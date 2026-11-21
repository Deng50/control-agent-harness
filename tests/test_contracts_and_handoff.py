from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from control_harness.artifacts import Workspace, WorkspaceBusy, atomic_json
from control_harness.contracts import ContractError, ModelInterface, validate_network_dimensions
from control_harness.control import MPCPolicy, partition_scenarios
from control_harness.data import trajectory_split
from control_harness.examples import example
from control_harness.modeling import DeclarativeDynamics, CorrectedDynamics, ResidualNetwork, register_model, tensor
from control_harness.policy import NeuralPolicy, network_config
from control_harness.runtime import OperationContext, Cancelled

torch.set_num_threads(1)
TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "unit-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class ContractTests(unittest.TestCase):
    def test_units_order_and_time_are_not_silently_coerced(self):
        original = example("linear")["nominal"]["interface"]
        a = ModelInterface.parse(original)
        for mutation in ("unit", "order", "time"):
            other = copy.deepcopy(original)
            if mutation == "unit":
                other["controls"][0]["unit"] = "kW"
            elif mutation == "order":
                other["states"].reverse()
            else:
                other["sample_time"] *= 10
            with self.assertRaises(ContractError):
                a.compatible(ModelInterface.parse(other))

    def test_network_dimension_edits_require_metadata_agreement(self):
        interface = ModelInterface.parse(example("linear")["nominal"]["interface"])
        self.assertEqual(validate_network_dimensions(interface, {}, residual=True), (7, 4))
        with self.assertRaises(ContractError):
            validate_network_dimensions(interface, {"output_dim": 3}, residual=True)

    def test_split_never_shares_a_trajectory(self):
        trajectory = np.repeat(np.arange(20), 8)
        split = trajectory_split(trajectory, 52)
        all_ids = []
        for group in split.values():
            all_ids.extend(group)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), set(range(20)))

    def test_controller_refuses_renamed_duplicate_holdouts(self):
        config = example("linear")
        config["controller"]["test_scenarios"] = copy.deepcopy(config["controller"]["tuning_scenarios"])
        config["controller"]["test_scenarios"][0]["name"] = "pretend-held-out"
        with self.assertRaises(ContractError):
            partition_scenarios(config["controller"], ModelInterface.parse(config["nominal"]["interface"]))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.workspace = Workspace(self.temp.name)
        self.spec = example("linear")["nominal"]

    def tearDown(self):
        self.temp.cleanup()

    def register(self):
        return register_model(self.workspace, self.spec, {"kind": "user_nominal"})

    def test_failed_stage_preserves_previous_accepted_output(self):
        old = self.register()
        with self.assertRaises(RuntimeError):
            with self.workspace.stage("nominal_model", ModelInterface.parse(self.spec["interface"]), [], {}) as stage:
                atomic_json(stage.root / "model.json", self.spec)
                raise RuntimeError("simulated numerical failure")
        self.assertEqual(self.workspace.active("nominal_model"), old)
        self.assertTrue(list((self.workspace.path / "failed").glob("*/failure.json")))
        self.assertFalse(self.workspace.lock_path.exists())

    def test_tampering_is_detected_before_handoff(self):
        artifact = self.register()
        root, _ = self.workspace.resolve(artifact)
        (root / "model.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ContractError):
            self.workspace.resolve(artifact)

    def test_upstream_revision_retires_defaults_but_preserves_frozen_consumers(self):
        parent = self.register()
        interface = ModelInterface.parse(self.spec["interface"])
        with self.workspace.stage("controller", interface, [parent], {}) as stage:
            child = stage.publish({"status": "verified"})
        with self.workspace.stage("expert_dataset", interface, [child], {}) as stage:
            grandchild = stage.publish({"status": "verified"})
        self.spec["A"][0][0] += .01
        replacement = self.register()
        self.assertNotEqual(parent, replacement)
        for item in (child, grandchild):
            root, _ = self.workspace.resolve(item)
            self.assertTrue((root / "artifact.json").is_file())
        self.assertNotIn("controller", self.workspace.manifest["active"])
        self.assertNotIn("expert_dataset", self.workspace.manifest["active"])
        self.workspace.resolve(parent)  # Immutable historical source remains inspectable.

    def test_conflicting_operations_are_refused(self):
        other_request = Workspace(self.workspace.path)
        with self.workspace.lock("first"):
            with self.workspace.lock("nested_stage"):
                pass  # A workflow owns one lock across its own nested stages.
            with self.assertRaises(WorkspaceBusy):
                with other_request.lock("second"):
                    pass

    def test_rejected_output_never_becomes_active_or_loadable(self):
        old = self.register()
        with self.workspace.stage("nominal_model", ModelInterface.parse(self.spec["interface"]), [], {}) as stage:
            rejected = stage.publish({"status": "rejected"})
        self.assertEqual(self.workspace.active("nominal_model"), old)
        with self.assertRaises(ContractError):
            self.workspace.resolve(rejected)

    def test_external_path_handoff_copies_verified_dependencies_and_is_source_independent(self):
        parent = self.register()
        interface = ModelInterface.parse(self.spec["interface"])
        with self.workspace.stage("controller", interface, [parent], {}) as stage:
            atomic_json(stage.root / "controller.json", {"prediction_model": parent})
            child = stage.publish({"status": "verified"})
        source_root, source_meta = self.workspace.resolve(child)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as target:
            destination = Workspace(target)
            imported = destination.import_artifact(source_root)
            self.assertEqual(imported, child)
            _, imported_meta = destination.resolve(child)
            self.assertEqual(imported_meta, source_meta)
            self.assertEqual(destination.active("controller"), child)
            self.assertIn(parent, destination.manifest["artifacts"])
            original_parent, _ = self.workspace.resolve(parent)
            (original_parent / "model.json").write_text("{}", encoding="utf-8")
            destination.resolve(child)  # The imported package no longer depends on the original path.


class NumericalConnectionTests(unittest.TestCase):
    def test_residual_changes_mpc_action_and_retains_action_gradients(self):
        spec = example("nonlinear")["nominal"]
        nominal = DeclarativeDynamics(spec)
        i = nominal.interface
        scalers = {"input_mean": [0] * (i.nx + i.nu + i.nd), "input_scale": [1] * (i.nx + i.nu + i.nd),
                   "residual_scale": [.1] * i.nx}
        residual = ResidualNetwork(i, {"hidden_dims": [8, 8]}, scalers)
        for p in residual.parameters():
            p.data.zero_()
        residual.network.net[-1].bias.data[0] = 1.0
        corrected = CorrectedDynamics(nominal, residual)
        x, u, d = tensor([0., 0.]), tensor([0.]).requires_grad_(True), tensor([0.])
        predicted = corrected(x, u, d)
        gradient, = torch.autograd.grad(predicted.sum(), u)
        self.assertGreater(abs(gradient.item()), .1)
        parameters = {"horizon": 3, "q": 1, "r": .001, "s": 0}
        base_action, _ = MPCPolicy(nominal, parameters).act([0, 0], [0], [0])
        corrected_action, _ = MPCPolicy(corrected, parameters).act([0, 0], [0], [0])
        self.assertLess(corrected_action[0], base_action[0] - .1)

    def test_three_networks_support_nonlegacy_dimensions_and_differentiation(self):
        i = ModelInterface.parse(example("linear")["nominal"]["interface"])
        for arch in ("mlp", "rnn", "lstm"):
            config = network_config(i, {"architecture": arch, "disturbance_forecast": "declared_constant",
                                       "hidden_dims": [8, 8], "hidden_size": 8, "preview_len": 3})
            scalers = {"input_mean": [0] * config["input_dim"], "input_scale": [1] * config["input_dim"],
                       "output_mean": [0] * i.nu, "output_scale": [1] * i.nu}
            policy = NeuralPolicy(i, config, scalers)
            state = tensor([[.1] * i.nx]).requires_grad_(True)
            result = policy(state, tensor([[0] * i.ny]), tensor([[0] * i.nd]))
            self.assertEqual(tuple(result.shape), (1, 2))
            grad, = torch.autograd.grad(result.sum(), state)
            self.assertTrue(torch.isfinite(grad).all())

    def test_recurrent_forecast_semantics_must_be_explicit(self):
        i = ModelInterface.parse(example("linear")["nominal"]["interface"])
        with self.assertRaises(ContractError):
            network_config(i, {"architecture": "lstm"})

    def test_cancellation_is_an_error_not_a_successful_empty_result(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as path:
            flag = Path(path) / "cancel"
            flag.write_text("cancel", encoding="utf-8")
            with self.assertRaises(Cancelled):
                OperationContext(cancel_file=flag).check()


if __name__ == "__main__":
    unittest.main()
