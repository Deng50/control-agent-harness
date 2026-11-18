"""Real conditional PINN gradients, full means, checkpoints and held-out evidence.

Short component runs make no frozen-F04 accuracy claim. The analytic solution
is an independent test oracle only; training consumes provided temperatures and
computes its own network derivatives from fixed physical collocation inputs.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace, file_hash, read_json
from control_harness.contracts import ContractError
from control_harness.material_registration import register_material
from control_harness.pinn import (train_pinn, resolve_pinn_config, normalized_features,
    normalized_physics_residual, gradient_probe, _components, _mean_component, _gradients, _component)
from control_harness.pinn_problem import prepare_pinn_problem, read_pinn_problem
from control_harness.pinn_runtime import load_pinn
from control_harness.runtime import Cancelled, OperationContext
from control_harness.supervised_export import build_supervised_network
from test_pinn_problem import configuration


PROJECT = Path(__file__).resolve().parents[1]
FROZEN = PROJECT / "test-output/v2-development/20261117T002006/fixtures"
OUTPUT = PROJECT / "test-output/pinn-tests"


class PinnTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        cls.base = Path(cls.temporary.name)
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        inputs = cls.base / "inputs"; inputs.mkdir()
        for name in ("F04_conditional_ode.json", "F04_analytic_trajectories.csv", "F04_training_physics_collocation.csv"):
            shutil.copyfile(FROZEN / name, inputs / name)
        ws = Workspace(cls.base / "prepared")
        material = register_material(ws, str(inputs / "F04_conditional_ode.json"), {
            "dependency_root": str(inputs), "dependencies": ["F04_analytic_trajectories.csv", "F04_training_physics_collocation.csv"]}, OperationContext())
        cls.problem_id = prepare_pinn_problem(ws, material, configuration(), OperationContext())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads); cls.temporary.cleanup()

    def setUp(self):
        self.temporary_case = tempfile.TemporaryDirectory(dir=OUTPUT); self.addCleanup(self.temporary_case.cleanup)
        self.root = Path(self.temporary_case.name)
        shutil.copytree(self.base / "prepared", self.root / "workspace")
        self.ws = Workspace(self.root / "workspace")
        self.problem = read_pinn_problem(self.ws, self.problem_id)
        self.definition, self.arrays = self.problem["definition"], self.problem["arrays"]
        self.network = {"hidden_dims": [5], "activation": "tanh", "epochs": 4, "learning_rate": .004,
            "batch_size": 128, "patience": 30, "seed": 20261121, "weight_decay": .001, "min_delta": 0., "grad_clip": 5.}
        self.config = {"network": self.network, "loss_weights": {"data": 1., "physics": 1., "initial": 2.}}

    def train(self, config=None, context=None, problem_id=None):
        artifact = train_pinn(self.ws, problem_id or self.problem_id, copy.deepcopy(config or self.config), context)
        root, metadata = self.ws.resolve(artifact, kind="pinn_model", allow_rejected=True)
        return artifact, root, metadata, read_json(root / "report.json")

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor): self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected: self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected): self.assert_nested_equal(first, second)
        else: self.assertEqual(actual, expected)

    def network_model(self, root, filename="model.pt"):
        model = build_supervised_network(read_json(root / "network_config.json"))
        value = torch.load(root / filename, map_location="cpu", weights_only=True)
        model.load_state_dict(value["network_state"] if filename == "model.pt" else value)
        return model

    def test_real_short_training_uses_physical_collocation_and_actual_hyperparameters(self):
        artifact, root, metadata, report = self.train()
        self.assertEqual(metadata["interface"]["interface_kind"], "material")
        self.assertEqual(metadata["interface"]["format"], "bundle")
        self.assertTrue(report["is_pinn"]); self.assertEqual(report["algorithm"], "conditional_ode_pinn")
        self.assertEqual(report["parameter_count"], 26)
        self.assertEqual(report["optimizer_updates"], 4); self.assertEqual(report["trained_epochs"], 4)
        self.assertEqual(report["acceptance"], "not_requested"); self.assertEqual(report["accuracy_claim"], "not_requested")
        for key, value in self.network.items(): self.assertEqual(report["resolved_config"]["network"][key], value)
        self.assertEqual(report["collocation"], {"fixed": True, "training_conditions_only": True, "points": 1028, "temperature_targets_used": False})
        self.assertFalse(report["test_used_for_training_or_selection"]); self.assertEqual(report["test_evaluations"], 1)
        self.assertEqual(report["selection"], "validation_weighted_data_physics_initial_loss_only")
        self.assertFalse(report["capability"]["one_step_transition"])
        self.assertEqual(metadata["source"]["problem"], self.problem_id)
        self.assertEqual(self.ws.manifest["active"]["pinn_model"], artifact)
        checkpoint = torch.load(root / "latest.pt", weights_only=True)
        self.assertEqual(checkpoint["contract"]["algorithm"], "conditional_ode_pinn")
        self.assertEqual(checkpoint["optimizer_state"]["param_groups"][0]["lr"], .004)
        self.assertEqual(set(checkpoint["model_state"]), {"0.weight", "0.bias", "2.weight", "2.bias"})
        self.assertTrue(all(row["optimizer_updates"] == row["epoch"] for row in checkpoint["log"]))
        model = load_pinn(root)
        self.assertFalse(hasattr(model, "step")); self.assertFalse(hasattr(model, "interface"))
        coverage = report["condition_coverage"]
        self.assertEqual(coverage["unique_training_initial_temperature_K"], [291.15, 295.15])
        self.assertEqual(coverage["unique_training_power_W"], [0., 200.])
        self.assertTrue(coverage["has_complete_2_by_2_training_subgrid"]); self.assertFalse(coverage["coverage_is_training_gate"])

    def test_seeded_gradient_probe_has_real_three_term_parameter_contributions(self):
        _, root, _, report = self.train()
        seeded = self.network_model(root, "seeded_initial_state.pt")
        train, _, _, _ = _components(self.arrays, self.definition)
        independent = gradient_probe(seeded, train, self.definition, 128, self.config["loss_weights"])
        self.assertEqual(independent["components"], report["gradient_probes"]["initial"]["components"])
        for name in ("data", "physics", "initial"):
            component = independent["components"][name]
            self.assertGreater(component["raw_loss"], 0.)
            self.assertGreater(component["raw_parameter_gradient_l2"], 1e-10)
            self.assertTrue(component["raw_gradient_nonzero"])
        self.assertTrue(independent["physics_derivative_create_graph"])
        self.assertEqual(independent["points"], {"data": 804, "physics": 1028, "initial": 4})

    def test_actual_physics_off_ablation_changes_updates_and_keeps_explicit_classification(self):
        active_id, active_root, _, active = self.train()
        active_checkpoint = torch.load(active_root / "latest.pt", weights_only=True)
        seeded = torch.load(active_root / "seeded_initial_state.pt", weights_only=True)
        off = {**self.config, "loss_weights": {"data": 1., "physics": 0., "initial": 2.}, "ablation": True}
        _, off_root, off_meta, ablation = self.train(off)
        off_checkpoint = torch.load(off_root / "latest.pt", weights_only=True)
        self.assertFalse(ablation["is_pinn"]); self.assertEqual(ablation["algorithm"], "conditional_ode_physics_off_ablation")
        self.assertEqual(off_meta["source"]["algorithm"], "conditional_ode_physics_off_ablation")
        self.assertEqual(ablation["checkpoint_algorithm"], "conditional_ode_pinn")
        physics = ablation["gradient_probes"]["initial"]["components"]["physics"]
        self.assertEqual(physics["raw_loss"], active["gradient_probes"]["initial"]["components"]["physics"]["raw_loss"])
        self.assertGreater(physics["raw_parameter_gradient_l2"], 0.); self.assertEqual(physics["weighted_parameter_gradient_l2"], 0.)
        self.assertFalse(physics["enabled_in_optimizer"])
        self.assertTrue(any(not torch.equal(value, off_checkpoint["model_state"][key]) for key, value in active_checkpoint["model_state"].items()))
        self.assertIsNone(ablation["log"][0]["training_components"]["physics"])
        self.assertFalse(read_json(off_root / "bundle/training.json")["physics_loss_enabled"])
        self.assert_nested_equal(torch.load(off_root / "seeded_initial_state.pt", weights_only=True), seeded)
        self.assertEqual(self.ws.manifest["active"]["pinn_model"], active_id)
        self.ws.resolve(active_id); self.assertEqual(off_meta["activation"], "none")

    def test_independent_analytic_solution_has_zero_residual_with_full_si_time_chain(self):
        definition = copy.deepcopy(self.definition)
        definition["normalization"].update(time_origin_s=7., time_scale_s=13., state_scale_K=3., power_scale_W=170., physics_scale_W=97.)
        definition["equation"]["parameters"].update(C_J_per_K=1234., G_W_per_K=37.)
        class Analytic(torch.nn.Module):
            def forward(self, z):
                p, n = definition["equation"]["parameters"], definition["normalization"]
                equilibrium = z[:, 2:3] * n["power_scale_W"] / (p["G_W_per_K"] * n["state_scale_K"])
                return equilibrium + (z[:, 1:2] - equilibrium) * torch.exp(-p["G_W_per_K"] * n["time_scale_s"] / p["C_J_per_K"] * z[:, :1])
        t, initial, power = np.array([7., 8.3, 13., 20.]), np.array([291.5, 294., 295., 292.]), np.array([0., 170., 40., 90.])
        z = normalized_features(t, initial, power, definition)
        with torch.no_grad(): residual = normalized_physics_residual(Analytic(), z, definition, create_graph=False)
        np.testing.assert_allclose(residual.detach().numpy(), 0., atol=2e-15, rtol=0)
        n = definition["normalization"]; epsilon = 1e-2
        shifted = {shift: Analytic()(normalized_features(t + shift * epsilon, initial, power, definition)).detach().numpy() for shift in (-2, -1, 1, 2)}
        dT = (-shifted[2] + 8 * shifted[1] - 8 * shifted[-1] + shifted[-2]) * n["state_scale_K"] / (12 * epsilon)
        theta = Analytic()(z).detach().numpy(); p = definition["equation"]["parameters"]
        independent = (p["C_J_per_K"] * dT + p["G_W_per_K"] * n["state_scale_K"] * theta - power[:, None]) / n["physics_scale_W"]
        np.testing.assert_allclose(independent, 0., atol=2e-10, rtol=0)

    def test_parameter_gradients_match_finite_difference_and_chunks_match_full_means(self):
        network = torch.nn.Sequential(torch.nn.Linear(3, 1, dtype=torch.float64))
        with torch.no_grad(): network[0].weight.copy_(torch.tensor([[.2, -.3, .4]], dtype=torch.float64)); network[0].bias.fill_(.07)
        points = torch.tensor([[.1, -.4, .3], [.5, .2, .8], [.9, .1, .6]], dtype=torch.float64)
        target = torch.tensor([[.2], [.3], [.7]], dtype=torch.float64)
        p, n = self.definition["equation"]["parameters"], self.definition["normalization"]
        theta = points.numpy() @ np.array([[.2], [-.3], [.4]]) + .07
        manual = (p["C_J_per_K"] * n["state_scale_K"] / n["time_scale_s"] * .2 + p["G_W_per_K"] * n["state_scale_K"] * theta - points[:, 2:3].numpy() * n["power_scale_W"]) / n["physics_scale_W"]
        np.testing.assert_allclose(normalized_physics_residual(network, points, self.definition).detach().numpy(), manual, atol=1e-16)
        for name in ("data", "physics", "initial"):
            network.zero_grad(set_to_none=True)
            full = _mean_component(network, name, points, target, self.definition, 3, OperationContext(), backward=True)
            gradient = _gradients(network).clone()
            network.zero_grad(set_to_none=True)
            chunked = _mean_component(network, name, points, target, self.definition, 2, OperationContext(), backward=True)
            self.assertAlmostEqual(full, chunked, places=14); torch.testing.assert_close(_gradients(network), gradient, rtol=1e-14, atol=1e-14)
            numerical = []
            for parameter in network.parameters():
                for index in range(parameter.numel()):
                    with torch.no_grad(): parameter.reshape(-1)[index] += 1e-6
                    plus = float(_component(network, name, points, target, self.definition, create_graph=False).detach())
                    with torch.no_grad(): parameter.reshape(-1)[index] -= 2e-6
                    minus = float(_component(network, name, points, target, self.definition, create_graph=False).detach())
                    with torch.no_grad(): parameter.reshape(-1)[index] += 1e-6
                    numerical.append((plus - minus) / 2e-6)
            np.testing.assert_allclose(gradient.numpy(), numerical, rtol=2e-9, atol=2e-10)

    def test_disconnected_time_derivative_is_rejected_not_replaced_with_reference_derivative(self):
        class Detached(torch.nn.Module):
            def forward(self, points): return points[:, :1].detach()
        class Ignored(torch.nn.Module):
            def __init__(self): super().__init__(); self.offset = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
            def forward(self, points): return self.offset.expand(len(points), 1)
        z = torch.zeros((2, 3), dtype=torch.float64)
        for network in (Detached(), Ignored()):
            with self.subTest(type=type(network).__name__), self.assertRaisesRegex(ContractError, "detached|disconnected"):
                normalized_physics_residual(network, z, self.definition)

    def test_cancel_after_committed_epoch_resumes_exact_model_adam_rng_log_and_best(self):
        _, full_root, _, full = self.train()
        expected = torch.load(full_root / "latest.pt", weights_only=True); best = torch.load(full_root / "model.pt", weights_only=True)
        events = []
        def emit(event):
            events.append(event)
            if event.get("stage") == "pinn_training" and event.get("epoch") == 2: raise Cancelled("Completed PINN epoch two")
        with self.assertRaises(Cancelled): self.train(context=OperationContext(emit=emit))
        checkpoint = Path(events[-1]["checkpoint"]); self.assertTrue(checkpoint.is_file())
        self.assertEqual(torch.load(checkpoint, weights_only=True)["epoch"], 2)
        _, resumed_root, _, resumed = self.train({**self.config, "resume_checkpoint": str(checkpoint), "resume_checkpoint_sha256": file_hash(checkpoint)})
        self.assertEqual(resumed["resumed_from_epoch"], 2)
        self.assert_nested_equal(torch.load(resumed_root / "latest.pt", weights_only=True), expected)
        self.assert_nested_equal(torch.load(resumed_root / "model.pt", weights_only=True), best)
        self.assertEqual(resumed["log"], full["log"])
        self.assertTrue(resumed["gradient_probes"]["initial"]["reconstructed_during_resume"])

    def test_epoch_zero_is_recoverable_and_target_total_epochs_refs_do_not_change_trajectory(self):
        short = {**self.config, "network": {**self.network, "epochs": 2}}
        _, root, _, _ = self.train(short)
        checkpoint = root / "latest.pt"
        _, continued_root, _, _ = self.train({**self.config, "resume_checkpoint": str(checkpoint), "plan_refs": {"training": {"plan_id": "new-authorization"}}})
        continued = torch.load(continued_root / "latest.pt", weights_only=True)
        events = []
        def emit(event):
            events.append(event)
            if event.get("stage") == "pinn_initial_checkpoint": raise Cancelled("Before any optimizer update")
        with self.assertRaises(Cancelled): self.train(context=OperationContext(emit=emit))
        checkpoint = Path(events[-1]["checkpoint"]); zero = torch.load(checkpoint, weights_only=True)
        self.assertEqual(zero["epoch"], 0); self.assertEqual(zero["updates"], 0); self.assertIsNone(zero["best_state"])
        _, resumed_root, _, _ = self.train({**self.config, "resume_checkpoint": str(checkpoint)})
        self.assert_nested_equal(torch.load(resumed_root / "latest.pt", weights_only=True), continued)

    def test_checkpoint_rejects_changed_loss_optimizer_or_network_and_preserves_active_on_failed_gate(self):
        accepted, root, _, _ = self.train(); checkpoint = root / "latest.pt"
        variations = [{**self.config, "loss_weights": {"data": 1., "physics": 2., "initial": 2.}},
            {**self.config, "network": {**self.network, "learning_rate": .005}},
            {**self.config, "network": {**self.network, "hidden_dims": [6]}}]
        for config in variations:
            with self.subTest(config=config), self.assertRaisesRegex(ContractError, "configuration differs"):
                self.train({**config, "resume_checkpoint": str(checkpoint)})
        rejected, _, metadata, report = self.train({**self.config, "acceptance": {"max_test_trajectory_nrmse": 0.}})
        self.assertNotEqual(accepted, rejected); self.assertEqual(metadata["validation"]["status"], "rejected")
        self.assertEqual(report["acceptance_failures"], ["max_test_trajectory_nrmse"])
        self.assertEqual(self.ws.manifest["active"]["pinn_model"], accepted); self.ws.resolve(accepted)
        changed_problem = self.prepare_changed(normalization={"state_scale_K": 5.})
        with self.assertRaisesRegex(ContractError, "configuration differs"):
            self.train({**self.config, "resume_checkpoint": str(checkpoint)}, problem_id=changed_problem)

    def test_final_test_arrays_reproduce_metrics_and_physical_runtime_residual(self):
        _, root, _, report = self.train(); runtime = load_pinn(root)
        with np.load(root / "test_predictions.npz", allow_pickle=False) as saved:
            prediction = runtime.predict(saved["time_s"], saved["initial_temperature_K"], saved["power_W"])
            np.testing.assert_allclose(prediction, saved["predicted_temperature_K"], atol=0., rtol=0.)
            residual = runtime.residual(*(torch.from_numpy(saved[key]) for key in ("time_s", "initial_temperature_K", "power_W")), create_graph=False).detach().numpy()
            # Portable runtime forms G*(T-Ta) in SI; the trainer uses G*scale*theta.
            # Reconstructing ~293K and subtracting Ta introduces ~1e-14 roundoff.
            np.testing.assert_allclose(residual, saved["normalized_physics_residual"], atol=2e-14, rtol=1e-14)
            self.assertAlmostEqual(float(np.sqrt(np.mean(residual**2))), report["test"]["normalized_physics_rms"], places=14)
            self.assertAlmostEqual(float(np.max(np.abs(saved["normalized_initial_residual"]))), report["test"]["normalized_initial_max_abs_error"], places=14)
            for row in report["test"]["per_condition"]:
                mask = saved["condition_index"] == row["condition_index"]
                rmse = float(np.sqrt(np.mean((prediction[mask] - saved["reference_temperature_K"][mask])**2)))
                self.assertAlmostEqual(rmse / self.definition["normalization"]["data_nrmse_scale_K"], row["nrmse"], places=13)

    def prepare_changed(self, *, test_shift=0., sparse=False, normalization=None):
        inputs = self.root / "changed-inputs"; shutil.copytree(self.base / "inputs", inputs)
        config = configuration()
        if normalization: config["normalization"].update(normalization)
        reference = inputs / "F04_analytic_trajectories.csv"
        with reference.open(encoding="utf-8", newline="") as stream: rows = list(csv.reader(stream))
        if sparse:
            removed = {row["id"] for row in config["conditions"][1:4]}
            config["conditions"] = [row for row in config["conditions"] if row["id"] not in removed]
            rows = [rows[0]] + [row for row in rows[1:] if row[0] not in removed]
            path = inputs / "F04_training_physics_collocation.csv"
            with path.open(encoding="utf-8", newline="") as stream: collocation = list(csv.reader(stream))
            with path.open("w", encoding="utf-8", newline="") as stream: csv.writer(stream).writerows([collocation[0]] + [row for row in collocation[1:] if row[0] not in removed])
        for row in rows[1:]:
            if row[1] == "test": row[5] = str(float(row[5]) + test_shift)
        with reference.open("w", encoding="utf-8", newline="") as stream: csv.writer(stream).writerows(rows)
        material = register_material(self.ws, str(inputs / "F04_conditional_ode.json"), {"dependency_root": str(inputs),
            "dependencies": ["F04_analytic_trajectories.csv", "F04_training_physics_collocation.csv"]}, OperationContext())
        return prepare_pinn_problem(self.ws, material, config, OperationContext())

    def test_test_label_change_cannot_change_actual_optimizer_or_validation_selection(self):
        _, root, _, report = self.train(); before = torch.load(root / "latest.pt", weights_only=True)
        changed_id = self.prepare_changed(test_shift=2.)
        _, changed_root, _, changed = self.train(problem_id=changed_id)
        after = torch.load(changed_root / "latest.pt", weights_only=True)
        for key in ("model_state", "optimizer_state", "best_state", "rng_state", "epoch", "updates", "wait", "log", "best_metric"):
            self.assert_nested_equal(after[key], before[key])
        self.assertNotEqual(after["training_signature"], before["training_signature"])
        self.assertEqual(changed["best_epoch"], report["best_epoch"])
        self.assertNotEqual(changed["test"]["rmse_K"], report["test"]["rmse_K"])

    def test_single_training_condition_is_reported_without_claiming_domain_generalization(self):
        problem = self.prepare_changed(sparse=True)
        _, _, _, report = self.train(problem_id=problem)
        coverage = report["condition_coverage"]
        self.assertEqual(coverage["training_conditions"], 1); self.assertFalse(coverage["has_complete_2_by_2_training_subgrid"])
        self.assertEqual(coverage["training_physical_pairs"], [[291.15, 0.]])
        self.assertFalse(coverage["coverage_is_training_gate"])
        self.assertIn("declared domain is not evidence", coverage["generalization_evidence"])

    def test_physics_only_objective_uses_no_data_loss_updates_but_keeps_reference_provenance(self):
        config = {**self.config, "network": {**self.network, "epochs": 2}, "loss_weights": {"data": 0., "physics": 1., "initial": 2.}}
        _, _, _, report = self.train(config)
        self.assertTrue(report["is_pinn"]); self.assertFalse(report["data_labels_used_for_optimizer"])
        self.assertTrue(report["provided_reference_data"])
        self.assertEqual(report["training_regime"], "physics_only_objective_with_provided_reference")
        self.assertTrue(all(row["training_components"]["data"] is None for row in report["log"]))

    def test_config_rejects_wrong_dims_rough_activation_hidden_resume_and_invalid_loss_claims(self):
        variants = []
        for change in ({"input_dim": 1}, {"output_dim": 2}, {"activation": "relu"}, {"max_test_nrmse": .1}, {"resume_checkpoint": "any.pt"}):
            variants.append({**self.config, "network": {**self.network, **change}})
        variants += [{**self.config, "loss_weights": {"data": 1., "physics": 0., "initial": 2.}},
            {**self.config, "ablation": True}, {**self.config, "loss_weights": {"data": 0., "physics": 0., "initial": 2.}},
            {**self.config, "loss_weights": {"data": 1., "physics": 1., "initial": 0.}},
            {**self.config, "loss_weights": {"data": 1., "physics": 10**1000, "initial": 2.}},
            {**self.config, "acceptance": {"max_normalized_physics_rms": -1}}, {**self.config, "accuracy": True}]
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ContractError): resolve_pinn_config(config)
        smooth_linear = resolve_pinn_config({**self.config, "network": {**self.network, "hidden_dims": [], "activation": "silu"}})
        self.assertEqual(smooth_linear["network_config"]["hidden_dims"], [])
        self.assertEqual(resolve_pinn_config({**self.config, "acceptance": {"max_test_trajectory_nrmse": 4.}})["acceptance"]["max_test_trajectory_nrmse"], 4.)

    def test_derivative_activation_budget_rejects_before_network_allocation(self):
        config = {**self.config, "network": {**self.network, "hidden_dims": [4096], "batch_size": 256}}
        with patch("control_harness.pinn.build_supervised_network", side_effect=AssertionError("Must reject before model allocation")):
            with self.assertRaisesRegex(ContractError, "derivative activation.*reduce batch_size"):
                self.train(config)


class AffinePinnTests(unittest.TestCase):
    """New unit-only data; no F04 reference or supplemental holdout is opened."""
    train = PinnTrainingTests.train
    assert_nested_equal = PinnTrainingTests.assert_nested_equal
    network_model = PinnTrainingTests.network_model

    @classmethod
    def setUpClass(cls):
        from control_harness.pinn_contract import UNITS
        OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.temporary = tempfile.TemporaryDirectory(dir=OUTPUT); cls.base = Path(cls.temporary.name)
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)
        inputs = cls.base / "inputs"; inputs.mkdir()
        conditions = [{"id": name, "split": part, "initial_temperature_K": t0, "power_W": power} for name, part, t0, power in (
            ("train00", "train", 298., -1.), ("train01", "train", 298., 3.),
            ("train10", "train", 302., -1.), ("train11", "train", 302., 3.),
            ("validation0", "validation", 299., 0.), ("validation1", "validation", 301., 2.),
            ("unit_test0", "test", 299.5, .5), ("unit_test1", "test", 300.5, 1.5))]
        config = {"schema_version": 1, "equation": {"template": "thermal_first_order", "forcing": "constant_per_condition",
            "parameters": {"C_J_per_K": 5., "G_W_per_K": 2., "Ta_K": 300.}},
            "domain": {"time_s": [0., 2.], "initial_temperature_K": [298., 302.], "power_W": [-1., 3.]},
            "normalization": {"time_origin_s": 0., "time_scale_s": 2., "state_offset_K": 300., "state_scale_K": 3., "power_scale_W": 3., "physics_scale_W": 4.},
            "units": dict(UNITS), "conditions": conditions,
            "reference_file": "source/reference.csv", "reference_columns": {key: {"index": index, "header": header} for key, index, header in (
                ("condition_id", 0, "id"), ("split", 1, "split"), ("time_s", 2, "t"), ("initial_temperature_K", 3, "T0"), ("power_W", 4, "P"), ("temperature_K", 5, "T"))},
            "collocation_file": "source/collocation.csv", "collocation_columns": {key: {"index": index, "header": header} for key, index, header in (
                ("condition_id", 0, "id"), ("time_s", 1, "t"), ("initial_temperature_K", 2, "T0"), ("power_W", 3, "P"))},
            "source": {"kind": "synthetic_analytic", "description": "Separate small analytic unit-only thermal package; no frozen F04 or supplemental labels"}}
        (inputs / "unit-problem.json").write_text(__import__("json").dumps(config), encoding="utf-8")
        with (inputs / "reference.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["id", "split", "t", "T0", "P", "T"])
            for condition in conditions:
                t0, power = condition["initial_temperature_K"], condition["power_W"]
                for time in np.linspace(0., 2., 21):
                    temperature = 300. + power / 2. + (t0 - 300. - power / 2.) * np.exp(-2. * time / 5.)
                    writer.writerow([condition["id"], condition["split"], time, t0, power, temperature])
        with (inputs / "collocation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["id", "t", "T0", "P"])
            for condition in conditions:
                if condition["split"] == "train":
                    for time in np.linspace(.03, 1.97, 9): writer.writerow([condition["id"], time, condition["initial_temperature_K"], condition["power_W"]])
        ws = Workspace(cls.base / "prepared")
        material = register_material(ws, str(inputs / "unit-problem.json"), {"dependency_root": str(inputs), "dependencies": ["reference.csv", "collocation.csv"]}, OperationContext())
        cls.problem_id = prepare_pinn_problem(ws, material, config, OperationContext())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads); cls.temporary.cleanup()

    def setUp(self):
        PinnTrainingTests.setUp(self)
        self.network["batch_size"] = 16

    def test_explicit_free_mode_is_identical_to_unchanged_default_raw_mlp(self):
        from control_harness.pinn_runtime import conditional_network
        _, first_root, _, first = self.train()
        expected = torch.load(first_root / "latest.pt", weights_only=True)
        selected = torch.load(first_root / "model.pt", weights_only=True)
        _, second_root, _, second = self.train({**self.config, "conditional_parameterization": "free_mlp"})
        self.assert_nested_equal(torch.load(second_root / "latest.pt", weights_only=True), expected)
        self.assert_nested_equal(torch.load(second_root / "model.pt", weights_only=True), selected)
        self.assertEqual(first["log"], second["log"]); self.assertEqual(second["conditional_parameterization"], "free_mlp")
        raw = self.network_model(second_root)
        self.assertIs(conditional_network(raw, self.definition), raw)

    def test_affine_condition_identity_time_and_parameter_gradients_are_real(self):
        from control_harness.pinn_runtime import conditional_network
        resolved = resolve_pinn_config(self.config); torch.manual_seed(20261121)
        raw = build_supervised_network(resolved["network_config"])
        view = conditional_network(raw, self.definition, "affine_initial_power")
        self.assertEqual([id(p) for p in view.parameters()], [id(p) for p in raw.parameters()])
        calls = []; handle = raw.register_forward_hook(lambda _module, args, _result: calls.append(len(args[0])))
        z = normalized_features([.7] * 5, [298., 302., 298., 302., 299.4], [-1., -1., 3., 3., .8], self.definition)
        values = view(z); handle.remove(); self.assertEqual(calls, [5, 5, 5])
        np.testing.assert_allclose((values[3] - values[1] - values[2] + values[0]).detach().numpy(), 0., atol=3e-16, rtol=0.)
        alpha, beta = (299.4 - 298.) / 4., (.8 + 1.) / 4.
        expected = values[0] + alpha * (values[1] - values[0]) + beta * (values[2] - values[0])
        torch.testing.assert_close(values[4], expected, atol=3e-16, rtol=0.)
        # Check time derivative through all three actual NN forwards, independently
        # from physical-time finite differences at an untrained interior condition.
        time = np.array([.31, .87, 1.61]); initial = np.array([299.2, 300.1, 301.3]); power = np.array([.1, .9, 2.2])
        points = normalized_features(time, initial, power, self.definition); n = self.definition["normalization"]
        epsilon = 1e-3
        shifted = {shift: view(normalized_features(time + shift * epsilon, initial, power, self.definition)).detach().numpy() for shift in (-2, -1, 1, 2)}
        derivative = (-shifted[2] + 8 * shifted[1] - 8 * shifted[-1] + shifted[-2]) * n["state_scale_K"] / (12 * epsilon)
        theta = view(points).detach().numpy(); p = self.definition["equation"]["parameters"]
        finite_difference = (p["C_J_per_K"] * derivative + p["G_W_per_K"] * n["state_scale_K"] * theta - power[:, None]) / n["physics_scale_W"]
        residual = normalized_physics_residual(view, points, self.definition)
        np.testing.assert_allclose(residual.detach().numpy(), finite_difference, atol=2e-11, rtol=0.)
        raw.zero_grad(set_to_none=True); residual.square().mean().backward(); gradient = _gradients(raw).clone().numpy()
        numerical = []
        for parameter in raw.parameters():
            for index in range(parameter.numel()):
                with torch.no_grad(): parameter.reshape(-1)[index] += 1e-6
                plus = float(normalized_physics_residual(view, points, self.definition, create_graph=False).square().mean().detach())
                with torch.no_grad(): parameter.reshape(-1)[index] -= 2e-6
                minus = float(normalized_physics_residual(view, points, self.definition, create_graph=False).square().mean().detach())
                with torch.no_grad(): parameter.reshape(-1)[index] += 1e-6
                numerical.append((plus - minus) / 2e-6)
        np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-7)
        self.assertGreater(float(np.linalg.norm(gradient)), 1e-6)
        raw.zero_grad(set_to_none=True)
        initial_points = normalized_features([0., 0.], [298., 302.], [-1., 3.], self.definition)
        initial_loss = _component(view, "initial", initial_points, None, self.definition, create_graph=True)
        initial_loss.backward(); self.assertGreater(float(initial_loss.detach()), 1e-8)
        self.assertGreater(float(torch.linalg.vector_norm(_gradients(raw))), 1e-8)

    def test_real_affine_training_export_and_exact_cancel_resume_keep_raw_checkpoints(self):
        config = {**self.config, "conditional_parameterization": "affine_initial_power"}
        _, root, metadata, report = self.train(config)
        expected = torch.load(root / "latest.pt", weights_only=True); selected = torch.load(root / "model.pt", weights_only=True)
        self.assertEqual(set(expected["model_state"]), {"0.weight", "0.bias", "2.weight", "2.bias"})
        self.assertEqual(expected["contract"]["config"]["conditional_parameterization"], "affine_initial_power")
        self.assertEqual(report["conditional_parameterization"], "affine_initial_power")
        self.assertEqual(metadata["source"]["conditional_parameterization"], "affine_initial_power")
        self.assertEqual(report["resource_boundary"]["prediction_forwards_per_query"], 3)
        self.assertGreater(report["gradient_probes"]["initial"]["components"]["initial"]["weighted_parameter_gradient_l2"], 0.)
        runtime = load_pinn(root)
        with np.load(root / "test_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(runtime.predict(saved["time_s"], saved["initial_temperature_K"], saved["power_W"]), saved["predicted_temperature_K"], atol=6e-14, rtol=0.)
        events = []
        def emit(event):
            events.append(event)
            if event.get("stage") == "pinn_training" and event.get("epoch") == 2: raise Cancelled("Affine completed epoch two")
        with self.assertRaises(Cancelled): self.train(config, OperationContext(emit=emit))
        checkpoint = Path(events[-1]["checkpoint"])
        with self.assertRaisesRegex(ContractError, "configuration differs"):
            self.train({**config, "conditional_parameterization": "free_mlp", "resume_checkpoint": str(checkpoint)})
        _, resumed_root, _, resumed = self.train({**config, "resume_checkpoint": str(checkpoint), "resume_checkpoint_sha256": file_hash(checkpoint)})
        self.assert_nested_equal(torch.load(resumed_root / "latest.pt", weights_only=True), expected)
        self.assert_nested_equal(torch.load(resumed_root / "model.pt", weights_only=True), selected)
        self.assertEqual(resumed["log"], report["log"])

    def test_three_forward_derivative_budget_and_parameterization_enum_are_enforced(self):
        wide = {**self.config, "network": {**self.network, "hidden_dims": [4096], "batch_size": 256}}
        with patch("control_harness.pinn.build_supervised_network", side_effect=AssertionError("Model allocation reached")):
            with self.assertRaisesRegex(ContractError, "derivative activation"):
                self.train({**wide, "conditional_parameterization": "affine_initial_power"})
            with self.assertRaisesRegex(AssertionError, "Model allocation reached"):
                self.train({**wide, "conditional_parameterization": "free_mlp"})
        for mode in ("generic_affine", True, None):
            with self.subTest(mode=mode), self.assertRaises(ContractError): resolve_pinn_config({**self.config, "conditional_parameterization": mode})


if __name__ == "__main__": unittest.main()
