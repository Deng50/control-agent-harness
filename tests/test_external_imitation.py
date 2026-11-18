"""Actual CPU MLP imitation, causal data-only contracts and exact continuation."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from control_harness.artifacts import Workspace, file_hash, read_json
from control_harness.contracts import ContractError, DataInterface
from control_harness.external_expert_data import import_expert_data, publish_external_expert_dataset
from control_harness.external_imitation import train_external_imitation, _load_members
from control_harness.material_registration import register_material
from control_harness.runtime import Cancelled, OperationContext


ROOT = Path(__file__).resolve().parents[1]


def definition(window=1, unit="1"):
    names = [("observation", "output", "observation", True), ("reference", "auxiliary", "reference", True),
             ("previous_a", "control", "previous_action", True), ("previous_b", "control", "previous_action", True),
             ("action_a", "control", "expert_action", True), ("action_b", "control", "expert_action", True)]
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "task": "dynamic_prediction", "sample_time": .2,
        "time_column": {"index": 0, "header": "time", "worksheet": None}, "trip_column": {"index": 1, "header": "trajectory", "worksheet": None},
        "signals": [{"name": name, "unit": unit if k == 0 else "1", "role": role, "expert_role": expert, "observable": available,
                     "delay_steps": 0, "column": {"index": k + 2, "header": name, "worksheet": None}} for k, (name, role, expert, available) in enumerate(names)],
        "features": [{"signal": "observation", "lag": 0, "window": window}] + [{"signal": name, "lag": 0, "window": 1} for name in ("reference", "previous_a", "previous_b")],
        "targets": [{"signal": name, "offset": 0} for name in ("action_a", "action_b")]}


def source_mapping():
    return {"signals": {"previous_a": {"time_semantics": "already_previous_applied_action", "action_target": "action_a"},
                        "previous_b": {"time_semantics": "already_previous_applied_action", "action_target": "action_b"}}}


class ExternalImitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def setUp(self):
        parent = ROOT / "test-output" / "external-imitation-tests"; parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.ws = Workspace(self.root / "case")
        self.network = {"hidden_dims": [7, 5], "activation": "tanh", "epochs": 6, "learning_rate": .004,
            "batch_size": 7, "patience": 50, "seed": 20261120, "weight_decay": .001, "min_delta": 0., "grad_clip": 5.}
        self.config = {"network": self.network, "action_metric_scale": [.3, .4]}

    def dataset(self, *, window=1, saturated_test=False, unit="1"):
        rng = np.random.default_rng(20261120); rows = []
        for trajectory in range(6):
            previous, last_obs = np.array([.2, -.1]), 0.
            for step in range(12):
                obs = rng.uniform(-.8, .8); ref = rng.uniform(-.4, .4)
                if saturated_test and trajectory >= 4: obs, ref = 5. + step * .02, 0.
                if saturated_test: action = np.clip([.8 * obs, -.7 * obs], -2., 2.)
                else: action = np.array([.35 * obs + .12 * ref + .1 * previous[0] + (.15 * last_obs if window > 1 else 0),
                                        -.22 * obs + .18 * ref + .08 * previous[1] - (.12 * last_obs if window > 1 else 0)])
                rows.append([step * .2, trajectory, obs, ref, *previous, *action]); previous, last_obs = action, obs
        values = np.asarray(rows); interface = DataInterface.parse(definition(window, unit))
        source = {"kind": "recorded_external_expert", "description": "Explicit handcrafted causal two-command external expert records; no internal MPC", "previous_action_semantics": "actual_applied_action_before_current_decision"}
        config = {"scaler": "standard", "action_constraints": {"bounds": [[-2., 2.], [-2., 2.]], "rate_bounds": [[-4., 4.], [-4., 4.]], "rate_semantics": "per_control_step"}}
        if window > 1:
            path = self.root / f"window-{window}-{unit}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(["time", "trajectory", "observation", "reference", "previous_a", "previous_b", "action_a", "action_b"]); writer.writerows(rows)
            material = register_material(self.ws, str(path), {}, OperationContext())
            return import_expert_data(self.ws, material, {**config, "interface": interface.to_dict(), "source_mapping": source_mapping(),
                "source": source, "split": {"mode": "trip", "train": .5, "validation": 1 / 6, "test": 1 / 3}}, OperationContext())
        return publish_external_expert_dataset(self.ws, {"X": values[:, 2:6], "y": values[:, 6:8], "actual_previous_u": values[:, 4:6],
            "trajectory_id": values[:, 1].astype(np.int64), "time": values[:, 0]}, interface,
            {"train": [0, 1, 2], "validation": [3], "test": [4, 5]}, config, OperationContext(), source=source, source_mapping=source_mapping())

    def train(self, dataset, config=None, context=None):
        artifact = train_external_imitation(self.ws, dataset, config or copy.deepcopy(self.config), context)
        root, meta = self.ws.resolve(artifact, kind="external_imitation_policy", allow_rejected=True)
        return artifact, root, meta, read_json(root / "report.json")

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor): self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected: self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected): self.assert_nested_equal(first, second)
        else: self.assertEqual(actual, expected)

    def test_real_configurable_mlp_with_arbitrary_data_interface_and_saved_test_predictions(self):
        source = self.dataset(); artifact, root, meta, report = self.train(source)
        from control_harness.external_policy import load_external_policy
        self.assertEqual(meta["interface"]["input_dim"], 4); self.assertEqual(meta["interface"]["output_dim"], 2)
        self.assertNotIn("nx", meta["interface"]); self.assertIsNone(report["physical_context"]["interface"])
        for key, value in self.network.items(): self.assertEqual(report["network"]["resolved_config"][key], value)
        self.assertEqual(report["network"]["parameter_count"], (4 + 1) * 7 + (7 + 1) * 5 + (5 + 1) * 2)
        self.assertGreater(report["network"]["optimizer_updates"], 0); self.assertEqual(report["closed_loop"], "not_run"); self.assertEqual(report["pgac"], "not_run")
        self.assertEqual(report["acceptance"], "not_requested"); self.assertFalse(report["test_used_for_training_or_selection"])
        nn_root, nn_meta = self.ws.resolve(report["supervised_model"])
        with np.load(root / "test_predictions.npz") as saved, np.load(nn_root / "test_predictions.npz") as original:
            np.testing.assert_array_equal(saved["raw_action"], original["prediction"])
            raw, truth, projected = saved["raw_action"], saved["expert_action"], saved["projected_action"]
            np.testing.assert_allclose(report["offline"]["test"]["raw"]["nrmse"], np.sqrt(np.mean((raw - truth) ** 2, axis=0)) / [.3, .4])
            np.testing.assert_allclose(report["offline"]["test"]["projected"]["nrmse"], np.sqrt(np.mean((projected - truth) ** 2, axis=0)) / [.3, .4])
        checkpoint = torch.load(nn_root / "latest.pt", weights_only=True)
        self.assertEqual([g["lr"] for g in checkpoint["optimizer_state"]["param_groups"]], [.004])
        self.assertEqual(checkpoint["updates"], report["network"]["optimizer_updates"])
        self.assertEqual(self.ws.manifest["active"]["external_imitation_policy"], artifact)
        self.assertNotIn("supervised_dataset", self.ws.manifest["active"]); self.assertNotIn("supervised_model", self.ws.manifest["active"])
        self.assertEqual(nn_meta["activation"], "none"); self.assertFalse(report["internal_controller_required"])
        policy = load_external_policy(root)
        source_root, _ = self.ws.resolve(source)
        with np.load(source_root / "expert.npz") as a:
            rows = np.flatnonzero(a["split_id"] == 2)
            np.testing.assert_allclose(policy.predict_raw(a["X"][rows]), raw, rtol=0, atol=1e-14)
            np.testing.assert_allclose(policy.predict(a["X"][rows], a["actual_previous_u"][rows]), projected, rtol=0, atol=1e-14)

    def test_projected_gate_is_explicit_and_raw_errors_are_diagnosed_without_hidden_gate(self):
        source = self.dataset(saturated_test=True)
        network = {**self.network, "hidden_dims": [], "epochs": 200, "learning_rate": .03, "weight_decay": 0., "patience": 200}
        _, _, meta, report = self.train(source, {"network": network, "action_metric_scale": [1., 1.], "max_test_action_nrmse": 0.})
        self.assertEqual(meta["validation"]["status"], "verified"); self.assertEqual(report["acceptance"], "passed")
        self.assertGreater(report["raw_out_of_bounds_fraction"], 0.)
        self.assertTrue(any(value > 0 for value in report["offline"]["test"]["raw"]["nrmse"]))
        self.assertEqual(report["offline"]["test"]["projected"]["nrmse"], [0., 0.])
        self.assertFalse(report["raw_metrics_gate_requested"])

    def test_rejected_candidate_keeps_active_policy_and_inactive_inner_artifacts(self):
        source = self.dataset(); accepted, _, _, first = self.train(source)
        rejected, _, meta, report = self.train(source, {**self.config, "max_test_action_nrmse": 0.})
        self.assertNotEqual(accepted, rejected); self.assertEqual(meta["validation"]["status"], "rejected")
        self.assertEqual(report["acceptance_failures"], ["max_test_action_nrmse"])
        self.assertEqual(self.ws.manifest["active"]["external_imitation_policy"], accepted)
        self.ws.resolve(accepted); self.assertEqual(report["supervised_dataset"], first["supervised_dataset"])
        self.assertNotIn("supervised_model", self.ws.manifest["active"])

    def test_cache_reuse_and_epoch_cancel_resume_preserve_actual_optimizer_rng_and_selected_weights(self):
        source = self.dataset(); _, _, _, full = self.train(source)
        expected_root, _ = self.ws.resolve(full["supervised_model"]); expected = torch.load(expected_root / "latest.pt", weights_only=True)
        events = []
        def emit(event):
            events.append(event)
            if event.get("stage") == "supervised_training" and event.get("epoch") == 2: raise Cancelled("Committed external imitation epoch two")
        with self.assertRaises(Cancelled): self.train(source, context=OperationContext(emit=emit))
        checkpoint = Path(events[-1]["checkpoint"]); self.assertTrue(checkpoint.is_file())
        _, _, _, resumed = self.train(source, {**self.config, "resume_checkpoint": str(checkpoint), "resume_checkpoint_sha256": file_hash(checkpoint)})
        self.assertEqual(resumed["supervised_dataset"], full["supervised_dataset"]); self.assertTrue(resumed["supervised_dataset_reused"])
        self.assertEqual(resumed["network"]["resumed_from_epoch"], 2)
        resumed_root, _ = self.ws.resolve(resumed["supervised_model"])
        self.assert_nested_equal(torch.load(resumed_root / "latest.pt", weights_only=True), expected)
        self.assert_nested_equal(torch.load(resumed_root / "model.pt", weights_only=True), torch.load(expected_root / "model.pt", weights_only=True))

    def test_epoch_zero_cancel_and_post_publication_failure_remain_recoverable(self):
        source = self.dataset(); checks = 0
        def cancelled():
            nonlocal checks
            checks += 1; return checks >= 5
        with self.assertRaises(Cancelled): self.train(source, context=OperationContext(cancelled=cancelled))
        checkpoints = list((self.ws.path / "failed").glob("*/latest.pt")); self.assertEqual(len(checkpoints), 1)
        self.assertEqual(torch.load(checkpoints[0], weights_only=True)["epoch"], 0)
        _, _, _, resumed = self.train(source, {**self.config, "resume_checkpoint": str(checkpoints[0])})
        published = []
        def emit(event):
            if event.get("checkpoint_status") == "published": published.append(event); raise RuntimeError("outer packaging interrupted after NN publish")
        with self.assertRaisesRegex(RuntimeError, "packaging interrupted"): self.train(source, context=OperationContext(emit=emit))
        checkpoint = Path(published[0]["checkpoint"]); self.assertEqual(checkpoint.parent.parent.name, "artifacts"); self.assertTrue(checkpoint.is_file())
        _, _, _, recovered = self.train(source, {**self.config, "resume_checkpoint": str(checkpoint)})
        self.assertEqual(recovered["network"]["resumed_from_epoch"], self.network["epochs"])
        self.assertEqual(recovered["supervised_dataset"], resumed["supervised_dataset"])

    def test_windowed_causal_features_train_without_a_physical_model_or_implicit_state_width(self):
        source = self.dataset(window=2); _, root, meta, report = self.train(source)
        self.assertEqual(meta["interface"]["input_dim"], 5)
        self.assertEqual([(x["signal"], x["lag"]) for x in report["feature_order"][:2]], [("observation", 1), ("observation", 0)])
        self.assertIsNone(report["physical_context"]["interface"]); self.assertFalse(report["physical_model_required"])
        from control_harness.external_policy import load_external_policy
        policy = load_external_policy(root)
        source_root, _ = self.ws.resolve(source)
        with np.load(source_root / "expert.npz") as a:
            rows = np.flatnonzero(a["split_id"] == 2); prediction = policy.predict(a["X"][rows], a["actual_previous_u"][rows])
            self.assertEqual(prediction.shape, (22, 2))
            with self.assertRaises((ContractError, ValueError)): policy.predict(a["X"][rows, :4], a["actual_previous_u"][rows])

    def test_training_only_metric_scale_and_explicit_scales_never_use_test_statistics(self):
        source = self.dataset(); _, _, _, report = self.train(source, {"network": self.network, "action_metric_scale": "train_std"})
        root, _ = self.ws.resolve(source)
        with np.load(root / "expert.npz") as a:
            expected = a["y"][a["split_id"] == 0].std(0, ddof=0)
            np.testing.assert_allclose(report["normalization"]["action_nrmse_scale"], expected, rtol=0, atol=1e-14)
        self.assertEqual(report["normalization"]["source"], "training_partition_only")
        self.assertFalse(report["normalization"]["fitted_using_test"])

    def test_generic_no_previous_action_no_rate_data_and_constant_action_fallback_are_explicit(self):
        rng = np.random.default_rng(56); d = definition(); d["features"] = d["features"][:2]
        features = rng.uniform(-.6, .6, (18, 2)); actions = np.tile([.2, -.3], (18, 1))
        source = publish_external_expert_dataset(self.ws, {"X": features, "y": actions, "trajectory_id": np.repeat([0, 1, 2], 6),
            "time": np.tile(np.arange(6) * .2, 3)}, DataInterface.parse(d), {"train": [0], "validation": [1], "test": [2]},
            {"scaler": "standard", "action_constraints": {"bounds": [[-2., 2.], [-2., 2.]], "rate_bounds": None, "rate_semantics": "per_control_step"}},
            OperationContext(), source={"kind": "recorded_external_expert", "description": "Constant recorded expert commands with no prior-action measurement", "previous_action_semantics": "not_recorded"}, source_mapping={})
        _, root, meta, report = self.train(source, {"network": self.network, "action_metric_scale": "train_std"})
        self.assertEqual(meta["interface"]["input_dim"], 2); self.assertIsNone(report["physical_context"]["interface"])
        np.testing.assert_allclose(report["normalization"]["action_nrmse_scale"], [1., 1.], rtol=0, atol=0)
        self.assertEqual(report["normalization"]["zero_std_actions"], [0, 1]); self.assertEqual(report["normalization"]["zero_std_fallback"], 1.)
        self.assertIsNone(report["offline"]["test"]["previous_action_source"])
        self.assertEqual(report["offline"]["test"]["projected_safety"]["rate_check"], "not_requested")
        from control_harness.external_policy import load_external_policy
        self.assertEqual(load_external_policy(root).predict(features[:2]).shape, (2, 2))

    def test_changed_network_or_source_semantics_cannot_force_old_optimizer_resume(self):
        source = self.dataset(); _, _, _, first = self.train(source)
        nn_root, _ = self.ws.resolve(first["supervised_model"])
        for changed in ({"hidden_dims": [8]}, {"learning_rate": .002}):
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                self.train(source, {**self.config, "network": {**self.network, **changed}, "resume_checkpoint": str(nn_root / "latest.pt")})
        other = self.dataset(unit="m")
        with self.assertRaises(ContractError): self.train(other, {**self.config, "resume_checkpoint": str(nn_root / "latest.pt")})

    def test_invalid_public_hyperparameters_scales_and_nested_gate_fail_before_creating_nn(self):
        source = self.dataset()
        cases = [{**self.config, "max_test_action_nrmse": True}, {**self.config, "action_metric_scale": [0., 1.]},
                 {**self.config, "action_metric_scale": [1.]}, {**self.config, "network": {**self.network, "input_dim": 99}},
                 {**self.config, "network": {**self.network, "resume_checkpoint": "fake.pt"}},
                 {**self.config, "network": {**self.network, "max_test_nrmse": .2}}, {**self.config, "unknown": 1}]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ContractError): self.train(source, config)
        self.assertFalse(any(row["kind"] in {"supervised_dataset", "supervised_model"} for row in self.ws.manifest["artifacts"].values()))

    def test_malicious_optional_previous_action_header_is_bounded_before_allocation(self):
        path = self.root / "malicious.npz"; member = io.BytesIO()
        np.lib.format.write_array_header_1_0(member, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 2)})
        member.write(b"\0" * 8)
        with zipfile.ZipFile(path, "w") as archive: archive.writestr("actual_previous_u.npy", member.getvalue())
        with self.assertRaisesRegex(ContractError, "shape/dtype/declared allocation"):
            _load_members(path, {"actual_previous_u": (3, 2)}, hashlib.sha256(path.read_bytes()).hexdigest(), optional=True)


if __name__ == "__main__": unittest.main()
