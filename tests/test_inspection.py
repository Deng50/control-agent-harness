"""Artifact protocol fixtures exercise dependency selection, integrity and bounded previews.

Published fixtures in these tests are protocol records, not numerical success evidence.
"""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from control_harness.artifacts import Workspace, atomic_json
from control_harness.contracts import ContractError, ModelInterface
from control_harness.data import simulate_transitions, register_transitions, read_transitions, training_coverage
from control_harness.examples import example
from control_harness.inspection import prepare_handoff, read_artifact_file, report_preview
from control_harness.modeling import register_model
from control_harness.operations import run_operation
from control_harness.residual import train_residual


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "inspection-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(self.temp.name)
        case = example("nonlinear")
        self.interface = ModelInterface.parse(case["nominal"]["interface"])
        self.nominal = register_model(self.workspace, case["nominal"], {"kind": "user_nominal"})
        self.reference = register_model(self.workspace, case["reference"], {"kind": "reference_simulator"})
        self.data = simulate_transitions(self.workspace, self.nominal, self.reference, {"trajectories": 6, "steps": 3})

    def record(self, kind, parents, files, status="verified"):
        with self.workspace.stage(kind, self.interface, parents, {"test_fixture": "protocol_only"}) as stage:
            for name, value in files.items():
                atomic_json(stage.root / name, value)
            return stage.publish({"status": status}, source={"kind": "protocol_test_fixture"})

    def test_exact_dependencies_are_used_instead_of_active_reference(self):
        other = copy.deepcopy(example("nonlinear")["reference"])
        other["bias"] = [.001, .001]
        register_model(self.workspace, other, {"kind": "reference_simulator"})
        # The active reference differs before this selected branch is built.
        data = simulate_transitions(self.workspace, self.nominal, self.reference, {"trajectories": 6, "steps": 3})
        corrected = self.record("corrected_model", [self.nominal, data], {"report.json": {}})
        result = prepare_handoff(self.workspace, corrected)
        self.assertEqual(result["handoff"]["operation"], "synthesize_mpc")
        self.assertEqual(result["handoff"]["inputs"], {"model": corrected})
        self.assertEqual(result["optional_handoffs"][0]["inputs"], {"model": corrected, "evaluation_model": self.reference})
        self.assertEqual(result["optional_handoffs"][0]["operation"], "tune_controller")
        self.assertIn("evaluation",result["optional_handoffs"][0]["missing_config"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["handoff"]["missing_config"], [])
        self.assertIn("test_scenarios", result["optional_handoffs"][0]["missing_config"])
        self.assertIn(self.reference, [row["id"] for row in result["source_versions"]])

    def test_measured_data_does_not_fabricate_an_evaluation_model(self):
        arrays, _, _ = read_transitions(self.workspace, self.data)
        measured = register_transitions(self.workspace, self.nominal, arrays,
                                       {"kind": "measured", "description": "Protocol test source attribution"})
        corrected = self.record("corrected_model", [self.nominal, measured], {"report.json": {}})
        result = prepare_handoff(self.workspace, corrected)
        self.assertEqual(result["handoff"]["inputs"], {"model": corrected})
        plan = result["optional_handoffs"][0]
        self.assertEqual(plan["inputs"]["evaluation_model"], "")
        self.assertEqual(plan["missing_inputs"], ["evaluation_model"])
        plan = prepare_handoff(self.workspace, measured)["handoff"]
        self.assertEqual(plan["inputs"], {"dataset": measured, "nominal_model": self.nominal})
        self.assertEqual(plan["config_suggestions"]["output_dim"], self.interface.nx)

    def test_design_delivery_only_offers_tuning_for_its_bound_numeric_predictor(self):
        from control_harness.design_only import synthesize_mpc
        numeric = synthesize_mpc(self.workspace,{"model":self.nominal},{"solver_check":False})
        delivered = prepare_handoff(self.workspace,numeric)
        self.assertIsNone(delivered["handoff"]["operation"],"Design-only delivery must remain a stopping point")
        optional = delivered["optional_handoffs"][0]
        self.assertEqual(optional["operation"],"tune_controller")
        self.assertEqual(optional["inputs"],{"model":self.nominal,"evaluation_model":""})
        self.assertEqual(optional["missing_inputs"],["evaluation_model"])
        self.assertIn("evaluation",optional["missing_config"])
        draft = synthesize_mpc(self.workspace,{"description":"spring damper parameters unknown"},{})
        self.assertEqual(prepare_handoff(self.workspace,draft)["optional_handoffs"],[],"A parameterized draft is not an executable tuning model")

    def test_controller_and_policy_handoffs_bind_full_chain(self):
        corrected = self.record("corrected_model", [self.nominal, self.data], {"report.json": {}})
        controller = self.record("controller", [corrected, self.reference], {"controller.json": {
            "prediction_model": corrected, "evaluation_model": self.reference}})
        expert = self.record("expert_dataset", [controller, self.reference], {"collection.json": {}})
        policy = self.record("imitation_policy", [expert], {"policy.json": {"expert_controller": controller}})
        self.assertEqual(prepare_handoff(self.workspace, controller)["handoff"]["inputs"],
                         {"controller": controller, "evaluation_model": self.reference})
        self.assertEqual(prepare_handoff(self.workspace, expert)["handoff"]["inputs"], {"dataset": expert})
        self.assertEqual(prepare_handoff(self.workspace, policy)["handoff"]["inputs"], {"policy": policy, "model": corrected})
        bad = self.record("controller", [corrected], {"controller.json": {"evaluation_model": "reference_model-unbound"}})
        with self.assertRaises(ContractError):
            prepare_handoff(self.workspace, bad)

    def test_rejected_or_stale_artifacts_cannot_be_handed_off_but_can_be_read(self):
        rejected = self.record("controller", [self.nominal, self.reference], {"controller.json": {}}, "rejected")
        with self.assertRaises(ContractError):
            prepare_handoff(self.workspace, rejected)
        preview = read_artifact_file(self.workspace, rejected, "controller.json", {})
        self.assertEqual(preview["status"], "rejected")
        with self.assertRaises(ContractError):
            read_artifact_file(self.workspace, rejected, "../manifest.json", {})
        manifest = self.workspace.manifest
        manifest["artifacts"][self.data]["status"] = "stale"
        atomic_json(self.workspace.path / "manifest.json", manifest)
        with self.assertRaises(ContractError):
            prepare_handoff(self.workspace, self.data)

    def test_text_preview_is_bounded_and_array_preview_has_real_numeric_summary(self):
        result = read_artifact_file(self.workspace, self.nominal, "model.json", {"max_chars": 21})
        self.assertEqual(len(result["file_preview"]["text"]), 21)
        self.assertTrue(result["file_preview"]["truncated"])
        preview = read_artifact_file(self.workspace, self.data, "transitions.npz", {})["file_preview"]
        arrays, _, _ = read_transitions(self.workspace, self.data)
        summary = next(row for row in preview["arrays"] if row["name"] == "x")
        self.assertEqual(summary["shape"], list(arrays["x"].shape))
        self.assertEqual(summary["min"], float(arrays["x"].min()))
        self.assertEqual(summary["max"], float(arrays["x"].max()))
        self.assertEqual(summary["nonfinite"], 0)
        for limit in (0, True, 32001):
            with self.assertRaises(ContractError):
                read_artifact_file(self.workspace, self.nominal, "model.json", {"max_chars": limit})

    def test_preview_never_deserializes_numpy_pickle_objects(self):
        with self.workspace.stage("transition_dataset", self.interface, [self.nominal], {}) as stage:
            np.savez_compressed(stage.root / "objects.npz", unsafe=np.array([{"not": "numeric"}], dtype=object))
            artifact = stage.publish({"status": "verified"}, source={"kind": "protocol_test_fixture"})
        with self.assertRaises(ValueError):
            read_artifact_file(self.workspace, artifact, "objects.npz", {})

    def test_forged_numpy_shape_is_rejected_before_array_allocation(self):
        import io
        import zipfile
        from unittest.mock import patch
        payload = io.BytesIO()
        np.lib.format.write_array_header_1_0(payload, {
            "descr": "<f8", "fortran_order": False, "shape": (2**40,)})
        with self.workspace.stage("transition_dataset", self.interface, [self.nominal], {}) as stage:
            with zipfile.ZipFile(stage.root / "forged.npz", "w") as archive:
                archive.writestr("huge.npy", payload.getvalue())
            artifact = stage.publish({"status": "verified"}, source={"kind": "protocol_test_fixture"})
        with patch("control_harness.inspection.np.load", side_effect=AssertionError("must not allocate")):
            with self.assertRaisesRegex(ContractError, "shape/dtype"):
                read_artifact_file(self.workspace, artifact, "forged.npz", {})

    def test_large_report_samples_traces_without_recomputing_metrics(self):
        path = Path(self.temp.name) / "large-report.json"
        atomic_json(path, {"test": {"runs": [{"metrics": {"rmse": .123}, "trajectory": [{"step": i} for i in range(1000)]}]}})
        preview = report_preview(path)
        run = preview["test"]["runs"][0]
        self.assertEqual(run["metrics"]["rmse"], .123)
        self.assertEqual(len(run["trajectory"]), 240)
        self.assertEqual((run["trajectory"][0]["step"], run["trajectory"][-1]["step"]), (0, 999))
        self.assertFalse(preview["display_summary"]["metrics_recomputed"])
        self.assertEqual(len(__import__('json').loads(path.read_text())["test"]["runs"][0]["trajectory"]), 1000)

    def test_operations_dispatch_and_explicit_derivative_training_rejection(self):
        result = run_operation({"workspace": str(self.workspace.path), "operation": "prepare_handoff", "inputs": {"artifact_id": self.data}})
        self.assertEqual(result["handoff"]["operation"], "train_residual")
        with self.assertRaisesRegex(ContractError, "derivative"):
            train_residual(self.workspace, self.nominal, self.data, {"residual_semantics": "state_derivative_residual"})

    def test_applicability_uses_training_exposure_without_holdout_leakage(self):
        arrays, masks, _ = read_transitions(self.workspace, self.data)
        coverage = training_coverage(self.interface, arrays, masks["train"])
        modified = {key: value.copy() for key, value in arrays.items()}
        modified["x"][masks["test"]] = 1000
        self.assertEqual(coverage, training_coverage(self.interface, modified, masks["train"]))
        self.assertFalse(coverage["extrapolation_validated"])
        self.assertFalse(coverage["joint_coverage_proved"])
        self.assertEqual(len(coverage["marginal_ranges"]), self.interface.nx+self.interface.nu+self.interface.nd)
        self.assertEqual(coverage["samples"], int(masks["train"].sum()))


if __name__ == "__main__":
    unittest.main()
