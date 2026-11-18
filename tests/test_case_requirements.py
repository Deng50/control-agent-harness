"""S03 workflow scope/source versions and frozen register-only execution."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from control_harness.artifacts import Workspace, atomic_json, canonical_hash, read_json
from control_harness.case_requirements import confirmation_requirements, freeze_workflow_case, resolve_workflow_case
from control_harness.case_plans import FIXTURE_KIND, PendingPlanConfirmation, propose_plan, validate_plan
from control_harness.contracts import ContractError, ModelInterface
from control_harness.examples import example
from control_harness.modeling import register_model
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
from control_harness.workflow import run_workflow

OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", Path(__file__).resolve().parents[1] / "test-output/case-requirement-tests"))
OUTPUT.mkdir(parents=True, exist_ok=True)


class CaseRequirementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.root = Path(self.temporary.name)
        self.workspace = Workspace(self.root / "workspace")
        self.model = self.root / "nominal.json"
        self.spec = example("linear")["nominal"]
        atomic_json(self.model, self.spec)
        self.data = self.root / "transitions.npz"
        self.data.write_bytes(b"declared file-version fixture, never trained")
        self.nominal = register_model(self.workspace, self.spec, {"kind": "user_nominal", "description": "static fixture"})

    def tearDown(self):
        self.temporary.cleanup()

    def request(self, steps, *, file=False):
        definition = {"schema_version": 1, "name": "局部流程", "steps": steps}
        if file:
            path = self.root / "workflow.json"
            atomic_json(path, definition)
            return {"operation": "run_workflow", "workspace": str(self.workspace.path),
                    "inputs": {"workflow_path": str(path)}, "config": {}}
        return {"operation": "run_workflow", "workspace": str(self.workspace.path), "inputs": {}, "config": definition}

    def model_step(self):
        return {"id": "model", "operation": "register_model", "inputs": {"model_path": str(self.model)},
                "config": {"source": {"kind": "user_nominal"}}}

    def import_step(self):
        return {"id": "data", "operation": "import_transitions", "inputs": {
            "nominal_model": self.nominal, "data_path": str(self.data)},
            "config": {"source": {"kind": "measured", "description": "only file mapping in this test"}}}

    def fixture_context(self, cls=OperationContext, **kwargs):
        directory = self.workspace.path / ".case-plans"
        directory.mkdir(exist_ok=True)
        atomic_json(directory / "test-fixture.json", {"kind": FIXTURE_KIND, "workspace": str(self.workspace.path)})
        return cls.test_fixture(self.workspace, **kwargs)

    def test_register_and_import_only_workflows_have_no_extra_confirmation_scopes(self):
        for steps in [[self.model_step()], [self.import_step()], [self.model_step(), self.import_step()]]:
            with self.subTest(stages=[s["operation"] for s in steps]):
                request = self.request(steps)
                self.assertEqual(confirmation_requirements(self.workspace, request), [])
                self.assertIn("workflow_inputs", resolve_workflow_case(self.workspace, request)["bindings"])

    def test_selected_steps_supply_the_minimum_union(self):
        imported = self.import_step()
        train = {"id": "network", "operation": "train_residual", "inputs": {
            "nominal_model": self.nominal, "dataset": {"stage": "data"}}, "config": {"epochs": 7}}
        sample = {"id": "samples", "operation": "simulate_transitions", "inputs": {
            "nominal_model": self.nominal, "reference_model": self.nominal}, "config": {"steps": 5}}
        self.assertEqual([x["scope"] for x in confirmation_requirements(self.workspace, self.request([imported, train]))], ["signals", "training"])
        self.assertEqual([x["scope"] for x in confirmation_requirements(self.workspace, self.request([sample]))], ["signals", "sampling"])
        self.assertEqual([x["scope"] for x in confirmation_requirements(self.workspace, self.request([sample, imported, train]))], ["signals", "sampling", "training"])

    def test_file_definition_is_fully_summarized_and_relative_sources_are_bound(self):
        step = self.model_step()
        step["inputs"]["model_path"] = "nominal.json"
        request = self.request([step], file=True)
        before = copy.deepcopy(request)
        case = resolve_workflow_case(self.workspace, request)
        self.assertEqual(request, before)
        self.assertEqual(case["definition"]["steps"][0]["inputs"]["model_path"], str(self.model))
        self.assertEqual(case["payload"]["workflow_definition"], case["definition"])
        self.assertEqual(case["bindings"]["workflow_file"]["path"], request["inputs"]["workflow_path"])
        self.assertIn("sha256", case["bindings"]["workflow_inputs"]["model.model_path"])

    def test_same_path_workflow_or_declared_data_change_invalidates_old_plan(self):
        train = {"id": "network", "operation": "train_residual", "inputs": {
            "nominal_model": self.nominal, "dataset": {"stage": "data"}}, "config": {"epochs": 7}}
        request = self.request([self.import_step(), train], file=True)
        first = confirmation_requirements(self.workspace, request)[0]
        plan = propose_plan(self.workspace, first["scope"], first["expected_payload"], first["bindings"])
        self.data.write_bytes(b"changed data at exactly the same path")
        changed = confirmation_requirements(self.workspace, request)[0]
        self.assertNotEqual(first["bindings"], changed["bindings"])
        with self.assertRaises(PendingPlanConfirmation):
            validate_plan(self.workspace, {"plan_id": plan["plan_id"], "version_hash": plan["version_hash"]},
                          first["scope"], expected_payload=changed["expected_payload"], bindings=changed["bindings"])
        path = Path(request["inputs"]["workflow_path"])
        definition = read_json(path)
        definition["steps"][-1]["config"]["epochs"] = 8
        atomic_json(path, definition)
        next_case = resolve_workflow_case(self.workspace, request)
        self.assertNotEqual(next_case["bindings"]["workflow_file"]["sha256"], first["bindings"]["workflow_file"]["sha256"])
        self.assertEqual(next_case["payload"]["workflow_definition"]["steps"][-1]["config"]["epochs"], 8)

    def test_external_directory_versions_include_and_verify_parent_closure(self):
        source = Workspace(self.root / "external")
        parent = register_model(source, self.spec, {"kind": "user_nominal"})
        with source.stage("corrected_model", ModelInterface.parse(self.spec["interface"]), [parent], {}) as stage:
            atomic_json(stage.root / "test-descriptor.json", {"static_fixture": True})
            child = stage.publish({"status": "verified"}, source={"kind": "explicit_test_fixture"})
        original = source.resolve(child)[0]
        request = self.request([{"id": "external", "operation": "import_artifact", "inputs": {"artifact_path": str(original)}}])
        case = resolve_workflow_case(self.workspace, request)
        self.assertEqual(case["scopes"], [])
        version = case["bindings"]["workflow_inputs"]["external.artifact_path"]
        self.assertEqual(set(version["artifacts"]), {parent, child})
        with self.workspace.lock("freeze-fixture"):
            frozen = freeze_workflow_case(self.workspace, case, self.workspace.path / ".case-plans/frozen")
        cached = Path(frozen["config"]["steps"][0]["inputs"]["artifact_path"])
        self.assertNotEqual(cached, original)
        self.assertEqual(read_json(cached / "test-descriptor.json"), {"static_fixture": True})
        (source.resolve(parent)[0] / "model.json").write_text("tampered parent", encoding="utf-8")
        with self.assertRaises(ContractError):
            resolve_workflow_case(self.workspace, request)
        self.assertTrue((cached / "artifact.json").exists())

    def test_frozen_files_stop_rereading_mutated_workflow_and_have_stable_resume_paths(self):
        request = self.request([self.model_step()], file=True)
        case = resolve_workflow_case(self.workspace, request)
        cache = self.workspace.path / ".case-plans/frozen"
        with self.workspace.lock("freeze-fixture"):
            first = freeze_workflow_case(self.workspace, case, cache)
            second = freeze_workflow_case(self.workspace, case, cache)
            self.assertEqual(first, second)
            changed = copy.deepcopy(self.spec)
            changed["A"][0][0] += .05
            atomic_json(self.model, changed)
            result = run_workflow(self.workspace, first["inputs"], first["config"], OperationContext())
        artifact = self.workspace.resolve(result["artifacts"]["model"])[0]
        self.assertEqual(read_json(artifact / "model.json")["A"], self.spec["A"])
        atomic_json(self.model, self.spec)
        resumed_request = copy.deepcopy(request)
        resumed_request["config"]["resume_run"] = result["workflow_run"]
        resumed_case = resolve_workflow_case(self.workspace, resumed_request)
        self.assertEqual(case["payload"], resumed_case["payload"])
        with self.workspace.lock("freeze-fixture"):
            resumed = freeze_workflow_case(self.workspace, resumed_case, cache)
            self.assertEqual(first["config"]["steps"], resumed["config"]["steps"])
            outcome = run_workflow(self.workspace, resumed["inputs"], resumed["config"], OperationContext())
        self.assertEqual(outcome["artifacts"], result["artifacts"])

    def test_change_between_confirmation_check_and_freezing_is_rejected(self):
        request = self.request([self.model_step()], file=True)
        case = resolve_workflow_case(self.workspace, request)
        self.model.write_text("changed after verification", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "changed after"):
            freeze_workflow_case(self.workspace, case, self.workspace.path / ".case-plans/frozen")

    def test_existing_dataonly_and_legacy_resume_fields_keep_same_summaries(self):
        for operation, config in [("prepare_supervised_dataset", {"interface": {"input_columns": [0], "output_columns": [1]}}),
                                  ("train_residual", {"epochs": 10, "learning_rate": .001})]:
            request = {"operation": operation, "workspace": str(self.workspace.path), "inputs": {}, "config": config}
            before = confirmation_requirements(self.workspace, request)
            request["config"].update(plan_refs={"signals": {}}, resume_checkpoint="not-read.pt", resume_checkpoint_sha256="x", resume_run="ignored")
            self.assertEqual(confirmation_requirements(self.workspace, request), before)

    def test_actual_run_operation_registration_only_fixture_needs_no_plans_and_uses_frozen_bytes(self):
        request = self.request([self.model_step()], file=True)
        changed = copy.deepcopy(self.spec)
        changed["A"][0][0] += .05
        def mutate_original(event):
            if event.get("stage") == "workflow" and event.get("step_status") == "running":
                atomic_json(self.model, changed)
        result = run_operation(request, self.fixture_context(emit=mutate_original))
        self.assertEqual(result["status"], "verified")
        root = self.workspace.resolve(result["artifacts"]["model"])[0]
        self.assertEqual(read_json(root / "model.json")["A"], self.spec["A"])
        self.assertEqual(read_json(self.model)["A"], changed["A"])
        self.assertFalse((self.workspace.path / ".case-plans/plans.json").exists())

    def test_actual_sampling_only_fixture_confirms_two_scopes_refuses_file_edit_and_resumes(self):
        reference = register_model(self.workspace, self.spec, {"kind": "reference_simulator", "description": "explicit mathematical fixture, not measurement"})
        request = self.request([{"id": "samples", "operation": "simulate_transitions", "inputs": {
            "nominal_model": self.nominal, "reference_model": reference},
            "config": {"trajectories": 5, "steps": 3, "seed": 23}}], file=True)
        requirements = confirmation_requirements(self.workspace, request)
        self.assertEqual([row["scope"] for row in requirements], ["signals", "sampling"])
        plans = [propose_plan(self.workspace, row["scope"], row["expected_payload"], row["bindings"]) for row in requirements]
        request["config"]["plan_refs"] = {plan["scope"]: {key: plan[key] for key in ("plan_id", "version_hash")} for plan in plans}
        context = self.fixture_context()
        first = run_operation(request, context)
        self.assertEqual(first["status"], "verified")
        self.assertEqual([row["scope"] for row in context.case_confirmation_sources], ["signals", "sampling"])
        self.assertTrue(all(row["confirmation_source"] == "test_fixture" for row in context.case_confirmation_sources))
        source_file = Path(request["inputs"]["workflow_path"])
        reviewed_bytes = source_file.read_bytes()
        changed = read_json(source_file)
        changed["steps"][0]["config"]["steps"] = 4
        atomic_json(source_file, changed)
        before = self.workspace.manifest
        with self.assertRaises(PendingPlanConfirmation):
            run_operation(request, self.fixture_context())
        self.assertEqual(self.workspace.manifest, before)
        source_file.write_bytes(reviewed_bytes)
        request["config"]["resume_run"] = first["workflow_run"]
        resumed = run_operation(request, self.fixture_context())
        self.assertEqual(resumed["artifacts"], first["artifacts"])

    def test_actual_worker_check_then_source_change_cannot_reenter_original_workflow_file(self):
        request = self.request([self.model_step()], file=True)
        path = Path(request["inputs"]["workflow_path"])
        class ChangedAfterCheck(OperationContext):
            def require_case_confirmation(context, candidate, requirements):
                result = super().require_case_confirmation(candidate, requirements)
                payload = read_json(path)
                payload["name"] = "changed after worker confirmation verification"
                atomic_json(path, payload)
                return result
        before = self.workspace.manifest
        with self.assertRaisesRegex(ContractError, "changed after"):
            run_operation(request, self.fixture_context(ChangedAfterCheck))
        self.assertEqual(self.workspace.manifest, before)

    def test_existing_legacy_cli_register_only_request_and_resume_stay_compatible(self):
        request = self.request([self.model_step()], file=True)
        environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "PYTHONUTF8": "1"}
        for key in list(environment):
            if key.startswith("CONTROL_CASE_RECEIPT_") or key == "CONTROL_CASE_NATIVE_SESSION":
                del environment[key]
        def call(candidate):
            child = subprocess.run([sys.executable, "-B", "-m", "control_harness.cli", "--stdin"],
                input=json.dumps(candidate), capture_output=True, text=True, encoding="utf-8", env=environment)
            self.assertEqual(child.returncode, 0, child.stderr + child.stdout)
            return next(json.loads(line)["result"] for line in child.stdout.splitlines() if json.loads(line)["event"] == "result")
        first = call(request)
        request["config"]["resume_run"] = first["workflow_run"]
        self.assertEqual(call(request)["artifacts"], first["artifacts"])


if __name__ == "__main__":
    unittest.main()
