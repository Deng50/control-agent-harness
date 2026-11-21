from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from control_harness.artifacts import Workspace, WorkspaceBusy, atomic_json, read_json
from control_harness.contracts import ContractError
from control_harness.contracts import canonical_hash
from control_harness.examples import example
from control_harness.operations import run_operation
from control_harness.runtime import Cancelled, OperationContext
from control_harness.workflow import plan_workflow, run_workflow

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / "test-output" / "workflow-tests"
TEMP.mkdir(parents=True, exist_ok=True)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP)
        self.workspace = Workspace(Path(self.temp.name) / "workspace")
        self.nominal = Path(self.temp.name) / "nominal.json"
        self.reference = Path(self.temp.name) / "reference.json"
        settings = example("nonlinear")
        atomic_json(self.nominal, settings["nominal"])
        atomic_json(self.reference, settings["reference"])
        self.config = {"schema_version": 1, "steps": [
            {"id": "nominal", "operation": "register_model", "inputs": {"model_path": str(self.nominal)}, "config": {"source": {"kind": "user_nominal"}}},
            {"id": "reference", "operation": "register_model", "inputs": {"model_path": str(self.reference)}, "config": {"source": {"kind": "reference_simulator"}}},
            {"id": "data", "operation": "simulate_transitions", "inputs": {"nominal_model": {"stage": "nominal"}, "reference_model": {"stage": "reference"}},
             "config": {"trajectories": 5, "steps": 8, "seed": 11}},
        ]}

    def tearDown(self):
        self.temp.cleanup()

    def test_full_examples_preflight_without_execution(self):
        for case in ("linear", "nonlinear"):
            plan = plan_workflow(self.workspace, {"workflow_path": str(ROOT / "examples" / case / "workflow.json")}, {})
            self.assertEqual(plan["errors"], [])
            self.assertEqual(len(plan["stages"]), 8)
        self.assertEqual(self.workspace.manifest["artifacts"], {})

    def test_cancel_then_resume_reuses_exact_committed_stage(self):
        def stop(event):
            if event.get("step") == "reference":
                raise Cancelled("Test actual stage boundary cancellation")
        with self.assertRaises(Cancelled):
            run_workflow(self.workspace, {}, self.config, OperationContext(emit=stop))
        record_path = next((self.workspace.path / "workflows").glob("*.json"))
        record = read_json(record_path)
        first_id = record["steps"]["nominal"]["artifact_id"]
        self.assertEqual(record["status"], "cancelled")
        result = run_workflow(self.workspace, {}, {**self.config, "resume_run": record["run_id"]}, OperationContext())
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["artifacts"]["nominal"], first_id)
        self.assertEqual(len(self.workspace.manifest["artifacts"]), 3)
        resumed_again = run_workflow(self.workspace, {}, {**self.config, "resume_run": record["run_id"]}, OperationContext())
        self.assertEqual(result["artifacts"], resumed_again["artifacts"])

    def test_recovers_publication_if_stage_acknowledgement_was_lost(self):
        result = run_workflow(self.workspace, {}, self.config, OperationContext())
        record_path = Path(result["record_path"])
        record = read_json(record_path)
        record["steps"]["data"] = {"status": "running"}
        record["status"] = "running"
        atomic_json(record_path, record)
        recovered = run_workflow(self.workspace, {}, {**self.config, "resume_run": result["workflow_run"]}, OperationContext())
        self.assertEqual(recovered["artifacts"], result["artifacts"])
        self.assertTrue(recovered["steps"]["data"]["recovered_publication"])

    def test_failed_training_record_and_checkpoint_preflight_use_the_same_generation(self):
        workflow = copy.deepcopy(self.config)
        training = {"id": "training", "operation": "train_residual",
                    "inputs": {"nominal_model": {"stage": "nominal"}, "dataset": {"stage": "data"}},
                    "config": {"epochs": 3, "hidden_dims": [4], "batch_size": 8}}
        workflow["steps"].append(training)

        def stop(event):
            if event.get("stage") == "residual_training" and event.get("epoch") == 1:
                raise Cancelled("cancelled after committed generation")

        with self.assertRaises(Cancelled):
            run_workflow(self.workspace, {}, workflow, OperationContext(emit=stop))
        record = read_json(next((self.workspace.path / "workflows").glob("*.json")))
        error = record["error"]
        self.assertEqual((error["epoch"], error["target_epochs"], error["checkpoint_status"]), (1, 3, "archived"))
        settings = training["config"]
        settings.update(resume_checkpoint=error["checkpoint"], resume_checkpoint_sha256=error["checkpoint_sha256"])
        plan = plan_workflow(self.workspace, {}, workflow)
        self.assertEqual(plan["errors"], [])
        self.assertEqual(plan["resources"][f"file:{error['checkpoint']}"], error["checkpoint_sha256"])
        settings["resume_checkpoint_sha256"] = "0" * 64
        self.assertIn("SHA-256", plan_workflow(self.workspace, {}, workflow)["errors"][0]["message"])

    def test_changed_input_or_config_cannot_resume_old_results(self):
        result = run_workflow(self.workspace, {}, self.config, OperationContext())
        altered = copy.deepcopy(self.config)
        altered["steps"][-1]["config"]["seed"] += 1
        with self.assertRaisesRegex(ContractError, "changed"):
            run_workflow(self.workspace, {}, {**altered, "resume_run": result["workflow_run"]}, OperationContext())
        spec = read_json(self.nominal)
        spec["A"][0][0] += .01
        atomic_json(self.nominal, spec)
        with self.assertRaisesRegex(ContractError, "changed"):
            run_workflow(self.workspace, {}, {**self.config, "resume_run": result["workflow_run"]}, OperationContext())

    def test_completed_record_cannot_substitute_another_valid_artifact(self):
        result = run_workflow(self.workspace, {}, self.config, OperationContext())
        record_path = Path(result["record_path"])
        record = read_json(record_path)
        unrelated = result["artifacts"]["reference"]
        _, meta = self.workspace.resolve(unrelated)
        record["steps"]["nominal"].update(artifact_id=unrelated, metadata_hash=canonical_hash(meta))
        atomic_json(record_path, record)
        with self.assertRaisesRegex(ContractError, "provenance mismatch"):
            run_workflow(self.workspace, {}, {**self.config, "resume_run": result["workflow_run"]}, OperationContext())

    def test_reviewed_model_override_does_not_execute_original_python(self):
        source = Path(self.temp.name) / "object.py"
        source.write_text("raise RuntimeError('This source must not execute')", encoding="utf-8")
        request = {"workspace": str(self.workspace.path), "operation": "register_model", "inputs": {"model_path": str(source)},
                   "config": {"model_spec": example("linear")["nominal"], "source": {"kind": "reconstructed_equations", "description": "Reviewed explicit equations"}}}
        result = run_operation(request)
        _, meta = self.workspace.resolve(result["artifact_id"])
        self.assertTrue(meta["source"]["explicit_model_override"])
        self.assertEqual(meta["source"]["kind"], "reconstructed_equations")

    def test_preflight_rejects_forward_refs_wrong_kind_and_interface(self):
        config = copy.deepcopy(self.config)
        config["steps"][-1]["inputs"]["nominal_model"] = {"stage": "future"}
        self.assertEqual(plan_workflow(self.workspace, {}, config)["status"], "invalid")
        config["steps"][-1]["inputs"]["nominal_model"] = {"stage": "reference"}
        self.assertIn("expects", plan_workflow(self.workspace, {}, config)["errors"][0]["message"])
        spec = read_json(self.reference)
        spec["interface"]["sample_time"] *= 2
        atomic_json(self.reference, spec)
        self.assertIn("Interface mismatch", plan_workflow(self.workspace, {}, self.config)["errors"][0]["message"])

    def test_rejected_stage_stops_before_downstream_and_remains_inspectable(self):
        config = copy.deepcopy(self.config)
        config["steps"].extend([
            {"id": "residual", "operation": "train_residual", "inputs": {"nominal_model": {"stage": "nominal"}, "dataset": {"stage": "data"}},
             "config": {"epochs": 1, "hidden_dims": [4], "seed": 10, "minimum_improvement": .999999}},
            {"id": "after", "operation": "simulate_transitions", "inputs": {"nominal_model": {"stage": "nominal"}, "reference_model": {"stage": "reference"}},
             "config": {"trajectories": 5, "steps": 1}},
        ])
        result = run_workflow(self.workspace, {}, config, OperationContext())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["steps"]["after"]["status"], "pending")
        rejected = result["steps"]["residual"]["artifact_id"]
        report = run_operation({"workspace": str(self.workspace.path), "operation": "inspect_artifact", "inputs": {"artifact_id": rejected}})
        self.assertEqual(report["status"], "rejected")
        with self.assertRaises(ContractError):
            self.workspace.resolve(rejected)

    def test_real_process_exit_releases_kernel_lock_without_age_guessing(self):
        code = "from control_harness.artifacts import Workspace; import sys,os; w=Workspace(sys.argv[1]); c=w.lock('child'); c.__enter__(); print('locked',flush=True); sys.stdin.read(1); os._exit(17)"
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        child = subprocess.Popen([sys.executable, "-u", "-c", code, str(self.workspace.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            self.assertEqual(child.stdout.readline().strip(), "locked")
            with self.assertRaises(WorkspaceBusy):
                with self.workspace.lock("conflicting"):
                    pass
            # Windows venv launchers can outlive or precede their interpreter;
            # trigger abrupt exit inside the process that actually owns the lock.
            child.stdin.write("x")
            child.stdin.flush()
            child.wait(timeout=15)
            with self.workspace.lock("recover_confirmed_dead_child"):
                self.assertEqual(read_json(self.workspace.lock_path)["pid"], os.getpid())
            self.assertTrue(list((self.workspace.path / "recovery").glob("lock-*.json")))
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=15)


if __name__ == "__main__":
    unittest.main()
