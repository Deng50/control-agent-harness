"""S10 shared dispatch/confirmation/handoff and real inner checkpoint recovery.

Real sources, collections, NN training and Python closed loops are exercised.
One explicitly labelled dispatch-only fixture tests a prepared-dict return; it
does not claim MATLAB execution.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Stage, Workspace, atomic_json, file_hash, read_json
from control_harness.case_plans import FIXTURE_KIND, PendingPlanConfirmation, propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.contracts import ContractError, DataInterface
from control_harness.external_expert import register_expert, collect_external_expert
from control_harness.external_expert_data import publish_external_expert_dataset
from control_harness.external_imitation import train_external_imitation
from control_harness.inspection import prepare_handoff
from control_harness.material_registration import register_material
from control_harness.operations import OPERATIONS, run_operation
from control_harness.plants import register_plant
from control_harness.runtime import OperationContext


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-output" / "external-imitation-operation-tests"
HARD_EXIT = r'''
import json,os,sys,torch
from pathlib import Path
from control_harness.artifacts import Workspace,atomic_json
from control_harness.external_imitation import train_external_imitation
from control_harness.runtime import OperationContext
torch.set_num_threads(1)
root=Path(sys.argv[1]);request=json.loads((root/'hard-exit-request.json').read_text())
def emit(event):
 if event.get('stage')=='supervised_training' and event.get('epoch')==1:
  atomic_json(root/'hard-exit-ready.json',{'stage_id':Path(event['checkpoint']).parent.name,'pid':os.getpid()})
  os._exit(73)
train_external_imitation(Workspace(root),request['dataset'],request['config'],OperationContext(emit=emit))
raise RuntimeError('Expected hard exit at an actual committed S05 epoch')
'''


class ExternalImitationOperations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        cls.fixture_temp = tempfile.TemporaryDirectory(dir=OUTPUT); cls.fixture = Path(cls.fixture_temp.name)
        cls.original_threads = torch.get_num_threads(); torch.set_num_threads(1)
        cls.source_ws = Workspace(cls.fixture / "workspace")
        var = lambda name, bound: {"name": name, "unit": "1", "lower": -bound, "upper": bound}
        cls.physical = {"schema_version": 1, "name": "shared actual scalar Plant", "sample_time": .1,
            "time_semantics": "discrete_state_transition", "states": [var("state", 3)], "controls": [var("command", 1)],
            "outputs": [var("output", 3)], "disturbances": []}
        plant = cls.fixture / "plant.py"; plant.write_text("import numpy as np\ndef step(x,u,d,dt,p):\n return np.array([.7*x[0]+.2*u[0]])\n", encoding="utf-8")
        material = register_material(cls.source_ws, str(plant), {}, OperationContext())
        cls.plant = register_plant(cls.source_ws, material, {"adapter": {"schema_version": 1, "backend": "python_callable",
            "source_file": "source/plant.py", "entrypoint": "step", "interface": cls.physical, "parameters": {}, "source_semantics": "transition", "C": [[1.]]}}, OperationContext())
        expert = cls.fixture / "expert.py"; expert.write_text("import numpy as np\ndef act(o,r,d,previous,memory,p):\n return np.array([.6*(r[0]-o[0])+.05*previous[0]]),{'calls':memory['calls']+1},'ok'\n", encoding="utf-8")
        material = register_material(cls.source_ws, str(expert), {}, OperationContext())
        spec = {"schema_version": 1, "backend": "python_callable", "source_file": "source/expert.py", "entrypoint": "act", "interface": cls.physical,
            "parameters": {}, "time_unit": "s", "control_period": .2, "observations": [{"name": "state", "source": "state", "index": 0,
                "plant_name": "state", "unit": "1", "availability": "measured", "delay_steps": 0}], "reference_available": True,
            "disturbance_available": True, "previous_action_available": True, "initial_memory": {"calls": 0},
            "action_mapping": [{"action_index": 0, "plant_control_index": 0, "plant_control_name": "command", "unit": "1"}], "failure_policy": "stop", "input_rate_bounds": [[-.4, .4]]}
        cls.expert = register_expert(cls.source_ws, material, {"adapter": spec})
        scenarios = [{"name": f"condition-{k}", "initial_state": [.05 + .04 * k], "reference": [.3], "disturbance": [], "steps": 3,
            "initial_previous_action": [0.], "split": "train" if k < 3 else "validation" if k == 3 else "test"} for k in range(6)]
        cls.dataset = collect_external_expert(cls.source_ws, cls.expert, cls.plant, {"scenarios": scenarios, "seed": 19})
        cls.config = {"network": {"hidden_dims": [4], "activation": "tanh", "epochs": 3, "learning_rate": .005,
            "batch_size": 4, "patience": 20, "seed": 20261120}}
        cls.policy = train_external_imitation(cls.source_ws, cls.dataset, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup(); torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name); shutil.copytree(self.source_ws.path, self.root / "case")
        self.ws = Workspace(self.root / "case")
        atomic_json(self.ws.path / ".case-plans/test-fixture.json", {"kind": FIXTURE_KIND, "workspace": str(self.ws.path)})
        self.context = OperationContext.test_fixture(self.ws)

    def request(self, operation, inputs=None, config=None):
        return {"operation": operation, "workspace": str(self.ws.path), "inputs": inputs or {}, "config": config or {}}

    def approve(self, request):
        refs = {}
        for requirement in confirmation_requirements(self.ws, request):
            plan = propose_plan(self.ws, requirement["scope"], requirement["expected_payload"], requirement["bindings"])
            refs[requirement["scope"]] = {key: plan[key] for key in ("plan_id", "version_hash")}
        request["config"]["plan_refs"] = refs

    def proposal(self, policy=None):
        return run_operation(self.request("prepare_imitation_evaluation", {"policy": policy or self.policy}), OperationContext())

    def test_public_train_requires_actual_scopes_then_reports_real_training(self):
        self.assertTrue({"train_external_imitation", "prepare_imitation_evaluation", "evaluate_external_imitation"}.issubset(OPERATIONS))
        request = self.request("train_external_imitation", {"dataset": self.dataset}, copy.deepcopy(self.config))
        requirements = confirmation_requirements(self.ws, request); self.assertEqual([row["scope"] for row in requirements], ["signals", "training"])
        before = set(self.ws.manifest["artifacts"])
        with self.assertRaises(PendingPlanConfirmation): run_operation(request, OperationContext())
        self.assertEqual(set(self.ws.manifest["artifacts"]), before)
        self.approve(request); result = run_operation(request, self.context)
        self.assertEqual(result["kind"], "external_imitation_policy")
        report = read_json(Path(result["artifact_path"]) / "report.json")
        self.assertEqual(report["network"]["trained_epochs"], 3)
        self.assertEqual([x["scope"] for x in report["confirmation_sources"]], ["signals", "training"])
        self.assertTrue(all(x["confirmation_source"] == "test_fixture" for x in report["confirmation_sources"]))

    def test_read_only_prepare_returns_concrete_heldout_conditions_without_admission(self):
        request = self.request("prepare_imitation_evaluation", {"policy": self.policy})
        self.assertEqual(confirmation_requirements(self.ws, request), [])
        before = copy.deepcopy(self.ws.manifest)
        result = run_operation(request, OperationContext())
        self.assertEqual(result["status"], "proposal"); self.assertFalse(result["source_executed"])
        self.assertEqual(self.ws.manifest, before)
        self.assertEqual(result["resolved_source_trajectory_ids"], [3, 4, 5])
        self.assertEqual(len(result["config_suggestions"]["scenarios"]), 3)
        self.assertEqual(result["control_period"], .2); self.assertEqual(result["plant_dt"], .1)

    def test_eval_empty_scenarios_or_missing_authority_are_rejected_before_numerics(self):
        before = set(self.ws.manifest["artifacts"])
        for settings in ({}, {"scenarios": []}, {"scenarios": None}):
            request = self.request("evaluate_external_imitation", {"policy": self.policy}, settings)
            with self.subTest(settings=settings), self.assertRaisesRegex(ContractError, "explicit evaluation scenarios"):
                confirmation_requirements(self.ws, request)
            with self.assertRaisesRegex(ContractError, "explicit evaluation scenarios"): run_operation(request, self.context)
        request = self.request("evaluate_external_imitation", {"policy": self.policy}, self.proposal()["config_suggestions"])
        with self.assertRaises(PendingPlanConfirmation): run_operation(request, OperationContext())
        self.assertEqual(set(self.ws.manifest["artifacts"]), before)

    def test_confirmed_explicit_original_plant_evaluation_keeps_weights_and_active_policy(self):
        config = self.proposal()["config_suggestions"]; config["benchmark"] = {"warmup": 1, "repeats": 2, "threads": 1, "batch_size": 1}
        request = self.request("evaluate_external_imitation", {"policy": self.policy}, config)
        requirements = confirmation_requirements(self.ws, request)
        self.assertEqual([row["scope"] for row in requirements], ["signals", "sampling"])
        self.assertEqual(requirements[1]["expected_payload"]["scenarios"], config["scenarios"])
        self.approve(request); before = copy.deepcopy(self.ws.manifest["active"])
        policy_root, _ = self.ws.resolve(self.policy); weight_hash = file_hash(policy_root / "bundle/model.pt")
        changed = copy.deepcopy(request); changed["config"]["scenarios"][0]["steps"] += 1
        with self.assertRaises(PendingPlanConfirmation): run_operation(changed, self.context)
        result = run_operation(request, self.context)
        self.assertEqual(result["kind"], "policy_evaluation")
        root = Path(result["artifact_path"]); report = read_json(root / "report.json")
        self.assertEqual(report["status"], "completed"); self.assertEqual(report["plant"], self.plant)
        self.assertEqual(self.ws.manifest["active"], before); self.assertEqual(file_hash(policy_root / "bundle/model.pt"), weight_hash)
        for name in ("expert", "imitation"):
            with np.load(root / (name + "-rollout.npz")) as data:
                np.testing.assert_allclose(data["next_x"], .49 * data["x"] + .34 * data["u"], rtol=0, atol=1e-12)
        self.assertEqual(prepare_handoff(self.ws, result["artifact_id"])["handoff"]["operation"], None)

    def test_inspection_offers_training_then_optional_prepare_without_auto_pgac(self):
        data = prepare_handoff(self.ws, self.dataset)
        self.assertEqual(data["handoff"]["operation"], "train_external_imitation")
        self.assertIn("network", data["handoff"]["config_suggestions"])
        policy = prepare_handoff(self.ws, self.policy)
        self.assertEqual(policy["status"], "complete"); self.assertIsNone(policy["handoff"]["operation"])
        self.assertEqual(policy["optional_handoffs"][0]["operation"], "prepare_imitation_evaluation")
        self.assertFalse(any(x.get("operation") == "optimize_policy" for x in policy["optional_handoffs"]))

    def test_matlab_prepared_dictionary_is_a_dispatch_protocol_result_not_an_artifact_id(self):
        request = self.request("evaluate_external_imitation", {"policy": self.policy}, self.proposal()["config_suggestions"]); self.approve(request)
        before = set(self.ws.manifest["artifacts"])
        prepared = {"status": "verified", "matlab_phase": "prepared", "batch_kind": "external_policy_evaluation", "test_scope": "dispatch_only_not_MATLAB_execution"}
        with patch("control_harness.external_policy_evaluation.evaluate_external_imitation", return_value=prepared) as dispatch:
            result = run_operation(request, self.context)
        self.assertEqual(result, prepared); self.assertEqual(dispatch.call_args.args[-1], request)
        self.assertEqual(set(self.ws.manifest["artifacts"]), before)

    def test_public_data_only_prepare_is_not_applicable_and_empty_eval_cannot_execute(self):
        source_root, meta = self.ws.resolve(self.dataset); data = read_json(source_root / "dataset.json")
        with np.load(source_root / "expert.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        source = publish_external_expert_dataset(self.ws, arrays, DataInterface.parse(meta["interface"]),
            read_json(source_root / "split.json")["groups"], {"scaler": "standard", "action_constraints": read_json(source_root / "action_constraints.json")},
            OperationContext(), source={"kind": "recorded_external_expert", "description": "Copied actual expert rows without an executable-source claim",
                "previous_action_semantics": "actual_applied_action_before_current_decision"}, source_mapping=data["source_mapping"])
        policy = train_external_imitation(self.ws, source, self.config)
        before = copy.deepcopy(self.ws.manifest)
        prepared = self.proposal(policy)
        self.assertEqual(prepared["status"], "not_applicable"); self.assertFalse(prepared["source_executed"])
        self.assertIsNone(prepared["config_suggestions"]); self.assertEqual(self.ws.manifest, before)
        with self.assertRaisesRegex(ContractError, "explicit evaluation scenarios"):
            run_operation(self.request("evaluate_external_imitation", {"policy": policy}), self.context)
        self.assertEqual(self.ws.manifest, before)

    def test_real_hard_exit_recovers_only_actual_inner_supervised_checkpoint_and_resumes(self):
        atomic_json(self.ws.path / "hard-exit-request.json", {"dataset": self.dataset, "config": self.config})
        child = subprocess.run([sys.executable, "-B", "-c", HARD_EXIT, str(self.ws.path)], capture_output=True, text=True,
            timeout=45, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        self.assertEqual(child.returncode, 73, child.stdout + child.stderr)
        marker = read_json(self.ws.path / "hard-exit-ready.json"); self.assertTrue(marker["stage_id"].startswith("supervised_model-"))
        result = run_operation(self.request("recover_interrupted_stage", {"stage_id": marker["stage_id"]}), self.context)
        self.assertEqual(result["checkpoint_kind"], "supervised_model"); self.assertEqual(result["algorithm"], "supervised_training")
        self.assertTrue(Path(result["checkpoint"]).is_file()); self.assertEqual(result["epoch"], 1)
        request = self.request("train_external_imitation", {"dataset": self.dataset}, {**copy.deepcopy(self.config),
            "resume_checkpoint": result["checkpoint"], "resume_checkpoint_sha256": result["checkpoint_sha256"]})
        self.approve(request); trained = run_operation(request, self.context)
        report = read_json(Path(trained["artifact_path"]) / "report.json")
        self.assertEqual(report["network"]["resumed_from_epoch"], 1)
        old_report = read_json(self.ws.resolve(self.policy)[0] / "report.json") if self.ws.manifest["artifacts"][self.policy]["status"] == "verified" else read_json(self.source_ws.resolve(self.policy)[0] / "report.json")
        actual_root, _ = self.ws.resolve(report["supervised_model"])
        expected_root, _ = self.source_ws.resolve(old_report["supervised_model"])
        actual = torch.load(actual_root / "latest.pt", weights_only=True); expected = torch.load(expected_root / "latest.pt", weights_only=True)
        self.assertEqual(actual["contract"], expected["contract"]); self.assertEqual(actual["updates"], expected["updates"])
        self.assertTrue(all(torch.equal(actual["model_state"][key], value) for key, value in expected["model_state"].items()))

    def test_outer_package_without_optimizer_is_not_reported_as_recoverable(self):
        root, metadata = self.ws.resolve(self.policy); interface = DataInterface.parse(metadata["interface"])
        with self.ws.lock("test_fixture_outer_package"):
            stage = Stage(self.ws, "external_imitation_policy", interface, [], {"test_fixture": "unpublished package; no optimizer"})
        with self.assertRaisesRegex(ContractError, "actual supervised_model/supervised_training"):
            run_operation(self.request("recover_interrupted_stage", {"stage_id": stage.id}), self.context)
        self.assertTrue(stage.root.is_dir()); self.assertFalse((self.ws.path / "failed" / stage.id).exists())


if __name__ == "__main__": unittest.main()
