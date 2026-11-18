"""Actual single-plant artifacts, confirmation reuse and failure preservation."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.case_plans import FIXTURE_KIND, PendingPlanConfirmation, propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.contracts import ContractError
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
from control_harness.sampling import SamplingStopped, SamplingCancelled


def interface():
    variable = lambda name: {"name": name, "unit": "1", "lower": -5., "upper": 5.}
    return {"schema_version": 1, "name": "single plant", "time_semantics": "discrete_state_transition",
        "sample_time": .2, "states": [variable("x0"), variable("x1")],
        "controls": [variable("u")], "outputs": [variable("position")], "disturbances": []}


class PlantOperations(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / "test-output/plant-operation-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root / "case")
        atomic_json(self.ws.path / ".case-plans/test-fixture.json", {"kind": FIXTURE_KIND, "workspace": str(self.ws.path)})
        self.ctx = OperationContext.test_fixture(self.ws)

    def operation(self, operation, inputs=None, config=None, context=None):
        return run_operation({"operation": operation, "workspace": str(self.ws.path),
            "inputs": inputs or {}, "config": config or {}}, context or self.ctx)

    def plant(self, body=None):
        source = self.root / "plant.py"
        source.write_text(body or "import numpy as np\ndef transition(x,u,d,dt,p):\n    return np.array([.8*x[0]+.1*x[1]+.2*u[0],.9*x[1]-.1*u[0]],dtype=np.float64)\n", encoding="utf-8")
        material = self.operation("register_material", {"local_path": str(source)})["artifact_id"]
        spec = {"schema_version": 1, "backend": "python_callable", "source_file": "source/plant.py",
            "interface": interface(), "parameters": {}, "source_semantics": "transition", "entrypoint": "transition", "C": [[1., 0.]]}
        return self.operation("register_plant", {"material": material}, {"adapter": spec})["artifact_id"]

    def request(self, plant):
        return {"operation": "sample_model", "workspace": str(self.ws.path), "inputs": {"model": plant},
            "config": {"sampling": {"mode": "expert", "seed": 17, "dt": .2, "steps": 12,
                "trajectories": [{"x0": [.1 * k, -.02 * k]} for k in range(6)], "u0": [0.], "d0": [],
                "u_profile": {"kind": "step", "times": [0., 1.2], "values": [[.1], [-.1]]},
                "d_profile": None, "control_rate_limits": [1.], "disturbance_rate_limits": [], "coupled_constraints": []},
                "trial": {"trajectories": 1, "steps": 3}, "sampling_run": "trial"}}

    def approve(self, request):
        refs = {}
        for row in confirmation_requirements(self.ws, request):
            plan = propose_plan(self.ws, row["scope"], row["expected_payload"], row["bindings"])
            refs[row["scope"]] = {key: plan[key] for key in ("plan_id", "version_hash")}
        request["config"]["plan_refs"] = refs

    def test_registration_and_recommendation_never_execute_source(self):
        plant = self.plant("raise RuntimeError('source must not run at registration')\ndef transition(x,u,d,dt,p): return x\n")
        root, metadata = self.ws.resolve(plant)
        self.assertFalse(metadata["source"]["source_executed"])
        self.assertEqual(metadata["activation"], "none")
        candidate = self.operation("recommend_sampling", {"model": plant}, {"overrides": {"steps": 4}})
        self.assertEqual(candidate["status"], "proposal")
        self.assertFalse(candidate["source_executed"])
        self.assertEqual(candidate["sampling"]["steps"], 4)
        self.assertEqual(self.operation("prepare_handoff", {"artifact_id": plant})["handoff"]["operation"], "recommend_sampling")

    def test_same_confirmed_plan_trial_and_full_match_independent_recurrence(self):
        plant = self.plant(); request = self.request(plant); self.approve(request)
        before = confirmation_requirements(self.ws, request)
        trial = run_operation(request, self.ctx)
        trial_root, trial_meta = self.ws.resolve(trial["artifact_id"])
        self.assertEqual(trial_meta["kind"], "sampling_trial")
        self.assertFalse((trial_root / "split.json").exists())
        handoff = self.operation("prepare_handoff", {"artifact_id": trial["artifact_id"]})["handoff"]
        self.assertEqual(handoff["inputs"], {"model": plant})
        self.assertEqual(handoff["config_suggestions"]["sampling_run"], "full")
        with np.load(trial_root / "transitions.npz") as data:
            trial_x = data["next_x"].copy()
        request["config"]["sampling_run"] = "full"
        self.assertEqual(before, confirmation_requirements(self.ws, request))
        result = run_operation(request, self.ctx)
        root, meta = self.ws.resolve(result["artifact_id"])
        self.assertEqual(meta["kind"], "sampled_dataset")
        handoff = self.operation("prepare_handoff", {"artifact_id": result["artifact_id"]})["handoff"]
        self.assertEqual(handoff["operation"], "train_predictive_model")
        self.assertEqual(handoff["config_suggestions"]["C"], [[1., 0.]])
        self.assertEqual([parent["id"] for parent in meta["parents"]], [plant])
        with np.load(root / "transitions.npz", allow_pickle=False) as data:
            self.assertEqual(data["x"].shape, (72, 2)); self.assertEqual(data["d"].shape, (72, 0))
            expected = []
            for k in range(6):
                x = np.array([.1*k, -.02*k])
                for j in range(12):
                    u = .1 if j < 6 else -.1
                    x = np.array([.8*x[0]+.1*x[1]+.2*u, .9*x[1]-.1*u])
                    expected.append(x)
            np.testing.assert_allclose(data["next_x"], expected, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(data["next_y"], data["next_x"][:, :1])
            np.testing.assert_array_equal(trial_x, data["next_x"][:3])
        split = read_json(root / "split.json")
        self.assertEqual(set(sum(split.values(), [])), set(range(6)))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        report = read_json(root / "report.json")
        self.assertEqual(report["completed_trajectories"], 6)
        self.assertFalse(meta["source"]["measured"])
        self.assertEqual(report["confirmation_sources"][0]["confirmation_source"], "test_fixture")

    def test_pending_json_flags_and_changed_budget_are_refused(self):
        request = self.request(self.plant())
        with self.assertRaises(PendingPlanConfirmation): run_operation(request, OperationContext())
        self.approve(request)
        request["config"]["confirmed"] = True
        with self.assertRaises(PendingPlanConfirmation): run_operation(request, OperationContext())
        request["config"].pop("confirmed")
        request["config"]["trial"]["steps"] += 1
        with self.assertRaises(ContractError): run_operation(request, self.ctx)

    def test_invalid_commands_are_rejected_before_any_source_import(self):
        request = self.request(self.plant("raise RuntimeError('unexpected source execution')\ndef transition(x,u,d,dt,p): return x\n"))
        request["config"]["sampling"]["u_profile"]["values"][0] = [6.]
        self.approve(request)
        with self.assertRaisesRegex(ContractError, "bounds|bound"):
            run_operation(request, self.ctx)
        self.assertFalse((self.ws.path / "failed").exists())

    def test_runtime_divergence_retains_only_valid_prefix_and_reason(self):
        plant = self.plant("import numpy as np\ndef transition(x,u,d,dt,p):\n    return np.array([x[0]+2.,x[1]],dtype=np.float64)\n")
        request = self.request(plant); request["config"]["sampling_run"] = "full"; self.approve(request)
        with self.assertRaises(SamplingStopped): run_operation(request, self.ctx)
        failed = list((self.ws.path / "failed").iterdir()); self.assertEqual(len(failed), 1)
        report = read_json(failed[0] / "report.json")
        self.assertEqual(report["failure"]["step_index"], 2)
        self.assertEqual(report["completed_trajectories"], 0)
        with np.load(failed[0] / "partial-transitions.npz") as data:
            self.assertEqual(len(data["x"]), 2); self.assertTrue(np.all(data["next_x"] <= 5))
        self.assertNotIn("sampled_dataset", self.ws.manifest["active"])

    def test_cancel_preserves_partial_without_a_published_dataset(self):
        request = self.request(self.plant()); self.approve(request)
        count = [0]
        def cancelled():
            count[0] += 1
            return count[0] >= 7
        context = OperationContext.test_fixture(self.ws, cancelled=cancelled)
        with self.assertRaises(SamplingCancelled): run_operation(request, context)
        failed = next((self.ws.path / "failed").iterdir())
        report = read_json(failed / "report.json")
        self.assertEqual(report["status"], "cancelled")
        self.assertTrue((failed / "partial-transitions.npz").is_file())
        self.assertNotIn("sampled_dataset", self.ws.manifest["active"])

    def test_existing_torch_model_samples_without_an_extra_nominal_or_reference(self):
        from control_harness.modeling import register_model
        spec = {"representation": "discrete_polynomial", "interface": interface(),
            "A": [[.8,.1],[0.,.9]], "B": [[.2],[-.1]], "C": [[1.,0.]]}
        model = register_model(self.ws, spec, {"kind": "user_nominal", "description": "Declared single Torch plant"})
        request = self.request(model); request["config"]["sampling_run"] = "full"; self.approve(request)
        result = run_operation(request, self.ctx)
        root, metadata = self.ws.resolve(result["artifact_id"])
        self.assertEqual(metadata["source"]["backend"], "registered_torch")
        self.assertEqual([parent["id"] for parent in metadata["parents"]], [model])
        with np.load(root / "transitions.npz") as arrays:
            self.assertEqual(arrays["x"].shape, (72,2))
            np.testing.assert_allclose(arrays["next_x"], arrays["x"] @ np.array(spec["A"]).T + arrays["u"] @ np.array(spec["B"]).T, atol=1e-15)


if __name__ == "__main__": unittest.main()
