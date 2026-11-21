"""Actual single-plant artifacts, confirmation reuse and failure preservation."""
import copy
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.case_plans import FIXTURE_KIND, PendingPlanConfirmation, propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.contracts import ContractError
from control_harness.modeling import load_model, register_model
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
        self.assertNotIn("plan_refs", handoff["config_suggestions"])
        self.assertEqual(handoff["config_suggestions"]["trial"], request["config"]["trial"])
        with np.load(trial_root / "transitions.npz") as data:
            trial_x = data["next_x"].copy()
        request["config"]["sampling_run"] = "full"
        self.assertEqual(before, confirmation_requirements(self.ws, request))
        result = run_operation(request, self.ctx)
        root, meta = self.ws.resolve(result["artifact_id"])
        self.assertEqual(meta["kind"], "sampled_dataset")
        prepared = self.operation("prepare_handoff", {"artifact_id": result["artifact_id"]})
        handoff = prepared["handoff"]
        self.assertEqual(handoff["operation"], "train_predictive_model")
        affine, = [row for row in prepared["optional_handoffs"] if row["config_suggestions"]["method"]=="sampled_affine"]
        self.assertEqual(affine["inputs"], {"plant": plant, "dataset": result["artifact_id"]})
        equations, = [row for row in prepared["optional_handoffs"] if row["config_suggestions"]["method"]=="declared_equations"]
        self.assertEqual(equations["missing_inputs"], ["equations"])
        source = self.operation("prepare_handoff", {"artifact_id":plant})
        self.assertFalse(source["artifact_capabilities"]["prediction"])
        self.assertTrue(source["artifact_capabilities"]["sampling"])
        for route in source["optional_handoffs"]:
            self.assertEqual(route["inputs"]["dataset"], "")
            self.assertEqual(route["candidate_datasets"], [result["artifact_id"]])
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

    def test_promoted_mpc_collects_and_evaluates_on_exact_original_callable(self):
        from control_harness.delivery import verify_design_delivery
        from control_harness.design_only import synthesize_mpc
        from control_harness.inspection import prepare_handoff
        from control_harness.plant_predictor import convert_plant_predictor
        from control_harness.policy import collect_expert, train_imitation

        # 故意留下小且可检出的方程误差，区分预测器滚动轨迹与实际原对象轨迹。
        plant = self.plant()
        sampling = self.request(plant)
        sampling["config"]["sampling_run"] = "full"
        self.approve(sampling)
        dataset = run_operation(sampling, self.ctx)["artifact_id"]
        equations = register_model(self.ws, {"interface": interface(), "representation": "discrete_polynomial",
            "A": [[.8, .1], [0., .9]], "B": [[.2], [-.1]], "E": [[], []],
            "C": [[1., 0.]], "bias": [.001, 0.]}, {"kind": "user_nominal"})
        predictor = convert_plant_predictor(self.ws, {"plant": plant, "dataset": dataset, "equations": equations},
                                           {"method": "declared_equations"}, self.ctx)
        design = synthesize_mpc(self.ws, {"model": predictor}, {"controller": {"horizon": 1}, "solver_check": False})
        controller = verify_design_delivery(self.ws, design, {"target": "workspace_controller",
            "case": {"x": [.1, 0.], "reference": [0.]}})
        routes = prepare_handoff(self.ws, controller)
        self.assertEqual(routes["handoff"]["inputs"], {"controller": controller, "evaluation_model": plant})
        cases = {"scenarios": [{"name": str(k), "initial_state": [.02*k, -.01], "reference": [.01],
                                "disturbance": [], "steps": 2} for k in range(5)]}
        with self.assertRaisesRegex(ContractError, "直接绑定"):
            collect_expert(self.ws, controller, self.plant(), cases)
        expert = collect_expert(self.ws, controller, plant, cases)
        root, meta = self.ws.resolve(expert, kind="expert_dataset")
        self.assertEqual(meta["source"]["evaluation_model"], plant)
        self.assertEqual(meta["source"]["evaluation_backend"], "original_python_callable")
        with np.load(root / "expert.npz", allow_pickle=False) as observed:
            x, u = observed["x"][::2], observed["u"][::2, 0]
            expected = np.column_stack((.8*x[:,0]+.1*x[:,1]+.2*u, .9*x[:,1]-.1*u))
            np.testing.assert_allclose(observed["x"][1::2], expected, rtol=0, atol=1e-14)
        policy = train_imitation(self.ws, expert, {"epochs": 1, "hidden_dims": [3], "seed": 7})
        policy_root, _ = self.ws.resolve(policy, kind="imitation_policy")
        report = read_json(policy_root / "report.json")
        self.assertEqual(report["evaluation_backend"], "original_python_callable")
        self.assertEqual(len(report["closed_loop"]["test"]), 1)
        self.assertIsNone(prepare_handoff(self.ws, policy)["handoff"]["operation"])

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

    def test_resume_reuses_only_completed_trajectory_and_rechecks_its_bytes(self):
        request = self.request(self.plant())
        request["config"]["sampling_run"] = "full"
        self.approve(request)
        receipts = []
        def interrupt(event):
            if "collection_checkpoint" in event:
                receipts.append(event["collection_checkpoint"])
        context = OperationContext.test_fixture(self.ws, emit=interrupt, cancelled=lambda: bool(receipts))
        context._case_receipt_key = base64.b64encode(os.urandom(32)).decode()
        context._case_native_session = "isolated-sampling-test"
        with self.assertRaises(SamplingCancelled):
            run_operation(request, context)
        self.assertEqual(receipts[0]["completed"], 1)
        self.assertNotIn("sampled_dataset", self.ws.manifest["active"])
        context.cancelled = lambda: False
        started = []
        context.emit = lambda event: started.append(event.get("trajectory_id")) if event.get("phase") == "started" else None
        with patch.dict(os.environ, {"CONTROL_COLLECTION_RESUME": json.dumps(receipts[0])}):
            result = run_operation(request, context)
            self.assertEqual(started, [1, 2, 3, 4, 5])
            report = read_json(Path(result["artifact_path"]) / "report.json")
            self.assertEqual(report["restored_trajectories"], 1)
            self.assertEqual(report["valid_rows"], 72)
            data = self.ws.path / "failed" / receipts[0]["stage_id"] / "trajectory-commits" / "0.npz"
            data.write_bytes(data.read_bytes() + b"changed")
            with self.assertRaisesRegex(SamplingStopped, "hash changed"):
                run_operation(request, context)

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

    def full_dataset(self, plant, config=None):
        # 复用采样工具、确认门禁和独立轨迹切分。测试夹具的确认来源记录为 test_fixture。
        request = self.request(plant)
        if config is not None:
            request["config"] = config
        request["config"]["sampling_run"] = "full"
        self.approve(request)
        return run_operation(request, self.ctx)["artifact_id"]

    def convert(self, inputs, config):
        request = {"operation": "convert_plant_predictor", "workspace": str(self.ws.path),
                   "inputs": inputs, "config": dict(config)}
        self.approve(request)
        return run_operation(request, self.ctx)["artifact_id"]

    def test_converted_polynomial_preserves_nonlinearity_and_rejects_inaccurate_affine(self):
        plant = self.plant("import numpy as np\ndef transition(x,u,d,dt,p):\n"
            "    return np.array([.8*x[0]+.1*x[1]+.2*u[0]+.1*x[0]**2,.9*x[1]-.1*u[0]],dtype=np.float64)\n")
        dataset = self.full_dataset(plant)
        spec = {"representation": "discrete_polynomial", "interface": interface(), "A": [[.8,.1],[0.,.9]],
                "B": [[.2],[-.1]], "C": [[1.,0.]], "terms": [{"state_index": 0, "powers": [2,0,0], "coefficient": .1}]}
        equations = register_model(self.ws, spec, {"kind": "user_nominal"})
        inputs = {"plant": plant, "dataset": dataset, "equations": equations}
        strict = {"max_one_step_nrmse": 1e-12, "max_rollout_nrmse": 1e-12}
        with self.assertRaises(PendingPlanConfirmation):
            self.operation("convert_plant_predictor", inputs, {"method": "declared_equations", **strict}, OperationContext())
        result = self.convert(inputs, {"method": "declared_equations", **strict})
        model = load_model(self.ws, result)
        self.assertEqual(model.terms, spec["terms"])
        np.testing.assert_allclose(model.step([.3,.1], [.1], []), [.279,.08], atol=1e-14)
        root, meta = self.ws.resolve(result)
        self.assertEqual(meta["validation"]["status"], "verified")
        self.assertEqual(meta["source"]["parent_versions"], {key: self.ws.manifest["artifacts"][key]["metadata_hash"] for key in inputs.values()})
        report = read_json(root / "report.json")
        self.assertTrue(report["acceptance"]["passed"])
        self.assertEqual(report["confirmation_sources"][0]["confirmation_source"], "test_fixture")
        self.assertFalse(meta["source"]["applicability"]["global_equivalence_proved"])
        # 同一非线性对象的仿射近似在严格门限下应拒收；仍保留失败报告供用户定位误差。
        affine = self.convert({"plant": plant, "dataset": dataset}, {"method": "sampled_affine", **strict})
        rejected, meta = self.ws.resolve(affine, allow_rejected=True)
        self.assertEqual(meta["validation"]["status"], "rejected")
        self.assertTrue(read_json(rejected / "report.json")["acceptance"]["reason"])
        with self.assertRaises(ContractError):
            load_model(self.ws, affine)
        self.assertEqual(self.ws.manifest["active"]["nominal_model"], equations)
        self.assertNotIn("identified_model", self.ws.manifest["active"])

    def test_conversion_refuses_another_plant_version_and_missing_output_mapping(self):
        plant = self.plant()
        dataset = self.full_dataset(plant)
        custom = self.plant("import numpy as np\ndef transition(x,u,d,dt,p): return .8*x + u[0]\n"
                            "def measure(x,p): return np.array([x[0]**2],dtype=np.float64)\n")
        root, _ = self.ws.resolve(custom)
        definition = read_json(root / "plant.json")
        with self.assertRaisesRegex(ContractError, "原对象.*版本"):
            self.convert({"plant": custom, "dataset": dataset}, {"method": "sampled_affine"})
        # 自定义非线性测量函数可用于采样，但仅凭输入输出维度无法得到状态投影矩阵。
        adapter = definition["adapter_spec"]
        del adapter["C"]
        adapter["output_entrypoint"] = "measure"
        custom_output = self.operation("register_plant", {"material": definition["material"]}, {"adapter": adapter})["artifact_id"]
        data = self.full_dataset(custom_output)
        with self.assertRaisesRegex(ContractError, "输出映射 C"):
            self.convert({"plant": custom_output, "dataset": data}, {"method": "sampled_affine"})

    def test_b03_actual_callable_through_equations_and_sampled_affine(self):
        inputs_dir = Path(__file__).resolve().parents[2] / "control-agent-benchmark/cases/B03_mpc_preferences/inputs"
        if not inputs_dir.is_dir():
            self.skipTest("B03 benchmark materials are not installed next to this checkout")
        material = self.operation("register_material", {"local_path": str(inputs_dir / "plant_numpy.py")})["artifact_id"]
        plant = self.operation("register_plant", {"material": material}, read_json(inputs_dir / "plant_adapter.json"))["artifact_id"]
        dataset = self.full_dataset(plant, read_json(inputs_dir / "identification_sampling_config.json"))
        equations = register_model(self.ws, read_json(inputs_dir / "declared_continuous_model.json"), {"kind": "user_nominal"})
        expected = read_json(inputs_dir / "declared_discrete_model.json")
        # 用独立提供的离散矩阵校验未参加拟合的状态与力组合，防止只检查“产物已登记”。
        x, u, d = [.05, -.02], [.23], [-.1]
        expected_next = np.asarray(expected["A"]) @ x + np.asarray(expected["B"]) @ u + np.asarray(expected["E"]) @ d
        for method in ("declared_equations", "sampled_affine"):
            inputs = {"plant": plant, "dataset": dataset}
            if method == "declared_equations":
                inputs["equations"] = equations
            with self.subTest(method=method):
                result = self.convert(inputs, {"method": method})
                root, meta = self.ws.resolve(result)
                model = load_model(self.ws, result)
                np.testing.assert_allclose(model.step(x, u, d), expected_next, rtol=0, atol=1e-12)
                self.assertEqual(meta["source"]["material"], material)
                self.assertEqual(meta["source"]["applicability"]["parameter_values"], {"mass": 1., "damping": .8, "stiffness": 2.})
                report = read_json(root / "report.json")
                self.assertEqual(sum(map(len, report["trajectory_split"].values())), 15)
                for part in ("validation", "test"):
                    for record in report["rollout"][part]:
                        self.assertEqual(record["steps"], 100)
                        self.assertFalse(record["teacher_forcing_after_initial_state"])
                        self.assertLess(max(record["state"]["nrmse_per_channel"]), 1e-12)
                if method == "declared_equations":
                    self.assertEqual(model.spec["integration"]["method"], "zoh")
                else:
                    fit = report["identification"]
                    self.assertEqual(fit["training_rank"], fit["parameter_count"])


if __name__ == "__main__": unittest.main()
