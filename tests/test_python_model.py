"""Real frozen Python dynamics, differentiability and the same downstream stages."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from control_harness.artifacts import Workspace, read_json
from control_harness.contracts import ContractError
from control_harness.estimation import estimator_config
from control_harness.model_inspection import inspect_model_source
from control_harness.modeling import load_model, tensor
from control_harness.operations import run_operation
from control_harness.python_model import (PythonTensorDynamics, python_spec, register_python_model,
                                         validate_python_spec)
from control_harness.workflow import plan_workflow, run_workflow

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = (PROJECT / "examples/python-tensor/model.py").read_text(encoding="utf-8")
SPEC = python_spec(SOURCE.encode())
TEMP = PROJECT / "test-output/python-model-tests"
TEMP.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


class PythonModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "model.py"
        self.path.write_text(SOURCE, encoding="utf-8")
        self.workspace = Workspace(Path(self.temp.name) / "workspace")

    def register(self, spec=None):
        return register_python_model(self.workspace, self.path, {"kind": "user_nominal"}, spec)

    def test_static_inspection_and_workflow_preflight_never_execute_source(self):
        self.path.write_text("raise RuntimeError('must not execute during inspection')\n"+SOURCE, encoding="utf-8")
        inspected = inspect_model_source(self.path)
        self.assertTrue(inspected["registration_ready"])
        self.assertEqual(inspected["registration_operation"], "register_python_model")
        self.assertFalse(inspected["candidates"][0]["runtime_execution_verified"])
        plan = plan_workflow(self.workspace, {}, {"schema_version": 1, "steps": [{"id": "source", "operation": "register_python_model",
            "inputs": {"model_path": str(self.path)}, "config": {"source": {"kind": "user_nominal"}}}]})
        self.assertEqual(plan["status"], "ready")
        self.assertFalse(self.workspace.manifest["artifacts"])
        with self.assertRaisesRegex(RuntimeError, "must not execute"):
            self.register()

    def test_registration_uses_frozen_source_and_real_nonlinear_output_gradients(self):
        artifact = self.register()
        root, metadata = self.workspace.resolve(artifact)
        model = load_model(self.workspace, artifact)
        x, u, d = tensor([.4, -.2, .3]), tensor([.2, -.1]).requires_grad_(True), tensor([.05])
        prediction = model(x, u, d)
        gu, = torch.autograd.grad(prediction.sum(), u)
        self.assertTrue(torch.all(gu > .05))
        expected = [.78*.4+.05*.3+.22*np.tanh(.2)+.025*.05,
                    .76*(-.2)+.04*np.sin(.4)+.2*np.tanh(-.1)+.02*.05,
                    .7*.3+.06*np.tanh(.4*(-.2))+.06*(.2-.1)]
        np.testing.assert_allclose(prediction.detach(), expected, atol=1e-13)
        np.testing.assert_allclose(model.output(x), [.4+.02*np.sin(.3), -.2], atol=1e-13)
        self.assertEqual(estimator_config(model, {"initial_estimate": [0, 0, 0]})["method"], "EKF")
        self.assertTrue(metadata["source"]["explicit_source_execution"])
        self.path.write_text("raise RuntimeError('original has changed')", encoding="utf-8")
        np.testing.assert_allclose(load_model(self.workspace, artifact).step(x, u.detach(), d), expected, atol=1e-13)
        self.assertEqual((root / "source.py").read_text(encoding="utf-8"), SOURCE)
        validation = read_json(root / "python-validation.json")
        self.assertEqual(len(validation["probes"]), 3)
        self.assertLess(max(row["max_gradient_error"] for row in validation["probes"]), 1e-6)

    def test_detached_numpy_random_or_wrong_shape_functions_do_not_replace_valid_model(self):
        accepted = self.register()
        invalid = [SOURCE.replace("torch.tanh(u[..., 0])", "torch.tanh(u[..., 0].detach())"),
                   SOURCE.replace("return torch.stack((", "return torch.stack((", 1).replace("), dim=-1)", "), dim=-1).detach().numpy()", 1),
                   SOURCE.replace(".78*x[..., 0]", ".78*x[..., 0] + .01*torch.randn_like(x[..., 0])"),
                   SOURCE.replace("), dim=-1)", "), dim=0)", 1)]
        for source in invalid:
            with self.subTest(source=source[-70:]):
                self.path.write_text(source, encoding="utf-8")
                with self.assertRaises(ContractError):
                    self.register()
                self.assertEqual(self.workspace.active("nominal_model"), accepted)

    def test_bad_output_gradients_external_imports_and_runtime_mismatch_are_rejected(self):
        self.path.write_text(SOURCE.replace("x[..., 0] + .02*torch.sin(x[..., 2])", "x[..., 0].detach() + .02*torch.sin(x[..., 2])"), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "output-map"):
            self.register()
        with self.assertRaisesRegex(ContractError, "imports"):
            python_spec(("from local_project import dynamics\n"+SOURCE).encode())
        self.path.write_text(SOURCE, encoding="utf-8")
        artifact = self.register()
        root, _ = self.workspace.resolve(artifact)
        spec = read_json(root / "model.json")
        spec["runtime_dependencies"]["torch"] = "different-runtime"
        with self.assertRaisesRegex(ContractError, "runtime dependencies"):
            PythonTensorDynamics(spec, root / "source.py")
        (root / "source.py").write_text("raise RuntimeError('must not execute tampered source')", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "file missing or changed"):
            load_model(self.workspace, artifact)

    def test_random_import_initialization_is_not_a_frozen_physical_model(self):
        source = SOURCE.replace("import torch", "import torch\nRANDOM_GAIN = torch.rand((), dtype=torch.float64)", 1)
        source = source.replace(".78*x[..., 0]", "RANDOM_GAIN*x[..., 0]")
        self.path.write_text(source, encoding="utf-8")
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ContractError, "Independent Python imports"):
            self.register()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(self.workspace.manifest["artifacts"])

    def test_continuous_python_derivative_integrates_once_with_autograd(self):
        spec = copy.deepcopy(SPEC)
        spec.update(representation="python_torch_derivative", time_unit="s", derivative_units=["1/s"]*3,
                    integration={"method": "rk4", "substeps": 4, "input_hold": "zero_order_hold"})
        spec["interface"].update(time_semantics="continuous_state_derivative", sample_time=.15)
        # The same explicit callable now declares dx/dt rather than next_state.
        artifact = self.register(spec)
        model = load_model(self.workspace, artifact)
        from scipy.integrate import solve_ivp
        x, u, d = np.array([.4, -.2, .3]), tensor([.2, -.1]), tensor([.05])
        def rhs(t, state):
            a, b, c = state
            return [.78*a+.05*c+.22*np.tanh(.2)+.025*.05,
                    .76*b+.04*np.sin(a)+.2*np.tanh(-.1)+.02*.05,
                    .7*c+.06*np.tanh(a*b)+.06*(.2-.1)]
        expected = solve_ivp(rhs, (0, .15), x, rtol=1e-11, atol=1e-13).y[:, -1]
        np.testing.assert_allclose(model.step(x, u, d), expected, atol=1e-9)
        self.assertEqual(model.interface.time_semantics, "discrete_state_transition")
        self.assertEqual(model.transition_adapter["source_representation"], "python_torch_derivative")
        with self.assertRaises(ContractError):
            validate_python_spec({**spec, "integration": {"method": "zoh", "substeps": 1, "input_hold": "zero_order_hold"}})


class PythonModelChainTests(unittest.TestCase):
    setUp = PythonModelTests.setUp
    # A separate real chain tests the non-polynomial source across all consumers.
    def test_real_python_model_residual_mpc_expert_imitation_optimization_and_import(self):
        rng = np.random.default_rng(54)
        def scenarios(prefix, count, steps):
            return [{"name": f"{prefix}-{index}", "initial_state": rng.uniform(-.4,.4,3).tolist(),
                     "reference": rng.uniform(-.25,.25,2).tolist(), "disturbance": rng.uniform(-.15,.15,1).tolist(),
                     "steps": steps} for index in range(count)]
        reference = copy.deepcopy(SPEC)
        reference["parameters"] = {"gains": [.24,.18], "bias": [.01,-.008,.004]}
        workflow = {"schema_version": 1, "steps": [
            {"id":"nominal","operation":"register_python_model","inputs":{"model_path":str(self.path)},"config":{"source":{"kind":"user_nominal"}}},
            {"id":"reference","operation":"register_python_model","inputs":{"model_path":str(self.path)},"config":{"model_spec":reference,"source":{"kind":"reference_simulator","description":"Explicit separate mathematical benchmark"}}},
            {"id":"data","operation":"simulate_transitions","inputs":{"nominal_model":{"stage":"nominal"},"reference_model":{"stage":"reference"}},"config":{"trajectories":12,"steps":16,"seed":54}},
            {"id":"corrected","operation":"train_residual","inputs":{"nominal_model":{"stage":"nominal"},"dataset":{"stage":"data"}},"config":{"epochs":100,"hidden_dims":[12,12],"learning_rate":.008,"batch_size":32,"patience":30,"seed":55}},
            {"id":"controller","operation":"design_controller","inputs":{"model":{"stage":"corrected"},"evaluation_model":{"stage":"reference"}},"config":{"parameters":{"horizon":3,"q":1,"r":.01,"s":.005},"budget":8,"seed":56,"tuning_scenarios":scenarios('tune',1,6),"verification_scenarios":scenarios('verify',1,7),"test_scenarios":scenarios('test',1,8)}},
            {"id":"expert","operation":"collect_expert","inputs":{"controller":{"stage":"controller"},"evaluation_model":{"stage":"reference"}},"config":{"scenarios":scenarios('expert',6,6),"seed":57}},
            {"id":"imitation","operation":"train_imitation","inputs":{"dataset":{"stage":"expert"}},"config":{"architecture":"mlp","epochs":50,"hidden_dims":[16,12],"batch_size":16,"learning_rate":.005,"seed":58}},
            {"id":"optimized","operation":"optimize_policy","inputs":{"policy":{"stage":"imitation"},"model":{"stage":"corrected"}},"config":{"epochs":2,"horizon":4,"learning_rate":.0002,"seed":59}},
        ]}
        plan = plan_workflow(self.workspace, {}, workflow)
        self.assertEqual(plan["status"], "ready", plan["errors"])
        result = run_operation({"workspace": str(self.workspace.path), "operation": "run_workflow", "config": workflow})
        self.assertEqual(result["status"], "verified")
        self.assertEqual(len(result["artifacts"]), 8)
        controller_root, _ = self.workspace.resolve(result["artifacts"]["controller"])
        descriptor = read_json(controller_root / "controller.json")
        self.assertTrue(descriptor["prediction_execution"]["nominal"]["frozen_python_source"])
        self.assertEqual(descriptor["evaluation_execution"]["source_representation"], "python_torch_transition")
        corrected_id = result["artifacts"]["corrected"]
        root, _ = self.workspace.resolve(corrected_id)
        report = read_json(root / "report.json")
        self.assertLess(report["test"]["corrected_mse"],report["test"]["nominal_mse"])
        corrected = load_model(self.workspace, corrected_id)
        self.assertIsInstance(corrected.nominal, PythonTensorDynamics)
        target = Workspace(Path(self.temp.name) / "imported")
        optimized = result["artifacts"]["optimized"]
        source_root, _ = self.workspace.resolve(optimized)
        target.import_artifact(source_root)
        self.path.unlink()
        np.testing.assert_allclose(load_model(target, corrected_id).step([.2,.1,0],[.1,-.2],[.05]),
                                   corrected.step([.2,.1,0],[.1,-.2],[.05]), atol=1e-12)
        self.assertEqual(len(target.manifest["artifacts"]),8)


if __name__ == "__main__":
    unittest.main()
