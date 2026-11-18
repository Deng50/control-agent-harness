"""Real predictor acquisition and portable assets; no GA or evaluation plant."""
from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.linalg import expm
import torch

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.contracts import ContractError, ModelInterface
from control_harness.modeling import CorrectedDynamics, ResidualNetwork, load_model, register_model
from control_harness.predictors import PredictionResult, export_predictor_assets, resolve_predictor, spring_damper_spec


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "test-output/v2-development/20261117T002006"
OUTPUT = ROOT / "test-output/predictor-tests"
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def parameters():
    return {"mass": 1., "damping": .8, "stiffness": 2., "sample_time": .2,
        "state_bounds": [[-2., 2.], [-2., 2.]], "input_bounds": [[-1., 1.]], "output_bounds": [[-2., 2.]]}


def nonlinear_spec():
    raw = spring_damper_spec(parameters())["interface"]; raw["time_semantics"] = "discrete_state_transition"
    return {"representation": "discrete_polynomial", "interface": raw, "A": [[.9, .1], [-.2, .8]],
        "B": [[.1], [.2]], "E": [[], []], "C": [[1., 0.]], "bias": [0., 0.],
        "terms": [{"state_index": 0, "coefficient": .2, "powers": [2, 0, 0]}]}


class PredictorTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = Workspace(self.root / "case")

    def register(self, spec=None):
        return register_model(self.ws, spec or nonlinear_spec(), {"kind": "user_nominal", "description": "explicit synthetic numerical test_fixture"})

    def test_spring_requires_every_physical_timing_bound_parameter_and_uses_exact_zoh(self):
        for missing in parameters():
            incomplete = parameters(); incomplete.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ContractError): spring_damper_spec(incomplete)
        for bad in ({"mass": True}, {"mass": 0}, {"sample_time": float("nan")}, {"state_bounds": [[1, -1], [-1, 1]]}):
            with self.assertRaises(ContractError): spring_damper_spec(parameters() | bad)
        result = resolve_predictor(self.ws, {}, {"method": "equations", "model_spec": spring_damper_spec(parameters())})
        augmented = np.zeros((3, 3)); augmented[:2, :2] = [[0, 1], [-2, -.8]]; augmented[1, 2] = 1
        oracle = expm(augmented * .2)
        np.testing.assert_allclose(result.model.A.numpy(), oracle[:2, :2], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.model.B.numpy(), oracle[:2, 2:], rtol=0, atol=1e-14)
        self.assertEqual(result.model.transition_adapter["method"], "zoh")
        self.assertEqual(result.interface.time_semantics, "discrete_state_transition")
        self.assertIsNone(result.artifact_id)
        self.assertFalse(self.ws.manifest["artifacts"])

    def test_direct_registered_nonlinear_model_remains_nmpc_without_forced_linear_fit(self):
        model_id = self.register()
        before = copy.deepcopy(self.ws.manifest)
        result = resolve_predictor(self.ws, {"model": model_id}, {"method": "direct"})
        self.assertTrue(result.instantiated); self.assertEqual(result.route, "direct")
        self.assertEqual(result.artifact_id, model_id); self.assertTrue(result.model.terms)
        self.assertTrue(result.report["nonlinear_MPC_supported_without_AB"])
        self.assertEqual(self.ws.manifest, before)
        x = torch.tensor([.3, -.2], dtype=torch.float64)
        u = torch.tensor([.4], dtype=torch.float64, requires_grad=True)
        value = result.model(x, u).sum(); gradient, = torch.autograd.grad(value, u)
        torch.testing.assert_close(gradient, torch.tensor([.3], dtype=torch.float64))

    def test_operating_point_linearization_keeps_affine_bias_dt_local_range_and_real_errors(self):
        model_id = self.register()
        config = {"method": "linearize", "operating_point": {"x": [.2, -.1], "u": [.1], "d": []},
            "validity_bounds": {"x": [[.1, .3], [-.2, 0.]], "u": [[-.2, .4]], "d": []}}
        result = resolve_predictor(self.ws, {"model": model_id}, config)
        np.testing.assert_allclose(result.model.A.numpy(), [[.98, .1], [-.2, .8]], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.model.bias.numpy(), [-.008, 0.], rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.report["max_state_abs_error"], [.002, 0.], rtol=0, atol=1e-14)
        self.assertEqual(result.interface.sample_time, .2)
        self.assertEqual([(v.lower, v.upper) for v in result.interface.states], [(.1, .3), (-.2, 0.)])
        self.assertEqual(self.ws.resolve(result.artifact_id)[1]["activation"], "none")
        self.assertNotIn("identified_model", self.ws.manifest["active"])
        self.assertEqual(self.ws.manifest["active"]["nominal_model"], model_id)
        self.assertIn("not_global", result.report["scope"])
        for bad in ({"validity_bounds": {"x": [[-3, .3], [-.2, 0]], "u": [[-.2, .4]], "d": []}},
                    {"operating_point": {"x": [1., -.1], "u": [.1], "d": []}}):
            with self.assertRaises(ContractError): resolve_predictor(self.ws, {"model": model_id}, config | bad)

    def test_frozen_s04_python_matlab_and_slx_affine_identification_match_independent_f01_oracle(self):
        evidence = read_json(RUN / "S06/predictive-training-rank-audit.json")
        expected_backends = {"python", "matlab_function", "simulink"}
        seen = set()
        checks = []
        augmented = np.zeros((4, 4)); augmented[:2, :2] = [[0, 1], [-2, -.8]]; augmented[1, 2:] = [1, 1]
        exact = expm(augmented * .2)
        for index, record in enumerate(evidence["checks"]):
            backend = record["backend"]
            if backend not in expected_backends: continue
            with self.subTest(backend=backend):
                source = Workspace(record["source_workspace"])
                source_root, source_meta = source.resolve(record["dataset_id"], kind="sampled_dataset")
                workspace = Workspace(self.root / ("c02-" + str(index)))
                imported = workspace.import_artifact(source_root)
                result = resolve_predictor(workspace, {"dataset": imported}, {"method": "identify_affine", "C": [[1., 0.]],
                    "acceptance": {"atol": 1e-6, "rtol": 1e-5}})
                np.testing.assert_allclose(result.model.A.numpy(), exact[:2, :2], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(result.model.B.numpy(), exact[:2, 2:3], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(result.model.E.numpy(), exact[:2, 3:4], rtol=1e-5, atol=1e-6)
                self.assertEqual(result.report["training_rank"], 5)
                self.assertFalse(result.report["test_used_for_fitting"])
                self.assertFalse(result.report["validation_used_for_fitting"])
                self.assertFalse(result.report["global_linearity_proved"])
                self.assertEqual(workspace.resolve(result.artifact_id)[1]["activation"], "none")
                independent_errors = []
                with np.load(source_root / "transitions.npz", allow_pickle=False) as data:
                    split = read_json(source_root / "split.json")
                    for group in split["test"]:
                        rows = np.flatnonzero(data["trajectory_id"] == group)
                        rows = rows[np.argsort(data["time"][rows], kind="stable")]
                        self.assertEqual(len(rows), 50)
                        actual = data["x"][rows[0]].copy(); reference = actual.copy()
                        for row in rows:
                            actual = result.model.step(actual, data["u"][row], data["d"][row])
                            reference = exact[:2, :2] @ reference + exact[:2, 2:3] @ data["u"][row] + exact[:2, 3:4] @ data["d"][row]
                            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
                            independent_errors.append(np.abs(actual - reference))
                checks.append({"backend": backend, "source_workspace": record["source_workspace"], "dataset_id": record["dataset_id"],
                    "source_files": source_meta["files"], "source_provenance": source_meta.get("source", {}),
                    "training_rank": result.report["training_rank"], "parameter_count": result.report["parameter_count"],
                    "coefficients": {key: getattr(result.model, key).numpy().tolist() for key in ("A", "B", "E", "bias", "C")},
                    "independent_exact_zoh_source": {"dt": .2, "A_continuous": [[0., 1.], [-2., -.8]], "B_continuous": [[0.], [1.]], "E_continuous": [[0.], [1.]]},
                    "independent_50_step_state_max_abs_error": np.max(independent_errors, axis=0).tolist(),
                    "heldout_one_step": result.report["one_step"], "heldout_rollout": result.report["rollout"],
                    "validation_used_for_fitting": result.report["validation_used_for_fitting"], "test_used_for_fitting": result.report["test_used_for_fitting"]})
                seen.add(backend)
        self.assertEqual(seen, expected_backends)
        atomic_json(RUN / "S07/c02-frozen-source-identification.json", {"schema_version": 1, "status": "PASS", "confirmation_source": "test_fixture",
            "execution_scope": "Actual train-only least-squares on existing frozen S04 numerical outputs; no new MATLAB/Simulink/native service execution",
            "acceptance": {"atol": 1e-6, "rtol": 1e-5, "heldout_steps_per_trajectory": 50}, "checks": checks})

    def synthetic_sampled(self, nonlinear=False, rank_deficient=False):
        spec = nonlinear_spec(); spec["terms"] = []
        i = ModelInterface.parse(spec["interface"])
        rng = np.random.default_rng(20261122)
        rows = {key: [] for key in ("x", "u", "d", "next_x", "y", "next_y", "trajectory_id", "time")}
        for group in range(6):
            x = np.array([-.3 + group * .1, .15 - group * .02])
            for step in range(12):
                u = np.zeros(1) if rank_deficient else rng.uniform(-.5, .5, 1)
                nx = np.array([[.7, .1], [-.1, .8]]) @ x + np.array([.1, .2]) * u[0]
                if nonlinear: nx[0] += .8 * x[0] ** 2
                for key, value in (("x", x.copy()), ("u", u), ("d", []), ("next_x", nx.copy()), ("y", [x[0]]),
                                   ("next_y", [nx[0]]), ("trajectory_id", group), ("time", step * .2)):
                    rows[key].append(value)
                x = nx
        with self.ws.stage("sampled_dataset", i, [], {"confirmation_source": "test_fixture"}) as stage:
            np.savez_compressed(stage.root / "transitions.npz", **{key: np.asarray(value) for key, value in rows.items()})
            atomic_json(stage.root / "split.json", {"train": [0, 1, 2, 3], "validation": [4], "test": [5]})
            return stage.publish({"status": "verified"}, source={"kind": "synthetic_test_fixture", "measured": False}, activate=False)

    def test_rank_deficient_or_nonlinear_data_cannot_be_claimed_exact_affine_predictor(self):
        for flags in ({"rank_deficient": True}, {"nonlinear": True}):
            source = self.synthetic_sampled(**flags)
            with self.subTest(flags=flags), self.assertRaises(ContractError):
                resolve_predictor(self.ws, {"dataset": source}, {"method": "identify_affine", "C": [[1., 0.]],
                    "acceptance": {"atol": 1e-6, "rtol": 1e-5}})
        self.assertNotIn("identified_model", self.ws.manifest["active"])

    def corrected(self):
        nominal = nonlinear_spec(); nominal["terms"] = []
        nominal_id = self.register(nominal)
        model = load_model(self.ws, nominal_id)
        config = {"hidden_dims": [4, 3], "residual_semantics": "next_state_residual_in_state_units"}
        scales = {"input_mean": [0., 0., 0.], "input_scale": [1., 1., 1.], "residual_scale": [.05, .03]}
        torch.manual_seed(20261122)
        residual = ResidualNetwork(model.interface, config, scales)
        with self.ws.stage("corrected_model", model.interface, [nominal_id], {"confirmation_source": "test_fixture"}) as stage:
            atomic_json(stage.root / "nominal.json", nominal); atomic_json(stage.root / "network.json", config); atomic_json(stage.root / "scalers.json", scales)
            torch.save({"network_state": residual.state_dict()}, stage.root / "model.pt")
            corrected_id = stage.publish({"status": "verified"}, source={"nominal_model": nominal_id, "kind": "manual_residual_tensor_fixture"}, activate=False)
        return resolve_predictor(self.ws, {"model": corrected_id}, {"method": "direct"})

    def assert_portable(self, result, name):
        from control_harness.control import MPCPolicy
        destination = self.root / (name + "-assets")
        export_predictor_assets(result, destination)
        x, u = [.2, -.1], [.15]
        d = [0.] * result.interface.nd
        expected = result.model.step(x, u, d)
        ux = torch.tensor(u, dtype=torch.float64, requires_grad=True)
        gradient, = torch.autograd.grad(result.model(torch.tensor(x, dtype=torch.float64), ux, torch.tensor(d, dtype=torch.float64)).sum(), ux)
        policy = MPCPolicy(result.model, {"horizon": 2, "q": 1., "r": .1, "s": .01, "max_iterations": 80})
        action, _ = policy.act(x, [0.], d)
        with tempfile.TemporaryDirectory(prefix="control-predictor-portable-") as external:
            external = Path(external); target = external / "prediction"
            shutil.copytree(destination, target)
            # Same production MPC numerical implementation, with only its context
            # dependency replaced by a fixed no-op test context.
            script = '''import hashlib,importlib.abc,importlib.util,json,math,runpy,sys,time,copy
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
class DenyProject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'):raise RuntimeError('repository import forbidden')
sys.meta_path.insert(0,DenyProject());sys.dont_write_bytecode=True
class OperationContext:
 def check(self):pass
class SolverFailure(RuntimeError):pass
ContractError=ValueError
def tensor(value):return torch.as_tensor(value,dtype=torch.float64)
''' + inspect.getsource(MPCPolicy) + '''
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('_portable_predictor',root/'predictor.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.load_predictor(root)
x=torch.tensor([.2,-.1],dtype=torch.float64);u=torch.tensor([.15],dtype=torch.float64,requires_grad=True);d=torch.zeros(model.interface.nd,dtype=torch.float64)
y=model(x,u,d);gradient,=torch.autograd.grad(y.sum(),u)
policy=MPCPolicy(model,{'horizon':2,'q':1.,'r':.1,'s':.01,'max_iterations':80});action,_=policy.act(x.numpy(),[0.],d.numpy())
print(json.dumps({'step':y.detach().numpy().tolist(),'gradient':gradient.detach().numpy().tolist(),'action':action.tolist()}))
'''
            environment = dict(os.environ); environment.pop("PYTHONPATH", None)
            completed = subprocess.run([sys.executable, "-I", "-c", script, str(target)], cwd=external,
                env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            actual = json.loads(completed.stdout)
            np.testing.assert_allclose(actual["step"], expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual["gradient"], gradient.numpy(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(actual["action"], action, rtol=1e-10, atol=1e-10)
            self.assertFalse(target.is_relative_to(ROOT))

    def test_polynomial_and_continuous_zoh_assets_run_new_process_step_gradient_and_real_mpc(self):
        direct = resolve_predictor(self.ws, {"model": self.register()}, {"method": "direct"})
        self.assert_portable(direct, "nonlinear")
        exact = resolve_predictor(self.ws, {}, {"method": "equations", "model_spec": spring_damper_spec(parameters())})
        self.assert_portable(exact, "continuous-zoh")

    def test_corrected_declarative_predictor_exports_actual_residual_semantics_and_new_process_mpc(self):
        result = self.corrected()
        self.assertIsInstance(result.model, CorrectedDynamics)
        self.assert_portable(result, "corrected")

    def test_learned_predictor_assets_are_self_contained_and_preserve_gradient_and_mpc(self):
        from test_learned_dynamics import physical, supervised_interface, scaler, settings
        from control_harness.learned_dynamics import export_learned_model
        from control_harness.supervised_export import build_supervised_network, export_supervised
        plant = physical()
        torch.manual_seed(20261122)
        network = build_supervised_network(settings(plant, [4]))
        supervised = self.root / "supervised"
        export_supervised(supervised / "bundle", network, settings(plant, [4]), supervised_interface(plant), scaler(plant), {"kind": "synthetic_test_fixture"})
        interface = ModelInterface.parse(plant)
        with self.ws.stage("learned_model", interface, [], {"kind": "manually_assigned_tensor_fixture"}) as stage:
            export_learned_model(stage.root, supervised, interface, [[1., 0.]], provenance={"input_path": str(self.root / "private-source.m"), "model_id": "fixture-nn"})
            artifact = stage.publish({"status": "verified"}, source={"kind": "manual_tensor_fixture"}, activate=False)
        result = resolve_predictor(self.ws, {"model": artifact}, {"method": "direct"})
        shutil.rmtree(supervised)
        (result.source_root / "bundle/unregistered-private-data.csv").write_text("private raw data must not be copied", encoding="utf-8")
        self.assert_portable(result, "learned")
        learned = read_json(self.root / "learned-assets/learned.json")
        self.assertEqual(learned["provenance"], {"model_id": "fixture-nn"})
        self.assertFalse((self.root / "learned-assets/unregistered-private-data.csv").exists())

    def test_registered_python_torch_direct_runtime_is_available_with_explicit_portability_limit(self):
        from control_harness.python_model import register_python_model
        source = self.root / "torch_source.py"
        source.write_text("import torch\ndef transition(x,u,d,parameters):\n return torch.stack((.9*x[...,0]+.2*u[...,0],.8*x[...,1]),dim=-1)\n", encoding="utf-8")
        spec = {"representation": "python_torch_transition", "interface": nonlinear_spec()["interface"],
            "entrypoint": "transition", "parameters": {}, "C": [[1., 0.]]}
        artifact = register_python_model(self.ws, source, {"kind": "user_nominal", "description": "synthetic registered Torch fixture"}, spec)
        result = resolve_predictor(self.ws, {"model": artifact}, {"method": "direct"})
        np.testing.assert_allclose(result.model.step([.2, -.1], [.3]), [.24, -.08], rtol=0, atol=1e-14)
        with self.assertRaisesRegex(ContractError, "portable adapter"):
            export_predictor_assets(result, self.root / "torch-unsupported")

    def test_export_omits_private_storage_metadata_but_keeps_hashes_and_numerical_report(self):
        result = resolve_predictor(self.ws, {"model": self.register()}, {"method": "direct"})
        result.report["source"].update(input_path=str(self.root / "private.py"), workspace=str(self.ws.path), sha256="f" * 64,
            nested={"source_root": str(self.ws.path), "value": .123, "paths": [str(self.root), "safe-id"]})
        result.spec["workspace"] = str(self.ws.path)
        destination = self.root / "private-free"
        export_predictor_assets(result, destination)
        report = read_json(destination / "prediction_report.json")
        self.assertNotIn("input_path", report["source"]); self.assertNotIn("workspace", report["source"])
        self.assertEqual(report["source"]["sha256"], "f" * 64)
        self.assertEqual(report["source"]["nested"], {"value": .123, "paths": ["safe-id"]})
        self.assertNotIn("workspace", read_json(destination / "model.json"))

    def test_unsupported_inputs_and_portable_source_never_silently_become_linear(self):
        with self.assertRaises(ContractError): resolve_predictor(self.ws, {"description": "unknown spring"}, {"method": "direct"})
        with self.assertRaises(ContractError): resolve_predictor(self.ws, {}, {"route": "equations"})
        invalid = PredictionResult(None, None, {"representation": "python_torch_transition"}, None, "direct", {}, True)
        with self.assertRaisesRegex(ContractError, "portable adapter"):
            export_predictor_assets(invalid, self.root / "unsupported")


if __name__ == "__main__": unittest.main()
