"""Actual frozen Tensor gradients and portable physical wrappers, no training claim."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
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

from control_harness.contracts import ContractError, ModelInterface
from control_harness.learned_dynamics import export_learned_model, load_learned_model
from control_harness.supervised_export import build_supervised_network, export_supervised


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-output/learned-dynamics-tests"
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def physical(nx=2, nu=1, nd=1):
    def variables(prefix, count):
        return [{"name": prefix + str(index + 1), "unit": prefix + "_unit_" + str(index + 1), "lower": -100., "upper": 100.}
                for index in range(count)]
    return {"schema_version": 1, "name": "synthetic explicit test_fixture", "sample_time": .2,
        "time_semantics": "discrete_state_transition", "states": variables("x", nx), "controls": variables("u", nu),
        "disturbances": variables("d", nd), "outputs": variables("y", 1)}


def supervised_interface(plant):
    signals = []
    for group, role in (("states", "state"), ("controls", "control"), ("disturbances", "disturbance")):
        for item in plant[group]:
            signals.append({"name": group + "::" + item["name"], "role": role, "unit": item["unit"], "observable": True,
                "delay_steps": 0, "column": {"index": len(signals), "header": "logical NPZ signal", "worksheet": None}})
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "name": "synthetic full-state transition fixture",
        "task": "dynamic_prediction", "sample_time": plant["sample_time"], "signals": signals,
        "features": [{"signal": item["name"], "window": 1, "lag": 0} for item in signals],
        "targets": [{"signal": "states::" + item["name"], "offset": 1} for item in plant["states"]],
        "state_mapping": {item["name"]: "states::" + item["name"] for item in plant["states"]},
        "time_column": {"index": len(signals), "header": "time", "worksheet": None}, "trip_column": None}


def settings(plant, hidden=None):
    return {"schema_version": 1, "architecture": "mlp", "input_dim": sum(len(plant[group]) for group in ("states", "controls", "disturbances")),
        "output_dim": len(plant["states"]), "hidden_dims": [] if hidden is None else hidden, "activation": "tanh", "dtype": "float64"}


def scaler(plant):
    width = sum(len(plant[group]) for group in ("states", "controls", "disturbances"))
    nx = len(plant["states"])
    return {"method": "standard", "fit_split": "train", "fit_samples": 20,
        "X": {"location": np.linspace(-.4, .5, width).tolist(), "scale": np.linspace(.6, 1.7, width).tolist()},
        "y": {"location": np.linspace(-.2, .3, nx).tolist(), "scale": np.linspace(.7, 2., nx).tolist()}}


def expected(network, scale, x, u, d):
    features = torch.cat((x, u, d), dim=-1)
    normalized = (features - torch.tensor(scale["X"]["location"], dtype=torch.float64)) / torch.tensor(scale["X"]["scale"], dtype=torch.float64)
    return network(normalized) * torch.tensor(scale["y"]["scale"], dtype=torch.float64) + torch.tensor(scale["y"]["location"], dtype=torch.float64)


class LearnedDynamicsTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plant = physical()
        self.source, self.network, self.scale = self.make_supervised("supervised", self.plant)
        self.C = [[1.2, -.4]]

    def make_supervised(self, name, plant, hidden=None, data=None):
        config = settings(plant, hidden)
        torch.manual_seed(20261121)
        network = build_supervised_network(config)
        with torch.no_grad():
            for index, parameter in enumerate(network.parameters()):
                parameter.copy_(torch.linspace(-.17, .23, parameter.numel(), dtype=torch.float64).reshape(parameter.shape) + index * .02)
        transform = scaler(plant)
        directory = self.root / name
        export_supervised(directory / "bundle", network, config, data or supervised_interface(plant), transform,
            {"confirmation_source": "test_fixture", "scope": "manually assigned weights; no training claim"})
        return directory, network, transform

    def export(self, name="learned", *, source=None, plant=None, C=None):
        directory = self.root / name
        export_learned_model(directory, source or self.source, plant or self.plant, self.C if C is None else C,
            provenance={"supervised_model": "synthetic-fixture-model", "sampled_dataset": "synthetic-fixture-dataset"})
        return directory, load_learned_model(directory)

    def test_actual_tensor_forward_inverse_scale_batch_and_no_extra_dt(self):
        directory, model = self.export()
        x = torch.tensor([[.2, -.1], [-.3, .7]], dtype=torch.float64)
        u = torch.tensor([[.4], [-.2]], dtype=torch.float64)
        d = torch.tensor([[.05], [-.03]], dtype=torch.float64)
        truth = expected(self.network, self.scale, x, u, d)
        torch.testing.assert_close(model(x, u, d), truth, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(model(x[0], u[0], d[0]), truth[0], rtol=1e-12, atol=1e-12)
        self.assertFalse(torch.allclose(model(x, u, d), truth * self.plant["sample_time"]))
        np.testing.assert_allclose(model.step(x.numpy(), u.numpy(), d.numpy()), truth.detach().numpy(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(model.output(x), x @ torch.tensor(self.C, dtype=torch.float64).T)
        self.assertIsInstance(model.interface, ModelInterface)
        self.assertIsNone(model.transition_adapter)
        self.assertFalse(hasattr(model, "A"))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(model.spec["representation"], "learned_discrete_mlp")
        with self.assertRaisesRegex(ContractError, "not a derivative"):
            model.derivative(x, u, d)

    def test_nontrivial_x_u_d_autograd_matches_central_finite_differences(self):
        source, network, scale = self.make_supervised("nonlinear", self.plant, hidden=[5, 4])
        _, model = self.export(source=source)
        x = torch.tensor([.27, -.18], dtype=torch.float64, requires_grad=True)
        u = torch.tensor([.31], dtype=torch.float64, requires_grad=True)
        d = torch.tensor([-.09], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([.7, -1.2], dtype=torch.float64)
        value = (model(x, u, d) * weights).sum()
        grads = torch.autograd.grad(value, (x, u, d))
        originals = [item.detach().clone() for item in (x, u, d)]
        for group, gradient in enumerate(grads):
            difference = []
            for index in range(len(originals[group])):
                plus, minus = [item.clone() for item in originals], [item.clone() for item in originals]
                plus[group][index] += 1e-6; minus[group][index] -= 1e-6
                difference.append(float(((model(*plus) - model(*minus)) * weights).sum() / 2e-6))
            self.assertTrue(torch.all(gradient.abs() > 1e-9))
            np.testing.assert_allclose(gradient.numpy(), difference, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(model(x, u, d), expected(network, scale, x, u, d), rtol=1e-12, atol=1e-12)
        output_grad = torch.autograd.functional.jacobian(model.output, x)
        torch.testing.assert_close(output_grad, torch.tensor(self.C, dtype=torch.float64))

    def test_four_states_two_controls_no_disturbance_and_explicit_empty_batches(self):
        plant = physical(4, 2, 0)
        source, network, scale = self.make_supervised("four-state", plant, hidden=[6])
        _, model = self.export(source=source, plant=plant, C=[[1., 0., 0., 0.]])
        x, u = torch.arange(12, dtype=torch.float64).reshape(3, 4) / 20, torch.ones((3, 2), dtype=torch.float64) * .2
        empty = torch.empty((3, 0), dtype=torch.float64)
        torch.testing.assert_close(model(x, u), model(x, u, empty))
        torch.testing.assert_close(model(x, u), expected(network, scale, x, u, empty), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(model.step(x.numpy(), u.numpy()), model(x, u).numpy())

    def test_runtime_rejects_missing_disturbance_broadcast_nonfinite_and_wrong_dtype(self):
        _, model = self.export()
        x, u, d = torch.zeros(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
        bad = [(x, u, None), (x.float(), u, d), (x, u.repeat(2), d), (x.repeat(2, 1), u, d),
               (x.unsqueeze(0).unsqueeze(0), u, d), (x * float("nan"), u, d)]
        for values in bad:
            with self.subTest(shapes=[None if value is None else list(value.shape) for value in values]), self.assertRaises(ValueError):
                model(*values)
        with self.assertRaises(ValueError): model.step([False, 0], [0], [0])

    def test_supported_wide_network_rejects_oversized_batch_before_forward(self):
        source, *_ = self.make_supervised("wide", self.plant, hidden=[4096])
        _, model = self.export(source=source)
        with patch.object(model.network, "forward", side_effect=AssertionError("must not allocate activations")):
            with self.assertRaisesRegex(ValueError, "activation-scalar batch budget"):
                model(torch.zeros((1300, 2), dtype=torch.float64), torch.zeros((1300, 1), dtype=torch.float64), torch.zeros((1300, 1), dtype=torch.float64))

    def test_physical_output_batch_and_c_matrix_have_separate_numeric_budgets(self):
        from control_harness.learned_dynamics import _numeric
        with self.assertRaisesRegex(ValueError, "numeric budget"):
            _numeric([], (65536, 65536), "C")
        plant = copy.deepcopy(self.plant)
        plant["outputs"] = [{"name": "output" + str(index), "unit": "declared_output_unit", "lower": -100., "upper": 100.} for index in range(4096)]
        _, model = self.export(plant=plant, C=np.tile([[1., 0.]], (4096, 1)))
        with self.assertRaisesRegex(ValueError, "output.*batch budget"):
            model.output(torch.zeros((1300, 2), dtype=torch.float64))

    def test_order_units_timing_history_hidden_and_target_mismatches_are_rejected(self):
        cases = {
            "state order": lambda plant: plant["states"].reverse(),
            "state units": lambda plant: plant["states"][0].update(unit="wrong_unit"),
            "dt": lambda plant: plant.update(sample_time=.3),
            "continuous": lambda plant: plant.update(time_semantics="continuous_state_derivative"),
            "dimensions": lambda plant: plant["states"].pop(),
            "bool dt": lambda plant: plant.update(sample_time=True),
        }
        for index, (name, mutate) in enumerate(cases.items()):
            plant = copy.deepcopy(self.plant); mutate(plant)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.export("bad-physical-" + str(index), plant=plant)
        for index, mutation in enumerate((lambda data: data["signals"][0].update(delay_steps=1),
                                         lambda data: data["features"][0].update(lag=1),
                                         lambda data: data["state_mapping"].pop("x1"),
                                         lambda data: data["targets"][0].update(offset=2))):
            data = supervised_interface(self.plant); mutation(data)
            source, *_ = self.make_supervised("bad-source-" + str(index), self.plant, data=data)
            with self.assertRaisesRegex(ValueError, "Signal|Only explicit|timing|Saved supervised"):
                self.export("bad-bound-" + str(index), source=source)

    def test_descriptor_file_hash_scaler_and_weight_shape_tampering_are_rejected(self):
        directory, _ = self.export()
        descriptor_path = directory / "learned.json"
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        weights_path = directory / "model.pt"
        weights = torch.load(weights_path, weights_only=True)
        weights["0.weight"] = weights["0.weight"][:, :-1]
        torch.save(weights, weights_path)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_learned_model(directory)
        descriptor["files"]["model.pt"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "keys or dimensions"):
            load_learned_model(directory)
        directory, _ = self.export("bad-scaler")
        scale = json.loads((directory / "scaler.json").read_text()); scale["X"]["scale"][0] = 0
        (directory / "scaler.json").write_text(json.dumps(scale))
        descriptor = json.loads((directory / "learned.json").read_text())
        descriptor["files"]["scaler.json"] = hashlib.sha256((directory / "scaler.json").read_bytes()).hexdigest()
        (directory / "learned.json").write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, "positive"):
            load_learned_model(directory)

    def test_rehashed_metadata_cannot_turn_boolean_timing_into_integer_steps(self):
        directory, _ = self.export()
        data = json.loads((directory / "data_interface.json").read_text())
        data["targets"][0]["offset"] = True
        (directory / "data_interface.json").write_text(json.dumps(data))
        descriptor = json.loads((directory / "learned.json").read_text())
        descriptor["files"]["data_interface.json"] = hashlib.sha256((directory / "data_interface.json").read_bytes()).hexdigest()
        (directory / "learned.json").write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, "one-step"):
            load_learned_model(directory)

    def test_kf_is_explicitly_rejected_and_existing_ekf_selects_real_input_jacobian(self):
        from control_harness.estimation import StateEstimator, estimator_config
        _, model = self.export()
        with self.assertRaisesRegex(ContractError, "KF requires"):
            estimator_config(model, {"method": "KF"})
        config = estimator_config(model, {"method": "auto", "initial_estimate": [0., 0.]})
        self.assertEqual(config["method"], "EKF")
        estimator = StateEstimator(model, {"method": "EKF", "initial_estimate": [0., 0.]})
        self.assertEqual(estimator.interface.nx, 2)

    def test_model_loading_and_export_preserve_global_rng_and_do_not_open_source_plant(self):
        before = torch.get_rng_state().clone()
        directory, model = self.export()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        shutil.rmtree(self.source)
        reloaded = load_learned_model(directory)
        torch.testing.assert_close(model(torch.zeros(2, dtype=torch.float64), torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)),
                                   reloaded(torch.zeros(2, dtype=torch.float64), torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)))

    def test_published_model_loads_through_shared_modeling_and_real_mpc(self):
        from control_harness.artifacts import Workspace
        from control_harness.control import MPCPolicy
        from control_harness.modeling import describe_execution, load_model
        workspace = Workspace(self.root / "workspace")
        interface = ModelInterface.parse(self.plant)
        with workspace.stage("learned_model", interface, [], {"confirmation_source": "test_fixture"}) as stage:
            export_learned_model(stage.root, self.source, interface, self.C, provenance={"kind": "manually_assigned_tensor_test_fixture"})
            artifact_id = stage.publish({"status": "verified", "checks": ["synthetic_unit_fixture_not_trained_quality"]})
        model = load_model(workspace, artifact_id)
        model.interface.compatible(interface)
        self.assertEqual(model.interface.fingerprint, interface.fingerprint)
        self.assertEqual(model.interface.to_dict(), interface.to_dict())
        self.assertEqual(describe_execution(model)["source_representation"], "learned_discrete_mlp")
        policy = MPCPolicy(model, {"horizon": 2, "q": 1., "r": .1, "s": .01, "max_iterations": 40})
        actions = torch.tensor([[.1], [.2]], dtype=torch.float64, requires_grad=True)
        cost, constraints = policy.rollout(actions, torch.zeros(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
        gradient, = torch.autograd.grad(cost, actions)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.all(gradient.abs() > 1e-12))
        self.assertTrue(torch.isfinite(constraints).all())
        action, report = policy.act([.2, -.1], [0.], [.02])
        self.assertEqual(action.shape, (1,))
        self.assertTrue(np.isfinite(action).all())
        self.assertLessEqual(report["constraint_violation"], 1e-6)

    def test_standalone_physical_copy_new_isolated_process_with_repository_imports_denied(self):
        directory, model = self.export()
        controls, disturbances = [[.1], [.2], [-.1]], [[.02], [.02], [0.]]
        x0 = [.2, -.1]
        states = [np.asarray(x0)]
        for u, d in zip(controls, disturbances): states.append(model.step(states[-1], u, d))
        expected_states = np.stack(states)
        with tempfile.TemporaryDirectory(prefix="control-physical-portable-") as external:
            external = Path(external)
            bundle = external / "bundle"
            shutil.copytree(directory / "bundle", bundle)
            # No runtime access to training or original source files is available.
            shutil.rmtree(self.source)
            source, target = external / "input.json", external / "output.json"
            source.write_text(json.dumps({"x0": x0, "controls": controls, "disturbances": disturbances}), encoding="utf-8")
            guard = '''import importlib.abc,runpy,sys
class DenyRepository(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'):raise RuntimeError('repository import forbidden')
sys.meta_path.insert(0,DenyRepository());sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')
'''
            environment = dict(os.environ); environment.pop("PYTHONPATH", None); environment["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run([sys.executable, "-I", "-c", guard, str(bundle / "simulate.py"), "--input", str(source), "--output", str(target)],
                cwd=external, env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = json.loads(target.read_text(encoding="utf-8"))
            np.testing.assert_allclose(actual["states"], expected_states, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual["outputs"], expected_states @ np.asarray(self.C).T, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual["time"], np.arange(4) * .2, rtol=0, atol=1e-15)
            self.assertEqual(actual["semantics"], "discrete_next_state_no_extra_dt")
            self.assertEqual(actual["state_units"], ["x_unit_1", "x_unit_2"])
            self.assertFalse((bundle / "__pycache__").exists())
            self.assertFalse(external.resolve().is_relative_to(ROOT))


if __name__ == "__main__": unittest.main()
