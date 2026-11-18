"""Actual Tensor/forward/export checks; synthetic test_fixture, no training."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from control_harness.supervised_export import build_supervised_network, export_supervised


OUTPUT = Path(__file__).resolve().parents[1] / "test-output/supervised-export-tests"
OUTPUT.mkdir(parents=True, exist_ok=True)


def configuration(width=3, activation="tanh"):
    return {"schema_version": 1, "architecture": "mlp", "input_dim": width, "output_dim": 2,
            "hidden_dims": [5, 4], "activation": activation, "dtype": "float64"}


def interface(dynamic=False):
    names = ("a", "b", "c", "position", "velocity")
    units = ("A", "V", "N", "m", "m/s")
    signals = [{"name": name, "role": "auxiliary" if i < 3 else "output", "unit": units[i],
                "column": {"index": i, "header": name, "worksheet": None},
                "delay_steps": 2 if dynamic and i == 0 else 0, "observable": i < 3}
               for i, name in enumerate(names)]
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "name": "synthetic test_fixture",
            "task": "dynamic_prediction" if dynamic else "static_regression", "signals": signals,
            "sample_time": .2 if dynamic else None,
            "time_column": {"index": 8, "header": "time", "worksheet": None} if dynamic else None,
            "trip_column": None,
            "features": ([{"signal": "a", "window": 3, "lag": 1}, {"signal": "b", "window": 1, "lag": 0}]
                         if dynamic else [{"signal": name, "window": 1, "lag": 0} for name in names[:3]]),
            "targets": [{"signal": name, "offset": 2 if dynamic else 0} for name in names[3:]]}


def scaler(width=3):
    return {"method": "standard", "fit_split": "train", "fit_samples": 12,
            "X": {"location": [10., -2., 3., 4.][:width], "scale": [2., .5, 4., 2.][:width]},
            "y": {"location": [100., -10.], "scale": [3., .25]}}


def load(directory):
    spec = importlib.util.spec_from_file_location("_export_test_inference", Path(directory) / "inference.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.load_bundle(directory)


class SupervisedExportTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(129)
        self.network = build_supervised_network(configuration())
        with torch.no_grad():
            for index, parameter in enumerate(self.network.parameters()):
                parameter.copy_(torch.linspace(-.3, .4, parameter.numel(), dtype=torch.float64).reshape(parameter.shape) + .01 * index)
        self.raw = np.array([[10., -2., 3.], [14., -.5, -1.], [8., -3., 7.]], dtype=np.float64)

    def export(self, name="bundle", *, network=None, network_config=None, data_interface=None, scaling=None, **kwargs):
        return export_supervised(self.root / name, network or self.network, network_config or configuration(),
                                 data_interface or interface(), scaling or scaler(),
                                 {"epochs": 7, "confirmation_source": "test_fixture", "scope": "manually assigned tensors; no training claim"}, **kwargs)

    def expected(self, raw=None, network=None, scaling=None):
        raw, network, scaling = self.raw if raw is None else raw, network or self.network, scaling or scaler()
        normalized = (raw - scaling["X"]["location"]) / scaling["X"]["scale"]
        with torch.no_grad(): output = network(torch.tensor(normalized, dtype=torch.float64)).numpy()
        return output * scaling["y"]["scale"] + scaling["y"]["location"]

    def test_shared_factory_has_exact_state_keys_float64_and_linear_output(self):
        for activation in ("tanh", "relu", "silu"):
            network = build_supervised_network(configuration(activation=activation))
            self.assertEqual(list(network.state_dict()), ["0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"])
            self.assertIsInstance(network[-1], nn.Linear)
            self.assertTrue(all(value.dtype == torch.float64 for value in network.state_dict().values()))
            self.assertEqual(tuple(network(torch.zeros((3, 3), dtype=torch.float64)).shape), (3, 2))

    def test_bad_dimensions_and_giant_network_are_rejected_before_tensor_allocation(self):
        for patch_value in ({"input_dim": True}, {"output_dim": 2.0}, {"hidden_dims": ["5"]},
                            {"hidden_dims": [65536, 65536]}, {"hidden_dims": [1] * 17},
                            {"hidden_dims": [-1]}, {"dtype": "float32"}, {"activation": "unknown"}):
            with self.subTest(config=patch_value), patch("control_harness.supervised_export.nn.Linear", side_effect=AssertionError("must not allocate")):
                with self.assertRaises(ValueError):
                    build_supervised_network(configuration() | patch_value)

    def test_actual_state_dict_forward_and_physical_scaler_roundtrip(self):
        exported = self.export()
        predictor = load(exported["bundle_root"])
        np.testing.assert_allclose(predictor.predict(self.raw), self.expected(), rtol=1e-12, atol=1e-12)
        state = torch.load(Path(exported["bundle_root"]) / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(set(state), set(self.network.state_dict()))
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in state.values()))
        self.assertEqual(exported["output_units"], ["m", "m/s"])
        self.assertNotIn("nx", predictor.interface)
        self.assertFalse(any("control_harness" in (Path(exported["bundle_root"]) / name).read_text(encoding="utf-8")
                             for name in ("inference.py", "predict.py")))

    def test_all_activation_bundles_match_their_actual_tensor_forward(self):
        for activation in ("tanh", "relu", "silu"):
            network = build_supervised_network(configuration(activation=activation))
            exported = self.export(activation, network=network, network_config=configuration(activation=activation))
            np.testing.assert_allclose(load(exported["bundle_root"]).predict(self.raw), self.expected(network=network), rtol=1e-12, atol=1e-12)

    def test_export_preserves_global_torch_rng_and_refuses_overwrite(self):
        before = torch.get_rng_state().clone()
        self.export()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaises(ValueError): self.export()

    def test_dynamic_history_metadata_keeps_lag_delay_units_and_does_not_guess_state(self):
        network = build_supervised_network(configuration(4))
        exported = self.export(network=network, network_config=configuration(4), data_interface=interface(True), scaling=scaler(4))
        predictor = load(exported["bundle_root"])
        self.assertEqual(predictor.features["feature_labels"], ["a@lag=5", "a@lag=4", "a@lag=3", "b@lag=0"])
        self.assertEqual(predictor.features["target_labels"], ["position@offset=2", "velocity@offset=2"])
        self.assertEqual(predictor.features["input_units"], ["A", "A", "A", "V"])
        self.assertEqual([row["availability_delay_steps"] for row in predictor.features["feature_order"]], [2, 2, 2, 0])
        self.assertNotIn("nx", predictor.interface)
        self.assertIn("not generated", (Path(exported["bundle_root"]) / "README.md").read_text(encoding="utf-8"))

    def test_raw_input_labels_width_nonfinite_bool_and_string_checks(self):
        exported = self.export(); predictor = load(exported["bundle_root"])
        labels = predictor.features["feature_labels"]
        np.testing.assert_allclose(predictor.predict(self.raw, labels), self.expected(), rtol=1e-12, atol=1e-12)
        for values in ([[1., 2.]], [[1., 2., float("nan")]], [[True, 2., 3.]], [["1", 2., 3.]], [1., 2., 3.], []):
            with self.subTest(values=values), self.assertRaises(ValueError): predictor.predict(values)
        with self.assertRaisesRegex(ValueError, "labels/order"):
            predictor.predict(self.raw, labels[::-1])

    def test_output_batch_budget_is_checked_before_tensor_forward(self):
        settings = configuration() | {"hidden_dims": [65536]}
        network = build_supervised_network(settings)
        exported = self.export(network=network, network_config=settings)
        predictor = load(exported["bundle_root"])
        # A genuine supported wide network is small enough to export; a tiny
        # feature matrix can still request an over-budget hidden activation.
        with patch.object(predictor.network, "forward", side_effect=AssertionError("must not allocate forward output")):
            with self.assertRaisesRegex(ValueError, "layer batch budget"):
                predictor.predict(np.zeros((100, 3), dtype=np.float64))

    def test_invalid_scaler_weights_interface_and_order_never_publish_a_bundle(self):
        for name, change in (("zero", lambda value: value["X"]["scale"].__setitem__(0, 0)),
                             ("negative", lambda value: value["y"]["scale"].__setitem__(0, -1)),
                             ("nan", lambda value: value["X"]["location"].__setitem__(0, float("nan"))),
                             ("width", lambda value: value["X"]["scale"].pop()),
                             ("holdout", lambda value: value.update(fit_split="test"))):
            scaling = scaler(); change(scaling)
            with self.subTest(name=name), self.assertRaises(ValueError): self.export(name, scaling=scaling)
            self.assertFalse((self.root / name / "bundle_manifest.json").exists())
        bad = copy.deepcopy(self.network)
        with torch.no_grad(): next(bad.parameters()).flatten()[0] = float("inf")
        with self.assertRaises(ValueError): self.export("bad-weights", network=bad)
        with self.assertRaises(ValueError): self.export("bad-order", feature_order=[{"signal": "guessed_state"}])
        with self.assertRaises(ValueError): self.export("bad-width", network_config=configuration(4))

    def test_file_hash_tampering_is_rejected(self):
        exported = self.export(); root = Path(exported["bundle_root"])
        (root / "scaler.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load(root)

    def copied_cli(self, *, npz=False, malicious_header=False):
        exported = self.export()
        # The copied bundle and fresh interpreter cwd are outside the repository.
        with tempfile.TemporaryDirectory(prefix="control-supervised-portable-") as external:
            external = Path(external)
            bundle = external / "bundle"
            shutil.copytree(exported["bundle_root"], bundle)
            source = external / ("input.npz" if npz else "input.json")
            labels = exported["feature_labels"]
            if malicious_header:
                header = io.BytesIO()
                np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (2**40, 3)})
                with zipfile.ZipFile(source, "w") as archive: archive.writestr("features.npy", header.getvalue())
            elif npz:
                np.savez_compressed(source, features=self.raw, feature_labels=np.asarray(labels))
            else:
                source.write_text(json.dumps({"features": self.raw.tolist(), "feature_labels": labels}), encoding="utf-8")
            target = external / "prediction.json"
            guard = r'''import importlib.abc,runpy,sys
class DenyProject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'): raise RuntimeError('repository import forbidden')
sys.meta_path.insert(0,DenyProject())
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            env = dict(os.environ); env.pop("PYTHONPATH", None); env["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run([sys.executable, "-I", "-c", guard, str(bundle / "predict.py"),
                                     "--input", str(source), "--output", str(target)], cwd=external,
                                    env=env, capture_output=True, text=True, timeout=60)
            if malicious_header:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("NPY shape/dtype", result.stderr)
                self.assertFalse(target.exists())
                return
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(target.read_text(encoding="utf-8"))
            np.testing.assert_allclose(output["predictions"], self.expected(), rtol=1e-12, atol=1e-12)
            self.assertEqual(output["output_units"], ["m", "m/s"])
            self.assertFalse((bundle / "__pycache__").exists())
            self.assertFalse(str(external.resolve()).startswith(str(OUTPUT.parents[1].resolve())))

    def test_json_cli_new_process_outside_repository_with_pythonpath_removed(self):
        self.copied_cli()

    def test_npz_cli_new_process_outside_repository_with_pythonpath_removed(self):
        self.copied_cli(npz=True)

    def test_npz_forged_shape_is_rejected_before_allocation(self):
        self.copied_cli(npz=True, malicious_header=True)


if __name__ == "__main__":
    unittest.main()
