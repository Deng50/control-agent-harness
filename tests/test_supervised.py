"""Real small data-only NNs, independent metrics and preallocation guards."""
from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.contracts import ContractError, DataInterface
from control_harness.datasets import prepare_dataset
from control_harness.supervised import train_supervised, _load_bounded_npz


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def data_interface():
    signals = [{"name": name, "role": role, "unit": unit, "observable": True, "delay_steps": 0,
        "column": {"index": index, "header": name, "worksheet": None}}
        for index, (name, role, unit) in enumerate((("a", "auxiliary", "V"), ("b", "auxiliary", "A"),
                                                    ("c", "output", "W"), ("constant", "output", "K")))]
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "task": "static_regression",
        "name": "data only fixture", "signals": signals,
        "features": [{"signal": name, "lag": 0, "window": 1} for name in ("a", "b")],
        "targets": [{"signal": name, "offset": 0} for name in ("c", "constant")]}


class SupervisedTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "test-output" / "supervised-tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workspace = Workspace(self.root / "case")
        self.config = {"hidden_dims": [8], "activation": "tanh", "epochs": 25, "learning_rate": .015,
            "batch_size": 10, "patience": 50, "seed": 20261120, "weight_decay": .001, "min_delta": 0, "grad_clip": 5}

    def dataset(self, mutate=None, scaler="standard", shift_test=0., rows=80, split=None):
        rng = np.random.default_rng(20261120)
        X = rng.uniform(-1., 1., size=(rows, 2))
        y = .7 * X[:, 0] - .4 * X[:, 1] + .1
        path = self.root / "fixture.csv"
        lines = ["a,b,c,constant"] + [f"{a:.17g},{b:.17g},{c:.17g},2" for (a, b), c in zip(X, y)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        interface = DataInterface.parse(data_interface())
        prepared = prepare_dataset(path, interface, split or {"mode": "time", "train": .6, "validation": .2, "test": .2}, scaler=scaler)
        arrays, transform = prepared.arrays, prepared.scaler
        if shift_test:
            arrays["y"][arrays["split_id"] == 2, 0] += shift_test
            arrays["y_scaled"] = (arrays["y"] - transform["y"]["location"]) / transform["y"]["scale"]
        if mutate: mutate(arrays, transform)
        with self.workspace.stage("supervised_dataset", interface, [], {"fixture": "synthetic unit fixture"}) as stage:
            np.savez_compressed(stage.root / "supervised.npz", **arrays)
            atomic_json(stage.root / "scaler.json", transform)
            atomic_json(stage.root / "dataset.json", prepared.metadata)
            artifact = stage.publish({"status": "verified"}, source={"kind": "synthetic", "description": "fixed small unit fixture; no plant model"})
        return artifact, arrays, transform, interface

    def artifact(self, dataset, config=None):
        artifact = train_supervised(self.workspace, dataset, config or self.config)
        root, metadata = self.workspace.resolve(artifact)
        return artifact, root, metadata, read_json(root / "report.json")

    def test_actual_nn_is_data_only_and_reports_independently_recomputed_per_output_metrics(self):
        dataset, arrays, scaler, interface = self.dataset()
        artifact, root, metadata, report = self.artifact(dataset)
        self.assertEqual(metadata["kind"], "supervised_model")
        self.assertEqual(metadata["interface"]["interface_kind"], "tabular_supervised")
        self.assertNotIn("nx", metadata["interface"])
        self.assertEqual([row["id"] for row in metadata["parents"]], [dataset])
        self.assertEqual(report["algorithm"], "supervised_training")
        self.assertEqual(report["parameter_count"], (2 + 1) * 8 + (8 + 1) * 2)
        self.assertEqual(report["resolved_config"]["epochs_semantics"], "target_total")
        self.assertEqual(report["optimizer_updates"], 5 * self.config["epochs"])
        self.assertLess(report["log"][-1]["training_loss"], report["log"][0]["training_loss"])
        self.assertFalse(report["test_used_for_training_or_selection"])
        self.assertEqual(report["test_evaluations"], 1)
        self.assertFalse(any(any("test" in key for key in row) for row in report["log"]))
        self.assertEqual(report["best_epoch"], 1 + int(np.argmin([row["validation_loss"] for row in report["log"]])))
        self.assertEqual(report["normalization"]["zero_std_outputs"], [1])
        self.assertEqual(report["normalization"]["nrmse_scale"][1], 1.)
        with np.load(root / "test_predictions.npz", allow_pickle=False) as final:
            expected_truth = arrays["y"][arrays["split_id"] == 2]
            np.testing.assert_array_equal(final["truth"], expected_truth)
            rmse = np.sqrt(np.mean((final["prediction"] - expected_truth) ** 2, axis=0))
        train_std = arrays["y"][arrays["split_id"] == 0].std(axis=0, ddof=0)
        train_std[train_std == 0] = 1.
        np.testing.assert_allclose(report["test"]["rmse"], rmse, rtol=0, atol=1e-15)
        np.testing.assert_allclose(report["test"]["nrmse"], rmse / train_std, rtol=0, atol=1e-15)
        self.assertEqual([row["signal"] for row in report["test"]["per_output"]], ["c", "constant"])
        checkpoint = torch.load(root / "latest.pt", weights_only=True)
        self.assertEqual(checkpoint["contract"]["algorithm"], "supervised_training")
        self.assertEqual(checkpoint["epoch"], self.config["epochs"])
        self.assertTrue(all(value.dtype == torch.float64 for value in checkpoint["model_state"].values()))
        self.assertEqual(report["acceptance"], "not_requested")
        self.assertTrue((root / "bundle" / "bundle_manifest.json").is_file())
        self.assertEqual(report["export"]["directory"], "bundle")
        self.assertNotIn("bundle_root", report["export"]["details"])

    def test_all_declared_activations_and_layer_widths_change_the_real_network(self):
        dataset, *_ = self.dataset()
        for activation in ("tanh", "relu", "silu"):
            with self.subTest(activation=activation):
                config = {**self.config, "hidden_dims": [5, 3], "activation": activation, "epochs": 3}
                _, root, _, report = self.artifact(dataset, config)
                self.assertEqual(read_json(root / "network_config.json")["activation"], activation)
                self.assertEqual(report["parameter_count"], 3 * 5 + 6 * 3 + 4 * 2)
                self.assertEqual(report["trained_epochs"], 3)
                self.assertEqual(report["resolved_config"]["learning_rate"], config["learning_rate"])

    def test_changing_only_test_targets_cannot_change_training_or_validation_selection(self):
        first, *_ = self.dataset()
        second, *_ = self.dataset(shift_test=20.)
        config = {**self.config, "epochs": 6}
        _, left, _, a = self.artifact(first, config)
        _, right, _, b = self.artifact(second, config)
        self.assertEqual(a["log"], b["log"])
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        wa = torch.load(left / "model.pt", weights_only=True)["network_state"]
        wb = torch.load(right / "model.pt", weights_only=True)["network_state"]
        self.assertTrue(all(torch.equal(wa[key], wb[key]) for key in wa))
        self.assertNotEqual(a["test"]["rmse"], b["test"]["rmse"])

    def test_predeclared_failed_test_gate_is_reviewable_and_does_not_replace_active_model(self):
        dataset, *_ = self.dataset()
        first, *_ = self.artifact(dataset, {**self.config, "epochs": 3})
        rejected = train_supervised(self.workspace, dataset, {**self.config, "epochs": 3, "max_test_nrmse": [0., 0.]})
        root, metadata = self.workspace.resolve(rejected, allow_rejected=True)
        report = read_json(root / "report.json")
        self.assertEqual(metadata["validation"]["status"], "rejected")
        self.assertEqual(report["acceptance"], "failed")
        self.assertEqual(metadata["activation"], "none")
        self.assertEqual(self.workspace.manifest["active"]["supervised_model"], first)
        self.assertTrue((root / "test_predictions.npz").is_file())

    def test_strict_config_rejects_widths_types_unknown_fields_and_preallocation_excess(self):
        dataset, *_ = self.dataset()
        bad = [{"input_dim": 3}, {"output_dim": 1}, {"input_dim": True}, {"hidden_dims": [True]}, {"hidden_dims": [4096, 4096]},
               {"activation": "sigmoid"}, {"activation": []}, {"epochs": 0}, {"epochs": 1.5}, {"learning_rate": "0.01"}, {"learning_rate": float("nan")},
               {"learning_rate": 0}, {"batch_size": 0}, {"patience": False}, {"seed": -1}, {"weight_decay": -1},
               {"min_delta": -1}, {"grad_clip": 0}, {"max_test_nrmse": [1.]}, {"max_test_nrmse": True}, {"nominal_model": "not-required"},
               {"resume_checkpoint": True}, {"resume_checkpoint_sha256": "bad-sha"}]
        for patch in bad:
            with self.subTest(patch=patch), self.assertRaises(ContractError):
                train_supervised(self.workspace, dataset, {**self.config, **patch})

    def test_missing_nonfinite_wrong_dimension_or_incomplete_splits_are_refused(self):
        mutations = [lambda arrays, scaler: arrays.__setitem__("X", arrays["X"][:, :1]),
                     lambda arrays, scaler: arrays["y"].__setitem__((0, 0), float("inf")),
                     lambda arrays, scaler: arrays["split_id"].__setitem__(arrays["split_id"] == 2, 1),
                     lambda arrays, scaler: arrays["split_id"].__setitem__(0, 3),
                     lambda arrays, scaler: arrays.pop("y_scaled")]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                dataset, *_ = self.dataset(mutate)
                with self.assertRaises(ContractError): train_supervised(self.workspace, dataset, self.config)

    def test_scaler_refit_on_all_rows_or_inconsistent_scaled_arrays_are_refused(self):
        def all_rows(arrays, scaler):
            scaler["X"]["location"] = arrays["X"].mean(0).tolist()
        def wrong_count(arrays, scaler): scaler["fit_samples"] += 1
        def wrong_split(arrays, scaler): scaler["fit_split"] = "all"
        def scaled(arrays, scaler): arrays["X_scaled"][0, 0] += .1
        for mutate in (all_rows, wrong_count, wrong_split, scaled):
            with self.subTest(mutate=mutate):
                dataset, *_ = self.dataset(mutate)
                with self.assertRaisesRegex(ContractError, "scaler|scaled arrays"):
                    train_supervised(self.workspace, dataset, self.config)

    def test_minmax_and_identity_scaling_remain_train_only_data_transforms(self):
        for method in ("minmax", "none"):
            with self.subTest(method=method):
                dataset, *_ = self.dataset(scaler=method)
                _, _, _, report = self.artifact(dataset, {**self.config, "epochs": 2})
                self.assertEqual(report["normalization"]["source"], "training_partition_only")

    def test_npz_huge_declared_shape_is_rejected_before_numpy_allocation(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (10**9, 2)})
        path = self.root / "shape-bomb.npz"
        with zipfile.ZipFile(path, "w") as archive:
            for name in ("X", "y", "X_scaled", "y_scaled", "split_id"):
                archive.writestr(name + ".npy", header.getvalue())
        with self.assertRaisesRegex(ContractError, "budget|member bytes"):
            _load_bounded_npz(path, DataInterface.parse(data_interface()))

    def test_loaded_npz_bytes_must_match_the_bound_artifact_version(self):
        dataset, *_ = self.dataset()
        root, metadata = self.workspace.resolve(dataset)
        with self.assertRaisesRegex(ContractError, "bytes changed"):
            _load_bounded_npz(root / "supervised.npz", DataInterface.parse(metadata["interface"]), "0" * 64)

    def test_small_parameter_network_with_giant_effective_batch_is_refused_before_training(self):
        # Only ~20k parameters and small raw arrays, but 2400*4100 hidden/input/
        # output activations would exceed the one-batch budget before backward.
        fractions = {"train": (.6, .2, .2), "validation": (.1, .8, .1), "test": (.1, .1, .8)}
        for part, ratio in fractions.items():
            with self.subTest(part=part):
                split = {"mode": "time", **dict(zip(("train", "validation", "test"), ratio))}
                dataset, *_ = self.dataset(rows=4000, split=split)
                with self.assertRaisesRegex(ContractError, part + r" batch.*activation.*reduce batch_size"):
                    train_supervised(self.workspace, dataset, {**self.config, "hidden_dims": [4096], "batch_size": 1_000_000, "epochs": 1})
        self.assertNotIn("supervised_model", self.workspace.manifest["active"])
        self.assertFalse(any((self.workspace.path / ".staging").glob("supervised_model-*")))

    def test_large_configured_batch_with_small_actual_splits_does_not_get_falsely_rejected(self):
        dataset, *_ = self.dataset()
        _, _, _, report = self.artifact(dataset, {**self.config, "epochs": 1, "batch_size": 1_000_000})
        self.assertEqual(report["optimizer_updates"], 1)
        self.assertEqual(report["resource_boundary"]["max_batch_activation_scalars"], 5_000_000)


if __name__ == "__main__": unittest.main()
