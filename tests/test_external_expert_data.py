"""S09 real tabular and common collector data contracts; no policy training."""
from __future__ import annotations

import copy
import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.contracts import ContractError, DataInterface, ModelInterface
from control_harness.external_expert_data import (import_expert_data,
    publish_external_expert_dataset, read_external_expert_dataset, validate_expert_interface)
from control_harness.material_registration import register_material
from control_harness.runtime import OperationContext


ROOT = Path(__file__).resolve().parents[1]
HEADERS = ["time", "trip", "observation", "reference", "previous_a", "previous_b", "action_a", "action_b", "memory"]


def interface():
    roles = [("obs", 2, "output", "observation", True), ("ref", 3, "auxiliary", "reference", True),
             ("prev_a", 4, "control", "previous_action", True), ("prev_b", 5, "control", "previous_action", True),
             ("act_a", 6, "control", "expert_action", True), ("act_b", 7, "control", "expert_action", True),
             ("memory", 8, "auxiliary", "diagnostic_memory", False)]
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "task": "dynamic_prediction",
        "sample_time": .2, "time_column": {"index": 0, "header": "time", "worksheet": None},
        "trip_column": {"index": 1, "header": "trip", "worksheet": None},
        "signals": [{"name": name, "unit": "N" if index in (4, 5, 6, 7) else "1", "role": role,
                     "expert_role": expert, "observable": observable, "delay_steps": 0,
                     "column": {"index": index, "header": HEADERS[index], "worksheet": None}}
                    for name, index, role, expert, observable in roles],
        "features": [{"signal": name, "lag": 0, "window": 1} for name in ("obs", "ref", "prev_a", "prev_b")],
        "targets": [{"signal": name, "offset": 0} for name in ("act_a", "act_b")]}


def mapping():
    return {"signals": {"prev_a": {"time_semantics": "already_previous_applied_action", "action_target": "act_a"},
                        "prev_b": {"time_semantics": "already_previous_applied_action", "action_target": "act_b"}}}


def config():
    return {"interface": interface(), "source_mapping": mapping(), "scaler": "standard",
        "split": {"mode": "trip", "train": .5, "validation": 1 / 6, "test": 1 / 3},
        "source": {"kind": "recorded_external_expert", "description": "Handcrafted two-action causal expert records; not measured plant data",
                   "previous_action_semantics": "actual_applied_action_before_current_decision"},
        "action_constraints": {"bounds": [[-10., 10.], [-10., 10.]], "rate_bounds": [[-1., 1.], [-1., 1.]],
                               "rate_semantics": "per_control_step"}}


class ExternalExpertDataTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "test-output" / "external-expert-data-tests"
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ws = Workspace(self.root / "case")
        self.context = OperationContext()

    def records(self):
        rows = []
        for trip in range(6):
            previous = np.array([1. + .5 * trip, -1. - .3 * trip])
            for step in range(12):
                action = previous + [.1, -.05]
                rows.append([step * .2, trip, trip + step * .1, .5 + trip, *previous, *action, step * 100])
                previous = action
        return rows

    def material(self, xlsx=False):
        path = self.root / ("external.xlsx" if xlsx else "external.csv")
        if xlsx:
            import openpyxl
            workbook = openpyxl.Workbook(); sheet = workbook.active; sheet.title = "expert"
            sheet.append(HEADERS)
            for row in self.records(): sheet.append(row)
            workbook.save(path); workbook.close()
        else:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(HEADERS); writer.writerows(self.records())
        return register_material(self.ws, str(path), {}, self.context)

    def imported(self, cfg=None, xlsx=False):
        cfg = copy.deepcopy(cfg or config())
        if xlsx:
            for item in cfg["interface"]["signals"]: item["column"]["worksheet"] = "expert"
            for key in ("time_column", "trip_column"): cfg["interface"][key]["worksheet"] = "expert"
            cfg["worksheet"] = "expert"
        artifact = import_expert_data(self.ws, self.material(xlsx), cfg, self.context)
        root, meta = self.ws.resolve(artifact)
        return artifact, root, meta

    def collector(self):
        definition = DataInterface.parse(interface())
        rows = np.asarray(self.records(), dtype=np.float64)
        arrays = {"X": rows[:, [2, 3, 4, 5]], "y": rows[:, [6, 7]], "trajectory_id": rows[:, 1].astype(np.int64),
                  "time": rows[:, 0], "actual_previous_u": rows[:, [4, 5]], "raw_true_x": rows[:, [2]]}
        split = {"train": [0, 1, 2], "validation": [3], "test": [4, 5]}
        cfg = config()
        source = {**cfg["source"], "kind": "declared_simulation"}
        return arrays, definition, split, cfg, source

    def publish(self, arrays=None, definition=None, split=None, cfg=None, source=None, **kwargs):
        a, i, s, c, p = self.collector()
        return publish_external_expert_dataset(self.ws, a if arrays is None else arrays,
            i if definition is None else definition, s if split is None else split, c if cfg is None else cfg,
            self.context, source=p if source is None else source, source_mapping=mapping(), **kwargs)

    def test_csv_two_action_import_keeps_actual_columns_split_and_train_only_scaler(self):
        artifact, root, meta = self.imported()
        arrays, data, description, scaler, loaded = read_external_expert_dataset(self.ws, artifact)
        self.assertEqual(meta["kind"], "external_expert_dataset")
        self.assertEqual((data.input_dim, data.output_dim), (4, 2))
        self.assertFalse(description["plant_dimensions_inferred"])
        self.assertNotIn("nx", meta["interface"])
        self.assertIsNone(read_json(root / "physical_context.json")["interface"])
        self.assertEqual(description["source_mapping"]["signals"]["prev_a"]["column"]["index"], 4)
        self.assertEqual(read_json(root / "split.json")["sample_counts"], {"train": 36, "validation": 12, "test": 24})
        train = arrays["split_id"] == 0
        np.testing.assert_allclose(scaler["X"]["location"], arrays["X"][train].mean(0), atol=1e-14)
        self.assertGreater(np.max(np.abs(np.asarray(scaler["X"]["location"]) - arrays["X"].mean(0))), .1)
        with np.load(root / "expert.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["actual_previous_u"], raw["X"][:, 2:4])
        self.assertNotIn("controller_id", meta["source"])
        self.assertEqual(read_json(root / "report.json")["training"], "not_run")
        self.assertEqual(meta, loaded)

    def test_xlsx_uses_same_causal_preparer_and_original_worksheet(self):
        _, root, _ = self.imported(xlsx=True)
        data = read_json(root / "dataset.json")
        self.assertEqual(data["source_mapping"]["source_format"], "xlsx")
        self.assertEqual(data["source_mapping"]["signals"]["obs"]["column"]["worksheet"], "expert")
        with np.load(root / "expert.npz") as raw: self.assertEqual(raw["y"].shape, (72, 2))

    def test_time_split_precedes_history_windows_no_fake_independent_trajectories(self):
        cfg = config(); cfg["split"] = {"mode": "time", "train": .5, "validation": .25, "test": .25}
        cfg["interface"]["features"][0]["window"] = 2
        _, root, _ = self.imported(cfg)
        split = read_json(root / "split.json")
        self.assertEqual(split["holdout_semantics"], "chronological_partitions_within_source_trips")
        with np.load(root / "expert.npz") as arrays:
            for index, part in enumerate(("train", "validation", "test")):
                allowed = set(split["raw_rows"][part]); mask = arrays["split_id"] == index
                self.assertTrue(set(arrays["feature_rows"][mask].flat).issubset(allowed))
                self.assertTrue(set(arrays["target_rows"][mask].flat).issubset(allowed))
            self.assertEqual(set(arrays["trajectory_id"][arrays["split_id"] == 0]), set(arrays["trajectory_id"][arrays["split_id"] == 2]))

    def test_same_target_explicit_lag_one_is_causal_and_not_shifted_twice(self):
        cfg = config(); cfg["interface"]["features"] = [{"signal": "obs", "lag": 0, "window": 1},
            {"signal": "act_a", "lag": 1, "window": 1}, {"signal": "act_b", "lag": 1, "window": 1}]
        _, root, _ = self.imported(cfg)
        with np.load(root / "expert.npz") as arrays:
            self.assertEqual(len(arrays["X"]), 66)
            np.testing.assert_allclose(arrays["actual_previous_u"], arrays["X"][:, 1:])
            np.testing.assert_allclose(arrays["y"] - arrays["actual_previous_u"], np.tile([.1, -.05], (66, 1)), atol=1e-14)
        self.assertEqual(read_json(root / "dataset.json")["feature_order"][1]["lag"], 1)

    def test_future_current_target_unobservable_and_memory_features_are_rejected(self):
        for mutate, expression in ((lambda d: d["features"].append({"signal": "act_a", "lag": 0, "window": 1}), "leaks"),
             (lambda d: d["features"][0].update(lag=-1), "integer"),
             (lambda d: d["signals"][0].update(observable=False), "Unobservable"),
             (lambda d: (d["signals"][-1].update(observable=True), d["features"].append({"signal": "memory", "lag": 0, "window": 1})), "memory"),
             (lambda d: d["targets"][0].update(offset=1), "offset 0")):
            with self.subTest(expression=expression):
                definition = interface(); mutate(definition)
                with self.assertRaisesRegex(ContractError, expression): validate_expert_interface(definition, mapping())

    def test_previous_action_column_requires_declared_time_and_target_binding(self):
        with self.assertRaisesRegex(ContractError, "previous_action column"):
            validate_expert_interface(interface(), {})
        wrong = mapping(); wrong["signals"]["prev_a"]["action_target"] = "obs"
        with self.assertRaises(ContractError): validate_expert_interface(interface(), wrong)

    def test_common_writer_preserves_predefined_groups_and_diagnostic_memory_only(self):
        artifact = self.publish(diagnostics={"memory_per_step": [{"counter": k} for k in range(72)]},
                                report={"expert_calls": 72, "measured": False})
        root, meta = self.ws.resolve(artifact)
        split = read_json(root / "split.json")
        self.assertEqual(split["groups"], {"train": [0, 1, 2], "validation": [3], "test": [4, 5]})
        self.assertEqual(read_json(root / "diagnostics.json")["memory_per_step"][1]["counter"], 1)
        self.assertFalse(read_json(root / "dataset.json")["diagnostic_memory_is_feature"])
        self.assertEqual(read_json(root / "report.json")["expert_calls"], 72)
        with np.load(root / "expert.npz", allow_pickle=False) as arrays:
            self.assertNotIn("memory", arrays.files)
            self.assertIn("raw_true_x", arrays.files)
        self.assertEqual(meta["parents"], [])

    def test_reuses_existing_collector_stage_without_second_publication(self):
        a, i, s, c, source = self.collector()
        with self.ws.stage("external_expert_dataset", i, [], c, compatible_parents=False) as stage:
            result = publish_external_expert_dataset(self.ws, a, i, s, c, self.context,
                source=source, source_mapping=mapping(), stage=stage)
            self.assertEqual(result, stage.id); self.assertTrue(stage.committed)
        self.assertEqual(len(self.ws.manifest["artifacts"]), 1)

    def test_bounds_rate_and_initial_previous_action_are_never_inferred(self):
        a, i, s, c, p = self.collector()
        no_bounds = copy.deepcopy(c); no_bounds["action_constraints"].pop("bounds")
        with self.assertRaisesRegex(ContractError, "action bounds"): self.publish(cfg=no_bounds)
        bad_bounds = copy.deepcopy(c); bad_bounds["action_constraints"]["bounds"][0] = [-.01, .01]
        with self.assertRaisesRegex(ContractError, "actuator bounds"): self.publish(cfg=bad_bounds)
        bad_rate = copy.deepcopy(c); bad_rate["action_constraints"]["rate_bounds"] = [[-.01, .01], [-1., 1.]]
        with self.assertRaisesRegex(ContractError, "rate bounds"): self.publish(cfg=bad_rate)
        # Remove previous features/evidence; no assumed initial zeros may pass.
        d = interface(); d["features"] = d["features"][:2]; a["X"] = a["X"][:, :2]; a.pop("actual_previous_u")
        with self.assertRaisesRegex(ContractError, "actual previous actions"):
            self.publish(arrays=a, definition=DataInterface.parse(d))
        c["action_constraints"]["rate_bounds"] = None; p["previous_action_semantics"] = "not_recorded"
        result = self.publish(arrays=a, definition=DataInterface.parse(d), cfg=c, source=p)
        root, _ = self.ws.resolve(result)
        self.assertEqual(read_json(root / "action_constraints.json")["rate_check"], "not_requested")

    def test_no_forged_controller_identity_or_unlabelled_measurement_provenance(self):
        _, _, _, _, source = self.collector()
        source["controller_id"] = "controller-invented"
        with self.assertRaisesRegex(ContractError, "forge"): self.publish(source=source)
        source.pop("controller_id"); source["kind"] = "measured"
        with self.assertRaisesRegex(ContractError, "recorded_external_expert"): self.publish(source=source)
        source["kind"] = "declared_simulation"; source["measured"] = True
        with self.assertRaisesRegex(ContractError, "real measured"): self.publish(source=source)

    def test_partial_prefix_is_archivable_but_cannot_publish_trainable_dataset(self):
        a, i, s, c, source = self.collector()
        with self.assertRaisesRegex(ContractError, "Only complete"):
            with self.ws.stage("external_expert_dataset", i, [], c, compatible_parents=False) as stage:
                np.savez_compressed(stage.root / "partial-expert.npz", **a)
                publish_external_expert_dataset(self.ws, a, i, s, c, self.context, source=source,
                    source_mapping=mapping(), report={"status": "partial", "valid_rows": len(a["X"])}, stage=stage)
        self.assertEqual(self.ws.manifest["artifacts"], {})
        failed = list((self.ws.path / "failed").iterdir())
        self.assertEqual(len(failed), 1)
        self.assertTrue((failed[0] / "partial-expert.npz").is_file())
        self.assertTrue((failed[0] / "failure.json").is_file())

    def test_split_overlap_missing_groups_time_errors_and_nonfinite_arrays_rejected(self):
        for split in ({"train": [0, 1], "validation": [1], "test": [2, 3, 4, 5]},
                      {"train": [0, 1, 2], "validation": [3], "test": [4]}):
            with self.assertRaises(ContractError): self.publish(split=split)
        a, *_ = self.collector(); a["time"][1] = 0
        with self.assertRaisesRegex(ContractError, "decision times"): self.publish(arrays=a)
        a, *_ = self.collector(); a["y"][0, 0] = np.nan
        with self.assertRaisesRegex(ContractError, "non-finite"): self.publish(arrays=a)

    def test_resources_object_complex_and_unicode_non_ids_fail_before_publication(self):
        for value in (np.array([[object()]]), np.array([[1 + 2j]]), np.array([["hidden"]])):
            a, *_ = self.collector(); a["bad"] = value
            with self.assertRaises(ContractError): self.publish(arrays=a)
        a, *_ = self.collector()
        # Zero-stride view represents a huge logical array without allocating it.
        a["oversized"] = np.broadcast_to(np.ones(1), (72, 120000))
        with self.assertRaisesRegex(ContractError, "budget"): self.publish(arrays=a)
        self.assertEqual(self.ws.manifest["artifacts"], {})

    def test_simulation_only_declared_observation_is_reported_not_hardware_claim(self):
        d = interface(); d["signals"][0]["availability"] = "simulation_only"
        artifact = self.publish(definition=DataInterface.parse(d))
        root, _ = self.ws.resolve(artifact)
        self.assertEqual(read_json(root / "report.json")["deployment_scope"], "simulation_only")

    def test_collector_array_mapping_must_match_archived_feature_and_target_values(self):
        a, i, s, c, source = self.collector()
        mapped = mapping(); mapped["signals"]["obs"] = {"array": "raw_true_x", "column": 0}
        mapped["signals"]["act_a"] = {"array": "raw_u", "column": 0}
        a["raw_u"] = a["y"].copy()
        result = publish_external_expert_dataset(self.ws, a, i, s, c, self.context, source=source, source_mapping=mapped)
        self.ws.resolve(result)
        a["raw_u"][0, 0] += .1
        with self.assertRaisesRegex(ContractError, "target values disagree"):
            publish_external_expert_dataset(self.ws, a, i, s, c, self.context, source=source, source_mapping=mapped)

    def test_physical_context_keeps_plant_dt_independent_of_expert_control_period(self):
        variable = lambda name: {"name": name, "unit": "1", "lower": -10., "upper": 10.}
        physical = ModelInterface.parse({"schema_version": 1, "name": "explicit independent context", "sample_time": .1,
            "time_semantics": "discrete_state_transition", "states": [variable("actual_state")],
            "controls": [variable("a"), variable("b")], "outputs": [variable("measurement")], "disturbances": []})
        _, _, _, cfg, _ = self.collector()
        cfg["physical_mapping"] = {"plant_dt": .1, "control_period": .2, "plant_substeps": 2}
        artifact = self.publish(cfg=cfg, physical_interface=physical)
        root, meta = self.ws.resolve(artifact); context = read_json(root / "physical_context.json")
        self.assertEqual(context["plant_dt"], .1); self.assertEqual(context["control_period"], .2)
        self.assertEqual(context["plant_substeps"], 2); self.assertEqual(len(context["interface"]["states"]), 1)
        self.assertEqual(meta["interface"]["input_dim"], 4)  # Never nx=feature width.
        cfg["physical_mapping"]["plant_substeps"] = 3
        with self.assertRaisesRegex(ContractError, "control periods"):
            self.publish(cfg=cfg, physical_interface=physical)

    def test_npz_declared_giant_shape_is_rejected_before_numpy_allocation(self):
        definition = DataInterface.parse(interface())
        with self.ws.stage("external_expert_dataset", definition, [], {"test_fixture": "malicious NPY header"}, compatible_parents=False) as stage:
            with zipfile.ZipFile(stage.root / "expert.npz", "w") as archive:
                for key, values in (("X", np.zeros((3, 4))), ("y", np.zeros((3, 2))), ("X_scaled", np.zeros((3, 4))),
                                    ("y_scaled", np.zeros((3, 2))), ("split_id", np.array([0, 1, 2]))):
                    buffer = io.BytesIO()
                    if key == "X":
                        np.lib.format.write_array_header_1_0(buffer, {"descr": "<f8", "fortran_order": False, "shape": (10**12, 4)})
                        buffer.write(b"\0" * 8)
                    else: np.save(buffer, values, allow_pickle=False)
                    archive.writestr(key + ".npy", buffer.getvalue())
            atomic_json(stage.root / "dataset.json", {"source_mapping": mapping()})
            artifact = stage.publish({"status": "verified"}, activate=False)
        with self.assertRaisesRegex(ContractError, "budget|member bytes"):
            read_external_expert_dataset(self.ws, artifact)

    def test_corrupted_artifact_bytes_rejected_by_bounded_reader(self):
        artifact = self.publish(); root, _ = self.ws.resolve(artifact)
        with (root / "expert.npz").open("ab") as stream: stream.write(b"tampered")
        with self.assertRaises(ContractError): read_external_expert_dataset(self.ws, artifact)


if __name__ == "__main__": unittest.main()
