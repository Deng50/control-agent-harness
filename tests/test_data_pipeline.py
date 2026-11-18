"""Synthetic test_fixture data only; no plant inference, acquisition or training."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from control_harness.contracts import ContractError
from control_harness.datasets import inspect_table, prepare_dataset


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "data-pipeline-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
HAS_XLSX = importlib.util.find_spec("openpyxl") is not None


class DataPipelineTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.headers = ["time", "trip", "signal", "signal", "disturbance", "target1", "target2"]
        self.rows = [[i * .2, f"trip-{trip}", trip * 100 + i, trip * 100 + 2 * i,
                      .1 * i, trip * 1000 + 3 * i, trip * 1000 - i]
                     for trip in range(6) for i in range(20)]
        self.path = self.write_csv(self.rows)

    def write_csv(self, rows, name="fixture.csv"):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.headers)
            writer.writerows(rows)
        return path

    def column(self, index, sheet=None):
        return {"index": index, "header": self.headers[index], "worksheet": sheet}

    def interface(self, *, task="dynamic_prediction", sheet=None):
        signals = []
        for name, index, role, unit in (("input_a", 2, "control", "N"),
                                         ("input_b", 3, "state", "m"),
                                         ("disturbance", 4, "disturbance", "N"),
                                         ("target_a", 5, "output", "m"),
                                         ("target_b", 6, "output", "m/s")):
            signals.append({"name": name, "role": role, "unit": unit,
                            "column": self.column(index, sheet), "delay_steps": 0,
                            "observable": name not in {"target_a", "target_b"}})
        return {"schema_version": 1, "interface_kind": "tabular_supervised", "task": task,
                "sample_time": .2 if task == "dynamic_prediction" else None,
                "time_column": self.column(0, sheet) if task == "dynamic_prediction" else None,
                "trip_column": self.column(1, sheet), "signals": signals,
                "features": [{"signal": name, "window": 1, "lag": 0}
                             for name in ("input_a", "input_b", "disturbance")],
                "targets": [{"signal": name, "offset": int(task == "dynamic_prediction")}
                            for name in ("target_a", "target_b")],
                "input_dim": 3, "output_dim": 2,
                "source": {"kind": "synthetic", "confirmation_source": "test_fixture"}}

    def prepare(self, interface=None, *, path=None, mode="trip", **kwargs):
        return prepare_dataset(path or self.path, interface or self.interface(),
                               {"mode": mode, "train": .5, "validation": .2, "test": .3}, **kwargs)

    def test_inspection_preserves_duplicate_positions_and_never_writes_source(self):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        result = inspect_table(self.path)
        table = result["worksheets"][0]
        self.assertEqual(table["duplicate_headers"], [{"header": "signal", "indices": [2, 3]}])
        self.assertEqual([table["columns"][i]["index"] for i in (2, 3)], [2, 3])
        self.assertEqual(result["source"]["sha256"], before)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        self.assertFalse(result["inferred_plant_dimensions"])
        self.assertTrue(result["read_only"])

    def test_static_three_inputs_two_outputs_is_data_only_not_a_plant(self):
        result = self.prepare(self.interface(task="static_regression"))
        self.assertEqual(result.arrays["X"].shape, (120, 3))
        self.assertEqual(result.arrays["y"].shape, (120, 2))
        self.assertEqual((result.metadata["input_dim"], result.metadata["output_dim"]), (3, 2))
        self.assertFalse(result.metadata["plant_dimensions_inferred"])
        self.assertFalse(result.metadata["trained"])
        self.assertFalse(any(key in result.metadata for key in ("nx", "nu", "ny", "states")))
        json.dumps({key: getattr(result, key) for key in ("metadata", "split", "statistics", "scaler")}, allow_nan=False)

    def test_delay_lag_window_order_is_oldest_to_newest_and_explicit(self):
        interface = self.interface()
        interface["features"] = [{"signal": "input_a", "window": 3, "lag": 1}]
        interface["signals"][0]["delay_steps"] = 2
        interface["input_dim"] = 3
        result = self.prepare(interface)
        self.assertEqual([item["lag"] for item in result.metadata["feature_order"]], [5, 4, 3])
        self.assertEqual(result.arrays["anchor_row"][0], 7)
        np.testing.assert_array_equal(result.arrays["feature_rows"][0], [2, 3, 4])
        np.testing.assert_array_equal(result.arrays["X"][0], [0, 1, 2])
        np.testing.assert_array_equal(result.arrays["target_rows"][0], [8, 8])

    def test_trip_splits_are_independent_and_windows_stay_within_trip_and_split(self):
        interface = self.interface()
        interface["features"][0]["window"] = 4
        interface["input_dim"] = 6
        result = self.prepare(interface)
        split_groups = [set(result.split["groups"][name]) for name in ("train", "validation", "test")]
        self.assertFalse(split_groups[0] & split_groups[1] or split_groups[0] & split_groups[2] or split_groups[1] & split_groups[2])
        self.assertTrue(result.split["split_before_windowing"])
        for label, name in enumerate(("train", "validation", "test")):
            allowed = set(result.split["raw_rows"][name])
            for row in np.flatnonzero(result.arrays["split_id"] == label):
                used = list(result.arrays["feature_rows"][row]) + list(result.arrays["target_rows"][row])
                self.assertTrue(set(used) <= allowed)
                trips = {self.rows[int(index) - 2][1] for index in used}
                self.assertEqual(len(trips), 1)

    def test_time_splits_happen_before_windowing_with_no_shared_raw_records(self):
        interface = self.interface()
        interface["features"][0]["window"] = 3
        interface["input_dim"] = 5
        result = self.prepare(interface, mode="time")
        raw = [set(result.split["raw_rows"][name]) for name in ("train", "validation", "test")]
        self.assertFalse(raw[0] & raw[1] or raw[0] & raw[2] or raw[1] & raw[2])
        for i in range(len(result.arrays["X"])):
            allowed = raw[int(result.arrays["split_id"][i])]
            self.assertTrue(set(result.arrays["feature_rows"][i]) <= allowed)
            self.assertTrue(set(result.arrays["target_rows"][i]) <= allowed)

    def test_holdout_outliers_cannot_change_train_fitted_scaler(self):
        before = self.prepare()
        changed = copy.deepcopy(self.rows)
        # Train is trips 0..2; mutate both validation and test only.
        for row in changed[60:]:
            for column in range(2, 7):
                row[column] = row[column] * 1e6 + 1e9
        after = self.prepare(path=self.write_csv(changed, "holdout-outliers.csv"))
        self.assertEqual(before.scaler, after.scaler)
        np.testing.assert_array_equal(before.arrays["X_scaled"][before.arrays["split_id"] == 0],
                                      after.arrays["X_scaled"][after.arrays["split_id"] == 0])
        self.assertEqual(before.scaler["fit_split"], "train")
        self.assertGreater(after.arrays["X_scaled"][after.arrays["split_id"] == 2].max(), 1e6)

    def test_minmax_constant_columns_and_none_scaler_are_finite(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row[4] = 7
        path = self.write_csv(rows, "constant.csv")
        result = self.prepare(path=path, scaler="minmax")
        self.assertEqual(result.scaler["X"]["constant_columns"], [2])
        self.assertTrue(np.isfinite(result.arrays["X_scaled"]).all())
        self.assertGreater(result.arrays["X_scaled"][result.arrays["split_id"] == 2, 0].max(), 1)
        unscaled = self.prepare(path=path, scaler="none")
        np.testing.assert_array_equal(unscaled.arrays["X"], unscaled.arrays["X_scaled"])

    def test_duplicate_header_still_requires_correct_index_header_and_sheet(self):
        for change in (lambda value: value["signals"][0]["column"].pop("index"),
                       lambda value: value["signals"][0]["column"].update(header="other"),
                       lambda value: value["signals"][0]["column"].update(worksheet="Sheet1")):
            interface = self.interface()
            change(interface)
            with self.assertRaises(ContractError):
                self.prepare(interface)

    def test_column_aliases_and_mismatched_dimensions_are_rejected(self):
        interface = self.interface()
        interface["signals"][1]["column"] = copy.deepcopy(interface["signals"][0]["column"])
        with self.assertRaisesRegex(ContractError, "alias"):
            self.prepare(interface)
        for field, value in (("input_dim", 2), ("output_dim", 3), ("input_dim", True), ("nx", 2)):
            interface = self.interface()
            interface[field] = value
            with self.assertRaises(ContractError):
                self.prepare(interface)

    def test_current_target_as_feature_is_leakage_but_observed_history_is_causal(self):
        interface = self.interface()
        interface["signals"][3]["observable"] = True
        interface["features"] = [{"signal": "target_a", "window": 1, "lag": 0}]
        interface["targets"][0]["offset"] = 0
        interface["input_dim"] = 1
        with self.assertRaisesRegex(ContractError, "leakage"):
            self.prepare(interface)
        interface["features"][0]["lag"] = 1
        result = self.prepare(interface)
        self.assertTrue(np.all(result.arrays["feature_rows"][:, 0] < result.arrays["target_rows"][:, 0]))
        interface["signals"][3]["observable"] = False
        with self.assertRaisesRegex(ContractError, "observable"):
            self.prepare(interface)

    def test_future_features_negative_delays_and_invalid_windows_are_rejected(self):
        for field, value in (("lag", -1), ("window", 0), ("window", True)):
            interface = self.interface()
            interface["features"][0][field] = value
            with self.assertRaises(ContractError):
                self.prepare(interface)
        interface = self.interface()
        interface["signals"][0]["delay_steps"] = -1
        with self.assertRaises(ContractError):
            self.prepare(interface)

    def test_nonfinite_missing_and_formula_cells_are_rejected(self):
        for bad in ("nan", "inf", "", "=1+1", None):
            with self.subTest(value=bad):
                rows = copy.deepcopy(self.rows)
                rows[5][2] = bad
                with self.assertRaises(ContractError):
                    self.prepare(path=self.write_csv(rows, "invalid.csv"))

    def test_time_duplicates_reverse_order_and_undeclared_resampling_are_rejected(self):
        for value in (.6, -.1, 9):
            rows = copy.deepcopy(self.rows)
            rows[5][0] = value
            with self.assertRaisesRegex(ContractError, "Time"):
                self.prepare(path=self.write_csv(rows, "bad-time.csv"))
        interface = self.interface()
        interface["time_column"] = None
        with self.assertRaisesRegex(ContractError, "time column"):
            self.prepare(interface)

    def test_missing_trip_and_forbidden_random_split_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[5][1] = ""
        with self.assertRaisesRegex(ContractError, "Trip"):
            self.prepare(path=self.write_csv(rows, "missing-trip.csv"))
        with self.assertRaisesRegex(ContractError, "random-row"):
            self.prepare(mode="random")
        interface = self.interface()
        interface["trip_column"] = None
        with self.assertRaisesRegex(ContractError, "trip_column"):
            self.prepare(interface)

    def test_insufficient_windows_and_invalid_split_fractions_are_rejected(self):
        interface = self.interface()
        interface["features"][0]["window"] = 50
        interface["input_dim"] = 52
        with self.assertRaisesRegex(ContractError, "no valid windows"):
            self.prepare(interface)
        for split in ({"mode": "trip", "train": .8, "validation": .2, "test": .2},
                      {"mode": "trip", "train": True}, {"mode": "trip", "test": 0}):
            with self.assertRaises(ContractError):
                prepare_dataset(self.path, self.interface(), split)

    def test_state_mapping_cannot_turn_an_output_or_control_into_plant_state(self):
        interface = self.interface()
        interface["state_mapping"] = {"x1": "input_a"}
        with self.assertRaisesRegex(ContractError, "state-role"):
            self.prepare(interface)
        interface["state_mapping"] = {"x1": "input_b"}
        self.assertFalse(self.prepare(interface).metadata["plant_dimensions_inferred"])

    def test_two_state_one_control_and_four_state_two_control_remain_explicit_data_mappings(self):
        for state_count, control_count in ((2, 1), (4, 2)):
            with self.subTest(states=state_count, controls=control_count):
                state_names = [f"x{i}" for i in range(state_count)]
                control_names = [f"u{i}" for i in range(control_count)]
                target_names = [f"next_x{i}" for i in range(state_count)]
                headers = ["trip"] + state_names + control_names + target_names
                path = self.root / f"explicit-{state_count}-{control_count}.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(headers)
                    for trip in range(6):
                        for step in range(10):
                            state = [step + i for i in range(state_count)]
                            control = [trip + i for i in range(control_count)]
                            writer.writerow([f"synthetic-trip-{trip}"] + state + control + [2 * x + .1 * control[0] for x in state])
                column = lambda index: {"index": index, "header": headers[index], "worksheet": None}
                signals = [{"name": name, "role": "state" if name in state_names else "control" if name in control_names else "output",
                            "unit": "synthetic_unit", "column": column(i), "delay_steps": 0,
                            "observable": name not in target_names} for i, name in enumerate(headers[1:], 1)]
                interface = {"schema_version": 1, "interface_kind": "tabular_supervised", "task": "static_regression",
                             "signals": signals, "features": [{"signal": name, "window": 1, "lag": 0} for name in state_names + control_names],
                             "targets": [{"signal": name, "offset": 0} for name in target_names],
                             "trip_column": column(0), "time_column": None, "sample_time": None,
                             "state_mapping": {name: name for name in state_names}}
                result = self.prepare(interface, path=path)
                self.assertEqual(result.arrays["X"].shape[1], state_count + control_count)
                self.assertEqual(result.arrays["y"].shape[1], state_count)
                self.assertFalse(result.metadata["plant_dimensions_inferred"])
                self.assertNotIn("nx", result.metadata)

    def test_ragged_rows_and_missing_declared_units_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0].pop()
        with self.assertRaisesRegex(ContractError, "Ragged"):
            inspect_table(self.write_csv(rows, "ragged.csv"))
        interface = self.interface()
        interface["signals"][0]["unit"] = ""
        with self.assertRaisesRegex(ContractError, "unit"):
            self.prepare(interface)

    @unittest.skipUnless(HAS_XLSX, "openpyxl optional dependency not installed")
    def test_xlsx_worksheet_and_duplicate_columns_are_positional_and_read_only(self):
        import openpyxl
        path = self.root / "synthetic.xlsx"
        workbook = openpyxl.Workbook()
        for name in ("measured", "other"):
            sheet = workbook.active if name == "measured" else workbook.create_sheet()
            sheet.title = name
            sheet.append(self.headers)
            for row in self.rows:
                sheet.append(row)
        workbook.save(path)
        workbook.close()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory = inspect_table(path)
        self.assertEqual([item["worksheet"] for item in inventory["worksheets"]], ["measured", "other"])
        self.assertEqual(inventory["worksheets"][0]["duplicate_headers"][0]["indices"], [2, 3])
        result = self.prepare(self.interface(sheet="measured"), path=path)
        self.assertEqual(result.metadata["source"]["worksheet"], "measured")
        self.assertEqual(result.arrays["X"][0, 0], self.rows[0][2])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        interface = self.interface(sheet="measured")
        interface["signals"][1]["column"]["worksheet"] = "other"
        with self.assertRaisesRegex(ContractError, "worksheet"):
            self.prepare(interface, path=path)

    @unittest.skipUnless(HAS_XLSX, "openpyxl optional dependency not installed")
    def test_xlsx_formulas_are_never_evaluated_or_replaced_by_stale_cached_values(self):
        import openpyxl
        path = self.root / "formula.xlsx"
        workbook = openpyxl.Workbook()
        workbook.active.title = "measured"
        workbook.active.append(self.headers)
        for row in self.rows:
            workbook.active.append(row)
        workbook.active["C2"] = "=1+1"
        workbook.save(path)
        workbook.close()
        self.assertEqual(inspect_table(path)["worksheets"][0]["preview"][0][2], "=1+1")
        with self.assertRaisesRegex(ContractError, "formulas are not evaluated"):
            self.prepare(self.interface(sheet="measured"), path=path)

    @unittest.skipUnless(HAS_XLSX, "openpyxl optional dependency not installed")
    def test_xlsx_archive_paths_and_dtd_entities_are_rejected_before_xml_parser(self):
        payloads = [("../outside.xml", b"<root/>"), ("C:/outside.xml", b"<root/>"),
                    ("xl/workbook.xml", b'<!DOCTYPE root [<!ENTITY x "expanded">]><root>&x;</root>'),
                    ("xl/workbook.xml", '<!DOCTYPE root [<!ENTITY x "expanded">]><root/>'.encode("utf-16")),
                    ("xl/workbook.xml", b" " * (64 * 1024 - 4) + b'<!DOCTYPE root><root/>')]
        for index, (name, content) in enumerate(payloads):
            path = self.root / f"unsafe-{index}.xlsx"
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(name, content)
            with self.subTest(member=name, case=index), patch("openpyxl.load_workbook", side_effect=AssertionError("must not parse")):
                with self.assertRaisesRegex(ContractError, "unsafe|DTD/ENTITY"):
                    inspect_table(path)


if __name__ == "__main__":
    unittest.main()
