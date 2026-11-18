"""Independent S03 contract/dependency checks using synthetic test_fixture data.

Expected widths come from declared physical signals/window counts, not from a
network or a learned model. No acquisition or model training is performed.
"""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from control_harness.artifacts import Workspace, atomic_json
from control_harness.contracts import (ContractError, DataInterface, MaterialInterface,
                                      ModelInterface, parse_interface,
                                      validate_network_dimensions)
from control_harness.datasets import prepare_dataset


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "data-contract-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def col(index, header, worksheet=None):
    return {"index": index, "header": header, "worksheet": worksheet}


def data_definition(*, dynamic=False):
    names = ("force", "sensor", "disturbance", "position", "velocity")
    headers = ("value", "value", "input3", "output1", "output2")
    roles = ("control", "auxiliary", "disturbance", "output", "output")
    units = ("N", "V", "N", "m", "m/s")
    return {"schema_version": 1, "interface_kind": "tabular_supervised", "name": "synthetic test_fixture",
            "task": "dynamic_prediction" if dynamic else "static_regression",
            "sample_time": .2 if dynamic else None,
            "time_column": col(0, "time") if dynamic else None, "trip_column": col(6, "trip"),
            "signals": [{"name": name, "role": role, "unit": unit, "column": col(i, header),
                         "delay_steps": 0, "observable": i <= 3}
                        for i, (name, header, role, unit) in enumerate(zip(names, headers, roles, units), 1)],
            "features": [{"signal": name, "window": 1, "lag": 0} for name in names[:3]],
            "targets": [{"signal": name, "offset": int(dynamic)} for name in names[3:]]}


def model_definition(nx, nu):
    variable = lambda name, unit: {"name": name, "unit": unit, "lower": -100, "upper": 100}
    return {"schema_version": 1, "name": "synthetic physical interface", "sample_time": .2,
            "time_semantics": "discrete_state_transition",
            "states": [variable(f"x{i}", "m") for i in range(nx)],
            "controls": [variable(f"u{i}", "N") for i in range(nu)],
            "outputs": [variable("y0", "m")], "disturbances": []}


class DataContractTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv = self.root / "synthetic.csv"
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["time", "value", "value", "input3", "output1", "output2", "trip"])
            for trip in range(6):
                for step in range(20):
                    writer.writerow([step * .2, step + trip, 2 * step, .1 * trip,
                                     3 * step + trip, -step, f"synthetic-trip-{trip}"])

    def test_three_inputs_two_outputs_are_supervised_widths_without_plant_dimensions(self):
        interface = DataInterface.parse(data_definition())
        self.assertEqual((interface.input_dim, interface.output_dim), (3, 2))
        self.assertFalse(hasattr(interface, "nx"))
        self.assertFalse(hasattr(interface, "nu"))
        for key in ("nx", "nu", "ny", "nd", "states"):
            self.assertNotIn(key, interface.to_dict())
            bad = data_definition()
            bad[key] = [] if key == "states" else 2
            with self.subTest(field=key), self.assertRaises(ContractError):
                DataInterface.parse(bad)

    def test_physical_dimensions_and_history_window_width_are_distinct(self):
        for nx, nu in ((2, 1), (4, 2)):
            with self.subTest(nx=nx, nu=nu):
                physical = ModelInterface.parse(model_definition(nx, nu))
                self.assertEqual((physical.nx, physical.nu, physical.ny), (nx, nu, 1))
                self.assertEqual(validate_network_dimensions(physical, {}, residual=True), (nx + nu, nx))
                raw = {"schema_version": 1, "interface_kind": "tabular_supervised", "task": "dynamic_prediction",
                       "sample_time": .2, "time_column": col(0, "time"), "signals": [], "features": [], "targets": []}
                for i, variable in enumerate(physical.states + physical.controls, 1):
                    is_state = i <= nx
                    raw["signals"].append({"name": variable.name, "role": "state" if is_state else "control",
                                           "unit": variable.unit, "column": col(i, variable.name), "delay_steps": 0, "observable": True})
                    raw["features"].append({"signal": variable.name, "window": 3, "lag": 0})
                    if is_state:
                        raw["targets"].append({"signal": variable.name, "offset": 1})
                raw["state_mapping"] = {state.name: state.name for state in physical.states}
                supervised = DataInterface.parse(raw)
                self.assertEqual((supervised.input_dim, supervised.output_dim), (3 * (nx + nu), nx))
                self.assertNotEqual(supervised.input_dim, nx + nu)
                self.assertFalse(hasattr(supervised, "nx"))

    def test_user_dimension_conflicts_are_rejected_before_data_read(self):
        for key, value in (("input_dim", 4), ("output_dim", 3), ("input_dim", True), ("output_dim", 2.0)):
            raw = data_definition()
            raw[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                DataInterface.parse(raw)

    def test_to_dict_is_independent_json_lists_and_prepares_real_arrays(self):
        raw = data_definition()
        raw["features"] = tuple(raw["features"])
        interface = DataInterface.parse(raw)
        encoded = interface.to_dict()
        self.assertIsInstance(encoded["signals"], list)
        self.assertIsInstance(encoded["features"], list)
        self.assertIsInstance(encoded["targets"], list)
        json.dumps(encoded, allow_nan=False)
        result = prepare_dataset(self.csv, encoded, {"mode": "trip"})
        self.assertEqual(result.arrays["X"].shape, (120, 3))
        self.assertEqual(result.arrays["y"].shape, (120, 2))
        fingerprint = interface.fingerprint
        encoded["signals"][0]["unit"] = "wrong"
        raw["signals"][0]["name"] = "mutated after parse"
        self.assertEqual(interface.fingerprint, fingerprint)

    def test_roles_units_availability_and_observability_are_explicit(self):
        changes = [("role", "input"), ("unit", ""), ("unit", None), ("delay_steps", -1),
                   ("delay_steps", True), ("observable", "true"), ("observable", False)]
        for field, value in changes:
            raw = data_definition()
            raw["signals"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                DataInterface.parse(raw)

    def test_duplicate_headers_are_allowed_but_duplicate_source_positions_are_not(self):
        raw = data_definition()
        self.assertEqual(raw["signals"][0]["column"]["header"], raw["signals"][1]["column"]["header"])
        self.assertEqual(DataInterface.parse(raw).input_dim, 3)
        raw["signals"][1]["column"]["index"] = 1
        with self.assertRaises(ContractError):
            DataInterface.parse(raw)
        for field in ("index", "header", "worksheet"):
            raw = data_definition()
            raw["signals"][0]["column"].pop(field)
            with self.subTest(missing=field), self.assertRaises(ContractError):
                DataInterface.parse(raw)

    def test_state_mapping_is_optional_explicit_and_never_infers_state_roles(self):
        raw = data_definition()
        raw["signals"][1]["role"] = "state"
        raw["state_mapping"] = {"position": "sensor"}
        interface = DataInterface.parse(raw)
        self.assertEqual(interface.to_dict()["state_mapping"], {"position": "sensor"})
        self.assertFalse(hasattr(interface, "nx"))
        for mapping in ({"x": "force"}, {"x": "missing"}, {"x": "sensor", "y": "sensor"},
                        {"": "sensor"}, [], False, "", {"x": []}):
            bad = copy.deepcopy(raw)
            bad["state_mapping"] = mapping
            with self.subTest(mapping=mapping), self.assertRaises(ContractError):
                DataInterface.parse(bad)

    def test_missing_or_unknown_targets_and_overlap_are_rejected(self):
        for targets in ([], None, [{"signal": "missing", "offset": 0}],
                        [{"signal": "position", "offset": 0}, {"signal": "position", "offset": 0}]):
            raw = data_definition()
            raw["targets"] = targets
            with self.subTest(targets=targets), self.assertRaises(ContractError):
                DataInterface.parse(raw)
        raw = data_definition()
        raw["features"].append(copy.deepcopy(raw["features"][0]))
        with self.assertRaises(ContractError):
            DataInterface.parse(raw)

    def test_target_leakage_is_rejected_but_observable_past_output_is_allowed(self):
        raw = data_definition(dynamic=True)
        raw["signals"][3]["observable"] = True
        raw["features"] = [{"signal": "position", "window": 2, "lag": 0}]
        raw["targets"] = [{"signal": "position", "offset": 0}]
        with self.assertRaises(ContractError):
            DataInterface.parse(raw)
        raw["features"][0]["lag"] = 1
        interface = DataInterface.parse(raw)
        result = prepare_dataset(self.csv, interface.to_dict(), {"mode": "trip"})
        self.assertTrue(np.all(result.arrays["feature_rows"][:, -1] < result.arrays["target_rows"][:, 0]))
        raw["features"][0]["lag"] = -1
        with self.assertRaises(ContractError):
            DataInterface.parse(raw)

    def test_static_and_dynamic_time_boundaries_agree_with_preparation(self):
        for key in ("time_column", "sample_time"):
            raw = data_definition(dynamic=True)
            raw[key] = None
            with self.subTest(missing=key), self.assertRaises(ContractError):
                DataInterface.parse(raw)
        for dt in (0, -.2, True, float("nan"), float("inf")):
            raw = data_definition(dynamic=True)
            raw["sample_time"] = dt
            with self.subTest(sample_time=dt), self.assertRaises(ContractError):
                DataInterface.parse(raw)
        # Calling an interface "static" must not authorize future labels or
        # history windows merely because a time column happens to exist.
        for change in (lambda raw: raw["targets"][0].update(offset=1),
                       lambda raw: raw["features"][0].update(window=2),
                       lambda raw: raw["signals"][0].update(delay_steps=1)):
            raw = data_definition()
            raw["time_column"] = col(0, "time")
            raw["sample_time"] = .2
            change(raw)
            with self.assertRaises(ContractError):
                DataInterface.parse(raw)

    def test_dispatch_and_material_formats_do_not_create_a_model_interface(self):
        self.assertIsInstance(parse_interface(data_definition()), DataInterface)
        material = {"schema_version": 1, "interface_kind": "material", "format": "csv", "name": "synthetic test_fixture"}
        self.assertIsInstance(parse_interface(material), MaterialInterface)
        self.assertFalse(hasattr(parse_interface(material), "nx"))
        self.assertIsInstance(parse_interface(model_definition(2, 1)), ModelInterface)
        for raw in (None, [], "csv", {"schema_version": 1, "interface_kind": "unknown"},
                    material | {"format": "exe"}):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                parse_interface(raw)


class DataArtifactContractTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = Workspace(self.root / "source")
        self.data_interface = DataInterface.parse(data_definition())
        self.material_interface = MaterialInterface.parse({"schema_version": 1, "interface_kind": "material", "format": "csv"})

    def publish(self, workspace, kind, interface, parents=(), *, compatible_parents=True, value=1):
        with workspace.stage(kind, interface, list(parents), {"confirmation_source": "test_fixture"},
                             compatible_parents=compatible_parents) as stage:
            atomic_json(stage.root / "fixture.json", {"synthetic": True, "value": value})
            return stage.publish({"status": "verified"}, source={"kind": "synthetic", "confirmation_source": "test_fixture"})

    def chain(self):
        material = self.publish(self.source, "source_material", self.material_interface)
        dataset = self.publish(self.source, "supervised_dataset", self.data_interface, [material], compatible_parents=False)
        report = self.publish(self.source, "dataset_report", self.data_interface, [dataset])
        return material, dataset, report

    def test_data_material_dependency_requires_explicit_incompatible_parent_opt_out(self):
        material = self.publish(self.source, "source_material", self.material_interface)
        before = self.source.manifest
        with self.assertRaises(ContractError):
            self.publish(self.source, "supervised_dataset", self.data_interface, [material])
        self.assertEqual(self.source.manifest, before)
        dataset = self.publish(self.source, "supervised_dataset", self.data_interface, [material], compatible_parents=False)
        _, metadata = self.source.resolve(dataset)
        self.assertEqual(metadata["parents"][0]["id"], material)
        self.assertIsInstance(parse_interface(metadata["interface"]), DataInterface)

    def test_default_legacy_model_dependencies_keep_strict_unit_and_dimension_checks(self):
        physical = ModelInterface.parse(model_definition(2, 1))
        parent = self.publish(self.source, "nominal_model", physical)
        self.publish(self.source, "controller", physical, [parent])
        for specification in (model_definition(4, 2), model_definition(2, 1)):
            if len(specification["states"]) == 2:
                specification["controls"][0]["unit"] = "kN"
            with self.subTest(specification=specification), self.assertRaises(ContractError):
                self.publish(self.source, "controller", ModelInterface.parse(specification), [parent])

    def test_data_only_artifact_closure_import_is_hash_checked_and_source_independent(self):
        material, dataset, report = self.chain()
        target = Workspace(self.root / "target")
        report_root, original_meta = self.source.resolve(report)
        self.assertEqual(target.import_artifact(report_root), report)
        self.assertEqual(set(target.manifest["artifacts"]), {material, dataset, report})
        self.assertEqual(target.resolve(report)[1], original_meta)
        material_root, _ = self.source.resolve(material)
        (material_root / "fixture.json").write_text('{"tampered": true}', encoding="utf-8")
        with self.assertRaises(ContractError):
            self.source.resolve(report)
        target.resolve(report)
        self.assertEqual(target.import_artifact(target.resolve(report)[0]), report)

    def test_material_replacement_invalidates_data_and_reports_transitively(self):
        material, dataset, report = self.chain()
        replacement = self.publish(self.source, "source_material", self.material_interface, value=2)
        self.assertNotEqual(material, replacement)
        self.source.resolve(material)  # Historical immutable input remains inspectable.
        for child in (dataset, report):
            with self.assertRaises(ContractError):
                self.source.resolve(child)
            self.assertEqual(self.source.manifest["artifacts"][child]["status"], "stale")
        self.assertNotIn("supervised_dataset", self.source.manifest["active"])
        self.assertNotIn("dataset_report", self.source.manifest["active"])

    def test_imported_data_dependency_can_be_explicitly_activated_then_revised(self):
        material, dataset, report = self.chain()
        target = Workspace(self.root / "target")
        target.import_artifact(self.source.resolve(report)[0])
        target.import_artifact(self.source.resolve(material)[0])
        replacement = self.publish(target, "source_material", self.material_interface, value=3)
        self.assertNotEqual(material, replacement)
        for child in (dataset, report):
            with self.assertRaises(ContractError):
                target.resolve(child)
        self.source.resolve(report)  # A target-workspace revision is independent.

    def test_opt_out_must_be_a_boolean_not_missing_or_false_like_configuration(self):
        material = self.publish(self.source, "source_material", self.material_interface)
        for value in (None, 0, ""):
            with self.subTest(compatible_parents=value), self.assertRaises(ContractError):
                self.publish(self.source, "supervised_dataset", self.data_interface, [material], compatible_parents=value)


if __name__ == "__main__":
    unittest.main()
