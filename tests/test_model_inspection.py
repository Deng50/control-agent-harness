from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from control_harness.contracts import ContractError
from control_harness.examples import example
from control_harness.model_inspection import inspect_model_source, matlab_runtime_status
from control_harness.runtime import Cancelled, OperationContext

TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "inspection-unit-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class ModelInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_complete_json_recovers_declared_interface_without_mutation(self):
        spec = example("linear")["nominal"]
        path = self.source("model.json", json.dumps(spec))
        before = path.read_bytes()
        report = inspect_model_source(path)
        self.assertTrue(report["registration_ready"])
        self.assertEqual(report["editable_model_spec"], spec)
        self.assertEqual(report["candidates"][0]["dimensions"], {"states": 4, "controls": 2, "outputs": 2, "disturbances": 1})
        self.assertFalse(report["source_executed"])
        self.assertFalse(report["registered"])
        self.assertEqual(path.read_bytes(), before)

    def test_missing_semantics_and_dimension_conflicts_remain_explicit(self):
        spec = example("nonlinear")["nominal"]
        del spec["interface"]["states"][0]["unit"]
        report = inspect_model_source(self.source("partial.json", json.dumps(spec)))
        self.assertFalse(report["registration_ready"])
        self.assertIn("interface.states[0].unit", report["missing"])
        spec["interface"]["states"][0]["unit"] = "1"
        spec["B"] = [[1, 2, 3]]
        report = inspect_model_source(self.source("bad-shape.json", json.dumps(spec)))
        self.assertFalse(report["registration_ready"])
        self.assertIn("B must be finite with shape", report["candidates"][0]["errors"][0])

    def test_python_top_level_side_effects_and_imports_are_never_run(self):
        marker = self.root / "must-not-exist"
        spec = example("nonlinear")["nominal"]
        source = ("from pathlib import Path\nimport nonexistent_unsafe_module\n"
                  f"Path({str(marker)!r}).write_text('executed')\n"
                  f"ENV_SPEC = {spec!r}\n"
                  "TOKEN = 'private-value'\n"
                  "class Plant:\n    def step(self, x, u, d=None):\n        return x\n")
        report = inspect_model_source(self.source("plant.py", source))
        self.assertFalse(marker.exists())
        self.assertTrue(report["registration_ready"])
        self.assertEqual(report["editable_model_spec"], spec)
        self.assertEqual(report["functions"][0]["name"], "Plant.step")
        self.assertEqual(report["functions"][0]["arguments"], ["self", "x", "u", "d"])
        self.assertNotIn("private-value", json.dumps(report))

    def test_dynamic_manifest_is_uncertain_without_keyword_inference(self):
        report = inspect_model_source(self.source("dynamic.py",
            "MANIFEST = build_spec()\nNX = 12\ndef step(state, control):\n    return state + control\n"))
        self.assertFalse(report["registration_ready"])
        self.assertEqual(report["candidates"], [])
        self.assertTrue(any(item["field"] == "MANIFEST" for item in report["uncertain"]))
        self.assertGreater(len(report["missing"]), 0)
        self.assertNotIn("dimensions", report)

    def test_multiple_valid_manifests_require_selection(self):
        first, second = example("linear")["nominal"], example("nonlinear")["nominal"]
        report = inspect_model_source(self.source("multiple.py", f"MODEL_SPEC={first!r}\nENV_SPEC={second!r}\n"))
        self.assertFalse(report["registration_ready"])
        self.assertNotIn("editable_model_spec", report)
        self.assertTrue(any(item["field"] == "model_selection" for item in report["uncertain"]))

    def test_slx_ports_are_static_evidence_not_resolved_interface_or_license(self):
        path = self.root / "plant.slx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("simulink/systems/system_root.xml", '<System><Block Name="heater" BlockType="Inport" SID="1"><P Name="Port">1</P></Block><Block Name="temp" BlockType="Outport" SID="2"/></System>')
        report = inspect_model_source(path)
        self.assertFalse(report["registration_ready"])
        self.assertEqual(len(report["simulink_structure"]["declared_port_blocks"]), 2)
        self.assertFalse(report["runtime"]["matlab_started"])
        self.assertEqual(report["runtime"]["license_status"], "not_checked")
        self.assertIsNone(report["runtime"]["native_execution_supported"])
        self.assertEqual(report["runtime"]["scope"], "static_material_inspection")
        self.assertEqual(report["runtime"]["runtime_verification"], "dedicated_matlab_session_required")
        self.assertEqual(report["runtime"]["blockers"], [])
        self.assertFalse(report["simulink_structure"]["compiled"])

    def test_slx_entity_xml_and_bogus_archive_are_reported(self):
        path = self.root / "bad.slx"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("simulink/blockdiagram.xml", '<!DOCTYPE System [<!ENTITY a "a">]><System/>')
        report = inspect_model_source(path)
        self.assertTrue(report["errors"])
        self.assertFalse(report["registration_ready"])
        path.write_bytes(b"not a zip")
        report = inspect_model_source(path)
        self.assertTrue(report["errors"])

    def test_runtime_discovery_does_not_claim_license_or_execute(self):
        executable = self.source("matlab.exe", "not an executable")
        with patch.dict(os.environ, {"SIMULINK_MPC_MATLAB_COMMAND": str(executable)}):
            report = matlab_runtime_status()
        self.assertTrue(report["matlab_executable_found"])
        self.assertFalse(report["matlab_started"])
        self.assertEqual(report["license_status"], "not_checked")

    def test_mdl_and_invalid_python_keep_missing_information(self):
        mdl = inspect_model_source(self.source("plant.mdl", 'Model { Name "plant" }'))
        self.assertFalse(mdl["registration_ready"])
        self.assertTrue(mdl["missing"])
        syntax = inspect_model_source(self.source("broken.py", "def foo(:\n"))
        self.assertTrue(syntax["errors"])
        self.assertFalse(syntax["registration_ready"])

    def test_cancellation_and_unsupported_format(self):
        path = self.source("model.json", "{}")
        with self.assertRaises(Cancelled):
            inspect_model_source(path, OperationContext(cancelled=lambda: True))
        with self.assertRaises(ContractError):
            inspect_model_source(self.source("model.bin", "binary"))


if __name__ == "__main__":
    unittest.main()
