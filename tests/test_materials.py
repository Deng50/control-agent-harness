"""S03 byte snapshots and static MATLAB inspection; no uploaded code execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from control_harness import materials
from control_harness.contracts import ContractError
from control_harness.materials import inspect_material, register_material
from control_harness.model_inspection import inspect_model_source
from control_harness.runtime import Cancelled, OperationContext


OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", Path(__file__).resolve().parents[1] / "test-output/material-unit-temp"))
OUTPUT.mkdir(parents=True, exist_ok=True)


class MaterialTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.primary = self.inputs / "measurements.csv"
        self.primary.write_bytes(b"time,current,voltage\n0,1,3.7\n1,2,3.6\n")
        self.stage = self.root / "stage"

    def tearDown(self):
        self.temporary.cleanup()

    def directory_link(self, link: Path, target: Path):
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)

    def assert_unpublished(self):
        self.assertFalse((self.stage / "source").exists())
        self.assertEqual(list(self.stage.glob(".material-snapshot-*")), [])

    def test_path_and_native_upload_local_path_share_content_semantics(self):
        before = self.primary.read_bytes()
        inspected = inspect_material(self.primary)
        first = register_material(self.primary, self.stage, expected_manifest=inspected)
        upload = self.root / "native-upload-without-extension"
        upload.write_bytes(before)
        second = register_material(upload, self.root / "uploaded-stage", source_kind="upload",
                                   original_name="measurements.csv", source_metadata={"attachment_id": "fixture"})
        self.assertEqual(first["content_hash"], second["content_hash"])
        self.assertEqual(first["source"]["kind"], "path")
        self.assertEqual(second["source"]["kind"], "upload")
        self.assertEqual(second["source"]["local_path"], str(upload))
        self.assertEqual(second["source"]["metadata"], {"attachment_id": "fixture"})
        self.assertEqual(first["primary_file"], "source/measurements.csv")
        self.assertEqual((self.stage / first["primary_file"]).read_bytes(), before)
        self.assertEqual(self.primary.read_bytes(), before)
        self.assertFalse(first["source_executed"])
        self.assertFalse(first["dimensions_inferred"])
        self.assertNotIn("nx", first)

    def test_explicit_dependency_tree_and_empty_directory_are_preserved(self):
        dependency = self.inputs / "helpers"
        (dependency / "empty").mkdir(parents=True)
        (dependency / "coefficients.json").write_text('{"gain": 2}', encoding="utf-8")
        (dependency / "model-data.sldd").write_bytes(b"unchanged binary dependency")
        report = register_material(self.primary, self.stage, dependencies=["helpers", "helpers/coefficients.json"])
        self.assertEqual(report["file_count"], 3)
        self.assertTrue((self.stage / "source/helpers/empty").is_dir())
        for item in report["files"]:
            copied = self.stage / item["path"]
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), item["sha256"])
            self.assertEqual(copied.stat().st_size, item["size_bytes"])
        self.assertEqual(report["file_hashes"], {item["path"]: item["sha256"] for item in report["files"]})

    def test_supported_primary_formats_are_snapshotted_as_bytes_without_execution(self):
        marker = self.root / "must-not-execute"
        for suffix in ["csv", "xlsx", "py", "m", "slx", "mdl", "json"]:
            with self.subTest(suffix=suffix):
                path = self.inputs / ("material." + suffix)
                payload = (f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n").encode()
                path.write_bytes(payload)
                inspected = inspect_material(path)
                report = register_material(path, self.root / ("stage-" + suffix))
                self.assertEqual(report["format"], suffix)
                self.assertEqual(inspected["format_detection"], "filename_extension_only")
                self.assertEqual((self.root / ("stage-" + suffix) / report["primary_file"]).read_bytes(), payload)
                self.assertFalse(marker.exists())

    def test_dependency_escape_and_traversal_are_refused(self):
        outside = self.root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        for dependency in [outside, "../outside.json"]:
            with self.subTest(dependency=dependency), self.assertRaises(ContractError):
                register_material(self.primary, self.stage, dependencies=[dependency])
        self.assertFalse(self.stage.exists())
        with self.assertRaises(ContractError):
            inspect_material(outside, dependency_root=self.inputs)

    def test_names_cannot_traverse_create_streams_or_alias_windows_reserved_names(self):
        for name in ["../x.csv", "..\\x.csv", "/x.csv", "C:\\x.csv", "x.csv:secret", "CON.csv", "x.csv ", "x.csv."]:
            with self.subTest(name=name), self.assertRaisesRegex(ContractError, "basename"):
                inspect_material(self.primary, original_name=name)
        with self.assertRaises(ContractError):
            inspect_material(self.primary, original_name="unsupported.exe")
        with self.assertRaises(ContractError):
            inspect_material(Path("relative.csv"))

    def test_source_and_dependency_directory_links_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "data.csv").write_text("u,y\n1,2\n", encoding="utf-8")
        link = self.inputs / "linked"
        self.directory_link(link, outside)
        with self.assertRaisesRegex(ContractError, "reparse|symlink"):
            inspect_material(link / "data.csv")
        with self.assertRaisesRegex(ContractError, "reparse|symlink"):
            register_material(self.primary, self.stage, dependencies=["linked"])
        self.assertTrue((outside / "data.csv").exists())
        self.assert_unpublished()

    def test_destination_directory_links_do_not_receive_writes(self):
        outside = self.root / "outside-target"
        outside.mkdir()
        self.directory_link(self.stage, outside)
        with self.assertRaisesRegex(ContractError, "reparse|symlink"):
            register_material(self.primary, self.stage)
        self.assertEqual(list(outside.iterdir()), [])

    def test_existing_stage_config_preserved_and_source_never_overwritten(self):
        self.stage.mkdir()
        (self.stage / "config.json").write_text("fixture-config", encoding="utf-8")
        register_material(self.primary, self.stage)
        before = (self.stage / "source/measurements.csv").read_bytes()
        with self.assertRaisesRegex(ContractError, "refusing overwrite"):
            register_material(self.primary, self.stage)
        self.assertEqual((self.stage / "config.json").read_text(encoding="utf-8"), "fixture-config")
        self.assertEqual((self.stage / "source/measurements.csv").read_bytes(), before)

    def test_destination_inside_selected_tree_is_rejected_before_copy(self):
        with self.assertRaisesRegex(ContractError, "inside a selected"):
            register_material(self.primary, self.inputs / "snapshot", dependencies=["."])
        self.assertFalse((self.inputs / "snapshot").exists())

    def test_change_between_inspect_and_copy_is_rejected_and_temp_cleaned(self):
        def change(event):
            if event["stage"] == "material_snapshot_started":
                self.primary.write_bytes(b"modified after inspection")
        with self.assertRaisesRegex(ContractError, "changed"):
            register_material(self.primary, self.stage, context=OperationContext(emit=change))
        self.assert_unpublished()

    def test_dependency_added_during_copy_invalidates_snapshot(self):
        dep = self.inputs / "dependencies"
        dep.mkdir()
        (dep / "initial.json").write_text("{}", encoding="utf-8")
        def change(event):
            if event["stage"] == "material_snapshot_copied":
                (dep / "new.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "changed"):
            register_material(self.primary, self.stage, dependencies=["dependencies"],
                              context=OperationContext(emit=change))
        self.assert_unpublished()

    def test_prior_inspection_detects_metadata_only_file_change(self):
        observed = inspect_material(self.primary)
        info = self.primary.stat()
        os.utime(self.primary, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
        self.assertEqual(inspect_material(self.primary)["content_hash"], observed["content_hash"])
        with self.assertRaisesRegex(ContractError, "changed since"):
            register_material(self.primary, self.stage, expected_manifest=observed)
        self.assertFalse(self.stage.exists())

    def test_file_count_size_and_depth_limits_are_enforced(self):
        with patch.object(materials, "MAX_FILE_BYTES", 3), self.assertRaisesRegex(ContractError, "per-file"):
            inspect_material(self.primary)
        with patch.object(materials, "MAX_TOTAL_BYTES", 3), self.assertRaisesRegex(ContractError, "total-byte"):
            inspect_material(self.primary)
        extra = self.inputs / "extra.json"
        extra.write_text("{}", encoding="utf-8")
        with patch.object(materials, "MAX_FILES", 1), self.assertRaisesRegex(ContractError, "file-count"):
            inspect_material(self.primary, dependencies=["extra.json"])
        deep = self.inputs / "a/b"
        deep.mkdir(parents=True)
        with patch.object(materials, "MAX_DEPTH", 1), self.assertRaisesRegex(ContractError, "depth"):
            inspect_material(self.primary, dependencies=["a"])

    def test_cancel_during_snapshot_cleans_only_its_temporary_tree(self):
        stop = [False]
        def cancel(event):
            if event["stage"] == "material_snapshot_started":
                stop[0] = True
        with self.assertRaises(Cancelled):
            register_material(self.primary, self.stage,
                              context=OperationContext(emit=cancel, cancelled=lambda: stop[0]))
        self.assert_unpublished()
        self.assertTrue(self.primary.exists())

    def test_logical_upload_filename_dependency_collision_is_rejected(self):
        (self.inputs / "renamed.csv").write_bytes(b"different dependency")
        with self.assertRaisesRegex(ContractError, "collide"):
            inspect_material(self.primary, original_name="renamed.csv", dependencies=["renamed.csv"])
        (self.inputs / "directory.csv").mkdir()
        with self.assertRaisesRegex(ContractError, "collides"):
            inspect_material(self.primary, original_name="directory.csv", dependencies=["directory.csv"])

    def test_source_metadata_must_remain_finite_json(self):
        with self.assertRaises(ContractError):
            register_material(self.primary, self.stage, source_kind="simulator")
        with self.assertRaises(ContractError):
            register_material(self.primary, self.stage, source_metadata={"bad": float("nan")})
        with self.assertRaises(ContractError):
            inspect_material(self.primary, dependencies=[None])
        with self.assertRaises(ContractError):
            inspect_material(None)
        self.assertFalse(self.stage.exists())


class MatlabMaterialInspectionTests(unittest.TestCase):
    def test_matlab_headers_and_numeric_literals_are_candidates_never_dimensions(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temporary:
            root = Path(temporary)
            source = root / "plant.m"
            source.write_text("% Declared oscillator, review mappings before execution\n"
                "% API_KEY = 'private-header-value'\n"
                "function [next_x, y] = plant(x, u, ...\n d)\n"
                "MASS = 1; % literal candidate\nA = [0 1; -2 -.8];\n"
                "DYNAMIC = build_matrix();\nSECRET = 'do-not-display';\n"
                "%{\nfunction fake = commented(x)\n%}\n"
                "system('must-not-execute');\nend\n", encoding="utf-8")
            report = inspect_model_source(source)
            structure = report["matlab_structure"]
            self.assertEqual(structure["functions"][0]["name"], "plant")
            self.assertEqual(structure["functions"][0]["arguments"], ["x", "u", "d"])
            self.assertEqual(structure["functions"][0]["outputs"], ["next_x", "y"])
            self.assertEqual(len(structure["functions"]), 1)
            values = {v["name"]: v["value"] for v in structure["numeric_literals"]}
            self.assertEqual(values["MASS"], 1)
            self.assertEqual(values["A"], [[0, 1], [-2, -.8]])
            self.assertNotIn("DYNAMIC", values)
            self.assertNotIn("do-not-display", json.dumps(report))
            self.assertNotIn("private-header-value", json.dumps(report))
            self.assertFalse(report["source_executed"])
            self.assertFalse(report["registration_ready"])
            self.assertFalse(structure["compiled"])
            self.assertNotIn("dimensions", report)
            self.assertEqual(report["runtime"]["required_products"], ["MATLAB"])

    def test_matlab_invalid_encoding_is_explicit_and_not_executed(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as temporary:
            path = Path(temporary) / "encoding.m"
            path.write_bytes(b"\xff\xfefunction plant")
            report = inspect_model_source(path)
            self.assertTrue(report["errors"])
            self.assertFalse(report["source_executed"])
            self.assertFalse(report["registration_ready"])


if __name__ == "__main__":
    unittest.main()
