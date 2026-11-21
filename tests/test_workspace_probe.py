"""用真实临时目录检查诊断和清理契约；模拟替换失败无需更改机器的 ACL。"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from control_harness.workspace_probe import inspect_workspace


class WorkspaceProbeTests(TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "test-output"
        root.mkdir(exist_ok=True)
        temporary = TemporaryDirectory(dir=root)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_read_only_and_direct_worker(self):
        report = inspect_workspace(self.directory, inspect_only=True)
        self.assertTrue(report["ok"], report)
        self.assertEqual(Path(report["workspace"]), self.directory)
        self.assertEqual(list(self.directory.iterdir()), [])
        with patch.dict(os.environ, {key: str(self.directory) for key in ("TEMP", "TMP", "TMPDIR")}):
            self.assertTrue(inspect_workspace(self.directory)["ok"])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_replace_preserves_system_error_and_removes_only_probe(self):
        existing = self.directory / "keep"
        existing.write_text("user data", encoding="utf-8")
        with patch.object(os, "replace", side_effect=PermissionError(13, "locked")):
            report = inspect_workspace(self.directory)
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"]["operation"], "atomic_replace")
        self.assertEqual(report["error"]["os_code"], 13)
        self.assertEqual(list(self.directory.iterdir()), [existing])
        self.assertEqual(existing.read_text(encoding="utf-8"), "user data")

    def test_read_only_missing_workspace_reports_failing_path(self):
        missing = self.directory / "missing"
        report = inspect_workspace(missing, inspect_only=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["error"]["operation"], "stat_directory")
        self.assertEqual(Path(report["error"]["path"]), missing)
        self.assertFalse(missing.exists())
