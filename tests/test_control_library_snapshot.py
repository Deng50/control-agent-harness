"""Regression checks for maintaining the vendored engineering/evaluation split."""
from pathlib import Path
import runpy
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SYNC = runpy.run_path(str(ROOT / "scripts/sync-control-library.py"))
SCRATCH = ROOT / "test-output"


def snapshot(payloads):
    manifest = {
        "schema_version": 1,
        "source": "control-skill-library",
        "files": {name: SYNC["record"](data) for name, data in payloads.items()},
    }
    return {**payloads, "snapshot.json": SYNC["json_bytes"](manifest)}


class ControlLibrarySnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(exist_ok=True)

    def test_update_is_idempotent_and_drift_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            root = Path(folder) / "snapshot"
            first = snapshot({"cases/a.m": b"a\r\n", "runs/b.mat": b"frozen"})
            second = snapshot({"cases/a.m": b"b\n"})
            SYNC["reconcile"](SYNC["plan"](root, first), check=False)
            self.assertFalse(SYNC["plan"](root, first).writes)
            SYNC["reconcile"](SYNC["plan"](root, second), check=False)
            self.assertFalse((root / "runs/b.mat").exists())
            SYNC["reconcile"](SYNC["plan"](root, None), check=True)
            (root / "cases/a.m").write_bytes(b"local change")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                SYNC["plan"](root, second)
            self.assertEqual((root / "cases/a.m").read_bytes(), b"local change")

    def test_rejects_unknown_files_and_out_of_tree_manifest_member(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            root = Path(folder)
            (root / "extra.md").touch()
            with self.assertRaisesRegex(ValueError, "Unlisted"):
                SYNC["plan"](root, snapshot({"a.md": b"expected"}))
            (root / "snapshot.json").write_bytes(snapshot({"../x": b"x"})["snapshot.json"])
            with self.assertRaisesRegex(ValueError, "Invalid asset path"):
                SYNC["plan"](root, None)

    def test_shipping_contents(self):
        bundle = SYNC["load_manifest"](SYNC["LIBRARY"])
        offline = SYNC["load_manifest"](SYNC["EVALUATION"])
        self.assertTrue(all(not p.startswith(SYNC["OFFLINE"]) for p in bundle["files"]))
        self.assertTrue(all(p.startswith(SYNC["OFFLINE"]) for p in offline["files"]))
        self.assertIn(".agents/skills/verify-mpc-prediction/scripts/verify_increment_matrix.m", bundle["files"])
        self.assertIn("runs/paired/library-snapshot.zip", offline["files"])
        for relative in bundle["local_study_only"]:
            self.assertFalse((SYNC["LIBRARY"] / relative).exists())


if __name__ == "__main__":
    unittest.main()
