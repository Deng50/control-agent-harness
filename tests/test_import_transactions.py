"""Artifact import fault tests use real registered models and simulated transitions."""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from control_harness import artifacts
from control_harness.artifacts import Workspace, atomic_json, canonical_hash, file_hash, read_json
from control_harness.contracts import ContractError
from control_harness.data import simulate_transitions
from control_harness.examples import example
from control_harness.modeling import register_model
from control_harness.runtime import OperationContext


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "import-transactions"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


class ImportTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.root = Path(self.temp.name)
        self.source = Workspace(self.root / "source")
        self.target = Workspace(self.root / "target")
        specs = example("linear")
        self.nominal = register_model(self.source, specs["nominal"], {"kind": "user_nominal", "description": "Import test nominal"})
        self.reference = register_model(self.source, specs["reference"], {"kind": "reference_simulator", "description": "Import test reference"})
        self.dataset = simulate_transitions(self.source, self.nominal, self.reference,
            {"trajectories": 5, "steps": 6, "seed": 37}, OperationContext())
        self.ids = [self.nominal, self.reference, self.dataset]
        self.source_path, _ = self.source.resolve(self.dataset)
        self.before = self.target.manifest

    def tearDown(self):
        self.temp.cleanup()

    def published_ids(self):
        root = self.target.path / "artifacts"
        return {item.name for item in root.iterdir()} if root.exists() else set()

    def diagnoses(self):
        return [read_json(path) for path in (self.target.path / "failed").glob("import-*/failure.json")]

    def assert_retry_succeeds(self):
        self.assertEqual(self.target.import_artifact(self.source_path), self.dataset)
        self.assertEqual(set(self.target.manifest["artifacts"]), set(self.ids))
        self.assertEqual(self.target.active("transition_dataset"), self.dataset)
        self.target.resolve(self.dataset)

    def orphan(self, artifact_id):
        root, _ = self.source.resolve(artifact_id)
        destination = self.target.path / "artifacts" / artifact_id
        shutil.copytree(root, destination)
        return destination

    def directory_link(self, link, target):
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            link.symlink_to(target, target_is_directory=True)

    def test_all_copies_precede_publication_and_only_one_manifest_commit_occurs(self):
        original_copy, original_json = shutil.copy2, atomic_json
        commits = []

        def checked_copy(src, dst, *args, **kwargs):
            self.assertEqual(self.target.manifest, self.before)
            self.assertEqual(self.published_ids(), set())
            return original_copy(src, dst, *args, **kwargs)

        def checked_json(path, value):
            if path == self.target.manifest_path:
                commits.append(copy.deepcopy(value))
                self.assertEqual(self.published_ids(), set(self.ids))
            return original_json(path, value)

        with patch.object(artifacts.shutil, "copy2", side_effect=checked_copy), patch.object(artifacts, "atomic_json", side_effect=checked_json):
            self.assert_retry_succeeds()
        self.assertEqual(len(commits), 1)

    def test_copy_failure_never_publishes_partial_closure_and_retry_succeeds(self):
        original, calls = shutil.copy2, 0

        def failing_copy(src, dst, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("injected copy I/O failure")
            return original(src, dst, *args, **kwargs)

        with patch.object(artifacts.shutil, "copy2", side_effect=failing_copy), self.assertRaisesRegex(OSError, "copy I/O"):
            self.target.import_artifact(self.source_path)
        self.assertEqual(self.target.manifest, self.before)
        self.assertEqual(self.published_ids(), set())
        self.assertEqual(self.diagnoses()[0]["published"], [])
        self.assert_retry_succeeds()

    def test_second_directory_publish_failure_withdraws_only_new_directories(self):
        original, calls = Path.rename, 0

        def failing_rename(path, destination):
            nonlocal calls
            if Path(destination).parent == self.target.path / "artifacts":
                calls += 1
                if calls == 2:
                    raise OSError("injected second publication failure")
            return original(path, destination)

        with patch.object(Path, "rename", failing_rename), self.assertRaisesRegex(OSError, "second publication"):
            self.target.import_artifact(self.source_path)
        self.assertEqual(self.target.manifest, self.before)
        self.assertEqual(self.published_ids(), set())
        self.assertEqual(len(self.diagnoses()[0]["published"]), 1)
        self.assert_retry_succeeds()

    def test_manifest_failure_rolls_back_closure_and_preserves_adopted_orphan(self):
        existing = self.orphan(self.nominal)
        original_bytes = (existing / "artifact.json").read_bytes()
        original = atomic_json

        def fail_manifest(path, value):
            if path == self.target.manifest_path:
                raise OSError("injected manifest replacement failure")
            return original(path, value)

        with patch.object(artifacts, "atomic_json", side_effect=fail_manifest), self.assertRaisesRegex(OSError, "manifest replacement"):
            self.target.import_artifact(self.source_path)
        self.assertEqual(self.target.manifest, self.before)
        self.assertEqual(self.published_ids(), {self.nominal})
        self.assertEqual((existing / "artifact.json").read_bytes(), original_bytes)
        self.assertFalse(self.diagnoses()[0]["manifest_committed"])
        self.assert_retry_succeeds()

    def test_failed_withdrawal_leaves_only_verified_orphan_that_retry_can_adopt(self):
        original, published = Path.rename, 0

        def failing_rename(path, destination):
            nonlocal published
            if Path(destination).parent == self.target.path / "artifacts":
                published += 1
                if published == 2:
                    raise OSError("injected publication failure")
            if path.parent == self.target.path / "artifacts":
                raise OSError("injected withdrawal failure")
            return original(path, destination)

        with patch.object(Path, "rename", failing_rename), self.assertRaisesRegex(RuntimeError, "inspect"):
            self.target.import_artifact(self.source_path)
        self.assertEqual(self.target.manifest, self.before)
        self.assertEqual(self.published_ids(), {self.nominal})
        self.assertIn("withdrawal failure", self.diagnoses()[0]["cleanup_errors"][0])
        self.assert_retry_succeeds()

    def test_error_reported_after_manifest_commit_never_withdraws_indexed_files(self):
        original = atomic_json

        def fail_after_commit(path, value):
            original(path, value)
            if path == self.target.manifest_path:
                raise OSError("injected post-commit notification failure")

        with patch.object(artifacts, "atomic_json", side_effect=fail_after_commit), self.assertRaisesRegex(OSError, "post-commit"):
            self.target.import_artifact(self.source_path)
        self.assertTrue(self.diagnoses()[0]["manifest_committed"])
        self.target.resolve(self.dataset)
        self.assert_retry_succeeds()

    def test_matching_hard_crash_orphans_are_adopted_without_copy_or_overwrite(self):
        paths = {item_id: self.orphan(item_id) for item_id in self.ids}
        before = {item_id: (root / "artifact.json").read_bytes() for item_id, root in paths.items()}
        with patch.object(artifacts.shutil, "copy2", side_effect=AssertionError("matching orphans must not be copied")):
            self.assert_retry_succeeds()
        self.assertEqual(before, {item_id: (root / "artifact.json").read_bytes() for item_id, root in paths.items()})

    def test_conflicting_orphan_hash_metadata_and_inventory_are_preserved_and_rejected(self):
        mutations = {
            "hash": lambda root: (root / "model.json").write_text("{}", encoding="utf-8"),
            "metadata": lambda root: atomic_json(root / "artifact.json", {**read_json(root / "artifact.json"), "source": {"changed": True}}),
            "extra": lambda root: (root / "unlisted.txt").write_text("keep me", encoding="utf-8"),
            "missing": lambda root: (root / "model.json").unlink(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                self.target = Workspace(self.root / f"target-{name}")
                before = self.target.manifest
                orphan = self.orphan(self.nominal)
                mutate(orphan)
                contents = {p.relative_to(orphan).as_posix(): p.read_bytes() for p in orphan.rglob("*") if p.is_file()}
                with self.assertRaises(ContractError):
                    self.target.import_artifact(self.source_path)
                self.assertEqual(self.target.manifest, before)
                self.assertEqual(self.published_ids(), {self.nominal})
                self.assertEqual(contents, {p.relative_to(orphan).as_posix(): p.read_bytes() for p in orphan.rglob("*") if p.is_file()})
                self.assertEqual(len(self.diagnoses()), 1)

    def test_source_mutation_after_copy_is_detected_before_publication(self):
        original, changed = shutil.copy2, False

        def changed_copy(src, dst, *args, **kwargs):
            nonlocal changed
            result = original(src, dst, *args, **kwargs)
            if not changed and Path(src).name == "model.json":
                changed = True
                Path(src).write_text("{}", encoding="utf-8")
            return result

        with patch.object(artifacts.shutil, "copy2", side_effect=changed_copy), self.assertRaisesRegex(ContractError, "missing or changed"):
            self.target.import_artifact(self.source_path)
        self.assertEqual(self.target.manifest, self.before)
        self.assertEqual(self.published_ids(), set())

    def test_cyclic_source_dependencies_are_rejected(self):
        root, meta = self.source.resolve(self.nominal)
        meta["parents"] = [{"id": self.nominal, "metadata_hash": "cyclic"}]
        atomic_json(root / "artifact.json", meta)
        manifest = self.source.manifest
        manifest["artifacts"][self.nominal]["metadata_hash"] = canonical_hash(meta)
        atomic_json(self.source.manifest_path, manifest)
        with self.assertRaisesRegex(ContractError, "Cyclic"):
            self.target.import_artifact(root)
        self.assertEqual(self.target.manifest, self.before)

    def test_source_file_escape_is_rejected_before_any_target_publication(self):
        root, meta = self.source.resolve(self.nominal)
        outside = root.parent / "outside.txt"
        outside.write_text("must not copy", encoding="utf-8")
        meta["files"]["../outside.txt"] = file_hash(outside)
        atomic_json(root / "artifact.json", meta)
        manifest = self.source.manifest
        manifest["artifacts"][self.nominal]["metadata_hash"] = canonical_hash(meta)
        atomic_json(self.source.manifest_path, manifest)
        with self.assertRaisesRegex(ContractError, "escapes artifact root"):
            self.target.import_artifact(root)
        self.assertEqual(self.target.manifest, self.before)
        self.assertEqual(self.published_ids(), set())

    def test_target_artifacts_directory_cannot_escape_workspace(self):
        outside = self.root / "outside-target"
        outside.mkdir()
        self.directory_link(self.target.path / "artifacts", outside)
        try:
            with self.assertRaisesRegex(ContractError, "escapes artifact root"):
                self.target.import_artifact(self.source_path)
            self.assertEqual(self.target.manifest, self.before)
            self.assertEqual(list(outside.iterdir()), [])
        finally:
            (self.target.path / "artifacts").rmdir() if os.name == "nt" else (self.target.path / "artifacts").unlink()

    def test_existing_artifact_alias_is_not_adopted_or_overwritten(self):
        real = self.target.path / "artifacts" / "existing-other-name"
        source, _ = self.source.resolve(self.nominal)
        shutil.copytree(source, real)
        link = real.parent / self.nominal
        self.directory_link(link, real)
        before = {p.relative_to(real).as_posix(): p.read_bytes() for p in real.rglob("*") if p.is_file()}
        try:
            with self.assertRaisesRegex(ContractError, "filesystem alias"):
                self.target.import_artifact(self.source_path)
            self.assertEqual(self.target.manifest, self.before)
            self.assertEqual(before, {p.relative_to(real).as_posix(): p.read_bytes() for p in real.rglob("*") if p.is_file()})
            self.assertEqual(len(self.diagnoses()), 1)
        finally:
            link.rmdir() if os.name == "nt" else link.unlink()


if __name__ == "__main__":
    unittest.main()
