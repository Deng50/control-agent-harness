"""Real worker hard exits, guarded archival, and numerical checkpoint continuation."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from control_harness.artifacts import Workspace, WorkspaceBusy, atomic_json, file_hash, read_json
from control_harness.checkpoints import HEAD, checkpoint_sidecar, current_checkpoint
from control_harness.contracts import ContractError, ModelInterface
from control_harness.data import simulate_transitions
from control_harness.locking import process_alive
from control_harness.modeling import register_model
from control_harness.recovery import recover_interrupted_stage
from control_harness.residual import train_residual


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "interrupted-recovery-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)

WORKER = r'''
import json, os, sys
from pathlib import Path
import torch
from control_harness.artifacts import Workspace, atomic_json
from control_harness.residual import train_residual
from control_harness.runtime import OperationContext
torch.set_num_threads(1)
root, mode = Path(sys.argv[1]), sys.argv[2]
request = json.loads((root / 'worker-request.json').read_text(encoding='utf-8'))
workspace = Workspace(root)
def emit(event):
    if event.get('epoch') == 2:
        stage_id = Path(event['checkpoint']).parent.name
        atomic_json(root / 'worker-ready.json', {'stage_id': stage_id, 'pid': os.getpid()})
        if mode == 'hold':
            sys.stdin.readline()
        os._exit(73)
train_residual(workspace, request['nominal'], request['dataset'], request['config'], OperationContext(emit=emit))
raise RuntimeError('Expected a hard exit before successful publication')
'''


class InterruptedStageRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        workspace = Workspace(cls.fixture.name)
        var = lambda name, bound: {"name": name, "unit": "1", "lower": -bound, "upper": bound}
        cls.spec = {"interface": {"schema_version": 1, "name": "hard-exit", "sample_time": .1,
                    "time_semantics": "discrete_state_transition", "states": [var("x", 3)],
                    "controls": [var("u", 1)], "outputs": [var("y", 3)], "disturbances": []},
                    "representation": "discrete_polynomial", "A": [[.7]], "B": [[.25]], "C": [[1]]}
        reference = {**copy.deepcopy(cls.spec), "A": [[.78]], "bias": [.02]}
        cls.nominal = register_model(workspace, cls.spec, {"kind": "user_nominal"})
        cls.reference = register_model(workspace, reference, {"kind": "reference_simulator"})
        cls.dataset = simulate_transitions(workspace, cls.nominal, cls.reference,
                                           {"trajectories": 6, "steps": 6, "seed": 12})
        cls.config = {"epochs": 5, "hidden_dims": [6], "batch_size": 5, "patience": 20,
                      "learning_rate": .01, "seed": 16, "physics_weight": .2}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        shutil.copytree(self.fixture.name, self.temp.name, dirs_exist_ok=True)
        self.workspace = Workspace(self.temp.name)
        self.children = []

    def tearDown(self):
        # Only processes created by this test are ever stopped.
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=15)
        self.temp.cleanup()

    def start_worker(self, mode="crash"):
        atomic_json(self.workspace.path / "worker-request.json",
                    {"nominal": self.nominal, "dataset": self.dataset, "config": self.config})
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        child = subprocess.Popen([sys.executable, "-c", WORKER, str(self.workspace.path), mode],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, env=env)
        self.children.append(child)
        ready = self.workspace.path / "worker-ready.json"
        deadline = time.monotonic() + 45
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        if not ready.exists():
            output, error = child.communicate(timeout=5)
            self.fail(f"Numerical worker did not commit a checkpoint: {output} {error}")
        record = read_json(ready)
        if mode == "crash":
            _, error = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 73, error)
            self.assertIs(process_alive(record["pid"]), False)
        return child, record

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_real_hard_exit_archives_verified_checkpoint_and_resumes_identically(self):
        full = train_residual(self.workspace, self.nominal, self.dataset, self.config)
        full_root = self.workspace.path / self.workspace.manifest["artifacts"][full]["path"]
        _, record = self.start_worker()
        stage_id = record["stage_id"]
        root = self.workspace.path / ".staging" / stage_id
        intent = read_json(root / "stage.intent.json")
        self.assertEqual(intent["owner"]["pid"], record["pid"])
        self.assertEqual(intent["owner"]["token"], read_json(self.workspace.lock_path)["token"])
        self.assertEqual(intent["algorithm"], "residual_training")
        self.assertEqual(intent["config"], self.config)
        self.assertEqual({row["id"] for row in intent["dependencies"]}, {self.nominal, self.dataset})
        self.assertEqual(intent["training"]["training_signature"], read_json(checkpoint_sidecar(current_checkpoint(root)))["training_signature"])
        manifest_before = self.workspace.manifest_path.read_bytes()
        result = recover_interrupted_stage(self.workspace, stage_id)
        self.assertEqual(result["status"], "archived")
        self.assertEqual(result["epoch"], 2)
        self.assertFalse(root.exists())
        self.assertEqual(self.workspace.manifest_path.read_bytes(), manifest_before)
        self.assertEqual(file_hash(Path(result["checkpoint"])), result["checkpoint_sha256"])
        receipt = read_json(Path(result["failed_path"]) / "interrupted-stage-recovery.json")
        self.assertEqual(receipt["owner_verification"], "os_confirmed_process_exit")
        self.assertEqual(receipt["original_owner"], intent["owner"])
        self.workspace.resolve(full)
        resumed = train_residual(self.workspace, self.nominal, self.dataset,
                                 {**self.config, "resume_checkpoint": result["checkpoint"],
                                  "resume_checkpoint_sha256": result["checkpoint_sha256"]})
        resumed_root = self.workspace.path / self.workspace.manifest["artifacts"][resumed]["path"]
        self.assert_nested_equal(torch.load(full_root / "latest.pt", weights_only=True),
                                 torch.load(resumed_root / "latest.pt", weights_only=True))
        self.assert_nested_equal(torch.load(full_root / "model.pt", weights_only=True),
                                 torch.load(resumed_root / "model.pt", weights_only=True))
        self.assertEqual(read_json(resumed_root / "report.json")["resumed_from_epoch"], 2)

    def test_live_worker_kernel_mutex_blocks_archive_then_exit_allows_it(self):
        child, record = self.start_worker("hold")
        root = self.workspace.path / ".staging" / record["stage_id"]
        original = file_hash(current_checkpoint(root))
        self.assertIs(process_alive(record["pid"]), True)
        with self.assertRaises(WorkspaceBusy):
            recover_interrupted_stage(self.workspace, record["stage_id"])
        self.assertEqual(file_hash(current_checkpoint(root)), original)
        self.assertTrue(root.is_dir())
        _, error = child.communicate("exit\n", timeout=15)
        self.assertEqual(child.returncode, 73, error)
        result = recover_interrupted_stage(self.workspace, record["stage_id"])
        self.assertTrue(Path(result["checkpoint"]).is_file())

    def test_unsafe_intents_corruption_dependencies_and_publication_are_refused(self):
        _, record = self.start_worker()
        stage_id = record["stage_id"]
        root = self.workspace.path / ".staging" / stage_id
        manifest_before = self.workspace.manifest_path.read_bytes()

        def refused(exception=ContractError):
            with self.assertRaises(exception):
                recover_interrupted_stage(self.workspace, stage_id)
            self.assertTrue(root.is_dir())
            self.assertFalse((self.workspace.path / "failed" / stage_id).exists())
            self.assertEqual(self.workspace.manifest_path.read_bytes(), manifest_before)

        intent_path = root / "stage.intent.json"
        original_intent = intent_path.read_bytes()
        for pid in (os.getpid(), None):
            intent = json.loads(original_intent)
            intent["owner"]["pid"] = pid
            atomic_json(intent_path, intent)
            refused(WorkspaceBusy)
        intent_path.write_bytes(original_intent)
        checkpoint = current_checkpoint(root)
        previous = root / (read_json(root / HEAD)["previous"] + ".pt")
        # 没有任何通过校验的代号时必须拒绝归档；只破坏当前代号的回退另测。
        previous_bytes = previous.read_bytes()
        previous.write_bytes(b"corrupt")
        for filename in ("config.json", HEAD, checkpoint_sidecar(checkpoint).name, checkpoint.name):
            with self.subTest(corruption=filename):
                path = root / filename
                original = path.read_bytes()
                path.write_bytes(b"corrupt")
                refused()
                path.write_bytes(original)
        previous.write_bytes(previous_bytes)
        marker = root / "artifact.json"
        marker.write_text("{}", encoding="utf-8")
        refused()
        marker.unlink()
        nominal_root, _ = self.workspace.resolve(self.nominal)
        model_path = nominal_root / "model.json"
        original_model = model_path.read_bytes()
        model_path.write_bytes(b"{}")
        refused()
        model_path.write_bytes(original_model)
        with self.assertRaisesRegex(ContractError, "exact unpublished"):
            recover_interrupted_stage(self.workspace, "../artifacts")
        result = recover_interrupted_stage(self.workspace, stage_id)
        self.assertEqual(result["epoch"], 2)

    def test_corrupt_current_generation_recovers_only_committed_verified_previous(self):
        _, record = self.start_worker()
        root = self.workspace.path / ".staging" / record["stage_id"]
        pointer = (root / HEAD).read_bytes()
        head = read_json(root / HEAD)
        previous = root / (head["previous"] + ".pt")
        previous_hash = file_hash(previous)
        damaged = current_checkpoint(root)
        damaged.write_bytes(b"damaged current")
        original_failure = {"type": "PermissionError", "message": "original write refused"}
        atomic_json(root / "failure.json", original_failure)
        with patch.object(Path, "rename", side_effect=PermissionError("archive rename refused")), \
                self.assertRaisesRegex(ContractError, "verified but not archived") as refused:
            recover_interrupted_stage(self.workspace, record["stage_id"])
        self.assertEqual(refused.exception.checkpoint_details["checkpoint"], str(previous))
        self.assertEqual(refused.exception.stage_failure["archive_state"], "retained")
        result = recover_interrupted_stage(self.workspace, record["stage_id"])
        self.assertEqual((result["epoch"], result["checkpoint_selection"]), (1, "previous"))
        self.assertIn("integrity", result["checkpoint_fallback_reason"])
        path = Path(result["checkpoint"])
        self.assertEqual(path.name, previous.name)
        self.assertEqual(file_hash(path), previous_hash)
        self.assertEqual((path.parent / HEAD).read_bytes(), pointer)
        self.assertEqual((path.parent / damaged.name).read_bytes(), b"damaged current")
        self.assertEqual(read_json(path.parent / "failure.json"), original_failure)
        resumed = train_residual(self.workspace, self.nominal, self.dataset, {**self.config,
                                 "resume_checkpoint": str(path), "resume_checkpoint_sha256": previous_hash})
        self.assertEqual(read_json(self.workspace.resolve(resumed, allow_rejected=True)[0] / "report.json")["resumed_from_epoch"], 1)

    def test_recovery_never_overwrites_an_existing_failed_destination(self):
        _, record = self.start_worker()
        destination = self.workspace.path / "failed" / record["stage_id"]
        destination.mkdir(parents=True)
        sentinel = destination / "keep.txt"
        sentinel.write_text("previous evidence", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "refusing overwrite"):
            recover_interrupted_stage(self.workspace, record["stage_id"])
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "previous evidence")
        self.assertTrue((self.workspace.path / ".staging" / record["stage_id"]).is_dir())

    def test_exception_after_manifest_commit_does_not_withdraw_a_published_stage(self):
        real_atomic = atomic_json
        def after_replace(path, value):
            real_atomic(path, value)
            if path == self.workspace.manifest_path:
                raise OSError("Injected failure after successful manifest replacement")
        stage_id = None
        with self.assertRaisesRegex(OSError, "after successful"), patch("control_harness.artifacts.atomic_json", after_replace):
            with self.workspace.stage("nominal_model", ModelInterface.parse(self.spec["interface"]), [], {}) as stage:
                stage_id = stage.id
                atomic_json(stage.root / "model.json", self.spec)
                stage.publish({"status": "verified"})
        root, _ = self.workspace.resolve(stage_id)
        self.assertEqual(root.parent, self.workspace.path / "artifacts")
        self.assertFalse((root / "failure.json").exists())
        self.assertFalse((self.workspace.path / "failed" / stage_id).exists())
        self.assertTrue((self.workspace.path / "recovery" / f"publication-{stage_id}.json").is_file())


if __name__ == "__main__":
    unittest.main()
