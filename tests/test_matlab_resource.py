"""Kernel-backed resource semantics; no MATLAB/Engine or numerical fixture is simulated as MATLAB."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from control_harness.matlab_resource import (MatlabResource, MatlabResourceError,
    MatlabResourceQuarantined, MatlabResourceWaitTimeout, process_identity)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "matlab-resource-tests"))
OUTPUT.mkdir(parents=True, exist_ok=True)
PROOF = {"kind": "synchronous_adapter_return", "producer": "unittest-trusted-synchronous-fixture"}


class MatlabResourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.path = Path(self.temporary.name)
        self.resource = MatlabResource(self.path, .01)
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
        self.temporary.cleanup()

    def spawn(self, source, *args):
        child = subprocess.Popen([sys.executable, "-u", "-c", source, *map(str, args)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        self.children.append(child)
        return child

    def test_reserved_cancel_and_explicit_synchronous_completion_release_without_quarantine(self):
        lease = self.resource.acquire("before-dispatch")
        lease.cancel_before_dispatch()
        with self.resource.acquire("trusted-sync") as lease:
            lease.mark_running()
            lease.complete(PROOF)
        state = self.resource.snapshot()
        self.assertIsNone(state["owner"])
        self.assertIsNone(state["quarantine"])
        self.assertEqual(state["queue"], [])
        self.assertEqual(state["last_completion"]["evidence"], PROOF)

    def test_exception_after_dispatch_quarantines_and_no_later_caller_is_admitted(self):
        with self.assertRaisesRegex(RuntimeError, "adapter failed"):
            with self.resource.acquire("raises") as lease:
                lease.mark_running()
                raise RuntimeError("adapter failed")
        state = self.resource.snapshot()
        self.assertEqual(state["quarantine"]["reason"], "adapter_left_without_verified_completion")
        with self.assertRaises(MatlabResourceQuarantined):
            self.resource.acquire("must-not-run")

    def test_promise_settlement_or_model_idle_flag_is_not_completion_evidence(self):
        with self.resource.acquire("untrusted-proof") as lease:
            lease.mark_running()
            with self.assertRaisesRegex(MatlabResourceError, "trusted synchronous-adapter"):
                lease.complete({"idle": True, "promiseSettled": True})
        self.assertIsNotNone(self.resource.snapshot()["quarantine"])

    def test_queue_deadline_removes_only_its_ticket_and_cannot_release_the_owner(self):
        owner = self.resource.acquire("holder")
        try:
            with self.assertRaises(MatlabResourceWaitTimeout):
                self.resource.acquire("deadline", timeout=.035)
            state = self.resource.snapshot()
            self.assertEqual(state["owner"]["token"], owner.owner["token"])
            self.assertEqual(state["queue"], [])
            self.assertIsNone(state["quarantine"])
        finally:
            owner.cancel_before_dispatch()

    def test_process_death_requires_matching_token_and_bound_session_os_exit_before_recovery(self):
        # This controlled sleeper is an explicit OS-liveness fixture, not MATLAB.
        session = self.spawn("import sys; print('ready', flush=True); sys.stdin.readline()")
        self.assertEqual(session.stdout.readline().strip(), "ready")
        binding = self.resource.bind_session(session.pid, "dedicated-session-fixture")
        crash = self.spawn(r'''
import os, sys
from control_harness.matlab_resource import MatlabResource
lease = MatlabResource(sys.argv[1], .01).acquire('crashing-host-fixture')
lease.mark_running()
print(lease.owner['token'], flush=True)
os._exit(73)
''', self.path)
        token = crash.stdout.readline().strip()
        _, error = crash.communicate(timeout=10)
        self.assertEqual(crash.returncode, 73, error)
        state = self.resource.snapshot()
        self.assertEqual(state["quarantine"]["token"], token)
        self.assertEqual(state["quarantine"]["reason"], "owner_process_exited")
        with self.assertRaisesRegex(MatlabResourceError, "token mismatch"):
            self.resource.recover_after_session_exit("wrong-token")
        with self.assertRaisesRegex(MatlabResourceError, "not been proven exited"):
            self.resource.recover_after_session_exit(token)
        with self.assertRaises(MatlabResourceQuarantined):
            self.resource.acquire("new-host-must-not-assume-idle")
        session.stdin.write("exit\n")
        session.stdin.flush()
        session.communicate(timeout=10)
        proof = self.resource.recover_after_session_exit(token)
        self.assertEqual(proof["kind"], "os_verified_bound_session_exit")
        self.assertEqual(proof["session"], binding)
        self.assertTrue((self.path / "recoveries" / f"{token}.json").is_file())
        with self.resource.acquire("after-verified-recovery") as lease:
            lease.mark_running()
            lease.complete(PROOF)
        self.assertIsNone(self.resource.snapshot()["quarantine"])

    def test_unbound_quarantine_cannot_be_reset_by_a_token_alone(self):
        with self.resource.acquire("unknown-session") as lease:
            lease.mark_running()
        token = self.resource.snapshot()["quarantine"]["token"]
        with self.assertRaisesRegex(MatlabResourceError, "No dedicated MATLAB process"):
            self.resource.recover_after_session_exit(token)

    def quarantined_process_fixture(self):
        self.resource.bind_session(os.getpid(), "legacy-non-uuid-session")
        with self.resource.acquire("host-fixture") as lease:
            lease.mark_running()
        return self.resource.snapshot()

    def test_invalid_or_conflicting_bindings_never_prove_exit(self):
        state = self.quarantined_process_fixture()
        token = state["quarantine"]["token"]
        good = state["session"]
        for changes in ({"identity": None}, {"identity": ""}, {"identity": "invalid"},
                        {"identity": "win32:0000000000000000"}, {"pid": True}, {"pid": 2**32},
                        {"host": "foreign-host.invalid"}, {"session_id": "  "},
                        {"session_id": "valid-but-different"}):
            with self.subTest(changes=changes):
                invalid = {**good, **changes}
                if invalid["identity"] is None:
                    del invalid["identity"]
                state["quarantine"]["owner"]["session"] = invalid
                state["session"] = good if changes.get("session_id") == "valid-but-different" else invalid
                self.resource.state_path.write_text(json.dumps(state), encoding="utf-8")
                # 即使操作系统报告 PID 不存在，也必须先有有效的、属于本机的完整绑定。
                with patch("control_harness.matlab_resource.process_alive", return_value=False) as check:
                    with self.assertRaises(MatlabResourceError):
                        self.resource.recover_after_session_exit(token)
                    check.assert_not_called()
                self.assertEqual(self.resource.snapshot()["quarantine"]["token"], token)
                self.assertFalse((self.path / "recoveries").exists())

    def test_unreadable_os_identity_does_not_release_quarantine(self):
        state = self.quarantined_process_fixture()
        token = state["quarantine"]["token"]
        with patch("control_harness.matlab_resource.process_alive", return_value=None), \
                patch("control_harness.matlab_resource.process_identity", return_value=None):
            with self.assertRaisesRegex(MatlabResourceError, "not been proven exited"):
                self.resource.recover_after_session_exit(token)
        self.assertEqual(self.resource.snapshot()["quarantine"]["token"], token)
        self.assertFalse((self.path / "recoveries").exists())

    def test_reused_pid_is_proof_of_originating_process_exit(self):
        state = self.quarantined_process_fixture()
        token = state["quarantine"]["token"]
        # 合成旧绑定，当前 PID 来自测试进程；实际核验读取 OS 提供的创建身份。
        prefix, value = state["session"]["identity"].rsplit(":", 1)
        previous = f"{int(value, 16) - 1:016x}" if os.name == "nt" else str(max(1, int(value) - 1))
        original = {**state["session"], "identity": f"{prefix}:{previous}"}
        self.assertNotEqual(original["identity"], process_identity(os.getpid()))
        state["session"] = state["quarantine"]["owner"]["session"] = original
        self.resource.state_path.write_text(json.dumps(state), encoding="utf-8")
        evidence = self.resource.recover_after_session_exit(token)
        self.assertEqual(evidence["session"], original)
        self.assertEqual(evidence["kind"], "os_verified_bound_session_exit")
        self.assertIsNone(self.resource.snapshot()["quarantine"])

    def test_live_dedicated_session_cannot_be_silently_rebound(self):
        first = self.resource.bind_session(os.getpid(), "first-session")
        self.assertEqual(first["identity"], process_identity(os.getpid()))
        with self.assertRaisesRegex(MatlabResourceError, "still alive"):
            self.resource.bind_session(os.getpid(), "different-session")

    def test_corrupt_metadata_and_relative_resource_directory_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            MatlabResource(Path("per-session-relative-dir"))
        self.resource.state_path.write_text('{"schema_version": 999}', encoding="utf-8")
        with self.assertRaisesRegex(MatlabResourceError, "schema/directory mismatch"):
            self.resource.acquire("must-not-recreate-state")
        self.assertEqual(json.loads(self.resource.state_path.read_text())["schema_version"], 999)


if __name__ == "__main__":
    unittest.main()
