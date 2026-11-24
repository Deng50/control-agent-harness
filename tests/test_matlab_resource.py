"""Kernel-backed resource semantics; no MATLAB/Engine or numerical fixture is simulated as MATLAB."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import uuid
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

    def test_host_observed_dispatch_denial_after_start_ack_releases_only_original_ticket(self):
        with self.resource.acquire("native-body-denial") as owner:
            owner.mark_running()
            with self.assertRaisesRegex(MatlabResourceError, "identify"):
                owner.not_dispatched("another-token")
            self.assertEqual(self.resource.snapshot()["owner"]["phase"], "running")
            owner.not_dispatched(owner.owner["token"])
        state = self.resource.snapshot()
        self.assertIsNone(state["owner"])
        self.assertIsNone(state["quarantine"])
        self.assertEqual(state["last_completion"]["evidence"]["kind"], "not_dispatched")

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

    def test_verification_deadline_does_not_limit_normal_fifo_wait(self):
        queued = threading.Event()
        def acquire():
            with self.resource.acquire("waiter", wait_for_recovery=True, pending_timeout=.03,
                                       timeout=4, on_queued=lambda _: queued.set()):
                pass
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.resource.acquire("ordinary-long-call"):
                waiter = executor.submit(acquire)
                self.assertTrue(queued.wait(2))
                with self.assertRaises(FutureTimeout):  # still queued past its pending-only limit
                    waiter.result(timeout=.1)
                self.assertFalse(waiter.done())
            waiter.result(timeout=2)
        self.resource.bind_session(os.getpid(), "waiting-for-recoverable-return")
        with self.resource.acquire("unknown-return") as lease:
            lease.mark_running(str(uuid.uuid4()))
        with self.assertRaisesRegex(MatlabResourceWaitTimeout, "pending verification"):
            self.resource.acquire("waiter-for-verification", wait_for_recovery=True, pending_timeout=.03)
        self.assertEqual(self.resource.snapshot()["queue"], [])

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

    def test_online_recovery_requires_original_dispatch_and_fresh_same_process_certificate(self):
        binding = self.resource.bind_session(os.getpid(), "original-session")
        nonce = str(uuid.uuid4())
        with self.resource.acquire("disconnected") as owner:
            owner.mark_running(nonce)
            token = owner.owner["token"]
        self.assertEqual(self.resource.bind_session(os.getpid(), "original-session", allow_existing_pending=True), binding)
        with self.assertRaises(MatlabResourceError):
            self.resource.bind_session(os.getpid(), "another-session", allow_existing_pending=True)
        with self.resource.recovery_lease(token) as (recovery, finish):
            self.assertEqual(recovery["completion_nonce"], nonce)
            with self.assertRaisesRegex(MatlabResourceError, "owns"):
                with self.resource.recovery_lease(token):
                    self.fail("concurrent recovery admitted")
            proof = {"kind": "synchronous_adapter_return", "producer": "control_harness_terminal_probe-v1",
                     "matlab_pid": os.getpid(), "session_id": binding["session_id"],
                     "nonce": recovery["probe_nonce"], "completion_nonce": nonce}
            for field in ("completion_nonce", "nonce", "session_id", "matlab_pid"):
                with self.subTest(field=field), self.assertRaisesRegex(MatlabResourceError, "same-process"):
                    finish({**proof, field: "wrong"})
            with patch("control_harness.matlab_resource._alive", return_value=False), self.assertRaisesRegex(MatlabResourceError, "same-process"):
                finish(proof)
            self.assertEqual(self.resource.snapshot()["quarantine"]["token"], token)
            finish(proof)
        state = self.resource.snapshot()
        self.assertIsNone(state["quarantine"])
        self.assertEqual(state["session"], binding)
        self.assertEqual(json.loads(Path(state["last_recovery"]).read_text())["evidence"], proof)
        with self.assertRaisesRegex(MatlabResourceError, "token mismatch"):
            with self.resource.recovery_lease(token):
                self.fail("stale recovery admitted")

    def test_legacy_pending_allows_diagnostic_probe_without_unlocking(self):
        state = self.quarantined_process_fixture()
        with self.assertRaisesRegex(MatlabResourceQuarantined, "缺少可关联的完成凭证"):
            self.resource.acquire("legacy-status", wait_for_recovery=True)
        self.assertEqual(self.resource.snapshot()["queue"], [])
        with self.resource.recovery_lease(state["quarantine"]["token"]) as (recovery, finish):
            self.assertIsNone(recovery["completion_nonce"])
            with self.assertRaisesRegex(MatlabResourceError, "same-process"):
                finish(dict(kind="synchronous_adapter_return", producer="control_harness_terminal_probe-v1",
                            nonce=recovery["probe_nonce"], completion_nonce=None,
                            session_id=state["session"]["session_id"], matlab_pid=os.getpid()))
        self.assertEqual(self.resource.snapshot()["quarantine"], state["quarantine"])

    def test_late_rpc_proof_is_bound_to_original_tool_transport_and_lease(self):
        binding = self.resource.bind_session(os.getpid(), "rpc-session")
        tool = "mcp__matlab__model_read"
        nonce = str(uuid.uuid4())
        with self.resource.acquire(tool + ":call-7") as owner:
            owner.mark_running(nonce, "mcp_response")
        token = self.resource.snapshot()["quarantine"]["token"]
        with self.resource.recovery_lease(token) as (recovery, finish):
            self.assertEqual(recovery["completion_source"], "mcp_response")
            proof = {"kind":"synchronous_adapter_return", "producer":"control_harness_terminal_probe-v1",
                     "nonce":recovery["probe_nonce"], "completion_source":"mcp_response", "completion_nonce":nonce,
                     "matlab_pid":binding["pid"], "session_id":binding["session_id"],
                     "lease_token":token, "tool_name":tool, "transport_generation":str(uuid.uuid4()), "rpc_request_id":7}
            for field, bad in (("tool_name","another-tool"), ("lease_token","another-lease"), ("rpc_request_id",None),
                               ("transport_generation","invalid"), ("completion_source","matlab_marker")):
                with self.subTest(field=field), self.assertRaises(MatlabResourceError):
                    finish({**proof, field:bad})
            finish(proof)
        self.assertIsNone(self.resource.snapshot()["quarantine"])

    def test_waiters_end_after_old_instance_exits_allowing_automatic_rebind(self):
        matlab = self.spawn("import sys; print('ready'); sys.stdin.readline()")
        self.assertEqual(matlab.stdout.readline().strip(), "ready")
        self.resource.bind_session(matlab.pid, "original")
        with self.resource.acquire("uncertain") as owner:
            owner.mark_running(str(uuid.uuid4()))
        with ThreadPoolExecutor(max_workers=1) as executor:
            queued = threading.Event()
            waiter = executor.submit(self.resource.acquire, "old-instance-only", wait_for_recovery=True,
                                     on_queued=lambda _: queued.set())
            self.assertTrue(queued.wait(2))
            matlab.communicate(timeout=5)
            with self.assertRaisesRegex(MatlabResourceError, "已退出"):
                waiter.result(timeout=3)
        state = self.resource.snapshot()
        self.assertEqual(state["queue"], [])
        self.assertIsNone(state["owner"])
        self.resource.recover_after_session_exit(state["quarantine"]["token"])
        self.assertEqual(self.resource.bind_session(os.getpid(), "replacement")["session_id"], "replacement")

    def test_reserved_broker_exit_does_not_pause_the_following_request(self):
        exited = self.spawn(r'''
import os, sys
from control_harness.matlab_resource import MatlabResource
lease = MatlabResource(sys.argv[1]).acquire("never-dispatched")
os._exit(0)
''', self.path)
        exited.communicate(timeout=10)
        self.assertEqual(exited.returncode, 0)
        with self.resource.acquire("next", timeout=1):
            pass
        self.assertIsNone(self.resource.snapshot()["quarantine"])

    def test_waiting_ticket_resumes_in_original_session_after_verified_recovery(self):
        binding = self.resource.bind_session(os.getpid(), "queued-same-instance")
        with self.resource.acquire("dispatched-with-late-return") as owner:
            nonce = str(uuid.uuid4())
            owner.mark_running(nonce)
            token = owner.owner["token"]
        child = self.spawn(r'''
import sys, time
from control_harness.matlab_resource import MatlabResource
clock = time.monotonic
polls = 0
def cancelled():
    global polls
    polls += 1
    if polls == 4:
        # Hours of waiting for one marked call must not expire a live caller.
        time.monotonic = lambda: clock() + 3600
        print("still waiting", flush=True)
    return False
with MatlabResource(sys.argv[1], .01).acquire("waiter", wait_for_recovery=True, cancelled=cancelled,
                     on_queued=lambda _: print("queued", flush=True)) as lease:
    print(lease.owner["session"]["session_id"], flush=True)
''', self.path)
        self.assertEqual(child.stdout.readline().strip(), "queued")
        self.assertEqual(child.stdout.readline().strip(), "still waiting")
        self.assertIsNone(self.resource.snapshot()["owner"])
        with self.resource.recovery_lease(token) as (recovery, finish):
            finish(dict(kind="synchronous_adapter_return", producer="control_harness_terminal_probe-v1",
                        nonce=recovery["probe_nonce"], completion_nonce=nonce,
                        session_id=binding["session_id"], matlab_pid=os.getpid()))
        output, error = child.communicate(timeout=10)
        self.assertEqual(child.returncode, 0, error)
        self.assertEqual(output.strip(), binding["session_id"])

    def test_replacement_host_uses_original_completion_nonce_over_broker_protocol(self):
        binding = self.resource.bind_session(os.getpid(), "host-restart-test")
        nonce = str(uuid.uuid4())
        broker = """
import sys
from control_harness.matlab_resource import MatlabResource, _broker
sys.exit(_broker(MatlabResource(sys.argv[1], .01)))
"""
        def send(process, command, event):
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
            answer = json.loads(process.stdout.readline())
            self.assertEqual(answer["event"], event, answer)
            return answer

        old = self.spawn(broker, self.path)
        send(old, {"command":"acquire", "label":"first-host"}, "queued")
        held = json.loads(old.stdout.readline())
        self.assertEqual(held["event"], "acquired")
        send(old, {"command":"start", "require_live_session":True, "completion_nonce":nonce}, "started")
        # Abrupt death leaves only the disk record and releases kernel handles.
        old.kill()
        old.communicate(timeout=10)
        observer = self.spawn(broker, self.path)
        state = send(observer, {"command":"inspect"}, "state")
        observer.communicate(timeout=10)
        self.assertIsNone(state["owner"])
        self.assertEqual(state["quarantine"]["owner"]["completion_nonce"], nonce)
        unfinished = self.spawn(broker, self.path)
        request = {"command":"recovery-acquire", "token":held["token"]}
        initial = send(unfinished, request, "recovery-acquired")
        send(unfinished, {"command":"defer"}, "deferred")
        unfinished.communicate(timeout=10)
        self.assertEqual(self.resource.snapshot()["quarantine"]["token"], held["token"])

        new = self.spawn(broker, self.path)
        recovery = send(new, request, "recovery-acquired")
        self.assertNotEqual(initial["probe_nonce"], recovery["probe_nonce"])
        self.assertEqual(recovery["session"], binding)
        self.assertEqual(recovery["completion_nonce"], nonce)
        proof = {"kind":"synchronous_adapter_return", "producer":"control_harness_terminal_probe-v1",
                 "nonce":recovery["probe_nonce"], "completion_nonce":nonce,
                 "matlab_pid":binding["pid"], "session_id":binding["session_id"]}
        send(new, {"command":"recover", "evidence":proof}, "recovered")
        new.communicate(timeout=10)
        self.assertIsNone(self.resource.snapshot()["quarantine"])

    def test_corrupt_metadata_and_relative_resource_directory_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            MatlabResource(Path("per-session-relative-dir"))
        self.resource.state_path.write_text('{"schema_version": 999}', encoding="utf-8")
        with self.assertRaisesRegex(MatlabResourceError, "schema/directory mismatch"):
            self.resource.acquire("must-not-recreate-state")
        self.assertEqual(json.loads(self.resource.state_path.read_text())["schema_version"], 999)


if __name__ == "__main__":
    unittest.main()
