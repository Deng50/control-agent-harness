"""One machine-local MATLAB resource, shared by Harness, batch and Python callers.

The kernel owns exclusion. Durable tickets own FIFO order. Durable running state
survives a caller/transport crash and *never* expires by age. A returned Promise,
cancel acknowledgement or released OS lock is not evidence that MATLAB stopped.
Only a trusted synchronous adapter may complete a lease; unverified outcomes are
quarantined. Recovery requires the bound MATLAB process to have actually exited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import errno
import json
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import time
from typing import Callable
import uuid

from .locking import kernel_mutex, process_alive


class MatlabResourceError(RuntimeError):
    """Fail-closed resource protocol error."""


class MatlabResourceQuarantined(MatlabResourceError):
    """MATLAB may still be working; ordinary admission is disabled."""


class MatlabResourceCancelled(MatlabResourceError):
    """A queued caller was cancelled without dispatching MATLAB work."""


class MatlabResourceWaitTimeout(MatlabResourceError):
    """A caller's queue deadline elapsed without dispatching MATLAB work."""


def process_identity(pid: int) -> str | None:
    """OS process creation identity, so PID reuse cannot revive an old owner."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                return None
            return f"win32:{created.dwHighDateTime:08x}{created.dwLowDateTime:08x}"
        finally:
            kernel.CloseHandle(handle)
    try:
        # The command field may contain spaces and ')'; the suffix starts at state.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"linux:{boot}:{fields[19]}"  # /proc field 22, process start ticks.
    except (OSError, IndexError):
        return None


def _alive(record: dict) -> bool | None:
    if record.get("host") != socket.gethostname():
        return None
    live = process_alive(record.get("pid"))
    if live is False:
        return False
    actual = process_identity(record.get("pid"))
    if actual is not None:
        return False if actual != record.get("identity") else live
    return None


def _process_record(pid: int) -> dict:
    identity = process_identity(pid)
    if identity is None or process_alive(pid) is not True:
        raise MatlabResourceError(f"Cannot establish a live OS process identity for PID {pid}")
    return {"host": socket.gethostname(), "pid": pid, "identity": identity}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Windows readers/virus scanners can briefly deny FILE_SHARE_DELETE.
        # Retry only the atomic publish; this never expires or removes a lock.
        publish_deadline = time.monotonic() + 2
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= publish_deadline:
                    raise
                time.sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def _busy(error: OSError) -> bool:
    # A denied file open is a configuration/ACL error, not a contended lock.
    return error.filename is None and error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}


class MatlabResource:
    """Use the same absolute directory for every consumer of one MATLAB session."""

    def __init__(self, directory: str | Path, poll_interval: float = .05):
        path = Path(directory)
        if not path.is_absolute():
            raise ValueError("MATLAB resource directory must be absolute and shared across Sessions")
        if not .005 <= poll_interval <= 5:
            raise ValueError("MATLAB resource poll interval must be between .005 and 5 seconds")
        path.mkdir(parents=True, exist_ok=True)
        self.directory = path.resolve()
        self.state_path = self.directory / "resource.json"
        self.poll_interval = poll_interval

    @contextmanager
    def _metadata(self):
        # The lock inode is permanent. Only the OS releases ownership, never age.
        mutex = kernel_mutex(self.directory / "queue.mutex")
        while True:
            try:
                mutex.__enter__()
                break
            except OSError as error:
                if not _busy(error):
                    raise
                time.sleep(self.poll_interval)
                mutex = kernel_mutex(self.directory / "queue.mutex")
        try:
            if self.state_path.exists():
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if state.get("schema_version") != 1 or state.get("directory") != str(self.directory):
                    raise MatlabResourceError("MATLAB resource metadata schema/directory mismatch")
            else:
                state = {"schema_version": 1, "directory": str(self.directory), "next_ticket": 0,
                         "queue": [], "owner": None, "quarantine": None, "session": None}
            yield state
        finally:
            mutex.__exit__(None, None, None)

    def _save(self, state: dict) -> None:
        _atomic_json(self.state_path, state)

    def _repair(self, state: dict) -> None:
        # Death releases the execution mutex but tells us nothing about MATLAB.
        owner = state.get("owner")
        if owner is not None and _alive(owner) is False:
            state["quarantine"] = {"token": owner["token"], "reason": "owner_process_exited",
                                   "owner": owner, "recorded_at": time.time()}
            state["owner"] = None
        state["queue"] = [entry for entry in state["queue"] if _alive(entry) is not False]

    def snapshot(self) -> dict:
        with self._metadata() as state:
            self._repair(state)
            self._save(state)
            return json.loads(json.dumps(state))

    def bind_session(self, pid: int, session_id: str) -> dict:
        """Administrative startup only: bind the dedicated MATLAB process identity.

        This is deliberately not a conversational tool. Reconnection must verify
        this identity separately; a namespace is not a MATLAB session identity.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("A nonempty dedicated MATLAB session id is required")
        binding = {**_process_record(pid), "session_id": session_id}
        with self._metadata() as state:
            self._repair(state)
            if state["owner"] or state["quarantine"] or state["queue"]:
                raise MatlabResourceError("Cannot change MATLAB session binding while owned, queued or quarantined")
            previous = state.get("session")
            if previous and previous != binding and _alive(previous) is not False:
                raise MatlabResourceError("Previous MATLAB session is still alive or its exit is unverified")
            state["session"] = binding
            self._save(state)
        return binding

    def acquire(self, label: str, *, cancelled: Callable[[], bool] = lambda: False,
                timeout: float | None = None, on_queued: Callable[[dict], None] = lambda _: None):
        """Register FIFO order durably; cancellation removes only this caller's ticket."""
        if not isinstance(label, str) or not label.strip() or len(label) > 256:
            raise ValueError("MATLAB resource label must be 1–256 characters")
        if timeout is not None and timeout <= 0:
            raise ValueError("Queue timeout must be positive")
        if cancelled():
            raise MatlabResourceCancelled("Cancelled before MATLAB resource admission")
        entry = {**_process_record(os.getpid()), "token": uuid.uuid4().hex, "label": label}
        started = time.monotonic()
        with self._metadata() as state:
            self._repair(state)
            if state["quarantine"]:
                self._save(state)
                raise MatlabResourceQuarantined(f"MATLAB resource is quarantined: {state['quarantine']['token']}")
            entry["ticket"] = state["next_ticket"]
            state["next_ticket"] += 1
            state["queue"].append(entry)
            self._save(state)
        try:
            on_queued(dict(entry))
            while True:
                if cancelled():
                    raise MatlabResourceCancelled("Cancelled while queued for MATLAB; no MATLAB call was dispatched")
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise MatlabResourceWaitTimeout("MATLAB resource queue deadline elapsed; no MATLAB call was dispatched")
                with self._metadata() as state:
                    self._repair(state)
                    if state["quarantine"]:
                        self._save(state)
                        raise MatlabResourceQuarantined(f"MATLAB resource is quarantined: {state['quarantine']['token']}")
                    if state["owner"] is None and state["queue"] and state["queue"][0]["token"] == entry["token"]:
                        execution = kernel_mutex(self.directory / "execution.mutex")
                        try:
                            execution.__enter__()
                        except OSError as error:
                            if not _busy(error):
                                raise
                            state["quarantine"] = {"token": entry["token"], "reason": "execution_mutex_has_unrecorded_owner",
                                                   "owner": None, "recorded_at": time.time()}
                            self._save(state)
                            raise MatlabResourceQuarantined("Execution mutex has an unrecorded owner; resource quarantined") from error
                        owner = {**entry, "phase": "reserved", "session": state.get("session")}
                        state["queue"].pop(0)
                        state["owner"] = owner
                        try:
                            self._save(state)
                        except BaseException:
                            execution.__exit__(None, None, None)
                            raise
                        return MatlabLease(self, owner, execution)
                    self._save(state)
                time.sleep(self.poll_interval)
        except BaseException:
            with self._metadata() as state:
                state["queue"] = [item for item in state["queue"] if item["token"] != entry["token"]]
                self._save(state)
            raise

    def recover_after_session_exit(self, token: str) -> dict:
        """Administrative recovery; OS proof of bound session exit, not idle=true.

        The ordinary LLM/MCP tool catalog intentionally has no reset operation.
        A fresh dedicated session can be bound only after this evidence is saved.
        """
        with self._metadata() as state:
            self._repair(state)
            quarantine = state["quarantine"]
            if not quarantine or quarantine.get("token") != token:
                raise MatlabResourceError("Quarantine token mismatch; inspect current resource state")
            binding = (quarantine.get("owner") or {}).get("session") or state.get("session")
            if not binding:
                raise MatlabResourceError("No dedicated MATLAB process was bound; automatic recovery is unavailable")
            if _alive(binding) is not False:
                raise MatlabResourceError("Bound MATLAB process has not been proven exited; resource remains quarantined")
            execution = kernel_mutex(self.directory / "execution.mutex")
            try:
                execution.__enter__()
            except OSError as error:
                if not _busy(error):
                    raise
                raise MatlabResourceError("Execution owner has not released its kernel lock") from error
            try:
                evidence = {"schema_version": 1, "kind": "os_verified_bound_session_exit", "token": token,
                            "directory": str(self.directory), "session": binding, "quarantine": quarantine,
                            "verifier": _process_record(os.getpid()), "verified_at": time.time()}
                archive = self.directory / "recoveries" / f"{token}.json"
                archive.parent.mkdir(exist_ok=True)
                _atomic_json(archive, evidence)
                state["quarantine"] = None
                state["session"] = None
                state["last_recovery"] = str(archive)
                self._save(state)
                return evidence
            finally:
                execution.__exit__(None, None, None)


class MatlabLease:
    def __init__(self, resource: MatlabResource, owner: dict, execution):
        self.resource, self.owner, self.execution = resource, owner, execution
        self.closed = False

    def _owned(self, state: dict) -> None:
        if self.closed or not state.get("owner") or state["owner"].get("token") != self.owner["token"]:
            raise MatlabResourceError("MATLAB resource lease is no longer owned by this caller")

    def mark_running(self) -> None:
        with self.resource._metadata() as state:
            self._owned(state)
            state["owner"]["phase"] = "running"
            self.resource._save(state)
            self.owner["phase"] = "running"

    def _release(self, *, evidence: dict | None = None, reason: str | None = None) -> None:
        try:
            with self.resource._metadata() as state:
                self._owned(state)
                if reason:
                    state["quarantine"] = {"token": self.owner["token"], "reason": reason,
                                           "owner": dict(state["owner"]), "recorded_at": time.time()}
                else:
                    state["last_completion"] = {"token": self.owner["token"], "evidence": evidence}
                state["owner"] = None
                self.resource._save(state)
        finally:
            self.closed = True
            self.execution.__exit__(None, None, None)

    def complete(self, evidence: dict) -> None:
        """Trusted adapter boundary only; never feed model-supplied arguments here."""
        if (not isinstance(evidence, dict) or evidence.get("kind") != "synchronous_adapter_return"
                or not isinstance(evidence.get("producer"), str) or not evidence["producer"].strip()):
            raise MatlabResourceError("Completion requires trusted synchronous-adapter evidence, not Promise settlement or idle=true")
        if self.owner["phase"] != "running":
            raise MatlabResourceError("Cannot complete a MATLAB call that was not marked running")
        self._release(evidence=evidence)

    def cancel_before_dispatch(self) -> None:
        if self.owner["phase"] != "reserved":
            raise MatlabResourceError("A dispatched MATLAB call cannot be released as queued cancellation")
        self._release(evidence={"kind": "cancelled_before_dispatch"})

    def quarantine(self, reason: str) -> None:
        self._release(reason=reason or "unverified_matlab_state")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if not self.closed:
            if self.owner["phase"] == "reserved":
                self.cancel_before_dispatch()
            else:
                self.quarantine("adapter_left_without_verified_completion")


def _broker(resource: MatlabResource) -> int:
    """Per-call lock holder, owned by Harness subprocess. No numerical execution."""
    inbox: queue.Queue = queue.Queue()
    stopped = threading.Event()

    def read_commands():
        try:
            for line in sys.stdin:
                if len(line) > 65536:
                    raise MatlabResourceError("Resource broker command exceeds 64 KiB")
                command = json.loads(line)
                if command.get("command") == "cancel":
                    stopped.set()
                inbox.put(command)
        finally:
            stopped.set()
            inbox.put({"command": "eof"})

    def emit(event: str, **value):
        print(json.dumps({"event": event, **value}, allow_nan=False), flush=True)

    threading.Thread(target=read_commands, daemon=True).start()
    lease = None
    try:
        first = inbox.get()
        if first.get("command") != "acquire":
            raise MatlabResourceError("First resource broker command must be acquire")
        lease = resource.acquire(first["label"], cancelled=stopped.is_set,
                                 timeout=first.get("wait_timeout"), on_queued=lambda entry: emit("queued", **entry))
        emit("acquired", token=lease.owner["token"], ticket=lease.owner["ticket"], session=lease.owner["session"])
        while True:
            command = inbox.get()
            kind = command.get("command")
            if kind == "start":
                if stopped.is_set():
                    lease.cancel_before_dispatch()
                    emit("cancelled")
                    return 0
                # 专用任务在排队结束后再次读取操作系统进程创建标识，防止
                # MATLAB 已退出或 PID 被重用时继续以旧的用户授权执行。
                if command.get("require_live_session") and _alive(lease.owner.get("session") or {}) is not True:
                    lease.cancel_before_dispatch()
                    emit("cancelled", reason="bound_session_identity_changed")
                    return 0
                lease.mark_running()
                emit("started")
            elif kind == "complete":
                evidence = command.get("evidence")
                # A cancel racing completion is deliberately quarantined.
                if stopped.is_set():
                    lease.quarantine("cancelled_after_matlab_dispatch")
                    emit("quarantined")
                elif (isinstance(evidence, dict) and evidence.get("call_outcome") == "synchronous_parameter_error"
                      and _alive(lease.owner.get("session") or {}) is not True):
                    # MATLAB 探针核验会话 ID 和 PID；在释放内核锁前再核对操作系统进程创建标识。
                    lease.quarantine("bound_session_identity_changed")
                    emit("quarantined", reason="bound_session_identity_changed")
                else:
                    lease.complete(evidence)
                    emit("completed")
                return 0
            elif kind == "quarantine":
                lease.quarantine(command.get("reason", "unverified_matlab_state"))
                emit("quarantined")
                return 0
            elif kind in {"cancel", "eof"}:
                if lease.owner["phase"] == "reserved":
                    lease.cancel_before_dispatch()
                    emit("cancelled")
                else:
                    lease.quarantine("caller_disconnected_or_cancelled_after_dispatch")
                    emit("quarantined")
                return 0
            else:
                raise MatlabResourceError(f"Unknown resource broker command: {kind}")
    except BaseException as error:
        if lease and not lease.closed:
            lease.__exit__(None, None, None)
        emit("error", error_type=type(error).__name__, message=str(error))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["broker", "inspect", "bind-session", "recover-after-session-exit"])
    parser.add_argument("--directory", required=True)
    parser.add_argument("--poll-ms", type=float, default=50)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--token")
    args = parser.parse_args()
    resource = MatlabResource(args.directory, args.poll_ms / 1000)
    if args.action == "broker":
        return _broker(resource)
    if args.action == "inspect":
        result = resource.snapshot()
    elif args.action == "bind-session":
        result = resource.bind_session(args.pid, args.session_id)
    else:
        result = resource.recover_after_session_exit(args.token)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
