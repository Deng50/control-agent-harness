"""Cooperative numerical-worker control; Harness owns the calling job lifecycle."""
from __future__ import annotations

import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class Cancelled(RuntimeError):
    pass


@dataclass
class OperationContext:
    emit: Callable[[dict], None] = lambda event: None
    cancel_file: Path | None = None
    deadline: float | None = None
    cancelled: Callable[[], bool] | None = None
    _case_receipt_body: str | None = field(init=False, default=None, repr=False)
    _case_receipt_key: str | None = field(init=False, default=None, repr=False)
    _case_receipt_signature: str | None = field(init=False, default=None, repr=False)
    _case_native_session: str | None = field(init=False, default=None, repr=False)
    _case_raw_request: str | None = field(init=False, default=None, repr=False)
    _test_fixture_workspace: Path | None = field(init=False, default=None, repr=False)
    case_confirmation_sources: list[dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # Only the Host's fixed subprocess environment can carry this receipt.
        # Numerical request JSON is deliberately never consulted for authority.
        self._case_receipt_body = os.environ.get("CONTROL_CASE_RECEIPT_BODY")
        self._case_receipt_key = os.environ.get("CONTROL_CASE_RECEIPT_KEY")
        self._case_receipt_signature = os.environ.get("CONTROL_CASE_RECEIPT_SIGNATURE")
        self._case_native_session = os.environ.get("CONTROL_CASE_NATIVE_SESSION")

    def bind_raw_request(self, text: str) -> None:
        self._case_raw_request = text.rstrip("\r\n")

    def require_case_confirmation(self, request: dict, requirements: list[dict]) -> list[dict]:
        from .case_plans import require_confirmation
        return require_confirmation(self, request, requirements)

    @classmethod
    def test_fixture(cls, workspace, **kwargs):
        """Explicit trusted Python test code only; a JSON field cannot call this."""
        from .case_plans import fixture_workspace
        fixture_workspace(workspace)
        context = cls(**kwargs)
        context._test_fixture_workspace = Path(getattr(workspace, "path", workspace)).resolve()
        return context

    def check(self) -> None:
        if self.cancelled and self.cancelled():
            raise Cancelled("Cancellation requested by the Harness job")
        if self.cancel_file and self.cancel_file.exists():
            raise Cancelled("Cancellation requested")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("Numerical operation exceeded its elapsed time budget")

    def progress(self, stage: str, **values) -> None:
        self.check()
        self.emit({"event": "progress", "stage": stage, **values})
