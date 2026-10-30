"""Cooperative numerical-worker control; Harness owns the calling job lifecycle."""
from __future__ import annotations

import time
from dataclasses import dataclass
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
