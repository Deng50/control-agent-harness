"""JSON-line numerical worker. No model calls, chat server or alternate agent loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import threading

from .operations import run_operation
from .runtime import OperationContext


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--request", type=Path)
    inputs.add_argument("--stdin", action="store_true")
    inputs.add_argument("--protocol", action="store_true", help="First JSON line is the request; subsequent lines carry cancellation")
    inputs.add_argument("--demo", choices=["linear", "nonlinear"])
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.demo and args.workspace is None:
        parser.error("--demo requires --workspace")
    import torch
    torch.set_num_threads(1)
    cancelled = threading.Event()
    context = OperationContext(emit=emit, cancel_file=args.cancel_file,
                               deadline=time.monotonic() + args.timeout, cancelled=cancelled.is_set)
    try:
        if args.protocol:
            request = json.loads(sys.stdin.readline())

            def control_reader():
                for line in sys.stdin:
                    try:
                        command = json.loads(line)
                    except ValueError:
                        cancelled.set()
                        return
                    if command.get("command") == "cancel":
                        cancelled.set()
                cancelled.set()  # A lost owner must not leave training running indefinitely.

            threading.Thread(target=control_reader, daemon=True).start()
        else:
            request = ({"operation": "demo", "workspace": str(args.workspace), "config": {"case": args.demo}}
                       if args.demo else json.loads(sys.stdin.read() if args.stdin else args.request.read_text(encoding="utf-8-sig")))
        emit({"event": "started", "pid": __import__("os").getpid(), "operation": request["operation"]})
        result = run_operation(request, context)
        emit({"event": "result", "result": result})
        return 2 if result.get("status") == "rejected" and request["operation"] not in {"inspect_artifact", "read_artifact_file"} else 0
    except Exception as exc:
        emit({"event": "error", "error_type": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
