"""JSON-line numerical worker. No model calls, chat server or alternate agent loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import threading

from .runtime import OperationContext
from .workspace_probe import inspect_workspace


def emit(event: dict) -> None:
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--request", type=Path)
    inputs.add_argument("--stdin", action="store_true")
    inputs.add_argument("--protocol", action="store_true", help="First JSON line is the request; subsequent lines carry cancellation")
    inputs.add_argument("--demo", choices=["linear", "nonlinear"])
    inputs.add_argument("--preflight", action="store_true", help="Check direct CLI workspace and TEMP using this process token")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if (args.demo or args.preflight) and args.workspace is None:
        parser.error("--demo and --preflight require --workspace")
    if args.preflight:
        report = inspect_workspace(args.workspace.absolute())
        emit({"actor": "direct_cli", **report})
        return 0 if report["ok"] else 1
    cancelled = threading.Event()
    context = OperationContext(emit=emit, cancel_file=args.cancel_file,
                               deadline=time.monotonic() + args.timeout, cancelled=cancelled.is_set)
    try:
        if args.protocol:
            raw_request = sys.stdin.readline()
            context.bind_raw_request(raw_request)
            request = json.loads(raw_request)

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

        else:
            request = ({"operation": "demo", "workspace": str(args.workspace), "config": {"case": args.demo}}
                       if args.demo else json.loads(sys.stdin.read() if args.stdin else args.request.read_text(encoding="utf-8-sig")))
            # 独立 CLI 沿用原有新建工作区行为，并在加载训练依赖前，用自己的令牌验写。
            # --protocol 由 Harness 的 Host + 受限 worker 门禁负责，避免重复启动探针。
            direct = Path(request["workspace"]).expanduser().absolute()
            report = inspect_workspace(direct, create=True, operation=request["operation"])
            if not report["ok"]:
                emit({"event": "error", "error_type": "WorkspacePreflightError", "actor": "direct_cli",
                      "stage": "preflight", "work_started": False, **report})
                return 1
        import torch
        from .operations import run_operation
        torch.set_num_threads(1)
        # 先完成数值扩展模块初始化，再让后台线程阻塞读取协议管道。
        if args.protocol:
            threading.Thread(target=control_reader, daemon=True).start()
        emit({"event": "started", "pid": __import__("os").getpid(), "operation": request["operation"]})
        result = run_operation(request, context)
        emit({"event": "result", "result": result})
        return 2 if result.get("status") == "rejected" and request["operation"] not in {"inspect_artifact", "read_artifact_file"} else 0
    except Exception as exc:
        from .artifacts import failure_fields
        emit({"event": "error", "error_type": type(exc).__name__, "message": str(exc), **failure_fields(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
