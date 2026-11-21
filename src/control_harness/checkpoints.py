"""Validated, atomic epoch-boundary checkpoints for numerical training workers."""
from __future__ import annotations

import os
import hashlib
import io
import json
import random
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import scipy
import torch

from .artifacts import Workspace, atomic_json, file_hash, read_json
from .contracts import ContractError, canonical_hash


FORMAT_VERSION = 1
HEAD = "checkpoint-head.json"
GENERATION = re.compile(r"checkpoint-[0-9a-f]{32}")


class CheckpointWriteError(OSError):
    """将系统错误和写入阶段原样保留给作业清单，避免把所有 Win32 5 当成文件占用。"""

    def __init__(self, operation: str, path: Path, cause: Exception):
        winerror = getattr(cause, "winerror", None)
        self.code = ("CHECKPOINT_FILE_BUSY" if winerror in {32, 33} else
                     "CHECKPOINT_ACCESS_DENIED" if winerror == 5 or isinstance(cause, PermissionError) else
                     "CHECKPOINT_IO_ERROR")
        super().__init__(f"{self.code}: {operation} ({path}): {cause}")
        self.io_failure = {"operation": operation, "path": str(path), "errno": getattr(cause, "errno", None), "winerror": winerror}
        if getattr(cause, "cleanup_errors", None):
            self.cleanup_errors = cause.cleanup_errors


def _head(root: Path) -> dict | None:
    """只认已发布的小指针。目录中较新的孤立文件可能是崩溃前的半成品。"""
    path = root / HEAD
    if path.resolve() != path.absolute():
        raise ContractError("Checkpoint pointer must be a regular local file")
    try:
        value = read_json(path)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        raise ContractError("Invalid committed checkpoint pointer") from exc
    if (not isinstance(value, dict) or value.get("format_version") != 2
            or not isinstance(value.get("current"), str) or not GENERATION.fullmatch(value["current"])
            or "previous" not in value or value["previous"] == value["current"]
            or (value["previous"] is not None and
                (not isinstance(value["previous"], str) or not GENERATION.fullmatch(value["previous"])))):
        raise ContractError("Invalid committed checkpoint pointer")
    return value


def current_checkpoint(root: Path) -> Path:
    head = _head(root)
    return root / (head["current"] + ".pt" if head is not None else "latest.pt")


def checkpoint_sidecar(path: Path) -> Path:
    if path.name == "latest.pt":
        return path.with_name("latest.checkpoint.json")
    if path.suffix != ".pt" or not GENERATION.fullmatch(path.stem):
        raise ContractError("Invalid checkpoint generation filename")
    return path.with_suffix(".json")


def _publish_pointer(path: Path, value: dict):
    """只对 Windows 明确报告的共享/锁冲突做两次短退避。其它错误立即返回。"""
    for delay in (0.0, .025, .075):
        if delay:
            time.sleep(delay)
        try:
            atomic_json(path, value)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {32, 33} or delay == .075:
                raise


def checkpoint_fields(path: Path, status: str) -> dict:
    sidecar = read_json(checkpoint_sidecar(path))
    return {"checkpoint": str(path), "checkpoint_sha256": sidecar["sha256"],
            "epoch": sidecar["epoch"], "optimizer_updates": sidecar["updates"],
            "target_epochs": sidecar.get("target_epochs"), "checkpoint_status": status}


def verified_recovery_point(root: Path, contract: dict, status: str) -> dict:
    """只为失败现场选择已提交且可验证的恢复点；用户指定代号和摘要的续训始终精确匹配。"""
    head = _head(root)
    candidates = [root / (head["current"] + ".pt") if head else root / "latest.pt"]
    if head and head["previous"]:
        candidates.append(root / (head["previous"] + ".pt"))
    current_error = None
    for path in candidates:
        try:
            checkpoint, digest = read_verified_checkpoint(path, contract)
            fields = checkpoint_fields(path, status)
            if fields["checkpoint_sha256"] != digest or fields["epoch"] != checkpoint["epoch"]:
                raise ContractError("Committed checkpoint metadata changed during recovery verification")
        except (ContractError, OSError, ValueError, KeyError) as exc:
            if path == candidates[-1]:
                if current_error is None:
                    raise
                raise ContractError(f"Neither committed generation verified. Current: {current_error}. Previous: {exc}") from exc
            current_error = str(exc)
            continue
        fields["checkpoint_selection"] = "previous" if current_error is not None else "current"
        if current_error is not None:
            fields["checkpoint_fallback_reason"] = current_error
        return fields
    raise ContractError("No committed checkpoint is available")


def snapshot_latest(root: Path) -> None:
    """仅在停止写入时生成旧消费者使用的一份 latest.pt；训练循环只操作不可变代号。"""
    path = current_checkpoint(root)
    if path.name == "latest.pt":
        return
    # 归档或发布间隔内底层文件有可能损坏，先核验数据再给只认 latest 的旧客户端制作副本。
    contract = read_json(root / "stage.intent.json")["training"]["contract"]
    read_verified_checkpoint(path, contract)
    for source, name in ((path, "latest.pt"), (checkpoint_sidecar(path), "latest.checkpoint.json")):
        destination = root / name
        if source.resolve() != source.absolute() or destination.resolve() != destination.absolute():
            raise ContractError("Checkpoint snapshot refuses a filesystem alias")
        if destination.exists():
            if file_hash(source) != file_hash(destination):
                raise ContractError("Existing checkpoint snapshot differs from the committed generation")
            continue
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())


def training_contract(workspace: Workspace, algorithm: str, dependencies: list[str],
                      effective_config: dict, interface_hash: str) -> dict:
    """Bind an optimizer trajectory to exact data/model versions and effective settings.

    Epoch budgets, checkpoint locations and engineering authorization references
    are not optimizer settings. All effective numerical settings stay bound.
    Resolving dependencies also verifies their complete transitive file closure.
    """
    parents = []
    for artifact_id in dict.fromkeys(dependencies):
        _, metadata = workspace.resolve(artifact_id)
        parents.append({"id": artifact_id, "metadata_hash": canonical_hash(metadata)})
    return {"algorithm": algorithm, "algorithm_version": 1, "interface_hash": interface_hash,
            "dependencies": parents,
            "config": {k: v for k, v in effective_config.items()
                       if k not in {"epochs", "resume_checkpoint", "resume_checkpoint_sha256", "plan_refs"}},
            "runtime": {"python": list(sys.version_info[:3]), "numpy": str(np.__version__),
                        "scipy": str(scipy.__version__), "torch": str(torch.__version__),
                        "device": "cpu", "dtype": "float64"}}


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_snapshot(generator=None, numpy_generator=None) -> dict:
    legacy = np.random.get_state()
    result = {"python": random.getstate(), "torch": torch.get_rng_state(),
              "numpy": {"kind": legacy[0], "keys": legacy[1].tolist(), "position": legacy[2],
                        "has_gauss": legacy[3], "cached_gaussian": legacy[4]}}
    if generator is not None:
        result["generator"] = generator.get_state()
    if numpy_generator is not None:
        result["numpy_generator"] = numpy_generator.bit_generator.state
    return result


def restore_rng(state: dict, generator=None, numpy_generator=None) -> None:
    try:
        legacy = state["numpy"]
        random.setstate(state["python"])
        np.random.set_state((legacy["kind"], np.asarray(legacy["keys"], dtype=np.uint32),
                             legacy["position"], legacy["has_gauss"], legacy["cached_gaussian"]))
        torch.set_rng_state(state["torch"])
        if generator is not None:
            generator.set_state(state["generator"])
        if numpy_generator is not None:
            numpy_generator.bit_generator.state = state["numpy_generator"]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ContractError(f"Invalid checkpoint random state: {exc}") from exc


def bind_checkpoint_intent(root: Path, contract: dict) -> str:
    """Bind a stage's immutable owner/config intent before its first checkpoint."""
    path = root / "stage.intent.json"
    intent = read_json(path)
    owner = read_json(root.parent.parent / ".operation.lock")
    if (root.parent.name != ".staging" or intent.get("stage_id") != root.name
            or intent.get("owner", {}).get("pid") != os.getpid()
            or owner.get("pid") != os.getpid()
            or owner.get("token") != intent.get("owner", {}).get("token")):
        raise ContractError("Checkpoint writer does not own the declared stage and workspace lock")
    if (intent.get("algorithm") != contract.get("algorithm")
            or intent.get("interface_hash") != contract.get("interface_hash")
            or intent.get("config_hash") != canonical_hash(read_json(root / "config.json"))
            or intent.get("config_hash") != canonical_hash(intent.get("config"))
            or any(parent not in contract.get("dependencies", []) for parent in intent.get("dependencies", []))):
        raise ContractError("Checkpoint contract conflicts with stage intent")
    training = {"contract": contract, "training_signature": canonical_hash(contract)}
    if intent.get("training") is None:
        intent["training"] = training
        atomic_json(path, intent)
    elif intent["training"] != training:
        raise ContractError("Checkpoint contract cannot change within a stage")
    return file_hash(path)


def save_checkpoint(root: Path, contract: dict, *, model, optimizer, epoch: int, updates: int, target_epochs: int,
                    best_state, best_metric, wait: int, log: list, generator=None,
                    numpy_generator=None) -> dict:
    """Commit only completed epochs; a partial epoch is replayed after cancellation.

    The sidecar detects interrupted writes/corruption even for unpublished failed
    stages, whose artifacts have no publication manifest yet.
    """
    try:
        intent_digest = bind_checkpoint_intent(root, contract)
        head = _head(root)
    except OSError as exc:
        raise CheckpointWriteError("read or bind checkpoint stage", root, exc) from exc
    payload = {"format_version": FORMAT_VERSION, "training_signature": canonical_hash(contract),
               "contract": contract, "model_state": model.state_dict(),
               "optimizer_state": optimizer.state_dict(), "epoch": epoch, "updates": updates,
               "best_state": best_state, "best_metric": best_metric, "wait": wait, "log": log,
               "rng_state": rng_snapshot(generator, numpy_generator)}
    generation = f"checkpoint-{uuid.uuid4().hex}"
    target = root / (generation + ".pt")
    sidecar = checkpoint_sidecar(target)
    operation, destination = "write generation", target
    try:
        # 数据与旁注均使用独占新建。即使进程在 fsync 后退出，也不会覆盖上一个完整 epoch。
        with target.open("xb") as stream:
            try:
                torch.save(payload, stream)
            except RuntimeError as serialization_error:
                # Torch 的 ZIP 写入器有时在结束记录时报 RuntimeError，保留底层文件错误分类。
                underlying = serialization_error.__context__
                cause = underlying if isinstance(underlying, OSError) else serialization_error
                raise CheckpointWriteError("serialize checkpoint generation", target, cause) from serialization_error
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_hash(target)
        metadata = {"format_version": FORMAT_VERSION, "generation": generation, "sha256": digest,
                    "training_signature": payload["training_signature"], "algorithm": contract["algorithm"],
                    "epoch": epoch, "updates": updates, "target_epochs": target_epochs, "stage_intent_sha256": intent_digest}
        operation, destination = "write generation metadata", sidecar
        with sidecar.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        # 使用恢复端的只读反序列化和计数检查验明刚写入的实际字节，再发布指针。
        operation, destination = "verify checkpoint generation", target
        try:
            read_verified_checkpoint(target, contract, target_epochs, digest)
        except ContractError as exc:
            if isinstance(exc.__cause__, OSError):
                raise CheckpointWriteError(operation, destination, exc.__cause__) from exc
            raise
        operation, destination = "commit checkpoint pointer", root / HEAD
        _publish_pointer(destination, {"format_version": 2, "current": generation,
                                      "previous": head["current"] if head else None})
    except CheckpointWriteError:
        raise
    except OSError as exc:
        # 若文件系统在替换生效后才报告错误，保留这代文件，让恢复端按实际指针核验。
        raise CheckpointWriteError(operation, destination, exc) from exc
    if head and head["previous"]:
        # 提交后只回收刚淘汰的那一代，绝不通过扫描猜测可恢复版本。
        # 删除受占用的旧文件失败不影响已持久化的两代。
        retired = root / (head["previous"] + ".pt")
        for path in (retired, checkpoint_sidecar(retired)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return {"checkpoint": str(root.parent.parent / "failed" / root.name / target.name),
            "checkpoint_sha256": digest, "target_epochs": target_epochs, "checkpoint_status": "committed_in_staging"}


def read_verified_checkpoint(path: Path, contract: dict, epochs: int | None = None,
                             expected_sha256: str | None = None) -> tuple[dict, str]:
    """Validate bytes and numerical state; caller must authorize the path and ownership.

    Recovery calls this while holding the workspace mutex. Normal continuation
    additionally confines the path to failed or published artifact directories.
    """
    try:
        sidecar_file = checkpoint_sidecar(path)
        if any(item.resolve() != item.absolute() for item in (path, sidecar_file, path.with_name("stage.intent.json"))):
            raise ContractError("Checkpoint or metadata path escapes its stage")
        sidecar = read_json(sidecar_file)
        if path.name != "latest.pt" and sidecar.get("generation") != path.stem:
            raise ContractError("Checkpoint generation metadata differs from its filename")
        if "generation" in sidecar and not isinstance(sidecar.get("stage_intent_sha256"), str):
            raise ContractError("Checkpoint generation is not bound to its stage intent")
        if path.name == "latest.pt" and "generation" in sidecar and _head(path.parent) is None:
            raise ContractError("Checkpoint generation snapshot has no committed pointer")
        if "generation" in sidecar and (type(sidecar.get("target_epochs")) is not int
                or type(sidecar.get("epoch")) is not int or sidecar["target_epochs"] < sidecar["epoch"]):
            raise ContractError("Invalid checkpoint target epoch budget")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise ContractError("Resume checkpoint differs from the preflight SHA-256")
        if sidecar.get("format_version") != FORMAT_VERSION or sidecar.get("sha256") != digest:
            raise ContractError("Resume checkpoint integrity check failed")
        if ("stage_intent_sha256" in sidecar
                and sidecar["stage_intent_sha256"] != file_hash(path.with_name("stage.intent.json"))):
            raise ContractError("Resume checkpoint stage intent integrity check failed")
        signature = canonical_hash(contract)
        if sidecar.get("training_signature") != signature:
            raise ContractError("Resume checkpoint dataset, model, runtime or training configuration differs")
        # Deserialize exactly the bytes that were hashed, even if the path changes meanwhile.
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if (checkpoint.get("format_version") != FORMAT_VERSION
                or checkpoint.get("training_signature") != signature
                or canonical_hash(checkpoint["contract"]) != signature):
            raise ContractError("Resume checkpoint contract differs")
        for counter in ("epoch", "updates", "wait"):
            if type(checkpoint[counter]) is not int or checkpoint[counter] < 0:
                raise ContractError(f"Invalid checkpoint {counter}")
        if epochs is not None and checkpoint["epoch"] > epochs:
            raise ContractError("Requested total epochs is below the checkpoint's completed epoch")
        if (not isinstance(checkpoint["log"], list) or len(checkpoint["log"]) != checkpoint["epoch"]
                or [event.get("epoch") for event in checkpoint["log"]] != list(range(1, checkpoint["epoch"] + 1))):
            raise ContractError("Checkpoint epoch/log counters are inconsistent")
        if checkpoint["epoch"] and checkpoint["best_state"] is None:
            raise ContractError("Checkpoint has no best validation state")
        for key in ("model_state", "optimizer_state", "best_state", "rng_state", "best_metric"):
            if key not in checkpoint:
                raise ContractError(f"Checkpoint is missing {key}")
        if (sidecar.get("algorithm") != contract.get("algorithm")
                or sidecar.get("epoch") != checkpoint["epoch"] or sidecar.get("updates") != checkpoint["updates"]):
            raise ContractError("Checkpoint sidecar counters or algorithm differ")
        for state in (checkpoint["model_state"], checkpoint["best_state"]):
            if state is not None and (not isinstance(state, dict) or not state
                    or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                           for value in state.values())):
                raise ContractError("Checkpoint model weights are invalid or non-finite")
        optimizer = checkpoint["optimizer_state"]
        if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict)
                or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]):
            raise ContractError("Checkpoint optimizer structure is invalid")
        for group in optimizer["state"].values():
            if not isinstance(group, dict) or any(isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all()
                                                  for tensor in group.values()):
                raise ContractError("Checkpoint optimizer state is invalid or non-finite")
        return checkpoint, digest
    except ContractError:
        raise
    except Exception as exc:
        # weights_only=True is deliberate: never fall back to arbitrary pickle execution.
        raise ContractError(f"Invalid or unsupported resume checkpoint: {exc}") from exc


def authorized_checkpoint_path(workspace: Workspace, value: str | Path) -> Path:
    """预检与加载共用工作区边界及已提交代号检查，避免先读取无关路径再报错。"""
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else workspace.path / candidate).resolve()
    try:
        relative = path.relative_to(workspace.path)
    except ValueError as exc:
        raise ContractError("Resume checkpoint must belong to this workspace") from exc
    if (len(relative.parts) != 3 or relative.parts[0] not in {"failed", "failures", "artifacts"}
            or path.name != "latest.pt" and (path.suffix != ".pt" or not GENERATION.fullmatch(path.stem))):
        raise ContractError("Resume requires a committed checkpoint from an artifact or failed stage")
    if not path.is_file():
        raise ContractError(f"Resume checkpoint not found: {path}")
    sidecar = checkpoint_sidecar(path)
    if sidecar.resolve() != sidecar.absolute():
        raise ContractError("Checkpoint metadata path escapes its stage")
    head = _head(path.parent)
    if path.name != "latest.pt" and (not head or path.stem not in (head["current"], head["previous"])):
        raise ContractError("Requested checkpoint generation is not committed")
    return path


def load_checkpoint(workspace: Workspace, value: str, contract: dict, epochs: int,
                    expected_sha256: str | None = None) -> dict:
    path = authorized_checkpoint_path(workspace, value)
    relative = path.relative_to(workspace.path)
    head = _head(path.parent)
    checkpoint, digest = read_verified_checkpoint(path, contract, epochs, expected_sha256)
    if head and path.name == "latest.pt":
        selected = path.with_name(head["current"] + ".pt")
        if file_hash(selected) != digest or read_json(checkpoint_sidecar(selected)) != read_json(checkpoint_sidecar(path)):
            raise ContractError("Checkpoint snapshot differs from its committed generation")
    if relative.parts[0] == "artifacts":
        row = workspace.manifest["artifacts"].get(relative.parts[1])
        if not row or row["status"] == "stale":
            raise ContractError("Resume checkpoint artifact is unindexed or stale")
        metadata = read_json(path.parent / "artifact.json")
        if (canonical_hash(metadata) != row["metadata_hash"]
                or metadata["files"].get(path.name) != digest
                or metadata["files"].get(checkpoint_sidecar(path).name) != file_hash(checkpoint_sidecar(path))
                or (HEAD in metadata["files"]) != bool(head)
                or (head and metadata["files"][HEAD] != file_hash(path.parent / HEAD))):
            raise ContractError("Published resume checkpoint was changed")
    return checkpoint


def restore_training(checkpoint, model, optimizer, generator=None, numpy_generator=None) -> None:
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("non-finite model state")
        for state in optimizer.state.values():
            if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("non-finite optimizer state")
        restore_rng(checkpoint["rng_state"], generator, numpy_generator)
    except ContractError:
        raise
    except (TypeError, ValueError, RuntimeError, KeyError) as exc:
        raise ContractError(f"Checkpoint state is incompatible: {exc}") from exc
