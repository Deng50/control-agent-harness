"""Validated, atomic epoch-boundary checkpoints for numerical training workers."""
from __future__ import annotations

import os
import hashlib
import io
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import scipy
import torch

from .artifacts import Workspace, atomic_json, file_hash, read_json
from .contracts import ContractError, canonical_hash


FORMAT_VERSION = 1


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


def save_checkpoint(root: Path, contract: dict, *, model, optimizer, epoch: int, updates: int,
                    best_state, best_metric, wait: int, log: list, generator=None,
                    numpy_generator=None) -> None:
    """Commit only completed epochs; a partial epoch is replayed after cancellation.

    The sidecar detects interrupted writes/corruption even for unpublished failed
    stages, whose artifacts have no publication manifest yet.
    """
    intent_digest = bind_checkpoint_intent(root, contract)
    payload = {"format_version": FORMAT_VERSION, "training_signature": canonical_hash(contract),
               "contract": contract, "model_state": model.state_dict(),
               "optimizer_state": optimizer.state_dict(), "epoch": epoch, "updates": updates,
               "best_state": best_state, "best_metric": best_metric, "wait": wait, "log": log,
               "rng_state": rng_snapshot(generator, numpy_generator)}
    target = root / "latest.pt"
    temporary = root / f".latest.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        atomic_json(root / "latest.checkpoint.json", {"format_version": FORMAT_VERSION,
                    "sha256": file_hash(target), "training_signature": payload["training_signature"],
                    "algorithm": contract["algorithm"], "epoch": epoch, "updates": updates,
                    "stage_intent_sha256": intent_digest})
    finally:
        temporary.unlink(missing_ok=True)


def read_verified_checkpoint(path: Path, contract: dict, epochs: int | None = None,
                             expected_sha256: str | None = None) -> tuple[dict, str]:
    """Validate bytes and numerical state; caller must authorize the path and ownership.

    Recovery calls this while holding the workspace mutex. Normal continuation
    additionally confines the path to failed or published artifact directories.
    """
    try:
        sidecar = read_json(path.with_name("latest.checkpoint.json"))
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
        return checkpoint, digest
    except ContractError:
        raise
    except Exception as exc:
        # weights_only=True is deliberate: never fall back to arbitrary pickle execution.
        raise ContractError(f"Invalid or unsupported resume checkpoint: {exc}") from exc


def load_checkpoint(workspace: Workspace, value: str, contract: dict, epochs: int,
                    expected_sha256: str | None = None) -> dict:
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else workspace.path / candidate).resolve()
    try:
        relative = path.relative_to(workspace.path)
    except ValueError as exc:
        raise ContractError("Resume checkpoint must belong to this workspace") from exc
    if (len(relative.parts) != 3 or relative.parts[0] not in {"failed", "failures", "artifacts"}
            or path.name != "latest.pt"):
        raise ContractError("Resume requires an artifact or failed-stage latest.pt checkpoint")
    if not path.is_file():
        raise ContractError(f"Resume checkpoint not found: {path}")
    checkpoint, digest = read_verified_checkpoint(path, contract, epochs, expected_sha256)
    if relative.parts[0] == "artifacts":
        row = workspace.manifest["artifacts"].get(relative.parts[1])
        if not row or row["status"] == "stale":
            raise ContractError("Resume checkpoint artifact is unindexed or stale")
        metadata = read_json(path.parent / "artifact.json")
        if (canonical_hash(metadata) != row["metadata_hash"]
                or metadata["files"].get("latest.pt") != digest
                or metadata["files"].get("latest.checkpoint.json") != file_hash(path.with_name("latest.checkpoint.json"))):
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
