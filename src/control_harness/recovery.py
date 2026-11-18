"""Explicit recovery of unpublished checkpoints left by an OS-confirmed dead worker."""
from __future__ import annotations

import os
import platform
import re
import time
from pathlib import Path

from .artifacts import Workspace, WorkspaceBusy, atomic_json, confined, file_hash, read_json
from .checkpoints import read_verified_checkpoint, training_contract
from .contracts import ContractError, canonical_hash
from .locking import process_alive
from .runtime import OperationContext


TRAINING_KINDS = {"corrected_model": "residual_training", "imitation_policy": "expert_imitation",
                  "optimized_policy": "physical_policy_optimization", "supervised_model": "supervised_training",
                  "pinn_model": "conditional_ode_pinn", "external_pgac_candidate": "external_physical_policy_optimization"}


def _checkpoint_algorithm(kind):
    # External imitation composes S05 training and then exports/projects it.
    # Only the real inner supervised_model stage owns optimizer/RNG state.
    # An interrupted outer package is not an optimizer checkpoint in disguise.
    if kind == "external_imitation_policy":
        raise ContractError("External imitation wrapper has no optimizer checkpoint; recover its actual supervised_model/supervised_training checkpoint stage")
    return TRAINING_KINDS.get(kind)


def _plain_directory(workspace: Workspace, relative: str) -> Path:
    lexical = workspace.path / relative
    resolved = confined(workspace.path, relative)
    # Reject aliases even when their target happens to remain inside the workspace.
    if resolved != lexical or lexical.is_symlink():
        raise ContractError(f"Interrupted-stage recovery refuses filesystem aliases: {relative}")
    return resolved


def _validate_unpublished(workspace: Workspace, stage_id: str, root: Path) -> None:
    manifest = workspace.manifest
    if stage_id in manifest["artifacts"] or (workspace.path / "artifacts" / stage_id).exists():
        raise ContractError("Stage has a published or publication-pending artifact; refusing recovery")
    for row in manifest["artifacts"].values():
        referenced = confined(workspace.path, row["path"])
        if referenced == root or referenced.is_relative_to(root):
            raise ContractError("Stage is referenced by the artifact manifest; refusing recovery")
    for path in root.rglob("*"):
        if path.resolve() != path or path.is_symlink():
            raise ContractError("Interrupted-stage contents contain a filesystem alias")
        if path.name == "artifact.json":
            raise ContractError("Stage contains publication metadata; refusing recovery")


def recover_interrupted_stage(workspace: Workspace, stage_id: str,
                              context: OperationContext | None = None) -> dict:
    """Archive one validated .staging checkpoint; never resume or terminate a process.

    The OS must positively confirm the original owner is dead. A timestamp,
    missing host registry entry or caller-provided cleanup flag is insufficient.
    After this operation the normal training API accepts the returned failed/
    checkpoint, subject to its existing dependency/configuration contract.
    """
    context = context or OperationContext()
    context.check()
    if not isinstance(stage_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*-[0-9a-f]{12}", stage_id):
        raise ContractError("stage_id must be an exact unpublished stage directory name")
    try:
        with workspace.lock("recover_interrupted_stage"):
            context.check()
            _plain_directory(workspace, ".staging")
            _plain_directory(workspace, "failed")
            root = _plain_directory(workspace, f".staging/{stage_id}")
            destination = _plain_directory(workspace, f"failed/{stage_id}")
            if not root.is_dir():
                raise ContractError("Interrupted stage was not found under .staging")
            if destination.exists():
                raise ContractError("Failed-stage destination already exists; refusing overwrite")
            _validate_unpublished(workspace, stage_id, root)
            intent = read_json(root / "stage.intent.json")
            if (intent.get("format_version") != 1 or intent.get("stage_id") != stage_id
                    or Path(intent.get("workspace", "")).resolve() != workspace.path
                    or intent.get("project_id") != workspace.manifest["project_id"]
                    or not stage_id.startswith(f"{intent.get('kind')}-")
                    or _checkpoint_algorithm(intent.get("kind")) != intent.get("algorithm")):
                raise ContractError("Interrupted stage intent identity or algorithm is unsupported")
            owner = intent.get("owner", {})
            if (owner.get("host") != platform.node()
                    or not isinstance(owner.get("token"), str)
                    or not re.fullmatch(r"[0-9a-f]{32}", owner["token"])):
                raise ContractError("Interrupted stage owner identity cannot be verified on this host")
            if process_alive(owner.get("pid")) is not False:
                raise WorkspaceBusy("Interrupted stage owner is alive or its exit cannot be verified")
            if (intent.get("config_hash") != canonical_hash(intent.get("config"))
                    or intent.get("config_hash") != canonical_hash(read_json(root / "config.json"))):
                raise ContractError("Interrupted stage configuration integrity check failed")
            training = intent.get("training")
            if not isinstance(training, dict) or not isinstance(training.get("contract"), dict):
                raise ContractError("Interrupted stage has no bound training checkpoint contract")
            contract = training["contract"]
            signature = canonical_hash(contract)
            if (training.get("training_signature") != signature
                    or contract.get("algorithm") != intent["algorithm"]
                    or contract.get("interface_hash") != intent.get("interface_hash")
                    or not isinstance(intent.get("dependencies"), list)
                    or any(parent not in contract.get("dependencies", []) for parent in intent["dependencies"])):
                raise ContractError("Interrupted stage checkpoint contract conflicts with its intent")
            # Resolving the exact dependency closure rejects changed, stale or rejected data/models.
            current = training_contract(workspace, contract["algorithm"],
                                        [parent["id"] for parent in contract["dependencies"]],
                                        contract["config"], contract["interface_hash"])
            if canonical_hash(current) != signature:
                raise ContractError("Interrupted checkpoint dependencies or numerical runtime have changed")
            checkpoint_path = root / "latest.pt"
            sidecar = read_json(root / "latest.checkpoint.json")
            if sidecar.get("stage_intent_sha256") != file_hash(root / "stage.intent.json"):
                raise ContractError("Interrupted checkpoint is not bound to this stage intent")
            checkpoint, digest = read_verified_checkpoint(checkpoint_path, contract)
            context.progress("interrupted_stage_recovery", stage_id=stage_id, epoch=checkpoint["epoch"],
                             optimizer_updates=checkpoint["updates"], owner_exit_verified=True)
            # Recheck immediately before mutation, under the same kernel-held workspace mutex.
            if process_alive(owner.get("pid")) is not False:
                raise WorkspaceBusy("Interrupted stage owner exit is no longer verifiable")
            _validate_unpublished(workspace, stage_id, root)
            if (file_hash(checkpoint_path) != digest
                    or file_hash(root / "stage.intent.json") != sidecar["stage_intent_sha256"]):
                raise ContractError("Interrupted checkpoint changed during recovery validation")
            result = {"status": "archived", "stage_id": stage_id, "workspace": str(workspace.path),
                      "algorithm": contract["algorithm"], "checkpoint_kind": intent["kind"], "epoch": checkpoint["epoch"],
                      "optimizer_updates": checkpoint["updates"], "training_signature": signature,
                      "checkpoint": str(destination / "latest.pt"), "checkpoint_sha256": digest,
                      "failed_path": str(destination)}
            receipt = {"format_version": 1, **result, "source": str(root), "original_owner": owner,
                       "owner_verification": "os_confirmed_process_exit", "recovered_at": time.time(),
                       "recovery_pid": os.getpid(), "project_id": intent["project_id"]}
            atomic_json(root / "interrupted-stage-recovery.json", receipt)
            atomic_json(root / "failure.json", {"type": "InterruptedWorker", "message":
                        "Unpublished checkpoint archived after the OS confirmed its owner exited",
                        "recovery": "interrupted-stage-recovery.json"})
            destination.parent.mkdir(exist_ok=True)
            root.rename(destination)
            return result
    except (ContractError, WorkspaceBusy):
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ContractError(f"Interrupted stage cannot be safely recovered: {exc}") from exc
