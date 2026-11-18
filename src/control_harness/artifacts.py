"""Immutable stage outputs with transactional publication and dependency checks."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import Iterator

from .contracts import ContractError, ModelInterface, SCHEMA_VERSION, canonical_hash, parse_interface
from .locking import kernel_mutex, process_alive


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confined(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ContractError(f"Path escapes artifact root: {relative}")
    return target


def activate_artifact(manifest: dict, kind: str, artifact_id: str) -> None:
    previous = manifest["active"].get(kind)
    manifest["active"][kind] = artifact_id
    obsolete = {previous} if previous and previous != artifact_id else set()
    changed = True
    while changed:
        changed = False
        for item_id, row in manifest["artifacts"].items():
            if item_id == artifact_id or item_id in obsolete:
                continue
            if any(parent in obsolete for parent in row["parents"]):
                obsolete.add(item_id)
                row.update(status="stale", stale_reason=f"Upstream {kind} replaced by {artifact_id}")
                changed = True
    for key, active in list(manifest["active"].items()):
        if manifest["artifacts"][active]["status"] == "stale":
            del manifest["active"][key]


class WorkspaceBusy(RuntimeError):
    pass


class Workspace:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        self.lock_path = self.path / ".operation.lock"
        self._lock_thread = None
        self.workflow_stage = None
        if not self.manifest_path.exists():
            with self.lock("initialize"):
                if not self.manifest_path.exists():
                    atomic_json(self.manifest_path, {"schema_version": SCHEMA_VERSION,
                        "project_id": uuid.uuid4().hex, "created_at": time.time(), "active": {}, "artifacts": {}})
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise ContractError("Unsupported workspace schema")

    @property
    def manifest(self) -> dict:
        return read_json(self.manifest_path)

    @contextlib.contextmanager
    def lock(self, operation: str) -> Iterator[None]:
        # A workflow owns the same lock across all stages, including their publication.
        # Reentrancy is confined to this Workspace instance and the owning thread.
        if self._lock_thread == threading.get_ident():
            yield
            return
        token = uuid.uuid4().hex
        mutex = kernel_mutex(self.path / ".workspace.mutex")
        try:
            mutex.__enter__()
        except OSError as exc:
            raise WorkspaceBusy(f"Workspace has a live operation lock: {self.path}") from exc
        try:
            if self.lock_path.exists():
                owner = read_json(self.lock_path)
                if process_alive(owner.get("pid")) is not False:
                    raise WorkspaceBusy(f"Previous lock owner is alive or cannot be verified: {self.lock_path}")
                # Kernel mutex excludes concurrent recovery and new writers. This
                # handles an observed dead process, not an aged metadata file.
                atomic_json(self.path / "recovery" / f"lock-{uuid.uuid4().hex}.json", owner)
                self.lock_path.unlink()
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "operation": operation, "token": token,
                           "created_at": time.time()}, stream)
            self._lock_thread = threading.get_ident()
            yield
        finally:
            self._lock_thread = None
            try:
                if self.lock_path.exists() and read_json(self.lock_path).get("token") == token:
                    self.lock_path.unlink()
            finally:
                mutex.__exit__(None, None, None)

    def resolve(self, artifact_id: str, *, kind: str | None = None, allow_stale: bool = False,
                allow_rejected: bool = False, _visiting: frozenset | None = None) -> tuple[Path, dict]:
        visiting = _visiting or frozenset()
        if artifact_id in visiting:
            raise ContractError(f"Cyclic artifact dependency: {artifact_id}")
        visiting = visiting | {artifact_id}
        index = self.manifest["artifacts"]
        if artifact_id not in index:
            raise ContractError(f"Unknown artifact: {artifact_id}")
        row = index[artifact_id]
        if row["status"] == "rejected" and not allow_rejected:
            raise ContractError(f"Artifact failed acceptance and cannot be consumed: {artifact_id}")
        if row["status"] == "stale" and not allow_stale:
            raise ContractError(f"Artifact is stale: {artifact_id}: {row.get('stale_reason', '')}")
        root = confined(self.path, row["path"])
        meta = read_json(root / "artifact.json")
        if canonical_hash(meta) != row["metadata_hash"]:
            raise ContractError(f"Artifact metadata changed: {artifact_id}")
        if kind and meta["kind"] != kind:
            raise ContractError(f"Expected {kind}, received {meta['kind']}")
        for relative, expected in meta["files"].items():
            file = confined(root, relative)
            if not file.is_file() or file_hash(file) != expected:
                raise ContractError(f"Artifact file missing or changed: {file}")
        for dependency in meta["parents"]:
            _, parent = self.resolve(dependency["id"], allow_stale=allow_stale, allow_rejected=allow_rejected, _visiting=visiting)
            if canonical_hash(parent) != dependency["metadata_hash"]:
                raise ContractError("Dependency version mismatch")
        return root, meta

    def active(self, kind: str) -> str:
        artifact_id = self.manifest["active"].get(kind)
        if not artifact_id:
            raise ContractError(f"No active {kind} artifact")
        self.resolve(artifact_id, kind=kind)
        return artifact_id

    def import_artifact(self, source_path: str | Path) -> str:
        """Stage a verified dependency closure, then publish it with one manifest commit."""
        source_root = Path(source_path).expanduser().resolve()
        if not (source_root / "artifact.json").is_file():
            raise ContractError("Select an artifact directory containing artifact.json")
        source_workspace_root = source_root.parent.parent
        if not (source_workspace_root / "manifest.json").is_file():
            raise ContractError("Source workspace manifest is required to verify all dependencies")
        source = Workspace(source_workspace_root)
        artifact_id = read_json(source_root / "artifact.json")["id"]
        resolved_root, requested = source.resolve(artifact_id)
        if resolved_root != source_root:
            raise ContractError("Source artifact directory does not match its workspace manifest")
        if source.path == self.path:
            return artifact_id
        ordered, visiting = {}, set()

        def visit(item_id):
            if item_id in ordered:
                return
            if item_id in visiting:
                raise ContractError(f"Cyclic artifact dependency: {item_id}")
            if not isinstance(item_id, str) or Path(item_id).name != item_id or item_id in {"", ".", ".."}:
                raise ContractError("Imported artifact ids must be single directory names")
            visiting.add(item_id)
            root, meta = source.resolve(item_id)
            if meta["id"] != item_id or meta["validation"].get("status") != "verified":
                raise ContractError("Imported artifact metadata identity or validation is inconsistent")
            for parent in meta["parents"]:
                visit(parent["id"])
            ordered[item_id] = (root, meta)
            visiting.remove(item_id)

        def verify_directory(root, metadata):
            if not root.is_dir() or canonical_hash(read_json(confined(root, "artifact.json"))) != canonical_hash(metadata):
                raise ContractError(f"Unindexed artifact metadata conflicts with source: {root}")
            expected_files = set(metadata["files"]) | {"artifact.json"}
            actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
            if actual_files != expected_files:
                raise ContractError(f"Unindexed artifact file inventory conflicts with source: {root}")
            for relative, expected in metadata["files"].items():
                if Path(relative).is_absolute() or relative == "artifact.json":
                    raise ContractError("Imported artifact file names must be relative and exclude artifact.json")
                file = confined(root, relative)
                if not file.is_file() or file_hash(file) != expected:
                    raise ContractError(f"Imported artifact file missing or changed: {file}")

        visit(artifact_id)
        with self.lock("import_artifact"):
            manifest = self.manifest
            transaction_id = f"import-{uuid.uuid4().hex}"
            transaction = confined(self.path, f".staging/{transaction_id}")
            artifact_base = confined(self.path, "artifacts")

            def destination_for(item_id):
                destination = confined(artifact_base, item_id)
                # An existing alias (including a dangling link) is content owned
                # by someone else, not an interrupted publication we may adopt.
                if destination != artifact_base / item_id or (artifact_base / item_id).is_symlink():
                    raise ContractError(f"Imported artifact destination is a filesystem alias: {item_id}")
                return destination

            staged, published, adopted = {}, [], []
            commit_attempted = False
            transaction.mkdir(parents=True)
            try:
                # Nothing enters artifacts/ until every missing dependency has a
                # complete, hash-checked copy and every existing id has been checked.
                for item_id, (root, meta) in ordered.items():
                    if item_id in manifest["artifacts"]:
                        _, existing = self.resolve(item_id)
                        if canonical_hash(existing) != canonical_hash(meta):
                            raise ContractError("Imported artifact id conflicts with local content")
                        continue
                    destination = destination_for(item_id)
                    if destination.exists():
                        verify_directory(destination, meta)
                        adopted.append(item_id)
                    else:
                        pending = confined(transaction, item_id)
                        pending.mkdir(parents=True)
                        for relative, expected in meta["files"].items():
                            if Path(relative).is_absolute() or relative == "artifact.json":
                                raise ContractError("Imported artifact file names must be relative and exclude artifact.json")
                            target = confined(pending, relative)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(confined(root, relative), target)
                            if file_hash(target) != expected:
                                raise ContractError("Source artifact changed during import")
                        atomic_json(pending / "artifact.json", meta)
                        verify_directory(pending, meta)
                        staged[item_id] = pending
                    manifest["artifacts"][item_id] = {
                        "path": destination.relative_to(self.path).as_posix(), "kind": meta["kind"], "status": "verified",
                        "metadata_hash": canonical_hash(meta), "parents": [p["id"] for p in meta["parents"]],
                        "metadata_file_sha256": file_hash((staged.get(item_id) or destination) / "artifact.json"),
                        "imported_from_project": source.manifest["project_id"], "imported_at": time.time()}
                for item_id, (root, meta) in ordered.items():
                    _, current = source.resolve(item_id)
                    if canonical_hash(current) != canonical_hash(meta):
                        raise ContractError("Source artifact metadata changed during import")
                artifact_base.mkdir(exist_ok=True)
                for item_id, pending in staged.items():
                    destination = destination_for(item_id)
                    if destination.exists() or destination.is_symlink():
                        raise ContractError(f"Artifact appeared during publication; refusing overwrite: {destination}")
                    pending.rename(destination)
                    published.append(item_id)
                for item_id in adopted:
                    verify_directory(destination_for(item_id), ordered[item_id][1])
                if requested.get("activation", "kind") == "kind":
                    activate_artifact(manifest, requested["kind"], artifact_id)
                commit_attempted = True
                atomic_json(self.manifest_path, manifest)
            except BaseException as exc:
                cleanup_errors = []
                # An I/O wrapper can report failure after replacement committed.
                # Never withdraw directories already referenced by that manifest.
                committed = False
                if commit_attempted:
                    try:
                        committed = canonical_hash(self.manifest) == canonical_hash(manifest)
                    except (OSError, ValueError) as check_error:
                        cleanup_errors.append(f"Manifest outcome could not be read: {check_error}")
                        committed = None
                if committed is False:
                    for item_id in reversed(published):
                        try:
                            confined(artifact_base, item_id).rename(confined(transaction, item_id))
                        except OSError as cleanup_error:
                            cleanup_errors.append(f"Could not withdraw {item_id}: {cleanup_error}")
                diagnosis = {"type": type(exc).__name__, "message": str(exc), "source": str(source_root),
                             "requested_artifact": artifact_id, "closure": list(ordered), "published": published,
                             "adopted_existing": adopted, "manifest_committed": committed, "cleanup_errors": cleanup_errors}
                try:
                    atomic_json(transaction / "failure.json", diagnosis)
                    failure = confined(self.path, f"failed/{transaction_id}")
                    failure.parent.mkdir(exist_ok=True)
                    transaction.rename(failure)
                except OSError as diagnosis_error:
                    raise RuntimeError(f"Artifact import failed: {exc}; diagnostic staging retained at {transaction}: {diagnosis_error}") from exc
                if cleanup_errors:
                    raise RuntimeError(f"Artifact import failed: {exc}; inspect {failure}/failure.json before retry") from exc
                raise
            # Publication is committed. A leftover empty transaction directory is
            # harmless; removal failure must not misreport a committed import.
            try:
                shutil.rmtree(transaction)
            except OSError:
                pass
        self.resolve(artifact_id)
        return artifact_id

    @contextlib.contextmanager
    def stage(self, kind: str, interface, parents: list[str], config: dict, *, compatible_parents: bool = True) -> Iterator["Stage"]:
        if type(compatible_parents) is not bool:
            raise ContractError("compatible_parents must be an explicit boolean")
        with self.lock(kind):
            dependencies = []
            for parent in parents:
                _, meta = self.resolve(parent)
                if compatible_parents:
                    interface.compatible(parse_interface(meta["interface"]))
                dependencies.append({"id": parent, "metadata_hash": canonical_hash(meta)})
            stage = Stage(self, kind, interface, dependencies, config)
            try:
                yield stage
                if not stage.committed:
                    raise RuntimeError("Stage ended without validated publication")
            except BaseException as exc:
                if stage.root.exists() and not stage.committed:
                    # Publication can fail after a directory or manifest replacement.
                    # Never withdraw a potentially published artifact as a training failure.
                    if ((stage.root / "artifact.json").exists()
                            or stage.id in self.manifest["artifacts"]
                            or stage.root.parent != self.path / ".staging"):
                        atomic_json(self.path / "recovery" / f"publication-{stage.id}.json",
                                    {"type": type(exc).__name__, "message": str(exc),
                                     "stage_id": stage.id, "retained_path": str(stage.root),
                                     "requires_publication_review": True})
                    else:
                        atomic_json(stage.root / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
                        failure = self.path / "failed" / stage.id
                        failure.parent.mkdir(exist_ok=True)
                        stage.root.rename(failure)
                raise


class Stage:
    def __init__(self, workspace: Workspace, kind: str, interface: ModelInterface, parents: list[dict], config: dict):
        self.workspace, self.kind, self.interface = workspace, kind, interface
        self.parents, self.config = parents, config
        self.workflow_stage = workspace.workflow_stage
        self.id = f"{kind}-{uuid.uuid4().hex[:12]}"
        self.root = workspace.path / ".staging" / self.id
        self.root.mkdir(parents=True)
        self.committed = False
        atomic_json(self.root / "config.json", config)
        owner = read_json(workspace.lock_path)
        if owner.get("pid") != os.getpid() or not owner.get("token"):
            raise WorkspaceBusy("Stage creation requires this process to own the workspace lock")
        algorithm = {"corrected_model": "residual_training", "imitation_policy": "expert_imitation",
                     "optimized_policy": "physical_policy_optimization", "supervised_model": "supervised_training",
                     "pinn_model": "conditional_ode_pinn", "external_pgac_candidate": "external_physical_policy_optimization"}.get(kind, kind)
        atomic_json(self.root / "stage.intent.json", {
            "format_version": 1, "stage_id": self.id, "kind": kind, "algorithm": algorithm,
            "workspace": str(workspace.path), "project_id": workspace.manifest["project_id"],
            "owner": {"pid": owner["pid"], "token": owner["token"], "host": platform.node()},
            "created_at": time.time(), "interface_hash": interface.fingerprint,
            "dependencies": parents, "config": config, "config_hash": canonical_hash(config),
            "workflow": self.workflow_stage, "training": None})

    def publish(self, validation: dict, *, preprocessing: dict | None = None, source: dict | None = None, activate: bool = True) -> str:
        if self.committed:
            raise RuntimeError("Stage already published")
        if validation.get("status") not in {"verified", "rejected"}:
            raise ContractError("Publication requires an explicit verified/rejected validation outcome")
        if type(activate) is not bool:
            raise ContractError("activate must be an explicit boolean")
        import numpy
        import scipy
        import torch
        files = {p.relative_to(self.root).as_posix(): file_hash(p)
                 for p in self.root.rglob("*") if p.is_file()}
        metadata = {"schema_version": SCHEMA_VERSION, "id": self.id, "kind": self.kind,
                    "activation": "kind" if activate else "none",
                    "created_at": time.time(), "interface": self.interface.to_dict(),
                    "interface_hash": self.interface.fingerprint, "parents": self.parents,
                    "files": files, "validation": validation, "preprocessing": preprocessing or {},
                    "source": source or {}, "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__}}
        if self.workflow_stage is not None:
            metadata["workflow"] = dict(self.workflow_stage)
        atomic_json(self.root / "artifact.json", metadata)
        destination = self.workspace.path / "artifacts" / self.id
        destination.parent.mkdir(exist_ok=True)
        self.root.rename(destination)
        self.root = destination
        manifest = self.workspace.manifest
        manifest["artifacts"][self.id] = {"path": destination.relative_to(self.workspace.path).as_posix(),
            "kind": self.kind, "status": validation["status"], "metadata_hash": canonical_hash(metadata),
            "metadata_file_sha256": file_hash(destination / "artifact.json"),
            "parents": [p["id"] for p in self.parents]}
        if validation["status"] == "verified" and activate:
            activate_artifact(manifest, self.kind, self.id)
        atomic_json(self.workspace.manifest_path, manifest)
        self.committed = True
        return self.id
