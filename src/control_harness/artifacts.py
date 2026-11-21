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


def record_cleanup_error(exc: BaseException, message: str) -> None:
    """使用独立字段保留次要异常，兼容没有 add_note 的 Python 3.10。"""
    if not hasattr(exc, "cleanup_errors"):
        exc.cleanup_errors = []
    exc.cleanup_errors.append(message)


def failure_fields(exc: BaseException) -> dict:
    """CLI 与工作流快照使用同一份现场事实，保留各自原有错误消息格式。"""
    fields = dict(getattr(exc, "checkpoint_details", {}))
    for attribute, key in (("stage_failure", "stage_failure"), ("cleanup_errors", "cleanup_errors"),
                           ("io_failure", "io_failure"), ("code", "error_code")):
        if hasattr(exc, attribute):
            fields[key] = getattr(exc, attribute)
    return fields


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
    except BaseException as exc:
        # 临时文件被占用时，保留替换操作本身的错误码，供调用端判定是否短暂重试。
        try:
            temp.unlink(missing_ok=True)
        except OSError as cleanup:
            record_cleanup_error(exc, f"Temporary file retained at {temp}: {cleanup}")
        raise


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
    # active 是默认选择。更换对象时撤下依赖旧对象的下游默认项；冻结的 ID、版本和验证记录
    # 继续代表原分支。真实文件变动、拒收和历史 stale 标记仍由 resolve 逐项检查。
    old_branch = {previous} if previous and previous != artifact_id else set()
    changed = True
    while changed:
        changed = False
        for item_id, row in manifest["artifacts"].items():
            if item_id == artifact_id or item_id in old_branch:
                continue
            if any(parent in old_branch for parent in row["parents"]):
                old_branch.add(item_id)
                changed = True
    for key, active in list(manifest["active"].items()):
        if active in old_branch or manifest["artifacts"][active]["status"] == "stale":
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
            original = sys.exc_info()[1]
            try:
                if self.lock_path.exists() and read_json(self.lock_path).get("token") == token:
                    self.lock_path.unlink()
            except (OSError, ValueError) as cleanup:
                if original is None:
                    raise
                record_cleanup_error(original, f"Workspace lock metadata retained at {self.lock_path}: {cleanup}")
            finally:
                active_error = sys.exc_info()[1] or original
                try:
                    mutex.__exit__(None, None, None)
                except OSError as cleanup:
                    if active_error is None:
                        raise
                    record_cleanup_error(active_error, f"Workspace mutex release failed: {cleanup}")

    def resolve(self, artifact_id: str, *, kind: str | None = None, allow_stale: bool = False,
                allow_rejected: bool = False) -> tuple[Path, dict]:
        verified = self.resolve_closure(artifact_id, allow_stale=allow_stale, allow_rejected=allow_rejected)
        root, meta = verified[artifact_id]
        if kind and meta["kind"] != kind:
            raise ContractError(f"Expected {kind}, received {meta['kind']}")
        return root, meta

    def resolve_closure(self, artifact_id: str, *, allow_stale=False, allow_rejected=False, restore_replaced=False) -> dict:
        """在同一份清单中核验闭包；多个父分支共享一个祖先时每个文件只散列一次。"""
        index, verified = self.manifest["artifacts"], {}
        self._verify_dependency(artifact_id, index, verified, frozenset(), allow_stale, allow_rejected, restore_replaced)
        return verified

    def _verify_dependency(self, artifact_id, index, verified, visiting, allow_stale, allow_rejected, restore_replaced):
        if artifact_id in verified:
            return
        if artifact_id in visiting:
            raise ContractError(f"Cyclic artifact dependency: {artifact_id}")
        visiting = visiting | {artifact_id}
        if artifact_id not in index:
            raise ContractError(f"Unknown artifact: {artifact_id}")
        row = index[artifact_id]
        if row["status"] == "rejected" and not allow_rejected:
            raise ContractError(f"Artifact failed acceptance and cannot be consumed: {artifact_id}")
        if row["status"] == "stale" and not allow_stale:
            # 只有显式跨工作区导入可恢复旧版“切换默认分支”标记。旧版也曾覆盖 rejected，
            # 因此仍须核对原始验证结论、全部文件和每一代父依赖的版本。
            reason = str(row.get("stale_reason", "")).split()
            replaced = (len(reason) == 5 and reason[0] == "Upstream" and reason[2:4] == ["replaced", "by"]
                        and index.get(reason[4], {}).get("kind") == reason[1])
            if not (restore_replaced and replaced):
                raise ContractError(f"Artifact is stale: {artifact_id}: {row.get('stale_reason', '')}")
        root = confined(self.path, row["path"])
        if not (root / "artifact.json").is_file():
            raise ContractError(f"Missing frozen artifact metadata: {artifact_id}; restore its source workspace dependency: {root}")
        meta = read_json(root / "artifact.json")
        if canonical_hash(meta) != row["metadata_hash"] or meta["id"] != artifact_id or row["kind"] != meta["kind"]:
            raise ContractError(f"Artifact metadata changed: {artifact_id}")
        if row["status"] not in {"verified", "rejected", "stale"} or ((row["status"] == "verified" or restore_replaced)
                and meta.get("validation", {}).get("status") != "verified"):
            raise ContractError(f"Artifact acceptance status differs from its manifest: {artifact_id}")
        for relative, expected in meta["files"].items():
            file = confined(root, relative)
            if not file.is_file() or file_hash(file) != expected:
                raise ContractError(f"Artifact file missing or changed: {file}")
        for dependency in meta["parents"]:
            self._verify_dependency(dependency["id"], index, verified, visiting, allow_stale, allow_rejected, restore_replaced)
            _, parent = verified[dependency["id"]]
            if canonical_hash(parent) != dependency["metadata_hash"]:
                raise ContractError(f"Dependency version mismatch: {artifact_id} requires {dependency['id']}@{dependency['metadata_hash']}")
        verified[artifact_id] = (root, meta)

    def active(self, kind: str) -> str:
        artifact_id = self.manifest["active"].get(kind)
        if not artifact_id:
            raise ContractError(f"No active {kind} artifact")
        self.resolve(artifact_id, kind=kind)
        return artifact_id

    def import_artifact(self, source_path: str | Path, *, expected_artifact_id=None,
                        expected_metadata_hash=None, expected_project_id=None) -> str:
        """Stage a verified dependency closure, then publish it with one manifest commit."""
        source_root = Path(source_path).expanduser().resolve()
        if not (source_root / "artifact.json").is_file():
            raise ContractError("Select an artifact directory containing artifact.json. 导入需要原工作区内完整的产物及父依赖，独立导出包不能恢复原始依赖。")
        source_workspace_root = source_root.parent.parent
        if not (source_workspace_root / "manifest.json").is_file():
            raise ContractError("Source workspace manifest is required to verify all dependencies")
        source = Workspace(source_workspace_root)
        artifact_id = read_json(source_root / "artifact.json")["id"]
        migration = source.path != self.path
        ordered = source.resolve_closure(artifact_id, restore_replaced=migration)
        resolved_root, requested = ordered[artifact_id]
        # 手工选目录仍兼容。网页交接同时锁住来源项目、选中 ID 和摘要，以便发现审阅之后的替换。
        for label, expected, actual in (("artifact_id", expected_artifact_id, artifact_id),
                ("metadata_hash", expected_metadata_hash, canonical_hash(requested)),
                ("project_id", expected_project_id, source.manifest["project_id"])):
            if expected is not None and expected != actual:
                raise ContractError(f"交接来源版本已变化：{label}，请重新检查所选原产物。")
        if resolved_root != source_root:
            raise ContractError("Source artifact directory does not match its workspace manifest")
        if source.path == self.path:
            return artifact_id
        for item_id, (_, meta) in ordered.items():
            if not isinstance(item_id, str) or Path(item_id).name != item_id or item_id in {"", ".", ".."}:
                raise ContractError("Imported artifact ids must be single directory names")
            if meta["id"] != item_id or meta["validation"].get("status") != "verified":
                raise ContractError("Imported artifact metadata identity or validation is inconsistent")

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
                current_closure = source.resolve_closure(artifact_id, restore_replaced=migration)
                for item_id, (_, meta) in ordered.items():
                    if item_id not in current_closure or canonical_hash(current_closure[item_id][1]) != canonical_hash(meta):
                        raise ContractError(f"Source artifact metadata changed during import: {item_id}")
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
                if not stage.committed:
                    stage.record_failure(exc)
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

    def record_failure(self, exc: BaseException) -> None:
        """即使诊断写入或目录归档失败，仍向宿主报告最初异常、真实位置及可验证的代号。"""
        details = {"stage_id": self.id, "archive_state": "retained", "retained_path": str(self.root),
                   "archive_errors": []}
        # 嵌套阶段已收集到的数值训练现场优先于外围包装阶段。
        if not hasattr(exc, "stage_failure"):
            exc.stage_failure = details
        errors = details["archive_errors"]
        try:
            published = self.workspace.path / "artifacts" / self.id
            if (published.exists() or (self.root / "artifact.json").exists() or self.root.parent != self.workspace.path / ".staging"
                    or self.id in self.workspace.manifest["artifacts"]):
                details["archive_state"] = "publication_review"
                if published.is_dir():
                    details["retained_path"] = str(published)
                atomic_json(self.workspace.path / "recovery" / f"publication-{self.id}.json",
                            {**details, "type": type(exc).__name__, "message": str(exc),
                             "requires_publication_review": True})
                return
            if not self.root.is_dir() or self.root.resolve() != self.root:
                errors.append("Stage directory disappeared or redirects elsewhere; its location requires review")
                details["archive_state"] = "publication_review"
                return
        except Exception as error:
            details["archive_state"] = "publication_review"
            errors.append(f"verify or record publication: {error}")
            return

        from .checkpoints import HEAD, snapshot_latest, verified_recovery_point
        try:
            atomic_json(self.root / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        except Exception as error:
            errors.append(f"write failure.json: {error}")
        try:
            destination = self.workspace.path / "failed" / self.id
            if destination.resolve() != destination or self.root.resolve() != self.root:
                raise ContractError("Failure archive refuses a filesystem alias")
            destination.parent.mkdir(exist_ok=True)
            if destination.exists():
                raise FileExistsError(f"Failed-stage destination already exists: {destination}")
            self.root.rename(destination)
            self.root = destination
            details.update(archive_state="archived", retained_path=str(destination))
        except Exception as error:
            errors.append(f"archive stage directory: {error}")
            # 有些文件系统封装在 rename 已经提交后才抛异常。只认同一 stage 的现场。
            try:
                if (not self.root.exists() and destination.resolve() == destination
                        and read_json(destination / "stage.intent.json")["stage_id"] == self.id):
                    self.root = destination
                    details.update(archive_state="archived", retained_path=str(destination))
            except (KeyError, OSError, ValueError) as check_error:
                errors.append(f"verify archive location: {check_error}")
        try:
            # 指针提交可能早于进度事件。直接验证现场，不能根据事件轮次猜测恢复点。
            if (self.root / HEAD).exists() or (self.root / "latest.pt").exists():
                contract = read_json(self.root / "stage.intent.json")["training"]["contract"]
                if not hasattr(exc, "checkpoint_details"):
                    exc.checkpoint_details = verified_recovery_point(self.root, contract, details["archive_state"])
                    if details["archive_state"] == "archived" and exc.checkpoint_details["checkpoint_selection"] == "current":
                        try:
                            snapshot_latest(self.root)
                        except Exception as error:
                            errors.append(f"write compatibility snapshot: {error}")
        except Exception as error:
            errors.append(f"verify retained checkpoint: {error}")

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
        from .checkpoints import HEAD, snapshot_latest
        # 旧读取方使用 latest.pt。发布前固化一次兼容快照，其哈希也进入原有 manifest。
        if (self.root / HEAD).is_file():
            snapshot_latest(self.root)
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
