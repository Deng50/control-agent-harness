"""Read-only material inspection and bounded byte snapshots for both entry paths.

No uploaded source is imported, evaluated or invoked. The caller owns artifact
publication and authorization of the source and destination paths. A supplied
dependency root confines all explicitly selected dependencies; directories are
not followed through symlinks or Windows reparse points.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import shutil
import uuid
from contextlib import contextmanager

from .contracts import ContractError, canonical_hash
from .runtime import OperationContext


FORMATS = {".csv", ".xlsx", ".py", ".m", ".slx", ".mdl", ".json"}
MAX_FILES = 2048
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_DEPTH = 32
_CHUNK = 1024 * 1024


@contextmanager
def _snapshot_directory(target: Path):
    # mkdtemp's mode0700 replaces the inherited Windows restricted-token ACEs.
    # This directory is inside an already authorized immutable stage; inherit
    # its normal permissions, while keeping an unpredictable exclusive name.
    pending = target / f".material-snapshot-{uuid.uuid4().hex}"
    pending.mkdir()
    try:
        yield pending
    finally:
        if pending.exists() or pending.is_symlink():
            _checked(pending, target)
            if pending.resolve().parent != target.resolve():
                raise ContractError("Snapshot cleanup target is not inside its own stage")
            shutil.rmtree(pending)


def _basename(value: str) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or any(char in value for char in '/\\:<>"|?*')
            or any(ord(char) < 32 for char in value) or value[-1] in {" ", "."}
            or PureWindowsPath(value).is_reserved()):
        raise ContractError("original_name must be one portable basename, without traversal or alternate streams")
    return value


def _absolute(value: str | Path, label: str) -> Path:
    try:
        path = Path(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must be an absolute filesystem path") from error
    if not path.is_absolute():
        raise ContractError(f"{label} must be an absolute path")
    return Path(os.path.abspath(path))


def _signature(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns, getattr(info, "st_file_attributes", 0)]


def _checked(path: Path, root: Path | None = None) -> os.stat_result:
    """Inspect ancestors before resolving; resolving first would hide a link."""
    if root is not None and not path.is_relative_to(root):
        raise ContractError(f"Material dependency escapes its declared root: {path}")
    for part in [*reversed(path.parents), path]:
        try:
            info = part.lstat()
        except OSError as error:
            raise ContractError(f"Material path is missing or unreadable: {part}") from error
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ContractError(f"Material symlink or Windows reparse point is not accepted: {part}")
    if root is not None and not path.resolve().is_relative_to(root.resolve()):
        raise ContractError(f"Material dependency resolves outside its declared root: {path}")
    return info


def _regular(path: Path, root: Path) -> os.stat_result:
    info = _checked(path, root)
    if not stat.S_ISREG(info.st_mode):
        raise ContractError(f"Material files must be regular files: {path}")
    if info.st_size > MAX_FILE_BYTES:
        raise ContractError("Material file exceeds the 64 MiB per-file snapshot limit")
    return info


def _read_file(path: Path, root: Path, context: OperationContext, target: Path | None = None) -> dict:
    context.check()
    before = _regular(path, root)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError(f"Cannot open material without following links: {path}") from error
    digest, size = hashlib.sha256(), 0
    with os.fdopen(descriptor, "rb") as source:
        if _signature(os.fstat(source.fileno())) != _signature(before):
            raise ContractError(f"Material changed before it could be read: {path}")
        output = target.open("xb") if target is not None else None
        try:
            while True:
                context.check()
                block = source.read(_CHUNK)
                if not block:
                    break
                size += len(block)
                if size > MAX_FILE_BYTES:
                    raise ContractError("Material file grew beyond the snapshot size limit")
                digest.update(block)
                if output is not None:
                    output.write(block)
            if output is not None:
                output.flush()
                os.fsync(output.fileno())
            if _signature(os.fstat(source.fileno())) != _signature(before):
                raise ContractError(f"Material changed while it was being read: {path}")
        finally:
            if output is not None:
                output.close()
    after = _regular(path, root)
    if _signature(after) != _signature(before) or size != before.st_size:
        raise ContractError(f"Material changed while it was being inspected/copied: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": size, "source_signature": _signature(after)}


def inspect_material(local_path: str | Path, *, original_name: str | None = None,
                     dependency_root: str | Path | None = None,
                     dependencies: list[str | Path] | tuple = (),
                     context: OperationContext | None = None) -> dict:
    """Inventory explicitly selected bytes, without mutation or dimension guesses."""
    context = context or OperationContext()
    context.check()
    primary = _absolute(local_path, "local_path")
    root = _absolute(dependency_root, "dependency_root") if dependency_root is not None else primary.parent
    if not stat.S_ISDIR(_checked(root).st_mode):
        raise ContractError("dependency_root must be a directory")
    _regular(primary, root)
    name = _basename(original_name if original_name is not None else primary.name)
    suffix = Path(name).suffix.lower()
    if suffix not in FORMATS:
        raise ContractError("Material format must be CSV/XLSX/Python/MATLAB/SLX/MDL/JSON")
    if not isinstance(dependencies, (list, tuple)):
        raise ContractError("dependencies must be a list of explicit files or directories")
    logical_primary = primary.relative_to(root).with_name(name).as_posix()
    entries, directories, selected, targets = {}, {}, [], {}

    def add(path: Path, *, primary_file: bool = False, depth: int = 0):
        context.check()
        if depth > MAX_DEPTH:
            raise ContractError("Material dependency directory exceeds the depth limit")
        info = _checked(path, root)
        relative = path.relative_to(root).as_posix()
        if len(path.relative_to(root).parts) > MAX_DEPTH:
            raise ContractError("Material dependency path exceeds the depth limit")
        for component in path.relative_to(root).parts:
            _basename(component)
        if stat.S_ISDIR(info.st_mode):
            if relative.casefold() in targets:
                raise ContractError("Material logical filename collides with a dependency directory")
            directories[relative] = _signature(info)
            if len(directories) > MAX_FILES:
                raise ContractError("Material dependency exceeds the directory-count limit")
            try:
                children = sorted(path.iterdir(), key=lambda child: child.name)
            except OSError as error:
                raise ContractError(f"Cannot enumerate material dependency directory: {path}") from error
            for child in children:
                add(child, depth=depth + 1)
            if _signature(_checked(path, root)) != _signature(info):
                raise ContractError(f"Material directory changed during inspection: {path}")
        elif stat.S_ISREG(info.st_mode):
            if str(path) in entries:
                return
            logical = logical_primary if primary_file or path == primary else relative
            if logical.casefold() in {value.casefold() for value in directories}:
                raise ContractError("Material logical filename collides with a dependency directory")
            collision = targets.get(logical.casefold())
            if collision is not None and collision != str(path):
                raise ContractError("Material filenames collide after logical-name/case normalization")
            targets[logical.casefold()] = str(path)
            observed = _read_file(path, root, context)
            entries[str(path)] = {"path": "source/" + logical, "relative_source": relative,
                                  "role": "primary" if path == primary else "dependency", **observed}
            if len(entries) > MAX_FILES or sum(item["size_bytes"] for item in entries.values()) > MAX_TOTAL_BYTES:
                raise ContractError("Material snapshot exceeds the file-count or total-byte limit")
        else:
            raise ContractError(f"Special device/socket/FIFO material is not accepted: {path}")

    add(primary, primary_file=True)
    for dependency in dependencies:
        try:
            path = Path(dependency)
        except (TypeError, ValueError) as error:
            raise ContractError("Each dependency must be an explicit filesystem path") from error
        if not path.is_absolute():
            if ".." in path.parts:
                raise ContractError("Dependency traversal is not accepted")
            path = root / path
        path = _absolute(path, "dependency")
        _checked(path, root)
        selected.append(str(path))
        add(path)
    files = sorted(entries.values(), key=lambda item: item["path"])
    directory_names = sorted(directories)
    semantic = {"format": suffix[1:], "primary": next(item["sha256"] for item in files if item["role"] == "primary"),
                "dependencies": [{key: item[key] for key in ("path", "sha256", "size_bytes")}
                                 for item in files if item["role"] == "dependency"],
                "directories": directory_names}
    version = {"root": str(root), "primary": str(primary), "files": files, "directories": directories}
    return {"schema_version": 1, "format": suffix[1:], "format_detection": "filename_extension_only",
            "primary_file": "source/" + logical_primary, "original_name": name,
            "local_path": str(primary), "dependency_root": str(root), "selected_dependencies": selected,
            "files": files, "directories": directory_names, "file_count": len(files),
            "total_bytes": sum(item["size_bytes"] for item in files),
            "content_hash": canonical_hash(semantic), "version_hash": canonical_hash(version),
            "source_executed": False, "dimensions_inferred": False}


def register_material(local_path: str | Path, destination: str | Path, *, source_kind: str = "path",
                      original_name: str | None = None, source_metadata: dict | None = None,
                      dependency_root: str | Path | None = None,
                      dependencies: list[str | Path] | tuple = (), expected_manifest: dict | None = None,
                      context: OperationContext | None = None) -> dict:
    """Snapshot into a new ``source`` child; the caller publishes the artifact.

    Existing stage configuration may remain in destination. An existing source
    child is never overwritten. Failure removes only this call's temporary tree.
    Origin/transport metadata do not enter the material's content fingerprint.
    """
    context = context or OperationContext()
    if source_kind not in {"path", "upload"}:
        raise ContractError("Material source_kind must be path or upload")
    if source_metadata is not None and not isinstance(source_metadata, dict):
        raise ContractError("Material source_metadata must be a JSON object")
    try:
        source_metadata = json.loads(json.dumps(source_metadata or {}, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ContractError("Material source_metadata must be finite JSON") from error
    observed = inspect_material(local_path, original_name=original_name, dependency_root=dependency_root,
                                dependencies=dependencies, context=context)
    if expected_manifest is not None:
        if (not isinstance(expected_manifest, dict) or expected_manifest.get("version_hash") != observed["version_hash"]
                or expected_manifest.get("content_hash") != observed["content_hash"]):
            raise ContractError("Material changed since its inspected manifest; inspect the current source again")
    target = _absolute(destination, "destination")
    # Validate existing ancestors before mkdir can traverse an unexpected link.
    ancestor = target
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    _checked(ancestor)
    for selected in observed["selected_dependencies"]:
        path = Path(selected)
        if path.is_dir() and target.is_relative_to(path):
            raise ContractError("Snapshot destination must not lie inside a selected dependency directory")
    if target == Path(observed["local_path"]) or Path(observed["local_path"]).is_relative_to(target):
        raise ContractError("Snapshot destination cannot contain the original primary material")
    target.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(_checked(target).st_mode):
        raise ContractError("Snapshot destination must be a directory")
    published = target / "source"
    if published.exists() or published.is_symlink():
        raise ContractError("Snapshot source directory already exists; refusing overwrite")
    root = Path(observed["dependency_root"])
    context.progress("material_snapshot_started", files=observed["file_count"], source_executed=False)
    with _snapshot_directory(target) as pending:
        for relative in observed["directories"]:
            (pending / relative).mkdir(parents=True, exist_ok=True)
        for entry in observed["files"]:
            context.check()
            copied = pending / Path(entry["path"]).relative_to("source")
            copied.parent.mkdir(parents=True, exist_ok=True)
            current = _read_file(root / entry["relative_source"], root, context, target=copied)
            if any(current[key] != entry[key] for key in ("sha256", "size_bytes", "source_signature")):
                raise ContractError("Material changed between inspection and snapshot copy")
        context.progress("material_snapshot_copied", files=observed["file_count"], source_executed=False)
        final = inspect_material(local_path, original_name=original_name, dependency_root=dependency_root,
                                 dependencies=dependencies, context=context)
        if final["version_hash"] != observed["version_hash"]:
            raise ContractError("Material file/directory changed during snapshot; nothing was published")
        context.check()
        _checked(target)
        if published.exists() or published.is_symlink():
            raise ContractError("Snapshot destination appeared during copy; refusing overwrite")
        pending.rename(published)
    public_files = [{key: item[key] for key in ("path", "sha256", "size_bytes", "role")} for item in observed["files"]]
    return {"schema_version": 1, "format": observed["format"], "primary_file": observed["primary_file"],
            "source": {"kind": source_kind, "local_path": observed["local_path"],
                       "original_name": observed["original_name"], "metadata": source_metadata},
            "files": public_files, "file_hashes": {item["path"]: item["sha256"] for item in public_files},
            "directories": ["source/" + value for value in observed["directories"]],
            "file_count": observed["file_count"], "total_bytes": observed["total_bytes"],
            "content_hash": observed["content_hash"], "version_hash": observed["version_hash"],
            "static_only": True, "source_executed": False, "dimensions_inferred": False}
