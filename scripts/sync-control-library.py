"""Import a local control-skill-library working tree, or audit the installed snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import re
import stat
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from control_harness.locking import kernel_mutex

LIBRARY = ROOT / "harness-plugin/control-skill-library"
EVALUATION = ROOT / "tests/fixtures/control-library-evaluation"
MANIFEST = "snapshot.json"
TOP_FILES = {"README.md", "index.json", "sources.json", "replay.py", "requirements.txt"}
ASSET_DIRS = (".agents/skills", "adapters", "cases", "controlkit", "evaluation", "knowledge", "runs")
EXTENSIONS = set(".md .py .m .mjs .json .slx .csv .npy .npz .pt .ts .txt .xlsx .mat .png .zip".split())
OFFLINE = ("evaluation/", "runs/paired/", "runs/valve-evaluation/")
# The source inventory flags these original MATLAB programs and model for local study.
LOCAL_ONLY = {"cases/matlab-scenes/thermal/" + filename for filename in
              ("Identification_ThermalModel.m", "estimate_function.m", "thermalModel_test.slx")}
CACHES = {"__pycache__", ".git", ".pytest_cache", "slprj", "node_modules"}


def record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def is_link(path):
    attrs = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def member(root, name):
    """Only resolve relative snapshot members; reject directory links as well as file links."""
    parts = PurePosixPath(name).parts
    if (not parts or name != "/".join(parts) or PureWindowsPath(name).drive
            or any(part in (".", "..", "/") or "\\" in part or ":" in part for part in parts)):
        raise ValueError(f"Invalid asset path: {name}")
    path = root
    for part in parts:
        path = path / part
        if os.path.lexists(path) and is_link(path):
            raise ValueError(f"Linked asset path: {path}")
    return path


def source_paths(source):
    for name in sorted(TOP_FILES):
        yield name, member(source, name)
    for name in ASSET_DIRS:
        directory = member(source, name)
        if not directory.is_dir():
            raise ValueError(f"Missing source directory: {directory}")
        for root, dirs, files in os.walk(directory):
            parent = Path(root)
            dirs[:] = sorted(d for d in dirs if d not in CACHES and not d.startswith(".")
                             and (parent / d).relative_to(source).as_posix() != "runs/checks")
            for child in dirs:
                if is_link(parent / child):
                    raise ValueError(f"Linked source directory: {parent / child}")
            for filename in sorted(files):
                path = parent / filename
                if not filename.startswith(".") and path.suffix.lower() in EXTENSIONS:
                    key = path.relative_to(source).as_posix()
                    yield key, member(source, key)


def portable_document(key, data):
    """Keep upstream prose, replacing links to materials outside the installed package."""
    doc = data.decode("utf-8")
    parent = posixpath.dirname(key) or "."
    provenance = posixpath.relpath("sources.json", parent)

    def reference(match):
        title, target = match.groups()
        dest = posixpath.normpath(posixpath.join(parent, target))
        if dest == "evaluation/README.md":
            return f"{title}（维护端文档 `docs/control-library-evaluation.md`）"
        if dest == "../素材" or dest.startswith("../素材/"):
            return f"{title}（外部 `{dest[3:]}`，[来源清单]({provenance})）"
        if dest in LOCAL_ONLY:
            return f"{title}（本地研究原件，[程序清单](../inventory.json)）"
        return match.group(0)

    doc = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", reference, doc)
    doc = doc.replace(r"Set-Location D:\projects\control-skill-library", "# 先切换到已复制的案例工作目录")
    doc = doc.replace(r"$py = 'D:\projects\control-agent-harness\.venv\Scripts\python.exe'",
                      "$py = 'python'  # 使用已安装 requirements.txt 依赖的环境")
    if key == "README.md":
        split = doc.index("\n")
        doc = (doc[:split] + "\n\n> Harness 内置资料。下文的运行指标来自源库的历史记录。"
               "复跑前复制到独立工作区，命令会重写 runs/。基础 Python 案例使用 requirements.txt。"
               "完整工程训练及 Simulink 场景所需原始材料见 sources.json 与场景 inputs.json。"
               "历史盲测仅在维护端按 docs/control-library-evaluation.md 复评分。\n" + doc[split:])
    if key in {"cases/matlab-scenes/README.md", "cases/matlab-scenes/thermal/README.md"}:
        split = doc.index("\n")
        doc = (doc[:split] + "\n\n> 此处列出源库历史执行成绩。热参数辨识案例的三个本地研究文件"
               "列在快照清单的 local_study_only 中。复跑该场景或完整 verify_scenes 时，"
               "从获准使用的源库补齐工作副本中的同名文件，并按清单核对 SHA-256。\n" + doc[split:])
    return doc.encode("utf-8")


def build(source):
    source = source.resolve(strict=True)
    if source in (LIBRARY, EVALUATION):
        raise ValueError("Use the original control-skill-library source directory.")
    assets, grader, omitted, adapted = {}, {}, {}, {}
    for key, path in source_paths(source):
        data = path.read_bytes()
        if key in LOCAL_ONLY:
            omitted[key] = record(data)
        elif key.startswith(OFFLINE):
            grader[key] = data
        else:
            original = data
            if key.endswith(".md"):
                data = portable_document(key, data)
            elif key.startswith("runs/matlab-scenes/") and key.endswith(".json") and b"Users" in data:
                # Match JSON-escaped paths, preserving the missing filename and diagnostic.
                data = re.sub(rb"[A-Za-z]:(?:\\\\[^\\\"\r\n]+)*?\\\\Users\\\\[^\\\"\r\n]+"
                              rb"(?:\\\\[^\\\"\r\n]+)*?\\\\(?=[^\\\"\r\n]+\.(?:m|mat|xlsx)\b)",
                              b"<historical-directory>/", data, flags=re.IGNORECASE)
                json.loads(data)
            if data != original:
                adapted[key] = record(original)["sha256"]
            assets[key] = data
    index = json.loads(assets["index.json"])
    for route in index:
        for field in ("instructions", "knowledge", "case", "code", "evidence"):
            if route[field] not in assets:
                raise ValueError(f"Missing {field} for {route['skill']}: {route[field]}")
    if {r["instructions"] for r in index} != {k for k in assets if k.endswith("/SKILL.md")}:
        raise ValueError("Index must cover every skill.")
    # Graders and participant submissions are installed outside the plugin and its read tools.
    # Their snapshot has a frozen, published hash. Keep even its text files byte-exact.
    if "runs/paired/library-snapshot.zip" in grader:
        frozen = json.loads(grader["runs/paired/results.json"])["snapshot_sha256"]
        if record(grader["runs/paired/library-snapshot.zip"])["sha256"] != frozen:
            raise ValueError("Historical evaluation snapshot checksum mismatch.")
    revision = None
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    for files, scope in ((assets, "engineering"), (grader, "offline-evaluation")):
        entries = {key: record(data) for key, data in files.items()}
        manifest = dict(schema_version=1, source="control-skill-library", scope=scope,
                        git_base=revision, content="working tree including uncommitted files", files=entries)
        if scope == "engineering":
            for key, original in adapted.items():
                entries[key]["source_sha256"] = original
            manifest.update(local_study_only=omitted, evidence="runs/ contains historical source-library results",
                            prerequisite_materials="sources.json and cases/matlab-scenes/inputs.json")
        files[MANIFEST] = json_bytes(manifest)
    return assets, grader


def load_manifest(root):
    path = member(root, MANIFEST)
    if not path.exists():
        return None
    manifest = json.loads(path.read_bytes())
    if manifest["schema_version"] != 1 or manifest["source"] != "control-skill-library":
        raise ValueError(f"Unexpected snapshot manifest: {path}")
    for key in manifest["files"]:
        member(root, key)
    return manifest


def write_atomic(path, data, scratch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(dir=scratch, prefix=".control-library-", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(data)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


@dataclass
class ChangeSet:
    root: Path
    files: dict | None
    count: int
    writes: list
    stale: list
    manifest_changed: bool


def plan(root, files):
    if root.exists() and is_link(root):
        raise ValueError(f"Linked snapshot directory: {root}")
    previous = load_manifest(root)
    if files is None and previous is None:
        raise ValueError(f"No snapshot in {root}")
    previous_entries = previous["files"] if previous else {}
    entries = json.loads(files[MANIFEST])["files"] if files is not None else previous_entries
    known = entries.keys() | previous_entries.keys() | {MANIFEST}
    if root.is_dir():
        foreign = [p.relative_to(root).as_posix() for p in root.rglob("*")
                   if (p.is_file() or is_link(p)) and p.relative_to(root).as_posix() not in known]
        if foreign:
            raise ValueError(f"Unlisted files in {root}: {', '.join(foreign[:8])}")
    writes, stale, conflicts = [], [], []
    for key in sorted(entries.keys() | previous_entries.keys()):
        path = member(root, key)
        actual = record(path.read_bytes())["sha256"] if path.is_file() else None
        wanted = entries.get(key, {}).get("sha256")
        if actual == wanted:
            if actual is not None and path.stat().st_size != entries[key]["bytes"]:
                raise ValueError(f"Incorrect manifest byte count: {key}")
            continue
        if actual is not None and actual != previous_entries.get(key, {}).get("sha256"):
            conflicts.append(key)
        elif wanted is None:
            stale.append(key)
        else:
            writes.append(key)
    if conflicts:
        raise ValueError(f"{root}: edited or conflicting file(s): {', '.join(conflicts[:8])}")
    manifest_changed = files is not None and (not previous or (root/MANIFEST).read_bytes() != files[MANIFEST])
    return ChangeSet(root, files, len(entries), writes, stale, manifest_changed)


def reconcile(change, check):
    root = change.root
    if check:
        if change.writes or change.stale or change.manifest_changed:
            raise ValueError(f"Snapshot differs: {root} ({len(change.writes)} changed/missing, "
                             f"{len(change.stale)} stale, manifest changed={change.manifest_changed})")
    else:
        for key in change.writes:
            write_atomic(member(root, key), change.files[key], root.parent)
        for key in change.stale:
            member(root, key).unlink()
        if change.manifest_changed:
            write_atomic(member(root, MANIFEST), change.files[MANIFEST], root.parent)
    print(f"{change.count} files {'verified' if check else 'synchronized'}: {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Source checkout (uses its current working files).")
    parser.add_argument("--check", action="store_true", help="Verify hashes offline; with --source also compare upstream.")
    args = parser.parse_args()
    if not args.check and args.source is None:
        parser.error("Import requires --source PATH. Use --check to audit the bundled files.")
    snapshots = build(args.source) if args.source else (None, None)
    lock = ROOT / "scripts/.control-library.mutex"
    with nullcontext() if args.check else kernel_mutex(lock):
        changes = [plan(root, files) for root, files in zip((LIBRARY, EVALUATION), snapshots)]
        for change in changes:
            reconcile(change, check=args.check)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
