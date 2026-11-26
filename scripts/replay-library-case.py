"""Replay a bundled control example in a fresh directory, with no historical outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


LIBRARY = Path(__file__).resolve().parents[1] / "harness-plugin" / "control-skill-library"
PYTHON_CASES = {
    "pinn": ("thermal",),
    "mpc": ("prediction", "spring"),
    "pgac": ("spring",),
}


def checked_bytes(relative, files):
    parts = PurePosixPath(relative).parts
    if not parts or relative != "/".join(parts) or ".." in parts or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid library path: {relative}")
    path = (LIBRARY / relative).resolve(strict=True)
    if not path.is_relative_to(LIBRARY):
        raise ValueError(f"Asset outside library: {relative}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != files[relative]["sha256"]:
        raise ValueError(f"Library fingerprint mismatch: {relative}")
    return data


def replay(args):
    manifest = (LIBRARY / "snapshot.json").read_bytes()
    files = json.loads(manifest)["files"]
    workspace = args.workspace.resolve()
    if workspace.is_relative_to(LIBRARY):
        raise ValueError("Choose a workspace outside the installed asset library.")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0")
    env.pop("PYTHONPATH", None)
    calibration_hashes = {}
    if args.task == "matlab":
        scenes = "cases/matlab-scenes"
        inputs = json.loads(checked_bytes(f"{scenes}/inputs.json", files))
        inventory = json.loads(checked_bytes(f"{scenes}/inventory.json", files))
        rows = [row for row in inventory if row["id"] == args.scene]
        if not rows:
            raise ValueError("Unknown scene. Open cases/matlab-scenes/README.md for scene IDs.")
        required = {f"{scenes}/{args.scene}/{row['file']}" for row in rows}
        missing = sorted(required.difference(files))
        if missing:
            raise ValueError(f"This snapshot provides references only for: {', '.join(missing)}")
        materials = args.materials.resolve() if args.materials else None
        for item in inputs[args.scene.replace("-", "_")]:
            if materials is None or not (materials / item["source"]).is_file():
                raise ValueError(f"Specify --materials with the calibration file {item['source']}")
            digest = hashlib.sha256((materials / item["source"]).read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"Wrong version of calibration file: {item['source']}")
            calibration_hashes[item["source"]] = digest
        executable = shutil.which("matlab")
        if not executable:
            raise ValueError("MATLAB executable was not found on PATH.")
        prefixes = (f"{scenes}/{args.scene}/", f"{scenes}/shared/")
        selected = {f"{scenes}/{name}" for name in ("run_scene.m", "case_sha256.m", "inventory.json", "inputs.json")}
        command = [executable, "-batch", "addpath(fullfile(pwd,'cases','matlab-scenes')); "
                   "run_scene(getenv('CONTROL_LIBRARY_SCENE'),getenv('CONTROL_LIBRARY_MATERIALS'),'fixed')"]
    else:
        if args.scene != "mpc1" or args.materials:
            raise ValueError("--scene and --materials apply to the MATLAB replay.")
        prefixes = ("controlkit/", *(f"cases/{name}/" for name in PYTHON_CASES[args.task]))
        selected = {"replay.py", "index.json", "requirements.txt"}
        command = [sys.executable, "-B", "replay.py", args.task]
    selected.update(path for path in files if path.startswith(prefixes))
    workspace.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"control-{args.task}-", dir=workspace))
    for relative in sorted(selected):
        data = checked_bytes(relative, files)
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if args.task == "matlab":
        env.update(CONTROL_LIBRARY_SCENE=args.scene, CONTROL_LIBRARY_MATERIALS=str(materials or run))
    # Upstream checks create scratch files; keep them with this run's other evidence.
    scratch = run / "runs" / "checks"
    scratch.mkdir(parents=True)
    env.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    print(f"Replay directory: {run}", flush=True)
    receipt = {"task": args.task, "snapshot_sha256": hashlib.sha256(manifest).hexdigest(),
               "command": command, "inputs": {p: files[p]["sha256"] for p in sorted(selected)}}
    if args.task == "matlab":
        receipt.update(scene=args.scene, materials=str(materials) if materials else None, calibrations=calibration_hashes)
    record = run / "replay-receipt.json"
    record.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(command, cwd=run, env=env, check=False)
    receipt["exit_code"] = result.returncode
    record.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[*PYTHON_CASES, "matlab"])
    parser.add_argument("--workspace", type=Path, required=True, help="Parent directory for a new replay")
    parser.add_argument("--scene", default="mpc1", help="MATLAB scene ID (default: mpc1)")
    parser.add_argument("--materials", type=Path, help="Root containing this MATLAB scene's original calibrations")
    try:
        sys.exit(replay(parser.parse_args()))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Replay failed: {error}\n")
