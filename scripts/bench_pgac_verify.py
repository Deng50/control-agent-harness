"""Run under python -I, load only the delivery package and original plant source."""
import importlib.util
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import zipfile

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The source can be a sealed artifact or an immutable historical reference.
    write_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_setting
    return module


def autonomous_trajectory(policy, plant, physical, condition):
    """The two benchmark sources have full current-state observations and one Plant tick per decision."""
    x = np.asarray(condition["initial_state"])
    previous = np.asarray(condition["initial_previous_action"])
    noise = np.random.default_rng(condition["noise_seed"]).normal(size=(condition["steps"], len(x)))
    noise *= condition["observation_noise_std"]
    rows = {k: [] for k in ("next_x", "next_y", "u", "previous_u", "reference")}
    c = np.asarray(physical["C"])
    for k, (r, d) in enumerate(zip(condition["reference_sequence"], condition["disturbance_sequence"], strict=True)):
        u = policy.predict(np.r_[x + noise[k], r, d, previous], previous_action=previous)
        x = plant.step(x, u, d, physical["interface"]["sample_time"], physical["parameters"])
        for field, value in (("next_x", x), ("next_y", c @ x), ("u", u), ("previous_u", previous), ("reference", r)):
            rows[field].append(np.asarray(value).copy())
        previous = u
    return {k: np.stack(values) for k, values in rows.items()}


def unpack_archive(artifact, archive, destination):
    """Extract only files in the sealed policy bundle; verify all bytes before importing any module."""
    receipt = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if receipt["artifact_id"] != artifact.name or digest != receipt["sha256"]:
        raise ValueError("Archive receipt does not match the selected artifact and ZIP")
    raw_metadata = (artifact / "artifact.json").read_bytes()
    if hashlib.sha256(raw_metadata).hexdigest() != receipt["artifact_metadata_sha256"]:
        raise ValueError("Archive receipt refers to a different artifact manifest")
    files = {name: digest for name, digest in json.loads(raw_metadata)["files"].items() if name.startswith("bundle/")}
    current = {p.relative_to(artifact).as_posix() for p in (artifact / "bundle").rglob("*") if p.is_file()}
    if current != set(files):
        raise ValueError("Current policy directory contains missing or unregistered files")
    expected = set(files) | {"DOWNLOAD_MANIFEST.json"}
    with zipfile.ZipFile(archive) as package:
        if set(package.namelist()) != expected or len(package.infolist()) != len(expected):
            raise ValueError("Archive members differ from the selected bundle")
        if hashlib.sha256(package.read("DOWNLOAD_MANIFEST.json")).hexdigest() != receipt["member_sha256"]["DOWNLOAD_MANIFEST.json"]:
            raise ValueError("ZIP download manifest digest mismatch")
        for name, expected_digest in files.items():
            original = (artifact / name).resolve()
            member = PurePosixPath(name)
            if (".." in member.parts or member.as_posix() != name or "\\" in name or ":" in name
                    or not original.is_relative_to((artifact / "bundle").resolve())):
                raise ValueError("Invalid sealed bundle member path: " + name)
            if package.getinfo(name).file_size != original.stat().st_size:
                raise ValueError("Archive member differs in size: " + name)
            data = package.read(name)
            if hashlib.sha256(data).hexdigest() != expected_digest or data != original.read_bytes():
                raise ValueError("Archive member differs from the selected bundle: " + name)
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
    return digest


def main(bundle=None, archive_digest=None):
    sys.dont_write_bytecode = True  # A deployment check must leave the sealed input bundle untouched.
    artifact, materials, chosen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    torch.set_num_threads(1)
    bundle = artifact / "bundle" if bundle is None else bundle
    policy = load_module(bundle / "policy_runtime.py", "shipped").load_external_policy(bundle)
    plant = load_module(materials / "plant.py", "original")
    config = json.loads((materials / "evaluation.json").read_text(encoding="utf-8"))
    physical = json.loads((materials / "plant-adapter.json").read_text(encoding="utf-8"))
    physical = physical.get("adapter", physical)
    bounds = physical["interface"]
    yrange, urange = [np.asarray([v["upper"] - v["lower"] for v in bounds[k]]) for k in ("outputs", "controls")]
    scores = []
    tracking_rows = []
    objective = config["evaluation"]
    tracking = objective.get("primary_metric") == "tracking_rmse"
    if tracking:
        outputs = [v["name"] for v in bounds["outputs"]]
        target = outputs.index(objective.get("tracking_output", outputs[0]))
    maximum = 0.
    with np.load(artifact / f"test-{chosen}-rollout.npz") as raw:
        for idx, condition in enumerate(config["scenarios"]):
            if condition["split"] != "test":
                continue
            mask = raw["trajectory_id"] == idx
            if mask.sum() != condition["steps"]:
                raise ValueError("Frozen test rows missing")
            trace = autonomous_trajectory(policy, plant, physical, condition)
            errors = trace["next_y"] - trace["reference"]
            action = np.mean((trace["u"] / urange)**2)
            increment = np.mean(((trace["u"] - trace["previous_u"]) / urange)**2)
            scores.append(objective["q"]*np.mean((errors / yrange)**2) + objective["r"]*action + objective["s"]*increment)
            maximum = max(maximum, *(float(np.max(np.abs(trace[k] - raw[k][mask]))) for k in ("next_x", "u")))
            if tracking:
                tracking_rows.append(np.sqrt((np.mean(errors[:, target]**2), action, increment)))
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    expected = report["closed_loop"]["test"][chosen]["quadratic"]["score"]
    actual = np.mean(scores)
    if maximum > 1e-10 or abs(actual - expected) > 1e-12 or any(m.startswith("control_harness") for m in sys.modules):
        raise ValueError(f"Deployment verification failed: max={maximum}, J={actual}, expected={expected}")
    result = dict(python_isolated=bool(sys.flags.isolated), product_package_imported=False,
                  selected_version=chosen, n_cases=len(scores), max_action_or_state_delta=maximum, J=actual,
                  J_delta=float(actual - expected))
    if tracking:
        names = ("tracking_rmse", "action_rms", "increment_rms")
        summary = report["closed_loop"]["test"][chosen]["quadratic"]["tracking"]
        reported = np.array([[r[k] for k in names] for r in summary["per_scenario"]])
        observed = np.asarray(tracking_rows)
        average = observed.mean(axis=0)
        delta = max(np.max(np.abs(observed - reported)), np.max(np.abs(average - [summary[k] for k in names])))
        if delta > 1e-10:
            raise ValueError(f"Independent per-condition physical or action RMS mismatch: {delta}")
        result.update(zip(names, average, strict=True))
        result["max_report_metric_delta"] = float(delta)
    if archive_digest is not None:
        result["loaded_from_verified_zip_sha256"] = archive_digest
    print(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) == 5:
        with tempfile.TemporaryDirectory(prefix="pgac-zip-check-",dir=Path(sys.argv[1]).resolve().parents[1]) as temporary:
            destination = Path(temporary)
            digest = unpack_archive(Path(sys.argv[1]), Path(sys.argv[4]), destination)
            main(destination / "bundle", digest)
    else:
        main()
