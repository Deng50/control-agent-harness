"""Repeat only the online training operation from the frozen baseline in a fresh workspace."""
import argparse
from pathlib import Path

import torch

from bench_online_pgac import Run
from bench_pgac_cases import read
from control_harness.artifacts import Workspace, atomic_json, file_hash
from control_harness.restricted_pgac import _state_hash


def repeat(source, destination, case):
    if (destination / case).exists():
        raise ValueError(f"Repeat requires a new directory: {destination / case}")
    old = source / case
    original = Workspace(old / "workspace")
    config_path = old / "materials/online.json"
    config_sha = read(source / "frozen-inputs.json")[case]["online.json"]
    if file_hash(config_path) != config_sha:
        raise ValueError("The archived training config differs from the pre-blind freeze")
    operations = read(old / "operations.json")
    request = operations["online"]
    config = read(config_path)
    if config != request["config"] or request["operation"] != "optimize_external_policy":
        raise ValueError("The recorded operation differs from the frozen training request")
    frozen = read(source / "frozen-candidates.json")[case]["online"]
    source_weights = original.resolve(request["artifact_id"])[0] / "bundle/model.pt"
    if file_hash(source_weights) != frozen:
        raise ValueError("The reference candidate differs from the pre-blind freeze")

    run = Run(destination, case)
    for artifact_id in request["inputs"].values():
        upstream = original.resolve(artifact_id)[0]
        run.workspace.import_artifact(upstream, expected_artifact_id=artifact_id)
    run.stage("online", request["operation"], request["inputs"], config)
    trained = run.artifact("online")
    repeated_weights = trained / "bundle/model.pt"
    before, after = [torch.load(p, map_location="cpu", weights_only=True) for p in (source_weights, repeated_weights)]
    tensors_equal = before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    report = read(trained / "report.json")
    output = dict(case=case, online_config_sha256=config_sha, frozen_file_sha256=frozen,
                  repeat_file_sha256=file_hash(repeated_weights), frozen_tensor_sha256=_state_hash(before),
                  repeat_tensor_sha256=_state_hash(after), tensor_count=len(before), all_tensors_equal=tensors_equal,
                  imitation_tensor_sha256=report["initial_weights_sha256"], optimizer_updates=report["optimizer_updates"],
                  student_control_steps=report["replay"]["student_control_steps"])
    atomic_json(run.root / "repeat.json", output)
    if not tensors_equal or output["repeat_file_sha256"] != frozen:
        raise ValueError(f"Repeated candidate differs; see {run.root / 'repeat.json'}")
    print(f"{case}: {frozen} ({len(before)} identical tensors)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original benchmark directory with both pre-blind hash manifests")
    parser.add_argument("destination", type=Path, help="Directory for new per-case workspaces")
    parser.add_argument("--cases", choices=("spring", "motor"), nargs="+", default=("spring", "motor"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name in args.cases:
        repeat(args.source.resolve(), args.destination.resolve(), name)
