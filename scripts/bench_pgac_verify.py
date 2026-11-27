"""Run under python -I, load only the delivery package and original plant source."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    sys.dont_write_bytecode = True  # A deployment check must leave the sealed input bundle untouched.
    artifact, materials, chosen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    torch.set_num_threads(1)
    bundle = artifact / "bundle"
    policy = load_module(bundle / "policy_runtime.py", "shipped").load_external_policy(bundle)
    plant = load_module(materials / "plant.py", "original")
    config = json.loads((materials / "evaluation.json").read_text(encoding="utf-8"))
    physical = json.loads((materials / "plant-adapter.json").read_text(encoding="utf-8"))
    physical = physical.get("adapter", physical)
    bounds = physical["interface"]
    yrange, urange = [np.asarray([v["upper"] - v["lower"] for v in bounds[k]]) for k in ("outputs", "controls")]
    c = np.asarray(physical["C"])
    scores = []
    maximum = 0.
    with np.load(artifact / f"test-{chosen}-rollout.npz") as raw:
        for idx, condition in enumerate(config["scenarios"]):
            if condition["split"] != "test":
                continue
            r_seq = condition["reference_sequence"]
            d_seq = condition["disturbance_sequence"]
            noise = np.random.default_rng(condition["noise_seed"]).normal(size=(condition["steps"], len(condition["observation_noise_std"])))
            noise *= condition["observation_noise_std"]
            x = np.asarray(condition["initial_state"])
            previous = np.asarray(condition["initial_previous_action"])
            costs = []
            mask = raw["trajectory_id"] == idx
            if mask.sum() != condition["steps"]:
                raise ValueError("Frozen test rows missing")
            for k, (r, d) in enumerate(zip(r_seq, d_seq, strict=True)):
                features = np.r_[x + noise[k], r, d, previous]
                u = policy.predict(features, previous_action=previous)
                x = plant.step(x, u, d, bounds["sample_time"], physical["parameters"])
                objective = config["evaluation"]
                costs.append(objective["q"] * np.mean(((c @ x - r) / yrange)**2)
                             + objective["r"] * np.mean((u / urange)**2) + objective["s"] * np.mean(((u - previous) / urange)**2))
                maximum = max(maximum, float(np.max(np.abs(x - raw["next_x"][mask][k]))),
                              float(np.max(np.abs(u - raw["u"][mask][k]))))
                previous = u
            scores.append(float(np.mean(costs)))
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    expected = report["closed_loop"]["test"][chosen]["quadratic"]["score"]
    actual = np.mean(scores)
    if maximum > 1e-10 or abs(actual - expected) > 1e-12 or any(m.startswith("control_harness") for m in sys.modules):
        raise ValueError(f"Deployment verification failed: max={maximum}, J={actual}, expected={expected}")
    print(json.dumps(dict(python_isolated=bool(sys.flags.isolated), product_package_imported=False,
                          selected_version=chosen, n_cases=len(scores), max_action_or_state_delta=maximum, J=actual)))


if __name__ == "__main__":
    main()
