"""Create and freeze physically disjoint tracking-primary scenarios for the two reference plants."""
import copy
import shutil

import numpy as np

from bench_pgac_cases import ASSETS, ROOT, freeze, read
from control_harness.pgac_scenarios import ensure_new_physical_conditions


TRACKING_ASSETS = ROOT / "examples/pgac-tracking"
PROTOCOL = read(TRACKING_ASSETS / "protocol.json")


def scenarios(case, partition):
    seed = PROTOCOL["partitions"][partition] + (350 if case == "motor" else 0)
    rng = np.random.default_rng(seed)
    motor = case == "motor"
    n = PROTOCOL["steps"]
    result = []
    for replicate in range(PROTOCOL["replicates"][partition]):
        for index, family in enumerate(PROTOCOL["families"]):
            if family == "negative":
                # The plant, disturbance, MPC cost, state and actuator boxes for both
                # fixtures are symmetric. Mirror the complete preceding positive step.
                row = copy.deepcopy(result[-1])
                for key in ("initial_state", "initial_previous_action", "reference", "disturbance",
                            "reference_sequence", "disturbance_sequence"):
                    row[key] = (-np.asarray(row[key])).tolist()
            else:
                sign = 1 if family == "positive" or replicate % 2 == 0 else -1
                x = [rng.uniform(-.5, .5), rng.uniform(-4, 4)] if motor else list(rng.uniform(-.16, .16, 2))
                r = sign * rng.uniform(.21, .39) * (38 if motor else 1)
                u0 = sign * rng.uniform(.03, .28) * (8 if motor else 1)
                if family == "midcourse":
                    x = [sign * rng.uniform(1.4, 2.8), -sign * rng.uniform(4, 8)] if motor else [
                        -sign * rng.uniform(.13, .25), sign * rng.uniform(.48, .82)]
                if family == "near_state_limit":
                    fraction = rng.uniform(.84, .91)
                    x = [-sign * rng.uniform(.3, 1.3), sign * fraction * 40] if motor else [
                        sign * fraction, -sign * rng.uniform(.20, .33)]
                    r = sign * rng.uniform(.43, .63) * (40 if motor else 1)
                if family == "slew_recovery":
                    u0 = sign * rng.uniform(.90, .96) * (24 if motor else 4)
                    r = -r
                ref = np.full((n, 1), r)
                if family == "cross_zero":
                    ref[rng.integers(n // 3, 2 * n // 3):] *= -rng.uniform(.65, 1.1)
                d = np.full((n, 1), rng.uniform(-.08, .08) * (.5 if motor else 1))
                d[rng.integers(n // 2, 3 * n // 4):] *= -rng.uniform(.4, 1.1)
                row = dict(initial_state=x, initial_previous_action=[u0], reference=ref[0].tolist(),
                           disturbance=d[0].tolist(), reference_sequence=ref.tolist(), disturbance_sequence=d.tolist(),
                           steps=n)
            # Both controllers and the expert receive the identical noise realization;
            # reference signs and the exogenous schedule remain independent of the Actor.
            row.update(name=f"rmse-{partition}-{family}-{replicate}", family=family,
                       noise_seed=seed + 10000 * index + replicate,
                       observation_noise_std=([.012, .029] if motor else [.0014, .0044]) if replicate % 2 else [0., 0.])
            if partition != "student_train":
                row["split"] = "validation" if partition == "validation" else "test"
            result.append(row)
    return result


def _physical(row):
    return dict(row, x0=row["initial_state"], initial_action=row["initial_previous_action"])


def make_materials(directory, case, historical):
    """Only read archived source files when a reference benchmark is supplied."""
    mats = directory / "materials"
    mats.mkdir(parents=True, exist_ok=True)
    source = historical / case / "materials"
    for key in ("plant-adapter", "expert-adapter", "model", "sampling", "collection", "imitation"):
        freeze(mats / (key + ".json"), read(source / (key + ".json")))
    for name in ("plant.py", "mpc.py"):
        target = mats / name
        if target.exists() and target.read_bytes() != (ASSETS / name).read_bytes():
            raise ValueError(f"Changed physical source {target}")
        if not target.exists():
            shutil.copyfile(ASSETS / name, target)
    old = read(source / "online.json")
    previous = [*read(source / "collection.json")["scenarios"], *old["replay"]["scenarios"],
                *old["original_evaluation"]["scenarios"]]
    new = {split: scenarios(case, split) for split in PROTOCOL["partitions"]}
    ensure_new_physical_conditions([_physical(row) for row in previous],
                                   [_physical(row) for split in new.values() for row in split])
    interface = read(mats / "plant-adapter.json")["interface"]
    output = interface["outputs"][0]["name"]
    evaluation = dict(condition_source="fresh_holdout", scenarios=new["validation"] + new["blind_test"],
                      evaluation={**old["original_evaluation"]["evaluation"], **PROTOCOL["acceptance"],
                                  "primary_metric": "tracking_rmse", "tracking_output": output},
                      benchmark=old["original_evaluation"]["benchmark"])
    for name, overrides in PROTOCOL["designs"].items():
        config = dict(old, **PROTOCOL["common_training"])
        config.update(overrides, original_evaluation=evaluation,
                      replay=dict(old["replay"], **PROTOCOL["replay"], scenarios=new["student_train"]))
        freeze(mats / (name + ".json"), config)
    freeze(mats / "evaluation.json", evaluation)
    freeze(mats / "protocol.json", PROTOCOL)
    return mats
