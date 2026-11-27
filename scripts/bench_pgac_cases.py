"""Freeze reproducible physical cases for the online PGAC benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
from scipy.signal import cont2discrete

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples/pgac-online"
PROTOCOL = json.loads((ASSETS / "protocol.json").read_text(encoding="utf-8"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != payload:
            raise ValueError(f"Frozen input changed: {path}")
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def controller_parameters(model, weights, increment_limit):
    a, b, e, c = (np.asarray(model[key]) for key in ("A", "B", "E", "C"))
    nx, horizon = len(a), weights["horizon"]
    free = np.vstack([np.linalg.matrix_power(a, k + 1) for k in range(horizon)])
    forced = np.zeros((nx * horizon, horizon))
    disturbance = np.zeros((nx * horizon, e.shape[1]))
    for k in range(horizon):
        rows = slice(nx * k, nx * (k + 1))
        for j in range(k + 1):
            power = np.linalg.matrix_power(a, k - j)
            forced[rows, j] = (power @ b)[:, 0]
            disturbance[rows] += power @ e
    output = np.kron(np.eye(horizon), c)
    output_control = output @ forced
    diff = np.eye(horizon) - np.eye(horizon, k=-1)
    interface = model["interface"]
    limit = interface["controls"][0]["upper"]
    yspan = interface["outputs"][0]["upper"] - interface["outputs"][0]["lower"]
    q = weights["q"] / yspan**2
    r, s = (weights[key] / (2 * limit)**2 for key in ("r", "s"))
    hessian = q * output_control.T @ output_control + r * np.eye(horizon) + s * diff.T @ diff
    # Both selected physical outputs are explicitly one of the bounded states with identical limits.
    bounds = np.array([[v["lower"], v["upper"]] for v in interface["states"]])
    return dict(horizon=horizon, limit=limit, rate=increment_limit, tracking=q, increment=s,
                state_free=free.tolist(), disturbance_free=disturbance.tolist(), outputs=output.tolist(),
                output_control=output_control.tolist(), difference=diff.tolist(),
                hessian=hessian.tolist(), hessian_inverse=np.linalg.inv(hessian).tolist(),
                state_lower=np.tile(bounds[:, 0], horizon).tolist(), state_upper=np.tile(bounds[:, 1], horizon).tolist(),
                constraint_matrix=np.vstack((forced, diff)).tolist())


def cases(case, partition, with_sequences=True):
    """Use distinct RNG streams, including seeded initial conditions and observation noise."""
    seed = PROTOCOL["partitions"][partition]
    rng = np.random.default_rng(seed + (100 if case == "motor" else 0))
    result = []
    motor = case == "motor"
    for family in PROTOCOL["families"]:
        for repeat in range(PROTOCOL["paired_noise_seeds_per_family"]):
            sign = -1 if family == "negative" else 1
            if family in ("midcourse", "cross_zero"):
                sign = [-1, 1][repeat]
            level = sign * rng.uniform(.21, .38) * (38 if motor else 1)
            travel = rng.uniform(.05, .22)
            x0 = [rng.uniform(-.5, .5), sign * travel * 30] if motor else [sign * travel, rng.uniform(-.22, .22)]
            if family == "midcourse":
                x0 = [sign * 2.0, -sign * rng.uniform(4, 7)] if motor else [-sign * travel, sign * rng.uniform(.53, .80)]
            initial_u = sign * rng.uniform(.03, .3) * (8 if motor else 1)
            d0 = rng.uniform(-.45, .45) * (.1 if motor else .2)
            steps = PROTOCOL["evaluation_steps"]
            row = dict(name=f"{partition}-{family}-{repeat}", initial_state=x0,
                       initial_previous_action=[initial_u], steps=steps, reference=[level], disturbance=[d0])
            if with_sequences:
                ref = np.full((steps, 1), level)
                dist = np.full((steps, 1), d0)
                if family == "cross_zero":
                    ref[steps // 2:] *= -.85
                dist[2 * steps // 3:] *= -.7
                row.update(family=family, reference_sequence=ref.tolist(), disturbance_sequence=dist.tolist(),
                           observation_noise_std=([.01, .025] if motor else [.001, .004]) if repeat else [0., 0.],
                           noise_seed=seed + 1000 * PROTOCOL["families"].index(family) + repeat)
            if partition != "student_train":
                row["split"] = "test" if partition == "blind_test" else "validation"
            result.append(row)
    return result


def prepare_materials(directory, case):
    materials = directory / "materials"
    materials.mkdir(parents=True, exist_ok=True)
    source = ROOT / "examples" / PROTOCOL["case_sources"][case]
    spec = read(source / "reference.json")
    interface = dict(spec["interface"], time_semantics="discrete_state_transition")
    b = np.hstack((spec["B"], spec["E"]))
    a, be, _, _, _ = cont2discrete((np.asarray(spec["A"]), b, np.asarray(spec["C"]), np.zeros((1, 2))), interface["sample_time"])
    parameters = dict(A=a.tolist(), B=be[:, :1].tolist(), E=be[:, 1:].tolist())
    model = dict(representation="discrete_polynomial", interface=interface, C=spec["C"], **parameters)
    weights = read(source / "controller.json")["parameters"]
    limit = interface["controls"][0]["upper"]
    rate = limit * .45
    adapter = dict(schema_version=1, backend="python_callable", source_file="source/plant.py",
                   entrypoint="step", source_semantics="transition", interface=interface, C=spec["C"], parameters=parameters)
    ctl = interface["controls"][0]
    observations = [dict(name=v["name"], source="state", index=k, plant_name=v["name"], unit=v["unit"],
                         availability="measured", delay_steps=0) for k, v in enumerate(interface["states"])]
    expert = dict(schema_version=1, backend="python_callable", source_file="source/mpc.py", entrypoint="control",
                  interface=interface, parameters=controller_parameters(model, weights, rate), time_unit="s",
                  control_period=interface["sample_time"], observations=observations, reference_available=True,
                  disturbance_available=True, previous_action_available=True, initial_memory={}, failure_policy="stop",
                  input_rate_bounds=[[-rate, rate]], action_mapping=[dict(action_index=0, plant_control_index=0,
                                                                        plant_control_name=ctl["name"], unit=ctl["unit"])])
    # Keep all published expert training trajectories and use new, separate IL validation/test trajectories.
    collection = read(source / "expert.json")
    for row in collection["scenarios"]:
        row.update(initial_previous_action=[0.], split="train")
    for k, row in enumerate(cases(case, "student_train", with_sequences=False)):
        row.update(name=f"il-holdout-{k}", split="validation" if k % 2 == 0 else "test")
        row["initial_state"] = [x * .71 for x in row["initial_state"]]
        collection["scenarios"].append(row)
    collection["scaler"] = "standard"
    sampling = dict(mode="expert", seed=2141, steps=50, trajectory_count=5, d0=[0.], d_profile=None,
                    control_rate_limits=None, disturbance_rate_limits=None, coupled_constraints=[], trajectories=[])
    for k in range(5):
        u = .01 * limit * (k - 2)
        sampling["trajectories"].append(dict(x0=[.015 * k, -.02 * k], u0=[u], d0=[0.],
                                             u_profile=dict(kind="step", times=[0.0], values=[[u]]), d_profile=None))
    evaluation = dict(condition_source="fresh_holdout", scenarios=cases(case, "validation") + cases(case, "blind_test"),
                      evaluation={**{k: weights[k] for k in ("q", "r", "s")}, "comparison_atol": 1e-12,
                                  **{k: PROTOCOL[k] for k in ("min_relative_improvement", "max_worst_case_relative_regression")}},
                      benchmark=dict(warmup=4, repeats=24, threads=1, batch_size=1))
    train = [r for r in collection["scenarios"] if r["split"] == "train"]
    families = {str(k): "positive" if r["reference"][0] >= 0 else "negative" for k, r in enumerate(train)}
    common = dict(PROTOCOL["optimizer"], domain={k: [[v["lower"], v["upper"]] for v in interface[key]]
                  for k, key in (("x", "states"), ("u", "controls"), ("d", "disturbances"))},
                  original_evaluation=evaluation, **{k: weights[k] for k in ("q", "r", "s")})
    replay = dict(capacity_steps=2048, refresh_every=4, updates_per_epoch=len(train), expert_families=families)
    for mode, config in (("existing", common),
                         ("balanced", dict(common, replay=dict(replay, mode="balanced_starts", student_fraction=0.0))),
                         ("online", dict(common, replay=dict(replay, mode="online", student_fraction=0.5,
                                                            scenarios=cases(case, "student_train"))))):
        freeze(materials / f"{mode}.json", config)
    for key, value in (("plant-adapter", adapter), ("expert-adapter", expert), ("model", model),
                       ("collection", collection), ("sampling", sampling), ("imitation", dict(network=PROTOCOL["network"])),
                       ("evaluation", evaluation), ("protocol", PROTOCOL)):
        freeze(materials / f"{key}.json", value)
    for name in ("plant.py", "mpc.py"):
        shutil.copyfile(ASSETS / name, materials / name)
    return materials
