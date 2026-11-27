"""Post-blind engineering stress check of the two already released policies (no fitting)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from bench_pgac_verify import load_module


# Coordinates are physical values. Each condition has a sign-reversed counterpart.
# High previous action is an initial actuator condition, not an executed command.
CONDITIONS = {
    "spring": [
        ("position_recovery", [.94, -.14], .36, 0.),
        ("velocity_braking", [-.05, 1.395], -.25, -2.60),
        ("actuator_slew", [.10, -.10], -.35, 3.80),
    ],
    "motor": [
        ("speed_recovery", [0., 37.6], 26., 0.),
        ("current_recovery", [11.16, 0.], -8., -4.0),
        ("actuator_slew", [.25, 4.], -14., 22.80),
    ],
}
STEPS = 24


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition(name, sign, initial, target, initial_action, policy, plant, adapter, slew):
    interface = adapter["interface"]
    states = interface["states"]
    state_limit = np.array([v["upper"] for v in states])
    limit = interface["controls"][0]["upper"]
    for v in states + interface["outputs"] + interface["controls"]:
        if v["lower"] != -v["upper"]:
            raise ValueError("The sign-paired stress design requires symmetric physical limits")
    output_limit = np.asarray([v["upper"] for v in interface["outputs"]])
    x = sign * np.asarray(initial)
    r = np.array([sign * target])
    d = np.zeros(len(interface["disturbances"]))
    up = np.array([sign * initial_action])
    initial_ratio = np.abs(x) / state_limit
    states_after, actions, increments, violations = [], [], [], []
    for k in range(STEPS):
        u = policy.predict(np.r_[x, r, d, up], previous_action=up)
        du = u - up
        x = plant.step(x, u, d, interface["sample_time"], adapter["parameters"])
        y = np.asarray(adapter["C"]) @ x
        if not np.isfinite(np.r_[x, u, y, du]).all():
            raise ValueError("Nonfinite stress-check output")
        states_after.append(x.copy())
        actions.append(float(u[0]))
        increments.append(float(du[0]))
        for field, value, bound in (("state", x, state_limit), ("output", y, output_limit),
                                    ("action", u, limit), ("increment", du, slew)):
            if np.any(np.abs(value) > bound + 1e-6):
                violations.append({"decision": k + 1, "field": field,
                                   "absolute_excess": float(np.max(np.abs(value) - bound))})
        if violations:
            break
        up = u
    x_peak = np.max(np.abs(states_after), axis=0)
    u = np.abs(actions)
    du = np.abs(increments)
    return dict(name=name, sign=sign, steps=len(u), initial_state=(sign*np.asarray(initial)).tolist(),
                reference=r.tolist(), initial_previous_action=sign*initial_action,
                initial_state_fraction_of_limit=initial_ratio.tolist(),
                post_step_peak_abs_state=x_peak.tolist(), post_step_peak_state_fraction_of_limit=(x_peak/state_limit).tolist(),
                post_step_state_steps_above_90pct=(np.abs(states_after) >= .9*state_limit).sum(axis=0).tolist(),
                max_applied_action=float(u.max()), applied_action_steps_above_90pct=int(np.sum(u >= .9*limit)),
                max_applied_increment=float(du.max()), applied_increment_steps_at_limit=int(np.sum(du >= slew-1e-6)),
                violations=violations)


def check_case(run, case):
    root = run / case
    materials = root / "materials"
    ops = read(root / "operations.json")
    metadata = read(run / "frozen-inputs.json")[case]
    for name in ("plant.py", "plant-adapter.json", "expert-adapter.json"):
        if sha(materials / name) != metadata[name]:
            raise ValueError("Original material differs from the frozen benchmark")
    artifacts = root / "workspace/artifacts"
    final = artifacts / ops["evaluate_online"]["artifact_id"]
    prior = read(final / "policy.json")
    source = artifacts / (prior["candidate"] if prior["selected_version"] == "candidate" else prior["imitation_policy"])
    release = final / "bundle"
    expected = torch.load(source / "bundle/model.pt", weights_only=True)
    delivered = torch.load(release / "model.pt", weights_only=True)
    if expected.keys() != delivered.keys() or not all(torch.equal(delivered[k], v) for k,v in expected.items()):
        raise ValueError("Deployed weights differ from the frozen accepted version")
    policy = load_module(release / "policy_runtime.py", "released").load_external_policy(release)
    plant = load_module(materials / "plant.py", "original")
    adapter = read(materials / "plant-adapter.json")
    expert = read(materials / "expert-adapter.json")
    slew = expert["input_rate_bounds"][0][1]
    rows = [condition(name, sign, x, r, u, policy, plant, adapter, slew)
            for name, x, r, u in CONDITIONS[case] for sign in (1, -1)]
    return dict(selected=prior["selected_version"], release_artifact=final.name, release_weights_sha256=sha(release / "model.pt"),
                state_limits=[v["upper"] for v in adapter["interface"]["states"]],
                action_limit=adapter["interface"]["controls"][0]["upper"], slew_per_decision=slew, results=rows)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    outcome = {"scope": "Post-blind stress characterization, excluded from all training and release decisions",
               "requested_steps_per_condition": STEPS, "seed": None, "noise": "none", "disturbance": "zero",
               "cases": {case: check_case(args.benchmark_root, case) for case in CONDITIONS}}
    args.output.write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)
