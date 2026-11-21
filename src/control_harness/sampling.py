"""Deterministic, explicitly bounded sampling plans and single-Plant collection.

Profiles are expanded before any adapter call. Rates mean absolute command
increments divided by dt, including the first command versus u0/d0. The caller
owns user confirmation, artifact publication and train-only data partitioning.
One expanded batch is limited to 5,000,000 logical scalar values across its
commands, endpoint times and transition arrays. Larger campaigns use batches;
this is a worker allocation guard, not a total experiment or goal limit.
"""
from __future__ import annotations

import copy
import json
import math
import numpy as np

from .contracts import ContractError, ModelInterface, canonical_hash
from .runtime import Cancelled, OperationContext

MAX_BATCH_SCALAR_VALUES = 5_000_000


class SamplingStopped(RuntimeError):
    """Runtime sampling stopped; only preceding valid transitions are retained."""
    def __init__(self, message, arrays, report):
        super().__init__(message)
        self.arrays, self.report = arrays, report


class SamplingCancelled(SamplingStopped, Cancelled):
    """Cancellation preserves partial arrays while retaining native semantics."""


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ContractError(f"{name} must be an explicit finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ContractError(f"{name} must be a finite number") from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise ContractError(f"{name} must be finite" + (" and positive" if positive else ""))
    return 0.0 if value == 0 else value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
        if (raw.dtype.kind not in "iuf" or raw.shape != shape
                or any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)):
            raise ValueError()
        result = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError(f"{name} must be a numeric array of shape {shape}; no broadcasting") from error
    if not np.isfinite(result).all():
        raise ContractError(f"{name} must be finite")
    result[result == 0] = 0.0
    return result


def _interface(value):
    raw = value.to_dict() if isinstance(value, ModelInterface) else copy.deepcopy(value)
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int:
        raise ContractError("Sampling requires an explicit discrete ModelInterface")
    for group in ("states", "controls", "outputs", "disturbances"):
        for variable in raw.get(group, []):
            for key in ("lower", "upper"):
                variable[key] = _number(variable[key], f"{group}.{key}")
    raw["sample_time"] = _number(raw.get("sample_time"), "sample_time", positive=True)
    return ModelInterface.parse(raw)


def _bounds(variables):
    return np.array([v.lower for v in variables]), np.array([v.upper for v in variables])


def _within(value, variables, name):
    lower, upper = _bounds(variables)
    if np.any(value < lower - 1e-12) or np.any(value > upper + 1e-12):
        raise ContractError(f"{name} is outside declared interface bounds")


def _rates(value, width, name):
    if value is None:
        return None
    result = _array(value, (width,), name)
    if np.any(result < 0):
        raise ContractError(f"{name} must be nonnegative absolute rates per second")
    return result


def _budget(interface, count, steps):
    # Input profile arrays and endpoint times coexist with collected transitions.
    # Count before building trajectory dictionaries or allocating NumPy arrays.
    scalars = count * (steps * (2 * interface.nx + 2 * interface.nu + 2 * interface.nd
                               + 2 * interface.ny + 3)
                       + interface.nx + interface.nu + interface.nd + 1)
    if scalars > MAX_BATCH_SCALAR_VALUES:
        raise ContractError(f"Sampling batch allocation budget exceeded: {scalars} logical scalar values "
                            f"> {MAX_BATCH_SCALAR_VALUES}; reduce steps/trajectories or collect in batches")
    return scalars


def _constraints(interface, items):
    if not isinstance(items, list):
        raise ContractError("coupled_constraints must be a list")
    result, names = [], set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) - {"name", "x", "u", "d", "lower", "upper"}:
            raise ContractError("Linear coupled constraints accept name/x/u/d/lower/upper only")
        name = item.get("name", f"linear-{index}")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ContractError("Linear coupled constraint names must be unique and nonempty")
        row = {"name": name}
        for key, width in (("x", interface.nx), ("u", interface.nu), ("d", interface.nd)):
            row[key] = _array(item.get(key, [0.0] * width), (width,), f"{name}.{key}").tolist()
        for key in ("lower", "upper"):
            row[key] = None if item.get(key) is None else _number(item[key], f"{name}.{key}")
        if (row["lower"] is None and row["upper"] is None) or (row["lower"] is not None and row["upper"] is not None and row["lower"] > row["upper"]):
            raise ContractError("Linear coupled constraint needs ordered finite endpoint(s)")
        if not any(value for key in ("x", "u", "d") for value in row[key]):
            raise ContractError("Linear coupled constraint coefficients cannot all be zero")
        names.add(name)
        result.append(row)
    return result


def _check_coupled(constraints, x, u, d, *, inputs_only=False):
    for item in constraints:
        if inputs_only and any(item["x"]):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(np.dot(item["x"], x) + np.dot(item["u"], u) + np.dot(item["d"], d))
        if not math.isfinite(value):
            raise ContractError(f"Linear coupled constraint {item['name']} evaluation is non-finite")
        if ((item["lower"] is not None and value < item["lower"] - 1e-12)
                or (item["upper"] is not None and value > item["upper"] + 1e-12)):
            raise ContractError(f"Linear coupled constraint {item['name']} is violated")


def _profile(spec, variables, initial, rate, dt, duration, steps, rng, name):
    width = len(variables)
    if width == 0:
        if spec is not None:
            raise ContractError(f"{name} must be absent when its declared width is zero")
        return np.empty((steps, 0)), None
    if not isinstance(spec, dict):
        raise ContractError(f"{name} requires an explicit profile")
    kind, time = spec.get("kind"), np.arange(steps, dtype=float) * dt
    lower, upper = _bounds(variables)
    if kind in {"step", "timeseries"}:
        time_key = "times" if kind == "step" else "time"
        allowed = {"kind", time_key, "values"} | ({"interpolation"} if kind == "timeseries" else set())
        if set(spec) - allowed:
            raise ContractError(f"Unknown {name} {kind} profile field")
        knots = np.asarray(spec.get(time_key, []))
        if knots.ndim != 1:
            raise ContractError(f"{name}.{time_key} must be a one-dimensional time array")
        knots = _array(knots, (len(knots),), f"{name}.{time_key}")
        values = _array(spec.get("values"), (len(knots), width), f"{name}.values")
        if not len(knots) or knots[0] != 0 or np.any(np.diff(knots) <= 0) or knots[-1] < time[-1] - 1e-12 and kind == "timeseries":
            raise ContractError(f"{name} requires strictly increasing time from zero covering sampled instants")
        _within(values, variables, name)
        if kind == "step":
            if np.any(knots > duration + 1e-12) or not np.allclose(knots / dt, np.rint(knots / dt), atol=1e-9, rtol=0):
                raise ContractError("Step switches must lie on the declared dt grid within duration")
            interpolation = "zoh"
        else:
            interpolation = spec.get("interpolation", "zoh")
            if interpolation not in {"zoh", "linear"}:
                raise ContractError("timeseries interpolation must be zoh or linear")
        if interpolation == "zoh":
            signal = values[np.maximum(0, np.searchsorted(knots, time + 1e-12, side="right") - 1)]
        else:
            signal = np.column_stack([np.interp(time, knots, values[:, k]) for k in range(width)])
        normalized = {"kind": kind, time_key: knots.tolist(), "values": values.tolist()}
        if kind == "timeseries":
            normalized["interpolation"] = interpolation
    elif kind == "chirp":
        if set(spec) - {"kind", "offset", "amplitude", "f0_hz", "f1_hz", "phase"}:
            raise ContractError(f"Unknown {name} chirp profile field")
        offset = _array(spec.get("offset"), (width,), f"{name}.offset")
        amplitude = _array(spec.get("amplitude"), (width,), f"{name}.amplitude")
        phase = _array(spec.get("phase", [0.0] * width), (width,), f"{name}.phase")
        f0, f1 = (_number(spec.get(key), f"{name}.{key}") for key in ("f0_hz", "f1_hz"))
        if np.any(amplitude < 0) or min(f0, f1) < 0 or max(f0, f1) <= 0 or max(f0, f1) > .5 / dt:
            raise ContractError("Chirp needs nonnegative amplitude/frequencies and a positive frequency at/below Nyquist")
        _within(offset - amplitude, variables, name)
        _within(offset + amplitude, variables, name)
        angle = 2 * math.pi * (f0 * time + .5 * (f1 - f0) / duration * time**2)
        signal = offset + amplitude * np.sin(angle[:, None] + phase)
        normalized = {"kind": kind, "offset": offset.tolist(), "amplitude": amplitude.tolist(),
                      "f0_hz": f0, "f1_hz": f1, "phase": phase.tolist()}
    elif kind == "bounded_random":
        if set(spec) - {"kind", "low", "high", "hold_steps"}:
            raise ContractError(f"Unknown {name} bounded_random profile field")
        low = _array(spec.get("low"), (width,), f"{name}.low")
        high = _array(spec.get("high"), (width,), f"{name}.high")
        hold = _integer(spec.get("hold_steps", 1), f"{name}.hold_steps")
        if np.any(low > high) or np.any(initial < low) or np.any(initial > high):
            raise ContractError("bounded_random needs ordered bounds containing the initial command")
        _within(low, variables, name)
        _within(high, variables, name)
        signal, previous = np.empty((steps, width)), initial.copy()
        for index in range(steps):
            if index % hold == 0:
                lo = low if rate is None else np.maximum(low, previous - rate * dt)
                hi = high if rate is None else np.minimum(high, previous + rate * dt)
                previous = rng.uniform(lo, hi)
            signal[index] = previous
        normalized = {"kind": kind, "low": low.tolist(), "high": high.tolist(), "hold_steps": hold}
    else:
        raise ContractError(f"Unsupported {name} profile; choose step/chirp/bounded_random/timeseries")
    signal = _array(signal, (steps, width), name)
    _within(signal, variables, name)
    if rate is not None and np.any(np.abs(np.diff(np.vstack([initial, signal]), axis=0)) > rate * dt + 1e-12):
        raise ContractError(f"{name} exceeds its per-second command rate limit (including initial command)")
    return signal, normalized


def _hash_plan(plan):
    return canonical_hash({key: value for key, value in plan.items() if key not in {"plan_hash", "config"}})


def build_sampling_plan(interface, config) -> dict:
    """Expand the same complete schema for recommended and expert conditions."""
    interface = _interface(interface)
    if not isinstance(config, dict) or set(config) - {"schema_version", "mode", "seed", "dt", "duration", "steps", "trajectory_count", "x0", "u0", "d0", "u_profile", "d_profile", "trajectories", "control_rate_limits", "disturbance_rate_limits", "coupled_constraints", "recommendation"}:
        raise ContractError("Unknown sampling configuration field")
    mode = config.get("mode", "expert")
    if mode not in {"recommended", "expert"} or type(config.get("schema_version", 1)) is not int or config.get("schema_version", 1) != 1:
        raise ContractError("Sampling schema_version must be 1 and mode recommended or expert")
    seed = _integer(config.get("seed", 0), "seed", 0)
    if seed > 2**53 - 1:
        raise ContractError("seed must remain exactly representable in JSON/JavaScript")
    dt = _number(config.get("dt", interface.sample_time), "dt", positive=True)
    if not math.isclose(dt, interface.sample_time, rel_tol=0, abs_tol=1e-12):
        raise ContractError("Sampling dt differs from the adapter's declared discrete sample_time")
    conditions = config.get("trajectories")
    if conditions is not None and (not isinstance(conditions, list) or not conditions):
        raise ContractError("trajectories must be a nonempty list of explicit conditions")
    count = _integer(config.get("trajectory_count", len(conditions) if conditions is not None else 1), "trajectory_count")
    if conditions is not None and count != len(conditions):
        raise ContractError("trajectory_count does not match its condition list")
    if "duration" not in config and "steps" not in config:
        raise ContractError("Sampling requires explicit duration or steps")
    if "steps" in config:
        _integer(config["steps"], "steps")
        _budget(interface, count, config["steps"])
    duration = _number(config.get("duration", config.get("steps", 0) * dt), "duration", positive=True)
    steps = _integer(config.get("steps", round(duration / dt)), "steps")
    if not math.isclose(duration, steps * dt, rel_tol=0, abs_tol=1e-9 * max(1.0, duration)):
        raise ContractError("duration must equal an integer number of dt intervals")
    _budget(interface, count, steps)
    duration = steps * dt
    ru = _rates(config.get("control_rate_limits"), interface.nu, "control_rate_limits")
    rd = _rates(config.get("disturbance_rate_limits"), interface.nd, "disturbance_rate_limits")
    coupled = _constraints(interface, config.get("coupled_constraints", []))
    if conditions is None:
        conditions = [{} for _ in range(count)]
    trajectories, normalized, ids = [], [], set()
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict) or set(condition) - {"trajectory_id", "x0", "u0", "d0", "u_profile", "d_profile"}:
            raise ContractError("Unknown trajectory condition field")
        identifier = _integer(condition.get("trajectory_id", index), "trajectory_id", 0)
        if identifier > 2**53 - 1:
            raise ContractError("trajectory_id must remain exactly representable in JSON/JavaScript")
        if identifier in ids:
            raise ContractError("trajectory_id must be unique")
        ids.add(identifier)
        x0 = _array(condition.get("x0", config.get("x0")), (interface.nx,), "x0")
        u0 = _array(condition.get("u0", config.get("u0")), (interface.nu,), "u0")
        d0 = _array(condition.get("d0", config.get("d0", [])), (interface.nd,), "d0")
        for value, variables, name in [(x0, interface.states, "x0"), (u0, interface.controls, "u0"), (d0, interface.disturbances, "d0")]:
            _within(value, variables, name)
        _check_coupled(coupled, x0, u0, d0)
        dp = condition.get("d_profile", config.get("d_profile"))
        if dp is None and interface.nd:
            dp = {"kind": "step", "times": [0.0], "values": [d0.tolist()]}
        u, up = _profile(condition.get("u_profile", config.get("u_profile")), interface.controls, u0, ru, dt, duration, steps,
                         np.random.default_rng(np.random.SeedSequence([seed, identifier, 0])), "u_profile")
        d, dp = _profile(dp, interface.disturbances, d0, rd, dt, duration, steps,
                         np.random.default_rng(np.random.SeedSequence([seed, identifier, 1])), "d_profile")
        for k in range(steps):
            _check_coupled(coupled, x0, u[k], d[k], inputs_only=k != 0)
        base = {"trajectory_id": identifier, "x0": x0.tolist(), "u0": u0.tolist(), "d0": d0.tolist()}
        normalized.append({**base, "u_profile": up, "d_profile": dp})
        trajectories.append({**base, "t": (np.arange(steps + 1) * dt).tolist(), "u": u.tolist(), "d": d.tolist()})
    settings = {"schema_version": 1, "mode": mode, "seed": seed, "dt": dt, "duration": duration, "steps": steps,
        "trajectory_count": len(trajectories), "control_rate_limits": None if ru is None else ru.tolist(),
        "disturbance_rate_limits": None if rd is None else rd.tolist(), "coupled_constraints": coupled, "trajectories": normalized}
    if "recommendation" in config:
        try:
            settings["recommendation"] = json.loads(json.dumps(config["recommendation"], allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ContractError("recommendation metadata must be finite JSON") from error
    plan = {key: copy.deepcopy(value) for key, value in settings.items() if key not in {"trajectories", "recommendation", "trajectory_count"}}
    plan.update(interface=json.loads(json.dumps(interface.to_dict(), allow_nan=False)), interface_hash=interface.fingerprint,
                trajectories=trajectories, config=settings)
    plan["plan_hash"] = _hash_plan(plan)
    return plan


def recommend_sampling(interface, overrides=None) -> dict:
    """Propose bounded candidates using declared limits, without model inference."""
    interface = _interface(interface)
    if overrides is not None and not isinstance(overrides, dict):
        raise ContractError("recommendation overrides must be an object")
    override = copy.deepcopy(overrides or {})
    count = override.get("trajectory_count", len(override["trajectories"]) if isinstance(override.get("trajectories"), list) else 5)
    count = _integer(count, "trajectory_count")
    dt = _number(override.get("dt", interface.sample_time), "dt", positive=True)
    lower, upper = _bounds(interface.states)
    ul, uh = _bounds(interface.controls)
    dl, dh = _bounds(interface.disturbances)
    u0 = _array(override.get("u0", ((ul + uh) / 2).tolist()), (interface.nu,), "u0")
    d0 = _array(override.get("d0", ((dl + dh) / 2).tolist()), (interface.nd,), "d0")
    settings = {"mode": "recommended", "seed": 42, "dt": dt, "steps": 50, "trajectory_count": count,
        "x0": ((lower + upper) / 2).tolist(), "u0": u0.tolist(), "d0": d0.tolist(),
        "control_rate_limits": ((uh - ul) / (10 * dt)).tolist(), "disturbance_rate_limits": [0.0] * interface.nd,
        "u_profile": {"kind": "bounded_random", "low": np.maximum(ul, u0 - .1 * (uh - ul)).tolist(),
                      "high": np.minimum(uh, u0 + .1 * (uh - ul)).tolist(), "hold_steps": 5},
        "d_profile": {"kind": "step", "times": [0.0], "values": [d0.tolist()]} if interface.nd else None,
        "coupled_constraints": [], "recommendation": {"basis": "Declared ordered interface and finite bounds only",
            "requires_user_review": True, "physical_model_inferred": False, "sufficient_excitation_proved": False,
            "rate_source": "Proposed sampling command limit; verify against actual actuator limits"}}
    if "duration" in override and "steps" not in override:
        settings.pop("steps")
    settings.update(override)
    budget_steps = (_integer(settings["steps"], "steps") if "steps" in settings else
                    round(_number(settings["duration"], "duration", positive=True) / dt))
    _budget(interface, count, _integer(budget_steps, "steps"))
    if "trajectories" not in override:
        settings["trajectories"] = [{} if "x0" in override else {"x0": ((lower + upper) / 2 + (upper - lower) * .05 * (0 if count == 1 else 2 * k / (count - 1) - 1)).tolist()} for k in range(count)]
    return build_sampling_plan(interface, settings)["config"]


def _prefix(plan, trial):
    if not isinstance(trial, dict) or set(trial) - {"trajectories", "steps"}:
        raise ContractError("trial accepts trajectories and steps only")
    count = _integer(trial.get("trajectories", len(plan["trajectories"])), "trial.trajectories")
    steps = _integer(trial.get("steps", plan["steps"]), "trial.steps")
    if count > len(plan["trajectories"]) or steps > plan["steps"]:
        raise ContractError("A trial must be a deterministic prefix of the full plan")
    result = copy.deepcopy(plan)
    result.update(steps=steps, duration=steps * plan["dt"], parent_plan_hash=plan["plan_hash"], trial={"trajectories": count, "steps": steps})
    result["trajectories"] = result["trajectories"][:count]
    for item in result["trajectories"]:
        item["t"], item["u"], item["d"] = item["t"][:steps + 1], item["u"][:steps], item["d"][:steps]
    result["plan_hash"] = _hash_plan(result)
    return result


def _validated_plan(plan):
    if not isinstance(plan, dict) or "interface" not in plan or "config" not in plan:
        raise ContractError("A complete expanded sampling plan is required")
    expected = build_sampling_plan(plan["interface"], plan["config"])
    _integer(plan.get("schema_version"), "plan.schema_version")
    _integer(plan.get("steps"), "plan.steps")
    _integer(plan.get("seed"), "plan.seed", 0)
    _number(plan.get("dt"), "plan.dt", positive=True)
    _number(plan.get("duration"), "plan.duration", positive=True)
    interface = _interface(plan["interface"])
    _rates(plan.get("control_rate_limits"), interface.nu, "plan.control_rate_limits")
    _rates(plan.get("disturbance_rate_limits"), interface.nd, "plan.disturbance_rate_limits")
    _constraints(interface, plan.get("coupled_constraints"))
    if "trial" in plan:
        if plan.get("parent_plan_hash") != expected["plan_hash"]:
            raise ContractError("Trial parent differs from its full sampling plan")
        expected = _prefix(expected, plan["trial"])
    if set(plan) != set(expected) or plan.get("plan_hash") != expected["plan_hash"]:
        raise ContractError("Sampling plan hash/content differs from its resolved configuration")
    for key in expected:
        if key != "config" and plan[key] != expected[key]:
            raise ContractError(f"Expanded sampling plan {key} was changed")
    for item in plan["trajectories"]:
        _integer(item.get("trajectory_id"), "plan.trajectory_id", 0)
        for key, shape in [("x0", (len(item["x0"]),)), ("u0", (len(item["u0"]),)), ("d0", (len(item["d0"]),)),
                           ("t", (plan["steps"] + 1,)), ("u", (plan["steps"], len(item["u0"]))), ("d", (plan["steps"], len(item["d0"])) )]:
            _array(item[key], shape, f"plan.{key}")
    return expected


def prefix_plan(plan, trial) -> dict:
    plan = _validated_plan(plan)
    if "trial" in plan:
        raise ContractError("Create a trial from the complete plan, not another trial")
    return _prefix(plan, trial)


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value if value is None or isinstance(value, str) else {"unsupported_type": type(value).__name__}


def collect(adapter, plan, context=None, journal=None):
    """Reset once per trajectory, stop at first invalid transition, retain prefix."""
    plan = _validated_plan(plan)
    interface = _interface(adapter.interface)
    interface.compatible(_interface(plan["interface"]))
    context = context or OperationContext()
    widths = {"x": interface.nx, "u": interface.nu, "d": interface.nd, "next_x": interface.nx, "y": interface.ny, "next_y": interface.ny}
    rows = {key: [] for key in [*widths, "trajectory_id", "time"]}
    report = {"schema_version": 1, "status": "running", "plan_hash": plan["plan_hash"],
        "parent_plan_hash": plan.get("parent_plan_hash"), "dt": plan["dt"], "requested_trajectories": len(plan["trajectories"]),
        "requested_steps_per_trajectory": plan["steps"], "completed_trajectories": 0, "valid_rows": 0,
        "rate_semantics": "abs(command[k]-previous_command)/dt including the first command",
        "data_semantics": "single-Plant simulated transitions; measurement attribution is caller-owned",
        "hidden_memory_used_as_state": False, "sufficient_excitation_proved": False, "stability_proved": False}
    attempt = {"trajectory_id": None, "step_index": None, "time": None}
    def arrays():
        return {key: np.asarray(values, dtype=np.float64).reshape((-1, width)) if width else np.empty((len(values), 0))
                for key, width in widths.items() for values in [rows[key]]} | {
                    "trajectory_id": np.asarray(rows["trajectory_id"], dtype=np.int64), "time": np.asarray(rows["time"], dtype=np.float64)}
    try:
        context.check()
        report["adapter"] = json.loads(json.dumps(adapter.describe(), allow_nan=False)) if callable(getattr(adapter, "describe", None)) else {"description_available": False}
        for index, trajectory in enumerate(plan["trajectories"]):
            attempt = {"trajectory_id": trajectory["trajectory_id"], "step_index": 0, "time": 0.0, "phase": "reset"}
            context.check()
            restored = journal.load(index, trajectory) if journal else None
            if restored is not None:
                committed, _ = restored
                for key in rows:
                    rows[key].extend(committed[key])
                journal.commit(index, trajectory, committed)
                report["completed_trajectories"] += 1
                continue
            x0 = _array(trajectory["x0"], (interface.nx,), "x0")
            returned = adapter.reset(x0.copy())
            state = _array(adapter.state, (interface.nx,), "adapter.state")
            if not np.array_equal(_array(returned, (interface.nx,), "adapter.reset"), x0) or not np.array_equal(state, x0):
                raise ContractError("Adapter reset did not restore exactly the declared x0")
            context.progress("sampling", trajectory_id=trajectory["trajectory_id"], phase="started", valid_rows=len(rows["x"]))
            for k in range(plan["steps"]):
                attempt = {"trajectory_id": trajectory["trajectory_id"], "step_index": k, "time": trajectory["t"][k], "phase": "before_step"}
                context.check()
                x = _array(adapter.state, (interface.nx,), "x")
                u, d = np.array(trajectory["u"][k]), np.array(trajectory["d"][k])
                attempt.update(x=x.copy(), u=u.copy(), d=d.copy())
                _within(x, interface.states, "x")
                _check_coupled(plan["coupled_constraints"], x, u, d)
                attempt["phase"] = "output"
                y = np.asarray(adapter.output())
                attempt["y"] = y.copy()
                y = _array(y, (interface.ny,), "y")
                _within(y, interface.outputs, "y")
                context.check()
                attempt["phase"] = "step"
                next_x = np.asarray(adapter.step(u.copy(), d.copy()))
                attempt["next_x"] = next_x.copy()
                context.check()
                next_x = _array(next_x, (interface.nx,), "next_x")
                _within(next_x, interface.states, "next_x")
                if not np.array_equal(_array(adapter.state, (interface.nx,), "adapter.state"), next_x):
                    raise ContractError("Adapter state disagrees with returned next_x")
                _check_coupled(plan["coupled_constraints"], next_x, u, d)
                attempt["phase"] = "next_output"
                next_y = np.asarray(adapter.output())
                attempt["next_y"] = next_y.copy()
                next_y = _array(next_y, (interface.ny,), "next_y")
                _within(next_y, interface.outputs, "next_y")
                for key, value in (("x", x), ("u", u), ("d", d), ("next_x", next_x), ("y", y), ("next_y", next_y)):
                    rows[key].append(value.copy())
                rows["trajectory_id"].append(trajectory["trajectory_id"])
                rows["time"].append(trajectory["t"][k])
            report["completed_trajectories"] += 1
            if journal:
                length = plan["steps"]
                journal.commit(index, trajectory, {key: np.asarray(values[-length:]) for key, values in rows.items()})
            context.progress("sampling", trajectory_id=trajectory["trajectory_id"], phase="completed", valid_rows=len(rows["x"]))
        context.check()
    except Exception as error:
        attempted = {key: value for key, value in attempt.items() if key in {"x", "u", "d", "next_x", "y", "next_y"}}
        nonfinite = {key: np.argwhere(~np.isfinite(value)).tolist() for key, value in attempted.items()
                     if isinstance(value, np.ndarray) and value.dtype.kind in "iuf" and not np.isfinite(value).all()}
        report.update(status="cancelled" if isinstance(error, Cancelled) else "failed", valid_rows=len(rows["x"]),
            failure={**{key: value for key, value in attempt.items() if key not in attempted},
                     "attempted": _safe(attempted), "nonfinite_indices": nonfinite,
                     "reason": str(error), "error_type": type(error).__name__},
            invalid_transition_in_arrays=False)
        failure = SamplingCancelled if isinstance(error, Cancelled) else SamplingStopped
        raise failure(str(error), arrays(), report) from error
    report.update(status="completed", valid_rows=len(rows["x"]), invalid_transition_in_arrays=False)
    return arrays(), report
