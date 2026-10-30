"""Explicitly sourced state-transition data with trajectory-level partitioning."""
from __future__ import annotations

import numpy as np

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, ModelInterface
from .modeling import load_model
from .runtime import OperationContext


def validate_transitions(interface: ModelInterface, arrays: dict) -> None:
    n = len(arrays["x"])
    if n == 0:
        raise ContractError("Dataset is empty")
    for key, width in (("x", interface.nx), ("u", interface.nu), ("d", interface.nd), ("next_x", interface.nx)):
        value = np.asarray(arrays[key])
        if value.shape != (n, width) or not np.isfinite(value).all():
            raise ContractError(f"Dataset {key} must be finite with shape {(n, width)}")
    trajectory = np.asarray(arrays["trajectory_id"])
    if trajectory.shape != (n,) or not np.issubdtype(trajectory.dtype, np.integer):
        raise ContractError("trajectory_id must be an integer for every row")
    if len(np.unique(trajectory)) < 5:
        raise ContractError("At least five independent trajectories are required for train/validation/test splits")


def trajectory_split(trajectory: np.ndarray, seed: int) -> dict:
    groups = np.unique(trajectory)
    groups = np.random.default_rng(seed).permutation(groups)
    nt = max(1, int(len(groups) * .2))
    nv = max(1, int(len(groups) * .2))
    return {"train": groups[nv + nt:].tolist(), "validation": groups[nt:nv + nt].tolist(),
            "test": groups[:nt].tolist()}


def register_transitions(workspace: Workspace, nominal_id: str, arrays: dict, source: dict, seed: int = 42,
                         reference_id: str | None = None) -> str:
    model = load_model(workspace, nominal_id)
    validate_transitions(model.interface, arrays)
    if source.get("kind") not in {"measured", "reference_simulator"} or not source.get("description"):
        raise ContractError("Data must identify measured or reference_simulator with a source description")
    parents = [nominal_id]
    if reference_id:
        reference = load_model(workspace, reference_id)
        model.interface.compatible(reference.interface)
        parents.append(reference_id)
    if source["kind"] == "reference_simulator" and not reference_id:
        raise ContractError("Simulated transitions require the exact reference model artifact")
    with workspace.stage("transition_dataset", model.interface, parents, {"seed": seed, "source": source}) as stage:
        np.savez_compressed(stage.root / "transitions.npz", **arrays)
        atomic_json(stage.root / "split.json", trajectory_split(arrays["trajectory_id"], seed))
        return stage.publish({"status": "verified", "checks": ["shape", "finite", "trajectory_split"],
                              "samples": len(arrays["x"])}, source=source)


def simulate_transitions(workspace: Workspace, nominal_id: str, reference_id: str, config: dict,
                         context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    nominal, reference = load_model(workspace, nominal_id), load_model(workspace, reference_id)
    nominal.interface.compatible(reference.interface)
    i = nominal.interface
    rng = np.random.default_rng(int(config.get("seed", 42)))
    count, steps = int(config.get("trajectories", 18)), int(config.get("steps", 35))
    lows, highs = np.array([v.lower for v in i.states]), np.array([v.upper for v in i.states])
    rows = {key: [] for key in ("x", "u", "d", "next_x", "trajectory_id")}
    for index in range(count):
        x = rng.uniform(lows * .5 + highs * .5 - (highs - lows) * .2,
                        lows * .5 + highs * .5 + (highs - lows) * .2)
        for _ in range(steps):
            context.check()
            u = rng.uniform([v.lower for v in i.controls], [v.upper for v in i.controls])
            d = rng.uniform([v.lower for v in i.disturbances], [v.upper for v in i.disturbances])
            xn = reference.step(x, u, d)
            for key, value in (("x", x.copy()), ("u", u), ("d", d), ("next_x", xn), ("trajectory_id", index)):
                rows[key].append(value)
            x = xn
            if np.any(x < lows) or np.any(x > highs):
                break
    arrays = {k: np.asarray(v, dtype=int if k == "trajectory_id" else float) for k, v in rows.items()}
    return register_transitions(workspace, nominal_id, arrays,
        {"kind": "reference_simulator", "description": "Explicit separate simulation model; not experimental plant measurements",
         "reference_artifact": reference_id}, int(config.get("seed", 42)), reference_id)


def read_transitions(workspace: Workspace, artifact_id: str):
    root, meta = workspace.resolve(artifact_id, kind="transition_dataset")
    with np.load(root / "transitions.npz", allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    validate_transitions(ModelInterface.parse(meta["interface"]), arrays)
    split = read_json(root / "split.json")
    masks = {key: np.isin(arrays["trajectory_id"], groups) for key, groups in split.items()}
    return arrays, masks, meta


def training_coverage(interface: ModelInterface, arrays: dict, train_mask: np.ndarray) -> dict:
    """Observed marginal ranges, fitted only on training rows; not a joint-domain guarantee."""
    rows = []
    for key, group, variables in (("x", "states", interface.states), ("u", "controls", interface.controls),
                                  ("d", "disturbances", interface.disturbances)):
        values = np.asarray(arrays[key])[train_mask]
        if len(values) == 0:
            raise ContractError("Training coverage requires nonempty training rows")
        for index, variable in enumerate(variables):
            rows.append({"group": group, "name": variable.name, "unit": variable.unit,
                         "observed_min": float(values[:, index].min()), "observed_max": float(values[:, index].max()),
                         "declared_lower": variable.lower, "declared_upper": variable.upper})
    return {"source": "training_partition_only", "samples": int(np.count_nonzero(train_mask)),
            "sample_time": interface.sample_time, "marginal_ranges": rows,
            "joint_coverage_proved": False, "extrapolation_validated": False,
            "scope": "Observed ranges describe training exposure, not a guarantee across declared operating bounds."}
