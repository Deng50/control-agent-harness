"""Learn a discrete next-state model from one recorded sampled dataset.

This is ordinary supervised NN regression, not a PINN or a nominal-residual
model.  The original physical interface is retained separately from the
derived feature/target interface, whose source is NPZ arrays rather than CSV.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from .artifacts import Workspace, atomic_json, confined, read_json
from .checkpoints import checkpoint_fields, current_checkpoint
from .contracts import ContractError, DataInterface, ModelInterface, canonical_hash
from .runtime import OperationContext
from .supervised import _read_data, _resolved, train_supervised


_SPLITS = ("train", "validation", "test")
_MAX_BYTES = 256 * 1024 * 1024
_MAX_VALUES = 8_000_000


def _has_bool(value):
    return isinstance(value, (bool, np.bool_)) or isinstance(value, (list, tuple)) and any(_has_bool(item) for item in value)


def _C(value, interface):
    try:
        raw = np.asarray(value)
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError("C must be an explicit finite [ny,nx] matrix") from error
    if (_has_bool(value) or raw.dtype.kind not in "fiu" or matrix.shape != (interface.ny, interface.nx) or not np.isfinite(matrix).all()):
        raise ContractError("C must be an explicit finite [ny,nx] matrix; identity is not inferred")
    return matrix


def _limit(value, width, name):
    if value is None: return None
    values = value if isinstance(value, list) else [value] * width
    try: numbers = [float(item) if type(item) in {int, float} else float("nan") for item in values]
    except (OverflowError, ValueError): numbers = [float("nan")]
    if len(numbers) != width or any(not math.isfinite(item) or item < 0 for item in numbers):
        raise ContractError(f"{name} must be a finite nonnegative scalar or list with width {width}")
    return numbers


def _config(config, interface):
    allowed = {"network", "C", "scaler", "plan_refs", "resume_checkpoint", "resume_checkpoint_sha256",
        "max_test_state_nrmse", "max_test_output_nrmse", "max_rollout_state_nrmse", "max_rollout_output_nrmse"}
    if not isinstance(config, dict) or set(config) - allowed: raise ContractError("Explicit predictive network/C/scaler config required; unknown fields are unsupported")
    if not isinstance(config.get("network"), dict): raise ContractError("network must contain the actual S05 supervised hyperparameters")
    if any(key in config["network"] for key in ("plan_refs", "resume_checkpoint", "resume_checkpoint_sha256")):
        raise ContractError("Predictive confirmation/resume references belong at the top level, not network")
    scaler = config.get("scaler", "standard")
    if not isinstance(scaler, str) or scaler not in {"standard", "minmax", "none"}: raise ContractError("scaler must be standard, minmax or none")
    limits = {name: _limit(config.get(name), width, name) for name, width in
        (("max_test_state_nrmse", interface.nx), ("max_test_output_nrmse", interface.ny),
         ("max_rollout_state_nrmse", interface.nx), ("max_rollout_output_nrmse", interface.ny))}
    network = dict(config["network"])
    for key in ("resume_checkpoint", "resume_checkpoint_sha256"):
        if key in config: network[key] = config[key]
    return _C(config.get("C"), interface), scaler, network, limits


def _read_sampled(workspace, artifact_id):
    root, metadata = workspace.resolve(artifact_id, kind="sampled_dataset")
    interface = ModelInterface.parse(metadata["interface"])
    widths = {"x": interface.nx, "u": interface.nu, "d": interface.nd, "next_x": interface.nx,
              "y": interface.ny, "next_y": interface.ny, "trajectory_id": None, "time": None}
    path = root / "transitions.npz"
    try:
        with path.open("rb") as stream: payload = stream.read(_MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES: raise ContractError("Sampled NPZ exceeds the 256 MiB single-job byte budget")
        if hashlib.sha256(payload).hexdigest() != metadata["files"].get("transitions.npz"):
            raise ContractError("Sampled NPZ differs from its bound artifact version")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist(); names = [entry.filename for entry in entries]
            if (len(names) > 32 or len(names) != len(set(names)) or sum(row.file_size for row in entries) > _MAX_BYTES or
                    any("/" in name or "\\" in name or not name.endswith(".npy") for name in names) or
                    not {key + ".npy" for key in widths}.issubset(names)):
                raise ContractError("Sampled NPZ has an unsupported inventory/expanded-byte budget")
            sizes, count = {}, 0
            for key, width in widths.items():
                member = archive.getinfo(key + ".npy")
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else: raise ContractError("Sampled NPZ requires numeric version-1/2 NPY arrays")
                    header = stream.tell()
                if (len(shape) != (1 if width is None else 2) or shape[0] < 1 or
                        width is not None and shape[1] != width or dtype.kind not in ("iu" if key == "trajectory_id" else "fiu")):
                    raise ContractError(f"Sampled {key} shape/type disagrees with the physical interface")
                logical = math.prod(shape); count += logical
                if count > _MAX_VALUES or logical * dtype.itemsize != member.file_size - header:
                    raise ContractError("Sampled NPZ logical shape/member bytes exceed the single-job budget")
                sizes[key] = shape[0]
            if len(set(sizes.values())) != 1: raise ContractError("Sampled arrays have inconsistent row counts")
            if next(iter(sizes.values())) * (2 * (2 * interface.nx + interface.nu + interface.nd) + 1) > _MAX_VALUES:
                raise ContractError("Derived supervised arrays exceed the single-job budget; prepare smaller batches")
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            if loaded["trajectory_id"].dtype.kind == "u" and np.any(loaded["trajectory_id"] > np.iinfo(np.int64).max):
                raise ContractError("Sampled trajectory IDs exceed int64")
            arrays = {key: np.asarray(loaded[key], dtype=np.int64 if key == "trajectory_id" else np.float64) for key in widths}
    except ContractError: raise
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error: raise ContractError(f"Invalid sampled NPZ: {error}") from error
    if any(not np.isfinite(value).all() for value in arrays.values()): raise ContractError("Sampled arrays must be finite")
    ids = set(np.unique(arrays["trajectory_id"]).tolist())
    with (root / "split.json").open("rb") as stream: split_payload = stream.read(16 * 1024 * 1024 + 1)
    if len(split_payload) > 16 * 1024 * 1024 or hashlib.sha256(split_payload).hexdigest() != metadata["files"].get("split.json"):
        raise ContractError("Sampled split metadata exceeds its budget or changed after version verification")
    try: split = json.loads(split_payload.decode("utf-8"))
    except (UnicodeError, ValueError) as error: raise ContractError("Invalid sampled trajectory split JSON") from error
    if not isinstance(split, dict) or set(split) != set(_SPLITS) or len(ids) < 5:
        raise ContractError("Sampled data require the original train/validation/test trajectory split and at least five trajectories")
    observed, masks = set(), {}
    for part in _SPLITS:
        groups = split[part]
        if (not isinstance(groups, list) or not groups or any(type(value) is not int for value in groups) or
                len(set(groups)) != len(groups) or observed.intersection(groups)):
            raise ContractError("Sampled trajectory partitions must be nonempty, disjoint integer-ID lists")
        observed.update(groups); masks[part] = np.isin(arrays["trajectory_id"], groups)
        if not masks[part].any(): raise ContractError("Every sampled split must have actual rows")
    if observed != ids: raise ContractError("Sampled split does not exactly cover the recorded trajectories")
    for key, variables in (("x", interface.states), ("next_x", interface.states), ("u", interface.controls),
                           ("d", interface.disturbances), ("y", interface.outputs), ("next_y", interface.outputs)):
        low, high = np.asarray([v.lower for v in variables]), np.asarray([v.upper for v in variables])
        if np.any(arrays[key] < low - 1e-12) or np.any(arrays[key] > high + 1e-12):
            raise ContractError(f"Recorded sampled {key} is outside its physical interface bounds")
    for group in sorted(ids):
        rows = np.flatnonzero(arrays["trajectory_id"] == group)
        rows = rows[np.argsort(arrays["time"][rows], kind="stable")]
        if len(rows) > 1 and (not np.allclose(np.diff(arrays["time"][rows]), interface.sample_time, rtol=1e-8, atol=1e-12) or
                not np.allclose(arrays["next_x"][rows[:-1]], arrays["x"][rows[1:]], rtol=1e-10, atol=1e-12)):
            raise ContractError("Recorded trajectory time/state continuity conflicts with sample_time")
    return arrays, masks, split, interface, metadata


def _data_interface(physical):
    signals, features, targets, mapping, source_mapping = [], [], [], {}, {}
    for key, group, role in (("x", "states", "state"), ("u", "controls", "control"), ("d", "disturbances", "disturbance")):
        for column, variable in enumerate(getattr(physical, group)):
            name = group + "::" + variable.name
            signals.append({"name": name, "unit": variable.unit, "role": role, "observable": True, "delay_steps": 0,
                "column": {"index": len(signals), "header": f"NPZ {key}[{column}]", "worksheet": None}})
            features.append({"signal": name, "lag": 0, "window": 1})
            source_mapping[name] = {"array": key, "column": column}
            if key == "x":
                targets.append({"signal": name, "offset": 1}); mapping[variable.name] = name
                source_mapping[name]["target_array"] = "next_x"
    return DataInterface.parse({"schema_version": 1, "interface_kind": "tabular_supervised", "task": "dynamic_prediction",
        "name": physical.name + " sampled next-state features", "sample_time": physical.sample_time,
        "signals": signals, "features": features, "targets": targets, "state_mapping": mapping,
        "time_column": {"index": len(signals), "header": "NPZ time", "worksheet": None},
        "trip_column": {"index": len(signals) + 1, "header": "NPZ trajectory_id", "worksheet": None},
        "source_format": "npz_transition_arrays", "column_semantics": "logical_signal_index_not_tabular_source_column",
        "source_mapping": {"file": "transitions.npz", "signals": source_mapping, "time": {"array": "time", "unit": "s"},
                           "trajectory": {"array": "trajectory_id"}, "row_semantics": "zero-based source NPZ row"}})


def _fit(values, method):
    with np.errstate(over="ignore", invalid="ignore"):
        location = values.mean(0) if method == "standard" else (values.min(0) if method == "minmax" else np.zeros(values.shape[1]))
        scale = values.std(0, ddof=0) if method == "standard" else (values.max(0) - values.min(0) if method == "minmax" else np.ones(values.shape[1]))
    if not np.isfinite(location).all() or not np.isfinite(scale).all(): raise ContractError("Derived train-only scaler statistics overflow")
    constant = np.flatnonzero(scale == 0).tolist(); scale[scale == 0] = 1.
    return {"location": location.tolist(), "scale": scale.tolist(), "constant_columns": constant}


def _prepare(workspace, source_id, arrays, masks, split, physical, source_meta, method):
    interface = _data_interface(physical)
    X, y = np.concatenate([arrays[key] for key in ("x", "u", "d")], axis=1), arrays["next_x"]
    labels = np.empty(len(X), dtype=np.int8)
    for index, part in enumerate(_SPLITS): labels[masks[part]] = index
    identity = {"algorithm": "sampled_transition_preparation", "version": 1, "source_id": source_id,
        "source_metadata_hash": canonical_hash(source_meta), "interface_hash": interface.fingerprint, "scaler": method, "split": split}
    cache_key = canonical_hash(identity)
    for artifact_id, row in workspace.manifest["artifacts"].items():
        if row["kind"] != "supervised_dataset" or row["status"] != "verified": continue
        candidate_root = confined(workspace.path, row["path"])
        candidate = read_json(candidate_root / "artifact.json")
        if candidate.get("source", {}).get("cache_key") != cache_key: continue
        stored, _, _, _, _, stored_interface, _, _ = _read_data(workspace, artifact_id)
        interface.compatible(stored_interface)
        if (not np.array_equal(stored["X"], X) or not np.array_equal(stored["y"], y) or not np.array_equal(stored["split_id"], labels) or
                [item["id"] for item in candidate["parents"]] != [source_id]):
            raise ContractError("Cached prepared dataset no longer matches the bound sampled rows/split")
        return artifact_id, True
    transform = {"method": method, "fit_split": "train", "fit_samples": int(masks["train"].sum()),
        "X": _fit(X[masks["train"]], method), "y": _fit(y[masks["train"]], method)}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        X_scaled = (X - transform["X"]["location"]) / transform["X"]["scale"]
        y_scaled = (y - transform["y"]["location"]) / transform["y"]["scale"]
    if not np.isfinite(X_scaled).all() or not np.isfinite(y_scaled).all(): raise ContractError("Derived scaled arrays are non-finite")
    definition = interface.to_dict(); by_name = {item["name"]: item for item in definition["signals"]}
    feature_order = [{"signal": item["signal"], "lag": 0, "unit": by_name[item["signal"]]["unit"], "availability_delay_steps": 0} for item in definition["features"]]
    target_order = [{"signal": item["signal"], "offset": 1, "unit": by_name[item["signal"]]["unit"]} for item in definition["targets"]]
    with workspace.stage("supervised_dataset", interface, [source_id], identity, compatible_parents=False) as stage:
        np.savez_compressed(stage.root / "supervised.npz", X=X, y=y, X_scaled=X_scaled, y_scaled=y_scaled, split_id=labels,
                            source_row=np.arange(len(X), dtype=np.int64), trajectory_id=arrays["trajectory_id"], time=arrays["time"])
        atomic_json(stage.root / "scaler.json", transform)
        atomic_json(stage.root / "split.json", {"mode": "source_trajectory_partition", "groups": split,
            "sample_counts": {part: int(masks[part].sum()) for part in _SPLITS}, "split_before_windows": True})
        atomic_json(stage.root / "physical_context.json", {"interface": physical.to_dict(), "sampled_dataset": source_id})
        atomic_json(stage.root / "dataset.json", {"schema_version": 1, "interface": definition,
            "interface_kind": "tabular_supervised", "task": "dynamic_prediction", "input_dim": interface.input_dim,
            "output_dim": interface.output_dim, "feature_order": feature_order, "target_order": target_order,
            "source": definition["source_mapping"], "source_mapping": definition["source_mapping"],
            "plant_dimensions_inferred": False, "physical_interface_is_explicit_context": True, "trained": False})
        return stage.publish({"status": "verified", "checks": ["source_array_mapping", "preserved_trajectory_split", "train_only_scaler"]},
            preprocessing=transform, source={**identity, "cache_key": cache_key, "kind": source_meta.get("source", {}).get("kind", "plant_simulator"),
                "description": "Recorded NPZ transitions; not CSV columns or inferred physical dimensions", "original_source": source_meta.get("source", {})}, activate=False), False


def _scale(values):
    result = values.std(0, ddof=0)
    if not np.isfinite(result).all(): raise ContractError("Train-only physical normalization overflows")
    result[result == 0] = 1.
    return result


def _metrics(prediction, truth, scale, variables):
    if len(truth) == 0: return {"samples": 0, "rmse": None, "nrmse": None, "per_variable": []}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=0)); nrmse = rmse / scale
    if not np.isfinite(rmse).all() or not np.isfinite(nrmse).all(): raise FloatingPointError("Physical predictive metrics are non-finite")
    return {"samples": len(truth), "rmse": rmse.tolist(), "nrmse": nrmse.tolist(),
        "per_variable": [{"name": item.name, "unit": item.unit, "rmse": float(rmse[index]), "nrmse": float(nrmse[index])}
                         for index, item in enumerate(variables)]}


def _outside(values, variables):
    return int(np.count_nonzero(np.any((values < np.asarray([v.lower for v in variables]) - 1e-12) |
        (values > np.asarray([v.upper for v in variables]) + 1e-12), axis=-1)))


def _coverage(arrays, train, interface):
    return {"source": "training_partition_only", "samples": int(train.sum()), "joint_coverage_proved": False,
        "extrapolation_validated": False, "marginal_ranges": [{"group": group, "name": variable.name, "unit": variable.unit,
            "observed_min": float(arrays[key][train, index].min()), "observed_max": float(arrays[key][train, index].max()),
            "declared_lower": variable.lower, "declared_upper": variable.upper}
            for key, group in (("x", "states"), ("u", "controls"), ("d", "disturbances"))
            for index, variable in enumerate(getattr(interface, group))]}


def train_predictive_model(workspace: Workspace, sampled_dataset_id: str, config: dict,
                           context: OperationContext | None = None) -> str:
    """Reuse S05 on source trajectory splits, then publish a physical NN wrapper."""
    from .learned_dynamics import export_learned_model, load_learned_model

    context = context or OperationContext()
    with workspace.lock("predictive_training"):
        arrays, masks, split, interface, source_meta = _read_sampled(workspace, sampled_dataset_id)
        C, method, network, limits = _config(config, interface)
        with np.errstate(over="ignore", invalid="ignore"):
            current_y, next_y = arrays["x"] @ C.T, arrays["next_x"] @ C.T
        if (not np.isfinite(current_y).all() or not np.isfinite(next_y).all() or
                not np.allclose(current_y, arrays["y"], rtol=1e-9, atol=1e-11) or not np.allclose(next_y, arrays["next_y"], rtol=1e-9, atol=1e-11)):
            raise ContractError("Declared C is inconsistent with sampled y/next_y; output mapping is not inferred")
        _resolved(network, _data_interface(interface))  # Reject unsupported NN config before creating derived artifacts.
        prepared_id, reused = _prepare(workspace, sampled_dataset_id, arrays, masks, split, interface, source_meta, method)
        supervised_id = train_supervised(workspace, prepared_id, network, context, activate=False)
        supervised_root, supervised_meta = workspace.resolve(supervised_id, allow_rejected=True)
        nn_report = read_json(supervised_root / "report.json")
        # The numerical stage has published and renamed its staging directory.
        # Record its real checkpoint before any cancellation/calibration check.
        context.emit({"event": "progress", "stage": "predictive_training", "supervised_model": supervised_id,
            **checkpoint_fields(current_checkpoint(supervised_root), "published")})
        context.check()
        if supervised_meta["validation"]["status"] != "verified":
            raise ContractError(f"Supervised NN failed its predeclared acceptance; inspect {supervised_id}")
        with workspace.stage("learned_model", interface, [sampled_dataset_id, prepared_id, supervised_id], config, compatible_parents=False) as stage:
            export_learned_model(stage.root, supervised_root, interface, C.tolist(), provenance={
                "sampled_dataset": sampled_dataset_id, "supervised_dataset": prepared_id, "supervised_model": supervised_id,
                "sampled_metadata_hash": canonical_hash(source_meta), "source_mapping": _data_interface(interface).to_dict()["source_mapping"]})
            model = load_learned_model(stage.root); model.eval()
            state_scale, output_scale = _scale(arrays["next_x"][masks["train"]]), _scale(arrays["next_y"][masks["train"]])
            test_rows = np.flatnonzero(masks["test"])
            predictions, outputs = [], []
            batch_size = nn_report["resolved_config"]["batch_size"]
            with torch.no_grad():
                for offset in range(0, len(test_rows), batch_size):
                    context.check(); rows = test_rows[offset:offset + batch_size]
                    prediction = model(torch.from_numpy(arrays["x"][rows]), torch.from_numpy(arrays["u"][rows]), torch.from_numpy(arrays["d"][rows]))
                    predictions.append(prediction.detach().cpu().numpy())
                    outputs.append(model.output(prediction).detach().cpu().numpy())
            predicted_x = np.concatenate(predictions); predicted_y = np.concatenate(outputs)
            if not np.isfinite(predicted_x).all() or not np.isfinite(predicted_y).all(): raise FloatingPointError("Learned one-step predictions are non-finite")
            one_step = {"test": {"state": _metrics(predicted_x, arrays["next_x"][test_rows], state_scale, interface.states),
                "output": _metrics(predicted_y, arrays["next_y"][test_rows], output_scale, interface.outputs)},
                "state_bound_violations": _outside(predicted_x, interface.states), "output_bound_violations": _outside(predicted_y, interface.outputs)}
            np.savez_compressed(stage.root / "one_step_predictions.npz", predicted_state=predicted_x, true_state=arrays["next_x"][test_rows],
                predicted_output=predicted_y, true_output=arrays["next_y"][test_rows], source_row=test_rows,
                input_time=arrays["time"][test_rows], prediction_time=arrays["time"][test_rows] + interface.sample_time)
            rollout_rows, rollout_states, rollout_outputs, trajectory_reports = [], [], [], []
            for group in split["test"]:
                rows = np.flatnonzero(arrays["trajectory_id"] == group); rows = rows[np.argsort(arrays["time"][rows], kind="stable")]
                x, completed, status = arrays["x"][rows[0]].copy(), 0, "completed"
                for row in rows:
                    context.check()
                    try:
                        x = np.asarray(model.step(x, arrays["u"][row], arrays["d"][row]), dtype=np.float64)
                        with torch.no_grad(): out = model.output(torch.from_numpy(x)).cpu().numpy()
                        if x.shape != (interface.nx,) or not np.isfinite(x).all() or not np.isfinite(out).all(): raise FloatingPointError("non-finite rollout")
                    except FloatingPointError:
                        status = "stopped_nonfinite"; break
                    rollout_rows.append(row); rollout_states.append(x.copy()); rollout_outputs.append(out.copy()); completed += 1
                    if _outside(x[None, :], interface.states) or _outside(out[None, :], interface.outputs):
                        status = "stopped_outside_declared_bounds"; break
                trajectory_reports.append({"trajectory_id": group, "requested_steps": len(rows), "completed_steps": completed, "status": status,
                    "initial_state": arrays["x"][rows[0]].tolist(), "teacher_forcing_after_initial_state": False})
            rollout_rows = np.asarray(rollout_rows, dtype=np.int64)
            rollout_states = np.asarray(rollout_states, dtype=np.float64).reshape(-1, interface.nx)
            rollout_outputs = np.asarray(rollout_outputs, dtype=np.float64).reshape(-1, interface.ny)
            rollout = {"test": {"state": _metrics(rollout_states, arrays["next_x"][rollout_rows], state_scale, interface.states),
                "output": _metrics(rollout_outputs, arrays["next_y"][rollout_rows], output_scale, interface.outputs)}, "trajectories": trajectory_reports,
                "all_trajectories_completed": all(row["status"] == "completed" for row in trajectory_reports),
                "prediction_time_semantics": "source.time+sample_time",
                "state_bound_violations": _outside(rollout_states, interface.states), "output_bound_violations": _outside(rollout_outputs, interface.outputs)}
            np.savez_compressed(stage.root / "rollout_predictions.npz", predicted_state=rollout_states, true_state=arrays["next_x"][rollout_rows],
                predicted_output=rollout_outputs, true_output=arrays["next_y"][rollout_rows], source_row=rollout_rows,
                trajectory_id=arrays["trajectory_id"][rollout_rows], input_time=arrays["time"][rollout_rows],
                prediction_time=arrays["time"][rollout_rows] + interface.sample_time)
            failures = []
            for name, metric in (("max_test_state_nrmse", one_step["test"]["state"]), ("max_test_output_nrmse", one_step["test"]["output"]),
                                  ("max_rollout_state_nrmse", rollout["test"]["state"]), ("max_rollout_output_nrmse", rollout["test"]["output"])):
                if limits[name] is not None and (metric["nrmse"] is None or np.any(np.asarray(metric["nrmse"]) > limits[name])): failures.append(name)
            if not rollout["all_trajectories_completed"]: failures.append("rollout_incomplete")
            if one_step["state_bound_violations"] or one_step["output_bound_violations"]: failures.append("one_step_bounds")
            passed = not failures
            accuracy_requested = any(value is not None for value in limits.values()) or nn_report["max_test_nrmse"] is not None
            structural_passed = rollout["all_trajectories_completed"] and not one_step["state_bound_violations"] and not one_step["output_bound_violations"]
            report = {"algorithm": "supervised_discrete_next_state_model", "runtime_time_semantics": "discrete_state_transition",
                "target_semantics": "next_x_in_state_units_no_dt_multiplication", "pinn": False, "physics_constraints_in_training": False,
                "nominal_or_reference_model_required": False, "sampled_dataset": sampled_dataset_id, "supervised_dataset": prepared_id,
                "supervised_dataset_reused": reused, "supervised_model": supervised_id, "C": C.tolist(),
                "C_validation": "consistent_with_recorded_data_uniqueness_not_proved", "physical_interface": interface.to_dict(),
                "one_step": one_step, "rollout": rollout, "normalization": {"source": "training_partition_only", "ddof": 0,
                    "state_nrmse_scale": state_scale.tolist(), "output_nrmse_scale": output_scale.tolist(), "zero_std_fallback": 1.},
                "coverage": _coverage(arrays, masks["train"], interface), "test_used_for_training_or_selection": False,
                "selection": "S05_validation_only", "network": {key: nn_report[key] for key in
                    ("network_config", "resolved_config", "parameter_count", "trained_epochs", "best_epoch", "best_validation_loss", "optimizer_updates", "resumed_from_epoch", "training_signature")},
                "prediction_limits": limits, "acceptance": ("passed" if accuracy_requested else "not_requested") if passed else "failed",
                "structural_finite_bounds_acceptance": "passed" if structural_passed else "failed", "acceptance_failures": failures,
                "resource_boundary": {"max_sampled_npz_bytes": _MAX_BYTES, "max_required_array_values": _MAX_VALUES,
                    "network": nn_report["resource_boundary"], "scope": "one prediction-training job; not the overall engineering goal"},
                "source_provenance": source_meta.get("source", {}), "physical_fidelity": "recorded_simulator_holdout_not_independent_plant_certification",
                "export": {"directory": "bundle", "descriptor": "learned.json"}, "confirmation_sources": context.case_confirmation_sources}
            atomic_json(stage.root / "physical_context.json", {"interface": interface.to_dict(), "C": C.tolist(), "source_split": split})
            atomic_json(stage.root / "report.json", report)
            context.check()
            return stage.publish({"status": "verified" if passed else "rejected", "checks": ["explicit_physical_interface", "recorded_C_consistency",
                "preserved_trajectory_split", "train_only_scaler", "validation_only_NN_selection", "one_step_test", "open_loop_trajectory_test"],
                "acceptance": report["acceptance"], "failures": failures}, source={"algorithm": report["algorithm"], "sampled_dataset": sampled_dataset_id,
                    "supervised_dataset": prepared_id, "supervised_model": supervised_id, "physical_fidelity": report["physical_fidelity"]}, activate=passed)
