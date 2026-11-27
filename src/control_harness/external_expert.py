"""Direct external-expert/Plant collection with explicit causal observations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .artifacts import Workspace, atomic_json, confined, file_hash, read_json
from .contracts import ContractError, DataInterface, MaterialInterface, ModelInterface, canonical_hash
from .data import trajectory_split
from .expert_adapter import ExpertFailure, load_external_expert, validate_expert_spec, validate_python_expert_source
from .pgac_scenarios import SCHEDULE_FIELDS, condition_hash, normalize_schedule, observation_noise, scenario_series
from .runtime import Cancelled, OperationContext


class ExpertCollectionStopped(RuntimeError):
    def __init__(self, message, arrays, report, diagnostics):
        super().__init__(message)
        self.arrays, self.report, self.diagnostics = arrays, report, diagnostics


class ExpertCollectionCancelled(ExpertCollectionStopped, Cancelled):
    """Cooperative cancellation retains its valid prefix and native cancel type."""


def register_expert(workspace, material_id, config, context=None):
    context = context or OperationContext()
    if not isinstance(config, dict) or set(config) != {"adapter"}: raise ContractError("Expert registration requires config.adapter only")
    root, metadata = workspace.resolve(material_id, kind="material")
    material = read_json(root / "material.json")
    spec = validate_expert_spec(config["adapter"])
    relative = spec["source_file"]
    if relative not in material["file_hashes"] or file_hash(confined(root, relative)) != material["file_hashes"][relative]:
        raise ContractError("Expert source must belong to the exact immutable material closure")
    if spec["backend"] == "python_callable": support = validate_python_expert_source(root, spec, metadata["files"])
    else:
        from .matlab_expert import validate_matlab_expert_source
        support = validate_matlab_expert_source(root, spec, metadata["files"])
    context.check()
    with workspace.stage("external_expert", MaterialInterface("bundle", "External explicit-memory expert"), [material_id], config, compatible_parents=False) as stage:
        atomic_json(stage.root / "expert.json", {"schema_version": 1, "adapter_spec": spec, "material": material_id,
            "material_content_hash": material["content_hash"]})
        atomic_json(stage.root / "report.json", {"kind": "external_expert_registration", "physical_interface": spec["interface"],
            "source_executed": False, "source_support": support, "status_contract": "string 'ok' is the only success",
            "controller_design": "not_run", "internal_memory": "explicit reset per trajectory; diagnostics only",
            "availability": "user-declared interface, not proof of measured data or a deployed estimator"})
        return stage.publish({"status": "verified", "checks": ["static_expert_contract", "source_hash", "explicit_observation_action_mapping"],
            "source_execution": "not_run"}, source={"kind": "declared_external_expert", "backend": spec["backend"],
                "material": material_id, "content_hash": material["content_hash"], "source_executed": False}, activate=False)


def _expert_selection(workspace, expert_id):
    root, metadata = workspace.resolve(expert_id, kind="external_expert")
    definition = read_json(root / "expert.json")
    spec = validate_expert_spec(definition["adapter_spec"])
    material_id = definition["material"]
    if material_id not in [row["id"] for row in metadata["parents"]]: raise ContractError("Expert material is not a bound parent")
    source, material_meta = workspace.resolve(material_id, kind="material")
    if read_json(source / "material.json")["content_hash"] != definition["material_content_hash"]:
        raise ContractError("Expert material content identity changed")
    return spec, source, metadata, material_meta


def _plant_selection(workspace, plant_id):
    from .plants import _selection
    root, metadata = workspace.resolve(plant_id)
    if metadata["kind"] == "learned_model":
        return ModelInterface.parse(metadata["interface"]), None, None, metadata, None
    return _selection(workspace, plant_id)


def _vector(value, width, label):
    if not isinstance(value, list) or len(value) != width or any(type(item) not in {int, float} or not math.isfinite(item) for item in value):
        raise ContractError(label + " must be an explicit finite numeric JSON vector")
    return [0. if float(item) == 0 else float(item) for item in value]


def _bounded(value, variables, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(variables),) or not np.isfinite(array).all(): raise ExpertFailure(label + " is nonfinite or has wrong dimensions")
    lower = np.asarray([item.lower for item in variables])
    upper = np.asarray([item.upper for item in variables])
    outside = (array < lower - 1e-12) | (array > upper + 1e-12)
    if outside.any():
        error = ExpertFailure(label + " violates declared physical bounds")
        error.bound_violations = [
            {"signal": variables[k].name, "unit": variables[k].unit, "value": float(array[k]),
             "lower": float(lower[k]), "upper": float(upper[k]),
             "excess": float(max(lower[k] - array[k], array[k] - upper[k]))}
            for k in np.flatnonzero(outside)]
        raise error
    return array


def build_expert_collection_plan(physical, expert_spec, config):
    interface = physical if isinstance(physical, ModelInterface) else ModelInterface.parse(physical)
    spec = validate_expert_spec(expert_spec); interface.compatible(ModelInterface.parse(spec["interface"]))
    if not isinstance(config, dict) or set(config) - {"scenarios", "seed", "plan_refs", "student_features", "scaler"}:
        raise ContractError("External collection requires explicit scenarios/seed and declared feature/scaler settings")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not 5 <= len(scenarios) <= 10000:
        raise ContractError("External expert collection requires at least five independent physical conditions")
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1: raise ContractError("seed must be an integer in 0..2**32-1")
    trajectories, seen, total = [], set(), 0
    explicit_split = any(isinstance(row, dict) and "split" in row for row in scenarios)
    partitions = {name: [] for name in ("train", "validation", "test")}
    for index, scenario in enumerate(scenarios):
        if (not isinstance(scenario, dict) or set(scenario) - SCHEDULE_FIELDS - {"name", "initial_state", "reference", "disturbance", "steps", "initial_previous_action", "split"}
                or type(scenario.get("steps")) is not int or not 1 <= scenario["steps"] <= 100000):
            raise ContractError("Unknown scenario fields or invalid explicit control-update steps")
        if "name" in scenario and (not isinstance(scenario["name"], str) or not scenario["name"].strip()): raise ContractError("Scenario name must be nonempty text")
        fields = {"x0": _vector(scenario.get("initial_state"), interface.nx, "initial_state"),
            "reference": _vector(scenario.get("reference"), interface.ny, "reference"),
            "disturbance": _vector(scenario.get("disturbance"), interface.nd, "disturbance"),
            "initial_action": _vector(scenario.get("initial_previous_action"), interface.nu, "initial_previous_action")}
        for key, variables in (("x0", interface.states), ("reference", interface.outputs), ("disturbance", interface.disturbances), ("initial_action", interface.controls)):
            try: _bounded(fields[key], variables, key)
            except ExpertFailure as error: raise ContractError(str(error)) from error
        fields.update(normalize_schedule(scenario, interface, spec))
        fingerprint = condition_hash({**fields, "steps": scenario["steps"]})
        if fingerprint in seen: raise ContractError("Duplicate physical expert conditions cannot be separated by names, duration, split or numeric representation")
        seen.add(fingerprint); total += scenario["steps"]
        group = scenario.get("split")
        if explicit_split and group not in partitions: raise ContractError("Every scenario needs an explicit train/validation/test split when any split is declared")
        if explicit_split: partitions[group].append(index)
        trajectories.append({"trajectory_id": index, "steps": scenario["steps"], **fields, "physical_case_hash": fingerprint})
    substeps = int(round(spec["control_period"] / interface.sample_time))
    logical_width = 2*interface.nx + 3*interface.nu + 2*interface.ny + interface.nd + len(spec["observations"]) + 8
    if total > 100000 or total * substeps > 500000 or total * logical_width > 5_000_000:
        raise ContractError("Expert collection exceeds one-job control/subtick/array budget; use smaller explicit batches")
    if explicit_split:
        if any(not ids for ids in partitions.values()): raise ContractError("Explicit whole-trajectory splits must all be nonempty")
    else: partitions = trajectory_split(np.arange(len(trajectories), dtype=np.int64), seed)
    membership = {trajectory: name for name, values in partitions.items() for trajectory in values}
    for row in trajectories:
        row["time"] = [step * spec["control_period"] for step in range(row["steps"])]
        row["split"] = membership[row["trajectory_id"]]
    plan = {"schema_version": 1, "interface": interface.to_dict(), "control_period": spec["control_period"], "plant_substeps": substeps,
        "trajectories": trajectories, "split": partitions, "split_origin": "explicit_scenario_partition" if explicit_split else "newly_partitioned_whole_trajectories",
        "seed": seed, "expert_spec_hash": canonical_hash(spec), "plant_dt": interface.sample_time,
        "resource_boundary": {"max_control_steps": 100000, "max_plant_subticks": 500000, "max_array_scalars": 5_000_000, "max_diagnostic_bytes": 8*1024*1024,
            "scope": "one collection job; not a limit on the overall engineering objective"}}
    plan["plan_hash"] = canonical_hash(plan)
    return plan


def expert_data_interface(spec, config):
    physical = ModelInterface.parse(spec["interface"])
    signals, mappings, candidates, default = [], {}, {}, []
    def add(name, unit, role, expert_role, array, column, available, availability=None):
        signal = {"name": name, "unit": unit, "role": role, "expert_role": expert_role, "observable": bool(available), "delay_steps": 0,
            "column": {"index": len(signals), "header": f"NPZ {array}[{column}]", "worksheet": None}}
        if availability is not None: signal["availability"] = availability
        signals.append(signal); mappings[name] = {"array": array, "column": column}; candidates[name] = signal
        if available: default.append(name)
    for index, obs in enumerate(spec["observations"]):
        add("observation::" + obs["name"], obs["unit"], "auxiliary", "observation", "observation", index,
            obs["availability"] != "simulation_only", obs["availability"])
    for variables, role, prefix, array, flag in ((physical.outputs, "reference", "reference", "reference", "reference_available"),
            (physical.disturbances, "disturbance", "disturbance", "d", "disturbance_available"),
            (physical.controls, "previous_action", "previous_action", "previous_u", "previous_action_available")):
        for index, variable in enumerate(variables):
            name = prefix + "::" + variable.name
            add(name, variable.unit, "disturbance" if role == "disturbance" else "auxiliary", role, array, index, spec[flag])
            if role == "previous_action": mappings[name].update(action_target="expert_action::" + variable.name, time_semantics="already_previous_applied_action")
    selected = config.get("student_features", default)
    if not isinstance(selected, list) or not selected or any(not isinstance(name, str) or name not in candidates for name in selected) or len(set(selected)) != len(selected):
        raise ContractError("student_features must explicitly select available causal observation/reference/disturbance/previous-action signals; raw state/memory/current targets are not implicit features")
    for name in selected:
        signal = candidates[name]
        if not signal["observable"] and signal.get("availability") != "simulation_only":
            raise ContractError("Unavailable expert interface signals cannot be supplied as student features")
        signal["observable"] = True  # Explicit simulation-only selection remains labeled, never default.
    targets = []
    for index, variable in enumerate(physical.controls):
        name = "expert_action::" + variable.name
        signals.append({"name": name, "unit": variable.unit, "role": "control", "expert_role": "expert_action", "observable": spec["previous_action_available"],
            "delay_steps": 0, "column": {"index": len(signals), "header": f"NPZ u[{index}]", "worksheet": None}})
        mappings[name] = {"array": "u", "column": index}; targets.append({"signal": name, "offset": 0})
    source_mapping = {"file": "expert.npz", "signals": mappings, "time": {"array": "time", "unit": "s"}, "trajectory": {"array": "trajectory_id"},
        "row_semantics": "control decision at k: causal available features -> expert applied action at k"}
    value = {"schema_version": 1, "interface_kind": "tabular_supervised", "task": "static_regression", "name": "Declared causal external expert policy data",
        "sample_time": spec["control_period"], "signals": signals, "features": [{"signal": name, "window": 1, "lag": 0} for name in selected], "targets": targets,
        "time_column": {"index": len(signals), "header": "NPZ time", "worksheet": None}, "trip_column": {"index": len(signals)+1, "header": "NPZ trajectory_id", "worksheet": None},
        "source_format": "npz_external_expert_decisions", "column_semantics": "logical_signal_index_not_tabular_source_column", "source_mapping": source_mapping}
    return DataInterface.parse(value), source_mapping


def _rows_arrays(rows, spec):
    i = ModelInterface.parse(spec["interface"])
    widths = {"observation": len(spec["observations"]), "reference": i.ny, "d": i.nd, "previous_u": i.nu, "u": i.nu,
        "x": i.nx, "next_x": i.nx, "plant_y": i.ny, "next_y": i.ny}
    arrays = {key: np.asarray(rows[key], dtype=np.float64).reshape(-1, width) if width else np.empty((len(rows["time"]), 0), dtype=np.float64) for key, width in widths.items()}
    arrays.update(trajectory_id=np.asarray(rows["trajectory_id"], dtype=np.int64), time=np.asarray(rows["time"], dtype=np.float64))
    return arrays


def _collect_python(plant, expert, plan, context, journal=None):
    i, spec = plant.interface, expert.spec
    rows = {key: [] for key in ("observation", "reference", "d", "previous_u", "u", "x", "next_x", "plant_y", "next_y", "trajectory_id", "time")}
    diagnostics, summaries, diagnostic_bytes = [], [], 0
    trajectory = step = 0
    try:
        for index, condition in enumerate(plan["trajectories"]):
            trajectory = condition["trajectory_id"]; completed = 0
            context.check()
            restored = journal.load(index, condition) if journal else None
            if restored is not None:
                committed, prior_diagnostics = restored
                for key in rows:
                    rows[key].extend(committed[key])
                diagnostics.extend(prior_diagnostics)
                diagnostic_bytes += sum(len(json.dumps(item, allow_nan=False).encode()) for item in prior_diagnostics)
                journal.commit(index, condition, committed, prior_diagnostics)
                summaries.append({"trajectory_id": trajectory, "requested_steps": condition["steps"],
                                  "completed_steps": condition["steps"], "status": "completed"})
                continue
            context.check(); plant.reset(condition["x0"]); expert.reset()
            previous = np.asarray(condition["initial_action"], dtype=np.float64)
            references = scenario_series(condition, "reference")
            disturbances = scenario_series(condition, "disturbance")
            noise = observation_noise(condition, len(spec["observations"]))
            for step in range(condition["steps"]):
                reference, disturbance = references[step], disturbances[step]
                context.check()
                x = _bounded(plant.state, i.states, "current state"); y = _bounded(plant.output(), i.outputs, "current output")
                observation = np.asarray([x[item["index"]] if item["source"] == "state" else y[item["index"]] for item in spec["observations"]], dtype=np.float64)
                observation += noise[step]
                action, diagnostic = expert.act(observation, reference, disturbance, previous)
                for tick in range(plan["plant_substeps"]):
                    context.check(); xn = _bounded(plant.step(action, disturbance), i.states, "next state at Plant subtick")
                    yn = _bounded(plant.output(), i.outputs, "next output at Plant subtick")
                diagnostic.update(trajectory_id=trajectory, step_index=step, time=condition["time"][step])
                size = len(json.dumps(diagnostic, allow_nan=False).encode())
                if diagnostic_bytes + size > plan["resource_boundary"]["max_diagnostic_bytes"]: raise ExpertFailure("Expert memory diagnostic budget exceeded")
                record = {"x": x, "next_x": xn, "plant_y": y, "next_y": yn, "observation": observation, "reference": reference,
                    "d": disturbance, "previous_u": previous.copy(), "u": action.copy(), "trajectory_id": trajectory, "time": condition["time"][step]}
                for key, value in record.items(): rows[key].append(value)
                diagnostics.append(diagnostic); diagnostic_bytes += size; completed += 1; previous = action.copy()
            summaries.append({"trajectory_id": trajectory, "requested_steps": condition["steps"], "completed_steps": completed, "status": "completed"})
            if journal:
                journal.commit(index, condition, _rows_arrays({key: values[-completed:] for key, values in rows.items()}, spec),
                               diagnostics[-completed:])
            context.progress("external_expert_collection", trajectory=trajectory, completed_trajectories=len(summaries), samples=len(rows["time"]))
    except Exception as error:
        completed = sum(value == trajectory for value in rows["trajectory_id"])
        after_completed = bool(summaries and summaries[-1]["trajectory_id"] == trajectory and summaries[-1]["status"] == "completed")
        if not after_completed: summaries.append({"trajectory_id": trajectory, "completed_steps": completed, "status": "partial" if completed else "failed"})
        report = {"status": "partial" if rows["time"] else "failed", "trajectories": summaries,
            "failure": {"trajectory_id": trajectory, "step_index": completed, "phase": "after_completed_trajectory" if after_completed else "reset_or_control_step", "type": type(error).__name__, "message": str(error), "expert_status": getattr(error, "status", None)},
            "valid_rows": len(rows["time"]), "failure_policy": "stop", "later_trajectories_started": False}
        if hasattr(error, "bound_violations"):
            report["failure"]["observed_bound_violations"] = error.bound_violations
        if hasattr(expert, "source_invocations"):
            report["expert_source_invocations_this_attempt"] = expert.source_invocations
        kind = ExpertCollectionCancelled if isinstance(error, Cancelled) else ExpertCollectionStopped
        stopped = kind(str(error), _rows_arrays(rows, spec), report, diagnostics)
        raise stopped from error
    return {"status": "completed", "arrays": _rows_arrays(rows, spec), "diagnostics": diagnostics,
        "report": {"status": "completed", "trajectories": summaries, "valid_rows": len(rows["time"]), "failure_policy": "stop", "expert": expert.describe(),
            "plant": plant.describe(), "plant_subticks_per_control": plan["plant_substeps"], "control_period": plan["control_period"], "plant_dt": i.sample_time}}


def _matlab_phase(workspace, expert_spec, expert_source, expert_meta, expert_material, plant_spec, plant_source, plant_meta, plant_material, plan, context):
    from .matlab_expert import prepare_matlab_expert_manifest, load_matlab_expert_batch
    phase = os.environ.get("CONTROL_MATLAB_PHASE")
    raw = context._case_raw_request
    digest, nonce = os.environ.get("CONTROL_MATLAB_REQUEST_SHA256", ""), os.environ.get("CONTROL_MATLAB_NONCE", "")
    if not context._case_receipt_body or phase not in {"prepare", "finalize"} or not raw or hashlib.sha256(raw.encode()).hexdigest() != digest or not re.fullmatch(r"[a-f0-9-]{36}", nonce):
        raise ContractError("External MATLAB collection requires one request-bound native Host MCP batch; direct Python does not start MATLAB")
    exchange = Path(os.environ.get("CONTROL_MATLAB_EXCHANGE_DIR", "")).resolve()
    if exchange.parent != (workspace.path / ".case-plans" / "matlab-exchanges").resolve() or not re.fullmatch(r"[a-f0-9-]{36}", exchange.name):
        raise ContractError("External expert MATLAB exchange must be this workspace's private fixed directory")
    def version(metadata): return {"id": metadata["id"], "metadata_hash": workspace.manifest["artifacts"][metadata["id"]]["metadata_hash"]}
    versions = {"request_sha256": digest, "nonce": nonce, "expert_version": version(expert_meta), "expert_material_version": version(expert_material),
        "plant_version": version(plant_meta), "plant_material_version": version(plant_material)}
    if phase == "prepare":
        result = prepare_matlab_expert_manifest(expert_spec, expert_source, plant_spec, plant_source, plan, exchange_dir=exchange, **versions)
        for meta in (expert_meta, expert_material, plant_meta, plant_material): workspace.resolve(meta["id"])
        return result
    path = Path(os.environ.get("CONTROL_MATLAB_RESULT_PATH", "")).resolve()
    if path.parent != exchange: raise ContractError("External expert result must belong to its fixed private exchange")
    return load_matlab_expert_batch(expert_spec, plant_spec, plan, result_path=path,
        result_sha256=os.environ.get("CONTROL_MATLAB_RESULT_SHA256", ""), **versions)


def collect_external_expert(workspace, expert_id, plant_id, config, context=None, request=None):
    context = context or OperationContext()
    spec, expert_source, expert_meta, expert_material = _expert_selection(workspace, expert_id)
    physical, plant_spec, plant_source, plant_meta, plant_material = _plant_selection(workspace, plant_id)
    plan = build_expert_collection_plan(physical, spec, config)
    interface, source_mapping = expert_data_interface(spec, config)
    matlab_expert = spec["backend"] == "matlab_function"
    matlab_plant = plant_spec is not None and plant_spec["backend"] in {"matlab_function", "simulink"}
    if matlab_expert != matlab_plant:
        raise ContractError("Mixed Python/MATLAB external expert and Plant backends require an explicit supported bridge; this operation never starts Engine or pointwise MCP")
    parents = [expert_id, plant_id]
    with workspace.lock("collect_external_expert"):
        result = None
        if matlab_expert:
            result = _matlab_phase(workspace, spec, expert_source, expert_meta, expert_material, plant_spec, plant_source, plant_meta, plant_material, plan, context)
            if result.get("matlab_phase") == "prepared": return result
        with workspace.stage("external_expert_dataset", interface, parents, config, compatible_parents=False) as stage:
            atomic_json(stage.root / "collection-plan.json", plan)
            expert = plant = None
            try:
                if result is None:
                    from .plants import _TorchAdapter
                    from .model_adapters import load_python_adapter
                    expert = load_external_expert(expert_source, spec, expected_file_hashes=expert_material["files"])
                    plant = (load_python_adapter(plant_source, plant_spec, expected_file_hashes=plant_material["files"])
                        if plant_spec else _TorchAdapter(workspace, plant_id))
                    from .trajectory_journal import TrajectoryJournal
                    journal = TrajectoryJournal(stage, context, plan, physical.sample_time)
                    result = _collect_python(plant, expert, plan, context, journal)
                    result["report"]["restored_trajectories"] = journal.restored
                if result["status"] != "completed":
                    raise ExpertCollectionStopped("External expert batch stopped; valid prefix is not a trainable dataset", result["arrays"], result["report"], result["diagnostics"])
                arrays = result["arrays"]
                columns = [source_mapping["signals"][feature["signal"]] for feature in interface.to_dict()["features"]]
                arrays["X"] = np.column_stack([arrays[item["array"]][:, item["column"]] for item in columns])
                arrays["y"] = arrays["u"].copy()
                simulation_only = any(signal.get("availability") == "simulation_only" and any(feature["signal"] == signal["name"] for feature in interface.to_dict()["features"]) for signal in interface.to_dict()["signals"])
                source = {"kind": "declared_simulation", "description": "External expert controls the explicitly selected simulator; no internal controller design or real measurement acquisition",
                    "expert": expert_id, "plant": plant_id, "expert_backend": spec["backend"], "plant_backend": plant_spec["backend"] if plant_spec else "registered_torch",
                    "measured": False, "previous_action_semantics": "actual_applied_action_before_current_decision",
                    "student_feature_scope": "simulation_only" if simulation_only else "user_declared_available_not_independently_certified",
                    "memory_is_student_feature": False, "split_origin": plan["split_origin"]}
                report = {**result["report"], "plan_hash": plan["plan_hash"], "controller_design": "not_run", "internal_MPC_required": False,
                    "training": "not_run", "hardware_execution": False, "confirmation_sources": context.case_confirmation_sources,
                    "control_period": plan["control_period"], "plant_dt": physical.sample_time, "plant_substeps": plan["plant_substeps"]}
                from .external_expert_data import publish_external_expert_dataset
                return publish_external_expert_dataset(workspace, arrays, interface, plan["split"],
                    {"scaler": config.get("scaler", "standard"), "action_constraints": {"bounds": [[v.lower, v.upper] for v in physical.controls],
                        "rate_bounds": spec.get("input_rate_bounds"), "rate_semantics": "per_control_step"},
                        "physical_mapping": {"plant_dt": physical.sample_time, "control_period": plan["control_period"], "plant_substeps": plan["plant_substeps"],
                            "observations": spec["observations"], "action_mapping": spec["action_mapping"]}}, context,
                    parents=parents, source=source, physical_interface=physical.to_dict(), source_mapping=source_mapping,
                    diagnostics=result["diagnostics"], report=report, stage=stage)
            except ExpertCollectionStopped as error:
                np.savez_compressed(stage.root / "partial-expert.npz", **error.arrays)
                atomic_json(stage.root / "diagnostics.json", error.diagnostics)
                atomic_json(stage.root / "report.json", {**error.report, "plan_hash": plan["plan_hash"], "controller_design": "not_run", "training": "not_run",
                    "data_source": "declared_simulation", "measured": False, "confirmation_sources": context.case_confirmation_sources})
                raise
            finally:
                if expert is not None: expert.close()
                if plant is not None: plant.close()
