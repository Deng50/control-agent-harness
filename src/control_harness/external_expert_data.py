"""Causal external-expert data, independent of an internal MPC or physical nx.

The common writer accepts complete collector arrays or the existing tabular
preparer's row-level partition proof. It never trains, executes an expert, or
infers actuator/plant constraints from observations. Observability/provenance
are explicit user declarations, not a hardware observability proof.
"""
from __future__ import annotations

from contextlib import nullcontext
import copy
import json
import re

import numpy as np

from .artifacts import atomic_json, read_json
from .contracts import ContractError, DataInterface, ModelInterface
from .datasets import prepare_dataset, _fit_scaler
from .runtime import OperationContext


SPLITS = ("train", "validation", "test")
EXPERT_ROLES = {"observation", "reference", "disturbance", "previous_action", "expert_action", "diagnostic_memory"}
MAX_SCALARS = 8_000_000
MAX_BYTES = 256 * 1024 * 1024
PREVIOUS = "actual_applied_action_before_current_decision"


def _json(value, label):
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (ValueError, TypeError) as exc:
        raise ContractError(f"{label} must be finite JSON") from exc


def _orders(interface):
    definition = interface.to_dict()
    signals = {item["name"]: item for item in definition["signals"]}
    features = [{"signal": item["signal"], "lag": item["lag"] + signals[item["signal"]]["delay_steps"] + history,
                 "unit": signals[item["signal"]]["unit"],
                 "availability_delay_steps": signals[item["signal"]]["delay_steps"]}
                for item in definition["features"] for history in reversed(range(item["window"]))]
    targets = [{"signal": item["signal"], "offset": item["offset"], "unit": signals[item["signal"]]["unit"]}
               for item in definition["targets"]]
    return signals, features, targets


def validate_expert_interface(interface, source_mapping=None):
    """Validate deployment causality; action history is allowed, memory is not."""
    interface = interface if isinstance(interface, DataInterface) else DataInterface.parse(interface)
    definition = interface.to_dict()
    mapping = _json(source_mapping if source_mapping is not None else definition.get("source_mapping", {}), "source_mapping")
    if not isinstance(mapping, dict) or not isinstance(mapping.get("signals", {}), dict):
        raise ContractError("source_mapping.signals must be an object")
    signals, features, targets = _orders(interface)
    for signal in signals.values():
        if not isinstance(signal.get("expert_role"), str) or signal["expert_role"] not in EXPERT_ROLES:
            raise ContractError("Every expert signal needs an explicit expert_role")
    target_names = {item["signal"] for item in targets}
    for target in targets:
        if target["offset"] != 0 or signals[target["signal"]]["expert_role"] != "expert_action":
            raise ContractError("Expert targets must be current expert_action at offset 0")
        if signals[target["signal"]]["role"] != "control":
            raise ContractError("Expert action targets must declare control role")
    for item in features:
        signal = signals[item["signal"]]
        role = signal["expert_role"]
        if role == "diagnostic_memory":
            raise ContractError("Diagnostic expert memory cannot be a deployment feature")
        if role == "expert_action" and (item["signal"] not in target_names or item["lag"] < 1):
            raise ContractError("Only explicitly historical expert action may be a feature")
        if role == "previous_action":
            evidence = mapping.get("signals", {}).get(item["signal"], {})
            if (not isinstance(evidence, dict) or evidence.get("time_semantics") != "already_previous_applied_action" or
                    evidence.get("action_target") not in target_names):
                raise ContractError("previous_action column requires explicit already_previous_applied_action and action_target mapping")
    simulation_only = any(signals[item["signal"]].get("availability") == "simulation_only" for item in features)
    return interface, mapping, {"feature_order": features, "target_order": targets,
        "deployment_scope": "simulation_only" if simulation_only else "declared_observable_inputs",
        "observability_is_user_declared": True, "diagnostic_memory_is_feature": False}


def _source(source):
    source = _json(source, "source")
    if (not isinstance(source, dict) or not isinstance(source.get("kind"), str) or source["kind"] not in {"recorded_external_expert", "declared_simulation"} or
            not isinstance(source.get("description"), str) or not source["description"].strip()):
        raise ContractError("Expert source requires recorded_external_expert/declared_simulation and a description")
    if not isinstance(source.get("previous_action_semantics"), str) or source["previous_action_semantics"] not in {PREVIOUS, "not_recorded"}:
        raise ContractError("Declare actual previous-action semantics or not_recorded explicitly")
    if any(key in source for key in ("controller_id", "internal_controller_id", "controller")):
        raise ContractError("External expert data cannot forge an internal controller identity")
    if source["kind"] == "declared_simulation":
        if source.get("measured", False) is not False:
            raise ContractError("Declared simulation cannot claim real measured acquisition")
        source["measured"] = False
    return source


def _array_id(value):
    if isinstance(value, np.integer): value = int(value)
    if type(value) is int: return ("int", value)
    if isinstance(value, str) and value and len(value) <= 256: return ("str", value)
    raise ContractError("Trajectory IDs must be explicit integers or nonempty strings (<=256 characters)")


def _arrays(raw, interface):
    if not isinstance(raw, dict) or not 2 <= len(raw) <= 32:
        raise ContractError("Expert arrays require X/y and at most 32 arrays")
    arrays, count, size = {}, 0, 0
    for key, value in raw.items():
        if not isinstance(key, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None:
            raise ContractError("Expert array names must be plain identifiers")
        # Native ndarray/header sizes are checked before conversion or copies.
        # Lists are an in-process collector convenience, not an unbounded file reader.
        try: array = np.asarray(value)
        except (TypeError, ValueError) as exc: raise ContractError(f"Expert array {key} is ragged or not numeric") from exc
        if array.ndim > 2 or array.dtype.kind not in "fiuU":
            raise ContractError(f"Expert array {key} must be rank <=2 real numeric (IDs may be Unicode)")
        if array.dtype.kind == "U" and key not in {"trajectory_id", "group_id"}:
            raise ContractError("Text is allowed only in explicit trajectory/group identifiers")
        count += array.size; size += array.nbytes
        if count > MAX_SCALARS or size > MAX_BYTES:
            raise ContractError("Expert dataset exceeds the 8-million-scalar/256-MiB one-job budget; import in bounded batches")
        if array.dtype.kind != "U" and not np.isfinite(array).all():
            raise ContractError(f"Expert array {key} contains non-finite values")
        arrays[key] = array
    for key, width in (("X", interface.input_dim), ("y", interface.output_dim)):
        if key not in arrays or arrays[key].dtype.kind == "U" or arrays[key].ndim != 2 or arrays[key].shape[1] != width:
            raise ContractError(f"{key} shape must match the declared DataInterface")
    rows = len(arrays["X"])
    if rows == 0 or any(array.ndim == 0 or len(array) != rows for array in arrays.values()):
        raise ContractError("All expert arrays must have the same nonzero row count")
    # Account for the two transformed copies before allocating them.
    if count + arrays["X"].size + arrays["y"].size > MAX_SCALARS:
        raise ContractError("Expert dataset including transformed arrays exceeds the logical scalar budget")
    if size + (arrays["X"].size + arrays["y"].size) * 8 > MAX_BYTES:
        raise ContractError("Expert dataset including transformed arrays exceeds the raw-byte budget")
    arrays["X"] = arrays["X"].astype(np.float64, copy=False)
    arrays["y"] = arrays["y"].astype(np.float64, copy=False)
    return arrays


def _partition(arrays, split, interface):
    split = _json(split, "split")
    n = len(arrays["X"])
    if not isinstance(split, dict): raise ContractError("Explicit expert split required")
    if "raw_rows" in split:
        # Proof emitted by prepare_dataset: preserve chronological partitions,
        # including same original trip in multiple partitions, without pretending
        # these are independent trajectory holdouts.
        if split.get("mode") not in {"time", "trip"} or split.get("split_before_windowing") is not True:
            raise ContractError("Tabular split requires split-before-windowing evidence")
        if not all(key in arrays for key in ("split_id", "feature_rows", "target_rows", "anchor_row", "group_id")):
            raise ContractError("Tabular split requires source-row provenance arrays")
        if arrays["anchor_row"].shape != (n,) or arrays["anchor_row"].dtype.kind not in "iu" or arrays["group_id"].shape != (n,):
            raise ContractError("Tabular anchor/group provenance requires one value per sample")
        labels = arrays["split_id"]
        if labels.ndim != 1 or labels.dtype.kind not in "iu" or set(labels.tolist()) != {0, 1, 2}:
            raise ContractError("All three nonempty expert partitions are required")
        raw_sets = []
        for index, part in enumerate(SPLITS):
            rows = split.get("raw_rows", {}).get(part)
            if not isinstance(rows, list) or not rows or any(type(row) is not int or row < 1 for row in rows) or len(rows) != len(set(rows)):
                raise ContractError("Raw source partition rows must be unique positive integers")
            allowed = set(rows); raw_sets.append(allowed)
            mask = labels == index
            if split.get("sample_counts", {}).get(part) != int(mask.sum()):
                raise ContractError("Tabular split sample count mismatch")
            for key, width in (("feature_rows", interface.input_dim), ("target_rows", interface.output_dim)):
                if arrays[key].shape != (n, width) or arrays[key].dtype.kind not in "iu" or not set(arrays[key][mask].flat).issubset(allowed):
                    raise ContractError("Window provenance crosses a raw source partition")
            if not set(arrays["anchor_row"][mask].tolist()).issubset(allowed):
                raise ContractError("Anchor row crosses a raw source partition")
        if any(raw_sets[i] & raw_sets[j] for i in range(3) for j in range(i)):
            raise ContractError("Raw expert partitions overlap")
        split["holdout_semantics"] = "independent_source_trips" if split["mode"] == "trip" else "chronological_partitions_within_source_trips"
        arrays.setdefault("trajectory_id", arrays["group_id"].copy())
    else:
        groups = split.get("groups", split)
        if not isinstance(groups, dict) or any(part not in groups for part in SPLITS):
            raise ContractError("Predefined train/validation/test trajectory IDs are required")
        if "trajectory_id" not in arrays or arrays["trajectory_id"].ndim != 1:
            raise ContractError("Collection requires one trajectory_id per row")
        keys = [_array_id(value) for value in arrays["trajectory_id"]]
        assignment, normalized = {}, {}
        for index, part in enumerate(SPLITS):
            ids = groups[part]
            if not isinstance(ids, list) or not ids: raise ContractError("All three trajectory partitions must be nonempty")
            normalized[part] = []
            for value in ids:
                key = _array_id(value)
                if key in assignment: raise ContractError("Expert trajectory partitions overlap or duplicate IDs")
                assignment[key] = index; normalized[part].append(value)
        if set(keys) != set(assignment): raise ContractError("Expert split must exactly cover recorded trajectories")
        labels = np.asarray([assignment[key] for key in keys], dtype=np.int8)
        if "split_id" in arrays and not np.array_equal(labels, arrays["split_id"]):
            raise ContractError("Provided split_id conflicts with predefined trajectory split")
        if set(labels.tolist()) != {0, 1, 2}: raise ContractError("All three expert partitions require samples")
        if "time" not in arrays or arrays["time"].shape != (n,) or arrays["time"].dtype.kind == "U":
            raise ContractError("Collection requires explicit decision time per row")
        dt = interface.to_dict().get("sample_time")
        for key in set(keys):
            times = arrays["time"][[item == key for item in keys]]
            if np.any(np.diff(times) <= 0) or (dt is not None and not np.allclose(np.diff(times), dt, atol=1e-9, rtol=1e-6)):
                raise ContractError("Expert decision times must be increasing and match declared control sample_time")
        split = {"mode": "predefined_trajectory", "groups": normalized, "split_before_windowing": True,
                 "holdout_semantics": "declared_source_trajectory_partition",
                 "sample_counts": {part: int((labels == index).sum()) for index, part in enumerate(SPLITS)}}
    arrays["split_id"] = np.asarray(labels, dtype=np.int8)
    return split


def _bounds(value, width, label):
    try: raw = np.asarray(value)
    except (TypeError, ValueError) as exc: raise ContractError(f"{label} must be a rectangular numeric matrix") from exc
    if raw.dtype.kind not in "fiu" or raw.shape != (width, 2) or not np.isfinite(raw).all():
        raise ContractError(f"Explicit {label} must be finite [action_width,2]")
    result = raw.astype(np.float64)
    if np.any(result[:, 0] >= result[:, 1]): raise ContractError(f"{label} must have strictly ordered lower/upper pairs")
    return result


def _array_mapping(arrays, mapping, order):
    """Check collector logical selectors against the arrays actually archived."""
    for role, matrix, entries in (("feature", "X", order["feature_order"]), ("target", "y", order["target_order"])):
        for index, entry in enumerate(entries):
            selector = mapping.get("signals", {}).get(entry["signal"], {})
            if not isinstance(selector, dict): raise ContractError("Signal source mapping must be an object")
            if "array" not in selector: continue  # Original CSV/XLSX columns use row provenance instead.
            name, column = selector["array"], selector.get("column")
            if (not isinstance(name, str) or name not in arrays or arrays[name].ndim != 2 or type(column) is not int or
                    not 0 <= column < arrays[name].shape[1] or arrays[name].dtype.kind == "U"):
                raise ContractError("Logical expert source selector does not address an archived numeric array")
            if role == "feature" and entry["lag"] != 0:
                raise ContractError("Collected array history requires explicit window preparation; import source rows or provide a thin causal adapter")
            if not np.array_equal(arrays[matrix][:, index], arrays[name][:, column]):
                raise ContractError(f"Expert {role} values disagree with their declared raw-array mapping")


def _actions(arrays, interface, config, source, mapping, order):
    constraints = config.get("action_constraints")
    if not isinstance(constraints, dict): raise ContractError("Explicit action_constraints are required; data extrema are not bounds")
    if "rate_bounds" not in constraints:
        raise ContractError("Declare rate_bounds explicitly (null when not requested)")
    if constraints.get("rate_semantics") != "per_control_step":
        raise ContractError("Action rate semantics must be per_control_step")
    bounds = _bounds(constraints.get("bounds"), interface.output_dim, "action bounds")
    if np.any((arrays["y"] < bounds[:, 0] - 1e-12) | (arrays["y"] > bounds[:, 1] + 1e-12)):
        raise ContractError("Expert target action violates declared actuator bounds")
    prev = None; evidence = None
    for key in ("actual_previous_u", "previous_u"):
        if key in arrays:
            candidate = arrays[key]
            if candidate.shape != arrays["y"].shape or candidate.dtype.kind == "U":
                raise ContractError(f"{key} must match the ordered action width")
            if prev is not None and not np.array_equal(prev, candidate):
                raise ContractError("Previous-action evidence arrays disagree")
            prev, evidence = candidate, key
    signals, features, targets = _orders(interface)
    has_previous_features = any(signals[item["signal"]]["expert_role"] in {"previous_action", "expert_action"} for item in features)
    if prev is None:
        columns = []
        for target in targets:
            choices = []
            for index, feature in enumerate(features):
                signal = signals[feature["signal"]]
                if ((signal["expert_role"] == "previous_action" and feature["lag"] == 0 and
                     mapping.get("signals", {}).get(feature["signal"], {}).get("action_target") == target["signal"]) or
                    (feature["signal"] == target["signal"] and feature["lag"] == 1)):
                    choices.append(index)
            if not choices: break
            first = arrays["X"][:, choices[0]]
            if any(not np.array_equal(first, arrays["X"][:, item]) for item in choices[1:]):
                raise ContractError("Mapped previous action features disagree")
            columns.append(first)
        if len(columns) == interface.output_dim:
            prev = np.column_stack(columns); evidence = "explicit_causal_feature_mapping"
    if (prev is not None or has_previous_features) and source["previous_action_semantics"] != PREVIOUS:
        raise ContractError("Previous command evidence requires actual_applied_action_before_current_decision provenance")
    rate = constraints.get("rate_bounds")
    if rate is not None:
        rate = _bounds(rate, interface.output_dim, "rate bounds")
        if prev is None:
            raise ContractError("Rate constraints require actual previous actions for every row; initial zero is not inferred")
        if np.any((arrays["y"] - prev < rate[:, 0] - 1e-12) | (arrays["y"] - prev > rate[:, 1] + 1e-12)):
            raise ContractError("Expert action increment violates declared rate bounds")
    if prev is not None:
        if np.any((prev < bounds[:, 0] - 1e-12) | (prev > bounds[:, 1] + 1e-12)):
            raise ContractError("Previous actual action violates declared actuator bounds")
        arrays["actual_previous_u"] = np.asarray(prev, dtype=np.float64)
    return {"bounds": bounds.tolist(), "rate_bounds": None if rate is None else rate.tolist(),
        "rate_semantics": "per_control_step", "target_order": order["target_order"],
        "constraints_are_explicit": True, "inferred_from_data_extrema": False,
        "previous_action_evidence": evidence, "amplitude_check": "passed",
        "rate_check": "not_requested" if rate is None else "passed"}


def publish_external_expert_dataset(workspace, arrays, interface, split, config, context=None, *,
        parents=(), source, physical_interface=None, source_mapping=None, diagnostics=None, report=None, stage=None):
    """Publish complete, checked arrays; collectors may supply their owned stage.

    This private-code ``stage`` argument is not a request-JSON authority. Failed
    collectors archive prefixes in their own failed stage and must not call this
    writer with incomplete data. No internal controller artifact is required.
    """
    context = context or OperationContext()
    context.check()
    interface, mapping, order = validate_expert_interface(interface, source_mapping)
    source = _source(source)
    physical = None if physical_interface is None else (physical_interface if isinstance(physical_interface, ModelInterface) else ModelInterface.parse(physical_interface)).to_dict()
    arrays = _arrays(arrays, interface)
    split = _partition(arrays, split, interface)
    _array_mapping(arrays, mapping, order)
    constraints = _actions(arrays, interface, config, source, mapping, order)
    method = config.get("scaler", "standard")
    if not isinstance(method, str) or method not in {"standard", "minmax", "none"}: raise ContractError("Unknown expert scaler")
    train = arrays["split_id"] == 0
    scaler = {"method": method, "fit_split": "train", "fit_samples": int(train.sum()),
              "X": _fit_scaler(arrays["X"][train], method), "y": _fit_scaler(arrays["y"][train], method)}
    for key in ("X", "y"):
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = (arrays[key] - scaler[key]["location"]) / scaler[key]["scale"]
        if not np.isfinite(transformed).all(): raise ContractError("Expert scaling overflows finite numbers")
        if f"{key}_scaled" in arrays and not np.allclose(arrays[f"{key}_scaled"], transformed, atol=1e-12, rtol=1e-12):
            raise ContractError("Supplied transformed arrays disagree with train-only scaler")
        arrays[f"{key}_scaled"] = transformed
    # Derived prior-action and scaled arrays also count against the final budget.
    if sum(value.size for value in arrays.values()) > MAX_SCALARS or sum(value.nbytes for value in arrays.values()) > MAX_BYTES:
        raise ContractError("Final expert arrays exceed the one-job resource budget")
    diagnostics = _json(diagnostics, "diagnostics")
    extra = {} if report is None else _json(report, "report")
    if not isinstance(extra, dict): raise ContractError("Collector report must be an object")
    if extra.get("status", "completed") != "completed":
        raise ContractError("Only complete expert collections may publish trainable data; archive partial prefixes as failed stages")
    confirmations = copy.deepcopy(getattr(context, "case_confirmation_sources", []))
    physical_mapping = _json(config.get("physical_mapping"), "physical_mapping")
    if physical_mapping is not None and not isinstance(physical_mapping, dict):
        raise ContractError("physical_mapping must be a separate explicit object")
    physical_context = {"interface": physical, "physical_mapping": physical_mapping, "inferred_from_feature_width": False}
    if physical_mapping is not None:
        physical_context.update({key: physical_mapping[key] for key in ("plant_dt", "control_period", "plant_substeps") if key in physical_mapping})
    if any(key in physical_context for key in ("plant_dt", "control_period", "plant_substeps")):
        plant_dt, control_dt, subticks = (physical_context.get(key) for key in ("plant_dt", "control_period", "plant_substeps"))
        if (physical is None or type(plant_dt) not in {int, float} or type(control_dt) not in {int, float} or
                type(subticks) is not int or subticks < 1 or plant_dt <= 0 or control_dt <= 0 or
                not np.isclose(plant_dt, physical["sample_time"], rtol=1e-12, atol=1e-12) or
                not np.isclose(control_dt, interface.to_dict().get("sample_time", -1), rtol=1e-12, atol=1e-12) or
                not np.isclose(control_dt, plant_dt * subticks, rtol=1e-12, atol=1e-12)):
            raise ContractError("Explicit Plant/expert control periods and substeps must agree without changing Plant dt")
    metadata = {"schema_version": 1, "kind": "external_expert_dataset", "interface_kind": "tabular_supervised",
        "interface": interface.to_dict(), "task": interface.to_dict()["task"],
        "input_dim": interface.input_dim, "output_dim": interface.output_dim,
        **order, "source": source, "source_mapping": mapping, "trained": False,
        "plant_dimensions_inferred": False, "physical_interface_is_explicit_context": physical is not None,
        "array_file": "expert.npz", "target_semantics": "current_expert_command", "target_offset": 0,
        "previous_action_feature_semantics": "explicit_source_mapping_without_automatic_extra_shift"}
    if stage is not None:
        if stage.workspace is not workspace or stage.kind != "external_expert_dataset" or stage.committed:
            raise ContractError("Collector stage must be an unpublished external_expert_dataset in this workspace")
        interface.compatible(stage.interface)
        if {item["id"] for item in stage.parents} != set(parents):
            raise ContractError("Collector stage parent dependencies differ from writer parents")
    manager = nullcontext(stage) if stage is not None else workspace.stage("external_expert_dataset", interface, list(parents), config, compatible_parents=False)
    with manager as owned:
        context.check()
        np.savez_compressed(owned.root / "expert.npz", **arrays)
        for name, value in (("dataset", metadata), ("scaler", scaler), ("split", split),
                            ("action_constraints", constraints), ("physical_context", physical_context),
                            ("diagnostics", diagnostics)):
            atomic_json(owned.root / f"{name}.json", value)
        atomic_json(owned.root / "report.json", {**extra, "kind": "external_expert_data_preparation", "training": "not_run",
            "source": source, "dimensions": {"input_dim": interface.input_dim, "output_dim": interface.output_dim},
            "deployment_scope": order["deployment_scope"], "split": split, "action_constraints": constraints,
            "physical_model": "not_provided" if physical is None else "explicit_context_only",
            "confirmation_sources": confirmations, "diagnostic_memory_is_feature": False})
        return owned.publish({"status": "verified", "checks": ["explicit_expert_signal_semantics", "causal_features",
            "source_partition_preserved", "train_only_scaler", "finite_bounded_arrays", "explicit_action_constraints"],
            "training": "not_run"}, preprocessing=scaler, source={**source,
            "provenance_is_user_declared": True, "confirmation_sources": confirmations})


def import_expert_data(workspace, material_id, config, context=None):
    """Prepare immutable CSV/XLSX expert records through the S03 preparer."""
    context = context or OperationContext()
    interface, mapping, _ = validate_expert_interface(config.get("interface"), config.get("source_mapping"))
    source = _source(config.get("source"))
    root, metadata = workspace.resolve(material_id, kind="material")
    material = read_json(root / "material.json")
    if material.get("format") not in {"csv", "xlsx"}:
        raise ContractError("Expert table import requires CSV or XLSX material")
    context.check()
    prepared = prepare_dataset(root / material["primary_file"], interface, config["split"],
        scaler=config.get("scaler", "standard"), **{key: config[key] for key in
            ("worksheet", "header_row", "delimiter", "encoding") if key in config})
    # Original positional selectors stay explicit; source-row provenance is
    # retained in expert.npz. No physical time is fabricated for static tables.
    mapping = {**mapping, "source_format": material["format"], "material_artifact": material_id,
        "material_content_hash": material["content_hash"], "column_semantics": "original_tabular_source_column",
        "signals": {item["name"]: {**mapping.get("signals", {}).get(item["name"], {}), "column": item["column"]}
                    for item in interface.to_dict()["signals"]}}
    return publish_external_expert_dataset(workspace, prepared.arrays, interface, prepared.split, config, context,
        parents=[material_id], source={**source, "material_artifact": material_id, "content_hash": material["content_hash"]},
        physical_interface=config.get("physical_interface"), source_mapping=mapping,
        report={"tabular_statistics": prepared.statistics, "original_source": prepared.metadata["source"],
                "source_rows_are_one_based_table_records": True})


def read_external_expert_dataset(workspace, artifact_id):
    """S10-ready bounded core arrays; Workspace validates every artifact byte."""
    from .supervised import _load_bounded_npz
    root, metadata = workspace.resolve(artifact_id, kind="external_expert_dataset")
    interface = DataInterface.parse(metadata["interface"])
    dataset = read_json(root / "dataset.json")
    validate_expert_interface(interface, dataset["source_mapping"])
    arrays = _load_bounded_npz(root / "expert.npz", interface, metadata["files"]["expert.npz"])
    interface.compatible(DataInterface.parse(dataset.get("interface")))
    if dataset.get("input_dim") != interface.input_dim or dataset.get("output_dim") != interface.output_dim:
        raise ContractError("Expert dataset dimensions disagree with its DataInterface")
    scaler = read_json(root / "scaler.json")
    if (not isinstance(scaler, dict) or scaler.get("method") not in {"standard", "minmax", "none"} or
            scaler.get("fit_split") != "train" or type(scaler.get("fit_samples")) is not int or
            scaler["fit_samples"] != int((arrays["split_id"] == 0).sum())):
        raise ContractError("Expert scaler must identify its training-only fit")
    for key in ("X", "y"):
        expected = _fit_scaler(arrays[key][arrays["split_id"] == 0], scaler["method"])
        item = scaler.get(key)
        if not isinstance(item, dict): raise ContractError("Expert scaler needs explicit X/y transformations")
        for field in ("location", "scale"):
            raw = np.asarray(item.get(field))
            if (raw.dtype.kind not in "fiu" or raw.shape != (arrays[key].shape[1],) or
                    not np.isfinite(raw).all() or not np.allclose(raw, expected[field], atol=1e-12, rtol=1e-12)):
                raise ContractError("Expert scaler disagrees with train-only population statistics")
        if item.get("constant_columns") != expected["constant_columns"]:
            raise ContractError("Expert scaler constant-column declaration disagrees with training data")
        with np.errstate(over="ignore", invalid="ignore"):
            expected_scaled = (arrays[key] - item["location"]) / item["scale"]
        if not np.isfinite(expected_scaled).all() or not np.allclose(expected_scaled, arrays[key + "_scaled"], atol=1e-12, rtol=1e-12):
            raise ContractError("Expert transformed arrays disagree with their scaler")
    return arrays, interface, dataset, scaler, metadata
