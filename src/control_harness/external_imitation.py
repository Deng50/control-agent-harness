"""External-expert imitation by composition of the existing S05 trainer.

DataInterface widths are regression features/actions, never physical nx. Test
network predictions are reused from S05's final saved prediction file; the
deployment projection is evaluated on recorded causal inputs/previous actions.
This teacher-forced offline check is not an autonomous closed-loop evaluation.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import zipfile

import numpy as np

from .artifacts import atomic_json, confined, read_json
from .contracts import ContractError, canonical_hash
from .external_expert_data import read_external_expert_dataset, validate_expert_interface, _actions
from .runtime import OperationContext
from .supervised import _read_data, _resolved, train_supervised


_MAX_BYTES = 256 * 1024 * 1024
_MAX_VALUES = 8_000_000
_PARTS = ("train", "validation", "test")


def _vector(value, width, label, *, positive):
    if (not isinstance(value, list) or len(value) != width or any(type(item) not in {int, float} or
            not math.isfinite(item) or item < 0 or (positive and item == 0) for item in value)):
        raise ContractError(f"{label} must contain {width} finite {'positive' if positive else 'nonnegative'} values")
    return np.asarray(value, dtype=np.float64)


def _config(config, interface, train_targets):
    allowed = {"network", "max_test_action_nrmse", "action_metric_scale", "resume_checkpoint", "resume_checkpoint_sha256", "plan_refs"}
    if not isinstance(config, dict) or set(config) - allowed or not isinstance(config.get("network"), dict):
        raise ContractError("External imitation requires config.network and supported outer metric/resume fields")
    network = dict(config["network"])
    if set(network) & {"resume_checkpoint", "resume_checkpoint_sha256", "plan_refs", "max_test_nrmse"}:
        raise ContractError("Place resume/plan_refs/action accuracy gate at the outer config, not inside network")
    for key in ("resume_checkpoint", "resume_checkpoint_sha256"):
        if key in config: network[key] = config[key]
    _resolved(network, interface)
    with np.errstate(over="ignore", invalid="ignore"):
        population = train_targets.std(axis=0, ddof=0)
    if not np.isfinite(population).all(): raise ContractError("Training action population standard deviation overflows")
    # Exactly repeated decimal values can acquire a tiny nonzero np.std through
    # mean-rounding (e.g. six copies of 0.2). This is mathematically a constant
    # column; do not normalize its error by that numerical residue. Nonconstant
    # small signals retain their actual std; there is no arbitrary epsilon gate.
    constant = np.all(train_targets == train_targets[0], axis=0)
    population[constant] = 0.
    declared = config.get("action_metric_scale", "train_std")
    if isinstance(declared, str) and declared == "train_std":
        scale = population.copy(); scale[scale == 0] = 1.
        normalization = {"source": "training_partition_only", "ddof": 0, "zero_std_fallback": 1.,
            "zero_std_actions": np.flatnonzero(population == 0).tolist()}
    else:
        scale = _vector(declared, interface.output_dim, "action_metric_scale", positive=True)
        normalization = {"source": "explicit_user_scales", "ddof": 0, "zero_std_fallback": "not_used", "zero_std_actions": []}
    normalization.update(action_nrmse_scale=scale.tolist(), training_action_population_std=population.tolist(),
                         constant_detection="exact equality within training column, not an epsilon threshold",
                         fitted_using_test=False, scales_fixed_before_training=True)
    threshold = config.get("max_test_action_nrmse")
    if threshold is not None:
        if isinstance(threshold, list): threshold = _vector(threshold, interface.output_dim, "max_test_action_nrmse", positive=False)
        else:
            if type(threshold) not in {int, float} or not math.isfinite(threshold) or threshold < 0:
                raise ContractError("max_test_action_nrmse must be a finite nonnegative scalar or one value per action")
            threshold = np.full(interface.output_dim, float(threshold))
    return network, scale, threshold, normalization


def _load_members(path, shapes, expected_sha256, *, optional=False):
    """Inspect exact optional/prior or prediction NPY headers before allocation."""
    if path.stat().st_size > _MAX_BYTES: raise ContractError("Imitation NPZ exceeds its 256-MiB byte budget")
    with path.open("rb") as stream: payload = stream.read(_MAX_BYTES + 1)
    if len(payload) > _MAX_BYTES or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError("Imitation NPZ changed after artifact version verification")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist(); names = [entry.filename for entry in entries]
            if (len(entries) > 32 or len(names) != len(set(names)) or any(not name.endswith(".npy") or "/" in name or "\\" in name for name in names)
                    or sum(item.file_size for item in entries) > _MAX_BYTES):
                raise ContractError("Imitation NPZ has an unsafe inventory or expanded-byte budget")
            selected = [key for key in shapes if key + ".npy" in names]
            if not optional and set(selected) != set(shapes): raise ContractError("Imitation prediction NPZ lacks required arrays")
            total = 0
            for key in selected:
                entry = archive.getinfo(key + ".npy")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else: raise ContractError("Imitation NPZ requires version-1/2 NPY headers")
                    header = stream.tell()
                total += math.prod(shape)
                kinds = "iu" if key == "row_index" else "fiu"
                if (shape != shapes[key] or dtype.kind not in kinds or dtype.hasobject or total > _MAX_VALUES or
                        math.prod(shape) * dtype.itemsize != entry.file_size - header):
                    raise ContractError("Imitation NPZ array shape/dtype/declared allocation disagrees with bounded expected rows")
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            result = {key: np.asarray(loaded[key], dtype=np.int64 if key == "row_index" else np.float64) for key in selected}
    except ContractError: raise
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise ContractError(f"Invalid imitation NPZ: {error}") from error
    if any(not np.isfinite(value).all() for value in result.values()): raise ContractError("Imitation numeric arrays must be finite")
    return result


def _prepare(workspace, source_id, arrays, interface, dataset, scaler, split, source_meta, constraints, physical):
    identity = {"algorithm": "external_expert_supervised_preparation", "version": 1, "external_expert_dataset": source_id,
        "source_metadata_hash": canonical_hash(source_meta), "source_file_hashes": source_meta["files"],
        "interface_hash": interface.fingerprint, "split": split, "scaler": scaler,
        "source_mapping": dataset["source_mapping"], "action_constraints": constraints, "physical_context": physical}
    cache_key = canonical_hash(identity)
    core_keys = ("X", "y", "X_scaled", "y_scaled", "split_id")
    for artifact_id, row in workspace.manifest["artifacts"].items():
        if row["kind"] != "supervised_dataset" or row["status"] != "verified": continue
        root = confined(workspace.path, row["path"]); candidate = read_json(root / "artifact.json")
        if candidate.get("source", {}).get("cache_key") != cache_key: continue
        stored, _, transform, description, metadata, declared, _, _ = _read_data(workspace, artifact_id)
        interface.compatible(declared)
        if (any(not np.array_equal(stored[key], arrays[key]) for key in core_keys) or transform != scaler or
                description.get("source_mapping") != dataset["source_mapping"] or read_json(root / "split.json") != split or
                [parent["id"] for parent in metadata["parents"]] != [source_id]):
            raise ContractError("Cached imitation dataset differs from its bound source arrays/split/scaler/mapping")
        return artifact_id, True
    with workspace.stage("supervised_dataset", interface, [source_id], identity, compatible_parents=False) as stage:
        np.savez_compressed(stage.root / "supervised.npz", **{key: arrays[key] for key in core_keys})
        atomic_json(stage.root / "dataset.json", {**dataset, "kind": "supervised_dataset", "array_file": "supervised.npz",
            "source_row_semantics": "same zero-based row/order as source expert.npz; exact source split_id retained",
            "external_expert_dataset": source_id, "prepared_for": "external_expert_imitation", "trained": False})
        atomic_json(stage.root / "scaler.json", scaler); atomic_json(stage.root / "split.json", split)
        return stage.publish({"status": "verified", "checks": ["exact_source_arrays", "source_partition_preserved", "train_only_scaler", "causal_features"]},
            preprocessing=scaler, source={**identity, "cache_key": cache_key, "kind": "external_expert",
                "description": "Inactive canonical S05 data view of the bound external expert artifact"}, activate=False), False


def _metrics(prediction, truth, scale, target_order):
    with np.errstate(over="ignore", invalid="ignore"):
        rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=0)); nrmse = rmse / scale
    if not np.isfinite(rmse).all() or not np.isfinite(nrmse).all(): raise FloatingPointError("Imitation error metrics are non-finite")
    return {"samples": len(truth), "rmse": rmse.tolist(), "nrmse": nrmse.tolist(),
        "per_action": [{**item, "rmse": float(rmse[index]), "nrmse": float(nrmse[index])} for index, item in enumerate(target_order)]}


def _safety(actions, previous, constraints):
    bounds = np.asarray(constraints["bounds"], dtype=np.float64)
    violation = np.maximum(bounds[:, 0] - actions, 0) + np.maximum(actions - bounds[:, 1], 0)
    report = {"amplitude_violation_count": int(np.count_nonzero(violation > 1e-12)), "max_amplitude_violation": float(violation.max()),
        "out_of_bounds_fraction": float(np.mean(np.any(violation > 1e-12, axis=1))),
        "out_of_bounds_coordinate_fraction": float(np.mean(violation > 1e-12)),
        "rate_violation_count": None, "max_rate_violation": None, "rate_check": "not_requested"}
    if constraints.get("rate_bounds") is not None:
        if previous is None: raise ContractError("Rate assessment requires actual previous applied actions; no implicit zeros")
        rate = np.asarray(constraints["rate_bounds"], dtype=np.float64); delta = actions - previous
        violation = np.maximum(rate[:, 0] - delta, 0) + np.maximum(delta - rate[:, 1], 0)
        report.update(rate_violation_count=int(np.count_nonzero(violation > 1e-12)), max_rate_violation=float(violation.max()), rate_check="recorded_actual_previous_actions")
    return report


def train_external_imitation(workspace, external_expert_dataset_id, config, context=None):
    """Train a real MLP, then package its constrained deployment policy."""
    from .external_policy import load_external_policy
    from .policy_export import export_external_policy

    context = context or OperationContext()
    with workspace.lock("external_expert_imitation"):
        arrays, interface, dataset, scaler, source_meta = read_external_expert_dataset(workspace, external_expert_dataset_id)
        source_root, _ = workspace.resolve(external_expert_dataset_id, kind="external_expert_dataset")
        split = read_json(source_root / "split.json"); constraints = read_json(source_root / "action_constraints.json")
        physical = read_json(source_root / "physical_context.json")
        _, mapping, order = validate_expert_interface(interface, dataset["source_mapping"])
        masks = {name: arrays["split_id"] == index for index, name in enumerate(_PARTS)}
        network, scale, threshold, normalization = _config(config, interface, arrays["y"][masks["train"]])
        # Optional diagnostic prior arrays are shape/header/hash checked before
        # NumPy allocation; otherwise only explicitly mapped causal X can supply
        # the prior. Current/future expert targets are never used as a fallback.
        prior = _load_members(source_root / "expert.npz", {key: arrays["y"].shape for key in ("actual_previous_u", "previous_u")},
                              source_meta["files"]["expert.npz"], optional=True)
        checked = {**arrays, **prior}
        checked_constraints = _actions(checked, interface, {"action_constraints": constraints}, dataset["source"], mapping, order)
        previous = checked.get("actual_previous_u")
        prepared_id, reused = _prepare(workspace, external_expert_dataset_id, arrays, interface, dataset, scaler, split, source_meta, constraints, physical)
        supervised_id = train_supervised(workspace, prepared_id, network, context, activate=False)
        nn_root, nn_meta = workspace.resolve(supervised_id, kind="supervised_model", allow_rejected=True)
        nn_report = read_json(nn_root / "report.json")
        context.emit({"event": "progress", "stage": "external_imitation", "supervised_model": supervised_id,
            "epoch": nn_report["trained_epochs"], "checkpoint": str(nn_root / "latest.pt"), "checkpoint_status": "published"})
        context.check()
        if nn_meta["validation"]["status"] != "verified": raise ContractError("Internal supervised training did not produce a verified model")
        test_rows = np.flatnonzero(masks["test"]); shape = (len(test_rows), interface.output_dim)
        prediction = _load_members(nn_root / "test_predictions.npz", {"prediction": shape, "truth": shape, "row_index": (len(test_rows),)},
                                   nn_meta["files"]["test_predictions.npz"])
        if not np.array_equal(prediction["row_index"], test_rows) or not np.array_equal(prediction["truth"], arrays["y"][test_rows]):
            raise ContractError("S05 final predictions do not address the exact source test rows")
        raw = prediction["prediction"]; test_previous = None if previous is None else previous[test_rows]
        with workspace.stage("external_imitation_policy", interface, [external_expert_dataset_id, prepared_id, supervised_id], config, compatible_parents=False) as stage:
            manifest = export_external_policy(stage.root / "bundle", nn_root, interface.to_dict(), constraints, physical,
                provenance={"external_expert_dataset": external_expert_dataset_id, "supervised_dataset": prepared_id,
                    "supervised_model": supervised_id, "source_metadata_hash": canonical_hash(source_meta),
                    "source_mapping": dataset["source_mapping"], "deployment_scope": dataset["deployment_scope"], "source": source_meta.get("source", {})})
            policy = load_external_policy(stage.root / "bundle")
            projected = np.asarray(policy.project_actions(raw, test_previous), dtype=np.float64)
            if projected.shape != raw.shape or not np.isfinite(projected).all(): raise FloatingPointError("Deployment action projection is non-finite or has the wrong shape")
            raw_metrics = _metrics(raw, prediction["truth"], scale, order["target_order"])
            projected_metrics = _metrics(projected, prediction["truth"], scale, order["target_order"])
            raw_safety, projected_safety = _safety(raw, test_previous, checked_constraints), _safety(projected, test_previous, checked_constraints)
            safe = projected_safety["amplitude_violation_count"] == 0 and projected_safety["rate_violation_count"] in (None, 0)
            accuracy_passed = threshold is None or bool(np.all(np.asarray(projected_metrics["nrmse"]) <= threshold))
            passed = safe and accuracy_passed
            changed = np.abs(projected - raw) > 1e-12
            np.savez_compressed(stage.root / "test_predictions.npz", raw_action=raw, projected_action=projected,
                expert_action=prediction["truth"], source_row=test_rows,
                **({"actual_previous_u": test_previous} if test_previous is not None else {}))
            descriptor = {"schema_version": 1, "representation": "external_supervised_policy", "bundle": "bundle",
                "external_expert_dataset": external_expert_dataset_id, "supervised_dataset": prepared_id, "supervised_model": supervised_id,
                "interface": interface.to_dict(), "source_mapping": dataset["source_mapping"], "closed_loop": "not_run"}
            atomic_json(stage.root / "policy.json", descriptor)
            train_validation = {}
            for part in ("train", "validation"):
                rmse = np.asarray(nn_report[part]["rmse"]); nrmse = rmse / scale
                train_validation[part] = {"raw": {"samples": nn_report[part]["samples"], "rmse": rmse.tolist(), "nrmse": nrmse.tolist(),
                    "per_action": [{**item, "rmse": float(rmse[index]), "nrmse": float(nrmse[index])} for index, item in enumerate(order["target_order"])],
                    "normalization": "outer_action_metric_scale"}}
            report = {"algorithm": "external_expert_supervised_imitation", "external_expert_dataset": external_expert_dataset_id,
                "supervised_dataset": prepared_id, "supervised_dataset_reused": reused, "supervised_model": supervised_id,
                "internal_controller_required": False, "physical_model_required": False, "physical_context": physical,
                "source_provenance": source_meta.get("source", {}), "deployment_scope": dataset["deployment_scope"],
                "feature_order": order["feature_order"], "target_order": order["target_order"], "normalization": normalization,
                "selection": "S05_validation_only", "test_used_for_training_or_selection": False, "test_evaluations": 1,
                "test_network_prediction_source": "reused_S05_final_test_predictions_no_second_test_forward",
                "offline": {**train_validation, "test": {"raw": raw_metrics, "projected": projected_metrics,
                    "raw_safety": raw_safety, "projected_safety": projected_safety,
                    "projection_modified_rows": int(np.count_nonzero(changed.any(axis=1))), "projection_modified_coordinates": int(changed.sum()),
                    "previous_action_source": checked_constraints["previous_action_evidence"],
                    "mode": "teacher_forced_recorded_features_and_actual_previous_actions", "autonomous_closed_loop": False}},
                "action_constraints": checked_constraints, "max_test_action_nrmse": None if threshold is None else threshold.tolist(),
                "acceptance_metric": "projected_test_action_nrmse", "acceptance": ("passed" if threshold is not None else "not_requested") if passed else "failed",
                "raw_out_of_bounds_fraction": raw_safety["out_of_bounds_fraction"],
                "structural_projection_acceptance": "passed" if safe else "failed", "raw_metrics_gate_requested": False,
                "acceptance_failures": ([] if safe else ["projected_action_constraints"]) + ([] if accuracy_passed else ["max_test_action_nrmse"]),
                "network": {key: nn_report[key] for key in ("network_config", "resolved_config", "parameter_count", "trained_epochs", "best_epoch",
                    "best_validation_loss", "optimizer_updates", "resumed_from_epoch", "training_signature", "resource_boundary")},
                "closed_loop": "not_run", "pgac": "not_run", "training_summary": "real CPU float64 Adam MLP via S05 train_supervised",
                "export": {"directory": "bundle", "manifest": "bundle/policy_manifest.json", "details": manifest},
                "confirmation_sources": context.case_confirmation_sources}
            atomic_json(stage.root / "report.json", report)
            atomic_json(stage.root / "training_summary.json", {key: report[key] for key in ("algorithm", "network", "normalization", "acceptance", "closed_loop", "pgac")})
            context.check()
            return stage.publish({"status": "verified" if passed else "rejected", "checks": ["causal_external_data", "exact_source_split",
                "train_only_scaler", "validation_only_selection", "saved_final_test_predictions", "raw_and_projected_metrics", "deployment_action_projection"],
                "acceptance": report["acceptance"], "failures": report["acceptance_failures"], "closed_loop": "not_run"},
                preprocessing=scaler, source={"algorithm": report["algorithm"], "external_expert_dataset": external_expert_dataset_id,
                    "supervised_dataset": prepared_id, "supervised_model": supervised_id, "source_provenance": source_meta.get("source", {}),
                    "confirmation_sources": context.case_confirmation_sources}, activate=passed)
