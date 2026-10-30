"""Full-state affine least-squares identification with held-out trajectory checks.

This estimates discrete dynamics from transition data; it does not optimize
controller parameters, infer hidden states from outputs, or substitute for GA.
"""
from __future__ import annotations

import copy
import math
import numpy as np

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, ModelInterface, canonical_hash
from .data import read_transitions
from .modeling import DeclarativeDynamics, load_model, tensor
from .runtime import OperationContext


def _independent_split(arrays: dict, masks: dict) -> dict:
    if set(masks) != {"train", "validation", "test"}:
        raise ContractError("Identification requires exactly train/validation/test trajectory splits")
    count = len(arrays["x"])
    membership = np.zeros(count, dtype=int)
    groups = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (count,) or not mask.any():
            raise ContractError(f"{name} must be a nonempty boolean dataset partition")
        membership += mask.astype(int)
        groups[name] = np.unique(arrays["trajectory_id"][mask]).tolist()
    if not np.all(membership == 1):
        raise ContractError("Identification partitions overlap or omit transitions")
    group_ids = [value for values in groups.values() for value in values]
    if len(group_ids) != len(set(group_ids)):
        raise ContractError("A trajectory cannot be shared across identification partitions")
    return groups


def fit_affine_transitions(interface: ModelInterface, arrays: dict, masks: dict, config: dict,
                           context: OperationContext | None = None) -> tuple[dict, dict]:
    """Fit only training rows. Return coefficients and independent partition diagnostics."""
    context = context or OperationContext()
    context.check()
    groups = _independent_split(arrays, masks)
    fit_bias = config.get("fit_bias", True)
    if type(fit_bias) is not bool:
        raise ContractError("fit_bias must be boolean")
    relative_tolerance = float(config.get("rank_relative_tolerance", 1e-10))
    max_condition = float(config.get("max_condition_number", 1e8))
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ContractError("rank_relative_tolerance must be in (0,1)")
    if not math.isfinite(max_condition) or max_condition < 1:
        raise ContractError("max_condition_number must be finite and at least one")
    features = np.concatenate([arrays[key] for key in ("x", "u", "d")], axis=1)
    targets = np.asarray(arrays["next_x"], dtype=float)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ContractError("Identification requires finite state transitions")
    train = masks["train"]
    feature_count = interface.nx + interface.nu + interface.nd
    if features.shape != (len(targets), feature_count) or targets.shape[1:] != (interface.nx,):
        raise ContractError("Identification features do not match x/u/d/next_x interface dimensions")
    center = features[train].mean(axis=0) if fit_bias else np.zeros(feature_count)
    scale = np.sqrt(np.mean((features[train] - center) ** 2, axis=0))
    if np.any(scale <= np.finfo(float).eps):
        raise ContractError("Training inputs/states lack excitation: a regressor has zero variance or magnitude")
    design = (features - center) / scale
    if fit_bias:
        design = np.column_stack((design, np.ones(len(design))))
    parameter_count = design.shape[1]
    if int(train.sum()) <= parameter_count:
        raise ContractError("More independent training rows than regression parameters are required")
    output_scale = np.array([v.upper - v.lower for v in interface.states])
    theta, _, rank, singular = np.linalg.lstsq(design[train], targets[train] / output_scale,
                                             rcond=relative_tolerance)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    if rank != parameter_count or not math.isfinite(condition) or condition > max_condition:
        raise ContractError(f"Training regression is not identifiable: rank={rank}/{parameter_count}, condition={condition}")
    context.progress("linear_identification_fit", training_rows=int(train.sum()), rank=int(rank),
                     parameter_count=parameter_count, condition_number=condition)
    physical = theta[:feature_count] * output_scale[None, :] / scale[:, None]
    bias = theta[-1] * output_scale - center @ physical if fit_bias else np.zeros(interface.nx)
    end_u = interface.nx + interface.nu
    coefficients = {"A": physical[:interface.nx].T.tolist(), "B": physical[interface.nx:end_u].T.tolist(),
                    "E": physical[end_u:].T.tolist(), "bias": bias.tolist()}
    predictions = features @ physical + bias
    metrics = {}
    for name, mask in masks.items():
        context.check()
        errors = predictions[mask] - targets[mask]
        metrics[name] = {"rows": int(mask.sum()), "trajectories": len(groups[name]),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "normalized_rmse": float(np.sqrt(np.mean((errors / output_scale) ** 2))),
            "rmse_per_state": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
            "max_abs_error_per_state": np.max(np.abs(errors), axis=0).tolist()}
    train_errors = predictions[train] - targets[train]
    return coefficients, {"algorithm": "numpy_linalg_lstsq_svd", "fit_bias": fit_bias,
        "training_rank": int(rank), "parameter_count": parameter_count, "condition_number": condition,
        "singular_values": singular.tolist(), "rank_relative_tolerance": relative_tolerance,
        "max_condition_number": max_condition, "trajectory_partitions": groups,
        "split_fingerprint": canonical_hash(groups), "metrics": metrics,
        "training_residual_covariance": (train_errors.T @ train_errors / (int(train.sum()) - parameter_count)).tolist(),
        "preprocessing": {"source": "training_partition_only", "feature_order": [
            v.name for group in (interface.states, interface.controls, interface.disturbances) for v in group],
            "feature_center": center.tolist(), "feature_scale": scale.tolist(),
            "target_scale": output_scale.tolist()},
        "validation_used_for_fitting": False, "test_used_for_fitting": False}


def identify_linear_model(workspace: Workspace, nominal_id: str, dataset_id: str, config: dict,
                          context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    nominal = load_model(workspace, nominal_id)
    if not isinstance(nominal, DeclarativeDynamics):
        raise ContractError("Linear identification requires a declarative source interface/output map")
    if nominal.terms and config.get("allow_linear_approximation") is not True:
        raise ContractError("Identifying an affine approximation to nonlinear source dynamics requires allow_linear_approximation=true")
    arrays, masks, data_meta = read_transitions(workspace, dataset_id)
    nominal.interface.compatible(ModelInterface.parse(data_meta["interface"]))
    parents = [parent["id"] for parent in data_meta["parents"]]
    if nominal_id not in parents:
        raise ContractError("Transition dataset must bind this exact nominal model version")
    limits = {name: float(config.get(f"max_{name}_normalized_rmse", .05)) for name in ("validation", "test")}
    if any(not math.isfinite(value) or value <= 0 for value in limits.values()):
        raise ContractError("Identification validation/test error limits must be finite and positive")
    with workspace.stage("identified_model", nominal.interface, [nominal_id, dataset_id], config) as stage:
        coefficients, report = fit_affine_transitions(nominal.interface, arrays, masks, config, context)
        spec = copy.deepcopy(nominal.spec)
        spec.update(coefficients, terms=[], representation="discrete_polynomial", interface=nominal.interface.to_dict())
        for continuous_field in ("integration", "derivative_units", "time_unit"):
            spec.pop(continuous_field, None)
        identified = DeclarativeDynamics(spec)
        nominal_predictions = nominal(tensor(arrays["x"]), tensor(arrays["u"]), tensor(arrays["d"])).detach().numpy()
        state_scale = np.array([v.upper - v.lower for v in nominal.interface.states])
        report["nominal_metrics"] = {name: {"normalized_rmse": float(np.sqrt(np.mean(
            ((nominal_predictions[mask] - arrays["next_x"][mask]) / state_scale) ** 2)))}
            for name, mask in masks.items()}
        checks = {f"{name}_prediction_error": report["metrics"][name]["normalized_rmse"] <= limit
                  for name, limit in limits.items()}
        radius = float(np.max(np.abs(np.linalg.eigvals(np.asarray(coefficients["A"])))))
        report["spectral_radius_A"] = radius
        if "max_spectral_radius" in config:
            limit = float(config["max_spectral_radius"])
            if not math.isfinite(limit) or limit <= 0:
                raise ContractError("max_spectral_radius must be finite and positive")
            checks["declared_spectral_radius_limit"] = radius <= limit
        report.update(acceptance={"checks": checks, "limits": limits, "passed": all(checks.values())},
                      source_dataset=dataset_id, nominal_model=nominal_id,
                      data_source=data_meta["source"], output_map="C retained from declared source; not fitted",
                      scope="fully_observed_state_affine_transition_identification_not_output_only_system_identification")
        source = {"kind": "identified_affine_state_transition", "dataset": dataset_id,
                  "source_model": nominal_id, "data_source": data_meta["source"],
                  "algorithm": report["algorithm"], "controller_parameters_modified": False,
                  "output_map_source": nominal_id, "nonlinear_terms_replaced": bool(nominal.terms)}
        atomic_json(stage.root / "model.json", spec)
        atomic_json(stage.root / "report.json", report)
        return stage.publish({"status": "verified" if all(checks.values()) else "rejected",
                              "train_regression_full_rank": True, "independent_validation_and_test": True,
                              "validation_checks": checks, "physical_fidelity": "data_and_declared_interface_limited"},
                             preprocessing=report["preprocessing"], source=source)


def load_identified_model(workspace: Workspace, artifact_id: str) -> DeclarativeDynamics:
    root, _ = workspace.resolve(artifact_id, kind="identified_model")
    return DeclarativeDynamics(read_json(root / "model.json"))
