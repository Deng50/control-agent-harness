"""Conditional thermal-ODE PINN with real time autodiff and epoch checkpoints.

The raw MLP maps normalized [time, initial temperature, constant power] to
normalized temperature. This is a bounded conditional trajectory solution,
not a one-step Plant, arbitrary forcing model, general PDE, or analytic solver.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import torch

from .artifacts import atomic_json
from .checkpoints import load_checkpoint, restore_training, save_checkpoint, seed_training, training_contract
from .contracts import ContractError, MaterialInterface, canonical_hash
from .pinn_contract import validate_pinn_definition, SPLIT_IDS
from .runtime import OperationContext
from .supervised import _resolved
from .supervised_export import build_supervised_network


_MAX_ACTIVATIONS = 5_000_000
_DERIVATIVE_FACTOR = 8
_GATES = ("max_test_trajectory_nrmse", "max_normalized_physics_rms", "max_initial_normalized_error")


def _number(value, label, *, positive=False):
    try: valid = type(value) in {int, float} and math.isfinite(float(value)) and value >= 0 and (not positive or value > 0)
    except (OverflowError, ValueError): valid = False
    if not valid:
        raise ContractError(f"{label} must be a finite {'positive' if positive else 'nonnegative'} number")
    return float(value)


def resolve_pinn_config(config):
    """Normalize explicit training options without reading a problem or training."""
    allowed = {"network", "loss_weights", "ablation", "acceptance", "conditional_parameterization", "resume_checkpoint", "resume_checkpoint_sha256", "plan_refs"}
    if not isinstance(config, dict) or set(config) - allowed or not isinstance(config.get("network"), dict):
        raise ContractError("PINN training requires config.network and supported physics/initial/acceptance options")
    network = dict(config["network"])
    if set(network) & {"resume_checkpoint", "resume_checkpoint_sha256", "plan_refs", "max_test_nrmse"}:
        raise ContractError("PINN resume/refs belong at top level; supervised max_test_nrmse is not a PINN gate")
    if network.get("activation", "tanh") not in {"tanh", "silu"}:
        raise ContractError("Conditional thermal PINN requires smooth tanh or silu activation")
    for key in ("resume_checkpoint", "resume_checkpoint_sha256"):
        if key in config: network[key] = config[key]
    resolved, architecture, parameters = _resolved(network, SimpleNamespace(input_dim=3, output_dim=1))
    resolved.pop("max_test_nrmse")
    resolved["loss"] = "weighted_normalized_data_physics_initial_mse"
    weights = config.get("loss_weights", {"data": 1., "physics": 1., "initial": 10.})
    if not isinstance(weights, dict) or set(weights) != {"data", "physics", "initial"}:
        raise ContractError("loss_weights must explicitly contain data, physics and initial")
    weights = {key: _number(value, f"loss_weights.{key}", positive=key == "initial") for key, value in weights.items()}
    if weights["data"] + weights["physics"] == 0: raise ContractError("At least data or physics loss must be enabled in addition to initial conditions")
    if not math.isfinite(sum(weights.values())): raise ContractError("PINN loss weight sum must be finite")
    ablation = config.get("ablation", False)
    if type(ablation) is not bool or ablation != (weights["physics"] == 0):
        raise ContractError("physics=0 requires explicit ablation=true; physics-on training must use ablation=false")
    gates = config.get("acceptance")
    if gates is None: gates = {}
    if not isinstance(gates, dict) or set(gates) - set(_GATES): raise ContractError("Unknown PINN acceptance gate")
    gates = {key: None if gates.get(key) is None else _number(gates[key], key) for key in _GATES}
    parameterization = config.get("conditional_parameterization", "free_mlp")
    if not isinstance(parameterization, str) or parameterization not in {"free_mlp", "affine_initial_power"}:
        raise ContractError("conditional_parameterization must be free_mlp or affine_initial_power")
    return {"network": resolved, "network_config": architecture, "parameter_count": parameters,
        "loss_weights": weights, "ablation": ablation, "acceptance": gates, "conditional_parameterization": parameterization,
        "optimizer_update_semantics": "one Adam update per complete epoch after full-mean chunked gradient accumulation"}


def normalized_features(time_s, initial_temperature_K, power_W, definition):
    n = definition["normalization"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.column_stack(((np.asarray(time_s).reshape(-1) - n["time_origin_s"]) / n["time_scale_s"],
            (np.asarray(initial_temperature_K).reshape(-1) - n["state_offset_K"]) / n["state_scale_K"],
            np.asarray(power_W).reshape(-1) / n["power_scale_W"]))
    if result.dtype.kind not in "fiu" or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ContractError("Normalized conditional PINN inputs must be finite [s,a,b]")
    return torch.from_numpy(np.asarray(result, dtype=np.float64))


def _prediction(network, points):
    value = network(points)
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float64 or
            value.shape != (len(points), 1) or not torch.isfinite(value).all()):
        raise ContractError("PINN output must be a finite CPU float64 normalized-temperature column")
    return value


@torch.enable_grad()
def _physics_prediction(network, points, definition, *, create_graph):
    # Autograd differentiates independent point rows with respect to s. No
    # finite differences, analytic reference derivative or precomputed label is
    # substituted for the network derivative during training/evaluation.
    z = points.detach().clone().requires_grad_(True)
    theta = _prediction(network, z)
    if not theta.requires_grad: raise ContractError("PINN normalized output is detached from the differentiable time input")
    try:
        gradient, = torch.autograd.grad(theta, z, torch.ones_like(theta), create_graph=create_graph, allow_unused=True)
    except RuntimeError as error:
        raise ContractError("PINN time derivative is disconnected from the network graph") from error
    if gradient is None or not torch.isfinite(gradient).all(): raise ContractError("PINN time derivative is disconnected or non-finite")
    p, n = definition["equation"]["parameters"], definition["normalization"]
    dT_dt = (n["state_scale_K"] / n["time_scale_s"]) * gradient[:, :1]
    power = z[:, 2:3] * n["power_scale_W"]
    residual = (p["C_J_per_K"] * dT_dt + p["G_W_per_K"] * n["state_scale_K"] * theta - power) / n["physics_scale_W"]
    if not torch.isfinite(residual).all(): raise FloatingPointError("PINN physical residual overflows finite SI values")
    return residual, theta


def normalized_physics_residual(network, points, definition, *, create_graph=True):
    """Actual [C*dT/dt+G*(T-Ta)-P]/physics_scale with the declared chain rule."""
    return _physics_prediction(network, points, definition, create_graph=create_graph)[0]


def _component(network, name, points, target, definition, *, create_graph):
    if name == "physics": residual = normalized_physics_residual(network, points, definition, create_graph=create_graph)
    elif name == "initial": residual = _prediction(network, points) - points[:, 1:2]
    else: residual = _prediction(network, points) - target
    loss = residual.square().mean()
    if not torch.isfinite(loss): raise FloatingPointError(f"PINN {name} loss is non-finite")
    return loss


def _gradients(network):
    return torch.cat([(torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).reshape(-1) for parameter in network.parameters()])


def _mean_component(network, name, points, target, definition, batch_size, context, *, backward=False, weight=1.):
    total = 0.
    for offset in range(0, len(points), batch_size):
        context.check(); chunk = points[offset:offset + batch_size]
        with torch.set_grad_enabled(backward or name == "physics"):
            loss = _component(network, name, chunk, None if target is None else target[offset:offset + batch_size], definition, create_graph=backward)
        factor = len(chunk) / len(points); total += float(loss.detach()) * factor
        if backward: (loss * (weight * factor)).backward()
    return total


def gradient_probe(network, components, definition, batch_size, weights, context=None):
    """Full-training-set raw/weighted parameter-gradient contribution evidence."""
    context = context or OperationContext()
    report, combined = {}, torch.zeros(sum(p.numel() for p in network.parameters()), dtype=torch.float64)
    for name, (points, target) in components.items():
        network.zero_grad(set_to_none=True)
        loss = _mean_component(network, name, points, target, definition, batch_size, context, backward=True)
        flat = _gradients(network).detach(); norm = float(torch.linalg.vector_norm(flat))
        if not math.isfinite(norm): raise FloatingPointError("PINN component parameter gradient is non-finite")
        combined.add_(flat, alpha=weights[name])
        report[name] = {"raw_loss": loss, "weight": weights[name], "enabled_in_optimizer": weights[name] > 0,
            "raw_parameter_gradient_l2": norm, "weighted_parameter_gradient_l2": norm * weights[name],
            "raw_gradient_nonzero": bool(torch.any(flat != 0)), "per_parameter": [
                {"name": key, "l2": float(torch.linalg.vector_norm(torch.zeros_like(parameter) if parameter.grad is None else parameter.grad))}
                for key, parameter in network.named_parameters()]}
    network.zero_grad(set_to_none=True)
    return {"components": report, "weighted_total_parameter_gradient_l2": float(torch.linalg.vector_norm(combined)),
        "points": {name: len(value[0]) for name, value in components.items()}, "parameter_count": len(combined),
        "physics_derivative_create_graph": True, "gradients_with_respect_to": "actual raw-network trainable parameters",
        "meaning": "Raw diagnostics are computed even for disabled terms; only enabled weighted gradients enter optimizer updates"}


def _components(arrays, definition):
    z = normalized_features(arrays["time_s"], arrays["initial_temperature_K"], arrays["power_W"], definition)
    n = definition["normalization"]; target = torch.from_numpy((arrays["temperature_K"] - n["state_offset_K"]) / n["state_scale_K"])
    collocation = normalized_features(arrays["collocation_time_s"], arrays["collocation_initial_temperature_K"], arrays["collocation_power_W"], definition)
    initial = normalized_features(arrays["initial_time_s"], arrays["initial_value_K"], arrays["initial_power_W"], definition)
    if not np.isfinite(target.numpy()).all(): raise ContractError("Normalized temperature targets overflow")
    train = arrays["split_id"] == SPLIT_IDS["train"]; validation = arrays["split_id"] == SPLIT_IDS["validation"]
    initial_train = arrays["initial_split_id"] == SPLIT_IDS["train"]; initial_val = arrays["initial_split_id"] == SPLIT_IDS["validation"]
    train_parts = {"data": (z[train], target[train]), "physics": (collocation, None), "initial": (initial[initial_train], None)}
    # Validation physics uses only validation-condition physical input coordinates
    # and the known equation, independent of all test labels/collocation.
    val_parts = {"data": (z[validation], target[validation]), "physics": (z[validation], None), "initial": (initial[initial_val], None)}
    return train_parts, val_parts, z, initial


def _condition_coverage(definition):
    conditions = definition["conditions"]
    train = [row for row in conditions if row["split"] == "train"]
    temperatures = sorted({row["initial_temperature_K"] for row in train})
    powers = sorted({row["power_W"] for row in train})
    pairs = sorted({(row["initial_temperature_K"], row["power_W"]) for row in train})
    # Diagnose a complete 2x2 subgrid without imposing a general training gate.
    # Compare along the smaller axis; sparse fixed-condition tasks are valid.
    by_temperature = {value: set() for value in temperatures}
    by_power = {value: set() for value in powers}
    for temperature, power in pairs:
        by_temperature[temperature].add(power); by_power[power].add(temperature)
    axis = by_temperature if len(by_temperature) <= len(by_power) else by_power
    groups = list(axis.items()); witness = None
    for index, (first, values) in enumerate(groups):
        if len(values) < 2: continue
        for second, other in groups[index + 1:]:
            if len(other) < 2: continue
            shared = sorted(values & other)
            if len(shared) >= 2:
                witness = {"initial_temperature_K": [first, second], "power_W": shared[:2]} if axis is by_temperature else {"initial_temperature_K": shared[:2], "power_W": [first, second]}
                break
        if witness is not None: break
    return {"training_conditions": len(train), "unique_training_initial_temperature_K": temperatures,
        "unique_training_power_W": powers, "training_physical_pairs": [list(pair) for pair in pairs],
        "has_complete_2_by_2_training_subgrid": witness is not None, "complete_2_by_2_witness": witness,
        "evaluated_validation_conditions": [row["id"] for row in conditions if row["split"] == "validation"],
        "evaluated_test_conditions": [row["id"] for row in conditions if row["split"] == "test"],
        "generalization_evidence": "Only the recorded validation/test conditions are evaluated; the declared domain is not evidence of coverage or arbitrary forcing generalization",
        "coverage_is_training_gate": False}


def _validation(network, components, definition, batch, weights, context):
    values = {name: _mean_component(network, name, points, target, definition, batch, context) for name, (points, target) in components.items()}
    values["total"] = math.fsum(weights[name] * values[name] for name in weights)
    if not math.isfinite(values["total"]): raise FloatingPointError("PINN weighted validation loss is non-finite")
    return values


def _evaluate(network, arrays, definition, points, initial, part, batch, context):
    rows = np.flatnonzero(arrays["split_id"] == SPLIT_IDS[part]); predictions, residuals = [], []
    n = definition["normalization"]
    for offset in range(0, len(rows), batch):
        context.check(); selected = rows[offset:offset + batch]
        residual, theta = _physics_prediction(network, points[selected], definition, create_graph=False)
        predictions.append(theta.detach().numpy() * n["state_scale_K"] + n["state_offset_K"]); residuals.append(residual.detach().numpy())
    predicted = np.concatenate(predictions); residual = np.concatenate(residuals); truth = arrays["temperature_K"][rows]
    conditions = []
    initial_indices = np.flatnonzero(arrays["initial_split_id"] == SPLIT_IDS[part]); initial_predictions, initial_sources = [], []
    for index in initial_indices:
        context.check(); condition = arrays["initial_condition_index"][index]
        existing = np.flatnonzero((arrays["condition_index"][rows] == condition) & (arrays["time_s"][rows, 0] == n["time_origin_s"]))
        if len(existing):
            initial_predictions.append(predicted[existing[0], 0]); initial_sources.append("reused_same_final_data_prediction_at_initial_time")
        else:
            with torch.no_grad(): value = float(_prediction(network, initial[index:index + 1])[0, 0])
            initial_predictions.append(n["state_offset_K"] + n["state_scale_K"] * value); initial_sources.append("explicit_initial_point_same_frozen_evaluation_phase")
    initial_predictions = np.asarray(initial_predictions, dtype=np.float64).reshape(-1, 1)
    initial_errors = (initial_predictions - arrays["initial_value_K"][initial_indices]) / n["state_scale_K"]
    for condition in np.unique(arrays["condition_index"][rows]):
        mask = arrays["condition_index"][rows] == condition; initial_mask = arrays["initial_condition_index"][initial_indices] == condition
        rmse = float(np.sqrt(np.mean((predicted[mask] - truth[mask]) ** 2)))
        conditions.append({**definition["conditions"][int(condition)], "condition_index": int(condition), "samples": int(mask.sum()),
            "rmse_K": rmse, "nrmse": rmse / n["data_nrmse_scale_K"], "normalized_physics_rms": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "normalized_initial_abs_error": float(np.max(np.abs(initial_errors[initial_mask])))})
    rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    summary = {"samples": len(rows), "rmse_K": rmse, "nrmse": rmse / n["data_nrmse_scale_K"],
        "normalized_physics_rms": float(np.sqrt(np.mean(residual ** 2))), "normalized_initial_max_abs_error": float(np.max(np.abs(initial_errors))),
        "per_condition": conditions, "initial_prediction_sources": initial_sources}
    if not all(math.isfinite(summary[key]) for key in ("rmse_K", "nrmse", "normalized_physics_rms", "normalized_initial_max_abs_error")):
        raise FloatingPointError("PINN frozen evaluation metrics are non-finite")
    saved = {"source_row": rows, "condition_index": arrays["condition_index"][rows], "time_s": arrays["time_s"][rows],
        "initial_temperature_K": arrays["initial_temperature_K"][rows], "power_W": arrays["power_W"][rows],
        "predicted_temperature_K": predicted, "reference_temperature_K": truth, "normalized_physics_residual": residual,
        "initial_condition_index": arrays["initial_condition_index"][initial_indices], "predicted_initial_temperature_K": initial_predictions,
        "normalized_initial_residual": initial_errors}
    return summary, saved


def train_pinn(workspace, problem_id, config, context=None):
    from .pinn_problem import read_pinn_problem
    from .pinn_export import export_pinn
    from .pinn_runtime import conditional_network

    context = context or OperationContext()
    with workspace.lock("conditional_ode_pinn"):
        problem = read_pinn_problem(workspace, problem_id); definition = validate_pinn_definition(problem["definition"])
        arrays, metadata = problem["arrays"], problem["metadata"]
        resolved = resolve_pinn_config(config); settings, weights = resolved["network"], resolved["loss_weights"]
        train, validation, points, initial = _components(arrays, definition)
        widths = [3, *resolved["network_config"]["hidden_dims"], 1]
        forward_factor = 3 if resolved["conditional_parameterization"] == "affine_initial_power" else 1
        for name, (group, _) in {**{f"train_{key}": value for key, value in train.items()}, **{f"validation_{key}": value for key, value in validation.items()}, "test": (points[arrays["split_id"] == SPLIT_IDS["test"]], None)}.items():
            estimate = min(settings["batch_size"], len(group)) * sum(widths) * _DERIVATIVE_FACTOR * forward_factor
            if estimate > _MAX_ACTIVATIONS: raise ContractError(f"PINN {name} batch exceeds 5,000,000 logical derivative activation values; reduce batch_size")
        seed_training(settings["seed"]); network = build_supervised_network(resolved["network_config"])
        prediction_network = conditional_network(network, definition, resolved["conditional_parameterization"])
        seeded_state = copy.deepcopy(network.state_dict())
        optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        interface = MaterialInterface("bundle", "Conditional thermal ODE PINN")
        effective = {**settings, "loss_weights": weights, "ablation": resolved["ablation"], "physics_derivative_create_graph": True,
            "normalization": definition["normalization"], "update_semantics": resolved["optimizer_update_semantics"], "data_loss_normalization": "state_scale_K",
            "conditional_parameterization": resolved["conditional_parameterization"]}
        contract = training_contract(workspace, "conditional_ode_pinn", [problem_id], effective, interface.fingerprint)
        signature = canonical_hash(contract); start = wait = updates = 0; log = []; best_state = None; best_loss = float("inf")
        if config.get("resume_checkpoint"):
            checkpoint = load_checkpoint(workspace, config["resume_checkpoint"], contract, settings["epochs"], expected_sha256=config.get("resume_checkpoint_sha256"))
            restore_training(checkpoint, network, optimizer); start, wait, updates, log = (checkpoint[key] for key in ("epoch", "wait", "updates", "log"))
            best_state, best_loss = checkpoint["best_state"], checkpoint["best_metric"]
        with workspace.stage("pinn_model", interface, [problem_id], config, compatible_parents=False) as stage:
            def checkpoint_epoch(epoch):
                return save_checkpoint(stage.root, contract, target_epochs=settings["epochs"], model=network, optimizer=optimizer, epoch=epoch, updates=updates,
                    best_state=best_state, best_metric=best_loss, wait=wait, log=log)
            for name, value in (("problem", definition), ("network_config", resolved["network_config"]), ("resolved_config", resolved), ("normalization", definition["normalization"])):
                atomic_json(stage.root / (name + ".json"), value)
            torch.save(seeded_state, stage.root / "seeded_initial_state.pt")
            saved = checkpoint_epoch(start)
            context.progress("pinn_initial_checkpoint", epoch=start, optimizer_updates=updates, **saved)
            current = copy.deepcopy(network.state_dict())
            try:
                network.load_state_dict(seeded_state, strict=True)
                initial_probe = gradient_probe(prediction_network, train, definition, settings["batch_size"], weights, context)
            finally: network.load_state_dict(current, strict=True); network.zero_grad(set_to_none=True)
            initial_probe.update(model_state="seeded_initial_network_before_optimizer_updates", reconstructed_during_resume=bool(config.get("resume_checkpoint")))
            atomic_json(stage.root / "initial_gradient_probe.json", initial_probe)
            for epoch in range(start, settings["epochs"]):
                if wait >= settings["patience"]: break
                context.check(); prediction_network.train(); optimizer.zero_grad(set_to_none=True); losses = {}
                for name, (group, target) in train.items():
                    losses[name] = (_mean_component(prediction_network, name, group, target, definition, settings["batch_size"], context, backward=True, weight=weights[name]) if weights[name] > 0 else None)
                total = math.fsum(weights[name] * losses[name] for name in weights if weights[name] > 0)
                before = float(torch.nn.utils.clip_grad_norm_(network.parameters(), settings["grad_clip"], error_if_nonfinite=True))
                after = float(torch.linalg.vector_norm(_gradients(network)))
                optimizer.step(); updates += 1; prediction_network.eval()
                val = _validation(prediction_network, validation, definition, settings["batch_size"], weights, context)
                selected = val["total"] < best_loss - settings["min_delta"]
                if selected: best_state, best_loss, wait = copy.deepcopy(network.state_dict()), val["total"], 0
                else: wait += 1
                event = {"epoch": epoch + 1, "training_loss": total, "training_components": losses, "validation_loss": val["total"], "validation_components": val,
                    "selected_by_validation": selected, "patience_wait": wait, "optimizer_updates": updates, "gradient_norm_before_clip": before, "gradient_norm_after_clip": after}
                log.append(event)
                saved = checkpoint_epoch(epoch + 1)
                context.progress("pinn_training", **event, **saved)
            if best_state is None: raise ContractError("No complete PINN epoch is available for validation-only selection")
            context.check(); network.load_state_dict(best_state, strict=True); prediction_network.eval()
            final_probe = gradient_probe(prediction_network, train, definition, settings["batch_size"], weights, context)
            final_probe["model_state"] = "best_validation_selected_network"
            atomic_json(stage.root / "final_gradient_probe.json", final_probe)
            torch.save({"network_state": best_state, "training_signature": signature}, stage.root / "model.pt")
            evaluations = {}
            for part in SPLIT_IDS:
                evaluations[part], saved = _evaluate(prediction_network, arrays, definition, points, initial, part, settings["batch_size"], context)
                if part == "test": np.savez_compressed(stage.root / "test_predictions.npz", **saved)
            test, gates = evaluations["test"], resolved["acceptance"]; failures = []
            if gates["max_test_trajectory_nrmse"] is not None and any(row["nrmse"] > gates["max_test_trajectory_nrmse"] for row in test["per_condition"]): failures.append("max_test_trajectory_nrmse")
            if gates["max_normalized_physics_rms"] is not None and test["normalized_physics_rms"] > gates["max_normalized_physics_rms"]: failures.append("max_normalized_physics_rms")
            if gates["max_initial_normalized_error"] is not None and test["normalized_initial_max_abs_error"] > gates["max_initial_normalized_error"]: failures.append("max_initial_normalized_error")
            passed = not failures; requested = any(value is not None for value in gates.values()); is_pinn = weights["physics"] > 0 and not resolved["ablation"]
            provenance = {"problem": problem_id, "problem_metadata_hash": canonical_hash(metadata), "physics_loss_enabled": weights["physics"] > 0,
                "physics_off_ablation": resolved["ablation"], "create_graph_for_physics_training": weights["physics"] > 0, "data_labels_used_for_optimizer": weights["data"] > 0}
            exported = export_pinn(stage.root / "bundle", network, definition, resolved["network_config"], training_config=resolved, provenance=provenance)
            atomic_json(stage.root / "pinn.json", {"schema_version": 1, "representation": "conditional_thermal_pinn" if is_pinn else "conditional_thermal_physics_off_ablation",
                "bundle": "bundle", "problem": problem_id, "capability": definition["capability"], "is_pinn": is_pinn, "conditional_parameterization": resolved["conditional_parameterization"]})
            algorithm = "conditional_ode_pinn" if is_pinn else "conditional_ode_physics_off_ablation"
            report = {"algorithm": algorithm, "checkpoint_algorithm": "conditional_ode_pinn", "problem": problem_id, "problem_metadata_hash": canonical_hash(metadata), "is_pinn": is_pinn,
                "training_regime": "physics_off_ablation" if resolved["ablation"] else "physics_only_objective_with_provided_reference" if weights["data"] == 0 else "physics_informed_supervised",
                "physical_equation": definition["equation"], "normalization": definition["normalization"], "capability": definition["capability"], "condition_coverage": _condition_coverage(definition),
                "conditional_parameterization": resolved["conditional_parameterization"],
                "resolved_config": resolved, "network_config": resolved["network_config"], "parameter_count": resolved["parameter_count"],
                "trained_epochs": len(log), "target_total_epochs": settings["epochs"], "best_epoch": next(row["epoch"] for row in reversed(log) if row["selected_by_validation"]),
                "best_validation_loss": best_loss, "optimizer_updates": updates, "resumed_from_epoch": start, "training_signature": signature,
                "stopping_reason": "patience" if wait >= settings["patience"] else "target_total_epochs",
                "loss_weights": weights, "loss_definitions": {"data": "mean normalized-temperature squared error using state_scale_K", "physics": "mean squared [C*(state/time)*dtheta/ds+G*state*theta-P]/physics_scale_W with create_graph=True", "initial": "mean squared theta(s=0,a,b)-a"},
                "gradient_probes": {"initial": initial_probe, "final": final_probe}, "collocation": {"fixed": True, "training_conditions_only": True, "points": len(train["physics"][0]), "temperature_targets_used": False},
                "data_labels_used_for_optimizer": weights["data"] > 0, "provided_reference_data": True, "selection": "validation_weighted_data_physics_initial_loss_only",
                "test_used_for_training_or_selection": False, "test_evaluations": 1,
                "test_evaluation_semantics": "single final frozen-weight phase; temperature and residual share each data forward; initial predictions reuse initial-time data rows when present",
                **evaluations, "acceptance": ("passed" if requested else "not_requested") if passed else "failed", "acceptance_failures": failures,
                "structural_finite_autodiff_acceptance": "passed", "accuracy_claim": "met_explicit_engineering_gates" if passed and requested else "not_requested" if not requested else "failed_requested_gates",
                "resource_boundary": {"max_parameters": 2_000_000, "max_logical_derivative_activation_values": _MAX_ACTIVATIONS, "derivative_factor": _DERIVATIVE_FACTOR,
                    "prediction_forwards_per_query": forward_factor,
                    "estimate": "min(batch_size,points)*sum(input,hidden,output widths)*8*prediction_forwards_per_query; conservative first/second derivative working set", "problem": definition["resource_boundary"]},
                "export": {"directory": "bundle", "details": exported}, "source_provenance": definition["source"], "confirmation_sources": context.case_confirmation_sources, "log": log}
            atomic_json(stage.root / "report.json", report); atomic_json(stage.root / "training_log.json", log)
            context.check()
            return stage.publish({"status": "verified" if passed else "rejected", "checks": ["explicit_SI_thermal_problem", "conditional_inputs", "autodiff_time_chain", "initial_conditions", "fixed_train_collocation", "full_mean_chunk_gradients", "validation_only_selection", "final_frozen_test"],
                "is_pinn": is_pinn, "acceptance": report["acceptance"], "failures": failures}, source={"algorithm": algorithm, "problem": problem_id,
                    "pinn_signals": definition["source"]["pinn_signals"], "capability": definition["capability"], "physics_loss_enabled": weights["physics"] > 0,
                    "physics_off_ablation": resolved["ablation"], "conditional_parameterization": resolved["conditional_parameterization"],
                    "confirmation_sources": context.case_confirmation_sources}, activate=passed and is_pinn)
