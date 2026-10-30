"""Residual supervision with declared physical constraints, not a generic PDE PINN solver."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .artifacts import Workspace, atomic_json, read_json
from .checkpoints import load_checkpoint, restore_training, save_checkpoint, seed_training, training_contract
from .contracts import ContractError, canonical_hash, validate_network_dimensions
from .data import read_transitions, training_coverage
from .modeling import CorrectedDynamics, ResidualNetwork, load_model, tensor
from .runtime import OperationContext


def mse_report(nominal, corrected, arrays, mask) -> dict:
    x, u, d, target = (tensor(arrays[k][mask]) for k in ("x", "u", "d", "next_x"))
    with torch.no_grad():
        base, fixed = (nominal(x, u, d) - target).square(), (corrected(x, u, d) - target).square()
    return {"nominal_mse": base.mean().item(), "corrected_mse": fixed.mean().item(),
            "nominal_mse_per_state": base.mean(0).tolist(), "corrected_mse_per_state": fixed.mean(0).tolist(),
            "samples": int(mask.sum())}


def train_residual(workspace: Workspace, nominal_id: str, dataset_id: str, config: dict,
                   context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    if config.get("residual_semantics", "next_state_residual_in_state_units") != "next_state_residual_in_state_units":
        raise ContractError("Residual training supports next_state_residual_in_state_units only; derivative residuals require a separate training adapter")
    nominal = load_model(workspace, nominal_id)
    if not hasattr(nominal, "spec"):
        raise ContractError("Train a residual against a nominal model, not an already corrected model")
    arrays, masks, data_meta = read_transitions(workspace, dataset_id)
    if nominal_id not in [p["id"] for p in data_meta["parents"]]:
        raise ContractError("Dataset was collected for a different nominal model version")
    interface = nominal.interface
    validate_network_dimensions(interface, config, residual=True)
    seed = int(config.get("seed", 42))
    epochs = int(config.get("epochs", 180))
    learning_rate = float(config.get("learning_rate", .003))
    if epochs < 1 or not 0 < learning_rate < 1:
        raise ContractError("epochs must be positive and learning_rate in (0,1)")
    seed_training(seed)
    generator = torch.Generator().manual_seed(seed)
    features = np.concatenate([arrays[k][masks["train"]] for k in ("x", "u", "d")], axis=1)
    x, u, d, target = (tensor(arrays[k]) for k in ("x", "u", "d", "next_x"))
    with torch.no_grad():
        residual_targets = target - nominal(x, u, d)
    scalers = {"input_mean": features.mean(0).tolist(), "input_scale": np.maximum(features.std(0), 1e-6).tolist(),
               "residual_scale": np.maximum(residual_targets[masks["train"]].numpy().std(0), 1e-5).tolist()}
    network_config = {"hidden_dims": config.get("hidden_dims", [32, 32]),
                      "input_dim": features.shape[1], "output_dim": interface.nx,
                      "residual_semantics": "next_state_residual_in_state_units"}
    residual = ResidualNetwork(interface, network_config, scalers)
    corrected = CorrectedDynamics(nominal, residual)
    weight_decay = float(config.get("weight_decay", 1e-6))
    physics_weight = float(config.get("physics_weight", .1))
    minimum_improvement = float(config.get("minimum_improvement", 0))
    if (not all(np.isfinite(v) and v >= 0 for v in (weight_decay, physics_weight))
            or not np.isfinite(minimum_improvement) or not 0 <= minimum_improvement < 1):
        raise ContractError("Invalid residual weight_decay, physics_weight or minimum_improvement")
    optimizer = torch.optim.Adam(residual.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lower, upper = tensor([v.lower for v in interface.states]), tensor([v.upper for v in interface.states])
    train_idx = torch.where(torch.as_tensor(masks["train"]))[0]
    validation_idx = torch.where(torch.as_tensor(masks["validation"]))[0]
    batch_size = int(config.get("batch_size", 128))
    patience = int(config.get("patience", 35))
    if batch_size < 1 or patience < 1:
        raise ContractError("batch_size and patience must be positive")
    balances = config.get("affine_balance_constraints", [])
    for balance in balances:
        if len(balance["coefficients"]) != interface.nx:
            raise ContractError("Balance constraints require one coefficient per state")
    best_loss, best_state, start, wait, updates, log = float("inf"), None, 0, 0, 0, []
    effective_config = {**config, **network_config, "learning_rate": learning_rate, "seed": seed,
                        "batch_size": batch_size, "affine_balance_constraints": balances,
                        "physics_weight": physics_weight, "weight_decay": weight_decay,
                        "patience": patience, "minimum_improvement": minimum_improvement, "grad_clip": 5}
    contract = training_contract(workspace, "residual_training", [nominal_id, dataset_id],
                                 effective_config, interface.fingerprint)
    signature = canonical_hash(contract)
    if config.get("resume_checkpoint"):
        checkpoint = load_checkpoint(workspace, config["resume_checkpoint"], contract, epochs,
                                     expected_sha256=config.get("resume_checkpoint_sha256"))
        restore_training(checkpoint, residual, optimizer, generator)
        start, wait, updates = checkpoint["epoch"], checkpoint["wait"], checkpoint["updates"]
        best_loss, best_state, log = checkpoint["best_metric"], checkpoint["best_state"], checkpoint["log"]
    with workspace.stage("corrected_model", interface, [nominal_id, dataset_id], config) as stage:
        def checkpoint_epoch(epoch):
            save_checkpoint(stage.root, contract, model=residual, optimizer=optimizer, epoch=epoch,
                            updates=updates, best_state=best_state, best_metric=best_loss, wait=wait,
                            log=log, generator=generator)

        atomic_json(stage.root / "nominal.json", nominal.spec)
        atomic_json(stage.root / "network.json", network_config)
        atomic_json(stage.root / "scalers.json", scalers)
        checkpoint_epoch(start)
        for epoch in range(start, epochs):
            if wait >= patience:
                break
            context.check()
            order = train_idx[torch.randperm(len(train_idx), generator=generator)]
            losses = []
            residual.train()
            for batch in order.split(batch_size):
                context.check()
                prediction = residual(x[batch], u[batch], d[batch])
                loss_data = ((prediction - residual_targets[batch]) / residual.residual_scale).square().mean()
                next_x = nominal(x[batch], u[batch], d[batch]) + prediction
                bound_loss = ((torch.relu(lower - next_x) + torch.relu(next_x - upper)) / (upper - lower)).square().mean()
                physical = bound_loss
                for balance in balances:
                    balance_error = ((next_x - x[batch]) @ tensor(balance["coefficients"]) / interface.sample_time
                                     - float(balance.get("rate", 0)))
                    physical = physical + balance_error.square().mean()
                loss = loss_data + physics_weight * physical
                if not torch.isfinite(loss):
                    raise FloatingPointError("Residual loss is non-finite")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(residual.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                losses.append(loss.item())
            residual.eval()
            with torch.no_grad():
                validation = ((corrected(x[validation_idx], u[validation_idx], d[validation_idx]) - target[validation_idx])
                              / residual.residual_scale).square().mean().item()
            if not np.isfinite(validation):
                raise FloatingPointError("Residual validation loss is non-finite")
            if validation < best_loss:
                best_loss, best_state, wait = validation, copy.deepcopy(residual.state_dict()), 0
            else:
                wait += 1
            event = {"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "validation_loss": validation}
            log.append(event)
            checkpoint_epoch(epoch + 1)
            context.progress("residual_training", **event, optimizer_updates=updates,
                             checkpoint=str(workspace.path / "failed" / stage.id / "latest.pt"))
            if wait >= patience:
                break
        if best_state is None:
            raise ContractError("No trained checkpoint available")
        residual.load_state_dict(best_state)
        residual.eval()
        torch.save({"network_state": best_state, "training_signature": signature}, stage.root / "model.pt")
        report = {part: mse_report(nominal, corrected, arrays, masks[part]) for part in masks}
        test = report["test"]
        accepted = test["corrected_mse"] <= test["nominal_mse"] * (1 - minimum_improvement)
        report.update({"algorithm": "supervised_discrete_residual_with_declared_physical_constraints",
                       "residual_semantics": "next_state_residual_in_state_units", "trained_epochs": len(log),
                       "optimizer_updates": updates, "resumed_from_epoch": start, "training_signature": signature,
                       "test_used_for_training_or_selection": False,
                       "applicability": training_coverage(interface, arrays, masks["train"]),
                       "physical_constraints": {"state_bounds": True, "affine_balances": balances},
                       "acceptance": "passed" if accepted else "failed", "log": log})
        atomic_json(stage.root / "report.json", report)
        return stage.publish({"status": "verified" if accepted else "rejected", "test": test,
                              "acceptance": report["acceptance"]}, preprocessing=scalers,
                             source={"algorithm": report["algorithm"], "data_source": data_meta["source"], "nominal_model": nominal_id})
