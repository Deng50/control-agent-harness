"""Expert imitation and physical-rollout policy optimization using reused networks."""
from __future__ import annotations

import copy
import time
import numpy as np
import torch

from .artifacts import Workspace, atomic_json, read_json
from .checkpoints import load_checkpoint, restore_training, save_checkpoint, seed_training, training_contract
from .contracts import ContractError, ModelInterface, canonical_hash
from .control import closed_loop, load_controller, validate_scenario
from .data import trajectory_split
from .modeling import DTYPE, load_model, tensor
from .plants import OriginalCallablePlant, load_evaluation_plant
from .reused.mlp import MLPNetwork
from .reused.lstm import LSTMNetwork
from .reused.rnn import RNNNetwork
from .runtime import OperationContext


def collect_expert(workspace: Workspace, controller_id: str, reference_id: str, config: dict,
                   context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    controller = load_controller(workspace, controller_id, context)
    plant = load_evaluation_plant(workspace, controller_id, reference_id)
    controller.interface.compatible(plant.interface)
    scenarios = config.get("scenarios", [])
    if len(scenarios) < 5:
        raise ContractError("Expert collection needs at least five independent scenarios")
    hashes = [canonical_hash({k: v for k, v in s.items() if k != "name"}) for s in scenarios]
    if len(set(hashes)) != len(hashes):
        raise ContractError("Duplicate expert scenarios cannot be treated as independent conditions")
    root, _ = workspace.resolve(controller_id)
    design_config = read_json(root / "config.json")
    reserved = {canonical_hash({k: v for k, v in s.items() if k != "name"})
                for group in ("verification_scenarios", "test_scenarios") for s in design_config[group]}
    if reserved.intersection(hashes):
        raise ContractError("Expert training scenarios overlap reserved controller verification/test cases")
    with workspace.stage("expert_dataset", plant.interface, [controller_id, reference_id], config) as stage:
        rows = {k: [] for k in ("x", "reference", "d", "u", "previous_u", "trajectory_id")}
        reports = []
        for index, scenario in enumerate(scenarios):
            run = closed_loop(plant, controller, scenario, context)
            reports.append({"scenario": scenario, "metrics": run["metrics"],
                            "constraint_checks": run["constraint_checks"]})
            previous = np.clip(np.zeros(plant.interface.nu), [v.lower for v in plant.interface.controls],
                               [v.upper for v in plant.interface.controls]).tolist()
            for row in run["trajectory"]:
                for key in ("x", "reference", "d", "u"):
                    rows[key].append(row[key])
                rows["trajectory_id"].append(index)
                rows["previous_u"].append(previous)
                previous = row["u"]
            context.progress("expert_collection", scenario=index + 1, total=len(scenarios))
        arrays = {k: np.asarray(v, dtype=int if k == "trajectory_id" else float) for k, v in rows.items()}
        np.savez_compressed(stage.root / "expert.npz", **arrays)
        atomic_json(stage.root / "split.json", trajectory_split(arrays["trajectory_id"], int(config.get("seed", 42))))
        passed = constraints_pass(reports)
        atomic_json(stage.root / "collection.json", {"scenarios": reports,
                    "constraint_checks": reports[0]["constraint_checks"],
                    "closed_loop_constraints_passed": passed,
                    "previous_action_semantics": "previous_trajectory_action_initial_amplitude_clipped_zero",
                    "forecast_semantics": "scenario_disturbances_are_declared_constant"})
        return stage.publish({"status": "verified" if passed else "rejected", "samples": len(arrays["u"]),
                              "scenarios": len(scenarios), "closed_loop_constraints_passed": passed,
                              "constraint_checks": reports[0]["constraint_checks"]},
                             source={"controller": controller_id, "evaluation_model": reference_id,
                                     "evaluation_backend": "original_python_callable" if isinstance(plant, OriginalCallablePlant) else "registered_dynamics",
                                     "kind": "simulated_expert_closed_loop"})


def network_config(interface, config: dict) -> dict:
    architecture = config.get("architecture", "mlp")
    base = interface.nx + interface.ny + interface.nd
    result = {"architecture": architecture, "output_dim": interface.nu,
              "input_dim": base, "dropout": float(config.get("dropout", 0.0))}
    if architecture == "mlp":
        result.update(hidden_dims=config.get("hidden_dims", [64, 32]), input_semantics="current_state_reference_disturbance")
    elif architecture in {"rnn", "lstm"}:
        if not interface.nd or config.get("disturbance_forecast") != "declared_constant":
            raise ContractError("Reused RNN/LSTM require disturbance dimensions and an explicit declared_constant forecast for this adapter")
        length = int(config.get("preview_len", 5))
        if length < 1:
            raise ContractError("preview_len must be positive")
        result.update(input_dim=base + length * interface.nd, static_features_dim=base,
                      preview_len=length, preview_features_dim=interface.nd,
                      hidden_size=int(config.get("hidden_size", 32)), num_layers=int(config.get("num_layers", 1)),
                      disturbance_forecast="declared_constant", input_semantics="current_features_plus_future_disturbance_preview")
    else:
        raise ContractError("architecture must be mlp, rnn or lstm")
    for key in ("input_dim", "output_dim"):
        if key in config and config[key] != result[key]:
            raise ContractError(f"{key} conflicts with metadata/preview; expected {result[key]}")
    return result


def build_network(config):
    arguments = {k: v for k, v in config.items() if k not in {
        "architecture", "input_semantics", "disturbance_forecast", "action_constraints"}}
    cls = {"mlp": MLPNetwork, "rnn": RNNNetwork, "lstm": LSTMNetwork}[config["architecture"]]
    return cls(**arguments).to(dtype=DTYPE)


def features(x, reference, d, config):
    static = torch.cat((x, reference, d), dim=-1)
    if config["architecture"] == "mlp":
        return static
    # The current adapter accepts constant-disturbance scenarios, explicitly declared in metadata.
    forecast = d.unsqueeze(-2).expand(*d.shape[:-1], config["preview_len"], d.shape[-1]).flatten(-2)
    return torch.cat((static, forecast), dim=-1)


class NeuralPolicy(torch.nn.Module):
    def __init__(self, interface, config, scalers, weights=None):
        super().__init__()
        self.interface, self.config = interface, config
        self.network = build_network(config)
        for key in ("input_mean", "input_scale", "output_mean", "output_scale"):
            self.register_buffer(key, tensor(scalers[key]))
        self.register_buffer("action_low", tensor([v.lower for v in interface.controls]))
        self.register_buffer("action_high", tensor([v.upper for v in interface.controls]))
        self.rate = copy.deepcopy(config.get("action_constraints", {}).get("input_rate_bounds"))
        if self.rate is not None:
            bounds = np.asarray(self.rate, dtype=float)
            if (bounds.shape != (interface.nu, 2) or not np.isfinite(bounds).all()
                    or np.any(bounds[:, 0] >= bounds[:, 1])):
                raise ContractError("Inherited input_rate_bounds must be finite nu-by-2 action increments per sample")
            self.register_buffer("rate_low", tensor(bounds[:, 0]), persistent=False)
            self.register_buffer("rate_high", tensor(bounds[:, 1]), persistent=False)
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x, reference, d, previous=None):
        z = (features(x, reference, d, self.config) - self.input_mean) / self.input_scale
        predicted = self.network(z) * self.output_scale + self.output_mean
        low, high = self.action_low, self.action_high
        if self.rate is not None:
            if previous is None:
                raise ContractError("Rate-constrained neural forward requires explicit previous action")
            previous = torch.as_tensor(previous, dtype=predicted.dtype, device=predicted.device)
            if previous.shape != predicted.shape or not torch.isfinite(previous).all():
                raise ContractError("Previous action must have the same finite shape as the neural control batch")
            low = torch.maximum(low, previous + self.rate_low)
            high = torch.minimum(high, previous + self.rate_high)
            if torch.any(low > high):
                raise ContractError("Input amplitude and inherited rate bounds have an empty intersection")
        return torch.clamp(predicted, low, high)

    def act(self, x, reference, disturbance=None, previous=None):
        if disturbance is None:
            if self.interface.nd:
                raise ContractError("Disturbance observation is required")
            disturbance = np.empty(0)
        self.eval()
        started = time.perf_counter()
        if previous is None:
            previous = torch.clamp(torch.zeros_like(self.action_low), self.action_low, self.action_high)
        with torch.no_grad():
            u = self(tensor(x).unsqueeze(0), tensor(reference).unsqueeze(0), tensor(disturbance).unsqueeze(0),
                     previous=tensor(previous).unsqueeze(0))[0].numpy()
        if not np.isfinite(u).all():
            raise FloatingPointError("Neural policy returned non-finite control")
        return u, {"inference_seconds": time.perf_counter() - started}


def inherited_action_constraints(workspace, controller_id, interface):
    root, meta = workspace.resolve(controller_id, kind="controller")
    interface.compatible(ModelInterface.parse(meta["interface"]))
    parameters = read_json(root / "controller.json")["parameters"]
    rate = parameters.get("input_rate_bounds")
    if rate is not None:
        bounds = np.asarray(rate, dtype=float)
        if (bounds.shape != (interface.nu, 2) or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] >= bounds[:, 1])):
            raise ContractError("Expert controller has invalid input_rate_bounds")
        rate = bounds.tolist()
    return {"input_rate_bounds": rate, "source_controller": controller_id,
            "rate_semantics": "action_increment_per_sample", "projection": "amplitude_rate_intersection_v1",
            "initial_previous": "amplitude_clipped_zero"}


def reject_constraint_override(config):
    if "input_rate_bounds" in config or "action_constraints" in config:
        raise ContractError("Policy action constraints are inherited from the expert controller; configure them there")


def policy_constraint_checks(policy):
    return {"state": True, "input": True, "output": True, "input_rate": policy.rate is not None}


def expert_previous_actions(data, interface):
    """Recover legacy datasets using the same initial action as closed_loop."""
    controls, trajectory_ids = np.asarray(data["u"]), np.asarray(data["trajectory_id"])
    if (controls.ndim != 2 or controls.shape[1] != interface.nu or not np.isfinite(controls).all()
            or trajectory_ids.shape != (len(controls),)):
        raise ContractError("Expert action/trajectory shapes are invalid")
    previous = np.empty_like(controls)
    initial = np.clip(np.zeros(interface.nu), [v.lower for v in interface.controls],
                      [v.upper for v in interface.controls])
    for group in np.unique(trajectory_ids):
        indexes = np.flatnonzero(trajectory_ids == group)
        previous[indexes[0]] = initial
        previous[indexes[1:]] = controls[indexes[:-1]]
    if "previous_u" in data:
        saved = np.asarray(data["previous_u"])
        if saved.shape != previous.shape or not np.allclose(saved, previous, rtol=0, atol=1e-12):
            raise ContractError("Expert previous_u does not match its trajectory action history")
    return tensor(previous)


def read_expert(workspace, dataset_id):
    root, meta = workspace.resolve(dataset_id, kind="expert_dataset")
    # 控制器和评价对象会在模仿训练的独立工况中再次执行，只允许使用冻结数据直接绑定的版本。
    if not {meta["source"].get("evaluation_model"), meta["source"].get("controller")}.issubset({p["id"] for p in meta["parents"]}):
        raise ContractError("专家数据所声明的控制器和评价对象必须同时绑定为该数据的父版本。")
    with np.load(root / "expert.npz", allow_pickle=False) as loaded:
        data = {k: loaded[k] for k in loaded.files}
    split = read_json(root / "split.json")
    masks = {k: np.isin(data["trajectory_id"], v) for k, v in split.items()}
    return data, masks, read_json(root / "collection.json"), meta


def policy_evaluation(plant, policy, collection, masks, data, context):
    result = {}
    for group in ("validation", "test"):
        scenario_ids = np.unique(data["trajectory_id"][masks[group]])
        result[group] = [closed_loop(plant, policy, collection["scenarios"][int(index)]["scenario"], context)
                         for index in scenario_ids]
    return result


def train_imitation(workspace: Workspace, dataset_id: str, config: dict,
                    context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    reject_constraint_override(config)
    data, masks, collection, meta = read_expert(workspace, dataset_id)
    interface = ModelInterface.parse(meta["interface"])
    reference_id = meta["source"]["evaluation_model"]
    plant = load_evaluation_plant(workspace, meta["source"]["controller"], reference_id)
    seed = int(config.get("seed", 42))
    seed_training(seed)
    generator = torch.Generator().manual_seed(seed)
    net_config = network_config(interface, config)
    net_config["action_constraints"] = inherited_action_constraints(workspace, meta["source"]["controller"], interface)
    x, ref, d, u = (tensor(data[k]) for k in ("x", "reference", "d", "u"))
    previous = expert_previous_actions(data, interface)
    observations = features(x, ref, d, net_config)
    train = masks["train"]
    scalers = {"input_mean": observations[train].mean(0).tolist(),
               "input_scale": observations[train].std(0, unbiased=False).clamp_min(1e-5).tolist(),
               "output_mean": u[train].mean(0).tolist(),
               "output_scale": u[train].std(0, unbiased=False).clamp_min(1e-5).tolist()}
    policy = NeuralPolicy(interface, net_config, scalers)
    epochs, patience = int(config.get("epochs", 150)), int(config.get("patience", 30))
    lr = float(config.get("learning_rate", .003))
    batch_size = int(config.get("batch_size", 64))
    if epochs < 1 or patience < 1 or batch_size < 1 or not 0 < lr < 1:
        raise ContractError("Invalid imitation epochs, patience, batch_size or learning_rate")
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    train_idx = torch.where(torch.as_tensor(train))[0]
    effective_config = {**config, **net_config, "seed": seed, "patience": patience,
                        "learning_rate": lr, "batch_size": batch_size, "grad_clip": 5,
                        "supervised_action_semantics": "projected_action_given_expert_previous_v1"}
    contract = training_contract(workspace, "expert_imitation", [dataset_id, reference_id],
                                 effective_config, interface.fingerprint)
    best, best_state, wait, log, start, updates = float("inf"), None, 0, [], 0, 0
    if config.get("resume_checkpoint"):
        checkpoint = load_checkpoint(workspace, config["resume_checkpoint"], contract, epochs,
                                     expected_sha256=config.get("resume_checkpoint_sha256"))
        restore_training(checkpoint, policy, optimizer, generator)
        start, updates, wait = checkpoint["epoch"], checkpoint["updates"], checkpoint["wait"]
        best, best_state, log = checkpoint["best_metric"], checkpoint["best_state"], checkpoint["log"]
    with workspace.stage("imitation_policy", interface, [dataset_id], config) as stage:
        def checkpoint_epoch(epoch):
            return save_checkpoint(stage.root, contract, target_epochs=epochs, model=policy, optimizer=optimizer, epoch=epoch,
                            updates=updates, best_state=best_state, best_metric=best, wait=wait,
                            log=log, generator=generator)

        saved = checkpoint_epoch(start)
        context.progress("imitation_initial_checkpoint", epoch=start, optimizer_updates=updates, **saved)
        for epoch in range(start, epochs):
            if wait >= patience:
                break
            context.check()
            policy.train()
            order = train_idx[torch.randperm(len(train_idx), generator=generator)]
            for batch in order.split(batch_size):
                context.check()
                predicted = policy(x[batch], ref[batch], d[batch], previous=previous[batch])
                loss = ((predicted - u[batch]) / policy.output_scale).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Imitation loss is non-finite")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            policy.eval()
            with torch.no_grad():
                val = ((policy(x[masks["validation"]], ref[masks["validation"]], d[masks["validation"]],
                               previous=previous[masks["validation"]])
                        - u[masks["validation"]]) / policy.output_scale).square().mean().item()
            if not np.isfinite(val):
                raise FloatingPointError("Imitation validation loss is non-finite")
            if val < best:
                best, best_state, wait = val, copy.deepcopy(policy.state_dict()), 0
            else:
                wait += 1
            event = {"epoch": epoch + 1, "training_loss": loss.item(), "validation_loss": val}
            log.append(event)
            saved = checkpoint_epoch(epoch + 1)
            context.progress("imitation_training", **event, optimizer_updates=updates,
                             **saved)
            if wait >= patience:
                break
        if best_state is None:
            raise ContractError("No trained policy checkpoint")
        policy.load_state_dict(best_state)
        policy.eval()
        evaluation = policy_evaluation(plant, policy, collection, masks, data, context)
        with torch.no_grad():
            error = policy(x[masks["test"]], ref[masks["test"]], d[masks["test"]],
                           previous=previous[masks["test"]]) - u[masks["test"]]
        report = {"algorithm": "expert_imitation", "input_semantics": net_config["input_semantics"],
                  "trained_epochs": len(log), "optimizer_updates": updates, "resumed_from_epoch": start,
                  "training_signature": canonical_hash(contract),
                  "action_constraints": net_config["action_constraints"],
                  "constraint_checks": policy_constraint_checks(policy),
                  "offline_action_metrics": "projected_action_given_expert_previous",
                  "evaluation_model": reference_id, "evaluation_backend": meta["source"].get("evaluation_backend", "registered_dynamics"),
                  "test_action_rmse": error.square().mean(0).sqrt().tolist(), "closed_loop": evaluation, "log": log,
                  "parameter_count": sum(p.numel() for p in policy.parameters()), "test_used_for_training": False}
        atomic_json(stage.root / "network.json", net_config)
        atomic_json(stage.root / "scalers.json", scalers)
        atomic_json(stage.root / "report.json", report)
        atomic_json(stage.root / "policy.json", {"dataset": dataset_id, "evaluation_model": reference_id,
                                               "expert_controller": meta["source"]["controller"],
                                               "action_constraints": net_config["action_constraints"]})
        torch.save({"policy_state": best_state}, stage.root / "policy.pt")
        from .legacy_policy_export import export_legacy_policy
        try:
            report["standalone_export"] = export_legacy_policy(stage.root / "bundle", interface, net_config, scalers, best_state,
                provenance={"dataset":dataset_id,"expert_controller":meta["source"]["controller"],"evaluation_model":reference_id,
                            "scope":"Existing internal imitation route; original supplied-state and declared-preview semantics retained"})
        except ContractError as error:
            report["standalone_export"] = {"status":"not_exported","reason":str(error)}
        atomic_json(stage.root / "report.json", report)
        return stage.publish({"status": "verified" if constraints_pass(evaluation["test"]) else "rejected", "test_action_rmse": report["test_action_rmse"],
                              "closed_loop_constraints_passed": constraints_pass(evaluation["test"]),
                              "constraint_checks": policy_constraint_checks(policy)},
                             preprocessing=scalers, source={"algorithm": "expert_imitation"})


def constraints_pass(runs):
    if not runs:
        return False
    for run in runs:
        checks = run.get("constraint_checks", {})
        # An absent rate declaration is not evidence that a rate constraint passed.
        if not all(checks.get(key) is True for key in ("state", "input", "output")):
            return False
        keys = ["max_state_violation", "max_input_violation", "max_output_violation"]
        if checks.get("input_rate") is True:
            keys.append("max_rate_violation")
        if any(not np.isfinite(run["metrics"].get(key, np.nan)) or run["metrics"][key] > 1e-6 for key in keys):
            return False
    return True


def load_policy(workspace: Workspace, artifact_id: str):
    root, meta = workspace.resolve(artifact_id)
    if meta["kind"] not in {"imitation_policy", "optimized_policy"}:
        raise ContractError("Artifact is not a neural control policy")
    interface, config = ModelInterface.parse(meta["interface"]), read_json(root / "network.json")
    descriptor = read_json(root / "policy.json")
    inherited = inherited_action_constraints(workspace, descriptor["expert_controller"], interface)
    saved = config.get("action_constraints")
    if saved != inherited and (saved is not None or inherited["input_rate_bounds"] is not None):
        raise ContractError("Neural policy action constraints differ from its expert controller; retrain the policy")
    if descriptor.get("action_constraints", saved) != saved:
        raise ContractError("Neural policy descriptor and network action constraints differ")
    checkpoint = torch.load(root / "policy.pt", weights_only=True, map_location="cpu")
    policy = NeuralPolicy(interface, config, read_json(root / "scalers.json"), checkpoint["policy_state"])
    policy.eval()
    return policy


def optimize_policy(workspace: Workspace, policy_id: str, model_id: str, config: dict,
                    context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    reject_constraint_override(config)
    policy = load_policy(workspace, policy_id)
    lineage = workspace.resolve_closure(policy_id)
    root, _ = lineage[policy_id]
    descriptor = read_json(root / "policy.json")
    dataset_id = descriptor["dataset"]
    if dataset_id not in lineage or lineage[dataset_id][1]["kind"] != "expert_dataset":
        raise ContractError("模仿或优化策略的专家数据集必须在其已验证的依赖链内。")
    data, masks, collection, meta = read_expert(workspace, dataset_id)
    model = load_model(workspace, model_id)
    if (descriptor["expert_controller"] != meta["source"]["controller"] or
            descriptor["evaluation_model"] != meta["source"]["evaluation_model"]):
        raise ContractError("策略记录中的专家和评价对象必须与所选专家数据的来源一致。")
    plant = load_evaluation_plant(workspace, descriptor["expert_controller"], descriptor["evaluation_model"])
    policy.interface.compatible(model.interface)
    expert = load_controller(workspace, descriptor["expert_controller"], context)
    controller_root, _ = workspace.resolve(descriptor["expert_controller"])
    if read_json(controller_root / "controller.json")["prediction_model"] != model_id:
        raise ContractError("Policy optimization model must match the expert prediction model version")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    seed = int(config.get("seed", 42))
    seed_training(seed)
    rng = np.random.default_rng(seed)
    learning_rate = float(config.get("learning_rate", .0003))
    batch_size = int(config.get("batch_size", 24))
    imitation_weight = float(config.get("imitation_weight", .1))
    constraint_weight = float(config.get("constraint_weight", 100))
    if (not 0 < learning_rate < 1 or batch_size < 1
            or not all(np.isfinite(v) and v >= 0 for v in (imitation_weight, constraint_weight))):
        raise ContractError("Invalid physical optimization learning_rate, batch_size or loss weights")
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    x, ref, d, expert_u = (tensor(data[k]) for k in ("x", "reference", "d", "u"))
    i = model.interface
    previous = expert_previous_actions(data, i)
    xlo, xhi = tensor([v.lower for v in i.states]), tensor([v.upper for v in i.states])
    ylo, yhi = tensor([v.lower for v in i.outputs]), tensor([v.upper for v in i.outputs])
    yscale = tensor([v.upper - v.lower for v in i.outputs])
    uscale = tensor([v.upper - v.lower for v in i.controls])
    groups = np.unique(data["trajectory_id"][masks["train"]])
    parameters = expert.config
    horizon, epochs = int(config.get("horizon", parameters["horizon"])), int(config.get("epochs", 40))
    if horizon < 1 or epochs < 1:
        raise ContractError("Physical optimization horizon/epochs must be positive")
    best_state = copy.deepcopy(policy.state_dict())

    def selection_score():
        policy.eval()
        indexes = np.unique(data["trajectory_id"][masks["validation"]])
        runs = [closed_loop(plant, policy, collection["scenarios"][int(k)]["scenario"], context) for k in indexes]
        if not constraints_pass(runs):
            return float("inf")
        return float(np.mean([r["metrics"]["normalized_rmse"] + .01 * r["metrics"]["control_effort"] for r in runs]))

    effective_config = {**config, "seed": seed, "learning_rate": learning_rate, "batch_size": batch_size,
                        "horizon": horizon, "imitation_weight": imitation_weight,
                        "constraint_weight": constraint_weight, "grad_clip": 5,
                        "constraint_penalty_semantics": "normalized_state_and_output_bounds_v1",
                        "action_constraints": inherited_action_constraints(workspace, descriptor["expert_controller"], i),
                        "rollout_action_semantics": "projected_action_with_explicit_previous_v1"}
    contract = training_contract(workspace, "physical_policy_optimization",
                                 [policy_id, model_id, dataset_id, descriptor["evaluation_model"],
                                  descriptor["expert_controller"]], effective_config, i.fingerprint)
    best_score, log, start, updates = None, [], 0, 0
    if config.get("resume_checkpoint"):
        checkpoint = load_checkpoint(workspace, config["resume_checkpoint"], contract, epochs,
                                     expected_sha256=config.get("resume_checkpoint_sha256"))
        restore_training(checkpoint, policy, optimizer, numpy_generator=rng)
        start, updates = checkpoint["epoch"], checkpoint["updates"]
        best_score, best_state, log = checkpoint["best_metric"], checkpoint["best_state"], checkpoint["log"]
    with workspace.stage("optimized_policy", i, [policy_id, model_id], config) as stage:
        def checkpoint_epoch(epoch):
            return save_checkpoint(stage.root, contract, target_epochs=epochs, model=policy, optimizer=optimizer, epoch=epoch,
                            updates=updates, best_state=best_state, best_metric=best_score, wait=0,
                            log=log, numpy_generator=rng)

        saved = checkpoint_epoch(start)
        context.progress("physical_initial_checkpoint", epoch=start, optimizer_updates=updates, **saved)
        if best_score is None:
            best_score = selection_score()
            saved = checkpoint_epoch(start)
            context.progress("physical_initial_checkpoint", epoch=start, optimizer_updates=updates, **saved)
        for epoch in range(start, epochs):
            context.check()
            # Every training condition participates each epoch, fixing the legacy tail-starvation issue.
            for group in rng.permutation(groups):
                context.check()
                indexes = np.flatnonzero((data["trajectory_id"] == group) & masks["train"])
                indexes = rng.choice(indexes, size=min(len(indexes), batch_size), replace=False)
                state, reference, disturbance = x[indexes], ref[indexes], d[indexes]
                original_state = state
                initial_previous = previous[indexes]
                last_action = initial_previous
                cost = state.new_zeros(())
                policy.eval()  # Deterministic rollout: dropout is disabled during physical optimization.
                for _ in range(horizon):
                    context.check()
                    action = policy(state, reference, disturbance, previous=last_action)
                    state = model(state, action, disturbance)
                    output = model.output(state)
                    cost = cost + float(parameters.get("q", 1)) * ((output - reference) / yscale).square().mean()
                    cost = cost + float(parameters.get("r", .01)) * (action / uscale).square().mean()
                    cost = cost + float(parameters.get("s", .01)) * ((action - last_action) / uscale).square().mean()
                    cost = cost + constraint_weight * ((torch.relu(xlo - state) + torch.relu(state - xhi)) / (xhi - xlo)).square().mean()
                    cost = cost + constraint_weight * ((torch.relu(ylo - output) + torch.relu(output - yhi)) / yscale).square().mean()
                    last_action = action
                protection = ((policy(original_state, reference, disturbance, previous=initial_previous)
                               - expert_u[indexes]) / uscale).square().mean()
                loss = cost / horizon + imitation_weight * protection
                before, before_optimizer = copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
                if not torch.isfinite(loss):
                    raise FloatingPointError("Physical rollout cost is non-finite")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in policy.parameters()):
                    policy.load_state_dict(before)
                    optimizer.load_state_dict(before_optimizer)
                    raise FloatingPointError("Non-finite policy update rolled back")
                updates += 1
            score = selection_score()
            if score < best_score:
                best_score, best_state = score, copy.deepcopy(policy.state_dict())
            event = {"epoch": epoch + 1, "loss": loss.item(), "validation_score": score if np.isfinite(score) else None,
                     "training_conditions_updated": len(groups)}
            log.append(event)
            saved = checkpoint_epoch(epoch + 1)
            context.progress("physical_policy_optimization", **event, optimizer_updates=updates,
                             **saved)
        policy.load_state_dict(best_state)
        policy.eval()
        original = load_policy(workspace, policy_id)
        test_ids = np.unique(data["trajectory_id"][masks["test"]])
        comparison = []
        for index in test_ids:
            scenario = collection["scenarios"][int(index)]["scenario"]
            comparison.append({"scenario": scenario,
                "expert": closed_loop(plant, expert, scenario, context),
                "imitation": closed_loop(plant, original, scenario, context),
                "optimized": closed_loop(plant, policy, scenario, context)})
        report = {"algorithm": "differentiable_physical_rollout_policy_gradient",
                  "trained_epochs": len(log), "optimizer_updates": updates, "resumed_from_epoch": start,
                  "training_signature": canonical_hash(contract),
                  "action_constraints": effective_config["action_constraints"],
                  "constraint_checks": policy_constraint_checks(policy),
                  "rollout_action_semantics": effective_config["rollout_action_semantics"],
                  "learned_critic": False, "corrected_dynamics_used": model_id,
                  "checkpoint_semantics": "one_shared_checkpoint_selected_on_validation_scenarios",
                  "test_used_for_updates_or_checkpoint_selection": False,
                  "comparison": comparison, "log": log}
        for filename in ("network.json", "scalers.json", "policy.json"):
            atomic_json(stage.root / filename, read_json(root / filename))
        atomic_json(stage.root / "report.json", report)
        torch.save({"policy_state": best_state}, stage.root / "policy.pt")
        return stage.publish({"status": "verified" if constraints_pass([r["optimized"] for r in comparison]) else "rejected", "closed_loop_constraints_passed": constraints_pass([r["optimized"] for r in comparison]),
                              "comparison_available": True, "constraint_checks": policy_constraint_checks(policy)},
                             preprocessing=read_json(root / "scalers.json"),
                             source={"algorithm": report["algorithm"], "prediction_model": model_id})
