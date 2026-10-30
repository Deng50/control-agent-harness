"""Receding-horizon constrained control and source-derived genetic tuning."""
from __future__ import annotations

import copy
import math
import random
import time
import numpy as np
import torch
from scipy.optimize import minimize

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, canonical_hash
from .modeling import describe_execution, load_model, tensor
from .reused.genetic import GeneticOperators
from .runtime import OperationContext


class SolverFailure(RuntimeError):
    pass


def validate_scenario(interface, scenario: dict) -> None:
    for key, n in (("initial_state", interface.nx), ("reference", interface.ny), ("disturbance", interface.nd)):
        value = np.asarray(scenario.get(key, []), dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ContractError(f"Scenario {key} must contain {n} finite values")
    if type(scenario.get("steps")) is not int or scenario["steps"] < 1:
        raise ContractError("Scenario steps must be a positive integer")


def partition_scenarios(config: dict, interface) -> None:
    seen = set()
    for group in ("tuning_scenarios", "verification_scenarios", "test_scenarios"):
        if not config.get(group):
            raise ContractError(f"Explicit independent {group} are required")
        for scenario in config[group]:
            validate_scenario(interface, scenario)
            signature = canonical_hash({k: v for k, v in scenario.items() if k != "name"})
            if signature in seen:
                raise ContractError("Controller tuning, verification and test scenarios overlap")
            seen.add(signature)


class MPCPolicy:
    def __init__(self, model, config: dict, context: OperationContext | None = None):
        self.model, self.interface, self.config = model, model.interface, copy.deepcopy(config)
        self.context = context or OperationContext()
        self.horizon = int(config.get("horizon", 5))
        if not 1 <= self.horizon <= 100:
            raise ContractError("MPC horizon must be between 1 and 100")
        self.q, self.r, self.s = (float(config.get(k, default)) for k, default in (("q", 1), ("r", .01), ("s", .01)))
        if not all(math.isfinite(v) and v >= 0 for v in (self.q, self.r, self.s)) or self.q == 0:
            raise ContractError("MPC requires q > 0, r >= 0 and s >= 0")
        self.lo = np.array([v.lower for v in self.interface.controls])
        self.hi = np.array([v.upper for v in self.interface.controls])
        self.u_scale = tensor(self.hi - self.lo)
        self.y_scale = tensor([v.upper - v.lower for v in self.interface.outputs])
        self.xlo = tensor([v.lower for v in self.interface.states])
        self.xhi = tensor([v.upper for v in self.interface.states])
        self.ylo = tensor([v.lower for v in self.interface.outputs])
        self.yhi = tensor([v.upper for v in self.interface.outputs])
        self.rate = config.get("input_rate_bounds")
        if self.rate is not None:
            bounds = np.asarray(self.rate, dtype=float)
            if bounds.shape != (self.interface.nu, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
                raise ContractError("input_rate_bounds must be finite nu-by-2 action increments per sample")
        for p in model.parameters():
            p.requires_grad_(False)
        self.warm = None

    def rollout(self, sequence, x, reference, disturbance, previous):
        states, increments, objective = [], [], sequence.new_zeros(())
        for action in sequence:
            x = self.model(x, action, disturbance)
            objective = objective + self.q * ((self.model.output(x) - reference) / self.y_scale).square().sum()
            objective = objective + self.r * (action / self.u_scale).square().sum()
            objective = objective + self.s * ((action - previous) / self.u_scale).square().sum()
            states.append(x)
            increments.append(action - previous)
            previous = action
        states = torch.stack(states)
        outputs = self.model.output(states)
        constraints = [(states - self.xlo).flatten(), (self.xhi - states).flatten(),
                       (outputs - self.ylo).flatten(), (self.yhi - outputs).flatten()]
        if self.rate is not None:
            bounds, delta = tensor(self.rate), torch.stack(increments)
            constraints.extend([(delta - bounds[:, 0]).flatten(), (bounds[:, 1] - delta).flatten()])
        return objective, torch.cat(constraints)

    def act(self, x, reference, disturbance=None, previous=None):
        self.context.check()
        i = self.interface
        if disturbance is None:
            disturbance = np.zeros(i.nd)
        if previous is None:
            previous = np.clip(np.zeros(i.nu), self.lo, self.hi)
        xt, rt, dt, pt = map(tensor, (x, reference, disturbance, previous))
        initial = (np.tile(previous, (self.horizon, 1)) if self.warm is None else self.warm).flatten()

        def objective(flat):
            self.context.check()
            action = tensor(flat).requires_grad_(True)
            value, _ = self.rollout(action.reshape(self.horizon, i.nu), xt, rt, dt, pt)
            gradient, = torch.autograd.grad(value, action)
            return value.item(), gradient.detach().numpy()

        def constraints(flat):
            self.context.check()
            with torch.no_grad():
                return self.rollout(tensor(flat).reshape(self.horizon, i.nu), xt, rt, dt, pt)[1].numpy()

        def constraint_jac(flat):
            self.context.check()
            jac = torch.autograd.functional.jacobian(
                lambda z: self.rollout(z.reshape(self.horizon, i.nu), xt, rt, dt, pt)[1], tensor(flat))
            return jac.numpy()

        started = time.perf_counter()
        result = minimize(objective, initial, jac=True, method="SLSQP", bounds=list(zip(self.lo, self.hi)) * self.horizon,
                          constraints=[{"type": "ineq", "fun": constraints, "jac": constraint_jac}],
                          options={"maxiter": int(self.config.get("max_iterations", 80)),
                                   "ftol": float(self.config.get("solver_tolerance", 1e-8))})
        elapsed = time.perf_counter() - started
        violation = float(max(0, -np.min(constraints(result.x))))
        if not result.success or not np.isfinite(result.fun) or violation > 1e-6:
            raise SolverFailure(f"MPC solve failed: {result.message}; predicted constraint violation={violation}")
        sequence = result.x.reshape(self.horizon, i.nu)
        self.warm = np.concatenate((sequence[1:], sequence[-1:]), axis=0)
        return sequence[0].copy(), {"solve_seconds": elapsed, "iterations": int(result.nit),
                                    "predicted_objective": float(result.fun), "constraint_violation": violation}


def closed_loop(plant, policy, scenario: dict, context: OperationContext | None = None) -> dict:
    context = context or OperationContext()
    i = plant.interface
    validate_scenario(i, scenario)
    x = np.array(scenario["initial_state"], dtype=float)
    reference = np.array(scenario["reference"], dtype=float)
    disturbance = np.array(scenario["disturbance"], dtype=float)
    previous = np.clip(np.zeros(i.nu), [v.lower for v in i.controls], [v.upper for v in i.controls])
    if hasattr(policy, "warm"):
        policy.warm = None
    rows, durations = [], []
    if hasattr(policy, "reset"):
        policy.reset()
    for step in range(scenario["steps"]):
        context.check()
        started = time.perf_counter()
        action, diagnostics = policy.act(x, reference, disturbance, previous)
        durations.append(time.perf_counter() - started)
        xn = plant.step(x, action, disturbance)
        y = plant.output(tensor(xn)).detach().numpy()
        rows.append({"step": step, "x": x.tolist(), "u": action.tolist(), "next_x": xn.tolist(),
                     "output": y.tolist(), "reference": reference.tolist(), "d": disturbance.tolist(),
                     "diagnostics": diagnostics})
        x, previous = xn, action
    states = np.array([row["next_x"] for row in rows])
    controls = np.array([row["u"] for row in rows])
    outputs = np.array([row["output"] for row in rows])
    xl, xh = np.array([v.lower for v in i.states]), np.array([v.upper for v in i.states])
    ul, uh = np.array([v.lower for v in i.controls]), np.array([v.upper for v in i.controls])
    violation = np.maximum(xl - states, 0) + np.maximum(states - xh, 0)
    uviolation = np.maximum(ul - controls, 0) + np.maximum(controls - uh, 0)
    yl, yh = np.array([v.lower for v in i.outputs]), np.array([v.upper for v in i.outputs])
    yviolation = np.maximum(yl - outputs, 0) + np.maximum(outputs - yh, 0)
    normalized_error = (outputs - reference) / np.array([v.upper - v.lower for v in i.outputs])
    delta = np.diff(np.vstack((np.clip(np.zeros(i.nu), ul, uh), controls)), axis=0)
    metrics = {"rmse": float(np.sqrt(np.mean((outputs - reference) ** 2))),
        "rmse_per_output": np.sqrt(np.mean((outputs - reference) ** 2, axis=0)).tolist(),
        "normalized_rmse": float(np.sqrt(np.mean(normalized_error ** 2))),
        "max_state_violation": float(violation.max()), "max_input_violation": float(uviolation.max()),
        "max_output_violation": float(yviolation.max()),
        "input_total_variation": float(np.abs(delta).sum()),
        "mean_control_seconds": float(np.mean(durations)), "p95_control_seconds": float(np.percentile(durations, 95)),
        "control_effort": float(np.mean((controls / (uh - ul)) ** 2))}
    rate = getattr(policy, "rate", None)
    if rate is not None:
        rate = np.asarray(rate)
        metrics["max_rate_violation"] = float((np.maximum(rate[:, 0] - delta, 0) + np.maximum(delta - rate[:, 1], 0)).max())
    else:
        metrics["max_rate_violation"] = 0.0
    return {"scenario": scenario, "metrics": metrics, "trajectory": rows,
            "constraint_checks": {"state": True, "input": True, "output": True, "input_rate": rate is not None}}


def evaluate_controller(model, plant, parameters, scenarios, context) -> dict:
    runs = [closed_loop(plant, MPCPolicy(model, parameters, context), s, context) for s in scenarios]
    feasible = all(max(r["metrics"][k] for k in ("max_state_violation", "max_input_violation", "max_output_violation", "max_rate_violation")) <= 1e-6 for r in runs)
    score = float(np.mean([r["metrics"]["normalized_rmse"] + .01 * r["metrics"]["control_effort"] for r in runs]))
    return {"runs": runs, "hard_constraints_passed": feasible, "score": score}


def design_controller(workspace: Workspace, model_id: str, reference_id: str, config: dict,
                      context: OperationContext | None = None) -> str:
    context = context or OperationContext()
    model, plant = load_model(workspace, model_id), load_model(workspace, reference_id)
    model.interface.compatible(plant.interface)
    partition_scenarios(config, model.interface)
    parameters = config.get("parameters", {"horizon": 5, "q": 1.0, "r": .01, "s": .01})
    MPCPolicy(model, parameters, context)
    budget = int(config.get("budget", 8))
    if budget < 8:
        raise ContractError("Genetic tuning requires budget >= 8 (two generations of at least four candidates)")
    space = config.get("parameter_space", {"r": {"lower": .0001, "upper": .05}, "s": {"lower": .0001, "upper": .05}})
    for name, bounds in space.items():
        if name not in {"q", "r", "s", "horizon"} or name not in parameters:
            raise ContractError("Only declared controller-owned parameters q/r/s/horizon may be tuned")
        low, high = float(bounds["lower"]), float(bounds["upper"])
        if not math.isfinite(low + high) or low >= high or low < 0 or not low <= float(parameters[name]) <= high:
            raise ContractError("Invalid parameter bounds or baseline outside bounds")
        if name in {"q", "horizon"} and low <= 0:
            raise ContractError("q and horizon must stay positive")
        if name == "horizon" and (bounds.get("integer") is not True or high > 100):
            raise ContractError("horizon requires bounded integer search")
    rng = random.Random(int(config.get("seed", 42)))
    size = max(4, min(8, budget // 2))
    with workspace.stage("controller", model.interface, [model_id, reference_id], config) as stage:
        baseline = evaluate_controller(model, plant, parameters, config["tuning_scenarios"], context)
        trials = []
        population = GeneticOperators._initial_population(space, size, rng, parameters)
        generation = 0
        while len(trials) < budget:
            ranked = []
            for index, candidate in enumerate(population):
                if len(trials) >= budget:
                    break
                context.check()
                trial_params = {**parameters, **candidate}
                trial_params.pop("__lineage", None)
                try:
                    result = evaluate_controller(model, plant, trial_params, config["tuning_scenarios"], context)
                    trial = {"parameters": {k: candidate[k] for k in space}, "score": result["score"],
                             "hard_constraints_passed": result["hard_constraints_passed"]}
                except SolverFailure as exc:
                    trial = {"parameters": {k: candidate[k] for k in space}, "score": None,
                             "hard_constraints_passed": False, "error": str(exc)}
                trial.update(index=len(trials), generation=generation, lineage=candidate.get("__lineage", {"kind": "initial"}))
                trials.append(trial)
                ranked.append(trial)
                context.progress("controller_tuning", trial=len(trials), budget=budget, score=trial["score"])
            ranked.sort(key=GeneticOperators._fitness)
            population = []
            for _ in range(size):
                a, b = GeneticOperators._tournament(ranked, rng), GeneticOperators._tournament(ranked, rng)
                child, crossover = GeneticOperators._crossover(a["parameters"], b["parameters"], space, rng)
                child, mutation = GeneticOperators._mutate(child, space, rng)
                child["__lineage"] = {"parents": [a["index"], b["index"]], "crossover": crossover, "mutation": mutation}
                population.append(child)
            generation += 1
        best = min(trials, key=GeneticOperators._fitness)
        if not math.isfinite(GeneticOperators._fitness(best)):
            raise SolverFailure("No feasible genetic candidate")
        candidate_parameters = {**parameters, **best["parameters"]}
        verification_base = evaluate_controller(model, plant, parameters, config["verification_scenarios"], context)
        verification_best = evaluate_controller(model, plant, candidate_parameters, config["verification_scenarios"], context)
        accepted = (verification_best["hard_constraints_passed"] and
                    verification_best["score"] < verification_base["score"] * (1 - float(config.get("minimum_improvement", 0))))
        chosen = candidate_parameters if accepted else parameters
        final = evaluate_controller(model, plant, chosen, config["test_scenarios"], context)
        descriptor = {"prediction_model": model_id, "evaluation_model": reference_id, "parameters": chosen,
                      "solver": "scipy_slsqp_with_torch_derivatives", "reference_semantics": "current_reference_held_over_horizon",
                      "execution_surface": "registered_prediction_and_evaluation_models_not_original_simulink",
                      "prediction_execution": describe_execution(model),
                      "evaluation_execution": describe_execution(plant)}
        atomic_json(stage.root / "controller.json", descriptor)
        atomic_json(stage.root / "report.json", {"algorithm": "genetic_algorithm", "baseline": baseline, "trials": trials,
            "verification_baseline": verification_base, "verification_candidate": verification_best,
            "tuned_candidate_accepted": accepted, "selection": "tuned" if accepted else "baseline_retained",
            "test": final, "test_used_for_tuning": False})
        return stage.publish({"status": "verified" if final["hard_constraints_passed"] else "rejected",
                              "independent_verification": True, "tuned_candidate_accepted": accepted,
                              "test_score": final["score"]}, source=descriptor)


def load_controller(workspace: Workspace, artifact_id: str, context: OperationContext | None = None):
    root, _ = workspace.resolve(artifact_id, kind="controller")
    descriptor = read_json(root / "controller.json")
    return MPCPolicy(load_model(workspace, descriptor["prediction_model"]), descriptor["parameters"], context)
