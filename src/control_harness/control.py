"""Receding-horizon constrained control and source-derived genetic tuning."""
from __future__ import annotations

import copy
import math
import random
import time
import numpy as np
import torch
from scipy.optimize import minimize

from .artifacts import Workspace, atomic_json, read_json, file_hash
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


def closed_loop(plant, policy, scenario: dict, context: OperationContext | None = None, *, check_finite: bool = False) -> dict:
    from .plants import OriginalCallablePlant
    context = context or OperationContext()
    context.check()
    validate_scenario(plant.interface, scenario)
    if isinstance(plant, OriginalCallablePlant):
        # 原对象只接收已求出的当前控制量。每个工况独立 reset，异常/取消均关闭快照运行时。
        with plant.trajectory(scenario["initial_state"]) as adapter:
            return _closed_loop(plant, policy, scenario, context, True, adapter)
    return _closed_loop(plant, policy, scenario, context, check_finite, None)


def _closed_loop(plant, policy, scenario, context, check_finite, adapter):
    i = plant.interface
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
        try:
            action, diagnostics = policy.act(x, reference, disturbance, previous)
            durations.append(time.perf_counter() - started)
            if adapter is None:
                xn = plant.step(x, action, disturbance)
                y = plant.output(tensor(xn)).detach().numpy()
            else:
                if isinstance(policy, MPCPolicy) and plant.sampled_limits is not None:
                    # warm 已左移一拍，附上刚选定的动作即可重建本次完整预测控制序列。
                    sequence = np.vstack((action, policy.warm[:-1]))
                    plant.check_prediction_domain(policy.model, x, sequence, disturbance)
                xn = adapter.step(action, disturbance)
                y = adapter.output()
            if check_finite and not all(np.isfinite(value).all() for value in (action, xn, y)):
                raise FloatingPointError("Registered-model action/state/output is nonfinite")
        except (SolverFailure, FloatingPointError) as error:
            error.partial_run = {"scenario": scenario, "trajectory": rows,
                "completed_steps": len(rows), "failure": {"step": step, "type": type(error).__name__, "message": str(error)}}
            raise
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


def evaluate_controller(model, plant, parameters, scenarios, context, evaluation=None) -> dict:
    if evaluation is not None:
        from .objectives import normalize_evaluation, trajectory_metrics, score_metrics
        frozen = normalize_evaluation(evaluation)
        runs = []
        for scenario in scenarios:
            context.check()
            initial_control = np.clip(np.zeros(plant.interface.nu), [v.lower for v in plant.interface.controls], [v.upper for v in plant.interface.controls])
            initial_output = None
            try:
                initial_output = plant.output(tensor(scenario["initial_state"])).detach().cpu().numpy()
                if not np.isfinite(initial_output).all(): raise FloatingPointError("Initial plant output is nonfinite")
                run = closed_loop(plant, MPCPolicy(model, parameters, context), scenario, context, check_finite=True)
                run.update(initial_output=initial_output.tolist(), initial_control=initial_control.tolist(),
                    input_rate_bounds=parameters.get("input_rate_bounds"), status="completed")
                metrics = trajectory_metrics(run, plant.interface, frozen)
                objective = score_metrics(metrics, frozen)
                run.update(objective_metrics=metrics, objective_result=objective)
            except (SolverFailure, FloatingPointError) as error:
                run = copy.deepcopy(getattr(error, "partial_run", {"scenario": scenario, "trajectory": []}))
                run.update(status="failed", initial_output=None if initial_output is None or not np.isfinite(initial_output).all() else initial_output.tolist(), initial_control=initial_control.tolist(),
                    input_rate_bounds=parameters.get("input_rate_bounds"), error={"type": type(error).__name__, "message": str(error)},
                    objective_metrics=None, objective_result={"score": None, "hard_constraints_passed": False,
                        "terms": {}, "checks": [{"name": "simulation_completed", "passed": False, "reason": str(error)}]})
            runs.append(run)
        feasible = bool(runs) and all(run["objective_result"]["hard_constraints_passed"] and
            isinstance(run["objective_result"]["score"], (int, float)) and math.isfinite(run["objective_result"]["score"]) for run in runs)
        score = float(np.mean([run["objective_result"]["score"] for run in runs])) if feasible else None
        return {"runs": runs, "hard_constraints_passed": feasible, "score": score,
            "evaluation": frozen, "evaluation_hash": canonical_hash(frozen), "aggregation": "equal_weight_mean_scenario_scores",
            "execution_surface": "registered_prediction_and_evaluation_models_not_original_simulink"}
    runs = [closed_loop(plant, MPCPolicy(model, parameters, context), s, context) for s in scenarios]
    feasible = all(max(r["metrics"][k] for k in ("max_state_violation", "max_input_violation", "max_output_violation", "max_rate_violation")) <= 1e-6 for r in runs)
    score = float(np.mean([r["metrics"]["normalized_rmse"] + .01 * r["metrics"]["control_effort"] for r in runs]))
    return {"runs": runs, "hard_constraints_passed": feasible, "score": score}


def _genetic_trials(model, plant, parameters, scenarios, space, budget, rng, context, evaluation=None, on_trial=None):
    """One source-derived GA loop shared by legacy and goal-driven tuning."""
    size = max(4, min(8, budget // 2))
    trials = []
    population = GeneticOperators._initial_population(space, size, rng, parameters)
    generation = 0
    while len(trials) < budget:
        ranked = []
        for candidate in population:
            if len(trials) >= budget: break
            context.check()
            trial_params = {**parameters, **candidate}
            trial_params.pop("__lineage", None)
            if evaluation is not None: trial_params["horizon"] = int(trial_params.get("horizon", 5))
            try:
                result = (evaluate_controller(model, plant, trial_params, scenarios, context) if evaluation is None else
                    evaluate_controller(model, plant, trial_params, scenarios, context, evaluation=evaluation))
                trial = {"parameters": {key: candidate[key] for key in space}, "score": result["score"],
                    "hard_constraints_passed": result["hard_constraints_passed"]}
                if evaluation is not None:
                    trial.update(complete_parameters=trial_params, tuning_evaluation=result,
                        status="failed" if any(run.get("status") == "failed" for run in result["runs"]) else "feasible" if result["hard_constraints_passed"] else "infeasible")
                    if trial["status"] == "failed": trial["error"] = "One or more actual registered-model simulations failed; see tuning_evaluation.runs"
            except (SolverFailure, FloatingPointError) as error:
                if evaluation is None and not isinstance(error, SolverFailure): raise
                trial = {"parameters": {key: candidate[key] for key in space}, "score": None,
                    "hard_constraints_passed": False, "error": str(error)}
                if evaluation is not None: trial.update(complete_parameters=trial_params, status="failed")
            trial.update(index=len(trials), generation=generation, lineage=candidate.get("__lineage", {"kind": "initial"}))
            trials.append(trial); ranked.append(trial)
            if on_trial is not None: on_trial(trial)
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
    return trials


def design_controller(workspace: Workspace, model_id: str, reference_id: str, config: dict,
                      context: OperationContext | None = None) -> str:
    if "evaluation" in config:
        return _design_goal_controller(workspace, model_id, reference_id, config, context)
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
    with workspace.stage("controller", model.interface, [model_id, reference_id], config) as stage:
        baseline = evaluate_controller(model, plant, parameters, config["tuning_scenarios"], context)
        trials = _genetic_trials(model, plant, parameters, config["tuning_scenarios"], space, budget, rng, context)
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


def _goal_settings(config, interface):
    if not isinstance(config, dict) or set(config) - {"parameters", "parameter_space", "budget", "seed", "minimum_improvement", "evaluation", "plan_refs",
            "tuning_scenarios", "verification_scenarios", "test_scenarios"}:
        raise ContractError("Unsupported goal-driven controller configuration field")
    parameters = {"horizon": 5, "q": 1., "r": .01, "s": .01, "max_iterations": 80, "solver_tolerance": 1e-8}
    declared = config.get("parameters", {})
    if not isinstance(declared, dict) or set(declared) - {*parameters, "input_rate_bounds"}:
        raise ContractError("Only actual MPC settings may be supplied as tuning parameters")
    parameters.update(copy.deepcopy(declared))
    for key in ("horizon", "max_iterations"):
        value = parameters[key]
        if type(value) is not int or value < 1 or (key == "horizon" and value > 100):
            raise ContractError(key + " must be an explicit positive integer in the supported MPC range")
    for key in ("q", "r", "s", "solver_tolerance"):
        value = parameters[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0 or (key in {"q", "solver_tolerance"} and value == 0):
            raise ContractError(key + " must be a finite MPC coefficient in its supported range")
        parameters[key] = float(value)
    if "input_rate_bounds" in parameters:
        bounds = parameters["input_rate_bounds"]
        if (not isinstance(bounds, list) or len(bounds) != interface.nu or any(not isinstance(pair, list) or len(pair) != 2
                or any(type(value) not in {int, float} or not math.isfinite(value) for value in pair) or pair[0] >= pair[1] for pair in bounds)):
            raise ContractError("input_rate_bounds must be finite ordered per-sample increments")
    budget = config.get("budget", 8)
    if type(budget) is not int or budget < 8:
        raise ContractError("Genetic tuning requires an integer budget >= 8")
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1: raise ContractError("seed must be an integer in 0..2**32-1")
    minimum = config.get("minimum_improvement", 0.)
    if type(minimum) not in {int, float} or not math.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ContractError("minimum_improvement must be a finite relative fraction in [0,1]")
    space = config.get("parameter_space", {"r": {"lower": .0001, "upper": .05}, "s": {"lower": .0001, "upper": .05}})
    if not isinstance(space, dict) or not space:
        raise ContractError("A nonempty controller-owned parameter_space is required")
    for name, bounds in space.items():
        if name not in {"q", "r", "s", "horizon"} or not isinstance(bounds, dict) or set(bounds) - {"lower", "upper", "integer"}:
            raise ContractError("Only bounded controller-owned q/r/s/horizon may be tuned")
        low, high = bounds.get("lower"), bounds.get("upper")
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in (low, high)) or low < 0 or low >= high or not low <= parameters[name] <= high:
            raise ContractError("Invalid parameter bounds or baseline outside bounds")
        if name in {"q", "horizon"} and low <= 0: raise ContractError("q and horizon bounds must stay positive")
        if "integer" in bounds and type(bounds["integer"]) is not bool: raise ContractError("integer must be an explicit boolean")
        if name == "horizon" and (bounds.get("integer") is not True or high > 100):
            raise ContractError("horizon requires bounded integer search")
        if bounds.get("integer") is True and any(float(value) != int(value) for value in (low, high, parameters[name])):
            raise ContractError("Integer parameter endpoints and baseline must be integral")
    return parameters, space, budget, seed, float(minimum)


def _goal_preflight(config, interface, parameters, space, budget):
    physical_conditions = set()
    for group in ("tuning_scenarios", "verification_scenarios", "test_scenarios"):
        for scenario in config[group]:
            if set(scenario) - {"name", "initial_state", "reference", "disturbance", "steps"}:
                raise ContractError("Unknown scenario fields cannot establish independent operating conditions")
            for key, variables in (("initial_state", interface.states), ("reference", interface.outputs), ("disturbance", interface.disturbances)):
                values = scenario[key]
                if not isinstance(values, list) or any(type(value) not in {int, float} or not math.isfinite(value) for value in values):
                    raise ContractError("Scenario " + key + " requires explicit finite numeric JSON values")
                if any(value < variable.lower or value > variable.upper for value, variable in zip(values, variables)):
                    raise ContractError("Scenario " + key + " is outside its declared physical bounds")
            # Different labels, record lengths and JSON representations of the
            # same coordinates do not create an independent physical condition.
            # Tuple float equality also treats integer/float and signed zero as
            # equal; all coordinates have already passed finite/type checks.
            signature = tuple(tuple(float(value) for value in scenario[key]) for key in ("initial_state", "reference", "disturbance"))
            if signature in physical_conditions:
                raise ContractError("Independent tuning, verification and test operating conditions overlap, including shared trajectory prefixes")
            physical_conditions.add(signature)
    max_horizon = int(space["horizon"]["upper"]) if "horizon" in space else parameters["horizon"]
    variables = max_horizon * interface.nu
    margins = max_horizon * (2 * interface.nx + 2 * interface.ny + (2 * interface.nu if "input_rate_bounds" in parameters else 0))
    if variables > 256 or variables * margins > 5_000_000:
        raise ContractError("Tuning MPC decision/Jacobian resource budget exceeded; reduce horizon/width before SLSQP")
    steps = {name: sum(scenario["steps"] for scenario in config[name + "_scenarios"]) for name in ("tuning", "verification", "test")}
    # Include complete runs, initial states and report copies, not merely input
    # widths. This bounds one job while leaving larger engineering goals open.
    per_row = 2 * interface.nx + interface.nu + 2 * interface.ny + interface.nd + 8
    estimated_scalars = 4 * per_row * ((budget + 1) * steps["tuning"] + 2 * steps["verification"] + steps["test"])
    if estimated_scalars > 5_000_000:
        raise ContractError("Tuning aggregate trajectory/report resource budget exceeded; use smaller explicit job batches")
    return {"max_decision_variables": 256, "max_dense_jacobian_values": 5_000_000,
        "max_estimated_trajectory_report_scalars": 5_000_000, "decision_variables_upper_bound": variables,
        "constraint_jacobian_values_upper_bound": variables * margins, "estimated_trajectory_report_scalars": estimated_scalars,
        "scope": "one tuning job; not a limit on the overall engineering objective"}


def _export_goal_bundle(stage, workspace, model_id, model, chosen, frozen, evaluation_hash, report):
    from .predictors import PredictionResult
    from .design_export import export_controller_design
    from .modeling import CorrectedDynamics, DeclarativeDynamics
    spec = copy.deepcopy(getattr(model, "spec", {"representation": "nominal_plus_learned_residual"}))
    representation = spec.get("representation")
    if representation not in {"discrete_polynomial", "continuous_polynomial", "learned_discrete_mlp", "nominal_plus_learned_residual"} or (
            isinstance(model, CorrectedDynamics) and not isinstance(model.nominal, DeclarativeDynamics)):
        return {"status": "unsupported_predictor_export", "reason": "This registered runtime requires an explicit portable predictor adapter; numerical tuning still uses its registered model."}
    model_root, _ = workspace.resolve(model_id)
    prediction = PredictionResult(model, model.interface, spec, model_id, "direct",
        {"route": "direct", "scope": "registered_prediction_model_for_chosen_tuned_controller"}, True, model_root, (model_id,))
    design = {"schema_version": 1, "mode": "tuned_controller_export", "status": "chosen_on_tuning_and_verification",
        "model_instantiated": True, "template": None, "parameters": {}, "required_parameters": [],
        "physical_interface": model.interface.to_dict(), "prediction": prediction.to_dict(),
        "controller_config": {key: value for key, value in chosen.items() if key not in {"q", "r", "s"}},
        "weights": {key: chosen[key] for key in ("q", "r", "s")},
        "external_evaluation": frozen, "evaluation_hash": evaluation_hash,
        "closed_loop": {"status": "registered_models_evaluated_before_export", "original_simulink_or_matlab_plant": "not_run"},
        "tuning_provenance": {"algorithm": "genetic_algorithm", "report": "tuning_report.json", "scope": "Bundle construction performs no additional GA or closed-loop execution."}}
    root = stage.root / "bundle"
    export_controller_design(root, design, prediction=prediction)
    # These files are still unpublished staging outputs. Preserve the S07
    # export-only scope and add separately scoped actual tuning provenance.
    atomic_json(root / "tuning_report.json", {"algorithm": report["algorithm"], "chosen_parameters": chosen,
        "selection": report["selection"], "tuned_candidate_accepted": report["tuned_candidate_accepted"],
        "evaluation": frozen, "evaluation_hash": evaluation_hash, "budget": report["budget"], "actual_trials": len(report["trials"]),
        "prediction_model": model_id, "evaluation_model": report["evaluation_model"],
        "execution_surface": report["execution_surface"], "test_used_for_selection": False,
        "registered_model_test": report["test"], "stability_proof": False,
        "bundle_construction": {"additional_GA": "not_run", "additional_closed_loop": "not_run"}})
    readme = root / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\n\n## Source tuning provenance\n\nThis package contains the chosen parameters from an already executed genetic search and registered-model closed-loop validation. See tuning_report.json for the frozen external objective and actual trial count. The export-only closed_loop=not_run / ga_trials=0 fields describe package construction and single-call inference, not the earlier tuning. Original MATLAB/Simulink plant closed loops and stability proofs were not performed.\n", encoding="utf-8")
    manifest_path = root / "bundle_manifest.json"
    manifest = read_json(manifest_path)
    manifest["files"].update({"tuning_report.json": file_hash(root / "tuning_report.json"), "README.md": file_hash(readme)})
    manifest["tuning_provenance"] = {"report": "tuning_report.json", "actual_GA_trials_before_export": len(report["trials"]),
        "registered_model_closed_loop_before_export": True, "original_matlab_simulink_closed_loop": "not_run"}
    atomic_json(manifest_path, manifest)
    return {"status": "exported", "directory": "bundle", "manifest": "bundle/bundle_manifest.json", "tuning_provenance": "bundle/tuning_report.json",
        "scope": "Chosen feasible controller; S07 export-only scope is distinct from the recorded registered-model tuning."}


def _design_goal_controller(workspace, model_id, reference_id, config, context=None):
    from .objectives import normalize_evaluation
    context = context or OperationContext()
    frozen = copy.deepcopy(normalize_evaluation(config["evaluation"]))
    evaluation_hash = canonical_hash(frozen)
    model, plant = load_model(workspace, model_id), load_model(workspace, reference_id)
    model.interface.compatible(plant.interface)
    if set(frozen["hard_gates"]["per_output"]) - {value.name for value in model.interface.outputs}:
        raise ContractError("External per_output gates must name declared physical outputs")
    for group in ("tuning_scenarios", "verification_scenarios", "test_scenarios"):
        if not isinstance(config.get(group), list) or not config[group] or any(not isinstance(scenario, dict) for scenario in config[group]):
            raise ContractError("Explicit independent scenario lists are required")
    partition_scenarios(config, model.interface)
    parameters, space, budget, seed, minimum = _goal_settings(config, model.interface)
    resources = _goal_preflight(config, model.interface, parameters, space, budget)
    MPCPolicy(model, parameters, context)
    execution = "registered_prediction_and_evaluation_models_not_original_simulink"
    def evaluate(settings, scenarios):
        return evaluate_controller(model, plant, settings, scenarios, context, evaluation=frozen)
    with workspace.stage("controller", model.interface, [model_id, reference_id], config) as stage:
        baseline = evaluate(parameters, config["tuning_scenarios"])
        atomic_json(stage.root / "baseline.json", baseline)
        trials = _genetic_trials(model, plant, parameters, config["tuning_scenarios"], space, budget, random.Random(seed), context,
            evaluation=frozen, on_trial=lambda trial: atomic_json(stage.root / "trials" / f"trial-{trial['index']:06d}.json", trial))
        feasible = [trial for trial in trials if math.isfinite(GeneticOperators._fitness(trial))]
        best = min(feasible, key=GeneticOperators._fitness) if feasible else None
        verification_base = evaluate(parameters, config["verification_scenarios"])
        verification_best = evaluate(best["complete_parameters"], config["verification_scenarios"]) if best is not None else {
            "runs": [], "score": None, "hard_constraints_passed": False, "status": "not_run", "reason": "no_feasible_genetic_candidate",
            "evaluation": frozen, "evaluation_hash": evaluation_hash}
        baseline_feasible = baseline["hard_constraints_passed"] and verification_base["hard_constraints_passed"]
        accepted = best is not None and verification_best["hard_constraints_passed"] and (not baseline_feasible or
            verification_best["score"] < verification_base["score"] * (1 - minimum))
        chosen = copy.deepcopy(best["complete_parameters"]) if accepted else copy.deepcopy(parameters) if baseline_feasible else None
        selection = "tuned" if accepted else "baseline_retained" if chosen is not None else "no_feasible_controller"
        final = evaluate(chosen, config["test_scenarios"]) if chosen is not None else {
            "runs": [], "score": None, "hard_constraints_passed": False, "status": "not_run", "reason": "no_verified_feasible_choice",
            "evaluation": frozen, "evaluation_hash": evaluation_hash}
        if canonical_hash(frozen) != evaluation_hash: raise ContractError("External evaluation changed during tuning")
        descriptor = {"prediction_model": model_id, "evaluation_model": reference_id, "parameters": chosen if chosen is not None else parameters,
            "parameters_status": "selected_before_test" if chosen is not None else "not_deployable_no_feasible_choice",
            "solver": "scipy_slsqp_with_torch_derivatives", "reference_semantics": "current_reference_held_over_horizon",
            "execution_surface": execution, "prediction_execution": describe_execution(model), "evaluation_execution": describe_execution(plant),
            "evaluation": frozen, "evaluation_hash": evaluation_hash}
        report = {"algorithm": "genetic_algorithm", "evaluation": frozen, "evaluation_hash": evaluation_hash,
            "execution_surface": execution, "prediction_model": model_id, "evaluation_model": reference_id,
            "prediction_execution": describe_execution(model), "evaluation_execution": describe_execution(plant),
            "baseline_parameters": parameters, "baseline": baseline, "trials": trials, "budget": budget,
            "best": best, "last": trials[-1], "failed_candidates": [trial for trial in trials if not trial["hard_constraints_passed"] or trial["score"] is None],
            "verification_baseline": verification_base, "verification_candidate": verification_best,
            "baseline_feasible_on_tuning_and_verification": baseline_feasible, "tuned_candidate_accepted": bool(accepted),
            "selection": selection, "selection_reason": "baseline_infeasible_candidate_verified" if accepted and not baseline_feasible else selection,
            "chosen_parameters": chosen, "minimum_improvement": minimum, "test": final, "test_used_for_tuning": False,
            "test_used_for_selection": False, "test_evaluations": 1 if chosen is not None else 0,
            "selection_partitions": ["tuning", "verification"], "scenario_partitions": {group: copy.deepcopy(config[group]) for group in
                ("tuning_scenarios", "verification_scenarios", "test_scenarios")},
            "resource_boundary": resources,
            "stability_proof": False, "original_matlab_simulink_closed_loop": "not_run",
            "confirmation_sources": context.case_confirmation_sources}
        report["export"] = (_export_goal_bundle(stage, workspace, model_id, model, chosen, frozen, evaluation_hash, report)
            if chosen is not None and final["hard_constraints_passed"] else {"status": "not_exported", "reason": "final_choice_failed_acceptance"})
        atomic_json(stage.root / "controller.json", descriptor); atomic_json(stage.root / "report.json", report)
        context.check()
        return stage.publish({"status": "verified" if final["hard_constraints_passed"] else "rejected", "independent_verification": True,
            "tuned_candidate_accepted": bool(accepted), "test_score": final["score"], "evaluation_hash": evaluation_hash,
            "stability_proof": False, "execution_surface": execution}, source=descriptor)


def load_controller(workspace: Workspace, artifact_id: str, context: OperationContext | None = None):
    root, _ = workspace.resolve(artifact_id, kind="controller")
    descriptor = read_json(root / "controller.json")
    return controller_from_descriptor(workspace, descriptor, context)


def controller_from_descriptor(workspace, descriptor, context=None):
    """工作区控制器及交付前检查共用同一个装载入口和 MPC 参数。"""
    return MPCPolicy(load_model(workspace, descriptor["prediction_model"]), descriptor["parameters"], context)
