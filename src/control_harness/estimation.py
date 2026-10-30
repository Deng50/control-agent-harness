"""Measurement-driven KF/EKF feedback with explicit simulation and artifact boundaries.

The estimator never receives a true state. Only the simulation adapter may turn
a reference simulator state into an output measurement; MPC receives x_hat(k|k).
"""
from __future__ import annotations

import copy
import math
import numpy as np
import torch

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, canonical_hash
from .control import MPCPolicy, closed_loop, validate_scenario
from .modeling import DeclarativeDynamics, load_model, tensor
from .runtime import OperationContext


def _vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ContractError(f"{name} must contain {size} finite values")
    return result.copy()


def _covariance(value, size: int, name: str, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.isfinite(result).all():
        raise ContractError(f"{name} must be a finite {size}-by-{size} covariance")
    tolerance = 1e-12 * max(1.0, float(np.linalg.norm(result, ord=2)))
    if not np.allclose(result, result.T, rtol=0, atol=tolerance):
        raise ContractError(f"{name} must be symmetric")
    result = (result + result.T) / 2
    eigenvalues, vectors = np.linalg.eigh(result)
    if eigenvalues[0] < -tolerance or (positive and eigenvalues[0] <= 0):
        raise ContractError(f"{name} must be positive {'definite' if positive else 'semidefinite'}")
    # Repair only floating-point roundoff, never a substantively invalid covariance.
    if eigenvalues[0] < 0:
        result = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
    return result


def estimator_config(model, config: dict) -> dict:
    """Compile dimensions/measurement order; covariance defaults are labelled assumptions."""
    i = model.interface
    indices = config.get("measurement_output_indices", list(range(i.ny)))
    if (not isinstance(indices, list) or not indices
            or any(type(k) is not int or not 0 <= k < i.ny for k in indices)
            or len(indices) != len(set(indices))):
        raise ContractError("measurement_output_indices must be unique declared output indices")
    linear = isinstance(model, DeclarativeDynamics) and not model.terms
    method = str(config.get("method", "auto")).upper()
    method = ("KF" if linear else "EKF") if method == "AUTO" else method
    if method not in {"KF", "EKF"} or (method == "KF" and not linear):
        raise ContractError("KF requires an affine declarative model; use EKF for nonlinear/corrected models")
    state_scales = np.array([v.upper - v.lower for v in i.states])
    measurement_scales = np.array([i.outputs[k].upper - i.outputs[k].lower for k in indices])
    defaults = {"Q": np.diag((state_scales * .001) ** 2),
                "R": np.diag((measurement_scales * .005) ** 2),
                "P0": np.diag((state_scales * .05) ** 2)}
    matrices = {name: _covariance(config.get(name, value), len(indices) if name == "R" else i.nx,
                                 name, positive=name == "R").tolist()
                for name, value in defaults.items()}
    initial = _vector(config.get("initial_estimate"), i.nx, "initial_estimate")
    relative = float(config.get("observability_relative_tolerance", 1e-8))
    if not math.isfinite(relative) or not 0 < relative < 1:
        raise ContractError("observability_relative_tolerance must be in (0,1)")
    return {"method": method, "measurement_output_indices": list(indices), **matrices,
            "initial_estimate": initial.tolist(), "observability_relative_tolerance": relative,
            "measurement_names": [i.outputs[k].name for k in indices],
            "measurement_units": [i.outputs[k].unit for k in indices],
            "measurement_time": "y(k)=h(x(k)); correct before u(k), then predict x(k+1)",
            "covariance_provenance": {name: "explicit_configuration" if name in config else
                "range_scaled_engineering_initialization_not_measured_noise" for name in defaults}}


def _rank_report(matrix: np.ndarray, nx: int, tolerance: float) -> dict:
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = tolerance * max(float(singular[0]), np.finfo(float).tiny)
    rank = int(np.count_nonzero(singular > threshold))
    return {"rank": rank, "required_rank": nx, "passed": rank == nx,
            "relative_tolerance": tolerance, "singular_values": singular.tolist(),
            "relative_min_singular_value": float(singular[-1] / singular[0]) if singular[0] else 0.0}


class StateEstimator:
    """KF or autograd-linearized EKF, with Joseph update and known applied controls."""
    def __init__(self, model, config: dict):
        self.model, self.interface = model, model.interface
        self.config = estimator_config(model, config)
        self.indices = self.config["measurement_output_indices"]
        self.Q, self.R = np.asarray(self.config["Q"]), np.asarray(self.config["R"])
        self.reset()
        if self.config["method"] == "KF":
            # Scale coordinates so units cannot decide the numerical rank.
            A = model.A.detach().numpy()
            H = model.C.detach().numpy()[self.indices]
            xs = np.array([v.upper - v.lower for v in self.interface.states])
            ys = np.array([self.interface.outputs[k].upper - self.interface.outputs[k].lower for k in self.indices])
            O = np.vstack([H @ np.linalg.matrix_power(A, k) for k in range(self.interface.nx)])
            O = O * xs[None, :] / np.tile(ys, self.interface.nx)[:, None]
            report = _rank_report(O, self.interface.nx, self.config["observability_relative_tolerance"])
            if not report["passed"]:
                raise ContractError(f"Declared measurements do not observe the linear state: rank {report['rank']}/{self.interface.nx}")

    def reset(self) -> None:
        self.x = np.array(self.config["initial_estimate"], dtype=float)
        self.P = np.array(self.config["P0"], dtype=float)
        self.history: list[dict] = []
        self.linearizations: list[tuple[np.ndarray, np.ndarray]] = []
        self._measurement_pending = False

    def measurement(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.output(x)[self.indices]

    def correct(self, measurement) -> dict:
        if self._measurement_pending:
            raise ContractError("Apply predict with the actual control before the next measurement")
        y = _vector(measurement, len(self.indices), "measurement")
        xt = tensor(self.x)
        expected = _vector(self.measurement(xt).detach().numpy(), len(self.indices), "predicted measurement")
        H = torch.autograd.functional.jacobian(self.measurement, xt).detach().numpy()
        if H.shape != (len(self.indices), self.interface.nx) or not np.isfinite(H).all():
            raise ContractError("Measurement Jacobian has invalid shape or values")
        innovation = y - expected
        S = _covariance(H @ self.P @ H.T + self.R, len(self.indices), "innovation covariance", positive=True)
        gain = np.linalg.solve(S, H @ self.P).T
        self.x = _vector(self.x + gain @ innovation, self.interface.nx, "corrected estimate")
        IKH = np.eye(self.interface.nx) - gain @ H
        self.P = _covariance(IKH @ self.P @ IKH.T + gain @ self.R @ gain.T,
                             self.interface.nx, "Joseph posterior covariance")
        self._H = H
        self._measurement_pending = True
        result = {"measurement": y.tolist(), "predicted_measurement": expected.tolist(),
                  "innovation": innovation.tolist(), "state_estimate": self.x.tolist(),
                  "posterior_covariance": self.P.tolist(),
                  "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(self.P)[0]),
                  "normalized_innovation_squared": float(innovation @ np.linalg.solve(S, innovation))}
        self.history.append(result)
        return copy.deepcopy(result)

    def predict(self, applied_control, disturbance) -> None:
        if not self._measurement_pending:
            raise ContractError("Correct the current-time measurement before predicting the next sample")
        i = self.interface
        u = tensor(_vector(applied_control, i.nu, "applied_control"))
        d = tensor(_vector(disturbance, i.nd, "disturbance"))
        x = tensor(self.x)
        F = torch.autograd.functional.jacobian(lambda value: self.model(value, u, d), x).detach().numpy()
        if F.shape != (i.nx, i.nx) or not np.isfinite(F).all():
            raise ContractError("State-transition Jacobian has invalid shape or values")
        self.x = _vector(self.model(x, u, d).detach().numpy(), i.nx, "predicted state")
        self.P = _covariance(F @ self.P @ F.T + self.Q, i.nx, "predicted covariance")
        self.linearizations.append((self._H.copy(), F))
        self.history[-1].update(predicted_next_state=self.x.tolist(), applied_control=u.tolist())
        self._measurement_pending = False

    def observability(self) -> dict:
        if len(self.linearizations) < self.interface.nx:
            raise ContractError("At least nx measured transitions are required for trajectory observability")
        xs = np.array([v.upper - v.lower for v in self.interface.states])
        ys = np.array([self.interface.outputs[k].upper - self.interface.outputs[k].lower for k in self.indices])
        phi, rows = np.eye(self.interface.nx), []
        for H, F in self.linearizations:
            rows.append((H @ phi) * xs[None, :] / ys[:, None])
            phi = F @ phi
        report = _rank_report(np.vstack(rows), self.interface.nx, self.config["observability_relative_tolerance"])
        report["scope"] = ("linear_model_observability" if self.config["method"] == "KF" else
                           "local_linearized_executed_estimate_trajectory_not_global_nonlinear_observability")
        return report


class EstimatedMPCPolicy:
    """Production measurement interface: act_measurement receives y(k), never x(k)."""
    def __init__(self, controller: MPCPolicy, config: dict):
        self.controller, self.interface = controller, controller.interface
        self.estimator = StateEstimator(controller.model, config)
        self.rate = controller.rate
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.controller.warm = None
        i = self.interface
        self.previous = np.clip(np.zeros(i.nu), [v.lower for v in i.controls], [v.upper for v in i.controls])

    def act_measurement(self, measurement, reference, disturbance, previous=None):
        self.controller.context.check()
        i = self.interface
        reference = _vector(reference, i.ny, "reference")
        disturbance = _vector(disturbance, i.nd, "disturbance")
        previous = self.previous if previous is None else _vector(previous, i.nu, "previous control")
        diagnostics = self.estimator.correct(measurement)
        # This is the only controller invocation: full plant state is unavailable here.
        action, control_diagnostics = self.controller.act(
            self.estimator.x.copy(), reference, disturbance, previous)
        self.estimator.predict(action, disturbance)
        self.previous = action.copy()
        return action, {**control_diagnostics, "estimator": diagnostics,
                        "controller_state_source": "posterior_state_estimate"}


class EstimatorSimulationAdapter:
    """Explicit simulated sensor boundary compatible with control.closed_loop."""
    def __init__(self, plant, policy: EstimatedMPCPolicy, *, seed: int, sensor_covariance):
        plant.interface.compatible(policy.interface)
        if type(seed) is not int or seed < 0:
            raise ContractError("Simulated sensor seed must be a nonnegative integer")
        self.plant, self.policy, self.seed, self.rate = plant, policy, seed, policy.rate
        self.sensor_covariance = _covariance(sensor_covariance, len(policy.estimator.indices), "sensor covariance")
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.rng = np.random.default_rng(self.seed)
        self.open_loop_estimate = np.array(self.policy.estimator.config["initial_estimate"])
        self.estimation_rows: list[dict] = []

    def act(self, true_state, reference, disturbance, previous):
        indices = self.policy.estimator.indices
        noiseless = self.plant.output(tensor(true_state)).detach().numpy()[indices]
        measurement = noiseless + self.rng.multivariate_normal(np.zeros(len(indices)), self.sensor_covariance)
        action, diagnostics = self.policy.act_measurement(measurement, reference, disturbance, previous)
        self.estimation_rows.append({"true_state": np.asarray(true_state).tolist(),
            "open_loop_estimate": self.open_loop_estimate.tolist(), **diagnostics["estimator"]})
        self.open_loop_estimate = self.policy.controller.model.step(self.open_loop_estimate, action, disturbance)
        return action, diagnostics


def estimator_closed_loop(plant, model, controller_parameters: dict, scenario: dict, config: dict,
                          context: OperationContext | None = None) -> dict:
    context = context or OperationContext()
    plant.interface.compatible(model.interface)
    validate_scenario(model.interface, scenario)
    if scenario["steps"] < model.interface.nx:
        raise ContractError("Estimator validation requires at least nx steps")
    policy = EstimatedMPCPolicy(MPCPolicy(model, controller_parameters, context), config)
    sensor_covariance = config.get("sensor_noise_covariance", policy.estimator.R)
    adapter = EstimatorSimulationAdapter(plant, policy, seed=config.get("seed", 42),
                                         sensor_covariance=sensor_covariance)
    result = closed_loop(plant, adapter, scenario, context)
    rows = adapter.estimation_rows
    actual = np.array([row["true_state"] for row in rows])
    estimate = np.array([row["state_estimate"] for row in rows])
    baseline = np.array([row["open_loop_estimate"] for row in rows])
    scales = np.array([v.upper - v.lower for v in model.interface.states])
    result["estimation"] = {"trajectory": rows, "method": policy.estimator.config["method"],
        "observability": policy.estimator.observability(),
        "metrics": {"estimation_rmse": float(np.sqrt(np.mean((estimate - actual) ** 2))),
            "normalized_estimation_rmse": float(np.sqrt(np.mean(((estimate - actual) / scales) ** 2))),
            "rmse_per_state": np.sqrt(np.mean((estimate - actual) ** 2, axis=0)).tolist(),
            "open_loop_estimation_rmse": float(np.sqrt(np.mean((baseline - actual) ** 2))),
            "minimum_covariance_eigenvalue": min(row["minimum_covariance_eigenvalue"] for row in rows),
            "mean_normalized_innovation_squared": float(np.mean([row["normalized_innovation_squared"] for row in rows]))},
        "noise": {"seed": adapter.seed, "sensor_covariance": adapter.sensor_covariance.tolist(),
            "source": "seeded_simulated_measurements_not_experimental_data", "injected_process_noise": False},
        "measurement_interface": policy.estimator.config}
    return result


def validate_estimator(workspace: Workspace, model_id: str, controller_id: str, reference_id: str,
                       config: dict, context: OperationContext | None = None) -> str:
    """Validate a fixed observer on disjoint scenarios, and publish only truthful acceptance."""
    context = context or OperationContext()
    model, plant = load_model(workspace, model_id), load_model(workspace, reference_id)
    model.interface.compatible(plant.interface)
    root, _ = workspace.resolve(controller_id, kind="controller")
    descriptor = read_json(root / "controller.json")
    if descriptor["prediction_model"] != model_id:
        raise ContractError("Estimator model must be the exact prediction model of the bound controller")
    compiled = estimator_config(model, config)
    seen = set()
    for name in ("verification_scenarios", "test_scenarios"):
        if not isinstance(config.get(name), list) or not config[name]:
            raise ContractError(f"Explicit independent {name} are required")
        for scenario in config[name]:
            validate_scenario(model.interface, scenario)
            signature = canonical_hash({k: value for k, value in scenario.items() if k != "name"})
            if signature in seen:
                raise ContractError("Estimator verification/test scenarios overlap")
            seen.add(signature)
    threshold = float(config.get("max_normalized_estimation_rmse", .25))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ContractError("max_normalized_estimation_rmse must be finite and positive")
    seed = config.get("seed", 42)
    if type(seed) is not int or seed < 0:
        raise ContractError("Estimator validation seed must be a nonnegative integer")
    with workspace.stage("estimator", model.interface, [model_id, controller_id, reference_id], config) as stage:
        runs = {}
        serial = 0
        for group in ("verification_scenarios", "test_scenarios"):
            runs[group] = []
            for scenario in config[group]:
                context.check()
                run_config = {**config, **compiled, "seed": seed + serial}
                result = estimator_closed_loop(plant, model, descriptor["parameters"], scenario, run_config, context)
                metrics, estimation = result["metrics"], result["estimation"]
                checks = {"local_observability": estimation["observability"]["passed"],
                    "estimation_error": estimation["metrics"]["normalized_estimation_rmse"] <= threshold,
                    "covariance_semidefinite": estimation["metrics"]["minimum_covariance_eigenvalue"] >= -1e-12,
                    "closed_loop_hard_constraints": max(metrics[k] for k in
                        ("max_state_violation", "max_input_violation", "max_output_violation", "max_rate_violation")) <= 1e-6}
                result["acceptance"] = {"checks": checks, "passed": all(checks.values())}
                runs[group].append(result)
                serial += 1
                context.progress("estimator_validation", group=group, scenario=scenario.get("name"),
                                 passed=result["acceptance"]["passed"])
        accepted = all(run["acceptance"]["passed"] for group in runs.values() for run in group)
        saved = {"prediction_model": model_id, "controller": controller_id, "evaluation_model": reference_id,
                 "estimator_config": compiled, "measurement_interface": "act_measurement(y_k,reference,disturbance)",
                 "controller_state_source": "posterior_state_estimate", "covariance_update": "Joseph"}
        atomic_json(stage.root / "estimator.json", saved)
        atomic_json(stage.root / "report.json", {**runs, "test_used_for_tuning": False,
            "max_normalized_estimation_rmse": threshold, "parameters_optimized": False,
            "scope": "reference_simulator_measurement_feedback_not_original_simulink", "accepted": accepted})
        return stage.publish({"status": "verified" if accepted else "rejected", "independent_test": True,
                              "actual_estimate_feedback": True, "measurement_only_runtime": True}, source=saved)


def load_estimator_controller(workspace: Workspace, artifact_id: str,
                              context: OperationContext | None = None) -> EstimatedMPCPolicy:
    root, _ = workspace.resolve(artifact_id, kind="estimator")
    saved = read_json(root / "estimator.json")
    controller_root, _ = workspace.resolve(saved["controller"], kind="controller")
    controller = read_json(controller_root / "controller.json")
    model = load_model(workspace, saved["prediction_model"])
    return EstimatedMPCPolicy(MPCPolicy(model, controller["parameters"], context), saved["estimator_config"])
