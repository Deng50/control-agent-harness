"""Explicit continuous-derivative to discrete-transition integration contracts."""
from __future__ import annotations

import copy
import torch

from .contracts import ContractError, ModelInterface


def continuous_contract(spec: dict) -> tuple[ModelInterface, dict]:
    """Validate source derivative units and return a separate discrete runtime interface."""
    raw = spec.get("interface")
    if not isinstance(raw, dict) or raw.get("time_semantics") != "continuous_state_derivative":
        raise ContractError("continuous_polynomial requires source time_semantics=continuous_state_derivative")
    if spec.get("time_unit") != "s":
        raise ContractError("Continuous integration currently requires explicit time_unit='s'; sample_time is in seconds")
    runtime = copy.deepcopy(raw)
    runtime["time_semantics"] = "discrete_state_transition"
    interface = ModelInterface.parse(runtime)
    declared = spec.get("derivative_units")
    if not isinstance(declared, list) or len(declared) != interface.nx:
        raise ContractError("derivative_units must explicitly declare one derivative unit per ordered state")
    normalized = []
    for state, derivative_unit in zip(interface.states, declared):
        unit = state.unit.replace(" ", "")
        allowed = {f"{unit}/s", f"({unit})/s"}
        if unit == "1":
            allowed.add("s^-1")
        if not isinstance(derivative_unit, str) or derivative_unit.replace(" ", "") not in allowed:
            raise ContractError(f"Derivative unit for {state.name} must explicitly be ({state.unit})/s; no implicit unit conversion")
        normalized.append(f"({state.unit})/s")
    integration = spec.get("integration")
    if not isinstance(integration, dict) or set(integration) != {"method", "substeps", "input_hold"}:
        raise ContractError("integration must explicitly contain method, substeps and input_hold")
    method, substeps = integration["method"], integration["substeps"]
    if method not in {"euler", "rk4", "zoh"}:
        raise ContractError("Supported continuous integrators: euler, rk4, zoh")
    if type(substeps) is not int or not 1 <= substeps <= 1024:
        raise ContractError("integration.substeps must be an integer between 1 and 1024")
    if integration["input_hold"] != "zero_order_hold":
        raise ContractError("Controls and disturbances must explicitly use zero_order_hold over the sample")
    if method == "zoh" and (spec.get("terms") or substeps != 1):
        raise ContractError("Exact zoh requires an affine derivative model without nonlinear terms and substeps=1")
    adapter = {"schema_version": 1, "source_representation": "continuous_polynomial",
        "source_time_semantics": "continuous_state_derivative", "runtime_time_semantics": "discrete_state_transition",
        "sample_time": interface.sample_time, "time_unit": "s", "method": method, "substeps": substeps,
        "substep_time": interface.sample_time / substeps, "input_hold": "zero_order_hold",
        "disturbance_hold": "zero_order_hold", "state_order": [state.name for state in interface.states],
        "state_units": [state.unit for state in interface.states], "declared_derivative_units": declared,
        "derivative_units": normalized,
        "output_semantics": "y(k)=C*x(k); after transition use y(k+1)=C*x(k+1)",
        "correction_semantics": "next_state_residual_in_state_units_after_full_integration",
        "declaration_sources": {"sample_time": "interface.sample_time", "method": "integration.method",
            "substeps": "integration.substeps", "derivative_units": "derivative_units"},
        "coefficient_dimensions": "assumed_consistent_with_declared_derivative_equations_not_symbolically_proved",
        "accuracy": "exact_affine_held_input_map" if method == "zoh" else "fixed_step_integration_no_error_tolerance_guarantee"}
    return interface, adapter


def affine_discrete_matrices(A, B, E, bias, adapter: dict):
    """Compile held-input affine dynamics without inverting A (including singular A)."""
    nx, nu, nd = A.shape[0], B.shape[1], E.shape[1]
    generator = A.new_zeros((nx + nu + nd + 1, nx + nu + nd + 1))
    generator[:nx, :nx] = A
    generator[:nx, nx:nx + nu] = B
    generator[:nx, nx + nu:nx + nu + nd] = E
    generator[:nx, -1] = bias
    method = adapter["method"]
    if method == "zoh":
        transition = torch.matrix_exp(generator * adapter["sample_time"])
    else:
        step = generator * adapter["substep_time"]
        identity = torch.eye(len(generator), dtype=A.dtype, device=A.device)
        if method == "euler":
            transition = identity + step
        else:
            step2 = step @ step
            step3 = step2 @ step
            step4 = step3 @ step
            transition = identity + step + step2 / 2 + step3 / 6 + step4 / 24
        transition = torch.linalg.matrix_power(transition, adapter["substeps"])
    if not torch.isfinite(transition).all():
        raise ContractError("Continuous affine integration produced non-finite discrete matrices")
    return (transition[:nx, :nx], transition[:nx, nx:nx + nu],
            transition[:nx, nx + nu:nx + nu + nd], transition[:nx, -1])


def integrate_polynomial(derivative, x, u, d, adapter: dict):
    """Differentiable fixed-step Euler/RK4; u and d are held through all substeps."""
    step = adapter["substep_time"]
    for _ in range(adapter["substeps"]):
        if adapter["method"] == "euler":
            x = x + step * derivative(x, u, d)
        elif adapter["method"] == "rk4":
            k1 = derivative(x, u, d)
            k2 = derivative(x + .5 * step * k1, u, d)
            k3 = derivative(x + .5 * step * k2, u, d)
            k4 = derivative(x + step * k3, u, d)
            x = x + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ContractError("A nonlinear derivative model requires Euler or RK4")
    return x
