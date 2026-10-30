"""Differentiable discrete models; corrected predictions are used by every consumer."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, ModelInterface
from .continuous import affine_discrete_matrices, continuous_contract, integrate_polynomial
from .reused.perception import PerceptionNet

DTYPE = torch.float64


def tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE)


def matrix(value, shape: tuple[int, ...], name: str) -> torch.Tensor:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return tensor(array)


class DeclarativeDynamics(nn.Module):
    """Discrete polynomial map or explicitly integrated continuous polynomial derivative.

    forward/step always return next states. derivative is a separate, explicitly
    named API; source coefficients and adapter provenance remain available.
    """
    def __init__(self, spec: dict):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.representation = spec.get("representation")
        self.transition_adapter = None
        if self.representation == "continuous_polynomial":
            self.interface, self.transition_adapter = continuous_contract(spec)
        elif self.representation == "discrete_polynomial":
            self.interface = ModelInterface.parse(spec["interface"])
            if any(key in spec for key in ("integration", "derivative_units", "time_unit")):
                raise ContractError("Discrete models cannot carry a continuous integration declaration")
        else:
            raise ContractError("Supported representations: discrete_polynomial or explicit continuous_polynomial")
        i = self.interface
        for key, value, shape in (
            ("A", spec["A"], (i.nx, i.nx)), ("B", spec["B"], (i.nx, i.nu)),
            ("E", spec.get("E", np.zeros((i.nx, i.nd))), (i.nx, i.nd)),
            ("C", spec["C"], (i.ny, i.nx)),
            ("bias", spec.get("bias", np.zeros(i.nx)), (i.nx,)),
        ):
            self.register_buffer(key, matrix(value, shape, key))
        self.terms = spec.get("terms", [])
        for term in self.terms:
            if not isinstance(term["state_index"], int) or not 0 <= term["state_index"] < i.nx:
                raise ContractError("Invalid polynomial output state index")
            powers = term["powers"]
            if len(powers) != i.nx + i.nu + i.nd or any(type(v) is not int or v < 0 for v in powers):
                raise ContractError("Polynomial powers must be nonnegative integers over [x,u,d]")
            if not np.isfinite(float(term["coefficient"])):
                raise ContractError("Polynomial coefficient must be finite")
        if self.transition_adapter is not None:
            for name in ("A", "B", "E", "bias"):
                self.register_buffer(f"derivative_{name}", getattr(self, name).clone())
            if not self.terms:
                # Linear consumers such as KF must see discrete A/B/E/bias,
                # never mistake continuous-time coefficients for a step map.
                discrete = affine_discrete_matrices(self.A, self.B, self.E, self.bias, self.transition_adapter)
                self.A, self.B, self.E, self.bias = discrete

    def forward(self, x: torch.Tensor, u: torch.Tensor, d: torch.Tensor | None = None) -> torch.Tensor:
        d = self._disturbances(x, d)
        if self.transition_adapter is not None and self.terms:
            return integrate_polynomial(self.derivative, x, u, d, self.transition_adapter)
        return self._polynomial(x, u, d, self.A, self.B, self.E, self.bias)

    def _disturbances(self, x, d):
        if d is None:
            if self.interface.nd:
                raise ContractError("Disturbances must be provided explicitly")
            d = x.new_empty((*x.shape[:-1], 0))
        return d

    def derivative(self, x: torch.Tensor, u: torch.Tensor, d: torch.Tensor | None = None) -> torch.Tensor:
        if self.transition_adapter is None:
            raise ContractError("A discrete next-state model does not expose a continuous derivative")
        return self._polynomial(x, u, self._disturbances(x, d), self.derivative_A,
                                self.derivative_B, self.derivative_E, self.derivative_bias)

    def _polynomial(self, x, u, d, A, B, E, bias):
        result = x @ A.T + u @ B.T + d @ E.T + bias
        features = torch.cat((x, u, d), dim=-1)
        for term in self.terms:
            powers = features.new_tensor(term["powers"])
            value = torch.prod(features ** powers, dim=-1) * term["coefficient"]
            basis = result.new_zeros(self.interface.nx)
            basis[term["state_index"]] = 1.0
            result = result + value.unsqueeze(-1) * basis
        return result

    def output(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.C.T

    def step(self, x, u, d=None) -> np.ndarray:
        with torch.no_grad():
            result = self(tensor(x), tensor(u), None if d is None else tensor(d))
        value = result.detach().cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError("Dynamics returned non-finite state")
        return value


class ResidualNetwork(nn.Module):
    """Scaled version of AIagent PerceptionNet, with configurable depth and disturbances."""
    def __init__(self, interface: ModelInterface, config: dict, scalers: dict):
        super().__init__()
        self.interface, self.config = interface, copy.deepcopy(config)
        self.residual_semantics = config.get("residual_semantics", "next_state_residual_in_state_units")
        if self.residual_semantics != "next_state_residual_in_state_units":
            raise ContractError("ResidualNetwork supports only discrete next-state residuals in state units, not derivative residuals")
        if interface.time_semantics != "discrete_state_transition":
            raise ContractError("Residual networks require an explicitly discrete runtime interface")
        hidden = config.get("hidden_dims", [64, 64])
        if not hidden or any(type(h) is not int or h <= 0 for h in hidden):
            raise ContractError("hidden_dims must contain positive integers")
        self.network = PerceptionNet(interface.nx, interface.nu + interface.nd, hidden=hidden[0])
        if hidden != [hidden[0], hidden[0]]:
            layers = []
            n = interface.nx + interface.nu + interface.nd
            for h in hidden:
                layers.extend([nn.Linear(n, h), nn.Tanh()])
                n = h
            layers.append(nn.Linear(n, interface.nx))
            self.network.net = nn.Sequential(*layers)
        for key in ("input_mean", "input_scale", "residual_scale"):
            self.register_buffer(key, tensor(scalers[key]))
        self.to(dtype=DTYPE)

    def forward(self, x, u, d=None):
        if d is None:
            d = x.new_empty((*x.shape[:-1], 0))
        features = (torch.cat((x, u, d), dim=-1) - self.input_mean) / self.input_scale
        return self.network(features[..., :self.interface.nx], features[..., self.interface.nx:]) * self.residual_scale


class CorrectedDynamics(nn.Module):
    def __init__(self, nominal: nn.Module, residual: ResidualNetwork):
        super().__init__()
        nominal.interface.compatible(residual.interface)
        if residual.residual_semantics != "next_state_residual_in_state_units":
            raise ContractError("CorrectedDynamics cannot add a derivative residual to a discrete next state")
        self.nominal, self.residual = nominal, residual
        self.interface = nominal.interface
        self.residual_semantics = residual.residual_semantics
        self.transition_adapter = copy.deepcopy(nominal.transition_adapter)

    def forward(self, x, u, d=None):
        return self.nominal(x, u, d) + self.residual(x, u, d)

    def output(self, x):
        return self.nominal.output(x)

    def step(self, x, u, d=None):
        with torch.no_grad():
            value = self(tensor(x), tensor(u), None if d is None else tensor(d)).cpu().numpy()
        if not np.isfinite(value).all():
            raise FloatingPointError("Corrected dynamics returned non-finite state")
        return value


def describe_execution(model) -> dict:
    """Describe the executable source rather than assuming a declarative plant."""
    if isinstance(model, CorrectedDynamics):
        return {"kind": "nominal_plus_learned_residual", "residual_semantics": model.residual_semantics,
                "nominal": describe_execution(model.nominal)}
    return {"source_representation": model.spec["representation"],
            "runtime_time_semantics": model.interface.time_semantics,
            "transition_adapter": model.transition_adapter,
            "frozen_python_source": model.spec["representation"].startswith("python_torch_")}


def register_model(workspace: Workspace, spec: dict, source: dict) -> str:
    model = DeclarativeDynamics(spec)
    if source.get("kind") not in {"user_nominal", "reference_simulator", "reconstructed_equations"}:
        raise ContractError("Model source must identify user_nominal, reference_simulator or reconstructed_equations")
    kind = "reference_model" if source["kind"] == "reference_simulator" else "nominal_model"
    with workspace.stage(kind, model.interface, [], {"source": source}) as stage:
        atomic_json(stage.root / "model.json", spec)
        if model.transition_adapter is not None:
            atomic_json(stage.root / "transition_adapter.json", model.transition_adapter)
        return stage.publish({"status": "verified", "checks": ["schema", "dimensions", "finite_parameters"],
                              "physical_fidelity": "user_supplied_not_independently_certified",
                              "runtime_time_semantics": "discrete_state_transition",
                              "continuous_adapter": model.transition_adapter},
                             source={**source, "source_representation": model.representation,
                                     "transition_adapter": model.transition_adapter})


def load_model(workspace: Workspace, artifact_id: str):
    root, meta = workspace.resolve(artifact_id)
    if meta["kind"] in {"nominal_model", "reference_model", "identified_model"}:
        spec = read_json(root / "model.json")
        if spec.get("representation") in {"python_torch_transition", "python_torch_derivative"}:
            from .python_model import load_python_model
            model = load_python_model(root, spec)
        else:
            model = DeclarativeDynamics(spec)
        model.interface.compatible(ModelInterface.parse(meta["interface"]))
        if model.transition_adapter is not None:
            if not (root / "transition_adapter.json").is_file():
                raise ContractError("Continuous model artifact lacks its explicit transition adapter declaration")
            if read_json(root / "transition_adapter.json") != model.transition_adapter:
                raise ContractError("Saved continuous transition adapter disagrees with the original derivative model")
        return model
    if meta["kind"] != "corrected_model":
        raise ContractError("Artifact is not an executable dynamics model")
    nominal_id = meta.get("source", {}).get("nominal_model", meta["parents"][0]["id"])
    if nominal_id not in [parent["id"] for parent in meta["parents"]]:
        raise ContractError("Corrected model nominal source is not a bound parent artifact")
    nominal = load_model(workspace, nominal_id)
    if not hasattr(nominal, "spec") or nominal.spec != read_json(root / "nominal.json"):
        raise ContractError("Corrected model nominal snapshot differs from its bound executable parent")
    nominal.interface.compatible(ModelInterface.parse(meta["interface"]))
    config = read_json(root / "network.json")
    residual = ResidualNetwork(nominal.interface, config, read_json(root / "scalers.json"))
    checkpoint = torch.load(root / "model.pt", weights_only=True, map_location="cpu")
    residual.load_state_dict(checkpoint["network_state"])
    residual.eval()
    for parameter in residual.parameters():
        parameter.requires_grad_(False)
    return CorrectedDynamics(nominal, residual)
