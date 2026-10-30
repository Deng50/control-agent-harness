"""Explicit non-polynomial mathematical benchmark; not measured plant evidence.

The runtime passes CPU float64 tensors and the frozen JSON parameters to these
pure functions. This file is inspected statically before explicit registration.
"""
import torch

MODEL_SPEC = {
    "representation": "python_torch_transition",
    "entrypoint": "transition",
    "output_entrypoint": "observe",
    "parameters": {"gains": [0.22, 0.20], "bias": [0.0, 0.0, 0.0]},
    "interface": {
        "schema_version": 1, "name": "standalone-nonpolynomial-benchmark", "sample_time": 0.1,
        "time_semantics": "discrete_state_transition",
        "states": [
            {"name": "x1", "unit": "1", "lower": -2, "upper": 2},
            {"name": "x2", "unit": "1", "lower": -2, "upper": 2},
            {"name": "x3", "unit": "1", "lower": -2, "upper": 2},
        ],
        "controls": [
            {"name": "u1", "unit": "1", "lower": -1.5, "upper": 1.5},
            {"name": "u2", "unit": "1", "lower": -1.5, "upper": 1.5},
        ],
        "outputs": [
            {"name": "y1", "unit": "1", "lower": -2.1, "upper": 2.1},
            {"name": "y2", "unit": "1", "lower": -2.1, "upper": 2.1},
        ],
        "disturbances": [{"name": "d1", "unit": "1", "lower": -0.3, "upper": 0.3}],
    },
}


def transition(x, u, d, parameters):
    gain, bias = parameters["gains"], parameters["bias"]
    return torch.stack((
        .78*x[..., 0] + .05*x[..., 2] + gain[0]*torch.tanh(u[..., 0]) + .025*d[..., 0] + bias[0],
        .76*x[..., 1] + .04*torch.sin(x[..., 0]) + gain[1]*torch.tanh(u[..., 1]) + .02*d[..., 0] + bias[1],
        .70*x[..., 2] + .06*torch.tanh(x[..., 0]*x[..., 1]) + .06*(u[..., 0]+u[..., 1]) + bias[2],
    ), dim=-1)


def observe(x, parameters):
    return torch.stack((x[..., 0] + .02*torch.sin(x[..., 2]), x[..., 1]), dim=-1)
