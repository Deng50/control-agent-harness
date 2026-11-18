"""Generate the synthetic S02 F01 oracle independently of production/MATLAB code.

SciPy's augmented matrix exponential gives exact zero-order-hold matrices.
Closed-form matrices and an independent scalar RK4 rollout cross-check the
oracle; this script never calls MATLAB, a server, or control_harness modules.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm


def generate() -> dict:
    dt = 0.2
    a = np.array([[0.0, 1.0], [-2.0, -0.8]], dtype=np.float64)
    b = np.array([[0.0], [1.0]], dtype=np.float64)
    c = np.array([[1.0, 0.0]], dtype=np.float64)
    augmented = np.zeros((3, 3), dtype=np.float64)
    augmented[:2, :2], augmented[:2, 2:] = a, b
    discrete = expm(augmented * dt)
    ad, bd = discrete[:2, :2], discrete[:2, 2:]
    x0 = np.array([0.1, -0.05], dtype=np.float64)
    times = np.arange(21, dtype=np.float64) * dt
    inputs = np.array([0.25] * 5 + [-0.30] * 5 + [0.40] * 5 + [0.0] * 6)
    disturbances = np.array([0.05] * 10 + [-0.03] * 11)

    # Analytic underdamped oscillator reference, independent of expm.
    omega = math.sqrt(2.0 - 0.4**2)
    s, co, decay = math.sin(omega * dt), math.cos(omega * dt), math.exp(-0.4 * dt)
    analytic_ad = decay * np.array([[co + 0.4 / omega * s, s / omega],
                                    [-2.0 / omega * s, co - 0.4 / omega * s]])
    analytic_bd = np.linalg.solve(a, (analytic_ad - np.eye(2)) @ b)
    matrix_error = float(max(np.max(np.abs(ad - analytic_ad)), np.max(np.abs(bd - analytic_bd))))
    assert matrix_error < 1e-12

    def rollout(gain: float) -> tuple[dict, float]:
        states = np.empty((21, 2), dtype=np.float64)
        rk4_states = np.empty_like(states)
        states[0] = rk4_states[0] = x0
        for index in range(20):
            force = gain * inputs[index] + disturbances[index]
            states[index + 1] = ad @ states[index] + bd[:, 0] * force
            # Scalar equations, fixed 200 substeps per held-input interval.
            position, velocity = map(float, rk4_states[index])
            h = dt / 200
            def derivative(p: float, v: float) -> tuple[float, float]:
                return v, force - 2.0 * p - 0.8 * v
            for _ in range(200):
                k1p, k1v = derivative(position, velocity)
                k2p, k2v = derivative(position + h * k1p / 2, velocity + h * k1v / 2)
                k3p, k3v = derivative(position + h * k2p / 2, velocity + h * k2v / 2)
                k4p, k4v = derivative(position + h * k3p, velocity + h * k3v)
                position += h * (k1p + 2 * k2p + 2 * k3p + k4p) / 6
                velocity += h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            rk4_states[index + 1] = [position, velocity]
        assert states.shape == (21, 2) and np.isfinite(states).all()
        error = float(np.max(np.abs(states - rk4_states)))
        assert error < 1e-10
        return {"gain": gain, "states": states.tolist(),
                "outputs": (states @ c.T)[:, 0].tolist()}, error

    baseline, baseline_error = rollout(1.0)
    edited, edited_error = rollout(1.25)
    assert np.max(np.abs(np.asarray(baseline["states"]) - edited["states"])) > 1e-5
    return {
        "fixtureVersion": "S02-F01-v1",
        "source": "Synthetic deterministic spring-mass-damper fixture; not measured vehicle data.",
        "generator": {"script": "generate_reference.py", "method": "SciPy augmented-matrix expm exact ZOH",
                      "scipy_version": scipy.__version__, "numpy_version": np.__version__,
                      "production_modules_used": False, "matlab_used": False},
        "semantics": {"states": ["position_m", "velocity_m_per_s"], "input": "force_N",
                      "disturbance": "additive_force_N", "output": "position_m",
                      "update": "x[k+1] = Ad*x[k] + Bd*(gain*u[k] + disturbance[k])",
                      "samples": "21 state/output samples t=0..4; first 20 input/disturbance samples drive intervals; final input sample is unused"},
        "dt": dt, "A": a.tolist(), "B": b.tolist(), "E": b.tolist(),
        "C": c.tolist(), "D": [[0.0]], "Ad": ad.tolist(), "Bd": bd.tolist(),
        "x0": x0.tolist(), "time": times.tolist(), "input": inputs.tolist(),
        "disturbance": disturbances.tolist(), "baseline": baseline, "edited": edited,
        "tolerance": {"atol": 1e-6, "rtol": 1e-5},
        "referenceChecks": {"closed_form_matrix_max_abs_error": matrix_error,
                            "scalar_RK4_substeps_per_interval": 200,
                            "baseline_RK4_state_max_abs_error": baseline_error,
                            "edited_RK4_state_max_abs_error": edited_error},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("f01-reference.json"))
    args = parser.parse_args()
    result = generate()
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "fixtureVersion": result["fixtureVersion"],
                      "samples": len(result["time"]), "checks": result["referenceChecks"]}))
