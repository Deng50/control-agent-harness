"""SISO receding-horizon quadratic controller, with physical and slew constraints."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def control(observation, reference, disturbance, previous, memory, parameters):
    p = parameters
    base = np.asarray(p["state_free"]) @ observation + np.asarray(p["disturbance_free"]) @ disturbance
    output_error = np.asarray(p["outputs"]) @ base - reference[0]
    initial = np.zeros(p["horizon"])
    initial[0] = previous[0]
    difference = np.asarray(p["difference"])
    linear = p["tracking"] * np.asarray(p["output_control"]).T @ output_error - p["increment"] * difference.T @ initial
    sequence = -np.asarray(p["hessian_inverse"]) @ linear
    lower = np.r_[p["state_lower"] - base, initial - p["rate"]]
    upper = np.r_[p["state_upper"] - base, initial + p["rate"]]
    matrix = np.asarray(p["constraint_matrix"])

    def feasible(actions):
        constrained = matrix @ actions
        return (np.all(actions >= -p["limit"] - 1e-9) and np.all(actions <= p["limit"] + 1e-9)
                and np.all(constrained >= lower - 1e-9) and np.all(constrained <= upper + 1e-9))

    if not feasible(sequence):
        hessian = np.asarray(p["hessian"])
        estimate = np.full(p["horizon"], previous[0])
        solved = minimize(lambda a: (0.5 * a @ hessian @ a + linear @ a, hessian @ a + linear),
                          estimate, jac=True, method="SLSQP", bounds=Bounds(-p["limit"], p["limit"]),
                          constraints=[LinearConstraint(matrix, lower, upper)],
                          options={"ftol": 1e-11, "maxiter": 120})
        if not solved.success or not feasible(solved.x):
            raise ValueError("MPC failed to find a feasible finite-horizon solution")
        sequence = solved.x
    return np.asarray([sequence[0]]), memory, "ok"
