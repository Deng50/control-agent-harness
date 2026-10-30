"""Explicit demonstration simulators; these are not measured engineering datasets."""
from __future__ import annotations
import copy
import numpy as np


def example(case: str) -> dict:
    if case == "linear":
        nx, nu, ny, nd = 4, 2, 2, 1
        a = [[.82, .04, 0, 0], [0, .78, .03, 0], [.01, 0, .72, .02], [0, .02, 0, .74]]
        b = [[.22, 0], [0, .20], [.09, .03], [.02, .10]]
        c = [[1, 0, 0, 0], [0, 1, 0, 0]]
        real_a = np.array(a) + np.diag([.055, .04, .03, .045])
        terms = []
    elif case == "nonlinear":
        nx, nu, ny, nd = 2, 1, 1, 1
        a = [[.86, .04], [.03, .78]]
        b, c = [[.22], [.10]], [[1, 0]]
        real_a = np.array(a) + np.diag([.04, .03])
        terms = [{"state_index": 0, "coefficient": -.025, "powers": [3, 0, 0, 0]},
                 {"state_index": 1, "coefficient": -.018, "powers": [0, 3, 0, 0]}]
    else:
        raise ValueError("case must be linear or nonlinear")

    def variables(prefix, count, bound):
        return [{"name": f"{prefix}{k + 1}", "unit": "1", "lower": -bound, "upper": bound} for k in range(count)]
    interface = {"schema_version": 1, "name": f"{case}_demonstration", "sample_time": .1,
                 "time_semantics": "discrete_state_transition", "states": variables("x", nx, 2),
                 "controls": variables("u", nu, 1.5), "outputs": variables("y", ny, 2),
                 "disturbances": variables("d", nd, .3)}
    nominal = {"interface": interface, "representation": "discrete_polynomial", "A": a, "B": b,
               "E": [[.025] for _ in range(nx)], "C": c, "terms": []}
    reference = copy.deepcopy(nominal)
    reference.update(A=real_a.tolist(), terms=terms)

    def scenario(seed, steps=12):
        rng = np.random.default_rng(seed)
        return {"name": f"condition-{seed}", "initial_state": rng.uniform(-.65, .65, nx).tolist(),
                "reference": rng.uniform(-.4, .4, ny).tolist(), "disturbance": rng.uniform(-.2, .2, nd).tolist(),
                "steps": steps}

    return {"case": case, "nominal": nominal, "reference": reference,
            "data": {"trajectories": 20, "steps": 24, "seed": 41},
            "residual": {"epochs": 130, "hidden_dims": [24, 24], "learning_rate": .006, "batch_size": 128,
                         "patience": 30, "physics_weight": .1, "seed": 42},
            "controller": {"parameters": {"horizon": 4, "q": 1, "r": .015, "s": .01},
                "parameter_space": {"r": {"lower": .0001, "upper": .04}, "s": {"lower": .0001, "upper": .04}},
                "budget": 8, "seed": 43, "tuning_scenarios": [scenario(101, 9)],
                "verification_scenarios": [scenario(201, 10)], "test_scenarios": [scenario(301, 11)]},
            "collection": {"scenarios": [scenario(400 + k, 12) for k in range(12)], "seed": 44},
            "imitation": {"architecture": "mlp", "epochs": 160, "hidden_dims": [48, 32],
                          "learning_rate": .004, "patience": 35, "batch_size": 64, "seed": 45},
            "optimization": {"epochs": 20, "learning_rate": .0005, "horizon": 4, "seed": 46,
                             "imitation_weight": .2}}
