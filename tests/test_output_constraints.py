import unittest
import numpy as np
import torch

from control_harness.control import MPCPolicy, closed_loop
from control_harness.examples import example
from control_harness.modeling import DeclarativeDynamics

torch.set_num_threads(1)


class OutputConstraintTests(unittest.TestCase):
    def test_mpc_enforces_output_limit_distinct_from_state_bounds(self):
        spec = example("nonlinear")["nominal"]
        spec["C"] = [[2., 0.]]
        spec["interface"]["outputs"][0].update(lower=-.1, upper=.1)
        model = DeclarativeDynamics(spec)
        policy = MPCPolicy(model, {"horizon": 3, "q": 1, "r": .001, "s": 0})
        scenario = {"initial_state": [0., 0.], "reference": [.5], "disturbance": [0.], "steps": 3}
        run = closed_loop(model, policy, scenario)
        self.assertLessEqual(run["metrics"]["max_output_violation"], 1e-6)
        self.assertTrue(run["constraint_checks"]["output"])
        self.assertFalse(run["constraint_checks"]["input_rate"])
        self.assertGreater(run["trajectory"][0]["output"][0], .09)

    def test_physical_output_violation_is_not_hidden_by_valid_state_and_input(self):
        spec = example("nonlinear")["nominal"]
        spec["interface"]["outputs"][0].update(lower=-.01, upper=.01)
        model = DeclarativeDynamics(spec)
        class ConstantPolicy:
            def act(self, *args):
                return np.array([1.]), {}
        run = closed_loop(model, ConstantPolicy(), {"initial_state": [0., 0.], "reference": [0.], "disturbance": [0.], "steps": 2})
        self.assertEqual(run["metrics"]["max_state_violation"], 0)
        self.assertEqual(run["metrics"]["max_input_violation"], 0)
        self.assertGreater(run["metrics"]["max_output_violation"], .1)


if __name__ == "__main__":
    unittest.main()
