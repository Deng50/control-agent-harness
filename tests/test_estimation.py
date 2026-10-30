from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.contracts import ContractError
from control_harness.control import MPCPolicy
from control_harness.estimation import (EstimatedMPCPolicy, StateEstimator, estimator_closed_loop,
                                        load_estimator_controller, validate_estimator)
from control_harness.modeling import DeclarativeDynamics, register_model
from control_harness.runtime import Cancelled, OperationContext

torch.set_num_threads(1)
TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "estimation-unit-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def specification(nonlinear=False, observable=True):
    def variables(prefix, n, bound):
        return [{"name": f"{prefix}{k}", "unit": "1", "lower": -bound, "upper": bound} for k in range(n)]
    return {"interface": {"schema_version": 1, "name": "observer-test", "sample_time": .1,
            "time_semantics": "discrete_state_transition", "states": variables("x", 2, 3),
            "controls": variables("u", 1, 2), "outputs": variables("y", 1, 3), "disturbances": []},
            "representation": "discrete_polynomial", "A": [[.88, .2 if observable else 0], [0, .82]],
            "B": [[.12], [.18]], "C": [[1, 0]],
            "terms": [{"state_index": 0, "coefficient": -.025, "powers": [3, 0, 0]}] if nonlinear else []}


def configuration():
    return {"initial_estimate": [-.4, .3], "Q": [[1e-5, 0], [0, 1e-5]], "R": [[.0001]],
            "P0": [[.5, 0], [0, .5]], "seed": 63, "sensor_noise_covariance": [[.0001]]}


def scenario(initial=None):
    return {"name": "verification", "initial_state": [.7, -.3] if initial is None else initial,
            "reference": [.1], "disturbance": [], "steps": 18}


PARAMETERS = {"horizon": 3, "q": 1, "r": .02, "s": .01, "input_rate_bounds": [[-.5, .5]]}


class EstimatorMathTests(unittest.TestCase):
    def test_joseph_update_matches_linear_conditional_gaussian(self):
        model = DeclarativeDynamics(specification())
        config = configuration()
        estimator = StateEstimator(model, config)
        prior, covariance = estimator.x.copy(), estimator.P.copy()
        C = model.C.numpy()
        measurement = np.array([.6])
        S = C @ covariance @ C.T + estimator.R
        gain = np.linalg.solve(S, C @ covariance).T
        expected_state = prior + gain @ (measurement - C @ prior)
        IKH = np.eye(2) - gain @ C
        expected_cov = IKH @ covariance @ IKH.T + gain @ estimator.R @ gain.T
        estimator.correct(measurement)
        np.testing.assert_allclose(estimator.x, expected_state, atol=1e-14)
        np.testing.assert_allclose(estimator.P, expected_cov, atol=1e-14)
        estimator.predict([.2], [])
        np.testing.assert_allclose(estimator.x, model.step(expected_state, [.2], []), atol=1e-14)
        np.testing.assert_allclose(estimator.P, model.A.numpy() @ expected_cov @ model.A.numpy().T + estimator.Q)

    def test_invalid_dimensions_and_covariance_are_rejected(self):
        model = DeclarativeDynamics(specification())
        for mutation in ({"Q": [[-1, 0], [0, 1]]}, {"R": [[0]]}, {"P0": [[1, .1], [0, 1]]},
                         {"initial_estimate": [0]}, {"measurement_output_indices": [0, 0]},
                         {"measurement_output_indices": [True]}):
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                StateEstimator(model, {**configuration(), **mutation})
        estimator = StateEstimator(model, configuration())
        with self.assertRaises(ContractError):
            estimator.correct([1, 2])
        with self.assertRaises(ContractError):
            estimator.predict([0], [])

    def test_unobservable_linear_state_and_nonlinear_kf_are_rejected(self):
        with self.assertRaisesRegex(ContractError, "observe"):
            StateEstimator(DeclarativeDynamics(specification(observable=False)), configuration())
        with self.assertRaisesRegex(ContractError, "KF requires"):
            StateEstimator(DeclarativeDynamics(specification(nonlinear=True)), {**configuration(), "method": "KF"})

    def test_controller_action_uses_posterior_estimate(self):
        model = DeclarativeDynamics(specification())
        config = {**configuration(), "initial_estimate": [-1.0, -.8], "P0": [[1e-9, 0], [0, 1e-9]], "R": [[10.0]]}
        policy = EstimatedMPCPolicy(MPCPolicy(model, PARAMETERS), config)
        action, diagnostics = policy.act_measurement([.7], [.1], [])
        posterior = diagnostics["estimator"]["state_estimate"]
        expected, _ = MPCPolicy(DeclarativeDynamics(specification()), PARAMETERS).act(posterior, [.1], [], [0])
        omniscient, _ = MPCPolicy(DeclarativeDynamics(specification()), PARAMETERS).act([.7, -.3], [.1], [], [0])
        np.testing.assert_allclose(action, expected, atol=1e-10)
        self.assertGreater(float(np.linalg.norm(action - omniscient)), .2)
        self.assertEqual(diagnostics["controller_state_source"], "posterior_state_estimate")

    def test_real_linear_and_nonlinear_mpc_measurement_feedback(self):
        for nonlinear in (False, True):
            with self.subTest(nonlinear=nonlinear):
                plant = DeclarativeDynamics(specification(nonlinear))
                result = estimator_closed_loop(plant, DeclarativeDynamics(specification(nonlinear)),
                                               PARAMETERS, scenario(), configuration())
                estimation = result["estimation"]
                self.assertEqual(estimation["method"], "EKF" if nonlinear else "KF")
                self.assertTrue(estimation["observability"]["passed"])
                self.assertLess(estimation["metrics"]["estimation_rmse"],
                                estimation["metrics"]["open_loop_estimation_rmse"] * .5)
                self.assertGreaterEqual(estimation["metrics"]["minimum_covariance_eigenvalue"], -1e-12)
                for key in ("max_state_violation", "max_input_violation", "max_output_violation", "max_rate_violation"):
                    self.assertLessEqual(result["metrics"][key], 1e-6)
                self.assertEqual(len(result["trajectory"]), scenario()["steps"])

    def test_sensor_seed_is_reproducible_and_changes_measurement(self):
        def run(seed):
            return estimator_closed_loop(DeclarativeDynamics(specification()), DeclarativeDynamics(specification()),
                PARAMETERS, {**scenario(), "steps": 4}, {**configuration(), "seed": seed})["estimation"]["trajectory"]
        first, repeated, changed = run(17), run(17), run(18)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first[0]["measurement"], changed[0]["measurement"])

    def test_cancellation_reaches_estimated_control(self):
        with self.assertRaises(Cancelled):
            estimator_closed_loop(DeclarativeDynamics(specification()), DeclarativeDynamics(specification()),
                                  PARAMETERS, scenario(), configuration(), OperationContext(cancelled=lambda: True))


class EstimatorArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.workspace = Workspace(self.temp.name)
        self.spec = specification()
        self.nominal = register_model(self.workspace, self.spec, {"kind": "user_nominal"})
        self.reference = register_model(self.workspace, self.spec, {"kind": "reference_simulator"})
        model = DeclarativeDynamics(self.spec)
        with self.workspace.stage("controller", model.interface, [self.nominal, self.reference], {}) as stage:
            atomic_json(stage.root / "controller.json", {"prediction_model": self.nominal,
                "evaluation_model": self.reference, "parameters": PARAMETERS})
            self.controller = stage.publish({"status": "verified", "fixture": "numerically tested fixed MPC parameters"})
        self.config = {**configuration(), "verification_scenarios": [scenario()],
            "test_scenarios": [{**scenario([-.4, .5]), "name": "held-out"}], "max_normalized_estimation_rmse": .1}

    def tearDown(self):
        self.temp.cleanup()

    def test_publish_reload_and_dependency_binding(self):
        artifact = validate_estimator(self.workspace, self.nominal, self.controller, self.reference, self.config)
        root, metadata = self.workspace.resolve(artifact, kind="estimator")
        self.assertEqual({p["id"] for p in metadata["parents"]}, {self.nominal, self.controller, self.reference})
        self.assertEqual(metadata["validation"]["status"], "verified")
        report = read_json(root / "report.json")
        self.assertFalse(report["test_used_for_tuning"])
        self.assertTrue(all(r["acceptance"]["passed"] for r in report["test_scenarios"]))
        policy = load_estimator_controller(self.workspace, artifact)
        action, diagnostics = policy.act_measurement([.4], [.1], [])
        self.assertEqual(action.shape, (1,))
        self.assertEqual(diagnostics["controller_state_source"], "posterior_state_estimate")

    def test_renamed_duplicate_test_scenario_cannot_pass(self):
        bad = copy.deepcopy(self.config)
        bad["test_scenarios"] = [{**bad["verification_scenarios"][0], "name": "renamed"}]
        with self.assertRaisesRegex(ContractError, "overlap"):
            validate_estimator(self.workspace, self.nominal, self.controller, self.reference, bad)

    def test_failed_error_gate_is_rejected_and_not_loadable(self):
        artifact = validate_estimator(self.workspace, self.nominal, self.controller, self.reference,
                                     {**self.config, "max_normalized_estimation_rmse": 1e-12})
        self.assertEqual(self.workspace.manifest["artifacts"][artifact]["status"], "rejected")
        self.assertNotIn("estimator", self.workspace.manifest["active"])
        with self.assertRaises(ContractError):
            load_estimator_controller(self.workspace, artifact)

    def test_actual_output_violation_rejects_otherwise_finite_estimator_run(self):
        # The evaluation plant has an unmodelled output gain. State/input bounds
        # remain satisfied, but true outputs violate the declared output bounds.
        changed = copy.deepcopy(self.spec)
        changed["C"] = [[10, 0]]
        reference = register_model(self.workspace, changed, {"kind": "reference_simulator"})
        with self.workspace.stage("controller", DeclarativeDynamics(self.spec).interface,
                                  [self.nominal, reference], {}) as stage:
            atomic_json(stage.root / "controller.json", {"prediction_model": self.nominal,
                "evaluation_model": reference, "parameters": PARAMETERS})
            controller = stage.publish({"status": "verified"})
        config = {**self.config, "initial_estimate": [0, 0], "P0": [[0, 0], [0, 0]],
                  "Q": [[0, 0], [0, 0]], "R": [[1]], "max_normalized_estimation_rmse": .5}
        artifact = validate_estimator(self.workspace, self.nominal, controller, reference, config)
        row = self.workspace.manifest["artifacts"][artifact]
        self.assertEqual(row["status"], "rejected")
        report = read_json(self.workspace.path / row["path"] / "report.json")
        run = report["verification_scenarios"][0]
        self.assertGreater(run["metrics"]["max_output_violation"], 0)
        self.assertLessEqual(run["metrics"]["max_state_violation"], 1e-6)
        self.assertLessEqual(run["metrics"]["max_input_violation"], 1e-6)
        self.assertFalse(run["acceptance"]["checks"]["closed_loop_hard_constraints"])


if __name__ == "__main__":
    unittest.main()
