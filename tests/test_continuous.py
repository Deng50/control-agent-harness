from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp
import torch

from control_harness.artifacts import Workspace, read_json
from control_harness.contracts import ContractError
from control_harness.control import MPCPolicy, closed_loop
from control_harness.data import simulate_transitions
from control_harness.estimation import StateEstimator
from control_harness.identification import identify_linear_model
from control_harness.model_inspection import inspect_model_source
from control_harness.modeling import (CorrectedDynamics, DeclarativeDynamics, ResidualNetwork,
                                      load_model, register_model, tensor)
from control_harness.residual import train_residual

torch.set_num_threads(1)
TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "continuous-unit-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def variables(prefix, count, unit="1", bound=2):
    return [{"name": f"{prefix}{k}", "unit": unit, "lower": -bound, "upper": bound} for k in range(count)]


def scalar_spec(method="zoh", substeps=1):
    return {"representation": "continuous_polynomial", "time_unit": "s", "derivative_units": ["K/s"],
            "integration": {"method": method, "substeps": substeps, "input_hold": "zero_order_hold"},
            "interface": {"schema_version": 1, "name": "continuous-first-order-test", "sample_time": .2,
                "time_semantics": "continuous_state_derivative", "states": variables("x", 1, "K"),
                "controls": variables("u", 1, "W", 1.5), "outputs": variables("y", 1, "K"), "disturbances": []},
            "A": [[-1.4]], "B": [[.8]], "C": [[1]], "E": [[]], "bias": [0], "terms": []}


def nonlinear_spec():
    return {"representation": "continuous_polynomial", "time_unit": "s", "derivative_units": ["1/s"] * 3,
            "integration": {"method": "rk4", "substeps": 4, "input_hold": "zero_order_hold"},
            "interface": {"schema_version": 1, "name": "continuous-nonlinear-three-state-test", "sample_time": .2,
                "time_semantics": "continuous_state_derivative", "states": variables("x", 3),
                "controls": variables("u", 2, bound=1.5), "outputs": variables("y", 2),
                "disturbances": variables("d", 1)},
            "A": [[-.7, .2, 0], [0, -1.1, .1], [.05, 0, -.5]],
            "B": [[.6, .1], [.2, .7], [.3, -.1]], "E": [[.1], [.2], [.05]],
            "C": [[1, 0, 0], [0, 1, .5]], "bias": [.01, 0, -.01],
            "terms": [{"state_index": 0, "coefficient": -.1, "powers": [3, 0, 0, 0, 0, 0]},
                      {"state_index": 1, "coefficient": .05, "powers": [1, 0, 0, 1, 0, 0]},
                      {"state_index": 2, "coefficient": -.08, "powers": [0, 0, 3, 0, 0, 0]}]}


class ContinuousMathTests(unittest.TestCase):
    def test_exact_zoh_matches_scalar_analytic_solution_and_input_gradient(self):
        model = DeclarativeDynamics(scalar_spec())
        x, u = tensor([.7]).requires_grad_(True), tensor([.4]).requires_grad_(True)
        result = model(x, u)
        decay = np.exp(-1.4 * .2)
        gain = .8 * (1 - decay) / 1.4
        np.testing.assert_allclose(result.detach().numpy(), [.7 * decay + .4 * gain], atol=1e-13)
        gx, gu = torch.autograd.grad(result.sum(), (x, u))
        np.testing.assert_allclose(gx, [decay], atol=1e-13)
        np.testing.assert_allclose(gu, [gain], atol=1e-13)
        np.testing.assert_allclose(model.A.numpy(), [[decay]], atol=1e-13)
        np.testing.assert_allclose(model.derivative(x, u).detach(), [-1.4 * .7 + .8 * .4])
        self.assertEqual(model.interface.time_semantics, "discrete_state_transition")
        self.assertEqual(model.spec["interface"]["time_semantics"], "continuous_state_derivative")

    def test_euler_refinement_and_rk4_match_expected_integration(self):
        exact = DeclarativeDynamics(scalar_spec()).step([.7], [.4])
        coarse = DeclarativeDynamics(scalar_spec("euler", 1)).step([.7], [.4])
        refined = DeclarativeDynamics(scalar_spec("euler", 20)).step([.7], [.4])
        self.assertLess(abs(refined[0] - exact[0]), abs(coarse[0] - exact[0]) / 10)
        rk4 = DeclarativeDynamics(scalar_spec("rk4", 4)).step([.7], [.4])
        np.testing.assert_allclose(rk4, exact, atol=3e-8, rtol=0)

    def test_singular_affine_matrix_zoh_with_multiple_inputs_and_disturbances(self):
        spec = nonlinear_spec()
        spec["A"], spec["terms"] = np.zeros((3, 3)).tolist(), []
        spec["integration"] = {"method": "zoh", "substeps": 1, "input_hold": "zero_order_hold"}
        model = DeclarativeDynamics(spec)
        x, u, d = np.array([.2, -.4, .6]), np.array([.1, .3]), np.array([-.2])
        expected = x + .2 * (np.array(spec["B"]) @ u + np.array(spec["E"]) @ d + spec["bias"])
        np.testing.assert_allclose(model.step(x, u, d), expected, atol=1e-13)

    def test_nonlinear_three_state_two_input_rk4_against_independent_ode_solver(self):
        model = DeclarativeDynamics(nonlinear_spec())
        x, u, d = np.array([.4, -.3, .2]), np.array([.2, -.1]), np.array([.05])
        expected = solve_ivp(lambda _, state: model.derivative(tensor(state), tensor(u), tensor(d)).numpy(),
                             (0, .2), x, rtol=1e-12, atol=1e-13).y[:, -1]
        np.testing.assert_allclose(model.step(x, u, d), expected, rtol=0, atol=1e-8)
        batch_x = tensor([x.tolist(), (x * .5).tolist()]).requires_grad_(True)
        batch_u = tensor([u.tolist(), (u * .5).tolist()]).requires_grad_(True)
        predicted = model(batch_x, batch_u, tensor([[.05], [.03]]))
        self.assertEqual(tuple(predicted.shape), (2, 3))
        gradients = torch.autograd.grad(predicted.square().sum(), (batch_x, batch_u))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(float(gradients[1].abs().sum()), .01)
        self.assertEqual(tuple(model.output(predicted).shape), (2, 2))

    def test_continuous_linear_kf_uses_discrete_transition_matrix(self):
        model = DeclarativeDynamics(scalar_spec())
        observer = StateEstimator(model, {"initial_estimate": [0], "Q": [[.001]], "R": [[.01]], "P0": [[.1]]})
        observer.correct([.4])
        posterior_x, posterior_covariance = observer.x.copy(), observer.P.copy()
        observer.predict([.2], [])
        np.testing.assert_allclose(observer.x, model.step(posterior_x, [.2]), atol=1e-12)
        np.testing.assert_allclose(observer.P, model.A.numpy() @ posterior_covariance @ model.A.numpy().T + [[.001]])

    def test_missing_or_ambiguous_continuous_contracts_are_rejected(self):
        mutations = [lambda s: s.pop("derivative_units"), lambda s: s.update(derivative_units=["K"]),
                     lambda s: s.update(time_unit="ms"), lambda s: s.pop("integration"),
                     lambda s: s["integration"].update(substeps=0),
                     lambda s: s["integration"].update(input_hold="unknown"),
                     lambda s: s["interface"].update(time_semantics="discrete_state_transition")]
        for mutation in mutations:
            spec = scalar_spec()
            mutation(spec)
            with self.subTest(spec=spec), self.assertRaises(ContractError):
                DeclarativeDynamics(spec)
        spec = nonlinear_spec()
        spec["integration"] = {"method": "zoh", "substeps": 1, "input_hold": "zero_order_hold"}
        with self.assertRaises(ContractError):
            DeclarativeDynamics(spec)

    def test_discrete_residual_is_added_after_integration_and_derivative_residual_rejected(self):
        model = DeclarativeDynamics(scalar_spec())
        scalers = {"input_mean": [0, 0], "input_scale": [1, 1], "residual_scale": [1]}
        with self.assertRaisesRegex(ContractError, "derivative"):
            ResidualNetwork(model.interface, {"residual_semantics": "state_derivative"}, scalers)
        residual = ResidualNetwork(model.interface, {"hidden_dims": [4, 4]}, scalers)
        for p in residual.parameters():
            p.data.zero_()
        residual.network.net[-1].bias.data[0] = .05
        corrected = CorrectedDynamics(model, residual)
        np.testing.assert_allclose(corrected.step([.7], [.4]), model.step([.7], [.4]) + .05, atol=1e-13)
        self.assertEqual(corrected.residual_semantics, "next_state_residual_in_state_units")


class ContinuousArtifactChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.workspace = Workspace(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_and_adapter_round_trip_and_inspection(self):
        spec = nonlinear_spec()
        artifact = register_model(self.workspace, spec, {"kind": "user_nominal"})
        root, metadata = self.workspace.resolve(artifact)
        self.assertEqual(read_json(root / "model.json"), spec)
        adapter = read_json(root / "transition_adapter.json")
        self.assertEqual(adapter["method"], "rk4")
        self.assertEqual(adapter["substeps"], 4)
        self.assertEqual(metadata["interface"]["time_semantics"], "discrete_state_transition")
        model = load_model(self.workspace, artifact)
        self.assertEqual(model.transition_adapter, adapter)
        report = inspect_model_source(root / "model.json")
        self.assertTrue(report["registration_ready"])
        self.assertEqual(report["candidates"][0]["source_interface"]["time_semantics"], "continuous_state_derivative")
        self.assertEqual(report["candidates"][0]["transition_adapter"], adapter)

    def test_transition_identification_does_not_integrate_fitted_discrete_coefficients_again(self):
        nominal = register_model(self.workspace, scalar_spec(), {"kind": "user_nominal"})
        reference_spec = scalar_spec()
        reference_spec.update(A=[[-.9]], B=[[1.]])
        reference = register_model(self.workspace, reference_spec, {"kind": "reference_simulator"})
        dataset = simulate_transitions(self.workspace, nominal, reference, {"trajectories": 10, "steps": 12, "seed": 71})
        identified_id = identify_linear_model(self.workspace, nominal, dataset, {})
        root, _ = self.workspace.resolve(identified_id)
        saved = read_json(root / "model.json")
        self.assertEqual(saved["representation"], "discrete_polynomial")
        self.assertNotIn("integration", saved)
        fitted, plant = load_model(self.workspace, identified_id), load_model(self.workspace, reference)
        np.testing.assert_allclose(fitted.step([.6], [.3], []), plant.step([.6], [.3], []), atol=1e-12)

    def test_continuous_model_real_residual_training_and_constrained_mpc_chain(self):
        nominal_spec, reference_spec = scalar_spec(), scalar_spec()
        reference_spec.update(A=[[-.9]], B=[[1.0]])
        nominal = register_model(self.workspace, nominal_spec, {"kind": "user_nominal"})
        reference = register_model(self.workspace, reference_spec, {"kind": "reference_simulator"})
        dataset = simulate_transitions(self.workspace, nominal, reference, {"trajectories": 12, "steps": 16, "seed": 72})
        corrected_id = train_residual(self.workspace, nominal, dataset, {"epochs": 100, "hidden_dims": [12, 12],
            "learning_rate": .01, "batch_size": 64, "patience": 25, "seed": 73, "physics_weight": .01})
        root, metadata = self.workspace.resolve(corrected_id, kind="corrected_model")
        report = read_json(root / "report.json")
        self.assertLess(report["test"]["corrected_mse"], report["test"]["nominal_mse"] * .3)
        self.assertEqual(report["residual_semantics"], "next_state_residual_in_state_units")
        self.assertEqual(report["applicability"]["source"], "training_partition_only")
        self.assertFalse(report["applicability"]["extrapolation_validated"])
        self.assertEqual(report["applicability"]["marginal_ranges"][0]["unit"], "K")
        self.assertEqual(read_json(root / "nominal.json"), nominal_spec)
        model, plant = load_model(self.workspace, corrected_id), load_model(self.workspace, reference)
        control = tensor([.1]).requires_grad_(True)
        derivative, = torch.autograd.grad(model(tensor([.4]), control, tensor([])).sum(), control)
        self.assertGreater(float(derivative), .05)
        result = closed_loop(plant, MPCPolicy(model, {"horizon": 3, "q": 1, "r": .01, "s": .005,
            "input_rate_bounds": [[-.5, .5]]}),
            {"initial_state": [.7], "reference": [.2], "disturbance": [], "steps": 10})
        self.assertEqual(len(result["trajectory"]), 10)
        for key in ("max_state_violation", "max_input_violation", "max_output_violation", "max_rate_violation"):
            self.assertLessEqual(result["metrics"][key], 1e-6)
        self.assertEqual(metadata["interface"]["time_semantics"], "discrete_state_transition")


if __name__ == "__main__":
    unittest.main()
