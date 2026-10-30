from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from control_harness.artifacts import Workspace, read_json
from control_harness.contracts import ContractError
from control_harness.control import MPCPolicy, closed_loop
from control_harness.data import register_transitions, trajectory_split
from control_harness.identification import fit_affine_transitions, identify_linear_model, load_identified_model
from control_harness.modeling import DeclarativeDynamics, register_model
from control_harness.runtime import Cancelled, OperationContext

torch.set_num_threads(1)
TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "identification-unit-temp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def specification():
    def variables(prefix, n, bound):
        return [{"name": f"{prefix}{k}", "unit": "1", "lower": -bound, "upper": bound} for k in range(n)]
    return {"interface": {"schema_version": 1, "name": "identification-test", "sample_time": .1,
            "time_semantics": "discrete_state_transition", "states": variables("x", 2, 3),
            "controls": variables("u", 1, 2), "outputs": variables("y", 1, 3),
            "disturbances": variables("d", 1, 1)},
            "representation": "discrete_polynomial", "A": [[.76, .2], [.04, .82]],
            "B": [[.12], [.18]], "E": [[.1], [.03]], "C": [[1, 0]], "bias": [.01, -.02], "terms": []}


def dataset():
    plant = DeclarativeDynamics(specification())
    rng = np.random.default_rng(723)
    rows = {key: [] for key in ("x", "u", "d", "next_x", "trajectory_id")}
    for group in range(10):
        x = rng.uniform(-1, 1, 2)
        for _ in range(24):
            u, d = rng.uniform(-.8, .8, 1), rng.uniform(-.4, .4, 1)
            next_x = plant.step(x, u, d)
            for key, value in (("x", x), ("u", u), ("d", d), ("next_x", next_x), ("trajectory_id", group)):
                rows[key].append(value)
            x = next_x
    arrays = {key: np.asarray(value, dtype=int if key == "trajectory_id" else float) for key, value in rows.items()}
    split = trajectory_split(arrays["trajectory_id"], 41)
    masks = {name: np.isin(arrays["trajectory_id"], groups) for name, groups in split.items()}
    return arrays, masks


class IdentificationMathTests(unittest.TestCase):
    def test_affine_dynamics_recovered_with_disturbance_and_bias(self):
        spec = specification()
        arrays, masks = dataset()
        coefficients, report = fit_affine_transitions(DeclarativeDynamics(spec).interface, arrays, masks, {})
        for name in ("A", "B", "E", "bias"):
            np.testing.assert_allclose(coefficients[name], spec[name], atol=1e-12)
        self.assertEqual(report["training_rank"], 5)
        self.assertFalse(report["validation_used_for_fitting"])
        self.assertFalse(report["test_used_for_fitting"])
        self.assertLess(report["metrics"]["test"]["normalized_rmse"], 1e-12)

    def test_heldout_data_cannot_change_fitted_coefficients_or_scalers(self):
        arrays, masks = dataset()
        interface = DeclarativeDynamics(specification()).interface
        original, report = fit_affine_transitions(interface, arrays, masks, {})
        mutated = {key: value.copy() for key, value in arrays.items()}
        holdout = masks["validation"] | masks["test"]
        mutated["x"][holdout] += 100
        mutated["next_x"][holdout] -= 1000
        changed, changed_report = fit_affine_transitions(interface, mutated, masks, {})
        self.assertEqual(original, changed)
        self.assertEqual(report["preprocessing"], changed_report["preprocessing"])
        self.assertGreater(changed_report["metrics"]["test"]["normalized_rmse"], 100)

    def test_insufficient_excitation_and_collinear_training_regressors_fail(self):
        arrays, masks = dataset()
        interface = DeclarativeDynamics(specification()).interface
        arrays["u"][masks["train"]] = arrays["x"][masks["train"], :1]
        with self.assertRaisesRegex(ContractError, "not identifiable"):
            fit_affine_transitions(interface, arrays, masks, {})
        arrays["u"][masks["train"]] = 0
        with self.assertRaisesRegex(ContractError, "excitation"):
            fit_affine_transitions(interface, arrays, masks, {})

    def test_overlap_missing_rows_and_shared_trajectory_are_rejected(self):
        arrays, masks = dataset()
        interface = DeclarativeDynamics(specification()).interface
        overlap = {key: value.copy() for key, value in masks.items()}
        overlap["test"] |= overlap["train"]
        with self.assertRaises(ContractError):
            fit_affine_transitions(interface, arrays, overlap, {})
        missing = {key: value.copy() for key, value in masks.items()}
        row = int(np.flatnonzero(missing["train"])[0])
        missing["train"][row] = False
        with self.assertRaises(ContractError):
            fit_affine_transitions(interface, arrays, missing, {})
        missing["test"][row] = True
        with self.assertRaisesRegex(ContractError, "trajectory"):
            fit_affine_transitions(interface, arrays, missing, {})

    def test_cancellation_is_observed_before_fit(self):
        arrays, masks = dataset()
        with self.assertRaises(Cancelled):
            fit_affine_transitions(DeclarativeDynamics(specification()).interface, arrays, masks, {},
                                   OperationContext(cancelled=lambda: True))


class IdentificationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        self.workspace = Workspace(self.temp.name)
        self.spec = specification()
        nominal = copy.deepcopy(self.spec)
        nominal["A"][0][0] -= .1
        self.nominal = register_model(self.workspace, nominal, {"kind": "user_nominal"})
        self.reference = register_model(self.workspace, self.spec, {"kind": "reference_simulator"})
        self.arrays, _ = dataset()

    def tearDown(self):
        self.temp.cleanup()

    def register_data(self, arrays):
        return register_transitions(self.workspace, self.nominal, arrays,
            {"kind": "reference_simulator", "description": "Seeded synthetic identification test transitions"},
            seed=41, reference_id=self.reference)

    def test_published_model_drives_real_mpc_and_preserves_provenance(self):
        data = self.register_data(self.arrays)
        artifact = identify_linear_model(self.workspace, self.nominal, data, {})
        root, metadata = self.workspace.resolve(artifact, kind="identified_model")
        self.assertEqual({p["id"] for p in metadata["parents"]}, {self.nominal, data})
        self.assertEqual(self.workspace.active("nominal_model"), self.nominal)
        self.assertEqual(metadata["source"]["data_source"]["kind"], "reference_simulator")
        report = read_json(root / "report.json")
        self.assertTrue(report["acceptance"]["passed"])
        model = load_identified_model(self.workspace, artifact)
        result = closed_loop(DeclarativeDynamics(self.spec), MPCPolicy(model, {"horizon": 3, "q": 1, "r": .02, "s": .01}),
            {"initial_state": [.5, -.3], "reference": [.1], "disturbance": [.1], "steps": 6})
        self.assertEqual(len(result["trajectory"]), 6)
        self.assertEqual(result["metrics"]["max_state_violation"], 0)
        self.assertEqual(result["metrics"]["max_input_violation"], 0)

    def test_bad_heldout_prediction_is_rejected_without_nominal_replacement(self):
        masks = {name: np.isin(self.arrays["trajectory_id"], groups)
                 for name, groups in trajectory_split(self.arrays["trajectory_id"], 41).items()}
        self.arrays["next_x"][masks["test"]] += 2
        data = self.register_data(self.arrays)
        artifact = identify_linear_model(self.workspace, self.nominal, data, {})
        self.assertEqual(self.workspace.manifest["artifacts"][artifact]["status"], "rejected")
        self.assertEqual(self.workspace.active("nominal_model"), self.nominal)
        with self.assertRaises(ContractError):
            load_identified_model(self.workspace, artifact)


if __name__ == "__main__":
    unittest.main()
