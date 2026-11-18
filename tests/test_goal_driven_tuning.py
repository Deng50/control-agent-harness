"""Real constrained MPC/GA objectives and fail-closed selection, synthetic fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace, canonical_hash, read_json
from control_harness.contracts import ContractError
from control_harness.control import design_controller, evaluate_controller, load_controller
from control_harness.modeling import load_model, register_model
from control_harness.objectives import normalize_evaluation
from control_harness.runtime import OperationContext


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-output/goal-driven-tuning-tests"
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def evaluation():
    return {"schema_version": 1, "recipe": "tracking", "energy_model": "action_proxy",
        "weights": {"rmse": 1., "energy": 0., "tv": 0., "overshoot": 0.},
        "scales": {"rmse": 1., "energy": 1., "tv": 1., "overshoot": 1.}}


def scenario(x, reference, steps=1):
    return {"initial_state": [x], "reference": [reference], "disturbance": [], "steps": steps}


def configuration():
    return {"parameters": {"horizon": 1, "q": .001, "r": 1., "s": .01},
        "parameter_space": {"q": {"lower": .001, "upper": 10000.}}, "budget": 8, "seed": 43,
        "tuning_scenarios": [scenario(0., .1)], "verification_scenarios": [scenario(.05, .2)],
        "test_scenarios": [scenario(.1, .25)], "evaluation": evaluation()}


class GoalDrivenTuningTests(unittest.TestCase):
    confirmation_source = "test_fixture"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=OUTPUT)
        self.addCleanup(self.temp.cleanup)
        self.workspace = Workspace(Path(self.temp.name) / "case")
        self.nominal, self.plant = self.models()

    def models(self, real_b=.45, nu=1):
        variable = lambda name: {"name": name, "unit": "1", "lower": -2., "upper": 2.}
        interface = {"schema_version": 1, "name": "explicit synthetic scalar tuning fixture", "sample_time": .2,
            "time_semantics": "discrete_state_transition", "states": [variable("x")], "controls": [variable("u" + str(index)) for index in range(nu)],
            "outputs": [variable("y")], "disturbances": []}
        spec = {"representation": "discrete_polynomial", "interface": interface, "A": [[.8]], "B": [[.3] * nu], "C": [[1.]], "terms": []}
        nominal = register_model(self.workspace, spec, {"kind": "user_nominal", "description": "synthetic test_fixture prediction"})
        other = copy.deepcopy(spec); other["B"] = [[real_b] * nu]
        plant = register_model(self.workspace, other, {"kind": "reference_simulator", "description": "explicit separate synthetic simulator; not measurements"})
        return nominal, plant

    def run_design(self, config=None, population=None, ids=None):
        ids = ids or (self.nominal, self.plant)
        if population is None:
            result = design_controller(self.workspace, *ids, configuration() if config is None else config)
        else:
            # Only the deterministic test initial population is fixed. Every
            # candidate still runs the real MPC solver and original GA children.
            with patch("control_harness.control.GeneticOperators._initial_population", return_value=copy.deepcopy(population)):
                result = design_controller(self.workspace, *ids, configuration() if config is None else config)
        root, metadata = self.workspace.resolve(result, allow_rejected=True)
        return result, root, metadata, read_json(root / "report.json")

    def test_external_metrics_use_real_initial_output_dt_and_fixed_normalization(self):
        model, plant = load_model(self.workspace, self.nominal), load_model(self.workspace, self.plant)
        parameters = {"horizon": 2, "q": 1., "r": .1, "s": .01, "input_rate_bounds": [[-.2, .2]]}
        objective = evaluation(); objective["weights"] = {"rmse": 2., "energy": .3, "tv": .4, "overshoot": .5}
        objective["scales"] = {"rmse": 2., "energy": 3., "tv": 4., "overshoot": .1}
        actual = evaluate_controller(model, plant, parameters, [scenario(.1, .6, 4)], OperationContext(), evaluation=objective)
        run = actual["runs"][0]; rows = run["trajectory"]
        y, u = np.asarray([row["output"] for row in rows]), np.asarray([row["u"] for row in rows])
        rmse = float(np.sqrt(np.mean((y - .6) ** 2)))
        energy = float(np.sum(u**2) * .2)
        tv = float(np.abs(np.diff(np.vstack(([0.], u)), axis=0)).sum())
        overshoot = float(max(0., (y.max() - .6) / (.6 - .1)))
        metrics = run["objective_metrics"]
        for key, expected in {"rmse": rmse, "energy": energy, "tv": tv, "overshoot": overshoot}.items():
            self.assertAlmostEqual(metrics[key], expected, places=13)
        expected = 2*rmse/2 + .3*energy/3 + .4*tv/4 + .5*overshoot/.1
        self.assertAlmostEqual(actual["score"], expected, places=13)
        self.assertEqual(run["initial_output"], [.1]); self.assertEqual(run["initial_control"], [0.])
        self.assertNotEqual(run["initial_output"], rows[0]["output"])
        self.assertEqual(actual["evaluation"], normalize_evaluation(objective))
        self.assertEqual(actual["evaluation_hash"], canonical_hash(normalize_evaluation(objective)))
        self.assertTrue(metrics["constraints"]["input_rate"]["available"])

    def test_real_nonfinite_evaluation_output_keeps_json_safe_failed_prefix(self):
        root, _ = self.workspace.resolve(self.plant)
        source = read_json(root / "model.json")
        source["B"] = [[1e308]]; source["C"] = [[1e308]]
        artifact = register_model(self.workspace, source, {"kind": "reference_simulator", "description": "explicit arithmetic-overflow negative test_fixture"})
        result = evaluate_controller(load_model(self.workspace, self.nominal), load_model(self.workspace, artifact),
            {"horizon": 1, "q": 1., "r": .1, "s": .01}, [scenario(0., .6)], OperationContext(), evaluation=evaluation())
        self.assertIsNone(result["score"]); self.assertFalse(result["hard_constraints_passed"])
        self.assertEqual(result["runs"][0]["status"], "failed")
        self.assertEqual(result["runs"][0]["completed_steps"], 0)
        self.assertEqual(result["runs"][0]["trajectory"], [])
        self.assertIn("nonfinite", result["runs"][0]["error"]["message"])
        json.dumps(result, allow_nan=False)

    def test_actual_unconverged_candidates_are_retained_and_feasible_baseline_survives(self):
        config = configuration(); config["parameters"].update(q=1e-10, max_iterations=2)
        config["parameter_space"] = {"q": {"lower": 1e-10, "upper": 1.}}
        _, root, metadata, report = self.run_design(config, population=[{"q": 1.}] * 4)
        self.assertEqual(metadata["validation"]["status"], "verified")
        self.assertEqual(report["selection"], "baseline_retained"); self.assertIsNone(report["best"])
        self.assertEqual(len(report["failed_candidates"]), 8)
        self.assertTrue(all(trial["score"] is None and trial["status"] == "failed" for trial in report["trials"]))
        self.assertTrue(all("Iteration limit reached" in trial["tuning_evaluation"]["runs"][0]["error"]["message"] for trial in report["trials"]))
        self.assertTrue(all(trial["tuning_evaluation"]["runs"][0]["failure"]["step"] == 0 for trial in report["trials"]))
        self.assertEqual(report["last"], report["trials"][-1])
        self.assertEqual(report["chosen_parameters"], report["baseline_parameters"])
        self.assertEqual(len(list((root / "trials").glob("trial-*.json"))), 8)
        self.assertEqual(report["test_evaluations"], 1)

    def test_failed_baseline_can_be_replaced_by_a_genuinely_feasible_candidate(self):
        config = configuration(); config["parameters"].update(q=1., max_iterations=2)
        config["parameter_space"] = {"q": {"lower": 1e-10, "upper": 1.}}
        _, _, metadata, report = self.run_design(config, population=[{"q": 1e-10}] * 4)
        self.assertFalse(report["baseline"]["hard_constraints_passed"])
        self.assertTrue(report["tuned_candidate_accepted"])
        self.assertEqual(report["selection_reason"], "baseline_infeasible_candidate_verified")
        self.assertTrue(report["verification_candidate"]["hard_constraints_passed"])
        self.assertEqual(metadata["validation"]["status"], "verified")

    def test_verification_failure_retains_baseline_despite_better_tuning_score(self):
        config = configuration(); config["verification_scenarios"] = [scenario(1.5, 1.8)]
        _, _, metadata, report = self.run_design(config, population=[{"q": 10000.}] * 4)
        self.assertLess(report["best"]["score"], report["baseline"]["score"])
        self.assertFalse(report["verification_candidate"]["hard_constraints_passed"])
        self.assertTrue(report["verification_baseline"]["hard_constraints_passed"])
        self.assertEqual(report["selection"], "baseline_retained")
        self.assertEqual(report["chosen_parameters"], report["baseline_parameters"])
        self.assertEqual(metadata["validation"]["status"], "verified")

    def test_rejected_test_never_reselects_baseline_or_replaces_active_controller(self):
        old, _, _, _ = self.run_design(population=[{"q": 10000.}] * 4)
        config = configuration(); config["test_scenarios"] = [scenario(1.5, 1.8)]
        result, root, metadata, report = self.run_design(config, population=[{"q": 10000.}] * 4)
        self.assertTrue(report["tuned_candidate_accepted"])
        self.assertEqual(report["selection"], "tuned")
        self.assertFalse(report["test"]["hard_constraints_passed"])
        self.assertEqual(report["test_evaluations"], 1)
        self.assertEqual(metadata["validation"]["status"], "rejected")
        self.assertEqual(self.workspace.manifest["active"]["controller"], old)
        self.workspace.resolve(old)
        self.assertFalse((root / "bundle").exists())
        with self.assertRaises(ContractError): load_controller(self.workspace, result)

    def test_shared_legacy_ga_path_keeps_original_score_and_report_contract(self):
        config = configuration(); config.pop("evaluation")
        _, _, metadata, report = self.run_design(config)
        self.assertNotIn("evaluation", report)
        self.assertEqual(len(report["trials"]), 8)
        self.assertEqual({trial["generation"] for trial in report["trials"]}, {0, 1})
        old = report["baseline"]
        expected = np.mean([run["metrics"]["normalized_rmse"] + .01*run["metrics"]["control_effort"] for run in old["runs"]])
        self.assertEqual(old["score"], expected)
        self.assertNotIn("objective_metrics", old["runs"][0])
        self.assertEqual(metadata["kind"], "controller")

    def test_unknown_or_invalid_objective_settings_and_overlapping_splits_fail_before_ga(self):
        cases = [{"evaluation": None}, {"budget": True}, {"seed": False}, {"minimum_improvement": 1.1},
            {"evaluation": {**evaluation(), "hard_gates": {"per_output": {"unknown-output": {"max_rmse": 1.}}}}},
            {"parameters": {"horizon": 1.5}}, {"parameters": {"q": True}},
            {"parameters": {"max_iterations": False}}, {"parameters": {"solver_tolerance": float("nan")}},
            {"tuning_scenarios": [scenario(3., .1)]}, {"test_scenarios": [scenario(0., .1)]},
            {"tuning_scenarios": [scenario(0., .1, 10**9)]}]
        for patch_value in cases:
            config = configuration(); config.update(patch_value)
            with self.subTest(patch=patch_value), patch("control_harness.control._genetic_trials", side_effect=AssertionError("must reject before GA")), self.assertRaises(ContractError):
                self.run_design(config)
        self.assertFalse(any(row["kind"] == "controller" for row in self.workspace.manifest["artifacts"].values()))

    def test_max_candidate_horizon_resource_limit_is_checked_before_any_solver(self):
        ids = self.models(nu=3)
        config = configuration(); config["parameters"]["horizon"] = 1
        config["parameter_space"] = {"horizon": {"lower": 1, "upper": 100, "integer": True}}
        with patch("control_harness.control.MPCPolicy", side_effect=AssertionError("no SLSQP before resource check")), self.assertRaisesRegex(ContractError, "resource budget"):
            self.run_design(config, ids=ids)

    def test_metadata_duration_and_numeric_spelling_cannot_create_independent_conditions(self):
        ids = self.models()
        for variant in ("metadata", "duration", "numeric_spelling"):
            with self.subTest(variant=variant):
                config = configuration()
                base = copy.deepcopy(config["tuning_scenarios"][0])
                for index, group in enumerate(("tuning_scenarios","verification_scenarios","test_scenarios")):
                    scenario = copy.deepcopy(base)
                    scenario["name"] = group
                    if variant == "metadata": scenario["comment"] = group
                    elif variant == "duration": scenario["steps"] = 9 + index
                    else:
                        scenario["initial_state"] = [[0],[0.0],[-0.0]][index]
                        scenario["reference"] = [[1],[1.0],[1.0]][index]
                    config[group] = [scenario]
                with patch("control_harness.control._genetic_trials",side_effect=AssertionError("No GA before independent physical conditions")), self.assertRaisesRegex(ContractError,"scenario fields|conditions overlap"):
                    self.run_design(config,ids=ids)
                self.assertFalse(any(row["kind"]=="controller" for row in self.workspace.manifest["artifacts"].values()))


if __name__ == "__main__": unittest.main()
