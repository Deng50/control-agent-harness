"""Real expert collection/training and exact deployed amplitude/rate projection checks."""
from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from control_harness.artifacts import Workspace, read_json
from control_harness.contracts import ContractError, ModelInterface
from control_harness.control import design_controller
from control_harness.modeling import register_model, tensor
from control_harness.policy import (NeuralPolicy, collect_expert, constraints_pass, expert_previous_actions,
                                  load_policy, network_config, optimize_policy, train_imitation)
from control_harness.runtime import Cancelled, OperationContext


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "rate-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


def scenario(value, reference=.6, steps=5):
    return {"name": str(value), "initial_state": [value], "reference": [reference],
            "disturbance": [.03], "steps": steps}


def controller_config(rate):
    parameters = {"horizon": 2, "q": 1, "r": .01, "s": .01}
    if rate is not None:
        parameters["input_rate_bounds"] = rate
    return {"parameters": parameters, "budget": 8, "seed": 13,
            "parameter_space": {"r": {"lower": .001, "upper": .02}},
            "tuning_scenarios": [scenario(.1)], "verification_scenarios": [scenario(.2)],
            "test_scenarios": [scenario(.3)]}


class PolicyRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        workspace = Workspace(cls.fixture.name)
        var = lambda name, bound: {"name": name, "unit": "1", "lower": -bound, "upper": bound}
        cls.interface = {"schema_version": 1, "name": "rate-fixture", "sample_time": .1,
                         "time_semantics": "discrete_state_transition", "states": [var("x", 3)],
                         "controls": [var("u", 1)], "outputs": [var("y", 3)], "disturbances": [var("d", .2)]}
        cls.nominal = {"interface": cls.interface, "representation": "discrete_polynomial", "A": [[.7]],
                       "B": [[.25]], "E": [[.02]], "C": [[1]], "bias": [0]}
        cls.nominal_id = register_model(workspace, cls.nominal, {"kind": "user_nominal"})
        reference = {**copy.deepcopy(cls.nominal), "A": [[.78]], "bias": [.015]}
        cls.reference_id = register_model(workspace, reference, {"kind": "reference_simulator"})
        cls.rate = [[-.08, .08]]
        cls.controller_id = design_controller(workspace, cls.nominal_id, cls.reference_id, controller_config(cls.rate))
        cls.collection = {"scenarios": [scenario(-.6 + k * .09) for k in range(6)], "seed": 14}
        cls.expert_id = collect_expert(workspace, cls.controller_id, cls.reference_id, cls.collection)
        cls.imitation_config = {"hidden_dims": [7], "epochs": 4, "seed": 15, "dropout": .2,
                                "batch_size": 3, "learning_rate": .003, "patience": 10}
        cls.policy_id = train_imitation(workspace, cls.expert_id, cls.imitation_config)
        cls.physical_config = {"epochs": 3, "seed": 16, "batch_size": 2, "horizon": 3,
                               "learning_rate": .001, "imitation_weight": .2}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        shutil.copytree(self.fixture.name, self.temp.name, dirs_exist_ok=True)
        self.workspace = Workspace(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def path(self, artifact_id):
        return self.workspace.path / self.workspace.manifest["artifacts"][artifact_id]["path"]

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_three_architectures_project_intersection_and_use_explicit_previous(self):
        spec = copy.deepcopy(self.interface)
        spec["controls"].append({"name": "v", "unit": "1", "lower": -1, "upper": 1})
        interface = ModelInterface.parse(spec)
        for architecture in ("mlp", "rnn", "lstm"):
            with self.subTest(architecture=architecture):
                config = network_config(interface, {"architecture": architecture, "hidden_dims": [5],
                    "hidden_size": 5, "preview_len": 2, "disturbance_forecast": "declared_constant"})
                config["action_constraints"] = {"input_rate_bounds": [[-.2, .1], [-.05, .3]]}
                scalers = {"input_mean": [0] * config["input_dim"], "input_scale": [1] * config["input_dim"],
                           "output_mean": [20, -20], "output_scale": [1, 1]}
                policy = NeuralPolicy(interface, config, scalers)
                with torch.no_grad():
                    for parameter in policy.parameters():
                        parameter.zero_()
                x, ref, d = tensor([[0]]), tensor([[0]]), tensor([[0]])
                first = policy(x, ref, d, previous=tensor([[.95, -.98]]))
                self.assertTrue(torch.equal(first, tensor([[1, -1]])))
                previous = tensor([[-.9, .9]]).requires_grad_(True)
                result = policy(x, ref, d, previous=previous)
                self.assertTrue(torch.allclose(result, tensor([[-.8, .85]])))
                gradient, = torch.autograd.grad(result.sum(), previous)
                self.assertTrue(torch.equal(gradient, torch.ones_like(previous)))
                deployed, _ = policy.act([0], [0], [0], [-.9, .9])
                np.testing.assert_allclose(deployed, result.detach().numpy()[0], rtol=0, atol=0)
                with self.assertRaisesRegex(ContractError, "explicit previous"):
                    policy(x, ref, d)
                with self.assertRaisesRegex(ContractError, "empty intersection"):
                    policy(x, ref, d, previous=tensor([[3, 0]]))
                with self.assertRaisesRegex(ContractError, "same finite shape"):
                    policy(x, ref, d, previous=tensor([0, 0]))

    def test_real_three_network_training_preserves_expert_constraints_and_deployment(self):
        for architecture in ("mlp", "rnn", "lstm"):
            with self.subTest(architecture=architecture):
                config = {**self.imitation_config, "architecture": architecture, "epochs": 2,
                          "hidden_size": 5, "preview_len": 2, "disturbance_forecast": "declared_constant"}
                policy_id = train_imitation(self.workspace, self.expert_id, config)
                optimized = optimize_policy(self.workspace, policy_id, self.nominal_id,
                                            {**self.physical_config, "epochs": 2})
                policy = load_policy(self.workspace, optimized)
                self.assertEqual(policy.rate, self.rate)
                for artifact_id in (policy_id, optimized):
                    root = self.path(artifact_id)
                    network = read_json(root / "network.json")
                    descriptor, report = read_json(root / "policy.json"), read_json(root / "report.json")
                    self.assertEqual(network["action_constraints"]["input_rate_bounds"], self.rate)
                    self.assertEqual(network["action_constraints"]["source_controller"], self.controller_id)
                    self.assertEqual(descriptor["action_constraints"], network["action_constraints"])
                    self.assertEqual(report["action_constraints"], network["action_constraints"])
                    self.assertTrue(report["constraint_checks"]["input_rate"])
                    self.assertGreater(report["optimizer_updates"], 0)
                comparison = read_json(self.path(optimized) / "report.json")["comparison"]
                for row in comparison:
                    for key in ("expert", "imitation", "optimized"):
                        self.assertTrue(constraints_pass([row[key]]))
                        self.assertTrue(row[key]["constraint_checks"]["input_rate"])
                        self.assertLessEqual(row[key]["metrics"]["max_rate_violation"], 1e-6)
                        actions = np.array([item["u"] for item in row[key]["trajectory"]])
                        deltas = np.diff(np.vstack(([[0]], actions)), axis=0)
                        self.assertTrue(np.all(np.abs(deltas) <= .080001))

    def test_previous_actions_reset_by_trajectory_and_legacy_reconstruction_agrees(self):
        with np.load(self.path(self.expert_id) / "expert.npz", allow_pickle=False) as loaded:
            data = dict(loaded)
        previous = expert_previous_actions(data, ModelInterface.parse(self.interface)).numpy()
        np.testing.assert_array_equal(previous, data["previous_u"])
        for trajectory in np.unique(data["trajectory_id"]):
            indexes = np.flatnonzero(data["trajectory_id"] == trajectory)
            np.testing.assert_array_equal(previous[indexes[0]], [0])
            np.testing.assert_array_equal(previous[indexes[1:]], data["u"][indexes[:-1]])
        del data["previous_u"]
        np.testing.assert_array_equal(expert_previous_actions(data, ModelInterface.parse(self.interface)), previous)
        data["previous_u"] = data["u"]
        with self.assertRaisesRegex(ContractError, "trajectory action history"):
            expert_previous_actions(data, ModelInterface.parse(self.interface))

    def test_rate_training_resume_matches_full_optimizer_state_and_signs_constraints(self):
        for kind in ("imitation", "physical"):
            with self.subTest(kind=kind):
                config = self.imitation_config if kind == "imitation" else self.physical_config
                def train(settings, context=None):
                    if kind == "imitation":
                        return train_imitation(self.workspace, self.expert_id, settings, context)
                    return optimize_policy(self.workspace, self.policy_id, self.nominal_id, settings, context)
                full = train(config)
                events = []
                def emit(event):
                    events.append(event)
                    if event.get("epoch") == 1:
                        raise Cancelled("intentional committed-epoch interruption")
                with self.assertRaises(Cancelled):
                    train(config, OperationContext(emit=emit))
                checkpoint = Path(events[-1]["checkpoint"])
                saved = torch.load(checkpoint, weights_only=True)
                action = saved["contract"]["config"]["action_constraints"]
                self.assertEqual(action["input_rate_bounds"], self.rate)
                self.assertEqual(action["projection"], "amplitude_rate_intersection_v1")
                resumed = train({**config, "resume_checkpoint": str(checkpoint)})
                self.assert_nested_equal(torch.load(self.path(full) / "latest.pt", weights_only=True),
                                         torch.load(self.path(resumed) / "latest.pt", weights_only=True))
                self.assert_nested_equal(torch.load(self.path(full) / "policy.pt", weights_only=True),
                                         torch.load(self.path(resumed) / "policy.pt", weights_only=True))
                with self.assertRaisesRegex(ContractError, "inherited from the expert"):
                    train({**config, "resume_checkpoint": str(checkpoint), "input_rate_bounds": [[-.2, .2]]})

    def test_rate_not_configured_is_explicitly_disabled_and_cannot_be_injected(self):
        controller = design_controller(self.workspace, self.nominal_id, self.reference_id, controller_config(None))
        expert = collect_expert(self.workspace, controller, self.reference_id, self.collection)
        policy_id = train_imitation(self.workspace, expert, {**self.imitation_config, "epochs": 2})
        policy = load_policy(self.workspace, policy_id)
        self.assertIsNone(policy.rate)
        report = read_json(self.path(policy_id) / "report.json")
        self.assertFalse(report["constraint_checks"]["input_rate"])
        self.assertIsNone(report["action_constraints"]["input_rate_bounds"])
        for run in report["closed_loop"]["test"]:
            self.assertFalse(run["constraint_checks"]["input_rate"])
            self.assertEqual(run["metrics"]["max_rate_violation"], 0)
        with self.assertRaisesRegex(ContractError, "inherited from the expert"):
            train_imitation(self.workspace, expert, {"epochs": 2, "input_rate_bounds": self.rate})

    def test_changed_expert_rate_rejects_resume_signature(self):
        events = []
        def emit(event):
            events.append(event)
            if event.get("epoch") == 1:
                raise Cancelled("retain checkpoint before changing the expert controller")
        with self.assertRaises(Cancelled):
            train_imitation(self.workspace, self.expert_id, self.imitation_config, OperationContext(emit=emit))
        checkpoint = Path(events[-1]["checkpoint"])
        controller = design_controller(self.workspace, self.nominal_id, self.reference_id,
                                       controller_config([[-.04, .04]]))
        expert = collect_expert(self.workspace, controller, self.reference_id, self.collection)
        with self.assertRaisesRegex(ContractError, "configuration differs"):
            train_imitation(self.workspace, expert, {**self.imitation_config, "resume_checkpoint": str(checkpoint)})

    def test_real_expert_collection_rejects_output_violation_and_blocks_training(self):
        nominal = copy.deepcopy(self.nominal)
        nominal["interface"]["outputs"][0].update(lower=-.5, upper=.5)
        reference = {**copy.deepcopy(nominal), "B": [[10]]}
        nominal_id = register_model(self.workspace, nominal, {"kind": "user_nominal"})
        reference_id = register_model(self.workspace, reference, {"kind": "reference_simulator"})
        config = controller_config(self.rate)
        for group, value in (("tuning_scenarios", .001), ("verification_scenarios", .002), ("test_scenarios", .003)):
            config[group] = [scenario(value, reference=0, steps=1)]
        controller = design_controller(self.workspace, nominal_id, reference_id, config)
        self.workspace.resolve(controller)
        expert = collect_expert(self.workspace, controller, reference_id,
                                {"scenarios": [scenario(k * .001, reference=.4, steps=1) for k in range(6)]})
        self.assertEqual(self.workspace.manifest["artifacts"][expert]["status"], "rejected")
        report = read_json(self.path(expert) / "collection.json")
        self.assertFalse(report["closed_loop_constraints_passed"])
        self.assertTrue(all(row["metrics"]["max_state_violation"] == 0 for row in report["scenarios"]))
        self.assertTrue(all(row["metrics"]["max_output_violation"] > .1 for row in report["scenarios"]))
        with self.assertRaisesRegex(ContractError, "failed acceptance"):
            train_imitation(self.workspace, expert, {"epochs": 1})


if __name__ == "__main__":
    unittest.main()
