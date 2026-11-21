"""Real CPU training: interrupted runs must match uninterrupted optimizer trajectories."""
from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from control_harness.artifacts import Workspace, file_hash, read_json
from control_harness.checkpoints import checkpoint_sidecar
from control_harness.contracts import ContractError
from control_harness.control import design_controller
from control_harness.data import simulate_transitions
from control_harness.modeling import register_model
from control_harness.policy import collect_expert, optimize_policy, train_imitation
from control_harness.residual import train_residual
from control_harness.runtime import Cancelled, OperationContext


TEMP_ROOT = Path(__file__).resolve().parents[1] / "test-output" / "resume-tests"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        workspace = Workspace(cls.fixture.name)
        var = lambda name, bound: {"name": name, "unit": "1", "lower": -bound, "upper": bound}
        interface = {"schema_version": 1, "name": "resume-fixture", "sample_time": .1,
                     "time_semantics": "discrete_state_transition", "states": [var("x", 3)],
                     "controls": [var("u", 1)], "outputs": [var("y", 3)], "disturbances": [var("d", .2)]}
        nominal = {"interface": interface, "representation": "discrete_polynomial", "A": [[.7]],
                   "B": [[.25]], "E": [[.02]], "C": [[1]], "bias": [0]}
        reference = {**copy.deepcopy(nominal), "A": [[.78]], "bias": [.015]}
        cls.nominal_id = register_model(workspace, nominal, {"kind": "user_nominal"})
        cls.reference_id = register_model(workspace, reference, {"kind": "reference_simulator"})
        cls.dataset_id = simulate_transitions(workspace, cls.nominal_id, cls.reference_id,
                                               {"trajectories": 6, "steps": 6, "seed": 12})

        def scenario(value):
            return {"name": str(value), "initial_state": [value], "reference": [.15],
                    "disturbance": [.03], "steps": 4}

        controller = {"parameters": {"horizon": 2, "q": 1, "r": .01, "s": .01}, "budget": 8,
                      "seed": 13, "parameter_space": {"r": {"lower": .001, "upper": .02}},
                      "tuning_scenarios": [scenario(.1)], "verification_scenarios": [scenario(.2)],
                      "test_scenarios": [scenario(.3)]}
        controller_id = design_controller(workspace, cls.nominal_id, cls.reference_id, controller)
        cls.expert_id = collect_expert(workspace, controller_id, cls.reference_id,
                                      {"scenarios": [scenario(-.6 + k * .09) for k in range(6)], "seed": 14})
        cls.policy_id = train_imitation(workspace, cls.expert_id,
                                        {"hidden_dims": [6], "epochs": 3, "seed": 15})
        cls.configs = {
            "residual": {"epochs": 5, "hidden_dims": [6], "batch_size": 5, "patience": 20,
                         "learning_rate": .01, "seed": 16, "physics_weight": .2, "weight_decay": .001},
            "imitation": {"epochs": 5, "hidden_dims": [7, 5], "batch_size": 3, "patience": 20,
                          "learning_rate": .003, "seed": 17, "dropout": .25},
            "physical": {"epochs": 4, "horizon": 2, "batch_size": 2, "learning_rate": .001,
                         "imitation_weight": .2, "constraint_weight": 50, "seed": 18}}

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEMP_ROOT)
        shutil.copytree(self.fixture.name, self.temp.name, dirs_exist_ok=True)
        self.workspace = Workspace(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def train(self, kind, config, context=None):
        if kind == "residual":
            return train_residual(self.workspace, self.nominal_id, self.dataset_id, config, context)
        if kind == "imitation":
            return train_imitation(self.workspace, self.expert_id, config, context)
        return optimize_policy(self.workspace, self.policy_id, self.nominal_id, config, context)

    def artifact_path(self, artifact_id):
        return self.workspace.path / self.workspace.manifest["artifacts"][artifact_id]["path"]

    def checkpoint(self, artifact_id):
        return torch.load(self.artifact_path(artifact_id) / "latest.pt", weights_only=True)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected), "Tensor differs after continuation")
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

    def interrupt_at_epoch(self, kind, config, exception=Cancelled):
        progress = []

        def emit(event):
            progress.append(event)
            if event.get("epoch") == 2:
                raise exception("intentional interruption after a committed epoch")

        with self.assertRaises(exception):
            self.train(kind, config, OperationContext(emit=emit))
        path = Path(progress[-1]["checkpoint"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent.parent.name, "failed")
        self.assertTrue((path.parent / "failure.json").is_file())
        self.assertFalse(self.workspace.lock_path.exists())
        return path

    def test_cancelled_training_resumes_identical_weights_optimizer_rng_and_history(self):
        for kind in ("residual", "imitation", "physical"):
            with self.subTest(kind=kind):
                config = self.configs[kind]
                continuous = self.train(kind, config)
                path = self.interrupt_at_epoch(kind, config)
                resumed = self.train(kind, {**config, "resume_checkpoint": str(path)})
                a, b = self.checkpoint(resumed), self.checkpoint(continuous)
                self.assert_nested_equal(a, b)
                report = read_json(self.artifact_path(resumed) / "report.json")
                self.assertEqual(report["resumed_from_epoch"], 2)
                self.assertEqual(report["trained_epochs"], config["epochs"])
                self.assertEqual(report["optimizer_updates"], a["updates"])
                self.assertGreater(a["updates"], 0)
                model_name = "model.pt" if kind == "residual" else "policy.pt"
                self.assert_nested_equal(torch.load(self.artifact_path(resumed) / model_name, weights_only=True),
                                         torch.load(self.artifact_path(continuous) / model_name, weights_only=True))

    def test_failure_checkpoint_and_relative_path_can_continue(self):
        config = self.configs["residual"]
        continuous = self.train("residual", config)
        path = self.interrupt_at_epoch("residual", config, RuntimeError)
        resumed = self.train("residual", {**config, "resume_checkpoint": path.relative_to(self.workspace.path).as_posix()})
        self.assert_nested_equal(self.checkpoint(resumed), self.checkpoint(continuous))

    def test_preflight_checkpoint_hash_is_enforced_by_all_training_workers(self):
        for kind in ("residual", "imitation", "physical"):
            with self.subTest(kind=kind):
                config = self.configs[kind]
                path = self.interrupt_at_epoch(kind, config)
                with self.assertRaisesRegex(ContractError, "preflight SHA-256"):
                    self.train(kind, {**config, "resume_checkpoint": str(path), "resume_checkpoint_sha256": "0" * 64})
                resumed = self.train(kind, {**config, "resume_checkpoint": str(path),
                                           "resume_checkpoint_sha256": file_hash(path)})
                self.assertNotIn("resume_checkpoint_sha256", self.checkpoint(resumed)["contract"]["config"])

    def test_partial_first_epoch_replays_from_initial_checkpoint(self):
        config = self.configs["imitation"]
        continuous = self.train("imitation", config)
        checks = 0

        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 4  # two optimizer updates occur before the first epoch completes

        with self.assertRaises(Cancelled):
            self.train("imitation", config, OperationContext(cancelled=cancelled))
        path = next((self.workspace.path / "failed").glob("*/latest.pt"))
        self.assertEqual(torch.load(path, weights_only=True)["epoch"], 0)
        resumed = self.train("imitation", {**config, "resume_checkpoint": str(path)})
        self.assert_nested_equal(self.checkpoint(resumed), self.checkpoint(continuous))

    def test_completed_run_extends_total_budget_without_resetting_optimizer(self):
        config = self.configs["imitation"]
        continuous = self.train("imitation", config)
        short = self.train("imitation", {**config, "epochs": 2})
        resumed = self.train("imitation", {**config, "resume_checkpoint": str(self.artifact_path(short) / "latest.pt")})
        self.assert_nested_equal(self.checkpoint(resumed), self.checkpoint(continuous))

    def test_early_stop_counter_is_preserved_when_total_budget_increases(self):
        config = {**self.configs["imitation"], "epochs": 10, "patience": 1,
                  "learning_rate": 1e-20, "dropout": 0.0}
        stopped = self.train("imitation", config)
        checkpoint = self.checkpoint(stopped)
        self.assertEqual(checkpoint["wait"], 1)
        self.assertLess(checkpoint["epoch"], config["epochs"])
        resumed = self.train("imitation", {**config, "epochs": 20,
                              "resume_checkpoint": str(self.artifact_path(stopped) / "latest.pt")})
        self.assert_nested_equal(self.checkpoint(resumed), checkpoint)

    def test_partial_physical_epoch_replays_all_conditions_from_committed_boundary(self):
        config = self.configs["physical"]
        continuous = self.train("physical", config)
        armed, checks = False, 0

        def emit(event):
            nonlocal armed
            if event.get("epoch") == 2:
                armed = True

        def cancelled():
            nonlocal checks
            if armed:
                checks += 1
            return checks >= 8

        with self.assertRaises(Cancelled):
            self.train("physical", config, OperationContext(emit=emit, cancelled=cancelled))
        path = next((self.workspace.path / "failed").glob("*/latest.pt"))
        self.assertEqual(torch.load(path, weights_only=True)["epoch"], 2)
        resumed = self.train("physical", {**config, "resume_checkpoint": str(path)})
        self.assert_nested_equal(self.checkpoint(resumed), self.checkpoint(continuous))

    def test_signature_covers_previously_omitted_residual_settings(self):
        config = self.configs["residual"]
        path = self.interrupt_at_epoch("residual", config)
        for field, value in (("weight_decay", .02), ("patience", 40), ("physics_weight", .3),
                             ("minimum_improvement", .1), ("seed", 99), ("hidden_dims", [8])):
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "configuration differs"):
                self.train("residual", {**config, field: value, "resume_checkpoint": str(path)})

    def test_policy_signatures_reject_changed_training_and_prediction_settings(self):
        for kind, field, value in (("imitation", "dropout", .1), ("physical", "horizon", 3),
                                   ("physical", "imitation_weight", .4)):
            with self.subTest(kind=kind, field=field):
                config = self.configs[kind]
                path = self.interrupt_at_epoch(kind, config)
                with self.assertRaisesRegex(ContractError, "configuration differs"):
                    self.train(kind, {**config, field: value, "resume_checkpoint": str(path)})

    def test_changed_dependency_is_rejected_before_resume(self):
        config = self.configs["imitation"]
        path = self.interrupt_at_epoch("imitation", config)
        root, _ = self.workspace.resolve(self.expert_id)
        (root / "split.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "missing or changed"):
            self.train("imitation", {**config, "resume_checkpoint": str(path)})

    def test_corrupted_unsupported_external_and_decreased_budget_checkpoints_are_rejected(self):
        config = self.configs["imitation"]
        path = self.interrupt_at_epoch("imitation", config)
        with self.assertRaisesRegex(ContractError, "below"):
            self.train("imitation", {**config, "epochs": 1, "resume_checkpoint": str(path)})
        with self.assertRaisesRegex(ContractError, "this workspace"):
            self.train("imitation", {**config, "resume_checkpoint": str(Path(self.fixture.name) / "latest.pt")})
        original = path.read_bytes()
        path.write_bytes(original[:-50])
        with self.assertRaisesRegex(ContractError, "integrity"):
            self.train("imitation", {**config, "resume_checkpoint": str(path)})
        path.write_bytes(original)
        checkpoint_sidecar(path).unlink()
        with self.assertRaisesRegex(ContractError, "unsupported"):
            self.train("imitation", {**config, "resume_checkpoint": str(path)})


if __name__ == "__main__":
    unittest.main()
