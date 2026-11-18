"""Real pre-confirmation checkpoint continuation; authorization refs are not optimizer settings."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch
from control_harness.artifacts import Workspace
from control_harness.contracts import ContractError
from control_harness.data import simulate_transitions
from control_harness.modeling import register_model
from control_harness.residual import train_residual
from control_harness.runtime import Cancelled, OperationContext

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("CONTROL_TEST_OUTPUT", ROOT / "test-output" / "confirmation-migration-tests"))
OUTPUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)


class ConfirmationCheckpointMigrationTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor): self.assertTrue(torch.equal(actual, expected), "Checkpoint tensor differs")
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected: self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected): self.assert_nested_equal(a, b)
        else: self.assertEqual(actual, expected)

    def test_old_no_refs_checkpoint_resumes_with_new_refs_identical_to_uninterrupted_training(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as directory:
            workspace = Workspace(directory)
            var = lambda name, limit: {"name": name, "unit": "1", "lower": -limit, "upper": limit}
            interface = {"schema_version": 1, "name": "pre-S03-checkpoint-fixture", "sample_time": .1,
                "time_semantics": "discrete_state_transition", "states": [var("x", 3)],
                "controls": [var("u", 1)], "outputs": [var("y", 3)], "disturbances": []}
            nominal = {"interface": interface, "representation": "discrete_polynomial", "A": [[.7]], "B": [[.25]], "C": [[1]]}
            reference = {**copy.deepcopy(nominal), "A": [[.78]], "bias": [.02]}
            model = register_model(workspace, nominal, {"kind": "user_nominal", "description": "explicit mathematical test fixture"})
            plant = register_model(workspace, reference, {"kind": "reference_simulator", "description": "explicit mathematical test fixture"})
            data = simulate_transitions(workspace, model, plant, {"trajectories": 6, "steps": 6, "seed": 57})
            config = {"epochs": 4, "hidden_dims": [5], "batch_size": 6, "seed": 58,
                      "learning_rate": .005, "physics_weight": .2, "patience": 20, "minimum_improvement": 0}
            continuous = train_residual(workspace, model, data, config)
            continuous_path = workspace.path / workspace.manifest["artifacts"][continuous]["path"]
            expected = torch.load(continuous_path / "latest.pt", weights_only=True)
            checkpoints = []
            def interrupt(event):
                if event.get("epoch") == 2:
                    checkpoints.append(Path(event["checkpoint"]))
                    raise Cancelled("explicit pre-S03 no-plan-refs epoch-boundary interruption")
            with self.assertRaises(Cancelled):
                train_residual(workspace, model, data, config, OperationContext(emit=interrupt))
            checkpoint = checkpoints[0]
            old = torch.load(checkpoint, weights_only=True)
            self.assertNotIn("plan_refs", old["contract"]["config"])
            self.assertEqual(old["epoch"], 2)
            refs = {"signals": {"plan_id": "plan-new-signals", "version_hash": "a" * 64},
                    "training": {"plan_id": "plan-new-training", "version_hash": "b" * 64}}
            # The direct numerical API is an explicit trusted test context. Host
            # provenance/receipt admission is independently exercised by Node.
            resumed_config = {**config, "plan_refs": refs, "resume_checkpoint": str(checkpoint)}
            with self.assertRaisesRegex(ContractError, "training configuration differs"):
                train_residual(workspace, model, data, {**resumed_config, "learning_rate": .01})
            resumed = train_residual(workspace, model, data, resumed_config)
            resumed_path = workspace.path / workspace.manifest["artifacts"][resumed]["path"]
            actual = torch.load(resumed_path / "latest.pt", weights_only=True)
            self.assert_nested_equal(actual, expected)
            self.assertNotIn("plan_refs", actual["contract"]["config"])
            self.assert_nested_equal(torch.load(resumed_path / "model.pt", weights_only=True),
                                     torch.load(continuous_path / "model.pt", weights_only=True))


if __name__ == "__main__": unittest.main()
