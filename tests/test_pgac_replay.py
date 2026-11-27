"""Replay windows, online original-Plant acquisition and interruption contracts."""
import copy
import unittest
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import torch

import test_pgac_gradients as fixtures
from control_harness.artifacts import file_hash
from control_harness.artifacts import atomic_json, read_json
from control_harness.contracts import ContractError
from control_harness.restricted_pgac import physical_rollout_loss, _segment_loss
from control_harness.pgac_replay import SegmentSampler
from control_harness.runtime import Cancelled, OperationContext


class WindowTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.PgacGradientTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.PgacGradientTests.tearDownClass.__func__)
    setUp = fixtures.PgacGradientTests.setUp

    def test_time_varying_reference_disturbance_noise_and_multirate_gradient(self):
        steps = 3
        refs = torch.stack([self.reference, -self.reference, self.reference * 0.5], dim=1)
        ds = torch.stack([self.d, -self.d, self.d * 1.5], dim=1)
        noise = torch.zeros((2, steps, 4), dtype=torch.float64)
        noise[:, 1, 0] = 0.08
        settings = {**self.settings, 'imitation_weight': 0.0, 'horizon': steps}

        def loss():
            return physical_rollout_loss(
                self.policy,
                self.model,
                self.x,
                refs[:, 0],
                ds[:, 0],
                self.previous,
                self.expert,
                settings,
                substeps=2,
                reference_sequence=refs,
                disturbance_sequence=ds,
                noise_sequence=noise,
            )

        x = self.x.numpy().copy()
        previous = self.previous.numpy().copy()
        expected = 0.0
        for k in range(steps):
            r = refs[:, k].numpy()
            d = ds[:, k].numpy()
            u = self.policy.predict(
                np.column_stack((x + noise[:, k].numpy(), r, d, previous)), previous_action=previous
            )
            for _ in range(2):
                x = x @ self.A.T + u @ self.B.T + d @ self.E.T
            y = x @ self.C.T
            expected += (
                settings['q'] * np.mean(((y - r) / 4.0) ** 2)
                + settings['r'] * np.mean((u / 3.0) ** 2)
                + settings['s'] * np.mean(((u - previous) / 3.0) ** 2)
            )
            previous = u
        self.assertAlmostEqual(float(loss().detach()), expected / steps, places=13)
        loss().backward()
        p = self.policy.network[-1].bias
        analytical = float(p.grad[0])
        initial = float(p.detach()[0])
        h = 1e-5
        with torch.no_grad():
            p[0] = initial + h
            plus = float(loss())
            p[0] = initial - h
            minus = float(loss())
            p[0] = initial
        self.assertAlmostEqual(analytical, (plus - minus) / (2 * h), places=9)
        self.assertGreater(abs(analytical), 1e-9)

    def test_student_actions_do_not_become_teacher_labels(self):
        kw = dict(anchor_mask=torch.tensor([False, True]), substeps=2)

        def run(actions):
            return physical_rollout_loss(
                self.policy,
                self.model,
                self.x,
                self.reference,
                self.d,
                self.previous,
                actions,
                self.settings,
                **kw
            )

        altered = self.expert.clone()
        altered[0] += 30.0
        torch.testing.assert_close(run(self.expert), run(altered), atol=0, rtol=0)
        altered[1] += 1.0
        self.assertGreater(float(run(altered).detach()), float(run(self.expert).detach()))

    def test_sampling_only_ablation_retains_old_loss_at_last_expert_row(self):
        episode = {
            'source': 'expert',
            'family': 'up',
            'x': self.x,
            'previous_u': self.previous,
            'u': self.expert,
            'reference': self.reference,
            'd': self.d,
            'noise': torch.zeros(2, 4, dtype=torch.float64),
        }
        settings = {**self.settings, 'batch_size': 1}
        sampler = SegmentSampler(
            [episode], [], {'mode': 'balanced_starts', 'student_fraction': 0.0}, self.model.interface
        )
        length = settings['horizon']
        measured, parts = _segment_loss(
            self.policy, self.model, sampler, {length: [(episode, 1)]}, settings, 2, OperationContext()
        )
        expected = physical_rollout_loss(
            self.policy,
            self.model,
            self.x[1:],
            self.reference[1:],
            self.d[1:],
            self.previous[1:],
            self.expert[1:],
            settings,
            substeps=2,
        )
        torch.testing.assert_close(measured, expected, atol=0, rtol=0)
        self.assertAlmostEqual(float(measured.detach()), sum(parts.values()), places=14)


class OnlineTrainingTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.RestrictedPgacTrainingTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.RestrictedPgacTrainingTests.tearDownClass.__func__)
    setUp = fixtures.RestrictedPgacTrainingTests.setUp
    train = fixtures.RestrictedPgacTrainingTests.train
    assert_nested_equal = fixtures.RestrictedPgacTrainingTests.assert_nested_equal

    def config_for_replay(self):
        c = copy.deepcopy(self.config)
        scenario = {
            'name': 'reverse',
            'family': 'reversal',
            'initial_state': [0.1],
            'initial_previous_action': [0.15],
            'reference': [0.4],
            'disturbance': [],
            'steps': 4,
            'reference_sequence': [[0.4], [0.4], [-0.2], [-0.2]],
        }
        short = {
            **scenario,
            'initial_state': [-0.14],
            'name': 'short',
            'steps': 1,
            'reference_sequence': [[0.4]],
        }
        c['replay'] = {
            'mode': 'online',
            'scenarios': [
                scenario,
                {**scenario, 'name': 'duplicate', 'noise_seed': 2, 'observation_noise_std': [0.0]},
                short,
            ],
            'student_fraction': 0.75,
            'capacity_steps': 10,
            'refresh_every': 1,
            'updates_per_epoch': 2,
        }
        return c

    def replay_fixture(self):
        from control_harness.external_policy import load_external_policy
        from control_harness.external_policy_evaluation import _selection
        from control_harness.pgac_replay import SegmentReplay, normalize_replay

        selected = _selection(self.ws, self.policy)
        config = self.config_for_replay()
        root = selected['dataset_root']
        with np.load(root / 'expert.npz', allow_pickle=False) as raw:
            replay = SegmentReplay(
                normalize_replay(config['replay']),
                selected,
                raw,
                read_json(root / 'collection-plan.json'),
                config['original_evaluation'],
                self.domain,
            )
        return replay, load_external_policy(selected['policy_root'])

    def test_restoration_rejects_wrong_schedule_initial_control_cursor_or_policy(self):
        replay, policy = self.replay_fixture()
        stamp = 'a' * 64
        replay.refresh(self.ws, policy, 0, stamp, OperationContext(), lambda epoch: {})
        good = copy.deepcopy(replay.state)
        replay.restore(copy.deepcopy(good), 0, stamp)
        variants = []
        for key in ('previous_u', 'reference', 'observation', 'noise'):
            variant = copy.deepcopy(good)
            variant['episodes'][0][key][0, 0] += 0.01
            variants.append(variant)
        for key in ('total_steps', 'next_id', 'completed_cases'):
            variants.append({**good, key: good[key] - 1})
        variants.append({**good, 'episodes': good['episodes'][1:]})
        for variant in variants:
            with self.assertRaises(ContractError):
                replay.restore(variant, 0, stamp)
        with self.assertRaisesRegex(ContractError, 'different checkpoint policy'):
            replay.restore(copy.deepcopy(good), 0, 'b' * 64)

    def test_student_factory_checks_original_outputs_and_memory_at_every_plant_subtick(self):
        from control_harness.model_adapters import load_python_adapter
        from control_harness.external_pgac_evaluation import _CoveragePlant
        from control_harness.expert_adapter import ExpertFailure

        replay, policy = self.replay_fixture()
        self.assertEqual(replay.substeps, 2)
        source = self.root / 'factory.py'
        template = """import numpy as np
class Plant:
    def reset(self,x0):
        self.state=x0.copy()
        self.count=0
    def step(self,u,d):
        self.state=-self.state
        self.count+=1
        return self.state.copy()
    def observe(self):
        return {observation}
def create(p): return Plant()
"""
        spec = {
            'schema_version': 1,
            'backend': 'python_factory',
            'source_file': 'factory.py',
            'source_semantics': 'transition',
            'factory': 'create',
            'reset_strategy': 'recreate',
            'output_entrypoint': 'observe',
            'parameters': {},
            'interface': self.physical,
        }
        # Both faults vanish at the next control decision (two original Plant steps).
        for observation, exception, message in [
            ('self.state.copy() + self.count % 2', ContractError, 'hidden memory'),
            ('np.where(self.state < 0., 10., 0.)', ExpertFailure, 'student output at Plant subtick'),
        ]:
            source.write_text(template.format(observation=observation), encoding='utf-8')
            with load_python_adapter(
                self.root, spec, expected_file_hashes={'factory.py': file_hash(source)}
            ) as plant:
                guarded = _CoveragePlant(plant, self.domain)
                with self.assertRaisesRegex(exception, message):
                    replay.collect(guarded, policy, replay.cases[0], 0, 'a' * 64, OperationContext())
                self.assertEqual(guarded.guard['passed_subticks'], 1)

    def test_refresh_full_episodes_reward_budget_and_verified_resume(self):
        c = self.config_for_replay()
        _, full, _, report = self.train(c)
        expected = torch.load(full / 'latest.pt', weights_only=True)
        replay = report['replay']
        self.assertEqual(replay['effective_student_conditions'], 2)
        self.assertEqual(replay['declared_student_conditions'], 3)
        self.assertEqual(replay['student_control_steps'], 15)
        self.assertEqual(replay['original_plant_substeps'], 30)
        self.assertEqual(replay['extra_mpc_calls'], 0)
        self.assertEqual(replay['retained_steps'], 10)
        # x, next_x, u, previous_u, observation, noise, reference are all scalar.
        self.assertEqual(replay['retained_tensor_bytes'], 10*7*8)
        self.assertEqual(replay['capacity_tensor_bytes'], 10*7*8)
        self.assertEqual(replay['last_committed_checkpoint_bytes'], (full/'latest.pt').stat().st_size)
        self.assertEqual(len({e['policy_sha256'] for e in replay['retained_episodes']}), 2)
        self.assertTrue(any(e['post_update_action_delta_max'] > 0 for e in report['log']))
        for episode in expected['replay_state']['episodes']:
            self.assertEqual(len(episode['reference']), len(episode['x']))
            torch.testing.assert_close(episode['next_x'][:-1], episode['x'][1:], atol=0, rtol=0)
            torch.testing.assert_close(episode['u'][:-1], episode['previous_u'][1:], atol=0, rtol=0)
        events = []

        def stop(event):
            if event.get('stage') == 'pgac_student_episode_committed' and event['episode'] == 2:
                events.append(event)
                raise Cancelled('after one complete episode in epoch 2')

        with self.assertRaises(Cancelled):
            self.train(c, OperationContext(emit=stop))
        # Failed-stage archival moves the checkpoint from staging.
        paths = [
            p
            for p in (self.ws.path / 'failed').glob('*/latest.pt')
            if file_hash(p) == events[0]['checkpoint_sha256']
        ]
        self.assertEqual(len(paths), 1)
        ids = []

        def observe(event):
            if event.get('stage') == 'pgac_student_episode_committed':
                ids.append(event['episode'])

        _, resumed, _, _ = self.train(
            {**c, 'resume_checkpoint': str(paths[0]), 'resume_checkpoint_sha256': file_hash(paths[0])},
            OperationContext(emit=observe),
        )
        self.assertEqual(ids, [3, 4, 5])
        actual = torch.load(resumed / 'latest.pt', weights_only=True)
        self.assert_nested_equal(actual, expected)

    def test_reject_heldout_alias_and_accept_expert_only_companion_mode(self):
        c = self.config_for_replay()
        held = copy.deepcopy(c['original_evaluation']['scenarios'][0])
        held.pop('split')
        held['name'] = 'renamed'
        held['steps'] = 1
        c['replay']['scenarios'] = [held]
        with self.assertRaisesRegex(ContractError, 'overlaps'):
            self.train(c)
        c['replay'] = {
            'mode': 'expert_segments',
            'expert_families': {'0': 'up', '1': 'up', '2': 'down'},
            'updates_per_epoch': 2,
        }
        _, _, _, report = self.train(c)
        self.assertEqual(report['replay']['student_control_steps'], 0)
        self.assertEqual(report['optimizer_updates'], 6)

    def test_public_recovery_after_process_death_in_middle_of_refresh(self):
        from control_harness.case_plans import FIXTURE_KIND
        from control_harness.operations import run_operation
        from test_pgac_operations import PublicDifferentiableOperations

        root = self.ws.path
        atomic_json(root / '.case-plans/test-fixture.json', {'kind': FIXTURE_KIND, 'workspace': str(root)})
        request = {
            'operation': 'optimize_external_policy',
            'workspace': str(root),
            'inputs': {'policy': self.policy, 'model': self.model, 'validation': self.validation},
            'config': {**self.config_for_replay(), 'epochs': 1},
        }
        PublicDifferentiableOperations.approve(self, request)
        atomic_json(root / 'replay-request.json', request)
        child = """
import os,sys,torch
from pathlib import Path
from control_harness.artifacts import Workspace,read_json,atomic_json
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
root=Path(sys.argv[1]); torch.set_num_threads(1)
def stop(event):
    if event.get('stage') == 'pgac_student_episode_committed':
        atomic_json(root/'interrupted.json',{'stage':Path(event['checkpoint']).parent.name})
        os._exit(47)
run_operation(read_json(root/'replay-request.json'), OperationContext.test_fixture(Workspace(root),emit=stop))
"""
        result = subprocess.run(
            [sys.executable, '-B', '-c', child, str(root)],
            timeout=120,
            capture_output=True,
            text=True,
            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')},
        )
        self.assertEqual(result.returncode, 47, result.stderr + result.stdout)
        committed = []
        ctx = OperationContext.test_fixture(
            self.ws,
            emit=lambda e: committed.append(e['episode'])
            if e.get('stage') == 'pgac_student_episode_committed'
            else None,
        )
        interrupted = read_json(root / 'interrupted.json')
        recovered = run_operation(
            {
                'operation': 'recover_interrupted_stage',
                'workspace': str(root),
                'inputs': {'stage_id': interrupted['stage']},
                'config': {},
            },
            ctx,
        )
        self.assertEqual(recovered['optimizer_updates'], 0)
        request['config'].update(
            resume_checkpoint=recovered['checkpoint'], resume_checkpoint_sha256=recovered['checkpoint_sha256']
        )
        resumed = run_operation(request, ctx)
        self.assertEqual(committed, [1])
        report = read_json(Path(resumed['artifact_path']) / 'report.json')
        self.assertEqual(report['replay']['student_control_steps'], 5)
        self.assertEqual(report['optimizer_updates'], 2)

    def test_new_blind_conditions_keep_previous_expert_holdouts_out_of_online_pool(self):
        c = self.config_for_replay()
        prior = copy.deepcopy(c['original_evaluation']['scenarios'])
        c['original_evaluation'].update(
            condition_source='fresh_holdout',
            scenarios=[
                {**row, 'initial_state': [-1.0 + i * 0.1], 'name': 'fresh-' + str(i)}
                for i, row in enumerate(prior)
            ],
        )
        saved = c['replay']['scenarios']
        alias = {
            **prior[0],
            'name': 'old-evaluation-alias',
            'steps': 1,
            'noise_seed': 23,
            'observation_noise_std': [0.01],
        }
        alias.pop('split')
        c['replay']['scenarios'] = [alias]
        with self.assertRaisesRegex(ContractError, 'overlaps'):
            self.train(c)
        c['replay']['scenarios'] = saved
        c['epochs'] = 1
        _, _, _, report = self.train(c)
        self.assertEqual(report['validation_trajectory_ids'], [])
        self.assertEqual(report['expert_validation_trajectory_ids'], [3])
        self.assertEqual([row['name'] for row in report['checkpoint_validation_conditions']], ['fresh-0'])


if __name__ == '__main__':
    unittest.main()
