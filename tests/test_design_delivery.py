"""验证交付证据的分层、独立运行和跨角色工作区控制器。无需模型服务。"""
import hashlib
import shutil
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.contracts import ContractError, MaterialInterface, ModelInterface
from control_harness.control import load_controller
from control_harness.delivery import verify_design_delivery
from control_harness.design_only import synthesize_mpc
from control_harness.examples import example
from control_harness.modeling import register_model
from control_harness.policy import collect_expert
from control_harness.reporting import artifact_delivery
from control_harness.runtime import Cancelled, OperationContext


class DesignDeliveryTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / 'test-output' / 'design-delivery'
        parent.mkdir(parents=True, exist_ok=True)
        self.ws = Workspace(parent / uuid.uuid4().hex)
        self.addCleanup(shutil.rmtree, self.ws.path)
        self.spec = example('nonlinear')['reference']
        self.model = register_model(self.ws, self.spec, {'kind': 'user_nominal'})
        self.ready = synthesize_mpc(self.ws, {'model': self.model}, {'controller': {'horizon': 2}, 'solver_check': False})
        self.case = {'x': [.1, -.05], 'reference': [.03], 'disturbance': [.01], 'previous': [0.]}

    def verify(self, target='standalone_python', **kwargs):
        return verify_design_delivery(self.ws, self.ready, {'target': target, 'case': self.case, **kwargs})

    def test_explicit_promotion_matches_real_loader_and_preserves_design_parent(self):
        old = artifact_delivery(self.ws, self.ready)
        self.assertIsNone(old['request_satisfied'])
        self.assertEqual(old['checks']['standalone'], 'not_run')
        artifact = self.verify('workspace_controller')
        root, meta = self.ws.resolve(artifact, kind='controller')
        report = read_json(root / 'report.json')
        verified = artifact_delivery(self.ws, artifact)
        self.assertTrue(verified['request_satisfied'])
        self.assertEqual({p['id'] for p in meta['parents']}, {self.ready, self.model})
        self.assertEqual(verified['bundle_artifact'], self.ready)
        self.assertEqual(verified['checks']['closed_loop'], 'not_run')
        self.assertNotIn('controller', self.ws.manifest['active'])
        self.assertEqual(read_json(root / 'controller.json')['prediction_model'], self.model)
        policy = load_controller(self.ws, artifact)
        action, _ = policy.act(self.case['x'], self.case['reference'], self.case['disturbance'], self.case['previous'])
        np.testing.assert_allclose(action, report['evidence']['standalone_result']['action'], rtol=0, atol=1e-6)
        # 单独登记评价对象后，经现有轻量化采数入口完成五个独立单步工况。
        # 这个冒烟只证明接口能协作，长时间跟踪及部署性能另行验收。
        plant = register_model(self.ws, self.spec, {'kind': 'reference_simulator'})
        scenarios = [{'name': f'condition-{k}', 'initial_state': [.01*k, -.02], 'reference': [.03],
                      'disturbance': [.01], 'steps': 1} for k in range(5)]
        dataset = collect_expert(self.ws, artifact, plant, {'scenarios': scenarios})
        data_root, data_meta = self.ws.resolve(dataset, kind='expert_dataset')
        self.assertEqual({p['id'] for p in data_meta['parents']}, {artifact, plant})
        with np.load(data_root / 'expert.npz', allow_pickle=False) as arrays:
            self.assertEqual(len(np.unique(arrays['trajectory_id'])), 5)
            self.assertEqual(arrays['u'].shape, (5, 1))
        self.assertFalse(list((self.ws.path / '.staging').glob('delivery-*')))
        index = self.ws.manifest
        index['artifacts'][self.model]['status'] = 'stale'
        atomic_json(self.ws.manifest_path, index)
        stale = artifact_delivery(self.ws, artifact)
        self.assertFalse(stale['request_satisfied'])
        self.assertIn('current_dependencies', stale['missing_requirements'])

    def test_requested_unmeasured_results_and_mismatched_action_are_visible(self):
        partial = artifact_delivery(self.ws, self.verify('workspace_controller', required_checks=['closed_loop', 'matlab', 'simulink']))
        self.assertEqual(partial['status'], 'partial')
        self.assertEqual(partial['missing_requirements'], ['closed_loop', 'matlab', 'simulink'])
        self.assertEqual(partial['checks']['standalone'], 'passed')
        self.assertFalse(partial['request_satisfied'])
        failed = artifact_delivery(self.ws, self.verify(expected_action=[200.], action_atol=.1))
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('expected_action', failed['missing_requirements'])
        self.assertEqual(failed['checks']['single_point'], 'passed')

    def test_draft_and_omitted_case_never_solve_or_promote(self):
        draft = synthesize_mpc(self.ws, {'description': '弹簧阻尼'}, {})
        with patch('control_harness.delivery._probe', side_effect=AssertionError('不应启动子进程')):
            for design, config, missing in [(draft, {'target': 'standalone_python', 'case': self.case}, 'mass'),
                                           (self.ready, {'target': 'workspace_controller'}, '工况')]:
                receipt = verify_design_delivery(self.ws, design, config)
                delivery = artifact_delivery(self.ws, receipt)
                self.assertIn(missing, delivery['details'])
                self.assertEqual(delivery['checks']['load'], 'not_run')
                self.assertFalse(delivery['request_satisfied'])

    def test_self_consistent_modified_template_never_executes(self):
        original, _ = self.ws.resolve(self.ready)
        with self.ws.stage('controller_design', MaterialInterface('bundle'), [self.model], {}, compatible_parents=False) as stage:
            shutil.copytree(original / 'bundle', stage.root / 'bundle')
            shutil.copyfile(original / 'design.json', stage.root / 'design.json')
            script = stage.root / 'bundle' / 'controller.py'
            script.write_text('# Foreign template version\n' + script.read_text(encoding='utf-8'), encoding='utf-8')
            manifest = stage.root / 'bundle' / 'bundle_manifest.json'
            payload = read_json(manifest)
            payload['files']['controller.py'] = hashlib.sha256(script.read_bytes()).hexdigest()
            atomic_json(manifest, payload)
            changed = stage.publish({'status': 'verified'}, activate=False)
        with patch('control_harness.delivery._probe', side_effect=AssertionError('不应执行变动模板')):
            receipt = verify_design_delivery(self.ws, changed, {'target': 'standalone_python', 'case': self.case})
        self.assertIn('模板', artifact_delivery(self.ws, receipt)['details'])
        (original / 'bundle' / 'controller.py').write_text('# changed', encoding='utf-8')
        with self.assertRaises(ContractError):
            self.verify()

    def test_cancel_during_independent_process_cleans_snapshot_without_receipt(self):
        cancel = {'active': False}
        context = OperationContext(cancelled=lambda: cancel['active'])
        context.emit = lambda event: cancel.update(active=True)
        with self.assertRaises(Cancelled):
            verify_design_delivery(self.ws, self.ready, {'target': 'standalone_python', 'case': self.case}, context)
        self.assertFalse(list((self.ws.path / '.staging').glob('delivery-*')))
        self.assertFalse(any(a['kind'] == 'delivery_validation' for a in self.ws.manifest['artifacts'].values()))

    def test_missing_bundle_is_not_reported_as_a_portable_design(self):
        with self.ws.stage('controller_design', MaterialInterface('bundle'), [], {}) as stage:
            atomic_json(stage.root / 'design.json', {'schema_version': 1, 'solver_check': {'status': 'passed'}})
            incomplete = stage.publish({'status': 'verified'}, activate=False)
        delivery = artifact_delivery(self.ws, incomplete)
        self.assertEqual(delivery['status'], 'partial')
        self.assertEqual(delivery['checks']['files'], 'failed')
        self.assertEqual(delivery['checks']['single_point'], 'not_run')
        self.assertIsNone(delivery['bundle_artifact'])

    def test_sampled_fit_rejects_requested_extrapolation_and_predicted_actions(self):
        interval = lambda lo, hi: {'min': lo, 'max': hi}
        limits = {'x': interval([-.15, -.15], [.15, .15]), 'u': interval([-.00001], [.00001]),
                  'd': interval([0.], [.03])}
        interface = ModelInterface.parse(self.spec['interface'])
        with self.ws.stage('identified_model', interface, [self.model], {}) as stage:
            atomic_json(stage.root / 'model.json', self.spec)
            model = stage.publish({'status': 'verified'}, source={'kind': 'plant_predictor_conversion',
                'method': 'sampled_affine', 'applicability': {'scope': 'affine_approximation_within_sampled_conditions_only',
                                                              'observed_training_ranges': limits}}, activate=False)
        design = synthesize_mpc(self.ws, {'model': model}, {'controller': {'horizon': 2}, 'solver_check': False})
        for case, label in [({**self.case, 'x': [.2, -.05]}, '指定工况'), (self.case, '预测序列')]:
            receipt = verify_design_delivery(self.ws, design, {'target': 'workspace_controller', 'case': case})
            outcome = artifact_delivery(self.ws, receipt)
            self.assertEqual(outcome['status'], 'partial')
            self.assertIn('sampled_domain', outcome['missing_requirements'])
            self.assertIn(label, outcome['details'])
            self.assertFalse(outcome['workspace_controller_published'])

    def test_ordinary_artifact_with_receipt_filename_cannot_claim_acceptance(self):
        with self.ws.stage('material', MaterialInterface('json'), [], {}) as stage:
            atomic_json(stage.root / 'delivery.json', {'request_satisfied': True, 'status': 'artifact_ready'})
            artifact = stage.publish({'status': 'verified'}, activate=False)
        self.assertIsNone(artifact_delivery(self.ws, artifact)['request_satisfied'])
        with self.ws.stage('material', MaterialInterface('json'), [self.ready], {}, compatible_parents=False) as stage:
            atomic_json(stage.root / 'delivery.json', {'request_satisfied': True})
            forged = stage.publish({'status': 'verified'},
                                   source={'kind': 'design_delivery_verification', 'design_artifact': self.ready}, activate=False)
        with self.assertRaisesRegex(ContractError, '产物类型'):
            artifact_delivery(self.ws, forged)
