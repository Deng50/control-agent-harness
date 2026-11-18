"""N05 independent actual-consumer intervention tests; no loader substitutes."""
import importlib.util
import os
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('n05_residual_consumption_probe',ROOT/'scripts/verify-residual-consumption.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
OUTPUT=Path(os.environ.get('CONTROL_TEST_OUTPUT',ROOT/'test-output/residual-consumption-tests'))


class ResidualConsumptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True,exist_ok=True)
        cls.path=OUTPUT/('n05-'+uuid.uuid4().hex[:8])
        def cleanup():
            assert cls.path.resolve().is_relative_to(OUTPUT.resolve())
            shutil.rmtree(cls.path,ignore_errors=True)
        cls.addClassCleanup(cleanup)
        cls.result=probe.run_evidence(cls.path)

    def test_frozen_f01_f02_residual_is_consumed_by_real_mpc_cost_and_interior_action(self):
        for name,row in self.result['results'].items():
            with self.subTest(fixture=name):
                a,b=row['mpc']['variants']
                self.assertGreater(abs(b['fixed_sequence_cost']-a['fixed_sequence_cost']),1e-7)
                self.assertGreater(np.max(np.abs(np.array(b['action'])-a['action'])),1e-4)
                self.assertTrue(all(min(x['action_bound_margin'])>.01 for x in (a,b)))
                self.assertTrue(all(x['solve']['constraint_violation']<=1e-6 for x in (a,b)))

    def test_actual_policy_optimizer_has_changed_loss_and_actor_gradient_without_loader_mock(self):
        self.assertTrue(self.result['no_model_loader_mock'])
        for name,row in self.result['results'].items():
            with self.subTest(fixture=name):
                policy=row['policy'];a,b=policy['variants']
                self.assertTrue(policy['same_actor_state_and_scalers'] and policy['same_expert_inputs'])
                self.assertEqual(policy['expert_action_loss_weight'],0.)
                self.assertLess(policy['expert_action_roundoff_max'],1e-12)
                self.assertGreater(abs(b['loss']-a['loss']),1e-7)
                self.assertGreater(policy['first_gradient_max_change'],1e-8)
                self.assertTrue(all(x['actual_adam_steps']>0 and x['action_margin']>1e-5 for x in (a,b)))

    def test_mpc_and_actual_actor_gradients_match_centered_differences_at_three_steps(self):
        for name,row in self.result['results'].items():
            for consumer in ('mpc','policy'):
                for variant in row[consumer]['variants']:
                    with self.subTest(fixture=name,consumer=consumer):
                        self.assertEqual([x['h'] for x in variant['finite_difference']],[1e-4,1e-5,1e-6])
                        for value in variant['finite_difference']:
                            np.testing.assert_allclose(value['value'],value['autograd'],atol=1e-5,rtol=1e-3)

    def test_no_extra_dt_and_actual_legacy_two_model_residual_training_route(self):
        for row in self.result['results'].values():
            np.testing.assert_allclose(row['mpc']['observed_prediction_delta'],row['mpc']['intervention_bias'],atol=1e-15,rtol=0)
            self.assertFalse(row['mpc']['extra_dt_applied'])
            self.assertEqual(row['mpc']['residual_semantics'],'next_state_residual_in_state_units')
        old=self.result['legacy_route']
        self.assertTrue(old['execution_completed'] and old['derivative_residual_rejected'])
        self.assertEqual(old['trained_epochs'],4)
        self.assertGreater(old['optimizer_updates'],0)
        self.assertNotEqual(old['nominal_id'],old['reference_id'])
        self.assertEqual(old['algorithm_name_unchanged'],'supervised_discrete_residual_with_declared_physical_constraints')


if __name__=='__main__':unittest.main()
