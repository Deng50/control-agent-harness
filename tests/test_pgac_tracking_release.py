"""Engineering acceptance follows the requested physical output, even when J favors another policy."""
import unittest
from types import SimpleNamespace

from control_harness.contracts import ContractError
from control_harness.external_pgac_evaluation import evaluation_config, final_acceptance, validation_choice, _verify_native_tracking
from control_harness.pgac_tracking import tracking_gate, tracking_index, tracking_summary


OBJECTIVE = dict(q=1., r=.2, s=.1, primary_metric='tracking_rmse', min_relative_improvement=.01,
                 max_worst_case_relative_regression=.05, max_action_rms_relative_increase=.15,
                 max_increment_rms_relative_increase=.25)


def score(rmses, *, j, action=.1, increment=.05):
    rows = [dict(trajectory_id=k, tracking_rmse=x, action_rms=action, increment_rms=increment)
            for k, x in enumerate(rmses)]
    return dict(score=j, feasible=True, tracking=tracking_summary(rows))


class TrackingRelease(unittest.TestCase):
    def test_contract_requires_both_budgets_and_unambiguous_physical_output(self):
        interface = SimpleNamespace(outputs=[SimpleNamespace(name='temperature'), SimpleNamespace(name='pressure')])
        for key in ['max_action_rms_relative_increase','max_increment_rms_relative_increase']:
            with self.assertRaises(ContractError):
                evaluation_config({k:v for k,v in OBJECTIVE.items() if k!=key})
            for invalid in (-.2,False,float('nan')):
                with self.assertRaises(ContractError):
                    evaluation_config({**OBJECTIVE,key:invalid})
        for metric in ('quadratic','unregistered'):
            with self.assertRaises(ContractError):
                evaluation_config({**OBJECTIVE,'primary_metric':metric})
        with self.assertRaises(ContractError):
            tracking_index(OBJECTIVE,interface)
        with self.assertRaises(ContractError):
            tracking_index({**OBJECTIVE,'tracking_output':'unknown'},interface)
        self.assertEqual(tracking_index({**OBJECTIVE,'tracking_output':'pressure'},interface),1)

    def test_selection_and_frozen_release_follow_rmse_and_both_budgets_before_j(self):
        baseline = score([1.,1.],j=.05)
        passing = score([.8,1.01],j=.08)
        scores = {'candidate':passing,'imitation':baseline}
        frozen = validation_choice(scores,OBJECTIVE)
        self.assertEqual(frozen['choice'],'candidate')
        self.assertEqual(final_acceptance(frozen,scores,OBJECTIVE)['selected_version'],'candidate')
        for rejected,reason in [
            (score([1.01,1.01],j=.03),'minimum_improvement'),
            (score([.5,1.06],j=.03),'worst_case_tracking'),
            (score([.8,.8],action=.116,j=.03),'action_rms'),
            (score([.8,.8],increment=.063,j=.03),'increment_rms')]:
            scores = {'candidate':rejected,'imitation':baseline}
            decision = validation_choice(scores,OBJECTIVE)
            self.assertFalse(decision['tracking_comparison']['checks'][reason])
            self.assertEqual(decision['choice'],'imitation')
            self.assertEqual(final_acceptance(frozen,scores,OBJECTIVE)['selected_version'],'imitation')
        self.assertFalse(final_acceptance(frozen,{'imitation':{**baseline,'feasible':False,'tracking':None},
                                                 'candidate':passing},OBJECTIVE)['approved'])

    def test_physical_rmse_is_macro_averaged_and_zero_cost_is_not_a_free_budget(self):
        base = score([0.,2.],j=1.,action=0.,increment=0.)['tracking']
        self.assertEqual(base['tracking_rmse'],1.)
        cand = score([0.,1.8],j=1.,action=.000001,increment=0.)['tracking']
        self.assertFalse(tracking_gate(base,cand,OBJECTIVE)['checks']['action_rms'])
        mismatched = [{**r,'trajectory_id':r['trajectory_id']+1} for r in cand['per_scenario']]
        self.assertFalse(tracking_gate(base,tracking_summary(mismatched),OBJECTIVE)['passed'])

    def test_native_metrics_are_cross_checked_per_trajectory(self):
        expected = score([1.,2.],j=.3)['tracking']
        _verify_native_tracking(expected,expected)
        tampered = {**expected,'per_scenario':list(reversed(expected['per_scenario']))}
        with self.assertRaises(ContractError):
            _verify_native_tracking(tampered,expected)
        with self.assertRaises(ContractError):
            _verify_native_tracking({**expected,'action_rms':0.},expected)

    def test_relative_limits_hold_for_tiny_physical_units_and_stationary_actuators(self):
        base = score([1e-13], j=1., action=0., increment=0.)['tracking']
        better = score([.96e-13], j=1., action=0., increment=0.)['tracking']
        self.assertTrue(tracking_gate(base,better,OBJECTIVE,validation=True)['passed'])
        worse = score([6e-13], j=1., action=0., increment=1e-15)['tracking']
        checks = tracking_gate(base,worse,OBJECTIVE,validation=False)['checks']
        self.assertFalse(checks['minimum_improvement'])
        self.assertFalse(checks['worst_case_tracking'])
        self.assertFalse(checks['increment_rms'])
