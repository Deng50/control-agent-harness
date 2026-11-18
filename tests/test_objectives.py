"""Frozen C03 arithmetic and independent C04 signed/MIMO/finite-record proofs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from control_harness.contracts import ContractError,ModelInterface
from control_harness.objectives import ATTRIBUTION,normalize_evaluation,trajectory_metrics,score_metrics


ROOT=Path(__file__).resolve().parents[1]
FROZEN=ROOT/'test-output/v2-development/20261117T002006/fixtures'
SCALES={'rmse':1.,'energy':100.,'tv':10.,'overshoot':.1}


def interface(ny=1,nu=1,dt=.5,output_units=None,control_units=None,output_bound=10.):
    variable=lambda name,unit,bound:{'name':name,'unit':unit,'lower':-bound,'upper':bound}
    return ModelInterface.parse({'schema_version':1,'name':'analytic objective fixture','sample_time':dt,
        'time_semantics':'discrete_state_transition','states':[variable('state','m',10)],
        'outputs':[variable('out'+str(k),(output_units or ['m']*ny)[k],output_bound) for k in range(ny)],
        'controls':[variable('act'+str(k),(control_units or ['N']*nu)[k],10.) for k in range(nu)],'disturbances':[]})


def run(outputs,references,controls=None,initial_y=None,initial_u=None):
    y=np.asarray(outputs,dtype=float)
    if y.ndim==1:y=y[:,None]
    ref=np.asarray(references,dtype=float)
    if ref.ndim==1:ref=np.repeat(ref[None,:],len(y),axis=0)
    controls=np.zeros((len(y),1)) if controls is None else np.asarray(controls,dtype=float)
    if controls.ndim==1:controls=controls[:,None]
    result={'scenario':{'initial_state':[0.]},'trajectory':[{'step':k,'next_x':[0.],
        'u':u.tolist(),'output':out.tolist(),'reference':r.tolist()} for k,(out,r,u) in enumerate(zip(y,ref,controls))],
        'metrics':{'rmse':0.,'energy':0.,'hard_constraint_violations':0},'constraint_checks':{'input_rate':False}}
    if initial_y is not None:result['initial_output']=initial_y
    if initial_u is not None:result['initial_control']=initial_u
    return result


def evaluation(**values):return {'recipe':'tracking','scales':copy.deepcopy(SCALES),**values}


class ObjectiveTests(unittest.TestCase):
    def test_C03_frozen_manual_oracle_exact_scores_rankings_and_hard_invalid_exclusion(self):
        path=FROZEN/'F05_expert_and_tradeoff_candidates.json'
        frozen=json.loads((FROZEN/'freeze_manifest.json').read_text(encoding='utf-8'))
        expected=next(row['sha256'] for row in frozen['files'] if row['path']=='fixtures/'+path.name)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        oracle=json.loads(path.read_text(encoding='utf-8'))['manual_scoring_oracle']
        self.assertEqual(oracle['scales'],SCALES)
        self.assertIn('NOT measured',oracle['provenance'])
        for recipe,weights in oracle['weights'].items():
            config=normalize_evaluation({'recipe':recipe,'weights':weights,'scales':oracle['scales']})
            self.assertEqual(normalize_evaluation({'recipe':recipe,'scales':SCALES})['weights'],weights)
            scores={}
            for candidate,metrics in oracle['candidates'].items():
                result=score_metrics(metrics,config)
                if candidate=='infeasible':
                    self.assertIsNone(result['score']);self.assertFalse(result['hard_constraints_passed']);continue
                self.assertTrue(result['hard_constraints_passed']);scores[candidate]=result['score']
                self.assertEqual(result['score'],oracle['expected'][recipe]['scores'][candidate])
            self.assertEqual(sorted(scores,key=scores.get),oracle['expected'][recipe]['ranking'])

    def test_normalization_is_idempotent_copies_input_and_keeps_search_scales_fixed(self):
        raw=evaluation();before=copy.deepcopy(raw)
        canonical=normalize_evaluation(raw)
        self.assertEqual(raw,before);self.assertEqual(normalize_evaluation(canonical),canonical)
        altered={'rmse':9.,'energy':1.,'tv':2.,'overshoot':0.,'hard_constraint_violations':0}
        score_metrics(altered,canonical)
        self.assertEqual(canonical['scales'],SCALES)
        canonical['weights']['rmse']=7.
        self.assertEqual(normalize_evaluation(raw)['weights']['rmse'],10.)

    def test_invalid_recipes_scales_weights_and_unknown_metric_definitions_fail_before_search(self):
        patches=[{'recipe':'auto'}, {'scales':None},{'scales':{**SCALES,'energy':0.}}, {'scales':{**SCALES,'rmse':True}},
            {'scales':{**SCALES,'tv':float('nan')}},{'scales':{**SCALES,'overshoot':'train_std'}},
            {'weights':dict.fromkeys(SCALES,0.)},{'weights':{'rmse':1.}}, {'weights':{**dict.fromkeys(SCALES,1.),'tv':-1.}},
            {'energy_model':'real_power'},{'hard_gates':{'stability':True}}, {'hard_gates':{'max_oscillation_crossings':1.5}},
            {'metrics':{'settling_hold_steps':False}},{'metrics':{'oscillation_rel_deadband':-1}}, {'candidate_normalization':True}]
        for patch in patches:
            with self.subTest(patch=patch),self.assertRaises(ContractError):normalize_evaluation(evaluation(**patch))

    def test_analytic_RMSE_energy_integral_TV_and_settling_are_independently_recomputed(self):
        sample=run([.8,1.2,1.,1.],[1.],[2.,1.,-1.,0.],initial_y=[0.],initial_u=[1.])
        config=evaluation(metrics={'settling_abs_band':.05,'settling_rel_band':0.,'settling_hold_steps':2,
            'oscillation_abs_deadband':.01,'oscillation_rel_deadband':0.})
        values=trajectory_metrics(sample,interface(),config)
        self.assertAlmostEqual(values['rmse'],math.sqrt(.02),places=14)
        self.assertEqual(values['energy'],3.) # .5 * (4+1+1+0), no actual power claim.
        self.assertEqual(values['tv'],5.) # |2-1|+|1-2|+|-1-1|+|0+1|.
        self.assertAlmostEqual(values['overshoot'],.2,places=14)
        self.assertEqual(values['settling_time'],1.5)
        self.assertEqual(values['oscillation_crossings'],1)
        self.assertEqual(values['energy_kind'],'action_proxy')
        self.assertIn('not measured power',values['metric_definitions']['energy'])
        self.assertEqual(values['per_control'][0]['energy_proxy_unit'],'N^2*s')
        self.assertAlmostEqual(score_metrics(values,config)['score'],10*math.sqrt(.02)+.1*3/100+.1*5/10,places=13)

    def test_C04_MIMO_up_down_and_zero_steps_have_signed_per_output_metrics_and_units(self):
        sample=run([[.5,1.,0.],[1.2,-.4,0.],[1.,0.,0.],[1.,0.,0.]], [1.,0.,0.],
            [[1.,2.]]*4,initial_y=[0.,2.,0.],initial_u=[0.,0.])
        config=evaluation(metrics={'settling_abs_band':.05,'settling_rel_band':0.,'settling_hold_steps':2})
        values=trajectory_metrics(sample,interface(3,2,1.,['m','Pa','1'],['N','A']),config)
        channels=values['per_output']
        self.assertAlmostEqual(channels[0]['overshoot_ratio'],.2,places=14)
        self.assertAlmostEqual(channels[1]['overshoot_ratio'],.2,places=14)
        self.assertEqual(channels[1]['segments'][0]['step_direction'],-1.)
        self.assertEqual(channels[1]['segments'][0]['step_amplitude'],2.)
        self.assertIsNone(channels[2]['overshoot_ratio']);self.assertIsNone(channels[2]['settling_time'])
        self.assertEqual(channels[2]['step_metrics_reason'],'no_step')
        self.assertEqual(values['settling_time'],3.)
        self.assertEqual([item['energy_proxy'] for item in values['per_control']],[4.,16.])
        self.assertTrue(values['metric_definitions']['mixed_output_units']);self.assertTrue(values['metric_definitions']['mixed_control_units'])
        self.assertIn('numeric coordinate indices',values['metric_definitions']['aggregate_units'])

    def test_reference_changes_use_previous_target_and_reset_crossing_count_at_boundaries(self):
        sample=run([.7,1.1,0.,-1.4,-1.,-1.],[[1.],[1.],[-1.],[-1.],[-1.],[-1.]],initial_y=[0.],initial_u=[0.])
        values=trajectory_metrics(sample,interface(),evaluation(metrics={'settling_abs_band':.05,'settling_rel_band':0.,'settling_hold_steps':2}))
        segments=values['per_output'][0]['segments']
        self.assertEqual(segments[1]['origin'],1.) # Not the previous actual output 1.1.
        self.assertAlmostEqual(values['overshoot'],.2,places=14)
        self.assertIsNone(values['settling_time']) # First commanded segment never settled.
        boundary=run([1.1,1.1,1.1,-1.1,-1.1,-1.1],[[1.]]*3+[[-1.]]*3,initial_y=[0.],initial_u=[0.])
        self.assertEqual(trajectory_metrics(boundary,interface(),evaluation())['oscillation_crossings'],0)

    def test_negative_target_step_normalizes_by_absolute_amplitude(self):
        values=trajectory_metrics(run([-1.,-2.5,-2.,-2.],[-2.],initial_y=[0.],initial_u=[0.]),interface(),evaluation())
        self.assertAlmostEqual(values['overshoot'],.25,places=14)
        self.assertEqual(values['per_output'][0]['segments'][0]['step_direction'],-1.)

    def test_zero_step_is_not_falsely_zero_overshoot_or_a_stability_certificate(self):
        values=trajectory_metrics(run([2.]*4,[2.],initial_y=[2.],initial_u=[0.]),interface(),evaluation())
        self.assertIsNone(values['overshoot']);self.assertIsNone(values['settling_time'])
        self.assertEqual(values['oscillation_crossings'],0)
        self.assertIn('not_assessed',values['metric_definitions']['stability'])
        self.assertIsNotNone(score_metrics(values,evaluation())['score'])
        weighted=evaluation(weights={'rmse':1.,'energy':0.,'tv':0.,'overshoot':1.})
        self.assertIsNone(score_metrics(values,weighted)['score'])
        gated=evaluation(hard_gates={'per_output':{'out0':{'max_overshoot':.1}}})
        self.assertFalse(score_metrics(values,gated)['hard_constraints_passed'])

    def test_missing_initial_output_never_uses_post_first_output_as_step_baseline(self):
        values=trajectory_metrics(run([.8,1.2,1.,1.],[1.],initial_u=[0.]),interface(),evaluation())
        self.assertFalse(values['initial_output_available']);self.assertIsNone(values['overshoot'])
        self.assertEqual(values['per_output'][0]['segments'][0]['overshoot_reason'],'missing_initial_output')
        self.assertIsNone(score_metrics(values,evaluation(hard_gates={'max_overshoot':10.}))['score'])

    def test_missing_initial_control_is_TV_NA_and_requested_rate_without_bounds_fails_closed(self):
        sample=run([.8,1.2,1.,1.],[1.],[2.,1.,-1.,0.],initial_y=[0.])
        values=trajectory_metrics(sample,interface(),evaluation())
        self.assertIsNone(values['tv']);self.assertIsNone(score_metrics(values,evaluation())['score'])
        sample['constraint_checks']['input_rate']=True
        sample['metrics']['max_rate_violation']=0. # Untrusted legacy claim is not sufficient.
        values=trajectory_metrics(sample,interface(),evaluation())
        self.assertFalse(values['hard_constraints_complete'])
        self.assertIsNone(score_metrics(values,evaluation(weights={'rmse':1.,'energy':0.,'tv':0.,'overshoot':0.}))['score'])

    def test_actual_amplitude_state_output_and_rate_breaches_are_hard_excluded(self):
        sample=run([.8,1.2,1.,1.],[1.],[2.,1.,-1.,0.],initial_y=[0.],initial_u=[1.])
        sample['input_rate_bounds']=[[-1.5,1.5]];sample['constraint_checks']['input_rate']=True
        values=trajectory_metrics(sample,interface(),evaluation())
        self.assertEqual(values['constraints']['input_rate']['count'],1)
        self.assertEqual(values['max_rate_violation'],.5)
        self.assertIsNone(score_metrics(values,evaluation())['score'])
        sample['input_rate_bounds']=[[-2.,2.]];sample['trajectory'][0]['next_x']=[20.]
        values=trajectory_metrics(sample,interface(),evaluation())
        self.assertGreater(values['hard_constraint_violations'],0);self.assertEqual(values['max_state_violation'],10.)
        self.assertFalse(score_metrics(values,evaluation())['hard_constraints_passed'])

    def test_named_output_and_scalar_gates_use_raw_units_and_metric_availability(self):
        values=trajectory_metrics(run([.8,1.2,1.,1.],[1.],initial_y=[0.],initial_u=[0.]),interface(),evaluation())
        self.assertIsNone(score_metrics(values,evaluation(hard_gates={'per_output':{'out0':{'max_rmse':.1}}}))['score'])
        self.assertIsNone(score_metrics(values,evaluation(hard_gates={'max_settling_time':1.}))['score'])
        with self.assertRaisesRegex(ContractError,'unknown outputs'):
            trajectory_metrics(run([1.]*4,[1.],initial_y=[0.],initial_u=[0.]),interface(),evaluation(hard_gates={'per_output':{'not_real':{'max_rmse':1.}}}))

    def test_settling_requires_observed_suffix_hold_and_never_claims_zero_delay(self):
        sample=run([.9,1.,1.],[1.],initial_y=[0.],initial_u=[0.])
        self.assertIsNone(trajectory_metrics(sample,interface(),evaluation())['settling_time'])
        value=trajectory_metrics(sample,interface(),evaluation(metrics={'settling_hold_steps':2}))
        self.assertEqual(value['settling_time'],1.)
        all_inside=trajectory_metrics(run([1.]*3,[1.],initial_y=[0.],initial_u=[0.]),interface(),evaluation())
        self.assertEqual(all_inside['settling_time'],.5)

    def test_oscillation_is_an_explicit_deadband_crossing_diagnostic(self):
        sample=run([1.1,.9,1.08,.92],[1.],initial_y=[0.],initial_u=[0.])
        loose=trajectory_metrics(sample,interface(),evaluation(metrics={'oscillation_abs_deadband':.09,'oscillation_rel_deadband':0.}))
        tight=trajectory_metrics(sample,interface(),evaluation(metrics={'oscillation_abs_deadband':.05,'oscillation_rel_deadband':0.}))
        self.assertEqual(loose['oscillation_crossings'],1);self.assertEqual(tight['oscillation_crossings'],3)
        short=trajectory_metrics(run([.9,1.1],[1.],initial_y=[0.],initial_u=[0.]),interface(),evaluation())
        self.assertIsNone(short['oscillation_crossings'])

    def test_nonfinite_and_overflow_metrics_have_null_score_and_valid_JSON(self):
        sample=run([.8,1.2,1.,1.],[1.],initial_y=[0.],initial_u=[0.]);sample['trajectory'][1]['u']=[float('nan')]
        values=trajectory_metrics(sample,interface(),evaluation());self.assertFalse(values['finite'])
        result=score_metrics(values,evaluation());self.assertIsNone(result['score'])
        json.dumps({'metrics':values,'score':result},allow_nan=False)
        huge=run([1e308]*4,[1e308],initial_y=[-1e308],initial_u=[0.])
        values=trajectory_metrics(huge,interface(output_bound=1.7e308),evaluation())
        self.assertFalse(values['finite']);json.dumps(values,allow_nan=False)
        scores=score_metrics({'rmse':1e308,'energy':1.,'tv':1.,'overshoot':0.,'hard_constraint_violations':0},
            evaluation(scales={**SCALES,'rmse':1e-308}))
        self.assertIsNone(scores['score']);json.dumps(scores,allow_nan=False)

    def test_malformed_shapes_and_excessive_reference_detail_are_explicit_resource_errors(self):
        sample=run([1.]*4,[1.],initial_y=[0.],initial_u=[0.]);sample['trajectory'][0]['output']=[[1.]]
        with self.assertRaises(ContractError):trajectory_metrics(sample,interface(),evaluation())
        n=10001;sample=run([0.]*n,[[float(k%2)] for k in range(n)],initial_y=[0.],initial_u=[0.])
        with self.assertRaisesRegex(ContractError,'reference-segment'):
            trajectory_metrics(sample,interface(),evaluation())
        near=run([1.]*n,[[1.+1.5e-12*(k%2)] for k in range(n)],initial_y=[0.],initial_u=[0.])
        with self.assertRaisesRegex(ContractError,'reference-segment'):
            trajectory_metrics(near,interface(),evaluation())

    def test_attribution_records_the_actual_v5_source_bytes_when_available(self):
        self.assertEqual(ATTRIBUTION['function'],'canonical_tracking_metrics')
        source=ROOT.parent/'mpc_agent_v5/app/performance.py'
        if source.is_file():self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),ATTRIBUTION['sha256'])


if __name__=='__main__':unittest.main()
