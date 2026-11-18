"""R01/R02 real numeric comparisons and an explicitly fixed synthetic proxy fit."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from control_harness.artifacts import Workspace,atomic_json,read_json
from control_harness.contracts import ContractError,ModelInterface
from control_harness.differentiable_validation import (validate_pair,validate_differentiable_model,
    read_differentiable_validation,resolve_validation_config)
from control_harness.material_registration import register_material
from control_harness.modeling import DeclarativeDynamics,register_model
from control_harness.plants import register_plant,sample_model
from control_harness.runtime import Cancelled,OperationContext
from control_harness.sampling import recommend_sampling
from control_harness.data import trajectory_split
from control_harness.predictive import train_predictive_model


PROJECT=Path(__file__).resolve().parents[1]
FROZEN=PROJECT/'test-output/v2-development/20261117T002006/fixtures'
OUTPUT=PROJECT/'test-output/differentiable-validation-tests';OUTPUT.mkdir(parents=True,exist_ok=True)


def fixture_context(workspace):
    atomic_json(workspace.path/'.case-plans/test-fixture.json',{'kind':'control-case-plan-test-fixture-v1','workspace':str(workspace.path),'confirmation_source':'test_fixture','synthetic':True})
    return OperationContext.test_fixture(workspace)


class ScalarRowReference:
    def __init__(self,spec):
        self.spec=spec;self.interface=ModelInterface.parse(spec['interface']);self.calls=0
    def __call__(self,x,u,d):
        self.calls+=1;s=self.spec
        return np.asarray([sum(a*b for a,b in zip(row,x))+sum(a*b for a,b in zip(s['B'][j],u))+sum(a*b for a,b in zip(s['E'][j],d)) for j,row in enumerate(s['A'])],dtype=np.float64)
    def output(self,x):return np.asarray([sum(a*b for a,b in zip(row,x)) for row in self.spec['C']],dtype=np.float64)


class SpringReference:
    def __init__(self,spec):self.interface=DeclarativeDynamics(spec).interface;self.calls=0
    def __call__(self,x,u,d):
        self.calls+=1;dt=self.interface.sample_time;alpha=.4;omega=math.sqrt(2.-alpha*alpha)
        phi=math.exp(-alpha*dt)*(math.cos(omega*dt)*np.eye(2)+math.sin(omega*dt)/omega*np.asarray([[alpha,1.],[-2.,-alpha]]))
        equilibrium=np.array([(u[0]+d[0])/2.,0.]);return equilibrium+phi@(x-equilibrium)
    def output(self,x):return np.array([x[0]],dtype=np.float64)


class Variant(nn.Module):
    def __init__(self,base,kind):super().__init__();self.base=base;self.interface=base.interface;self.kind=kind
    def forward(self,x,u,d):
        if self.kind=='mixed_batch' and x.ndim==2:return self.base(x,u[:1].expand_as(u),d)
        value=self.base(x,u,d)
        if self.kind=='wrong_numeric':return value+.002
        if self.kind=='detached':return value.detach()+x.sum(dim=-1,keepdim=True)*0.
        if self.kind=='float32':return value.float()
        return value
    def output(self,x):return self.base.output(x)


def pure_config(interface,*,steps=50):
    trajectories=[]
    for group,phase in [(101,0.),(201,.7)]:
        if interface.nx==4:x0=[.1+phase*.05,-.2,.3,-.1]
        else:x0=[.12+phase*.03,-.08]
        controls=[[.12*math.sin(.19*k+phase)+(.03 if j==0 else -.02) for j in range(interface.nu)] for k in range(steps)]
        disturbance=[[.03*math.cos(.11*k+phase) for _ in range(interface.nd)] for k in range(steps)]
        trajectories.append({'id':group,'initial_state':x0,'u':controls,'d':disturbance})
    return {'mode':'exact_rewrite','steps':steps,'probes':[{'row':3,'kind':'smooth'},{'row':steps+7,'kind':'smooth'}],
        'numeric':{'atol':1e-6,'rtol':1e-5},'gradient':{'atol':1e-5,'rtol':1e-3,'finite_difference_steps':[1e-4,1e-5,1e-6]},'trajectories':trajectories}


class DifferentiablePairTests(unittest.TestCase):
    confirmation_source='test_fixture'
    @classmethod
    def setUpClass(cls):
        cls.spring_spec=read_json(FROZEN/'F01_spring_2x1.json')['source_model']
        cls.linear_spec=read_json(FROZEN/'F02_existing_linear_4x2.json')['nominal']
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_f01_exact_zoh_50step_mixed_batch_and_all_physical_jacobians(self):
        model=DeclarativeDynamics(self.spring_spec);reference=SpringReference(self.spring_spec)
        result=validate_pair(reference,model,pure_config(model.interface));r=result.report
        self.assertTrue(r['qualified_for_pgac']);self.assertTrue(r['exact_equivalence_pass']);self.assertTrue(r['gradient_checks_passed'])
        self.assertEqual([t['completed_steps'] for t in r['trajectories']],[50,50]);self.assertTrue(all(not t['teacher_forcing_after_initial_state'] for t in r['trajectories']))
        self.assertLess(max(r['numeric']['rollout_state']['max_abs_error']),2e-14)
        # Independent analytic Phi / equilibrium derivative, not a domain Jacobian.
        alpha=.4;dt=.2;omega=math.sqrt(2-alpha*alpha)
        phi=math.exp(-alpha*dt)*(math.cos(omega*dt)*np.eye(2)+math.sin(omega*dt)/omega*np.array([[alpha,1.],[-2.,-alpha]]))
        held=(np.eye(2)-phi)@np.array([.5,0.]);expected=np.column_stack((phi,held,held))
        np.testing.assert_allclose(result.arrays['autograd_jacobian'][0,:2],expected,atol=2e-15,rtol=0.)
        self.assertEqual(result.arrays['finite_difference_jacobian'].shape,(2,3,4,4))
        self.assertTrue(result.arrays['finite_difference_evaluated'].all());self.assertGreater(abs(expected[0,2]),.01)

    def test_f02_distinct_batch_full_output_rows_and_true_nonzero_control_gradients(self):
        model=DeclarativeDynamics(self.linear_spec);reference=ScalarRowReference(self.linear_spec)
        result=validate_pair(reference,model,pure_config(model.interface))
        self.assertTrue(result.report['qualified_for_pgac']);self.assertTrue(all(p['agreement_by_h']==[True]*3 for p in result.report['gradient_probes']))
        expected=np.column_stack((self.linear_spec['A'],self.linear_spec['B'],self.linear_spec['E']))
        np.testing.assert_array_equal(result.arrays['autograd_jacobian'][0,:4],expected)
        self.assertEqual(result.arrays['autograd_jacobian'].shape,(2,8,7));self.assertGreater(np.linalg.norm(expected[:,4:6]),.1)
        self.assertEqual(result.report['numeric']['mixed_batch_next_x']['rows'],100)

    def test_true_gradient_with_wrong_values_is_rejected_separately(self):
        base=DeclarativeDynamics(self.linear_spec);candidate=Variant(base,'wrong_numeric')
        result=validate_pair(ScalarRowReference(self.linear_spec),candidate,pure_config(base.interface))
        self.assertTrue(result.report['gradient_checks_passed']);self.assertFalse(result.report['exact_equivalence_pass']);self.assertFalse(result.report['qualified_for_pgac'])
        self.assertGreater(max(result.report['numeric']['single_next_x']['max_abs_error']),.001)

    def test_numerically_exact_detached_transition_is_rejected_by_full_jacobian(self):
        base=DeclarativeDynamics(self.linear_spec);result=validate_pair(ScalarRowReference(self.linear_spec),Variant(base,'detached'),pure_config(base.interface))
        self.assertTrue(result.report['exact_equivalence_pass']);self.assertFalse(result.report['gradient_checks_passed']);self.assertFalse(result.report['qualified_for_pgac'])
        self.assertGreater(result.report['gradient_probes'][0]['gradient_max_abs_error_by_h'][0],.1)

    def test_cross_row_broadcast_bug_passes_single_gradient_but_fails_mixed_batch(self):
        base=DeclarativeDynamics(self.linear_spec);result=validate_pair(ScalarRowReference(self.linear_spec),Variant(base,'mixed_batch'),pure_config(base.interface))
        self.assertTrue(result.report['numeric']['single_next_x']['within_tolerance']);self.assertTrue(result.report['gradient_checks_passed'])
        self.assertFalse(result.report['numeric']['mixed_batch_next_x']['within_tolerance']);self.assertFalse(result.report['qualified_for_pgac'])

    def test_proxy_budget_does_not_rename_approximation_as_exact_equivalence(self):
        base=DeclarativeDynamics(self.linear_spec);config=pure_config(base.interface);config.update(mode='bounded_surrogate',domain={k:[[v.lower,v.upper] for v in variables] for k,variables in [('x',base.interface.states),('u',base.interface.controls),('d',base.interface.disturbances)]})
        without=validate_pair(ScalarRowReference(self.linear_spec),Variant(base,'wrong_numeric'),config)
        self.assertIsNone(without.report['bounded_proxy_acceptance']);self.assertFalse(without.report['qualified_for_pgac'])
        config['proxy_budget']={key:[.1]*width for key,width in [('one_step_state_rmse',4),('one_step_output_rmse',2),('rollout_state_rmse',4),('rollout_output_rmse',2)]}
        result=validate_pair(ScalarRowReference(self.linear_spec),Variant(base,'wrong_numeric'),config)
        self.assertFalse(result.report['exact_equivalence_pass']);self.assertTrue(result.report['bounded_proxy_acceptance']);self.assertTrue(result.report['qualified_for_pgac'])
        self.assertEqual(result.report['gradient_semantics'],'surrogate_gradient_not_physical_true_gradient');self.assertFalse(result.report['coverage']['joint_coverage_proved'])

    def saturation_fixture(self,absolute=False):
        spec={'schema_version':1,'name':'synthetic_switch_test_fixture','sample_time':.1,'time_semantics':'discrete_state_transition',
            'states':[{'name':'x','unit':'m','lower':-2.,'upper':2.}], 'controls':[{'name':'force','unit':'N','lower':-2.,'upper':2.}],
            'outputs':[{'name':'x','unit':'m','lower':-2.,'upper':2.}], 'disturbances':[]}
        interface=ModelInterface.parse(spec)
        class Original:
            def __call__(self,x,u,d):return np.array([.5*abs(x[0])+.05*u[0] if absolute else .6*x[0]+.2*np.clip(u[0],-.5,.5)])
            def output(self,x):return x.copy()
        class Candidate(nn.Module):
            def forward(self,x,u,d):return .5*torch.abs(x)+.05*u if absolute else .6*x+.2*torch.clamp(u,-.5,.5)
            def output(self,x):return x
        original=Original();original.interface=interface;candidate=Candidate();candidate.interface=interface
        return original,candidate

    def test_correct_zero_saturation_and_disclosed_kink_pass_without_smoothing_nd0(self):
        original,candidate=self.saturation_fixture();steps=50
        config={'steps':steps,'probes':[{'row':0,'kind':'saturation_interior'},{'row':25,'kind':'switch_boundary','note':'Physical actuator upper clamp at .5N'}],
            'domain':{'x':[[-2.,2.]],'u':[[-2.,2.]],'d':[]},
            'switching_planes':[{'name':'upper','axis':'u','index':0,'value':.5,'unit':'N'},{'name':'lower','axis':'u','index':0,'value':-.5,'unit':'N'}],
            'trajectories':[{'id':1,'initial_state':[.1],'u':[[.9]]*25+[[.5]]*25,'d':[[] for _ in range(steps)]}]}
        result=validate_pair(original,candidate,config)
        self.assertTrue(result.report['qualified_for_pgac']);self.assertEqual(result.arrays['reference_d'].shape,(50,0))
        np.testing.assert_array_equal(result.arrays['autograd_jacobian'][0,:,1],np.zeros(3))
        boundary=result.report['gradient_probes'][1];self.assertIsNone(boundary['passed']);self.assertTrue(boundary['boundary_excluded_from_smooth_acceptance']);self.assertTrue(boundary['nonsmoothness_detected'])
        self.assertFalse(result.report['source_code_rewritten_or_switches_smoothed'])

    def test_absolute_value_kink_not_hidden_by_zero_central_difference(self):
        original,candidate=self.saturation_fixture(absolute=True)
        config={'steps':50,'probes':[{'row':0,'kind':'smooth'}],'trajectories':[{'id':1,'initial_state':[0.],'u':[[.1]]*50,'d':[[] for _ in range(50)]}]}
        result=validate_pair(original,candidate,config)
        self.assertTrue(result.report['exact_equivalence_pass']);self.assertFalse(result.report['gradient_checks_passed']);self.assertTrue(result.report['gradient_probes'][0]['nonsmoothness_detected'])

    def test_hidden_buffer_mutation_rejected_and_original_rng_model_mode_restored(self):
        base=DeclarativeDynamics(self.linear_spec)
        class Mutating(Variant):
            def __init__(self):super().__init__(base,'normal');self.register_buffer('counter',torch.zeros((),dtype=torch.float64))
            def forward(self,x,u,d):self.counter.add_(1);return self.base(x,u,d)
        candidate=Mutating();candidate.train();rng=torch.get_rng_state().clone()
        result=validate_pair(ScalarRowReference(self.linear_spec),candidate,pure_config(base.interface))
        self.assertFalse(result.report['qualified_for_pgac']);self.assertTrue(any(e['stage']=='candidate_state' for e in result.report['errors']))
        self.assertEqual(candidate.counter.item(),0.);self.assertTrue(candidate.training);torch.testing.assert_close(torch.get_rng_state(),rng,atol=0,rtol=0)

    def test_dtype_config_budget_and_cancellation_reject_without_fake_success(self):
        base=DeclarativeDynamics(self.linear_spec);result=validate_pair(ScalarRowReference(self.linear_spec),Variant(base,'float32'),pure_config(base.interface))
        self.assertFalse(result.report['qualified_for_pgac']);self.assertTrue(result.report['errors'])
        for patch_config in [{'steps':True},{'mode':[]},{'partition':'test'},{'gradient':{'atol':1e-5,'rtol':1e-3,'finite_difference_steps':[True,1e-5,1e-6]}}]:
            cfg=pure_config(base.interface);cfg.update(patch_config);reference=ScalarRowReference(self.linear_spec)
            with self.subTest(config=patch_config),self.assertRaises(ContractError):validate_pair(reference,base,cfg)
            self.assertEqual(reference.calls,0)
        with self.assertRaises(Cancelled):validate_pair(ScalarRowReference(self.linear_spec),base,pure_config(base.interface),OperationContext(cancelled=lambda:True))
        raw=copy.deepcopy(self.linear_spec['interface']);raw['states']=[{'name':f'x{n}','unit':'1','lower':-2.,'upper':2.} for n in range(1000)]
        huge=ModelInterface.parse(raw)
        with self.assertRaisesRegex(ContractError,'Jacobian'):resolve_validation_config({'probes':[{'row':n,'kind':'smooth'} for n in range(32)]},huge)


_SPRING_SOURCE='''import math
import numpy as np
def transition(x,u,d,dt,parameters):
    mass=parameters['mass'];damping=parameters['damping'];stiffness=parameters['stiffness']
    alpha=damping/(2*mass);omega=math.sqrt(stiffness/mass-alpha*alpha)
    phi=math.exp(-alpha*dt)*(math.cos(omega*dt)*np.eye(2)+math.sin(omega*dt)/omega*np.array([[alpha,1.],[-stiffness/mass,-alpha]]))
    equilibrium=np.array([(u[0]+d[0])/stiffness,0.])
    return equilibrium+phi@(x-equilibrium)
'''


class DifferentiableArtifactTests(unittest.TestCase):
    confirmation_source='test_fixture'
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=OUTPUT);cls.base=Path(cls.temp.name);cls.prepared=Workspace(cls.base/'prepared');cls.threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.spec=read_json(FROZEN/'F01_spring_2x1.json')['source_model'];physical=DeclarativeDynamics(cls.spec).interface
        source=cls.base/'original.py';source.write_text(_SPRING_SOURCE,encoding='utf-8')
        material=register_material(cls.prepared,str(source),{},OperationContext())
        original=register_plant(cls.prepared,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/original.py','source_semantics':'transition',
            'interface':json.loads(json.dumps(physical.to_dict())),'entrypoint':'transition','parameters':{'mass':1.,'damping':.8,'stiffness':2.},'C':[[1.,0.]]}},OperationContext())
        sampling=recommend_sampling(physical,{'trajectory_count':5,'steps':50,'seed':20261118,'x0':[.1,-.05]})
        data=sample_model(cls.prepared,original,{'sampling':sampling,'sampling_run':'full'},fixture_context(cls.prepared))
        cls.original=original;cls.data=data
        cls.candidate=register_model(cls.prepared,cls.spec,{'kind':'reconstructed_equations','description':'Synthetic F01 test_fixture Torch exact ZOH rewrite of explicit original equations'})
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads);cls.temp.cleanup()
    def setUp(self):
        self.temp_case=tempfile.TemporaryDirectory(dir=self.base);self.addCleanup(self.temp_case.cleanup);self.ws=Workspace(Path(self.temp_case.name)/'workspace')
        self.ws.import_artifact(self.prepared.path/'artifacts'/self.data);self.ws.import_artifact(self.prepared.path/'artifacts'/self.candidate)
        root,_=self.ws.resolve(self.data);self.split=read_json(root/'split.json')
        with np.load(root/'transitions.npz',allow_pickle=False) as loaded:self.arrays={k:loaded[k].copy() for k in loaded.files}
        self.val=np.flatnonzero(np.isin(self.arrays['trajectory_id'],self.split['validation']))
        self.config={'steps':50,'probes':[{'row':int(self.val[3]),'kind':'smooth'}]}

    def test_actual_python_sampling_reference_public_receipt_and_current_binding_accessor(self):
        aid=validate_differentiable_model(self.ws,self.original,self.candidate,self.config,fixture_context(self.ws),reference_dataset_id=self.data)
        report=read_differentiable_validation(self.ws,aid,candidate_id=self.candidate,original_id=self.original)
        self.assertTrue(report['qualified_for_pgac']);self.assertEqual(report['reference_backend'],'python_callable');self.assertEqual(report['coverage']['reference_rows'],50)
        self.assertEqual(report['bindings']['dataset']['id'],self.data);self.assertFalse(report['test_used_for_selection']);self.assertFalse(report['original_plant_closed_loop_evaluated'])
        root,meta=self.ws.resolve(aid);self.assertEqual(meta['activation'],'none');self.assertEqual(meta['interface']['interface_kind'],'material')
        self.assertEqual({p['id'] for p in meta['parents']},{self.original,self.candidate,self.data})
        self.assertEqual(meta['validation']['status'],'verified');self.assertNotIn('differentiable_validation',self.ws.manifest['active'])
        with np.load(root/'validation_arrays.npz',allow_pickle=False) as data:np.testing.assert_array_equal(data['reference_source_row'],self.val)
        with self.assertRaisesRegex(ContractError,'different candidate'):read_differentiable_validation(self.ws,aid,candidate_id='other')
        material_id=self.ws.resolve(self.original)[1]['parents'][0]['id'];material_root,_=self.ws.resolve(material_id)
        (material_root/'source/original.py').write_text('# changed original bytes',encoding='utf-8')
        with self.assertRaises(ContractError):read_differentiable_validation(self.ws,aid,candidate_id=self.candidate)

    def test_numeric_wrong_candidate_publishes_rejected_inactive_diagnostic(self):
        spec=copy.deepcopy(self.spec);spec['bias']=[.01,0.]
        candidate=register_model(self.ws,spec,{'kind':'reconstructed_equations','description':'Synthetic deliberately wrong differentiable equation negative fixture'})
        aid=validate_differentiable_model(self.ws,self.original,candidate,self.config,reference_dataset_id=self.data)
        report=read_differentiable_validation(self.ws,aid,require_qualified=False)
        self.assertFalse(report['exact_equivalence_pass']);self.assertTrue(report['gradient_checks_passed']);self.assertFalse(report['qualified_for_pgac'])
        with self.assertRaises(ContractError):read_differentiable_validation(self.ws,aid,candidate_id=candidate)

    def test_unreserved_partition_wrong_source_and_missing_proxy_coverage_stop_before_candidate_load(self):
        for changed in [{'trajectory_ids':self.split['train']},{'probes':[{'row':int(np.flatnonzero(np.isin(self.arrays['trajectory_id'],self.split['test']))[0]),'kind':'smooth'}]},
                {'mode':'bounded_surrogate','domain':{k:[[v.lower,v.upper] for v in variables] for k,variables in [('x',DeclarativeDynamics(self.spec).interface.states),('u',DeclarativeDynamics(self.spec).interface.controls),('d',DeclarativeDynamics(self.spec).interface.disturbances)]}}]:
            config={**self.config,**changed}
            with self.subTest(config=changed),patch('control_harness.modeling.load_model',side_effect=AssertionError('Candidate execution must not start')),self.assertRaises(ContractError):
                validate_differentiable_model(self.ws,self.original,self.candidate,config,reference_dataset_id=self.data)
        with patch('control_harness.modeling.load_model',side_effect=AssertionError('Candidate execution must not start')),self.assertRaisesRegex(ContractError,'exact original'):
            validate_differentiable_model(self.ws,self.candidate,self.candidate,self.config,reference_dataset_id=self.data)

    def test_mixed_row_reordering_not_hidden_by_sorted_reference_reader(self):
        broken={k:v.copy() for k,v in self.arrays.items()};a,b=int(self.val[3]),int(self.val[4])
        for values in broken.values():values[[a,b]]=values[[b,a]]
        interface=DeclarativeDynamics(self.spec).interface
        with self.ws.stage('sampled_dataset',interface,[self.original],{'kind':'synthetic_malformed_test_fixture'},compatible_parents=False) as stage:
            np.savez_compressed(stage.root/'transitions.npz',**broken);atomic_json(stage.root/'split.json',self.split)
            bad=stage.publish({'status':'verified','scope':'intentionally malformed negative admission test only'},source={'kind':'plant_simulator','model':self.original,'measured':False,'synthetic_test_fixture':True},activate=False)
        with patch('control_harness.modeling.load_model',side_effect=AssertionError('Candidate execution must not start')),self.assertRaisesRegex(ContractError,'order'):
            validate_differentiable_model(self.ws,self.original,self.candidate,self.config,reference_dataset_id=bad)


class RealLearnedProxyTests(unittest.TestCase):
    """One preregistered S06 fit; no search or held-out metric selection."""
    confirmation_source='test_fixture'

    def test_actual_sampled_to_learned_proxy_qualification_and_training_domain_boundary(self):
        registration=read_json(PROJECT/'test-output/v2-development/20261117T002006/S12/unit-proxy-registration.json')
        temporary=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(temporary.cleanup)
        base=Path(temporary.name);workspace=Workspace(base/'workspace');threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        interface=ModelInterface.parse(registration['interface'])
        split=trajectory_split(np.arange(6,dtype=np.int64),registration['seed'])
        self.assertEqual(split,registration['split'])
        source=base/'synthetic_proxy_source.py'
        source.write_text('import numpy as np\ndef transition(x,u,d,dt,p):\n return np.array([.35*x[0]+.1*u[0]],dtype=np.float64)\n',encoding='utf-8')
        material=register_material(workspace,str(source),{},OperationContext())
        original=register_plant(workspace,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/synthetic_proxy_source.py',
            'source_semantics':'transition','interface':interface.to_dict(),'entrypoint':'transition','parameters':{},'C':[[1.]]}},OperationContext())
        sampling=recommend_sampling(interface,registration['sampling'])
        dataset=sample_model(workspace,original,{'sampling':sampling,'sampling_run':'full'},fixture_context(workspace))
        data_root,_=workspace.resolve(dataset)
        self.assertEqual(read_json(data_root/'split.json'),split)
        with np.load(data_root/'transitions.npz',allow_pickle=False) as loaded:arrays={k:loaded[k].copy() for k in loaded.files}
        learned=train_predictive_model(workspace,dataset,registration['training'],fixture_context(workspace))
        learned_root,learned_meta=workspace.resolve(learned)
        self.assertEqual(learned_meta['kind'],'learned_model')
        report=read_json(learned_root/'report.json');coverage=report['coverage']
        train=np.isin(arrays['trajectory_id'],split['train'])
        self.assertEqual(coverage['source'],'training_partition_only');self.assertEqual(coverage['samples'],200)
        self.assertIn(dataset,{p['id'] for p in learned_meta['parents']})
        for row,key in zip(coverage['marginal_ranges'],['x','u']):
            self.assertEqual(row['observed_min'],float(arrays[key][train,0].min()))
            self.assertEqual(row['observed_max'],float(arrays[key][train,0].max()))
        # The configured 300 epochs, not test metrics, determine this single fit.
        inner_root,_=workspace.resolve(report['supervised_model'])
        self.assertEqual(read_json(inner_root/'report.json')['trained_epochs'],300)
        val=np.flatnonzero(np.isin(arrays['trajectory_id'],split['validation']))
        config={**registration['validation'],'probes':[{'row':int(val[3]),'kind':'smooth'}]}
        aid=validate_differentiable_model(workspace,original,learned,config,fixture_context(workspace),reference_dataset_id=dataset)
        validated=read_differentiable_validation(workspace,aid,candidate_id=learned,original_id=original)
        self.assertTrue(validated['qualified_for_pgac']);self.assertTrue(validated['bounded_proxy_acceptance'])
        self.assertFalse(validated['exact_equivalence_pass']);self.assertTrue(validated['gradient_checks_passed'])
        self.assertEqual(validated['proxy_training_coverage']['training_coverage'],coverage)
        self.assertEqual(validated['bindings']['dataset']['id'],dataset)
        self.assertEqual(validated['coverage']['trajectory_ids'],split['validation'])
        self.assertFalse(validated['coverage']['joint_coverage_proved'])
        self.assertEqual(validated['gradient_semantics'],'surrogate_gradient_not_physical_true_gradient')
        changed=copy.deepcopy(config);changed['domain']['u']=[[-.7,.7]]
        with patch('control_harness.modeling.load_model',side_effect=AssertionError('Out-of-training-domain candidate must not execute')),self.assertRaisesRegex(ContractError,'training marginal'):
            validate_differentiable_model(workspace,original,learned,changed,reference_dataset_id=dataset)


if __name__=='__main__':unittest.main()
