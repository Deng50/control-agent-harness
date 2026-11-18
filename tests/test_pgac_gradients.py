"""Actual external MLP rollout gradients and short numerical continuation.

Synthetic component-only systems exercise real policy/export/source APIs.
No frozen F05 candidate, native Host, MATLAB, critic, PPO or SAC is executed.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace,read_json,file_hash
from control_harness.contracts import ContractError,ModelInterface
from control_harness.modeling import DeclarativeDynamics,register_model
from control_harness.policy_export import export_external_policy_state
from control_harness.external_policy import load_external_policy
from control_harness.restricted_pgac import (physical_rollout_loss,normalize_pgac_config,require_current_full_state,
    tensor_policy_action,RolloutDomainError,_predictor_validation,train_restricted_pgac)
from control_harness.supervised_export import build_supervised_network
from control_harness.runtime import Cancelled,OperationContext
from test_policy_export import fixture

PROJECT=Path(__file__).resolve().parents[1]
OUTPUT=PROJECT/'test-output/pgac-gradient-tests'


class PgacGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True,exist_ok=True);cls.threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        data,context,constraints=fixture();self.context=context
        self.config={'schema_version':1,'architecture':'mlp','input_dim':9,'output_dim':2,'hidden_dims':[4],'activation':'tanh','dtype':'float64'}
        self.scaler={'method':'none','fit_split':'train','fit_samples':17,'X':{'location':[0.]*9,'scale':[1.]*9},'y':{'location':[0.,0.],'scale':[1.,1.]}}
        network=build_supervised_network(self.config)
        with torch.no_grad():
            for layer in network:
                if isinstance(layer,torch.nn.Linear):layer.weight.copy_(torch.linspace(-.04,.05,layer.weight.numel(),dtype=torch.float64).reshape(layer.weight.shape));layer.bias.fill_(.01)
        export_external_policy_state(self.root/'bundle',network.state_dict(),self.config,data,self.scaler,constraints,context,provenance={'purpose':'synthetic gradient component only'})
        self.policy=load_external_policy(self.root/'bundle');self.policy.network.requires_grad_(True)
        self.A=np.diag([.7,.6,.5,.4]);self.B=np.array([[.14,.02],[.03,.12],[.05,-.02],[-.01,.04]])
        self.E=np.array([[.05],[.02],[-.01],[.03]]);self.C=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
        self.model=DeclarativeDynamics({'representation':'discrete_polynomial','interface':context['interface'],'A':self.A.tolist(),'B':self.B.tolist(),'C':self.C.tolist(),'E':self.E.tolist()})
        self.domain={'x':[[-2.,2.]]*4,'u':[[-1.5,1.5]]*2,'d':[[-.3,.3]]}
        self.settings={'horizon':3,'q':1.2,'r':.02,'s':.03,'imitation_weight':.1,'constraint_weight':100.,'domain':self.domain}
        self.x=torch.tensor([[.2,-.1,.03,.05],[-.1,.15,-.02,.04]],dtype=torch.float64)
        self.reference=torch.tensor([[.3,-.2],[-.2,.2]],dtype=torch.float64);self.d=torch.tensor([[.02],[-.01]],dtype=torch.float64)
        self.previous=torch.tensor([[.03,-.02],[0.,.01]],dtype=torch.float64);self.expert=torch.tensor([[.2,-.1],[.1,.05]],dtype=torch.float64)

    def loss(self,horizon=3,x=None,previous=None):
        return physical_rollout_loss(self.policy,self.model,self.x if x is None else x,self.reference,self.d,self.previous if previous is None else previous,
            self.expert,{**self.settings,'horizon':horizon},substeps=2)

    def manual_loss(self,horizon,settings=None):
        settings=self.settings if settings is None else settings
        state=self.x.numpy().copy();reference=self.reference.numpy();d=self.d.numpy();previous=self.previous.numpy().copy()
        initial_state=state.copy();initial_previous=previous.copy();parameters={key:value.detach().numpy().copy() for key,value in self.policy.network.state_dict().items()}
        def action(x,p):
            features=np.column_stack((x,reference,d,p));z=np.tanh(features@parameters['0.weight'].T+parameters['0.bias'])
            raw=z@parameters['2.weight'].T+parameters['2.bias']
            return np.maximum(np.minimum(raw,np.minimum(1.5,p+.25)),np.maximum(-1.5,p-.25))
        cost=0.
        for _ in range(horizon):
            u=action(state,previous)
            for _ in range(2):state=state@self.A.T+u@self.B.T+d@self.E.T
            y=state@self.C.T
            cost+=settings['q']*np.mean(((y-reference)/4.)**2)+settings['r']*np.mean((u/3.)**2)+settings['s']*np.mean(((u-previous)/3.)**2)
            previous=u
        anchor=np.mean(((action(initial_state,initial_previous)-self.expert.numpy())/3.)**2)
        return cost/horizon+settings['imitation_weight']*anchor

    def test_three_horizons_match_independent_matrix_loss_and_parameter_finite_differences(self):
        require_current_full_state(self.policy,self.model.interface)
        for horizon in (1,3,5):
            self.policy.network.zero_grad(set_to_none=True);loss=self.loss(horizon);loss.backward()
            self.assertAlmostEqual(float(loss.detach()),self.manual_loss(horizon),places=14)
            gradient=np.concatenate([p.grad.detach().numpy().reshape(-1) for p in self.policy.network.parameters()])
            self.assertGreater(float(np.linalg.norm(gradient)),1e-8)
            for h in (1e-4,1e-5,1e-6):
                numerical=[]
                for parameter in self.policy.network.parameters():
                    for index in range(parameter.numel()):
                        original=float(parameter.detach().reshape(-1)[index])
                        with torch.no_grad():parameter.reshape(-1)[index]=original+h
                        plus=self.manual_loss(horizon)
                        with torch.no_grad():parameter.reshape(-1)[index]=original-h
                        minus=self.manual_loss(horizon)
                        with torch.no_grad():parameter.reshape(-1)[index]=original
                        numerical.append((plus-minus)/(2*h))
                np.testing.assert_allclose(gradient,numerical,atol=1e-9,rtol=1e-5)

    def test_tracking_only_actor_gradient_passes_through_frozen_physical_dynamics(self):
        settings={**self.settings,'r':0.,'s':0.,'imitation_weight':0.,'constraint_weight':0.}
        loss=physical_rollout_loss(self.policy,self.model,self.x,self.reference,self.d,self.previous,self.expert,settings,substeps=2)
        self.policy.network.zero_grad(set_to_none=True);loss.backward()
        gradient=np.concatenate([p.grad.detach().numpy().reshape(-1) for p in self.policy.network.parameters()])
        self.assertGreater(float(np.linalg.norm(gradient)),1e-7);self.assertAlmostEqual(float(loss.detach()),self.manual_loss(3,settings),places=14)
        parameter=self.policy.network[-1].bias;original=float(parameter.detach()[0]);h=1e-5
        with torch.no_grad():parameter[0]=original+h
        plus=self.manual_loss(3,settings)
        with torch.no_grad():parameter[0]=original-h
        minus=self.manual_loss(3,settings)
        with torch.no_grad():parameter[0]=original
        self.assertAlmostEqual(float(parameter.grad[0]),(plus-minus)/(2*h),places=9)

    def test_nonfinite_differentiable_model_fault_is_rejected_before_an_optimizer_update(self):
        class NumericalFault(torch.nn.Module):
            interface=self.model.interface
            def forward(self,x,u,d):return x+torch.sqrt(x.new_tensor(-1.))*u.sum(-1,keepdim=True)
            def output(self,x):return x[...,:2]
        before={key:value.clone() for key,value in self.policy.network.state_dict().items()}
        with self.assertRaises(FloatingPointError):
            physical_rollout_loss(self.policy,NumericalFault(),self.x,self.reference,self.d,self.previous,self.expert,self.settings,substeps=2)
        self.assertTrue(all(torch.equal(before[key],value) for key,value in self.policy.network.state_dict().items()))

    def test_current_feature_order_and_real_state_previous_action_gradients_keep_two_subticks(self):
        assembled=torch.cat((self.x,self.reference,self.d,self.previous),dim=-1)
        torch.testing.assert_close(tensor_policy_action(self.policy,self.model,self.x,self.reference,self.d,self.previous),self.policy(assembled,self.previous),rtol=0,atol=0)
        x=self.x.clone().requires_grad_(True);previous=self.previous.clone().requires_grad_(True)
        gradients=torch.autograd.grad(self.loss(3,x,previous),(x,previous))
        for source,gradient in zip((x,previous),gradients):
            numerical=np.zeros_like(source.detach().numpy())
            for index in np.ndindex(*source.shape):
                plus=source.detach().clone();minus=plus.clone();plus[index]+=1e-6;minus[index]-=1e-6
                args_plus=(plus,previous.detach()) if source is x else (x.detach(),plus)
                args_minus=(minus,previous.detach()) if source is x else (x.detach(),minus)
                numerical[index]=(float(self.loss(3,*args_plus).detach())-float(self.loss(3,*args_minus).detach()))/2e-6
            np.testing.assert_allclose(gradient.detach().numpy(),numerical,atol=1e-9,rtol=1e-5)
            self.assertGreater(float(torch.linalg.vector_norm(gradient)),1e-8)
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.parameters()))

    def test_true_saturation_has_zero_parameter_gradient_and_domain_failure_never_calls_model(self):
        with torch.no_grad():
            for parameter in self.policy.network.parameters():parameter.zero_()
            self.policy.network[-1].bias.copy_(torch.tensor([5.,-5.],dtype=torch.float64))
        self.policy.network.zero_grad(set_to_none=True);self.loss().backward()
        self.assertTrue(all(torch.count_nonzero(p.grad)==0 for p in self.policy.network.parameters()))
        calls=[];handle=self.model.register_forward_hook(lambda *_:calls.append(1));x=self.x.clone();x[0,0]=2.1
        with self.assertRaises(RolloutDomainError):self.loss(x=x)
        handle.remove();self.assertEqual(calls,[])

    def test_partial_state_and_history_are_not_promoted_to_restricted_pgac(self):
        self.policy.binding=copy.deepcopy(self.policy.binding);self.policy.binding['current_features']=self.policy.binding['current_features'][1:]
        with self.assertRaisesRegex(ContractError,'every physical state'):require_current_full_state(self.policy,self.model.interface)
        self.policy.binding['status']='unavailable'
        with self.assertRaisesRegex(ContractError,'history'):require_current_full_state(self.policy,self.model.interface)

    def test_output_observations_never_evaluate_an_out_of_domain_current_or_predicted_state(self):
        from control_harness.external_expert import expert_data_interface
        context=copy.deepcopy(self.context);observations=context['physical_mapping']['observations']
        observations.append({'name':'current-output','source':'output','index':0,'plant_name':'y0','unit':'m','availability':'measured','delay_steps':0})
        spec={'interface':context['interface'],'control_period':context['control_period'],'observations':observations,
            'reference_available':True,'disturbance_available':True,'previous_action_available':True}
        selected=['observation::'+row['name'] for row in observations]+['reference::y0','reference::y1','disturbance::wind','previous_action::u0','previous_action::u1']
        interface,mapping=expert_data_interface(spec,{'student_features':selected});config={**self.config,'input_dim':10}
        network=build_supervised_network(config);scaler={**self.scaler,'X':{'location':[0.]*10,'scale':[1.]*10}}
        export_external_policy_state(self.root/'output-observation',network.state_dict(),config,interface.to_dict(),scaler,self.policy.action_constraints,context,provenance={'source_mapping':mapping})
        policy=load_external_policy(self.root/'output-observation');policy.network.requires_grad_(True);require_current_full_state(policy,self.model.interface)
        calls=[]
        class Instrumented(torch.nn.Module):
            interface=self.model.interface
            def __init__(self,base):super().__init__();self.base=base
            def forward(self,x,u,d):calls.append('transition');return self.base(x,u,d)
            def output(self,x):calls.append('output');return self.base.output(x)
        model=Instrumented(self.model);outside=self.x.clone();outside[0,0]=2.1
        with self.assertRaises(RolloutDomainError):physical_rollout_loss(policy,model,outside,self.reference,self.d,self.previous,self.expert,self.settings,substeps=2)
        self.assertEqual(calls,[])
        original={'scenarios':[{'name':'outside','split':'validation','initial_state':[2.1,0.,0.,0.],'reference':[.1,0.],'disturbance':[0.],
            'initial_previous_action':[0.,0.],'steps':1}],'evaluation':{'q':1.,'r':.01,'s':.01,'comparison_atol':1e-12}}
        with self.assertRaises(RolloutDomainError):_predictor_validation(policy,model,original,self.settings,2,OperationContext())
        self.assertEqual(calls,[])
        # A permitted transition may predict OOD; never call output on that new state.
        with torch.no_grad():self.model.A[0,0]=2.
        narrow={**self.settings,'domain':{**self.domain,'x':[[-.25,.25]]*4}}
        with self.assertRaises(RolloutDomainError):physical_rollout_loss(policy,model,self.x,self.reference,self.d,self.previous,self.expert,narrow,substeps=2)
        self.assertEqual(calls,['output','transition'])

    def test_only_validation_scenarios_are_forwarded_and_canonical_domain_config_is_strict(self):
        validation={'name':'validation','split':'validation','initial_state':[.1,0.,0.,0.],'reference':[.2,0.], 'disturbance':[0.],'initial_previous_action':[0.,0.],'steps':3}
        never=dict(validation,name='test-never-executed',split='test',initial_state=[50.,0.,0.,0.])
        original={'scenarios':[validation,never],'evaluation':{'q':1.,'r':.01,'s':.01,'comparison_atol':1e-12}}
        result=_predictor_validation(self.policy,self.model,original,self.settings,2,OperationContext())
        self.assertEqual(len(result['runs']),1);self.assertEqual(result['runs'][0]['name'],'validation');self.assertFalse(result['test_used'])
        config={key:self.settings[key] for key in ('horizon','q','r','s','imitation_weight','constraint_weight','domain')}
        self.assertEqual(normalize_pgac_config(config,self.model.interface,self.domain)['grad_clip'],5.)
        for extra in ({'critic':True},{'learning_rate':0.},{'seed':True},{'domain':{'x':[[-3.,3.]]*4,'u':self.domain['u'],'d':self.domain['d']}}):
            with self.subTest(extra=extra),self.assertRaises(ContractError):normalize_pgac_config({**config,**extra},self.model.interface,self.domain)


class RestrictedPgacTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from control_harness.material_registration import register_material
        from control_harness.plants import register_plant,sample_model
        from control_harness.external_expert import register_expert,collect_external_expert
        from control_harness.external_imitation import train_external_imitation
        from control_harness.external_policy_evaluation import prepare_external_imitation_evaluation
        from control_harness.differentiable_validation import validate_differentiable_model
        OUTPUT.mkdir(parents=True,exist_ok=True);cls.temporary=tempfile.TemporaryDirectory(dir=OUTPUT);cls.base=Path(cls.temporary.name)
        cls.threads=torch.get_num_threads();torch.set_num_threads(1);ws=Workspace(cls.base/'prepared');context=OperationContext()
        variable=lambda name,bound:{'name':name,'unit':'1','lower':-bound,'upper':bound}
        cls.physical={'schema_version':1,'name':'short actual PGAC scalar unit source','sample_time':.1,'time_semantics':'discrete_state_transition',
            'states':[variable('state',3.)],'controls':[variable('command',1.)],'outputs':[variable('output',3.)],'disturbances':[]}
        plant_file=cls.base/'plant.py';plant_file.write_text('import numpy as np\ndef step(x,u,d,dt,p):\n return np.array([.7*x[0]+.2*u[0]])\n',encoding='utf-8')
        material=register_material(ws,str(plant_file),{},context)
        cls.plant=register_plant(ws,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/plant.py','entrypoint':'step','interface':cls.physical,
            'parameters':{},'source_semantics':'transition','C':[[1.]]}},context)
        expert_file=cls.base/'expert.py';expert_file.write_text('import numpy as np\ndef control(obs,r,d,previous,memory,p):\n return (np.array([.7*(r[0]-obs[0])+.05*previous[0]]), {"calls":memory["calls"]+1}, "ok")\n',encoding='utf-8')
        material=register_material(ws,str(expert_file),{},context)
        expert=register_expert(ws,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/expert.py','entrypoint':'control','interface':cls.physical,
            'parameters':{},'time_unit':'s','control_period':.2,'observations':[{'name':'state','source':'state','index':0,'plant_name':'state','unit':'1','availability':'measured','delay_steps':0}],
            'reference_available':True,'disturbance_available':True,'previous_action_available':True,'initial_memory':{'calls':0},
            'action_mapping':[{'action_index':0,'plant_control_index':0,'plant_control_name':'command','unit':'1'}],'failure_policy':'stop','input_rate_bounds':[[-.4,.4]]}})
        collection={'seed':20261129,'scaler':'standard','scenarios':[{'name':'unit'+str(k),'initial_state':[.05+.04*k],'reference':[.3],'disturbance':[],'steps':6,
            'initial_previous_action':[0.],'split':'train' if k<3 else 'validation' if k==3 else 'test'} for k in range(6)]}
        cls.dataset=collect_external_expert(ws,expert,cls.plant,collection)
        cls.policy=train_external_imitation(ws,cls.dataset,{'network':{'hidden_dims':[4],'activation':'tanh','epochs':3,'learning_rate':.004,'batch_size':8,'patience':20,'seed':20261129}})
        cls.model=register_model(ws,{'representation':'discrete_polynomial','interface':cls.physical,'A':[[.7]],'B':[[.2]],'C':[[1.]],'E':[[]]}, {'kind':'reconstructed_equations','description':'Actual exact numerical unit-only rewrite'})
        sampling={'mode':'expert','seed':20261129,'steps':50,'trajectory_count':5,'d0':[],'d_profile':None,'control_rate_limits':None,'disturbance_rate_limits':None,'coupled_constraints':[],
            'trajectories':[{'x0':[.15+.03*k],'u0':[.1+.01*k],'d0':[],'u_profile':{'kind':'step','times':[0.],'values':[[.1+.01*k]]},'d_profile':None} for k in range(5)]}
        sampled=sample_model(ws,cls.plant,{'sampling':sampling},context);sampled_root,_=ws.resolve(sampled)
        split=read_json(sampled_root/'split.json')
        with np.load(sampled_root/'transitions.npz') as rows:probe=int(np.flatnonzero(rows['trajectory_id']==split['validation'][0])[0])+1
        cls.domain={'x':[[-3.,3.]],'u':[[-1.,1.]],'d':[]}
        cls.validation=validate_differentiable_model(ws,cls.plant,cls.model,{'mode':'exact_rewrite','steps':50,'probes':[{'row':probe,'kind':'smooth'}],'domain':cls.domain},reference_dataset_id=sampled)
        original=prepare_external_imitation_evaluation(ws,cls.policy)['config_suggestions'];original.pop('metrics',None);original.pop('benchmark',None)
        original['evaluation']={'q':1.,'r':.01,'s':.01,'comparison_atol':1e-12}
        cls.config={'epochs':3,'learning_rate':.0005,'batch_size':2,'horizon':3,'seed':20261129,'q':1.,'r':.01,'s':.01,
            'imitation_weight':.1,'constraint_weight':100.,'domain':cls.domain,'original_evaluation':original}

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads);cls.temporary.cleanup()

    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        shutil.copytree(self.base/'prepared',self.root/'workspace');self.ws=Workspace(self.root/'workspace')

    def train(self,config=None,context=None,validation=None):
        artifact=train_restricted_pgac(self.ws,self.policy,self.model,copy.deepcopy(config or self.config),context,validation_id=validation or self.validation)
        root,meta=self.ws.resolve(artifact,kind='external_pgac_candidate');return artifact,root,meta,read_json(root/'report.json')

    def assert_nested_equal(self,actual,expected):
        if isinstance(expected,torch.Tensor):self.assertTrue(torch.equal(actual,expected))
        elif isinstance(expected,dict):
            self.assertEqual(set(actual),set(expected))
            for key in expected:self.assert_nested_equal(actual[key],expected[key])
        elif isinstance(expected,(tuple,list)):
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual,expected):self.assert_nested_equal(a,b)
        else:self.assertEqual(actual,expected)

    def test_actual_imitation_weights_only_raw_network_optimized_and_candidate_stays_inactive(self):
        baseline_root,_=self.ws.resolve(self.policy);before=file_hash(baseline_root/'bundle/model.pt');active=copy.deepcopy(self.ws.manifest['active'])
        _,root,meta,report=self.train()
        original=torch.load(baseline_root/'bundle/model.pt',weights_only=True);self.assert_nested_equal(torch.load(root/'initial_imitation_state.pt',weights_only=True),original)
        self.assertTrue(report['initialized_from_actual_imitation']);self.assertTrue(report['model_parameters_frozen']);self.assertEqual(report['optimizer_updates'],9)
        self.assertEqual(report['first_update_in_this_run']['state_sha256_before'],report['initial_weights_sha256']);self.assertEqual(report['first_update_in_this_run']['learning_rate'],.0005)
        self.assertEqual(report['training_trajectory_ids'],[0,1,2]);self.assertEqual(report['validation_trajectory_ids'],[3]);self.assertFalse(report['test_used_for_updates_or_checkpoint_selection'])
        self.assertEqual(report['original_plant_closed_loop'],'not_run');self.assertFalse(report['learned_critic']);self.assertFalse(report['ppo']);self.assertFalse(report['sac'])
        self.assertEqual(meta['activation'],'none');self.assertEqual(self.ws.manifest['active'],active);self.assertEqual(file_hash(baseline_root/'bundle/model.pt'),before)
        checkpoint=torch.load(root/'latest.pt',weights_only=True);self.assertEqual(set(checkpoint['model_state']),{'0.weight','0.bias','2.weight','2.bias'})
        self.assertTrue(any(not torch.equal(value,checkpoint['model_state'][key]) for key,value in original.items()))
        policy=load_external_policy(root);self.assertFalse(any(p.requires_grad for p in policy.network.parameters()))

    def test_cancel_resume_model_adam_rng_best_and_log_exact_and_changed_contract_rejects(self):
        _,root,_,_=self.train();expected=torch.load(root/'latest.pt',weights_only=True);best=torch.load(root/'model.pt',weights_only=True);events=[]
        def emit(event):
            events.append(event)
            if event.get('stage')=='restricted_pgac_training' and event.get('epoch')==1:raise Cancelled('Completed real PGAC epoch one')
        with self.assertRaises(Cancelled):self.train(context=OperationContext(emit=emit))
        checkpoint=Path(events[-1]['checkpoint']);self.assertTrue(checkpoint.is_file())
        for change in ({'learning_rate':.0006},{'q':1.1},{'domain':{'x':[[-2.,2.]],'u':[[-1.,1.]],'d':[]}}):
            with self.subTest(change=change),self.assertRaisesRegex(ContractError,'configuration differs'):
                self.train({**self.config,**change,'resume_checkpoint':str(checkpoint)})
        _,resumed,_,report=self.train({**self.config,'resume_checkpoint':str(checkpoint),'resume_checkpoint_sha256':file_hash(checkpoint)})
        self.assertEqual(report['resumed_from_epoch'],1);self.assert_nested_equal(torch.load(resumed/'latest.pt',weights_only=True),expected)
        self.assert_nested_equal(torch.load(resumed/'model.pt',weights_only=True),best)

    def test_epoch_zero_cancel_and_out_of_domain_failure_preserve_actual_baseline(self):
        baseline,_=self.ws.resolve(self.policy);before=file_hash(baseline/'bundle/model.pt');events=[]
        def emit(event):
            events.append(event)
            if event.get('stage')=='pgac_initial_checkpoint':raise Cancelled('Before any PGAC Adam step')
        with self.assertRaises(Cancelled):self.train(context=OperationContext(emit=emit))
        checkpoint=Path(events[-1]['checkpoint']);self.assertEqual(torch.load(checkpoint,weights_only=True)['epoch'],0)
        _,resumed,_,_=self.train({**self.config,'resume_checkpoint':str(checkpoint)})
        _,continuous,_,_=self.train();self.assert_nested_equal(torch.load(resumed/'latest.pt',weights_only=True),torch.load(continuous/'latest.pt',weights_only=True))
        with self.assertRaises(RolloutDomainError):self.train({**self.config,'domain':{'x':[[-.02,.02]],'u':[[-1.,1.]],'d':[]}})
        self.assertEqual(file_hash(baseline/'bundle/model.pt'),before);self.assertEqual(self.ws.manifest['active']['external_imitation_policy'],self.policy)
        zero=[path for path in (self.ws.path/'failed').glob('*/latest.pt') if torch.load(path,weights_only=True)['epoch']==0]
        self.assertGreaterEqual(len(zero),2)

    def test_unqualified_missing_receipt_and_oversized_rollout_are_rejected_without_training(self):
        with patch('control_harness.restricted_pgac.physical_rollout_loss',side_effect=AssertionError('No optimizer rollout expected')):
            with self.assertRaises(ContractError):self.train(validation='missing-qualification')
            from control_harness.external_imitation import train_external_imitation
            wide=train_external_imitation(self.ws,self.dataset,{'network':{'hidden_dims':[512],'activation':'tanh','epochs':1,'learning_rate':.004,'batch_size':8,'patience':20,'seed':20261129}})
            with self.assertRaisesRegex(ContractError,'activation budget'):
                train_restricted_pgac(self.ws,wide,self.model,{**self.config,'horizon':256,'batch_size':100_000},validation_id=self.validation)


if __name__=='__main__':unittest.main()
