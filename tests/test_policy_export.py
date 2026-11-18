"""Actual frozen tensors, independent numeric arithmetic and portable inference.

All networks/arrays here are explicit synthetic test fixtures, not a training
or original MATLAB/Simulink execution claim. MATLAB tests validate the emitted
numeric payload and supported-source boundary; native evaluation is separate.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from control_harness.contracts import ContractError, DataInterface
from control_harness.external_expert import expert_data_interface
from control_harness.external_policy import load_external_policy, project_actions
from control_harness.policy_export import export_external_policy
from control_harness.supervised_export import build_supervised_network, export_supervised


PROJECT=Path(__file__).resolve().parents[1]
OUTPUT=PROJECT/'test-output/policy-export-tests'
OUTPUT.mkdir(parents=True,exist_ok=True)


def fixture():
    v=lambda name,unit,bound:{'name':name,'unit':unit,'lower':-bound,'upper':bound}
    physical={'schema_version':1,'name':'synthetic explicit policy physical context','sample_time':.1,'time_semantics':'discrete_state_transition',
        'states':[v('x'+str(j),'m',2.) for j in range(4)],'controls':[v('u'+str(j),'N',1.5) for j in range(2)],
        'outputs':[v('y'+str(j),'m',2.) for j in range(2)],'disturbances':[v('wind','N',.3)]}
    observations=[{'name':'sensor'+str(j),'source':'state','index':j,'plant_name':'x'+str(j),'unit':'m',
        'availability':'measured' if j<3 else 'simulation_only','delay_steps':0} for j in range(4)]
    spec={'interface':physical,'control_period':.2,'observations':observations,'reference_available':True,'disturbance_available':True,'previous_action_available':True}
    selected=['observation::sensor'+str(j) for j in range(4)]+['reference::y0','reference::y1','disturbance::wind','previous_action::u0','previous_action::u1']
    interface,mapping=expert_data_interface(spec,{'student_features':selected})
    context={'interface':physical,'inferred_from_feature_width':False,'physical_mapping':{'observations':observations,'action_mapping':[
        {'action_index':0,'plant_control_index':0,'plant_control_name':'u0','unit':'N'},
        {'action_index':1,'plant_control_index':1,'plant_control_name':'u1','unit':'N'}],
        'plant_dt':.1,'control_period':.2,'plant_substeps':2},'plant_dt':.1,'control_period':.2,'plant_substeps':2}
    constraints={'bounds':[[-1.5,1.5],[-1.5,1.5]],'rate_bounds':[[-.25,.25],[-.25,.25]],'rate_semantics':'per_control_step'}
    return interface.to_dict(),context,constraints


def manual_raw(features,state,settings,scaler):
    value=(np.asarray(features)-scaler['X']['location'])/scaler['X']['scale']
    widths=[settings['input_dim'],*settings['hidden_dims'],settings['output_dim']]
    for j in range(len(widths)-1):
        value=value@state[str(2*j)+'.weight'].numpy().T+state[str(2*j)+'.bias'].numpy()
        if j<len(widths)-2:
            if settings['activation']=='tanh':value=np.tanh(value)
            elif settings['activation']=='relu':value=np.maximum(value,0.)
            else:value=value/(1.+np.exp(-value))
    return value*scaler['y']['scale']+scaler['y']['location']


class PolicyExportTests(unittest.TestCase):
    confirmation_source='test_fixture'

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def export(self,name='case',activation='tanh',data=None,context=None,constraints=None,hidden=None,none_context=False):
        default_data,default_context,default_constraints=fixture();data=copy.deepcopy(default_data if data is None else data)
        data=DataInterface.parse(data).to_dict();context=None if none_context else copy.deepcopy(default_context if context is None else context)
        constraints=copy.deepcopy(default_constraints if constraints is None else constraints)
        settings={'schema_version':1,'architecture':'mlp','input_dim':data['input_dim'],'output_dim':data['output_dim'],
            'hidden_dims':[5,3] if hidden is None else hidden,'activation':activation,'dtype':'float64'}
        with torch.random.fork_rng(devices=[]):network=build_supervised_network(settings)
        with torch.no_grad():
            for module in network:
                if isinstance(module,torch.nn.Linear):
                    module.weight.copy_(torch.linspace(-.3,.35,module.weight.numel(),dtype=torch.float64).reshape(module.weight.shape))
                    module.bias.zero_()
        scaler={'method':'standard','fit_split':'train','fit_samples':13,
            'X':{'location':np.linspace(-.15,.2,data['input_dim']).tolist(),'scale':np.linspace(.8,1.4,data['input_dim']).tolist(),'constant_columns':[]},
            'y':{'location':[.1,-.05],'scale':[1.5,.75],'constant_columns':[]}}
        source=self.root/name/'source';export_supervised(source/'bundle',network,settings,data,scaler,{'purpose':'synthetic test_fixture; no training'})
        destination=self.root/name/'portable';manifest=export_external_policy(destination,source,data,constraints,context,
            provenance={'confirmation_source':'test_fixture','supervised_model':'synthetic-source','workspace':str(self.root),'source_root':str(source)})
        return destination,manifest,network,settings,scaler,data,context,constraints,source

    def rehash(self,root,name,value):
        path=root/name
        if isinstance(value,(dict,list)):path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
        else:path.write_bytes(value)
        manifest=json.loads((root/'policy_manifest.json').read_text());manifest['files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'policy_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')

    def test_actual_tensors_numpy_inverse_scaler_projection_and_feature_labels_match_independent_arithmetic(self):
        for activation in ['tanh','relu','silu']:
            with self.subTest(activation=activation):
                root,manifest,network,settings,scaler,*_=self.export(activation,activation)
                policy=load_external_policy(root);features=np.linspace(-.4,.6,27).reshape(3,9);previous=np.array([[.1,-.2],[0.,.2],[-.1,.1]])
                expected=manual_raw(features,{k:v.detach() for k,v in network.state_dict().items()},settings,scaler)
                np.testing.assert_allclose(policy.predict_raw(features),expected,atol=1e-14,rtol=1e-13)
                np.testing.assert_allclose(policy.numpy_raw(features),expected,atol=1e-14,rtol=1e-13)
                projected=np.maximum(np.minimum(expected,np.minimum(1.5,previous+.25)),np.maximum(-1.5,previous-.25))
                np.testing.assert_allclose(policy.predict(features,previous),projected,atol=1e-14,rtol=1e-13)
                np.testing.assert_allclose(policy.project_actions(expected,previous),projected,atol=1e-14,rtol=1e-13)
                np.testing.assert_allclose(project_actions(expected,previous,policy.action_constraints),projected,atol=1e-14,rtol=1e-13)
                with self.assertRaises(ContractError):policy.predict(features,previous,list(reversed(policy.layout['feature_labels'])))
                self.assertFalse(any(p.requires_grad for p in policy.network.parameters()))
                self.assertEqual(manifest['physical_act']['binding']['availability_scope'],'simulation_only')

    def test_torch_input_and_previous_action_gradients_are_real_finite_differences_with_frozen_weights(self):
        root,_,_,_,scaler,*_=self.export();policy=load_external_policy(root)
        x=torch.tensor([scaler['X']['location']],dtype=torch.float64,requires_grad=True);previous=torch.zeros((1,2),dtype=torch.float64)
        coefficients=torch.tensor([[.7,-.4]],dtype=torch.float64);loss=(policy(x,previous)*coefficients).sum();gradient=torch.autograd.grad(loss,x)[0].detach().numpy()
        finite=np.zeros_like(gradient);h=1e-6
        for j in range(9):
            plus=x.detach().numpy().copy();minus=plus.copy();plus[0,j]+=h;minus[0,j]-=h
            finite[0,j]=np.sum((policy.predict(plus,previous)-policy.predict(minus,previous))*coefficients.numpy())/(2*h)
        self.assertGreater(float(np.linalg.norm(gradient)),1e-4);np.testing.assert_allclose(gradient,finite,atol=1e-9,rtol=1e-6)
        raw=torch.tensor([[3.,-3.]],dtype=torch.float64,requires_grad=True);prior=torch.tensor([[.1,-.2]],dtype=torch.float64,requires_grad=True)
        clipped=policy.project_tensor(raw,prior);grad_raw,grad_prior=torch.autograd.grad(clipped.sum(),(raw,prior))
        np.testing.assert_array_equal(grad_raw.numpy(),[[0.,0.]]);np.testing.assert_array_equal(grad_prior.numpy(),[[1.,1.]])
        for j in range(2):
            plus=prior.detach().numpy().copy();minus=plus.copy();plus[0,j]+=h;minus[0,j]-=h
            actual=(project_actions(raw.detach().numpy(),plus,policy.action_constraints)-project_actions(raw.detach().numpy(),minus,policy.action_constraints))/(2*h)
            np.testing.assert_allclose(actual,np.eye(2)[j:j+1],atol=1e-9)
        self.assertTrue(all(p.grad is None for p in policy.network.parameters()))

    def test_physical_act_uses_explicit_mapping_control_period_and_batch_shapes(self):
        root,manifest,*_=self.export();policy=load_external_policy(root)
        x=torch.tensor([.1,.2,-.1,.05],dtype=torch.float64,requires_grad=True);r=torch.tensor([.2,-.2],dtype=torch.float64)
        d=torch.tensor([.02],dtype=torch.float64);previous=torch.tensor([.1,-.1],dtype=torch.float64)
        assembled=torch.cat([x,r,d,previous]);action=policy.act(x,r,d,previous)
        torch.testing.assert_close(action,policy(assembled,previous),atol=1e-14,rtol=1e-13)
        self.assertTrue(action.requires_grad);self.assertEqual(policy.binding['plant_substeps'],2)
        self.assertEqual(policy.binding['plant_dt'],.1);self.assertEqual(policy.binding['control_period'],.2)
        batch=policy.act(x.repeat(2,1),r.repeat(2,1),d.repeat(2,1),previous.repeat(2,1));self.assertEqual(batch.shape,(2,2))
        with self.assertRaises(ContractError):policy.act(x.repeat(2,1),r,d,previous)
        self.assertEqual(policy.reset()['hidden_state'],'none');self.assertEqual(policy.decisions,0)
        self.assertTrue(manifest['physical_act']['supported'])

    def test_missing_actual_previous_wrong_shape_nonfinite_and_infeasible_projection_are_rejected(self):
        root,*_=self.export();policy=load_external_policy(root);x=np.zeros((2,9));previous=np.zeros((2,2))
        for args in [(x,None),(x,np.zeros(2)),(x,[[True,0],[0,0]])]:
            with self.assertRaises(ContractError):policy.predict(*args)
        for invalid in [np.full((2,9),np.nan),np.zeros((2,1,9)),[['0']*9]*2,[[False]*9]*2]:
            with self.assertRaises(ContractError):policy.predict(invalid,previous)
        with self.assertRaises(ContractError):project_actions([[0.,0.]],[[1.5,1.5]],{'bounds':[[-1.5,1.5]]*2,'rate_bounds':[[.1,.2]]*2,'rate_semantics':'per_control_step'})
        with self.assertRaises(ContractError):policy.project_actions([[np.inf,0.]],[[0.,0.]])
        no_rate={'bounds':[[-1.5,1.5]]*2,'rate_bounds':None,'rate_semantics':'per_control_step'}
        np.testing.assert_array_equal(project_actions([[2.,-2.]],None,no_rate),[[1.5,-1.5]])

    def test_history_delay_feature_order_is_exact_caller_managed_and_declines_physical_matlab_adapter(self):
        data,context,constraints=fixture();data['signals'][0]['delay_steps']=1;data['features'][0].update(window=3,lag=2)
        data['task']='dynamic_prediction'
        data.pop('input_dim');data.pop('output_dim');data=DataInterface.parse(data).to_dict()
        root,manifest,*_=self.export('history',data=data,context=context,constraints=constraints);policy=load_external_policy(root)
        self.assertEqual([f['lag'] for f in policy.layout['feature_order'][:3]],[5,4,3])
        self.assertEqual([f['signal'] for f in policy.layout['feature_order'][:3]],['observation::sensor0']*3)
        values=np.linspace(-.1,.2,2*data['input_dim']).reshape(2,data['input_dim']);self.assertEqual(policy.predict(values,np.zeros((2,2))).shape,(2,2))
        self.assertEqual(manifest['matlab']['status'],'unsupported');self.assertIn('history',manifest['matlab']['reason'])
        self.assertFalse(manifest['physical_act']['supported'])
        with self.assertRaises(ContractError):policy.act(np.zeros(4),np.zeros(2),np.zeros(1),np.zeros(2))
        self.assertIn('no implicit priming',policy.reset()['history'])

    def test_data_only_policy_does_not_invent_physical_dimensions_or_require_a_plant(self):
        root,manifest,*_=self.export('data-only',none_context=True);policy=load_external_policy(root)
        self.assertIsNone(policy.physical_context['interface']);self.assertEqual(policy.predict(np.zeros((2,9)),np.zeros((2,2))).shape,(2,2))
        self.assertFalse(manifest['physical_act']['supported']);self.assertEqual(manifest['matlab']['status'],'unsupported')
        self.assertEqual(manifest['deployment_scope'],'simulation_only');self.assertEqual(policy.describe()['deployment_scope'],'simulation_only')
        with self.assertRaises(ContractError):policy.act(np.zeros(4),np.zeros(2),np.zeros(1),np.zeros(2))

    def test_whole_bundle_copied_outside_repo_new_isolated_process_needs_no_harness_original_sources(self):
        root,manifest,*tail=self.export();policy=load_external_policy(root);features=np.linspace(-.3,.5,18).reshape(2,9);previous=np.array([[.1,-.2],[0.,.2]])
        expected=policy.predict(features,previous);source=tail[-1];shutil.rmtree(source)
        outside_root=PROJECT.parent/'external-policy-portable-tests';outside_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='external-policy-standalone-',dir=outside_root) as outside:
            outside=Path(outside);portable=outside/'portable';shutil.copytree(root,portable)
            input_file=outside/'input.json';output=outside/'result.json';input_file.write_text(json.dumps({'features':features.tolist(),'previous_action':previous.tolist(),'feature_labels':policy.layout['feature_labels']}),encoding='utf-8')
            environment={key:value for key,value in os.environ.items() if key!='PYTHONPATH'}
            result=subprocess.run([sys.executable,'-I','-B',str(portable/'infer.py'),'--input',str(input_file),'--output',str(output)],cwd=outside,env=environment,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr);actual=json.loads(output.read_text());np.testing.assert_allclose(actual['action'],expected,atol=1e-13,rtol=1e-12)
            blocker="""import importlib.abc,importlib.util,json,sys\nfrom pathlib import Path\nimport torch\nclass Block(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.startswith('control_harness'):raise RuntimeError('No project imports allowed')\nsys.meta_path.insert(0,Block())\nroot=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('portable_policy',root/'policy_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);policy=module.load_external_policy(root);x=torch.zeros((1,policy.input_dim),dtype=torch.float64,requires_grad=True);p=torch.zeros((1,policy.output_dim),dtype=torch.float64);y=policy.raw_forward(x);g=torch.autograd.grad(y.sum(),x)[0];print(json.dumps({'finite':bool(torch.isfinite(g).all()),'gradient':g.tolist(),'frozen':all(not v.requires_grad for v in policy.network.parameters())}))"""
            result=subprocess.run([sys.executable,'-I','-B','-c',blocker,str(portable)],cwd=outside,env=environment,capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,0,result.stderr);result=json.loads(result.stdout);self.assertTrue(result['finite'] and result['frozen'])
            for path in portable.glob('*.json'):self.assertNotIn(str(self.root),path.read_text())

    def test_numeric_matlab_assets_match_same_torch_weights_and_static_six_input_three_output_boundary(self):
        from control_harness.matlab_expert import validate_matlab_expert_source
        for activation in ['tanh','relu','silu']:
            with self.subTest(activation=activation):
                root,manifest,*_=self.export('matlab-'+activation,activation);policy=load_external_policy(root);matlab=manifest['matlab'];self.assertEqual(matlab['status'],'supported')
                p=matlab['parameters'];x=np.linspace(-.2,.4,9);previous=x[-2:];value=(x-np.asarray(p['x_location']))/np.asarray(p['x_scale'])
                for k,layer in enumerate(p['layers']):
                    value=np.asarray(layer['W'])@value+np.asarray(layer['b'])
                    if k<len(p['layers'])-1:
                        if p['activation_code']==1:value=np.tanh(value)
                        elif p['activation_code']==2:value=np.maximum(value,0.)
                        else:value=value/(1+np.exp(-value))
                raw=value*np.asarray(p['y_scale'])+np.asarray(p['y_location']);expected=project_actions(raw,previous,policy.action_constraints)
                np.testing.assert_allclose(raw,policy.predict_raw(x),atol=1e-14,rtol=1e-13);np.testing.assert_allclose(expected,policy.predict(x,previous),atol=1e-14,rtol=1e-13)
                snapshot=self.root/('matlab-source-'+activation);(snapshot/'source').mkdir(parents=True);target=snapshot/'source/external_policy_nn.m';shutil.copyfile(root/'external_policy_nn.m',target)
                support=validate_matlab_expert_source(snapshot,matlab['adapter'],{'source/external_policy_nn.m':hashlib.sha256(target.read_bytes()).hexdigest()})
                self.assertEqual(support['status'],'supported_subset');self.assertFalse(support['source_executed'])
                self.assertEqual(matlab['adapter']['initial_memory'],{'calls':0});self.assertEqual(matlab['adapter']['parameters'],p)

    def test_metadata_unit_order_scaler_and_hash_conflicts_are_refused(self):
        root,*_=self.export();original=json.loads((root/'physical_context.json').read_text())
        corrupted=self.root/'bad-hash';shutil.copytree(root,corrupted);(corrupted/'network.json').write_text('{}')
        with self.assertRaises(ContractError):load_external_policy(corrupted)
        cases=[]
        scaler=json.loads((root/'scaler.json').read_text());scaler['X']['scale'][0]=0.;cases.append(('scaler.json',scaler))
        context=copy.deepcopy(original);context['plant_substeps']=1;cases.append(('physical_context.json',context))
        context=copy.deepcopy(original);context['physical_mapping']['observations'][0]['unit']='K';cases.append(('physical_context.json',context))
        data=json.loads((root/'data_interface.json').read_text());data['targets'].reverse();cases.append(('data_interface.json',data))
        data=json.loads((root/'data_interface.json').read_text());data['source_mapping']['signals']['previous_action::u0'].pop('time_semantics');cases.append(('data_interface.json',data))
        for j,(name,value) in enumerate(cases):
            destination=self.root/('semantic-'+str(j));shutil.copytree(root,destination);self.rehash(destination,name,value)
            with self.subTest(name=name),self.assertRaises(ContractError):load_external_policy(destination)

    def test_batch_resource_gate_precedes_network_and_does_not_run_another_test_forward_for_projection(self):
        root,*_=self.export('budget',hidden=[128,128]);policy=load_external_policy(root)
        values=np.zeros((20000,9))
        with patch.object(policy.network,'forward',side_effect=AssertionError('Must reject before network forward')):
            with self.assertRaisesRegex(ContractError,'activation resource budget'):policy.predict_raw(values)
            with self.assertRaisesRegex(ContractError,'activation resource budget'):policy.numpy_raw(values)
            projected=policy.project_actions(np.array([[3.,-3.]]),np.zeros((1,2)));np.testing.assert_array_equal(projected,[[.25,-.25]])

    def test_separate_bound_dataset_mapping_survives_real_csv_import_training_and_bundle_without_interface_mutation(self):
        from control_harness.artifacts import Workspace,read_json
        from control_harness.external_expert_data import import_expert_data
        from control_harness.external_imitation import train_external_imitation
        from control_harness.material_registration import register_material
        from control_harness.runtime import OperationContext
        path=self.root/'separate-map.csv';rows=[]
        for trip in range(6):
            previous=.1
            for step in range(8):
                observation=-.4+.07*step+.03*trip;action=.3*observation+.15*previous
                rows.append([step*.2,trip,observation,previous,action]);previous=action
        with path.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['time','trip','observation','previous','action']);writer.writerows(rows)
        def signal(name,index,role,expert_role):return {'name':name,'unit':'N' if name!='observation' else 'm','role':role,'expert_role':expert_role,'observable':True,'delay_steps':0,'column':{'index':index,'header':name,'worksheet':None}}
        data=DataInterface.parse({'schema_version':1,'interface_kind':'tabular_supervised','task':'static_regression','sample_time':.2,
            'signals':[signal('observation',2,'auxiliary','observation'),signal('previous',3,'auxiliary','previous_action'),signal('action',4,'control','expert_action')],
            'features':[{'signal':name,'window':1,'lag':0} for name in ['observation','previous']],'targets':[{'signal':'action','offset':0}],
            'time_column':{'index':0,'header':'time','worksheet':None},'trip_column':{'index':1,'header':'trip','worksheet':None}}).to_dict()
        self.assertNotIn('source_mapping',data);original=copy.deepcopy(data)
        mapping={'signals':{'previous':{'time_semantics':'already_previous_applied_action','action_target':'action'}}}
        ws=Workspace(self.root/'actual-crosscomponent');context=OperationContext();material=register_material(ws,str(path),{},context)
        dataset=import_expert_data(ws,material,{'interface':data,'source_mapping':mapping,'split':{'mode':'trip','train':.5,'validation':1/6,'test':1/3},'scaler':'standard',
            'source':{'kind':'recorded_external_expert','description':'Synthetic test_fixture recorded causal action recurrence, no hardware claim','previous_action_semantics':'actual_applied_action_before_current_decision'},
            'action_constraints':{'bounds':[[-2.,2.]],'rate_bounds':[[-4.,4.]],'rate_semantics':'per_control_step'}},context)
        source_root,source_meta=ws.resolve(dataset);source_definition=read_json(source_root/'dataset.json')
        self.assertNotIn('source_mapping',source_meta['interface']);self.assertEqual(source_definition['source_mapping']['signals']['previous']['action_target'],'action')
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:artifact=train_external_imitation(ws,dataset,{'network':{'hidden_dims':[4],'activation':'tanh','epochs':2,'batch_size':8,'patience':10,'learning_rate':.003,'seed':51},'action_metric_scale':[1.]},context)
        finally:torch.set_num_threads(old_threads)
        root,meta=ws.resolve(artifact,kind='external_imitation_policy');descriptor=read_json(root/'policy.json');policy=load_external_policy(root)
        self.assertEqual(data,original);self.assertEqual(meta['interface'],source_meta['interface'])
        self.assertEqual(read_json(root/'bundle/data_interface.json'),source_meta['interface']);self.assertNotIn('source_mapping',policy.data_interface)
        self.assertEqual(policy.deployment_mapping['signals']['previous']['time_semantics'],'already_previous_applied_action')
        for name in ['supervised_dataset','supervised_model']:
            member,metadata=ws.resolve(descriptor[name]);self.assertEqual(metadata['interface'],source_meta['interface'])
            if name=='supervised_model':self.assertEqual(read_json(member/'bundle/data_interface.json'),source_meta['interface'])
        with np.load(source_root/'expert.npz',allow_pickle=False) as raw,np.load(root/'test_predictions.npz',allow_pickle=False) as saved:
            index=np.flatnonzero(raw['split_id']==2);np.testing.assert_allclose(policy.predict(raw['X'][index],raw['actual_previous_u'][index]),saved['projected_action'],atol=1e-14,rtol=0)

    def test_embedded_and_bound_mapping_conflicts_reject_but_nonconflicting_enrichment_is_preserved(self):
        root,manifest,_,_,_,data,context,constraints,source=self.export('map-base')
        mapping=copy.deepcopy(data['source_mapping']);mapping['signals']['previous_action::u0']['original_source_column']={'index':7,'header':'prior_command'}
        external=self.root/'map-enriched';export_external_policy(external,source,data,constraints,context,provenance={'source_mapping':mapping})
        policy=load_external_policy(external);self.assertEqual(policy.deployment_mapping['signals']['previous_action::u0']['original_source_column']['index'],7)
        self.assertEqual(json.loads((external/'data_interface.json').read_text()),data)
        conflict=copy.deepcopy(mapping);conflict['signals']['previous_action::u0']['action_target']='expert_action::u1'
        with self.assertRaisesRegex(ContractError,'Conflicting'):export_external_policy(self.root/'map-conflict',source,data,constraints,context,provenance={'source_mapping':conflict})
        with self.assertRaisesRegex(ContractError,'mapping'):export_external_policy(self.root/'bad-map-type',source,data,constraints,context,provenance={'source_mapping':{'signals':[]}})


if __name__=='__main__':unittest.main()
