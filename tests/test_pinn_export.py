"""Real autograd/optimizer and independently portable conditional thermal NN tests."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np
import torch

from control_harness import pinn_contract as pc
from control_harness.contracts import ContractError
from control_harness.pinn_runtime import ConditionalThermalPINN,conditional_network,load_pinn,PINN_MAX_ACTIVATION_VALUES
from control_harness.pinn_export import export_pinn
from control_harness.supervised_export import build_supervised_network


ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'test-output/pinn-export-tests'
PORTABLE=ROOT.parent/'control-agent-harness-pinn-portable-tests'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
write=lambda path,value:path.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n',encoding='utf-8')


class PinnExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous=torch.get_num_threads();torch.set_num_threads(1);cls.addClassCleanup(torch.set_num_threads,previous)

    def setUp(self):
        OUTPUT.mkdir(parents=True,exist_ok=True);self.root=OUTPUT/uuid.uuid4().hex;self.root.mkdir()
        self.addCleanup(self.cleanup_root);self.source=self.root/'source';self.source.mkdir();self.definition=self.make_definition()

    def cleanup_root(self):
        assert self.root.resolve().is_relative_to(OUTPUT.resolve());shutil.rmtree(self.root,ignore_errors=True)

    def make_definition(self,constant_train=False):
        # The independent analytic expression exists only in this test's CSV
        # generator, never in the trained network, runtime or export source.
        conditions=[{'id':'fit','split':'train','initial_temperature_K':300.,'power_W':40.},
            {'id':'validation','split':'validation','initial_temperature_K':305.,'power_W':60.},
            {'id':'heldout','split':'test','initial_temperature_K':310.,'power_W':80.}]
        if constant_train:conditions[0].update(initial_temperature_K=290.,power_W=0.)
        reference_headers=['condition_id','split','time_s','initial_temperature_K','power_W','temperature_K']
        collocation_headers=['condition_id','time_s','initial_temperature_K','power_W'];training_labels=[]
        with (self.source/'reference.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(reference_headers)
            for condition in conditions:
                for time in [2.,3.,4.5,6.]:
                    temperature=self.analytic(time,condition['initial_temperature_K'],condition['power_W'])
                    writer.writerow([condition['id'],condition['split'],time,condition['initial_temperature_K'],condition['power_W'],temperature])
                    if condition['split']=='train':training_labels.append(temperature)
        with (self.source/'collocation.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(collocation_headers)
            for time in np.linspace(2.,6.,12):writer.writerow(['fit',float(time),300.,40.])
        selectors=lambda names:{name:{'index':index,'header':name} for index,name in enumerate(names)}
        config={'schema_version':1,'equation':{'template':'thermal_first_order','forcing':'constant_per_condition',
            'parameters':{'C_J_per_K':20.,'G_W_per_K':2.,'Ta_K':290.}},
            'domain':{'time_s':[2.,6.],'initial_temperature_K':[285.,320.],'power_W':[0.,100.]},
            'normalization':{'time_origin_s':2.,'time_scale_s':4.,'state_offset_K':290.,'state_scale_K':20.,'power_scale_W':100.,'physics_scale_W':100.},
            'units':dict(pc.UNITS),'conditions':conditions,'reference_file':'reference.csv','collocation_file':'collocation.csv',
            'reference_columns':selectors(reference_headers),'collocation_columns':selectors(collocation_headers),
            'source':{'kind':'synthetic_analytic','description':'Explicit independent component-test thermal CSV, not a native F04 acceptance run'}}
        definition=pc.validate_pinn_config(config)['definition']
        constant=all(value==training_labels[0] for value in training_labels);population=0. if constant else float(np.std(training_labels))
        definition['normalization'].update(data_nrmse_scale_K=20. if constant else population,fit_split='train',fit_samples=len(training_labels),ddof=0,
            data_population_std_K=population,data_nrmse_scale_source='engineering_state_scale_for_exact_constant_training_targets' if constant else 'training_population_std',
            data_nrmse_fallback_K=20. if constant else None,exact_constant_training_targets=constant)
        files={name:sha(self.source/name) for name in ['reference.csv','collocation.csv']}
        source={'kind':config['source']['kind'],'description':config['source']['description'],'material':{'id':'explicit-component-fixture',
            'metadata_hash':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),'content_hash':hashlib.sha256(json.dumps(files).encode()).hexdigest()},
            'reference':{'file':'reference.csv','sha256':files['reference.csv'],'columns':config['reference_columns']},
            'collocation':{'file':'collocation.csv','sha256':files['collocation.csv'],'columns':config['collocation_columns']},'provenance_is_user_declared':True}
        definition['source']=source;definition['resource_boundary']={'max_conditions':pc.MAX_CONDITIONS,'max_points_per_table':pc.MAX_POINTS,
            'max_logical_values':pc.MAX_LOGICAL_VALUES,'max_npz_bytes':pc.MAX_NPZ_BYTES,'scope':'one problem artifact; not a limit on the overall engineering objective'}
        source['pinn_signals']=pc.pinn_signals(definition)
        return pc.validate_pinn_definition(definition)

    @staticmethod
    def analytic(time,initial,power):
        equilibrium=290.+power/2.
        return equilibrium+(initial-equilibrium)*math.exp(-2.*(time-2.)/20.)

    def network(self,hidden=None,activation='tanh',seed=123):
        config={'schema_version':1,'architecture':'mlp','input_dim':3,'output_dim':1,'hidden_dims':[6,5] if hidden is None else hidden,
            'activation':activation,'dtype':'float64'}
        with torch.random.fork_rng(devices=[]):torch.manual_seed(seed);network=build_supervised_network(config)
        return network,config

    def query(self):return [torch.tensor(value,dtype=torch.float64,requires_grad=True) for value in [[2.5,4.,5.5],[299.,305.,312.],[30.,55.,78.]]]

    def export(self,name='bundle',provenance=None,hidden=None,activation='tanh',parameterization='free_mlp'):
        network,config=self.network(hidden,activation);path=self.root/name
        manifest=export_pinn(path,network,self.definition,config,training_config={'epochs':10,'conditional_parameterization':parameterization,'resume_checkpoint':str(self.root/'private.pt')},
            provenance=provenance or {'scope':'Untrained explicit component NN; export makes no learned accuracy/physics claim','workspace':str(self.root)})
        return path,manifest,network,config

    def test_temperature_input_gradients_match_physical_finite_differences(self):
        network,_=self.network();model=ConditionalThermalPINN(network,self.definition);inputs=self.query()
        actual=model.temperature(*inputs);gradients=torch.autograd.grad(actual.sum(),inputs)
        for column,gradient in enumerate(gradients):
            h=1e-5;plus=[x.detach().numpy().copy() for x in inputs];minus=[x.detach().numpy().copy() for x in inputs];plus[column]+=h;minus[column]-=h
            finite_difference=(model.predict(*plus)-model.predict(*minus))/(2*h)
            np.testing.assert_allclose(gradient.numpy(),finite_difference,atol=2e-7,rtol=2e-6)
        self.assertEqual(actual.shape,(3,));self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertFalse(hasattr(model,'step'));self.assertFalse(hasattr(model,'interface'))

    def test_physics_residual_has_full_time_chain_and_independent_scalar_broadcast(self):
        network,_=self.network();model=ConditionalThermalPINN(network,self.definition);time,initial,power=self.query()
        temperature=model.temperature(time,initial,power);dT=torch.autograd.grad(temperature.sum(),time,create_graph=True)[0]
        expected=(20*dT+2*(temperature-290)-power)/100
        torch.testing.assert_close(model.residual(time,initial,power,create_graph=True),expected,atol=0,rtol=0)
        h=1e-5;numeric_derivative=(model.predict(time.detach().numpy()+h,initial.detach().numpy(),power.detach().numpy())-
            model.predict(time.detach().numpy()-h,initial.detach().numpy(),power.detach().numpy()))/(2*h)
        np.testing.assert_allclose(model.residual(time,initial,power).detach().numpy(),
            (20*numeric_derivative+2*(temperature.detach().numpy()-290)-power.detach().numpy())/100,atol=1e-7,rtol=1e-6)
        common=torch.tensor(3.,dtype=torch.float64,requires_grad=True)
        together=model.residual(common,initial,power,create_graph=True)
        independent=torch.stack([model.residual(common,initial[k],power[k],create_graph=True) for k in range(3)])
        torch.testing.assert_close(together,independent,atol=1e-14,rtol=1e-14)
        columns=model.residual(time.reshape(-1,1),initial.reshape(-1,1),power.reshape(-1,1),normalized=False)
        torch.testing.assert_close(columns.squeeze(-1),expected*100,atol=1e-13,rtol=1e-13)

    def test_create_graph_retains_derivative_only_weight_gradient_and_residual_fd(self):
        network,_=self.network(hidden=[5],seed=91);model=ConditionalThermalPINN(network,self.definition,freeze=False);inputs=self.query()
        temperature=model.temperature(*inputs);derivative=torch.autograd.grad(temperature.sum(),inputs[0],create_graph=True)[0]
        gradient=torch.autograd.grad(derivative.square().mean(),network[0].weight,retain_graph=True)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().max()),1e-8)
        loss=model.residual(*inputs,create_graph=True).square().mean();actual=torch.autograd.grad(loss,network[0].weight)[0][0,0].item()
        original=network[0].weight[0,0].item();h=1e-5
        with torch.no_grad():network[0].weight[0,0]=original+h
        plus=model.residual(*inputs,create_graph=True).square().mean().item()
        with torch.no_grad():network[0].weight[0,0]=original-h
        minus=model.residual(*inputs,create_graph=True).square().mean().item()
        with torch.no_grad():network[0].weight[0,0]=original
        self.assertAlmostEqual(actual,(plus-minus)/(2*h),delta=1e-7)

    def test_short_real_optimizer_uses_data_initial_and_connected_physics_loss(self):
        network,_=self.network(hidden=[12,12],seed=17);model=ConditionalThermalPINN(network,self.definition,freeze=False)
        time=torch.linspace(2.,6.,25,dtype=torch.float64);initial=torch.full_like(time,300.);power=torch.full_like(time,40.)
        reference=torch.tensor([self.analytic(float(t),300.,40.) for t in time],dtype=torch.float64)
        optimizer=torch.optim.Adam(network.parameters(),lr=.01)
        def losses():
            data=((model.temperature(time,initial,power)-reference)/20).square().mean()
            physics=model.residual(time,initial,power,create_graph=True).square().mean()
            ic=((model.temperature(torch.tensor(2.,dtype=torch.float64),torch.tensor(300.,dtype=torch.float64),torch.tensor(40.,dtype=torch.float64))-300)/20).square()
            return data,physics,ic
        before=[value.item() for value in losses()];old=[p.detach().clone() for p in network.parameters()]
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True);terms=losses();loss=sum(terms);loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),10.,error_if_nonfinite=True);optimizer.step()
        after=[value.item() for value in losses()]
        self.assertLess(sum(after),sum(before)*.1);self.assertLess(after[1],before[1]);self.assertTrue(any(not torch.equal(a,b) for a,b in zip(old,network.parameters())))
        self.assertTrue(optimizer.state_dict()['state']);self.assertGreater(before[1],0)
        # Positive physics provenance is tested with the actually optimized
        # component NN, rather than labeling an untrained fixture a PINN.
        bundle=self.root/'trained-component-bundle'
        manifest=export_pinn(bundle,model,self.definition,model.network_config,training_config={'component_updates':160},
            provenance={'physics_loss_enabled':True,'physics_off_ablation':False,'create_graph_for_physics_training':True,'is_pinn':True,
                'scope':'Short real component physics training; not F04/native acceptance'})
        self.assertEqual(manifest['training_classification'],'conditional_pinn_training_declared')
        torch.testing.assert_close(load_pinn(bundle).temperature(time,initial,power),model.temperature(time,initial,power),atol=0,rtol=0)

    def test_export_roundtrip_has_raw_state_keys_and_preserves_original_training_state(self):
        network,config=self.network();network.train();before=torch.get_rng_state().clone();state={k:v.detach().clone() for k,v in network.state_dict().items()}
        path=self.root/'bundle';export_pinn(path,ConditionalThermalPINN(network,self.definition,freeze=False),self.definition,config)
        self.assertTrue(network.training);self.assertTrue(all(p.requires_grad for p in network.parameters()));torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
        saved=torch.load(path/'model.pt',map_location='cpu',weights_only=True);self.assertEqual(set(saved),set(state));self.assertFalse(any(key.startswith('network.') for key in saved))
        loaded=load_pinn(path);inputs=self.query()
        original=ConditionalThermalPINN(network,self.definition,freeze=False)
        torch.testing.assert_close(loaded.temperature(*inputs),original.temperature(*inputs),atol=0,rtol=0)
        torch.testing.assert_close(loaded.residual(*inputs,create_graph=True),original.residual(*inputs,create_graph=True),atol=0,rtol=0)
        write(self.root/'pinn.json',{'schema_version':1,'bundle':'bundle'});torch.testing.assert_close(load_pinn(self.root).temperature(*inputs),loaded.temperature(*inputs),atol=0,rtol=0)

    def test_repo_external_new_isolated_process_preserves_outputs_gradients_residual_and_cli(self):
        path,_,_,_=self.export();model=load_pinn(path);inputs=self.query();expected=model.temperature(*inputs)
        gradients=torch.autograd.grad(expected.sum(),inputs);residual=model.residual(*inputs,create_graph=True)
        PORTABLE.mkdir(parents=True,exist_ok=True);external=PORTABLE/uuid.uuid4().hex;external.mkdir()
        def cleanup():
            assert external.resolve().is_relative_to(PORTABLE.resolve()) and not external.resolve().is_relative_to(ROOT.resolve());shutil.rmtree(external,ignore_errors=True)
        self.addCleanup(cleanup);bundle=external/'bundle';shutil.copytree(path,bundle)
        assert path.resolve().is_relative_to(self.root.resolve()) and self.source.resolve().is_relative_to(self.root.resolve())
        shutil.rmtree(path);shutil.rmtree(self.source)
        query={name:value.detach().tolist() for name,value in zip(['time_s','initial_temperature_K','power_W'],inputs)};write(external/'query.json',query)
        code=r'''import importlib.abc,importlib.util,json,sys,torch
from pathlib import Path
class RejectHarness(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'):raise ModuleNotFoundError('Independent PINN must not import project code')
sys.meta_path.insert(0,RejectHarness());torch.set_num_threads(1);root=Path(sys.argv[1]);query=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('_independent_pinn',root/'pinn_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.load_pinn(root)
inputs=[torch.tensor(query[k],dtype=torch.float64,requires_grad=True) for k in ['time_s','initial_temperature_K','power_W']];temperature=model.temperature(*inputs);grads=torch.autograd.grad(temperature.sum(),inputs);residual=model.residual(*inputs,create_graph=True)
assert not hasattr(model,'step') and not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)
print(json.dumps({'temperature':temperature.tolist(),'gradients':[g.tolist() for g in grads],'residual':residual.tolist(),'capability':model.capability},allow_nan=False))'''
        env={**os.environ,'PYTHONPATH':'','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
        process=subprocess.run([sys.executable,'-I','-B','-c',code,str(bundle),str(external/'query.json')],cwd=external,env=env,text=True,encoding='utf-8',capture_output=True,check=True)
        actual=json.loads(process.stdout);np.testing.assert_allclose(actual['temperature'],expected.detach().numpy(),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(actual['gradients'],[g.numpy() for g in gradients],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(actual['residual'],residual.detach().numpy(),atol=1e-12,rtol=1e-12)
        cli=subprocess.run([sys.executable,'-I','-B',str(bundle/'predict.py'),'--input',str(external/'query.json'),'--output',str(external/'out.json')],cwd=external,env=env,text=True,encoding='utf-8',capture_output=True,check=True)
        self.assertEqual(cli.returncode,0);np.testing.assert_allclose(read(external/'out.json')['temperature_K'],expected.detach().numpy(),atol=1e-12,rtol=1e-12)

    def test_domain_dtype_shape_and_query_budget_reject_before_network_execution(self):
        network,_=self.network(hidden=[6]);model=ConditionalThermalPINN(network,self.definition);calls=[];handle=network.register_forward_pre_hook(lambda *_:calls.append(1));self.addCleanup(handle.remove)
        scalar=lambda value:torch.tensor(value,dtype=torch.float64)
        for inputs in [(torch.tensor(3.,dtype=torch.float32),scalar(300),scalar(40)),(scalar(1.9),scalar(300),scalar(40)),
                       (scalar(3),scalar(0),scalar(40)),(scalar(3),scalar(300),scalar(float('nan'))),
                       (torch.ones((2,2),dtype=torch.float64)*3,scalar(300),scalar(40)),
                       (torch.tensor([3.,4.],dtype=torch.float64),torch.tensor([[300.],[300.]],dtype=torch.float64),scalar(40))]:
            with self.subTest(inputs=inputs),self.assertRaises(ContractError):model.temperature(*inputs)
        count=PINN_MAX_ACTIVATION_VALUES//(model._width_sum*4)+1
        with self.assertRaisesRegex(ContractError,'activation budget'):model.temperature(torch.full((count,),3.,dtype=torch.float64),scalar(300),scalar(40))
        self.assertEqual(calls,[])
        for bad in [True,[True,False],['3.'],[float('inf')]]:
            with self.subTest(value=bad),self.assertRaises(ContractError):model.predict(bad,300.,40.)
        self.assertEqual(model.predict([3.,4.],300.,40.).shape,(2,));self.assertEqual(model.predict([[3.],[4.]],300.,40.).shape,(2,1))

    def test_tampered_bundle_budget_capability_domain_and_ablation_claims_are_rejected(self):
        path,_,_,_=self.export()
        for mode in ['weights','capability','domain','network_budget']:
            target=self.root/mode;shutil.copytree(path,target);manifest=read(target/'pinn_manifest.json')
            if mode=='weights':(target/'model.pt').write_bytes((target/'model.pt').read_bytes()+b'tamper')
            elif mode=='capability':manifest['capability']['one_step_transition']=True;write(target/'pinn_manifest.json',manifest)
            elif mode=='domain':manifest['domain']['power_W']=[0,1000];write(target/'pinn_manifest.json',manifest)
            else:
                config=read(target/'network.json');config['hidden_dims']=[4096]*8;write(target/'network.json',config);manifest['files']['network.json']=sha(target/'network.json');write(target/'pinn_manifest.json',manifest)
            with self.subTest(mode=mode),self.assertRaises((ContractError,ValueError)):load_pinn(target)
        with self.assertRaises(ContractError):self.export('invalid-training',provenance={'physics_loss_enabled':True,'create_graph_for_physics_training':False})
        with self.assertRaises(ContractError):self.export('invalid-ablation',provenance={'physics_loss_enabled':False,'physics_off_ablation':True,'create_graph_for_physics_training':False,'is_pinn':True})
        with self.assertRaises(ContractError):self.export('contradictory-pinn-label',provenance={'physics_loss_enabled':True,'create_graph_for_physics_training':True,'is_pinn':False})

    def test_physics_off_export_is_explicit_and_matches_silu_or_linear_weights(self):
        provenance={'physics_loss_enabled':False,'physics_off_ablation':True,'create_graph_for_physics_training':False,'is_pinn':False}
        for name,hidden,activation in [('silu',[4],'silu'),('linear',[],'tanh')]:
            path,manifest,network,_=self.export(name,provenance=provenance,hidden=hidden,activation=activation)
            self.assertEqual(manifest['training_classification'],'conditional_mlp_physics_off_ablation');self.assertFalse(manifest['physics_loss_enabled'])
            loaded=load_pinn(path);self.assertTrue(loaded.training_declaration['physics_off_ablation']);self.assertNotIn('resume_checkpoint',read(path/'training.json')['config'])
            model=ConditionalThermalPINN(network,self.definition,freeze=False);inputs=self.query();torch.testing.assert_close(loaded.temperature(*inputs),model.temperature(*inputs),atol=0,rtol=0)
        bad=read(path/'pinn_manifest.json');bad['training_classification']='conditional_pinn_training_declared';write(path/'pinn_manifest.json',bad)
        with self.assertRaisesRegex(ContractError,'ablation labels'):load_pinn(path)

    def test_constant_training_reference_engineering_fallback_survives_portable_contract(self):
        definition=self.make_definition(constant_train=True);network,config=self.network(hidden=[4])
        path=self.root/'constant-bundle';export_pinn(path,network,definition,config)
        model=load_pinn(path);normalization=model.definition['normalization']
        self.assertTrue(normalization['exact_constant_training_targets']);self.assertEqual(normalization['data_population_std_K'],0.)
        self.assertEqual(normalization['data_nrmse_scale_K'],20.);self.assertEqual(normalization['data_nrmse_fallback_K'],20.)
        self.assertEqual(normalization['data_nrmse_scale_source'],'engineering_state_scale_for_exact_constant_training_targets')
        self.assertTrue(np.isfinite(model.predict([2.5,3.],290.,0.)).all())

    def test_affine_helper_shares_parameters_rng_and_three_real_corner_queries(self):
        network,_=self.network();network.train();before=torch.get_rng_state().clone();parameters=list(network.parameters())
        self.assertIs(conditional_network(network,self.definition),network)
        view=conditional_network(network,self.definition,'affine_initial_power')
        torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
        self.assertTrue(network.training);self.assertTrue(all(p.requires_grad for p in parameters))
        self.assertEqual([id(p) for p in view.parameters()],[id(p) for p in parameters])
        n=self.definition['normalization'];d=self.definition['domain']
        alo=(d['initial_temperature_K'][0]-n['state_offset_K'])/n['state_scale_K'];ahi=(d['initial_temperature_K'][1]-n['state_offset_K'])/n['state_scale_K']
        blo=d['power_W'][0]/n['power_scale_W'];bhi=d['power_W'][1]/n['power_scale_W']
        alpha,beta=.23,.61;z=torch.tensor([[.2,alo+alpha*(ahi-alo),blo+beta*(bhi-blo)],[.7,ahi,bhi]],dtype=torch.float64)
        calls=[];handle=network.register_forward_pre_hook(lambda _,args:calls.append(args[0].detach().clone()))
        actual=view(z);handle.remove();self.assertEqual(len(calls),3)
        expected_inputs=[torch.column_stack((z[:,0],torch.full((2,),a,dtype=torch.float64),torch.full((2,),b,dtype=torch.float64))) for a,b in [(alo,blo),(ahi,blo),(alo,bhi)]]
        for call,expected in zip(calls,expected_inputs):torch.testing.assert_close(call,expected,atol=0,rtol=0)
        f00,f10,f01=[network(value) for value in expected_inputs]
        expected=f00+(z[:,1:2]-alo)/(ahi-alo)*(f10-f00)+(z[:,2:3]-blo)/(bhi-blo)*(f01-f00)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        # The fourth corner is reconstructed, not a fourth hidden network call.
        torch.testing.assert_close(actual[1],(f10+f01-f00)[1],atol=2e-16,rtol=2e-16)
        for fraction in [.1,.37,.83]:
            left=z.clone();left[0,1]=alo;right=z.clone();right[0,1]=ahi;middle=z.clone();middle[0,1]=alo+fraction*(ahi-alo)
            torch.testing.assert_close(view(middle)[0],(1-fraction)*view(left)[0]+fraction*view(right)[0],atol=3e-16,rtol=3e-16)
        model=ConditionalThermalPINN(network,self.definition,freeze=False);inputs=self.query()
        features=torch.stack(((inputs[0]-2)/4,(inputs[1]-290)/20,inputs[2]/100),dim=-1)
        torch.testing.assert_close(model.temperature(*inputs),290+20*network(features).squeeze(-1),atol=0,rtol=0)
        with self.assertRaises(ContractError):conditional_network(network,self.definition,'unknown')

    def test_affine_si_input_weight_derivatives_and_initial_loss_are_not_hard_coded(self):
        network,_=self.network(hidden=[5],seed=91);model=ConditionalThermalPINN(network,self.definition,parameterization='affine_initial_power',freeze=False);inputs=self.query()
        temperature=model.temperature(*inputs);gradients=torch.autograd.grad(temperature.sum(),inputs,create_graph=True)
        for column,gradient in enumerate(gradients):
            h=1e-5;plus=[x.detach().numpy().copy() for x in inputs];minus=[x.detach().numpy().copy() for x in inputs];plus[column]+=h;minus[column]-=h
            np.testing.assert_allclose(gradient.detach().numpy(),(model.predict(*plus)-model.predict(*minus))/(2*h),atol=2e-7,rtol=2e-6)
        expected=(20*gradients[0]+2*(temperature-290)-inputs[2])/100
        torch.testing.assert_close(model.residual(*inputs,create_graph=True),expected,atol=0,rtol=0)
        derivative_gradient=torch.autograd.grad(gradients[0].square().mean(),network[0].weight,retain_graph=True)[0]
        self.assertGreater(float(derivative_gradient.abs().max()),1e-8)
        loss=model.residual(*inputs,create_graph=True).square().mean();gradient=torch.autograd.grad(loss,network[0].weight)[0][0,0].item()
        original=network[0].weight[0,0].item();h=1e-5
        with torch.no_grad():network[0].weight[0,0]=original+h
        plus=model.residual(*inputs,create_graph=True).square().mean().item()
        with torch.no_grad():network[0].weight[0,0]=original-h
        minus=model.residual(*inputs,create_graph=True).square().mean().item()
        with torch.no_grad():network[0].weight[0,0]=original
        self.assertAlmostEqual(gradient,(plus-minus)/(2*h),delta=1e-7)
        scalar=lambda value:torch.tensor(value,dtype=torch.float64)
        initial_loss=((model.temperature(scalar(2),scalar(300),scalar(40))-300)/20).square()
        self.assertGreater(initial_loss.item(),1e-6)
        ic_gradient=torch.autograd.grad(initial_loss,list(network.parameters()))
        self.assertTrue(all(torch.isfinite(value).all() for value in ic_gradient));self.assertGreater(max(float(value.abs().max()) for value in ic_gradient),1e-6)
        # No duplicate representation weights enter runtime state/checkpoints.
        self.assertEqual(set(model.state_dict()),{'network.'+key for key in network.state_dict()})

    def test_affine_three_query_budget_rejects_before_any_raw_forward(self):
        network,_=self.network(hidden=[6]);model=ConditionalThermalPINN(network,self.definition,parameterization='affine_initial_power');calls=[]
        handle=network.register_forward_pre_hook(lambda *_:calls.append(1));self.addCleanup(handle.remove)
        count=PINN_MAX_ACTIVATION_VALUES//(model._width_sum*4*3)+1
        self.assertLess(count*model._width_sum*4,PINN_MAX_ACTIVATION_VALUES)
        inputs=[torch.full((count,),value,dtype=torch.float64) for value in [3.,300.,40.]]
        with self.assertRaisesRegex(ContractError,'activation budget'):model.temperature(*inputs)
        view=conditional_network(network,self.definition,'affine_initial_power')
        with self.assertRaisesRegex(ContractError,'activation budget'):view(torch.tensor([.25,.5,.4],dtype=torch.float64).expand(count,3))
        self.assertEqual(calls,[])

    def test_affine_export_mode_is_bound_and_legacy_free_bundle_still_loads(self):
        path,manifest,network,config=self.export('affine',parameterization='affine_initial_power');model=load_pinn(path)
        self.assertEqual(model.parameterization,'affine_initial_power');self.assertEqual(manifest['conditional_parameterization'],'affine_initial_power')
        self.assertEqual(manifest['resource_boundary']['raw_network_evaluations_per_query'],3)
        self.assertEqual(read(path/'training.json')['config']['conditional_parameterization'],'affine_initial_power')
        self.assertNotIn('conditional_parameterization',read(path/'network.json'))
        original=ConditionalThermalPINN(network,self.definition,parameterization='affine_initial_power',freeze=False)
        torch.testing.assert_close(model.temperature(*self.query()),original.temperature(*self.query()),atol=0,rtol=0)
        for wrapped in [original,conditional_network(network,self.definition,'affine_initial_power')]:
            with self.assertRaisesRegex(ContractError,'parameterization'):export_pinn(self.root/uuid.uuid4().hex,wrapped,self.definition,config)
        for mode in ['manifest_free','training_free','config_free','unsupported','resource_one','bad_config_type']:
            target=self.root/mode;shutil.copytree(path,target);m=read(target/'pinn_manifest.json');training=read(target/'training.json')
            if mode=='manifest_free':m['conditional_parameterization']='free_mlp'
            elif mode=='training_free':training['conditional_parameterization']='free_mlp'
            elif mode=='config_free':training['config']['conditional_parameterization']='free_mlp'
            elif mode=='unsupported':training['conditional_parameterization']='analytic_solution'
            elif mode=='resource_one':m['resource_boundary']['raw_network_evaluations_per_query']=1
            else:training['config']=[]
            write(target/'training.json',training);m['files']['training.json']=sha(target/'training.json');write(target/'pinn_manifest.json',m)
            with self.subTest(mode=mode),self.assertRaises(ContractError):load_pinn(target)
        legacy,_,raw,_=self.export('legacy-free');m=read(legacy/'pinn_manifest.json');training=read(legacy/'training.json')
        m.pop('conditional_parameterization');m['resource_boundary'].pop('raw_network_evaluations_per_query');training.pop('conditional_parameterization');training['config'].pop('conditional_parameterization')
        write(legacy/'training.json',training);m['files']['training.json']=sha(legacy/'training.json');write(legacy/'pinn_manifest.json',m)
        loaded=load_pinn(legacy);self.assertEqual(loaded.parameterization,'free_mlp')
        torch.testing.assert_close(loaded.temperature(*self.query()),ConditionalThermalPINN(raw,self.definition).temperature(*self.query()),atol=0,rtol=0)

    def test_affine_repo_external_isolated_process_uses_same_three_evaluations_and_gradients(self):
        path,_,_,_=self.export(parameterization='affine_initial_power');model=load_pinn(path);inputs=self.query();temperature=model.temperature(*inputs)
        gradients=torch.autograd.grad(temperature.sum(),inputs);residual=model.residual(*inputs,create_graph=True)
        state={key:value.tolist() for key,value in model.network.state_dict().items()}
        PORTABLE.mkdir(parents=True,exist_ok=True);external=PORTABLE/uuid.uuid4().hex;external.mkdir()
        def cleanup():
            assert external.resolve().is_relative_to(PORTABLE.resolve()) and not external.resolve().is_relative_to(ROOT.resolve());shutil.rmtree(external,ignore_errors=True)
        self.addCleanup(cleanup);bundle=external/'bundle';shutil.copytree(path,bundle)
        assert path.resolve().is_relative_to(self.root.resolve()) and self.source.resolve().is_relative_to(self.root.resolve())
        shutil.rmtree(path);shutil.rmtree(self.source)
        query={name:value.detach().tolist() for name,value in zip(['time_s','initial_temperature_K','power_W'],inputs)};write(external/'query.json',query)
        code=r'''import importlib.abc,importlib.util,json,sys,torch
from pathlib import Path
class RejectHarness(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'):raise ModuleNotFoundError('Independent affine PINN must not import project code')
sys.meta_path.insert(0,RejectHarness());torch.set_num_threads(1);root=Path(sys.argv[1]);query=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('_independent_affine_pinn',root/'pinn_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.load_pinn(root)
calls=[];handle=model.network.register_forward_pre_hook(lambda _,args:calls.append(args[0].detach().tolist()))
inputs=[torch.tensor(query[k],dtype=torch.float64,requires_grad=True) for k in ['time_s','initial_temperature_K','power_W']];temperature=model.temperature(*inputs);grads=torch.autograd.grad(temperature.sum(),inputs);residual=model.residual(*inputs,create_graph=True)
assert model.parameterization=='affine_initial_power' and len(calls)==6 and not hasattr(model,'step') and not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)
print(json.dumps({'temperature':temperature.tolist(),'gradients':[g.tolist() for g in grads],'residual':residual.tolist(),'calls':calls,'state':{k:v.tolist() for k,v in model.network.state_dict().items()}},allow_nan=False))'''
        env={**os.environ,'PYTHONPATH':'','PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
        process=subprocess.run([sys.executable,'-I','-B','-c',code,str(bundle),str(external/'query.json')],cwd=external,env=env,text=True,encoding='utf-8',capture_output=True,check=True)
        actual=json.loads(process.stdout);self.assertEqual(actual['state'],state)
        np.testing.assert_allclose(actual['temperature'],temperature.detach().numpy(),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(actual['gradients'],[g.numpy() for g in gradients],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(actual['residual'],residual.detach().numpy(),atol=1e-12,rtol=1e-12)
        for offset in [0,3]:
            np.testing.assert_allclose(np.asarray(actual['calls'][offset])[:,1:],np.tile([-.25,0.],(3,1)),atol=0,rtol=0)
            np.testing.assert_allclose(np.asarray(actual['calls'][offset+1])[:,1:],np.tile([1.5,0.],(3,1)),atol=0,rtol=0)
            np.testing.assert_allclose(np.asarray(actual['calls'][offset+2])[:,1:],np.tile([-.25,1.],(3,1)),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
