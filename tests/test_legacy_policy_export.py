"""Actual legacy numerical bodies reloaded without the project in a new process."""
from __future__ import annotations
import copy
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from control_harness.artifacts import atomic_json
from control_harness.contracts import ContractError,ModelInterface
from control_harness.examples import example
from control_harness.legacy_policy_export import export_legacy_policy
from control_harness.policy import NeuralPolicy,network_config

ROOT=Path(__file__).resolve().parents[1]
OUTER=ROOT.parent/'legacy-policy-independent-exports'
OUTER.mkdir(exist_ok=True)
torch.set_num_threads(1)

class LegacyPolicyExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=OUTER);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.assertTrue(self.root.is_relative_to(OUTER.resolve()))
        self.interface=ModelInterface.parse(example('linear')['nominal']['interface'])
        self.point={'x':[.1,.2,-.1,.03],'reference':[.08,-.05],'disturbance':[.01],'previous':[.05,-.03]}
    def policy(self,architecture='mlp'):
        torch.manual_seed(20261122)
        extra={'hidden_dims':[5,4]} if architecture=='mlp' else {'hidden_size':4,'num_layers':1,'preview_len':3,'disturbance_forecast':'declared_constant'}
        cfg=network_config(self.interface,{'architecture':architecture,'dropout':.1,**extra})
        cfg['action_constraints']={'input_rate_bounds':[[-.12,.12],[-.12,.12]]}
        scalers={'input_mean':(np.arange(cfg['input_dim'])*.01).tolist(),'input_scale':[.9]*cfg['input_dim'],
            'output_mean':[.03,-.02],'output_scale':[.4,.5]}
        policy=NeuralPolicy(self.interface,cfg,scalers)
        optimizer=torch.optim.Adam(policy.parameters(),lr=.003)
        x=torch.randn(6,4,dtype=torch.float64)*.1;r=torch.randn(6,2,dtype=torch.float64)*.1;d=torch.zeros(6,1,dtype=torch.float64)
        loss=policy(x,r,d,previous=torch.zeros(6,2,dtype=torch.float64)).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        policy.eval();return policy,cfg,scalers
    def export(self,architecture='mlp'):
        policy,cfg,scalers=self.policy(architecture);bundle=self.root/architecture
        export_legacy_policy(bundle,self.interface,cfg,scalers,policy.state_dict(),provenance={'kind':'one_actual_optimizer_step_numerical_fixture_not_MPC_performance_claim'})
        return policy,bundle,cfg,scalers
    def load(self,bundle):
        name='legacy_bundle_'+bundle.name
        spec=importlib.util.spec_from_file_location(name,bundle/'policy_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.load_policy(bundle)
    def test_all_three_trained_architectures_reload_in_new_repo_free_process(self):
        code="""import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('Project imports are forbidden')
sys.meta_path.insert(0,Block());sys.path.insert(0,sys.argv[1])
from policy_runtime import load_policy
p=load_policy();v=json.loads(sys.argv[2]);u,_=p.act(v['x'],v['reference'],v['disturbance'],v['previous']);assert not any(k.startswith('control_harness') for k in sys.modules);print(json.dumps(u.tolist()))"""
        for architecture in ('mlp','rnn','lstm'):
            with self.subTest(architecture=architecture):
                policy,bundle,_,_=self.export(architecture)
                expected=policy.act(self.point['x'],self.point['reference'],self.point['disturbance'],self.point['previous'])[0]
                env={**os.environ,'PYTHONPATH':'','PYTHONUTF8':'1'}
                result=subprocess.run([sys.executable,'-I','-B','-c',code,str(bundle),json.dumps(self.point)],cwd=self.root,env=env,text=True,capture_output=True,check=True)
                np.testing.assert_allclose(json.loads(result.stdout),expected,atol=1e-12,rtol=1e-12)
                self.assertNotIn(str(ROOT),str(bundle))
    def test_reset_actual_previous_and_rate_projection_keep_real_memory(self):
        policy,cfg,scalers=self.policy();policy.network.net[-1].bias.data.fill_(20.)
        bundle=self.root/'rate';export_legacy_policy(bundle,self.interface,cfg,scalers,policy.state_dict())
        exported=self.load(bundle)
        with self.assertRaisesRegex(ValueError,'previous'):exported.act(self.point['x'],self.point['reference'],self.point['disturbance'])
        with self.assertRaisesRegex(ValueError,'previous'):exported.reset()
        exported.reset(self.point['previous'])
        first,_=exported.act(self.point['x'],self.point['reference'],self.point['disturbance']);second,_=exported.act(self.point['x'],self.point['reference'],self.point['disturbance'])
        np.testing.assert_allclose(first,np.asarray(self.point['previous'])+.12,atol=1e-12)
        np.testing.assert_allclose(second,first+.12,atol=1e-12)
        exported.reset([1.5,1.5]);action,_=exported.act(self.point['x'],self.point['reference'],self.point['disturbance']);self.assertTrue(np.all(action<=1.5))
    def test_hash_tamper_and_nonfinite_raw_before_projection_are_rejected(self):
        _,bundle,_,_=self.export();path=bundle/'policy.pt';path.write_bytes(path.read_bytes()+b'tampered')
        with self.assertRaisesRegex(ValueError,'hash'):self.load(bundle)
        policy,cfg,scalers=self.policy();scalers['output_scale']=[1e308,1e308]
        policy=NeuralPolicy(self.interface,cfg,scalers);policy.network.net[-1].bias.data.fill_(1e308)
        raw=self.root/'overflow';export_legacy_policy(raw,self.interface,cfg,scalers,policy.state_dict());loaded=self.load(raw)
        with self.assertRaisesRegex(FloatingPointError,'before action projection'):loaded.act(self.point['x'],self.point['reference'],self.point['disturbance'],self.point['previous'])
    def test_large_configuration_is_rejected_before_real_parameter_allocation(self):
        _,bundle,_,_=self.export();path=bundle/'policy.json';document=json.loads(path.read_text(encoding='utf-8'));document['network']['hidden_dims']=[65536,65536];atomic_json(path,document)
        manifest=json.loads((bundle/'policy_manifest.json').read_text(encoding='utf-8'));manifest['files']['policy.json']=hashlib.sha256(path.read_bytes()).hexdigest();atomic_json(bundle/'policy_manifest.json',manifest)
        original=torch.empty
        def checked(*args,**kwargs):
            if torch.get_default_device().type!='meta' and args and isinstance(args[0],tuple) and np.prod(args[0])>5_000_000:raise AssertionError('Real oversized allocation attempted')
            return original(*args,**kwargs)
        with patch('torch.empty',side_effect=checked),self.assertRaisesRegex(ValueError,'tensor budget'):self.load(bundle)
    def test_export_preserves_rng_refuses_overwrite_and_invalid_weights(self):
        policy,cfg,scalers=self.policy();before=torch.get_rng_state().clone();bundle=self.root/'safe'
        export_legacy_policy(bundle,self.interface,cfg,scalers,policy.state_dict());self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with self.assertRaisesRegex(ContractError,'overwrite'):export_legacy_policy(bundle,self.interface,cfg,scalers,policy.state_dict())
        bad=copy.deepcopy(policy.state_dict());next(iter(bad.values())).fill_(float('nan'))
        with self.assertRaisesRegex(ContractError,'finite'):export_legacy_policy(self.root/'bad',self.interface,cfg,scalers,bad)

    def test_valid_parameter_budget_cannot_allocate_an_oversized_recurrent_activation(self):
        cfg=network_config(self.interface,{'architecture':'rnn','hidden_size':1000,'num_layers':1,'preview_len':49993,'disturbance_forecast':'declared_constant'})
        scalers={'input_mean':[0.]*cfg['input_dim'],'input_scale':[1.]*cfg['input_dim'],'output_mean':[0.]*2,'output_scale':[1.]*2}
        source=NeuralPolicy(self.interface,cfg,scalers);self.assertLess(sum(v.numel() for v in source.state_dict().values()),5_000_000)
        bundle=self.root/'long-preview';export_legacy_policy(bundle,self.interface,cfg,scalers,source.state_dict());policy=self.load(bundle)
        with patch.object(policy.module,'act',side_effect=AssertionError('Large recurrent inference executed')),self.assertRaisesRegex(ValueError,'activation budget'):
            policy.act(self.point['x'],self.point['reference'],self.point['disturbance'],self.point['previous'])
        with patch.object(policy.module,'forward',side_effect=AssertionError('Large recurrent forward executed')),self.assertRaisesRegex(ValueError,'activation budget'):
            policy.forward(torch.zeros(1,4),torch.zeros(1,2),torch.zeros(1,1),torch.zeros(1,2))

    def test_differentiable_forward_rejects_hidden_nonfinite_inputs_and_preserves_valid_gradient(self):
        for architecture in ('mlp','rnn','lstm'):
            with self.subTest(architecture=architecture):
                source,bundle,_,_=self.export(architecture);policy=self.load(bundle)
                x=torch.tensor([self.point['x']],dtype=torch.float64,requires_grad=True);r=torch.tensor([self.point['reference']],dtype=torch.float64);d=torch.tensor([self.point['disturbance']],dtype=torch.float64);p=torch.tensor([self.point['previous']],dtype=torch.float64)
                expected=source(x,r,d,previous=p);actual=policy.forward(x,r,d,p);torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                expected_grad=torch.autograd.grad(expected.sum(),x,retain_graph=True)[0];actual_grad=torch.autograd.grad(actual.sum(),x)[0];torch.testing.assert_close(actual_grad,expected_grad,atol=1e-12,rtol=1e-12)
                for index in range(4):
                    inputs=[x.detach().clone(),r.clone(),d.clone(),p.clone()];inputs[index][0,0]=float('inf')
                    with self.assertRaisesRegex(ValueError,'finite'):policy.forward(*inputs)
                with self.assertRaisesRegex(ValueError,'batch'):policy.forward(x,r.repeat(2,1),d,p)

if __name__=='__main__':unittest.main()
