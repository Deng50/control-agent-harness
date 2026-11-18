"""Pure MATLAB expert exchange/static-boundary tests; no MATLAB is executed.

Native result documents here are synthetic protocol fixtures. They do not
substitute for L02's actual native MATLAB expert/plant integration.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np

from control_harness.contracts import ContractError,canonical_hash
from control_harness.expert_adapter import validate_expert_spec
from control_harness.matlab_adapter import validate_matlab_spec
from control_harness.matlab_expert import (prepare_matlab_expert_manifest,load_matlab_expert_batch,validate_matlab_expert_source)
from control_harness.matlab_source import validate_matlab_sources

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'test-output/matlab-expert-tests'
EXPERT="""function [action,next_memory,status] = external_expert(obs,r,d,previous,memory,parameters)
action=parameters.gain*(r-obs(1:numel(r)));
next_memory=struct('count',memory.count+1);
status='ok';
end
"""
PLANT="""function next = plant_step(x,u,d,dt,parameters)
next=parameters.A*x+parameters.B*u+parameters.E*d;
end
"""


def write_json(path,value):
    path.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n',encoding='utf-8')


class MatlabExpertTests(unittest.TestCase):
    def setUp(self):
        OUTPUT.mkdir(parents=True,exist_ok=True);self.root=OUTPUT/('e-'+uuid.uuid4().hex[:8]);self.root.mkdir();self.addCleanup(self.cleanup)
        variable=lambda name,bound:{'name':name,'unit':'1','lower':-bound,'upper':bound}
        self.interface={'schema_version':1,'name':'explicit synthetic boundary fixture','time_semantics':'discrete_state_transition','sample_time':.2,
            'states':[variable('x1',2),variable('x2',2)],'controls':[variable('u1',1)],'outputs':[variable('y1',2)],'disturbances':[variable('d1',.2)]}
        self.expert={'schema_version':1,'backend':'matlab_function','source_file':'source/external_expert.m','entrypoint':'external_expert','interface':copy.deepcopy(self.interface),
            'parameters':{'gain':.2},'time_unit':'s','control_period':.4,'observations':[{'name':'measured_'+v['name'],'source':'state','index':k,
                'plant_name':v['name'],'unit':v['unit'],'availability':'measured','delay_steps':0} for k,v in enumerate(self.interface['states'])],
            'reference_available':True,'disturbance_available':True,'previous_action_available':True,'initial_memory':{'count':0},
            'action_mapping':[{'action_index':0,'plant_control_index':0,'plant_control_name':'u1','unit':'1'}],'failure_policy':'stop','input_rate_bounds':[[-.5,.5]]}
        self.plant={'schema_version':1,'backend':'matlab_function','source_file':'source/plant_step.m','entrypoint':'plant_step','interface':copy.deepcopy(self.interface),
            'parameters':{'A':[[.9,.05],[0,.85]],'B':[[.1],[.03]],'E':[[.01],[.01]]},'source_semantics':'transition','C':[[1,0]]}
        self.expert_root,self.expert_material=self.material('expert',{'source/external_expert.m':EXPERT})
        self.plant_root,self.plant_material=self.material('plant',{'source/plant_step.m':PLANT})
        self.versions={'expert_version':{'id':'external-expert-declared','metadata_hash':'1'*64},'expert_material_version':self.expert_material,
            'plant_version':{'id':'plant-declared','metadata_hash':'2'*64},'plant_material_version':self.plant_material}
        self.plan={'schema_version':1,'interface':copy.deepcopy(self.interface),'control_period':.4,'plant_substeps':2,'seed':20261122,
            'trajectories':[{'trajectory_id':k,'steps':2,'x0':[.02*k,.1],'reference':[.1],'disturbance':[.01],'initial_action':[0.],
                'time':[0.,.4],'split':part} for k,part in enumerate(['train','validation','test'])],
            'split':{'train':[0],'validation':[1],'test':[2]}}
        self.rehash_plan();self.nonce=str(uuid.uuid4());self.request_sha='3'*64;self.exchange=self.root/'exchange'

    def cleanup(self):
        assert self.root.resolve().is_relative_to(OUTPUT.resolve());shutil.rmtree(self.root,ignore_errors=True)

    def material(self,name,files):
        root=self.root/name;root.mkdir();hashes={}
        for relative,text in files.items():
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8');hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        metadata={'schema_version':1,'id':'material-'+name,'kind':'material','files':hashes};write_json(root/'artifact.json',metadata)
        return root,{'id':metadata['id'],'metadata_hash':canonical_hash(metadata)}

    def rehash_plan(self):self.plan['plan_hash']=canonical_hash({k:v for k,v in self.plan.items() if k!='plan_hash'})

    def prepare(self):
        result=prepare_matlab_expert_manifest(self.expert,self.expert_root,self.plant,self.plant_root,self.plan,exchange_dir=self.exchange,
            request_sha256=self.request_sha,nonce=self.nonce,**self.versions)
        self.manifest=json.loads(Path(result['manifest_path']).read_text(encoding='utf-8'));return result

    def result(self):
        self.prepare();expert=self.manifest['expert']['spec'];plant=self.manifest['plant']['spec'];arrays={key:[] for key in
            ['observation','reference','d','previous_u','u','x','next_x','plant_y','next_y','trajectory_id','time']};summaries=[];diagnostics=[]
        A=np.asarray(plant['parameters']['A']);B=np.asarray(plant['parameters']['B']);E=np.asarray(plant['parameters']['E']);C=np.asarray(plant['C'])
        for trajectory in self.plan['trajectories']:
            x=np.asarray(trajectory['x0']);previous=np.asarray(trajectory['initial_action']);memory={'count':0};reference=np.asarray(trajectory['reference']);d=np.asarray(trajectory['disturbance'])
            for k in range(trajectory['steps']):
                y=C@x;obs=np.array([x[row['index']] if row['source']=='state' else y[row['index']] for row in expert['observations']]);action=.2*(reference-obs[:len(reference)])
                u=np.zeros(len(self.interface['controls']))
                for mapping in expert['action_mapping']:u[mapping['plant_control_index']]=action[mapping['action_index']]
                next_x=x.copy()
                for _ in range(self.plan['plant_substeps']):next_x=A@next_x+B@u+E@d
                next_y=C@next_x
                row={'observation':obs,'reference':reference,'d':d,'previous_u':previous,'u':u,'x':x,'next_x':next_x,'plant_y':y,'next_y':next_y,
                    'trajectory_id':trajectory['trajectory_id'],'time':trajectory['time'][k]}
                for key,value in row.items():arrays[key].append(value.tolist() if isinstance(value,np.ndarray) else value)
                after={'count':memory['count']+1};diagnostics.append({'trajectory_id':trajectory['trajectory_id'],'step_index':k,'status':'ok','memory_before':memory,
                    'memory_after':after,'expert_action':action.item() if action.size==1 else action.tolist()});x=next_x;previous=u;memory=after
            summaries.append({'trajectory_id':trajectory['trajectory_id'],'completed_steps':trajectory['steps'],'status':'completed'})
        bound={key:self.manifest[key] for key in ['schema_version','nonce','request_sha256','spec_hash',*self.versions]}
        return {**bound,'kind':'control-matlab-expert-batch-result-v1','plan_hash':self.plan['plan_hash'],'status':'completed','arrays':arrays,
            'trajectories':summaries,'diagnostics':diagnostics,'failure':[],'stop_detection':'synthetic_protocol_fixture_no_MATLAB_execution'}

    def load(self,result):
        path=self.exchange/'result.json';write_json(path,result)
        return load_matlab_expert_batch(self.expert,self.plant,self.plan,result_path=path,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            request_sha256=self.request_sha,nonce=self.nonce,**self.versions)

    def test_expert_signature_is_explicit_and_default_plant_contract_remains_unchanged(self):
        meta=json.loads((self.expert_root/'artifact.json').read_text())
        report=validate_matlab_expert_source(self.expert_root,self.expert,meta['files'])
        self.assertEqual(report['entrypoint_signature'],[6,3]);self.assertFalse(report['source_executed'])
        with self.assertRaises(ContractError):validate_matlab_sources(self.expert_root,self.expert,meta['files'])

    def test_quadprog_and_optimoptions_are_allowed_only_by_trusted_expert_caller(self):
        # Static source-support fixture only; no options are guessed or solver run.
        text="""function [action,next_memory,status] = external_expert(obs,r,d,previous,memory,parameters)
options=optimoptions('quadprog');
[action,value,flag]=quadprog(parameters.H,parameters.f,[],[],[],[],parameters.lower,parameters.upper,previous,options);
assert(flag>0 && isfinite(value));
next_memory=memory;status='ok';
end
"""
        root,version=self.material('qp',{'source/external_expert.m':text});files=json.loads((root/'artifact.json').read_text())['files']
        self.assertEqual(validate_matlab_expert_source(root,self.expert,files)['extra_numeric_builtins'],['optimoptions','quadprog','spdiags','speye'])
        with self.assertRaises(ContractError):validate_matlab_sources(root,{**self.expert,'signature':[6,3],'extra_numeric_builtins':['quadprog']},files)
        self.assertTrue(version)

    def test_numeric_extension_cannot_grant_io_dynamic_calls_or_arbitrary_signatures(self):
        files=json.loads((self.expert_root/'artifact.json').read_text())['files']
        for kwargs in ({'signature':(True,3)},{'signature':(1,1)},{'extra_numeric_builtins':['eval']},{'extra_numeric_builtins':['fopen']},{'extra_numeric_builtins':'quadprog'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ContractError):validate_matlab_sources(self.expert_root,self.expert,files,**kwargs)

    def test_hidden_memory_and_dynamic_or_io_expert_helpers_are_rejected(self):
        for index,body in enumerate(['persistent hidden; action=obs(1);','global hidden; action=obs(1);',"action=eval('obs(1)');", "fopen('file'); action=obs(1);",'action=rand(1);']):
            text='function [action,next_memory,status]=external_expert(obs,r,d,previous,memory,parameters)\n'+body+"\nnext_memory=memory;status='ok';\nend\n"
            root,_=self.material('bad'+str(index),{'source/external_expert.m':text});files=json.loads((root/'artifact.json').read_text())['files']
            with self.subTest(body=body),self.assertRaises(ContractError):validate_matlab_expert_source(root,self.expert,files)

    def test_prepare_binds_two_materials_versions_hashes_and_no_matlab_import(self):
        before={name for name in sys.modules if name.startswith('matlab')};prepared=self.prepare()
        self.assertEqual(prepared['batch_kind'],'external_expert');self.assertEqual(prepared['plan_hash'],self.plan['plan_hash'])
        self.assertEqual(self.manifest['spec_hash'],canonical_hash({'expert':validate_expert_spec(self.expert),'plant':validate_matlab_spec(self.plant)}))
        for role in ['expert','plant']:
            source=self.manifest[role]
            for relative,sha in source['source_inventory'].items():self.assertEqual(hashlib.sha256((Path(source['source_root'])/relative).read_bytes()).hexdigest(),sha)
        self.assertEqual(before,{name for name in sys.modules if name.startswith('matlab')})

    def test_complete_result_recovers_arrays_and_explicit_memory_per_trajectory(self):
        loaded=self.load(self.result());self.assertEqual(loaded['status'],'completed');self.assertEqual(loaded['arrays']['x'].shape,(6,2));self.assertEqual(loaded['arrays']['u'].shape,(6,1))
        self.assertEqual([loaded['diagnostics'][k]['memory_before'] for k in (0,2,4)],[{'count':0}]*3)
        self.assertEqual(loaded['report']['training'],'not_run');self.assertFalse(loaded['report']['hardware_execution'])

    def test_zero_disturbance_json_empty_array_restores_n_by_zero(self):
        self.interface['disturbances']=[];self.expert['interface']=copy.deepcopy(self.interface);self.plant['interface']=copy.deepcopy(self.interface);self.plant['parameters']['E']=[[],[]]
        self.plan['interface']=copy.deepcopy(self.interface)
        for row in self.plan['trajectories']:row['disturbance']=[]
        self.rehash_plan();result=self.result();result['arrays']['d']=[]
        self.assertEqual(self.load(result)['arrays']['d'].shape,(6,0))

    def test_four_state_two_action_permutation_preserves_plant_previous_and_rate_order(self):
        frozen=json.loads((ROOT/'test-output/v2-development/20261117T002006/fixtures/F02_existing_linear_4x2.json').read_text(encoding='utf-8'))
        model=frozen['reference'];self.interface=copy.deepcopy(model['interface']);self.expert['interface']=copy.deepcopy(self.interface)
        self.expert['observations']=[{'name':'observed_'+item['name'],'source':'state','index':k,'plant_name':item['name'],
            'unit':item['unit'],'availability':'measured','delay_steps':0} for k,item in enumerate(self.interface['states'])]
        self.expert['action_mapping']=[{'action_index':0,'plant_control_index':1,'plant_control_name':'u2','unit':'1'},
            {'action_index':1,'plant_control_index':0,'plant_control_name':'u1','unit':'1'}]
        self.expert['input_rate_bounds']=[[-.5,.5],[-.4,.4]]
        self.plant.update(interface=copy.deepcopy(self.interface),parameters={key:model[key] for key in ['A','B','E']},C=model['C'])
        self.plan['interface']=copy.deepcopy(self.interface)
        self.plan['plant_substeps']=4
        for k,row in enumerate(self.plan['trajectories']):row.update(x0=[.02*k,.1,.03,-.02],reference=[.1,-.1],initial_action=[.02,-.03])
        self.rehash_plan();result=self.result();loaded=self.load(result)
        self.assertEqual(loaded['arrays']['x'].shape,(6,4));self.assertEqual(loaded['arrays']['u'].shape,(6,2))
        np.testing.assert_array_equal(loaded['arrays']['previous_u'][0],[.02,-.03])
        np.testing.assert_array_equal(loaded['arrays']['u'][0],np.asarray(loaded['diagnostics'][0]['expert_action'])[[1,0]])
        bad=copy.deepcopy(result);bad['arrays']['previous_u'][0].reverse()
        with self.assertRaises(ContractError):self.load(bad)

    def test_nonce_request_spec_source_version_and_plan_binding_cannot_be_reused(self):
        original=self.result()
        for key,value in [('nonce',str(uuid.uuid4())),('request_sha256','4'*64),('spec_hash','5'*64),('plan_hash','6'*64),('expert_version',{'id':'other','metadata_hash':'7'*64})]:
            changed=copy.deepcopy(original);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):self.load(changed)

    def test_same_path_changed_plan_cannot_reuse_prepared_result(self):
        result=self.result();self.plan['trajectories'][0]['x0'][0]+=.01;self.rehash_plan()
        with self.assertRaises(ContractError):self.load(result)

    def test_changed_private_snapshot_or_escaped_entrypoint_is_rejected(self):
        result=self.result();source=Path(self.manifest['expert']['source_file']);original=source.read_bytes();source.write_bytes(original+b'\n% changed\n')
        with self.assertRaises(ContractError):self.load(result)
        source.write_bytes(original);self.manifest['expert']['source_file']=str(self.root/'outside.m');write_json(self.exchange/'manifest.json',self.manifest)
        with self.assertRaises(ContractError):self.load(result)

    def test_wrong_numeric_shapes_types_nonfinite_and_row_order_are_rejected(self):
        original=self.result()
        for key,value in [('x',[[0,0]]*5),('x',[[[0,0]]]*6),('u',[[True]]*6),('trajectory_id',[.0]*6),('time',[[[0]]]*6),('trajectory_id',[0,1,0,1,2,2])]:
            result=copy.deepcopy(original);result['arrays'][key]=value
            with self.subTest(key=key,value=str(value)[:30]),self.assertRaises(ContractError):self.load(result)
        result=copy.deepcopy(original);result['arrays']['u'][0]=[None]
        with self.assertRaises(ContractError):self.load(result)

    def test_x0_history_memory_and_observation_mapping_cannot_be_forged(self):
        original=self.result()
        mutations=[lambda r:r['arrays']['x'][0].__setitem__(0,.4),lambda r:r['arrays']['previous_u'][1].__setitem__(0,.4),
            lambda r:r['arrays']['observation'][0].__setitem__(0,.4),lambda r:r['diagnostics'][2].__setitem__('memory_before',{'count':99}),
            lambda r:r['diagnostics'][0].__setitem__('expert_action',.4),lambda r:r['diagnostics'][0].__setitem__('status','failed')]
        for mutate in mutations:
            result=copy.deepcopy(original);mutate(result)
            with self.subTest(mutate=mutate),self.assertRaises(ContractError):self.load(result)

    def test_complete_rows_must_obey_bounds_rate_and_output_mapping_even_if_status_ok(self):
        original=self.result()
        for key,value in [('u',[1.1]),('u',[.6]),('next_x',[2.1,0.]),('plant_y',[.5]),('d',[.3])]:
            result=copy.deepcopy(original);result['arrays'][key][0]=value
            with self.subTest(key=key,value=value),self.assertRaises(ContractError):self.load(result)

    def test_partial_prefix_is_preserved_and_never_replayed_as_complete(self):
        original=self.result();result=copy.deepcopy(original)
        result['arrays']={key:value[:1] for key,value in result['arrays'].items()};result['diagnostics']=result['diagnostics'][:1]
        result.update(status='partial',trajectories=[{'trajectory_id':0,'completed_steps':1,'status':'partial'}],failure={'trajectory_id':0,'step_index':1,'reason':'expert status failed'})
        loaded=self.load(result);self.assertEqual(loaded['status'],'partial');self.assertEqual(len(loaded['arrays']['u']),1)
        for change in [{'status':'completed'},{'status':'failed'},{'failure':{'trajectory_id':0,'step_index':0}},{'trajectories':result['trajectories']+[{'trajectory_id':1,'completed_steps':0,'status':'failed'}]}]:
            with self.subTest(change=change),self.assertRaises(ContractError):self.load({**result,**change})

    def test_failed_zero_prefix_and_all_completed_failed_status_are_distinct(self):
        original=self.result();failed=copy.deepcopy(original);failed['arrays']={key:[] for key in original['arrays']};failed['diagnostics']=[]
        failed.update(status='failed',trajectories=[{'trajectory_id':0,'completed_steps':0,'status':'failed'}],failure={'trajectory_id':0,'step_index':0})
        self.assertEqual(self.load(failed)['status'],'failed')
        with self.assertRaises(ContractError):self.load({**original,'status':'failed'})
        with self.assertRaises(ContractError):self.load({**failed,'status':'completed'})

    def test_plan_time_period_initial_bounds_and_expansion_reject_before_source_copy(self):
        for mutate in [lambda p:p.__setitem__('control_period',.3),lambda p:p['trajectories'][0].__setitem__('time',[0,.2]),
                lambda p:p['trajectories'][0].__setitem__('initial_action',[1.1]),lambda p:p['trajectories'][0].__setitem__('steps',10**12)]:
            old=copy.deepcopy(self.plan);mutate(self.plan);self.rehash_plan()
            with self.subTest(mutate=mutate),self.assertRaises(ContractError):self.prepare()
            self.assertFalse(self.exchange.exists());self.plan=old

    def test_material_byte_change_and_unregistered_dependencies_are_rejected(self):
        path=self.expert_root/'source/external_expert.m';path.write_text(EXPERT+'\n% new bytes\n',encoding='utf-8')
        with self.assertRaises(ContractError):self.prepare()

    def test_native_helper_keeps_local_batch_and_explicit_slx_state_reset_contract(self):
        source=(ROOT/'matlab/control_harness_expert_batch.m').read_text(encoding='utf-8')
        self.assertIn('Simulink.SimulationInput',source);self.assertIn('in.setExternalInput(ds)',source)
        self.assertIn('out.xout.numElements == nx',source);self.assertIn('InitialState',source)
        self.assertIn('memory=expert.initial_memory',source);self.assertIn("strcmp(status,'ok')",source)
        self.assertNotIn('start_matlab',source);self.assertNotIn('save_system(',source);self.assertNotIn('load_system(',source)


if __name__=='__main__':unittest.main()
