"""Real Python comparisons and explicitly synthetic native-transport tests.

The policy is trained by the actual external-imitation trainer. No expert,
Plant/model loader, NN forward or timing function is mocked.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace,read_json,file_hash,atomic_json
from control_harness.contracts import ContractError,DataInterface
from control_harness.material_registration import register_material
from control_harness.plants import register_plant
from control_harness.external_expert import register_expert,collect_external_expert
from control_harness.external_expert import build_expert_collection_plan,expert_data_interface
from control_harness.external_expert_data import publish_external_expert_dataset
from control_harness.external_imitation import train_external_imitation
from control_harness.external_policy_evaluation import evaluate_external_imitation,benchmark_config,prepare_external_imitation_evaluation
from control_harness.runtime import OperationContext,Cancelled

ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/'test-output/external-policy-evaluation-tests'


class ExternalPolicyEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT.mkdir(parents=True,exist_ok=True);cls.base=OUTPUT/('source-'+uuid.uuid4().hex[:8]);cls.base.mkdir();cls.addClassCleanup(cls.cleanup_base)
        original_threads=torch.get_num_threads();torch.set_num_threads(1);cls.addClassCleanup(torch.set_num_threads,original_threads)
        cls.ws=Workspace(cls.base/'workspace');var=lambda name,bound:{'name':name,'unit':'1','lower':-bound,'upper':bound}
        cls.physical={'schema_version':1,'name':'independent explicit scalar source','sample_time':.1,'time_semantics':'discrete_state_transition',
            'states':[var('state',3)],'controls':[var('command',1)],'outputs':[var('output',3)],'disturbances':[]}
        plant_file=cls.base/'plant.py';plant_file.write_text('import numpy as np\ndef step(x,u,d,dt,p):\n return np.array([p["a"]*x[0]+p["b"]*u[0]])\n',encoding='utf-8')
        material=register_material(cls.ws,str(plant_file),{},OperationContext())
        cls.plant=register_plant(cls.ws,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/plant.py','entrypoint':'step',
            'interface':cls.physical,'parameters':{'a':.7,'b':.2},'source_semantics':'transition','C':[[1.]]}},OperationContext())
        expert_file=cls.base/'expert.py';expert_file.write_text('import numpy as np\ndef control(obs,r,d,previous,memory,p):\n return (np.array([p["gain"]*(r[0]-obs[0])+.05*previous[0]]), {"count":memory["count"]+1}, "ok")\n',encoding='utf-8')
        material=register_material(cls.ws,str(expert_file),{},OperationContext())
        cls.expert_spec={'schema_version':1,'backend':'python_callable','source_file':'source/expert.py','entrypoint':'control','interface':cls.physical,
            'parameters':{'gain':.7},'time_unit':'s','control_period':.2,'observations':[{'name':'state','source':'state','index':0,'plant_name':'state','unit':'1','availability':'measured','delay_steps':0}],
            'reference_available':True,'disturbance_available':True,'previous_action_available':True,'initial_memory':{'count':0},
            'action_mapping':[{'action_index':0,'plant_control_index':0,'plant_control_name':'command','unit':'1'}],'failure_policy':'stop','input_rate_bounds':[[-.4,.4]]}
        cls.expert=register_expert(cls.ws,material,{'adapter':cls.expert_spec})
        cls.collection={'seed':20261122,'scaler':'standard','scenarios':[{'name':'case-'+str(k),'initial_state':[.05+.04*k],'reference':[.3],'disturbance':[],
            'steps':6,'initial_previous_action':[0.],'split':'train' if k<3 else 'validation' if k==3 else 'test'} for k in range(6)]}
        cls.dataset=collect_external_expert(cls.ws,cls.expert,cls.plant,copy.deepcopy(cls.collection));cls.network={'hidden_dims':[5],'activation':'tanh','epochs':3,
            'learning_rate':.005,'batch_size':8,'patience':20,'seed':20261120}
        cls.policy=train_external_imitation(cls.ws,cls.dataset,{'network':cls.network})
        source,_=cls.ws.resolve(cls.dataset);data=read_json(source/'dataset.json')
        with np.load(source/'expert.npz',allow_pickle=False) as archive:arrays={key:archive[key].copy() for key in archive.files}
        constraints=read_json(source/'action_constraints.json')
        data_only_workspace=Workspace(cls.base/'data-only-workspace')
        cls.data_only=publish_external_expert_dataset(data_only_workspace,arrays,DataInterface.parse(data['interface']),{'train':[0,1,2],'validation':[3],'test':[4,5]},
            {'scaler':'standard','action_constraints':constraints},source={'kind':'recorded_external_expert','description':'Explicit causal action records without an executable original expert or Plant',
                'previous_action_semantics':'actual_applied_action_before_current_decision'},source_mapping=data['source_mapping'])
        other_policy=train_external_imitation(data_only_workspace,cls.data_only,{'network':cls.network})
        other_root,_=data_only_workspace.resolve(other_policy)
        cls.data_only_policy=cls.ws.import_artifact(other_root)
        # Native-transport protocol fixture only: these arrays came from the
        # equivalent real Python source above. No MATLAB execution is claimed.
        matws=Workspace(cls.base/'mat-protocol-workspace');matplant=cls.base/'matplant.m';matexpert=cls.base/'matexpert.m'
        matplant.write_text('function next=matplant(x,u,d,dt,P)\nnext=P.a*x+P.b*u;\nend\n',encoding='utf-8')
        matexpert.write_text("function [action,next_memory,status]=matexpert(obs,r,d,previous,memory,P)\naction=P.gain*(r-obs)+.05*previous;next_memory=struct('count',memory.count+1);status='ok';\nend\n",encoding='utf-8')
        material=register_material(matws,str(matplant),{},OperationContext());mp=register_plant(matws,material,{'adapter':{'schema_version':1,'backend':'matlab_function',
            'source_file':'source/matplant.m','entrypoint':'matplant','interface':cls.physical,'parameters':{'a':.7,'b':.2},'source_semantics':'transition','C':[[1.]]}},OperationContext())
        matspec={**copy.deepcopy(cls.expert_spec),'backend':'matlab_function','source_file':'source/matexpert.m','entrypoint':'matexpert'}
        material=register_material(matws,str(matexpert),{},OperationContext());me=register_expert(matws,material,{'adapter':matspec})
        matinterface,mapping=expert_data_interface(matspec,cls.collection);plan=build_expert_collection_plan(cls.physical,matspec,cls.collection)
        config={'scaler':'standard','action_constraints':constraints,'physical_mapping':{'plant_dt':.1,'control_period':.2,'plant_substeps':2,
            'observations':matspec['observations'],'action_mapping':matspec['action_mapping']}}
        with matws.stage('external_expert_dataset',matinterface,[me,mp],config,compatible_parents=False) as stage:
            atomic_json(stage.root/'collection-plan.json',plan)
            md=publish_external_expert_dataset(matws,copy.deepcopy(arrays),matinterface,plan['split'],config,source={'kind':'declared_simulation',
                'description':'Static native transport protocol fixture using equivalent Python-reference arrays; no MATLAB execution',
                'expert':me,'plant':mp,'protocol_fixture_no_MATLAB_execution':True,'previous_action_semantics':'actual_applied_action_before_current_decision'},
                physical_interface=cls.physical,source_mapping=mapping,parents=[me,mp],report={'status':'completed','execution_origin':'Python reference, not MATLAB'},stage=stage)
        matpolicy=train_external_imitation(matws,md,{'network':cls.network});matroot,_=matws.resolve(matpolicy);cls.mat_policy=cls.ws.import_artifact(matroot)

    @classmethod
    def cleanup_base(cls):
        assert cls.base.resolve().is_relative_to(OUTPUT.resolve());shutil.rmtree(cls.base,ignore_errors=True)

    def setUp(self):
        self.root=OUTPUT/('eval-'+uuid.uuid4().hex[:8]);shutil.copytree(self.base/'workspace',self.root);self.addCleanup(self.cleanup)
        self.workspace=Workspace(self.root)

    def cleanup(self):
        assert self.root.resolve().is_relative_to(OUTPUT.resolve());shutil.rmtree(self.root,ignore_errors=True)

    def evaluate(self,policy=None,config=None):
        artifact=evaluate_external_imitation(self.workspace,policy or self.policy,config or {'benchmark':{'warmup':1,'repeats':4,'batch_size':1,'threads':1}})
        root,metadata=self.workspace.resolve(artifact,kind='policy_evaluation',allow_rejected=True)
        return root,metadata,read_json(root/'report.json')

    def test_real_original_source_rollouts_use_own_states_and_previous_actions(self):
        before=copy.deepcopy(self.workspace.manifest['active']);policy_root,_=self.workspace.resolve(self.policy);weights=file_hash(policy_root/'bundle/model.pt')
        root,meta,report=self.evaluate();self.assertEqual(report['status'],'completed');self.assertEqual(meta['activation'],'none');self.assertFalse(report['stability_proof'])
        self.assertEqual(report['expert'],self.expert);self.assertEqual(report['plant'],self.plant);self.assertEqual(report['condition_source'],'bound_dataset_validation_and_test')
        self.assertEqual([row['trajectory_id'] for row in report['plan']['trajectories']],[3,4,5]);self.assertTrue(report['independent_of_training'])
        with np.load(root/'expert-rollout.npz') as a,np.load(root/'imitation-rollout.npz') as b:
            self.assertGreater(float(np.max(np.abs(a['u']-b['u']))),1e-5)
            self.assertGreater(float(np.max(np.abs(a['x']-b['x']))),1e-6)
            for trajectory in [3,4,5]:
                mask=b['trajectory_id']==trajectory;x=b['x'][mask];u=b['u'][mask];next_x=b['next_x'][mask];previous=b['previous_u'][mask]
                np.testing.assert_allclose(next_x,.49*x+.34*u,atol=1e-12,rtol=0)  # Two true original-Plant subticks.
                np.testing.assert_array_equal(previous[1:],u[:-1]);np.testing.assert_array_equal(x[1:],next_x[:-1])
        self.assertEqual(self.workspace.manifest['active'],before);self.assertEqual(file_hash(policy_root/'bundle/model.pt'),weights)
        self.assertEqual(report['policy_replacement'],'not_performed');self.assertIn('raw_tracking_rmse_increase_on_conditions',report['degradation'])

    def test_controller_only_timing_is_single_thread_and_separate_from_plant(self):
        _,_,report=self.evaluate();benchmark=report['controller_benchmark'];self.assertEqual(benchmark['status'],'completed')
        self.assertEqual(benchmark['settings'],{'warmup':1,'repeats':4,'batch_size':1,'threads':1})
        self.assertEqual(benchmark['hardware']['torch_threads'],1);self.assertTrue(all(v['num_threads']==1 for v in benchmark['hardware']['threadpools']))
        for name in ['expert','imitation']:
            stats=benchmark[name];self.assertEqual(stats['samples'],4);self.assertEqual(len(stats['raw_seconds']),4);self.assertGreater(stats['median'],0)
            np.testing.assert_allclose(stats['median'],np.median(stats['raw_seconds']),atol=0,rtol=0)
            self.assertGreater(report['closed_loop'][name]['plant_timing']['separate_plant_timing_seconds'],0)
        self.assertTrue(benchmark['not_a_deployment_latency_guarantee']);self.assertIn('S09 repeated/fresh',benchmark['scope'])

    def test_data_only_policy_is_explicitly_not_applicable_without_plant_execution(self):
        _,_,report=self.evaluate(self.data_only_policy);self.assertEqual(report['status'],'not_applicable')
        self.assertEqual(report['closed_loop']['plant_execution'],'not_run');self.assertEqual(report['controller_benchmark']['status'],'not_applicable')
        self.assertNotIn('plant',report);self.assertIn('recorded actions are not executable',report['reason'])

    def test_train_prefix_cannot_be_renamed_or_resized_into_a_heldout_case(self):
        row=copy.deepcopy(self.collection['scenarios'][0]);row.update(name='renamed-test',split='test',steps=17)
        with self.assertRaisesRegex(ContractError,'training physical prefix'):self.evaluate(config={'scenarios':[row],'benchmark':{'warmup':1,'repeats':2}})

    def test_custom_independent_conditions_are_labeled_and_source_active_is_not_substituted(self):
        # A new unrelated Plant may coexist, but the evaluation must keep its exact ancestry.
        material=register_material(self.workspace,str(self.base/'plant.py'),{},OperationContext())
        other=register_plant(self.workspace,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/plant.py','entrypoint':'step',
            'interface':self.physical,'parameters':{'a':.1,'b':.8},'source_semantics':'transition','C':[[1.]]}},OperationContext())
        row={'name':'new-independent-test','initial_state':[-.23],'reference':[.17],'disturbance':[],'steps':4,'initial_previous_action':[.03]}
        _,_,report=self.evaluate(config={'scenarios':[row],'benchmark':{'warmup':1,'repeats':2}})
        self.assertNotEqual(other,self.plant);self.assertEqual(report['plant'],self.plant);self.assertEqual(report['condition_source'],'explicit_custom_independent_conditions')

    def test_invalid_timing_or_metric_settings_fail_before_source_execution(self):
        for settings in [{'threads':2},{'batch_size':2},{'repeats':True},{'warmup':0},{'repeats':10**12}]:
            with self.subTest(settings=settings),self.assertRaises(ContractError):benchmark_config(settings)
        with self.assertRaises(ContractError):self.evaluate(config={'metrics':{'settling_rel_band':-1}})

    def test_cancelled_evaluation_retains_failed_stage_numeric_prefix_without_activation(self):
        before=copy.deepcopy(self.workspace.manifest['active'])
        def emit(event):
            if event.get('stage')=='external_expert_collection':raise Cancelled('Independent cancellation after first complete held-out trajectory')
        with self.assertRaises(Cancelled):evaluate_external_imitation(self.workspace,self.policy,{'benchmark':{'warmup':1,'repeats':2}},OperationContext(emit=emit))
        failures=list((self.root/'failed').glob('policy_evaluation-*'));self.assertTrue(failures)
        report=read_json(failures[-1]/'report.json');self.assertEqual(report['status'],'cancelled');self.assertTrue((failures[-1]/'expert-partial-rollout.npz').is_file())
        self.assertEqual(self.workspace.manifest['active'],before)

    def test_finite_but_degraded_real_nn_branch_is_reported_and_does_not_replace_baseline(self):
        original_root,meta=self.workspace.resolve(self.policy);original_hash=file_hash(original_root/'bundle/model.pt');before=copy.deepcopy(self.workspace.manifest['active'])
        with self.workspace.stage('external_imitation_policy',DataInterface.parse(meta['interface']),[self.policy,self.dataset],
                {'test_fixture':'explicit real NN weight intervention, not a trained accuracy/improvement claim'},compatible_parents=False) as stage:
            for name in meta['files']:
                if name in {'config.json','stage.intent.json'}:continue
                target=stage.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original_root/name,target)
            state=torch.load(stage.root/'bundle/model.pt',weights_only=True)
            for value in state.values():value.zero_()
            biases=[key for key in state if key.endswith('.bias')];state[biases[-1]].fill_(-1e6);torch.save(state,stage.root/'bundle/model.pt')
            manifest=read_json(stage.root/'bundle/policy_manifest.json');manifest['files']['model.pt']=file_hash(stage.root/'bundle/model.pt')
            manifest['matlab']={'status':'unsupported','reason':'Explicit Python-only test_fixture weight intervention; no stale MATLAB weights claimed.'};atomic_json(stage.root/'bundle/policy_manifest.json',manifest)
            atomic_json(stage.root/'report.json',{'test_fixture':'real NN weights changed before publication solely to test degradation reporting','trained_improvement_claim':False})
            bad=stage.publish({'status':'verified','scope':'test_fixture_real_network_intervention'},source={'kind':'test_fixture_weight_intervention','external_expert_dataset':self.dataset},activate=False)
        root,_,report=self.evaluate(bad);self.assertEqual(report['status'],'completed');self.assertTrue(report['candidate_feasible'])
        self.assertTrue(report['degradation']['raw_tracking_rmse_increase_on_conditions']);self.assertFalse(report['effect_improvement_required'])
        self.assertEqual(self.workspace.manifest['active'],before);self.assertEqual(file_hash(original_root/'bundle/model.pt'),original_hash)
        with np.load(root/'imitation-rollout.npz') as arrays:self.assertLessEqual(float(arrays['u'].max()),-.4+1e-12)

    def test_native_scalar_timing_and_non_numeric_timing_boundary(self):
        from control_harness.external_policy_evaluation import _stats
        self.assertEqual(_stats(.0001)['samples'],1)
        for bad in [[True,.1],[-.1],[None],[[.1]],['.01']]:
            with self.subTest(value=bad),self.assertRaises(ContractError):_stats(bad)

    def test_read_only_proposal_resolves_explicit_cases_and_never_loads_sources(self):
        before=copy.deepcopy(self.workspace.manifest);proposal=prepare_external_imitation_evaluation(self.workspace,self.policy)
        self.assertEqual(proposal['status'],'proposal');self.assertFalse(proposal['source_executed']);self.assertEqual(proposal['resolved_source_trajectory_ids'],[3,4,5])
        self.assertEqual(proposal['config_suggestions']['benchmark'],{'warmup':5,'repeats':50,'batch_size':1,'threads':1})
        self.assertEqual([r['initial_state'] for r in proposal['config_suggestions']['scenarios']],
            [self.collection['scenarios'][k]['initial_state'] for k in [3,4,5]])
        self.assertEqual(self.workspace.manifest,before)
        data_only=prepare_external_imitation_evaluation(self.workspace,self.data_only_policy)
        self.assertEqual(data_only['status'],'not_applicable');self.assertIsNone(data_only['config_suggestions'])

    def test_matlab_route_requires_native_phase_and_never_starts_engine(self):
        with self.assertRaisesRegex(ContractError,'native Host batch'):self.evaluate(self.mat_policy)

    def _native_prepare(self):
        nonce=str(uuid.uuid4());exchange=self.root/'.case-plans/matlab-exchanges'/nonce;raw='{"test":"private_native_transport_protocol_only"}'
        context=OperationContext();context._case_receipt_body='explicit trusted unittest protocol fixture, not a real Host receipt';context.bind_raw_request(raw)
        env={'CONTROL_MATLAB_PHASE':'prepare','CONTROL_MATLAB_EXCHANGE_DIR':str(exchange),'CONTROL_MATLAB_REQUEST_SHA256':hashlib.sha256(raw.encode()).hexdigest(),'CONTROL_MATLAB_NONCE':nonce}
        settings={'benchmark':{'warmup':1,'repeats':2}}
        with patch.dict(os.environ,env):result=evaluate_external_imitation(self.workspace,self.mat_policy,settings,context)
        return result,exchange,env,context,settings

    def test_matlab_pure_prepare_binds_two_existing_s09_manifests_to_one_original_plant(self):
        result,exchange,_,_,_=self._native_prepare();self.assertEqual(result['batch_kind'],'external_policy_evaluation');top=read_json(exchange/'manifest.json')
        expert=read_json(exchange/'expert/manifest.json');imitation=read_json(exchange/'imitation/manifest.json')
        self.assertEqual(expert['plant_version'],imitation['plant_version']);self.assertEqual(expert['plant_material_version'],imitation['plant_material_version'])
        self.assertEqual(expert['plan']['trajectories'],imitation['plan']['trajectories']);self.assertNotEqual(expert['expert_version'],imitation['expert_version'])
        self.assertEqual(imitation['expert_version'],top['policy_version']);self.assertEqual(imitation['expert_material_version'],top['policy_version'])
        self.assertEqual(top['children']['expert']['manifest_sha256'],file_hash(exchange/'expert/manifest.json'))
        self.assertEqual(top['children']['imitation']['manifest_sha256'],file_hash(exchange/'imitation/manifest.json'))
        self.assertFalse(list((self.root/'artifacts').glob('policy_evaluation-*')),'Prepare cannot publish a numerical evaluation')

    def test_matlab_finalize_rejects_changed_root_nonce_without_claiming_a_native_run(self):
        _,exchange,env,context,settings=self._native_prepare();top=read_json(exchange/'manifest.json')
        bad={key:top[key] for key in ['schema_version','nonce','request_sha256','spec_hash','policy_version','dataset_version','expert_version','expert_material_version','plant_version','plant_material_version']}
        bad.update(kind='control-matlab-policy-evaluation-result-v1',plan_hash=top['plan']['plan_hash'],nonce=str(uuid.uuid4()))
        atomic_json(exchange/'result.json',bad);env.update(CONTROL_MATLAB_PHASE='finalize',CONTROL_MATLAB_RESULT_PATH=str(exchange/'result.json'),CONTROL_MATLAB_RESULT_SHA256=file_hash(exchange/'result.json'))
        with patch.dict(os.environ,env),self.assertRaisesRegex(ContractError,'no longer match'):evaluate_external_imitation(self.workspace,self.mat_policy,settings,context)

    def test_matlab_finalize_rejects_missing_terminal_status_and_changed_child_manifest(self):
        # These are authenticated transport fixtures only, never MATLAB evidence.
        for mode in ['missing_status','changed_child']:
            with self.subTest(mode=mode):
                _,exchange,env,context,settings=self._native_prepare();top=read_json(exchange/'manifest.json')
                result={key:top[key] for key in ['schema_version','nonce','request_sha256','spec_hash','policy_version','dataset_version','expert_version','expert_material_version','plant_version','plant_material_version']}
                result.update(kind='control-matlab-policy-evaluation-result-v1',plan_hash=top['plan']['plan_hash'])
                if mode=='changed_child':
                    result['status']='completed';child=read_json(exchange/'expert/manifest.json');child['protocol_fixture_tampered']=True;atomic_json(exchange/'expert/manifest.json',child)
                atomic_json(exchange/'result.json',result);env.update(CONTROL_MATLAB_PHASE='finalize',CONTROL_MATLAB_RESULT_PATH=str(exchange/'result.json'),CONTROL_MATLAB_RESULT_SHA256=file_hash(exchange/'result.json'))
                with patch.dict(os.environ,env),self.assertRaisesRegex(ContractError,'terminal status|child MATLAB evaluation manifest changed'):
                    evaluate_external_imitation(self.workspace,self.mat_policy,settings,context)

    def test_cancel_after_rollouts_before_report_keeps_both_real_numeric_prefixes(self):
        from control_harness.external_policy_evaluation import _finish,_selection
        root,_,report=self.evaluate();results={}
        for name in ['expert','imitation']:
            with np.load(root/(name+'-rollout.npz'),allow_pickle=False) as data:arrays={key:data[key].copy() for key in data.files}
            results[name]={'arrays':arrays,'report':report['closed_loop'][name]['report'],'diagnostics':read_json(root/(name+'-diagnostics.json'))}
        with self.assertRaises(Cancelled):
            _finish(self.workspace,_selection(self.workspace,self.policy),{},OperationContext(cancelled=lambda:True),
                {'policy':self.policy},report['plan'],results,report['controller_benchmark'])
        failed=list((self.root/'failed').glob('policy_evaluation-*'));self.assertTrue(failed)
        failure=read_json(failed[-1]/'report.json');self.assertEqual(failure['status'],'cancelled')
        for name in results:
            with np.load(failed[-1]/(name+'-partial-rollout.npz'),allow_pickle=False) as data:np.testing.assert_array_equal(data['u'],results[name]['arrays']['u'])


if __name__=='__main__':unittest.main()
