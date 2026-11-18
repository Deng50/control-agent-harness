"""Actual Python original-Plant gates and synthetic MATLAB transport boundaries.

No test invokes MATLAB/Native, alters F05, or claims a weight intervention is a
PGAC improvement. The baseline is trained by the real S10 trainer; validation
receipts use the real R01/R02 validator and sampled numeric source data.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from tests import test_external_policy_evaluation as fixture
from control_harness.artifacts import Workspace,read_json,atomic_json,file_hash
from control_harness.contracts import ContractError,DataInterface,ModelInterface
from control_harness.runtime import OperationContext,Cancelled
from control_harness import external_policy_evaluation as ep
from control_harness import external_pgac_evaluation as pg
from control_harness.external_policy import load_external_policy
from control_harness.policy_export import export_external_policy_state
from control_harness.modeling import register_model
from control_harness.plants import sample_model
from control_harness.sampling import recommend_sampling
from control_harness.differentiable_validation import validate_differentiable_model

ROOT=Path(__file__).resolve().parents[1]


def fixture_context(ws):
    atomic_json(ws.path/'.case-plans/test-fixture.json',{'kind':'control-case-plan-test-fixture-v1','workspace':str(ws.path),'confirmation_source':'test_fixture','synthetic':True})
    return OperationContext.test_fixture(ws)


class ExternalPgacEvaluationTests(unittest.TestCase):
    cleanup_base=classmethod(fixture.ExternalPolicyEvaluationTests.cleanup_base.__func__)
    setUp=fixture.ExternalPolicyEvaluationTests.setUp
    cleanup=fixture.ExternalPolicyEvaluationTests.cleanup

    @classmethod
    def setUpClass(cls):
        fixture.ExternalPolicyEvaluationTests.setUpClass.__func__(cls)
        spec={'representation':'discrete_polynomial','interface':cls.physical,'A':[[.7]],'B':[[.2]],'E':[[]],'C':[[1.]],'terms':[]}
        cls.model=register_model(cls.ws,spec,{'kind':'reconstructed_equations','description':'Explicit exact scalar test-fixture rewrite'})
        sampling=recommend_sampling(ModelInterface.parse(cls.physical),{'trajectory_count':5,'steps':50,'seed':20261122,'x0':[.1]})
        data=sample_model(cls.ws,cls.plant,{'sampling':sampling,'sampling_run':'full'},fixture_context(cls.ws))
        source,_=cls.ws.resolve(data);split=read_json(source/'split.json')
        with np.load(source/'transitions.npz',allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
        row=int(np.flatnonzero(np.isin(arrays['trajectory_id'],split['validation']))[3]);config={'steps':50,'probes':[{'row':row,'kind':'smooth'}]}
        cls.validation=validate_differentiable_model(cls.ws,cls.plant,cls.model,config,fixture_context(cls.ws),reference_dataset_id=data)
        mat=ep._selection(cls.ws,cls.mat_policy)
        # Transport-only synthetic fixture: the source numbers are actual Python
        # scalar-formula outputs. This is not original MATLAB execution evidence.
        with cls.ws.stage('sampled_dataset',ModelInterface.parse(cls.physical),[mat['plant_id']],{'protocol_fixture':True},compatible_parents=False) as stage:
            np.savez_compressed(stage.root/'transitions.npz',**arrays);atomic_json(stage.root/'split.json',split)
            md=stage.publish({'status':'verified','scope':'synthetic MATLAB transport boundary, no MATLAB execution'},
                source={'kind':'plant_simulator','model':mat['plant_id'],'backend':'matlab_function','measured':False,'protocol_fixture_no_MATLAB_execution':True},activate=False)
        cls.mat_validation=validate_differentiable_model(cls.ws,mat['plant_id'],cls.model,config,reference_dataset_id=md)

    def make_candidate(self,*,mat=False,eligible=True,change=None,validation=None,benchmark=None,domain=None,network_config=None):
        policy_id=self.mat_policy if mat else self.policy;selection=ep._selection(self.workspace,policy_id);baseline=load_external_policy(selection['policy_root'])
        state={k:v.detach().clone() for k,v in baseline.network.state_dict().items()}
        if network_config is not None:
            from control_harness.supervised_export import build_supervised_network
            with torch.random.fork_rng(devices=[]):state=build_supervised_network(network_config).state_dict()
        if change is not None:change(state)
        proposed=ep.prepare_external_imitation_evaluation(self.workspace,policy_id)['config_suggestions']
        config=pg.normalize_original_evaluation(selection,{'scenarios':proposed['scenarios'],'evaluation':{'q':1.,'r':.015,'s':.01,'comparison_atol':1e-12},**({'benchmark':benchmark} if benchmark else {})})
        refs={'imitation_policy':policy_id,'rollout_model':self.model,'model_validation':validation or (self.mat_validation if mat else self.validation),
            'external_expert_dataset':selection['dataset_id'],'expert':selection['expert_id'],'plant':selection['plant_id']}
        with self.workspace.stage('external_pgac_candidate',DataInterface.parse(selection['policy_meta']['interface']),list(dict.fromkeys(refs.values())),{'component_fixture':True},compatible_parents=False) as stage:
            source=selection['policy_root']/'bundle'
            export_external_policy_state(stage.root/'bundle',state,network_config or read_json(source/'network.json'),selection['policy_meta']['interface'],read_json(source/'scaler.json'),
                read_json(source/'action_constraints.json'),read_json(source/'physical_context.json'),provenance={'source_mapping':selection['data']['source_mapping'],'purpose':'Explicit component weight intervention; no PGAC effect claim'})
            torch.save(state,stage.root/'model.pt');atomic_json(stage.root/'policy.json',{'schema_version':1,'representation':'external_pgac_candidate','bundle':'bundle',**refs})
            atomic_json(stage.root/'report.json',{'original_evaluation_config':config,'rollout_domain_checks_passed':eligible,'domain':domain or {'x':[[-3.,3.]],'u':[[-1.,1.]],'d':[]},'component_fixture_no_optimization_claim':True})
            candidate=stage.publish({'status':'verified','scope':'actual finite raw-state component fixture only'},source=refs,activate=False)
        return candidate,config

    def run_evaluation(self,candidate,config,context=None):
        aid=pg.evaluate_external_pgac(self.workspace,candidate,config,context);root,meta=self.workspace.resolve(aid,allow_rejected=True)
        return root,meta,read_json(root/'report.json')

    def test_raw_state_export_is_real_and_portable_without_a_supervised_artifact(self):
        candidate,_=self.make_candidate(change=lambda state:state['0.bias'].add_(.03));root,_=self.workspace.resolve(candidate);model=load_external_policy(root)
        old=load_external_policy(self.workspace.resolve(self.policy)[0]);features=np.asarray([[.1,.3,0.],[.2,.3,.1]])
        self.assertGreater(float(np.max(np.abs(model.predict_raw(features)-old.predict_raw(features)))),1e-6)
        outside=ROOT.parent/'control-agent-harness-pgac-portable-tests'/uuid.uuid4().hex;outside.mkdir(parents=True)
        def cleanup():
            assert outside.resolve().is_relative_to((ROOT.parent/'control-agent-harness-pgac-portable-tests').resolve()) and not outside.resolve().is_relative_to(ROOT)
            shutil.rmtree(outside,ignore_errors=True)
        self.addCleanup(cleanup);shutil.copytree(root/'bundle',outside/'bundle')
        query=outside/'query.json';atomic_json(query,{'features':features.tolist(),'previous_action':[[0.],[.1]]})
        code="""import importlib.abc,importlib.util,json,sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'):raise ModuleNotFoundError('No project imports')
sys.meta_path.insert(0,Block());root=Path(sys.argv[1]);query=json.loads(Path(sys.argv[2]).read_text())
spec=importlib.util.spec_from_file_location('portable',root/'policy_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);policy=module.load_external_policy(root)
print(json.dumps(policy.predict(query['features'],query['previous_action']).tolist()))"""
        child=subprocess.run([sys.executable,'-I','-B','-c',code,str(outside/'bundle'),str(query)],cwd=outside,env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'},check=True,capture_output=True,text=True)
        np.testing.assert_allclose(json.loads(child.stdout),model.predict(features,np.array([[0.],[.1]])),atol=1e-12,rtol=1e-12)
        self.assertEqual(read_json(root/'bundle/policy.json')['weight_source'],'explicit_actual_raw_state_dict')

    def test_real_three_way_original_source_replay_and_selection_freezes_before_test(self):
        candidate,config=self.make_candidate();base=copy.deepcopy(self.workspace.manifest['active']);snapshots=[]
        def emit(event):
            if event['stage']=='pgac_original_validation_selection_frozen':
                staged=next((self.root/'.staging').glob('external_optimized_policy-*'));saved=read_json(staged/'selection.json');snapshots.append(saved)
                self.assertFalse(list(staged.glob('test-*-rollout.npz')));self.assertEqual(saved['decision']['choice'],'imitation')
        root,meta,report=self.run_evaluation(candidate,config,OperationContext(emit=emit))
        self.assertEqual(len(snapshots),1);self.assertEqual(report['selected_version'],'imitation');self.assertTrue(report['final_acceptance']['approved']);self.assertFalse(report['final_acceptance']['candidate_accepted'])
        self.assertEqual(meta['activation'],'kind');self.assertFalse(report['test_used_for_checkpoint_or_hyperparameter_selection']);self.assertEqual(report['selection_sha256'],file_hash(root/'selection.json'))
        self.assertEqual(report['versions'],{'expert':self.expert,'imitation':self.policy,'optimized_candidate':candidate})
        for part in ['validation','test']:
            for name in pg.NAMES:
                with np.load(root/(part+'-'+name+'-rollout.npz'),allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
                np.testing.assert_allclose(arrays['next_x'],.49*arrays['x']+.34*arrays['u'],atol=1e-12,rtol=0)
                rows=[]
                for identifier in np.unique(arrays['trajectory_id']):
                    mask=arrays['trajectory_id']==identifier;u=arrays['u'][mask];previous=arrays['previous_u'][mask]
                    np.testing.assert_allclose(previous[1:],u[:-1],atol=0,rtol=0)
                    values=((arrays['next_y'][mask]-arrays['reference'][mask])/6.)**2+.015*(u/2.)**2+.01*((u-previous)/2.)**2
                    rows.append(float(np.mean(values)))
                self.assertAlmostEqual(report['closed_loop'][part][name]['quadratic']['score'],float(np.mean(rows)),delta=1e-15)
        for kind,value in base.items():self.assertEqual(self.workspace.manifest['active'][kind],value)
        self.assertEqual(file_hash(root/'bundle/model.pt'),read_json(root/'bundle/policy.json')['source_weight_sha256'])

    def test_domain_ineligible_candidate_retains_real_baseline_and_never_activates_candidate(self):
        candidate,config=self.make_candidate(eligible=False);_,_,report=self.run_evaluation(candidate,config)
        self.assertFalse(report['selection']['candidate_eligible']);self.assertEqual(report['selected_version'],'imitation');self.assertEqual(self.workspace.resolve(candidate)[1]['activation'],'none')

    def test_original_plant_coverage_escape_keeps_baseline_despite_physical_feasibility(self):
        candidate,config=self.make_candidate(domain={'x':[[-.001,.001]],'u':[[-1.,1.]],'d':[]});_,_,report=self.run_evaluation(candidate,config)
        result=report['closed_loop']['validation']['candidate']['quadratic'];self.assertTrue(result['checks']['state']);self.assertFalse(result['checks']['qualified_domain'])
        self.assertEqual(report['selected_version'],'imitation');self.assertFalse(report['final_acceptance']['candidate_accepted'])
        self.assertIn('not_implemented_by_standalone_MLP',report['deployment_scope']['online_ood_fallback'])

    def test_actual_original_subtick_exit_and_return_is_caught_before_a_valid_control_endpoint(self):
        from control_harness.material_registration import register_material
        from control_harness.plants import register_plant
        from control_harness.model_adapters import load_python_adapter
        from control_harness.expert_adapter import load_external_expert
        from control_harness.external_expert import _collect_python,ExpertCollectionStopped
        source=self.root/'exit_return.py';source.write_text('import numpy as np\ndef step(x,u,d,dt,p):\n return np.array([1.-x[0]])\n',encoding='utf-8')
        material=register_material(self.workspace,str(source),{},OperationContext());plant_id=register_plant(self.workspace,material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/exit_return.py',
            'entrypoint':'step','interface':self.physical,'parameters':{},'source_semantics':'transition','C':[[1.]]}},OperationContext())
        selection=ep._selection(self.workspace,self.policy);physical,spec,root,_,material_meta=ep._plant_selection(self.workspace,plant_id)
        original=load_python_adapter(root,spec,expected_file_hashes=material_meta['files']);original.reset([.2])
        np.testing.assert_allclose(original.step([0.],[]),[.8],atol=1e-15,rtol=0);np.testing.assert_allclose(original.step([0.],[]),[.2],atol=1e-15,rtol=0)
        domain={'x':[[-.3,.3]],'u':[[-1.,1.]],'d':[]};guarded=pg._CoveragePlant(original,domain);plan,_=ep._evaluation_plan(selection,{})
        plan=pg._subplan(plan,'validation');plan['trajectories'][0]['x0']=[.2]
        expert=load_external_expert(selection['expert_root'],selection['expert'],expected_file_hashes=selection['expert_material']['files'])
        try:
            with self.assertRaises(ExpertCollectionStopped) as stopped:_collect_python(guarded,expert,plan,OperationContext())
            self.assertEqual(len(stopped.exception.arrays['time']),0);guard=guarded.describe()['coverage_guard']
            self.assertEqual(guard['before_checks'],1);self.assertEqual(guard['after_checks'],1);self.assertEqual(guard['passed_subticks'],0);self.assertEqual(guard['violations'],1)
            self.assertEqual(guard['status'],'failed');self.assertIn('actual physical subtick',stopped.exception.report['failure']['message'])
        finally:expert.close();original.close()

    def test_real_short_pgac_candidate_enters_the_same_original_source_gate(self):
        from control_harness.restricted_pgac import train_restricted_pgac
        selection=ep._selection(self.workspace,self.policy);proposed=ep.prepare_external_imitation_evaluation(self.workspace,self.policy)['config_suggestions']
        original=pg.normalize_original_evaluation(selection,{'scenarios':proposed['scenarios'],'evaluation':{'q':1.,'r':.015,'s':.01,'comparison_atol':1e-12}})
        settings={'epochs':1,'learning_rate':.0001,'batch_size':2,'horizon':2,'seed':20261129,'q':1.,'r':.015,'s':.01,
            'domain':{'x':[[-3.,3.]],'u':[[-1.,1.]],'d':[]},'original_evaluation':original}
        candidate=train_restricted_pgac(self.workspace,self.policy,self.model,settings,validation_id=self.validation)
        candidate_root,candidate_meta=self.workspace.resolve(candidate);trained=read_json(candidate_root/'report.json')
        self.assertGreater(trained['optimizer_updates'],0);self.assertEqual(candidate_meta['activation'],'none')
        initial=torch.load(candidate_root/'initial_imitation_state.pt',weights_only=True,map_location='cpu');baseline=load_external_policy(selection['policy_root'])
        for name,value in baseline.network.state_dict().items():torch.testing.assert_close(initial[name],value,atol=0,rtol=0)
        _,_,report=self.run_evaluation(candidate,original);self.assertTrue(report['final_acceptance']['approved'])
        self.assertEqual(report['versions']['optimized_candidate'],candidate);self.assertEqual(report['original_evaluation_config'],original)

    def test_changed_objective_case_duration_or_missing_condition_is_rejected_before_evaluation(self):
        candidate,config=self.make_candidate();before=len(self.workspace.manifest['artifacts'])
        for mode in ['objective','duration','missing']:
            altered=copy.deepcopy(config)
            if mode=='objective':altered['evaluation']['q']=2.
            elif mode=='duration':altered['scenarios'][0]['steps']+=1
            else:altered['scenarios'].pop()
            with self.subTest(mode=mode),self.assertRaises(ContractError):pg.evaluate_external_pgac(self.workspace,candidate,altered)
        self.assertEqual(len(self.workspace.manifest['artifacts']),before)

    def test_arbitrary_verified_material_cannot_replace_a_real_model_validation_receipt(self):
        material=self.workspace.resolve(self.plant)[1]['parents'][0]['id'];candidate,config=self.make_candidate(validation=material)
        with self.assertRaises(ContractError):pg.evaluate_external_pgac(self.workspace,candidate,config)

    def test_a_new_network_architecture_is_not_a_pgac_continuation_of_actual_imitation_weights(self):
        root,_=self.workspace.resolve(self.policy);settings=read_json(root/'bundle/network.json');settings['hidden_dims']=[6]
        candidate,config=self.make_candidate(network_config=settings)
        with self.assertRaisesRegex(ContractError,'network.json'):pg.evaluate_external_pgac(self.workspace,candidate,config)

    def test_fixed_validation_and_test_gates_keep_no_improvement_failed_and_degraded_baselines(self):
        good=lambda score:{'feasible':True,'score':score};bad={'feasible':False,'score':None};objective={'q':1,'r':.015,'s':.01,'comparison_atol':1e-12}
        for candidate,baseline,choice in [(good(.1),good(.2),'candidate'),(good(.2),good(.2),'imitation'),(bad,good(.2),'imitation'),(bad,bad,None),(good(.1),bad,'candidate')]:
            actual=pg.validation_choice({'candidate':candidate,'imitation':baseline},objective);self.assertEqual(actual['choice'],choice)
        fixed=pg.validation_choice({'candidate':good(.1),'imitation':good(.2)},objective)
        rejected=pg.final_acceptance(fixed,{'candidate':good(.3),'imitation':good(.2)},objective)
        self.assertEqual(rejected['selected_version'],'imitation');self.assertFalse(rejected['candidate_accepted'])
        self.assertFalse(pg.final_acceptance(fixed,{'candidate':bad,'imitation':bad},objective)['approved'])
        baseline_fixed=pg.validation_choice({'candidate':good(.2),'imitation':good(.1)},objective)
        self.assertFalse(pg.final_acceptance(baseline_fixed,{'candidate':good(.01),'imitation':bad},objective)['approved'],'Test cannot promote an unselected candidate')
        no_val_baseline=pg.validation_choice({'candidate':good(.1),'imitation':bad},objective)
        self.assertFalse(pg.final_acceptance(no_val_baseline,{'candidate':bad,'imitation':good(.2)},objective)['approved'],'A validation-infeasible baseline cannot become an approved fallback based on test alone')
        for bad_cfg in [{'q':0,'r':0,'s':0},{'q':1,'r':-1,'s':0},{'q':True,'r':0,'s':0},{'q':1,'r':0,'s':0,'comparison_atol':.1}]:
            with self.assertRaises(ContractError):pg.evaluation_config(bad_cfg)

    def test_cancel_after_validation_archives_actual_prefix_and_preserves_active(self):
        candidate,config=self.make_candidate();before=copy.deepcopy(self.workspace.manifest['active'])
        def emit(event):
            if event['stage']=='pgac_original_validation_selection_frozen':raise Cancelled('Explicit component cancellation at the saved gate')
        with self.assertRaises(Cancelled):pg.evaluate_external_pgac(self.workspace,candidate,config,OperationContext(emit=emit))
        failure=next((self.root/'failed').glob('external_optimized_policy-*'));self.assertEqual(read_json(failure/'report.json')['status'],'cancelled')
        self.assertTrue((failure/'selection.json').is_file());self.assertEqual(len(list(failure.glob('validation-*-rollout.npz'))),3);self.assertFalse(list(failure.glob('test-*-rollout.npz')))
        self.assertEqual(self.workspace.manifest['active'],before)

    def test_optional_three_way_python_timing_uses_real_calls_and_one_thread(self):
        candidate,config=self.make_candidate(benchmark={'warmup':1,'repeats':2,'threads':1,'batch_size':1})
        _,_,report=self.run_evaluation(candidate,config);benchmark=report['controller_benchmark']
        self.assertEqual(benchmark['status'],'completed');self.assertEqual(benchmark['hardware']['torch_threads'],1)
        self.assertTrue(all(row['num_threads']==1 for row in benchmark['hardware']['threadpools']))
        for name in pg.NAMES:self.assertEqual(benchmark[name]['samples'],2);self.assertTrue(all(value>=0 for value in benchmark[name]['raw_seconds']))
        self.assertEqual(benchmark['candidate_speedup_median'],benchmark['expert']['median']/benchmark['candidate']['median'])
        self.assertEqual(benchmark['expert_repeated_for_candidate_pair']['samples'],2)

    def native_prepare(self):
        candidate,config=self.make_candidate(mat=True);nonce=str(uuid.uuid4());exchange=self.root/'.case-plans/matlab-exchanges'/nonce;raw='{"synthetic_transport_fixture":"no native or MATLAB call"}'
        context=OperationContext();context._case_receipt_body='Explicit unit protocol fixture, not real Host authorization';context.bind_raw_request(raw)
        env={'CONTROL_MATLAB_PHASE':'prepare','CONTROL_MATLAB_EXCHANGE_DIR':str(exchange),'CONTROL_MATLAB_REQUEST_SHA256':hashlib.sha256(raw.encode()).hexdigest(),'CONTROL_MATLAB_NONCE':nonce}
        with patch.dict(os.environ,env):prepared=pg.evaluate_external_pgac(self.workspace,candidate,config,context)
        return candidate,config,context,env,exchange,prepared

    def test_matlab_prepare_binds_six_original_source_children_without_execution_or_publication(self):
        _,_,_,_,exchange,prepared=self.native_prepare();top=read_json(exchange/'manifest.json');self.assertEqual(prepared['batch_kind'],'external_pgac_evaluation')
        self.assertEqual(set(top['children']),{'validation','test'})
        for phase in top['children']:
            for name in pg.NAMES:
                child=read_json(exchange/phase/name/'manifest.json');self.assertEqual(child['plant_version'],top['plant_version']);self.assertEqual(child['plant_material_version'],top['plant_material_version'])
                self.assertEqual(child['nonce'],top['nonce']);self.assertTrue(all(row['split']==phase for row in child['plan']['trajectories']))
                self.assertEqual(file_hash(exchange/phase/name/'manifest.json'),top['children'][phase][name]['manifest_sha256'])
        self.assertFalse(list((self.root/'artifacts').glob('external_optimized_policy-*')))

    def test_matlab_finalize_rejects_wrong_nonce_and_test_without_a_frozen_selection(self):
        for mode in ['nonce','missing_selection']:
            candidate,config,context,env,exchange,_=self.native_prepare();top=read_json(exchange/'manifest.json')
            result={key:top[key] for key in ['schema_version','nonce','request_sha256','spec_hash','policy_version','candidate_version','dataset_version','expert_version','expert_material_version','plant_version','plant_material_version']}
            result.update(kind='control-matlab-pgac-evaluation-result-v1',plan_hash=top['plan']['plan_hash'])
            if mode=='nonce':result['nonce']=str(uuid.uuid4())
            atomic_json(exchange/'result.json',result);env.update(CONTROL_MATLAB_PHASE='finalize',CONTROL_MATLAB_RESULT_PATH=str(exchange/'result.json'),CONTROL_MATLAB_RESULT_SHA256=file_hash(exchange/'result.json'))
            with self.subTest(mode=mode),patch.dict(os.environ,env),self.assertRaises((ContractError,FileNotFoundError)):pg.evaluate_external_pgac(self.workspace,candidate,config,context)

    def synthetic_native_results(self,candidate,exchange):
        """Protocol fixture with actual Torch actions and independent scalar Plant.

        This is deliberately labeled synthetic; it never executes the .m helper
        and cannot stand in for R03's real native MATLAB acceptance.
        """
        top=read_json(exchange/'manifest.json');selected=pg._selection(self.workspace,candidate);controllers={'imitation':load_external_policy(selected['policy_root']),'candidate':load_external_policy(selected['candidate_root'])}
        results={};hashes={}
        for phase in ['validation','test']:
            results[phase]={};hashes[phase]={}
            for name in pg.NAMES:
                child=read_json(exchange/phase/name/'manifest.json');plan=child['plan'];spec=child['expert']['spec'];rows={key:[] for key in ['x','next_x','plant_y','next_y','u','previous_u','reference','d','observation','trajectory_id','time']};diagnostics=[];summaries=[]
                for trajectory in plan['trajectories']:
                    x=np.asarray(trajectory['x0']);previous=np.asarray(trajectory['initial_action']);r=np.asarray(trajectory['reference']);memory=copy.deepcopy(spec['initial_memory']);key=next(iter(memory))
                    for step in range(trajectory['steps']):
                        action=.7*(r-x)+.05*previous if name=='expert' else controllers[name].predict(np.r_[x,r,previous],previous_action=previous)
                        next_x=.49*x+.34*action;after={key:memory[key]+1}
                        values={'x':x,'next_x':next_x,'plant_y':x,'next_y':next_x,'u':action,'previous_u':previous,'reference':r,'d':np.empty(0),'observation':x,
                            'trajectory_id':trajectory['trajectory_id'],'time':trajectory['time'][step]}
                        for field,value in values.items():rows[field].append(value.copy() if isinstance(value,np.ndarray) else value)
                        diagnostics.append({'trajectory_id':trajectory['trajectory_id'],'step_index':step,'status':'ok','memory_before':memory,'memory_after':after,'expert_action':action.tolist()});x=next_x;previous=action;memory=after
                    summaries.append({'trajectory_id':trajectory['trajectory_id'],'completed_steps':trajectory['steps'],'status':'completed'})
                arrays={field:np.asarray(value,dtype=np.int64 if field=='trajectory_id' else np.float64) for field,value in rows.items()}
                value={**{key:child[key] for key in ['schema_version','nonce','request_sha256','spec_hash','expert_version','expert_material_version','plant_version','plant_material_version']},
                    'kind':'control-matlab-expert-batch-result-v1','plan_hash':plan['plan_hash'],'status':'completed','arrays':{k:v.tolist() for k,v in arrays.items()},
                    'trajectories':summaries,'diagnostics':diagnostics,'failure':[],'stop_detection':'synthetic_protocol_fixture_no_MATLAB_execution'}
                guard=None
                if name=='candidate':
                    ticks=len(arrays['time'])*plan['plant_substeps'];guard={'schema_version':1,'enabled':True,'domain_hash':child['coverage_guard_hash'],
                        'before_checks':ticks,'after_checks':ticks,'passed_subticks':ticks,'violations':0,'status':'passed'}
                    value['coverage_guard']=guard
                path=exchange/phase/name/'result.json';atomic_json(path,value);hashes[phase][name]=file_hash(path)
                results[phase][name]={'status':'completed','arrays':arrays,'diagnostics':diagnostics,'report':{'trajectories':summaries,**({'coverage_guard':guard} if guard else {})}}
            if phase=='validation':
                decision=pg.validation_choice(pg._scores(selected,results[phase],pg._subplan(top['plan'],phase),top['evaluation']),top['evaluation'],candidate_eligible=True)
                snapshot={'phase':'frozen_before_any_test_rollout','evaluation':top['evaluation'],'plan_hash':top['plan']['plan_hash'],'nonce':top['nonce'],'request_sha256':top['request_sha256'],'decision':decision}
                atomic_json(exchange/'selection.json',snapshot)
        selection_sha=file_hash(exchange/'selection.json');result={key:top[key] for key in ['schema_version','nonce','request_sha256','spec_hash','policy_version','candidate_version','dataset_version','expert_version','expert_material_version','plant_version','plant_material_version']}
        result.update(kind='control-matlab-pgac-evaluation-result-v1',plan_hash=top['plan']['plan_hash'],status='completed',children=hashes,selection_sha256=selection_sha,
            test_started_after_selection=True,test_selection_sha256={name:selection_sha for name in pg.NAMES},controller_benchmark={'status':'not_requested'},protocol_fixture_no_MATLAB_execution=True)
        atomic_json(exchange/'result.json',result);return result

    def test_matlab_synthetic_protocol_recomputes_scores_and_rejects_a_forged_choice(self):
        candidate,config,context,env,exchange,_=self.native_prepare();result=self.synthetic_native_results(candidate,exchange)
        frozen=read_json(exchange/'selection.json');original=copy.deepcopy(frozen);frozen['decision']['choice']='candidate';atomic_json(exchange/'selection.json',frozen)
        wrong_sha=file_hash(exchange/'selection.json');result.update(selection_sha256=wrong_sha,test_selection_sha256={name:wrong_sha for name in pg.NAMES});atomic_json(exchange/'result.json',result)
        env.update(CONTROL_MATLAB_PHASE='finalize',CONTROL_MATLAB_RESULT_PATH=str(exchange/'result.json'),CONTROL_MATLAB_RESULT_SHA256=file_hash(exchange/'result.json'))
        with patch.dict(os.environ,env),self.assertRaisesRegex(ContractError,'choice disagrees'):pg.evaluate_external_pgac(self.workspace,candidate,config,context)
        atomic_json(exchange/'selection.json',original);real_sha=file_hash(exchange/'selection.json');result.update(selection_sha256=real_sha,test_selection_sha256={name:real_sha for name in pg.NAMES});atomic_json(exchange/'result.json',result)
        env['CONTROL_MATLAB_RESULT_SHA256']=file_hash(exchange/'result.json')
        with patch.dict(os.environ,env):aid=pg.evaluate_external_pgac(self.workspace,candidate,config,context)
        root,meta=self.workspace.resolve(aid);report=read_json(root/'report.json')
        self.assertEqual(report['selected_version'],'imitation');self.assertEqual(report['selection_sha256'],real_sha);self.assertEqual(meta['activation'],'kind')
        self.assertEqual(len(list(root.glob('*-rollout.npz'))),6)

    def test_matlab_synthetic_protocol_requires_actual_subtick_guard_counts_and_hash(self):
        candidate,config,context,env,exchange,_=self.native_prepare();result=self.synthetic_native_results(candidate,exchange)
        path=exchange/'validation/candidate/result.json';original=read_json(path)
        for mode in ['missing','count','hash','boolean_count']:
            changed=copy.deepcopy(original);top=copy.deepcopy(result)
            if mode=='missing':changed.pop('coverage_guard')
            elif mode=='count':changed['coverage_guard']['passed_subticks']-=1
            elif mode=='hash':changed['coverage_guard']['domain_hash']='0'*64
            else:changed['coverage_guard']['before_checks']=True
            atomic_json(path,changed);top['children']['validation']['candidate']=file_hash(path);atomic_json(exchange/'result.json',top)
            env.update(CONTROL_MATLAB_PHASE='finalize',CONTROL_MATLAB_RESULT_PATH=str(exchange/'result.json'),CONTROL_MATLAB_RESULT_SHA256=file_hash(exchange/'result.json'))
            with self.subTest(mode=mode),patch.dict(os.environ,env),self.assertRaisesRegex(ContractError,'coverage'):pg.evaluate_external_pgac(self.workspace,candidate,config,context)


if __name__=='__main__':unittest.main()
