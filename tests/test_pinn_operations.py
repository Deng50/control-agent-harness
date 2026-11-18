"""Actual S11 dispatch, confirmation, trajectory handoff and dead-worker recovery."""
from __future__ import annotations
import copy,json,os
from pathlib import Path
import shutil,subprocess,sys,tempfile,unittest
import torch
from control_harness.artifacts import Workspace,atomic_json,read_json
from control_harness.case_plans import FIXTURE_KIND,PendingPlanConfirmation,propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.contracts import ContractError
from control_harness.design_only import synthesize_mpc
from control_harness.inspection import prepare_handoff
from control_harness.material_registration import register_material
from control_harness.modeling import load_model
from control_harness.operations import run_operation
from control_harness.pinn import train_pinn
from control_harness.pinn_problem import prepare_pinn_problem
from control_harness.runtime import OperationContext
from test_pinn_problem import configuration,FROZEN

PROJECT=Path(__file__).resolve().parents[1];OUTPUT=PROJECT/'test-output/pinn-operation-tests';OUTPUT.mkdir(parents=True,exist_ok=True)
HARD_EXIT='''import json,os,sys
from pathlib import Path
from control_harness.artifacts import Workspace,atomic_json
from control_harness.pinn import train_pinn
from control_harness.runtime import OperationContext
root=Path(sys.argv[1]);request=json.loads((root/'hard-exit-request.json').read_text())
def emit(event):
 if event.get('stage')=='pinn_training' and event.get('epoch')==1:
  atomic_json(root/'hard-exit-ready.json',{'stage_id':Path(event['checkpoint']).parent.name,'pid':os.getpid()});os._exit(73)
train_pinn(Workspace(root),request['problem'],request['config'],OperationContext(emit=emit))
raise RuntimeError('Expected an actual committed PINN epoch')
'''

class PinnOperationsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.base_temp=tempfile.TemporaryDirectory(dir=OUTPUT);cls.base=Path(cls.base_temp.name);cls.threads=torch.get_num_threads();torch.set_num_threads(1)
  inputs=cls.base/'inputs';inputs.mkdir()
  for name in ['F04_conditional_ode.json','F04_analytic_trajectories.csv','F04_training_physics_collocation.csv']:shutil.copyfile(FROZEN/name,inputs/name)
  cls.base_ws=Workspace(cls.base/'workspace');cls.material=register_material(cls.base_ws,str(inputs/'F04_conditional_ode.json'),{'dependency_root':str(inputs),'dependencies':['F04_analytic_trajectories.csv','F04_training_physics_collocation.csv']},OperationContext())
  cls.problem=prepare_pinn_problem(cls.base_ws,cls.material,configuration())
  cls.config={'network':{'hidden_dims':[5],'activation':'tanh','epochs':3,'patience':30,'batch_size':256,'learning_rate':.003,'seed':20261121},'loss_weights':{'data':1.,'physics':1.,'initial':10.}}
 @classmethod
 def tearDownClass(cls):torch.set_num_threads(cls.threads);cls.base_temp.cleanup()
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=OUTPUT);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);shutil.copytree(self.base_ws.path,self.root/'case');self.ws=Workspace(self.root/'case')
  atomic_json(self.ws.path/'.case-plans/test-fixture.json',{'kind':FIXTURE_KIND,'workspace':str(self.ws.path),'confirmation_source':'test_fixture'});self.context=OperationContext.test_fixture(self.ws)
 def request(self,operation,inputs,config=None):return {'operation':operation,'workspace':str(self.ws.path),'inputs':inputs,'config':copy.deepcopy(config or {})}
 def approve(self,request):
  refs={}
  for row in confirmation_requirements(self.ws,request):
   plan=propose_plan(self.ws,row['scope'],row['expected_payload'],row['bindings']);refs[row['scope']]={k:plan[k] for k in ['plan_id','version_hash']}
  request['config']['plan_refs']=refs
 def train(self):
  request=self.request('train_pinn',{'problem':self.problem},self.config);self.approve(request);return run_operation(request,self.context)
 def test_prepare_full_physics_declaration_requires_review_and_changes_invalidate_it(self):
  request=self.request('prepare_pinn_problem',{'material':self.material},configuration());before=set(self.ws.manifest['artifacts'])
  with self.assertRaises(PendingPlanConfirmation):run_operation(request,OperationContext())
  self.assertEqual(set(self.ws.manifest['artifacts']),before);self.approve(request)
  changed=copy.deepcopy(request);changed['config']['normalization']['time_scale_s']*=2
  with self.assertRaises(PendingPlanConfirmation):run_operation(changed,self.context)
  self.assertEqual(set(self.ws.manifest['artifacts']),before)
  result=run_operation(request,self.context);self.assertEqual(result['kind'],'conditional_ode_problem')
  report=read_json(Path(result['artifact_path'])/'report.json');self.assertEqual(report['training'],'not_run');self.assertFalse(report['source_executed']);self.assertEqual(report['confirmation_sources'][0]['confirmation_source'],'test_fixture')
 def test_actual_training_needs_both_scopes_and_preserves_complete_stop(self):
  request=self.request('train_pinn',{'problem':self.problem},self.config);requirements=confirmation_requirements(self.ws,request);self.assertEqual([x['scope'] for x in requirements],['signals','training'])
  self.assertEqual(requirements[0]['expected_payload'],self.ws.resolve(self.problem)[1]['source']['pinn_signals']);before=set(self.ws.manifest['artifacts'])
  with self.assertRaises(PendingPlanConfirmation):run_operation(request,OperationContext())
  self.assertEqual(set(self.ws.manifest['artifacts']),before);self.approve(request);result=run_operation(request,self.context)
  report=read_json(Path(result['artifact_path'])/'report.json');self.assertTrue(report['is_pinn']);self.assertEqual(report['acceptance'],'not_requested');self.assertEqual(report['trained_epochs'],3)
  self.assertEqual([x['scope'] for x in report['confirmation_sources']],['signals','training'])
  handoff=prepare_handoff(self.ws,result['artifact_id']);self.assertIsNone(handoff['handoff']['operation']);self.assertFalse(handoff['handoff']['capabilities']['automatic_mpc_pgac_handoff'])
 def test_changed_problem_or_loss_cannot_reuse_unchanged_training_confirmation(self):
  request=self.request('train_pinn',{'problem':self.problem},self.config);self.approve(request);before=set(self.ws.manifest['artifacts'])
  for key,value in [('loss_weights',{'data':1.,'physics':2.,'initial':10.}),('network',{**self.config['network'],'learning_rate':.004}),('conditional_parameterization','affine_initial_power')]:
   changed=copy.deepcopy(request);changed['config'][key]=value
   with self.assertRaises(PendingPlanConfirmation):run_operation(changed,self.context)
  changed=configuration();changed['normalization']['time_scale_s']=80.;other=prepare_pinn_problem(self.ws,self.material,changed)
  request['inputs']['problem']=other
  with self.assertRaises(PendingPlanConfirmation):run_operation(request,self.context)
  self.assertEqual(len(self.ws.manifest['artifacts']),len(before)+1)
 def test_published_conditional_network_is_rejected_as_a_direct_mpc_plant(self):
  result=self.train();before=copy.deepcopy(self.ws.manifest)
  with self.assertRaisesRegex(ContractError,'conditional.*state-transition'):load_model(self.ws,result['artifact_id'])
  with self.assertRaisesRegex(ContractError,'conditional.*state-transition'):synthesize_mpc(self.ws,{'model':result['artifact_id']},{'solver_check':True},OperationContext())
  self.assertEqual(self.ws.manifest,before);self.assertFalse(any(x['kind']=='controller_design' for x in self.ws.manifest['artifacts'].values()))
 def test_ablation_is_labeled_and_does_not_replace_the_active_physics_network(self):
  result=self.train();active=copy.deepcopy(self.ws.manifest['active']);request=self.request('train_pinn',{'problem':self.problem},{**self.config,'ablation':True,'loss_weights':{'data':1.,'physics':0.,'initial':10.}});self.approve(request)
  control=run_operation(request,self.context);report=read_json(Path(control['artifact_path'])/'report.json')
  self.assertFalse(report['is_pinn']);self.assertEqual(report['algorithm'],'conditional_ode_physics_off_ablation');self.assertEqual(self.ws.manifest['active'],active)
  self.assertFalse(prepare_handoff(self.ws,control['artifact_id'])['handoff']['is_pinn'])
 def test_actual_dead_pinn_worker_checkpoint_recovers_as_its_real_algorithm(self):
  atomic_json(self.ws.path/'hard-exit-request.json',{'problem':self.problem,'config':self.config})
  env={**os.environ,'PYTHONPATH':str(PROJECT/'src'),'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'}
  child=subprocess.run([sys.executable,'-B','-c',HARD_EXIT,str(self.ws.path)],cwd=PROJECT,env=env,capture_output=True,text=True,timeout=60)
  self.assertEqual(child.returncode,73,child.stderr);ready=read_json(self.ws.path/'hard-exit-ready.json')
  recovered=run_operation(self.request('recover_interrupted_stage',{'stage_id':ready['stage_id']}),OperationContext())
  self.assertEqual(recovered['algorithm'],'conditional_ode_pinn');self.assertEqual(recovered['checkpoint_kind'],'pinn_model');self.assertEqual(recovered['epoch'],1)
  continuous=train_pinn(self.ws,self.problem,copy.deepcopy(self.config));resume=self.request('train_pinn',{'problem':self.problem},{**self.config,'resume_checkpoint':recovered['checkpoint'],'resume_checkpoint_sha256':recovered['checkpoint_sha256']});self.approve(resume)
  resumed=run_operation(resume,self.context);a=torch.load(self.ws.resolve(continuous)[0]/'model.pt',weights_only=True);b=torch.load(Path(resumed['artifact_path'])/'model.pt',weights_only=True)
  for key in a['network_state']:self.assertTrue(torch.equal(a['network_state'][key],b['network_state'][key]))

if __name__=='__main__':unittest.main()
