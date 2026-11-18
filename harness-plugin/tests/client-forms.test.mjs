import assert from 'node:assert/strict';
import test from 'node:test';
import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';
import {agents} from '../src/roles.mjs';
import {applySummary,inputLabel,networkConfig,networkPath,operationDefaults,operationGroups,operations,parseNumericInput,parseObject,reportReadOperation,requestSignature,setPath,signature,uploadAccept,workspaceKey} from '../src/client/form-schema.mjs';

test('all V2 routes are exposed through their actual native role, with no unknown default request keys',()=>{
  const required={model:['register_material','prepare_supervised_dataset','train_supervised','register_plant','recommend_sampling','sample_model','train_predictive_model','prepare_pinn_problem','train_pinn'],
    controller:['synthesize_mpc','tune_controller'],lightweight:['register_expert','collect_external_expert','import_expert_data','train_external_imitation','prepare_imitation_evaluation','evaluate_external_imitation','validate_differentiable_model','optimize_external_policy','evaluate_external_pgac']};
  for(const [role,names] of Object.entries(required)){const listed=operationGroups(role).flatMap(g=>g.names);assert.equal(new Set(listed).size,listed.length);
    for(const name of listed){assert.ok(agents[role].operations.includes(name));assert.ok(operations[name]);const defaults=operationDefaults(name);
      for(const key of Object.keys(defaults.inputs))assert.ok(Object.hasOwn(operations[name].inputsSchema.properties,key),name+' unknown input '+key);
      for(const key of Object.keys(defaults.config))assert.ok(Object.hasOwn(operations[name].configSchema.properties,key),name+' unknown config '+key);}
    for(const name of names)assert.ok(listed.includes(name),role+' missing '+name);
  }
});
test('upload and input labels cover actual supported source formats without fabricating a model',()=>{
  for(const ext of ['.py','.m','.slx','.csv','.xlsx','.json','.npz'])assert.ok(uploadAccept.split(',').includes(ext));
  assert.match(inputLabel('material'),/材料/);assert.match(inputLabel('validation'),/验证/);
  const plant=operationDefaults('register_plant');assert.equal(plant.config.adapter.interface,undefined);assert.equal(plant.config.adapter.C,undefined);assert.equal(plant.config.adapter.source_file,'');
  const design=operationDefaults('synthesize_mpc');assert.equal(design.config.parameters,undefined);assert.equal(design.inputs.model,undefined);
  const data=operationDefaults('import_expert_data');assert.deepEqual(data.config.action_constraints.bounds,[]);assert.equal(data.config.physical_interface,undefined);assert.equal(data.config.source.kind,undefined);
});
test('ordinary NN edits compile into exactly the advanced JSON settings, preserving unrelated and nested values',()=>{
  for(const operation of ['train_supervised','train_predictive_model','train_external_imitation','train_pinn']){
    const draft=operationDefaults(operation),prefix=networkPath(operation),path=name=>prefix?prefix+'.'+name:name;
    const edited=setPath(setPath(draft.config,path('hidden_dims'),[17,9]),path('learning_rate'),.0037),advanced=parseObject(JSON.stringify(edited));
    assert.deepEqual(advanced,edited);assert.equal(networkConfig(advanced,operation).learning_rate,.0037);assert.deepEqual(networkConfig(advanced,operation).hidden_dims,[17,9]);
    assert.deepEqual(operationDefaults(operation),draft,'Editing one role must not mutate shared defaults');
  }
});
test('numerical/dependency edits invalidate review fingerprints; mere mode/format/compatible continuation does not',()=>{
  const request={agent:'model',operation:'train_predictive_model',workspace:'D:/p',inputs:{dataset:'real-id'},config:operationDefaults('train_predictive_model').config};
  const initial=requestSignature(request),changed={...request,config:setPath(request.config,'network.learning_rate',.0123)};assert.notEqual(requestSignature(changed),initial);
  assert.notEqual(requestSignature({...request,inputs:{dataset:'new-real-version'}}),initial);
  assert.equal(requestSignature({...request,config:{...request.config,plan_refs:{training:{plan_id:'p',version_hash:'h'}},resume_checkpoint:'failed/a/latest.pt',resume_checkpoint_sha256:'hash'}}),initial);
  const sampling={...request,operation:'sample_model',config:{sampling:{seed:42},trial:{steps:3,trajectories:1},sampling_run:'trial'}};
  assert.equal(requestSignature(sampling),requestSignature({...sampling,config:{...sampling.config,sampling_run:'full'}}));
  assert.notEqual(requestSignature(sampling),requestSignature({...sampling,config:{...sampling.config,trial:{steps:4,trajectories:1}}}));
  assert.equal(signature({x:-0,y:1}),signature({y:1,x:0}));assert.notEqual(signature({x:false}),signature({x:0}));
});
test('editing a displayed training/sampling summary never imports authority flags or replaces its checkpoint',()=>{
  const previous={network:{epochs:20},resume_checkpoint:'failed/original/latest.pt',resume_checkpoint_sha256:'original-sha',plan_refs:{training:{plan_id:'old'}}};
  const edited=applySummary(previous,{network:{epochs:40},plan_refs:{training:{plan_id:'forged'}},resume_checkpoint:'other/latest.pt'});
  assert.equal(edited.resume_checkpoint,previous.resume_checkpoint);assert.equal(edited.resume_checkpoint_sha256,previous.resume_checkpoint_sha256);assert.equal(edited.plan_refs,undefined);assert.equal(edited.network.epochs,40);
});
test('browser drafts reject overflow and isolate workspace/role keys without changing actual paths',()=>{
  assert.throws(()=>parseObject('{"learning_rate":1e400}'),/有限/);assert.throws(()=>parseObject('[]'),/对象/);
  assert.equal(workspaceKey(' D:\\Cases\\A\\ ','model'),workspaceKey('d:/cases/a','model'));
  assert.notEqual(workspaceKey('D:/Cases/A','model'),workspaceKey('D:/Cases/A','lightweight'));
  assert.notEqual(workspaceKey('D:/Cases/A','model'),workspaceKey('D:/Cases/B','model'));
});
test('actual report callback default max_rows/filter requests read saved array rows; ordinary file preview remains metadata-only',()=>{
  for(const cfg of [{max_rows:200},{filter:{key:'trajectory_id',value:3}},{keys:['time','next_y']},{max_rows:200,filter:{key:'condition_index',value:4}}])assert.equal(reportReadOperation('test-candidate-rollout.npz',cfg),'read_report_arrays');
  assert.equal(reportReadOperation('expert.npz',{}),'read_artifact_file');assert.equal(reportReadOperation('report.json',{max_rows:200}),'read_artifact_file');
});
test('bad or nonfinite number entry retains the prior declared setting and blocks execution instead of deleting it',()=>{
  for(const [raw,validity] of [['',{badInput:true}],['1e309',{}],['-1',{rangeUnderflow:true}],['3.5',{stepMismatch:true}]]){
    let config={epochs:7};const decision=parseNumericInput(raw,validity);if(decision.accepted)config=setPath(config,'epochs',decision.value);
    assert.equal(decision.accepted,false);assert.deepEqual(config,{epochs:7});assert.match(decision.error,/原配置保留/);
  }
  assert.deepEqual(parseNumericInput('',{}),{accepted:true,value:undefined,empty:true});assert.deepEqual(parseNumericInput('17',{}),{accepted:true,value:17,empty:false});
});

test('real Python branch normalizers accept the browser NN layout and reject wrong PINN nesting, without training',()=>{
  const project=resolve(import.meta.dirname,'../..'),request=Object.fromEntries(['train_supervised','train_predictive_model','train_external_imitation','train_pinn'].map(name=>[name,operationDefaults(name).config]));
  const source=String.raw`import copy,json,sys,numpy as np
from types import SimpleNamespace
from control_harness.contracts import ContractError,ModelInterface
from control_harness.supervised import _resolved
from control_harness.predictive import _config as predictive_config,_data_interface
from control_harness.external_imitation import _config as imitation_config
from control_harness.pinn import resolve_pinn_config
cfg=json.load(sys.stdin);interface=SimpleNamespace(input_dim=3,output_dim=2)
result={};resolved,architecture,count=_resolved(cfg['train_supervised'],interface);assert resolved['learning_rate']==.001 and architecture['hidden_dims']==[32,32];result['supervised']={'architecture':architecture,'resolved_config':resolved}
physical=ModelInterface.parse({'schema_version':1,'name':'explicit normalization-only unit fixture','sample_time':.1,'time_semantics':'discrete_state_transition',
 'states':[{'name':'x1','unit':'1','lower':-2,'upper':2},{'name':'x2','unit':'1','lower':-2,'upper':2}],
 'controls':[{'name':'u1','unit':'1','lower':-1,'upper':1}],
 'outputs':[{'name':'y1','unit':'1','lower':-2,'upper':2},{'name':'y2','unit':'1','lower':-2,'upper':2}],'disturbances':[]})
predictive=copy.deepcopy(cfg['train_predictive_model']);assert predictive['C']==[],'UI must leave C unknown until explicitly supplied';predictive['C']=[[1,0],[0,1]]
C,scaler,network,limits=predictive_config(predictive,physical);resolved,architecture,count=_resolved(network,_data_interface(physical));result['predictive']={'architecture':architecture,'scaler':scaler,'resolved_config':resolved,'test_fixture_C_supplied_explicitly':True}
network,scale,threshold,normalization=imitation_config(cfg['train_external_imitation'],interface,np.asarray([[-1.,1.],[1.,-1.]],dtype=np.float64));resolved,architecture,count=_resolved(network,interface);result['external_imitation']={'architecture':architecture,'resolved_config':resolved}
pinn=resolve_pinn_config(cfg['train_pinn']);assert pinn['network_config']['input_dim']==3 and pinn['network_config']['output_dim']==1 and pinn['network']['epochs']==150;result['pinn']=pinn
wrong={**cfg['train_pinn']['network'],'loss_weights':cfg['train_pinn']['loss_weights']}
try:resolve_pinn_config(wrong)
except ContractError:result['wrong_PINN_top_level_rejected']=True
else:raise AssertionError('Wrong PINN nesting was accepted')
print(json.dumps({'status':'PASS_NORMALIZATION_ONLY','branches':result,'training_or_NN_forwards':0},allow_nan=False))`;
  const child=spawnSync(resolve(project,'.venv/Scripts/python.exe'),['-B','-c',source],{cwd:project,input:JSON.stringify(request),encoding:'utf8',windowsHide:true,env:{...process.env,PYTHONPATH:resolve(project,'src'),PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'}});
  assert.equal(child.status,0,child.stderr);const value=JSON.parse(child.stdout);assert.equal(value.status,'PASS_NORMALIZATION_ONLY');assert.equal(value.training_or_NN_forwards,0);assert.equal(value.branches.wrong_PINN_top_level_rejected,true);
  assert.deepEqual(value.branches.pinn.network_config.hidden_dims,request.train_pinn.network.hidden_dims);
});
