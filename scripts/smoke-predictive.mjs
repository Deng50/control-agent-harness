/** S06 N04: real native single-plant sampled-data NN training, three source formats. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { isAbsolute,relative,resolve,sep } from 'node:path';
import { parseArgs } from 'node:util';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence } from './harness-rpc.mjs';

const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},'source-audit':{type:'string'},
  fixtures:{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-predictive.mjs --workspace-root <isolated parent> --evidence <json> [--source-audit <train-rank audit>] [--fixtures <S00 fixtures>] [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values['workspace-root']&&values.evidence,'Explicit workspace-root/evidence required');
const destination=resolve(values['workspace-root']),evidencePath=resolve(values.evidence);
const relation=relative(resolve(project,'test-output'),destination);
assert.ok(relation&&!isAbsolute(relation)&&relation!=='..'&&!relation.startsWith(`..${sep}`));
const json=path=>readFile(path,'utf8').then(JSON.parse),hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const runRoot=resolve(project,'test-output/v2-development/20261117T002006');
const audit=await json(resolve(values['source-audit']??resolve(runRoot,'S06/predictive-training-rank-audit.json')));
assert.equal(audit.status,'TRAINING_INPUT_RANK_FULL_ON_DECLARED_CASES');
const fixtures=resolve(values.fixtures??resolve(runRoot,'fixtures'));
const f01Bytes=await readFile(resolve(fixtures,'F01_spring_2x1.json'));
assert.equal(hash(f01Bytes),'22525d4196bda85f3e3ddba0de0493859a91f515fe301f99cd54f40a84e55824');
const f01=JSON.parse(f01Bytes),C=f01.source_model.C;
assert.deepEqual(C,[[1,0]],'C must come from frozen F01, never inferred identity');
// First candidate declaration. This linear NN is trained by the actual S05
// optimizer, not initialized by copying the known physical A/B/E matrices.
const config={network:{hidden_dims:[],activation:'tanh',epochs:4000,learning_rate:.01,batch_size:32,patience:300,
  seed:20261120,weight_decay:0,min_delta:1e-12,grad_clip:5,max_test_nrmse:.15},C,scaler:'standard',
  max_test_state_nrmse:.15,max_test_output_nrmse:.15,max_rollout_state_nrmse:.15,max_rollout_output_nrmse:.15};
let evidence=values.resume?await json(evidencePath):{schema_version:1,stage:'S06',group:'N04',status:'RUNNING',
  destination,startedAt:new Date().toISOString(),confirmation_source:'test_fixture',preRegisteredConfig:config,
  sourceAudit:audit,fixtureSha256:hash(f01Bytes),sessions:{},jobs:{},admissions:[],confirmations:[],checks:[],
  scope:'Three independent native workspaces import actual S04 sampled artifacts; data->trained next-state NN->portable physical reload. No new sampling/Plant execution, LLM, PINN, MPC or policy stage.'};
assert.equal(evidence.destination,destination);assert.deepEqual(evidence.preRegisteredConfig,config);
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const runPython=(code,args,extra={})=>execFileSync(python,['-B','-c',code,...args],{encoding:'utf8',windowsHide:true,maxBuffer:8*1024*1024,
  env:{...process.env,PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',...extra}});
const runtimeProbe=String.raw`import json,sys,numpy as np,torch
from pathlib import Path
from control_harness.artifacts import Workspace
from control_harness.modeling import load_model
torch.set_num_threads(1)
ws=Workspace(sys.argv[1]);model=load_model(ws,sys.argv[2]);root=Path(sys.argv[3]);out=Path(sys.argv[4])
split=json.loads((root/'split.json').read_text(encoding='utf-8'))
with np.load(root/'transitions.npz',allow_pickle=False) as loaded:a={key:loaded[key] for key in loaded.files}
train=np.isin(a['trajectory_id'],split['train']);test=np.isin(a['trajectory_id'],split['test']);rows=np.flatnonzero(test)
with torch.no_grad():one=model(*[torch.from_numpy(a[k][rows]) for k in ('x','u','d')]).numpy()
group=split['test'][0];sequence=np.flatnonzero(a['trajectory_id']==group);sequence=sequence[np.argsort(a['time'][sequence])]
case={'x0':a['x'][sequence[0]].tolist(),'controls':a['u'][sequence].tolist(),'disturbances':a['d'][sequence].tolist()}
out.write_text(json.dumps(case,allow_nan=False)+'\n',encoding='utf-8')
x=np.array(case['x0']);states=[x.copy()];outputs=[model.output(torch.from_numpy(x)).detach().numpy()]
for u,d in zip(case['controls'],case['disturbances']):
    x=model.step(x,u,d);states.append(x.copy());outputs.append(model.output(torch.from_numpy(x)).detach().numpy())
model_root=ws.resolve(sys.argv[2])[0];report=json.loads((model_root/'report.json').read_text(encoding='utf-8'))
supervised=ws.resolve(report['supervised_model'])[0];checkpoint=torch.load(supervised/'latest.pt',weights_only=True,map_location='cpu')
print(json.dumps({'one_step':one.tolist(),'states':np.asarray(states).tolist(),'outputs':np.asarray(outputs).tolist(),
    'time':(np.arange(len(states))*model.interface.sample_time).tolist(),'steps':len(sequence),
    'test_rows':rows.tolist(),'test_trajectory':int(group),'train_rows':int(train.sum()),
    'state_scale':a['next_x'][train].std(0).tolist(),'output_scale':a['next_y'][train].std(0).tolist(),
    'source_truth_states':a['next_x'][sequence].tolist(),'source_truth_outputs':a['next_y'][sequence].tolist(),
    'model_weight_shapes':{k:list(v.shape) for k,v in model.network.state_dict().items()},
    'actual_optimizer_lr':[g['lr'] for g in checkpoint['optimizer_state']['param_groups']],
    'actual_optimizer_updates':checkpoint['updates'],'actual_epoch':checkpoint['epoch'],
    'train_ids':split['train'],'validation_ids':split['validation'],'test_ids':split['test']},allow_nan=False))`;
const npz=String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as a: print(json.dumps({key:a[key].tolist() for key in a.files},allow_nan=False))`;
const isolated=String.raw`import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('No project imports in independent physical bundle')
sys.meta_path.insert(0,Block());p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')
assert not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)`;
const near=(a,b,label,atol=1e-6,rtol=1e-5)=>{
  if(Array.isArray(b)){assert.equal(a.length,b.length,label);b.forEach((v,k)=>near(a[k],v,`${label}[${k}]`,atol,rtol));}
  else assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=atol+rtol*Math.abs(b),`${label}: ${a} vs ${b}`);
};
const metric=(prediction,truth,scale)=>scale.map((s,k)=>Math.sqrt(prediction.reduce((sum,row,i)=>sum+(row[k]-truth[i][k])**2,0)/truth.length)/s);

try{
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG});
  for(const source of audit.checks){
    const format=source.backend,workspace=resolve(destination,format);await prepareSmokeWorkspace(workspace);
    assert.equal(hash(await readFile(resolve(source.dataset_root,'transitions.npz'))),source.transitions_sha256);
    assert.equal(hash(await readFile(resolve(source.dataset_root,'split.json'))),source.split_sha256);
    if(!values.resume){
      try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'Use --resume for existing jobs');}catch(e){if(e.code!=='ENOENT')throw e;}
      await writeSmokeEvidence(resolve(workspace,'.case-plans/test-fixture.json'),{kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true});
    }
    if(!evidence.sessions[format]){const s=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});assert.equal(s.agentPreset,'control-model');evidence.sessions[format]=s.sessionId;await save();}
    const sessionId=evidence.sessions[format];
    const stage=async(key,operation,inputs={},settings={})=>{
      const request={agent:'model',operation,workspace,inputs,config:settings},requestHash=hash(JSON.stringify(request));
      let entry=evidence.jobs[`${format}:${key}`];
      if(entry){assert.equal(entry.requestHash,requestHash);assert.ok(entry.jobId,'Uncertain admission: reconcile saved history; never resubmit');if(entry.result)return entry.result;}
      else{entry=evidence.jobs[`${format}:${key}`]={operation,requestHash,status:'ADMISSION_PENDING'};await save();
        const start=await command(rpc,sessionId,`/control ${JSON.stringify(request)}`);Object.assign(entry,{jobId:start.jobId,attemptId:start.attemptId,status:'RUNNING'});
        evidence.admissions.push({format,operation,sessionId,jobId:entry.jobId,attemptId:entry.attemptId});await save();}
      for(;;){const state=await command(rpc,sessionId,`/control-status ${JSON.stringify({workspace,attemptId:entry.attemptId})}`);
        if(['completed','failed','killed'].includes(state.record.state)){entry.status=state.record.state;await save();assert.equal(entry.status,'completed',JSON.stringify(state));entry.result=state.record.result;assert.ok(entry.result);await save();return entry.result;}
        assert.notEqual(state.record.state,'interrupted','Host ownership lost: reconcile original job');await new Promise(done=>setTimeout(done,300));}
    };
    const imported=await stage('import','import_artifact',{artifact_path:source.dataset_root});
    assert.equal(imported.artifact_id,source.dataset_id);assert.equal(imported.kind,'sampled_dataset');
    assert.equal(hash(await readFile(resolve(imported.artifact_path,'transitions.npz'))),source.transitions_sha256);
    const training=structuredClone(config),request={operation:'train_predictive_model',workspace,inputs:{dataset:imported.artifact_id},config:training};
    const requirements=await confirmationRequirements(request);assert.deepEqual(requirements.map(item=>item.scope),['signals','training']);
    const refs={};
    for(const row of requirements){const plan=await stage(`plan-${row.scope}`,'propose_case_plan',{}, {scope:row.scope,payload:row.expectedPayload,bindings:row.bindings});
      refs[row.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};const confirmed=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace,...refs[row.scope]})}`);
      assert.equal(confirmed.confirmation_source,'test_fixture');evidence.confirmations.push({format,scope:row.scope,...refs[row.scope],confirmation_source:'test_fixture'});await save();}
    const result=await stage('training','train_predictive_model',request.inputs,{...training,plan_refs:refs});
    assert.equal(result.kind,'learned_model');assert.equal(result.status,'verified');
    const report=await json(resolve(result.artifact_path,'report.json')),descriptor=await json(resolve(result.artifact_path,'learned.json'));
    assert.deepEqual(report.C,C);assert.deepEqual(descriptor.C,C);assert.equal(report.nominal_or_reference_model_required,false);
    assert.equal(report.pinn,false);assert.equal(report.test_used_for_training_or_selection,false);assert.equal(report.selection,'S05_validation_only');
    assert.equal(report.target_semantics,'next_x_in_state_units_no_dt_multiplication');assert.equal(report.normalization.source,'training_partition_only');
    for(const [key,value]of Object.entries(config.network))assert.deepEqual(report.network.resolved_config[key],value,`actual ${format} network ${key}`);
    const input=resolve(workspace,'portable-case.json'),output=resolve(workspace,'portable-rollout.json');
    const internal=JSON.parse(runPython(runtimeProbe,[workspace,result.artifact_id,imported.artifact_path,input],{PYTHONPATH:resolve(project,'src')}));
    assert.equal(internal.steps,50);assert.equal(internal.train_rows,150);assert.deepEqual(internal.train_ids,source.train_trajectories);
    assert.deepEqual(internal.model_weight_shapes,{'0.weight':[2,4],'0.bias':[2]},'A real single-layer NN was not trained/exported');
    assert.deepEqual(internal.actual_optimizer_lr,[.01]);assert.equal(internal.actual_epoch,report.network.trained_epochs);
    assert.equal(internal.actual_optimizer_updates,report.network.optimizer_updates);
    near(report.normalization.state_nrmse_scale,internal.state_scale,'train-only state std',1e-12,1e-12);
    near(report.normalization.output_nrmse_scale,internal.output_scale,'train-only output std',1e-12,1e-12);
    const one=JSON.parse(runPython(npz,[resolve(result.artifact_path,'one_step_predictions.npz')]));
    near(one.predicted_state,internal.one_step,'reloaded teacher-forced predictions');
    assert.deepEqual(one.source_row,internal.test_rows);
    near(metric(one.predicted_state,one.true_state,internal.state_scale),report.one_step.test.state.nrmse,'independent one-step state NRMSE',1e-10,1e-10);
    near(metric(one.predicted_output,one.true_output,internal.output_scale),report.one_step.test.output.nrmse,'independent one-step output NRMSE',1e-10,1e-10);
    const bundle=resolve(result.artifact_path,'bundle');
    execFileSync(python,['-B','-c',isolated,resolve(bundle,'simulate.py'),'--input',input,'--output',output],{cwd:bundle,encoding:'utf8',windowsHide:true,
      env:{...process.env,PYTHONPATH:'',PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'}});
    const portable=await json(output);assert.equal(portable.semantics,'discrete_next_state_no_extra_dt');assert.equal(portable.sample_time,.2);
    near(portable.states,internal.states,'independent portable states');near(portable.outputs,internal.outputs,'independent portable outputs');near(portable.time,internal.time,'portable dt grid',1e-12,1e-12);
    assert.deepEqual(portable.state_order,f01.state_order);assert.deepEqual(portable.output_order,f01.output_order);
    assert.deepEqual(portable.state_units,['m','m/s']);assert.deepEqual(portable.output_units,['m']);
    near(metric(portable.states.slice(1),internal.source_truth_states,internal.state_scale),report.rollout.test.state.nrmse,'independent 50-step state NRMSE',1e-9,1e-9);
    near(metric(portable.outputs.slice(1),internal.source_truth_outputs,internal.output_scale),report.rollout.test.output.nrmse,'independent 50-step output NRMSE',1e-9,1e-9);
    for(const item of [report.one_step.test.state,report.one_step.test.output,report.rollout.test.state,report.rollout.test.output])assert.ok(item.nrmse.every(value=>value<=.15),'Pre-registered physical test gate failed; do not retune on held-out test');
    assert.equal(report.rollout.all_trajectories_completed,true);assert.equal(report.acceptance,'passed');
    const metadata=await json(resolve(result.artifact_path,'artifact.json'));
    for(const[name,expected]of Object.entries(metadata.files))assert.equal(hash(await readFile(resolve(result.artifact_path,name))),expected);
    const current=await json(resolve(workspace,'manifest.json'));
    assert.ok(!Object.values(current.artifacts).some(row=>['nominal_model','reference_model','controller','expert_dataset','imitation_policy','optimized_policy'].includes(row.kind)),'Single-plant training fabricated a second model or downstream control branch');
    const records=(await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:10000}})).records;
    const eventTypes=[...new Set(records.map(row=>row.event.type))];assert.ok(!eventTypes.some(type=>type.startsWith('turn/')));
    const check={format,workspace,source:source.dataset_root,dataset:imported.artifact_id,model:result.artifact_id,preRegisteredConfig:config,
      sameSourceBytes:true,rankAudit:source,teacherForced:report.one_step,independent50Step:report.rollout,normalization:report.normalization,
      actualNetwork:report.network,actualWeightShapes:internal.model_weight_shapes,actualOptimizerLR:internal.actual_optimizer_lr,
      exportIndependentProcess:true,projectImportsBlocked:true,originalPlantExecution:false,eventTypes,confirmation_source:'test_fixture'};
    const index=evidence.checks.findIndex(item=>item.format===format);if(index<0)evidence.checks.push(check);else evidence.checks[index]=check;await save();
  }
  assert.equal(evidence.checks.length,3);evidence.status='PASS';evidence.finishedAt=new Date().toISOString();
}catch(error){evidence.status='FAIL';evidence.error=error instanceof Error?error.stack:String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,checks:evidence.checks.length,
  jobs:evidence.admissions.map(({format,operation,jobId})=>({format,operation,jobId})),confirmation_source:'test_fixture'}));}
