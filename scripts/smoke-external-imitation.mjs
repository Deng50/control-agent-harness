/** S10 L04/L05 native acceptance. Uses an already running Host; never starts MATLAB. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFile,readdir} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},'training-config':{type:'string'},
  'training-config-sha256':{type:'string'},'defer-evaluation':{type:'boolean'},fixtures:{type:'string'},'mcp-audit':{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-external-imitation.mjs --workspace-root <isolated parent> --evidence <json> --training-config <preregistered JSON> --mcp-audit <trusted observer JSONL> [--fixtures <S00 directory>] [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values['workspace-root']&&values.evidence&&values['training-config']&&values['mcp-audit'],'Explicit workspace/evidence/frozen training config/MCP audit required');
const destination=resolve(values['workspace-root']),evidencePath=resolve(values.evidence),configPath=resolve(values['training-config']),auditPath=resolve(values['mcp-audit']);
const descendant=relative(resolve(project,'test-output'),destination);assert.ok(descendant&&!isAbsolute(descendant)&&descendant!=='..'&&!descendant.startsWith(`..${sep}`));
const json=path=>readFile(path,'utf8').then(JSON.parse),hash=value=>createHash('sha256').update(value).digest('hex');
const runRoot=resolve(project,'test-output/v2-development/20261117T002006'),fixtures=resolve(values.fixtures??resolve(runRoot,'fixtures'));
const bytes=await readFile(configPath),registration=JSON.parse(bytes);assert.equal(hash(bytes),values['training-config-sha256']??'c43f2826fa283dd7f557aa8b96e06f7faa202d814db2bdf9c58e1f010f5493d9','Use the explicitly preregistered immutable S10 configuration');assert.equal(registration.schema_version,1);assert.ok(Array.isArray(registration.cases)&&registration.cases.length>0);
assert.equal(new Set(registration.cases.map(row=>row.key)).size,registration.cases.length);
const f02File=resolve(fixtures,'F02_existing_linear_4x2.json'),f02Hash=hash(await readFile(f02File));assert.equal(f02Hash,'ac6364b26915a81abc790a88c6346e2fe91fdd62950898ecac0cc0c67f6ae7d9');
const sources=[];
for(const item of registration.cases){assert.match(item.key,/^[A-Za-z0-9_-]+$/);assert.ok(item.dataset_artifact_path&&item.training_config?.network);
  const path=resolve(item.dataset_artifact_path),metadata=await json(resolve(path,'artifact.json'));assert.equal(metadata.kind,'external_expert_dataset');
  if(item.expected_dataset_id)assert.equal(metadata.id,item.expected_dataset_id);
  sources.push({key:item.key,path,id:metadata.id,metadata_sha256:hash(await readFile(resolve(path,'artifact.json'))),expert_npz_sha256:hash(await readFile(resolve(path,'expert.npz')))});
}
const preRegistered={training_config_path:configPath,training_config_sha256:hash(bytes),registration,sources,f02_sha256:f02Hash,
  selection:'Actual S05 validation-only model selection. Source whole-trajectory train/validation/test split is preserved; test never selects hyperparameters.',
  native_counts:{business_batch:1,fixed_guard_probes:2,total:3},benchmark_claim:'Measured controller-only latency on this machine; no mandatory speedup or stability guarantee'};
const portableRoot=resolve(project,'..','control-agent-harness-s10-portable',hash(destination).slice(0,12));
assert.ok(relative(project,portableRoot).startsWith('..'+sep),'Portable exports must be outside the repository');
let evidence=values.resume?await json(evidencePath):{schema_version:1,stage:'S10',groups:['L04','L05'],status:'RUNNING',destination,auditPath,preRegistered,
  started_at:new Date().toISOString(),confirmation_source:'test_fixture',sessions:{},jobs:{},admissions:[],confirmations:[],rejections:[],checks:[]};
assert.equal(evidence.destination,destination);assert.equal(evidence.auditPath,auditPath);assert.deepEqual(evidence.preRegistered,preRegistered,'Resume preserves source bytes, candidates, metrics and gates');
if(values.resume){evidence.observer_history??=[];if(evidence.error)evidence.observer_history.push({error:evidence.error,status:evidence.status,resumed_at:new Date().toISOString()});delete evidence.error;delete evidence.finished_at;evidence.status='RUNNING';}
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const isolated=(code,args)=>JSON.parse(execFileSync(python,['-I','-B','-c',code,...args],{encoding:'utf8',windowsHide:true,maxBuffer:16*1024*1024,
  env:{...process.env,PYTHONPATH:'',PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'}}));
const auditRows=async sessionId=>{let value='';try{value=await readFile(auditPath,'utf8');}catch(error){if(error.code!=='ENOENT')throw error;}
  return value.split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line)).filter(row=>row.source==='trusted_test_observer'&&row.sessionId===sessionId);};
function auditCalls(rows,matlab){
  if(!matlab){assert.equal(rows.length,0,'Python-only imitation/evaluation must not call MATLAB');return {business_batch_calls:0,fixed_guard_probe_calls:0,total_actual_mcp_calls:0};}
  const starts=rows.filter(row=>row.event==='started'),returns=rows.filter(row=>row.event==='returned');assert.equal(starts.length,3);assert.equal(returns.length,3);assert.equal(rows.length,6);
  const seen=new Set();let batches=0,probes=0;
  for(const row of starts){const id=row.callId??row.rootCallId;assert.ok(id&&!seen.has(id));seen.add(id);assert.equal(row.name,'mcp__matlab__evaluate_matlab_code');
    const pair=returns.filter(other=>(other.callId??other.rootCallId)===id);assert.equal(pair.length,1);assert.equal(pair[0].isError,false);
    const key=[row.callId,row.rootCallId].find(value=>value?.startsWith('matlab-batch-')||value?.startsWith('matlab-probe-'));
    if(key?.startsWith('matlab-batch-'))batches++;else if(key?.startsWith('matlab-probe-'))probes++;else assert.fail('Unexpected actual native tool identity '+id);
  }
  assert.equal(batches,1);assert.equal(probes,2);return {business_batch_calls:batches,fixed_guard_probe_calls:probes,total_actual_mcp_calls:3,calls:starts,
    source:'Actual trusted optional around-next observer; paired returned calls, not inferred from numerical jobs or synthetic Session events'};
}

const offlineAudit=String.raw`import hashlib,importlib.abc,importlib.util,json,shutil,sys
from pathlib import Path
import numpy as np,torch
class NoHarness(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='control_harness' or fullname.startswith('control_harness.'):raise ModuleNotFoundError('Portable policy must not import Harness')
sys.meta_path.insert(0,NoHarness());torch.set_num_threads(1)
policy,source,workspace,case=[Path(x) for x in sys.argv[1:5]];case=json.loads(case.read_text(encoding='utf-8'));read=lambda p:json.loads(p.read_text(encoding='utf-8'))
meta=read(policy/'artifact.json');report=read(policy/'report.json');dataset=read(source/'dataset.json');split=read(source/'split.json');definition=read(policy/'policy.json')
for directory in [source,policy]:
 m=read(directory/'artifact.json')
 for name,digest in m['files'].items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
assert meta['kind']=='external_imitation_policy' and meta['validation']['status']=='verified'
assert definition['external_expert_dataset']==read(source/'artifact.json')['id']==report['external_expert_dataset']
with np.load(source/'expert.npz',allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
with np.load(policy/'test_predictions.npz',allow_pickle=False) as a:predictions={k:a[k].copy() for k in a.files}
assert len(arrays['y'])==case.get('expected_samples',1280);assert arrays['X'].shape[1]==9 and arrays['y'].shape[1]==2
groups=split['groups'];expected=case.get('expected_split_counts',{'train':20,'validation':6,'test':6});assert {k:len(v) for k,v in groups.items()}==expected
for part,index in [('train',0),('validation',1),('test',2)]:assert set(arrays['trajectory_id'][arrays['split_id']==index].tolist())==set(groups[part])
selected=np.flatnonzero(arrays['split_id']==2);np.testing.assert_array_equal(predictions['source_row'],selected);np.testing.assert_array_equal(predictions['expert_action'],arrays['y'][selected])
train=arrays['y'][arrays['split_id']==0];std=train.std(axis=0);constant=np.all(train==train[0],axis=0);std[constant]=0
assert not constant.any(),'Fixed F05 action targets must have nonconstant training distributions'
declared=case['training_config'].get('action_metric_scale','train_std')
if isinstance(declared,str) and declared=='train_std':
 action_scale=std;assert report['normalization']['source']=='training_partition_only'
else:
 action_scale=np.asarray(declared,dtype=float);assert report['normalization']['source']=='explicit_user_scales'
np.testing.assert_array_equal(report['normalization']['action_nrmse_scale'],action_scale)
np.testing.assert_allclose(report['normalization']['training_action_population_std'],std,atol=1e-12,rtol=1e-12)
assert report['normalization']['fitted_using_test'] is False and report['normalization']['scales_fixed_before_training'] is True
derived=workspace/'artifacts'/definition['supervised_dataset'];nn=workspace/'artifacts'/definition['supervised_model']
assert read(derived/'split.json')==split
with np.load(derived/'supervised.npz',allow_pickle=False) as a:
 for name in ['X','y','X_scaled','y_scaled','split_id']:np.testing.assert_array_equal(a[name],arrays[name])
scaler=read(derived/'scaler.json');assert scaler['fit_split']=='train' and scaler['fit_samples']==int((arrays['split_id']==0).sum())
for name in ['X','y']:
 a=arrays[name][arrays['split_id']==0];loc=a.mean(0);scale=a.std(0);scale[scale==0]=1
 np.testing.assert_allclose(scaler[name]['location'],loc,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(scaler[name]['scale'],scale,atol=1e-12,rtol=1e-12)
network=case['training_config']['network'];resolved=report['network']['resolved_config']
for key,value in network.items():assert resolved[key]==value,(key,resolved.get(key),value)
checkpoint=torch.load(nn/'latest.pt',map_location='cpu',weights_only=True);assert checkpoint['optimizer_state']['param_groups'][0]['lr']==network['learning_rate'];assert checkpoint['updates']>0
assert report['test_used_for_training_or_selection'] is False and report['test_evaluations']==1 and report['selection']=='S05_validation_only'
assert report['closed_loop']=='not_run' and report['pgac']=='not_run' and report['internal_controller_required'] is False
assert report['offline']['test']['mode']=='teacher_forced_recorded_features_and_actual_previous_actions'
portable=Path(sys.argv[5]).resolve();project=Path(sys.argv[6]).resolve();bundle=portable/'bundle';portable.mkdir(parents=True,exist_ok=True)
assert not bundle.is_relative_to(workspace) and not bundle.is_relative_to(project)
if bundle.exists():
 original={p.relative_to(policy/'bundle').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (policy/'bundle').rglob('*') if p.is_file()}
 copied={p.relative_to(bundle).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file()};assert copied==original,'Existing portable copy is immutable; never overwrite a changed copy'
else:shutil.copytree(policy/'bundle',bundle)
spec=importlib.util.spec_from_file_location('_independent_policy',bundle/'policy_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.load_external_policy(bundle)
X=arrays['X'][selected];previous=arrays['previous_u'][selected];raw=model.predict_raw(X);actions=model.predict(X,previous_action=previous)
np.testing.assert_allclose(raw,predictions['raw_action'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(actions,predictions['projected_action'],atol=1e-12,rtol=1e-12)
rmse=np.sqrt(np.mean((actions-arrays['y'][selected])**2,axis=0));nrmse=rmse/action_scale;np.testing.assert_allclose(report['offline']['test']['projected']['nrmse'],nrmse,atol=1e-12,rtol=1e-12)
threshold=case['training_config']['max_test_action_nrmse'];assert np.all(nrmse<=np.asarray(threshold))
bounds=np.asarray(read(source/'action_constraints.json')['bounds']);rate=np.asarray(read(source/'action_constraints.json')['rate_bounds']);delta=actions-previous
assert np.all(actions>=bounds[:,0]-1e-12) and np.all(actions<=bounds[:,1]+1e-12) and np.all(delta>=rate[:,0]-1e-12) and np.all(delta<=rate[:,1]+1e-12)
assert not any(name=='control_harness' or name.startswith('control_harness.') for name in sys.modules)
print(json.dumps({'status':'PASS','samples':len(arrays['y']),'test_samples':len(selected),'split_counts':expected,'training_action_std':std.tolist(),'frozen_action_metric_scale':action_scale.tolist(),'normalization_source':report['normalization']['source'],'projected_test_nrmse':nrmse.tolist(),
 'raw_test_nrmse':report['offline']['test']['raw']['nrmse'],'portable_projected_max_error':float(np.max(np.abs(actions-predictions['projected_action']))),'portable_process':'python -I; repo-external explicit inherited-ACL mkdir/copy; Harness imports blocked','portable_directory':str(bundle),
 'actual_optimizer_learning_rate':checkpoint['optimizer_state']['param_groups'][0]['lr'],'actual_weight_shapes':{k:list(v.shape) for k,v in checkpoint['model_state'].items()},'train_only_scaler':True,'test_selected_hyperparameters':False},allow_nan=False))`;

const closedLoopAudit=String.raw`import hashlib,json,sys,numpy as np
from pathlib import Path
root,workspace,frozen=[Path(x) for x in sys.argv[1:4]];read=lambda p:json.loads(p.read_text(encoding='utf-8'));report=read(root/'report.json');meta=read(root/'artifact.json');fixture=read(frozen)
for name,digest in meta['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert report['status']=='completed' and report['policy_activation']=='not_changed' and report['policy_replacement']=='not_performed' and report['stability_proof'] is False
plant=read(workspace/'artifacts'/report['plant']/'plant.json')['adapter_spec'];parameters=plant['parameters'];match=[]
for key in ['nominal','reference']:
 if all(np.allclose(parameters[k],fixture[key][k],atol=0,rtol=0) for k in ['A','B','E']):match.append(key)
assert len(match)==1;reference=fixture[match[0]];A,B,C,E=[np.asarray(reference[k]) for k in ['A','B','C','E']];i=report['physical_interface'];plan=report['plan'];checks={};allactions={}
for name in ['expert','imitation']:
 with np.load(root/(name+'-rollout.npz'),allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
 assert report['closed_loop'][name]['feedback'].startswith('own past applied actions');max_error=0.;allactions[name]=arrays['u'];dt=plan['control_period'];metrics={r['trajectory_id']:r['metrics'] for r in report['closed_loop'][name]['metrics']}
 for case in plan['trajectories']:
  ids=np.flatnonzero(arrays['trajectory_id']==case['trajectory_id']);assert len(ids)==case['steps'];x=np.array(case['x0']);previous=np.array(case['initial_action']);errors=[];energy=tv=0.
  for row in ids:
   np.testing.assert_allclose(arrays['x'][row],x,atol=1e-10,rtol=1e-10);np.testing.assert_allclose(arrays['plant_y'][row],C@x,atol=1e-10,rtol=1e-10);np.testing.assert_array_equal(arrays['previous_u'][row],previous)
   u=arrays['u'][row];d=arrays['d'][row]
   for tick in range(plan['plant_substeps']):x=A@x+B@u+E@d
   y=C@x;max_error=max(max_error,float(np.max(np.abs(x-arrays['next_x'][row]))));np.testing.assert_allclose(arrays['next_x'][row],x,atol=1e-9,rtol=1e-9);np.testing.assert_allclose(arrays['next_y'][row],y,atol=1e-9,rtol=1e-9)
   errors.append(y-arrays['reference'][row]);energy+=float(np.sum(u*u))*dt;tv+=float(np.sum(np.abs(u-previous)));previous=u
  actual=metrics[case['trajectory_id']];np.testing.assert_allclose([actual['rmse'],actual['energy'],actual['tv']],[np.sqrt(np.mean(np.square(errors))),energy,tv],atol=1e-9,rtol=1e-9)
 checks[name]={'samples':len(arrays['time']),'independent_matrix_max_error':max_error,'raw_metrics_replayed':True,'own_previous_actions_verified':True}
bench=report['controller_benchmark'];assert bench['status']=='completed' and bench['settings']['threads']==bench['settings']['batch_size']==1
for name in ['expert','imitation']:
 raw=np.asarray(bench[name]['raw_seconds']);assert len(raw)==bench['settings']['repeats'] and np.isfinite(raw).all() and (raw>=0).all();np.testing.assert_allclose(bench[name]['median'],np.median(raw),atol=0,rtol=0)
if 'hardware' in bench:assert bench['hardware']['torch_threads']==1 and all(p['num_threads']==1 for p in bench['hardware']['threadpools'])
else:assert bench['matlab_max_computational_threads']==1
print(json.dumps({'status':'PASS','frozen_matrix_variant':match[0],'rollouts':checks,'own_action_sequence_difference':float(np.max(np.abs(allactions['expert']-allactions['imitation']))),
 'candidate_feasible':report['candidate_feasible'],'physical_degradation':report['degradation'],'benchmark':bench,'stability_proof':False,'independent_oracle':'NumPy frozen matrices and direct raw metric arithmetic; no original Plant function, Harness module or MPC solver imported'},allow_nan=False))`;

try{
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  for(const item of registration.cases){
    const workspace=resolve(destination,item.key),source=sources.find(row=>row.key===item.key);await prepareSmokeWorkspace(workspace);
    if(!evidence.sessions[item.key]){try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'Fresh workspace required; resume saved handles');}catch(error){if(error.code!=='ENOENT')throw error;}}
    const markerPath=resolve(workspace,'.case-plans/test-fixture.json'),markerSource={kind:'S10_preregistered_native_acceptance',registration_sha256:hash(bytes),dataset_id:source.id,
      original_dataset_artifact_path:source.path,original_dataset_metadata_sha256:source.metadata_sha256};let marker;
    try{marker=await json(markerPath);}catch(error){if(error.code!=='ENOENT')throw error;assert.ok(!evidence.sessions[item.key],'A saved Session requires its retained original fixture marker');
      marker={kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true,source:markerSource};await writeSmokeEvidence(markerPath,marker);}
    assert.equal(marker.kind,'control-case-plan-test-fixture-v1');assert.equal(marker.workspace,workspace);assert.equal(marker.confirmation_source,'test_fixture');assert.equal(marker.synthetic,true);assert.deepEqual(marker.source,markerSource,'Retain the original fixture authorization/source instead of rewriting it on resume');
    evidence.fixture_markers??={};const markerRecord={path:markerPath,sha256:hash(await readFile(markerPath)),source:markerSource};
    if(evidence.fixture_markers[item.key])assert.deepEqual(evidence.fixture_markers[item.key],markerRecord);else{evidence.fixture_markers[item.key]=markerRecord;await save();}
    if(!evidence.sessions[item.key]){
      const session=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});assert.equal(session.agentPreset,'control-lightweight');evidence.sessions[item.key]=session.sessionId;await save();}
    const sessionId=evidence.sessions[item.key],events=async()=>{
      const page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:10000}});assert.equal(page.hasMore,false,'Full Session events are required; a truncated page cannot prove zero LLM turns/jobs');
      const path=resolve(dirname(evidencePath),`external-imitation-${hash(destination).slice(0,12)}-${item.key}-session-events.json`);await writeSmokeEvidence(path,page);
      evidence.session_events??={};evidence.session_events[item.key]={path,sha256:hash(await readFile(path)),sessionId,records:page.records.length,hasMore:page.hasMore};await save();return page.records;
    };
    const snapshot=async()=>{let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(name=>/^[0-9a-f-]{36}$/.test(name)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}
      const history=await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace})}`),records=await events();return {attempts,history:history.attempts.map(row=>row.attemptId).sort(),jobs:records.filter(row=>row.event.type==='job/registered').map(row=>row.event.data.jobId),mcp:(await auditRows(sessionId)).length};};
    const deny=async(label,operation,inputs,config)=>{label=item.key+':'+label;if(evidence.rejections.some(row=>row.label===label))return;
      const before=await snapshot(),response=await rpc('commands/execute',{agentId:sessionId,line:`/control ${JSON.stringify({agent:'lightweight',operation,workspace,inputs,config})}`,submittedAttachments:[]});
      assert.equal(response.result.kind,'error');assert.match(response.result.text,/confirm|plan|version|review/i);const after=await snapshot();assert.deepEqual(after,before);evidence.rejections.push({label,status:'PASS_ZERO_ADMISSION',message:response.result.text,before,after});await save();};
    const stage=async(key,operation,inputs={},config={})=>{const request={agent:'lightweight',operation,workspace,inputs,config},signature=hash(JSON.stringify(request)),id=item.key+':'+key;let row=evidence.jobs[id];
      if(row){assert.equal(row.signature,signature);assert.ok(row.jobId,'Uncertain admission: reconcile existing native history, never resubmit');if(row.result)return row.result;}
      else{row=evidence.jobs[id]={operation,signature,status:'ADMISSION_PENDING',sessionId};await save();const started=await command(rpc,sessionId,`/control ${JSON.stringify(request)}`);Object.assign(row,{jobId:started.jobId,attemptId:started.attemptId,status:'RUNNING'});evidence.admissions.push({key:id,operation,sessionId,jobId:row.jobId,attemptId:row.attemptId});await save();}
      for(;;){const state=await command(rpc,sessionId,`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);if(['completed','failed','killed'].includes(state.record.state)){row.state=state.record;row.status=state.record.state;await save();assert.equal(row.status,'completed',JSON.stringify({operation,attemptId:row.attemptId,status:row.status,detail:state.record.detail,result:state.record.result}));row.result=state.record.result;assert.ok(row.result);await save();return row.result;}
        assert.notEqual(state.record.state,'interrupted','Inspect/recover saved attempt rather than admitting a replacement');await new Promise(done=>setTimeout(done,400));}
    };
    const confirm=async(key,operation,inputs,config)=>{const request={operation,workspace,inputs,config},requirements=await confirmationRequirements(request),refs={};
      assert.deepEqual(requirements.map(row=>row.scope),operation==='train_external_imitation'?['signals','training']:['signals','sampling']);
      for(const row of requirements){const plan=await stage(key+'-plan-'+row.scope,'propose_case_plan',{}, {scope:row.scope,payload:row.expectedPayload,bindings:row.bindings});refs[row.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};}
      const statuses={};for(const row of requirements){const status=await command(rpc,sessionId,`/control-plan-status ${JSON.stringify({workspace,...refs[row.scope]})}`);assert.equal(status.session_id,sessionId);assert.equal(status.plan_id,refs[row.scope].plan_id);assert.equal(status.version_hash,refs[row.scope].version_hash);assert.equal(status.scope,row.scope);assert.ok(['pending','confirmed'].includes(status.confirmation_state));statuses[row.scope]=status;}
      if(Object.values(statuses).some(row=>row.confirmation_state==='pending'))await deny(key+'-before-confirmation',operation,inputs,{...config,plan_refs:refs});
      else assert.ok(evidence.rejections.some(row=>row.label===item.key+':'+key+'-before-confirmation'&&row.status==='PASS_ZERO_ADMISSION'),'Already-confirmed plans are reused; missing historical denial evidence cannot be invented');
      for(const row of requirements){let confirmed=statuses[row.scope],reused=confirmed.confirmation_state==='confirmed';
        if(!reused)confirmed=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace,...refs[row.scope]})}`);
        assert.equal(confirmed.confirmation_source,'test_fixture');assert.equal(confirmed.session_id,sessionId);assert.equal(confirmed.plan_id,refs[row.scope].plan_id);assert.equal(confirmed.version_hash,refs[row.scope].version_hash);assert.equal(confirmed.scope,row.scope);assert.equal(confirmed.source.kind,'native_user_command');
        const existing=evidence.confirmations.find(record=>record.key===item.key&&record.phase===key&&record.scope===row.scope&&record.sessionId===sessionId&&record.plan_id===refs[row.scope].plan_id&&record.version_hash===refs[row.scope].version_hash);
        const record={key:item.key,phase:key,scope:row.scope,sessionId,...refs[row.scope],confirmation_source:'test_fixture',source:confirmed.source};
        if(existing)assert.deepEqual(existing,record);else evidence.confirmations.push(record);
        if(reused){evidence.confirmation_reuses??=[];evidence.confirmation_reuses.push({...record,reused_at:new Date().toISOString(),verified_via:'control-plan-status'});}await save();}return refs;
    };
    assert.equal(hash(await readFile(resolve(source.path,'artifact.json'))),source.metadata_sha256);
    const dataset=await stage('import','import_artifact',{artifact_path:source.path});assert.equal(dataset.artifact_id,source.id);assert.equal(dataset.kind,'external_expert_dataset');
    const inputs={dataset:dataset.artifact_id},trainRefs=await confirm('train','train_external_imitation',inputs,item.training_config);
    const policy=await stage('train','train_external_imitation',inputs,{...item.training_config,plan_refs:trainRefs});assert.equal(policy.kind,'external_imitation_policy');assert.equal(policy.status,'verified');
    const changed=structuredClone(item.training_config);changed.network.learning_rate*=1.01;await deny('changed-network-old-confirmation','train_external_imitation',inputs,{...changed,plan_refs:trainRefs});
    const caseFile=resolve(workspace,'s10-preregistered-case.json');await writeSmokeEvidence(caseFile,item);
    const offline=isolated(offlineAudit,[policy.artifact_path,dataset.artifact_path,workspace,caseFile,resolve(portableRoot,item.key,policy.artifact_id),project]);assert.equal(offline.status,'PASS');
    const checkStaticNoMatlab=async()=>{evidence.static_mcp_checks??={};if(evidence.jobs[item.key+':evaluate']?.jobId){assert.equal(evidence.static_mcp_checks[item.key]?.status,'PASS');return;}
      assert.equal((await auditRows(sessionId)).length,0,'Training and static prepare must not execute MATLAB');evidence.static_mcp_checks[item.key]={status:'PASS',sessionId,actual_mcp_events:0,before_evaluation_admission:true};await save();};
    await checkStaticNoMatlab();
    const proposal=await stage('prepare-evaluation','prepare_imitation_evaluation',{policy:policy.artifact_id},{benchmark:registration.benchmark??{warmup:5,repeats:50,batch_size:1,threads:1}});
    assert.equal(proposal.status,'proposal');assert.equal(proposal.source_executed,false);assert.equal(proposal.config_suggestions.scenarios.length,12);
    assert.ok(proposal.config_suggestions.scenarios.every(row=>['validation','test'].includes(row.split)));await checkStaticNoMatlab();
    if(values['defer-evaluation']){
      evidence.deferred_cases??=[];
      const partial={key:item.key,workspace,sessionId,dataset,policy,proposal,offline,status:'TRAINED_AND_PORTABLE_VERIFIED_EVALUATION_NOT_RUN',reason:'Wait for concurrent regression/study processes to finish before the controller timing window'};
      const index=evidence.deferred_cases.findIndex(row=>row.key===item.key);if(index<0)evidence.deferred_cases.push(partial);else evidence.deferred_cases[index]=partial;
      await save();continue;
    }
    const evaluationConfig=proposal.config_suggestions,evaluationInputs={policy:policy.artifact_id};const evalRefs=await confirm('evaluate','evaluate_external_imitation',evaluationInputs,evaluationConfig);
    const activeBefore=(await json(resolve(workspace,'manifest.json'))).active;
    const evaluation=await stage('evaluate','evaluate_external_imitation',evaluationInputs,{...evaluationConfig,plan_refs:evalRefs});assert.equal(evaluation.kind,'policy_evaluation');assert.equal(evaluation.status,'verified');
    const changedEvaluation=structuredClone(evaluationConfig);changedEvaluation.benchmark.repeats+=1;await deny('changed-benchmark-old-confirmation','evaluate_external_imitation',evaluationInputs,{...changedEvaluation,plan_refs:evalRefs});
    assert.deepEqual((await json(resolve(workspace,'manifest.json'))).active,activeBefore,'Report-only comparison cannot replace the policy');
    const closedLoop=isolated(closedLoopAudit,[evaluation.artifact_path,workspace,f02File]);assert.equal(closedLoop.status,'PASS');
    const datasetMeta=await json(resolve(dataset.artifact_path,'artifact.json')),expertMeta=await json(resolve(workspace,'artifacts',datasetMeta.source.expert,'expert.json'));
    const matlab=expertMeta.adapter_spec.backend==='matlab_function',calls=auditCalls(await auditRows(sessionId),matlab),records=await events();assert.ok(!records.some(row=>row.event.type.startsWith('turn/')),'No numerical samples or decisions may use LLM turns');
    const manifest=await json(resolve(workspace,'manifest.json'));assert.ok(!Object.values(manifest.artifacts).some(row=>['controller','controller_design','imitation_policy','optimized_policy'].includes(row.kind)),'External imitation/evaluation must end without internal MPC or PGAC');
    assert.equal(hash(await readFile(resolve(source.path,'artifact.json'))),source.metadata_sha256);assert.equal(hash(await readFile(resolve(source.path,'expert.npz'))),source.expert_npz_sha256);
    const check={key:item.key,status:'PASS',workspace,sessionId,dataset,policy,evaluation,proposal,offline,closed_loop:closedLoop,actual_calls:calls,
      confirmation_source:'test_fixture',job_ids:['train','prepare-evaluation','evaluate'].map(key=>evidence.jobs[item.key+':'+key].jobId)};
    const index=evidence.checks.findIndex(row=>row.key===item.key);if(index<0)evidence.checks.push(check);else evidence.checks[index]=check;await save();
  }
  if(values['defer-evaluation']){assert.equal(evidence.deferred_cases.length,registration.cases.length);evidence.status='TRAINED_AWAITING_EVALUATION';}
  else{assert.equal(evidence.checks.length,registration.cases.length);assert.equal(evidence.rejections.length,4*registration.cases.length);evidence.status='PASS';evidence.finished_at=new Date().toISOString();}
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,checks:evidence.checks.map(row=>({key:row.key,policy:row.policy.artifact_id,evaluation:row.evaluation.artifact_id,projected_test_nrmse:row.offline.projected_test_nrmse,total_actual_mcp_calls:row.actual_calls.total_actual_mcp_calls})),confirmation_source:'test_fixture'}));}
