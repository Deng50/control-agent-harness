/** S10 native external-imitation cancel/resume gate; no Host/MATLAB lifecycle. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {cp,mkdir,readFile,readdir,stat} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{workspace:{type:'string'},'dataset-path':{type:'string'},evidence:{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-external-imitation-recovery.mjs --workspace <isolated S10/cr> --dataset-path <fresh actual CSV expert artifact> --evidence <json> [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values.workspace&&values['dataset-path']&&values.evidence);
const workspace=resolve(values.workspace),sourceRoot=resolve(values['dataset-path']),evidencePath=resolve(values.evidence),json=path=>readFile(path,'utf8').then(JSON.parse),sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const rel=relative(resolve(project,'test-output'),workspace);assert.ok(rel&&!isAbsolute(rel)&&rel!=='..'&&!rel.startsWith(`..${sep}`));
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const py=(code,args,isolated=false)=>execFileSync(python,[...(isolated?['-I']:[]),'-B','-c',code,...args],{encoding:'utf8',windowsHide:true,maxBuffer:16*1024*1024,
  env:{...process.env,PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',PYTHONPATH:isolated?'':resolve(project,'src')}});
const sourceMetaBytes=await readFile(resolve(sourceRoot,'artifact.json')),sourceMeta=JSON.parse(sourceMetaBytes),sourceSplit=await json(resolve(sourceRoot,'split.json')),sourcePhysical=await json(resolve(sourceRoot,'physical_context.json'));
assert.equal(sourceMeta.kind,'external_expert_dataset');assert.equal(sourceMeta.interface.input_dim,9);assert.equal(sourceMeta.interface.output_dim,2);assert.equal(sourcePhysical.interface,null);assert.deepEqual(sourceSplit.sample_counts,{train:800,validation:240,test:240});
const config={network:{hidden_dims:[16,8],activation:'tanh',learning_rate:.003,batch_size:64,epochs:300,patience:300,seed:20261129,weight_decay:.001,min_delta:1e-10,grad_clip:10}};
const preRegistered={source:{path:sourceRoot,id:sourceMeta.id,metadata_sha256:sha(sourceMetaBytes),expert_npz_sha256:sha(await readFile(resolve(sourceRoot,'expert.npz'))),split:sourceSplit,physical_context:sourcePhysical},
  config,scope:'Native lifecycle/structural gate only; no business accuracy gate and no replacement for the separate frozen F05 .2 acceptance.',cancel_after_committed_epoch_at_least:3,compare:'actual selected/last model, Adam state, RNG state, full epoch log and update count exact'};
let evidence;
if(values.resume){evidence=await json(evidencePath);assert.equal(evidence.workspace,workspace);assert.deepEqual(evidence.preRegistered,preRegistered);
  if(evidence.error){(evidence.observer_history??=[]).push({error:evidence.error,status:evidence.status,at:new Date().toISOString()});delete evidence.error;}evidence.status='RUNNING';}
else{try{await readFile(evidencePath);throw new Error('Evidence already exists; use --resume to observe saved handles');}catch(error){if(error.code!=='ENOENT')throw error;}
  evidence={schema_version:1,stage:'S10',group:'external_imitation_native_cancel_resume',status:'RUNNING',workspace,preRegistered,started_at:new Date().toISOString(),confirmation_source:'test_fixture',jobs:{},admissions:[],confirmations:[]};}
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const terminal=new Set(['completed','failed','killed']),pause=()=>new Promise(done=>setTimeout(done,100));
const checkpointInfo=String.raw`import hashlib,json,sys,torch
from pathlib import Path
from control_harness.locking import process_alive
p=Path(sys.argv[1]);c=torch.load(p,map_location='cpu',weights_only=True);s=json.loads(p.with_name('latest.checkpoint.json').read_text());intent=json.loads(p.with_name('stage.intent.json').read_text());actual=hashlib.sha256(p.read_bytes()).hexdigest()
assert actual==s['sha256'];assert s['stage_intent_sha256']==hashlib.sha256(p.with_name('stage.intent.json').read_bytes()).hexdigest()
print(json.dumps({'path':str(p),'sha256':actual,'epoch':c['epoch'],'updates':c['updates'],'wait':c['wait'],'algorithm':c['contract']['algorithm'],'training_signature':c['training_signature'],
 'checkpoint_kind':intent['kind'],'owner_pid':intent['owner']['pid'],'owner_alive':process_alive(intent['owner']['pid']),'sidecar':s,
 'optimizer_lr':[g['lr'] for g in c['optimizer_state']['param_groups']],'optimizer_steps':[float(v['step']) for v in c['optimizer_state']['state'].values()],
 'weight_shapes':{k:list(v.shape) for k,v in c['model_state'].items()},'epochs':len(c['log'])},allow_nan=False))`;
const compare=String.raw`import hashlib,json,sys,torch
from pathlib import Path
torch.set_num_threads(1)
a,b=map(Path,sys.argv[1:3]);left=torch.load(a/'latest.pt',weights_only=True,map_location='cpu');right=torch.load(b/'latest.pt',weights_only=True,map_location='cpu');checks={'tensors':0,'scalars':0}
def same(x,y,path):
 if isinstance(y,torch.Tensor):assert isinstance(x,torch.Tensor) and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y),path;checks['tensors']+=1
 elif isinstance(y,dict):
  assert isinstance(x,dict) and set(x)==set(y),path
  for k in y:same(x[k],y[k],path+'.'+str(k))
 elif isinstance(y,(list,tuple)):
  assert type(x)==type(y) and len(x)==len(y),path
  for k,(xx,yy) in enumerate(zip(x,y)):same(xx,yy,path+'.'+str(k))
 else:assert type(x)==type(y) and x==y,path;checks['scalars']+=1
same(left,right,'complete_checkpoint');same(torch.load(a/'model.pt',weights_only=True),torch.load(b/'model.pt',weights_only=True),'selected_model')
assert left['epoch']==300 and left['updates']==3900
print(json.dumps({'status':'PASS_EXACT','all_checkpoint_fields_equal':True,'selected_model_equal':True,'last_model_equal':True,'adam_equal':True,'rng_equal':True,'log_equal':True,
 'epoch':left['epoch'],'updates':left['updates'],'training_signature':left['training_signature'],'log_epochs':len(left['log']),'field_checks':checks,
 'resumed_latest_file_sha256':hashlib.sha256((a/'latest.pt').read_bytes()).hexdigest(),'continuous_latest_file_sha256':hashlib.sha256((b/'latest.pt').read_bytes()).hexdigest()},allow_nan=False))`;
const bundleInput=String.raw`import json,sys,numpy as np
from pathlib import Path
from control_harness.external_policy import load_external_policy
torch=__import__('torch');torch.set_num_threads(1)
policy=load_external_policy(sys.argv[1]);data=Path(sys.argv[2]);out=Path(sys.argv[3])
with np.load(data/'expert.npz',allow_pickle=False) as a:x=a['X'][:11].copy();previous=a['actual_previous_u'][:11].copy()
out.write_text(json.dumps({'features':x.tolist(),'previous_action':previous.tolist(),'feature_labels':policy.layout['feature_labels']})+'\n')
print(json.dumps({'raw':policy.predict_raw(x).tolist(),'projected':policy.predict(x,previous).tolist()},allow_nan=False))`;
const portable=String.raw`import importlib.abc,importlib.util,json,sys
from pathlib import Path
import torch
torch.set_num_threads(1)
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('Project imports forbidden in independent recovery bundle')
sys.meta_path.insert(0,Block());root=Path(sys.argv[1]);loader=importlib.util.spec_from_file_location('independent_recovery_policy',root/'policy_runtime.py');module=importlib.util.module_from_spec(loader);sys.modules[loader.name]=module;loader.loader.exec_module(module)
policy=module.load_external_policy(root);x=json.loads(Path(sys.argv[2]).read_text());raw=policy.predict_raw(x['features']);projected=policy.predict(x['features'],x['previous_action']);expected=json.loads(Path(sys.argv[3]).read_text())
np=__import__('numpy');np.testing.assert_array_equal(raw,np.asarray(expected['raw']));np.testing.assert_array_equal(projected,np.asarray(expected['projected']))
assert not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)
print(json.dumps({'status':'PASS','rows':len(raw),'max_raw_error':float(np.max(np.abs(raw-expected['raw']))),'max_projected_error':float(np.max(np.abs(projected-expected['projected']))),'project_imports':False,'original_source_imports':False},allow_nan=False))`;

try{
  await prepareSmokeWorkspace(workspace);
  const markerPath=resolve(workspace,'.case-plans/test-fixture.json'),markerSource={kind:'S10_actual_native_external_imitation_recovery',source_dataset:sourceMeta.id,source_metadata_sha256:preRegistered.source.metadata_sha256};
  let marker;try{marker=await json(markerPath);}catch(error){if(error.code!=='ENOENT')throw error;assert.ok(!evidence.sessionId&&!evidence.admissions.length,'Never relabel a previously admitted Session');
    marker={kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true,source:markerSource};await writeSmokeEvidence(markerPath,marker);}
  assert.equal(marker.kind,'control-case-plan-test-fixture-v1');assert.equal(marker.workspace,workspace);assert.equal(marker.confirmation_source,'test_fixture');assert.deepEqual(marker.source,markerSource);
  const markerRecord={path:markerPath,sha256:sha(await readFile(markerPath))};if(evidence.fixture_marker)assert.deepEqual(evidence.fixture_marker,markerRecord);else evidence.fixture_marker=markerRecord;await save();
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  if(!evidence.sessionId){const s=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});assert.equal(s.agentPreset,'control-lightweight');evidence.sessionId=s.sessionId;await save();}
  const sessionId=evidence.sessionId;
  const status=async entry=>(await command(rpc,sessionId,`/control-status ${JSON.stringify({workspace,attemptId:entry.attemptId})}`)).record;
  const admit=async(key,line)=>{const request_sha256=sha(line);let entry=evidence.jobs[key];
    if(entry){assert.equal(entry.request_sha256,request_sha256);assert.ok(entry.jobId,'Uncertain admission; reconcile saved original handle, never re-submit');return entry;}
    entry=evidence.jobs[key]={request_sha256,status:'ADMISSION_PENDING'};await save();const started=await command(rpc,sessionId,line);
    Object.assign(entry,{jobId:started.jobId,attemptId:started.attemptId,recoveredFrom:started.recoveredFrom??null,status:'RUNNING'});assert.ok(entry.jobId&&entry.attemptId);evidence.admissions.push({key,...started});await save();return entry;};
  const wait=async(entry,expected='completed')=>{for(;;){const record=await status(entry);if(terminal.has(record.state)){entry.status=record.state;entry.record=record;await save();assert.equal(record.state,expected,JSON.stringify(record));return record;}
      assert.notEqual(record.state,'interrupted','Reconcile saved native handle; do not submit another worker');await pause();}};
  const operation=async(key,name,inputs={},settings={})=>{const entry=await admit(key,`/control ${JSON.stringify({agent:'lightweight',operation:name,workspace,inputs,config:settings})}`);return (await wait(entry)).result;};
  const events=async(label)=>{
    const pages=[];let beforeSeq,throughSeq;const records=new Map();
    for(;;){const page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:500,...(beforeSeq===undefined?{}:{beforeSeq,throughSeq})}});if(throughSeq===undefined)throughSeq=page.cursor;
      pages.push(page);for(const record of page.records)records.set(record.event.seq,record);if(!page.hasMore)break;
      assert.ok(page.records.length,'A truncated page must advance');const oldest=Math.min(...page.records.map(x=>x.event.seq));assert.ok(beforeSeq===undefined||oldest<beforeSeq);beforeSeq=oldest;}
    const value={sessionId,throughSeq,pages,records:[...records.values()].sort((a,b)=>a.event.seq-b.event.seq),all_pages_complete:true};
    const path=resolve(dirname(evidencePath),`external-imitation-recovery-${label}-session-events.json`);await writeSmokeEvidence(path,value);
    (evidence.session_events??={})[label]={path,sha256:sha(await readFile(path)),pages:pages.length,records:value.records.length,hasMore:false,throughSeq};await save();return value.records;};
  const imported=await operation('import','import_artifact',{artifact_path:sourceRoot});assert.equal(imported.kind,'external_expert_dataset');assert.equal(imported.artifact_id,sourceMeta.id);
  evidence.dataset_id=imported.artifact_id;evidence.dataset_path=imported.artifact_path;await save();
  if(!evidence.plan_refs){const requirements=await confirmationRequirements({operation:'train_external_imitation',workspace,inputs:{dataset:imported.artifact_id},config});assert.deepEqual(requirements.map(x=>x.scope),['signals','training']);const refs={};
    for(const row of requirements){const p=await operation('plan-'+row.scope,'propose_case_plan',{}, {scope:row.scope,payload:row.expectedPayload,bindings:row.bindings,dependencies:structuredClone(refs)});refs[row.scope]={plan_id:p.plan_id,version_hash:p.version_hash};
      const confirmed=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace,...refs[row.scope]})}`);assert.equal(confirmed.confirmation_source,'test_fixture');evidence.confirmations.push({scope:row.scope,...refs[row.scope],source:confirmed.source,confirmation_source:confirmed.confirmation_source});await save();}
    evidence.plan_refs=refs;await save();}
  const request={agent:'lightweight',operation:'train_external_imitation',workspace,inputs:{dataset:imported.artifact_id},config:{...config,plan_refs:evidence.plan_refs}},line=`/control ${JSON.stringify(request)}`;
  const original=await admit('cancelled-training',line);
  if(!evidence.cancel_requested){for(;;){const record=await status(original);assert.ok(!terminal.has(record.state),'Training finished before a real >=3-epoch cancellation; no automatic replacement experiment');assert.notEqual(record.state,'interrupted');
      if((record.progress?.epoch??0)>=3){evidence.observed_before_cancel={state:record.state,progress:record.progress,at:new Date().toISOString()};evidence.cancel_requested={status:'ADMISSION_PENDING',jobId:original.jobId,method:'job/kill'};await save();
        const receipt=await rpc('job/kill',{request:{sessionId,jobId:original.jobId}});assert.equal(receipt.outcome,'requested');Object.assign(evidence.cancel_requested,{status:'requested',receipt});await save();break;}await pause();}}
  const killed=await wait(original,'killed');assert.equal(killed.outcome.cleanupVerified,true);assert.ok(!killed.recoveryBlocked);assert.ok(killed.checkpoint);
  const checkpoint=resolve(killed.checkpoint);assert.ok(relative(workspace,checkpoint).startsWith('failed'+sep));
  const before=JSON.parse(py(checkpointInfo,[checkpoint]));assert.equal(before.owner_alive,false);assert.equal(before.checkpoint_kind,'supervised_model');assert.equal(before.algorithm,'supervised_training');assert.ok(before.epoch>=3&&before.epoch<300);
  assert.deepEqual(before.optimizer_lr,[.003]);assert.ok(before.optimizer_steps.every(x=>x===before.updates));
  const attemptPath=resolve(workspace,'.control-jobs',original.attemptId),requestFile=resolve(attemptPath,'request.json'),requestHash=sha(await readFile(requestFile));
  const ledger=[];for(const file of (await readdir(attemptPath)).filter(x=>/^\d{12}\.json$/.test(x)).sort())ledger.push(await json(resolve(attemptPath,file)));
  const started=ledger.find(x=>x.workerEvent?.event==='started')?.workerEvent;assert.ok(started?.pid);assert.equal(started.pid,before.owner_pid);
  try{await stat(resolve(workspace,'.operation.lock'));assert.fail('Workspace operation lock remained after OS-confirmed cleanup');}catch(error){if(error.code!=='ENOENT')throw error;}
  evidence.cancelled={native_job:original.jobId,attemptId:original.attemptId,worker_pid:started.pid,worker_os_alive:before.owner_alive,cleanup_verified:true,workspace_lock_released:true,checkpoint:before};await save();
  const beforeRecoveryEvents=await events('before-resume'),confirmationCount=beforeRecoveryEvents.filter(x=>x.event.type==='command/run'&&x.event.data.name==='control-confirm').length;assert.equal(confirmationCount,2);
  const recovered=await admit('resumed-training',`/control-recover ${JSON.stringify({workspace,attemptId:original.attemptId,mode:'resume'})}`);assert.notEqual(recovered.jobId,original.jobId);assert.equal(recovered.recoveredFrom,original.attemptId);
  const completed=await wait(recovered);assert.equal(completed.recoveryMode,'resume');assert.equal(completed.recoveredFrom,original.attemptId);assert.deepEqual(completed.request.config.plan_refs,evidence.plan_refs);assert.deepEqual(completed.request.config.network,config.network);assert.equal(resolve(completed.request.config.resume_checkpoint),checkpoint);
  const recoveredResult=completed.result;assert.equal(recoveredResult.kind,'external_imitation_policy');assert.equal(recoveredResult.status,'verified');const resumedReport=await json(resolve(recoveredResult.artifact_path,'report.json'));
  assert.equal(resumedReport.network.resumed_from_epoch,before.epoch);assert.equal(resumedReport.network.trained_epochs,300);assert.equal(resumedReport.acceptance,'not_requested');assert.equal(resumedReport.closed_loop,'not_run');assert.equal(resumedReport.pgac,'not_run');
  const continuous=await admit('continuous-baseline',line),baseline=await wait(continuous),baselineResult=baseline.result;assert.equal(baselineResult.kind,'external_imitation_policy');assert.equal(baselineResult.status,'verified');
  const baselineReport=await json(resolve(baselineResult.artifact_path,'report.json'));assert.equal(baselineReport.network.resumed_from_epoch,0);assert.equal(baselineReport.supervised_dataset,resumedReport.supervised_dataset);
  const manifest=await json(resolve(workspace,'manifest.json')),resumedNN=resolve(workspace,manifest.artifacts[resumedReport.supervised_model].path),baselineNN=resolve(workspace,manifest.artifacts[baselineReport.supervised_model].path);
  evidence.exact_comparison=JSON.parse(py(compare,[resumedNN,baselineNN]));assert.equal(evidence.exact_comparison.status,'PASS_EXACT');assert.equal(sha(await readFile(checkpoint)),before.sha256);assert.equal(sha(await readFile(requestFile)),requestHash);
  evidence.resumed={result:recoveredResult,supervised_model:resumedReport.supervised_model,checkpoint:JSON.parse(py(checkpointInfo,[resolve(resumedNN,'latest.pt')]))};
  evidence.continuous={result:baselineResult,supervised_model:baselineReport.supervised_model,checkpoint:JSON.parse(py(checkpointInfo,[resolve(baselineNN,'latest.pt')]))};await save();
  const proposal=await operation('prepare-data-only','prepare_imitation_evaluation',{policy:baselineResult.artifact_id});assert.equal(proposal.status,'not_applicable');assert.equal(proposal.source_executed,false);assert.equal(proposal.config_suggestions,null);evidence.prepare_imitation_evaluation=proposal;
  const input=resolve(workspace,'portable-input.json'),expected=resolve(workspace,'portable-expected.json'),internal=JSON.parse(py(bundleInput,[baselineResult.artifact_path,imported.artifact_path,input]));await writeSmokeEvidence(expected,internal);
  const portableRoot=resolve(project,'..','control-agent-harness-s10-recovery-portable',sha(workspace).slice(0,10));assert.ok(relative(project,portableRoot).startsWith('..'+sep));
  try{await stat(portableRoot);}catch(error){if(error.code!=='ENOENT')throw error;await mkdir(dirname(portableRoot),{recursive:true});await cp(resolve(baselineResult.artifact_path,'bundle'),portableRoot,{recursive:true,errorOnExist:true,force:false});}
  const portableManifest=await json(resolve(portableRoot,'policy_manifest.json'));for(const [file,digest]of Object.entries(portableManifest.files))assert.equal(sha(await readFile(resolve(portableRoot,file))),digest);
  evidence.portable={root:portableRoot,...JSON.parse(py(portable,[portableRoot,input,expected],true))};assert.equal(evidence.portable.status,'PASS');
  const finalEvents=await events('final');assert.equal(finalEvents.filter(x=>x.event.type==='command/run'&&x.event.data.name==='control-confirm').length,confirmationCount,'Compatible resume/baseline unexpectedly reconfirmed');
  assert.ok(!finalEvents.some(x=>x.event.type.startsWith('turn/')),'Fixed lifecycle fixture must not open a model turn');assert.ok(!finalEvents.some(x=>x.event.type==='tool/call'&&JSON.stringify(x.event.data).includes('mcp__matlab__')));
  const last=(await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace})}`)).attempts;assert.ok(last.every(x=>terminal.has(x.state)));
  evidence.native_terminal=last.map(x=>({attemptId:x.attemptId,jobId:x.nativeJobId,state:x.state,cleanup_verified:x.outcome?.cleanupVerified,operation:x.request.operation}));
  const workerPids=[];for(const attempt of last){const directory=resolve(workspace,'.control-jobs',attempt.attemptId);for(const name of (await readdir(directory)).filter(x=>/^\d{12}\.json$/.test(x))){const event=await json(resolve(directory,name));if(event.workerEvent?.event==='started')workerPids.push(event.workerEvent.pid);}}
  evidence.worker_terminal=JSON.parse(py("import json,sys\nfrom control_harness.locking import process_alive\nprint(json.dumps([{'pid':p,'alive':process_alive(p)} for p in json.loads(sys.argv[1])]))",[JSON.stringify([...new Set(workerPids)])]));assert.ok(evidence.worker_terminal.every(x=>x.alive===false));
  assert.ok(!Object.values(manifest.artifacts).some(x=>['controller','controller_design','imitation_policy','optimized_policy'].includes(x.kind)));
  evidence.source_hashes={};for(const file of ['src/control_harness/external_imitation.py','src/control_harness/supervised.py','src/control_harness/checkpoints.py','src/control_harness/external_policy.py','src/control_harness/policy_export.py','src/control_harness/case_requirements.py','harness-plugin/src/case-requirements.mjs','harness-plugin/src/index.mjs','harness-plugin/src/job-history.mjs','harness-plugin/src/worker.mjs','scripts/smoke-external-imitation-recovery.mjs'])evidence.source_hashes[file]=sha(await readFile(resolve(project,file)));
  evidence.recovery={old_job:original.jobId,new_job:recovered.jobId,old_attempt:original.attemptId,new_attempt:recovered.attemptId,new_confirmations:0,same_plan_refs:true,old_checkpoint_unchanged:true,original_request_unchanged:true};
  evidence.no_matlab_calls=true;evidence.no_mpc_or_pgac=true;evidence.status='PASS';evidence.error=null;evidence.finished_at=new Date().toISOString();
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,sessionId:evidence.sessionId,jobs:Object.fromEntries(Object.entries(evidence.jobs).map(([key,value])=>[key,{jobId:value.jobId,attemptId:value.attemptId,status:value.status}])),workers:evidence.worker_terminal??[],confirmation_source:'test_fixture'}));}
