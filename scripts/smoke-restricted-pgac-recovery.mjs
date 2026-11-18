/** S12 real native cancellation -> exact raw-policy checkpoint continuation. */
import assert from 'node:assert/strict';
import {readFile,readdir} from 'node:fs/promises';
import {resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {fileURLToPath} from 'node:url';
import {command,connectHarness,project,writeSmokeEvidence} from './harness-rpc.mjs';
import {artifact,assertAudit,checkpointInspect,freezeAudit,hash,json,observer,openEvidence,preflight,runPython,terminalWorkers} from './pgac-native-observer.mjs';

const {values}=parseArgs({options:{registration:{type:'string'},'registration-sha256':{type:'string'},'validation-evidence':{type:'string'},
  'validation-proof':{type:'string'},'validation-proof-sha256':{type:'string'},'gradient-probes':{type:'string'},'gradient-probes-sha256':{type:'string'},
  'gradient-oracle':{type:'string'},'gradient-oracle-sha256':{type:'string'},evidence:{type:'string'},'mcp-audit':{type:'string'},'web-log':{type:'string'},
  resume:{type:'boolean'},'validate-only':{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-restricted-pgac-recovery.mjs --registration <R03 JSON> --registration-sha256 <SHA> --validation-evidence <R01 PASS> --validation-proof <independent JSON> --validation-proof-sha256 <SHA> --gradient-probes <TRAIN points JSON> --gradient-probes-sha256 <SHA> --gradient-oracle <post-oracle.py> --gradient-oracle-sha256 <SHA> --evidence <new JSON> --mcp-audit <actual observer JSONL> [--web-log <Host log>] [--resume] [--validate-only]');
  console.log('Uses a separate fresh Python-only native Session in the preserved R01 workspace. Cancels only its own recorded job after committed epoch progress; continues exact target20 and compares all checkpoint fields with an uninterrupted target20.');process.exit(0);}
assert.ok(values.evidence&&values['mcp-audit']);const evidencePath=resolve(values.evidence),auditPath=resolve(values['mcp-audit']);
const prepared=await preflight(values,[fileURLToPath(import.meta.url),resolve(project,'scripts/pgac-native-observer.mjs')]);prepared.pinned.mcp_audit_path=auditPath;
if(values['validate-only']){console.log(JSON.stringify({status:'PASS_PREFLIGHT_ONLY',case:prepared.registration.recovery.case,native_submitted:false}));process.exit(0);}
const e=await openEvidence(evidencePath,values,prepared,'S12-R03-native-recovery'),save=()=>writeSmokeEvidence(evidencePath,e);
const exactCompare=String.raw`import hashlib,json,sys,torch
from pathlib import Path
torch.set_num_threads(1);a,b=map(Path,sys.argv[1:3]);left=torch.load(a/'latest.pt',weights_only=True,map_location='cpu');right=torch.load(b/'latest.pt',weights_only=True,map_location='cpu');checks={'tensors':0,'scalars':0}
def same(x,y,path):
 if isinstance(y,torch.Tensor):assert isinstance(x,torch.Tensor) and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y),path;checks['tensors']+=1
 elif isinstance(y,dict):
  assert isinstance(x,dict) and set(x)==set(y),path
  for k in y:same(x[k],y[k],path+'.'+str(k))
 elif isinstance(y,(list,tuple)):
  assert type(x)==type(y) and len(x)==len(y),path
  for i,(xx,yy) in enumerate(zip(x,y)):same(xx,yy,path+'.'+str(i))
 else:assert type(x)==type(y) and x==y,path;checks['scalars']+=1
same(left,right,'complete_checkpoint');same(torch.load(a/'model.pt',weights_only=True),torch.load(b/'model.pt',weights_only=True),'selected_raw_network')
assert left['epoch']==20 and left['updates']==400 and left['contract']['algorithm']=='external_physical_policy_optimization'
print(json.dumps({'status':'PASS_EXACT','all_checkpoint_fields_equal':True,'selected_model_equal':True,'last_model_equal':True,'Adam_equal':True,'RNG_equal':True,'log_equal':True,
 'epoch':left['epoch'],'updates':left['updates'],'training_signature':left['training_signature'],'field_checks':checks,
 'resumed_checkpoint_sha256':hashlib.sha256((a/'latest.pt').read_bytes()).hexdigest(),'continuous_checkpoint_sha256':hashlib.sha256((b/'latest.pt').read_bytes()).hexdigest()},allow_nan=False))`;
try{
  const item=prepared.cases.find(row=>row.key===prepared.registration.recovery.case);assert.ok(item&&item.key==='f05-python');
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true}),o=observer(rpc,e,evidencePath,item,{sessionKey:'recovery-python'});
  const cfg=structuredClone(prepared.registration.recovery.config),inputs={policy:item.policy_id,model:item.actual.candidate,validation:item.actual.validation};assert.equal(cfg.epochs,20);assert.equal(cfg.seed,20261127);
  const refs=await o.phaseRefs('lifecycle','cancelled-training','optimize_external_policy',inputs,cfg,['signals','training']);
  const original=await o.admit('cancelled-training','optimize_external_policy',inputs,{...cfg,plan_refs:refs},{expected:'killed'});
  // An observer interruption cannot silently substitute another experiment or
  // issue a second kill to an unrelated job. An uncertain kill stops for review.
  if(e.cancel_requested){assert.equal(e.cancel_requested.jobId,original.jobId);assert.equal(e.cancel_requested.status,'requested','Uncertain own-job kill must be reconciled, never automatically repeated');}
  else for(;;){const state=await o.status(original),record=o.assertRecord(original,state);assert.ok(!['completed','failed','killed','interrupted'].includes(record.state),'Training already terminal before an actual epoch cancellation; do not replace the experiment');
    if(record.progress?.stage==='restricted_pgac_training'&&(record.progress.epoch??0)>=1){e.observed_before_cancel={state:record.state,progress:record.progress,at:new Date().toISOString()};e.cancel_requested={status:'ADMISSION_PENDING',jobId:original.jobId,attemptId:original.attemptId,method:'job/kill'};await save();
      const receipt=await rpc('job/kill',{request:{sessionId:await o.session(),jobId:original.jobId}});assert.equal(receipt.outcome,'requested');Object.assign(e.cancel_requested,{status:'requested',receipt});await save();break;}
    await new Promise(done=>setTimeout(done,100));}
  await o.wait(original,'killed');const killed=original.record;assert.equal(killed.outcome.cleanupVerified,true);assert.ok(!killed.recoveryBlocked);assert.ok(killed.checkpoint);
  const checkpointPath=resolve(killed.checkpoint);assert.ok(checkpointPath.startsWith(resolve(item.workspace,'failed')+sep));
  const info=runPython(checkpointInspect,[checkpointPath]);assert.equal(info.checkpoint_kind,'external_pgac_candidate');assert.equal(info.algorithm,'external_physical_policy_optimization');assert.equal(info.owner_alive,false);assert.ok(info.epoch>=1&&info.epoch<20);assert.deepEqual(info.learning_rates,[.0003]);assert.equal(info.updates,info.epoch*20);
  const attemptPath=resolve(item.workspace,'.control-jobs',original.attemptId),requestPath=resolve(attemptPath,'request.json'),oldRequestSha=hash(await readFile(requestPath));
  const ledger=[];for(const file of (await readdir(attemptPath)).filter(file=>/^\d{12}\.json$/.test(file)).sort())ledger.push(await json(resolve(attemptPath,file)));
  const started=ledger.find(row=>row.workerEvent?.event==='started')?.workerEvent;assert.equal(started?.pid,info.owner_pid);
  const cancelled={nativeJobId:original.jobId,attemptId:original.attemptId,worker_pid:started.pid,worker_os_alive:false,cleanupVerified:true,checkpoint:info,original_request_sha256:oldRequestSha};
  if(e.cancelled)assert.deepEqual(e.cancelled,cancelled);else e.cancelled=cancelled;await save();
  const before=await o.events(),confirmationCount=before.filter(row=>row.event.type==='command/run'&&row.event.data.name==='control-confirm').length;assert.equal(confirmationCount,2);
  const recoverArgs={workspace:item.workspace,attemptId:original.attemptId,mode:'resume'},line='/control-recover '+JSON.stringify(recoverArgs),recoverKey=o.key('resumed-training');
  const recoveredRequest={agent:'lightweight',operation:'optimize_external_policy',workspace:item.workspace,inputs,config:{...cfg,plan_refs:refs,resume_checkpoint:checkpointPath}};
  let recovered=e.jobs[recoverKey];
  if(recovered){assert.equal(recovered.command_sha256,hash(line));assert.deepEqual(recovered.request,recoveredRequest);assert.ok(recovered.jobId&&recovered.attemptId,'Uncertain recovery admission must be reconciled; never resend');}
  else{recovered=e.jobs[recoverKey]={key:recoverKey,operation:'optimize_external_policy',owner:'lightweight',sessionId:await o.session(),request:recoveredRequest,signature:hash(JSON.stringify(recoveredRequest)),command_sha256:hash(line),expected:'completed',status:'ADMISSION_PENDING'};await save();
    const receipt=await command(rpc,await o.session(),line);assert.ok(receipt.jobId&&receipt.attemptId);assert.equal(receipt.recoveredFrom,original.attemptId);assert.notEqual(receipt.jobId,original.jobId);
    Object.assign(recovered,{jobId:receipt.jobId,attemptId:receipt.attemptId,status:'RUNNING',recoveredFrom:receipt.recoveredFrom});e.admissions.push({key:recoverKey,caseKey:item.key,sessionKey:o.sessionKey,operation:'optimize_external_policy',owner:'lightweight',sessionId:recovered.sessionId,jobId:recovered.jobId,attemptId:recovered.attemptId,signature:recovered.signature,command_sha256:recovered.command_sha256,recoveredFrom:original.attemptId});await save();}
  const resumed=await o.wait(recovered);assert.equal(recovered.record.recoveryMode,'resume');assert.equal(recovered.record.recoveredFrom,original.attemptId);assert.deepEqual(recovered.record.request.config.plan_refs,refs);assert.equal(resumed.kind,'external_pgac_candidate');assert.equal(resumed.status,'verified');
  const resumedArtifact=await artifact(item.workspace,resumed.artifact_id),rr=await resumedArtifact.member('report.json');assert.equal(rr.resumed_from_epoch,info.epoch);assert.equal(rr.trained_epochs,20);assert.equal(rr.optimizer_updates,400);assert.equal(resumedArtifact.metadata.activation,'none');
  const continuous=await o.stage('continuous-training','optimize_external_policy',inputs,{...cfg,plan_refs:refs});assert.equal(continuous.status,'verified');const continuousArtifact=await artifact(item.workspace,continuous.artifact_id),cr=await continuousArtifact.member('report.json');assert.equal(cr.resumed_from_epoch,0);assert.equal(cr.trained_epochs,20);assert.equal(continuousArtifact.metadata.activation,'none');
  const compared=runPython(exactCompare,[resumedArtifact.root,continuousArtifact.root]);assert.equal(compared.status,'PASS_EXACT');assert.equal(hash(await readFile(checkpointPath)),info.sha256);assert.equal(hash(await readFile(requestPath)),oldRequestSha);
  const baseline=await artifact(item.workspace,item.policy_id);assert.equal(hash(await baseline.memberBytes('bundle/model.pt')),item.baseline_weights_sha256);
  await o.phaseRefs('lifecycle','cancelled-training','optimize_external_policy',inputs,cfg,['signals','training']);
  const finalEvents=await o.events();assert.equal(finalEvents.filter(row=>row.event.type==='command/run'&&row.event.data.name==='control-confirm').length,confirmationCount,'Compatible continuation/baseline re-confirmed unchanged scopes');assert.ok(!finalEvents.some(row=>row.event.type.startsWith('turn/')));
  e.exact_comparison=compared;e.resumed={artifact_id:resumed.artifact_id,checkpoint:runPython(checkpointInspect,[resolve(resumedArtifact.root,'latest.pt')])};e.continuous={artifact_id:continuous.artifact_id,checkpoint:runPython(checkpointInspect,[resolve(continuousArtifact.root,'latest.pt')])};
  e.recovery={old_job:original.jobId,new_job:recovered.jobId,old_attempt:original.attemptId,new_attempt:recovered.attemptId,new_confirmations:0,same_target_epochs:20,same_plan_refs:true,old_checkpoint_unchanged:true,original_request_unchanged:true,baseline_unchanged:true};
  e.terminal=await terminalWorkers(o,e);e.actual_mcp=await assertAudit(auditPath,await o.session(),false);await freezeAudit(auditPath,e,evidencePath,'R03-recovery');
  for(const [name,digest] of Object.entries(prepared.pinned.source_sha256))assert.equal(hash(await readFile(resolve(project,name))),digest);
  e.status='PASS';e.native_result='PASS';e.error=null;e.finished_at=new Date().toISOString();await save();console.log(JSON.stringify({status:e.status,evidence:evidencePath,jobs:e.admissions.length,actual_MATLAB_calls:0,exact_checkpoint_comparison:true,new_confirmations_on_resume:0}));
}catch(error){e.status='FAIL';e.error={name:error.name,message:error.message,stack:error.stack};e.finished_at=new Date().toISOString();await save();throw error;}
