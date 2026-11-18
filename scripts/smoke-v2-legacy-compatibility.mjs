/** E01: the full public linear/nonlinear legacy workflows through native Harness jobs. */
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {readFile,readdir,mkdir,realpath,writeFile} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{workspace:{type:'string'},evidence:{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},'prepare-only':{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-v2-legacy-compatibility.mjs --workspace <fresh project test-output root> --evidence <JSON> --web-log <existing normal Host log> [--prepare-only] [--resume]');
  console.log('Copies all three original public example files per case unchanged, prepares test-fixture markers before canonical native Workspace/Session creation, reviews/explicitly confirms the actual three scopes, runs all eight original stages and reuses the completed run. No LLM/MATLAB/Host startup. An uncertain admission is never retried.');process.exit(0);}
assert.ok(values.workspace&&values.evidence,'Explicit fresh workspace root and evidence file are required');
const inside=(base,target)=>{const r=relative(base,target);return Boolean(r)&&!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep);};
const root=resolve(values.workspace),evidencePath=resolve(values.evidence),registrationPath=resolve(dirname(evidencePath),'legacy-compatibility-registration-v1.json');
assert.ok(inside(resolve(project,'test-output'),root),'Use a project test-output descendant; never an old artifact workspace');
const json=p=>readFile(p,'utf8').then(JSON.parse),hash=b=>createHash('sha256').update(b).digest('hex');
const normalize=v=>{if(typeof v==='string')v=JSON.parse(v);assert.ok(v&&typeof v==='object'&&!Array.isArray(v));return v;};
const sleep=ms=>new Promise(done=>setTimeout(done,ms));
const operations=['register_model','register_model','simulate_transitions','train_residual','design_controller','collect_expert','train_imitation','optimize_policy'];
const driverSHA=hash(await readFile(new URL(import.meta.url)));
let evidence;
try{evidence=await json(evidencePath);assert.ok(values.resume||values['prepare-only'],'Existing evidence must be observed with --resume, never silently replaced');}
catch(error){if(error.code!=='ENOENT')throw error;assert.ok(!values.resume,'No original evidence exists to resume');}

if(!evidence){
  await prepareSmokeWorkspace(root);
  const registration={schema_version:1,phase:'S14-E01-legacy-compatibility',created_at:new Date().toISOString(),driver_sha256:driverSHA,confirmation_source:'test_fixture',synthetic:true,
    source:'Unmodified public examples/linear and examples/nonlinear: all original eight operations and their numerical settings',cases:[],component_sha256:{},limitations:['Demonstration simulation workflows, not original hardware or MATLAB evaluation.','No LLM conversation or new installation claim.']};
  for(const name of ['src/control_harness/workflow.py','src/control_harness/examples.py','src/control_harness/operations.py','src/control_harness/control.py','src/control_harness/policy.py','src/control_harness/residual.py','src/control_harness/case_requirements.py','harness-plugin/src/index.mjs','harness-plugin/src/case-plans.mjs','harness-plugin/src/case-requirements.mjs','harness-plugin/src/review.mjs','harness-plugin/src/session-workspace.mjs','scripts/harness-rpc.mjs'])registration.component_sha256[name]=hash(await readFile(resolve(project,name)));
  for(const key of ['linear','nonlinear']){
    const workspace=resolve(root,key==='linear'?'l':'n');await prepareSmokeWorkspace(workspace);assert.equal(await realpath(workspace),workspace,'Workspace spelling must be canonical before any native Session');
    for(const name of ['manifest.json','.control-jobs']){try{await readFile(resolve(workspace,name));assert.fail('Fresh compatibility workspace required');}catch(error){if(!['ENOENT','EISDIR'].includes(error.code))throw error;if(error.code==='EISDIR')assert.fail('Existing job directory must be investigated, never reused as fresh');}}
    await mkdir(resolve(workspace,'fixtures'),{recursive:true});await mkdir(resolve(workspace,'.case-plans'),{recursive:true});
    const files=[];
    for(const name of ['nominal.json','reference.json','workflow.json']){
      const source=resolve(project,'examples',key,name),destination=resolve(workspace,'fixtures',name),raw=await readFile(source);await writeFile(destination,raw,{flag:'wx'});files.push({name,source,path:destination,bytes:raw.length,sha256:hash(raw)});
    }
    const definition=await json(resolve(workspace,'fixtures/workflow.json'));assert.deepEqual(definition.steps.map(x=>x.operation),operations,'Do not drop any original legacy stage');
    const markerPath=resolve(workspace,'.case-plans/test-fixture.json');await writeFile(markerPath,JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true,source:'S14/E01 full original public '+key+' workflow'})+'\n',{flag:'wx'});
    registration.cases.push({key,workspace,workflow_path:resolve(workspace,'fixtures/workflow.json'),files,definition,marker:{path:markerPath,sha256:hash(await readFile(markerPath))}});
  }
  await writeFile(registrationPath,JSON.stringify(registration,null,2)+'\n',{flag:'wx'});
  evidence={schema_version:1,phase:registration.phase,status:'PREPARED_NOT_RUN',confirmation_source:'test_fixture',synthetic:true,started_at:new Date().toISOString(),
    preRegistered:{path:registrationPath,sha256:hash(await readFile(registrationPath)),driver_sha256:driverSHA},sessions:{},native_workspaces:{},session_intents:{},jobs:{},admissions:[],reviews:{},phase_refs:{},confirmations:[],rejections:[],checks:[],session_events:{},
    actions:{LLM:0,MATLAB:0,Host_start_stop:0,algorithm_changes:0,test_case_or_threshold_reduction:0},scope:'Real native jobs for original linear/nonlinear public workflows and completed reuse; test_fixture confirmations are not production-user authorization.'};
  await writeSmokeEvidence(evidencePath,evidence);
}
const registration=await json(registrationPath);assert.equal(hash(await readFile(registrationPath)),evidence.preRegistered.sha256);assert.equal(driverSHA,evidence.preRegistered.driver_sha256);
for(const [name,digest]of Object.entries(registration.component_sha256))assert.equal(hash(await readFile(resolve(project,name))),digest,'Registered component changed: '+name);
for(const item of registration.cases){assert.deepEqual(item.definition.steps.map(x=>x.operation),operations);for(const file of item.files){assert.equal(hash(await readFile(file.source)),file.sha256);assert.equal(hash(await readFile(file.path)),file.sha256);}assert.equal(hash(await readFile(item.marker.path)),item.marker.sha256);}
if(values['prepare-only']){console.log(JSON.stringify({status:'PREPARED_NOT_RUN',registration:registrationPath,registration_sha256:evidence.preRegistered.sha256,evidence:evidencePath,cases:registration.cases.map(x=>x.key),native_calls:0}));process.exit(0);}
if(evidence.error){evidence.observer_history??=[];evidence.observer_history.push({status:evidence.status,error:evidence.error,observed_at:new Date().toISOString()});delete evidence.error;}
evidence.status='RUNNING';const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();

try{
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  evidence.authentication={helper:'connectHarness',private_startup_log_read_internally:true,token_or_cookie_in_evidence:false,browser_token_injection:false};await save();
  for(const item of registration.cases){
    const workspace=item.workspace,key=part=>item.key+'/'+part;
    if(!evidence.native_workspaces[item.key]){assert.ok(!evidence.session_intents[key('workspace')],'Uncertain Workspace admission must be investigated first');evidence.session_intents[key('workspace')]={at:new Date().toISOString(),path:workspace};await save();
      const native=await rpc('workspace/create',{request:{path:workspace}});assert.equal(native.workspace.path,workspace);evidence.native_workspaces[item.key]=native;await save();}
    if(!evidence.sessions[item.key]){assert.ok(!evidence.session_intents[key('session')],'Uncertain Session creation must be investigated first');evidence.session_intents[key('session')]={at:new Date().toISOString(),workspaceId:evidence.native_workspaces[item.key].workspace.workspaceId};await save();
      const created=await rpc('session/create',{request:{workspaceId:evidence.native_workspaces[item.key].workspace.workspaceId,agentPreset:'control-model'}});assert.equal(created.agentPreset,'control-model');evidence.sessions[item.key]=created.sessionId;await save();}
    const sessionId=evidence.sessions[item.key];
    const attached=await command(rpc,sessionId,'/control-workspace {}');assert.equal(attached.sessionId,sessionId);assert.equal(attached.workspace,workspace);assert.equal(attached.preset.agentPreset,'control-model');assert.equal(attached.preset.durable_hint,true);
    const status=row=>command(rpc,sessionId,'/control-status '+JSON.stringify({workspace,attemptId:row.attemptId}));
    const stage=async(part,operation,inputs={},config={})=>{
      const name=key(part),request={agent:'workflow',operation,workspace,inputs,config},signature=hash(JSON.stringify(request));let row=evidence.jobs[name];
      if(row){assert.deepEqual(row.request,request);assert.equal(row.signature,signature);assert.ok(row.jobId&&row.attemptId,'Uncertain saved admission has no handle; do not resubmit');}
      else{row=evidence.jobs[name]={caseKey:item.key,sessionId,request,signature,status:'ADMISSION_PENDING',admission_started_at:new Date().toISOString()};await save();
        const admitted=await command(rpc,sessionId,'/control '+JSON.stringify(request));assert.ok(admitted.jobId&&admitted.attemptId);Object.assign(row,{jobId:admitted.jobId,attemptId:admitted.attemptId,status:'RUNNING'});evidence.admissions.push({key:name,caseKey:item.key,operation,sessionId,jobId:row.jobId,attemptId:row.attemptId,signature});await save();}
      const deadline=Date.now()+30*60*1000;
      for(;;){const state=await status(row),record=state.record;assert.equal(record.attemptId,row.attemptId);assert.equal(record.nativeJobId,row.jobId);assert.equal(record.owner,sessionId);assert.equal(record.agentName,'workflow');const {agent,...expected}=request;assert.deepEqual(record.request,expected);
        row.last_observation={at:new Date().toISOString(),state:record.state,native_status:state.job?.status??null,progress:record.progress};
        if(['completed','failed','killed'].includes(record.state)&&record.outcome?.cleanupVerified===true&&(!state.job||['completed','failed','killed'].includes(state.job.status))){row.status=record.state;row.record=record;row.rawControlStatus=state;await save();assert.equal(record.state,'completed',name+' terminal failure (preserve original handle)');row.result=normalize(record.result);await save();return row.result;}
        await save();if(Date.now()>deadline){row.observer_deadline_at=new Date().toISOString();await save();throw Error('Observer deadline; job remains '+row.jobId+' / '+row.attemptId+'; use --resume to observe this exact handle, never submit again');}await sleep(1200);
      }
    };
    const events=async()=>{
      let page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:500}});const cursor=page.cursor,pages=[page];
      while(page.hasMore){const beforeSeq=Math.min(...page.records.map(x=>x.event.seq));page=await rpc('session/events',{request:{address:{kind:'session',sessionId},throughSeq:cursor,beforeSeq,maxMessages:500}});assert.equal(page.cursor,cursor);pages.push(page);}
      const rows=pages.flatMap(p=>p.records).sort((a,b)=>a.event.seq-b.event.seq);assert.equal(rows.length,cursor+1);rows.forEach((r,i)=>assert.equal(r.event.seq,i));return {sessionId,cursor,pages,records:rows,hasMore:false};
    };
    const snapshot=async()=>{let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(x=>/^[a-f0-9-]{36}$/.test(x)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}const native=await events();return {attempts,registered:native.records.filter(r=>r.event.type==='job/registered').map(r=>r.event.data.jobId),confirmation_commands:native.records.filter(r=>r.event.type==='command/run'&&r.event.data.name==='control-confirm').length};};
    const inputs={workflow_path:item.workflow_path},config={};
    const plan=await stage('plan','plan_workflow',inputs,config);assert.equal(plan.status,'ready');assert.deepEqual(plan.stages.map(x=>x.operation),operations);assert.deepEqual(plan.definition,item.definition.steps.some(s=>Object.values(s.inputs).some(v=>typeof v==='string'&&!isAbsolute(v)))?{...item.definition,steps:item.definition.steps.map(s=>({...s,inputs:Object.fromEntries(Object.entries(s.inputs).map(([n,v])=>[n,typeof v==='string'&&!isAbsolute(v)?resolve(dirname(item.workflow_path),v):v]))}))}:item.definition);
    let refs=evidence.phase_refs[item.key];
    if(!refs){const request={agent:'workflow',operation:'run_workflow',workspace,inputs,config};const reviewed=await command(rpc,sessionId,'/control-review '+JSON.stringify(request));assert.equal(reviewed.authority,'read_only_requirements_not_confirmation');assert.equal(reviewed.jobs_started,0);assert.deepEqual(reviewed.requirements,await confirmationRequirements(request));assert.deepEqual(reviewed.requirements.map(x=>x.scope),['signals','sampling','training']);evidence.reviews[item.key]={request,result:reviewed,at:new Date().toISOString()};await save();
      refs={};for(const req of reviewed.requirements){const proposed=await stage('propose-'+req.scope,'propose_case_plan',{}, {scope:req.scope,payload:req.expectedPayload,bindings:req.bindings,dependencies:structuredClone(refs)});assert.deepEqual(proposed.dependencies,refs);refs[req.scope]={plan_id:proposed.plan_id,version_hash:proposed.version_hash};}
      evidence.phase_refs[item.key]=refs;await save();
      const before=await snapshot(),pending=await rpc('commands/execute',{agentId:sessionId,line:'/control '+JSON.stringify({...request,config:{plan_refs:refs}}),submittedAttachments:[]});assert.equal(pending.result.kind,'error');assert.match(pending.result.text,/confirm|plan|pending|review/i);const after=await snapshot();assert.deepEqual(after,before,'Pending confirmation must not create any native job/history');evidence.rejections.push({caseKey:item.key,label:'pending full-workflow confirmation',nativeKind:'error',message:pending.result.text,before,after,new_jobs:0,new_attempts:0});await save();
    }
    for(const [scope,ref]of Object.entries(refs)){const state=await command(rpc,sessionId,'/control-plan-status '+JSON.stringify({workspace,...ref}));
      const saved=evidence.confirmations.find(x=>x.caseKey===item.key&&x.scope===scope);
      if(saved){assert.equal(state.confirmation_state,'confirmed');assert.equal(state.session_id,sessionId);assert.equal(state.source.command_id,saved.source.command_id);continue;}
      const confirmed=state.confirmation_state==='confirmed'?state:await command(rpc,sessionId,'/control-confirm '+JSON.stringify({workspace,...ref}));assert.equal(confirmed.confirmation_source,'test_fixture');assert.equal(confirmed.session_id,sessionId);assert.equal(confirmed.source.kind,'native_user_command');const checked=await command(rpc,sessionId,'/control-plan-status '+JSON.stringify({workspace,...ref}));assert.equal(checked.confirmation_state,'confirmed');evidence.confirmations.push({caseKey:item.key,scope,...confirmed});await save();}
    const run=await stage('run','run_workflow',inputs,{plan_refs:refs});assert.equal(run.status,'verified');assert.deepEqual(Object.keys(run.artifacts),item.definition.steps.map(s=>s.id));
    const manifestBefore=await json(resolve(workspace,'manifest.json')),artifactsBefore=Object.fromEntries(Object.entries(manifestBefore.artifacts).map(([id,row])=>[id,row]));
    assert.equal(Object.keys(artifactsBefore).length,8);for(const id of Object.values(run.artifacts))assert.equal(artifactsBefore[id].status,'verified');
    const metadataHashes={};for(const [id,row]of Object.entries(artifactsBefore))metadataHashes[id]=hash(await readFile(resolve(workspace,row.path,'artifact.json')));
    const beforeResume=await snapshot(),resumed=await stage('resume','run_workflow',inputs,{plan_refs:refs,resume_run:run.workflow_run});assert.equal(resumed.status,'verified');assert.equal(resumed.workflow_run,run.workflow_run);assert.deepEqual(resumed.artifacts,run.artifacts);assert.deepEqual(resumed.steps,run.steps);
    const manifestAfter=await json(resolve(workspace,'manifest.json'));assert.deepEqual(manifestAfter.artifacts,artifactsBefore,'Completed resume must preserve exact artifact identities/metadata versions');
    for(const [id,row]of Object.entries(manifestAfter.artifacts))assert.equal(hash(await readFile(resolve(workspace,row.path,'artifact.json'))),metadataHashes[id]);
    const afterResume=await snapshot();assert.equal(afterResume.confirmation_commands,beforeResume.confirmation_commands,'Completed reuse must not mint new confirmations');
    const raw=evidence.jobs[key('resume')].rawControlStatus.output.chunks.map(x=>x.text).join(''),progress=raw.split('\n').flatMap(line=>{try{return [JSON.parse(line)];}catch{return [];}}).filter(x=>x.event==='progress');
    const reused=progress.filter(x=>x.step_status==='reused_verified');assert.deepEqual(reused.map(x=>x.step),item.definition.steps.map(s=>s.id));assert.ok(!progress.some(x=>x.step_status==='running'),'Completed reuse cannot execute a numerical stage');
    const native=await events();assert.equal(native.records.filter(x=>x.event.type==='turn/start'||x.event.type==='request/header').length,0,'Keyless native acceptance must create no model turns');
    const eventPath=resolve(dirname(evidencePath),'legacy-'+item.key+'-session-events.json');await writeSmokeEvidence(eventPath,native);evidence.session_events[item.key]={path:eventPath,sha256:hash(await readFile(eventPath)),sessionId,cursor:native.cursor,hasMore:false,records:native.records.length,model_turns:0};
    const check={caseKey:item.key,workspace,sessionId,workflow_run:run.workflow_run,artifacts:run.artifacts,original_operations:operations,all_eight_verified:true,completed_resume:{same_workflow:true,same_artifacts:true,unchanged_metadata_sha256:metadataHashes,reused_steps:reused.map(x=>x.step),new_confirmations:0,new_numeric_stages:0},confirmation_source:'test_fixture'};
    if(evidence.checks.some(x=>x.caseKey===item.key))assert.deepEqual(evidence.checks.find(x=>x.caseKey===item.key),check);else evidence.checks.push(check);await save();
  }
  for(const [name,digest]of Object.entries(registration.component_sha256))assert.equal(hash(await readFile(resolve(project,name))),digest,'Original component changed during acceptance '+name);
  for(const item of registration.cases)for(const file of item.files){assert.equal(hash(await readFile(file.source)),file.sha256);assert.equal(hash(await readFile(file.path)),file.sha256);}
  evidence.source_versions_unchanged=true;evidence.status='PASS';evidence.finished_at=new Date().toISOString();await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,sha256:hash(await readFile(evidencePath)),cases:evidence.checks.map(x=>x.caseKey),native_jobs:evidence.admissions.length,LLM:0,MATLAB:0}));
}catch(error){evidence.status='FAIL';evidence.error={name:error.name,message:error.message,stack:error.stack};evidence.finished_at=new Date().toISOString();await save();console.error(JSON.stringify({status:'FAIL',evidence:evidencePath,error:error.message,handles:Object.entries(evidence.jobs).map(([key,r])=>({key,jobId:r.jobId,attemptId:r.attemptId,status:r.status}))}));process.exitCode=1;}
