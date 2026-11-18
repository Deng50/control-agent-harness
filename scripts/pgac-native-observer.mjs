/** Test-only S12 observer. No connection is made by importing this module. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFile,readdir,writeFile} from 'node:fs/promises';
import {basename,dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,project,writeSmokeEvidence} from './harness-rpc.mjs';

export const hash=value=>createHash('sha256').update(value).digest('hex');
export const json=path=>readFile(path,'utf8').then(JSON.parse);
export const normalize=value=>{if(typeof value==='string')value=JSON.parse(value);assert.ok(value&&typeof value==='object'&&!Array.isArray(value),'Expected one actual JSON result, at most one display serialization');return value;};
export const inside=(base,target)=>{const path=relative(base,target);return path!==''&&!isAbsolute(path)&&path!=='..'&&!path.startsWith(`..${sep}`);};
export const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
export const runPython=(code,args=[])=>JSON.parse(execFileSync(python,['-B','-c',code,...args],{encoding:'utf8',windowsHide:true,maxBuffer:32*1024*1024,
  env:{...process.env,PYTHONPATH:resolve(project,'src'),PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'}}));

export async function artifact(workspace,id){
  const manifest=await json(resolve(workspace,'manifest.json')),entry=manifest.artifacts[id];assert.ok(entry,'Actual artifact is absent: '+id);
  const root=resolve(workspace,entry.path);assert.ok(inside(resolve(workspace,'artifacts'),root));
  const bytes=await readFile(resolve(root,'artifact.json'));assert.equal(hash(bytes),entry.metadata_file_sha256,'Artifact metadata bytes changed');const metadata=JSON.parse(bytes);assert.equal(metadata.id,id);
  const memberBytes=async name=>{const path=resolve(root,name);assert.ok(inside(root,path));const bytes=await readFile(path);assert.ok(bytes.length<=256*1024*1024);assert.equal(hash(bytes),metadata.files[name],'Artifact member changed: '+name);return bytes;};
  return {root,metadata,memberBytes,member:async name=>JSON.parse(await memberBytes(name))};
}

/** Validate pinned R01 facts, not its mutable observer timestamp/current status page. */
export function validationBindings(upstream){return {
  registration_sha256:upstream.preRegistered.registration_sha256,
  source_file_hashes:upstream.preRegistered.source_file_hashes,
  component_source_sha256:upstream.preRegistered.component_source_sha256,
  driver_sha256:upstream.preRegistered.driver_sha256,
  artifacts:upstream.checks.map(({caseKey,workspace,sessionId,policy,plant,candidate,dataset,validation,negative})=>({caseKey,workspace,sessionId,...(policy?{policy}:{}),plant,candidate,dataset,validation,...(negative?{negative}:{})})),
  native_admissions:upstream.admissions.map(({key,jobId,attemptId,signature})=>({key,jobId,attemptId,signature})),
  native_requests:Object.fromEntries(Object.entries(upstream.jobs).map(([key,row])=>{assert.equal(hash(JSON.stringify(row.request)),row.signature);return [key,{request:row.request,signature:row.signature,jobId:row.jobId,attemptId:row.attemptId}];})),
  fixture_markers:upstream.fixture_markers,confirmations:upstream.confirmations,
  audit_snapshot:upstream.mcp_audit?{path:upstream.mcp_audit.path,sha256:upstream.mcp_audit.sha256,through_bytes:upstream.mcp_audit.through_bytes}:null
};}

export async function preflight(values,driverFiles,{needProbes=true}={}){
  const requirePinned=async(name)=>{const path=values[name]&&resolve(values[name]),digest=values[name+'-sha256'];assert.ok(path&&/^[a-f0-9]{64}$/.test(digest??''),'Required pinned --'+name+' and --'+name+'-sha256');
    const bytes=await readFile(path);assert.equal(hash(bytes),digest,name+' bytes changed');return {path,sha256:digest,bytes,value:JSON.parse(bytes)};};
  const registered=await requirePinned('registration'),registration=registered.value;assert.equal(registration.phase,'S12-R03');assert.equal(registration.cases.length,2);
  const sources={};for(const [name,digest] of Object.entries(registration.source_sha256)){const path=resolve(project,name);assert.ok(inside(project,path));assert.equal(hash(await readFile(path)),digest,'Frozen product changed: '+name);sources[name]=digest;}
  assert.ok(values['validation-evidence'],'Explicit actual R01 evidence required');const upstreamPath=resolve(values['validation-evidence']),upstreamBytes=await readFile(upstreamPath),upstream=JSON.parse(upstreamBytes);
  assert.equal(upstream.status,'PASS','R01 must be terminal PASS before any new Session');assert.equal(upstream.native_result,'PASS');
  assert.equal(upstream.preRegistered.registration_sha256,registration.validation_registration_sha256);
  assert.equal(hash(await readFile(resolve(registration.validation_registration_path))),registration.validation_registration_sha256);
  const proof=await requirePinned('validation-proof');assert.equal(proof.value.status,'PASS','Independent R01 numeric proof must pass before any new Session');
  const bindings=validationBindings(upstream),proofBindings=proof.value.input_bindings;
  assert.ok(proofBindings,'Independent R01 proof must expose its immutable input_bindings');
  assert.equal(proofBindings.registration_sha256,bindings.registration_sha256);
  assert.deepEqual(proofBindings.artifacts,bindings.artifacts.map(({negative,...row})=>row),'Independent proof must bind all exact core artifact IDs; negative Native cases remain separately pinned');
  assert.deepEqual(proofBindings.native_admissions,bindings.native_admissions);
  if(proofBindings.source_file_hashes)assert.deepEqual(proofBindings.source_file_hashes,bindings.source_file_hashes);
  if(proofBindings.component_source_sha256)assert.deepEqual(proofBindings.component_source_sha256,bindings.component_source_sha256);
  for(const [name,digest] of Object.entries(bindings.source_file_hashes))assert.equal(hash(await readFile(name)),digest);
  for(const [name,digest] of Object.entries(bindings.component_source_sha256))assert.equal(hash(await readFile(resolve(project,name))),digest);
  if(bindings.audit_snapshot){assert.equal(hash(await readFile(bindings.audit_snapshot.path)),bindings.audit_snapshot.sha256);}
  const cases=[];
  for(const item of registration.cases){const actual=upstream.checks.find(row=>row.caseKey===item.key);assert.ok(actual?.status==='PASS');assert.equal(actual.policy,item.policy_id);
    const workspace=resolve(actual.workspace);assert.ok(inside(resolve(project,'test-output'),workspace));if(item.workspace)assert.equal(workspace,resolve(item.workspace));for(const key of ['policy','plant','candidate','dataset','validation'])assert.equal(typeof actual[key],'string');
    const marker=bindings.fixture_markers[item.key];assert.ok(marker);assert.equal(hash(await readFile(marker.path)),marker.sha256,'R01 marker must be preserved byte-for-byte');
    const parsed=await json(marker.path);assert.equal(parsed.workspace,workspace);assert.equal(parsed.confirmation_source,'test_fixture');assert.deepEqual(parsed.source,marker.source);
    const policy=await artifact(workspace,item.policy_id);assert.equal(hash(await readFile(resolve(policy.root,'artifact.json'))),item.policy_metadata_sha256);
    const independentlyChecked=proof.value.cases?.find(row=>row.caseKey===item.key);assert.ok(independentlyChecked?.artifact_metadata_sha256,'Independent proof must retain exact source/array artifact versions');
    for(const [id,digest] of Object.entries(independentlyChecked.artifact_metadata_sha256)){
      const bound=await artifact(workspace,id);assert.equal(hash(await readFile(resolve(bound.root,'artifact.json'))),digest,'Independent R01 source/array metadata changed: '+id);
      for(const name of Object.keys(bound.metadata.files))await bound.memberBytes(name);
    }
    await artifact(workspace,actual.candidate);await artifact(workspace,actual.validation);cases.push({...item,workspace,actual,marker,policyRoot:policy.root,baseline_weights_sha256:hash(await policy.memberBytes('bundle/model.pt'))});
  }
  const pinned={registration_path:registered.path,registration_sha256:registered.sha256,registration,source_sha256:sources,
    validation_evidence_path:upstreamPath,validation_bindings:bindings,validation_proof_path:proof.path,validation_proof_sha256:proof.sha256,script_sha256:{}};
  for(const path of driverFiles)pinned.script_sha256[resolve(path)]=hash(await readFile(path));
  if(needProbes){const probes=await requirePinned('gradient-probes');assert.equal(probes.value.status,'PASS');assert.ok(Array.isArray(probes.value.cases));
    for(const item of cases){const point=probes.value.cases.find(row=>row.caseKey===item.key);assert.ok(point,'A pre-training TRAIN probe is required per case');
      assert.equal(point.policy,item.policy_id);assert.equal(point.model,item.actual.candidate);assert.equal(point.validation,item.actual.validation);assert.equal(resolve(point.workspace),resolve(item.workspace));
      assert.equal(point.baseline_weights_sha256,item.baseline_weights_sha256);assert.ok(Number.isInteger(point.row)&&point.row>=0);assert.equal(point.TRAIN_only,true);assert.equal(point.validation_or_test_values_used_for_selection,false);assert.equal(point.optimizer_steps,0);}
    for(const item of cases){const point=probes.value.cases.find(row=>row.caseKey===item.key);assert.equal(point.original,item.actual.plant);assert.deepEqual(point.domain,item.config.domain);}
    const oraclePath=values['gradient-oracle']&&resolve(values['gradient-oracle']),oracleHash=values['gradient-oracle-sha256'];assert.ok(oraclePath&&/^[a-f0-9]{64}$/.test(oracleHash??''));assert.equal(hash(await readFile(oraclePath)),oracleHash);
    assert.equal(probes.value.registration_sha256,registered.sha256);assert.equal(probes.value.validation_proof_sha256,proof.sha256);
    const code=probes.value.code_sha256;assert.ok(code&&code['actor_probe_numeric.py']&&code['verify-physical-actor-gradient-v2.py']);
    for(const [name,digest] of Object.entries(code))assert.equal(hash(await readFile(resolve(dirname(oraclePath),name))),digest,'Probe/oracle helper changed');
    assert.equal(code['verify-physical-actor-gradient-v2.py'],oracleHash);
    pinned.gradient_probes={path:probes.path,sha256:probes.sha256,cases:probes.value.cases};pinned.gradient_oracle={path:oraclePath,sha256:oracleHash,execution:'not_run_by_this_native_driver; independent post-training oracle consumes saved initial_imitation_state and pre-training TRAIN point'};
  }
  return {registration,cases,pinned,initialUpstreamSha256:hash(upstreamBytes)};
}

export async function openEvidence(path,values,prepared,phase){
  let e;if(values.resume){assert.ok(!values.supersedes,'Use --supersedes only to create one explicit new observer evidence; use --resume thereafter');e=await json(path);}
  else{try{await readFile(path);assert.fail('Existing evidence requires --resume; never overwrite original handles');}catch(error){if(error.code!=='ENOENT')throw error;}
    if(prepared.superseded){e=structuredClone(prepared.superseded.evidence);e.original_failure={status:e.status,error:e.error,finished_at:e.finished_at};
      e.observer_parent={path:prepared.superseded.path,sha256:prepared.superseded.sha256,status:'FAIL'};
      e.observer_revision={kind:'S12_explicit_phase_dependencies_v1',recorded_at:new Date().toISOString(),previous_script_sha256:e.preRegistered.script_sha256,
        script_sha256:prepared.pinned.script_sha256,reason:'Explicit phase proposal dependencies and MATLAB native-vs-recomputed selection core-field assertions only. Preserve completed signals and failed training proposal; no numerical fit or MCP occurred.',
        allowed_failed_proposal:prepared.superseded.failedKey};
      e.allowed_proposal_repairs={[prepared.superseded.failedKey]:{old_jobId:e.jobs[prepared.superseded.failedKey].jobId,old_attemptId:e.jobs[prepared.superseded.failedKey].attemptId,
        old_signature:e.jobs[prepared.superseded.failedKey].signature,only_change:'config.dependencies',reason:'Explicit dependency on this phase signals, excluding unrelated stale R01 sampling'}};
      delete e.error;delete e.finished_at;e.status='RUNNING';e.observer_started_at=new Date().toISOString();}
    else e={schema_version:1,phase,status:'RUNNING',started_at:new Date().toISOString(),confirmation_source:'test_fixture',preRegistered:prepared.pinned,
      first_validation_evidence_sha256:prepared.initialUpstreamSha256,sessions:{},jobs:{},admissions:[],confirmations:[],confirmation_reuses:[],phase_requests:{},rejections:[],historical_phase_checks:[],checks:[]};}
  assert.equal(e.phase,phase);
  const withoutScripts=({script_sha256,...rest})=>rest;assert.deepEqual(withoutScripts(e.preRegistered),withoutScripts(prepared.pinned),'Observer cannot revise numerical registration, source, proof or probe bytes');
  assert.deepEqual(e.observer_revision?.script_sha256??e.preRegistered.script_sha256,prepared.pinned.script_sha256,'Observer source revision must be explicit, never silently replace old driver hashes');
  if(e.observer_parent)assert.equal(hash(await readFile(e.observer_parent.path)),e.observer_parent.sha256,'Original failed observer evidence must remain byte-for-byte preserved');
  if(values.resume){e.observer_history??=[];if(e.error)e.observer_history.push({status:e.status,error:e.error,resumed_at:new Date().toISOString()});delete e.error;delete e.finished_at;e.status='RUNNING';}
  await writeSmokeEvidence(path,e);return e;
}

/** One reviewed observer-only migration. Does not write or contact the Host. */
export async function validateSuperseded(values,prepared){
  if(!values.supersedes){assert.ok(!values['supersedes-sha256']);return;}
  assert.ok(!values.resume&&/^[a-f0-9]{64}$/.test(values['supersedes-sha256']??''));const path=resolve(values.supersedes),bytes=await readFile(path);assert.equal(hash(bytes),values['supersedes-sha256']);
  assert.notEqual(path,resolve(values.evidence));const e=JSON.parse(bytes);assert.equal(e.phase,'S12-R03');assert.equal(e.status,'FAIL');assert.ok(!e.observer_revision);
  const withoutScripts=({script_sha256,...rest})=>rest;assert.deepEqual(withoutScripts(e.preRegistered),withoutScripts(prepared.pinned),'Supersession changes observer source only');
  for(const [script,digest] of Object.entries(e.preRegistered.script_sha256))assert.equal(hash(await readFile(resolve(dirname(path),'driver-v1-predependency',basename(script)))),digest,'Original executed observer source snapshot differs');
  const keys=Object.keys(e.jobs);assert.deepEqual(keys,['f05-python/optimize-plan-signals','f05-python/optimize-plan-training']);assert.equal(e.admissions.length,2);assert.equal(e.confirmations.length,0);assert.equal(e.checks.length,0);assert.equal(Object.keys(e.phase_requests).length,0);
  const failedKey=keys[1];for(let index=0;index<keys.length;index++){const key=keys[index],row=e.jobs[key];assert.ok(row.jobId&&row.attemptId);assert.equal(row.operation,'propose_case_plan');assert.equal(row.status,index?'failed':'completed');assert.equal(row.record.state,row.status);assert.equal(row.record.outcome.cleanupVerified,true);
    assert.equal(row.request.config.scope,index?'training':'signals');assert.equal(row.request.config.dependencies,undefined);assert.equal(hash(JSON.stringify(row.request)),row.signature);
    assert.equal(row.sessionId,e.sessions['f05-python']);assert.equal(row.record.agentName,'lightweight');assert.equal(row.record.nativeJobId,row.jobId);assert.equal(row.record.owner,row.sessionId);assert.equal(row.record.attemptId,row.attemptId);const {agent,...request}=row.request;assert.equal(agent,'lightweight');assert.deepEqual(row.record.request,request);
    const actual=await json(resolve(row.request.workspace,'.control-jobs',row.attemptId,'request.json'));assert.equal(actual.nativeJobId??row.jobId,row.jobId);assert.equal(actual.attemptId,row.attemptId);assert.equal(actual.owner,row.sessionId);assert.deepEqual(actual.request,request);
    assert.equal(e.admissions[index].jobId,row.jobId);assert.equal(e.admissions[index].attemptId,row.attemptId);assert.equal(e.admissions[index].signature,row.signature);
    const directory=resolve(row.request.workspace,'.control-jobs',row.attemptId),events=[];for(const name of (await readdir(directory)).filter(name=>/^\d{12}\.json$/.test(name)).sort())events.push(await json(resolve(directory,name)));
    assert.equal(events.find(event=>event.nativeJobId)?.nativeJobId,row.jobId);const outcomes=events.filter(event=>event.outcome);assert.ok(outcomes.length);assert.equal(outcomes.at(-1).outcome.status,row.status);assert.equal(outcomes.at(-1).outcome.cleanupVerified,true);
  }
  assert.match(e.jobs[failedKey].record.detail??'',/^PendingPlanConfirmation: Plan dependency version is stale$/);
  assert.equal((await auditRows(prepared.pinned.mcp_audit_path,e.jobs[keys[0]].sessionId)).length,0,'Supersession may not hide an actual original-source/MCP invocation');
  const item=prepared.cases.find(row=>row.key==='f05-python'),requirements=await confirmationRequirements({operation:'optimize_external_policy',workspace:item.workspace,inputs:{policy:item.policy_id,model:item.actual.candidate,validation:item.actual.validation},config:item.config});
  for(let index=0;index<keys.length;index++){const req=requirements[index],request=e.jobs[keys[index]].request;assert.equal(request.config.scope,req.scope);assert.deepEqual(request.config.payload,req.expectedPayload);assert.deepEqual(request.config.bindings,req.bindings);}
  prepared.superseded={path,sha256:hash(bytes),evidence:e,failedKey};
}

export function observer(rpc,e,evidencePath,item,{sessionKey=item.key}={}){
  const workspace=resolve(item.workspace),owner='lightweight',key=part=>sessionKey+'/'+part,save=()=>writeSmokeEvidence(evidencePath,e);
  const session=async()=>{assert.equal(hash(await readFile(item.marker.path)),item.marker.sha256);
    if(!e.sessions[sessionKey]){const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});assert.equal(created.agentPreset,'control-lightweight');assert.notEqual(created.sessionId,item.actual.sessionId);e.sessions[sessionKey]=created.sessionId;await save();}return e.sessions[sessionKey];};
  const events=async()=>{const sessionId=await session();let page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:500}});const cursor=page.cursor,pages=[page];let before;
    while(page.hasMore){assert.ok(page.records.length);const next=Math.min(...page.records.map(row=>row.event.seq));assert.ok(before===undefined||next<before);before=next;page=await rpc('session/events',{request:{address:{kind:'session',sessionId},throughSeq:cursor,beforeSeq:before,maxMessages:500}});assert.equal(page.cursor,cursor);pages.push(page);}
    const records=pages.flatMap(row=>row.records).sort((a,b)=>a.event.seq-b.event.seq);assert.equal(records.length,cursor+1);records.forEach((row,index)=>assert.equal(row.event.seq,index));assert.equal(page.hasMore,false);
    const path=resolve(dirname(evidencePath),'pgac-'+sessionKey+'-events-'+cursor+'.json');await writeSmokeEvidence(path,{source:'actual complete native Session pages',sessionId,cursor,pages,records,hasMore:false});
    e.session_events??={};e.session_events[sessionKey]={path,sha256:hash(await readFile(path)),cursor,sessionId,pages:pages.length,records:records.length,hasMore:false};await save();return records;};
  const status=row=>command(rpc,row.sessionId,`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);
  const assertRecord=(row,state)=>{const r=state.record;assert.equal(r.owner,row.sessionId);assert.equal(r.agentName,owner);assert.equal(r.nativeJobId,row.jobId);assert.equal(r.attemptId,row.attemptId);
    const {agent,...request}=row.request;assert.equal(agent,owner);assert.deepEqual(r.request,request);return r;};
  const admit=async(part,operation,inputs={},config={},options={})=>{const name=key(part),request={agent:owner,operation,workspace,inputs,config},signature=hash(JSON.stringify(request));let row=e.jobs[name];
    if(row){assert.equal(row.signature,signature);assert.deepEqual(row.request,request);assert.equal(row.expected,options.expected??'completed');assert.ok(row.jobId&&row.attemptId,'Uncertain admission must be reconciled; never resubmit');return row;}
    row=e.jobs[name]={key:name,operation,owner,sessionId:await session(),request,signature,expected:options.expected??'completed',status:'ADMISSION_PENDING'};await save();
    const started=await command(rpc,row.sessionId,'/control '+JSON.stringify(request));assert.ok(started.jobId&&started.attemptId);Object.assign(row,{jobId:started.jobId,attemptId:started.attemptId,status:'RUNNING'});
    e.admissions.push({key:name,caseKey:item.key,sessionKey,operation,owner,sessionId:row.sessionId,jobId:row.jobId,attemptId:row.attemptId,signature});await save();return row;};
  const wait=async(row,expected=row.expected)=>{const deadline=Date.now()+30*60*1000;
    for(;;){const state=await status(row),record=assertRecord(row,state);row.last_observation={at:new Date().toISOString(),state:record.state,native_status:state.job?.status??null};
      if(['completed','failed','killed'].includes(record.state)&&record.outcome?.cleanupVerified===true&&(!state.job||['completed','failed','killed'].includes(state.job.status))){Object.assign(row,{status:record.state,record,rawControlStatus:state});if(record.result!=null)row.result=normalize(record.result);await save();assert.equal(record.state,expected,`${row.key} ended ${record.state}: ${record.detail??''}`);return row.result??null;}
      assert.notEqual(record.state,'interrupted','Original owner lost; inspect the saved handle, do not resubmit');if(Date.now()>=deadline){row.observer_deadline_reached=true;await save();throw new Error('Observer deadline; resume saved original handle '+row.jobId);}
      await new Promise(done=>setTimeout(done,100));}
  };
  const stage=async(part,operation,inputs={},config={},options={})=>wait(await admit(part,operation,inputs,config,options));
  const snapshot=async()=>{let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(name=>/^[0-9a-f-]{36}$/.test(name)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}
    const history=await command(rpc,await session(),`/control-history ${JSON.stringify({workspace})}`),records=await events();return {attempts,owned_history:history.attempts.map(row=>row.attemptId).sort(),native_jobs:records.filter(row=>row.event.type==='job/registered').map(row=>row.event.data.jobId)};};
  const deny=async(label,operation,inputs,config)=>{label=key(label);if(e.rejections.some(row=>row.label===label))return;
    const before=await snapshot(),request={agent:owner,operation,workspace,inputs,config},response=await rpc('commands/execute',{agentId:await session(),line:'/control '+JSON.stringify(request),submittedAttachments:[]});
    assert.equal(response.result.kind,'error','Unconfirmed changed request was admitted');assert.match(response.result.text,/confirm|plan|version|review/i);const after=await snapshot();assert.deepEqual(after,before,'Denied request created history/native job');
    e.rejections.push({label,status:'PASS_ZERO_ADMISSION',request,reason:response.result.text,before,after,confirmation_source:'test_fixture'});await save();};
  const phaseRefs=async(phase,target,operation,inputs,config,scopes)=>{const phaseKey=key(phase),admitted=e.jobs[key(target)];
    if(!admitted){const reqs=await confirmationRequirements({operation,workspace,inputs,config});assert.deepEqual(reqs.map(row=>row.scope),scopes);const refs={},states={};
      for(const req of reqs){const originalPart=phase+'-plan-'+req.scope,originalKey=key(originalPart),old=e.jobs[originalKey];let part=originalPart,proposalConfig={scope:req.scope,payload:req.expectedPayload,bindings:req.bindings,dependencies:structuredClone(refs)};
        if(old?.status==='completed'){
          // Keep the original successful request/job. Its resolved dependency
          // envelope must be exactly the dependencies now explicitly intended.
          const resolved=normalize(old.result??old.record?.result);assert.deepEqual(resolved.dependencies,refs);assert.deepEqual(old.request.config.payload,req.expectedPayload);assert.deepEqual(old.request.config.bindings,req.bindings);proposalConfig=old.request.config;
        }else if(old?.status==='failed'){
          const allowed=e.allowed_proposal_repairs?.[originalKey];assert.ok(allowed,'Failed proposal cannot be automatically retried');assert.equal(allowed.old_jobId,old.jobId);assert.equal(allowed.old_attemptId,old.attemptId);assert.equal(allowed.old_signature,old.signature);
          assert.equal(old.request.config.dependencies,undefined);const {dependencies,...unchanged}=proposalConfig;assert.deepEqual(unchanged,old.request.config,'A repaired proposal may change only explicit dependencies');
          part=originalPart+'-explicit-dependencies-v1';e.proposal_repairs??={};const repair={old_key:originalKey,old_jobId:old.jobId,old_attemptId:old.attemptId,old_signature:old.signature,
            new_key:key(part),new_request_sha256:hash(JSON.stringify({agent:owner,operation:'propose_case_plan',workspace,inputs:{},config:proposalConfig})),only_change:'config.dependencies',dependencies};
          if(e.proposal_repairs[originalKey])assert.deepEqual(e.proposal_repairs[originalKey],repair);else e.proposal_repairs[originalKey]=repair;await save();
        }
        const plan=await stage(part,'propose_case_plan',{},proposalConfig);assert.deepEqual(plan.dependencies,refs);e.phase_proposals??={};e.phase_proposals[phaseKey]??={};e.phase_proposals[phaseKey][req.scope]=key(part);await save();refs[req.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};
        const state=await command(rpc,await session(),'/control-plan-status '+JSON.stringify({workspace,...refs[req.scope]}));assert.equal(state.session_id,await session());assert.equal(state.scope,req.scope);states[req.scope]=state;}
      if(Object.values(states).some(row=>row.confirmation_state==='pending'))await deny(phase+'-pending',operation,inputs,{...config,plan_refs:refs});
      for(const req of reqs){let confirmed=states[req.scope],reused=confirmed.confirmation_state==='confirmed';if(!reused){assert.equal(confirmed.confirmation_state,'pending');confirmed=await command(rpc,await session(),'/control-confirm '+JSON.stringify({workspace,...refs[req.scope]}));}
        assert.equal(confirmed.session_id,await session());assert.equal(confirmed.confirmation_source,'test_fixture');assert.equal(confirmed.source.kind,'native_user_command');
        const receipt={phase:phaseKey,caseKey:item.key,sessionKey,owner,sessionId:confirmed.session_id,scope:req.scope,...refs[req.scope],confirmation_source:'test_fixture',source:confirmed.source};
        const old=e.confirmations.find(row=>row.phase===phaseKey&&row.scope===req.scope);if(old)assert.deepEqual(old,receipt);else e.confirmations.push(receipt);if(reused)e.confirmation_reuses.push(receipt);await save();}
      const request={agent:owner,operation,workspace,inputs,config:{...config,plan_refs:refs}};if(e.phase_requests[phaseKey])assert.deepEqual(e.phase_requests[phaseKey],request);else e.phase_requests[phaseKey]=request;await save();return refs;
    }
    // Historical evidence remains authoritative after later scope heads move.
    const request=e.phase_requests[phaseKey];assert.ok(request);assert.equal(hash(JSON.stringify(request)),admitted.signature);assert.deepEqual(admitted.request,request);const state=await status(admitted),record=assertRecord(admitted,state);
    assert.equal(request.operation,operation);assert.deepEqual(request.inputs,inputs);const {plan_refs:refs,...actual}=request.config;assert.deepEqual(actual,config);const records=await events(),sources=[];
    for(const scope of scopes){const ref=refs[scope],receipt=e.confirmations.find(row=>row.phase===phaseKey&&row.scope===scope);assert.ok(receipt);assert.equal(receipt.sessionId,await session());assert.equal(receipt.plan_id,ref.plan_id);assert.equal(receipt.version_hash,ref.version_hash);
      const proposal=e.jobs[e.phase_proposals?.[phaseKey]?.[scope]??key(phase+'-plan-'+scope)];assert.ok(proposal?.jobId);const previous=await status(proposal),proposed=assertRecord(proposal,previous);assert.equal(proposed.state,'completed');const p=normalize(proposed.result);assert.equal(p.plan_id,ref.plan_id);assert.equal(p.version_hash,ref.version_hash);
      const run=records[receipt.source.seq]?.event;assert.equal(run?.type,'command/run');assert.equal(run.data.name,'control-confirm');assert.equal(run.data.commandId,receipt.source.command_id);assert.equal(run.data.source.kind,'user');assert.deepEqual(JSON.parse(run.data.args),{workspace,...ref});
      const done=records.find(row=>row.event.type==='command/done'&&row.event.data.commandId===run.data.commandId)?.event;assert.equal(done?.data.kind,'success');const confirmed=JSON.parse(done.data.text);assert.equal(confirmed.session_id,await session());assert.equal(confirmed.scope,scope);assert.deepEqual(confirmed.source,receipt.source);assert.equal(confirmed.confirmation_source,'test_fixture');sources.push({scope,...ref,confirmation_source:'test_fixture',source:receipt.source});}
    if(record.result){const result=normalize(record.result);if(result.artifact_id){const a=await artifact(workspace,result.artifact_id);if(a.metadata.files['report.json'])assert.deepEqual((await a.member('report.json')).confirmation_sources,sources);}}
    e.historical_phase_checks.push({sessionKey,phase:phaseKey,jobId:admitted.jobId,attemptId:admitted.attemptId,signature:admitted.signature,confirmation_sources:sources,current_scope_head_used:false});await save();return refs;
  };
  return {workspace,sessionKey,key,save,session,events,status,assertRecord,admit,wait,stage,snapshot,deny,phaseRefs};
}

export async function auditRows(path,sessionId){const bytes=await readFile(path);return bytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line)).filter(row=>row.source==='trusted_test_observer'&&row.sessionId===sessionId);}
export async function assertAudit(path,sessionId,matlab){
  const rows=await auditRows(path,sessionId);if(!matlab){assert.equal(rows.length,0,'Pure Python workflow called MATLAB');return {status:'PASS',total_actual_mcp_calls:0,business_batch_calls:0,fixed_guard_probe_calls:0,rows};}
  const starts=rows.filter(row=>row.event==='started'),returns=rows.filter(row=>row.event==='returned');assert.equal(rows.length,6);assert.equal(starts.length,3);assert.equal(returns.length,3);let batches=0,probes=0;const ids=new Set();
  for(const row of starts){const id=row.callId??row.rootCallId;assert.ok(id&&!ids.has(id));ids.add(id);assert.equal(row.name,'mcp__matlab__evaluate_matlab_code');const pairs=returns.filter(other=>(other.callId??other.rootCallId)===id);assert.equal(pairs.length,1);assert.equal(pairs[0].isError,false);
    const label=[row.callId,row.rootCallId].find(x=>x?.startsWith('matlab-batch-')||x?.startsWith('matlab-probe-'));if(label?.startsWith('matlab-batch-'))batches++;else if(label?.startsWith('matlab-probe-'))probes++;else assert.fail('Unrecognized actual MATLAB identity');}
  assert.equal(batches,1);assert.equal(probes,2);return {status:'PASS',total_actual_mcp_calls:3,business_batch_calls:batches,fixed_guard_probe_calls:probes,rows};
}
export async function freezeAudit(path,e,evidencePath,label){
  if(e.mcp_audit){const snapshot=await readFile(e.mcp_audit.path),live=await readFile(path);assert.equal(hash(snapshot),e.mcp_audit.sha256);assert.ok(live.length>=snapshot.length&&live.subarray(0,snapshot.length).equals(snapshot));return;}
  const bytes=await readFile(path),target=resolve(dirname(evidencePath),label+'-audit-snapshot.jsonl');try{await writeFile(target,bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;assert.ok((await readFile(target)).equals(bytes));}
  e.mcp_audit={path:target,sha256:hash(bytes),through_bytes:bytes.length,source_live_path:path,scope:'Immutable actual observer prefix; R01 prefix untouched'};await writeSmokeEvidence(evidencePath,e);
}

export const checkpointInspect=String.raw`import hashlib,json,sys,torch
from pathlib import Path
from control_harness.locking import process_alive
p=Path(sys.argv[1]);c=torch.load(p,map_location='cpu',weights_only=True)
side=json.loads(p.with_name('latest.checkpoint.json').read_text(encoding='utf-8'));sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==side['sha256']
intent=json.loads(p.with_name('stage.intent.json').read_text(encoding='utf-8'));assert side['stage_intent_sha256']==hashlib.sha256(p.with_name('stage.intent.json').read_bytes()).hexdigest()
def digest(v):
 h=hashlib.sha256()
 if isinstance(v,torch.Tensor):h.update(str(v.dtype).encode());h.update(str(list(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 elif isinstance(v,dict):
  for k in sorted(v,key=str):h.update(str(k).encode());h.update(digest(v[k]).encode())
 elif isinstance(v,(list,tuple)):
  for x in v:h.update(digest(x).encode())
 else:h.update(repr(v).encode())
 return h.hexdigest()
print(json.dumps({'path':str(p),'sha256':sha,'epoch':c['epoch'],'updates':c['updates'],'training_signature':c['training_signature'],'algorithm':c['contract']['algorithm'],
 'model_sha256':digest(c['model_state']),'optimizer_sha256':digest(c['optimizer_state']),'best_state_sha256':digest(c['best_state']),'rng_sha256':digest(c['rng_state']),
 'learning_rates':[x['lr'] for x in c['optimizer_state']['param_groups']],'log_epochs':[r['epoch'] for r in c['log']],
 'checkpoint_kind':intent['kind'],'owner_pid':intent['owner']['pid'],'owner_alive':process_alive(intent['owner']['pid']),'sidecar':side},allow_nan=False))`;

export async function terminalWorkers(o,e){
  const records=[],pids=[];
  for(const row of Object.values(e.jobs).filter(row=>row.sessionId===e.sessions[o.sessionKey])){
    assert.ok(row.jobId&&row.attemptId,'Unresolved native admission');const state=await o.status(row),record=o.assertRecord(row,state);
    assert.ok(['completed','failed','killed'].includes(record.state));assert.equal(record.outcome?.cleanupVerified,true);records.push({jobId:row.jobId,attemptId:row.attemptId,operation:row.operation,state:record.state,cleanupVerified:true});
    const directory=resolve(o.workspace,'.control-jobs',row.attemptId);for(const name of (await readdir(directory)).filter(name=>/^\d{12}\.json$/.test(name))){const event=await json(resolve(directory,name));if(event.workerEvent?.event==='started')pids.push(event.workerEvent.pid);}
  }
  const workers=runPython("import json,sys\nfrom control_harness.locking import process_alive\nprint(json.dumps([{'pid':p,'alive':process_alive(p)} for p in json.loads(sys.argv[1])]))",[JSON.stringify([...new Set(pids)])]);assert.ok(workers.every(row=>row.alive===false),'Owned numerical worker still alive');
  try{await readFile(resolve(o.workspace,'.operation.lock'));assert.fail('Workspace still has an operation owner');}catch(error){if(error.code!=='ENOENT')throw error;}
  return {records,workers,workspace_operation_lock_released:true};
}
