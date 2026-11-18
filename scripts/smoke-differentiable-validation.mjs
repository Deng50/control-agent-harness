/** S12 R01/R02 native source/rewrite validation. Never starts a Host or MATLAB. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFile,readdir,writeFile} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{registration:{type:'string'},'registration-sha256':{type:'string'},evidence:{type:'string'},
  'web-log':{type:'string'},'mcp-audit':{type:'string'},resume:{type:'boolean'},'validate-only':{type:'boolean'},verifier:{type:'string'},'verifier-sha256':{type:'string'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-differentiable-validation.mjs --registration <prospective JSON> --registration-sha256 <SHA256> --evidence <JSON> [--web-log <Host log>] [--mcp-audit <trusted optional observer JSONL>] [--resume] [--validate-only] [--verifier <independent Python script> --verifier-sha256 <SHA256>]');
  console.log('Each registration case declares its fresh project test-output workspace. Validate-only checks immutable files and schema without any Session/RPC. Resume observes saved job/attempt handles and original confirmations; uncertain admissions are never resubmitted. Optional verifier receives --evidence, --registration and --output.');process.exit(0);}
assert.ok(values.registration&&values.evidence&&/^[a-f0-9]{64}$/.test(values['registration-sha256']??''),'Explicit registration, exact SHA256 and evidence path required');
assert.equal(Boolean(values.verifier),Boolean(values['verifier-sha256']),'Pin an optional verifier with its exact SHA256');
const json=path=>readFile(path,'utf8').then(JSON.parse),hash=value=>createHash('sha256').update(value).digest('hex');
const registrationPath=resolve(values.registration),evidencePath=resolve(values.evidence),registrationBytes=await readFile(registrationPath);
assert.equal(hash(registrationBytes),values['registration-sha256'],'Prospective registration bytes changed');
const registration=JSON.parse(registrationBytes);assert.equal(registration.schema_version,1);assert.equal(registration.phase,'S12-R01-R02');
assert.ok(Array.isArray(registration.cases)&&registration.cases.length>0);assert.equal(new Set(registration.cases.map(row=>row.key)).size,registration.cases.length);
const inside=(base,target)=>{const path=relative(base,target);return path!==''&&!isAbsolute(path)&&path!=='..'&&!path.startsWith(`..${sep}`);};
const sourceHashes={},componentHashes={};
for(const [path,digest] of Object.entries(registration.source_file_hashes)){assert.ok(isAbsolute(path));assert.equal(hash(await readFile(path)),digest,'Frozen source changed: '+path);sourceHashes[path]=digest;}
for(const [name,digest] of Object.entries(registration.component_source_sha256)){const path=resolve(project,name);assert.ok(inside(project,path));assert.equal(hash(await readFile(path)),digest,'Registered component changed: '+name);componentHashes[name]=digest;}
for(const item of registration.cases){assert.match(item.key,/^[A-Za-z0-9_-]+$/);assert.ok(isAbsolute(item.workspace)&&inside(resolve(project,'test-output'),resolve(item.workspace)));
  assert.ok(item.original&&item.candidate?.model_path&&item.candidate.source);assert.ok(item.original.policy_artifact?item.original.original_id:item.original.material_path&&item.original.plant_config);
  assert.ok(Object.hasOwn(sourceHashes,resolve(item.candidate.model_path)));if(item.wrong_candidate)assert.ok(Object.hasOwn(sourceHashes,resolve(item.wrong_candidate.model_path)));
  assert.equal(item.validation_config.mode,'exact_rewrite');assert.equal(item.validation_config.partition,'validation');assert.equal(item.validation_config.steps,50);
  assert.deepEqual(item.validation_config.numeric,{atol:1e-6,rtol:1e-5});assert.deepEqual(item.validation_config.gradient,{atol:1e-5,rtol:1e-3,finite_difference_steps:[1e-4,1e-5,1e-6]});
  assert.equal(item.sampling_config.sampling_run,'full');assert.equal(item.sampling_config.sampling.steps,50);
}
const verifier=values.verifier?resolve(values.verifier):null;
const auditPath=values['mcp-audit']?resolve(values['mcp-audit']):null;
if(verifier)assert.equal(hash(await readFile(verifier)),values['verifier-sha256'],'Independent verifier bytes changed');
const preRegistered={registration_path:registrationPath,registration_sha256:hash(registrationBytes),registration,source_file_hashes:sourceHashes,component_source_sha256:componentHashes,
  driver_sha256:hash(await readFile(new URL(import.meta.url))),mcp_audit_path:auditPath,verifier,verifier_sha256:values['verifier-sha256']??null};
if(values['validate-only']){console.log(JSON.stringify({status:'PASS_REGISTRATION_ONLY',cases:registration.cases.map(row=>row.key),source_files:Object.keys(sourceHashes).length,component_files:Object.keys(componentHashes).length,native_submitted:false}));process.exit(0);}
let evidence;
if(values.resume)evidence=await json(evidencePath);
else{try{await readFile(evidencePath);assert.fail('Evidence already exists; inspect saved attempts and pass --resume, never overwrite');}catch(error){if(error.code!=='ENOENT')throw error;}
  evidence={schema_version:1,phase:'S12-R01-R02',status:'RUNNING',started_at:new Date().toISOString(),confirmation_source:'test_fixture',preRegistered,
    sessions:{},jobs:{},admissions:[],confirmations:[],confirmation_reuses:[],phase_requests:{},rejections:[],historical_phase_checks:[],checks:[]};}
assert.deepEqual(evidence.preRegistered,preRegistered,'Observer resume cannot revise source, candidates, gates or verifier');
if(values.resume){evidence.observer_history??=[];if(evidence.error)evidence.observer_history.push({status:evidence.status,error:evidence.error,resumed_at:new Date().toISOString()});delete evidence.error;delete evidence.finished_at;evidence.status='RUNNING';}
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const normalize=value=>{if(typeof value==='string')value=JSON.parse(value);assert.ok(value&&typeof value==='object'&&!Array.isArray(value),'Native result must be a JSON object (at most one serialized display layer)');return value;};
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');

async function artifact(workspace,id){const manifest=await json(resolve(workspace,'manifest.json')),entry=manifest.artifacts[id];assert.ok(entry,'Artifact is absent from its actual workspace index');
  const root=resolve(workspace,entry.path);assert.ok(inside(resolve(workspace,'artifacts'),root));const bytes=await readFile(resolve(root,'artifact.json'));assert.equal(hash(bytes),entry.metadata_file_sha256,'Artifact metadata version changed');const metadata=JSON.parse(bytes);assert.equal(metadata.id,id);
  return {root,metadata,member:async name=>{const path=resolve(root,name);assert.ok(inside(root,path));const bytes=await readFile(path);assert.ok(bytes.length<=64*1024*1024,'Bounded evidence member read exceeded 64 MiB');assert.equal(hash(bytes),metadata.files[name],'Artifact file changed: '+name);return JSON.parse(bytes);}};}
async function audit(sessionId,matlab){
  if(!auditPath)return {status:'NOT_AUDITED',scope:'No actual tool-call count inferred from a numerical job, backend label or synthetic event.'};
  let bytes;
  if(evidence.mcp_audit){bytes=await readFile(evidence.mcp_audit.path);assert.equal(hash(bytes),evidence.mcp_audit.sha256);assert.equal(bytes.length,evidence.mcp_audit.through_bytes);
    const live=await readFile(auditPath);assert.ok(live.length>=bytes.length);assert.ok(live.subarray(0,bytes.length).equals(bytes),'Original live observer prefix changed');}
  else try{bytes=await readFile(auditPath);}catch(error){if(error.code!=='ENOENT')throw error;bytes=Buffer.alloc(0);}
  const all=bytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
  const rows=all.filter(row=>row.source==='trusted_test_observer'&&row.sessionId===sessionId),starts=rows.filter(row=>row.event==='started'),returns=rows.filter(row=>row.event==='returned');
  if(!matlab){assert.equal(rows.length,0,'Python-only validation must not call MATLAB');return {status:'PASS',business_batch_calls:0,fixed_guard_probe_calls:0,total_actual_mcp_calls:0,rows};}
  assert.equal(rows.length,6);assert.equal(starts.length,3);assert.equal(returns.length,3);const seen=new Set();let batches=0,probes=0;
  for(const row of starts){const id=row.callId??row.rootCallId;assert.ok(id&&!seen.has(id));seen.add(id);assert.equal(row.name,'mcp__matlab__evaluate_matlab_code');
    const paired=returns.filter(other=>(other.callId??other.rootCallId)===id);assert.equal(paired.length,1);assert.equal(paired[0].isError,false);
    const call=[row.callId,row.rootCallId].find(value=>value?.startsWith('matlab-batch-')||value?.startsWith('matlab-probe-'));
    if(call?.startsWith('matlab-batch-'))batches++;else if(call?.startsWith('matlab-probe-'))probes++;else assert.fail('Unknown actual MATLAB invocation identity: '+id);}
  assert.equal(batches,1);assert.equal(probes,2);return {status:'PASS',business_batch_calls:1,fixed_guard_probe_calls:2,total_actual_mcp_calls:3,rows,
    source:'Actual trusted test-only around-next observer; started/returned pairs bound to this exact native Session, never numerical substitute events'};
}
async function freezeAudit(){
  if(!auditPath)return;
  if(evidence.mcp_audit){const bytes=await readFile(evidence.mcp_audit.path),live=await readFile(auditPath);assert.equal(hash(bytes),evidence.mcp_audit.sha256);assert.equal(bytes.length,evidence.mcp_audit.through_bytes);
    assert.ok(live.length>=bytes.length&&live.subarray(0,bytes.length).equals(bytes),'The frozen R01 actual observer prefix changed');return;}
  const bytes=await readFile(auditPath),path=resolve(dirname(evidencePath),'R01audit-snapshot.jsonl');
  try{await writeFile(path,bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;assert.ok((await readFile(path)).equals(bytes),'Never replace an earlier actual observer snapshot');}
  const rows=bytes.toString('utf8').split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
  for(const checked of evidence.checks)assert.deepEqual(rows.filter(row=>row.source==='trusted_test_observer'&&row.sessionId===checked.sessionId),checked.mcp_actual.rows);
  evidence.mcp_audit={path,sha256:hash(bytes),snapshot_path:path,snapshot_sha256:hash(bytes),source_live_path:auditPath,through_bytes:bytes.length,source_live_prefix_sha256:hash(bytes),rows:rows.length,
    source:'Exclusive immutable R01 prefix snapshot of the actual live observer. Later R03 appends are outside through_bytes and never revise these per-Session tool pairs.'};await save();
}

try{
  // Marker and Windows ACL preparation precede the first native Session.
  for(const item of registration.cases){const workspace=resolve(item.workspace);await prepareSmokeWorkspace(workspace);
    if(!evidence.sessions[item.key]){try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'New case needs an empty isolated workspace');}catch(error){if(error.code!=='ENOENT')throw error;}}
    const markerPath=resolve(workspace,'.case-plans/test-fixture.json'),marker={kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true,
      source:{kind:'S12_prospective_rewrite_validation',case_key:item.key,registration_sha256:preRegistered.registration_sha256,truth:item.truth??'Synthetic program-model test fixture; not measured hardware data'}};
    try{assert.deepEqual(await json(markerPath),marker,'Original fixture marker changed');}catch(error){if(error.code!=='ENOENT')throw error;assert.ok(!evidence.sessions[item.key],'Never recreate a lost marker for an existing Session');await writeSmokeEvidence(markerPath,marker);}
    evidence.fixture_markers??={};const record={path:markerPath,sha256:hash(await readFile(markerPath)),source:marker.source};
    if(evidence.fixture_markers[item.key])assert.deepEqual(evidence.fixture_markers[item.key],record);else evidence.fixture_markers[item.key]=record;
  }await save();
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  for(const item of registration.cases){const workspace=resolve(item.workspace),owner='lightweight',prefix=item.key+'/',key=part=>prefix+part;
    const session=async()=>{assert.equal(hash(await readFile(evidence.fixture_markers[item.key].path)),evidence.fixture_markers[item.key].sha256);
      if(!evidence.sessions[item.key]){const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});assert.equal(created.agentPreset,'control-lightweight');evidence.sessions[item.key]=created.sessionId;await save();}return evidence.sessions[item.key];};
    const events=async()=>{const sessionId=await session();let page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:500}});const cursor=page.cursor,pages=[page];let before;
      while(page.hasMore){assert.ok(page.records.length);const next=Math.min(...page.records.map(row=>row.event.seq));assert.ok(before===undefined||next<before,'Native pagination must advance');before=next;
        page=await rpc('session/events',{request:{address:{kind:'session',sessionId},throughSeq:cursor,beforeSeq:before,maxMessages:500}});assert.equal(page.cursor,cursor);pages.push(page);}
      const records=pages.flatMap(row=>row.records).sort((a,b)=>a.event.seq-b.event.seq);assert.equal(records.length,cursor+1);records.forEach((row,index)=>assert.equal(row.event.seq,index));assert.equal(page.hasMore,false);
      const path=resolve(dirname(evidencePath),'rewrite-'+item.key+'-session-events-'+cursor+'.json');await writeSmokeEvidence(path,{source:'Actual native complete session/events pages at one cursor',sessionId,cursor,pages,records,hasMore:false});
      evidence.session_events??={};evidence.session_events[item.key]={path,sha256:hash(await readFile(path)),sessionId,cursor,records:records.length,pages:pages.length,hasMore:false};await save();return records;};
    const status=async row=>command(rpc,await session(),`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);
    const assertRecord=(row,state,request)=>{const record=state.record;assert.equal(record.attemptId,row.attemptId);assert.equal(record.nativeJobId,row.jobId);assert.equal(record.owner,row.sessionId);assert.equal(record.agentName,owner);
      const {agent,...nativeRequest}=request;assert.equal(agent,owner);assert.deepEqual(record.request,nativeRequest,'Observe the exact original request, not revised current scope heads');return record;};
    const snapshot=async()=>{let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(name=>/^[0-9a-f-]{36}$/.test(name)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}
      const history=await command(rpc,await session(),`/control-history ${JSON.stringify({workspace})}`),records=await events();return {attempts,owned_history:history.attempts.map(row=>row.attemptId).sort(),native_jobs:records.filter(row=>row.event.type==='job/registered').map(row=>row.event.data.jobId)};};
    const deny=async(label,operation,inputs,config)=>{label=key(label);if(evidence.rejections.some(row=>row.label===label))return;
      const before=await snapshot(),request={agent:owner,operation,workspace,inputs,config},response=await rpc('commands/execute',{agentId:await session(),line:'/control '+JSON.stringify(request),submittedAttachments:[]});
      assert.equal(response.result.kind,'error','Unconfirmed/changed request was admitted');assert.match(response.result.text,/confirm|plan|version|review/i);const after=await snapshot();assert.deepEqual(after,before,'Rejected request created an attempt/history/native job');
      evidence.rejections.push({label,status:'PASS_ZERO_ADMISSION',request,reason:response.result.text,before,after,confirmation_source:'test_fixture'});await save();};
    const stage=async(part,operation,inputs={},config={},expectedFailure=false)=>{const name=key(part),request={agent:owner,operation,workspace,inputs,config},signature=hash(JSON.stringify(request));let row=evidence.jobs[name];
      if(row){assert.equal(row.signature,signature);assert.equal(row.expectedFailure,expectedFailure);assert.deepEqual(row.request,request);assert.ok(row.jobId&&row.attemptId,'Uncertain admission must be reconciled; do not resubmit');}
      else{row=evidence.jobs[name]={key:name,operation,owner,sessionId:await session(),request,signature,expectedFailure,status:'ADMISSION_PENDING'};await save();
        const started=await command(rpc,row.sessionId,'/control '+JSON.stringify(request));assert.ok(started.jobId&&started.attemptId);Object.assign(row,{jobId:started.jobId,attemptId:started.attemptId,status:'RUNNING'});
        evidence.admissions.push({key:name,caseKey:item.key,operation,owner,sessionId:row.sessionId,jobId:row.jobId,attemptId:row.attemptId,signature});await save();}
      const deadline=Date.now()+420000;
      for(;;){const state=await status(row),record=assertRecord(row,state,request);row.last_observation={at:new Date().toISOString(),state:record.state,native_status:state.job?.status??null};
        if(['completed','failed','killed'].includes(record.state)&&record.outcome?.cleanupVerified===true&&(!state.job||['completed','failed','killed'].includes(state.job.status))){row.status=record.state;row.record=record;row.rawControlStatus=state;
          assert.equal(record.state,expectedFailure?'failed':'completed',`${name} actual terminal ${record.state}: ${record.detail??''}`);const domainResult=normalize(record.result);if(row.result)assert.deepEqual(row.result,domainResult,'Saved native outcome changed');row.result=domainResult;await save();
          if(expectedFailure)assert.equal(row.result.status,'rejected','Expected actual domain rejection, not a worker crash');return row.result;}
        assert.notEqual(record.state,'interrupted','Original job ownership lost; inspect saved attempt instead of replacing it');
        if(Date.now()>=deadline){row.observer_deadline_reached=true;await save();throw new Error(`Observer deadline; original job ${row.jobId}/${row.attemptId} remains owned. Resume observes this handle; never resubmit.`);}
        await new Promise(done=>setTimeout(done,400));}
    };
    const phaseRefs=async(phase,target,operation,inputs,config,scopes)=>{const phaseKey=key(phase),targetKey=key(target),admitted=evidence.jobs[targetKey];
      if(!admitted){const requirements=await confirmationRequirements({operation,workspace,inputs,config});assert.deepEqual(requirements.map(row=>row.scope),scopes);const refs={},states={};
        for(const requirement of requirements){const plan=await stage(phase+'-plan-'+requirement.scope,'propose_case_plan',{}, {scope:requirement.scope,payload:requirement.expectedPayload,bindings:requirement.bindings});
          refs[requirement.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};const current=await command(rpc,await session(),`/control-plan-status ${JSON.stringify({workspace,...refs[requirement.scope]})}`);
          assert.equal(current.session_id,await session());assert.equal(current.scope,requirement.scope);assert.equal(current.plan_id,plan.plan_id);assert.equal(current.version_hash,plan.version_hash);states[requirement.scope]=current;}
        if(Object.values(states).some(row=>row.confirmation_state==='pending'))await deny(phase+'-pending',operation,inputs,{...config,plan_refs:refs});
        for(const requirement of requirements){let confirmed=states[requirement.scope],reused=confirmed.confirmation_state==='confirmed';
          if(!reused){assert.equal(confirmed.confirmation_state,'pending');confirmed=await command(rpc,await session(),`/control-confirm ${JSON.stringify({workspace,...refs[requirement.scope]})}`);}
          assert.equal(confirmed.session_id,await session());assert.equal(confirmed.confirmation_source,'test_fixture');assert.equal(confirmed.source.kind,'native_user_command');
          const receipt={phase:phaseKey,caseKey:item.key,owner,sessionId:confirmed.session_id,scope:requirement.scope,...refs[requirement.scope],confirmation_source:'test_fixture',source:confirmed.source};
          const old=evidence.confirmations.find(row=>row.phase===phaseKey&&row.scope===requirement.scope);if(old)assert.deepEqual(old,receipt);else evidence.confirmations.push(receipt);
          if(reused)evidence.confirmation_reuses.push({...receipt,verified_via:'control-plan-status for this exact newly proposed requirement'});await save();}
        const request={agent:owner,operation,workspace,inputs,config:{...config,plan_refs:refs}};if(evidence.phase_requests[phaseKey])assert.deepEqual(evidence.phase_requests[phaseKey],request);else evidence.phase_requests[phaseKey]=request;await save();return refs;
      }
      // A later phase may have advanced a scope. Reuse the original signed
      // admission/proposal/command snapshots, never reauthorize old work.
      assert.ok(admitted.jobId&&admitted.attemptId);const state=await status(admitted),request=evidence.phase_requests[phaseKey];assert.ok(request);const record=assertRecord(admitted,state,request);
      assert.equal(hash(JSON.stringify(request)),admitted.signature);assert.equal(request.operation,operation);assert.deepEqual(request.inputs,inputs);const {plan_refs:refs,...actualConfig}=request.config;assert.deepEqual(actualConfig,config);
      const records=await events(),sources=[];
      for(const scope of scopes){const reference=refs[scope],receipt=evidence.confirmations.find(row=>row.phase===phaseKey&&row.scope===scope);assert.ok(receipt);assert.equal(receipt.sessionId,await session());assert.equal(receipt.owner,owner);
        assert.equal(receipt.plan_id,reference.plan_id);assert.equal(receipt.version_hash,reference.version_hash);const proposal=evidence.jobs[key(phase+'-plan-'+scope)];assert.ok(proposal?.jobId&&proposal?.attemptId&&proposal.result);
        const old=await status(proposal),proposed=assertRecord(proposal,old,proposal.request);assert.equal(proposed.state,'completed');const proposalResult=normalize(proposed.result);assert.equal(proposalResult.plan_id,reference.plan_id);assert.equal(proposalResult.version_hash,reference.version_hash);
        const run=records[receipt.source.seq]?.event;assert.equal(run?.type,'command/run');assert.equal(run.data.name,'control-confirm');assert.equal(run.data.commandId,receipt.source.command_id);assert.equal(run.data.source.kind,'user');assert.deepEqual(JSON.parse(run.data.args),{workspace,...reference});
        const done=records.find(row=>row.event.type==='command/done'&&row.event.data.commandId===run.data.commandId)?.event;assert.equal(done?.data.kind,'success');const confirmed=JSON.parse(done.data.text);
        assert.equal(confirmed.session_id,await session());assert.equal(confirmed.scope,scope);assert.equal(confirmed.plan_id,reference.plan_id);assert.equal(confirmed.version_hash,reference.version_hash);assert.deepEqual(confirmed.source,receipt.source);assert.equal(confirmed.confirmation_source,'test_fixture');
        sources.push({scope,...reference,confirmation_source:'test_fixture',source:receipt.source});}
      if(['completed','failed'].includes(record.state)&&record.result){const value=normalize(record.result);if(value.artifact_id){const saved=await artifact(workspace,value.artifact_id);if(saved.metadata.files['report.json'])assert.deepEqual((await saved.member('report.json')).confirmation_sources,sources);}}
      evidence.historical_phase_checks.push({caseKey:item.key,phase:phaseKey,target:targetKey,jobId:admitted.jobId,attemptId:admitted.attemptId,request_sha256:admitted.signature,state:record.state,confirmation_sources:sources,verified_via:'Original request/proposal attempts and actual historical command/run/done; current scope head not read'});await save();return refs;
    };

    let plant,policy;
    if(item.original.policy_artifact){const imported=await stage('import-policy','import_artifact',{artifact_path:item.original.policy_artifact});policy=imported.artifact_id;assert.equal(policy,item.original.policy_id);assert.equal(imported.kind,'external_imitation_policy');plant=item.original.original_id;
      const source=await artifact(workspace,policy),original=await artifact(workspace,plant);assert.equal(hash(await readFile(resolve(source.root,'artifact.json'))),item.original.policy_metadata_sha256);assert.equal(hash(await readFile(resolve(original.root,'artifact.json'))),item.original.original_metadata_sha256);
    }else{const material=await stage('material','register_material',{local_path:item.original.material_path},{source_kind:'path',source_metadata:{kind:'S12_fixed_synthetic_original',registration_sha256:preRegistered.registration_sha256,measured:false}});
      const original=await stage('plant','register_plant',{material:material.artifact_id},item.original.plant_config);assert.equal(original.kind,'plant_model');plant=original.artifact_id;}
    // Register the negative first: registration activates nominal_model and
    // would otherwise stale an already-published positive child receipt.
    const wrongCandidate=item.wrong_candidate?await stage('wrong-candidate','register_python_model',{model_path:item.wrong_candidate.model_path},{source:item.wrong_candidate.source}):null;
    if(wrongCandidate)assert.equal(wrongCandidate.status,'verified');
    const candidate=await stage('candidate','register_python_model',{model_path:item.candidate.model_path},{source:item.candidate.source});assert.equal(candidate.status,'verified');
    const sampleInputs={model:plant},sampleConfig=structuredClone(item.sampling_config),sampleRefs=await phaseRefs('sample','sample','sample_model',sampleInputs,sampleConfig,['signals','sampling']);
    const sample=await stage('sample','sample_model',sampleInputs,{...sampleConfig,plan_refs:sampleRefs});assert.equal(sample.kind,'sampled_dataset');assert.equal(sample.status,'verified');
    const changedSample=structuredClone(sampleConfig);changedSample.sampling.seed+=1;await deny('sample-changed-summary','sample_model',sampleInputs,{...changedSample,plan_refs:sampleRefs});
    const sampled=await artifact(workspace,sample.artifact_id);assert.ok(sampled.metadata.parents.some(row=>row.id===plant));assert.equal(sampled.metadata.source.model,plant);assert.equal(sampled.metadata.source.kind,'plant_simulator');assert.equal(sampled.metadata.source.measured,false);
    if(item.expected_split)assert.deepEqual(await sampled.member('split.json'),item.expected_split);if(item.expected_samples)assert.equal(sampled.metadata.validation.samples,item.expected_samples);
    const validationInputs={evaluation_model:plant,model:candidate.artifact_id,dataset:sample.artifact_id},validationConfig=structuredClone(item.validation_config),refs=await phaseRefs('validation','validation','validate_differentiable_model',validationInputs,validationConfig,['signals']);
    const validated=await stage('validation','validate_differentiable_model',validationInputs,{...validationConfig,plan_refs:refs});assert.equal(validated.kind,'differentiable_validation');assert.equal(validated.status,'verified');
    const changedValidation=structuredClone(validationConfig);changedValidation.steps-=1;await deny('validation-changed-summary','validate_differentiable_model',validationInputs,{...changedValidation,plan_refs:refs});
    const receipt=await artifact(workspace,validated.artifact_id),report=await receipt.member('report.json');assert.equal(receipt.metadata.activation,'none');assert.equal(report.qualified_for_pgac,true);assert.equal(report.exact_equivalence_pass,true);assert.equal(report.gradient_checks_passed,true);
    if(item.frozen_physical_interface)assert.deepEqual(report.physical_interface,item.frozen_physical_interface);
    assert.equal(report.bindings.original.id,plant);assert.equal(report.bindings.candidate.id,candidate.artifact_id);assert.equal(report.bindings.dataset.id,sample.artifact_id);assert.equal(report.reference_partition,'validation');assert.equal(report.test_used_for_selection,false);
    assert.deepEqual(new Set(receipt.metadata.parents.map(row=>row.id)),new Set([plant,candidate.artifact_id,sample.artifact_id]));assert.ok(report.trajectories.every(row=>row.completed_steps===50&&row.teacher_forcing_after_initial_state===false));
    assert.ok(report.gradient_probes.every(row=>row.agreement_by_h.every(Boolean)));assert.equal(report.numeric.mixed_batch_next_x.within_tolerance,true);assert.ok(report.numeric.mixed_batch_next_x.rows>1);
    const checked={caseKey:item.key,workspace,sessionId:await session(),...(policy?{policy}:{}),plant,candidate:candidate.artifact_id,dataset:sample.artifact_id,validation:validated.artifact_id};
    if(wrongCandidate){const wrong=wrongCandidate;
      const inputs={evaluation_model:plant,model:wrong.artifact_id,dataset:sample.artifact_id},wrongRefs=await phaseRefs('wrong-validation','wrong-validation','validate_differentiable_model',inputs,validationConfig,['signals']);
      const rejected=await stage('wrong-validation','validate_differentiable_model',inputs,{...validationConfig,plan_refs:wrongRefs},true),negative=await artifact(workspace,rejected.artifact_id),negativeReport=await negative.member('report.json');
      assert.equal(negative.metadata.validation.status,'rejected');assert.equal(negative.metadata.activation,'none');assert.equal(negativeReport.gradient_checks_passed,true);assert.equal(negativeReport.exact_equivalence_pass,false);assert.equal(negativeReport.qualified_for_pgac,false);
      checked.negative={candidate:wrong.artifact_id,validation:rejected.artifact_id,jobId:evidence.jobs[key('wrong-validation')].jobId,attemptId:evidence.jobs[key('wrong-validation')].attemptId};}
    // On a clean run this also records the same historical proof required by
    // --resume, after later signal versions have legitimately advanced.
    await phaseRefs('sample','sample','sample_model',sampleInputs,sampleConfig,['signals','sampling']);await phaseRefs('validation','validation','validate_differentiable_model',validationInputs,validationConfig,['signals']);
    const records=await events();assert.equal(records.filter(row=>row.event.type.startsWith('turn/')).length,0,'Native acceptance must not call an LLM');
    checked.mcp_actual=await audit(checked.sessionId,String(sampled.metadata.source.backend).startsWith('matlab'));
    checked.status='PASS';checked.confirmation_source='test_fixture';const old=evidence.checks.find(row=>row.caseKey===item.key);if(old)assert.deepEqual(old,checked);else evidence.checks.push(checked);await save();
  }
  await freezeAudit();
  for(const [path,digest] of Object.entries(sourceHashes))assert.equal(hash(await readFile(path)),digest,'Original fixture source changed during execution');
  for(const [name,digest] of Object.entries(componentHashes))assert.equal(hash(await readFile(resolve(project,name))),digest,'Production component changed during execution');
  evidence.source_hashes_after={source_file_hashes:sourceHashes,component_source_sha256:componentHashes};evidence.native_result='PASS';evidence.status=verifier?'NATIVE_PASS_AWAITING_INDEPENDENT_VERIFIER':'PASS';await save();
  if(verifier){const output=resolve(dirname(evidencePath),'rewrite-independent-numeric.json'),inputBindings={registration_sha256:preRegistered.registration_sha256,
      artifacts:evidence.checks.map(({caseKey,workspace,sessionId,policy,plant,candidate,dataset,validation,negative})=>({caseKey,workspace,sessionId,...(policy?{policy}:{}),plant,candidate,dataset,validation,...(negative?{negative}:{})})),
      native_admissions:evidence.admissions.map(({key,jobId,attemptId,signature})=>({key,jobId,attemptId,signature}))};
    if(evidence.independent_numeric_verification){const previous=evidence.independent_numeric_verification;assert.equal(previous.path,output);assert.equal(previous.verifier,verifier);assert.equal(previous.verifier_sha256,preRegistered.verifier_sha256);
      assert.deepEqual(previous.input_bindings,inputBindings,'Completed verifier belongs to different registration/artifacts/admissions');assert.equal(hash(await readFile(output)),previous.sha256);assert.equal((await json(output)).status,'PASS');}
    else{assert.ok(!(await readFile(output).then(()=>true,error=>{if(error.code==='ENOENT')return false;throw error;})),'Unrecorded verifier output must be reconciled; never rerun or overwrite it');
      const stdout=execFileSync(python,['-I','-B',verifier,'--evidence',evidencePath,'--registration',registrationPath,'--output',output,'--final-ready'],{encoding:'utf8',windowsHide:true,maxBuffer:16*1024*1024,env:{...process.env,PYTHONPATH:'',PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'}});
      const result=await json(output);assert.equal(result.status,'PASS');evidence.independent_numeric_verification={path:output,sha256:hash(await readFile(output)),verifier,verifier_sha256:preRegistered.verifier_sha256,input_bindings:inputBindings,stdout};}
    evidence.status='PASS';}
  else evidence.independent_numeric_verification={status:'NOT_REQUESTED_BY_DRIVER',scope:'Native artifact/report assertions only; independent numeric verifier remains a separate gate.'};
  evidence.finished_at=new Date().toISOString();await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,cases:evidence.checks.length,jobs:evidence.admissions.length,zero_admissions:evidence.rejections.length,LLM_turns:0}));
}catch(error){evidence.status='FAIL';evidence.error={name:error.name,message:error.message,stack:error.stack};evidence.finished_at=new Date().toISOString();await save();throw error;}
