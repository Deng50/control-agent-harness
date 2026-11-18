/** S11 P01/P02/P03 native acceptance preparation; never starts a Host. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {constants} from 'node:fs';
import {copyFile,mkdir,readFile,readdir} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{workspace:{type:'string'},evidence:{type:'string'},registration:{type:'string'},'registration-sha256':{type:'string'},
  fixtures:{type:'string'},verifier:{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-conditional-pinn.mjs --workspace <isolated test-output> --evidence <json> --registration <predeclared JSON> --registration-sha256 <exact SHA256> [--verifier <versioned independent verifier>] [--fixtures <S00>] [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values.workspace&&values.evidence&&values.registration&&/^[a-f0-9]{64}$/.test(values['registration-sha256']??''),'Explicit workspace/evidence/registration and exact SHA required');
const workspace=resolve(values.workspace),evidencePath=resolve(values.evidence),registrationPath=resolve(values.registration),registrationSha=values['registration-sha256'];
const inside=relative(resolve(project,'test-output'),workspace);assert.ok(inside&&!isAbsolute(inside)&&inside!=='..'&&!inside.startsWith(`..${sep}`));
const json=path=>readFile(path,'utf8').then(JSON.parse),hash=value=>createHash('sha256').update(value).digest('hex');
const registrationBytes=await readFile(registrationPath);assert.equal(hash(registrationBytes),registrationSha);const registration=JSON.parse(registrationBytes);assert.equal(registration.schema_version,1);
const runRoot=resolve(project,'test-output/v2-development/20261117T002006'),fixtureRoot=resolve(values.fixtures??resolve(runRoot,'fixtures'));
for(const [name,digest] of Object.entries(registration.fixture_hashes))assert.equal(hash(await readFile(resolve(fixtureRoot,name))),digest,'Frozen fixture changed: '+name);
const packet=resolve(registration.packet),packetFiles={};for(const name of ['reference.csv','collocation.csv','problem-input.json'])packetFiles[name]=hash(await readFile(resolve(packet,name)));
assert.equal(packetFiles['reference.csv'],registration.fixture_hashes['F04_analytic_trajectories.csv']);assert.equal(packetFiles['collocation.csv'],registration.fixture_hashes['F04_training_physics_collocation.csv']);
assert.deepEqual(await json(resolve(packet,'problem-input.json')),registration.problem_config);assert.equal(registration.problem_config.reference_file,'source/reference.csv');assert.equal(registration.problem_config.collocation_file,'source/collocation.csv');
const verifier=resolve(values.verifier??resolve(runRoot,'S11/verify-native-pinn.py')),verifierSha=hash(await readFile(verifier));
const preRegistered={registration_path:registrationPath,registration_sha256:registrationSha,registration,packet_file_hashes:packetFiles,verifier,verifier_sha256:verifierSha,
  scope:'Actual conditional smooth NN training, independent SI/analytic/export verification, physics-off diagnostic and explicit non-Plant capability rejection. No general PDE, transition, MPC solution, PGAC, LLM or MATLAB execution.'};
let evidence=values.resume?await json(evidencePath):{schema_version:1,stage:'S11',groups:['P01','P02','P03'],status:'RUNNING',workspace,preRegistered,
  started_at:new Date().toISOString(),confirmation_source:'test_fixture',sessions:{},jobs:{},admissions:[],confirmations:[],confirmation_reuses:[],rejections:[],boundary_checks:[],checks:[]};
assert.equal(evidence.workspace,workspace);assert.deepEqual(evidence.preRegistered,preRegistered,'Observer resume cannot revise candidates, files, numerical gates or verifier');
if(values.resume){evidence.observer_history??=[];if(evidence.error)evidence.observer_history.push({status:evidence.status,error:evidence.error,resumed_at:new Date().toISOString()});delete evidence.error;delete evidence.finished_at;evidence.status='RUNNING';}
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();await prepareSmokeWorkspace(workspace);
if(Object.keys(evidence.sessions).length===0){try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'New registration requires an empty isolated workspace');}catch(error){if(error.code!=='ENOENT')throw error;}}
const markerPath=resolve(workspace,'.case-plans/test-fixture.json'),markerSource={kind:'S11_preregistered_conditional_thermal_acceptance',registration_sha256:registrationSha,packet,packet_file_hashes:packetFiles};let marker;
try{marker=await json(markerPath);}catch(error){if(error.code!=='ENOENT')throw error;assert.equal(Object.keys(evidence.sessions).length,0,'Never recreate a lost original authorization marker for an existing Session');
  marker={kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true,source:markerSource};await writeSmokeEvidence(markerPath,marker);}
assert.equal(marker.kind,'control-case-plan-test-fixture-v1');assert.equal(marker.workspace,workspace);assert.equal(marker.confirmation_source,'test_fixture');assert.equal(marker.synthetic,true);assert.deepEqual(marker.source,markerSource);
const markerRecord={path:markerPath,sha256:hash(await readFile(markerPath)),source:markerSource};if(evidence.fixture_marker)assert.deepEqual(evidence.fixture_marker,markerRecord);else{evidence.fixture_marker=markerRecord;await save();}
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const portableRoot=resolve(project,'..','control-agent-harness-pinn-native-portable',hash(workspace).slice(0,12));assert.ok(relative(project,portableRoot).startsWith('..'+sep));
const runVerifier=args=>JSON.parse(execFileSync(python,['-I','-B',verifier,...args],{cwd:portableRoot,encoding:'utf8',windowsHide:true,maxBuffer:16*1024*1024,
  env:{...process.env,PYTHONPATH:'',PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1'}}));

try{
  await mkdir(portableRoot,{recursive:true});const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  const session=async key=>{assert.deepEqual(await json(markerPath),marker,'Retain fixture source on every Session entry/resume');
    if(!evidence.sessions[key]){const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:key==='model'?'control-model':'control-controller'}});assert.equal(created.agentPreset,key==='model'?'control-model':'control-controller');evidence.sessions[key]=created.sessionId;await save();}return evidence.sessions[key];};
  const events=async key=>{const id=await session(key);let page=await rpc('session/events',{request:{address:{kind:'session',sessionId:id},maxMessages:500}});const cursor=page.cursor,pages=[page];let before;
    while(page.hasMore){assert.ok(page.records.length);const next=Math.min(...page.records.map(row=>row.event.seq));assert.ok(before===undefined||next<before,'Pagination must advance to older records');before=next;
      page=await rpc('session/events',{request:{address:{kind:'session',sessionId:id},throughSeq:cursor,beforeSeq:before,maxMessages:500}});assert.equal(page.cursor,cursor);pages.push(page);}
    assert.equal(page.hasMore,false);const records=pages.flatMap(row=>row.records).sort((a,b)=>a.event.seq-b.event.seq);const seen=new Set();for(const row of records){assert.ok(!seen.has(row.event.seq));seen.add(row.event.seq);}
    assert.equal(records.length,cursor+1);records.forEach((row,index)=>assert.equal(row.event.seq,index));
    const path=resolve(dirname(evidencePath),`conditional-pinn-${hash(workspace).slice(0,12)}-${key}-session-events.json`);
    await writeSmokeEvidence(path,{source:'Actual native session/events pages at one immutable cursor; records below are their sorted complete union',sessionId:id,cursor,pages,records,hasMore:false});
    evidence.session_events??={};evidence.session_events[key]={path,sha256:hash(await readFile(path)),sessionId:id,cursor,records:records.length,pages:pages.length,hasMore:false};await save();return records;};
  const snapshot=async key=>{const id=await session(key);let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(name=>/^[0-9a-f-]{36}$/.test(name)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}
    const history=await command(rpc,id,`/control-history ${JSON.stringify({workspace})}`),records=await events(key);return {attempts,owned_history:history.attempts.map(row=>row.attemptId).sort(),native_jobs:records.filter(row=>row.event.type==='job/registered').map(row=>row.event.data.jobId)};};
  const deny=async(label,key,operation,inputs,config)=>{if(evidence.rejections.some(row=>row.label===label))return;const before=await snapshot(key),response=await rpc('commands/execute',{agentId:await session(key),line:`/control ${JSON.stringify({agent:key,operation,workspace,inputs,config})}`,submittedAttachments:[]});
    assert.equal(response.result.kind,'error');assert.match(response.result.text,/confirm|plan|version|review/i);const after=await snapshot(key);assert.deepEqual(after,before);evidence.rejections.push({label,status:'PASS_ZERO_ADMISSION',reason:response.result.text,before,after});await save();};
  const stage=async(key,owner,operation,inputs={},config={},expectedFailure=false)=>{const request={agent:owner,operation,workspace,inputs,config},signature=hash(JSON.stringify(request));let row=evidence.jobs[key];
    if(row){assert.equal(row.signature,signature);assert.equal(row.expectedFailure,expectedFailure);assert.ok(row.jobId,'Uncertain admission must be reconciled from saved history; never submit twice');if(row.result)return row.result;if(row.failure)return row.failure;}
    else{row=evidence.jobs[key]={operation,signature,owner,sessionId:await session(owner),status:'ADMISSION_PENDING',expectedFailure};await save();const started=await command(rpc,row.sessionId,`/control ${JSON.stringify(request)}`);Object.assign(row,{jobId:started.jobId,attemptId:started.attemptId,status:'RUNNING'});evidence.admissions.push({key,operation,owner,jobId:row.jobId,attemptId:row.attemptId});await save();}
    for(;;){const state=await command(rpc,row.sessionId,`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);
      if(['completed','failed','killed'].includes(state.record.state)){row.status=state.record.state;row.record=state.record;await save();
        if(expectedFailure){assert.equal(row.status,'failed','The explicit capability-negative job must fail, never solve/export');const text=[state.record.detail,...(state.output?.chunks??[]).map(chunk=>chunk.text)].filter(Boolean).join('\n');assert.match(text,/conditional|state-transition|MPC\/PGAC/i);
          const path=resolve(dirname(evidencePath),`conditional-pinn-negative-${key}.json`);await writeSmokeEvidence(path,state);row.failure={status:'PASS_AUTHORIZED_NON_PLANT_REJECTION',jobId:row.jobId,attemptId:row.attemptId,detail:state.record.detail??null,evidence:path};await save();return row.failure;}
        assert.equal(row.status,'completed',`${operation} ${row.attemptId}: ${state.record.detail??row.status}`);row.result=state.record.result;assert.ok(row.result);await save();return row.result;}
      assert.notEqual(state.record.state,'interrupted','Original execution lost ownership; inspect/recover this saved attempt, never admit a replacement');await new Promise(done=>setTimeout(done,400));}
  };
  const confirm=async(key,owner,operation,inputs,config,scopes)=>{const request={operation,workspace,inputs,config},requirements=await confirmationRequirements(request);assert.deepEqual(requirements.map(row=>row.scope),scopes);const refs={},statuses={};
    for(const requirement of requirements){const plan=await stage(key+'-plan-'+requirement.scope,owner,'propose_case_plan',{}, {scope:requirement.scope,payload:requirement.expectedPayload,bindings:requirement.bindings});refs[requirement.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};
      const status=await command(rpc,await session(owner),`/control-plan-status ${JSON.stringify({workspace,...refs[requirement.scope]})}`);assert.equal(status.session_id,await session(owner));assert.equal(status.scope,requirement.scope);assert.equal(status.plan_id,plan.plan_id);assert.equal(status.version_hash,plan.version_hash);statuses[requirement.scope]=status;}
    if(Object.values(statuses).some(row=>row.confirmation_state==='pending'))await deny(key+'-before-confirmation',owner,operation,inputs,{...config,plan_refs:refs});
    else assert.ok(evidence.rejections.some(row=>row.label===key+'-before-confirmation'),'Do not repeat denial with already-confirmed references or invent its historical result');
    for(const requirement of requirements){let result=statuses[requirement.scope],reused=result.confirmation_state==='confirmed';if(!reused){assert.equal(result.confirmation_state,'pending');result=await command(rpc,await session(owner),`/control-confirm ${JSON.stringify({workspace,...refs[requirement.scope]})}`);}
      assert.equal(result.confirmation_source,'test_fixture');assert.equal(result.session_id,await session(owner));assert.equal(result.source.kind,'native_user_command');const record={phase:key,owner,sessionId:result.session_id,scope:requirement.scope,...refs[requirement.scope],confirmation_source:'test_fixture',source:result.source};
      const old=evidence.confirmations.find(row=>row.phase===key&&row.scope===requirement.scope);if(old)assert.deepEqual(old,record);else evidence.confirmations.push(record);if(reused)evidence.confirmation_reuses.push({...record,verified_via:'control-plan-status',at:new Date().toISOString()});await save();}return refs;};
  const phaseRefs=async(phase,target,owner,operation,inputs,config,scopes)=>{
    evidence.phase_requests??={};const admitted=evidence.jobs[target];
    if(!admitted){const refs=await confirm(phase,owner,operation,inputs,config,scopes);const request={agent:owner,operation,workspace,inputs,config:{...config,plan_refs:refs}};
      if(evidence.phase_requests[phase])assert.deepEqual(evidence.phase_requests[phase],request);else{evidence.phase_requests[phase]=request;await save();}return refs;}
    assert.ok(admitted.jobId&&admitted.attemptId,'Uncertain admission must be reconciled; do not re-confirm or resubmit');
    const state=await command(rpc,await session(owner),`/control-status ${JSON.stringify({workspace,attemptId:admitted.attemptId})}`),record=state.record,request=record.request;
    assert.equal(record.owner,await session(owner));assert.equal(record.attemptId,admitted.attemptId);assert.equal(record.nativeJobId,admitted.jobId);
    assert.equal(record.agentName,owner);assert.equal(request.operation,operation);assert.equal(request.workspace,workspace);assert.deepEqual(request.inputs,inputs);
    const {plan_refs:refs,...actualConfig}=request.config;assert.deepEqual(actualConfig,config);
    const original=evidence.phase_requests[phase];assert.ok(original);assert.equal(hash(JSON.stringify(original)),admitted.signature);const {agent:declaredAgent,...nativeOriginal}=original;assert.equal(declaredAgent,owner);
    assert.deepEqual(request,nativeOriginal,'Use the originally frozen native request, never current scope heads');
    if(admitted.result){assert.equal(record.state,'completed');assert.deepEqual(record.result,admitted.result);}
    if(admitted.failure)assert.equal(record.state,'failed');
    const records=await events(owner),sources=[];
    for(const scope of scopes){const reference=refs[scope],snapshot=evidence.confirmations.find(row=>row.phase===phase&&row.scope===scope);
      assert.ok(snapshot);assert.equal(snapshot.owner,owner);assert.equal(snapshot.sessionId,await session(owner));assert.equal(snapshot.plan_id,reference.plan_id);assert.equal(snapshot.version_hash,reference.version_hash);
      const proposal=evidence.jobs[phase+'-plan-'+scope];assert.ok(proposal?.jobId&&proposal?.attemptId&&proposal.result);
      const old=await command(rpc,await session(owner),`/control-status ${JSON.stringify({workspace,attemptId:proposal.attemptId})}`);
      assert.equal(old.record.owner,await session(owner));assert.equal(old.record.agentName,owner);assert.equal(old.record.state,'completed');
      assert.equal(hash(JSON.stringify({agent:owner,operation:old.record.request.operation,workspace:old.record.request.workspace,inputs:old.record.request.inputs,config:old.record.request.config})),proposal.signature);
      assert.deepEqual(old.record.request,proposal.record.request);assert.equal(old.record.request.config.scope,scope);assert.equal(old.record.result.plan_id,reference.plan_id);assert.equal(old.record.result.version_hash,reference.version_hash);
      const run=records[snapshot.source.seq]?.event;assert.equal(run?.type,'command/run');assert.equal(run.data.name,'control-confirm');assert.equal(run.data.commandId,snapshot.source.command_id);assert.equal(run.data.source.kind,'user');
      assert.deepEqual(JSON.parse(run.data.args),{workspace,...reference});const done=records.find(row=>row.event.type==='command/done'&&row.event.data.commandId===run.data.commandId)?.event;
      assert.equal(done?.data.kind,'success');const confirmed=JSON.parse(done.data.text);assert.equal(confirmed.session_id,await session(owner));assert.equal(confirmed.scope,scope);assert.equal(confirmed.plan_id,reference.plan_id);assert.equal(confirmed.version_hash,reference.version_hash);assert.equal(confirmed.confirmation_source,'test_fixture');assert.deepEqual(confirmed.source,snapshot.source);
      sources.push({scope,...reference,confirmation_source:'test_fixture',source:snapshot.source});
    }
    if(record.state==='completed'&&record.result?.artifact_path){const root=resolve(record.result.artifact_path),path=relative(workspace,root);assert.ok(path&&!isAbsolute(path)&&path!=='..'&&!path.startsWith(`..${sep}`));
      const metadata=await json(resolve(root,'artifact.json')),reportBytes=await readFile(resolve(root,'report.json'));assert.equal(hash(reportBytes),metadata.files['report.json']);const report=JSON.parse(reportBytes);
      assert.deepEqual(report.confirmation_sources,sources,'Completed artifact preserves its own original receipt/confirmation sources');}
    evidence.historical_phase_checks??=[];evidence.historical_phase_checks.push({phase,target,attemptId:record.attemptId,jobId:admitted.jobId,request_sha256:admitted.signature,state:record.state,
      confirmation_sources:sources,verified_via:'Original native attempt request + historical proposal attempts + actual command/run/done and artifact receipt snapshots; current scope/status not read',at:new Date().toISOString()});await save();return refs;
  };
  const sources=resolve(workspace,'sources');await mkdir(sources,{recursive:true});for(const [name,digest] of Object.entries(packetFiles)){const target=resolve(sources,name);try{assert.equal(hash(await readFile(target)),digest);}catch(error){if(error.code!=='ENOENT')throw error;await copyFile(resolve(packet,name),target,constants.COPYFILE_EXCL);}assert.equal(hash(await readFile(target)),digest);}
  const material=await stage('material','model','register_material',{local_path:resolve(sources,'problem-input.json')},{dependency_root:sources,dependencies:['reference.csv','collocation.csv'],source_kind:'path',source_metadata:{kind:'S11_frozen_analytic_reference_and_train_collocation',registration_sha256:registrationSha,measured:false}});
  const problemRefs=await phaseRefs('prepare','problem','model','prepare_pinn_problem',{material:material.artifact_id},registration.problem_config,['signals']);
  const problem=await stage('problem','model','prepare_pinn_problem',{material:material.artifact_id},{...registration.problem_config,plan_refs:problemRefs});assert.equal(problem.kind,'conditional_ode_problem');assert.equal(problem.status,'verified');
  const activeRefs=await phaseRefs('active','active','model','train_pinn',{problem:problem.artifact_id},registration.active_config,['signals','training']);
  const active=await stage('active','model','train_pinn',{problem:problem.artifact_id},{...registration.active_config,plan_refs:activeRefs});assert.equal(active.kind,'pinn_model');assert.equal(active.status,'verified','Preregistered active candidate did not satisfy its immutable gates; preserve this attempt, no automatic retuning');
  const changed=structuredClone(registration.active_config);changed.loss_weights.physics*=.5;await deny('changed-physics-weight-old-confirmation','model','train_pinn',{problem:problem.artifact_id},{...changed,plan_refs:activeRefs});
  const copyBundle=async result=>{const source=resolve(result.artifact_path,'bundle'),target=resolve(portableRoot,result.artifact_id,'bundle');await mkdir(target,{recursive:true});const manifest=await json(resolve(source,'pinn_manifest.json'));
    for(const name of [...Object.keys(manifest.files),'pinn_manifest.json']){const expected=hash(await readFile(resolve(source,name))),destination=resolve(target,name);try{assert.equal(hash(await readFile(destination)),expected,'Existing portable member is immutable');}catch(error){if(error.code!=='ENOENT')throw error;await copyFile(resolve(source,name),destination,constants.COPYFILE_EXCL);}assert.equal(hash(await readFile(destination)),expected);}
    assert.deepEqual((await readdir(target)).sort(),[...Object.keys(manifest.files),'pinn_manifest.json'].sort());return resolve(target,'..');};
  const auditModel=async(result,kind)=>{const portable=await copyBundle(result),audit=runVerifier(['model','--artifact',result.artifact_path,'--problem',problem.artifact_path,'--workspace',workspace,'--registration',registrationPath,'--portable',portable,'--kind',kind]);assert.equal(audit.status,'PASS_INDEPENDENT_REPLAY');
    const path=resolve(dirname(evidencePath),`conditional-pinn-${hash(workspace).slice(0,12)}-${kind}-independent.json`);await writeSmokeEvidence(path,audit);return {path,sha256:hash(await readFile(path)),...audit};};
  evidence.active_independent=await auditModel(active,'active');await save();
  const direct=runVerifier(['boundaries','--repo',project,'--workspace',workspace,'--ids',problem.artifact_id,active.artifact_id]);assert.equal(direct.status,'PASS_EXPLICIT_CAPABILITY_REJECTION');evidence.direct_capability_rejection=direct;await save();
  for(const [label,id] of [['problem',problem.artifact_id],['pinn',active.artifact_id]]){
    if(evidence.boundary_checks.some(row=>row.label===label))continue;
    const config={prediction:{method:'direct'},controller:{horizon:3,max_iterations:20,solver_tolerance:1e-8},weights:{q:1,r:.01,s:.01},solver_check:false},inputs={model:id};
    const refs=await phaseRefs('reject-mpc-'+label,'reject-mpc-'+label,'controller','synthesize_mpc',inputs,config,['signals']);
    const rejected=await stage('reject-mpc-'+label,'controller','synthesize_mpc',inputs,{...config,plan_refs:refs},true);
    const manifest=await json(resolve(workspace,'manifest.json'));assert.ok(!Object.values(manifest.artifacts).some(row=>['controller','controller_design','optimized_policy'].includes(row.kind)));
    evidence.boundary_checks.push({label,artifact_id:id,...rejected,authorization:'Explicit native test_fixture signal confirmation; failed at the conditional-model capability boundary before controller publication, no solve requested'});await save();
  }
  const beforeOff=(await json(resolve(workspace,'manifest.json'))).active.pinn_model;assert.equal(beforeOff,active.artifact_id);
  const offRefs=await phaseRefs('off','off','model','train_pinn',{problem:problem.artifact_id},registration.ablation_config,['signals','training']);
  const off=await stage('off','model','train_pinn',{problem:problem.artifact_id},{...registration.ablation_config,plan_refs:offRefs});assert.equal(off.kind,'pinn_model');assert.equal(off.status,'verified');
  assert.equal((await json(resolve(workspace,'manifest.json'))).active.pinn_model,active.artifact_id,'Physics-off diagnostic must not replace accepted active PINN');
  evidence.off_independent=await auditModel(off,'off');evidence.ablation=runVerifier(['ablation','--active',active.artifact_path,'--off',off.artifact_path,'--problem',problem.artifact_path,'--registration',registrationPath]);assert.equal(evidence.ablation.status,'PASS_REAL_GRADIENT_ABLATION');await save();
  const handoff=await stage('handoff','model','prepare_handoff',{artifact_id:active.artifact_id});assert.equal(handoff.handoff.operation,null);assert.equal(handoff.handoff.capabilities.automatic_mpc_pgac_handoff,false);assert.equal(handoff.handoff.capabilities.arbitrary_state_transition,false);evidence.handoff=handoff;
  for(const key of Object.keys(evidence.sessions)){const records=await events(key);assert.ok(!records.some(row=>row.event.type.startsWith('turn/')),'No numerical point or optimizer step may invoke a model turn');}
  for(const [name,digest] of Object.entries(packetFiles)){assert.equal(hash(await readFile(resolve(packet,name))),digest);assert.equal(hash(await readFile(resolve(sources,name))),digest);}
  assert.equal(hash(await readFile(registrationPath)),registrationSha);assert.equal(hash(await readFile(verifier)),verifierSha);assert.equal(hash(await readFile(markerPath)),markerRecord.sha256);
  evidence.checks=[{status:'PASS',problem,active,off,source_unchanged:true,physics_off_inactive:true,active_preserved:true,controller_solution:'not_produced',PGAC:'not_run',confirmation_source:'test_fixture'}];
  evidence.status='PASS';evidence.finished_at=new Date().toISOString();
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,registration_sha256:registrationSha,modelSession:evidence.sessions.model,controllerSession:evidence.sessions.controller,
  active:evidence.jobs.active?.result?.artifact_id,off:evidence.jobs.off?.result?.artifact_id,active_test:evidence.active_independent?{per_condition:evidence.active_independent.per_condition_test_nrmse,physics_rms:evidence.active_independent.normalized_physics_rms,initial_error:evidence.active_independent.normalized_initial_max_abs_error}:null,
  actual_job_handles:evidence.admissions.map(row=>({key:row.key,jobId:row.jobId,attemptId:row.attemptId})),confirmation_source:'test_fixture'}));}
