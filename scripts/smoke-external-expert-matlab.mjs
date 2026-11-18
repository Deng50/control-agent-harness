/** S09 L02 real native preparation. Never starts/reloads Host or MATLAB. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {constants} from 'node:fs';
import {copyFile,mkdir,readFile,readdir} from 'node:fs/promises';
import {isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},fixtures:{type:'string'},'mcp-audit':{type:'string'},
  'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-external-expert-matlab.mjs --workspace-root <isolated parent> --evidence <json> --mcp-audit <trusted observer JSONL> [--fixtures <S09 fixture directory>] [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values['workspace-root']&&values.evidence&&values['mcp-audit'],'Explicit workspace-root/evidence/mcp-audit required');
const destination=resolve(values['workspace-root']),evidencePath=resolve(values.evidence),auditPath=resolve(values['mcp-audit']);
const rel=relative(resolve(project,'test-output'),destination);assert.ok(rel&&!isAbsolute(rel)&&rel!=='..'&&!rel.startsWith(`..${sep}`));
const run=resolve(project,'test-output/v2-development/20261117T002006'),fixtures=resolve(values.fixtures??resolve(run,'S09/matlab-expert-fixtures'));
const json=path=>readFile(path,'utf8').then(JSON.parse),hash=value=>createHash('sha256').update(value).digest('hex');
const manifestBytes=await readFile(resolve(fixtures,'fixture-manifest.json')),fixture=JSON.parse(manifestBytes);assert.equal(fixture.status,'PRE_REGISTERED_NOT_EXECUTED');
assert.deepEqual(fixture.cases.map(item=>item.key).sort(),['f01-simulink','f02-matlab']);assert.deepEqual(fixture.tolerances,{atol:1e-6,rtol:1e-5});
const preRegistered={fixture_manifest_sha256:hash(manifestBytes),options_probe_sha256:fixture.options_probe_sha256,cases:fixture.cases.map(item=>({key:item.key,
  collection:item.collection,expert_adapter:item.expert_adapter,expert_source_sha256:item.expert_source_sha256,expected_samples:item.expected_samples,reference_sha256:item.independent_reference_sha256})),
  tolerances:fixture.tolerances,mcp:{business_batch:1,fixed_identity_idle_probes:2,total_actual_tool_calls:3},
  scope:'External expert and declared plant collection only; no internal controller synthesis, training, Engine, cross-backend bridge or structural model edit.'};
let evidence=values.resume?await json(evidencePath):{schema_version:1,stage:'S09',test:'L02',status:'RUNNING',destination,auditPath,preRegistered,
  started_at:new Date().toISOString(),confirmation_source:'test_fixture',sessions:{},jobs:{},admissions:[],confirmations:[],rejections:[],checks:[]};
assert.equal(evidence.destination,destination);assert.equal(evidence.auditPath,auditPath);assert.deepEqual(evidence.preRegistered,preRegistered,'Resume must preserve all predeclared source/condition/gate inputs');
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const auditRows=async sessionId=>{
  let text='';try{text=await readFile(auditPath,'utf8');}catch(error){if(error.code!=='ENOENT')throw error;}
  return text.split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line)).filter(row=>row.source==='trusted_test_observer'&&row.sessionId===sessionId);
};
function auditCalls(rows){
  const starts=rows.filter(row=>row.event==='started'),returns=rows.filter(row=>row.event==='returned');assert.equal(starts.length,3,'Expected 1 business batch and 2 fixed guard probes');assert.equal(returns.length,3);assert.equal(rows.length,6,'No hidden retries/pointwise calls or thrown native tools');
  const ids=new Set();let batches=0,probes=0;
  for(const row of starts){const id=row.callId??row.rootCallId;assert.ok(id&&!ids.has(id));ids.add(id);assert.equal(row.name,'mcp__matlab__evaluate_matlab_code');
    const returned=returns.filter(other=>(other.callId??other.rootCallId)===id);assert.equal(returned.length,1);assert.equal(returned[0].isError,false);
    const classified=[row.callId,row.rootCallId].find(value=>value?.startsWith('matlab-batch-')||value?.startsWith('matlab-probe-'));
    if(classified?.startsWith('matlab-batch-'))batches++;else if(classified?.startsWith('matlab-probe-'))probes++;else assert.fail('Unexpected native tool identity '+id);
  }
  assert.equal(batches,1);assert.equal(probes,2);return{business_batch_calls:batches,fixed_guard_probe_calls:probes,total_actual_mcp_calls:starts.length,
    calls:starts.map(row=>({name:row.name,callId:row.callId,rootCallId:row.rootCallId,sessionId:row.sessionId})),source:'trusted optional around-execute observer; actual next() returned without isError, not inferred from job count'};
}
const independentAudit=String.raw`import hashlib,json,sys,numpy as np
from pathlib import Path
root=Path(sys.argv[1]);case=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));expected=json.loads(Path(case['independent_reference']).read_text(encoding='utf-8'))
assert hashlib.sha256(Path(case['independent_reference']).read_bytes()).hexdigest()==case['independent_reference_sha256']
read=lambda name:json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
meta=read('artifact');report=read('report');dataset=read('dataset');scaler=read('scaler');split=read('split');constraints=read('action_constraints');context=read('physical_context');plan=read('collection-plan');diagnostics=read('diagnostics')
for name,digest in meta['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
with np.load(root/'expert.npz',allow_pickle=False) as archive:arrays={name:archive[name].copy() for name in archive.files}
assert len(arrays['y'])==case['expected_samples']==len(diagnostics);errors={}
for key,value in expected['arrays'].items():
 a=arrays[key];b=np.asarray(value);assert a.shape==b.shape,key+' shape'
 np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5,err_msg=key+' independent active-set QP/scalar plant reference');errors[key]=float(np.max(np.abs(a-b))) if a.size else 0.
np.testing.assert_array_equal(arrays['y'],arrays['u'])
selectors=dataset['source_mapping']['signals'];feature_order=dataset['feature_order'];rawX=np.column_stack([arrays[selectors[item['signal']]['array']][:,selectors[item['signal']]['column']] for item in feature_order]);np.testing.assert_array_equal(rawX,arrays['X'])
assert not any('memory' in item['signal'] for item in feature_order)
assert split['mode']=='predefined_trajectory' and split['split_before_windowing'] is True
groups=split['groups']
for part in ['train','validation','test']:
 ids=[k for k,s in enumerate(case['collection']['scenarios']) if s['split']==part];assert groups[part]==ids
assert scaler['fit_split']=='train';train=np.isin(arrays['trajectory_id'],groups['train']);assert scaler['fit_samples']==int(train.sum())
for name in ['X','y']:
 location=arrays[name][train].mean(axis=0);scale=arrays[name][train].std(axis=0);scale[scale==0]=1
 np.testing.assert_allclose(scaler[name]['location'],location,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(scaler[name]['scale'],scale,atol=1e-12,rtol=1e-12)
 np.testing.assert_allclose(arrays[name+'_scaled'],(arrays[name]-location)/scale,atol=1e-12,rtol=1e-12)
assert report['status']=='completed' and report['training']=='not_run' and report['controller_design']=='not_run' and report['internal_MPC_required'] is False
assert report['diagnostic_memory_is_feature'] is False and report['source']['kind']=='declared_simulation' and report['source']['measured'] is False
physical=case['expert_adapter']['interface'];nu=len(physical['controls']);assert constraints['rate_semantics']=='per_control_step'
np.testing.assert_array_equal(constraints['rate_bounds'],case['expert_adapter']['input_rate_bounds'])
bounds=np.array([[v['lower'],v['upper']] for v in physical['controls']]);np.testing.assert_array_equal(constraints['bounds'],bounds)
assert np.all(arrays['u']>=bounds[:,0]-1e-12) and np.all(arrays['u']<=bounds[:,1]+1e-12)
rate=np.asarray(constraints['rate_bounds']);delta=arrays['u']-arrays['previous_u'];assert np.all(delta>=rate[:,0]-1e-12) and np.all(delta<=rate[:,1]+1e-12)
for k,s in enumerate(case['collection']['scenarios']):
 where=np.flatnonzero(arrays['trajectory_id']==k);assert len(where)==s['steps'];first=where[0]
 np.testing.assert_array_equal(arrays['previous_u'][first],s['initial_previous_action']);assert diagnostics[first]['memory_before']=={'count':0}
 for j,index in enumerate(where):
  row=diagnostics[index];assert row['status']=='ok' and row['memory_before']['count']==j and row['memory_after']['count']==j+1
  assert row['memory_after']['exitflag']>0 and row['memory_after']['violation']<=1e-6
assert context['plant_substeps']*context['plant_dt']==context['control_period']==case['expert_adapter']['control_period']
print(json.dumps({'status':'PASS','samples':len(arrays['y']),'feature_dim':arrays['X'].shape[1],'action_dim':nu,'train_samples':int(train.sum()),'split_counts':{key:len(groups[key]) for key in groups},
 'max_error_per_array':errors,'train_only_scaler':True,'memory_reset_all_trajectories':True,'actual_solver_statuses_ok':True,'actual_plant_bounds_rate_ok':True,
 'training':'not_run','independent_reference':'S00 active-set QP + scalar evaluation dynamics; no production MPC/MATLAB rerun by this audit'}))`;

try{
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  for(const item of fixture.cases){
    const workspace=resolve(destination,item.key);await prepareSmokeWorkspace(workspace);
    if(!evidence.sessions[item.key]){try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'Use --resume for admitted work');}catch(error){if(error.code!=='ENOENT')throw error;}
      await writeSmokeEvidence(resolve(workspace,'.case-plans/test-fixture.json'),{kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true});}
    if(!evidence.sessions[item.key]){const session=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});assert.equal(session.agentPreset,'control-lightweight');evidence.sessions[item.key]=session.sessionId;await save();}
    const sessionId=evidence.sessions[item.key],events=async()=>(await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:10000}})).records;
    const snapshot=async()=>{let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(name=>/^[0-9a-f-]{36}$/.test(name)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}
      const history=await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace})}`),records=await events();return{attempts,history:history.attempts.map(row=>row.attemptId).sort(),jobs:records.filter(row=>row.event.type==='job/registered').map(row=>row.event.data.jobId),mcp:(await auditRows(sessionId)).length};};
    const deny=async(label,request)=>{if(evidence.rejections.some(row=>row.label===label))return;const before=await snapshot();
      const response=await rpc('commands/execute',{agentId:sessionId,line:`/control ${JSON.stringify({agent:'lightweight',...request})}`,submittedAttachments:[]});
      assert.equal(response.result.kind,'error');assert.match(response.result.text,/confirm|plan|version|review/i);const after=await snapshot();assert.deepEqual(after,before);
      evidence.rejections.push({label,status:'PASS_ZERO_ADMISSION',message:response.result.text,newJobs:0,newHistory:0,newMcp:0,before,after});await save();};
    const stage=async(key,operation,inputs={},config={})=>{
      const request={agent:'lightweight',operation,workspace,inputs,config},signature=hash(JSON.stringify(request)),id=`${item.key}:${key}`;let row=evidence.jobs[id];
      if(row){assert.equal(row.signature,signature);assert.ok(row.jobId,'Uncertain admission must be reconciled from existing history, never resubmitted');if(row.result)return row.result;}
      else{row=evidence.jobs[id]={operation,signature,status:'ADMISSION_PENDING',sessionId};await save();const started=await command(rpc,sessionId,`/control ${JSON.stringify(request)}`);Object.assign(row,{jobId:started.jobId,attemptId:started.attemptId,status:'RUNNING'});evidence.admissions.push({key:id,operation,sessionId,jobId:row.jobId,attemptId:row.attemptId});await save();}
      for(;;){const state=await command(rpc,sessionId,`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);
        if(['completed','failed','killed'].includes(state.record.state)){row.state=state.record;row.status=state.record.state;await save();assert.equal(row.status,'completed',JSON.stringify(state));row.result=state.record.result;assert.ok(row.result);await save();return row.result;}
        assert.notEqual(state.record.state,'interrupted','Inspect/recover saved existing job; no replacement admission');await new Promise(done=>setTimeout(done,400));}
    };
    const register=async(key,file,adapter,operation)=>{
      await mkdir(resolve(workspace,'sources'),{recursive:true});const local=resolve(workspace,'sources',file.split(/[\\/]/).at(-1));
      try{await readFile(local);}catch(error){if(error.code!=='ENOENT')throw error;await copyFile(file,local,constants.COPYFILE_EXCL);}assert.equal(hash(await readFile(local)),hash(await readFile(file)));
      const material=await stage(key+'-material','register_material',{local_path:local},{source_kind:'path',source_metadata:{confirmation_source:'test_fixture',description:'Preregistered external MATLAB source; no internal controller synthesis'}});
      const info=await json(resolve(material.artifact_path,'material.json'));return stage(key,operation,{material:material.artifact_id},{adapter:{...adapter,source_file:info.primary_file}});
    };
    assert.equal(hash(await readFile(item.independent_reference)),item.independent_reference_sha256);
    const expertSourceHash=hash(await readFile(item.expert_file));assert.equal(expertSourceHash,item.expert_source_sha256);const expert=await register('expert',item.expert_file,item.expert_adapter,'register_expert');
    let plant;if(item.plant.kind==='import'){assert.equal(hash(await readFile(item.plant.original_slx)),item.plant.original_slx_sha256);plant=await stage('plant-import','import_artifact',{artifact_path:item.plant.artifact_path});}
    else{assert.equal(hash(await readFile(item.plant.file)),item.plant.source_sha256);plant=await register('plant',item.plant.file,item.plant.adapter,'register_plant');}
    evidence.static_mcp_checks??={};
    if(!evidence.static_mcp_checks[item.key]){assert.ok(!evidence.jobs[`${item.key}:collect`]?.jobId);assert.equal((await auditRows(sessionId)).length,0,'Static registrations must not call MATLAB');evidence.static_mcp_checks[item.key]={status:'PASS',actual_mcp_events:0};await save();}
    const inputs={expert:expert.artifact_id,plant:plant.artifact_id},request={operation:'collect_external_expert',workspace,inputs,config:item.collection};
    const requirements=await confirmationRequirements(request);assert.deepEqual(requirements.map(row=>row.scope),['signals','sampling']);const refs={};
    for(const row of requirements){const plan=await stage('plan-'+row.scope,'propose_case_plan',{}, {scope:row.scope,payload:row.expectedPayload,bindings:row.bindings});refs[row.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};}
    await deny(item.key+':before-confirmation',{...request,config:{...item.collection,plan_refs:refs}});
    for(const row of requirements){const confirmed=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace,...refs[row.scope]})}`);assert.equal(confirmed.confirmation_source,'test_fixture');evidence.confirmations.push({key:item.key,scope:row.scope,...refs[row.scope],confirmation_source:'test_fixture'});await save();}
    const result=await stage('collect','collect_external_expert',inputs,{...item.collection,plan_refs:refs});assert.equal(result.kind,'external_expert_dataset');assert.equal(result.status,'verified');
    const changed=structuredClone(item.collection);changed.scenarios[0].steps+=1;await deny(item.key+':changed-condition-old-confirmation',{...request,config:{...changed,plan_refs:refs}});
    const caseFile=resolve(fixtures,item.key,'case.json');assert.deepEqual(await json(caseFile),item);const independent=JSON.parse(execFileSync(python,['-I','-B','-c',independentAudit,result.artifact_path,caseFile],{cwd:workspace,encoding:'utf8',windowsHide:true,maxBuffer:8*1024*1024,env:{...process.env,PYTHONPATH:'',PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'}}));
    assert.equal(independent.status,'PASS');assert.equal(hash(await readFile(item.expert_file)),expertSourceHash);
    if(item.plant.kind==='import')assert.equal(hash(await readFile(item.plant.original_slx)),item.plant.original_slx_sha256);
    const records=await events();assert.ok(!records.some(row=>row.event.type.startsWith('turn/')),'No LLM turn may run numerical samples');const actualCalls=auditCalls(await auditRows(sessionId));
    const final=await json(resolve(workspace,'manifest.json'));assert.ok(!Object.values(final.artifacts).some(row=>['controller','controller_design','imitation_policy','optimized_policy','supervised_model'].includes(row.kind)),'External collection must stop before internal design/training');
    const check={key:item.key,status:'PASS',workspace,sessionId,artifact_id:result.artifact_id,artifact_path:result.artifact_path,job:evidence.jobs[`${item.key}:collect`],
      actual_calls:actualCalls,independent,source_expert_sha256:expertSourceHash,source_slx_sha256:item.plant.original_slx_sha256??null,confirmation_source:'test_fixture'};
    const old=evidence.checks.findIndex(row=>row.key===item.key);if(old<0)evidence.checks.push(check);else evidence.checks[old]=check;await save();
  }
  assert.equal(evidence.checks.length,2);assert.equal(evidence.rejections.length,4);evidence.status='PASS';evidence.finished_at=new Date().toISOString();
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,checks:evidence.checks.map(row=>({key:row.key,artifact_id:row.artifact_id,samples:row.independent.samples,totalActualMcp:row.actual_calls.total_actual_mcp_calls})),admissions:evidence.admissions.map(row=>({operation:row.operation,jobId:row.jobId,attemptId:row.attemptId})),confirmation_source:'test_fixture'}));}
