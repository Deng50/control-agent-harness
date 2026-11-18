/** E02: real configured-provider business conversations. No mock model or alternate agent loop. */
import assert from 'node:assert/strict';
import {randomUUID,createHash} from 'node:crypto';
import {mkdir,readFile,writeFile,rename,copyFile,stat} from 'node:fs/promises';
import {dirname,resolve,join,isAbsolute,relative,sep} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {parseArgs,isDeepStrictEqual} from 'node:util';
import {execFileSync} from 'node:child_process';
import {connectHarness,prepareSmokeWorkspace,project} from './harness-rpc.mjs';
import {isRecordedReadOnlyFailure,verifyRecoveredInspector,verifyCorrectedMaterial} from './e02-continuation.mjs';

const self=fileURLToPath(import.meta.url),sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const read=async path=>JSON.parse((await readFile(path,'utf8')).replace(/^\uFEFF/,''));
const ref=async path=>({path:resolve(path),sha256:sha(await readFile(path))});
const equal=(a,b)=>isDeepStrictEqual(JSON.parse(JSON.stringify(a)),JSON.parse(JSON.stringify(b)));
const wait=ms=>new Promise(done=>setTimeout(done,ms));
const inside=(base,path)=>{const r=relative(base,path);return !!r&&!isAbsolute(r)&&r!=='..'&&!r.startsWith('..'+sep);};
const roleSkills={model:'train-object-model',controller:'design-and-tune-mpc',lightweight:'distill-controller'};

async function immutable(path,bytes){await mkdir(dirname(path),{recursive:true});try{await writeFile(path,bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;assert.deepEqual(await readFile(path),Buffer.from(bytes));}}

async function allEvents(rpc,sessionId){
  const pages=[],events=new Map();let beforeSeq,throughSeq;
  for(let i=0;i<200;i++){
    const page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:100,...(beforeSeq===undefined?{}:{beforeSeq}),...(throughSeq===undefined?{}:{throughSeq})}});
    if(throughSeq===undefined)throughSeq=page.cursor;assert.equal(page.cursor,throughSeq);pages.push(page);
    for(const row of page.records)events.set((row.event??row).seq,row);
    if(!page.hasMore)return{throughSeq,pages,records:[...events.values()].sort((a,b)=>(a.event??a).seq-(b.event??b).seq)};
    const first=Math.min(...page.records.map(row=>(row.event??row).seq));assert.ok(Number.isFinite(first)&&first>0);beforeSeq=first;
  }
  throw Error('Native journal pagination budget exceeded; preserve original Session, no retry');
}

const checkArtifactCode=String.raw`import json,sys,hashlib,math,shutil,subprocess,os
from pathlib import Path
import numpy as np,torch
from control_harness.artifacts import Workspace
role,workspace,artifact_id,packet,output=sys.argv[1:]
ws=Workspace(workspace);root,meta=ws.resolve(artifact_id);report=json.loads((root/'report.json').read_text());case=json.loads(Path(packet).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=False)
assert meta['validation']['status']=='verified'
for source in report.get('confirmation_sources',[]):assert source['confirmation_source']=='test_fixture'
result={'artifact_id':artifact_id,'kind':meta['kind'],'root':str(root),'metadata_sha256':hashlib.sha256((root/'artifact.json').read_bytes()).hexdigest(),'report_sha256':hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),'all_parent_and_file_hashes_verified':True}
if role=='model':
 assert meta['kind']=='supervised_model' and report['acceptance']=='passed';assert report['network_config']['input_dim']==3 and report['network_config']['output_dim']==2
 for key,value in case['training_config'].items():assert report['resolved_config'][key]==value,(key,report['resolved_config'].get(key),value)
 assert report['test_used_for_training_or_selection'] is False and report['test_evaluations']==1
 assert max(report['test']['nrmse'])<=.15
 data_root,data_meta=ws.resolve(report['data_provenance']['dataset_id']);a=np.load(data_root/'supervised.npz',allow_pickle=False);rows=np.flatnonzero(a['split_id']==2)[:12];X=a['X'][rows]
 saved=np.load(root/'test_predictions.npz',allow_pickle=False);expected=saved['prediction'][:12]
 payload={'features':X.tolist()};script='predict.py';result.update(test=report['test'],network=report['network_config'],trained_epochs=report['trained_epochs'],optimizer_updates=report['optimizer_updates'],physical_model_required=False)
 checkpoint=torch.load(root/'latest.pt',map_location='cpu',weights_only=True);assert checkpoint['optimizer_state']['param_groups'][0]['lr']==case['training_config']['learning_rate'];assert checkpoint['updates']==report['optimizer_updates']
elif role=='lightweight':
 assert meta['kind']=='external_imitation_policy' and report['acceptance']=='passed';assert report['network']['network_config']['input_dim']==9 and report['network']['network_config']['output_dim']==2 and report['network']['parameter_count']==98
 for key,value in case['training_config']['network'].items():assert report['network']['resolved_config'][key]==value,key
 assert report['closed_loop']=='not_run' and report['pgac']=='not_run' and report['test_used_for_training_or_selection'] is False and report['test_evaluations']==1
 assert report['physical_context']['interface'] is None and max(report['offline']['test']['projected']['nrmse'])<=.2
 data_root,data_meta=ws.resolve(report['external_expert_dataset']);a=np.load(data_root/'expert.npz',allow_pickle=False);rows=np.flatnonzero(a['split_id']==2)[:12];X=a['X'][rows];previous=a['actual_previous_u'][rows]
 saved=np.load(root/'test_predictions.npz',allow_pickle=False);expected=saved['projected_action'][:12]
 payload={'features':X.tolist(),'previous_action':previous.tolist()};script='infer.py';result.update(test=report['offline']['test'],network=report['network'],closed_loop='not_run',pgac='not_run')
else:
 assert meta['kind']=='controller_design';d=json.loads((root/'design.json').read_text());assert d['model_instantiated'] is True and d['closed_loop']['status']=='not_run' and d['closed_loop']['ga_trials']==0 and d['solver_check']['status']=='passed'
 payload=case['solver_check'];script='controller.py';expected=np.asarray(d['solver_check']['action']);result.update(model_instantiated=True,closed_loop=d['closed_loop'],solver_check=d['solver_check'])
shutil.copytree(root/'bundle',out/'bundle');input_path=out/'input.json';input_path.write_text(json.dumps(payload));output_path=out/'prediction.json'
block="import importlib.abc,sys,runpy\nclass Block(importlib.abc.MetaPathFinder):\n def find_spec(self,fullname,path=None,target=None):\n  if fullname=='control_harness' or fullname.startswith('control_harness.'): raise ModuleNotFoundError('Standalone must not import project')\nsys.meta_path.insert(0,Block())\nscript=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')"
completed=subprocess.run([sys.executable,'-I','-c',block,str(out/'bundle'/script),'--input',str(input_path),'--output',str(output_path)],cwd=out,capture_output=True,text=True,env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME'}})
assert completed.returncode==0,(completed.stdout,completed.stderr)
value=json.loads(output_path.read_text());actual=np.asarray(value['predictions'] if role=='model' else value['action'] if role=='lightweight' else value['u']);assert np.allclose(actual,expected,atol=1e-6,rtol=1e-5),(actual,expected)
result.update(independent_bundle=str(out/'bundle'),independent_subprocess_exit=0,independent_inference_max_abs_error=float(np.max(np.abs(actual-expected))),bundle_file_hashes={str(p.relative_to(out/'bundle')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'bundle').rglob('*') if p.is_file() and '__pycache__' not in p.parts})
print(json.dumps(result,allow_nan=False))`;

async function main(){
 const {values}=parseArgs({options:{'prepare-registration':{type:'boolean'},'validate-only':{type:'boolean'},'run-approved':{type:'boolean'},registration:{type:'string'},'registration-sha256':{type:'string'},'workspace-root':{type:'string'},evidence:{type:'string'},'input-registration':{type:'string'},'host-record':{type:'string'},help:{type:'boolean',short:'h'},resume:{type:'boolean'}}});
 if(values.help){console.log('node scripts/smoke-v2-user-paths.mjs --prepare-registration --workspace-root <fresh> --evidence <E02-json> --input-registration <frozen-inputs> --host-record <actual-normal-record> --registration <new-json> | --registration <json> --registration-sha256 <sha> --validate-only | --run-approved [--resume observer-only]');return;}
 const registrationPath=resolve(values.registration??'test-output/v2-development/20261117T002006/S14/E02-registration-v1.json');
 if(values['prepare-registration']){
   assert.ok(values['workspace-root']&&values.evidence&&values['input-registration']&&values['host-record']);const inputs=await read(resolve(values['input-registration']));assert.equal(inputs.status,'FROZEN_INPUTS_NO_TRAINING_NO_LLM');
   const host=await read(resolve(values['host-record']));assert.equal(host.status,'PASS');assert.equal(host.started.fault_overlay_loaded,false);assert.equal(host.started.production_timeout_ms,300000);
   const packet=resolve(project,'examples/v2'),files=[];for(const item of inputs.files){const p=resolve(packet,item.path);assert.equal(sha(await readFile(p)),item.sha256);files.push(await ref(p));}
   const registration={schema_version:1,status:'REGISTERED_BEFORE_E02_LLM_OR_NUMERICAL_ADMISSION',at:new Date().toISOString(),script:await ref(self),workspace_root:resolve(values['workspace-root']),evidence:resolve(values.evidence),inputs:await ref(resolve(values['input-registration'])),public_root:packet,files,host:await ref(resolve(values['host-record'])),cases:{model:await read(join(packet,'model-nn.json')),controller:await read(join(packet,'mpc-design.json')),lightweight:await read(join(packet,'imitation.json'))},lineage:inputs.lineage,source_sha256:await Promise.all(['harness-plugin/src/index.mjs','harness-plugin/src/session-workspace.mjs','harness-plugin/src/case-requirements.mjs','harness-plugin/src/case-plans.mjs','harness-plugin/src/roles.mjs','scripts/harness-rpc.mjs','src/control_harness/supervised.py','src/control_harness/external_imitation.py','src/control_harness/design_only.py'].map(p=>ref(resolve(project,p)))),protocol:{configured_provider:'catalog.default only; no credential fallback',confirmation_source:'test_fixture native user command; model never confirms',scope:'Three actual business conversations, no mock or second LLM grader',uncertain_admission:'stop and observe original handle; never retry automatically',test_selection:false,roles_sequential:true,LLM_chooses_stage_and_proposes:true}};
   assert.ok(inside(resolve(project,'test-output'),registration.workspace_root));await immutable(registrationPath,JSON.stringify(registration,null,2)+'\n');console.log(JSON.stringify({status:registration.status,registration:await ref(registrationPath)}));return;
 }
 assert.equal(sha(await readFile(registrationPath)),values['registration-sha256']);const reg=await read(registrationPath);assert.equal(sha(await readFile(self)),reg.script.sha256);for(const item of [...reg.files,...reg.source_sha256])assert.equal(sha(await readFile(item.path)),item.sha256);assert.equal(sha(await readFile(reg.host.path)),reg.host.sha256);const host=await read(reg.host.path);
 let continuationProof,priorEvidence,materialContinuationProof;
 if(reg.observer_revision){const revision=reg.observer_revision;for(const item of [revision.original_registration,revision.original_evidence,revision.original_driver_snapshot,revision.helper])assert.equal(sha(await readFile(item.path)),item.sha256);const original=await read(revision.original_registration.path);for(const key of ['workspace_root','inputs','public_root','files','host','cases','lineage','source_sha256','protocol'])assert.deepEqual(reg[key],original[key],`Observer revision changed original engineering registration: ${key}`);assert.notEqual(reg.evidence,original.evidence);priorEvidence=await read(revision.original_evidence.path);assert.equal(priorEvidence.registration_sha256,revision.original_registration.sha256);continuationProof=await verifyRecoveredInspector(priorEvidence,revision);if(revision.latest_evidence){for(const item of [revision.latest_evidence,revision.previous_registration,...revision.previous_observer_sources])assert.equal(sha(await readFile(item.path)),item.sha256);const previous=await read(revision.previous_registration.path);for(const key of ['workspace_root','inputs','public_root','files','host','cases','lineage','source_sha256','protocol'])assert.deepEqual(reg[key],previous[key]);priorEvidence=await read(revision.latest_evidence.path);assert.equal(priorEvidence.registration_sha256,revision.previous_registration.sha256);materialContinuationProof=await verifyCorrectedMaterial(priorEvidence,revision);}}
 if(values['validate-only']){console.log(JSON.stringify({status:'READY_STATIC_NO_RPC',registration:await ref(registrationPath),roles:Object.keys(reg.cases),LLM:0,numerical_jobs:0}));return;}
 assert.ok(values['run-approved']);let e;try{e=await read(reg.evidence);assert.ok(values.resume,'Existing evidence; no blind repeat');assert.equal(e.registration_sha256,values['registration-sha256']);if(e.status==='PASS'){console.log(JSON.stringify({status:'PASS_HISTORICAL_REUSE',evidence:await ref(reg.evidence)}));return;}}catch(error){if(error.code!=='ENOENT')throw error;assert.ok(!values.resume);if(priorEvidence){e=structuredClone(priorEvidence);e.status='RUNNING';e.observer_failure_history??=[];if(e.original_failure)e.observer_failure_history.push(e.original_failure);e.observer_failure_history.push(e.error);e.original_failure=e.error;delete e.error;e.registration=registrationPath;e.registration_sha256=values['registration-sha256'];e.prior_observer_revision=e.observer_revision;e.observer_revision={...reg.observer_revision,verified_original_pair:continuationProof,verified_corrected_material:materialContinuationProof,continued_at:new Date().toISOString(),no_completed_role_reconfirmation_or_training:true};e.roles.controller.turns.draft.state='COMPLETED';e.roles.controller.turns.draft.observer_completion=continuationProof;}else e={schema_version:1,stage:'S14',group:'E02',status:'RUNNING',started_at:new Date().toISOString(),registration:registrationPath,registration_sha256:values['registration-sha256'],roles:{},confirmations:[],mutations:[],checks:[],authentication:{auth_material_used:true,private_startup_read_only_inside_connectHarness:true,token_or_cookie_output:false,browser_injection:false}};}
 let serial=0;const save=async()=>{await mkdir(dirname(reg.evidence),{recursive:true});const snapshot=reg.evidence+`.snapshot-${process.pid}-${++serial}.json`;await writeFile(snapshot,JSON.stringify(e,null,2)+'\n',{flag:'wx'});const tmp=reg.evidence+`.${process.pid}.tmp`;await writeFile(tmp,JSON.stringify(e,null,2)+'\n');await rename(tmp,reg.evidence);};await save();
 try{
  const rpc=await connectHarness({logPath:host.started.private_stdout_path,connectionClose:true});
  const command=async(sessionId,name,args)=>{const raw=`/${name} ${JSON.stringify(args)}`,response=await rpc('commands/execute',{agentId:sessionId,line:raw,submittedAttachments:[]});assert.equal(response.result.kind,'success',response.result.text);return{value:JSON.parse(response.result.text),response,raw};};
  const mutate=async(key,method,args)=>{assert.ok(!values.resume,'Observer-only resume cannot admit new work');assert.ok(!e.mutations.some(row=>row.key===key));const row={key,method,args,at:new Date().toISOString(),state:'INTENT_NO_RETRY'};e.mutations.push(row);await save();row.response=await rpc(method,args);row.state='RETURNED';await save();return row.response;};
  const catalogue=await rpc('session/modelCatalog',{});const selected=catalogue.default;assert.ok(selected?.provider&&selected?.model);assert.ok(catalogue.groups.find(g=>g.id===selected.provider)?.models.some(m=>m.id===selected.model),'Configured default is absent; do not silently switch providers');e.configured_model={...selected};await save();
  const history=async role=>{const row=e.roles[role],h=(await command(row.sessionId,'control-history',{workspace:row.workspace})).value;row.history=h;return h.attempts;};
  const capture=async(role,label)=>{const row=e.roles[role],events=await allEvents(rpc,row.sessionId);row.events=events;row.event_snapshots??=[];const path=resolve(dirname(reg.evidence),`E02-${role}-${label}-native-events.json`);await immutable(path,JSON.stringify(events,null,2)+'\n');row.event_snapshots.push(await ref(path));await history(role);await save();return events;};
  if(values.resume){e.observer_only_resumes??=[];const observation={at:new Date().toISOString(),roles:[]};for(const [role,row]of Object.entries(e.roles)){if(row.sessionId){observation.roles.push({role,sessionId:row.sessionId,state:await rpc('session/state',{request:{sessionId:row.sessionId}}),native_events:await allEvents(rpc,row.sessionId),history:await history(role)});}}e.observer_only_resumes.push(observation);await save();console.log(JSON.stringify({status:'OBSERVED_EXISTING_HANDLES_NO_NEW_ADMISSION',evidence:reg.evidence}));return;}
  const turn=async(role,key,text)=>{
   const row=e.roles[role];row.turns??={};let t=row.turns[key];
   if(!t){assert.ok(!values.resume,'Resume cannot invent a prompt');const state=await rpc('session/state',{request:{sessionId:row.sessionId}});assert.equal(state.running,false);assert.deepEqual(state.pendingInput,[]);t=row.turns[key]={requestId:randomUUID(),text,text_sha256:sha(text),afterSeq:state.cursor,state:'INTENT_NO_RETRY',at:new Date().toISOString()};await save();t.admission=await mutate(`${role}:prompt:${key}`,'session/prompt',{request:{sessionId:row.sessionId,requestId:t.requestId,mode:'queue',content:[{type:'text',text}]}});t.state='ADMITTED';await save();}
   if(t.state!=='COMPLETED'){
    const deadline=Date.now()+1200000;let state;
    do{state=await rpc('session/wait',{request:{sessionId:row.sessionId,timeoutMs:1000,maxMessages:5,afterSeq:t.afterSeq}});if(state.reason!=='timeout')break;}while(Date.now()<deadline);
    t.wait=state;await capture(role,key);assert.equal(state.reason,'idle','Native input/timeout/failure: preserve existing handle, no retry');assert.equal(state.state.running,false);assert.deepEqual(state.state.pendingInput,[]);
    const records=row.events.records.map(r=>r.event??r);assert.ok(records.some(ev=>ev.type==='user/message'&&JSON.stringify(ev.data).includes(t.requestId)),'Prompt source request ID is absent');assert.equal(records.findLast(ev=>ev.type==='turn/end')?.data.reason?.kind,'completed');
    const failed=row.history.attempts.filter(a=>['failed','killed','interrupted'].includes(a.state)&&!isRecordedReadOnlyFailure(a,reg.observer_revision));assert.equal(failed.length,0,'A real native job failed; preserve and report before any further work');t.state='COMPLETED';await save();
   }
   console.log(JSON.stringify({role,phase:key,status:t.state,sessionId:row.sessionId}));
  };
  const results=async(role,operation)=>{const attempts=await history(role);return attempts.filter(a=>a.request.operation===operation&&a.state==='completed'&&a.result);};
  const onlyResult=async(role,operation,predicate=()=>true)=>{const rows=(await results(role,operation)).filter(predicate);assert.equal(rows.length,1,`${role}/${operation} must have exactly one actual accepted stage`);return rows[0].result;};
  const summary=async(role,key,operation,inputs,config)=>{
   const row=e.roles[role],request={agent:role,operation,workspace:row.workspace,inputs,config};const review=(await command(row.sessionId,'control-review',request)).value;row.reviews??={};row.reviews[key]=review;await save();
   const prompt=`这是本次业务下一阶段的真实Host只读审阅，不是批准：${JSON.stringify({request,requirements:review.requirements})}。请保持材料版本、payload和bindings逐字语义一致，用本角色领域工具自行提出每个实际scope的版本化提案，并用原生job工具读取真实结果。signals dependencies必须{}；training dependencies仅引用本轮signals的实际plan_id/version_hash，不能默认带入其他路线摘要。此回合仅提案和展示摘要，不执行受保护的${operation}，不调用确认命令、终端代签或ask_user_question，完成后停止等我的下一用户turn。`;
   await turn(role,`${key}-propose`,(role==='controller'&&key==='numeric-design'?(reg.observer_revision?.controller_correction_prefix??''):'')+prompt);const refs={};
   for(const req of review.requirements){const proposals=await results(role,'propose_case_plan');const deps=req.scope==='signals'?{}:{signals:refs.signals};const match=proposals.filter(a=>a.result.scope===req.scope&&equal(a.result.payload,req.expectedPayload)&&equal(a.result.bindings,req.bindings)&&equal(a.result.dependencies,deps));assert.ok(match.length,'LLM did not propose the exact actual Host-reviewed scope; do not synthesize a plan');const plan=match.at(-1).result;refs[req.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};
    const confirmed=(await command(row.sessionId,'control-confirm',{workspace:row.workspace,...refs[req.scope]}));assert.equal(confirmed.value.confirmation_source,'test_fixture');assert.equal(confirmed.value.session_id,row.sessionId);e.confirmations.push({role,key,scope:req.scope,...confirmed.value,native_command_response:confirmed.response});await save();}
   return{...config,plan_refs:refs};
  };
  const executeText=(operation,inputs,config)=>`我已通过本会话原生用户命令确认刚才实际提出的这些版本。这是明确的test_fixture用户答复，不是你代签。请按原请求执行当前${operation}任务，输入${JSON.stringify(inputs)}，完整配置${JSON.stringify(config)}。你自行选择本角色正确领域工具，必须等待真实job终态并读取结果。完成后检查该实际产物及完整导出文件，简短报告实际ID、指标和未运行边界并结束，不粘贴完整逐epoch日志；不要追加后续业务、改参数、重投、不调用ask_user_question。`;
  for(const role of ['model','controller','lightweight']){
   if(e.roles[role]?.status==='PASS'){assert.ok(reg.observer_revision&&(role==='model'||(role==='controller'&&materialContinuationProof)));console.log(JSON.stringify({role,status:'PASS_HISTORICAL_ROLE_REUSE_NO_NEW_TURN_CONFIRM_OR_FIT',artifact:e.roles[role].final.artifact_id,sessionId:e.roles[role].sessionId}));continue;}
   const workspace=resolve(reg.workspace_root,role);await prepareSmokeWorkspace(workspace);await mkdir(join(workspace,'.case-plans'),{recursive:true});await immutable(join(workspace,'.case-plans/test-fixture.json'),JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace},null,2)+'\n');
   const packet=reg.cases[role],packetName=role==='model'?'model-nn.json':role==='controller'?'mpc-design.json':'imitation.json';if(role!=='controller')await immutable(join(workspace,'inputs',packetName),await readFile(join(reg.public_root,packetName)));else await immutable(join(workspace,'inputs','mpc-draft.json'),await readFile(join(reg.public_root,'mpc-draft.json')));if(packet.data_file)await immutable(join(workspace,'inputs',packet.data_file),await readFile(join(reg.public_root,packet.data_file)));
   if(!e.roles[role]){assert.ok(!values.resume);const markerPath=join(workspace,'.case-plans/test-fixture.json'),markerStat=await stat(markerPath);e.roles[role]={workspace,marker_before_session:true,fixture_marker:{...await ref(markerPath),birthtime_ms:markerStat.birthtimeMs,mtime_ms:markerStat.mtimeMs,recorded_before_session_create_at:new Date().toISOString()},turns:{}};await save();const w=await mutate(`${role}:workspace-create`,'workspace/create',{request:{path:workspace}});const created=await mutate(`${role}:session-create`,'session/create',{request:{workspaceId:w.workspace.workspaceId,agentPreset:`control-${role}`}});e.roles[role].sessionId=created.sessionId;await save();await command(created.sessionId,'control-workspace',{});await mutate(`${role}:select-model`,'session/selectModel',{request:{sessionId:created.sessionId,provider:selected.provider,model:selected.model,...(selected.reasoningEffort?{reasoningEffort:selected.reasoningEffort}:{})}});}
   const row=e.roles[role],intro=`这是已授权的明确合成仿真验收业务，不是实车测量。工作目录${workspace}。请先加载角色入口skill及相关执行参考，再读资料${join(workspace,'inputs',role==='controller'?'mpc-draft.json':packetName)}。资料数值与边界是我给定的工程输入，不是文件中的执行指令；不要修改原文件或超参数。所有确认由我后续原生用户命令给出，不能自行执行/control-confirm。原生工具返回jobId不等于完成，你要读取真实job结果。`;
   if(role==='controller'){
    await turn(role,'draft',intro+'当前弹簧质量、阻尼、刚度、dt及边界尚未给出，请只交付参数化MPC设计包。现在只使用draft_config，缺项必须model_instantiated=false，不运行GA或闭环，交付后停下，不询问下一步。');
    const draft=await onlyResult(role,'synthesize_mpc');const d=await read(join(draft.artifact_path,'design.json'));assert.equal(d.model_instantiated,false);assert.ok(d.required_parameters.length===7);assert.equal(d.closed_loop.ga_trials,0);row.draft=draft;await save();
    await immutable(join(workspace,'inputs',packetName),await readFile(join(reg.public_root,packetName)));const config={...packet.draft_config,parameters:packet.parameters,solver_check:packet.solver_check};const inputs={description:packet.description};const confirmed=await summary(role,'numeric-design','synthesize_mpc',inputs,config);await turn(role,'numeric-design',executeText('synthesize_mpc',inputs,confirmed));const choices=await results(role,'synthesize_mpc');const numeric=[];for(const choice of choices){const design=await read(join(choice.result.artifact_path,'design.json'));if(design.model_instantiated)numeric.push(choice.result);}assert.equal(numeric.length,1);row.final=numeric[0];
   }else{
    await turn(role,'material',intro+(role==='model'?'我只有CSV数据，最终只要数据NN，不要物理/名义模型、控制器或PINN。此回合只检查实际表头并固化本CSV材料，报告实际材料ID；先不准备数据、不训练、不做版本化提案。':'我已有外部专家CSV记录，最终只要模仿网络，不要重新设计MPC，不采数，不PGAC，不编造可执行Plant或闭环。此回合只检查实际表头并固化本CSV材料，报告实际材料ID；先不导入专家数据、不训练、不做版本化提案。'));
    const material=role==='lightweight'&&materialContinuationProof?materialContinuationProof.selected_material:await onlyResult(role,'register_material');row.material=material;if(role==='lightweight'&&materialContinuationProof)row.material_header_observation=materialContinuationProof;else assert.equal((await results(role,'inspect_table')).length,1,'Actual headers must be inspected');const operation=role==='model'?'prepare_supervised_dataset':'import_expert_data',inputs={material:material.artifact_id};
    const dataConfig=await summary(role,'data',operation,inputs,packet.data_config);await turn(role,'data',executeText(operation,inputs,dataConfig));const data=await onlyResult(role,operation);row.dataset=data;await save();
    const trainOp=role==='model'?'train_supervised':'train_external_imitation',trainInputs={dataset:data.artifact_id},training=await summary(role,'training',trainOp,trainInputs,packet.training_config);await turn(role,'training',executeText(trainOp,trainInputs,training));row.final=await onlyResult(role,trainOp);
   }
   assert.equal(row.final.status,'verified','Engineering gate rejected; no test-driven retry');await capture(role,'final');
   const eventRows=row.events.records.map(v=>v.event??v),toolCalls=eventRows.filter(v=>v.type==='tool/call');assert.ok(toolCalls.some(v=>v.data.name==='skill'&&JSON.parse(v.data.arguments).name===roleSkills[role]));assert.ok(toolCalls.some(v=>v.data.name===`control_${role}`));assert.ok(toolCalls.some(v=>v.data.name==='job_output'));assert.ok(!toolCalls.some(v=>v.data.name.startsWith('mcp__matlab__')),'These three explicit business paths require no MATLAB');assert.ok(!toolCalls.some(v=>v.data.name.startsWith('control_')&&!['control_help',`control_${role}`].includes(v.data.name)),'No cross-role or forced workflow');
   const forbidden=['run_workflow','demo','design_controller','tune_controller','collect_expert','collect_external_expert','optimize_policy','optimize_external_policy','evaluate_external_pgac','train_pinn'];assert.ok(!row.history.attempts.some(a=>forbidden.includes(a.request.operation)));
   const output=resolve(dirname(reg.evidence),'E02-independent',role);const checked=JSON.parse(execFileSync(resolve(project,'.venv/Scripts/python.exe'),['-B','-c',checkArtifactCode,role,workspace,row.final.artifact_id,join(workspace,'inputs',packetName),output],{cwd:project,encoding:'utf8',windowsHide:true,maxBuffer:16*1024*1024,env:{...process.env,PYTHONPATH:resolve(project,'src'),PYTHONUTF8:'1'}}));e.checks.push({role,...checked});row.status='PASS';await save();console.log(JSON.stringify({role,status:'PASS',artifact:row.final.artifact_id,sessionId:row.sessionId}));
  }
  assert.equal(e.checks.length,3);e.status='PASS';e.finished_at=new Date().toISOString();await save();
 }catch(error){e.status='FAIL_RETAIN_ACTUAL_HANDLES_NO_AUTOMATIC_RETRY';e.error={name:error.name,message:String(error.message).replace(/https?:\/\/\S+/g,'[URL_REDACTED]').slice(0,6000)};await save();throw Error(`E02 failed; preserve ${reg.evidence}; no automatic retry`);}
 finally{console.log(JSON.stringify({status:e.status,evidence:reg.evidence,roles:Object.fromEntries(Object.entries(e.roles).map(([name,row])=>[name,{sessionId:row.sessionId,status:row.status,artifact:row.final?.artifact_id}])),confirmations:e.confirmations.length}));}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await main();
