/** S04 real native-MCP MATLAB and SLX batches against the frozen F01 oracle. */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, isAbsolute, relative, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { command, connectHarness, prepareSmokeWorkspace, project, writeSmokeEvidence } from './harness-rpc.mjs';

const {values} = parseArgs({options:{workspace:{type:'string'},evidence:{type:'string'},fixtures:{type:'string'},'web-log':{type:'string'}}});
assert.ok(values.fixtures && isAbsolute(values.fixtures), 'Pass the explicit frozen --fixtures directory');
const fixtures=resolve(values.fixtures), workspace=resolve(values.workspace??resolve(project,'test-output',`plant-matlab-${randomUUID()}`));
const evidencePath=resolve(values.evidence??resolve(project,'test-output/smoke-evidence',`plant-matlab-${randomUUID()}.json`));
const distance=relative(resolve(project,'test-output'),workspace);
assert.ok(distance&&!isAbsolute(distance)&&distance!=='..'&&!distance.startsWith(`..${sep}`),'Use an isolated project test-output test_fixture workspace');
const json=path=>readFile(path,'utf8').then(JSON.parse), sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const sleep=ms=>new Promise(done=>setTimeout(done,ms));
const evidence={schema_version:1,status:'RUNNING',workspace,confirmation_source:'test_fixture',startedAt:new Date().toISOString(),
  scope:'S04 actual native jobs and one local MCP batch per collection; no LLM turn, Engine or NN training',admissions:[],stages:[],checks:[],rejections:[],confirmations:[]};
const save=()=>writeSmokeEvidence(evidencePath,evidence);
await save();
const python=resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const arraysAt=artifact=>JSON.parse(execFileSync(python,['-c',
  'import sys,json,numpy as np\nwith np.load(sys.argv[1],allow_pickle=False) as d: print(json.dumps({k:d[k].tolist() for k in d.files},allow_nan=False))',
  resolve(artifact.artifact_path,'transitions.npz')],{encoding:'utf8',windowsHide:true,maxBuffer:8*1024*1024,env:{...process.env,PYTHONUTF8:'1',PYTHONIOENCODING:'utf-8'}}));
function near(actual,expected,gate,label){
  if(Array.isArray(expected)){assert.equal(actual.length,expected.length,label);return expected.reduce((m,x,k)=>Math.max(m,near(actual[k],x,gate,`${label}[${k}]`)),0);}
  assert.ok(Number.isFinite(actual),label);const error=Math.abs(actual-expected);
  assert.ok(error<=gate.atol+gate.rtol*Math.abs(expected),`${label}: ${actual} != ${expected}`);return error;
}
function spring(x,u,d,dt,p){
  const a=p.damping/(2*p.mass),w=Math.sqrt(p.spring/p.mass-a*a),q=(u[0]+d[0])/p.spring,z=x[0]-q;
  const e=Math.exp(-a*dt),s=Math.sin(w*dt)/w,c=Math.cos(w*dt);
  return [q+e*((c+a*s)*z+s*x[1]),e*(-p.spring/p.mass*s*z+(c-a*s)*x[1])];
}
try{
  await prepareSmokeWorkspace(workspace);
  try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts).length,0,'Use a fresh workspace');}catch(e){if(e.code!=='ENOENT')throw e;}
  const freeze=await json(resolve(fixtures,'freeze_manifest.json')),frozen={};
  for(const name of ['F01_spring_2x1.json','acceptance_thresholds.json']){
    const bytes=await readFile(resolve(fixtures,name));assert.equal(sha(bytes),freeze.files.find(x=>basename(x.path)===name)?.sha256,'Frozen oracle changed');frozen[name]=JSON.parse(bytes);
  }
  const f=frozen['F01_spring_2x1.json'],gate=frozen['acceptance_thresholds.json'].gates.F01_exact_ZOH_single_and_multi_step;
  const iface={...structuredClone(f.source_model.interface),time_semantics:'discrete_state_transition'};
  const physics={mass:f.physical_parameters.mass_kg,damping:f.physical_parameters.damping_Ns_per_m,spring:f.physical_parameters.spring_N_per_m};
  evidence.frozenVersion=freeze.version;evidence.gate=gate;
  await mkdir(resolve(workspace,'.case-plans'),{recursive:true});
  await writeSmokeEvidence(resolve(workspace,'.case-plans/test-fixture.json'),{kind:'control-case-plan-test-fixture-v1',workspace});
  await mkdir(resolve(workspace,'sources'),{recursive:true});
  for(const name of ['f01_exact_step.m','f01_derivative.m','f01_spring.slx'])await copyFile(resolve(project,'matlab/fixtures',name),resolve(workspace,'sources',name));
  await writeFile(resolve(workspace,'sources/f01_diverge.m'),'function next_x=f01_diverge(x,u,d,dt,p)\nnext_x=[x(1)+.3;x(2)];\nend\n');
  const originalSlxHash=sha(await readFile(resolve(workspace,'sources/f01_spring.slx')));
  const rpc=await connectHarness({logPath:values['web-log']??resolve(project,'harness-web.log')});
  const session=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});evidence.sessionId=session.sessionId;
  // One admission per request. Observation expiration records the same handle;
  // it never creates a replacement for an active external calculation.
  async function stage(operation,inputs={},config={},expected='completed',card_id){
    const started=await command(rpc,session.sessionId,`/control ${JSON.stringify({agent:'model',operation,workspace,inputs,config,card_id})}`);
    evidence.admissions.push({...started,operation});await save();
    const deadline=Date.now()+420000;let state;
    while(Date.now()<deadline){
      state=await command(rpc,session.sessionId,`/control-status ${started.jobId}`);
      if(['completed','failed','killed'].includes(state.job.status))break;
      await sleep(500);
    }
    if(!['completed','failed','killed'].includes(state?.job.status))throw new Error(`Observer expired; resume the existing job ${started.jobId}, never resubmit`);
    const text=state.output.chunks.map(x=>x.text).join('');
    const events=text.split('\n').flatMap(line=>{try{return[JSON.parse(line)];}catch{return[];}});
    const result=events.findLast(x=>x.event==='result')?.result;
    evidence.stages.push({operation,...started,status:state.job.status,detail:state.job.detail,result,workerResultPhases:events.filter(x=>x.event==='result').map(x=>x.result.matlab_phase??x.result.kind??null)});await save();
    assert.equal(state.job.status,expected,JSON.stringify(state));
    return expected==='completed'?(assert.ok(result,'No terminal result'),result):{state,events};
  }
  async function register(name,declaration){
    const material=await stage('register_material',{local_path:resolve(workspace,'sources',name)},{source_kind:'path'});
    const adapter={...declaration,source_file:`source/${name}`};
    const plant=await stage('register_plant',{material:material.artifact_id},{adapter});
    const report=await json(resolve(plant.artifact_path,'report.json'));assert.equal(report.source_executed,false);
    return plant;
  }
  async function confirm(request){
    const card=await command(rpc,session.sessionId,`/control-prepare-review ${JSON.stringify({agent:'model',...request})}`);
    assert.equal(card.matlab_scope.operation,request.operation);
    request.card_id=card.card_id;
    return async()=>{
      const accepted=await command(rpc,session.sessionId,`/control-confirm ${JSON.stringify({workspace,card_id:card.card_id})}`);
      assert.equal(accepted.confirmation_source,'test_fixture');
      assert.equal(accepted.matlab_grant.state,'active');
      evidence.confirmations.push({card_id:card.card_id,source:accepted.source,grant:accepted.matlab_grant});
      await save();
    };
  }
  async function pending(request){
    const before=(await readdir(resolve(workspace,'.control-jobs'))).sort();
    const result=await rpc('commands/execute',{agentId:session.sessionId,line:`/control ${JSON.stringify({agent:'model',...request})}`,submittedAttachments:[]});
    assert.equal(result.result.kind,'error');assert.deepEqual((await readdir(resolve(workspace,'.control-jobs'))).sort(),before);
    evidence.rejections.push({kind:'pending_before_native_job',zeroNewHistory:true,detail:result.result.text});await save();
  }
  const conditions=[{x0:f.known_probe.x0,u:[f.known_probe.u],d:[f.known_probe.d]},
    {x0:[-.1,.05],u:[.2],d:[.03]},{x0:[0,0],u:[-.15],d:[-.01]},
    {x0:[.05,-.05],u:[.1],d:[0]},{x0:[-.15,.08],u:[0],d:[.02]}];
  const sampling={mode:'expert',seed:f.seed,dt:iface.sample_time,steps:50,control_rate_limits:null,disturbance_rate_limits:null,coupled_constraints:[],
    trajectories:conditions.map((x,trajectory_id)=>({trajectory_id,x0:x.x0,u0:x.u,d0:x.d,u_profile:{kind:'step',times:[0],values:[x.u]},d_profile:{kind:'step',times:[0],values:[x.d]}}))};
  async function compare(artifact,backend){
    const plan=await json(resolve(artifact.artifact_path,'sampling-plan.json')),data=arraysAt(artifact);let row=0,error=0;
    for(const trajectory of plan.trajectories){let x=trajectory.x0;
      for(let k=0;k<plan.steps;k++,row++){
        assert.equal(data.trajectory_id[row],trajectory.trajectory_id);assert.deepEqual(data.u[row],trajectory.u[k]);assert.deepEqual(data.d[row],trajectory.d[k]);
        near(data.time[row],trajectory.t[k],{atol:1e-12,rtol:0},'time');error=Math.max(error,near(data.x[row],x,gate,'state'));
        near(data.y[row],[x[0]],gate,'output');x=spring(x,trajectory.u[k],trajectory.d[k],plan.dt,physics);
        error=Math.max(error,near(data.next_x[row],x,gate,'next state'));near(data.next_y[row],[x[0]],gate,'next output');
      }
    }
    assert.equal(data.x.length,row);const report=await json(resolve(artifact.artifact_path,'report.json'));
    assert.equal(report.adapter.backend,backend);assert.equal(report.training,'not_run');
    assert.ok(report.confirmation_sources.every(x=>x.confirmation_source==='test_fixture'));
    assert.equal(report.adapter.stop_detection,backend==='simulink'?'post_trajectory_validation':'stepwise_validation');
    return{data,error,report};
  }
  const common={schema_version:1,interface:iface,parameters:{},source_semantics:'transition'};
  const declarations=[['f01_exact_step.m',{...common,backend:'matlab_function',entrypoint:'f01_exact_step',C:f.source_model.C}],
    ['f01_derivative.m',{...common,backend:'matlab_function',source_semantics:'derivative',entrypoint:'f01_derivative',C:f.source_model.C,
      time_unit:'s',derivative_units:f.source_model.derivative_units,integration:{method:'rk4',substeps:8,input_hold:'zero_order_hold'}}],
    ['f01_spring.slx',{...common,backend:'simulink',simulink:{controls:[{port:1,sid:'1'}],disturbances:[{port:2,sid:'2'}],outputs:[{port:1,sid:'15'}],
      states:[{sid:'5',initial_condition_variable:'S02_x01'},{sid:'6',initial_condition_variable:'S02_x02'}],sample_time_variable:'S02_dt',solver:'FixedStepDiscrete'}}]];
  for(const[name,spec]of declarations){
    const plant=await register(name,spec),request={operation:'sample_model',workspace,inputs:{model:plant.artifact_id},config:{sampling,trial:{trajectories:1,steps:3},sampling_run:'trial'}};
    const approve=await confirm(request);await pending(request);await approve();
    const trial=await stage('sample_model',request.inputs,request.config,'completed',request.card_id),trialCheck=await compare(trial,spec.backend);
    request.config.sampling_run='full';const full=await stage('sample_model',request.inputs,request.config,'completed',request.card_id),actual=await compare(full,spec.backend);
    assert.deepEqual(trialCheck.data.next_x,actual.data.next_x.slice(0,3));
    await approve(); // Repeat a consumed full batch only after a fresh fixture user command.
    const repeated=await stage('sample_model',request.inputs,request.config,'completed',request.card_id),second=await compare(repeated,spec.backend);
    near(second.data.next_x,actual.data.next_x,{atol:1e-13,rtol:1e-13},'batch reset repeat');
    assert.equal(sha(await readFile(resolve(workspace,'sources/f01_spring.slx'))),originalSlxHash,'Original SLX changed');
    evidence.checks.push({fixture:name,backend:spec.backend,trial:trial.artifact_id,full:full.artifact_id,repeated:repeated.artifact_id,
      samples:actual.data.x.length,maxAbsError:actual.error,trialPrefix:true,confirmationReusedForFull:true,repeatPassed:true,stopDetection:actual.report.adapter.stop_detection});await save();
  }
  const divergent=await register('f01_diverge.m',{...common,backend:'matlab_function',entrypoint:'f01_diverge',C:f.source_model.C});
  const bad={operation:'sample_model',workspace,inputs:{model:divergent.artifact_id},config:{sampling}};await(await confirm(bad))();
  const beforeFailures=new Set(await readdir(resolve(workspace,'failed')).catch(e=>e.code==='ENOENT'?[]:Promise.reject(e)));
  const rejected=await stage('sample_model',bad.inputs,bad.config,'failed',bad.card_id);
  const added=(await readdir(resolve(workspace,'failed'))).filter(x=>!beforeFailures.has(x));assert.equal(added.length,1);
  const failure=await json(resolve(workspace,'failed',added[0],'report.json'));
  assert.equal(failure.status,'failed');assert.equal(failure.completed_trajectories,0);assert.ok(failure.valid_rows<50);
  assert.equal(failure.adapter.stop_detection,'stepwise_validation');
  assert.ok((await readdir(resolve(workspace,'failed',added[0]))).includes('partial-transitions.npz'));
  evidence.rejections.push({kind:'real_matlab_runtime_divergence',job:rejected.state.job.id,failedStage:added[0],validRows:failure.valid_rows,reason:failure.failure.reason});
  const events=await rpc('session/events',{request:{address:{kind:'session',sessionId:session.sessionId},maxMessages:20000}});
  evidence.eventTypes=[...new Set(events.records.map(x=>x.event.type))];assert.ok(!evidence.eventTypes.some(x=>x.startsWith('turn/')));
  evidence.originalSlxSha256=originalSlxHash;evidence.status='PASS';
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{evidence.finishedAt=new Date().toISOString();await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,workspace,admissions:evidence.admissions.length,checks:evidence.checks.length,rejections:evidence.rejections.length}));}
