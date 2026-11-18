/** One additional real recommended-mode case on an explicitly selected verified plant. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command, connectHarness, project, runStage, writeSmokeEvidence } from './harness-rpc.mjs';
const {values}=parseArgs({options:{workspace:{type:'string'},plant:{type:'string'},evidence:{type:'string'},fixtures:{type:'string'}}});
assert.ok(values.workspace&&values.plant&&values.evidence&&values.fixtures,'Pass explicit --workspace --plant --evidence --fixtures');
const workspace=resolve(values.workspace),path=resolve(values.evidence);
const evidence={status:'RUNNING',confirmation_source:'test_fixture',workspace,model:values.plant,started_at:new Date().toISOString(),jobs:[]};
const save=()=>writeSmokeEvidence(path,evidence);
await save();
try{
  const fixtureRoot=resolve(values.fixtures),frozen=JSON.parse(await readFile(resolve(fixtureRoot,'freeze_manifest.json'),'utf8'));
  const thresholdBytes=await readFile(resolve(fixtureRoot,'acceptance_thresholds.json'));
  assert.equal(createHash('sha256').update(thresholdBytes).digest('hex'),frozen.files.find(x=>x.path.endsWith('/acceptance_thresholds.json'))?.sha256);
  const gate=JSON.parse(thresholdBytes).gates.F01_exact_ZOH_single_and_multi_step;
  const rpc=await connectHarness(),session=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});
  evidence.session_id=session.sessionId;
  const stage=async(operation,inputs={},config={})=>{
    const result=await runStage(rpc,session.sessionId,{agent:'model',operation,workspace,inputs,config});
    evidence.jobs.push({operation,...result});await save();return result.result;
  };
  const proposed=await stage('recommend_sampling',{model:values.plant},{overrides:{steps:50,seed:20261117}});
  assert.equal(proposed.source_executed,false);assert.equal(proposed.sampling.mode,'recommended');
  const request={operation:'sample_model',workspace,inputs:{model:values.plant},
    config:{sampling:proposed.sampling,trial:{trajectories:1,steps:3},sampling_run:'trial'}};
  const dependencies={},refs={};
  for(const r of await confirmationRequirements(request)){
    const p=await stage('propose_case_plan',{}, {scope:r.scope,payload:r.expectedPayload,bindings:r.bindings,dependencies:structuredClone(dependencies)});
    refs[p.scope]={plan_id:p.plan_id,version_hash:p.version_hash};dependencies[p.scope]=refs[p.scope];
  }
  request.config.plan_refs=refs;
  for(const ref of Object.values(refs)){
    const proof=await command(rpc,session.sessionId,`/control-confirm ${JSON.stringify({workspace,...ref})}`);
    assert.equal(proof.confirmation_source,'test_fixture');
  }
  const trial=await stage('sample_model',request.inputs,request.config);
  request.config.sampling_run='full';const full=await stage('sample_model',request.inputs,request.config);
  const read=async artifact=>({plan:JSON.parse(await readFile(resolve(artifact.artifact_path,'sampling-plan.json'),'utf8')),
    arrays:JSON.parse(execFileSync(resolve(project,'.venv/Scripts/python.exe'),['-c',
      'import numpy as np,sys,json\nwith np.load(sys.argv[1],allow_pickle=False) as a: print(json.dumps({k:a[k].tolist() for k in a.files}))',
      resolve(artifact.artifact_path,'transitions.npz')],{encoding:'utf8',windowsHide:true,maxBuffer:4*1024*1024}))});
  const a=await read(trial),b=await read(full);
  assert.equal(b.plan.mode,'recommended');assert.equal(b.arrays.x.length,250);
  assert.deepEqual(a.arrays.next_x,b.arrays.next_x.slice(0,3));
  assert.deepEqual(b.plan.config,proposed.sampling,'The executed proposal must retain exact normalized conditions');
  let maxError=0;
  for(let k=0;k<b.arrays.x.length;k++){
    const x=b.arrays.x[k],force=b.arrays.u[k][0]+b.arrays.d[k][0],dt=b.plan.dt;
    const alpha=.4,w=Math.sqrt(2-alpha*alpha),q=force/2,z=x[0]-q,e=Math.exp(-alpha*dt),s=Math.sin(w*dt)/w,c=Math.cos(w*dt);
    const expected=[q+e*((c+alpha*s)*z+s*x[1]),e*(-2*s*z+(c-alpha*s)*x[1])];
    expected.forEach((v,i)=>{const error=Math.abs(v-b.arrays.next_x[k][i]);maxError=Math.max(maxError,error);assert.ok(error<=gate.atol+gate.rtol*Math.abs(v));});
  }
  evidence.proposed_config=proposed.sampling;evidence.trial=trial;evidence.full=full;evidence.max_abs_error=maxError;
  evidence.plan_hash=b.plan.plan_hash;evidence.trial_exact_prefix=true;evidence.same_confirmation=true;evidence.status='PASS';
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{evidence.finished_at=new Date().toISOString();await save();console.log(JSON.stringify({status:evidence.status,evidence:path,jobs:evidence.jobs.length}));}
