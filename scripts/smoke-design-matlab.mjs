/** S07 optional generated MATLAB single-point checks through the real native Host. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {command,connectHarness,writeSmokeEvidence} from './harness-rpc.mjs';
const {values}=parseArgs({options:{'design-evidence':{type:'string'},evidence:{type:'string'},'web-log':{type:'string'}}});
assert.ok(values['design-evidence']&&values.evidence,'Pass --design-evidence and --evidence');
const json=path=>readFile(path,'utf8').then(JSON.parse),path=resolve(values.evidence),source=await json(resolve(values['design-evidence']));
assert.equal(source.status,'PASS');assert.ok(Array.isArray(source.bundles));
assert.deepEqual(source.bundles.map(bundle=>bundle.key).sort(),[
  'draft:draft','explicit-model:direct','matlab_function:identify','nonlinear:direct','nonlinear:local','simulink:identify',
].sort(),'All frozen S07 routes must be present exactly once');
const evidence={schema_version:1,status:'RUNNING',started_at:new Date().toISOString(),confirmation_source:'test_fixture',
  source_evidence:resolve(values['design-evidence']),scope:'Actual generated MATLAB prediction/cost/one-QP check only; closed_loop not_run',
  predeclared_tolerances:{prediction:{atol:1e-6,rtol:1e-5},fixed_sequence_cost_and_margins:{atol:1e-9,rtol:1e-8},
    cross_solver_action:{atol:.005,rtol:1e-4},constraint_violation_max:1e-6},checks:[],jobs:[]};
const save=()=>writeSmokeEvidence(path,evidence),sleep=ms=>new Promise(done=>setTimeout(done,ms));await save();
function near(actual,expected,tolerance,label){
  if(Array.isArray(expected)){
    if(typeof actual==='number'&&expected.length===1)actual=[actual];
    assert.ok(Array.isArray(actual),label+' array');assert.equal(actual.length,expected.length,label+' width');
    return expected.reduce((maximum,value,index)=>Math.max(maximum,near(actual[index],value,tolerance,`${label}[${index}]`)),0);
  }
  assert.ok(Number.isFinite(actual)&&Number.isFinite(expected),label+' finite');
  const error=Math.abs(actual-expected);assert.ok(error<=tolerance.atol+tolerance.rtol*Math.abs(expected),`${label}: ${actual} != ${expected}`);return error;
}
try{
  const rpc=await connectHarness({logPath:values['web-log']});
  for(const bundle of source.bundles){
    if(bundle.matlab_kind==='unsupported_nonlinear_predictor'){
      evidence.checks.push({key:bundle.key,status:'NOT_APPLICABLE_NONLINEAR_PREDICTOR',scope:'Python NMPC remains the validated route; no forced affine conversion'});await save();continue;
    }
    const session=await rpc('session/create',{request:{cwd:bundle.workspace,agentPreset:'control-controller'}});
    let casePath=bundle.case_path;
    if(!casePath){casePath=resolve(bundle.workspace,'s07-draft-matlab-input.json');await writeFile(casePath,'{}\n');}
    const args={artifact_path:bundle.artifact_path,case_path:casePath,...(bundle.parameters_path?{parameters_path:bundle.parameters_path}:{})};
    const job=await command(rpc,session.sessionId,`/design-matlab-check ${JSON.stringify(args)}`);
    evidence.jobs.push({key:bundle.key,session_id:session.sessionId,...job});await save();
    const deadline=Date.now()+420000;let state;
    while(Date.now()<deadline){state=await command(rpc,session.sessionId,`/control-status ${job.jobId}`);
      if(['completed','failed','killed'].includes(state.job.status))break;await sleep(400);}
    assert.equal(state?.job.status,'completed',`Inspect this existing job before any retry: ${job.jobId}; ${JSON.stringify(state)}`);
    const rows=state.output.chunks.flatMap(chunk=>chunk.text.split('\n')).flatMap(line=>{try{return[JSON.parse(line)];}catch{return[];}});
    const result=rows.findLast(row=>row.event==='result')?.result;assert.ok(result,'No actual MATLAB result');
    const actual=result.result;assert.equal(actual.closed_loop,'not_run');assert.equal(actual.ga_trials,0);
    assert.equal(actual.wrapper_nonce,result.nonce,'Trusted wrapper result must match the submitted nonce');
    assert.equal(resolve(result.casePath),resolve(casePath));assert.equal(resolve(result.artifact),resolve(bundle.artifact_path));
    const checked={key:bundle.key,job_id:job.jobId,result,scope:'single-point diagnostic only'};
    if(!bundle.model_instantiated){assert.equal(actual.status,'not_instantiated');assert.equal(actual.model_instantiated,false);assert.equal(actual.missing_parameters_rejected,true);checked.status='PASS_PARAMETERIZED';}
    else{
      assert.equal(actual.model_instantiated,true);
      checked.prediction_error=near(actual.prediction.next_x,bundle.python_reference.prediction.next_x,evidence.predeclared_tolerances.prediction,'next state');
      near(actual.prediction.next_y,bundle.python_reference.prediction.next_y,evidence.predeclared_tolerances.prediction,'next output');
      near(actual.prediction.sample_time,bundle.python_reference.prediction.sample_time,{atol:1e-12,rtol:0},'dt');
      checked.fixed_sequence_cost_error=near(actual.cost,bundle.python_reference.cost,evidence.predeclared_tolerances.fixed_sequence_cost_and_margins,'same control-sequence cost');
      checked.fixed_sequence_margin_error=near(actual.constraints,bundle.python_reference.margins,evidence.predeclared_tolerances.fixed_sequence_cost_and_margins,'same control-sequence margins');
      if(actual.status==='blocked_toolbox_license'){checked.status='BLOCKED_OPTIMIZATION_LICENSE';evidence.checks.push(checked);await save();continue;}
      assert.equal(actual.status,'single_point_solved');assert.equal(actual.optimization_toolbox_license,true);
      assert.equal(actual.optimization_installed,true);assert.equal(actual.optimization_licensed,true);
      assert.equal(actual.solve.solver,'quadprog');assert.ok(actual.solve.exitflag>0);
      assert.ok(Number.isFinite(actual.solve.constraint_violation)&&actual.solve.constraint_violation<=evidence.predeclared_tolerances.constraint_violation_max);
      checked.action_error=near(actual.action,bundle.python_reference.action,evidence.predeclared_tolerances.cross_solver_action,'quadprog versus SLSQP action');
      checked.status='PASS_SINGLE_POINT';
    }
    evidence.checks.push(checked);await save();
  }
  evidence.status=evidence.checks.some(row=>row.status.startsWith('BLOCKED'))?'BLOCKED':'PASS';
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{evidence.finished_at=new Date().toISOString();await save();console.log(JSON.stringify({status:evidence.status,evidence:path,checks:evidence.checks.length,jobs:evidence.jobs.length}));}
