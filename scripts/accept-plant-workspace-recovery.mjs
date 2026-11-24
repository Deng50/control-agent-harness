/** Deliberate leaf ACL fault and model compilation fault, through the real Host. */
import assert from 'node:assert/strict';
import {randomUUID, createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {readFile, writeFile, copyFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {openLayer, project} from './matlab-layer-harness.mjs';
import {matlabString} from '../harness-plugin/src/simulink-native.mjs';

const [fixture,source,caseParameters] = process.argv.slice(2);
assert.ok(caseParameters,'Pass fixture directory, original uploaded plant_numpy.py, and case physical_parameters.json');
const id = randomUUID();
const name = 'acl-plant-' + id;
const workspace = join(project,'test-output',name);
const ps = (script,args) => JSON.parse(execFileSync('powershell.exe',['-NoProfile','-File',join(project,'scripts',script),...args],
  {encoding:'utf8',windowsHide:true}));
const before = ps('new-workspace-acl-fixture.ps1',['-Name',name]);
const evidence = {workspace,before,status:'running'};
const record = () => writeFile(join(workspace,'recovery.json'),JSON.stringify(evidence,null,2));
const sourceHash = createHash('sha256').update(await readFile(source)).digest('hex');
assert.equal(sourceHash,'c4f054989498454dd25c2f4f78668fcba29bad2231e51fba57d7ae737dbea7b9');
await copyFile(source,join(workspace,'plant_numpy.py'));
const p = JSON.parse(await readFile(caseParameters,'utf8'));
const dt=p.sample_time, physics={mass:p.mass,damping:p.damping,stiffness:p.stiffness};
const model = 'AclPlant_' + id.slice(0,8);
const args = {model,Ac:[[0,1],[-p.stiffness/p.mass,-p.damping/p.mass]],Bc:[[0,0],[1/p.mass,1/p.mass]],
  inputNames:['ControlForce','LoadForce'],sampleTime:dt,initialCondition:[0.5,0],referenceFile:'python-reference.json'};
const trajectories=[[0.5,0],[-0.27,0.17],[0,0]].map((initialCondition,i)=>({initialCondition,
  controls:Array.from({length:19},(_,k)=>[Math.sin(0.33*k+i)]),
  disturbances:Array.from({length:19},(_,k)=>[(k+i)%4===0 ? 0.11 : -0.07])}));
await writeFile(join(workspace,'request.json'),JSON.stringify({parameters:physics,sampleTime:dt,absTolerance:1e-12,trajectories}));
execFileSync(join(project,'.venv/Scripts/python.exe'),['-I','-B',join(project,'scripts/plant-step-reference.py'),
  '--source',join(workspace,'plant_numpy.py'),'--request',join(workspace,'request.json'),'--output',join(workspace,args.referenceFile)],{windowsHide:true});
const layer = await openLayer({fixture:resolve(fixture),workspace,managed:true});
async function success(tool,arg) {
  const r=await layer.call(tool,arg);assert.equal(r.isError,false,r.error?.message);return r.value;
}
async function fixtureAction(fixtureModel, action) {
  const dir=join(project,'tests/fixtures/matlab-scope'),target=join(workspace,fixtureModel+'.slx');
  const body="addpath("+matlabString(dir)+",'-end'); disp(jsonencode(control_compile_regression("+
    [fixtureModel,target,action].map(matlabString).join(',')+")));";
  const r=await success('mcp__matlab__evaluate_matlab_code',{code:body});
  return JSON.parse(r.content.map(c=>c.text).join('').trim());
}
try {
  evidence.pid=layer.marker.matlab_pid; evidence.session=layer.agent.id; evidence.source_sha256=sourceHash;
  const first = await layer.call('control_build_zoh_plant',args);
  assert.equal(first.isError,true);
  evidence.refusal=JSON.parse(first.error.message);
  assert.equal(evidence.refusal.stage,'workspace_preflight');
  assert.match(first.error.message,/check_WRITE_OWNER/);
  const denied=await layer.call('mcp__matlab__evaluate_matlab_code',{code:'disp(314159);'});
  assert.equal(denied.isError,true);
  assert.match(denied.error.message,/check_WRITE_OWNER/);
  evidence.sibling_denial={message:denied.error.message,info:denied.error.info};
  await record();
  evidence.repair=ps('prepare-workspace.ps1',['-Workspace',workspace,'-Prepare']);
  assert.ok(evidence.repair.prepared);
  assert.deepEqual(evidence.repair.added_leaf_rights,['WRITE_OWNER']);
  evidence.recheck=await success('control_recheck_plant_workspace',{});
  evidence.build=await success('control_build_zoh_plant',args);
  assert.equal(evidence.build.session_verified,true);
  assert.equal(evidence.build.matlab_pid,evidence.pid);
  assert.equal(evidence.build.comparison.samples,60);
  assert.ok(evidence.build.comparison.max_abs_error<=1e-12);
  const invalidSolver=await layer.call('control_simulate_simulink_model',{model,solver:'auto',solverType:'Fixed-step'});
  assert.equal(invalidSolver.isError,true);
  evidence.bad_solver=JSON.parse(invalidSolver.error.message);
  assert.equal(evidence.bad_solver.identifier,'ControlHarness:SolverCombination');
  assert.equal(evidence.bad_solver.session_verified,true);
  evidence.solver_retry=await success('control_simulate_simulink_model',{model,solver:'FixedStepDiscrete',
    solverType:'Fixed-step',stopTime:0.5,fixedStep:0.1});
  assert.equal(evidence.solver_retry.final_time,0.5);
  await record();
  const badModel='CompilePair_'+id.slice(0,8);
  evidence.nativeFixture=await fixtureAction(badModel,'create');
  assert.equal(evidence.nativeFixture.solver.identifier,'Simulink:Commands:BadSetParamValue');
  assert.equal(evidence.nativeFixture.width.identifier,'MATLAB:MException:MultipleErrors');
  assert.equal(evidence.nativeFixture.solver_recognized_in_fixed_operation,true);
  assert.equal(evidence.nativeFixture.solver_recognized_in_other_operation,false);
  assert.equal(evidence.nativeFixture.width_recognized,true);
  assert.equal(evidence.nativeFixture.mixed_tree_recognized,false);
  const failed=await layer.call('control_simulate_simulink_model',{model:badModel,stopTime:0.5,fixedStep:0.1});
  assert.equal(failed.isError,true);
  evidence.compile=JSON.parse(failed.error.message);
  await record();
  assert.equal(evidence.compile.stage,'compile');
  assert.equal(evidence.compile.session_verified,true);
  assert.equal(evidence.compile.retryable,true);
  assert.equal(evidence.compile.matlab_pid,evidence.pid);
  assert.deepEqual(evidence.compile.causes.map(c=>c.identifier).sort(),[
    'Simulink:Engine:PortDimsMismatch21','Simulink:blocks:MuxInValidPortWidths'].sort());
  evidence.fixed=await fixtureAction(badModel,'fix');
  evidence.rerun=await success('control_simulate_simulink_model',{model:badModel,stopTime:0.5,fixedStep:0.1});
  assert.equal(evidence.rerun.final_time,0.5);
  assert.equal(evidence.rerun.matlab_pid,evidence.pid);
  assert.equal(evidence.rerun.session_verified,true);
  const continuousModel='Continuous_'+id.slice(0,8);
  evidence.continuousFixture=await fixtureAction(continuousModel,'continuous');
  assert.equal(evidence.continuousFixture.saved_solver_error.identifier,'Simulink:Engine:InvalidSolver');
  const badContinuous=await layer.call('control_simulate_simulink_model',{model:continuousModel});
  assert.equal(badContinuous.isError,true);
  evidence.saved_solver_failure=JSON.parse(badContinuous.error.message);
  assert.equal(evidence.saved_solver_failure.identifier,'Simulink:Engine:InvalidSolver');
  assert.equal(evidence.saved_solver_failure.retryable,true);
  assert.equal(evidence.saved_solver_failure.matlab_pid,evidence.pid);
  evidence.solver_repair=await success('control_simulate_simulink_model',{model:continuousModel,
    stopTime:0.5,fixedStep:0.1,solver:'ode4',solverType:'Fixed-step'});
  assert.equal(evidence.solver_repair.final_time,0.5);
  assert.equal(evidence.solver_repair.matlab_pid,evidence.pid);
  assert.deepEqual(evidence.solver_repair.verified,['validation','load','compile','simulate']);
  evidence.status='passed';
} catch(error) {evidence.status='failed';evidence.reason=error.message;throw error;}
finally {await record();await layer.close();console.log(join(workspace,'recovery.json'));}
