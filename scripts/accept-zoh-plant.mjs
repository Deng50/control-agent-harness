/** Offline R2024b acceptance. Requires a fresh chat workspace and the private official fixture. */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { join, resolve, dirname } from 'node:path';
import { parseArgs } from 'node:util';
import { attachFixtureSource, configureFixtureFileGeneration, openLayer, project } from './matlab-layer-harness.mjs';
import { matlabString } from '../harness-plugin/src/simulink-native.mjs';

const {values} = parseArgs({options:{fixture:{type:'string'},case:{type:'string'},workspace:{type:'string'},source:{type:'string'}}});
assert.ok(values.fixture && values.case, 'Pass --fixture and --case (benchmark B01 inputs directory).');
const workspace = resolve(values.workspace ?? join(project,'test-output','zoh-chat-' + randomUUID()));
assert.equal(dirname(workspace),join(project,'test-output'));
await mkdir(workspace); // Fresh leaf only.
const report = {workspace,fixture:resolve(values.fixture),model_calls:[],status:'running'};
const evidencePath = join(workspace,'acceptance.json');
const json = path => readFile(path,'utf8').then(JSON.parse);
const checkpoint = () => writeFile(evidencePath,JSON.stringify(report,null,2));
const caseDir = resolve(values.case);
const initials = [[0.15,-0.1],[0,0],[-0.3,0.22]];
const trajectories = initials.map((initialCondition,index) => ({
  initialCondition,
  controls:Array.from({length:20},(_,k)=>[k<4 ? 0 : (1.1*Math.sin(0.71*k+index))]),
  disturbances:Array.from({length:20},(_,k)=>[k%4===index ? 0.13 : -0.03]),
}));
let layer;
try {
  // Acceptance invokes only this reviewed, synthetic benchmark file.
  const upload = resolve(values.source ?? join(caseDir,'plant_numpy.py'));
  const sourceData=await readFile(upload);
  const digest = createHash('sha256').update(sourceData).digest('hex');
  assert.equal(digest,'c4f054989498454dd25c2f4f78668fcba29bad2231e51fba57d7ae737dbea7b9');
  report.source = upload;
  report.source_sha256 = digest;
  execFileSync('powershell.exe',['-NoProfile','-File',join(project,'scripts/prepare-workspace.ps1'),'-Workspace',workspace,'-Prepare'],
    {windowsHide:true,stdio:'pipe'});
  layer = await openLayer({fixture:values.fixture,workspace,managed:true});
  await attachFixtureSource(layer,resolve(values.fixture),{name:'plant_numpy.py',data:sourceData});
  await configureFixtureFileGeneration(layer,resolve(values.fixture));
  report.session = layer.agent.session.id;
  report.matlab_pid = layer.marker.matlab_pid;
  const params = await json(join(caseDir,'physical_parameters.json'));
  const suffix = randomUUID().slice(0,8);
  const model = 'SpringPlant_' + suffix;
  const target = join(workspace,model+'.slx');
  const fixtureHelpers = join(project,'tests/fixtures/matlab-scope');
  const installed = await layer.call('mcp__matlab__evaluate_matlab_code',{code:
    'addpath(' + matlabString(fixtureHelpers) + ", '-end'); disp(which('control_harness_scope_regression'));"});
  assert.equal(installed.isError,false,installed.error?.message);
  assert.ok(installed.value.content.some(item => item.text?.includes(fixtureHelpers)));
  // Run the deliberate fault as a test fixture. The production wrapper and
  // terminal probe continue to resolve from the real installation directory.
  const scopeCall=await layer.call('mcp__matlab__evaluate_matlab_code',{
    code:"assert(strcmpi(which('control_harness_scope_regression'),"+matlabString(join(fixtureHelpers,'control_harness_scope_regression.m'))+
      ")); disp(jsonencode(control_harness_scope_regression("+[model,target,layer.agent.session.id].map(matlabString).join(',')+")));",
    project_path:join(project,'matlab')});
  assert.equal(scopeCall.isError,false,scopeCall.error?.message);
  const scopeReceipt=JSON.parse(scopeCall.value.content.map(item=>item.text).join('').trim());
  const resource=await json(join(layer.resource,'resource.json'));
  assert.ok(!resource.owner && !resource.quarantine);
  report.scope_terminal=resource.last_completion.evidence;
  assert.equal(report.scope_terminal.session_id,layer.marker.session_id);
  assert.equal(report.scope_terminal.matlab_pid,layer.marker.matlab_pid);
  report.scope = scopeReceipt;
  await checkpoint();
  assert.equal(scopeReceipt.status,'failed');
  assert.equal(scopeReceipt.identifier,'MATLAB:UndefinedFunction');
  assert.match(scopeReceipt.reason,/Ac/);
  assert.equal(scopeReceipt.recoverable,true);
  assert.equal(scopeReceipt.path,target);
  assert.equal(scopeReceipt.model,model);
  assert.equal(scopeReceipt.file_exists,false);
  assert.equal(scopeReceipt.matlab_pid,layer.marker.matlab_pid);
  // A second real Harness session chooses exactly the same cwd, model and
  // target. It may observe the refusal but cannot adopt the partial model.
  const otherChat = await openLayer({fixture:values.fixture,workspace,managed:true});
  try {
    assert.notEqual(otherChat.agent.session.id,layer.agent.session.id);
    const collision = await otherChat.call('control_build_zoh_plant',{model,Ac:[[0,1],[-2,-0.8]],
      Bc:[[0,0],[1,1]],initialCondition:initials[0],sampleTime:0.1,inputNames:['ControlForce','DisturbanceForce']});
    assert.equal(collision.isError,true);
    const refusal = JSON.parse(collision.error.message);
    assert.equal(refusal.identifier,'ControlHarness:BuildTarget');
    assert.deepEqual(refusal.changes,[]);
    assert.equal(refusal.file_exists,false);
    assert.equal(refusal.session_verified,true);
    report.other_chat = {session:otherChat.agent.session.id,refusal};
    await checkpoint();
  } finally {
    await otherChat.close();
  }
  const secondary = await json(join(project,'examples/spring-damper/reference.json'));
  const variants = [
    {model,source:join(caseDir,'physical_parameters.json'),dt:params.sample_time,
      physics:{mass:params.mass,damping:params.damping,stiffness:params.stiffness}},
    {model:'SpringPlantAlt_' + suffix,source:join(project,'examples/spring-damper/README.md'),
      dt:secondary.interface.sample_time,physics:{mass:1.1,damping:1.0,stiffness:2.3}},
  ];
  for (const [index,v] of variants.entries()) {
    const {mass,damping,stiffness} = v.physics;
    const Ac = [[0,1],[-stiffness/mass,-damping/mass]], Bc = [[0,0],[1/mass,1/mass]];
    if (index===1) {
      assert.ok(Math.abs(Ac[1][0]-secondary.A[1][0])<1e-14);
      assert.equal(Bc[1][0],secondary.B[1][0]);
    }
    const replay=await layer.call('control_replay_plant_source',{source:'plant_numpy.py',
      request:{parameters:v.physics,sampleTime:v.dt,absTolerance:1e-12,trajectories}});
    assert.equal(replay.isError,false,replay.error?.message);
    assert.equal(replay.value.source_sha256,digest);
    const result = await layer.call('control_build_zoh_plant',{model:v.model,Ac,Bc,
      initialCondition:initials[0],sampleTime:v.dt,inputNames:['ControlForce','DisturbanceForce'],
      pythonSource:'plant_numpy.py',referenceFile:replay.value.referenceFile});
    if (result.isError) throw new Error(result.error?.message ?? JSON.stringify(result.content));
    const receipt = result.value;
    report.model_calls.push({parameter_source:v.source,parameters:v.physics,dt:v.dt,replay:replay.value,receipt});
    await checkpoint();
    assert.equal(receipt.status,'simulated');
    assert.equal(receipt.matlab_pid,report.matlab_pid);
    assert.equal(receipt.session_verified,true);
    assert.deepEqual(receipt.reloaded_parameters,v.physics);
    assert.equal(receipt.comparison.basis,'python_replay');
    assert.equal(receipt.comparison.source.sha256,digest);
    assert.equal(receipt.comparison.samples,63);
    assert.ok(receipt.comparison.max_abs_error <= 1e-12);
    assert.equal((await stat(receipt.path)).size,receipt.bytes);
    if (index===0) {
      assert.deepEqual(receipt.prior_values.Ac,[[0,1],[-2,-0.5]]);
      assert.ok(receipt.prior_model_workspace.includes('Ac'));
      assert.deepEqual(receipt.updated_parameters.damping,{before:0.5,after:0.8});
    }
  }
  report.status = 'passed';
} catch (error) {
  report.status = 'failed'; report.error = error.message; throw error;
} finally {
  if (layer) await layer.close();
  await checkpoint();
  console.log(evidencePath);
}
