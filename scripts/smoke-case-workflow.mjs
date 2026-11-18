/** S03 native workflow scope/freeze/resume acceptance; fixed synthetic fixtures only. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command, connectHarness, prepareSmokeWorkspace, project, runStage, smokeOptions, writeSmokeEvidence } from './harness-rpc.mjs';

const options = smokeOptions('smoke-case-workflow');
if (options.help) process.exit(0);
const workspace = options.workspace;
const local = relative(resolve(project, 'test-output'), workspace);
assert.ok(local && !isAbsolute(local) && local !== '..' && !local.startsWith(`..${sep}`), 'Use a fresh project test-output fixture workspace.');
await prepareSmokeWorkspace(workspace);
try {
  const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
  assert.equal(Object.keys(manifest.artifacts ?? {}).length, 0, 'Use a new workspace; inspect existing job handles before rerunning.');
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const evidence = { schema_version: 1, status: 'RUNNING', confirmation_source: 'test_fixture', workspace,
  startedAt: new Date().toISOString(), scope: 'Real Harness workflow jobs, minimum confirmation scopes, immutable input freeze and verified-stage reuse. Synthetic simulator only; no LLM, NN training, hardware acquisition or service launch.',
  admissions: [], stages: [], confirmations: [], rejections: [] };
const hash = value => createHash('sha256').update(value).digest('hex');
const json = path => readFile(path, 'utf8').then(JSON.parse);
const python = process.env.CONTROL_PYTHON ?? resolve(project, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
await writeSmokeEvidence(options.evidence, evidence);

const generator = String.raw`import json,sys
from pathlib import Path
from control_harness.examples import example
root=Path(sys.argv[1]); fixture=example('linear')
for name in ('nominal','reference'):
    (root/(name+'.json')).write_text(json.dumps(fixture[name],indent=2)+'\n',encoding='utf-8')
print(json.dumps({'case':'linear','source':'public examples.example','confirmation_source':'test_fixture','synthetic':True,'data':{**fixture['data'],'trajectories':6,'steps':4}}))
`;
const readNpz = String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as values:
    result={name:values[name].tolist() for name in values.files}
print(json.dumps(result,allow_nan=False))
`;
function near(actual, expected, label) {
  assert.ok(Number.isFinite(actual), `${label} must be finite`);
  assert.ok(Math.abs(actual - expected) <= 1e-11 + 1e-10 * Math.abs(expected), `${label}: ${actual} != ${expected}`);
}

try {
  const fixtures = resolve(workspace, 'fixtures');
  await mkdir(fixtures, { recursive: true });
  await mkdir(resolve(workspace, '.case-plans'), { recursive: true });
  await writeSmokeEvidence(resolve(workspace, '.case-plans/test-fixture.json'), {
    kind: 'control-case-plan-test-fixture-v1', workspace, confirmation_source: 'test_fixture', synthetic: true,
  });
  evidence.fixture = JSON.parse(execFileSync(python, ['-c', generator, fixtures], {
    env: { ...process.env, PYTHONPATH: resolve(project, 'src') }, encoding: 'utf8', windowsHide: true,
  }));
  const nominalPath = resolve(fixtures, 'nominal.json'), referencePath = resolve(fixtures, 'reference.json');
  const nominal = await json(nominalPath), reference = await json(referencePath);
  const nominalHash = hash(await readFile(nominalPath)), referenceHash = hash(await readFile(referencePath));
  const registrationPath = resolve(fixtures, 'registration-only.json');
  const samplingPath = resolve(fixtures, 'sampling-only.json');
  const nominalStep = { id: 'nominal', operation: 'register_model', inputs: { model_path: nominalPath },
    config: { source: { kind: 'user_nominal', description: 'Fixed public linear mathematical test_fixture; synthetic, not measured.' } } };
  await writeSmokeEvidence(registrationPath, { schema_version: 1, name: 'test_fixture registration only', steps: [nominalStep] });
  await writeSmokeEvidence(samplingPath, { schema_version: 1, name: 'test_fixture sampling only', steps: [
    nominalStep,
    { id: 'reference', operation: 'register_model', inputs: { model_path: referencePath },
      config: { source: { kind: 'reference_simulator', description: 'Separate public linear mathematical test_fixture; synthetic, not measured.' } } },
    { id: 'samples', operation: 'simulate_transitions', inputs: { nominal_model: { stage: 'nominal' }, reference_model: { stage: 'reference' } },
      config: evidence.fixture.data },
  ] });
  evidence.sources = { nominal: { path: nominalPath, sha256: nominalHash }, reference: { path: referencePath, sha256: referenceHash },
    registrationWorkflow: { path: registrationPath, sha256: hash(await readFile(registrationPath)) },
    samplingWorkflow: { path: samplingPath, sha256: hash(await readFile(samplingPath)) }, confirmation_source: 'test_fixture' };

  const baseRpc = await connectHarness({ logPath: options.webLog });
  const rpc = async (method, payload) => {
    const value = await baseRpc(method, payload);
    if (method === 'commands/execute' && value.result?.kind === 'success') {
      const body = JSON.parse(value.result.text);
      if (body.jobId) {
        evidence.admissions.push({ jobId: body.jobId, attemptId: body.attemptId,
          operation: payload.line.startsWith('/control ') ? JSON.parse(payload.line.slice(9)).operation : payload.line });
        await writeSmokeEvidence(options.evidence, evidence);
      }
    }
    return value;
  };
  const created = await rpc('session/create', { request: { cwd: workspace, agentPreset: 'control-model' } });
  assert.equal(created.agentPreset, 'control-model');
  const sessionId = created.sessionId;
  evidence.sessionId = sessionId;
  const stage = async (operation, inputs = {}, config = {}) => {
    // The workflow executor is a registered tool role in the real model Session.
    const value = await runStage(rpc, sessionId, { agent: 'workflow', operation, workspace, inputs, config });
    evidence.stages.push({ operation, jobId: value.jobId, attemptId: value.attemptId, result: value.result });
    await writeSmokeEvidence(options.evidence, evidence);
    return value;
  };
  const events = async () => (await rpc('session/events', { request: { address: { kind: 'session', sessionId }, maxMessages: 10000 } })).records;
  const snapshot = async () => {
    let attempts = [];
    try { attempts = (await readdir(resolve(workspace, '.control-jobs'))).filter(name => /^[0-9a-f-]{36}$/.test(name)).sort(); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const history = await command(rpc, sessionId, `/control-history ${JSON.stringify({ workspace })}`);
    const records = await events();
    return { attempts, history: history.attempts.map(row => row.attemptId).sort(),
      registeredJobs: records.filter(row => row.event.type === 'job/registered').map(row => row.event.data.jobId),
      confirmationCommands: records.filter(row => row.event.type === 'command/run' && row.event.data.name === 'control-confirm').length };
  };
  const deny = async (label, request) => {
    const before = await snapshot();
    const response = await rpc('commands/execute', { agentId: sessionId, line: `/control ${JSON.stringify({ agent: 'workflow', ...request })}`, submittedAttachments: [] });
    assert.equal(response.result.kind, 'error', `${label} was admitted`);
    assert.match(response.result.text, /plan|reviewed|confirmation|version/i);
    const after = await snapshot();
    assert.deepEqual(after, before, `${label} created history/job or added confirmation`);
    evidence.rejections.push({ label, nativeKind: response.result.kind, message: response.result.text,
      newHistory: 0, newJobs: 0, before, after, confirmation_source: 'test_fixture' });
    await writeSmokeEvidence(options.evidence, evidence);
  };

  const registration = { operation: 'run_workflow', workspace, inputs: { workflow_path: registrationPath }, config: {} };
  assert.deepEqual(await confirmationRequirements(registration), []);
  assert.ok(!Object.hasOwn(registration.config, 'plan_refs'));
  const registrationPlan = await stage('plan_workflow', registration.inputs, registration.config);
  assert.equal(registrationPlan.result.status, 'ready');
  const registered = await stage('run_workflow', registration.inputs, registration.config);
  assert.equal(registered.result.status, 'verified');
  assert.deepEqual(Object.keys(registered.result.artifacts), ['nominal']);
  const afterRegistration = await json(resolve(workspace, 'manifest.json'));
  assert.deepEqual([...new Set(Object.values(afterRegistration.artifacts).map(row => row.kind))], ['nominal_model']);
  try { await readFile(resolve(workspace, '.case-plans/plans.json')); assert.fail('Registration-only workflow unnecessarily created engineering plans'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  evidence.registrationOnly = { scopes: [], requiredPlanRefs: false, result: registered.result, confirmation_source: 'test_fixture' };

  const request = { operation: 'run_workflow', workspace, inputs: { workflow_path: samplingPath }, config: {} };
  const requirements = await confirmationRequirements(request);
  assert.deepEqual(requirements.map(row => row.scope), ['signals', 'sampling']);
  assert.equal(requirements[0].bindings.workflow_inputs['nominal.model_path'].sha256, nominalHash);
  assert.equal(requirements[0].bindings.workflow_inputs['reference.model_path'].sha256, referenceHash);
  assert.equal(requirements[0].bindings.workflow_file.sha256, evidence.sources.samplingWorkflow.sha256);
  const plans = [];
  for (const requirement of requirements) {
    const proposed = await stage('propose_case_plan', {}, { scope: requirement.scope,
      payload: requirement.expectedPayload, bindings: requirement.bindings });
    plans.push(proposed.result);
  }
  request.config.plan_refs = Object.fromEntries(plans.map(plan => [plan.scope, { plan_id: plan.plan_id, version_hash: plan.version_hash }]));
  await deny('sampling workflow before confirmation', request);
  for (const plan of plans) {
    const reference = { workspace, plan_id: plan.plan_id, version_hash: plan.version_hash };
    const confirmed = await command(rpc, sessionId, `/control-confirm ${JSON.stringify(reference)}`);
    assert.equal(confirmed.confirmation_source, 'test_fixture');
    const status = await command(rpc, sessionId, `/control-plan-status ${JSON.stringify(reference)}`);
    assert.equal(status.confirmation_state, 'confirmed');
    assert.equal(status.confirmation_source, 'test_fixture');
    evidence.confirmations.push({ ...confirmed, confirmation_source: 'test_fixture' });
  }
  const preflight = await stage('plan_workflow', request.inputs, request.config);
  assert.equal(preflight.result.status, 'ready');
  assert.deepEqual(preflight.result.stages.map(row => row.operation), ['register_model', 'register_model', 'simulate_transitions']);
  const sampled = await stage('run_workflow', request.inputs, request.config);
  assert.equal(sampled.result.status, 'verified');
  assert.deepEqual(Object.keys(sampled.result.artifacts), ['nominal', 'reference', 'samples']);
  const manifest = await json(resolve(workspace, 'manifest.json'));
  const dataId = sampled.result.artifacts.samples;
  assert.equal(manifest.artifacts[dataId].kind, 'transition_dataset');
  assert.ok(Object.values(manifest.artifacts).every(row => ['nominal_model', 'reference_model', 'transition_dataset'].includes(row.kind)), 'Unexpected training/controller artifact');
  const inspected = await stage('inspect_artifact', { artifact_id: dataId });
  assert.equal(inspected.result.metadata.source.kind, 'reference_simulator');
  assert.equal(inspected.result.metadata.source.reference_artifact, sampled.result.artifacts.reference);
  assert.match(inspected.result.metadata.source.description, /not experimental/);
  assert.equal(inspected.result.metadata.validation.samples, 24);
  const preview = await stage('read_artifact_file', { artifact_id: dataId, file: 'transitions.npz' });
  assert.ok(preview.result.file_preview.arrays.some(row => row.name === 'x' && row.shape[0] === 24 && row.shape[1] === 4));
  const dataRoot = resolve(workspace, manifest.artifacts[dataId].path);
  const arrays = JSON.parse(execFileSync(python, ['-c', readNpz, resolve(dataRoot, 'transitions.npz')], { encoding: 'utf8', windowsHide: true }));
  assert.deepEqual([...new Set(arrays.trajectory_id)], [0, 1, 2, 3, 4, 5]);
  for (let trajectory = 0; trajectory < 6; trajectory++) assert.equal(arrays.trajectory_id.filter(id => id === trajectory).length, 4);
  for (let row = 0; row < 24; row++) {
    for (let state = 0; state < 4; state++) {
      const expected = reference.A[state].reduce((sum, value, col) => sum + value * arrays.x[row][col], 0)
        + reference.B[state].reduce((sum, value, col) => sum + value * arrays.u[row][col], 0)
        + reference.E[state].reduce((sum, value, col) => sum + value * arrays.d[row][col], 0);
      near(arrays.next_x[row][state], expected, `synthetic transition ${row}/${state}`);
    }
  }
  const record = await json(sampled.result.record_path);
  assert.equal(record.status, 'completed');
  const frozen = [];
  for (const [stageId, expected] of [['nominal', nominalHash], ['reference', referenceHash]]) {
    const path = record.definition.steps.find(row => row.id === stageId).inputs.model_path;
    assert.notEqual(path, stageId === 'nominal' ? nominalPath : referencePath);
    assert.ok(path.includes('workflow-cache'), 'Confirmed workflow did not use its frozen input cache');
    assert.equal(hash(await readFile(path)), expected);
    frozen.push({ stageId, path, sha256: expected });
  }
  evidence.samplingOnly = { scopes: requirements.map(row => row.scope), result: sampled.result,
    synthetic: true, trajectories: 6, stepsPerTrajectory: 4, samples: 24, independentlyCheckedTransitions: 24,
    noTrainingStage: true, frozen, confirmation_source: 'test_fixture' };

  const beforeResume = await snapshot();
  const artifactIds = Object.keys(manifest.artifacts).sort();
  const resume = structuredClone(request);
  resume.config.resume_run = sampled.result.workflow_run;
  assert.deepEqual(await confirmationRequirements(resume), requirements);
  const resumed = await stage('run_workflow', resume.inputs, resume.config);
  assert.equal(resumed.result.workflow_run, sampled.result.workflow_run);
  assert.deepEqual(resumed.result.artifacts, sampled.result.artifacts);
  assert.deepEqual(resumed.result.steps, sampled.result.steps);
  assert.deepEqual(Object.keys((await json(resolve(workspace, 'manifest.json'))).artifacts).sort(), artifactIds);
  const afterResume = await snapshot();
  assert.equal(afterResume.confirmationCommands, beforeResume.confirmationCommands, 'Completed-run reuse requested another confirmation');
  const resumeOutput = await command(rpc, sessionId, `/control-status ${resumed.jobId}`);
  const progress = resumeOutput.output.chunks.map(chunk => chunk.text).join('').split('\n').flatMap(line => {
    try { return [JSON.parse(line)]; } catch { return []; }
  }).filter(row => row.event === 'progress' && row.step_status === 'reused_verified');
  assert.deepEqual(progress.map(row => row.step), ['nominal', 'reference', 'samples']);
  evidence.resume = { result: resumed.result, newArtifacts: 0, newConfirmations: 0,
    reusedStages: progress.map(row => row.step), newNativeJob: true, confirmation_source: 'test_fixture' };

  const changed = structuredClone(nominal);
  changed.A[0][0] += .001;
  await writeSmokeEvidence(nominalPath, changed);
  const changedHash = hash(await readFile(nominalPath));
  assert.notEqual(changedHash, nominalHash);
  await deny('same model path with changed bytes and old refs', resume);
  for (const file of frozen) assert.equal(hash(await readFile(file.path)), file.sha256, 'Original-file edit changed frozen bytes');
  evidence.changedSource = { path: nominalPath, oldSha256: nominalHash, newSha256: changedHash,
    samePath: true, oldRefsRejectedBeforeJob: true, frozenInputsUnchanged: true, confirmation_source: 'test_fixture' };
  const records = await events();
  evidence.eventTypes = [...new Set(records.map(row => row.event.type))];
  assert.ok(!evidence.eventTypes.some(type => type.startsWith('turn/')), 'S03 fixed fixture opened an LLM turn');
  evidence.status = 'PASS';
} catch (error) {
  evidence.status = 'FAIL';
  evidence.error = error instanceof Error ? error.stack : String(error);
  throw error;
} finally {
  evidence.finishedAt = new Date().toISOString();
  await writeSmokeEvidence(options.evidence, evidence);
  console.log(JSON.stringify({ status: evidence.status, confirmation_source: 'test_fixture', workspace,
    evidence: options.evidence, admissions: evidence.admissions.map(row => ({ operation: row.operation, jobId: row.jobId })),
    rejections: evidence.rejections.length, scopes: evidence.samplingOnly?.scopes, reusedStages: evidence.resume?.reusedStages }));
}
