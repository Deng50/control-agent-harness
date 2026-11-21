/** Keyless acceptance over the real authenticated Web RPC and native commands/jobs. */
import { readFile, mkdir, writeFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const project = resolve(import.meta.dirname, '..');
const log = await readFile(resolve(project, 'harness-web.log'), 'utf8');
const startupUrl = log.match(/dsh web: (http[^\s]+)/)?.[1];
assert.ok(startupUrl, 'The real Harness Web process must be running');
const url = new URL(startupUrl);
const auth = await fetch(url, { redirect: 'manual', signal: AbortSignal.timeout(10000) });
const cookie = auth.headers.getSetCookie().map(item => item.split(';')[0]).join('; ');
assert.ok(cookie, 'Harness browser authentication did not return a cookie');
async function rpc(method, payload) {
  const response = await fetch(new URL(`/api/${method}`, url), {
    method: 'POST', headers: { cookie, origin: url.origin, 'content-type': 'application/json' },
    body: JSON.stringify({ type: 'client-request', rpcId: randomUUID(), method, payload: { args: payload } }), signal: AbortSignal.timeout(30000),
  });
  assert.equal(response.status, 200, `HTTP ${response.status}: ${await response.clone().text()}`);
  const envelope = await response.json();
  assert.ok(envelope.result?.ok ?? envelope.ok, JSON.stringify(envelope));
  return (envelope.result ?? envelope).value;
}

const workspace = resolve(project, 'test-output', `harness-smoke-${randomUUID()}`);
await mkdir(workspace, { recursive: true });
if (process.platform === 'win32') execFileSync('pwsh.exe', ['-NoProfile', '-File', resolve(project, 'scripts/prepare-workspace.ps1'), '-Prepare', '-Workspace', workspace]);
const created = await rpc('session/create', { request: { cwd: workspace } });
let sessionId = created.sessionId;
const commands = await rpc('commands/list', { agentId: sessionId });
assert.ok(commands.some(item => item.name === 'control'));
async function command(line, submittedAttachments = []) {
  const value = await rpc('commands/execute', { agentId: sessionId, line, submittedAttachments });
  assert.equal(value.result.kind, 'success', JSON.stringify(value));
  return JSON.parse(value.result.text);
}
const started = await command(`/control ${JSON.stringify({ agent: 'model', operation: 'inspect_workspace', workspace })}`);
let observation;
const deadline = Date.now() + 30000;
while (Date.now() < deadline) {
  observation = await command(`/control-status ${started.jobId}`);
  if (['completed', 'failed', 'killed'].includes(observation.job.status)) break;
  await new Promise(resolve => setTimeout(resolve, 400));
}
assert.equal(observation.job.status, 'completed', JSON.stringify(observation));
const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
assert.equal(manifest.schema_version, 1);
const rejected = await rpc('commands/execute', { agentId: sessionId, line: '/control {"agent":"model","operation":"invented","workspace":"ignored"}', submittedAttachments: [] });
assert.equal(rejected.result.kind, 'error');
const modelBytes = await readFile(resolve(project, 'examples/linear/nominal.json'));
const upload = await rpc('fileUploads/upload', { agentId: sessionId, request: { data: modelBytes.toString('base64'), name: 'nominal.json' } });
const admitted = await command('/control-upload', [{ type: 'file', receiptId: upload.receiptId }]);
assert.equal(admitted.bytes, modelBytes.length);
const imported = await command(`/control ${JSON.stringify({ agent: 'model', operation: 'register_model', workspace,
  inputs: { model_path: admitted.path }, config: { source: { kind: 'user_nominal', description: 'Explicit linear demonstration equations uploaded through Harness' } } })}`);
const importDeadline = Date.now() + 30000;
let importObservation;
while (Date.now() < importDeadline) {
  importObservation = await command(`/control-status ${imported.jobId}`);
  if (['completed', 'failed', 'killed'].includes(importObservation.job.status)) break;
  await new Promise(resolve => setTimeout(resolve, 400));
}
assert.equal(importObservation.job.status, 'completed', JSON.stringify(importObservation));
const afterImport = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
assert.ok(afterImport.active.nominal_model);
const evidence = { checkedAt: new Date().toISOString(), sessionId, jobId: started.jobId,
  status: observation.job.status, workspace, invalidOperationRejected: true, uploadedModelImported: afterImport.active.nominal_model,
  scope: 'Real Harness Web authentication, generated RPC, command registry, native job, confined Python subprocess, staged file upload/admission and model artifact. No model API call.' };
await writeFile(resolve(project, 'test-output', 'harness-smoke.json'), JSON.stringify(evidence, null, 2));
console.log(JSON.stringify(evidence, null, 2));

// Optional full numerical acceptance under the same native Harness sandbox.
const workflowMode = process.argv[2]?.startsWith('workflow-');
const demoCase = process.argv[2]?.replace(/^workflow-/, '');
if (demoCase) {
  assert.ok(['linear', 'nonlinear'].includes(demoCase));
  const demoWorkspace = resolve(project, 'test-output', `harness-${demoCase}-${randomUUID()}`);
  await mkdir(demoWorkspace, { recursive: true });
  if (process.platform === 'win32') execFileSync('pwsh.exe', ['-NoProfile', '-File', resolve(project, 'scripts/prepare-workspace.ps1'), '-Prepare', '-Workspace', demoWorkspace]);
  sessionId = (await rpc('session/create', { request: { cwd: demoWorkspace } })).sessionId;
  const workflowInput = { workflow_path: resolve(project, 'examples', demoCase, 'workflow.json') };
  const workflowRequest = { agent: 'workflow', operation: 'run_workflow', workspace: demoWorkspace, inputs: workflowInput };
  const demo = await command(`/control ${JSON.stringify(workflowMode ? workflowRequest : { agent: 'workflow', operation: 'demo', workspace: demoWorkspace, config: { case: demoCase } })}`);
  console.log(`Native Harness demo started: ${demo.jobId}`);
  const demoDeadline = Date.now() + 600000;
  let demoStatus;
  while (Date.now() < demoDeadline) {
    demoStatus = await command(`/control-status ${demo.jobId}`);
    if (['completed', 'failed', 'killed'].includes(demoStatus.job.status)) break;
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  assert.equal(demoStatus.job.status, 'completed', JSON.stringify(demoStatus));
  let demoResult;
  if (workflowMode) {
    const records = (await readdir(resolve(demoWorkspace, 'workflows'))).filter(name => name.endsWith('.json'));
    assert.equal(records.length, 1);
    const record = JSON.parse(await readFile(resolve(demoWorkspace, 'workflows', records[0]), 'utf8'));
    assert.equal(record.status, 'completed');
    demoResult = { status: 'verified', workflow_run: record.run_id, artifacts: Object.fromEntries(Object.entries(record.steps).map(([name, step]) => [name, step.artifact_id])) };
    const history = await command(`/control-history ${JSON.stringify({ workspace: demoWorkspace })}`);
    assert.equal(history.attempts.find(attempt => attempt.attemptId === demo.attemptId).state, 'completed');
    const resumed = await command(`/control ${JSON.stringify({ ...workflowRequest, config: { resume_run: record.run_id } })}`);
    const resumeDeadline = Date.now() + 60000;
    let resumedStatus;
    while (Date.now() < resumeDeadline) {
      resumedStatus = await command(`/control-status ${resumed.jobId}`);
      if (['completed', 'failed', 'killed'].includes(resumedStatus.job.status)) break;
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    assert.equal(resumedStatus.job.status, 'completed', JSON.stringify(resumedStatus));
    const unchanged = JSON.parse(await readFile(resolve(demoWorkspace, 'manifest.json'), 'utf8'));
    assert.equal(Object.keys(unchanged.artifacts).length, 8, 'Resume must reuse, not duplicate completed artifacts');
    demoResult.completedStageResumeVerified = true;
  } else {
    demoResult = JSON.parse(await readFile(resolve(demoWorkspace, 'demo-result.json'), 'utf8'));
  }
  assert.equal(Object.keys(demoResult.artifacts).length, 8);
  const demoManifest = JSON.parse(await readFile(resolve(demoWorkspace, 'manifest.json'), 'utf8'));
  for (const artifactId of Object.values(demoResult.artifacts)) assert.equal(demoManifest.artifacts[artifactId].status, 'verified');
  const demoEvidence = { checkedAt: new Date().toISOString(), sessionId, jobId: demo.jobId,
    workspace: demoWorkspace, status: demoStatus.job.status, ...demoResult };
  await writeFile(resolve(project, 'test-output', `harness-${workflowMode ? 'workflow-' : ''}${demoCase}-evidence.json`), JSON.stringify(demoEvidence, null, 2));
  console.log(JSON.stringify(demoEvidence, null, 2));
}
