/** S05 N03 real native cancellation/recovery; requires an existing isolated dataset. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { access, readFile, readdir } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command, connectHarness, project, writeSmokeEvidence } from './harness-rpc.mjs';

const { values } = parseArgs({ options: { workspace: { type: 'string' }, dataset: { type: 'string' },
  evidence: { type: 'string' }, 'web-log': { type: 'string' }, resume: { type: 'boolean' }, help: { type: 'boolean', short: 'h' } } });
if (values.help) {
  console.log('node scripts/smoke-supervised-recovery.mjs --workspace <existing isolated workspace> --dataset <supervised_dataset id> --evidence <json> [--web-log <log>] [--resume]');
  console.log('No Host is started and no table is rebuilt. Native job/kill targets only this script\'s saved job; observer resume never resubmits uncertain admission.');
  process.exit(0);
}
assert.ok(values.workspace && values.dataset && values.evidence, 'Explicit workspace, dataset and evidence are required.');
const workspace = resolve(values.workspace), datasetId = values.dataset, evidencePath = resolve(values.evidence);
const descendant = relative(resolve(project, 'test-output'), workspace);
assert.ok(descendant && !isAbsolute(descendant) && descendant !== '..' && !descendant.startsWith(`..${sep}`), 'Native test confirmations require an isolated project test-output workspace.');
const json = path => readFile(path, 'utf8').then(JSON.parse);
const sha = value => createHash('sha256').update(value).digest('hex');
const marker = await json(resolve(workspace, '.case-plans/test-fixture.json'));
assert.equal(marker.kind, 'control-case-plan-test-fixture-v1'); assert.equal(marker.workspace, workspace);
const manifest = await json(resolve(workspace, 'manifest.json'));
assert.equal(manifest.artifacts[datasetId]?.kind, 'supervised_dataset');
assert.equal(manifest.artifacts[datasetId]?.status, 'verified');
const datasetRoot = resolve(workspace, manifest.artifacts[datasetId].path);
const relativeDataset = relative(workspace, datasetRoot);
assert.ok(relativeDataset && !isAbsolute(relativeDataset) && relativeDataset !== '..' && !relativeDataset.startsWith(`..${sep}`));
const datasetSplit = await json(resolve(datasetRoot, 'split.json'));
assert.deepEqual(datasetSplit.sample_counts, { train: 640, validation: 192, test: 192 }, 'Reuse the main frozen F03 dataset, not another row partition.');
const datasetDescription = await json(resolve(datasetRoot, 'dataset.json'));
assert.equal(datasetDescription.input_dim, 3); assert.equal(datasetDescription.output_dim, 2);
assert.deepEqual(datasetDescription.feature_order.map(item => item.signal), ['f1', 'f2', 'f3']);
// Lifecycle candidate fixed before its first native run. Patience exceeds the
// total target so cancellation/recovery cannot be masked by early stopping.
const trainingConfig = { hidden_dims: [24, 16], activation: 'tanh', input_dim: 3, output_dim: 2,
  epochs: 300, learning_rate: .005, batch_size: 16, patience: 1000, seed: 20261120,
  weight_decay: 0, min_delta: 0, grad_clip: 5, max_test_nrmse: .15 };
let evidence = values.resume ? await json(evidencePath) : { schema_version: 1, status: 'RUNNING', workspace, datasetId,
  startedAt: new Date().toISOString(), confirmation_source: 'test_fixture', preRegisteredConfig: trainingConfig,
  scope: 'S05 N03 native job cancel -> same-summary checkpoint resume -> new native job; synthetic existing F03 dataset, no LLM or onward MPC.',
  jobs: {}, admissions: [], confirmations: [], rejections: [] };
assert.equal(evidence.workspace, workspace); assert.equal(evidence.datasetId, datasetId);
assert.deepEqual(evidence.preRegisteredConfig, trainingConfig, 'Observer resume cannot change the lifecycle candidate.');
const save = () => writeSmokeEvidence(evidencePath, evidence);
await save();
const python = process.env.CONTROL_PYTHON ?? resolve(project, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const runPython = (code, args, env = {}) => execFileSync(python, ['-B', '-c', code, ...args], { encoding: 'utf8', windowsHide: true,
  maxBuffer: 4 * 1024 * 1024, env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', ...env } });
const checkpointInfo = String.raw`import hashlib,json,sys,torch
from pathlib import Path
p=Path(sys.argv[1]);c=torch.load(p,map_location='cpu',weights_only=True)
def digest(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items(),key=lambda item:str(item[0])):
        h.update(str(key).encode())
        if isinstance(value,dict): h.update(digest(value).encode())
        elif isinstance(value,torch.Tensor): h.update(str(value.dtype).encode());h.update(str(list(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else: h.update(repr(value).encode())
    return h.hexdigest()
sidecar=json.loads(p.with_name('latest.checkpoint.json').read_text(encoding='utf-8'))
actual=hashlib.sha256(p.read_bytes()).hexdigest();assert sidecar['sha256']==actual
print(json.dumps({'sha256':actual,'epoch':c['epoch'],'updates':c['updates'],'training_signature':c['training_signature'],
    'model_sha256':digest(c['model_state']),'optimizer_sha256':digest(c['optimizer_state']),
    'optimizer_steps':[float(state['step']) for state in c['optimizer_state']['state'].values()],
    'optimizer_learning_rates':[group['lr'] for group in c['optimizer_state']['param_groups']],
    'weight_shapes':{key:list(value.shape) for key,value in c['model_state'].items()},'log_epochs':[row['epoch'] for row in c['log']]},allow_nan=False))`;
const internalProbe = String.raw`import json,sys,numpy as np,torch
from pathlib import Path
from control_harness.supervised_export import build_supervised_network
torch.set_num_threads(1)
root,dataset,output=map(Path,sys.argv[1:4])
network=build_supervised_network(json.loads((root/'network_config.json').read_text(encoding='utf-8')))
network.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True)['network_state'],strict=True);network.eval()
scaler=json.loads((root/'scaler.json').read_text(encoding='utf-8'))
with np.load(dataset/'supervised.npz',allow_pickle=False) as a: raw=a['X'][:11].copy()
output.write_text(json.dumps({'features':raw.tolist(),'feature_labels':['f1@lag=0','f2@lag=0','f3@lag=0']})+'\n',encoding='utf-8')
with torch.no_grad(): y=network(torch.from_numpy((raw-scaler['X']['location'])/scaler['X']['scale'])).numpy()*scaler['y']['scale']+scaler['y']['location']
print(json.dumps({'predictions':y.tolist()},allow_nan=False))`;
const isolatedBundle = String.raw`import importlib.abc,runpy,sys
class BlockProject(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='control_harness' or fullname.startswith('control_harness.'): raise ModuleNotFoundError('No project import in independent bundle')
sys.meta_path.insert(0,BlockProject());script=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')
assert not any(key=='control_harness' or key.startswith('control_harness.') for key in sys.modules)`;
const terminal = new Set(['completed', 'failed', 'killed']);
const pause = () => new Promise(done => setTimeout(done, 100));
const near = (a, b) => {
  if (Array.isArray(b)) { assert.equal(a.length, b.length); b.forEach((value, k) => near(a[k], value)); }
  else assert.ok(Number.isFinite(a) && Math.abs(a - b) <= 1e-6 + 1e-5 * Math.abs(b));
};

try {
  const rpc = await connectHarness({ logPath: values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG });
  for (const key of ['sessionId', 'peerSessionId']) {
    if (!evidence[key]) {
      const session = await rpc('session/create', { request: { cwd: workspace, agentPreset: 'control-model' } });
      assert.equal(session.agentPreset, 'control-model'); evidence[key] = session.sessionId; await save();
    }
  }
  const sessionId = evidence.sessionId;
  const status = entry => command(rpc, sessionId, `/control-status ${JSON.stringify({ workspace, attemptId: entry.attemptId })}`);
  const admit = async (key, line) => {
    let entry = evidence.jobs[key]; const requestSha = sha(line);
    if (entry) {
      assert.equal(entry.requestSha, requestSha, 'Observer resume changed an admitted request');
      assert.ok(entry.jobId, 'Uncertain admission: reconcile native history and attach its original job ID before observing; never resubmit.');
      return entry;
    }
    entry = evidence.jobs[key] = { requestSha, status: 'ADMISSION_PENDING' }; await save();
    const started = await command(rpc, sessionId, line);
    Object.assign(entry, { jobId: started.jobId, attemptId: started.attemptId, recoveredFrom: started.recoveredFrom ?? null, status: 'RUNNING' });
    assert.ok(entry.jobId && entry.attemptId);
    evidence.admissions.push({ key, jobId: entry.jobId, attemptId: entry.attemptId, recoveredFrom: entry.recoveredFrom });
    await save(); return entry;
  };
  const wait = async (entry, expected = 'completed') => {
    for (;;) {
      const state = await status(entry);
      if (terminal.has(state.record.state)) {
        entry.status = state.record.state; await save();
        assert.equal(state.record.state, expected, JSON.stringify(state)); return state.record;
      }
      assert.notEqual(state.record.state, 'interrupted', 'Host ownership was lost; stop and reconcile the saved job rather than resubmit.');
      await pause();
    }
  };
  const support = async (key, operation, config) => {
    const entry = await admit(key, `/control ${JSON.stringify({ agent: 'model', operation, workspace, inputs: {}, config })}`);
    return (await wait(entry)).result;
  };
  if (!evidence.planRefs) {
    const request = { operation: 'train_supervised', workspace, inputs: { dataset: datasetId }, config: trainingConfig };
    const requirements = await confirmationRequirements(request);
    assert.deepEqual(requirements.map(item => item.scope), ['signals', 'training']);
    const refs = {};
    for (const row of requirements) {
      const plan = await support(`plan:${row.scope}`, 'propose_case_plan', { scope: row.scope, payload: row.expectedPayload, bindings: row.bindings });
      refs[row.scope] = { plan_id: plan.plan_id, version_hash: plan.version_hash };
      const confirmed = await command(rpc, sessionId, `/control-confirm ${JSON.stringify({ workspace, ...refs[row.scope] })}`);
      assert.equal(confirmed.confirmation_source, 'test_fixture');
      evidence.confirmations.push({ scope: row.scope, ...refs[row.scope], confirmation_source: 'test_fixture' }); await save();
    }
    evidence.planRefs = refs; await save();
  }
  const request = { agent: 'model', operation: 'train_supervised', workspace, inputs: { dataset: datasetId },
    config: { ...trainingConfig, plan_refs: evidence.planRefs } };
  const original = await admit('training', `/control ${JSON.stringify(request)}`);
  if (!evidence.cancelRequested) {
    for (;;) {
      const state = await status(original);
      assert.ok(!terminal.has(state.record.state), 'Training finished before a real >=2-epoch cancellation; do not count this as N03 or rerun automatically.');
      assert.notEqual(state.record.state, 'interrupted', 'Execution ownership lost before cancellation');
      if ((state.record.progress?.epoch ?? 0) >= 2) {
        evidence.observedBeforeCancel = { progress: state.record.progress, nativeStatus: state.record.state, observedAt: new Date().toISOString() };
        evidence.cancelRequested = { status: 'ADMISSION_PENDING', jobId: original.jobId, method: 'job/kill' }; await save();
        // Actual native JobController API; no OS process enumeration or kill.
        const receipt = await rpc('job/kill', { request: { sessionId, jobId: original.jobId } });
        assert.equal(receipt.outcome, 'requested'); evidence.cancelRequested.receipt = receipt; evidence.cancelRequested.status = 'requested'; await save();
        break;
      }
      await pause();
    }
  }
  const killed = await wait(original, 'killed');
  assert.equal(killed.outcome.cleanupVerified, true); assert.ok(!killed.recoveryBlocked);
  try { await access(resolve(workspace, '.operation.lock')); assert.fail('Workspace lock remains after verified native cancellation'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  assert.ok(killed.checkpoint);
  const checkpoint = resolve(killed.checkpoint), checkpointRelative = relative(workspace, checkpoint);
  assert.ok(checkpointRelative.startsWith(`failed${sep}`) && checkpoint.endsWith(`${sep}latest.pt`));
  const before = JSON.parse(runPython(checkpointInfo, [checkpoint]));
  assert.ok(before.epoch >= 2 && before.epoch < trainingConfig.epochs);
  assert.ok(before.optimizer_steps.every(step => step === before.updates));
  assert.deepEqual(before.optimizer_learning_rates, [trainingConfig.learning_rate]);
  evidence.killed = { jobId: original.jobId, attemptId: original.attemptId, checkpoint, cleanupVerified: true,
    lockReleased: true, progress: killed.progress, checkpointBefore: before }; await save();
  const attemptDirectory = resolve(workspace, '.control-jobs', original.attemptId);
  const originalRequestHash = sha(await readFile(resolve(attemptDirectory, 'request.json')));
  const snapshot = async () => {
    const attempts = (await readdir(resolve(workspace, '.control-jobs'))).filter(name => /^[a-f0-9-]{36}$/.test(name)).sort();
    const sessions = [];
    for (const id of [sessionId, evidence.peerSessionId]) {
      const history = await command(rpc, id, `/control-history ${JSON.stringify({ workspace })}`);
      const records = (await rpc('session/events', { request: { address: { kind: 'session', sessionId: id }, maxMessages: 10000 } })).records;
      sessions.push({ sessionId: id, attempts: history.attempts.map(row => row.attemptId).sort(),
        registeredJobs: records.filter(row => row.event.type === 'job/registered').map(row => row.event.data.jobId ?? row.event.data.job?.id ?? row.seq) });
    }
    return { attempts, sessions };
  };
  const reject = async (key, id, line) => {
    if (evidence.rejections.some(row => row.key === key)) return;
    const old = await snapshot();
    const response = await rpc('commands/execute', { agentId: id, line, submittedAttachments: [] });
    assert.equal(response.result.kind, 'error', `${key} unexpectedly admitted a job`);
    assert.deepEqual(await snapshot(), old, `${key} created a native job or history attempt`);
    assert.equal(sha(await readFile(resolve(attemptDirectory, 'request.json'))), originalRequestHash, 'Original append-only request was modified');
    evidence.rejections.push({ key, nativeKind: response.result.kind, detail: response.result.text, newAttempts: 0, newJobs: 0 }); await save();
  };
  for (const [key, changed] of [['changed-layers', { hidden_dims: [25, 16] }], ['changed-learning-rate', { learning_rate: .004 }]]) {
    await reject(key, sessionId, `/control ${JSON.stringify({ ...request, config: { ...request.config, ...changed, resume_checkpoint: checkpoint } })}`);
  }
  await reject('foreign-session-recovery', evidence.peerSessionId, `/control-recover ${JSON.stringify({ workspace, attemptId: original.attemptId, mode: 'resume' })}`);
  if (!evidence.jobs.recovery) {
    try { await access(resolve(attemptDirectory, 'recovery.json')); assert.fail('Rejected recovery claimed the original attempt'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  const confirmationCount = evidence.confirmations.length;
  const recovered = await admit('recovery', `/control-recover ${JSON.stringify({ workspace, attemptId: original.attemptId, mode: 'resume' })}`);
  assert.notEqual(recovered.jobId, original.jobId); assert.notEqual(recovered.attemptId, original.attemptId);
  assert.equal(recovered.recoveredFrom, original.attemptId);
  const completed = await wait(recovered);
  assert.equal(completed.recoveredFrom, original.attemptId); assert.equal(completed.recoveryMode, 'resume');
  assert.deepEqual(completed.request.config.plan_refs, evidence.planRefs);
  assert.equal(completed.request.config.epochs, trainingConfig.epochs);
  assert.equal(resolve(completed.request.config.resume_checkpoint), checkpoint);
  assert.equal(evidence.confirmations.length, confirmationCount, 'Compatible recovery demanded a new engineering-summary confirmation');
  const originalAfter = (await status(original)).record;
  assert.equal(originalAfter.recovery.attemptId, recovered.attemptId);
  assert.equal(originalAfter.recovery.mode, 'resume');
  const result = completed.result;
  assert.equal(result.kind, 'supervised_model'); assert.equal(result.status, 'verified');
  const report = await json(resolve(result.artifact_path, 'report.json'));
  const after = JSON.parse(runPython(checkpointInfo, [resolve(result.artifact_path, 'latest.pt')]));
  assert.equal(report.resumed_from_epoch, before.epoch);
  assert.equal(after.epoch, trainingConfig.epochs); assert.equal(report.trained_epochs, trainingConfig.epochs);
  assert.equal(after.training_signature, before.training_signature);
  assert.ok(after.updates > before.updates); assert.equal(after.updates, Math.ceil(640 / trainingConfig.batch_size) * after.epoch);
  assert.ok(after.optimizer_steps.every(step => step === after.updates), 'Optimizer steps were reset rather than restored');
  assert.deepEqual(after.optimizer_learning_rates, before.optimizer_learning_rates);
  assert.deepEqual(after.weight_shapes, before.weight_shapes);
  assert.notEqual(after.model_sha256, before.model_sha256); assert.notEqual(after.optimizer_sha256, before.optimizer_sha256);
  assert.deepEqual(after.log_epochs, Array.from({ length: trainingConfig.epochs }, (_, k) => k + 1));
  assert.equal(sha(await readFile(checkpoint)), before.sha256, 'Recovery mutated its original failed checkpoint');
  const input = resolve(workspace, 'fixtures/recovery-export-input.json'), output = resolve(workspace, 'fixtures/recovery-export-output.json');
  const internal = JSON.parse(runPython(internalProbe, [result.artifact_path, datasetRoot, input], { PYTHONPATH: resolve(project, 'src') }));
  const bundle = resolve(result.artifact_path, 'bundle');
  execFileSync(python, ['-B', '-c', isolatedBundle, resolve(bundle, 'predict.py'), '--input', input, '--output', output], {
    cwd: bundle, windowsHide: true, encoding: 'utf8', env: { ...process.env, PYTHONPATH: '', PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' },
  });
  near((await json(output)).predictions, internal.predictions);
  const metadata = await json(resolve(result.artifact_path, 'artifact.json'));
  for (const [name, expected] of Object.entries(metadata.files)) assert.equal(sha(await readFile(resolve(result.artifact_path, name))), expected);
  const beforeDuplicate = await snapshot();
  const duplicate = await command(rpc, sessionId, `/control-recover ${JSON.stringify({ workspace, attemptId: original.attemptId, mode: 'resume' })}`);
  assert.equal(duplicate.existing, true); assert.equal(duplicate.jobId, recovered.jobId); assert.equal(duplicate.attemptId, recovered.attemptId);
  assert.deepEqual(await snapshot(), beforeDuplicate, 'Duplicate recover created another native job or history attempt');
  assert.equal(sha(await readFile(resolve(attemptDirectory, 'request.json'))), originalRequestHash, 'Recovery rewrote its original append-only request');
  const finalManifest = await json(resolve(workspace, 'manifest.json'));
  assert.ok(Object.values(finalManifest.artifacts).every(row => ['material', 'supervised_dataset', 'supervised_model'].includes(row.kind)), 'Recovery continued into a physical/MPC/policy branch');
  const events = (await rpc('session/events', { request: { address: { kind: 'session', sessionId }, maxMessages: 10000 } })).records;
  evidence.eventTypes = [...new Set(events.map(row => row.event.type))];
  assert.ok(!evidence.eventTypes.some(type => type.startsWith('turn/')), 'Fixed recovery smoke opened a model turn');
  evidence.recovery = { oldJobId: original.jobId, oldAttemptId: original.attemptId, newJobId: recovered.jobId,
    newAttemptId: recovered.attemptId, recoveredFrom: completed.recoveredFrom, recoveryMode: completed.recoveryMode,
    sameSummaryPlanRefs: completed.request.config.plan_refs, newConfirmations: 0, checkpointBefore: before, checkpointAfter: after,
    report: { resumed_from_epoch: report.resumed_from_epoch, trained_epochs: report.trained_epochs, optimizer_updates: report.optimizer_updates },
    artifactId: result.artifact_id, exportSeparateProcess: true, exportProjectImportsBlocked: true, duplicateRecoveryReusedSameJob: true };
  evidence.status = 'PASS'; evidence.finishedAt = new Date().toISOString();
} catch (error) {
  evidence.status = 'FAIL'; evidence.error = error instanceof Error ? error.stack : String(error);
  evidence.requiresObserveOrReconcile = Object.values(evidence.jobs).some(row => !terminal.has(row.status));
  throw error;
} finally {
  await save();
  console.log(JSON.stringify({ status: evidence.status, workspace, datasetId, evidence: evidencePath,
    jobs: Object.fromEntries(Object.entries(evidence.jobs).map(([key, row]) => [key, { jobId: row.jobId, attemptId: row.attemptId, status: row.status }])),
    confirmation_source: 'test_fixture' }));
}
