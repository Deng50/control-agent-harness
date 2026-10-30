import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile, symlink, access } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { spawn } from 'node:child_process';
import { JobHistory, authorizeWorkspace, recoveryRequest, persistedJob } from '../src/job-history.mjs';
import { createHost, project } from './history-host.mjs';

const root = resolve(project, 'test-output/job-history-tests');
await mkdir(root, { recursive: true });
const emptyJobs = { list: () => [] };
const workspace = () => mkdtemp(join(root, 'case-'));
const unpack = value => { assert.equal(value.kind, 'success', value.text); return JSON.parse(value.text); };

test('current sandbox policy and physical path bound host history writes', async () => {
  const allowed = await workspace(), outside = await workspace();
  await assert.rejects(authorizeWorkspace(allowed, { mode: 'read-only', workspaceRoot: allowed }, true), /does not allow/);
  await assert.rejects(authorizeWorkspace(outside, { mode: 'workspace-write', workspaceRoot: allowed }, true), /outside/);
  const redirected = join(allowed, 'redirect');
  await symlink(outside, redirected, process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(authorizeWorkspace(redirected, { mode: 'workspace-write', workspaceRoot: allowed }, true), /outside/);
  assert.equal(await authorizeWorkspace(join(allowed, 'new'), { mode: 'workspace-write', workspaceRoot: allowed }, true), join(allowed, 'new'));
});

test('concurrent events are committed atomically and a foreign Session cannot read them', async () => {
  const ws = await workspace(), store = new JobHistory();
  const writer = await store.create({ workspace: ws, owner: 'owner', agentName: 'model', request: { operation: 'inspect_workspace', workspace: ws } });
  await Promise.all(Array.from({ length: 40 }, (_, epoch) => writer.append({ workerEvent: { event: 'progress', epoch } })));
  await writer.append({ state: 'completed', outcome: { status: 'completed' } });
  const record = await new JobHistory().read(ws, writer.record.attemptId, 'owner', emptyJobs);
  assert.equal(record.progress.epoch, 39);
  assert.equal(record.state, 'completed');
  assert.deepEqual((await store.list(ws, 'foreign', emptyJobs)), []);
  await assert.rejects(store.read(ws, writer.record.attemptId, 'foreign', emptyJobs), /another Session/);
  assert.equal((await readFile(join(ws, '.control-jobs', writer.record.attemptId, 'request.json'), 'utf8')).includes('inspect_workspace'), true);
});

test('native state wins and missing ownership of an existing host is never assumed dead', async () => {
  const ws = await workspace(), store = new JobHistory();
  const writer = await store.create({ workspace: ws, owner: 'owner', agentName: 'model', request: { operation: 'inspect_workspace', workspace: ws } });
  await writer.append({ state: 'running', nativeJobId: 'native-id' });
  const record = await store.read(ws, writer.record.attemptId, 'owner', { list: () => [{ id: 'native-id', status: 'stopping', owner: 'owner' }] });
  assert.equal(record.state, 'stopping');
  const interrupted = await new JobHistory().read(ws, writer.record.attemptId, 'owner', emptyJobs);
  assert.equal(interrupted.state, 'interrupted');
  assert.ok(interrupted.recoveryBlocked);
  await assert.rejects(recoveryRequest(interrupted, 'retry'), /host PID/);
  const forced = await store.read(ws, writer.record.attemptId, 'owner', { list: () => [{ id: 'native-id', status: 'failed', owner: 'owner' }] });
  await assert.rejects(recoveryRequest(forced, 'retry'), /cleanup has not been verified/);
  await writer.append({ state: 'failed', outcome: { status: 'failed', cleanupVerified: false } });
  await assert.rejects(recoveryRequest(await store.read(ws, writer.record.attemptId, 'owner', emptyJobs), 'retry'), /cleanup has not been verified/);
});

test('recovery is first-wins across independent history writers and checkpoint scope is enforced', async () => {
  const ws = await workspace(), store = new JobHistory();
  const request = { operation: 'train_residual', workspace: ws, config: { epochs: 5 }, inputs: { dataset: 'explicit' } };
  const writer = await store.create({ workspace: ws, owner: 'owner', agentName: 'model', request });
  await writer.append({ state: 'failed', outcome: { status: 'failed' } });
  const record = await store.read(ws, writer.record.attemptId, 'owner', emptyJobs);
  const claims = await Promise.all(Array.from({ length: 10 }, (_, i) => new JobHistory().claimRecovery(ws, writer.record.attemptId, `replacement-${i}`, 'retry')));
  assert.equal(new Set(claims).size, 1);
  assert.deepEqual(await recoveryRequest(record, 'retry'), request);
  const other = await workspace(); await writeFile(join(other, 'latest.pt'), 'test');
  await assert.rejects(recoveryRequest(record, 'resume', join(other, 'latest.pt')), /inside this workspace/);
  await writeFile(join(ws, 'latest.pt'), 'checkpoint');
  const resumed = await recoveryRequest(record, 'resume', 'latest.pt');
  assert.equal(resumed.config.resume_checkpoint, join(ws, 'latest.pt'));
  assert.equal(record.request.config.resume_checkpoint, undefined);
  assert.equal((await recoveryRequest({ ...record, workflowRun: 'run-123', request: { ...request, operation: 'run_workflow' } }, 'resume')).config.resume_run, 'run-123');
});

test('cancel before persistence completes never starts a worker', async () => {
  const ws = await workspace(), store = new JobHistory();
  const writer = await store.create({ workspace: ws, owner: 'owner', agentName: 'model', request: { operation: 'inspect_workspace', workspace: ws } });
  let started = false;
  const hooks = persistedJob({ writer, job: { id: 'cancel-before-start' }, start: () => { started = true; } });
  hooks.cancel();
  assert.equal((await hooks.done).status, 'killed');
  assert.equal(started, false);
  assert.equal((await store.read(ws, writer.record.attemptId, 'owner', emptyJobs)).state, 'killed');
});

test('real native job and Python result survive a newly composed host with the same Session', { timeout: 45000 }, async () => {
  const ws = await workspace();
  const host = await createHost(ws);
  let started;
  try {
    started = unpack(await host.command('control', { agent: 'model', operation: 'inspect_workspace', workspace: ws }));
    const native = await host.ctx.jobs.wait(started.jobId, 30000, host.agent.id);
    assert.equal(native.status, 'completed', native.detail);
    assert.ok(unpack(await host.command('control-status', { workspace: ws, attemptId: started.attemptId })).record.result.project_id);
  } finally { await host.close(); }
  const next = await createHost(ws);
  try {
    assert.equal(next.ctx.jobs.list(next.agent.id).length, 0);
    const record = unpack(await next.command('control-status', { workspace: ws, attemptId: started.attemptId })).record;
    assert.equal(record.state, 'completed');
    assert.equal(record.authority, 'durable_outcome');
    assert.ok(record.result.project_id);
    assert.equal(unpack(await next.command('control-history', { workspace: ws })).attempts.length, 1);
  } finally { await next.close(); }
});

test('killed native attempt can be explicitly retried once despite concurrent commands', { timeout: 45000 }, async () => {
  const ws = await workspace(), host = await createHost(ws);
  try {
    const initial = unpack(await host.command('control', { agent: 'model', operation: 'inspect_workspace', workspace: ws }));
    host.ctx.jobs.kill(initial.jobId, host.agent.id);
    await host.ctx.jobs.wait(initial.jobId, 30000, host.agent.id);
    const args = { workspace: ws, attemptId: initial.attemptId, mode: 'retry' };
    const replies = (await Promise.all(Array.from({ length: 5 }, () => host.command('control-recover', args)))).map(unpack);
    assert.equal(new Set(replies.map(reply => reply.attemptId)).size, 1);
    const winner = replies.find(reply => reply.jobId);
    assert.ok(winner);
    assert.equal((await host.ctx.jobs.wait(winner.jobId, 30000, host.agent.id)).status, 'completed');
    assert.equal(host.ctx.jobs.list(host.agent.id).length, 2);
  } finally { await host.close(); }
});

test('real checkpoint cancellation and resume use a new native job with restored training state', { timeout: 90000 }, async () => {
  const ws = await workspace(), host = await createHost(ws);
  try {
    async function run(operation, inputs = {}, config = {}) {
      const started = unpack(await host.command('control', { agent: 'model', operation, workspace: ws, inputs, config }));
      const ended = await host.ctx.jobs.wait(started.jobId, 30000, host.agent.id);
      assert.equal(ended.status, 'completed', ended.detail);
      return unpack(await host.command('control-status', { workspace: ws, attemptId: started.attemptId })).record.result;
    }
    const nominal = await run('register_model', { model_path: resolve(project, 'examples/linear/nominal.json') },
      { source: { kind: 'user_nominal', description: 'Explicit resume test equations' } });
    const reference = await run('register_model', { model_path: resolve(project, 'examples/linear/reference.json') },
      { source: { kind: 'reference_simulator', description: 'Explicit resume test reference' } });
    const data = await run('simulate_transitions', { nominal_model: nominal.artifact_id, reference_model: reference.artifact_id },
      { trajectories: 5, steps: 12, seed: 53 });
    let cancelled = false;
    const stopWatching = host.ctx.jobs.events.subscribe({ owner: host.agent.id }, event => {
      if (event.type === 'progress' && event.job.progress?.startsWith('residual_training:') && !cancelled) {
        cancelled = true; host.ctx.jobs.kill(event.job.id, host.agent.id);
      }
    });
    const training = unpack(await host.command('control', { agent: 'model', operation: 'train_residual', workspace: ws,
      inputs: { nominal_model: nominal.artifact_id, dataset: data.artifact_id },
      config: { epochs: 12, batch_size: 16, hidden_dims: [8], seed: 54, minimum_improvement: 0 } }));
    const stopped = await host.ctx.jobs.wait(training.jobId, 30000, host.agent.id);
    stopWatching();
    assert.equal(stopped.status, 'killed', stopped.detail);
    const before = unpack(await host.command('control-status', { workspace: ws, attemptId: training.attemptId })).record;
    assert.ok(before.checkpoint);
    await access(before.checkpoint);
    const resumed = unpack(await host.command('control-recover', { workspace: ws, attemptId: training.attemptId, mode: 'resume' }));
    assert.notEqual(resumed.jobId, training.jobId);
    const outcome = await host.ctx.jobs.wait(resumed.jobId, 30000, host.agent.id);
    const after = unpack(await host.command('control-status', { workspace: ws, attemptId: resumed.attemptId })).record;
    assert.ok(after.result?.artifact_path, outcome.detail);
    const report = JSON.parse(await readFile(join(after.result.artifact_path, 'report.json'), 'utf8'));
    assert.ok(report.resumed_from_epoch >= 1);
    assert.ok(report.optimizer_updates > before.progress.optimizer_updates);
    assert.equal(report.trained_epochs, 12);
  } finally { await host.close(); }
});

test('hard host termination leaves an interrupted record; fresh host exposes an explicit recovery path', { timeout: 90000 }, async () => {
  const ws = await workspace();
  const child = spawn(process.execPath, [resolve(import.meta.dirname, 'history-host.mjs'), '--crash-fixture', ws],
    { cwd: project, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '', pending = '', initial;
  child.stderr.on('data', chunk => { stderr += chunk; });
  const exited = new Promise(resolveExit => child.once('exit', resolveExit));
  const ready = new Promise((resolveReady, reject) => {
    child.once('error', reject);
    child.once('exit', code => reject(new Error(`Host exited before progress: ${code}: ${stderr}`)));
    child.stdout.on('data', chunk => {
      pending += chunk;
      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline); pending = pending.slice(newline + 1);
        let value; try { value = JSON.parse(line); } catch { continue; }
        if (value.type === 'started') initial = value;
        if (value.type === 'ready_to_crash') resolveReady(value);
      }
    });
  });
  const timer = setTimeout(() => child.kill('SIGKILL'), 60000);
  try {
    await ready;
    child.kill('SIGKILL');
    await exited;
    const host = await createHost(ws);
    try {
      const record = unpack(await host.command('control-status', { workspace: ws, attemptId: initial.attemptId })).record;
      assert.equal(record.state, 'interrupted');
      assert.equal(record.recoveryBlocked, undefined);
      assert.equal(host.ctx.jobs.list(host.agent.id).length, 0);
      const request = await recoveryRequest(record, 'retry');
      assert.deepEqual(request, record.request);
      // Do not relaunch the entire demo as a fake training continuation. A changed request is a separate new job.
      await assert.rejects(recoveryRequest(record, 'resume'), /no checkpoint resume support/);
      const probe = unpack(await host.command('control', { agent: 'model', operation: 'register_model', workspace: ws,
        inputs: { model_path: resolve(project, 'examples/linear/nominal.json') },
        config: { source: { kind: 'user_nominal', description: 'Lock recovery probe after an explicitly terminated host' } } }));
      const probeResult = await host.ctx.jobs.wait(probe.jobId, 30000, host.agent.id);
      assert.equal(probeResult.status, 'completed', probeResult.detail);
      await assert.rejects(access(join(ws, '.operation.lock')));
    } finally { await host.close(); }
  } finally { clearTimeout(timer); if (child.exitCode === null) child.kill('SIGKILL'); await exited; }
});
