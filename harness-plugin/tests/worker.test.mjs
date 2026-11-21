import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, access } from 'node:fs/promises';
import { resolve } from 'node:path';
import { WorkerSlots, numericalJob } from '../src/worker.mjs';

const project = resolve(import.meta.dirname, '../..');
const temp = resolve(project, 'test-output/worker-tests');
await mkdir(temp, { recursive: true });

// This adapter exercises the actual Python protocol. Full Cordis composition is a separate acceptance gate.
const ctx = { subprocess: { spawn(spec) {
  const child = spawn(spec.argv[0], spec.argv.slice(1), {
    cwd: spec.cwd, env: { ...process.env, ...spec.env }, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-65536); });
  const done = new Promise((resolveDone, reject) => {
    child.on('error', reject);
    child.on('close', (code, signal) => resolveDone({ exitCode: code, signal }));
  });
  return { stdin: child.stdin, stdout: child.stdout, done,
    collected: { stderr: { readFrom: () => ({ text: stderr }) } },
    terminate: () => child.kill(), waitForExit: async () => { await done; return true; } };
} } };

const config = { codeRoot: project, pythonPath: resolve(project, '.venv/Scripts/python.exe'), timeoutSeconds: 120,
                 cancellationGraceMs: 10000 };

test('global worker admission is bounded, FIFO and queued cancellation frees its entry', async () => {
  const slots = new WorkerSlots(1);
  const release = await slots.acquire(new AbortController().signal);
  const cancelled = new AbortController();
  const queued = slots.acquire(cancelled.signal);
  const assertion = assert.rejects(queued);
  cancelled.abort(new Error('no longer needed'));
  await assertion;
  assert.equal(slots.waiting.length, 0);
  release();
  assert.equal(slots.active, 0);
});

test('worker queue has a separate deadline and times out before dispatch', async () => {
  const slots = new WorkerSlots(1), release = await slots.acquire(new AbortController().signal);
  const guard = setTimeout(release,5000);
  let dispatched = false;
  try {
    const hooks = numericalJob({ctx:{subprocess:{spawn(){dispatched=true;}}},slots,
      config:{...config,queueTimeoutSeconds:1},policy:{mode:'workspace-write'},
      job:{updateProgress(){},append(){}},request:{operation:'inspect_workspace'}});
    const outcome = await hooks.done;
    assert.equal(outcome.status,'failed');
    assert.match(outcome.detail,/WORKER_QUEUE_TIMEOUT/);
    assert.equal(outcome.cleanupVerified,true);
    assert.equal(dispatched,false);
    assert.equal(slots.waiting.length,0);
  } finally {clearTimeout(guard);release();}
});

test('real worker returns an actual project manifest', async () => {
  const workspace = await mkdtemp(resolve(temp, 'inspect-'));
  const outputs = [];
  const hooks = numericalJob({ ctx, slots: new WorkerSlots(1), config,
    policy: { mode: 'danger-full-access' },
    job: { updateProgress() {}, append: value => outputs.push(value) },
    request: { operation: 'inspect_workspace', workspace },
  });
  const outcome = await hooks.done;
  assert.equal(outcome.status, 'completed', outcome.detail);
  const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
  assert.equal(JSON.parse(outcome.result).project_id, manifest.project_id);
  assert.ok(outputs.some(line => line.includes('"event": "started"')));
});

test('worker validation errors are failed native-job outcomes', async () => {
  const workspace = await mkdtemp(resolve(temp, 'invalid-'));
  const hooks = numericalJob({ ctx, slots: new WorkerSlots(1), config,
    policy: { mode: 'danger-full-access' }, job: { updateProgress() {}, append() {} },
    request: { operation: 'not_a_real_operation', workspace },
  });
  const result = await hooks.done;
  assert.equal(result.status, 'failed');
  assert.match(result.detail, /Unknown operation/);
});

test('a genuinely trained but rejected residual is a failed job and never becomes active', async () => {
  const workspace = await mkdtemp(resolve(temp, 'rejected-'));
  const slots = new WorkerSlots(1);
  async function run(operation, inputs = {}, stageConfig = {}) {
    return numericalJob({ ctx, slots, config, policy: { mode: 'danger-full-access' },
      job: { updateProgress() {}, append() {} }, request: { operation, workspace, inputs, config: stageConfig },
    }).done;
  }
  const nominal = await run('register_model', { model_path: resolve(project, 'examples/linear/nominal.json') },
    { source: { kind: 'user_nominal', description: 'Test nominal equations' } });
  assert.equal(nominal.status, 'completed', nominal.detail);
  const reference = await run('register_model', { model_path: resolve(project, 'examples/linear/reference.json') },
    { source: { kind: 'reference_simulator', description: 'Explicit test simulator' } });
  assert.equal(reference.status, 'completed', reference.detail);
  const nominalId = JSON.parse(nominal.result).artifact_id;
  const data = await run('simulate_transitions', { nominal_model: nominalId, reference_model: JSON.parse(reference.result).artifact_id },
    { trajectories: 5, steps: 12, seed: 202 });
  assert.equal(data.status, 'completed', data.detail);
  const trained = await run('train_residual', { nominal_model: nominalId, dataset: JSON.parse(data.result).artifact_id },
    { epochs: 1, hidden_dims: [4], seed: 203, minimum_improvement: 0.999999 });
  assert.equal(trained.status, 'failed');
  assert.match(trained.detail, /failed domain acceptance/);
  const result = JSON.parse(trained.result);
  assert.equal(result.status, 'rejected');
  const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
  assert.equal(manifest.active.corrected_model, undefined);
  const report = JSON.parse(await readFile(resolve(result.artifact_path, 'report.json'), 'utf8'));
  assert.equal(report.trained_epochs, 1);
  assert.equal(report.acceptance, 'failed');
  const inspection = await run('inspect_artifact', { artifact_id: result.artifact_id });
  assert.equal(inspection.status, 'completed', inspection.detail);
  assert.equal(JSON.parse(inspection.result).status, 'rejected');
  assert.equal(JSON.parse(inspection.result).report.acceptance, 'failed');
});

test('cooperative cancellation stops real training and releases the workspace lock', async () => {
  const workspace = await mkdtemp(resolve(temp, 'cancel-'));
  let hooks, requested = false;
  hooks = numericalJob({ ctx, slots: new WorkerSlots(1), config,
    policy: { mode: 'danger-full-access' },
    job: { updateProgress() {}, append(line) {
      if (!requested && line.includes('"stage": "residual_training"')) {
        requested = true;
        hooks.cancel();
      }
    } },
    request: { operation: 'demo', workspace, config: { case: 'linear' } },
  });
  const result = await hooks.done;
  assert.ok(requested);
  assert.equal(result.status, 'killed', result.detail);
  await assert.rejects(access(resolve(workspace, '.operation.lock')));
  await assert.rejects(access(resolve(workspace, 'demo-result.json')));
});
