/** Append-only execution history; native Harness jobs remain the live authority. */
import { randomUUID } from 'node:crypto';
import { hostname } from 'node:os';
import { mkdir, readFile, readdir, realpath, stat, writeFile, rename, link, unlink } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

const TERMINAL = new Set(['completed', 'failed', 'killed']);
const ATTEMPT = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const TRAINING = new Set(['train_residual', 'train_imitation', 'optimize_policy', 'train_supervised', 'train_predictive_model', 'train_external_imitation', 'train_pinn', 'optimize_external_policy']);

function inside(root, target) {
  const path = relative(root, target);
  return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith(`..${sep}`));
}

/** Resolve existing ancestors, including junctions, before checking session write roots. */
async function physical(path) {
  try { return await realpath(path); } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const parent = dirname(path);
    if (parent === path) throw error;
    return join(await physical(parent), relative(parent, path));
  }
}

export async function authorizeWorkspace(workspace, policy, write = false) {
  if (typeof workspace !== 'string' || !isAbsolute(workspace)) throw new Error('workspace must be an absolute path');
  const target = await physical(resolve(workspace));
  if (write && policy.mode !== 'danger-full-access') {
    if (policy.mode !== 'workspace-write') throw new Error('Current session sandbox does not allow persistent job writes');
    const roots = await Promise.all([policy.workspaceRoot, ...(policy.additionalWritableRoots ?? [])].filter(Boolean).map(physical));
    if (!roots.some(root => inside(root, target))) throw new Error('Workspace is outside current session writable roots');
  }
  // A ledger must never redirect host writes through an existing symlink/junction.
  const ledger = join(target, '.control-jobs');
  if (!inside(target, await physical(ledger))) throw new Error('Job history escapes its workspace');
  return target;
}

async function atomicJson(path, value, exclusive = false) {
  const temp = `${path}.${randomUUID()}.tmp`;
  await writeFile(temp, `${JSON.stringify(value)}\n`, { flag: 'wx', mode: 0o600 });
  try {
    if (exclusive) await link(temp, path);
    else await rename(temp, path);
  } finally {
    await unlink(temp).catch(error => { if (error.code !== 'ENOENT') throw error; });
  }
}

/** One history service per plugin instance, with no scheduling or process ownership. */
export class JobHistory {
  constructor() { this.runtimeId = randomUUID(); }

  async directory(workspace, attemptId) {
    if (!ATTEMPT.test(attemptId)) throw new Error('Invalid durable attempt id');
    const root = join(workspace, '.control-jobs');
    const path = join(root, attemptId);
    if (!inside(workspace, await physical(root)) || !inside(root, await physical(path))) throw new Error('Job history path escapes workspace');
    return path;
  }

  async create({ workspace, owner, agentName, request, recoveredFrom = null, recoveryMode = null }) {
    const attemptId = randomUUID();
    const directory = await this.directory(workspace, attemptId);
    await mkdir(directory, { recursive: true });
    const record = { version: 1, attemptId, owner, agentName, workspace, runtimeId: this.runtimeId, hostPid: process.pid, host: hostname(),
      createdAt: new Date().toISOString(), recoveredFrom, recoveryMode, request: structuredClone(request) };
    await atomicJson(join(directory, 'request.json'), record, true);
    return this.writer(record, directory);
  }

  writer(record, directory) {
    let sequence = 0, chain = Promise.resolve(), failure;
    return {
      record,
      append(event) {
        const entry = { ...structuredClone(event), sequence: ++sequence, recordedAt: new Date().toISOString() };
        chain = chain.then(() => atomicJson(join(directory, `${String(entry.sequence).padStart(12, '0')}.json`), entry, true));
        // A failed journal stops new execution; retain it for flush without unhandled rejections.
        chain.catch(error => { failure ??= error; });
        return chain;
      },
      async flush() { await chain; if (failure) throw failure; },
    };
  }

  async read(workspace, attemptId, owner, jobs) {
    const directory = await this.directory(workspace, attemptId);
    const record = JSON.parse(await readFile(join(directory, 'request.json'), 'utf8'));
    if (record.version !== 1 || record.attemptId !== attemptId || record.workspace !== workspace) throw new Error('Invalid job history record');
    if (record.owner !== owner) throw new Error('Job history belongs to another Session');
    let state = { ...record, state: 'prepared', nativeJobId: null, progress: null, result: null, checkpoint: null, workflowRun: null };
    const files = (await readdir(directory)).filter(name => /^\d{12}\.json$/.test(name)).sort();
    for (const file of files) {
      const event = JSON.parse(await readFile(join(directory, file), 'utf8'));
      if (event.state) state.state = event.state;
      if (event.nativeJobId) state.nativeJobId = event.nativeJobId;
      if (event.workerEvent) {
        const value = event.workerEvent;
        if (value.event === 'progress') state.progress = value;
        if (value.event === 'result') state.result = value.result;
        if (value.checkpoint) state.checkpoint = value.checkpoint;
        if (value.workflow_run) state.workflowRun = value.workflow_run;
        if (value.result?.workflow_run) state.workflowRun = value.result.workflow_run;
        if (value.result?.checkpoint) state.checkpoint = value.result.checkpoint;
      }
      if (event.outcome) { state.outcome = event.outcome; state.detail = event.outcome.detail; }
      if (event.nativeSettlement) { state.nativeSettlement = event.nativeSettlement; state.detail = event.nativeSettlement.detail ?? state.detail; }
      if (event.recoveryBlocked) state.recoveryBlocked = event.recoveryBlocked;
      state.updatedAt = event.recordedAt;
    }
    // list(owner) is the registry's access-checked view; do not mask get() authorization errors.
    const native = state.nativeJobId ? jobs.list(owner).find(item => item.id === state.nativeJobId) : undefined;
    if (native) {
      state.native = native;
      state.state = native.status;
      state.authority = 'native_job';
    } else if (!TERMINAL.has(state.state)) {
      state.persistedState = state.state;
      state.state = 'interrupted';
      state.detail = 'Native execution ownership is unavailable. Original process state is unknown; no automatic restart was attempted.';
      state.authority = 'ownership_lost';
      // PID is only a conservative recovery veto; it is never used to identify or kill a process.
      // A reused PID can delay manual recovery, but cannot admit duplicate execution.
      if (record.host !== hostname()) state.recoveryBlocked = 'Original execution belongs to another host; liveness cannot be verified here';
      else {
        try { process.kill(record.hostPid, 0); state.recoveryBlocked = 'Original host PID still exists; execution ownership cannot be established'; }
        catch (error) { if (error.code !== 'ESRCH') state.recoveryBlocked = 'Original host liveness could not be verified'; }
      }
    } else state.authority = 'durable_outcome';
    if (TERMINAL.has(state.state) && (!state.outcome || state.outcome.cleanupVerified === false)) {
      state.recoveryBlocked = 'Producer resource cleanup has not been verified; inspect the original execution before recovery';
    }
    try { state.recovery = JSON.parse(await readFile(join(directory, 'recovery.json'), 'utf8')); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    return state;
  }

  async list(workspace, owner, jobs) {
    let directories;
    try { directories = await readdir(join(workspace, '.control-jobs')); }
    catch (error) { if (error.code === 'ENOENT') return []; throw error; }
    const records = [];
    for (const id of directories.filter(value => ATTEMPT.test(value))) {
      const directory = await this.directory(workspace, id);
      let snapshot;
      try { snapshot = JSON.parse(await readFile(join(directory, 'request.json'), 'utf8')); }
      catch (error) { if (error.code === 'ENOENT') continue; throw error; }
      if (snapshot.owner === owner) records.push(await this.read(workspace, id, owner, jobs));
    }
    return records.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  /** Atomic first-wins retry link; concurrent clicks can never admit two replacement jobs. */
  async claimRecovery(workspace, attemptId, childAttemptId, mode) {
    const directory = await this.directory(workspace, attemptId);
    try {
      await atomicJson(join(directory, 'recovery.json'), { attemptId: childAttemptId, mode, createdAt: new Date().toISOString() }, true);
      return childAttemptId;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
      return JSON.parse(await readFile(join(directory, 'recovery.json'), 'utf8')).attemptId;
    }
  }
}

/** Derive a new request; a checkpoint resumes numerical state, never the old process. */
export async function recoveryRequest(record, mode, checkpoint) {
  if (!['retry', 'resume'].includes(mode)) throw new Error('Recovery mode must be retry or resume');
  if (!['interrupted', 'failed', 'killed'].includes(record.state)) throw new Error('Only interrupted, failed or killed attempts may be recovered');
  if (record.recoveryBlocked) throw new Error(record.recoveryBlocked);
  const request = structuredClone(record.request);
  request.config ??= {};
  if (mode === 'retry') {
    return request;
  }
  if (request.operation === 'run_workflow') {
    if (!record.workflowRun) throw new Error('No recorded workflow run id; inspect workflow state before resuming');
    request.config.resume_run = record.workflowRun;
    return request;
  }
  if (!TRAINING.has(request.operation)) throw new Error('This operation has no checkpoint resume support; use retry');
  const selected = checkpoint ?? record.checkpoint;
  if (typeof selected !== 'string' || !selected) throw new Error('A saved training checkpoint is required for resume');
  const path = await realpath(resolve(record.workspace, selected));
  if (!inside(record.workspace, path) || !(await stat(path)).isFile()) throw new Error('Checkpoint must be a file inside this workspace');
  request.config.resume_checkpoint = path;
  return request;
}

/** Persist worker lifecycle before native settlement. This does not own or recreate jobs. */
export function persistedJob({ writer, job, start }) {
  let hooks, cancelled = false;
  const done = (async () => {
    let outcome;
    try {
      await writer.append({ state: 'queued', nativeJobId: job.id });
      if (cancelled) outcome = { status: 'killed', detail: 'Cancelled before numerical worker admission' };
      else {
        hooks = start(event => {
          writer.append({ workerEvent: event, ...(event.event === 'started' ? { state: 'running' } : {}) })
            .catch(() => { try { hooks?.cancel(); } catch { /* done remains the cleanup authority */ } });
        });
        if (cancelled) hooks.cancel();
        outcome = await hooks.done;
      }
      await writer.flush();
      await writer.append({ state: outcome.status, outcome });
    } catch (error) {
      if (hooks) {
        try { hooks.cancel(); await hooks.done; }
        catch (cleanupError) { return { status: 'failed', detail: `History failed: ${error.message}; cleanup could not be verified: ${cleanupError.message}` }; }
      }
      outcome = { status: 'failed', detail: `Durable job history failed: ${error.message}` };
    }
    return outcome;
  })();
  return { cancel() { cancelled = true; hooks?.cancel(); }, done };
}
