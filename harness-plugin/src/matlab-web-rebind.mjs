import { execFile } from 'node:child_process';
import { rmSync } from 'node:fs';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { promisify } from 'node:util';

const run = promisify(execFile);
const jobActive = job => job.status === 'running' || job.status === 'stopping';

function hostBusy(ctx, agents) {
  return ctx.jobs.list().some(jobActive) || agents.some(agent =>
    agent.status !== 'idle' || agent.inbox.nextTurn.length || agent.inbox.nextStep.length ||
    ctx.jobs.list(agent.id).some(jobActive));
}

function releasedBinding(state, identity) {
  if (state.session_alive !== false || state.session?.pid !== identity.pid ||
      state.session.session_id !== identity.session_id || state.session.host !== identity.host ||
      state.session.identity !== identity.identity || state.owner !== null || !Array.isArray(state.queue) || state.queue.length) return false;
  const recorded = state.quarantine?.owner?.session;
  return !recorded || (recorded.session_id === state.session.session_id && recorded.pid === state.session.pid &&
    recorded.identity === state.session.identity && recorded.host === state.session.host);
}

/** Recheck immediately before calling the launcher's orderly whole-Host disposal. */
export async function requestWebRebind(ctx, identity, inspect, exit, noOpen, signal) {
  const agents = ctx.agents.list();
  if (hostBusy(ctx, agents) || !releasedBinding(await inspect(), identity)) return false;
  await Promise.all(agents.map(agent => agent.whenIdle())); // includes between-turn maintenance
  // A turn or a new owner could have appeared while the OS identity was checked.
  if (signal.aborted || !releasedBinding(await inspect(), identity)) return false;
  const current = ctx.agents.list();
  if (signal.aborted || current.some(agent => !agents.includes(agent)) || hostBusy(ctx, current)) return false;
  exit(noOpen ? 76 : 75);
  return true;
}

export async function superviseWebBinding(ctx, identity, signal, env) {
  const root = env.CONTROL_CODE_ROOT, resource = env.CONTROL_MATLAB_RESOURCE_DIRECTORY;
  const directory = join(root, '.dsh', 'oneclick');
  const marker = join(directory, `web-${process.pid}.json`);
  const request = join(directory, `web-${process.pid}-${identity.session_id}.rebind`);
  const inspect = async () => {
    const {stdout} = await run(env.CONTROL_PYTHON,
      ['-m','control_harness.matlab_resource','inspect','--directory',resource],
      {cwd:root, env:{...process.env, PYTHONPATH:join(root,'src')}, windowsHide:true, timeout:10000});
    return JSON.parse(stdout);
  };
  // Capture the broker's OS creation identity while the MCP handshake and
  // the bound process both agree on the same live session.
  const startup = await inspect();
  const binding = startup.session;
  if (startup.session_alive !== true || binding?.pid !== identity.matlab_pid ||
      binding.session_id !== identity.session_id || !binding.host || !/^win32:[0-9a-f]{16}$/.test(binding.identity)) {
    throw new Error('网页启动期间专用 MATLAB 进程或身份已变化。');
  }
  await mkdir(directory, {recursive:true});
  await rm(request, {force:true});
  await writeFile(marker + '.tmp', JSON.stringify({web_pid:process.pid, project_root:root,
    resource_directory:resource, matlab_pid:identity.matlab_pid, session_id:identity.session_id}));
  await rename(marker + '.tmp', marker);
  const runId = env.CONTROL_ONECLICK_HOST_RUN;
  const readyFile = /^[0-9a-f]{32}$/.test(runId ?? '') ? join(directory,`ready-${runId}.json`) : null;
  let published = false;
  async function publishReady() {
    const port = ctx.webServer?.port;
    if (!readyFile || published || !port) return;
    const response = await fetch(`http://127.0.0.1:${port}/`, {
      redirect:'error', signal:AbortSignal.any([signal,AbortSignal.timeout(2000)]),
    });
    if (response.status !== 401) { await response.body?.cancel(); return; }
    if ((await response.text()).trim() !== 'dsh web authentication required; reopen the URL printed by dsh web.') return;
    if (signal.aborted) return;
    await writeFile(readyFile+'.tmp', JSON.stringify({launch_id:runId, web_pid:process.pid,
      project_root:root, resource_directory:resource, port, session:binding}));
    await rename(readyFile+'.tmp',readyFile);
    published = true;
  }
  // Retire the handoff before the CLI begins bounded teardown. Its process
  // exit deadline may fire before asynchronous plugin disposers are run.
  const retire = () => {
    rmSync(request, {force:true});
    rmSync(marker, {force:true});
    if (readyFile) rmSync(readyFile, {force:true});
  };
  const restart = code => {
    retire();
    ctx.get('appExit')(code);
  };
  let checking = false;
  let nextIdentityCheck = 0;
  const timer = setInterval(async () => {
    if (checking || signal.aborted) return;
    checking = true;
    try {
      try { await publishReady(); }
      catch (error) { if (!signal.aborted) ctx.logger?.debug?.(`尚未记录网页就绪回执：${error.message}`); }
      let mode;
      try { mode = await readFile(request, 'utf8'); }
      catch (error) { if (error.code !== 'ENOENT') throw error; }
      const requested = mode === 'open' || mode === 'no-open';
      if (!requested && Date.now() < nextIdentityCheck) return;
      nextIdentityCheck = Date.now() + 15000;
      // The supervisor restarts on the same port. Keep the current tab on
      // automatic recovery; an explicit launcher request may open a new tab.
      if (await requestWebRebind(ctx, binding, inspect, restart, mode !== 'open', signal)) clearInterval(timer);
    } catch (error) {
      if (error.code !== 'ENOENT' && !signal.aborted) ctx.logger.warn(`专用 MATLAB 重新绑定核验未通过：${error.message}`);
    } finally { checking = false; }
  }, 1500);
  timer.unref();
  ctx.effect(() => () => {
    clearInterval(timer);
    retire();
  });
}
