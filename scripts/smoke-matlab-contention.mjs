/** S02 M03: real Python/batch MATLAB consumer versus two native Harness Sessions. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { readFile, readdir, stat } from 'node:fs/promises';
import { isAbsolute, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { pathToFileURL } from 'node:url';
import { parseMatlabTerminalProbe } from '../harness-plugin/src/matlab-terminal.mjs';
import { connectHarness, command, prepareSmokeWorkspace, project, writeSmokeEvidence } from './harness-rpc.mjs';

const script = resolve(import.meta.dirname, 'smoke-matlab-contention.mjs');
const wait = ms => new Promise(resolveWait => setTimeout(resolveWait, ms));
const hash = text => createHash('sha256').update(text).digest('hex');
const readState = directory => readFile(join(directory, 'resource.json'), 'utf8').then(JSON.parse);

/** Use the already installed official SDK; no new dependency or lock mutation. */
export async function resolveOfficialSdk() {
  const store = resolve(project, 'vendor/deepseek-harness/node_modules/.pnpm');
  const matches = (await readdir(store)).filter(name => name.startsWith('@modelcontextprotocol+sdk@'));
  assert.equal(matches.length, 1, 'Expected one already-installed official @modelcontextprotocol/sdk version');
  const root = resolve(store, matches[0], 'node_modules/@modelcontextprotocol/sdk');
  const manifest = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
  assert.equal(manifest.name, '@modelcontextprotocol/sdk');
  return { root, version: manifest.version };
}

/** Runs only as the Python holder's child. There is no user-supplied code. */
async function batchActor(options) {
  let client, transport, stderrBytes = 0, stage = 'dedicated-identity-and-resource-binding';
  const record = { schema_version: 1, confirmation_source: 'test_fixture', status: 'RUNNING',
    sdkNodePid: process.pid, startedAt: new Date().toISOString(), calls: [] };
  try {
    const identity = JSON.parse(await readFile(join(options.privateAppdata, 'control-harness-session.json'), 'utf8'));
    assert.equal(identity.kind, 'control-harness-dedicated-matlab-v1');
    assert.equal(identity.status, 'ready');
    assert.equal(resolve(identity.private_appdata), options.privateAppdata);
    const state = await readState(options.resourceDirectory);
    assert.equal(state.owner?.phase, 'running', 'Python actor must own a durable running lease before starting its SDK child');
    assert.equal(state.owner.pid, Number(process.env.CONTROL_S02_FIXTURE_OWNER_PID));
    const session = state.owner.session;
    assert.equal(session?.pid, identity.matlab_pid);
    assert.equal(session?.session_id, identity.session_id);
    const sdk = await resolveOfficialSdk();
    const [{ Client }, { StdioClientTransport }] = await Promise.all([
      import(pathToFileURL(join(sdk.root, 'dist/esm/client/index.js'))),
      import(pathToFileURL(join(sdk.root, 'dist/esm/client/stdio.js'))),
    ]);
    record.sdk = { name: '@modelcontextprotocol/sdk', version: sdk.version };
    record.session = { session_id: identity.session_id, matlab_pid: identity.matlab_pid, identity: session.identity };
    client = new Client({ name: 'control-harness-s02-fixed-batch-fixture', version: '1.0.0' }, { capabilities: {} });
    transport = new StdioClientTransport({ command: options.mcpExecutable,
      args: ['--matlab-session-mode=existing', '--disable-telemetry=true'],
      env: { APPDATA: options.privateAppdata }, cwd: project, stderr: 'pipe' });
    // Drain the owned server's diagnostics without printing session descriptors
    // or credentials. Public successful math/probe output is recorded below.
    transport.stderr?.on('data', chunk => { stderrBytes += chunk.length; });
    stage = 'official-sdk-connect-to-existing-dedicated-session';
    await client.connect(transport);
    record.ownedMcpPid = transport.pid;

    const probe = async () => {
      const nonce = randomUUID();
      const code = `fprintf('CONTROL_HARNESS_PROBE_${nonce}:%s\\n',jsonencode(control_harness_terminal_probe('${nonce}')));`;
      const result = await client.callTool({ name: 'evaluate_matlab_code', arguments: { code } }, undefined, { timeout: 30000 });
      assert.notEqual(result.isError, true, 'Real batch terminal probe returned an MCP error');
      return parseMatlabTerminalProbe(result, { nonce, identity, session,
        helperFile: resolve(identity.project_root, 'matlab/control_harness_terminal_probe.m') });
    };
    for (let index = 1; index <= 3; index += 1) {
      stage = `batch-${index}-before-terminal-probe`;
      const before = await probe();
      const code = `fprintf('CONTROL_HARNESS_S02_BATCH_${index}:%d\\n',6*7);`;
      const startedAt = new Date().toISOString(), started = Date.now();
      stage = `batch-${index}-fixed-arithmetic`;
      const result = await client.callTool({ name: 'evaluate_matlab_code', arguments: { code } }, undefined, { timeout: 30000 });
      assert.notEqual(result.isError, true, 'Real batch arithmetic returned an MCP error');
      const actualOutput = result.content.filter(item => item.type === 'text').map(item => item.text).join('\n').trim();
      assert.equal(actualOutput, `CONTROL_HARNESS_S02_BATCH_${index}:42`);
      stage = `batch-${index}-after-terminal-probe`;
      const after = await probe();
      record.calls.push({ index, confirmation_source: 'test_fixture', fixedCode: code, actualOutput,
        startedAt, settledAt: new Date().toISOString(), durationMs: Date.now() - started, before, after });
    }
    // All synchronous calls and their terminal probes have settled before close.
    // Closing this owned MCP client is not used as evidence that MATLAB stopped.
    stage = 'owned-sdk-client-close-after-verified-terminal-calls';
    await client.close();
    client = undefined;
    record.clientClosedAt = new Date().toISOString();
    record.status = 'PASS';
  } catch (error) {
    record.status = 'FAIL';
    record.failureStage = stage;
    record.errorType = error?.name ?? 'Error';
    record.diagnosticSha256 = hash(String(error?.stack ?? error));
    record.error = 'Official SDK batch fixture failed; raw transport diagnostics are deliberately not printed';
    process.exitCode = 1;
  } finally {
    if (client) {
      try { await client.close(); record.clientClosedAfterFailure = true; }
      catch { record.clientCloseUnverified = true; }
    }
    record.stderrBytes = stderrBytes;
    record.settledAt = new Date().toISOString();
    console.log(JSON.stringify(record));
  }
}

// The actor never executes user materials. Its only numerical child is this
// script's fixed SDK batch mode; stdin has one literal release command.
const pythonHolder = String.raw`
import json, os, subprocess, sys, time
from control_harness.matlab_resource import MatlabResource
from control_harness.locking import process_alive
directory, node, script, private, executable = sys.argv[1:]
def emit(event, **data): print(json.dumps({'event':event,'pythonPid':os.getpid(),**data}),flush=True)
try:
    resource = MatlabResource(directory,.01)
    with resource.acquire('S02 confirmation_source=test_fixture Python real MATLAB batch',on_queued=lambda item:emit('queued',ticket=item['ticket'])) as lease:
        lease.mark_running()
        emit('acquired',token=lease.owner['token'],ticket=lease.owner['ticket'],session=lease.owner['session'])
        env = {**os.environ,'CONTROL_S02_FIXTURE_OWNER_PID':str(os.getpid())}
        child = subprocess.run([node,script,'--batch-actor','--resource-directory',directory,'--private-appdata',private,'--mcp-executable',executable],capture_output=True,text=True,encoding='utf-8',env=env)
        try: batch = json.loads(child.stdout)
        except Exception: batch = {'status':'FAIL','error':'SDK actor did not return one valid JSON evidence record'}
        if child.returncode != 0 or batch.get('status') != 'PASS' or len(batch.get('calls',[])) != 3:
            lease.quarantine('s02_real_batch_terminal_state_unverified')
            emit('error',message='Real MATLAB batch failed; resource quarantined',batch=batch)
            sys.exit(2)
        deadline = time.monotonic() + 5
        while process_alive(batch.get('ownedMcpPid')) is not False and time.monotonic() < deadline:
            time.sleep(.025)
        if process_alive(batch.get('ownedMcpPid')) is not False:
            lease.quarantine('s02_owned_batch_mcp_exit_unverified')
            emit('error',message='Owned SDK MCP process exit could not be verified; resource quarantined')
            sys.exit(2)
        batch['sdkNodeExitCode'] = child.returncode
        batch['ownedMcpExitVerified'] = True
        emit('batch_done',batch=batch)
        if sys.stdin.readline().strip() != 'release':
            lease.quarantine('s02_fixture_parent_disconnected_before_release')
            emit('error',message='Fixture controller disconnected; resource quarantined')
            sys.exit(2)
        lease.complete({'kind':'synchronous_adapter_return','producer':'S02-official-sdk-fixed-real-batch','confirmation_source':'test_fixture','completed_calls':3,'session_id':batch['session']['session_id'],'matlab_pid':batch['session']['matlab_pid'],'client_closed_at':batch['clientClosedAt']})
        emit('released')
except Exception as error:
    emit('error',message='Python resource/batch fixture failed',error_type=type(error).__name__)
    sys.exit(2)
`;

function startHolder(options) {
  const python = resolve(project, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
  const child = spawn(python, ['-u', '-c', pythonHolder, options.resourceDirectory, process.execPath,
    script, options.privateAppdata, options.mcpExecutable], { cwd: project,
    env: { ...process.env, PYTHONPATH: resolve(project, 'src'), PYTHONDONTWRITEBYTECODE: '1' },
    windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  const events = [], waiters = new Set();
  let pending = '', exited = false, stderrBytes = 0;
  const flush = () => {
    for (const waiter of waiters) {
      const error = events.find(item => item.event === 'error');
      const result = events.find(item => item.event === waiter.event);
      if (result) { waiters.delete(waiter); clearTimeout(waiter.timer); waiter.resolve(result); }
      else if (error || exited) {
        waiters.delete(waiter); clearTimeout(waiter.timer);
        waiter.reject(new Error(error?.message ?? `Python holder exited before ${waiter.event}`));
      }
    }
  };
  child.stderr.on('data', chunk => { stderrBytes += chunk.length; });
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', chunk => {
    pending += chunk;
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, newline).trim(); pending = pending.slice(newline + 1);
      if (line) {
        try { events.push(JSON.parse(line)); }
        catch { events.push({ event: 'error', message: 'Python holder emitted an invalid evidence line' }); }
      }
    }
    flush();
  });
  const done = new Promise((resolveDone, reject) => {
    child.once('error', reject);
    child.once('close', code => { exited = true; flush(); resolveDone({ exitCode: code, stderrBytes }); });
  });
  return { child, events, done,
    wait(event) { return new Promise((resolveWait, reject) => {
      const waiter = { event, resolve: resolveWait, reject };
      waiter.timer = setTimeout(() => { waiters.delete(waiter); reject(new Error(`Python holder did not reach ${event} within 120 seconds`)); }, 120000);
      waiters.add(waiter); flush();
    }); },
    release() { if (!exited && !child.stdin.destroyed) child.stdin.end('release\n'); } };
}

function options() {
  const { values } = parseArgs({ options: { workspace: { type: 'string' }, evidence: { type: 'string' },
    'web-log': { type: 'string' }, 'resource-directory': { type: 'string' },
    'private-appdata': { type: 'string' }, 'mcp-executable': { type: 'string' },
    'batch-actor': { type: 'boolean' }, help: { type: 'boolean', short: 'h' } } });
  if (values.help) {
    console.log('node scripts/smoke-matlab-contention.mjs --resource-directory <absolute> --private-appdata <absolute> [--mcp-executable <absolute>] [--workspace <absolute>] [--evidence <json>] [--web-log <log>]');
    console.log('Requires a healthy dedicated existing MATLAB session and the optional /matlab-lifecycle acceptance plugin. Executes a fixed real 3-call SDK batch under Python OS ownership, then two native role Sessions contend. No Host/session restart or recovery.');
    return { help: true };
  }
  const selected = { batchActor: !!values['batch-actor'],
    resourceDirectory: values['resource-directory'] ?? process.env.CONTROL_MATLAB_RESOURCE_DIRECTORY,
    privateAppdata: values['private-appdata'] ?? process.env.CONTROL_MATLAB_SESSION_APPDATA,
    mcpExecutable: values['mcp-executable'] ?? process.env.CONTROL_MATLAB_MCP_EXECUTABLE,
    workspace: resolve(values.workspace ?? process.env.CONTROL_SMOKE_WORKSPACE ?? resolve(project, 'test-output', `matlab-contention-${randomUUID()}`)),
    evidence: resolve(values.evidence ?? process.env.CONTROL_SMOKE_EVIDENCE ?? resolve(project, 'test-output/smoke-evidence', `matlab-contention-${randomUUID()}.json`)),
    webLog: resolve(values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG ?? resolve(project, 'harness-web.log')) };
  for (const key of ['resourceDirectory', 'privateAppdata', 'mcpExecutable']) {
    assert.ok(typeof selected[key] === 'string' && isAbsolute(selected[key]), `${key} must be an explicit absolute path or its CONTROL_MATLAB_* environment value`);
    selected[key] = resolve(selected[key]);
  }
  return selected;
}

async function main(options) {
  const evidence = { schema_version: 1, confirmation_source: 'test_fixture', status: 'RUNNING',
    startedAt: new Date().toISOString(), workspace: options.workspace,
    resourceDirectory: options.resourceDirectory, privateAppdata: options.privateAppdata,
    scope: 'Real OS-owned Python resource, official SDK existing MATLAB batch and two native role Sessions; no model requests or recovery', sessions: [] };
  let holder;
  const commands = [];
  await writeSmokeEvidence(options.evidence, evidence);
  try {
    assert.ok((await stat(options.mcpExecutable)).isFile(), 'Explicit installed MATLAB MCP executable is missing');
    evidence.workspacePreparation = await prepareSmokeWorkspace(options.workspace);
    evidence.resourceBefore = await readState(options.resourceDirectory);
    assert.equal(evidence.resourceBefore.owner, null, 'Start from a healthy idle resource');
    assert.equal(evidence.resourceBefore.quarantine, null, 'This acceptance script never recovers or resets a quarantine');
    assert.deepEqual(evidence.resourceBefore.queue, []);
    const rpc = await connectHarness({ logPath: options.webLog });
    const roster = await rpc('agentPresets/list', {});
    for (const role of ['model', 'controller']) {
      assert.ok(roster.presets.some(item => item.id === `control-${role}` && !item.broken));
      const workspace = resolve(options.workspace, role);
      await prepareSmokeWorkspace(workspace);
      const created = await rpc('session/create', { request: { cwd: workspace, agentPreset: `control-${role}` } });
      const available = await rpc('commands/list', { agentId: created.sessionId });
      assert.ok(available.some(item => item.name === 'matlab-lifecycle'), 'Load the optional fixed /matlab-lifecycle acceptance plugin');
      evidence.sessions.push({ role, workspace, sessionId: created.sessionId });
    }
    assert.notEqual(evidence.sessions[0].sessionId, evidence.sessions[1].sessionId);
    await writeSmokeEvidence(options.evidence, evidence);
    holder = startHolder(options);
    evidence.pythonOwner = await holder.wait('acquired');
    await writeSmokeEvidence(options.evidence, evidence);
    const batch = await holder.wait('batch_done');
    evidence.realBatch = batch.batch;
    assert.equal(evidence.realBatch.status, 'PASS');
    assert.equal(evidence.realBatch.calls.length, 3);
    assert.ok(evidence.realBatch.calls.every(item => /:42$/.test(item.actualOutput) && item.before && item.after));
    assert.ok(evidence.realBatch.clientClosedAt, 'SDK client closed only after all real calls and terminal probes settled');
    evidence.batchDoneWhilePythonStillOwns = await readState(options.resourceDirectory);
    assert.equal(evidence.batchDoneWhilePythonStillOwns.owner.pid, evidence.pythonOwner.pythonPid);
    await writeSmokeEvidence(options.evidence, evidence);

    for (const session of evidence.sessions) {
      const row = { session, settled: false };
      row.promise = command(rpc, session.sessionId, '/matlab-lifecycle queue-one')
        .then(result => { row.settled = true; row.result = result; return result; },
          error => { row.settled = true; row.error = error.message; return undefined; });
      commands.push(row);
    }
    const deadline = Date.now() + 15000;
    while (true) {
      const snapshot = await readState(options.resourceDirectory);
      if (snapshot.queue.length === 2) {
        assert.equal(snapshot.owner.pid, evidence.pythonOwner.pythonPid);
        assert.ok(commands.every(row => !row.settled), 'Both native commands must still be waiting behind the real Python owner');
        assert.ok(snapshot.queue.every(row => row.label.startsWith('mcp__matlab__evaluate_matlab_code:')));
        assert.ok(snapshot.queue[0].ticket < snapshot.queue[1].ticket, 'Persisted FIFO tickets establish arrival order');
        evidence.twoSessionsQueuedWhilePythonOwns = snapshot;
        evidence.queueObservedAt = new Date().toISOString();
        break;
      }
      assert.ok(commands.every(row => !row.settled), 'A native command settled before it was observed in the shared queue');
      assert.ok(Date.now() < deadline, 'Both real native Sessions did not enter the shared FIFO within 15 seconds');
      await wait(30);
    }
    await writeSmokeEvidence(options.evidence, evidence);
    evidence.pythonReleaseRequestedAt = new Date().toISOString();
    holder.release();
    evidence.pythonReleased = await holder.wait('released');
    evidence.pythonExit = await holder.done;
    assert.equal(evidence.pythonExit.exitCode, 0);
    await Promise.all(commands.map(row => row.promise));
    for (const row of commands) {
      assert.equal(row.error, undefined, row.error);
      assert.equal(row.result?.status, 'PASS', row.result?.error);
      row.session.result = row.result;
      assert.ok(Date.parse(row.result.calls[0].settledAt) >= Date.parse(evidence.pythonReleaseRequestedAt),
        'A queued native tool cannot settle successfully before Python releases the shared resource');
      const events = await rpc('session/events', { request: { address: { kind: 'session', sessionId: row.session.sessionId }, maxMessages: 100 } });
      row.session.eventTypes = [...new Set(events.records.map(item => item.event.type))];
      assert.ok(!row.session.eventTypes.some(type => type.startsWith('turn/')));
    }
    evidence.resourceAfter = await readState(options.resourceDirectory);
    assert.equal(evidence.resourceAfter.owner, null);
    assert.equal(evidence.resourceAfter.quarantine, null);
    assert.deepEqual(evidence.resourceAfter.queue, []);
    evidence.pythonEvents = holder.events;
    evidence.status = 'PASS';
  } catch (error) {
    evidence.status = 'FAIL'; evidence.error = error.message;
    throw error;
  } finally {
    // A literal release is consumed only after the batch has returned verified
    // terminal results. Failed/unknown batch state quarantines instead. Never
    // kill a possibly computing MATLAB process to make this test pass.
    if (holder) {
      holder.release();
      try { evidence.pythonExit ??= await holder.done; } catch (error) { evidence.pythonExitError = error.message; }
      evidence.pythonEvents = holder.events;
    }
    await Promise.allSettled(commands.map(row => row.promise));
    for (const row of commands) if (row.result) row.session.result = row.result;
    try { evidence.resourceAfter = await readState(options.resourceDirectory); } catch { /* Retain existing failure. */ }
    evidence.settledAt = new Date().toISOString();
    await writeSmokeEvidence(options.evidence, evidence);
    console.log(JSON.stringify({ status: evidence.status, evidence: options.evidence, workspace: options.workspace,
      pythonPid: evidence.pythonOwner?.pythonPid, sessionIds: evidence.sessions.map(item => item.sessionId),
      actualBatchCalls: evidence.realBatch?.calls?.length ?? 0, ...(evidence.error ? { error: evidence.error } : {}) }));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const selected = options();
  if (!selected.help) {
    if (selected.batchActor) await batchActor(selected);
    else await main(selected);
  }
}
