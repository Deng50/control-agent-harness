/** Real OS processes and native Cordis dispatch; remote-work flags are explicit failure-boundary fixtures. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installMatlabResource, dispatchMatlabTaskBatch } from '../src/matlab-resource.mjs';
import { createMatlabTerminalVerifier } from '../src/matlab-terminal.mjs';

const project = resolve(import.meta.dirname, '../..');
const upstream = resolve(project, 'vendor/deepseek-harness');
const python = resolve(project, '.venv/Scripts/python.exe');
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/matlab-resource-tests'));
const builtUrl = path => pathToFileURL(resolve(upstream, path)).href;
const built = path => import(builtUrl(path));
await mkdir(output, { recursive: true });
const proof = { kind: 'synchronous_adapter_return', producer: 'node-test-trusted-synchronous-adapter' };

async function context(directory, options = {}) {
  const [{ Context }, { default: Prompt }, { default: Tools }, Executor, { default: Local }, mcp] = await Promise.all([
    built('vendor/cordis/lib/index.js'), built('packages/core/system-prompt/lib/index.js'),
    built('packages/core/tools/lib/index.js'), built('packages/core/tool-executor-native/lib/index.js'),
    built('packages/subprocess/subprocess-local/lib/index.js'), built('packages/mcp/mcp-client/lib/index.js'),
  ]);
  const ctx = new Context();
  await ctx.plugin(Prompt);
  await ctx.plugin(Tools);
  await ctx.plugin(Executor);
  await ctx.plugin(Local);
  const resource = installMatlabResource(ctx, { pythonPath: python, codeRoot: project,
    matlabResource: { directory, pollIntervalMs: 10, toolPrefixes: ['mcp__matlab__'] } },
  { verifyBeforeDispatch: async () => proof, ...options });
  const register = (name, call) => ctx.tools.register(mcp.createMcpToolDefinition(ctx, {
    name, rawName: name, description: 'Explicit MCP adapter boundary fixture',
    inputSchema: { type: 'object', properties: {} }, call,
  }));
  return { ctx, resource, register };
}

async function execute(ctx, name, signal = new AbortController().signal, args = {}) {
  const result = await ctx.tools.execute({ calls: [{ type: 'tool_call', id: `call-${Math.random()}`, name, arguments: JSON.stringify(args) }], signal });
  return result.results[0].result;
}

async function fixture(callback) {
  const temporary = await mkdtemp(join(output, 'resource-node-'));
  try { return await callback(temporary); }
  finally { await rm(temporary, { recursive: true, force: true }); }
}

const state = directory => readFile(join(directory, 'resource.json'), 'utf8').then(JSON.parse);
const mcpSuccess = text => ({ content: [{ type: 'text', text }] });

// 重放两次真实会话的 MCP 错误：原生工具桥会在会话记录外层另加 "Error: "。
// 仅 MATLAB 服务端为替身；权限检查、探针解析和 Python 资源代理使用真实实现。
const inport = {
  label: 'Inport Value',
  text: "matlab error: Inport block 没有名为 'Value' 的参数\n \n",
  invalid: {code:"set_param('MassSpringDamper/u_force','Value','1');"},
  corrected: {code:"set_param('MassSpringDamper/u_force','Port','1');"},
};
const plant = {
  label: 'event 277 Plant B',
  text: `matlab error: 参数 'B' 的 '<a href="matlab:open_and_hilite_hyperlink ('MPC_MassSpringDamper/Plant','error')">MPC_MassSpringDamper/Plant</a>' 中的设置无效。\n \n`,
  invalid: {code:"set_param('MPC_MassSpringDamper/Plant','A','eye(2)');"},
  corrected: {code:"set_param('MPC_MassSpringDamper/Plant','A','eye(2)','B','eye(2)','C','eye(2)','D','zeros(2)','InitialCondition','[1;0]','SampleTime','0.01');"},
};
const terminalConditions = ['none', 'busy', 'different-session', 'cancel-during-probe', 'timeout', 'transport', 'owner-exited'];
const parameterScenarios = [
  {...inport, conditions:terminalConditions}, {...plant, conditions:terminalConditions},
  {...inport, label:'English Inport', text:"matlab error: Inport block does not have a parameter named 'Value'."},
  {...plant, label:'Chinese plain text', text:"matlab error: 参数 'B' 的 '模型/对象' 中的设置无效。"},
  {...inport, label:'mixed SDK failure', text:inport.text + 'SDK request deadline elapsed', classified:false},
  {...plant, label:'extra diagnostic', text:plant.text + 'MCP transport disconnected', classified:false},
  {...plant, label:'conflicting link target', text:plant.text.replace('>MPC_MassSpringDamper/Plant<','>AnotherModel/Plant<'), classified:false},
  {...plant, label:'unrecognized markup', text:"matlab error: 参数 'B' 的 '<b>model/Plant</b>' 中的设置无效。", classified:false},
];
for (const scenario of parameterScenarios) for (const fault of scenario.conditions ?? ['none']) {
  test(`${scenario.label} diagnostic, terminal condition: ${fault}`, async () => fixture(async directory => {
    const identityFile = join(directory, 'identity.json'), sessionId = randomUUID();
    const bound = spawnSync(python, ['-m', 'control_harness.matlab_resource', 'bind-session',
      '--directory', directory, '--pid', String(process.pid), '--session-id', sessionId],
    { cwd: project, env: {...process.env, PYTHONPATH:resolve(project,'src')}, encoding:'utf8', windowsHide:true });
    assert.equal(bound.status, 0, bound.stderr);
    if (fault === 'owner-exited') {
      const initial = await state(directory);
      initial.session.identity = 'stale-process-birth-identity';
      await writeFile(join(directory,'resource.json'), JSON.stringify(initial));
    }
    await writeFile(identityFile, JSON.stringify({ schema_version:1, kind:'control-harness-dedicated-matlab-v1',
      status:'ready', project_root:project, matlab_pid:process.pid, session_id:sessionId }));
    const abort = new AbortController(), evalTool = 'mcp__matlab__evaluate_matlab_code';
    let ctx, requested = 0, probes = 0, permissions = 0;
    const verify = args => createMatlabTerminalVerifier(ctx, {identityFile, codeRoot:project})(args);
    const mounted = await context(directory, {verifyBeforeDispatch:verify, verifyTerminal:verify});
    ctx = mounted.ctx;
    const {invalid, corrected, text:errorText} = scenario;
    const qualified = scenario.classified !== false && !['timeout','transport'].includes(fault);
    const retryable = qualified && fault === 'none';
    try {
      ctx.tools.guard(()=>{ permissions++; });
      mounted.register(evalTool, async ({code}) => {
        const nonce = code.match(/CONTROL_HARNESS_PROBE_([a-f0-9-]{36}):/)?.[1];
        if (nonce) {
          probes++;
          assert.ok(code.includes(resolve(project,'matlab/control_harness_terminal_probe.m')));
          if (probes === 2 && fault === 'cancel-during-probe') abort.abort();
          return mcpSuccess(`CONTROL_HARNESS_PROBE_${nonce}:` + JSON.stringify({
            nonce, schema_version:1, matlab_pid:process.pid,
            session_id:probes === 2 && fault === 'different-session' ? randomUUID() : sessionId,
            helper_file:resolve(project,'matlab/control_harness_terminal_probe'), idle:true, busy_models:[],
            background_futures:0, parallel_futures:0, running_timers:probes === 2 && fault === 'busy' ? 1 : 0,
          }));
        }
        requested++;
        if (fault === 'timeout') throw new Error('SDK request deadline elapsed');
        if (fault === 'transport') throw new Error('MCP transport disconnected');
        if (code === invalid.code) return {isError:true, content:[{type:'text',text:errorText}]};
        assert.equal(code, corrected.code);
        return mcpSuccess('corrected parameter');
      });
      const failed = await execute(ctx,evalTool,abort.signal,invalid);
      assert.equal(failed.isError,true);
      if (qualified) {
        assert.equal(failed.error.message,errorText);
        assert.equal(failed.meta.matlab_parameter_retryable,retryable);
      }
      const ledger = await state(directory);
      if (retryable) {
        assert.match(failed.content.at(-1).text,/修正参数/);
        assert.equal(ledger.owner,null); assert.equal(ledger.quarantine,null);
        assert.equal(ledger.last_completion.evidence.call_outcome,'synchronous_parameter_error');
        assert.equal(ledger.last_completion.evidence.matlab_pid,process.pid);
        assert.equal((await execute(ctx,evalTool,undefined,corrected)).isError,false);
        assert.equal(requested,2); assert.equal(probes,4); assert.equal(permissions,6);
      } else {
        assert.ok(ledger.quarantine,fault);
        assert.equal(probes,qualified?2:1);
        assert.equal((await execute(ctx,evalTool,undefined,corrected)).isError,true);
        assert.equal(requested,1);
      }
    } finally {await ctx.fiber.dispose();}
  }));
}

test('a scoped task batches the two signed identity probes under one native permission and one broker lease',async()=>fixture(async directory=>{
  // The bound PID is this test process. A fixture MCP callback substitutes for MATLAB.
  const identity=randomUUID();
  const bound=spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,'--pid',String(process.pid),'--session-id',identity],
    {cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')},windowsHide:true,encoding:'utf8'});
  assert.equal(bound.status,0,bound.stderr);
  const binding=JSON.parse(bound.stdout), scope={helper:'control_harness_collect_batch',native_tool:'mcp__matlab__evaluate_matlab_code',dedicated_session:binding};
  const {ctx,register}=await context(directory,{verifyTerminal:async()=>assert.fail('no separate probe')});
  const root=resolve(project,'matlab'), helper=resolve(root,'control_harness_terminal_probe');
  let permissions=0,dispatched=0,allow=true,omitPost=false;
  try {
    ctx.tools.guard(()=>{permissions++;return allow?undefined:'Fixture native denial';});
    register(scope.native_tool,async ({code,project_path})=>{
      assert.equal(project_path,root);
      const [nonce,session,before,after]=[...code.matchAll(/'([a-f0-9-]{36})'/g)].map(match=>match[1]);
      assert.equal(session,identity);assert.ok(nonce);
      return mcpSuccess((omitPost?[before]:[before,after]).map(token=>'CONTROL_HARNESS_PROBE_'+token+':'+JSON.stringify({
        schema_version:1,nonce:token,session_id:identity,matlab_pid:process.pid,helper_file:helper,
        busy_models:[],background_futures:0,parallel_futures:0,running_timers:0,idle:true})).join('\n'));
    });
    const dispatch=()=>dispatchMatlabTaskBatch({ctx,grant:{scope,validate:async lease=>assert.deepEqual(lease,binding)},
      onDispatch(){dispatched++;},signal:new AbortController().signal,manifestPath:resolve(directory,'manifest.json'),
      manifestSha:'a'.repeat(64),nonce:randomUUID(),helperRoot:root});
    allow=false;assert.equal((await dispatch()).results[0].result.isError,true);
    assert.equal(dispatched,0);assert.equal((await state(directory)).quarantine,null);
    allow=true;assert.equal((await dispatch()).results[0].result.isError,false);
    assert.equal(dispatched,1);assert.equal(permissions,2);
    assert.equal((await state(directory)).quarantine,null);
    // 即便卡片和 broker 文件一致，也要重新向 OS 核对创建时间。
    const healthy=await state(directory);
    const stale={...binding,identity:'stale-process-creation-fixture'};
    await writeFile(resolve(directory,'resource.json'),JSON.stringify({...healthy,session:stale}));
    scope.dedicated_session=stale;
    const refused=await dispatchMatlabTaskBatch({ctx,grant:{scope,validate:async lease=>assert.deepEqual(lease,stale)},
      onDispatch(){assert.fail('No dispatch to stale PID');},signal:new AbortController().signal,manifestPath:resolve(directory,'manifest.json'),
      manifestSha:'a'.repeat(64),nonce:randomUUID(),helperRoot:root});
    assert.equal(refused.results[0].result.isError,true);
    assert.equal((await state(directory)).owner,null);
    assert.equal((await state(directory)).quarantine,null);
    await writeFile(resolve(directory,'resource.json'),JSON.stringify(healthy));
    scope.dedicated_session=binding;
    omitPost=true;assert.equal((await dispatch()).results[0].result.isError,true);
    assert.equal(dispatched,2);assert.equal(permissions,4);
    assert.ok((await state(directory)).quarantine);
  } finally {await ctx.fiber.dispose();}
}));

test('native permissions reject before admission; unconfigured Python-only mode adds no MATLAB hook', async () => fixture(async directory => {
  const { ctx, register } = await context(directory);
  let calls = 0;
  try {
    register('mcp__matlab__denied', async () => { calls += 1; return mcpSuccess('must not run'); });
    ctx.tools.guard(exec => exec.name === 'mcp__matlab__denied' ? 'Native permission fixture denies MATLAB' : undefined);
    const denied = await execute(ctx, 'mcp__matlab__denied');
    assert.equal(denied.isError, true);
    assert.equal(calls, 0);
    await assert.rejects(readFile(join(directory, 'resource.json')), { code: 'ENOENT' });
    assert.equal(installMatlabResource(ctx, { pythonPath: python, codeRoot: project }), undefined);
  } finally { await ctx.fiber.dispose(); }
}));

test('trusted terminal probe re-enters native permissions while one lease permits consecutive calls', async () => fixture(async directory => {
  let ctx, probeCalls = 0, bodyCalls = 0, permissionChecks = 0;
  const verify = async ({ lease }) => lease.withInternalProbe(async () => {
    const result = await execute(ctx, 'mcp__matlab__fixed_probe');
    assert.equal(result.isError, false);
    return proof;
  });
  const mounted = await context(directory, { verifyBeforeDispatch: verify, verifyTerminal: verify });
  ctx = mounted.ctx;
  try {
    mounted.register('mcp__matlab__sync', async () => { bodyCalls += 1; return mcpSuccess('synchronous fixture completed'); });
    mounted.register('mcp__matlab__fixed_probe', async () => { probeCalls += 1; return mcpSuccess('trusted fixed probe fixture'); });
    ctx.tools.guard(() => { permissionChecks += 1; return undefined; });
    for (let i = 0; i < 2; i += 1) assert.equal((await execute(ctx, 'mcp__matlab__sync')).isError, false);
    assert.equal(bodyCalls, 2, 'next dispatches each requested tool exactly once');
    assert.equal(probeCalls, 4, 'fixed pre/post probes execute under the same lease');
    assert.equal(permissionChecks, 6, 'internal probing cannot bypass native permission guards');
    assert.equal((await state(directory)).quarantine, null);
  } finally { await ctx.fiber.dispose(); }
}));

test('native queued cancellation preserves the owner and returns canonical abort before tool dispatch', async () => fixture(async directory => {
  const { ctx, resource, register } = await context(directory, { verifyTerminal: async () => proof });
  const owner = await resource.acquire('other-host-owner-fixture', new AbortController().signal);
  let calls = 0;
  try {
    register('mcp__matlab__queued', async () => { calls += 1; return mcpSuccess('should dispatch only after release'); });
    const abort = new AbortController();
    const pending = execute(ctx, 'mcp__matlab__queued', abort.signal);
    const deadline = Date.now() + 10000;
    while ((await state(directory)).queue.length !== 1) {
      assert.ok(Date.now() < deadline, 'Native call must actually enqueue before cancellation');
      await new Promise(resolveWait => setTimeout(resolveWait, 10));
    }
    abort.abort(new Error('Queued native invocation cancelled'));
    const result = await pending;
    assert.equal(result.isError, true);
    assert.equal(result.error.info.code, 'ABORTED_BEFORE_DISPATCH');
    assert.equal(calls, 0);
    const remaining = await state(directory);
    assert.equal(remaining.owner.token, owner.token);
    assert.equal(remaining.quarantine, null);
    assert.deepEqual(remaining.queue, []);
    await owner.cancelBeforeDispatch();
    assert.equal((await execute(ctx, 'mcp__matlab__queued')).isError, false);
    assert.equal(calls, 1);
  } finally { if (!owner.closed) await owner.cancelBeforeDispatch(); await ctx.fiber.dispose(); }
}));

for (const precondition of ['wrong-session', 'busy', 'missing-verifier']) {
  test(`${precondition} pre-dispatch probe prevents the original MATLAB tool from running`, async () => fixture(async directory => {
    let calls = 0;
    const verifyBeforeDispatch = precondition === 'missing-verifier' ? undefined : async ({ lease }) => {
      assert.equal(lease.started, true, 'Preflight is durable and runs while this caller owns the execution lease');
      if (precondition === 'wrong-session') throw new Error('Dedicated session UUID/PID mismatch');
      return undefined; // Fixed probe reported active simulation/background work.
    };
    const { ctx, register } = await context(directory, { verifyBeforeDispatch, verifyTerminal: async () => assert.fail('No post-verification without original dispatch') });
    try {
      register('mcp__matlab__must_not_execute', async () => { calls += 1; return mcpSuccess('must not run'); });
      assert.equal((await execute(ctx, 'mcp__matlab__must_not_execute')).isError, true);
      assert.equal(calls, 0);
      assert.ok((await state(directory)).quarantine);
    } finally { await ctx.fiber.dispose(); }
  }));
}

test('an ordinary successful MCP result without trusted terminal proof remains quarantined', async () => fixture(async directory => {
  const { ctx, register } = await context(directory);
  let calls = 0;
  try {
    register('mcp__matlab__async_start', async () => { calls += 1; return mcpSuccess('accepted background work; idle=true is untrusted'); });
    assert.equal((await execute(ctx, 'mcp__matlab__async_start')).isError, false);
    assert.equal((await state(directory)).quarantine.reason, 'mcp_success_without_trusted_terminal_proof');
    assert.equal((await execute(ctx, 'mcp__matlab__async_start')).isError, true);
    assert.equal(calls, 1, 'unknown background state prevents later MATLAB dispatch');
  } finally { await ctx.fiber.dispose(); }
}));

test('cancellation arriving during terminal verification cannot turn into a clean release', async () => fixture(async directory => {
  const abort = new AbortController();
  const { ctx, register } = await context(directory, { verifyTerminal: async () => {
    abort.abort(new Error('Cancelled during terminal probe'));
    return proof;
  } });
  try {
    register('mcp__matlab__sync', async () => mcpSuccess('body returned before cancellation'));
    assert.equal((await execute(ctx, 'mcp__matlab__sync', abort.signal)).isError, true);
    assert.equal((await state(directory)).quarantine.reason, 'mcp_cancelled_state_unknown');
  } finally { await ctx.fiber.dispose(); }
}));

test('a failed fixed terminal verifier quarantines even after a successful MCP body', async () => fixture(async directory => {
  const { ctx, register } = await context(directory, { verifyTerminal: async () => { throw new Error('Dedicated session identity cannot be verified'); } });
  try {
    register('mcp__matlab__sync', async () => mcpSuccess('body returned'));
    assert.equal((await execute(ctx, 'mcp__matlab__sync')).isError, true);
    assert.equal((await state(directory)).quarantine.reason, 'mcp_dispatch_or_wrapper_failed_state_unknown');
  } finally { await ctx.fiber.dispose(); }
}));

for (const boundary of ['transport', 'mcp-isError', 'timeout', 'cancel']) {
  test(`${boundary} settlement quarantines while remote-work termination remains unverified`, async () => fixture(async directory => {
    const abort = new AbortController();
    let remoteWorkActive = false, calls = 0, admitted;
    const started = new Promise(resolveStarted => { admitted = resolveStarted; });
    const { ctx, register } = await context(directory, { verifyTerminal: async () => assert.fail('Error outcomes must not invoke the terminal-success verifier') });
    try {
      register('mcp__matlab__failure', async (_args, exec) => {
        calls += 1;
        remoteWorkActive = true;  // Deliberately outlives SDK settlement in this fixture.
        admitted();
        if (boundary === 'mcp-isError') return { isError: true, content: [{ type: 'text', text: 'MATLAB reported an error' }] };
        if (boundary === 'timeout') await new Promise((_resolve, reject) => setTimeout(() => reject(new Error('SDK request deadline elapsed')), 20));
        if (boundary === 'cancel') await new Promise((_resolve, reject) => exec.signal.addEventListener('abort', () => reject(new Error('SDK cancelled request')), { once: true }));
        throw new Error('MCP transport disconnected');
      });
      const pending = execute(ctx, 'mcp__matlab__failure', abort.signal);
      await started;
      if (boundary === 'cancel') abort.abort(new Error('User cancelled while MATLAB may still run'));
      assert.equal((await pending).isError, true);
      assert.equal(remoteWorkActive, true, 'Promise settlement is not a computation-stop certificate');
      assert.ok((await state(directory)).quarantine);
      assert.equal((await execute(ctx, 'mcp__matlab__failure')).isError, true);
      assert.equal(calls, 1, 'quarantine is shared, not a per-call/Session lock');
    } finally { remoteWorkActive = false; await ctx.fiber.dispose(); }
  }));
}

/** Small external actors. They perform only resource admission, never MATLAB work. */
const pythonActor = String.raw`
import json, os, sys
from control_harness.matlab_resource import MatlabResource
def event(name, **data): print(json.dumps({'event': name, 'pid': os.getpid(), **data}), flush=True)
with MatlabResource(sys.argv[1], .01).acquire(sys.argv[2], on_queued=lambda item: event('queued', ticket=item['ticket'])) as lease:
    lease.mark_running()
    event('acquired', ticket=lease.owner['ticket'])
    sys.stdin.readline()
    lease.complete({'kind':'synchronous_adapter_return','producer':'cross-process-python-fixture'})
    event('completed')
`;

const nodeActor = `
import { createInterface } from 'node:readline';
import { Context } from ${JSON.stringify(builtUrl('vendor/cordis/lib/index.js'))};
import Local from ${JSON.stringify(builtUrl('packages/subprocess/subprocess-local/lib/index.js'))};
import { MatlabResource } from ${JSON.stringify(pathToFileURL(resolve(project, 'harness-plugin/src/matlab-resource.mjs')).href)};
const directory = process.argv[1];
const abort = new AbortController();
const event = (event, data={}) => console.log(JSON.stringify({event,pid:process.pid,...data}));
let release;
const released = new Promise(resolve => { release = resolve; });
const input = createInterface({input:process.stdin});
input.on('line', line => line === 'cancel' ? abort.abort(new Error('queued actor cancelled')) : release());
const ctx = new Context();
try {
  await ctx.plugin(Local);
  const resource = new MatlabResource({subprocess:ctx.subprocess,pythonPath:${JSON.stringify(python)},codeRoot:${JSON.stringify(project)},config:{directory,pollIntervalMs:10}});
  const lease = await resource.acquire('cross-process-node-fixture',abort.signal,item=>event('queued',{ticket:item.ticket}));
  await lease.start();
  event('acquired',{ticket:lease.ticket});
  await released;
  await lease.complete(${JSON.stringify(proof)});
  event('completed');
} catch(error) {
  if(abort.signal.aborted) event('cancelled'); else { console.error(error.stack); process.exitCode=1; }
} finally { input.close(); await ctx.fiber.dispose(); }
`;

function actor(language, directory, label) {
  const child = spawn(language === 'python' ? python : process.execPath,
    language === 'python' ? ['-u', '-c', pythonActor, directory, label] : ['--input-type=module', '-e', nodeActor, directory],
    { cwd: project, env: { ...process.env, PYTHONPATH: resolve(project, 'src') }, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  const events = [], waiters = new Set();
  let pending = '', stderr = '', exited = false;
  const flush = () => {
    for (const waiter of waiters) {
      const event = events.find(item => item.event === waiter.name);
      if (event) { waiters.delete(waiter); clearTimeout(waiter.timer); waiter.resolve(event); }
      else if (exited) { waiters.delete(waiter); clearTimeout(waiter.timer); waiter.reject(new Error(`Actor exited before ${waiter.name}: ${stderr}`)); }
    }
  };
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', chunk => { stderr += chunk; });
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', chunk => {
    pending += chunk;
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, newline).trim(); pending = pending.slice(newline + 1);
      if (line) events.push(JSON.parse(line));
    }
    flush();
  });
  const done = new Promise((resolveDone, reject) => { child.once('error', reject); child.once('close', code => { exited = true; flush(); resolveDone({ code, stderr }); }); });
  return { child, events, done, wait(name) {
    return new Promise((resolveWait, reject) => {
      const waiter = { name, resolve: resolveWait, reject };
      waiter.timer = setTimeout(() => { waiters.delete(waiter); reject(new Error(`Timed out waiting for ${language} actor ${name}: ${stderr}`)); }, 15000);
      waiters.add(waiter); flush();
    });
  }, send(line) { child.stdin.write(`${line}\n`); } };
}

test('independent Node Hosts and Python consumers share FIFO exclusion; queued cancellation removes only itself', { timeout: 30000 }, async () => fixture(async directory => {
  const actors = [];
  const start = (language, label) => { const value = actor(language, directory, label); actors.push(value); return value; };
  try {
    const holder = start('python', 'python-first-owner');
    const first = await holder.wait('acquired');
    const node = start('node');
    const second = await node.wait('queued');
    const pythonNext = start('python', 'python-next');
    const third = await pythonNext.wait('queued');
    const cancelled = start('node');
    const fourth = await cancelled.wait('queued');
    assert.ok(first.ticket < second.ticket && second.ticket < third.ticket && third.ticket < fourth.ticket);
    cancelled.send('cancel');
    await cancelled.wait('cancelled');
    assert.equal((await cancelled.done).code, 0);
    const queued = await state(directory);
    assert.deepEqual(queued.queue.map(item => item.ticket), [second.ticket, third.ticket]);
    assert.equal(node.events.some(item => item.event === 'acquired'), false);
    assert.equal(pythonNext.events.some(item => item.event === 'acquired'), false);
    holder.send('release');
    await holder.wait('completed');
    const nodeOwned = await node.wait('acquired');
    assert.equal(nodeOwned.ticket, second.ticket);
    assert.equal(pythonNext.events.some(item => item.event === 'acquired'), false, 'Python cannot overlap the Node Host lease');
    node.send('release');
    await node.wait('completed');
    const pythonOwned = await pythonNext.wait('acquired');
    assert.equal(pythonOwned.ticket, third.ticket);
    assert.equal(new Set([first.pid, nodeOwned.pid, pythonOwned.pid, fourth.pid]).size, 4, 'Actual independent processes took part');
    pythonNext.send('release');
    await pythonNext.wait('completed');
    for (const item of actors) assert.equal((await item.done).code, 0);
    const final = await state(directory);
    assert.equal(final.owner, null);
    assert.equal(final.quarantine, null);
    assert.deepEqual(final.queue, []);
  } finally {
    for (const item of actors) { if (item.child.exitCode === null) item.child.kill(); await item.done; }
  }
}));
