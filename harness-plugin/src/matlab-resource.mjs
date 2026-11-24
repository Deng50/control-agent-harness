/** Shared MATLAB admission; the per-call Python broker owns the same OS lock as batch/Python. */
import { isAbsolute, resolve } from 'node:path';
import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { parseMatlabTerminalProbe, terminalProbeCall } from './matlab-terminal.mjs';
import { calculationCall, calculationScope } from './matlab-calculation.mjs';
import { checkInteractiveModelEdits } from './simulink-save-routing.mjs';
import { withMcpReplyTracking, forgetMcpReply } from './matlab-mcp-reply.mjs';
import { partialToolkitValue } from './matlab-presentation.mjs';

// Only trusted Host code obtains a lease. No model arguments or public tool
// names can create this re-entry capability for the fixed terminal probe.
const internalProbe = new AsyncLocalStorage();
const scopedDispatch = new AsyncLocalStorage();
const controlledModelDispatch = new AsyncLocalStorage();
const calculationDispatch = new AsyncLocalStorage();
const recoveryProbe = new AsyncLocalStorage();
const quote = text => "'" + String(text).replaceAll("'", "''") + "'";

export function isControlledModelDispatch(exec) {
  const operation=controlledModelDispatch.getStore();
  if (!operation || operation.agent!==exec.agent || exec.name!==operation.call.name) return false;
  const probe=internalProbe.getStore();
  return probe ? !probe.lease.closed
    : exec.callId===operation.call.id && JSON.stringify(exec.arguments)===operation.call.arguments;
}

// A public tool can supply model data. Only a Host helper can supply this
// per-call receipt parser. It runs while the broker still holds the OS lock.
export async function dispatchControlledModelOperation(tools, agent, signal, call, inspect) {
  const invocation = {agent, call, inspect};
  try {
    const batch = await controlledModelDispatch.run(invocation, () => tools.execute({agent, signal, calls:[call]}));
    const result = batch.results[0]?.result;
    if (!result || result.isError) throw new MatlabResourceError(result?.error?.message ?? '未收到原生 MATLAB 调用结果。');
    if (!invocation.receipt) throw new MatlabResourceError('缺少专用 MATLAB 资源核验记录。');
    return invocation.receipt;
  } catch (error) {
    throw Object.assign(new MatlabResourceError(error.message),
      {modelReceipt:invocation.observed, quarantined:invocation.quarantined === true,
        sessionVerified:invocation.sessionVerified === true, dispatched:invocation.dispatched === true});
  }
}

export const quarantineRecovery = 'MATLAB 等待终态核验。可继续检查工程文件和诊断记录。Host 会核对原调用完成标记、实例身份与空闲状态；若仍无法核实，保留现有模型并查看专用实例状态。';

/** Only this Host-generated expression can carry the bound pair of in-process probes. */
export async function dispatchMatlabTaskBatch({ctx,agent,grant,signal,manifestPath,manifestSha,nonce,helperRoot,onDispatch}) {
  const {scope} = grant, before = randomUUID(), after = randomUUID();
  const helper = resolve(helperRoot,'control_harness_task_batch.m'), callId = 'matlab-task-'+nonce;
  const probeFile = resolve(helperRoot,'control_harness_terminal_probe.m');
  const identity = {session_id:scope.dedicated_session.session_id,matlab_pid:scope.dedicated_session.pid};
  const args = [scope.helper,manifestPath,manifestSha,nonce,identity.session_id,before,after];
  // The MCP working directory is the installed helper directory. No source from
  // the caller's workspace can shadow the code that proves process identity.
  const code = "clear('control_harness_task_batch'); assert(strcmp(which('control_harness_task_batch'),"+quote(helper)+")); "
    +"control_harness_task_batch("+args.map(quote).join(',')+","+identity.matlab_pid+");";
  const params = {code,project_path:helperRoot};
  const invocation = {agent,callId,params,grant,onDispatch,nonce,verified:false,
    verify(result,session) {
      for (const token of [before,after]) {
        const proof = parseMatlabTerminalProbe(result.value,{nonce:token,identity,session,helperFile:probeFile});
        if (token === after) return proof;
      }
    }};
  const batch = await scopedDispatch.run(invocation,() => ctx.tools.execute({agent,signal,calls:[{type:'tool_call',
    id:callId,name:scope.native_tool,arguments:JSON.stringify(params)}]}));
  if (!batch.results[0]?.result?.isError && !invocation.verified) throw new MatlabResourceError('MATLAB 专用资源门禁未返回终态证明。');
  return batch;
}

export class MatlabResourceError extends Error {
  constructor(message) { super(message); this.name = 'MatlabResourceError'; }
}

function validate(config) {
  if (!isAbsolute(config.directory)) throw new Error('matlabResource.directory must be one absolute shared Host directory');
  const prefixes = config.toolPrefixes ?? ['mcp__matlab__'];
  if (!prefixes.length || prefixes.some(prefix => !/^mcp__[A-Za-z0-9_-]+__$/.test(prefix))) {
    throw new Error('matlabResource.toolPrefixes must contain explicit native MCP server prefixes');
  }
  const pollIntervalMs = config.pollIntervalMs ?? 50;
  if (!Number.isFinite(pollIntervalMs) || pollIntervalMs < 5 || pollIntervalMs > 5000) {
    throw new Error('matlabResource.pollIntervalMs must be between 5 and 5000');
  }
  return { directory: resolve(config.directory), toolPrefixes: [...new Set(prefixes)], pollIntervalMs };
}

/** A tiny JSON-lines client; process completion includes native process-range cleanup. */
export class Broker {
  constructor(handle) {
    this.handle = handle;
    this.events = [];
    this.waiters = new Set();
    this.failure = undefined;
    this.exited = false;
    let pending = '';
    const fail = error => { this.failure ??= error; this.flush(); };
    handle.stdout.setEncoding('utf8');
    handle.stdout.on('data', chunk => {
      pending += chunk;
      if (pending.length > 65536) {
        fail(new MatlabResourceError('MATLAB resource broker output exceeded 64 KiB'));
        handle.terminate();
        return;
      }
      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline).trim();
        pending = pending.slice(newline + 1);
        if (!line) continue;
        try {
          const event = JSON.parse(line);
          if (event.event === 'error') fail(Object.assign(new MatlabResourceError(`${event.error_type}: ${event.message}`), {brokerError:event.error_type}));
          else this.events.push(event);
        } catch (error) { fail(error); }
        this.flush();
      }
    });
    handle.stdout.on('error', fail);
    handle.stdin.on('error', fail);
    this.done = (async () => {
      try {
        const result = await handle.done;
        await handle.waitForExit();
        if (result.exitCode !== 0) {
          fail(new MatlabResourceError(`MATLAB resource broker exited ${result.exitCode}: ${handle.collected?.stderr?.readFrom(0).text ?? ''}`));
        }
      } catch (error) { fail(error); }
      finally { this.exited = true; this.flush(); }
    })();
  }

  flush() {
    for (const waiter of this.waiters) {
      const index = this.events.findIndex(event => waiter.names.includes(event.event));
      if (index >= 0) {
        this.waiters.delete(waiter);
        waiter.resolve(this.events.splice(index, 1)[0]);
      } else if (this.failure || this.exited) {
        this.waiters.delete(waiter);
        waiter.reject(this.failure ?? new MatlabResourceError('MATLAB resource broker exited without the expected acknowledgement'));
      }
    }
  }

  wait(...names) {
    return new Promise((resolveWait, reject) => {
      this.waiters.add({ names, resolve: resolveWait, reject });
      this.flush();
    });
  }

  send(command) {
    if (this.exited || this.failure) throw this.failure ?? new MatlabResourceError('MATLAB resource broker is closed');
    this.handle.stdin.write(`${JSON.stringify(command)}\n`);
  }

  async finish(command, ...events) {
    this.send(command);
    const event = await this.wait(...events);
    this.handle.stdin.end();
    await this.done;
    if (this.failure) throw this.failure;
    return event;
  }
}

export class MatlabResource {
  constructor({ subprocess, pythonPath, codeRoot, config, logger }) {
    if (!subprocess?.spawn || !isAbsolute(pythonPath) || !isAbsolute(codeRoot)) {
      throw new Error('MATLAB admission needs the native subprocess service and absolute Python/code paths');
    }
    this.subprocess = subprocess;
    this.pythonPath = pythonPath;
    this.codeRoot = codeRoot;
    this.config = validate(config);
    this.logger = logger;
  }

  matches(name) { return this.config.toolPrefixes.some(prefix => name.startsWith(prefix)); }

  withInternalProbe(lease, callback) {
    if (lease.resource !== this || lease.closed) throw new MatlabResourceError('Terminal probe requires this resource\'s live lease');
    return internalProbe.run({ resource: this, lease }, callback);
  }

  withRecoveryProbe(probeNonce, callback) {
    return recoveryProbe.run({resource:this,call:terminalProbeCall(this.codeRoot,probeNonce)},callback);
  }

  spawnBroker() {
    return new Broker(this.subprocess.spawn({
      argv: [this.pythonPath, '-m', 'control_harness.matlab_resource', 'broker',
        '--directory', this.config.directory, '--poll-ms', String(this.config.pollIntervalMs)],
      cwd: this.codeRoot,
      env: { PYTHONPATH: resolve(this.codeRoot, 'src'), PYTHONDONTWRITEBYTECODE: '1', PYTHONUNBUFFERED: '1' },
      graceMs: 1000,
      stdio: { stdin: 'pipe', stdout: 'pipe', stderr: { maxBytes: 65536 } },
    }));
  }

  async acquire(label, signal, onQueued = () => {}) {
    signal.throwIfAborted();
    const broker = this.spawnBroker();
    const abortQueued = () => {
      try { broker.send({ command: 'cancel' }); } catch { /* Process exit is observed below. */ }
    };
    signal.addEventListener('abort', abortQueued, { once: true });
    try {
      broker.send({ command: 'acquire', label, ...(this.recoveryEnabled ? {wait_for_recovery:true} : {}) });
      const queued = await broker.wait('queued');
      onQueued(queued);
      const acquired = await broker.wait('acquired');
      const lease = new MatlabLease(this, broker, signal, acquired);
      if (signal.aborted) {
        await lease.cancelBeforeDispatch();
        signal.throwIfAborted();
      }
      return lease;
    } catch (error) {
      // EOF removes only a queued/reserved ticket. If dispatch ever started,
      // the Python broker persists quarantine instead of claiming MATLAB idle.
      broker.handle.stdin.end();
      await broker.done;
      if (signal.aborted) signal.throwIfAborted();
      throw error;
    } finally { signal.removeEventListener('abort', abortQueued); }
  }
}

class MatlabLease {
  constructor(resource, broker, signal, acquired) {
    this.resource = resource;
    this.broker = broker;
    this.signal = signal;
    this.token = acquired.token;
    this.ticket = acquired.ticket;
    this.session = acquired.session;
    this.started = false;
    this.closed = false;
    this.released = false;
    this.quarantined = false;
  }

  withInternalProbe(callback) { return this.resource.withInternalProbe(this, callback); }

  async start(requireLiveSession = false, completionNonce, completionSource) {
    this.signal.throwIfAborted();
    this.broker.send({ command: 'start',require_live_session:requireLiveSession,
      ...(completionNonce ? {completion_nonce:completionNonce} : {}), ...(completionSource ? {completion_source:completionSource} : {}) });
    const event = await this.broker.wait('started', 'cancelled', 'quarantined');
    if (event.event !== 'started') {
      this.closed = true;
      this.quarantined = event.event === 'quarantined';
      this.broker.handle.stdin.end();
      await this.broker.done;
      this.signal.throwIfAborted();
      if (event.event === 'quarantined') {
        throw Object.assign(new MatlabResourceError('专用 MATLAB 进程已退出、PID 已重用或创建身份无法验证。' + quarantineRecovery), {quarantined:true});
      }
      if (event.reason === 'bound_session_identity_changed') {
        throw new MatlabResourceError('专用 MATLAB 的进程身份已变化。本次调用尚未派发，请核对专用会话绑定。');
      }
      throw new MatlabResourceError('MATLAB broker cancelled before dispatch');
    }
    this.started = true;
  }

  async cancelBeforeDispatch() {
    if (this.closed) return;
    if (this.started) throw new MatlabResourceError('A dispatched MATLAB call cannot be released as queued cancellation');
    this.closed = true;
    const event = await this.broker.finish({ command: 'cancel' }, 'cancelled', 'quarantined');
    this.quarantined = event.event === 'quarantined';
  }

  async complete(evidence) {
    if (this.closed) throw new MatlabResourceError('MATLAB resource lease is closed');
    if (this.signal.aborted) {
      await this.quarantine('cancelled_before_completion_certificate');
      this.signal.throwIfAborted();
    }
    this.closed = true;
    const event = await this.broker.finish({ command: 'complete', evidence }, 'completed', 'quarantined');
    if (event.event !== 'completed') {
      this.quarantined = true;
      throw new MatlabResourceError(event.reason === 'bound_session_identity_changed'
        ? '专用 MATLAB 的进程身份复核失败，正在等待原实例的终态证据。'
        : 'MATLAB 终态与取消同时发生，正在核验原调用是否结束。');
    }
    this.released = true;
  }

  async confirmNotDispatched() {
    if (this.closed || !this.started) throw new MatlabResourceError('Expected a live started MATLAB lease');
    this.closed = true;
    await this.broker.finish({command:'not-dispatched',token:this.token},'not-dispatched');
    this.released = true;
  }

  async quarantine(reason) {
    if (this.closed) return;
    this.closed = true;
    await this.broker.finish({ command: 'quarantine', reason }, 'quarantined');
    this.quarantined = true;
  }
}

// These names only preserve UI hints for existing parameter/wiring clients.
function editCategory(code) {
  if (/\badd_line\s*\(/.test(code)) return 'wiring';
  if (/\bset_param\s*\(/.test(code)) return 'parameter';
}

const syncToolkitTools = new Set(['model_overview','model_read','model_query_params','model_resolve_params',
  'model_edit','model_read_diagnostics','model_check','model_scan','model_test']);
const synchronousNativeTools = new Set([...syncToolkitTools,'run_matlab_file','run_matlab_test_file','check_matlab_code','detect_matlab_toolboxes']);

// SATK writes its result as a top-level status field, sometimes inside a
// displayed MATLAB string. Do not search block names, diagnostic logs or
// arbitrary MATLAB output for words like "error".
function toolkitStatus(value) {
  const structured = value?.structuredContent;
  if (structured && typeof structured === 'object' && Object.hasOwn(structured,'status')) return structured.status;
  if (value?.content?.length !== 1 || value.content[0].type !== 'text') return;
  const output = value.content[0].text.trim();
  if (output.startsWith('{')) {
    try { return JSON.parse(output)?.status; } catch { return; }
  }
  const body = output.replace(/^\[Autolayout\] applied:[^\r\n]*\r?\n\s*/, '');
  const displayed = /^ans\s*=\s*"([\s\S]*)"$/.exec(body);
  return /^status:[ \t]*(error|partial)[ \t]*(?:\r?\n|$)/.exec(displayed ? displayed[1] : body)?.[1];
}

function presentToolkitResult(result,status) {
  const structured = result.value.structuredContent;
  const original = result.value.content.filter(item=>item.type==='text').map(item=>item.text).join('\n');
  const structuredOnly = !original && structured ? [{type:'text',text:JSON.stringify(structured)}] : [];
  const meta = {...result.meta,matlab_tool_status:status,matlab_session_verified:true,
    ...(structured ? {matlab_tool_result:structured} : {})};
  if (status === 'partial') return {...result,meta,value:partialToolkitValue(result.value,structuredOnly,meta)};
  return {isError:true,error:{message:original || JSON.stringify(structured)},meta,
    content:[...result.content,...structuredOnly,{type:'text',text:'Simulink 工具返回错误。专用 MATLAB 已核验空闲，可读回现场并修正失败步骤。'}]};
}

// 错误结果仍原样返回；记录的是专用资源可以承接下一次独立调用，而非本次任务成功。
function sameVerifiedSession(before, after, session) {
  if (!session?.identity || !session.session_id || !Number.isSafeInteger(session.pid)) return false;
  return [before, after].every(probe => probe?.kind === 'synchronous_adapter_return'
    && probe.producer === 'control_harness_terminal_probe-v1' && probe.session_id === session.session_id
    && probe.matlab_pid === session.pid && /^[a-f0-9-]{36}$/i.test(probe.nonce ?? ''))
    && before.nonce !== after.nonce;
}

async function verifyFailedCall({exec,result,before,verifyTerminal,resource,lease,completionNonce,scope}) {
  try {
    const after = await resource.withInternalProbe(lease,()=>verifyTerminal({exec,result,lease,completionNonce}));
    const identityProof = scope
      ? after?.producer === 'control_harness_terminal_probe-v1' && after.session_id === lease.session?.session_id
        && after.matlab_pid === lease.session.pid
      : sameVerifiedSession(before,after,lease.session);
    if (!identityProof || (completionNonce && after.completion_nonce !== completionNonce)) {
      throw new Error('专用 MATLAB 未确认本次调用已结束。');
    }
    const category = editCategory(exec.arguments.code);
    const outputLost = after.completion_outcome === 'returned';
    const outcome = outputLost ? 'returned_without_tool_result' : category ? `synchronous_${category}_error` : 'synchronous_matlab_error';
    await lease.complete({...after,call_outcome:outcome});
    const note = outputLost
      ? 'MATLAB 中的代码已经返回，但原生工具没有交付完整结果。请核实模型与磁盘状态。'
      : '专用 MATLAB 身份与空闲状态已核实。请读回实际端口和已有连线、参数及文件，只修正失败步骤。';
    return {...result,meta:{...result.meta,matlab_session_verified:true,
      matlab_error:{identifier:after.completion_identifier ?? '',outcome,partial_changes_possible:true},
      ...(category?{[`matlab_${category}_retryable`]:!outputLost}:{})},content:[...result.content,{type:'text',text:note}]};
  } catch (error) {
    if (!lease.closed) await lease.quarantine('mcp_error_awaiting_terminal_proof');
    return {...result,meta:{...result.meta,matlab_session_verified:false},
      content:[...result.content,{type:'text',text:error.message+' '+quarantineRecovery}]};
  }
}

async function runNestedEvaluation(ctx, exec, resource, lease, call, completionNonce) {
  const nested = {call,resource,lease,agent:exec.agent,completionNonce,dispatched:false};
  const releaseUnsent = () => lease.started ? lease.confirmNotDispatched() : lease.cancelBeforeDispatch();
  let batch;
  try {
    batch = await calculationDispatch.run(nested, () => ctx.tools.execute({
      agent:exec.agent, signal:exec.signal, parent:exec.token, calls:[call],
    }));
  } catch (error) {
    if (!nested.dispatched && !lease.closed) await releaseUnsent();
    throw error;
  }
  const result = batch.results[0]?.result;
  if (result?.isError && (!nested.dispatched || result.error?.info?.code === 'ABORTED_BEFORE_DISPATCH')) {
    if (!lease.closed) await releaseUnsent();
    return {denied:true,result};
  }
  return {result};
}

async function executeCalculation(ctx, exec, scope, resource, lease, before, verifyTerminal, completionNonce) {
  const {call,parse} = calculationCall(exec.arguments, scope, resource.codeRoot, completionNonce);
  const {result,denied} = await runNestedEvaluation(ctx,exec,resource,lease,call,completionNonce);
  if (denied) return result;
  if (result?.isError || exec.signal.aborted) {
    await lease.quarantine(exec.signal.aborted ? 'mcp_cancelled_state_unknown' : 'mcp_failed_state_unknown');
    throw new MatlabResourceError(result?.error?.message ?? 'MATLAB 计算中断。');
  }
  const {receipt,content} = parse(result,lease.session);
  const {nonce,helper,certifiable,...diagnostic} = receipt;
  try {
    if (!certifiable) throw new MatlabResourceError('计算涉及被遮蔽的函数或工作区对象，无法核实其副作用。');
    exec.signal.throwIfAborted();
    const after = await resource.withInternalProbe(lease, () => verifyTerminal({exec,result,lease}));
    if (!sameVerifiedSession(before,after,lease.session)) throw new MatlabResourceError('计算后的专用会话身份或空闲状态无法核实。');
    await lease.complete({...after,call_outcome:receipt.status === 'failed' ? 'synchronous_calculation_error' : 'synchronous_calculation_success'});
  } catch (error) {
    if (!lease.closed) await lease.quarantine(exec.signal.aborted ? 'mcp_cancelled_state_unknown' : 'mcp_calculation_terminal_unverified');
    const message = [receipt.message,error.message,quarantineRecovery].filter(Boolean).join('\n');
    return {isError:true,error:{message},content:[...content,{type:'text',text:message}],
      meta:{...result.meta,matlab_calculation:{...diagnostic,stage:'terminal_verification',session_verified:false,
        retryable:false,verification_error:error.message}}};
  }
  const meta = {...result.meta,matlab_session_verified:true,matlab_calculation:{...diagnostic,stage:'evaluate',session_verified:true,retryable:receipt.status==='failed'}};
  if (receipt.status === 'failed') {
    const message = receipt.message;
    return {isError:true,error:{message},meta,content:[...content,{type:'text',text:
      `${receipt.identifier}: ${message}\n专用 MATLAB 会话身份和空闲状态已核验。已执行的赋值可能保留，可修正计算后重试。`}]};
  }
  return {...result,meta,content,value:{...result.value,content}};
}

// Run the originally authorised text exactly once in the MCP base workspace.
// A fresh Host-owned function frame records actual unwind, including MATLAB
// exceptions. SDK timeout/transport strings cannot manufacture this evidence.
function recordedEvaluation(exec, codeRoot, nonce) {
  const file = resolve(codeRoot,'matlab/control_harness_evaluate.m');
  const source = '[' + exec.arguments.code.split(/\r\n?|\n/).map(quote).join(' char(10) ') + ']';
  const code = `assert(strcmpi(which('control_harness_evaluate'),${quote(file)})); control_harness_evaluate(${source},${quote(nonce)});`;
  return {type:'tool_call',id:'matlab-eval-'+nonce,name:exec.name,arguments:JSON.stringify({...exec.arguments,code})};
}

function cancelledBeforeDispatch() {
  const message = 'MATLAB call cancelled before dispatch';
  return {isError:true,error:{message,info:{name:'AbortError',code:'ABORTED_BEFORE_DISPATCH'}},content:[{type:'text',text:message}]};
}

/** Native permissions and the shared broker lease cover the original call and its terminal probe. */
export function installMatlabResource(ctx, config, { verifyBeforeDispatch, verifyTerminal } = {}) {
  if (config.matlabResource === undefined) return undefined;
  const resource = new MatlabResource({ subprocess: ctx.subprocess, pythonPath: config.pythonPath,
    codeRoot: config.codeRoot, config: config.matlabResource, logger: ctx.logger });
  ctx.on('tools/execute', async (exec, next) => {
    if (!resource.matches(exec.name)) return next();
    const recovery = recoveryProbe.getStore();
    if (recovery?.resource === resource && exec.callId === recovery.call.id && exec.name === recovery.call.name
        && JSON.stringify(exec.arguments) === recovery.call.arguments) return next();
    const probe = internalProbe.getStore();
    if (probe?.resource === resource && !probe.lease.closed) return next();
    const calculation = calculationDispatch.getStore();
    if (calculation?.resource === resource && calculation.agent === exec.agent &&
        !calculation.lease.closed && !calculation.lease.started && exec.callId === calculation.call.id &&
        exec.name === calculation.call.name && JSON.stringify(exec.arguments) === calculation.call.arguments) {
      // Both outer and nested permission decisions have settled. Until here,
      // cancellation or refusal only removes a reservation.
      await calculation.lease.start(true,calculation.completionNonce);
      if (exec.signal.aborted) {
        await calculation.lease.confirmNotDispatched();
        return cancelledBeforeDispatch();
      }
      calculation.dispatched = true;
      return next();
    }
    const scoped = scopedDispatch.getStore();
    const controlled = controlledModelDispatch.getStore();
    if (!scoped) checkInteractiveModelEdits(exec);
    let lease;
    try { lease = await resource.acquire(`${exec.name}:${exec.callId}`, exec.signal); }
    catch (error) {
      if (exec.signal.aborted) return cancelledBeforeDispatch();
      if (error.brokerError === 'MatlabResourceQuarantined') {
        if (controlled) controlled.quarantined = true;
        throw new MatlabResourceError(error.message + '\n' + quarantineRecovery);
      }
      throw error;
    }
    let result, trackReply;
    try {
      if (scoped) {
        if (scoped.callId !== exec.callId || scoped.agent !== exec.agent || exec.name !== scoped.grant.scope.native_tool
            || exec.arguments?.code !== scoped.params.code || exec.arguments?.project_path !== scoped.params.project_path
            || Object.keys(exec.arguments).length !== 2) throw new MatlabResourceError('MATLAB 批处理调用与 Host 已授权参数不匹配。');
        // A queued call can outlive the reviewed binary, session or artifact.
        // A failed check here releases only the reservation and never calls MATLAB.
        await scoped.grant.validate(lease.session);
        await lease.start(true,scoped.nonce);
        if (exec.signal.aborted) {
          await lease.confirmNotDispatched();
          exec.signal.throwIfAborted();
        }
        scoped.onDispatch();
        result = await next();
        if (result.error?.info?.code === 'ABORTED_BEFORE_DISPATCH') {
          await lease.confirmNotDispatched();
          return result;
        }
        if (result.isError && !exec.signal.aborted && verifyTerminal) {
          return await verifyFailedCall({exec,result,verifyTerminal,resource,lease,completionNonce:scoped.nonce,scope:true});
        }
        if (result.isError || exec.signal.aborted) {
          await lease.quarantine('scoped_mcp_failed_or_cancelled_state_unknown');
        } else {
          const terminal = scoped.verify(result,lease.session);
          await lease.complete(terminal);
          scoped.verified = true;
        }
        return result;
      }
      const before = verifyBeforeDispatch
        ? await resource.withInternalProbe(lease, () => verifyBeforeDispatch({ exec, lease })) : undefined;
      if (before?.kind !== 'synchronous_adapter_return' || typeof before.producer !== 'string' || !before.producer.trim()) {
        throw new MatlabResourceError('Dedicated MATLAB session identity and idle state could not be verified before dispatch');
      }
      exec.signal.throwIfAborted();
      if (controlled && (controlled.agent !== exec.agent || controlled.call.name !== exec.name ||
          controlled.call.id !== exec.callId || controlled.call.arguments !== JSON.stringify(exec.arguments))) {
        throw new MatlabResourceError('本次模型工具与 Host 发起的原生调用不一致。');
      }
      const isEval = exec.name === 'mcp__matlab__evaluate_matlab_code' && typeof exec.arguments?.code === 'string';
      const scope = !controlled && isEval && lease.session && verifyTerminal && calculationScope(exec.arguments.code);
      const record = !controlled && !scope && isEval && lease.session && verifyTerminal;
      const nativeName = exec.name.startsWith('mcp__matlab__') ? exec.name.slice('mcp__matlab__'.length) : undefined;
      trackReply = lease.session && verifyTerminal && synchronousNativeTools.has(nativeName);
      const completionNonce = controlled ? controlled.call.id.replace(/^model-/, '') : scope || record || trackReply ? randomUUID() : undefined;
      if (scope) return await executeCalculation(ctx,exec,scope,resource,lease,before,verifyTerminal,completionNonce);
      if (record) {
        const nested = await runNestedEvaluation(ctx,exec,resource,lease,recordedEvaluation(exec,resource.codeRoot,completionNonce),completionNonce);
        if (nested.denied) return nested.result;
        result = nested.result;
      } else {
        await lease.start(Boolean(lease.session) || Boolean(controlled),completionNonce,trackReply?'mcp_response':undefined);
        if (exec.signal.aborted) {
          await lease.confirmNotDispatched();
          return cancelledBeforeDispatch();
        }
        if (controlled) controlled.dispatched = true;
        result = trackReply ? await withMcpReplyTracking(resource,lease,exec,completionNonce,next) : await next();
        if (result.error?.info?.code === 'ABORTED_BEFORE_DISPATCH') {
          await lease.confirmNotDispatched();
          return result;
        }
      }
      if (controlled) {
        if (exec.signal.aborted) throw new MatlabResourceError('执行中的 MATLAB 操作被取消。');
        let receipt;
        if (!result.isError) {
          try {
            receipt = controlled.inspect(result.value);
            if (receipt.matlab_pid !== lease.session?.pid) throw new Error('建模回执的 PID 与绑定进程不符。');
          } catch (error) {
            result = {isError:true,error:{message:error.message},content:[{type:'text',text:error.message}]};
          }
        }
        if (result.isError) {
          const failure = await verifyFailedCall({exec,result,before,verifyTerminal,resource,lease,completionNonce});
          controlled.sessionVerified = failure.meta.matlab_session_verified;
          controlled.quarantined = !controlled.sessionVerified;
          return failure;
        }
        controlled.observed = receipt;
        const after = verifyTerminal && await resource.withInternalProbe(lease, () => verifyTerminal({exec, result, lease}));
        if (!sameVerifiedSession(before, after, lease.session)) throw new Error('前后两次专用会话身份和空闲核验不一致。');
        await lease.complete({...after, call_outcome: receipt.status === 'failed' ? 'synchronous_model_error' : 'synchronous_model_success'});
        controlled.receipt = {...receipt, retryable:receipt.status === 'failed', session_verified:true};
        return result;
      }
      const toolkit = syncToolkitTools.has(nativeName);
      const repliedFromMatlab = result.error?.info?.code === 'MCP_TOOL_RETURNED_ERROR'
        && result.error.message.startsWith('matlab error:') && trackReply;
      if (result.isError && !exec.signal.aborted && verifyTerminal && (record || repliedFromMatlab)) {
        return await verifyFailedCall({exec,result,before,verifyTerminal,resource,lease,completionNonce:record?completionNonce:undefined});
      }
      const proof = !result.isError && !exec.signal.aborted && verifyTerminal
        ? await resource.withInternalProbe(lease, () => verifyTerminal({ exec, result, lease })) : undefined;
      if (proof && !exec.signal.aborted) {
        const status = toolkit && toolkitStatus(result.value);
        if (status === 'partial' || status === 'error') {
          await lease.complete({...proof,call_outcome:'synchronous_tool_'+status});
          return presentToolkitResult(result,status);
        }
        await lease.complete(proof);
      }
      else {
        await lease.quarantine(exec.signal.aborted ? 'mcp_cancelled_state_unknown'
          : result.isError ? 'mcp_failed_state_unknown' : 'mcp_success_without_trusted_terminal_proof');
        ctx.logger.warn(`MATLAB is awaiting the original completion evidence after ${exec.name}`);
        const diagnostic = result.error?.message ?? '缺少专用 MATLAB 空闲核验证据。';
        return {...result, isError:true, error:{...result.error, message:diagnostic},
          content:[...(result.content ?? []), {type:'text', text:quarantineRecovery}]};
      }
      return result;
    } catch (error) {
      if (!lease.closed) {
        if (lease.started) await lease.quarantine('mcp_dispatch_or_wrapper_failed_state_unknown');
        else await lease.cancelBeforeDispatch();
      }
      if (!lease.started && exec.signal.aborted) return cancelledBeforeDispatch();
      if (controlled && error.quarantined) controlled.quarantined = true;
      if (lease.started && lease.quarantined) {
        if (controlled) controlled.quarantined = true;
        throw new MatlabResourceError(error.message + '\n' + quarantineRecovery);
      }
      throw error;
    } finally {
      if (trackReply && lease.released) {
        await forgetMcpReply(resource.config.directory,lease.token).catch(error=>resource.logger?.warn?.(error.message));
      }
    }
  });
  return resource;
}
