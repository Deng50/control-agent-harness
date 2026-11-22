/** Shared MATLAB admission; the per-call Python broker owns the same OS lock as batch/Python. */
import { isAbsolute, resolve } from 'node:path';
import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { parseMatlabTerminalProbe } from './matlab-terminal.mjs';
import { checkInteractiveModelSave } from './simulink-save-routing.mjs';

// Only trusted Host code obtains a lease. No model arguments or public tool
// names can create this re-entry capability for the fixed terminal probe.
const internalProbe = new AsyncLocalStorage();
const scopedDispatch = new AsyncLocalStorage();
const quote = text => "'" + String(text).replaceAll("'", "''") + "'";

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
  const invocation = {agent,callId,params,grant,onDispatch,verified:false,
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
class Broker {
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
          if (event.event === 'error') fail(new MatlabResourceError(`${event.error_type}: ${event.message}`));
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
    if (lease.resource !== this || lease.closed || !lease.started) throw new MatlabResourceError('Terminal probe requires this resource\'s live running lease');
    return internalProbe.run({ resource: this, lease }, callback);
  }

  async acquire(label, signal, onQueued = () => {}) {
    signal.throwIfAborted();
    const broker = new Broker(this.subprocess.spawn({
      argv: [this.pythonPath, '-m', 'control_harness.matlab_resource', 'broker',
        '--directory', this.config.directory, '--poll-ms', String(this.config.pollIntervalMs)],
      cwd: this.codeRoot,
      env: { PYTHONPATH: resolve(this.codeRoot, 'src'), PYTHONDONTWRITEBYTECODE: '1', PYTHONUNBUFFERED: '1' },
      graceMs: 1000,
      stdio: { stdin: 'pipe', stdout: 'pipe', stderr: { maxBytes: 65536 } },
    }));
    const abortQueued = () => {
      try { broker.send({ command: 'cancel' }); } catch { /* Process exit is observed below. */ }
    };
    signal.addEventListener('abort', abortQueued, { once: true });
    try {
      broker.send({ command: 'acquire', label });
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
  }

  withInternalProbe(callback) { return this.resource.withInternalProbe(this, callback); }

  async start(requireLiveSession = false) {
    this.signal.throwIfAborted();
    this.broker.send({ command: 'start',require_live_session:requireLiveSession });
    const event = await this.broker.wait('started', 'cancelled');
    if (event.event !== 'started') {
      this.closed = true;
      this.broker.handle.stdin.end();
      await this.broker.done;
      this.signal.throwIfAborted();
      if (event.reason === 'bound_session_identity_changed') throw new MatlabResourceError('专用 MATLAB 进程已退出，PID 已重用或无法验证。');
      throw new MatlabResourceError('MATLAB broker cancelled before dispatch');
    }
    this.started = true;
  }

  async cancelBeforeDispatch() {
    if (this.closed) return;
    if (this.started) throw new MatlabResourceError('A dispatched MATLAB call cannot be released as queued cancellation');
    this.closed = true;
    await this.broker.finish({ command: 'cancel' }, 'cancelled', 'quarantined');
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
      throw new MatlabResourceError(event.reason === 'bound_session_identity_changed'
        ? '专用 MATLAB 的进程身份复核失败，资源已隔离。'
        : 'MATLAB 终态与取消同时发生，资源已隔离。');
    }
  }

  async quarantine(reason) {
    if (this.closed) return;
    this.closed = true;
    await this.broker.finish({ command: 'quarantine', reason }, 'quarantined');
  }
}

// 固定上游的 MCP bridge 把服务端工具错误压成文本。这里仅选择可进行终态复核的错误；
// 只有完整匹配的一行明确块参数诊断才触发探针。中文/英文参数名和块类型均不写死。
function synchronousParameterError(exec, result) {
  if (exec.name !== 'mcp__matlab__evaluate_matlab_code' || !result.isError || result.error?.info) return false;
  const diagnostic = result.error?.message?.trim() ?? '';
  if (diagnostic.length > 1024 || /[\r\n\u2028\u2029]/u.test(diagnostic)) return false;
  // R2024b 会把块路径显示成可点击的 MATLAB 链接。仅提取已知诊断链接，
  // 链接实参与可见块路径必须完全相同；不执行链接，也不对任意 HTML 去标签。
  const blockLink = /<a href="matlab:open_and_hilite_hyperlink[ \t]*\('([^'"<>]+)',[ \t]*'error'\)">\1<\/a>/gu;
  const plain = diagnostic.replace(blockLink, '$1');
  const missing = /^matlab error: ['"]?[\w /()-]+['"]? block (?:没有名为 ['"][\w.]+['"] 的参数|does not have a parameter named ['"][\w.]+['"])[.。]?$/iu;
  const invalidEnglish = /^matlab error: Invalid setting in '[^'<>]+' for parameter '[\w.]+'[.]?$/iu;
  const invalidChinese = /^matlab error: 参数 '[\w.]+' 的 '[^'<>]+' 中的设置无效[。.]?$/u;
  return missing.test(plain) || invalidEnglish.test(plain) || invalidChinese.test(plain);
}

// 错误结果仍原样返回；记录的是专用资源可以承接下一次独立调用，而非本次任务成功。
function recoveredParameterError(before, after, session) {
  if (!session?.identity || !session.session_id || !Number.isSafeInteger(session.pid)) return false;
  return [before, after].every(probe => probe?.kind === 'synchronous_adapter_return'
    && probe.producer === 'control_harness_terminal_probe-v1' && probe.session_id === session.session_id
    && probe.matlab_pid === session.pid && /^[a-f0-9-]{36}$/i.test(probe.nonce ?? ''))
    && before.nonce !== after.nonce;
}

async function finishParameterError({ exec, result, before, verifyTerminal, resource, lease }) {
  let retryable = false;
  try {
    // 留在同一租约内重新发出独立 nonce 的受信探针，避免其他任务在两次证明之间进入。
    const after = !exec.signal.aborted && verifyTerminal
      ? await resource.withInternalProbe(lease, () => verifyTerminal({exec, result, lease})) : undefined;
    if (!exec.signal.aborted && recoveredParameterError(before, after, lease.session)) {
      await lease.complete({...after, call_outcome:'synchronous_parameter_error'});
      retryable = true;
    }
  } catch (error) {
    // 保留触发本次核验的原始 MATLAB 错误；探针拒绝、忙或 broker 拒绝完成时不允许重试。
    resource.logger?.warn(`MATLAB parameter error could not be certified terminal: ${error.message}`);
  } finally {
    if (!lease.closed) {
      await lease.quarantine(exec.signal.aborted ? 'mcp_cancelled_state_unknown' : 'mcp_parameter_error_identity_or_idle_unverified');
    }
  }
  const text = retryable ? '已核验专用 MATLAB 会话身份和空闲状态，可修正参数后单独重试此操作。'
    : '专用 MATLAB 终态复核未完成。请检查隔离状态后恢复专用会话。';
  return {...result, content:[...result.content,{type:'text',text}],
    meta:{...result.meta, matlab_parameter_retryable:retryable}};
}

/** Native permissions and the shared broker lease cover the original call and its terminal probe. */
export function installMatlabResource(ctx, config, { verifyBeforeDispatch, verifyTerminal } = {}) {
  if (config.matlabResource === undefined) return undefined;
  const resource = new MatlabResource({ subprocess: ctx.subprocess, pythonPath: config.pythonPath,
    codeRoot: config.codeRoot, config: config.matlabResource, logger: ctx.logger });
  ctx.on('tools/execute', async (exec, next) => {
    if (!resource.matches(exec.name)) return next();
    const probe = internalProbe.getStore();
    if (probe?.resource === resource && !probe.lease.closed && probe.lease.started) return next();
    const scoped = scopedDispatch.getStore();
    if (!scoped) checkInteractiveModelSave(exec);
    let lease;
    try { lease = await resource.acquire(`${exec.name}:${exec.callId}`, exec.signal); }
    catch (error) {
      // Let the native body gate produce its canonical pre-dispatch abort.
      // It rechecks the original caller signal before invoking any tool body.
      if (exec.signal.aborted) return next();
      throw error;
    }
    let result;
    try {
      if (scoped) {
        if (scoped.callId !== exec.callId || scoped.agent !== exec.agent || exec.name !== scoped.grant.scope.native_tool
            || exec.arguments?.code !== scoped.params.code || exec.arguments?.project_path !== scoped.params.project_path
            || Object.keys(exec.arguments).length !== 2) throw new MatlabResourceError('MATLAB 批处理调用与 Host 已授权参数不匹配。');
        // A queued call can outlive the reviewed binary, session or artifact.
        // A failed check here releases only the reservation and never calls MATLAB.
        await scoped.grant.validate(lease.session);
        await lease.start(true);
        exec.signal.throwIfAborted();
        scoped.onDispatch();
        result = await next();
        if (result.isError || exec.signal.aborted) {
          await lease.quarantine('scoped_mcp_failed_or_cancelled_state_unknown');
        } else {
          const terminal = scoped.verify(result,lease.session);
          await lease.complete(terminal);
          scoped.verified = true;
        }
        return result;
      }
      await lease.start();
      const before = verifyBeforeDispatch
        ? await resource.withInternalProbe(lease, () => verifyBeforeDispatch({ exec, lease })) : undefined;
      if (before?.kind !== 'synchronous_adapter_return' || typeof before.producer !== 'string' || !before.producer.trim()) {
        await lease.quarantine('mcp_pre_dispatch_identity_or_idle_unverified');
        throw new MatlabResourceError('Dedicated MATLAB session identity and idle state could not be verified before dispatch');
      }
      exec.signal.throwIfAborted();
      result = await next();
      if (synchronousParameterError(exec, result)) {
        return await finishParameterError({exec, result, before, verifyTerminal, resource, lease});
      }
      const proof = !result.isError && !exec.signal.aborted && verifyTerminal
        ? await resource.withInternalProbe(lease, () => verifyTerminal({ exec, result, lease })) : undefined;
      if (proof && !exec.signal.aborted) await lease.complete(proof);
      else {
        await lease.quarantine(exec.signal.aborted ? 'mcp_cancelled_state_unknown'
          : result.isError ? 'mcp_failed_state_unknown' : 'mcp_success_without_trusted_terminal_proof');
        ctx.logger.warn(`MATLAB resource quarantined after ${exec.name}; dedicated session exit must be verified before reuse`);
      }
      return result;
    } catch (error) {
      if (!lease.closed) {
        if (lease.started) await lease.quarantine('mcp_dispatch_or_wrapper_failed_state_unknown');
        else await lease.cancelBeforeDispatch();
      }
      if (!lease.started && exec.signal.aborted) return next();
      throw error;
    }
  });
  return resource;
}
