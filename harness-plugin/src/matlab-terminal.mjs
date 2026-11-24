/** Fixed Host-owned identity/quiescence probe. Caller output never certifies idle. */
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';

const TOOL = 'mcp__matlab__evaluate_matlab_code';
const normalized = value => resolve(value).replaceAll('\\', '/').toLowerCase();

export function parseMatlabTerminalProbe(value, { nonce, identity, session, helperFile, completionNonce }) {
  const prefix = `CONTROL_HARNESS_PROBE_${nonce}:`;
  const lines = (value?.content ?? []).filter(item => item.type === 'text')
    .flatMap(item => String(item.text).split(/\r?\n/)).filter(line => line.startsWith(prefix));
  if (lines.length !== 1) throw new Error('Fixed MATLAB probe did not return exactly one nonce certificate');
  const probe = JSON.parse(lines[0].slice(prefix.length));
  if (probe.schema_version !== 1 || probe.nonce !== nonce || probe.session_id !== identity.session_id
    || probe.matlab_pid !== identity.matlab_pid || session?.pid !== identity.matlab_pid
    || session?.session_id !== identity.session_id || !session?.identity
    || typeof probe.helper_file !== 'string' || normalized(`${probe.helper_file}.m`) !== normalized(helperFile)) {
    throw new Error('Dedicated MATLAB identity or fixed helper location does not match the bound resource');
  }
  if (!Array.isArray(probe.busy_models) || !probe.busy_models.every(name => typeof name === 'string') ||
      ![probe.running_timers, probe.background_futures, probe.parallel_futures].every(n => Number.isSafeInteger(n) && n >= 0)) {
    throw new Error('专用 MATLAB 空闲核验数据不完整。');
  }
  if (probe.idle !== true || probe.busy_models.length || probe.background_futures || probe.parallel_futures || probe.running_timers) {
    throw new Error(`专用 MATLAB 的空闲核验未通过。忙模型 ${JSON.stringify(probe.busy_models)}，`
      + `运行中计时器 ${probe.running_timers}，后台池任务 ${probe.background_futures}，并行池任务 ${probe.parallel_futures}。`);
  }
  if (completionNonce && probe.completion_nonce !== completionNonce) {
    throw new Error('尚未观察到原 MATLAB 调用的完成标记。');
  }
  return { kind: 'synchronous_adapter_return', producer: 'control_harness_terminal_probe-v1',
    observed_at: new Date().toISOString(), session_id: probe.session_id,
    matlab_pid: probe.matlab_pid, nonce, probe_scope: probe.probe_scope,
    ...(completionNonce ? {completion_nonce:completionNonce,
      ...(probe.completion_outcome === 'returned' || probe.completion_outcome === 'failed'
        ? {completion_outcome:probe.completion_outcome,completion_identifier:String(probe.completion_identifier ?? '')} : {})} : {}) };
}

export function terminalProbeCall(codeRoot, nonce) {
  if (!/^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i.test(nonce)) throw new Error('Invalid MATLAB probe UUID');
  const helperFile = resolve(codeRoot,'matlab/control_harness_terminal_probe.m');
  const code = `assert(strcmpi(which('control_harness_terminal_probe'),'${helperFile.replaceAll("'", "''")}')); `
    + `fprintf('CONTROL_HARNESS_PROBE_${nonce}:%s\\n',jsonencode(control_harness_terminal_probe('${nonce}')));`;
  return { type:'tool_call', id:`matlab-probe-${nonce}`, name:TOOL, arguments:JSON.stringify({code,project_path:resolve(codeRoot,'matlab')}) };
}

export function createMatlabTerminalVerifier(ctx, { identityFile, codeRoot }) {
  if (!isAbsolute(identityFile)) throw new Error('MATLAB identityFile must be absolute');
  if (typeof codeRoot !== 'string' || !isAbsolute(codeRoot)) throw new Error('MATLAB probe needs the Host installation directory');
  const helperFile = resolve(codeRoot, 'matlab/control_harness_terminal_probe.m');
  return async ({ exec, lease, completionNonce, probeNonce }) => {
    const identity = JSON.parse(await readFile(identityFile, 'utf8'));
    if (identity.schema_version !== 1 || identity.kind !== 'control-harness-dedicated-matlab-v1'
      || identity.status !== 'ready' || !Number.isSafeInteger(identity.matlab_pid)
      || !/^[a-f0-9-]{36}$/i.test(identity.session_id ?? '') || !isAbsolute(identity.project_root ?? '')
      || normalized(identity.project_root) !== normalized(codeRoot)) {
      throw new Error('Dedicated MATLAB ready marker is invalid');
    }
    const nonce = probeNonce ?? randomUUID();
    // A fixed local function and Host-generated UUID only. No caller code, cwd,
    // credentials, model arguments, or claimed idle flags enter this expression.
    // 在调用前核对安装路径。工作区同名函数不能充当身份及空闲证明的签发方。
    const batch = await ctx.tools.execute({ agent: exec.agent, signal: exec.signal,
      calls: [terminalProbeCall(codeRoot,nonce)] });
    const result = batch.results[0]?.result;
    if (!result || result.isError) throw new Error('Fixed MATLAB probe was denied or failed');
    return parseMatlabTerminalProbe(result.value, { nonce, identity, session: lease.session, helperFile, completionNonce });
  };
}
