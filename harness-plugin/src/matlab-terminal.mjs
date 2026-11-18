/** Fixed Host-owned identity/quiescence probe. Caller output never certifies idle. */
import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';

const TOOL = 'mcp__matlab__evaluate_matlab_code';
const normalized = value => resolve(value).replaceAll('\\', '/').toLowerCase();

export function parseMatlabTerminalProbe(value, { nonce, identity, session, helperFile }) {
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
  if (probe.idle !== true || !Array.isArray(probe.busy_models) || probe.busy_models.length
    || probe.background_futures !== 0 || probe.parallel_futures !== 0 || probe.running_timers !== 0) {
    throw new Error('Dedicated MATLAB still has managed computation; resource remains quarantined');
  }
  return { kind: 'synchronous_adapter_return', producer: 'control_harness_terminal_probe-v1',
    observed_at: new Date().toISOString(), session_id: probe.session_id,
    matlab_pid: probe.matlab_pid, nonce, probe_scope: probe.probe_scope };
}

export function createMatlabTerminalVerifier(ctx, { identityFile }) {
  if (!isAbsolute(identityFile)) throw new Error('MATLAB identityFile must be absolute');
  return async ({ exec, lease }) => {
    const identity = JSON.parse(await readFile(identityFile, 'utf8'));
    if (identity.schema_version !== 1 || identity.kind !== 'control-harness-dedicated-matlab-v1'
      || identity.status !== 'ready' || !Number.isSafeInteger(identity.matlab_pid)
      || !/^[a-f0-9-]{36}$/i.test(identity.session_id ?? '') || !isAbsolute(identity.project_root ?? '')) {
      throw new Error('Dedicated MATLAB ready marker is invalid');
    }
    const nonce = randomUUID();
    // A fixed local function and Host-generated UUID only. No caller code, cwd,
    // credentials, model arguments, or claimed idle flags enter this expression.
    const code = `fprintf('CONTROL_HARNESS_PROBE_${nonce}:%s\\n',jsonencode(control_harness_terminal_probe('${nonce}')));`;
    const batch = await ctx.tools.execute({ agent: exec.agent, signal: exec.signal,
      calls: [{ type: 'tool_call', id: `matlab-probe-${nonce}`, name: TOOL, arguments: JSON.stringify({ code }) }] });
    const result = batch.results[0]?.result;
    if (!result || result.isError) throw new Error('Fixed MATLAB probe was denied or failed');
    return parseMatlabTerminalProbe(result.value, { nonce, identity, session: lease.session,
      helperFile: resolve(identity.project_root, 'matlab/control_harness_terminal_probe.m') });
  };
}
