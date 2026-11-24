import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';
import z from '@deepseek-ai/schemastery';
import { superviseWebBinding } from './matlab-web-rebind.mjs';

export const name = 'control-matlab-startup-check';
// controlEngineering 仅在业务工具、资源锁及终态探针安装完毕后由主插件发布。
export const inject = ['tools', 'controlEngineering', 'webServer', 'agents', 'jobs', 'appExit'];
export const Config = z.object({ identityFile: z.string().required() });

const evaluate = 'mcp__matlab__evaluate_matlab_code';
const readModel = 'mcp__matlab__model_read';

/** 在将要提供网页的同一个 Host 内调用其 MCP。探测只发送固定表达式和本次随机回执编号。 */
export async function checkStartupConnection(tools, identity, signal) {
  signal.throwIfAborted();
  const discoveryDeadline = Date.now() + 45000;
  while (!tools.get(evaluate) || !tools.get(readModel)) {
    signal.throwIfAborted();
    if (Date.now() >= discoveryDeadline) {
      throw new Error('MATLAB MCP 工具未就绪，请检查 MCP Server 与 Simulink Toolkit 的 tools/tools.json。');
    }
    await delay(150, undefined, { signal });
  }
  const nonce = randomUUID();
  const prefix = `CONTROL_MATLAB_READY_${nonce}:`;
  const code = `fprintf('${prefix}%s\\n',jsonencode(struct('release',version('-release'),`
    + "'matlab_pid',feature('getpid'),'session_id',getenv('CONTROL_HARNESS_SESSION_ID'),'sum',2+3)));";
  const batch = await tools.execute({
    signal, calls: [{ type: 'tool_call', id: nonce, name: evaluate, arguments: JSON.stringify({code}) }],
  });
  signal.throwIfAborted();
  const result = batch.results[0]?.result;
  if (!result || result.isError) {
    const reason = result?.error?.message ?? '';
    if (/quarantin|隔离/i.test(reason)) throw new Error('上次 MATLAB 调用的终态待核验，请检查专用会话的恢复记录。');
    throw new Error('实际 MATLAB 调用失败，请检查专用会话日志、资源状态和 MCP 工具权限。');
  }
  const records = (result.value?.content ?? []).filter(item => item.type === 'text')
    .flatMap(item => item.text.split(/\r?\n/)).filter(line => line.startsWith(prefix));
  if (records.length !== 1) throw new Error('实际 MATLAB 调用未返回本轮唯一的连通回执。');
  let data;
  try { data = JSON.parse(records[0].slice(prefix.length)); }
  catch { throw new Error('MATLAB 返回的连通回执不是有效 JSON。'); }
  if (!Number.isSafeInteger(data?.matlab_pid) || data.matlab_pid < 1 || data.matlab_pid !== identity.matlab_pid ||
      !/^[a-f\d-]{36}$/i.test(data.session_id) || data.session_id !== identity.session_id ||
      typeof data.release !== 'string' || !/^\d{4}[ab]$/.test(data.release) ||
      data.release !== identity.matlab_release || data.sum !== 5) {
    throw new Error('MATLAB 版本、会话、进程或 2+3 的实际结果与专用会话预期不一致。');
  }
  return { release: data.release, matlabPid: data.matlab_pid, sum: data.sum };
}

export async function apply(ctx, config) {
  const stop = new AbortController();
  ctx.on('dispose', () => stop.abort());
  const identity = JSON.parse(await readFile(config.identityFile, 'utf8'));
  const signal = AbortSignal.any([stop.signal, AbortSignal.timeout(120000)]);
  const recovery = ctx.get('controlEngineering')?.matlabRecovery;
  let pending = await recovery?.pending({inspectOwner:true});
  let receipt;
  if (!pending) {
    try { receipt = await checkStartupConnection(ctx.tools, identity, signal); }
    catch (error) {
      // A concurrent dispatch can enter recovery during the startup request.
      // Keep the Web Host available if the broker now owns a pending record.
      pending = await recovery?.pending({inspectOwner:true});
      if (!pending || stop.signal.aborted) throw error;
    }
  }
  if (pending) {
    const binding = pending.owner?.session;
    if (binding && (binding.session_id !== identity.session_id || binding.pid !== identity.matlab_pid)) {
      throw new Error('待核验的 MATLAB 调用属于另一个实例，请检查专用进程绑定。');
    }
    // Keep the UI and file tools available. Only the Host verifier can settle
    // the outstanding execution. Startup must not send a second normal call.
    receipt = {status:'pending_verification', sessionId:identity.session_id, matlabPid:identity.matlab_pid};
    void recovery.check();
  }
  if (process.env.CONTROL_ONECLICK_SUPERVISE === '1') {
    await superviseWebBinding(ctx, identity, stop.signal, process.env);
  }
  ctx.provide('controlMatlabReady', receipt);
  if (pending) console.log('网页已启动。上次 MATLAB 调用的结果待核实，工程文件和作业记录可用。');
  else console.log(`MATLAB 连接验证通过：R${receipt.release}，专用 PID ${receipt.matlabPid}，2 + 3 = ${receipt.sum}。`);
}
