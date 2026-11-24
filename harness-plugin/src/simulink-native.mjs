import { randomUUID } from 'node:crypto';
import { join, isAbsolute, relative } from 'node:path';
import { dispatchControlledModelOperation } from './matlab-resource.mjs';

const evaluate = 'mcp__matlab__evaluate_matlab_code';

export function matlabString(value) {
  return "'" + value.replaceAll("'", "''") + "'";
}

export function requireSessionMatlab(ctx, config, agent) {
  const resource = config.matlabResource;
  const prefixes = resource?.toolPrefixes ?? ['mcp__matlab__'];
  if (!resource?.directory || !resource.identityFile || !prefixes.some(prefix => evaluate.startsWith(prefix)) ||
      !ctx.tools.definitions(agent).some(tool => tool.name === evaluate)) {
    throw new Error('请通过专用 MATLAB overlay 启动 Web 并在此会话启用原生 MATLAB 工具。');
  }
}

/** helper 和 tag 由本项目指定。参数表达式由调用者从已校验的数值或字符串字面量生成。 */
function nativeHelperCall(directory, helper, tag, expressions) {
  const callId = randomUUID();
  const prefix = 'CONTROL_HARNESS_' + tag + '_' + callId + ':';
  const body = [
    `clear('${helper}');`,
    `assert(strcmp(which('${helper}'),${matlabString(join(directory, helper + '.m'))}));`,
    `fprintf(${matlabString(prefix + '%s\\n')},jsonencode(${helper}(${expressions.join(',')})));`,
  ].join(' ');
  const code = tag === 'MODEL'
    ? `assert(strcmp(which('control_harness_evaluate'),${matlabString(join(directory,'control_harness_evaluate.m'))})); `
      + `control_harness_evaluate(${matlabString(body)},${matlabString(callId)});`
    : body;
  const call = {
    type: 'tool_call', id: tag.toLowerCase() + '-' + callId, name: evaluate,
    arguments: JSON.stringify({code, project_path: directory}),
  };
  function readReceipt(value) {
    const lines = (value?.content ?? []).filter(item => item.type === 'text')
      .flatMap(item => item.text.split(/\r?\n/)).filter(line => line.startsWith(prefix));
    if (lines.length !== 1) throw new Error('MATLAB 没有返回唯一的本次操作回执。');
    return JSON.parse(lines[0].slice(prefix.length));
  }
  return {call, readReceipt};
}

export async function runSimulinkHelper(tools, codeRoot, agent, signal, helper, tag, expressions) {
  const {call, readReceipt} = nativeHelperCall(join(codeRoot, 'matlab'), helper, tag, expressions);
  const batch = await tools.execute({agent, signal, calls:[call]});
  signal.throwIfAborted();
  const result = batch.results[0]?.result;
  if (!result || result.isError) {
    const diagnostic = result?.error?.message ?? '原生 MATLAB 调用失败，请核对审批和专用资源状态。';
    // 原生中间件完成身份和空闲复核后才会附加提示。外层工具继续返回失败并保留这项结论。
    const hasTerminalNote = typeof result?.meta?.matlab_session_verified === 'boolean'
      || typeof result?.meta?.matlab_parameter_retryable === 'boolean'
      || typeof result?.meta?.matlab_wiring_retryable === 'boolean';
    const note = hasTerminalNote ? result.content?.at(-1)?.text : undefined;
    throw Object.assign(new Error(note ? diagnostic + '\n' + note : diagnostic), {
      sessionVerified:result?.meta?.matlab_session_verified,
      identifier:result?.meta?.matlab_error?.identifier,
    });
  }
  return readReceipt(result.value);
}

export async function runControlledModelHelper(tools, directory, agent, signal, helper, expressions, {model,target,validate}) {
  const {call, readReceipt} = nativeHelperCall(directory, helper, 'MODEL', expressions);
  return dispatchControlledModelOperation(tools, agent, signal, call, value => {
    const receipt = readReceipt(value);
    if (receipt?.schema_version !== 1 || !['simulated','failed'].includes(receipt.status) ||
        typeof receipt.stage !== 'string' || !receipt.stage || receipt.model !== model ||
        typeof receipt.path !== 'string' || !isAbsolute(receipt.path) || relative(target, receipt.path) !== '' ||
        !Number.isSafeInteger(receipt.matlab_pid) || !receipt.release ||
        !Array.isArray(receipt.verified) || !Array.isArray(receipt.changes) || typeof receipt.file_exists !== 'boolean' ||
        (receipt.status === 'failed' && (typeof receipt.identifier !== 'string' || typeof receipt.reason !== 'string' ||
          typeof receipt.recoverable !== 'boolean')) ||
        (receipt.status === 'simulated' && (!receipt.file_exists || !Number.isSafeInteger(receipt.bytes) || receipt.bytes < 4))) {
      throw new Error('MATLAB 回执的阶段、模型、文件或结果与本次操作不符。');
    }
    validate?.(receipt);
    return receipt;
  });
}
