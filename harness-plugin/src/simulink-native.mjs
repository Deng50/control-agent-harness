import { randomUUID } from 'node:crypto';
import { join } from 'node:path';

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
export async function runSimulinkHelper(tools, codeRoot, agent, signal, helper, tag, expressions) {
  const directory = join(codeRoot, 'matlab');
  const callId = randomUUID();
  const prefix = 'CONTROL_HARNESS_' + tag + '_' + callId + ':';
  const code = [
    `clear('${helper}');`,
    `assert(strcmp(which('${helper}'),${matlabString(join(directory, helper + '.m'))}));`,
    `fprintf(${matlabString(prefix + '%s\\n')},jsonencode(${helper}(${expressions.join(',')})));`,
  ].join(' ');
  const batch = await tools.execute({agent, signal, calls: [{
    type: 'tool_call', id: tag.toLowerCase() + '-' + callId, name: evaluate,
    arguments: JSON.stringify({code, project_path: directory}),
  }]});
  signal.throwIfAborted();
  const result = batch.results[0]?.result;
  if (!result || result.isError) {
    const diagnostic = result?.error?.message ?? '原生 MATLAB 调用失败，请核对审批和专用资源状态。';
    // 原生中间件完成身份和空闲复核后才会附加提示。外层工具继续返回失败并保留这项结论。
    const hasTerminalNote = typeof result?.meta?.matlab_parameter_retryable === 'boolean'
      || typeof result?.meta?.matlab_wiring_retryable === 'boolean';
    const note = hasTerminalNote ? result.content?.at(-1)?.text : undefined;
    throw new Error(note ? diagnostic + '\n' + note : diagnostic);
  }
  const lines = (result.value?.content ?? []).filter(item => item.type === 'text')
    .flatMap(item => item.text.split(/\r?\n/)).filter(line => line.startsWith(prefix));
  if (lines.length !== 1) throw new Error('MATLAB 没有返回唯一的本次操作回执。');
  return JSON.parse(lines[0].slice(prefix.length));
}
