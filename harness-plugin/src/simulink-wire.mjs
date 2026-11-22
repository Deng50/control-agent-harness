import { defineTool } from '@deepseek-ai/dsh-tools';
import { matlabString, requireSessionMatlab, runSimulinkHelper } from './simulink-native.mjs';

const endpointFields = {
  sourceBlock: {type: 'string', required: true, description: 'Exact output block path relative to model'},
  sourcePort: {type: 'integer', required: true, description: 'Physical Outport number, starting at 1'},
  destinationBlock: {type: 'string', required: true, description: 'Exact input block path relative to model'},
  destinationPort: {type: 'integer', required: true, description: 'Physical Inport number, starting at 1'},
};

function validPath(path) {
  return typeof path === 'string' && path.trim().length > 0 && !path.startsWith('/') && !/[\x00-\x1f\x7f]/u.test(path);
}

function validSignalWidth(value) {
  return (Number.isSafeInteger(value) && value > 0) || (Array.isArray(value) && value.length === 0);
}

export async function wireSimulink(ctx, config, {model, connections}, {agent, signal}) {
  if (typeof model !== 'string' || !/^[A-Za-z][A-Za-z0-9_]{0,62}$/u.test(model)) {
    throw new Error('请提供已加载的顶层 Simulink 模型名。');
  }
  if (!agent?.session || agent.session.id !== agent.id) throw new Error('接线需要当前对话的 MATLAB 工具权限。');
  requireSessionMatlab(ctx, config, agent);
  if (!Array.isArray(connections) || !connections.length) throw new Error('请至少提供一条待核对的接线。');
  for (const edge of connections) {
    if (!edge || !validPath(edge.sourceBlock) || !validPath(edge.destinationBlock) ||
        !Number.isSafeInteger(edge.sourcePort) || edge.sourcePort < 1 ||
        !Number.isSafeInteger(edge.destinationPort) || edge.destinationPort < 1 ||
        Object.keys(edge).some(key => !Object.hasOwn(endpointFields, key))) {
      throw new Error('每条接线需要模型内的源块和目标块路径，以及从 1 开始的物理输出、输入端口号。');
    }
  }
  // JSON 作为数据交给 MATLAB。块名可包含引号，无需把名称拼接成 MATLAB 接线命令。
  const data = matlabString(JSON.stringify(connections));
  const receipt = await runSimulinkHelper(ctx.tools, config.codeRoot, agent, signal,
    'control_harness_wire_simulink', 'WIRE', [matlabString(model), 'jsondecode(' + data + ')']);
  if (receipt?.status === 'refused' && receipt.stage === 'preflight' && receipt.created === 0) {
    throw new Error('接线预检未通过，本次新建 0 条线。' + receipt.reason);
  }
  const edges = receipt?.connections;
  if (receipt?.status !== 'wired' || receipt.stage !== 'readback' || receipt.model !== model ||
      !Array.isArray(edges) || edges.length !== connections.length) {
    throw new Error('缺少与本次模型和接线计划匹配的完整读回记录，请重新核对实际连线。');
  }
  let created = 0;
  for (const [i, edge] of edges.entries()) {
    if (!edge || !Object.keys(endpointFields).every(key => edge[key] === connections[i][key]) ||
        !Number.isSafeInteger(edge.sourcePortCount) || edge.sourcePortCount < edge.sourcePort ||
        !Number.isSafeInteger(edge.destinationPortCount) || edge.destinationPortCount < edge.destinationPort ||
        !validSignalWidth(edge.sourceSignalWidth) || !validSignalWidth(edge.destinationSignalWidth) ||
        !['created', 'existing'].includes(edge.action)) {
      throw new Error('接线读回与请求端点不一致，请核对实际连线。');
    }
    if (edge.action === 'created') created++;
  }
  if (receipt.created !== created || receipt.existing !== edges.length - created) {
    throw new Error('接线计数与端点读回不一致，请核对实际连线。');
  }
  return receipt;
}

export function simulinkWireTool(ctx, config) {
  return defineTool({
    name: 'control_wire_simulink',
    description: 'Wire ordinary Simulink signal ports on a loaded model. Preflight the entire batch against physical ports and existing lines, add missing lines and return verified endpoints and port counts. Signal widths are positive when compiled and known, [] when unknown. Use a Mux to feed separate scalars to a vector port.',
    parameters: {
      model: {type: 'string', required: true, description: 'Loaded top-level model name without .slx'},
      connections: {type: 'array', required: true, items: {type: 'object', additionalProperties: false, properties: endpointFields},
        description: 'Desired connections in order. Each source and destination must share a parent system.'},
    },
    output: {schema: {type: 'object', additionalProperties: true}, render: (_args, value) => [{type: 'text', text: JSON.stringify(value)}]},
    execute: (args, execution) => wireSimulink(ctx, config, args, execution),
  });
}
