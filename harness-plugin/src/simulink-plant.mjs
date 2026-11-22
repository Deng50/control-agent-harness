import { defineTool } from '@deepseek-ai/dsh-tools';
import { matlabString, requireSessionMatlab, runSimulinkHelper } from './simulink-native.mjs';

const numbers = {type: 'array', items: {type: 'number'}};
const matrix = description => ({type: 'array', items: numbers, required: true, description});

function checkMatrix(name, rows, columns, value) {
  if (!rows || !columns || !Array.isArray(value) || value.length !== rows ||
      !value.every(row => Array.isArray(row) && row.length === columns && row.every(Number.isFinite))) {
    const expected = rows && columns ? ` ${rows}×${columns}` : '非空';
    throw new Error(`${name} 须为${expected}的有限数值矩阵，按行传入二维数组。`);
  }
}

function matrixLiteral(name, rows, columns, value) {
  checkMatrix(name, rows, columns, value);
  // 只由有限数字生成 MATLAB 字面量，保留列方向；不接收调用方提供的 MATLAB 表达式。
  return '[' + value.map(row => row.join(' ')).join(';') + ']';
}

// MATLAB jsonencode 将标量、一行和一列矩阵压缩。只恢复已知维度，不对错误尺寸静默整形。
function matrixReadback(name, rows, columns, value) {
  let result = value;
  if (!Array.isArray(value)) result = [[value]];
  else if (rows === 1 && !Array.isArray(value[0])) result = [value];
  else if (columns === 1 && !Array.isArray(value[0])) result = value.map(number => [number]);
  checkMatrix(name, rows, columns, result);
  return result;
}

export async function configureSimulinkPlant(ctx, config, args, {agent, signal}) {
  const {model, block, A, B, C, D, initialCondition, sampleTime} = args;
  if (typeof model !== 'string' || !/^[A-Za-z][A-Za-z0-9_]{0,62}$/.test(model) ||
      typeof block !== 'string' || !block.trim() || block.startsWith('/') || /[\x00-\x1f\x7f]/.test(block)) {
    throw new Error('请提供已加载的模型名和模型内部的相对块路径，例如 Plant 或 Dynamics/Plant。');
  }
  if (!agent?.session || agent.session.id !== agent.id) throw new Error('设置 Plant 需要当前对话的 MATLAB 工具权限。');
  requireSessionMatlab(ctx, config, agent);
  const states = A?.length, inputs = B?.[0]?.length, outputs = C?.length;
  const matrices = [
    matrixLiteral('A', states, states, A), matrixLiteral('B', states, inputs, B),
    matrixLiteral('C', outputs, states, C), matrixLiteral('D', outputs, inputs, D),
  ];
  if (!Array.isArray(initialCondition) || initialCondition.length !== states || !initialCondition.every(Number.isFinite)) {
    throw new Error(`initialCondition 需要 ${states} 个有限数值，顺序与状态一致。`);
  }
  if (!Number.isFinite(sampleTime) || sampleTime <= 0) throw new Error('sampleTime 须为以秒为单位的有限正数。');
  const receipt = await runSimulinkHelper(ctx.tools, config.codeRoot, agent, signal,
    'control_harness_configure_discrete_plant', 'PLANT', [
      matlabString(model), matlabString(block), ...matrices, '[' + initialCondition.join(';') + ']', String(sampleTime),
    ]);
  if (receipt.status === 'refused' || receipt.status === 'unverified') {
    throw new Error(`Plant ${receipt.stage}：${receipt.reason}（参数已写入：${receipt.parameters_applied === true ? '是' : '否'}）`);
  }
  if (receipt.status !== 'configured' || receipt.stage !== 'readback' || receipt.parameters_applied !== true ||
      receipt.model !== model || receipt.block_path !== model + '/' + block ||
      receipt.states !== states || receipt.inputs !== inputs || receipt.outputs !== outputs ||
      !Number.isFinite(receipt.sample_time) || receipt.sample_time <= 0 || receipt.diagram_update !== 'not_run') {
    throw new Error('Plant 读回记录与本次配置不匹配，请重新查询实际块参数。');
  }
  return {...receipt,
    A: matrixReadback('A', states, states, receipt.A), B: matrixReadback('B', states, inputs, receipt.B),
    C: matrixReadback('C', outputs, states, receipt.C), D: matrixReadback('D', outputs, inputs, receipt.D),
    initial_condition: matrixReadback('initial_condition', states, 1, receipt.initial_condition).flat(),
  };
}

export function simulinkPlantTool(ctx, config) {
  return defineTool({
    name: 'control_configure_simulink_plant',
    description: 'Configure a loaded Discrete State-Space block in one parameter group and return its actual numeric readback. Add the block first. Wire the declared input/output vectors and update the diagram after configuration.',
    parameters: {
      model: {type: 'string', required: true, description: 'Loaded top-level model name without .slx'},
      block: {type: 'string', required: true, description: 'Block path relative to model, e.g. Dynamics/Plant'},
      A: matrix('State transition, n by n'), B: matrix('Input matrix, n by m'),
      C: matrix('Output matrix, r by n'), D: matrix('Feedthrough, r by m'),
      initialCondition: {...numbers, required: true, description: 'n state values in model order'},
      sampleTime: {type: 'number', required: true, description: 'Positive discrete sample period in seconds'},
    },
    output: {schema: {type: 'object', additionalProperties: true}, render: (_args, value) => [{type:'text', text:JSON.stringify(value)}]},
    execute: (args, execution) => configureSimulinkPlant(ctx, config, args, execution),
  });
}
