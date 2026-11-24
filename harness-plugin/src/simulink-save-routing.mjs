/** Route common interactive wiring and save calls before any part of a mixed batch executes. */
export function checkInteractiveModelEdits(exec) {
  const code = exec.arguments?.code;
  if (exec.name !== 'mcp__matlab__evaluate_matlab_code' || !exec.agent?.session ||
      typeof code !== 'string' || !/\b(?:save_system|add_line|sim)\b/.test(code)) return;

  // 仅识别对话中直接写出的调用；忽略注释、消息文字和引号内的模型名。
  // 转置符保留，避免它被错认为后续字符串的起始引号。
  const expressions = code.replace(/^[ \t]*%\{[ \t]*\r?\n[\s\S]*?^[ \t]*%\}[ \t]*\r?$|%[^\n]*|"(?:[^"\n]|"")*"|(?<![\w)\]}.'])'(?:[^'\n]|'')*'/gm, item => {
    if (item.trimStart().startsWith('%')) return '\n';
    const literal = item.slice(1, -1);
    if (/^(?:[A-Z]:[\\/]|\\\\|\/)/i.test(literal)) return ' ABSOLUTE_LITERAL ';
    if (/^(StopTime|FixedStep|Solver|SolverType)$/i.test(literal)) return ' SIM_SETTING_LITERAL ';
    return /\/[0-9]+$/.test(literal) ? ' PORT_LITERAL ' : ' TEXT_LITERAL ';
  });
  // 常见的“系统、源块/序号、目标块/序号”用结构化工具预检整个批次。
  // 保留使用句柄、坐标或特殊物理端口的原生工作流。底层资源门禁仍处理 MATLAB 实际返回的错误。
  const joined = expressions.replace(/\.\.\.[^\n]*\n/g, ' ');
  if (/\badd_line\s*\(\s*[^,;\n]+,\s*PORT_LITERAL\s*,\s*PORT_LITERAL\s*[,)]/.test(joined)) {
    throw new Error('请将本批普通信号线交给 control_wire_simulink，提交源块、物理输出端口、目标块和物理输入端口。工具会先检查实际端口和已有接线。整段代码尚未派发。');
  }
  if (!/\bsave_system\s*\(/.test(expressions)) {
    // A model-name sim with name/value settings (including #253) has a fixed
    // synchronous entry. Leave SimulationInput, parsim and advanced workflows
    // with the native executor and its conservative unknown-outcome policy.
    if (/\bsim\s*\(\s*TEXT_LITERAL\s*\)|\bsim\s*\(\s*(?:TEXT_LITERAL|[A-Za-z]\w*)\s*,\s*SIM_SETTING_LITERAL\s*,/.test(joined)) {
      throw new Error('请用 control_simulate_simulink_model 按当前对话的已保存模型名提交仿真，可指定数值 stopTime 和 fixedStep。工具检查实际文件并返回编译或仿真的 MException 和各项原因。本批代码尚未派发。');
    }
    return;
  }
  const implicitTarget = /\bsave_system\s*\(\s*(?:\w+\s*)?\)/.test(expressions);
  const relativeTarget = /\bsave_system\s*\(\s*\w+\s*,\s*(?:TEXT|PORT)_LITERAL\s*[,)]/.test(expressions);
  if (/\bsim\s*\(/.test(expressions) || implicitTarget || relativeTarget) {
    throw new Error('请先用 control_save_simulink_model({model:"模型名"}) 保存到本次对话工作区，取得实际 .slx 路径后再单独运行仿真。本次代码尚未派发。');
  }
}
