/** Catch common interactive save mistakes before dispatch. Native MATLAB remains responsible for executing code. */
export function checkInteractiveModelSave(exec) {
  const code = exec.arguments?.code;
  if (exec.name !== 'mcp__matlab__evaluate_matlab_code' || !exec.agent?.session ||
      typeof code !== 'string' || !code.includes('save_system')) return;

  // 仅识别对话中直接写出的调用；忽略注释、消息文字和引号内的模型名。
  // 转置符保留，避免它被错认为后续字符串的起始引号。
  const expressions = code.replace(/^[ \t]*%\{[ \t]*\r?\n[\s\S]*?^[ \t]*%\}[ \t]*\r?$|%[^\n]*|"(?:[^"\n]|"")*"|(?<![\w)\]}.'])'(?:[^'\n]|'')*'/gm, item => {
    if (item.trimStart().startsWith('%')) return '\n';
    return /^(?:[A-Z]:[\\/]|\\\\|\/)/i.test(item.slice(1, -1)) ? ' ABSOLUTE_LITERAL ' : ' TEXT_LITERAL ';
  });
  if (!/\bsave_system\s*\(/.test(expressions)) return;
  const implicitTarget = /\bsave_system\s*\(\s*(?:\w+\s*)?\)/.test(expressions);
  const relativeTarget = /\bsave_system\s*\(\s*\w+\s*,\s*TEXT_LITERAL\s*[,)]/.test(expressions);
  if (/\bsim\s*\(/.test(expressions) || implicitTarget || relativeTarget) {
    throw new Error('请先用 control_save_simulink_model({model:"模型名"}) 保存到本次对话工作区，取得实际 .slx 路径后再单独运行仿真。本次代码尚未派发。');
  }
}
