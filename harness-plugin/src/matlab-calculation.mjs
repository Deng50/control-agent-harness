import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';

// Recognize self-contained numerical scripts by their operations BEFORE dispatch.
// An MCP error string, in any language, cannot select this path. Scripts outside
// this subset retain the native evaluator (including plotting, I/O and jobs).
const functions = new Set((
  'abs acos asin assert atan atan2 ceil chol cond conj cos cosh cross ctrb c2d d2c d2d dcgain det diag disp dlqr dot eig ' +
  'error exp expm eye floor fprintf freqresp imag inv isfinite isempty jsonencode length linspace log log10 lqr max mean min ' +
  'mod norm numel obsv ones pinv pole poly rank rcond real reshape roots round sin sinh size sqrt ss ssdata sum svd ' +
  'tan tanh tf trace transpose zeros zpk pi eps inf nan'
).split(' '));
const properties = new Set(['A','B','C','D','E','Ts']);
const numericalOptions = new Set(['zoh','foh','tustin','matched','impulse','s','z','econ','vector','matrix','fro','inf','all']);
const lexeme = /[ \t\r]+|\n|%[^\n]*|'(?:''|[^'\n])*'|"(?:[^"\n]|"")*"|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z][A-Za-z0-9_]*|==|~=|<=|>=|\.[*\/\\^]|[.=,;:()\[\]+*\/\\^<>~-]/gy;
const identifier = /^[A-Za-z]/;
const quote = value => "'" + value.replaceAll("'", "''") + "'";

export function calculationScope(code) {
  if (typeof code !== 'string' || !code.trim() || code.length > 16384 || code.includes('...') || /^\s*%[{}]/m.test(code)) return;
  const tokens = [];
  const lexer = new RegExp(lexeme);
  while (lexer.lastIndex < code.length) {
    const token = lexer.exec(code)?.[0];
    if (!token) return;
    if (!/^[ \r\t%]/.test(token)) tokens.push(token);
  }
  const variables = new Set(), called = new Set(), nesting = [];
  let assignment;
  for (let i = 0; i < tokens.length; i++) {
    const name = tokens[i], previous = tokens[i-1], next = tokens[i+1];
    if (name === '(' || name === '[') nesting.push(name);
    if (name === ')' || name === ']') {
      if (nesting.pop() !== (name === ')' ? '(' : '[')) return;
    }
    if (!nesting.length && [';',',','\n'].includes(name) && assignment) {
      variables.add(assignment);
      assignment = undefined;
    }
    if (name === '.') {
      if (!properties.has(next) || tokens[i+2] === '(') return;
      continue;
    }
    // Ambiguous transpose/string combinations and command-form arguments
    // retain the native path. In this subset strings are expression literals.
    if (/^['"]/.test(name)) {
      const textArgument = previous === '(' && ['fprintf','disp','error','assert'].includes(tokens[i-2]);
      if (!textArgument && (!['(',','].includes(previous) || !numericalOptions.has(name.slice(1,-1)))) return;
    }
    if (!identifier.test(name) || previous === '.') continue;
    if (!nesting.length && next === '=' && (previous === undefined || [';',',','\n'].includes(previous))) {
      if (functions.has(name) || assignment) return;
      assignment = name;
    } else if (functions.has(name)) {
      if (next !== '(' && !['pi','eps','inf','nan'].includes(name)) return;
      if (name === 'fprintf' && !/^['"]/.test(tokens[i+2] ?? '')) return;
      called.add(name);
    } else if (!variables.has(name)) return;
  }
  if (nesting.length) return;
  if (assignment) variables.add(assignment);
  return {variables:[...variables],functions:[...called]};
}

export function calculationCall(args, scope, codeRoot, nonce = randomUUID()) {
  const helper = 'control_harness_calculation';
  const path = resolve(codeRoot, 'matlab', helper + '.m');
  const prefix = 'CONTROL_HARNESS_CALC_' + nonce + ':';
  const literal = '[' + args.code.split(/\r\n?|\n/).map(quote).join(' char(10) ') + ']';
  const cell = names => '{' + names.map(quote).join(',') + '}';
  // Keep the public call's project_path. The original code is visible to the
  // native permission checks on both the outer and the nested invocation.
  const code = `assert(strcmpi(which('${helper}'),${quote(path)})); ` +
    `${helper}(${literal},${quote(nonce)},${cell(scope.variables)},${cell(scope.functions)});`;
  const call = {type:'tool_call',id:'matlab-calc-'+nonce,name:'mcp__matlab__evaluate_matlab_code',arguments:JSON.stringify({...args,code})};
  return {call, parse(result,session) {
    if (result?.isError || !result?.value) throw new Error(result?.error?.message ?? 'MATLAB 计算调用未返回回执。');
    const output = result.value.content;
    if (!Array.isArray(output) || output.some(item => item.type !== 'text')) throw new Error('MATLAB 计算返回了未知格式。');
    const text = output.map(item => item.text).join('\n');
    const index = text.lastIndexOf(prefix);
    if (index < 0 || (index > 0 && text[index-1] !== '\n') || text.indexOf(prefix) !== index) throw new Error('MATLAB 计算缺少唯一回执。');
    const receipt = JSON.parse(text.slice(index+prefix.length));
    if (receipt.nonce !== nonce || receipt.matlab_pid !== session?.pid || receipt.session_id !== session?.session_id ||
        !['returned','failed'].includes(receipt.status) || typeof receipt.certifiable !== 'boolean' ||
        typeof receipt.helper !== 'string' || resolve(receipt.helper + '.m').toLowerCase() !== path.toLowerCase() ||
        (receipt.status === 'failed' && (typeof receipt.identifier !== 'string' || typeof receipt.message !== 'string' || !receipt.message))) {
      throw new Error('MATLAB 计算回执与本次调用或专用进程不一致。');
    }
    // The helper inserts one newline immediately before the receipt.
    const content = [{type:'text',text:text.slice(0,index).replace(/\r?\n$/,'')}];
    return {receipt,content};
  }};
}
