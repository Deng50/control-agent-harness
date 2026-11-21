import assert from 'node:assert/strict';
import test from 'node:test';
import { checkInteractiveModelSave } from '../src/simulink-save-routing.mjs';

const name = 'mcp__matlab__evaluate_matlab_code';
const agent = {session: {header: {cwd: 'D:/workspaces/demo'}}};
const check = code => checkInteractiveModelSave({name, agent, arguments: {code}});

test('会话中误用默认目录、相对文件名或合并仿真时，在派发前提示专用保存入口', () => {
  for (const code of [
    "save_system('MassSpringDamper'); set_param('MassSpringDamper/F','Value','1'); sim('MassSpringDamper');",
    'save_system(bdroot)', 'save_system()', 'save_system("Demo")',
    "save_system(m,'out.slx')", "save_system('Demo','sub/Demo.slx')",
    "save_system(m,'D:/workspaces/demo/Demo.slx'); sim(m);",
  ]) assert.throws(() => check(code), /control_save_simulink_model/, code);
});

test('分开的保存和仿真可派发；注释、字符串与矩阵转置不会触发分流', () => {
  for (const code of [
    String.raw`save_system(m, 'D:\workspaces\demo\Demo.slx')`,
    "sim('Demo')", "disp('save_system(m)'); sim(m);",
    'disp("message ""save_system(m)"""); sim(m);', '% save_system(m)\nsim(m);',
    '%{\r\nsave_system(m)\r\n%}\r\nsim(m);',
    "x = A'; save_system(m,'/workspace/Demo.slx'); % sim(m)",
  ]) assert.doesNotThrow(() => check(code), code);
  assert.doesNotThrow(() => checkInteractiveModelSave({name: 'another_tool', agent, arguments: {code:'save_system(m)'}}));
});
