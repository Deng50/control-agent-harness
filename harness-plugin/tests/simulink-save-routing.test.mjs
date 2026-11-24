import assert from 'node:assert/strict';
import test from 'node:test';
import { checkInteractiveModelEdits } from '../src/simulink-save-routing.mjs';

const name = 'mcp__matlab__evaluate_matlab_code';
const agent = {session: {header: {cwd: 'D:/workspaces/demo'}}};
const check = code => checkInteractiveModelEdits({name, agent, arguments: {code}});

test('原始 #253 带 StopTime 和 FixedStep 的 sim 批次只走受控仿真，保留 SimulationInput',()=>{
  assert.throws(()=>check("simOut = sim(modelName, 'StopTime','0.5','FixedStep','0.1'); disp(simOut);"),/control_simulate_simulink_model/);
  assert.throws(()=>check("sim('Demo')"),/control_simulate_simulink_model/);
  assert.doesNotThrow(()=>check("si=Simulink.SimulationInput('Demo'); y=sim(si);"));
  assert.doesNotThrow(()=>check("sim('Demo','SimulationMode','accelerator')"));
});

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
    "disp('save_system(m)'); sim(m);",
    'disp("message ""save_system(m)"""); sim(m);', '% save_system(m)\nsim(m);',
    '%{\r\nsave_system(m)\r\n%}\r\nsim(m);',
    "x = A'; save_system(m,'/workspace/Demo.slx'); % sim(m)",
  ]) assert.doesNotThrow(() => check(code), code);
  assert.doesNotThrow(() => checkInteractiveModelEdits({name: 'another_tool', agent, arguments: {code:'save_system(m)'}}));
});

test('会话中的普通端口字面量接线整批改走预检，原生句柄和物理连线仍可用', () => {
  for (const code of [
    "add_line(m,'Ref_Pos/1','Input_Mux/3');\nadd_line(m,'Disturbance/1','Plant/2');",
    'add_line("Demo", ...\n "Axis 1/Force/1", ... % 第二个参数\n "Mux/2", "autorouting", "on");',
  ]) assert.throws(() => check(code), /control_wire_simulink/);
  for (const code of [
    "disp(\"add_line(m,'u/1','Plant/2')\");", "% add_line(m,'u/1','y/1')",
    'add_line(m,sourceHandle,destinationHandle);', "add_line(m,'A/RConn1','B/LConn1');",
  ]) assert.doesNotThrow(() => check(code));
});
