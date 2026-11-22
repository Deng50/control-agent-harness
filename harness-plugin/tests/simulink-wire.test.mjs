import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import test from 'node:test';
import { wireSimulink } from '../src/simulink-wire.mjs';

const model = 'WireDemo';
const edge = {sourceBlock: "Axes/Drive's force", sourcePort: 1, destinationBlock: 'Axes/Mux', destinationPort: 2};
const verified = {...edge, sourcePortCount:1, destinationPortCount:2, sourceSignalWidth:[], destinationSignalWidth:[], action:'existing'};
const receipt = {model, status: 'wired', stage: 'readback', created: 0, existing: 1,
  connections: [verified]};
const native = 'mcp__matlab__evaluate_matlab_code';

function fixture(reply = receipt) {
  const agent = {id: 'web-test', session: {id: 'web-test'}};
  const config = {codeRoot: resolve(import.meta.dirname, '../..'), matlabResource: {directory: 'test', identityFile: 'test'}};
  const signal = new AbortController().signal;
  const calls = [];
  const tools = {
    definitions: () => [{name: native}],
    async execute(request) {
      assert.equal(request.agent, agent);
      assert.equal(request.signal, signal);
      const [call] = request.calls;
      assert.equal(call.name, native);
      calls.push(JSON.parse(call.arguments));
      const marker = 'CONTROL_HARNESS_WIRE_' + call.id.slice('wire-'.length) + ':';
      return {results: [{result: {isError: false, value: {content: [{type:'text', text: marker + JSON.stringify(reply)}]}}}]};
    },
  };
  return {agent, calls, tools, run: (args = {model, connections: [edge]}) => wireSimulink({tools}, config, args, {agent, signal})};
}

test('结构化端点作为数据发送，核对同一模型与每个端点读回', async () => {
  const f = fixture();
  assert.deepEqual(await f.run(), receipt);
  assert.equal(f.calls.length, 1);
  assert.ok(f.calls[0].code.includes("which('control_harness_wire_simulink')"));
  assert.ok(f.calls[0].code.includes("jsondecode('" + JSON.stringify([edge]).replaceAll("'", "''") + "')"));
});

test('无权限或端点输入不合约定时，不派发到 MATLAB', async () => {
  const f = fixture();
  for (const edit of [{sourceBlock: '/Other/Source'}, {destinationPort: 0}, {sourcePort: 1.5}, {changeExisting: true}]) {
    await assert.rejects(f.run({model, connections: [{...edge, ...edit}]}));
  }
  await assert.rejects(f.run({model, connections: []}));
  f.tools.definitions = () => [];
  await assert.rejects(f.run(), /启用原生 MATLAB/);
  f.agent.session.id = 'different';
  await assert.rejects(f.run(), /当前对话/);
  assert.equal(f.calls.length, 0);
});

test('预检零写入与读回不符均保留失败，已有同源连线可验证', async () => {
  const refused = {model, status: 'refused', stage: 'preflight', created: 0, reason: 'Plant 只有一个输入端口'};
  await assert.rejects(fixture(refused).run(), /0 条线。Plant/);
  for (const other of [{model: 'other'}, {stage: 'applied'}, {connections: []},
    {connections: [{...verified, sourceBlock: 'Another'}]}, {connections: [{...verified, destinationPortCount: 1}]},
    {connections: [{...verified, sourceSignalWidth: -1}]}, {connections: [{...verified, action: 'pending'}]},
    {created: 1}, {existing: 0}]) {
    await assert.rejects(fixture({...receipt, ...other}).run(), /读回/);
  }
});
