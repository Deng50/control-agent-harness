import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { configureSimulinkPlant } from '../src/simulink-plant.mjs';

const args = {
  model: 'ControlDemo', block: "Axis 2/Plant's dynamics",
  A: [[0.9]], B: [[0.1, -0.2]], C: [[1], [2]], D: [[0, 0], [0, 0]],
  initialCondition: [1], sampleTime: 0.1,
};
const nativeTool = 'mcp__matlab__evaluate_matlab_code';
const receipt = {status: 'configured', stage: 'readback', parameters_applied: true, model: args.model,
  block_path: args.model + '/' + args.block, states: 1, inputs: 2, outputs: 2, diagram_update: 'not_run',
  ports: {input_count: 1, output_count: 1, required_input_width: 2, required_output_width: 2},
  A: 0.9, B: [0.1, -0.2], C: [1, 2], D: [[0, 0], [0, 0]], initial_condition: 1, sample_time: 0.1};

function fixture(reply = receipt) {
  const codeRoot = resolve(import.meta.dirname, '../..');
  const agent = {id: 'test', session: {id: 'test'}}, signal = new AbortController().signal;
  const calls = [];
  const tools = {
    definitions: () => [{name: nativeTool}],
    async execute(request) {
      assert.equal(request.agent, agent);
      assert.equal(request.signal, signal);
      const call = request.calls[0];
      assert.equal(call.name, nativeTool);
      calls.push(JSON.parse(call.arguments));
      const line = 'CONTROL_HARNESS_PLANT_' + call.id.slice('plant-'.length) + ':' + JSON.stringify(reply);
      return {results: [{result: {isError: false, value: {content: [{type:'text', text:line}]}}}]};
    },
  };
  const config = {codeRoot, matlabResource: {directory: 'fixture', identityFile: 'fixture'}};
  return {calls, tools, run: (values = args) => configureSimulinkPlant({tools}, config, values, {agent, signal})};
}

test('numeric tool uses one guarded native call, preserves block name and returns rectangular JSON readback', async () => {
  const f = fixture();
  const actual = await f.run();
  assert.equal(f.calls.length, 1);
  assert.ok(f.calls[0].code.includes("which('control_harness_configure_discrete_plant')"));
  assert.ok(f.calls[0].code.includes("'Axis 2/Plant''s dynamics',[0.9],[0.1 -0.2],[1;2]"));
  for (const key of ['A', 'B', 'C', 'D']) assert.deepEqual(actual[key], args[key]);
  assert.deepEqual(actual.initial_condition, args.initialCondition);
  assert.equal(actual.sample_time, args.sampleTime);
  assert.deepEqual(actual.ports, receipt.ports);
});

test('malformed dimensions, expressions, invalid sample times and unavailable native tools stop before dispatch', async () => {
  const f = fixture();
  for (const invalid of [{A: []}, {B: [[1], [1, 2]]}, {C: [['gain']]}, {D: [[0], [0]]},
    {initialCondition: [NaN]}, {sampleTime: 0}, {sampleTime: Infinity}]) {
    await assert.rejects(f.run({...args, ...invalid}));
  }
  f.tools.definitions = () => [];
  await assert.rejects(f.run(), /启用原生 MATLAB/);
  assert.deepEqual(f.calls, []);
});

test('native refusals, uncertain readback, and a receipt for another block never report configured', async () => {
  for (const alternate of [{...receipt, block_path:'Another/Plant'}, {...receipt, C: [1]},
    {...receipt, sample_time: null}, {...receipt, ports: {...receipt.ports, required_input_width: 1}},
    {...receipt, status:'refused', stage:'validation', reason:'wrong block', parameters_applied:false},
    {...receipt, status:'unverified', reason:'mismatch', parameters_applied:true}]) {
    await assert.rejects(fixture(alternate).run());
  }
  const f = fixture();
  const diagnostic = "matlab error: 参数 'B' 的设置无效。";
  f.tools.execute = async () => ({results: [{result: {isError: true, error: {message: diagnostic},
    meta: {matlab_parameter_retryable: true}, content: [{type: 'text', text: '复核通过，可纠正参数'}]}}]});
  await assert.rejects(f.run(), error => error.message.includes(diagnostic) && error.message.includes('复核通过'));
});
