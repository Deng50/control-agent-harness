import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { checkStartupConnection } from '../src/matlab-startup-check.mjs';

const identity = { matlab_pid: 12345, matlab_release: '2025b', session_id: randomUUID() };
const sample = { matlab_pid: identity.matlab_pid, release: identity.matlab_release, session_id: identity.session_id, sum: 5 };
const value = text => ({ results: [{ result: { isError: false, value: {content: [{type: 'text', text}]} } }] });

function check(render) {
  const tools = {
    get: () => true,
    execute: async ({calls}) => {
      assert.equal(calls.length, 1);
      assert.equal(calls[0].name, 'mcp__matlab__evaluate_matlab_code');
      const { code } = JSON.parse(calls[0].arguments);
      const prefix = 'CONTROL_MATLAB_READY_' + calls[0].id + ':';
      assert.ok(code.includes(prefix));
      assert.ok(code.includes('2+3'));
      return render(prefix);
    },
  };
  return checkStartupConnection(tools, identity, AbortSignal.timeout(1000));
}

test('年度版本、MATLAB 实际 PID、会话及算术都相同才通过', async () => {
  assert.deepEqual(await check(prefix => value(prefix + JSON.stringify(sample))), {
    release: '2025b', matlabPid: identity.matlab_pid, sum: 5,
  });
  for (const delta of [{release:'2024b'}, {matlab_pid:12346}, {session_id:randomUUID()}, {sum:4}]) {
    await assert.rejects(check(prefix => value(prefix + JSON.stringify({...sample, ...delta}))), /不一致/);
  }
});

test('陈旧回执、重复回执、损坏回执、隔离均不能报告已连通', async () => {
  await assert.rejects(check(() => value('other:' + JSON.stringify(sample))), /唯一/);
  await assert.rejects(check(prefix => value([1,2].map(() => prefix + JSON.stringify(sample)).join('\n'))), /唯一/);
  await assert.rejects(check(prefix => value(prefix + '{')), /JSON/);
  await assert.rejects(check(() => ({results:[{result:{isError:true, error:{message:'MatlabResourceQuarantined'}}}]})), /隔离/);
});

test('缺少 Toolkit 的注册或启动中止时不派发 MATLAB 调用', async () => {
  let dispatched = false;
  const tools = { get: name => name.endsWith('evaluate_matlab_code'), execute: () => { dispatched = true; } };
  const stop = new AbortController();
  const pending = checkStartupConnection(tools, identity, stop.signal);
  stop.abort();
  await assert.rejects(pending, {name:'AbortError'});
  assert.equal(dispatched, false);
});
