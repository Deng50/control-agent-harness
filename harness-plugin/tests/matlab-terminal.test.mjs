import test from 'node:test';
import assert from 'node:assert/strict';
import { parseMatlabTerminalProbe, createMatlabTerminalVerifier } from '../src/matlab-terminal.mjs';

const nonce = '70c47945-12cc-44b9-85b1-007fba65252f';
const identity = { session_id: 'f5035701-c660-4ec1-9124-71f514447f65', matlab_pid: 12345 };
const session = { session_id: identity.session_id, pid: identity.matlab_pid, identity: 'os-birth-identity' };
const helperFile = 'D:/projects/control-agent-harness/matlab/control_harness_terminal_probe.m';
const probe = { schema_version: 1, nonce, session_id: identity.session_id, matlab_pid: identity.matlab_pid,
  helper_file: helperFile.slice(0, -2), idle: true, busy_models: [], background_futures: 0,
  parallel_futures: 0, running_timers: 0, probe_scope: 'unit fixture only' };
const value = data => ({ content: [{ type: 'text', text: `CONTROL_HARNESS_PROBE_${nonce}:${JSON.stringify(data)}` }] });
const options = { nonce, identity, session, helperFile };

test('only fixed nonce, helper, session and idle state certify a synchronous managed return', () => {
  const result = parseMatlabTerminalProbe(value(probe), options);
  assert.equal(result.producer, 'control_harness_terminal_probe-v1');
  assert.equal(result.matlab_pid, identity.matlab_pid);
  for (const patch of [{ nonce: 'spoof' }, { session_id: 'other' }, { matlab_pid: 999 },
    { helper_file: 'D:/other/control_harness_terminal_probe' }, { idle: false }, { busy_models: ['running'] },
    { background_futures: 1 }, { parallel_futures: 1 }, { running_timers: 1 }]) {
    assert.throws(() => parseMatlabTerminalProbe(value({ ...probe, ...patch }), options));
  }
  assert.throws(() => parseMatlabTerminalProbe(value(probe), { ...options, session: { ...session, pid: 1 } }));
  assert.throws(() => parseMatlabTerminalProbe(value(probe), { ...options, session: undefined }));
  assert.throws(() => parseMatlabTerminalProbe({ content: [{ type: 'text', text: 'idle=true' }] }, options));
  const duplicated = value(probe); duplicated.content.push(...duplicated.content);
  assert.throws(() => parseMatlabTerminalProbe(duplicated, options));
});

test('verifier requires an explicit absolute private identity path', () => {
  assert.throws(() => createMatlabTerminalVerifier({}, { identityFile: 'relative/marker.json' }));
});
