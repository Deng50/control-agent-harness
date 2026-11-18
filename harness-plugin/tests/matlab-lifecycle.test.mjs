/** Command validation/orchestration only. These fixtures are not real MATLAB acceptance. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { apply } from '../../scripts/smoke-matlab-lifecycle.mjs';

const project = resolve(import.meta.dirname, '../..');
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/matlab-lifecycle-command-tests'));
await mkdir(output, { recursive: true });

async function fixture(run) {
  const directory = await mkdtemp(join(output, 'lifecycle-command-'));
  const previous = process.env.CONTROL_MATLAB_RESOURCE_DIRECTORY;
  process.env.CONTROL_MATLAB_RESOURCE_DIRECTORY = directory;
  const state = { schema_version: 1, directory, queue: [], owner: null, quarantine: null };
  const save = () => writeFile(join(directory, 'resource.json'), JSON.stringify(state));
  await save();
  let command, calls = [];
  const ctx = { commands: { register: value => { command = value; } },
    agentPresets: { presetForSession: () => 'control-model' },
    workingDirectory: { ensure: async () => directory },
    tools: { execute: async request => {
      calls.push(request);
      return { results: [{ result: await execute(request, calls.length) }] };
    } } };
  let execute = async () => assert.fail('Unexpected native tool request in command validation fixture');
  apply(ctx);
  try {
    return await run({ state, save, calls, directory,
      setExecute: callback => { execute = callback; },
      invoke: async phase => JSON.parse((await command.handler({ rawInput: phase,
        agent: { id: 'unit-command-session', session: {} }, signal: new AbortController().signal })).text) });
  } finally {
    if (previous === undefined) delete process.env.CONTROL_MATLAB_RESOURCE_DIRECTORY;
    else process.env.CONTROL_MATLAB_RESOURCE_DIRECTORY = previous;
    await rm(directory, { recursive: true, force: true });
  }
}

test('acceptance command rejects caller code, paths and unknown phases before any native tool call', async () => fixture(async ({ invoke, calls }) => {
  for (const input of ['', 'queue-one;disp(42)', 'cancel D:/arbitrary/path', '{"code":"quit"}', 'reset']) {
    const result = await invoke(input);
    assert.equal(result.status, 'FAIL');
    assert.match(result.error, /exactly queue-one, cancel, long or fault/);
  }
  assert.equal(calls.length, 0);
}));

test('fixed queue arithmetic uses the native MATLAB tool and records a reusable terminal state', async () => fixture(async ({ invoke, calls, setExecute }) => {
  setExecute(async request => {
    assert.equal(request.calls[0].name, 'mcp__matlab__evaluate_matlab_code');
    assert.equal(JSON.parse(request.calls[0].arguments).code, "fprintf('CONTROL_HARNESS_S02_QUEUE_ONE:%d\\n',6*7);");
    return { isError: false, value: { content: [{ type: 'text', text: 'CONTROL_HARNESS_S02_QUEUE_ONE:42' }] }, content: [] };
  });
  const result = await invoke('queue-one');
  assert.equal(result.status, 'PASS');
  assert.equal(result.confirmation_source, 'test_fixture');
  assert.equal(result.calls.length, 1);
  assert.equal(calls.length, 1);
}));

test('expected fault is PASS only with its diagnostic, quarantine and a fresh secondary rejection', async () => fixture(async ({ invoke, state, save, setExecute }) => {
  setExecute(async (request, count) => {
    assert.equal(request.signal.aborted, false, 'Secondary quarantine test must not reuse a locally aborted signal');
    if (count === 1) {
      assert.match(JSON.parse(request.calls[0].arguments).code, /error\('ControlHarness:S02FixtureFailure'/);
      state.quarantine = { token: 'unit-fixture-quarantine', reason: 'expected-exception' };
      await save();
      return { isError: true, error: { message: 'ControlHarness:S02FixtureFailure: explicit command-unit diagnostic' }, content: [] };
    }
    return { isError: true, error: { message: 'MATLAB resource is quarantined: unit-fixture-quarantine' }, content: [] };
  });
  const result = await invoke('fault');
  assert.equal(result.status, 'PASS');
  assert.equal(result.secondaryDidNotExecute, true);
  assert.equal(result.calls.length, 2);
}));

test('a MATLAB error alone cannot pass the lifecycle isolation acceptance', async () => fixture(async ({ invoke, setExecute, calls }) => {
  setExecute(async () => ({ isError: true, error: { message: 'ControlHarness:S02FixtureFailure' }, content: [] }));
  const result = await invoke('fault');
  assert.equal(result.status, 'FAIL');
  assert.match(result.error, /must quarantine/);
  assert.equal(calls.length, 1);
}));

test('a pre-existing quarantine is never reset or bypassed by the fixture command', async () => fixture(async ({ invoke, state, save, calls }) => {
  state.quarantine = { token: 'pre-existing-test-token' };
  await save();
  const result = await invoke('queue-one');
  assert.equal(result.status, 'FAIL');
  assert.match(result.error, /never recovers/);
  assert.equal(calls.length, 0);
  assert.equal(result.resourceAfter.quarantine.token, 'pre-existing-test-token');
}));
