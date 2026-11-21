/** S02-only, fixed native MATLAB checks. Optional Cordis acceptance plugin + RPC CLI. */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { command, connectHarness, prepareSmokeWorkspace, project, writeSmokeEvidence } from './harness-rpc.mjs';

// Reuse the fixed Harness dependency; no extra package or upstream change.
const yaml = createRequire(resolve(project, 'vendor/deepseek-harness/package.json'))('js-yaml');

export const name = 'control-matlab-native-acceptance';
export const inject = ['commands', 'tools', 'jobs', 'workingDirectory', 'agentPresets'];
const phases = ['connect', 'simulink', 'denial'];
const sha256 = value => createHash('sha256').update(value).digest('hex');
const mq = value => `'${String(value).replaceAll("'", "''")}'`;
const phaseTimeout = { connect: 120000, simulink: 300000, denial: 30000 };

function texts(value, result = [], depth = 0) {
  if (depth > 12) return result;
  if (typeof value === 'string') {
    result.push(value);
    try { texts(JSON.parse(value), result, depth + 1); } catch { /* ordinary MATLAB text */ }
  } else if (Array.isArray(value)) value.forEach(item => texts(item, result, depth + 1));
  else if (value && typeof value === 'object') Object.values(value).forEach(item => texts(item, result, depth + 1));
  return result;
}

function markerJSON(value) {
  for (const text of texts(value)) {
    for (const match of text.matchAll(/CONTROL_S02_JSON:([^\r\n]+)/g)) {
      try { return JSON.parse(match[1]); } catch { /* another enclosing string may hold decoded text */ }
    }
  }
  throw new Error('MATLAB tool did not return the fixed CONTROL_S02_JSON result marker.');
}

async function runPhase(ctx, agent, phase, signal, job) {
  const workspace = ctx.workingDirectory.get(agent.session);
  const directory = resolve(workspace, 'matlab-smoke', `${phase}-${randomUUID()}`);
  await mkdir(directory, { recursive: true });
  const evidence = { schema_version: 1, status: 'RUNNING', confirmation_source: 'test_fixture', phase, directory, workspace, sessionId: agent.id,
    preset: ctx.agentPresets.presetForSession(agent.session), startedAt: new Date().toISOString(), calls: [],
    scope: 'Fixed S02 native tools in a real Harness Session; no LLM, Engine install, or service launch.' };
  async function call(tool, args, expectError = false) {
    signal.throwIfAborted();
    job.updateProgress(`S02 ${phase}: ${tool}`);
    const callId = `s02-${randomUUID()}`, started = Date.now();
    const batch = await ctx.tools.execute({ agent, signal, calls: [{ type: 'tool-call', id: callId, name: tool, arguments: JSON.stringify(args) }] });
    assert.equal(batch.results.length, 1);
    const result = batch.results[0].result;
    const file = resolve(directory, `${String(evidence.calls.length + 1).padStart(2, '0')}-${tool}.json`);
    const serialized = `${JSON.stringify(result, null, 2)}\n`;
    await writeFile(file, serialized);
    evidence.calls.push({ tool, callId, durationMs: Date.now() - started, isError: !!result.isError,
      argumentsSha256: sha256(JSON.stringify(args)), raw_result: file, resultSha256: sha256(serialized) });
    assert.equal(!!result.isError, expectError, `${tool} returned native error; inspect ${file}`);
    return expectError ? result : result.value;
  }
  const evaluate = code => call('mcp__matlab__evaluate_matlab_code', { code, project_path: directory });
  try {
    assert.ok(/^control-(model|controller|lightweight)$/.test(evidence.preset), 'Use a real engineering role Session.');
    for (const tool of ['detect_matlab_toolboxes', 'evaluate_matlab_code', 'run_matlab_file']) {
      assert.ok(ctx.tools.definitions(agent).some(item => item.name === `mcp__matlab__${tool}`), `Missing native MATLAB tool ${tool}`);
    }
    if (phase === 'connect') {
      await call('mcp__matlab__detect_matlab_toolboxes', {});
      const identity = markerJSON(await evaluate("disp(['CONTROL_S02_JSON:' jsonencode(struct('release',version('-release'),'arithmetic',6*7,'session_id',getenv('CONTROL_HARNESS_SESSION_ID'),'matlab_pid',feature('getpid')))])"));
      assert.equal(identity.release, '2024b');
      assert.equal(identity.arithmetic, 42);
      const expected = JSON.parse(await readFile(process.env.CONTROL_MATLAB_IDENTITY_FILE, 'utf8'));
      assert.equal(identity.session_id, expected.session_id);
      assert.equal(identity.matlab_pid, expected.matlab_pid);
      const script = resolve(directory, 's02_core_arithmetic.m');
      const source = "% S02 fixed public arithmetic fixture. No model or user data.\ns02_core_result = struct('release',version('-release'),'arithmetic',6*7);\ndisp(['CONTROL_S02_JSON:' jsonencode(s02_core_result)])\n";
      await writeFile(script, source);
      const fileResult = markerJSON(await call('mcp__matlab__run_matlab_file', { script_path: script }));
      assert.equal(fileResult.arithmetic, 42);
      assert.equal(fileResult.release, '2024b');
      evidence.identity = identity;
      evidence.m_file = { path: script, sha256: sha256(source), result: fileResult };
    } else if (phase === 'denial') {
      const dispose = ctx.tools.guard(exec => exec.agent?.id === agent.id && exec.name.startsWith('mcp__matlab__') ? 'S02_FIXED_NATIVE_DENIAL' : undefined);
      try {
        const result = await call('mcp__matlab__detect_matlab_toolboxes', {}, true);
        assert.ok(JSON.stringify(result).includes('S02_FIXED_NATIVE_DENIAL'), 'The denial did not come from the fixed native guard.');
        evidence.nativeDenial = true;
      } finally { dispose(); evidence.guardDisposed = true; }
      evidence.resourceComparison = 'Root CLI must compare resource ticket/last_completion before and after this phase.';
    } else {
      await runSimulink({ ctx, agent, signal, directory, evidence, call, evaluate });
    }
    evidence.status = 'PASS';
  } catch (error) {
    evidence.status = 'FAIL';
    evidence.error = error instanceof Error ? error.message : String(error);
  }
  evidence.finishedAt = new Date().toISOString();
  await writeSmokeEvidence(resolve(directory, 'result.json'), evidence);
  job.append(`${JSON.stringify({ event: 'result', result: evidence })}\n`, { channel: 'stdout' });
  return { status: evidence.status === 'PASS' ? 'completed' : 'failed', detail: evidence.error ?? `S02 ${phase} passed`, result: JSON.stringify(evidence) };
}

function objectIn(value, predicate) {
  const visited = new Set();
  function visit(item, depth = 0) {
    if (depth > 12 || item === null || item === undefined) return undefined;
    if (typeof item === 'object') {
      if (visited.has(item)) return undefined;
      visited.add(item);
      if (predicate(item)) return item;
      for (const child of Object.values(item)) { const found = visit(child, depth + 1); if (found !== undefined) return found; }
    } else if (typeof item === 'string') {
      for (const candidate of [item, item.slice(item.indexOf('{'), item.lastIndexOf('}') + 1)]) {
        try { const found = visit(JSON.parse(candidate), depth + 1); if (found !== undefined) return found; } catch { /* non-JSON presentation */ }
      }
    }
    return undefined;
  }
  return visit(value);
}

// SATK 2026.11.a renders its structured result as YAML inside MATLAB's
// displayed char/string value. Keep the fixed acceptance fields strict rather
// than treating a successful MCP transport as a successful model edit.
function satkDisplayText(text) {
  const displayed = text.match(/(?:^|\n)([ \t]*)(['"])(status:[\s\S]*)$/);
  if (!displayed) return text;
  const body = displayed[3].trimEnd();
  assert.ok(body.endsWith(displayed[2]), 'Unterminated MATLAB display wrapper around SATK result.');
  const lines = body.slice(0, -1).split(/\r?\n/);
  const displayIndent = `${displayed[1]} `;
  return lines.map((line, index) => {
    if (index === 0 || !line.trim()) return line;
    assert.ok(line.startsWith(displayIndent), 'Unexpected indentation in MATLAB display wrapper.');
    return line.slice(displayIndent.length);
  }).join('\n');
}

function satkStatus(value) {
  const structured = objectIn(value, item => typeof item.status === 'string');
  if (structured) return structured.status;
  for (const text of texts(value)) {
    const match = satkDisplayText(text).match(/(?:^|\n)\s*status:\s*([a-z_-]+)\s*(?:\r?\n|$)/i);
    if (match) return match[1];
  }
  throw new Error('SATK result has no recognized status field.');
}

function satkCreated(value) {
  const structured = objectIn(value, item => item.created && typeof item.created === 'object');
  if (structured) return structured.created;
  for (const text of texts(value)) {
    const lines = satkDisplayText(text).split(/\r?\n/);
    const start = lines.findIndex(line => /^\s*created:\s*$/.test(line));
    if (start < 0) continue;
    const indentation = lines[start].match(/^\s*/)[0].length;
    const created = {};
    for (const line of lines.slice(start + 1)) {
      if (!line.trim()) continue;
      if (line.match(/^\s*/)[0].length <= indentation) break;
      const match = line.match(/^\s+([A-Za-z][A-Za-z0-9_]*):\s*(blk_\d+)\s*$/);
      assert.ok(match, 'Unrecognized entry in SATK created block map.');
      created[match[1]] = match[2];
    }
    if (Object.keys(created).length) return created;
  }
  throw new Error('SATK edit did not return a created block map.');
}

function assertHealthyCheck(value) {
  assert.equal(satkStatus(value), 'healthy', 'Model structural check did not report healthy.');
  assert.ok(!objectIn(value, item => item.severity === 'error') &&
    !texts(value).some(text => /(?:^|\n)\s*(?:-\s*)?severity:\s*error\b/.test(text)),
  'Model structural check found an error.');
}

function satkDocument(value) {
  const structured = objectIn(value, item => typeof item.status === 'string');
  if (structured) return structured;
  for (const text of texts(value)) {
    const display = satkDisplayText(text).trim();
    if (!display.startsWith('status:')) continue;
    const document = yaml.load(display, { schema: yaml.JSON_SCHEMA });
    assert.ok(document && typeof document === 'object', 'SATK result is not a structured YAML document.');
    return document;
  }
  throw new Error('SATK result has no structured document.');
}

function resolvedScalar(value) {
  if (typeof value === 'number') {
    assert.ok(Number.isFinite(value), 'Resolved parameter is not finite.');
    return value;
  }
  if (typeof value === 'string') {
    try { return resolvedScalar(JSON.parse(value)); } catch { /* reject presentation-only values below */ }
  }
  if (Array.isArray(value) && value.length === 1) return resolvedScalar(value[0]);
  if (value && typeof value === 'object' && Object.hasOwn(value, 'value')) return resolvedScalar(value.value);
  throw new Error('Resolver did not expose an unambiguous numeric scalar value.');
}

function assertNear(actual, expected, tolerance, label) {
  assert.equal(actual.length, expected.length, `${label} shape`);
  let maximum = 0;
  for (let i = 0; i < expected.length; i++) {
    if (Array.isArray(expected[i])) {
      maximum = Math.max(maximum, assertNear(actual[i], expected[i], tolerance, `${label}[${i}]`));
    } else {
      assert.ok(Number.isFinite(actual[i]), `${label}[${i}] must be finite`);
      const error = Math.abs(actual[i] - expected[i]);
      assert.ok(error <= tolerance.atol + tolerance.rtol * Math.abs(expected[i]), `${label}[${i}] error ${error}`);
      maximum = Math.max(maximum, error);
    }
  }
  return maximum;
}

async function runSimulink({ directory, evidence, call, evaluate }) {
  // The installed skills are actually loaded in this role; no MATLAB model
  // inspection or construction occurs before the library gate result is read.
  for (const skill of ['building-simulink-models', 'simulating-simulink-models']) await call('skill', { name: skill });
  const gate = markerJSON(await evaluate("s02_gate = library.settingsLookup(); disp(['CONTROL_S02_JSON:' jsonencode(s02_gate)]);"));
  evidence.libraryGate = gate;
  assert.ok(gate.found === false || gate.gatePass === true, 'Library/policy prerequisites are incomplete; no model was created.');
  if (gate.found && gate.kgIndexPath) await call('read', { file_path: gate.kgIndexPath, limit: 120 });

  const sourceDirectory = resolve(project, 'tests/fixtures/matlab-s02');
  const referencePath = resolve(directory, 'f01-reference.json');
  const helperPath = resolve(directory, 'simulate_f01.m');
  await copyFile(resolve(sourceDirectory, 'f01-reference.json'), referencePath);
  await copyFile(resolve(sourceDirectory, 'simulate_f01.m'), helperPath);
  const reference = JSON.parse(await readFile(referencePath, 'utf8'));
  assert.equal(reference.dt, 0.2);
  assert.deepEqual(reference.A, [[0, 1], [-2, -0.8]]);
  assert.deepEqual(reference.B, [[0], [1]]);
  assert.deepEqual(reference.C, [[1, 0]]);
  evidence.reference = { file: referencePath, sha256: sha256(await readFile(referencePath)), generator: reference.generator, source: 'Independent SciPy ZOH and synthetic bounded fixed input; not measured data.' };
  evidence.helper = { file: helperPath, sha256: sha256(await readFile(helperPath)) };

  // A pair of scalar Unit Delay states implements the same exact ZOH model.
  // This avoids the installed model_edit's per-parameter updates exposing
  // temporarily incompatible A/B/C dimensions on a Discrete State-Space block.
  const library = markerJSON(await evaluate("load_system('simulink'); s02_lib = struct('type',get_param('simulink/Discrete/Unit Delay','BlockType'),'parameters',{fieldnames(get_param('simulink/Discrete/Unit Delay','DialogParameters'))},'gain_parameters',{fieldnames(get_param('simulink/Math Operations/Gain','DialogParameters'))},'sum_parameters',{fieldnames(get_param('simulink/Math Operations/Sum','DialogParameters'))}); disp(['CONTROL_S02_JSON:' jsonencode(s02_lib)]);"));
  for (const parameter of ['InitialCondition', 'SampleTime']) assert.ok(library.parameters.includes(parameter), `Installed delay parameter ${parameter} is unavailable`);
  assert.ok(library.gain_parameters.includes('Gain'), 'Installed Gain parameter is unavailable');
  assert.ok(library.sum_parameters.includes('Inputs'), 'Installed Sum parameter is unavailable');
  evidence.installedParameters = library;
  evidence.libraryBlocks = ['Inport', 'Gain', 'Sum', library.type, 'Outport'];
  evidence.realization = 'Two scalar Unit Delay states with six named exact-ZOH coefficient gains; x(k+1)=Ad*x(k)+Bd*(gain*u(k)+d(k)), y(k)=x1(k).';
  const originalName = `ch_s02_${randomUUID().replaceAll('-', '').slice(0, 12)}`;
  const editedName = `${originalName}_copy`;
  const originalPath = resolve(directory, `${originalName}.slx`), editedPath = resolve(directory, `${editedName}.slx`);
  evidence.models = { originalName, editedName, originalPath, editedPath };
  await evaluate(`assert(~bdIsLoaded(${mq(originalName)})); new_system(${mq(originalName)}); open_system(${mq(originalName)}); s02_ref=jsondecode(fileread(${mq(referencePath)})); s02_aug=[s02_ref.A s02_ref.B; zeros(1,3)]; s02_zoh=expm(s02_aug*s02_ref.dt); s02_ws=get_param(${mq(originalName)},'ModelWorkspace'); s02_ws.assignin('S02_Ad',s02_zoh(1:2,1:2)); s02_ws.assignin('S02_Bd',s02_zoh(1:2,3)); s02_ws.assignin('S02_C',s02_ref.C); s02_ws.assignin('S02_x0',s02_ref.x0(:)); s02_ws.assignin('S02_Ad11',s02_zoh(1,1)); s02_ws.assignin('S02_Ad12',s02_zoh(1,2)); s02_ws.assignin('S02_Ad21',s02_zoh(2,1)); s02_ws.assignin('S02_Ad22',s02_zoh(2,2)); s02_ws.assignin('S02_Bd1',s02_zoh(1,3)); s02_ws.assignin('S02_Bd2',s02_zoh(2,3)); s02_ws.assignin('S02_x01',s02_ref.x0(1)); s02_ws.assignin('S02_x02',s02_ref.x0(2)); s02_ws.assignin('S02_dt',s02_ref.dt); s02_ws.assignin('S02_input_gain',1); disp(['CONTROL_S02_JSON:' jsonencode(struct('created',true))]);`);
  await call('mcp__matlab__model_read', { model: originalName, scope: 'root', depth: '0' });
  const operations = [
    { op: 'add_block', type: 'Inport', name: 'Force', ref: 'force', params: { Port: '1' } },
    { op: 'add_block', type: 'Inport', name: 'Disturbance', ref: 'disturbance', params: { Port: '2' } },
    { op: 'add_block', type: 'Gain', name: 'InputGain', ref: 'gain', params: { Gain: 'S02_input_gain' } },
    { op: 'add_block', type: 'Sum', name: 'TotalForce', ref: 'sum', params: { Inputs: '++' } },
    { op: 'add_block', type: library.type, name: 'State1', ref: 'x1', params: { InitialCondition: 'S02_x01', SampleTime: 'S02_dt' } },
    { op: 'add_block', type: library.type, name: 'State2', ref: 'x2', params: { InitialCondition: 'S02_x02', SampleTime: 'S02_dt' } },
    { op: 'add_block', type: 'Gain', name: 'Ad11', ref: 'a11', params: { Gain: 'S02_Ad11' } },
    { op: 'add_block', type: 'Gain', name: 'Ad12', ref: 'a12', params: { Gain: 'S02_Ad12' } },
    { op: 'add_block', type: 'Gain', name: 'Bd1', ref: 'b1', params: { Gain: 'S02_Bd1' } },
    { op: 'add_block', type: 'Gain', name: 'Ad21', ref: 'a21', params: { Gain: 'S02_Ad21' } },
    { op: 'add_block', type: 'Gain', name: 'Ad22', ref: 'a22', params: { Gain: 'S02_Ad22' } },
    { op: 'add_block', type: 'Gain', name: 'Bd2', ref: 'b2', params: { Gain: 'S02_Bd2' } },
    { op: 'add_block', type: 'Sum', name: 'NextState1', ref: 'next1', params: { Inputs: '+++' } },
    { op: 'add_block', type: 'Sum', name: 'NextState2', ref: 'next2', params: { Inputs: '+++' } },
    { op: 'add_block', type: 'Outport', name: 'Displacement', ref: 'output', params: { Port: '1' } },
    { op: 'connect', target: '#force.y1 -> #gain.u1' },
    { op: 'connect', target: '#gain.y1 -> #sum.u1' },
    { op: 'connect', target: '#disturbance.y1 -> #sum.u2' },
    { op: 'connect', target: '#x1.y1 -> #a11.u1' },
    { op: 'connect', target: '#x2.y1 -> #a12.u1' },
    { op: 'connect', target: '#sum.y1 -> #b1.u1' },
    { op: 'connect', target: '#a11.y1 -> #next1.u1' },
    { op: 'connect', target: '#a12.y1 -> #next1.u2' },
    { op: 'connect', target: '#b1.y1 -> #next1.u3' },
    { op: 'connect', target: '#next1.y1 -> #x1.u1' },
    { op: 'connect', target: '#x1.y1 -> #a21.u1' },
    { op: 'connect', target: '#x2.y1 -> #a22.u1' },
    { op: 'connect', target: '#sum.y1 -> #b2.u1' },
    { op: 'connect', target: '#a21.y1 -> #next2.u1' },
    { op: 'connect', target: '#a22.y1 -> #next2.u2' },
    { op: 'connect', target: '#b2.y1 -> #next2.u3' },
    { op: 'connect', target: '#next2.y1 -> #x2.u1' },
    { op: 'connect', target: '#x1.y1 -> #output.u1' },
    { op: 'configure', target: `config:${originalName}`, params: { SolverType: 'Fixed-step', Solver: 'FixedStepDiscrete', FixedStep: '0.2', StopTime: '4' } },
  ];
  const build = await call('mcp__matlab__model_edit', { model: originalName, scope: 'root', operations: JSON.stringify(operations), layout_mode: 'full' });
  // Both diagnostics run immediately even if an edit reports partial completion.
  const structure = await call('mcp__matlab__model_read', { model: originalName, scope: 'root', depth: 'inf' });
  const checked = await call('mcp__matlab__model_check', { model: originalName, scope: 'root', checks: '["unconnected_ports","unconnected_lines"]' });
  assert.equal(satkStatus(build), 'ok', 'Model edit did not complete; inspect the saved read/check diagnostics.');
  assertHealthyCheck(checked);
  const created = satkCreated(build);
  for (const operation of operations.filter(item => item.op === 'add_block')) {
    assert.match(created[operation.ref] ?? '', /^blk_\d+$/, `Missing created ID for ${operation.ref}`);
    assert.ok(texts(structure).join('\n').includes(created[operation.ref]), `model_read did not confirm ${operation.ref}`);
  }
  evidence.createdBlockIds = created;
  await evaluate(`save_system(${mq(originalName)},${mq(originalPath)}); disp(['CONTROL_S02_JSON:' jsonencode(struct('saved',true))]);`);
  const expectedParameters = { S02_Ad11: reference.Ad[0][0], S02_Ad12: reference.Ad[0][1],
    S02_Ad21: reference.Ad[1][0], S02_Ad22: reference.Ad[1][1],
    S02_Bd1: reference.Bd[0][0], S02_Bd2: reference.Bd[1][0],
    S02_x01: reference.x0[0], S02_x02: reference.x0[1], S02_dt: reference.dt, S02_input_gain: 1 };
  const resolved = await call('mcp__matlab__model_resolve_params', { model: originalPath, expressions: JSON.stringify(Object.keys(expectedParameters)) });
  const resolution = satkDocument(resolved);
  assert.equal(resolution.status, 'ok', 'Parameter resolver did not report success.');
  assert.ok(resolution.errors === undefined || resolution.errors === null ||
    (typeof resolution.errors === 'object' && Object.keys(resolution.errors).length === 0),
  'Parameter resolver reported errors.');
  const actualParameters = {};
  for (const [parameter, expected] of Object.entries(expectedParameters)) {
    const container = objectIn(resolution.results, item => Object.hasOwn(item, parameter));
    assert.ok(container, `Resolver omitted ${parameter}`);
    const actual = resolvedScalar(container[parameter]);
    assertNear([actual], [expected], reference.tolerance, `resolved ${parameter}`);
    actualParameters[parameter] = actual;
  }
  evidence.resolvedParameters = { expected: expectedParameters, actual: actualParameters, tolerance: reference.tolerance };
  const originalHash = sha256(await readFile(originalPath));
  const originalOutput = resolve(directory, 'original-simulation.json');
  const stateIds = mq(JSON.stringify({ x1: created.x1, x2: created.x2 }));
  const baseline = markerJSON(await evaluate(`addpath(${mq(directory)}); s02_result=simulate_f01(${mq(originalName)},${mq(referencePath)},${mq(originalOutput)},jsondecode(${stateIds})); disp(['CONTROL_S02_JSON:' jsonencode(s02_result)]);`));
  assertNear(baseline.time, reference.time, reference.tolerance, 'original time');
  const baselineError = { states: assertNear(baseline.states, reference.baseline.states, reference.tolerance, 'original states'),
    outputs: assertNear(baseline.outputs, reference.baseline.outputs, reference.tolerance, 'original outputs') };
  await evaluate(`close_system(${mq(originalName)},0); disp(['CONTROL_S02_JSON:' jsonencode(struct('closed',true))]);`);
  assert.equal(sha256(await readFile(originalPath)), originalHash, 'Original model changed during non-destructive simulation');
  await copyFile(originalPath, editedPath);
  await evaluate(`open_system(${mq(editedPath)}); disp(['CONTROL_S02_JSON:' jsonencode(struct('opened',true))]);`);
  await call('mcp__matlab__model_read', { model: editedPath, scope: 'root', depth: '0' });
  const edited = await call('mcp__matlab__model_edit', { model: editedPath, scope: 'root', operations: JSON.stringify([{ op: 'configure', target: created.gain, params: { Gain: String(reference.edited.gain) } }]), layout_mode: 'incremental' });
  await call('mcp__matlab__model_read', { model: editedPath, scope: 'root', depth: '0' });
  const editCheck = await call('mcp__matlab__model_check', { model: editedPath, scope: 'root', checks: '["unconnected_ports","unconnected_lines"]' });
  assert.equal(satkStatus(edited), 'ok', 'Copy edit did not complete; inspect read/check evidence');
  assertHealthyCheck(editCheck);
  const editedOutput = resolve(directory, 'edited-simulation.json');
  await evaluate(`save_system(${mq(editedName)},${mq(editedPath)});`);
  const variant = markerJSON(await evaluate(`s02_result=simulate_f01(${mq(editedName)},${mq(referencePath)},${mq(editedOutput)},jsondecode(${stateIds})); disp(['CONTROL_S02_JSON:' jsonencode(s02_result)]);`));
  assertNear(variant.time, reference.time, reference.tolerance, 'edited time');
  const editedError = { states: assertNear(variant.states, reference.edited.states, reference.tolerance, 'edited states'),
    outputs: assertNear(variant.outputs, reference.edited.outputs, reference.tolerance, 'edited outputs') };
  assert.ok(Math.max(...variant.outputs.map((value, i) => Math.abs(value - baseline.outputs[i]))) > 1e-4, 'Copy Gain edit did not change the actual output');
  await evaluate(`close_system(${mq(editedName)},0); disp(['CONTROL_S02_JSON:' jsonencode(struct('closed',true))]);`);
  assert.equal(sha256(await readFile(originalPath)), originalHash, 'Original model changed after editing the copy');
  evidence.comparison = { tolerance: reference.tolerance, baselineError, editedError, originalUnchanged: true };
  evidence.files = [];
  for (const file of [originalPath, editedPath, originalOutput, editedOutput]) evidence.files.push({ path: file, sha256: sha256(await readFile(file)) });
}

export function apply(ctx) {
  ctx.commands.register({
    definitionId: '@control-engineering/acceptance/matlab-native-smoke', name: 'matlab-native-smoke',
    description: 'S02 fixed-phase native MATLAB acceptance; no arbitrary tool/code/path input',
    handler({ agent, signal, rawInput }) {
      signal.throwIfAborted();
      const args = JSON.parse(rawInput);
      assert.deepEqual(Object.keys(args), ['phase']);
      assert.ok(phases.includes(args.phase), 'Unknown fixed S02 phase');
      const jobId = ctx.jobs.start({ owner: agent.id, kind: 'control-engineering', label: `S02 MATLAB ${args.phase}`, run(job) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(new Error(`S02 ${args.phase} bounded phase expired`)), phaseTimeout[args.phase]);
        timer.unref?.();
        const done = runPhase(ctx, agent, args.phase, controller.signal, job).finally(() => clearTimeout(timer));
        return { done, cancel(reason) { controller.abort(new Error(reason ?? 'S02 acceptance cancelled')); } };
      } });
      return { kind: 'success', text: JSON.stringify({ jobId, phase: args.phase }) };
    },
  });
}

async function main() {
  const { values } = parseArgs({ options: { workspace: { type: 'string' }, evidence: { type: 'string' },
    'web-log': { type: 'string' }, phase: { type: 'string', default: 'connect' }, role: { type: 'string', default: 'model' }, help: { type: 'boolean', short: 'h' } } });
  if (values.help) {
    console.log('node scripts/smoke-matlab-native.mjs --phase connect|simulink|denial --workspace <directory> --evidence <json> --web-log <Host-log> [--role model|controller|lightweight]');
    console.log('Fixed test-only Cordis plugin must be loaded by the existing Host. No service/session launch and no LLM.');
    return;
  }
  assert.ok(phases.includes(values.phase));
  assert.ok(['model', 'controller', 'lightweight'].includes(values.role));
  const workspace = resolve(values.workspace ?? process.env.CONTROL_SMOKE_WORKSPACE ?? resolve(project, 'test-output', `matlab-native-${randomUUID()}`));
  const evidencePath = resolve(values.evidence ?? process.env.CONTROL_SMOKE_EVIDENCE ?? resolve(workspace, 'evidence.json'));
  await prepareSmokeWorkspace(workspace);
  const evidence = { status: 'RUNNING', confirmation_source: 'test_fixture', phase: values.phase, workspace, checkedAt: new Date().toISOString(), scope: 'S02 real native Session and MCP; no LLM' };
  await writeSmokeEvidence(evidencePath, evidence);
  try {
    const rpc = await connectHarness({ logPath: values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG });
    const created = await rpc('session/create', { request: { cwd: workspace, agentPreset: `control-${values.role}` } });
    evidence.sessionId = created.sessionId;
    const started = await command(rpc, created.sessionId, `/matlab-native-smoke ${JSON.stringify({ phase: values.phase })}`);
    evidence.jobId = started.jobId;
    await writeSmokeEvidence(evidencePath, evidence);
    const deadline = Date.now() + phaseTimeout[values.phase] + 30000;
    while (Date.now() < deadline) {
      const state = await command(rpc, created.sessionId, `/control-status ${started.jobId}`);
      evidence.nativeStatus = state.job.status;
      if (['completed', 'failed', 'killed'].includes(state.job.status)) {
        const output = state.output.chunks.map(chunk => chunk.text).join('');
        const result = output.split('\n').flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } }).findLast(row => row.event === 'result')?.result;
        assert.ok(result, `Native job ended without fixed phase evidence; inspect ${started.jobId}`);
        evidence.result = result;
        evidence.status = result.status;
        const page = await rpc('session/events', { request: { address: { kind: 'session', sessionId: created.sessionId }, maxMessages: 100 } });
        evidence.eventTypes = [...new Set(page.records.map(row => row.event.type))];
        assert.ok(!evidence.eventTypes.some(type => type.startsWith('turn/')), 'S02 non-LLM check opened a model turn');
        assert.equal(state.job.status, 'completed', result.error);
        assert.equal(result.status, 'PASS', result.error);
        return;
      }
      await new Promise(done => setTimeout(done, 400));
    }
    throw new Error(`S02 native job ${started.jobId} is still not terminal; inspect this handle and do not resubmit.`);
  } catch (error) {
    evidence.error = error instanceof Error ? error.message : String(error);
    if (!['running', 'stopping'].includes(evidence.nativeStatus)) evidence.status = 'FAIL';
    throw error;
  } finally {
    await writeSmokeEvidence(evidencePath, evidence);
    console.log(JSON.stringify({ status: evidence.status, phase: evidence.phase, evidence: evidencePath, sessionId: evidence.sessionId, jobId: evidence.jobId }));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
