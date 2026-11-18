/**
 * S02 acceptance only: fixed real MATLAB calls through native Session tools.
 * Load this optional Cordis plugin only in the local acceptance overlay. The
 * command accepts a phase name, never user code/tool names/paths. It never
 * restarts, resets, recovers or terminates MATLAB or its MCP process.
 */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { connectHarness, command, prepareSmokeWorkspace, project, writeSmokeEvidence } from './harness-rpc.mjs';

export const name = 'control-matlab-lifecycle-acceptance';
export const inject = ['commands', 'tools', 'workingDirectory', 'agentPresets'];
export const phases = ['queue-one', 'cancel', 'long', 'fault'];
const TOOL = 'mcp__matlab__evaluate_matlab_code';
const FAILURE = 'ControlHarness:S02FixtureFailure';
const roles = ['model', 'controller', 'lightweight'];
const pause = ms => new Promise(resolveWait => setTimeout(resolveWait, ms));
const matlabString = value => `'${value.replaceAll("'", "''")}'`;
const resultText = result => JSON.stringify(result.error ?? result.value ?? result.content);

async function doesExist(path) {
  try { await stat(path); return true; }
  catch (error) { if (error.code === 'ENOENT') return false; throw error; }
}

function startedCode(path, payload, { wait = false } = {}) {
  return `s02FixtureFile=fopen(${matlabString(path.replaceAll('\\', '/'))},'w');` +
    `assert(s02FixtureFile>=0,'ControlHarness:S02MarkerWriteFailed','Cannot write fixed acceptance marker');` +
    `fprintf(s02FixtureFile,'%s',${matlabString(JSON.stringify(payload))});fclose(s02FixtureFile);` +
    (wait ? "disp('CONTROL_HARNESS_S02_STARTED');pause(15);disp('CONTROL_HARNESS_S02_FINISHED');" : "disp('CONTROL_HARNESS_S02_SECOND_CALL_MUST_NOT_RUN');");
}

async function waitForMarker(path, expected, pending, signal) {
  let settled = false, outcome;
  pending.then(result => { settled = true; outcome = result; }, error => { settled = true; outcome = { error: String(error) }; });
  const deadline = Date.now() + 20000;
  while (true) {
    signal.throwIfAborted();
    if (await doesExist(path)) {
      let actual;
      try { actual = JSON.parse(await readFile(path, 'utf8')); }
      catch (error) {
        // fopen makes the file visible before fprintf/fclose has finished.
        if (!(error instanceof SyntaxError)) throw error;
        assert.ok(Date.now() < deadline, 'MATLAB started marker remained incomplete');
        await pause(40);
        continue;
      }
      assert.deepEqual(actual, expected, 'The real MATLAB marker must match this fixed test invocation');
      return { path, observedAt: new Date().toISOString(), content: actual };
    }
    if (settled) throw new Error(`MATLAB call settled before its real started marker: ${resultText(outcome)}`);
    assert.ok(Date.now() < deadline, 'MATLAB did not write the fixed started marker within 20 seconds');
    await pause(40);
  }
}

export function apply(ctx) {
  ctx.commands.register({
    definitionId: '@control-engineering/acceptance/matlab-lifecycle', name: 'matlab-lifecycle',
    description: 'Local S02 acceptance: fixed MATLAB queue/cancel/disconnect/error fixtures, no model request',
    input: { hint: 'One fixed phase: queue-one | cancel | long | fault' },
    async handler({ agent, signal, rawInput }) {
      const evidence = { schema_version: 1, confirmation_source: 'test_fixture',
        startedAt: new Date().toISOString(), sessionId: agent.id, calls: [], status: 'RUNNING' };
      let localAbort, pending;
      try {
        const phase = rawInput.trim();
        assert.ok(phases.includes(phase), 'This test-only command accepts exactly queue-one, cancel, long or fault; no code or paths');
        evidence.phase = phase;
        const preset = ctx.agentPresets.presetForSession(agent.session);
        assert.ok(roles.some(role => preset === `control-${role}`), 'Use a real control role preset Session');
        evidence.preset = preset;
        evidence.workspace = await ctx.workingDirectory.ensure(agent);
        assert.ok(isAbsolute(evidence.workspace), 'The real Session workspace must be absolute');
        const directory = process.env.CONTROL_MATLAB_RESOURCE_DIRECTORY;
        assert.ok(directory && isAbsolute(directory), 'The acceptance Host needs its configured shared MATLAB resource directory');
        evidence.resourceDirectory = directory;
        const snapshot = async () => JSON.parse(await readFile(resolve(directory, 'resource.json'), 'utf8'));
        evidence.resourceBefore = await snapshot();
        assert.equal(evidence.resourceBefore.quarantine, null, 'Start each phase with an explicitly healthy resource; this command never recovers it');

        async function call(stage, code, callSignal = signal) {
          const startedAt = new Date().toISOString(), started = Date.now();
          const batch = await ctx.tools.execute({ agent, signal: callSignal,
            calls: [{ type: 'tool_call', id: `matlab-lifecycle-${randomUUID()}`, name: TOOL, arguments: JSON.stringify({ code }) }] });
          assert.equal(batch.results.length, 1);
          const result = batch.results[0].result;
          evidence.calls.push({ stage, tool: TOOL, fixedCode: code, startedAt,
            settledAt: new Date().toISOString(), durationMs: Date.now() - started, result });
          return result;
        }

        let result;
        if (phase === 'queue-one') {
          result = await call('queue-arithmetic', "fprintf('CONTROL_HARNESS_S02_QUEUE_ONE:%d\\n',6*7);");
          assert.equal(result.isError, false, resultText(result));
          assert.match(JSON.stringify(result.value), /CONTROL_HARNESS_S02_QUEUE_ONE:42/);
          evidence.resourceAfter = await snapshot();
          assert.equal(evidence.resourceAfter.quarantine, null, 'Verified synchronous arithmetic should remain reusable');
          evidence.expectedOutcome = 'One short real MATLAB call, admitted through the shared FIFO and terminal verifier';
        } else {
          const nonce = randomUUID();
          const secondaryMarker = resolve(evidence.workspace, 'matlab-lifecycle-rejected-secondary.started.json');
          assert.equal(await doesExist(secondaryMarker), false, 'Use a fresh fixture workspace; preserve pre-existing markers');
          if (phase === 'fault') {
            result = await call('actual-matlab-exception', `error('${FAILURE}','${FAILURE}: Fixed S02 acceptance exception');`);
            assert.equal(result.isError, true, 'The actual MATLAB exception must produce a native error outcome');
            assert.ok(JSON.stringify(result).includes(FAILURE), 'The actual MATLAB diagnostic must identify this fixed exception');
          } else {
            const markerPath = resolve(evidence.workspace, `matlab-lifecycle-${phase}.started.json`);
            assert.equal(await doesExist(markerPath), false, 'Use a fresh fixture workspace; preserve pre-existing markers');
            const payload = { kind: 'control-harness-s02-lifecycle-started', confirmation_source: 'test_fixture', phase, nonce };
            localAbort = new AbortController();
            const callSignal = phase === 'cancel' ? AbortSignal.any([signal, localAbort.signal]) : signal;
            pending = call('actual-matlab-pause', startedCode(markerPath, payload, { wait: true }), callSignal);
            evidence.marker = await waitForMarker(markerPath, payload, pending, signal);
            if (phase === 'cancel') {
              evidence.cancelRequestedAt = new Date().toISOString();
              evidence.cancelSource = 'Host observed the real fixed MATLAB started marker, then aborted only this native invocation';
              localAbort.abort(new Error('ControlHarness S02 test fixture cancellation after real MATLAB started marker'));
            }
            result = await pending;
            pending = undefined;
            assert.equal(result.isError, true, phase === 'long'
              ? 'Expected root-controlled MCP disconnect/timeout did not occur before the fixed call returned'
              : 'The real cancelled SDK invocation must settle as an error, not proof that MATLAB stopped');
          }
          evidence.resourceAfterFailure = await snapshot();
          assert.ok(evidence.resourceAfterFailure.quarantine, 'An error/cancel/disconnect must quarantine the MATLAB resource');
          // Deliberately use the original, non-locally-aborted command signal.
          // A second fresh native call must fail due to quarantine itself.
          const rejected = await call('quarantine-rejects-secondary', startedCode(secondaryMarker,
            { kind: 'must-not-execute-after-quarantine', confirmation_source: 'test_fixture', nonce }), signal);
          assert.equal(rejected.isError, true);
          assert.match(resultText(rejected), /quarantin/i, 'The second call must be rejected by shared resource quarantine');
          assert.equal(await doesExist(secondaryMarker), false, 'The rejected second MATLAB body must not write its marker');
          evidence.secondaryDidNotExecute = true;
          evidence.resourceAfter = await snapshot();
          assert.equal(evidence.resourceAfter.quarantine.token, evidence.resourceAfterFailure.quarantine.token);
          evidence.expectedOutcome = 'Real expected negative case: error diagnostic plus durable isolation, secondary body did not execute';
          evidence.recovery = 'Not attempted. Root must verify dedicated-session stop/exit and coordinate explicit recovery.';
        }
        evidence.status = 'PASS';
      } catch (error) {
        evidence.status = 'FAIL';
        evidence.error = error instanceof Error ? error.message : String(error);
        // Never abandon an admitted native Promise. Cancelling this fixture can
        // quarantine it but cannot be interpreted as terminating MATLAB work.
        if (pending) {
          localAbort?.abort(new Error('S02 acceptance command cleanup after fixture assertion failed'));
          try { await pending; } catch (settlement) { evidence.pendingSettlementError = String(settlement); }
        }
        if (evidence.resourceDirectory) {
          try { evidence.resourceAfter = JSON.parse(await readFile(resolve(evidence.resourceDirectory, 'resource.json'), 'utf8')); }
          catch (snapshotError) { evidence.snapshotError = String(snapshotError); }
        }
      }
      evidence.settledAt = new Date().toISOString();
      return { kind: 'success', text: JSON.stringify(evidence) };
    },
  });
}

function options() {
  const { values } = parseArgs({ options: { workspace: { type: 'string' }, evidence: { type: 'string' },
    'web-log': { type: 'string' }, phase: { type: 'string' }, role: { type: 'string' }, help: { type: 'boolean', short: 'h' } } });
  if (values.help) {
    console.log('node scripts/smoke-matlab-lifecycle.mjs --phase queue-one|cancel|long|fault [--role model|controller|lightweight] [--workspace <absolute>] [--evidence <json>] [--web-log <log>]');
    console.log('Requires this file as an optional test-only Cordis plugin. Negative cases intentionally leave MATLAB quarantined; no reset, restart or termination is attempted.');
    return { help: true };
  }
  const phase = values.phase ?? process.env.CONTROL_MATLAB_LIFECYCLE_PHASE ?? 'queue-one';
  const role = values.role ?? process.env.CONTROL_MATLAB_LIFECYCLE_ROLE ?? 'model';
  assert.ok(phases.includes(phase), 'Select one fixed lifecycle phase');
  assert.ok(roles.includes(role), 'Select one of the three fixed control roles');
  return { phase, role,
    workspace: resolve(values.workspace ?? process.env.CONTROL_SMOKE_WORKSPACE ?? resolve(project, 'test-output', `matlab-lifecycle-${phase}-${randomUUID()}`)),
    evidence: resolve(values.evidence ?? process.env.CONTROL_SMOKE_EVIDENCE ?? resolve(project, 'test-output/smoke-evidence', `matlab-lifecycle-${phase}-${randomUUID()}.json`)),
    webLog: resolve(values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG ?? resolve(project, 'harness-web.log')) };
}

async function main() {
  const selected = options();
  if (selected.help) return;
  const evidence = { schema_version: 1, confirmation_source: 'test_fixture', startedAt: new Date().toISOString(),
    status: 'RUNNING', phase: selected.phase, role: selected.role, workspace: selected.workspace,
    scope: 'Real authenticated native role Session, MATLAB MCP and shared resource; no model request, session restart or recovery' };
  await writeSmokeEvidence(selected.evidence, evidence);
  try {
    evidence.workspacePreparation = await prepareSmokeWorkspace(selected.workspace);
    const rpc = await connectHarness({ logPath: selected.webLog });
    const roster = await rpc('agentPresets/list', {});
    const presetId = `control-${selected.role}`;
    const preset = roster.presets.find(row => row.id === presetId);
    assert.ok(preset && !preset.broken, 'The chosen real control preset is unavailable');
    const created = await rpc('session/create', { request: { cwd: selected.workspace, agentPreset: presetId } });
    evidence.sessionId = created.sessionId;
    await writeSmokeEvidence(selected.evidence, evidence);
    const commands = await rpc('commands/list', { agentId: created.sessionId });
    assert.ok(commands.some(row => row.name === 'matlab-lifecycle'), 'Load this optional test-only Cordis acceptance plugin before running the CLI');
    evidence.result = await command(rpc, created.sessionId, `/matlab-lifecycle ${selected.phase}`);
    assert.equal(evidence.result.status, 'PASS', evidence.result.error);
    const page = await rpc('session/events', { request: { address: { kind: 'session', sessionId: created.sessionId }, maxMessages: 100 } });
    evidence.eventTypes = [...new Set(page.records.map(record => record.event.type))];
    assert.ok(!evidence.eventTypes.some(type => type.startsWith('turn/')), 'The fixed lifecycle command unexpectedly opened a model turn');
    evidence.status = 'PASS';
  } catch (error) {
    evidence.status = 'FAIL';
    evidence.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    evidence.settledAt = new Date().toISOString();
    await writeSmokeEvidence(selected.evidence, evidence);
    console.log(JSON.stringify({ status: evidence.status, phase: selected.phase, role: selected.role,
      workspace: selected.workspace, evidence: selected.evidence, sessionId: evidence.sessionId,
      ...(evidence.error ? { error: evidence.error } : {}) }));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
