/**
 * S01 acceptance only. The CLI talks to the real Web Host; this module's optional
 * Cordis export runs a fixed tool checklist in that Host's real role Session.
 * Load it only with the local acceptance overlay, never as a product tool bridge.
 * No tool/provider/Agent/LLM implementation is replaced and no model is requested.
 */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { connectHarness, command, smokeOptions, writeSmokeEvidence } from './harness-rpc.mjs';

export const name = 'control-native-capability-acceptance';
export const inject = ['commands', 'tools', 'workingDirectory', 'agentPresets'];
const entries = { model: 'train-object-model', controller: 'design-and-tune-mpc', lightweight: 'distill-controller' };
const referenceTasks = {
  model: ['仅有温度和功率的时序 CSV，按整条工况训练并留出测试', 'audit-train-tabular-model', 'controlkit/table.py'],
  controller: ['用 Simulink 搭二阶对象做闭环 MPC，和 ADRC 比较', 'build-native-mpc', 'cases/matlab-scenes/mpc2/Run_MPC_For_2nd_order_Plant_auto_tuning.m'],
  lightweight: ['把耗时的 MPC 蒸馏成一个实时可运行的小网络', 'clone-control-policy', 'controlkit/policy.py'],
};
const fixtureName = 'native-capability-fixture.txt';
const marker = 'CONTROL_NATIVE_CAPABILITY_FIXTURE';
const sha256 = value => createHash('sha256').update(value).digest('hex');

export function apply(ctx) {
  ctx.commands.register({
    definitionId: '@control-engineering/acceptance/native-capability-smoke',
    name: 'native-capability-smoke',
    description: 'Local S01 acceptance: fixed non-LLM native tool checks in this role Session',
    async handler({ agent, signal, rawInput }) {
      const evidence = { sessionId: agent.id, checkedAt: new Date().toISOString(), calls: [] };
      try {
        const libraryOnly = rawInput.trim() === 'library';
        assert.ok(libraryOnly || !rawInput.trim(), 'Use no arguments or the fixed library checklist.');
        const preset = ctx.agentPresets.presetForSession(agent.session);
        const role = preset?.replace(/^control-/, '');
        assert.ok(entries[role], 'Use a real control-model/controller/lightweight preset Session.');
        evidence.role = role;
        evidence.preset = preset;
        evidence.workspace = ctx.workingDirectory.get(agent.session);
        const definitions = ctx.tools.definitions(agent);
        evidence.tools = definitions.map(tool => tool.name).sort();
        const shell = process.platform === 'win32' ? 'pwsh' : 'bash';
        for (const tool of ['skill', 'read', 'glob', 'grep', shell, 'job_list', 'job_output', 'job_kill', 'ask_user_question', 'subagent', 'subagent_fork', 'list_agents', 'workflow', 'todo_write', 'present']) {
          assert.ok(evidence.tools.includes(tool), `${preset} is missing native tool ${tool}`);
        }
        for (const other of Object.keys(entries).filter(item => item !== role)) assert.ok(!evidence.tools.includes(`control_${other}`));
        assert.ok(evidence.tools.includes(`control_${role}`));

        async function call(tool, args, expectError = false) {
          signal.throwIfAborted();
          const started = Date.now();
          const batch = await ctx.tools.execute({ agent, signal, calls: [{ type: 'tool-call', id: `native-smoke-${randomUUID()}`, name: tool, arguments: JSON.stringify(args) }] });
          assert.equal(batch.results.length, 1);
          const result = batch.results[0].result;
          evidence.calls.push({ tool, arguments: args, isError: !!result.isError, durationMs: Date.now() - started,
            resultSha256: sha256(JSON.stringify(result)), ...(result.isError ? { error: result.error ?? null } : {}) });
          assert.equal(!!result.isError, expectError, `${tool} failed: ${JSON.stringify(result.error ?? result.content)}`);
          return result.value;
        }
        const [query, expectedSkill, expectedCode] = referenceTasks[role];
        const result = await call('control_library_search', { role, query, limit: 1 });
        const first = result.matches[0];
        assert.equal(first?.skill, expectedSkill);
        assert.equal(first.code, expectedCode);
        assert.equal(first.role, role);
        const loaded = await call('skill', { name: first.skill });
        assert.equal(loaded.name, first.skill);
        assert.ok(loaded.content.length > 100);
        const lib = resolve(import.meta.dirname, '../harness-plugin/control-skill-library');
        const skillSource = await readFile(resolve(lib, first.instructions), 'utf8');
        assert.ok(skillSource.includes(loaded.content.trim()), 'The native skill must match the bundled instructions.');
        const code = await call('control_library_read', { path: first.code, length: 1000 });
        assert.equal(code.path, first.code);
        assert.equal(code.snapshot_sha256, result.snapshot_sha256);
        assert.equal(code.sha256, sha256(await readFile(resolve(lib, first.code))));
        assert.ok(code.text.length > 100);
        evidence.library = { query, skill: loaded.name, provider: loaded.provider, code: code.path,
          codeSha256: code.sha256, snapshotSha256: result.snapshot_sha256 };
        if (libraryOnly) {
          evidence.status = 'PASS';
          return { kind: 'success', text: JSON.stringify(evidence) };
        }
        evidence.skills = [];
        for (const skill of [entries[role], 'simulating-simulink-models']) {
          const loaded = await call('skill', { name: skill });
          assert.equal(loaded.name, skill);
          assert.ok(loaded.content.length > 100);
          evidence.skills.push({ name: skill, provider: loaded.provider, resourceBase: loaded.resourceBase,
            contentLength: loaded.content.length, contentSha256: sha256(loaded.content) });
        }
        const fixture = resolve(evidence.workspace, fixtureName);
        const read = await call('read', { file_path: fixture, limit: 5 });
        assert.ok(JSON.stringify(read).includes(marker));
        assert.ok(JSON.stringify(await call('glob', { path: evidence.workspace, pattern: fixtureName })).includes(fixtureName));
        assert.ok(JSON.stringify(await call('grep', { path: evidence.workspace, pattern: marker })).includes(marker));
        const shellResult = await call(shell, { description: 'S01 native shell arithmetic', command: process.platform === 'win32' ? 'Write-Output (6 * 7)' : "printf '%s\\n' 42", workdir: evidence.workspace, timeoutMs: 10000 });
        assert.equal(shellResult.exitCode, 0);
        assert.match(shellResult.stdout.text, /42/);
        // Completed foreground calls intentionally remove their unexposed job id.
        // An explicit short background call exercises the durable job tool surface.
        const background = await call(shell, { description: 'S01 native background job arithmetic', command: process.platform === 'win32' ? 'Write-Output (6 * 7)' : "printf '%s\\n' 42", workdir: evidence.workspace, run_in_background: true });
        assert.equal(background.kind, 'background');
        const jobs = await call('job_list', {});
        assert.ok(jobs.some(job => job.id === background.jobId && job.kind === shell), 'The explicit background job was not listed');
        const output = await call('job_output', { job_id: background.jobId, wait: true, timeout_ms: 10000 });
        assert.equal(output.job.status, 'completed');
        assert.match(output.text, /42/);
        assert.deepEqual(await call('list_agents', {}), []);
        const workflow = await call('workflow', { meta: { name: 'native-capability-smoke', description: 'S01 arithmetic orchestration without agent or model calls' },
          script: 'const values = await parallel([async () => 1, async () => 2]); return { nativeSmoke: true, sum: values.reduce((a,b) => a+b,0) };' });
        assert.equal(workflow.agentsStarted, 0);
        assert.deepEqual(workflow.result, { nativeSmoke: true, sum: 3 });
        const denied = Object.keys(entries).find(item => item !== role);
        await call(`control_${denied}`, {}, true);
        evidence.subagentScope = 'Native subagent tools listed and list_agents executed; no child LLM activation requested.';
        evidence.workflowScope = 'Real native workflow engine executed a zero-agent script; no LLM requests.';
        evidence.matlabScope = 'Loaded the installed Simulink skill only; no MATLAB/MCP call or S02 work.';
        evidence.status = 'PASS';
        return { kind: 'success', text: JSON.stringify(evidence) };
      } catch (error) {
        evidence.status = 'FAIL';
        evidence.error = error instanceof Error ? error.message : String(error);
        return { kind: 'success', text: JSON.stringify(evidence) };
      }
    },
  });
}

async function main() {
  const options = smokeOptions('smoke-native-capabilities');
  if (options.help) return;
  await mkdir(options.workspace, { recursive: true });
  const fixture = resolve(options.workspace, fixtureName);
  const fixtureText = `${marker}\n6 * 7 = 42\n`;
  try { await writeFile(fixture, fixtureText, { flag: 'wx' }); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    assert.equal(await readFile(fixture, 'utf8'), fixtureText, 'The selected workspace already contains a different fixture; preserve it and choose another workspace.');
  }
  const evidence = { checkedAt: new Date().toISOString(), workspace: options.workspace, status: 'RUNNING',
    scope: 'Real authenticated Web RPC, native role Sessions, tool executor and providers; no model calls or MATLAB execution.', roles: [] };
  await writeSmokeEvidence(options.evidence, evidence);
  try {
    const rpc = await connectHarness({ logPath: options.webLog });
    const roster = await rpc('agentPresets/list', {});
    for (const role of Object.keys(entries)) {
      const preset = roster.presets.find(row => row.id === `control-${role}`);
      assert.ok(preset && !preset.broken, `Role preset ${role} is unavailable or broken`);
      const created = await rpc('session/create', { request: { cwd: options.workspace, agentPreset: preset.id } });
      const commands = await rpc('commands/list', { agentId: created.sessionId });
      assert.ok(commands.some(row => row.name === 'native-capability-smoke'), 'Load this script as the optional S01 acceptance Cordis plugin before running this smoke.');
      const row = await command(rpc, created.sessionId, '/native-capability-smoke');
      evidence.roles.push(row);
      await writeSmokeEvidence(options.evidence, evidence);
      assert.equal(row.status, 'PASS', row.error);
      const page = await rpc('session/events', { request: { address: { kind: 'session', sessionId: created.sessionId }, maxMessages: 100 } });
      row.eventTypes = [...new Set(page.records.map(record => record.event.type))];
      assert.ok(!row.eventTypes.some(type => type.startsWith('turn/')), 'A non-LLM acceptance command opened a model turn');
      await writeSmokeEvidence(options.evidence, evidence);
    }
    evidence.status = 'PASS';
  } catch (error) {
    evidence.status = 'FAIL';
    evidence.error = error instanceof Error ? error.message : String(error);
    throw error;
  } finally {
    await writeSmokeEvidence(options.evidence, evidence);
    console.log(JSON.stringify({ status: evidence.status, evidence: options.evidence, roles: evidence.roles.map(row => ({ role: row.role, status: row.status, calls: row.calls.length })) }));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
