/** Isolated real native job/subprocess services; no Web port or model API. */
import { resolve, join, relative, isAbsolute, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { apply } from '../src/index.mjs';
import { confirmationRequirements } from '../src/case-requirements.mjs';

export const project = resolve(import.meta.dirname, '../..');
const upstream = resolve(project, 'vendor/deepseek-harness');
const built = async path => import(pathToFileURL(resolve(upstream, path)));

export async function createHost(workspace, owner = 'durable-test-session', { confirmFixturePlans = true, policy, pythonPath } = {}) {
  const { Context } = await built('vendor/cordis/lib/index.js');
  const { Session } = await built('packages/core/session/lib/index.js');
  const { default: Projection } = await built('packages/session/session-projection/lib/index.js');
  const { default: Agents } = await built('packages/core/agent/lib/index.js');
  const { default: Jobs } = await built('packages/jobs/jobs-local/lib/index.js');
  const { default: Subprocess } = await built('packages/subprocess/subprocess-local/lib/index.js');
  const { default: Commands } = await built('packages/interaction/commands/lib/index.js');
  const ctx = new Context();
  const fibers = [];
  for (const service of [Projection, Agents, Jobs, Subprocess, Commands]) {
    const fiber = ctx.plugin(service); fibers.push(fiber); await fiber;
  }
  ctx.jobs.attachController('durable-history-test');
  const agentFiber = ctx.plugin(() => {});
  const initial = Session.create(owner);
  const session = policy ? Session.create(owner, undefined, {...initial.header, cwd:workspace}) : initial;
  const agent = { id: owner, session, ctx: agentFiber.ctx, status: 'idle', options: {},
    model: {}, inbox: {}, send() {}, followup() {}, inject() {}, cancel() {}, whenIdle: async () => {},
    runMaintenance: fn => fn(new AbortController().signal) };
  const unregister = await ctx.agents.register(agent);
  // Native commands/projections preserve the real user-source receipt seam.
  // Only the unused model-facing tool registry is captured in this job fixture.
  apply({ jobs: ctx.jobs, subprocess: ctx.subprocess, logger: ctx.logger, on: ctx.on.bind(ctx), provide: ctx.provide.bind(ctx),
    commands: ctx.commands, sessionProjections: ctx.sessionProjections, tools: { register() {} }, attachments: {}, sandbox: {},
    sandboxPolicy: { resolve: policy ?? (() => ({ mode: 'danger-full-access', workspaceRoot: workspace })) },
  }, { codeRoot: project, pythonPath: pythonPath ?? resolve(project, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python'),
    timeoutSeconds: 120, cancellationGraceMs: 5000, maxConcurrentWorkers: 1 });

  const nativeCommand = async (name, args) => {
    const result = await ctx.commands.execute(agent, `/${name} ${typeof args === 'string' ? args : JSON.stringify(args)}`, [], new AbortController().signal);
    return result.result;
  };
  async function prepareFixtureConfirmation(args) {
    const request = { operation: args.operation, workspace: args.workspace, inputs: args.inputs ?? {}, config: args.config ?? {} };
    const requirements = await confirmationRequirements(request, agent);
    if (!requirements.length) return args;
    const distance = relative(resolve(project, 'test-output'), resolve(workspace));
    if (isAbsolute(distance) || distance === '..' || distance.startsWith(`..${sep}`)) throw new Error('History confirmation preparation is limited to isolated test-output workspaces');
    const directory = join(workspace, '.case-plans');
    await mkdir(directory, { recursive: true });
    const marker = { kind: 'control-case-plan-test-fixture-v1', workspace: resolve(workspace) };
    const markerPath = join(directory, 'test-fixture.json');
    try { await writeFile(markerPath, JSON.stringify(marker), { flag: 'wx' }); }
    catch (error) {
      if (error.code !== 'EEXIST') throw error;
      if (JSON.stringify(JSON.parse(await readFile(markerPath, 'utf8'))) !== JSON.stringify(marker)) throw new Error('History fixture marker belongs to a different workspace');
    }
    const python = resolve(project, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
    const source = `import json,sys\nfrom control_harness.case_plans import propose_plan\nx=json.load(sys.stdin); plans=[]; deps={}\nfor row in x['requirements']:\n p=propose_plan(x['workspace'],row['scope'],row['expectedPayload'],row['bindings'],dict(deps)); plans.append(p); deps[row['scope']]={k:p[k] for k in ('plan_id','version_hash')}\nprint(json.dumps(plans))`;
    const proposed = spawnSync(python, ['-c', source], { input: JSON.stringify({ workspace, requirements }),
      env: { ...process.env, PYTHONPATH: resolve(project, 'src'), PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }, encoding: 'utf8', windowsHide: true });
    if (proposed.status !== 0) throw new Error(`History fixture proposal failed: ${proposed.stderr}`);
    const plans = JSON.parse(proposed.stdout), refs = {};
    for (const plan of plans) {
      refs[plan.scope] = { plan_id: plan.plan_id, version_hash: plan.version_hash };
      const confirmation = await nativeCommand('control-confirm', { workspace, ...refs[plan.scope] });
      if (confirmation.kind !== 'success') throw new Error(confirmation.text);
      if (JSON.parse(confirmation.text).confirmation_source !== 'test_fixture') throw new Error('History fixture confirmation must never claim a real human answer');
    }
    return { ...args, config: { ...(args.config ?? {}), plan_refs: refs } };
  }
  return {
    ctx, agent,
    async command(name, args) {
      if (confirmFixturePlans && name === 'control') args = await prepareFixtureConfirmation(typeof args === 'string' ? JSON.parse(args) : args);
      return nativeCommand(name, args);
    },
    async close() { await unregister(); await agentFiber.dispose(); for (const fiber of fibers.reverse()) await fiber.dispose(); },
  };
}

if (process.argv[2] === '--crash-fixture') {
  const workspace = process.argv[3];
  const host = await createHost(workspace);
  const result = await host.command('control', { agent: 'workflow', operation: 'demo', workspace, config: { case: 'linear' } });
  if (result.kind !== 'success') throw new Error(result.text);
  const started = JSON.parse(result.text);
  process.stdout.write(`${JSON.stringify({ type: 'started', ...started })}\n`);
  const interval = setInterval(async () => {
    const response = await host.command('control-status', { workspace, attemptId: started.attemptId });
    const record = JSON.parse(response.text).record;
    if (record.progress?.stage === 'residual_training') {
      clearInterval(interval);
      process.stdout.write(`${JSON.stringify({ type: 'ready_to_crash', ...started })}\n`);
    }
  }, 10);
}
