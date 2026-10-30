/** Isolated real native job/subprocess services; no Web port or model API. */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { apply } from '../src/index.mjs';

export const project = resolve(import.meta.dirname, '../..');
const upstream = resolve(project, 'vendor/deepseek-harness');
const built = async path => import(pathToFileURL(resolve(upstream, path)));

export async function createHost(workspace, owner = 'durable-test-session') {
  const { Context } = await built('vendor/cordis/lib/index.js');
  const { Session } = await built('packages/core/session/lib/index.js');
  const { default: Projection } = await built('packages/session/session-projection/lib/index.js');
  const { default: Agents } = await built('packages/core/agent/lib/index.js');
  const { default: Jobs } = await built('packages/jobs/jobs-local/lib/index.js');
  const { default: Subprocess } = await built('packages/subprocess/subprocess-local/lib/index.js');
  const ctx = new Context();
  const fibers = [];
  for (const service of [Projection, Agents, Jobs, Subprocess]) {
    const fiber = ctx.plugin(service); fibers.push(fiber); await fiber;
  }
  ctx.jobs.attachController('durable-history-test');
  const agentFiber = ctx.plugin(() => {});
  const agent = { id: owner, session: Session.create(owner), ctx: agentFiber.ctx, status: 'idle', options: {},
    model: {}, inbox: {}, send() {}, followup() {}, inject() {}, cancel() {}, whenIdle: async () => {},
    runMaintenance: fn => fn(new AbortController().signal) };
  const unregister = await ctx.agents.register(agent);
  const commands = new Map();
  // Only the command/tool registries are captured; native ownership and subprocess cleanup are real.
  apply({ jobs: ctx.jobs, subprocess: ctx.subprocess, logger: ctx.logger, on: ctx.on.bind(ctx), provide: ctx.provide.bind(ctx),
    commands: { register: item => commands.set(item.name, item) }, tools: { register() {} }, attachments: {}, sandbox: {},
    sandboxPolicy: { resolve: () => ({ mode: 'danger-full-access', workspaceRoot: workspace }) },
  }, { codeRoot: project, pythonPath: resolve(project, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python'),
    timeoutSeconds: 120, cancellationGraceMs: 5000, maxConcurrentWorkers: 1 });
  return {
    ctx, agent,
    async command(name, args) {
      return commands.get(name).handler({ rawInput: typeof args === 'string' ? args : JSON.stringify(args),
        agent, signal: new AbortController().signal, attachments: [] });
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
