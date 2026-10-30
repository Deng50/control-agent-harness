/** External Cordis plugin: all model interaction and background jobs stay inside Harness. */
import { isAbsolute } from 'node:path';
import z from '@deepseek-ai/schemastery';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { WorkerSlots, numericalJob } from './worker.mjs';
import { JobHistory, authorizeWorkspace, recoveryRequest, persistedJob } from './job-history.mjs';
import { agents } from './roles.mjs';
import { operationHelp, operations } from './catalog.mjs';

export { agents } from './roles.mjs';

export const name = 'control-engineering';
export const inject = ['tools', 'commands', 'attachments', 'jobs', 'subprocess', 'sandbox', 'sandboxPolicy'];
export const Config = z.object({
  pythonPath: z.string().required(),
  codeRoot: z.string().required(),
  maxConcurrentWorkers: z.natural().min(1).default(1),
  timeoutSeconds: z.number().min(1).default(1800),
  cancellationGraceMs: z.number().min(1000).default(10000),
});

export function apply(ctx, config) {
  if (!isAbsolute(config.pythonPath) || !isAbsolute(config.codeRoot)) {
    throw new Error('Control engineering pythonPath and codeRoot must be absolute paths');
  }
  const slots = new WorkerSlots(config.maxConcurrentWorkers);
  const history = new JobHistory();
  const pendingWrites = new Set();
  const writers = new Map();
  ctx.jobs.events.subscribe({ owners: 'scope' }, event => {
    if (event.type !== 'settled') return;
    const writer = writers.get(event.job.id);
    if (!writer) return;
    writers.delete(event.job.id);
    const write = writer.append({ state: event.job.status, nativeSettlement: {
      status: event.job.status, detail: event.job.detail, cause: event.cause,
    } });
    pendingWrites.add(write);
    write.catch(error => ctx.logger.warn(`Control history settlement could not be saved: ${error.message}`))
      .finally(() => pendingWrites.delete(write));
  });
  ctx.on('dispose', async () => { await Promise.allSettled([...pendingWrites]); });
  // Both the conversational tools and the human form command use this owner.
  // Commands are recorded by Harness and never require an LLM round trip.
  const submit = async (agentName, args, agent, signal, toolCall, preparedWriter) => {
    signal.throwIfAborted();
    const spec = agents[agentName];
    if (!spec || !spec.operations.includes(args.operation)) throw new Error('Operation is not available for this control agent');
    if (typeof args.workspace !== 'string' || !isAbsolute(args.workspace)) throw new Error('workspace must be an absolute path');
    for (const key of ['inputs', 'config']) {
      if (args[key] !== undefined && (args[key] === null || typeof args[key] !== 'object' || Array.isArray(args[key]))) {
        throw new Error(`${key} must be a JSON object`);
      }
    }
    const policy = ctx.sandboxPolicy.resolve({ session: agent.session });
    const workspace = await authorizeWorkspace(args.workspace, policy, true);
    const request = { operation: args.operation, workspace, inputs: args.inputs ?? {}, config: args.config ?? {} };
    const writer = preparedWriter ?? await history.create({ workspace, owner: agent.id, agentName, request });
    let jobId;
    try {
      signal.throwIfAborted();
      jobId = ctx.jobs.start({
      kind: 'control-engineering', label: `${agentName}: ${args.operation}`, owner: agent.id,
      ...(toolCall ? { toolCall } : {}),
      run: job => {
        writers.set(job.id, writer);
        return persistedJob({ writer, job,
          start: onEvent => numericalJob({ ctx, job, slots, config, request, policy, onEvent }),
        });
      },
    }); } catch (error) {
      await writer.append({ state: 'failed', outcome: { status: 'failed', detail: `Native job admission failed: ${error.message}` } });
      throw error;
    }
    await writer.flush();
    return { jobId, attemptId: writer.record.attemptId, workspace,
      ...(writer.record.recoveredFrom ? { recoveredFrom: writer.record.recoveredFrom } : {}) };
  };
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/help', name: 'control-help',
    description: 'Read the operation catalog, required inputs and engineering prerequisites',
    input: { hint: 'Operation name, or JSON {operation?, agent?}; empty lists all operations' },
    handler: ({ rawInput }) => {
      try {
        const text = rawInput.trim();
        const args = text.startsWith('{') ? JSON.parse(text) : text ? { operation: text } : {};
        return { kind: 'success', text: JSON.stringify(operationHelp(args)) };
      } catch (error) { return { kind: 'error', text: error instanceof Error ? error.message : String(error) }; }
    },
  });
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/run', name: 'control',
    description: 'Execute a control engineering form using native Harness jobs',
    input: { hint: 'JSON: {agent, operation, workspace, inputs, config}' },
    handler: async ({ rawInput, agent, signal }) => {
      try {
        const args = JSON.parse(rawInput);
        if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Expected a JSON request object');
        return { kind: 'success', text: JSON.stringify(await submit(args.agent, args, agent, signal)) };
      } catch (error) {
        return { kind: 'error', text: error instanceof Error ? error.message : String(error) };
      }
    },
  });
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/upload', name: 'control-upload',
    description: 'Admit one uploaded model or dataset for a control stage',
    input: { hint: 'Attach one model JSON or transition NPZ', attachments: true },
    handler: ({ attachments }) => {
      if (attachments.length !== 1 || attachments[0].type !== 'file') return { kind: 'error', text: 'Attach exactly one file' };
      const file = attachments[0].attachment;
      const path = ctx.attachments.fileHostPath(file);
      if (!path) return { kind: 'error', text: 'This attachment backend does not expose a local worker-readable file' };
      return { kind: 'success', text: JSON.stringify({ path, name: file.name, bytes: file.bytes, sha256: file.attachmentId }) };
    },
  });
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/status', name: 'control-status',
    description: 'Inspect a control job without consuming the model output cursor',
    input: { hint: 'Native Harness job id or JSON {workspace, attemptId}' },
    handler: async ({ rawInput, agent }) => {
      if (rawInput.trim().startsWith('{')) {
        const args = JSON.parse(rawInput);
        const workspace = await authorizeWorkspace(args.workspace, ctx.sandboxPolicy.resolve({ session: agent.session }));
        const record = await history.read(workspace, args.attemptId, agent.id, ctx.jobs);
        const output = record.native ? ctx.jobs.readAt(record.nativeJobId, 0, agent.id) : null;
        return { kind: 'success', text: JSON.stringify({ record, job: record.native ?? null, output }) };
      }
      const jobId = rawInput.trim();
      const job = ctx.jobs.get(jobId, agent.id);
      const output = ctx.jobs.readAt(jobId, 0, agent.id);
      return { kind: 'success', text: JSON.stringify({ job, output }) };
    },
  });
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/history', name: 'control-history',
    description: 'Read this Session\'s durable control attempts, including interrupted executions',
    input: { hint: 'JSON {workspace}' },
    handler: async ({ rawInput, agent }) => {
      const args = JSON.parse(rawInput);
      const workspace = await authorizeWorkspace(args.workspace, ctx.sandboxPolicy.resolve({ session: agent.session }));
      return { kind: 'success', text: JSON.stringify({ workspace, attempts: await history.list(workspace, agent.id, ctx.jobs) }) };
    },
  });
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/recover', name: 'control-recover',
    description: 'Explicitly retry an interrupted/failed attempt or resume its saved training/workflow state as a new native job',
    input: { hint: 'JSON {workspace, attemptId, mode: retry|resume, checkpoint?}' },
    handler: async ({ rawInput, agent, signal }) => {
      try {
        const args = JSON.parse(rawInput);
        const workspace = await authorizeWorkspace(args.workspace, ctx.sandboxPolicy.resolve({ session: agent.session }), true);
        const previous = await history.read(workspace, args.attemptId, agent.id, ctx.jobs);
        if (previous.recovery) {
          const replacement = await history.read(workspace, previous.recovery.attemptId, agent.id, ctx.jobs);
          return { kind: 'success', text: JSON.stringify({ jobId: replacement.nativeJobId, attemptId: replacement.attemptId,
            workspace, recoveredFrom: previous.attemptId, existing: true, state: replacement.state }) };
        }
        const request = await recoveryRequest(previous, args.mode, args.checkpoint);
        const writer = await history.create({ workspace, owner: agent.id, agentName: previous.agentName,
          request, recoveredFrom: previous.attemptId, recoveryMode: args.mode });
        const winner = await history.claimRecovery(workspace, previous.attemptId, writer.record.attemptId, args.mode);
        if (winner !== writer.record.attemptId) {
          await writer.append({ state: 'killed', recoveryBlocked: `Duplicate recovery; inspect replacement ${winner}`,
            outcome: { status: 'killed', detail: 'Duplicate recovery admission suppressed; no worker started' } });
          const replacement = await history.read(workspace, winner, agent.id, ctx.jobs);
          return { kind: 'success', text: JSON.stringify({ jobId: replacement.nativeJobId, attemptId: winner, workspace,
            recoveredFrom: previous.attemptId, existing: true, state: replacement.state }) };
        }
        return { kind: 'success', text: JSON.stringify(await submit(previous.agentName, request, agent, signal, undefined, writer)) };
      } catch (error) { return { kind: 'error', text: error instanceof Error ? error.message : String(error) }; }
    },
  });
  ctx.tools.register(defineTool({
    name: 'control_help',
    description: 'Read control-engineering operation instructions before submitting an unfamiliar stage. With no operation, lists the catalog. With an operation, returns required inputs, config JSON schemas, prerequisites, output meaning and a request example. Read-only; no workspace, numerical job or model service is needed.',
    parameters: {
      operation: { type: 'string', enum: Object.keys(operations), description: 'Operation to describe in detail; omit for the index.' },
      agent: { type: 'string', enum: Object.keys(agents), description: 'Optional role filter for the index.' },
    },
    output: { schema: { type: 'object', additionalProperties: true }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }] },
    execute: args => operationHelp(args),
  }));
  for (const [agentName, spec] of Object.entries(agents)) {
    ctx.tools.register(defineTool({
      name: `control_${agentName}`,
      description: `${spec.description} First read control_help({operation}) for required inputs, config and prerequisites. Returns a native Harness job id; use job_output/job_list/job_kill to follow or cancel it. Model dimensions/units are validated, and failed acceptance never becomes the active downstream artifact.`,
      parameters: {
        operation: { type: 'string', enum: spec.operations, required: true },
        workspace: { type: 'string', required: true, description: 'Absolute project path containing manifest.json; selected by the user.' },
        inputs: { type: 'object', additionalProperties: true, description: 'Operation-specific required artifact ids/source paths. Obtain the exact inputsSchema from control_help({operation}); do not infer names.' },
        config: { type: 'object', additionalProperties: true, description: 'Operation-specific settings. Obtain configSchema and prerequisites from control_help({operation}); preserve user-confirmed dimensions, units and conditions.' },
      },
      output: {
        schema: { type: 'object', additionalProperties: false,
          properties: { jobId: { type: 'string', required: true }, attemptId: { type: 'string', required: true }, workspace: { type: 'string', required: true } } },
        render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
      },
      execute(args, execution) {
        if (!execution.agent) throw new Error('Control tools require a Harness agent Session');
        return submit(agentName, args, execution.agent, execution.signal,
          { callId: execution.callId, rootCallId: execution.rootCallId });
      },
    }));
  }
  // Preset roles wait for this service so restriction names always exist.
  ctx.provide('controlEngineering', { operations: Object.keys(operations) });
}
