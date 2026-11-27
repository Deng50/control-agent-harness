/** External Cordis plugin: all model interaction and background jobs stay inside Harness. */
import { isAbsolute } from 'node:path';
import z from '@deepseek-ai/schemastery';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { WorkerSlots, numericalJob } from './worker.mjs';
import { JobHistory, authorizeWorkspace, recoveryRequest, persistedJob } from './job-history.mjs';
import { agents } from './roles.mjs';
import { installLibraryTools } from './library-tools.mjs';
import { operationHelp, operations } from './catalog.mjs';
import { installMatlabResource } from './matlab-resource.mjs';
import { installMatlabRecovery } from './matlab-recovery.mjs';
import { createMatlabTerminalVerifier } from './matlab-terminal.mjs';
import { saveSessionSlx } from './simulink-save.mjs';
import { simulinkPlantTool } from './simulink-plant.mjs';
import { buildZohPlantTool } from './simulink-build.mjs';
import { simulateModelTool } from './simulink-simulate.mjs';
import { installPlantWorkspace } from './plant-workspace.mjs';
import { simulinkWireTool } from './simulink-wire.mjs';
import { MatlabTaskGrants } from './matlab-task-grants.mjs';
import { preflightWorkspace } from './workspace-preflight.mjs';
import { installCasePlans } from './case-plans.mjs';
import { confirmationRequirements } from './case-requirements.mjs';
import { prepareExecution, prepareControlCard, reviewControlRequest } from './review.mjs';
import { installSessionWorkspace, ensureProfessionalPreset, installManagedShellWorkspace } from './session-workspace.mjs';
import { requiresMatlabBatch, assertMatlabBatchAvailable, matlabCollectionJob } from './matlab-job.mjs';

export { agents } from './roles.mjs';

export const name = 'control-engineering';
export const inject = ['tools', 'commands', 'attachments', 'jobs', 'subprocess', 'sandbox', 'sandboxPolicy', 'sessionProjections', 'sessionProjectionCache', 'workspaceRegistry', 'agentPresets', 'agents'];
export const Config = z.object({
  pythonPath: z.string().required(),
  codeRoot: z.string().required(),
  maxConcurrentWorkers: z.natural().min(1).default(1),
  timeoutSeconds: z.number().min(1).default(1800),
  queueTimeoutSeconds: z.number().min(1).max(3600).default(120),
  cancellationGraceMs: z.number().min(1000).default(10000),
  // Omission is Python-only mode; one Host-wide directory must cover every
  // consumer of the same dedicated MATLAB session, including batch/Python.
  matlabResource: z.object({
    directory: z.string().required(),
    toolPrefixes: z.array(z.string()).default(['mcp__matlab__']),
    pollIntervalMs: z.number().min(5).max(5000).default(50),
    identityFile: z.string(),
  }).default(undefined),
});

export function apply(ctx, config) {
  if (!isAbsolute(config.pythonPath) || !isAbsolute(config.codeRoot)) {
    throw new Error('Control engineering pythonPath and codeRoot must be absolute paths');
  }
  const verifyMatlab = config.matlabResource?.identityFile
    ? createMatlabTerminalVerifier(ctx, { identityFile: config.matlabResource.identityFile, codeRoot:config.codeRoot }) : undefined;
  const matlabResource=installMatlabResource(ctx,config,{verifyBeforeDispatch:verifyMatlab,verifyTerminal:verifyMatlab});
  const matlabRecovery = verifyMatlab && installMatlabRecovery(ctx,matlabResource,verifyMatlab);
  // Propose all scopes in one Python transaction. Reviews have their own
  // bounded slot so a long training job cannot block a user trying to review.
  const reviewSlots = new WorkerSlots(1);
  installLibraryTools(ctx, config);
  const preflight = options => preflightWorkspace({ctx,config,...options});
  if (verifyMatlab) {
    const {checkBuild,verifyReference}=installPlantWorkspace(ctx,config,preflight);
    ctx.tools.register(buildZohPlantTool(ctx,config,checkBuild,verifyReference));
    ctx.tools.register(simulateModelTool(ctx,config,preflight));
    ctx.tools.register(simulinkPlantTool(ctx, config));
    ctx.tools.register(simulinkWireTool(ctx, config));
    ctx.tools.register(defineTool({
      name:'control_save_simulink_model',
      description:'Save one loaded top-level Simulink model to this conversation workspace. Returns its verified absolute .slx path and byte count. Request simulation separately after saving.',
      parameters:{model:{type:'string',required:true,description:'Loaded model name, without a path or .slx extension'}},
      output:{schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]},
      execute:(args,execution)=>saveSessionSlx(ctx,config,args,execution,preflight),
    }));
  }
  const matlabGrants = new MatlabTaskGrants(config);
  const proposeBatch = async ({request, intent_id, requirements, agent, signal, retire_scopes = []}) => {
    const timeout = AbortSignal.timeout(30000);
    const stop = signal ? AbortSignal.any([signal, timeout]) : timeout;
    stop.throwIfAborted();
    const policy = ctx.sandboxPolicy.resolve({session:agent.session});
    await preflight({workspace:request.workspace,policy,signal:stop,operation:'propose_case_plans',agent});
    const worker = numericalJob({ctx, slots:reviewSlots, config:{...config, timeoutSeconds:30},
      policy, job:{updateProgress() {}, append() {}},
      request:{operation:'propose_case_plans',workspace:request.workspace,inputs:{},config:{intent_id,retire_scopes,
        requirements:requirements.map(({scope,reviewPayload,reviewBindings,bindings}) => ({scope,reviewPayload,reviewBindings,bindings}))}}});
    stop.addEventListener('abort', worker.cancel, {once:true});
    try {
      const outcome = await worker.done;
      stop.throwIfAborted();
      if (outcome.status !== 'completed') throw new Error(outcome.detail ?? 'Engineering review preparation failed');
      return JSON.parse(outcome.result).plans;
    } finally { stop.removeEventListener('abort', worker.cancel); }
  };
  const casePlans = installCasePlans(ctx, { codeRoot: config.codeRoot, resolveRequirements: confirmationRequirements, proposeBatch, matlabGrants });
  installSessionWorkspace(ctx,preflight);
  installManagedShellWorkspace(ctx,config);
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
  const resourceLimits = grant => ({workerQueueSeconds:config.queueTimeoutSeconds ?? 120,
    workerSeconds:config.timeoutSeconds, matlabSeconds:grant?.scope.dispatch_timeout_seconds});
  async function selectCardVersions(request, card_id, agent, role) {
    const card = await casePlans.cardStatus({workspace:request.workspace, card_id, agent, role});
    const continuation = card.operation === 'prepare_supervised_dataset' && request.operation === 'train_supervised' && card.plan_refs.training;
    if (card.operation !== request.operation && !continuation) throw new Error('The selected review card does not cover this stage');
    if (Object.values(card.statuses).some(status => status.confirmation_state !== 'confirmed')) {
      throw new Error('本卡仍待用户确认，请核对当前版本并点击“确认以上摘要”。');
    }
    const refs = request.operation === 'prepare_supervised_dataset' ? {signals:card.plan_refs.signals} : card.plan_refs;
    for (const [scope, ref] of Object.entries(request.config.plan_refs ?? {})) {
      const selected = card.plan_refs[scope];
      if (!Object.hasOwn(card.plan_refs,scope) || ref?.plan_id !== selected.plan_id || ref?.version_hash !== selected.version_hash) {
        throw new Error('The supplied plan_refs differ from the selected review card');
      }
    }
    request.config = {...request.config,plan_refs:refs};
    return card;
  }
  async function admitResources(request, agent, policy, signal, card) {
    await preflight({workspace:request.workspace,policy,signal,operation:request.operation,agent});
    if (!await requiresMatlabBatch(request)) return null;
    assertMatlabBatchAvailable(ctx,config,agent);
    if (!card?.matlab_scope) throw new Error('MATLAB_TASK_GRANT_REQUIRED：请为本次操作生成确认卡，核对专用 MATLAB 的范围并点击授权。');
    return matlabGrants.reserve({request,agent,card});
  }
  async function admit(agentName,args,agent,signal) {
    signal.throwIfAborted();
    const policy = ctx.sandboxPolicy.resolve({ session: agent.session });
    const request = await prepareExecution({...args,agent:agentName}, policy, {structured:args.card_id !== undefined});
    if (args.next_training !== undefined) throw new Error('next_training belongs to the review card only');
    await ensureProfessionalPreset(ctx, agent);
    const card = args.card_id !== undefined ? await selectCardVersions(request,args.card_id,agent,agentName) : undefined;
    const confirmation = await casePlans.authorizeRequest({ request, agent, role:agentName });
    if (args.card_id !== undefined && !confirmation) throw new Error('This request no longer uses the reviewed engineering scopes');
    const grant = await admitResources(request,agent,policy,signal,card);
    return {request,agentName,agent,policy,confirmation,grant};
  }

  async function startAdmitted(admission,signal,toolCall,writer,collectionResume) {
    const {request,agentName,agent,policy,confirmation,grant}=admission;
    const {workspace}=request;
    let jobId;
    try {
      signal.throwIfAborted();
      writer ??= await history.create({ workspace, owner: agent.id, agentName, request, limits:resourceLimits(grant) });
      jobId = ctx.jobs.start({
      kind: 'control-engineering', label: `${agentName}: ${request.operation}`, owner: agent.id,
      ...(toolCall ? { toolCall } : {}),
      run: job => {
        writers.set(job.id, writer);
        let started = false;
        const hooks = persistedJob({ writer, job,
          start: onEvent => {
            const worker = grant
              ? matlabCollectionJob({ctx,job,agent,slots,config,request,policy,onEvent,confirmation,grant})
              : numericalJob({ctx,job,slots,config,request,policy,onEvent,confirmation,collectionResume});
            started = true;
            return worker;
          },
        });
        return {cancel:hooks.cancel,done:hooks.done.finally(() => { if (!started) grant?.revoke('job_start_failed'); })};
      },
    }); } catch (error) {
      grant?.revoke('job_start_failed');
      await writer?.append({ state: 'failed', outcome: { status: 'failed', cleanupVerified:true, detail: `Native job admission failed: ${error.message}` } });
      throw error;
    }
    await writer.flush();
    return { jobId, attemptId: writer.record.attemptId, workspace,
      ...(writer.record.recoveredFrom ? { recoveredFrom: writer.record.recoveredFrom } : {}) };
  }
  const submit = async (agentName,args,agent,signal,toolCall) =>
    startAdmitted(await admit(agentName,args,agent,signal),signal,toolCall);
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
    definitionId:'@control-engineering/harness-bundle/preflight',name:'control-preflight',
    description:'Inspect selected artifact versions and engineering interfaces before review or execution',
    handler:async ({rawInput,agent})=>{
      try {const result=await prepareExecution(JSON.parse(rawInput),ctx.sandboxPolicy.resolve({session:agent.session}),{probe:true});
        return {kind:'success',text:JSON.stringify(result)};}
      catch(error){return {kind:'error',text:error.message};}
    },
  });
  ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/review', name: 'control-review',
    description: 'Read exact signal/sampling/training summaries required by this request; does not propose, confirm or execute',
    input: { hint: 'JSON: {agent, operation, workspace, inputs, config}' },
    handler: async ({rawInput, agent, signal}) => {
      try {
        signal.throwIfAborted();
        const value = await reviewControlRequest(JSON.parse(rawInput), ctx.sandboxPolicy.resolve({session:agent.session}));
        signal.throwIfAborted();
        return {kind:'success', text:JSON.stringify(value)};
      } catch (error) { return {kind:'error', text:error instanceof Error ? error.message : String(error)}; }
    },
  });
  ctx.commands.register({
    definitionId:'@control-engineering/harness-bundle/prepare-review',name:'control-prepare-review',
    description:'Prepare a versioned engineering review card in this Session',
    handler:async ({rawInput,agent,signal}) => {
      try {
        await ensureProfessionalPreset(ctx, agent);
        const card = await prepareControlCard(JSON.parse(rawInput), ctx.sandboxPolicy.resolve({session:agent.session}), casePlans, agent, signal);
        return {kind:'success',text:JSON.stringify(card)};
      } catch (error) {return {kind:'error',text:error.message};}
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
    input: { hint: 'Attach one Python/MATLAB/SLX/MDL model, CSV/XLSX dataset, JSON or transition NPZ', attachments: true },
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
    input: { hint: 'Use the attempt in the sidebar. For migration select its signed card in the same Session; the legacy plan_refs selector is also supported.' },
    handler: async ({ rawInput, agent, signal }) => {
      try {
        const args = JSON.parse(rawInput);
        const workspace = await authorizeWorkspace(args.workspace, ctx.sandboxPolicy.resolve({ session: agent.session }), true);
        await ensureProfessionalPreset(ctx, agent);
        const previous = await history.read(workspace, args.attemptId, agent.id, ctx.jobs);
        if (previous.recovery) {
          const replacement = await history.read(workspace, previous.recovery.attemptId, agent.id, ctx.jobs);
          return { kind: 'success', text: JSON.stringify({ jobId: replacement.nativeJobId, attemptId: replacement.attemptId,
            workspace, recoveredFrom: previous.attemptId, existing: true, state: replacement.state }) };
        }
        const recovered = await recoveryRequest(previous, args.mode, args.checkpoint);
        if (args.plan_refs !== undefined && (!args.plan_refs || typeof args.plan_refs !== 'object' || Array.isArray(args.plan_refs))) throw new Error('plan_refs must be an object selecting current confirmed plan versions');
        if (args.card_id !== undefined && args.plan_refs !== undefined) throw new Error('Select one confirmation card or legacy plan versions for recovery');
        const selection = { operation: recovered.operation, workspace, inputs: recovered.inputs ?? {},
          config: { ...(recovered.config ?? {}), ...(args.plan_refs === undefined ? {} : { plan_refs: args.plan_refs }) } };
        // Admission is read-only and precedes every durable recovery mutation.
        // A legacy request may select newly confirmed plan refs, never mint them.
        if (args.card_id !== undefined) {
          delete selection.config.plan_refs;
          selection.card_id=args.card_id;
        }
        const admission=await admit(previous.agentName,selection,agent,signal);
        const {request,grant}=admission;
        const collectionResume = args.mode === 'resume' && ['sample_model','collect_external_expert'].includes(request.operation)
          ? previous.collectionCheckpoint : undefined;
        if (grant && collectionResume) { grant.revoke('job_start_failed'); throw new Error('MATLAB 批任务需要完整且已核验的交换结果。'); }
        let writer, winner;
        try {
          signal.throwIfAborted();
          writer = await history.create({ workspace, owner: agent.id, agentName: previous.agentName,
            request, recoveredFrom: previous.attemptId, recoveryMode: args.mode, limits:resourceLimits(grant) });
          winner = await history.claimRecovery(workspace, previous.attemptId, writer.record.attemptId, args.mode);
        } catch (error) {
          grant?.revoke('job_start_failed');
          await writer?.append({state:'failed',outcome:{status:'failed',cleanupVerified:true,
            detail:`Recovery admission failed before a worker started: ${error.message}`}});
          throw error;
        }
        if (winner !== writer.record.attemptId) {
          grant?.revoke('job_start_failed');
          await writer.append({ state: 'killed', recoveryBlocked: `Duplicate recovery; inspect replacement ${winner}`,
            outcome: { status: 'killed', cleanupVerified:true, detail: 'Duplicate recovery admission suppressed; no worker started' } });
          const replacement = await history.read(workspace, winner, agent.id, ctx.jobs);
          return { kind: 'success', text: JSON.stringify({ jobId: replacement.nativeJobId, attemptId: winner, workspace,
            recoveredFrom: previous.attemptId, existing: true, state: replacement.state }) };
        }
        return { kind: 'success', text: JSON.stringify(await startAdmitted(admission,signal,undefined,writer,collectionResume)) };
      } catch (error) { return { kind: 'error', text: error instanceof Error ? error.message : String(error) }; }
    },
  });
  ctx.tools.register(defineTool({
    name: 'control_preflight',
    description: '检查目标操作、所选产物版本、类型和信号接口，返回具体字段和补充材料。只读检查，不启动作业或产生确认。',
    parameters:{agent:{type:'string',enum:Object.keys(agents),required:true},operation:{type:'string',enum:Object.keys(operations),required:true},
      workspace:{type:'string',required:true},inputs:{type:'object',additionalProperties:true},config:{type:'object',additionalProperties:true},
      config_file:{type:'string',description:'Lightweight only: workspace-root JSON configuration filename, instead of inline config.'}},
    output:{schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]},
    execute:(args,{agent})=>prepareExecution(args,ctx.sandboxPolicy.resolve({session:agent.session}),{probe:true}),
  }));
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
  ctx.tools.register(defineTool({
    name:'control_prepare_review',
    description:'Prepare the exact engineering request as a clickable human review card in this Session. Use the same intent_id for edits; choose a distinct task label for independent work. For spreadsheet preparation include next_training only when the user wants to train. Wait for the user to click, then submit the selected operation and card_id. Preparing the card does not confirm or start the work.',
    parameters:{
      agent:{type:'string',enum:Object.keys(agents),required:true}, operation:{type:'string',enum:Object.keys(operations),required:true},
      workspace:{type:'string',required:true},inputs:{type:'object',additionalProperties:true},config:{type:'object',additionalProperties:true},
      config_file:{type:'string',description:'Lightweight only: workspace-root JSON configuration filename, instead of inline config. Its full contents are reviewed.'},
      intent_id:{type:'string',description:'Task label or returned handle. Reuse across revisions of the same task.'},
      next_training:{type:'object',additionalProperties:false,properties:{config:{type:'object',additionalProperties:true,required:true}},description:'Optional frozen train_supervised settings to review together with dataset preparation'},
    },
    output:{schema:{type:'object',additionalProperties:true},render:(_args,card)=>[{type:'text',text:JSON.stringify({
      workspace:card.workspace,role:card.agent,operation:card.operation,card_id:card.card_id,intent_id:card.intent_id,
      confirmation_scope_count:card.confirmation_scope_count,statuses:Object.fromEntries(Object.entries(card.statuses).map(([scope,s]) => [scope,s.confirmation_state])),
    })}]},
    execute:async (args,{agent,signal}) => {
      if (!agent) throw new Error('Engineering review requires a Harness Session');
      await ensureProfessionalPreset(ctx, agent);
      return prepareControlCard(args,ctx.sandboxPolicy.resolve({session:agent.session}),casePlans,agent,signal);
    },
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
        ...(agentName === 'lightweight' ? {config_file:{type:'string',description:'Use the same workspace-root JSON filename reviewed for this job, instead of inline config.'}} : {}),
        card_id: {type:'string',description:'Returned structured review card from this Session. The Host selects and verifies the exact approved versions.'},
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
  ctx.provide('controlEngineering', { operations: Object.keys(operations), matlabRecovery });
}
