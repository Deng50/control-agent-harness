/** Normalize the same request at review, admission and recovery. */
import { isAbsolute } from 'node:path';
import { validateJsonSchemaValue } from '@deepseek-ai/dsh-tools';
import { agents } from './roles.mjs';
import { operations } from './catalog.mjs';
import { authorizeWorkspace } from './job-history.mjs';
import { confirmationRequirements, resolveWorkflowCase, stagePreflight } from './case-requirements.mjs';
import { readWorkspaceBytes } from './workspace-file.mjs';

const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);

function validateSettings(operation, config) {
  const schema = operations[operation].configSchema;
  // plan_refs are selected by the Host from the reviewed card. Older numerical
  // configurations can still carry them explicitly.
  const {plan_refs, ...settings} = config;
  const required = (schema.required ?? []).filter(key => key !== 'plan_refs');
  const errors = validateJsonSchemaValue({...schema, required}, settings, 'config');
  if (errors.length) throw new Error(errors.join('\n'));
}

export async function prepareExecution(args, policy, {structured = false, probe = false} = {}) {
  if (!object(args)) throw new Error('Expected a JSON request object');
  if (!Object.hasOwn(agents, args.agent) || !agents[args.agent].operations.includes(args.operation)) {
    throw new Error('Operation is not available for this control agent');
  }
  if (typeof args.workspace !== 'string' || !isAbsolute(args.workspace)) throw new Error('workspace must be an absolute path');
  for (const key of ['inputs', 'config']) {
    if (args[key] !== undefined && !object(args[key])) throw new Error(`${key} must be a JSON object`);
  }
  const workspace = await authorizeWorkspace(args.workspace, policy, true);
  const inputs = args.inputs ?? {};
  let config = args.config ?? {};
  if (args.config_file !== undefined) {
    if (args.agent !== 'lightweight' || args.config !== undefined || typeof args.config_file !== 'string' ||
        !/^[^<>:"/\\|?*\x00-\x1f]+\.json$/i.test(args.config_file)) {
      throw new Error('config_file requires one workspace JSON filename for the lightweight role, in place of inline config');
    }
    const bytes = await readWorkspaceBytes(workspace, args.config_file, 2 * 1024 * 1024);
    config = JSON.parse(new TextDecoder('utf-8', {fatal:true}).decode(bytes));
    if (!object(config) || Object.hasOwn(config,'plan_refs')) throw new Error('config_file must contain one operation config object without plan_refs');
    validateSettings(args.operation, config);
  }
  const request = {operation:args.operation,workspace,inputs,config};
  const preflight = await stagePreflight(request);
  if (probe) {
    if (['plan_workflow','run_workflow'].includes(request.operation)) {
      try {await resolveWorkflowCase(request);}
      catch(error) {
        const issue=error.issue ?? {status:error.code==='ENOENT' || /changed|stale or rejected/.test(error.message)?'stale_dependency':'clarify',
          field:request.inputs.workflow_path?'inputs.workflow_path':'config.steps',message:error.message};
        preflight.issues.push(issue);preflight.status=issue.status;
      }
    }
    const errors=validateJsonSchemaValue(operations[args.operation].inputsSchema,inputs,'inputs');
    try {validateSettings(args.operation,config);} catch(error) {errors.push(error.message);}
    for(const message of errors) {
      const field=message.match(/(?:inputs|config)(?:\.[\w-]+)*/)?.[0] ?? 'config';
      if(!preflight.issues.some(i=>i.field===field)) preflight.issues.push({status:'clarify',field,message});
    }
    if(preflight.issues.length && preflight.status==='ready') preflight.status='clarify';
    return preflight;
  }
  if (preflight.status !== 'ready') throw new Error(preflight.status+' '+preflight.issues.map(i=>i.field+': '+i.message).join('\n'));
  if (structured) {
    const errors = validateJsonSchemaValue(operations[args.operation].inputsSchema, inputs, 'inputs');
    if (errors.length) throw new Error(errors.join('\n'));
    validateSettings(args.operation, config);
  }
  return request;
}

export async function reviewControlRequest(args, policy) {
  const request = await prepareExecution(args, policy);
  const requirements = await confirmationRequirements(request);
  return {agent:args.agent, operation:request.operation, workspace:request.workspace, requirements,
    confirmation_scope_count:requirements.length, authority:'read_only_requirements_not_confirmation', jobs_started:0};
}

export async function prepareControlCard(args, policy, authority, agent, signal) {
  const request = await prepareExecution(args, policy, {structured:true});
  const role = args.agent;
  const preset = agent.session.header?.agentPreset;
  if (Object.values(agents).some(spec => spec.preset && spec.preset === preset) &&
      role !== 'workflow' && agents[role].preset !== preset) throw new Error('The review role differs from the professional Session');
  const intent_id = authority.intentFor({agent, role, intent_id:args.intent_id});
  const derivationKey = request.operation === 'train_supervised' ? await authority.key() : undefined;
  const requirements = await confirmationRequirements(request, agent, {intent_id, derivationKey});
  const preparation = request.operation === 'prepare_supervised_dataset';
  if (args.next_training !== undefined) {
    if (!preparation || !object(args.next_training) || Object.keys(args.next_training).length !== 1 || !object(args.next_training.config)) {
      throw new Error('next_training requires a supervised dataset preparation and an explicit training config');
    }
    const training = args.next_training.config;
    if (['plan_refs','resume_checkpoint','resume_checkpoint_sha256','resume_run'].some(key => Object.hasOwn(training,key))) {
      throw new Error('A planned first training stage cannot select existing approvals or a recovery checkpoint');
    }
    validateSettings('train_supervised', training);
    const signals = requirements.find(row => row.scope === 'signals');
    requirements.push({scope:'training', expectedPayload:training, bindings:signals.bindings,
      reviewPayload:{operation:'train_supervised',config:training,preparation:signals.reviewPayload.config},
      reviewBindings:signals.reviewBindings});
  }
  signal.throwIfAborted();
  if (!requirements.length) return {workspace:request.workspace,agent:role,role,operation:request.operation,intent_id,
    card_id:null,plan_refs:{},requirements:[],statuses:{},confirmation_scope_count:0,jobs_started:0};
  const card = await authority.prepareCard({request, agent, role, intent_id, requirements, signal,
    retire_scopes:preparation && !args.next_training ? ['training'] : []});
  return {...card,workspace:request.workspace,agent:role,operation:request.operation,confirmation_scope_count:requirements.length,jobs_started:0};
}
