/** Read the exact existing admission requirements without minting authority. */
import {isAbsolute} from 'node:path';
import {agents} from './roles.mjs';
import {authorizeWorkspace} from './job-history.mjs';
import {confirmationRequirements} from './case-requirements.mjs';

export async function reviewControlRequest(args, policy) {
  if (!args || typeof args !== 'object' || Array.isArray(args)) throw new Error('Expected a JSON request object');
  if (!agents[args.agent]?.operations.includes(args.operation)) throw new Error('Operation is not available for this control agent');
  if (typeof args.workspace !== 'string' || !isAbsolute(args.workspace)) throw new Error('workspace must be an absolute path');
  for (const key of ['inputs', 'config']) {
    if (args[key] !== undefined && (!args[key] || typeof args[key] !== 'object' || Array.isArray(args[key]))) throw new Error(`${key} must be a JSON object`);
  }
  const workspace = await authorizeWorkspace(args.workspace, policy, true);
  const requirements = await confirmationRequirements({operation:args.operation, workspace, inputs:args.inputs ?? {}, config:args.config ?? {}});
  return {agent:args.agent, operation:args.operation, workspace, requirements, confirmation_scope_count:requirements.length,
    authority:'read_only_requirements_not_confirmation', jobs_started:0};
}
