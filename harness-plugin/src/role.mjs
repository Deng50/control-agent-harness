/** Native preset contribution: scoped persona and professional-tool routing only. */
import z from '@deepseek-ai/schemastery';
import { agents, roleIds, rolePrompt } from './roles.mjs';

export const name = 'control-engineering-role';
export const inject = ['tools', 'systemPrompt', 'controlEngineering'];
export const Config = z.object({ role: z.union(roleIds).required() });

export function apply(ctx, config) {
  if (!roleIds.includes(config.role)) throw new Error(`Unknown control role: ${config.role}`);
  // The host readiness service is published after all control tools register.
  // Native restrictions affect declarations and dispatch, not just prompt text.
  ctx.tools.restrict({ deny: roleIds.filter(role => role !== config.role).map(role => `control_${role}`) });
  ctx.systemPrompt.section({ name: 'deployment:persona-prefix',
    order: ctx.systemPrompt.getSectionOrder('DEPLOYMENT_PERSONA_PREFIX'),
    interpolate: false, text: rolePrompt(config.role) });
  ctx.systemPrompt.section({ name: 'deployment:persona-suffix',
    order: ctx.systemPrompt.getSectionOrder('DEPLOYMENT_PERSONA_SUFFIX'), text: '' });
}
