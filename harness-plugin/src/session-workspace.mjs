/** Attach only the invoking native Session through Harness's own Workspace API. */
import { isAbsolute } from 'node:path';
import { authorizeWorkspace } from './job-history.mjs';

const professionalPresets = new Set(['control-model', 'control-controller', 'control-lightweight']);

/** Restore a creation-time professional role using only the native selection producer. */
export async function ensureProfessionalPreset(ctx, agent) {
  if (!agent?.session || agent.id !== agent.session.id) throw new Error('Native Session identity does not match its Agent');
  const expected = agent.session.header?.agentPreset;
  // Ordinary native Sessions may still use the shared domain tools explicitly.
  if (!professionalPresets.has(expected)) return { managed: false };
  const cwd = agent.session.header?.cwd;
  const canonicalCwd = await authorizeWorkspace(cwd, ctx.sandboxPolicy.resolve({ session: agent.session }), true);
  // Native blank restoration compares the immutable header cwd to the native
  // Workspace path exactly. A legacy slash/case/alias spelling can therefore
  // fall through to another blank's default preset even with a valid cache.
  // Keep that historical Session intact; only an explicit user migration may
  // replace its professional entry with a new canonical Workspace-id Session.
  if (cwd !== canonicalCwd) {
    throw new Error('CONTROL_PROFESSIONAL_SESSION_CANONICAL_WORKSPACE_REQUIRED：旧专业会话保存的路径写法与原生工作区不一致，刷新可能切到其他空白会话。请点击“新建专业会话并保留旧记录”；旧任务和确认记录仍可读取，新会话需重新审阅确认。');
  }
  const registry = ctx.agentPresets;
  const before = { recorded: registry.presetForSession(agent.session), composed: registry.composedPreset(agent.ctx) };
  if (before.recorded !== expected || before.composed !== expected) {
    try { await registry.select(agent, expected); }
    catch (error) {
      if (error.code === 'agent-preset/locked') throw new Error('CONTROL_PROFESSIONAL_SESSION_LOCKED：原专业会话的预设已改变且会话已经开始。请点击“新建专业会话并保留旧记录”；旧任务和确认记录仍保留，不能自动迁移授权。', { cause: error });
      throw error;
    }
  }
  if (registry.presetForSession(agent.session) !== expected || registry.composedPreset(agent.ctx) !== expected) throw new Error('Native professional preset selection did not settle');
  // The native blank-Session restore path reads the cold listing's projection
  // hint. A correct creation header/runtime alone cannot guarantee that the
  // cache's asynchronous creation/write-behind checkpoint has landed. Use its
  // official durability barrier before navigation or any domain admission;
  // selecting the same preset is a native no-op and cannot seal that gap.
  const keys = new Set(['agentPreset']);
  await ctx.sessionProjectionCache.write(agent.session);
  if (ctx.sessionProjectionCache.cachedSnapshot(agent.session.header, keys)?.values.agentPreset !== expected) {
    throw new Error('Native professional preset cold-list hint was not durably recorded');
  }
  return { managed: true, agentPreset: expected, repaired: before.recorded !== expected || before.composed !== expected, durable_hint: true };
}

export async function ensureSessionWorkspace(ctx, agent, args = {}, preflight, signal) {
  if (!args || typeof args !== 'object' || Array.isArray(args) || Object.keys(args).length) {
    throw new Error('control-workspace accepts no arguments; only the invoking native Session can be attached');
  }
  if (!agent?.session || agent.id !== agent.session.id) throw new Error('Native Session identity does not match its Agent');
  const cwd = agent.session.header?.cwd;
  if (typeof cwd !== 'string' || !isAbsolute(cwd)) throw new Error('Native Session has no stored absolute working directory');
  const policy = ctx.sandboxPolicy.resolve({ session: agent.session });
  const workspacePath = await authorizeWorkspace(cwd, policy, true);
  const diagnostic = await preflight?.({workspace:workspacePath,policy,signal,agent});
  const preset = await ensureProfessionalPreset(ctx, agent);
  const workspace = await ctx.workspaceRegistry.create(workspacePath);
  // The native entity independently checks the Session's immutable stored cwd.
  await workspace.attachSession(agent.id);
  if (!workspace.sessionIds.includes(agent.id)) throw new Error('Native Workspace attachment was not projected');
  return { sessionId: agent.id, workspaceId: workspace.id, workspace: workspace.path, attached: true, preset,
    ...(diagnostic ? {preflight:diagnostic} : {}) };
}

export function installSessionWorkspace(ctx, preflight) {
  return ctx.commands.register({
    definitionId: '@control-engineering/harness-bundle/workspace', name: 'control-workspace',
    description: 'Attach this existing Session to its native Workspace without creating another Session or starting work',
    input: { hint: 'No arguments; uses only this Session and its stored working directory' },
    handler: async ({ agent, rawInput, signal }) => {
      try {
        const args = rawInput.trim() ? JSON.parse(rawInput) : {};
        return { kind: 'success', text: JSON.stringify(await ensureSessionWorkspace(ctx, agent, args, preflight, signal)) };
      } catch (error) { return { kind: 'error', text: error instanceof Error ? error.message : String(error) }; }
    },
  });
}
