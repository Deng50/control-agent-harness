/** Engineering confirmation authority: native user commands, versioned plans and worker receipts. */
import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto';
import { mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { authorizeWorkspace } from './job-history.mjs';

export const planScopes = ['signals', 'sampling', 'training'];
const sha256 = text => createHash('sha256').update(text, 'utf8').digest('hex');
const inside = (root, target) => { const path = relative(root, target); return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith(`..${sep}`)); };
const signature = (key, body) => createHmac('sha256', key).update(body, 'utf8').digest('hex');
const jsonTransportValue = value => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const sameJsonValue = (left, right) => isDeepStrictEqual(jsonTransportValue(left), jsonTransportValue(right));
export class PendingPlanConfirmation extends Error { constructor(message) { super(message); this.name = 'PendingPlanConfirmation'; } }

function verifySigned(record, key) {
  if (typeof record?.body !== 'string' || !/^[a-f0-9]{64}$/.test(record?.signature ?? '')) throw new PendingPlanConfirmation('Invalid stored native confirmation');
  const expected = Buffer.from(signature(key, record.body), 'hex');
  if (!timingSafeEqual(Buffer.from(record.signature, 'hex'), expected)) throw new PendingPlanConfirmation('Stored confirmation was not created by this Host authority');
  return JSON.parse(record.body);
}

async function planDirectory(workspace) {
  const root = await realpath(workspace), directory = await realpath(join(root, '.case-plans'));
  if (!inside(root, directory)) throw new Error('Case plan directory escapes its workspace');
  return { root, directory };
}

export async function readCasePlan(workspace, reference, scope) {
  if (!reference || !/^[A-Za-z0-9_-]{1,100}$/.test(reference.plan_id ?? '') || !/^[a-f0-9]{64}$/.test(reference.version_hash ?? '')) {
    throw new PendingPlanConfirmation(`Missing current config.plan_refs.${scope}`);
  }
  let directory;
  try { ({ directory } = await planDirectory(workspace)); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    throw new PendingPlanConfirmation(`The ${scope} plan is pending; propose and review it before creating a numerical job`);
  }
  const document = JSON.parse(await readFile(join(directory, 'plans.json'), 'utf8'));
  const record = document.plans?.[reference.plan_id];
  if (document.schema_version !== 1 || !record || record.scope !== scope || record.status === 'stale'
    || record.version_hash !== reference.version_hash || typeof record.canonical_payload_json !== 'string'
    || sha256(record.canonical_payload_json) !== record.version_hash) {
    throw new PendingPlanConfirmation(`The ${scope} plan version changed; review and confirm the current summary`);
  }
  const envelope = JSON.parse(record.canonical_payload_json);
  if (!isDeepStrictEqual(envelope, Object.fromEntries(['scope', 'payload', 'bindings', 'dependencies'].map(key => [key, record[key]])))) {
    throw new PendingPlanConfirmation('Plan payload does not match its Python-owned canonical version');
  }
  for (const [dependency, ref] of Object.entries(record.dependencies)) {
    const active = document.plans?.[document.active_by_scope?.[dependency]];
    if (!active || active.plan_id !== ref.plan_id || active.version_hash !== ref.version_hash || active.status === 'stale') {
      throw new PendingPlanConfirmation(`The upstream ${dependency} plan changed`);
    }
  }
  return record;
}

async function loadHostKey(codeRoot) {
  const directory = join(codeRoot, '.dsh');
  await mkdir(directory, { recursive: true });
  const path = join(directory, 'case-plan-authority.key');
  try { await writeFile(path, randomBytes(32), { flag: 'wx', mode: 0o600 }); }
  catch (error) { if (error.code !== 'EEXIST') throw error; }
  const key = await readFile(path);
  if (key.length !== 32) throw new Error('Invalid Host case-plan authority key');
  return key;
}

function userCommandProjection(codeRoot) {
  const require = createRequire(resolve(codeRoot, 'vendor/deepseek-harness/packages/interaction/commands/package.json'));
  const { z } = require('zod');
  return { key: 'controlCaseConfirmationCommands', stateVersion: 1,
    stateSchema: z.object({ ownStart: z.number(), commands: z.record(z.string(), z.object({ args: z.string(), seq: z.number() })) }),
    init: (_header, inheritedEventCount) => ({ ownStart: inheritedEventCount, commands: {} }),
    apply: (state, event) => {
      if (event.seq < state.ownStart || event.type !== 'command/run' || event.data.name !== 'control-confirm' || event.data.source?.kind !== 'user') return state;
      return { ...state, commands: { ...state.commands, [event.data.commandId]: { args: event.data.args ?? '', seq: event.seq } } };
    } };
}

export class CasePlanAuthority {
  constructor(ctx, { codeRoot, resolveRequirements, key } = {}) {
    this.ctx = ctx; this.codeRoot = codeRoot; this.resolveRequirements = resolveRequirements;
    this.keyPromise = key ? Promise.resolve(key) : undefined;
    if (ctx.sessionProjections) {
      this.projection = Promise.resolve(ctx.sessionProjections.register(userCommandProjection(codeRoot)));
      this.projection.catch(() => {}); // Handlers surface errors; no unobserved eager rejection.
    }
  }
  key() { return this.keyPromise ??= loadHostKey(this.codeRoot); }

  async confirmationPath(workspace, sessionId, plan, { create = false } = {}) {
    const { directory } = await planDirectory(workspace);
    const confirmations = join(directory, 'confirmations');
    try { if (!inside(directory, await realpath(confirmations))) throw new Error('Case confirmation directory escapes its workspace'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const parent = join(confirmations, sha256(`${resolve(workspace)}\0${sessionId}`), plan.plan_id);
    if (create) await mkdir(parent, { recursive: true });
    try { if (!inside(directory, await realpath(parent))) throw new Error('Case confirmation path escapes its workspace'); }
    catch (error) { if (error.code !== 'ENOENT' || create) throw error; }
    return join(parent, `${plan.version_hash}.json`);
  }

  async confirmPlan({ agent, commandId, rawInput, args }) {
    if (!this.projection || typeof commandId !== 'string') throw new PendingPlanConfirmation('Confirmation requires an actual native user command/run event');
    const projected = (await this.projection).stateOf(agent.session).commands[commandId];
    if (!projected || projected.args.trim() !== rawInput.trim() || !isDeepStrictEqual(JSON.parse(projected.args), args)) {
      throw new PendingPlanConfirmation('Native user command source does not match this plan confirmation');
    }
    if (Object.keys(args).some(key => !['workspace', 'plan_id', 'version_hash'].includes(key))) throw new Error('Confirmation accepts only workspace, plan_id and version_hash');
    const workspace = await authorizeWorkspace(args.workspace, this.ctx.sandboxPolicy.resolve({ session: agent.session }), true);
    const { directory } = await planDirectory(workspace);
    const document = JSON.parse(await readFile(join(directory, 'plans.json'), 'utf8'));
    const scope = document.plans?.[args.plan_id]?.scope;
    if (!planScopes.includes(scope)) throw new PendingPlanConfirmation('Unknown engineering plan');
    const plan = await readCasePlan(workspace, args, scope);
    let confirmationSource = 'native_user';
    try {
      const marker = JSON.parse(await readFile(join(directory, 'test-fixture.json'), 'utf8'));
      if (inside(resolve(this.codeRoot, 'test-output'), workspace) && marker.kind === 'control-case-plan-test-fixture-v1'
        && resolve(marker.workspace) === resolve(workspace)) confirmationSource = 'test_fixture';
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const key = await this.key();
    const path = await this.confirmationPath(workspace, agent.id, plan, { create: true });
    try {
      const existing = verifySigned(JSON.parse(await readFile(path, 'utf8')), key);
      if (existing.session_id === agent.id && existing.workspace === workspace && existing.scope === scope
        && existing.plan_id === plan.plan_id && existing.version_hash === plan.version_hash) return { ...existing, reused: true };
      throw new PendingPlanConfirmation('Stored confirmation belongs to another workspace, plan or scope');
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const body = JSON.stringify({ schema_version: 1, workspace, session_id: agent.id, scope,
      plan_id: plan.plan_id, version_hash: plan.version_hash, confirmation_source: confirmationSource,
      source: { kind: 'native_user_command', command_id: commandId, seq: projected.seq }, confirmed_at: new Date().toISOString() });
    const signed = { body, signature: signature(key, body) };
    try { await writeFile(path, `${JSON.stringify(signed)}\n`, { flag: 'wx', mode: 0o600 }); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    return verifySigned(JSON.parse(await readFile(path, 'utf8')), key);
  }

  async statusPlan({ workspace, reference, agent }) {
    const document = JSON.parse(await readFile(join((await planDirectory(workspace)).directory, 'plans.json'), 'utf8'));
    const scope = document.plans?.[reference?.plan_id]?.scope;
    if (!planScopes.includes(scope)) throw new PendingPlanConfirmation('Unknown engineering plan');
    const plan = await readCasePlan(workspace, reference, scope);
    const path = await this.confirmationPath(workspace, agent.id, plan);
    const result = { scope, plan_id: plan.plan_id, version_hash: plan.version_hash, session_id: agent.id, confirmation_state: 'pending' };
    let confirmed;
    try { confirmed = verifySigned(JSON.parse(await readFile(path, 'utf8')), await this.key()); }
    catch (error) { if (error.code === 'ENOENT') return result; throw error; }
    if (confirmed.workspace === workspace && confirmed.session_id === agent.id && confirmed.scope === scope
      && confirmed.plan_id === plan.plan_id && confirmed.version_hash === plan.version_hash) {
      return { ...result, confirmation_state: 'confirmed', confirmation_source: confirmed.confirmation_source, source: confirmed.source };
    }
    return result;
  }

  async authorizeRequest({ request, agent }) {
    const requirements = this.resolveRequirements ? await this.resolveRequirements(request, agent) : [];
    if (!this.resolveRequirements && request.operation === 'prepare_supervised_dataset') throw new PendingPlanConfirmation('Case confirmation requirement resolver is unavailable');
    if (!requirements.length) return undefined;
    const key = await this.key(), plans = [], confirmationSources = [];
    for (const requirement of requirements) {
      const scope = requirement.scope;
      const plan = await readCasePlan(request.workspace, request.config?.plan_refs?.[scope], scope);
      if (!sameJsonValue(plan.payload, requirement.expectedPayload ?? requirement.expected_payload)
        || !sameJsonValue(plan.bindings, requirement.bindings)) throw new PendingPlanConfirmation(`The ${scope} request or material/data/dependency version differs from the reviewed summary`);
      const path = await this.confirmationPath(request.workspace, agent.id, plan);
      let confirmed;
      try { confirmed = verifySigned(JSON.parse(await readFile(path, 'utf8')), key); }
      catch (error) { if (error.code !== 'ENOENT') throw error; throw new PendingPlanConfirmation(`The ${scope} plan is pending user confirmation in this native Session`); }
      if (confirmed.session_id !== agent.id || confirmed.workspace !== request.workspace || confirmed.plan_id !== plan.plan_id
        || confirmed.version_hash !== plan.version_hash || confirmed.source?.kind !== 'native_user_command') {
        throw new PendingPlanConfirmation('Confirmation belongs to another request version or native Session');
      }
      plans.push({ scope, plan_id: plan.plan_id, version_hash: plan.version_hash });
      confirmationSources.push({ scope, plan_id: plan.plan_id, version_hash: plan.version_hash,
        confirmation_source: confirmed.confirmation_source, source: confirmed.source });
    }
    const rawRequest = JSON.stringify(request);
    const body = JSON.stringify({ schema_version: 1, workspace: request.workspace, session_id: agent.id,
      operation: request.operation, request_sha256: sha256(rawRequest), plans, confirmation_sources: confirmationSources });
    return { rawRequest, env: { CONTROL_CASE_RECEIPT_BODY: body, CONTROL_CASE_RECEIPT_SIGNATURE: signature(key, body),
      CONTROL_CASE_RECEIPT_KEY: key.toString('base64'), CONTROL_CASE_NATIVE_SESSION: agent.id } };
  }
}

export function installCasePlans(ctx, options) {
  const authority = new CasePlanAuthority(ctx, options);
  ctx.commands.register({ definitionId: '@control-engineering/harness-bundle/confirm-plan', name: 'control-confirm',
    description: 'Confirm the exact reviewed engineering plan version in this native Session',
    input: { hint: 'JSON {workspace, plan_id, version_hash}; review the displayed plan before submitting' },
    handler: async invocation => {
      try { return { kind: 'success', text: JSON.stringify(await authority.confirmPlan({ ...invocation, args: JSON.parse(invocation.rawInput) })) }; }
      catch (error) { return { kind: 'error', text: error.message }; }
    } });
  ctx.commands.register({ definitionId: '@control-engineering/harness-bundle/plan-status', name: 'control-plan-status',
    description: 'Read whether this exact engineering plan version is already confirmed in this native Session',
    input: { hint: 'JSON {workspace, plan_id, version_hash}' },
    handler: async ({ rawInput, agent }) => {
      try {
        const args = JSON.parse(rawInput);
        if (Object.keys(args).some(key => !['workspace', 'plan_id', 'version_hash'].includes(key))) throw new Error('Status accepts only workspace, plan_id and version_hash');
        const workspace = await authorizeWorkspace(args.workspace, ctx.sandboxPolicy.resolve({ session: agent.session }));
        return { kind: 'success', text: JSON.stringify(await authority.statusPlan({ workspace, reference: args, agent })) };
      } catch (error) { return { kind: 'error', text: error.message }; }
    } });
  return authority;
}
