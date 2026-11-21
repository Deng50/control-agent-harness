/** Engineering confirmation authority: native user commands, versioned plans and worker receipts. */
import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto';
import { mkdir, readFile, readdir, realpath, rename, rmdir, stat, unlink, writeFile } from 'node:fs/promises';
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
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const referenceOf = plan => ({ plan_id: plan.plan_id, version_hash: plan.version_hash });
const cardPattern = /^card-[a-f0-9]{16}-[a-f0-9]{48}$/;
export class PendingPlanConfirmation extends Error { constructor(message) { super(message); this.name = 'PendingPlanConfirmation'; } }

// Recognize a second pasted command only after the JSON object has closed.
// Slashes, escapes and physical newlines inside legitimate input stay intact.
function parseArguments(raw) {
  const input = raw.trim();
  try {
    const value = JSON.parse(input);
    if (object(value)) return value;
  } catch {
    let depth = 0, quoted = false, escaped = false;
    if (input.startsWith('{')) for (let i = 0; i < input.length; i++) {
      const char = input[i];
      if (quoted) {
        if (escaped) escaped = false;
        else if (char === '\\') escaped = true;
        else if (char === '"') quoted = false;
      } else if (char === '"') quoted = true;
      else if (char === '{' || char === '[') depth++;
      else if ((char === '}' || char === ']') && --depth === 0) {
        if (/^(?:\/control-[\w-]+\b|\{)/.test(input.slice(i + 1).trimStart())) {
          throw new Error('ONE_COMMAND_PER_SUBMISSION：每次提交一个命令。使用工程确认卡可一次确认本次显示的全部内容。');
        }
        break;
      }
    }
  }
  throw new Error(`INVALID_CONFIRM_JSON：请点击工程确认卡。手工命令需要标准 JSON，键名和字符串须使用双引号。示例 /control-confirm ${JSON.stringify({workspace:'D:\\control workspace',plan_id:'plan-example',version_hash:'0'.repeat(64)})}`);
}

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

async function planDocument(directory) {
  const path = join(directory,'plans.json');
  if (relative(path,await realpath(path)) !== '') throw new Error('Case plan file redirects outside its identity');
  return JSON.parse(await readFile(path,'utf8'));
}

function checkedPlan(document, reference, scope, visiting = new Set()) {
  const record = document.plans?.[reference.plan_id];
  if (![1, 2].includes(document.schema_version) || !record || record.scope !== scope || record.status === 'stale'
    || record.version_hash !== reference.version_hash || typeof record.canonical_payload_json !== 'string'
    || sha256(record.canonical_payload_json) !== record.version_hash) {
    throw new PendingPlanConfirmation(`The ${scope} plan version changed; review and confirm the current summary`);
  }
  const v2 = record.schema_version === 2;
  const active = v2 ? document.active_by_intent?.[record.intent_id] : document.active_by_scope;
  const keys = ['scope', 'payload', 'bindings', 'dependencies'];
  if (v2) keys.push('intent_id','revision');
  if (v2 && Object.hasOwn(record,'effective_settings')) keys.push('effective_settings');
  if ((v2 && (active?.[scope] !== record.plan_id || record.review_hash !== record.version_hash
    || !sameJsonValue(record.review_payload, record.payload)))
    || !isDeepStrictEqual(JSON.parse(record.canonical_payload_json), Object.fromEntries(keys.map(key => [key, record[key]])))) {
    throw new PendingPlanConfirmation('Plan version changed or payload does not match its Python-owned canonical version');
  }
  if (visiting.has(record.plan_id)) throw new PendingPlanConfirmation('Cyclic engineering plan dependencies');
  visiting.add(record.plan_id);
  for (const [dependency, ref] of Object.entries(record.dependencies)) {
    const upstream = document.plans?.[active?.[dependency]];
    if (!planScopes.slice(0,planScopes.indexOf(scope)).includes(dependency) || !upstream
      || upstream.plan_id !== ref.plan_id || upstream.version_hash !== ref.version_hash || upstream.status === 'stale'
      || (v2 && (upstream.intent_id !== record.intent_id || upstream.schema_version !== 2))) {
      throw new PendingPlanConfirmation(`The upstream ${dependency} plan changed`);
    }
    if (v2) checkedPlan(document, ref, dependency, visiting);
  }
  visiting.delete(record.plan_id);
  return record;
}

export async function readCasePlan(workspace, reference, scope) {
  if (!reference || !/^[A-Za-z0-9_-]{1,100}$/.test(reference.plan_id ?? '') || !/^[a-f0-9]{64}$/.test(reference.version_hash ?? '')) {
    throw new PendingPlanConfirmation(`请通过 control_prepare_review 生成工程确认卡，用户点击后提交 card_id。Missing current config.plan_refs.${scope}`);
  }
  let directory;
  try { ({ directory } = await planDirectory(workspace)); }
  catch (error) {
    if (error.code !== 'ENOENT') throw error;
    throw new PendingPlanConfirmation(`The ${scope} plan is pending; propose and review it before creating a numerical job`);
  }
  const document = await planDocument(directory);
  return checkedPlan(document, reference, scope);
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

function differences(previous, current, path = '') {
  if (sameJsonValue(previous, current)) return [];
  if (object(previous) && object(current)) {
    return [...new Set([...Object.keys(previous), ...Object.keys(current)])].sort()
      .flatMap(key => differences(previous[key], current[key], path ? `${path}.${key}` : key));
  }
  return [{ path, before: previous === undefined ? '未设置' : previous, after: current === undefined ? '未设置' : current }];
}

// Publish a nonempty directory. Rename cannot replace another nonempty
// directory, even on POSIX. Concurrent clicks therefore have exactly one
// winner, without requiring Windows hard-link privileges or a persistent lock.
async function publishCard(path, value, key) {
  const body = JSON.stringify(value), temp = `${path}.${randomBytes(12).toString('hex')}.tmp`;
  await mkdir(temp);
  const file = join(temp,'record.json');
  try {
    await writeFile(file, JSON.stringify({ body, signature: signature(key, body) }), { flag:'wx',mode:0o600 });
    try { await rename(temp,path); return true; }
    catch (error) {
      if (!['EEXIST','ENOTEMPTY','EACCES','EPERM'].includes(error.code) || !(await stat(path).catch(() => null))?.isDirectory()) throw error;
      return false;
    }
  } finally {
    await unlink(file).catch(error => { if (error.code !== 'ENOENT') throw error; });
    await rmdir(temp).catch(error => { if (error.code !== 'ENOENT') throw error; });
  }
}

const cardPrefix = intentId => `card-${sha256(intentId).slice(0,16)}-`;

export class CasePlanAuthority {
  constructor(ctx, { codeRoot, resolveRequirements, proposeBatch, key, matlabGrants } = {}) {
    this.ctx = ctx; this.codeRoot = codeRoot; this.resolveRequirements = resolveRequirements; this.proposeBatch = proposeBatch;
    this.keyPromise = key ? Promise.resolve(key) : undefined;
    this.matlabGrants = matlabGrants;
    if (ctx.sessionProjections) {
      this.projection = Promise.resolve(ctx.sessionProjections.register(userCommandProjection(codeRoot)));
      this.projection.catch(() => {}); // Handlers surface errors; no unobserved eager rejection.
    }
  }
  key() { return this.keyPromise ??= loadHostKey(this.codeRoot); }

  intentFor({ agent, role, intent_id }) {
    if (!agent?.id || typeof role !== 'string' || !role.length) throw new Error('An engineering task requires a Session and role');
    const prefix = `intent-${sha256(JSON.stringify([agent.id,role])).slice(0,20)}-`;
    if (intent_id == null) return `${prefix}${randomBytes(12).toString('hex')}`;
    if (typeof intent_id !== 'string' || !intent_id.trim() || intent_id.length > 128) throw new Error('Invalid engineering intent');
    if (intent_id.startsWith('intent-')) {
      if (!intent_id.startsWith(prefix) || !/^[a-f0-9]{24}$/.test(intent_id.slice(prefix.length))) {
        throw new PendingPlanConfirmation('Engineering intent belongs to another role or Session');
      }
      return intent_id;
    }
    return `${prefix}${sha256(intent_id.trim()).slice(0,24)}`;
  }

  async cardDirectory(workspace, agent, create = false) {
    let {directory} = await planDirectory(workspace);
    for (const name of ['review-cards', sha256(JSON.stringify([workspace, agent.id]))]) {
      const path = join(directory, name);
      if (create) await mkdir(path).catch(error => { if (error.code !== 'EEXIST') throw error; });
      const actual = await realpath(path);
      if (relative(path, actual) !== '') throw new Error('Review card directory redirects outside its identity');
      directory = actual;
    }
    return directory;
  }

  async cardFile(directory, id, workspace, agent, accepted = false) {
    if (!cardPattern.test(id)) throw new PendingPlanConfirmation('Invalid engineering card handle');
    const path = join(directory, `${id}${accepted ? '.accepted' : '.review'}`,'record.json');
    if (relative(path, await realpath(path)) !== '') throw new Error('Review card file redirects outside its identity');
    const card = verifySigned(JSON.parse(await readFile(path, 'utf8')), await this.key());
    if (card.kind !== (accepted ? 'case_card_acceptance_v2' : 'case_card_review_v2')
      || card.card_id !== id || card.workspace !== workspace || card.session_id !== agent.id) {
      throw new PendingPlanConfirmation('Engineering card belongs to another workspace, version or Session');
    }
    return card;
  }

  async cardsForIntent({workspace, agent, intent_id, accepted = false}) {
    let directory;
    try { directory = await this.cardDirectory(workspace, agent); }
    catch (error) { if (error.code === 'ENOENT') return []; throw error; }
    const suffix = accepted ? '.accepted' : '.review', result = [];
    for (const name of await readdir(directory)) {
      if (!name.startsWith(cardPrefix(intent_id)) || !name.endsWith(suffix)) continue;
      const id = name.slice(0, -suffix.length);
      if (cardPattern.test(id)) result.push(await this.cardFile(directory, id, workspace, agent, accepted));
    }
    return result.filter(card => card.intent_id === intent_id).sort((a,b) => b.created_at.localeCompare(a.created_at));
  }

  async prepareCard({ request, agent, role, intent_id, requirements, retire_scopes, signal }) {
    if (!Array.isArray(requirements) || !requirements.length) throw new Error('This operation has no engineering scopes to confirm');
    intent_id = this.intentFor({agent,role,intent_id});
    const workspace = await authorizeWorkspace(request.workspace, this.ctx.sandboxPolicy.resolve({session:agent.session}), true);
    const scopes = requirements.map(row => row.scope);
    if (scopes.some(scope => !planScopes.includes(scope)) || new Set(scopes).size !== scopes.length) throw new Error('Invalid engineering review scope set');
    const previous = (await this.cardsForIntent({workspace,agent,intent_id}))[0];
    const matlabScope = await this.matlabGrants?.review({...request,workspace},agent,intent_id);
    const proposed = await this.proposeBatch({request:{...request,workspace},intent_id,requirements,retire_scopes,agent,signal});
    const plan_refs = {}, changes = {}, reviewed = [];
    for (const row of requirements) {
      const record = proposed.find(plan => plan.scope === row.scope);
      if (!record || record.intent_id !== intent_id || record.schema_version !== 2) throw new Error('Plan writer returned a different engineering task');
      const reference = referenceOf(record);
      const plan = await readCasePlan(workspace, reference, row.scope);
      if (!sameJsonValue(plan.payload,row.reviewPayload) || !sameJsonValue(plan.bindings,row.reviewBindings)) throw new Error('Plan proposal differs from the current review');
      plan_refs[row.scope] = reference;
      const displayed = {...row};
      if (Object.hasOwn(plan,'effective_settings')) displayed.effective_settings = plan.effective_settings;
      reviewed.push(displayed);
      const old = previous?.requirements.find(item => item.scope === row.scope);
      const businessSettings = item => ({parameters:item.reviewPayload, sources:item.reviewBindings, effective_settings:item.effective_settings});
      changes[row.scope] = old ? differences(businessSettings(old),businessSettings(displayed)) : [];
    }
    if (previous?.operation === 'prepare_supervised_dataset' && request.operation === previous.operation
      && Boolean(previous.plan_refs.training) !== Boolean(plan_refs.training)) {
      const stopAfter = refs => refs.training ? '训练并导出网络' : '准备数据集';
      changes.signals.push({path:'交付停止点',before:stopAfter(previous.plan_refs),after:stopAfter(plan_refs)});
    }
    const snapshot = {workspace,session_id:agent.id,role,intent_id,operation:request.operation,plan_refs,requirements:reviewed,
      ...(matlabScope ? {matlab_scope:matlabScope} : {})};
    const card_id = cardPrefix(intent_id) + sha256(JSON.stringify(snapshot)).slice(0,48);
    const directory = await this.cardDirectory(workspace,agent,true);
    await publishCard(join(directory,`${card_id}.review`), {kind:'case_card_review_v2', ...snapshot,card_id,changes,created_at:new Date().toISOString()}, await this.key());
    return this.cardStatus({workspace,card_id,agent,role});
  }

  async cardStatus({workspace, card_id, agent, role}) {
    workspace = await authorizeWorkspace(workspace, this.ctx.sandboxPolicy.resolve({session:agent.session}));
    let card;
    try { card = await this.cardFile(await this.cardDirectory(workspace,agent),card_id,workspace,agent); }
    catch (error) { if (error.code !== 'ENOENT') throw error; throw new PendingPlanConfirmation('No such review card in the current workspace and Session'); }
    if (role && card.role !== role) throw new PendingPlanConfirmation('Engineering card belongs to another role');
    const statuses = {};
    const document = await planDocument((await planDirectory(workspace)).directory);
    const receipts = await this.cardsForIntent({workspace,agent,intent_id:card.intent_id,accepted:true});
    for (const [scope,reference] of Object.entries(card.plan_refs)) {
      const plan = checkedPlan(document,reference,scope);
      statuses[scope] = await this.statusForPlan({workspace,plan,agent,receipts});
    }
    return {...card,statuses,...(card.matlab_scope ? {matlab_grant:this.matlabGrants.status(card)} : {})};
  }

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
    if (!projected || projected.args.trim() !== rawInput.trim() || !isDeepStrictEqual(parseArguments(projected.args), args)) {
      throw new PendingPlanConfirmation('Native user command source does not match this plan confirmation');
    }
    const grouped = Object.hasOwn(args,'card_id');
    const allowed = grouped ? ['workspace','card_id'] : ['workspace','plan_id','version_hash'];
    if (Object.keys(args).some(key => !allowed.includes(key))) throw new Error('Select a review card or an exact individual plan version');
    const workspace = await authorizeWorkspace(args.workspace, this.ctx.sandboxPolicy.resolve({ session: agent.session }), true);
    const { directory } = await planDirectory(workspace);
    let confirmationSource = 'native_user';
    try {
      const marker = JSON.parse(await readFile(join(directory, 'test-fixture.json'), 'utf8'));
      if (inside(resolve(this.codeRoot, 'test-output'), workspace) && marker.kind === 'control-case-plan-test-fixture-v1'
        && resolve(marker.workspace) === resolve(workspace)) confirmationSource = 'test_fixture';
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const key = await this.key();
    const source = {kind:'native_user_command',command_id:commandId,seq:projected.seq};
    if (grouped) {
      const card = await this.cardStatus({workspace,card_id:args.card_id,agent});
      const cardDirectory = await this.cardDirectory(workspace,agent);
      // Inspect every head in one atomic plans.json snapshot before publication.
      const document = await planDocument(directory);
      for (const [scope, ref] of Object.entries(card.plan_refs)) checkedPlan(document,ref,scope);
      const acceptance = {kind:'case_card_acceptance_v2',workspace,session_id:agent.id,intent_id:card.intent_id,
        card_id:card.card_id,plan_refs:card.plan_refs,confirmation_source:confirmationSource,source,created_at:new Date().toISOString()};
      const created = await publishCard(join(cardDirectory,`${card.card_id}.accepted`),acceptance,key);
      const committed = await this.cardFile(cardDirectory,card.card_id,workspace,agent,true);
      // Only this invocation has a verified native user event. Reading an old
      // receipt, a projected plan status or a tool result cannot issue a grant.
      if (card.matlab_scope) this.matlabGrants.issue(card,source);
      return {...await this.cardStatus({workspace,card_id:card.card_id,agent}), source:committed.source,
        confirmation_source:committed.confirmation_source,reused:!created};
    }
    const document = await planDocument(directory);
    const scope = document.plans?.[args.plan_id]?.scope;
    if (!planScopes.includes(scope)) throw new PendingPlanConfirmation('Unknown engineering plan');
    const plan = checkedPlan(document,args,scope);
    if (plan.schema_version === 2) throw new PendingPlanConfirmation('此版本属于结构化确认卡，请打开完整卡片核对后一次确认。');
    const path = await this.confirmationPath(workspace, agent.id, plan, { create: true });
    try {
      const existing = verifySigned(JSON.parse(await readFile(path, 'utf8')), key);
      if (existing.session_id === agent.id && existing.workspace === workspace && existing.scope === scope
        && existing.plan_id === plan.plan_id && existing.version_hash === plan.version_hash) return { ...existing, reused: true };
      throw new PendingPlanConfirmation('Stored confirmation belongs to another workspace, plan or scope');
    } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const confirmation = { schema_version: 1, workspace, session_id: agent.id, scope,
      plan_id: plan.plan_id, version_hash: plan.version_hash, confirmation_source: confirmationSource,
      source, confirmed_at: new Date().toISOString() };
    const body = JSON.stringify(confirmation);
    try { await writeFile(path,JSON.stringify({body,signature:signature(key,body)}),{flag:'wx',mode:0o600}); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    return verifySigned(JSON.parse(await readFile(path, 'utf8')), key);
  }

  async statusPlan({ workspace, reference, agent }) {
    const document = await planDocument((await planDirectory(workspace)).directory);
    const scope = document.plans?.[reference?.plan_id]?.scope;
    if (!planScopes.includes(scope)) throw new PendingPlanConfirmation('Unknown engineering plan');
    return this.statusForPlan({workspace,plan:checkedPlan(document,reference,scope),agent});
  }

  async statusForPlan({workspace,plan,agent,receipts}) {
    const {scope} = plan;
    const result = { scope, plan_id: plan.plan_id, version_hash: plan.version_hash, session_id: agent.id, confirmation_state: 'pending' };
    if (plan.schema_version === 2) {
      receipts ??= await this.cardsForIntent({workspace,agent,intent_id:plan.intent_id,accepted:true});
      const accepted = receipts.find(receipt => sameJsonValue(receipt.plan_refs?.[scope], referenceOf(plan)));
      if (!accepted) return result;
      return {...result,confirmation_state:'confirmed',confirmation_source:accepted.confirmation_source,source:accepted.source,card_id:accepted.card_id};
    }
    const path = await this.confirmationPath(workspace, agent.id, plan);
    let confirmed;
    try { confirmed = verifySigned(JSON.parse(await readFile(path, 'utf8')), await this.key()); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (confirmed && confirmed.workspace === workspace && confirmed.session_id === agent.id && confirmed.scope === scope
      && confirmed.plan_id === plan.plan_id && confirmed.version_hash === plan.version_hash) {
      return { ...result, confirmation_state: 'confirmed', confirmation_source: confirmed.confirmation_source, source: confirmed.source };
    }
    return result;
  }

  async authorizeRequest({ request, agent, role }) {
    const signalsRef = request.config?.plan_refs?.signals;
    const signalsPlan = signalsRef ? await readCasePlan(request.workspace,signalsRef,'signals') : null;
    const options = signalsPlan?.schema_version === 2 ? {intent_id:signalsPlan.intent_id} : {};
    if (options.intent_id && request.operation === 'train_supervised') options.derivationKey = await this.key();
    if (!this.resolveRequirements) throw new PendingPlanConfirmation('Case confirmation requirement resolver is unavailable');
    const requirements = await this.resolveRequirements(request, agent, options);
    if (!requirements.length) return undefined;
    const key = await this.key(), plans = [], confirmationSources = [], executionBindings = [];
    let hasV2 = false, intentId, receipts;
    for (const requirement of requirements) {
      const scope = requirement.scope;
      const plan = await readCasePlan(request.workspace, request.config?.plan_refs?.[scope], scope);
      const v2 = plan.schema_version === 2;
      if (plans.length && (hasV2 !== v2 || (v2 && intentId !== plan.intent_id))) {
        throw new PendingPlanConfirmation('Selected plans belong to different engineering tasks or protocols');
      }
      if (v2) {
        hasV2 = true; intentId = plan.intent_id;
        if (role) this.intentFor({agent,role,intent_id:intentId});
      }
      if (!sameJsonValue(plan.payload, v2 ? requirement.reviewPayload : requirement.expectedPayload ?? requirement.expected_payload)
        || !sameJsonValue(plan.bindings, v2 ? requirement.reviewBindings : requirement.bindings)) throw new PendingPlanConfirmation(`The ${scope} request or material/data/dependency version differs from the reviewed summary`);
      if (v2) receipts ??= await this.cardsForIntent({workspace:request.workspace,agent,intent_id:intentId,accepted:true});
      const confirmed = await this.statusForPlan({workspace:request.workspace,plan,agent,receipts});
      if (confirmed.confirmation_state !== 'confirmed') throw new PendingPlanConfirmation(`The ${scope} plan is pending user confirmation in this native Session`);
      if (confirmed.source?.kind !== 'native_user_command' || !['test_fixture','native_user'].includes(confirmed.confirmation_source)) {
        throw new PendingPlanConfirmation('Confirmation belongs to another request version or native Session');
      }
      plans.push({ scope, plan_id: plan.plan_id, version_hash: plan.version_hash });
      confirmationSources.push({ scope, plan_id: plan.plan_id, version_hash: plan.version_hash,
        confirmation_source: confirmed.confirmation_source, source: confirmed.source });
      executionBindings.push({scope,bindings:requirement.bindings});
    }
    const rawRequest = JSON.stringify(request);
    const body = JSON.stringify({ schema_version: 1, workspace: request.workspace, session_id: agent.id,
      operation: request.operation, request_sha256: sha256(rawRequest), plans, confirmation_sources: confirmationSources,
      ...(hasV2 ? {execution_bindings:executionBindings} : {}) });
    return { rawRequest, env: { CONTROL_CASE_RECEIPT_BODY: body, CONTROL_CASE_RECEIPT_SIGNATURE: signature(key, body),
      CONTROL_CASE_RECEIPT_KEY: key.toString('base64'), CONTROL_CASE_NATIVE_SESSION: agent.id } };
  }
}

export function installCasePlans(ctx, options) {
  const authority = new CasePlanAuthority(ctx, options);
  ctx.commands.register({ definitionId: '@control-engineering/harness-bundle/confirm-plan', name: 'control-confirm',
    description: 'Confirm the exact reviewed engineering plan version in this native Session',
    input: { hint: '通过工程确认卡提交本次审阅。高级入口接受一条标准 JSON 命令。' },
    handler: async invocation => {
      try { return { kind: 'success', text: JSON.stringify(await authority.confirmPlan({ ...invocation, args: parseArguments(invocation.rawInput) })) }; }
      catch (error) { return { kind: 'error', text: error.message }; }
    } });
  ctx.commands.register({ definitionId: '@control-engineering/harness-bundle/plan-status', name: 'control-plan-status',
    description: 'Read whether this exact engineering plan version is already confirmed in this native Session',
    input: { hint: '读取指定版本的确认状态' },
    handler: async ({ rawInput, agent }) => {
      try {
        const args = parseArguments(rawInput);
        if (Object.keys(args).some(key => !['workspace', 'plan_id', 'version_hash'].includes(key))) throw new Error('Status accepts only workspace, plan_id and version_hash');
        const workspace = await authorizeWorkspace(args.workspace, ctx.sandboxPolicy.resolve({ session: agent.session }));
        return { kind: 'success', text: JSON.stringify(await authority.statusPlan({ workspace, reference: args, agent })) };
      } catch (error) { return { kind: 'error', text: error.message }; }
    } });
  ctx.commands.register({definitionId:'@control-engineering/harness-bundle/card-status',name:'control-card-status',
    description:'读取当前原生会话中此工程确认卡的内容、变更和签署状态', input:{hint:'由网页确认卡提供工作区和卡片句柄'},
    handler:async ({rawInput,agent}) => {
      try {
        const args = parseArguments(rawInput);
        if (Object.keys(args).some(key => !['workspace','card_id'].includes(key))) throw new Error('Card status accepts only workspace and card_id');
        return {kind:'success',text:JSON.stringify(await authority.cardStatus({...args,agent}))};
      } catch (error) { return {kind:'error',text:error.message}; }
    }});
  return authority;
}
