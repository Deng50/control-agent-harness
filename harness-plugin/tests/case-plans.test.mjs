/** Native command provenance plus real Python verification; no numerical work. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installCasePlans, PendingPlanConfirmation } from '../src/case-plans.mjs';
import { createHost } from './history-host.mjs';

const project = resolve(import.meta.dirname, '../..');
const python = resolve(project, '.venv/Scripts/python.exe');
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/case-plan-tests'));
await mkdir(output, { recursive: true });
const built = path => import(pathToFileURL(resolve(project, 'vendor/deepseek-harness', path)));
const payload = { inputs: ['温度'], output: '速度', sample_time: .1, scale: 1.0 };
const binding = { material: { id: 'material-1', metadata_hash: 'a'.repeat(64), content_hash: 'b'.repeat(64) } };
const ref = plan => ({ plan_id: plan.plan_id, version_hash: plan.version_hash });
const environment = { ...process.env, PYTHONPATH: resolve(project, 'src'), PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' };

function propose(workspace, scope = 'signals', value = payload, bindings = binding) {
  const run = spawnSync(python, ['-c', 'import json,sys; from control_harness.case_plans import propose_plan; x=json.load(sys.stdin); print(json.dumps(propose_plan(**x)))'], {
    input: JSON.stringify({ workspace, scope, payload: value, bindings }), encoding: 'utf8', windowsHide: true, env: environment,
  });
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout);
}

async function fixture(run) {
  const workspace = await mkdtemp(join(output, 'native-plan-'));
  const [{ Context }, { default: Projection }, { default: Commands }, { Session }] = await Promise.all([
    built('vendor/cordis/lib/index.js'), built('packages/session/session-projection/lib/index.js'),
    built('packages/interaction/commands/lib/index.js'), built('packages/core/session/lib/index.js'),
  ]);
  const ctx = new Context();
  await ctx.plugin(Projection); await ctx.plugin(Commands);
  ctx.provide('sandboxPolicy', { resolve: () => ({ mode: 'danger-full-access', workspaceRoot: workspace }) });
  let expectedPayload = payload, bindings = binding;
  const authority = installCasePlans(ctx, { codeRoot: project, key: Buffer.alloc(32, 7),
    resolveRequirements: async request => request.operation === 'prepare_supervised_dataset'
      ? [{ scope: 'signals', expectedPayload, bindings }] : [] });
  const plan = propose(workspace);
  await writeFile(join(workspace, '.case-plans/test-fixture.json'), JSON.stringify({ kind: 'control-case-plan-test-fixture-v1', workspace }));
  const agent = { id: 'native-plan-session', session: Session.create('native-plan-session') };
  const request = () => ({ operation: 'prepare_supervised_dataset', workspace, inputs: { material: 'material-1' },
    config: { interface: expectedPayload, plan_refs: { signals: ref(plan) } } });
  const confirm = async (owner = agent, version = plan) => {
    const result = await ctx.commands.execute(owner, `/control-confirm ${JSON.stringify({ workspace, ...ref(version) })}`, [], new AbortController().signal);
    assert.equal(result.result.kind, 'success', result.result.text);
    return JSON.parse(result.result.text);
  };
  try { return await run({ workspace, ctx, authority, plan, agent, request, confirm, Session,
    expected: (value, versions = bindings) => { expectedPayload = value; bindings = versions; } }); }
  finally { await ctx.fiber.dispose(); await rm(workspace, { recursive: true, force: true }); }
}

test('pending plans and forged request flags cannot obtain a Host worker receipt', async () => fixture(async ({ authority, request, agent }) => {
  const candidate = request();
  candidate.config.confirmed = true;
  candidate.config.confirmation_source = 'native_user';
  candidate.config.command_id = 'model-invented-command';
  await assert.rejects(authority.authorizeRequest({ request: candidate, agent }), PendingPlanConfirmation);
}));

test('confirmation requires an actual native command and is reused only in the same Session/version', async () => fixture(async ({ authority, agent, plan, workspace, request, confirm, Session, ctx }) => {
  const args = { workspace, ...ref(plan) };
  const before = await ctx.commands.execute(agent, `/control-plan-status ${JSON.stringify(args)}`, [], new AbortController().signal);
  assert.equal(JSON.parse(before.result.text).confirmation_state, 'pending');
  await assert.rejects(stat(join(workspace, '.case-plans/confirmations')), { code: 'ENOENT' });
  await assert.rejects(authority.confirmPlan({ agent, commandId: 'invented', args, rawInput: JSON.stringify(args) }), /Native user command source/);
  const first = await confirm();
  assert.equal(first.confirmation_source, 'test_fixture');
  assert.equal(first.source.kind, 'native_user_command');
  const second = await confirm();
  assert.equal(second.reused, true);
  assert.equal(second.source.command_id, first.source.command_id);
  const status = await authority.statusPlan({ workspace, reference: ref(plan), agent });
  assert.equal(status.confirmation_state, 'confirmed');
  assert.ok((await authority.authorizeRequest({ request: request(), agent })).env.CONTROL_CASE_RECEIPT_SIGNATURE);
  const other = { id: 'different-native-session', session: Session.create('different-native-session') };
  await assert.rejects(authority.authorizeRequest({ request: request(), agent: other }), /pending user confirmation/);
}));

test('copying plan and confirmation files to another workspace does not confer confirmation there', async () => fixture(async ({ authority, workspace, agent, plan, confirm }) => {
  await confirm();
  const copied = await mkdtemp(join(output, 'copied-plan-'));
  try {
    await cp(join(workspace, '.case-plans'), join(copied, '.case-plans'), { recursive: true });
    assert.equal((await authority.statusPlan({ workspace: copied, reference: ref(plan), agent })).confirmation_state, 'pending');
  } finally { await rm(copied, { recursive: true, force: true }); }
}));

test('explicit professional-session migration cannot retarget old plan refs or alter the original native confirmation',async()=>fixture(async({authority,agent,request,confirm,Session,workspace})=>{
  const confirmation=await confirm();const oldReceipt=await authority.authorizeRequest({request:request(),agent});assert.ok(oldReceipt.env.CONTROL_CASE_RECEIPT_SIGNATURE);
  const path=await authority.confirmationPath(workspace,agent.id,request().config.plan_refs.signals);const before=await readFile(path,'utf8');
  const migrated={id:'explicit-new-professional-owner',session:Session.create('explicit-new-professional-owner')};
  await assert.rejects(authority.authorizeRequest({request:request(),agent:migrated}),PendingPlanConfirmation);
  assert.equal(await readFile(path,'utf8'),before);assert.equal((await authority.statusPlan({workspace,reference:request().config.plan_refs.signals,agent})).confirmation_state,'confirmed');assert.equal(confirmation.confirmation_source,'test_fixture');
  await assert.rejects(stat(join(workspace,'.control-jobs')),{code:'ENOENT'});
}));

test('edited plan payload and changed material versions invalidate authorization', async () => fixture(async ({ authority, agent, request, confirm, workspace, expected }) => {
  await confirm();
  expected(payload, { material: { ...binding.material, content_hash: 'c'.repeat(64) } });
  await assert.rejects(authority.authorizeRequest({ request: request(), agent }), /material\/data\/dependency/);
  expected(payload, binding);
  propose(workspace, 'signals', { ...payload, sample_time: .2 });
  await assert.rejects(authority.authorizeRequest({ request: request(), agent }), /version changed/);
}));

test('a real Host receipt is verified in a separate Python worker and rejects request/Session/signature mutation', async () => fixture(async ({ authority, agent, request, confirm }) => {
  await confirm();
  const receipt = await authority.authorizeRequest({ request: request(), agent });
  const source = `import json,sys\nfrom control_harness.runtime import OperationContext\nx=sys.stdin.read(); context=OperationContext(); context.bind_raw_request(x); request=json.loads(x)\ntry:\n context.require_case_confirmation(request,[{'scope':'signals','expectedPayload':request['config']['interface'],'bindings':${JSON.stringify(binding)}}]); print(json.dumps({'ok':True}))\nexcept Exception as error:\n print(json.dumps({'ok':False,'error_type':type(error).__name__}))`;
  function worker(raw = receipt.rawRequest, overrides = {}) {
    const result = spawnSync(python, ['-c', source], { input: raw, env: { ...environment, ...receipt.env, ...overrides }, encoding: 'utf8', windowsHide: true });
    assert.equal(result.status, 0, result.stderr);
    return JSON.parse(result.stdout);
  }
  assert.equal(worker().ok, true);
  const tampered = JSON.parse(receipt.rawRequest); tampered.config.confirmed = true;
  assert.equal(worker(JSON.stringify(tampered)).ok, false);
  assert.equal(worker(receipt.rawRequest, { CONTROL_CASE_NATIVE_SESSION: 'other-session' }).ok, false);
  assert.equal(worker(receipt.rawRequest, { CONTROL_CASE_RECEIPT_SIGNATURE: '0'.repeat(64) }).ok, false);
  assert.equal(worker(receipt.rawRequest, { CONTROL_CASE_RECEIPT_KEY: '', CONTROL_CASE_RECEIPT_BODY: '', CONTROL_CASE_RECEIPT_SIGNATURE: '' }).ok, false);
}));

test('native confirmation normalizes negative zero JSON semantics and keeps bool distinct from number', async () => fixture(async ({ authority, agent, workspace, request, confirm, expected }) => {
  const negativeZeroPayload = { ...payload, lower: -0, enabled: false };
  const negativeZeroBindings = { ...binding, offset: -0 };
  const plan = propose(workspace, 'signals', negativeZeroPayload, negativeZeroBindings);
  expected(negativeZeroPayload, negativeZeroBindings);
  const candidate = request();
  candidate.config.plan_refs.signals = ref(plan);
  await confirm(agent, plan);
  const receipt = await authority.authorizeRequest({ request: candidate, agent });
  assert.ok(receipt.env.CONTROL_CASE_RECEIPT_SIGNATURE);
  const source = `import json,sys\nfrom control_harness.runtime import OperationContext\nx=sys.stdin.read(); c=OperationContext(); c.bind_raw_request(x); r=json.loads(x); p={**r['config']['interface'],'lower':-0.0}; b=${JSON.stringify(negativeZeroBindings)}; b['offset']=-0.0\nc.require_case_confirmation(r,[{'scope':'signals','expectedPayload':p,'bindings':b}]); print('verified-negative-zero')`;
  const worker = spawnSync(python, ['-c', source], { input: receipt.rawRequest,
    env: { ...environment, ...receipt.env }, encoding: 'utf8', windowsHide: true });
  assert.equal(worker.status, 0, worker.stderr);
  assert.match(worker.stdout, /verified-negative-zero/);
  expected({ ...negativeZeroPayload, enabled: 0 }, negativeZeroBindings);
  await assert.rejects(authority.authorizeRequest({ request: candidate, agent }), /differs from the reviewed/);
}));

test('actual Host submit rejects pending and forged confirmations before any history, job or worker admission', async () => {
  const workspace = await mkdtemp(join(output, 'host-pending-'));
  const source = 'import json,sys; from pathlib import Path; from control_harness.artifacts import Workspace; from control_harness.modeling import register_model; root=Path(sys.argv[1]); ws=Workspace(sys.argv[2]); ids=[register_model(ws,json.loads((root/"examples/linear"/name).read_text()),{"kind":kind,"description":"explicit test fixture with existing equations; no sampling or training"}) for name,kind in (("nominal.json","user_nominal"),("reference.json","reference_simulator"))]; print(json.dumps(ids))';
  const initialized = spawnSync(python, ['-c', source, project, workspace], { env: environment, encoding: 'utf8', windowsHide: true });
  assert.equal(initialized.status, 0, initialized.stderr);
  const [nominal, reference] = JSON.parse(initialized.stdout);
  const host = await createHost(workspace, 'pending-admission-native-session', { confirmFixturePlans: false });
  const originalStart = host.ctx.jobs.start, originalSpawn = host.ctx.subprocess.spawn;
  let jobs = 0, workers = 0;
  host.ctx.jobs.start = () => { jobs += 1; throw new Error('Pending request reached native job admission'); };
  host.ctx.subprocess.spawn = () => { workers += 1; throw new Error('Pending request reached a numerical worker'); };
  try {
    for (const fake of [{}, { confirmed: true }, { confirmed: true, confirmation_source: 'test_fixture', commandId: 'invented' }]) {
      const result = await host.command('control', { agent: 'model', operation: 'simulate_transitions', workspace,
        inputs: { nominal_model: nominal, reference_model: reference }, config: { trajectories: 2, steps: 3, seed: 42, ...fake } });
      assert.equal(result.kind, 'error');
      assert.match(result.text, /plan is pending|Missing current config\.plan_refs/);
    }
    assert.equal(jobs, 0); assert.equal(workers, 0);
    assert.deepEqual(host.ctx.jobs.list('pending-admission-native-session'), []);
    await assert.rejects(stat(join(workspace, '.control-jobs')), { code: 'ENOENT' });
  } finally {
    host.ctx.jobs.start = originalStart; host.ctx.subprocess.spawn = originalSpawn;
    await host.close(); await rm(workspace, { recursive: true, force: true });
  }
});
