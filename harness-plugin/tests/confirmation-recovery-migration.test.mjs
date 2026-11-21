/** Real native confirmation and numerical retry of a pre-S03 interrupted ledger. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createHost, project } from './history-host.mjs';
import { JobHistory } from '../src/job-history.mjs';
import { confirmationRequirements } from '../src/case-requirements.mjs';

const python = resolve(project, '.venv/Scripts/python.exe');
const environment = { ...process.env, PYTHONPATH: resolve(project, 'src'), PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' };
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/confirmation-migration-tests'));
await mkdir(output, { recursive: true });
const unpack = value => { assert.equal(value.kind, 'success', value.text); return JSON.parse(value.text); };

for (const selector of ['legacy refs','structured card']) test(`legacy interrupted no-refs recovery via ${selector}; pending recovery creates no attempt or job`, { timeout: 60000 }, async () => {
  const workspace = await mkdtemp(join(output, 'legacy-recovery-'));
  let host;
  try {
    const source = 'import json,sys; from pathlib import Path; from control_harness.artifacts import Workspace; from control_harness.modeling import register_model; root=Path(sys.argv[1]); ws=Workspace(sys.argv[2]); ids=[register_model(ws,json.loads((root/"examples/linear"/name).read_text()),{"kind":kind,"description":"explicit pre-upgrade migration fixture"}) for name,kind in (("nominal.json","user_nominal"),("reference.json","reference_simulator"))]; print(json.dumps(ids))';
    const initialized = spawnSync(python, ['-c', source, project, workspace], { env: environment, encoding: 'utf8', windowsHide: true });
    assert.equal(initialized.status, 0, initialized.stderr);
    const [nominal, reference] = JSON.parse(initialized.stdout);
    host = await createHost(workspace, 'legacy-migration-native-session', { confirmFixturePlans: false });
    const request = { operation: 'simulate_transitions', workspace, inputs: { nominal_model: nominal, reference_model: reference },
      config: { trajectories: 6, steps: 6, seed: 59 } };
    const history = new JobHistory();
    const writer = await history.create({ workspace, owner: host.agent.id, agentName: 'model', request });
    await writer.append({ state: 'running', nativeJobId: 'pre-upgrade-native-job-no-longer-present' });
    await writer.flush();
    // Construct a real migration fixture, not a fake human confirmation. Its
    // previous producer is an actual short-lived process whose exit we joined.
    const exited = spawnSync(process.execPath, ['-e', ''], { windowsHide: true });
    assert.equal(exited.status, 0);
    const recordPath = join(workspace, '.control-jobs', writer.record.attemptId, 'request.json');
    const legacyRecord = JSON.parse(await readFile(recordPath, 'utf8'));
    legacyRecord.hostPid = exited.pid;
    await writeFile(recordPath, `${JSON.stringify(legacyRecord)}\n`);
    const record = unpack(await host.command('control-status', { workspace, attemptId: writer.record.attemptId })).record;
    assert.equal(record.state, 'interrupted');
    assert.equal(record.recoveryBlocked, undefined);
    assert.equal(record.request.config.plan_refs, undefined);
    const attempts = async () => (await readdir(join(workspace, '.control-jobs'))).sort();
    const baseline = await attempts();
    const registrations = () => host.agent.session.snapshotEvents().filter(event => event.type === 'job/registered').length;
    const assertPending = async args => {
      const beforeJobs = registrations();
      const result = await host.command('control-recover', { workspace, attemptId: writer.record.attemptId, mode: 'retry', ...args });
      assert.equal(result.kind, 'error');
      assert.match(result.text, /plan_refs|pending user confirmation|待用户确认/);
      assert.deepEqual(await attempts(), baseline, 'Pending recovery must not create dangling attempt history');
      assert.equal(registrations(), beforeJobs, 'Pending recovery must not register a native job');
      await assert.rejects(stat(join(workspace, '.control-jobs', writer.record.attemptId, 'recovery.json')), { code: 'ENOENT' });
    };
    await assertPending({ confirmed: true, confirmation_source: 'test_fixture' });
    const requirements = await confirmationRequirements(request);
    const plansDirectory = join(workspace, '.case-plans');
    await mkdir(plansDirectory, { recursive: true });
    await writeFile(join(plansDirectory, 'test-fixture.json'), JSON.stringify({ kind: 'control-case-plan-test-fixture-v1', workspace }));
    const proposalSource = `import json,sys\nfrom control_harness.case_plans import propose_plan\nx=json.load(sys.stdin); plans=[]; deps={}\nfor r in x['requirements']:\n p=propose_plan(x['workspace'],r['scope'],r['expectedPayload'],r['bindings'],dict(deps)); plans.append(p); deps[r['scope']]={k:p[k] for k in ('plan_id','version_hash')}\nprint(json.dumps(plans))`;
    const proposed = spawnSync(python, ['-c', proposalSource], { input: JSON.stringify({ workspace, requirements }), env: environment, encoding: 'utf8', windowsHide: true });
    assert.equal(proposed.status, 0, proposed.stderr);
    const plans = JSON.parse(proposed.stdout);
    const refs = Object.fromEntries(plans.map(plan => [plan.scope, { plan_id: plan.plan_id, version_hash: plan.version_hash }]));
    await assertPending({ plan_refs: refs });
    for (const reference of Object.values(refs)) {
      const confirmation = unpack(await host.command('control-confirm', { workspace, ...reference }));
      assert.equal(confirmation.confirmation_source, 'test_fixture');
      assert.equal(confirmation.source.kind, 'native_user_command');
    }
    let selection = {plan_refs:refs}, expectedRefs = refs;
    if (selector === 'structured card') {
      const card = unpack(await host.command('control-prepare-review',{...request,agent:'model'}));
      selection = {card_id:card.card_id};
      expectedRefs = card.plan_refs;
      await assertPending(selection);
      const signed = unpack(await host.command('control-confirm',{workspace,...selection}));
      assert.equal(signed.confirmation_source,'test_fixture');
    }
    const started = unpack(await host.command('control-recover', { workspace, attemptId: writer.record.attemptId, mode: 'retry', ...selection }));
    const ended = await host.ctx.jobs.wait(started.jobId, 30000, host.agent.id);
    assert.equal(ended.status, 'completed', ended.detail);
    const replacement = unpack(await host.command('control-status', { workspace, attemptId: started.attemptId })).record;
    assert.equal(replacement.result.kind, 'transition_dataset');
    assert.equal(replacement.result.status, 'verified');
    assert.equal(replacement.recoveredFrom, writer.record.attemptId);
    assert.deepEqual(replacement.request.config.plan_refs, expectedRefs);
    assert.equal(JSON.parse(await readFile(recordPath, 'utf8')).request.config.plan_refs, undefined, 'Migration must preserve the old immutable request');
    const after = await attempts();
    assert.equal(after.length, baseline.length + 1);
    const duplicate = unpack(await host.command('control-recover', { workspace, attemptId: writer.record.attemptId, mode: 'retry', ...selection }));
    assert.equal(duplicate.attemptId, started.attemptId);
    assert.deepEqual(await attempts(), after);
    assert.equal(registrations(), 1);
  } finally { if (host) await host.close(); await rm(workspace, { recursive: true, force: true }); }
});
