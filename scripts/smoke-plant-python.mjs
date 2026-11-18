/** S04 real native Python plant trials/batches against frozen independent F01/F02 references. */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, isAbsolute, relative, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command, connectHarness, prepareSmokeWorkspace, project, runStage, writeSmokeEvidence } from './harness-rpc.mjs';

const { values } = parseArgs({ options: { workspace: { type: 'string' }, evidence: { type: 'string' },
  'web-log': { type: 'string' }, fixtures: { type: 'string' }, help: { type: 'boolean', short: 'h' } } });
if (values.help) {
  console.log('node scripts/smoke-plant-python.mjs --fixtures <absolute frozen fixtures directory> [--workspace <test-output descendant>] [--evidence <json>] [--web-log <log>]');
  process.exit(0);
}
const fixtures = values.fixtures ?? process.env.CONTROL_FROZEN_FIXTURES;
assert.ok(fixtures && isAbsolute(fixtures), 'Select the explicit S00 frozen fixtures directory with --fixtures or CONTROL_FROZEN_FIXTURES');
const workspace = resolve(values.workspace ?? process.env.CONTROL_SMOKE_WORKSPACE ?? resolve(project, 'test-output', `plant-python-${randomUUID()}`));
const evidencePath = resolve(values.evidence ?? process.env.CONTROL_SMOKE_EVIDENCE ?? resolve(project, 'test-output/smoke-evidence', `plant-python-${randomUUID()}.json`));
const webLog = resolve(values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG ?? resolve(project, 'harness-web.log'));
const distance = relative(resolve(project, 'test-output'), workspace);
assert.ok(distance && !isAbsolute(distance) && distance !== '..' && !distance.startsWith(`..${sep}`), 'Use an isolated project test-output workspace for fixed test_fixture confirmation');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const json = path => readFile(path, 'utf8').then(JSON.parse);
const sleep = ms => new Promise(resolveWait => setTimeout(resolveWait, ms));
const python = process.env.CONTROL_PYTHON ?? resolve(project, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const pythonEnv = { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' };
const evidence = { schema_version: 1, confirmation_source: 'test_fixture', status: 'RUNNING', workspace,
  startedAt: new Date().toISOString(), scope: 'S04 real native registration/trial/sampling only; frozen independent references; no NN training, model turns or service launch',
  admissions: [], stages: [], confirmations: [], rejections: [], checks: [] };
await writeSmokeEvidence(evidencePath, evidence);

const f01Source = `import numpy as np
def dynamics(x,u,d,dt,p):
    return np.array([x[1],(u[0]+d[0]-p['damping']*x[1]-p['spring']*x[0])/p['mass']],dtype=np.float64)
`;
const f02Source = `import numpy as np
class Plant:
    def __init__(self,p): self.p=p; self.sample_time=p['dt']
    def reset(self,x0): self.state=np.asarray(x0,dtype=np.float64).copy()
    def step(self,u,d):
        self.state=np.asarray(self.p['A'],dtype=np.float64)@self.state+np.asarray(self.p['B'],dtype=np.float64)@u+np.asarray(self.p['E'],dtype=np.float64)@d
        return self.state.copy()
def create_plant(p): return Plant(p)
`;
const poisonSource = `raise RuntimeError('S04_SOURCE_EXECUTED_BEFORE_PLAN_VALIDATION')
def dynamics(x,u,d,dt,p): return x
`;
const readArrays = String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as data: print(json.dumps({k:data[k].tolist() for k in data.files},allow_nan=False))
`;

function same(actual, expected, label) { assert.deepEqual(actual, expected, label); }
function near(actual, expected, gate, label) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, `${label} width`);
    return expected.reduce((max, value, index) => Math.max(max, near(actual[index], value, gate, `${label}[${index}]`)), 0);
  }
  assert.ok(Number.isFinite(actual), `${label} is not finite`);
  const error = Math.abs(actual - expected);
  assert.ok(error <= gate.atol + gate.rtol * Math.abs(expected), `${label}: ${actual} != ${expected}`);
  return error;
}
const matvec = (matrix, vector) => matrix.map(row => row.reduce((sum, coefficient, index) => sum + coefficient * vector[index], 0));
function springReference(x, u, d, dt, parameters) {
  const alpha = parameters.damping / (2 * parameters.mass), w = Math.sqrt(parameters.spring / parameters.mass - alpha * alpha);
  const equilibrium = (u[0] + d[0]) / parameters.spring, position = x[0] - equilibrium;
  const decay = Math.exp(-alpha * dt), sine = Math.sin(w * dt) / w, cosine = Math.cos(w * dt);
  return [equilibrium + decay * ((cosine + alpha * sine) * position + sine * x[1]),
    decay * (-parameters.spring / parameters.mass * sine * position + (cosine - alpha * sine) * x[1])];
}
function linearReference(x, u, d, model) {
  return model.A.map((row, i) => row.reduce((sum, value, j) => sum + value * x[j], 0)
    + model.B[i].reduce((sum, value, j) => sum + value * u[j], 0) + model.E[i].reduce((sum, value, j) => sum + value * d[j], 0));
}
function explicitSampling(interfaceValue, conditions, seed) {
  return { mode: 'expert', seed, steps: 50, trajectory_count: conditions.length,
    dt: interfaceValue.sample_time, control_rate_limits: null, disturbance_rate_limits: null, coupled_constraints: [],
    trajectories: conditions.map((item, trajectory_id) => ({ trajectory_id, x0: item.x0.slice(), u0: item.u.slice(), d0: item.d.slice(),
      u_profile: { kind: 'step', times: [0], values: [item.u.slice()] }, d_profile: { kind: 'step', times: [0], values: [item.d.slice()] } })) };
}

try {
  await prepareSmokeWorkspace(workspace);
  try { assert.equal(Object.keys((await json(resolve(workspace, 'manifest.json'))).artifacts).length, 0, 'Use a fresh workspace'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const freeze = await json(resolve(fixtures, 'freeze_manifest.json'));
  const frozen = {};
  for (const name of ['F01_spring_2x1.json', 'F02_existing_linear_4x2.json', 'acceptance_thresholds.json']) {
    const declared = freeze.files.find(row => basename(row.path) === name);
    const bytes = await readFile(resolve(fixtures, name));
    assert.equal(sha(bytes), declared?.sha256, 'Frozen fixtures changed; no candidate code may redefine this oracle');
    frozen[name] = JSON.parse(bytes);
  }
  evidence.frozenVersion = freeze.version;
  const f01 = frozen['F01_spring_2x1.json'], f02 = frozen['F02_existing_linear_4x2.json'];
  const gates = frozen['acceptance_thresholds.json'].gates;
  const physics = { mass: f01.physical_parameters.mass_kg, damping: f01.physical_parameters.damping_Ns_per_m, spring: f01.physical_parameters.spring_N_per_m };
  const f01Interface = { ...structuredClone(f01.source_model.interface), time_semantics: 'discrete_state_transition' };
  const f01Conditions = [
    { x0: f01.known_probe.x0, u: [f01.known_probe.u], d: [f01.known_probe.d] },
    { x0: [-.1,.05], u: [.2], d: [.03] }, { x0: [0,0], u: [-.15], d: [-.01] },
    { x0: [.05,-.05], u: [.1], d: [0] }, { x0: [-.15,.08], u: [0], d: [.02] },
  ];
  const known = f02.known_probes.nominal;
  const f02Conditions = [{ x0: known.x, u: known.u, d: known.d },
    { x0: [-.2,.1,.2,-.1], u: [.1,.15], d: [-.03] }, { x0: [0,0,0,0], u: [-.1,.05], d: [.01] },
    { x0: [.3,-.1,-.2,.1], u: [0,0], d: [0] }, { x0: [-.1,-.2,.1,.2], u: [-.2,-.1], d: [.02] }];
  await mkdir(resolve(workspace, '.case-plans'), { recursive: true });
  await writeSmokeEvidence(resolve(workspace, '.case-plans/test-fixture.json'), { kind: 'control-case-plan-test-fixture-v1', workspace });
  await mkdir(resolve(workspace, 'sources'), { recursive: true });
  for (const [name, text] of [['f01_numpy.py', f01Source], ['f02_factory.py', f02Source], ['poison.py', poisonSource]]) {
    await writeFile(resolve(workspace, 'sources', name), text);
  }
  const baseRpc = await connectHarness({ logPath: webLog });
  const rpc = async (method, payload) => {
    const response = await baseRpc(method, payload);
    if (method === 'commands/execute' && response.result?.kind === 'success') {
      try {
        const body = JSON.parse(response.result.text);
        if (body.jobId) { evidence.admissions.push({ jobId: body.jobId, attemptId: body.attemptId, sessionId: payload.agentId }); await writeSmokeEvidence(evidencePath, evidence); }
      } catch (error) { if (!(error instanceof SyntaxError)) throw error; }
    }
    return response;
  };
  const session = await rpc('session/create', { request: { cwd: workspace, agentPreset: 'control-model' } });
  evidence.sessionId = session.sessionId;
  const stage = async (operation, inputs = {}, config = {}) => {
    const row = await runStage(rpc, session.sessionId, { agent: 'model', operation, workspace, inputs, config });
    evidence.stages.push({ operation, jobId: row.jobId, attemptId: row.attemptId, result: row.result });
    await writeSmokeEvidence(evidencePath, evidence);
    return row.result;
  };
  async function register(filename, declaration) {
    const material = await stage('register_material', { local_path: resolve(workspace, 'sources', filename) },
      { source_kind: 'path', source_metadata: { confirmation_source: 'test_fixture', description: 'Fixed S04 Python source from frozen mathematical oracle, no learned coefficients' } });
    const materialInfo = await json(resolve(material.artifact_path, 'material.json'));
    const adapter = { ...declaration, source_file: materialInfo.primary_file };
    const plant = await stage('register_plant', { material: material.artifact_id }, { adapter });
    const report = await json(resolve(plant.artifact_path, 'report.json'));
    assert.equal(report.source_executed, false);
    return { ...plant, adapter, material: material.artifact_id };
  }
  async function confirm(request) {
    const requirements = await confirmationRequirements(request);
    const refs = {}, dependencies = {};
    for (const requirement of requirements) {
      const plan = await stage('propose_case_plan', {}, { scope: requirement.scope, payload: requirement.expectedPayload,
        bindings: requirement.bindings, dependencies: structuredClone(dependencies) });
      refs[plan.scope] = { plan_id: plan.plan_id, version_hash: plan.version_hash };
      dependencies[plan.scope] = refs[plan.scope];
    }
    request.config.plan_refs = refs;
    return async () => {
      for (const reference of Object.values(refs)) {
        const result = await command(rpc, session.sessionId, `/control-confirm ${JSON.stringify({ workspace, ...reference })}`);
        assert.equal(result.confirmation_source, 'test_fixture'); evidence.confirmations.push(result);
      }
      await writeSmokeEvidence(evidencePath, evidence);
    };
  }
  const snapshot = async () => {
    let attempts = [];
    try { attempts = (await readdir(resolve(workspace, '.control-jobs'))).sort(); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const page = await rpc('session/events', { request: { address: { kind: 'session', sessionId: session.sessionId }, maxMessages: 10000 } });
    return { attempts, registered: page.records.filter(row => row.event.type === 'job/registered').map(row => row.event.data.jobId) };
  };
  async function unconfirmed(request) {
    const before = await snapshot();
    const response = await rpc('commands/execute', { agentId: session.sessionId, line: `/control ${JSON.stringify({ agent: 'model', ...request })}`, submittedAttachments: [] });
    assert.equal(response.result.kind, 'error');
    assert.deepEqual(await snapshot(), before);
    evidence.rejections.push({ kind: 'unconfirmed', zeroHistoryAndJobs: true, detail: response.result.text });
  }
  function compare(artifact, reference, outputMatrix, gate, { strict = true } = {}) {
    return Promise.all([json(resolve(artifact.artifact_path, 'sampling-plan.json')),
      Promise.resolve(JSON.parse(execFileSync(python, ['-c', readArrays, resolve(artifact.artifact_path, 'transitions.npz')], { encoding: 'utf8', env: pythonEnv, windowsHide: true, maxBuffer: 4 * 1024 * 1024 })))])
      .then(([plan, arrays]) => {
        let row = 0, maxError = 0;
        const checkGate = strict ? gate : { atol: Infinity, rtol: 0 };
        for (const trajectory of plan.trajectories) {
          let x = trajectory.x0.slice();
          for (let k = 0; k < plan.steps; k += 1, row += 1) {
            same(arrays.trajectory_id[row], trajectory.trajectory_id, 'trajectory id');
            same(arrays.u[row], trajectory.u[k], 'planned controls'); same(arrays.d[row], trajectory.d[k], 'planned disturbance');
            near(arrays.time[row], trajectory.t[k], { atol: 1e-13, rtol: 0 }, 'sample time');
            maxError = Math.max(maxError, near(arrays.x[row], x, checkGate, 'independent current state'));
            near(arrays.y[row], matvec(outputMatrix, x), checkGate, 'independent current output');
            x = reference(x, trajectory.u[k], trajectory.d[k], plan.dt);
            maxError = Math.max(maxError, near(arrays.next_x[row], x, checkGate, 'independent next state'));
            near(arrays.next_y[row], matvec(outputMatrix, x), checkGate, 'independent next output');
          }
        }
        assert.equal(arrays.x.length, row);
        return { plan, arrays, maxError, samples: row };
      });
  }
  async function collect(plant, sampling, fixtureId, reference, C, gate, strict = true) {
    const request = { operation: 'sample_model', workspace, inputs: { model: plant.artifact_id },
      config: { sampling, trial: { trajectories: 1, steps: 3 }, sampling_run: 'trial' } };
    const approve = await confirm(request);
    await unconfirmed(request); await approve();
    const trial = await stage('sample_model', request.inputs, request.config);
    assert.equal(trial.kind, 'sampling_trial');
    const preview = await compare(trial, reference, C, gate, { strict });
    request.config.sampling_run = 'full';
    // Same exact engineering version, selected full run; no second approval.
    const full = await stage('sample_model', request.inputs, request.config);
    assert.equal(full.kind, 'sampled_dataset');
    const actual = await compare(full, reference, C, gate, { strict });
    same(actual.arrays.x.slice(0, 3), preview.arrays.x, 'trial preserves exact full-plan prefix');
    same(actual.arrays.next_x.slice(0, 3), preview.arrays.next_x, 'trial next-state prefix');
    const report = await json(resolve(full.artifact_path, 'report.json'));
    assert.equal(report.training, 'not_run');
    assert.ok(report.confirmation_sources.every(item => item.confirmation_source === 'test_fixture'));
    evidence.checks.push({ fixtureId, model: plant.artifact_id, trial: trial.artifact_id, full: full.artifact_id,
      samples: actual.samples, maxAbsStateError: actual.maxError, frozenGate: gate, finalAccuracyGateApplied: strict,
      trialSharesFullPrefix: true, repeatedConfirmationForFull: false, confirmation_source: 'test_fixture' });
    await writeSmokeEvidence(evidencePath, evidence);
    return { request, full, ...actual };
  }

  const springSpec = substeps => ({ schema_version: 1, backend: 'python_callable', interface: f01Interface,
    parameters: physics, source_semantics: 'derivative', entrypoint: 'dynamics', C: f01.source_model.C,
    time_unit: 's', derivative_units: f01.source_model.derivative_units,
    integration: { method: 'rk4', substeps, input_hold: 'zero_order_hold' } });
  const springPlan = explicitSampling(f01Interface, f01Conditions, f01.seed);
  const springOracle = (x, u, d, dt) => springReference(x, u, d, dt, physics);
  const coarse = await collect(await register('f01_numpy.py', springSpec(1)), springPlan, 'F01_RK4_1_convergence', springOracle, f01.source_model.C, gates.F01_exact_ZOH_single_and_multi_step, false);
  const fine = await collect(await register('f01_numpy.py', springSpec(8)), springPlan, 'F01_RK4_8', springOracle, f01.source_model.C, gates.F01_exact_ZOH_single_and_multi_step);
  assert.ok(fine.maxError < coarse.maxError / 16, 'RK4 refinement must converge toward the frozen independent analytic solution');
  near(fine.arrays.next_x[0], f01.known_probe.next_x, gates.F01_exact_ZOH_single_and_multi_step, 'Frozen F01 known probe');
  evidence.springConvergence = { coarseSubsteps: 1, fineSubsteps: 8, coarseError: coarse.maxError, fineError: fine.maxError, claimsExactZoh: false };
  assert.deepEqual(f02.nominal.terms, []);
  const factorySpec = { schema_version: 1, backend: 'python_factory', interface: f02.nominal.interface,
    factory: 'create_plant', reset_strategy: 'recreate', source_semantics: 'transition', C: f02.nominal.C,
    parameters: { A: f02.nominal.A, B: f02.nominal.B, E: f02.nominal.E, dt: f02.sample_time_s } };
  const factory = await register('f02_factory.py', factorySpec);
  const linear = await collect(factory, explicitSampling(f02.nominal.interface, f02Conditions, f02.seed), 'F02_numpy_factory',
    (x, u, d) => linearReference(x, u, d, f02.nominal), f02.nominal.C, gates.F02_discrete_single_and_multi_step);
  near(linear.arrays.next_x[0], known.next_x, gates.F02_discrete_single_and_multi_step, 'Frozen F02 known probe');
  const repeated = await stage('sample_model', linear.request.inputs, linear.request.config);
  const repeatedData = await compare(repeated, (x, u, d) => linearReference(x, u, d, f02.nominal), f02.nominal.C, gates.F02_discrete_single_and_multi_step);
  same(repeatedData.arrays, linear.arrays, 'Independent factory resets produce identical repeated batches');
  evidence.factoryBatchRepeat = { first: linear.full.artifact_id, second: repeated.artifact_id, exactArrays: true };

  // A source which would throw on import provides observable evidence that a
  // bad plan is rejected before loading it, rather than merely skipping step().
  const poison = await register('poison.py', { schema_version: 1, backend: 'python_callable', interface: f01Interface,
    parameters: {}, source_semantics: 'transition', entrypoint: 'dynamics', C: f01.source_model.C });
  const proposed = await stage('recommend_sampling', { model: poison.artifact_id }, { overrides: { steps: 50 } });
  assert.equal(proposed.source_executed, false);
  const badSampling = explicitSampling(f01Interface, f01Conditions, f01.seed);
  badSampling.trajectories[0].x0[0] = f01Interface.states[0].upper + 1;
  const bad = { operation: 'sample_model', workspace, inputs: { model: poison.artifact_id }, config: { sampling: badSampling } };
  await (await confirm(bad))();
  const before = await json(resolve(workspace, 'manifest.json'));
  const started = await command(rpc, session.sessionId, `/control ${JSON.stringify({ agent: 'model', ...bad })}`);
  let rejected;
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    rejected = await command(rpc, session.sessionId, `/control-status ${started.jobId}`);
    if (['completed', 'failed', 'killed'].includes(rejected.job.status)) break;
    await sleep(200);
  }
  assert.equal(rejected.job.status, 'failed');
  const text = JSON.stringify(rejected);
  assert.match(text, /outside declared interface bounds/);
  assert.ok(!text.includes('S04_SOURCE_EXECUTED_BEFORE_PLAN_VALIDATION'), 'Out-of-bounds plan imported poison source');
  assert.deepEqual((await json(resolve(workspace, 'manifest.json'))).active, before.active, 'Invalid sampling cannot replace active datasets');
  evidence.rejections.push({ kind: 'out_of_bounds_before_source_import', jobId: started.jobId,
    workerValidationRan: true, plantSourceExecuted: false, importPoisonAvoided: true, detail: rejected.job.detail });
  const page = await rpc('session/events', { request: { address: { kind: 'session', sessionId: session.sessionId }, maxMessages: 10000 } });
  evidence.eventTypes = [...new Set(page.records.map(row => row.event.type))];
  assert.ok(!evidence.eventTypes.some(type => type.startsWith('turn/')));
  evidence.status = 'PASS';
} catch (error) {
  evidence.status = 'FAIL'; evidence.error = error.stack ?? String(error); throw error;
} finally {
  evidence.finishedAt = new Date().toISOString(); await writeSmokeEvidence(evidencePath, evidence);
  console.log(JSON.stringify({ status: evidence.status, evidence: evidencePath, workspace, sessionId: evidence.sessionId,
    checks: evidence.checks.length, rejections: evidence.rejections.length, admissions: evidence.admissions.length }));
}
