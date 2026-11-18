/** S05 real native jobs on frozen synthetic F03; no LLM, physical plant or MPC. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command, connectHarness, prepareSmokeWorkspace, project, writeSmokeEvidence } from './harness-rpc.mjs';

const { values } = parseArgs({ options: { workspace: { type: 'string' }, evidence: { type: 'string' },
  fixtures: { type: 'string' }, 'web-log': { type: 'string' }, resume: { type: 'boolean' }, help: { type: 'boolean', short: 'h' } } });
if (values.help) {
  console.log('node scripts/smoke-supervised.mjs --workspace <isolated test-output path> --evidence <json> [--fixtures <S00 frozen fixtures>] [--web-log <Host log>] [--resume]');
  console.log('The script never starts/reloads a Host. --resume observes saved job IDs and never resubmits an uncertain admission.');
  process.exit(0);
}
assert.ok(values.workspace && values.evidence, 'Explicit workspace/evidence are required.');
const workspace = resolve(values.workspace), evidencePath = resolve(values.evidence);
const fixtureRoot = resolve(values.fixtures ?? process.env.CONTROL_S05_FIXTURES ?? resolve(project, 'test-output/v2-development/20261117T002006/fixtures'));
const descendant = relative(resolve(project, 'test-output'), workspace);
assert.ok(descendant && !isAbsolute(descendant) && descendant !== '..' && !descendant.startsWith(`..${sep}`), 'Fixed auto-confirmation is restricted to an isolated project test-output workspace.');
const python = process.env.CONTROL_PYTHON ?? resolve(project, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = path => readFile(path, 'utf8').then(JSON.parse);
const frozen = {
  'freeze_manifest.json': '8f820c97225be46a9d8a8172f7535767d6189a362d9b91f6d15011bb4aa2a7a7',
  'F03_spec.json': '63d0c812e7000dbec9727a07f70105856705be4d1348e0a5de5ea2cb2746af51',
  'F03_synthetic_3in2out.csv': '11a7547ecd08d46b7c04f9b02f707753479f18a4d062c2a2499206fa20343715',
  'acceptance_thresholds.json': '6c75841d83f18fdcf4f3794bb008227f32ca391c780fbbceed5705be0c48394f',
};
// Fixed before any candidate submission; final test errors never revise this.
const baseTrainingConfig = { hidden_dims: [32, 32], activation: 'tanh', input_dim: 3, output_dim: 2,
  epochs: 600, learning_rate: .003, batch_size: 64, patience: 80, seed: 20261120,
  weight_decay: 0, min_delta: 1e-8, grad_clip: 5, max_test_nrmse: .15 };
const trainingConfigs = { csv: { ...baseTrainingConfig, hidden_dims: [24, 16], learning_rate: .005 },
  xlsx: structuredClone(baseTrainingConfig) };
let evidence = values.resume ? await json(evidencePath) : { schema_version: 1, status: 'RUNNING', workspace,
  startedAt: new Date().toISOString(), confirmation_source: 'test_fixture',
  scope: 'S05 real native fixed synthetic data-only MLP training/export; no LLM, physical plant, PINN, MPC or policy training.',
  preRegisteredExpertConfigs: trainingConfigs, fixtures: {}, stages: {}, admissions: [], confirmations: [], rejections: [], checks: [] };
assert.equal(evidence.workspace, workspace, 'Saved run belongs to a different workspace.');
assert.deepEqual(evidence.preRegisteredExpertConfigs, trainingConfigs, 'A resumed observer must retain the original registered candidates.');
const save = () => writeSmokeEvidence(evidencePath, evidence);
await prepareSmokeWorkspace(workspace);
if (!values.resume) {
  try { assert.equal(Object.keys((await json(resolve(workspace, 'manifest.json'))).artifacts ?? {}).length, 0, 'Fresh script requires a new empty workspace; use --resume for existing handles.'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}
await save();

const near = (actual, expected, label, atol = 1e-11, rtol = 1e-10) => {
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), `${label} must be an array`);
    assert.equal(actual.length, expected.length, `${label} width`);
    expected.forEach((item, k) => near(actual[k], item, `${label}[${k}]`, atol, rtol));
  } else {
    assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= atol + rtol * Math.abs(expected), `${label}: ${actual} != ${expected}`);
  }
};
const expectedScaler = rows => {
  const location = rows[0].map((_, k) => rows.reduce((sum, row) => sum + row[k], 0) / rows.length);
  const scale = location.map((mean, k) => Math.sqrt(rows.reduce((sum, row) => sum + (row[k] - mean) ** 2, 0) / rows.length) || 1);
  return { location, scale };
};
const npzReader = String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as values: result={key:values[key].tolist() for key in values.files}
print(json.dumps(result,allow_nan=False))`;
const xlsxGenerator = String.raw`import csv,json,sys,openpyxl
with open(sys.argv[1],encoding='utf-8-sig',newline='') as stream:
    rows=list(csv.reader(stream))
book=openpyxl.Workbook(); sheet=book.active; sheet.title='frozen_F03'; sheet.append(rows[0])
for row in rows[1:]: sheet.append([row[0],int(row[1]),float(row[2]),row[3],*map(float,row[4:])])
book.save(sys.argv[2]);book.close()
print(json.dumps({'generator':'openpyxl','version':openpyxl.__version__,'source':'same frozen F03 table; not new candidate data'}))`;
const internalPredictor = String.raw`import json,sys,numpy as np,torch
from pathlib import Path
from control_harness.supervised_export import build_supervised_network
torch.set_num_threads(1)
root=Path(sys.argv[1]);network=build_supervised_network(json.loads((root/'network_config.json').read_text(encoding='utf-8')))
network.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True)['network_state'],strict=True);network.eval()
scaler=json.loads((root/'scaler.json').read_text(encoding='utf-8'))
raw=np.asarray(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))['features'],dtype=np.float64)
with torch.no_grad(): prediction=network(torch.from_numpy((raw-scaler['X']['location'])/scaler['X']['scale'])).numpy()*scaler['y']['scale']+scaler['y']['location']
print(json.dumps({'predictions':prediction.tolist(),'source':'internal selected model; separate process'},allow_nan=False))`;
const isolatedBundleRunner = String.raw`import importlib.abc,runpy,sys
class BlockProject(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='control_harness' or fullname.startswith('control_harness.'):
            raise ModuleNotFoundError('Independent bundle must not import control_harness')
sys.meta_path.insert(0,BlockProject())
script=sys.argv[1];sys.argv=sys.argv[1:]
runpy.run_path(script,run_name='__main__')
assert not any(key=='control_harness' or key.startswith('control_harness.') for key in sys.modules)`;
const actualTrainingState = String.raw`import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);selected=torch.load(root/'model.pt',map_location='cpu',weights_only=True)['network_state']
checkpoint=torch.load(root/'latest.pt',map_location='cpu',weights_only=True)
print(json.dumps({'selected_weight_shapes':{key:list(value.shape) for key,value in selected.items()},
    'checkpoint_weight_shapes':{key:list(value.shape) for key,value in checkpoint['model_state'].items()},
    'optimizer_param_groups':[{'lr':group['lr'],'weight_decay':group['weight_decay'],'parameter_tensor_count':len(group['params'])} for group in checkpoint['optimizer_state']['param_groups']],
    'optimizer_state_tensors':len(checkpoint['optimizer_state']['state']),'epoch':checkpoint['epoch'],'updates':checkpoint['updates']},allow_nan=False))`;
const runPython = (code, args, extra = {}) => execFileSync(python, ['-B', '-c', code, ...args], {
  encoding: 'utf8', windowsHide: true, maxBuffer: 8 * 1024 * 1024,
  env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', ...extra },
});

try {
  for (const [name, expected] of Object.entries(frozen)) {
    const actual = hash(await readFile(resolve(fixtureRoot, name)));
    assert.equal(actual, expected, `Frozen S00 bytes changed: ${name}`);
    evidence.fixtures[name] = { source: resolve(fixtureRoot, name), sha256: actual };
  }
  const specification = await json(resolve(fixtureRoot, 'F03_spec.json'));
  const thresholds = await json(resolve(fixtureRoot, 'acceptance_thresholds.json'));
  assert.equal(thresholds.gates.F03_NN.per_output_test_NRMSE_max, .15);
  near(thresholds.normalization.F03_targets.values, [.5815299806958877, .4532018605396638], 'frozen train-only target std', 0, 0);
  const text = await readFile(resolve(fixtureRoot, 'F03_synthetic_3in2out.csv'), 'utf8');
  const lines = text.trim().split(/\r?\n/), headers = lines.shift().split(',');
  assert.deepEqual(headers, ['trip_id', 'sample_id', 'time_s', 'split', 'f1', 'f2', 'f3', 'target_1', 'target_2']);
  const rows = lines.map(line => { const row = line.split(','); return [row[0], Number(row[1]), Number(row[2]), row[3], ...row.slice(4).map(Number)]; });
  assert.equal(rows.length, 1024);
  const groupOrder = [...new Set(rows.map(row => row[0]))];
  assert.equal(groupOrder.length, 32);
  assert.deepEqual(groupOrder, [...specification.group_split.train, ...specification.group_split.validation, ...specification.group_split.test]);
  for (const group of groupOrder) assert.equal(rows.filter(row => row[0] === group).length, 32);
  for (const row of rows) near(row.slice(7), [.8*row[4]-.5*row[5]+.3*row[6]+.2, -.2*row[4]+.4*row[5]+.6*row[6]+.1*Math.sin(Math.PI*row[4])], 'frozen analytic label');
  await mkdir(resolve(workspace, 'fixtures'), { recursive: true });
  const csvPath = resolve(workspace, 'fixtures/F03-frozen.csv'), xlsxPath = resolve(workspace, 'fixtures/F03-same-table.xlsx');
  if (!values.resume) {
    await copyFile(resolve(fixtureRoot, 'F03_synthetic_3in2out.csv'), csvPath);
    evidence.xlsxGenerator = JSON.parse(runPython(xlsxGenerator, [csvPath, xlsxPath]));
    evidence.generatedFixtureHashes = { csv: hash(await readFile(csvPath)), xlsx: hash(await readFile(xlsxPath)) };
    await writeSmokeEvidence(resolve(workspace, '.case-plans/test-fixture.json'), {
      kind: 'control-case-plan-test-fixture-v1', workspace, confirmation_source: 'test_fixture', synthetic: true,
    });
  }
  assert.equal(hash(await readFile(csvPath)), frozen['F03_synthetic_3in2out.csv']);
  assert.equal(hash(await readFile(xlsxPath)), evidence.generatedFixtureHashes.xlsx, 'The same generated workbook must survive observer resume.');
  await save();
  const rpc = await connectHarness({ logPath: values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG });
  if (!evidence.sessionId) {
    const session = await rpc('session/create', { request: { cwd: workspace, agentPreset: 'control-model' } });
    evidence.sessionId = session.sessionId;
    assert.equal(session.agentPreset, 'control-model');
    await save();
  }
  const sessionId = evidence.sessionId;
  const run = async (key, operation, inputs = {}, config = {}) => {
    const request = { agent: 'model', operation, workspace, inputs, config };
    const requestHash = hash(JSON.stringify(request));
    let entry = evidence.stages[key];
    if (entry) {
      assert.equal(entry.requestHash, requestHash, `${key} was changed while resuming its observer`);
      assert.ok(entry.jobId, `${key} has uncertain admission; reconcile native history before attaching its job ID. Never submit it twice.`);
      if (entry.result) return entry.result;
    } else {
      entry = evidence.stages[key] = { key, operation, requestHash, status: 'ADMISSION_PENDING' };
      await save();
      const started = await command(rpc, sessionId, `/control ${JSON.stringify(request)}`);
      Object.assign(entry, { jobId: started.jobId, attemptId: started.attemptId, status: 'RUNNING' });
      assert.ok(entry.jobId);
      evidence.admissions.push({ key, operation, jobId: entry.jobId, attemptId: entry.attemptId, sessionId });
      await save(); // Persist handles before the first observation can time out.
    }
    for (;;) {
      const state = await command(rpc, sessionId, `/control-status ${entry.jobId}`);
      entry.nativeStatus = state.job.status;
      if (['completed', 'failed', 'killed'].includes(state.job.status)) {
        entry.status = state.job.status;
        entry.terminal = { detail: state.job.detail ?? null, observedAt: new Date().toISOString() };
        await save();
        assert.equal(state.job.status, 'completed', JSON.stringify(state));
        const messages = state.output.chunks.map(chunk => chunk.text).join('').split(/\r?\n/).flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } });
        entry.result = messages.findLast(message => message.event === 'result')?.result;
        assert.ok(entry.result, `No complete worker result for existing job ${entry.jobId}`);
        await save();
        return entry.result;
      }
      await new Promise(done => setTimeout(done, 400));
    }
  };
  const authorize = async (key, operation, inputs, config) => {
    const request = { operation, workspace, inputs, config };
    const requirements = await confirmationRequirements(request);
    assert.deepEqual(requirements.map(item => item.scope), operation === 'train_supervised' ? ['signals', 'training'] : ['signals']);
    const refs = {};
    for (const item of requirements) {
      const plan = await run(`${key}:plan:${item.scope}`, 'propose_case_plan', {}, { scope: item.scope, payload: item.expectedPayload, bindings: item.bindings });
      refs[item.scope] = { plan_id: plan.plan_id, version_hash: plan.version_hash };
    }
    if (!evidence.rejections.some(item => item.key === key)) {
      const before = await command(rpc, sessionId, `/control-history ${JSON.stringify({ workspace })}`);
      const response = await rpc('commands/execute', { agentId: sessionId, line: `/control ${JSON.stringify({ agent: 'model', ...request, config: { ...config, plan_refs: refs } })}`, submittedAttachments: [] });
      // On resumed scripts a previously confirmed same payload is legitimate.
      // Only fresh authorization paths are tested before their confirmations.
      assert.equal(response.result.kind, 'error', `${key} was admitted without native confirmation`);
      const after = await command(rpc, sessionId, `/control-history ${JSON.stringify({ workspace })}`);
      assert.deepEqual(after.attempts.map(row => row.attemptId), before.attempts.map(row => row.attemptId), 'Unconfirmed training created a worker history attempt');
      evidence.rejections.push({ key, kind: response.result.kind, newHistory: 0, message: response.result.text });
      await save();
    }
    for (const ref of Object.values(refs)) {
      const confirmed = await command(rpc, sessionId, `/control-confirm ${JSON.stringify({ workspace, ...ref })}`);
      assert.equal(confirmed.confirmation_source, 'test_fixture');
      evidence.confirmations.push({ key, ...ref, confirmation_source: 'test_fixture' });
    }
    await save();
    return { ...config, plan_refs: refs };
  };
  const interfaceFor = worksheet => {
    const column = index => ({ index, header: headers[index], worksheet });
    return { schema_version: 1, interface_kind: 'tabular_supervised', name: 'Frozen synthetic F03 static 3in2out; no plant',
      task: 'static_regression', sample_time: null, time_column: null, trip_column: column(0),
      signals: headers.slice(4).map((name, k) => ({ name, role: k < 3 ? 'auxiliary' : 'output', unit: '1',
        column: column(k + 4), delay_steps: 0, observable: k < 3 })),
      features: headers.slice(4, 7).map(signal => ({ signal, window: 1, lag: 0 })),
      targets: headers.slice(7).map(signal => ({ signal, offset: 0 })), input_dim: 3, output_dim: 2 };
  };
  for (const [format, source, worksheet] of [['csv', csvPath, null], ['xlsx', xlsxPath, 'frozen_F03']]) {
    const trainingConfig = trainingConfigs[format];
    const material = await run(`${format}:material`, 'register_material', { local_path: source }, { source_kind: 'path',
      source_metadata: { kind: 'synthetic', description: 'Frozen F03 table; not real vehicle measurements' } });
    const dataConfig = await authorize(`${format}:dataset`, 'prepare_supervised_dataset', { material: material.artifact_id }, {
      interface: interfaceFor(worksheet), split: { mode: 'trip', train: .625, validation: .1875, test: .1875 }, scaler: 'standard',
      ...(worksheet ? { worksheet } : {}), source: { kind: 'synthetic', description: 'F03 S00 frozen synthetic groups; no physical model' },
    });
    const dataset = await run(`${format}:dataset`, 'prepare_supervised_dataset', { material: material.artifact_id }, dataConfig);
    assert.equal(dataset.kind, 'supervised_dataset');
    const split = await json(resolve(dataset.artifact_path, 'split.json'));
    assert.deepEqual(split.sample_counts, { train: 640, validation: 192, test: 192 });
    assert.equal(split.split_before_windowing, true);
    assert.equal(split.ordering, 'first-seen trip order');
    for (const name of ['train', 'validation', 'test']) assert.deepEqual(split.groups[name].map(item => item.replace(/^str:/, '')), specification.group_split[name]);
    const arrays = JSON.parse(runPython(npzReader, [resolve(dataset.artifact_path, 'supervised.npz')]));
    near(arrays.X, rows.map(row => row.slice(4, 7)), `${format} original feature order`);
    near(arrays.y, rows.map(row => row.slice(7)), `${format} original target order`);
    assert.deepEqual(arrays.split_id, [...Array(640).fill(0), ...Array(192).fill(1), ...Array(192).fill(2)]);
    const scaler = await json(resolve(dataset.artifact_path, 'scaler.json'));
    assert.equal(scaler.fit_split, 'train'); assert.equal(scaler.fit_samples, 640);
    for (const [key, expected] of [['X', expectedScaler(rows.slice(0, 640).map(row => row.slice(4, 7)))], ['y', expectedScaler(rows.slice(0, 640).map(row => row.slice(7)))]]) {
      near(scaler[key].location, expected.location, `${format} training-only ${key} location`);
      near(scaler[key].scale, expected.scale, `${format} training-only ${key} scale`);
    }
    near(scaler.y.scale, thresholds.normalization.F03_targets.values, `${format} frozen normalization`);
    const config = await authorize(`${format}:training`, 'train_supervised', { dataset: dataset.artifact_id }, structuredClone(trainingConfig));
    const trained = await run(`${format}:training`, 'train_supervised', { dataset: dataset.artifact_id }, config);
    assert.equal(trained.kind, 'supervised_model'); assert.equal(trained.status, 'verified');
    const metadata = await json(resolve(trained.artifact_path, 'artifact.json'));
    const report = await json(resolve(trained.artifact_path, 'report.json'));
    const resolvedConfig = await json(resolve(trained.artifact_path, 'resolved_config.json'));
    for (const [key, expected] of Object.entries(trainingConfig)) assert.deepEqual(resolvedConfig[key], expected, `${format} expert setting ${key}`);
    const actualState = JSON.parse(runPython(actualTrainingState, [trained.artifact_path]));
    const widths = [3, ...trainingConfig.hidden_dims, 2], expectedShapes = {};
    widths.slice(0, -1).forEach((left, layer) => {
      expectedShapes[`${2 * layer}.weight`] = [widths[layer + 1], left];
      expectedShapes[`${2 * layer}.bias`] = [widths[layer + 1]];
    });
    assert.deepEqual(actualState.selected_weight_shapes, expectedShapes, `${format} actual selected MLP layer widths`);
    assert.deepEqual(actualState.checkpoint_weight_shapes, expectedShapes, `${format} actual checkpoint layer widths`);
    assert.equal(actualState.optimizer_param_groups.length, 1);
    assert.equal(actualState.optimizer_param_groups[0].lr, trainingConfig.learning_rate, `${format} actual Adam learning rate`);
    assert.equal(actualState.optimizer_param_groups[0].weight_decay, trainingConfig.weight_decay);
    assert.equal(actualState.optimizer_param_groups[0].parameter_tensor_count, Object.keys(expectedShapes).length);
    assert.equal(actualState.optimizer_state_tensors, Object.keys(expectedShapes).length);
    assert.equal(actualState.epoch, report.trained_epochs);
    assert.equal(actualState.updates, Math.ceil(640 / trainingConfig.batch_size) * report.trained_epochs);
    assert.equal(actualState.updates, report.optimizer_updates);
    assert.deepEqual(metadata.parents.map(parent => parent.id), [dataset.artifact_id]);
    for (const key of ['nx', 'nu', 'ny', 'nd', 'states']) assert.ok(!(key in metadata.interface), `Data-only model fabricated ${key}`);
    assert.equal(report.algorithm, 'supervised_training');
    assert.equal(report.test_used_for_training_or_selection, false); assert.equal(report.test_evaluations, 1);
    assert.equal(report.normalization.source, 'training_partition_only');
    near(report.normalization.nrmse_scale, thresholds.normalization.F03_targets.values, `${format} reported NRMSE scale`);
    assert.equal(report.acceptance, 'passed'); assert.ok(report.best_epoch <= report.trained_epochs);
    assert.ok(report.confirmation_sources.every(item => item.confirmation_source === 'test_fixture'));
    const best = report.log.filter(item => item.selected_by_validation).at(-1);
    assert.equal(best.epoch, report.best_epoch);
    assert.equal(best.validation_loss, report.best_validation_loss);
    const selectedPredictions = JSON.parse(runPython(npzReader, [resolve(trained.artifact_path, 'test_predictions.npz')]));
    near(selectedPredictions.truth, rows.slice(832).map(row => row.slice(7)), `${format} held-out truth`);
    assert.deepEqual(selectedPredictions.row_index, Array.from({ length: 192 }, (_, k) => k + 832));
    const independentNRMSE = [0, 1].map(k => Math.sqrt(selectedPredictions.prediction.reduce((sum, row, i) => sum + (row[k] - rows[i + 832][k + 7]) ** 2, 0) / 192) / thresholds.normalization.F03_targets.values[k]);
    near(report.test.nrmse, independentNRMSE, `${format} independent test NRMSE`);
    assert.ok(independentNRMSE.every(value => value <= .15), `${format} frozen per-output quality gate failed`);
    const predictionInput = resolve(workspace, `fixtures/${format}-export-input.json`), predictionOutput = resolve(workspace, `fixtures/${format}-export-output.json`);
    await writeSmokeEvidence(predictionInput, { features: rows.slice(832).map(row => row.slice(4, 7)), feature_labels: ['f1@lag=0', 'f2@lag=0', 'f3@lag=0'] });
    const bundle = resolve(trained.artifact_path, report.export.directory);
    // A separate interpreter and cwd; no project PYTHONPATH is available to the bundle.
    execFileSync(python, ['-B', '-c', isolatedBundleRunner, resolve(bundle, 'predict.py'), '--input', predictionInput, '--output', predictionOutput], {
      cwd: bundle, encoding: 'utf8', windowsHide: true,
      env: { ...process.env, PYTHONPATH: '', PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' },
    });
    const exported = await json(predictionOutput);
    const internal = JSON.parse(runPython(internalPredictor, [trained.artifact_path, predictionInput], { PYTHONPATH: resolve(project, 'src') }));
    const tolerance = thresholds.gates.same_implementation_export_reload;
    near(exported.predictions, internal.predictions, `${format} standalone/internal process inference`, tolerance.atol, tolerance.rtol);
    near(exported.predictions, selectedPredictions.prediction, `${format} standalone/final frozen evaluation`, tolerance.atol, tolerance.rtol);
    assert.deepEqual(exported.output_units, ['1', '1']);
    for (const [name, expected] of Object.entries(metadata.files)) assert.equal(hash(await readFile(resolve(trained.artifact_path, name))), expected, `${format} artifact hash ${name}`);
    const check = { format, material: material.artifact_id, dataset: dataset.artifact_id, model: trained.artifact_id,
      split: split.sample_counts, group_split: specification.group_split, input_dim: 3, output_dim: 2, noPhysicalModel: true,
      expertConfig: trainingConfig, resolvedConfig, actualTrainingState: actualState, trained_epochs: report.trained_epochs, best_epoch: report.best_epoch,
      optimizer_updates: report.optimizer_updates, test_nrmse: independentNRMSE, normalization: report.normalization,
      standaloneInference: { separateProcess: true, projectImportRequired: false, projectImportsBlocked: true, tolerance, samples: 192 }, confirmation_source: 'test_fixture' };
    const previous = evidence.checks.findIndex(item => item.format === format);
    if (previous < 0) evidence.checks.push(check); else evidence.checks[previous] = check;
    await save();
  }
  const manifest = await json(resolve(workspace, 'manifest.json'));
  assert.ok(Object.values(manifest.artifacts).every(item => ['material', 'supervised_dataset', 'supervised_model'].includes(item.kind)), 'NN-only branch unexpectedly created a controller/physical model/policy.');
  const events = (await rpc('session/events', { request: { address: { kind: 'session', sessionId }, maxMessages: 10000 } })).records;
  evidence.eventTypes = [...new Set(events.map(row => row.event.type))];
  assert.ok(!evidence.eventTypes.some(type => type.startsWith('turn/')), 'Fixed native smoke unexpectedly opened an LLM turn.');
  for (const [name, expected] of Object.entries(frozen)) assert.equal(hash(await readFile(resolve(fixtureRoot, name))), expected);
  evidence.status = 'PASS'; evidence.finishedAt = new Date().toISOString();
} catch (error) {
  evidence.status = 'FAIL'; evidence.error = error instanceof Error ? error.stack : String(error);
  evidence.requiresObserverResumeOrHistoryReconciliation = Object.values(evidence.stages).some(item => !['completed', 'failed', 'killed'].includes(item.status));
  throw error;
} finally {
  await save();
  console.log(JSON.stringify({ status: evidence.status, workspace, evidence: evidencePath, checks: evidence.checks.length,
    admissions: evidence.admissions.map(({ operation, jobId }) => ({ operation, jobId })), confirmation_source: 'test_fixture' }));
}
