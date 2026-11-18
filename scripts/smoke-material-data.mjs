/** S03 fixed synthetic fixtures over real native Sessions/jobs; no LLM or training. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { confirmationRequirements } from '../harness-plugin/src/case-requirements.mjs';
import { command, connectHarness, prepareSmokeWorkspace, project, runStage, smokeOptions, writeSmokeEvidence } from './harness-rpc.mjs';

const options = smokeOptions('smoke-material-data');
if (options.help) process.exit(0);
const workspace = options.workspace;
const fixtureRoot = resolve(project, 'test-output');
const relativeWorkspace = relative(fixtureRoot, workspace);
assert.ok(relativeWorkspace && !isAbsolute(relativeWorkspace) && relativeWorkspace !== '..' && !relativeWorkspace.startsWith(`..${sep}`),
  'A fixed-confirmation smoke must use an isolated project test-output descendant.');
await prepareSmokeWorkspace(workspace);
try {
  const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
  assert.equal(Object.keys(manifest.artifacts ?? {}).length, 0, 'Use a fresh workspace; inspect previous job handles before rerunning.');
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const evidence = { schema_version: 1, status: 'RUNNING', confirmation_source: 'test_fixture', workspace,
  startedAt: new Date().toISOString(), scope: 'S03 real native material/upload/confirmation/data pipeline; fixed synthetic test_fixture; no LLM, NN training or service launch.',
  sessions: [], stages: [], admissions: [], rejections: [], fixtures: [], checks: [] };
await writeSmokeEvidence(options.evidence, evidence);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const json = file => readFile(file, 'utf8').then(JSON.parse);
const python = process.env.CONTROL_PYTHON ?? resolve(project, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const headers = ['time', 'trip', 'current', 'current', 'auxiliary', 'response_a', 'response_b'];
const rows = Array.from({ length: 120 }, (_, index) => {
  const trip = Math.floor(index / 20), step = index % 20;
  const requested = 20 * trip + step, measured = 20 * trip + .8 * step + .1, auxiliary = .3 * trip + .2 * step;
  return [step * .2, `synthetic-trip-${trip}`, requested, measured, auxiliary,
    2 * requested - .5 * measured + auxiliary, .1 * requested + 1.2 * measured - auxiliary];
});

function interfaceFor(worksheet) {
  const column = index => ({ index, header: headers[index], worksheet });
  return { schema_version: 1, interface_kind: 'tabular_supervised', name: 'S03 synthetic test_fixture 3in2out',
    task: 'static_regression', sample_time: null, time_column: null, trip_column: column(1),
    signals: [
      { name: 'requested_current', role: 'control', unit: 'A', column: column(2), delay_steps: 0, observable: true },
      { name: 'measured_current', role: 'output', unit: 'A', column: column(3), delay_steps: 0, observable: true },
      { name: 'auxiliary', role: 'auxiliary', unit: 'V', column: column(4), delay_steps: 0, observable: true },
      { name: 'response_a', role: 'output', unit: 'm', column: column(5), delay_steps: 0, observable: false },
      { name: 'response_b', role: 'output', unit: 'm/s', column: column(6), delay_steps: 0, observable: false },
    ], features: ['requested_current', 'measured_current', 'auxiliary'].map(signal => ({ signal, window: 1, lag: 0 })),
    targets: ['response_a', 'response_b'].map(signal => ({ signal, offset: 0 })), input_dim: 3, output_dim: 2 };
}

function near(actual, expected, label) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length, `${label} shape`);
    expected.forEach((value, i) => near(actual[i], value, `${label}[${i}]`));
  } else {
    assert.ok(Number.isFinite(actual), `${label} must be finite`);
    assert.ok(Math.abs(actual - expected) <= 1e-11 + 1e-10 * Math.abs(expected), `${label}: ${actual} != ${expected}`);
  }
}

function expectedScaler(values) {
  const location = values[0].map((_, column) => values.reduce((sum, row) => sum + row[column], 0) / values.length);
  const scale = location.map((mean, column) => Math.sqrt(values.reduce((sum, row) => sum + (row[column] - mean) ** 2, 0) / values.length) || 1);
  return { location, scale };
}

const xlsxGenerator = String.raw`import json,sys,openpyxl
from pathlib import Path
fixture=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
book=openpyxl.Workbook(); sheet=book.active; sheet.title='measured'
sheet.append(fixture['headers'])
for row in fixture['rows']: sheet.append(row)
book.save(sys.argv[2]); book.close()
print(json.dumps({'generator':'openpyxl','version':openpyxl.__version__,'confirmation_source':'test_fixture'}))
`;
const npzReader = String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as values:
    result={name:values[name].tolist() for name in values.files}
print(json.dumps(result,allow_nan=False))
`;

try {
  await mkdir(resolve(workspace, 'fixtures'), { recursive: true });
  await mkdir(resolve(workspace, '.case-plans'), { recursive: true });
  await writeSmokeEvidence(resolve(workspace, '.case-plans/test-fixture.json'), {
    kind: 'control-case-plan-test-fixture-v1', workspace, confirmation_source: 'test_fixture', synthetic: true,
  });
  const csvPath = resolve(workspace, 'fixtures/synthetic-3in2out.csv');
  const xlsxPath = resolve(workspace, 'fixtures/synthetic-3in2out.xlsx');
  const fixturePath = resolve(workspace, 'fixtures/synthetic-source.json');
  await writeSmokeEvidence(fixturePath, { confirmation_source: 'test_fixture', headers, rows });
  await writeFile(csvPath, [headers, ...rows].map(row => row.join(',')).join('\n') + '\n');
  evidence.xlsxGenerator = JSON.parse(execFileSync(python, ['-c', xlsxGenerator, fixturePath, xlsxPath], { encoding: 'utf8', windowsHide: true }));
  for (const path of [csvPath, xlsxPath]) evidence.fixtures.push({ path, sha256: hash(await readFile(path)), confirmation_source: 'test_fixture' });

  const baseRpc = await connectHarness({ logPath: options.webLog });
  // Persist admission handles as soon as /control returns, including a live
  // handle if runStage later times out. Never start a second worker on retry.
  const rpc = async (method, payload) => {
    const value = await baseRpc(method, payload);
    if (method === 'commands/execute' && value.result?.kind === 'success') {
      try {
        const body = JSON.parse(value.result.text);
        if (body.jobId) {
          evidence.admissions.push({ jobId: body.jobId, attemptId: body.attemptId, sessionId: payload.agentId,
            operation: payload.line.startsWith('/control ') ? JSON.parse(payload.line.slice(9)).operation : payload.line.split(' ')[0] });
          await writeSmokeEvidence(options.evidence, evidence);
        }
      } catch (error) { if (error instanceof SyntaxError) { /* non-JSON unrelated command */ } else throw error; }
    }
    return value;
  };
  const session = await rpc('session/create', { request: { cwd: workspace, agentPreset: 'control-model' } });
  const peer = await rpc('session/create', { request: { cwd: workspace, agentPreset: 'control-model' } });
  assert.equal(session.agentPreset, 'control-model');
  assert.equal(peer.agentPreset, 'control-model');
  evidence.sessions.push(session.sessionId, peer.sessionId);
  const stage = async (operation, inputs = {}, config = {}) => {
    const result = await runStage(rpc, session.sessionId, { agent: 'model', operation, workspace, inputs, config });
    evidence.stages.push({ operation, jobId: result.jobId, attemptId: result.attemptId, result: result.result });
    await writeSmokeEvidence(options.evidence, evidence);
    return result.result;
  };
  const events = async sessionId => (await rpc('session/events', { request: { address: { kind: 'session', sessionId }, maxMessages: 10000 } })).records;
  const snapshot = async () => {
    let attempts = [];
    try { attempts = (await readdir(resolve(workspace, '.control-jobs'))).filter(name => /^[0-9a-f-]{36}$/.test(name)).sort(); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const sessions = [];
    for (const sessionId of evidence.sessions) {
      const history = await command(rpc, sessionId, `/control-history ${JSON.stringify({ workspace })}`);
      const records = await events(sessionId);
      sessions.push({ sessionId, history: history.attempts.map(item => item.attemptId).sort(),
        registeredJobs: records.filter(row => row.event.type === 'job/registered').map(row => row.event.data.jobId ?? row.event.data.job?.id ?? row.seq) });
    }
    return { attempts, sessions };
  };
  const denied = async (label, request, sessionId = session.sessionId) => {
    const before = await snapshot();
    const response = await rpc('commands/execute', { agentId: sessionId, line: `/control ${JSON.stringify({ agent: 'model', ...request })}`, submittedAttachments: [] });
    assert.equal(response.result.kind, 'error', `${label} was admitted`);
    const after = await snapshot();
    assert.deepEqual(after, before, `${label} created history or registered a native job`);
    evidence.rejections.push({ label, nativeKind: response.result.kind, message: response.result.text,
      newHistory: 0, newNativeJobs: 0, before, after, confirmation_source: 'test_fixture' });
    await writeSmokeEvidence(options.evidence, evidence);
  };

  const sourceMetadata = { confirmation_source: 'test_fixture', description: 'Fixed bounded synthetic S03 fixture, no measured plant or training.' };
  const csvMaterial = await stage('register_material', { local_path: csvPath }, { source_kind: 'path', source_metadata: sourceMetadata });
  const xlsxPathMaterial = await stage('register_material', { local_path: xlsxPath }, { source_kind: 'path', source_metadata: sourceMetadata });
  const uploaded = await rpc('fileUploads/upload', { agentId: session.sessionId,
    request: { data: (await readFile(xlsxPath)).toString('base64'), name: 'synthetic-3in2out.xlsx' } });
  assert.equal(uploaded.file.bytes, (await readFile(xlsxPath)).length);
  assert.equal(uploaded.file.attachmentId, `sha256:${hash(await readFile(xlsxPath))}`);
  const attachment = await rpc('commands/execute', { agentId: session.sessionId, line: '/control-upload',
    submittedAttachments: [{ type: 'file', receiptId: uploaded.receiptId }] });
  assert.equal(attachment.result.kind, 'success', attachment.result.text);
  const admitted = JSON.parse(attachment.result.text);
  assert.equal(admitted.sha256, uploaded.file.attachmentId);
  const xlsxMaterial = await stage('register_material', { local_path: admitted.path }, {
    source_kind: 'upload', original_name: admitted.name,
    source_metadata: { ...sourceMetadata, attachment_id: uploaded.file.attachmentId, bytes: uploaded.file.bytes },
  });
  const xlsxPathDescription = await json(resolve(xlsxPathMaterial.artifact_path, 'material.json'));
  const xlsxDescription = await json(resolve(xlsxMaterial.artifact_path, 'material.json'));
  assert.equal(xlsxPathDescription.content_hash, xlsxDescription.content_hash);
  assert.equal(xlsxDescription.source.kind, 'upload');
  evidence.upload = { transport: 'native fileUploads/upload RPC', actualNativeUpload: true,
    receiptIdSha256: hash(uploaded.receiptId), file: uploaded.file, content_hash: xlsxDescription.content_hash,
    pathContentHash: xlsxPathDescription.content_hash, guiUploadValidated: false, guiGate: 'S13 remains required', confirmation_source: 'test_fixture' };

  for (const [format, material, sheet] of [['csv', csvMaterial, null], ['xlsx', xlsxMaterial, 'measured']]) {
    const description = await json(resolve(material.artifact_path, 'material.json'));
    const table = await stage('inspect_table', { data_path: resolve(material.artifact_path, description.primary_file) }, sheet ? { worksheet: sheet } : {});
    assert.deepEqual(table.worksheets[0].duplicate_headers, [{ header: 'current', indices: [2, 3] }]);
    assert.equal(table.worksheets[0].row_count, 120);
    const config = { interface: interfaceFor(sheet), split: { mode: 'trip', train: .5, validation: .2, test: .3 }, scaler: 'standard',
      source: { kind: 'synthetic', description: 'Fixed S03 test_fixture; no measured plant, data acquisition or NN training.' } };
    const request = { operation: 'prepare_supervised_dataset', workspace, inputs: { material: material.artifact_id }, config };
    const requirements = await confirmationRequirements(request);
    assert.equal(requirements.length, 1);
    assert.equal(requirements[0].scope, 'signals');
    const plan = await stage('propose_case_plan', {}, { scope: 'signals', payload: config.interface, bindings: requirements[0].bindings });
    const reference = { plan_id: plan.plan_id, version_hash: plan.version_hash };
    config.plan_refs = { signals: reference };
    await denied(`${format}: unconfirmed plan`, request);
    const confirmed = await command(rpc, session.sessionId, `/control-confirm ${JSON.stringify({ workspace, ...reference })}`);
    assert.equal(confirmed.confirmation_source, 'test_fixture');
    assert.equal(confirmed.version_hash, reference.version_hash);
    const status = await command(rpc, session.sessionId, `/control-plan-status ${JSON.stringify({ workspace, ...reference })}`);
    assert.equal(status.confirmation_state, 'confirmed');
    assert.equal(status.confirmation_source, 'test_fixture');
    const changedUnit = structuredClone(request);
    changedUnit.config.interface.signals[0].unit = 'mA';
    await denied(`${format}: changed unit`, changedUnit);
    const changedMaterial = structuredClone(request);
    changedMaterial.inputs.material = format === 'csv' ? xlsxPathMaterial.artifact_id : csvMaterial.artifact_id;
    await denied(`${format}: changed material ID`, changedMaterial);
    await denied(`${format}: different Session`, request, peer.sessionId);
    const prepared = await stage('prepare_supervised_dataset', request.inputs, config);
    assert.equal(prepared.kind, 'supervised_dataset');
    assert.equal(prepared.status, 'verified');
    const inspected = await stage('inspect_artifact', { artifact_id: prepared.artifact_id });
    assert.equal(inspected.report.training, 'not_run');
    assert.equal(inspected.report.physical_model, 'not_provided');
    assert.deepEqual(inspected.report.dimensions, { input_dim: 3, output_dim: 2 });
    assert.ok(inspected.report.confirmation_sources.length > 0);
    assert.ok(inspected.report.confirmation_sources.every(item => item.confirmation_source === 'test_fixture'));
    assert.ok(inspected.metadata.source.confirmation_sources.every(item => item.confirmation_source === 'test_fixture'));
    for (const name of ['nx', 'nu', 'ny', 'nd', 'states']) assert.ok(!(name in inspected.metadata.interface), `${format} fabricated ${name}`);
    const preview = await stage('read_artifact_file', { artifact_id: prepared.artifact_id, file: 'supervised.npz' });
    const rawArrays = JSON.parse(execFileSync(python, ['-c', npzReader, resolve(prepared.artifact_path, 'supervised.npz')], { encoding: 'utf8', windowsHide: true, maxBuffer: 2 * 1024 * 1024 }));
    const data = await json(resolve(prepared.artifact_path, 'dataset.json'));
    const scaler = await json(resolve(prepared.artifact_path, 'scaler.json'));
    const split = await json(resolve(prepared.artifact_path, 'split.json'));
    near(rawArrays.X, rows.map(row => row.slice(2, 5)), `${format} actual X`);
    near(rawArrays.y, rows.map(row => row.slice(5, 7)), `${format} actual y`);
    assert.deepEqual(rawArrays.split_id, [...Array(60).fill(0), ...Array(20).fill(1), ...Array(40).fill(2)]);
    assert.deepEqual(split.sample_counts, { train: 60, validation: 20, test: 40 });
    assert.equal(split.split_before_windowing, true);
    assert.equal(scaler.fit_split, 'train');
    assert.equal(scaler.fit_samples, 60);
    const expectedX = expectedScaler(rows.slice(0, 60).map(row => row.slice(2, 5)));
    const expectedY = expectedScaler(rows.slice(0, 60).map(row => row.slice(5, 7)));
    near(scaler.X.location, expectedX.location, `${format} train-only X mean`);
    near(scaler.X.scale, expectedX.scale, `${format} train-only X std`);
    near(scaler.y.location, expectedY.location, `${format} train-only y mean`);
    near(scaler.y.scale, expectedY.scale, `${format} train-only y std`);
    near(rawArrays.X_scaled, rows.map(row => row.slice(2, 5).map((value, i) => (value - expectedX.location[i]) / expectedX.scale[i])), `${format} transformed X`);
    assert.equal(data.plant_dimensions_inferred, false);
    assert.equal(data.trained, false);
    assert.equal(data.input_dim, 3);
    assert.equal(data.output_dim, 2);
    assert.ok(preview.file_preview.arrays.some(item => item.name === 'X' && item.shape[0] === 120 && item.shape[1] === 3));
    const fileHashes = {};
    for (const [name, expected] of Object.entries(inspected.metadata.files)) {
      fileHashes[name] = hash(await readFile(resolve(prepared.artifact_path, name)));
      assert.equal(fileHashes[name], expected, `${format} artifact ${name} hash`);
    }
    evidence.checks.push({ format, material: material.artifact_id, dataset: prepared.artifact_id,
      plan: reference, confirmed, confirmationStatus: status, confirmation_source: 'test_fixture',
      dimensions: { input_dim: 3, output_dim: 2 }, noPlant: true, training: 'not_run',
      samples: split.sample_counts, scalerIndependentlyChecked: true, fileHashes });
    await writeSmokeEvidence(options.evidence, evidence);
  }
  for (const fixture of evidence.fixtures) assert.equal(hash(await readFile(fixture.path)), fixture.sha256, 'Original fixture bytes changed');
  evidence.eventTypes = {};
  for (const sessionId of evidence.sessions) {
    const records = await events(sessionId);
    evidence.eventTypes[sessionId] = [...new Set(records.map(row => row.event.type))];
    assert.ok(!evidence.eventTypes[sessionId].some(type => type.startsWith('turn/')), 'Fixed S03 smoke opened an LLM turn');
  }
  evidence.status = 'PASS';
} catch (error) {
  evidence.status = 'FAIL';
  evidence.error = error instanceof Error ? error.stack : String(error);
  throw error;
} finally {
  evidence.finishedAt = new Date().toISOString();
  await writeSmokeEvidence(options.evidence, evidence);
  console.log(JSON.stringify({ status: evidence.status, confirmation_source: 'test_fixture', workspace,
    evidence: options.evidence, admissions: evidence.admissions.map(item => ({ operation: item.operation, jobId: item.jobId })),
    checks: evidence.checks.length, rejectionChecks: evidence.rejections.length, actualNativeUpload: evidence.upload?.actualNativeUpload ?? false }));
}
