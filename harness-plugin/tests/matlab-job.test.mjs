/** Synthetic protocol fixtures; these tests never start MATLAB or a worker. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
import test from 'node:test';
import { assertMatlabBatchAvailable, matlabCollectionJob, requiresMatlabBatch, taskHelpers } from '../src/matlab-job.mjs';

const root = resolve(import.meta.dirname, '../../test-output/matlab-job-tests');
await mkdir(root, { recursive: true });
const sha = value => createHash('sha256').update(value).digest('hex');
const save = (path, value) => writeFile(path, JSON.stringify(value));
async function fixture(t) {
  const workspace = await mkdtemp(resolve(root, 'fixture-'));
  t.after(() => rm(workspace, { recursive: true, force: true }));
  const request = { operation: 'sample_model', workspace, inputs: { model: 'plant-model-test' }, config: { sampling: { fixture: true } } };
  const rawRequest = JSON.stringify(request);
  const config = { codeRoot: resolve(import.meta.dirname, '../..'), matlabResource: { directory: resolve(workspace, 'resource'), identityFile: resolve(workspace, 'identity.json') } };
  const job = { updateProgress() {}, append() {} };
  const grant = {get scope(){return {helper:taskHelpers[request.operation],dispatch_timeout_seconds:300};},
    deadline:Date.now()+1800000,signal:new AbortController().signal,revoke(){},finish(){}};
  return { workspace, request, grant, confirmation: { rawRequest, env: {} }, config, job, policy: { mode: 'workspace-write' }, agent: { id: 'test-session' }, slots: {}, phases: [] };
}
function protocol(f, { finalStatus = 'completed', toolError = false, hold = false, wrongKind = false } = {}) {
  let release, enteredResolve;
  const entered = new Promise(resolve => { enteredResolve = resolve; });
  const pending = new Promise(resolve => { release = resolve; });
  const numerical = args => {
    assert.strictEqual(args.request, f.request);
    assert.strictEqual(args.confirmation, f.confirmation);
    assert.strictEqual(args.policy, f.policy);
    f.phases.push(args.extraEnv.CONTROL_MATLAB_PHASE);
    return { cancel() {}, done: (async () => {
      if (args.extraEnv.CONTROL_MATLAB_PHASE === 'prepare') {
        const env = args.extraEnv, exchange = env.CONTROL_MATLAB_EXCHANGE_DIR;
        assert.equal(resolve(exchange, '..'), resolve(f.workspace, '.case-plans/matlab-exchanges'));
        await mkdir(exchange);
        const manifest = { nonce: env.CONTROL_MATLAB_NONCE, request_sha256: env.CONTROL_MATLAB_REQUEST_SHA256, plan: { plan_hash: 'plan-fixture' }, spec_hash: 'spec-fixture' };
        if(f.request.operation==='collect_external_expert')manifest.kind=wrongKind?'unsupported-batch':'control-matlab-expert-batch-v1';
        if(f.request.operation==='evaluate_external_imitation')manifest.kind=wrongKind?'unsupported-batch':'control-matlab-policy-evaluation-v1';
        const manifestPath = resolve(exchange, 'manifest.json'); await save(manifestPath, manifest);
        const prepared = { status: 'verified', matlab_phase: 'prepared', nonce: manifest.nonce,
          request_sha256: manifest.request_sha256, manifest_path: manifestPath,
          manifest_sha256: sha(await readFile(manifestPath)), exchange_dir: exchange,
          result_path: resolve(exchange, 'result.json'), plan_hash: 'plan-fixture' };
        if(f.request.operation==='collect_external_expert')prepared.batch_kind='external_expert';
        if(f.request.operation==='evaluate_external_imitation')prepared.batch_kind='external_policy_evaluation';
        args.onEvent({ event: 'result', result: prepared });
        return { status: 'completed', cleanupVerified: true, result: JSON.stringify(prepared) };
      }
      assert.equal(args.extraEnv.CONTROL_MATLAB_REQUEST_SHA256, sha(f.confirmation.rawRequest));
      assert.equal(sha(await readFile(args.extraEnv.CONTROL_MATLAB_RESULT_PATH)), args.extraEnv.CONTROL_MATLAB_RESULT_SHA256);
      args.onEvent({ event: 'result', result: { status: finalStatus === 'completed' ? 'verified' : 'rejected', artifact_id: 'fixture-artifact' } });
      return { status: finalStatus, cleanupVerified: true, detail: finalStatus === 'failed' ? 'fixture domain failure' : undefined };
    })() };
  };
  const ctx = { tools: {
    definitions: () => [{ name: 'mcp__matlab__evaluate_matlab_code' }],
    async execute(args) {
      assert.strictEqual(args.agent, f.agent);
      assert.equal(args.calls.length, 1);
      const call = args.calls[0]; assert.equal(call.name, 'mcp__matlab__evaluate_matlab_code');
      const params = JSON.parse(call.arguments);
      assert.ok(params.code.includes(f.request.operation==='evaluate_external_imitation'?'control_harness_policy_evaluation':f.request.operation==='collect_external_expert'?'control_harness_expert_batch':'control_harness_collect_batch'));
      const manifest = JSON.parse(await readFile(resolve(params.project_path, 'manifest.json'), 'utf8'));
      enteredResolve();
      if (hold) await pending;
      if (toolError) return { results: [{ result: { isError: true, message: 'fixture native denial' } }] };
      args.signal.throwIfAborted();
      const resultPath = resolve(params.project_path, 'result.json');
      await save(resultPath, { nonce: manifest.nonce, request_sha256: manifest.request_sha256,
        plan_hash: manifest.plan.plan_hash, spec_hash: manifest.spec_hash });
      return { results: [{ result: { isError: false, value: { content: [{ type: 'text',
        text: `CONTROL_MATLAB_BATCH_${manifest.nonce}:${JSON.stringify({ nonce: manifest.nonce, status: 'completed', result_path: resultPath })}` }] } } }] };
    },
  } };
  // Stand in only for the separate native resource/guard fixture. Both worker
  // phases still see the original exact request and exchange paths.
  const dispatch = ({ctx,agent,grant,signal,manifestPath,nonce,onDispatch}) => { onDispatch(); return ctx.tools.execute({agent,signal,calls:[{
    name:'mcp__matlab__evaluate_matlab_code',id:'fixture-'+nonce,
    arguments:JSON.stringify({code:grant.scope.helper, project_path:resolve(manifestPath,'..')})}]}); };
  return { ctx, numerical, entered, release, dispatch };
}

test('prepare result never settles the native job before MCP and finalization', async t => {
  const f = await fixture(t), p = protocol(f, { hold: true });
  const events = [], producer = matlabCollectionJob({ ...f, ...p, onEvent: event => events.push(event) });
  let settled = false; producer.done.then(() => { settled = true; });
  await p.entered;
  assert.equal(settled, false);
  assert.deepEqual(f.phases, ['prepare']);
  assert.equal(events[0].result.matlab_phase, 'prepared');
  p.release();
  assert.equal((await producer.done).status, 'completed');
  assert.deepEqual(f.phases, ['prepare', 'finalize']);
  assert.equal(events.at(-1).result.artifact_id, 'fixture-artifact');
});

test('finalize domain failure overrides a completed prepare phase', async t => {
  const f = await fixture(t), p = protocol(f, { finalStatus: 'failed' });
  const outcome = await matlabCollectionJob({ ...f, ...p }).done;
  assert.equal(outcome.status, 'failed');
  assert.equal(outcome.cleanupVerified, true);
  assert.deepEqual(f.phases, ['prepare', 'finalize']);
});

test('native error never runs finalize or claims MATLAB cleanup', async t => {
  const f = await fixture(t), p = protocol(f, { toolError: true });
  const outcome = await matlabCollectionJob({ ...f, ...p }).done;
  assert.equal(outcome.status, 'failed');
  assert.equal(outcome.cleanupVerified, false);
  assert.deepEqual(f.phases, ['prepare']);
});

test('cancel during the MCP phase preserves uncertainty and never finalizes', async t => {
  const f = await fixture(t), p = protocol(f, { hold: true });
  const producer = matlabCollectionJob({ ...f, ...p });
  await p.entered; producer.cancel('fixture cancellation'); p.release();
  const outcome = await producer.done;
  assert.equal(outcome.status, 'killed');
  assert.equal(outcome.cleanupVerified, false);
  assert.deepEqual(f.phases, ['prepare']);
});

test('missing optional overlay and changed receipt are rejected before preparation', async t => {
  const f = await fixture(t), p = protocol(f);
  assert.throws(() => assertMatlabBatchAvailable(p.ctx, {}, f.agent), /optional.*overlay/);
  const changed = { ...f.confirmation, rawRequest: f.confirmation.rawRequest + ' ' };
  const outcome = await matlabCollectionJob({ ...f, ...p, confirmation: changed }).done;
  assert.equal(outcome.status, 'failed');
  assert.deepEqual(f.phases, []);
});

test('routing reads only the selected hash-bound plant descriptor', async t => {
  const f = await fixture(t), directory = resolve(f.workspace, 'artifacts/fixture');
  await mkdir(directory, { recursive: true });
  const descriptor = { adapter_spec: { backend: 'matlab_function' } };
  await save(resolve(directory, 'plant.json'), descriptor);
  const meta = { id: f.request.inputs.model, kind: 'plant_model', files: { 'plant.json': sha(await readFile(resolve(directory, 'plant.json'))) } };
  await save(resolve(directory, 'artifact.json'), meta);
  await save(resolve(f.workspace, 'manifest.json'), { artifacts: { [meta.id]: { kind: meta.kind, status: 'verified', path: 'artifacts/fixture', metadata_file_sha256: sha(await readFile(resolve(directory, 'artifact.json'))) } } });
  assert.equal(await requiresMatlabBatch(f.request), true);
  assert.equal(await requiresMatlabBatch({ ...f.request, operation: 'inspect_artifact' }), false);
  await save(resolve(directory, 'plant.json'), { adapter_spec: { backend: 'python_callable' } });
  await assert.rejects(requiresMatlabBatch(f.request), /descriptor changed/);
});

test('expert batches use the fixed helper and preserve the same confirmed request through both workers',async t=>{
  const f=await fixture(t);f.request.operation='collect_external_expert';f.confirmation.rawRequest=JSON.stringify(f.request);
  const p=protocol(f);assert.equal((await matlabCollectionJob({...f,...p}).done).status,'completed');
  assert.deepEqual(f.phases,['prepare','finalize']);
});

test('an expert preparation cannot dispatch the plant-only or an unknown batch protocol',async t=>{
  const f=await fixture(t);f.request.operation='collect_external_expert';f.confirmation.rawRequest=JSON.stringify(f.request);
  const p=protocol(f,{wrongKind:true}),result=await matlabCollectionJob({...f,...p}).done;
  assert.equal(result.status,'failed');assert.equal(result.cleanupVerified,true);assert.match(result.detail,/expert batch protocol/);
  assert.deepEqual(f.phases,['prepare']);
});

test('external routing verifies both descriptors, backend pairing and the physical plant kind',async t=>{
  const f=await fixture(t),artifacts={};
  async function publish(id,kind,descriptorName,descriptor){
    const root=resolve(f.workspace,'artifacts',id);await mkdir(root,{recursive:true});const files={};
    if(descriptorName){await save(resolve(root,descriptorName),descriptor);files[descriptorName]=sha(await readFile(resolve(root,descriptorName)));}
    await save(resolve(root,'artifact.json'),{id,kind,files});
    artifacts[id]={kind,status:'verified',path:`artifacts/${id}`,metadata_file_sha256:sha(await readFile(resolve(root,'artifact.json')))};
    await save(resolve(f.workspace,'manifest.json'),{artifacts});return root;
  }
  const expert=await publish('expert','external_expert','expert.json',{adapter_spec:{backend:'python_callable'}});
  await publish('plant','learned_model');const request={operation:'collect_external_expert',workspace:f.workspace,inputs:{expert:'expert',plant:'plant'}};
  assert.equal(await requiresMatlabBatch(request),false);
  await publish('expert','external_expert','expert.json',{adapter_spec:{backend:'matlab_function'}});
  await assert.rejects(requiresMatlabBatch(request),/colocated/);
  await publish('plant','plant_model','plant.json',{adapter_spec:{backend:'simulink'}});
  assert.equal(await requiresMatlabBatch(request),true);
  await save(resolve(expert,'expert.json'),{adapter_spec:{backend:'python_callable'}});
  await assert.rejects(requiresMatlabBatch(request),/descriptor changed/);
  await publish('expert','external_expert','expert.json',{adapter_spec:{backend:'python_callable'}});
  await publish('plant','material');await assert.rejects(requiresMatlabBatch(request),/physical plant/);
});

test('policy evaluation dispatches only the fixed comparison helper with original receipt',async t=>{
  const f=await fixture(t);f.request.operation='evaluate_external_imitation';f.confirmation.rawRequest=JSON.stringify(f.request);
  const p=protocol(f);assert.equal((await matlabCollectionJob({...f,...p}).done).status,'completed');assert.deepEqual(f.phases,['prepare','finalize']);
});

test('policy evaluation rejects the wrong batch protocol before native execution',async t=>{
  const f=await fixture(t);f.request.operation='evaluate_external_imitation';f.confirmation.rawRequest=JSON.stringify(f.request);
  const p=protocol(f,{wrongKind:true});const outcome=await matlabCollectionJob({...f,...p}).done;
  assert.equal(outcome.status,'failed');assert.equal(outcome.cleanupVerified,true);assert.match(outcome.detail,/comparison batch protocol/);assert.deepEqual(f.phases,['prepare']);
});

test('policy routing follows hash-bound original expert/Plant ancestors instead of active artifacts',async t=>{
  const f=await fixture(t),artifacts={};
  async function publish(id,kind,file,descriptor,parents=[],source={}){
    const root=resolve(f.workspace,'artifacts',id);await mkdir(root,{recursive:true});await save(resolve(root,file),descriptor);
    await save(resolve(root,'artifact.json'),{id,kind,parents:parents.map(id=>({id})),source,files:{[file]:sha(await readFile(resolve(root,file)))}});
    artifacts[id]={kind,status:'verified',path:`artifacts/${id}`,metadata_file_sha256:sha(await readFile(resolve(root,'artifact.json')))};
    await save(resolve(f.workspace,'manifest.json'),{artifacts,active:{plant_model:'unrelated-active'}});return root;
  }
  await publish('expert','external_expert','expert.json',{adapter_spec:{backend:'matlab_function'}});
  await publish('plant','plant_model','plant.json',{adapter_spec:{backend:'simulink'}});
  await publish('dataset','external_expert_dataset','dataset.json',{},['expert','plant'],{expert:'expert',plant:'plant'});
  const policy=await publish('policy','external_imitation_policy','policy.json',{external_expert_dataset:'dataset'},['dataset']);
  const request={operation:'evaluate_external_imitation',workspace:f.workspace,inputs:{policy:'policy'}};
  assert.equal(await requiresMatlabBatch(request),true);
  await publish('expert','external_expert','expert.json',{adapter_spec:{backend:'python_callable'}});
  await assert.rejects(requiresMatlabBatch(request),/colocated/);
  await publish('plant','plant_model','plant.json',{adapter_spec:{backend:'python_callable'}});assert.equal(await requiresMatlabBatch(request),false);
  await publish('dataset','external_expert_dataset','dataset.json',{},[],{});assert.equal(await requiresMatlabBatch(request),false,'Recorded data has no executable expert to route');
  await publish('dataset','external_expert_dataset','dataset.json',{},[],{expert:'expert',plant:'plant'});await assert.rejects(requiresMatlabBatch(request),/ancestors/);
  await save(resolve(policy,'policy.json'),{external_expert_dataset:'other'});await assert.rejects(requiresMatlabBatch(request),/descriptor changed/);
});
