/** Actual Python/Node workflow admission parity; no sampling, training or model API. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { confirmationRequirements, resolveWorkflowCase, workflowStageInputs, synthesisScopes } from '../src/case-requirements.mjs';

const project = resolve(import.meta.dirname, '../..');
const python = resolve(project, '.venv/Scripts/python.exe');
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/case-requirement-tests'));
const env = { ...process.env, PYTHONPATH: resolve(project, 'src'), PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' };
await mkdir(output, { recursive: true });
function py(source, args = [], input = '') {
  const result = spawnSync(python, ['-B', '-c', source, ...args], { cwd: project, env, input, encoding: 'utf8', windowsHide: true });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout);
}
const make = String.raw`
import copy,json,sys
from pathlib import Path
from control_harness.artifacts import Workspace,atomic_json
from control_harness.examples import example
from control_harness.modeling import register_model
from control_harness.case_requirements import confirmation_requirements,resolve_workflow_case
from control_harness.workflow import STAGES
root=Path(sys.argv[1]);ws=Workspace(root/'workspace');spec=example('linear')['nominal']
model=root/'nominal.json';atomic_json(model,spec);data=root/'input.npz';data.write_bytes(b'explicit source hash fixture, no execution')
nominal=register_model(ws,spec,{'kind':'user_nominal'})
source=Workspace(root/'source');external=register_model(source,spec,{'kind':'user_nominal'});external_path=source.resolve(external)[0]
register={'id':'model','operation':'register_model','inputs':{'model_path':str(model)},'config':{'source':{'kind':'user_nominal'}}}
imported={'id':'data','operation':'import_transitions','inputs':{'nominal_model':nominal,'data_path':str(data)}}
train={'id':'network','operation':'train_residual','inputs':{'nominal_model':nominal,'dataset':{'stage':'data'}},'config':{'epochs':7,'learning_rate':.001}}
sample={'id':'samples','operation':'simulate_transitions','inputs':{'nominal_model':nominal,'reference_model':nominal},'config':{'steps':5}}
external_step={'id':'external','operation':'import_artifact','inputs':{'artifact_path':str(external_path)}}
requests=[]
for steps in [[register],[imported],[external_step],[imported,train],[sample],[sample,imported,train]]:
 requests.append({'operation':'run_workflow','workspace':str(ws.path),'inputs':{},'config':{'schema_version':1,'name':'局部阶段','steps':steps}})
definition={'schema_version':1,'name':'明确文件流程','steps':copy.deepcopy([register,imported,train])}
definition['steps'][0]['inputs']['model_path']='nominal.json'
path=root/'workflow.json';atomic_json(path,definition)
requests.append({'operation':'run_workflow','workspace':str(ws.path),'inputs':{'workflow_path':str(path)},'config':{}})
def rows(req):
 return [{'scope':r['scope'],'expectedPayload':r['expected_payload'],'bindings':r['bindings']} for r in confirmation_requirements(ws,req)]
print(json.dumps({'requests':requests,'requirements':[rows(r) for r in requests],'snapshots':[resolve_workflow_case(ws,r)['bindings'] for r in requests],
 'stage_inputs':{op:{key:('artifact' if isinstance(kind,set) else kind) for key,kind in item[0].items()} for op,item in STAGES.items()},
 'model':str(model),'data':str(data),'workflow':str(path),'external':str(external_path)},ensure_ascii=False))
`;
const query = String.raw`
import json,sys
from control_harness.artifacts import Workspace
from control_harness.case_requirements import confirmation_requirements
r=json.load(sys.stdin)
try:
 rows=confirmation_requirements(Workspace(r['workspace']),r)
 print(json.dumps({'ok':True,'rows':[{'scope':x['scope'],'expectedPayload':x['expected_payload'],'bindings':x['bindings']} for x in rows]},ensure_ascii=False))
except Exception as error: print(json.dumps({'ok':False,'kind':type(error).__name__}))
`;

test('real STAGES, minimum scopes and source bindings match Python including file and artifact paths', async () => {
  const directory = await mkdtemp(join(output, 'workflow-parity-'));
  try {
    const fixture = py(make, [directory]);
    assert.deepEqual(workflowStageInputs, fixture.stage_inputs, 'New stage support must update the exact native mirror');
    const expectedScopes = [[], [], [], ['signals', 'training'], ['signals', 'sampling'], ['signals', 'sampling', 'training'], ['signals', 'training']];
    for (let index = 0; index < fixture.requests.length; index += 1) {
      const request = fixture.requests[index];
      assert.deepEqual(await confirmationRequirements(request), fixture.requirements[index]);
      assert.deepEqual((await resolveWorkflowCase(request)).bindings, fixture.snapshots[index]);
      assert.deepEqual(fixture.requirements[index].map(row => row.scope), expectedScopes[index]);
    }
    const request = fixture.requests.at(-1), before = await confirmationRequirements(request);
    await writeFile(fixture.data, 'changed bytes at the same exact path');
    const after = await confirmationRequirements(request), fromPython = py(query, [], JSON.stringify(request));
    assert.equal(fromPython.ok, true);
    assert.deepEqual(after, fromPython.rows);
    assert.notDeepEqual(after[0].bindings, before[0].bindings, 'Path equality cannot authorize different source bytes');
    const document = JSON.parse(await readFile(fixture.workflow, 'utf8'));
    document.steps.at(-1).config.epochs = 9;
    await writeFile(fixture.workflow, JSON.stringify(document));
    const changed = await confirmationRequirements(request);
    assert.notEqual(changed[0].bindings.workflow_file.sha256, after[0].bindings.workflow_file.sha256);
    assert.equal(changed[0].expectedPayload.workflow_definition.steps.at(-1).config.epochs, 9);
    assert.deepEqual(changed, py(query, [], JSON.stringify(request)).rows);

    const externalRequest = fixture.requests[2];
    await writeFile(join(fixture.external, 'model.json'), 'tampered artifact bytes');
    await assert.rejects(resolveWorkflowCase(externalRequest), /changed/);
    assert.equal(py(query, [], JSON.stringify(externalRequest)).ok, false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('resume execution fields stay outside ordinary engineering summaries in both runtimes', async () => {
  const directory = await mkdtemp(join(output, 'resume-parity-'));
  try {
    const fixture = py(make, [directory]), workspace = fixture.requests[0].workspace;
    for (const [operation, config] of [['prepare_supervised_dataset', { interface: { inputs: [0], outputs: [1] } }], ['train_residual', { epochs: 10, learning_rate: .001 }]]) {
      const request = { operation, workspace, inputs: {}, config };
      const before = await confirmationRequirements(request);
      Object.assign(config, { plan_refs: { signals: {} }, resume_run: 'new-run', resume_checkpoint: 'not-read.pt', resume_checkpoint_sha256: 'ignored' });
      const after = await confirmationRequirements(request);
      assert.deepEqual(after, before);
      assert.deepEqual(after, py(query, [], JSON.stringify(request)).rows);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('trial/full selector reuses reviewed limits while changing a limit invalidates both runtimes', async () => {
  const directory = await mkdtemp(join(output, 'sampling-selection-parity-'));
  try {
    const fixture = py(make, [directory]), workspace = fixture.requests[0].workspace;
    const model = fixture.requests[1].config.steps[0].inputs.nominal_model;
    const request = { operation: 'sample_model', workspace, inputs: { model },
      config: { sampling: { steps: 20, trajectories: 6 }, trial: { trajectories: 1, steps: 3 }, sampling_run: 'trial' } };
    const trial = await confirmationRequirements(request);
    assert.deepEqual(trial, py(query, [], JSON.stringify(request)).rows);
    request.config.sampling_run = 'full';
    assert.deepEqual(await confirmationRequirements(request), trial);
    assert.deepEqual(await confirmationRequirements(request), py(query, [], JSON.stringify(request)).rows);
    request.config.trial.steps += 1;
    assert.notDeepEqual(await confirmationRequirements(request), trial);
    assert.deepEqual(await confirmationRequirements(request), py(query, [], JSON.stringify(request)).rows);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('MPC draft versus numerical design scopes are identical across Host and worker', () => {
  const params = { mass:1, damping:.8, stiffness:2, sample_time:.2,
    state_bounds:[[-1,1],[-1.5,1.5]], input_bounds:[[-4,4]], output_bounds:[[-1,1]] };
  const cases = [
    {inputs:{description:'二阶弹簧振子 MPC'},config:{}},
    {inputs:{description:'spring damper'},config:{parameters:{mass:1}}},
    {inputs:{description:'spring damper'},config:{parameters:{...params,mass:null}}},
    {inputs:{description:'generic nonlinear design'},config:{parameters:params}},
    {inputs:{description:'spring damper'},config:{parameters:params}},
    {inputs:{},config:{template:'spring_damper',parameters:params}},
    {inputs:{model:'exact-model-id'},config:{}},
    {inputs:{dataset:'sampled-id'},config:{}},
    {inputs:{dataset:'sampled-id'},config:{prediction:{method:'identify_affine'}}},
    {inputs:{model:'exact-model-id'},config:{prediction:{method:'linearize'}}},
    {inputs:{description:'arbitrary description'},config:{prediction:{method:'equations',model_spec:{representation:'discrete_polynomial'}}}},
  ].map(row=>({operation:'synthesize_mpc',...row}));
  const expected=[[],[],[],[],['signals'],['signals'],['signals'],['signals','training'],['signals','training'],['signals'],['signals']];
  assert.deepEqual(cases.map(synthesisScopes),expected);
  const result=py('import json,sys\nfrom control_harness.case_requirements import synthesis_scopes\nprint(json.dumps([synthesis_scopes(r) for r in json.load(sys.stdin)]))',[],JSON.stringify(cases));
  assert.deepEqual(result,expected);
  for (const row of cases) row.config.confirmed=true;
  assert.deepEqual(cases.map(synthesisScopes),expected,'A model-written flag must never change authority scopes');
});

test('preference tuning binds the evaluation recipe and scales in both runtimes', async () => {
  const directory = await mkdtemp(join(output, 'tuning-preference-parity-'));
  try {
    const fixture = py(make, [directory]), workspace = fixture.requests[0].workspace;
    const model = fixture.requests[1].config.steps[0].inputs.nominal_model;
    const request = { operation:'tune_controller', workspace, inputs:{model,evaluation_model:model},
      config:{ parameters:{horizon:4,q:1,r:.015,s:.01}, evaluation:{recipe:'tracking',scales:{rmse:1,energy:100,tv:10,overshoot:.1}} } };
    const original = await confirmationRequirements(request);
    assert.deepEqual(original.map(row=>row.scope),['signals','training']);
    assert.deepEqual(original,py(query,[],JSON.stringify(request)).rows);
    for (const evaluation of [
      {recipe:'energy',scales:{rmse:1,energy:100,tv:10,overshoot:.1}},
      {recipe:'tracking',scales:{rmse:1,energy:200,tv:10,overshoot:.1}},
      {recipe:'tracking',scales:{rmse:1,energy:100,tv:10,overshoot:.1},weights:{rmse:10,energy:1,tv:0,overshoot:0}},
    ]) {
      const changed = {...request,config:{...request.config,evaluation}};
      const requirements = await confirmationRequirements(changed);
      assert.deepEqual(requirements,py(query,[],JSON.stringify(changed)).rows);
      assert.deepEqual(requirements[0],original[0],'Physical signals unchanged');
      assert.notDeepEqual(requirements[1],original[1],'Changed preference requires the revised numerical summary');
    }
  } finally { await rm(directory,{recursive:true,force:true}); }
});
