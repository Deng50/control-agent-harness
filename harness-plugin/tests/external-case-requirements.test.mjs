/** Admission-only protocol parity over real immutable registrations; no execution or sampling. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { confirmationRequirements } from '../src/case-requirements.mjs';
import { createHost } from './history-host.mjs';

const project = resolve(import.meta.dirname, '../..');
const python = resolve(project, '.venv/Scripts/python.exe');
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? join(project, 'test-output/external-confirmation-tests'));
const env = { ...process.env, PYTHONPATH: resolve(project, 'src'), PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' };
await mkdir(output, { recursive: true });
function py(source, args = [], input = '') {
  const result = spawnSync(python, ['-B', '-c', source, ...args], { cwd: project, env, input, encoding: 'utf8', windowsHide: true });
  assert.equal(result.status, 0, result.stderr || result.error?.message);
  return JSON.parse(result.stdout);
}

const make = String.raw`
import copy,json,sys
from pathlib import Path
from control_harness.artifacts import Workspace
from control_harness.contracts import DataInterface
from control_harness.material_registration import register_material
from control_harness.external_expert import register_expert
from control_harness.plants import register_plant
from control_harness.runtime import OperationContext
root=Path(sys.argv[1]);ws=Workspace(root/'workspace');context=OperationContext()
v=lambda name,unit,bound:{'name':name,'unit':unit,'lower':-bound,'upper':bound}
physical={'schema_version':1,'name':'synthetic admission-only physical fixture','sample_time':.1,'time_semantics':'discrete_state_transition',
 'states':[v('position','m',2.)],'controls':[v('force','N',1.)],'outputs':[v('sensor','m',2.)],'disturbances':[v('wind','N',.2)]}
# These declarations can be registered and inspected, but would fail if admission accidentally ran their code.
ep=root/'expert.py';ep.write_text("raise RuntimeError('admission must never execute expert source')\ndef expert(obs,r,d,previous,memory,parameters):\n return (previous,memory,'ok')\n",encoding='utf-8')
pm=root/'plant.py';pm.write_text("raise RuntimeError('admission must never execute Plant source')\ndef transition(x,u,d,dt,parameters):\n return x\n",encoding='utf-8')
expert_material=register_material(ws,str(ep),{},context);plant_material=register_material(ws,str(pm),{},context)
spec={'schema_version':1,'backend':'python_callable','source_file':'source/expert.py','entrypoint':'expert','interface':physical,
 'parameters':{'gain':.4},'time_unit':'s','control_period':.2,
 'observations':[{'name':'measured_position','source':'output','index':0,'plant_name':'sensor','unit':'m','availability':'measured','delay_steps':0}],
 'reference_available':True,'disturbance_available':False,'previous_action_available':True,
 'initial_memory':{'integral':0.,'mode':'start'},'action_mapping':[{'action_index':0,'plant_control_index':0,'plant_control_name':'force','unit':'N'}],
 'failure_policy':'stop','input_rate_bounds':[[-.2,.2]]}
experts={};specs={}
for name in ['base','parameters','observation','memory','period','rate']:
 s=copy.deepcopy(spec)
 if name=='parameters':s['parameters']['gain']=.7
 if name=='observation':s['observations'][0]['availability']='estimated'
 if name=='memory':s['initial_memory']['integral']=.125
 if name=='period':s['control_period']=.4
 if name=='rate':s['input_rate_bounds']=[[-.1,.15]]
 experts[name]=register_expert(ws,expert_material,{'adapter':s});specs[name]=s
plants={}
for name,gain in [('base',.2),('other',.3)]:
 plants[name]=register_plant(ws,plant_material,{'adapter':{'schema_version':1,'backend':'python_callable','source_file':'source/plant.py',
  'entrypoint':'transition','interface':physical,'parameters':{'gain':gain},'source_semantics':'transition','C':[[1.]]}},context)
csv=root/'recording.csv';csv.write_text('observation,reference,previous,action,time,trip\n'+''.join(f'.1,.2,0,.05,{step*.2},synthetic-trip-{trip}\n' for trip in range(6) for step in range(4)),encoding='utf-8')
recording=register_material(ws,str(csv),{},context)
signals=[]
for index,(name,unit,role,expert_role) in enumerate([('observation','m','auxiliary','observation'),('reference','m','auxiliary','reference'),('previous','N','auxiliary','previous_action'),('action','N','control','expert_action')]):
 signals.append({'name':name,'unit':unit,'role':role,'expert_role':expert_role,'observable':True,'delay_steps':0,
  'column':{'index':index,'header':name,'worksheet':None}})
data=DataInterface.parse({'schema_version':1,'interface_kind':'tabular_supervised','task':'static_regression','name':'synthetic imported external policy channels',
 'sample_time':.2,'signals':signals,'features':[{'signal':name,'lag':0,'window':1} for name in ['observation','reference','previous']],
 'targets':[{'signal':'action','offset':0}],'time_column':{'index':4,'header':'time','worksheet':None},'trip_column':{'index':5,'header':'trip','worksheet':None},
 'source_mapping':{'signals':{'previous':{'time_semantics':'already_previous_applied_action','action_target':'action'}}}}).to_dict()
config={'seed':43,'scaler':'standard','scenarios':[{'name':'synthetic-'+str(k),'initial_state':[-.3+.1*k],
 'reference':[.2+.01*k],'disturbance':[.01*k],'initial_previous_action':[0.],'steps':3,
 'split':'train' if k<4 else 'validation' if k==4 else 'test'} for k in range(6)]}
collect={'operation':'collect_external_expert','workspace':str(ws.path),'inputs':{'expert':experts['base'],'plant':plants['base']},'config':config}
imported={'operation':'import_expert_data','workspace':str(ws.path),'inputs':{'material':recording},'config':{'interface':data,'scaler':'standard',
 'source_mapping':copy.deepcopy(data['source_mapping']),
 'split':{'mode':'trip','train':.6,'validation':.2,'test':.2},
 'action_constraints':{'bounds':[[-1.,1.]],'rate_bounds':[[-.2,.2]],'rate_semantics':'per_control_step'},
 'source':{'kind':'recorded_external_expert','description':'Explicit synthetic admission fixture; not measured hardware data',
  'previous_action_semantics':'actual_applied_action_before_current_decision'}}}
registration={'operation':'register_expert','workspace':str(ws.path),'inputs':{'material':expert_material},'config':{'adapter':spec}}
print(json.dumps({'collect':collect,'imported':imported,'registration':registration,'experts':experts,'specs':specs,'plants':plants,
 'expert_path':str(ws.resolve(experts['base'])[0]/'expert.json'),'material_source':str(ws.resolve(expert_material)[0]/'source/expert.py'),
 'kinds':[row['kind'] for row in ws.manifest['artifacts'].values()]},ensure_ascii=False))
`;

const query = String.raw`
import json,sys
from control_harness.artifacts import Workspace
from control_harness.case_requirements import confirmation_requirements
request=json.load(sys.stdin)
try:
 rows=confirmation_requirements(Workspace(request['workspace']),request)
 print(json.dumps({'ok':True,'rows':[{'scope':row['scope'],'expectedPayload':row['expected_payload'],'bindings':row['bindings']} for row in rows]},ensure_ascii=False))
except Exception as error:
 print(json.dumps({'ok':False,'kind':type(error).__name__,'message':str(error)},ensure_ascii=False))
`;
const copy = value => JSON.parse(JSON.stringify(value));
async function parity(request) {
  const expected = py(query, [], JSON.stringify(request));
  assert.equal(expected.ok, true, expected.message);
  const actual = await confirmationRequirements(request);
  assert.deepEqual(actual, expected.rows);
  return actual;
}
async function fixture(t) {
  const directory = await mkdtemp(join(output, 'external-admission-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return py(make, [directory]);
}

test('static registration has no authority scopes; real bound expert collection captures explicit semantics in both runtimes', async t => {
  const f = await fixture(t);
  assert.deepEqual(await parity(f.registration), []);
  const rows = await parity(f.collect);
  assert.deepEqual(rows.map(row => row.scope), ['signals', 'sampling']);
  assert.deepEqual(rows[0].expectedPayload.plant_interface, f.specs.base.interface);
  const spec = rows[0].expectedPayload.expert_adapter;
  assert.equal(spec.control_period, .2);
  assert.equal(spec.interface.sample_time, .1, 'Plant dt must not become the control period');
  assert.equal(spec.time_unit, 's');
  assert.deepEqual(spec.observations, f.specs.base.observations);
  assert.deepEqual(spec.initial_memory, { integral: 0, mode: 'start' });
  assert.deepEqual(spec.input_rate_bounds, [[-.2, .2]]);
  assert.deepEqual(spec.action_mapping, f.specs.base.action_mapping);
  assert.equal(spec.disturbance_available, false);
  assert.equal(spec.previous_action_available, true);
  assert.equal(spec.failure_policy, 'stop');
  assert.equal(Object.hasOwn(spec, 'parameters'), false, 'Large parameters are sealed by immutable binding, not copied into signal summary');
  assert.deepEqual(rows[1].expectedPayload, f.collect.config);
  assert.deepEqual(Object.keys(rows[0].bindings).sort(), ['expert', 'plant']);
  assert.equal(rows[0].bindings.expert.id, f.experts.base);
  assert.equal(rows[0].bindings.plant.id, f.plants.base);
  assert.match(rows[0].bindings.expert.metadata_hash, /^[a-f0-9]{64}$/);
  assert.ok(f.kinds.every(kind => ['material', 'external_expert', 'plant_model'].includes(kind)), 'Admission creates no demonstration, controller, training or numerical result');
});

test('collection config and immutable expert/plant changes invalidate the corresponding payload or dependency in both runtimes', async t => {
  const f = await fixture(t), base = await parity(f.collect);
  const changedConfig = copy(f.collect);
  changedConfig.config.scenarios[0].steps += 1;
  const sampling = await parity(changedConfig);
  assert.deepEqual(sampling[0], base[0], 'Unchanged interface remains the same signals review');
  assert.notDeepEqual(sampling[1], base[1], 'Changed collection duration requires revised sampling review');
  for (const name of ['parameters', 'observation', 'memory', 'period', 'rate']) {
    const request = copy(f.collect); request.inputs.expert = f.experts[name];
    const changed = await parity(request);
    assert.notDeepEqual(changed[0].bindings, base[0].bindings, `Changed ${name} expert must change signal dependencies`);
    assert.notDeepEqual(changed[1].bindings, base[1].bindings, `Changed ${name} expert must change sampling dependencies`);
    assert.deepEqual(changed[1].expectedPayload, base[1].expectedPayload, 'Same collection config; dependency rather than accidental config difference binds expert');
    if (name === 'parameters') assert.deepEqual(changed[0].expectedPayload, base[0].expectedPayload);
    else assert.notDeepEqual(changed[0].expectedPayload, base[0].expectedPayload, `Actual ${name} semantics require revised signal summary`);
  }
  const changedPlant = copy(f.collect); changedPlant.inputs.plant = f.plants.other;
  const plantRows = await parity(changedPlant);
  assert.deepEqual(plantRows[0].expectedPayload, base[0].expectedPayload);
  assert.notDeepEqual(plantRows[0].bindings.plant, base[0].bindings.plant, 'Same dimensions cannot authorize a different physical implementation');
});

test('recorded import reviews DataInterface with causal actions and never invents physical nx', async t => {
  const f = await fixture(t), rows = await parity(f.imported);
  assert.deepEqual(rows.map(row => row.scope), ['signals']);
  assert.deepEqual(rows[0].expectedPayload,f.imported.config,'One signals review binds the entire declared external import contract');
  const data = rows[0].expectedPayload.interface;
  assert.equal(data.interface_kind, 'tabular_supervised');
  assert.equal(data.input_dim, 3);
  assert.equal(data.output_dim, 1);
  for (const key of ['nx', 'nu', 'ny', 'states', 'controls']) assert.equal(Object.hasOwn(data, key), false);
  assert.deepEqual(data.features.map(item => item.signal), ['observation', 'reference', 'previous']);
  assert.equal(data.targets[0].signal, 'action');
  assert.equal(data.targets[0].offset, 0);
  assert.equal(data.source_mapping.signals.previous.time_semantics, 'already_previous_applied_action');
  assert.deepEqual(Object.keys(rows[0].bindings), ['material']);
  const changed = copy(f.imported); changed.config.interface.signals[0].unit = 'cm';
  const changedRows = await parity(changed);
  assert.notDeepEqual(changedRows[0].expectedPayload, rows[0].expectedPayload, 'Unit changes require new signal review');
  assert.deepEqual(changedRows[0].bindings, rows[0].bindings, 'Unit declaration change does not pretend source bytes changed');
});

function importChanges(request) {
  const cases=[];
  const add=(name,change)=>{const value=copy(request);change(value.config);cases.push({name,request:value});};
  add('action amplitude bounds',c=>c.action_constraints.bounds=[[-2,2]]);
  add('action increment bounds',c=>c.action_constraints.rate_bounds=[[-.3,.3]]);
  add('previous action mapping',c=>c.source_mapping.signals.previous.action_target='different-action');
  add('source provenance',c=>c.source.description+=' revised claim');
  add('source kind',c=>c.source.kind='declared_simulation');
  add('split',c=>c.split={mode:'trip',train:.5,validation:.25,test:.25});
  add('scaler',c=>c.scaler='minmax');
  add('physical interface',c=>c.physical_interface={schema_version:1,name:'separate explicit context'});
  add('physical mapping',c=>c.physical_mapping={observations:['changed explicit binding']});
  add('worksheet',c=>c.worksheet='other');
  add('header row',c=>c.header_row=2);
  add('delimiter',c=>c.delimiter=';');
  add('encoding',c=>c.encoding='utf-16');
  return cases;
}

test('all external import semantics and preprocessing change the same single scope, while refs and ordinary S03 prep retain their rules',async t=>{
  const f=await fixture(t),base=await parity(f.imported);
  for(const row of importChanges(f.imported)){
    const actual=await parity(row.request);assert.deepEqual(actual.map(x=>x.scope),['signals']);
    assert.notDeepEqual(actual[0].expectedPayload,base[0].expectedPayload,row.name+' must require revised review');
    assert.deepEqual(actual[0].bindings,base[0].bindings,'Same material bytes remain the same dependency');
  }
  const resumed=copy(f.imported);Object.assign(resumed.config,{plan_refs:{signals:{plan_id:'not-consulted-by-resolver'}},resume_checkpoint:'unused.pt',resume_checkpoint_sha256:'f'.repeat(64),resume_run:'unused'});
  assert.deepEqual(await parity(resumed),base,'Confirmation/resume locations are excluded from the engineering summary');
  const ordinary={...copy(f.imported),operation:'prepare_supervised_dataset'},first=await parity(ordinary);
  const changed=copy(ordinary);changed.config.scaler='none';changed.config.source.description+=' ordinary prep remains unchanged';
  assert.deepEqual(first,await parity(changed));assert.deepEqual(first[0].expectedPayload,ordinary.config.interface);
});

test('real native confirmation of an external import rejects changed contracts before any job or history admission',async t=>{
  const f=await fixture(t),workspace=f.imported.workspace,requirements=await parity(f.imported);
  await mkdir(join(workspace,'.case-plans'),{recursive:true});
  await writeFile(join(workspace,'.case-plans/test-fixture.json'),JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace}));
  const plan=py("import json,sys\nfrom control_harness.case_plans import propose_plan\nx=json.load(sys.stdin);print(json.dumps(propose_plan(x['workspace'],'signals',x['payload'],x['bindings'])))",[],JSON.stringify({workspace,payload:requirements[0].expectedPayload,bindings:requirements[0].bindings}));
  const host=await createHost(workspace,'external-import-scope-native',{confirmFixturePlans:false});t.after(()=>host.close());
  const ref={plan_id:plan.plan_id,version_hash:plan.version_hash};
  const confirmed=await host.command('control-confirm',{workspace,...ref});assert.equal(confirmed.kind,'success',confirmed.text);assert.equal(JSON.parse(confirmed.text).confirmation_source,'test_fixture');
  const snapshot=async()=>{const history=await host.command('control-history',{workspace});assert.equal(history.kind,'success',history.text);return{jobs:host.ctx.jobs.list(host.agent.id).map(x=>x.id),attempts:JSON.parse(history.text).attempts.map(x=>x.attemptId)};};
  const before=await snapshot();assert.deepEqual(before,{jobs:[],attempts:[]});
  for(const row of importChanges(f.imported)){
    const response=await host.command('control',{agent:'lightweight',...row.request,config:{...row.request.config,plan_refs:{signals:ref}}});
    assert.equal(response.kind,'error',row.name+' changed contract was admitted');assert.match(response.text,/reviewed summary|version|differ/i);
    assert.deepEqual(await snapshot(),before,row.name+' created a native job or durable attempt');
  }
});

test('tampered expert declaration and transitive source closure are rejected by both actual readers', async t => {
  const f = await fixture(t);
  await parity(f.collect);
  const original = await readFile(f.expert_path);
  const tampered = JSON.parse(original.toString('utf8'));
  tampered.adapter_spec.observations[0].availability = 'simulation_only';
  await writeFile(f.expert_path, JSON.stringify(tampered));
  await assert.rejects(confirmationRequirements(f.collect), /changed/);
  assert.equal(py(query, [], JSON.stringify(f.collect)).ok, false);
  await writeFile(f.expert_path, original);
  await parity(f.collect);
  await writeFile(f.material_source, 'def expert(obs,r,d,previous,memory,p):\n return (previous,memory,"ok")\n');
  await assert.rejects(confirmationRequirements(f.collect), /changed/);
  assert.equal(py(query, [], JSON.stringify(f.collect)).ok, false, 'Restored expert bytes do not authorize a mutated material parent');
});
