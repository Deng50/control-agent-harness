/** Real immutable PINN preparation, then pure Host/worker authorization parity. */
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdir,mkdtemp,rm} from 'node:fs/promises';
import {resolve} from 'node:path';
import test from 'node:test';
import {confirmationRequirements} from '../src/case-requirements.mjs';
const project=resolve(import.meta.dirname,'../..'),output=resolve(project,'test-output/pinn-case-tests');await mkdir(output,{recursive:true});
function py(code,args=[],input=''){
 const result=spawnSync(resolve(project,'.venv/Scripts/python.exe'),['-B','-c',code,...args],{cwd:project,input,encoding:'utf8',windowsHide:true,
  env:{...process.env,PYTHONPATH:resolve(project,'src'),PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'}});
 assert.equal(result.status,0,result.stderr||result.error?.message);return JSON.parse(result.stdout);
}
const make=String.raw`import json,sys
from pathlib import Path
from control_harness.artifacts import Workspace
from control_harness.material_registration import register_material
from control_harness.pinn_problem import prepare_pinn_problem
from control_harness.runtime import OperationContext
r=json.loads(Path('test-output/v2-development/20261117T002006/S11/native-pinn-registration-v2.json').read_text());packet=Path(r['packet']);w=Workspace(Path(sys.argv[1])/'workspace')
m=register_material(w,str(packet/'problem-input.json'),{'dependency_root':str(packet),'dependencies':['reference.csv','collocation.csv']},OperationContext())
p=prepare_pinn_problem(w,m,r['problem_config'])
print(json.dumps({'prepare':{'operation':'prepare_pinn_problem','workspace':str(w.path),'inputs':{'material':m},'config':r['problem_config']},
 'train':{'operation':'train_pinn','workspace':str(w.path),'inputs':{'problem':p},'config':{'network':{'hidden_dims':[5],'epochs':3},'loss_weights':{'data':1,'physics':1,'initial':10}}},'material':m}))`;
const query=String.raw`import json,sys
from control_harness.artifacts import Workspace
from control_harness.case_requirements import confirmation_requirements
r=json.load(sys.stdin)
try:print(json.dumps({'ok':True,'rows':[{'scope':x['scope'],'expectedPayload':x['expected_payload'],'bindings':x['bindings']} for x in confirmation_requirements(Workspace(r['workspace']),r)]}))
except Exception as e:print(json.dumps({'ok':False,'error':str(e)}))`;
async function parity(request){const expected=py(query,[],JSON.stringify(request));assert.equal(expected.ok,true,expected.error);const actual=await confirmationRequirements(request);assert.deepEqual(actual,expected.rows);return actual;}

test('PINN preparation reviews complete SI/condition/mapping declarations and training binds the resolved problem',async t=>{
 const directory=await mkdtemp(resolve(output,'case-'));t.after(()=>rm(directory,{recursive:true,force:true}));const fixture=py(make,[directory]);
 const prepared=await parity(fixture.prepare);assert.deepEqual(prepared.map(x=>x.scope),['signals']);assert.deepEqual(prepared[0].expectedPayload,fixture.prepare.config);
 const trained=await parity(fixture.train);assert.deepEqual(trained.map(x=>x.scope),['signals','training']);const signals=trained[0].expectedPayload;
 assert.equal(signals.problem_kind,'conditional_ode');assert.equal(signals.equation.template,'thermal_first_order');assert.equal(signals.coordinates.length,3);assert.equal(signals.state.unit,'K');assert.equal(signals.time_derivative.unit,'K/s');assert.equal(signals.capability.one_step_transition,false);
 const changed=structuredClone(fixture.prepare);changed.config.normalization.time_scale_s*=2;const revised=await parity(changed);assert.notDeepEqual(revised[0].expectedPayload,prepared[0].expectedPayload);assert.deepEqual(revised[0].bindings,prepared[0].bindings);
 for(const request of [fixture.prepare,fixture.train]){const before=await parity(request);request.config.plan_refs={signals:{plan_id:'selector-only'}};request.config.resume_checkpoint='not-read.pt';assert.deepEqual(await parity(request),before);}
});

test('PINN training changes invalidate training summary without silently changing physics signals; wrong problem kind is rejected',async t=>{
 const directory=await mkdtemp(resolve(output,'case-'));t.after(()=>rm(directory,{recursive:true,force:true}));const fixture=py(make,[directory]),initial=await parity(fixture.train);
 for(const patch of [{loss_weights:{data:1,physics:2,initial:10}},{network:{hidden_dims:[7],epochs:3}},{conditional_parameterization:'affine_initial_power'},{ablation:true,loss_weights:{data:1,physics:0,initial:10}}]){
  const changed={...fixture.train,config:{...fixture.train.config,...patch}};const requirements=await parity(changed);assert.deepEqual(requirements[0],initial[0]);assert.notDeepEqual(requirements[1].expectedPayload,initial[1].expectedPayload);
 }
 const invalid={...fixture.train,inputs:{problem:fixture.material}};await assert.rejects(confirmationRequirements(invalid),/conditional ODE problem/);const result=py(query,[],JSON.stringify(invalid));assert.equal(result.ok,false);assert.match(result.error,/conditional ODE problem/);
});
