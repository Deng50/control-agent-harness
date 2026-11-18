/** Actual Python source/NN/R01 fixture, Host scopes and real worker receipts.
 * Persistent isolated files are retained under test-output for inspection.
 * Native commands here use Cordis' real command/projection mechanism with an
 * explicit test_fixture marker; no production Host, F05 or MATLAB is run.
 */
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {cp,mkdir,mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {delimiter,join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import test from 'node:test';
import {confirmationRequirements} from '../src/case-requirements.mjs';
import {installCasePlans} from '../src/case-plans.mjs';

const project=resolve(import.meta.dirname,'../..'),python=resolve(project,'.venv/Scripts/python.exe');
const output=resolve(project,'test-output/pgac-case-tests');await mkdir(output,{recursive:true});
const environment={...process.env,PYTHONPATH:[resolve(project,'src'),resolve(project,'tests')].join(delimiter),PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'};
function py(code,args=[],input='',extraEnv={}){
 const result=spawnSync(python,['-B','-c',code,...args],{cwd:project,input,encoding:'utf8',windowsHide:true,env:{...environment,...extraEnv},timeout:180000});
 assert.equal(result.status,0,result.stderr||result.error?.message);return JSON.parse(result.stdout);
}
const make=String.raw`import copy,json,shutil,sys,torch
from pathlib import Path
from test_pgac_gradients import RestrictedPgacTrainingTests as Fixture
from control_harness.artifacts import Workspace,read_json,atomic_json
from control_harness.differentiable_validation import read_differentiable_validation,validate_differentiable_model
from control_harness.restricted_pgac import train_restricted_pgac
torch.set_num_threads(1);Fixture.setUpClass()
try:
 root=Path(sys.argv[1]);shutil.copytree(Fixture.base/'prepared',root/'workspace');w=Workspace(root/'workspace')
 atomic_json(w.path/'.case-plans/test-fixture.json',{'kind':'control-case-plan-test-fixture-v1','workspace':str(w.path)})
 report=read_differentiable_validation(w,Fixture.validation);allowed=('mode','partition','steps','probes','numeric','gradient','domain','switching_planes')
 vc={k:copy.deepcopy(report['resolved_config'][k]) for k in allowed}
 alternate=validate_differentiable_model(w,Fixture.plant,Fixture.model,{**vc,'steps':49},reference_dataset_id=report['bindings']['dataset']['id'])
 config={**copy.deepcopy(Fixture.config),'epochs':1}
 candidate=train_restricted_pgac(w,Fixture.policy,Fixture.model,config,validation_id=Fixture.validation)
 cr,cm=w.resolve(candidate);pr,pm=w.resolve(Fixture.policy);nn=read_json(pr/'report.json');trained=read_json(cr/'report.json')
 data={'workspace':str(w.path),'validate':{'operation':'validate_differentiable_model','workspace':str(w.path),'inputs':{'model':Fixture.model,'evaluation_model':Fixture.plant,'dataset':report['bindings']['dataset']['id']},'config':vc},
  'optimize':{'operation':'optimize_external_policy','workspace':str(w.path),'inputs':{'policy':Fixture.policy,'model':Fixture.model,'validation':Fixture.validation},'config':config},
  'evaluate':{'operation':'evaluate_external_pgac','workspace':str(w.path),'inputs':{'candidate':candidate},'config':trained['original_evaluation_config']},
  'alternate_validation':alternate,'model_relative_path':w.manifest['artifacts'][Fixture.model]['path'],
  'numeric_proof':{'actual_source_and_sampling':True,'validation_qualified':report['qualified_for_pgac'],'recorded_reference_steps':report['resolved_config']['steps'],
   'imitation_epochs':nn['network']['trained_epochs'],'pgac_epochs':trained['trained_epochs'],'pgac_updates':trained['optimizer_updates'],'pgac_inactive':cm['activation']=='none','f05_or_matlab_run':False}}
 atomic_json(root/'fixture.json',data);print(json.dumps(data))
finally:Fixture.tearDownClass()`;
const query=String.raw`import json,sys
from control_harness.artifacts import Workspace
from control_harness.case_requirements import confirmation_requirements
result=[]
for request in json.load(sys.stdin):
 try:result.append({'ok':True,'rows':[{'scope':r['scope'],'expectedPayload':r['expected_payload'],'bindings':r['bindings']} for r in confirmation_requirements(Workspace(request['workspace']),request)]})
 except Exception as error:result.append({'ok':False,'error':str(error),'error_type':type(error).__name__})
print(json.dumps(result))`;
const base=await mkdtemp(join(output,'actual-source-'));
const fixture=py(make,[base]);
assert.equal(fixture.numeric_proof.validation_qualified,true);assert.equal(fixture.numeric_proof.imitation_epochs,3);
assert.equal(fixture.numeric_proof.pgac_epochs,1);assert.equal(fixture.numeric_proof.pgac_updates,3);assert.equal(fixture.numeric_proof.pgac_inactive,true);

async function clone(){
 const directory=await mkdtemp(join(output,'case-')),workspace=join(directory,'workspace');await cp(fixture.workspace,workspace,{recursive:true});
 await mkdir(join(workspace,'.case-plans'),{recursive:true});await writeFile(join(workspace,'.case-plans/test-fixture.json'),JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace}));
 const value=structuredClone(fixture);for(const key of ['validate','optimize','evaluate'])value[key].workspace=workspace;
 return {...value,workspace,directory};
}
async function parity(requests){
 const expected=py(query,[],JSON.stringify(requests));const actual=[];
 for(let i=0;i<requests.length;i++){assert.equal(expected[i].ok,true,expected[i].error);const rows=await confirmationRequirements(requests[i]);assert.deepEqual(rows,expected[i].rows);actual.push(rows);}
 return actual;
}
const ref=plan=>({plan_id:plan.plan_id,version_hash:plan.version_hash});
function propose(workspace,requirement){
 return py('import json,sys;from control_harness.case_plans import propose_plan; print(json.dumps(propose_plan(**json.load(sys.stdin))))',[],
  JSON.stringify({workspace,scope:requirement.scope,payload:requirement.expectedPayload,bindings:requirement.bindings}));
}
async function authorityFixture(value,run){
 const built=path=>import(pathToFileURL(resolve(project,'vendor/deepseek-harness',path)));
 const [{Context},{default:Projection},{default:Commands},{Session}]=await Promise.all([built('vendor/cordis/lib/index.js'),built('packages/session/session-projection/lib/index.js'),built('packages/interaction/commands/lib/index.js'),built('packages/core/session/lib/index.js')]);
 const ctx=new Context();await ctx.plugin(Projection);await ctx.plugin(Commands);
 ctx.provide('sandboxPolicy',{resolve:()=>({mode:'danger-full-access',workspaceRoot:value.workspace})});
 const authority=installCasePlans(ctx,{codeRoot:project,key:Buffer.alloc(32,19),resolveRequirements:confirmationRequirements});
 const agent={id:'pgac-case-session',session:Session.create('pgac-case-session')};
 try{return await run({ctx,authority,agent,Session});}finally{await ctx.fiber.dispose();}
}

test('three PGAC operations use actual immutable numeric dependencies and matching Host/Python scopes',async()=>{
 const value=await clone();const [validation,training,evaluation]=await parity([value.validate,value.optimize,value.evaluate]);
 assert.deepEqual(validation.map(x=>x.scope),['signals']);assert.deepEqual(training.map(x=>x.scope),['signals','training']);assert.deepEqual(evaluation.map(x=>x.scope),['signals','sampling']);
 assert.equal(training[0].bindings.validation.id,value.optimize.inputs.validation);assert.equal(training[0].expectedPayload.validated_source.kind,'bounded_differentiable_validation');
 assert.deepEqual(evaluation[1].expectedPayload.scenarios,value.evaluate.config.scenarios);
 const changed=structuredClone(value.optimize);changed.config.learning_rate*=2;
 const domain=structuredClone(value.optimize);domain.config.domain.x=[[-2,2]];
 const source=structuredClone(value.optimize);source.inputs.validation=value.alternate_validation;
 const [lrRows,domainRows,sourceRows]=await parity([changed,domain,source]);
 assert.deepEqual(lrRows[0],training[0]);assert.notDeepEqual(lrRows[1].expectedPayload,training[1].expectedPayload);
 assert.deepEqual(domainRows[0],training[0]);assert.notDeepEqual(domainRows[1].expectedPayload,training[1].expectedPayload);
 assert.notDeepEqual(sourceRows[0].bindings,training[0].bindings);
 const resumed=structuredClone(value.optimize);resumed.config.resume_checkpoint='not-opened.pt';resumed.config.resume_checkpoint_sha256='a'.repeat(64);resumed.config.plan_refs={training:{plan_id:'selector-only'}};
 assert.deepEqual((await parity([resumed]))[0],training);
});

test('actual native-command fixture confirmations create Session-bound worker receipts; changed optimization contracts fail',async()=>{
 const value=await clone();await authorityFixture(value,async({ctx,authority,agent,Session})=>{
  const request=structuredClone(value.optimize);request.config.plan_refs={};
  for(const requirement of await confirmationRequirements(request)){
   const plan=propose(value.workspace,requirement);request.config.plan_refs[requirement.scope]=ref(plan);
   const result=await ctx.commands.execute(agent,`/control-confirm ${JSON.stringify({workspace:value.workspace,...ref(plan)})}`,[],new AbortController().signal);
   assert.equal(result.result.kind,'success',result.result.text);assert.equal(JSON.parse(result.result.text).confirmation_source,'test_fixture');
  }
  const receipt=await authority.authorizeRequest({request,agent});
  const verifier=String.raw`import json,sys
from control_harness.artifacts import Workspace
from control_harness.case_requirements import confirmation_requirements
from control_harness.runtime import OperationContext
raw=sys.stdin.read();context=OperationContext();context.bind_raw_request(raw);request=json.loads(raw)
try:
 rows=context.require_case_confirmation(request,confirmation_requirements(Workspace(request['workspace']),request));print(json.dumps({'ok':True,'scopes':[r['scope'] for r in rows],'sources':[r['confirmation_source'] for r in context.case_confirmation_sources]}))
except Exception as error:print(json.dumps({'ok':False,'error_type':type(error).__name__}))`;
  const worker=(raw=receipt.rawRequest,env={})=>py(verifier,[],raw,{...receipt.env,...env});
  assert.deepEqual(worker(),{ok:true,scopes:['signals','training'],sources:['test_fixture','test_fixture']});
  const rawChanged=JSON.parse(receipt.rawRequest);rawChanged.config.learning_rate*=2;assert.equal(worker(JSON.stringify(rawChanged)).ok,false);
  assert.equal(worker(receipt.rawRequest,{CONTROL_CASE_NATIVE_SESSION:'other-session'}).ok,false);
  for(const change of [{learning_rate:request.config.learning_rate*2},{domain:{x:[[-2,2]],u:[[-1,1]],d:[]}}]){
   const candidate=structuredClone(request);Object.assign(candidate.config,change);await assert.rejects(authority.authorizeRequest({request:candidate,agent}),/reviewed|pending|version|differs/);
  }
  const other=structuredClone(request);other.inputs.validation=value.alternate_validation;await assert.rejects(authority.authorizeRequest({request:other,agent}),/dependency|material|version|pending/);
  await assert.rejects(authority.authorizeRequest({request,agent:{id:'other-session',session:Session.create('other-session')}}),/pending/);
 });
});

test('empty evaluation scenarios and changed bound artifact bytes fail both resolvers before numeric admission',async()=>{
 const value=await clone();const missing=structuredClone(value.evaluate);missing.config.scenarios=[];
 await assert.rejects(confirmationRequirements(missing),/scenarios/);assert.equal(py(query,[],JSON.stringify([missing]))[0].ok,false);
 const path=join(value.workspace,value.model_relative_path,'model.json');await writeFile(path,(await readFile(path,'utf8'))+' ');
 await assert.rejects(confirmationRequirements(value.optimize),/changed/);const pythonResult=py(query,[],JSON.stringify([value.optimize]))[0];assert.equal(pythonResult.ok,false);assert.match(pythonResult.error,/changed|hash|integrity/i);
});
