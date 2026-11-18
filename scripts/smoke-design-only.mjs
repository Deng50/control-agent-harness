/** S07 native basic-design acceptance. Never starts a Host or MATLAB session. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {copyFile,readFile} from 'node:fs/promises';
import {isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},'source-audit':{type:'string'},
  fixtures:{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-design-only.mjs --workspace-root <isolated parent> --evidence <json> [--source-audit <S06 rank audit>] [--fixtures <S00>] [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values['workspace-root']&&values.evidence,'Explicit workspace-root/evidence required');
const destination=resolve(values['workspace-root']),evidencePath=resolve(values.evidence),hash=v=>createHash('sha256').update(v).digest('hex');
const relation=relative(resolve(project,'test-output'),destination);assert.ok(relation&&!isAbsolute(relation)&&relation!=='..'&&!relation.startsWith(`..${sep}`));
const json=p=>readFile(p,'utf8').then(JSON.parse),runRoot=resolve(project,'test-output/v2-development/20261117T002006');
const fixtures=resolve(values.fixtures??resolve(runRoot,'fixtures'));
const frozenBytes=await readFile(resolve(fixtures,'F01_spring_2x1.json'));
assert.equal(hash(frozenBytes),'22525d4196bda85f3e3ddba0de0493859a91f515fe301f99cd54f40a84e55824');
const frozen=JSON.parse(frozenBytes);
const audit=await json(resolve(values['source-audit']??resolve(runRoot,'S06/predictive-training-rank-audit.json')));
assert.equal(audit.status,'TRAINING_INPUT_RANK_FULL_ON_DECLARED_CASES');
const controller={horizon:3,max_iterations:100,solver_tolerance:1e-10},weights={q:1,r:.03,s:.005};
const linearPoint={x:[.12,-.04],reference:[.03],disturbance:[.01],previous:[0.]};
const localPoint={x:[.12,-.04],u:[.03],d:[.01]},localBounds={x:[[.05,.2],[-.1,.05]],u:[[-.2,.2]],d:[[-.03,.03]]};
const cases=[{name:'draft',source:'description'}, {name:'explicit-model',source:'file',model:resolve(project,'examples/spring-damper/nominal.json')},
  ...audit.checks.filter(row=>['matlab_function','simulink'].includes(row.backend)).map(row=>({name:row.backend,source:'sampled',data:row})),
  {name:'nonlinear',source:'file',model:resolve(project,'examples/nonlinear/reference.json')}];
let evidence=values.resume?await json(evidencePath):{schema_version:1,stage:'S07',status:'RUNNING',destination,
  startedAt:new Date().toISOString(),confirmation_source:'test_fixture',preRegistered:{controller,weights,linearPoint,localPoint,localBounds,identificationAcceptance:{atol:1e-6,rtol:1e-5}},
  scope:'C01 parameterized draft; C02 explicit/recorded-.M/SLX/local predictors and independent standalone MPC; no GA, original-plant closed loop, LLM or MATLAB service operation.',
  sessions:{},modelSessions:{},jobs:{},admissions:[],confirmations:[],checks:[],bundles:[]};
assert.equal(evidence.destination,destination);
assert.deepEqual(evidence.preRegistered,{controller,weights,linearPoint,localPoint,localBounds,identificationAcceptance:{atol:1e-6,rtol:1e-5}},'Observer resume cannot change engineering assumptions');
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const py=(code,args,extra={})=>execFileSync(python,['-B','-c',code,...args],{encoding:'utf8',windowsHide:true,maxBuffer:8*1024*1024,
  env:{...process.env,PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',...extra}});
const independentZoh=String.raw`import json,sys,numpy as np
from scipy.linalg import expm
s=json.loads(open(sys.argv[1],encoding='utf-8').read())['source_model'];A=np.asarray(s['A']);B=np.asarray(s['B']);E=np.asarray(s['E']);nx,nu,nd=len(A),B.shape[1],E.shape[1]
G=np.zeros((nx+nu+nd+1,nx+nu+nd+1));G[:nx,:nx]=A;G[:nx,nx:nx+nu]=B;G[:nx,nx+nu:nx+nu+nd]=E
T=expm(G*s['interface']['sample_time']);print(json.dumps({'A':T[:nx,:nx].tolist(),'B':T[:nx,nx:nx+nu].tolist(),'E':T[:nx,nx+nu:nx+nu+nd].tolist(),'bias':T[:nx,-1].tolist(),'C':s['C']}))`;
const internalReference=String.raw`import json,sys,numpy as np,torch
from pathlib import Path
from control_harness.artifacts import Workspace
from control_harness.control import MPCPolicy
from control_harness.modeling import load_model,tensor
torch.set_num_threads(1)
ws=Workspace(sys.argv[1]);root=Path(sys.argv[2]);casepath=Path(sys.argv[3]);design=json.loads((root/'design.json').read_text(encoding='utf-8'))
prediction=design['prediction'];model=load_model(ws,prediction['artifact_id']);cfg={**design['controller_config'],**design['weights']};policy=MPCPolicy(model,cfg)
point=design['solver_check']['point'];sequence=[[.03]*model.interface.nu for _ in range(policy.horizon)];case={**point,'sequence':sequence};casepath.write_text(json.dumps(case)+'\n',encoding='utf-8')
xt,rt,dt,pt=[tensor(point[k]) for k in ('x','reference','disturbance','previous')]
cost,margins=policy.rollout(tensor(sequence),xt,rt,dt,pt);next_x=model(xt,tensor(sequence[0]),dt);next_y=model.output(next_x)
action,diag=policy.act(point['x'],point['reference'],point['disturbance'],point['previous'])
result={'case':case,'prediction':{'next_x':next_x.detach().tolist(),'next_y':next_y.detach().tolist(),'sample_time':model.interface.sample_time},'cost':cost.item(),'margins':margins.detach().tolist(),'action':action.tolist(),'diagnostics':diag}
if all(hasattr(model,k) for k in ('A','B','E','bias','C')) and not getattr(model,'terms',[]):result['affine']={k:getattr(model,k).detach().tolist() for k in ('A','B','E','bias','C')}
print(json.dumps(result,allow_nan=False))`;
const draftProbe=String.raw`import importlib.util,json,sys
from pathlib import Path
root=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('controller',root/'controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=m.load_design(root)
try:d.instantiate()
except m.NotInstantiated as e:print(json.dumps({'status':'not_instantiated','missing_rejected':True,'message':str(e),'model_instantiated':d.model_instantiated,'ga_trials':0,'closed_loop':'not_run'}))
else:raise RuntimeError('Missing physical parameters were silently instantiated')`;
const isolated=String.raw`import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('No repository import from independent controller')
sys.meta_path.insert(0,Block());p=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(p,run_name='__main__')
assert not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)`;
const near=(a,b,label,atol=1e-6,rtol=1e-5)=>{if(Array.isArray(b)){assert.equal(a.length,b.length,label);b.forEach((v,k)=>near(a[k],v,`${label}[${k}]`,atol,rtol));}else assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=atol+rtol*Math.abs(b),`${label}: ${a} != ${b}`);};

try{
  const zoh=JSON.parse(py(independentZoh,[resolve(fixtures,'F01_spring_2x1.json')]));evidence.independentZohReference=zoh;
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG});
  for(const item of cases){
    const workspace=resolve(destination,item.name);await prepareSmokeWorkspace(workspace);
    if(!values.resume){try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'Use --resume for existing admissions');}catch(e){if(e.code!=='ENOENT')throw e;}
      await writeSmokeEvidence(resolve(workspace,'.case-plans/test-fixture.json'),{kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true});}
    if(!evidence.sessions[item.name]){const session=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-controller'}});assert.equal(session.agentPreset,'control-controller');evidence.sessions[item.name]=session.sessionId;await save();}
    const sessionId=evidence.sessions[item.name];
    const stage=async(key,operation,inputs={},config={})=>{
      let owner=sessionId,role='controller';
      if(operation==='register_model'){
        evidence.modelSessions??={};
        if(!evidence.modelSessions[item.name]){const session=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});assert.equal(session.agentPreset,'control-model');evidence.modelSessions[item.name]=session.sessionId;await save();}
        owner=evidence.modelSessions[item.name];role='model';
      }
      const request={agent:role,operation,workspace,inputs,config},signature=hash(JSON.stringify(request));let row=evidence.jobs[`${item.name}:${key}`];
      if(row){assert.equal(row.signature,signature);assert.ok(row.jobId,'Unknown admission: reconcile original history before attaching ID, never resubmit');if(row.result)return row.result;}
      else{row=evidence.jobs[`${item.name}:${key}`]={operation,signature,status:'ADMISSION_PENDING'};await save();const started=await command(rpc,owner,`/control ${JSON.stringify(request)}`);
        Object.assign(row,{jobId:started.jobId,attemptId:started.attemptId,sessionId:owner,status:'RUNNING'});evidence.admissions.push({case:item.name,operation,sessionId:owner,jobId:row.jobId,attemptId:row.attemptId});await save();}
      for(;;){const state=await command(rpc,row.sessionId??owner,`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);
        if(['completed','failed','killed'].includes(state.record.state)){row.status=state.record.state;await save();assert.equal(row.status,'completed',JSON.stringify(state));row.result=state.record.result;assert.ok(row.result);
          const text=state.output?.chunks?.map(c=>c.text).join('')??'';row.gaProgressEvents=text.split(/\r?\n/).filter(line=>{try{return JSON.parse(line).stage==='controller_tuning';}catch{return false;}}).length;assert.equal(row.gaProgressEvents,0);await save();return row.result;}
        assert.notEqual(state.record.state,'interrupted','Host ownership lost: inspect saved job');await new Promise(done=>setTimeout(done,250));}
    };
    const synthesize=async(key,inputs,settings)=>{
      const request={operation:'synthesize_mpc',workspace,inputs,config:settings},requirements=await confirmationRequirements(request),refs={};
      for(const row of requirements){const plan=await stage(`${key}:plan:${row.scope}`,'propose_case_plan',{}, {scope:row.scope,payload:row.expectedPayload,bindings:row.bindings});refs[row.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};
        const confirmed=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace,...refs[row.scope]})}`);assert.equal(confirmed.confirmation_source,'test_fixture');evidence.confirmations.push({case:item.name,scope:row.scope,...refs[row.scope],confirmation_source:'test_fixture'});await save();}
      const result=await stage(key,'synthesize_mpc',inputs,{...settings,...(Object.keys(refs).length?{plan_refs:refs}:{})});assert.equal(result.kind,'controller_design');assert.equal(result.status,'verified');
      const report=await json(resolve(result.artifact_path,'report.json')),metadata=await json(resolve(result.artifact_path,'artifact.json'));
      assert.equal(metadata.activation,'none');assert.equal(report.controller_parameters_tuned,false);assert.equal(report.closed_loop.status,'not_run');assert.equal(report.closed_loop.ga_trials,0);assert.equal(report.closed_loop.plant_rollout_steps,0);
      const bundle=resolve(result.artifact_path,'bundle'),runtime=await readFile(resolve(bundle,'mpc_runtime.py'),'utf8');assert.ok(!runtime.includes('GeneticOperators'));
      for(const[name,expected]of Object.entries(metadata.files))assert.equal(hash(await readFile(resolve(result.artifact_path,name))),expected);
      let reference=null,caseFile=null;
      if(!report.model_instantiated){const probe=JSON.parse(py(draftProbe,[bundle],{PYTHONPATH:''}));assert.equal(probe.missing_rejected,true);assert.equal(report.solver_check.status,'not_run');assert.deepEqual(report.parameters,{});assert.equal(report.required_parameters.length,7);
        caseFile=resolve(workspace,`${key}-case.json`);await writeSmokeEvidence(caseFile,{purpose:'syntax and missing-parameter rejection only',no_numeric_conditions:true});}
      else{
        assert.equal(report.solver_check.status,'passed');assert.equal(report.solver_check.solves,1);
        caseFile=resolve(workspace,`${key}-case.json`);reference=JSON.parse(py(internalReference,[workspace,result.artifact_path,caseFile],{PYTHONPATH:resolve(project,'src')}));
        const output=resolve(workspace,`${key}-portable.json`);
        execFileSync(python,['-I','-B','-c',isolated,resolve(bundle,'controller.py'),'--input',caseFile,'--output',output],{cwd:bundle,encoding:'utf8',windowsHide:true,env:{...process.env,PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1'}});
        const portable=await json(output);near(portable.u,reference.action,'independent native package action');near(portable.diagnostics.predicted_objective,reference.diagnostics.predicted_objective,'independent native package cost');assert.equal(portable.closed_loop,'not_run');assert.equal(portable.ga_trials,0);
        if(['explicit-model','matlab_function','simulink'].includes(item.name)){for(const name of ['A','B','E','bias','C'])near(reference.affine[name],zoh[name],`${item.name} independent exact-ZOH ${name}`);}
        if(key==='local'){
          const p=report.prediction.report;assert.equal(p.scope,'local_jacobian_not_global_linear_model');assert.deepEqual(p.operating_point,localPoint);assert.deepEqual(p.validity_bounds,localBounds);
          const affine=report.prediction.spec;const expected=p.f0.map((value,k)=>value-affine.A[k].reduce((s,v,j)=>s+v*localPoint.x[j],0)-affine.B[k].reduce((s,v,j)=>s+v*localPoint.u[j],0)-affine.E[k].reduce((s,v,j)=>s+v*localPoint.d[j],0));near(affine.bias,expected,'local nonzero affine bias',1e-12,1e-12);assert.equal(p.continuous_derivative_used,false);
        }
      }
      const matlab=await json(resolve(bundle,'matlab_model.json'));
      const entry={case:item.name,key:`${item.name}:${key}`,workspace,sessionId,artifact_id:result.artifact_id,artifact_path:result.artifact_path,
        bundle,model_instantiated:report.model_instantiated,matlab_kind:matlab.kind,case_path:caseFile,
        python_reference:reference,matlab_validation:'not_run',matlab_license:'not_checked'};
      const index=evidence.bundles.findIndex(row=>row.key===entry.key);if(index<0)evidence.bundles.push(entry);else evidence.bundles[index]=entry;
      evidence.checks.push({case:item.name,key,artifact_id:result.artifact_id,design_status:report.status,model_instantiated:report.model_instantiated,solver_check:report.solver_check,closed_loop:report.closed_loop,prediction:report.prediction,packageIndependentProcess:!!reference,confirmation_source:'test_fixture'});await save();return result;
    };
    if(item.source==='description')await synthesize('draft',{description:'请设计弹簧振子MPC，只做基础设计，不要调参、采数或仿真；物理参数尚未提供。'}, {template:'spring_damper',controller,weights,solver_check:false});
    else if(item.source==='sampled'){
      assert.equal(hash(await readFile(resolve(item.data.dataset_root,'transitions.npz'))),item.data.transitions_sha256);
      const source=await stage('import','import_artifact',{artifact_path:item.data.dataset_root});
      await synthesize('identify',{dataset:source.artifact_id},{prediction:{method:'identify_affine',C:frozen.source_model.C,fit:{fit_bias:true},acceptance:{atol:1e-6,rtol:1e-5}},controller,weights,solver_check:linearPoint});
    }else{
      const file=resolve(workspace,'explicit-source.json');if(!values.resume)await copyFile(item.model,file);
      if(item.name==='explicit-model')assert.equal(hash(await readFile(file)),frozen.source.sha256,'Frozen explicit spring source changed');
      const model=await stage('register','register_model',{model_path:file},{source:{kind:'user_nominal',description:'Explicit fixed synthetic predictor for independent basic design; no evaluation plant'}});
      if(item.name==='nonlinear'){
        await synthesize('direct',{model:model.artifact_id},{prediction:{method:'direct'},controller,weights,solver_check:linearPoint});
        await synthesize('local',{model:model.artifact_id},{prediction:{method:'linearize',operating_point:localPoint,validity_bounds:localBounds},controller,weights,solver_check:linearPoint});
      }else await synthesize('direct',{model:model.artifact_id},{prediction:{method:'direct'},controller,weights,solver_check:linearPoint});
    }
    const records=(await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:10000}})).records;
    assert.ok(!records.some(row=>row.event.type.startsWith('turn/')),'Fixed acceptance opened a model turn');
    if(evidence.modelSessions?.[item.name]){const sourceEvents=(await rpc('session/events',{request:{address:{kind:'session',sessionId:evidence.modelSessions[item.name]},maxMessages:10000}})).records;assert.ok(!sourceEvents.some(row=>row.event.type.startsWith('turn/')));}
    const final=await json(resolve(workspace,'manifest.json'));assert.ok(!Object.values(final.artifacts).some(row=>['controller','reference_model','expert_dataset','imitation_policy','optimized_policy'].includes(row.kind)),'Design-only branch created an evaluation/tuned/controller-policy chain');
  }
  assert.equal(evidence.bundles.length,6);evidence.status='PASS';evidence.finishedAt=new Date().toISOString();
}catch(error){evidence.status='FAIL';evidence.error=error instanceof Error?error.stack:String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,bundles:evidence.bundles.length,jobs:evidence.admissions.map(({case:name,operation,jobId})=>({case:name,operation,jobId})),confirmation_source:'test_fixture'}));}
