/** S08 C05 native GA acceptance; never starts a Host or MATLAB session. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {constants} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {copyFile,readFile,readdir,writeFile} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},fixtures:{type:'string'},
  'web-log':{type:'string'},resume:{type:'boolean'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-goal-tuning.mjs --workspace-root <isolated parent> --evidence <json> [--fixtures <S00>] [--web-log <Host log>] [--resume]');process.exit(0);}
assert.ok(values['workspace-root']&&values.evidence,'Pass explicit workspace-root and evidence');
const destination=resolve(values['workspace-root']),evidencePath=resolve(values.evidence),runRoot=resolve(project,'test-output/v2-development/20261117T002006');
const fixtures=resolve(values.fixtures??resolve(runRoot,'fixtures')),hash=value=>createHash('sha256').update(value).digest('hex');
const relation=relative(resolve(project,'test-output'),destination);assert.ok(relation&&!isAbsolute(relation)&&relation!=='..'&&!relation.startsWith(`..${sep}`));
const json=path=>readFile(path,'utf8').then(JSON.parse),frozenBytes=await readFile(resolve(fixtures,'F02_existing_linear_4x2.json'));
assert.equal(hash(frozenBytes),'ac6364b26915a81abc790a88c6346e2fe91fdd62950898ecac0cc0c67f6ae7d9');
const frozen=JSON.parse(frozenBytes),f05Bytes=await readFile(resolve(fixtures,'F05_expert_and_tradeoff_candidates.json'));
assert.equal(hash(f05Bytes),'66493849a0fa57eaba303e576077d40e8ad5779118617fa0a34d982231a73d2b');
const f05=JSON.parse(f05Bytes),controllerFile=resolve(project,'examples/linear/controller.json'),controllerBytes=await readFile(controllerFile);
assert.equal(hash(controllerBytes),frozen.source_files.controller.sha256);const original=JSON.parse(controllerBytes);
assert.deepEqual(original,f05.source_expert_controller.config);
const recipes=['tracking','energy'];
const configurations=Object.fromEntries(recipes.map(recipe=>[recipe,{...structuredClone(original),minimum_improvement:0,
  evaluation:{schema_version:1,recipe,weights:structuredClone(f05.manual_scoring_oracle.weights[recipe]),
    scales:structuredClone(f05.manual_scoring_oracle.scales),energy_model:'action_proxy',hard_gates:{constraint_tolerance:1e-6}}}]));
const preRegistered={fixture_sha256:hash(frozenBytes),preference_source_sha256:hash(f05Bytes),controller_source_sha256:hash(controllerBytes),
  configurations,resolved_solver_defaults:{max_iterations:80,solver_tolerance:1e-8},metric_semantics:'energy=sum(action_coordinate^2)*dt is an action proxy, not measured electrical/mechanical energy',
  selection:'Tuning and verification only; frozen test scenarios evaluated after choosing parameters, never choose preferences using test.',
  independent_tolerance:{atol:1e-6,rtol:1e-5}};
let evidence=values.resume?await json(evidencePath):{schema_version:1,stage:'S08',status:'RUNNING',destination,started_at:new Date().toISOString(),
  confirmation_source:'test_fixture',preRegistered,sessions:{},modelSessions:{},jobs:{},admissions:[],confirmations:[],rejections:[],checks:[]};
assert.equal(evidence.destination,destination);assert.deepEqual(evidence.preRegistered,preRegistered,'Observer resume cannot change preferences, scenarios, scales or search');
if(values.resume){
  evidence.observer_history??=[];
  if(evidence.error)evidence.observer_history.push({status:evidence.status,error:evidence.error,finished_at:evidence.finished_at??null,resumed_at:new Date().toISOString()});
  delete evidence.error;delete evidence.finished_at;evidence.status='RUNNING';
}
const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const near=(actual,expected,label,atol=1e-6,rtol=1e-5)=>{
  if(Array.isArray(expected)){assert.ok(Array.isArray(actual),label+' array');assert.equal(actual.length,expected.length,label+' length');expected.forEach((v,k)=>near(actual[k],v,`${label}[${k}]`,atol,rtol));}
  else assert.ok(Number.isFinite(actual)&&Number.isFinite(expected)&&Math.abs(actual-expected)<=atol+rtol*Math.abs(expected),`${label}: ${actual} != ${expected}`);
};
function auditEvaluation(group,evaluation,label){
  assert.deepEqual(group.evaluation,evaluation,label+' frozen evaluation');const scores=[];
  for(const [index,run]of group.runs.entries()){
    const name=`${label}.run${index}`,objective=run.objective_result;
    if(run.status==='failed'){assert.equal(objective.score,null);assert.equal(objective.hard_constraints_passed,false);scores.push(null);continue;}
    assert.equal(run.status,'completed');assert.equal(run.trajectory.length,run.scenario.steps);
    const rows=run.trajectory,i=frozen.reference.interface,dt=i.sample_time,initialX=run.scenario.initial_state;
    const product=(matrix,vector)=>matrix.map(row=>row.reduce((sum,value,k)=>sum+value*vector[k],0));
    const initialY=product(frozen.reference.C,initialX),initialU=i.controls.map(v=>Math.min(v.upper,Math.max(v.lower,0)));
    near(run.initial_output,initialY,name+' initial output',1e-12,1e-12);near(run.initial_control,initialU,name+' initial action',1e-12,1e-12);
    let error2=0,energy=0,tv=0,previous=initialU.slice(),violations=0;const perOutput=Array(i.outputs.length).fill(0),perOvershoot=Array(i.outputs.length).fill(0);
    const tolerance=evaluation.hard_gates.constraint_tolerance;
    const count=(values,variables)=>{for(let j=0;j<variables.length;j++)if(Math.max(variables[j].lower-values[j],0)+Math.max(values[j]-variables[j].upper,0)>tolerance)violations++;};
    count(initialX,i.states);count(initialY,i.outputs);count(initialU,i.controls);
    for(const [k,row]of rows.entries()){
      assert.equal(row.step,k);near(row.reference,run.scenario.reference,name+' held reference',1e-12,1e-12);
      count(row.next_x,i.states);count(row.output,i.outputs);count(row.u,i.controls);
      row.output.forEach((value,j)=>{const error=value-row.reference[j];error2+=error*error;perOutput[j]+=error*error;
        const step=row.reference[j]-initialY[j];if(Math.abs(step)>evaluation.metrics.step_atol)perOvershoot[j]=Math.max(perOvershoot[j],Math.max(0,(value-row.reference[j])*Math.sign(step))/Math.abs(step));});
      row.u.forEach((value,j)=>{energy+=value*value*dt;const delta=value-previous[j];tv+=Math.abs(delta);
        if(run.input_rate_bounds){const [lo,hi]=run.input_rate_bounds[j];if(Math.max(lo-delta,0)+Math.max(delta-hi,0)>tolerance)violations++;}});previous=row.u;
    }
    const applicable=perOvershoot.filter((_,j)=>Math.abs(run.scenario.reference[j]-initialY[j])>evaluation.metrics.step_atol);
    const metrics={rmse:Math.sqrt(error2/(rows.length*i.outputs.length)),energy,tv,overshoot:applicable.length?Math.max(...applicable):null};
    for(const key of Object.keys(metrics))if(metrics[key]===null)assert.equal(run.objective_metrics[key],null,name+' unavailable '+key);else near(run.objective_metrics[key],metrics[key],name+' '+key,1e-10,1e-9);
    perOutput.forEach((value,j)=>near(run.objective_metrics.per_output[j].rmse,Math.sqrt(value/rows.length),name+' channel RMSE',1e-10,1e-9));
    assert.equal(run.objective_metrics.hard_constraint_violations,violations,name+' independent violations');
    let score=0;
    for(const key of Object.keys(metrics)){
      const term=objective.terms[key];assert.equal(term.weight,evaluation.weights[key]);assert.equal(term.scale,evaluation.scales[key]);
      const weighted=evaluation.weights[key]===0?0:evaluation.weights[key]*metrics[key]/evaluation.scales[key];near(term.weighted,weighted,name+' weighted '+key,1e-10,1e-9);score+=weighted;
    }
    if(violations){assert.equal(objective.score,null);assert.equal(objective.hard_constraints_passed,false);scores.push(null);}
    else{assert.equal(objective.hard_constraints_passed,true);near(objective.score,score,name+' score',1e-10,1e-9);scores.push(score);}
  }
  if(scores.some(value=>value===null)){assert.equal(group.score,null);assert.equal(group.hard_constraints_passed,false);}
  else near(group.score,scores.reduce((a,b)=>a+b,0)/scores.length,label+' equal-scenario mean score',1e-10,1e-9);
  return{runs:group.runs.length,scores,metric_reference:'independent scalar recorded-row formulas; original fixed scales and preferences'};
}
// The controller is loaded from the published bundle in a fresh isolated Python
// process. The evaluation plant below is an independent scalar matrix formula;
// no Workspace/load_model/closed_loop/objectives implementation is imported.
const independentReplay=String.raw`import hashlib,importlib.abc,importlib.util,json,math,sys
from pathlib import Path
import numpy as np,torch
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name=='control_harness' or name.startswith('control_harness.'):raise ModuleNotFoundError('Harness imports forbidden in portable replay')
sys.meta_path.insert(0,Block());torch.set_num_threads(1)
bundle=Path(sys.argv[1]);fixture=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));report=json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(report['evaluation'],sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()==report['evaluation_hash']
loader=importlib.util.spec_from_file_location('portable_goal_controller',bundle/'controller.py');module=importlib.util.module_from_spec(loader);sys.modules[loader.name]=module;loader.loader.exec_module(module)
design=module.load_design(bundle);policy=design.instantiate();plant=fixture['reference'];pred=fixture['nominal'];interface=plant['interface'];A=plant['A'];B=plant['B'];E=plant['E'];C=plant['C'];dt=interface['sample_time'];nx=len(A);nu=len(B[0]);ny=len(C)
def row_sum(matrix,values):return [math.fsum(coefficient*values[k] for k,coefficient in enumerate(row)) for row in matrix]
def step(x,u,d):
 ax=row_sum(A,x);bu=row_sum(B,u);ed=row_sum(E,d)
 return [ax[k]+bu[k]+ed[k] for k in range(nx)]
def compare(a,b,label):
 aa=np.asarray(a,float);bb=np.asarray(b,float);assert aa.shape==bb.shape,label+' shape';assert np.isfinite(aa).all() and np.isfinite(bb).all(),label+' finite'
 np.testing.assert_allclose(aa,bb,atol=1e-6,rtol=1e-5,err_msg=label);return float(np.max(np.abs(aa-bb))) if aa.size else 0.
chosen=report['chosen_parameters'];actual={k:policy.config[k] for k in chosen};assert actual==chosen,'Bundle does not use chosen parameters'
maximum_state=0.;maximum_action=0.;maximum_output=0.;runs=[]
for saved in report['test']['runs']:
 case=saved['scenario'];x=list(case['initial_state']);reference=list(case['reference']);d=list(case['disturbance']);previous=[min(max(0.,v['lower']),v['upper']) for v in interface['controls']];initial_control=list(previous);initial_output=row_sum(C,x);policy.warm=None;rows=[]
 assert len(saved['trajectory'])==case['steps']
 for index in range(case['steps']):
  u,diagnostics=policy.act(x,reference,d,previous);u=u.tolist();xn=step(x,u,d);y=row_sum(C,xn);record=saved['trajectory'][index]
  assert record['step']==index;maximum_state=max(maximum_state,compare(x,record['x'],'pre-state'),compare(xn,record['next_x'],'next-state'));maximum_action=max(maximum_action,compare(u,record['u'],'action'));maximum_output=max(maximum_output,compare(y,record['output'],'output'));compare(d,record['d'],'disturbance');compare(reference,record['reference'],'reference')
  rows.append({'x':list(x),'u':u,'next_x':xn,'output':y});x=xn;previous=u
 controls=np.asarray([r['u'] for r in rows]);outputs=np.asarray([r['output'] for r in rows]);errors=outputs-np.asarray(reference);increments=np.diff(np.vstack([initial_control,controls]),axis=0)
 metrics={'rmse':float(np.sqrt(np.mean(errors**2))),'energy':float(np.sum(controls**2)*dt),'tv':float(np.abs(increments).sum())}
 overs=[]
 for j in range(ny):
  delta=reference[j]-initial_output[j]
  overs.append(None if abs(delta)<=1e-12 else max(0.,max((row[j]-reference[j])*math.copysign(1.,delta) for row in outputs))/abs(delta))
 metrics['overshoot']=max((v for v in overs if v is not None),default=None)
 for key,value in metrics.items():
  if value is None:assert saved['objective_metrics'][key] is None
  else:compare([value],[saved['objective_metrics'][key]],'independent physical '+key)
 violation_count=0;tolerance=report['evaluation']['hard_gates']['constraint_tolerance']
 for name,group,values in [('state',interface['states'],[case['initial_state']]+[r['next_x'] for r in rows]),('input',interface['controls'],[initial_control]+[r['u'] for r in rows]),('output',interface['outputs'],[initial_output]+[r['output'] for r in rows])]:
  for values_row in values:
   for j,v in enumerate(values_row):violation_count+=int(max(group[j]['lower']-v,0.)+max(v-group[j]['upper'],0.)>tolerance)
 if policy.rate is not None:
  for delta in increments:
   for j,v in enumerate(delta):violation_count+=int(max(policy.rate[j][0]-v,0.)+max(v-policy.rate[j][1],0.)>tolerance)
 assert violation_count==saved['objective_metrics']['hard_constraint_violations']
 runs.append({'scenario':case,'metrics':metrics,'hard_constraint_violations':violation_count,'per_output_overshoot':overs,'initial_output':initial_output,'initial_control':initial_control})
assert not any(k=='control_harness' or k.startswith('control_harness.') for k in sys.modules)
print(json.dumps({'status':'PASS','controller_parameters':actual,'plant_reference':'frozen F02 scalar row sums, no domain transition','new_process_isolated':True,'project_imports':False,'max_state_error':maximum_state,'max_action_error':maximum_action,'max_output_error':maximum_output,'runs':runs},allow_nan=False))`;

try{
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  for(const recipe of recipes){
    const workspace=resolve(destination,recipe);await prepareSmokeWorkspace(workspace);
    if(!values.resume){try{assert.equal(Object.keys((await json(resolve(workspace,'manifest.json'))).artifacts??{}).length,0,'Use --resume to observe existing admissions');}catch(error){if(error.code!=='ENOENT')throw error;}}
    const markerPath=resolve(workspace,'.case-plans/test-fixture.json');let marker;
    try{marker=await json(markerPath);}catch(error){
      if(error.code!=='ENOENT')throw error;
      assert.ok(!evidence.admissions.some(row=>row.recipe===recipe),'Missing test marker after admission: preserve this workspace as invalid evidence; do not relabel prior confirmations');
      marker={kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true};
      await writeSmokeEvidence(markerPath,marker);
    }
    assert.equal(marker.kind,'control-case-plan-test-fixture-v1');assert.equal(resolve(marker.workspace),workspace);
    for(const [group,preset]of [['sessions','control-controller'],['modelSessions','control-model']])if(!evidence[group][recipe]){
      const session=await rpc('session/create',{request:{cwd:workspace,agentPreset:preset}});assert.equal(session.agentPreset,preset);evidence[group][recipe]=session.sessionId;await save();}
    const sessionId=evidence.sessions[recipe],modelSession=evidence.modelSessions[recipe];
    const snapshot=async()=>{
      let attempts=[];try{attempts=(await readdir(resolve(workspace,'.control-jobs'))).filter(name=>/^[0-9a-f-]{36}$/.test(name)).sort();}catch(error){if(error.code!=='ENOENT')throw error;}
      const history=await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace})}`),records=(await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:10000}})).records;
      return{attempts,history:history.attempts.map(row=>row.attemptId).sort(),registeredJobs:records.filter(row=>row.event.type==='job/registered').map(row=>row.event.data.jobId),
        confirmationCommands:records.filter(row=>row.event.type==='command/run'&&row.event.data.name==='control-confirm').length};
    };
    const deny=async(label,request)=>{
      evidence.rejections??=[];if(evidence.rejections.some(row=>row.label===label))return;
      const before=await snapshot(),response=await rpc('commands/execute',{agentId:sessionId,line:`/control ${JSON.stringify({agent:'controller',...request})}`,submittedAttachments:[]});
      assert.equal(response.result.kind,'error',label+' was admitted');assert.match(response.result.text,/plan|confirmation|version|review/i);
      const after=await snapshot();assert.deepEqual(after,before,label+' created native job/history/attempt or confirmation');
      evidence.rejections.push({label,status:'PASS_ZERO_ADMISSION',nativeKind:response.result.kind,message:response.result.text,before,after,newJobs:0,newHistory:0,newAttempts:0,confirmation_source:'test_fixture'});await save();
    };
    const stage=async(key,operation,inputs={},config={})=>{
      const register=operation==='register_model',owner=register?modelSession:sessionId,request={agent:register?'model':'controller',operation,workspace,inputs,config},signature=hash(JSON.stringify(request)),id=`${recipe}:${key}`;
      let row=evidence.jobs[id];
      if(row){assert.equal(row.signature,signature,'Do not change an admitted request');assert.ok(row.jobId,'Unknown admission: reconcile saved history before continuing, never submit twice');if(row.result)return row.result;}
      else{row=evidence.jobs[id]={operation,signature,status:'ADMISSION_PENDING',sessionId:owner};await save();const admission=await command(rpc,owner,`/control ${JSON.stringify(request)}`);
        Object.assign(row,{jobId:admission.jobId,attemptId:admission.attemptId,status:'RUNNING'});evidence.admissions.push({recipe,key,operation,sessionId:owner,jobId:row.jobId,attemptId:row.attemptId});await save();}
      for(;;){const state=await command(rpc,owner,`/control-status ${JSON.stringify({workspace,attemptId:row.attemptId})}`);
        if(['completed','failed','killed'].includes(state.record.state)){row.status=state.record.state;row.lastRecord=state.record;await save();assert.equal(row.status,'completed',JSON.stringify(state));row.result=state.record.result;assert.ok(row.result);await save();return row.result;}
        assert.notEqual(state.record.state,'interrupted','Observe/recover the saved existing attempt; do not resubmit');await new Promise(done=>setTimeout(done,300));}
    };
    const models={};
    for(const name of ['nominal','reference']){
      const source=resolve(project,`examples/linear/${name}.json`),local=resolve(workspace,`${name}-source.json`);assert.equal(hash(await readFile(source)),frozen.source_files[name].sha256);
      try{await readFile(local);}catch(error){
        if(error.code!=='ENOENT')throw error;
        assert.ok(!evidence.jobs[`${recipe}:register-${name}`],'A previously admitted source disappeared; inspect the saved attempt');
        await copyFile(source,local,constants.COPYFILE_EXCL);
      }
      assert.equal(hash(await readFile(local)),frozen.source_files[name].sha256);
      models[name]=await stage(`register-${name}`,'register_model',{model_path:local},{source:{kind:name==='reference'?'reference_simulator':'user_nominal',description:'Frozen F02 explicit synthetic '+name+'; no fidelity claim for a production plant'}});
    }
    const inputs={model:models.nominal.artifact_id,evaluation_model:models.reference.artifact_id},config=configurations[recipe],request={operation:'tune_controller',workspace,inputs,config};
    const required=await confirmationRequirements(request);assert.deepEqual(required.map(r=>r.scope),['signals','training']);const refs={};
    for(const item of required){const plan=await stage(`plan-${item.scope}`,'propose_case_plan',{}, {scope:item.scope,payload:item.expectedPayload,bindings:item.bindings});refs[item.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};}
    await deny(`${recipe}:before-confirmation`,{...request,config:{...config,plan_refs:refs}});
    for(const item of required){
      if(evidence.confirmations.some(row=>row.recipe===recipe&&row.scope===item.scope&&row.plan_id===refs[item.scope].plan_id&&row.version_hash===refs[item.scope].version_hash))continue;
      const confirmed=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace,...refs[item.scope]})}`);assert.equal(confirmed.confirmation_source,'test_fixture');evidence.confirmations.push({recipe,scope:item.scope,...refs[item.scope],confirmation_source:'test_fixture'});await save();
    }
    const result=await stage('tune','tune_controller',inputs,{...config,plan_refs:refs});assert.equal(result.kind,'controller');assert.equal(result.status,'verified');
    const changedPreference=structuredClone(config);changedPreference.evaluation.scales.energy*=2;
    await deny(`${recipe}:changed-scale-with-old-confirmation`,{...request,config:{...changedPreference,plan_refs:refs}});
    const artifact=resolve(result.artifact_path),reportBytes=await readFile(resolve(artifact,'report.json')),report=JSON.parse(reportBytes),metadata=await json(resolve(artifact,'artifact.json')),descriptor=await json(resolve(artifact,'controller.json'));
    for(const[name,digest]of Object.entries(metadata.files))assert.equal(hash(await readFile(resolve(artifact,name))),digest);
    assert.equal(report.algorithm,'genetic_algorithm');assert.equal(report.test_used_for_tuning,false);assert.equal(report.test_used_for_selection,false);assert.equal(report.test_evaluations,1);assert.deepEqual(report.selection_partitions,['tuning','verification']);
    assert.equal(report.trials.length,config.budget);assert.equal(report.baseline.hard_constraints_passed,true);assert.equal(report.test.hard_constraints_passed,true);assert.equal(report.export.status,'exported');
    assert.ok(['tuned','baseline_retained'].includes(report.selection));assert.equal(report.tuned_candidate_accepted,report.selection==='tuned');assert.deepEqual(report.chosen_parameters,descriptor.parameters);
    assert.ok(report.best&&report.last&&Array.isArray(report.failed_candidates),'Best/last/failed records are mandatory even if no candidate fails');
    assert.deepEqual(report.test.runs.map(r=>r.scenario),config.test_scenarios);assert.deepEqual(report.verification_baseline.runs.map(r=>r.scenario),config.verification_scenarios);
    assert.deepEqual(report.baseline.runs.map(r=>r.scenario),config.tuning_scenarios);
    const expectedBaseline={...preRegistered.resolved_solver_defaults,...config.parameters};assert.deepEqual(report.baseline_parameters,expectedBaseline);
    if(!report.tuned_candidate_accepted)assert.deepEqual(descriptor.parameters,expectedBaseline,'No improvement retains the actual feasible baseline');
    assert.ok(report.evaluation&&report.evaluation_hash);assert.deepEqual(descriptor.evaluation,report.evaluation,'Published and reported preference contracts agree');assert.equal(descriptor.evaluation_hash,report.evaluation_hash);
    for(const key of ['recipe','weights','scales','energy_model'])assert.deepEqual(report.evaluation[key],config.evaluation[key],'Predeclared '+key+' must not change during search');
    const metricAudits={};
    for(const key of ['baseline','verification_baseline','verification_candidate','test']){assert.equal(report[key].evaluation_hash,report.evaluation_hash);metricAudits[key]=auditEvaluation(report[key],report.evaluation,key);}
    for(const trial of report.trials)if(trial.tuning_evaluation){assert.equal(trial.tuning_evaluation.evaluation_hash,report.evaluation_hash);metricAudits[`trial${trial.index}`]=auditEvaluation(trial.tuning_evaluation,report.evaluation,`trial${trial.index}`);assert.equal(trial.score,trial.tuning_evaluation.score);}
    const rawReport=resolve(dirname(evidencePath),`native-goal-tuning-${recipe}-raw-report.json`);await writeFile(rawReport,reportBytes);
    const replay=JSON.parse(execFileSync(python,['-I','-B','-c',independentReplay,resolve(artifact,'bundle'),resolve(fixtures,'F02_existing_linear_4x2.json'),rawReport],
      {cwd:workspace,windowsHide:true,encoding:'utf8',maxBuffer:16*1024*1024,env:{...process.env,PYTHONPATH:'',PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1'}}));
    assert.equal(replay.status,'PASS');assert.equal(replay.project_imports,false);
    const check={recipe,status:'PASS',workspace,sessionId,artifact_id:result.artifact_id,artifact_path:artifact,job:evidence.jobs[`${recipe}:tune`],
      raw_report:rawReport,raw_report_sha256:hash(reportBytes),chosen_parameters:descriptor.parameters,selection:report.selection,tuned_candidate_accepted:report.tuned_candidate_accepted,
      baseline:report.baseline,best:report.best,last:report.last,failed_candidates:report.failed_candidates,verification_baseline:report.verification_baseline,
      verification_candidate:report.verification_candidate,test:report.test,evaluation:report.evaluation,evaluation_hash:report.evaluation_hash,independent_metric_audits:metricAudits,portable_independent_replay:replay};
    for(const[group,preset]of [['sessions','controller'],['modelSessions','model']]){
      const records=(await rpc('session/events',{request:{address:{kind:'session',sessionId:evidence[group][recipe]},maxMessages:10000}})).records;
      assert.ok(!records.some(row=>row.event.type.startsWith('turn/')),`Fixed ${preset} acceptance must not open an LLM turn`);
    }
    assert.ok(!Object.values((await json(resolve(workspace,'manifest.json'))).artifacts).some(row=>['expert_dataset','imitation_policy','optimized_policy'].includes(row.kind)),'MPC branch must stop after tuning');
    const previous=evidence.checks.findIndex(row=>row.recipe===recipe);if(previous<0)evidence.checks.push(check);else evidence.checks[previous]=check;await save();
  }
  assert.deepEqual(evidence.checks.map(r=>r.recipe).sort(),recipes.toSorted());assert.equal(evidence.rejections.length,4);evidence.status='PASS';evidence.finished_at=new Date().toISOString();
}catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:evidencePath,recipes:evidence.checks.map(r=>({recipe:r.recipe,selection:r.selection,artifact_id:r.artifact_id})),admissions:evidence.admissions.map(r=>({recipe:r.recipe,operation:r.operation,jobId:r.jobId,attemptId:r.attemptId})),confirmation_source:'test_fixture'}));}
