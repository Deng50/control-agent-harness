/** S09 L01: actual external NumPy MPC expert. No Host/service lifecycle changes. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {mkdir,readFile,readdir,writeFile} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';
import {confirmationRequirements} from '../harness-plugin/src/case-requirements.mjs';
import {command,connectHarness,prepareSmokeWorkspace,project,writeSmokeEvidence} from './harness-rpc.mjs';

export const runRoot=resolve(project,'test-output/v2-development/20261117T002006');
export const hash=value=>createHash('sha256').update(value).digest('hex');
export const json=path=>readFile(path,'utf8').then(JSON.parse);
export const python=process.env.CONTROL_PYTHON??resolve(project,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
export function py(code,args=[],env={}){return execFileSync(python,['-I','-B','-c',code,...args],{encoding:'utf8',windowsHide:true,maxBuffer:32*1024*1024,
  env:{...process.env,PYTHONUTF8:'1',PYTHONDONTWRITEBYTECODE:'1',...env}});}
export function near(a,b,label,atol=1e-10,rtol=1e-9){if(Array.isArray(b)){assert.ok(Array.isArray(a),label);assert.equal(a.length,b.length,label);b.forEach((v,k)=>near(a[k],v,`${label}[${k}]`,atol,rtol));}
  else assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=atol+rtol*Math.abs(b),`${label}: ${a} vs ${b}`);}
export async function immutableFile(path,bytes){await mkdir(dirname(path),{recursive:true});try{assert.equal(hash(await readFile(path)),hash(bytes),'Existing source bytes changed: '+path);}catch(error){if(error.code!=='ENOENT')throw error;await writeFile(path,bytes,{flag:'wx'});}return path;}
export async function createObserver(values,group,preRegistered){
  const destination=resolve(values['workspace-root']),evidencePath=resolve(values.evidence),inside=relative(resolve(project,'test-output'),destination);
  assert.ok(inside&&!isAbsolute(inside)&&inside!=='..'&&!inside.startsWith(`..${sep}`),'Use an isolated test-output workspace root');
  let evidence;
  if(values.resume){evidence=await json(evidencePath);assert.equal(evidence.destination,destination);assert.deepEqual(evidence.preRegistered,preRegistered);
    if(evidence.error){(evidence.observer_history??=[]).push({status:evidence.status,error:evidence.error,finished_at:evidence.finished_at,resumed_at:new Date().toISOString()});delete evidence.error;delete evidence.finished_at;}}
  else{try{await readFile(evidencePath);throw new Error('Evidence exists; preserve it and use --resume to observe saved handles');}catch(error){if(error.code!=='ENOENT')throw error;}
    evidence={schema_version:1,stage:'S09',group,status:'PREPARED',destination,preRegistered,started_at:new Date().toISOString(),confirmation_source:'test_fixture',sessions:{},jobs:{},admissions:[],confirmations:[],rejections:[],checks:[]};}
  const save=()=>writeSmokeEvidence(evidencePath,evidence);await save();
  let rpc;
  const connect=async()=>rpc??=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  const workspace=async branch=>{
    const workspace=resolve(destination,branch);await prepareSmokeWorkspace(workspace);
    const markerPath=resolve(workspace,'.case-plans/test-fixture.json');let marker;
    try{marker=await json(markerPath);}catch(error){if(error.code!=='ENOENT')throw error;
      assert.ok(!evidence.admissions.some(row=>row.branch===branch),'Marker disappeared after an admission; never relabel prior native confirmations');
      // A resumed evidence file is not authority to relabel untracked old jobs.
      let existing=[];try{existing=await readdir(resolve(workspace,'.control-jobs'));}catch(e){if(e.code!=='ENOENT')throw e;}
      assert.ok(!existing.some(name=>/^[0-9a-f-]{36}$/.test(name)),'Existing jobs without marker require manual evidence reconciliation');
      marker={kind:'control-case-plan-test-fixture-v1',workspace,confirmation_source:'test_fixture',synthetic:true};await writeSmokeEvidence(markerPath,marker);}
    assert.equal(marker.kind,'control-case-plan-test-fixture-v1');assert.equal(resolve(marker.workspace),workspace);assert.equal(marker.confirmation_source,'test_fixture');
    return workspace;
  };
  const session=async(branch,role)=>{const ws=await workspace(branch),key=`${branch}:${role}`;await connect();
    if(!evidence.sessions[key]){const s=await rpc('session/create',{request:{cwd:ws,agentPreset:`control-${role}`}});assert.equal(s.agentPreset,`control-${role}`);evidence.sessions[key]=s.sessionId;await save();}return evidence.sessions[key];};
  const stage=async(branch,key,role,operation,inputs={},config={})=>{
    const ws=await workspace(branch),sessionId=await session(branch,role),request={agent:role,operation,workspace:ws,inputs,config},requestHash=hash(JSON.stringify(request)),id=`${branch}:${key}`;
    let row=evidence.jobs[id];
    if(row){assert.equal(row.requestHash,requestHash,'Never mutate an admitted request');assert.ok(row.jobId,'Uncertain admission: reconcile saved history; never retry mutation blindly');if(row.result)return row.result;}
    else{row=evidence.jobs[id]={operation,requestHash,status:'ADMISSION_PENDING',sessionId};await save();const admission=await command(rpc,sessionId,`/control ${JSON.stringify(request)}`);
      Object.assign(row,{jobId:admission.jobId,attemptId:admission.attemptId,status:'RUNNING'});evidence.admissions.push({branch,key,operation,sessionId,jobId:row.jobId,attemptId:row.attemptId});await save();}
    const deadline=Date.now()+20*60*1000;
    while(Date.now()<deadline){const state=await command(rpc,sessionId,`/control-status ${JSON.stringify({workspace:ws,attemptId:row.attemptId})}`);
      if(['completed','failed','killed'].includes(state.record.state)){row.status=state.record.state;row.record=state.record;await save();assert.equal(row.status,'completed',JSON.stringify(state));row.result=state.record.result;assert.ok(row.result);await save();return row.result;}
      assert.notEqual(state.record.state,'interrupted','Reconcile the original attempt; do not resubmit');await new Promise(done=>setTimeout(done,350));}
    throw new Error('Existing job still active; saved handles retained; do not resubmit');
  };
  const confirm=async(branch,key,role,operation,inputs,config)=>{
    const ws=await workspace(branch),sessionId=await session(branch,role),requirements=await confirmationRequirements({operation,workspace:ws,inputs,config}),refs={};
    for(const required of requirements){const plan=await stage(branch,`${key}-plan-${required.scope}`,role,'propose_case_plan',{}, {scope:required.scope,payload:required.expectedPayload,bindings:required.bindings});
      refs[required.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};const id=`${branch}:${key}:${required.scope}`;
      let row=evidence.confirmations.find(item=>item.key===id);
      if(row){assert.equal(row.sessionId,sessionId);assert.deepEqual(row.ref,refs[required.scope]);const status=await command(rpc,sessionId,`/control-plan-status ${JSON.stringify({workspace:ws,...refs[required.scope]})}`);assert.equal(status.confirmation_source,'test_fixture');assert.equal(status.confirmation_state,'confirmed');}
      else{const result=await command(rpc,sessionId,`/control-confirm ${JSON.stringify({workspace:ws,...refs[required.scope]})}`);assert.equal(result.confirmation_source,'test_fixture');row={key:id,branch,scope:required.scope,sessionId,ref:refs[required.scope],confirmation_source:result.confirmation_source};evidence.confirmations.push(row);await save();}}
    return refs;
  };
  const snapshot=async(branch,role)=>{const ws=await workspace(branch),sessionId=await session(branch,role);let attempts=[];
    try{attempts=(await readdir(resolve(ws,'.control-jobs'))).filter(x=>/^[0-9a-f-]{36}$/.test(x)).sort();}catch(e){if(e.code!=='ENOENT')throw e;}
    const history=await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace:ws})}`),events=(await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:10000}})).records;
    return{attempts,history:history.attempts.map(x=>x.attemptId).sort(),jobs:events.filter(x=>x.event.type==='job/registered').map(x=>x.event.data.jobId),confirmations:events.filter(x=>x.event.type==='command/run'&&x.event.data.name==='control-confirm').length};};
  const deny=async(branch,label,role,operation,inputs,config)=>{if(evidence.rejections.some(row=>row.label===label))return;const ws=await workspace(branch),sessionId=await session(branch,role),before=await snapshot(branch,role);
    const response=await rpc('commands/execute',{agentId:sessionId,line:`/control ${JSON.stringify({agent:role,operation,workspace:ws,inputs,config})}`,submittedAttachments:[]});assert.equal(response.result.kind,'error',label+' was admitted');assert.match(response.result.text,/plan|confirm|version|review/i);
    const after=await snapshot(branch,role);assert.deepEqual(after,before,label+' created job/history/confirmation');evidence.rejections.push({branch,label,status:'PASS_ZERO_ADMISSION',before,after,message:response.result.text,newJobs:0,newHistory:0});await save();};
  return{destination,evidencePath,evidence,save,connect,workspace,session,stage,confirm,deny,snapshot,get rpc(){return rpc;}};
}

// Fixed QP mathematics extracted from the independent S00 F05 calibration.
// Provenance SHA of that source is captured before any native admission below.
const expertSource=String.raw`import numpy as np

def _qp(model,p,x,target,d,previous):
 A,B,C,E=[np.asarray(model[k],dtype=np.float64) for k in ('A','B','C','E')]
 nx,nu,ny=A.shape[0],B.shape[1],C.shape[0];h=int(p['horizon']);width=h*nu;i=model['interface']
 ul=np.array([v['lower'] for v in i['controls']]);uh=np.array([v['upper'] for v in i['controls']]);xl=np.array([v['lower'] for v in i['states']]);xh=np.array([v['upper'] for v in i['states']]);yl=np.array([v['lower'] for v in i['outputs']]);yh=np.array([v['upper'] for v in i['outputs']])
 ur,yr=uh-ul,yh-yl;H=np.zeros((width,width));g=np.zeros(width);G=[];limits=[];c=np.array(x);M=np.zeros((nx,width));P=np.zeros((nu,width));pc=np.array(previous)
 for k in range(h):
  S=np.zeros((nu,width));S[:,k*nu:(k+1)*nu]=np.eye(nu);c=A@c+E@d;M=A@M+B@S;oc,OM=C@c,C@M
  for w,F,b in ((p['q'],OM/yr[:,None],(oc-target)/yr),(p['r'],S/ur[:,None],np.zeros(nu)),(p['s'],(S-P)/ur[:,None],-pc/ur)):
   H+=2*w*(F.T@F);g+=2*w*(F.T@b)
  for F,b,lo,hi in ((M,c,xl,xh),(OM,oc,yl,yh),(S,np.zeros(nu),ul,uh),(S-P,-pc,np.full(nu,-.25),np.full(nu,.25))):G.extend([F,-F]);limits.extend([hi-b,b-lo])
  P,pc=S,np.zeros(nu)
 return H,g,np.vstack(G),np.concatenate(limits)

def _solve(H,g,G,b,initial):
 z=np.array(initial,dtype=np.float64,copy=True);active=[]
 if np.max(G@z-b)>1e-9:return None,{'failure':'initial_infeasible'}
 for iteration in range(1,201):
  grad=H@z+g
  if active:
   W=G[active];K=np.block([[H,W.T],[W,np.zeros((len(active),len(active)))]]);solution=np.linalg.lstsq(K,np.r_[-grad,np.zeros(len(active))],rcond=None)[0];direction,multipliers=solution[:len(z)],solution[len(z):]
  else:direction=np.linalg.solve(H,-grad);multipliers=np.empty(0)
  if np.linalg.norm(direction,ord=np.inf)<1e-11:
   if not active or np.min(multipliers)>=-1e-9:
    stationarity=grad+(G[active].T@multipliers if active else 0);violation=float(max(0.,np.max(G@z-b)));residual=float(np.max(np.abs(stationarity)))
    if violation>1e-8 or residual>1e-8:return None,{'failure':'kkt_failed'}
    return z,{'iterations':iteration,'violation':violation,'stationarity_inf':residual}
   del active[int(np.argmin(multipliers))];continue
  step,blocker=1.,None;gd=G@direction;slack=b-G@z
  for index in range(len(b)):
   if index in active or gd[index]<=1e-13:continue
   limit=max(0.,float(slack[index]))/float(gd[index])
   if limit<step-1e-13:step,blocker=limit,index
  z+=step*direction
  if blocker is not None and np.linalg.matrix_rank(G[active+[blocker]],tol=1e-10)==len(active)+1:active.append(blocker)
 return None,{'failure':'iteration_guard'}

def expert_step(observation,reference,disturbance,previous_action,memory,parameters):
 p=parameters;H,g,G,b=_qp(p['model'],p['controller'],observation,reference,disturbance,previous_action)
 z,diagnostic=_solve(H,g,G,b,np.tile(previous_action,int(p['controller']['horizon'])))
 if z is None:return np.empty(0,dtype=np.float64),dict(memory),'qp_failed'
 return z[:len(previous_action)].copy(),{'calls':int(memory['calls'])+1,'solver':diagnostic},'ok'
`;
const plantSource=String.raw`import numpy as np
def transition(x,u,d,dt,parameters):
 if abs(dt-parameters['sample_time'])>1e-12:raise ValueError('Plant dt changed')
 return np.asarray(parameters['A'],dtype=np.float64)@x+np.asarray(parameters['B'],dtype=np.float64)@u+np.asarray(parameters['E'],dtype=np.float64)@d
`;

const readArrays=String.raw`import json,sys,numpy as np
with np.load(sys.argv[1],allow_pickle=False) as a:print(json.dumps({key:a[key].tolist() for key in a.files},allow_nan=False))`;
// Deliberately different numerical solver and uncondensed recurrence from the
// supplied active-set QP. Only observed rows are replayed; labels are never replaced.
const independentActions=String.raw`import json,sys,numpy as np
from scipy.optimize import minimize,nnls
from pathlib import Path
fixture=json.loads(Path(sys.argv[1]).read_text());data=json.loads(Path(sys.argv[2]).read_text());substeps=int(sys.argv[3]);model=fixture['nominal'];i=model['interface'];A,B,C,E=[np.asarray(model[k]) for k in ('A','B','C','E')];h=4;nu=2
ul=np.array([v['lower'] for v in i['controls']]);uh=np.array([v['upper'] for v in i['controls']]);xl=np.array([v['lower'] for v in i['states']]);xh=np.array([v['upper'] for v in i['states']]);yl=np.array([v['lower'] for v in i['outputs']]);yh=np.array([v['upper'] for v in i['outputs']])
def replay(flat,x,r,d,prev):
 cost=0.;margins=[]
 for u in flat.reshape(h,nu):
  for tick in range(substeps):x=A@x+B@u+E@d
  y=C@x;cost+=np.sum(((y-r)/(yh-yl))**2)+.015*np.sum((u/(uh-ul))**2)+.01*np.sum(((u-prev)/(uh-ul))**2)
  margins.extend([x-xl,xh-x,y-yl,yh-y,u-prev+.25,.25-u+prev]);prev=u
 return float(cost),np.concatenate(margins)
ids=np.asarray(data['trajectory_id']);checks=[];maximum=0.
for trajectory in np.unique(ids):
 rows=np.flatnonzero(ids==trajectory)
 for row in sorted(set([int(rows[0]),int(rows[len(rows)//2]),int(rows[-1])])):
  x=np.array(data['x'][row]);r=np.array(data['reference'][row]);d=np.array(data['d'][row]);prev=np.array(data['previous_u'][row]);initial=np.tile(prev,h)
  # The independent uncondensed recurrence is exactly quadratic in the
  # horizon actions and its margins are affine. Unit-basis identities extract
  # their exact derivatives without a tiny finite-difference step or any
  # production QP condensation/gradient code. The original numerical-gradient
  # ftol=1e-12 stopped too early for a weakly curved direction (row 540).
  width=len(initial);basis=np.eye(width);zero=np.zeros(width);j0,m0=replay(zero,x,r,d,prev)
  plus=[replay(e,x,r,d,prev) for e in basis];minus=[replay(-e,x,r,d,prev) for e in basis]
  gradient0=np.array([(p[0]-m[0])/2 for p,m in zip(plus,minus)]);H=np.zeros((width,width))
  for a in range(width):
   H[a,a]=plus[a][0]+minus[a][0]-2*j0
   for b in range(a):H[a,b]=H[b,a]=replay(basis[a]+basis[b],x,r,d,prev)[0]-plus[a][0]-plus[b][0]+j0
  M=np.column_stack([(p[1]-m[1])/2 for p,m in zip(plus,minus)])
  assert float(np.linalg.eigvalsh(H).min())>0,'Independent cost is not strictly convex'
  result=minimize(lambda z:replay(z,x,r,d,prev)[0],initial,jac=lambda z:H@z+gradient0,method='SLSQP',bounds=list(zip(ul,uh))*h,
   constraints=[{'type':'ineq','fun':lambda z:replay(z,x,r,d,prev)[1],'jac':lambda z:M}],options={'maxiter':250,'ftol':1e-14})
  assert result.success,result.message
  margins=replay(result.x,x,r,d,prev)[1];all_margins=np.r_[margins,result.x-np.tile(ul,h),np.tile(uh,h)-result.x];all_jac=np.vstack([M,np.eye(width),-np.eye(width)])
  active=all_margins<1e-7;gradient=H@result.x+gradient0;multipliers,residual=nnls(all_jac[active].T,gradient) if np.any(active) else (np.zeros(0),float(np.linalg.norm(gradient)))
  stationarity=gradient-(all_jac[active].T@multipliers if np.any(active) else 0);kkt_inf=float(np.max(np.abs(stationarity)))
  assert float(all_margins.min())>=-1e-9 and kkt_inf<1e-7,('Independent reference KKT',row,float(all_margins.min()),kkt_inf)
  error=float(np.max(np.abs(result.x[:nu]-np.array(data['u'][row]))));maximum=max(maximum,error)
  assert error<2e-5,(int(trajectory),row,error)
  checks.append({'trajectory_id':int(trajectory),'row':row,'first_action_max_error':error,'independent_solver':'SciPy SLSQP; exact quadratic/affine unit-basis derivatives of independent recurrence',
   'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'ftol':1e-14,'min_margin':float(all_margins.min()),'kkt_stationarity_inf':kkt_inf,
   'kkt_complementarity_inf':float(np.max(np.abs(multipliers*all_margins[active]))) if len(multipliers) else 0.,'minimum_hessian_eigenvalue':float(np.linalg.eigvalsh(H).min()),
   'quadratic_identity_error':abs(float(result.fun)-(j0+float(gradient0@result.x)+float(result.x@H@result.x)/2))})
print(json.dumps({'status':'PASS','sampled_decisions':len(checks),'max_action_error':maximum,'action_tolerance_unchanged':2e-5,'checks':checks,'harness_imports':False},allow_nan=False))`;

function matrixProduct(a,b){return a.map(row=>b[0].map((_,j)=>row.reduce((sum,v,k)=>sum+v*b[k][j],0)));}
function composed(model,n){const result=structuredClone(model);if(n===1)return result;
  const A=matrixProduct(model.A,model.A),AB=matrixProduct(model.A,model.B),AE=matrixProduct(model.A,model.E);
  result.A=A;result.B=AB.map((r,k)=>r.map((v,j)=>v+model.B[k][j]));result.E=AE.map((r,k)=>r.map((v,j)=>v+model.E[k][j]));result.interface.sample_time*=n;return result;}

export async function mainPython(){
  const {values}=parseArgs({options:{'workspace-root':{type:'string'},evidence:{type:'string'},fixtures:{type:'string'},'web-log':{type:'string'},resume:{type:'boolean'},'prepare-only':{type:'boolean'},help:{type:'boolean',short:'h'}}});
  if(values.help){console.log('node scripts/smoke-external-expert-python.mjs --workspace-root <isolated S09 parent> --evidence <json> [--fixtures <frozen fixtures>] [--web-log <Host log>] [--resume] [--prepare-only]');return;}
  assert.ok(values['workspace-root']&&values.evidence);
  const fixtures=resolve(values.fixtures??resolve(runRoot,'fixtures')),f02Bytes=await readFile(resolve(fixtures,'F02_existing_linear_4x2.json')),f05Bytes=await readFile(resolve(fixtures,'F05_expert_and_tradeoff_candidates.json'));
  assert.equal(hash(f02Bytes),'ac6364b26915a81abc790a88c6346e2fe91fdd62950898ecac0cc0c67f6ae7d9');assert.equal(hash(f05Bytes),'66493849a0fa57eaba303e576077d40e8ad5779118617fa0a34d982231a73d2b');
  const f02=JSON.parse(f02Bytes),f05=JSON.parse(f05Bytes),attributionPath=resolve(runRoot,'S00/correct_f05_action_normalization.py'),attributionBytes=await readFile(attributionPath);
  const preRegistered={fixture_sha256:hash(f02Bytes),expert_cases_sha256:hash(f05Bytes),expert_source_sha256:hash(expertSource),plant_source_sha256:hash(plantSource),
    source_attribution:{path:attributionPath,sha256:hash(attributionBytes),functions:['make_qp','solve_feasible_qp'],scope:'independent frozen fixed reference QP; extracted math only, no internal MPC/controller artifact'},
    branches:{f05:{trajectories:32,steps:40,split_counts:{train:20,validation:6,test:6},substeps:1},'period-two':{trajectories:5,steps:3,split_counts:{train:3,validation:1,test:1},substeps:2}},
    weights:f05.source_expert_controller.config.parameters,action_delta_bounds:[[-.25,.25],[-.25,.25]],independent_action_tolerance:2e-5,
    scope:'External explicit-memory NumPy QP calls + ordinary Python plant. No internal controller design, imitation, PGAC, or LLM.'};
  const obs=await createObserver(values,'L01',preRegistered),{evidence,save}=obs;
  try{
    const sourceDir=resolve(dirname(obs.evidencePath),'external-expert-python-sources');await immutableFile(resolve(sourceDir,'f05_external_qp.py'),expertSource);await immutableFile(resolve(sourceDir,'f02_reference_step.py'),plantSource);
    if(values['prepare-only']){evidence.status='PREPARED';await save();return;}
    evidence.status='RUNNING';await save();
    for(const branch of ['f05','period-two']){
      const workspace=await obs.workspace(branch),substeps=branch==='f05'?1:2,period=.1*substeps;
      const localExpert=resolve(workspace,'source-inputs/f05_external_qp.py'),localPlant=resolve(workspace,'source-inputs/f02_reference_step.py');
      await immutableFile(localExpert,expertSource);await immutableFile(localPlant,plantSource);
      const plantMaterial=await obs.stage(branch,'material-plant','model','register_material',{local_path:localPlant});
      const plant=await obs.stage(branch,'plant','model','register_plant',{material:plantMaterial.artifact_id},{adapter:{schema_version:1,backend:'python_callable',source_file:'source/f02_reference_step.py',entrypoint:'transition',interface:f02.reference.interface,
        parameters:{A:f02.reference.A,B:f02.reference.B,E:f02.reference.E,sample_time:.1},source_semantics:'transition',C:f02.reference.C}});
      const expertMaterial=await obs.stage(branch,'material-expert','lightweight','register_material',{local_path:localExpert});
      const physical=f02.reference.interface,spec={schema_version:1,backend:'python_callable',source_file:'source/f05_external_qp.py',entrypoint:'expert_step',interface:physical,time_unit:'s',control_period:period,
        parameters:{model:composed(f02.nominal,substeps),controller:f05.source_expert_controller.config.parameters},initial_memory:{calls:0},observations:physical.states.map((v,index)=>({name:v.name,source:'state',index,plant_name:v.name,unit:v.unit,availability:'simulation_only',delay_steps:0})),
        reference_available:true,disturbance_available:true,previous_action_available:true,action_mapping:physical.controls.map((v,index)=>({action_index:index,plant_control_index:index,plant_control_name:v.name,unit:v.unit})),failure_policy:'stop',input_rate_bounds:[[-.25,.25],[-.25,.25]]};
      const expert=await obs.stage(branch,'expert','lightweight','register_expert',{material:expertMaterial.artifact_id},{adapter:spec});
      const selected=branch==='f05'?f05.collection_scenarios:f05.collection_scenarios.slice(0,5).map((s,k)=>({...s,steps:3,split:k<3?'train':k===3?'validation':'test'}));
      const scenarios=selected.map(s=>({name:s.id,initial_state:s.initial_state,reference:s.reference,disturbance:s.disturbance,steps:s.steps,initial_previous_action:[0,0],split:s.split}));
      const student_features=[...physical.states.map(v=>'observation::'+v.name),...physical.outputs.map(v=>'reference::'+v.name),...physical.disturbances.map(v=>'disturbance::'+v.name),...physical.controls.map(v=>'previous_action::'+v.name)];
      const inputs={expert:expert.artifact_id,plant:plant.artifact_id},config={scenarios,seed:f05.seed,student_features,scaler:'standard'};
      await obs.deny(branch,`${branch}:missing-confirmation`,'lightweight','collect_external_expert',inputs,config);
      const refs=await obs.confirm(branch,'collection','lightweight','collect_external_expert',inputs,config);
      const changed=structuredClone(config);changed.scenarios[0].steps++;
      await obs.deny(branch,`${branch}:changed-sampling-config`,'lightweight','collect_external_expert',inputs,{...changed,plan_refs:refs});
      const result=await obs.stage(branch,'collect','lightweight','collect_external_expert',inputs,{...config,plan_refs:refs});assert.equal(result.kind,'external_expert_dataset');assert.equal(result.status,'verified');
      const root=result.artifact_path,meta=await json(resolve(root,'artifact.json')),dataset=await json(resolve(root,'dataset.json')),split=await json(resolve(root,'split.json')),report=await json(resolve(root,'report.json')),diagnostics=await json(resolve(root,'diagnostics.json'));
      assert.equal(dataset.input_dim,9);assert.equal(dataset.output_dim,2);assert.equal(dataset.plant_dimensions_inferred,false);assert.equal(dataset.diagnostic_memory_is_feature,false);assert.equal(dataset.deployment_scope,'simulation_only');
      const arr=JSON.parse(py(readArrays,[resolve(root,'expert.npz')]));assert.equal(arr.X.length,branch==='f05'?1280:15);assert.equal(arr.y.length,arr.X.length);assert.equal(diagnostics.length,arr.X.length);
      const groups={train:[],validation:[],test:[]};scenarios.forEach((s,k)=>groups[s.split].push(k));assert.deepEqual(split.groups,groups);
      const rowProduct=(matrix,vector)=>matrix.map(row=>row.reduce((sum,v,k)=>sum+v*vector[k],0));let previousId=null,previousAction,previousNext;
      for(let k=0;k<arr.X.length;k++){
        const id=arr.trajectory_id[k],start=id!==previousId,scenario=scenarios[id];if(start){previousAction=[0,0];previousNext=scenario.initial_state;}
        near(arr.x[k],previousNext,'recorded pre-state');near(arr.previous_u[k],previousAction,'actual previous command');near(arr.reference[k],scenario.reference,'held reference');near(arr.d[k],scenario.disturbance,'held disturbance');
        let expected=arr.x[k];for(let tick=0;tick<substeps;tick++){const ax=rowProduct(f02.reference.A,expected),bu=rowProduct(f02.reference.B,arr.u[k]),ed=rowProduct(f02.reference.E,arr.d[k]);expected=ax.map((v,j)=>v+bu[j]+ed[j]);}
        near(arr.next_x[k],expected,'independent scalar physical recurrence');near(arr.plant_y[k],rowProduct(f02.reference.C,arr.x[k]),'initial physical output');near(arr.next_y[k],rowProduct(f02.reference.C,expected),'next physical output');
        near(arr.X[k],[...arr.x[k],...arr.reference[k],...arr.d[k],...arr.previous_u[k]],'causal 9 inputs');near(arr.y[k],arr.u[k],'expert current labels');
        assert.equal(diagnostics[k].memory_before.calls,start?0:diagnostics[k-1].memory_after.calls);assert.equal(diagnostics[k].memory_after.calls,diagnostics[k].memory_before.calls+1);
        assert.ok(diagnostics[k].memory_after.solver.violation<=1e-8);assert.ok(diagnostics[k].memory_after.solver.stationarity_inf<=1e-8);assert.equal(diagnostics[k].status,'ok');previousId=id;previousAction=arr.u[k];previousNext=arr.next_x[k];
      }
      const recorded=resolve(workspace,'independent-recorded-rows.json');await immutableFile(recorded,JSON.stringify(arr));const independent=JSON.parse(py(independentActions,[resolve(fixtures,'F02_existing_linear_4x2.json'),recorded,String(substeps)]));assert.equal(independent.status,'PASS');
      for(const [name,expected]of Object.entries(meta.files))assert.equal(hash(await readFile(resolve(root,name))),expected);
      const context=await json(resolve(root,'physical_context.json'));assert.equal(context.plant_dt,.1);assert.equal(context.control_period,period);assert.equal(context.plant_substeps,substeps);
      const manifest=await json(resolve(workspace,'manifest.json'));assert.ok(!Object.values(manifest.artifacts).some(r=>['controller','controller_design','expert_dataset','imitation_policy','optimized_policy'].includes(r.kind)));
      const check={branch,status:'PASS',workspace,sessionId:evidence.sessions[`${branch}:lightweight`],artifact_id:result.artifact_id,artifact_path:root,expert_id:expert.artifact_id,plant_id:plant.artifact_id,rows:arr.X.length,dimensions:[9,2],split:split.groups,
        expert_npz_sha256:hash(await readFile(resolve(root,'expert.npz'))),metadata_sha256:hash(await readFile(resolve(root,'artifact.json'))),physical_scalar_rows_checked:arr.X.length,independent_action_replay:independent,
        control_period:period,plant_dt:.1,plant_substeps:substeps,training:'not_run',confirmation_sources:report.confirmation_sources};
      assert.ok(check.confirmation_sources.length===2&&check.confirmation_sources.every(r=>r.source==='test_fixture'||r.confirmation_source==='test_fixture'));
      const prior=evidence.checks.findIndex(r=>r.branch===branch);if(prior<0)evidence.checks.push(check);else evidence.checks[prior]=check;await save();
    }
    assert.equal(evidence.checks.length,2);assert.equal(evidence.rejections.length,4);evidence.status='PASS';evidence.error=null;evidence.finished_at=new Date().toISOString();
  }catch(error){evidence.status='FAIL';evidence.error=error.stack??String(error);throw error;}
  finally{await save();console.log(JSON.stringify({status:evidence.status,evidence:obs.evidencePath,jobs:evidence.admissions.length,checks:evidence.checks.map(r=>({branch:r.branch,artifact_id:r.artifact_id,rows:r.rows}))}));}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await mainPython();
