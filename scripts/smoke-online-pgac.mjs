/** Native Python Plant replay, interrupted worker recovery and deployment gate. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {mkdir, readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {command, connectHarness, prepareSmokeWorkspace, project, writeSmokeEvidence} from './harness-rpc.mjs';
import {artifact, hash, observer, runPython} from './pgac-native-observer.mjs';

const {values} = parseArgs({options:{'web-log':{type:'string'},tracking:{type:'boolean',default:false}}});
const directory = resolve(project,'test-output','online-pgac-native',randomUUID());
const workspace = resolve(directory,'workspace');
await mkdir(directory,{recursive:true});
await prepareSmokeWorkspace(workspace);
const path = resolve(directory,'result.json');
// Provision the native worker on the empty leaf before creating artifact children.
// On Windows a new restricted worker SID may be granted only when its first job starts.
const rpc=await connectHarness({logPath:values['web-log'],connectionClose:true});
const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});
const sessionId=created.sessionId;
const startup=await command(rpc,sessionId,'/control '+JSON.stringify({agent:'lightweight',workspace,operation:'inspect_workspace',inputs:{},config:{}}));
const startupDeadline=Date.now()+120000;
for (;;) {
  const {record}=await command(rpc,sessionId,'/control-status '+JSON.stringify({workspace,attemptId:startup.attemptId}));
  if(record.state==='completed') break;
  assert.ok(!['failed','killed','interrupted'].includes(record.state),record.detail);
  assert.ok(Date.now()<startupDeadline,'Workspace preparation timed out');
  await new Promise(done=>setTimeout(done,150));
}
const fixture = runPython(String.raw`import json,shutil,sys
from pathlib import Path
from control_harness.artifacts import atomic_json
sys.path.insert(0,str(Path.cwd()/'tests'))
from test_pgac_gradients import RestrictedPgacTrainingTests as Fixture
Fixture.setUpClass()
try:
    root=Path(sys.argv[1])
    shutil.copytree(Fixture.base/'prepared',root,dirs_exist_ok=True)
    atomic_json(root/'.case-plans/test-fixture.json',{'kind':'control-case-plan-test-fixture-v1','workspace':str(root)})
    cfg=dict(Fixture.config,epochs=8,horizon=3,batch_size=4,
             replay=dict(mode='online',capacity_steps=256,refresh_every=1,updates_per_epoch=12,student_fraction=.5))
    print(json.dumps(dict(inputs=dict(policy=Fixture.policy,model=Fixture.model,validation=Fixture.validation),config=cfg)))
finally:
    Fixture.tearDownClass()`,[workspace]);

const e={phase:'online_pgac_worker_recovery',status:'RUNNING',workspace,sessions:{online:sessionId},jobs:{},admissions:[],
  confirmations:[],confirmation_reuses:[],phase_requests:{},rejections:[],historical_phase_checks:[]};
const markerPath=resolve(workspace,'.case-plans','test-fixture.json');
const item={key:'online',workspace,actual:{},marker:{path:markerPath,sha256:hash(await readFile(markerPath))}};
const o=observer(rpc,e,path,item);
try {
  const {inputs,config}=fixture;
  if (values.tracking) Object.assign(config.original_evaluation.evaluation, {
    primary_metric:'tracking_rmse', min_relative_improvement:.01, max_worst_case_relative_regression:.05,
    max_action_rms_relative_increase:.15, max_increment_rms_relative_increase:.25,
  });
  const refs=await o.phaseRefs('training','interrupted','optimize_external_policy',inputs,config,['signals','training']);
  const settings={...config,plan_refs:refs};
  const job=await o.admit('interrupted','optimize_external_policy',inputs,settings,{expected:'killed'});
  const until=Date.now()+5*60*1000;
  for (;;) {
    const state=await o.status(job);
    const record=o.assertRecord(job,state);
    assert.ok(!['completed','killed','failed','interrupted'].includes(record.state),'Worker ended before interruption');
    const progress=record.progress;
    if (['pgac_student_episode_committed','restricted_pgac_training'].includes(progress?.stage)) {
      e.interrupted_after=progress;
      const stopped=await rpc('job/kill',{request:{sessionId:await o.session(),jobId:job.jobId}});
      assert.equal(stopped.outcome,'requested');
      break;
    }
    assert.ok(Date.now()<until,'No committed replay observed');
    await new Promise(done=>setTimeout(done,80));
  }
  await o.wait(job,'killed');
  const receipt=await command(rpc,await o.session(),'/control-recover '+JSON.stringify({workspace,attemptId:job.attemptId,mode:'resume'}));
  assert.equal(receipt.recoveredFrom,job.attemptId);
  const resumedStatus=await command(rpc,sessionId,'/control-status '+JSON.stringify({workspace,attemptId:receipt.attemptId}));
  const recovered={key:o.key('resumed'),jobId:receipt.jobId,attemptId:receipt.attemptId,sessionId,
    request:{agent:'lightweight',...resumedStatus.record.request},expected:'completed'};
  e.jobs[recovered.key]=recovered;
  const resumed=await o.wait(recovered);
  const continuous=await o.stage('continuous','optimize_external_policy',inputs,settings);
  const left=await artifact(workspace,resumed.artifact_id);
  const right=await artifact(workspace,continuous.artifact_id);
  e.comparison=runPython(String.raw`import json,sys,torch
from pathlib import Path
def check(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: check(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): check(x,y)
    else: assert a==b,(a,b)
a,b=[torch.load(Path(p)/'latest.pt',map_location='cpu',weights_only=True) for p in sys.argv[1:]]
for field in ('model_state','best_state','optimizer_state','rng_state','replay_state','updates','log'): check(a[field],b[field])
print(json.dumps(dict(exact=True,epochs=a['epoch'],updates=a['updates'],student_steps=a['replay_state']['total_steps'])))`,[left.root,right.root]);
  const handoff=await o.stage('handoff','prepare_handoff',{artifact_id:resumed.artifact_id});
  const next=handoff.handoff;
  const validationRefs=await o.phaseRefs('validation','evaluation',next.operation,next.inputs,next.config_suggestions,['signals','sampling']);
  const evaluated=await o.stage('evaluation',next.operation,next.inputs,{...next.config_suggestions,plan_refs:validationRefs});
  const evaluatedArtifact=await artifact(workspace,evaluated.artifact_id);
  const report=await evaluatedArtifact.member('report.json');
  e.acceptance=report.final_acceptance;
  if (values.tracking) {
    assert.equal(report.engineering_objective?.primary_metric,'tracking_rmse');
    for (const phase of ['validation','test']) {
      assert.equal(typeof report.engineering_objective.gates[phase].checks.action_rms,'boolean');
      assert.equal(typeof report.engineering_objective.gates[phase].checks.increment_rms,'boolean');
    }
    e.engineering_objective=report.engineering_objective;
  }
  assert.equal(e.acceptance.approved,true);
  e.status='PASS';
} catch(error) {
  e.status='FAIL';e.error=error.message;
  throw error;
} finally {
  await writeSmokeEvidence(path,e);
  console.log(JSON.stringify({status:e.status,path,comparison:e.comparison}));
}
