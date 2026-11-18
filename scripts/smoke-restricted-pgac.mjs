/** S12 R03 prospective real native PGAC + original-Plant gate. No Host lifecycle. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {fileURLToPath} from 'node:url';
import {connectHarness,project,writeSmokeEvidence} from './harness-rpc.mjs';
import {artifact,assertAudit,auditRows,checkpointInspect,freezeAudit,hash,observer,openEvidence,preflight,runPython,terminalWorkers,validateSuperseded} from './pgac-native-observer.mjs';

const {values}=parseArgs({options:{registration:{type:'string'},'registration-sha256':{type:'string'},'validation-evidence':{type:'string'},
  'validation-proof':{type:'string'},'validation-proof-sha256':{type:'string'},'gradient-probes':{type:'string'},'gradient-probes-sha256':{type:'string'},
  'gradient-oracle':{type:'string'},'gradient-oracle-sha256':{type:'string'},evidence:{type:'string'},'mcp-audit':{type:'string'},'web-log':{type:'string'},
  resume:{type:'boolean'},'validate-only':{type:'boolean'},supersedes:{type:'string'},'supersedes-sha256':{type:'string'},help:{type:'boolean',short:'h'}}});
if(values.help){console.log('node scripts/smoke-restricted-pgac.mjs --registration <R03 JSON> --registration-sha256 <SHA> --validation-evidence <R01 PASS> --validation-proof <independent PASS JSON> --validation-proof-sha256 <SHA> --gradient-probes <pre-training TRAIN points JSON> --gradient-probes-sha256 <SHA> --gradient-oracle <post-oracle.py> --gradient-oracle-sha256 <SHA> --evidence <new JSON> --mcp-audit <actual observer JSONL> [--web-log <Host log>] [--resume] [--validate-only]');
  console.log('No Host/session restart, no altered source or gates. Validate-only makes no RPC. Resume observes original handles/commands without admitting completed phases again. Post-training gradient oracle is a separate explicit independent gate.');
  console.log('Reviewed dependency-only repair: use a NEW --evidence with --supersedes <original two-proposal FAIL JSON> --supersedes-sha256 <exact SHA>. Original evidence/preRegistered and both job handles are retained; completed signals is reused, failed training gets one explicit-dependencies request. Thereafter resume the new evidence with --resume.');process.exit(0);}
assert.ok(values.evidence&&values['mcp-audit']);const evidencePath=resolve(values.evidence),auditPath=resolve(values['mcp-audit']);
const prepared=await preflight(values,[fileURLToPath(import.meta.url),resolve(project,'scripts/pgac-native-observer.mjs')]);
prepared.pinned.mcp_audit_path=auditPath;
await validateSuperseded(values,prepared);
if(values['validate-only']){console.log(JSON.stringify({status:'PASS_PREFLIGHT_ONLY',cases:prepared.cases.map(row=>row.key),R01_independent_proof:prepared.pinned.validation_proof_sha256,native_submitted:false}));process.exit(0);}
const e=await openEvidence(evidencePath,values,prepared,'S12-R03'),save=()=>writeSmokeEvidence(evidencePath,e);

const tensorAudit=String.raw`import hashlib,json,sys,torch
from pathlib import Path
from control_harness.artifacts import Workspace
from control_harness.restricted_pgac import _state_hash
torch.set_num_threads(1);w=Workspace(sys.argv[1]);root,meta=w.resolve(sys.argv[2],kind='external_pgac_candidate');base,bm=w.resolve(sys.argv[3],kind='external_imitation_policy')
r=json.loads((root/'report.json').read_text(encoding='utf-8'));initial=torch.load(root/'initial_imitation_state.pt',weights_only=True,map_location='cpu');actual=torch.load(base/'bundle/model.pt',weights_only=True,map_location='cpu')
assert initial.keys()==actual.keys() and all(torch.equal(initial[k],actual[k]) for k in actual)
c=torch.load(root/'latest.pt',weights_only=True,map_location='cpu');selected=torch.load(root/'model.pt',weights_only=True,map_location='cpu')
assert c['contract']['algorithm']=='external_physical_policy_optimization';assert c['epoch']==int(sys.argv[4]);assert c['updates']==c['epoch']*20
assert len(c['log'])==c['epoch'] and all(torch.equal(selected[k],c['best_state'][k]) for k in selected)
assert r['initial_weights_sha256']==_state_hash(actual)==r['first_update_in_this_run']['state_sha256_before']
assert r['initial_imitation_bundle_weights_sha256']==hashlib.sha256((base/'bundle/model.pt').read_bytes()).hexdigest()
assert r['first_update_in_this_run']['learning_rate']==.0003 and r['first_update_in_this_run']['weight_decay']==0
assert all(float(s['step'])==c['updates'] for s in c['optimizer_state']['state'].values())
assert any(not torch.equal(actual[k],c['model_state'][k]) for k in actual),'Actual Adam state must update, even when unchanged baseline is selected'
print(json.dumps({'status':'PASS','initial_tensor_exact':True,'first_Adam_parameters_exact':True,'initial_state_sha256':_state_hash(initial),'updates':c['updates'],
 'epochs':c['epoch'],'Adam_steps':[float(s['step']) for s in c['optimizer_state']['state'].values()],'last_state_changed':True,
 'selected_best_state_exact':True,'baseline_weight_file_sha256':hashlib.sha256((base/'bundle/model.pt').read_bytes()).hexdigest()},allow_nan=False))`;

try{
  const rpc=await connectHarness({logPath:values['web-log']??process.env.CONTROL_HARNESS_WEB_LOG,connectionClose:true});
  for(const item of prepared.cases){const o=observer(rpc,e,evidencePath,item),inputs={policy:item.policy_id,model:item.actual.candidate,validation:item.actual.validation},cfg=structuredClone(item.config);
    const refs=await o.phaseRefs('optimize','optimize','optimize_external_policy',inputs,cfg,['signals','training']);
    const changedLr=structuredClone(cfg);changedLr.learning_rate*=2;await o.deny('changed-learning-rate-oldrefs','optimize_external_policy',inputs,{...changedLr,plan_refs:refs});
    const changedDomain=structuredClone(cfg);changedDomain.domain.x[0]=[-1.9,1.9];await o.deny('changed-domain-oldrefs','optimize_external_policy',inputs,{...changedDomain,plan_refs:refs});
    await o.deny('changed-model-dependency-oldrefs','optimize_external_policy',{...inputs,model:item.actual.plant},{...cfg,plan_refs:refs});
    const result=await o.stage('optimize','optimize_external_policy',inputs,{...cfg,plan_refs:refs});assert.equal(result.kind,'external_pgac_candidate');assert.equal(result.status,'verified');
    const candidate=await artifact(item.workspace,result.artifact_id),report=await candidate.member('report.json');assert.equal(candidate.metadata.activation,'none');
    assert.equal(report.algorithm,item.expected_algorithm);assert.equal(report.initialized_from_actual_imitation,true);assert.equal(report.learned_critic,false);assert.equal(report.ppo,false);assert.equal(report.sac,false);assert.equal(report.model_parameters_frozen,true);
    assert.equal(report.test_used_for_updates_or_checkpoint_selection,false);assert.equal(report.original_plant_closed_loop,'not_run');assert.equal(report.policy_activation,'not_changed');assert.equal(report.requires_original_plant_gate,true);assert.equal(report.rollout_domain_checks_passed,true);
    assert.equal(report.imitation_policy,item.policy_id);assert.equal(report.rollout_model,item.actual.candidate);assert.equal(report.model_validation,item.actual.validation);assert.equal(report.plant,item.actual.plant);
    assert.equal(report.trained_epochs,cfg.epochs);assert.equal(report.optimizer_updates,60);assert.equal(report.resumed_from_epoch,0);assert.deepEqual(report.training_trajectory_ids,Array.from({length:20},(_,i)=>i));assert.deepEqual(report.validation_trajectory_ids,[20,21,22,23,24,25]);
    assert.deepEqual(report.original_evaluation_config,cfg.original_evaluation);assert.equal(report.initial_imitation_bundle_weights_sha256,item.baseline_weights_sha256);
    await o.deny('changed-policy-source-oldrefs','optimize_external_policy',{...inputs,policy:result.artifact_id},{...cfg,plan_refs:refs});
    const tensors=runPython(tensorAudit,[item.workspace,result.artifact_id,item.policy_id,String(cfg.epochs)]);
    const handoff=await o.stage('handoff','prepare_handoff',{artifact_id:result.artifact_id});assert.equal(handoff.handoff.operation,'evaluate_external_pgac');assert.deepEqual(handoff.handoff.inputs,{candidate:result.artifact_id});assert.deepEqual(handoff.handoff.config_suggestions,cfg.original_evaluation);
    const evalInputs={candidate:result.artifact_id},evalCfg=structuredClone(cfg.original_evaluation),evalRefs=await o.phaseRefs('evaluate','evaluate','evaluate_external_pgac',evalInputs,evalCfg,['signals','sampling']);
    const changedObjective=structuredClone(evalCfg);changedObjective.evaluation.q=2;await o.deny('postfit-objective-oldrefs','evaluate_external_pgac',evalInputs,{...changedObjective,plan_refs:evalRefs});
    const evaluated=await o.stage('evaluate','evaluate_external_pgac',evalInputs,{...evalCfg,plan_refs:evalRefs});assert.equal(evaluated.kind,'external_optimized_policy');assert.equal(evaluated.status,'verified','An actually feasible original baseline must remain an honest approved fallback');
    const final=await artifact(item.workspace,evaluated.artifact_id),finalReport=await final.member('report.json'),selection=await final.member('selection.json');
    assert.equal(selection.phase,'frozen_before_any_test_rollout');assert.equal(hash(await final.memberBytes('selection.json')),finalReport.selection_sha256);
    if(item.key==='f05-matlab'){
      const native=selection.decision,recomputed=finalReport.selection,nullable=value=>Array.isArray(value)&&value.length===0?null:value;
      assert.equal(nullable(native.choice),recomputed.choice);for(const key of ['reason','candidate_eligible','selection_partition','test_used','comparison_atol'])assert.equal(native[key],recomputed[key]);
      for(const name of ['expert','imitation','candidate']){assert.equal(native.scores[name].feasible,recomputed.scores[name].feasible);const a=nullable(native.scores[name].score),b=recomputed.scores[name].score;
        if(b===null)assert.equal(a,null);else assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=1e-12,'Actual MATLAB selection score differs from saved Python physical-row recomputation');}
    }else assert.deepEqual(selection.decision,finalReport.selection);
    assert.equal(finalReport.selection.selection_partition,'validation');assert.equal(finalReport.selection.test_used,false);assert.equal(finalReport.test_used_for_checkpoint_or_hyperparameter_selection,false);assert.equal(finalReport.stability_proof,false);assert.equal(finalReport.improvement_required_for_stage_completion,false);
    assert.deepEqual(finalReport.original_evaluation_config,evalCfg);assert.equal(finalReport.final_acceptance.approved,true);assert.ok(['candidate','imitation'].includes(finalReport.final_acceptance.selected_version));
    for(const phase of ['validation','test'])for(const name of ['expert','imitation','candidate']){const run=finalReport.closed_loop[phase][name];assert.ok(run&&run.feedback.includes('own states'));await final.memberBytes(run.rollout_file);
      assert.equal(typeof run.quadratic.feasible,'boolean');if(run.quadratic.feasible){assert.ok(Number.isFinite(run.quadratic.score));assert.ok(Object.values(run.quadratic.checks).every(Boolean));}
      if(name==='candidate'&&run.quadratic.feasible){assert.equal(run.coverage_guard?.status,'passed');assert.equal(run.coverage_guard.violations,0);assert.equal(run.coverage_guard.passed_subticks,6*40*finalReport.plan.plant_substeps);}}
    const selected=finalReport.final_acceptance.selected_version;for(const phase of ['validation','test'])assert.equal(finalReport.closed_loop[phase][selected].quadratic.feasible,true);
    if(selected==='candidate'){const val=finalReport.closed_loop.validation,ts=finalReport.closed_loop.test,tol=1e-12;if(val.imitation.quadratic.feasible)assert.ok(val.candidate.quadratic.score<val.imitation.quadratic.score-tol);if(ts.imitation.quadratic.feasible)assert.ok(ts.candidate.quadratic.score<=ts.imitation.quadratic.score+tol);}
    const actualAudit=await assertAudit(auditPath,await o.session(),item.key==='f05-matlab');
    // Confirm the altered engineering summary, then prove the immutable trained
    // contract rejects it before the original source/MATLAB is dispatched.
    const changedRefs=await o.phaseRefs('postfit-changed','postfit-changed','evaluate_external_pgac',evalInputs,changedObjective,['signals','sampling']);
    const before=await auditRows(auditPath,await o.session());await o.stage('postfit-changed','evaluate_external_pgac',evalInputs,{...changedObjective,plan_refs:changedRefs},{expected:'failed'});
    assert.match(e.jobs[o.key('postfit-changed')].record.detail??'',/changed after PGAC|evaluation.*changed/i);assert.deepEqual(await auditRows(auditPath,await o.session()),before,'Postfit changed rules executed original MATLAB');
    assert.deepEqual(await assertAudit(auditPath,await o.session(),item.key==='f05-matlab'),actualAudit);
    if(item.key===prepared.registration.negative_domain.case){const bad=structuredClone(prepared.registration.negative_domain.config),badRefs=await o.phaseRefs('negative-domain','negative-domain','optimize_external_policy',inputs,bad,['signals','training']);
      const prior=await auditRows(auditPath,await o.session());await o.stage('negative-domain','optimize_external_policy',inputs,{...bad,plan_refs:badRefs},{expected:'failed'});const row=e.jobs[o.key('negative-domain')];assert.match(row.record.detail??'',/domain|coverage/i);assert.ok(row.record.checkpoint);
      const checkpoint=runPython(checkpointInspect,[resolve(row.record.checkpoint)]);assert.equal(checkpoint.epoch,0);assert.equal(checkpoint.updates,0);assert.equal(checkpoint.algorithm,'external_physical_policy_optimization');assert.equal(checkpoint.checkpoint_kind,'external_pgac_candidate');assert.equal(checkpoint.owner_alive,false);
      e.negative_domain={status:'PASS_GENUINE_FAILED_STAGE',jobId:row.jobId,attemptId:row.attemptId,checkpoint,baseline_unchanged:true};assert.deepEqual(await auditRows(auditPath,await o.session()),prior);await save();}
    const baseline=await artifact(item.workspace,item.policy_id);assert.equal(hash(await baseline.memberBytes('bundle/model.pt')),item.baseline_weights_sha256);assert.equal(baseline.metadata.validation.status,'verified');
    await o.phaseRefs('optimize','optimize','optimize_external_policy',inputs,cfg,['signals','training']);await o.phaseRefs('evaluate','evaluate','evaluate_external_pgac',evalInputs,evalCfg,['signals','sampling']);
    const records=await o.events();assert.equal(records.filter(row=>row.event.type.startsWith('turn/')).length,0);const checked={caseKey:item.key,workspace:item.workspace,sessionId:await o.session(),
      imitation_policy:item.policy_id,rollout_model:item.actual.candidate,model_validation:item.actual.validation,candidate:result.artifact_id,evaluated:evaluated.artifact_id,
      tensors,selection:finalReport.selection,final_acceptance:finalReport.final_acceptance,actual_mcp:actualAudit,terminal:await terminalWorkers(o,e),confirmation_source:'test_fixture',status:'PASS',
      scope:'Finite original-source scenarios only; no global stability or standalone online OOD fallback claim'};
    const old=e.checks.find(row=>row.caseKey===item.key);if(old)assert.deepEqual(old,checked);else e.checks.push(checked);await save();
  }
  for(const [name,digest] of Object.entries(prepared.pinned.source_sha256))assert.equal(hash(await readFile(resolve(project,name))),digest);
  await freezeAudit(auditPath,e,evidencePath,'R03');e.native_result='PASS';e.status='PASS';e.independent_gradient_oracle={status:'REQUIRED_SEPARATE_GATE',...prepared.pinned.gradient_oracle,probes:prepared.pinned.gradient_probes};e.finished_at=new Date().toISOString();e.error=null;await save();
  console.log(JSON.stringify({status:e.status,evidence:evidencePath,cases:e.checks.length,jobs:e.admissions.length,zero_admissions:e.rejections.length,independent_gradient_oracle:'PENDING_SEPARATE_GATE',LLM_turns:0}));
}catch(error){e.status='FAIL';e.error={name:error.name,message:error.message,stack:error.stack};e.finished_at=new Date().toISOString();await save();throw error;}
