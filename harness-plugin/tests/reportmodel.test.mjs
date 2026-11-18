import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeReport, arrayCharts, splitSegments, availableRawFiles, formatValue } from '../src/client/reportmodel.mjs';

const physical={sample_time:.2,states:[{name:'position',unit:'m'},{name:'velocity',unit:'m/s'}],controls:[{name:'force',unit:'N'}],outputs:[{name:'position',unit:'m'}]};
const artifact=(kind,files)=>({kind,metadata:{interface:physical},files:files.map(name=>({name,path:`D:/confined/${name}`}))});
const preview=(values,rows=[7,9,11])=>({array_rows:{values,source_rows:rows,returned_rows:rows.length}});

test('sample rows remain sample rows and already physical predictions are not inverse scaled twice',()=>{
  const iface={interface_kind:'tabular_supervised',signals:[{name:'temperature',unit:'K'}],targets:[{signal:'temperature'}],input_dim:9,output_dim:1};
  const model=normalizeReport({report:{algorithm:'supervised_training',train:{rmse:[1]},test:{rmse:[2],nrmse:[.1]},normalization:{scale:100},log:[]},modelInterface:iface});
  assert.match(model.notices.join(' '),/没有据此推定物理状态/);assert.equal(model.conditions.length,0);
  const charts=arrayCharts(model,preview({prediction:[[301],[302],[303]],truth:[[300],[301],[302]]}));
  assert.deepEqual(charts[0].series[0].points,[[7,301,'one'],[9,302,'one'],[11,303,'one']]);assert.match(charts[0].title,/temperature \/ K/);assert.match(charts[0].xLabel,/非物理时间/);
  assert.equal(model.metricRows.find(row=>row.label==='测试 · NRMSE').value[0],.1);
});
test('nonfinite samples, trajectory boundaries and reset time split actual polylines',()=>{
  const points=[[0,1,'a'],[1,2,'a'],[2,null,'a'],[3,4,'a'],[4,5,'b'],[0,6,'b'],[1,7,'b'],[2,NaN,'b']];
  assert.deepEqual(splitSegments(points),[[[0,1],[1,2]],[[3,4]],[[4,5]],[[0,6],[1,7]]]);
  assert.equal(formatValue(null),'未报告 / 不适用');assert.equal(formatValue(false),'否');
});
test('S12 keyed validation/test closed loops preserve baseline fallback and actual metrics',()=>{
  const run={status:'completed',metrics:[{trajectory_id:20,metrics:{rmse:.3,energy:9,tv:2,overshoot:null,per_output:[{name:'T',unit:'K',rmse:.3,overshoot_ratio:null,oscillation_crossings:2}]}}]};
  const model=normalizeReport({artifact:artifact('external_optimized_policy',['validation-candidate-rollout.npz','test-imitation-rollout.npz','bundle/policy_manifest.json']),report:{
    algorithm:'differentiable_physical_rollout_policy_gradient',selection:{choice:'imitation',reason:'candidate_rejected',test_used:false},
    final_acceptance:{selected_version:'imitation',candidate_accepted:false,approved:true},closed_loop:{validation:{expert:run,imitation:run,candidate:run},test:{expert:run,imitation:run,candidate:run}},stability_proof:false}});
  assert.equal(model.conditions.length,0);assert.equal(model.closedMetrics.filter(row=>row.label.endsWith('RMSE（物理单位）')).length,6);
  assert.equal(model.metricDetails.length,6);assert.equal(model.summary.find(row=>row.label==='选择 / choice').value,'imitation');assert.ok(model.bundleAvailable);
  assert.match(model.notices.join(' '),/不构成稳定性证明/);assert.equal(model.rawFiles.length,2);
});
test('S07 draft does not invent any closed loop or GA data',()=>{
  const model=normalizeReport({report:{algorithm:'mpc_design_only',status:'parameterized_draft',model_instantiated:false,required_parameters:['mass','sample_time'],closed_loop:{status:'not_run',ga_trials:0},solver_check:{status:'not_run'}},artifact:artifact('controller_design',['report.json','design.json','bundle/bundle_manifest.json'])});
  assert.match(model.notices.join(' '),/仅设计：未执行闭环/);assert.equal(model.trials.length,0);assert.equal(model.conditions.length,0);assert.equal(model.metricRows.length,0);assert.equal(model.summary.find(row=>row.label==='仍缺参数').value[0],'mass');
});
test('PINN plots real temperature and residual values with distinct units, including physics-off warning',()=>{
  const model=normalizeReport({report:{algorithm:'conditional_ode_physics_off_ablation',is_pinn:false,test:{rmse_K:.2,normalized_physics_rms:.4,normalized_initial_max_abs_error:.01}},artifact:artifact('pinn_model',['test_predictions.npz'])});
  const charts=arrayCharts(model,preview({time_s:[[0],[1],[2]],condition_index:[1,1,2],predicted_temperature_K:[[300],[301],[302]],reference_temperature_K:[[299],[300],[301]],normalized_physics_residual:[[.1],[.2],[.3]]}));
  assert.equal(charts.length,2);assert.match(charts[0].title,/温度 \/ K/);assert.match(charts[1].title,/无量纲/);
  assert.deepEqual(charts[1].series[0].points,[[0,.1,1],[1,.2,1],[2,.3,2]]);assert.equal(splitSegments(charts[0].series[0].points).length,2);assert.match(model.notices.join(' '),/没有作为训练损失优化/);
});
test('S08 consumes objective_metrics and retains failed last candidate without a fabricated score',()=>{
  const run={scenario:{name:'holdout'},metrics:{rmse:.1},objective_metrics:{per_output:[{name:'position',unit:'m',rmse:.1,overshoot_ratio:.02,settling_time:1.2,oscillation_crossings:3}]},trajectory:[]};
  const model=normalizeReport({report:{algorithm:'genetic_algorithm',baseline_parameters:{q:1},baseline:{score:2,hard_constraints_passed:true,runs:[run]},best:{score:1,complete_parameters:{q:2}},last:{score:null,hard_constraints_passed:false,error:'solver_failed'},trials:[{score:1},{score:null}],test:{score:1,runs:[run]},chosen_parameters:{q:2},tuned_candidate_accepted:false,original_matlab_simulink_closed_loop:'not_run'},modelInterface:physical});
  assert.equal(model.candidateRows.find(row=>row.label==='最后一次候选（可能失败）').value.score,null);
  assert.deepEqual(model.candidateRows.find(row=>row.label==='最好可行候选').value.parameters,{q:2});assert.equal(model.metricDetails.length,2);assert.equal(model.metricDetails[0].row.oscillation_crossings,3);
  assert.match(model.notices.join(' '),/Simulink 闭环未运行/);
});
test('offline projected action diagnostics are not presented as closed-loop observations',()=>{
  const model=normalizeReport({report:{algorithm:'external_supervised_imitation',offline:{test:{raw:{rmse:[1],nrmse:[.5]},projected:{rmse:[.2],nrmse:[.1]}}}},artifact:artifact('external_imitation_policy',['test_predictions.npz','model.pt'])});
  assert.equal(model.conditions.length,0);assert.equal(model.closedMetrics.length,0);assert.match(model.notices.join(' '),/teacher forcing/);assert.equal(model.bundleAvailable,false);
  const charts=arrayCharts(model,preview({raw_action:[[3],[4],[5]],projected_action:[[1],[1],[1]],expert_action:[[.9],[.8],[.7]]}));assert.equal(charts[0].series.length,3);assert.equal(charts[0].series[0].points[0][1],3);
});
test('registered file choices exclude weights, traversal and unrecognized archives',()=>{
  const files=availableRawFiles(artifact('supervised_model',['../test_predictions.npz','model.pt','unregistered.npz','one_step_predictions.npz','rollout_predictions.npz']));
  assert.deepEqual(files.map(file=>file.name),['one_step_predictions.npz','rollout_predictions.npz']);
});

test('MPC curve choices preserve actual partition/version and distinct stored conditions without inventing runs',()=>{
  const makeRun=(name,position,force)=>({scenario:{name},trajectory:[{step:0,output:[position],reference:[.7],u:[force]},{step:1,output:[position+.1],reference:[.7],u:[force-.1]}]});
  const tuning=makeRun('condition-101',.1,.2),baseline=makeRun('condition-201',.3,.4),candidate=makeRun('condition-201',.5,.6),heldout=makeRun('condition-301',.8,.9);
  const report={baseline:{runs:[tuning]},verification_baseline:{runs:[baseline]},verification_candidate:{runs:[candidate]},test:{runs:[heldout]},best:{tuning_evaluation:{runs:[{scenario:{name:'not-a-real-curve'},trajectory:[]}]}}};
  const original=JSON.stringify(report);
  const model=normalizeReport({report,modelInterface:physical});
  assert.deepEqual(model.conditions.map(row=>[row.partition,row.version,row.source]),[
    ['test','selected','test.runs[0]'],['validation','candidate','verification_candidate.runs[0]'],
    ['validation','baseline','verification_baseline.runs[0]'],['tuning','baseline','baseline.runs[0]']]);
  assert.match(model.conditions[0].label,/测试.*已选参数.*condition-301/);
  assert.match(model.conditions[1].label,/独立复验.*候选.*condition-201/);
  assert.match(model.conditions[2].label,/独立复验.*基线.*condition-201/);
  assert.match(model.conditions[3].label,/调参.*基线.*condition-101/);
  assert.equal(model.conditions[0].traces[0].run,heldout);assert.equal(model.conditions[1].traces[0].run,candidate);
  assert.equal(model.conditions[2].traces[0].run,baseline);assert.equal(model.conditions[3].traces[0].run,tuning);
  assert.equal(JSON.stringify(report),original);
  const empty=normalizeReport({report:{test:{runs:[{scenario:{name:'missing-real-values'},trajectory:[{step:0}]}]},verification_candidate:{runs:[]}},modelInterface:physical});
  assert.equal(empty.conditions.length,0);
});
