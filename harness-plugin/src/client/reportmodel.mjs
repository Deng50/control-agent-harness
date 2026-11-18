/** Persisted report/array presentation only. No metrics or physical states are inferred. */
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const list = value => Array.isArray(value) ? value : [];
export const finite = value => typeof value === 'number' && Number.isFinite(value);
export const formatValue = value => {
  if (value === null || value === undefined) return '未报告 / 不适用';
  if (typeof value === 'boolean') return value ? '是' : '否';
  if (finite(value)) return value !== 0 && Math.abs(value) < .001 ? value.toExponential(4) : String(Number(value.toPrecision(6)));
  if (Array.isArray(value)) return value.map(formatValue).join(' / ');
  return typeof value === 'string' ? value : '—';
};
const labels = {
  train:'训练',validation:'验证',test:'测试',expert:'外部专家',imitation:'模仿基线',candidate:'PGAC 候选',optimized:'优化策略',
  rmse:'RMSE（物理单位）',nrmse:'NRMSE',rmse_K:'温度 RMSE / K',normalized_physics_rms:'归一化物理残差 RMS',
  normalized_initial_max_abs_error:'归一化初值最大误差',normalized_initial_abs_error:'归一化初值误差',energy:'动作平方积分（代理）',tv:'动作总变差',overshoot:'超调比',
  overshoot_ratio:'超调比',settling_time:'调节时间 / s',oscillation_crossings:'振荡过零次数',
  hard_constraint_violations:'硬约束违规数',hard_constraints_complete:'硬约束核验完整',finite:'数值有限',
  max_state_violation:'状态最大越界',max_input_violation:'动作最大越界',max_output_violation:'输出最大越界',max_rate_violation:'动作增量最大越界',
  input_total_variation:'动作总变差',samples:'样本数',score:'评分',hard_constraints_passed:'硬门通过',nominal_mse:'名义 MSE',corrected_mse:'校正 MSE',
};
const scalarMetricKeys = Object.keys(labels).filter(key => !['train','validation','test','expert','imitation','candidate','optimized'].includes(key));
function metricRows(value, prefix, rows = [], depth = 0) {
  if (!object(value) || depth > 4) return rows;
  for (const key of scalarMetricKeys) if (Object.hasOwn(value,key)) {
    const raw = value[key];
    if (raw === null || typeof raw !== 'object' || Array.isArray(raw) && raw.every(v => v === null || finite(v))) rows.push({label:`${prefix} · ${labels[key]}`,value:raw});
  }
  for (const key of ['state','output','raw','projected','metrics']) if (object(value[key])) metricRows(value[key],`${prefix} / ${{state:'状态',output:'输出',raw:'原始动作',projected:'投影后动作',metrics:'指标'}[key]}`,rows,depth+1);
  return rows;
}
function metricSections(report) {
  const rows=[];
  for (const key of ['train','validation','test']) metricRows(report[key],labels[key],rows);
  for (const kind of ['one_step','rollout']) for (const partition of ['validation','test']) metricRows(report[kind]?.[partition],`${kind==='one_step'?'一步预测':'递归多步预测'} / ${labels[partition]}`,rows);
  for (const partition of ['train','validation','test']) metricRows(report.offline?.[partition],`离线 ${labels[partition]}`,rows);
  return rows;
}
function closedRuns(report) {
  const runs=[];
  for (const [partition,section] of [['验证',report.closed_loop?.validation],['测试',report.closed_loop?.test]]) {
    if (object(section)) for (const name of ['expert','imitation','candidate']) if (object(section[name])) runs.push({label:`${partition} / ${labels[name]}`,run:section[name]});
  }
  for (const name of ['expert','imitation','candidate']) if (object(report.closed_loop?.[name])) runs.push({label:labels[name],run:report.closed_loop[name]});
  return runs;
}
function legacyConditions(report) {
  if (Array.isArray(report.comparison) && report.comparison.some(row => ['expert','imitation','optimized'].some(key => object(row[key])))) return report.comparison.map((row,index)=>({
    label:row.scenario?.name ?? `工况 ${index+1}`,traces:['expert','imitation','optimized'].filter(key=>object(row[key])).map(key=>({label:labels[key],run:row[key]}))}));
  const hasCurve=run=>list(run?.trajectory).some(row=>finite(row.step)&&['output','reference','u'].every(key=>Array.isArray(row[key])&&row[key].length&&row[key].every(finite)));
  const choices=[];
  // These are saved evaluations of different conditions and controller versions.
  // A validation/tuning run must never inherit the test partition's label.
  const sections=[
    {key:'test',partition:'test',version:'selected',label:'测试 / 已选参数'},
    {key:'verification_candidate',partition:'validation',version:'candidate',label:'独立复验 / 候选'},
    {key:'verification_baseline',partition:'validation',version:'baseline',label:'独立复验 / 初始基线'},
    {key:'baseline',partition:'tuning',version:'baseline',label:'调参工况 / 初始基线'},
  ];
  for(const section of sections)for(const [index,run]of list(report[section.key]?.runs).entries())if(hasCurve(run)){
    const label=`${section.label} / ${run.scenario?.name??`工况 ${index+1}`}`;
    choices.push({label,partition:section.partition,version:section.version,source:`${section.key}.runs[${index}]`,traces:[{label,run}]});
  }
  if(choices.length)return choices;
  return list(report.closed_loop?.test).filter(hasCurve).map((run,index)=>({label:`测试 / ${run.scenario?.name??`工况 ${index+1}`}`,partition:'test',source:`closed_loop.test[${index}]`,traces:[{label:run.scenario?.name??`工况 ${index+1}`,run}]}));
}
function filesOf(artifact) {
  if (Array.isArray(artifact?.files)) return artifact.files.map(item=>typeof item==='string'?item:item?.name??item?.path).filter(v=>typeof v==='string');
  return Object.keys(artifact?.metadata?.files??artifact?.files??{});
}
const rawLabels={test_predictions:'测试预测',one_step_predictions:'一步预测',rollout_predictions:'递归多步预测',transitions:'采样状态转移',expert:'外部专家数据'};
export function availableRawFiles(artifact) {
  return filesOf(artifact).filter(name=>/^(?:test_predictions|one_step_predictions|rollout_predictions|transitions|expert|(?:validation-|test-)?(?:expert|imitation|candidate)-rollout)\.npz$/.test(name))
    .map(name=>({name,label:rawLabels[name.slice(0,-4)]??name}));
}
export function normalizeReport({report,modelInterface,artifact}={}) {
  const r=object(report)?report:{};const kind=artifact?.kind??artifact?.metadata?.kind;
  const iface=r.physical_interface??r.plan?.interface??modelInterface??artifact?.metadata?.interface??{};
  const notices=['指标直接显示已保存的完整报告值；图中抽样点不参与重新计算指标。'];
  if (!object(report)) notices.push('此产物没有可用报告；文件存在不代表已训练或已完成闭环验证。');
  if (r.display_summary?.omitted) notices.push(r.display_summary.omitted);
  if (r.display_summary?.sampled_fields) notices.push('报告预览已抽样；完整结果以已登记 report.json 为准。');
  if (r.algorithm==='mpc_design_only') notices.push('仅设计：未执行闭环验收或 GA 调参；单点求解诊断不代表控制性能或稳定性。');
  if (kind==='supervised_model' || r.algorithm==='supervised_training') notices.push('监督回归：输入是用户定义的特征列和历史窗口；没有据此推定物理状态或名义模型基线。');
  if (r.algorithm==='supervised_discrete_next_state_model') notices.push('离散 next_x 预测，输出已反归一化到状态单位；多步图为递归预测。录制模拟器留出结果不等于独立实物验证。');
  if (kind==='pinn_model' || 'is_pinn' in r) notices.push(r.is_pinn===false?'物理项关闭的消融模型：物理残差仅作诊断，没有作为训练损失优化。':'条件热 ODE PINN：仅支持声明范围内的时间、初值及恒定功率轨迹；不是任意状态转移 Plant。');
  if (r.offline) notices.push('离线动作误差使用记录的特征及真实前一动作（teacher forcing）；这不是自主闭环响应。');
  if (kind==='external_pgac_candidate' || r.requires_original_plant_gate===true) notices.push('PGAC 数值候选尚未通过原始 Plant 验收，保持非激活；预测模型上的验证评分不替代原 Plant 闭环。');
  if (r.execution_surface) notices.push(`实际评价范围：${r.execution_surface}`);
  if (r.original_matlab_simulink_closed_loop==='not_run') notices.push('原 MATLAB / Simulink 闭环未运行。');
  if (r.closed_loop==='not_run' || r.original_plant_closed_loop==='not_run') notices.push('原 Plant 闭环未运行。');
  if (r.stability_proof===false) notices.push('这些有限工况结果不构成稳定性证明或硬件认证。');
  if (r.controller_benchmark) notices.push('运行时间只显示记录的测量范围；控制器推理与 Plant 仿真时间分开，不把整批墙钟时间当推理延迟。');
  const summary=[];
  if(artifact?.status)summary.push({label:'产物状态（非精度证明）',value:artifact.status});
  if(object(r.network))for(const key of ['trained_epochs','best_epoch','parameter_count','optimizer_updates'])if(key in r.network)summary.push({label:`内部监督网络 / ${key}`,value:r.network[key]});
  for (const [key,label] of Object.entries({acceptance:'报告验收',accuracy_claim:'精度声明',status:'报告状态',model_instantiated:'数值模型已实例化',required_parameters:'仍缺参数',closed_loop_validation:'闭环验证',controller_parameters_tuned:'参数已调优',trained_epochs:'已训练轮数',best_epoch:'验证集选定轮次',optimizer_updates:'优化器更新数',parameter_count:'网络参数数',conditional_parameterization:'条件参数化',selection:'选择规则',selection_reason:'选择原因',selected_imitation_baseline:'保留模仿基线',tuned_candidate_accepted:'接受调参候选',test_evaluations:'最终测试次数'})) {
    if (key in r && !object(r[key])) summary.push({label,value:r[key]});
  }
  if (object(r.selection)) for (const key of ['choice','reason','candidate_eligible','selection_partition','test_used']) if (key in r.selection) summary.push({label:`选择 / ${key}`,value:r.selection[key]});
  if (object(r.final_acceptance)) for (const [key,value] of Object.entries(r.final_acceptance)) if (!object(value)) summary.push({label:`原 Plant 终验 / ${key}`,value});
  if (r.solver_check) summary.push({label:'单点求解诊断',value:r.solver_check.status});
  const training=list(r.log);const logSeries=[];
  for (const [key,label] of Object.entries({training_loss:'训练损失',validation_loss:'验证损失',validation_score:'验证评分',predictor_validation_score:'预测模型验证评分'})) if (training.some(row=>finite(row[key]))) logSeries.push({label,points:training.map(row=>[row.epoch,row[key]])});
  if (training.some(row=>object(row.training_components))) for (const [key,label] of [['data','数据项（未加权）'],['physics','物理项（未加权）'],['initial','初值项（未加权）']]) logSeries.push({label,points:training.map(row=>[row.epoch,row.training_components?.[key]])});
  const runs=closedRuns(r);const closedMetrics=[];
  for (const {label,run} of runs) {
    closedMetrics.push({label:`${label} / 执行状态`,value:run.status});
    if (Array.isArray(run.metrics)) for (const row of run.metrics) metricRows(row.metrics,`${label} / 工况 ${row.trajectory_id}`,closedMetrics);
    else metricRows(run.metrics,label,closedMetrics);
    metricRows(run.quadratic,`${label} / 冻结目标`,closedMetrics);
  }
  const metricDetails=[];
  function detail(value,prefix) { if (!object(value)) return;for (const key of ['per_output','per_control','per_action','per_target','per_condition']) for (const row of list(value[key])) metricDetails.push({label:`${prefix} / ${row.name??row.signal??row.condition_index??row.trajectory_id??''}${row.unit?` · ${row.unit}`:''}`,row}); }
  for (const partition of ['train','validation','test']) {detail(r[partition],labels[partition]);detail(r.offline?.[partition]?.raw,`离线${labels[partition]} 原始`);detail(r.offline?.[partition]?.projected,`离线${labels[partition]} 投影`);}
  for (const {label,run} of runs) for (const row of list(run.metrics)) detail(row.metrics,`${label} / 工况 ${row.trajectory_id}`);
  const trials=list(r.trials);const candidateRows=[];
  for (const key of ['baseline','best','last','verification_baseline','verification_candidate',...(r.trials?['test']:[])]) if (object(r[key])) {
    candidateRows.push({label:{baseline:'初始基线',best:'最好可行候选',last:'最后一次候选（可能失败）',verification_baseline:'独立复验 / 基线',verification_candidate:'独立复验 / 候选',test:'最终测试 / 已选参数'}[key],value:{...r[key],parameters:r[key].complete_parameters??r[key].parameters??(key==='baseline'?r.baseline_parameters:key==='test'?r.chosen_parameters:undefined)}});
    for (const run of list(r[key].runs??r[key].tuning_evaluation?.runs)) detail(run.objective_metrics,`${key} / ${run.scenario?.name??''}`);
  }
  if(Object.hasOwn(r,'best')&&r.best===null)candidateRows.splice(1,0,{label:'最好可行候选',value:{score:null,hard_constraints_passed:false,error:'没有可行候选'}});
  const benchmark=[];for (const name of ['expert','imitation','candidate','optimized']) if (object(r.controller_benchmark?.[name])) benchmark.push({label:labels[name],...r.controller_benchmark[name]});
  return {algorithm:r.algorithm??'未报告',kind,interface:iface,notices:[...new Set(notices)],summary,metricRows:metricSections(r),closedMetrics,metricDetails,
    logSeries,conditions:legacyConditions(r),trials,candidateRows,benchmark,rawFiles:availableRawFiles(artifact),coverage:r.coverage??r.applicability,
    bundleAvailable:filesOf(artifact).some(name=>/^bundle\/(?:policy_manifest|pinn_manifest|dynamics_manifest|bundle_manifest)\.json$/.test(name)),report:r};
}

/** Split rather than bridge a nonfinite value, different trajectory or reset time. */
export function splitSegments(points) {
  const segments=[];let segment=[];let priorGroup,priorX;
  for (const point of points) {
    const [x,y,group]=point;
    if (!finite(x)||!finite(y)) {if(segment.length)segments.push(segment);segment=[];priorX=undefined;continue;}
    if (segment.length && (group!==priorGroup || x<=priorX)) {segments.push(segment);segment=[];}
    segment.push([x,y]);priorGroup=group;priorX=x;
  }
  if(segment.length)segments.push(segment);return segments;
}
function vector(value,index) { const raw=Array.isArray(value)?value[index]:index===0?value:undefined;return finite(raw)?raw:null; }
function width(values) {const row=list(values).find(v=>Array.isArray(v));return row?row.length:values?.length?1:0;}
function variables(model,key) {if(key==='state')return list(model.interface?.states);if(key==='output')return list(model.interface?.outputs);if(key==='action')return list(model.interface?.controls);return[];}
function targetVariable(model,index) {
  const targets=list(model.interface?.targets);const signal=targets[index]?.signal;return list(model.interface?.signals).find(row=>row.name===signal)??{name:signal??`目标 ${index+1}`};
}
export function arrayCharts(model,preview) {
  const slice=preview?.array_rows;if(!object(slice?.values))return[];
  const a=slice.values;const n=slice.returned_rows;const groups=a.trajectory_id??a.condition_index??Array(n).fill('one');
  const timeKey=['prediction_time','time','time_s','input_time'].find(key=>Array.isArray(a[key]));
  const x=timeKey?a[timeKey].map(row=>vector(row,0)):slice.source_rows;
  const xLabel=timeKey?'时间 / s（实际保存值）':'原始样本行号（非物理时间）';
  const charts=[];
  function add(title,keys,groupName,forcedVariables,units) {
    const present=keys.filter(([key])=>Array.isArray(a[key]));if(!present.length)return;
    const channels=Math.max(...present.map(([key])=>width(a[key])));
    for(let channel=0;channel<Math.min(channels,16);channel++) {
      const v=forcedVariables?.[channel]??variables(model,groupName)[channel]??targetVariable(model,channel);
      charts.push({title:`${title} · ${v?.name??`通道 ${channel+1}`}${units??v?.unit?` / ${units??v.unit}`:''}`,xLabel,
        series:present.map(([key,label,dashed])=>({label,dashed,points:a[key].map((row,i)=>[x[i],vector(row,channel),groups[i]])}))});
    }
  }
  if(a.predicted_temperature_K) {
    add('条件温度轨迹',[['predicted_temperature_K','网络温度'],['reference_temperature_K','参考温度',true]],'output',[{name:'温度',unit:'K'}]);
    add('实际 autograd 物理残差',[['normalized_physics_residual','归一化残差']],'output',[{name:'残差',unit:'无量纲'}]);
  } else if(a.predicted_state||a.predicted_output) {
    add('状态预测',[['predicted_state','预测'],['true_state','参考',true]],'state');
    add('输出预测',[['predicted_output','预测'],['true_output','参考',true]],'output');
  } else if(a.prediction) add('物理单位回归预测',[['prediction','预测'],['truth','记录目标',true]],'target');
  else if(a.projected_action||a.raw_action) add('离线动作预测（记录特征）',[['raw_action','原始网络'],['projected_action','投影后'],['expert_action','专家记录',true]],'action');
  else {
    add('当前输出（实际保存）',[['plant_y','当前输出'],['reference','目标',true]],'output');
    add('动作（控制更新起点）',[['u','实际施加动作']],'action');
    add('状态转移终点（横轴为区间起点）',[['next_x','下一状态']],'state');
    if(!a.plant_y)add('输出（实际保存）',[['y','当前输出'],['next_y','下一输出'],['reference','目标',true]],'output');
  }
  return charts;
}
