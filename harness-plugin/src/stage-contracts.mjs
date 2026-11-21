import contracts from '../../src/control_harness/stage_contracts.json' with {type:'json'};

export const stages = contracts.stages;
export const payloads = contracts.payloads;
export function allowedKinds(type) {
  if (Array.isArray(type)) return type;
  if (type.startsWith('@')) return contracts.groups[type.slice(1)];
  return ['file','artifact_directory'].includes(type) ? [] : [type];
}
export const workflowStageInputs = Object.fromEntries(Object.entries(stages)
  .filter(([,stage]) => Object.hasOwn(stage,'workflow_output'))
  .map(([name,stage]) => [name,Object.fromEntries(Object.entries(stage.inputs)
    .map(([key,type]) => [key,allowedKinds(type).length ? 'artifact' : type]))]));

export function kindIssue(field, type, actual) {
  const expected = allowedKinds(type);
  if (expected.includes(actual)) return;
  let hint = '请选择具有相应能力的已核验产物。';
  if (actual === 'plant_model' && expected.includes('nominal_model')) hint = '先 sample_model 完整采样，再用 convert_plant_predictor 核验方程或有界辨识。';
  if (actual === 'controller_design' && expected.includes('controller')) hint = '先用 verify_design_delivery 核验 workspace_controller 和用户工况。';
  return {status:'unsupported',field,expected,actual,message:`expected ${expected.join(' / ')}, received ${actual}。${hint}`};
}

// 比较有物理含义的端口、顺序和采样语义。显示名称不同可兼容。
export function capabilityIssues(request, metadata) {
  const issues = [], op=request.operation, config=request.config ?? {};
  const issue=(status,field,message)=>issues.push({status,field,message});
  const plant=metadata.plant, dataset=metadata.dataset;
  const original=op === 'convert_plant_predictor' ? plant : op === 'validate_differentiable_model' ? metadata.evaluation_model : null;
  if (original && dataset && (dataset.source?.model !== original.id || !dataset.parents.some(p=>p.id === original.id))) {
    issue('unsupported','inputs.dataset','请选择由指定原对象版本完整采样的数据。');
  }
  if (op === 'convert_plant_predictor') {
    if (!config.method) issue('clarify','config.method','请选择核验显式方程或按完整轨迹进行有界仿射辨识。');
    if (plant && plant.source?.backend !== 'python_callable') issue('unsupported','inputs.plant','本转换入口需要 python_callable 采样对象。');
    if (config.method === 'declared_equations' && !metadata.equations) issue('clarify','inputs.equations','请登记具有显式状态、输出、单位和采样周期的方程模型。');
    if (metadata.equations && !['discrete_polynomial','continuous_polynomial'].includes(metadata.equations.descriptor?.representation)) issue('unsupported','inputs.equations','请选择已登记的声明式连续或离散方程。');
    if (config.method === 'sampled_affine' && plant && !plant.descriptor?.adapter_spec?.C) issue('clarify','inputs.plant.adapter_spec.C','仿射路线需要明确的状态到输出映射 C；请选择带 C 的对象或改选已经登记的显式方程。');
    if (config.method === 'sampled_affine' && metadata.equations) issue('unsupported','inputs.equations','采样辨识仅使用绑定数据，请清除无关方程或选择方程核验。');
  }
  if (op==='collect_expert' && metadata.evaluation_model?.kind==='plant_model' && metadata.controller) {
    const selected=metadata.evaluation_model, predictor=metadata.controller.predictor;
    if (selected.descriptor?.adapter_spec?.backend!=='python_callable' || predictor?.source?.kind!=='plant_predictor_conversion' || predictor.source.plant!==selected.id || !predictor.parents.some(p=>p.id===selected.id)) issue('unsupported','inputs.evaluation_model','请选控制器预测模型来源闭包中绑定的同一 python_callable 原对象。');
  }
  if (op === 'collect_external_expert' && metadata.expert && plant) {
    const spec=metadata.expert.descriptor?.adapter_spec;
    if ((spec?.backend === 'matlab_function') !== ['matlab_function','simulink'].includes(plant.descriptor?.adapter_spec?.backend)) issue('unsupported','inputs.plant','专家和原对象需要同一受支持的 Python 或 MATLAB 执行入口。');
    const declared=spec?.interface;
    if (declared && ['states','outputs','controls','disturbances','sample_time'].some(k=>!samePhysical(declared[k],plant.interface[k]))) issue('unsupported','inputs.expert.interface','专家端口顺序、单位、约束或采样周期与对象不匹配。');
  }
  if (op === 'verify_design_delivery' && metadata.design?.descriptor?.model_instantiated) {
    const design=metadata.design.descriptor, iface=design.physical_interface, prediction=design.prediction;
    if (!['continuous_polynomial','discrete_polynomial'].includes(prediction?.spec?.representation)) {
      issue('unsupported','inputs.design.prediction','该表示尚无独立交付核验器。仍可检查原设计及其导出清单。');
    } else if (config.target==='workspace_controller' && !metadata.design.parents.some(p=>p.id===prediction.artifact_id)) {
      issue('clarify','inputs.design.prediction.artifact_id','工作区控制器需要绑定已登记的数值预测器。请用该预测器生成设计，或选择独立 Python 包交付。');
    }
    for (const [key,axis] of Object.entries({x:'states',reference:'outputs',disturbance:'disturbances',previous:'controls'})) {
      const row=config.case?.[key], width=iface?.[axis]?.length;
      const required=key==='x'||key==='reference'||key==='disturbance'&&width>0||key==='previous'&&design.controller_config?.input_rate_bounds;
      if (row===undefined && required) issue('clarify','config.case.'+key,'请给出本次验收的有序信号值。');
      else if (row!==undefined && (!Array.isArray(row)||row.length!==width)) issue('unsupported','config.case.'+key,`应为 ${width} 维向量。`);
    }
    if (config.expected_action && config.expected_action.length!==iface?.controls?.length) issue('unsupported','config.expected_action','预期控制向量必须具有相同的控制通道数。');
  }
  const physical = Object.entries(metadata).filter(([,m])=>Array.isArray(m.interface?.states));
  const first = physical[0]?.[1].interface;
  for (const [key,m] of physical.slice(1)) {
    for (const axis of ['sample_time','time_semantics','states','controls','outputs','disturbances']) {
      if (!samePhysical(first[axis],m.interface[axis])) issue('unsupported',`inputs.${key}.interface.${axis}`,'物理接口的采样语义、信号顺序、单位或约束与所选上游不同。');
    }
  }
  const data = dataset?.interface;
  let widths;
  if (['train_supervised','train_external_imitation'].includes(op) && data) widths=[data.input_dim,data.output_dim];
  if (op === 'train_predictive_model' && data) widths=[data.states.length+data.controls.length+data.disturbances.length,data.states.length];
  const net = op === 'train_supervised' ? config : config.network ?? {};
  for (const [i,key] of ['input_dim','output_dim'].entries()) {
    if (widths?.[i] !== undefined && net[key] !== undefined && widths[i] !== net[key]) issue('unsupported',`config.${op==='train_supervised'?'':'network.'}${key}`,`按已登记特征顺序需要 ${widths[i]}，收到 ${net[key]}。`);
  }
  if (op === 'verify_design_delivery' && metadata.design?.source?.model_instantiated === false) issue('clarify','inputs.design','当前设计仍为模板。请补齐物理参数、实例化预测模型后再验收。');
  return issues;
}

function samePhysical(a,b) {
  if (a===b) return true;
  if (a==null || b==null || typeof a!=='object' || typeof b!=='object' || Array.isArray(a)!==Array.isArray(b)) return false;
  const keys=Object.keys(a);
  return keys.length===Object.keys(b).length && keys.every(k=>Object.hasOwn(b,k) && samePhysical(a[k],b[k]));
}
