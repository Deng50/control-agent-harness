import React,{createContext,useContext,useEffect,useId,useState} from 'react';
import {networkOps,networkConfig,networkPath,operations,parseNumericInput,parseObject,pretty} from './form-schema.mjs';

const FieldValidation=createContext(null);

export function JsonField({label,value,onChange,onInvalid,rows=3,placeholder}){
  const valueText=value===undefined?'':pretty(value),[text,setText]=useState(valueText),[error,setError]=useState('');
  useEffect(()=>{setText(valueText);setError('');onInvalid?.(label,'');},[valueText]);
  return <label>{label}<textarea rows={rows} value={text} placeholder={placeholder??'填写实际 JSON；留空表示未声明'} spellCheck={false} onChange={event=>{
    const next=event.target.value;setText(next);try{const parsed=next.trim()?JSON.parse(next):undefined;parseObject({value:parsed});onChange(parsed);setError('');onInvalid?.(label,'');}catch(failure){setError('JSON 尚未完整或数值无效，暂不能执行');onInvalid?.(label,failure.message);}
  }}/>{error&&<small role="alert">{error}</small>}</label>;
}
export function NumericField({label,value,onChange,min,step='any'}){
  const notify=useContext(FieldValidation),id=useId(),valueText=value===undefined?'':String(value),[text,setText]=useState(valueText),[error,setError]=useState('');
  useEffect(()=>{setText(valueText);setError('');notify?.(id,'');},[valueText]);
  useEffect(()=>()=>notify?.(id,''),[]);
  return <label>{label}<input type="number" min={min} step={step} value={text} aria-invalid={Boolean(error)} onChange={event=>{
    const raw=event.target.value,next=parseNumericInput(raw,event.target.validity);setText(raw);
    if(!next.accepted){setError(next.error);notify?.(id,next.error);return;}
    setError('');notify?.(id,'');onChange(next.value);
  }}/>{error&&<small role="alert">{error}</small>}</label>;
}
export function Choice({label,value,values,onChange}){return <label>{label}<select value={value??''} onChange={event=>onChange(event.target.value||undefined)}><option value="">请选择</option>{values.map(row=><option key={typeof row==='string'?row:row[0]} value={typeof row==='string'?row:row[0]}>{typeof row==='string'?row:row[1]}</option>)}</select></label>;}

export function NetworkFields({operation,config,set,onInvalid}){
  if(!networkOps.has(operation))return null;
  const prefix=networkPath(operation),n=networkConfig(config,operation),path=name=>prefix?prefix+'.'+name:name;
  const allowed=prefix?null:operations[operation]?.configSchema?.properties??{},has=name=>!allowed||Object.hasOwn(allowed,name),recurrent=operation==='train_imitation'&&['rnn','lstm'].includes(n.architecture);
  return <fieldset><legend>实际网络与训练参数</legend><div className="control-parameters">
    {!recurrent&&<JsonField label="隐藏层节点数 hidden_dims" value={n.hidden_dims} onChange={v=>set(path('hidden_dims'),v)} onInvalid={onInvalid} rows={2} placeholder="例如 [32,16]；[] 为线性网络"/>}
    {has('activation')&&<Choice label="激活函数" value={n.activation} values={operation==='train_pinn'?['tanh','silu']:['tanh','relu','silu']} onChange={v=>set(path('activation'),v)}/>}
    {[[ 'learning_rate','学习率',0],['epochs','目标总轮次',1],['batch_size','批量大小',1],['seed','随机种子',0]].map(([name,label,min])=><NumericField key={name} label={label} value={n[name]} min={min} onChange={v=>set(path(name),v)}/>)}
    {[['patience','早停耐心轮次',1],['weight_decay','权重衰减',0],['min_delta','验证改善阈值',0],['grad_clip','梯度裁剪阈值',0],['input_dim','输入特征宽度',1],['output_dim','输出目标宽度',1]].filter(([name])=>has(name)).map(([name,label,min])=><NumericField key={name} label={label} value={n[name]} min={min} onChange={v=>set(path(name),v)}/>)}
    {operation==='train_imitation'&&<Choice label="旧内部专家网络" value={n.architecture} values={['mlp','rnn','lstm']} onChange={v=>set(path('architecture'),v)}/>}
    {recurrent&&[['hidden_size','循环隐藏宽度'],['num_layers','循环层数'],['preview_len','显式预览长度']].map(([name,label])=><NumericField key={name} label={label} value={n[name]} min={1} onChange={v=>set(path(name),v)}/>)}
  </div><small>这些字段写入同一份阶段配置；宽度必须符合实际 DataInterface，不代表物理状态维度。best 由验证集选择；目标轮次包含已恢复轮次。</small></fieldset>;
}

function ColumnEditor({label,column,onChange}){const c=column??{};return <div className="control-column-editor"><strong>{label}</strong><div className="control-parameters">
  <NumericField label="列位置（从 0 开始）" value={c.index} min={0} onChange={v=>onChange({...c,index:v})}/><label>该位置的完整表头<input value={c.header??''} onChange={e=>onChange({...c,header:e.target.value})}/></label>
  <label>Excel 工作表（如适用）<input value={c.worksheet??''} onChange={e=>onChange({...c,worksheet:e.target.value||null})}/></label>
</div></div>;}
export function DataInterfaceFields({value,onChange,external=false}){
  const i=value??{},signals=i.signals??[],features=i.features??[],targets=i.targets??[];
  const edit=(group,index,patch)=>onChange({...i,[group]:(i[group]??[]).map((row,k)=>k===index?{...row,...patch}:row)});
  const remove=(group,index)=>onChange({...i,[group]:(i[group]??[]).filter((_,k)=>k!==index)});
  return <fieldset><legend>数据列、因果特征与目标</legend><div className="control-parameters"><label>接口名称<input value={i.name??''} onChange={e=>onChange({...i,name:e.target.value})}/></label>
    <Choice label="数据任务" value={i.task} values={['static_regression','dynamic_prediction']} onChange={v=>onChange({...i,task:v})}/>
    <NumericField label="采样周期（秒，动态数据需声明）" value={i.sample_time} min={0} onChange={v=>onChange({...i,sample_time:v})}/></div>
    <p>按真实列位置和表头绑定；同名表头必须用位置区分。单位、可观测性与延迟由用户声明。</p>
    {signals.map((s,index)=><details key={index} open><summary>信号 {index+1} · {s.name||'待命名'}</summary><div className="control-parameters">
      <label>信号名<input value={s.name??''} onChange={e=>edit('signals',index,{name:e.target.value})}/></label><label>单位<input value={s.unit??''} onChange={e=>edit('signals',index,{unit:e.target.value})}/></label>
      <Choice label="角色" value={s.role} values={['state','control','disturbance','output','auxiliary']} onChange={v=>edit('signals',index,{role:v})}/>
      <NumericField label="信号延迟（步）" value={s.delay_steps} min={0} onChange={v=>edit('signals',index,{delay_steps:v})}/>
      <label className="control-checkbox"><input type="checkbox" checked={s.observable===true} onChange={e=>edit('signals',index,{observable:e.target.checked})}/>部署时可获得此信号</label>
      {external&&<Choice label="专家数据语义" value={s.expert_role} values={['observation','reference','disturbance','previous_action','expert_action','diagnostic_memory']} onChange={v=>edit('signals',index,{expert_role:v})}/>}
    </div><ColumnEditor label="原始数据列" column={s.column} onChange={v=>edit('signals',index,{column:v})}/><button type="button" onClick={()=>remove('signals',index)}>删除此信号</button></details>)}
    <button type="button" onClick={()=>onChange({...i,signals:[...signals,{name:'',role:'auxiliary',unit:'',column:{header:''},delay_steps:0,observable:true}]})}>添加数据列</button>
    <h4>输入特征（按网络输入顺序）</h4>{features.map((f,index)=><div className="control-mapping-row" key={index}>
      <Choice label="信号" value={f.signal} values={signals.map(s=>s.name).filter(Boolean)} onChange={v=>edit('features',index,{signal:v})}/>
      <NumericField label="历史窗口 window" value={f.window} min={1} onChange={v=>edit('features',index,{window:v})}/><NumericField label="额外滞后 lag" value={f.lag} min={0} onChange={v=>edit('features',index,{lag:v})}/><button type="button" onClick={()=>remove('features',index)}>删除</button>
    </div>)}<button type="button" onClick={()=>onChange({...i,features:[...features,{signal:'',window:1,lag:0}]})}>添加输入特征</button>
    <h4>训练目标（按网络输出顺序）</h4>{targets.map((t,index)=><div className="control-mapping-row" key={index}>
      <Choice label="目标信号" value={t.signal} values={signals.map(s=>s.name).filter(Boolean)} onChange={v=>edit('targets',index,{signal:v})}/>
      <NumericField label="目标前移 offset" value={t.offset} min={0} onChange={v=>edit('targets',index,{offset:v})}/><button type="button" onClick={()=>remove('targets',index)}>删除</button>
    </div>)}<button type="button" onClick={()=>onChange({...i,targets:[...targets,{signal:'',offset:0}]})}>添加训练目标</button>
    <details><summary>时间 / 轨迹分组列</summary>{[['time_column','时间列'],['trip_column','完整轨迹 / 分组列']].map(([key,label])=><ColumnEditor key={key} label={label} column={i[key]} onChange={v=>onChange({...i,[key]:v})}/>)}</details>
    <small>先按完整轨迹或时间拆分，再构造窗口与训练集缩放。专家目标为当前动作；历史动作必须有明确滞后 / 已是上一动作的来源声明。诊断记忆不能默认成为部署特征。</small>
  </fieldset>;
}

export function ParameterFields({operation,config,set,onInvalid}){
  const object=(path,label,rows=3)=><JsonField key={path} label={label} value={path.split('.').reduce((x,k)=>x?.[k],config)} onChange={v=>set(path,v)} onInvalid={onInvalid} rows={rows}/>;
  const number=(path,label,min=0)=><NumericField key={path} label={label} value={path.split('.').reduce((x,k)=>x?.[k],config)} min={min} onChange={v=>set(path,v)}/>;
  const text=(path,label)=><label key={path}>{label}<input value={path.split('.').reduce((x,k)=>x?.[k],config)??''} onChange={e=>set(path,e.target.value||undefined)}/></label>;
  const data=['prepare_supervised_dataset','import_expert_data'].includes(operation),adapter=['register_plant','register_expert'].includes(operation);
  return <FieldValidation.Provider value={onInvalid}>
    <small>数值留空表示未声明：有后端默认时采用默认，必填项需补全；坏输入会阻止执行，实际值以解析配置 / 报告为准。</small>
    <NetworkFields operation={operation} config={config} set={set} onInvalid={onInvalid}/>
    {data&&<><DataInterfaceFields value={config.interface} onChange={v=>set('interface',v)} external={operation==='import_expert_data'}/><div className="control-parameters">
      <Choice label="拆分方式" value={config.split?.mode} values={['trip','time']} onChange={v=>set('split.mode',v)}/>
      {number('split.train','训练比例')}{number('split.validation','验证比例')}{number('split.test','测试比例')}
      <Choice label="缩放（仅训练集拟合）" value={config.scaler} values={['standard','minmax','none']} onChange={v=>set('scaler',v)}/>
      <Choice label="数据实际来源 * 必填" value={config.source?.kind} values={operation==='import_expert_data'?['recorded_external_expert','declared_simulation']:['measured','synthetic','external_expert']} onChange={v=>set('source.kind',v)}/>
      {text('source.description','来源说明')}{text('worksheet','Excel 工作表')}{text('delimiter','CSV 分隔符')}{number('header_row','表头行（从 1 开始）',1)}
    </div>{operation==='import_expert_data'&&<><Choice label="上一动作的记录语义" value={config.source?.previous_action_semantics} values={['actual_applied_action_before_current_decision','not_recorded']} onChange={v=>set('source.previous_action_semantics',v)}/>{object('action_constraints','显式动作 / 增量边界')}{object('source_mapping','外置来源映射（如已有上一动作列）')}{object('physical_interface','可选独立物理接口；数据-only 可为 null')}{object('physical_mapping','可选物理映射')}</>}</>}
    {adapter&&<fieldset><legend>{operation==='register_plant'?'对象适配声明':'外部专家适配声明'}</legend><div className="control-parameters">
      <Choice label="适配后端" value={config.adapter?.backend} values={operation==='register_expert'?['python_callable','matlab_function']:['python_callable','python_factory','matlab_function','simulink']} onChange={v=>set('adapter.backend',v)}/>
      {text('adapter.source_file','快照内源文件，例如 source/model.py')}{text('adapter.entrypoint','明确入口函数')}
      {operation==='register_plant'&&<Choice label="源函数语义" value={config.adapter?.source_semantics} values={['transition','derivative']} onChange={v=>set('adapter.source_semantics',v)}/>}
      {operation==='register_expert'&&number('adapter.control_period','专家控制周期（秒）')}
    </div>{object('adapter.interface','有序物理接口：状态 / 控制 / 输出 / 扰动及单位边界',7)}{object('adapter.parameters','源函数参数')}
    {operation==='register_plant'?<>{object('adapter.C','状态到输出矩阵 C（不推断 identity）')}{object('adapter.integration','导数积分声明（仅 derivative）')}</>:<>{object('adapter.observations','明确可用观测映射')}{object('adapter.action_mapping','动作到物理执行器映射')}{object('adapter.initial_memory','逐轨重置的显式 JSON 记忆')}{object('adapter.input_rate_bounds','每控制步动作变化率边界')}
      <div className="control-parameters">{['reference_available','disturbance_available','previous_action_available'].map(name=><label className="control-checkbox" key={name}><input type="checkbox" checked={config.adapter?.[name]===true} onChange={e=>set('adapter.'+name,e.target.checked)}/>{name}</label>)}</div></>}
    <small>登记阶段不执行原材料。复杂 factory、导数单位与 SLX 端口 / SID / 初值变量映射请在高级 JSON 显式填写；任意工程解析与隐藏状态不在支持范围。</small></fieldset>}
    {operation==='sample_model'&&<fieldset><legend>实际采样计划与试跑</legend>{object('sampling','采样计划（先从推荐操作取得完整候选）',7)}<div className="control-parameters">{number('trial.trajectories','试跑前几条轨迹',1)}{number('trial.steps','试跑每条前几步',1)}<Choice label="本次执行范围" value={config.sampling_run} values={[['trial','已确认计划的固定试跑前缀'],['full','已确认的完整采样计划']]} onChange={v=>set('sampling_run',v)}/></div><small>切换试跑 / 全量复用同一计划确认；改变工况、步长或试跑上限会使对应版本失效。</small></fieldset>}
    {operation==='recommend_sampling'&&object('overrides','采样建议的显式覆盖参数')}
    {operation==='train_predictive_model'&&<>{object('C','显式输出映射 C')}{object('max_rollout_state_nrmse','预先声明的逐状态递推门限（可选）')}</>}
    {operation==='train_supervised'&&object('max_test_nrmse','预先声明的测试 NRMSE 门限（可选）')}
    {operation==='train_external_imitation'&&<>{object('action_metric_scale','动作指标尺度："train_std" 或显式正数列表')}{object('max_test_action_nrmse','预先声明的动作 NRMSE 门限（可选）')}</>}
    {operation==='train_pinn'&&<fieldset><legend>真实物理损失与能力范围</legend><Choice label="条件参数化" value={config.conditional_parameterization} values={['free_mlp','affine_initial_power']} onChange={v=>set('conditional_parameterization',v)}/><div className="control-parameters">{number('loss_weights.data','数据项权重')}{number('loss_weights.physics','物理残差权重')}{number('loss_weights.initial','初值项权重')}</div><label className="control-checkbox"><input type="checkbox" checked={config.ablation===true} onChange={e=>set('ablation',e.target.checked)}/>明确执行关闭物理项的对照</label>{object('acceptance','预先声明的温度 / 残差 / 初值门限（可选）')}<small>当前仅支持热过程条件 ODE 轨迹；关闭物理项会标记 ablation，不能作为任意 Plant step 交给 MPC。</small></fieldset>}
    {operation==='prepare_pinn_problem'&&<>{object('equation','热 ODE 方程模板与明确 SI 参数')}{object('domain','时间、初温、恒定功率条件域')}{object('normalization','明确的时间 / 温度 / 功率 / 物理残差尺度')}{object('conditions','完整条件及 train/validation/test 分组')}{text('reference_file','材料内参考 CSV')}{object('reference_columns','参考 CSV 列位置与表头')}{text('collocation_file','材料内固定训练配点 CSV')}{object('collocation_columns','配点 CSV 列位置与表头')}{object('units','各字段单位')}{object('source','参考数据实际来源')}</>}
    {['synthesize_mpc','tune_controller','design_controller'].includes(operation)&&<fieldset><legend>MPC 设计 / 调参参数</legend>
      {operation==='synthesize_mpc'&&<><Choice label="仅设计模板" value={config.template} values={['generic','spring_damper']} onChange={v=>set('template',v)}/>{object('parameters','用户提供的物理参数；缺失值保持参数化，不填示例值')}{object('prediction','预测器路线 / 显式方程 / 辨识或线性化声明')}
        <div className="control-parameters">{number('controller.horizon','预测时域',1)}{number('controller.max_iterations','单点求解最大迭代',1)}{number('controller.solver_tolerance','求解容差')}</div>{object('weights','内部归一化 q/r/s 权重')}{object('controller.input_rate_bounds','每步控制增量边界')}</>}
      {operation!=='synthesize_mpc'&&<><div className="control-parameters">{number('budget','实际调参候选预算',1)}{number('seed','随机种子')}</div>{object('tuning_scenarios','仅调参工况')}{object('verification_scenarios','独立验证工况')}{object('test_scenarios','最终测试工况')}{object('evaluation','外部评分配方、固定尺度与硬门限')}</>}
      <small>内部 q/r/s 与外部评分分别声明。仅设计可在未实例化模板处结束；单点求解不等于原对象闭环或稳定性验证。</small></fieldset>}
    {['collect_external_expert','evaluate_external_imitation','evaluate_external_pgac','collect_expert'].includes(operation)&&<>{object('scenarios','明确的逐轨工况及分区',7)}{operation.startsWith('evaluate_')&&object('benchmark','可选控制器计时配置（与对象 / MCP 耗时分开）')}{operation==='evaluate_external_pgac'&&object('evaluation','必须与训练前冻结的原对象 q/r/s 完全一致')}</>}
    {operation==='validate_differentiable_model'&&<><Choice label="模型资格路线" value={config.mode} values={['exact_rewrite','existing_torch','bounded_surrogate']} onChange={v=>set('mode',v)}/>{number('steps','自主递推步数',1)}{object('probes','明确 smooth / saturation / switch 探针')}{object('domain','明确 x/u/d 验证域')}{object('numeric','物理数值一致性容差')}{object('gradient','Jacobian / 差分步长与容差')}{config.mode==='bounded_surrogate'&&object('proxy_budget','代理近似误差预算')}</>}
    {['optimize_external_policy','optimize_policy'].includes(operation)&&<fieldset><legend>实际物理梯度优化</legend><div className="control-parameters">{number('learning_rate','学习率')}{number('epochs','目标总轮次',1)}{number('seed','随机种子')}{operation==='optimize_external_policy'&&<>{number('batch_size','批量大小',1)}{number('horizon','滚动预测时域',1)}{number('q','内部 q')}{number('r','内部 r')}{number('s','内部 s')}{number('imitation_weight','模仿锚点权重')}{number('constraint_weight','边界惩罚权重')}</>}</div>{operation==='optimize_external_policy'&&<>{object('domain','已合格模型范围内的 x/u/d 域')}{object('original_evaluation','训练前冻结的完整原对象验证 / 测试及评分',8)}</>}<small>从实际模仿权重开始，仅训练策略网络。候选数值完成仍须原对象接受 / 可行基线回退；无 Critic / PPO / SAC。</small></fieldset>}
  </FieldValidation.Provider>;
}
