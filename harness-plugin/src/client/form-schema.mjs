/** Browser form metadata. The Host remains the only confirmation authority. */
import {operations} from '../catalog.mjs';
import {agents} from '../roles.mjs';

export {operations};
export const roles=[
  {id:'model',title:'模型与数据',sub:'数据 NN · 单对象预测 · 条件 PINN'},
  {id:'controller',title:'控制器设计',sub:'仅设计 · 目标调参 · 独立验证'},
  {id:'lightweight',title:'控制器轻量化',sub:'外部专家 · 模仿 · 可选 PGAC'},
];
export const labels={inspect_workspace:'查看工作区',inspect_artifact:'检查产物与报告',read_artifact_file:'读取产物文件',prepare_handoff:'准备下一阶段',
  convert_plant_predictor:'从采样对象核验预测模型',verify_design_delivery:'验收控制器交付',
  check_matlab_design:'在专用 MATLAB 核验导出控制器',
  read_report_arrays:'读取已保存曲线数据',prepare_bundle_archive:'准备完整导出 ZIP',inspect_table:'查看 CSV / Excel 实际表头',
  inspect_material:'检查本地材料',register_material:'登记路径 / 上传材料',prepare_supervised_dataset:'配置 CSV / Excel 数据',train_supervised:'纯数据 NN 训练',
  register_plant:'登记 Python / MATLAB / SLX 对象',recommend_sampling:'生成可编辑采样建议',sample_model:'单对象试跑 / 全量采样',train_predictive_model:'学习离散下一状态模型',
  prepare_pinn_problem:'准备条件热 ODE 问题',train_pinn:'条件 ODE PINN 训练',synthesize_mpc:'仅设计 MPC / NMPC',tune_controller:'按目标调优控制器',
  register_expert:'登记外部专家',collect_external_expert:'原专家与对象闭环采数',import_expert_data:'导入已有专家 CSV / Excel',train_external_imitation:'训练外部专家模仿策略',
  prepare_imitation_evaluation:'准备原对象复验工况',evaluate_external_imitation:'原对象专家 / 模仿对比',validate_differentiable_model:'验证可微预测器',
  optimize_external_policy:'从实际模仿权重微调 PGAC',evaluate_external_pgac:'原对象三版本接受 / 回退',
  inspect_model_source:'检查旧对象材料',register_model:'登记声明式 Torch 模型',register_python_model:'登记独立 Torch 动力学',
  import_transitions:'导入旧转移 NPZ',simulate_transitions:'双模型生成转移',train_residual:'名义模型残差训练',identify_linear_model:'旧双模型线性辨识',
  design_controller:'旧设计与调参完整流程',validate_estimator:'KF / EKF 测量反馈验证',collect_expert:'旧内部控制器专家采数',train_imitation:'旧内部专家模仿',optimize_policy:'旧内部策略物理梯度',
  import_artifact:'导入已有产物及依赖',recover_interrupted_stage:'校验未发布检查点',read_case_plan:'读取工程计划',propose_case_plan:'提案工程计划'};
const groups={
  model:[['材料与数据',['inspect_workspace','inspect_material','register_material','prepare_supervised_dataset','train_supervised']],
    ['单对象与预测',['register_plant','recommend_sampling','sample_model','convert_plant_predictor','train_predictive_model']],['条件 ODE PINN',['prepare_pinn_problem','train_pinn']],
    ['兼容旧流程',['inspect_model_source','register_model','register_python_model','import_transitions','simulate_transitions','train_residual','identify_linear_model']]],
  controller:[['设计与调参',['inspect_workspace','convert_plant_predictor','synthesize_mpc','verify_design_delivery','check_matlab_design','tune_controller','validate_estimator']],['材料与对象',['inspect_material','register_material','register_plant','recommend_sampling','sample_model']],['兼容旧流程',['design_controller']]],
  lightweight:[['外部专家与模仿',['inspect_workspace','register_material','register_expert','collect_external_expert','import_expert_data','train_external_imitation','prepare_imitation_evaluation','evaluate_external_imitation']],
    ['可选受限 PGAC',['register_python_model','train_predictive_model','validate_differentiable_model','optimize_external_policy','evaluate_external_pgac']],
    ['兼容旧流程',['collect_expert','train_imitation','optimize_policy']]],
};
export function operationGroups(role){const allowed=new Set(agents[role]?.operations??[]),seen=new Set();const result=(groups[role]??[]).map(([title,names])=>({title,names:names.filter(name=>allowed.has(name)&&!seen.has(name)&&seen.add(name))}));
  result.push({title:'产物与工具',names:[...allowed].filter(name=>!seen.has(name)&&!['propose_case_plan','read_case_plan'].includes(name))});return result.filter(row=>row.names.length);}
export const uploadAccept='.py,.m,.slx,.mdl,.csv,.xlsx,.json,.npz';
export const parseObject=text=>{const value=typeof text==='string'?JSON.parse(text):text;if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('请输入 JSON 对象');
  const finite=v=>{if(typeof v==='number'&&!Number.isFinite(v))throw new Error('JSON 中的数值必须有限');if(v&&typeof v==='object')Object.values(v).forEach(finite);};finite(value);return value;};
export const pretty=value=>JSON.stringify(value,null,2);
const ordered=value=>Array.isArray(value)?value.map(ordered):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,ordered(value[key])])):value;
export const signature=value=>JSON.stringify(ordered(value));
export function workspaceKey(path,role){const clean=String(path??'').trim().replace(/\\/g,'/').replace(/\/+$/,'');return `${/^[A-Za-z]:/.test(clean)?clean.toLowerCase():clean}::${role}`;}
export const initialDraft=()=>({operation:'inspect_workspace',inputs:'{}',config:'{}',chat:'',jobId:'',attemptId:'',sessionId:'',mode:'normal',review:null});
export const resumeFields=['resume_checkpoint','resume_checkpoint_sha256','resume_run'];
export function summaryConfig(config){const result=structuredClone(config);delete result.plan_refs;for(const name of resumeFields)delete result[name];return result;}
export function requestSignature(request){const config=summaryConfig(request.config??{});if(request.operation==='sample_model')delete config.sampling_run;return signature({...request,config});}
export function setPath(value,path,next){const result=structuredClone(value),parts=path.split('.');let target=result;for(const part of parts.slice(0,-1)){if(!target[part]||typeof target[part]!=='object'||Array.isArray(target[part]))target[part]={};target=target[part];}
  if(next===undefined)delete target[parts.at(-1)];else target[parts.at(-1)]=next;delete result.plan_refs;return result;}
export function applySummary(config,payload){const result=structuredClone(parseObject(payload));delete result.plan_refs;for(const name of resumeFields){delete result[name];if(Object.hasOwn(config,name))result[name]=config[name];}return result;}
export function normalizedResult(value){if(typeof value==='string'){try{value=JSON.parse(value);}catch{return null;}}return value&&typeof value==='object'&&!Array.isArray(value)?value:null;}
export function reportReadOperation(name,config={}){return name.toLowerCase().endsWith('.npz')&&['keys','max_rows','filter'].some(key=>Object.hasOwn(config,key))?'read_report_arrays':'read_artifact_file';}
export function parseNumericInput(raw,validity={}){
  if(validity.badInput||validity.typeMismatch||validity.rangeUnderflow||validity.rangeOverflow||validity.stepMismatch)return {accepted:false,error:'数值无法解析或超出此字段允许范围；原配置保留，请修正后再执行'};
  if(raw==='')return {accepted:true,value:undefined,empty:true};
  const value=Number(raw);return Number.isFinite(value)?{accepted:true,value,empty:false}:{accepted:false,error:'数值必须有限；原配置保留，请修正后再执行'};
}

const nn={hidden_dims:[32,32],activation:'tanh',epochs:150,learning_rate:.001,batch_size:64,patience:30,seed:42,weight_decay:0,min_delta:0,grad_clip:5};
const emptyData=()=>({schema_version:1,interface_kind:'tabular_supervised',name:'用户数据接口',task:'static_regression',signals:[],features:[],targets:[]});
const split={mode:'trip',train:.6,validation:.2,test:.2};
const defaults={
  convert_plant_predictor:{inputs:{plant:'',dataset:''},config:{method:'declared_equations'}},
  verify_design_delivery:{inputs:{design:''},config:{target:'standalone_python',case:{}}},
  check_matlab_design:{inputs:{design:''},config:{case:{x:[],reference:[]}}},
  register_material:{inputs:{local_path:''},config:{source_kind:'path',source_metadata:{description:''}}},inspect_material:{inputs:{local_path:''},config:{}},inspect_table:{inputs:{data_path:''},config:{preview_rows:5}},
  prepare_supervised_dataset:{inputs:{material:''},config:{interface:emptyData(),split,scaler:'standard',source:{description:''}}},
  train_supervised:{inputs:{dataset:''},config:nn},register_plant:{inputs:{material:''},config:{adapter:{schema_version:1,backend:'python_callable',source_file:'',entrypoint:'',parameters:{},source_semantics:'transition'}}},
  recommend_sampling:{inputs:{model:''},config:{overrides:{}}},sample_model:{inputs:{model:''},config:{sampling:{},trial:{trajectories:1,steps:3},sampling_run:'trial'}},
  train_predictive_model:{inputs:{dataset:''},config:{network:nn,C:[],scaler:'standard'}},
  prepare_pinn_problem:{inputs:{material:''},config:{}},train_pinn:{inputs:{problem:''},config:{network:{...nn,input_dim:3,output_dim:1},conditional_parameterization:'free_mlp',loss_weights:{data:1,physics:1,initial:10},ablation:false,acceptance:null}},
  synthesize_mpc:{inputs:{description:''},config:{template:'generic',prediction:{},controller:{horizon:8,max_iterations:80,solver_tolerance:1e-6},weights:{q:1,r:.01,s:.01}}},
  tune_controller:{inputs:{model:'',evaluation_model:''},config:{budget:8,seed:42,tuning_scenarios:[],verification_scenarios:[],test_scenarios:[],evaluation:{recipe:'tracking',scales:{}}}},
  register_expert:{inputs:{material:''},config:{adapter:{schema_version:1,backend:'python_callable',source_file:'',entrypoint:'',parameters:{},time_unit:'s',observations:[],initial_memory:{},action_mapping:[],failure_policy:'stop'}}},
  collect_external_expert:{inputs:{expert:'',plant:''},config:{scenarios:[],seed:42,scaler:'standard'}},
  import_expert_data:{inputs:{material:''},config:{interface:emptyData(),split,scaler:'standard',source:{description:''},action_constraints:{bounds:[],rate_bounds:null,rate_semantics:'per_control_step'}}},
  train_external_imitation:{inputs:{dataset:''},config:{network:nn,action_metric_scale:'train_std'}},prepare_imitation_evaluation:{inputs:{policy:''},config:{}},
  evaluate_external_imitation:{inputs:{policy:''},config:{scenarios:[]}},validate_differentiable_model:{inputs:{model:'',evaluation_model:'',dataset:''},config:{mode:'exact_rewrite',partition:'validation',steps:50,probes:[]}},
  optimize_external_policy:{inputs:{policy:'',model:'',validation:''},config:{epochs:20,learning_rate:.0003,batch_size:24,horizon:4,seed:42,q:1,r:.01,s:.01,imitation_weight:.1,constraint_weight:100,domain:{x:[],u:[],d:[]},original_evaluation:{scenarios:[],evaluation:{q:1,r:.01,s:.01,comparison_atol:1e-12}}}},
  evaluate_external_pgac:{inputs:{candidate:''},config:{scenarios:[],evaluation:{q:1,r:.01,s:.01,comparison_atol:1e-12}}},
  inspect_model_source:{inputs:{model_path:''},config:{}},register_model:{inputs:{model_path:''},config:{source:{kind:'user_nominal',description:''}}},
  register_python_model:{inputs:{model_path:''},config:{source:{kind:'user_nominal',description:''}}},import_artifact:{inputs:{artifact_path:''},config:{}},
  import_transitions:{inputs:{nominal_model:'',data_path:''},config:{source:{description:''},seed:42}},simulate_transitions:{inputs:{nominal_model:'',reference_model:''},config:{trajectories:20,steps:24,seed:42}},
  train_residual:{inputs:{nominal_model:'',dataset:''},config:{hidden_dims:[32,32],epochs:150,learning_rate:.001,batch_size:64,patience:30,seed:42,weight_decay:0}},design_controller:{inputs:{model:'',evaluation_model:''},config:{seed:42,budget:8,tuning_scenarios:[],verification_scenarios:[],test_scenarios:[]}},
  collect_expert:{inputs:{controller:'',evaluation_model:''},config:{scenarios:[],seed:42}},train_imitation:{inputs:{dataset:''},config:{architecture:'mlp',hidden_dims:[64,32],learning_rate:.001,epochs:150,seed:42}},
  optimize_policy:{inputs:{policy:'',model:''},config:{learning_rate:.0001,epochs:20,seed:42}},identify_linear_model:{inputs:{nominal_model:'',dataset:''},config:{}},
  validate_estimator:{inputs:{model:'',controller:'',evaluation_model:''},config:{initial_estimate:[],verification_scenarios:[],test_scenarios:[],seed:42}},
};
export function operationDefaults(name){return structuredClone(defaults[name]??{inputs:{},config:{}});}
export function inputLabel(name){return {equations:'已登记的显式方程模型 ID',expected_artifact_id:'交接产物 ID',expected_project_id:'来源项目 ID',expected_metadata_hash:'来源版本摘要',design:'已导出的控制器设计产物 ID',material:'材料产物 ID',model:'预测 / 对象模型 ID',dataset:'数据集产物 ID',policy:'策略产物 ID',validation:'可微模型验证报告 ID',candidate:'待原对象复验候选 ID',
  expert:'外部专家产物 ID',plant:'原对象产物 ID',controller:'控制器产物 ID',evaluation_model:'原对象 / 评价模型 ID',nominal_model:'名义模型 ID',reference_model:'参考模型 ID',problem:'条件 ODE 问题 ID',
  local_path:'材料绝对路径',model_path:'模型源文件绝对路径',data_path:'转移数据绝对路径',artifact_path:'已有产物目录绝对路径',description:'设计任务描述'}[name]??name;}
export const networkOps=new Set(['train_supervised','train_predictive_model','train_external_imitation','train_pinn','train_residual','train_imitation']);
export const trainingOps=new Set([...networkOps,'optimize_external_policy','optimize_policy']);
export function networkPath(operation){return ['train_predictive_model','train_external_imitation','train_pinn'].includes(operation)?'network':'';}
export function networkConfig(config,operation){return networkPath(operation)?config.network??{}:config;}
export const checkpointOperations={residual_training:'train_residual',expert_imitation:'train_imitation',physical_policy_optimization:'optimize_policy',
  supervised_training:'train_supervised',conditional_ode_pinn:'train_pinn',external_physical_policy_optimization:'optimize_external_policy'};
