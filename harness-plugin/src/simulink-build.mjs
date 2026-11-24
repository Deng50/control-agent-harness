import { lstat } from 'node:fs/promises';
import { join } from 'node:path';
import { readWorkspaceBytes } from './workspace-file.mjs';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { matlabString, runControlledModelHelper } from './simulink-native.mjs';
import { matrixLiteral } from './simulink-plant.mjs';
import { verifySavedFile, withSessionSlxTarget } from './simulink-save.mjs';
import { quarantineRecovery } from './matlab-resource.mjs';

const vector = {type:'array', items:{type:'number'}};
const matrix = {type:'array', items:vector};
const maxReferenceBytes = 2 * 1024 * 1024;
const numericVector = (v, size) => Array.isArray(v) && v.length === size && v.every(Number.isFinite);

class ModelOperationFailure extends Error {
  constructor(receipt) { super(JSON.stringify(receipt)); }
}

export async function readPlantReference(workspace, name) {
  if (typeof name !== 'string' || !/^[\p{L}\p{N}_-][\p{L}\p{N}_. -]*\.json$/iu.test(name) ||
      /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\./i.test(name)) throw new Error('referenceFile 需要当前工作目录中的单个 JSON 文件名。');
  return JSON.parse((await readWorkspaceBytes(workspace,name,maxReferenceBytes)).toString('utf8'));
}

function referenceArgument(reference, n, m, sampleTime) {
  if (reference === undefined) return matlabString('{}');
  if (!Number.isFinite(reference?.absTolerance) || reference.absTolerance <= 0 ||
      (reference.sampleTime !== undefined && reference.sampleTime !== sampleTime) ||
      !Array.isArray(reference.trajectories) || !reference.trajectories.length) {
    throw new Error('参考轨迹需要正数 absTolerance、与模型相同的采样周期以及至少一条 trajectories。');
  }
  const trajectories = reference.trajectories.map(({initialCondition, inputs, states}) => {
    if (!numericVector(initialCondition,n) || !Array.isArray(inputs) || !inputs.length || !inputs.every(row => numericVector(row,m)) ||
        !Array.isArray(states) || states.length !== inputs.length + 1 || !states.every(row => numericVector(row,n)) ||
        !states[0].every((v,i) => v === initialCondition[i])) throw new Error('每条参考轨迹需要初态、K 行有序输入和 K+1 行状态，首行必须与初态相同。');
    return {initialCondition,inputs,states};
  });
  const json = JSON.stringify({sampleTime,absTolerance:reference.absTolerance,trajectories});
  if (Buffer.byteLength(json) > maxReferenceBytes) throw new Error('参考轨迹超过 2 MiB。');
  return matlabString(json);
}

function verifySimulation(receipt, reference) {
  if (receipt.status !== 'simulated') return;
  const expected = ['validation','create','configure','wire','save','reload','simulate'];
  if (reference) expected.push('compare');
  const c = receipt.comparison;
  if (expected.some((stage,i) => receipt.verified[i] !== stage) ||
      !Number.isSafeInteger(receipt.default_samples) || receipt.default_samples < 2 || c?.compared !== Boolean(reference)) {
    throw new Error('缺少与本次请求对应的 Simulink 保存、重载或仿真验证步骤。');
  }
  if (reference && (c.trajectories !== reference.trajectories.length ||
      c.samples !== reference.trajectories.reduce((n,t) => n+t.states.length,0) || c.tolerance !== reference.absTolerance ||
      !Number.isFinite(c.max_abs_error) || c.max_abs_error < 0 || c.max_abs_error > c.tolerance)) {
    throw new Error('数值对照回执中的轨迹数、采样数、误差或容差不符。');
  }
}

async function fileExists(path) {
  try { return (await lstat(path)).isFile(); }
  catch (error) { if (error.code === 'ENOENT') return false; return null; }
}

export async function buildZohPlant(ctx, config, args, execution, preflight, verifyReference) {
  const {model, Ac, Bc, initialCondition, sampleTime, inputNames, reference, referenceFile, parameters} = args;
  const chatId = execution.agent?.session?.id;
  if (typeof chatId !== 'string' || !chatId || /[\r\n\0]/.test(chatId)) throw new Error('需要当前对话的身份才能创建或继续部分模型。');
  const n = Ac?.length, m = inputNames?.length;
  if (!Array.isArray(inputNames) || !m || !inputNames.every(s => typeof s === 'string' && /^[A-Za-z][A-Za-z0-9_]{0,62}$/.test(s)) ||
      new Set(inputNames).size !== m) throw new Error('inputNames 需要互异的输入端口标识符，按控制量、扰动量的顺序。');
  const a = matrixLiteral('Ac',n,n,Ac), b = matrixLiteral('Bc',n,m,Bc);
  if (!numericVector(initialCondition,n) || !Number.isFinite(sampleTime) || sampleTime <= 0) throw new Error('请提供完整初态和正数采样周期。');
  if (reference !== undefined && referenceFile !== undefined) throw new Error('reference 与 referenceFile 只能选择一项。');
  return withSessionSlxTarget(ctx, config, model, execution, preflight, 'build_zoh_plant', async (target,workspace) => {
    const ref = referenceFile === undefined ? reference : await readPlantReference(workspace, referenceFile);
    await verifyReference?.(execution.agent,referenceFile,ref);
    // Python case files can contain other settings. Only finite scalar physical
    // parameters are copied to MATLAB; model expressions never enter this interface.
    const physical = {};
    for (const source of [ref?.parameters, parameters]) {
      if (source === undefined) continue;
      if (!source || typeof source !== 'object' || Array.isArray(source)) throw new Error('物理参数需要命名数值对象。');
      for (const [key,value] of Object.entries(source)) {
        if (!/^[A-Za-z][A-Za-z0-9_]{0,62}$/.test(key) || !Number.isFinite(value)) {
          if (source === parameters) throw new Error('parameters 只接受 MATLAB 标识符和有限数值标量。');
          continue;
        }
        if (Object.hasOwn(physical,key) && physical[key] !== value) throw new Error('物理参数与 Python 源回放不一致：' + key);
        physical[key] = value;
      }
    }
    const expressions = [matlabString(model), matlabString(target), a, b, '[' + initialCondition.join(';') + ']',
      String(sampleTime), '{' + inputNames.map(matlabString).join(',') + '}', referenceArgument(ref,n,m,sampleTime),
      matlabString(chatId), matlabString(JSON.stringify(physical))];
    const existedBefore = await fileExists(target);
    let receipt;
    try {
      receipt = await runControlledModelHelper(ctx.tools, join(config.codeRoot,'matlab'), execution.agent, execution.signal,
        'control_harness_build_zoh_plant', expressions, {model,target,validate:result=>verifySimulation(result,ref)});
      if (receipt.status === 'simulated') {
        const file = await verifySavedFile(target, receipt.bytes);
        return {...receipt,...file,file_created_by_operation:true};
      }
    } catch (error) {
      const observed = receipt ?? error.modelReceipt;
      const exists = await fileExists(target);
      throw new ModelOperationFailure({...observed, model, path:target, file_exists:exists,
        file_created_by_operation:exists && !existedBefore && error.dispatched !== false ? (observed?.verified?.includes('save') ? true : null) : false,
        status:error.quarantined ? 'quarantined' : error.dispatched === false ? 'not_dispatched' : error.sessionVerified ? 'failed' : 'interrupted',
        stage:receipt ? 'host_file_check' : observed?.stage ?? (error.dispatched === false ? (error.quarantined ? 'resource_precheck' : 'admission') : 'dispatch'),
        verified:observed?.verified ?? [], changes:observed?.changes ?? [], reason:observed?.reason || error.message,
        verification_error:error.message, retryable:false, session_verified:receipt?.session_verified === true || error.sessionVerified === true,
        recovery:error.quarantined ? quarantineRecovery : error.sessionVerified || receipt?.session_verified
          ? 'MATLAB 可继续使用。先核对当前模型和已有文件，只纠正失败步骤。'
          : '请核对上方的文件或工具权限诊断。'});
    }
    const exists = await fileExists(target);
    const preserve = exists || receipt.identifier === 'ControlHarness:BuildTarget';
    const recovery = exists && receipt.verified.includes('save')
      ? '文件已保存。保留该版本，按已核验的绝对路径读回模型并纠正失败的验证步骤。'
      : preserve
      ? '专用进程身份与空闲状态已核验。请核对目标路径和已验证阶段，保留已有文件或同名模型，纠正输入后用新模型名重建。'
      : '专用进程身份与空闲状态已核验。请根据原始诊断修正输入，并在同一对话中读回已有模型后继续。';
    throw new ModelOperationFailure({...receipt, file_exists:exists,
      file_created_by_operation:exists && !existedBefore ? (receipt.verified.includes('save') ? true : null) : false, recovery});
  });
}

export function buildZohPlantTool(ctx,config,preflight,verifyReference) {
  return defineTool({
    name:'control_build_zoh_plant',
    description:'Build a discrete Simulink plant from inspected continuous LTI dynamics using exact zero-order hold. Supply case or user parameters, ordered control then disturbance inputs, initial state and period. Saves to this conversation, closes, reloads and simulates the SLX. For an uploaded Python source, use control_replay_plant_source and pass its returned referenceFile for independent sample comparison.',
    parameters:{
      model:{type:'string', required:true, description:'Unique MATLAB model name'},
      Ac:{...matrix, required:true, description:'Continuous state matrix, n × n'},
      Bc:{...matrix, required:true, description:'Continuous input matrix, n × m, control then disturbance columns'},
      initialCondition:{...vector,required:true}, sampleTime:{type:'number',required:true},
      inputNames:{type:'array',items:{type:'string'},required:true},
      parameters:{type:'object',additionalProperties:true,description:'Optional named finite scalar physical parameters from the user or case; must agree with Python replay when supplied'},
      referenceFile:{type:'string',description:'Optional reference JSON filename in the current conversation workspace (up to 2 MiB)'},
      reference:{type:'object',additionalProperties:false,description:'Alternatively supply trajectories inline',properties:{sampleTime:{type:'number'},absTolerance:{type:'number'},
        parameters:{type:'object',additionalProperties:true},source_sha256:{type:'string'},
        trajectories:{type:'array',items:{type:'object',additionalProperties:false,properties:{initialCondition:vector,inputs:matrix,states:matrix}}}}},
    },
    output:{schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]},
    async execute(args,execution) {
      try { return await buildZohPlant(ctx,config,args,execution,options=>preflight(options,execution),verifyReference); }
      catch (error) {
        if (error instanceof ModelOperationFailure) throw error;
        throw new ModelOperationFailure({status:'failed', stage:error.diagnostic ? 'workspace_preflight' : 'validation',
          verified:[], changes:[], file_exists:null, model:args?.model, reason:error.message, retryable:false,
          diagnostic:error.diagnostic, ...(error.diagnostic?{preflight_observation:'fresh',work_started:false}:{}),
          recovery:'尚未取得本次模型派发回执。请按诊断核对输入、参考文件和当前对话工作目录。'});
      }
    },
  });
}
