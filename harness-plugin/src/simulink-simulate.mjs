import { join } from 'node:path';
import { lstat } from 'node:fs/promises';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { matlabString, runControlledModelHelper } from './simulink-native.mjs';
import { verifySavedFile, withSessionSlxTarget } from './simulink-save.mjs';
import { quarantineRecovery } from './matlab-resource.mjs';

/** Simulate the verified workspace file, with numeric SimulationInput overrides only. */
export function simulateModelTool(ctx,config,preflight) {
  return defineTool({name:'control_simulate_simulink_model',
    description:'Compile and simulate the saved SLX in this chat workspace. Uses numeric SimulationInput overrides without editing the file. Returns the failing phase, original MATLAB error and nested causes, plus dedicated process and idle verification.',
    parameters:{model:{type:'string',required:true},stopTime:{type:'number',description:'Optional nonnegative stop time'},
      fixedStep:{type:'number',description:'Optional positive numeric fixed step'},
      solver:{type:'string',description:'Optional Simulink solver name, e.g. FixedStepDiscrete'},
      solverType:{type:'string',enum:['Fixed-step','Variable-step']}},
    output:{schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]},
    async execute({model,stopTime,fixedStep,solver,solverType},execution) {
      let observed, path, stage='validation', exists=null;
      try {
        if (stopTime !== undefined && (!Number.isFinite(stopTime) || stopTime<0) ||
            fixedStep !== undefined && (!Number.isFinite(fixedStep) || fixedStep<=0)) throw new Error('仿真结束时间需为非负数，固定步长需为正数。');
        if (solver!==undefined && !/^[A-Za-z][A-Za-z0-9_]{0,62}$/.test(solver)) throw new Error('solver 需要有效的求解器名。');
        return await withSessionSlxTarget(ctx,config,model,execution,preflight,'simulate_simulink_model',async target=>{
          path=target;
          try { await lstat(target); exists=true; }
          catch (error) { if (error.code==='ENOENT') { exists=false; throw new Error('工作区没有该模型的已保存 SLX。'); } throw error; }
          stage='dispatch';
          const expressions=[matlabString(model),matlabString(target),matlabString(execution.agent.session.id),
            stopTime === undefined ? '[]' : String(stopTime), fixedStep === undefined ? '[]' : String(fixedStep),
            matlabString(solver ?? ''),matlabString(solverType ?? '')];
          observed=await runControlledModelHelper(ctx.tools,join(config.codeRoot,'matlab'),execution.agent,execution.signal,
            'control_harness_simulate_model',expressions,{model,target,validate:r=>{
              if (r.status==='simulated' && (r.verified.join(',')!=='validation,load,compile,simulate' ||
                  !Number.isSafeInteger(r.default_samples) || r.default_samples < 1)) throw new Error('缺少本次已保存模型的载入、编译、仿真或采样证据。');
            }});
          if (observed.status==='failed') throw new Error(observed.reason);
          stage='host_file_check';
          return {...observed,...await verifySavedFile(target,observed.bytes),file_created_by_operation:false};
        });
      } catch (error) {
        const r=observed ?? error.modelReceipt;
        const quarantined=error.quarantined===true;
        if (path) {
          try { exists=(await lstat(path)).isFile(); }
          catch (statError) { exists=statError.code==='ENOENT' ? false : null; }
        }
        throw new Error(JSON.stringify({...r,model,path,file_exists:exists,file_created_by_operation:false,
          status:quarantined?'quarantined':r?.status==='failed'?'failed':error.dispatched===false?'not_dispatched':'failed',
          stage:error.diagnostic?'workspace_preflight':stage==='host_file_check'?stage:r?.stage ?? (error.dispatched===false?'admission':stage),
          verified:r?.verified ?? [],changes:r?.changes ?? [],reason:r?.reason || error.message,
          diagnostic:error.diagnostic,verification_error:quarantined?error.message:undefined,
            retryable:!quarantined && r?.retryable===true,session_verified:r?.session_verified===true || error.sessionVerified===true,
          recovery:quarantined?quarantineRecovery:'请核对原始诊断、已保存模型和验证阶段，修正后再执行受控仿真。'}));
      }
    },
  });
}
