import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { basename, dirname, isAbsolute, relative, resolve } from 'node:path';
import { lstat,realpath } from 'node:fs/promises';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { isControlledModelDispatch } from './matlab-resource.mjs';
import { replayTool } from './plant-replay.mjs';
import { plantWorkspacePolicy } from './workspace-policy.mjs';
import { canPrepareSessionWorkspace } from './workspace-preflight.mjs';
import { readWorkspaceBytes, workspacePythonName, maxPythonBytes } from './workspace-file.mjs';

const build = 'control_build_zoh_plant';
const recheck = 'control_recheck_plant_workspace';
const replay = 'control_replay_plant_source';
const gateCode = 'PLANT_WORKSPACE_PREFLIGHT_REQUIRED';
const safeTools = new Set(['read','read_image','glob','grep','skill','ask_user_question','control_help']);
const deliveryCode = 'PLANT_DELIVERY_COMPLETE';
const recovery = '请保留该工作区和对话，按权限诊断修复后用 control_recheck_plant_workspace 重新检查，再回放源码并构建。也可以在可写目录新建建模会话。';
const output = {schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]};

function receipt(diagnostic,observation) {
  return {status:'failed',stage:'workspace_preflight',diagnostic,verified:[],changes:[],file_exists:null,
    file_created_by_operation:false,reason:observation==='fresh'?'本次预检失败：'+diagnostic.detail:'沿用上次失败记录：'+diagnostic.detail,
    preflight_observation:observation,work_started:false,retryable:false,recovery};
}

function readFailure(text) {
  try {
    const parsed = JSON.parse(text.replace(/^Error:\s*/, ''));
    return parsed.stage === 'workspace_preflight' && parsed.diagnostic?.code === 'WORKSPACE_PREFLIGHT_FAILED'
      ? parsed.diagnostic : undefined;
  } catch { return undefined; }
}

function completedDelivery(result) {
  if (result.isError) return null;
  let r;
  try { r=JSON.parse(result.content[0].text); } catch { return null; }
  if (r?.status!=='simulated' || r.session_verified!==true || r.file_exists!==true ||
      !Array.isArray(r.verified) || !['save','reload','simulate','compare'].every(s=>r.verified.includes(s)) ||
      !r.comparison?.compared || typeof r.model!=='string' || typeof r.path!=='string' || !isAbsolute(r.path) ||
      !/^[a-f0-9]{64}$/.test(r.sha256) || !Number.isSafeInteger(r.bytes) || r.bytes<4) return null;
  return {model:r.model,path:r.path,sha256:r.sha256,bytes:r.bytes};
}

const deliveredNote = file => '本次转换已交付并完成重载、仿真和源码对照：'+file.path+'。请向用户报告现有文件和实测误差。用户在下一条消息另行要求调试时可继续操作。';
const hasSource = state => state.sources.length>0 || state.workspaceSources.length>0;

// Project only actual user requests and native tool outcomes. Assistant text,
// injected context and an agent's claims about permissions have no authority.
function projection(codeRoot) {
  const {z} = createRequire(resolve(codeRoot,'vendor/deepseek-harness/packages/interaction/commands/package.json'))('zod');
  return {key:'controlPlantWorkspaceAdmission',stateVersion:8,stateSchema:z.object({
    ownStart:z.number(),pythonSource:z.boolean(),awaitingSource:z.boolean(),active:z.boolean(),epoch:z.number(),blocked:z.record(z.string(),z.unknown()).nullable(),
    userSeq:z.number(),delivered:z.object({model:z.string(),path:z.string(),sha256:z.string(),bytes:z.number()}).nullable(),
    sources:z.array(z.object({attachmentId:z.string(),name:z.string(),bytes:z.number()})),pending:z.record(z.string(),z.object({name:z.string(),userSeq:z.number()})),
    cwd:z.string(),workspaceSources:z.array(z.string()),needsReplay:z.boolean(),
    replays:z.record(z.string(),z.object({source_sha256:z.string(),reference_data_sha256:z.string(),source:z.string().optional(),source_kind:z.string().optional()})),
  }),init:(header,ownStart)=>({ownStart,cwd:header.cwd??'',workspaceSources:[],needsReplay:false,userSeq:-1,delivered:null,pythonSource:false,awaitingSource:false,sources:[],active:false,epoch:-1,blocked:null,pending:{},replays:{}}),
  apply(state,event) {
    if (event.seq < state.ownStart) return state;
    const {type,data}=event;
    if (type === 'user/message' && data.source?.kind === 'user') {
      const blocks=data.content;
      const text=blocks.filter(b=>b.type==='text').map(b=>b.text).join(' ');
      const uploads=blocks.filter(b=>b.type==='file' && /\.py$/i.test(b.attachment?.name)).map(b=>b.attachment);
      // A new native upload replaces the source selection for this task.
      // A later text-only prompt or parameter attachment keeps that selection.
      const sources=uploads.length ? uploads.filter((f,i)=>uploads.findIndex(other=>
        f.attachmentId===other.attachmentId && f.name===other.name)===i) : state.sources;
      const pythonSource = state.pythonSource || uploads.length>0 || state.workspaceSources.length>0;
      const wantsModel=/simulink|\bslx\b/i.test(text) && /转换|转成|转为|转[到换]|改[成为]|弄成|生成|创建|搭建|建模|建立|构建|做[成个]|被控对象|convert|build|creat|generat|produce|turn.+into|model (?:it|this)|export/i.test(text);
      const request=(pythonSource || /\bpython\b|\.py\b/i.test(text)) && (wantsModel || state.awaitingSource && uploads.length>0);
      const awaitingSource=!request && (state.awaitingSource || wantsModel);
      return {...state,userSeq:event.seq,delivered:null,pythonSource,sources,workspaceSources:uploads.length?[]:state.workspaceSources,needsReplay:state.needsReplay || request,awaitingSource,active:state.active || request,replays:uploads.length?{}:state.replays,
        epoch:uploads.length || request ? event.seq : state.epoch};
    }
    if (type === 'command/run' && data.name === 'control-end-plant-task' && data.source?.kind === 'user') {
      return {...state,userSeq:event.seq,delivered:null,active:false,needsReplay:false,pythonSource:false,awaitingSource:false,sources:[],workspaceSources:[],blocked:null,epoch:event.seq,pending:{},replays:{}};
    }
    if (type==='tool/call' && data.name==='read')
      return {...state,pending:{...state.pending,[data.callId]:{name:'read',userSeq:state.userSeq}}};
    if (type === 'tool/call' && [build,recheck,replay].includes(data.name)) {
      if (state.delivered) return state;
      const active=state.active || data.name!==recheck;
      return {...state,active,needsReplay:state.needsReplay || data.name===replay,epoch:active && !state.active ? event.seq : state.epoch,
        pending:{...state.pending,[data.callId]:{name:data.name,userSeq:state.userSeq}}};
    }
    if (type !== 'tool/result') return state;
    const msg=data.message, invocation=state.pending[msg.toolCallId], tool=invocation?.name;
    if (!tool && data.error?.code !== gateCode) return state;
    const pending={...state.pending};
    delete pending[msg.toolCallId];
    if (tool==='read') {
      // Native read metadata contains the resolved path even for a relative argument.
      const name=!msg.isError && workspacePythonName(state.cwd,data.meta?.path);
      if (!name || state.workspaceSources.includes(name) || invocation.userSeq<state.epoch && invocation.userSeq!==state.userSeq) return {...state,pending};
      return {...state,pending,workspaceSources:[...state.workspaceSources,name],pythonSource:true,
        active:state.active || state.awaitingSource,awaitingSource:false,epoch:state.awaitingSource?event.seq:state.epoch};
    }
    if (msg.isError) {
      const diagnostic=[data.error?.reason,...msg.content.filter(b=>b.type==='text').map(b=>b.text)]
        .filter(x=>typeof x==='string').map(readFailure).find(Boolean);
      return {...state,pending,blocked:diagnostic ?? state.blocked};
    }
    if (tool === build) {
      const delivered=hasSource(state) ? completedDelivery(msg) : null;
      if (hasSource(state) && !delivered) return {...state,pending};
      const sameRequest=invocation.userSeq===state.userSeq;
      if (!sameRequest && state.epoch>invocation.userSeq) return {...state,pending};
      return {...state,active:false,needsReplay:false,delivered:sameRequest?delivered:null,pythonSource:hasSource(state),
        awaitingSource:false,blocked:null,epoch:event.seq,pending,replays:{}};
    }
    if (tool === replay) {
      try {
        const r=JSON.parse(msg.content[0].text);
        if (r.status==='replayed') return {...state,pending,blocked:null,replays:{...state.replays,[r.referenceFile]:{
          source_sha256:r.source_sha256,reference_data_sha256:r.reference_data_sha256,source:r.source,source_kind:r.source_kind}}};
      } catch { /* A missing typed replay receipt cannot authorize a comparison. */ }
    }
    return {...state,pending,blocked:tool === recheck ? null : state.blocked};
  }};
}

export function installPlantWorkspace(ctx,config,preflight) {
  // Before automatic preparation existed, the Session stored a genuine Host
  // denial. Its fields allow one fresh controlled admission on the selected
  // cwd. Only the current OS check can authorize the eventual ACL edit.
  function eligiblePreviousDenial(session, blocked) {
    const cwd=session.header.cwd;
    return blocked?.operation==='check_WRITE_OWNER' && !blocked.workspace_preparation &&
      typeof cwd==='string' && typeof blocked.path==='string' && !relative(cwd,blocked.path) &&
      canPrepareSessionWorkspace(config.codeRoot,cwd,blocked.security);
  }
  let projectionHandle;
  const registered=Promise.resolve(ctx.sessionProjections.register(projection(config.codeRoot))).then(handle=>projectionHandle=handle);
  registered.catch(()=>{}); // The first admission reports registration failure.
  const checks=new WeakMap();
  const finalDeliveries=new WeakMap();
  const runningBuilds=new WeakMap();
  const deliveredFile=(session,state)=>state.delivered ??
    (finalDeliveries.get(session)?.userSeq===state.userSeq ? finalDeliveries.get(session).file : null);
  function conversionActive(session,state) {
    // Direct registry clients may omit the native transcript. Retire a completed
    // request on the next user message, including when only its hot receipt exists.
    const previous=finalDeliveries.get(session);
    return state.active && (!previous || previous.userSeq===state.userSeq || state.epoch>previous.userSeq);
  }
  // This final native event runs after cancellation, policy and rendering have
  // settled, before the next exclusive sibling. The Session projection restores
  // the same receipt from the recorded call/result pair after a restart.
  ctx.on('tools/result',(exec,result)=>{
    if (exec.name!==build || !exec.agent?.session || !projectionHandle) return;
    const session=exec.agent.session, state=projectionHandle.stateOf(session);
    const calls=runningBuilds.get(session);
    const started=state.pending[exec.callId]?.userSeq ?? calls?.get(exec.callId);
    calls?.delete(exec.callId);
    if (started===undefined || (started!==state.userSeq && state.epoch>started) ||
        finalDeliveries.get(session)?.userSeq>started) return;
    const file=hasSource(state) && completedDelivery(result);
    if (file) finalDeliveries.set(session,{userSeq:started,file});
  });
  const prompt=ctx.get('systemPrompt');
  prompt?.context({name:'control:uploaded-plant-workspace',order:prompt.getContextOrder('SANDBOX_POLICY')+2,text:({agent})=>{
    if (!projectionHandle || !agent?.session) return '';
    const state=projectionHandle.stateOf(agent.session);
    const delivered=deliveredFile(agent.session,state);
    if (delivered) return deliveredNote(delivered);
    const local=checks.get(agent.session);
    const blocked=local?.epoch===state.epoch ? local.blocked : state.blocked;
    if (eligiblePreviousDenial(agent.session,blocked)) return '上次记录显示当前项目工作区缺少 WRITE_OWNER。用户继续建模时，请直接调用 control_recheck_plant_workspace 或 control_replay_plant_source。Host 将重新核验并准备符合条件的目录。';
    if (blocked) return '上次工作区预检失败：'+blocked.operation+' '+blocked.path+'。请说明哪些步骤尚未启动，并转述 Host 给当前普通账户的恢复方法。用户要求重新检查或确认修复后，用 control_recheck_plant_workspace 核验当前权限，通过后继续回放源码并建模。此前保留现场，停止工具重试。';
    if (conversionActive(agent.session,state)) return '用户正在将 Python 被控对象转换为 Simulink。从本次附件或已用 read 读取的当前工作区 Python 源文件中选择正确文件，用 control_replay_plant_source 回放原始 step，将其 referenceFile 交给 control_build_zoh_plant。缺少参数时直接向用户询问。';
    return '';
  }});
  async function context(agent) {
    const session=agent?.session;
    if (!session) return;
    const state=(await registered).stateOf(session);
    let local=checks.get(session);
    if (!local || local.epoch !== state.epoch) {
      local={epoch:state.epoch,blocked:state.blocked,verified:false,replays:new Map()};
      checks.set(session,local);
    }
    return {session,state,local};
  }
  async function check(agent,signal,current) {
    current ??= await context(agent);
    const {session,local}=current ?? {};
    const workspace=session?.header?.cwd;
    if (session?.id !== agent?.id || !workspace || !isAbsolute(workspace)) throw new Error('需要已绑定当前工作目录的原生会话。');
    if (local.checking) return local.checking;
    const original=ctx.sandboxPolicy.resolve({session});
    const policy=plantWorkspacePolicy(ctx,session,original);
    local.checking=(async()=>{
      try {
        const result=await preflight({workspace,policy,signal,operation:'build_zoh_plant',agent});
        if (result?.ok !== true) throw new Error('工作区未返回完整预检结果。');
        local.verified=true;
        local.blocked=null;
        return {status:'verified',stage:'workspace_preflight',workspace:result.workspace,preflight_observation:'fresh',checked_at:new Date().toISOString(),
          session_sandbox_mode:original.mode,sandbox_mode:result.sandbox_mode,
          ...(result.host?.workspace_preparation ? {workspace_preparation:result.host.workspace_preparation} : {})};
      } catch (error) {
        local.verified=false;
        local.blocked={...(error.diagnostic ?? {code:'WORKSPACE_PREFLIGHT_FAILED',operation:'verify_workspace',path:workspace,
          detail:error.message,work_started:false}),checked_at:new Date().toISOString(),session_sandbox_mode:original.mode,sandbox_mode:policy.mode};
        throw new Error(JSON.stringify(receipt(local.blocked,'fresh')));
      }
    })();
    try { return await local.checking; } finally { delete local.checking; }
  }

  async function writableRequestJson(exec,workspace) {
    if (exec.name!=='write' || typeof exec.arguments?.file_path!=='string' || typeof exec.arguments.content!=='string') return false;
    const path=resolve(workspace,exec.arguments.file_path);
    if (relative(workspace,dirname(path)) || !/^[\p{L}\p{N}_-][\p{L}\p{N}_. -]*\.json$/iu.test(basename(path)) ||
        /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\./i.test(basename(path))) return false;
    // Raw relative native writes may resolve against working_directory. Require
    // an absolute JSON target so the checked path is exactly what the tool uses.
    if (!isAbsolute(exec.arguments.file_path) || Buffer.byteLength(exec.arguments.content)>2*1024*1024) return false;
    try { JSON.parse(exec.arguments.content); } catch { return false; }
    try {
      const file=await lstat(path);
      return file.isFile() && !file.isSymbolicLink() && file.nlink===1 && !relative(path,await realpath(path));
    } catch (error) { return error.code==='ENOENT'; }
  }
  async function refusal(exec) {
    if (!exec.agent?.session || safeTools.has(exec.name) || (exec.name==='working_directory' && exec.arguments?.cd === undefined)) return;
    const c=await context(exec.agent);
    const delivered=deliveredFile(c.session,c.state);
    if (delivered) return {code:deliveryCode,text:JSON.stringify({status:'not_dispatched',stage:'delivery_complete',code:deliveryCode,work_started:false,
      verified_delivery:delivered,reason:deliveredNote(delivered)})};
    if (exec.name===recheck) return;
    // The build body runs its own real preflight, including when it is the
    // first explicit operation and no natural-language conversion was found.
    if (c.local.blocked && !([build,replay].includes(exec.name) && eligiblePreviousDenial(c.session,c.local.blocked)))
      return {code:gateCode,text:JSON.stringify(receipt(c.local.blocked,'previous_failure'))};
    if (exec.name===build || !conversionActive(c.session,c.state) && !c.local.blocked) return;
    if (!c.local.verified) {
      try { await check(exec.agent,exec.signal,c); } catch (error) { return {code:gateCode,text:error.message}; }
    }
    if (['control_replay_plant_source','control_simulate_simulink_model'].includes(exec.name) || isControlledModelDispatch(exec) ||
        await writableRequestJson(exec,c.session.header.cwd)) return;
    return {code:'PLANT_CONTROLLED_TOOL_REQUIRED',text:JSON.stringify({status:'not_dispatched',stage:'plant_tool_selection',work_started:false,
      reason:'当前任务是将选定的 Python 源码转换为 SLX。请读取附件或工作区中的源文件，使用 control_replay_plant_source 回放、control_build_zoh_plant 建模和 control_simulate_simulink_model 仿真。'})};
  }
  ctx.on('tools/pre-execute',async(exec,next)=>{
    if (exec.name===build && exec.agent?.session) {
      const {session,state}=await context(exec.agent);
      if (!runningBuilds.has(session)) runningBuilds.set(session,new Map());
      runningBuilds.get(session).set(exec.callId,state.userSeq);
    }
    const reason=await refusal(exec);
    return reason ? {kind:'deny',reason:reason.text,info:{name:'PlantWorkspaceError',code:reason.code}} : next();
  },{prepend:true});
  ctx.on('tools/execute',async(exec,next)=>{
    const reason=await refusal(exec);
    if (reason) return {isError:true,content:[{type:'text',text:'Error: '+reason.text}],
      error:{message:reason.text,info:{name:'PlantWorkspaceError',code:reason.code}}};
    return next();
  },{prepend:true});
  ctx.tools.register(defineTool({name:recheck,description:'Recheck the current conversation workspace ACL, Host publication and confined worker I/O after a plant conversion was blocked. Takes no paths; releases the workspace gate only after successful verification.',
    parameters:{},output,execute:(_,{agent,signal})=>check(agent,signal)}));
  ctx.tools.register(replayTool(ctx,config,context,check));
  ctx.commands.register({definitionId:'@control-engineering/harness-bundle/end-plant-task',name:'control-end-plant-task',
    description:'End this Python-to-Simulink conversion in the current chat to start unrelated work.',
    handler:()=>({kind:'success',text:'已结束本次对象转换。'})});
  // The existing target helper uses this callback inside its normal workspace
  // authorization. A failed preflight becomes visible to sibling native calls
  // immediately, before the agent-loop logs the outer tool result.
  async function checkBuild(options,execution) {
    const c=await context(execution.agent);
    try {
      const result=await check(execution.agent,options.signal,c);
      return result;
    } catch (error) {
      if (c?.local.blocked) error.diagnostic=c.local.blocked;
      throw error;
    }
  }
  async function verifyReference(agent,name,data) {
    const c=await context(agent);
    if (!c?.state.active || !c.state.needsReplay && !hasSource(c.state)) return;
    const proof=typeof name==='string' && (c.local.replays.get(name) ?? c.state.replays[name]);
    if (!proof || data?.source_sha256!==proof.source_sha256 ||
        createHash('sha256').update(JSON.stringify(data)).digest('hex')!==proof.reference_data_sha256)
      throw new Error('本次转换需要 control_replay_plant_source 以原始源码回放生成的完整 referenceFile。');
    if (proof.source_kind==='workspace' && c.state.workspaceSources.includes(proof.source)) {
      const bytes=await readWorkspaceBytes(c.session.header.cwd,proof.source,maxPythonBytes);
      if (createHash('sha256').update(bytes).digest('hex')!==proof.source_sha256)
        throw new Error('工作区源码在回放后已变化，请重新读取并回放当前文件。');
    } else if (!c.state.sources.some(f=>f.attachmentId==='sha256:'+proof.source_sha256))
      throw new Error('回放源码与本次会话选择的文件不符。');
  }
  return {checkBuild,verifyReference};
}
