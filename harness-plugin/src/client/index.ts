import { ControlPanel } from './panel.jsx';
import { WorkbenchEntry } from './workbench-entry.jsx';
import { PlantResult, WorkspaceRecheck, WorkspaceGateTail, plantGateDefinition, workspaceRecoveryView } from './plant-chat.jsx';
import { ChatReviewTool, ChatReviewTail, reviewTurnDefinition } from './confirmation-card.jsx';

export const inject = ['slots', 'layout', 'remote', 'remote.session', 'remote.workspace', 'remote.commands', 'jobs', 'uiWorkspace', 'fileUpload', 'uiConversation', 'sessions'];

const panelId = 'control-engineering';
const unwrap = result => {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
};

/** Native slots, sessions, commands and job streams; no second chat server. */
export function apply(ctx) {
  const execute = async (sessionId, name, args) => {
    const execution = unwrap(await ctx.remote.commands.execute(sessionId, `/${name} ${JSON.stringify(args)}`, []));
    if (!execution || execution.result.kind !== 'success') {
      const message = execution?.result.text ?? 'Native command is unavailable';
      let diagnostic;
      try { diagnostic = JSON.parse(message); } catch { /* Other commands return ordinary text. */ }
      if (diagnostic?.code === 'WORKSPACE_PREFLIGHT_FAILED') {
        const actor = {host:'主机进程',worker:'数值进程',sandbox:'沙箱启动器',policy:'会话写入策略'}[diagnostic.actor] ?? diagnostic.actor;
        throw Object.assign(new Error(`工作区预检失败，计算尚未开始。${actor}在 ${diagnostic.path} 执行 ${diagnostic.operation} 失败（${diagnostic.os_code ?? '无系统错误码'}）。${diagnostic.detail}。${diagnostic.hint}`),
          {workspaceRecovery:workspaceRecoveryView(diagnostic,'fresh')});
      }
      throw new Error(message);
    }
    return JSON.parse(execution.result.text);
  };
  const fetchVerifiedDownload = async (descriptor, mime = 'application/octet-stream') => {
    const limit = 50 * 1024 * 1024;
    if (!descriptor || typeof descriptor.host_path !== 'string' || !/^[a-f0-9]{64}$/.test(descriptor.sha256)
      || typeof descriptor.filename !== 'string' || !descriptor.filename || /[\\/]/.test(descriptor.filename)
      || descriptor.bytes !== undefined && (!Number.isSafeInteger(descriptor.bytes) || descriptor.bytes < 0)) throw new Error('下载描述缺少已校验文件路径、文件名或 SHA-256');
    if (descriptor.bytes > limit) throw new Error('文件超过当前原生下载入口 50 MiB 上限；请从产物目录取得完整文件。没有下载或截断文件。');
    const url = new URL('api/file', document.baseURI); url.searchParams.set('path', descriptor.host_path);
    const response = await fetch(url, { credentials: 'same-origin', redirect: 'follow' });
    if (!response.ok) throw new Error(`原生文件下载失败（HTTP ${response.status}）；未交付部分文件。`);
    const length = response.headers.get('content-length');
    if (length !== null && Number(length) > limit) throw new Error('原生文件超过 50 MiB 上限，未保存部分文件');
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength > limit || descriptor.bytes !== undefined && bytes.byteLength !== descriptor.bytes) throw new Error('下载字节数与已校验文件不一致或超过原生上限');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
    if (digest !== descriptor.sha256) throw new Error('下载 SHA-256 与已校验文件不一致');
    const blobUrl = URL.createObjectURL(new Blob([bytes], { type: mime }));
    const anchor = document.createElement('a'); anchor.href = blobUrl; anchor.download = descriptor.filename; anchor.click();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
    return { ...descriptor, bytes: bytes.byteLength, sha256: digest, status: 'downloaded_verified', hash_verified: true };
  };
  // A tool row and its visible Turn tail can mount together. A status read is
  // a native command too, so share the in-flight read for that exact version.
  const readingCards = new Map(), cardListeners = new Map();
  const cardStatus = (sessionId, {workspace,card_id}) => {
    const key = JSON.stringify([sessionId,workspace,card_id]);
    if (!readingCards.has(key)) readingCards.set(key,
      execute(sessionId,'control-card-status',{workspace,card_id}).finally(()=>readingCards.delete(key)));
    return readingCards.get(key);
  };
  const watchCard = (sessionId,{workspace,card_id},listener) => {
    const key=JSON.stringify([sessionId,workspace,card_id]);
    if(!cardListeners.has(key))cardListeners.set(key,new Set());
    const listeners=cardListeners.get(key);
    listeners.add(listener);
    return ()=>{listeners.delete(listener);if(!listeners.size)cardListeners.delete(key);};
  };
  const confirmCard = async (sessionId, {workspace,card_id}) => {
    const result=await execute(sessionId,'control-confirm',{workspace,card_id});
    if(result.card_id!==card_id || !Array.isArray(result.requirements))throw new Error('收到的确认卡与当前版本不匹配');
    const listeners=cardListeners.get(JSON.stringify([sessionId,workspace,card_id]));
    for(const listener of listeners??[])listener(result);
    return result;
  };
  const chatReviewActions = sessionId => ({
    loadCard: handle => cardStatus(sessionId,handle),
    confirmCard: handle => confirmCard(sessionId,handle),
    watchCard: (handle,listener)=>watchCard(sessionId,handle,listener),
  });
  const actions = {
    async createSession(workspace, title, role, pendingId) {
      if (!['model', 'controller', 'lightweight'].includes(role)) throw new Error('Unknown control role');
      let id=pendingId;
      if (!id) {
        const nativeWorkspace = unwrap(await ctx.remote.workspace.create({ path: workspace })).workspace;
        const created = unwrap(await ctx.remote.session.create({workspaceId:nativeWorkspace.workspaceId,agentPreset:`control-${role}`}));
        if (created.agentPreset !== `control-${role}`) throw new Error('新角色会话的原生预设不匹配；请保留原生返回记录并检查。');
        id=created.sessionId;
      }
      try {
        unwrap(await ctx.remote.session.rename({sessionId:id,title}));
        await execute(id,'control-workspace',{});
        return id;
      } catch (failure) {
        // Native creation has already succeeded. Keep the same id for repair
        // so each failed filesystem probe cannot create another chat.
        throw Object.assign(failure,{pendingSessionId:id});
      }
    },
    async openSession(sessionId) {
      await execute(sessionId, 'control-workspace', {});
      ctx.uiWorkspace.openSession(sessionId);
    },
    listDirectory: (path, signal) => ctx.uiWorkspace.listDirectory(path, signal),
    async submit(sessionId, request) {
      return execute(sessionId, 'control', request);
    },
    review: (sessionId, request) => execute(sessionId, 'control-review', request),
    prepareCard: (sessionId, request) => execute(sessionId,'control-prepare-review',request),
    cardStatus, confirmCard,
    planStatus: (sessionId, workspace, reference) => execute(sessionId, 'control-plan-status', { workspace, ...reference }),
    status: (sessionId, workspace, attemptId) => execute(sessionId, 'control-status', { workspace, attemptId }),
    async history(sessionId, workspace) {
      const execution = unwrap(await ctx.remote.commands.execute(sessionId, `/control-history ${JSON.stringify({ workspace })}`, []));
      if (execution.result.kind !== 'success') throw new Error(execution.result.text);
      return JSON.parse(execution.result.text);
    },
    recover: (sessionId,workspace,attemptId,mode,card_id,checkpoint) =>
      execute(sessionId,'control-recover',{workspace,attemptId,mode,...(card_id?{card_id}:{}),...(checkpoint?{checkpoint}:{})}),
    async download(descriptor) {
      if (!Number.isSafeInteger(descriptor?.bytes) || descriptor.bytes < 1) throw new Error('归档缺少已核实的完整字节数');
      return fetchVerifiedDownload(descriptor, 'application/zip');
    },
    downloadVerifiedFile: (filePath, sha256, filename, bytes) => fetchVerifiedDownload({ host_path: filePath, sha256, filename, bytes }),
    async upload(sessionId, file, signal, onProgress) {
      const receipt = unwrap(await ctx.fileUpload.upload(sessionId, file, file.name, signal, onProgress));
      const execution = unwrap(await ctx.remote.commands.execute(sessionId, '/control-upload', [{ type: 'file', receiptId: receipt.receiptId }], signal));
      if (!execution || execution.result.kind !== 'success') throw new Error(execution?.result.text ?? 'Upload admission is unavailable');
      return JSON.parse(execution.result.text);
    },
    async prompt(sessionId, text, files = [], signal) {
      await execute(sessionId, 'control-workspace', {});
      const content: ({type:'text', text:string} | {type:'file', receiptId:string})[] = [{ type: 'text', text }];
      for (const file of files) {
        signal?.throwIfAborted();
        const receipt = unwrap(await ctx.fileUpload.upload(sessionId, file, file.name, signal));
        content.push({ type: 'file', receiptId: receipt.receiptId });
      }
      signal?.throwIfAborted();
      unwrap(await ctx.remote.session.prompt({ sessionId, requestId: crypto.randomUUID(), mode: 'queue', content }));
      ctx.uiWorkspace.openSession(sessionId);
    },
    watchJobs: sessionId => ctx.jobs.watchRows(sessionId),
    observeJob: (sessionId, jobId) => ctx.jobs.observe(sessionId, jobId),
    async cancel(sessionId, jobId) { unwrap(await ctx.jobs.kill(sessionId, jobId)); },
    hooks: { jobs: ctx.jobs.state },
  };
  ctx.slots.inject('main', () => ctx.slots.register({ name: 'main', key: panelId, inject: () => actions }, ControlPanel));
  ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({
    name:'tool.call.toolview', key:'control_build_zoh_plant', inject:() => ({download:actions.downloadVerifiedFile}),
  },PlantResult));
  ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({
    name:'tool.call.toolview', key:'control_recheck_plant_workspace', inject:() => ({}),
  },WorkspaceRecheck));
  ctx.slots.inject('conversation.chat.turnTail', () => {
    ctx.effect(()=>ctx.uiConversation.events.register(plantGateDefinition),'control: plant workspace gate');
    return ctx.slots.register({name:'conversation.chat.turnTail',id:'control-plant-workspace-gate',inject:sessionId=>{
      const chat=ctx.uiConversation.binding(ctx.sessions.binding(sessionId)).target('chat');
      return {keyedHooks:{workspaceGates:turn=>chat.getSnapshot().nodes.turnDataSource(Number(turn),plantGateDefinition.kind)}};
    }},WorkspaceGateTail);
  });
  ctx.slots.inject('tool.call.toolview', () => ctx.slots.register({
    name:'tool.call.toolview',key:'control_prepare_review',inject:chatReviewActions,
  },ChatReviewTool));
  ctx.slots.inject('conversation.chat.turnTail', () => {
    ctx.effect(()=>ctx.uiConversation.events.register(reviewTurnDefinition),'control: turn review cards');
    return ctx.slots.register({name:'conversation.chat.turnTail',id:panelId,inject:sessionId=>{
      const chat=ctx.uiConversation.binding(ctx.sessions.binding(sessionId)).target('chat');
      return {...chatReviewActions(sessionId),keyedHooks:{reviewCards:turn=>chat.getSnapshot().nodes.turnDataSource(Number(turn),'control-engineering-review')}};
    }},ChatReviewTail);
  });
  ctx.slots.inject('sidebar.footer.action', () => ctx.slots.register({
    name: 'sidebar.footer.action', id: panelId, order: -50,
    inject: () => ({ panelId, openWorkbench: () => ctx.layout.selectPanel(panelId) }),
  }, WorkbenchEntry));
}
