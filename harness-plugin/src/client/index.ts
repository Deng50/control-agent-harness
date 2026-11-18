import { ControlPanel, ControlIcon } from './panel.jsx';

export const inject = ['slots', 'layout', 'remote', 'remote.session', 'remote.workspace', 'remote.commands', 'jobs', 'uiWorkspace', 'fileUpload'];

const panelId = 'control-engineering';
const unwrap = result => {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
};

/** Native slots, sessions, commands and job streams; no second chat server. */
export function apply(ctx) {
  const execute = async (sessionId, name, args) => {
    const execution = unwrap(await ctx.remote.commands.execute(sessionId, `/${name} ${JSON.stringify(args)}`, []));
    if (!execution || execution.result.kind !== 'success') throw new Error(execution?.result.text ?? 'Native command is unavailable');
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
  const actions = {
    async createSession(workspace, title, role) {
      if (!['model', 'controller', 'lightweight'].includes(role)) throw new Error('Unknown control role');
      const nativeWorkspace = unwrap(await ctx.remote.workspace.create({ path: workspace })).workspace;
      const created = unwrap(await ctx.remote.session.create({ workspaceId: nativeWorkspace.workspaceId, agentPreset: `control-${role}` }));
      if (created.agentPreset !== `control-${role}`) throw new Error('新角色会话的原生预设不匹配；请保留原生返回记录并检查。');
      unwrap(await ctx.remote.session.rename({ sessionId: created.sessionId, title }));
      await execute(created.sessionId, 'control-workspace', {});
      return created.sessionId;
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
    confirm: (sessionId, workspace, reference) => execute(sessionId, 'control-confirm', { workspace, ...reference }),
    planStatus: (sessionId, workspace, reference) => execute(sessionId, 'control-plan-status', { workspace, ...reference }),
    status: (sessionId, workspace, attemptId) => execute(sessionId, 'control-status', { workspace, attemptId }),
    async history(sessionId, workspace) {
      const execution = unwrap(await ctx.remote.commands.execute(sessionId, `/control-history ${JSON.stringify({ workspace })}`, []));
      if (execution.result.kind !== 'success') throw new Error(execution.result.text);
      return JSON.parse(execution.result.text);
    },
    async recover(sessionId, workspace, attemptId, mode) {
      const execution = unwrap(await ctx.remote.commands.execute(sessionId, `/control-recover ${JSON.stringify({ workspace, attemptId, mode })}`, []));
      if (execution.result.kind !== 'success') throw new Error(execution.result.text);
      return JSON.parse(execution.result.text);
    },
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
    async prompt(sessionId, text) {
      await execute(sessionId, 'control-workspace', {});
      unwrap(await ctx.remote.session.prompt({ sessionId, requestId: crypto.randomUUID(), mode: 'queue', content: [{ type: 'text', text }] }));
      ctx.uiWorkspace.openSession(sessionId);
    },
    watchJobs: sessionId => ctx.jobs.watchRows(sessionId),
    observeJob: (sessionId, jobId) => ctx.jobs.observe(sessionId, jobId),
    async cancel(sessionId, jobId) { unwrap(await ctx.jobs.kill(sessionId, jobId)); },
    hooks: { jobs: ctx.jobs.state },
  };
  ctx.slots.inject('main', () => ctx.slots.register({ name: 'main', key: panelId, inject: () => actions }, ControlPanel));
  ctx.slots.inject('sidebar.panellist', () => ctx.slots.register({
    name: 'sidebar.panellist', id: panelId, order: -50, label: () => '控制工程',
  }, ControlIcon));
}
