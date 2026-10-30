import { ControlPanel, ControlIcon } from './panel.jsx';

export const inject = ['slots', 'layout', 'remote', 'remote.session', 'remote.commands', 'jobs', 'uiWorkspace', 'fileUpload'];

const panelId = 'control-engineering';
const unwrap = result => {
  if (!result.ok) throw new Error(`${result.error.code}: ${result.error.message}`);
  return result.value;
};

/** Native slots, sessions, commands and job streams; no second chat server. */
export function apply(ctx) {
  const actions = {
    async createSession(workspace, title, role) {
      if (!['model', 'controller', 'lightweight'].includes(role)) throw new Error('Unknown control role');
      const created = unwrap(await ctx.remote.session.create({ cwd: workspace, agentPreset: `control-${role}` }));
      unwrap(await ctx.remote.session.rename({ sessionId: created.sessionId, title }));
      return created.sessionId;
    },
    openSession: sessionId => ctx.uiWorkspace.openSession(sessionId),
    listDirectory: (path, signal) => ctx.uiWorkspace.listDirectory(path, signal),
    async submit(sessionId, request) {
      const execution = unwrap(await ctx.remote.commands.execute(sessionId, `/control ${JSON.stringify(request)}`, []));
      if (!execution || execution.result.kind !== 'success') throw new Error(execution?.result.text ?? 'Control command is unavailable');
      return JSON.parse(execution.result.text);
    },
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
    async upload(sessionId, file, signal, onProgress) {
      const receipt = unwrap(await ctx.fileUpload.upload(sessionId, file, file.name, signal, onProgress));
      const execution = unwrap(await ctx.remote.commands.execute(sessionId, '/control-upload', [{ type: 'file', receiptId: receipt.receiptId }], signal));
      if (!execution || execution.result.kind !== 'success') throw new Error(execution?.result.text ?? 'Upload admission is unavailable');
      return JSON.parse(execution.result.text);
    },
    async prompt(sessionId, text) {
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
