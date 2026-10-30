import React, { useEffect, useMemo, useState } from 'react';
import './panel.css';
import { ArtifactReport } from './report.jsx';

const roles = [
  { id: 'model', title: '模型校正', sub: '物理模型 + 残差学习', operations: [['inspect_workspace','查看工作区'],['inspect_model_source','检查对象材料'],['register_model','导入对象模型'],['register_python_model','运行并导入 Python 动力学'],['import_transitions','导入转移数据'],['simulate_transitions','生成指定仿真数据'],['train_residual','训练残差网络'],['identify_linear_model','最小二乘线性辨识'],['import_artifact','接收外部产物']] },
  { id: 'controller', title: '控制器设计', sub: 'MPC / NMPC + 遗传调参', operations: [['inspect_workspace','查看工作区'],['design_controller','设计与调参'],['validate_estimator','KF / EKF 测量反馈验证'],['import_artifact','接收外部产物']] },
  { id: 'lightweight', title: '控制器轻量化', sub: '专家数据 → 模仿 → 物理梯度', operations: [['inspect_workspace','查看工作区'],['collect_expert','采集专家数据'],['train_imitation','训练模仿策略'],['optimize_policy','物理梯度优化'],['import_artifact','接收外部产物']] },
];
const storageKey = 'control-engineering.v1';
const handoffs = {
  transition_dataset: ['model','train_residual','dataset'],
  nominal_model: ['controller','design_controller','model'],
  corrected_model: ['controller','design_controller','model'],
  identified_model: ['controller','design_controller','model'],
  controller: ['lightweight','collect_expert','controller'],
  expert_dataset: ['lightweight','train_imitation','dataset'],
  imitation_policy: ['lightweight','optimize_policy','policy'],
};
const initialDraft = () => ({ operation: 'inspect_workspace', inputs: '{}', config: '{}', chat: '', jobId: '', sessionId: '' });
const defaults = {
  inspect_model_source: { inputs: { model_path: '' }, config: {} },
  register_model: { inputs: { model_path: '' }, config: { source: { kind: 'user_nominal', description: '' } } },
  register_python_model: { inputs: { model_path: '' }, config: { source: { kind: 'user_nominal', description: 'Reviewed standalone PyTorch dynamics' } } },
  import_artifact: { inputs: { artifact_path: '' }, config: {} },
  import_transitions: { inputs: { nominal_model: '', data_path: '' }, config: { source: { kind: 'measured', description: '' }, seed: 42 } },
  simulate_transitions: { inputs: { nominal_model: '', reference_model: '' }, config: { trajectories: 20, steps: 24, seed: 42 } },
  train_residual: { inputs: { nominal_model: '', dataset: '' }, config: { hidden_dims: [32,32], learning_rate: 0.002, epochs: 150, seed: 42 } },
  design_controller: { inputs: { model: '', evaluation_model: '' }, config: { seed: 42, budget: 8, tuning_scenarios: [], verification_scenarios: [], test_scenarios: [] } },
  collect_expert: { inputs: { controller: '', evaluation_model: '' }, config: { scenarios: [], seed: 42 } },
  train_imitation: { inputs: { dataset: '' }, config: { architecture: 'mlp', hidden_dims: [64,32], learning_rate: 0.001, epochs: 150, seed: 42 } },
  optimize_policy: { inputs: { policy: '', model: '' }, config: { learning_rate: 0.0001, epochs: 20, seed: 42 } },
  identify_linear_model: { inputs: { nominal_model: '', dataset: '' }, config: {} },
  validate_estimator: { inputs: { model: '', controller: '', evaluation_model: '' }, config: { initial_estimate: [], verification_scenarios: [], test_scenarios: [], seed: 42 } },
};
function load() {
  try { const saved = JSON.parse(localStorage.getItem(storageKey)); if (saved?.version === 1) return saved; } catch { /* A malformed browser draft cannot become a task. */ }
  return { version: 1, workspace: '', role: 'model', projects: {} };
}
function json(text) { const value = JSON.parse(text); if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('请输入 JSON 对象'); return value; }
function events(text) { return text.split('\n').flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } }); }
const pretty = value => JSON.stringify(value, null, 2);

export function ControlIcon({ size = 20 }) { return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="M3 18h18M5 15l4-8 5 5 5-9"/><circle cx="9" cy="7" r="2"/><circle cx="14" cy="12" r="2"/></svg>; }

/** Keeps only form drafts in browser storage; authoritative tasks live in Harness. */
export function ControlPanel({ createSession, openSession, listDirectory, submit, history, recover, upload, prompt, watchJobs, observeJob, cancel, useJobs }) {
  const [state, setState] = useState(load);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState('');
  const [uploadController, setUploadController] = useState(null);
  const [directory, setDirectory] = useState(null);
  const [attempts, setAttempts] = useState([]);
  const [selectedAttempt, setSelectedAttempt] = useState(null);
  const jobs = useJobs(value => value);
  const role = roles.find(item => item.id === state.role) ?? roles[0];
  const key = `${state.workspace.trim()}::${role.id}`;
  const draft = state.projects[key] ?? initialDraft();
  const patch = update => setState(value => ({ ...value, projects: { ...value.projects, [key]: { ...(value.projects[key] ?? initialDraft()), ...update } } }));
  useEffect(() => { localStorage.setItem(storageKey, JSON.stringify(state)); }, [state]);
  useEffect(() => draft.sessionId ? watchJobs(draft.sessionId) : undefined, [draft.sessionId, watchJobs]);
  useEffect(() => draft.sessionId && draft.jobId ? observeJob(draft.sessionId, draft.jobId) : undefined, [draft.sessionId, draft.jobId, observeJob]);
  useEffect(() => () => uploadController?.abort(), [uploadController]);
  useEffect(() => { setAttempts([]); setSelectedAttempt(null); }, [key]);
  const rows = jobs.rows[draft.sessionId] ?? [];
  const observed = jobs.observed[draft.jobId];
  const stream = useMemo(() => events(observed?.text ?? ''), [observed?.text]);
  const result = selectedAttempt ? selectedAttempt.result : stream.findLast(item => item.event === 'result')?.result;
  const job = rows.find(item => item.id === draft.jobId);
  const terminal = job && ['completed','failed','killed'].includes(job.status);
  const progress = stream.filter(item => item.event === 'progress');
  const logLines = (observed?.text ?? '').split('\n');
  const visibleLog = logLines.slice(-60).map(line => {
    try { const row = JSON.parse(line); return row.event === 'result' ? `[result] ${row.result?.artifact_id ?? row.result?.workflow_run ?? '工作区结果'} · ${row.result?.status ?? '已读取'}（详见下方报告与产物）` : line; } catch { return line; }
  }).join('\n');
  useEffect(() => {
    if (!observed?.error?.includes('unknown job') || !draft.attemptId || !draft.sessionId) return;
    let current = true;
    history(draft.sessionId, state.workspace.trim()).then(value => {
      const record = value.attempts.find(item => item.attemptId === draft.attemptId);
      if (current && record) { setSelectedAttempt(record); setAttempts(value.attempts); patch({ jobId: '' }); }
    }).catch(failure => { if (current) setError(`持久任务记录读取失败：${failure.message}`); });
    return () => { current = false; };
  }, [observed?.error, draft.attemptId, draft.sessionId, key, history]);
  async function action(fn) { setBusy(true); setError(''); try { await fn(); } catch (failure) { setError(failure.message ?? String(failure)); } finally { setBusy(false); } }
  async function session(forWork = false) {
    if (!state.workspace.trim()) throw new Error('请先填写绝对工作目录');
    if (!forWork && draft.sessionId) return draft.sessionId;
    if (draft.roleSessionId) { patch({ sessionId: draft.roleSessionId }); return draft.roleSessionId; }
    const id = await createSession(state.workspace.trim(), `${role.title} · ${state.workspace.trim().split(/[\\/]/).at(-1)}`, role.id);
    const legacySessionIds = [...new Set([...(draft.legacySessionIds ?? []), draft.sessionId].filter(value => value && value !== id))];
    patch({ sessionId: id, roleSessionId: id, legacySessionIds, jobId: '' });
    return id;
  }
  function selectOperation(operation) {
    const preset = defaults[operation] ?? { inputs: {}, config: {} };
    patch({ operation, inputs: pretty(preset.inputs), config: pretty(preset.config) });
  }
  async function run(request) {
    const sessionId = await session(true);
    const started = await submit(sessionId, request ?? { agent: role.id, operation: draft.operation, workspace: state.workspace.trim(), inputs: json(draft.inputs), config: json(draft.config) });
    setSelectedAttempt(null);
    patch({ sessionId, jobId: started.jobId, attemptId: started.attemptId });
  }
  async function recoverAttempt(record, mode) {
    const sessionId = await session();
    const started = await recover(sessionId, state.workspace.trim(), record.attemptId, mode);
    setSelectedAttempt(null);
    patch({ sessionId, jobId: started.jobId, attemptId: started.attemptId });
  }
  function parameter(name, value) { try { patch({ config: pretty({ ...json(draft.config), [name]: value }) }); } catch (failure) { setError(failure.message); } }
  let config = {}; try { config = json(draft.config); } catch { /* JSON editor keeps incomplete input until submission. */ }
  function adoptHandoff(plan) {
    const { agent: next, operation, inputs, config_suggestions } = plan.handoff;
    const destination = `${state.workspace.trim()}::${next}`;
    const preset = defaults[operation];
    setState(value => ({ ...value, role: next, projects: { ...value.projects, [destination]: { ...(value.projects[destination] ?? initialDraft()), operation,
      inputs: pretty(inputs), config: pretty({ ...preset.config, ...config_suggestions }) } } }));
    setSelectedAttempt(null);
  }
  function handoff(artifactId) {
    return action(() => run({ agent: role.id, operation: 'prepare_handoff', workspace: state.workspace.trim(), inputs: { artifact_id: artifactId } }));
  }
  return <div className="control-panel">
    <header><p className="control-kicker">CONTROL ENGINEERING / HARNESS</p><h1>从物理模型到轻量化控制</h1><p>三个独立会话，共享可校验的工程产物。计算由 Harness 后台作业执行。</p></header>
    <label className="control-workspace">项目工作目录<input placeholder="D:\projects\control-agent-harness\workspaces\project-01" value={state.workspace} onChange={event => setState(value => ({ ...value, workspace: event.target.value }))}/><small>Windows 工作区需由当前用户拥有并授予其完全控制，供 Harness 配置沙箱。路径改变会切换该项目的会话与草稿。</small></label>
    <button disabled={busy} onClick={() => action(async () => setDirectory(await listDirectory(state.workspace.trim() || undefined)))}>浏览工作目录</button>
    {directory && <section className="control-card" aria-label="工作目录浏览"><div className="control-row">{directory.crumbs.map(crumb => <button key={crumb.path} onClick={() => action(async () => setDirectory(await listDirectory(crumb.path)))}>{crumb.name}</button>)}</div><div className="control-directory">{directory.entries.map(entry => <button key={entry.path} onClick={() => action(async () => setDirectory(await listDirectory(entry.path)))}>{entry.name}</button>)}</div>{directory.truncated && <small>目录过多，当前仅显示前一部分。</small>}<div className="control-row"><button onClick={() => { setState(value => ({ ...value, workspace: directory.path })); setDirectory(null); }}>选择此目录</button><button onClick={() => setDirectory(null)}>关闭浏览</button></div></section>}
    <nav className="control-stages">{roles.map((item,index) => <button key={item.id} aria-pressed={role.id === item.id} onClick={() => setState(value => ({ ...value, role: item.id }))}><span>0{index+1}</span><strong>{item.title}</strong><small>{item.sub}</small></button>)}</nav>
    {error && <div role="alert" className="control-error">{error}</div>}
    <div className="control-columns"><section className="control-card"><h2>{role.title}</h2>
      <div className="control-row"><label>操作<select value={draft.operation} onChange={event => selectOperation(event.target.value)}>{role.operations.map(([value,label]) => <option key={value} value={value}>{label}</option>)}</select></label><button disabled={busy} onClick={() => action(async () => openSession(await session(true)))}>打开独立会话</button></div>
      <label>输入产物 ID / 文件路径<textarea aria-label="输入产物" rows={5} spellCheck={false} value={draft.inputs} onChange={event => patch({ inputs: event.target.value })}/></label>
      {['inspect_model_source','register_model','register_python_model','import_transitions'].includes(draft.operation) && <label>上传{draft.operation === 'import_transitions' ? '数据 NPZ' : '对象材料'}<input type="file" disabled={busy} accept={draft.operation === 'import_transitions' ? '.npz' : draft.operation === 'inspect_model_source' ? '.json,.py,.slx,.mdl' : draft.operation === 'register_python_model' ? '.py' : '.json'} onChange={event => { const file = event.target.files?.[0]; if (!file) return; action(async () => { const id = await session(true); const controller = new AbortController(); setUploadController(controller); try { const receipt = await upload(id, file, controller.signal, progress => setUploadProgress(`已上传 ${(progress.loaded/1024).toFixed(0)} KB`)); patch({ inputs: pretty({ ...json(draft.inputs), [draft.operation === 'import_transitions' ? 'data_path' : 'model_path']: receipt.path }) }); setUploadProgress(`${receipt.name} · ${receipt.bytes} 字节 · SHA-256 ${receipt.sha256}`); } finally { setUploadController(null); } }); }}/><small>{uploadProgress}</small>{uploadController && <button onClick={() => uploadController.abort()}>取消上传</button>}</label>}
      {['register_model','register_python_model'].includes(draft.operation) && config.model_spec?.interface && <ModelInterfaceEditor spec={config.model_spec} onChange={value => parameter('model_spec', value)}/>}
      {draft.operation === 'register_python_model' && <p>此操作会在 Harness 数值作业中执行已检查的独立 PyTorch 源码，核对形状、重复性与梯度，并保存源码快照。请先检查对象材料和接口声明。</p>}
      {['train_residual','train_imitation','optimize_policy'].includes(draft.operation) && <div className="control-parameters">
        <label>学习率<input type="number" min="0.0000001" step="0.0001" value={config.learning_rate ?? ''} onChange={event => parameter('learning_rate', Number(event.target.value))}/></label>
        <label>训练轮数<input type="number" min="1" value={config.epochs ?? ''} onChange={event => parameter('epochs', Number(event.target.value))}/></label>
        {draft.operation !== 'optimize_policy' && <><label>输入维度<input type="number" min="1" placeholder="按模型协议校验" value={config.input_dim ?? ''} onChange={event => parameter('input_dim', Number(event.target.value))}/></label>
        <label>输出维度<input type="number" min="1" placeholder="按模型协议校验" value={config.output_dim ?? ''} onChange={event => parameter('output_dim', Number(event.target.value))}/></label>
        {['rnn','lstm'].includes(config.architecture) ? <><label>循环层数<input type="number" min="1" value={config.num_layers ?? 1} onChange={event => parameter('num_layers', Number(event.target.value))}/></label><label>每层节点数<input type="number" min="1" value={config.hidden_size ?? 32} onChange={event => parameter('hidden_size', Number(event.target.value))}/></label><label>未来扰动预览长度<input type="number" min="1" value={config.preview_len ?? 5} onChange={event => parameter('preview_len', Number(event.target.value))}/></label><label>扰动预览来源<select value={config.disturbance_forecast ?? ''} onChange={event => parameter('disturbance_forecast', event.target.value)}><option value="">必须显式声明</option><option value="declared_constant">已知恒定扰动工况</option></select></label></> : <label>隐藏层节点数<input placeholder="32,32" value={config.hidden_dims?.join(',') ?? ''} onChange={event => { const nodes = event.target.value.split(',').map(Number); if (nodes.every(n => Number.isInteger(n) && n > 0)) parameter('hidden_dims', nodes); }}/></label>}</>}
        {draft.operation === 'train_imitation' && <label>网络<select value={config.architecture ?? 'mlp'} onChange={event => parameter('architecture', event.target.value)}>{['mlp','rnn','lstm'].map(name => <option key={name} value={name}>{name.toUpperCase()}</option>)}</select></label>}
      </div>}
      <label>阶段配置<textarea aria-label="阶段配置" rows={9} spellCheck={false} value={draft.config} onChange={event => patch({ config: event.target.value })}/></label>
      {['train_residual','train_imitation','optimize_policy'].includes(draft.operation) && <label>训练检查点（留空从头训练）<input value={config.resume_checkpoint ?? ''} placeholder="failed/.../latest.pt 或产物内的检查点" onChange={event => { const next = { ...config }; delete next.resume_checkpoint_sha256; if (event.target.value.trim()) next.resume_checkpoint = event.target.value.trim(); else delete next.resume_checkpoint; patch({ config: pretty(next) }); }}/><small>训练轮数是总目标。恢复会核对依赖、配置、优化器和随机状态。</small></label>}
      <div className="control-row"><button className="control-primary" disabled={busy} onClick={() => action(() => run())}>{busy ? '提交中…' : '执行阶段'}</button><button disabled={busy} onClick={() => action(() => run({ agent: role.id, operation: 'inspect_workspace', workspace: state.workspace.trim() }))}>刷新产物</button></div>
      <details><summary>用对话调整任务</summary><textarea rows={3} value={draft.chat} placeholder="说明目标和希望修改的参数…" onChange={event => patch({ chat: event.target.value })}/><button disabled={busy || !draft.chat.trim()} onClick={() => action(async () => { const id = await session(true); await prompt(id, `当前角色：${role.title}。工作目录：${state.workspace.trim()}。\n当前请求：${pretty({ operation: draft.operation, inputs: json(draft.inputs), config: json(draft.config) })}\n用户要求：${draft.chat}\n使用 control_${role.id} 工具读取必要产物并执行；维度不一致时先解释并确认，不能默默截断或补零。`); })}>提交到 Harness 对话</button><small>使用 Harness 设置中的模型和凭据。</small></details>
      <details><summary>用户项目串联工作流</summary><p>选择含 steps 的流程 JSON。阶段通过 {`{"stage":"前一步ID"}`} 接力，也可直接填写已有产物 ID。先预检，再运行。</p><label>流程文件路径<input aria-label="流程文件路径" value={draft.workflowPath ?? ''} onChange={event => patch({ workflowPath: event.target.value })} placeholder="D:\project\workflow.json"/></label><div className="control-row">{[['plan_workflow','检查前置条件'],['run_workflow','运行串联流程']].map(([operation,label]) => <button key={operation} disabled={busy || !draft.workflowPath?.trim()} onClick={() => action(() => run({ agent: 'workflow', operation, workspace: state.workspace.trim(), inputs: { workflow_path: draft.workflowPath.trim() } }))}>{label}</button>)}</div><small>已完成阶段会保存产物。中断后在任务记录中选择续接；输入改变时须新建运行。</small></details>
      <details><summary>完整流程演示（明确的仿真对象）</summary><p>使用全新工作目录，串联残差训练、MPC 调参、专家采集和策略优化。演示数据不是实测数据。</p>{['linear','nonlinear'].map(name => <button key={name} disabled={busy} onClick={() => action(() => run({ agent: 'workflow', operation: 'demo', workspace: state.workspace.trim(), config: { case: name } }))}>{name === 'linear' ? '线性 4 状态 / 2 控制' : '非线性 2 状态 / 1 控制'}</button>)}</details>
    </section><section className="control-card"><h2>作业与结果</h2>
      {draft.legacySessionIds?.length > 0 && <label>查看会话任务<select aria-label="查看会话任务" value={draft.sessionId} onChange={event => { patch({ sessionId: event.target.value, jobId: '' }); setAttempts([]); setSelectedAttempt(null); }}><option value={draft.roleSessionId}>当前角色会话</option>{draft.legacySessionIds.map(id => <option key={id} value={id}>旧会话 · {id}</option>)}</select><small>旧会话及其任务记录仍可读取；新任务使用当前角色配置。</small></label>}
      <details><summary>持久任务记录与恢复</summary><button disabled={busy} onClick={() => action(async () => { const value = await history(await session(), state.workspace.trim()); setAttempts(value.attempts); })}>刷新任务记录</button><p>记录属于当前角色会话。重试按原请求重新执行；续训恢复检查点，流程续接复用已核验阶段。</p>{attempts.slice(0,20).map(record => <div className="control-artifact" key={record.attemptId}><strong>{record.request.operation} · {record.state}</strong><code>{record.attemptId}</code><small>{record.detail}</small><button onClick={() => { setSelectedAttempt(record); patch({ jobId: record.authority === 'native_job' ? record.nativeJobId : '' }); }}>查看记录结果</button>{['failed','killed','interrupted'].includes(record.state) && <><button disabled={busy || !!record.recoveryBlocked} onClick={() => action(() => recoverAttempt(record, 'retry'))}>按原请求重试</button>{(record.checkpoint || record.workflowRun) && <button disabled={busy || !!record.recoveryBlocked} onClick={() => action(() => recoverAttempt(record, 'resume'))}>{record.workflowRun ? '从已完成阶段续接' : '从检查点续训'}</button>}</>}{record.recoveryBlocked && <small>{record.recoveryBlocked}</small>}</div>)}</details>
      <select aria-label="选择作业" value={draft.jobId} onChange={event => patch({ jobId: event.target.value })}><option value="">选择作业</option>{rows.map(item => <option key={item.id} value={item.id}>{item.label} · {item.status}</option>)}</select>
      {job && <div className="control-status"><strong>{job.status}</strong><span>{job.progress}</span>{!terminal && <button onClick={() => action(() => cancel(draft.sessionId, job.id))}>取消作业</button>}{terminal && job.status !== 'completed' && <button onClick={() => action(() => run())}>按当前配置重试</button>}<small>{job.detail}</small></div>}
      {(observed?.error || observed?.gapBefore) && <p className="control-error">{observed.error ?? '较早日志已超出显示窗口，请查看工作区文件。'}</p>}
      <LossChart values={progress}/>
      <pre className="control-logs">{visibleLog || '执行阶段后，这里显示原生作业日志和指标。'}</pre>
      {logLines.length > 60 && <small>当前显示最近60行；较早的事件保存在工作目录的 .control-jobs 中。</small>}
      {selectedAttempt && <><p>已读取持久任务记录 · {selectedAttempt.request.operation} · {selectedAttempt.state}</p><details><summary>持久记录详情</summary><pre className="control-logs">{pretty(selectedAttempt)}</pre></details></>}
      {result?.candidates && <div><h3>对象材料检查：{result.registration_ready ? '有完整候选，待确认' : '需要补充信息'}</h3><p>本次仅静态检查，未执行上传源码或原始 Simulink。</p><pre className="control-logs">{pretty({ source: result.source, candidates: result.candidates, missing: result.missing, uncertain: result.uncertain, errors: result.errors, runtime: result.runtime })}</pre>{result.editable_model_spec && <button onClick={() => patch({ operation: result.registration_operation ?? 'register_model', inputs: pretty({ model_path: result.source.path }), config: pretty({ model_spec: result.editable_model_spec, source: { kind: result.registration_operation === 'register_python_model' || result.source.format === 'json' ? 'user_nominal' : 'reconstructed_equations', description: result.registration_operation === 'register_python_model' ? 'User-reviewed standalone PyTorch dynamics source' : 'User-reviewed declared equations from inspected source' } }) })}>检查并修改模型接口</button>}</div>}
      {result?.stages && <div><h3>流程前置条件：{result.status === 'ready' ? '通过' : '待修正'}</h3><small>{result.validation_scope}</small>{result.stages.map(stage => <p key={stage.id}>{stage.id} → {stage.output_kind} · {stage.status}</p>)}{result.errors?.map((item,index) => <p className="control-error" key={index}>{item.id}: {item.message}</p>)}</div>}
      {result?.workflow_run && <div><h3>流程运行</h3><code>{result.workflow_run}</code><pre className="control-logs">{pretty(result.steps)}</pre></div>}
      {result?.handoff && <div className="control-handoff"><h3>已校验的阶段接力</h3><p>{result.handoff.agent} → {result.handoff.operation}</p><pre className="control-logs">{pretty(result.handoff.inputs)}</pre>{result.handoff.missing_inputs?.length > 0 && <p>待补充输入：{result.handoff.missing_inputs.join('、')}</p>}{result.handoff.missing_config?.length > 0 && <p>待设置工况：{result.handoff.missing_config.join('、')}</p>}<small>接力依赖选中产物及其确切版本。采用后请检查工况配置，再执行阶段。</small><button disabled={busy} onClick={() => adoptHandoff(result)}>采用配置并切换</button></div>}
      <details><summary>强制中断后的未发布检查点</summary><p>输入日志中记录的 .staging 阶段目录名。仅在操作系统确认原进程已退出、检查点及依赖校验通过后归档。</p><label>未发布阶段 ID<input value={draft.interruptedStage ?? ''} placeholder="corrected_model-12位十六进制ID" onChange={event => patch({ interruptedStage: event.target.value })}/></label><button disabled={busy || !draft.interruptedStage?.trim()} onClick={() => action(() => run({ agent: role.id, operation: 'recover_interrupted_stage', workspace: state.workspace.trim(), inputs: { stage_id: draft.interruptedStage.trim() } }))}>校验并归档检查点</button></details>
      {result?.status === 'archived' && result.checkpoint && <div><h3>检查点已归档</h3><p>已训练 {result.epoch} 轮。归档未启动训练。</p><code>{result.checkpoint}</code><small>SHA-256 {result.checkpoint_sha256}</small><button onClick={() => action(async () => { const operation = {residual_training:'train_residual',expert_imitation:'train_imitation',physical_policy_optimization:'optimize_policy'}[result.algorithm]; if (draft.operation !== operation) throw new Error(`请先选择 ${operation} 并恢复原训练输入与配置，再采用此检查点`); patch({ config: pretty({ ...json(draft.config), resume_checkpoint: result.checkpoint, resume_checkpoint_sha256: result.checkpoint_sha256 }) }); })}>填入当前训练配置</button></div>}
      {result?.artifact_id && <div><h3>阶段产物</h3><code>{result.artifact_id}</code><p>{result.artifact_path}</p><p>状态：{result.status}</p>{result.status === 'verified' && handoffs[result.kind] && <button onClick={() => handoff(result.artifact_id, result.kind)}>交给下一阶段</button>}<button onClick={() => action(() => run({ agent: role.id, operation: 'inspect_artifact', workspace: state.workspace.trim(), inputs: { artifact_id: result.artifact_id } }))}>校验并查看报告</button></div>}
      {result?.report && <><ArtifactReport key={result.artifact_id} report={result.report} modelInterface={result.metadata?.interface}/><details><summary>独立验证报告 JSON</summary><pre className="control-logs">{pretty(result.report)}</pre></details></>}
      {result?.files && <details><summary>生成文件与接口元数据</summary>{result.files.map(file => <div key={file.name} className="control-artifact"><code>{file.name}</code><small>{file.path}</small><button disabled={busy} onClick={() => action(() => run({ agent: role.id, operation: 'read_artifact_file', workspace: state.workspace.trim(), inputs: { artifact_id: result.artifact_id, file: file.name } }))}>查看文件 · {file.name}</button></div>)}<pre className="control-logs">{pretty({ interface: result.metadata.interface, validation: result.metadata.validation })}</pre></details>}
      {result?.file_preview && <div><h3>文件预览 · {result.file_preview.name}</h3><small>{result.file_preview.bytes} 字节 · SHA-256 {result.file_preview.sha256}</small>{result.file_preview.text && <pre className="control-logs">{result.file_preview.text}</pre>}{result.file_preview.arrays && <pre className="control-logs">{pretty(result.file_preview.arrays)}</pre>}{result.file_preview.format === 'binary' && <p>二进制文件保存在产物目录中，可由相应模型加载器读取。</p>}{result.file_preview.truncated && <small>当前为有界预览，完整文件保存在产物目录。</small>}</div>}
      {result?.artifacts && <div><h3>工作区产物</h3>{Object.entries(result.artifacts).map(([id,item]) => typeof item === 'object' ? <div className="control-artifact" key={id}><strong>{item.kind} · {item.status}</strong><code>{id}</code><small>{item.path}</small>{handoffs[item.kind] && item.status === 'verified' && <button onClick={() => handoff(id,item.kind)}>选择并接力</button>}<button onClick={() => action(() => run({ agent: role.id, operation: 'inspect_artifact', workspace: state.workspace.trim(), inputs: { artifact_id: id } }))}>查看报告</button></div> : <p key={id}>{id}: {item}</p>)}</div>}
    </section></div>
  </div>;
}
function LossChart({ values }) {
  const lastStage = values.findLast(row => Number.isFinite(row.validation_loss))?.stage;
  const rows = values.filter(row => row.stage === lastStage && Number.isFinite(row.validation_loss));
  if (rows.length < 2) return null;
  const y = rows.map(row => row.validation_loss), low = Math.min(...y), high = Math.max(...y);
  const points = y.map((value,index) => `${10+index/(y.length-1)*380},${90-(value-low)/(high-low || 1)*70}`).join(' ');
  return <figure className="control-chart"><figcaption>验证集损失 · {rows.at(-1).stage}</figcaption><svg viewBox="0 0 400 105" role="img" aria-label="验证集损失变化"><path d="M10 10v80h380" fill="none" stroke="currentColor" opacity=".3"/><polyline points={points} fill="none" stroke="#259a8c" strokeWidth="2"/></svg><small>{y.at(-1).toExponential(3)}</small></figure>;
}

function ModelInterfaceEditor({ spec, onChange }) {
  const edit = (group, index, name, value) => {
    const updated = structuredClone(spec);
    updated.interface[group][index][name] = value;
    onChange(updated);
  };
  return <details open><summary>模型接口确认</summary><label>采样周期（秒）<input type="number" min="0.000001" step="0.01" value={spec.interface.sample_time} onChange={event => onChange({ ...spec, interface: { ...spec.interface, sample_time: Number(event.target.value) } })}/></label>{[['states','状态'],['controls','控制'],['outputs','输出'],['disturbances','扰动']].map(([group,label]) => <div key={group}><strong>{label} · {(spec.interface[group] ?? []).length} 维（按此顺序）</strong>{(spec.interface[group] ?? []).map((variable,index) => <div className="control-parameters" key={index}>{[['name','名称'],['unit','单位'],['lower','下界'],['upper','上界']].map(([field,title]) => <label key={field}>{label}{index + 1} {title}<input type={['lower','upper'].includes(field) ? 'number' : 'text'} value={variable[field]} onChange={event => edit(group,index,field,['lower','upper'].includes(field) ? Number(event.target.value) : event.target.value)}/></label>)}</div>)}</div>)}<small>维度、矩阵和方程可在下方 JSON 中修改；提交时会验证一致性，不会自动补零或截断。单位或采样周期修改不会自动转换方程。</small></details>;
}
