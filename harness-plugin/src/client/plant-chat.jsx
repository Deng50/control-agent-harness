import React, { useEffect, useRef, useState } from 'react';

/** Upload through the same native Session used for the eventual user turn. */
export function PlantChat({ submit, busy }) {
  const [files, setFiles] = useState([]);
  const [request, setRequest] = useState('把上传的 Python 对象转换成 Simulink 被控对象，保存到本次对话工作区，重载并验证仿真。');
  const active = useRef(null);
  useEffect(() => () => active.current?.abort(), []);
  return <details><summary>上传 Python 对象并转换成 Simulink</summary>
    <label>对象源码及参数文件<input type="file" multiple accept=".py,.json,.md" disabled={busy}
      onChange={event => setFiles(Array.from(event.target.files ?? []))}/></label>
    <label>建模要求<textarea rows={3} value={request} onChange={event => setRequest(event.target.value)}/></label>
    <button type="button" disabled={busy || !request.trim() || !files.some(file => /\.py$/i.test(file.name))}
      onClick={async () => {
        if (active.current) return;
        const controller = new AbortController();
        active.current = controller;
        try { await submit(request, files, controller.signal); }
        finally { active.current = null; }
      }}>发送到建模对话</button>
  </details>;
}

const stages = {validation:'核对输入', create:'创建模型', configure:'配置矩阵和初态', wire:'连接端口',
  save:'保存 SLX', reload:'关闭后重载', simulate:'运行仿真', compare:'逐点数值比较', dispatch:'等待已派发调用回执',
  admission:'原生工具权限检查', resource_precheck:'专用会话派发前核验', workspace_preflight:'工作区写入预检', host_file_check:'主机核验文件'};
const stageName = value => stages[value] ?? String(value);

function parsedReceipt(content) {
  const text=(content ?? []).filter(item=>item.type==='text').map(item=>item.text).join('\n');
  try { return JSON.parse(text.replace(/^Error:\s*/,'')); } catch { return undefined; }
}

// Older transcripts have no observation tag. Preserve them as historical records.
export function workspaceRecoveryView(diagnostic, observation) {
  const recovery=diagnostic?.workspace_recovery;
  return recovery && {...recovery, observation, checked_at:diagnostic.checked_at,
    preparation_detail:diagnostic.workspace_preparation ? diagnostic.detail : undefined};
}

export function WorkspaceRecheck({phase,block}) {
  if (phase !== 'result') return <p>正在重新检查当前工作区权限…</p>;
  const result=parsedReceipt(block.content);
  if (result?.diagnostic?.workspace_recovery) return <WorkspaceRecovery recovery={workspaceRecoveryView(result.diagnostic,result.preflight_observation)}/>;
  if (!block.isError && result?.status === 'verified')
    return <p>工作区已通过重新检查，可以继续回放 Python 并建模。<br/><code>{result.workspace}</code>
      {result.sandbox_mode && <> · 核验模式 <code>{result.sandbox_mode}</code></>}</p>;
  return <p role="alert">{result?.reason || (block.content ?? []).filter(item=>item.type==='text').map(item=>item.text).join('\n')}</p>;
}

// Keep a genuine Host tool denial visible even when the chat folds native shell/tool rows.
export const plantGateDefinition = {
  kind:'control-plant-workspace-gate',target:'chat',
  match(event) {
    if (event.type!=='tool/result' || event.surfaceOp!=='append' || event.data.error?.code!=='PLANT_WORKSPACE_PREFLIGHT_REQUIRED') return null;
    return parsedReceipt(event.data.message.content)?.diagnostic?.workspace_recovery ? {id:event.data.message.toolCallId,role:'start'} : null;
  },
  start(_context,{event}) {
    const result=parsedReceipt(event.data.message.content);
    return workspaceRecoveryView(result.diagnostic,result.preflight_observation);
  },
  update({state}) { return state; },
  buildViewNode(context) {
    return {key:context.key,kind:'control-plant-workspace-gate',id:context.id,target:'chat',anchorSeq:context.start.event.seq,
      location:context.start.location,visibility:'hidden',data:context.state};
  },
};

export function WorkspaceGateTail({turn,useWorkspaceGates}) {
  const failures=useWorkspaceGates(String(turn.turn));
  return failures?.length ? <WorkspaceRecovery recovery={failures.at(-1)}/> : null;
}

export function WorkspaceRecovery({ recovery }) {
  const cached=recovery.observation==='previous_failure';
  const prefix=cached?'上次预检记录缺少':recovery.observation==='fresh'?'本次预检发现缺少':'预检记录中缺少';
  return <section className="control-card control-workspace-recovery" aria-label="恢复工作区权限">
    <h4>{prefix} {recovery.missing_rights.join('、')}</h4>
    <p>工作区预检已停止此操作，尚未派发计算。受影响的目录为 <code>{recovery.workspace}</code>。</p>
    {recovery.preparation_detail && <p>{recovery.preparation_detail}</p>}
    {recovery.checked_at && <p>权限检查时间 <time dateTime={recovery.checked_at}>{recovery.checked_at}</time></p>}
    {cached && <p>本次拦截沿用上次记录。若已修复，直接在原会话发送“重新检查工作区并继续建模”来核实当前权限。</p>}
    <p>在启动 Harness 的同一 Windows 账户下打开普通 PowerShell，运行以下命令查看权限。</p>
    <pre>{recovery.diagnose_command}</pre>
    {recovery.can_prepare && recovery.prepare_command ? <>
      <p>若仍缺权限，运行以下命令仅补齐这个目录自身的缺失项。</p>
      <pre>{recovery.prepare_command}</pre>
      <p>确认输出的 <code>prepared</code> 为 <code>true</code> 后，在原页面重试。若已进入建模对话，发送“重新检查工作区并继续建模”。</p>
    </> : <p>请在项目的 workspaces 下选择自己拥有的独立 NTFS 工作目录，在该目录使用已有源码或上传附件。若要继续使用原目录，请联系目录所有者。</p>}
  </section>;
}

export function PlantResult({ phase, block, download }) {
  const [downloading, setDownloading] = useState(false);
  const [downloadMessage, setDownloadMessage] = useState('');
  if (phase !== 'result') return <p>正在专用 MATLAB 中构建模型…</p>;
  const text = (block.content ?? []).filter(item => item.type === 'text').map(item => item.text).join('\n');
  let result;
  try { result = JSON.parse(text.replace(/^Error:\s*/, '')); } catch { /* Native approvals and dispatch errors are plain text. */ }
  if (!result || typeof result.stage !== 'string') return <p role="status">{text || '工具尚未返回可核实的模型记录。'}</p>;
  if (result.stage === 'workspace_preflight' && result.diagnostic?.workspace_recovery)
    return <WorkspaceRecovery recovery={workspaceRecoveryView(result.diagnostic,result.preflight_observation)}/>;
  if (result.code === 'PLANT_DELIVERY_COMPLETE' && result.stage === 'delivery_complete' && result.verified_delivery?.path)
    return <p role="status">本轮模型已保存、重载并完成仿真。Host 已停止这次额外调用。
      <br/><code>{result.verified_delivery.path}</code></p>;
  const passed = result.status === 'simulated' && !block.isError;
  return <section className="control-card" aria-label="Simulink 建模结果">
    <h4>{passed ? '模型已重载并完成仿真' : '建模在「' + stageName(result.stage) + '」停止'}</h4>
    {Array.isArray(result.verified) && result.verified.length > 0 &&
      <p>已验证　{result.verified.map(stageName).join('、')}</p>}
    <p>{result.file_exists === true ? '目标文件已存在' : result.file_exists === false ? '目标路径尚无文件' : '文件状态待核实'}
      {result.file_exists === true && result.file_created_by_operation === false && '，本次未创建'}
      {result.file_exists === true && result.file_created_by_operation === null && '，归属待核实'}
      {result.path && <><br/><code>{result.path}</code></>}</p>
    {passed && result.file_exists && result.sha256 && <button type="button" disabled={downloading}
      onClick={async () => {
        setDownloading(true);
        try {
          await download(result.path, result.sha256, result.model + '.slx', result.bytes);
          setDownloadMessage('文件摘要校验通过，已提交浏览器下载。');
        } catch (error) { setDownloadMessage(error.message); }
        finally { setDownloading(false); }
      }}>下载已验证的 SLX</button>}
    {downloadMessage && <p role="status">{downloadMessage}</p>}
    {result.comparison?.compared === true && <p>已比较 {result.comparison.samples} 个状态采样点，最大绝对误差 {result.comparison.max_abs_error}，容差 {result.comparison.tolerance}。</p>}
    {passed && !result.comparison?.compared && <p>本次没有源程序的逐点比较记录。</p>}
    {!passed && <><p role="alert">{[result.identifier, result.reason].filter(Boolean).join('　')}</p>
      {result.verification_error && result.verification_error !== result.reason && <p>{result.verification_error}</p>}
      {Array.isArray(result.changes) && result.changes.length > 0 && <p>现场修改　{result.changes.join('、')}</p>}
      {result.session_verified && <p>MATLAB 已核验，可以在原会话继续。</p>}
      {['quarantined','pending_verification'].includes(result.status) && <p>原调用的终态待核验，仍可查看工作区文件和日志。</p>}
      <p>{result.recovery || (result.retryable && result.session_verified ? '读回当前模型，修正失败步骤。' : '保留现场，核对现有文件和工具诊断。')}</p></>}
  </section>;
}
