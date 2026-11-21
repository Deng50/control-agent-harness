import React, { useEffect, useRef, useState } from 'react';
import { labels } from './form-schema.mjs';

const scopeNames = { signals: '信号、材料与接口', sampling: '采样与验证工况', training: '训练或设计配置' };
const batchNames = {trial:'试采',full:'全量采样',run:'本次执行'};
const grantStates = {pending:'待授权',active:'可用',running:'正在执行',expired:'已到期',consumed:'额度已用完',revoked:'已撤销'};
const fieldNames = { interface: '接口', signals: '信号', features: '输入特征', targets: '预测目标', states: '状态',
  controls: '控制量', outputs: '输出', disturbances: '扰动', name: '名称', unit: '单位', task: '学习目标',
  sample_time: '采样周期', timestep: '步长', worksheet: '工作表', split: '数据划分', scaler: '数据缩放',
  hidden_dims: '隐藏层宽度', learning_rate: '学习率', batch_size: '批量大小', epochs: '目标轮次', seed: '随机种子',
  input_dim: '输入宽度', output_dim: '输出宽度', offset: '时间偏移', column: '数据列', window: '历史窗口',
  train: '训练集', validation: '验证集', test: '测试集', mode: '方式', source: '来源', operation: '操作',
  activation: '激活函数', patience: '早停耐心轮次', weight_decay: '权重衰减', min_delta: '最小验证改善量',
  grad_clip: '梯度裁剪阈值', max_test_nrmse: '测试 NRMSE 上限',
  lower: '下限', upper: '上限', config: '阶段配置', next_training: '后续训练', stop_after: '交付停止点' };
const valueText = value => value === undefined ? '未设置' : typeof value === 'string' ? value : JSON.stringify(value);
const payloadOf = row => row.reviewPayload ?? row.review_payload ?? row.expectedPayload;

function Fields({ value }) {
  if (!value || typeof value !== 'object') return valueText(value);
  const entries = Object.entries(value);
  if (!entries.length) return Array.isArray(value) ? '[]' : '{}';
  if (Array.isArray(value) && value.every(v => typeof v !== 'object' || v === null)) return valueText(value);
  return <dl className="control-review-fields">{entries.map(([key, val]) =>
    <React.Fragment key={key}><dt>{Array.isArray(value) ? Number(key) + 1 : Object.hasOwn(fieldNames,key) ? fieldNames[key] : key}</dt>
      <dd><Fields value={val}/></dd></React.Fragment>)}</dl>;
}

/** Both surfaces display the Host snapshot. Neither a tool output nor a browser draft can confirm a plan. */
export function ConfirmationCard({ card, disabled, onConfirm, onRefresh, inChat=false, recovery=false }) {
  const rows = card.requirements;
  const signed = rows.length > 0 && rows.every(row => card.statuses?.[row.scope]?.confirmation_state === 'confirmed');
  const complete = signed && (!card.matlab_scope || ['active','running'].includes(card.matlab_grant?.state));
  return <section className="control-confirmation" aria-label="工程确认卡">
    <h4>{labels[card.operation] ?? card.operation}</h4>
    <p>{card.workspace}</p>
    {card.operation==='prepare_supervised_dataset' && <p>本次范围：{rows.some(row=>row.scope==='training')?'准备数据集，训练并导出网络':'准备并交付数据集'}</p>}
    {rows.length === 0 && <p>此操作无需工程确认。</p>}
    {rows.map(row => {
      const status = card.statuses?.[row.scope], changes = card.changes?.[row.scope] ?? [];
      return <details key={row.scope} open>
        <summary>{scopeNames[row.scope] ?? row.scope} · {status?.confirmation_state === 'confirmed' ? '已确认' : '待确认'}</summary>
        {changes.length > 0 && <div className="control-review-changes"><strong>与上一版的差异</strong><ul>
          {changes.map(change => <li key={change.path}>{change.path}：{valueText(change.before)} → {valueText(change.after)}</li>)}
        </ul></div>}
        <Fields value={payloadOf(row)}/>
        {row.effective_settings && <div><strong>实际训练参数（含自动取值）</strong>
          <Fields value={row.effective_settings}/></div>}
        {row.reviewBindings && Object.keys(row.reviewBindings).length > 0 && <details><summary>来源与依赖版本</summary>
          <Fields value={row.reviewBindings}/></details>}
      </details>;
    })}
    {card.matlab_scope && <section aria-label="专用 MATLAB 任务授权"><h4>本次 MATLAB 执行范围</h4>
      <p>{card.matlab_scope.operation} · {card.matlab_scope.variants.map(value=>batchNames[value]??value).join(' / ')} · 有效期 {card.matlab_scope.ttl_seconds} 秒 · 单次调用含排队上限 {card.matlab_scope.dispatch_timeout_seconds} 秒</p>
      <p>授权状态：{grantStates[card.matlab_grant?.state]??'待核验'}。{card.matlab_grant?.remaining?.length>0 && `可用步骤：${card.matlab_grant.remaining.map(value=>batchNames[value]??value).join('、')}。`}</p>
      <p>专用进程 PID {card.matlab_scope.dedicated_session.pid}，写入目录 {card.matlab_scope.write_directory}。原生工具会按当前会话策略审批。</p>
      <details><summary>程序、材料、批函数与授权状态</summary><Fields value={card.matlab_scope}/><Fields value={card.matlab_grant}/></details>
    </section>}
    {rows.length > 0 && <div className="control-row">
      <button type="button" disabled={disabled || complete} onClick={onConfirm}>{complete ? '以上摘要已确认' : card.matlab_scope ? (signed ? '重新授权本次 MATLAB 操作' : '确认摘要并授权本次 MATLAB 操作') : '确认以上摘要'}</button>
      {onRefresh && <button type="button" disabled={disabled} onClick={onRefresh}>读取原生确认状态</button>}
    </div>}
    {complete && <p>{card.matlab_grant?.state==='running'?'本批正在执行，请查看原生作业记录。':inChat?'已确认。在对话中发送“继续”，即可按此方案执行。':recovery?'已确认。请选择重试或续接原任务。':'已确认。点击“执行阶段”提交当前操作。'}</p>}
  </section>;
}

export function recordedCard(content) {
  if (content?.length !== 1 || content[0]?.type !== 'text' || typeof content[0].text !== 'string' || !content[0].text.includes('"card_id"')) return null;
  try {
    const {workspace, card_id} = JSON.parse(content[0].text);
    if (typeof workspace === 'string' && typeof card_id === 'string' && card_id) return { workspace, card_id };
  } catch { /* An incomplete or failed proposal cannot become a clickable confirmation. */ }
  return null;
}

function ChatCard({ handle, loadCard, confirmCard, watchCard }) {
  const [card, setCard] = useState(null), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  const version = useRef(0), submitting = useRef(false);
  async function readOrConfirm(confirm) {
    if (submitting.current) return;
    const generation = ++version.current;
    if (confirm) submitting.current = true;
    setBusy(true); setError('');
    try {
      const result = await (confirm ? confirmCard : loadCard)(handle);
      if (generation !== version.current) return;
      if (result.card_id !== handle.card_id || !Array.isArray(result.requirements)) throw new Error('收到的确认卡与当前版本不匹配');
      setCard(result);
    } catch (err) {
      if (generation === version.current) { setCard(null); setError(err.message ?? String(err)); }
    } finally {
      if (generation === version.current) { setBusy(false); submitting.current = false; }
    }
  }
  useEffect(() => {
    const stop = watchCard(handle,result=>{
      version.current++; submitting.current=false; setBusy(false); setError(''); setCard(result);
    });
    setCard(null); void readOrConfirm(false);
    return () => { stop(); version.current++; submitting.current = false; };
  }, [handle.workspace, handle.card_id, loadCard, confirmCard, watchCard]);
  return <div className="control-chat-review">
    {card ? <ConfirmationCard card={card} inChat disabled={busy} onRefresh={()=>readOrConfirm(false)} onConfirm={()=>readOrConfirm(true)}/> :
      <p>{error || '正在读取当前会话的工程确认卡…'}{error && <button type="button" onClick={()=>readOrConfirm(false)}>重新读取</button>}</p>}
  </div>;
}

export function ChatReviewTool({ phase, block, loadCard, confirmCard, watchCard }) {
  if (phase !== 'result') return <p>正在生成工程确认卡…</p>;
  if (block.isError) return <p role="alert">{block.content?.filter(item=>item.type==='text').map(item=>item.text).join('\n') || '生成确认卡失败'}</p>;
  const handle = recordedCard(block.content);
  return handle ? <ChatCard key={handle.card_id} handle={handle} loadCard={loadCard} confirmCard={confirmCard} watchCard={watchCard}/> : <p>本次审阅没有可确认的版本。</p>;
}

// Turn tails remain visible when Harness folds the completed Turn's tool calls.
export const reviewTurnDefinition = {
  kind: 'control-engineering-review', target: 'chat',
  match(event) {
    if (event.type === 'tool/call' && event.data.name === 'control_prepare_review') return {id:event.data.callId,role:'start'};
    if (event.type === 'tool/result' && event.surfaceOp === 'append' && !event.data.message.isError && recordedCard(event.data.message.content))
      return {id:event.data.message.source.callId,role:'update'};
    if (event.type === 'tool/ptc-dispatch' && event.data.name === 'control_prepare_review' && !event.data.isError && recordedCard(event.data.content))
      return {id:event.data.subCallId,role:'start'};
    return null;
  },
  start(_context,{event}) { return {handle:event.type==='tool/ptc-dispatch' ? recordedCard(event.data.content) : null}; },
  update({state},{event}) { return event.type==='tool/result' ? {handle:recordedCard(event.data.message.content)} : state; },
  buildViewNode(context) {
    if (!context.state?.handle || !context.start) return null;
    return {key:context.key,kind:'control-engineering-review',id:context.id,target:'chat',anchorSeq:context.start.event.seq,
      location:context.start.location,visibility:'hidden',data:context.state.handle};
  },
};

export function ChatReviewTail({turn, useReviewCards, loadCard, confirmCard, watchCard}) {
  const handles = useReviewCards(String(turn.turn));
  const unique = new Map((handles??[]).map(handle=>[handle.card_id,handle]));
  return Array.from(unique.values(), handle=><ChatCard key={handle.card_id} handle={handle} loadCard={loadCard} confirmCard={confirmCard} watchCard={watchCard}/>);
}
