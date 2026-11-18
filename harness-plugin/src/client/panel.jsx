import React, { useEffect, useMemo, useRef, useState } from 'react';
import './panel.css';
import { ArtifactReport } from './report.jsx';
import { ParameterFields } from './form-fields.jsx';
import { applySummary, initialDraft as newDraft, inputLabel, labels, normalizedResult, operationDefaults, operationGroups, operations, parseObject, reportReadOperation, requestSignature, roles as roleMetadata, setPath, signature, trainingOps, uploadAccept, workspaceKey } from './form-schema.mjs';

const roles = roleMetadata.map(role => ({ ...role, operations: [...operationGroups(role.id).flatMap(group => group.names).map(name => [name, labels[name] ?? name]), ['plan_workflow','预检显式工作流'], ['run_workflow','执行已审阅工作流'],['demo','完整仿真演示（需摘要确认）']] }));
const storageKey = 'control-engineering.v1';
const handoffs = Object.fromEntries(['transition_dataset','nominal_model','corrected_model','identified_model','learned_model','controller','controller_design','expert_dataset','imitation_policy','supervised_dataset','supervised_model','plant_model','sampled_dataset','external_expert','external_expert_dataset','external_imitation_policy','conditional_ode_problem','pinn_model','differentiable_validation','external_pgac_candidate','external_optimized_policy'].map(kind => [kind,true]));
const initialDraft = newDraft;
const defaults = new Proxy({}, { get: (_,name) => operationDefaults(name) });
const requestAgent = (role,operation) => ['plan_workflow','run_workflow','demo'].includes(operation) ? 'workflow' : role;
function useScopedState(key,initial) {
  const [values,setValues] = useState({});
  return [Object.hasOwn(values,key) ? values[key] : initial, next => setValues(previous => ({ ...previous, [key]: typeof next === 'function' ? next(Object.hasOwn(previous,key) ? previous[key] : initial) : next }))];
}
function load() {
  try { const saved = JSON.parse(localStorage.getItem(storageKey)); if (saved?.version === 1) return { ...saved, projects: Object.fromEntries(Object.entries(saved.projects ?? {}).map(([key,value]) => { const index=key.lastIndexOf('::'); return [workspaceKey(key.slice(0,index),key.slice(index+2)),{...initialDraft(),...value}]; })) }; } catch { /* A malformed browser draft cannot become a task. */ }
  return { version: 1, workspace: '', role: 'model', projects: {} };
}
const json = parseObject;
function events(text) { return text.split('\n').flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } }); }
const pretty = value => JSON.stringify(value, null, 2);

export function ControlIcon({ size = 20 }) { return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"><path d="M3 18h18M5 15l4-8 5 5 5-9"/><circle cx="9" cy="7" r="2"/><circle cx="14" cy="12" r="2"/></svg>; }

/** Keeps only form drafts in browser storage; authoritative tasks live in Harness. */
export function ControlPanel({ createSession, openSession, listDirectory, submit, review, confirm, planStatus, status, history, recover, upload, download, downloadVerifiedFile, prompt, watchJobs, observeJob, cancel, useJobs }) {
  const [state, renderState] = useState(load);
  const stateRef=useRef(state),persistenceError=useRef(''),sessionCache=useRef(new Map()),sessionPending=useRef(new Map()),artifactSelection=useRef(new Map()),artifactEpoch=useRef(new Map()),previewEpoch=useRef(new Map()),tableEpoch=useRef(new Map());
  const role = roles.find(item => item.id === state.role) ?? roles[0];
  const key = workspaceKey(state.workspace, role.id);
  const draft = state.projects[key] ?? initialDraft();
  if(!artifactSelection.current.has(key)&&draft.artifactId)artifactSelection.current.set(key,draft.artifactId);
  const [busy,setBusy] = useScopedState(key,false), [error,setError] = useScopedState(key,'');
  const [uploadProgress,setUploadProgress] = useScopedState(key,''), [uploadController,setUploadController] = useScopedState(key,null);
  const [directory,setDirectory] = useScopedState(key,null), [attempts,setAttempts] = useScopedState(key,[]), [selectedAttempt,setSelectedAttempt] = useScopedState(key,null);
  const [reportResult,setReportResult] = useScopedState(key,null), [filePreview,setFilePreview] = useScopedState(key,null), [downloaded,setDownloaded] = useScopedState(key,null);
  const [fieldErrors,setFieldErrors] = useScopedState(key,{}), [summaryEdits,setSummaryEdits] = useScopedState(key,{}), [supportProgress,setSupportProgress] = useScopedState(key,'');
  const [handoffResult,setHandoffResult]=useScopedState(key,null),[artifactOptions,setArtifactOptions]=useScopedState(key,{});
  const [tablePreview,setTablePreview]=useScopedState(key,null);
  useEffect(()=>setFieldErrors({}),[key,draft.operation]);
  const jobs = useJobs(value => value);
  // Native navigation can unmount this panel before a passive effect runs.
  // Merge into the latest synchronous draft and save before returning a handle.
  function setState(update) {
    const next=typeof update==='function'?update(stateRef.current):update;
    stateRef.current=next;
    try {localStorage.setItem(storageKey,JSON.stringify(next));persistenceError.current='';}
    catch(failure){persistenceError.current='浏览器草稿未能保存；原生任务记录仍存在。请先解决存储问题再打开会话，以免丢失关联。';setError(persistenceError.current);}
    renderState(next);
  }
  const patch = update => setState(value => ({ ...value, projects: { ...value.projects, [key]: { ...(value.projects[key] ?? initialDraft()), ...update } } }));
  useEffect(() => draft.sessionId ? watchJobs(draft.sessionId) : undefined, [draft.sessionId, watchJobs]);
  useEffect(() => draft.sessionId && draft.jobId ? observeJob(draft.sessionId, draft.jobId) : undefined, [draft.sessionId, draft.jobId, observeJob]);
  useEffect(() => () => uploadController?.abort(), [uploadController]);
  useEffect(() => () => uploadController?.abort(), [key]);
  const rows = jobs.rows[draft.sessionId] ?? [];
  const observed = jobs.observed[draft.jobId];
  const stream = useMemo(() => events(observed?.text ?? ''), [observed?.text]);
  const result = normalizedResult(selectedAttempt ? selectedAttempt.result : stream.findLast(item => item.event === 'result')?.result);
  const selectedReport = reportResult??(result?.report?result:null);
  useEffect(() => { if(result?.report && result?.files&&(!artifactSelection.current.has(key)||artifactSelection.current.get(key)===result.artifact_id)) { artifactSelection.current.set(key,result.artifact_id);setReportResult(result); patch({ artifactId: result.artifact_id }); } }, [result?.artifact_id,Boolean(result?.report)]);
  useEffect(()=>{if(result?.artifacts)setArtifactOptions(result.artifacts);},[key,signature(result?.artifacts??{})]);
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
    const current=stateRef.current.projects[key]??draft;
    if (!forWork && current.sessionId) return current.sessionId;
    const existing=current.roleSessionId??sessionCache.current.get(key);
    if (existing) { patch({ sessionId:existing });if(persistenceError.current)throw new Error(persistenceError.current);return existing; }
    if(sessionPending.current.has(key))return sessionPending.current.get(key);
    const pending=(async()=>{const id=await createSession(state.workspace.trim(),`${role.title} · ${state.workspace.trim().split(/[\\/]/).at(-1)}`,role.id);
      sessionCache.current.set(key,id);const latest=stateRef.current.projects[key]??draft;
      const legacySessionIds=[...new Set([...(latest.legacySessionIds??[]),latest.sessionId].filter(value=>value&&value!==id))];patch({sessionId:id,roleSessionId:id,legacySessionIds,jobId:'',attemptId:''});if(persistenceError.current)throw new Error(persistenceError.current);return id;})();
    sessionPending.current.set(key,pending);try{return await pending;}finally{sessionPending.current.delete(key);}
  }
  async function newProfessionalSession() {
    if(!state.workspace.trim())throw new Error('请先填写绝对工作目录');
    if(sessionPending.current.has(key))throw new Error('当前角色会话正在创建，请等待原请求结束');
    const pending=(async()=>{
      const id=await createSession(state.workspace.trim(),`${role.title} · ${state.workspace.trim().split(/[\\/]/).at(-1)}`,role.id);
      const latest=stateRef.current.projects[key]??draft;
      const legacySessionIds=[...new Set([...(latest.legacySessionIds??[]),latest.roleSessionId,latest.sessionId].filter(value=>value&&value!==id))];
      let configText=latest.config;try{const stripRefs=value=>Array.isArray(value)?value.map(stripRefs):value&&typeof value==='object'?Object.fromEntries(Object.entries(value).filter(([name])=>name!=='plan_refs').map(([name,item])=>[name,stripRefs(item)])):value;configText=pretty(stripRefs(json(configText)));}catch{/* Invalid drafts stay editable and cannot run. */}
      artifactSelection.current.delete(key);for(const epoch of [artifactEpoch,previewEpoch,tableEpoch])epoch.current.set(key,(epoch.current.get(key)??0)+1);
      sessionCache.current.set(key,id);setState(value=>({...value,projects:{...value.projects,[key]:{...initialDraft(),operation:latest.operation,mode:latest.mode??'normal',inputs:latest.inputs,config:configText,sessionId:id,roleSessionId:id,legacySessionIds}}}));setSelectedAttempt(null);setReportResult(null);setHandoffResult(null);setFilePreview(null);setDownloaded(null);setTablePreview(null);setArtifactOptions({});setUploadProgress('');setSupportProgress('');
      if(persistenceError.current)throw new Error(persistenceError.current);
      await openSession(id);return id;
    })();
    sessionPending.current.set(key,pending);try{return await pending;}finally{sessionPending.current.delete(key);}
  }
  function selectOperation(operation) {
    const preset = defaults[operation] ?? { inputs: {}, config: {} };
    editRequest({ operation, inputs: pretty(preset.inputs), config: pretty(preset.config) });setFieldErrors({});
  }
  const currentRequest = () => ({ agent: requestAgent(role.id,draft.operation), operation:draft.operation, workspace:state.workspace.trim(), inputs:json(draft.inputs), config:json(draft.config) });
  async function support(operation,inputs={},config={}) {
    const sessionId=await session(operation==='propose_case_plan'), started=await submit(sessionId,{agent:requestAgent(role.id,operation),operation,workspace:state.workspace.trim(),inputs,config});
    const current=stateRef.current.projects[key]??draft;patch({supportJobs:[...(current.supportJobs??[]).slice(-19),{...started,operation,sessionId}]});
    setSupportProgress(`${operation} · ${started.jobId}`);
    try { const deadline=Date.now()+1800000;
      for(;;) { const value=await status(sessionId,state.workspace.trim(),started.attemptId),record=value.record;
        if(record.owner!==sessionId||record.nativeJobId!==started.jobId) throw new Error('原生作业归属不匹配');
        if(['completed','failed','killed'].includes(record.state)&&record.outcome?.cleanupVerified===true) { if(record.state!=='completed') throw new Error(record.detail ?? record.state); return normalizedResult(record.result); }
        if(record.state==='interrupted') throw new Error(`执行归属丢失；保留原句柄 ${started.attemptId}，不会自动重提`);
        if(Date.now()>deadline) throw new Error(`观察超时；请读取原任务 ${started.attemptId}，不会自动重提`);
        await new Promise(done=>setTimeout(done,350));
      }
    } finally { setSupportProgress(''); }
  }
  async function prepareReview() {
    const request=currentRequest(),sessionId=await session(true),actual=await review(sessionId,request),refs={},statuses={};
    const snapshot={sessionId,requestSignature:requestSignature(request),requirements:actual.requirements,refs:{},statuses:{}};
    patch({review:snapshot});
    for(const row of actual.requirements) {
      const plan=await support('propose_case_plan',{}, {scope:row.scope,payload:row.expectedPayload,bindings:row.bindings,dependencies:structuredClone(refs)});
      refs[row.scope]={plan_id:plan.plan_id,version_hash:plan.version_hash};statuses[row.scope]=await planStatus(sessionId,state.workspace.trim(),refs[row.scope]);
      patch({review:{...snapshot,refs:structuredClone(refs),statuses:structuredClone(statuses)}});
    }
    patch({config:pretty({...request.config,plan_refs:refs}),review:{...snapshot,refs,statuses}});
  }
  async function confirmReview() {
    const snapshot=draft.review,request=currentRequest(),sessionId=await session(true),statuses={};
    if(!snapshot||snapshot.requestSignature!==requestSignature(request)||snapshot.sessionId!==sessionId) throw new Error('请求或会话已改变，请重新生成实际摘要');
    for(const row of snapshot.requirements) { const value=await planStatus(sessionId,state.workspace.trim(),snapshot.refs[row.scope]);
      statuses[row.scope]=value.confirmation_state==='confirmed'?value:{...await confirm(sessionId,state.workspace.trim(),snapshot.refs[row.scope]),confirmation_state:'confirmed'}; }
    patch({review:{...snapshot,statuses}});
  }
  async function refreshReview() { const snapshot=draft.review;if(!snapshot)return;const sessionId=await session(true),statuses={};
    for(const row of snapshot.requirements) statuses[row.scope]=await planStatus(sessionId,state.workspace.trim(),snapshot.refs[row.scope]);patch({review:{...snapshot,statuses}}); }
  async function run(request) {
    if(Object.values(fieldErrors).some(Boolean)) throw new Error('请先完成结构化字段中的 JSON');
    const target=request??currentRequest();
    if(target.operation==='inspect_table'){tableEpoch.current.set(key,(tableEpoch.current.get(key)??0)+1);setTablePreview(null);}
    const sessionId=await session(true),actual=await review(sessionId,target),snapshot=draft.review;
    if(actual.requirements.length) {
      if(!snapshot||snapshot.sessionId!==sessionId||snapshot.requestSignature!==requestSignature(target)||signature(snapshot.requirements)!==signature(actual.requirements)) throw new Error('请先审阅并确认本次实际摘要；尚未派发计算。');
      for(const row of actual.requirements) if((await planStatus(sessionId,state.workspace.trim(),snapshot.refs[row.scope])).confirmation_state!=='confirmed') throw new Error('当前版本尚未确认；尚未派发计算。');
      target.config.plan_refs=snapshot.refs;
    }
    const started=await submit(sessionId,target);setSelectedAttempt(null);patch({sessionId,jobId:started.jobId,attemptId:started.attemptId});
  }
  async function inspectArtifact(id) { const token=(artifactEpoch.current.get(key)??0)+1;artifactEpoch.current.set(key,token);artifactSelection.current.set(key,id);setFilePreview(null);setDownloaded(null);setHandoffResult(null);
    const value=await support('inspect_artifact',{artifact_id:id});if(artifactEpoch.current.get(key)===token){setReportResult(value);patch({artifactId:id});}return value; }
  async function inspectUploadedTable() { const token=(tableEpoch.current.get(key)??0)+1;tableEpoch.current.set(key,token);
    const value=await support('inspect_table',{data_path:draft.lastUpload.path},{preview_rows:5});if(tableEpoch.current.get(key)===token)setTablePreview(value);return value; }
  async function readFile(id,name,config={}) { const token=(previewEpoch.current.get(key)??0)+1;previewEpoch.current.set(key,token);const value=await support(reportReadOperation(name,config),{artifact_id:id,file:name},config);if(artifactSelection.current.get(key)===id&&previewEpoch.current.get(key)===token)setFilePreview({...value.file_preview,artifact_id:id});return value; }
  async function downloadBundle(id) { const descriptor=await support('prepare_bundle_archive',{artifact_id:id}),value=await download(descriptor);if(artifactSelection.current.get(key)===id)setDownloaded(value);return value; }
  async function downloadFile(artifact,file) {
    const known=artifact.files?.find(row=>row.name===file.name&&row.path===file.path),digest=artifact.metadata?.files?.[file.name];
    if(!known||!file.path||!/^[a-f0-9]{64}$/.test(digest??''))throw new Error('只允许下载已检查产物文件列表中的完整文件及已绑定 SHA-256');
    const value=await downloadVerifiedFile(file.path,digest,`${artifact.artifact_id}-${file.name.replace(/[\\/]/g,'_')}`,file.bytes);
    if(artifactSelection.current.get(key)===artifact.artifact_id)setDownloaded(value);return value;
  }
  function editRequest(change) { let next={...change};try {const config=json(next.config??draft.config),inputs=json(next.inputs??draft.inputs),operation=next.operation??draft.operation;
      const unchanged=requestSignature({agent:requestAgent(role.id,operation),operation,workspace:state.workspace.trim(),inputs,config})===requestSignature(currentRequest());
      if(!unchanged){delete config.plan_refs;next={...next,config:pretty(config),review:null};}else if(draft.review?.refs)next.config=pretty({...config,plan_refs:draft.review.refs});
    }catch {next.review=null;}patch(next); }
  async function recoverAttempt(record, mode) {
    const sessionId = await session();
    if(record.owner!==sessionId||record.recoveryBlocked||record.outcome?.cleanupVerified!==true) throw new Error('原会话归属或资源退出尚未核实，不能恢复');
    const started = await recover(sessionId, state.workspace.trim(), record.attemptId, mode);
    setSelectedAttempt(null);
    patch({ sessionId, jobId: started.jobId, attemptId: started.attemptId });
  }
  function parameter(name,value) { try { editRequest({ config:pretty(setPath(json(draft.config),name,value)) }); } catch(failure) {setError(failure.message);} }
  let config = {}; try { config = json(draft.config); } catch { /* JSON editor keeps incomplete input until submission. */ }
  function adoptHandoff(plan) {
    const { agent: next, operation, inputs, config_suggestions } = plan.handoff;
    if(!operation) return;
    const destination = workspaceKey(state.workspace,next);
    const preset = defaults[operation];
    setState(value => ({ ...value, role: next, projects: { ...value.projects, [destination]: { ...(value.projects[destination] ?? initialDraft()), operation,
      inputs: pretty(inputs), config: pretty({ ...preset.config, ...config_suggestions }),review:null } } }));
    setSelectedAttempt(null);
  }
  function handoff(artifactId) { return action(async()=>setHandoffResult(await support('prepare_handoff',{artifact_id:artifactId}))); }
  const handoffPlan=handoffResult??result;
  useEffect(()=>{if(!draft.artifactId||!draft.roleSessionId||reportResult)return;let active=true;
    inspectArtifact(draft.artifactId).catch(failure=>{if(active)setError(`报告恢复失败：${failure.message}`);});return()=>{active=false;};},[key,draft.artifactId,draft.roleSessionId]);
  useEffect(()=>{if(!draft.sessionId||!draft.attemptId)return;let active=true,timer;
    const poll=async()=>{try{const value=await status(draft.sessionId,state.workspace.trim(),draft.attemptId);if(!active)return;setSelectedAttempt(value.record);
      if(['completed','failed','killed'].includes(value.record.state)&&value.record.outcome?.cleanupVerified===true||value.record.state==='interrupted')return;
    }catch(failure){if(active)setError(`持久任务读取失败：${failure.message}`);}if(active)timer=setTimeout(poll,1200);};void poll();return()=>{active=false;clearTimeout(timer);};},[key,draft.sessionId,draft.attemptId]);
  useEffect(()=>{if(!draft.review?.requirements?.length||!draft.roleSessionId)return;let active=true;const saved=draft.review;
    Promise.all(saved.requirements.map(async row=>[row.scope,await planStatus(saved.sessionId,state.workspace.trim(),saved.refs[row.scope])])).then(rows=>{if(active)patch({review:{...saved,statuses:Object.fromEntries(rows)}});}).catch(failure=>{if(active){patch({review:{...saved,statuses:{}}});setError(`确认版本需重新审阅：${failure.message}`);}});return()=>{active=false;};},[key,draft.roleSessionId,draft.review?.requestSignature]);
  return <div className="control-panel">
    <header><p className="control-kicker">CONTROL ENGINEERING / HARNESS</p><h1>模型、控制器与轻量化</h1><p>按所需阶段开始或结束。三个角色各自使用原生会话，计算由 Harness 后台作业执行。</p></header>
    <label className="control-workspace">项目工作目录<input placeholder="D:\projects\control-agent-harness\workspaces\project-01" value={state.workspace} onChange={event => setState(value => ({ ...value, workspace: event.target.value }))}/><small>Windows 工作区需由当前用户拥有并授予其完全控制，供 Harness 配置沙箱。路径改变会切换该项目的会话与草稿。</small></label>
    <button disabled={busy} onClick={() => action(async () => setDirectory(await listDirectory(state.workspace.trim() || undefined)))}>浏览工作目录</button>
    {directory && <section className="control-card" aria-label="工作目录浏览"><div className="control-row">{directory.crumbs.map(crumb => <button key={crumb.path} onClick={() => action(async () => setDirectory(await listDirectory(crumb.path)))}>{crumb.name}</button>)}</div><div className="control-directory">{directory.entries.map(entry => <button key={entry.path} onClick={() => action(async () => setDirectory(await listDirectory(entry.path)))}>{entry.name}</button>)}</div>{directory.truncated && <small>目录过多，当前仅显示前一部分。</small>}<div className="control-row"><button onClick={() => { setState(value => ({ ...value, workspace: directory.path })); setDirectory(null); }}>选择此目录</button><button onClick={() => setDirectory(null)}>关闭浏览</button></div></section>}
    <nav className="control-stages">{roles.map((item,index) => <button key={item.id} aria-pressed={role.id === item.id} onClick={() => setState(value => ({ ...value, role: item.id }))}><span>0{index+1}</span><strong>{item.title}</strong><small>{item.sub}</small></button>)}</nav>
    {error && <div role="alert" className="control-error">{error}</div>}
    <div className="control-columns"><section className="control-card"><h2>{role.title}</h2>
      <div className="control-row"><label>操作<select aria-label="任务 / 阶段" disabled={busy} value={draft.operation} onChange={event => selectOperation(event.target.value)}>{role.operations.map(([value,label]) => <option key={value} value={value}>{label}</option>)}</select></label><button disabled={busy} onClick={() => action(async () => openSession(await session(true)))}>打开独立会话</button><button disabled={busy} onClick={()=>action(newProfessionalSession)}>新建专业会话并保留旧记录</button></div><small>新会话保留当前输入与数值配置；旧任务可在“查看会话任务”中读取，旧摘要确认不会迁移，新计算需重新确认。</small>
      <div className="control-row" role="group" aria-label="配置显示模式">{[['normal','普通模式'],['expert','专家模式']].map(([mode,label])=><button key={mode} aria-pressed={(draft.mode??'normal')===mode} onClick={()=>patch({mode})}>{label}</button>)}<small>显示模式共用同一份配置，切换不改变参数或确认。</small></div>
      <p>{operations[draft.operation]?.summary}</p>
      <fieldset disabled={busy}><legend>明确输入</legend>{Object.keys({...operations[draft.operation]?.inputsSchema?.properties,...(()=>{try{return json(draft.inputs);}catch{return {};}})()}).map(name=><label key={name}>{inputLabel(name)}<input list={name.endsWith('path')?undefined:'artifact-options-'+role.id} value={(()=>{try{return json(draft.inputs)[name]??'';}catch{return '';}})()} onChange={event=>editRequest({inputs:pretty({...json(draft.inputs),[name]:event.target.value})})}/></label>)}</fieldset><datalist id={'artifact-options-'+role.id}>{Object.entries(artifactOptions).map(([id,item])=><option key={id} value={id}>{item.kind} · {item.status}</option>)}</datalist>
      <details open={draft.mode==='expert'}><summary>高级输入 JSON</summary><label>输入产物<textarea disabled={busy} aria-label="输入产物" rows={5} spellCheck={false} value={draft.inputs} onChange={event=>editRequest({inputs:event.target.value})}/></label></details>
      {['register_material','inspect_material','inspect_table','inspect_model_source','register_model','register_python_model','import_transitions'].includes(draft.operation) && <label>上传{draft.operation === 'import_transitions' ? '数据 NPZ' : '对象材料'}<input type="file" disabled={busy} accept={uploadAccept} onChange={event => { const file = event.target.files?.[0]; if (!file) return; action(async () => { const id = await session(true); const controller = new AbortController(); setUploadController(controller); try { const receipt = await upload(id, file, controller.signal, progress => setUploadProgress(`已上传 ${(progress.loaded/1024).toFixed(0)} KB`)); patch({ inputs: pretty({ ...json(draft.inputs), [['import_transitions','inspect_table'].includes(draft.operation) ? 'data_path' : ['register_material','inspect_material'].includes(draft.operation) ? 'local_path' : 'model_path']: receipt.path }) }); if(draft.operation==='register_material') patch({config:pretty({...json(draft.config),source_kind:'upload',original_name:receipt.name}),review:null});patch({lastUpload:receipt});setUploadProgress(`${receipt.name} · ${receipt.bytes} 字节 · SHA-256 ${receipt.sha256}`); } finally { setUploadController(null); } }); }}/><small>{uploadProgress}</small>{uploadController && <button onClick={() => uploadController.abort()}>取消上传</button>}</label>}
      {draft.lastUpload&&/\.(csv|xlsx)$/i.test(draft.lastUpload.name)&&<button disabled={busy} onClick={()=>action(inspectUploadedTable)}>读取本角色刚上传文件的实际表头</button>}
      {['register_model','register_python_model'].includes(draft.operation) && config.model_spec?.interface && <ModelInterfaceEditor spec={config.model_spec} onChange={value => parameter('model_spec', value)}/>}
      {draft.operation === 'register_python_model' && <p>此操作会在 Harness 数值作业中执行已检查的独立 PyTorch 源码，核对形状、重复性与梯度，并保存源码快照。请先检查对象材料和接口声明。</p>}
      <fieldset disabled={busy}><ParameterFields operation={draft.operation} config={config} set={parameter} onInvalid={(name,value)=>setFieldErrors(previous=>({...previous,[name]:value}))}/></fieldset>
      <details open={draft.mode==='expert'}><summary>高级配置 JSON · 与普通表单共用</summary><label>阶段配置<textarea disabled={busy} aria-label="阶段配置" rows={9} spellCheck={false} value={draft.config} onChange={event => editRequest({ config: event.target.value })}/></label></details>
      {trainingOps.has(draft.operation) && <label>训练检查点（留空从头训练）<input value={config.resume_checkpoint ?? ''} placeholder="failed/.../latest.pt 或产物内的检查点" onChange={event => { const next = { ...config }; delete next.resume_checkpoint_sha256; if (event.target.value.trim()) next.resume_checkpoint = event.target.value.trim(); else delete next.resume_checkpoint; patch({ config: pretty(next) }); }}/><small>训练轮数是总目标。恢复会核对依赖、配置、优化器和随机状态。</small></label>}
      <section className="control-review" aria-label="实际摘要与原生确认"><h3>执行前审阅</h3><button disabled={busy||!state.workspace.trim()||Object.values(fieldErrors).some(Boolean)} onClick={()=>action(prepareReview)}>生成 / 更新实际摘要</button>
      {draft.review&&<><p>{draft.review.requirements.length ? `本阶段需要 ${draft.review.requirements.length} 项摘要确认。` : '此操作没有工程确认范围，可直接执行。'}</p>{draft.review.requirements.map(row=><details key={row.scope} open><summary>{{signals:'信号与来源',sampling:'采样 / 仿真工况',training:'训练 / 数值配置'}[row.scope]} · {draft.review.statuses[row.scope]?.confirmation_state??'提案中'}</summary><small>{draft.review.refs[row.scope]?.plan_id} · {draft.review.refs[row.scope]?.version_hash}</small><pre className="control-summary">{pretty(row.expectedPayload)}</pre>
      {row.scope!=='signals'&&<details><summary>编辑实际配置摘要</summary><textarea rows={8} value={summaryEdits[row.scope]??pretty(row.expectedPayload)} onChange={event=>setSummaryEdits(previous=>({...previous,[row.scope]:event.target.value}))}/><button disabled={busy} onClick={()=>action(async()=>editRequest({config:pretty(applySummary(config,summaryEdits[row.scope]??row.expectedPayload))}))}>应用到同一配置并重新审阅</button></details>}
      {row.scope==='signals'&&<small>信号来自本次输入及其确切版本；需要修改时编辑接口 / 映射或重新登记材料，再更新摘要。</small>}<details><summary>材料 / 数据 / 依赖版本</summary><pre className="control-summary">{pretty(row.bindings)}</pre></details></details>)}
      {draft.review.requirements.length>0&&<div className="control-row"><button disabled={busy||draft.review.requirements.some(row=>!draft.review.refs[row.scope])||draft.review.requirements.every(row=>draft.review.statuses[row.scope]?.confirmation_state==='confirmed')} onClick={()=>action(confirmReview)}>确认以上摘要</button><button disabled={busy} onClick={()=>action(refreshReview)}>读取原生确认状态</button></div>}</>}
      </section>{Object.values(fieldErrors).some(Boolean)&&<p className="control-error">结构化 JSON 尚未完整，请先修正再执行。</p>}{supportProgress&&<small>{supportProgress}</small>}
      <div className="control-row"><button className="control-primary" disabled={busy||Object.values(fieldErrors).some(Boolean)} onClick={() => action(() => run())}>{busy ? '提交中…' : '执行阶段'}</button><button disabled={busy} onClick={() => action(() => run({ agent: role.id, operation: 'inspect_workspace', workspace: state.workspace.trim() }))}>刷新产物</button></div>
      <details><summary>用对话调整任务</summary><textarea rows={3} value={draft.chat} placeholder="说明目标和希望修改的参数…" onChange={event => patch({ chat: event.target.value })}/><button disabled={busy || !draft.chat.trim()} onClick={() => action(async () => { const id = await session(true); await prompt(id, `当前角色：${role.title}。工作目录：${state.workspace.trim()}。\n当前请求：${pretty({ operation: draft.operation, inputs: json(draft.inputs), config: json(draft.config) })}\n用户要求：${draft.chat}\n使用 control_${role.id} 工具读取必要产物并执行；维度不一致时先解释并确认，不能默默截断或补零。`); })}>提交到 Harness 对话</button><small>使用 Harness 设置中的模型和凭据。</small></details>
      <details><summary>用户项目串联工作流</summary><p>选择含 steps 的流程 JSON。阶段通过 {`{"stage":"前一步ID"}`} 接力，也可直接填写已有产物 ID。先预检，再运行。</p><label>流程文件路径<input aria-label="流程文件路径" value={draft.workflowPath ?? ''} onChange={event => patch({ workflowPath: event.target.value })} placeholder="D:\project\workflow.json"/></label><div className="control-row">{[['plan_workflow','检查前置条件'],['run_workflow','载入流程请求以审阅']].map(([operation,label]) => <button key={operation} disabled={busy || !draft.workflowPath?.trim()} onClick={() => operation==='run_workflow' ? editRequest({operation,inputs:pretty({workflow_path:draft.workflowPath.trim()}),config:'{}'}) : action(() => run({ agent:'workflow',operation,workspace:state.workspace.trim(),inputs:{workflow_path:draft.workflowPath.trim()},config:{} }))}>{label}</button>)}</div><small>已完成阶段会保存产物。中断后在任务记录中选择续接；输入改变时须新建运行。</small></details>
      <details><summary>完整流程演示（明确的仿真对象）</summary><p>使用全新工作目录，串联残差训练、MPC 调参、专家采集和策略优化。演示数据不是实测数据。</p>{['linear','nonlinear'].map(name => <button key={name} disabled={busy} onClick={() => editRequest({operation:'demo',inputs:'{}',config:pretty({case:name})})}>{name === 'linear' ? '线性 4 状态 / 2 控制' : '非线性 2 状态 / 1 控制'}</button>)}</details>
    </section><section className="control-card"><h2>作业与结果</h2>
      {draft.legacySessionIds?.length > 0 && <label>查看会话任务<select aria-label="查看会话任务" value={draft.sessionId} onChange={event => { patch({ sessionId: event.target.value, jobId: '' }); setAttempts([]); setSelectedAttempt(null); }}><option value={draft.roleSessionId}>当前角色会话</option>{draft.legacySessionIds.map(id => <option key={id} value={id}>旧会话 · {id}</option>)}</select><small>旧会话及其任务记录仍可读取；新任务使用当前角色配置。</small></label>}
      <details><summary>持久任务记录与恢复</summary><button disabled={busy} onClick={() => action(async () => { const value = await history(await session(), state.workspace.trim()); setAttempts(value.attempts); })}>刷新任务记录</button><p>记录属于当前角色会话。重试按原请求重新执行；续训恢复检查点，流程续接复用已核验阶段。</p>{attempts.slice(0,20).map(record => <div className="control-artifact" key={record.attemptId}><strong>{record.request.operation} · {record.state}</strong><code>{record.attemptId}</code><small>{record.detail}</small><button onClick={() => { setSelectedAttempt(record); patch({ jobId: record.authority === 'native_job' ? record.nativeJobId : '', attemptId:record.attemptId }); }}>查看记录结果</button>{['failed','killed','interrupted'].includes(record.state) && <><button disabled={busy || !!record.recoveryBlocked || record.outcome?.cleanupVerified!==true} onClick={() => action(() => recoverAttempt(record, 'retry'))}>按原请求重试</button>{(record.checkpoint || record.workflowRun) && <button disabled={busy || !!record.recoveryBlocked || record.outcome?.cleanupVerified!==true} onClick={() => action(() => recoverAttempt(record, 'resume'))}>{record.workflowRun ? '从已完成阶段续接' : '从检查点续训'}</button>}</>}{record.recoveryBlocked && <small>{record.recoveryBlocked}</small>}</div>)}</details>
      <select aria-label="选择作业" value={draft.jobId} onChange={event => { patch({ jobId:event.target.value,attemptId:'' });setSelectedAttempt(null); }}><option value="">选择作业</option>{rows.map(item => <option key={item.id} value={item.id}>{item.label} · {item.status}</option>)}</select>
      {job && <div className="control-status"><strong>{job.status}</strong><span>{job.progress}</span>{!terminal && <button onClick={() => action(() => cancel(draft.sessionId, job.id))}>取消作业</button>}{terminal && job.status !== 'completed' && <small>读取原任务记录后显式重试 / 恢复；不自动重复派发。</small>}<small>{job.detail}</small></div>}
      {(observed?.error || observed?.gapBefore) && <p className="control-error">{observed.error ?? '较早日志已超出显示窗口，请查看工作区文件。'}</p>}
      <LossChart values={progress}/>
      <pre className="control-logs">{visibleLog || '执行阶段后，这里显示原生作业日志和指标。'}</pre>
      {logLines.length > 60 && <small>当前显示最近60行；较早的事件保存在工作目录的 .control-jobs 中。</small>}
      {selectedAttempt && <><p>已读取持久任务记录 · {selectedAttempt.request.operation} · {selectedAttempt.state}</p><details><summary>持久记录详情</summary><pre className="control-logs">{pretty(selectedAttempt)}</pre></details></>}
      {(tablePreview||((selectedAttempt?.request?.operation??draft.operation)==='inspect_table'&&result))&&<details open aria-label="实际数据表头"><summary>实际数据表头、列位置与工作表</summary><pre className="control-logs">{pretty(tablePreview??result)}</pre><small>只展示实际文件数据；名称不自动决定单位、信号角色或可观测性，请在数据接口中明确绑定。</small></details>}
      {result?.candidates && <div><h3>对象材料检查：{result.registration_ready ? '有完整候选，待确认' : '需要补充信息'}</h3><p>本次仅静态检查，未执行上传源码或原始 Simulink。</p><pre className="control-logs">{pretty({ source: result.source, candidates: result.candidates, missing: result.missing, uncertain: result.uncertain, errors: result.errors, runtime: result.runtime })}</pre>{result.editable_model_spec && <button onClick={() => patch({ operation: result.registration_operation ?? 'register_model', inputs: pretty({ model_path: result.source.path }), config: pretty({ model_spec: result.editable_model_spec, source: { kind: result.registration_operation === 'register_python_model' || result.source.format === 'json' ? 'user_nominal' : 'reconstructed_equations', description: result.registration_operation === 'register_python_model' ? 'User-reviewed standalone PyTorch dynamics source' : 'User-reviewed declared equations from inspected source' } }) })}>检查并修改模型接口</button>}</div>}
      {result?.sampling&&<div className="control-handoff"><h3>采样建议已生成，尚未执行源对象</h3><small>{result.scope}</small><button disabled={busy} onClick={()=>editRequest({operation:'sample_model',inputs:pretty({model:result.model}),config:pretty({...operationDefaults('sample_model').config,sampling:result.sampling})})}>采用完整计划为采样草稿</button></div>}
      {result?.stages && <div><h3>流程前置条件：{result.status === 'ready' ? '通过' : '待修正'}</h3><small>{result.validation_scope}</small>{result.stages.map(stage => <p key={stage.id}>{stage.id} → {stage.output_kind} · {stage.status}</p>)}{result.errors?.map((item,index) => <p className="control-error" key={index}>{item.id}: {item.message}</p>)}</div>}
      {result?.workflow_run && <div><h3>流程运行</h3><code>{result.workflow_run}</code><pre className="control-logs">{pretty(result.steps)}</pre></div>}
      {handoffPlan?.handoff && <div className="control-handoff"><h3>{handoffPlan.handoff.operation?'已校验的阶段接力':'已校验的产物与后续选择'}</h3>{handoffPlan.handoff.operation?<p>{handoffPlan.handoff.agent} → {handoffPlan.handoff.operation}</p>:<p>当前没有自动接续的阶段，可在此结束。后续任务需明确选择。</p>}<pre className="control-logs">{pretty(handoffPlan.handoff.inputs)}</pre>{handoffPlan.handoff.missing_inputs?.length > 0 && <p>待补充输入：{handoffPlan.handoff.missing_inputs.join('、')}</p>}{handoffPlan.handoff.missing_config?.length > 0 && <p>待设置工况：{handoffPlan.handoff.missing_config.join('、')}</p>}{handoffPlan.handoff.operation&&<><small>接力依赖选中产物及其确切版本。采用后请检查工况配置，再执行阶段。</small><button disabled={busy} onClick={() => adoptHandoff(handoffPlan)}>采用配置并切换</button></>}</div>}
      <details><summary>强制中断后的未发布检查点</summary><p>输入日志中记录的 .staging 阶段目录名。仅在操作系统确认原进程已退出、检查点及依赖校验通过后归档。</p><label>未发布阶段 ID<input value={draft.interruptedStage ?? ''} placeholder="corrected_model-12位十六进制ID" onChange={event => patch({ interruptedStage: event.target.value })}/></label><button disabled={busy || !draft.interruptedStage?.trim()} onClick={() => action(() => run({ agent: role.id, operation: 'recover_interrupted_stage', workspace: state.workspace.trim(), inputs: { stage_id: draft.interruptedStage.trim() } }))}>校验并归档检查点</button></details>
      {result?.status === 'archived' && result.checkpoint && <div><h3>检查点已归档</h3><p>已训练 {result.epoch} 轮。归档未启动训练。</p><code>{result.checkpoint}</code><small>SHA-256 {result.checkpoint_sha256}</small><button onClick={() => action(async () => { const operation = {residual_training:'train_residual',expert_imitation:'train_imitation',physical_policy_optimization:'optimize_policy',supervised_training:result.checkpoint_kind==='supervised_model'?'train_supervised':'train_external_imitation',conditional_ode_pinn:'train_pinn',external_physical_policy_optimization:'optimize_external_policy'}[result.algorithm]; if (draft.operation !== operation) throw new Error(`请先选择 ${operation} 并恢复原训练输入与配置，再采用此检查点`); patch({ config: pretty({ ...json(draft.config), resume_checkpoint: result.checkpoint, resume_checkpoint_sha256: result.checkpoint_sha256 }) }); })}>填入当前训练配置</button></div>}
      {result?.artifact_id && <div><h3>阶段产物</h3><code>{result.artifact_id}</code><p>{result.artifact_path}</p><p>状态：{result.status}</p>{result.status === 'verified' && handoffs[result.kind] && <button onClick={() => handoff(result.artifact_id, result.kind)}>交给下一阶段</button>}<button onClick={() => action(() => inspectArtifact(result.artifact_id))}>校验并查看报告</button></div>}
      {selectedReport&&<section aria-label="所选产物报告"><h3>所选报告 · {selectedReport.kind} · {selectedReport.status}</h3><code>{selectedReport.artifact_id}</code><small>{selectedReport.artifact_path}</small>
      <button disabled={busy} onClick={()=>action(()=>downloadBundle(selectedReport.artifact_id))}>下载已校验导出 ZIP</button>{downloaded&&<small>{downloaded.filename} · {downloaded.bytes} 字节 · SHA-256 {downloaded.sha256}</small>}
      {selectedReport.report&&<><ArtifactReport key={selectedReport.artifact_id} report={selectedReport.report} modelInterface={selectedReport.metadata?.interface} artifact={selectedReport} onReadFile={(name,config)=>readFile(selectedReport.artifact_id,name,config)} onDownloadBundle={()=>downloadBundle(selectedReport.artifact_id)}/><details><summary>有界报告预览 JSON（完整 report.json 见文件下载）</summary><pre className="control-logs">{pretty(selectedReport.report)}</pre></details></>}
      <details><summary>已校验文件与接口</summary>{selectedReport.files?.map(file=><div key={file.name} className="control-artifact"><code>{file.name}</code><small>{file.path}</small><button disabled={busy} onClick={()=>action(()=>readFile(selectedReport.artifact_id,file.name))}>预览此文件</button><button disabled={busy||!selectedReport.metadata?.files?.[file.name]} onClick={()=>action(()=>downloadFile(selectedReport,file))}>下载完整文件 · {file.name}</button></div>)}<pre className="control-logs">{pretty(selectedReport.metadata?.interface)}</pre></details></section>}
      {filePreview&&<details open><summary>文件预览 · {filePreview.name}</summary><small>{filePreview.bytes} 字节 · SHA-256 {filePreview.sha256}</small><pre className="control-logs">{filePreview.text??pretty(filePreview)}</pre>{filePreview.truncated&&<small>有界预览；完整文件保留在原产物中。</small>}</details>}
      {result?.file_preview && <div><h3>文件预览 · {result.file_preview.name}</h3><small>{result.file_preview.bytes} 字节 · SHA-256 {result.file_preview.sha256}</small>{result.file_preview.text && <pre className="control-logs">{result.file_preview.text}</pre>}{result.file_preview.arrays && <pre className="control-logs">{pretty(result.file_preview.arrays)}</pre>}{result.file_preview.format === 'binary' && <p>二进制文件保存在产物目录中，可由相应模型加载器读取。</p>}{result.file_preview.truncated && <small>当前为有界预览，完整文件保存在产物目录。</small>}</div>}
      {result?.artifacts && <div><h3>工作区产物</h3>{Object.entries(result.artifacts).map(([id,item]) => typeof item === 'object' ? <div className="control-artifact" key={id}><strong>{item.kind} · {item.status}</strong><code>{id}</code><small>{item.path}</small>{handoffs[item.kind] && item.status === 'verified' && <button onClick={() => handoff(id,item.kind)}>选择并接力</button>}<button onClick={() => action(() => inspectArtifact(id))}>查看报告</button></div> : <p key={id}>{id}: {item}</p>)}</div>}
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
