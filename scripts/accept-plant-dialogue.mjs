/** Live, billable Harness dialogue. Use an isolated Web Host and a new test workspace. */
import assert from 'node:assert/strict';
import {createHash, randomUUID} from 'node:crypto';
import {readFile, realpath, writeFile} from 'node:fs/promises';
import {basename, dirname, join, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {connectHarness, project} from './harness-rpc.mjs';

const {values} = parseArgs({options:{action:{type:'string'}, fixture:{type:'string'}, workspace:{type:'string'},
  source:{type:'string'}, text:{type:'string'},presentation:{type:'boolean',default:false}}});
assert.ok(['start','switch-mode','resume','inspect','wait','verify'].includes(values.action), 'Select start, switch-mode, resume, inspect, wait or verify.');
const fixture = await realpath(resolve(values.fixture));
const workspace = await realpath(resolve(values.workspace));
const testRoot = await realpath(join(project,'test-output'));
assert.equal(dirname(fixture),testRoot);
assert.equal(dirname(workspace),testRoot);
assert.notEqual(fixture,workspace);
const path = join(fixture,'paid-plant-dialogue.json');
assert.ok(!values.presentation || ['inspect','verify'].includes(values.action),'Presentation Host supports observation only.');
const rpc = await connectHarness({logPath:join(fixture,values.presentation?'private-presentation.log':'private-web-startup.log'),connectionClose:true});
let record;
if (values.action === 'start') {
  const source = await readFile(resolve(values.source));
  const digest = createHash('sha256').update(source).digest('hex');
  assert.equal(digest,'c4f054989498454dd25c2f4f78668fcba29bad2231e51fba57d7ae737dbea7b9','Only upload the original reviewed benchmark');
  const catalog = await rpc('session/modelCatalog',{});
  const group = catalog.groups.find(item => item.id === 'deepseek-official');
  const model = group?.models.find(item => item.id === catalog.default.model) ?? group?.models[0];
  assert.ok(model,'Configure the DeepSeek provider in the isolated Harness Host.');
  record = {workspace,provider:group.id,model:model.id,source_sha256:digest,turns:[]};
  await writeFile(path,JSON.stringify(record,null,2),{flag:'wx'});
  const created = await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});
  record.sessionId = created.sessionId;
  await save();
  await rpc('session/selectModel',{request:{sessionId:record.sessionId,provider:group.id,model:model.id,reasoningEffort:'low'}});
  const upload = await rpc('fileUploads/upload',{agentId:record.sessionId,request:{data:source.toString('base64'),name:basename(values.source)}});
  assert.ok(upload.receiptId);
  record.upload = {name:basename(values.source),bytes:source.length};
  await submit(values.text ?? '请阅读上传的 Python 源文件，把这个对象转换成可仿真的 Simulink 被控对象，在本次工作区新建 SLX。质量 1 kg，阻尼 0.8，刚度 2，采样周期 0.1 秒，默认初态 [0.15,-0.1]。保留控制力、扰动力和完整状态。关闭后从磁盘重载，用原始 step 比较三组不同初态，每组至少 20 步独立变化的控制与扰动，绝对容差 1e-12。请给出文件路径和实测最大误差。',upload.receiptId);
} else {
  record = JSON.parse(await readFile(path,'utf8'));
  assert.equal(record.workspace,workspace);
  assert.ok(record.sessionId);
  if (values.action === 'switch-mode') {
    assert.ok(!record.permission_switch && record.observed?.turnsEnded===1 &&
      record.observed.results.some(r=>r.diagnostic==='check_WRITE_OWNER'),'Switch only a new acceptance chat after its first denied turn.');
    const command=await rpc('commands/execute',{agentId:record.sessionId,line:'/permission danger-full-access',submittedAttachments:[]});
    assert.equal(command.result.kind,'success');
    record.permission_switch={requested_at:new Date().toISOString(),result:command.result.text};
    await save();
  }
  if (values.action === 'resume') await submit(values.text ?? '我已经按网页指引准备好了同一工作区。请重新检查权限，在这个对话继续完成上一条请求，用我上传的原文件建模和验证。');
}
async function save() { await writeFile(path,JSON.stringify(record,null,2)); }
async function submit(text,receiptId) {
  const requestId = randomUUID();
  record.turns.push({requestId,text,submitted:new Date().toISOString()});
  await save();
  await rpc('session/prompt',{request:{sessionId:record.sessionId,requestId,mode:'queue',
    content:[{type:'text',text},...(receiptId ? [{type:'file',receiptId}] : [])]}});
  console.log(JSON.stringify({sessionId:record.sessionId,model:record.model,turn:record.turns.length,admitted:true}));
}
const visible = content => typeof content === 'string' ? content : (content ?? []).filter(x=>x.type==='text').map(x=>x.text).join('\n');
const deadline = Date.now() + 12*60_000;
let previous = -1;
for (;;) {
  const page = await rpc('session/events',{request:{address:{kind:'session',sessionId:record.sessionId},maxMessages:300}});
  const events = page.records.map(r=>r.event ?? r);
  const end = events.filter(r=>r.type==='turn/end');
  const calls = events.filter(r=>r.type==='tool/call').map(r=>{
    const args=typeof r.data.arguments==='string'?JSON.parse(r.data.arguments):r.data.arguments;
    const cmd=args?.command ?? '';
    return {seq:r.seq,turn:r.data.turn,name:r.data.name,callId:r.data.callId,
      ...(r.data.name==='read'?{file:args.file_path}:{}),
      ...(r.data.name==='working_directory'?{changes_directory:args?.cd!==undefined}:{}),
      ...(cmd.includes('plant-step-reference.py')?{source_replay:true}:{}),
      ...(r.data.name==='control_build_zoh_plant'?{requested_model:args.model,reference_file:args.referenceFile}:{})};
  });
  const results = events.filter(r=>r.type==='tool/result').map(r=>{
    const m=r.data.message, text=visible(m.content);
    let v;
    try { v=JSON.parse(text.replace(/^Error:\s*/,'')); } catch { /* ordinary tool result */ }
    return {seq:r.seq,turn:r.data.turn,error:m.isError,callId:m.toolCallId,
      ...(r.data.error?.code?{host_code:r.data.error.code}:{}),
      ...(v?.status==='replayed'?{status:'replayed',source:v.source,source_sha256:v.source_sha256,reference_file:v.referenceFile,samples:v.samples,sandbox_mode:v.sandbox_mode}:{}),
      ...(v?.stage?{stage:v.stage,status:v.status,diagnostic:v.diagnostic?.operation,retryable:v.retryable}:{}),
      ...(v?.stage==='workspace_preflight'?{observation:v.preflight_observation,checked_at:v.checked_at??v.diagnostic?.checked_at,
        session_mode:v.session_sandbox_mode??v.diagnostic?.session_sandbox_mode,sandbox_mode:v.sandbox_mode??v.diagnostic?.sandbox_mode,
        work_started:v.work_started}:{}),
      ...(v?.status==='simulated'?{path:v.path,sha256:v.sha256,comparison:v.comparison,verified:v.verified,
        matlab_pid:v.matlab_pid,session_verified:v.session_verified}:{}),
      ...(m.isError&&!v?.stage?{message:text.slice(0,400)}:{})};
  });
  record.observed = {turnsEnded:end.length,ends:end.map(r=>r.data.reason),calls,results,
    modes:events.filter(r=>r.type==='sandbox/mode').map(r=>({seq:r.seq,mode:r.data.mode})),
    answer:visible(events.filter(r=>r.type==='assistant/message').at(-1)?.data.message.content)};
  await save();
  if (previous !== calls.length) {
    console.log(JSON.stringify({turnsEnded:end.length,tools:calls.slice(-4).map(c=>c.name),lastResult:results.at(-1)}));
    previous=calls.length;
  }
  if (['inspect','verify'].includes(values.action) || end.length>=record.turns.length) break;
  if (Date.now()>=deadline) throw Error('Wait expired; use --action wait to continue observing the same turn.');
  await new Promise(done=>setTimeout(done,2500));
}
if (values.action==='verify') {
  delete record.validated;
  await save();
  assert.equal(record.observed.turnsEnded,record.turns.length,'Wait for the entire agent turn before accepting its delivery.');
  assert.equal(record.observed.ends.at(-1)?.kind,'completed');
  assert.ok(record.observed.results.some(r=>r.diagnostic==='check_WRITE_OWNER' || r.message?.includes('WRITE_OWNER')),
    'The same live chat must first observe the workspace failure.');
  const completed=record.observed.results.findLast(r=>r.status==='simulated' && !r.error && r.comparison?.compared);
  assert.ok(completed?.session_verified && completed.verified?.includes('reload') && completed.verified.includes('compare'));
  // A saved model can be numerically correct even if a later call quarantines MATLAB.
  const readOnlyTools=new Set(['read','read_image','glob','grep','skill','ask_user_question','control_help']);
  for (const call of record.observed.calls.filter(c=>c.turn===completed.turn && c.callId!==completed.callId)) {
    const result=record.observed.results.find(r=>r.callId===call.callId);
    if (call.seq<completed.seq && result?.seq<completed.seq) continue;
    if (readOnlyTools.has(call.name) || call.name==='working_directory' && !call.changes_directory) continue;
    assert.ok(result?.error && result.host_code==='PLANT_DELIVERY_COMPLETE',
      'Side effect after verified delivery was not stopped by the Host: '+call.name+' at #'+call.seq);
  }
  assert.equal(dirname(await realpath(completed.path)),workspace);
  assert.equal(createHash('sha256').update(await readFile(completed.path)).digest('hex'),completed.sha256);
  assert.ok(completed.comparison.trajectories>=3 && completed.comparison.samples>=63);
  assert.ok(completed.comparison.max_abs_error<=1e-12);
  const builder=record.observed.calls.find(c=>c.name==='control_build_zoh_plant' && c.callId===completed.callId);
  assert.ok(builder?.reference_file,'The model must supply its actual original-step reference file.');
  const read=record.observed.calls.find(c=>c.name==='read' && c.file?.includes(record.source_sha256));
  assert.ok(read && record.observed.results.some(r=>r.callId===read.callId && !r.error),'Missing successful source read.');
  const replay=record.observed.results.find(r=>r.status==='replayed' && !r.error && r.reference_file===builder.reference_file);
  assert.ok(replay?.source_sha256===record.source_sha256 && replay.samples===completed.comparison.samples);
  if(record.permission_switch) {
    const changed=record.observed.modes.find(r=>r.mode==='danger-full-access');
    assert.ok(changed && replay.sandbox_mode==='workspace-write');
    assert.ok(record.observed.results.some(r=>r.seq>changed.seq && r.seq<replay.seq &&
      r.status==='verified' && r.observation==='fresh' && r.session_mode==='danger-full-access' && r.sandbox_mode==='workspace-write'));
  }
  const referencePath=await realpath(join(workspace,builder.reference_file));
  assert.equal(dirname(referencePath),workspace);
  const reference=JSON.parse(await readFile(referencePath,'utf8'));
  assert.equal(reference.source_sha256,record.source_sha256);
  record.validated={path:completed.path,sha256:completed.sha256,source_sha256:reference.source_sha256,
    source_read_seq:read.seq,replay_seq:replay.seq,model_receipt_seq:completed.seq,
    matlab_pid:completed.matlab_pid,comparison:completed.comparison};
  await save();
}
console.log(JSON.stringify({sessionId:record.sessionId,observedTurns:record.observed.turnsEnded,
  receipt:record.observed.results.findLast(r=>r.status==='simulated'),evidence:path}));
