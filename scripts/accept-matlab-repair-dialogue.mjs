/** Billable acceptance against an isolated Web Host and a dedicated synthetic MATLAB fixture. */
import assert from 'node:assert/strict';
import {createHash, randomUUID} from 'node:crypto';
import {readFile, realpath, stat, writeFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import {setTimeout as sleep} from 'node:timers/promises';
import {connectHarness} from './harness-rpc.mjs';
import {configureFixtureFileGeneration,openLayer,readFixture} from './matlab-layer-harness.mjs';

const [directory, scenario] = process.argv.slice(2);
assert.ok(directory && ['controller','controller-eval','partial'].includes(scenario), 'Pass the disposable fixture and controller, controller-eval or partial.');
const {fixture,marker,resource} = await readFixture(resolve(directory));
const workspace = await realpath(join(fixture,scenario === 'partial' ? 'partial' : 'workspace'));
const reportPath = join(fixture,scenario+'-dialogue.json');
const model = scenario === 'partial' ? 'SaveRetryBench' : scenario === 'controller-eval' ? 'MpcEvalRetryBench' : 'MpcRetryBench';
const target = join(workspace,model+'.slx');
const basePrompt = scenario === 'partial'
  ? '请执行一个自包含的 Simulink 测试。创建 SaveRetryBench，让常数 2.5 接顶层输出 y。使用独立的保存工具落盘，保留返回的绝对路径。然后故意用 control_simulate_simulink_model 对这个已保存模型提交 solver=NotARealSolver；必须实际观察错误。在同一回合从报错恢复，只纠正仿真输入，用正确设置仿真 1 秒。从仿真结果读出实际输出值，打印 CHECK_Y=数值，不重复建模、保存或覆盖文件。完成后让此模型保持已加载，在其模型工作区设置未保存的变量 KeepOtherChatValue=1701，供另一聊天之后核验。只处理本测试模型，仅使用 MATLAB 和项目建模工具。'
  : '请执行 MPC 闭环建模验收。对象是合成离散标量 x(k+1)=0.8*x(k)+0.2*u(k)，y=x，周期 0.1 秒，初态零，参考恒定为 1。先为 MpcRetryBench 创建部分内存模型和 mpc 对象，再实际尝试添加旧工程标注的 mpc_lib/MPC Controller；暂时不要改写该路径或用 try/catch 吸收异常。我需要看到失败的原生工具回执。收到实际错误后在同一轮自主查询本机安装中的真实库块，读回部分模型、保留已有块和对象，从出错的操作继续。完成 MPC、被控对象、参考与测量反馈的接线并把 y 接顶层 Outport。仅编辑 MpcRetryBench，其他会话已有模型保持原状。用独立工具保存本工作区的 SLX，关闭自己的模型，从回执的绝对路径重载，闭环仿真 4 秒；从实际仿真时序打印 CHECK_Y=最后一个输出值，并检查前后符合跟踪目标。结尾用 MATLAB 读回另一个聊天的 SaveRetryBench 已加载且模型工作区 KeepOtherChatValue 仍为 1701，打印 OTHER_CHAT=1701 后报告结果。仅使用 MATLAB 和项目建模工具；不要读取任何用户文件或配置。';
const prompt = basePrompt.replaceAll('MpcRetryBench',model) + (scenario==='controller-eval' ? ' 本轮须精确复现原生 evaluate_matlab_code 通道：错误的第一条加控制器块操作请用 mcp__matlab__evaluate_matlab_code 直接调用 add_block 用旧库路径和控制器目标块，不用 model_edit 来触发此错误。后续查询、修改与仿真步骤请根据回执自主决定。' : '');
const rpc = await connectHarness({logPath:join(fixture,'private-web-startup.log'),connectionClose:true});
const save = async () => writeFile(reportPath, JSON.stringify(report,null,2)+'\n');
let report;
try { report = JSON.parse(await readFile(reportPath,'utf8')); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (!report) {
  const setup=await openLayer({fixture,workspace,managed:true});
  try {await configureFixtureFileGeneration(setup,fixture);}
  finally {await setup.close();}
  const catalog = await rpc('session/modelCatalog',{});
  const group = catalog.groups.find(entry => entry.id === 'deepseek-official');
  const route = group?.models.find(entry => entry.id === catalog.default.model) ?? group?.models[0];
  assert.ok(route,'Official DeepSeek route required');
  const {sessionId} = await rpc('session/create',{request:{cwd:workspace,agentPreset:scenario === 'partial'?'control-model':'control-controller'}});
  report = {scenario,sessionId,workspace,model,provider:group.id,route:route.id,prompt,requestId:randomUUID(),
    matlab_pid:marker.matlab_pid,matlab_session:marker.session_id,submitted:false};
  await writeFile(reportPath,JSON.stringify(report,null,2),{flag:'wx'});
  await rpc('session/selectModel',{request:{sessionId,provider:group.id,model:route.id,reasoningEffort:'low'}});
  // Persist the request id before sending so resuming never pays for another turn.
  report.submitted = true; await save();
  await rpc('session/prompt',{request:{sessionId,requestId:report.requestId,mode:'queue',content:[{type:'text',text:prompt}]}});
}
assert.equal(report.workspace,workspace);
assert.equal(report.matlab_session,marker.session_id);
assert.ok(report.submitted,'Fixture stopped before submission; inspect its recorded session.');
let previousCalls = -1;
const deadline = Date.now()+18*60_000;
for (;;) {
  const page = await rpc('session/events',{request:{address:{kind:'session',sessionId:report.sessionId},maxMessages:600}});
  const events = page.records.map(row=>row.event ?? row);
  const calls = events.filter(e=>e.type==='tool/call').map(e=>{
    const data=e.data;
    const args=typeof data.arguments==='string' ? JSON.parse(data.arguments) : data.arguments;
    const r=events.find(v=>v.type==='tool/result' && v.data.message.toolCallId===data.callId);
    const content=r?.data.message.content.filter(c=>c.type==='text').map(c=>c.text).join('\n') ?? '';
    return {seq:e.seq,turn:data.turn,name:data.name,id:data.callId,args,result_seq:r?.seq,error:r?.data.message.isError,content};
  });
  const end=events.find(e=>e.type==='turn/end');
  report.calls=calls; report.end=end?.data.reason;
  const state=JSON.parse(await readFile(join(resource,'resource.json'),'utf8'));
  assert.equal(state.session?.pid,marker.matlab_pid);
  assert.equal(state.session?.session_id,marker.session_id);
  report.quarantine=state.quarantine?.reason ?? null;
  const firstSave=calls.find(c=>c.name==='control_save_simulink_model' && c.error===false);
  if (firstSave && !report.saved) {
    const receipt=JSON.parse(firstSave.content);
    assert.equal(receipt.path,target);
    report.saved={sha256:receipt.sha256,bytes:receipt.bytes,mtime:(await stat(target)).mtimeMs,seq:firstSave.result_seq};
  }
  await save();
  if(calls.length!==previousCalls) console.log(JSON.stringify({scenario,calls:calls.length,last:calls.at(-1)?.name,ended:!!end}));
  previousCalls=calls.length;
  if(end) break;
  assert.ok(Date.now()<deadline,'Observation timeout; rerun to observe the same session');
  await sleep(1200);
}
assert.equal(report.end?.kind,'completed',JSON.stringify(report.end));
assert.equal(report.quarantine,null);
assert.ok(report.saved,'No verified save receipt');
const saves=report.calls.filter(c=>c.name==='control_save_simulink_model' && c.error===false);
const delivered=JSON.parse(saves.at(-1).content);
assert.equal(delivered.path,target);
assert.equal(createHash('sha256').update(await readFile(target)).digest('hex'),delivered.sha256);
const fault=report.calls.find(c=>(c.error || (c.name==='mcp__matlab__model_edit' && c.content.includes('status: error'))) && (scenario==='partial'
  ? c.name==='control_simulate_simulink_model' && c.args.solver==='NotARealSolver'
  : (c.name==='mcp__matlab__evaluate_matlab_code' || scenario==='controller' && c.name==='mcp__matlab__model_edit') && JSON.stringify(c.args).includes('mpc_lib/MPC Controller') && /mpc_lib/.test(c.content)));
assert.ok(fault,'Expected original diagnostic from an actual tool execution');
const later=report.calls.filter(c=>c.seq>fault.result_seq && c.turn===fault.turn && c.error===false);
const output=later.flatMap(c=>[...c.content.matchAll(/(?:^|\n)\s*CHECK_Y\s*=\s*([-+\d.eE]+)/g)].map(m=>Number(m[1]))).at(-1);
assert.ok(Number.isFinite(output),'No observed numeric sample after error');
if (scenario==='partial') {
  assert.equal(saves.length,1);
  assert.equal((await stat(target)).mtimeMs,report.saved.mtime,'SLX was rewritten after first verified save');
  assert.ok(report.saved.seq<fault.seq && later.some(c=>c.name==='control_simulate_simulink_model'));
  assert.ok(Math.abs(output-2.5)<1e-9);
} else {
  assert.ok(later.some(c=>/model_read|model_overview/.test(c.name)||/find_system|get_param/.test(c.args.code??'')),'Missing model readback');
  assert.ok(later.some(c=>/mpclib/i.test(c.content) && /find_system|which|dir|load_system|model_read/.test(c.args.code??c.name)),'No installed-library discovery result');
  assert.ok(later.some(c=>/load_system|open_system/.test(c.args.code??'') && (c.args.code.includes(target) || c.args.code.includes(target.replaceAll('\\','/')))),'No observed reload from delivered path');
  assert.ok(later.some(c=>/(?:^|\n)\s*OTHER_CHAT\s*=\s*1701\b/.test(c.content)),'Other chat model state lost');
  assert.ok(Math.abs(1-output)<0.1,'Closed loop did not track the synthetic reference');
}
report.verified={target,output,turn:fault.turn,fault_seq:fault.result_seq,save_seq:saves.at(-1).result_seq,sha256:delivered.sha256,final_pid:marker.matlab_pid,final_session:marker.session_id};
await save();
console.log(JSON.stringify(report.verified));
