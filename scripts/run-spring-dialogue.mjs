/** Run the spring benchmark through three real Harness role Sessions; save public turn evidence only. */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { connectHarness, command, project } from './harness-rpc.mjs';

const workspace = resolve(process.argv[2] ?? resolve(project, 'workspaces', `spring-damper-${new Date().toISOString().slice(0,10)}-${randomUUID().slice(0,8)}`));
const caseRoot = resolve(project, 'examples/spring-damper');
const evidenceRoot = resolve(workspace, 'presentation/agent-outputs');
await mkdir(evidenceRoot, {recursive:true});
execFileSync('pwsh.exe', ['-NoProfile','-File',resolve(project,'scripts/prepare-workspace.ps1'),'-Workspace',workspace]);
const rpc = await connectHarness();
const provider = 'deepseek-official', model = 'deepseek-flash';
const recordPath = resolve(workspace, 'case-run.json');
let record;
try { record = JSON.parse(await readFile(recordPath,'utf8')); }
catch (error) { if(error.code !== 'ENOENT') throw error; record = {case:'spring-damper',workspace,createdAt:new Date().toISOString(),provider,model,status:'running',artifacts:{},phases:[]}; }
const persist = () => writeFile(recordPath, JSON.stringify(record,null,2));
const settings = async name => JSON.parse(await readFile(resolve(caseRoot,`${name}.json`),'utf8'));

function visibleText(records) {
  const texts=[];
  function visit(value) {
    if(!value || typeof value !== 'object') return;
    if(value.type === 'block-end' && value.block?.type === 'text' && typeof value.block.text === 'string') { texts.push(value.block.text); return; }
    if(value.type === 'text' && typeof value.text === 'string') { texts.push(value.text); return; }
    for(const [key,item] of Object.entries(value)) if(!['reasoning','thinking'].includes(key)) {
      if(Array.isArray(item)) item.forEach(visit); else if(typeof item === 'object') visit(item);
    }
  }
  for(const row of records) if(row.event.type === 'assistant/message') visit(row.event.data);
  return [...new Set(texts)].join('\n\n');
}

async function phase(name, preset, definition, explanation) {
  let entry=record.phases.find(row => row.name===name);
  if(entry?.status === 'verified') return entry;
  const workflowPath=resolve(workspace,`${name}-workflow.json`);
  if(!entry) {
    await writeFile(workflowPath,JSON.stringify(definition,null,2));
    const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:preset}});
    const sessionId=created.sessionId;
    await rpc('session/selectModel',{request:{sessionId,provider,model,reasoningEffort:'low'}});
    const prompt=`请执行二阶质量—弹簧—阻尼系统演示的${name}阶段。工作目录：${workspace}；本阶段明确批准的流程文件：${workflowPath}。${explanation}\n先用 control_help 查询 plan_workflow 和 run_workflow，再用 control_workflow 预检本阶段文件；通过后仅运行一次该文件。严格按文件执行，不改参数、数据、工况、文件或验收门，不调用 demo。作业异步执行时用 job_output 等待真实终态，不要把 jobId 当成完成，不要重复提交。末尾用真实产物和报告写中文交接说明：输入与输出产物ID、维度和单位、数据来源、评价结果、下一角色应读取的产物。mean_control_seconds和p95_control_seconds单位为秒，显示毫秒时必须乘1000；control_effort是归一化平方力而不是物理能量。没有声明input_rate_bounds时速率检查是不适用而非失败。单种子不能推断显著性或判定改进属于随机噪声。明说是数学参考仿真而非实测；模型校正为离散残差学习，策略优化为可微物理rollout策略梯度，不能称作完整ODE PINN、PPO或SAC。若失败请停下如实报告。`;
    entry={name,preset,sessionId,workflowPath,prompt,status:'submitted',submittedAt:new Date().toISOString()};
    record.phases.push(entry); await persist();
    await writeFile(resolve(evidenceRoot,`${name}-prompt.txt`),prompt);
    await rpc('session/prompt',{request:{sessionId,requestId:randomUUID(),mode:'queue',content:[{type:'text',text:prompt}]}});
    console.log(JSON.stringify({phase:name,sessionId,workspace,status:'submitted'}));
  } else console.log(JSON.stringify({phase:name,sessionId:entry.sessionId,status:'resuming_existing_session'}));
  const deadline=Date.now()+1800000;
  let last=null;
  while(Date.now()<deadline) {
    const state=await rpc('session/wait',{request:{sessionId:entry.sessionId,timeoutMs:1000,maxMessages:5}});
    if(state.reason !== 'timeout') { last=state; break; }
  }
  const history=await command(rpc,entry.sessionId,`/control-history ${JSON.stringify({workspace})}`);
  const runs=history.attempts.filter(row=>row.request.operation==='run_workflow' && row.request.inputs?.workflow_path===workflowPath);
  const completed=runs.find(row=>row.state==='completed' && row.result?.status==='verified');
  const events=await rpc('session/events',{request:{address:{kind:'session',sessionId:entry.sessionId},maxMessages:200}});
  const transcript=visibleText(events.records);
  await writeFile(resolve(evidenceRoot,`${name}-assistant.txt`),transcript || '未提取到文本；参见Session事件类型记录。');
  const safeEvents=events.records.filter(row=>['turn/begin','turn/end','tool/call','tool/result','assistant/message'].includes(row.event.type));
  // Store only the visible transcript and domain attempts; raw replay/credentials are never exported.
  const attempts=history.attempts.map(row=>({attemptId:row.attemptId,nativeJobId:row.nativeJobId,state:row.state,request:row.request,result:row.result}));
  await writeFile(resolve(evidenceRoot,`${name}-attempts.json`),JSON.stringify(attempts,null,2));
  Object.assign(entry,{status:completed?'verified':'not_verified',waitReason:last?.reason,eventTypes:[...new Set(safeEvents.map(row=>row.event.type))],workflowAttempts:runs.length,
    ...(completed?{jobId:completed.nativeJobId,workflowRun:completed.result.workflow_run,artifacts:completed.result.artifacts}:{})});
  if(completed) Object.assign(record.artifacts,completed.result.artifacts);
  await persist();
  console.log(JSON.stringify({phase:name,status:entry.status,workflowAttempts:runs.length,artifacts:entry.artifacts}));
  assert.equal(runs.length,1,'Inspect existing attempts before retrying a phase');
  assert.ok(completed,`Phase ${name} is not verified; inspect the existing Session and job rather than starting another`);
  return entry;
}

const modelPhase=await settings('model-phase-workflow');
// File references are resolved relative to this saved phase file in the workspace.
for(const step of modelPhase.steps) if(step.inputs?.model_path) step.inputs.model_path=resolve(caseRoot,step.inputs.model_path);
await phase('01-model','control-model',modelPhase,'连续对象为 m q双点+c q单点+k q=u+d；状态[q(m),v(m/s)]，控制u(N)，输出q(m)，扰动d(N)，dt=.2秒。名义m=1,c=.8,k=2；参考m=1.1,c=1,k=2.3。残差网络[q,v,u,d]→下一状态残差[Δq,Δv]，4入2出。');
const a=record.artifacts;
await phase('02-controller','control-controller',{schema_version:1,name:'Spring MPC design and independent genetic tuning',steps:[{
  id:'controller',operation:'design_controller',inputs:{model:a.corrected,evaluation_model:a.reference},config:await settings('controller')
}]},`必须读取上一阶段已验证校正模型 ${a.corrected} 和参考对象 ${a.reference}；固定动力学和horizon=6，仅GA调r/s，调参/复验/最终测试互不重叠。不要把调参工况的分数与另一个工况的分数相除称为改善。`);
await phase('03-lightweight','control-lightweight',{schema_version:1,name:'Spring expert imitation and physical policy improvement',steps:[
  {id:'expert',operation:'collect_expert',inputs:{controller:a.controller,evaluation_model:a.reference},config:await settings('expert')},
  {id:'imitation',operation:'train_imitation',inputs:{dataset:{stage:'expert'}},config:await settings('imitation')},
  {id:'optimized',operation:'optimize_policy',inputs:{policy:{stage:'imitation'},model:a.corrected},config:await settings('optimize')}
]},`必须接力上一阶段控制器 ${a.controller}、参考对象 ${a.reference} 和校正模型 ${a.corrected}。专家采集12个独立工况；策略输入[q,v,目标位移,d]为4维、输出u为1维；按工况划分数据，测试不参与训练或checkpoint选择。给出held-out同工况专家/模仿/优化对比，如实报告优化是否改善。`);
const manifest=JSON.parse(await readFile(resolve(workspace,'manifest.json'),'utf8'));
for(const id of Object.values(record.artifacts)) assert.equal(manifest.artifacts[id]?.status,'verified');
record.status='verified'; record.completedAt=new Date().toISOString(); await persist();
await copyFile(resolve(caseRoot,'nominal.json'),resolve(workspace,'presentation/nominal.json'));
await copyFile(resolve(caseRoot,'reference.json'),resolve(workspace,'presentation/reference.json'));
console.log(JSON.stringify({status:record.status,workspace,artifacts:record.artifacts}));
