/** Actual DeepSeek conversation plans and runs all eight stages through native tools. */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { connectHarness, command, project } from './harness-rpc.mjs';

const rpc=await connectHarness();
const catalog=await rpc('session/modelCatalog',{});
const provider='deepseek-official';
const offered=catalog.groups.find(group=>group.id===provider)?.models;
assert.ok(offered?.length,'Configure the official DeepSeek provider first');
const model=offered.find(item=>item.id===catalog.default.model)?.id??offered[0].id;
const workspace=resolve(project,'test-output',`harness-business-dialogue-${randomUUID()}`);
await mkdir(workspace,{recursive:true});
if(process.platform==='win32')execFileSync('pwsh.exe',['-NoProfile','-File',resolve(project,'scripts/prepare-workspace.ps1'),'-Prepare', '-Workspace',workspace]);
const workflowPath=resolve(project,'examples/python-tensor/workflow.json');
const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});
const sessionId=created.sessionId;
await rpc('session/selectModel',{request:{sessionId,provider,model,reasoningEffort:'low'}});
const evidence={checkedAt:new Date().toISOString(),workspace,workflowPath,...created,provider,model,status:'running',scope:'Actual DeepSeek conversation must plan and run the reviewed standalone Python benchmark through control_workflow and wait for real native jobs. No mocked model or numerical results.'};
const evidencePath=resolve(project,'docs/harness-business-dialogue-evidence.json');
await writeFile(evidencePath,JSON.stringify(evidence,null,2));
await rpc('session/prompt',{request:{sessionId,requestId:randomUUID(),mode:'queue',content:[{type:'text',text:`请执行已授权的完整控制工程工作流。工作目录：${workspace}。流程文件：${workflowPath}。这是明确声明的数学参考仿真案例；其中独立PyTorch源码已检查，允许在原生数值作业中执行。请先用 control_help 查询 plan_workflow 和 run_workflow，使用 control_workflow 预检并读取作业最终结果；预检通过后按原文件配置运行八阶段流程，再使用原生 job_output 等待运行作业终态。工具只返回jobId并不代表完成，异步作业仍运行时不要结束任务。不得改动源文件、参数、工况或验收门，不要调用demo或重复提交相同流程。完成后读取真实产物和报告，给出每阶段ID、数据来源、约束结果，以及神经策略是否实际改善。遇到失败如实报告，不得伪装成功。`} ]}});
console.log(JSON.stringify({sessionId,workspace,workflowPath,status:'model_request_admitted'}));
const deadline=Date.now()+600000;
let state;
while(Date.now()<deadline){
  state=await rpc('session/wait',{request:{sessionId,timeoutMs:1000,maxMessages:10}});
  if(state.reason!=='timeout')break;
}
const history=await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace})}`);
const planned=history.attempts.find(row=>row.request.operation==='plan_workflow'&&row.state==='completed'&&row.result?.status==='ready');
const runs=history.attempts.filter(row=>row.request.operation==='run_workflow');
const completed=runs.find(row=>row.state==='completed'&&row.result?.status==='verified');
const manifest=JSON.parse(await readFile(resolve(workspace,'manifest.json'),'utf8'));
const page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:60}});
const providerError=page.records.findLast(row=>row.event.type==='turn/end')?.event.data.reason?.error;
Object.assign(evidence,{waitReason:state?.reason,preflightPassed:!!planned,workflowAttempts:runs.length,status:completed?'verified':'not_verified',
  ...(completed?{jobId:completed.nativeJobId,workflowRun:completed.result.workflow_run,artifacts:completed.result.artifacts}:{}),
  artifactCount:Object.keys(manifest.artifacts).length,eventTypes:[...new Set(page.records.map(row=>row.event.type))],
  ...(providerError?{providerError:{code:providerError.code,status:providerError.status}}:{}),
  attempts:history.attempts.map(row=>({attemptId:row.attemptId,operation:row.request.operation,nativeJobId:row.nativeJobId,state:row.state}))});
if(completed){
  evidence.reports={};
  for(const [stage,id]of Object.entries(completed.result.artifacts)){
    const row=manifest.artifacts[id]; assert.equal(row.status,'verified');
    if(['corrected','controller','imitation','optimized'].includes(stage))evidence.reports[stage]=JSON.parse(await readFile(resolve(workspace,row.path,'report.json'),'utf8'));
  }
}
await writeFile(evidencePath,JSON.stringify(evidence,null,2));
console.log(JSON.stringify({sessionId,status:evidence.status,preflightPassed:evidence.preflightPassed,workflowAttempts:runs.length,artifactCount:evidence.artifactCount,...(completed?{jobId:completed.nativeJobId}: {})}));
assert.ok(planned,'The real model did not complete a successful workflow preflight');
assert.equal(runs.length,1,'Exactly one workflow attempt is expected; inspect existing live jobs before retrying');
assert.ok(completed,'The actual dialogue did not finish its numerical job; inspect this Session/job rather than starting another');
assert.equal(evidence.artifactCount,8);
