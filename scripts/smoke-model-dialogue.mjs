/** Actual DeepSeek requests and native tool jobs. Reads only explicit simulation fixtures. */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { connectHarness, command, prepareSmokeWorkspace, smokeOptions, writeSmokeEvidence } from './harness-rpc.mjs';

const options = smokeOptions('smoke-model-dialogue');
if (options.help) process.exit(0);
const workspace = options.workspace;
const workspacePreparation = await prepareSmokeWorkspace(workspace);
const rpc = await connectHarness({ logPath: options.webLog });
const catalog = await rpc('session/modelCatalog', {});
const group = catalog.groups.find(item=>item.id==='deepseek-official');
assert.ok(group?.models.length,'Configure the official DeepSeek provider in Harness first');
const selected = group.models.find(model=>model.id===catalog.default.model) ?? group.models[0];
const evidence={checkedAt:new Date().toISOString(),provider:group.id,model:selected.id,workspace,workspacePreparation,scope:'Actual provider requests; each fresh role Session must itself submit inspect_workspace and observe its native job. No numerical success is simulated.',roles:[]};
for (const role of ['model','controller','lightweight']) {
  const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:`control-${role}`}});
  const sessionId=created.sessionId;
  await rpc('session/selectModel',{request:{sessionId,provider:group.id,model:selected.id,reasoningEffort:'low'}});
  await rpc('session/prompt',{request:{sessionId,requestId:randomUUID(),mode:'queue',content:[{type:'text',text:`这是已授权的本机联调。工作目录：${workspace}，允许为空目录，产物数量可以为零。请先调用 control_help 查询 inspect_workspace，然后通过 control_${role} 执行 inspect_workspace，使用 job_output 读取到作业结束，最后报告实际产物数量。只允许该检查自身必要的工作区初始化及作业记录；不得训练、改动对象材料、运行其他阶段或以猜测代替工具结果。`} ]}});
  console.log(`Actual model request admitted for ${role}: ${sessionId}`);
  const deadline=Date.now()+180000;
  let state;
  while (Date.now()<deadline) {
    state=await rpc('session/wait',{request:{sessionId,timeoutMs:1000,maxMessages:10}});
    if (state.reason!=='timeout') break;
  }
  const history=await command(rpc,sessionId,`/control-history ${JSON.stringify({workspace})}`);
  const inspections=history.attempts.filter(record=>record.request.operation==='inspect_workspace');
  const completed=inspections.find(record=>record.state==='completed'&&record.result?.artifacts);
  const page=await rpc('session/events',{request:{address:{kind:'session',sessionId},maxMessages:40}});
  const providerError=page.records.findLast(row=>row.event.type==='turn/end')?.event.data.reason?.error;
  const status=providerError?.status;
  const failure=status===401?'provider_authentication_401':status===402?'provider_balance_402':status===429?'provider_rate_limit_429':undefined;
  const row={role,...created,waitReason:state?.reason,nativeInspectionAttempts:inspections.length,passed:!!completed,
    ...(completed?{jobId:completed.nativeJobId,artifactCount:Object.keys(completed.result.artifacts).length}:{}),
    ...(failure?{failure}:{}),...(providerError?{providerError:{code:providerError.code,status}}:{}),eventTypes:[...new Set((page.records??[]).map(item=>(item.event??item).type))]};
  evidence.roles.push(row);
  await writeSmokeEvidence(options.evidence,evidence);
  console.log(JSON.stringify(row));
  if (!completed) throw new Error(`Actual ${role} dialogue did not complete its native inspection${failure?`: ${failure}`:''}; inspect the recorded Session. No model success claimed.`);
}
evidence.status = 'PASS';
await writeSmokeEvidence(options.evidence, evidence);
console.log(JSON.stringify({ status: evidence.status, evidence: options.evidence, workspace }));
