/** 真实回归。参数为 test-output 下专门为本测试启动 MATLAB 的目录，内含 session/appdata。 */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, mkdtemp, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { apply } from '../harness-plugin/src/index.mjs';

const project = await realpath(resolve(import.meta.dirname, '..'));
const testOutput = await realpath(join(project, 'test-output'));
const prepare = process.argv[3] === '--prepare-workspace';
if (process.argv.length !== (prepare ? 4 : 3)) throw new Error('用法：node scripts/smoke-matlab-inport-save.mjs <test-output/matlab-inport-测试目录> [--prepare-workspace]');
const fixture = await realpath(resolve(process.argv[2]));
assert.equal(dirname(fixture), testOutput);
assert.match(basename(fixture), /^matlab-inport-[\w-]+$/);
const appdata = join(fixture, 'session', 'appdata');
assert.equal(await realpath(appdata), appdata);
const identityFile = join(appdata, 'control-harness-session.json');
const marker = JSON.parse(await readFile(identityFile, 'utf8'));
assert.equal(marker.kind, 'control-harness-dedicated-matlab-v1');
assert.equal(marker.status, 'ready');
assert.equal(resolve(marker.project_root), project);
assert.equal(resolve(marker.private_appdata), appdata);
const pythonPath = join(project, '.venv', 'Scripts', 'python.exe');
const resource = join(fixture, 'resource');
await mkdir(resource, {recursive:true});
assert.equal(await realpath(resource), resource);
const readState = () => readFile(join(resource, 'resource.json'), 'utf8').then(JSON.parse);
// 一个测试进程只绑定一个 broker。重试时保留旧隔离与租约，避免绕过终态校验。
try {
  const previous = await readState();
  assert.equal(previous.session?.session_id, marker.session_id);
  assert.equal(previous.session?.pid, marker.matlab_pid);
  assert.equal(previous.quarantine, null, '测试 MATLAB 已隔离，请另启一份专用测试会话');
  assert.equal(previous.owner, null, '测试 MATLAB 正被占用');
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  const bound = spawnSync(pythonPath, ['-m','control_harness.matlab_resource','bind-session',
    '--directory',resource,'--pid',String(marker.matlab_pid),'--session-id',marker.session_id],
  {cwd:project,env:{...process.env,PYTHONPATH:join(project,'src')},windowsHide:true,encoding:'utf8'});
  assert.equal(bound.status, 0, bound.stderr || bound.error?.message);
}
const workspace = await mkdtemp(join(fixture, 'workspace-'));
if (prepare) {
  // 仅当显式请求时，按项目既有启动方式准备刚创建的测试叶目录，不修改它的父目录。
  const prepared = spawnSync('powershell.exe',['-NoProfile','-NonInteractive','-File',
    join(project,'scripts','prepare-workspace.ps1'),'-Workspace',workspace,'-Prepare'],
  {cwd:project,windowsHide:true,encoding:'utf8'});
  assert.equal(prepared.status,0,prepared.stderr || prepared.error?.message);
}
const model = 'InportSmoke_' + randomUUID().slice(0,8);
const built = path => import(pathToFileURL(resolve(project, 'vendor/deepseek-harness', path, 'lib/index.js')));
const [{Context}, {Session}, prompt, tools, executor, local, projection, agents, jobs, commands, policy, sandbox, mcp] =
  await Promise.all(['vendor/cordis','packages/core/session','packages/core/system-prompt','packages/core/tools',
    'packages/core/tool-executor-native','packages/subprocess/subprocess-local','packages/session/session-projection',
    'packages/core/agent','packages/jobs/jobs-local','packages/interaction/commands','packages/sandbox/sandbox-policy',
    'packages/sandbox/sandbox-local','packages/mcp/mcp-client'].map(built));
const ctx = new Context();
const report = {status:'RUNNING',matlabPid:marker.matlab_pid,workspace,model,stages:[]};
const evidencePath = join(workspace,'smoke-evidence.json');
let stage = 'mount-harness';
try {
  for (const component of [prompt.default, tools.default, executor, local.default, projection.default,
    agents.default, jobs.default, commands.default, sandbox.default]) await ctx.plugin(component);
  await ctx.plugin(policy.default, {mode:'workspace-write',workspaceRoot:workspace});
  await ctx.plugin(mcp, {transport:'stdio',serverName:'matlab',
    command:process.env.CONTROL_MATLAB_MCP_EXECUTABLE || resolve(marker.toolkit_root,'../bin/matlab-mcp-server.exe'),
    args:['--matlab-session-mode=existing','--extension-file='+resolve(marker.toolkit_root,'tools/tools.json'),'--disable-telemetry=true'],
    env:{APPDATA:appdata},cwd:project,toolCallTimeoutMs:120000,failOnStartupError:true});
  apply(ctx, {codeRoot:project,pythonPath,maxConcurrentWorkers:1,timeoutSeconds:120,cancellationGraceMs:5000,
    matlabResource:{directory:resource,identityFile,toolPrefixes:['mcp__matlab__'],pollIntervalMs:25}});
  const id = 'session-' + randomUUID();
  const session = Session.create(id,undefined,{...Session.create(id).header,cwd:workspace});
  const agent = {id,session,ctx:ctx.plugin(()=>{}).ctx,options:{}};
  const evaluate = 'mcp__matlab__evaluate_matlab_code';
  const call = async (name,args) => {
    const batch = await ctx.tools.execute({agent,signal:AbortSignal.timeout(150000),
      calls:[{type:'tool_call',id:randomUUID(),name,arguments:JSON.stringify(args)}]});
    return batch.results[0].result;
  };
  const success = result => {
    assert.equal(result.isError,false,result.error?.message || '原生工具报错');
    return result.value;
  };
  const checkpoint = async (name,data) => {
    report.stages.push({name,...data});
    await writeFile(evidencePath,JSON.stringify(report,null,2));
  };
  stage = 'create-model';
  success(await call(evaluate,{code:[
    `new_system('${model}');`,
    `add_block('simulink/Sources/In1','${model}/u');`,
    `add_block('simulink/Sinks/Out1','${model}/y');`,
    `add_line('${model}','u/1','y/1');`
  ].join('\n'),project_path:workspace}));
  await checkpoint(stage,{});
  stage = 'save-model';
  const saved = success(await call('control_save_simulink_model',{model}));
  assert.equal(saved.path,join(workspace,model+'.slx'));
  assert.equal(saved.status,'saved');
  assert.equal(saved.simulation,'not_run');
  assert.equal(saved.bytes,(await stat(saved.path)).size);
  await checkpoint(stage,{receipt:saved});
  stage = 'invalid-inport-parameter';
  const failed = await call(evaluate, {code:`set_param('${model}/u','Value','1');`,project_path:workspace});
  assert.equal(failed.isError,true);
  assert.match(failed.error.message,/Inport block[^\n]*Value/i);
  const settled = await readState();
  assert.equal(settled.owner,null);
  assert.equal(settled.quarantine,null,'同步参数错误后，身份与空闲探针必须通过');
  assert.equal(settled.last_completion.evidence.matlab_pid,marker.matlab_pid);
  assert.equal(settled.last_completion.evidence.call_outcome,'synchronous_parameter_error');
  await checkpoint(stage,{diagnostic:failed.error.message.trim(),owner:settled.owner,quarantine:settled.quarantine,
    outcome:settled.last_completion.evidence.call_outcome,proof:settled.last_completion.evidence.producer});
  stage = 'reload-and-simulate';
  // 重新从确切路径加载，Dataset 喂给 Inport，Simulink 输出末值应等于已知输入。
  const quote = value => "'" + value.replaceAll("'","''") + "'";
  const simulation = success(await call(evaluate,{code:[
    `close_system('${model}',0); load_system(${quote(saved.path)});`,
    'ds = Simulink.SimulationData.Dataset;',
    "ds = ds.addElement(timeseries([2;2],[0;1]),'u');",
    `si = Simulink.SimulationInput('${model}'); si = si.setExternalInput(ds);`,
    "si = si.setModelParameter('StopTime','1','SaveOutput','on','OutputSaveName','yout','SaveFormat','Dataset');",
    "so = sim(si); y = so.get('yout'); value = y.getElement(1).Values.Data(end);",
    "fprintf('INPORT_SMOKE_FINAL=%.17g\\n',value);",
    `close_system('${model}',0);`
  ].join('\n'),project_path:workspace}));
  const result = simulation.content.filter(item=>item.type==='text').map(item=>item.text).join('\n');
  assert.match(result,/^INPORT_SMOKE_FINAL=2\s*$/m);
  const final = await readState();
  assert.equal(final.owner,null); assert.equal(final.quarantine,null);
  assert.equal(final.queue.length,0);
  await checkpoint(stage,{input:2,output:2,stopTimeSeconds:1,queueLength:final.queue.length,owner:null,quarantine:null});
  report.status = 'PASS';
} catch (error) {
  report.status = 'FAIL'; report.failedStage = stage;
  // 只输出固定测试步骤的诊断。启动阶段的外部服务器消息可能包含连接描述符。
  report.error = stage === 'mount-harness' ? 'Harness/MCP 服务未完成初始化' : String(error.message).slice(0,1200);
  process.exitCode = 1;
} finally {
  await ctx.fiber.dispose();
  await writeFile(evidencePath,JSON.stringify(report,null,2));
  console.log(JSON.stringify({status:report.status,evidencePath,failedStage:report.failedStage,error:report.error}));
}
