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
let launch = 'session';
try {
  launch = (await readFile(join(fixture,'active-test-launch.txt'),'utf8')).trim();
  assert.match(launch,/^session-[a-zA-Z0-9-]+$/);
} catch (error) { if (error.code !== 'ENOENT') throw error; }
const appdata = join(fixture, launch, 'appdata');
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
const report = {status:'RUNNING',matlabRelease:marker.matlab_release,matlabPid:marker.matlab_pid,workspace,model,stages:[]};
const evidencePath = join(workspace,'smoke-evidence.json');
let stage = 'mount-harness';
try {
  for (const component of [prompt.default, tools.default, executor, local.default, projection.default,
    agents.default, jobs.default, commands.default, sandbox.default]) await ctx.plugin(component);
  await ctx.plugin(policy.default, {mode:'workspace-write',workspaceRoot:workspace});
  await ctx.plugin(mcp, {transport:'stdio',serverName:'matlab',
    command:marker.mcp_executable || process.env.CONTROL_MATLAB_MCP_EXECUTABLE || resolve(marker.toolkit_root,'../bin/matlab-mcp-server.exe'),
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
  // 直接核对 broker 的释放凭证，同时要求调用者仍能看见原始 MATLAB 错误。
  const verifyRetryableError = async (name,result,expectedDiagnostic,outcome='synchronous_parameter_error') => {
    assert.equal(result.isError,true);
    assert.match(result.error.message,expectedDiagnostic);
    const settled = await readState();
    assert.equal(settled.owner,null);
    assert.equal(settled.quarantine,null,'同步错误后，身份与空闲探针必须通过');
    assert.equal(settled.last_completion.evidence.matlab_pid,marker.matlab_pid);
    assert.equal(settled.last_completion.evidence.session_id,marker.session_id);
    assert.equal(settled.last_completion.evidence.call_outcome,outcome);
    await checkpoint(name,{diagnostic:result.error.message.trim(),matlabPid:settled.last_completion.evidence.matlab_pid,
      matchedDedicatedSession:true,owner:settled.owner,quarantine:settled.quarantine,
      outcome:settled.last_completion.evidence.call_outcome,proof:settled.last_completion.evidence.producer});
  };
  const valueText = value => value.content.filter(item=>item.type==='text').map(item=>item.text).join('\n');
  const endpoint = (sourceBlock,sourcePort,destinationBlock,destinationPort) =>
    ({sourceBlock,sourcePort,destinationBlock,destinationPort});
  stage = 'toolkit-library-settings';
  const settings = success(await call(evaluate,{project_path:workspace,
    code:"settings = library.settingsLookup(); fprintf('SETTINGS=%s\\n',jsonencode(struct('found',settings.found,'gatePass',settings.gatePass)));"}));
  const gate = JSON.parse(valueText(settings).match(/^SETTINGS=(\{[^\r\n]*\})$/m)[1]);
  assert.ok(!gate.found || gate.gatePass,'测试工作区 Toolkit 库配置未就绪');
  await checkpoint(stage,gate);
  stage = 'create-model';
  success(await call(evaluate,{code:[
    `new_system('${model}');`,
    `add_block('simulink/Sources/In1','${model}/u');`,
    `add_block('simulink/Sinks/Out1','${model}/y');`
  ].join('\n'),project_path:workspace}));
  const passthrough = success(await call('control_wire_simulink',{model,connections:[endpoint('u',1,'y',1)]}));
  assert.equal(passthrough.status,'wired');
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
  await verifyRetryableError(stage,failed,/Inport block[^\n]*Value/i);
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

  // 原会话 #271 仅在新块上指定位置。完全重建默认块，避免之前的内存参数影响 #277 的复现。
  const plantModel = 'PlantSmoke_' + randomUUID().slice(0,8);
  const plant = plantModel + '/Plant';
  stage = 'create-default-plant';
  success(await call(evaluate,{project_path:workspace,code:[
    `new_system('${plantModel}');`,
    `add_block('simulink/Discrete/Discrete State-Space','${plant}');`,
  ].join('\n')}));
  stage = 'invalid-plant-parameter';
  const problem = await call(evaluate,{project_path:workspace,
    code:`set_param('${plant}','A','[0.99950087 0.00997338; -0.09973379 0.99451418]');`});
  await verifyRetryableError(stage,problem,/(参数\s*'B'.*设置无效|Invalid setting in .* for parameter 'B')/i);

  // 两列 B 分别接收控制力和扰动力；同时读回矩阵、初态、采样周期，以便核对纠正后的实际模型。
  const requested = {model:plantModel,block:'Plant',
    A:[[0.99950087,0.00997338],[-0.09973379,0.99451418]],
    B:[[0.00004991,0.00004991],[0.00997338,0.00997338]],C:[[1,0],[0,1]],D:[[0,0],[0,0]],
    initialCondition:[1,0],sampleTime:0.01};
  stage = 'configure-and-read-back-plant';
  const configured = success(await call('control_configure_simulink_plant',requested));
  assert.equal(configured.status,'configured');
  assert.equal(configured.stage,'readback');
  assert.equal(configured.model,plantModel);
  assert.equal(configured.block_path,plant);
  assert.equal(configured.parameters_applied,true);
  for (const key of ['states','inputs','outputs']) assert.equal(configured[key],2);
  assert.deepEqual(configured.ports,{input_count:1,output_count:1,required_input_width:2,required_output_width:2});
  for (const key of ['A','B','C','D']) assert.deepEqual(configured[key],requested[key],key);
  assert.deepEqual(configured.initial_condition,requested.initialCondition);
  assert.equal(configured.sample_time,requested.sampleTime);
  assert.equal(configured.diagram_update,'not_run');
  await checkpoint(stage,{receipt:configured});

  stage = 'prepare-wiring-fixture';
  // 通过与网页相同的 Toolkit 入口建立两个标量源和一个 Mux，确保两个分量的次序可以核对。
  const structure = {model:plantModel,scope:'root',depth:'0'};
  success(await call('mcp__matlab__model_read',structure));
  const added = success(await call('mcp__matlab__model_edit',{model:plantModel,scope:'root',layout_mode:'incremental',
    operations:JSON.stringify([
      {op:'add_block',type:'Constant',name:'Control',ref:'control',params:{Value:'2'}},
      {op:'add_block',type:'Constant',name:'Disturbance',ref:'dist',params:{Value:'-1'}},
      {op:'add_block',type:'Mux',name:'Inputs',ref:'mux',params:{Inputs:'2'}},
      {op:'add_block',type:'Outport',name:'State',ref:'state'},
    ])}));
  const blocks = valueText(success(await call('mcp__matlab__model_read',structure)));
  assert.match(valueText(added),/status:\s*ok/i);
  for (const name of ['Control','Disturbance','Inputs','Plant','State']) assert.ok(blocks.includes(name),name);
  const plan = [endpoint('Control',1,'Inputs',1),endpoint('Disturbance',1,'Inputs',2),
    endpoint('Inputs',1,'Plant',1),endpoint('Plant',1,'State',1)];
  const runWiring = connections => call('control_wire_simulink',{model:plantModel,connections});
  const verifyReadback = (receipt,actions) => {
    assert.equal(receipt.status,'wired'); assert.equal(receipt.stage,'readback');
    assert.equal(receipt.model,plantModel); assert.equal(receipt.connections.length,plan.length);
    assert.equal(receipt.created,actions.filter(x=>x==='created').length);
    assert.equal(receipt.existing,actions.filter(x=>x==='existing').length);
    receipt.connections.forEach((connection,index)=>{
      for (const key of Object.keys(plan[index])) assert.equal(connection[key],plan[index][key],key);
      assert.equal(connection.action,actions[index]);
      // 两个标量接入 Mux 的两个物理输入；其两个分量在 Plant 上共用一个物理输入。
      assert.equal(connection.sourcePortCount,1);
      assert.equal(connection.destinationPortCount,index < 2 ? 2 : 1);
      for (const width of [connection.sourceSignalWidth,connection.destinationSignalWidth]) {
        // 未编译的模型可以明确返回未知；有已编译宽度时必须与这次模型相符。
        if (Array.isArray(width)) assert.deepEqual(width,[]);
        else assert.equal(width,index < 2 ? 1 : 2);
      }
    });
  };
  stage = 'reject-literal-wiring-before-dispatch';
  const batchCode = `add_line('${plantModel}','Control/1','Inputs/1'); add_line('${plantModel}','Disturbance/1','Plant/2');`;
  const completionBefore = (await readState()).last_completion;
  const redirected = await call(evaluate,{project_path:workspace,code:batchCode});
  assert.equal(redirected.isError,true);
  assert.match(redirected.error.message,/control_wire_simulink/);
  assert.deepEqual((await readState()).last_completion,completionBefore);
  await checkpoint(stage,{dispatched:false});

  stage = 'reject-invalid-batch-before-any-line';
  // 第二条引用不存在的端口。独立查询第一条的目标是否仍未接线，直接验证预检的零写入。
  const refused = await runWiring([plan[0],endpoint('Disturbance',1,'Plant',2)]);
  assert.equal(refused.isError,true);
  assert.match(refused.error.message,/Plant/);
  assert.match(refused.error.message,/Mux/);
  const unchanged = success(await call(evaluate,{project_path:workspace,code:[
    `ports = get_param('${plantModel}/Inputs','PortHandles');`,
    "fprintf('EMPTY_INPUTS=%d\\n',all(arrayfun(@(p) get_param(p,'Line') == -1, ports.Inport)));",
  ].join('\n')}));
  assert.match(valueText(unchanged),/^EMPTY_INPUTS=1\s*$/m);
  await checkpoint(stage,{diagnostic:refused.error.message,unchangedInputs:2});

  stage = 'invalid-wiring-after-partial-legacy-batch';
  // 模拟旧建模脚本：第一条成功，第二条引用不存在的端口。原生 MATLAB 同步返回 #321，
  // 随后必须以同一 PID、同一专用会话的终态证明释放资源；此前成功的连线应由后续工具读回。
  const badWire = await call(evaluate,{project_path:workspace,code:[
    "sources = {'Control/1','Disturbance/1'}; destinations = {'Inputs/1','Plant/2'};",
    `for k = 1:2, add_line('${plantModel}',sources{k},destinations{k}); end`,
  ].join('\n')});
  await verifyRetryableError(stage,badWire,/Simulink 对象名称无效:\s*'Plant\/2'。|Invalid Simulink object name:\s*'Plant\/2'\./i,
    'synchronous_wiring_error');
  assert.equal(badWire.meta.matlab_wiring_retryable,true);

  stage = 'reject-conflicting-existing-line';
  // 已有线保持原状；即使计划把一条可用的新线放在冲突前，也不能留下多余修改。
  const conflict = await runWiring([plan[1],endpoint('Disturbance',1,'Inputs',1)]);
  assert.equal(conflict.isError,true);
  assert.match(conflict.error.message,/Inputs/);
  await checkpoint(stage,{diagnostic:conflict.error.message});
  stage = 'correct-and-read-back-wiring';
  const wired = success(await runWiring(plan));
  verifyReadback(wired,['existing','created','created','created']);
  const repeated = success(await runWiring(plan));
  verifyReadback(repeated,['existing','existing','existing','existing']);
  await checkpoint(stage,{first:wired,repeated});

  stage = 'update-and-save-corrected-plant';
  success(await call(evaluate,{project_path:workspace,code:[
    `set_param('${plantModel}','Solver','FixedStepDiscrete','FixedStep','0.01','StopTime','0.01');`,
    `set_param('${plantModel}','SimulationCommand','update');`,
  ].join('\n')}));
  const plantSaved = success(await call('control_save_simulink_model',{model:plantModel}));
  assert.equal(plantSaved.status,'saved'); assert.equal(plantSaved.simulation,'not_run');
  assert.equal(plantSaved.path,join(workspace,plantModel+'.slx'));
  assert.equal(plantSaved.bytes,(await stat(plantSaved.path)).size);
  await checkpoint(stage,{receipt:plantSaved});

  stage = 'reload-verify-and-simulate-corrected-plant';
  // 从工具返回的确切绝对路径重新加载，复核磁盘模型的接线，再从 t=0 开始仿真。
  success(await call(evaluate,{project_path:workspace,code:[
    `close_system('${plantModel}',0); assert(~bdIsLoaded('${plantModel}')); load_system(${quote(plantSaved.path)});`,
    `assert(strcmpi(get_param('${plantModel}','FileName'),${quote(plantSaved.path)}));`,
  ].join('\n')}));
  const reloaded = success(await runWiring(plan));
  verifyReadback(reloaded,['existing','existing','existing','existing']);
  const stepped = success(await call(evaluate,{project_path:workspace,code:[
    `si = Simulink.SimulationInput('${plantModel}');`,
    "si = si.setModelParameter('SaveOutput','on','OutputSaveName','yout','SaveFormat','Dataset');",
    "so = sim(si); series = so.get('yout').getElement(1).Values;",
    "x = getdatasamples(series,numel(series.Time));",
    "fprintf('PLANT_ONE_STEP=%s\\n',jsonencode([series.Time(end),reshape(x,1,[])]));",
    `close_system('${plantModel}',0);`
  ].join('\n')}));
  const text = stepped.content.filter(item=>item.type==='text').map(item=>item.text).join('\n');
  const sample = text.match(/^PLANT_ONE_STEP=(\[[^\r\n]*\])$/m);
  assert.ok(sample,'缺少本次图更新后的采样输出');
  const [time,...state] = JSON.parse(sample[1]);
  assert.ok(Math.abs(time-0.01) < 1e-12);
  const expected = requested.A.map((row,i)=>row[0]+2*requested.B[i][0]-requested.B[i][1]);
  assert.equal(state.length,2);
  state.forEach((value,i)=>assert.ok(Math.abs(value-expected[i]) < 1e-10,`state[${i}]=${value}, expected ${expected[i]}`));
  const released = await readState();
  assert.equal(released.owner,null); assert.equal(released.quarantine,null); assert.equal(released.queue.length,0);
  await checkpoint(stage,{path:plantSaved.path,wiresVerifiedAfterReload:reloaded.existing,
    initialState:requested.initialCondition,forces:[2,-1],time,state,expected,
    queueLength:released.queue.length,owner:null,quarantine:null});
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
