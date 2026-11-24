/** Real OS processes and native Cordis dispatch; remote-work flags are explicit failure-boundary fixtures. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { copyFile, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installMatlabResource, dispatchMatlabTaskBatch } from '../src/matlab-resource.mjs';
import { createMatlabTerminalVerifier } from '../src/matlab-terminal.mjs';
import { calculationScope } from '../src/matlab-calculation.mjs';
import { buildZohPlant } from '../src/simulink-build.mjs';
import {installMatlabRecovery} from '../src/matlab-recovery.mjs';
import {observeMatlabTransport,readMcpReply,withMcpReplyTracking} from '../src/matlab-mcp-reply.mjs';
import {setTimeout as sleep} from 'node:timers/promises';

const project = resolve(import.meta.dirname, '../..');
const upstream = resolve(project, 'vendor/deepseek-harness');
const python = resolve(project, '.venv/Scripts/python.exe');
const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/matlab-resource-tests'));
const builtUrl = path => pathToFileURL(resolve(upstream, path)).href;
const built = path => import(builtUrl(path));
await mkdir(output, { recursive: true });
const proof = { kind: 'synchronous_adapter_return', producer: 'node-test-trusted-synchronous-adapter' };

async function context(directory, options = {}) {
  const [{ Context }, { default: Prompt }, { default: Tools }, Executor, { default: Local }, mcp] = await Promise.all([
    built('vendor/cordis/lib/index.js'), built('packages/core/system-prompt/lib/index.js'),
    built('packages/core/tools/lib/index.js'), built('packages/core/tool-executor-native/lib/index.js'),
    built('packages/subprocess/subprocess-local/lib/index.js'), built('packages/mcp/mcp-client/lib/index.js'),
  ]);
  const ctx = new Context();
  await ctx.plugin(Prompt);
  await ctx.plugin(Tools);
  await ctx.plugin(Executor);
  await ctx.plugin(Local);
  const resource = installMatlabResource(ctx, { pythonPath: python, codeRoot: project,
    matlabResource: { directory, pollIntervalMs: 10, toolPrefixes: ['mcp__matlab__'] } },
  { verifyBeforeDispatch: async () => proof, ...options });
  const register = (name, call) => ctx.tools.register(mcp.createMcpToolDefinition(ctx, {
    name, rawName: name, description: 'Explicit MCP adapter boundary fixture',
    inputSchema: { type: 'object', properties: {} }, call,
  }));
  return { ctx, resource, register };
}

async function execute(ctx, name, signal = new AbortController().signal, args = {}) {
  const result = await ctx.tools.execute({ calls: [{ type: 'tool_call', id: `call-${Math.random()}`, name, arguments: JSON.stringify(args) }], signal });
  return result.results[0].result;
}

for (const [tool,kind] of [['model_read','completed'],['run_matlab_file','completed'],['model_read','wrong-id'],['run_matlab_file','server-timeout'],
  ['run_matlab_file','explicit-abort'],['run_matlab_file','abort-after-deadline']]) {
  test(`actual stdio after SDK timeout: ${tool} ${kind}`,{timeout:15000},async()=>fixture(async directory=>{
    const session=randomUUID(),name='mcp__matlab__'+tool;
    const bound=spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
      '--pid',String(process.pid),'--session-id',session],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')}});
    assert.equal(bound.status,0);
    const verify=async({probeNonce})=>({kind:'synchronous_adapter_return',producer:'control_harness_terminal_probe-v1',
      session_id:session,matlab_pid:process.pid,nonce:probeNonce??randomUUID()});
    const {ctx,resource}=await context(directory,{verifyBeforeDispatch:verify,verifyTerminal:verify});
    try {
      const bridge=await import('../src/matlab-mcp-client.mjs');
      await bridge.apply(ctx,{serverName:'matlab',transport:'stdio',command:process.execPath,env:{},cwd:directory,
        args:[resolve(import.meta.dirname,'fixtures/late-matlab-reply.mjs'),directory,kind],
        toolCallTimeoutMs:200,failOnStartupError:true});
      const recovery=installMatlabRecovery(ctx,resource,verify);
      const controller=new AbortController();
      const running=execute(ctx,name,controller.signal);
      if (kind==='explicit-abort') {
        while (!(await stat(join(directory,'dispatch.log')).catch(()=>undefined))) await sleep(5);
        controller.abort(new Error('operator requested stop'));
      }
      const failed=await running;
      assert.equal(failed.isError,true);
      assert.match(failed.error.message,kind==='explicit-abort' ? /aborted|operator requested stop/i : /timed out/i);
      if (kind==='abort-after-deadline') controller.abort(new Error('operator stopped after timeout'));
      const pending=(await state(directory)).quarantine;
      assert.equal(pending.owner.completion_source,'mcp_response');
      await recovery.check();
      assert.equal((await state(directory)).quarantine.token,pending.token,'idle alone cannot recover the original request');
      await writeFile(join(directory,'release'),'1');
      for (let i=0;i<6 && (await state(directory)).quarantine;i++) {await sleep(70);await recovery.check();}
      const settled=await state(directory);
      if (kind==='explicit-abort' || kind==='abort-after-deadline') {
        assert.match(await readFile(join(directory,'cancel.log'),'utf8'),/^\d+\n$/,'a genuine abort reached the server exactly once');
      } else {
        await assert.rejects(readFile(join(directory,'cancel.log')),{code:'ENOENT'},'an observation deadline must not cancel server execution');
      }
      if (kind==='completed') {
        assert.equal(settled.quarantine,null);
        const archive=JSON.parse(await readFile(settled.last_recovery,'utf8'));
        assert.equal(archive.evidence.completion_source,'mcp_response');
        assert.equal(archive.evidence.lease_token,pending.token);
        assert.equal(archive.evidence.tool_name,name);
        assert.ok(archive.evidence.transport_generation);
        assert.equal(archive.session.pid,process.pid);
      } else assert.equal(settled.quarantine.token,pending.token);
      assert.equal((await readFile(join(directory,'dispatch.log'),'utf8')).trim(),tool,'the original request was never replayed');
    } finally {await ctx.fiber.dispose();}
  }));
}

test('RPC id reuse on another connection cannot certify an interrupted MATLAB file',async()=>fixture(async directory=>{
  const schema={validate:value=>({value})};
  const first=observeMatlabTransport({send(){}},schema), second=observeMatlabTransport({send(){}},schema);
  const nonce=randomUUID(), token=randomUUID().replaceAll('-',''),name='mcp__matlab__run_matlab_file';
  const session={pid:process.pid,session_id:randomUUID(),identity:'bound PID creation fixture'};
  const lease={token,session,completion_nonce:nonce};
  const args={script_path:'temporary.m'};
  withMcpReplyTracking({config:{directory}},lease,{name,arguments:args},nonce,()=>first.send({
    jsonrpc:'2.0',id:7,method:'tools/call',params:{name:'run_matlab_file',arguments:args}}));
  const response={jsonrpc:'2.0',id:7,result:mcpSuccess('script completed')};
  for (const result of [{...response.result,resultType:'input_required',requestState:'awaiting input'},
    {...response.result,task:{taskId:'still-running'}}]) {
    const transport=observeMatlabTransport({send(){}},schema);
    withMcpReplyTracking({config:{directory}},lease,{name,arguments:args},nonce,()=>transport.send({
      jsonrpc:'2.0',id:7,method:'tools/call',params:{name:'run_matlab_file',arguments:args}}));
    transport.onmessage({...response,result});
    assert.equal(await readMcpReply(directory,lease,name),undefined,'an intermediate response cannot certify completion');
  }
  second.onmessage(response);
  assert.equal(await readMcpReply(directory,lease,name),undefined);
  first.onmessage(response);
  assert.equal((await readMcpReply(directory,lease,name)).rpc_request_id,7);
  assert.equal(await readMcpReply(directory,{...lease,completion_nonce:randomUUID()},name),undefined);
  assert.equal(await readMcpReply(directory,{...lease,session:{...session,identity:'reused pid'}},name),undefined);
}));

for (const tool of ['model_read','model_edit']) {
  for (const origin of ['matlab','sdk','server-transport']) {
    test(`project MCP bridge distinguishes ${tool} errors from ${origin}`, async()=>fixture(async directory=>{
      const session=randomUUID(), name='mcp__matlab__'+tool;
      const bound=spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
        '--pid',String(process.pid),'--session-id',session],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')}});
      assert.equal(bound.status,0);
      const trace=[];
      const verify=async()=>{
        trace.push('verify');
        return {kind:'synchronous_adapter_return',producer:'control_harness_terminal_probe-v1',
          nonce:randomUUID(),session_id:session,matlab_pid:process.pid};
      };
      const {ctx}=await context(directory,{verifyBeforeDispatch:verify,verifyTerminal:verify});
      try {
        const bridge=await import('../src/matlab-mcp-client.mjs');
        const message=origin==='server-transport' ? 'Connection closed' : 'matlab error: 图块不存在\nMATLAB multistep diagnostic';
        ctx.tools.register(bridge.createMcpToolDefinition(ctx,{name,rawName:tool,description:'test fixture',
          inputSchema:{type:'object',properties:{}}, call:async()=>{
            trace.push('call');
            if (origin==='sdk') throw new Error(message);
            return {isError:true,content:[{type:'text',text:message}]};
          }}));
        const result=await execute(ctx,name);
        assert.equal(result.isError,true);
        assert.equal(result.error.message,message);
        assert.equal(result.error.info?.code,origin==='sdk'?undefined:'MCP_TOOL_RETURNED_ERROR');
        assert.deepEqual(trace,origin==='matlab'?['verify','call','verify']:['verify','call']);
        assert.equal(Boolean((await state(directory)).quarantine),origin!=='matlab');
      } finally {await ctx.fiber.dispose();}
    }));
  }
}

async function fixture(callback) {
  const temporary = await mkdtemp(join(output, 'resource-node-'));
  try { return await callback(temporary); }
  finally { await rm(temporary, { recursive: true, force: true }); }
}

test('SATK reports only top-level error and partial statuses, preserving raw diagnostics and partial data',async()=>fixture(async directory=>{
  const {ctx}=await context(directory,{verifyTerminal:async()=>proof});
  const missing='[Autolayout] applied: model=Bench, scope=root, mode=full, duration_ms=647.2\n\nans = \n\n    "status: error\n     error_code: INVALID_PARAM\n     message: "找不到系统或文件 \'mpc_lib\'。""\n';
  const cases=[
    {tool:'model_edit',text:missing,status:'error'},
    {tool:'model_query_params',text:'status: partial\nparameters: [A,B]\nunknown: [C]',status:'partial'},
    {tool:'model_read',text:'ans = "status: ok\n    message: status: error"'},
    {tool:'model_read',text:'warning: observed an error\nstatus: error\nlog from subsystem'},
    {tool:'model_test',text:'structured failure',structuredContent:{status:'error'},status:'error'},
    {tool:'evaluate_matlab_code',text:'status: error\nprinted by the user code'},
  ];
  let active;
  try {
    const bridge=await import('../src/matlab-mcp-client.mjs');
    for (const tool of new Set(cases.map(item=>item.tool))) ctx.tools.register(bridge.createMcpToolDefinition(ctx,{
      name:'mcp__matlab__'+tool,rawName:tool,description:'test fixture',inputSchema:{type:'object'},
      call:async()=>({...mcpSuccess(active.text),...(active.structuredContent?{structuredContent:active.structuredContent}:{})})}));
    for (active of cases) {
      const result=await execute(ctx,'mcp__matlab__'+active.tool);
      assert.equal(result.isError, active.status==='error');
      assert.equal(result.meta?.matlab_tool_status,active.status);
      assert.ok(result.content.some(item=>item.text?.includes(active.text)));
      if (active.status==='error') assert.equal(result.error.message,active.text);
      if (active.status==='partial') {
        assert.match(result.content.at(-1).text,/部分完成/);
        assert.equal(result.value.content[0].text,active.text);
        assert.equal((await state(directory)).last_completion.evidence.call_outcome,'synchronous_tool_partial');
      }
      assert.equal((await state(directory)).quarantine,null);
    }
  } finally {await ctx.fiber.dispose();}
}));

const state = directory => readFile(join(directory, 'resource.json'), 'utf8').then(JSON.parse);
const mcpSuccess = text => ({ content: [{ type: 'text', text }] });

test('only closed numerical calculations select the synchronous receipt adapter', () => {
  assert.ok(calculationScope("A=[1 2;3 4]; B=ones(3,1); disp(A*B)"));
  assert.ok(calculationScope("p=ss([0 1;-1 -2],[0;1],[1 0],0); d=c2d(p,0.1,'zoh'); d.A"));
  for (const code of ['A=parfeval(@disp,0,1);', 'q=unlistedFunction(0);', 'x=x(1);', 'x=[1;x(1)];',
    "x=1; fprintf(x,'data');", "A=ones(3,'distributed');", "s='gpuArray'; A=zeros(2,s);",
    'q=1; q.delete;', 'a=1;\n!myprogram', 'x=timer;', 'A=otherWorkspaceVariable*2;']) {
    assert.equal(calculationScope(code),undefined,code);
  }
});

for (const fault of ['ordinary','timeout','cancel','cancel-at-proof','missing-receipt','wrong-pid','shadow','shadow-success','busy','denied']) {
  test(`native calculation requires a receipt and independent same-session proof: ${fault}`, async () => fixture(async directory => {
    const session = randomUUID(), abort = new AbortController();
    const native = 'mcp__matlab__evaluate_matlab_code';
    const bound = spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
      '--pid',String(process.pid),'--session-id',session],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')}});
    assert.equal(bound.status,0,bound.stderr?.toString());
    let probes=0, bodies=0;
    const verify = async () => {
      probes++;
      if (fault === 'busy' && probes === 2) throw new Error('Busy timer remains');
      if (fault === 'cancel-at-proof' && probes === 2) abort.abort();
      return {kind:'synchronous_adapter_return',producer:'control_harness_terminal_probe-v1',nonce:randomUUID(),
        session_id:session,matlab_pid:process.pid,observed_at:new Date().toISOString()};
    };
    const {ctx,register}=await context(directory,{verifyBeforeDispatch:verify,verifyTerminal:verify});
    try {
      if (fault==='denied') ctx.tools.guard(exec=>exec.arguments?.code?.includes('control_harness_calculation') ? 'nested native policy refused' : undefined);
      register(native,async ({code}) => {
        bodies++;
        const nonce = code.match(/'([a-f0-9-]{36})'/)[1];
        if (fault === 'timeout') throw new Error('SDK request deadline elapsed');
        if (fault === 'cancel') { abort.abort(); throw new Error('Cancelled during execution'); }
        if (fault === 'missing-receipt') return mcpSuccess('partial output');
        const receipt={nonce,session_id:session,matlab_pid:process.pid+(fault==='wrong-pid'?1:0),
          helper:resolve(project,'matlab/control_harness_calculation'),certifiable:!fault.startsWith('shadow'),
          status:fault==='shadow-success'?'returned':'failed',identifier:'MATLAB:fixture',message:'原始诊断'};
        return mcpSuccess('prior output\nCONTROL_HARNESS_CALC_'+nonce+':'+JSON.stringify(receipt));
      });
      const result=await execute(ctx,native,abort.signal,{code:'A=[1 2]; A*ones(3,1)',project_path:directory});
      assert.equal(result.isError,true);
      const ledger=await state(directory);
      if (fault==='ordinary') {
        assert.equal(result.meta.matlab_calculation.identifier,'MATLAB:fixture');
        assert.equal(result.meta.matlab_calculation.retryable,true);
        assert.equal(result.error.message,'原始诊断');
        assert.equal(ledger.last_completion.evidence.call_outcome,'synchronous_calculation_error');
      }
      if (['busy','shadow','shadow-success'].includes(fault)) {
        assert.equal(result.meta.matlab_calculation.identifier,'MATLAB:fixture');
        assert.equal(result.meta.matlab_calculation.retryable,false);
        assert.match(result.error.message,/原始诊断/);
      }
      assert.equal(Boolean(ledger.quarantine),!['ordinary','denied'].includes(fault));
      assert.equal(probes,['ordinary','busy','cancel-at-proof'].includes(fault)?2:1);
      assert.equal(bodies,fault==='denied'?0:1);
    } finally {await ctx.fiber.dispose();}
  }));
}

test('pending MATLAB work blocks only MATLAB; non-MATLAB tools remain usable in this and other turns',async()=>fixture(async directory=>{
  const {ctx,register}=await context(directory);
  const [projection,agents,{Session}]=await Promise.all([
    built('packages/session/session-projection/lib/index.js'), built('packages/core/agent/lib/index.js'),
    built('packages/core/session/lib/index.js'),
  ]);
  try {
    await ctx.plugin(projection.default);
    await ctx.plugin(agents.default);
    function chat() {
      const id=randomUUID();
      return {id,session:Session.create(id),ctx:ctx.plugin(()=>{}).ctx,options:{}};
    }
    const first=chat(), other=chat();
    for (const chat of [first,other]) chat.session.append('turn/start',{turn:1});
    const bodies={pwsh:0,write_file:0};
    for (const name of Object.keys(bodies)) register(name,async()=>{bodies[name]++;return mcpSuccess('ordinary work');});
    let matlab=0;
    register('mcp__matlab__evaluate_matlab_code',async()=>{matlab++;throw new Error('test-only MCP transport interruption');});
    const call=async(agent,name)=>{
      const batch=await ctx.tools.execute({agent,signal:new AbortController().signal,
        calls:[{type:'tool_call',id:randomUUID(),name,arguments:'{}'}]});
      return batch.results[0].result;
    };
    const failed=await call(first,'mcp__matlab__evaluate_matlab_code');
    assert.ok(failed.isError); assert.ok((await state(directory)).quarantine);
    assert.equal((await call(first,'pwsh')).isError,false);
    assert.equal((await call(first,'write_file')).isError,false);
    assert.equal((await call(other,'pwsh')).isError,false);
    assert.equal((await call(first,'mcp__matlab__evaluate_matlab_code')).isError,true);
    first.session.append('turn/end',{turn:1,reason:{kind:'completed'}});
    first.session.append('turn/start',{turn:2});
    assert.equal((await call(first,'write_file')).isError,false);
    assert.equal((await call(first,'mcp__matlab__evaluate_matlab_code')).isError,true);
    assert.equal(matlab,1); assert.deepEqual(bodies,{pwsh:2,write_file:2});
  } finally {await ctx.fiber.dispose();}
}));

for (const fault of ['recoverable','unknown','busy','wrong-pid','bad-receipt','bad-receipt-completed','raw-error-completed','transport','permission','success']) {
  test(`controlled ZOH model call retains native and resource outcomes: ${fault}`, async () => fixture(async directory => {
    const session = randomUUID(), model = 'f01_spring', target = join(directory,model+'.slx');
    const native = 'mcp__matlab__evaluate_matlab_code';
    const bound = spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
      '--pid',String(process.pid),'--session-id',session],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')},encoding:'utf8'});
    assert.equal(bound.status,0,bound.stderr);
    const identityFile = join(directory,'marker.json');
    await writeFile(identityFile,JSON.stringify({schema_version:1,kind:'control-harness-dedicated-matlab-v1',
      status:'ready',project_root:project,matlab_pid:process.pid,session_id:session}));
    let ctx, probes=0, bodies=0, completed='';
    const verify = options => createMatlabTerminalVerifier(ctx,{identityFile,codeRoot:project})(options);
    const mounted = await context(directory,{verifyBeforeDispatch:verify,verifyTerminal:verify});
    ctx = mounted.ctx;
    const stageList = ['validation','create','configure','wire','save','reload','simulate','compare'];
    try {
      mounted.register(native,async ({code}) => {
        const nonce = code.match(/CONTROL_HARNESS_PROBE_([a-f0-9-]{36}):/)?.[1];
        if (nonce) {
          probes++;
          return mcpSuccess('CONTROL_HARNESS_PROBE_'+nonce+':'+JSON.stringify({schema_version:1,nonce,session_id:session,
            matlab_pid:process.pid,helper_file:resolve(project,'matlab/control_harness_terminal_probe'),idle:true,busy_models:[],
            completion_nonce:completed,background_futures:0,parallel_futures:0,running_timers:fault === 'busy' && probes === 2 ? 2 : 0}));
        }
        bodies++;
        if (fault === 'transport') throw new Error('Fixture MCP connection lost');
        assert.ok(code.includes(agent.id));
        assert.ok(code.includes('{"damping":0.8}'));
        const marker = code.match(/CONTROL_HARNESS_MODEL_[a-f0-9-]{36}:/)[0];
        if (fault.endsWith('-completed')) completed=marker.slice('CONTROL_HARNESS_MODEL_'.length,-1);
        if (fault==='raw-error-completed') return {isError:true,content:[{type:'text',text:'MATLAB failed before model receipt\n仿真数据错误'}]};
        const receipt = {schema_version:1, status:'failed', stage:'configure',model,path:target,verified:['create'],
          changes:['block creation'],file_exists:false,release:'2024b',matlab_pid:process.pid,identifier:'ControlHarness:Fixture',
          reason:'fixture modeling diagnostic', recoverable:fault !== 'unknown'};
        if (fault === 'wrong-pid') receipt.matlab_pid++;
        if (fault.startsWith('bad-receipt')) receipt.path = 'wrong';
        if (fault === 'success') {
          await copyFile(join(project,'matlab/fixtures/f01_spring.slx'),target);
          Object.assign(receipt,{status:'simulated',stage:'compare',verified:stageList,file_exists:true,bytes:(await stat(target)).size,
            default_samples:11,comparison:{compared:true,trajectories:1,samples:2,max_abs_error:0,tolerance:1e-8}});
        }
        return mcpSuccess(marker+JSON.stringify(receipt));
      });
      if (fault === 'permission') ctx.tools.guard(()=> 'fixture native refusal');
      const id=randomUUID();
      // Native plugin resolves agent-scoped services using a real Cordis child context.
      const agent={id,session:{id,header:{cwd:directory}},ctx:ctx.plugin(()=>{}).ctx,options:{}};
      const host={tools:ctx.tools,sandboxPolicy:{resolve:()=>({mode:'workspace-write',workspaceRoot:directory})}};
      const config={codeRoot:project,matlabResource:{directory,identityFile}};
      const reference = {absTolerance:1e-8,parameters:{damping:0.8},trajectories:[{initialCondition:[1],inputs:[[0]],states:[[1],[1]]}]};
      await writeFile(join(directory,'trace.json'),JSON.stringify(reference));
      const build = () => buildZohPlant(host,config,{model,Ac:[[0]],Bc:[[1]],initialCondition:[1],
        sampleTime:0.1,inputNames:['u'],referenceFile:'trace.json'},{agent,signal:new AbortController().signal},async()=>{});
      if (fault === 'success') {
        const built = await build();
        assert.equal(built.status,'simulated');
        assert.ok(built.session_verified && built.file_exists && built.comparison.compared);
      } else {
        let error;
        try { await build(); } catch (caught) { error=caught; }
        const r = JSON.parse(error.message);
        if (['recoverable','unknown'].includes(fault)) {
          assert.equal(r.identifier,'ControlHarness:Fixture'); assert.equal(r.reason,'fixture modeling diagnostic');
          assert.equal(r.session_verified,true); assert.equal(r.retryable,true); assert.deepEqual(r.changes,['block creation']);
        } else if (fault.endsWith('-completed')) {
          assert.equal(r.status,'failed'); assert.equal(r.session_verified,true); assert.equal(r.retryable,false);
        } else {
          assert.equal(r.status,fault === 'permission' ? 'not_dispatched' : 'quarantined');
          assert.equal(r.retryable,false);
          if (fault === 'busy') assert.match(r.verification_error,/运行中计时器 2/);
        }
      }
      const ledger=await state(directory), settled=['success','recoverable','unknown','permission'].includes(fault) || fault.endsWith('-completed');
      assert.equal(Boolean(ledger.quarantine),!settled);
      if (!settled) await assert.rejects(build(), /quarantined/);
      assert.equal(bodies,fault === 'permission'?0:1);
      assert.equal(probes,fault === 'permission'?0:2);
    } finally {await ctx.fiber.dispose();}
  }));
}

// 重放实际会话中的 MCP 错误：原生工具桥会在会话记录外层另加 "Error: "。
// 仅 MATLAB 服务端为替身；权限检查、探针解析和 Python 资源代理使用真实实现。
const inport = {
  label: 'Inport Value',
  text: "matlab error: Inport block 没有名为 'Value' 的参数\n \n",
  invalid: {code:"set_param('MassSpringDamper/u_force','Value','1');"},
  corrected: {code:"set_param('MassSpringDamper/u_force','Port','1');"},
};
const plant = {
  label: 'event 277 Plant B',
  text: `matlab error: 参数 'B' 的 '<a href="matlab:open_and_hilite_hyperlink ('MPC_MassSpringDamper/Plant','error')">MPC_MassSpringDamper/Plant</a>' 中的设置无效。\n \n`,
  invalid: {code:"set_param('MPC_MassSpringDamper/Plant','A','eye(2)');"},
  corrected: {code:"set_param('MPC_MassSpringDamper/Plant','A','eye(2)','B','eye(2)','C','eye(2)','D','zeros(2)','InitialCondition','[1;0]','SampleTime','0.01');"},
};
const wiring = {
  label: 'event 321 Plant/2', category:'wiring',
  text: "matlab error: Simulink 对象名称无效: 'Plant/2'。\n \n",
  invalid: {code:"add_line(modelName,'Disturbance/1','Plant/2');"},
  corrected: {code:"add_line(modelName,'PlantInputMux/1','Plant/1');"},
};
const missingLibrary = {
  label:'missing MPC library with preceding output',category:'generic',
  text:"discrete model created\n工具库缺失：mpc_lib/MPC Controller\n" + '先前输出'.repeat(400),
  invalid:{code:"add_block('mpc_lib/MPC Controller','plant/MPC');"},
  corrected:{code:"add_block('mpclib/MPC Controller','plant/MPC');"},
};

for (const scenario of [inport,plant,wiring,missingLibrary]) {
  for (const fault of ['none','busy','different-session','missing-marker','stale-marker','timeout','transport','cancel']) {
    test(`${scenario.label}: completion evidence ${fault}`, async () => fixture(async directory => {
      const identityFile = join(directory,'identity.json'), sessionId = randomUUID();
      const bound = spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
        '--pid',String(process.pid),'--session-id',sessionId],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')}});
      assert.equal(bound.status,0);
      await writeFile(identityFile,JSON.stringify({schema_version:1,kind:'control-harness-dedicated-matlab-v1',
        status:'ready',project_root:project,matlab_pid:process.pid,session_id:sessionId}));
      let ctx, completion='',probes=0,bodies=0,permissions=0;
      const verify = options=>createMatlabTerminalVerifier(ctx,{identityFile,codeRoot:project})(options);
      const mounted=await context(directory,{verifyBeforeDispatch:verify,verifyTerminal:verify}); ctx=mounted.ctx;
      const controller=new AbortController(), name='mcp__matlab__evaluate_matlab_code';
      try {
        ctx.tools.guard(()=>{permissions++;});
        mounted.register(name, async ({code,project_path})=>{
          const probe=code.match(/CONTROL_HARNESS_PROBE_([a-f0-9-]{36}):/)?.[1];
          if (probe) {
            probes++;
            return mcpSuccess('CONTROL_HARNESS_PROBE_'+probe+':'+JSON.stringify({schema_version:1,nonce:probe,
              session_id:probes===2 && fault==='different-session' ? randomUUID():sessionId,matlab_pid:process.pid,
              helper_file:resolve(project,'matlab/control_harness_terminal_probe'),idle:true,busy_models:[],
              background_futures:0,parallel_futures:0,running_timers:fault==='busy' && probes===2 ? 1:0,
              completion_nonce:completion,completion_outcome:'failed',completion_identifier:'Simulink:Fixture'}));
          }
          bodies++;
          assert.equal(project_path,directory);
          assert.ok(code.includes('control_harness_evaluate'));
          const token=code.match(/'([a-f0-9-]{36})'\);$/)?.[1];
          assert.ok(token);
          if (fault==='timeout') throw new Error('Request timed out');
          if (fault==='transport') throw new Error('Connection closed');
          if (fault==='cancel') {controller.abort();throw new Error('cancel fixture');}
          completion=fault==='stale-marker' ? randomUUID() : fault==='missing-marker'?'':token;
          if (bodies===1) {
            assert.ok(code.includes(scenario.invalid.code.replaceAll("'","''")));
            return {isError:true,content:[{type:'text',text:scenario.text}]};
          }
          assert.ok(code.includes(scenario.corrected.code.replaceAll("'","''")));
          return mcpSuccess('corrected');
        });
        const result=await execute(ctx,name,controller.signal,{...scenario.invalid,project_path:directory});
        assert.equal(result.isError,true);
        const ledger=await state(directory);
        if (fault==='none') {
          assert.equal(ledger.quarantine,null);
          assert.equal(result.meta.matlab_session_verified,true);
          assert.equal(result.meta.matlab_error.identifier,'Simulink:Fixture');
          assert.equal(result.error.message,scenario.text);
          if (scenario.category!=='generic') assert.ok(result.meta[`matlab_${scenario.category??'parameter'}_retryable`]);
          assert.equal((await execute(ctx,name,undefined,{...scenario.corrected,project_path:directory})).isError,false);
          assert.equal(bodies,2); assert.equal(probes,4); assert.equal(permissions,8);
        } else {
          assert.ok(ledger.quarantine?.owner.completion_nonce);
          assert.equal((await execute(ctx,name,undefined,{...scenario.corrected,project_path:directory})).isError,true);
          assert.equal(bodies,1);
        }
      } finally {await ctx.fiber.dispose();}
    }));
  }
}

test('a scoped task batches the two signed identity probes under one native permission and one broker lease',async()=>fixture(async directory=>{
  // The bound PID is this test process. A fixture MCP callback substitutes for MATLAB.
  const identity=randomUUID();
  const bound=spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,'--pid',String(process.pid),'--session-id',identity],
    {cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')},windowsHide:true,encoding:'utf8'});
  assert.equal(bound.status,0,bound.stderr);
  const binding=JSON.parse(bound.stdout), scope={helper:'control_harness_collect_batch',native_tool:'mcp__matlab__evaluate_matlab_code',dedicated_session:binding};
  const {ctx,register}=await context(directory,{verifyTerminal:async()=>assert.fail('no separate probe')});
  const root=resolve(project,'matlab'), helper=resolve(root,'control_harness_terminal_probe');
  let permissions=0,dispatched=0,allow=true,omitPost=false;
  try {
    ctx.tools.guard(()=>{permissions++;return allow?undefined:'Fixture native denial';});
    register(scope.native_tool,async ({code,project_path})=>{
      assert.equal(project_path,root);
      const [nonce,session,before,after]=[...code.matchAll(/'([a-f0-9-]{36})'/g)].map(match=>match[1]);
      assert.equal(session,identity);assert.ok(nonce);
      return mcpSuccess((omitPost?[before]:[before,after]).map(token=>'CONTROL_HARNESS_PROBE_'+token+':'+JSON.stringify({
        schema_version:1,nonce:token,session_id:identity,matlab_pid:process.pid,helper_file:helper,
        busy_models:[],background_futures:0,parallel_futures:0,running_timers:0,idle:true})).join('\n'));
    });
    const dispatch=()=>dispatchMatlabTaskBatch({ctx,grant:{scope,validate:async lease=>assert.deepEqual(lease,binding)},
      onDispatch(){dispatched++;},signal:new AbortController().signal,manifestPath:resolve(directory,'manifest.json'),
      manifestSha:'a'.repeat(64),nonce:randomUUID(),helperRoot:root});
    allow=false;assert.equal((await dispatch()).results[0].result.isError,true);
    assert.equal(dispatched,0);assert.equal((await state(directory)).quarantine,null);
    allow=true;assert.equal((await dispatch()).results[0].result.isError,false);
    assert.equal(dispatched,1);assert.equal(permissions,2);
    assert.equal((await state(directory)).quarantine,null);
    // 即便卡片和 broker 文件一致，也要重新向 OS 核对创建时间。
    const healthy=await state(directory);
    const stale={...binding,identity:'stale-process-creation-fixture'};
    await writeFile(resolve(directory,'resource.json'),JSON.stringify({...healthy,session:stale}));
    scope.dedicated_session=stale;
    const refused=await dispatchMatlabTaskBatch({ctx,grant:{scope,validate:async lease=>assert.deepEqual(lease,stale)},
      onDispatch(){assert.fail('No dispatch to stale PID');},signal:new AbortController().signal,manifestPath:resolve(directory,'manifest.json'),
      manifestSha:'a'.repeat(64),nonce:randomUUID(),helperRoot:root});
    assert.equal(refused.results[0].result.isError,true);
    assert.equal((await state(directory)).owner,null);
    assert.equal((await state(directory)).quarantine,null);
    // Restore this synthetic ledger before the separate missing-certificate case.
    await writeFile(resolve(directory,'resource.json'),JSON.stringify(healthy));
    scope.dedicated_session=binding;
    omitPost=true;assert.equal((await dispatch()).results[0].result.isError,true);
    assert.equal(dispatched,2);assert.equal(permissions,4);
    assert.ok((await state(directory)).quarantine);
  } finally {await ctx.fiber.dispose();}
}));

test('native permissions reject before admission; unconfigured Python-only mode adds no MATLAB hook', async () => fixture(async directory => {
  const { ctx, register } = await context(directory);
  let calls = 0;
  try {
    register('mcp__matlab__denied', async () => { calls += 1; return mcpSuccess('must not run'); });
    ctx.tools.guard(exec => exec.name === 'mcp__matlab__denied' ? 'Native permission fixture denies MATLAB' : undefined);
    const denied = await execute(ctx, 'mcp__matlab__denied');
    assert.equal(denied.isError, true);
    assert.equal(calls, 0);
    await assert.rejects(readFile(join(directory, 'resource.json')), { code: 'ENOENT' });
    assert.equal(installMatlabResource(ctx, { pythonPath: python, codeRoot: project }), undefined);
  } finally { await ctx.fiber.dispose(); }
}));

test('cancellation while wrapped MATLAB code awaits native permission releases only the reservation',async()=>fixture(async directory=>{
  const session=randomUUID(),name='mcp__matlab__evaluate_matlab_code',abort=new AbortController();
  const bound=spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
    '--pid',String(process.pid),'--session-id',session],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')}});
  assert.equal(bound.status,0);
  const {ctx,register}=await context(directory,{verifyTerminal:async()=>assert.fail('No MATLAB command was sent')});
  let enter,release,calls=0;
  const checking=new Promise(resolve=>{enter=resolve;});
  const answer=new Promise(resolve=>{release=resolve;});
  try {
    ctx.on('tools/pre-execute',async (exec,next)=>{
      if (exec.arguments?.code?.includes('control_harness_evaluate')) {enter(); await answer;}
      return next();
    });
    register(name,async()=>{calls++;return mcpSuccess('unexpected body');});
    const pending=execute(ctx,name,abort.signal,{code:"add_block('mpclib/MPC Controller','plant/MPC');"});
    await checking;
    assert.equal((await state(directory)).owner.phase,'reserved');
    abort.abort(); release();
    assert.equal((await pending).isError,true);
    const ledger=await state(directory);
    assert.equal(calls,0); assert.equal(ledger.quarantine,null); assert.equal(ledger.owner,null);
  } finally {release();await ctx.fiber.dispose();}
}));

test('failed scoped batch can release its resource without publishing a successful job',async()=>fixture(async directory=>{
  const session=randomUUID(), name='mcp__matlab__evaluate_matlab_code';
  const bound=spawnSync(python,['-m','control_harness.matlab_resource','bind-session','--directory',directory,
    '--pid',String(process.pid),'--session-id',session],{cwd:project,env:{...process.env,PYTHONPATH:resolve(project,'src')},encoding:'utf8'});
  assert.equal(bound.status,0);
  const scope={helper:'control_harness_collect_batch',native_tool:name,dedicated_session:JSON.parse(bound.stdout)};
  const nonce=randomUUID(), errorMessage='MATLAB batch error\n输入数据维度无效';
  const {ctx,register}=await context(directory,{verifyTerminal:async({completionNonce})=>{
    assert.equal(completionNonce,nonce);
    return {kind:'synchronous_adapter_return',producer:'control_harness_terminal_probe-v1',
      session_id:session,matlab_pid:process.pid,completion_nonce:nonce,nonce:randomUUID()};
  }});
  try {
    let calls=0;
    register(name,async()=>{
      calls++;
      return {isError:true,content:[{type:'text',text:errorMessage}]};
    });
    const batch=await dispatchMatlabTaskBatch({ctx,grant:{scope,validate:async()=>{}},onDispatch(){},
      signal:new AbortController().signal,manifestPath:resolve(directory,'manifest.json'),
      manifestSha:'a'.repeat(64),nonce,helperRoot:resolve(project,'matlab')});
    const result=batch.results[0].result;
    assert.equal(calls,1); assert.equal(result.isError,true);
    assert.equal(result.error.message,errorMessage); assert.equal(result.meta.matlab_session_verified,true);
    assert.equal((await state(directory)).quarantine,null);
  } finally {await ctx.fiber.dispose();}
}));

test('trusted terminal probe re-enters native permissions while one lease permits consecutive calls', async () => fixture(async directory => {
  let ctx, probeCalls = 0, bodyCalls = 0, permissionChecks = 0;
  const verify = async ({ lease }) => lease.withInternalProbe(async () => {
    const result = await execute(ctx, 'mcp__matlab__fixed_probe');
    assert.equal(result.isError, false);
    return proof;
  });
  const mounted = await context(directory, { verifyBeforeDispatch: verify, verifyTerminal: verify });
  ctx = mounted.ctx;
  try {
    mounted.register('mcp__matlab__sync', async () => { bodyCalls += 1; return mcpSuccess('synchronous fixture completed'); });
    mounted.register('mcp__matlab__fixed_probe', async () => { probeCalls += 1; return mcpSuccess('trusted fixed probe fixture'); });
    ctx.tools.guard(() => { permissionChecks += 1; return undefined; });
    for (let i = 0; i < 2; i += 1) assert.equal((await execute(ctx, 'mcp__matlab__sync')).isError, false);
    assert.equal(bodyCalls, 2, 'next dispatches each requested tool exactly once');
    assert.equal(probeCalls, 4, 'fixed pre/post probes execute under the same lease');
    assert.equal(permissionChecks, 6, 'internal probing cannot bypass native permission guards');
    assert.equal((await state(directory)).quarantine, null);
  } finally { await ctx.fiber.dispose(); }
}));

test('native queued cancellation preserves the owner and returns canonical abort before tool dispatch', async () => fixture(async directory => {
  const { ctx, resource, register } = await context(directory, { verifyTerminal: async () => proof });
  const owner = await resource.acquire('other-host-owner-fixture', new AbortController().signal);
  let calls = 0;
  try {
    register('mcp__matlab__queued', async () => { calls += 1; return mcpSuccess('should dispatch only after release'); });
    const abort = new AbortController();
    const pending = execute(ctx, 'mcp__matlab__queued', abort.signal);
    const deadline = Date.now() + 10000;
    while ((await state(directory)).queue.length !== 1) {
      assert.ok(Date.now() < deadline, 'Native call must actually enqueue before cancellation');
      await new Promise(resolveWait => setTimeout(resolveWait, 10));
    }
    abort.abort(new Error('Queued native invocation cancelled'));
    const result = await pending;
    assert.equal(result.isError, true);
    assert.equal(result.error.info.code, 'ABORTED_BEFORE_DISPATCH');
    assert.equal(calls, 0);
    const remaining = await state(directory);
    assert.equal(remaining.owner.token, owner.token);
    assert.equal(remaining.quarantine, null);
    assert.deepEqual(remaining.queue, []);
    await owner.cancelBeforeDispatch();
    assert.equal((await execute(ctx, 'mcp__matlab__queued')).isError, false);
    assert.equal(calls, 1);
  } finally { if (!owner.closed) await owner.cancelBeforeDispatch(); await ctx.fiber.dispose(); }
}));

for (const precondition of ['wrong-session', 'busy', 'missing-verifier']) {
  test(`${precondition} pre-dispatch probe prevents the original MATLAB tool from running`, async () => fixture(async directory => {
    let calls = 0;
    const verifyBeforeDispatch = precondition === 'missing-verifier' ? undefined : async ({ lease }) => {
      assert.equal(lease.started, false, 'Only the reserved caller may probe; its business command is not yet running');
      if (precondition === 'wrong-session') throw new Error('Dedicated session UUID/PID mismatch');
      return undefined; // Fixed probe reported active simulation/background work.
    };
    const { ctx, register } = await context(directory, { verifyBeforeDispatch, verifyTerminal: async () => assert.fail('No post-verification without original dispatch') });
    try {
      register('mcp__matlab__must_not_execute', async () => { calls += 1; return mcpSuccess('must not run'); });
      assert.equal((await execute(ctx, 'mcp__matlab__must_not_execute')).isError, true);
      assert.equal(calls, 0);
      assert.equal((await state(directory)).quarantine,null);
    } finally { await ctx.fiber.dispose(); }
  }));
}

test('an ordinary successful MCP result without trusted terminal proof remains quarantined', async () => fixture(async directory => {
  const { ctx, register } = await context(directory);
  let calls = 0;
  try {
    register('mcp__matlab__async_start', async () => { calls += 1; return mcpSuccess('accepted background work; idle=true is untrusted'); });
    assert.equal((await execute(ctx, 'mcp__matlab__async_start')).isError, true);
    assert.equal((await state(directory)).quarantine.reason, 'mcp_success_without_trusted_terminal_proof');
    assert.equal((await execute(ctx, 'mcp__matlab__async_start')).isError, true);
    assert.equal(calls, 1, 'unknown background state prevents later MATLAB dispatch');
  } finally { await ctx.fiber.dispose(); }
}));

test('cancellation arriving during terminal verification cannot turn into a clean release', async () => fixture(async directory => {
  const abort = new AbortController();
  const { ctx, register } = await context(directory, { verifyTerminal: async () => {
    abort.abort(new Error('Cancelled during terminal probe'));
    return proof;
  } });
  try {
    register('mcp__matlab__sync', async () => mcpSuccess('body returned before cancellation'));
    assert.equal((await execute(ctx, 'mcp__matlab__sync', abort.signal)).isError, true);
    assert.equal((await state(directory)).quarantine.reason, 'mcp_cancelled_state_unknown');
  } finally { await ctx.fiber.dispose(); }
}));

test('a failed fixed terminal verifier quarantines even after a successful MCP body', async () => fixture(async directory => {
  const { ctx, register } = await context(directory, { verifyTerminal: async () => { throw new Error('Dedicated session identity cannot be verified'); } });
  try {
    register('mcp__matlab__sync', async () => mcpSuccess('body returned'));
    assert.equal((await execute(ctx, 'mcp__matlab__sync')).isError, true);
    assert.equal((await state(directory)).quarantine.reason, 'mcp_dispatch_or_wrapper_failed_state_unknown');
  } finally { await ctx.fiber.dispose(); }
}));

for (const boundary of ['transport', 'mcp-isError', 'timeout', 'cancel']) {
  test(`${boundary} settlement quarantines while remote-work termination remains unverified`, async () => fixture(async directory => {
    const abort = new AbortController();
    let remoteWorkActive = false, calls = 0, admitted;
    const started = new Promise(resolveStarted => { admitted = resolveStarted; });
    const { ctx, register } = await context(directory, { verifyTerminal: async () => assert.fail('Error outcomes must not invoke the terminal-success verifier') });
    try {
      register('mcp__matlab__failure', async (_args, exec) => {
        calls += 1;
        remoteWorkActive = true;  // Deliberately outlives SDK settlement in this fixture.
        admitted();
        if (boundary === 'mcp-isError') return { isError: true, content: [{ type: 'text', text: 'MATLAB reported an error' }] };
        if (boundary === 'timeout') await new Promise((_resolve, reject) => setTimeout(() => reject(new Error('SDK request deadline elapsed')), 20));
        if (boundary === 'cancel') await new Promise((_resolve, reject) => exec.signal.addEventListener('abort', () => reject(new Error('SDK cancelled request')), { once: true }));
        throw new Error('MCP transport disconnected');
      });
      const pending = execute(ctx, 'mcp__matlab__failure', abort.signal);
      await started;
      if (boundary === 'cancel') abort.abort(new Error('User cancelled while MATLAB may still run'));
      assert.equal((await pending).isError, true);
      assert.equal(remoteWorkActive, true, 'Promise settlement is not a computation-stop certificate');
      assert.ok((await state(directory)).quarantine);
      assert.equal((await execute(ctx, 'mcp__matlab__failure')).isError, true);
      assert.equal(calls, 1, 'quarantine is shared, not a per-call/Session lock');
    } finally { remoteWorkActive = false; await ctx.fiber.dispose(); }
  }));
}

/** Small external actors. They perform only resource admission, never MATLAB work. */
const pythonActor = String.raw`
import json, os, sys
from control_harness.matlab_resource import MatlabResource
def event(name, **data): print(json.dumps({'event': name, 'pid': os.getpid(), **data}), flush=True)
with MatlabResource(sys.argv[1], .01).acquire(sys.argv[2], on_queued=lambda item: event('queued', ticket=item['ticket'])) as lease:
    lease.mark_running()
    event('acquired', ticket=lease.owner['ticket'])
    sys.stdin.readline()
    lease.complete({'kind':'synchronous_adapter_return','producer':'cross-process-python-fixture'})
    event('completed')
`;

const nodeActor = `
import { createInterface } from 'node:readline';
import { Context } from ${JSON.stringify(builtUrl('vendor/cordis/lib/index.js'))};
import Local from ${JSON.stringify(builtUrl('packages/subprocess/subprocess-local/lib/index.js'))};
import { MatlabResource } from ${JSON.stringify(pathToFileURL(resolve(project, 'harness-plugin/src/matlab-resource.mjs')).href)};
const directory = process.argv[1];
const abort = new AbortController();
const event = (event, data={}) => console.log(JSON.stringify({event,pid:process.pid,...data}));
let release;
const released = new Promise(resolve => { release = resolve; });
const input = createInterface({input:process.stdin});
input.on('line', line => line === 'cancel' ? abort.abort(new Error('queued actor cancelled')) : release());
const ctx = new Context();
try {
  await ctx.plugin(Local);
  const resource = new MatlabResource({subprocess:ctx.subprocess,pythonPath:${JSON.stringify(python)},codeRoot:${JSON.stringify(project)},config:{directory,pollIntervalMs:10}});
  const lease = await resource.acquire('cross-process-node-fixture',abort.signal,item=>event('queued',{ticket:item.ticket}));
  await lease.start();
  event('acquired',{ticket:lease.ticket});
  await released;
  await lease.complete(${JSON.stringify(proof)});
  event('completed');
} catch(error) {
  if(abort.signal.aborted) event('cancelled'); else { console.error(error.stack); process.exitCode=1; }
} finally { input.close(); await ctx.fiber.dispose(); }
`;

function actor(language, directory, label) {
  const child = spawn(language === 'python' ? python : process.execPath,
    language === 'python' ? ['-u', '-c', pythonActor, directory, label] : ['--input-type=module', '-e', nodeActor, directory],
    { cwd: project, env: { ...process.env, PYTHONPATH: resolve(project, 'src') }, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  const events = [], waiters = new Set();
  let pending = '', stderr = '', exited = false;
  const flush = () => {
    for (const waiter of waiters) {
      const event = events.find(item => item.event === waiter.name);
      if (event) { waiters.delete(waiter); clearTimeout(waiter.timer); waiter.resolve(event); }
      else if (exited) { waiters.delete(waiter); clearTimeout(waiter.timer); waiter.reject(new Error(`Actor exited before ${waiter.name}: ${stderr}`)); }
    }
  };
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', chunk => { stderr += chunk; });
  child.stdout.setEncoding('utf8');
  child.stdout.on('data', chunk => {
    pending += chunk;
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      const line = pending.slice(0, newline).trim(); pending = pending.slice(newline + 1);
      if (line) events.push(JSON.parse(line));
    }
    flush();
  });
  const done = new Promise((resolveDone, reject) => { child.once('error', reject); child.once('close', code => { exited = true; flush(); resolveDone({ code, stderr }); }); });
  return { child, events, done, wait(name) {
    return new Promise((resolveWait, reject) => {
      const waiter = { name, resolve: resolveWait, reject };
      waiter.timer = setTimeout(() => { waiters.delete(waiter); reject(new Error(`Timed out waiting for ${language} actor ${name}: ${stderr}`)); }, 15000);
      waiters.add(waiter); flush();
    });
  }, send(line) { child.stdin.write(`${line}\n`); } };
}

test('independent Node Hosts and Python consumers share FIFO exclusion; queued cancellation removes only itself', { timeout: 30000 }, async () => fixture(async directory => {
  const actors = [];
  const start = (language, label) => { const value = actor(language, directory, label); actors.push(value); return value; };
  try {
    const holder = start('python', 'python-first-owner');
    const first = await holder.wait('acquired');
    const node = start('node');
    const second = await node.wait('queued');
    const pythonNext = start('python', 'python-next');
    const third = await pythonNext.wait('queued');
    const cancelled = start('node');
    const fourth = await cancelled.wait('queued');
    assert.ok(first.ticket < second.ticket && second.ticket < third.ticket && third.ticket < fourth.ticket);
    cancelled.send('cancel');
    await cancelled.wait('cancelled');
    assert.equal((await cancelled.done).code, 0);
    const queued = await state(directory);
    assert.deepEqual(queued.queue.map(item => item.ticket), [second.ticket, third.ticket]);
    assert.equal(node.events.some(item => item.event === 'acquired'), false);
    assert.equal(pythonNext.events.some(item => item.event === 'acquired'), false);
    holder.send('release');
    await holder.wait('completed');
    const nodeOwned = await node.wait('acquired');
    assert.equal(nodeOwned.ticket, second.ticket);
    assert.equal(pythonNext.events.some(item => item.event === 'acquired'), false, 'Python cannot overlap the Node Host lease');
    node.send('release');
    await node.wait('completed');
    const pythonOwned = await pythonNext.wait('acquired');
    assert.equal(pythonOwned.ticket, third.ticket);
    assert.equal(new Set([first.pid, nodeOwned.pid, pythonOwned.pid, fourth.pid]).size, 4, 'Actual independent processes took part');
    pythonNext.send('release');
    await pythonNext.wait('completed');
    for (const item of actors) assert.equal((await item.done).code, 0);
    const final = await state(directory);
    assert.equal(final.owner, null);
    assert.equal(final.quarantine, null);
    assert.deepEqual(final.queue, []);
  } finally {
    for (const item of actors) { if (item.child.exitCode === null) item.child.kill(); await item.done; }
  }
}));
