import assert from 'node:assert/strict';
import test from 'node:test';
import {randomUUID} from 'node:crypto';
import {join,resolve} from 'node:path';
import {mkdtemp,mkdir,readFile,writeFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {defineTool} from '@deepseek-ai/dsh-tools';
import {installPlantWorkspace} from '../src/plant-workspace.mjs';
import {WorkspacePreflightError} from '../src/workspace-policy.mjs';
import {preflightWorkspace} from '../src/workspace-preflight.mjs';

const project=resolve(import.meta.dirname,'../..');
const built=path=>import(pathToFileURL(join(project,'vendor/deepseek-harness',path,'lib/index.js')));
const [{Context},{Session},prompt,tools,executor,projection,commands,policy,llm]=await Promise.all([
  'vendor/cordis','packages/core/session','packages/core/system-prompt','packages/core/tools',
  'packages/core/tool-executor-native','packages/session/session-projection','packages/interaction/commands',
  'packages/sandbox/sandbox-policy','packages/llm/llm',
].map(built));
const output={schema:{type:'object',additionalProperties:true},render:()=>[{type:'text',text:'{}'}]};

test('verified delivery stops native siblings, both live and after replay, until the next genuine user request',async()=>{
  const cwd=join(project,'test-output'), original=new Context(), restored=new Context();
  const raw='mcp__matlab__evaluate_matlab_code', build='control_build_zoh_plant', recheck='control_recheck_plant_workspace';
  const result={status:'simulated',model:'TestPlant',path:join(cwd,'TestPlant.slx'),sha256:'b'.repeat(64),bytes:9234,
    verified:['validation','save','reload','simulate','compare'],comparison:{compared:true},
    file_exists:true,session_verified:true};
  let buildBody=()=>result, rawCalls=0, checks=0;
  const makeCall=name=>({type:'tool_call',id:randomUUID(),name,arguments:'{}'});
  async function setup(ctx) {
    for (const p of [prompt.default,tools.default,executor,projection.default,commands.default]) await ctx.plugin(p);
    await ctx.plugin(policy.default,{mode:'workspace-write',workspaceRoot:cwd});
    installPlantWorkspace(ctx,{codeRoot:project},async()=>{checks++;return {ok:true,workspace:cwd,sandbox_mode:'workspace-write'};});
    for (const name of [raw,build,'read','pwsh']) ctx.tools.register(defineTool({
      name,description:'Count the native tool body',parameters:{},output:{...output,render:(_,v)=>[{type:'text',text:JSON.stringify(v)}]},
      execute:async()=>{if(name===build)return buildBody(); if(name===raw)rawCalls++;return {};},
    }));
  }
  const user=(a,text,file=false)=>a.session.append('user/message',llm.createUserMessage({source:{kind:'user'},
    content:[{type:'text',text},...(file?[{type:'file',attachment:{name:'plant.py',attachmentId:'sha256:'+'a'.repeat(64),bytes:17}}]:[])]}),{surfaceOp:'append'});
  const chat=()=>{const id=randomUUID(),session=Session.create(id,undefined,{...Session.create(id).header,cwd});
    session.append('turn/start',{turn:1});return {id,session,ctx:original.plugin(()=>{}).ctx,options:{}};};
  const call=async(ctx,a,names,logged=true,signal=AbortSignal.timeout(5000))=>(await ctx.tools.execute({agent:ctx===original?a:{...a,ctx:ctx.plugin(()=>{}).ctx},signal,
    ...(logged?{log:{turn:1,step:1}}:{}),calls:names.map(makeCall)})).results.map(r=>r.result);
  try {
    await setup(original);await setup(restored);
    for(const logged of [true,false]) {
      const a=chat();user(a,'将我上传的 Python 转成 Simulink 被控对象',true);
      const [done,blocked]=await call(original,a,[build,raw],logged);
      assert.equal(done.isError,false);assert.equal(blocked.error.info.code,'PLANT_DELIVERY_COMPLETE');assert.equal(rawCalls,0);
      assert.deepEqual(JSON.parse(blocked.error.message).verified_delivery,{model:result.model,path:result.path,sha256:result.sha256,bytes:result.bytes});
      a.session.append('turn/end',{turn:1,reason:{kind:'completed'}});
      a.session.append('turn/start',{turn:2});
      policy.setSandboxMode(a.session,'danger-full-access');
      a.session.append('user/message',llm.createUserMessage({source:{kind:'runtime-context'},
        content:[{type:'text',text:'Runtime sandbox policy changed.'}]}),{surfaceOp:'append'});
      const view=logged?restored:original;
      for(const name of [build,raw,recheck,'pwsh']) assert.equal((await call(view,a,[name],logged))[0].error.info.code,'PLANT_DELIVERY_COMPLETE');
      assert.equal((await call(view,a,['read'],logged))[0].isError,false);assert.equal(checks,0);
      user(a,'请继续调试刚才的模型，检查块的实际属性。');
      const debug=(await call(view,a,[raw],logged))[0];
      assert.equal(debug.isError,false,JSON.stringify({logged,error:debug.error}));
      assert.equal(rawCalls,1);rawCalls=0;
    }
    const ended=chat();user(ended,'将 Python 转换成 Simulink',true);
    await call(original,ended,[build],false);
    assert.equal((await call(original,chat(),[raw]))[0].isError,false);
    assert.equal((await call(original,ended,[raw],false))[0].error.info.code,'PLANT_DELIVERY_COMPLETE');
    assert.equal((await original.commands.execute(ended,'/control-end-plant-task',[],new AbortController().signal)).result.kind,'success');
    assert.equal((await call(original,ended,[raw],false))[0].isError,false);
    // In-flight success after a new user message cannot capture that new request.
    let settle,notify;
    buildBody=()=>new Promise(resolve=>{settle=resolve;notify();});
    for(const logged of [true,false]) {
      const late=chat();user(late,'把 Python 上传转成 Simulink',true);
      const started=new Promise(r=>{notify=r;});
      const finishing=call(original,late,[build],logged);
      await started;
      user(late,'请检查模型默认输入值。');
      settle(result);await finishing;
      assert.equal((await call(logged?restored:original,late,[raw],logged))[0].isError,false);
    }
    // Body return can resemble a verified delivery even though native cancellation
    // supersedes that return. Only the final native result can release the task.
    const cancelled=chat();user(cancelled,'将 Python 转换成 Simulink',true);
    const cancel=new AbortController();
    const running=new Promise(r=>{notify=r;});
    const interrupted=call(original,cancelled,[build],true,cancel.signal);
    await running;cancel.abort();settle(result);
    assert.equal((await interrupted)[0].isError,true);
    assert.equal((await call(original,cancelled,[raw]))[0].error.info.code,'PLANT_CONTROLLED_TOOL_REQUIRED');
    // A successful-looking but incomplete body does not claim verified delivery.
    const incomplete=chat();user(incomplete,'把 Python 上传转成 Simulink',true);
    buildBody=()=>({...result,session_verified:false});
    await call(original,incomplete,[build]);
    assert.equal((await call(original,incomplete,[raw]))[0].error.info.code,'PLANT_CONTROLLED_TOOL_REQUIRED');
  } finally { await restored.fiber.dispose();await original.fiber.dispose(); }
});

test('native attachment conversion gates the first side effect, subsequent turns, cold replay and post-repair dispatch',async()=>{
  const ctx=new Context();
  await mkdir(join(project,'test-output'),{recursive:true});
  const cwd=await mkdtemp(join(project,'test-output','workspace-admission-'));
  const counts=new Map(), config={codeRoot:project};
  let checks=0, healthy=false;
  try {
    for (const plugin of [prompt.default,tools.default,executor,projection.default,commands.default]) await ctx.plugin(plugin);
    await ctx.plugin(policy.default,{mode:'workspace-write',workspaceRoot:cwd});
    installPlantWorkspace(ctx,config,async({workspace,policy})=>{
      checks++;
      if (!healthy) throw new WorkspacePreflightError({code:'WORKSPACE_PREFLIGHT_FAILED',operation:'check_WRITE_OWNER',
        os_code:5,path:workspace,detail:'WRITE_OWNER',workspace_recovery:{workspace,missing_rights:['WRITE_OWNER']}});
      return {ok:true,workspace,sandbox_mode:policy.mode};
    });
    for (const name of ['pwsh','workflow','mcp__matlab__evaluate_matlab_code','read','write','working_directory']) {
      ctx.tools.register(defineTool({name,description:'Test body',parameters:name==='write'
        ? {file_path:{type:'string'},content:{type:'string'}} : {},output,execute:()=>{counts.set(name,1+(counts.get(name)??0));return {};} }));
    }
    const chat=()=>{ const id=randomUUID();return {id,session:Session.create(id,undefined,{...Session.create(id).header,cwd}),
      ctx:ctx.plugin(()=>{}).ctx,options:{}}; };
    const first=chat(), other=chat();
    const user=(agent,content)=>agent.session.append('user/message',llm.createUserMessage({source:{kind:'user'},content}),{surfaceOp:'append'});
    const call=async(agent,name,args={})=>{
      const id=randomUUID();
      const batch=await ctx.tools.execute({agent,signal:AbortSignal.timeout(5000),
        calls:[{id,type:'tool_call',name,arguments:JSON.stringify(args)}]});
      return batch.results[0].result;
    };
    const source={type:'file',attachment:{attachmentId:'sha256:'+'a'.repeat(64),name:'plant_numpy.py',bytes:128}};
    user(first,[source,{type:'text',text:'这是源文件。'}]);
    assert.equal((await call(first,'pwsh')).isError,false);
    user(first,[{type:'text',text:'把附件转成 Simulink 被控对象，交付 slx'}]);
    assert.equal((await call(first,'read')).isError,false);
    const initial=await call(first,'pwsh');
    assert.equal(initial.error.info.code,'PLANT_WORKSPACE_PREFLIGHT_REQUIRED');
    const issue=JSON.parse(initial.error.message);
    assert.equal(issue.diagnostic.operation,'check_WRITE_OWNER');
    assert.equal(issue.stage,'workspace_preflight');
    assert.equal(issue.preflight_observation,'fresh');
    assert.match((await ctx.systemPrompt.assemble({agent:first})).contexts.find(c=>c.name==='control:uploaded-plant-workspace').text,/停止工具重试/);
    assert.equal(checks,1);
    user(first,[{type:'text',text:'请继续处理'}]);
    for (const name of ['pwsh','workflow','write','mcp__matlab__evaluate_matlab_code']) {
      const denied=await call(first,name);
      assert.equal(denied.error.info.code,'PLANT_WORKSPACE_PREFLIGHT_REQUIRED');
      assert.equal(JSON.parse(denied.error.message).preflight_observation,'previous_failure');
    }
    assert.equal(checks,1);
    assert.equal((await call(other,'workflow')).isError,false);
    assert.equal((await call(first,'control_recheck_plant_workspace')).isError,true);
    healthy=true;
    assert.equal((await call(first,'pwsh')).isError,true);
    assert.equal((await call(first,'control_recheck_plant_workspace')).isError,false);
    for (const name of ['pwsh','workflow','mcp__matlab__evaluate_matlab_code']) {
      assert.equal((await call(first,name)).error.info.code,'PLANT_CONTROLLED_TOOL_REQUIRED');
    }
    assert.equal((await call(first,'write',{file_path:join(cwd,'request.json'),content:'{"sampleTime":0.1}'})).isError,false);
    assert.equal((await call(first,'write',{file_path:join(cwd,'manual.slx'),content:'{}'})).error.info.code,'PLANT_CONTROLLED_TOOL_REQUIRED');
    assert.equal(counts.get('pwsh'),1);
    // Restore the original failed agent loop's contracted tool result (the
    // native executor alone intentionally does not persist agent-loop events).
    const cold=chat();
    user(cold,[source,{type:'text',text:'创建 Simulink 被控对象'}]);
    const callId=randomUUID();
    cold.session.append('tool/call',{turn:1,step:1,callId,name:'control_build_zoh_plant',arguments:'{}'});
    cold.session.append('tool/result',{turn:1,step:1,message:llm.createToolResultMessage({callId,
      isError:true,content:[{type:'text',text:'Error: '+initial.error.message}]})},{surfaceOp:'append'});
    assert.equal((await call(cold,'pwsh')).error.info.code,'PLANT_WORKSPACE_PREFLIGHT_REQUIRED');
    assert.equal((await ctx.commands.execute(first,'/control-end-plant-task',[],new AbortController().signal)).result.kind,'success');
    assert.equal((await call(first,'pwsh')).isError,false);
  } finally { await ctx.fiber.dispose(); }
});

test('real ACL failure, broader Session mode, fresh rechecks, leaf repair and low-integrity source replay in the same Session',{skip:process.platform!=='win32'},async()=>{
  const ctx=new Context();
  await mkdir(join(project,'test-output'),{recursive:true});
  const root=await mkdtemp(join(project,'test-output','native-source-replay-'));
  const ps=(file,args)=>JSON.parse(execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-File',join(project,'scripts',file),...args],
    {windowsHide:true,encoding:'utf8'}));
  const fixture=ps('new-workspace-acl-fixture.ps1',['-Name','acl-plant-'+randomUUID()]);
  const cwd=fixture.workspace;
  const [subprocess,sandbox,attachments]=await Promise.all(['packages/subprocess/subprocess-local',
    'packages/sandbox/sandbox-local','packages/attachment/attachment-local'].map(built));
  const config={codeRoot:project,pythonPath:join(project,'.venv/Scripts/python.exe')};
  try {
    for (const plugin of [prompt.default,tools.default,executor,projection.default,commands.default,subprocess.default,sandbox.default]) await ctx.plugin(plugin);
    await ctx.plugin(policy.default,{mode:'workspace-write',workspaceRoot:cwd});
    await ctx.plugin(attachments.default,{dshHome:join(root,'private-home')});
    const guard=installPlantWorkspace(ctx,config,opts=>preflightWorkspace({ctx,config,...opts}));
    const source=await ctx.attachments.saveFile({name:'user_plant.py',data:Buffer.from([
      'from pathlib import Path','import runpy',
      'inspect_token = runpy.run_path('+JSON.stringify(join(project,'src/control_harness/workspace_probe.py'))+')["_windows_security"]',
      'assert inspect_token(Path.cwd())["integrity_sid"] == "S-1-16-4096", "source replay must inherit the confined low-integrity token"',
      'def step(x,u,d,dt,parameters):','    return [x[0]+dt*(parameters["gain"]*u[0]-d[0])]',
    ].join('\n'))});
    const id=randomUUID(), session=Session.create(id,undefined,{...Session.create(id).header,cwd});
    const agent={id,session,ctx:ctx.plugin(()=>{}).ctx,options:{}};
    session.append('user/message',llm.createUserMessage({source:{kind:'user'},content:[{type:'file',attachment:source},
      {type:'text',text:'把附件转成 Simulink 被控对象'}]}),{surfaceOp:'append'});
    const request={absTolerance:1e-12,sampleTime:0.25,parameters:{gain:2},
      trajectories:[{initialCondition:[4],controls:[[3],[2]],disturbances:[[1],[-4]]}]};
    const call=async (args,name='control_replay_plant_source')=>(await ctx.tools.execute({agent,log:{turn:1,step:1},signal:AbortSignal.timeout(90000),
      calls:[{type:'tool_call',id:randomUUID(),name,arguments:JSON.stringify(args)}]})).results[0].result;
    const recheck=()=>call({},'control_recheck_plant_workspace');
    const first=JSON.parse((await recheck()).error.message);
    assert.equal(first.diagnostic.operation,'check_WRITE_OWNER');
    assert.equal(first.diagnostic.session_sandbox_mode,'workspace-write');
    policy.setSandboxMode(session,'danger-full-access');
    const second=JSON.parse((await recheck()).error.message);
    assert.equal(second.preflight_observation,'fresh');
    assert.equal(second.diagnostic.session_sandbox_mode,'danger-full-access');
    assert.equal(second.diagnostic.sandbox_mode,'workspace-write');
    assert.equal(second.diagnostic.operation,'check_WRITE_OWNER');
    const repair=ps('prepare-workspace.ps1',['-Workspace',cwd,'-Prepare']);
    assert.deepEqual(repair.added_leaf_rights,['WRITE_OWNER']);
    const recovered=(await recheck()).value;
    assert.equal(recovered.status,'verified');
    assert.equal(recovered.session_sandbox_mode,'danger-full-access');
    assert.equal(recovered.sandbox_mode,'workspace-write');
    assert.ok((await call({source:'other_chat.py',request})).isError);
    const success=await call({source:source.name,request});
    assert.equal(success.isError,false,success.error?.message);
    const proof=JSON.parse(success.content[0].text), ref=JSON.parse(await readFile(join(cwd,proof.referenceFile),'utf8'));
    assert.equal(proof.samples,3);
    assert.equal(proof.sandbox_mode,'workspace-write');
    assert.deepEqual(ref.trajectories[0].states,[[4],[5.25],[7.25]]);
    assert.equal('sha256:'+proof.source_sha256,source.attachmentId);
    await guard.verifyReference(agent,proof.referenceFile,ref);
    await assert.rejects(guard.verifyReference(agent,'invented.json',ref),/回放/);
    await assert.rejects(guard.verifyReference(agent,proof.referenceFile,{...ref,parameters:{gain:3}}),/回放/);
    policy.setSandboxMode(session,'read-only');
    const readOnly=JSON.parse((await recheck()).error.message);
    assert.equal(readOnly.diagnostic.sandbox_mode,'read-only');
    assert.equal(readOnly.diagnostic.operation,'authorize_workspace');
    assert.equal((await call({source:source.name,request})).isError,true);
    const reopened=new Context();
    try {
      for (const plugin of [prompt.default,tools.default,projection.default,commands.default]) await reopened.plugin(plugin);
      const coldGuard=installPlantWorkspace(reopened,config,()=>assert.fail('Verifying the original source receipt must not start a process'));
      await coldGuard.verifyReference(agent,proof.referenceFile,ref);
      // A newly uploaded file becomes the only source eligible for the next conversion.
      const replacement=await ctx.attachments.saveFile({name:'next.py',data:Buffer.from('def step(x,u,d,dt,parameters): return x')});
      session.append('user/message',llm.createUserMessage({source:{kind:'user'},content:[{type:'file',attachment:replacement},
        {type:'text',text:'把这个也转成 Simulink'}]}),{surfaceOp:'append'});
      await assert.rejects(coldGuard.verifyReference(agent,proof.referenceFile,ref),/回放/);
    } finally { await reopened.fiber.dispose(); }
    await writeFile(join(root,'evidence.json'),JSON.stringify({session:id,workspace:cwd,fixture,first,second,recovered,
      repaired_rights:repair.added_leaf_rights,readOnly,source:source.name,proof,states:ref.trajectories[0].states},null,2));
    console.log('原生上传、受限 Python 回放及来源校验：',join(root,'evidence.json'));
  } finally { await ctx.fiber.dispose(); }
});
