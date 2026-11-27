import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash, randomUUID} from 'node:crypto';
import {mkdir, mkdtemp, readFile, writeFile} from 'node:fs/promises';
import {join, resolve} from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {defineTool} from '@deepseek-ai/dsh-tools';
import {installPlantWorkspace} from '../src/plant-workspace.mjs';
import {preflightWorkspace} from '../src/workspace-preflight.mjs';

const project=resolve(import.meta.dirname,'../..');
const importBuilt=path=>import(pathToFileURL(join(project,'vendor/deepseek-harness',path,'lib/index.js')));
const [{Context},{Session},llm,prompt,tools,executor,projection,commands,policy,subprocess,sandbox,fs,wd,toolFs,attachments]=
  await Promise.all(['vendor/cordis','packages/core/session','packages/llm/llm','packages/core/system-prompt',
    'packages/core/tools','packages/core/tool-executor-native','packages/session/session-projection','packages/interaction/commands',
    'packages/sandbox/sandbox-policy','packages/subprocess/subprocess-local','packages/sandbox/sandbox-local',
    'packages/fs/fs-local','packages/session/working-directory','packages/fs/tool-fs','packages/attachment/attachment-local'].map(importBuilt));

test('a native read in this chat binds a confined workspace-source replay and its verified delivery',
  {skip:process.platform!=='win32'}, async()=>{
    const testRoot=join(project,'test-output');
    await mkdir(testRoot,{recursive:true});
    const root=await mkdtemp(join(testRoot,'native-workspace-source-'));
    const cwd=join(root,'workspace');
    await mkdir(cwd);
    const prepared=JSON.parse(execFileSync('powershell.exe',['-NoProfile','-NonInteractive','-File',
      join(project,'scripts/prepare-workspace.ps1'),'-Workspace',cwd,'-Prepare'],{encoding:'utf8',windowsHide:true}));
    assert.equal(prepared.prepared,true);

    const name='plant_numpy.py', sourcePath=join(cwd,name), unreadPath=join(cwd,'unread.py'), outsidePath=join(root,name);
    const source=[
      'from pathlib import Path','import runpy',
      'inspect_token = runpy.run_path('+JSON.stringify(join(project,'src/control_harness/workspace_probe.py'))+')["_windows_security"]',
      'assert inspect_token(Path.cwd())["integrity_sid"] == "S-1-16-4096"',
      'def step(x,u,d,dt,parameters):',
      '    return [x[0]+dt*(parameters["gain"]*u[0]-d[0])]',
    ].join('\n');
    await Promise.all([writeFile(sourcePath,source),writeFile(unreadPath,source),writeFile(outsidePath,source)]);
    const config={codeRoot:project,pythonPath:join(project,'.venv/Scripts/python.exe')};
    const ctx=new Context(), restored=new Context();
    const request={absTolerance:1e-12,sampleTime:0.25,parameters:{gain:2},
      trajectories:[{initialCondition:[4],controls:[[3],[2]],disturbances:[[1],[-4]]}]};
    const sessionId=randomUUID(), session=Session.create(sessionId,undefined,{...Session.create(sessionId).header,cwd});
    session.append('turn/start',{turn:1});
    const agentOf=context=>({id:sessionId,session,ctx:context.plugin(()=>{}).ctx,options:{}});
    const nativeCall=async(context,agent,name,args={})=>(await context.tools.execute({agent,log:{turn:1,step:1},
      signal:AbortSignal.timeout(90000),calls:[{type:'tool_call',id:randomUUID(),name,arguments:JSON.stringify(args)}]})).results[0].result;

    try {
      for (const plugin of [prompt.default,tools.default,executor,projection.default,commands.default,subprocess.default,sandbox.default]) {
        await ctx.plugin(plugin);
      }
      await ctx.plugin(policy.default,{mode:'workspace-write',workspaceRoot:cwd});
      await ctx.plugin(fs.default,{cwd});
      await ctx.plugin(wd.default,{defaultDirectory:cwd});
      await ctx.plugin(attachments.default,{dshHome:join(root,'private-home')});
      await ctx.plugin(toolFs,{read:true,readImage:false,write:true,edit:false});
      const guard=installPlantWorkspace(ctx,config,opts=>preflightWorkspace({ctx,config,...opts}));
      const agent=agentOf(ctx);
      session.append('user/message',llm.createUserMessage({source:{kind:'user'},
        content:[{type:'text',text:'读取本工作区的 plant_numpy.py，把 Python 被控对象转换成 Simulink。'}]}),{surfaceOp:'append'});
      const replay=source=>nativeCall(ctx,agent,'control_replay_plant_source',{source,request});
      assert.equal((await replay(name)).isError,true,'A name in user text is not a successful source read.');
      const observed=await nativeCall(ctx,agent,'read',{file_path:sourcePath});
      assert.equal(observed.isError,false,observed.error?.message);
      assert.equal(observed.meta.path,sourcePath);
      // Reading a script does not turn every independent matrix/reference pair into a source conversion.
      const measured={sampleTime:0.25,absTolerance:1e-12,trajectories:[{initialCondition:[4],inputs:[[3,1]],states:[[4],[5.25]]}]};
      await guard.verifyReference(agent,'measured-reference.json',measured);
      await assert.rejects(guard.verifyReference(agent,'measured-reference.json',measured,name),/回放/);
      const notes=join(cwd,'接口说明.md');
      const write=await nativeCall(ctx,agent,'write',{file_path:notes,content:'状态与输入采用 SI 单位。'});
      assert.equal(write.isError,false,write.error?.message);
      assert.equal(await readFile(notes,'utf8'),'状态与输入采用 SI 单位。');
      // Even an actual read of a same-name file outside the chat root grants no replay authority.
      assert.equal((await nativeCall(ctx,agent,'read',{file_path:outsidePath})).isError,false);
      assert.equal((await replay(outsidePath)).isError,true);
      assert.equal((await replay(unreadPath)).isError,true);

      const sourceSha=createHash('sha256').update(source).digest('hex');
      let proof,reference;
      for (const selector of [name,sourcePath]) {
        const result=await replay(selector);
        assert.equal(result.isError,false,result.error?.message);
        proof=JSON.parse(result.content[0].text);
        assert.equal(proof.source_kind,'workspace');
        assert.equal(proof.source,name);
        assert.equal(proof.source_sha256,sourceSha);
        assert.equal(proof.sandbox_mode,'workspace-write');
        reference=JSON.parse(await readFile(join(cwd,proof.referenceFile),'utf8'));
        assert.equal(reference.source_sha256,sourceSha);
        assert.deepEqual(reference.trajectories[0].states,[[4],[5.25],[7.25]]);
        await guard.verifyReference(agent,proof.referenceFile,reference);
      }

      for (const plugin of [prompt.default,tools.default,executor,projection.default,commands.default]) await restored.plugin(plugin);
      const coldGuard=installPlantWorkspace(restored,config,()=>assert.fail('Using a recorded proof must not run a new worker.'));
      const coldAgent=agentOf(restored);
      await coldGuard.verifyReference(coldAgent,proof.referenceFile,reference);
      await coldGuard.verifyReference(coldAgent,proof.referenceFile,reference,name);
      await assert.rejects(coldGuard.verifyReference(coldAgent,proof.referenceFile,reference,'unread.py'));
      await assert.rejects(coldGuard.verifyReference(coldAgent,'invented.json',reference));
      await assert.rejects(coldGuard.verifyReference(coldAgent,proof.referenceFile,{...reference,parameters:{gain:3}}));
      await writeFile(sourcePath,source+'\n# The workspace source changed.\n');
      await assert.rejects(guard.verifyReference(agent,proof.referenceFile,reference));
      await assert.rejects(coldGuard.verifyReference(coldAgent,proof.referenceFile,reference));
      await writeFile(sourcePath,source);

      // This is a receipt-boundary test. Numerical model building is exercised separately against R2024b.
      const completion={status:'simulated',model:'WorkspacePlant',path:join(cwd,'WorkspacePlant.slx'),bytes:1024,
        sha256:'b'.repeat(64),session_verified:true,file_exists:true,comparison:{compared:true},verified:['save','reload','simulate','compare']};
      const output={schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]};
      ctx.tools.register(defineTool({name:'control_build_zoh_plant',description:'Finish the verified test source',parameters:{},output,
        execute:async()=>{await guard.verifyReference(agent,proof.referenceFile,reference);return completion;}}));
      const completed=await nativeCall(ctx,agent,'control_build_zoh_plant');
      assert.equal(completed.isError,false,completed.error?.message);
      const raw='mcp__matlab__evaluate_matlab_code';
      restored.tools.register(defineTool({name:raw,description:'Observe the permitted next MATLAB operation',parameters:{},output,
        execute:()=>({dispatched:true})}));
      const next=await nativeCall(restored,coldAgent,raw);
      assert.deepEqual(next.value,{dispatched:true},
        'A workspace-source replay/build in ordinary chat must leave other independently authorized tools available.');
      const evidence=join(root,'workspace-source.json');
      await writeFile(evidence,JSON.stringify({workspace:cwd,session:sessionId,read:observed.meta.path,sourceSha,
        replay:proof,states:reference.trajectories[0].states,cold_proof_verified:true,
        changed_source_rejected:true,unread_and_outside_rejected:true,ordinary_chat_continued:next.value.dispatched},null,2));
      console.log('原生工作区源码回放及来源校验：',evidence);
    } finally {
      await restored.fiber.dispose();
      await ctx.fiber.dispose();
    }
});
