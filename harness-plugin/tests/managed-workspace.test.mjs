import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {randomUUID} from 'node:crypto';
import {lstat,mkdir,mkdtemp,readFile,readdir,realpath,rename,rmdir,unlink,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {installPlantWorkspace} from '../src/plant-workspace.mjs';
import {ensureSessionWorkspace,installManagedShellWorkspace} from '../src/session-workspace.mjs';
import {inspectHostSecurity} from '../src/windows-host-security.mjs';
import {preflightWorkspace,prepareShellWorkspaceGrant} from '../src/workspace-preflight.mjs';
import {createHost} from './history-host.mjs';

const project=resolve(import.meta.dirname,'../..');
const upstream=path=>import(pathToFileURL(join(project,'vendor/deepseek-harness',path,'lib/index.js')));
const config={codeRoot:project,pythonPath:join(project,'.venv/Scripts/python.exe')};
const powershell=(args,workspace)=>execFileSync('powershell.exe',['-NoProfile','-NonInteractive',...args],{
  encoding:'utf8',windowsHide:true, ...(workspace ? {env:{...process.env,CONTROL_TEST_LEAF:workspace}} : {})
});
const dacl=path=>powershell(['-Command',"[IO.DirectoryInfo]::new($env:CONTROL_TEST_LEAF).GetAccessControl().GetSecurityDescriptorSddlForm('Access')"],path).trim();

test('each new managed Session leaf prepares only itself before panel admission or direct source replay', {
  skip:process.platform!=='win32',timeout:90000
}, async()=>{
  const [{Context},sessions,domains,storage,json,registry,tools,executor,prompt,projections,commands,policy,subprocess,sandbox,attachments,llm]=
    await Promise.all(['vendor/cordis','packages/core/session','packages/storage/storage-domain','packages/storage/storage','packages/storage/storage-json',
      'packages/workspace/workspace','packages/core/tools','packages/core/tool-executor-native','packages/core/system-prompt','packages/session/session-projection',
      'packages/interaction/commands','packages/sandbox/sandbox-policy','packages/subprocess/subprocess-local','packages/sandbox/sandbox-local',
      'packages/attachment/attachment-local','packages/llm/llm'].map(upstream));
  await mkdir(join(project,'test-output'),{recursive:true});
  const evidenceDir=await mkdtemp(join(project,'test-output','managed-workspace-'));
  const fixture=JSON.parse(powershell(['-File',join(project,'scripts/new-workspace-acl-fixture.ps1'),'-Name','acl-plant-'+randomUUID(),'-ManagedWorkspace']));
  const parent=fixture.workspace, first=join(parent,"first case's [1]"), second=join(parent,'nested','second'),
    historical=join(parent,'resumed'), denied=join(parent,'denied'), shellLeaf=join(parent,'chat-shell');
  const ctx=new Context();
  const business=[];
  const preflight=options=>preflightWorkspace({ctx,config,...options});
  const agent=(cwd,agentPreset)=>{
    const id=randomUUID(),session=ctx.sessions.create(id,{meta:{cwd,agentPreset}});
    return {id,session,ctx:ctx.plugin(()=>{}).ctx,options:{}};
  };
  const references=[];
  try {
    await ctx.plugin(storage.default);
    ctx.storage.backend.register('json',new json.JsonStorageBackend(join(evidenceDir,'store')));
    const facility=new domains.DomainFacility(ctx,{backend:'json',routes:{}});
    ctx.storage.mount('domain',facility);ctx.provide('storageDomain',facility);ctx.provide('sessionPersistence',{list:async()=>[]});
    for(const p of [sessions.default,registry.default,tools.default,executor,prompt.default,projections.default,commands.default,
      subprocess.default,sandbox.default]) await ctx.plugin(p);
    await ctx.plugin(attachments.default,{dshHome:join(evidenceDir,'home')});
    await ctx.plugin(policy.default,{mode:'workspace-write'});
    const admissions=[];
    installPlantWorkspace(ctx,config,async options=>{
      const result=await preflight(options);
      admissions.push(result);
      return result;
    });
    const parentAcl=dacl(parent);
    await mkdir(first);await mkdir(denied);
    const siblingAcl=dacl(denied);
    assert.equal(inspectHostSecurity(first).right_errors.WRITE_OWNER,5);
    const panel=agent(first);
    const attach=()=>ensureSessionWorkspace(ctx,panel,{},preflight);
    const [attached,same]=await Promise.all([attach(),attach()]);
    assert.equal(attached.attached,true);assert.equal(same.workspaceId,attached.workspaceId);
    const a=attached.preflight;
    assert.deepEqual(a.host.workspace_preparation.added_leaf_rights,['WRITE_OWNER']);
    assert.equal(a.worker.integrity_sid,'S-1-16-4096');
    const leafAcl=dacl(first);
    const reused=(await attach()).preflight;
    assert.equal(reused.host.workspace_preparation,undefined);
    assert.equal(dacl(first),leafAcl);assert.equal(dacl(denied),siblingAcl);assert.equal(dacl(parent),parentAcl);

    await assert.rejects(lstat(second),{code:'ENOENT'});
    const chat=agent(second), att=await ctx.attachments.saveFile({name:'plant_numpy.py',data:Buffer.from('def step(x,u,d,dt,parameters): return [x[0]+dt*(u[0]+d[0])]')});
    chat.session.append('user/message',llm.createUserMessage({source:{kind:'user'},content:[{type:'file',attachment:att},
      {type:'text',text:'将附件转换成 Simulink 对象'}]}),{surfaceOp:'append'});
    const call=async(name,args,current=chat)=>(await ctx.tools.execute({agent:current,signal:AbortSignal.timeout(30000),calls:[
      {type:'tool_call',id:randomUUID(),name,arguments:JSON.stringify(args)}]})).results[0].result;
    const args={source:att.name,request:{parameters:{},sampleTime:.25,absTolerance:1e-12,
      trajectories:[{initialCondition:[2],controls:[[1],[0]],disturbances:[[0],[-1]]}]}};
    const replay=await call('control_replay_plant_source',args);
    assert.equal(replay.isError,false,replay.error?.message);
    const reference=join(second,replay.value.referenceFile);references.push(reference);
    assert.deepEqual(JSON.parse(await readFile(reference,'utf8')).trajectories[0].states,[[2],[2.25],[2]]);
    const b=admissions.find(p=>p.host.workspace_preparation);
    assert.deepEqual(b.host.workspace_preparation.added_leaf_rights,['WRITE_OWNER']);
    assert.equal(b.worker.integrity_sid,'S-1-16-4096');assert.equal(dacl(first),leafAcl);assert.equal(dacl(parent),parentAcl);

    // Read an old-style failure from native Session history before doing a
    // replay in the same chat. The original preflight had no preparation flag.
    const resumed=agent(historical), callId=randomUUID();
    resumed.session.append('user/message',llm.createUserMessage({source:{kind:'user'},content:[{type:'file',attachment:att},
      {type:'text',text:'把对象转成 Simulink 模型'}]}),{surfaceOp:'append'});
    let old;
    await assert.rejects(preflight({workspace:historical,policy:ctx.sandboxPolicy.resolve({session:resumed.session})}),
      error=>{old=error.diagnostic;return old?.operation==='check_WRITE_OWNER';});
    resumed.session.append('tool/call',{turn:1,step:1,callId,name:'control_replay_plant_source',arguments:'{}'});
    resumed.session.append('tool/result',{turn:1,step:1,message:llm.createToolResultMessage({callId,isError:true,
      content:[{type:'text',text:JSON.stringify({stage:'workspace_preflight',diagnostic:old})}]})},{surfaceOp:'append'});
    resumed.session.append('user/message',llm.createUserMessage({source:{kind:'user'},content:[{type:'text',text:'继续'}]}),{surfaceOp:'append'});
    const resumedReplay=await call('control_replay_plant_source',args,resumed);
    assert.equal(resumedReplay.isError,false,resumedReplay.error?.message);
    references.push(join(historical,resumedReplay.value.referenceFile));
    powershell(['-Command',[
      "$ErrorActionPreference='Stop'",'$dir=[IO.DirectoryInfo]::new($env:CONTROL_TEST_LEAF)','$acl=$dir.GetAccessControl()',
      "$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new([Security.Principal.WindowsIdentity]::GetCurrent().User,'TakeOwnership','Deny'))",
      '$dir.SetAccessControl($acl)'
    ].join('\n')],denied);
    const deniedAcl=dacl(denied);
    let deniedReceipt;
    await assert.rejects(ensureSessionWorkspace(ctx,agent(denied),{},preflight),e=>{
      deniedReceipt=e.diagnostic;
      return deniedReceipt?.workspace_preparation?.reason==='applicable_deny_ace' && deniedReceipt.workspace_recovery.can_prepare===false;
    });
    assert.equal(dacl(denied),deniedAcl);assert.deepEqual(await readdir(denied),[]);

    // A new native professional chat has a preset and cwd but no panel command
    // or plant conversion. Run the upstream pwsh tool with its real ACL worker.
    const [pwsh, pwshTool, shellEnv]=await Promise.all(['packages/shell/pwsh-sandbox',
      'packages/shell/tool-pwsh','packages/shell/shell-env'].map(upstream));
    ctx.provide('workingDirectory',{ensure:async a=>a.session.header.cwd});
    const shellAgent=agent(shellLeaf,'control-lightweight');
    let composed='standard';
    ctx.provide('agentPresets',{presetForSession:s=>s.header.agentPreset,composedPreset:()=>composed});
    await ctx.plugin(shellEnv,{dshHome:join(evidenceDir,'home')});
    await ctx.plugin(pwsh.default,{pwshPath:join(process.env.SystemRoot,'System32/WindowsPowerShell/v1.0/powershell.exe')});
    await ctx.plugin(pwshTool,{enableRunInBackground:false});
    installManagedShellWorkspace(ctx,config);
    const shellArgs={description:'Check this workspace',command:"Write-Output 'shell-ready'"};
    await mkdir(shellLeaf);
    const unprepared=await call('pwsh',shellArgs,shellAgent);
    assert.match(unprepared.error.message,/SetNamedSecurityInfoW failed \(Win32 5\): grantWrite/);
    assert.equal(inspectHostSecurity(shellLeaf).right_errors.WRITE_OWNER,5);
    composed='control-lightweight';
    for (const mode of ['read-only','danger-full-access']) {
      shellAgent.session.append('sandbox/mode',{mode});
      const result=await call('pwsh',shellArgs,shellAgent);
      assert.equal(result.isError,false,result.error?.message);
      assert.equal(result.value.sandbox.mode,mode);
      assert.equal(inspectHostSecurity(shellLeaf).right_errors.WRITE_OWNER,5);
    }
    shellAgent.session.append('sandbox/mode',{mode:'workspace-write'});
    await prepareShellWorkspaceGrant({sandbox:{confine:async()=>({argv:['custom-runner']})}},config,
      ctx.sandboxPolicy.resolve({session:shellAgent.session}),AbortSignal.timeout(3000));
    assert.equal(inspectHostSecurity(shellLeaf).right_errors.WRITE_OWNER,5);
    for (let i=0;i<2;i++) {
      const result=await call('pwsh',shellArgs,shellAgent);
      assert.equal(result.isError,false,result.error?.message);
      assert.equal(result.value.sandbox.mode,'workspace-write');
      assert.match(result.value.stdout.text,/shell-ready/);
    }
    assert.equal(inspectHostSecurity(shellLeaf).effective_rights.WRITE_OWNER,true);
    const blockedShell=await call('pwsh',shellArgs,agent(denied,'control-lightweight'));
    assert.ok(blockedShell.isError);assert.match(blockedShell.error.message,/applicable_deny_ace/);
    assert.equal(dacl(denied),deniedAcl);assert.equal(dacl(parent),parentAcl);
    // Each business entry opens a new workspace without ever opening the panel.
    const stages=[];
    for(const command of ['control-prepare-review','control']) {
      const workspace=join(parent,command),owner=randomUUID();
      await mkdir(workspace);business.push(workspace);
      const before=inspectHostSecurity(workspace);
      assert.equal(before.right_errors.WRITE_OWNER,5);
      const executionPolicy={mode:'workspace-write',workspaceRoot:workspace,sessionId:owner};
      for(const other of [panel,{id:owner,session:agent(workspace).session}]) {
        await assert.rejects(preflight({workspace,agent:other,policy:executionPolicy}),e=>e.diagnostic?.operation==='check_WRITE_OWNER');
        assert.equal(inspectHostSecurity(workspace).right_errors.WRITE_OWNER,5);
      }
      const service=await createHost(workspace,owner,{confirmFixturePlans:false,policy:()=>executionPolicy,sandbox:ctx.sandbox});
      try {
        const request=command==='control' ? {agent:'model',operation:'inspect_workspace',workspace}
          : {agent:'controller',operation:'synthesize_mpc',workspace,inputs:{description:'二阶弹簧系统'},config:{
            parameters:{mass:1,damping:.5,stiffness:2,sample_time:.1,state_bounds:[[-2,2],[-3,3]],input_bounds:[[-1,1]],output_bounds:[[-2,2]]}}};
        const result=await service.command(command,request);
        assert.equal(result.kind,'success',result.text);
        const value=JSON.parse(result.text);
        if(command==='control') {
          const end=await service.ctx.jobs.wait(value.jobId,30000,owner);
          assert.equal(end.status,'completed',end.detail);
        } else assert.ok(value.card_id,result.text);
        const after=inspectHostSecurity(workspace);
        assert.equal(after.effective_rights.WRITE_OWNER,true);
        stages.push({command,workspace,before:before.effective_rights,after:after.effective_rights,
          card_id:value.card_id,job_id:value.jobId});
      } finally { await service.close(); }
    }
    await writeFile(join(evidenceDir,'evidence.json'),JSON.stringify({first:a,reused,second:b,replay:replay.value,resumed:{old,replay:resumedReplay.value},denied:deniedReceipt,
      business:stages,parent_acl_unchanged:dacl(parent)===parentAcl,other_leaf_acl_unchanged:dacl(first)===leafAcl},null,2));
    console.log('Managed Session workspace verification:',join(evidenceDir,'evidence.json'));
  } finally {
    await ctx.fiber.dispose();
    for(const reference of references) await unlink(reference);
    // Retain the two new native job/review ledgers with the evidence.
    assert.equal(await realpath(evidenceDir),evidenceDir);
    for(const path of business) {
      assert.ok(path===join(parent,'control') || path===join(parent,'control-prepare-review'));
      assert.equal(await realpath(path),path);
      await rename(path,join(evidenceDir,path.slice(parent.length+1)));
    }
    for(const leaf of [second,join(parent,'nested'),first,historical,denied,shellLeaf,parent]) {
      try { await rmdir(leaf); } catch(e) { if(e.code!=='ENOENT') throw e; }
    }
  }
});
