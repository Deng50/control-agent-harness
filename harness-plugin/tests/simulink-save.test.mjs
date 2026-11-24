import test from 'node:test';
import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, stat, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { saveSessionSlx } from '../src/simulink-save.mjs';

const project=resolve(import.meta.dirname,'../..');
const output=join(project,'test-output/simulink-save');
const registered='mcp__matlab__evaluate_matlab_code';
const fixturePath=join(project,'matlab/fixtures/f01_spring.slx');

async function fixture(behavior=async f=>{
  await copyFile(fixturePath,f.target);
  return {status:'saved',model:f.model,path:f.target,bytes:(await stat(f.target)).size};
}) {
  await mkdir(output,{recursive:true});
  const workspace=await mkdtemp(join(output,"会话 A's "));
  const model='f01_spring', target=join(workspace,model+'.slx'), agent={id:'a',session:{id:'a',header:{cwd:workspace}}};
  const policy={mode:'workspace-write',workspaceRoot:workspace}, abort=new AbortController();
  const config={codeRoot:project,matlabResource:{directory:workspace,identityFile:'fixture'}};
  const seen=[];
  const ctx={sandboxPolicy:{resolve:({session})=>{assert.equal(session,agent.session);return policy;}},
    tools:{definitions:()=>[{name:registered}],execute:async request=>{
      seen.push('native');
      assert.equal(request.agent,agent);assert.equal(request.signal,abort.signal);
      assert.equal(request.calls.length,1);
      const call=request.calls[0];
      assert.equal(call.name,registered);
      const {code,project_path}=JSON.parse(call.arguments);
      assert.equal(project_path,join(project,'matlab'));
      assert.ok(code.includes(target.replaceAll("'","''")));
      assert.ok(code.includes("which('control_harness_save_slx')"));
      assert.doesNotMatch(code,/\bsim\(|\bset_param\(|\bcd\(/);
      const marker='CONTROL_HARNESS_SLX_'+call.id.slice(4)+':';
      const record=await behavior({model,target});
      return {results:[{result:{isError:false,value:{content:[{type:'text',text:marker+JSON.stringify(record)}]}}}]};
    }}};
  const preflight=async options=>{
    seen.push('preflight');assert.equal(options.workspace,workspace);assert.equal(options.policy,policy);assert.equal(options.agent,agent);
  };
  return {ctx,config,seen,agent,workspace,target,policy,abort,
    save:(args={model})=>saveSessionSlx(ctx,config,args,{agent,signal:abort.signal},preflight)};
}

test('save uses one fixed native call in caller Session, quotes workspace and returns verified absolute SLX path',async()=>{
  const f=await fixture(), receipt=await f.save();
  assert.deepEqual(f.seen,['preflight','native']);
  assert.equal(receipt.path,f.target);assert.equal(receipt.simulation,'not_run');
  assert.equal(receipt.status,'saved');assert.equal(receipt.bytes,(await stat(f.target)).size);
});

test('save rejects alternate workspace, MATLAB code, denied writes and absent overlay before dispatch',async()=>{
  const f=await fixture();
  for (const args of [{model:'../x'},{model:"m');sim('m"},{model:'NUL'},{model:'m',workspace:project}]) {
    await assert.rejects(f.save(args),/模型名/);
  }
  f.policy.mode='read-only'; await assert.rejects(f.save(),/persistent job writes/);
  f.policy.mode='workspace-write';f.config.matlabResource=undefined;
  await assert.rejects(f.save(),/overlay/);
  assert.deepEqual(f.seen,[]);
});

test('refusals, wrong path, missing files and non-SLX contents never report a successful save',async()=>{
  const refused=await fixture(async()=>({status:'refused',reason:'同名模型已绑定其他路径'}));
  await assert.rejects(refused.save(),/其他路径/);
  const wrong=await fixture(async f=>({status:'saved',model:f.model,path:fixturePath,bytes:1}));
  await assert.rejects(wrong.save(),/位置/);
  const missing=await fixture(async f=>({status:'saved',model:f.model,path:f.target,bytes:1}));
  await assert.rejects(missing.save(),/ENOENT/);
  const invalid=await fixture(async f=>{
    await writeFile(f.target,'text');
    return {status:'saved',model:f.model,path:f.target,bytes:4};
  });
  await assert.rejects(invalid.save(), error => {
    const record=JSON.parse(error.message);
    assert.match(record.reason,/ZIP/);
    assert.equal(record.stage,'host_file_check');
    assert.deepEqual(record.verified,['matlab_save']);
    assert.equal(record.session_verified,true);
    assert.equal(record.file_exists,true);
    assert.equal(record.path,invalid.target);
    return true;
  });
});
