import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash,randomUUID } from 'node:crypto';
import {mkdir,mkdtemp,writeFile,rm} from 'node:fs/promises';
import {resolve} from 'node:path';
import {MatlabTaskGrants} from '../src/matlab-task-grants.mjs';

const codeRoot=resolve(import.meta.dirname,'../..');
const output=resolve(codeRoot,'test-output/matlab-task-grants');
const sha=s=>createHash('sha256').update(s).digest('hex');

async function fixture(t) {
  await mkdir(output,{recursive:true});
  const workspace=await mkdtemp(resolve(output,'case-'));
  t.after(()=>rm(workspace,{recursive:true,force:true}));
  const config={codeRoot,matlabResource:{directory:workspace,identityFile:resolve(workspace,'marker.json')}};
  const env={CONTROL_MATLAB_BOUND_EXECUTABLE:resolve(workspace,'matlab.fixture'),CONTROL_MATLAB_MCP_EXECUTABLE:resolve(workspace,'mcp.fixture')};
  const state={session:{session_id:randomUUID(),pid:5321,host:'test',identity:'creation-fixture'},quarantine:null};
  const agent={id:'session1'},intent_id='files-task';
  const id='model-1', descriptor=JSON.stringify({adapter_spec:{backend:'matlab_function'}}),root=resolve(workspace,'plant');
  const metadata=JSON.stringify({id,kind:'plant_model',files:{'plant.json':sha(descriptor)}});
  const manifest={artifacts:{[id]:{kind:'plant_model',status:'verified',path:'plant',metadata_hash:sha(metadata),metadata_file_sha256:sha(metadata)}}};
  await mkdir(root);
  await Promise.all([writeFile(resolve(root,'plant.json'),descriptor),writeFile(resolve(root,'artifact.json'),metadata),
    writeFile(resolve(workspace,'manifest.json'),JSON.stringify(manifest)),writeFile(resolve(workspace,'resource.json'),JSON.stringify(state)),
    writeFile(config.matlabResource.identityFile,JSON.stringify({schema_version:1,kind:'control-harness-dedicated-matlab-v1',
      status:'ready',project_root:codeRoot,session_id:state.session.session_id,matlab_pid:state.session.pid})),
    ...Object.values(env).map(path=>writeFile(path,'synthetic binary fixture'))]);
  let timestamp=Date.now();
  const clock={now:()=>timestamp,env}, grants=new MatlabTaskGrants(config,clock);
  const request={operation:'sample_model',workspace,inputs:{model:id},config:{sampling:{steps:25},trial:{steps:1}}};
  const card={card_id:'fixture-card',operation:request.operation,workspace,session_id:agent.id,intent_id,
    matlab_scope:await grants.review(request,agent,intent_id)};
  const source={kind:'native_user_command',command_id:'native1',seq:55};
  return {agent,grants,card,request,state,source,env,config,clock,expire(){timestamp+=31*60000;}};
}

test('one native click bounds trial and full; parallel requests cannot reuse the same step', async t=>{
  const f=await fixture(t);
  assert.throws(()=>f.grants.issue(f.card,{kind:'assistant_text'}),/真实用户/);
  await assert.rejects(f.grants.reserve(f),/失效/);
  f.grants.issue(f.card,f.source);
  const results=await Promise.allSettled([f.grants.reserve(f),f.grants.reserve(f)]);
  assert.equal(results.filter(row=>row.status==='fulfilled').length,1);
  const ticket=results.find(row=>row.status==='fulfilled').value;
  await ticket.validate(f.state.session); ticket.finish();
  assert.deepEqual(f.grants.status(f.card).remaining,['full']);
  f.grants.issue(f.card,{...f.source,command_id:'native2',seq:56});
  await assert.rejects(f.grants.reserve(f),/已消耗/);
  f.request.config.sampling_run='full';
  const full=await f.grants.reserve(f);
  await full.validate(f.state.session);full.finish();
  assert.equal(f.grants.status(f.card).state,'consumed');
  assert.equal(f.grants.issue(f.card,f.source).state,'consumed');
});

test('expired native receipt, wrong session, changed MCP program, restart and cancelled job have no implicit permission',async t=>{
  const f=await fixture(t);f.grants.issue(f.card,f.source);
  await assert.rejects(f.grants.reserve({...f,agent:{id:'other'}}),/当前会话/);
  assert.equal(new MatlabTaskGrants(f.config,f.clock).status(f.card).state,'pending');
  f.expire();assert.equal(f.grants.issue(f.card,f.source).state,'expired');
  const renew={...f.source,command_id:'renew',seq:57};f.grants.issue(f.card,renew);
  const ticket=await f.grants.reserve(f);
  await assert.rejects(ticket.validate({...f.state.session,identity:'reused pid'}),/进程发生变化/);
  await writeFile(f.env.CONTROL_MATLAB_MCP_EXECUTABLE,'changed fixture');
  await assert.rejects(ticket.validate(f.state.session),/已变化/);
  ticket.revoke('job_cancelled');ticket.finish();
  assert.equal(f.grants.status(f.card).state,'revoked');
  f.grants.issue(f.card,{...renew,seq:58,command_id:'renew2'});
  await assert.rejects(f.grants.reserve(f),/新卡片/);
  assert.equal(f.grants.status(f.card).state,'revoked');
});

test('a new confirmed scope for the same task revokes its old grant; quarantine blocks admission',async t=>{
  const f=await fixture(t);f.grants.issue(f.card,f.source);
  f.request.config.sampling.steps=30;
  const current=await f.grants.review(f.request,f.agent,f.card.intent_id);
  const next={...f.card,card_id:'fixture-card2',matlab_scope:current};
  f.grants.issue(next,{...f.source,command_id:'native2',seq:56});
  assert.equal(f.grants.status(f.card).reason,'superseded_card');
  f.state.quarantine={reason:'unknown_outcome'};
  await writeFile(resolve(f.card.workspace,'resource.json'),JSON.stringify(f.state));
  await assert.rejects(f.grants.reserve({...f,card:next}),/隔离/);
});
