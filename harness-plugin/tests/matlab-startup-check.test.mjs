import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { checkStartupConnection, apply } from '../src/matlab-startup-check.mjs';
import { requestWebRebind } from '../src/matlab-web-rebind.mjs';

const identity = { matlab_pid: 12345, matlab_release: '2025b', session_id: randomUUID() };
const sample = { matlab_pid: identity.matlab_pid, release: identity.matlab_release, session_id: identity.session_id, sum: 5 };
const value = text => ({ results: [{ result: { isError: false, value: {content: [{type: 'text', text}]} } }] });

test('an existing legacy record starts the interface in verification-only mode', async t => {
  const root = await mkdtemp(join(tmpdir(),'matlab-ready-'));
  t.after(() => rm(root,{recursive:true,force:true}));
  const identityFile = join(root,'identity.json');
  await writeFile(identityFile,JSON.stringify(identity));
  const binding = {pid:identity.matlab_pid, session_id:identity.session_id};
  let receipt, checks=0;
  const ctx = {on(){}, get:()=>({matlabRecovery:{
    pending:async options=>{assert.equal(options.inspectOwner,true); return {token:'legacy',owner:{session:binding}};}, check:async()=>{ checks++; },
  }}), provide:(_name,r)=>{ receipt=r; }};
  const supervision = process.env.CONTROL_ONECLICK_SUPERVISE;
  delete process.env.CONTROL_ONECLICK_SUPERVISE;
  t.after(() => { if (supervision !== undefined) process.env.CONTROL_ONECLICK_SUPERVISE = supervision; });
  await apply(ctx,{identityFile});
  assert.equal(receipt.status,'pending_verification');
  assert.equal(checks,1);
  let reads=0;
  ctx.get = () => ({matlabRecovery:{
    pending:async()=>++reads === 1 ? null : {token:'new',owner:{session:binding}},
    check:async()=>{ checks++; },
  }});
  ctx.tools = {get:()=>true, execute:async()=>({results:[{result:{isError:true,error:{message:'connection ended'}}}]})};
  await apply(ctx,{identityFile});
  assert.equal(receipt.status,'pending_verification');
  assert.equal(checks,2);
  binding.session_id = 'another-instance';
  await assert.rejects(apply(ctx,{identityFile}),/另一个实例/);
});

function check(render) {
  const tools = {
    get: () => true,
    execute: async ({calls}) => {
      assert.equal(calls.length, 1);
      assert.equal(calls[0].name, 'mcp__matlab__evaluate_matlab_code');
      const { code } = JSON.parse(calls[0].arguments);
      const prefix = 'CONTROL_MATLAB_READY_' + calls[0].id + ':';
      assert.ok(code.includes(prefix));
      assert.ok(code.includes('2+3'));
      return render(prefix);
    },
  };
  return checkStartupConnection(tools, identity, AbortSignal.timeout(1000));
}

test('年度版本、MATLAB 实际 PID、会话及算术都相同才通过', async () => {
  assert.deepEqual(await check(prefix => value(prefix + JSON.stringify(sample))), {
    release: '2025b', matlabPid: identity.matlab_pid, sum: 5,
  });
  for (const delta of [{release:'2024b'}, {matlab_pid:12346}, {session_id:randomUUID()}, {sum:4}]) {
    await assert.rejects(check(prefix => value(prefix + JSON.stringify({...sample, ...delta}))), /不一致/);
  }
});

test('陈旧回执、重复回执、损坏回执、隔离均不能报告已连通', async () => {
  await assert.rejects(check(() => value('other:' + JSON.stringify(sample))), /唯一/);
  await assert.rejects(check(prefix => value([1,2].map(() => prefix + JSON.stringify(sample)).join('\n'))), /唯一/);
  await assert.rejects(check(prefix => value(prefix + '{')), /JSON/);
  await assert.rejects(check(() => ({results:[{result:{isError:true, error:{message:'MatlabResourceQuarantined'}}}]})), /待核验/);
});

test('缺少 Toolkit 的注册或启动中止时不派发 MATLAB 调用', async () => {
  let dispatched = false;
  const tools = { get: name => name.endsWith('evaluate_matlab_code'), execute: () => { dispatched = true; } };
  const stop = new AbortController();
  const pending = checkStartupConnection(tools, identity, stop.signal);
  stop.abort();
  await assert.rejects(pending, {name:'AbortError'});
  assert.equal(dispatched, false);
});

test('rebind waits for the bound process to exit, all agent inboxes and owner-scoped jobs to drain', async () => {
  const terminal = {session:{pid:identity.matlab_pid,session_id:identity.session_id,identity:'birth',host:'pc'},
    owner:null,queue:[],session_alive:false};
  const agent = {id:'a', status:'idle', inbox:{nextTurn:[],nextStep:[]}, whenIdle:async () => {}};
  let jobs = [];
  const ctx = {agents:{list:() => [agent]}, jobs:{list:owner => owner === agent.id ? jobs : []}};
  const exits = [];
  const check = (inspect, noOpen=true) => requestWebRebind(ctx,terminal.session,inspect,code => exits.push(code),noOpen,new AbortController().signal);
  for (const bad of [{session_alive:true},{session_alive:null},{owner:{}},{queue:[{}]},
    {session:{...terminal.session,session_id:'other'}},{session:{...terminal.session,identity:'changed'}},
    {quarantine:{owner:{session:{...terminal.session,identity:'changed'}}}}]) {
    assert.equal(await check(async () => ({...terminal,...bad})), false);
  }
  assert.equal(await check(async () => {agent.status='running'; return terminal;}), false);
  agent.status='idle';
  agent.inbox.nextTurn.push({});
  assert.equal(await check(async () => terminal), false);
  agent.inbox.nextTurn=[];
  jobs=[{status:'stopping'}];
  assert.equal(await check(async () => terminal), false);
  jobs=[];
  assert.equal(await check(async () => terminal), true);
  assert.equal(await check(async () => terminal,false), true);
  assert.deepEqual(exits,[76,75]);
});
