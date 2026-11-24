import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import { randomUUID } from 'node:crypto';
import test from 'node:test';

const require = createRequire(new URL('../package.json', import.meta.url));
const React = require('react');
const {renderToStaticMarkup} = require('react-dom/server');
const bundle = await readFile(new URL('../lib/client.js', import.meta.url), 'utf8');

function client(rejectFile, commandError) {
  let descriptor, actions, resultView, recheckView, gateDefinition, gateTail;
  const calls = [];
  const ok = value => ({ok:true, value});
  vm.runInNewContext(bundle, {window:{__ModuleLoader__:{load: module => { descriptor = module; }}}, crypto:{randomUUID}, structuredClone});
  descriptor.factory(require).apply({
    slots:{
      inject: (name, setup) => { if (['main','tool.call.toolview','conversation.chat.turnTail'].includes(name)) setup(); },
      register: (spec, view) => {
        if (spec.name === 'main') actions = spec.inject();
        if (spec.key === 'control_build_zoh_plant') resultView = view;
        if (spec.key === 'control_recheck_plant_workspace') recheckView = view;
        if (spec.id === 'control-plant-workspace-gate') gateTail=view;
      },
    },
    jobs:{state:() => {}},effect:fn=>fn(),
    uiConversation:{events:{register:def=>{if (def.kind==='control-plant-workspace-gate') gateDefinition=def;}}},
    fileUpload:{upload: async (session, file) => {
      calls.push({upload:file.name, session});
      if (file.name === rejectFile) return {ok:false, error:{code:'upload', message:'rejected'}};
      return ok({receiptId:'receipt-' + file.name});
    }},
    remote:{
      commands:{execute:async () => ok({result:commandError ? {kind:'error', text:JSON.stringify(commandError)} : {kind:'success', text:'{}'}})},
      workspace:{create:async () => ok({workspace:{workspaceId:'testWorkspace'}})},
      session:{create:async () => ok({sessionId:'user-chat',agentPreset:'control-model'}),rename:async()=>ok({}),
        prompt: async request => { calls.push({request}); return ok({}); }},
    },
    uiWorkspace:{openSession: session => calls.push({open:session})},
  });
  return {actions, calls, gateDefinition,
    renderRecheck:(result,isError)=>renderToStaticMarkup(React.createElement(recheckView,
      {phase:'result',block:{content:[{type:'text',text:JSON.stringify(result)}],isError}})),
    renderGate:recovery=>renderToStaticMarkup(React.createElement(gateTail,{turn:{turn:1},useWorkspaceGates:()=>[recovery]})),
    render: (text, isError) => renderToStaticMarkup(React.createElement(resultView,
    {phase:'result', block:{content:[{type:'text',text}], isError}}))};
}

test('new Python source and parameter files are native attachments on the same natural-language turn', async () => {
  const {actions, calls} = client();
  const files = [{name:'plant_numpy.py'}, {name:'case.json'}];
  await actions.prompt('fresh-session', '转换成 Simulink 被控对象', files, new AbortController().signal);
  assert.deepEqual(calls.slice(0,2), files.map(file => ({upload:file.name, session:'fresh-session'})));
  const {request} = calls[2];
  assert.equal(request.sessionId, 'fresh-session');
  assert.deepEqual(JSON.parse(JSON.stringify(request.content)), [{type:'text',text:'转换成 Simulink 被控对象'},
    ...files.map(file => ({type:'file',receiptId:'receipt-' + file.name}))]);
  assert.deepEqual(calls[3], {open:'fresh-session'});
  const failed = client('case.json');
  await assert.rejects(failed.actions.prompt('fresh-session','建模',files), /rejected/);
  assert.ok(failed.calls.every(row => row.upload));
});

test('build receipt distinguishes source verification, a saved file, and an uncertain terminal state', () => {
  const {render} = client();
  const path = 'D:\\new-workspace\\plant.slx';
  const success = render(JSON.stringify({status:'simulated',stage:'compare',verified:['save','reload','simulate','compare'],
    path, file_exists:true, sha256:'c'.repeat(64), bytes:1001, comparison:{compared:true,samples:22,max_abs_error:1e-15,tolerance:1e-11}}), false);
  for (const value of [path, '下载已验证的 SLX', '22', '1e-15', '1e-11']) assert.ok(success.includes(value));
  const failure = render('Error: ' + JSON.stringify({status:'quarantined',stage:'simulate',verified:['save','reload'],
    path,file_exists:true,reason:'injected Simulink diagnostic',verification_error:'running timers: 1',recovery:'保留现场'}), true);
  for (const value of ['运行仿真', '目标文件已存在', path, 'injected Simulink diagnostic', 'running timers: 1','保留现场']) assert.ok(failure.includes(value));
  assert.ok(!failure.includes('下载已验证的 SLX'));
  assert.match(render('permission denied', true), /permission denied/);
  const afterDelivery=render('Error: '+JSON.stringify({code:'PLANT_DELIVERY_COMPLETE',stage:'delivery_complete',
    status:'not_dispatched',verified_delivery:{path}}),true);
  assert.ok(afterDelivery.includes(path) && afterDelivery.includes('本轮模型已保存、重载并完成仿真'));
  assert.ok(!afterDelivery.includes('文件状态待核实'));
});

test('permission failure shows the actual workspace and recovery command for its owning Windows account', () => {
  const {render}=client();
  const workspace="D:\\new-workspace\\用户's case";
  const recovery={workspace,missing_rights:['WRITE_OWNER'],diagnose_command:'inspect workspace',can_prepare:true,prepare_command:'prepare only this directory'};
  const markup=() => render('Error: '+JSON.stringify({status:'failed',stage:'workspace_preflight',
    diagnostic:{workspace_recovery:recovery}}),true);
  for (const text of ['WRITE_OWNER','用户','inspect workspace','prepare only this directory','重新检查工作区并继续建模']) assert.ok(markup().includes(text));
  recovery.can_prepare=false;
  assert.ok(!markup().includes(recovery.prepare_command));
  assert.ok(markup().includes('选择自己拥有的独立 NTFS 工作目录'));
});

test('the panel preserves its new session and the Host recovery instructions when attachment preflight fails', async () => {
  const recovery={workspace:'D:\\test',missing_rights:['WRITE_OWNER'],can_prepare:false,diagnose_command:'inspect'};
  const {actions,calls}=client(undefined,{code:'WORKSPACE_PREFLIGHT_FAILED',operation:'check_WRITE_OWNER',actor:'host',workspace_recovery:recovery});
  await assert.rejects(actions.createSession(recovery.workspace,'test','model'), error =>
    error.pendingSessionId==='user-chat' && error.workspaceRecovery.diagnose_command==='inspect');
  assert.equal(calls.length,0);
});

test('a genuine Host refusal on a native shell call keeps recovery visible after the tool row folds', () => {
  const {gateDefinition,renderGate}=client();
  const recovery={workspace:'D:\\test',missing_rights:['WRITE_OWNER'],can_prepare:true,diagnose_command:'inspect',prepare_command:'prepare'};
  const checked_at='2026-11-22T10:36:34.733Z';
  const content=[{type:'text',text:'Error: '+JSON.stringify({stage:'workspace_preflight',preflight_observation:'previous_failure',diagnostic:{workspace_recovery:recovery,checked_at}})}];
  const event={seq:48,type:'tool/result',surfaceOp:'append',data:{message:{toolCallId:'pwsh-1',content,isError:true}}};
  assert.equal(gateDefinition.match(event),null);
  event.data.error={code:'PLANT_WORKSPACE_PREFLIGHT_REQUIRED'};
  assert.equal(gateDefinition.match(event).role,'start');
  const state=gateDefinition.start({},{event});
  const view=gateDefinition.buildViewNode({key:'turn-1',id:'pwsh-1',start:{event,location:{}},state});
  const markup=renderGate(view.data);
  for (const value of ['沿用上次记录','重新检查工作区并继续建模',checked_at]) assert.ok(markup.includes(value));
});

test('a recheck renders a current denial or a verified workspace with its effective sandbox policy', () => {
  const {renderRecheck}=client();
  const workspace='D:\\test';
  const recovery={workspace,missing_rights:['WRITE_OWNER'],diagnose_command:'inspect'};
  const diagnostic={workspace_recovery:recovery};
  assert.match(renderRecheck({diagnostic,preflight_observation:'fresh'},true),/本次预检发现缺少/);
  assert.match(renderRecheck({diagnostic},true),/预检记录中缺少/);
  const passed=renderRecheck({workspace,status:'verified',sandbox_mode:'workspace-write',session_sandbox_mode:'danger-full-access'},false);
  for (const value of [workspace,'workspace-write','可以继续回放 Python']) assert.ok(passed.includes(value));
});
