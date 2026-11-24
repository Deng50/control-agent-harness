/** Actual compiled React form; isolated native transport peers, no jobs/provider/Host. */
import {clientRequire} from './client-require.mjs';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import {workspaceKey} from '../src/client/form-schema.mjs';
const require=createRequire(resolve(import.meta.dirname,'../package.json')),vendorRequire=createRequire(resolve(import.meta.dirname,'../../vendor/deepseek-harness/package.json'));
const React=require('react'),{JSDOM}=vendorRequire('jsdom'),bundle=await readFile(process.env.CONTROL_TEST_CLIENT_BUNDLE??resolve(import.meta.dirname,'../lib/client.js'),'utf8');

test('only an explicit migration creates a new professional Session, keeps old history reachable and clears all old draft authority',async()=>{
  const dom=new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>',{url:'http://localhost:5288/'}),previous={window:globalThis.window,document:globalThis.document,navigator:Object.getOwnPropertyDescriptor(globalThis,'navigator'),act:globalThis.IS_REACT_ACT_ENVIRONMENT};
  globalThis.window=dom.window;globalThis.document=dom.window.document;Object.defineProperty(globalThis,'navigator',{value:dom.window.navigator,configurable:true});globalThis.IS_REACT_ACT_ENVIRONMENT=true;
  const {createRoot}=require('react-dom/client'),{act}=React,workspace='D:\\unit-explicit-professional-migration',key=workspaceKey(workspace,'model'),old='old-started-standard',fresh='new-explicit-control-model',storageKey='control-engineering.v1';
  const config={epochs:13,learning_rate:.004,plan_refs:{signals:{plan_id:'old-signals',version_hash:'old-version'}},network:{hidden_dims:[7],plan_refs:{training:{plan_id:'old-training',version_hash:'old-version'}}}};
  dom.window.localStorage.setItem(storageKey,JSON.stringify({version:1,workspace,role:'model',projects:{[key]:{operation:'train_supervised',inputs:'{"dataset":"existing-data"}',config:JSON.stringify(config),sessionId:old,roleSessionId:old,jobId:'old-job',attemptId:'old-attempt',supportJobs:[{jobId:'old-support'}],review:{requirements:[],refs:{signals:config.plan_refs.signals},statuses:{}}}}}));
  let descriptor,Panel,actions,created=0;const calls=[],navigation=[];
  vm.runInNewContext(bundle,{window:{__ModuleLoader__:{load:value=>descriptor=value}},document:dom.window.document,localStorage:dom.window.localStorage,console,structuredClone,AbortController,setTimeout,clearTimeout});
  const ctx={slots:{inject:(name,fn)=>{if(name==='main')fn();},register:(spec,component)=>{Panel=component;actions=spec.inject();}},jobs:{state:()=>{},watchRows:()=>()=>{},observe:()=>()=>{}},
    remote:{workspace:{create:async()=>({ok:true,value:{workspace:{workspaceId:'native-workspace'}}})},session:{create:async request=>{created++;assert.equal(request.workspaceId,'native-workspace');assert.equal(request.agentPreset,'control-model');return {ok:true,value:{sessionId:fresh,agentPreset:request.agentPreset}};},rename:async()=>({ok:true,value:{}})},commands:{execute:async(sessionId,line)=>{calls.push({sessionId,line});assert.ok(!line.startsWith('/control '),'Migration cannot admit a numerical/domain job');
      if(line.startsWith('/control-workspace ')){return {ok:true,value:{result:sessionId===old?{kind:'error',text:'CONTROL_PROFESSIONAL_SESSION_LOCKED: explicit migration required'}:{kind:'success',text:JSON.stringify({sessionId,attached:true,preset:{managed:true,agentPreset:'control-model'}})}}};}
      if(line.startsWith('/control-history '))return {ok:true,value:{result:{kind:'success',text:JSON.stringify({attempts:[]})}}};
      if(line.startsWith('/control-status '))return {ok:true,value:{result:{kind:'success',text:JSON.stringify({record:{state:'completed',owner:old,nativeJobId:'old-job',request:{operation:'train_supervised'},outcome:{cleanupVerified:true}}})}}};
      throw new Error('Unexpected native call '+line);}}},uiWorkspace:{openSession:id=>navigation.push(id)}};
  descriptor.factory(clientRequire).apply(ctx);const root=createRoot(dom.window.document.getElementById('root'));
  const button=text=>{const node=[...dom.window.document.querySelectorAll('button')].find(item=>item.textContent===text);assert.ok(node,'Missing actual migration button '+text);return node;};
  const click=node=>node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true}));const press=async text=>act(async()=>{click(button(text));await new Promise(done=>setTimeout(done,0));});
  try{await act(async()=>root.render(React.createElement(Panel,{...actions,useJobs:pick=>pick({rows:{},observed:{}})})));
    await press('打开独立会话');assert.equal(created,0);assert.deepEqual(navigation,[]);assert.match(dom.window.document.querySelector('[role="alert"]').textContent,/CONTROL_PROFESSIONAL_SESSION_LOCKED/);
    await press('新建专业会话并保留旧记录');assert.equal(created,1);assert.deepEqual(navigation,[fresh]);const draft=JSON.parse(dom.window.localStorage.getItem(storageKey)).projects[key];
    assert.equal(draft.roleSessionId,fresh);assert.equal(draft.sessionId,fresh);assert.ok(draft.legacySessionIds.includes(old));assert.equal(draft.inputs,'{"dataset":"existing-data"}');assert.equal(draft.operation,'train_supervised');assert.deepEqual(JSON.parse(draft.config),{epochs:13,learning_rate:.004,network:{hidden_dims:[7]}});assert.equal(draft.review,null);assert.equal(draft.jobId,'');assert.equal(draft.attemptId,'');assert.ok(!draft.supportJobs?.length);
    const select=dom.window.document.querySelector('select[aria-label="查看会话任务"]');assert.ok([...select.options].some(row=>row.value===old));await act(async()=>{select.value=old;select.dispatchEvent(new dom.window.Event('change',{bubbles:true}));});await press('刷新任务记录');assert.equal(calls.at(-1).sessionId,old);assert.match(calls.at(-1).line,/^\/control-history /);assert.ok(!calls.some(row=>row.line.startsWith('/control-confirm ')));assert.ok(!calls.some(row=>row.line.startsWith('/control ')));
  }finally{await act(async()=>root.unmount());dom.window.close();globalThis.window=previous.window;globalThis.document=previous.document;if(previous.navigator)Object.defineProperty(globalThis,'navigator',previous.navigator);else delete globalThis.navigator;globalThis.IS_REACT_ACT_ENVIRONMENT=previous.act;}
});
