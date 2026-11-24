/** Real React + jsdom unmount boundary; all Native calls are isolated stubs. */
import {clientRequire} from './client-require.mjs';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import {workspaceKey} from '../src/client/form-schema.mjs';

const require=createRequire(resolve(import.meta.dirname,'../package.json'));
const vendorRequire=createRequire(resolve(import.meta.dirname,'../../vendor/deepseek-harness/package.json'));
const {JSDOM}=vendorRequire('jsdom'),React=require('react');
const source=await readFile(process.env.CONTROL_TEST_CLIENT_BUNDLE??resolve(import.meta.dirname,'../lib/client.js'),'utf8');
const storageKey='control-engineering.v1',workspace='D:\\unit-persistence-only',modelKey=workspaceKey(workspace,'model'),controllerKey=workspaceKey(workspace,'controller');

async function fixture({deferCreate=false,refuseFirst=false}={}){
  const dom=new JSDOM('<!doctype html><html><head></head><body><div id="root"></div></body></html>',{url:'http://localhost:5288/'});
  const previous={window:globalThis.window,document:globalThis.document,navigator:Object.getOwnPropertyDescriptor(globalThis,'navigator'),act:globalThis.IS_REACT_ACT_ENVIRONMENT};
  globalThis.window=dom.window;globalThis.document=dom.window.document;Object.defineProperty(globalThis,'navigator',{value:dom.window.navigator,configurable:true});globalThis.IS_REACT_ACT_ENVIRONMENT=true;
  const {createRoot}=require('react-dom/client'),{act}=React;let descriptor,Panel,actions,root,created=0,releaseCreate,onOpen;
  const navigations=[];
  dom.window.localStorage.setItem(storageKey,JSON.stringify({version:1,workspace,role:'model',projects:{[modelKey]:{operation:'inspect_workspace',inputs:'{}',config:'{}',chat:'original'}}}));
  const context={window:{__ModuleLoader__:{load:value=>descriptor=value}},document:dom.window.document,localStorage:dom.window.localStorage,console,structuredClone,AbortController,setTimeout,clearTimeout};
  vm.runInNewContext(source,context,{filename:'actual-built-control-client.js'});
  const ctx={slots:{inject:(name,fn)=>{if(name==='main')fn();},register:(spec,component)=>{Panel=component;actions=spec.inject();}},
    jobs:{state:()=>{},watchRows:()=>()=>{},observe:()=>()=>{}},
    remote:{workspace:{create:async()=>({ok:true,value:{workspace:{workspaceId:'unit-native-workspace'}}})},session:{create:async request=>{assert.equal(request.workspaceId,'unit-native-workspace');assert.equal(request.cwd,undefined);created++;const value={ok:true,value:{sessionId:'unit-created-session-'+created,agentPreset:request.agentPreset}};if(deferCreate)await new Promise(done=>releaseCreate=done);return value;},rename:async()=>({ok:true,value:{}})},commands:{execute:async(sessionId,command)=>{assert.match(command,/^\/control-workspace /);if(refuseFirst){refuseFirst=false;return {ok:true,value:{result:{kind:'error',text:JSON.stringify({code:'WORKSPACE_PREFLIGHT_FAILED',actor:'worker',operation:'replace',path:workspace,os_code:5,detail:'denied',hint:'修复后重试'})}}};}return {ok:true,value:{result:{kind:'success',text:JSON.stringify({sessionId,attached:true})}}};}}},
    uiWorkspace:{openSession:sessionId=>{navigations.push({sessionId,storage:JSON.parse(dom.window.localStorage.getItem(storageKey))});root.unmount();onOpen?.();}}};
  descriptor.factory(clientRequire).apply(ctx);
  async function mount(){root=createRoot(dom.window.document.getElementById('root'));await act(async()=>root.render(React.createElement(Panel,{...actions,useJobs:pick=>pick({rows:{},observed:{}})})));}
  function button(text){const value=[...dom.window.document.querySelectorAll('button')].find(node=>node.textContent===text);assert.ok(value,'Button missing: '+text);return value;}
  function click(node){node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true}));}
  async function openAndUnmount(){await act(async()=>{const boundary=new Promise(done=>onOpen=done);click(button('打开独立会话'));await boundary;});}
  async function changeText(node,text){await act(async()=>{const setter=Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype,'value').set;setter.call(node,text);node.dispatchEvent(new dom.window.Event('input',{bubbles:true}));});}
  async function close(){try{await act(async()=>root?.unmount());}finally{dom.window.close();globalThis.window=previous.window;globalThis.document=previous.document;if(previous.navigator)Object.defineProperty(globalThis,'navigator',previous.navigator);else delete globalThis.navigator;globalThis.IS_REACT_ACT_ENVIRONMENT=previous.act;}}
  await mount();return {dom,mount,button,click,act,openAndUnmount,changeText,close,navigations,get created(){return created;},release:()=>releaseCreate?.()};
}

test('created role Session is persisted before native navigation unmount; reopening reuses its exact handle',async()=>{
  const f=await fixture();try{await f.openAndUnmount();assert.equal(f.navigations[0].storage.projects[modelKey].roleSessionId,'unit-created-session-1','Persistence must complete synchronously before openSession can unmount the panel');
    await f.mount();await f.openAndUnmount();assert.equal(f.created,1);assert.deepEqual(f.navigations.map(row=>row.sessionId),['unit-created-session-1','unit-created-session-1']);
  }finally{await f.close();}
});

test('workspace probe failure displays actor and code; retry attaches the existing native session',async()=>{
  const f=await fixture({refuseFirst:true});try {
    await f.act(async()=>f.click(f.button('打开独立会话')));
    assert.match(f.dom.window.document.body.textContent,/预检失败，计算尚未开始。数值进程.*replace.*5/);
    assert.equal(f.navigations.length,0);
    await f.openAndUnmount();
    assert.equal(f.created,1);
    assert.equal(f.navigations[0].sessionId,'unit-created-session-1');
  }finally{await f.close();}
});

test('deferred Session creation merges into latest drafts without losing same-key chat or a later role configuration',async()=>{
  const f=await fixture({deferCreate:true});try{
    await f.act(async()=>f.click(f.button('打开独立会话')));
    const chat=f.dom.window.document.querySelector('textarea[placeholder="说明目标和希望修改的参数…"]');assert.ok(chat);await f.changeText(chat,'edited while Session create was pending');
    await f.act(async()=>{const role=[...f.dom.window.document.querySelectorAll('nav button')].find(node=>node.textContent.includes('控制算法设计与调优'));f.click(role);});
    const config=f.dom.window.document.querySelector('textarea[aria-label="阶段配置"]');assert.ok(config);await f.changeText(config,'{"seed":73}');
    await f.act(async()=>{f.release();await new Promise(done=>setTimeout(done,0));});
    assert.equal(f.navigations.length,1);const saved=f.navigations[0].storage;assert.equal(saved.role,'controller');assert.equal(saved.projects[modelKey].chat,'edited while Session create was pending');assert.deepEqual(JSON.parse(saved.projects[controllerKey].config),{seed:73});assert.equal(saved.projects[modelKey].roleSessionId,'unit-created-session-1');
  }finally{await f.close();}
});
