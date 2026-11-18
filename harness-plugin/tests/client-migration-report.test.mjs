/** Actual React race: an old Session report response cannot overwrite a migrated Session. */
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import {workspaceKey} from '../src/client/form-schema.mjs';
const require=createRequire(resolve(import.meta.dirname,'../package.json')),vendorRequire=createRequire(resolve(import.meta.dirname,'../../vendor/deepseek-harness/package.json'));
const React=require('react'),{JSDOM}=vendorRequire('jsdom'),bundle=await readFile(resolve(import.meta.dirname,'../lib/client.js'),'utf8');

test('migration releases old report selection and invalidates an older pending inspection so a new Session result remains selected',async()=>{
  const dom=new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>',{url:'http://localhost:5288/'}),previous={window:globalThis.window,document:globalThis.document,navigator:Object.getOwnPropertyDescriptor(globalThis,'navigator'),act:globalThis.IS_REACT_ACT_ENVIRONMENT};
  globalThis.window=dom.window;globalThis.document=dom.window.document;Object.defineProperty(globalThis,'navigator',{value:dom.window.navigator,configurable:true});globalThis.IS_REACT_ACT_ENVIRONMENT=true;
  const {createRoot}=require('react-dom/client'),{act}=React,workspace='D:\\unit-migration-report-race',key=workspaceKey(workspace,'model'),old='old-role',fresh='new-role';
  const report=id=>({artifact_id:id,kind:'supervised_model',status:'verified',report:{parameter_count:1,scope:'Isolated component fixture; no trained accuracy claim'},files:[],metadata:{}}),A=report('old-artifact-A'),B=report('new-artifact-B');
  const jobs={rows:{[old]:[{id:'old-job',label:'old report',status:'completed'}]},observed:{'old-job':{text:JSON.stringify({event:'result',result:A})+'\n'}}};let descriptor,Panel,actions,releaseOld,pending=false;
  dom.window.localStorage.setItem('control-engineering.v1',JSON.stringify({version:1,workspace,role:'model',projects:{[key]:{operation:'inspect_workspace',inputs:'{}',config:'{}',sessionId:old,roleSessionId:old,jobId:'old-job',artifactId:A.artifact_id}}}));
  vm.runInNewContext(bundle,{window:{__ModuleLoader__:{load:value=>descriptor=value}},document:dom.window.document,localStorage:dom.window.localStorage,console,structuredClone,AbortController,setTimeout,clearTimeout});
  const ctx={slots:{inject:(name,fn)=>{if(name==='main')fn();},register:(spec,component)=>{Panel=component;actions=spec.inject();}},jobs:{state:()=>{},watchRows:()=>()=>{},observe:()=>()=>{}},uiWorkspace:{openSession:()=>{}},
    remote:{workspace:{create:async()=>({ok:true,value:{workspace:{workspaceId:'native-workspace'}}})},session:{create:async request=>({ok:true,value:{sessionId:fresh,agentPreset:request.agentPreset}}),rename:async()=>({ok:true,value:{}})},commands:{execute:async(sessionId,line)=>{
      if(line.startsWith('/control-workspace '))return {ok:true,value:{result:{kind:'success',text:JSON.stringify({sessionId,attached:true})}}};
      if(line.startsWith('/control ')){const request=JSON.parse(line.slice('/control '.length));assert.equal(sessionId,old);assert.equal(request.operation,'inspect_artifact');pending=true;return {ok:true,value:{result:{kind:'success',text:JSON.stringify({jobId:'old-inspect',attemptId:'old-inspect-attempt'})}}};}
      if(line.startsWith('/control-status ')){assert.equal(sessionId,old);await new Promise(done=>releaseOld=done);return {ok:true,value:{result:{kind:'success',text:JSON.stringify({record:{owner:old,nativeJobId:'old-inspect',state:'completed',outcome:{cleanupVerified:true},result:A}})}}};}
      throw new Error('Unexpected native fixture request');}}}};
  descriptor.factory(require).apply(ctx);const root=createRoot(dom.window.document.getElementById('root'));
  const click=node=>node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true}));
  try{await act(async()=>{root.render(React.createElement(Panel,{...actions,useJobs:pick=>pick(jobs)}));await new Promise(done=>setTimeout(done,0));});assert.equal(pending,true);
    await act(async()=>{const button=[...dom.window.document.querySelectorAll('button')].find(row=>row.textContent==='新建专业会话并保留旧记录');assert.ok(button);click(button);await new Promise(done=>setTimeout(done,0));});
    jobs.rows[fresh]=[{id:'new-job-B',label:'new report',status:'completed'}];jobs.observed['new-job-B']={text:JSON.stringify({event:'result',result:B})+'\n'};
    await act(async()=>root.render(React.createElement(Panel,{...actions,useJobs:pick=>pick(jobs)})));
    const select=dom.window.document.querySelector('select[aria-label="选择作业"]');await act(async()=>{select.value='new-job-B';select.dispatchEvent(new dom.window.Event('change',{bubbles:true}));});
    const selected=()=>dom.window.document.querySelector('[aria-label="所选产物报告"] code')?.textContent;assert.equal(selected(),'new-artifact-B');
    await act(async()=>{releaseOld();await new Promise(done=>setTimeout(done,0));});assert.equal(selected(),'new-artifact-B','An old Session response must remain invalid after migration');
  }finally{await act(async()=>root.unmount());dom.window.close();globalThis.window=previous.window;globalThis.document=previous.document;if(previous.navigator)Object.defineProperty(globalThis,'navigator',previous.navigator);else delete globalThis.navigator;globalThis.IS_REACT_ACT_ENVIRONMENT=previous.act;}
});
