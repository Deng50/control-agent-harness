/** Actual built client + React DOM flow. Native results are isolated, shaped fixtures. */
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
const workspace='D:\\unit-table-preview-only',storageKey='control-engineering.v1';
const xlsx='D:\\uploaded\\model-actual.xlsx',csv='D:\\uploaded\\direct-actual.csv';
const preview=(path,worksheet)=>({schema_version:1,read_only:true,source:{path,sha256:path.endsWith('.csv')?'c'.repeat(64):'a'.repeat(64)},worksheets:[{worksheet,header_row:1,columns:[{index:0,header:path.endsWith('.csv')?'csv_actual_column':'xlsx_actual_column',worksheet}]}]});

async function fixture({deferSupport=false}={}){
  const dom=new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>',{url:'http://localhost:5288/'});
  const previous={window:globalThis.window,document:globalThis.document,navigator:Object.getOwnPropertyDescriptor(globalThis,'navigator'),act:globalThis.IS_REACT_ACT_ENVIRONMENT};
  globalThis.window=dom.window;globalThis.document=dom.window.document;Object.defineProperty(globalThis,'navigator',{value:dom.window.navigator,configurable:true});globalThis.IS_REACT_ACT_ENVIRONMENT=true;
  const {createRoot}=require('react-dom/client'),{act}=React;
  let descriptor,Panel,actions,releaseSupport,number=0;
  const requests=[],records=new Map(),jobs={rows:{},observed:{}};
  const projects=Object.fromEntries(['model','controller'].map(role=>[workspaceKey(workspace,role),{operation:'inspect_table',inputs:JSON.stringify({data_path:csv}),config:'{}',sessionId:role+'-session',roleSessionId:role+'-session',lastUpload:role==='model'?{path:xlsx,name:'model-actual.xlsx'}:null}]));
  dom.window.localStorage.setItem(storageKey,JSON.stringify({version:1,workspace,role:'model',projects}));
  vm.runInNewContext(source,{window:{__ModuleLoader__:{load:value=>descriptor=value}},document:dom.window.document,localStorage:dom.window.localStorage,console,structuredClone,AbortController,setTimeout,clearTimeout},{filename:'actual-built-control-client.js'});
  descriptor.factory(clientRequire).apply({slots:{inject:(name,fn)=>{if(name==='main')fn();},register:(spec,component)=>{Panel=component;actions=spec.inject();}},jobs:{state:()=>{}}});
  const root=createRoot(dom.window.document.getElementById('root'));
  const props={...actions,useJobs:pick=>pick(jobs),watchJobs:()=>()=>{},observeJob:()=>()=>{},review:async()=>({requirements:[]}),
    submit:async(sessionId,request)=>{const n=++number,jobId='unit-table-job-'+n,attemptId='unit-table-attempt-'+n;
      requests.push({sessionId,request:structuredClone(request),jobId,attemptId});
      const isSupport=request.config?.preview_rows===5;
      records.set(attemptId,{owner:sessionId,nativeJobId:jobId,state:'completed',request:structuredClone(request),outcome:{cleanupVerified:true},result:preview(request.inputs.data_path,request.inputs.data_path.endsWith('.xlsx')?'Frozen_F03':null),isSupport});
      return {jobId,attemptId};},
    status:async(sessionId,ws,attemptId)=>{const record=records.get(attemptId);assert.ok(record);assert.equal(record.owner,sessionId);assert.equal(ws,workspace);
      if(deferSupport&&record.isSupport){await new Promise(done=>releaseSupport=done);}return {record};}};
  await act(async()=>root.render(React.createElement(Panel,props)));
  const button=text=>{const value=[...dom.window.document.querySelectorAll('button')].find(node=>node.textContent===text);assert.ok(value,'Missing button '+text);return value;};
  const click=node=>node.dispatchEvent(new dom.window.MouseEvent('click',{bubbles:true}));
  async function press(text){await act(async()=>{click(button(text));await new Promise(done=>setTimeout(done,0));});}
  async function role(name){await act(async()=>{const value=[...dom.window.document.querySelectorAll('nav button')].find(node=>node.textContent.includes(name));assert.ok(value);click(value);});}
  function actual(){const node=dom.window.document.querySelector('[aria-label="实际数据表头"] pre');return node?JSON.parse(node.textContent):null;}
  async function close(){try{await act(async()=>root.unmount());}finally{dom.window.close();globalThis.window=previous.window;globalThis.document=previous.document;if(previous.navigator)Object.defineProperty(globalThis,'navigator',previous.navigator);else delete globalThis.navigator;globalThis.IS_REACT_ACT_ENVIRONMENT=previous.act;}}
  return {press,role,actual,requests,close,act,release:()=>releaseSupport?.()};
}

test('support XLSX header is superseded by the actual direct CSV inspect result with null worksheet and remains role scoped',async()=>{
  const f=await fixture();try{
    await f.press('读取本角色刚上传文件的实际表头');assert.equal(f.actual().source.path,xlsx);assert.equal(f.actual().worksheets[0].worksheet,'Frozen_F03');
    await f.press('执行阶段');assert.equal(f.requests.length,2);assert.equal(f.requests[1].request.operation,'inspect_table');assert.equal(f.requests[1].request.inputs.data_path,csv);
    assert.equal(f.actual().source.path,csv,'A retained support preview must not shadow a later direct result');assert.equal(f.actual().worksheets[0].worksheet,null);assert.equal(f.actual().worksheets[0].columns[0].header,'csv_actual_column');
    await f.role('控制算法设计与调优');assert.equal(f.actual(),null,'Another role must not inherit the model preview');
    await f.role('对象模型自动化训练');assert.equal(f.actual().source.path,csv);assert.equal(f.actual().worksheets[0].worksheet,null);
  }finally{await f.close();}
});

test('late support header response stays in its originating role while another role completes a direct CSV request',async()=>{
  const f=await fixture({deferSupport:true});try{
    await f.press('读取本角色刚上传文件的实际表头');await f.role('控制算法设计与调优');await f.press('执行阶段');
    assert.equal(f.actual().source.path,csv);assert.equal(f.actual().worksheets[0].worksheet,null);
    await f.act(async()=>{f.release();await new Promise(done=>setTimeout(done,0));});
    assert.equal(f.actual().source.path,csv);assert.equal(f.actual().worksheets[0].worksheet,null);
    await f.role('对象模型自动化训练');assert.equal(f.actual().source.path,xlsx);
  }finally{await f.close();}
});
