/** Actual compiled client factory + React SSR; never contacts Harness or a model. */
import {clientRequire} from './client-require.mjs';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import vm from 'node:vm';
import {createHash,webcrypto} from 'node:crypto';
import test from 'node:test';
import {operationDefaults,workspaceKey} from '../src/client/form-schema.mjs';

const require=createRequire(resolve(import.meta.dirname,'../package.json')),React=require('react'),{renderToStaticMarkup}=require('react-dom/server');
const source=await readFile(resolve(import.meta.dirname,'../lib/client.js'),'utf8');
test('sidebar project entry opens the workbench from the expanded sidebar and collapsed rail',()=>{
  let descriptor,entry;
  const selected=[];
  vm.runInNewContext(source,{window:{__ModuleLoader__:{load:value=>{descriptor=value;}}},structuredClone});
  descriptor.factory(clientRequire).apply({
    slots:{inject:(name,mount)=>{if(name==='sidebar.footer.action') mount();},
      register:(spec,component)=>{entry={spec,component};}},
    jobs:{state:()=>{}}, layout:{selectPanel:id=>selected.push(id)},
  });
  const props={...entry.spec.inject(),usePanelInfo:select=>select({activePanelId:'control-engineering'})};
  for(const wide of [true,false]){
    const element=entry.component({...props,wide});
    const html=renderToStaticMarkup(element);
    assert.match(html,/aria-label="控制工程工作台"/);
    assert.match(html,/aria-current="page"/);
    if(wide) assert.match(html,/天津大学<\/span><span>上海联合汽车电子有限公司/);
    else assert.match(html,/data-compact="true"/);
    element.props.onClick();
  }
  assert.deepEqual(selected,['control-engineering','control-engineering']);
});

function render(role,operation,mode='normal',change={}){
  const workspace='D:\\explicit-user-project',defaults=operationDefaults(operation),config={...defaults.config,...change},key=workspaceKey(workspace,role);
  let descriptor,Panel,actions;let executions=0;
  const storage=JSON.stringify({version:1,workspace,role,projects:{[key]:{operation,mode,inputs:JSON.stringify(defaults.inputs),config:JSON.stringify(config),sessionId:'',jobId:''}}});
  const context={window:{__ModuleLoader__:{load:value=>descriptor=value}},console,localStorage:{getItem:()=>storage},structuredClone,setTimeout,clearTimeout,AbortController};
  vm.runInNewContext(source,context,{filename:'actual-harness-client-factory.js'});
  const exported=descriptor.factory(clientRequire);exported.apply({slots:{inject:(name,fn)=>{if(name==='main')fn();},register:(spec,component)=>{Panel=component;actions=spec.inject();}},jobs:{state:()=>{}},remote:new Proxy({},{get:()=>{executions++;throw Error('SSR must not execute any Native request');}})});
  const html=renderToStaticMarkup(React.createElement(Panel,{...actions,useJobs:pick=>pick({rows:{},observed:{}})}));
  assert.equal(executions,0);return html;
}
test('actual three-role client renders V2 task selection, material formats and original native-chat entry',()=>{
  const model=render('model','register_material');assert.match(model,/accept="\.py,\.m,\.slx,\.mdl,\.csv,\.xlsx,\.json,\.npz"/);assert.match(model,/项目工作目录/);assert.match(model,/Harness 对话/);
  assert.match(render('controller','synthesize_mpc'),/仅设计模板/);assert.match(render('lightweight','train_external_imitation'),/原对象三版本接受/);
});
test('ordinary and expert render the same requested NN hyperparameters without rewriting their payload',()=>{
  const cfg={hidden_dims:[17,9],learning_rate:.0047,epochs:7,batch_size:11};
  const ordinary=render('model','train_supervised','normal',cfg),expert=render('model','train_supervised','expert',cfg);
  for(const html of [ordinary,expert]){assert.match(html,/实际网络与训练参数/);assert.match(html,/value="0\.0047"/);assert.match(html,/value="7"/);assert.match(html,/17/);assert.match(html,/9/);assert.match(html,/确认以上摘要|生成 \/ 更新实际摘要/);}
});
test('actual nested PINN and PGAC forms display their capability and pre-training contract boundaries',()=>{
  const pinn=render('model','train_pinn');assert.match(pinn,/物理残差权重/);assert.match(pinn,/条件 ODE 轨迹/);assert.match(pinn,/ablation/);
  const pgac=render('lightweight','optimize_external_policy');assert.match(pgac,/训练前冻结的完整原对象/);assert.match(pgac,/Critic \/ PPO \/ SAC/);assert.match(pgac,/已合格模型范围内/);
});
test('data-only and external data forms expose causal column/target/units rather than inventing physical dimensions',()=>{
  const data=render('model','prepare_supervised_dataset');for(const text of ['列位置','输入特征','训练目标','单位','时间 / 轨迹分组列'])assert.match(data,new RegExp(text));
  const external=render('lightweight','import_expert_data');assert.match(external,/动作 \/ 增量边界/);assert.match(external,/不自动|先按完整轨迹/);assert.match(external,/数据-only 可为 null/);
});

function downloadClient(payload,{status=200}={}){
  let descriptor,actions;const clicks=[],requests=[];
  class BrowserURL extends URL {static createObjectURL(){return 'blob:verified-unit-file';}static revokeObjectURL(){}}
  const context={window:{__ModuleLoader__:{load:value=>descriptor=value}},console,localStorage:{getItem:()=>null},structuredClone,AbortController,
    URL:BrowserURL,Blob,crypto:webcrypto,Uint8Array,setTimeout:fn=>{fn();return 1;},clearTimeout,
    document:{baseURI:'http://localhost:5288/',querySelector:()=>({}),createElement:()=>({click(){clicks.push({filename:this.download,href:this.href});}})},
    fetch:async(url,options)=>{requests.push({url:String(url),options});return {ok:status===200,status,headers:{get:()=>String(payload.byteLength)},arrayBuffer:async()=>payload.buffer.slice(payload.byteOffset,payload.byteOffset+payload.byteLength)};}};
  vm.runInNewContext(source,context);descriptor.factory(clientRequire).apply({slots:{inject:(name,fn)=>{if(name==='main')fn();},register:spec=>{actions=spec.inject();}},jobs:{state:()=>{}}});
  return {actions,clicks,requests};
}
test('actual client downloads a complete >32K report through native authentication and verifies its entire SHA before saving',async()=>{
  const payload=Buffer.from(JSON.stringify({saved_log:'actual full file test bytes '.repeat(6000)})),sha=createHash('sha256').update(payload).digest('hex'),c=downloadClient(payload);
  const result=await c.actions.downloadVerifiedFile('D:\\project\\artifacts\\result\\report.json',sha,'result-report.json',payload.byteLength);
  assert.ok(payload.byteLength>32000);assert.equal(result.bytes,payload.byteLength);assert.equal(result.hash_verified,true);assert.equal(result.sha256,sha);assert.equal(c.clicks.length,1);
  assert.equal(c.requests[0].options.credentials,'same-origin');assert.equal(c.requests[0].options.redirect,'follow');assert.match(c.requests[0].url,/^http:\/\/localhost:5288\/api\/file\?path=/);
});
test('actual client rejects hash mismatch, byte mismatch and native limit failures without saving a truncated file',async()=>{
  const payload=Buffer.from('complete source bytes'),sha=createHash('sha256').update(payload).digest('hex');
  for(const scenario of ['hash','bytes','413']){const c=downloadClient(payload,{status:scenario==='413'?413:200});await assert.rejects(c.actions.downloadVerifiedFile('D:\\p\\report.json',scenario==='hash'?'0'.repeat(64):sha,'report.json',scenario==='bytes'?payload.byteLength+1:payload.byteLength));assert.equal(c.clicks.length,0);}
  const c=downloadClient(payload);await assert.rejects(c.actions.download({host_path:'D:\\p\\bundle.zip',filename:'bundle.zip',sha256:sha,bytes:50*1024*1024+1}),/50 MiB/);assert.equal(c.requests.length,0);assert.equal(c.clicks.length,0);
});
