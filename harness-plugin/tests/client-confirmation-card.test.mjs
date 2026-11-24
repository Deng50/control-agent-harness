/** Exercises the built Web plugin in React/jsdom with a deterministic native transport. */
import {clientRequire} from './client-require.mjs';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import vm from 'node:vm';
import test from 'node:test';
import { workspaceKey } from '../src/client/form-schema.mjs';

const require = createRequire(resolve(import.meta.dirname,'../package.json'));
const vendor = createRequire(resolve(import.meta.dirname,'../../vendor/deepseek-harness/package.json'));
const {JSDOM}=vendor('jsdom'), React=require('react');
const source=await readFile(process.env.CONTROL_TEST_CLIENT_BUNDLE??resolve(import.meta.dirname,'../lib/client.js'),'utf8');
const workspace='D:\\control cases\\现场 数据', sessionId='native-model-session', handle={workspace,card_id:'immutable-card-1'};
const scopes=['signals','training'];

async function fixture({attempt,professionalSession=sessionId,matlab=false}={}) {
  const dom=new JSDOM('<!doctype html><html><head></head><body><div id="root"></div></body></html>',{url:'http://localhost/'});
  const previous={window:globalThis.window,document:globalThis.document,act:globalThis.IS_REACT_ACT_ENVIRONMENT};
  globalThis.window=dom.window;globalThis.document=dom.window.document;globalThis.IS_REACT_ACT_ENVIRONMENT=true;
  const {createRoot}=require('react-dom/client'), root=createRoot(dom.window.document.getElementById('root'));
  const entries=new Map(),calls=[];let descriptor,confirmed=false,definition,stale=false;
  function snapshot() {return {...handle,role:'model',intent_id:'intent-1',operation:'prepare_supervised_dataset',
    ...(matlab ? {operation:'sample_model',matlab_scope:{operation:'sample_model',variants:['trial','full'],
      dedicated_session:{pid:235},ttl_seconds:1800,dispatch_timeout_seconds:300,write_directory:workspace},
      matlab_grant:{state:confirmed?'active':'expired'}} : {}),
    requirements:scopes.map(scope=>({scope,reviewPayload:scope==='signals'?{worksheet:'现场记录',split:{mode:'trip'}}:{epochs:21},
      ...(scope==='training'?{effective_settings:{epochs:21,hidden_dims:[17,9]}}:{})})),
    statuses:Object.fromEntries(scopes.map(scope=>[scope,{confirmation_state:matlab||confirmed?'confirmed':'pending'}])),
    changes:{training:[{path:'epochs',before:18,after:21}]} };}
  dom.window.localStorage.setItem('control-engineering.v1',JSON.stringify({version:1,workspace,role:'model',projects:{
    [workspaceKey(workspace,'model')]:{sessionId,roleSessionId:professionalSession,legacySessionIds:professionalSession!==sessionId?[sessionId]:[],
      operation:'prepare_supervised_dataset',inputs:'{"material":"input"}',config:'{}'}}}));
  vm.runInNewContext(source,{window:{__ModuleLoader__:{load:value=>descriptor=value}},document:dom.window.document,localStorage:dom.window.localStorage,
    crypto:globalThis.crypto,structuredClone,AbortController,setTimeout,clearTimeout,console});
  const plugin=descriptor.factory(clientRequire);
  plugin.apply({slots:{inject:(name,fn)=>fn(),register:(spec,component)=>entries.set(spec.name,{spec,component})},
    effect:(fn)=>fn(),uiConversation:{events:{register:value=>{definition=value;return ()=>{};}}},
    jobs:{state:()=>{},watchRows:()=>()=>{},observe:()=>()=>{}},
    remote:{commands:{execute:async(s,line)=>{
      const at=line.indexOf(' '),name=line.slice(1,at),args=JSON.parse(line.slice(at+1));
      calls.push({name,args});
      if(stale || s!==sessionId) return {ok:true,value:{result:{kind:'error',text:'旧版本或其他会话的确认卡不可用'}}};
      let value;
      if(name==='control-card-status' || name==='control-confirm'){
        assert.deepEqual(args,handle,'Both actions address the immutable card in this workspace');
        if(name==='control-confirm') confirmed=true;
        value=snapshot();
      } else if(name==='control-prepare-review') value=snapshot();
      else if(name==='control') {if(args.operation!=='recover_interrupted_stage')assert.ok(confirmed);value={jobId:'job',attemptId:'attempt'};}
      else if(name==='control-history') value={attempts:attempt?[attempt]:[]};
      else if(name==='control-recover') value={jobId:'replacement',attemptId:'replacement-attempt'};
      else if(name==='control-status') value={record:attempt&&args.attemptId===attempt.attemptId?attempt:
        {owner:sessionId,nativeJobId:'job',request:{operation:'prepare_supervised_dataset'},state:'completed',outcome:{cleanupVerified:true}}};
      else throw new Error('Unexpected native command '+name);
      return {ok:true,value:{result:{kind:'success',text:JSON.stringify(value)}}};
    }}}});
  const render=async(element)=>React.act(async()=>root.render(element));
  const button=name=>{const found=[...dom.window.document.querySelectorAll('button')].find(item=>item.textContent===name);assert.ok(found,'Missing '+name);return found;};
  const click=async(name,count=1)=>React.act(async()=>{const target=button(name);for(let i=0;i<count;i++)target.click();});
  async function close(){await React.act(async()=>root.unmount());dom.window.close();globalThis.window=previous.window;globalThis.document=previous.document;globalThis.IS_REACT_ACT_ENVIRONMENT=previous.act;}
  return {dom,entries,calls,definition,render,button,click,close,expire:()=>{stale=true;}};
}

test('chat renders current Host values, shows diffs and sends one native human confirmation for two scopes',async()=>{
  const f=await fixture();
  try{
    const row=f.entries.get('tool.call.toolview');assert.equal(row.spec.key,'control_prepare_review');
    const content=[{type:'text',text:JSON.stringify({...handle,requirements:[{scope:'training',expectedPayload:'unauthenticated text'}]})}];
    await f.render(React.createElement(row.component,{phase:'result',block:{content},...row.spec.inject(sessionId)}));
    const body=f.dom.window.document.body;assert.match(body.textContent,/现场记录/);assert.match(body.textContent,/18 → 21/);assert.doesNotMatch(body.textContent,/unauthenticated/);
    assert.match(body.textContent,/实际训练参数（含自动取值）/);assert.match(body.textContent,/\[17,9\]/);
    assert.deepEqual(f.calls.map(call=>call.name),['control-card-status']);
    await f.click('确认以上摘要',2);
    assert.deepEqual(f.calls.map(call=>call.name),['control-card-status','control-confirm']);
    assert.equal(f.button('以上摘要已确认').disabled,true);assert.match(body.textContent,/发送“继续”/);
    const definition=f.definition;
    const call={type:'tool/call',seq:8,data:{name:'control_prepare_review',callId:'a'}}, result={type:'tool/result',surfaceOp:'append',data:{message:{source:{callId:'a'},content}}};
    assert.equal(definition.match(call).role,'start');assert.equal(definition.match(result).id,'a');
    const state=definition.update({state:definition.start({}, {event:call})},{event:result});
    const node=definition.buildViewNode({key:'review:a',id:'a',state,start:{event:call,location:{kind:'turn'}}});
    assert.equal(node.visibility,'hidden');assert.deepEqual(JSON.parse(JSON.stringify(node.data)),handle);
    const nested={type:'tool/ptc-dispatch',data:{name:'control_prepare_review',subCallId:'nested',content}};
    assert.equal(definition.match(nested).id,'nested');assert.equal(definition.start({}, {event:nested}).handle.card_id,handle.card_id);
    assert.equal(definition.match({...nested,data:{...nested.data,name:'other_tool'}}),null);
  }finally{await f.close();}
});

test('expired task grant remains actionable even when the business summary was already signed',async()=>{
  const f=await fixture({matlab:true});try{
    const row=f.entries.get('tool.call.toolview'),content=[{type:'text',text:JSON.stringify(handle)}];
    await f.render(React.createElement(row.component,{phase:'result',block:{content},...row.spec.inject(sessionId)}));
    assert.match(f.dom.window.document.body.textContent,/专用进程 PID 235/);
    assert.equal(f.button('重新授权本次 MATLAB 操作').disabled,false);
    await f.click('重新授权本次 MATLAB 操作',2);
    assert.deepEqual(f.calls.map(call=>call.name),['control-card-status','control-confirm']);
    assert.equal(f.button('以上摘要已确认').disabled,true);
  }finally{await f.close();}
});

test('handoff prepares a pinned draft in the selected role or workspace without dispatching or copying authority',async()=>{
  for(const migrate of [false,true]){
    const f=await fixture();try{
      const target='D:/control-cases/target',selected='design-immutable',sha='0'.repeat(64);
      const transfer={mode:'explicit_artifact_closure',workspace,project_id:'source-project',
        artifact_path:workspace+'/artifacts/'+selected,artifact_id:selected,metadata_hash:sha};
      const result={artifact_id:selected,kind:'controller_design',handoff:{agent:'controller',operation:null},
        optional_handoffs:[{agent:'controller',operation:'verify_design_delivery',inputs:{design:selected},
          missing_config:['case.x','case.reference'],config_suggestions:{target:'standalone_python',case:{},plan_refs:{signals:'old-authority'}}}],
        transfer,source_versions:[{id:selected,metadata_hash:sha}],delivery:{status:migrate?'artifact_ready':'partial',target:migrate?'published_artifact':'standalone_python',
          artifact_id:selected,source_artifact:selected,bundle_artifact:selected,request_satisfied:migrate?null:false,
          checks:{files:'passed',load:'not_run',standalone:'not_run',closed_loop:'not_run'},missing_requirements:['指定验收工况']}};
      const storage=f.dom.window.localStorage,state=JSON.parse(storage.getItem('control-engineering.v1'));
      state.projects[workspaceKey(workspace,'model')].jobId='source';
      state.projects[workspaceKey(migrate?target:workspace,'controller')]={sessionId:'wrong-legacy-session',roleSessionId:'target-session',
        review:{card_id:'old'},intentId:'old-intent',trainingReview:{card_id:'unrelated'},attemptId:'old-attempt',jobId:'old-job'};
      storage.setItem('control-engineering.v1',JSON.stringify(state));
      const jobs={rows:{},observed:{source:{text:JSON.stringify({event:'result',result})}}},entry=f.entries.get('main');
      await f.render(React.createElement(entry.component,{...entry.spec.inject(),useJobs:pick=>pick(jobs)}));
      const doc=f.dom.window.document;
      const evidence=doc.querySelector('[aria-label="交付验收证据"]').textContent;
      assert.match(evidence,/独立运行 · 尚未运行/);assert.match(evidence,migrate?/用户要求：待核实/:/用户要求：尚未满足/);
      assert.match(doc.querySelector('[aria-label="产物交接"]').textContent,/case.reference/);
      if(migrate){
        const input=[...doc.querySelectorAll('label')].find(row=>row.textContent==='目标工作区绝对路径').querySelector('input');
        await React.act(async()=>{
          Object.getOwnPropertyDescriptor(f.dom.window.HTMLInputElement.prototype,'value').set.call(input,target);
          input.dispatchEvent(new f.dom.window.Event('input',{bubbles:true}));
        });
        assert.equal(f.button('在目标工作区准备导入').disabled,false);
        await f.click('在目标工作区准备导入');
      }else await f.click('准备独立 Python 运行草稿');
      const saved=JSON.parse(storage.getItem('control-engineering.v1')),draft=saved.projects[workspaceKey(saved.workspace,saved.role)];
      assert.equal(saved.role,'controller');assert.equal(saved.workspace,migrate?target:workspace);
      assert.equal(draft.operation,migrate?'import_artifact':'verify_design_delivery');
      assert.deepEqual(JSON.parse(draft.inputs),migrate?{artifact_path:transfer.artifact_path,expected_artifact_id:selected,
        expected_metadata_hash:sha,expected_project_id:transfer.project_id}:{design:selected});
      assert.equal(draft.sessionId,'target-session');assert.equal(draft.jobId,'');assert.equal(draft.review,null);
      assert.equal(draft.intentId,null);assert.equal(draft.trainingReview,null);assert.ok(!JSON.parse(draft.config).plan_refs);
      assert.equal(f.calls.length,0);
    }finally{await f.close();}
  }
});

test('foreign session, failed status and a revision changed after display never show accepted or dispatch',async()=>{
  const f=await fixture();try{
    const row=f.entries.get('tool.call.toolview'),content=[{type:'text',text:JSON.stringify(handle)}];
    const render=id=>f.render(React.createElement(row.component,{key:id,phase:'result',block:{content},...row.spec.inject(id)}));
    await render('other-session');
    assert.match(f.dom.window.document.body.textContent,/其他会话/);
    assert.equal(f.dom.window.document.querySelector('[aria-label="工程确认卡"]'),null);
    assert.equal(f.calls.filter(call=>call.name==='control-confirm').length,0);
    await render(sessionId);f.expire();await f.click('确认以上摘要');
    assert.match(f.dom.window.document.body.textContent,/旧版本/);
    assert.doesNotMatch(f.dom.window.document.body.textContent,/以上摘要已确认/);
    assert.equal(f.calls.filter(call=>call.name==='control').length,0);
  }finally{await f.close();}
});

test('sidebar prepares combined scope only on selection, signs once and submits dataset and training with same handle',async()=>{
  const f=await fixture();try{
    const entry=f.entries.get('main');
    await f.render(React.createElement(entry.component,{...entry.spec.inject(),useJobs:pick=>pick({rows:{},observed:{}})}));
    const doc=f.dom.window.document, scope=doc.querySelector('[aria-label="本次任务范围"]');
    assert.equal(scope.value,'dataset');
    await React.act(async()=>{scope.value='training';scope.dispatchEvent(new f.dom.window.Event('change',{bubbles:true}));});
    await f.click('生成 / 更新实际摘要');
    const prepare=f.calls.find(call=>call.name==='control-prepare-review');assert.ok(prepare.args.next_training.config.hidden_dims);
    assert.equal(f.button('执行阶段').disabled,true);assert.equal(f.calls.filter(call=>call.name==='control-confirm').length,0);
    await f.click('确认以上摘要');assert.equal(f.button('执行阶段').disabled,false);
    await f.click('执行阶段');
    const operation=doc.querySelector('[aria-label="任务 / 阶段"]');
    await React.act(async()=>{operation.value='train_supervised';operation.dispatchEvent(new f.dom.window.Event('change',{bubbles:true}));});
    const dataInput=doc.querySelector('fieldset input[list]');
    await React.act(async()=>{
      Object.getOwnPropertyDescriptor(f.dom.window.HTMLInputElement.prototype,'value').set.call(dataInput,'derived-dataset');
      dataInput.dispatchEvent(new f.dom.window.Event('input',{bubbles:true}));
    });
    await f.click('执行阶段');
    assert.deepEqual(f.calls.filter(call=>call.name==='control').map(call=>[call.args.operation,call.args.card_id]),[['prepare_supervised_dataset',handle.card_id],['train_supervised',handle.card_id]]);
    assert.equal(f.calls.filter(call=>call.name==='control-confirm').length,1);
    assert.equal(f.calls.filter(call=>call.name==='control').at(-1).args.inputs.dataset,'derived-dataset');
    assert.ok(f.calls.filter(call=>call.name==='control').every(call=>!('next_training'in call.args)&&!('plan_refs'in call.args.config)));
  }finally{await f.close();}
});

test('legacy interrupted attempt gets reviewed and recovered in its original selected Session without manual refs',async()=>{
  const attempt={attemptId:'old-no-refs',agentName:'model',owner:sessionId,state:'interrupted',authority:'ownership_lost',
    request:{operation:'prepare_supervised_dataset',workspace,inputs:{material:'old-material'},config:{worksheet:'实测表'}}};
  const f=await fixture({attempt,professionalSession:'different-current-session'});
  try{
    const entry=f.entries.get('main');
    await f.render(React.createElement(entry.component,{...entry.spec.inject(),useJobs:pick=>pick({rows:{},observed:{}})}));
    await f.click('刷新任务记录');await f.click('载入原请求以审阅');
    await f.click('生成 / 更新实际摘要');
    const prepared=f.calls.find(call=>call.name==='control-prepare-review');
    assert.deepEqual(prepared.args.inputs,attempt.request.inputs);assert.deepEqual(prepared.args.config,attempt.request.config);
    assert.equal(f.button('按已审阅请求重试').disabled,true);assert.equal(f.calls.filter(call=>call.name==='control-recover').length,0);
    await f.click('确认以上摘要');await f.click('按已审阅请求重试');
    assert.deepEqual(f.calls.find(call=>call.name==='control-recover').args,{workspace,attemptId:attempt.attemptId,mode:'retry',card_id:handle.card_id});
    assert.deepEqual(f.calls.filter(call=>call.name==='control-confirm').map(call=>call.args),[handle]);
    assert.equal(f.calls.some(call=>call.name==='control'),false);
  }finally{await f.close();}
});

test('retained checkpoint is archived in its original session before numerical resume is enabled',async()=>{
  const attempt={attemptId:'old-stage',agentName:'model',owner:sessionId,state:'failed',workspace,
    checkpoint:'failed/model/point.pt',unpublishedStageId:'supervised_model-abcdef123456',
    outcome:{cleanupVerified:true},request:{operation:'train_supervised',workspace}};
  const f=await fixture({attempt,professionalSession:'different-current-session'});try{
    const entry=f.entries.get('main');
    await f.render(React.createElement(entry.component,{...entry.spec.inject(),useJobs:pick=>pick({rows:{},observed:{}})}));
    await f.click('刷新任务记录');
    assert.equal(f.button('从检查点续训').disabled,true);
    await f.click('校验并归档未发布检查点');
    assert.deepEqual(f.calls.find(call=>call.name==='control').args,{workspace,agent:'model',operation:'recover_interrupted_stage',
      inputs:{stage_id:attempt.unpublishedStageId},config:{}});
    assert.ok(!f.calls.some(call=>call.name==='control-recover'));
  }finally{await f.close();}
});

test('ordinary recovery keeps original refs, edited legacy request cannot recover with a mismatched card',async()=>{
  const attempt={attemptId:'legacy-training',agentName:'model',owner:sessionId,state:'failed',checkpoint:'failed/old/latest.pt',
    outcome:{cleanupVerified:true},request:{operation:'train_supervised',workspace,inputs:{dataset:'frozen'},config:{epochs:9,
      resume_checkpoint:'failed/earlier/latest.pt',plan_refs:{signals:{plan_id:'v1-old'}}}}};
  const f=await fixture({attempt});
  try{
    const entry=f.entries.get('main');
    await f.render(React.createElement(entry.component,{...entry.spec.inject(),useJobs:pick=>pick({rows:{},observed:{}})}));
    await f.click('刷新任务记录');await f.click('从检查点续训');
    assert.deepEqual(f.calls.find(call=>call.name==='control-recover').args,{workspace,attemptId:attempt.attemptId,mode:'resume'});
    await f.click('载入原请求以审阅');
    const config=f.dom.window.document.querySelector('textarea[aria-label="阶段配置"]');
    assert.equal(JSON.parse(config.value).resume_checkpoint,attempt.request.config.resume_checkpoint);
    await React.act(async()=>{
      Object.getOwnPropertyDescriptor(f.dom.window.HTMLTextAreaElement.prototype,'value').set.call(config,'{"epochs":99}');
      config.dispatchEvent(new f.dom.window.Event('input',{bubbles:true}));
    });
    await f.click('生成 / 更新实际摘要');await f.click('按原请求重试');
    assert.equal(f.calls.some(call=>call.name==='control-prepare-review'),false);
    assert.equal(f.calls.filter(call=>call.name==='control-recover').length,1);
  }finally{await f.close();}
});
