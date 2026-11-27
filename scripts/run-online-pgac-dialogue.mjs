/** Observe a real lightweight-role task; fixture clicks accept only the supplied configurations. */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash, randomUUID} from 'node:crypto';
import {copyFile, readFile, readdir, stat, writeFile} from 'node:fs/promises';
import {basename, dirname, resolve} from 'node:path';
import {isDeepStrictEqual, parseArgs} from 'node:util';
import {setTimeout as sleep} from 'node:timers/promises';
import {command, connectHarness, prepareSmokeWorkspace, project, writeSmokeEvidence} from './harness-rpc.mjs';

const {values} = parseArgs({options:{workspace:{type:'string'},materials:{type:'string'},
  'web-log':{type:'string'},action:{type:'string',default:'start'},text:{type:'string'},python:{type:'string'}}});
const workspace = resolve(values.workspace);
const root = resolve(project,'test-output');
assert.equal(dirname(workspace),root,'Use a fresh direct test-output child for fixture confirmations');
const evidence = resolve(root,basename(workspace)+'-observer.json');
const json = async file => JSON.parse(await readFile(file,'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const userTask = '我有这个被控对象和 MPC，请采集专家数据，训练轻量网络，再用 PGAC 做在线闭环微调，并在独立工况上对比三种控制器。工作区内的材料说明.md 给出了原始程序、接口、工况和我的验收条件。请按这些文件执行并交付可加载的策略包，告知最终选用哪一版。';
const rpc = values.action === 'verify' ? undefined :
  await connectHarness({logPath:values['web-log'],connectionClose:true});
let record;

async function submit(text) {
  await rpc('session/prompt',{request:{sessionId:record.sessionId,requestId:randomUUID(),mode:'queue',content:[{type:'text',text}]}});
  record.turns.push(text);
}
async function start() {
  await assert.rejects(stat(evidence),e=>e.code==='ENOENT');
  await assert.rejects(stat(workspace),e=>e.code==='ENOENT');
  await prepareSmokeWorkspace(workspace);
  const materials=resolve(values.materials);
  const evalConfig=await json(resolve(materials,'evaluation.json'));
  const tracking=evalConfig.evaluation.primary_metric==='tracking_rmse';
  const names=['plant.py','mpc.py','model.json','collection.json','imitation.json','online.json','evaluation.json'];
  for(const name of names)await copyFile(resolve(materials,name),resolve(workspace,name));
  if(tracking){
    assert.equal(basename(dirname(materials)),'spring','This dialogue fixture supports the synthetic spring only');
    const protocol=await json(resolve(materials,'protocol.json'));
    await writeSmokeEvidence(resolve(workspace,'protocol.json'),{
      object:'spring',primary_metric:'tracking_rmse',output:evalConfig.evaluation.tracking_output,
      acceptance:protocol.acceptance,action_metrics:protocol.action_metrics,hard_constraints:protocol.hard_constraints,
      partitions:protocol.partitions,replicates:protocol.replicates,families:protocol.families,steps:protocol.steps});
  }else{
    await copyFile(resolve(materials,'protocol.json'),resolve(workspace,'protocol.json'));
  }
  for (const [source,field] of [['plant-adapter','adapter'],['expert-adapter','adapter'],['sampling','sampling']]){
    await writeSmokeEvidence(resolve(workspace,source+'.json'),{[field]:await json(resolve(materials,source+'.json'))});
  }
  const registration=await json(resolve(dirname(materials),'operations.json'));
  await writeSmokeEvidence(resolve(workspace,'qualification.json'),registration.receipt.config);
  await writeSmokeEvidence(resolve(workspace,'model-source.json'),registration.model.config);
  await writeFile(resolve(workspace,'材料说明.md'),[
    '# 弹簧对象与专家控制器',
    'plant.py 提供完整状态离散对象，mpc.py 提供 MPC。model.json 为与原对象对应的精确离散方程。protocol.json 规定物理边界、独立工况与验收门限。',
    'plant-adapter.json、expert-adapter.json 已按正式登记接口封装。model-source.json 为方程来源，sampling.json 为原对象采样方案，qualification.json 是对应确定性验证划分上的实测探针配置。',
    'collection.json 为专家工况，imitation.json 为网络架构和训练设置，online.json 含在线回放设置与冻结验收条件，evaluation.json 为独立原对象评价设置。各文件已是完整的操作配置，可用公开角色工具的 config_file 文件名引用。请沿用这些预算和时序。',
    '我希望看到实际求解次数、学生仿真步数、三方独立表现、是否达到预先约定目标以及选择的策略包。'
  ].join('\n')+'\n','utf8');
  const pinned={};
  for(const file of await readdir(workspace))if(/\.(py|json|md)$/u.test(file))pinned[file]=hash(await readFile(resolve(workspace,file)));
  const catalog=await rpc('session/modelCatalog',{});
  const group=catalog.groups.find(g=>g.id==='deepseek-official');
  const model=group?.models.find(m=>m.id==='deepseek-flash')??group?.models[0];
  assert.ok(model,'The local host offers no configured official DeepSeek model');
  const {sessionId}=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-lightweight'}});
  record={workspace,sessionId,role:'control-lightweight',provider:group.id,model:model.id,started:new Date().toISOString(),turns:[],confirmations:[],pinned};
  await writeSmokeEvidence(evidence,record);
  await rpc('session/selectModel',{request:{sessionId,provider:group.id,model:model.id,reasoningEffort:'low'}});
  const trackingRequest='这次以位移跟踪 RMSE 为主指标，同时限制力和每拍力增量开销、要求零越界。请按这些冻结门限汇报正负阶跃、反转和接管等工况的实际改善或退化，说明交付哪一版。仅使用本工作区的合成弹簧材料、角色技能与此次正式操作结果；跳过其他案例、历史运行记录和其他工作区。';
  await submit(tracking ? userTask+trackingRequest : userTask);
}
const confirmedFiles={
  register_plant:'plant-adapter.json',register_expert:'expert-adapter.json',register_model:'model-source.json',
  sample_model:'sampling.json',validate_differentiable_model:'qualification.json',collect_external_expert:'collection.json',
  train_external_imitation:'imitation.json',optimize_external_policy:'online.json',evaluate_external_pgac:'evaluation.json'
};
async function observe() {
  let page=await rpc('session/events',{request:{address:{kind:'session',sessionId:record.sessionId},maxMessages:150}});
  const records=[...page.records];const cursor=page.cursor;
  while(page.hasMore){
    page=await rpc('session/events',{request:{address:{kind:'session',sessionId:record.sessionId},throughSeq:cursor,
      beforeSeq:Math.min(...page.records.map(r=>r.event.seq)),maxMessages:150}});
    records.push(...page.records);
  }
  const events=records.map(r=>r.event).sort((a,b)=>a.seq-b.seq);
  const visible=m=>(m?.content??[]).filter(x=>x.type==='text').map(x=>x.text).join('\n');
  const parse=text=>{try{return JSON.parse(text);}catch{return {};}};
  const results=new Map(events.filter(e=>e.type==='tool/result').map(e=>[e.data.message.toolCallId,e.data.message]));
  const calls=events.filter(e=>e.type==='tool/call').map(e=>{
    const {turn,name,arguments:args,callId}=e.data;return {turn,name,args:typeof args==='string'?parse(args):args,
      result:results.get(callId),seq:e.seq};
  });
  const history=await command(rpc,record.sessionId,'/control-history '+JSON.stringify({workspace}));
  record.steps=history.attempts.map(row=>({operation:row.request.operation,state:row.state,job:row.nativeJobId,
    artifact:row.result?.artifact_id,status:row.result?.status,detail:row.state==='failed'?row.detail?.slice(0,400):undefined}));
  record.calls=calls.map(c=>({seq:c.seq,turn:c.turn,name:c.name,operation:c.args?.operation,skill:c.args?.skill??c.args?.name,
    card_id:c.name==='control_prepare_review'&&c.result?parse(visible(c.result)).card_id:undefined,
    file:c.name==='read'?c.args?.file_path:undefined,config_file:c.args?.config_file,error:c.result?.isError||
    undefined,diagnostic:c.result?.isError?visible(c.result).slice(0,260):undefined}));
  const ends=events.filter(e=>e.type==='turn/end');
  record.ended=ends.length;record.lastReason=ends.at(-1)?.data.reason?.kind;
  const err=ends.at(-1)?.data.reason?.error;
  if(err)record.providerError={code:err.code,status:err.status,text:String(err.message??'').slice(0,200)};
  record.lastAnswer=events.filter(e=>e.type==='assistant/message').map(e=>visible(e.data.message)).filter(Boolean).at(-1);
  const receipts=calls.filter(c=>c.name==='control_prepare_review'&&c.result&&!c.result.isError);
  const pending=receipts.filter(c=>{
    const card=parse(visible(c.result));
    return card.card_id && !record.confirmations.includes(card.card_id);
  });
  if(values.action==='confirm'||values.action==='drive'&&pending.length){
    // Let the restricted worker create its ledger. Pre-creating this directory as the
    // observer can prevent the worker from writing on Windows.
    const markerPath=resolve(workspace,'.case-plans/test-fixture.json');
    assert.ok((await stat(resolve(workspace,'.case-plans/plans.json'))).isFile());
    try {
      await writeFile(markerPath,JSON.stringify({kind:'control-case-plan-test-fixture-v1',
        workspace,confirmation_source:'test_fixture',source:'frozen spring engineering configurations'}),{flag:'wx'});
    } catch(error) { if(error.code!=='EEXIST')throw error; }
    const marker=await json(resolve(workspace,'.case-plans/test-fixture.json'));
    assert.equal(marker.workspace,workspace);assert.equal(marker.confirmation_source,'test_fixture');
    const c=pending[0]; assert.ok(c,'No pending review');
    const card=parse(visible(c.result));assert.ok(card.card_id,'Nothing to confirm');
    assert.equal(c.args.workspace,workspace);assert.equal(c.args.agent,'lightweight');
    assert.ok(!record.confirmations.includes(card.card_id),'This review already has a fixture receipt');
    const file=confirmedFiles[c.args.operation];
    assert.ok(file,'Unregistered benchmark stage: '+c.args.operation);
    assert.ok(c.args.config_file===file || (!c.args.config_file&&isDeepStrictEqual(c.args.config,await json(resolve(workspace,file)))),'Review differs from benchmark config');
    for(const [name,digest] of Object.entries(record.pinned))assert.equal(hash(await readFile(resolve(workspace,name))),digest);
    const result=await command(rpc,record.sessionId,'/control-confirm '+JSON.stringify({workspace,card_id:card.card_id}));
    assert.equal(result.confirmation_source,'test_fixture');record.confirmations.push(card.card_id);
    if(record.turns.length===record.ended)await submit('已核对并确认本次工况与配置卡，请继续本任务。');
  }
  if(values.action==='drive'&&!pending.length&&record.turns.length===record.ended &&
      receipts.some(c=>c.turn===record.ended&&record.confirmations.includes(parse(visible(c.result)).card_id)) &&
      !record.steps.some(s=>s.operation==='evaluate_external_pgac'&&s.state==='completed')){
    await submit('已核对并确认本次配置卡，请继续完成工程任务。');
  }
  if (values.action === 'reply') {
    assert.ok(values.text);
    if (record.ended === record.turns.length &&
        record.steps.some(s => s.operation === 'evaluate_external_pgac' && s.state === 'completed')) {
      (record.postDeliveryQuestions ??= []).push({turn:record.ended, priorAnswer:record.lastAnswer, question:values.text});
    }
    await submit(values.text);
  }
}
async function verify() {
  const observed = await json(evidence);
  assert.equal(observed.workspace, workspace);
  assert.equal(observed.ended, observed.turns.length, 'Wait for the final model response before archiving');
  const succeeded = observed.steps.filter(s => s.state === 'completed' && s.artifact).reverse();
  const resultOf = operation => {
    const result = succeeded.find(s => s.operation === operation);
    assert.ok(result, 'Missing completed operation: ' + operation);
    return result;
  };
  const training = resultOf('optimize_external_policy');
  const evaluation = resultOf('evaluate_external_pgac');
  const artifact = resolve(workspace, 'artifacts', evaluation.artifact);
  const report = await json(resolve(artifact, 'report.json'));
  const candidate = await json(resolve(workspace, 'artifacts', training.artifact, 'report.json'));
  const collection = await json(resolve(workspace, 'artifacts', resultOf('collect_external_expert').artifact, 'report.json'));
  assert.equal(report.candidate, training.artifact);
  assert.equal(report.imitation_policy, resultOf('train_external_imitation').artifact);
  assert.ok(report.final_acceptance.approved, 'No approved delivery package in this run');
  const selected = report.final_acceptance.selected_version;
  const python = values.python ?? resolve(project, process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
  const fileHash = async path => hash(await readFile(path));
  const skillCalls = observed.calls.filter(c => c.name === 'skill' && !c.error);
  const confirmations = observed.calls.filter(c => c.name === 'control_prepare_review' && !c.error);
  const acceptedOps = new Set(succeeded.map(s => s.operation));
  const archiveDirectory = resolve(workspace, '.report-downloads');
  const archives = [];
  if (succeeded.some(s => s.operation === 'prepare_bundle_archive')) {
    for (const entry of await readdir(archiveDirectory)) {
      if (/^[a-f0-9]{64}\.zip$/.test(entry)) {
        const archivePath=resolve(archiveDirectory,entry);
        const receipt=await json(archivePath.replace(/\.zip$/,'.json'));
        if (receipt.artifact_id!==evaluation.artifact) continue;
        const archiveVerification=JSON.parse(execFileSync(python,
          ['-I','-B',resolve(project,'scripts/bench_pgac_verify.py'),artifact,workspace,selected,archivePath],
          {encoding:'utf8',windowsHide:true}));
        archives.push({path:'.report-downloads/' + entry, sha256:await fileHash(archivePath),
          isolated_process_verification:archiveVerification});
      }
    }
  }
  assert.ok(archives.length,'Prepare the selected delivery ZIP before independent verification');
  const independent=archives[0].isolated_process_verification;
  // ExternalExpertAdapter.act checks each Python decision twice in the same runtime
  // and once in a fresh module. These executions are real MPC work.
  assert.equal(collection.source.expert_backend, 'python_callable');
  assert.ok(collection.trajectories.every(t => t.status === 'completed'));
  const expertTestSteps = report.closed_loop.test.expert.report.valid_rows;
  const expertEvaluationSteps = report.closed_loop.validation.expert.report.valid_rows + expertTestSteps;
  const timing = report.controller_benchmark;
  // The published Python benchmark pairs the expert once with each student. Each
  // pair includes one preflight per distinct timing context plus warmup and repeats.
  const timingPairs = timing.status === 'completed' ? 2 : 0;
  const timingSourceCalls = timingPairs * (Math.min(expertTestSteps, timing.settings?.repeats ?? 0)
    + (timing.settings?.warmup ?? 0) + (timing.settings?.repeats ?? 0));
  const published = {
    started: observed.started, session_id: observed.sessionId, role: observed.role, provider: observed.provider,
    model: observed.model, workspace: basename(workspace), user_request: observed.turns[0],
    additional_operator_messages: observed.turns.slice(1).filter(t => !t.startsWith('已核对并确认')
      && !observed.postDeliveryQuestions?.some(q => q.question === t)),
    completed_turns: observed.ended, final_turn_reason: observed.lastReason,
    final_answer_sha256:hash(observed.lastAnswer),
    post_delivery_questions:(observed.postDeliveryQuestions ?? []).map(({turn,priorAnswer,question}) =>
      ({turn, prior_answer_sha256:hash(priorAnswer), original_accounting_excerpt:priorAnswer.split('\n').filter(s => s.includes('MPC 求解')),
        delivery_answer_excerpt:priorAnswer.split('\n').filter(s=>s.includes('交付 PGAC') || s.includes('盲测等权') || s.includes('动作 RMS')),
        question})),
    final_answer_accounting_excerpt:observed.lastAnswer.split('\n').filter(s => s.includes('执行次数') || s.includes('执行合计')),
    final_archive_check_excerpt:observed.lastAnswer.split('\n').filter(s=>s.startsWith('结论') && s.includes('缓存')),
    loaded_skills: skillCalls.map(({seq, skill}) => ({seq, skill})),
    confirmed_cards: {source:'test_fixture', count:observed.confirmations.length,
      prepared:confirmations.map(({seq, operation, config_file, card_id}) =>
        ({seq, operation, config_file, received_confirmation:observed.confirmations.includes(card_id)}))},
    role_tool_calls: observed.calls.filter(c => c.name === 'control_lightweight' && acceptedOps.has(c.operation))
      .map(({seq, operation, config_file, error}) => ({seq, operation, config_file, error})),
    completed_operations: succeeded.map(({operation, job, artifact, status}) => ({operation, job, artifact, status})),
    failed_attempts: observed.steps.filter(s => s.state === 'failed').map(({operation, job}) => ({operation, job})),
    expert_work: {accounting:'Successful Python expert decisions are checked by two calls in the live module and one in a fresh module. The timing harness invokes the raw function directly.',
      collection_decisions:collection.valid_rows, collection_function_calls:collection.expert.source_invocations_this_adapter ?? 3 * collection.valid_rows,
      train_decisions:collection.split.sample_counts.train, train_function_calls:3 * collection.split.sample_counts.train,
      evaluation_decisions:expertEvaluationSteps, evaluation_function_calls:3 * expertEvaluationSteps,
      timing_pairs:timingPairs, timing_raw_function_calls:timingSourceCalls, additional_pgac_training_calls:candidate.replay.extra_mpc_calls},
    training: {best_epoch:candidate.best_epoch, trained_epochs:candidate.trained_epochs,
      optimizer_updates:candidate.optimizer_updates, initial_action_parity:candidate.initial_action_parity, replay:candidate.replay},
    final_acceptance: report.final_acceptance,
    ...(report.engineering_objective ? {engineering_objective: report.engineering_objective} : {}),
    test: Object.fromEntries(Object.entries(report.closed_loop.test).map(([version, score]) =>
      [version, {J:score.quadratic.score, feasible:score.quadratic.feasible,
        ...(score.quadratic.tracking ? {tracking:score.quadratic.tracking} : {})}])),
    isolated_process_verification: independent,
    package: {path:'artifacts/' + evaluation.artifact + '/bundle', model_sha256:await fileHash(resolve(artifact, 'bundle/model.pt')),
      report_sha256:await fileHash(resolve(artifact, 'report.json')), archives},
    pinned_fixture_sha256: observed.pinned
  };
  const output = resolve(root, basename(workspace) + '-verified.json');
  await writeSmokeEvidence(output, published);
  console.log(JSON.stringify({output, ...independent}));
}
if (values.action === 'verify') {
  await verify();
} else {
  if (values.action === 'start') {
    await start();
  } else {
    assert.ok(['inspect', 'confirm', 'reply', 'drive'].includes(values.action));
    record = await json(evidence);
    assert.equal(record.workspace, workspace);
    do {
      await observe();
      await writeSmokeEvidence(evidence, record);
      if (values.action !== 'drive' || record.ended >= record.turns.length) break;
      await sleep(4000);
    } while (true);
  }
  await writeSmokeEvidence(evidence, record);
  console.log(JSON.stringify({sessionId:record.sessionId, ended:record.ended, turns:record.turns.length,
    calls:record.calls?.slice(-5), steps:record.steps?.slice(-3), lastReason:record.lastReason, providerError:record.providerError,
    ...(record.ended===record.turns.length ? {answer:record.lastAnswer?.slice(-550)} : {})}));
}
