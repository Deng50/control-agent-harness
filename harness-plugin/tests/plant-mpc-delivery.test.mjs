/** B03 原始输入，真实 Host 命令、隔离 Session、测试来源确认及独立 Python 子进程。 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {mkdir,mkdtemp,readFile,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {createHost,project} from './history-host.mjs';

const materials=resolve(process.env.CONTROL_BENCHMARK_ROOT ?? join(project,'../control-agent-benchmark'),'cases/B03_mpc_preferences/inputs');
const json=async path=>JSON.parse(await readFile(path,'utf8'));
const unpack=result=>{assert.equal(result.kind,'success',result.text);return JSON.parse(result.text);};
const output=join(project,'test-output/p0c-native');

test('B03 callable → confirmed conversion → pinned transfer → MPC → independent delivery', {
  timeout:300000,skip:!existsSync(join(materials,'plant_numpy.py')) && 'B03 benchmark inputs unavailable',
},async t=>{
  await mkdir(output,{recursive:true});
  const source=await mkdtemp(join(output,'plant-')), target=await mkdtemp(join(output,'controller-'));
  const setup=async (workspace,owner)=>{
    const host=await createHost(workspace,owner,{confirmFixturePlans:false});
    t.after(()=>host.close());
    await mkdir(join(workspace,'.case-plans'),{recursive:true});
    await writeFile(join(workspace,'.case-plans/test-fixture.json'),JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace}));
    const call=(cmd,args)=>host.command(cmd,{workspace,...args});
    const submit=async (agent,operation,inputs,config={},review=false)=>{
      const request={agent,operation,inputs,config};
      if(review) {
        const card=unpack(await call('control-prepare-review',request));
        assert.equal(unpack(await call('control-confirm',{card_id:card.card_id})).confirmation_source,'test_fixture');
        request.card_id=card.card_id;
      }
      const job=unpack(await call('control',request));
      const end=await host.ctx.jobs.wait(job.jobId,75000,host.agent.id);
      assert.equal(end.status,'completed',end.detail);
      return unpack(await call('control-status',{attemptId:job.attemptId})).record.result;
    };
    return {call,submit,registered:()=>host.agent.session.snapshotEvents().filter(e=>e.type==='job/registered').length};
  };
  const model=await setup(source,'p0c-model'), controller=await setup(target,'p0c-controller');
  const material=await model.submit('model','register_material',{local_path:join(materials,'plant_numpy.py')});
  const plant=(await model.submit('model','register_plant',{material:material.artifact_id},await json(join(materials,'plant_adapter.json')))).artifact_id;
  const data=(await model.submit('model','sample_model',{model:plant},{...await json(join(materials,'identification_sampling_config.json')),sampling_run:'full'},true)).artifact_id;
  const bad={agent:'controller',operation:'synthesize_mpc',inputs:{model:plant}};
  const count=model.registered();
  assert.equal(unpack(await model.call('control-preflight',bad)).status,'unsupported');
  assert.match((await model.call('control',bad)).text,/convert_plant_predictor/);
  const gap=unpack(await model.call('control-preflight',{agent:'model',operation:'convert_plant_predictor',inputs:{plant,dataset:data},config:{method:'declared_equations'}}));
  assert.ok(gap.issues.some(i=>i.field==='inputs.equations' && i.status==='clarify'));
  assert.equal(model.registered(),count);
  const predictor=(await model.submit('model','convert_plant_predictor',{plant,dataset:data},{method:'sampled_affine'},true)).artifact_id;
  const handoff=await model.submit('model','prepare_handoff',{artifact_id:predictor});
  const transfer=handoff.transfer;
  assert.equal(transfer.artifact_id,predictor);
  const transferProbe={agent:'controller',operation:'import_artifact',inputs:{artifact_path:transfer.artifact_path,expected_metadata_hash:'0'.repeat(64)}};
  const beforeImport=controller.registered();
  assert.equal(unpack(await controller.call('control-preflight',transferProbe)).status,'stale_dependency');
  assert.equal((await controller.call('control',transferProbe)).kind,'error');
  assert.equal(controller.registered(),beforeImport);
  const imported=await controller.submit('controller','import_artifact',{artifact_path:transfer.artifact_path,expected_artifact_id:predictor,
    expected_metadata_hash:transfer.metadata_hash,expected_project_id:transfer.project_id});
  assert.equal(imported.artifact_id,predictor);
  const design=await controller.submit('controller','synthesize_mpc',{model:predictor},
    {prediction:{method:'direct'},controller:{horizon:4,max_iterations:50,solver_tolerance:1e-12},weights:{q:1,r:.08,s:.04}},true);
  assert.equal(design.delivery.checks.standalone,'not_run');
  // 用 benchmark 给出的独立离散矩阵求解这个线性二次问题。该点全部约束均不活跃。
  const conditions=await json(join(materials,'comparison_conditions.json')), row=conditions.tuning_scenarios[0];
  const point={x:row.initial_state,reference:row.reference,disturbance:row.disturbance,previous:[0]};
  assert.deepEqual([point.x,point.disturbance],[[0,0],[0]]);
  const oracle=`import json,sys,numpy as np
m=json.load(open(sys.argv[1],encoding="utf-8"))
a,b,c=map(np.asarray,(m["A"],m["B"],m["C"]))
n=4
g=np.array([[(c@np.linalg.matrix_power(a,i-j)@b).item() if j<=i else 0. for j in range(n)] for i in range(n)])
h=np.eye(n)-np.eye(n,k=-1)
s=m["interface"];lo,hi=lambda k:s[k][0]["lower"],lambda k:s[k][0]["upper"]
factor=((hi("outputs")-lo("outputs"))/(hi("controls")-lo("controls")))**2
u=np.linalg.solve(g.T@g+factor*(.08*np.eye(n)+.04*h.T@h), g.T@np.full(n,float(sys.argv[2])))
x=np.zeros(2)
for v in u:
    assert lo("controls")<v<hi("controls")
    x=a@x+b.ravel()*v
    assert all(p["lower"]<z<p["upper"] for p,z in zip(s["states"],x))
    assert lo("outputs")<(c@x).item()<hi("outputs")
print(float(u[0]))`;
  const python=resolve(project,process.platform==='win32'?'.venv/Scripts/python.exe':'.venv/bin/python');
  const expected=Number(execFileSync(python,['-I','-c',oracle,join(materials,'declared_discrete_model.json'),String(point.reference[0])],{encoding:'utf8'}));
  assert.ok(expected>0);
  const deliver=await controller.submit('controller','verify_design_delivery',{design:design.artifact_id},
    {target:'workspace_controller',case:point,expected_action:[expected],action_atol:1e-4},true);
  assert.equal(deliver.kind,'controller');
  for(const key of ['files','load','standalone','single_point','workspace_controller']) assert.equal(deliver.delivery.checks[key],'passed');
  assert.equal(deliver.delivery.checks.closed_loop,'not_run');
  const partial=await controller.submit('controller','verify_design_delivery',{design:design.artifact_id},
    {target:'workspace_controller',case:point,required_checks:['closed_loop']},true);
  assert.equal(partial.kind,'delivery_validation');
  assert.equal(partial.delivery.status,'partial');
  assert.equal(partial.delivery.request_satisfied,false);
  assert.ok(partial.delivery.missing_requirements.includes('closed_loop'));
  const downstream=await controller.submit('lightweight','prepare_handoff',{artifact_id:deliver.artifact_id});
  assert.equal(downstream.transfer.artifact_id,deliver.artifact_id);
  const cases=[...conditions.tuning_scenarios,...conditions.verification_scenarios,...conditions.test_scenarios].slice(0,5).map((s,i)=>({...s,name:'p0c-original-'+i,steps:2}));
  const collection=await controller.submit('lightweight','collect_expert',{controller:deliver.artifact_id,evaluation_model:plant},{scenarios:cases,seed:42},true);
  assert.equal(collection.kind,'expert_dataset');
  const collected=await json(join(collection.artifact_path,'artifact.json'));
  assert.equal(collected.source.evaluation_model,plant);
  assert.equal(collected.validation.samples,10);
  t.diagnostic(JSON.stringify({source,target,expected,partial:partial.artifact_id,expert_dataset:collection.artifact_id,plant,dataset:data,predictor,design:design.artifact_id,controller:deliver.artifact_id,
    delivery:join(deliver.artifact_path,'report.json')}));
});
