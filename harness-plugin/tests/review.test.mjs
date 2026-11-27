import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,readdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {prepareExecution,reviewControlRequest} from '../src/review.mjs';

const output=resolve(import.meta.dirname,'../../test-output/review-command-tests');
await mkdir(output,{recursive:true});

test('lightweight file settings resolve to the current exact config at review and admission',async()=>{
  const workspace=await mkdtemp(resolve(output,'settings-'));
  const policy={mode:'workspace-write',workspaceRoot:workspace};
  const request={agent:'lightweight',operation:'register_material',workspace,inputs:{local_path:resolve(workspace,'plant.py')}};
  await writeFile(request.inputs.local_path,'def step(x): return x\n');
  const file=resolve(workspace,'settings.json');
  for(const label of ['initial','updated']){
    const config={name:label,source_kind:'path'};
    await writeFile(file,JSON.stringify(config));
    assert.deepEqual(await prepareExecution({...request,config_file:'settings.json'},policy,{structured:true}),
      await prepareExecution({...request,config},policy,{structured:true}));
  }
  for(const bad of [{config_file:'../settings.json'},{config_file:'settings.json',config:{}},{agent:'model',config_file:'settings.json'}])
    await assert.rejects(prepareExecution({...request,...bad},policy));
  for(const value of [[],{plan_refs:{}},{unknown_key:1}]){
    await writeFile(file,JSON.stringify(value));
    await assert.rejects(prepareExecution({...request,config_file:'settings.json'},policy));
  }
});

test('review returns the actual branch scopes and exact editable config without jobs, plans or ledgers',async()=>{
  const workspace=await mkdtemp(resolve(output,'requirements-'));
  const policy={mode:'workspace-write',workspaceRoot:workspace};
  const request={agent:'controller',operation:'synthesize_mpc',workspace,inputs:{description:'二阶弹簧系统'},config:{}};
  const draft=await reviewControlRequest(request,policy);
  assert.deepEqual(draft.requirements,[]);
  const parameters={mass:1,damping:.5,stiffness:2,sample_time:.1,state_bounds:[[-2,2],[-3,3]],input_bounds:[[-1,1]],output_bounds:[[-2,2]]};
  const numeric=await reviewControlRequest({...request,config:{parameters}},policy);
  assert.deepEqual(numeric.requirements.map(row=>row.scope),['signals']);
  assert.deepEqual(numeric.requirements[0].expectedPayload,{inputs:request.inputs,config:{parameters}});
  assert.equal(numeric.jobs_started,0);assert.equal(numeric.authority,'read_only_requirements_not_confirmation');
  assert.deepEqual(await readdir(workspace),[]);
  const changed=await reviewControlRequest({...request,config:{parameters:{...parameters,mass:2}}},policy);
  assert.notDeepEqual(changed.requirements,numeric.requirements);
  assert.deepEqual(await readdir(workspace),[]);
});

test('review rejects wrong roles, malformed request objects and out-of-root execution targets without writes',async()=>{
  const workspace=await mkdtemp(resolve(output,'denied-'));
  const policy={mode:'workspace-write',workspaceRoot:workspace};
  for(const request of [null,[],{agent:'model',operation:'tune_controller',workspace},{agent:'model',operation:'inspect_workspace',workspace,config:[]},
    {agent:'model',operation:'inspect_workspace',workspace:output}]){
    await assert.rejects(reviewControlRequest(request,policy));
  }
  assert.deepEqual(await readdir(workspace),[]);
});
