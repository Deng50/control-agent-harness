import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,mkdtemp,readdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {reviewControlRequest} from '../src/review.mjs';

const output=resolve(import.meta.dirname,'../../test-output/review-command-tests');
await mkdir(output,{recursive:true});

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
