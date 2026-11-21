import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mkdir,mkdtemp,writeFile} from 'node:fs/promises';
import {join,resolve} from 'node:path';
import {stagePreflight,verifyPublication} from '../src/case-requirements.mjs';

const hash=text=>createHash('sha256').update(text).digest('hex');
const base=resolve(import.meta.dirname,'../../test-output/contract-preflight');

test('an indexed kind cannot replace the predictor representation or an intact payload',async()=>{
  await mkdir(base,{recursive:true});
  const workspace=await mkdtemp(join(base,'model-')),id='nominal_model-fixture',path='artifacts/'+id;
  const dir=join(workspace,path), model='{"representation":"unknown"}';
  await mkdir(dir,{recursive:true});
  await writeFile(join(dir,'model.json'),model);
  const meta=JSON.stringify({id,kind:'nominal_model',validation:{status:'verified'},parents:[],interface:{},files:{'model.json':hash(model)}});
  await writeFile(join(dir,'artifact.json'),meta);
  await writeFile(join(workspace,'manifest.json'),JSON.stringify({artifacts:{[id]:{kind:'nominal_model',status:'verified',path,metadata_file_sha256:hash(meta)}}}));
  const request={workspace,operation:'synthesize_mpc',inputs:{model:id}};
  assert.equal((await stagePreflight(request)).status,'unsupported');
  await writeFile(join(dir,'model.json'),'{}');
  const stale=await stagePreflight(request);
  assert.equal(stale.status,'stale_dependency');
  assert.equal(stale.issues[0].field,'inputs.model');
  assert.equal(stale.jobs_started,0);
});

test('missing output, unregistered path and unrecorded positive delivery fail publication',async()=>{
  await mkdir(base,{recursive:true});
  const workspace=await mkdtemp(join(base,'output-')),id='controller_design-fixture',path='artifacts/'+id;
  const dir=join(workspace,path);
  await mkdir(dir,{recursive:true});
  const meta=JSON.stringify({id,files:{'report.json':hash('{}')}});
  await writeFile(join(dir,'artifact.json'),meta);
  await writeFile(join(workspace,'manifest.json'),JSON.stringify({artifacts:{[id]:{kind:'controller_design',status:'verified',path,metadata_file_sha256:hash(meta)}}}));
  const result={artifact_path:dir,artifact_id:id,kind:'controller_design',status:'verified',delivery:{artifact_id:id}};
  await assert.rejects(verifyPublication(workspace,result));
  await writeFile(join(dir,'report.json'),'{}');
  await verifyPublication(workspace,result);
  await assert.rejects(verifyPublication(workspace,{...result,artifact_path:workspace}));
  await assert.rejects(verifyPublication(workspace,{...result,delivery:{...result.delivery,target:'standalone_python',request_satisfied:true}}));
});
