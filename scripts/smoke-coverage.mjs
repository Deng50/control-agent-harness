/** Real native artifact import, residual training and training-only applicability report. */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { connectHarness, runStage, project } from './harness-rpc.mjs';

const rpc=await connectHarness();
const source=resolve(project,'test-output/harness-nonlinear-809e8b2c-f74c-45f1-957a-ddbd0540b8d8');
const manifest=JSON.parse(await readFile(resolve(source,'manifest.json'),'utf8'));
const dataset=manifest.active.transition_dataset, nominal=manifest.active.nominal_model;
const workspace=resolve(project,'test-output',`harness-coverage-${randomUUID()}`);
await mkdir(workspace,{recursive:true});
if(process.platform==='win32')execFileSync('pwsh.exe',['-NoProfile','-File',resolve(project,'scripts/prepare-workspace.ps1'),'-Prepare', '-Workspace',workspace]);
const created=await rpc('session/create',{request:{cwd:workspace,agentPreset:'control-model'}});
const imported=await runStage(rpc,created.sessionId,{agent:'model',operation:'import_artifact',workspace,inputs:{artifact_path:resolve(source,manifest.artifacts[dataset].path)}});
assert.equal(imported.result.artifact_id,dataset);
const trained=await runStage(rpc,created.sessionId,{agent:'model',operation:'train_residual',workspace,inputs:{nominal_model:nominal,dataset},config:{epochs:25,hidden_dims:[6],batch_size:16,patience:20,learning_rate:.003,seed:42}});
assert.equal(trained.result.status,'verified');
const inspected=await runStage(rpc,created.sessionId,{agent:'model',operation:'inspect_artifact',workspace,inputs:{artifact_id:trained.result.artifact_id}});
const coverage=inspected.result.report.applicability;
assert.equal(coverage.source,'training_partition_only');
assert.equal(coverage.marginal_ranges.length,4);
assert.equal(coverage.extrapolation_validated,false);
assert.ok(coverage.samples>0);
const evidence={checkedAt:new Date().toISOString(),workspace,...created,importJobId:imported.jobId,trainingJobId:trained.jobId,reportJobId:inspected.jobId,artifact_id:trained.result.artifact_id,coverage,test:inspected.result.report.test,scope:'Actual native import of exact external dependency closure, CPU training and inspected saved report. No model API call.'};
await writeFile(resolve(project,'docs/harness-coverage-evidence.json'),JSON.stringify(evidence,null,2));
console.log(JSON.stringify(evidence));
