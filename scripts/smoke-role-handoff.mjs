/** Native role composition, exact artifact relay and bounded report/file operations. No LLM API. */
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { connectHarness, command, runStage, project } from './harness-rpc.mjs';

const rpc = await connectHarness();
const roster = await rpc('agentPresets/list', {});
const evidence = { checkedAt: new Date().toISOString(), roles: {}, relay: [], scope: 'Real native presets, sessions, commands and numerical subprocesses; no model API call.' };
const workspace = resolve(project, 'test-output/harness-nonlinear-809e8b2c-f74c-45f1-957a-ddbd0540b8d8');
const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
for (const role of ['model','controller','lightweight']) {
  const id = `control-${role}`;
  const preset = roster.presets.find(item => item.id === id);
  assert.ok(preset, `Missing preset ${id}`);
  assert.equal(preset.broken, undefined, `Preset activation failed: ${preset.broken}`);
  const created = await rpc('session/create', { request: { cwd: workspace, agentPreset: id } });
  assert.equal(created.agentPreset, id);
  const help = await command(rpc, created.sessionId, `/control-help ${JSON.stringify({agent:role})}`);
  assert.ok(help.operations.some(item => item.operation === 'recover_interrupted_stage'));
  evidence.roles[role] = created;
}
for (const [kind, role] of [['corrected_model','model'],['controller','controller'],['expert_dataset','lightweight'],['imitation_policy','lightweight']]) {
  const artifact_id = manifest.active[kind];
  assert.ok(artifact_id);
  const result = await runStage(rpc, evidence.roles[role].sessionId, {agent:role,operation:'prepare_handoff',workspace,inputs:{artifact_id}});
  const plan = result.result.handoff;
  if (kind === 'corrected_model') assert.equal(plan.inputs.evaluation_model, manifest.active.reference_model);
  if (kind === 'controller') assert.equal(plan.inputs.evaluation_model, manifest.active.reference_model);
  if (kind === 'imitation_policy') assert.equal(plan.inputs.model, manifest.active.corrected_model);
  evidence.relay.push({artifact_id,jobId:result.jobId,...result.result});
}
const inspected = await runStage(rpc,evidence.roles.controller.sessionId,{agent:'controller',operation:'inspect_artifact',workspace,inputs:{artifact_id:manifest.active.controller}});
assert.ok(inspected.result.report.test.runs[0].trajectory.length > 1);
const preview = await runStage(rpc,evidence.roles.controller.sessionId,{agent:'controller',operation:'read_artifact_file',workspace,inputs:{artifact_id:manifest.active.controller,file:'controller.json'}});
assert.equal(JSON.parse(preview.result.file_preview.text).prediction_model, manifest.active.corrected_model);
evidence.reportJobId = inspected.jobId;
evidence.previewJobId = preview.jobId;
await mkdir(resolve(project,'docs'),{recursive:true});
await writeFile(resolve(project,'docs/harness-role-handoff-evidence.json'),JSON.stringify(evidence,null,2));
console.log(JSON.stringify({roles:evidence.roles,relay:evidence.relay.map(row=>({kind:row.kind,status:row.status,jobId:row.jobId})),reportJobId:inspected.jobId,previewJobId:preview.jobId}));
