/** Native role composition, exact artifact relay and bounded report/file operations. No LLM API. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { connectHarness, command, runStage, smokeOptions, writeSmokeEvidence } from './harness-rpc.mjs';

const options = smokeOptions('smoke-role-handoff', { requireWorkspace: true });
if (options.help) process.exit(0);
const workspace = options.workspace;
const manifest = JSON.parse(await readFile(resolve(workspace, 'manifest.json'), 'utf8'));
for (const kind of ['corrected_model', 'reference_model', 'controller', 'expert_dataset', 'imitation_policy']) {
  const id = manifest.active?.[kind];
  assert.ok(id && manifest.artifacts?.[id]?.status === 'verified', `Explicit workspace lacks a verified ${kind}; generate and verify its fixture first.`);
}
const rpc = await connectHarness({ logPath: options.webLog });
const roster = await rpc('agentPresets/list', {});
const evidence = { checkedAt: new Date().toISOString(), workspace, roles: {}, relay: [], scope: 'Real native presets, sessions, commands and numerical subprocesses over an explicitly selected verified fixture; no model API call.' };
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
  if (kind === 'corrected_model') {
    assert.equal(plan.operation, 'synthesize_mpc');
    assert.deepEqual(plan.inputs, { model: artifact_id });
    assert.equal(result.result.optional_handoffs.find(item => item.operation === 'tune_controller').inputs.evaluation_model, manifest.active.reference_model);
  }
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
evidence.status = 'PASS';
await writeSmokeEvidence(options.evidence, evidence);
console.log(JSON.stringify({status:evidence.status,evidence:options.evidence,workspace,roles:evidence.roles,relay:evidence.relay.map(row=>({kind:row.kind,status:row.status,jobId:row.jobId})),reportJobId:inspected.jobId,previewJobId:preview.jobId}));
