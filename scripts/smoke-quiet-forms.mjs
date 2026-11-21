/** Native form jobs must finish without opening an unsolicited model turn. */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { connectHarness, runStage, project } from './harness-rpc.mjs';

const rpc = await connectHarness();
const evidence = { checkedAt: new Date().toISOString(), scope: 'Real native form commands and job settlement; no model call is needed.', roles: [] };
for (const role of ['model', 'controller', 'lightweight']) {
  const workspace = resolve(project, 'test-output', `quiet-form-${role}-${randomUUID()}`);
  await mkdir(workspace, { recursive: true });
  if (process.platform === 'win32') execFileSync('pwsh.exe', ['-NoProfile', '-File', resolve(project, 'scripts/prepare-workspace.ps1'), '-Prepare', '-Workspace', workspace]);
  const { sessionId } = await rpc('session/create', { request: { cwd: workspace, agentPreset: `control-${role}` } });
  const started = await runStage(rpc, sessionId, { agent: role, operation: 'inspect_workspace', workspace });
  const page = await rpc('session/events', { request: { address: { kind: 'session', sessionId }, maxMessages: 100 } });
  const types = [...new Set(page.records.map(row => row.event.type))];
  assert.ok(types.includes('job/settled'), 'The real native job has not settled');
  assert.ok(!types.some(type => type.startsWith('turn/')), 'A form completion opened an unsolicited model turn');
  evidence.roles.push({ role, sessionId, workspace, jobId: started.jobId, passed: true, eventTypes: types });
  console.log(JSON.stringify({ role, passed: true, modelTurns: 0 }));
}
await writeFile(resolve(project, 'docs/harness-quiet-forms-evidence.json'), JSON.stringify(evidence, null, 2));
