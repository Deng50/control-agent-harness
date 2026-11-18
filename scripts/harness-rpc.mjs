/** Authenticated native RPC for local acceptance scripts; never prints the startup token. */
import assert from 'node:assert/strict';
import { mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { execFileSync } from 'node:child_process';
import { parseArgs } from 'node:util';

export const project = resolve(import.meta.dirname, '..');
export function smokeOptions(name, { requireWorkspace = false } = {}) {
  const { values } = parseArgs({ options: {
    workspace: { type: 'string' }, evidence: { type: 'string' },
    'web-log': { type: 'string' }, help: { type: 'boolean', short: 'h' },
  } });
  if (values.help) {
    console.log(`node scripts/${name}.mjs --workspace <absolute-directory> --evidence <json-path> [--web-log <log-path>]`);
    console.log(requireWorkspace ? 'An explicit existing artifact workspace is required (or CONTROL_SMOKE_WORKSPACE).' : 'Omitting workspace creates a fresh test-output workspace; an empty workspace is supported.');
    if (name === 'smoke-model-dialogue') console.log('Windows project-local test directories use prepare-workspace.ps1 without taking ownership. Explicit external directories must already exist and be prepared by the caller; their permissions are not changed.');
    return { help: true };
  }
  const requested = values.workspace ?? process.env.CONTROL_SMOKE_WORKSPACE;
  for (const [key, value] of Object.entries({ workspace: requested, evidence: values.evidence ?? process.env.CONTROL_SMOKE_EVIDENCE,
    webLog: values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG })) {
    assert.ok(value === undefined || value.trim().length > 0, `${key} must not be blank`);
  }
  assert.ok(!requireWorkspace || requested, 'Pass --workspace or CONTROL_SMOKE_WORKSPACE; historical workspaces are never selected automatically.');
  return {
    workspace: resolve(requested ?? resolve(project, 'test-output', `${name}-${randomUUID()}`)),
    evidence: resolve(values.evidence ?? process.env.CONTROL_SMOKE_EVIDENCE ?? resolve(project, 'test-output/smoke-evidence', `${name}-${randomUUID()}.json`)),
    webLog: resolve(values['web-log'] ?? process.env.CONTROL_HARNESS_WEB_LOG ?? resolve(project, 'harness-web.log')),
  };
}

/** Prepare only a caller-owned project descendant; external directories are never ACL-ed. */
export async function prepareSmokeWorkspace(directory) {
  const workspace = resolve(directory);
  const inside = (base, target) => {
    const path = relative(base, target);
    return path.length > 0 && path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path);
  };
  if (!inside(project, workspace)) {
    assert.ok((await stat(workspace)).isDirectory(), 'Explicit external workspace must already exist and be prepared by the caller.');
    return { workspace, projectLocal: false, aclPrepared: false };
  }
  const physicalProject = await realpath(project);
  let ancestor = workspace, physicalAncestor;
  while (physicalAncestor === undefined) {
    try { physicalAncestor = await realpath(ancestor); }
    catch (error) {
      if (error.code !== 'ENOENT') throw error;
      const parent = dirname(ancestor);
      assert.notEqual(parent, ancestor, 'Unable to resolve workspace ancestor');
      ancestor = parent;
    }
  }
  assert.ok(physicalAncestor === physicalProject || inside(physicalProject, physicalAncestor), 'Project-local workspace ancestor resolves outside the project; no directory or permissions changed.');
  await mkdir(workspace, { recursive: true });
  assert.ok(inside(physicalProject, await realpath(workspace)), 'Project-local workspace resolves outside the project; no permissions changed.');
  if (process.platform === 'win32') execFileSync('pwsh.exe', ['-NoProfile', '-File', resolve(project, 'scripts/prepare-workspace.ps1'), '-Workspace', workspace], { windowsHide: true });
  return { workspace, projectLocal: true, aclPrepared: process.platform === 'win32' };
}

export async function writeSmokeEvidence(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`);
}

export async function connectHarness({ logPath = process.env.CONTROL_HARNESS_WEB_LOG ?? resolve(project, 'harness-web.log'), connectionClose = false } = {}) {
  assert.equal(typeof connectionClose,'boolean');
  // A synchronous independent Python audit can outlast the Host keep-alive.
  // Callers can avoid stale pooled sockets without retrying mutating RPCs.
  const connectionHeaders = connectionClose ? { connection:'close' } : {};
  const log = await readFile(resolve(logPath), 'utf8');
  // A log may retain more than one launch; never authenticate against the first, stale URL.
  const startup = [...log.matchAll(/dsh web: (http[^\s]+)/g)].at(-1)?.[1];
  assert.ok(startup, 'Harness Web has not emitted its startup URL');
  const url = new URL(startup);
  assert.ok(['localhost', '127.0.0.1', '[::1]'].includes(url.hostname), 'Acceptance RPC only connects to a local Harness Host.');
  const response = await fetch(url, { redirect: 'manual', headers:connectionHeaders, signal: AbortSignal.timeout(10000) });
  const cookie = response.headers.getSetCookie().map(item => item.split(';')[0]).join('; ');
  assert.ok(cookie, 'Harness authentication did not issue a cookie');
  return async (method, payload) => {
    const response = await fetch(new URL(`/api/${method}`, url), {
      method: 'POST', headers: { cookie, origin: url.origin, 'content-type': 'application/json', ...connectionHeaders },
      body: JSON.stringify({ type: 'client-request', rpcId: randomUUID(), method, payload: { args: payload } }),
      signal: AbortSignal.timeout(30000),
    });
    assert.equal(response.status, 200, `Harness RPC HTTP status ${response.status}`);
    const envelope = await response.json();
    const result = envelope.result ?? envelope;
    assert.ok(result.ok, JSON.stringify(result.error));
    return result.value;
  };
}

export async function command(rpc, sessionId, line) {
  const response = await rpc('commands/execute', { agentId: sessionId, line, submittedAttachments: [] });
  assert.equal(response.result.kind, 'success', response.result.text);
  return JSON.parse(response.result.text);
}

export async function runStage(rpc, sessionId, request) {
  const started = await command(rpc, sessionId, `/control ${JSON.stringify(request)}`);
  const deadline = Date.now() + 120000;
  while (Date.now() < deadline) {
    const state = await command(rpc, sessionId, `/control-status ${started.jobId}`);
    if (['completed','failed','killed'].includes(state.job.status)) {
      assert.equal(state.job.status, 'completed', JSON.stringify(state));
      const text = state.output.chunks.map(chunk => chunk.text).join('');
      const result = text.split('\n').flatMap(line => { try { return [JSON.parse(line)]; } catch { return []; } }).findLast(row => row.event === 'result')?.result;
      assert.ok(result, 'Native job did not emit a domain result');
      return { ...started, result };
    }
    await new Promise(resolve => setTimeout(resolve, 400));
  }
  throw new Error(`Native job still has no terminal result: ${started.jobId}; inspect that job before retrying`);
}
