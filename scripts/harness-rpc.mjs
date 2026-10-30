/** Authenticated native RPC for local acceptance scripts; never prints the startup token. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';

export const project = resolve(import.meta.dirname, '..');
export async function connectHarness() {
  const log = await readFile(resolve(project, 'harness-web.log'), 'utf8');
  const startup = log.match(/dsh web: (http[^\s]+)/)?.[1];
  assert.ok(startup, 'Harness Web has not emitted its startup URL');
  const url = new URL(startup);
  const response = await fetch(url, { redirect: 'manual', signal: AbortSignal.timeout(10000) });
  const cookie = response.headers.getSetCookie().map(item => item.split(';')[0]).join('; ');
  assert.ok(cookie, 'Harness authentication did not issue a cookie');
  return async (method, payload) => {
    const response = await fetch(new URL(`/api/${method}`, url), {
      method: 'POST', headers: { cookie, origin: url.origin, 'content-type': 'application/json' },
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
