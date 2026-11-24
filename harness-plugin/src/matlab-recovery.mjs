import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { readMcpReply, forgetMcpReply } from './matlab-mcp-reply.mjs';

// The caller that timed out may have gone away. A Host-owned verifier holds
// the same execution mutex as ordinary callers and never resubmits their code.
export function installMatlabRecovery(ctx, resource, verifyTerminal) {
  const stop = new AbortController();
  const stateFile = join(resource.config.directory, 'resource.json');
  let inFlight, observedLegacy, lastDiagnostic, ownerToken, nextOwnerCheck = 0;

  async function pending({inspectOwner = false} = {}) {
    try {
      const state = JSON.parse(await readFile(stateFile, 'utf8'));
      if (state.quarantine || !state.owner) return state.quarantine ?? null;
      // A crashed broker cannot write a pending record. Inspect long-lived
      // owners under the metadata mutex; the broker checks OS process birth.
      if (ownerToken !== state.owner.token) {
        ownerToken = state.owner.token;
        nextOwnerCheck = Date.now() + 15000;
      }
      if ((inspectOwner || Date.now() >= nextOwnerCheck) && !stop.signal.aborted) {
        nextOwnerCheck = Date.now() + 15000;
        const broker = resource.spawnBroker();
        try { return (await broker.finish({command:'inspect'}, 'state')).quarantine; }
        finally { if (!broker.handle.stdin.writableEnded) broker.handle.stdin.end(); }
      }
      return null;
    }
    catch (error) { if (error.code === 'ENOENT') return null; throw error; }
  }

  async function reconcile() {
    const record = await pending();
    if (!record || stop.signal.aborted || !ctx.tools.get('mcp__matlab__evaluate_matlab_code')) return;
    if (record.token === observedLegacy) return;
    const broker = resource.spawnBroker();
    const signal = AbortSignal.any([stop.signal, AbortSignal.timeout(45000)]);
    const endInput = () => broker.handle.stdin.end();
    signal.addEventListener('abort', endInput, {once:true});
    try {
      broker.send({command:'recovery-acquire', token:record.token});
      const lease = await broker.wait('recovery-acquired');
      signal.throwIfAborted();
      const wire = lease.completion_source === 'mcp_response';
      const reply = wire && await readMcpReply(resource.config.directory,lease,String(lease.label).split(':')[0]);
      if (wire && !reply) return;
      const idle = await resource.withRecoveryProbe(lease.probe_nonce, () => verifyTerminal({
        exec:{signal}, lease, probeNonce:lease.probe_nonce, completionNonce:wire ? undefined : lease.completion_nonce ?? undefined,
      }));
      signal.throwIfAborted();
      if (!lease.completion_nonce) {
        // An old ledger has no command completion marker. A successful idle
        // probe is diagnostic only; it cannot settle an unknown earlier write.
        observedLegacy = record.token;
        return;
      }
      const proof = wire ? {...reply,...idle} : idle;
      if (proof.completion_nonce !== lease.completion_nonce) return;
      await broker.finish({command:'recover', evidence:proof}, 'recovered');
      if (wire) await forgetMcpReply(resource.config.directory,lease.token);
      lastDiagnostic = undefined;
      ctx.logger?.info('已核验原 MATLAB 调用结束，专用会话已恢复。');
    } finally {
      signal.removeEventListener('abort', endInput);
      if (!broker.handle.stdin.writableEnded) endInput();
      await broker.done;
    }
  }

  function check() {
    if (!inFlight) inFlight = reconcile().catch(error => {
      if (!stop.signal.aborted && error.message !== lastDiagnostic) {
        lastDiagnostic = error.message;
        ctx.logger?.debug?.(`MATLAB recovery check: ${error.message}`);
      }
    }).finally(() => { inFlight = undefined; });
    return inFlight;
  }

  resource.recoveryEnabled = true;
  const timer = setInterval(check, 1500);
  timer.unref();
  ctx.on('dispose', () => {
    clearInterval(timer);
    stop.abort();
    return inFlight;
  });
  return {check, pending};
}
