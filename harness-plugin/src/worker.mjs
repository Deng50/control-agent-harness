/** Bounded global worker admission and JSON-lines execution over Harness subprocess handles. */
export class WorkerSlots {
  constructor(limit) {
    this.limit = limit;
    this.active = 0;
    this.waiting = [];
  }

  acquire(signal) {
    signal.throwIfAborted();
    if (this.active < this.limit) {
      this.active += 1;
      return Promise.resolve(this.releaseHandle());
    }
    return new Promise((resolve, reject) => {
      const entry = { resolve, reject, signal, onAbort: null };
      entry.onAbort = () => {
        this.waiting = this.waiting.filter(item => item !== entry);
        reject(signal.reason);
      };
      this.waiting.push(entry);
      signal.addEventListener('abort', entry.onAbort, { once: true });
    });
  }

  releaseHandle() {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      const next = this.waiting.shift();
      if (next) {
        next.signal.removeEventListener('abort', next.onAbort);
        next.resolve(this.releaseHandle());
      } else {
        this.active -= 1;
      }
    };
  }
}

/** Returns synchronous native-job hooks; done includes process-range cleanup. */
export function numericalJob({ ctx, job, slots, config, request, policy, onEvent = () => {} }) {
  const abort = new AbortController();
  let process, forceTimer, cancelled = false, cleanupVerified = true;
  const cancel = () => {
    cancelled = true;
    if (!process) {
      abort.abort(new Error('Cancelled before numerical worker started'));
      return;
    }
    if (forceTimer) return;
    if (process.stdin?.writable && !process.stdin.destroyed) process.stdin.write('{"command":"cancel"}\n');
    forceTimer = setTimeout(() => process.terminate(), config.cancellationGraceMs);
    forceTimer.unref?.();
  };
  const done = (async () => {
    let release;
    try {
      job.updateProgress('Waiting for a numerical worker slot');
      release = await slots.acquire(abort.signal);
      let argv = [config.pythonPath, '-m', 'control_harness', '--protocol', '--timeout', String(config.timeoutSeconds)];
      if (policy.mode !== 'danger-full-access') argv = (await ctx.sandbox.confine(argv, policy, abort.signal)).argv;
      abort.signal.throwIfAborted();
      process = ctx.subprocess.spawn({ argv, cwd: config.codeRoot, graceMs: config.cancellationGraceMs,
        stdio: { stdin: 'pipe', stdout: 'pipe', stderr: { maxBytes: 65536 } },
        env: { PYTHONPATH: `${config.codeRoot}/src`, PYTHONDONTWRITEBYTECODE: '1', PYTHONUNBUFFERED: '1' },
      });
      cleanupVerified = false;
      let pending = '', result, workerError;
      process.stdout.setEncoding('utf8');
      process.stdout.on('data', chunk => {
        pending += chunk;
        if (pending.length > 1048576) {
          workerError = 'Numerical worker exceeded maximum JSON line size';
          process.terminate();
          return;
        }
        let newline;
        while ((newline = pending.indexOf('\n')) >= 0) {
          const line = pending.slice(0, newline).trim();
          pending = pending.slice(newline + 1);
          if (!line) continue;
          job.append(`${line}\n`, { channel: 'stdout' });
          try {
            const event = JSON.parse(line);
            onEvent(event);
            if (event.event === 'result') result = event.result;
            if (event.event === 'error') workerError = `${event.error_type}: ${event.message}`;
            if (event.event === 'progress') job.updateProgress(`${event.stage}: ${event.epoch ?? event.trial ?? event.scenario ?? ''}`);
          } catch {
            workerError = 'Invalid JSON from numerical worker';
            process.terminate();
          }
        }
      });
      process.stdin.on('error', error => { workerError ??= error.message; });
      process.stdin.write(`${JSON.stringify(request)}\n`);
      if (cancelled) cancel();
      const outcome = await process.done;
      await process.waitForExit();
      cleanupVerified = true;
      const stderr = process.collected.stderr?.readFrom(0).text;
      if (stderr) job.append(stderr, { channel: 'stderr' });
      if (cancelled) return { status: 'killed', detail: 'Cancelled; check failed-stage checkpoint for resumable training' };
      if (!workerError && !pending.trim() && result?.status === 'rejected' && !['inspect_artifact', 'read_artifact_file'].includes(request.operation) && [0, 2].includes(outcome.exitCode)) {
        return { status: 'failed', detail: 'Stage ran but failed domain acceptance', result: JSON.stringify(result) };
      }
      if (outcome.exitCode !== 0 || workerError || result === undefined || pending.trim()) {
        return { status: 'failed', detail: workerError ?? `Worker exited ${outcome.exitCode} without a complete result: ${stderr ?? ''}` };
      }
      return { status: 'completed', result: JSON.stringify(result) };
    } catch (error) {
      let cleanupError;
      if (process) {
        try {
          process.terminate();
          await process.done.catch(() => {});
          await process.waitForExit();
          cleanupVerified = true;
        } catch (failure) {
          cleanupError = failure instanceof Error ? failure.message : String(failure);
        }
      }
      const detail = error instanceof Error ? error.message : String(error);
      return { status: cleanupError ? 'failed' : cancelled ? 'killed' : 'failed',
        detail: cleanupError ? `${detail}; process cleanup could not be verified: ${cleanupError}` : detail };
    } finally {
      if (forceTimer) clearTimeout(forceTimer);
      release?.();
    }
  })().then(outcome => ({ ...outcome, cleanupVerified }));
  return { cancel, done };
}
