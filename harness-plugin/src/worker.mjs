/** Bounded global worker admission and JSON-lines execution over Harness subprocess handles. */
import {stages} from './stage-contracts.mjs';
import {verifyPublication} from './case-requirements.mjs';

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
export function numericalJob({ ctx, job, slots, config, request, policy, onEvent = () => {}, confirmation, extraEnv = {}, collectionResume }) {
  const privateKeys = new Set(['CONTROL_MATLAB_PHASE', 'CONTROL_MATLAB_EXCHANGE_DIR', 'CONTROL_MATLAB_REQUEST_SHA256',
    'CONTROL_MATLAB_NONCE', 'CONTROL_MATLAB_RESULT_PATH', 'CONTROL_MATLAB_RESULT_SHA256']);
  if (Object.entries(extraEnv).some(([key,value]) => !privateKeys.has(key) || typeof value !== 'string')) {
    throw new Error('Only fixed private MATLAB batch fields may extend a numerical worker environment');
  }
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
      const queueMs = Math.ceil((config.queueTimeoutSeconds ?? 120) * 1000);
      const queueDeadline = AbortSignal.timeout(queueMs);
      job.updateProgress(`等待数值计算名额，排队上限 ${queueMs / 1000} 秒`);
      try {
        release = await slots.acquire(AbortSignal.any([abort.signal, queueDeadline]));
      } catch (error) {
        if (queueDeadline.aborted && !abort.signal.aborted) {
          throw new Error(`WORKER_QUEUE_TIMEOUT：等待名额超过 ${queueMs / 1000} 秒，计算尚未开始，可稍后重试。`);
        }
        throw error;
      }
      let argv = [config.pythonPath, '-m', 'control_harness', '--protocol', '--timeout', String(config.timeoutSeconds)];
      if (policy.mode !== 'danger-full-access') argv = (await ctx.sandbox.confine(argv, policy, abort.signal)).argv;
      abort.signal.throwIfAborted();
      process = ctx.subprocess.spawn({ argv, cwd: config.codeRoot, graceMs: config.cancellationGraceMs,
        stdio: { stdin: 'pipe', stdout: 'pipe', stderr: { maxBytes: 65536 } },
        env: { ...extraEnv, PYTHONPATH: `${config.codeRoot}/src`, PYTHONDONTWRITEBYTECODE: '1', PYTHONUNBUFFERED: '1',
          PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8',
          CONTROL_COLLECTION_RESUME: collectionResume ? JSON.stringify(collectionResume) : '',
          CONTROL_CASE_RECEIPT_BODY: confirmation?.env.CONTROL_CASE_RECEIPT_BODY ?? '',
          CONTROL_CASE_RECEIPT_SIGNATURE: confirmation?.env.CONTROL_CASE_RECEIPT_SIGNATURE ?? '',
          CONTROL_CASE_RECEIPT_KEY: confirmation?.env.CONTROL_CASE_RECEIPT_KEY ?? '',
          CONTROL_CASE_NATIVE_SESSION: confirmation?.env.CONTROL_CASE_NATIVE_SESSION ?? '' },
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
      const rawRequest = confirmation?.rawRequest ?? JSON.stringify(request);
      if (confirmation && rawRequest !== JSON.stringify(request)) throw new Error('Confirmed request changed before worker dispatch');
      process.stdin.write(`${rawRequest}\n`);
      if (cancelled) cancel();
      const outcome = await process.done;
      await process.waitForExit();
      cleanupVerified = true;
      const stderr = process.collected.stderr?.readFrom(0).text;
      if (stderr) job.append(stderr, { channel: 'stderr' });
      if (cancelled) return { status: 'killed', detail: 'Cancelled; check failed-stage checkpoint for resumable training' };
      const artifactStage = request.operation === 'register_material' || (Object.hasOwn(stages,request.operation)
        && !['recommend_sampling','prepare_imitation_evaluation'].includes(request.operation));
      const matlabPreparation = result?.matlab_phase === 'prepared' &&
        ['sample_model','collect_external_expert','evaluate_external_imitation','evaluate_external_pgac','check_matlab_design'].includes(request.operation);
      if (!workerError && !pending.trim() && [0,2].includes(outcome.exitCode) && !matlabPreparation && (artifactStage || result?.delivery)) {
        try {await verifyPublication(request.workspace,result);}
        catch(error) {return {status:'failed',detail:'产物发布核验失败：'+error.message};}
      }
      if (!workerError && !pending.trim() && result?.status === 'rejected' && !['inspect_artifact', 'read_artifact_file'].includes(request.operation) && [0, 2].includes(outcome.exitCode)) {
        if (request.operation==='verify_design_delivery' && result.delivery?.status==='partial') {
          return {status:'completed',detail:'已记录交付验收，所需证据仍有缺项。',result:JSON.stringify(result)};
        }
        return { status: 'failed', detail: 'Stage ran but failed domain acceptance', result: JSON.stringify(result) };
      }
      if (outcome.exitCode !== 0 || workerError || result === undefined || pending.trim()) {
        return { status: 'failed', detail: workerError ?? `Worker exited ${outcome.exitCode} without a complete result: ${stderr ?? ''}` };
      }
      if (!['inspect_artifact','read_artifact_file'].includes(request.operation) && result?.delivery?.status === 'failed') {
        return {status:'failed',detail:'交付核验未通过，请查看逐项证据。',result:JSON.stringify(result)};
      }
      return { status: 'completed', ...(result?.delivery?.status === 'partial' ? {detail:'产物已生成，交付条件尚有缺项。'} : {}), result: JSON.stringify(result) };
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
