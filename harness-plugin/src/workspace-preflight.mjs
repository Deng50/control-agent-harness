/** Pre-admission checks run before case mutations and native job registration. */
import { link, lstat, mkdir, open, readFile, realpath, rename, rmdir, unlink } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, relative, resolve } from 'node:path';
import { authorizeWorkspace } from './job-history.mjs';
import { inspectHostSecurity, pinDirectoryPath } from './windows-host-security.mjs';
import { WorkspacePreflightError } from './workspace-policy.mjs';
export { WorkspacePreflightError };

function failure(actor, operation, path, error, extra = {}) {
  const nativeCode = error.winerror ?? error.errno ?? error.message?.match(/\bWin32\s+(\d+)/i)?.[1];
  return new WorkspacePreflightError({ code: 'WORKSPACE_PREFLIGHT_FAILED', actor, operation, path,
    os_code: nativeCode ?? error.code ?? null,
    os_domain: error.winerror || /\bWin32\s+\d+/i.test(error.message) ? 'Win32' : error.errno ? 'libuv' : null,
    detail: error.message,
    hint: actor === 'policy' ? '请将会话工作目录设为实际本地工作区，再重试。'
      : actor === 'sandbox' || operation.startsWith('check_WRITE')
      ? '检查会话根目录的所有者、WRITE_DAC、WRITE_OWNER 和私有 TEMP；可用 prepare-workspace.ps1 只读诊断。'
      : '请检查报错路径的文件占用和当前进程权限，修复后重试。', ...extra });
}

async function hostIO(root, probePath, runWorker, checkHardlink = true) {
  let directory, current, candidate, published, releasePin, operation = 'create_probe_directory', target = root, problem, worker;
  const owned = new Set();
  try {
    await mkdir(probePath, {mode:process.platform === 'win32' ? 0o777 : 0o700});
    directory = probePath;
    releasePin = pinDirectoryPath(directory);
    current = join(directory, 'current');
    candidate = join(directory, 'candidate');
    published = join(directory, 'published');
    operation = 'write_exclusive';
    for (const [file, content] of [[current,'old'], [candidate,'new']]) {
      target = file;
      const handle = await open(file, 'wx');
      owned.add(file);
      try { await handle.writeFile(content); } finally { await handle.close(); }
    }
    if (checkHardlink) {
      operation = 'publish_history_hardlink'; target = published;
      await link(current, published);
      owned.add(published);
    }
    operation = 'atomic_replace'; target = current;
    await rename(candidate, current);
    owned.delete(candidate);
    operation = 'read_replacement';
    if (await readFile(current, 'utf8') !== 'new') throw new Error('Atomic replacement read-back failed');
    if (runWorker) {
      worker = await runWorker();
      operation = 'read_worker_replacement';
      if (await readFile(current, 'utf8') !== 'worker') throw new Error('Host could not verify the worker replacement');
    }
  } catch (error) {
    problem = error instanceof WorkspacePreflightError ? error : failure('host', operation, target, error);
  } finally {
    if (directory) {
      // Only remove the named probe files. An unexpected file must survive cleanup.
      for (const entry of [...owned, directory]) {
        try {
          if (entry === directory) { releasePin?.(); await rmdir(entry); }
          else await unlink(entry);
        }
        catch (error) {
          if (error.code !== 'ENOENT') {
            const cleanup = failure('host', 'cleanup', entry, error);
            if (problem) problem = new WorkspacePreflightError({ ...problem.diagnostic, cleanup:problem.diagnostic.cleanup ?? cleanup.diagnostic });
            else problem = cleanup;
          }
        }
      }
    }
  }
  if (problem) throw problem;
  return worker;
}

async function prepareProbe(ctx, config, workspace, policy, stop, crossFile, operation) {
  const windowsAcl = process.platform === 'win32' && policy.mode === 'workspace-write';
  const actor = 'worker';
  let argv = [config.pythonPath, '-I', '-S', '-B', join(config.codeRoot, 'src/control_harness/workspace_probe.py'),
    '--cross-file', crossFile];
  if (operation) argv.push('--operation', operation);
  if (windowsAcl) argv.push('--windows-acl', '--ambient-temp', tmpdir());
  argv.push('--', workspace);
  let expectedTemp;
  if (policy.mode !== 'danger-full-access') {
    try {
      const wrapped = await ctx.sandbox.confine(argv, policy, stop);
      const args = wrapped.argv.slice(0, -argv.length);
      if (windowsAcl && args.includes('--temp-write-sid')) expectedTemp = args[args.indexOf('--temp') + 1];
      argv = wrapped.argv;
    } catch (error) { throw failure('sandbox', 'confine', policy.workspaceRoot, error); }
  }
  return () => runProbe({ctx, config, workspace, actor, stop, argv, expectedTemp});
}

async function runProbe({ctx, config, workspace, actor, stop, argv, expectedTemp}) {
  let handle;
  try {
    stop.throwIfAborted();
    handle = ctx.subprocess.spawn({ argv, cwd: config.codeRoot, signal: stop, graceMs: 1000,
      stdio: { stdin: 'ignore', stdout: { maxBytes: 32768 }, stderr: { maxBytes: 4096 } } });
    const outcome = await handle.done;
    await handle.waitForExit();
    stop.throwIfAborted();
    const output = handle.collected.stdout.readFrom(0).text;
    let report;
    try { report = JSON.parse(output); } catch {
      throw failure(actor, 'launch_probe', config.pythonPath,
        new Error(handle.collected.stderr.readFrom(0).text.trim() || 'Probe did not produce a diagnostic'), { exit_code: outcome.exitCode });
    }
    if (!report.ok || outcome.exitCode !== 0) {
      const detail = report.error ?? {};
      throw failure(actor, detail.operation ?? 'probe', detail.path ?? workspace, new Error(detail.message), {
        os_code:detail.os_code ?? null, os_domain:detail.os_domain ?? null, probe:report,
      });
    }
    const expected = {workspace, cwd:await realpath(config.codeRoot)};
    if (expectedTemp) expected.temp = await realpath(expectedTemp);
    for (const [field, value] of Object.entries(expected)) {
      if (relative(await realpath(report[field]), value)) {
        throw failure(actor, 'verify_probe_context', workspace, new Error('Runner context differs from dispatch'),
          { field, expected:value, observed:report[field] });
      }
    }
    return report;
  } catch (error) {
    if (error instanceof WorkspacePreflightError) throw error;
    throw failure(actor, stop.aborted ? 'cancel_or_timeout' : 'spawn_probe', config.pythonPath, error);
  }
}

export async function preflightWorkspace({ ctx, config, workspace, policy, signal, operation }) {
  const timeout = AbortSignal.timeout(15000);
  const stop = signal ? AbortSignal.any([signal, timeout]) : timeout;
  let root;
  try {
    stop.throwIfAborted();
    root = await authorizeWorkspace(workspace, policy, true);
  } catch (error) {
    if (error instanceof WorkspacePreflightError) throw error;
    throw failure('policy','authorize_workspace',workspace,error);
  }
  const releases = [];
  try {
    releases.push(pinDirectoryPath(root, true));
    return await checkAuthorizedWorkspace({ctx,config,root,policy,stop,releases,operation});
  } catch (error) {
    if (error instanceof WorkspacePreflightError) throw error;
    throw failure('host',error.operation ?? 'pin_workspace',root,error);
  } finally {
    while (releases.length) releases.pop()();
  }
}

async function checkAuthorizedWorkspace({ctx,config,root,policy,stop,releases,operation}) {
  // Preserve support for a fresh authorized workspace. Do not create ledger, plans,
  // checkpoints, or an attempt during admission. Never chmod or alter an ACL here.
  try {
    await mkdir(root, { recursive:true });
    releases.push(pinDirectoryPath(root));
    if (!(await lstat(root)).isDirectory() || relative(root, await realpath(root))) throw new Error('Workspace path changed');
  } catch (error) { throw failure('host', 'open_workspace', root, error); }
  const existing = {};
  for (const name of ['.control-jobs', '.case-plans', '.staging', 'artifacts', 'failed', '.report-downloads']) {
    const path = join(root, name);
    try {
      const entry = await lstat(path);
      if (!entry.isDirectory() || entry.isSymbolicLink() || relative(path, await realpath(path))) {
        throw new Error('Reserved workspace directory redirects or is not a directory');
      }
      releases.push(pinDirectoryPath(path));
      existing[name] = path;
    } catch (error) {
      if (error.code !== 'ENOENT') throw failure('host', 'validate_reserved_directory', path, error);
    }
  }
  let host = { pid:process.pid, cwd:process.cwd() };
  if (process.platform === 'win32') {
    const grantRoot = policy.mode === 'workspace-write' ? resolve(policy.workspaceRoot) : root;
    let security;
    try { security = inspectHostSecurity(grantRoot); }
    catch (error) { throw failure('host', error.operation ?? 'inspect_security', grantRoot, error); }
    host = { ...host, security };
    if (policy.mode === 'workspace-write') {
      if (security.filesystem.toUpperCase() !== 'NTFS') {
        throw failure('host', 'filesystem', grantRoot, new Error('Windows ACL backend requires local NTFS'), {security});
      }
      for (const [right, code] of Object.entries(security.right_errors)) {
        throw failure('host', 'check_' + right, grantRoot, Object.assign(new Error(right), {winerror:code}), {security});
      }
    }
  }
  const directory = join(root, '.control-preflight-' + randomUUID());
  // Materialize native session grants before creating any Host-owned probe files.
  // Existing files can carry a different integrity label even after root admission.
  const launch = await prepareProbe(ctx, config, root, policy, stop, join(directory, 'current'), operation);
  if (existing['.control-jobs']) {
    const ledger = existing['.control-jobs'];
    await hostIO(ledger, join(ledger, '.control-preflight-' + randomUUID()));
  }
  if (existing['.case-plans'] && (!operation || operation === 'propose_case_plans' || operation === 'propose_case_plan')) {
    const plans = existing['.case-plans'];
    await hostIO(plans, join(plans, '.control-preflight-' + randomUUID()), undefined, false);
  }
  const worker = await hostIO(root, directory, launch);
  return { ok:true, workspace:root, host, worker, sandbox_mode:policy.mode };
}
