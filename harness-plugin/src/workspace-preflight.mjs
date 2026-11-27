/** Pre-admission checks run before case mutations and native job registration. */
import { link, lstat, mkdir, open, readFile, realpath, rename, rmdir, unlink } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { tmpdir } from 'node:os';
import { isAbsolute, join, relative, resolve, sep } from 'node:path';
import { authorizeWorkspace } from './job-history.mjs';
import { inspectHostSecurity, pinDirectoryPath } from './windows-host-security.mjs';
import { WorkspacePreflightError } from './workspace-policy.mjs';
export { WorkspacePreflightError };

const pendingPreparations = new Map();

function insideDirectory(parent, path) {
  const child = relative(parent,path);
  return child !== '' && !isAbsolute(child) && child !== '..' && !child.startsWith('..' + sep);
}

export function isManagedWorkspace(codeRoot, workspace) {
  return typeof workspace === 'string' && isAbsolute(workspace) && insideDirectory(join(codeRoot,'workspaces'),workspace);
}

export function canPrepareSessionWorkspace(codeRoot, workspace, security) {
  return isManagedWorkspace(codeRoot,workspace) &&
    security?.owner_is_current_user === true && security.filesystem?.toUpperCase() === 'NTFS' &&
    security.effective_rights?.WRITE_DAC === true && security.right_errors?.WRITE_OWNER === 5;
}

// The path comes from the verified native Session cwd. All callers share the
// same bounded, idempotent ACL edit, then independently verify their own worker.
function prepareOwnedWorkspace(ctx, config, workspace) {
  const key = workspace.toLowerCase();
  if (pendingPreparations.has(key)) return pendingPreparations.get(key);
  const prepare = (async () => {
    const executable = join(process.env.SystemRoot,'System32','WindowsPowerShell','v1.0','powershell.exe');
    const proc = ctx.subprocess.spawn({argv:[executable,'-NoProfile','-NonInteractive','-File',
      join(config.codeRoot,'scripts','prepare-workspace.ps1'),'-Workspace',workspace,'-PrepareForSession'],
      cwd:config.codeRoot, signal:AbortSignal.timeout(12000), graceMs:1000,
      stdio:{stdin:'ignore', stdout:{maxBytes:32768}, stderr:{maxBytes:4096}}});
    const done = await proc.done;
    await proc.waitForExit();
    if (done.exitCode !== 0) throw new Error(proc.collected.stderr.readFrom(0).text.trim() || 'Workspace preparation failed');
    const result = JSON.parse(proc.collected.stdout.readFrom(0).text);
    if (relative(workspace,result.workspace) || !Array.isArray(result.added_leaf_rights) ||
        !result.added_leaf_rights.every(r=>r === 'WRITE_OWNER')) throw new Error('Invalid workspace preparation receipt');
    return result;
  })().finally(()=>pendingPreparations.delete(key));
  pendingPreparations.set(key,prepare);
  return prepare;
}

// A native shell does not publish engineering ledgers. Prepare only the
// standing workspace grant it will actually consume; its own confined
// dispatch verifies the result. An alternate runner that already works
// needs no ACL adjustment.
export async function prepareShellWorkspaceGrant(ctx, config, policy, signal) {
  const workspace = policy.workspaceRoot;
  let security;
  try { security = inspectHostSecurity(workspace); } catch { return; }
  if (!canPrepareSessionWorkspace(config.codeRoot,workspace,security)) return;
  signal.throwIfAborted();
  try {
    // confine prepares the provider's grant but does not spawn a process.
    await ctx.sandbox.confine([config.pythonPath,'-V'],policy,signal);
    return;
  } catch (error) {
    if (error.api !== 'SetNamedSecurityInfoW' || error.win32Code !== 5 ||
        !error.message.endsWith(`: grantWrite(${workspace})`)) throw error;
  }
  signal.throwIfAborted();
  const receipt = await prepareOwnedWorkspace(ctx,config,workspace);
  if (!receipt.prepared) throw new Error('目录的 ACL 不允许自动准备 PowerShell 工作区：' + (receipt.reason ?? workspace));
}

function windowsRecovery(codeRoot, workspace, security) {
  const quote = value => "'" + value.replaceAll("'", "''") + "'";
  const command = '& ' + quote(join(codeRoot, 'scripts', 'prepare-workspace.ps1')) + ' -Workspace ' + quote(workspace);
  const inside = ['workspaces', 'test-output'].some(folder => insideDirectory(join(codeRoot,folder),workspace));
  const missing = Object.keys(security.right_errors);
  const canPrepare = inside && security.owner_is_current_user && security.filesystem.toUpperCase() === 'NTFS' &&
    missing.every(right => security.right_errors[right] === 5);
  return {workspace, missing_rights:missing, diagnose_command:command, can_prepare:canPrepare,
    ...(canPrepare ? {prepare_command:command + ' -Prepare'} : {})};
}

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

export async function preflightWorkspace({ ctx, config, workspace, policy, signal, operation, agent }) {
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
    const cwd = agent?.session?.header?.cwd;
    const prepareSessionWorkspace = typeof agent?.id === 'string' && agent.id === agent.session?.id &&
      typeof cwd === 'string' && isAbsolute(cwd) && !relative(cwd,workspace) && !relative(cwd,root);
    return await checkAuthorizedWorkspace({ctx,config,root,policy,stop,releases,operation,prepareSessionWorkspace});
  } catch (error) {
    if (error instanceof WorkspacePreflightError) throw error;
    throw failure('host',error.operation ?? 'pin_workspace',root,error);
  } finally {
    while (releases.length) releases.pop()();
  }
}

async function checkAuthorizedWorkspace({ctx,config,root,policy,stop,releases,operation,prepareSessionWorkspace}) {
  // Preserve support for a fresh authorized workspace. Do not create ledger, plans,
  // checkpoints, or an attempt during admission.
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
    if (policy.mode === 'workspace-write') {
      if (security.filesystem.toUpperCase() !== 'NTFS') {
        throw failure('host', 'filesystem', grantRoot, new Error('Windows ACL backend requires local NTFS'), {security});
      }
      let prepareError;
      if (prepareSessionWorkspace && !relative(root,grantRoot) && canPrepareSessionWorkspace(config.codeRoot,root,security)) {
        stop.throwIfAborted();
        const before = security.effective_rights;
        try {
          const result = await prepareOwnedWorkspace(ctx,config,root);
          host.workspace_preparation = {prepared:result.prepared === true, before, added_leaf_rights:result.added_leaf_rights,
            ...(result.reason ? {reason:result.reason} : {})};
        } catch (error) {
          prepareError = error;
          host.workspace_preparation = {prepared:false,before,reason:'prepare_failed'};
        }
        stop.throwIfAborted();
        security = inspectHostSecurity(root);
        if (!Object.keys(security.right_errors).length && (prepareError || !host.workspace_preparation.prepared ||
            !security.owner_is_current_user || security.filesystem.toUpperCase() !== 'NTFS')) {
          throw failure('host','verify_prepared_workspace',root,prepareError ?? new Error('The owner or preparation receipt changed during workspace setup.'),
            {security,workspace_preparation:host.workspace_preparation});
        }
      }
      for (const [right, code] of Object.entries(security.right_errors)) {
        const recovery = windowsRecovery(config.codeRoot, grantRoot, security);
        const denied = host.workspace_preparation?.reason === 'applicable_deny_ace';
        if (denied) { recovery.can_prepare = false; delete recovery.prepare_command; }
        const message = denied ? '目录 ACL 中存在针对当前账户的拒绝规则，请使用其他自有工作区或联系目录所有者。'
          : prepareError ? right + '：自动准备失败。' + prepareError.message : right;
        throw failure('host', 'check_' + right, grantRoot, Object.assign(new Error(message), {winerror:code}),
          {security, ...(host.workspace_preparation ? {workspace_preparation:host.workspace_preparation} : {}),
            ...(code === 5 ? {workspace_recovery:recovery} : {})});
      }
    }
    host.security = security;
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
