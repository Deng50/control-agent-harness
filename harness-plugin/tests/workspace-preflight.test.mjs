import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, rmdir, symlink, unlink, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { preflightWorkspace } from '../src/workspace-preflight.mjs';
import { JobHistory } from '../src/job-history.mjs';
import { createHost } from './history-host.mjs';
import { inspectHostSecurity } from '../src/windows-host-security.mjs';

const project = resolve(import.meta.dirname, '../..');
const upstream = path => import(pathToFileURL(resolve(project, 'vendor/deepseek-harness', path)));
const config = { codeRoot:project, pythonPath:join(project, '.venv/Scripts/python.exe') };

test('unsupported policy fails before subprocess or job admission', async () => {
  const workspace = resolve(project, 'test-output');
  await assert.rejects(preflightWorkspace({ctx:{}, config, workspace, policy:{mode:'read-only', workspaceRoot:workspace}}),
    error => error.diagnostic?.actor === 'policy' && error.diagnostic.work_started === false);
});

test('native submit, explicit recovery and session attachment reject before registration or durable mutation', {
  timeout:45000,
}, async () => {
  const root = resolve(project, 'test-output');
  await mkdir(root, {recursive:true});
  const workspace = await mkdtemp(join(root, 'p0b admission ')), owner = 'preflight-' + randomUUID();
  const history = new JobHistory(), request = {operation:'inspect_workspace', inputs:{}, config:{}, workspace};
  const prior = await history.create({workspace, owner, agentName:'model', request});
  await prior.append({state:'failed', outcome:{status:'failed', cleanupVerified:true}});
  const attemptId = prior.record.attemptId;
  const ledger = join(workspace,'.control-jobs'), state = join(ledger,attemptId);
  const names = await readdir(state);
  const bytes = () => Promise.all(names.map(name => readFile(join(state,name),'utf8')));
  const before = await bytes();
  let policy = {mode:'read-only', workspaceRoot:workspace};
  const host = await createHost(workspace, owner, {confirmFixturePlans:false, policy:() => policy,
    pythonPath:join(workspace,'missing-interpreter.exe')});
  try {
    const commands = [
      ['control', {agent:'model', ...request}], ['control-recover', {workspace, attemptId, mode:'retry'}],
      ['control-workspace', {}],
    ];
    // The missing interpreter yields a real managed-subprocess failure under the
    // historical fixture's existing unrestricted policy; the native runner has a separate test below.
    const modes = ['read-only','danger-full-access'];
    // Ordinary inherited "Modify" on local Windows often lacks WRITE_OWNER.
    // Exercise the observed real Host denial when present, without altering ACLs to manufacture it.
    if (process.platform === 'win32' && Object.keys(inspectHostSecurity(workspace).right_errors).length) modes.push('workspace-write');
    for (const mode of modes) {
      policy = {mode, workspaceRoot:workspace};
      for (const [command, args] of commands) {
        const response = await host.command(command, args);
        assert.equal(response.kind,'error', response.text);
        if (mode === 'read-only') assert.match(response.text,/does not allow/);
        else {
          const detail = JSON.parse(response.text);
          assert.equal(detail.actor, mode === 'workspace-write' ? 'host' : 'worker');
          if (mode === 'workspace-write') assert.match(detail.operation,/^check_WRITE/);
          assert.equal(detail.work_started, false);
        }
        assert.equal(host.ctx.jobs.list(owner).length, 0);
        assert.deepEqual(await readdir(ledger), [attemptId]);
        assert.deepEqual(await readdir(state), names);
        assert.deepEqual(await bytes(), before);
        assert.deepEqual(await readdir(workspace), ['.control-jobs']);
      }
    }
  } finally {
    await host.close();
    for (const name of names) await unlink(join(state,name));
    await rmdir(state); await rmdir(ledger); await rmdir(workspace);
  }
});

test('Windows rejects UNC, drive-relative, additional root and redirected session root before probing', {
  skip:process.platform !== 'win32',
}, async () => {
  const root = join(project, 'test-output');
  await mkdir(root, {recursive:true});
  const workspace = await mkdtemp(join(root, 'p0b path '));
  const alias = join(workspace, 'junction');
  const expectPolicyFailure = (path, policy) => assert.rejects(preflightWorkspace({ctx:{}, config, workspace:path, policy}),
    error => error.diagnostic?.actor === 'policy' && error.diagnostic.work_started === false);
  try {
    await symlink(root, alias, 'junction');
    for (const path of ['D:relative', '\\\\server\\share', alias]) {
      await expectPolicyFailure(path, {mode:'workspace-write', workspaceRoot:path});
    }
    await expectPolicyFailure(workspace, {mode:'workspace-write', workspaceRoot:workspace, additionalWritableRoots:[root]});
  } finally {
    await unlink(alias);
    await rmdir(workspace);
  }
});

test('Windows Host and real Low-integrity worker exercise isolated workspace and private TEMP', {
  skip: process.platform !== 'win32', timeout: 45000,
}, async () => {
  const root = resolve(project, 'test-output');
  await mkdir(root, {recursive:true});
  const workspace = await mkdtemp(join(root, 'p0b preflight '));
  const [{Context}, {default:Sandbox}, {default:Subprocess}] = await Promise.all([
    upstream('vendor/cordis/lib/index.js'), upstream('packages/sandbox/sandbox-local/lib/index.js'),
    upstream('packages/subprocess/subprocess-local/lib/index.js'),
  ]);
  const ctx = new Context();
  try {
    // Give only the fixture owner the two missing ACL setup rights on this fresh leaf.
    const helper = join(project, 'scripts/prepare-workspace.ps1');
    const prepare = JSON.parse(execFileSync('powershell.exe', ['-NoProfile','-File',helper,'-Prepare','-Workspace',workspace],
      {encoding:'utf8',windowsHide:true,timeout:20000}));
    assert.equal(prepare.prepared, true);
    assert.equal(prepare.after.filesystem, 'NTFS');
    assert.deepEqual(prepare.after.effective_rights, {WRITE_DAC:true, WRITE_OWNER:true});
    const acl = () => execFileSync('icacls.exe', [workspace], {encoding:'utf8', windowsHide:true});
    const before = acl();
    const inspection = JSON.parse(execFileSync('powershell.exe', ['-NoProfile','-File',helper,'-Workspace',workspace],
      {encoding:'utf8',windowsHide:true,timeout:10000}));
    assert.equal(inspection.ok, true);
    assert.equal(acl(), before);
    await ctx.plugin(Subprocess);
    await ctx.plugin(Sandbox);
    const policy = {
      mode:'workspace-write', workspaceRoot:workspace, sessionId:randomUUID(), additionalWritableRoots:[],
    };
    const diagnostic = await preflightWorkspace({ctx,config,workspace,policy});
    assert.equal(diagnostic.ok, true);
    assert.equal(diagnostic.host.security.user_sid, diagnostic.worker.user_sid);
    assert.equal(diagnostic.worker.integrity_sid, 'S-1-16-4096');
    assert.notEqual(diagnostic.worker.temp, process.env.TEMP);
    assert.deepEqual(await readdir(workspace), []);
    const names = ['.control-jobs','.case-plans','.staging','artifacts','failed'];
    for (const name of names) await mkdir(join(workspace,name));
    try {
      const training = await preflightWorkspace({ctx,config,workspace,policy,operation:'train_supervised'});
      assert.deepEqual(training.worker.writable_business_directories, ['.staging','artifacts','failed'].map(name=>join(workspace,name)));
      // Model a legacy ACL on only the disposable staging directory. The ordinary
      // Host is still able to access it; this exact session's worker loses add-subdir.
      const wrapped = await ctx.sandbox.confine([config.pythonPath,'--version'], policy);
      const capability = wrapped.argv[wrapped.argv.indexOf('--write-sid') + 1];
      assert.match(capability,/^S-1-4-\d+-\d+$/);
      execFileSync('powershell.exe', ['-NoProfile','-NonInteractive','-Command', [
        "$ErrorActionPreference = 'Stop'",
        "$directory = [IO.DirectoryInfo]::new($env:CONTROL_TEST_STAGING)",
        "$sid = [Security.Principal.SecurityIdentifier]::new($env:CONTROL_TEST_CAPABILITY)",
        "$acl = $directory.GetAccessControl()",
        "$acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($sid,'CreateDirectories','Deny'))",
        "$directory.SetAccessControl($acl)",
      ].join('\n')], {windowsHide:true, env:{...process.env, CONTROL_TEST_STAGING:join(workspace,'.staging'), CONTROL_TEST_CAPABILITY:capability}});
      await assert.rejects(preflightWorkspace({ctx,config,workspace,policy,operation:'train_supervised'}),
        error => error.diagnostic?.actor === 'worker' && error.diagnostic.operation === 'create_probe_directory'
          && error.diagnostic.path === join(workspace,'.staging'));
      const inspect = await preflightWorkspace({ctx,config,workspace,policy,operation:'inspect_workspace'});
      assert.deepEqual(inspect.worker.writable_business_directories, []);
      for (const name of names) assert.deepEqual(await readdir(join(workspace,name)), []);
    } finally { for (const name of names) await rmdir(join(workspace,name)); }
  } finally {
    await ctx.fiber.dispose();
    // Preserve an original probe/ACL error if Windows denies cleanup; do not recursively remove residue.
    if (!(await readdir(workspace)).length) await rmdir(workspace);
  }
});

test('pre-existing staging junction fails before runner grant and leaves the external target untouched', async () => {
  const root = resolve(project, 'test-output');
  await mkdir(root, {recursive:true});
  const workspace = await mkdtemp(join(root, 'p0b redirected ')), external = await mkdtemp(join(root, 'p0b outside '));
  const alias = join(workspace,'.staging'), marker = join(external,'user-record');
  try {
    await writeFile(marker,'original',{flag:'wx'});
    await symlink(external, alias, process.platform === 'win32' ? 'junction' : 'dir');
    await assert.rejects(preflightWorkspace({ctx:{},config,workspace,policy:{mode:'workspace-write', workspaceRoot:workspace}}),
      error => error.diagnostic?.operation === 'validate_reserved_directory' && error.diagnostic.work_started === false);
    assert.deepEqual(await readdir(external), ['user-record']);
    assert.equal(await readFile(marker,'utf8'), 'original');
    assert.deepEqual(await readdir(workspace), ['.staging']);
  } finally { await unlink(alias); await unlink(marker); await rmdir(workspace); await rmdir(external); }
});
