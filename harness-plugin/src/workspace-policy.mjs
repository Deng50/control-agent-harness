import { lstat } from 'node:fs/promises';
import { dirname, isAbsolute, resolve } from 'node:path';

export class WorkspacePreflightError extends Error {
  constructor(diagnostic) {
    const result = { ...diagnostic, stage:'preflight', work_started:false };
    super(JSON.stringify(result));
    this.name = 'WorkspacePreflightError';
    this.diagnostic = result;
  }
}

/** Reject ambiguous Windows native paths before resolve() can silently anchor them to the Host drive. */
export async function assertWorkspacePolicyPath(workspace, policy, write = false) {
  const validate = path => {
    if (typeof path !== 'string' || !isAbsolute(path) ||
        (process.platform === 'win32' && !/^[A-Za-z]:[\\/]/.test(path))) {
      throw new Error('workspace must be an absolute local drive path; UNC, device and drive-relative paths are unsupported');
    }
    if (path.split(process.platform === 'win32' ? /[\\/]/ : '/').includes('..')) {
      throw new Error('Workspace must use its physical path without parent traversal');
    }
  };
  validate(workspace);
  if (!write || policy.mode !== 'workspace-write') return;
  if (process.platform === 'win32' && policy.additionalWritableRoots?.length) {
    throw new Error('Windows ACL confinement does not support additional writable directories; select the workspace as Session cwd');
  }
  for (const root of [policy.workspaceRoot, ...(policy.additionalWritableRoots ?? [])]) {
    validate(root);
    for (let path = resolve(root); ; path = dirname(path)) {
      try {
        if ((await lstat(path)).isSymbolicLink()) throw new Error('Sandbox root redirects outside its lexical path; select the physical workspace');
      } catch (error) { if (error.code !== 'ENOENT') throw error; }
      if (dirname(path) === path) break;
    }
  }
}
