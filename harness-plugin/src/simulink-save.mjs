import { lstat, open, realpath } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { isAbsolute, join, relative } from 'node:path';
import { authorizeWorkspace } from './job-history.mjs';
import { pinDirectoryPath } from './windows-host-security.mjs';
import { plantWorkspacePolicy } from './workspace-policy.mjs';

import { matlabString, requireSessionMatlab, runSimulinkHelper } from './simulink-native.mjs';
const MODEL_IDENTIFIER = /^[A-Za-z][A-Za-z0-9_]{0,62}$/;
const RESERVED_FILENAME = /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$/i;

function sameAbsolutePath(candidate, expected) {
  return typeof candidate === 'string' && isAbsolute(candidate) && relative(candidate, expected) === '';
}

async function regularTarget(target) {
  const file = await lstat(target);
  if (!file.isFile() || file.isSymbolicLink() || file.nlink !== 1 ||
      !sameAbsolutePath(await realpath(target), target)) {
    throw new Error('SLX 目标必须是当前会话工作区内的普通文件，不能使用链接。');
  }
  return file;
}

/** 经原生工具和它的审批、专用资源中间件执行；读取本次调用的唯一回执。 */
async function runNativeSave(tools, codeRoot, agent, signal, model, target) {
  const receipt = await runSimulinkHelper(tools, codeRoot, agent, signal, 'control_harness_save_slx', 'SLX',
    [matlabString(model), matlabString(target)]);
  if (receipt.status === 'refused') throw Object.assign(new Error(receipt.reason),{stage:'save_precheck',sessionVerified:true});
  if (receipt.status !== 'saved' || receipt.model !== model || !sameAbsolutePath(receipt.path, target)) {
    throw Object.assign(new Error('MATLAB 实际保存位置与当前会话的 SLX 目标不同。'),{stage:'save_receipt',sessionVerified:true});
  }
  return receipt;
}

/** MATLAB 已读取模型包清单，此处从 Host 打开目标，核对文件身份、大小和 ZIP 文件头。 */
export async function verifySavedFile(target, expectedBytes) {
  const file = await regularTarget(target);
  if (!Number.isSafeInteger(expectedBytes) || file.size !== expectedBytes || file.size < 4) {
    throw new Error('SLX 文件长度与 MATLAB 保存记录不同。');
  }
  const handle = await open(target, 'r');
  try {
    const openedFile = await handle.stat();
    const header = Buffer.alloc(4);
    const {bytesRead} = await handle.read(header, 0, header.length, 0);
    if (openedFile.dev !== file.dev || openedFile.ino !== file.ino || openedFile.size !== file.size ||
        bytesRead !== header.length || header.readUInt32LE() !== 0x04034b50) {
      throw new Error('保存后的 SLX 文件发生变化或缺少 ZIP 文件头。');
    }
    const hash = createHash('sha256');
    for await (const chunk of handle.createReadStream({start:0,autoClose:false})) hash.update(chunk);
    const after = await handle.stat();
    const published = await regularTarget(target);
    if (after.size !== openedFile.size || after.mtimeMs !== openedFile.mtimeMs ||
        published.dev !== after.dev || published.ino !== after.ino || published.size !== after.size ||
        published.mtimeMs !== after.mtimeMs) throw new Error('生成下载校验值时，SLX 目标发生变化。');
    return {bytes:after.size,sha256:hash.digest('hex')};
  } finally {
    await handle.close();
  }
}

/** 外部参数只能指定模型名。保存目录取自发起调用的会话，并受该会话写入策略约束。 */
export async function saveSessionSlx(ctx, config, args, {agent, signal}, preflight) {
  const model = args?.model;
  if (Object.keys(args ?? {}).some(key => key !== 'model')) throw new Error('保存参数只接受模型名。');
  return withSessionSlxTarget(ctx, config, model, {agent,signal}, preflight, 'save_simulink_model', async target => {
    let receipt;
    try {
      receipt = await runNativeSave(ctx.tools, config.codeRoot, agent, signal, model, target);
      const file = await verifySavedFile(target, receipt.bytes);
      return {status:'saved', model, path:target, ...file, simulation:'not_run'};
    } catch (error) {
      signal.throwIfAborted();
      let exists = null;
      try { exists = (await regularTarget(target)).isFile(); }
      catch (statError) { if (statError.code === 'ENOENT') exists = false; }
      throw new Error(JSON.stringify({status:'failed', stage:receipt ? 'host_file_check' : error.stage ?? 'save',
        model, path:target, reason:error.message, identifier:error.identifier, file_exists:exists,
        verified:receipt ? ['matlab_save'] : [], session_verified:receipt ? true : error.sessionVerified ?? null,
        recovery:'先读回目标文件和模型的 FileName，核对保存结果后再决定下一步。'}));
    }
  });
}

export async function withSessionSlxTarget(ctx, config, model, {agent, signal}, preflight, operation, action) {
  if (typeof model !== 'string' || !MODEL_IDENTIFIER.test(model) || RESERVED_FILENAME.test(model)) {
    throw new Error('请输入已加载的 Simulink 模型名，例如 MassSpringDamper（不含路径或扩展名）。');
  }
  const session = agent?.session;
  const cwd = session?.header?.cwd;
  if (!session || session.id !== agent.id || typeof cwd !== 'string' ||
      !isAbsolute(cwd) || /[\r\n\0]/.test(cwd)) {
    throw new Error('保存 SLX 需要本次会话绑定的绝对工作目录。');
  }
  requireSessionMatlab(ctx, config, agent);
  const policy = operation === 'save_simulink_model' ? ctx.sandboxPolicy.resolve({session}) : plantWorkspacePolicy(ctx,session);
  const workspace = await authorizeWorkspace(cwd, policy, true);
  await preflight({workspace, policy, signal, operation, agent});
  signal.throwIfAborted();
  const release = pinDirectoryPath(workspace);
  try {
    const target = join(workspace, model + '.slx');
    try {
      await regularTarget(target);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    return await action(target, workspace);
  } finally {
    release();
  }
}
