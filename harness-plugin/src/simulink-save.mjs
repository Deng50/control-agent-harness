import { lstat, open, realpath } from 'node:fs/promises';
import { isAbsolute, join, relative } from 'node:path';
import { authorizeWorkspace } from './job-history.mjs';
import { pinDirectoryPath } from './windows-host-security.mjs';

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
  if (receipt.status === 'refused') throw new Error(receipt.reason);
  if (receipt.status !== 'saved' || receipt.model !== model || !sameAbsolutePath(receipt.path, target)) {
    throw new Error('MATLAB 实际保存位置与当前会话的 SLX 目标不同。');
  }
  return receipt;
}

/** MATLAB 已读取模型包清单，此处从 Host 打开目标，核对文件身份、大小和 ZIP 文件头。 */
async function verifySavedFile(target, expectedBytes) {
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
    return openedFile.size;
  } finally {
    await handle.close();
  }
}

/** 外部参数只能指定模型名。保存目录取自发起调用的会话，并受该会话写入策略约束。 */
export async function saveSessionSlx(ctx, config, args, {agent, signal}, preflight) {
  const model = args?.model;
  if (Object.keys(args ?? {}).some(key => key !== 'model') || typeof model !== 'string' ||
      !MODEL_IDENTIFIER.test(model) || RESERVED_FILENAME.test(model)) {
    throw new Error('请输入已加载的 Simulink 模型名，例如 MassSpringDamper（不含路径或扩展名）。');
  }
  const session = agent?.session;
  const cwd = session?.header?.cwd;
  if (!session || session.id !== agent.id || typeof cwd !== 'string' ||
      !isAbsolute(cwd) || /[\r\n\0]/.test(cwd)) {
    throw new Error('保存 SLX 需要本次会话绑定的绝对工作目录。');
  }
  requireSessionMatlab(ctx, config, agent);
  const policy = ctx.sandboxPolicy.resolve({session});
  const workspace = await authorizeWorkspace(cwd, policy, true);
  await preflight({workspace, policy, signal, operation: 'save_simulink_model'});
  signal.throwIfAborted();
  const release = pinDirectoryPath(workspace);
  try {
    const target = join(workspace, model + '.slx');
    try {
      await regularTarget(target);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    const receipt = await runNativeSave(ctx.tools, config.codeRoot, agent, signal, model, target);
    const bytes = await verifySavedFile(target, receipt.bytes);
    return {status: 'saved', model, path: target, bytes, simulation: 'not_run'};
  } finally {
    release();
  }
}
