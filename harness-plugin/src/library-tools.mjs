import { join } from 'node:path';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { roleIds } from './roles.mjs';

const output = {
  schema: { type: 'object', additionalProperties: true },
  render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
};

// A host-owned, read-only catalog, like control_help. Only the installed module
// runs here. It accepts indexed paths and never imports or executes case code.
async function lookup(ctx, config, args, signal) {
  const timeout = AbortSignal.timeout(15000);
  const stop = signal ? AbortSignal.any([signal, timeout]) : timeout;
  stop.throwIfAborted();
  const child = ctx.subprocess.spawn({
    argv: [config.pythonPath, '-B', '-S', '-E', '-X', 'utf8', '-m', 'control_harness.library', ...args],
    cwd: join(config.codeRoot, 'src'),
    signal: stop,
    graceMs: 1000,
    stdio: { stdin: 'ignore', stdout: { maxBytes: 256 * 1024 }, stderr: { maxBytes: 4096 } },
  });
  const result = await child.done;
  await child.waitForExit();
  stop.throwIfAborted();
  if (result.exitCode !== 0) {
    const detail = child.collected.stderr.readFrom(0).text.trim();
    throw new Error(detail || `Control library lookup exited with code ${result.exitCode}`);
  }
  const reply = child.collected.stdout.readFrom(0);
  if (reply.lossy) throw new Error('Control library reply exceeded the output limit');
  return JSON.parse(reply.text);
}

export function installLibraryTools(ctx, config) {
  ctx.tools.register(defineTool({
    name: 'control_library_search',
    description: '查找适合当前控制任务的本地技能、案例、源码、知识文档和已有验证记录。请在选取建模、调参或轻量化路线时，给出具体对象、语言、方法和交付物。跨角色任务可以省略 role。',
    parameters: {
      query: { type: 'string', required: true, description: '要完成的任务，可用中文或英文。' },
      role: { type: 'string', enum: roleIds, description: '仅搜索该角色的技能。' },
      limit: { type: 'integer', description: '返回 1 至 8 个候选，默认 3。' },
    },
    output,
    execute({ query, role, limit }, { signal }) {
      const args = [`--query=${query}`];
      if (role !== undefined) args.push(`--role=${role}`);
      if (limit !== undefined) args.push(`--limit=${limit}`);
      return lookup(ctx, config, args, signal);
    },
  }));
  ctx.tools.register(defineTool({
    name: 'control_library_read',
    description: '读取控制库中登记的技能说明、知识、案例、Python/MATLAB 源码或历史报告，返回内容、路径、文件哈希和续读位置。取用搜索结果中的相对路径。',
    parameters: {
      path: { type: 'string', required: true, description: '库内相对路径，保留原目录和扩展名。' },
      offset: { type: 'integer', description: '从零开始的字符偏移，可用上一次的 next_offset 续读。' },
      length: { type: 'integer', description: '读取 1 至 12000 个字符，默认 4000。' },
    },
    output,
    execute({ path, offset, length }, { signal }) {
      const args = [`--read=${path}`];
      if (offset !== undefined) args.push(`--offset=${offset}`);
      if (length !== undefined) args.push(`--length=${length}`);
      return lookup(ctx, config, args, signal);
    },
  }));
}
