import { createHash } from 'node:crypto';
import { readdir, readFile, realpath } from 'node:fs/promises';
import { resolve } from 'node:path';

// Cordis mounts this optional contribution only when dsh-knowledge provides
// its service. The role tools and fixed asset lookup have no dependency on it.
export const name = 'control-library-knowledge';
const baseName = '控制工程技能库';
const description = 'control-skill-library 随包知识文档，由控制工程工作台同步。';

export async function syncKnowledge(knowledge, directory) {
  const source = await realpath(directory);
  const entries = await readdir(source, { withFileTypes: true });
  const snapshot = JSON.parse(await readFile(resolve(source, '../snapshot.json'), 'utf8'));
  const articles = Object.keys(snapshot.files).filter(path => path.startsWith('knowledge/'));
  if (!articles.length || entries.length !== articles.length) throw new Error('知识目录与快照的文章数不一致。');
  for (const entry of entries) {
    const expected = snapshot.files[`knowledge/${entry.name}`];
    if (!expected || !entry.isFile() || !entry.name.endsWith('.md')) throw new Error('知识目录包含未登记的文档。');
    const bytes = await readFile(resolve(source, entry.name));
    if (createHash('sha256').update(bytes).digest('hex') !== expected.sha256) {
      throw new Error(`知识文档与快照版本不一致：${entry.name}`);
    }
  }
  await knowledge.whenReady();
  // 0.5.4 inherits a non-empty global reranker even for explicit lexical
  // searches; an empty per-base rerankModel cannot override that setting.
  const reranker = knowledge.getConfig().rerankModel?.trim();
  if (reranker && !reranker.startsWith('local:')) throw new Error('知识库启用了远程重排。请使用本地领域检索。');
  const bases = knowledge.listBases();
  const managed = bases.filter(base => base.description === description);
  if (managed.length > 1 || (!managed.length && bases.some(base => base.name === baseName))) {
    throw new Error('知识库名称或来源标记冲突，请在知识库面板核对来源。');
  }
  const base = managed[0] ?? await knowledge.createBase({
    name: baseName, description,
    config: {
      embeddingProvider: 'none', searchMode: 'lexical', documentProcessorProvider: 'builtin',
      semanticChunk: false, imageCaptionProvider: 'off', autoRetrieve: false,
    },
  });
  const config = knowledge.getConfigFor(base.id);
  const baseReranker = config.rerankModel?.trim();
  if (config.embeddingProvider !== 'none' || config.documentProcessorProvider !== 'builtin'
      || config.imageCaptionProvider !== 'off' || config.semanticChunk
      || (baseReranker && !baseReranker.startsWith('local:'))) {
    throw new Error('知识库使用了自定义文档处理或向量化设置，请在知识库面板手动同步。');
  }
  if (!managed.length && knowledge.isEnabled()) {
    // Empty means all bases, including ones added in the future. Respect
    // that setting and a user's later decision to exclude this base. Set
    // initial visibility before ingestion so a failed import can be retried.
    const selected = knowledge.getEnabledBaseIds();
    if (selected.length && !selected.includes(base.id)) await knowledge.setEnabledBaseIds([...selected, base.id]);
  }
  const result = await knowledge.importDirectoryTree(base.id, source);
  if (result.sync.failed || result.errors.length) {
    throw new Error(`控制工程知识文档未全部同步：${result.errors.map(item => item.file).join(', ')}`);
  }
  return { baseId: base.id, sync: result.sync };
}

export function apply(ctx) {
  ctx.inject(['knowledge'], async scope => {
    try {
      await syncKnowledge(scope.knowledge, resolve(import.meta.dirname, '../control-skill-library/knowledge'));
    } catch (error) {
      scope.logger.warn(`控制工程知识库同步失败。领域检索工具仍可使用。 ${error.message}`);
    }
  });
}
