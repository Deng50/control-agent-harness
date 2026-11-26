import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { mkdir, mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installLibraryTools } from '../src/library-tools.mjs';
import * as libraryKnowledge from '../src/library-knowledge.mjs';

const root = resolve(import.meta.dirname, '../..');
const library = resolve(root, 'harness-plugin/control-skill-library');
const load = path => import(pathToFileURL(resolve(root, path)));
const vendor = 'vendor/deepseek-harness';
const knowledgePlugin = '.dsh/profiles/web/node_modules/dsh-knowledge/lib/knowledge/index.js';
const knowledgeTools = '.dsh/profiles/web/node_modules/dsh-knowledge/lib/tool-knowledge/index.js';

test('domain tools query and read the bundled catalog through the native subprocess service', async () => {
  const { Context } = await load(`${vendor}/vendor/cordis/lib/index.js`);
  const subprocess = await load(`${vendor}/packages/subprocess/subprocess-local/lib/index.js`);
  const tools = await load(`${vendor}/packages/core/tools/lib/index.js`);
  const ctx = new Context();
  try {
    const prompt = await load(`${vendor}/packages/core/system-prompt/lib/index.js`);
    // Professional presets also boot without an installed knowledge bundle.
    await ctx.plugin(libraryKnowledge);
    assert.equal(ctx.get('knowledge'), undefined);
    await ctx.plugin(prompt.default);
    await ctx.plugin(tools.default);
    await ctx.plugin(subprocess.default);
    const pythonPath = process.env.CONTROL_PYTHON ?? resolve(root,
      process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python');
    installLibraryTools(ctx, { codeRoot: root, pythonPath });
    const search = ctx.tools.get('control_library_search');
    const read = ctx.tools.get('control_library_read');
    const execution = { signal: new AbortController().signal };
    for (const [role, query, skill] of [
      ['model', '灰箱辨识', 'identify-greybox-parameters'],
      ['controller', 'MATLAB Function 原生MPC', 'build-native-mpc'],
      ['lightweight', '热管理轻量化', 'distill-thermal-expert'],
    ]) {
      const { matches } = await search.execute({ query, role, limit: 1 }, execution);
      assert.equal(matches[0]?.skill, skill);
      const path = matches[0].code;
      const source = await readFile(resolve(library, path));
      const first = await read.execute({ path, length: 18 }, execution);
      const second = await read.execute({ path, offset: first.next_offset, length: 18 }, execution);
      assert.equal(first.sha256, createHash('sha256').update(source).digest('hex'));
      assert.equal(first.text + second.text, [...source.toString('utf8').replace(/^\uFEFF/, '')].slice(0, 36).join(''));
    }
    await assert.rejects(read.execute({ path: '../../README.md' }, execution));
    await assert.rejects(search.execute({ query: 'MPC', limit: 0 }, execution));
    assert.deepEqual((await search.execute({ query: '--qqqzzzz' }, execution)).matches, []);
  } finally { await ctx.fiber.dispose(); }
});

test('optional knowledge plugin imports packaged articles locally and reuses its own directory', {
  skip: !existsSync(resolve(root, knowledgePlugin)) && 'dsh-knowledge is not installed in the Web profile',
}, async t => {
  const originalHome = process.env.DSH_HOME;
  const output = resolve(root, 'test-output');
  await mkdir(output, { recursive: true });
  const home = await mkdtemp(join(output, 'control-knowledge-test-'));
  process.env.DSH_HOME = home;
  const { Context } = await load(`${vendor}/vendor/cordis/lib/index.js`);
  const ctx = new Context();
  try {
    // The Web loader supplies host packages to external bundles. Standalone
    // node:test resolves those same two imports from the pinned build.
    const hostPackages = new Map([
      ['@deepseek-ai/cordis', 'vendor/cordis/lib/index.js'],
      ['@deepseek-ai/dsh-storage-domain', 'packages/storage/storage-domain/lib/index.js'],
      ['@deepseek-ai/dsh-tools', 'packages/core/tools/lib/index.js'],
    ]);
    const pluginModules = [knowledgePlugin, knowledgeTools].map(path => pathToFileURL(resolve(root, path)).href);
    const hooks = registerHooks({ resolve(specifier, context, next) {
      if (pluginModules.includes(context.parentURL) && hostPackages.has(specifier)) {
        return next(pathToFileURL(resolve(root, vendor, hostPackages.get(specifier))).href, context);
      }
      return next(specifier, context);
    } });
    let serviceModule, toolsModule;
    try { [serviceModule, toolsModule] = await Promise.all([load(knowledgePlugin), load(knowledgeTools)]); }
    finally { hooks.deregister(); }
    ctx.provide('webServer', { register: () => () => {} });
    const storage = await load(`${vendor}/packages/storage/storage/lib/index.js`);
    const json = await load(`${vendor}/packages/storage/storage-json/lib/index.js`);
    const domain = await load(`${vendor}/packages/storage/storage-domain/lib/index.js`);
    await ctx.plugin(storage.default);
    await ctx.plugin(json, { root: join(home, 'storages') });
    await ctx.plugin(domain, { backend: 'json' });
    const prompt = await load(`${vendor}/packages/core/system-prompt/lib/index.js`);
    const tools = await load(`${vendor}/packages/core/tools/lib/index.js`);
    await ctx.plugin(prompt.default);
    await ctx.plugin(tools.default);
    await ctx.plugin(serviceModule.KnowledgeService);
    await ctx.plugin(toolsModule);
    const service = ctx.get('knowledge');
    const userBase = await service.createBase({ name: 'existing-project' });
    await service.setEnabledBaseIds([userBase.id]);
    const docs = join(library, 'knowledge');
    const files = await readdir(docs);
    const imported = Promise.withResolvers();
    const importDirectory = service.importDirectoryTree.bind(service);
    const observe = t.mock.method(service, 'importDirectoryTree', (...args) => {
      const result = importDirectory(...args);
      imported.resolve(result);
      return result;
    });
    await ctx.plugin(libraryKnowledge);
    await imported.promise;
    observe.mock.restore();
    const managed = service.listBases().find(base => base.name === '控制工程技能库');
    assert.ok(managed);
    assert.deepEqual(service.listDocuments(managed.id).filter(doc => doc.sourceType === 'file')
      .map(doc => doc.title).sort(), files.sort());
    const second = await libraryKnowledge.syncKnowledge(service, docs);
    assert.equal(second.baseId, managed.id);
    assert.equal(second.sync.status, 'unchanged');
    assert.deepEqual(service.getEnabledBaseIds(), [userBase.id, managed.id]);
    assert.equal(service.listDocuments(userBase.id).length, 0);
    const search = ctx.tools.get('knowledge_search');
    const params = { baseId: managed.id, query: 'MPC', mode: 'lexical', topK: 3 };
    const execution = { signal: new AbortController().signal };
    const result = await search.execute(params, execution);
    assert.ok(result.hits.length);
    assert.ok(result.hits.every(hit => files.includes(hit.documentTitle)));
    await service.setEnabledBaseIds([userBase.id]);
    await libraryKnowledge.syncKnowledge(service, docs);
    await assert.rejects(search.execute(params, execution), /not enabled/);
    await service.deleteBase(managed.id);
    await service.setEnabledBaseIds([]);
    await libraryKnowledge.syncKnowledge(service, docs);
    assert.deepEqual(service.getEnabledBaseIds(), []);
  } finally {
    await ctx.fiber.dispose();
    if (originalHome === undefined) delete process.env.DSH_HOME;
    else process.env.DSH_HOME = originalHome;
    await rm(home, { recursive: true, force: true });
  }
});
