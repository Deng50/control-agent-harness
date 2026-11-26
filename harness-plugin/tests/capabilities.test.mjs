/** Native composition parity plus real scoped skill/tool registries; no model API calls. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { resolve, join, relative } from 'node:path';
import { pathToFileURL } from 'node:url';
import { runInNewContext } from 'node:vm';
import { spawnSync } from 'node:child_process';
import * as rolePlugin from '../src/role.mjs';
import { rolePrompt } from '../src/roles.mjs';
import {
  project, upstream, standardRelative, upstreamCommit, patchPath, roleIds,
  YAML, jsTag, skillRootExpression, librarySkillRootExpression, parseTaggedYaml, insertedRows, verifyUpstreamRevision,
} from '../../scripts/sync-control-presets.mjs';

const built = async path => import(pathToFileURL(resolve(upstream, path)));
const patch = parseTaggedYaml(await readFile(patchPath, 'utf8'));
const standard = parseTaggedYaml(await readFile(resolve(upstream, standardRelative), 'utf8'));
const nativePlugins = insertedRows(standard).items[0].getIn(['config', 'plugins'], true);
const controlPreset = role => insertedRows(patch).items.find(row => row.get('id') === `preset-control-${role}`);

function rows(sequence) {
  return sequence.items.flatMap(row => [row, ...(row.get('group') === true ? rows(row.get('config', true)) : [])]);
}

function rowById(sequence, id) {
  const row = rows(sequence).find(item => item.get('id') === id);
  assert.ok(row, `Missing composition row ${id}`);
  return row;
}

/** Compare semantics, including YAML tags, rather than formatting/quote styles. */
function semantic(node) {
  if (YAML.isScalar(node)) return node.tag ? { tag: node.tag, value: node.value } : node.value;
  if (YAML.isSeq(node)) return node.items.map(semantic);
  if (YAML.isMap(node)) return Object.fromEntries(node.items.map(pair => [pair.key.value, semantic(pair.value)]));
  return node;
}

function executableConfig(row) {
  function value(node) {
    if (YAML.isScalar(node) && node.tag === jsTag) {
      return runInNewContext(node.value, { process: { env: { CONTROL_CODE_ROOT: project }, platform: process.platform } });
    }
    if (YAML.isScalar(node)) return node.value;
    if (YAML.isSeq(node)) return node.items.map(value);
    if (YAML.isMap(node)) return Object.fromEntries(node.items.map(pair => [pair.key.value, value(pair.value)]));
    return node;
  }
  return value(row.get('config', true));
}

test('three control presets retain every pinned standard plugin, config and isolation group', async () => {
  assert.equal(await verifyUpstreamRevision(), upstreamCommit);
  const baseline = nativePlugins.clone();
  rowById(baseline, 'tool-plugin-market').delete('disabled');
  const expected = semantic(baseline);
  for (const role of roleIds) {
    const plugins = controlPreset(role).getIn(['config', 'plugins'], true);
    const flat = rows(plugins);
    assert.equal(flat.filter(row => row.get('name') === '@control-engineering/harness-bundle/role').length, 1);
    assert.equal(flat.filter(row => row.get('name') === '@deepseek-ai/dsh-persona').length, 0);
    assert.equal(flat.filter(row => row.get('name') === '@deepseek-ai/dsh-skill-filesystem').length, 1);
    assert.equal(rowById(plugins, 'persona').getIn(['config', 'role']), role);
    assert.equal(rowById(plugins, 'tool-jobs').getIn(['config', 'completionDelivery']), 'quiet');
    const skill = rowById(plugins, 'skill-filesystem');
    assert.equal(skill.getIn(['config', 'includeDefaultRoots']), true);
    const roots = skill.getIn(['config', 'customSkillDirs'], true).items;
    assert.ok(roots.every(item => item.tag === jsTag));
    assert.deepEqual(roots.map(item => item.value), [skillRootExpression, librarySkillRootExpression]);
    assert.equal(rowById(plugins, 'tool-plugin-market').get('disabled'), undefined);
    const spawn = rowById(plugins, 'tool-subagent');
    assert.deepEqual(spawn.getIn(['config', 'allowedAgentPresets']).toJSON(), roleIds.map(id => `control-${id}`));
    assert.equal(rowById(plugins, 'tool-subagent-fork').getIn(['config', 'allowedAgentPresets']), undefined);
    assert.equal(rowById(plugins, 'workflow-ptc').getIn(['config', 'agentPreset']), undefined);
    assert.equal(rowById(plugins, 'tool-workflow').getIn(['config', 'agentPreset']), undefined);

    // Undo only the documented domain deltas, then compare the full nested composition.
    const comparison = plugins.clone();
    for (const id of ['persona', 'tool-jobs', 'skill-filesystem', 'tool-subagent']) {
      const actual = rowById(comparison, id);
      const original = rowById(nativePlugins, id);
      actual.set('name', original.get('name'));
      const config = original.get('config', true);
      if (config === undefined) actual.delete('config'); else actual.set('config', config.clone());
    }
    assert.deepEqual(semantic(comparison), expected, `${role}: native composition drift`);
  }
});

test('static patch synchronization is idempotent and preserves executable YAML tags', () => {
  const checked = spawnSync(process.execPath, [resolve(project, 'scripts/sync-control-presets.mjs'), '--check'], {
    encoding: 'utf8', windowsHide: true,
  });
  assert.equal(checked.status, 0, checked.stderr || checked.stdout);
  assert.equal(JSON.parse(checked.stdout).changed, false);
  const roundTrip = parseTaggedYaml(patch.toString({ lineWidth: 0 }));
  assert.deepEqual(semantic(roundTrip.contents), semantic(patch.contents));
  for (const role of roleIds) {
    const plugins = controlPreset(role).getIn(['config', 'plugins'], true);
    assert.equal(rowById(plugins, 'tool-plugin-manager').get('disabled', true).tag, jsTag);
    assert.equal(rowById(plugins, 'tool-plugin-manager').get('disabled'), "!ctx.get('profileContext')");
    assert.equal(rowById(plugins, 'tool-pwsh').get('disabled', true).tag, jsTag);
  }
});

test('native spawn tool advertises only control presets and rejects other choices before dispatch', async () => {
  const [cordis, prompt, tools, projections, subagentTool, sessionModule] = await Promise.all([
    built('vendor/cordis/lib/index.js'), built('packages/core/system-prompt/lib/index.js'),
    built('packages/core/tools/lib/index.js'), built('packages/session/session-projection/lib/index.js'),
    built('packages/subagent/tool-subagent/lib/index.js'), built('packages/core/session/lib/index.js'),
  ]);
  const ctx = new cordis.Context();
  let dispatches = 0;
  try {
    await ctx.plugin(prompt.default);
    await ctx.plugin(tools.default);
    await ctx.plugin(projections.default);
    // Test collaborator stops at the native tool boundary; real provider startup is
    // checked by smoke-native-capabilities rather than calling a model here.
    const provider = {
      name: 'spawn', inheritsParentContext: false,
      capabilities: { agentPreset: true, depthLimit: true, agentOptions: true },
    };
    ctx.provide('agentPresets', {});
    ctx.provide('subagents', {
      getProvider: name => name === 'spawn' ? provider : undefined,
      resolveMaxDepth: value => value,
      startActivation: async () => { dispatches += 1; throw new Error('Unexpected native provider dispatch'); },
    });
    const nativeSpawn = executableConfig(rowById(controlPreset('model').getIn(['config', 'plugins'], true), 'tool-subagent'));
    await ctx.plugin(subagentTool, nativeSpawn);
    const definition = ctx.tools.get('subagent');
    assert.ok(definition, 'Actual native spawn tool must register');
    assert.deepEqual(definition.parameters.properties.agent_preset.enum, roleIds.map(role => `control-${role}`));
    const session = sessionModule.Session.create('preset-preflight', [], {
      version: sessionModule.SESSION_FORMAT_VERSION, id: 'preset-preflight', cwd: project,
      createdAt: 0, isSeeded: false,
    });
    await assert.rejects(definition.execute({
      description: 'Validate preset boundary', prompt: 'Do not run', agent_preset: 'standard',
    }, { agent: { id: 'preset-preflight', session }, signal: new AbortController().signal }),
    /agent_preset.*must be one of.*control-model.*control-controller.*control-lightweight/);
    assert.equal(dispatches, 0, 'An invalid preset must fail before any child starts');
  } finally { await ctx.fiber.dispose(); }
});

test('skills are packaged and loaded by each real native scope alongside default MATLAB-style roots', async () => {
  const [cordis, prompt, tools, executor, projections, agents, skills, filesystem, toolSkill, scopeModule, sessionModule] = await Promise.all([
    built('vendor/cordis/lib/index.js'), built('packages/core/system-prompt/lib/index.js'),
    built('packages/core/tools/lib/index.js'), built('packages/core/tool-executor-native/lib/index.js'),
    built('packages/session/session-projection/lib/index.js'), built('packages/core/agent/lib/index.js'),
    built('packages/skill/skill/lib/index.js'), built('packages/skill/skill-filesystem/lib/index.js'),
    built('packages/skill/tool-skill/lib/index.js'), built('packages/core/scope/lib/index.js'),
    built('packages/core/session/lib/index.js'),
  ]);
  const output = resolve(process.env.CONTROL_TEST_OUTPUT ?? resolve(project, 'test-output/capability-tests'));
  await mkdir(output, { recursive: true });
  const temporary = await mkdtemp(join(output, 'capability-fixture-'));
  const ctx = new cordis.Context();
  try {
    const customRoot = resolve(project, 'harness-plugin/skills');
    const libraryRoot = resolve(project, 'harness-plugin/control-skill-library/.agents/skills');
    const manifest = JSON.parse(await readFile(resolve(project, 'harness-plugin/package.json'), 'utf8'));
    assert.ok(manifest.files.includes('skills'), 'Installed plugin must carry its skills');
    assert.ok(manifest.files.includes('control-skill-library'));
    const sources = new Map();
    for (const root of [customRoot, libraryRoot]) {
      for (const dir of await readdir(root, { withFileTypes: true })) {
        if (!dir.isDirectory()) continue;
        let text;
        try { text = await readFile(join(root, dir.name, 'SKILL.md'), 'utf8'); }
        catch (error) { if (error.code === 'ENOENT') continue; throw error; }
        const frontmatter = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text);
        assert.ok(frontmatter, `Missing skill frontmatter: ${dir.name}`);
        const name = YAML.parse(frontmatter[1]).name;
        if (!sources.has(name)) sources.set(name, root);
      }
    }
    const shipped = [...sources.keys()];
    for (const name of ['fit-state-flow', 'build-native-mpc', 'clone-control-policy']) assert.equal(sources.get(name), libraryRoot);
    const defaultRoot = join(temporary, 'user-agents/skills/matlab-capability-fixture');
    await mkdir(defaultRoot, { recursive: true });
    await writeFile(join(defaultRoot, 'SKILL.md'), '---\nname: matlab-capability-fixture\ndescription: Default user MATLAB skill discovery fixture\n---\nUse the existing MATLAB connector when the task requires MATLAB.\n');
    const workspace = join(temporary, 'workspace');
    await mkdir(workspace);
    await ctx.plugin(prompt.default);
    await ctx.plugin(tools.default);
    await ctx.plugin(projections.default);
    await ctx.plugin(executor);
    await ctx.plugin(agents.default);
    await ctx.plugin(skills.default);
    ctx.provide('controlEngineering', {});
    const marketTools = await built('packages/boot/plugin-market/lib/types/tools.js');
    ctx.provide('pluginMarket', { listIndexes: () => [] });
    // A fixed-directory collaborator, as used by native tool-skill tests; no shell/LLM runs.
    ctx.provide('workingDirectory', { ensure: async agent => agent.session.header.cwd });
    for (const role of roleIds) ctx.tools.register(tools.defineTool({
      name: `control_${role}`, description: 'Scoped role restriction fixture', parameters: {},
      output: { schema: { type: 'string' }, render: () => [] }, execute: () => '',
    }));
    for (const role of roleIds) {
      const agent = { id: `capability-${role}`, session: sessionModule.Session.create(`capability-${role}`, [], {
        version: sessionModule.SESSION_FORMAT_VERSION, id: `capability-${role}`, cwd: workspace,
        createdAt: 0, isSeeded: false,
      }) };
      const scope = scopeModule.createScope(ctx, agent);
      try {
        const plugins = controlPreset(role).getIn(['config', 'plugins'], true);
        const config = executableConfig(rowById(plugins, 'skill-filesystem'));
        assert.deepEqual(config.customSkillDirs.map(path => resolve(path)), [customRoot, libraryRoot]);
        await scope.ctx.plugin(rolePlugin, { role });
        await scope.ctx.plugin(filesystem, {
          ...config, dshHome: join(temporary, 'user-dsh'), agentsHome: join(temporary, 'user-agents'), watch: false,
        });
        await scope.ctx.plugin(toolSkill);
        await scope.ctx.plugin(marketTools);
        const market = ctx.tools.get('plugin_market', agent);
        assert.deepEqual(JSON.parse(await market.execute({ action: 'list_indexes' }, {
          agent, signal: new AbortController().signal,
        })), { indexes: [] });
        const lookup = { cwd: workspace, scope: agent };
        const discovered = await ctx.skills.list(lookup);
        const names = discovered.map(item => item.name);
        for (const name of shipped) assert.ok(names.includes(name), `${role} did not discover ${name}`);
        assert.ok(names.includes('matlab-capability-fixture'), 'Default user roots must remain enabled');
        assert.equal(new Set(names).size, names.length, 'One merged skill catalog must not duplicate names');
        const tool = ctx.tools.get('skill', agent);
        assert.ok(tool, 'Native skill loader tool is visible in the actual scoped registry');
        for (const name of [...shipped, 'matlab-capability-fixture']) {
          const loaded = await tool.execute({ name }, { agent, signal: new AbortController().signal });
          assert.equal(loaded.name, name);
          assert.ok(loaded.content.length > 0);
          if (sources.has(name)) assert.ok(!relative(sources.get(name), loaded.resourceBase.path).startsWith('..'), name);
        }
        const assembled = await ctx.systemPrompt.assemble({ scope: agent });
        const personas = assembled.sections.filter(section => section.name === 'deployment:persona-prefix');
        assert.equal(personas.length, 1);
        assert.equal(personas[0].text, rolePrompt(role));
        for (const other of roleIds) assert.equal(Boolean(ctx.tools.get(`control_${other}`, agent)), other === role);
      } finally { await scope.dispose(); }
    }
    assert.equal(ctx.tools.get('skill'), undefined, 'Preset skill loader must not leak into host scope');
  } finally {
    await ctx.fiber.dispose();
    await rm(temporary, { recursive: true, force: true });
  }
});
