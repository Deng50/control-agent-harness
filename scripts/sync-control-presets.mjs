/** Copy the pinned native standard composition; do not maintain a second tool roster. */
import { readFile, writeFile, stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { agents } from '../harness-plugin/src/roles.mjs';

export const project = resolve(import.meta.dirname, '..');
export const upstream = resolve(project, 'vendor/deepseek-harness');
export const upstreamCommit = 'f3803c96974ad4a94195fb97e5c637e9c2a3afe3';
export const standardRelative = 'packages/bundle/web-app/presets/standard.patch.yml';
export const patchPath = resolve(project, 'harness-plugin/cordis.patch.yml');
export const roleIds = ['model', 'controller', 'lightweight'];
export const jsTag = 'tag:yaml.org,2002:js';
export const skillRootExpression = "process.env.CONTROL_CODE_ROOT + '/harness-plugin/skills'";
export const librarySkillRootExpression = "process.env.CONTROL_CODE_ROOT + '/harness-plugin/control-skill-library/.agents/skills'";

// YAML is an already installed dependency of the pinned native skill provider.
// Resolving there avoids a new global dependency or a second lockfile owner.
const require = createRequire(resolve(upstream, 'packages/skill/skill-filesystem/package.json'));
export const YAML = require('yaml');

export function parseTaggedYaml(text) {
  const document = YAML.parseDocument(text, {
    // Always quote expressions: a leading ! is YAML tag syntax, not a JS token.
    customTags: [{ tag: jsTag, resolve: value => value, stringify: item => JSON.stringify(item.value) }],
  });
  if (document.errors.length) throw new Error(document.errors.map(error => error.message).join('\n'));
  return document;
}

export function insertedRows(document) {
  const insertion = document.contents?.items.find(item => item.has?.('insert'));
  const rows = insertion?.get('insert', true);
  if (!YAML.isSeq(rows)) throw new Error('Expected an insert sequence in the preset patch');
  return rows;
}

/** Read git metadata only; setup-harness already pins this separate checkout. */
export async function verifyUpstreamRevision() {
  const marker = resolve(upstream, '.git');
  const gitDirectory = (await stat(marker)).isDirectory()
    ? marker
    : resolve(upstream, (await readFile(marker, 'utf8')).trim().replace(/^gitdir:\s*/, ''));
  const head = (await readFile(resolve(gitDirectory, 'HEAD'), 'utf8')).trim();
  let revision = head;
  if (head.startsWith('ref: ')) {
    const reference = head.slice(5);
    try { revision = (await readFile(resolve(gitDirectory, reference), 'utf8')).trim(); }
    catch (error) {
      if (error.code !== 'ENOENT') throw error;
      const packed = await readFile(resolve(gitDirectory, 'packed-refs'), 'utf8');
      revision = packed.split('\n').find(line => line.endsWith(` ${reference}`))?.split(' ')[0];
    }
  }
  if (revision !== upstreamCommit) throw new Error(`Expected native Harness ${upstreamCommit}; found ${revision}`);
  return revision;
}

function expression(value) {
  const node = new YAML.Scalar(value);
  node.tag = jsTag;
  return node;
}

function findRow(sequence, id) {
  for (const row of sequence.items) {
    if (row.get?.('id') === id) return row;
    if (row.get?.('group') === true) {
      const nested = row.get('config', true);
      if (YAML.isSeq(nested)) {
        const found = findRow(nested, id);
        if (found) return found;
      }
    }
  }
  return undefined;
}

export async function expectedPatch() {
  await verifyUpstreamRevision();
  const source = parseTaggedYaml(await readFile(resolve(upstream, standardRelative), 'utf8'));
  const native = insertedRows(source).items.find(row => row.get('id') === 'preset-standard');
  const plugins = native?.getIn(['config', 'plugins'], true);
  if (!YAML.isSeq(plugins)) throw new Error('Pinned standard preset has no plugins sequence');
  const target = parseTaggedYaml(await readFile(patchPath, 'utf8'));
  const targetRows = insertedRows(target);
  for (const role of roleIds) {
    const preset = targetRows.items.find(row => row.get('id') === `preset-control-${role}`);
    if (!preset) throw new Error(`Missing control-${role} declaration`);
    preset.setIn(['config', 'name'], agents[role].title);
    const copied = plugins.clone();
    const persona = findRow(copied, 'persona');
    const jobs = findRow(copied, 'tool-jobs');
    const skills = findRow(copied, 'skill-filesystem');
    const subagent = findRow(copied, 'tool-subagent');
    const market = findRow(copied, 'tool-plugin-market');
    if (!persona || !jobs || !skills || !subagent || !market) throw new Error('Pinned standard composition changed required rows');
    persona.set('name', '@control-engineering/harness-bundle/role');
    persona.set('config', target.createNode({ role }));
    jobs.set('config', target.createNode({ completionDelivery: 'quiet' }));
    skills.set('config', target.createNode({ includeDefaultRoots: true, customSkillDirs: [] }));
    const skillRoots = skills.getIn(['config', 'customSkillDirs'], true);
    // The harness-specific entry skills precede their upstream counterparts.
    skillRoots.add(expression(skillRootExpression));
    skillRoots.add(expression(librarySkillRootExpression));
    // Native catalog tools register when a profile supplies the optional market service.
    market.delete('disabled');
    subagent.setIn(['config', 'allowedAgentPresets'], target.createNode(roleIds.map(id => `control-${id}`)));
    preset.setIn(['config', 'plugins'], copied);
  }
  target.commentBefore = [
    ' Generated by scripts/sync-control-presets.mjs --write.',
    ` Native standard composition: ${upstreamCommit}`,
    ` Source: ${standardRelative}`,
    ' Intentional changes: domain persona, quiet jobs, role and library skill roots, market lookup, spawn role selection.',
    ' Keep native groups/isolation and !!js expressions; do not hand-maintain the common tool list.',
  ].join('\n');
  return target.toString({ lineWidth: 0 });
}

async function main() {
  const flags = process.argv.slice(2);
  if (flags.length !== 1 || !['--check', '--write'].includes(flags[0])) {
    throw new Error('Usage: node scripts/sync-control-presets.mjs --check|--write');
  }
  const expected = await expectedPatch();
  const actual = await readFile(patchPath, 'utf8');
  const changed = actual !== expected;
  if (flags[0] === '--write') await writeFile(patchPath, expected);
  console.log(JSON.stringify({ upstreamCommit, roles: roleIds, changed, action: flags[0].slice(2) }));
  if (flags[0] === '--check' && changed) process.exitCode = 1;
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
