import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile, link } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { buildZohPlantTool, readPlantReference } from '../src/simulink-build.mjs';

const project=resolve(import.meta.dirname,'../..');

test('the public ZOH tool schema compiles with nested trajectory arrays and advertises the confined source replay tool', () => {
  const tool = buildZohPlantTool({}, {codeRoot:project,pythonPath:join(project,'.venv/Scripts/python.exe')}, () => {});
  assert.equal(tool.name,'control_build_zoh_plant');
  assert.match(tool.description,/control_replay_plant_source/);
});

test('reference file is bounded, plain and inside the current workspace', async () => {
  const root=join(project,'test-output','host-zoh-regression');
  await mkdir(root,{recursive:true});
  const directory=await mkdtemp(join(root,'reference-'));
  await writeFile(join(directory,'轨迹.json'),'{"absTolerance":0.00001}');
  assert.deepEqual(await readPlantReference(directory,'轨迹.json'),{absTolerance:0.00001});
  for (const name of ['../轨迹.json','C:/elsewhere.json','sub/trace.json','NUL.json','trace.txt']) {
    await assert.rejects(readPlantReference(directory,name),/单个 JSON/);
  }
  await link(join(directory,'轨迹.json'),join(directory,'alias.json'));
  await assert.rejects(readPlantReference(directory,'alias.json'),/单链接/);
  await writeFile(join(directory,'large.json'),Buffer.alloc(2*1024*1024+1));
  await assert.rejects(readPlantReference(directory,'large.json'),/2 MiB/);
});
