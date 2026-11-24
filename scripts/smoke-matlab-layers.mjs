import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { openLayer, readFixture, requireUnboundFixture, vendor } from './matlab-layer-harness.mjs';

const {fixture,marker,config,resource} = await readFixture(process.argv[2]);
await requireUnboundFixture(resource);
const reportPath = join(fixture, 'layer-evidence-' + randomUUID() + '.json');
const workspace = await mkdtemp(join(fixture,'layers-'));
const report = {workspace, stages:[]};
const evaluate = 'evaluate_matlab_code';
const code = "fprintf('ACTUAL=%s\\n',jsonencode(struct('release',version('-release'),'pid',feature('getpid')," +
  "'session',getenv('CONTROL_HARNESS_SESSION_ID'),'connector',which('shareMATLABSession')," +
  "'modelEdit',which('model_edit'),'librarySettings',library.settingsLookup(),'sum',2+3)));";
const args = {code, project_path:workspace};
function verify(raw) {
  const line = raw.content.flatMap(x => x.type === 'text' ? x.text.split(/\r?\n/) : []).find(x => x.startsWith('ACTUAL='));
  assert.ok(line);
  const actual = JSON.parse(line.slice(7));
  assert.equal(actual.pid,marker.matlab_pid); assert.equal(actual.session,marker.session_id);
  assert.equal(actual.release,marker.matlab_release); assert.equal(actual.sum,5);
  assert.equal(resolve(actual.connector),join(marker.toolbox_root,'shareMATLABSession.m'));
  assert.equal(resolve(actual.modelEdit),join(marker.toolkit_root,'tools/model_edit/model_edit.p'));
}
let current;
try {
  // Real MCP SDK stdio, before loading any Cordis or project tool plugin.
  current = 'official-stdio';
  const require = createRequire(join(vendor, 'packages/mcp/mcp-client/package.json'));
  const sdk = name => import(pathToFileURL(require.resolve(name)));
  const [{Client},{StdioClientTransport}] = await Promise.all([sdk('@modelcontextprotocol/client'),sdk('@modelcontextprotocol/client/stdio')]);
  const client = new Client({name:'control-harness-official-baseline',version:'1.0.0'},
    {capabilities:{},versionNegotiation:{mode:'auto'}});
  try {
    await client.connect(new StdioClientTransport({...config,env:{...config.env,
      SystemRoot:process.env.SystemRoot, LOCALAPPDATA:process.env.LOCALAPPDATA, USERPROFILE:process.env.USERPROFILE}}));
    const names = (await client.listTools()).tools.map(tool => tool.name);
    const raw = await client.callTool({name:evaluate,arguments:args},{timeout:120000});
    report.stages.push({layer:current, tools:names, raw});
    assert.ok(names.includes('model_read') && names.includes(evaluate));
    verify(raw);
  } finally { await client.close(); }
  for (const managed of [false,true]) {
    current = managed ? 'harness-managed' : 'harness-native';
    const layer = await openLayer({fixture,workspace,managed});
    try {
      const names = layer.ctx.tools.definitions().map(tool => tool.name).filter(name=>name.startsWith('mcp__matlab__'));
      const raw = await layer.call('mcp__matlab__' + evaluate,args);
      report.stages.push({layer:current,tools:names,raw});
      assert.equal(raw.isError,false,raw.error?.message);
      verify(raw.value);
      if (managed) {
        const state = JSON.parse(await readFile(join(resource,'resource.json'),'utf8'));
        assert.equal(state.owner,null); assert.equal(state.quarantine,null);
        assert.equal(state.last_completion.evidence.matlab_pid,marker.matlab_pid);
        report.stages.at(-1).terminalEvidence = state.last_completion.evidence;
      }
    } finally { await layer.close(); }
  }
  report.status = 'PASS';
} catch (error) {
  report.status = 'FAIL'; report.stage = current; process.exitCode = 1;
  report.error = error.message;
} finally {
  await writeFile(reportPath,JSON.stringify(report,null,2));
  console.log(JSON.stringify({status:report.status, stage:report.stage, reportPath}));
}
