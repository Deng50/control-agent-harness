/** Keyless acceptance harness. Only mounts a private, already-running test MATLAB. */
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, realpath } from 'node:fs/promises';
import { dirname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { apply } from '../harness-plugin/src/index.mjs';

export const project = resolve(import.meta.dirname, '..');
export const vendor = resolve(project, 'vendor/deepseek-harness');
const built = path => import(pathToFileURL(join(vendor, path, 'lib/index.js')));

export async function readFixture(fixture) {
  for (const name of ['MW_MCP_SERVER_MATLAB_SESSION_CONNECTION_DETAILS','MW_MCP_SERVER_MATLAB_ROOT',
    'MW_MCP_SERVER_INITIAL_WORKING_FOLDER','MW_MCP_SERVER_MATLAB_DISPLAY_MODE']) {
    assert.ok(!Object.hasOwn(process.env,name), 'An inherited MCP selector conflicts with the dedicated fixture: ' + name);
  }
  fixture = await realpath(resolve(fixture));
  assert.equal(dirname(fixture), await realpath(join(project, 'test-output')));
  let launch = 'session';
  try {
    launch = (await readFile(join(fixture, 'active-test-launch.txt'), 'utf8')).trim();
    assert.match(launch, /^session-[a-zA-Z0-9-]+$/);
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const appdata = join(fixture, launch, 'appdata');
  assert.equal(await realpath(appdata), appdata);
  const identityFile = join(appdata, 'control-harness-session.json');
  const marker = JSON.parse(await readFile(identityFile, 'utf8'));
  assert.equal(marker.kind, 'control-harness-dedicated-matlab-v1');
  assert.equal(marker.status, 'ready');
  assert.equal(resolve(marker.private_appdata), appdata);
  assert.equal(resolve(marker.project_root), project);
  assert.ok(marker.mcp_executable && marker.toolbox_root);
  const config = {command: marker.mcp_executable, args: ['--matlab-session-mode=existing',
    '--extension-file=' + join(marker.toolkit_root,'tools/tools.json'), '--disable-telemetry=true'],
    env: {APPDATA: appdata}, cwd: project};
  return {fixture, marker, identityFile, config, resource: join(fixture, 'resource')};
}

// Once a broker owns this resource, even test calls must use its managed path.
export async function requireUnboundFixture(resource) {
  try {
    const state = JSON.parse(await readFile(join(resource,'resource.json'),'utf8'));
    assert.ok(!state.quarantine && !state.owner && !state.queue?.length && !state.session,
      'Direct controls require a fresh unbound fixture; inspect the existing broker.');
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
}

/** Keep compiler caches from raw model edits and simulations inside this disposable fixture. */
export async function configureFixtureFileGeneration(layer,fixture) {
  const folder = join(fixture,'generated');
  const literal = path => "'" + path.replaceAll("'","''") + "'";
  const code = "Simulink.fileGenControl('set','CacheFolder'," + literal(join(folder,'cache')) +
    ",'CodeGenFolder'," + literal(join(folder,'codegen')) + ",'keepPreviousPath',true,'createDir',true);";
  const result = await layer.call('mcp__matlab__evaluate_matlab_code',{code});
  assert.equal(result.isError,false,result.error?.message);
}

/** Attach the reviewed fixture to one native chat, using only its private store. */
export async function attachFixtureSource(layer,fixture,file) {
  const [attachments,llm]=await Promise.all(['packages/attachment/attachment-local','packages/llm/llm'].map(built));
  await layer.ctx.plugin(attachments.default,{dshHome:join(fixture,'test-homes',layer.agent.id)});
  const attachment=await layer.ctx.attachments.saveFile(file);
  layer.agent.session.append('turn/start',{turn:1});
  layer.agent.session.append('user/message',llm.createUserMessage({source:{kind:'user'},
    content:[{type:'file',attachment},{type:'text',text:'请用附件的原始 step 验证模型轨迹。'}]}),{surfaceOp:'append'});
  return attachment;
}

export async function openLayer({fixture, workspace, managed, toolCallTimeoutMs=120000}) {
  const {marker, identityFile, config, resource} = await readFixture(fixture);
  if (!managed) await requireUnboundFixture(resource);
  workspace = await realpath(resolve(workspace));
  assert.ok(workspace.startsWith(await realpath(join(project,'test-output')) + sep));
  const pythonPath = join(project, '.venv/Scripts/python.exe');
  if (managed) {
    await mkdir(resource, {recursive:true});
    assert.equal(await realpath(resource), resource);
    const bound = spawnSync(pythonPath, ['-m','control_harness.matlab_resource','bind-session',
      '--directory',resource,'--pid',String(marker.matlab_pid),'--session-id',marker.session_id],
      {cwd:project, env:{...process.env, PYTHONPATH:join(project,'src')}, windowsHide:true, encoding:'utf8'});
    assert.equal(bound.status, 0, bound.stderr || bound.error?.message);
  }
  const [{Context},{Session},prompt,tools,executor,local,projection,agents,jobs,commands,policy,sandbox,mcp] =
    await Promise.all(['vendor/cordis','packages/core/session','packages/core/system-prompt','packages/core/tools',
      'packages/core/tool-executor-native','packages/subprocess/subprocess-local','packages/session/session-projection',
      'packages/core/agent','packages/jobs/jobs-local','packages/interaction/commands','packages/sandbox/sandbox-policy',
      'packages/sandbox/sandbox-local'].map(built).concat(import('../harness-plugin/src/matlab-mcp-client.mjs')));
  const ctx = new Context();
  try {
    for (const component of [prompt.default, tools.default, executor, local.default, projection.default,
      agents.default, jobs.default, commands.default, sandbox.default]) await ctx.plugin(component);
    await ctx.plugin(policy.default, {mode:'workspace-write',workspaceRoot:workspace});
    await ctx.plugin(mcp, {...config,transport:'stdio',serverName:'matlab',toolCallTimeoutMs,
      failOnStartupError:true,reconnect:{enabled:false}});
    if (managed) apply(ctx, {codeRoot:project,pythonPath,maxConcurrentWorkers:1,timeoutSeconds:120,cancellationGraceMs:5000,
      matlabResource:{directory:resource,identityFile,toolPrefixes:['mcp__matlab__'],pollIntervalMs:25}});
    function createAgent(agentPreset) {
      const id = 'session-' + randomUUID();
      const session = Session.create(id,undefined,{...Session.create(id).header,cwd:workspace,agentPreset});
      return {id,session,ctx:ctx.plugin(()=>{}).ctx,options:{}};
    }
    const agent = createAgent('control-model');
    async function call(name,args,caller=agent) {
      const batch = await ctx.tools.execute({agent:caller,signal:AbortSignal.timeout(150000),
        calls:[{type:'tool_call',id:randomUUID(),name,arguments:JSON.stringify(args)}]});
      return batch.results[0].result;
    }
    return {ctx, agent, marker, resource, createAgent, call, close:() => ctx.fiber.dispose()};
  } catch (error) {
    await ctx.fiber.dispose();
    throw error;
  }
}
