/** Actual native preset selection/composition and Session replay; no provider or numerical work. */
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,stat} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {resolve,join,dirname} from 'node:path';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';
import test from 'node:test';
import {JobHistory} from '../src/job-history.mjs';

const upstream=resolve(import.meta.dirname,'../../vendor/deepseek-harness');
const built=path=>import(pathToFileURL(resolve(upstream,path)));
const nativeRequire=createRequire(resolve(upstream,'packages/preset/agent-preset-registry/package.json'));
const named=name=>import(pathToFileURL(nativeRequire.resolve(name)));
const [cordis,storage,jsonStorage,domain,sessions,workspaces,loader,group,llm,projection,prompt,tools,executor,agents,loop,presets,projectionCache]=await Promise.all([
  built('vendor/cordis/lib/index.js'),built('packages/storage/storage/lib/index.js'),built('packages/storage/storage-json/lib/index.js'),built('packages/storage/storage-domain/lib/index.js'),
  built('packages/core/session/lib/index.js'),built('packages/workspace/workspace/lib/index.js'),named('@deepseek-ai/cordis-plugin-loader'),named('@deepseek-ai/cordis-plugin-group'),
  built('packages/llm/llm/lib/index.js'),built('packages/session/session-projection/lib/index.js'),built('packages/core/system-prompt/lib/index.js'),built('packages/core/tools/lib/index.js'),
  built('packages/core/tool-executor-native/lib/index.js'),built('packages/core/agent/lib/index.js'),built('packages/core/agent-loop/lib/index.js'),built('packages/preset/agent-preset-registry/lib/index.js'),built('packages/session/session-projection-cache/lib/index.js')]);
const module=await import(pathToFileURL(process.env.CONTROL_TEST_SESSION_WORKSPACE_MODULE??resolve(import.meta.dirname,'../src/session-workspace.mjs')));
const root=resolve(import.meta.dirname,'../../test-output/professional-session-tests');await mkdir(root,{recursive:true});
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');

async function fixture(original='control-model',legacySlash=false,second=false){
  const directory=await mkdtemp(join(root,'native-')),cwd=join(directory,'project');await mkdir(cwd);
  const ctx=new cordis.Context();await ctx.plugin(loader.default);ctx.loader.builtins.group=group.default;
  for(const service of [llm.default,sessions.default,projection.default,prompt.default,tools.default])await ctx.plugin(service);
  await ctx.plugin(executor);await ctx.plugin(agents.default);await ctx.plugin(loop.default,{agents:[]});await ctx.plugin(presets.default,{default:'standard'});
  for(const id of ['standard','control-model','control-controller','control-lightweight'])await ctx.agentPresets.register({id,plugins:[]});
  await ctx.plugin(storage.default);const backend=new jsonStorage.JsonStorageBackend(join(directory,'json'));ctx.storage.backend.register('json',backend);
  const facility=new domain.DomainFacility(ctx,{backend:'json',routes:{}});ctx.storage.mount('domain',facility);ctx.provide('storageDomain',facility);
  ctx.provide('sandboxPolicy',{resolve:()=>({mode:'workspace-write',workspaceRoot:cwd})});
  const handle=await ctx.agents.create({sessionId:'unit-professional-role',meta:{cwd:legacySlash?cwd.replaceAll('\\','/'):cwd,agentPreset:original},setup:async scoped=>{await ctx.agentPresets.mount(scoped,original);}});
  const peer=second?(await ctx.agents.create({sessionId:'unit-second-professional-role',meta:{cwd,agentPreset:'control-controller'},setup:async scoped=>{await ctx.agentPresets.mount(scoped,'control-controller');}})).agent:undefined;
  // Create through the native in-memory Agent factory before adding the Workspace's cold-list peer.
  ctx.provide('sessionPersistence',{list:async()=>[]});await ctx.plugin(workspaces.default);
  // Deliberately start the real cache after creation: it reproduces an absent
  // initial cold-list hint without forging or mutating any native cache row.
  await ctx.plugin(projectionCache.default,{writeEveryEvents:10000,writeIntervalMs:60000});
  return {ctx,cwd,directory,agent:handle.agent,peer,close:async()=>{await ctx.fiber.dispose();await backend.close();}};
}

// Execute the unchanged upstream navigation/Session controller implementations.
// Transpilation is in memory; local TS imports and native dependencies retain
// their package resolution. No copied restoreBlank or preset-selection logic.
const ts=nativeRequire('typescript'),nativeModules=new Map();
function nativeSource(file){
  if(nativeModules.has(file))return nativeModules.get(file).exports;
  const ownRequire=createRequire(file),loaded={exports:{}};nativeModules.set(file,loaded);
  const source=ownRequire('node:fs').readFileSync(file,'utf8');
  const output=ts.transpileModule(source,{fileName:file,compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,experimentalDecorators:true}}).outputText;
  const localRequire=name=>name==='@deepseek-ai/dsh-api-gateway/client'?nativeSource(resolve(upstream,'packages/api/gateway/src/client/index.ts')):name.startsWith('.')&&name.endsWith('.ts')?nativeSource(resolve(dirname(file),name)):ownRequire(name);
  vm.runInThisContext(`(function(require,module,exports){${output}\n})`,{filename:file})(localRequire,loaded,loaded.exports);
  return loaded.exports;
}
const {UiWorkspaceService}=nativeSource(resolve(upstream,'packages/client/ui-workspace/src/client/navigation.ts'));
const {ApiSessionAgentController}=nativeSource(resolve(upstream,'packages/api/session-controller/src/agent.ts'));
const {SessionCommandController}=nativeSource(resolve(upstream,'packages/api/session-controller/src/commands.ts'));
const {ClientSessions}=nativeSource(resolve(upstream,'packages/api/session-controller/src/client/sessions/service.ts'));
const {ApiSessionList}=nativeSource(resolve(upstream,'packages/api/session-controller/src/list.ts'));

test('actual native ClientSessions preserves a selected canonical professional blank across startup with another blank in the same Workspace',async()=>{
  const f=await fixture('control-model',false,true),clientCtx=new cordis.Context(),originalStorage=globalThis.localStorage;
  try{
    await module.ensureSessionWorkspace(f.ctx,f.agent);await module.ensureSessionWorkspace(f.ctx,f.peer);
    const workspace=f.ctx.workspaceRegistry.list()[0];assert.equal(f.agent.session.header.cwd,workspace.path);assert.equal(f.peer.session.header.cwd,workspace.path);
    f.ctx.provide('attachments',{imageLimits:{}});f.ctx.provide('sessionQuery',{listSessions:async()=>f.ctx.agents.list().map(agent=>({header:agent.session.header,formatStatus:'current'}))});
    const list=new ApiSessionList(f.ctx,16);await list.initialize();
    const server=new SessionCommandController(f.ctx,new ApiSessionAgentController(f.ctx),f.cwd),requests=[];
    const keys=new Set(['agentPreset','sessionListMetadata']);
    const remote={commands:{execute:async()=>{throw Error('No domain jobs or messages permitted');}},subagents:{},session:{
      list:async()=>({ok:true,value:{items:await list.list()}}),
      projections:async({sessionId})=>({ok:true,value:{kind:'sequenced',...f.ctx.sessionProjections.snapshot(f.ctx.sessions.get(sessionId),keys)}}),
      create:async request=>{requests.push(request);return{ok:true,value:await server.create(request)};},
      async *follow(request,signal){const actual=f.ctx.sessions.get(request.address.sessionId);yield{type:'snapshot',header:actual.header,cursor:actual.seq-1,records:[],hasMore:false,projections:f.ctx.sessionProjections.snapshot(actual,keys),assistantStream:{revision:0}};await new Promise(done=>{if(signal.aborted)done();else signal.addEventListener('abort',done,{once:true});});}
    }};
    const client=new ClientSessions(clientCtx,remote),backing=new Map([['dsh.sessions.current',JSON.stringify({sessionId:f.peer.id})]]);
    globalThis.localStorage={getItem:key=>backing.get(key)??null,setItem:(key,value)=>backing.set(key,value),removeItem:key=>backing.delete(key)};
    clientCtx.provide('layout',{beginNavigation:()=>new AbortController().signal,selectPanel(){}});
    const state={phase:'ready',items:[{workspaceId:workspace.id,path:workspace.path,sessionIds:[...workspace.sessionIds],createdAt:new Date(0).toISOString()}],archivedSessionIds:[]};
    new UiWorkspaceService(clientCtx,{}, {list:{getSnapshot:()=>state,subscribe:()=>()=>{}}},client,{},()=>{});
    await client.refresh();const deadline=Date.now()+3000;while(requests.length===0&&Date.now()<deadline)await new Promise(done=>setTimeout(done,5));
    assert.deepEqual(requests,[{workspaceId:workspace.id,sessionId:f.peer.id,agentPreset:'control-controller'}]);
    assert.equal(JSON.parse(backing.get('dsh.sessions.current')).sessionId,f.peer.id);
    assert.equal(client.list.getSnapshot().byId[f.peer.id].projectionValues.agentPreset,'control-controller');
    assert.equal(f.ctx.agentPresets.presetForSession(f.peer.session),'control-controller');assert.equal(f.ctx.agentPresets.presetForSession(f.agent.session),'control-model');
    for(const agent of [f.agent,f.peer])assert.ok(!agent.session.snapshotEvents().some(event=>['turn/start','user/message','agent-preset/selected'].includes(event.type)));
    assert.equal(f.ctx.agents.list().length,2);assert.equal(f.ctx.get('jobs'),undefined);
  }finally{globalThis.localStorage=originalStorage;await clientCtx.fiber.dispose();await f.close();}
});

test('fresh header-only professional Session seals a durable cold preset hint before actual native blank restoration',async()=>{
  const f=await fixture();const originalStorage=globalThis.localStorage;
  try{
    const keys=new Set(['agentPreset']),id=f.agent.id,session=f.agent.session;
    assert.equal(f.ctx.sessionProjectionCache.cachedSnapshot(session.header,keys),undefined);
    assert.equal(f.ctx.sessionProjections.snapshot(session,keys).values.agentPreset,'control-model');
    assert.equal(session.snapshotEvents().filter(e=>e.type==='agent-preset/selected').length,0,'A correct header has no selection event; seq cannot prove otherwise');
    let flushes=0;f.ctx.on('session/flush',()=>{flushes++;});
    await module.ensureSessionWorkspace(f.ctx,f.agent);
    const cached=f.ctx.sessionProjectionCache.cachedSnapshot(session.header,keys);
    assert.equal(cached?.values.agentPreset,'control-model','The real cold listing must receive the expected hint before navigation');
    assert.ok(flushes>0,'Cache seal passes through native SessionStore.flush');
    const stored=JSON.parse(await readFile(join(f.directory,'json','session_projcache','sessions',id+'.json'),'utf8'));
    assert.equal(stored.record.rows.agentPreset.val,'control-model','Cold hint is on the actual JSON domain medium');
    const workspace=f.ctx.workspaceRegistry.list().find(w=>w.sessionIds.includes(id));assert.ok(workspace);
    const commands=new SessionCommandController(f.ctx,new ApiSessionAgentController(f.ctx),f.cwd),requests=[],retained=[];
    // Same public transport boundary as the UI: list is initially loading,
    // then exposes the actual cold cache view, not a test-supplied role string.
    let state={phase:'loading',ids:[id],byId:{[id]:{id,cwd:f.cwd,blank:true,updatedAt:0,projectionValues:cached.values}}};
    const clientSessions={list:{getSnapshot:()=>state,subscribe:()=>()=>{}},refreshProjections:async()=>{},
      create:async request=>{requests.push(request);return(await commands.create(request)).sessionId;},
      retain:target=>{retained.push(target);return{sessionId:target,release(){},ready:Promise.resolve()};},subagentAddress:()=>undefined};
    const nativeWorkspaces={phase:'ready',items:[{workspaceId:workspace.id,path:workspace.path,sessionIds:[...workspace.sessionIds],createdAt:new Date(0).toISOString()}],archivedSessionIds:[]};
    const backing=new Map([['dsh.sessions.current',JSON.stringify({sessionId:id})]]);
    globalThis.localStorage={getItem:key=>backing.get(key)??null,setItem:(key,value)=>backing.set(key,value),removeItem:key=>backing.delete(key)};
    f.ctx.provide('layout',{beginNavigation:()=>new AbortController().signal,selectPanel(){}});
    const navigation=new UiWorkspaceService(f.ctx,{}, {list:{getSnapshot:()=>nativeWorkspaces,subscribe:()=>()=>{}}},clientSessions,{},()=>{});
    state={...state,phase:'ready'};
    await navigation.restoreSelection(nativeWorkspaces,state);
    assert.deepEqual(requests,[{workspaceId:workspace.id,sessionId:id,agentPreset:'control-model'}]);
    assert.deepEqual(retained,[id]);assert.equal(f.ctx.agents.list().length,1);assert.equal(f.ctx.agents.get(id),f.agent);
    assert.equal(f.ctx.agentPresets.presetForSession(session),'control-model');assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'control-model');
    assert.ok(!session.snapshotEvents().some(e=>['turn/start','user/message','agent-preset/selected'].includes(e.type)),'Flush does not invent messages, turns or selection events');
    assert.equal(f.ctx.get('jobs'),undefined,'No job service or numerical admission exists in this fixture');
    await module.ensureSessionWorkspace(f.ctx,f.agent);assert.equal(f.ctx.sessionProjectionCache.cachedSnapshot(session.header,keys).values.agentPreset,'control-model');
  }finally{globalThis.localStorage=originalStorage;await f.close();}
});

test('a real blank professional Session is repaired through official select, preserving its identity/jobs and producing replayable native selection',async()=>{
  const f=await fixture();try{
    await f.ctx.agentPresets.select(f.agent,'standard');assert.equal(f.ctx.agentPresets.presetForSession(f.agent.session),'standard');assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'standard');
    const writer=await new JobHistory().create({workspace:f.cwd,owner:f.agent.id,agentName:'model',request:{operation:'inspect_workspace',workspace:f.cwd}});await writer.flush();
    const file=join(f.cwd,'.control-jobs',writer.record.attemptId,'request.json'),before=hash(await readFile(file)),prefix=f.agent.session.snapshotEvents(),id=f.agent.id;
    await module.ensureSessionWorkspace(f.ctx,f.agent);
    assert.equal(f.agent.id,id);assert.equal(f.agent.session.header.agentPreset,'control-model');assert.equal(f.ctx.agentPresets.presetForSession(f.agent.session),'control-model');assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'control-model');
    const events=f.agent.session.snapshotEvents();assert.deepEqual(events.slice(0,prefix.length),prefix);assert.equal(events.length,prefix.length+1);assert.equal(events.at(-1).type,'agent-preset/selected');assert.equal(events.at(-1).data.agentPreset,'control-model');assert.ok(!events.some(e=>['turn/start','user/message'].includes(e.type)));
    assert.equal(hash(await readFile(file)),before);assert.equal(f.ctx.agents.list().length,1);
    const replayFile=join(f.directory,'official-selected-events.json');await writeFile(replayFile,JSON.stringify(events));const restored=sessions.Session.create(id,JSON.parse(await readFile(replayFile,'utf8')),f.agent.session.header);
    assert.equal(f.ctx.agentPresets.presetForSession(restored),'control-model');
  }finally{await f.close();}
});

test('an already correct professional Session with a real completed turn is a no-op, not an invalid locked selection',async()=>{
  const f=await fixture();try{f.agent.session.append('turn/start',{turn:1});f.agent.session.append('turn/end',{turn:1,reason:{kind:'completed'}});const before=JSON.stringify(f.agent.session.snapshotEvents());await module.ensureSessionWorkspace(f.ctx,f.agent);assert.equal(JSON.stringify(f.agent.session.snapshotEvents()),before);assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'control-model');}finally{await f.close();}
});

test('a changed professional Session that has started is refused without replacing its identity or appending repair events',async()=>{
  const f=await fixture();try{await f.ctx.agentPresets.select(f.agent,'standard');f.agent.session.append('turn/start',{turn:1});f.agent.session.append('turn/end',{turn:1,reason:{kind:'completed'}});const before=JSON.stringify(f.agent.session.snapshotEvents());await assert.rejects(module.ensureSessionWorkspace(f.ctx,f.agent),/CONTROL_PROFESSIONAL_SESSION_LOCKED/);assert.equal(JSON.stringify(f.agent.session.snapshotEvents()),before);assert.equal(f.ctx.agents.list().length,1);assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'standard');assert.equal(f.agent.id,'unit-professional-role');}finally{await f.close();}
});

test('a real legacy slash-spelled professional header refuses Open and admission without preset selection, Workspace attachment or jobs',async()=>{
  const f=await fixture('control-controller',true);try{
    assert.notEqual(f.agent.session.header.cwd,f.cwd);const events=JSON.stringify(f.agent.session.snapshotEvents()),id=f.agent.id;
    await assert.rejects(module.ensureProfessionalPreset(f.ctx,f.agent),/CONTROL_PROFESSIONAL_SESSION_CANONICAL_WORKSPACE_REQUIRED/);
    await assert.rejects(module.ensureSessionWorkspace(f.ctx,f.agent),/CONTROL_PROFESSIONAL_SESSION_CANONICAL_WORKSPACE_REQUIRED/);
    assert.equal(f.agent.id,id);assert.equal(JSON.stringify(f.agent.session.snapshotEvents()),events);
    assert.equal(f.ctx.agentPresets.presetForSession(f.agent.session),'control-controller');assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'control-controller');
    assert.equal(f.ctx.workspaceRegistry.list().length,0);assert.equal(f.ctx.agents.list().length,1);assert.equal(f.ctx.get('jobs'),undefined);
    await assert.rejects(stat(join(f.cwd,'.control-jobs')),error=>error.code==='ENOENT');
    assert.equal(f.ctx.sessionProjectionCache.cachedSnapshot(f.agent.session.header,new Set(['agentPreset'])),undefined,'Refusal occurs before writing a fresh role hint');
  }finally{await f.close();}
});

test('generic standard Sessions remain generic and may attach without professional role or canonical-spelling coercion',async()=>{
  const f=await fixture('standard',true);try{f.agent.session.append('turn/start',{turn:1});f.agent.session.append('turn/end',{turn:1,reason:{kind:'completed'}});const before=JSON.stringify(f.agent.session.snapshotEvents());await module.ensureSessionWorkspace(f.ctx,f.agent);assert.equal(JSON.stringify(f.agent.session.snapshotEvents()),before);assert.equal(f.ctx.agentPresets.composedPreset(f.agent.ctx),'standard');assert.notEqual(f.agent.session.header.cwd,f.cwd);}finally{await f.close();}
});
