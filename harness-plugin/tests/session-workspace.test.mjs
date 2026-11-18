/** Actual native SessionStore + WorkspaceRegistry + durable JSON domain; no Host/LLM/RPC. */
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';
import test from 'node:test';
import {installSessionWorkspace,ensureSessionWorkspace} from '../src/session-workspace.mjs';
import {JobHistory} from '../src/job-history.mjs';

const upstream=resolve(import.meta.dirname,'../../vendor/deepseek-harness');
const built=path=>import(pathToFileURL(resolve(upstream,path)));
const [cordis,storage,jsonStorage,domain,sessions,workspaces]=await Promise.all([
  built('vendor/cordis/lib/index.js'),built('packages/storage/storage/lib/index.js'),
  built('packages/storage/storage-json/lib/index.js'),built('packages/storage/storage-domain/lib/index.js'),
  built('packages/core/session/lib/index.js'),built('packages/workspace/workspace/lib/index.js')]);
const require=createRequire(resolve(import.meta.dirname,'../package.json'));
const bundle=await readFile(process.env.CONTROL_TEST_CLIENT_BUNDLE??resolve(import.meta.dirname,'../lib/client.js'),'utf8');
const root=resolve(import.meta.dirname,'../../test-output/session-workspace-tests');await mkdir(root,{recursive:true});
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');

async function fixture(){
  const directory=await mkdtemp(join(root,'native-')),cwd=join(directory,'project'),other=join(directory,'other');await mkdir(cwd);await mkdir(other);
  const ctx=new cordis.Context();await ctx.plugin(storage.default);
  const backend=new jsonStorage.JsonStorageBackend(join(directory,'domain-json'));ctx.storage.backend.register('json',backend);
  const facility=new domain.DomainFacility(ctx,{backend:'json',routes:{}});ctx.storage.mount('domain',facility);ctx.provide('storageDomain',facility);
  // There is no historical persistence fixture: actual live SessionStore owns immutable headers.
  ctx.provide('sessionPersistence',{list:async()=>[]});await ctx.plugin(sessions.default);await ctx.plugin(workspaces.default);
  let policy={mode:'workspace-write',workspaceRoot:cwd};ctx.provide('sandboxPolicy',{resolve:()=>policy});
  const original=ctx.sessions.create('unit-existing-role',{meta:{cwd}});original.append('turn/start',{turn:1});original.append('turn/end',{turn:1,reason:{kind:'completed'}});
  const agent={id:original.id,session:original};let command;
  ctx.provide('commands',{register:spec=>{assert.equal(spec.name,'control-workspace');command=spec;return()=>{};}});installSessionWorkspace(ctx);
  let descriptor,actions,created=0,commands=0;const navigations=[];
  vm.runInNewContext(bundle,{window:{__ModuleLoader__:{load:value=>descriptor=value}},console,structuredClone,AbortController,setTimeout,clearTimeout});
  const client={slots:{inject:(name,fn)=>{if(name==='main')fn();},register:spec=>{actions=spec.inject();}},jobs:{state:()=>{}},
    remote:{workspace:{create:async({path})=>{const w=await ctx.workspaceRegistry.create(path);return {ok:true,value:{workspace:{workspaceId:w.id,path:w.path,sessionIds:[...w.sessionIds]}}};}},
      session:{create:async request=>{assert.equal(request.cwd,undefined,'Client must use native Workspace creation, not cwd-only');const w=ctx.workspaceRegistry.get(request.workspaceId);assert.ok(w);const live=ctx.sessions.create('unit-new-role-'+(++created),{meta:{cwd:w.path}});await w.attachSession(live.id);return {ok:true,value:{sessionId:live.id,agentPreset:request.agentPreset}};},rename:async()=>({ok:true,value:{}})},
      commands:{execute:async(sessionId,raw)=>{commands++;const live=ctx.sessions.get(sessionId);assert.ok(live);assert.match(raw,/^\/control-workspace /);return {ok:true,value:{result:await command.handler({agent:{id:live.id,session:live},rawInput:raw.slice('/control-workspace '.length)})}};}}},
    uiWorkspace:{openSession:sessionId=>navigations.push(sessionId)}};
  descriptor.factory(require).apply(client);
  return {ctx,cwd,other,original,agent,command,actions,navigations,get created(){return created;},get commands(){return commands;},setPolicy:value=>policy=value,
    close:async()=>{await ctx.fiber.dispose();await backend.close();}};
}

test('actual compiled client opens the same existing native Session after idempotent validated Workspace attachment without rewriting jobs or history',async()=>{
  const f=await fixture();try{
    const writer=await new JobHistory().create({workspace:f.cwd,owner:f.agent.id,agentName:'model',request:{operation:'inspect_workspace',workspace:f.cwd,inputs:{},config:{}}});await writer.flush();
    const requestPath=join(f.cwd,'.control-jobs',writer.record.attemptId,'request.json'),before=sha(await readFile(requestPath)),events=JSON.stringify(f.original.snapshotEvents());
    assert.equal(f.ctx.workspaceRegistry.list().filter(w=>w.sessionIds.includes(f.agent.id)).length,0,'Real legacy cwd-only Session has no native Workspace owner');
    await f.actions.openSession(f.agent.id);
    const owner=f.ctx.workspaceRegistry.list().find(w=>w.sessionIds.includes(f.agent.id));assert.ok(owner,'Native chat requires an actual Workspace membership, not only a cwd or navigation selection');assert.equal(owner.path,f.cwd);
    await f.actions.openSession(f.agent.id);assert.deepEqual([...owner.sessionIds],[f.agent.id]);assert.deepEqual(f.navigations,[f.agent.id,f.agent.id]);assert.equal(f.created,0);assert.equal(f.commands,2);
    assert.equal(JSON.stringify(f.original.snapshotEvents()),events);assert.equal(sha(await readFile(requestPath)),before);assert.equal(writer.record.owner,f.agent.id);
  }finally{await f.close();}
});

test('new role creation uses the real native Workspace id before creating its Session and preserves the requested preset',async()=>{
  const f=await fixture();try{const id=await f.actions.createSession(f.cwd,'Role title','controller');assert.equal(f.created,1);const owner=f.ctx.workspaceRegistry.list().find(w=>w.sessionIds.includes(id));assert.ok(owner);assert.equal(owner.path,f.cwd);assert.equal(f.ctx.sessions.get(id).header.cwd,f.cwd);await f.actions.openSession(id);assert.deepEqual(f.navigations,[id]);}finally{await f.close();}
});

test('Workspace adapter refuses supplied Session ids, mismatched Agent identity and read-only policy before registering a Workspace',async()=>{
  const f=await fixture();try{
    const count=f.ctx.workspaceRegistry.list().length;
    for(const args of [{sessionId:'other'},{workspace:f.other},[],null]){const result=await f.command.handler({agent:f.agent,rawInput:JSON.stringify(args)});assert.equal(result.kind,'error');}
    await assert.rejects(ensureSessionWorkspace(f.ctx,{id:'other',session:f.original}),/identity/);
    f.setPolicy({mode:'read-only',workspaceRoot:f.cwd});await assert.rejects(ensureSessionWorkspace(f.ctx,f.agent),/sandbox/);
    assert.equal(f.ctx.workspaceRegistry.list().length,count);assert.equal(f.created,0);assert.equal(f.navigations.length,0);
    // The native entity itself enforces the immutable cwd even outside our adapter.
    const otherWorkspace=await f.ctx.workspaceRegistry.create(f.other);await assert.rejects(otherWorkspace.attachSession(f.agent.id),/cwd resolves/);assert.deepEqual([...otherWorkspace.sessionIds],[]);
  }finally{await f.close();}
});
