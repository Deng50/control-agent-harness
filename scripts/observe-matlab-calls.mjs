/** Optional read-only execution observer for actual native MCP acceptance. */
import assert from 'node:assert/strict';
import {appendFile,mkdir,readFile,realpath,writeFile} from 'node:fs/promises';
import {dirname,isAbsolute,relative,resolve,sep} from 'node:path';
export const name='control-matlab-call-acceptance-observer';
export const inject=['tools','workingDirectory'];
const project=resolve(import.meta.dirname,'..');
const inside=(root,path)=>{const name=relative(root,path);return name===''||!isAbsolute(name)&&name!=='..'&&!name.startsWith(`..${sep}`);};
export async function apply(ctx,config){
  assert.ok(config&&typeof config.directory==='string'&&isAbsolute(config.directory));
  const base=await realpath(resolve(project,'test-output')),parent=await realpath(dirname(config.directory));
  assert.ok(inside(base,parent),'Acceptance observations must remain inside test-output');
  try{await mkdir(config.directory);}catch(error){if(error.code!=='EEXIST')throw error;}
  const directory=await realpath(config.directory);assert.ok(inside(base,directory));
  const file=resolve(directory,'events.jsonl');
  await writeFile(resolve(directory,'observer-ready.json'),JSON.stringify({schema_version:1,status:'ready',
    observer:name,host_pid:process.pid,directory,at:new Date().toISOString()})+'\n','utf8');
  ctx.on('tools/execute',async(exec,next)=>{
    if(!exec.name.startsWith('mcp__matlab__')||!exec.agent?.session)return next();
    let workspace;
    try{
      workspace=await realpath(ctx.workingDirectory.get(exec.agent.session));
      if(!inside(base,workspace))return next();
      const fixture=JSON.parse(await readFile(resolve(workspace,'.case-plans/test-fixture.json'),'utf8'));
      if(fixture.kind!=='control-case-plan-test-fixture-v1'||resolve(fixture.workspace)!==workspace)return next();
    }catch{return next();}
    const identity={source:'trusted_test_observer',name:exec.name,callId:exec.callId,rootCallId:exec.rootCallId??null,
      sessionId:exec.agent.id,workspace};
    const record=async(event,fields={})=>{
      try{await appendFile(file,JSON.stringify({...identity,event,at:new Date().toISOString(),...fields})+'\n','utf8');}
      catch{ctx.logger.warn('MATLAB acceptance observer could not write an audit row');}
    };
    await record('started');
    try{const result=await next();await record('returned',{isError:result.isError===true});return result;}
    catch(error){await record('threw',{errorName:error?.name??'Error'});throw error;}
  });
}
