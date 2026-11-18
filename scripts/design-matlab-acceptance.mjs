/** Test-only S07 command. Native MCP permissions and the production resource hook apply. */
import assert from 'node:assert/strict';
import {createHash,randomUUID} from 'node:crypto';
import {readFile,realpath} from 'node:fs/promises';
import {isAbsolute,relative,resolve,sep} from 'node:path';

export const name='control-design-matlab-acceptance';
export const inject=['commands','tools','jobs','workingDirectory'];
const project=resolve(import.meta.dirname,'..'),tool='mcp__matlab__evaluate_matlab_code';
const inside=(root,target)=>{const name=relative(root,target);return name===''||!isAbsolute(name)&&name!=='..'&&!name.startsWith(`..${sep}`);};
const quote=value=>`'${String(value).replaceAll("'","''")}'`;
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');

async function fixtureInputs(ctx,agent,args){
  assert.ok(args&&typeof args==='object'&&!Array.isArray(args));
  assert.deepEqual(Object.keys(args).sort(),['artifact_path','case_path',...(args.parameters_path===undefined?[]:['parameters_path'])].sort());
  const workspace=await realpath(ctx.workingDirectory.get(agent.session)),testRoot=await realpath(resolve(project,'test-output'));
  assert.ok(inside(testRoot,workspace),'This optional command is confined to an isolated test-output workspace');
  const marker=JSON.parse(await readFile(resolve(workspace,'.case-plans/test-fixture.json'),'utf8'));
  assert.equal(marker.kind,'control-case-plan-test-fixture-v1');assert.equal(resolve(marker.workspace),workspace);
  const artifact=await realpath(args.artifact_path);assert.ok(inside(workspace,artifact));
  const metadataBytes=await readFile(resolve(artifact,'artifact.json')),metadata=JSON.parse(metadataBytes.toString('utf8'));
  assert.equal(metadata.kind,'controller_design');
  const manifest=JSON.parse(await readFile(resolve(workspace,'manifest.json'),'utf8')),row=manifest.artifacts[metadata.id];
  assert.ok(row);assert.equal(await realpath(resolve(workspace,row.path)),artifact);
  if(row.metadata_file_sha256)assert.equal(hash(metadataBytes),row.metadata_file_sha256);
  for(const[name,expected]of Object.entries(metadata.files)){
    const path=await realpath(resolve(artifact,name));assert.ok(inside(artifact,path));assert.equal(hash(await readFile(path)),expected);
  }
  const bundle=await realpath(resolve(artifact,'bundle'));assert.ok(inside(artifact,bundle));
  assert.ok(metadata.files['bundle/selfcheck_controller.m'],'Only a recorded generated self-check can execute');
  const casePath=await realpath(args.case_path);assert.ok(inside(workspace,casePath));const caseBytes=await readFile(casePath);JSON.parse(caseBytes.toString('utf8'));
  let parametersPath='',parametersHash=null;if(args.parameters_path!==undefined){parametersPath=await realpath(args.parameters_path);assert.ok(inside(workspace,parametersPath));const bytes=await readFile(parametersPath);JSON.parse(bytes.toString('utf8'));parametersHash=hash(bytes);}
  return{workspace,artifact,bundle,casePath,caseHash:hash(caseBytes),parametersPath,parametersHash,metadataHash:hash(metadataBytes)};
}

export function apply(ctx){
  ctx.commands.register({definitionId:'@control-engineering/acceptance/design-matlab',name:'design-matlab-check',
    description:'Test-fixture-only generated MPC MATLAB prediction/cost/one-solve check; no closed-loop or tuning',
    async handler({agent,signal,rawInput}){
      signal.throwIfAborted();const input=await fixtureInputs(ctx,agent,JSON.parse(rawInput));
      const jobId=ctx.jobs.start({owner:agent.id,kind:'control-engineering',label:'S07 generated MATLAB design check',run(job){
        const abort=new AbortController();
        const done=(async()=>{
          try{
            const nonce=randomUUID(),helperRoot=resolve(project,'matlab'),helper=resolve(helperRoot,'control_harness_design_check.m');
            const code=`addpath(${quote(helperRoot)}); assert(strcmp(which('control_harness_design_check'),${quote(helper)})); fprintf('CONTROL_S07_DESIGN_${nonce}:%s\\n',jsonencode(control_harness_design_check(${quote(input.bundle)},${quote(input.casePath)},${quote(input.parametersPath)},${quote(nonce)})));`;
            job.updateProgress('Checking generated MATLAB controller through native MCP');
            const batch=await ctx.tools.execute({agent,signal:abort.signal,calls:[{type:'tool_call',id:`design-${nonce}`,name:tool,arguments:JSON.stringify({code,project_path:input.workspace})}]});
            const result=batch.results[0]?.result;assert.ok(result&&!result.isError,'Native MATLAB design check failed');
            const prefix=`CONTROL_S07_DESIGN_${nonce}:`,lines=(result.value?.content??[]).filter(x=>x.type==='text').flatMap(x=>String(x.text).split(/\r?\n/)).filter(x=>x.startsWith(prefix));
            assert.equal(lines.length,1,'One nonce-bound MATLAB result is required');const value=JSON.parse(lines[0].slice(prefix.length));assert.equal(value.wrapper_nonce,nonce);
            assert.equal(hash(await readFile(input.casePath)),input.caseHash,'Case file changed during MATLAB execution');
            if(input.parametersPath)assert.equal(hash(await readFile(input.parametersPath)),input.parametersHash,'Physical parameters changed during MATLAB execution');
            const evidence={schema_version:1,confirmation_source:'test_fixture',scope:'generated prediction/cost/single solver check only',...input,nonce,result:value,closed_loop:'not_run'};
            job.append(JSON.stringify({event:'result',result:evidence})+'\n',{channel:'stdout'});return{status:'completed',result:JSON.stringify(evidence)};
          }catch(error){return{status:abort.signal.aborted?'killed':'failed',detail:error.stack??String(error)};}
        })();return{done,cancel(reason){abort.abort(new Error(reason??'S07 MATLAB test cancelled'));}};
      }});return{kind:'success',text:JSON.stringify({jobId})};
    }});
}
