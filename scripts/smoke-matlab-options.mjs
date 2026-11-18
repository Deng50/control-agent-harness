/** Optional confined S09 native MATLAB option-inspection command and observer. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {readFile,realpath} from 'node:fs/promises';
import {isAbsolute,relative,resolve,sep} from 'node:path';
import {pathToFileURL} from 'node:url';
import {command,connectHarness,prepareSmokeWorkspace,project,smokeOptions,writeSmokeEvidence} from './harness-rpc.mjs';
export const name='control-matlab-options-acceptance';
export const inject=['commands','tools','workingDirectory'];
const inside=(base,path)=>{const name=relative(base,path);return name!==''&&!isAbsolute(name)&&name!=='..'&&!name.startsWith(`..${sep}`);};
const quote=value=>`'${String(value).replaceAll("'","''")}'`;
export function apply(ctx){
  ctx.commands.register({definitionId:'@control-engineering/acceptance/matlab-options',name:'matlab-options-check',
    description:'Test-fixture-only read of installed quadprog properties; no optimization run',
    async handler({agent,signal,rawInput}){
      assert.equal(String(rawInput??'').trim(),'');
      const workspace=await realpath(ctx.workingDirectory.get(agent.session));
      assert.ok(inside(await realpath(resolve(project,'test-output')),workspace));
      const fixture=JSON.parse(await readFile(resolve(workspace,'.case-plans/test-fixture.json'),'utf8'));
      assert.equal(fixture.kind,'control-case-plan-test-fixture-v1');assert.equal(resolve(fixture.workspace),workspace);
      const nonce=randomUUID(),helperRoot=resolve(project,'matlab'),helper=resolve(helperRoot,'control_harness_optimization_options.m');
      const code=`addpath(${quote(helperRoot)}); assert(strcmp(which('control_harness_optimization_options'),${quote(helper)})); fprintf('CONTROL_OPTIONS_${nonce}:%s\\n',jsonencode(control_harness_optimization_options(${quote(nonce)})));`;
      const batch=await ctx.tools.execute({agent,signal,calls:[{type:'tool_call',id:`options-${nonce}`,name:'mcp__matlab__evaluate_matlab_code',arguments:JSON.stringify({code,project_path:workspace})}]});
      const result=batch.results[0]?.result;assert.ok(result&&!result.isError,'Native MATLAB option inspection failed');
      const prefix=`CONTROL_OPTIONS_${nonce}:`,lines=(result.value?.content??[]).filter(x=>x.type==='text').flatMap(x=>String(x.text).split(/\r?\n/)).filter(x=>x.startsWith(prefix));
      assert.equal(lines.length,1);const value=JSON.parse(lines[0].slice(prefix.length));assert.equal(value.nonce,nonce);
      return{kind:'success',text:JSON.stringify({status:'PASS',confirmation_source:'test_fixture',result:value})};
    }});
}
async function main(){
  const options=smokeOptions('smoke-matlab-options');if(options.help)return;
  await prepareSmokeWorkspace(options.workspace);
  await writeSmokeEvidence(resolve(options.workspace,'.case-plans/test-fixture.json'),{kind:'control-case-plan-test-fixture-v1',workspace:options.workspace,confirmation_source:'test_fixture'});
  const rpc=await connectHarness({logPath:options.webLog,connectionClose:true});
  const session=await rpc('session/create',{request:{cwd:options.workspace,agentPreset:'control-controller'}});
  const evidence=await command(rpc,session.sessionId,'/matlab-options-check');
  assert.equal(evidence.result.release,'2024b');assert.ok(Number.isSafeInteger(evidence.result.matlab_pid)&&evidence.result.matlab_pid>0);
  assert.equal(evidence.result.optimization_licensed,true);assert.equal(evidence.result.solver_executed,false);
  for(const key of ['Algorithm','Display','MaxIterations','OptimalityTolerance','ConstraintTolerance'])assert.ok(evidence.result.option_names.includes(key));
  await writeSmokeEvidence(options.evidence,{...evidence,session_id:session.sessionId,workspace:options.workspace});
  console.log(JSON.stringify({status:'PASS',evidence:options.evidence,matlab_pid:evidence.result.matlab_pid,defaults:evidence.result.selected_defaults}));
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await main();
