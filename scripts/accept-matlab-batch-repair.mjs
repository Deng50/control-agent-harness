/** Exercise the production grant, confined workers, native dispatcher and durable job ledger without an LLM. */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdtemp,readFile,readdir,writeFile} from 'node:fs/promises';
import {dirname,join,resolve} from 'node:path';
import {openLayer,project,readFixture} from './matlab-layer-harness.mjs';

const {fixture,marker,resource}=await readFixture(resolve(process.argv[2]));
const workspace=await mkdtemp(join(fixture,'workspace-'));
const json=file=>readFile(file,'utf8').then(JSON.parse);
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const ledger=join(workspace,'.case-plans','matlab-exchanges');
const evidence={confirmation_source:'test_fixture',workspace,pid:marker.matlab_pid,uuid:marker.session_id,jobs:[]};
const report=join(workspace,'batch-repair-evidence.json');
const persist=()=>writeFile(report,JSON.stringify(evidence,null,2));
execFileSync('powershell.exe',['-NoProfile','-File',join(project,'scripts/prepare-workspace.ps1'),'-Workspace',workspace,'-Prepare'],{windowsHide:true});
process.env.CONTROL_MATLAB_MCP_EXECUTABLE=marker.mcp_executable;
assert.ok(Number.isSafeInteger(marker.matlab_pid) && marker.matlab_pid>0);
const boundExecutable=execFileSync('powershell.exe',['-NoProfile','-Command',
  `(Get-Process -Id ${marker.matlab_pid} -ErrorAction Stop).Path`],
  {windowsHide:true,encoding:'utf8'}).trim();
const launch=await json(join(dirname(marker.private_appdata),'launch.json'));
assert.equal(launch.session_id,marker.session_id);
assert.ok(boundExecutable.toLowerCase()===launch.matlab_executable.toLowerCase() ||
  boundExecutable.toLowerCase()===join(dirname(launch.matlab_executable),'win64','MATLAB.exe').toLowerCase());
process.env.CONTROL_MATLAB_BOUND_EXECUTABLE=boundExecutable;
const layer=await openLayer({fixture,workspace,managed:true});
const {ctx}=layer;
// The native job harness has no UI preset registry. Invoke the ordinary session's shared commands.
const agent=layer.createAgent('standard');
ctx.jobs.attachController('synthetic-live-batch');
Object.assign(agent,{status:'idle',model:{},inbox:{},send(){},followup(){},inject(){},cancel(){},whenIdle:async()=>{},
  runMaintenance:fn=>fn(new AbortController().signal)});
const unregister=await ctx.agents.register(agent);
async function command(name,args) {
  const {result}=await ctx.commands.execute(agent,'/'+name+' '+JSON.stringify({workspace,...args}),[],new AbortController().signal);
  assert.equal(result.kind,'success',name+': '+result.text);
  return JSON.parse(result.text);
}
async function run(stage,operation,inputs,config) {
  const args={agent:'controller',operation,inputs,config};
  const card=await command('control-prepare-review',args);
  if(card.card_id) {
    const markerFile=join(workspace,'.case-plans','test-fixture.json');
    try { await writeFile(markerFile,JSON.stringify({kind:'control-case-plan-test-fixture-v1',workspace}),{flag:'wx'}); }
    catch (error) { if(error.code!=='EEXIST') throw error; }
    const confirmation=await command('control-confirm',{card_id:card.card_id});
    assert.equal(confirmation.confirmation_source,'test_fixture');
    args.card_id=card.card_id;
  }
  if (operation==='check_matlab_design') assert.ok(card.card_id && card.matlab_scope?.dedicated_session.pid===marker.matlab_pid);
  const admitted=await command('control',args);
  const record={stage,operation,card_id:card.card_id,...admitted}; evidence.jobs.push(record); await persist();
  record.final=await ctx.jobs.wait(admitted.jobId,180000,agent.id);
  record.history=(await command('control-status',{attemptId:admitted.attemptId})).record;
  record.resource=await json(join(resource,'resource.json')); await persist();
  assert.equal(record.resource.quarantine,null);
  assert.equal(record.resource.session.pid,marker.matlab_pid);
  return record;
}
try {
  const spec=await json(join(project,'examples/v2/mpc-design.json'));
  const draft=await run('design','synthesize_mpc',{description:spec.description},spec.draft_config);
  assert.equal(draft.final.status,'completed',draft.final.detail);
  const inputs={design:draft.history.result.artifact_id};
  const before=new Set(await readdir(ledger).catch(e=>e.code==='ENOENT'?[]:Promise.reject(e)));
  for(const valid of [false,true]) {
    const parameters={...spec.parameters,mass:valid ? spec.parameters.mass : -1};
    const r=await run(valid?'correct physical mass':'invalid physical mass','check_matlab_design',inputs,{case:spec.solver_check,parameters});
    const changes=(await readdir(ledger)).filter(n=>!before.has(n));assert.equal(changes.length,1,'One new exchange required per paid-for scope');
    const nonce=changes[0]; before.add(nonce); const dir=join(ledger,nonce),manifestPath=join(dir,'manifest.json');
    const manifest=await json(manifestPath);
    assert.equal(manifest.nonce,nonce); assert.ok(/^[a-f0-9]{64}$/.test(manifest.request_sha256));
    r.exchange={nonce,manifest_path:manifestPath,manifest_sha256:digest(await readFile(manifestPath)),request_sha256:manifest.request_sha256,plan_hash:manifest.plan.plan_hash};
    for(const member of manifest.members) assert.equal(digest(await readFile(join(dir,'bundle',member.name))),member.sha256);
    const outcome=await readFile(join(dir,'result.json')).catch(e=>{if(e.code!=='ENOENT') throw e;});
    if(valid) {
      assert.equal(r.final.status,'completed',r.final.detail);
      const value=JSON.parse(outcome);
      assert.equal(value.nonce,nonce); assert.equal(value.spec_hash,manifest.spec_hash);
      assert.equal(value.request_sha256,manifest.request_sha256);assert.equal(value.plan_hash,manifest.plan.plan_hash);
      assert.equal(value.report.matlab_pid,marker.matlab_pid);assert.equal(value.report.wrapper_nonce,nonce);
      assert.equal(value.report.status,'single_point_solved');assert.equal(value.report.solve.solver,'quadprog');
      assert.ok(value.report.solve.exitflag>0 && Number.isFinite(value.report.action));
      const artifact=r.history.result.artifact_path;
      const metadata=await json(join(artifact,'artifact.json'));
      const archived=await readFile(join(artifact,'report.json'));
      assert.equal(digest(archived),metadata.files['report.json']);
      assert.deepEqual(JSON.parse(archived),{...value.report,design:inputs.design,case:spec.solver_check});
      r.exchange.result_sha256=digest(outcome);
      r.exchange.archived_report_sha256=digest(archived);
      r.numerical={action:value.report.action,solve:value.report.solve};
    } else {
      assert.equal(r.final.status,'failed'); assert.match(r.final.detail,/Invalid SI physical parameters/);
      assert.equal(outcome,undefined,'Failed helper must not produce a successful design result');
      assert.equal(r.history.attemptId,r.attemptId);assert.equal(r.history.state,'failed');
      assert.equal(r.history.outcome.cleanupVerified,true);
      const attemptDir=join(workspace,'.control-jobs',r.attemptId);
      const entries=(await readdir(attemptDir)).filter(name=>/^\d+\.json$/.test(name)).sort();
      r.exchange.failure_record=join(attemptDir,entries.at(-1));
      assert.equal((await json(r.exchange.failure_record)).state,'failed');
    }
    await persist();
  }
  evidence.passed=true;
} catch(error){evidence.error=error.message;throw error;}
finally {await persist();console.log(report);await unregister();await layer.close();}
