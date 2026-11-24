/** One already-launched disposable R2024b instance per fault. No model API. */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { basename, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { openLayer } from './matlab-layer-harness.mjs';
import { matlabString } from '../harness-plugin/src/simulink-native.mjs';

const [directory,phase] = process.argv.slice(2);
const fixture = resolve(directory);
assert.ok(['busy','cancel','timeout','file-timeout'].includes(phase));
assert.ok(basename(fixture).startsWith('host-fault-'+phase+'-'));
const workspace=join(fixture,'workspace'), markerPath=join(workspace,'started.txt');
const timed=phase.endsWith('timeout');
const nonce=randomUUID(), timeoutMs=timed?20000:120000;
const layer=await openLayer({fixture,workspace,managed:true,toolCallTimeoutMs:timeoutMs});
const evidence={phase,matlab_pid:layer.marker.matlab_pid,release:layer.marker.matlab_release,
  dedicated_session:layer.marker.session_id,sdk_timeout_ms:timeoutMs,workspace};
try {
  const ready=await layer.call('mcp__matlab__evaluate_matlab_code',{code:'disp(3+4);'});
  assert.equal(ready.isError,false,ready.error?.message);
  const controller=new AbortController();
  const timeout=AbortSignal.timeout(140000);
  const body=phase==='busy' ? "start(timer('Name','ControlHarnessFaultTimer','StartDelay',8,'TimerFcn',@(~,~)disp(42)));" : timed ? 'pause(31);' : 'pause(8);';
  const code="f=fopen("+matlabString(markerPath)+",'a'); assert(f~=-1); fprintf(f,'%s',"+matlabString(nonce)+"); fclose(f); "+body;
  const fixtureScript=join(workspace,'run_scoped_write.m');
  if (phase==='file-timeout') await writeFile(fixtureScript,code,{flag:'wx'});
  const call=phase==='file-timeout'
    ? {name:'mcp__matlab__run_matlab_file',arguments:JSON.stringify({script_path:fixtureScript})}
    : {name:'mcp__matlab__evaluate_matlab_code',arguments:JSON.stringify({code,project_path:workspace})};
  const request=layer.ctx.tools.execute({agent:layer.agent,signal:AbortSignal.any([controller.signal,timeout]),
    calls:[{type:'tool_call',id:randomUUID(),...call}]});
  const pending=request.then(batch=>({result:batch.results[0]?.result}),error=>({error:error.message}));
  const end=Date.now()+35000;
  while (Date.now()<end) {
    let marker;
    try { marker=await readFile(markerPath,'utf8'); } catch (e) {if (e.code!=='ENOENT') throw e;}
    if (marker===nonce) {evidence.body_started=true; break;}
    await sleep(50);
  }
  assert.ok(evidence.body_started,'MATLAB did not write the scoped execution marker');
  // A second role enters the same resource queue while the first owns it.
  evidence.peer_role=phase==='file-timeout' ? 'control-lightweight' : 'control-controller';
  const peer = layer.createAgent(evidence.peer_role);
  const secondary = layer.call('mcp__matlab__evaluate_matlab_code',
    {code:"fprintf('SECONDARY_PID=%d',feature('getpid'));"},peer);
  const queuedDeadline=Date.now()+6000;
  while (Date.now()<queuedDeadline) {
    const snapshot=JSON.parse(await readFile(join(layer.resource,'resource.json'),'utf8'));
    if (snapshot.queue.length && (snapshot.owner || snapshot.quarantine)) {
      evidence.waiting_peer_ticket=snapshot.queue[0].ticket;
      break;
    }
    await sleep(30);
  }
  assert.ok(Number.isSafeInteger(evidence.waiting_peer_ticket),'Second role must queue behind the original call');
  if (phase==='cancel') {
    controller.abort(new Error('Dedicated test cancellation after MATLAB wrote the started marker'));
    evidence.cancel_requested_after_marker=controller.signal.aborted;
  }
  const lookup=await layer.call('control_help',{operation:'synthesize_mpc'},peer);
  assert.equal(lookup.isError,false,lookup.error?.message);
  evidence.local_diagnostic_available=true;
  const {result,error}=await pending;
  evidence.native_error=result?.error ?? {message:error};
  evidence.native_content=result?.content;
  assert.ok(result?.isError || error,'Fault must interrupt verified completion');
  const state=JSON.parse(await readFile(join(layer.resource,'resource.json'),'utf8'));
  evidence.quarantine=state.quarantine;
  evidence.during_recovery = {owner:state.owner?.label ?? null, queued:state.queue.length};
  if (phase==='busy') assert.match(evidence.native_error.message,/运行中计时器 [1-9]/);
  if (timed) assert.match(evidence.native_error.message,/timed out|timeout/i);
  if (phase==='cancel') {
    assert.ok(evidence.cancel_requested_after_marker);
    assert.ok(evidence.native_error.info?.code==='ABORTED' || evidence.native_error.message.includes(controller.signal.reason.message));
  }
  const next=await secondary;
  assert.equal(next.isError,false,next.error?.message);
  assert.ok(JSON.stringify(next.value).includes('SECONDARY_PID='+evidence.matlab_pid));
  const settled=JSON.parse(await readFile(join(layer.resource,'resource.json'),'utf8'));
  assert.equal(settled.quarantine,null);
  assert.equal(settled.session.session_id,evidence.dedicated_session);
  assert.deepEqual(settled.queue,[]);
  assert.equal(await readFile(markerPath,'utf8'),nonce,'Original write must execute exactly once');
  evidence.peer_session_id=peer.id;
  evidence.peer_matlab_pid=settled.session.pid;
  evidence.peer_matlab_session=settled.session.session_id;
  evidence.last_recovery=settled.last_recovery;
  if (phase==='file-timeout') {
    const archive=JSON.parse(await readFile(settled.last_recovery,'utf8'));
    assert.equal(archive.evidence.completion_source,'mcp_response');
    assert.equal(archive.evidence.tool_name,call.name);
    assert.equal(archive.evidence.completion_nonce,archive.quarantine.owner.completion_nonce);
    evidence.completion_source=archive.evidence.completion_source;
  }
  evidence.original_write_count=1;
  evidence.passed=true;
} catch (error) {
  evidence.passed=false; evidence.error=error.message; throw error;
} finally {
  await writeFile(join(fixture,'fault.json'),JSON.stringify(evidence,null,2));
  await layer.close();
}
