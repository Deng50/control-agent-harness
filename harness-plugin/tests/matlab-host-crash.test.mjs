import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn, execFile } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { promisify } from 'node:util';
import { Context } from '../../vendor/deepseek-harness/vendor/cordis/lib/index.js';
import Local from '../../vendor/deepseek-harness/packages/subprocess/subprocess-local/lib/index.js';
import { MatlabResource } from '../src/matlab-resource.mjs';
import { installMatlabRecovery } from '../src/matlab-recovery.mjs';
import { createMatlabTerminalVerifier } from '../src/matlab-terminal.mjs';

const project = resolve(import.meta.dirname, '../..');
const python = join(project, '.venv', 'Scripts', 'python.exe');
const env = {...process.env, PYTHONPATH:join(project,'src'), PYTHONDONTWRITEBYTECODE:'1'};
const originalHost = `
import json, os, subprocess, sys
from pathlib import Path
directory, nonce = sys.argv[1:]
broker = subprocess.Popen([sys._base_executable, '-u', '-m', 'control_harness.matlab_resource',
    'broker', '--directory', directory], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def send(command):
    broker.stdin.write(json.dumps(command) + '\\n')
    broker.stdin.flush()
    return json.loads(broker.stdout.readline())
try:
    assert send({'command':'acquire','label':'mcp__matlab__evaluate_matlab_code:original'})['event']=='queued'
    held = json.loads(broker.stdout.readline())
    assert held['event']=='acquired'
    assert send({'command':'start','require_live_session':True,'completion_nonce':nonce})['event']=='started'
    Path(directory,'fixture-request').write_text(nonce)
    print(json.dumps({'token':held['token'],'broker':broker.pid}), flush=True)
    sys.stdin.readline()
finally:
    broker.kill()
    broker.wait(timeout=8)
os._exit(73)
`;

test('replacement Host reconciles an abruptly lost Host and broker, then admits a queued write exactly once',
  {skip:process.platform!=='win32', timeout:30000}, async t => {
    const output = join(project,'test-output','matlab-resource-tests');
    await mkdir(output,{recursive:true});
    const directory = await mkdtemp(join(output,'host-crash-'));
    const state = () => readFile(join(directory,'resource.json'),'utf8').then(JSON.parse);
    const session = randomUUID(), nonce = randomUUID();
    const opts = {cwd:project,env,windowsHide:true,stdio:['pipe','pipe','inherit']};
    const actor = (source,...args) => spawn(python,['-u','-c',source,...args],opts);
    // Stable process stands in for the native MATLAB server. Its completion
    // token comes from the dispatched request after we allow it to finish.
    const matlab = actor(`from pathlib import Path
import sys
root=Path(sys.argv[1])
sys.stdin.readline()
(root/'fixture-completion').write_text((root/'fixture-request').read_text())
print('returned',flush=True)
sys.stdin.readline()`, directory);
    const matlabExit = once(matlab,'exit');
    const host = new Context();
    let old, recoveryStop;
    t.after(async () => {
      old?.stdin.end();
      await recoveryStop?.();
      await host.fiber.dispose();
      matlab.stdin.end();
      await matlabExit;
      await rm(directory,{recursive:true,force:true});
    });
    const bound = await promisify(execFile)(python,['-m','control_harness.matlab_resource','bind-session',
      '--directory',directory,'--pid',String(matlab.pid),'--session-id',session],opts);
    const binding = JSON.parse(bound.stdout);
    old = actor(originalHost,directory,nonce);
    const oldExit = once(old,'exit');
    const lines = createInterface({input:old.stdout});
    const [line] = await once(lines,'line');
    const held = JSON.parse(line);
    old.stdin.end('fault injection\n');
    assert.equal((await oldExit)[0],73);
    const stale = await state();
    assert.equal(stale.owner.pid,held.broker);
    assert.equal(stale.owner.phase,'running');
    assert.equal(stale.quarantine,null);

    await host.plugin(Local);
    const resource = new MatlabResource({subprocess:host.subprocess,codeRoot:project,pythonPath:python,
      config:{directory,pollIntervalMs:10}});
    const marker = join(directory,'identity.json');
    await writeFile(marker,JSON.stringify({schema_version:1,kind:'control-harness-dedicated-matlab-v1',
      status:'ready',session_id:session,matlab_pid:matlab.pid,project_root:project}));
    // Only the native tool is substituted. The fresh Host uses the production
    // recovery scheduler, terminal parser, Python broker and OS execution lock.
    const tools = {get:()=>true,execute:async ({calls:[call]})=>{
      const probeNonce = call.id.slice('matlab-probe-'.length);
      const observed = await readFile(join(directory,'fixture-completion'),'utf8')
        .catch(error=>{if (error.code==='ENOENT') return ''; throw error;});
      const proof = {schema_version:1,nonce:probeNonce,session_id:session,matlab_pid:matlab.pid,
        helper_file:join(project,'matlab','control_harness_terminal_probe'),idle:true,busy_models:[],
        running_timers:0,parallel_futures:0,background_futures:0,completion_nonce:observed};
      const text = 'CONTROL_HARNESS_PROBE_'+probeNonce+':'+JSON.stringify(proof);
      return {results:[{result:{value:{content:[{type:'text',text}]}}}]};
    }};
    const scheduler = installMatlabRecovery({tools,on:(_name,dispose)=>{recoveryStop=dispose;}},
      resource,createMatlabTerminalVerifier({tools},{identityFile:marker,codeRoot:project}));
    assert.equal((await scheduler.pending({inspectOwner:true})).token,held.token);
    let enqueued;
    const waiting = new Promise(resolve=>{enqueued=resolve;});
    let writes = 0;
    const writer = resource.acquire('queued-write',AbortSignal.timeout(12000),enqueued).then(async lease=>{
      assert.deepEqual(lease.session,binding);
      await lease.start(true);
      writes++;
      await writeFile(join(directory,'output'),'one corrected operation',{flag:'wx'});
      await lease.complete({kind:'synchronous_adapter_return',producer:'queued-write-fixture'});
    });
    await waiting;
    await scheduler.check();
    assert.equal(writes,0);
    assert.equal((await state()).quarantine.token,held.token);
    const returned = once(createInterface({input:matlab.stdout}),'line');
    matlab.stdin.write('finish\n');
    await returned;
    await scheduler.check();
    await writer;
    const final = await state();
    assert.deepEqual(final.session,binding);
    assert.equal(final.owner,null);
    assert.equal(final.quarantine,null);
    assert.deepEqual(final.queue,[]);
    assert.equal(JSON.parse(await readFile(final.last_recovery,'utf8')).evidence.completion_nonce,nonce);
    await scheduler.check();
    assert.equal(writes,1);
  });
