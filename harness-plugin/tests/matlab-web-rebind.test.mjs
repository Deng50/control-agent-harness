import test from 'node:test';
import assert from 'node:assert/strict';
import { execFile, spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { superviseWebBinding } from '../src/matlab-web-rebind.mjs';

test('supervised Web automatically rebinds after the original OS process exits without a launcher request',
  {skip:process.platform !== 'win32', timeout:30000}, async t => {
    const project = fileURLToPath(new URL('../../', import.meta.url));
    const output = join(project, 'test-output');
    await mkdir(output, {recursive:true});
    const root = await mkdtemp(join(output, 'control-web-auto-'));
    const python = join(project, '.venv', 'Scripts', 'python.exe');
    const source = join(root, 'src', 'control_harness');
    const directory = join(root, 'resource');
    await mkdir(source, {recursive:true});
    for (const file of ['locking.py','matlab_resource.py']) {
      await copyFile(join(project, 'src', 'control_harness', file), join(source, file));
    }
    await writeFile(join(source, '__init__.py'), '');
    const child = spawn(python, ['-c','import sys; sys.stdin.readline()'], {windowsHide:true, stdio:'pipe'});
    const stopped = once(child, 'exit');
    await once(child, 'spawn');
    const stop = new AbortController();
    let dispose;
    t.after(async () => {
      stop.abort();
      dispose?.();
      child.stdin.end();
      await stopped;
      await rm(root, {recursive:true, force:true});
    });
    const identity = {session_id:randomUUID(), matlab_pid:child.pid};
    const runId = randomUUID().replaceAll('-','');
    const endpoint = createServer((_req,res) => {
      res.writeHead(401);
      res.end('dsh web authentication required; reopen the URL printed by dsh web.\n');
    });
    endpoint.listen(0,'127.0.0.1');
    await once(endpoint,'listening');
    t.after(()=>endpoint.close());
    const env = {...process.env, CONTROL_ONECLICK_HOST_RUN:runId, CONTROL_CODE_ROOT:root, CONTROL_PYTHON:python,
      CONTROL_MATLAB_RESOURCE_DIRECTORY:directory, PYTHONPATH:join(root,'src')};
    await promisify(execFile)(python, ['-m','control_harness.matlab_resource','bind-session',
      '--directory',directory,'--pid',String(child.pid),'--session-id',identity.session_id],
      {cwd:root, env, windowsHide:true});
    const codes = [];
    let observe, alarm;
    const restart = new Promise((resolve,reject) => {
      observe = resolve;
      alarm = setTimeout(() => reject(new Error('Supervisor did not observe the exited instance')), 25000);
    });
    t.after(()=>clearTimeout(alarm));
    const ctx = {webServer:{port:endpoint.address().port}, agents:{list:()=>[]}, jobs:{list:()=>[]}, logger:{warn:msg=>assert.fail(msg)},
      effect:setup=>{ dispose = setup(); }, get:name=>{
        assert.equal(name,'appExit');
        return code=>{codes.push(code); observe();};
      }};
    await superviseWebBinding(ctx,identity,stop.signal,env);
    const readyFile = join(root,'.dsh','oneclick','ready-'+runId+'.json');
    let receipt;
    const deadline = Date.now()+8000;
    while (!receipt && Date.now()<deadline) {
      try { receipt = JSON.parse(await readFile(readyFile,'utf8')); }
      catch (error) { if (error.code!=='ENOENT') throw error; await delay(100); }
    }
    assert.equal(receipt?.launch_id,runId);
    assert.equal(receipt.session.session_id,identity.session_id);
    assert.equal(receipt.port,ctx.webServer.port);
    child.stdin.end();
    await stopped;
    await restart;
    assert.deepEqual(codes,[76]);
  });
