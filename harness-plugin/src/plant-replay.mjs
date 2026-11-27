import { createHash,randomUUID } from 'node:crypto';
import { isAbsolute, join } from 'node:path';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { readPlantReference } from './simulink-build.mjs';
import { authorizeWorkspace } from './job-history.mjs';
import { pinDirectoryPath } from './windows-host-security.mjs';
import { plantWorkspacePolicy } from './workspace-policy.mjs';
import { readWorkspaceBytes, workspacePythonName, maxPythonBytes } from './workspace-file.mjs';

export function replayTool(ctx,config,getContext,check) {
  return defineTool({name:'control_replay_plant_source',
    description:'Generate Python reference trajectories for a source-to-Simulink build. Requires a module-level step(x,u,d,dt,parameters). Select an attached file by name or ID, or a previously inspected .py in the workspace root. Runs in the native workspace-write sandbox. Supply {parameters,sampleTime,absTolerance,trajectories:[{initialCondition,controls,disturbances}]} or requestFile. Use an empty disturbance row per step when there are no disturbances. Returns referenceFile, source SHA-256 and sample count.',
    parameters:{source:{type:'string',required:true},request:{type:'object',additionalProperties:true},
      requestFile:{type:'string',description:'Single workspace JSON filename, alternative to inline request'}},
    output:{schema:{type:'object',additionalProperties:true},render:(_args,value)=>[{type:'text',text:JSON.stringify(value)}]},
    async execute({source,request,requestFile},{agent,signal}) {
      const c=await getContext(agent);
      const sources=c?.state.sources.filter(s=>s.name===source || s.attachmentId===source) ?? [];
      const requestedName=c && isAbsolute(source) ? workspacePythonName(c.session.header.cwd,source) : source;
      const localName=c?.state.workspaceSources.find(name=>name===requestedName ||
        process.platform==='win32' && name.toLowerCase()===requestedName?.toLowerCase());
      if (sources.length>1) throw new Error('同名附件请按 attachmentId 选择。');
      if (!sources.length && !localName) throw new Error('请指定本次对话的 Python 附件，或先用 read 读取本工作区根目录的 Python 文件再提交文件名。');
      if ((request === undefined)===(requestFile === undefined)) throw new Error('请提供 request 或工作区内的 requestFile。');
      await check(agent,signal,c);
      const policy=plantWorkspacePolicy(ctx,agent.session);
      if (policy.mode!=='workspace-write') throw new Error('源回放需要本次对话授权在工作区内写入。');
      const workspace=await authorizeWorkspace(agent.session.header.cwd,policy,true);
      const release=pinDirectoryPath(workspace);
      try {
        // Freeze the inspected workspace file in the native content-addressed store.
        // Only the confined Python process executes those bytes.
        const attachment=sources[0] ?? await ctx.attachments.saveFile({name:localName,data:await readWorkspaceBytes(workspace,localName,maxPythonBytes)});
        const sourcePath=ctx.attachments.fileHostPath(attachment);
        const match=/^sha256:([a-f0-9]{64})$/.exec(attachment.attachmentId);
        if (!sourcePath || !match) throw new Error('当前附件后端未提供可核验的本地源文件。');
        const input=JSON.stringify(request ?? await readPlantReference(workspace,requestFile));
        if (Buffer.byteLength(input)>2*1024*1024) throw new Error('回放请求超过 2 MiB。');
        const referenceFile='plant-reference-'+randomUUID()+'.json';
        const target=join(workspace,referenceFile);
        const deadline=AbortSignal.timeout(60000);
        const stop=AbortSignal.any([signal,deadline]);
        const argv=[config.pythonPath,'-I','-B',join(config.codeRoot,'scripts','plant-step-reference.py'),
          '--source',sourcePath,'--expect-sha256',match[1],'--request','-','--output',target];
        const wrapped=await ctx.sandbox.confine(argv,policy,stop);
        const child=ctx.subprocess.spawn({argv:wrapped.argv,cwd:workspace,signal:stop,graceMs:1000,
          stdio:{stdin:'pipe',stdout:{maxBytes:4096},stderr:{maxBytes:8192}}});
        child.stdin.on('error',()=>{}); // Early startup failure is reported by the child's outcome.
        child.stdin.end(input);
        const outcome=await child.done;
        await child.waitForExit();
        stop.throwIfAborted();
        if (outcome.exitCode!==0) throw new Error('Python 源回放失败：'+child.collected.stderr.readFrom(0).text.trim().slice(-8192));
        const ref=await readPlantReference(workspace,referenceFile);
        if (ref.source_sha256!==match[1] || !Array.isArray(ref.trajectories)) throw new Error('回放结果与原始附件不符。');
        const proof={status:'replayed',referenceFile,source:attachment.name,source_kind:sources.length?'attachment':'workspace',source_sha256:match[1],sandbox_mode:policy.mode,
          reference_data_sha256:createHash('sha256').update(JSON.stringify(ref)).digest('hex'),
          samples:ref.trajectories.reduce((n,t)=>n+t.states.length,0)};
        c.local.replays.set(referenceFile,proof);
        return proof;
      } finally { release(); }
    },
  });
}
