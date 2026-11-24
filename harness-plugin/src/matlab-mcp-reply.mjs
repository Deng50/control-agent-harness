import { AsyncLocalStorage } from 'node:async_hooks';
import { randomUUID } from 'node:crypto';
import { readFile, unlink } from 'node:fs/promises';
import { renameSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';

const execution = new AsyncLocalStorage();
const replyFile = (root, token) => {
  if (!/^[a-f0-9]{32}$/.test(token)) throw new Error('Invalid MATLAB lease token');
  return join(root, 'mcp-reply-' + token + '.json');
};

export function withMcpReplyTracking(resource, lease, exec, nonce, next) {
  return execution.run({directory:resource.config.directory,logger:resource.logger,
    token:lease.token,nonce,session:lease.session,tool:exec.name,args:exec.arguments,signal:exec.signal},next);
}

// The SDK normally cancels the server's execution at the caller's deadline.
// Go MCP servers then discard the reply, even when MATLAB continues running.
// Give only the original, broker-bound call a separate observation lifetime.
// A real caller abort still reaches the SDK, including after the UI has timed out.
export async function callWithMatlabDeadline(client, makeTimeoutError, params, options) {
  const call = execution.getStore();
  if (!call || 'mcp__matlab__'+params.name !== call.tool || !isDeepStrictEqual(params.arguments,call.args)) {
    return client.callTool(params,options);
  }
  const signal = call.signal && call.signal !== options.signal
    ? AbortSignal.any([call.signal,options.signal]) : options.signal;
  let timer;
  const deadline = new Promise((_,reject)=>{
    timer = setTimeout(()=>reject(makeTimeoutError(options.timeout)),options.timeout);
    timer.unref();
  });
  try {
    const response = client.callTool(params,{...options,signal,timeout:2_147_483_647});
    return await Promise.race([response,deadline]);
  } finally {clearTimeout(timer);}
}

// IDs are scoped to this actual transport. Observe the correlated reply before
// the SDK normalizes it. The durable copy survives a Host restart after receipt.
export function observeMatlabTransport(transport, schema) {
  const generation = randomUUID(), requests = new Map();
  const send = transport.send.bind(transport);
  transport.send = (message, options) => {
    const call = execution.getStore();
    if (call && !call.sent && message.jsonrpc === '2.0' && message.method === 'tools/call' &&
        (typeof message.id === 'string' || Number.isSafeInteger(message.id)) &&
        'mcp__matlab__'+message.params?.name === call.tool && isDeepStrictEqual(message.params.arguments,call.args)) {
      call.sent = true;
      requests.set(message.id,call);
    }
    return send(message,options);
  };
  const previous = transport.onmessage;
  transport.onmessage = (message, extra) => {
    const call = requests.get(message.id);
    if (call && message.jsonrpc === '2.0' && !('method' in message) && ('result' in message || 'error' in message)) {
      requests.delete(message.id);
      try {
        // A JSON-RPC/protocol error or a server-side connection failure can
        // arrive before MATLAB finishes. Only a synchronous tool result or
        // a completed MATLAB diagnostic counts. A task handle is not a result.
        const result = message.result;
        const final = !message.error && result && Array.isArray(result.content) && !('task' in result)
          && (result.resultType === undefined || result.resultType === 'complete');
        const parsed = final ? schema.validate(result) : undefined;
        if (parsed && !parsed.issues && parsed.value && (parsed.value.isError !== true ||
            parsed.value.content.filter(item=>item.type==='text').map(item=>item.text).join('\n').startsWith('matlab error:'))) {
          const receipt = {completion_source:'mcp_response',completion_nonce:call.nonce,lease_token:call.token,
            tool_name:call.tool,session_id:call.session.session_id,matlab_pid:call.session.pid,
            process_identity:call.session.identity,transport_generation:generation,rpc_request_id:message.id};
          const path=replyFile(call.directory,call.token), temp=path+'.'+randomUUID()+'.tmp';
          try {
            writeFileSync(temp,JSON.stringify(receipt),{flag:'wx'});
            renameSync(temp,path);
          } finally {rmSync(temp,{force:true});}
        }
      } catch (error) {call.logger?.warn?.('MATLAB 原生回执未能核验或保存：'+error.message);}
    }
    return previous?.call(transport,message,extra);
  };
  const previousClose = transport.onclose;
  transport.onclose = () => { requests.clear(); return previousClose?.call(transport); };
  return transport;
}

export async function readMcpReply(directory, lease, tool) {
  let receipt;
  try {receipt=JSON.parse(await readFile(replyFile(directory,lease.token),'utf8'));}
  catch (error) {if (error.code==='ENOENT') return; throw error;}
  if (receipt.completion_source==='mcp_response' && receipt.completion_nonce===lease.completion_nonce &&
      receipt.lease_token===lease.token && receipt.tool_name===tool && receipt.session_id===lease.session.session_id &&
      receipt.matlab_pid===lease.session.pid && receipt.process_identity===lease.session.identity &&
      /^[a-f0-9-]{36}$/i.test(receipt.transport_generation) &&
      (typeof receipt.rpc_request_id==='string' || Number.isSafeInteger(receipt.rpc_request_id))) return receipt;
}

export async function forgetMcpReply(directory,token) {
  try {await unlink(replyFile(directory,token));} catch (error) {if (error.code!=='ENOENT') throw error;}
}
