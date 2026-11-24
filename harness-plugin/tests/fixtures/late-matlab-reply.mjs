import { appendFileSync,existsSync } from 'node:fs';
import {createInterface} from 'node:readline';
import {setTimeout as wait} from 'node:timers/promises';
import {join} from 'node:path';

const [directory,kind] = process.argv.slice(2);
const names=['model_read','run_matlab_file','evaluate_matlab_code'];
const cancelled = new Set();
createInterface({input:process.stdin}).on('line',async line=>{
  const request=JSON.parse(line);
  if (request.method==='notifications/cancelled') {
    cancelled.add(request.params.requestId);
    appendFileSync(join(directory,'cancel.log'),String(request.params.requestId)+'\n');
  }
  if (request.id===undefined) return;
  const emit=body=>process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:request.id,...body})+'\n');
  if (request.method==='server/discover') return emit({error:{code:-32601,message:'Legacy MCP server'}});
  if (request.method==='initialize') return emit({result:{protocolVersion:request.params.protocolVersion,
    capabilities:{tools:{}},serverInfo:{name:'late-reply-test',version:'1'}}});
  if (request.method==='tools/list') return emit({result:{tools:names.map(name=>({name,inputSchema:{type:'object'}}))}});
  if (request.method==='tools/call') {
    appendFileSync(join(directory,'dispatch.log'),request.params.name+'\n');
    while (!existsSync(join(directory,'release'))) await wait(10);
    if (cancelled.has(request.id)) return; // Go's MCP server suppresses a cancelled request's result.
    const success={content:[{type:'text',text:'fixture terminated'}]};
    if (kind==='wrong-id') return process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:request.id+1,result:success})+'\n');
    if (kind==='server-timeout') return emit({result:{isError:true,content:[{type:'text',text:'Connection closed'}]}});
    return emit({result:success});
  }
});
