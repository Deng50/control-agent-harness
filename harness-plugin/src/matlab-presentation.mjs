import {randomUUID} from 'node:crypto';

// The native executor rematerializes success from output.value. Pass a
// single-use projection receipt to the owning MCP definition so only a
// verified Host dispatch can produce session_verified presentation metadata.
const receipts = new Map();
export function partialToolkitValue(value, additional, meta) {
  const id = randomUUID();
  receipts.set(id,meta);
  // Normalization is synchronous once the middleware returns. Bound entries
  // left behind by aborted or invalid output even if no later projection runs.
  if (receipts.size>256) receipts.delete(receipts.keys().next().value);
  return {...value,content:[...value.content,...additional,{type:'text',_meta:{control_partial:id},
    text:'此批操作仅部分完成。保留返回的数据与已生效的更改，读回模型后只补做未完成的部分。'}]};
}

export function matlabOutput(output, rawName) {
  if (!rawName.startsWith('model_')) return output;
  return {...output,presentationMeta(args,value){
    const id=value.content.at(-1)?._meta?.control_partial;
    const meta=receipts.get(id);
    if (!meta) return output.presentationMeta?.(args,value) ?? null;
    receipts.delete(id);
    return meta;
  }};
}
