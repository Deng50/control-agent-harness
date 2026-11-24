// The pinned native bridge flattens a validated server error and a failed
// transport into the same Error. Preserve the distinction for this plugin
// instance only. Load the original bridge with its own dependency resolution.
import { registerHooks } from 'node:module';

const url = new URL('../../vendor/deepseek-harness/packages/mcp/mcp-client/lib/index.js', import.meta.url);
url.search = 'control-matlab';
const original = 'if (result.isError === true) throw new Error(text);';
const typed = "if (result.isError === true) throw new HarnessError(text, 'MCP_TOOL_RETURNED_ERROR');";
const createTransport = 'transport = createTransport(config);';
const nativeCall = 'call: (args, execution) => client.callTool({';
const observer = JSON.stringify(new URL('./matlab-mcp-reply.mjs',import.meta.url).href);
const output = 'output: createOutput(rawName, supportedOutputSchema(options.outputSchema)),';
const presentation = JSON.stringify(new URL('./matlab-presentation.mjs',import.meta.url).href);
const hook = registerHooks({
  load(specifier, context, nextLoad) {
    const result = nextLoad(specifier, context);
    if (specifier !== url.href) return result;
    const source = Buffer.from(result.source).toString();
    if ([original,createTransport,output,nativeCall].some(marker=>source.split(marker).length !== 2)) {
      throw new Error('Unexpected pinned DeepSeek MCP bridge. Check its tool-error boundary before loading MATLAB.');
    }
    return {...result, source:"import { HarnessError } from '@deepseek-ai/dsh-llm';\n" +
      "import { SdkError, SdkErrorCode } from '@modelcontextprotocol/client';\n" +
      `import { observeMatlabTransport, callWithMatlabDeadline } from ${observer};\nimport { matlabOutput } from ${presentation};\n` + source.replace(original, typed)
        .replace(nativeCall,"call: (args, execution) => callWithMatlabDeadline(client, timeout => new SdkError(SdkErrorCode.RequestTimeout, 'Request timed out', {timeout}), {")
        .replace(output,'output: matlabOutput(createOutput(rawName, supportedOutputSchema(options.outputSchema)), rawName),')
        .replace(createTransport,'transport = observeMatlabTransport(createTransport(config), specTypeSchemas.CallToolResult["~standard"]);')};
  },
});
let bridge;
try { bridge = await import(url.href); }
finally { hook.deregister(); }

export const {name, inject, Config, apply, createMcpToolDefinition} = bridge;
