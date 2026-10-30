/** Native scoped registries, not a model API or numerical acceptance fixture. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import * as rolePlugin from '../src/role.mjs';
import { agents, roleIds, rolePrompt } from '../src/roles.mjs';
import { operationHelp, operations } from '../src/catalog.mjs';

const upstream = resolve(import.meta.dirname, '../../vendor/deepseek-harness');
const built = async path => import(pathToFileURL(resolve(upstream, path)));

test('role prompt and tool restrictions are scoped in real native registries', async () => {
  const { Context } = await built('vendor/cordis/lib/index.js');
  const { default: Prompt } = await built('packages/core/system-prompt/lib/index.js');
  const { default: Tools, defineTool } = await built('packages/core/tools/lib/index.js');
  const Executor = await built('packages/core/tool-executor-native/lib/index.js');
  const { createScope } = await built('packages/core/scope/lib/index.js');
  const ctx = new Context();
  try {
    await ctx.plugin(Prompt);
    await ctx.plugin(Tools);
    await ctx.plugin(Executor);
    ctx.provide('controlEngineering', {});
    const names = ['control_model','control_controller','control_lightweight','control_workflow','control_help','job_output','pwsh'];
    for (const name of names) ctx.tools.register(defineTool({name,description:'Registry isolation test fixture',parameters:{},output:{schema:{type:'string'},render:(_args,value)=>[{type:'text',text:value}]},execute:()=>''}));
    for (const role of roleIds) {
      const key = {}, scope = createScope(ctx,key);
      try {
        await scope.ctx.plugin(rolePlugin,{role});
        const actual = ctx.tools.definitions(key).map(item=>item.name);
        for (const other of roleIds) assert.equal(actual.includes(`control_${other}`), other === role);
        for (const common of ['control_workflow','control_help','job_output','pwsh']) assert.ok(actual.includes(common));
        assert.equal(ctx.tools.get(`control_${roleIds.find(other=>other!==role)}`,key),undefined);
        const prompt = await ctx.systemPrompt.assemble({scope:key});
        const persona = prompt.sections.find(section=>section.name==='deployment:persona-prefix').text;
        assert.equal(persona,rolePrompt(role));
        assert.equal(ctx.tools.definitions().length,names.length,'a role must not mask global/sibling tools');
      } finally { await scope.dispose(); }
    }
  } finally { await ctx.fiber.dispose(); }
});

test('every admitted role operation has discoverable required inputs and recovery semantics', () => {
  for (const [role,settings] of Object.entries(agents)) for (const operation of settings.operations) {
    assert.ok(operations[operation],`${role} admits undocumented ${operation}`);
    const help=operationHelp({operation});
    assert.ok(help.agents.includes(role));
    assert.equal(help.inputsSchema.type,'object');
    assert.equal(help.configSchema.type,'object');
    assert.ok(help.prerequisites.length);
  }
  assert.deepEqual(operationHelp({operation:'recover_interrupted_stage'}).inputsSchema.required,['stage_id']);
  assert.throws(()=>operationHelp({operation:'invented'}),/Unknown/);
});
