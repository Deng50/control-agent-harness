/** Pure admission requirements; workflow JSON and declared resources are data only. */
import { open, readFile, realpath, stat } from 'node:fs/promises';
import { createHash, createHmac, timingSafeEqual } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

import {stages, payloads, allowedKinds, kindIssue, capabilityIssues, workflowStageInputs} from './stage-contracts.mjs';
export {workflowStageInputs};

export const protectedOperations = {
  convert_plant_predictor: ['signals'], verify_design_delivery: ['signals'],
  check_matlab_design: ['signals'],
  prepare_supervised_dataset: ['signals'],
  train_supervised: ['signals', 'training'],
  train_predictive_model: ['signals', 'training'],
  train_external_imitation: ['signals', 'training'],
  evaluate_external_imitation: ['signals', 'sampling'],
  prepare_pinn_problem: ['signals'], train_pinn: ['signals', 'training'],
  validate_differentiable_model: ['signals'], optimize_external_policy: ['signals','training'], evaluate_external_pgac: ['signals','sampling'],
  sample_model: ['signals', 'sampling'],
  simulate_transitions: ['signals', 'sampling'], collect_expert: ['signals', 'sampling'],
  validate_estimator: ['signals', 'sampling'],
  train_residual: ['signals', 'training'], train_imitation: ['signals', 'training'],
  optimize_policy: ['signals', 'training'], identify_linear_model: ['signals', 'training'],
  design_controller: ['signals', 'training'], run_workflow: [], demo: ['signals', 'sampling', 'training'],
  tune_controller: ['signals', 'training'],
  collect_external_expert: ['signals', 'sampling'], import_expert_data: ['signals'],
};
const scopesInOrder = ['signals', 'sampling', 'training'];
const resumeFields = ['plan_refs', 'resume_checkpoint', 'resume_checkpoint_sha256', 'resume_run'];
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const clone = value => JSON.parse(JSON.stringify(value));
const configSummary = config => clone(Object.fromEntries(Object.entries(config).filter(([key]) => !resumeFields.includes(key))));

export function synthesisScopes(request) {
  const inputs = request.inputs ?? {}, config = request.config ?? {};
  const prediction = object(config.prediction) ? config.prediction : {};
  const method = prediction.method;
  let numeric = inputs.model != null || inputs.dataset != null || prediction.model_spec != null;
  if (!numeric) {
    const description = typeof inputs.description === 'string' ? inputs.description : '';
    const template = config.template ?? (/spring/i.test(description) || description.includes('弹簧') ? 'spring_damper' : 'generic');
    const required = ['mass','damping','stiffness','sample_time','state_bounds','input_bounds','output_bounds'];
    numeric = template === 'spring_damper' && object(config.parameters) && required.every(key => config.parameters[key] != null);
  }
  if (!numeric) return [];
  return method === 'identify_affine' || method == null && inputs.dataset != null ? ['signals','training'] : ['signals'];
}
const inside = (root, target) => {
  const path = relative(root, target);
  return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith(`..${sep}`));
};
const expand = value => value === '~' ? homedir() : /^~[/\\]/.test(value) ? join(homedir(), value.slice(2)) : value;
const signature = info => [info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].map(String).join(':');

async function stableFile(value, content = false) {
  if (typeof value !== 'string' || !isAbsolute(expand(value))) throw new Error('Declared workflow resources require absolute paths');
  const path = await realpath(expand(value)), before = await stat(path, { bigint: true });
  if (!before.isFile()) throw new Error('Declared workflow resource is not a file');
  const handle = await open(path, 'r'), hash = createHash('sha256'), chunks = [];
  let count = 0;
  try {
    if (signature(await handle.stat({ bigint: true })) !== signature(before)) throw new Error('Workflow source changed before reading');
    for await (const chunk of handle.createReadStream({ autoClose: false })) {
      hash.update(chunk); count += chunk.length;
      if (content) chunks.push(chunk);
    }
    if (signature(await handle.stat({ bigint: true })) !== signature(before)
      || signature(await stat(path, { bigint: true })) !== signature(before) || BigInt(count) !== before.size) throw new Error('Workflow source changed while reading');
  } finally { await handle.close(); }
  return { record: { kind: 'file', path, sha256: hash.digest('hex'), size_bytes: count }, content: content ? Buffer.concat(chunks) : undefined };
}
const fileHash = async path => (await stableFile(path)).record.sha256;
async function readManifest(root, required = false) {
  try { return JSON.parse(await readFile(join(root, 'manifest.json'), 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT' || required) throw error; return { artifacts: {} }; }
}

function artifactReader(root, manifest, restoreReplaced = false) {
  const cache = new Map(), versions = new Map(), visiting = new Set();
  async function artifact(id) {
    if (cache.has(id)) return cache.get(id);
    if (visiting.has(id)) throw new Error('Cyclic case artifact dependency');
    const row = manifest.artifacts?.[id];
    const old = row?.stale_reason?.match(/^Upstream (\S+) replaced by (\S+)$/);
    const migrated = restoreReplaced && row?.status==='stale' && old && manifest.artifacts?.[old[2]]?.kind===old[1];
    if (!row || (row.status!=='verified' && !migrated)) throw new Error('Case dependency is missing, stale or rejected: '+id);
    visiting.add(id);
    const directory = await realpath(resolve(root, row.path)), file = await realpath(join(directory, 'artifact.json'));
    if (!inside(root, directory) || !inside(directory, file)) throw new Error('Case dependency escapes its workspace');
    const observed = await stableFile(file, true);
    if (row.metadata_file_sha256 && observed.record.sha256 !== row.metadata_file_sha256) throw new Error('Case artifact metadata changed');
    const meta = JSON.parse(observed.content.toString('utf8'));
    if (meta.validation?.status !== 'verified' || meta.kind !== row.kind) throw new Error('Artifact validation or kind differs from the registered version: '+id);
    if (meta.id !== id || !object(meta.files) || Object.keys(meta.files).length > 10000) throw new Error('Invalid case artifact identity or file inventory');
    for (const [name, expected] of Object.entries(meta.files)) {
      const actual = await realpath(resolve(directory, name));
      if (!inside(directory, actual) || await fileHash(actual) !== expected) throw new Error('Case material/artifact file changed; review the new version');
    }
    for (const parent of meta.parents ?? []) {
      if (manifest.artifacts?.[parent.id]?.metadata_hash !== parent.metadata_hash) throw new Error('Case parent version changed');
      await artifact(parent.id);
    }
    versions.set(id, { id, metadata_hash: row.metadata_hash, metadata_file_sha256: observed.record.sha256,
      content_hash: meta.source?.content_hash ?? null, files: clone(meta.files), parents: clone(meta.parents ?? []) });
    visiting.delete(id); cache.set(id, meta);
    return meta;
  }
  async function closure(id) {
    await artifact(id);
    const selected = {};
    function visit(key) {
      if (selected[key]) return;
      selected[key] = versions.get(key);
      for (const parent of selected[key].parents) visit(parent.id);
    }
    visit(id); return selected;
  }
  async function descriptor(id,name) {
    const meta=await artifact(id);
    const directory=await realpath(resolve(root,manifest.artifacts[id].path));
    const path=await realpath(resolve(directory,name));
    if (!Object.hasOwn(meta.files,name)||!inside(directory,path)) throw new Error('Descriptor escapes the selected artifact');
    const observed=await stableFile(path,true);
    if (observed.record.sha256!==meta.files[name]) throw new Error('Selected descriptor changed during preflight');
    return JSON.parse(observed.content.toString('utf8'));
  }
  return { artifact, closure, descriptor };
}

// Python 已在发布时验证父闭包。Host 核实清单已提交，且返回路径和实际产物/交付报告是同一版本。
export async function verifyPublication(workspace,result) {
  if (!result?.artifact_id || !result.artifact_path || result.delivery?.artifact_id!==result.artifact_id) throw new Error('缺少产物路径或交付证据');
  const root=await realpath(workspace), manifest=await readManifest(root,true);
  const row=manifest.artifacts?.[result.artifact_id];
  if (!row || row.kind!==result.kind || row.status!==result.status) throw new Error('交付结果未绑定当前工作区清单');
  const directory=await realpath(resolve(root,row.path));
  if (!inside(root,directory) || directory!==await realpath(result.artifact_path)) throw new Error('交付结果与登记路径不同');
  const descriptor=await realpath(join(directory,'artifact.json'));
  if (!inside(directory,descriptor)) throw new Error('产物描述越出登记目录');
  const observed=await stableFile(descriptor,true), meta=JSON.parse(observed.content.toString('utf8'));
  if (meta.id!==result.artifact_id || (row.metadata_file_sha256 && observed.record.sha256!==row.metadata_file_sha256)) throw new Error('交付产物描述版本不符');
  for (const filename of Object.keys(meta.files)) {
    const target=await realpath(join(directory,filename));
    if (!inside(directory,target) || !(await stat(target)).isFile()) throw new Error('缺少交付文件 '+filename);
  }
  if (result.delivery.target && result.delivery.request_satisfied===true) {
    const proof=await realpath(join(directory,'delivery.json'));
    if (!inside(directory,proof) || await fileHash(proof)!==meta.files['delivery.json']) throw new Error('交付凭证版本不符');
  }
}

export async function resolveWorkflowCase(request) {
  const root = await realpath(request.workspace), manifest = await readManifest(root), reader = artifactReader(root, manifest);
  const inputs = request.inputs ?? {}, config = request.config ?? {}, bindings = {}, versions = {}, selectedScopes = new Set(), seen = new Map();
  let definition, base;
  if (inputs.workflow_path) {
    if (Object.hasOwn(config, 'steps')) throw new Error('Choose either workflow_path or inline steps');
    const observed = await stableFile(inputs.workflow_path, true);
    bindings.workflow_file = observed.record;
    definition = JSON.parse(observed.content.toString('utf8')); base = dirname(observed.record.path);
  } else definition = clone(Object.fromEntries(['schema_version', 'name', 'steps'].filter(key => Object.hasOwn(config, key)).map(key => [key, config[key]])));
  if (!object(definition) || definition.schema_version !== 1 || !Array.isArray(definition.steps) || definition.steps.length < 1 || definition.steps.length > 50) throw new Error('Workflow requires schema_version 1 and 1-50 explicit stages');
  for (const step of definition.steps) {
    if (!object(step) || !/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(step.id ?? '') || seen.has(step.id)) throw new Error('Workflow stage ids must be unique valid identifiers');
    if (!Object.hasOwn(workflowStageInputs, step.operation)) throw new Error(`Unsupported workflow operation: ${step.operation}`);
    const stage = stages[step.operation];
    const required = workflowStageInputs[step.operation];
    let output = stage.workflow_output;
    if (['register_model','register_python_model'].includes(step.operation) && step.config?.source?.kind === 'reference_simulator') output='reference_model';
    const supplied = step.inputs ?? {};
    if (!object(supplied) || !object(step.config ?? {}) || Object.keys(required).some(key => !Object.hasOwn(supplied, key))) throw new Error('Workflow stage inputs/config are incomplete');
    for (const scope of protectedOperations[step.operation] ?? []) selectedScopes.add(scope);
    for (const [name, value] of Object.entries(supplied)) {
      const expected = step.operation === 'import_transitions' && name === 'reference_model' ? 'artifact' : required[name];
      if (!expected) throw new Error(`Unknown workflow input ${name} for ${step.operation}`);
      const key = `${step.id}.${name}`;
      const type = stage.inputs[name] ?? stage.optional?.[name];
      const checkKind = actual => {const issue=kindIssue(key,type,actual); if(issue) throw Object.assign(new Error(key+': '+issue.message),{issue});};
      if (object(value)) {
        if (expected !== 'artifact' || Object.keys(value).length !== 1 || !seen.has(value.stage)) throw new Error('Workflow reference must name an earlier stage output');
        checkKind(seen.get(value.stage));
        continue;
      }
      if (typeof value !== 'string' || !value.trim()) throw new Error('Workflow input must be a declared path, artifact id or stage reference');
      if (expected === 'file' || expected === 'artifact_directory') {
        let path = expand(value);
        if (!isAbsolute(path) && base) path = resolve(base, path);
        if (!isAbsolute(path)) throw new Error('Inline workflow paths must be absolute');
        path = await realpath(path); supplied[name] = path;
        if (expected === 'file') versions[key] = (await stableFile(path)).record;
        else {
          const meta = JSON.parse(await readFile(join(path, 'artifact.json'), 'utf8')), sourceRoot = dirname(dirname(path));
          const sourceManifest = await readManifest(sourceRoot, true), sourceReader = artifactReader(sourceRoot, sourceManifest);
          if (!sourceManifest.artifacts?.[meta.id] || await realpath(resolve(sourceRoot, sourceManifest.artifacts[meta.id].path)) !== path) throw new Error('External artifact directory does not match its source manifest');
          output=meta.kind;
          versions[key] = { kind: 'artifact_directory', path, workspace: sourceRoot, project_id: sourceManifest.project_id,
            id: meta.id, artifacts: await sourceReader.closure(meta.id) };
        }
      } else {
        checkKind((await reader.artifact(value)).kind);
        versions[key] = { kind: 'workspace_artifact', id: value, artifacts: await reader.closure(value) };
      }
    }
    seen.set(step.id,output);
  }
  bindings.workflow_inputs = versions;
  return { definition, bindings, scopes: scopesInOrder.filter(scope => selectedScopes.has(scope)),
    payload: { inputs: clone(inputs), config: configSummary(config), workflow_definition: clone(definition) } };
}

// 预检只读所选版本，不具备确认或作业授权。
export async function stagePreflight(request) {
  const stage = stages[request.operation], issues = [], metadata = {};
  if (request.operation === 'import_artifact' && request.inputs?.artifact_path) {
    const input=request.inputs, field='inputs.artifact_path';
    try {
      if (!isAbsolute(input.artifact_path)) throw new Error('交接需要来源工作区中的完整绝对目录。');
      const directory=await realpath(input.artifact_path), sourceRoot=dirname(dirname(directory));
      const manifest=await readManifest(sourceRoot,true);
      // 从已登记目录反查 ID，核对声明版本后才读取选中的父闭包。
      const entry=Object.entries(manifest.artifacts).find(([,row])=>resolve(sourceRoot,row.path)===directory);
      if (!entry) throw new Error('来源目录未登记，独立导出包不能替代带原始依赖的工作区产物。');
      const [id,row]=entry;
      for (const [key,actual] of [['expected_artifact_id',id],['expected_metadata_hash',row.metadata_hash],['expected_project_id',manifest.project_id]]) {
        if (input[key]!==undefined && input[key]!==actual) throw new Error('交接版本已变化：inputs.'+key);
      }
      await artifactReader(sourceRoot,manifest,sourceRoot!==await realpath(request.workspace)).artifact(id);
    } catch(error) {issues.push({status:'stale_dependency',field,message:error.message});}
  }
  if (stage) {
    const root = await realpath(request.workspace).catch(error=>{if(error.code==='ENOENT')return resolve(request.workspace);throw error;});
    const manifest = await readManifest(root);
    const reader = artifactReader(root, manifest);
    for (const [key,type] of Object.entries({...stage.inputs,...stage.optional})) {
      if (!allowedKinds(type).length) continue;
      const id = request.inputs?.[key], field = 'inputs.'+key;
      if (id == null || id === '') {
        if (Object.hasOwn(stage.inputs,key)) issues.push({status:'clarify',field,message:'请选择该步骤需要的产物。'});
        continue;
      }
      try {
        const meta = await reader.artifact(id);
        const mismatch = kindIssue(field,type,meta.kind);
        if (mismatch) {issues.push(mismatch); continue;}
        const required=payloads[meta.kind] ?? [];
        const missing=required.filter(name=>!Object.hasOwn(meta.files,name));
        if(missing.length) {issues.push({status:'unsupported',field,message:'所选产物的文件清单缺少 '+missing.join(', ')});continue;}
        const name=required[0], declared=['model.json','plant.json','expert.json','controller.json','design.json'].includes(name)
          ? await reader.descriptor(id,name) : undefined;
        if (name==='model.json' && !['discrete_polynomial','continuous_polynomial','python_torch_transition','python_torch_derivative'].includes(declared.representation)) {
          issues.push({status:'unsupported',field,message:'预测器加载入口不支持 model.json 的表示 '+declared.representation});continue;
        }
        if(name==='controller.json' && (!object(declared.parameters)||!meta.parents.some(p=>p.id===declared.prediction_model))) {
          issues.push({status:'unsupported',field,message:'控制器 descriptor 未绑定预测模型或可执行参数。'});continue;
        }
        metadata[key]={...meta,descriptor:declared};
        if(name==='controller.json') metadata[key].predictor=await reader.artifact(declared.prediction_model);
      } catch(error) {issues.push({status:'stale_dependency',field,artifact_id:id,message:error.message});}
    }
    issues.push(...capabilityIssues(request,metadata));
    try {validateStructuredInputs(request,metadata);} catch(error) {issues.push({status:'unsupported',field:error.message.split(':')[0],message:error.message});}
  }
  const status=['stale_dependency','unsupported','clarify'].find(s=>issues.some(i=>i.status===s)) ?? 'ready';
  return {operation:request.operation,status,issues,jobs_started:0,verification:'artifact_contract_only'};
}

function validateStructuredInputs(request, metadata) {
  if (request.operation === 'prepare_supervised_dataset' && metadata.material && !['csv','xlsx'].includes(metadata.material?.interface?.format)) {
    throw new Error('inputs.material: supervised data preparation requires a registered CSV or XLSX material');
  }
  const iface = metadata.evaluation_model?.interface ?? metadata.plant?.interface ?? metadata.model?.interface;
  for (const group of ['scenarios','tuning_scenarios','verification_scenarios','test_scenarios']) {
    const rows = request.config?.[group];
    if (!Array.isArray(rows)) continue;
    if (!rows.length) throw new Error(`config.${group} requires explicit operating conditions`);
    for (const row of rows) {
      for (const [key, axis] of Object.entries({initial_state:'states',reference:'outputs',disturbance:'disturbances',initial_previous_action:'controls'})) {
        if (Array.isArray(row[key]) && Array.isArray(iface?.[axis]) && row[key].length !== iface[axis].length) {
          throw new Error(`config.${group}.${key}: expected ${iface[axis].length} values for the selected interface`);
        }
      }
    }
  }
}

// Only the fixed, signed preparation stage can bind a future training request
// to a material. A copied, imported or old dataset remains eligible for an
// independent review of its exact dataset version.
async function preparationEvidence({root, manifest, reader, dataset, agent, options, request}) {
  const proof = dataset.source?.preparation_proof;
  if (!proof || !options.derivationKey) return;
  const fail = () => { throw new Error('The supervised dataset preparation proof does not match its verified inputs or output'); };
  if (typeof proof.body !== 'string') fail();
  const body = JSON.parse(proof.body);
  // A foreign workspace cannot extend a local grant. It takes the exact
  // dataset review route even when its original Host used a different key.
  if (typeof body?.workspace !== 'string') fail();
  if (body.workspace !== root) return;
  if (!/^[a-f0-9]{64}$/.test(proof.signature ?? '')) fail();
  const expected = createHmac('sha256',options.derivationKey).update(proof.body,'utf8').digest();
  if (!timingSafeEqual(expected,Buffer.from(proof.signature,'hex'))) fail();
  const parent = dataset.parents?.[0];
  if (body.kind !== 'control-supervised-preparation-v2' || dataset.parents?.length !== 1 ||
      body.dataset_id !== dataset.id || parent.id !== dataset.source.material_artifact) fail();
  const material = await reader.artifact(parent.id);
  const binding = {id:parent.id,metadata_hash:parent.metadata_hash,content_hash:material.source?.content_hash};
  if (material.kind !== 'material' || !binding.content_hash || dataset.source.content_hash !== binding.content_hash ||
      !isDeepStrictEqual(body.material,binding) || !isDeepStrictEqual(body.interface,dataset.interface)) fail();
  for (const file of ['config.json','supervised.npz','dataset.json','split.json','scaler.json']) {
    if (!dataset.files[file] || body.files?.[file] !== dataset.files[file]) fail();
  }
  const directory = await realpath(resolve(root,manifest.artifacts[dataset.id].path));
  const configPath = await realpath(join(directory,'config.json'));
  if (!inside(directory,configPath)) fail();
  const observed = await stableFile(configPath,true);
  if (observed.record.sha256 !== dataset.files['config.json']) fail();
  const preparation = JSON.parse(observed.content.toString('utf8'));
  if (!isDeepStrictEqual(body.config,configSummary(preparation)) ||
      !isDeepStrictEqual(body.signals,preparation.plan_refs?.signals)) fail();
  if (body.session_id !== agent?.id || body.intent_id !== options.intent_id) return;
  const document = await readManifestPlans(root);
  const active = document.active_by_intent?.[options.intent_id]?.signals;
  const record = document.plans?.[active];
  if (!record || active !== body.signals.plan_id || record.version_hash !== body.signals.version_hash) return;
  if (request.config?.plan_refs?.signals && !isDeepStrictEqual(request.config.plan_refs.signals,body.signals)) return;
  return {material:binding,config:body.config};
}

async function readManifestPlans(root) {
  const path = await realpath(join(root,'.case-plans','plans.json'));
  if (!inside(root,path)) throw new Error('Case plan directory escapes its workspace');
  return JSON.parse(await readFile(path,'utf8'));
}

export async function confirmationRequirements(request, agent, options) {
  if (request.operation === 'run_workflow') {
    const selected = await resolveWorkflowCase(request);
    return selected.scopes.map(scope => ({ scope, expectedPayload: selected.payload, bindings: selected.bindings,
      ...(options?.intent_id ? {reviewPayload:{operation:request.operation,summary:selected.payload},reviewBindings:selected.bindings} : {}) }));
  }
  const scopes = request.operation === 'synthesize_mpc' ? synthesisScopes(request) : protectedOperations[request.operation];
  if (!scopes) return [];
  const root = await realpath(request.workspace), manifest = await readManifest(root), reader = artifactReader(root, manifest);
  const bindings = {}, metadata = {};
  for (const [key, value] of Object.entries(request.inputs ?? {})) {
    const row = typeof value === 'string' ? manifest.artifacts?.[value] : undefined;
    if (!row) continue;
    const meta = await reader.artifact(value);
    bindings[key] = { id: value, metadata_hash: row.metadata_hash, content_hash: meta.source?.content_hash ?? null };
    metadata[key] = meta;
  }
  const config = configSummary(request.config ?? {});
  if (options?.intent_id) validateStructuredInputs(request,metadata);
  if(request.operation==='evaluate_external_imitation'&&(!Array.isArray(config.scenarios)||config.scenarios.length===0))throw new Error('Prepare and review explicit evaluation scenarios before confirming an original-Plant comparison');
  if (request.operation === 'sample_model') delete config.sampling_run;
  const blueprint = { inputs: request.inputs ?? {}, config };
  const selected = ['nominal_model', 'model', 'dataset', 'controller', 'policy', 'evaluation_model'].map(key => metadata[key]?.interface).find(Boolean);
  let signals = request.operation === 'prepare_supervised_dataset' ? config.interface : selected ?? blueprint;
  // The single external-import signals review includes the complete declared
  // contract; configSummary already excludes confirmation/resume references.
  if (['convert_plant_predictor','verify_design_delivery'].includes(request.operation)) signals=blueprint;
  if(['import_expert_data','prepare_pinn_problem'].includes(request.operation))signals=config;
  if(request.operation==='train_pinn'){
    const problem=metadata.problem;signals=problem?.source?.pinn_signals;
    if(problem?.kind!=='conditional_ode_problem'||!object(signals))throw new Error('Select an explicit conditional ODE problem with equation, time/condition units and normalization before PINN training');
  }
  if(request.operation==='validate_differentiable_model')signals={candidate_interface:metadata.model?.interface??null,original_interface:metadata.evaluation_model?.interface??null,validation_config:config};
  if(request.operation==='optimize_external_policy')signals={rollout_model_interface:metadata.model?.interface??null,policy_signal_interface:metadata.policy?.interface??null,validated_source:metadata.validation?.source??null};
  if(request.operation==='evaluate_external_pgac'){
    if(!Array.isArray(config.scenarios)||!config.scenarios.length)throw new Error('Review the frozen original-Plant validation/test scenarios before confirming PGAC evaluation');
    signals=metadata.candidate?.interface;
  }
  if(request.operation==='collect_external_expert'){
    if(!metadata.plant||metadata.expert?.kind!=='external_expert')throw new Error('Select bound external expert and plant artifacts before confirming collection');
    const expertRoot=await realpath(resolve(root,manifest.artifacts[request.inputs.expert].path)),path=await realpath(resolve(expertRoot,'expert.json'));
    if(!inside(expertRoot,path))throw new Error('External expert declaration escapes its artifact');
    const observed=await stableFile(path,true);
    if(observed.record.sha256!==metadata.expert.files?.['expert.json'])throw new Error('External expert declaration changed before confirmation');
    const spec=JSON.parse(observed.content.toString('utf8')).adapter_spec;
    const fields=['interface','control_period','time_unit','observations','reference_available','disturbance_available','previous_action_available','initial_memory','action_mapping','failure_policy','input_rate_bounds'];
    signals={plant_interface:metadata.plant.interface,expert_adapter:Object.fromEntries(fields.filter(key=>Object.hasOwn(spec,key)).map(key=>[key,spec[key]]))};
  }
  if (!object(signals)) throw new Error('A complete signal interface is required before confirmation');
  const rows = scopes.map(scope => ({ scope, expectedPayload: scope === 'signals' ? signals : request.operation === 'demo' ? blueprint : config, bindings }));
  if (!options?.intent_id) return rows;
  const operation = request.operation;
  const derived = operation === 'train_supervised' && metadata.dataset
    ? await preparationEvidence({root,manifest,reader,dataset:metadata.dataset,agent,options,request}) : undefined;
  for (const row of rows) {
    row.reviewPayload = {operation,summary:row.expectedPayload};
    row.reviewBindings = bindings;
    if (operation === 'synthesize_mpc' && row.scope === 'signals') Object.assign(row.reviewPayload,{config,inputs:request.inputs ?? {}});
    if (operation === 'prepare_supervised_dataset') row.reviewPayload = {operation,config};
    if (derived) {
      row.reviewPayload = row.scope === 'signals' ? {operation:'prepare_supervised_dataset',config:derived.config}
        : {operation,config,preparation:derived.config};
      row.reviewBindings = {material:derived.material};
    }
  }
  return rows;
}
