/** Pure admission requirements; workflow JSON and declared resources are data only. */
import { open, readFile, realpath, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

export const protectedOperations = {
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
// Cross-language tests compare this mirror with the actual Python workflow.STAGES.
export const workflowStageInputs = {
  register_model: { model_path: 'file' }, register_python_model: { model_path: 'file' },
  import_artifact: { artifact_path: 'artifact_directory' },
  import_transitions: { nominal_model: 'artifact', data_path: 'file' },
  simulate_transitions: { nominal_model: 'artifact', reference_model: 'artifact' },
  train_residual: { nominal_model: 'artifact', dataset: 'artifact' },
  design_controller: { model: 'artifact', evaluation_model: 'artifact' },
  collect_expert: { controller: 'artifact', evaluation_model: 'artifact' },
  train_imitation: { dataset: 'artifact' }, optimize_policy: { policy: 'artifact', model: 'artifact' },
  identify_linear_model: { nominal_model: 'artifact', dataset: 'artifact' },
  validate_estimator: { model: 'artifact', controller: 'artifact', evaluation_model: 'artifact' },
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

function artifactReader(root, manifest) {
  const cache = new Map(), versions = new Map(), visiting = new Set();
  async function artifact(id) {
    if (cache.has(id)) return cache.get(id);
    if (visiting.has(id)) throw new Error('Cyclic case artifact dependency');
    const row = manifest.artifacts?.[id];
    if (!row || row.status !== 'verified') throw new Error('Case dependency is missing, stale or rejected');
    visiting.add(id);
    const directory = await realpath(resolve(root, row.path)), file = await realpath(join(directory, 'artifact.json'));
    if (!inside(root, directory) || !inside(directory, file)) throw new Error('Case dependency escapes its workspace');
    const observed = await stableFile(file, true);
    if (row.metadata_file_sha256 && observed.record.sha256 !== row.metadata_file_sha256) throw new Error('Case artifact metadata changed');
    const meta = JSON.parse(observed.content.toString('utf8'));
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
  return { artifact, closure };
}

export async function resolveWorkflowCase(request) {
  const root = await realpath(request.workspace), manifest = await readManifest(root), reader = artifactReader(root, manifest);
  const inputs = request.inputs ?? {}, config = request.config ?? {}, bindings = {}, versions = {}, selectedScopes = new Set(), seen = new Set();
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
    const required = workflowStageInputs[step.operation];
    const supplied = step.inputs ?? {};
    if (!object(supplied) || !object(step.config ?? {}) || Object.keys(required).some(key => !Object.hasOwn(supplied, key))) throw new Error('Workflow stage inputs/config are incomplete');
    for (const scope of protectedOperations[step.operation] ?? []) selectedScopes.add(scope);
    for (const [name, value] of Object.entries(supplied)) {
      const expected = step.operation === 'import_transitions' && name === 'reference_model' ? 'artifact' : required[name];
      if (!expected) throw new Error(`Unknown workflow input ${name} for ${step.operation}`);
      const key = `${step.id}.${name}`;
      if (object(value)) {
        if (expected !== 'artifact' || Object.keys(value).length !== 1 || !seen.has(value.stage)) throw new Error('Workflow reference must name an earlier stage output');
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
          versions[key] = { kind: 'artifact_directory', path, workspace: sourceRoot, project_id: sourceManifest.project_id,
            id: meta.id, artifacts: await sourceReader.closure(meta.id) };
        }
      } else {
        await reader.artifact(value);
        versions[key] = { kind: 'workspace_artifact', id: value, artifacts: await reader.closure(value) };
      }
    }
    seen.add(step.id);
  }
  bindings.workflow_inputs = versions;
  return { definition, bindings, scopes: scopesInOrder.filter(scope => selectedScopes.has(scope)),
    payload: { inputs: clone(inputs), config: configSummary(config), workflow_definition: clone(definition) } };
}

export async function confirmationRequirements(request) {
  if (request.operation === 'run_workflow') {
    const selected = await resolveWorkflowCase(request);
    return selected.scopes.map(scope => ({ scope, expectedPayload: selected.payload, bindings: selected.bindings }));
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
  if(request.operation==='evaluate_external_imitation'&&(!Array.isArray(config.scenarios)||config.scenarios.length===0))throw new Error('Prepare and review explicit evaluation scenarios before confirming an original-Plant comparison');
  if (request.operation === 'sample_model') delete config.sampling_run;
  const blueprint = { inputs: request.inputs ?? {}, config };
  const selected = ['nominal_model', 'model', 'dataset', 'controller', 'policy', 'evaluation_model'].map(key => metadata[key]?.interface).find(Boolean);
  let signals = request.operation === 'prepare_supervised_dataset' ? config.interface : selected ?? blueprint;
  // The single external-import signals review includes the complete declared
  // contract; configSummary already excludes confirmation/resume references.
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
  return scopes.map(scope => ({ scope, expectedPayload: scope === 'signals' ? signals : request.operation === 'demo' ? blueprint : config, bindings }));
}
