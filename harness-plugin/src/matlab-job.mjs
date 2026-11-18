/** One native job: sandboxed prepare, ordinary native MCP call, sandboxed finalize. */
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, realpath } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { numericalJob } from './worker.mjs';

const TOOL = 'mcp__matlab__evaluate_matlab_code';
const sha = value => createHash('sha256').update(value).digest('hex');
const inside = (root, target) => { const name = relative(root, target); return name === '' || (!isAbsolute(name) && name !== '..' && !name.startsWith(`..${sep}`)); };
const quote = value => `'${String(value).replaceAll("'", "''")}'`;
const json = path => readFile(path, 'utf8').then(JSON.parse);

/** Called after ordinary case authorization, before a history/job is created. */
export async function requiresMatlabBatch(request) {
  if(request.operation==='evaluate_external_pgac')return requiresMatlabPgacEvaluation(request);
  if(request.operation==='evaluate_external_imitation')return requiresMatlabPolicyEvaluation(request);
  if(request.operation==='collect_external_expert')return requiresMatlabExpertBatch(request);
  if (request.operation !== 'sample_model') return false;
  const workspace = await realpath(request.workspace);
  const manifest = await json(resolve(workspace, 'manifest.json'));
  const id = request.inputs?.model;
  const row = typeof id === 'string' ? manifest.artifacts?.[id] : undefined;
  if (!row || row.status !== 'verified') throw new Error('Select a verified plant/model artifact for sampling');
  const root = await realpath(resolve(workspace, row.path));
  if (!inside(workspace, root)) throw new Error('Selected plant artifact escapes its workspace');
  const metadataPath = await realpath(resolve(root, 'artifact.json'));
  if (!inside(root, metadataPath)) throw new Error('Selected plant metadata escapes its artifact');
  const metadataBytes = await readFile(metadataPath);
  if (row.metadata_file_sha256 && sha(metadataBytes) !== row.metadata_file_sha256) throw new Error('Selected plant metadata changed');
  const metadata = JSON.parse(metadataBytes.toString('utf8'));
  if (metadata.id !== id || metadata.kind !== row.kind) throw new Error('Selected plant artifact identity is inconsistent');
  if (metadata.kind !== 'plant_model') return false;
  const path = await realpath(resolve(root, 'plant.json'));
  if (!inside(root, path) || !metadata.files?.['plant.json']) throw new Error('Selected plant has no confined registered descriptor');
  const bytes = await readFile(path);
  if (sha(bytes) !== metadata.files['plant.json']) throw new Error('Selected plant descriptor changed');
  const descriptor = JSON.parse(bytes.toString('utf8'));
  const backend = descriptor.adapter_spec?.backend;
  if (!['matlab_function', 'simulink', 'python_callable', 'python_factory'].includes(backend)) throw new Error('Selected plant backend is unsupported');
  return backend === 'matlab_function' || backend === 'simulink';
}

async function requiresMatlabPgacEvaluation(request){
  const workspace=await realpath(request.workspace),manifest=await json(resolve(workspace,'manifest.json'));
  const id=request.inputs?.candidate,row=typeof id==='string'?manifest.artifacts?.[id]:undefined;
  if(!row||row.status!=='verified'||row.kind!=='external_pgac_candidate')throw new Error('Select a verified inactive external_pgac_candidate');
  const root=await realpath(resolve(workspace,row.path)),metadataPath=await realpath(resolve(root,'artifact.json'));
  if(!inside(workspace,root)||!inside(root,metadataPath))throw new Error('PGAC candidate escapes its workspace');
  const metadataBytes=await readFile(metadataPath);
  if(row.metadata_file_sha256&&sha(metadataBytes)!==row.metadata_file_sha256)throw new Error('PGAC candidate metadata changed');
  const metadata=JSON.parse(metadataBytes.toString('utf8'));
  if(metadata.id!==id||metadata.kind!=='external_pgac_candidate')throw new Error('PGAC candidate identity differs');
  const descriptorPath=await realpath(resolve(root,'policy.json'));
  if(!inside(root,descriptorPath)||!metadata.files?.['policy.json'])throw new Error('PGAC candidate has no confined source descriptor');
  const bytes=await readFile(descriptorPath);if(sha(bytes)!==metadata.files['policy.json'])throw new Error('PGAC candidate source descriptor changed');
  const descriptor=JSON.parse(bytes.toString('utf8')),baseline=descriptor.imitation_policy;
  if(typeof baseline!=='string'||!metadata.parents?.some(parent=>parent.id===baseline))throw new Error('PGAC candidate must bind its actual imitation baseline parent');
  return requiresMatlabPolicyEvaluation({...request,inputs:{policy:baseline}});
}

async function requiresMatlabPolicyEvaluation(request){
  const workspace=await realpath(request.workspace),manifest=await json(resolve(workspace,'manifest.json'));
  async function select(id,kind,file){
    const row=typeof id==='string'?manifest.artifacts?.[id]:undefined;
    if(!row||row.status!=='verified'||row.kind!==kind)throw new Error(`Select a verified ${kind} artifact`);
    const root=await realpath(resolve(workspace,row.path)),metaPath=await realpath(resolve(root,'artifact.json'));
    if(!inside(workspace,root)||!inside(root,metaPath))throw new Error('Policy evaluation source escapes its artifact');
    const bytes=await readFile(metaPath);if(row.metadata_file_sha256&&sha(bytes)!==row.metadata_file_sha256)throw new Error('Policy evaluation metadata changed');
    const metadata=JSON.parse(bytes.toString('utf8'));
    if(metadata.id!==id||metadata.kind!==kind)throw new Error('Policy evaluation source identity is inconsistent');
    const path=await realpath(resolve(root,file));if(!inside(root,path)||!metadata.files?.[file])throw new Error('Policy evaluation source descriptor is not confined');
    const content=await readFile(path);if(sha(content)!==metadata.files[file])throw new Error('Policy evaluation source descriptor changed');
    return {metadata,descriptor:JSON.parse(content.toString('utf8'))};
  }
  const policy=await select(request.inputs?.policy,'external_imitation_policy','policy.json');
  const datasetId=policy.descriptor.external_expert_dataset;
  if(!policy.metadata.parents?.some(row=>row.id===datasetId))throw new Error('Policy evaluation dataset is not a bound policy ancestor');
  const dataset=await select(datasetId,'external_expert_dataset','dataset.json');
  const expert=dataset.metadata.source?.expert,plant=dataset.metadata.source?.plant;
  if(expert==null&&plant==null)return false;
  if(!dataset.metadata.parents?.some(row=>row.id===expert)||!dataset.metadata.parents?.some(row=>row.id===plant))throw new Error('Policy evaluation expert/Plant are not bound dataset ancestors');
  return requiresMatlabExpertBatch({...request,inputs:{expert,plant}});
}

async function requiresMatlabExpertBatch(request){
  const workspace=await realpath(request.workspace),manifest=await json(resolve(workspace,'manifest.json'));
  async function selected(id,label,descriptorName){
    const row=typeof id==='string'?manifest.artifacts?.[id]:undefined;
    if(!row||row.status!=='verified')throw new Error(`Select a verified ${label} artifact`);
    const root=await realpath(resolve(workspace,row.path));if(!inside(workspace,root))throw new Error(`Selected ${label} escapes its workspace`);
    const metadataPath=await realpath(resolve(root,'artifact.json'));if(!inside(root,metadataPath))throw new Error(`Selected ${label} metadata escapes its artifact`);
    const bytes=await readFile(metadataPath);if(row.metadata_file_sha256&&sha(bytes)!==row.metadata_file_sha256)throw new Error(`Selected ${label} metadata changed`);
    const metadata=JSON.parse(bytes.toString('utf8'));if(metadata.id!==id||metadata.kind!==row.kind)throw new Error(`Selected ${label} identity is inconsistent`);
    let descriptor;
    if(descriptorName){
      const path=await realpath(resolve(root,descriptorName));if(!inside(root,path)||!metadata.files?.[descriptorName])throw new Error(`Selected ${label} has no confined descriptor`);
      const content=await readFile(path);if(sha(content)!==metadata.files[descriptorName])throw new Error(`Selected ${label} descriptor changed`);
      descriptor=JSON.parse(content.toString('utf8'));
    }
    return{metadata,descriptor};
  }
  const expert=await selected(request.inputs?.expert,'external expert','expert.json');
  if(expert.metadata.kind!=='external_expert')throw new Error('Expert collection requires a registered external expert');
  const plantRow=manifest.artifacts?.[request.inputs?.plant];
  const plant=await selected(request.inputs?.plant,'plant',plantRow?.kind==='plant_model'?'plant.json':undefined);
  if(!['plant_model','nominal_model','reference_model','identified_model','corrected_model','learned_model'].includes(plant.metadata.kind))throw new Error('External expert collection requires an executable physical plant artifact');
  const expertBackend=expert.descriptor.adapter_spec?.backend;
  const plantBackend=plant.metadata.kind==='plant_model'?plant.descriptor.adapter_spec?.backend:'registered_torch';
  const pythonPlant=['python_callable','python_factory','registered_torch'].includes(plantBackend);
  const matlabPlant=['matlab_function','simulink'].includes(plantBackend);
  if(expertBackend==='python_callable'&&pythonPlant)return false;
  if(expertBackend==='matlab_function'&&matlabPlant)return true;
  throw new Error('This collection adapter requires colocated Python or MATLAB expert/plant backends; select a supported explicit bridge instead of per-sample MCP calls');
}

function marker(value, nonce) {
  const prefix = `CONTROL_MATLAB_BATCH_${nonce}:`;
  const lines = (value?.content ?? []).filter(item => item.type === 'text').flatMap(item => String(item.text).split(/\r?\n/))
    .filter(line => line.startsWith(prefix));
  if (lines.length !== 1) throw new Error('Fixed MATLAB batch did not return one nonce-bound terminal marker');
  return JSON.parse(lines[0].slice(prefix.length));
}

export function assertMatlabBatchAvailable(ctx, config, agent) {
  if (!config.matlabResource?.identityFile || !config.matlabResource?.directory) {
    throw new Error('MATLAB sampling requires the optional dedicated-session MATLAB overlay and shared resource identity');
  }
  if (!ctx.tools.definitions(agent).some(item => item.name === TOOL)) throw new Error('Native MATLAB MCP tool is unavailable in this Session');
}

/** Original request/receipt are never rewritten; only private worker env changes. */
export function matlabCollectionJob({ ctx, job, agent, slots, config, request, policy,
  onEvent = () => {}, confirmation, numerical = numericalJob }) {
  const abort = new AbortController();
  let active, cancelled = false, matlabDispatched = false, matlabReturned = false;
  const cancel = reason => {
    cancelled = true;
    abort.abort(new Error(reason ?? 'MATLAB collection cancelled'));
    active?.cancel(reason);
  };
  const done = (async () => {
    try {
      assertMatlabBatchAvailable(ctx, config, agent);
      const raw = confirmation?.rawRequest ?? JSON.stringify(request);
      if (raw !== JSON.stringify(request)) throw new Error('Confirmed request changed before MATLAB preparation');
      const requestSha = sha(raw), nonce = randomUUID();
      const workspace = await realpath(request.workspace);
      const parent = resolve(workspace, '.case-plans/matlab-exchanges');
      await mkdir(parent, { recursive: true });
      if (!inside(workspace, await realpath(parent))) throw new Error('MATLAB exchange directory escapes its workspace');
      const exchange = resolve(parent, nonce);
      const common = { CONTROL_MATLAB_EXCHANGE_DIR: exchange, CONTROL_MATLAB_REQUEST_SHA256: requestSha,
        CONTROL_MATLAB_NONCE: nonce };
      let prepared;
      job.updateProgress('Preparing the confirmed MATLAB sampling exchange');
      active = numerical({ ctx, job, slots, config, request, policy, confirmation,
        extraEnv: { ...common, CONTROL_MATLAB_PHASE: 'prepare' }, onEvent(event) {
          onEvent(event); if (event.event === 'result') prepared = event.result;
        } });
      if (cancelled) active.cancel();
      const preparation = await active.done;
      active = undefined;
      if (preparation.status !== 'completed') return preparation;
      abort.signal.throwIfAborted();
      prepared ??= preparation.result ? JSON.parse(preparation.result) : undefined;
      const manifestPath = resolve(exchange, 'manifest.json'), resultPath = resolve(exchange, 'result.json');
      if (prepared?.matlab_phase !== 'prepared' || prepared.status !== 'verified' ||
          resolve(prepared.manifest_path ?? '') !== manifestPath || resolve(prepared.exchange_dir ?? '') !== exchange ||
          resolve(prepared.result_path ?? '') !== resultPath || prepared.request_sha256 !== requestSha || prepared.nonce !== nonce) {
        throw new Error('Sandboxed prepare did not return this job\'s fixed MATLAB exchange');
      }
      const manifestBytes = await readFile(manifestPath);
      if (sha(manifestBytes) !== prepared.manifest_sha256) throw new Error('MATLAB exchange manifest changed after preparation');
      const manifest = JSON.parse(manifestBytes.toString('utf8'));
      if(request.operation==='collect_external_expert'&&(manifest.kind!=='control-matlab-expert-batch-v1'||prepared.batch_kind!=='external_expert')){
        throw new Error('External expert preparation did not identify the fixed expert batch protocol');
      }
      if(request.operation==='evaluate_external_imitation'&&(manifest.kind!=='control-matlab-policy-evaluation-v1'||prepared.batch_kind!=='external_policy_evaluation')){
        throw new Error('Policy evaluation preparation did not identify the fixed local comparison batch protocol');
      }
      if(request.operation==='evaluate_external_pgac'&&(manifest.kind!=='control-matlab-pgac-evaluation-v1'||prepared.batch_kind!=='external_pgac_evaluation')){
        throw new Error('PGAC evaluation preparation did not identify the fixed original-Plant comparison batch protocol');
      }
      if (manifest.nonce !== nonce || manifest.request_sha256 !== requestSha || manifest.plan?.plan_hash !== prepared.plan_hash) {
        throw new Error('MATLAB manifest is not bound to the confirmed request and sampling plan');
      }
      const helperFunction=request.operation==='evaluate_external_pgac'?'control_evaluate_pgac_batch':request.operation==='evaluate_external_imitation'?'control_harness_policy_evaluation':request.operation==='collect_external_expert'?'control_harness_expert_batch':'control_harness_collect_batch';
      const helper = resolve(config.codeRoot, `matlab/${helperFunction}.m`);
      const helperRoot = resolve(config.codeRoot, 'matlab');
      // The dedicated session can outlive a plugin update. Refresh only the
      // trusted changed PGAC helpers under this same guarded native call.
      const refresh=request.operation==='evaluate_external_pgac'?"clear('control_harness_expert_batch','control_evaluate_pgac_batch'); ":'';
      const code = `addpath(${quote(helperRoot)}); ${refresh}assert(strcmp(which('${helperFunction}'),${quote(helper)}),'ControlHarness:HelperIdentity','Unexpected batch helper'); fprintf('CONTROL_MATLAB_BATCH_${nonce}:%s\\n',jsonencode(${helperFunction}(${quote(manifestPath)},${quote(nonce)})));`;
      job.updateProgress('Collecting one bounded batch through the native MATLAB MCP tool');
      matlabDispatched = true;
      const batch = await ctx.tools.execute({ agent, signal: abort.signal,
        calls: [{ type: 'tool_call', id: `matlab-batch-${nonce}`, name: TOOL,
          arguments: JSON.stringify({ code, project_path: exchange }) }] });
      const result = batch.results[0]?.result;
      if (!result || result.isError) throw new Error('Native MATLAB batch was denied or failed; inspect the resource state before recovery');
      const certificate = marker(result.value, nonce);
      if (certificate.nonce !== nonce || resolve(certificate.result_path ?? '') !== resultPath || !['completed', 'partial', 'failed'].includes(certificate.status)) {
        throw new Error('Native MATLAB batch terminal marker does not identify this exchange');
      }
      // The native around hook has already verified dedicated identity/idle and
      // completed its lease before tools.execute returns a successful result.
      matlabReturned = true;
      abort.signal.throwIfAborted();
      const resultBytes = await readFile(resultPath);
      const resultDocument = JSON.parse(resultBytes.toString('utf8'));
      if (resultDocument.nonce !== nonce || resultDocument.request_sha256 !== requestSha ||
          resultDocument.plan_hash !== prepared.plan_hash || resultDocument.spec_hash !== manifest.spec_hash) {
        throw new Error('Native MATLAB result is not bound to this job\'s manifest');
      }
      job.updateProgress('Archiving the native MATLAB batch through the original confirmed worker request');
      active = numerical({ ctx, job, slots, config, request, policy, confirmation, onEvent,
        extraEnv: { ...common, CONTROL_MATLAB_PHASE: 'finalize', CONTROL_MATLAB_RESULT_PATH: resultPath,
          CONTROL_MATLAB_RESULT_SHA256: sha(resultBytes) } });
      if (cancelled) active.cancel();
      return await active.done;
    } catch (error) {
      return { status: cancelled ? 'killed' : 'failed', detail: error instanceof Error ? error.message : String(error),
        cleanupVerified: !matlabDispatched || matlabReturned };
    }
  })();
  return { cancel, done };
}
