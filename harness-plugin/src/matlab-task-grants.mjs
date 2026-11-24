/** Process-local MATLAB capability. Only the Host's projected native command may issue it. */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, realpath } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { requiresMatlabBatch, taskHelpers } from './matlab-job.mjs';

const readJson = path => readFile(path,'utf8').then(JSON.parse);
const ordered = value => Array.isArray(value) ? value.map(ordered) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, ordered(value[key])])) : value;
const digest = value => createHash('sha256').update(JSON.stringify(ordered(value))).digest('hex');

async function fingerprint(file) {
  if (!file || !isAbsolute(file)) throw new Error('缺少已绑定的 MATLAB / MCP 程序绝对路径，请从专用 MATLAB 启动入口进入。');
  const path = await realpath(file), hash = createHash('sha256');
  for await (const bytes of createReadStream(path)) hash.update(bytes);
  return {path,sha256:hash.digest('hex')};
}

export class MatlabTaskGrants {
  constructor(config,{now = Date.now, env = process.env} = {}) {
    this.config = config;
    this.now = now;
    this.env = env;
    this.entries = new Map();
    this.commands = new Map();
  }

  async review(request,agent,intent_id) {
    if (!await requiresMatlabBatch(request)) return null;
    const {matlabResource: resource,codeRoot} = this.config;
    if (!resource?.directory || !resource.identityFile) throw new Error('本操作需要专用 MATLAB 启动配置。');
    const workspace = await realpath(request.workspace);
    const [marker,state,manifest,matlab_program,mcp_program] = await Promise.all([
      readJson(resource.identityFile),readJson(resolve(resource.directory,'resource.json')),readJson(resolve(workspace,'manifest.json')),
      fingerprint(this.env.CONTROL_MATLAB_BOUND_EXECUTABLE),fingerprint(this.env.CONTROL_MATLAB_MCP_EXECUTABLE),
    ]);
    const binding = state.session;
    const pendingBinding = state.quarantine?.owner?.session;
    if ((pendingBinding && !isDeepStrictEqual(pendingBinding,binding)) ||
      marker.kind !== 'control-harness-dedicated-matlab-v1' || marker.status !== 'ready'
      || marker.schema_version !== 1 || !binding?.identity || !Number.isSafeInteger(binding.pid) || binding.pid <= 0
      || !/^[a-f0-9-]{36}$/i.test(binding.session_id ?? '') || binding.session_id !== marker.session_id
      || binding.pid !== marker.matlab_pid || await realpath(marker.project_root) !== await realpath(codeRoot)) {
      throw new Error('专用 MATLAB 进程身份不匹配，请核查专用资源。');
    }
    const settings = {...request.config};
    delete settings.plan_refs;
    if (request.operation === 'sample_model') delete settings.sampling_run;
    const read_inputs = {};
    for (const [key,id] of Object.entries(request.inputs ?? {})) {
      const row = manifest.artifacts?.[id];
      if (typeof id !== 'string' || row?.status !== 'verified' || !/^[a-f0-9]{64}$/.test(row.metadata_hash ?? '')) {
        throw new Error('MATLAB 批处理只允许读取本次已核对版本的工作区产物。');
      }
      read_inputs[key] = {id,metadata_hash:row.metadata_hash};
    }
    const helper = taskHelpers[request.operation];
    const dependencies = ['control_harness_task_batch','control_harness_terminal_probe',helper];
    if (['evaluate_external_imitation','evaluate_external_pgac'].includes(request.operation)) dependencies.push('control_harness_expert_batch');
    if (request.operation === 'check_matlab_design') dependencies.push('control_harness_design_check');
    const helpers = await Promise.all(dependencies.map(name => fingerprint(resolve(codeRoot,'matlab',name+'.m'))));
    return {schema_version:1, workspace,session_id:agent.id,intent_id,operation:request.operation,
      native_tool:'mcp__matlab__evaluate_matlab_code',helper,helpers,matlab_program,mcp_program,
      dedicated_session:binding,matlab_release:marker.matlab_release ?? null,read_inputs,write_directory:resolve(workspace,'.case-plans/matlab-exchanges'),
      request_sha256:digest({inputs:request.inputs,settings}),ttl_seconds:1800,dispatch_timeout_seconds:300,
      variants:request.operation === 'sample_model' ? (settings.trial ? ['trial','full'] : ['full']) : ['run']};
  }

  status(card) {
    const entry = this.entries.get(card.card_id);
    if (!entry) return {state:'pending',remaining:[]};
    const state = entry.revoked ? 'revoked' : this.now() >= entry.deadline ? 'expired' :
      entry.running ? 'running' : entry.remaining.length ? 'active' : 'consumed';
    return {state,remaining:[...entry.remaining],expires_at:new Date(entry.deadline).toISOString(),reason:entry.revoked,
      authorizing_command:{command_id:entry.source.command_id,seq:entry.source.seq}};
  }

  issue(card,source) {
    const scope = card.matlab_scope;
    if (!scope) return null;
    if (source?.kind !== 'native_user_command' || !source.command_id || !Number.isSafeInteger(source.seq)
        || scope.session_id !== card.session_id || scope.intent_id !== card.intent_id
        || scope.operation !== card.operation || scope.workspace !== card.workspace) throw new Error('MATLAB 任务授权需要本卡真实用户点击记录。');
    const eventKey = JSON.stringify([card.session_id,source.command_id]);
    const seen = this.commands.get(eventKey);
    if (seen && seen !== card.card_id) throw new Error('MATLAB 授权来源已用于另一张卡。');
    if (seen) return this.status(card);
    this.commands.set(eventKey,card.card_id);
    const status = this.status(card);
    if (['active','running'].includes(status.state)) return status;
    const preceding = this.entries.get(card.card_id);
    if (preceding?.running) throw new Error('已有 MATLAB 批次仍在结算，请等待资源状态。');
    for (const [id,older] of this.entries) {
      if (id !== card.card_id && older.scope.workspace === scope.workspace && older.scope.session_id === scope.session_id
        && older.scope.intent_id === scope.intent_id && !older.revoked) {
        older.revoked='superseded_card';
        older.abort.abort(new Error('MATLAB 任务范围已由新版授权卡替代。'));
      }
    }
    const entry = {scope,source,remaining:[...scope.variants],deadline:this.now()+scope.ttl_seconds*1000, abort:new AbortController()};
    this.entries.set(card.card_id,entry);
    return this.status(card);
  }

  async reserve({request,agent,card}) {
    const scope = card?.matlab_scope;
    if (!scope || scope.session_id !== agent.id || card.session_id !== agent.id || scope.workspace !== card.workspace) throw new Error('此 MATLAB 操作需要当前会话的结构化任务授权卡。');
    const entry = this.entries.get(card.card_id);
    if (!isDeepStrictEqual(entry?.scope,scope) || this.status(card).state !== 'active') {
      throw new Error('MATLAB 任务授权已失效、正在使用或范围已变化，请复核并点击当前卡片。');
    }
    let current;
    try { current=await this.review(request,agent,card.intent_id); }
    catch(error) {
      entry.revoked='scope_unavailable';
      entry.abort.abort(error);
      throw error;
    }
    if (!isDeepStrictEqual(current,scope)) {
      entry.revoked='scope_changed';
      entry.abort.abort(new Error('MATLAB 任务范围已变化。'));
      throw new Error('MATLAB 任务材料、设置、目录或专用程序身份已变化，请核对新卡片。');
    }
    // 并发请求可能在读取版本的等待期间已占用同一授权，最后一次检查必须是同步的。
    if (this.entries.get(card.card_id) !== entry || this.status(card).state !== 'active') throw new Error('MATLAB 授权正在使用或已过期。');
    const variant = request.operation === 'sample_model' ? (request.config?.sampling_run ?? (request.config?.trial ? 'trial':'full')) : 'run';
    const index = entry.remaining.indexOf(variant);
    if (index < 0) throw new Error('该 MATLAB 子步骤未授权或已消耗，请重新核对任务。');
    entry.remaining.splice(index,1);
    entry.running = true;
    let completed = false, started = false;
    const revoke = reason => {
      entry.revoked ??= reason;
      entry.abort.abort(new Error(reason));
    };
    const authority = this;
    return {scope, deadline:entry.deadline, signal:entry.abort.signal,
      audit:{card_id:card.card_id,command_id:entry.source.command_id,seq:entry.source.seq,variant,
        helper:scope.helper,session_id:scope.dedicated_session.session_id,matlab_pid:scope.dedicated_session.pid,
        scope_sha256:digest(scope),expires_at:new Date(entry.deadline).toISOString()},
      begin() {
        if (completed || started || entry.revoked) throw new Error('MATLAB 批任务已失效或已开始。');
        started=true;
      },
      async validate(binding) {
        if (completed || !entry.running || entry.revoked || authority.now() >= entry.deadline
            || !isDeepStrictEqual(binding,scope.dedicated_session)) throw new Error('MATLAB 排队期间授权失效或专用进程发生变化。');
        if (!isDeepStrictEqual(await currentScope(),scope)) throw new Error('MATLAB 执行前程序、助手、来源或目录已变化，请重新确认。');
      },
      revoke(reason) {
        revoke(reason);
        if (!started) { completed=true; entry.running=false; }
      },
      finish() { completed=true; entry.running=false; },
    };
    function currentScope() {return authority.review(request,agent,card.intent_id);}
  }
}
