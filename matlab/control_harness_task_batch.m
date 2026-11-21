function control_harness_task_batch(helper, manifestPath, manifestHash, nonce, sessionId, beforeNonce, afterNonce, expectedPid)
% 一个已核对的工程步骤只进入一次原生工具。先证明进程与空闲状态，
% 身份不符时不接触模型；数值函数返回后再复核，缺少终态证据时 Host 隔离资源。
trustedRoot = fileparts(mfilename('fullpath'));
allowed = {'control_harness_collect_batch','control_harness_expert_batch', ...
    'control_harness_policy_evaluation','control_evaluate_pgac_batch','control_harness_design_batch'};
assert(any(strcmp(helper,allowed)), 'ControlHarness:TaskHelper','Unknown task helper.');
assert(strcmp(getenv('CONTROL_HARNESS_SESSION_ID'),sessionId) && feature('getpid')==expectedPid, ...
    'ControlHarness:TaskSession','Dedicated MATLAB session identity changed.');
oldPath = path;
oldDir = pwd;
restore = onCleanup(@() restore_environment(oldPath,oldDir));
cd(trustedRoot);
addpath(trustedRoot);
dependencies = {'control_harness_terminal_probe', helper};
if any(strcmp(helper,{'control_evaluate_pgac_batch','control_harness_policy_evaluation'}))
    dependencies{end+1} = 'control_harness_expert_batch';
elseif strcmp(helper,'control_harness_design_batch')
    dependencies{end+1} = 'control_harness_design_check';
end
for index=1:numel(dependencies)
    name = dependencies{index};
    clear(name);
    assert(strcmp(which(name),fullfile(trustedRoot,[name '.m'])), ...
        'ControlHarness:TaskIdentity','Unexpected helper path.');
end
print_probe(beforeNonce,sessionId,expectedPid,trustedRoot);
assert(strcmp(char(java.io.File(manifestPath).getCanonicalPath()),manifestPath), ...
    'ControlHarness:TaskManifest','The task exchange redirects to a different location.');
assert(strcmp(file_hash(manifestPath),manifestHash),'ControlHarness:TaskManifest','Reviewed manifest changed.');
certificate = feval(helper,manifestPath,nonce);
% 不信任业务函数给出的 idle 字段。重新定位项目探针，并检查 MATLAB 管理的作业。
cd(trustedRoot);
assert(strcmp(which('control_harness_terminal_probe'),fullfile(trustedRoot,'control_harness_terminal_probe.m')), ...
    'ControlHarness:TaskIdentity','Unexpected terminal probe after task.');
print_probe(afterNonce,sessionId,expectedPid,trustedRoot);
fprintf('CONTROL_MATLAB_BATCH_%s:%s\n',nonce,jsonencode(certificate));
end

function print_probe(nonce,sessionId,pid,root)
result = control_harness_terminal_probe(nonce);
assert(strcmp(result.session_id,sessionId) && result.matlab_pid==pid && result.idle && ...
    strcmp([result.helper_file '.m'],fullfile(root,'control_harness_terminal_probe.m')), ...
    'ControlHarness:TaskIdle','Dedicated process identity or idle state could not be proved.');
fprintf('CONTROL_HARNESS_PROBE_%s:%s\n',nonce,jsonencode(result));
end

function restore_environment(oldPath,oldDir)
% 不在其他 MATLAB 会话中改 path 或当前目录，本进程中的变化在异常退出时也复原。
path(oldPath);
cd(oldDir);
end

function digest=file_hash(file)
handle=fopen(file,'rb');
assert(handle>=0,'ControlHarness:TaskManifest','Cannot read the prepared manifest.');
cleanup=onCleanup(@() fclose(handle));
hasher=javaMethod('getInstance','java.security.MessageDigest','SHA-256');
while true
    bytes=fread(handle,1024*1024,'*uint8');
    if isempty(bytes),break;end
    hasher.update(typecast(bytes,'int8'));
end
digest=lower(reshape(dec2hex(typecast(hasher.digest(),'uint8'),2).',1,[]));
end
