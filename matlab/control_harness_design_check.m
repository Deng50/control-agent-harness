function result = control_harness_design_check(bundle, casePath, parametersPath, nonce)
% 生产单点检查和验收脚本共用此入口。执行前逐一核对函数实际路径，
% 异常退出也恢复进入本函数时的搜索路径和当前目录。
oldPath = path;
oldDirectory = pwd;
restore = onCleanup(@() restore_environment(oldPath, oldDirectory));
clear_generated_functions();
addpath(bundle);
names = {'selfcheck_controller','prediction_model','predict_step','mpc_cost','mpc_control'};
for index=1:numel(names)
    assert(strcmp(which(names{index}),fullfile(bundle,[names{index} '.m'])), ...
        'ControlHarness:DesignIdentity','Unexpected exported controller function.');
end
if isempty(parametersPath)
    result = selfcheck_controller(casePath);
else
    result = selfcheck_controller(casePath, parametersPath);
end
assert(isstruct(result) && isscalar(result), 'ControlHarness:DesignResult', ...
    'Controller self-check must return one explicit result structure.');
result.wrapper_nonce = nonce;
result.matlab_pid = feature('getpid');
result.optimization_installed = ~isempty(which('quadprog'));
result.optimization_licensed = logical(license('test', 'Optimization_Toolbox'));
result.matlab_release = version('-release');
installed = ver('optim');
result.optimization_toolbox_version = '';
if ~isempty(installed),result.optimization_toolbox_version = installed(1).Version;end
result.closed_loop = 'not_run';
end

function clear_generated_functions()
% Clear only the declared numerical test functions, never the whole session.
clear('prediction_model', 'predict_step', 'mpc_cost', 'mpc_control', 'selfcheck_controller');
end

function restore_environment(oldPath, oldDirectory)
clear_generated_functions();
path(oldPath);
cd(oldDirectory);
end
