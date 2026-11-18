function result = control_harness_design_check(bundle, casePath, parametersPath, nonce)
% Fixed S07 acceptance wrapper; all executable bundle members are Host-hashed.
oldPath = path;
oldDirectory = pwd;
restore = onCleanup(@() restore_environment(oldPath, oldDirectory));
clear_generated_functions();
addpath(bundle);
assert(strcmp(which('selfcheck_controller'), fullfile(bundle, 'selfcheck_controller.m')), ...
    'ControlHarness:DesignIdentity', 'Unexpected controller self-check function.');
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
