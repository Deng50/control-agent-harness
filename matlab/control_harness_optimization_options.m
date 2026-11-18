function result = control_harness_optimization_options(nonce)
% Fixed read-only acceptance probe in the dedicated Harness MATLAB session.
options = optimoptions('quadprog');
names = properties(options);
selected = {'Algorithm','Display','MaxIterations','OptimalityTolerance','ConstraintTolerance'};
values = struct();
for index = 1:numel(selected)
    key = selected{index};
    if any(strcmp(names,key))
        values.(key) = options.(key);
    end
end
result = struct('nonce',nonce,'release',version('-release'),'version',version, ...
    'matlab_pid',feature('getpid'),'optimization_licensed',logical(license('test','Optimization_Toolbox')), ...
    'solver','quadprog','option_names',{names},'selected_defaults',values, ...
    'solver_executed',false,'scope','Installed solver option inspection only');
end
