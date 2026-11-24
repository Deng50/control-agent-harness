function r = control_compile_regression(model,target,action)
% Dedicated acceptance only. All model edits belong to a newly created fixture.
[parent,name,extension] = fileparts(target);
assert(strcmp(name,model) && strcmp(extension,'.slx') && isfolder(parent));
cache = Simulink.fileGenControl('getConfig');
restore = onCleanup(@() Simulink.fileGenControl('setConfig','config',cache)); %#ok<NASGU>
Simulink.fileGenControl('set','CacheFolder',fullfile(parent,'fixture-cache'), ...
    'CodeGenFolder',fullfile(parent,'fixture-code'),'keepPreviousPath',true,'createDir',true);
r = struct('pid',feature('getpid'),'model',model,'action',action);
if strcmp(action,'create')
    assert(~bdIsLoaded(model) && ~isfile(target));
    new_system(model,'Model','ErrorIfShadowed');
    set_param(model,'SolverType','Fixed-step');
    try
        set_param(model,'Solver','auto');
        error('ControlHarness:MissingFixtureError','Solver=auto unexpectedly succeeded.');
    catch err
        r.solver = describe(err);
        r.solver_recognized_in_fixed_operation = control_harness_model_error('configure_solver',err);
        r.solver_recognized_in_other_operation = control_harness_model_error('simulate',err);
    end
    set_param(model,'Solver','FixedStepDiscrete','FixedStep','0.1','StopTime','0.5');
    add_block('built-in/Inport',[model '/force']);
    add_block('built-in/Inport',[model '/load_force'],'Port','2');
    add_block('built-in/Mux',[model '/Mux'],'Inputs','2');
    add_block('simulink/Discrete/Discrete State-Space',[model '/Plant'],'A','eye(2)', ...
        'B','[0;1]','C','eye(2)','D','[0;0]','InitialCondition','[0;0]','SampleTime','0.1');
    add_block('built-in/Outport',[model '/State']);
    add_line(model,'force/1','Mux/1'); add_line(model,'load_force/1','Mux/2');
    add_line(model,'Mux/1','Plant/1'); add_line(model,'Plant/1','State/1');
    save_system(model,target);
    try
        set_param(model,'SimulationCommand','update');
        error('ControlHarness:MissingFixtureError','Scalar Plant compiled with a 2-wide drive.');
    catch err
        r.width = describe(err);
        r.width_recognized = control_harness_model_error('compile',err);
        mixed = addCause(err,MException('ControlHarness:OpaqueCallback','Synthetic unclassified callback.'));
        r.mixed_tree_recognized = control_harness_model_error('compile',mixed);
    end
    close_system(model,0);
elseif strcmp(action,'continuous')
    assert(~bdIsLoaded(model) && ~isfile(target));
    new_system(model,'Model','ErrorIfShadowed');
    set_param(model,'SolverType','Fixed-step','Solver','FixedStepDiscrete','FixedStep','0.1');
    add_block('built-in/Constant',[model '/Constant']);
    add_block('built-in/Integrator',[model '/Integrator']);
    add_block('built-in/Outport',[model '/State']);
    add_line(model,'Constant/1','Integrator/1');
    add_line(model,'Integrator/1','State/1');
    save_system(model,target);
    try
        set_param(model,'SimulationCommand','update');
        error('ControlHarness:MissingFixtureError','Continuous model unexpectedly compiled with a discrete solver.');
    catch err
        r.saved_solver_error = describe(err);
    end
    close_system(model,0);
elseif strcmp(action,'fix')
    if ~bdIsLoaded(model), load_system(target); end
    assert(strcmpi(get_param(model,'FileName'),target) && strcmp(get_param(model,'Dirty'),'off'));
    set_param([model '/Plant'],'B','[0 0;1 1]','D','zeros(2)');
    save_system(model,target); close_system(model,0);
else
    error('ControlHarness:InvalidFixtureAction','Unexpected fixture operation.');
end
end

function d = describe(err)
d = struct('identifier',err.identifier,'message',err.message, ...
    'causes',{cellfun(@describe,err.cause,'UniformOutput',false)});
end
