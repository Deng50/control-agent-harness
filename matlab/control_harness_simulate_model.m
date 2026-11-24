function r = control_harness_simulate_model(model,target,sessionId,stopTime,fixedStep,solver,solverType)
% Compile and simulate a saved, workspace-bound model without writing the SLX.
r = struct('schema_version',1,'status','failed','stage','validation','verified',{{}}, ...
    'changes',{{}},'file_exists',false,'file_created_by_operation',false,'path',target,'model',model, ...
    'bytes',0,'matlab_pid',feature('getpid'),'release',version('-release'),'recoverable',false, ...
    'identifier','','reason','','causes',{{}});
if nargin < 4, stopTime = []; end
if nargin < 5, fixedStep = []; end
if nargin < 6, solver = ''; end
if nargin < 7, solverType = ''; end
try
    [directory,name,ext] = fileparts(target);
    assert(ischar(sessionId) && isrow(sessionId) && ~isempty(strtrim(sessionId)) && ...
        isvarname(model) && strcmp(name,model) && strcmp(ext,'.slx') && isfolder(directory) && ...
        strcmpi(char(java.io.File(target).getCanonicalPath()),target) && isfile(target), ...
        'ControlHarness:SimulationTarget','Select the existing, saved SLX in the current chat.');
    assert((isempty(stopTime) || validScalar(stopTime) && stopTime>=0) && ...
        (isempty(fixedStep) || validScalar(fixedStep) && fixedStep>0), ...
        'ControlHarness:SimulationInput','Invalid stop time or fixed step.');
    assert((isempty(solver) || (ischar(solver) && isrow(solver) && isvarname(solver))) && ...
        (isempty(solverType) || ismember(solverType,{'Fixed-step','Variable-step'})), ...
        'ControlHarness:SimulationInput','Invalid solver identifier or solver type.');
    file = dir(target); r.bytes = file.bytes; r.file_exists = true;
    r.verified{end+1} = r.stage;
    r.stage = 'load';
    private = getenv('CONTROL_MATLAB_SESSION_APPDATA');
    assert(isfolder(private),'ControlHarness:MissingSessionCache','No private dedicated runtime directory.');
    previous = Simulink.fileGenControl('getConfig');
    restore = onCleanup(@() Simulink.fileGenControl('setConfig','config',previous)); %#ok<NASGU>
    generated = fullfile(fileparts(private),'simulink-generated');
    Simulink.fileGenControl('set','CacheFolder',fullfile(generated,'cache'),'CodeGenFolder',fullfile(generated,'code'), ...
        'keepPreviousPath',true,'createDir',true);
    if bdIsLoaded(model)
        assert(strcmpi(get_param(model,'FileName'),target) && strcmp(get_param(model,'Dirty'),'off') && ...
            strcmp(get_param(model,'SimulationStatus'),'stopped'), ...
            'ControlHarness:SimulationTarget','A different, modified or busy memory model owns this name.');
        r.source = 'already_loaded_clean';
    else
        load_system(target);
        r.source = 'loaded_from_disk';
    end
    assert(strcmpi(get_param(model,'FileName'),target), ...
        'ControlHarness:SimulationTarget','The actual model filename differs from the requested SLX.');
    r.verified{end+1} = r.stage;
    r.stage = 'configure_solver';
    effectiveType = solverType;
    if isempty(effectiveType), effectiveType = get_param(model,'SolverType'); end
    assert(~(strcmp(effectiveType,'Fixed-step') && strcmpi(solver,'auto')), ...
        'ControlHarness:SolverCombination','Fixed-step with Solver=auto is invalid in R2024b. Use FixedStepDiscrete for a discrete model.');
    compileSavedSettings = (isempty(solver) || strcmp(solver,get_param(model,'Solver'))) && ...
        (isempty(solverType) || strcmp(solverType,get_param(model,'SolverType'))) && ...
        (isempty(fixedStep) || isequal(fixedStep,str2double(get_param(model,'FixedStep'))));
    if compileSavedSettings
        r.stage = 'compile';
        set_param(model,'SimulationCommand','update');
        r.verified{end+1} = r.stage;
    end
    r.stage = 'simulate';
    in = Simulink.SimulationInput(model);
    in = in.setModelParameter('SaveTime','on','TimeSaveName','tout','ReturnWorkspaceOutputs','on');
    if ~isempty(solverType), in = in.setModelParameter('SolverType',solverType); end
    if ~isempty(solver), in = in.setModelParameter('Solver',solver); end
    if ~isempty(stopTime), in = in.setModelParameter('StopTime',mat2str(stopTime,17)); end
    if ~isempty(fixedStep), in = in.setModelParameter('FixedStep',mat2str(fixedStep,17)); end
    out = sim(in);
    if ~compileSavedSettings
        % sim compiled the diagram with the requested transient settings.
        r.verified{end+1} = 'compile';
    end
    time = out.get('tout');
    assert(isnumeric(time) && isvector(time) && ~isempty(time) && all(isfinite(time)), ...
        'ControlHarness:SimulationReadback','Simulink did not return a finite simulation time vector.');
    r.default_samples = numel(time); r.final_time = time(end);
    r.verified{end+1} = r.stage; r.status = 'simulated';
catch err
    [known,diagnostic] = control_harness_model_error(r.stage,err);
    if known && strcmp(r.stage,'simulate')
        % These engine diagnostics identify diagram/solver compilation failures.
        r.stage = 'compile';
    end
    r.identifier = err.identifier; r.reason = err.message; r.causes = diagnostic.causes;
    r.recoverable = known || ismember(err.identifier,{'ControlHarness:SimulationTarget','ControlHarness:SimulationInput','ControlHarness:SolverCombination'});
    r.file_exists = isfile(target);
end
end

function ok = validScalar(value)
ok = isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value);
end
