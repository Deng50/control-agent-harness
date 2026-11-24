function r = control_harness_build_zoh_plant(model, target, Ac, Bc, x0, dt, inputNames, referenceJson, sessionId, parametersJson)
% Build a state-output LTI plant with one scalar Inport per column of Bc.
% Only this operation's own new/partly built memory model may be edited.
% The Host binds target to the chat and arbitrates retry and resource ownership.
r = struct('schema_version',1,'status','failed','stage','validation','verified',{{}}, ...
    'changes',{{}},'file_exists',false,'path',target,'model',model,'bytes',0, ...
    'matlab_pid',feature('getpid'),'release',version('-release'),'recoverable',false, ...
    'identifier','','reason','','comparison',struct('compared',false));
try
    assert(ischar(sessionId) && isrow(sessionId) && ~isempty(strtrim(sessionId)), ...
        'ControlHarness:BuildTarget','The Host session identity is required.');
    owner = struct('path',target,'session_id',sessionId);
    gate = library.settingsLookup();
    assert(~gate.found || gate.gatePass, 'ControlHarness:BuildInput', ...
        'The configured Simulink library index for this project is incomplete.');
    [directory, name, ext] = fileparts(target);
    assert(isvarname(model) && strcmp(name,model) && strcmp(ext,'.slx') && isfolder(directory) ...
        && strcmpi(char(java.io.File(target).getCanonicalPath()),target), ...
        'ControlHarness:BuildTarget','Expected an absolute canonical path to the chat model.');
    n = size(Ac,1); m = size(Bc,2);
    assert(n >= 1 && m >= 1 && isnumeric(Ac) && isnumeric(Bc) && isnumeric(x0) && isnumeric(dt) ...
        && isreal(Ac) && isreal(Bc) && isreal(x0) && isreal(dt) ...
        && isequal(size(Ac),[n n]) && isequal(size(Bc),[n m]) && isvector(x0) ...
        && numel(x0)==n && isscalar(dt) && dt>0 && all(isfinite([Ac(:);Bc(:);x0(:);dt])), ...
        'ControlHarness:BuildInput','Check Ac, Bc, initial state and positive sample time.');
    inputNames = cellstr(inputNames);
    assert(numel(inputNames)==m && all(cellfun(@isvarname,inputNames)) && ...
        numel(unique(lower(string([inputNames(:);{'InputVector';'Plant';'State'}]))))==m+3, ...
        'ControlHarness:BuildInput','Each input column needs a unique Simulink port name.');
    reference = jsondecode(referenceJson);
    parameters = jsondecode(parametersJson);
    assert(isstruct(parameters) && isscalar(parameters), 'ControlHarness:BuildInput', ...
        'Source parameters must be an object of finite real scalars.');
    keys = fieldnames(parameters);
    for j=1:numel(keys)
        value = parameters.(keys{j});
        assert(isvarname(keys{j}) && isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value), ...
            'ControlHarness:BuildInput','Invalid scalar source parameter %s.',keys{j});
    end
    hasReference = isfield(reference,'trajectories');
    if hasReference
        validateReference(reference,n,m,dt);
    end
    % Compute entirely in the function workspace, then explicitly publish the
    % parameters used by the blocks into the model's embedded workspace.
    transition = expm([Ac Bc; zeros(m,n+m)] * dt);
    assert(all(isfinite(transition(:))), 'ControlHarness:BuildInput', ...
        'The requested ZOH transition is non-finite. Check the plant parameters and dt.');
    Ad = transition(1:n,1:n); Bd = transition(1:n,n+1:end);
    assert(~isfile(target), 'ControlHarness:BuildTarget', ...
        'A file already exists at the destination. Choose a fresh model name.');
    r.verified{end+1} = 'validation';
    r.stage = 'create';
    if bdIsLoaded(model)
        ws = get_param(model,'ModelWorkspace');
        assert(ws.hasVariable('ControlHarnessBuildTarget') && ...
            isequal(ws.getVariable('ControlHarnessBuildTarget'),owner) && isempty(get_param(model,'FileName')) && ...
            strcmp(get_param(model,'SimulationStatus'),'stopped'), ...
            'ControlHarness:BuildTarget','A different or already saved in-memory model owns this name.');
    else
        new_system(model,'Model','ErrorIfShadowed');
        r.changes{end+1} = 'created memory model';
        ws = get_param(model,'ModelWorkspace');
        ws.assignin('ControlHarnessBuildTarget',owner);
    end
    ws = get_param(model,'ModelWorkspace');
    assert(strcmp(ws.DataSource,'Model File'),'ControlHarness:BuildTarget','Expected an embedded model workspace.');
    % Keep Simulink's working files outside the helper path and user files.
    privateAppData = getenv('CONTROL_MATLAB_SESSION_APPDATA');
    assert(isfolder(privateAppData),'ControlHarness:MissingSessionCache','No dedicated runtime directory.');
    previousFolders = Simulink.fileGenControl('getConfig');
    restoreFolders = onCleanup(@() Simulink.fileGenControl('setConfig','config',previousFolders)); %#ok<NASGU>
    generated = fullfile(fileparts(privateAppData),'simulink-generated');
    Simulink.fileGenControl('set','CacheFolder',fullfile(generated,'cache'), ...
        'CodeGenFolder',fullfile(generated,'code'),'keepPreviousPath',true,'createDir',true);
    % Keep a readback of the unfinished graph before correcting it. Avoid
    % reading the base workspace or resolving any user-supplied expression.
    r.prior_blocks = find_system(model,'SearchDepth',1,'Type','block');
    previous = whos(ws);
    r.prior_model_workspace = {previous.name};
    for key = {'Ac','Bc','dt'}
        if ws.hasVariable(key{1}), r.prior_values.(key{1}) = ws.getVariable(key{1}); end
    end
    evalc('model_read(model,''root'',''0'')');
    blockNames = [inputNames(:); {'InputVector';'Plant';'State'}];
    blockTypes = [repmat({'Inport'},m,1); {'Mux';'DiscreteStateSpace';'Outport'}];
    assert(all(ismember(r.prior_blocks, strcat(model,'/',blockNames))), ...
        'ControlHarness:BuildTarget','The partial model contains blocks outside the controlled plan.');
    operations = {};
    for j = 1:numel(blockNames)
        if getSimulinkBlockHandle([model '/' blockNames{j}]) == -1
            operations{end+1} = struct('op','add_block','type',blockTypes{j},'name',blockNames{j}); %#ok<AGROW>
        end
    end
    if ~isempty(operations)
        r.changes{end+1} = 'block creation attempted; inspect actual graph on retry';
        ops = jsonencode(operations);
        editReport = evalc('model_edit(model,''root'',ops,''full'')');
        evalc('model_read(model,''root'',''0'')');
        evalc('model_check(model,''root'',''["unconnected_ports","unconnected_lines"]'')');
    else
        editReport = '';
    end
    blocks = zeros(numel(blockNames),1);
    for j = 1:numel(blockNames)
        blocks(j) = getSimulinkBlockHandle([model '/' blockNames{j}]);
        assert(blocks(j)~=-1 && strcmp(get_param(blocks(j),'BlockType'),blockTypes{j}), ...
            'ControlHarness:BuildBlocks','Missing or mismatched %s. Toolkit result: %s',blockNames{j},editReport);
    end
    r.verified{end+1} = 'create';
    r.stage = 'configure';
    r.changes{end+1} = 'model workspace and block configuration attempted';
    % Replace only the plant coefficients. Other variables in a recovered
    % partial model remain available for inspection.
    data = struct('Ac',Ac,'Bc',Bc,'Ad',Ad,'Bd',Bd,'Cd',eye(n),'Dd',zeros(n,m),'x0',x0(:),'dt',dt);
    % Ac and Bc were derived from the inspected source. Keep its parameter
    % snapshot with the generated coefficients. Refresh preexisting scalar
    % aliases from that exact snapshot without clearing unrelated variables.
    r.updated_parameters = struct();
    protected = [fieldnames(data); {'ControlHarnessBuildTarget';'SourceParameters'}];
    if ~isempty(keys)
        data.SourceParameters = parameters;
        for j=1:numel(keys)
            key = keys{j};
            if ~ismember(key,protected) && ws.hasVariable(key)
                old = ws.getVariable(key);
                if isa(old,'double') && isreal(old) && isscalar(old) && isfinite(old)
                    data.(key) = parameters.(key);
                    r.updated_parameters.(key) = struct('before',old,'after',data.(key));
                end
            end
        end
    end
    variables = fieldnames(data);
    for j=1:numel(variables), ws.assignin(variables{j},data.(variables{j})); end
    for j=1:m, set_param(blocks(j),'Port',num2str(j),'PortDimensions','1','Interpolate','off'); end
    set_param(blocks(m+1),'Inputs',num2str(m));
    set_param(blocks(m+2),'A','Ad','B','Bd','C','Cd','D','Dd','InitialCondition','x0','SampleTime','dt');
    r.stage = 'configure_solver';
    set_param(model,'SolverType','Fixed-step','Solver','FixedStepDiscrete','FixedStep',mat2str(dt,17), ...
        'StopTime',mat2str(10*dt,17),'SaveOutput','on','OutputSaveName','yout','SaveFormat','Dataset', ...
        'ReturnWorkspaceOutputs','on','LoadExternalInput','off');
    r.stage = 'configure';
    for j=1:m
        assert(str2double(get_param(blocks(j),'Port'))==j,'ControlHarness:BuildReadback','Wrong Inport order.');
    end
    r.verified{end+1} = 'configure';
    r.stage = 'wire';
    edge = @(from,to,port) struct('sourceBlock',from,'sourcePort',1,'destinationBlock',to,'destinationPort',port);
    for j=m:-1:1, edges(j) = edge(inputNames{j},'InputVector',j); end
    edges(end+1) = edge('InputVector','Plant',1);
    edges(end+1) = edge('Plant','State',1);
    r.changes{end+1} = 'wiring attempted; identical existing edges retained';
    wired = control_harness_wire_simulink(model,edges);
    if ~strcmp(wired.status,'wired'), error(wired.identifier,'%s',wired.reason); end
    r.verified{end+1} = 'wire';
    r.stage = 'compile';
    set_param(model,'SimulationCommand','update');
    r.diagram_update = 'verified';
    r.stage = 'save';
    % The construction ownership marker is transient. No chat path is needed
    % by the delivered model; a bound or pre-existing file is never overwritten.
    evalin(ws,'clear ControlHarnessBuildTarget');
    try
        saved = control_harness_save_slx(model,target);
        if ~strcmp(saved.status,'saved'), error('ControlHarness:BuildTarget','%s',saved.reason); end
    catch saveError
        ws.assignin('ControlHarnessBuildTarget',owner);
        rethrow(saveError);
    end
    r.changes{end+1} = 'saved file';
    r.bytes = saved.bytes; r.file_exists = true; r.verified{end+1} = 'save';
    r.stage = 'reload';
    close_system(model,0);
    assert(~bdIsLoaded(model),'ControlHarness:BuildReadback','The original memory model is still loaded.');
    load_system(target);
    ws = get_param(model,'ModelWorkspace');
    assert(strcmpi(get_param(model,'FileName'),target),'ControlHarness:BuildReadback','Reloaded the wrong file.');
    for j=1:numel(variables)
        assert(isequal(ws.getVariable(variables{j}),data.(variables{j})), ...
            'ControlHarness:BuildReadback','Reloaded model parameter %s differs.', variables{j});
    end
    if ~isempty(keys), r.reloaded_parameters = ws.getVariable('SourceParameters'); end
    r.verified{end+1} = 'reload';
    r.stage = 'simulate';
    % First run uses only embedded initial state and ordinary root Inports.
    in = Simulink.SimulationInput(model);
    out = sim(in);
    [time, states] = readStates(out,n);
    assert(numel(time)>=2 && time(1)==0 && max(abs(states(1,:)'-x0(:))) < ...
        1e-12*(1+norm(x0,inf)), 'ControlHarness:BuildReadback','Default simulation initial condition differs.');
    r.default_samples = numel(time);
    r.verified{end+1} = 'simulate';
    if hasReference
        r.stage = 'compare';
        r.comparison = compareTrajectories(model,inputNames,dt,reference,n);
        assert(r.comparison.max_abs_error <= reference.absTolerance, ...
            'ControlHarness:ReferenceMismatch','State sample error %.17g exceeds absolute tolerance %.17g.', ...
            r.comparison.max_abs_error, reference.absTolerance);
        r.verified{end+1} = 'compare';
    end
    r.status = 'simulated';
catch err
    r.identifier = err.identifier; r.reason = err.message;
    r.file_exists = isfile(target);
    if r.file_exists, file = dir(target); r.bytes = file.bytes; end
    % Unknown callbacks, runtime and IO exceptions never certify completion.
    [known,diagnostic] = control_harness_model_error(r.stage,err);
    r.causes = diagnostic.causes;
    r.recoverable = known || ismember(err.identifier,{'ControlHarness:BuildInput','ControlHarness:BuildTarget', ...
        'ControlHarness:BuildBlocks','ControlHarness:WirePort','ControlHarness:WireOccupied', ...
        'ControlHarness:ReferenceMismatch','ControlHarness:ReferenceShape'});
end
end

function validateReference(ref,n,m,dt)
assert(isfield(ref,'absTolerance') && isnumeric(ref.absTolerance) && isscalar(ref.absTolerance) && ...
    isfinite(ref.absTolerance) && ref.absTolerance>0 && (~isfield(ref,'sampleTime') || isequal(ref.sampleTime,dt)) && ...
    isstruct(ref.trajectories) && ~isempty(ref.trajectories) && ...
    all(isfield(ref.trajectories,{'initialCondition','inputs','states'})), ...
    'ControlHarness:ReferenceShape','Expected sampled Python state trajectories and a positive absolute tolerance.');
for j=1:numel(ref.trajectories)
    t = ref.trajectories(j);
    assert(isnumeric(t.inputs) && isnumeric(t.states) && isnumeric(t.initialCondition) && ...
        ismatrix(t.inputs) && size(t.inputs,1)>=1 && size(t.inputs,2)==m && isvector(t.initialCondition) && ...
        numel(t.initialCondition)==n && isequal(size(t.states),[size(t.inputs,1)+1 n]) && ...
        all(isfinite([t.inputs(:);t.states(:);t.initialCondition(:)])) && ...
        isequal(t.states(1,:)',t.initialCondition(:)), ...
        'ControlHarness:ReferenceShape','Trajectory %d must contain K inputs and K+1 full state samples.',j);
end
end

function report = compareTrajectories(model,names,dt,ref,n)
report = struct('compared',true,'trajectories',numel(ref.trajectories),'samples',0, ...
    'max_abs_error',0,'tolerance',ref.absTolerance);
for j=1:numel(ref.trajectories)
    t = ref.trajectories(j); time = (0:size(t.inputs,1))'*dt;
    dataset = Simulink.SimulationData.Dataset;
    for k=1:numel(names)
        values = [t.inputs(:,k);t.inputs(end,k)];
        series = timeseries(values,time);
        series = setinterpmethod(series,'zoh');
        dataset = dataset.addElement(series,names{k});
    end
    in = Simulink.SimulationInput(model);
    in = in.setExternalInput(dataset);
    in = in.setVariable('x0',t.initialCondition(:),'Workspace',model);
    in = in.setModelParameter('StopTime',mat2str(time(end),17));
    out = sim(in);
    [actualTime,actual] = readStates(out,n);
    assert(isequal(size(actualTime),size(time)) && max(abs(actualTime-time))<=32*eps(max(1,time(end))), ...
        'ControlHarness:ReferenceMismatch','Sample times differ on trajectory %d.',j);
    report.samples = report.samples + numel(time);
    report.max_abs_error = max(report.max_abs_error,max(abs(actual-t.states),[],'all'));
end
end

function [time,states] = readStates(out,n)
trace = out.get('yout').getElement(1).Values;
time = trace.Time(:);
states = zeros(numel(time),n);
for i=1:numel(time)
    sample = getdatasamples(trace,i);
    assert(numel(sample)==n && all(isfinite(sample(:))), ...
        'ControlHarness:BuildReadback','State output is non-finite or has the wrong width.');
    states(i,:) = sample(:)';
end
end
