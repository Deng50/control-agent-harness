function certificate = control_harness_collect_batch(manifestPath, nonce)
% Trusted S04 entrypoint: one synchronous batch, never a model-supplied script.
% .m models implement f(x,u,d,dt,parameters); outputs are g(x,parameters) or C*x.
% SLX uses SimulationInput/Dataset and explicit scalar logged-state mappings.
% Bounds on SLX are detected after each bounded trajectory, not online.
manifest = jsondecode(fileread(manifestPath));
assert(manifest.schema_version == 1 && strcmp(manifest.kind, 'control-matlab-batch-v1') && ...
    strcmp(manifest.nonce, nonce), 'ControlHarness:Manifest', 'Invalid fixed MATLAB exchange.');
[exchange, ~, ~] = fileparts(manifestPath);
assert(strcmp(manifest.result_path, fullfile(exchange, 'result.json')), ...
    'ControlHarness:Manifest', 'Unexpected fixed result path.');
spec = manifest.spec;
plan = manifest.plan;
verify_sources(manifest);
interface = spec.interface;
nx = numel(interface.states); nu = numel(interface.controls);
ny = numel(interface.outputs); nd = numel(interface.disturbances);
oldPath = path; oldDirectory = pwd;
restore = onCleanup(@() restore_environment(oldPath, oldDirectory));
cd(exchange);
if strcmp(spec.backend, 'matlab_function')
    addpath(manifest.function_path_root);
    assert(strcmp(which(spec.entrypoint), manifest.source_file), ...
        'ControlHarness:FunctionIdentity', 'MATLAB entrypoint does not resolve to the frozen source.');
    if isfield(spec, 'output_entrypoint')
        outputFile = which(spec.output_entrypoint);
        assert(startsWith(outputFile, [manifest.source_root filesep]), ...
            'ControlHarness:FunctionIdentity', 'Output function is not in the frozen source bundle.');
    end
    stopDetection = 'stepwise_validation';
else
    addpath(fileparts(manifest.source_file));
    stopDetection = 'post_trajectory_validation';
end
maximum = numel(plan.trajectories) * plan.steps;
arrays = struct('x', zeros(maximum,nx), 'u', zeros(maximum,nu), ...
    'd', zeros(maximum,nd), 'next_x', zeros(maximum,nx), 'y', zeros(maximum,ny), ...
    'next_y', zeros(maximum,ny), 'trajectory_id', zeros(maximum,1), 'time', zeros(maximum,1));
count = 0; summaries = cell(0,1); failure = [];
for trajectoryIndex = 1:numel(plan.trajectories)
    trajectory = plan.trajectories(trajectoryIndex);
    summary = struct('trajectory_id',trajectory.trajectory_id,'completed_steps',0, ...
        'status','completed','stop_detection',stopDetection,'error_identifier','','error_message','');
    stepIndex = 0;
    attempted = struct('x',[],'u',[],'d',[],'next_x',[],'y',[],'next_y',[]);
    try
        if strcmp(spec.backend, 'simulink')
            [allStates, allOutputs] = run_slx(manifest, trajectory, nx,nu,ny,nd);
        else
            x = vector(trajectory.x0,nx,'initial state');
            y = output_value(spec,x,ny);
        end
        for k = 1:plan.steps
            stepIndex = k - 1;
            u = channel_row(trajectory.u,k,nu);
            d = channel_row(trajectory.d,k,nd);
            if strcmp(spec.backend, 'simulink')
                x = allStates(k,:).'; y = allOutputs(k,:).';
            end
            attempted = struct('x',x,'u',u,'d',d,'next_x',[],'y',y,'next_y',[]);
            check_point(x,u,d,y,interface,plan);
            if strcmp(spec.backend, 'matlab_function')
                next = next_state(spec,x,u,d,plan.dt,nx);
                attempted.next_x = next;
                nextY = output_value(spec,next,ny);
            else
                next = allStates(k+1,:).'; nextY = allOutputs(k+1,:).';
            end
            attempted.next_x = next; attempted.next_y = nextY;
            check_point(next,u,d,nextY,interface,plan);
            count = count + 1;
            arrays.x(count,:) = x.'; arrays.u(count,:) = u.'; arrays.d(count,:) = d.';
            arrays.next_x(count,:) = next.'; arrays.y(count,:) = y.'; arrays.next_y(count,:) = nextY.';
            arrays.trajectory_id(count) = trajectory.trajectory_id;
            arrays.time(count) = trajectory.t(k);
            summary.completed_steps = summary.completed_steps + 1;
            x = next; y = nextY;
        end
    catch problem
        if summary.completed_steps > 0
            summary.status = 'partial';
        else
            summary.status = 'failed';
        end
        summary.error_identifier = problem.identifier;
        summary.error_message = problem.message;
        nonfinite = struct;
        fields = fieldnames(attempted);
        for fieldIndex = 1:numel(fields)
            field = fields{fieldIndex};
            nonfinite.(field) = find(~isfinite(attempted.(field))).';
        end
        failure = struct('trajectory_id',trajectory.trajectory_id,'step_index',stepIndex, ...
            'time',trajectory.t(min(stepIndex+1,numel(trajectory.t))), ...
            'reason',problem.message,'error_identifier',problem.identifier, ...
            'attempted',attempted,'nonfinite_indices',nonfinite,'stop_detection',stopDetection);
        failure.error_report = getReport(problem,'extended','hyperlinks','off');
    end
    summaries{end+1,1} = summary; %#ok<AGROW> bounded by approved trajectory count
    if ~strcmp(summary.status,'completed')
        break; % Never start another trajectory after the first failure.
    end
end
fields = fieldnames(arrays);
for fieldIndex = 1:numel(fields)
    field = fields{fieldIndex}; arrays.(field) = arrays.(field)(1:count,:);
end
status = 'completed';
if ~isempty(failure)
    if count > 0, status = 'partial'; else, status = 'failed'; end
end
verify_sources(manifest);
result = struct('schema_version',1,'kind','control-matlab-batch-result-v1','nonce',nonce, ...
    'request_sha256',manifest.request_sha256,'plan_hash',plan.plan_hash, ...
    'model_version',manifest.model_version,'material_version',manifest.material_version, ...
    'spec_hash',manifest.spec_hash,'status',status,'arrays',arrays,'trajectories',{summaries}, ...
    'failure',failure,'stop_detection',stopDetection,'hardware_execution',false, ...
    'matlab_release',version('-release'),'matlab_pid',feature('getpid'), ...
    'session_id',getenv('CONTROL_HARNESS_SESSION_ID'),'helper_file',mfilename('fullpath'));
result.repeatability = 'same-input pair and cleared-entrypoint comparison at visited inputs; not a formal proof';
temporary = fullfile(exchange,['result-' nonce '.tmp']);
fid = fopen(temporary,'w','n','UTF-8');
assert(fid >= 0,'ControlHarness:ResultWrite','Cannot write native batch result.');
fileCleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true,ConvertInfAndNaN=true));
clear fileCleanup
movefile(temporary,manifest.result_path,'f');
certificate = struct('nonce',nonce,'status',status,'result_path',manifest.result_path, ...
    'samples',count,'stop_detection',stopDetection);
end

function next = next_state(spec,x,u,d,dt,nx)
if strcmp(spec.source_semantics,'transition')
    next = evaluate_function(spec.entrypoint,{x,u,d,dt,spec.parameters},nx,'next state');
    return
end
h = dt/spec.integration.substeps;
next = x;
for substep = 1:spec.integration.substeps
    k1 = evaluate_function(spec.entrypoint,{next,u,d,dt,spec.parameters},nx,'derivative');
    if strcmp(spec.integration.method,'euler')
        next = next + h*k1;
    else
        k2 = evaluate_function(spec.entrypoint,{next+h*k1/2,u,d,dt,spec.parameters},nx,'derivative');
        k3 = evaluate_function(spec.entrypoint,{next+h*k2/2,u,d,dt,spec.parameters},nx,'derivative');
        k4 = evaluate_function(spec.entrypoint,{next+h*k3,u,d,dt,spec.parameters},nx,'derivative');
        next = next + h*(k1+2*k2+2*k3+k4)/6;
    end
    next = vector(next,nx,'integrated state');
end
end

function y = output_value(spec,x,ny)
if isfield(spec,'output_entrypoint')
    y = evaluate_function(spec.output_entrypoint,{x,spec.parameters},ny,'output');
else
    y = spec.C*x;
end
y = vector(y,ny,'output');
end

function value = evaluate_function(name,args,width,label)
first = vector(feval(name,args{:}),width,label);
second = vector(feval(name,args{:}),width,label);
clear(name); % Only this declared numerical entrypoint, never the session.
fresh = vector(feval(name,args{:}),width,label);
assert(isequal(first,second) && isequal(first,fresh),'ControlHarness:HiddenMemory', ...
    'Declared numerical function has hidden memory/nondeterminism at a visited input.');
value = first;
end

function value = vector(value,width,label)
assert(isnumeric(value) && isreal(value) && numel(value) == width && ...
    (isempty(value) || isvector(value)), 'ControlHarness:VectorShape', ...
    '%s must be the explicitly declared real vector.',label);
value = double(value(:));
assert(all(isfinite(value)), 'ControlHarness:Nonfinite', '%s contains NaN/Inf.',label);
end

function value = channel_row(data,index,width)
if width == 0
    value = zeros(0,1);
else
    value = vector(data(index,:),width,'approved input');
end
end

function check_point(x,u,d,y,interface,plan)
groups = {'states','controls','disturbances','outputs'};
values = {x,u,d,y};
for groupIndex = 1:numel(groups)
    variables = interface.(groups{groupIndex}); value = values{groupIndex};
    assert(all(isfinite(value)), 'ControlHarness:Nonfinite','Nonfinite state/input/output.');
    for index = 1:numel(variables)
        assert(value(index) >= variables(index).lower-1e-12 && value(index) <= variables(index).upper+1e-12, ...
            'ControlHarness:Bounds','%s variable %s is outside its declared bounds.',groups{groupIndex},variables(index).name);
    end
end
for index = 1:numel(plan.coupled_constraints)
    constraint = plan.coupled_constraints(index);
    value = constraint.x(:).'*x + constraint.u(:).'*u + constraint.d(:).'*d;
    assert(isfinite(value),'ControlHarness:NonfiniteConstraint','Coupled constraint sum is nonfinite.');
    if ~isempty(constraint.lower)
        assert(value >= constraint.lower-1e-12,'ControlHarness:CoupledConstraint', ...
            'Coupled constraint %s violates its lower bound.',constraint.name);
    end
    if ~isempty(constraint.upper)
        assert(value <= constraint.upper+1e-12,'ControlHarness:CoupledConstraint', ...
            'Coupled constraint %s violates its upper bound.',constraint.name);
    end
end
end

function [states,outputs] = run_slx(manifest,trajectory,nx,nu,ny,nd)
spec = manifest.spec; mapping = spec.simulink; plan = manifest.plan;
[modelDirectory,modelName,~] = fileparts(manifest.source_file);
assert(~bdIsLoaded(modelName),'ControlHarness:ModelAlreadyLoaded', ...
    'Refusing to replace/close a pre-existing model with this basename.');
% SimulationInput opens the model for simulation; no persistent set_param,
% load_system, open_system or save_system changes the model configuration.
modelCleanup = onCleanup(@() close_owned_model(modelName));
in = Simulink.SimulationInput(modelName);
for index = 1:nx
    in = in.setVariable(mapping.states(index).initial_condition_variable,trajectory.x0(index),Workspace=modelName);
end
in = in.setVariable(mapping.sample_time_variable,plan.dt,Workspace=modelName);
names = fieldnames(spec.parameters);
for index = 1:numel(names)
    name = names{index};
    assert(~strcmp(name,mapping.sample_time_variable) && ...
        ~any(strcmp(name,{mapping.states.initial_condition_variable})), ...
        'ControlHarness:VariableConflict','Parameters cannot override approved dt/x0 variables.');
    in = in.setVariable(name,spec.parameters.(name),Workspace=modelName);
end
ds = Simulink.SimulationData.Dataset;
for index = 1:nu
    channel = mapping.controls(index);
    input = [trajectory.u(:,index); trajectory.u(end,index)];
    signal = timeseries(input,trajectory.t(:)); signal = setinterpmethod(signal,'zoh');
    ds{channel.port} = signal;
end
for index = 1:nd
    channel = mapping.disturbances(index);
    input = [trajectory.d(:,index); trajectory.d(end,index)];
    signal = timeseries(input,trajectory.t(:)); signal = setinterpmethod(signal,'zoh');
    ds{channel.port} = signal;
end
in = in.setExternalInput(ds);
in = in.setModelParameter('StartTime','0','StopTime',sprintf('%.17g',trajectory.t(end)), ...
    'SolverType','Fixed-step','Solver','FixedStepDiscrete','FixedStep',sprintf('%.17g',plan.dt), ...
    'SimulationMode','normal','LoadInitialState','off', ...
    'SaveTime','on','TimeSaveName','tout','SaveOutput','on','OutputSaveName','yout', ...
    'SaveState','on','StateSaveName','xout','SaveFormat','Dataset', ...
    'SignalInfNanChecking','error','ReturnWorkspaceOutputs','on');
out = sim(in);
% Fixed output properties were explicitly requested above. No speculative
% isfield/try-catch probing of SimulationOutput is used.
assert(strcmp(get_param(modelName,'FileName'),manifest.source_file), ...
    'ControlHarness:ModelIdentity','Simulation opened a different model file.');
assert(numel(find_system(modelName,'SearchDepth',1,'BlockType','Inport')) == nu+nd && ...
    numel(find_system(modelName,'SearchDepth',1,'BlockType','Outport')) == ny, ...
    'ControlHarness:PortCoverage','Every scalar root inport/outport must be explicitly mapped.');
inputs = [mapping.controls; mapping.disturbances];
for index = 1:numel(inputs)
    block = Simulink.ID.getFullName([modelName ':' inputs(index).sid]);
    assert(strcmp(get_param(block,'BlockType'),'Inport') && str2double(get_param(block,'Port')) == inputs(index).port, ...
        'ControlHarness:PortIdentity','Input SID/port does not match its declared role.');
end
assert(out.xout.numElements == nx && out.yout.numElements == ny, ...
    'ControlHarness:StateCoverage','Logged state/output count differs; hidden or vector states are unsupported.');
states = zeros(numel(trajectory.t),nx); outputs = zeros(numel(trajectory.t),ny);
seen = false(1,nx);
expectedPaths = cell(1,nx);
for index = 1:nx
    expectedPaths{index} = Simulink.ID.getFullName([modelName ':' mapping.states(index).sid]);
    assert(strcmp(get_param(expectedPaths{index},'BlockType'),'UnitDelay') && ...
        strcmp(strtrim(get_param(expectedPaths{index},'InitialCondition')),mapping.states(index).initial_condition_variable) && ...
        strcmp(strtrim(get_param(expectedPaths{index},'SampleTime')),mapping.sample_time_variable), ...
        'ControlHarness:StateIdentity','S04 requires the declared scalar Unit Delay and explicit x0/dt variables.');
end
for datasetIndex = 1:out.xout.numElements
    state = out.xout.getElement(datasetIndex);
    stateIndex = find(strcmp(expectedPaths,state.BlockPath.getBlock(1)));
    assert(isscalar(stateIndex) && ~seen(stateIndex),'ControlHarness:StateIdentity','Unexpected or duplicate logged-state path.');
    assert_grid(state.Values.Time,trajectory.t);
    data = state.Values.Data;
    assert(numel(data) == numel(trajectory.t),'ControlHarness:StateShape','Each declared state log must be scalar.');
    states(:,stateIndex) = data(:); seen(stateIndex) = true;
end
for index = 1:ny
    channel = mapping.outputs(index);
    expected = Simulink.ID.getFullName([modelName ':' channel.sid]);
    assert(strcmp(get_param(expected,'BlockType'),'Outport') && str2double(get_param(expected,'Port')) == channel.port, ...
        'ControlHarness:OutputIdentity','Output SID/port does not match its declaration.');
    output = out.yout.getElement(channel.port);
    assert(strcmp(output.BlockPath.getBlock(1),expected),'ControlHarness:OutputIdentity','Unexpected logged output path.');
    assert_grid(output.Values.Time,trajectory.t);
    data = output.Values.Data;
    assert(numel(data) == numel(trajectory.t),'ControlHarness:OutputShape','Each declared output must be scalar.');
    outputs(:,index) = data(:);
end
assert(all(seen) && max(abs(states(1,:).'-trajectory.x0(:))) <= 1e-10, ...
    'ControlHarness:InitialState','Simulation did not start from the approved complete x0.');
assert(strcmp(fileparts(get_param(modelName,'FileName')),modelDirectory), ...
    'ControlHarness:ModelIdentity','Simulation escaped the frozen model folder.');
end

function assert_grid(actual,expected)
assert(numel(actual) == numel(expected) && max(abs(actual(:)-expected(:))) <= 1e-10, ...
    'ControlHarness:TimeGrid','Simulation logs do not match the approved sample grid.');
end

function close_owned_model(modelName)
if bdIsLoaded(modelName), close_system(modelName,0); end
end

function restore_environment(oldPath,oldDirectory)
path(oldPath); cd(oldDirectory);
end

function verify_sources(manifest)
% Path keys in JSON objects are normalized by jsondecode; source_files retains
% the exact original relative path strings for this byte-integrity check.
assert(isfield(manifest,'source_files') && ~isempty(manifest.source_files), ...
    'ControlHarness:SourceIntegrity','No bound source-file inventory.');
for index = 1:numel(manifest.source_files)
    entry = manifest.source_files(index);
    source = fullfile(manifest.source_root,entry.relative_path);
    assert(strcmp(file_sha256(source),entry.sha256), ...
        'ControlHarness:SourceIntegrity','Frozen source bytes changed before/after execution.');
end
end

function digest = file_sha256(source)
fid = fopen(source,'rb');
assert(fid >= 0,'ControlHarness:SourceIntegrity','Cannot read frozen source for SHA256.');
cleanup = onCleanup(@() fclose(fid));
hasher = javaMethod('getInstance','java.security.MessageDigest','SHA-256');
while true
    bytes = fread(fid,1024*1024,'*uint8');
    if isempty(bytes), break; end
    hasher.update(typecast(bytes,'int8'));
end
bytes = typecast(hasher.digest(),'uint8');
digest = lower(reshape(dec2hex(bytes,2).',1,[]));
end
