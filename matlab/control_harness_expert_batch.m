function certificate = control_harness_expert_batch(manifestPath, nonce)
% Trusted S09 one-call external expert + declared MATLAB/flat-SLX plant batch.
% No Harness/LLM callback occurs in the local numerical control loop.
manifest = jsondecode(fileread(manifestPath));
assert(manifest.schema_version==1 && strcmp(manifest.kind,'control-matlab-expert-batch-v1') && strcmp(manifest.nonce,nonce), ...
    'ControlHarness:ExpertManifest','Invalid fixed external-expert exchange.');
exchange=fileparts(manifestPath);
assert(strcmp(manifest.result_path,fullfile(exchange,'result.json')),'ControlHarness:ExpertManifest','Unexpected fixed result path.');
expert=manifest.expert.spec; plant=manifest.plant.spec; plan=manifest.plan; interface=plant.interface;
nx=numel(interface.states);nu=numel(interface.controls);ny=numel(interface.outputs);nd=numel(interface.disturbances);no=numel(expert.observations);
coverage=[];coverageGuard=[];
if isfield(manifest,'coverage_domain')
    coverage=manifest.coverage_domain;
    assert(isstruct(coverage)&&isequal(sort(fieldnames(coverage)),sort({'x';'u';'d'}))&& ...
        isfield(manifest,'coverage_guard_hash')&&~isempty(regexp(manifest.coverage_guard_hash,'^[a-f0-9]{64}$','once')), ...
        'ControlHarness:CoverageManifest','Invalid optional per-subtick coverage contract.');
    coverageGuard=struct('schema_version',1,'enabled',true,'domain_hash',manifest.coverage_guard_hash, ...
        'before_checks',0,'after_checks',0,'passed_subticks',0,'violations',0,'status','passed');
end
verify_sources(manifest.expert);verify_sources(manifest.plant);
namesToClear={expert.entrypoint};if strcmp(plant.backend,'matlab_function'),namesToClear{end+1}=plant.entrypoint;if isfield(plant,'output_entrypoint'),namesToClear{end+1}=plant.output_entrypoint;end,end
oldPath=path;oldDirectory=pwd;restore=onCleanup(@() restore_environment(oldPath,oldDirectory,namesToClear));
for k=1:numel(namesToClear),clear(namesToClear{k});end
cd(exchange);addpath(manifest.expert.function_path_root);
assert(strcmp(which(expert.entrypoint),manifest.expert.source_file),'ControlHarness:ExpertIdentity','Expert resolves outside its bound snapshot.');
if strcmp(plant.backend,'matlab_function')
    addpath(manifest.plant.function_path_root);
    assert(strcmp(which(plant.entrypoint),manifest.plant.source_file),'ControlHarness:PlantIdentity','Plant resolves outside its bound snapshot.');
    if isfield(plant,'output_entrypoint'),assert(startsWith(which(plant.output_entrypoint),[manifest.plant.source_root filesep]),'ControlHarness:PlantIdentity','Output is outside its snapshot.');end
    stopDetection='stepwise_plant_tick_validation';
else
    addpath(fileparts(manifest.plant.source_file));stopDetection='post_control_interval_simulation_validation';
end
% Check installed numerical solver identity only when a source declares its use.
used={};for k=1:numel(manifest.expert.source_validation.files),used=[used;manifest.expert.source_validation.files(k).numerical_builtins(:)];end %#ok<AGROW>
for name={'quadprog','optimoptions'}
    if any(strcmp(used,name{1})),resolved=which(name{1});assert(~isempty(resolved)&&contains(lower(resolved),lower(matlabroot)), ...
        'ControlHarness:SolverIdentity','A declared numerical solver resolves outside the installed MATLAB tree.');end
end
maximum=sum([plan.trajectories.steps]);
arrays=struct('observation',zeros(maximum,no),'reference',zeros(maximum,ny),'d',zeros(maximum,nd), ...
    'previous_u',zeros(maximum,nu),'u',zeros(maximum,nu),'x',zeros(maximum,nx),'next_x',zeros(maximum,nx), ...
    'plant_y',zeros(maximum,ny),'next_y',zeros(maximum,ny),'trajectory_id',zeros(maximum,1),'time',zeros(maximum,1));
count=0;summaries=cell(0,1);diagnostics=cell(0,1);failure=[];diagnosticBytes=0;diagnosticLimit=8*1024*1024;
if isfield(plan,'resource_boundary'),diagnosticLimit=plan.resource_boundary.max_diagnostic_bytes;end
for trajectoryIndex=1:numel(plan.trajectories)
    trajectory=plan.trajectories(trajectoryIndex);memory=expert.initial_memory;check_memory(memory,0);
    previous=vector(trajectory.initial_action,nu,'initial applied action');x=vector(trajectory.x0,nx,'initial state');
    reference=vector(trajectory.reference,ny,'reference');d=vector(trajectory.disturbance,nd,'disturbance');
    summary=struct('trajectory_id',trajectory.trajectory_id,'completed_steps',0,'status','completed','stop_detection',stopDetection,'error_identifier','','error_message','');
    stepIndex=0;plantSubstep=0;attempted=struct('x',x,'u',[],'d',d,'next_x',[],'plant_y',[],'next_y',[],'expert_status','not_called');
    try
        if strcmp(plant.backend,'matlab_function'),y=output_value(plant,x,ny);
        else
            initial=struct('x0',x,'u',previous.','d',d.','t',0);
            segment=manifest.plant;segment.plan=struct('dt',interface.sample_time);
            [initialStates,initialOutputs]=run_slx(segment,initial,nx,nu,ny,nd); %#ok<ASGLU>
            y=initialOutputs(1,:).';
        end
        for k=1:trajectory.steps
            stepIndex=k-1;plantSubstep=0;check_point(x,previous,d,y,interface);observation=zeros(no,1);
            for j=1:no,item=expert.observations(j);if strcmp(item.source,'state'),observation(j)=x(item.index+1);else,observation(j)=y(item.index+1);end,end
            rArg=reference;if ~expert.reference_available,rArg=zeros(0,1);end
            dArg=d;if ~expert.disturbance_available,dArg=zeros(0,1);end
            pArg=previous;if ~expert.previous_action_available,pArg=zeros(0,1);end
            memoryBefore=memory;
            attempted=struct('x',x,'u',[],'d',d,'next_x',[],'plant_y',y,'next_y',[],'expert_status','not_returned');
            [action,memoryAfter,status]=expert_call(expert,{observation,rArg,dArg,pArg,memory,expert.parameters},nu);
            attempted=struct('x',x,'u',[],'d',d,'next_x',[],'plant_y',y,'next_y',[],'expert_status',status);
            assert(strcmp(status,'ok'),'ControlHarness:ExpertStatus','External expert did not report ok: %s',status);
            u=zeros(nu,1);for j=1:numel(expert.action_mapping),item=expert.action_mapping(j);u(item.plant_control_index+1)=action(item.action_index+1);end
            attempted.u=u;check_point(x,u,d,y,interface);
            if isfield(expert,'input_rate_bounds')
                delta=u-previous;rate=expert.input_rate_bounds;
                assert(all(delta>=rate(:,1)-1e-12 & delta<=rate(:,2)+1e-12),'ControlHarness:ExpertRate','Expert action increment violates its declared bound; no clamp/fallback.');
            end
            next=x;nextY=y;
            if strcmp(plant.backend,'matlab_function')
                for tick=1:plan.plant_substeps
                    plantSubstep=tick;
                    if isstruct(coverage)
                        coverageGuard=coverage_check(coverageGuard,next,u,d,coverage,false);
                        assert(coverageGuard.violations==0,'ControlHarness:CoverageDomain','Original Plant leaves qualified x/u/d coverage before a physical subtick.');
                    end
                    next=next_state(plant,next,u,d,interface.sample_time,nx);attempted.next_x=next;
                    if isstruct(coverage)
                        coverageGuard=coverage_check(coverageGuard,next,u,d,coverage,true);
                        assert(coverageGuard.violations==0,'ControlHarness:CoverageDomain','Original Plant leaves qualified x coverage after a physical subtick.');
                    end
                    nextY=output_value(plant,next,ny);attempted.next_y=nextY;check_point(next,u,d,nextY,interface);
                end
            else
                local=struct('x0',x,'u',u.','d',d.','t',(0:plan.plant_substeps).'*interface.sample_time);
                segment=manifest.plant;segment.plan=struct('dt',interface.sample_time);
                [states,outputs]=run_slx(segment,local,nx,nu,ny,nd);
                for tick=1:plan.plant_substeps
                    plantSubstep=tick;
                    if isstruct(coverage)
                        coverageGuard=coverage_check(coverageGuard,states(tick,:).',u,d,coverage,false);
                        assert(coverageGuard.violations==0,'ControlHarness:CoverageDomain','Original SLX logged state leaves qualified x/u/d coverage before a physical subtick.');
                    end
                    next=states(tick+1,:).';nextY=outputs(tick+1,:).';attempted.next_x=next;attempted.next_y=nextY;
                    if isstruct(coverage)
                        coverageGuard=coverage_check(coverageGuard,next,u,d,coverage,true);
                        assert(coverageGuard.violations==0,'ControlHarness:CoverageDomain','Original SLX logged state leaves qualified x coverage after a physical subtick.');
                    end
                    check_point(next,u,d,nextY,interface);
                end
            end
            diagnostic=struct('trajectory_id',trajectory.trajectory_id,'step_index',stepIndex,'status',status, ...
                'memory_before',memoryBefore,'memory_after',memoryAfter,'expert_action',action.');
            bytes=numel(unicode2native(jsonencode(diagnostic),'UTF-8'));
            assert(diagnosticBytes+bytes<=diagnosticLimit,'ControlHarness:ExpertMemory','Single-batch diagnostic storage exceeds the approved common memory budget; collect in batches.');
            diagnosticBytes=diagnosticBytes+bytes;
            count=count+1;arrays.observation(count,:)=observation.';arrays.reference(count,:)=reference.';arrays.d(count,:)=d.';
            arrays.previous_u(count,:)=previous.';arrays.u(count,:)=u.';arrays.x(count,:)=x.';arrays.next_x(count,:)=next.';
            arrays.plant_y(count,:)=y.';arrays.next_y(count,:)=nextY.';arrays.trajectory_id(count)=trajectory.trajectory_id;arrays.time(count)=trajectory.time(k);
            diagnostics{count,1}=diagnostic; %#ok<AGROW>
            summary.completed_steps=summary.completed_steps+1;x=next;y=nextY;previous=u;memory=memoryAfter;
        end
    catch problem
        if summary.completed_steps>0,summary.status='partial';else,summary.status='failed';end
        summary.error_identifier=problem.identifier;summary.error_message=problem.message;
        failure=struct('trajectory_id',trajectory.trajectory_id,'step_index',stepIndex,'plant_substep',plantSubstep, ...
            'time',trajectory.time(min(stepIndex+1,numel(trajectory.time))),'reason',problem.message,'error_identifier',problem.identifier, ...
            'attempted',attempted,'stop_detection',stopDetection,'error_report',getReport(problem,'extended','hyperlinks','off'));
    end
    summaries{end+1,1}=summary; %#ok<AGROW>
    if ~strcmp(summary.status,'completed'),break;end
end
names=fieldnames(arrays);for k=1:numel(names),arrays.(names{k})=arrays.(names{k})(1:count,:);end
status='completed';if ~isempty(failure),if count>0,status='partial';else,status='failed';end,end
verify_sources(manifest.expert);verify_sources(manifest.plant);
result=struct('schema_version',1,'kind','control-matlab-expert-batch-result-v1','nonce',nonce,'request_sha256',manifest.request_sha256, ...
    'spec_hash',manifest.spec_hash,'plan_hash',plan.plan_hash,'expert_version',manifest.expert_version,'expert_material_version',manifest.expert_material_version, ...
    'plant_version',manifest.plant_version,'plant_material_version',manifest.plant_material_version,'status',status,'arrays',arrays, ...
    'trajectories',{summaries},'diagnostics',{diagnostics},'failure',failure,'stop_detection',stopDetection, ...
    'hardware_execution',false,'matlab_release',version('-release'),'matlab_pid',feature('getpid'), ...
    'session_id',getenv('CONTROL_HARNESS_SESSION_ID'),'helper_file',mfilename('fullpath'));
result.repeatability='same-input pair and cleared declared entrypoint comparison at visited inputs; explicit memory reset per trajectory, not formal purity proof';
if isstruct(coverage),result.coverage_guard=coverageGuard;end
temporary=fullfile(exchange,['result-' nonce '.tmp']);fid=fopen(temporary,'w','n','UTF-8');
assert(fid>=0,'ControlHarness:ResultWrite','Cannot write expert result.');cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true,ConvertInfAndNaN=true));clear cleanup;movefile(temporary,manifest.result_path,'f');
certificate=struct('nonce',nonce,'status',status,'result_path',manifest.result_path,'samples',count,'stop_detection',stopDetection);
end

function guard = coverage_check(guard,x,u,d,domain,after)
if after,guard.after_checks=guard.after_checks+1;else,guard.before_checks=guard.before_checks+1;end
valid=coverage_contains(x,domain.x);
if ~after,valid=valid&&coverage_contains(u,domain.u)&&coverage_contains(d,domain.d);end
if ~valid,guard.violations=1;guard.status='failed';elseif after,guard.passed_subticks=guard.passed_subticks+1;end
end

function valid = coverage_contains(value,bounds)
value=value(:);
if isempty(value),valid=isempty(bounds);return;end
valid=isnumeric(bounds)&&isreal(bounds)&&isequal(size(bounds),[numel(value),2])&&all(isfinite(value))&&all(isfinite(bounds),'all') ...
    &&all(value>=bounds(:,1)&value<=bounds(:,2));
end

function [action,memory,status] = expert_call(spec,args,width)
[a,m,s]=feval(spec.entrypoint,args{:});s=status_text(s);
if ~strcmp(s,'ok'),action=a;memory=m;status=s;return;end % Failed solver status never applies even an empty/nonfinite action.
a=vector(a,width,'expert action');check_memory(m,0);
[b,n,t]=feval(spec.entrypoint,args{:});b=vector(b,width,'expert action');check_memory(n,0);t=status_text(t);
clear(spec.entrypoint); % Clear only the declared expert function, never the session.
[c,o,v]=feval(spec.entrypoint,args{:});c=vector(c,width,'expert action');check_memory(o,0);v=status_text(v);
assert(isequal(a,b)&&isequal(a,c)&&isequal(m,n)&&isequal(m,o)&&strcmp(s,t)&&strcmp(s,v), ...
    'ControlHarness:ExpertHiddenMemory','External expert depends on hidden memory/nondeterminism; expose it in explicit memory.');
action=a;memory=m;status=s;
end

function value = status_text(value)
assert((ischar(value)&&isrow(value))||(isstring(value)&&isscalar(value)),'ControlHarness:ExpertStatus','Expert status must be a scalar explicit string.');
value=char(value);assert(numel(value)<=1024,'ControlHarness:ExpertStatus','Expert status exceeds its diagnostic budget.');
end

function check_memory(value,depth)
assert(isstruct(value)&&isscalar(value),'ControlHarness:ExpertMemory','Expert memory must be a finite explicit JSON object.');
check_json(value,depth);
assert(numel(unicode2native(jsonencode(value),'UTF-8'))<=65536,'ControlHarness:ExpertMemory','Expert memory exceeds its 64KiB diagnostic limit.');
end

function check_json(value,depth)
assert(depth<=32,'ControlHarness:ExpertMemory','Expert memory nesting exceeds its diagnostic limit.');
if isstruct(value)
    assert(isscalar(value),'ControlHarness:ExpertMemory','Memory objects must be scalar.');names=fieldnames(value);
    for k=1:numel(names),check_json(value.(names{k}),depth+1);end
elseif iscell(value)
    assert(numel(value)<=65536,'ControlHarness:ExpertMemory','Memory array exceeds its limit.');for k=1:numel(value),check_json(value{k},depth+1);end
elseif isnumeric(value)
    assert(isreal(value)&&all(isfinite(value),'all')&&ndims(value)<=2&&numel(value)<=65536,'ControlHarness:ExpertMemory','Memory numeric values must be finite JSON arrays.');
else
    assert(islogical(value)||ischar(value)||(isstring(value)&&isscalar(value)),'ControlHarness:ExpertMemory','Opaque memory objects/functions are unsupported.');
end
end

function next = next_state(spec,x,u,d,dt,nx)
if strcmp(spec.source_semantics,'transition'),next=evaluate_function(spec.entrypoint,{x,u,d,dt,spec.parameters},nx,'next state');return;end
h=dt/spec.integration.substeps;next=x;
for k=1:spec.integration.substeps
    k1=evaluate_function(spec.entrypoint,{next,u,d,dt,spec.parameters},nx,'derivative');
    if strcmp(spec.integration.method,'euler'),next=next+h*k1;
    else
        k2=evaluate_function(spec.entrypoint,{next+h*k1/2,u,d,dt,spec.parameters},nx,'derivative');
        k3=evaluate_function(spec.entrypoint,{next+h*k2/2,u,d,dt,spec.parameters},nx,'derivative');
        k4=evaluate_function(spec.entrypoint,{next+h*k3,u,d,dt,spec.parameters},nx,'derivative');next=next+h*(k1+2*k2+2*k3+k4)/6;
    end
    next=vector(next,nx,'integrated state');
end
end

function y = output_value(spec,x,ny)
if isfield(spec,'output_entrypoint'),y=evaluate_function(spec.output_entrypoint,{x,spec.parameters},ny,'output');else,y=spec.C*x;end
y=vector(y,ny,'output');
end

function value = evaluate_function(name,args,width,label)
first=vector(feval(name,args{:}),width,label);second=vector(feval(name,args{:}),width,label);clear(name);fresh=vector(feval(name,args{:}),width,label);
assert(isequal(first,second)&&isequal(first,fresh),'ControlHarness:HiddenMemory','Declared Plant has hidden memory/nondeterminism.');value=first;
end

function value = vector(value,width,label)
assert(isnumeric(value)&&isreal(value)&&numel(value)==width&&(isempty(value)||isvector(value)),'ControlHarness:Shape','%s width/type mismatch.',label);
value=double(value(:));assert(all(isfinite(value)),'ControlHarness:Nonfinite','%s contains nonfinite values.',label);
end

function check_point(x,u,d,y,interface)
groups={'states','controls','disturbances','outputs'};values={x,u,d,y};
for k=1:numel(groups)
    variables=interface.(groups{k});value=values{k};assert(all(isfinite(value)),'ControlHarness:Nonfinite','Nonfinite physical sample.');
    for j=1:numel(variables),assert(value(j)>=variables(j).lower-1e-12&&value(j)<=variables(j).upper+1e-12,'ControlHarness:Bounds','%s %s is outside physical bounds.',groups{k},variables(j).name);end
end
end

function restore_environment(oldPath,oldDirectory,namesToClear)
for k=1:numel(namesToClear),clear(namesToClear{k});end
path(oldPath);cd(oldDirectory);
end

function verify_sources(source)
assert(strcmp(source.source_file,fullfile(source.source_root,source.spec.source_file)),'ControlHarness:SourceIntegrity','Entrypoint escaped its fixed snapshot.');
for k=1:numel(source.source_files)
    entry=source.source_files(k);assert(strcmp(file_sha256(fullfile(source.source_root,entry.relative_path)),entry.sha256),'ControlHarness:SourceIntegrity','Frozen source bytes changed.');
end
end

function digest = file_sha256(source)
fid=fopen(source,'rb');assert(fid>=0,'ControlHarness:SourceIntegrity','Cannot read frozen source.');cleanup=onCleanup(@() fclose(fid));
hasher=javaMethod('getInstance','java.security.MessageDigest','SHA-256');
while true,bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end;hasher.update(typecast(bytes,'int8'));end
digest=lower(reshape(dec2hex(typecast(hasher.digest(),'uint8'),2).',1,[]));
end

% Same explicit logged-state/SID validation as the established S04 boundary.
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
    input = repmat(trajectory.u(1,index),numel(trajectory.t),1);
    signal = timeseries(input,trajectory.t(:)); signal = setinterpmethod(signal,'zoh');
    ds{channel.port} = signal;
end
for index = 1:nd
    channel = mapping.disturbances(index);
    input = repmat(trajectory.d(1,index),numel(trajectory.t),1);
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

