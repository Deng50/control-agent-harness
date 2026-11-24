function control_replay_delivered_plant(directory)
% Cold-session oracle consumer. Read the actual State-Space block's discrete
% state log, including x(0); do not reuse the product comparison helper.
request = jsondecode(fileread(fullfile(directory,'verification-request.json')));
reference = jsondecode(fileread(fullfile(directory,'python-reference.json')));
[~,model] = fileparts(request.model);
assert(~bdIsLoaded(model),'Expected a new MATLAB verification process.');
assert(strcmp(version('-release'),'2024b'));
cd(directory);
Simulink.fileGenControl('set','CacheFolder',fullfile(directory,'cache'), ...
    'CodeGenFolder',fullfile(directory,'codegen'),'createDir',true);
load_system(request.model);
assert(strcmpi(get_param(model,'FileName'),request.model));
closeOwned = onCleanup(@() close_system(model,0)); %#ok<NASGU>
plant = find_system(model,'FindAll','on','Type','block','BlockType','DiscreteStateSpace');
assert(isscalar(plant),'Expected a single full-state discrete plant.');
plantPath = getfullname(plant);
ports = find_system(model,'SearchDepth',1,'BlockType','Inport');
indices = cellfun(@(p) str2double(get_param(p,'Port')),ports);
[~,order] = sort(indices);
ports = ports(order);
assert(isequal(sort(indices(:))',[1 2]));
inputNames = cellfun(@(p) get_param(p,'Name'),ports,'UniformOutput',false);
ws = get_param(model,'ModelWorkspace');
result = struct('schema_version',1,'status','running','matlab_pid',feature('getpid'), ...
    'release',version('-release'),'model',request.model,'input_names',{inputNames}, ...
    'source_sha256',reference.source_sha256,'parameters',reference.parameters,'sample_time',reference.sampleTime, ...
    'max_abs_error',0,'tolerance',reference.absTolerance,'trajectories',{{}});
if ws.hasVariable('SourceParameters'), result.embedded_parameters = ws.getVariable('SourceParameters'); end
for j=1:numel(reference.trajectories)
    ref = reference.trajectories(j);
    times = (0:size(ref.inputs,1))' * reference.sampleTime;
    dataset = Simulink.SimulationData.Dataset;
    for k=1:numel(ports)
        samples = [ref.inputs(:,k);ref.inputs(end,k)];
        dataset = dataset.addElement(setinterpmethod(timeseries(samples,times),'zoh'),inputNames{k});
    end
    in = Simulink.SimulationInput(model);
    in = in.setModelParameter('StopTime',num2str(times(end),17),'SaveState','on','StateSaveName','xout', ...
        'SaveFormat','Dataset','ReturnWorkspaceOutputs','on');
    if j>1
        in = in.setBlockParameter(plantPath,'InitialCondition',mat2str(ref.initialCondition(:),17));
    end
    in = in.setExternalInput(dataset);
    out = sim(in);
    assert(strcmpi(get_param(model,'FileName'),request.model));
    matched = false;
    for s=1:out.xout.numElements
        state = out.xout.getElement(s);
        if strcmp(state.BlockPath.getBlock(1),plantPath)
            assert(~matched);
            matched = true; ys=state.Values;
        end
    end
    assert(matched && isequal(size(ys.Time(:)),size(times)) && max(abs(ys.Time(:)-times))<1e-12);
    actual = zeros(size(ref.states));
    for k=1:numel(times)
        value = getdatasamples(ys,k);
        actual(k,:) = value(:)';
    end
    diff = abs(actual-ref.states);
    assert(all(isfinite(diff(:))));
    result.trajectories{end+1}=struct('default_initial_condition',j==1,'inputs',ref.inputs, ...
        'initial',actual(1,:),'time',times,'states',actual,'python_states',ref.states, ...
        'max_abs_error',max(diff(:)));
    result.max_abs_error = max(result.max_abs_error,max(diff(:)));
end
result.status='failed';
if result.max_abs_error<=result.tolerance, result.status='passed'; end
fid=fopen(fullfile(directory,'independent-simulation.json'),'w','n','UTF-8');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',jsonencode(result));
assert(strcmp(result.status,'passed'),'Numerical discrepancy %.17g exceeds %.17g.',result.max_abs_error,result.tolerance);
end
