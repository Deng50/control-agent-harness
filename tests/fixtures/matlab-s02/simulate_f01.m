function result = simulate_f01(modelName, referencePath, outputPath, stateIds)
% Fixed S02 fixture runner. Runtime settings never alter the saved .slx file.
reference = jsondecode(fileread(referencePath));
ds = Simulink.SimulationData.Dataset;
ds{1} = timeseries(reference.input(:), reference.time(:));
ds{2} = timeseries(reference.disturbance(:), reference.time(:));
in = Simulink.SimulationInput(modelName);
in = in.setExternalInput(ds);
in = in.setModelParameter('StartTime', '0', ...
    'StopTime', sprintf('%.17g', reference.time(end)), ...
    'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
    'FixedStep', sprintf('%.17g', reference.dt), ...
    'SaveTime', 'on', 'TimeSaveName', 'tout', ...
    'SaveOutput', 'on', 'OutputSaveName', 'yout', ...
    'SaveState', 'on', 'StateSaveName', 'xout', ...
    'SaveFormat', 'Dataset', 'ReturnWorkspaceOutputs', 'on');
out = sim(in);
time = out.yout.getElement(1).Values.Time(:);
outputs = out.yout.getElement(1).Values.Data(:);
assert(out.xout.numElements == 2, 'ControlHarness:StateLogCount', ...
    'Expected exactly the two scalar Unit Delay state logs.');
expectedIds = {stateIds.x1, stateIds.x2};
expectedNames = {'State1', 'State2'};
expectedPaths = cell(1, 2);
for stateIndex = 1:2
    token = regexp(expectedIds{stateIndex}, '^blk_(\d+)$', 'tokens', 'once');
    assert(~isempty(token), 'ControlHarness:StateId', 'Missing fixed state block ID.');
    expectedPaths{stateIndex} = Simulink.ID.getFullName([modelName ':' token{1}]);
    assert(strcmp(get_param(expectedPaths{stateIndex}, 'BlockType'), 'UnitDelay') && ...
        strcmp(get_param(expectedPaths{stateIndex}, 'Name'), expectedNames{stateIndex}), ...
        'ControlHarness:StateIdentity', 'The returned ID does not identify the fixed scalar state.');
end
states = zeros(numel(time), 2);
seen = false(1, 2);
loggedStates = repmat(struct('block_id', '', 'block_path', '', 'dataset_index', 0), 1, 2);
for datasetIndex = 1:out.xout.numElements
    state = out.xout.getElement(datasetIndex);
    blockPath = state.BlockPath.getBlock(1);
    stateIndex = find(strcmp(expectedPaths, blockPath));
    assert(isscalar(stateIndex) && ~seen(stateIndex), ...
        'ControlHarness:StateLogIdentity', 'State log identity is missing, duplicated, or unexpected.');
    stateValues = state.Values.Data(:);
    stateTime = state.Values.Time(:);
    assert(numel(stateValues) == numel(time) && isequal(stateTime, time), ...
        'ControlHarness:StateLogTime', 'Scalar state log must align with every output sample.');
    states(:, stateIndex) = stateValues;
    seen(stateIndex) = true;
    loggedStates(stateIndex) = struct('block_id', expectedIds{stateIndex}, ...
        'block_path', blockPath, 'dataset_index', datasetIndex);
end
assert(all(seen), 'ControlHarness:StateLogIdentity', 'Both fixed states must be present.');
assert(isequal(size(states), [numel(time), 2]), ...
    'ControlHarness:StateLogShape', 'Expected two discrete states at every output sample.');
result = struct('time', time, 'outputs', outputs, 'states', states, ...
    'model_name', modelName, 'logged_states', loggedStates, ...
    'runtime', 'SimulationInput with ordered two-element Dataset; state columns mapped by verified SID');
fid = fopen(outputPath, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'ControlHarness:ResultWrite', 'Cannot write S02 simulation result.');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(result, PrettyPrint=true));
end
