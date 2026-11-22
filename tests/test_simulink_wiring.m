function test_simulink_wiring()
% 使用随机命名的独立内存模型，验证预检拒绝不产生写入、分支线读回以及重复调用的结果。
    project = fileparts(fileparts(mfilename('fullpath')));
    previousPath = path();
    restore = onCleanup(@() path(previousPath)); %#ok<NASGU>
    addpath(fullfile(project, 'matlab'));
    model = ['WiringRegression_' strrep(char(java.util.UUID.randomUUID()), '-', '')];
    new_system(model);
    cleanup = onCleanup(@() close_system(model, 0)); %#ok<NASGU>
    add_block('built-in/Constant', [model '/U']);
    add_block('built-in/Constant', [model '/D']);
    add_block('built-in/Mux', [model '/Forces'], 'Inputs', '2');
    add_block('simulink/Discrete/Discrete State-Space', [model '/Plant'], ...
        'A', '0.5', 'B', '[1 2]', 'C', '1', 'D', '[0 0]', 'SampleTime', '0.1');
    add_block('built-in/Terminator', [model '/Monitor']);
    add_block('built-in/Subsystem', [model '/Nested']);
    add_block('built-in/Constant', [model '/Nested/Source']);
    add_block('built-in/Terminator', [model '/Nested/Sink']);
    edge = @(s, sp, d, dp) struct('sourceBlock', s, 'sourcePort', sp, 'destinationBlock', d, 'destinationPort', dp);
    wire = @(edges) control_harness_wire_simulink(model, edges);
    u = edge('U', 1, 'Forces', 1);
    d = edge('D', 1, 'Forces', 2);
    ports = get_param([model '/Forces'], 'PortHandles');

    % 最后一条才有错。前面的合法线不能提前写入；返回实际端口数与向量输入提示。
    refused = wire([u, edge('D', 1, 'Plant', 2)]);
    assert(strcmp(refused.identifier, 'ControlHarness:WirePort') && contains(refused.reason, 'Mux'));
    assert(strcmp(refused.status, 'refused') && refused.created == 0 && get_param(ports.Inport(1), 'Line') == -1);
    refused = wire([u, edge('D', 1, 'Forces', 1)]);
    assert(strcmp(refused.identifier, 'ControlHarness:WirePlanConflict') && get_param(ports.Inport(1), 'Line') == -1);
    refused = wire([u, edge('U', 1, 'Nested/Sink', 1)]);
    assert(strcmp(refused.identifier, 'ControlHarness:WireScope') && get_param(ports.Inport(1), 'Line') == -1);
    refused = wire([u, edge('Monitor', 1, 'Forces', 2)]);
    assert(strcmp(refused.identifier, 'ControlHarness:WirePort') && get_param(ports.Inport(1), 'Line') == -1);

    % 模拟上次调用已接上一条线。新计划仅补齐剩余端口，并保留同一信号到监视器的分支。
    add_line(model, 'U/1', 'Forces/1');
    refused = wire([d, edge('D', 1, 'Forces', 1)]);
    assert(strcmp(refused.identifier, 'ControlHarness:WireOccupied') && get_param(ports.Inport(2), 'Line') == -1);
    monitor = edge('U', 1, 'Monitor', 1);
    nested = edge('Nested/Source', 1, 'Nested/Sink', 1);
    plan = [u, d, edge('Forces', 1, 'Plant', 1), monitor, monitor, nested];
    receipt = wire(plan);
    assert(strcmp(receipt.status, 'wired') && strcmp(receipt.stage, 'readback'), '%s', receipt.reason);
    assert(receipt.created == 4 && receipt.existing == 2);
    muxInput = receipt.connections{1};
    plantInput = receipt.connections{3};
    assert(muxInput.destinationPortCount == 2 && plantInput.destinationPortCount == 1 && plantInput.sourcePortCount == 1);
    assert(isempty(plantInput.sourceSignalWidth) && isempty(plantInput.destinationSignalWidth));
    assert(isequal(cellfun(@(v) v.action, receipt.connections, 'UniformOutput', false)', ...
        {'existing', 'created', 'created', 'created', 'existing', 'created'}));
    replay = wire(plan);
    assert(strcmp(replay.status, 'wired') && replay.created == 0 && replay.existing == numel(plan));
    payload = jsondecode(jsonencode(wire(nested)));
    assert(numel(payload.connections) == 1 && strcmp(payload.connections.destinationBlock, 'Nested/Sink'));
    % 测试代码显式编译该临时模型后再发起幂等读回，生产接线函数本身不会编译模型。
    feval(model, [], [], [], 'compile');
    terminate = onCleanup(@() feval(model, [], [], [], 'term'));
    compiled = wire(plan(3));
    observed = compiled.connections{1};
    assert(compiled.created == 0 && observed.sourcePortCount == 1 && observed.destinationPortCount == 1);
    assert(isequal(observed.sourceSignalWidth, 2) && isequal(observed.destinationSignalWidth, 2));
    clear terminate
    fprintf('PASS %s: all-plan preflight, occupied ports, existing branches, nested wiring and replay; one Plant port, unknown while editing, width 2 when compiled\n', version('-release'));
end
