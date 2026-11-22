function receipt = control_harness_wire_simulink(model, connections)
% 对已加载模型的普通 Simulink 信号线执行整批预检，按需补线并读回。
% sourcePort 和 destinationPort 都是从 1 开始的物理端口号，不是向量元素号。
% 每条线的两端必须位于同一层。已有相同连线原样保留，也不拆除冲突或悬空的线。
    receipt = struct('model', '', 'stage', 'preflight', 'status', 'refused', ...
        'reason', '', 'identifier', '', 'created', 0, 'existing', 0, 'connections', {{}});
    try
        model = textValue(model);
        receipt.model = model;
        assert(isvarname(model) && bdIsLoaded(model) && ...
            strcmp(get_param(model, 'BlockDiagramType'), 'model'), 'ControlHarness:WireModel', ...
            '请指定已经加载的 Simulink 模型名。');
        fields = {'sourceBlock', 'sourcePort', 'destinationBlock', 'destinationPort'};
        assert(isstruct(connections) && isvector(connections) && ~isempty(connections) && ...
            all(isfield(connections, fields)), 'ControlHarness:WirePlan', ...
            '至少需要一条含源块、输出端口、目的块、输入端口的连线。');
        n = numel(connections);
        sources = zeros(n, 1);
        destinations = zeros(n, 1);
        parents = cell(n, 1);
        create = false(n, 1);
        details = cell(n, 1);
        for i = 1:n
            edge = connections(i);
            [sources(i), parents{i}] = port(model, edge.sourceBlock, 'Outport', edge.sourcePort);
            [destinations(i), destinationParent] = port(model, edge.destinationBlock, 'Inport', edge.destinationPort);
            assert(strcmp(parents{i}, destinationParent), 'ControlHarness:WireScope', ...
                '第 %d 条线跨越子系统边界。请用该子系统的输入或输出端口分别在同一层接线。', i);
            earlier = find(destinations(1:i-1) == destinations(i), 1);
            assert(isempty(earlier) || sources(earlier) == sources(i), 'ControlHarness:WirePlanConflict', ...
                '第 %d 条线与前一条线要求连接同一输入 %s/%d，但指定的来源不同。', ...
                i, edge.destinationBlock, edge.destinationPort);

            % 区分未接线、已接同源、其他来源以及悬空线。只在预检全通过后调用 add_line。
            line = get_param(destinations(i), 'Line');
            if line == -1
                create(i) = isempty(earlier);
            else
                actual = sourceOfLine(line);
                assert(actual ~= -1, 'ControlHarness:WireDangling', ...
                    '%s/%d 已有一条没有来源的线，请先核查该线。', edge.destinationBlock, edge.destinationPort);
                if actual ~= sources(i)
                    error('ControlHarness:WireOccupied', '%s/%d 已连接到 %s/%d，保留原接线。', ...
                        edge.destinationBlock, edge.destinationPort, get_param(actual, 'Parent'), get_param(actual, 'PortNumber'));
                end
            end
            details{i} = struct('sourceBlock', edge.sourceBlock, 'sourcePort', edge.sourcePort, ...
                'destinationBlock', edge.destinationBlock, 'destinationPort', edge.destinationPort, 'action', 'existing');
        end
    catch exception
        % 只有上述只读预检主动给出的业务拒绝可以直接返回。意外的 MATLAB 异常交给资源门禁。
        if ~startsWith(exception.identifier, 'ControlHarness:Wire')
            rethrow(exception);
        end
        receipt.identifier = exception.identifier;
        receipt.reason = exception.message;
        return
    end

    % 使用已解析的真实端口句柄；从已接线的输出再接一个目的端时由 Simulink 创建分支。
    % 应用或读回异常直接抛出，使宿主处理不确定的终态。失败后可能已有部分新线，须重新预检。
    for i = 1:n
        if create(i)
            add_line(parents{i}, sources(i), destinations(i), 'autorouting', 'on');
            details{i}.action = 'created';
        end
    end
    % 重新按请求的块路径和端口号定位。接线若触发动态端口变化，旧句柄不能代表最终接口。
    hasCompiledState = any(strcmp(get_param(model, 'SimulationStatus'), {'compiled', 'paused', 'running'}));
    for i = 1:n
        edge = connections(i);
        [actualSource, ~, details{i}.sourcePortCount] = port(model, edge.sourceBlock, 'Outport', edge.sourcePort);
        [actualDestination, ~, details{i}.destinationPortCount] = port(model, edge.destinationBlock, 'Inport', edge.destinationPort);
        line = get_param(actualDestination, 'Line');
        assert(line ~= -1 && sourceOfLine(line) == actualSource, 'ControlHarness:WireReadback', ...
            '第 %d 条连线读回时，实际来源与计划不一致。请重新检查当前模型。', i);
        details{i}.sourceSignalWidth = compiledWidth(actualSource, hasCompiledState);
        details{i}.destinationSignalWidth = compiledWidth(actualDestination, hasCompiledState);
    end
    receipt.stage = 'readback';
    receipt.status = 'wired';
    receipt.created = nnz(create);
    receipt.existing = n - receipt.created;
    receipt.connections = details;
end

function [handle, parent, count] = port(model, relative, direction, index)
% 只在当前模型中解析真实块，对实际方向的 PortHandles 计数。无需编译、加载其他模型或求值 B。
    relative = textValue(relative);
    assert(~isempty(relative) && relative(1) ~= '/', 'ControlHarness:WirePath', '请提供相对于模型的块路径。');
    assert(isnumeric(index) && isreal(index) && isscalar(index) && isfinite(index) && ...
        index >= 1 && index == fix(index), 'ControlHarness:WirePort', '物理端口号必须是从 1 开始的整数。');
    block = getSimulinkBlockHandle([model '/' relative], false);
    assert(block ~= -1 && strcmp(get_param(block, 'Type'), 'block') && strcmp(getfullname(bdroot(block)), model), ...
        'ControlHarness:WireBlock', '模型 %s 中找不到块 %s。请先读取当前模型结构。', model, relative);
    parent = get_param(block, 'Parent');
    handles = get_param(block, 'PortHandles');
    active = handles.(direction);
    count = numel(active);
    if index > count
        hint = '';
        if strcmp(direction, 'Inport') && strcmp(get_param(block, 'BlockType'), 'DiscreteStateSpace')
            hint = 'B 的各列对应同一个输入端口上的向量分量，可用 Mux 按列顺序组合后接到端口 1。';
        end
        error('ControlHarness:WirePort', '块 %s 实有 %d 个 %s 端口，请求的物理端口 %d 不存在。%s', ...
            relative, count, direction, index, hint);
    end
    handle = active(index);
end

function width = compiledWidth(portHandle, hasCompiledState)
% 正整数表示 Simulink 当前编译所得的信号宽度；空数组表示未知。只读可用的编译元数据，
% 不为获取宽度而编译模型或求值块参数。单端口既可承载标量也可承载向量，不能用端口数推断宽度。
    width = [];
    if hasCompiledState
        value = get_param(portHandle, 'CompiledPortWidth');
        if isnumeric(value) && isscalar(value) && isfinite(value) && value > 0 && value == fix(value)
            width = value;
        end
    end
end

function source = sourceOfLine(line)
% 从目的端读到的可能是分支线句柄。必要时向上找主线，比较输出端口句柄而非线的句柄或名称。
    source = get_param(line, 'SrcPortHandle');
    while source == -1
        line = get_param(line, 'LineParent');
        if line == -1
            return
        end
        source = get_param(line, 'SrcPortHandle');
    end
end

function value = textValue(value)
    if isstring(value) && isscalar(value) && ~ismissing(value)
        value = char(value);
    end
    assert(ischar(value) && isrow(value), 'ControlHarness:WirePath', '模型名和块路径须为单条文本。');
end
