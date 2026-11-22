function receipt = control_harness_configure_discrete_plant(model, blockPath, A, B, C, D, initialCondition, sampleTime)
% 在当前已加载的模型中，一次设置离散状态空间块的六个耦合参数，并从同一块读回。
% 输入只接受数值，A/B/C/D 的行列分别对应状态、输入和输出；初态统一写成列向量。
% 尚未接线的模型也能使用此入口。整图更新由调用方在接线完成后单独执行。
    receipt = struct('status', 'refused', 'stage', 'validation', 'parameters_applied', false, ...
        'reason', '', 'identifier', '', 'model', '', 'block_path', '', 'diagram_update', 'not_run');
    try
        model = scalarText(model);
        blockPath = scalarText(blockPath);
        assert(isvarname(model) && bdIsLoaded(model), 'ControlHarness:PlantModel', ...
            '请指定已经加载的顶层模型名。');
        assert(strcmp(get_param(model, 'BlockDiagramType'), 'model'), 'ControlHarness:PlantModel', ...
            '目标必须是已加载的模型。');
        assert(~isempty(blockPath) && blockPath(1) ~= '/', 'ControlHarness:PlantPath', ...
            '块路径必须相对于该模型，例如 Plant 或 Dynamics/Plant。');
        receipt.model = model;
        receipt.block_path = [model '/' blockPath];
        block = get_param(receipt.block_path, 'Handle');
        assert(strcmp(getfullname(bdroot(block)), model) && ...
            strcmp(get_param(block, 'BlockType'), 'DiscreteStateSpace'), 'ControlHarness:PlantBlockType', ...
            '目标必须是该模型中的 Discrete State-Space 块。');

        names = {'A', 'B', 'C', 'D', 'InitialCondition', 'SampleTime'};
        values = {A, B, C, D, initialCondition, sampleTime};
        for i = 1:numel(values)
            value = values{i};
            assert(isnumeric(value) && isreal(value) && ismatrix(value) && ~isempty(value) && ...
                ~issparse(value) && all(isfinite(value(:))), 'ControlHarness:PlantNumeric', ...
                '参数 %s 必须是非空的有限实数矩阵。', names{i});
            values{i} = double(value);
        end
        n = size(A, 1);
        m = size(B, 2);
        r = size(C, 1);
        assert(size(A, 2) == n, 'ControlHarness:PlantShapeA', 'A 必须为方阵。');
        assert(size(B, 1) == n, 'ControlHarness:PlantShapeB', 'B 应有 %d 行，每行对应一个状态。', n);
        assert(size(C, 2) == n, 'ControlHarness:PlantShapeC', 'C 应有 %d 列，每列对应一个状态。', n);
        assert(isequal(size(D), [r m]), 'ControlHarness:PlantShapeD', ...
            'D 尺寸应为 %d×%d（输出数×输入数）。', r, m);
        assert(isvector(initialCondition) && numel(initialCondition) == n, 'ControlHarness:PlantInitialCondition', ...
            'InitialCondition 必须含有 %d 个初始状态。', n);
        assert(isscalar(sampleTime) && sampleTime > 0, 'ControlHarness:PlantSampleTime', ...
            'SampleTime 必须是大于零的有限采样周期（秒）。');
        values{5} = values{5}(:);

        % 由数值生成 17 位有效数字的 MATLAB 字面量，不从工作区求值任何表达式。
        literals = cellfun(@(value) mat2str(value, 17), values, 'UniformOutput', false);
        pairs = [names; literals];
        receipt.stage = 'apply';
        % set_param 同次校验所有名值对；逐个写 A、B 等会让默认标量暂时尺寸冲突。
        set_param(block, pairs{:});
        receipt.parameters_applied = true;
        receipt.stage = 'readback';
        for i = 1:numel(names)
            actual = get_param(block, names{i});
            assert(ischar(actual) && strcmp(strtrim(actual), literals{i}), 'ControlHarness:PlantReadback', ...
                '参数 %s 的实际存储值与写入的数值字面量不一致。', names{i});
        end
    catch exception
        % apply 阶段可能来自模型回调或非参数异常；保留原生 MATLAB 错误，
        % 让持有租约的 Host 判定是否可执行身份及空闲复核，不将未知异常伪装成业务拒绝。
        if strcmp(receipt.stage, 'apply')
            rethrow(exception);
        end
        receipt.identifier = exception.identifier;
        receipt.reason = exception.message;
        if receipt.parameters_applied
            % 写入已经成功，读回失败不代表更改已经回滚。
            receipt.status = 'unverified';
        end
        return
    end

    % 上面已逐项核实块上保存的字面量。维度字段可用于还原 JSON 折叠的行/列向量。
    receipt.status = 'configured';
    receipt.states = n;
    receipt.inputs = m;
    receipt.outputs = r;
    receipt.A = values{1};
    receipt.B = values{2};
    receipt.C = values{3};
    receipt.D = values{4};
    receipt.initial_condition = values{5};
    receipt.sample_time = values{6};
end

function text = scalarText(input)
% 仅接受一个标量字符串或一行字符，避免多个模型/块被隐式拼接。
    if isstring(input) && isscalar(input) && ~ismissing(input)
        input = char(input);
    end
    assert(ischar(input) && isrow(input), 'ControlHarness:PlantPath', ...
        '模型名和块路径均应为单行文本。');
    text = input;
end
