function test_simulink_plant_parameters()
% R2024b 原生回归。在同一临时模型中完成块参数拒绝、纠正、读回、接线、编译和短仿真。
% 所有块均属于本次随机命名的内存模型，不保存或覆盖用户文件。
    project = fileparts(fileparts(mfilename('fullpath')));
    previousPath = path();
    restorePath = onCleanup(@() path(previousPath)); %#ok<NASGU>
    addpath(fullfile(project, 'matlab'));
    model = ['PlantRegression_' strrep(char(java.util.UUID.randomUUID()), '-', '')];
    new_system(model);
    closeModel = onCleanup(@() close_system(model, 0)); %#ok<NASGU>
    set_param(model, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
        'FixedStep', '0.01', 'StopTime', '0.01', 'ReturnWorkspaceOutputs', 'on');

    % 会话中的单质量弹簧阻尼对象。分别保留控制力 u、负载力 d 的两个输入。
    dt = 0.01;
    transition = expm([0 1 0; -10 -0.5 1; 0 0 0] * dt);
    A = transition(1:2, 1:2);
    Bd = transition(1:2, 3);
    B = [Bd Bd];
    C = eye(2);
    D = zeros(2);
    x0 = [1; 0];
    libraryBlock = 'simulink/Discrete/Discrete State-Space';
    % 在干净的默认块上，记录旧调用的具体失败位置，不从错误文案反推堆栈。
    % 此处故意逐项设置；生产入口必须只调用一次 set_param。
    probe = [model '/PreviousSequence'];
    add_block(libraryBlock, probe);
    oldNames = {'A', 'B', 'C', 'D', 'X0', 'SampleTime'};
    oldValues = {A, B, C, D, x0, dt};
    failedAt = '';
    for i = 1:numel(oldNames)
        try
            set_param(probe, oldNames{i}, mat2str(oldValues{i}, 8));
        catch exception
            failedAt = oldNames{i};
            fprintf('Old sequential write failed at %s (%s): %s\n', failedAt, ...
                exception.identifier, getReport(exception, 'basic', 'hyperlinks', 'off'));
            break
        end
    end
    assert(~isempty(failedAt), 'The old sequential configuration unexpectedly succeeded.');
    delete_block(probe);

    target = [model '/Plant'];
    add_block(libraryBlock, target);
    names = {'A', 'B', 'C', 'D', 'InitialCondition', 'SampleTime'};
    readGroup = @() cellfun(@(name) get_param(target, name), names, 'UniformOutput', false);
    before = readGroup();
    refused = control_harness_configure_discrete_plant(model, 'Plant', A, 1, C, D, x0, dt);
    assert(strcmp(refused.status, 'refused') && strcmp(refused.identifier, 'ControlHarness:PlantShapeB'));
    assert(~refused.parameters_applied && isequal(readGroup(), before));

    % 即使此前因无效 B 被拒绝，同一模型、同一个块仍可提交正确的一组参数。
    receipt = control_harness_configure_discrete_plant(model, 'Plant', A, B, C, D, x0, dt);
    assert(strcmp(receipt.status, 'configured'), '%s: %s', receipt.identifier, receipt.reason);
    assert(receipt.parameters_applied && strcmp(receipt.stage, 'readback'));
    assert(strcmp(receipt.block_path, target) && strcmp(receipt.diagram_update, 'not_run'));
    assert(isequal([receipt.states receipt.inputs receipt.outputs], [2 2 2]));
    assert(isequal({receipt.A, receipt.B, receipt.C, receipt.D, receipt.initial_condition, receipt.sample_time}, ...
        {A, B, C, D, x0, dt}));
    expectedLiterals = cellfun(@(v) mat2str(v, 17), {A,B,C,D,x0,dt}, 'UniformOutput', false);
    assert(isequal(readGroup(), expectedLiterals));

    % 继续搭建一个确定性反馈夹具以检验 Plant 的实际端口、状态顺序及一步行为。
    % 使用 u=K*x 和独立常值扰动 d，便于独立计算参考；此控制律只用于本测试。
    K = [-1 -0.1];
    d = 0.25;
    add_block('simulink/Math Operations/Gain', [model '/Feedback'], ...
        'Multiplication', 'Matrix(K*u)', 'Gain', mat2str(K,17));
    add_block('simulink/Sources/Constant', [model '/Disturbance'], 'Value', mat2str(d,17));
    add_block('simulink/Signal Routing/Mux', [model '/Forces'], 'Inputs', '2');
    add_block('simulink/Sinks/To Workspace', [model '/StateTrace'], ...
        'VariableName', 'plantCheckStates', 'SaveFormat', 'Timeseries');
    add_line(model, 'Plant/1', 'Feedback/1');
    add_line(model, 'Feedback/1', 'Forces/1');
    add_line(model, 'Disturbance/1', 'Forces/2');
    add_line(model, 'Forces/1', 'Plant/1');
    add_line(model, 'Plant/1', 'StateTrace/1');
    set_param(model, 'SimulationCommand', 'update');
    output = sim(model);
    trace = output.get('plantCheckStates');
    assert(numel(trace.Time) == 2 && abs(trace.Time(2)-dt) < 1e-14);
    actual = getdatasamples(trace, 2);
    expected = A*x0 + B*[K*x0;d];
    errorNorm = norm(actual(:) - expected, inf);
    assert(errorNorm < 1e-12, 'One-step mismatch: %.17g', errorNorm);

    % 另一组尺寸（1 状态、2 输入、3 输出）和嵌套路径，检查没有为弹簧对象写特例。
    add_block('built-in/Subsystem', [model '/Nested']);
    add_block(libraryBlock, [model '/Nested/Other']);
    other = control_harness_configure_discrete_plant(model, 'Nested/Other', ...
        0.9, [0.4 -0.2], [1; -1; 0.2], zeros(3,2), 2, 0.1);
    assert(strcmp(other.status,'configured'), '%s: %s', other.identifier, other.reason);
    assert(isequal([other.states other.inputs other.outputs],[1 2 3]));
    assert(isequal(other.B, [0.4 -0.2]) && isequal(other.C,[1; -1; 0.2]));
    refused = control_harness_configure_discrete_plant(model, 'Feedback', A, B, C, D, x0, dt);
    assert(strcmp(refused.identifier, 'ControlHarness:PlantBlockType'));
    fprintf('PASS %s: Plant readback=%s; continued feedback simulation x1=%s, max_error=%.3g; nested dimensions 1/2/3.\n', ...
        version('-release'), jsonencode(receipt), mat2str(actual(:), 17), errorNorm);
end
