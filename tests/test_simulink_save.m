function test_simulink_save()
% 独立 MATLAB 批处理回归。所有写入位于本检出目录的 test-output。
% 两个工作目录共用一个 MATLAB，以实际 FileName 验证同名文件保护。
    project = fileparts(fileparts(mfilename('fullpath')));
    previousPath = path();
    cleanupPath = onCleanup(@() path(previousPath)); %#ok<NASGU>
    addpath(fullfile(project, 'matlab'));
    output = fullfile(project, 'test-output');
    if ~isfolder(output), mkdir(output); end
    root = tempname(output);
    first = fullfile(root, '会话 A');
    other = fullfile(root, '会话 B');
    mkdir(first);
    mkdir(other);
    model = ['SaveFixture_' strrep(char(java.util.UUID.randomUUID()), '-', '')];
    target = fullfile(first, [model '.slx']);
    otherTarget = fullfile(other, [model '.slx']);
    new_system(model);
    cleanupModel = onCleanup(@() close_system(model, 0)); %#ok<NASGU>
    add_block('simulink/Sources/In1', [model '/u_force']);
    add_block('simulink/Sinks/Out1', [model '/y']);
    add_line(model, 'u_force/1', 'y/1');

    % 已存在的其他文件不被覆盖。
    fid = fopen(target, 'w');
    assert(fid >= 0);
    fprintf(fid,'fixture');
    fclose(fid);
    blocked = control_harness_save_slx(model, target);
    assert(strcmp(blocked.status, 'refused') && strcmp(fileread(target), 'fixture'));
    delete(target);

    saved = control_harness_save_slx(model, target);
    assert(strcmp(saved.status, 'saved') && strcmp(saved.path, target));
    assert(saved.bytes == dir(target).bytes);
    % 复保存仍只写已绑定的文件。同一 MATLAB 中的另一个对话不能保存此模型。
    again = control_harness_save_slx(model, target);
    assert(strcmp(again.status, 'saved'));
    blocked = control_harness_save_slx(model, otherTarget);
    assert(strcmp(blocked.status, 'refused') && ~isfile(otherTarget));

    % 验证单独的错误调用不会抹掉已经落盘的模型。随后从实际保存路径独立加载。
    observed = false;
    try
        set_param([model '/u_force'], 'Value', '1');
    catch error
        observed = contains(error.message, 'Value');
        fprintf('Inport parameter error identifier: %s\n', error.identifier);
    end
    assert(observed && isfile(target));
    close_system(model, 0);
    load_system(target);
    assert(strcmp(get_param([model '/u_force'], 'BlockType'), 'Inport'));
    fprintf('Verified SLX path: %s (%d bytes)\n', saved.path, saved.bytes);
end
