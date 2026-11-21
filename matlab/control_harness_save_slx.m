function receipt = control_harness_save_slx(model, destination)
% 将当前内存中的顶层模型保存到 Host 从本次会话 cwd 生成的绝对文件名。
% 此入口不创建、加载或关闭模型，不修改输入、仿真参数或调用者的当前目录。
% 普通的前置条件失败返回说明；实际保存或校验异常保留原始异常，交由资源层处理。
    receipt = struct('status', 'refused', 'model', model, 'reason', '');
    if ~ischar(model) || ~isvarname(model) || ~bdIsLoaded(model)
        receipt.reason = '指定模型尚未加载，请核对本次会话创建的顶层模型名。';
        return
    end
    [parent, leaf, suffix] = fileparts(destination);
    target = java.io.File(destination);
    if ~target.isAbsolute() || ~strcmp(leaf, model) || ~strcmp(suffix, '.slx') || ~isfolder(parent)
        receipt.reason = '保存路径须为会话工作目录内与已加载模型同名的 .slx 文件。';
        return
    end
    if ~sameFileName(char(target.getCanonicalPath()), destination) || java.nio.file.Files.isSymbolicLink(target.toPath())
        receipt.reason = '目标文件或上级目录发生重定向，已停止保存。';
        return
    end
    % 多个对话共用同一 MATLAB。内存模型若已绑定其他路径，不能用同名请求将另一会话的模型写入本目录。
    source = get_param(model, 'FileName');
    if ~isempty(source) && ~sameFileName(char(java.io.File(source).getCanonicalPath()), destination)
        receipt.reason = '内存中的同名模型已绑定其他路径，请选择属于当前会话的模型。';
        return
    end
    if isempty(source) && target.exists()
        receipt.reason = '当前目录已存在同名文件。请先核对并打开该版本，或为新模型选择其他名称。';
        return
    end

    % 总是传入显式绝对路径。保持 Simulink 的磁盘版本冲突检查；其他被引用模型、外部工作区不随之写入。
    actual = save_system(model, destination, 'AllowPrompt', false, 'OverwriteIfChangedOnDisk', false, ...
        'SaveDirtyReferencedModels', false, 'SaveModelWorkspace', false);
    if iscell(actual) && isscalar(actual)
        actual = actual{1};
    end
    assert(ischar(actual) || (isstring(actual) && isscalar(actual)), 'ControlHarness:SaveReceipt', ...
        'save_system 未返回一个可核对的文件名。');
    actual = char(java.io.File(char(actual)).getCanonicalPath());
    assert(sameFileName(actual, destination) && target.isFile(), 'ControlHarness:SaveLocation', ...
        'Simulink 实际保存的文件不在预期位置。');
    % 不加载或仿真模型；解析刚保存的 SLX 清单，避免把同名的普通文件误报为模型。
    info = Simulink.MDLInfo(actual);
    assert(strcmp(info.BlockDiagramName, model), 'ControlHarness:SavedDiagram', ...
        '保存文件中的模型名与请求不同。');
    receipt = struct('status', 'saved', 'model', model, 'path', actual, 'bytes', double(target.length()));
end

function equal = sameFileName(a, b)
% Windows 本地盘路径忽略大小写；其他平台保留大小写语义。
    if ispc
        equal = strcmpi(a, b);
    else
        equal = strcmp(a, b);
    end
end
