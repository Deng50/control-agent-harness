function control_install_matlab_addons(receiptPath, mcpPackage, simulinkPackage)
% 在启动器明确选定的 MATLAB 中定位附加工具箱；两条包路径为空时仅读取当前搜索路径。
% 安装器只传入已按官方 SHA-256 核验的本地文件。由 MATLAB 自身完成安装及路径注册，
% 支持用户自定义 Add-On 目录。返回的回执只含两个安装目录，不含会话凭据。
    if ~isempty(mcpPackage)
        matlab.addons.toolbox.installToolbox(mcpPackage);
    end
    if ~isempty(simulinkPackage)
        matlab.addons.toolbox.installToolbox(simulinkPackage);
    end
    mcpRoot = fileparts(which('shareMATLABSession'));
    toolkitRoot = fileparts(which('satk_initialize'));
    if isempty(toolkitRoot) || ~isfile(fullfile(toolkitRoot, 'tools', 'tools.json'))
        % 新版安装包在 tools/model_edit/ 下注册 model_edit，并附带同级工具清单。
        editFile = which('model_edit');
        toolkitRoot = fileparts(fileparts(fileparts(editFile)));
    end
    if isempty(toolkitRoot) || ~isfile(fullfile(toolkitRoot, 'tools', 'tools.json'))
        toolkitRoot = '';
    end
    [fid, msg] = fopen(receiptPath, 'w', 'n', 'UTF-8');
    assert(fid >= 0, 'ControlHarness:AddonReceipt', '无法记录工具箱安装位置：%s', msg);
    closeFile = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid, '%s', jsonencode(struct('mcpRoot', mcpRoot, 'toolkitRoot', toolkitRoot)));
end
