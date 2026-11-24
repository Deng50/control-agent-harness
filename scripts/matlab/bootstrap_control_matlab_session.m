function bootstrap_control_matlab_session()
% Initialize only an explicitly launched dedicated engineering MATLAB session.
% No Engine installation, global startup edit, session close, or user-path reset.
projectRoot = getenv('CONTROL_MATLAB_PROJECT_ROOT');
privateAppData = getenv('CONTROL_MATLAB_SESSION_APPDATA');
sessionId = getenv('CONTROL_MATLAB_DEDICATED_SESSION_ID');
toolkitRoot = getenv('CONTROL_SIMULINK_TOOLKIT_ROOT');
toolboxRoot = getenv('CONTROL_MATLAB_MCP_TOOLBOX_ROOT');
serverExecutable = getenv('CONTROL_MATLAB_MCP_EXECUTABLE');
required = {projectRoot, privateAppData, sessionId, toolkitRoot, toolboxRoot, serverExecutable};
assert(all(~cellfun(@isempty, required)), 'ControlHarness:MissingLaunchContext', ...
    'Use the explicit dedicated-session launcher; no MATLAB session was changed.');
assert(strcmp(sessionId, getenv('CONTROL_HARNESS_SESSION_ID')), ...
    'ControlHarness:SessionMarkerMismatch', 'The explicit session identity environment does not agree.');
assert(ispc, 'ControlHarness:UnsupportedDiscoveryNamespace', ...
    'This private APPDATA discovery binding is verified on Windows only.');
assert(isfolder(projectRoot) && isfolder(privateAppData), 'ControlHarness:InvalidDirectory', ...
    'The dedicated project and private session directories must already exist.');
assert(isfile(fullfile(toolkitRoot, 'tools', 'tools.json')), ...
    'ControlHarness:MissingToolkit', 'The Simulink extension manifest is missing.');
assert(isfile(fullfile(toolboxRoot, 'shareMATLABSession.m')), ...
    'ControlHarness:MissingMCPToolbox', 'The installed MATLAB MCP Server Toolbox was not found.');
originalAppData = getenv('APPDATA');
assert(~strcmpi(privateAppData, originalAppData), 'ControlHarness:GlobalDiscoveryRejected', ...
    'The dedicated session must not publish through the user global APPDATA.');
addpath(toolboxRoot);
addpath(toolkitRoot);
assert(isfolder(fullfile(projectRoot, 'matlab')), 'ControlHarness:MissingProjectMATLAB', ...
    'The project MATLAB helpers must exist before session bootstrap.');
addpath(fullfile(projectRoot, 'matlab'));
% Both the installed Toolbox and pinned MCP Server discover this namespace.
% Restore this process environment on success and on every MATLAB exception.
restoreEnvironment = onCleanup(@() setenv('APPDATA', originalAppData));
setenv('APPDATA', privateAppData);
% A private toolbox install uses the official package's declared path order.
% No Add-On registration, user preferences or shared connector are modified.
pathManifest = fullfile(fileparts(toolkitRoot), 'metadata', 'configuration.xml');
if isfile(pathManifest)
    nodes = xmlread(pathManifest).getElementsByTagName('matlabPath');
    for n = nodes.getLength()-1:-1:0
        rel = regexprep(char(nodes.item(n).getTextContent()), '^/', '');
        addpath(fullfile(toolkitRoot, rel));
    end
    assert(strcmpi(fileparts(which('shareMATLABSession')), toolboxRoot), ...
        'ControlHarness:UnexpectedConnector', 'The selected private connector was not resolved.');
    shareMATLABSession();
elseif isfile(fullfile(toolkitRoot, 'satk_initialize.p')) || isfile(fullfile(toolkitRoot, 'satk_initialize.m'))
    satk_initialize(MCPServerPath=serverExecutable);
else
    assert(strcmpi(fileparts(which('shareMATLABSession')), toolboxRoot), ...
        'ControlHarness:UnexpectedConnector', 'The connector resolves outside the selected MCP Toolbox.');
    shareMATLABSession();
end
% 两种布局都要求实际加载所选扩展中的编辑函数，防止只有清单或解析到了其他版本。
assert(strcmpi(fileparts(which('model_edit')), fullfile(toolkitRoot, 'tools', 'model_edit')), ...
    'ControlHarness:ToolkitNotRegistered', ...
    'model_edit does not resolve to the selected toolkit. Enable the Simulink Agentic Toolkit in this MATLAB.');
detailsPath = fullfile(privateAppData, 'MathWorks', 'MATLAB MCP Server', 'v1', 'sessionDetails.json');
assert(isfile(detailsPath), 'ControlHarness:MissingConnector', ...
    'The connector did not publish in the private directory.');
clear restoreEnvironment;
cd(projectRoot);
marker = struct('schema_version', 1, 'kind', 'control-harness-dedicated-matlab-v1', 'status', 'ready', ...
    'session_id', sessionId, 'pid', feature('getpid'), 'matlab_pid', feature('getpid'), 'project_root', projectRoot, ...
    'private_appdata', privateAppData, 'matlab_release', version('-release'), ...
    'toolkit_root', toolkitRoot, 'toolbox_root', toolboxRoot, ...
    'mcp_executable', serverExecutable, 'createdAt', char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''')));
markerPath = fullfile(privateAppData, 'control-harness-session.json');
% 先关闭临时文件再在同一目录发布；轮询就绪文件的启动器只会看到完整 JSON。
tempMarker = [markerPath '.' sessionId '.tmp'];
fid = fopen(tempMarker, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'ControlHarness:MarkerWriteFailed', 'Cannot write dedicated-session marker.');
closeMarker = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(marker, PrettyPrint=true));
clear closeMarker;
[published, detail] = movefile(tempMarker, markerPath);
assert(published, 'ControlHarness:MarkerPublishFailed', 'Cannot publish ready marker: %s', detail);
fprintf('Dedicated control MATLAB ready. PID=%d release=%s\n', marker.pid, marker.matlab_release);
end
