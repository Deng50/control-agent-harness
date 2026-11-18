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
assert(isfile(fullfile(toolkitRoot, 'satk_initialize.p')) || ...
    isfile(fullfile(toolkitRoot, 'satk_initialize.m')), 'ControlHarness:MissingToolkit', ...
    'The installed Simulink Toolkit initialization entry was not found.');
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
satk_initialize(MCPServerPath=serverExecutable);
detailsPath = fullfile(privateAppData, 'MathWorks', 'MATLAB MCP Server', 'v1', 'sessionDetails.json');
details = jsondecode(fileread(detailsPath));
assert(double(details.pid) == feature('getpid'), 'ControlHarness:WrongSession', ...
    'The private connection descriptor does not identify this dedicated MATLAB.');
clear restoreEnvironment;
cd(projectRoot);
marker = struct('schema_version', 1, 'kind', 'control-harness-dedicated-matlab-v1', 'status', 'ready', ...
    'session_id', sessionId, 'pid', feature('getpid'), 'matlab_pid', feature('getpid'), 'project_root', projectRoot, ...
    'private_appdata', privateAppData, 'matlab_release', version('-release'), ...
    'toolkit_root', toolkitRoot, 'createdAt', char(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS''Z''')));
markerPath = fullfile(privateAppData, 'control-harness-session.json');
fid = fopen(markerPath, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'ControlHarness:MarkerWriteFailed', 'Cannot write dedicated-session marker.');
closeMarker = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(marker, PrettyPrint=true));
clear closeMarker;
fprintf('Dedicated control MATLAB ready. PID=%d release=%s\n', marker.pid, marker.matlab_release);
end
