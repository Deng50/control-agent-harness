function result = control_harness_terminal_probe(nonce)
% Read-only certificate for one dedicated session and MATLAB-managed work.
% This is not a proof about arbitrary detached OS processes or remote jobs.
arguments
    nonce (1,:) char
end
assert(~isempty(regexp(nonce, '^[a-f0-9-]{36}$', 'once')), ...
    'ControlHarness:InvalidProbeNonce', 'Expected a Host-generated UUID.');
busyModels = {};
models = find_system('SearchDepth', 0, 'Type', 'block_diagram');
for index = 1:numel(models)
    if strcmp(get_param(models{index}, 'BlockDiagramType'), 'model') && ...
            ~strcmp(get_param(models{index}, 'SimulationStatus'), 'stopped')
        busyModels{end+1} = models{index}; %#ok<AGROW>
    end
end
backgroundFutures = 0;
backgroundAvailable = true;
try
    pool = backgroundPool;
    backgroundFutures = numel(pool.FevalQueue.RunningFutures) + numel(pool.FevalQueue.QueuedFutures);
catch exception
    if strcmp(exception.identifier, 'MATLAB:UndefinedFunction')
        backgroundAvailable = false;
    else
        rethrow(exception);
    end
end
parallelFutures = 0;
if exist('gcp', 'file') == 2
    pool = gcp('nocreate');
    if ~isempty(pool)
        parallelFutures = numel(pool.FevalQueue.RunningFutures) + numel(pool.FevalQueue.QueuedFutures);
    end
end
timers = timerfindall;
runningTimers = 0;
for index = 1:numel(timers)
    runningTimers = runningTimers + strcmp(timers(index).Running, 'on');
end
result = struct('schema_version', 1, 'nonce', nonce, ...
    'session_id', getenv('CONTROL_HARNESS_SESSION_ID'), 'matlab_pid', feature('getpid'), ...
    'helper_file', mfilename('fullpath'), 'busy_models', {busyModels}, ...
    'background_futures', backgroundFutures, 'parallel_futures', parallelFutures, ...
    'running_timers', runningTimers, 'background_available', backgroundAvailable, ...
    'idle', isempty(busyModels) && backgroundFutures == 0 && parallelFutures == 0 && runningTimers == 0, ...
    'probe_scope', 'dedicated-session synchronous calls, Simulink, local futures, timers');
end
