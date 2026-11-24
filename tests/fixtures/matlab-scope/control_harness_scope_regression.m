function r = control_harness_scope_regression(model,target,sessionId)
% Fixture for session c9b2 event #43-44. Leave the owned model in memory.
r = struct('schema_version',1,'status','failed','stage','scope','verified',{{}}, ...
    'changes',{{}},'file_exists',isfile(target),'path',target,'model',model,'bytes',0, ...
    'matlab_pid',feature('getpid'),'release',version('-release'),'recoverable',false, ...
    'identifier','','reason','');
assert(isvarname(model) && ~bdIsLoaded(model) && ~isfile(target), ...
    'ControlHarness:TestCollision','The scope fixture requires a fresh model and destination.');
new_system(model);
set_param(model,'StopTime','10','SolverType','Fixed-step','FixedStep','0.01','SaveOutput','on','SaveTime','on');
hws = get_param(model,'ModelWorkspace');
hws.assignin('ControlHarnessBuildTarget',struct('path',target,'session_id',sessionId));
hws.assignin('mass',1.0); hws.assignin('damping',0.5); hws.assignin('stiffness',2.0);
hws.assignin('dt',0.01); hws.assignin('Ac',[0 1; -2.0/1.0 -0.5/1.0]); hws.assignin('Bc',[0;1/1.0]);
r.verified = {'create'};
r.changes = {'created memory model','assigned mass,damping,stiffness,dt,Ac,Bc in model workspace'};
try
    aug = [Ac Bc; 0 0 0]; %#ok<NASGU,NODEF> deliberate failure from #43
catch err
    r.identifier = err.identifier;
    r.reason = err.message;
    r.recoverable = strcmp(err.identifier,'MATLAB:UndefinedFunction') && hws.hasVariable('Ac') && ~exist('Ac','var');
    return
end
error('ControlHarness:ScopeNotReproduced','The out-of-scope matrix unexpectedly resolved.');
end
