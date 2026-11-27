function certificate = control_evaluate_pgac_batch(manifestPath,nonce)
% S12: six local S09 batches in ONE native business call, never per-step MCP.
manifest=jsondecode(fileread(manifestPath));exchange=fileparts(manifestPath);
assert(manifest.schema_version==1&&strcmp(manifest.kind,'control-matlab-pgac-evaluation-v1')&&strcmp(manifest.nonce,nonce), ...
    'ControlHarness:PGACManifest','Unexpected fixed PGAC evaluation manifest.');
assert(strcmp(manifest.result_path,fullfile(exchange,'result.json'))&&strcmp(manifest.selection_path,fullfile(exchange,'selection.json')), ...
    'ControlHarness:PGACManifest','Fixed result/selection paths differ.');
oldPath=path;oldDirectory=pwd;oldThreads=maxNumCompThreads(1);restore=onCleanup(@() restore_environment(oldPath,oldDirectory,oldThreads));
assert(maxNumCompThreads==1,'ControlHarness:PGACThreads','One computational thread is required.');
names={'expert','imitation','candidate'};parts={'validation','test'};children=struct;certificates=struct;loaded=struct;hashes=struct;testBindings=struct;total=0;allComplete=true;
for phaseIndex=1:2
    phase=parts{phaseIndex};
    for k=1:3
        name=names{k};prepared=manifest.children.(phase).(name);expected=fullfile(exchange,phase,name,'manifest.json');
        assert(strcmp(prepared.manifest_path,expected)&&strcmp(file_sha256(expected),prepared.manifest_sha256), ...
            'ControlHarness:PGACChild','Prepared original-source child manifest changed.');
        children.(phase).(name)=jsondecode(fileread(expected));child=children.(phase).(name);
        assert(strcmp(child.nonce,nonce)&&strcmp(child.request_sha256,manifest.request_sha256),'ControlHarness:PGACChild','Child identity differs.');
        if phaseIndex==2
            assert(exist(manifest.selection_path,'file')==2&&strcmp(file_sha256(manifest.selection_path),selectionSHA), ...
                'ControlHarness:PGACSelection','Frozen validation selection changed before a test rollout.');
            testBindings.(name)=selectionSHA;
        end
        certificates.(phase).(name)=control_harness_expert_batch(expected,nonce);
        loaded.(phase).(name)=jsondecode(fileread(child.result_path));hashes.(phase).(name)=file_sha256(child.result_path);
        total=total+certificates.(phase).(name).samples;allComplete=allComplete&&strcmp(certificates.(phase).(name).status,'completed');
    end
    if phaseIndex==1
        scores=struct;
        for k=1:3
            name=names{k};domain=[];guardHash='';
            if strcmp(name,'candidate')
                domain=manifest.candidate_domain;child=children.validation.(name);
                assert(isequal(child.coverage_domain,domain),'ControlHarness:PGACCoverage','Candidate child coverage differs from its confirmed parent.');guardHash=child.coverage_guard_hash;
            end
            scores.(name)=quadratic_score(loaded.validation.(name),children.validation.(name).plan,manifest.evaluation,domain,guardHash);
        end
        decision=validation_decision(scores,manifest.evaluation,manifest.candidate_eligible);
        selection=struct('schema_version',1,'phase','frozen_before_any_test_rollout','nonce',nonce,'request_sha256',manifest.request_sha256, ...
            'plan_hash',manifest.plan.plan_hash,'evaluation',manifest.evaluation,'decision',decision,'validation_result_sha256',hashes.validation);
        write_json(manifest.selection_path,selection,nonce);selectionSHA=file_sha256(manifest.selection_path);
    end
end
benchmark=struct('status','not_requested','scope','No new latency claim; prior S10 timing remains separate.');
if isstruct(manifest.benchmark)
    try
        benchmark=controller_benchmark(children.test.expert,children.test.imitation,children.test.candidate,loaded.test.expert,manifest.benchmark);
    catch problem
        benchmark=struct('status','failed','settings',manifest.benchmark,'reason',problem.message,'error_identifier',problem.identifier);
    end
end
status='completed';if ~allComplete,if total>0,status='partial';else,status='failed';end,end
result=struct('schema_version',1,'kind','control-matlab-pgac-evaluation-result-v1','nonce',nonce,'request_sha256',manifest.request_sha256, ...
    'spec_hash',manifest.spec_hash,'plan_hash',manifest.plan.plan_hash,'status',status,'children',hashes,'selection_sha256',selectionSHA, ...
    'test_started_after_selection',true,'test_selection_sha256',testBindings,'controller_benchmark',benchmark, ...
    'matlab_release',version('-release'),'matlab_pid',feature('getpid'),'hardware_execution',false, ...
    'scope','Actual original Plant; own controller feedback; validation fixes one version before all final test diagnostics; no analytic stability certificate.');
versions={'policy_version','candidate_version','dataset_version','expert_version','expert_material_version','plant_version','plant_material_version'};
for k=1:numel(versions),name=versions{k};result.(name)=manifest.(name);end
write_json(manifest.result_path,result,nonce);
certificate=struct('nonce',nonce,'status',status,'result_path',manifest.result_path,'samples',total,'stop_detection','original_S09_local_batches');
end

function decision = validation_decision(scores,evaluation,eligible)
base=scores.imitation;candidate=scores.candidate;choice=[];reason='no_approved_validation_fallback';
better=false;
if eligible && candidate.feasible && base.feasible
    gain=base.score-candidate.score;
    better=gain>evaluation.comparison_atol;
    if isfield(evaluation,'min_relative_improvement')
        better=better && gain+evaluation.comparison_atol>=base.score*evaluation.min_relative_improvement;
    end
    if isfield(evaluation,'max_worst_case_relative_regression')
        for k=1:numel(base.per_scenario)
            bound=base.per_scenario(k).score*(1+evaluation.max_worst_case_relative_regression);
            better=better && candidate.per_scenario(k).score<=bound+evaluation.comparison_atol;
        end
    end
end
if eligible&&candidate.feasible&&(~base.feasible||better)
    choice='candidate';if base.feasible,reason='candidate_strictly_better_on_original_validation';else,reason='candidate_only_feasible_original_validation';end
elseif base.feasible
    choice='imitation';reason='feasible_imitation_retained_without_required_improvement';
end
decision=struct('choice',choice,'reason',reason,'candidate_eligible',logical(eligible),'scores',scores, ...
    'comparison_atol',evaluation.comparison_atol,'selection_partition','validation','test_used',false);
end

function score = quadratic_score(result,plan,evaluation,domain,guardHash)
physical=plan.interface;nx=numel(physical.states);nu=numel(physical.controls);ny=numel(physical.outputs);count=numel(result.arrays.time);
a=result.arrays;x=matrix(a.x,count,nx);xn=matrix(a.next_x,count,nx);y=matrix(a.plant_y,count,ny);yn=matrix(a.next_y,count,ny);
u=matrix(a.u,count,nu);previous=matrix(a.previous_u,count,nu);reference=matrix(a.reference,count,ny);ids=reshape(a.trajectory_id,[],1);time=reshape(a.time,[],1);
expected=sum([plan.trajectories.steps]);checks=struct('completed_all_trajectories',strcmp(result.status,'completed')&&count==expected, ...
    'state',bounded(x,physical.states)&&bounded(xn,physical.states),'output',bounded(y,physical.outputs)&&bounded(yn,physical.outputs), ...
    'input',bounded(u,physical.controls),'input_rate',true,'causal_chain',true,'finite',all(isfinite([x(:);xn(:);y(:);yn(:);u(:);previous(:);reference(:)])), ...
    'qualified_domain',true,'subtick_coverage',true);
if isstruct(domain)
    d=matrix(a.d,count,numel(physical.disturbances));
    checks.qualified_domain=inside_domain(x,domain.x)&&inside_domain(xn,domain.x)&&inside_domain(u,domain.u)&&inside_domain(d,domain.d);
    checks.subtick_coverage=false;
    if isfield(result,'coverage_guard')
        guard=result.coverage_guard;ticks=expected*plan.plant_substeps;
        checks.subtick_coverage=guard.enabled&&strcmp(guard.domain_hash,guardHash)&&strcmp(guard.status,'passed')&&guard.violations==0 ...
            &&guard.before_checks==ticks&&guard.after_checks==ticks&&guard.passed_subticks==ticks;
    end
    checks.qualified_domain=checks.qualified_domain&&checks.subtick_coverage;
end
yr=reshape([physical.outputs.upper]-[physical.outputs.lower],1,[]);ur=reshape([physical.controls.upper]-[physical.controls.lower],1,[]);
values=[];records=struct('trajectory_id',{},'steps',{},'score',{});
for k=1:numel(plan.trajectories)
    condition=plan.trajectories(k);indices=find(ids==condition.trajectory_id);n=numel(indices);
    checks.completed_all_trajectories=checks.completed_all_trajectories&&n==condition.steps;
    if n==0,records(end+1)=struct('trajectory_id',condition.trajectory_id,'steps',0,'score',[]);continue;end %#ok<AGROW>
    p=previous(indices,:);commands=u(indices,:);states=x(indices,:);next=xn(indices,:);
    checks.causal_chain=checks.causal_chain&&close(states(1,:),condition.x0)&&close(p(1,:),condition.initial_action) ...
        &&close(states(2:end,:),next(1:end-1,:))&&close(p(2:end,:),commands(1:end-1,:)) ...
        &&close(time(indices),condition.time(1:n))&&close(reference(indices,:),repmat(reshape(condition.reference,1,[]),n,1));
    % S09 checks the declared per-update rate and every physical state/output
    % subtick before a row is accepted. Python independently repeats row checks.
    tracking=mean(((yn(indices,:)-reference(indices,:))./yr).^2,2);action=mean((commands./ur).^2,2);increment=mean(((commands-p)./ur).^2,2);
    value=evaluation.q*mean(tracking)+evaluation.r*mean(action)+evaluation.s*mean(increment);checks.finite=checks.finite&&isfinite(value);
    values(end+1)=value;records(end+1)=struct('trajectory_id',condition.trajectory_id,'steps',n,'score',value); %#ok<AGROW>
end
feasible=all(structfun(@(v) logical(v),checks));value=[];if feasible,value=mean(values);end
score=struct('feasible',logical(feasible),'score',value,'checks',checks,'per_scenario',records);
end

function result = inside_domain(array,bounds)
if isempty(bounds),result=isempty(array);return;end
result=all(isfinite(array),'all')&&all(array>=reshape(bounds(:,1),1,[])&array<=reshape(bounds(:,2),1,[]),'all');
end

function result = bounded(array,variables)
low=reshape([variables.lower],1,[]);high=reshape([variables.upper],1,[]);
result=all(isfinite(array),'all')&&all(array>=low-1e-6&array<=high+1e-6,'all');
end

function result = close(left,right)
if isvector(left)&&isvector(right),right=reshape(right,size(left));end
result=isequal(size(left),size(right))&&all(abs(left-right)<=1e-12,'all');
end

function value = matrix(array,count,width)
assert(isnumeric(array)&&isreal(array)&&numel(array)==count*width,'ControlHarness:PGACShape','Original-source numeric array shape differs.');
value=reshape(double(array),count,width);
end

function write_json(destination,value,nonce)
temporary=[destination '.' nonce '.tmp'];fid=fopen(temporary,'w','n','UTF-8');assert(fid>=0,'ControlHarness:PGACWrite','Cannot write trusted PGAC evaluation evidence.');
cleanup=onCleanup(@() fclose(fid));fprintf(fid,'%s\n',jsonencode(value,PrettyPrint=true,ConvertInfAndNaN=true));clear cleanup;movefile(temporary,destination,'f');
end


function benchmark = controller_benchmark(expertManifest,nnManifest,candidateManifest,reference,settings)
assert(settings.batch_size==1&&settings.threads==1&&settings.warmup>=1&&settings.repeats>=1, ...
    'ControlHarness:TimingSettings','Timing requires explicit warmup/repeats, batch one and one computational thread.');
count=numel(reference.arrays.trajectory_id);
if count==0,benchmark=struct('status','not_applicable','reason','No valid expert operating contexts');return;end
sources={expertManifest.expert,nnManifest.expert,candidateManifest.expert};names={'expert','imitation','candidate'};raw=struct;contexts=cell(3,min(count,settings.repeats));
for sourceIndex=1:3
    source=sources{sourceIndex};addpath(source.function_path_root);clear(source.spec.entrypoint);
    assert(strcmp(which(source.spec.entrypoint),source.source_file),'ControlHarness:PolicyTimingIdentity','A timed controller resolves outside its frozen snapshot.');
    for index=1:size(contexts,2)
        x=row(reference.arrays.x,index,numel(source.spec.interface.states));y=row(reference.arrays.plant_y,index,numel(source.spec.interface.outputs));
        observation=zeros(numel(source.spec.observations),1);
        for j=1:numel(source.spec.observations),item=source.spec.observations(j);if strcmp(item.source,'state'),observation(j)=x(item.index+1);else,observation(j)=y(item.index+1);end,end
        r=row(reference.arrays.reference,index,numel(source.spec.interface.outputs));d=row(reference.arrays.d,index,numel(source.spec.interface.disturbances));
        previous=row(reference.arrays.previous_u,index,numel(source.spec.interface.controls));
        if ~source.spec.reference_available,r=zeros(0,1);end
        if ~source.spec.disturbance_available,d=zeros(0,1);end
        if ~source.spec.previous_action_available,previous=zeros(0,1);end
        if sourceIndex==1,memory=reference.diagnostics(index).memory_before;else,memory=source.spec.initial_memory;end
        contexts{sourceIndex,index}={observation,r,d,previous,memory,source.spec.parameters};
    end
end
for sourceIndex=1:3
    source=sources{sourceIndex};name=names{sourceIndex};nu=numel(source.spec.interface.controls);values=zeros(settings.repeats,1);
    % Both NN versions intentionally share the generated function basename.
    % Rebind and verify each exact snapshot outside its timed section.
    addpath(source.function_path_root);clear(source.spec.entrypoint);
    assert(strcmp(which(source.spec.entrypoint),source.source_file),'ControlHarness:PolicyTimingIdentity','A timed controller resolves outside its own frozen snapshot.');
    for index=1:size(contexts,2),args=contexts{sourceIndex,index};[action,memory,status]=feval(source.spec.entrypoint,args{:});validate_result(action,memory,status,source,row(reference.arrays.previous_u,index,nu));end
    for index=1:settings.warmup,contextIndex=mod(index-1,size(contexts,2))+1;args=contexts{sourceIndex,contextIndex};[action,memory,status]=feval(source.spec.entrypoint,args{:});validate_result(action,memory,status,source,row(reference.arrays.previous_u,contextIndex,nu));end
    for index=1:settings.repeats
        contextIndex=mod(index-1,size(contexts,2))+1;args=contexts{sourceIndex,contextIndex};
        clock=tic;[action,memory,status]=feval(source.spec.entrypoint,args{:});values(index)=toc(clock);
        validate_result(action,memory,status,source,row(reference.arrays.previous_u,contextIndex,nu)); % Validation outside timing.
    end
    raw.(name)=struct('raw_seconds',values,'unit','seconds');
    verify_sources(source);
    clear(source.spec.entrypoint);
end
benchmark=struct('status','completed','settings',settings,'expert',raw.expert,'imitation',raw.imitation,'candidate',raw.candidate, ...
    'matlab_max_computational_threads',maxNumCompThreads,'matlab_release',version('-release'),'architecture',computer('arch'), ...
    'scope','Raw expert function vs numeric NN function including normalization/projection; excludes Plant, MCP, imports and S09 triple/fresh validation.', ...
    'input_contexts','Same recorded held-out expert physical states/outputs/reference/disturbance/previous actions; prepared before timing.', ...
    'not_a_deployment_latency_guarantee',true,'stability_proof',false);
end

function validate_result(action,memory,status,source,previous)
width=numel(source.spec.interface.controls);
assert(isnumeric(action)&&isreal(action)&&numel(action)==width&&all(isfinite(action)), ...
    'ControlHarness:TimingAction','Raw timed controller returned a nonfinite/incorrect action.');
assert(isstruct(memory)&&isscalar(memory)&&strcmp(status,'ok'),'ControlHarness:TimingStatus','Raw timed controller did not return an explicit successful result.');
applied=zeros(width,1);for k=1:numel(source.spec.action_mapping),item=source.spec.action_mapping(k);applied(item.plant_control_index+1)=action(item.action_index+1);end
for k=1:width,item=source.spec.interface.controls(k);assert(applied(k)>=item.lower-1e-12&&applied(k)<=item.upper+1e-12,'ControlHarness:TimingBounds','Timed action violates its actuator bounds.');end
if isfield(source.spec,'input_rate_bounds')
    delta=applied-previous;rate=source.spec.input_rate_bounds;assert(all(delta>=rate(:,1)-1e-12&delta<=rate(:,2)+1e-12),'ControlHarness:TimingRate','Timed action violates its actual per-update increment.');
end
end

function value = row(array,index,width)
if width==0,value=zeros(0,1);elseif width==1,value=double(array(index));else,value=double(array(index,:).');end
end

function verify_sources(source)
for k=1:numel(source.source_files),entry=source.source_files(k);assert(strcmp(file_sha256(fullfile(source.source_root,entry.relative_path)),entry.sha256), ...
    'ControlHarness:TimingSource','Frozen controller source changed during timing.');end
end

function digest = file_sha256(source)
fid=fopen(source,'rb');assert(fid>=0,'ControlHarness:TimingSource','Cannot read frozen controller.');cleanup=onCleanup(@() fclose(fid));hasher=javaMethod('getInstance','java.security.MessageDigest','SHA-256');
while true,bytes=fread(fid,1024*1024,'*uint8');if isempty(bytes),break;end;hasher.update(typecast(bytes,'int8'));end
digest=lower(reshape(dec2hex(typecast(hasher.digest(),'uint8'),2).',1,[]));
end

function restore_environment(oldPath,oldDirectory,oldThreads)
path(oldPath);cd(oldDirectory);maxNumCompThreads(oldThreads);
end
