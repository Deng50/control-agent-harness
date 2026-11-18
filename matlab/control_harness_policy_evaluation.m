function certificate = control_harness_policy_evaluation(manifestPath,nonce)
% S10 fixed native entrypoint: two original-Plant rollouts, ONE MCP business call.
% S09 trusted batches retain their unchanged source/reset/shape/bounds protocol.
manifest=jsondecode(fileread(manifestPath));exchange=fileparts(manifestPath);
assert(manifest.schema_version==1&&strcmp(manifest.kind,'control-matlab-policy-evaluation-v1')&&strcmp(manifest.nonce,nonce), ...
    'ControlHarness:PolicyEvaluationManifest','Unexpected fixed evaluation manifest.');
assert(strcmp(manifest.result_path,fullfile(exchange,'result.json')),'ControlHarness:PolicyEvaluationManifest','Unexpected result path.');
oldPath=path;oldDirectory=pwd;oldThreads=maxNumCompThreads(1);
restore=onCleanup(@() restore_environment(oldPath,oldDirectory,oldThreads));
assert(maxNumCompThreads==1,'ControlHarness:TimingThreads','One computational thread is required.');
names={'expert','imitation'};children=struct;certificates=struct;elapsed=struct;
for k=1:2
    name=names{k};child=manifest.children.(name);expected=fullfile(exchange,name,'manifest.json');
    assert(strcmp(child.manifest_path,expected)&&strcmp(file_sha256(expected),child.manifest_sha256), ...
        'ControlHarness:PolicyEvaluationIntegrity','A prepared child manifest changed or escaped its fixed directory.');
    children.(name)=jsondecode(fileread(expected));
    assert(strcmp(children.(name).nonce,nonce)&&strcmp(children.(name).request_sha256,manifest.request_sha256), ...
        'ControlHarness:PolicyEvaluationIntegrity','Child request identity changed.');
    clock=tic;certificates.(name)=control_harness_expert_batch(expected,nonce);elapsed.(name)=toc(clock);
end
expertResult=jsondecode(fileread(children.expert.result_path));imitationResult=jsondecode(fileread(children.imitation.result_path));
try
    benchmark=controller_benchmark(children.expert,children.imitation,expertResult,manifest.benchmark);
catch problem
    benchmark=struct('status','failed','settings',manifest.benchmark,'reason',problem.message,'error_identifier',problem.identifier, ...
        'scope','No speed claim: raw controller benchmark failed. Plant simulation is excluded from attempted timing.');
end
status='completed';
if ~strcmp(certificates.expert.status,'completed')||~strcmp(certificates.imitation.status,'completed')||strcmp(benchmark.status,'failed')
    if certificates.expert.samples+certificates.imitation.samples>0,status='partial';else,status='failed';end
end
result=struct('schema_version',1,'kind','control-matlab-policy-evaluation-result-v1','nonce',nonce, ...
    'request_sha256',manifest.request_sha256,'spec_hash',manifest.spec_hash,'plan_hash',manifest.plan.plan_hash, ...
    'policy_version',manifest.policy_version,'dataset_version',manifest.dataset_version,'expert_version',manifest.expert_version, ...
    'expert_material_version',manifest.expert_material_version,'plant_version',manifest.plant_version,'plant_material_version',manifest.plant_material_version, ...
    'status',status,'expert_result_sha256',file_sha256(children.expert.result_path),'imitation_result_sha256',file_sha256(children.imitation.result_path), ...
    'expert_batch_seconds',elapsed.expert,'imitation_batch_seconds',elapsed.imitation,'controller_benchmark',benchmark, ...
    'matlab_release',version('-release'),'matlab_pid',feature('getpid'),'hardware_execution',false, ...
    'scope','Original bound Plant on identical conditions; imitation uses its own action/state feedback; package construction does not certify stability.');
temporary=fullfile(exchange,['result-' nonce '.tmp']);fid=fopen(temporary,'w','n','UTF-8');
assert(fid>=0,'ControlHarness:PolicyEvaluationWrite','Cannot write authenticated evaluation result.');cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(result,PrettyPrint=true,ConvertInfAndNaN=true));clear cleanup;movefile(temporary,manifest.result_path,'f');
certificate=struct('nonce',nonce,'status',status,'result_path',manifest.result_path, ...
    'samples',certificates.expert.samples+certificates.imitation.samples,'stop_detection','original_S09_local_batches');
end

function benchmark = controller_benchmark(expertManifest,nnManifest,reference,settings)
assert(settings.batch_size==1&&settings.threads==1&&settings.warmup>=1&&settings.repeats>=1, ...
    'ControlHarness:TimingSettings','Timing requires explicit warmup/repeats, batch one and one computational thread.');
count=numel(reference.arrays.trajectory_id);
if count==0,benchmark=struct('status','not_applicable','reason','No valid expert operating contexts');return;end
sources={expertManifest.expert,nnManifest.expert};names={'expert','imitation'};raw=struct;contexts=cell(2,min(count,settings.repeats));
for sourceIndex=1:2
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
for sourceIndex=1:2
    source=sources{sourceIndex};name=names{sourceIndex};nu=numel(source.spec.interface.controls);values=zeros(settings.repeats,1);
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
benchmark=struct('status','completed','settings',settings,'expert',raw.expert,'imitation',raw.imitation, ...
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
