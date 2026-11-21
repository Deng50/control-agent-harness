function certificate = control_harness_design_batch(manifestPath,nonce)
% 单点验算只使用准备阶段复制的固定函数与纯数据文件，不把原始用户目录加入 path。
manifest=jsondecode(fileread(manifestPath));
exchange=fileparts(manifestPath);
bundle=fullfile(exchange,'bundle');
allowed={'prediction_model.m','predict_step.m','mpc_cost.m','mpc_control.m','selfcheck_controller.m', ...
    'design.json','matlab_model.json','case.json','parameters.json'};
assert(manifest.schema_version==1 && strcmp(manifest.kind,'control-matlab-design-check-v1') && ...
    strcmp(manifest.nonce,nonce) && strcmp(manifest.result_path,fullfile(exchange,'result.json')), ...
    'ControlHarness:DesignManifest','Invalid dedicated design exchange.');
entries=dir(bundle);
entries=entries(~ismember({entries.name},{'.','..'}));
assert(~any([entries.isdir]) && isequal(sort({entries.name}),sort(allowed)) && ...
    isequal(sort({manifest.members.name}),sort(allowed)), ...
    'ControlHarness:DesignFiles','Only the fixed design sources and data may enter the solver path.');
for index=1:numel(manifest.members)
    member=manifest.members(index);
    file=fullfile(bundle,member.name);
    assert(strcmp(char(java.io.File(file).getCanonicalPath()),file) && strcmp(file_hash(file),member.sha256), ...
        'ControlHarness:DesignFileIdentity','A registered design member changed.');
end
report=control_harness_design_check(bundle,fullfile(bundle,'case.json'),fullfile(bundle,'parameters.json'),nonce);
result=struct('schema_version',1,'nonce',nonce,'request_sha256',manifest.request_sha256, ...
    'plan_hash',manifest.plan.plan_hash,'spec_hash',manifest.spec_hash,'report',report);
% 写完后关闭文件，先发布完整结果，再把同一文件摘要交给 Host，避免将半成品归档。
temporary=fullfile(exchange,['result-' nonce '.tmp']);
fid=fopen(temporary,'wb');
assert(fid>=0,'ControlHarness:DesignResult','Cannot create the design result.');
cleanup=onCleanup(@() fclose(fid));
bytes=unicode2native(jsonencode(result),'UTF-8');
assert(fwrite(fid,bytes,'uint8')==numel(bytes),'ControlHarness:DesignResult','Incomplete design result.');
clear cleanup;
[ok,message]=movefile(temporary,manifest.result_path);
assert(ok,'ControlHarness:DesignResult','%s',message);
certificate=struct('nonce',nonce,'status','completed','result_path',manifest.result_path, ...
    'result_sha256',file_hash(manifest.result_path));
end

function digest=file_hash(file)
fid=fopen(file,'rb');
assert(fid>=0,'ControlHarness:DesignFiles','Cannot read the fixed bundle member.');
cleanup=onCleanup(@() fclose(fid));
hasher=javaMethod('getInstance','java.security.MessageDigest','SHA-256');
while true
    bytes=fread(fid,1024*1024,'*uint8');
    if isempty(bytes),break;end
    hasher.update(typecast(bytes,'int8'));
end
digest=lower(reshape(dec2hex(typecast(hasher.digest(),'uint8'),2).',1,[]));
end
