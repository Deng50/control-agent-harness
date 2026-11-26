function report=verify_scenes()
% 从磁盘重新核对文件覆盖、实际执行版本、全程时标、关键物理误差与优化结果。
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(here));out=fullfile(root,'runs','matlab-scenes');
rows=jsondecode(fileread(fullfile(here,'inventory.json')));records=cell(size(rows));sourceFailures={}; models=0;
for k=1:numel(rows)
    r=rows(k);fixed=readReport(out,r.id,'fixed',r.file);original=readReport(out,r.id,'original',r.file);
    assert(fixed.success && strcmp(fixed.executed_sha256,case_sha256(fullfile(here,r.id,r.file))),'修正版报告与文件不匹配：%s',r.source);
    assert(strcmp(original.executed_sha256,r.sha256),'原件版本不匹配：%s',r.source);
    if ~original.success,sourceFailures{end+1}=r.source;end %#ok<AGROW>
    assert(isfile(fullfile(here,r.id,'README.md')));
    m=fixed.metrics;
    if endsWith(r.file,'.slx')
        models=models+1; assert(m.start_s==0 && m.end_s>0);
        for key=string(fieldnames(m.signals))'
            [~,name]=fileparts(r.file);v=readmatrix(fullfile(out,r.id,'fixed',name+"-"+key+".csv"));
            assert(all(isfinite(v),'all') && all(diff(v(:,1))>=0) && v(1,1)==0 && abs(v(end,1)-m.end_s)<1e-6);
            stored=m.signals.(key).final_channels; assert(max(abs(v(end,2:end)-stored(:)'))<1e-7*max(1,max(abs(stored))));
        end
    end
    if isfield(m,'mat_file')
        p=fullfile(out,r.id,'fixed',m.mat_file);info=whos('-file',p);assert(~isempty(info));
    end
    records{k}=struct('case_id',r.id,'file',r.file,'success',true);
end
% 这些门限针对随库的参考工况；新对象应先确定单位、参考和误差预算。
checks=struct;
m=readReport(out,'egr','fixed','NLS.m');checks.egr_interpolation=m.metrics.diagnostics.heldout_rmse_kg_s<.005;
m=readReport(out,'engine','fixed','P_ident.m');checks.engine_speed=m.metrics.fit_rmse<1;
for id=["ident4","ident-root"]
    m=readReport(out,id,'fixed','Model_parameters_Ident.m');checks.(matlab.lang.makeValidName(id))=m.metrics.fit_rmse<1e-8;
end
probes=jsondecode(fileread(fullfile(out,'mpc-matrix.json')));isel=string({probes.mode})=="fixed";
mx=arrayfun(@(p)p.metrics.max_abs,probes);
checks.lifted_prediction=nnz(isel)==8 && all(mx(isel)<1e-11) && nnz(mx(~isel)>1e-11)==7;
checks.mixed_increments=all(arrayfun(@(p)p.metrics.mixed_increments.passed,probes(isel)));
addpath(fullfile(root,'.agents','skills','verify-mpc-prediction','scripts'));
% A=0 时两输入的精确增量矩阵为下三角单位阵与 (C*B+D) 的 Kronecker 积。
md=struct('A',zeros(2),'B',eye(2),'C',[1 -1;0 2],'D',[.5 0;0 -.2],'E',[.3;1]);
P=kron(tril(ones(3)),md.C*md.B+md.D);moves=[.1 -.05;-.3 .2;.15 .4];
ok=verify_increment_matrix(md,P,[.2;.1],[.1;-.1],moves,[0;.1;-.1]);
P(6,4)=P(6,4)+.5;bad=verify_increment_matrix(md,P,[.2;.1],[.1;-.1],moves,[0;.1;-.1]);
checks.mimo_feedthrough=ok.passed && ~bad.passed && bad.first_step==3 && bad.first_channel==2 && abs(bad.signed_error-.1)<1e-12;
for id=["mpc2","pi"]
    if id=="mpc2",f='Run_MPC_For_2nd_order_Plant_auto_tuning.m';else,f='Run_PID_For_2nd_order_Plant_auto_tuning.m';end
    a=readReport(out,id,'fixed',f);d=a.metrics.diagnostics;
    checks.(id+"_tuning")=d.exitflag>0 && d.tuned_cost<d.baseline_cost;
end
for id=["mpc2","mpc55"]
    a=readReport(out,id,'fixed','ADRC_2nd_Order_Kang.slx');s=a.metrics.signals;
    checks.(id+"_adrc")=abs(s.sid154.final-s.sid174.final)<1e-4;
end
a=readReport(out,'mpc1','fixed','LMPC_1st_order_Kang_2.slx');checks.first_order_tracking=abs(a.metrics.signals.sid1737.final-120)<1e-5;
for id=["online-adrc","eturbo"]
    if id=="eturbo",f='Run_eTurbo_with_Torque_mode_2018bp.slx';else,f='RLS_ADRC_2nd_order_Kang.slx';end
    a=readReport(out,id,'fixed',f);v=a.metrics.signals.theta_rls;
    checks.(matlab.lang.makeValidName(id))=numel(v.final_channels)==4 && v.max>v.min;
end
s=readReport(out,'sfunction','fixed','Plant_Model.m');checks.sfunction_euler=s.metrics.res.last<1e-10;
for tag=["U","Ub","V1","V2"]
    a=readReport(out,'pulse','fixed',"RLS_"+tag+".m");d=a.metrics.diagnostics;
    checks.("battery_"+tag)=d.accepted>0 && d.first_physical_sample<=d.total && d.voltage_rmse_V<.03 && abs(d.dt_s-1.33)<1e-6;
end
a=readReport(out,'soc','fixed','SOCekf.m');checks.soc=a.metrics.soc_rmse<.01;
a=readReport(out,'thermal','fixed','Identification_ThermalModel.m');v=a.metrics.diagnostics.rmse_surface_core_K;
checks.thermal_transfer=all(v(2,:)<.5) && a.metrics.diagnostics.ga_exitflag>0;
% 独立逐拍欧拉解验证 GA 目标使用的离散滤波形式，输入包含阶跃和负脉冲。
addpath(fullfile(here,'thermal'),fullfile(here,'pulse'));
q=[zeros(10,1);2*ones(18,1);-ones(12,1)]; x=[7 30 5 12];ambient=8;
[surface,core]=case_temperatures(q,ambient,x,1);z=ambient*ones(2,numel(q));
for k=2:numel(q)
    transfer=(z(1,k-1)-z(2,k-1))/x(1);loss=(z(2,k-1)-ambient)/x(3);
    z(:,k)=z(:,k-1)+[(q(k-1)-transfer)/x(2);(transfer-loss)/x(4)];
end
checks.thermal_filter_equivalence=max(abs([core surface]-z'),[],'all')<1e-9;
% 从已知的两极点电路合成离散系数，再恢复全部五个物理量。
poles=[.6;.98];resistance=[.002;.005];r0=.001;g=resistance.*(1-poles);
coeff=[sum(poles);-prod(poles);-r0;r0*sum(poles)-sum(g);-r0*prod(poles)+g(1)*poles(2)+g(2)*poles(1)];
[rc,k0]=case_rc_params(repmat(coeff,1,5),1.33);
truth=[r0;resistance;-1.33./(log(poles).*resistance)];
checks.rc_inverse=k0==3 && max(abs(rc(:,end)./truth-1))<1e-9;
a=readReport(out,'dekf','fixed','DEKF_Rs_1_10_20.m');v=a.metrics.diagnostics.final_Rs_surface_core_rmse;
checks.dekf=all(v(:,1)>0) && all(v(:,2)<1) && all(v(:,3)<1.5);
assert(all(structfun(@(x)logical(x),checks)),'场景数值验收有失败项');
backup=jsondecode(fileread(fullfile(here,'backups.json')));backupSuccess=0;
for k=1:numel(backup),b=backup(k);z=readReport(out,b.id,'backup',b.archive);assert(strcmp(z.executed_sha256,b.sha256));backupSuccess=backupSuccess+z.success;end
report=struct('version',version('-release'),'scenarios',numel(unique(string({rows.id}))),...
    'passed',true,'fixed_programs',numel(rows),'models',models,'matlab_files',numel(rows)-models,...
    'source_execution_failures',{sourceFailures},'backup_files',numel(backup),'backup_executed_successfully',backupSuccess,...
    'numerical_checks',checks,'programs',{records});
writelines(string(jsonencode(report,PrettyPrint=true)),fullfile(out,'acceptance.json'),Encoding='UTF-8');
fprintf('%d 个正式程序，%d 个独立数值判据，%d 个历史快照已复核。\n',numel(rows),numel(fieldnames(checks)),numel(backup));
end
function s=readReport(out,id,mode,file)
s=jsondecode(fileread(fullfile(out,id,mode,string(file)+".json")));
end
