function results = run_scene(id, materials, mode, only)
% 从素材复制配套数据，在临时工作目录实跑；同名模型不会跨案例解析。
% 使用 matlab -batch "addpath('cases/matlab-scenes'); run_scene('mpc1')"。
arguments
    id (1,1) string
    materials (1,1) string = ""
    mode (1,1) string {mustBeMember(mode,["fixed","original","backup"])} = "fixed"
    only (1,1) string = ""
end
assert(batchStartupOptionUsed,'请使用独立 matlab -batch 会话，原脚本会清空基础工作区。');
here=fileparts(mfilename('fullpath')); lib=fileparts(fileparts(here));
if materials=="", materials=fullfile(fileparts(lib),'素材'); end
[found,sourceDir]=fileattrib(materials);assert(found,'素材目录不存在');materials=string(sourceDir.Name);
inventory=jsondecode(fileread(fullfile(here,'inventory.json')));
data=jsondecode(fileread(fullfile(here,'inputs.json')));
ids=unique(string({inventory.id}),'stable');
if id=="all", for k=1:numel(ids), run_scene(ids(k),materials,mode,only); end; results=[];return; end
assert(any(ids==id),'未知案例'); rows=inventory(string({inventory.id})==id);
folder=tempname; mkdir(folder); old=pwd; cd(folder);
clean=onCleanup(@() finishScene(old,folder)); %#ok<NASGU>
outdir=fullfile(lib,'runs','matlab-scenes',id,mode); if ~isfolder(outdir),mkdir(outdir);end
for k=1:numel(rows)
    if mode~="fixed", src=fullfile(materials,rows(k).source); else, src=fullfile(here,id,rows(k).file); end
    copyfile(src,fullfile(folder,rows(k).file));
end
% 附加适配代码与 Level-2 S-function 的最小宿主单独存放。
helpers=[dir(fullfile(here,id,'case_*.m'));dir(fullfile(here,'shared','case_*.m'))];
for h=helpers', copyfile(fullfile(h.folder,h.name),folder);end
infiles=data.(replace(id,'-','_'));
for k=1:numel(infiles)
    dest=fullfile(folder,infiles(k).target); if ~isfolder(fileparts(dest)),mkdir(fileparts(dest));end
    copyfile(fullfile(materials,infiles(k).source),dest);
    assert(strcmp(case_sha256(dest),infiles(k).sha256),'案例输入与已记录的版本不同：%s',infiles(k).source);
end
if isfolder('data'),addpath(fullfile(folder,'data'));end
if isfolder('数据'),addpath(fullfile(folder,'数据'));end
if mode=="backup"
    allback=jsondecode(fileread(fullfile(here,'backups.json')));
    rows=allback(string({allback.id})==id);
end
set(groot,'defaultFigureVisible','off');
results={};
for k=1:numel(rows)
    f=string(rows(k).file); if only~="" && ~contains(f,only),continue;end
    [~,name,ext]=fileparts(f);
    % 各入口均从独立原标定开始，恢复前一个采集/辨识脚本可能生成的同名数据。
    for j=1:numel(infiles)
        if any(string(infiles(j).target)==["model_param.mat","Data4Ident.mat","tempParameters.mat"])
            copyfile(fullfile(materials,infiles(j).source),fullfile(folder,infiles(j).target));
        end
    end
    evalin('base','clearvars'); bdclose('all'); close all force; rng(876,'twister');
    if mode=="backup",copyfile(fullfile(materials,rows(k).source),fullfile(folder,f));rehash;end
    result=struct('source',rows(k).source,'file',f,'mode',mode,'matlab',version('-release'),'success',false,...
        'executed_sha256',case_sha256(fullfile(folder,f)),'source_sha256',rows(k).sha256);
    started=tic;
    try
        setupWorkspace(id);
        if mode=="backup" && rows(k).archive=="SOCekf.asv"
            % 快照硬编码 44724 点；HPPC.mat 提供同长、同变量名的存档电流。
            evalin('base',"load('HPPC.mat','dstI')");
        end
        if ext==".slx", result.metrics=runModel(name,outdir);
        else, result.metrics=runProgram(name,id,outdir,mode);end
        result.success=true;
    catch ex
        result.error=string(getReport(ex,'basic','hyperlinks','off'));
        result.stack=string({ex.stack.name});
    end
    result.seconds=toc(started);
    reportName=f;if mode=="backup",reportName=string(rows(k).archive);end
    path=fullfile(outdir,reportName+".json"); writelines(string(jsonencode(result,PrettyPrint=true)),path,Encoding="UTF-8");
    fprintf('CASE %s %s %d (%.2fs)\n',id,f,result.success,result.seconds);
    if ~result.success,disp(result.error);end
    results{end+1}=result; %#ok<AGROW>
end
if mode=="fixed" && nargout==0
    assert(~isempty(results) && all(cellfun(@(r)r.success,results)),'案例有失败或未匹配的程序，请查看对应 JSON。');
end
end

function finishScene(old,folder)
bdclose('all');close all force;cd(old);
for d=["data","数据"], p=fullfile(folder,d); if isfolder(p) && contains(path,p),rmpath(p);end;end
if isfolder(folder),rmdir(folder,'s');end
end

function setupWorkspace(id)
% 模型单独启动所需的初值来自各自脚本，标定文件按场景复制。
cmd="T_sample=.01; a=-4.2;c=-9;b=9;f0=1;Q1=1;Q2=.015;R=.1;R1=.1;base=0; fluctuation_range=1;max_sim_time=20;";
if isfile('model_param.mat'),cmd=cmd+"load model_param.mat; a=x_model(1);c=x_model(2);b=x_model(3);f0=x_model(4);";end
if isfile('Data4Ident.mat'),cmd=cmd+"load Data4Ident.mat;time=Data_collect(:,1);u=Data_collect(:,2);y=Data_collect(:,3);";end
if id=="engine", cmd=cmd+"srcout=sim('plant_model_with_dither_input','ReturnWorkspaceOutputs','on');time=srcout.Engine_Speed_rpm.Time;input_data=srcout.Fueling_rate_mgPcycle.Data;output_data=srcout.Engine_Speed_rpm.Data; K1=150;K2=.02;eta_i=.35;J=10;";end
if id=="eturbo",cmd=cmd+"load model_param2.mat;load model_param6.mat;load model_param7.mat;k=x_model2(2);intercept=x_model2(3);k2=x_model2(4);wn=x_model6(1);kesi=x_model6(2);wn1=10;J_TC=x_model7(1);alpha=x_model7(2);alpha2=x_model7(3);T_sample=.001;";end
if id=="soc",cmd=cmd+"load DSTI.mat dstI; load VandI.mat I V;tuvar=[(0:numel(I)-1)' I(:)];";end
if id=="thermal",cmd=cmd+"load tempParameters.mat;Tf=8;d=case_thermal_input(1);time=d.t;T_core=d.tc;T_surf=d.ts;Q_total=d.heat';";end
evalin('base',cmd);
end

function m=runModel(name,outdir)
% 从物理端口记录实际信号，同时保留仿真自身的导出项。
load_system(name); targets=struct('ADRC_2nd_Order_Kang',[154 174], ...
    'LMPC_1st_order_Kang_2',[1654 1737 1676],'LMPC_2nd_order_Kang_2',[156 189 157],...
    'RLS_ADRC_2nd_order_Kang',[195 639 196 460],...
    'Run_eTurbo_with_Torque_mode_2018bp',[1896 2347 2337 1897],...
    'thermalModel_test',[309 279],'two_RC_model',[143 157],...
    'PID2_2nd_order_Kang_2',[365 370],'PID_for_2nd_order_sysm',[375 430],...
    'Linear_2nd_order_model4DataProduction',[273 251], 'Linear_2nd_order_model4Ident',[251],...
    'demo_ident',[181],'plant_model_with_dither_input',[39]);
% 2018b 的转速辨识快照使用另一子系统 SID，并额外引用旧版摩擦参数。
if name=="demo_ident" && getSimulinkBlockHandle('demo_ident/Subsystem1')>0
    targets.demo_ident=133;
end
for sid=targets.(name)
    blk=Simulink.ID.getFullName(char(name+":"+sid)); ports=get_param(blk,'PortHandles');
    if isempty(ports.Outport) || strcmp(get_param(blk,'Commented'),'on'),continue;end
    port=ports.Outport(1); if get_param(port,'Line')==-1,continue;end
    set_param(port,'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName',char("sid"+sid));
    if any(sid==[196 1897]) && numel(ports.Outport)>=2
        set_param(ports.Outport(2),'DataLogging','on','DataLoggingNameMode','Custom','DataLoggingName','theta_rls');
    end
end
in=Simulink.SimulationInput(name); in=in.setModelParameter('ReturnWorkspaceOutputs','on','SignalLogging','on','SignalLoggingName','scene_logs','SaveTime','on','LimitDataPoints','off','TimeSaveName','tout');
out=sim(in); t=out.tout; assert(~isempty(t) && all(isfinite(t)) && t(end)>t(1));
m=struct('time_samples',numel(t),'start_s',t(1),'end_s',t(end),'signals',struct);
if any(string(who(out))=="scene_logs")
    ds=out.scene_logs;
    for j=1:ds.numElements
        sig=ds.getElement(j); v=sig.Values; if ~isa(v,'timeseries'),continue;end
        val=double(v.Data); assert(isreal(val) && all(isfinite(val),'all'),'模型信号非有限'); 
        key=matlab.lang.makeValidName(sig.Name); m.signals.(key)=struct('min',min(val,[],'all'),'max',max(val,[],'all'),'final',val(end));
        values=reshape(val,[],numel(v.Time))'; if size(val,1)==numel(v.Time),values=reshape(val,numel(v.Time),[]);end
        m.signals.(key).final_channels=values(end,:);
        writematrix([v.Time(:),values],fullfile(outdir,name+"-"+key+".csv"));
    end
end
assert(~isempty(fieldnames(m.signals)),'无可验收的已连接输出'); 
end

function m=runProgram(name,id,outdir,mode)
switch name
    case "HpEGR_MassFlow"
        cmd="D=readmatrix('HP_EGR valve.xls');D=D(all(isfinite(D(:,1:5)),2),:);res=HpEGR_MassFlow([.0074 2.2 287 .1 1 0],D(:,1:4));";
    case "LS_demo_ident"
        cmd="res=LS_demo_ident([150,.02,.35,10],[input_data output_data]);";
    case "Linear_2nd_order_Sys_Model"
        if mode~="fixed" && id=="ident-root",args="[0 1 -9 -4.2 0 9]";else,args="[-4.2 -9 9 1]";end
        cmd="res=Linear_2nd_order_Sys_Model("+args+",[u y]);";
    case "MPC_QP"
        cmd="res=MPC_QP(10,diag([1 .015]),.1,.01,[0 1;-9 -4.2],[0;9],[0;1],[.1;.2],[0;0],[1;0],0);";
    case "MPC_QP_1st"
        cmd="res=MPC_QP_1st(10,1,.1,.01,-2,3,1,.2,1,0);";
    case {"ModelTest","model_copy"}
        cmd="load LiBatt_PulseDataEVE.mat; cur=-PulseData.current(1:100); vv=PulseData.voltage(1:100); z=ones(1,100);res="+name+"(cur,vv,vv,100,.001*z,.002*z,.001*z,1000*z,4000*z);";
    case "estimate_function"
        cmd="res=estimate_function(T_surf,T_core,Tf,Q_total,[Rc Cc Rs Cs],1);";
    case "Plant_Model"
        cmd="res=case_sfunction();";
    otherwise
        cmd=name+";";
end
evalin('base',cmd);
vars=string(evalin('base','who')); m=struct;
for key=["res","R2","x_model","x_opt","fval","HpEGR_MassFlow_est","y_est","Data_collect","V_error_mean","V_error_max","theta_est","Tc_pre","Ts_pre","Socekf","R1","R2","C1","C2","rc_param"]
    if any(key==["R1","R2","C1","C2"]) && ~any(id==["pulse","soc"]) && ~(key=="R2" && ismember("y_est",vars)),continue;end
    if ~ismember(key,vars),continue;end
    v=evalin('base',key);if ~isnumeric(v) || isempty(v),continue;end
    m.(key)=struct('size',size(v),'finite_real',isreal(v)&&all(isfinite(v),'all'),'last',real(v(end)));
    if numel(v)<=10 && isreal(v) && all(isfinite(v),'all'),m.(key).values=v;end
end
if ismember('y_est',vars),m.fit_rmse=evalin('base','sqrt(mean((y_est(:)-ydata(:)).^2))');end
if ismember('HpEGR_MassFlow_est',vars),m.flow_rmse=evalin('base','sqrt(mean((HpEGR_MassFlow_est(:)-W_HP_EGR(:)).^2))');end
if name=="SOCekf",m.soc_rmse=evalin('base','sqrt(mean((Socekf-Soc).^2))');end
if name=="DEKF_Rs_1_10_20",m.core_rmse=evalin('base','sqrt(mean((Tc_pre-Tc).^2))');m.surface_rmse=evalin('base','sqrt(mean((Ts_pre-Ts).^2))');end
if name=="Identification_ThermalModel",m.validation_core_rmse=evalin('base','sqrt(mean((Tc_pre-T_core).^2))');m.validation_surface_rmse=evalin('base','sqrt(mean((Ts_pre-T_surf).^2))');end
if name=="Understand_Newtons_Law",m.axes=numel(findall(groot,'Type','axes'));assert(m.axes>=3);end
if any(name==["Run_MPC_For_2nd_order_Plant","Run_VGT_Ctrl_using_Spd_and_Torque_Mode_scrpt"]) && ismember('scene_metrics',vars)
    assert(all(isfinite(evalin('base','scene_metrics.end_s'))));
end
if ismember('scene_metrics',vars),m.diagnostics=evalin('base','scene_metrics');end
if ismember('Data_collect',vars)&&contains(name,'Datacollection'),writematrix(evalin('base','Data_collect'),fullfile(outdir,'collection.csv'));end
if mode=="fixed"
    fields=fieldnames(m);for j=1:numel(fields),v=m.(fields{j});if isstruct(v)&&isfield(v,'finite_real'),assert(v.finite_real,'输出 %s 为复数或含非有限值',fields{j});end;end
    % 将优化结果、实测/预测序列和物理量一并留存，供下游 MATLAB 工作流读取。
    switch name
        case "NLS", keep=["x_HpEGR_MassFlow","xdata","ydata","HpEGR_MassFlow_est","holdout","pred","xf"];
        case {"Model_parameters_Ident","P_ident"}, keep=["x_model","xdata","ydata","y_est"];
        case "Linear_2nd_order_Sys_Model4Datacollection"
            keep="Data_collect";copyfile('Data4Ident.xlsx',fullfile(outdir,'Data4Ident.xlsx'));
        case {"RLS_U","RLS_Ub","RLS_V1","RLS_V2"},keep=["time","idx","accepted","volt","V_error","R0","R1","R2","C1","C2"];
        case "SOCekf",keep=["t","i","Soc","Socekf","Uoc","Uocekf"];
        case "Untitled",keep=["t0","V0","R1","R2","C1","C2"];
        case "Identification_ThermalModel",keep=["rc_param","err","time","Ts_pre","Tc_pre","T_core","T_surf"];
        case "DEKF_Rs_1_10_20",keep=["initials","results","history","Ts","Tc"];
        case {"Run_MPC_For_2nd_order_Plant_auto_tuning","Run_PID_For_2nd_order_Plant_auto_tuning"},keep=["x_opt","fval","scene_metrics"];
        otherwise,keep=strings(1,0);
    end
    if ~isempty(keep)
        payload=struct;for key=keep,payload.(key)=evalin('base',key);end
        m.mat_file=name+".mat";save(fullfile(outdir,m.mat_file),'-struct','payload');
    end
end
end
