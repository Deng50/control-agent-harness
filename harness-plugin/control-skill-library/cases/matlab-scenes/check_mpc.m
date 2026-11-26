function results=check_mpc(materials)
% 对各代文件本身提取预测矩阵，包含注释掉的对照支路；仿真运行状态另见逐文件报告。
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(here));
if nargin<1,materials=fullfile(fileparts(root),'素材');end
rows=jsondecode(fileread(fullfile(here,'inventory.json')));addpath(fullfile(here,'shared'));
addpath(fullfile(root,'.agents','skills','verify-mpc-prediction','scripts'));
old=pwd;tmp=tempname;mkdir(tmp);cd(tmp);clean=onCleanup(@()finish(old,tmp)); %#ok<NASGU>
results={};
for row=rows'
    [~,name,ext]=fileparts(row.file);
    if ~(startsWith(name,'LMPC_') || startsWith(name,'MPC_QP') || startsWith(name,'PID2_') || startsWith(name,'PID_for_')),continue;end
    for mode=["original","fixed"]
        if mode=="original",src=fullfile(materials,row.source);else,src=fullfile(here,row.id,row.file);end
        if strcmp(ext,'.m'),scripts={fileread(src)};paths={row.file};
        else
            load_system(src);charts=sfroot().find('-isa','Stateflow.EMChart');
            scripts={};paths={};
            for c=charts',if startsWith(c.Path,[name '/']) && contains(c.Script,'inputMatrix'),scripts{end+1}=c.Script;paths{end+1}=c.Path;end;end %#ok<AGROW>
            close_system(name,0);
        end
        for i=1:numel(scripts)
            nx=2;if contains(name,'1st'),nx=1;end
            m=case_mpc_check(scripts{i},nx,strcmp(name,'MPC_QP_1st'));
            results{end+1}=struct('source',row.source,'mode',mode,'implementation',paths{i},'metrics',m); %#ok<AGROW>
            if mode=="fixed",assert(m.max_abs<1e-11,'预测矩阵仍不匹配：%s',paths{i});end
        end
    end
end
dest=fullfile(root,'runs','matlab-scenes','mpc-matrix.json');
writelines(string(jsonencode(results,PrettyPrint=true)),dest,Encoding='UTF-8');fprintf('%d 组控制器矩阵核验完成\n',numel(results));
end
function finish(old,tmp)
cd(old);rmdir(tmp,'s');
end
