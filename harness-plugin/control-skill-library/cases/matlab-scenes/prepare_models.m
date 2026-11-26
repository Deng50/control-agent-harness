function prepare_models(materials,only)
% 从原 SLX 重新生成案例副本的中文图注、内嵌预测修复；输出始终位于案例库。
here=fileparts(mfilename('fullpath')); if nargin<1 || strlength(materials)==0,materials=fullfile(fileparts(fileparts(fileparts(here))),'素材');end
if nargin<2,only="";end
rows=jsondecode(fileread(fullfile(here,'inventory.json')));
tmp=tempname;mkdir(tmp);cleanupTmp=onCleanup(@()rmdir(tmp,'s')); %#ok<NASGU>
for row=rows'
    [~,name,ext]=fileparts(row.file);if ~strcmp(ext,'.slx'),continue;end
    if strlength(only)>0 && ~strcmp(name,only),continue;end
    source=fullfile(materials,row.source);dest=fullfile(here,row.id,row.file);
    local=fullfile(tmp,row.file);copyfile(source,local);load_system(local);cleanup=onCleanup(@()close_system(name,0));
    root=sfroot; charts=root.find('-isa','Stateflow.EMChart');
    for ch=charts'
        if ~startsWith(ch.Path,[name '/']),continue;end
        s=ch.Script;
        if contains(s,'stop = start + stateDim - 1;')
            ch.Script=strrep(s,'stop = start + stateDim - 1;',sprintf('%% 第 j 个增量对所有剩余预测步均有影响。\n    stop = predictionHorizon * stateDim;'));
        elseif contains(s,'Theta_k_1')
            ch.Script=strrep(s,'%Kang Song',sprintf('%%Kang Song\n%% 遗忘因子用于协方差更新；控制器采用新参数还要通过外层切换判据。'));
        end
    end
    switch name
        case 'two_RC_model', note='二阶 RC：核对工况电流的时标与放电符号，再读取两路极化电压和 SOC。';
        case 'thermalModel_test',note='两节点热模型：以热功率为输入，对照核心和表面温度；参数顺序见本案例 README。';
        case 'demo_ident',note='辨识：同一喷油序列下对比预测与观测转速，目标量为 rpm。';
        case 'plant_model_with_dither_input',note='采数：供油量阶梯激励，记录每循环毫克数与发动机转速。';
        case 'Linear_2nd_order_model4Ident',note='辨识：工作区输入按 time 对齐，积分顺序为加速度、速度、位移。';
        case 'Linear_2nd_order_model4DataProduction',note='采数：记录施加的总输入和积分器位置输出，两者共用仿真时间轴。';
        case 'ADRC_2nd_Order_Kang',note='自抗扰闭环：检查参考、观测器扰动估计和实际输出，带宽与采样周期共同决定动态响应。';
        case 'RLS_ADRC_2nd_order_Kang',note='在线辨识：先看 RLS 估计，再看学习使能与误差判据；估计值和控制采用值分别记录。';
        case 'Run_eTurbo_with_Torque_mode_2018bp'
            % 四个“真值”From 在源档案中没有生产者，只通向 Scope2。
            % 保留已连接的估计通道，避免把缺测曲线画成零真值。
            for sid=[1890 1891 1892 1893]
                mux=Simulink.ID.getFullName(sprintf('%s:%d',name,sid));
                ports=get_param(mux,'PortHandles');
                missingLine=get_param(ports.Inport(1),'Line');
                missing=get_param(missingLine,'SrcBlockHandle');
                validLine=get_param(ports.Inport(2),'Line'); srcPort=get_param(validLine,'SrcPortHandle');
                outLine=get_param(ports.Outport(1),'Line'); dstPort=get_param(outLine,'DstPortHandle');
                assert(strcmp(get_param(missing,'BlockType'),'From') && numel(dstPort)==1);
                delete_line(missingLine);delete_line(validLine);delete_line(outLine);
                delete_block(missing);delete_block(mux);add_line(name,srcPort,dstPort,'autorouting','on');
            end
            note='电增压：转矩驱动转速，再影响压力。Scope2 只显示可取得的参数，原档案没有四个参数真值。';
        case 'LMPC_1st_order_Kang_2',note='一阶 MPC：参考、实际输出、控制增量逐拍核对；独立 .m 函数与图内实现分别验证。';
        case 'LMPC_2nd_order_Kang_2',note='二阶增量 MPC：后续控制增量须传播至时域末端；R 与 R1 由本模型连线确定。';
        otherwise,note='PI 闭环：Q2 与 R 对应启用 PI 支路的比例、积分增益，记录实际输出及操纵量。';
    end
    if strcmp(name,'Linear_2nd_order_model4Ident')
        % 原采数端为零阶保持；辨识输入在阶跃间使用同一保持约定。
        set_param(Simulink.ID.getFullName([name ':129']),'OutputAfterFinalValue','Holding final value','Interpolate','off');
    end
    an=Simulink.Annotation(name,note); an.Position=[15 -100 720 -55];
    save_system(name,local);clear cleanup;copyfile(local,dest);
end
end
