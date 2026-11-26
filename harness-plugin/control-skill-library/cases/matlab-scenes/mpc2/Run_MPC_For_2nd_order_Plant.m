% 二阶对象使用实际连线的 R，采样周期 0.01 s；区分状态跟踪与执行器消耗。

% 即使在空工作区启动，也要为两个模型显式装载四参数标定。
p=load('model_param.mat'); a=p.x_model(1);c=p.x_model(2);b=p.x_model(3);f0=p.x_model(4);T_sample=.01;
Q1 = 1;
Q2 = 0.015;
R = 0.1;
outM=sim('LMPC_2nd_order_Kang_2','ReturnWorkspaceOutputs','on');

outA=sim('ADRC_2nd_Order_Kang','ReturnWorkspaceOutputs','on');

scene_metrics=struct('end_s',[outM.tout(end) outA.tout(end)]);
