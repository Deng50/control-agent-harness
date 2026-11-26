% 此版本实际输入权重为 R1；ADRC 与 MPC 在同一个二阶标定对象上分别运行。

% 即使在空工作区启动，也要为两个模型显式装载四参数标定。
p=load('model_param.mat'); a=p.x_model(1);c=p.x_model(2);b=p.x_model(3);f0=p.x_model(4);T_sample=.01;
Q1 = 1;
Q2 = 0.015;
R1 = 0.1;
outM=sim('LMPC_2nd_order_Kang_2','ReturnWorkspaceOutputs','on');

outA=sim('ADRC_2nd_Order_Kang','ReturnWorkspaceOutputs','on');

scene_metrics=struct('end_s',[outM.tout(end) outA.tout(end)]);
