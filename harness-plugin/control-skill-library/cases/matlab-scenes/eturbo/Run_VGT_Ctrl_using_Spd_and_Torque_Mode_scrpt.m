% 辨识参数分别对应转速与进气压力通道，采样周期为 1 ms。

load("model_param2.mat")
x = x_model2;
wn1 = x(1);
k = x(2);
intercept = x(3);
k2 = x(4);


load("model_param6.mat")
T_sample = 0.001;
x = x_model6;
wn= x(1);
kesi = x(2);

wn1 = 10;

load("model_param7.mat")

x = x_model7;
J_TC = x(1);
alpha = x(2);
alpha2 = x(3);


out=sim('Run_eTurbo_with_Torque_mode_2018bp','ReturnWorkspaceOutputs','on');
scene_metrics=struct('end_s',out.tout(end),'dt_s',T_sample,'wn',wn,'damping',kesi);
