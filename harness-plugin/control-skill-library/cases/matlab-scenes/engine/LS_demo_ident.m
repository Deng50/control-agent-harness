function ydata=LS_demo_ident(x,xdata)
% 以每循环喷油量为激励，记录转速 rpm，辨识用仿真转速必须与观测时刻对齐。

% define the time and the corresponding  
T_sample = 0.01;
time = [0*T_sample:1*T_sample:(length(xdata)-1)*T_sample]';

input_data = xdata(:,1);
output_data = xdata(:,2);

K1 = x(1);
K2 = x(2);
eta_i = x(3);
J = x(4);
%K7 = x(7);

% 显式传参，优化器的每次调用均以相同的初值和采样时标运行。
in=Simulink.SimulationInput('demo_ident');
names={'K1','K2','eta_i','J','T_sample','time','input_data','output_data'};
vals={K1,K2,eta_i,J,T_sample,time,input_data,output_data};
for k=1:numel(names),in=in.setVariable(names{k},vals{k});end
in=in.setModelParameter('ReturnWorkspaceOutputs','on'); out=sim(in); ydata=out.out_data.Data;
