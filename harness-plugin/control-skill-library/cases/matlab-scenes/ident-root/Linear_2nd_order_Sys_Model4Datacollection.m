% 运行时单独解析本目录的同名模型；原模型端口为标量 a、c、b、f0。
clc;clear;

kesi = 0.7;
wn = 3;
K = 1;

a = -2*kesi* wn;
c = -wn^2;
b = K*wn^2;
f0 = 1;
T_sample = 0.01;

%% 激励对象的参数设置
fluctuation_range  = 1;%方波波动范围
base = 0;%方波的最小值
max_sim_time = 20;
Time = [0: T_sample: max_sim_time]';

% 明确接收返回值；MAT 与表格的列均为真实时标、总输入、位移。
out=sim('Linear_2nd_order_model4DataProduction','ReturnWorkspaceOutputs','on');
input_data=out.input_data;out_data=out.out_data;Time=out_data.Time;

figure(1)
subplot(211)
plot(input_data);
subplot(212)
plot(out_data);

filename = 'Data4Ident.xlsx';
Data_collect = [Time,input_data.Data,out_data.Data];

save("Data4Ident.mat","Data_collect")

writematrix(Data_collect,filename);
