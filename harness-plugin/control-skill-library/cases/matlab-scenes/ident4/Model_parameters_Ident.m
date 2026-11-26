% 本目录使用四参数 [a,c,b,f0]，速度、位移依次积分；输入幅值与偏置按采数模型核对。
clc; clear;close all;

% 二阶振荡环节
T_sample = 0.01;% 采样周期10ms

load Data4Ident.mat

% 3列分别是：time(s）,u,y

Start_Data_Num = 1;
End_Data_Num = length(Data_collect)/1;

time = Data_collect(Start_Data_Num:End_Data_Num,1);
u = Data_collect(Start_Data_Num:End_Data_Num,2);
y = Data_collect(Start_Data_Num:End_Data_Num,3);


figure(1)

subplot(211);plot(time,u); 
subplot(212);plot(time,y); 


% ********************** using LS for parameters identification******************************%
%%
% model: y_dot_dot = a*y_dot + c*y + b*u + f0
a = -1;
c = -1;
b = 1;
f0 = 0;

x0=[a,c,b,f0];


xdata = [u,y];
ydata = [y];


% trust-region-reflective
options=optimset('TolFun',1e-10,'TolX',1e-9,'MaxFunEvals',500,'MaxIter',200, 'Algorithm','levenberg-marquardt', 'display', 'iter');
[x_model,~,~,~,~,~,jac]=lsqcurvefit(@Linear_2nd_order_Sys_Model,x0,xdata,ydata,[ ],[ ],options);%中括号为对待拟参数取值范围进行限定，前者为下限值，后者为上限值。
y_est = Linear_2nd_order_Sys_Model(x_model,xdata);
R2=corrcoef(ydata,y_est).^2;
save("model_param.mat","x_model")
load("model_param.mat")
a = x_model(1);
c = x_model(2);
b = x_model(3);
f0 = x_model(4);

1
