% 以每循环喷油量为激励，记录转速 rpm，辨识用仿真转速必须与观测时刻对齐。
clc

% design input dither signal for the real plant
T_sample = 0.01;% 采样周期10ms

out=sim('plant_model_with_dither_input','ReturnWorkspaceOutputs','on');

xdata = [out.Fueling_rate_mgPcycle.data,out.Engine_Speed_rpm.data];
ydata = [out.Engine_Speed_rpm.data];


% ********************** using LS for parameters identification******************************%
%x0=[1,0 0,1,0,0];%带拟参数赋初值，多尝试。
x0=[100,0,0.3, 5];
% trust-region-reflective
options=optimset('TolFun',1e-10,'TolX',1e-9,'MaxFunEvals',500,'MaxIter',10, 'Algorithm','levenberg-marquardt', 'display', 'iter');
[x_model,~,~,~,~,~,jac]=lsqcurvefit(@LS_demo_ident,x0,xdata,ydata,[],[],options);%中括号为对待拟参数取值范围进行限定，前者为下限值，后者为上限值。

y_est = LS_demo_ident(x_model,xdata);

R2=corrcoef(ydata,y_est).^2
