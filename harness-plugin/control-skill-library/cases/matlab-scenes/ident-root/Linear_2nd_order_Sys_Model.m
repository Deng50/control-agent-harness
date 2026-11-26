function ydata=Linear_2nd_order_Sys_Model(x,xdata)
% 该 SLX 实际有四个标量参数；六系数的通用状态空间对象需使用独立 S-function。
assert(numel(x)==4 && size(xdata,2)==2,'需要 [a c b f0] 和 [输入,输出]');
dt=.01;time=(0:size(xdata,1)-1)'*dt;
in=Simulink.SimulationInput('Linear_2nd_order_model4Ident');
names={'a','c','b','f0','T_sample','time','u','y'};
vals={x(1),x(2),x(3),x(4),dt,time,xdata(:,1),xdata(:,2)};
for k=1:numel(names),in=in.setVariable(names{k},vals{k});end
in=in.setModelParameter('ReturnWorkspaceOutputs','on');out=sim(in);ydata=out.out_data.Data;
end
