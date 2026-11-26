function ydata=HpEGR_MassFlow(x,xdata)
% 按排气压力、进气压力、排温和阀位的列序计算高压 EGR 质量流量。

%
%mc_dot=ydata;
Ahp_EGR_max=x(1);
pem=xdata(:,1);
pim=xdata(:,2);
Tem=xdata(:,3);
Theta_hpEGR=xdata(:,4);

% 正向可压缩流假设：背压不能超过上游压力，开度映射和气体温度须在实数域。
assert(all(pem>0 & pim>=0 & pim<=pem & Tem>0),'压力或温度超出正向流量定义域');
assert(x(3)>0 && all(x(4).*Theta_hpEGR+x(5)>0),'对数阀口映射或气体常数非正');
k2=x(2);
R=x(3);
a1=x(4);
a2=x(5);
a3=x(6);

% Ahp_EGR=Ahp_EGR_max.*(a1.*Theta_hpEGR.^2+a2.*Theta_hpEGR);

% Ahp_EGR=Ahp_EGR_max.*log(a1.*Theta_hpEGR+a2);
 % 阀口闭合时有效面积下限为零，正向压差下不应给出负流量。
 Ahp_EGR=Ahp_EGR_max.*max(0,log(a1.*Theta_hpEGR+a2)+a3);


ydata=Ahp_EGR.*pem.*sqrt(1-(pim./pem).^k2)./sqrt(R.*Tem);
