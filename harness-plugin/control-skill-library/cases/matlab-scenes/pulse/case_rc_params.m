function [params,start,accepted]=case_rc_params(coeff,T)
% 由离散 ARX 的两个稳定极点和部分分式恢复 R0、两支 R/C。
% 放电电流为正，端电压的直接通道为 -R0；初次得到物理解之前不报电压。
n=size(coeff,2); params=nan(5,n); accepted=false(1,n);
for k=3:n
    a=coeff(:,k); p=sort(roots([1 -a(1) -a(2)]));
    if ~isreal(p) || p(1)<=0 || p(2)>=1 || diff(p)<1e-8,continue;end
    r0=-a(3); total=r0*sum(p)-a(4); weighted=a(5)+r0*prod(p);
    g1=(weighted-p(1)*total)/diff(p); g2=total-g1;
    r=[g1;g2]./(1-p); c=-T./(log(p).*r);
    one=[r0;r;c];
    if all(isfinite(one)) && all(one>0),params(:,k)=one;accepted(k)=true;end
end
start=find(accepted,1); assert(~isempty(start),'该回归未产生正实双 RC 参数');
for k=start+1:n
    if ~accepted(k), params(:,k)=params(:,k-1);end
end
end
