function voltage=case_rc_voltage(current,measured,ocv,n,param,dt)
% 区间内保持上一拍电流，采样瞬间的欧姆压降使用当前电流。
assert(n>=1 && all(isfinite(param),'all') && all(param>0,'all') && dt>0);
state=zeros(2,1); voltage=zeros(1,n);voltage(1)=measured(1);
for k=2:n
    r=param(2:3,k); c=param(4:5,k); pole=exp(-dt./(r.*c));
    state=pole.*state+r.*(1-pole)*current(k-1);
    voltage(k)=ocv(k)-sum(state)-param(1,k)*current(k);
end
end
