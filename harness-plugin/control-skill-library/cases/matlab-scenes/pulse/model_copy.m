function Vt=model_copy(I,Ut,uoc,N,R0,R1,R2,C1,C2,T)
% 放电为正，以输入时标（秒）恢复端电压；九参数历史调用沿用 1 秒。
if nargin<10,T=1;end
param=[R0(:) R1(:) R2(:) C1(:) C2(:)]';
Vt=case_rc_voltage(I,Ut,uoc,N,param,T);
end
