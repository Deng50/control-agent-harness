% 原笔记把“错误公式”再次赋给正确结果，并引用未保存的小写 v。
% matlab.mat 保存 0% SOC 松弛段 V0 及拟合系数；以本段自己的时标作图。
load('matlab.mat','V0','b','c','d','f');
V0=smoothdata(V0(424:end),'movmean',101);
t0=(0:numel(V0)-1)'; R1=b/30;R2=d/30;C1=1/(R1*c);C2=1/(R2*f);
assert(all([R1 R2 C1 C2]>0));plot(t0,V0);xlabel('段内采样点');ylabel('端电压 / V');
