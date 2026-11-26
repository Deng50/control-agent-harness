function d=case_thermal_input(group)
% 电流按档案数值及 2.3 Ah 容量使用 A，熵系数由 mV/K 转 V/K。
% 只在电流与温度的共同观测区间取样，避免末点补零或外推。
v=load('Voltage_current_data.mat'); z=load('Temperature_data.mat'); vc=v.(sprintf('VC_data_%d',group)); tc=z.(sprintf('T_data_%d',group));
d.t=(ceil(max(vc(1,1),tc(1,1))):floor(min(vc(end,1),tc(end,1))))';
d.i=interp1(vc(:,1),vc(:,2),d.t); d.v=interp1(vc(:,1),vc(:,3),d.t);
d.ts=interp1(tc(:,1),tc(:,2),d.t); d.tc=interp1(tc(:,1),tc(:,3),d.t);
a=load('OCV_SOC.mat'); b=load('dUdT.mat');
soc=.5+[0;cumsum(d.i(1:end-1))]/(2.3*3600);
ocv=polyval(polyfit(a.SOC,a.OCV0,8),soc);
d.irrev=abs(d.i.*(ocv-d.v)); d.ent=d.i.*polyval(polyfit(b.SOC_ent,b.dUdt,5),soc)/1000;
d.heat=d.irrev+d.ent.*(273.15+d.tc);
assert(all(isfinite([d.t d.i d.v d.ts d.tc d.heat]),'all'));
end
