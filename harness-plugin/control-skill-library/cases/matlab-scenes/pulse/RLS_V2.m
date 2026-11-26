% 约定放电电流为正；先辨识电压回归系数，再检查可实现的两个正时间常数。
clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%加载数据%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('LiBatt_PulseDataEVE.mat');
time = PulseData.time;
Vol = PulseData.voltage;
Cur = -PulseData.current; 
Temp = -PulseData.temperature;
Capacity = 287; % 电池容量
Len = round(length(Vol) * 0.8); % 取前80%数据进行辨识，SOC低于20%时已辨识不准

% 参数初始值（包含 k1-k5）
cs0 = [0; 0; 0; 0; 0]; % 初始化参数矩阵，包含 k1-k5
cs = [cs0, cs0, zeros(5, Len-2)]; % 初始化参数存储矩阵
T=median(diff(time(1:Len))); % 采样时间

%%%%%%%%%初始化数据，通过 MATLAB 自带函数进行拟合%%%%%%%%%%%%%%%
ocv_lut = [2.789261, 3.20478, 3.242465, 3.275661, 3.289715, 3.290776, 3.293613, 3.329685, 3.330853, 3.33225, 3.380374];
soc_lut = 0.02439:0.09756:1;
pfit = polyfit(soc_lut, ocv_lut, 7);

SOC = zeros(Len, 1);
SOC(1) = 1;
Vocv = zeros(Len, 1);
Vocv(1) = polyval(pfit, SOC(1));
for i = 2:Len
    SOC(i) = SOC(i-1) - Cur(i) * T / 3600 / Capacity;
    Vocv(i) = polyval(pfit, SOC(i));
end

U = Vol(1:Len); % U是端电压

% 初始化 fai 矩阵（数据向量矩阵），包含常数项 1
F1 = [0; 0; Cur(1); 0; 0]; % 初值 k=1 时刻
F2 = [U(1); 0; Cur(2); Cur(1); 0]; % 初值 k=2 时刻
fai = [F1, F2, zeros(5, Len-2)]; % 初始化数据向量矩阵
Pk_1 = 10^4 * eye(5, 5); % 初始化协方差矩阵
lamda = 1; % 遗忘因子
e_ksum = zeros(1, Len-2); % 初始化误差存储

%%%%%%%%%%%%%RLS辨识开始%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 3:Len
    fai(:, k) = [U(k-1)-Vocv(k); U(k-2)-Vocv(k); Cur(k); Cur(k-1); Cur(k-2)]; % 数据向量

    % 当前误差
    e_k = U(k)-Vocv(k)-fai(:,k)'*cs(:,k-1); % 计算当前误差
    e_ksum(:, k) = e_k;

    % 递归最小二乘法更新所有参数 k1-k5
    K_k = Pk_1 * fai(:, k) / (lamda + fai(:, k)' * Pk_1 * fai(:, k)); % 增益矩阵
    cs(:, k) = cs(:, k-1) + K_k * e_k; % 更新参数矩阵 k1-k5

    % 更新协方差矩阵
    P_k = (Pk_1 - K_k * fai(:, k)' * Pk_1) / lamda;
    Pk_1 = P_k; % 更新协方差矩阵
end


% 物理可实现性筛选：两实极点在 (0,1)，R、C 为正。失效时保持上次有效估计。
[rc,k0,accepted]=case_rc_params(cs(end-4:end,:),T);
idx=k0:Len; R0=rc(1,idx);R1=rc(2,idx);R2=rc(3,idx);C1=rc(4,idx);C2=rc(5,idx);
volt=ModelTest(Cur(idx),Vol(idx),Vocv(idx),numel(idx),R0,R1,R2,C1,C2,T);
V_error=Vol(idx)'-volt;V_error_mean=mean(abs(V_error));V_error_max=max(abs(V_error));
scene_metrics=struct('first_physical_sample',k0,'accepted',nnz(accepted),'total',Len,'dt_s',T,'voltage_rmse_V',sqrt(mean(V_error.^2)));
plot(time(idx),[Vol(idx),volt']);legend('实测端电压','递推端电压');xlabel('秒');ylabel('伏');
