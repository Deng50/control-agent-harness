% 启用的 PI 支路以 Q2 为比例增益、R 为积分增益；性能积分使用记录时间轴。

%% 简化版MPC权重优化
clear; clc;
load("model_param.mat")
a = x_model(1);
c = x_model(2);
b = x_model(3);
f0 = x_model(4);
T_sample = 0.01;

% 优化配置
options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'MaxIterations', 300);

% 定义优化问题
problem.objective = @(x) mpc_cost_function(x);
problem.x0 = [1, 1];  % [Q2, R]
problem.lb = [0.00001, 0.000001];
problem.ub = [10000, 10000];
problem.solver = 'fmincon';
problem.options = options;

% 运行优化
% 以同一对象、参考和积分口径对照初始参数与搜索结果。
baseline_cost=problem.objective(problem.x0);
[x_opt, fval, exitflag] = fmincon(problem);
assert(isfinite(fval) && fval<=baseline_cost+1e-7,'整定后目标恶化');
scene_metrics=struct('baseline_cost',baseline_cost,'tuned_cost',fval,'exitflag',exitflag);

fprintf('优化完成！\n');
fprintf('最优Q2: %.4f\n', x_opt(1));
fprintf('最优R: %.4f\n', x_opt(2));

%% 成本函数
function cost = mpc_cost_function(x)
    % 设置参数到工作空间
    Q1 = 1;
    Q2 = x(1);
    R = x(2);
    T_sample = 0.01;
    
    assignin('base', 'Q1', Q1);
    assignin('base', 'Q2', Q2);
    assignin('base', 'R', R);
    
    % 运行Simulink
    sim_out = sim('PID2_2nd_order_Kang_2.slx', 'StopTime', '10');
    
    % 提取数据
    % 以实际输出时标积分。首列为位置，控制量采用实际记录。
    y=sim_out.y.Data(:,1); r=sim_out.r.Data(:,1); t=sim_out.y.Time;
    u=sim_out.u.Data(:,1); tu=sim_out.u.Time;
    cost=trapz(t,t.*(r-y).^2)+0.1*trapz(tu,u.^2);
    assert(isfinite(cost),'整定目标非有限');
end
