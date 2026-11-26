function delta_u_k = MPC_QP_1st(predictionHorizon, stateWeight, inputWeight, samplingTime, A_continuous, B_continuous, C_continuous, currentState, reference, lastControl)
% 一阶增量预测把上一实际动作并入状态，每拍只执行最优序列的首个增量。
% MPC_QP_1st: 基于QP二次规划的一阶(或一般)系统增量式模型预测控制
% 
% 输入参数:
% - predictionHorizon: 预测步长 (N)
% - stateWeight: 状态/输出误差的权重矩阵 (Q)
% - inputWeight: 控制增量的权重矩阵 (R)
% - samplingTime: 离散化采样时间 (T_s)
% - A_continuous, B_continuous, C_continuous: 连续时间系统状态空间矩阵
% - currentState: 当前时刻系统的状态值 (x(k))
% - reference: 期望的参考输出值 (r(k))
% - lastControl: 上一时刻的控制量 u(k-1)
%
% 输出参数:
% - delta_u_k: 当前时刻计算出的最优控制增量 delta_u(k)

    % 1. 系统离散化 (采用欧拉法)
    stateDim = size(A_continuous, 1);
    inputDim = size(B_continuous, 2);
    outputDim = size(C_continuous, 1);
    
    A_discrete = eye(stateDim) + A_continuous .* samplingTime;
    B_discrete = B_continuous .* samplingTime;
    C_discrete = C_continuous;
    
    % 2. 构造增量式模型 (Augmented Model)
    % 新的状态变量: x_aug(k) = [delta_x(k); y(k)]
    % 其中 delta_x(k) = x(k) - x(k-1), y(k) = C * x(k)
    % 但更方便的做法是直接构建关于 delta_u 的预测模型:
    % X_predict = F * x(k) + Phi * U
    % U = u(k-1) * ones(N, 1) + L * Delta_U
    % Y_predict = C_aug * X_predict
    
    % 更标准的方法: 构建增广矩阵 A_aug, B_aug, C_aug
    % x_aug(k+1) = A_aug * x_aug(k) + B_aug * delta_u(k)
    % y(k) = C_aug * x_aug(k)
    % 假设系统状态就是真实的 x(k)，为了不依赖 x(k-1)，我们用另一种等效的增广方式：
    % x_aug(k) = [x(k); u(k-1)]
    
    A_aug = [A_discrete, B_discrete;
             zeros(inputDim, stateDim), eye(inputDim)];
    B_aug = [B_discrete;
             eye(inputDim)];
    C_aug_sys = [C_discrete, zeros(outputDim, inputDim)];
    
    augStateDim = stateDim + inputDim;
    
    % 当前的增广状态
    x_aug_k = [currentState; lastControl];
    
    % 3. 构建预测矩阵 F 和 Phi
    % 预测方程: X_aug_predict = F * x_aug(k) + Phi * Delta_U
    % 预测输出: Y_predict = C_aug_predict * X_aug_predict
    
    F = zeros(predictionHorizon * augStateDim, augStateDim);
    Phi = zeros(predictionHorizon * augStateDim, predictionHorizon * inputDim);
    
    tempA = eye(augStateDim);
    for i = 1:predictionHorizon
        tempA = tempA * A_aug;
        F((i-1)*augStateDim + 1 : i*augStateDim, :) = tempA;
        
        tempB = B_aug;
        for j = 1:i
            Phi((i-1)*augStateDim + 1 : i*augStateDim, (i-j)*inputDim + 1 : (i-j+1)*inputDim) = tempB;
            tempB = A_aug * tempB;
        end
    end
    
    % 构建扩展的C矩阵 (C_aug_predict)
    C_aug_predict = kron(eye(predictionHorizon), C_aug_sys);
    
    % 计算输出预测矩阵 F_y 和 Phi_y
    F_y = C_aug_predict * F;
    Phi_y = C_aug_predict * Phi;
    
    % 4. 构建预测域内的全局权重矩阵 Q_bar 和 R_bar
    Q_bar = kron(eye(predictionHorizon), stateWeight);
    R_bar = kron(eye(predictionHorizon), inputWeight);
    
    % 5. 目标函数转换与二次规划(QP)求解准备
    % 代价函数: J = (Y - R_ref)^T * Q_bar * (Y - R_ref) + Delta_U^T * R_bar * Delta_U
    
    % 扩展参考输出矩阵
    % 确保 reference 维度与 outputDim 匹配
    if length(reference) > outputDim
        reference = reference(1:outputDim);
    end
    R_ref = repmat(reference, predictionHorizon, 1);
    
    % 计算Hesse矩阵 H 和一次项系数向量 f
    H = Phi_y' * Q_bar * Phi_y + R_bar;
    % 强制 H 矩阵对称，消除因数值计算误差导致的非对称警告
    H = (H + H') / 2;
    f = Phi_y' * Q_bar * (F_y * x_aug_k - R_ref);
    
    % 6. 求解二次规划(QP)问题，得到最优控制增量序列 Delta_U
    H_qp = 2 * H;
    f_qp = 2 * f;
    
    coder.extrinsic('quadprog');
    
    Delta_U_opt = zeros(predictionHorizon * inputDim, 1);
    x0 = zeros(predictionHorizon * inputDim, 1);
    
    % 在 Simulink 中使用 coder.extrinsic 调用 quadprog 时，
    % 需要通过赋值一个预定义的空矩阵（而不是直接使用 []）或者明确传入大小为 0 的矩阵
    % 以免 Coder 传递给 MATLAB 的参数变成不可识别的空类型
    A_ineq = zeros(0, predictionHorizon * inputDim);
    b_ineq = zeros(0, 1);
    A_eq = zeros(0, predictionHorizon * inputDim);
    b_eq = zeros(0, 1);
    
    % options = optimoptions('quadprog', 'Algorithm', 'active-set', 'Display', 'off');
    % 注意：optimoptions 在某些 Simulink coder.extrinsic 场景下也可能解析异常，
    % 更稳妥的做法是使用 optimset 或者在外部定义。这里我们改用 optimset。
    coder.extrinsic('optimset');
    options = optimset('Algorithm', 'active-set', 'Display', 'off');
    
    % 调用 quadprog 时使用明确预定义的空矩阵变量
    Delta_U_opt_tmp = zeros(predictionHorizon * inputDim, 1);
    [Delta_U_opt_tmp, ~] = quadprog(H_qp, f_qp, A_ineq, b_ineq, A_eq, b_eq, [], [], x0, options);
    
    if ~isempty(Delta_U_opt_tmp)
        Delta_U_opt = Delta_U_opt_tmp;
    else
        Delta_U_opt = -H \ f;
    end
    
    % 7. 提取当前时刻的控制增量 delta_u(k)
    delta_u_k = Delta_U_opt(1:inputDim);

end
