function controlInput = MPC_QP(predictionHorizon, stateWeight, inputWeight, samplingTime, A_continuous, B_continuous, C_continuous, currentState, stateChange, referenceState, disturbance)
% 二阶对象使用实际连线的 R，采样周期 0.01 s；区分状态跟踪与执行器消耗。
% Model Predictive Control (MPC) based on Quadratic Programming (QP)
% 
% Inputs:
% - predictionHorizon: The prediction horizon for MPC.
% - stateWeight: Weight matrix for state deviation in the cost function.
% - inputWeight: Weight matrix for control input in the cost function.
% - samplingTime: Time step for discretization.
% - A_continuous, B_continuous, C_continuous: Continuous-time system matrices.
% - currentState: The current state of the system.
% - stateChange: The difference between the current state and the previous state.
% - referenceState: The desired reference state for the system.
% - disturbance: Estimated disturbance for the system.
%
% Outputs:
% - controlInput: The computed control input based on MPC.

%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the dimension of the state vector
stateDim = size(A_continuous, 1); 

% Discretize the continuous-time system matrices using Euler method
A_discrete = A_continuous .* samplingTime + eye(stateDim);
B_discrete = B_continuous .* samplingTime;
C_discrete = C_continuous .* samplingTime;

% Compute the dimension of the input vector
inputDim = size(B_discrete, 2);

% Construct the extended weight matrices for the entire prediction horizon using the Kronecker product
Q_bar = kron(eye(predictionHorizon), stateWeight);
R_bar = kron(eye(predictionHorizon), inputWeight);

% Pre-compute certain operations to avoid repeated calculations and enhance efficiency
Q_bar_transpose_times_Q_bar = Q_bar' * Q_bar;
R_bar_transpose_times_R_bar = R_bar' * R_bar;

% Use repmat to create a vector by repeating the reference state for the entire prediction horizon
referenceStateAugmented = repmat(referenceState, predictionHorizon, 1);

% Initialization of matrices that will represent the prediction model over the horizon
stateMatrix = [A_discrete; zeros((predictionHorizon-1)*stateDim, stateDim)];
identityMatrix = [eye(stateDim); zeros((predictionHorizon-1)*stateDim, stateDim)];
controlMatrix = [C_discrete; zeros((predictionHorizon-1)*stateDim, inputDim)];
inputMatrix = zeros(predictionHorizon*stateDim, predictionHorizon*inputDim);

% Temporary matrices to assist in constructing the prediction model
tmpMatrix1 = A_discrete;
tmpMatrix2 = C_discrete;

% Build the prediction model matrices over the horizon
for i = 1:predictionHorizon-1
    rows = i*stateDim + (1:stateDim);
    tmpMatrix1 = A_discrete + tmpMatrix1 * A_discrete;
    stateMatrix(rows, :) = tmpMatrix1;
    identityMatrix(rows, :) = eye(stateDim);
    tmpMatrix2 = C_discrete + A_discrete * tmpMatrix2;
    controlMatrix(rows, :) = tmpMatrix2;
end

% Populate the input matrix over the horizon
inputMatrix(1:stateDim, 1:inputDim) = B_discrete;
tmpMatrix3 = B_discrete;

for i = 1:predictionHorizon-1
    rows = i*stateDim + (1:stateDim);
    tmpMatrix3 = B_discrete + A_discrete * tmpMatrix3;
    inputMatrix(rows, 1:inputDim) = tmpMatrix3;
end

% Adjust the input matrix based on the prediction horizon
for j = 1:predictionHorizon-1
    columns = j*inputDim + (1:inputDim);
    start = j*stateDim + 1;
    % 第二个及以后动作也会影响余下全部步。
    stop = predictionHorizon*stateDim;
    inputMatrix(start:stop, columns) = inputMatrix(1:stop-start+1, 1:inputDim);
end

if size(controlMatrix, 2) ~= size(disturbance, 1)
    error('Dimensions of controlMatrix and disturbance do not match for multiplication');
end


% Compute the prediction error over the horizon
errorPrediction = referenceStateAugmented - stateMatrix * stateChange - identityMatrix * currentState - controlMatrix * disturbance;

% Compute the optimal control matrix using the pre-computed matrices
controlMatrixOptimal = inputMatrix' * Q_bar * inputMatrix + R_bar;

% Solve the quadratic program to determine the change in control input over the horizon
deltaControl = controlMatrixOptimal \ inputMatrix' * Q_bar * errorPrediction;

% Return the first control input from the delta control sequence
controlInput = deltaControl(1:inputDim);

end
