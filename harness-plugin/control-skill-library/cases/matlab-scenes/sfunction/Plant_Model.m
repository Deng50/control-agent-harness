function Plant_Model(block)
% Level-2 S-function 用两个 DWork 状态以 0.01 s 欧拉步长更新，输出为第一状态。

setup(block);

end

%% ==================================================
%% 初始化
%% ==================================================
function setup(block)

block.NumDialogPrms  = 0;

block.NumInputPorts  = 1;
block.NumOutputPorts = 1;

block.SetPreCompInpPortInfoToDynamic;
block.SetPreCompOutPortInfoToDynamic;

% 输入 u
block.InputPort(1).Dimensions = 1;
block.InputPort(1).DirectFeedthrough = false;

% 输出 y
block.OutputPort(1).Dimensions = 1;

block.SampleTimes = [0.01 0];  % 仿真步长

block.SimStateCompliance = 'DefaultSimState';

block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
block.RegBlockMethod('InitializeConditions', @InitializeConditions);
block.RegBlockMethod('Outputs', @Outputs);
block.RegBlockMethod('Update', @Update);

end

%% ==================================================
%% 状态变量
%% ==================================================
function DoPostPropSetup(block)

block.NumDworks = 1;

block.Dwork(1).Name = 'x';
block.Dwork(1).Dimensions = 2;
block.Dwork(1).DatatypeID = 0;
block.Dwork(1).Complexity = 'Real';
block.Dwork(1).UsedAsDiscState = true;

end


%% ==================================================
%% 初始化状态
%% ==================================================
function InitializeConditions(block)

block.Dwork(1).Data = [0;0];

end


%% ==================================================
%% 输出
%% ==================================================
function Outputs(block)

% D=0，Outputs 不读取本拍输入，符合非直接馈通的端口声明。
x = block.Dwork(1).Data; block.OutputPort(1).Data=x(1);
end

function Update(block)

x = block.Dwork(1).Data;

u = block.InputPort(1).Data;

% a = evalin('base','a');
% b = evalin('base','b');
x_model = evalin('base', 'x_model');
a(1) = x_model(1);
a(2) = x_model(2);
a(3) = x_model(3);
a(4) = x_model(4);
b(1) = x_model(5);
b(2) = x_model(6);

A = [a(1) a(2);
     a(3) a(4)];

B = [b(1);
     b(2)];

Ts = block.SampleTimes(1);

% Euler离散化
x_next = x + Ts*(A*x + B*u);

block.Dwork(1).Data = x_next;

end
