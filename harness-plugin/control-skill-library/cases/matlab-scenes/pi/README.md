# 二阶对象 PI 整定与多回路比较

PID2 模型启用的控制器是 PI。Q2 连接比例支路，R 连接积分支路。另一 SLX 另含 MPC 和 PI 的比较回路。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink、Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('pi')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [PID2_2nd_order_Kang_2.slx](PID2_2nd_order_Kang_2.slx) | 外层整定使用的 PI 支路 | 10 s，末端 参考 sid365 = 1；对象输出 sid370 = 0.42638 | [修正后](../../../runs/matlab-scenes/pi/fixed/PID2_2nd_order_Kang_2.slx.json) · [原版](../../../runs/matlab-scenes/pi/original/PID2_2nd_order_Kang_2.slx.json) |
| [PID_for_2nd_order_sysm.slx](PID_for_2nd_order_sysm.slx) | 含 MPC 和 PI 的比较模型 | 10 s，末端 参考 sid375 = 1, 0；对象输出 sid430 = 0.99998 | [修正后](../../../runs/matlab-scenes/pi/fixed/PID_for_2nd_order_sysm.slx.json) · [原版](../../../runs/matlab-scenes/pi/original/PID_for_2nd_order_sysm.slx.json) |
| [Run_PID_For_2nd_order_Plant_auto_tuning.m](Run_PID_For_2nd_order_Plant_auto_tuning.m) | 搜索 PI 比例和积分增益 | 同口径外层目标 2.6124 → 1.5726 | [修正后](../../../runs/matlab-scenes/pi/fixed/Run_PID_For_2nd_order_Plant_auto_tuning.m.json) · [原版](../../../runs/matlab-scenes/pi/original/Run_PID_For_2nd_order_Plant_auto_tuning.m.json) |

## 修复和适用条件

原误差项写死 0.1，动作项直接求和。修正为按两个记录信号的实际时间积分，输入能量的显式权重取 0.1。记录初始及整定目标。两图保留的 MPC 计算也接受预测矩阵检验。

图内 chart 与独立函数的矩阵可以单独检查。[MATLAB 核验入口](../check_mpc.m) · [原版与修正后的逐图结果](../../../runs/matlab-scenes/mpc-matrix.json) · [原生硬约束 MPC](../../native-mpc/README.md)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `PID2_2nd_order_Kang_2.slx`
- `PID_for_2nd_order_sysm.slx`
- `Run_PID_For_2nd_order_Plant_auto_tuning.m`

配套文件为`55-系统辨识MPC-ADRC-PID/Data4Ident.mat`、`55-系统辨识MPC-ADRC-PID/model_param.mat`。

## 可在 MATLAB 中继续使用的产物

[Run_PID_For_2nd_order_Plant_auto_tuning.mat](../../../runs/matlab-scenes/pi/fixed/Run_PID_For_2nd_order_Plant_auto_tuning.mat)
