# 二阶 MPC、ADRC 对照与闭环整定（根目录版）

装入四参数模型，运行原生增量 MPC 或 ADRC。对 MPC 使用 fmincon 搜索 Q2 与实际连接的 R。外层评价由时间加权跟踪误差和实际动作能量组成。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink、Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('mpc2')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [ADRC_2nd_Order_Kang.slx](ADRC_2nd_Order_Kang.slx) | 扩张状态观测与扰动补偿闭环 | 20 s，末端 对象位置 sid174 = 1；阶跃参考 sid154 = 1 | [修正后](../../../runs/matlab-scenes/mpc2/fixed/ADRC_2nd_Order_Kang.slx.json) · [原版](../../../runs/matlab-scenes/mpc2/original/ADRC_2nd_Order_Kang.slx.json) |
| [LMPC_2nd_order_Kang_2.slx](LMPC_2nd_order_Kang_2.slx) | 原生图内增量预测与闭环 | 10 s，末端 参考 sid156 = 1, 0；子系统输出 sid157 = 0.88842；对象状态 sid189 = 0.99998 | [修正后](../../../runs/matlab-scenes/mpc2/fixed/LMPC_2nd_order_Kang_2.slx.json) · [原版](../../../runs/matlab-scenes/mpc2/original/LMPC_2nd_order_Kang_2.slx.json) |
| [MPC_QP.m](MPC_QP.m) | 独立二阶 MPC 矩阵和首拍增量 | 输出 [1, 1]，末值 0.6477397 | [修正后](../../../runs/matlab-scenes/mpc2/fixed/MPC_QP.m.json) · [原版](../../../runs/matlab-scenes/mpc2/original/MPC_QP.m.json) |
| [Run_MPC_For_2nd_order_Plant.m](Run_MPC_For_2nd_order_Plant.m) | 装入标定后分别启动 MPC 与 ADRC | 终止时刻 [10, 20] s | [修正后](../../../runs/matlab-scenes/mpc2/fixed/Run_MPC_For_2nd_order_Plant.m.json) · [原版](../../../runs/matlab-scenes/mpc2/original/Run_MPC_For_2nd_order_Plant.m.json) |
| [Run_MPC_For_2nd_order_Plant_auto_tuning.m](Run_MPC_For_2nd_order_Plant_auto_tuning.m) | 外层 fmincon 搜索权重 | 同口径外层目标 29.388 → 5.9809 | [修正后](../../../runs/matlab-scenes/mpc2/fixed/Run_MPC_For_2nd_order_Plant_auto_tuning.m.json) · [原版](../../../runs/matlab-scenes/mpc2/original/Run_MPC_For_2nd_order_Plant_auto_tuning.m.json) |

## 修复和适用条件

原运行脚本只声明 R1，图中读取 R；修正为 R。实际内嵌和独立 .m 中均补齐后续增量对剩余预测步的影响。外层代价改为沿记录时间积分并登记搜索前后的同口径目标。原控制器未施加硬约束。

图内 chart 与独立函数的矩阵可以单独检查。[MATLAB 核验入口](../check_mpc.m) · [原版与修正后的逐图结果](../../../runs/matlab-scenes/mpc-matrix.json) · [原生硬约束 MPC](../../native-mpc/README.md)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `ADRC_2nd_Order_Kang.slx`
- `LMPC_2nd_order_Kang_2.slx`
- `MPC_QP.m`
- `Run_MPC_For_2nd_order_Plant.m`
- `Run_MPC_For_2nd_order_Plant_auto_tuning.m`

配套文件为`55-系统辨识MPC-ADRC-PID/Data4Ident.mat`、`55-系统辨识MPC-ADRC-PID/model_param.mat`。

## 可在 MATLAB 中继续使用的产物

[Run_MPC_For_2nd_order_Plant_auto_tuning.mat](../../../runs/matlab-scenes/mpc2/fixed/Run_MPC_For_2nd_order_Plant_auto_tuning.mat)
