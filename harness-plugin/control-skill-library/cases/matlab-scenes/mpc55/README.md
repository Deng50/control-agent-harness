# 二阶 MPC 与 ADRC 阶跃对照（55 版）

以同一四参数二阶对象分别运行增量 MPC 和 ADRC。控制器的真实输入权重叫 R1。ADRC 的跟踪、扰动估计和补偿由原图内子系统实现。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('mpc55')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [ADRC_2nd_Order_Kang.slx](ADRC_2nd_Order_Kang.slx) | 扩张状态观测与扰动补偿闭环 | 20 s，末端 对象位置 sid174 = 1；阶跃参考 sid154 = 1 | [修正后](../../../runs/matlab-scenes/mpc55/fixed/ADRC_2nd_Order_Kang.slx.json) · [原版](../../../runs/matlab-scenes/mpc55/original/ADRC_2nd_Order_Kang.slx.json) |
| [LMPC_2nd_order_Kang_2.slx](LMPC_2nd_order_Kang_2.slx) | 原生图内增量预测与闭环 | 10 s，末端 参考 sid156 = 1, 0；子系统输出 sid157 = 0.88842；对象状态 sid189 = 0.99998 | [修正后](../../../runs/matlab-scenes/mpc55/fixed/LMPC_2nd_order_Kang_2.slx.json) · [原版](../../../runs/matlab-scenes/mpc55/original/LMPC_2nd_order_Kang_2.slx.json) |
| [Run_MPC_For_2nd_order_Plant.m](Run_MPC_For_2nd_order_Plant.m) | 装入标定后分别启动 MPC 与 ADRC | 终止时刻 [10, 20] s | [修正后](../../../runs/matlab-scenes/mpc55/fixed/Run_MPC_For_2nd_order_Plant.m.json) · [原版](../../../runs/matlab-scenes/mpc55/original/Run_MPC_For_2nd_order_Plant.m.json) |

## 修复和适用条件

独立加载标定，去除桌面打开模型的依赖，修复 MPC 未来增量列；从图中端口取得参考与真实位移。ADRC 看整段阶跃过程和稳态，不能只看模型编译。

图内 chart 与独立函数的矩阵可以单独检查。[MATLAB 核验入口](../check_mpc.m) · [原版与修正后的逐图结果](../../../runs/matlab-scenes/mpc-matrix.json) · [原生硬约束 MPC](../../native-mpc/README.md)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `55-系统辨识MPC-ADRC-PID/ADRC_2nd_Order_Kang.slx`
- `55-系统辨识MPC-ADRC-PID/LMPC_2nd_order_Kang_2.slx`
- `55-系统辨识MPC-ADRC-PID/Run_MPC_For_2nd_order_Plant.m`

配套文件为`55-系统辨识MPC-ADRC-PID/Data4Ident.mat`、`55-系统辨识MPC-ADRC-PID/model_param.mat`。
