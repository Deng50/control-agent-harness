# 电增压器转矩、转速与进气压力

原 VGT 脚本分别读取三套参数，设 1 ms 步长，运行 50 s；分别记录转速、压力动态和 RLS 输出。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('eturbo')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [Run_eTurbo_with_Torque_mode_2018bp.slx](Run_eTurbo_with_Torque_mode_2018bp.slx) | 电增压转速、压力及在线学习 | 50 s，末端 脉冲 sid1896 = 20；RLS 增益 sid1897 = 4.7719e-05, 1.533e-05, -4.6956e-08, -0.00078518；RLS 参数 theta_rls = -6.3069, -9.9011, 0.009886, 990.51；压力环节 sid2337 = 105；转速环节 sid2347 = 5004.1 | [修正后](../../../runs/matlab-scenes/eturbo/fixed/Run_eTurbo_with_Torque_mode_2018bp.slx.json) · [原版](../../../runs/matlab-scenes/eturbo/original/Run_eTurbo_with_Torque_mode_2018bp.slx.json) |
| [Run_VGT_Ctrl_using_Spd_and_Torque_Mode_scrpt.m](Run_VGT_Ctrl_using_Spd_and_Torque_Mode_scrpt.m) | 按档案版本给电增压动态赋初值 | 终止时刻 50 s | [修正后](../../../runs/matlab-scenes/eturbo/fixed/Run_VGT_Ctrl_using_Spd_and_Torque_Mode_scrpt.m.json) · [原版](../../../runs/matlab-scenes/eturbo/original/Run_VGT_Ctrl_using_Spd_and_Torque_Mode_scrpt.m.json) |

## 修复和适用条件

根目录缺两份同名标定，取 55 目录实存文件并核对摘要。原转速通道在部分工况取负值，使用其作为绝对机械转速前需确认信号基准、物理范围和限幅。

## 原件和配套输入

原图的 Scope2 试图同时显示估计值与四条参数“真值”。追踪连线发现 b0_real、a_real、f0_real、c_real 对应的 From 均无生产者，四条线路只通往 Scope2。副本移除这些无数据的比较通道，保留四路实际信号。仿真仍提示控制子系统的一个可选输出与一个 Goto 没有下游，这两项不参加本例数值验收。

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `Run_eTurbo_with_Torque_mode_2018bp.slx`
- `Run_VGT_Ctrl_using_Spd_and_Torque_Mode_scrpt.m`

配套文件为`55-系统辨识MPC-ADRC-PID/model_param2.mat`、`model_param6.mat`、`55-系统辨识MPC-ADRC-PID/model_param7.mat`。
