# 在线 RLS、参数切换与二阶 ADRC

运行 250 s 的脉冲工况。遗忘因子 RLS 根据观测更新四参数，学习使能和预测误差决定何时切换控制器采用的参数。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('online-adrc')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [RLS_ADRC_2nd_order_Kang.slx](RLS_ADRC_2nd_order_Kang.slx) | 遗忘因子 RLS、质量判据、参数切换 | 250 s，末端 辨识激励 sid195 = 1；RLS 增益 sid196 = 0.00040134, 0.00046116, -0.00037759, -0.00069066；RLS 参数 theta_rls = -8.157, -15.621, 7.8015, 19.512；实际对象 sid639 = 9.4211e-07；切换参数 sid460 = -8.1549 | [修正后](../../../runs/matlab-scenes/online-adrc/fixed/RLS_ADRC_2nd_order_Kang.slx.json) · [原版](../../../runs/matlab-scenes/online-adrc/original/RLS_ADRC_2nd_order_Kang.slx.json) |

## 修复和适用条件

原控制逻辑保留，在 RLS 内嵌函数内注释协方差更新及切换关系。物理对象输出、RLS 增益、四通道估计和切换端分别记录；不能仅凭无发散判断学习参数已被控制器采纳。

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `RLS_ADRC_2nd_order_Kang.slx`

配套文件为`55-系统辨识MPC-ADRC-PID/Data4Ident.mat`、`55-系统辨识MPC-ADRC-PID/model_param.mat`。
