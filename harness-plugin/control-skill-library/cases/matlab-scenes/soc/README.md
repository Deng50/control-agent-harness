# 电池二阶 RC、SOC EKF 与松弛标定

Simulink 图回放 RC 电压、SOC；独立脚本按 38 Ah 和 DST 电流生成带噪端电压并估计 [SOC,U1,U2]。另有松弛曲线换算支路 R/C 的计算笔记。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink；原 SOC 脚本另需 Symbolic Math Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('soc')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [SOCekf.m](SOCekf.m) | 带噪端电压更新 SOC 及两支路电压 | SOC RMSE 0.4123 个百分点 | [修正后](../../../runs/matlab-scenes/soc/fixed/SOCekf.m.json) · [原版](../../../runs/matlab-scenes/soc/original/SOCekf.m.json) |
| [two_RC_model.slx](two_RC_model.slx) | 二阶 RC 对象记录电压与 SOC | 5000 s，末端 端电压 sid143 = 3.3094；子系统输出 sid157 = 0.81263 | [修正后](../../../runs/matlab-scenes/soc/fixed/two_RC_model.slx.json) · [原版](../../../runs/matlab-scenes/soc/original/two_RC_model.slx.json) |
| [Untitled.m](Untitled.m) | 由松弛系数恢复支路 R/C | R1、R2 为 0.002506、0.004006667 Ω | [修正后](../../../runs/matlab-scenes/soc/fixed/Untitled.m.json) · [原版](../../../runs/matlab-scenes/soc/original/Untitled.m.json) |

## 修复和适用条件

移除 EKF 更新对模拟真值的访问。端电压观测对两极化电压的导数均为 −1；噪声按标准差注入，使用 Joseph 形式更新协方差。松弛脚本显式装入保存的数据，避免错误示例覆盖正确结果。SOC 指标为仿真真值对照。

## 原件和配套输入

原图保留若干未接线的 Signal Builder、查表和备选支路，Simulink 会给出端口告警。本例核验已连接的端电压与 SOC 输出。要采用备选工况，需先接入相应信号并再跑验证。

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `最小二乘法进行电池等效电路参数辨识/多工况要求/SOCekf.m`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/two_RC_model.slx`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/Untitled.m`

配套文件为`最小二乘法进行电池等效电路参数辨识/多工况要求/DSTI.mat`、`最小二乘法进行电池等效电路参数辨识/多工况要求/HPPC.mat`、`最小二乘法进行电池等效电路参数辨识/多工况要求/matlab.mat`、`最小二乘法进行电池等效电路参数辨识/多工况要求/R0R1R2C1C2.mat`、`最小二乘法进行电池等效电路参数辨识/多工况要求/SOC90.mat`、`最小二乘法进行电池等效电路参数辨识/多工况要求/UDDSI.mat`、`最小二乘法进行电池等效电路参数辨识/多工况要求/VandI.mat`。

旧快照 `SOCekf.asv` 以 HPPC.mat 的 44,724 点同名电流执行。[实际记录](../../../runs/matlab-scenes/soc/backup/SOCekf.asv.json)

旧快照 `two_RC_model.slx.original` 历史回放未通过。[实际记录](../../../runs/matlab-scenes/soc/backup/two_RC_model.slx.original.json)

旧快照 `two_RC_model.slx.r2021b` 历史回放未通过。[实际记录](../../../runs/matlab-scenes/soc/backup/two_RC_model.slx.r2021b.json)

旧快照 `Untitled.asv` 历史回放未通过。[实际记录](../../../runs/matlab-scenes/soc/backup/Untitled.asv.json)

## 可在 MATLAB 中继续使用的产物

[SOCekf.mat](../../../runs/matlab-scenes/soc/fixed/SOCekf.mat) · [Untitled.mat](../../../runs/matlab-scenes/soc/fixed/Untitled.mat)
