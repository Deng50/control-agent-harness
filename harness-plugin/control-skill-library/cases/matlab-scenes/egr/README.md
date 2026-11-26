# 高压 EGR 阀质量流量

按排气绝压、进气绝压、排气绝对温度、阀开度四列拟合六参数有效面积与质量流量。第五列为流量 kg/s。

## 如何运行

从库根执行。依赖为 MATLAB、Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('egr')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [HpEGR_MassFlow.m](HpEGR_MassFlow.m) | 压力、温度、阀位到质量流量 | 输出 [195, 1]，末值 0.01920632 | [修正后](../../../runs/matlab-scenes/egr/fixed/HpEGR_MassFlow.m.json) · [原版](../../../runs/matlab-scenes/egr/original/HpEGR_MassFlow.m.json) |
| [NLS.m](NLS.m) | 读取工作表，拟合流量并换行验证 | 39/195 行留出，流量 RMSE 0.001553 kg/s | [修正后](../../../runs/matlab-scenes/egr/fixed/NLS.m.json) · [原版](../../../runs/matlab-scenes/egr/original/NLS.m.json) |
| [Understand_Newtons_Law.m](Understand_Newtons_Law.m) | 代价、梯度、曲率三幅教学图 | 3 个绘图坐标区 | [修正后](../../../runs/matlab-scenes/egr/fixed/Understand_Newtons_Law.m.json) · [原版](../../../runs/matlab-scenes/egr/original/Understand_Newtons_Law.m.json) |

## 修复和适用条件

原方程可能在反向压差、负绝对温度或非正对数自变量下产生复数。修正版检查物理定义域，闭阀处把有效面积下限设为零，使用有界最小二乘，另留每五行中一行检查插值。参数间有尺度耦合，流量拟合好不能单独证明面积和气体常数都辨识正确。

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `06 - Nonlinear LS_EGR mass flow rate model/HpEGR_MassFlow.m`
- `06 - Nonlinear LS_EGR mass flow rate model/NLS.m`
- `06 - Nonlinear LS_EGR mass flow rate model/Understand_Newtons_Law.m`

配套文件为`06 - Nonlinear LS_EGR mass flow rate model/HP_EGR valve.xls`。

## 可在 MATLAB 中继续使用的产物

[NLS.mat](../../../runs/matlab-scenes/egr/fixed/NLS.mat)
