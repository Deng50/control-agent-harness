# 脉冲工况下的二阶 RC 在线 RLS

287 Ah，约定放电电流为正，使用脉冲记录前 80%。比较电压减 OCV、时变 OCV 修正、显式截距、当拍局部 OCV 四种回归。

## 如何运行

从库根执行。依赖为 MATLAB。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('pulse')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [model_copy.m](model_copy.m) | 回算函数的另一版本与兼容入口 | 输出 [1, 100]，末值 2.681 | [修正后](../../../runs/matlab-scenes/pulse/fixed/model_copy.m.json) · [原版](../../../runs/matlab-scenes/pulse/original/model_copy.m.json) |
| [ModelTest.m](ModelTest.m) | 按采样间隔递推端电压 | 输出 [1, 100]，末值 2.681 | [修正后](../../../runs/matlab-scenes/pulse/fixed/ModelTest.m.json) · [原版](../../../runs/matlab-scenes/pulse/original/ModelTest.m.json) |
| [RLS_U.m](RLS_U.m) | 对电压减 OCV 回归 | 首个物理解第 539 点，接受 15340/15878 次，电压 RMSE 0.00890 V | [修正后](../../../runs/matlab-scenes/pulse/fixed/RLS_U.m.json) · [原版](../../../runs/matlab-scenes/pulse/original/RLS_U.m.json) |
| [RLS_Ub.m](RLS_Ub.m) | 按自身时刻补偿 OCV 的五参数回归 | 首个物理解第 539 点，接受 15340/15878 次，电压 RMSE 0.00899 V | [修正后](../../../runs/matlab-scenes/pulse/fixed/RLS_Ub.m.json) · [原版](../../../runs/matlab-scenes/pulse/original/RLS_Ub.m.json) |
| [RLS_V1.m](RLS_V1.m) | 学习独立电压截距的六参数回归 | 首个物理解第 540 点，接受 15338/15878 次，电压 RMSE 0.01334 V | [修正后](../../../runs/matlab-scenes/pulse/fixed/RLS_V1.m.json) · [原版](../../../runs/matlab-scenes/pulse/original/RLS_V1.m.json) |
| [RLS_V2.m](RLS_V2.m) | 当拍 OCV 前馈的五参数回归 | 首个物理解第 539 点，接受 15340/15878 次，电压 RMSE 0.00943 V | [修正后](../../../runs/matlab-scenes/pulse/fixed/RLS_V2.m.json) · [原版](../../../runs/matlab-scenes/pulse/original/RLS_V2.m.json) |

## 修复和适用条件

原脚本可产生复时间常数或负阻容，部分版本甚至把总电阻放进时间常数排序。修正版从两个 (0,1) 实极点和部分分式恢复 R/C，未获得物理解前不计电压误差，后续无效更新保留上次有效估计。辨识与回算共用实测采样周期，报告有效起点及接受次数。

[恢复物理参数](case_rc_params.m) · [端电压递推](case_rc_voltage.m)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/model_copy.m`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/ModelTest.m`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/RLS_U.m`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/RLS_Ub.m`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/RLS_V1.m`
- `最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/RLS_V2.m`

配套文件为`最小二乘法进行电池等效电路参数辨识/多工况要求/zuix/rls/LiBatt_PulseDataEVE.mat`。

## 可在 MATLAB 中继续使用的产物

[RLS_U.mat](../../../runs/matlab-scenes/pulse/fixed/RLS_U.mat) · [RLS_Ub.mat](../../../runs/matlab-scenes/pulse/fixed/RLS_Ub.mat) · [RLS_V1.mat](../../../runs/matlab-scenes/pulse/fixed/RLS_V1.mat) · [RLS_V2.mat](../../../runs/matlab-scenes/pulse/fixed/RLS_V2.mat)
