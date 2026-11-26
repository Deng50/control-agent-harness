# 电池热阻与温度双 EKF

两节点状态均相对环境温度。用表温创新量更新状态与表面对环境热阻 Rs（K/W）。从 20、10、1 K/W 分别初始化，各跑 3500 个点。

## 如何运行

从库根执行。依赖为 MATLAB。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('dekf')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [DEKF_Rs_1_10_20.m](DEKF_Rs_1_10_20.m) | 三组初始热阻下的双 EKF | 三组 Rs 末值 4.6380, 4.6380, 4.6380 K/W | [修正后](../../../runs/matlab-scenes/dekf/fixed/DEKF_Rs_1_10_20.m.json) · [原版](../../../runs/matlab-scenes/dekf/original/DEKF_Rs_1_10_20.m.json) |

## 修复和适用条件

每组独立复位状态、状态协方差及参数协方差，传播热阻灵敏度。产热用上拍估计芯温，实测芯温只在事后计分。热阻真实值没有独立测量，温度 RMSE 不能代替热阻真值验收。

[电热数据预处理](../shared/case_thermal_input.m)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `DEKF电池热状态参数在线估计/DEKF_Rs_1_10_20.m`

配套文件为`DEKF电池热状态参数在线估计/data/tempParameters.mat`、`DEKF电池热状态参数在线估计/data/dUdT.mat`、`DEKF电池热状态参数在线估计/data/OCV_SOC.mat`、`DEKF电池热状态参数在线估计/data/Temperature_data.mat`、`DEKF电池热状态参数在线估计/data/Voltage_current_data.mat`。

## 可在 MATLAB 中继续使用的产物

[DEKF_Rs_1_10_20.mat](../../../runs/matlab-scenes/dekf/fixed/DEKF_Rs_1_10_20.mat)
