# 电池热参数遗传辨识与 Simscape 热网络

> 此处列出源库历史执行成绩。热参数辨识案例的三个本地研究文件列在快照清单的 local_study_only 中。复跑该场景或完整 verify_scenes 时，从获准使用的源库补齐工作副本中的同名文件，并按清单核对 SHA-256。


用工况一拟合核心至表面的热阻、核心热容、表面至环境的热阻以及表面热容。工况二冻结参数，分别与实测表温、芯温比较。GA 保留种群 50、最多 200 代。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink、Simscape、Control System Toolbox、Global Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('thermal')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| estimate_function.m（本地研究原件，[程序清单](../inventory.json)） | 给定 R/C 的两温度联合误差 | 输出 [1, 1]，末值 0.3762503 | [修正后](../../../runs/matlab-scenes/thermal/fixed/estimate_function.m.json) · [原版](../../../runs/matlab-scenes/thermal/original/estimate_function.m.json) |
| Identification_ThermalModel.m（本地研究原件，[程序清单](../inventory.json)） | GA 训练与跨工况复验 | 工况二表面 / 核心 RMSE 0.1201 / 0.2964 K | [修正后](../../../runs/matlab-scenes/thermal/fixed/Identification_ThermalModel.m.json) · [原版](../../../runs/matlab-scenes/thermal/original/Identification_ThermalModel.m.json) |
| thermalModel_test.slx（本地研究原件，[程序清单](../inventory.json)） | Simscape 热网络与对照信号 | 5972 s，末端 温度传感通道 sid309 = 0.1373；温度传感通道 sid279 = 0.20727 | [修正后](../../../runs/matlab-scenes/thermal/fixed/thermalModel_test.slx.json) · [原版](../../../runs/matlab-scenes/thermal/original/thermalModel_test.slx.json) |

## 修复和适用条件

在电压、电流与温度共同时间段插值，修复末点补零和温度缺失掩码错误。熵热采用绝对温度；GA 用与原双节点欧拉递推等价的滤波计算加速，两个工况均运行 Simscape 图。离线产热用到实测芯温。源目录有学习示例的版权声明，对外分发原件前需核实授权。

[数据同步与产热](../shared/case_thermal_input.m) · [双节点热传递](case_temperatures.m)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `遗传算法热模型参数辨识/estimate_function.m`
- `遗传算法热模型参数辨识/Identification_ThermalModel.m`
- `遗传算法热模型参数辨识/thermalModel_test.slx`

配套文件为`遗传算法热模型参数辨识/tempParameters.mat`、`遗传算法热模型参数辨识/版权声明.txt`、`遗传算法热模型参数辨识/说明.txt`、`遗传算法热模型参数辨识/数据/dUdT.mat`、`遗传算法热模型参数辨识/数据/OCV_SOC.mat`、`遗传算法热模型参数辨识/数据/Temperature_data.mat`、`遗传算法热模型参数辨识/数据/Voltage_current_data.mat`。

## 可在 MATLAB 中继续使用的产物

[Identification_ThermalModel.mat](../../../runs/matlab-scenes/thermal/fixed/Identification_ThermalModel.mat)
