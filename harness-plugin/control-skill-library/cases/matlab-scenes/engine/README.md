# 发动机供油激励与转速辨识

阶梯改变供油量（mg/循环），从 960 rpm 启动，采样 0.01 s，共 60 s。先采供油与转速，再由仿真回调拟合摩擦、效率、惯量。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink、Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('engine')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [demo_ident.slx](demo_ident.slx) | 摩擦、效率和转动惯量参与的动态对象 | 60 s，末端 转速 sid181 = 643.37 | [修正后](../../../runs/matlab-scenes/engine/fixed/demo_ident.slx.json) · [原版](../../../runs/matlab-scenes/engine/original/demo_ident.slx.json) |
| [LS_demo_ident.m](LS_demo_ident.m) | 预测转速的优化器回调 | 输出 [6001, 1]，末值 643.3701 | [修正后](../../../runs/matlab-scenes/engine/fixed/LS_demo_ident.m.json) · [原版](../../../runs/matlab-scenes/engine/original/LS_demo_ident.m.json) |
| [P_ident.m](P_ident.m) | 供油采数和转速参数优化入口 | 同激励预测 RMSE 0.5143464 | [修正后](../../../runs/matlab-scenes/engine/fixed/P_ident.m.json) · [原版](../../../runs/matlab-scenes/engine/original/P_ident.m.json) |
| [plant_model_with_dither_input.slx](plant_model_with_dither_input.slx) | 阶梯供油以及转速采数 | 60 s，末端 转速 sid39 = 639.35 | [修正后](../../../runs/matlab-scenes/engine/fixed/plant_model_with_dither_input.slx.json) · [原版](../../../runs/matlab-scenes/engine/original/plant_model_with_dither_input.slx.json) |

## 修复和适用条件

显式接收 SimulationOutput，为每次调用传入参数和真正施加的喷油序列。data_ident.mat 第一列是静态转速，不能当作从零开始的秒数。用速度拟合 RMSE 判断重构，参数耦合仍需独立激励辨别。本例验证的是转速辨识。

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `07 - Dyamic_model - engine speed model/demo_ident.slx`
- `07 - Dyamic_model - engine speed model/LS_demo_ident.m`
- `07 - Dyamic_model - engine speed model/P_ident.m`
- `07 - Dyamic_model - engine speed model/plant_model_with_dither_input.slx`

配套文件为`07 - Dyamic_model - engine speed model/data_ident.mat`。

旧快照 `demo_ident.slx.r2018b` 历史回放未通过。[实际记录](../../../runs/matlab-scenes/engine/backup/demo_ident.slx.r2018b.json)

## 可在 MATLAB 中继续使用的产物

[P_ident.mat](../../../runs/matlab-scenes/engine/fixed/P_ident.mat)
