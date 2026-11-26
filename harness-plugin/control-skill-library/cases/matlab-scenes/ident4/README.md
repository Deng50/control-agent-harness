# 二阶对象采数与四参数灰箱辨识（55 版）

阻尼比 0.7、自然频率 3、增益 1、常量偏置 1。采集 [t,u,y]，用 y″=a y′+c y+b u+f₀ 做同初态仿真辨识。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink、Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('ident4')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [Linear_2nd_order_model4DataProduction.slx](Linear_2nd_order_model4DataProduction.slx) | 总输入及位置记录端 | 20 s，末端 位移 sid251 = 0.10971；实际输入 sid273 = 1 | [修正后](../../../runs/matlab-scenes/ident4/fixed/Linear_2nd_order_model4DataProduction.slx.json) · [原版](../../../runs/matlab-scenes/ident4/original/Linear_2nd_order_model4DataProduction.slx.json) |
| [Linear_2nd_order_model4Ident.slx](Linear_2nd_order_model4Ident.slx) | 回放输入与可调参数的两积分器对象 | 20 s，末端 位移 sid251 = 0.10971 | [修正后](../../../runs/matlab-scenes/ident4/fixed/Linear_2nd_order_model4Ident.slx.json) · [原版](../../../runs/matlab-scenes/ident4/original/Linear_2nd_order_model4Ident.slx.json) |
| [Linear_2nd_order_Sys_Model.m](Linear_2nd_order_Sys_Model.m) | 给定参数返回完整位移预测序列 | 输出 [2001, 1]，末值 0.1101795 | [修正后](../../../runs/matlab-scenes/ident4/fixed/Linear_2nd_order_Sys_Model.m.json) · [原版](../../../runs/matlab-scenes/ident4/original/Linear_2nd_order_Sys_Model.m.json) |
| [Linear_2nd_order_Sys_Model4Datacollection.m](Linear_2nd_order_Sys_Model4Datacollection.m) | 按工况采数并导出原生 MAT 和表格 | 时间 / 输入 / 位移，数组 [2001, 3] | [修正后](../../../runs/matlab-scenes/ident4/fixed/Linear_2nd_order_Sys_Model4Datacollection.m.json) · [原版](../../../runs/matlab-scenes/ident4/original/Linear_2nd_order_Sys_Model4Datacollection.m.json) |
| [Model_parameters_Ident.m](Model_parameters_Ident.m) | 按位移残差优化灰箱参数 | 同激励预测 RMSE 1.139795e-16 | [修正后](../../../runs/matlab-scenes/ident4/fixed/Model_parameters_Ident.m.json) · [原版](../../../runs/matlab-scenes/ident4/original/Model_parameters_Ident.m.json) |

## 修复和适用条件

MAT 与 XLSX 均取记录端真实时间。原仿真对象采用方波保持，辨识输入修成相同的零阶保持，末点保留最后的输入。单独运行辨识使用归档记录，避免采数程序覆盖原始数据。

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `55-系统辨识MPC-ADRC-PID/Linear_2nd_order_model4DataProduction.slx`
- `55-系统辨识MPC-ADRC-PID/Linear_2nd_order_model4Ident.slx`
- `55-系统辨识MPC-ADRC-PID/Linear_2nd_order_Sys_Model.m`
- `55-系统辨识MPC-ADRC-PID/Linear_2nd_order_Sys_Model4Datacollection.m`
- `55-系统辨识MPC-ADRC-PID/Model_parameters_Ident.m`

配套文件为`55-系统辨识MPC-ADRC-PID/Data4Ident.mat`、`55-系统辨识MPC-ADRC-PID/model_param.mat`。

旧快照 `Linear_2nd_order_Sys_Model4Datacollection.asv` 已执行。[实际记录](../../../runs/matlab-scenes/ident4/backup/Linear_2nd_order_Sys_Model4Datacollection.asv.json)

旧快照 `Model_parameters_Ident.asv` 历史回放未通过。[实际记录](../../../runs/matlab-scenes/ident4/backup/Model_parameters_Ident.asv.json)

## 可在 MATLAB 中继续使用的产物

[Data4Ident.xlsx](../../../runs/matlab-scenes/ident4/fixed/Data4Ident.xlsx) · [Linear_2nd_order_Sys_Model4Datacollection.mat](../../../runs/matlab-scenes/ident4/fixed/Linear_2nd_order_Sys_Model4Datacollection.mat) · [Model_parameters_Ident.mat](../../../runs/matlab-scenes/ident4/fixed/Model_parameters_Ident.mat)
