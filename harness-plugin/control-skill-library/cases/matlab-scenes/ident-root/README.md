# 二阶对象采数与同名接口修复（根目录版）

对象、采数和辨识构成一个场景。原回调读六个元素，外层只传四个；模型实际连入四个标量 a、c、b、f₀。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink、Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('ident-root')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [Linear_2nd_order_model4DataProduction.slx](Linear_2nd_order_model4DataProduction.slx) | 总输入及位置记录端 | 20 s，末端 位移 sid251 = 0.10971；实际输入 sid273 = 1 | [修正后](../../../runs/matlab-scenes/ident-root/fixed/Linear_2nd_order_model4DataProduction.slx.json) · [原版](../../../runs/matlab-scenes/ident-root/original/Linear_2nd_order_model4DataProduction.slx.json) |
| [Linear_2nd_order_model4Ident.slx](Linear_2nd_order_model4Ident.slx) | 回放输入与可调参数的两积分器对象 | 20 s，末端 位移 sid251 = 0.10971 | [修正后](../../../runs/matlab-scenes/ident-root/fixed/Linear_2nd_order_model4Ident.slx.json) · [原版](../../../runs/matlab-scenes/ident-root/original/Linear_2nd_order_model4Ident.slx.json) |
| [Linear_2nd_order_Sys_Model.m](Linear_2nd_order_Sys_Model.m) | 给定参数返回完整位移预测序列 | 输出 [2001, 1]，末值 0.1101795 | [修正后](../../../runs/matlab-scenes/ident-root/fixed/Linear_2nd_order_Sys_Model.m.json) · [原版](../../../runs/matlab-scenes/ident-root/original/Linear_2nd_order_Sys_Model.m.json) |
| [Linear_2nd_order_Sys_Model4Datacollection.m](Linear_2nd_order_Sys_Model4Datacollection.m) | 按工况采数并导出原生 MAT 和表格 | 时间 / 输入 / 位移，数组 [2001, 3] | [修正后](../../../runs/matlab-scenes/ident-root/fixed/Linear_2nd_order_Sys_Model4Datacollection.m.json) · [原版](../../../runs/matlab-scenes/ident-root/original/Linear_2nd_order_Sys_Model4Datacollection.m.json) |
| [Model_parameters_Ident.m](Model_parameters_Ident.m) | 按位移残差优化灰箱参数 | 同激励预测 RMSE 1.139795e-16 | [修正后](../../../runs/matlab-scenes/ident-root/fixed/Model_parameters_Ident.m.json) · [原版](../../../runs/matlab-scenes/ident-root/original/Model_parameters_Ident.m.json) |

## 修复和适用条件

修复回调的参数形状与信号时标，显式传入工作区变量，令输入保持与采数端一致。六系数一般状态空间对象另在 sfunction 案例验证。根目录和 55 目录的同名程序分别实跑。

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `Linear_2nd_order_model4DataProduction.slx`
- `Linear_2nd_order_model4Ident.slx`
- `Linear_2nd_order_Sys_Model.m`
- `Linear_2nd_order_Sys_Model4Datacollection.m`
- `Model_parameters_Ident.m`

配套文件为`55-系统辨识MPC-ADRC-PID/Data4Ident.mat`、`55-系统辨识MPC-ADRC-PID/model_param.mat`。

## 可在 MATLAB 中继续使用的产物

[Data4Ident.xlsx](../../../runs/matlab-scenes/ident-root/fixed/Data4Ident.xlsx) · [Linear_2nd_order_Sys_Model4Datacollection.mat](../../../runs/matlab-scenes/ident-root/fixed/Linear_2nd_order_Sys_Model4Datacollection.mat) · [Model_parameters_Ident.mat](../../../runs/matlab-scenes/ident-root/fixed/Model_parameters_Ident.mat)
