# 二状态 Level-2 MATLAB S-function 对象

六个参数为 A11、A12、A21、A22、B1、B2。标量控制输入驱动两状态，在 0.01 s 欧拉步长下更新；输出第一状态。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('sfunction')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [Plant_Model.m](Plant_Model.m) | 注册并执行 DWork 状态回调 | 输出 [1, 1]，末值 0 | [修正后](../../../runs/matlab-scenes/sfunction/fixed/Plant_Model.m.json) · [原版](../../../runs/matlab-scenes/sfunction/original/Plant_Model.m.json) |

## 修复和适用条件

Outputs 只读 DWork，符合非直接馈通端口声明；实际输入在 Update 读取。新增 Simulink 测试宿主，以给定六参数作阶跃并与独立递推逐点核对。model_param6.mat 中实际是两个电增压系数。

[S-function 仿真宿主及欧拉参考解](case_sfunction.m)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `Plant_Model.m`

采集时一并核对过的电增压标定为`55-系统辨识MPC-ADRC-PID/model_param2.mat`、`model_param6.mat`、`55-系统辨识MPC-ADRC-PID/model_param7.mat`。

宿主自带六个测试系数，本例不把上述电增压标定作为六系数对象使用。
