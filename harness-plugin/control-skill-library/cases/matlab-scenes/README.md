# MATLAB / Simulink 控制案例目录

> 此处列出源库历史执行成绩。热参数辨识案例的三个本地研究文件列在快照清单的 local_study_only 中。复跑该场景或完整 verify_scenes 时，从获准使用的源库补齐工作副本中的同名文件，并按清单核对 SHA-256。


从 素材目录（外部 `素材`，[来源清单](../../sources.json)） 的源文件按物理对象及执行入口整理。**15 个场景，30 份 MATLAB 源文件，17 个 Simulink 模型**。下表每个目录都有中文 README、带注释的程序、原件位置和本次实测结果。库内的控制与验收入口使用 MATLAB。正式文件逐个运行，额外将 4 个自动保存脚本和 3 个旧模型按快照回放。[正式文件与源摘要](inventory.json) · [数据与标定版本](inputs.json) · [旧版本清单](backups.json) · [机器验收](../../runs/matlab-scenes/acceptance.json)

| 场景 | 可直接处理的任务 | 文件数 |
|---|---|---:|
| [一阶对象 MPC](mpc1/README.md) | 一阶传递对象阶跃跟踪，比较增广 QP 函数与图内控制器 | 2 |
| [二阶 MPC 和闭环整定](mpc2/README.md) | 对接根目录版 R 端口，联合仿真搜索 Q2、R，对照 ADRC | 5 |
| [二阶 MPC 与 ADRC（55 版）](mpc55/README.md) | 核对 R1 端口，对同一对象分别运行原生 MPC 与自抗扰闭环 | 3 |
| [二阶 PI 整定与多回路比较](pi/README.md) | 识别启用的比例和积分支路，按跟踪和用量统一计分 | 3 |
| [在线辨识与 ADRC](online-adrc/README.md) | 从脉冲、RLS 估计到质量判据和学习参数切换 | 1 |
| [发动机转速辨识](engine/README.md) | 供油激励采数，以转速残差估计参数 | 4 |
| [电增压转速、转矩与压力](eturbo/README.md) | 配齐三份标定，执行 1 ms 控制和压力动态仿真 | 2 |
| [高压 EGR 阀质量流量](egr/README.md) | 有界非线性最小二乘、定义域检查和留出流量评价 | 3 |
| [二阶采数与四参数辨识](ident4/README.md) | 生成 MAT、Excel，重放源工况并拟合带偏置方程 | 5 |
| [根目录二阶对象接口修复](ident-root/README.md) | 解决四参数优化入口与六参数同名回调冲突 | 5 |
| [状态空间 S-function](sfunction/README.md) | 接入 Level-2 MATLAB S-function，按独立欧拉解验收 | 1 |
| [电池二阶 RC 在线辨识](pulse/README.md) | 四种 RLS 变体、可实现性筛选和端电压复算 | 6 |
| [电池 SOC、二阶 RC 与松弛标定](soc/README.md) | 运行电路图，校正三状态 EKF 的观测方程和雅可比 | 3 |
| [电池热参数遗传辨识与 Simscape](thermal/README.md) | 工况一拟合四个热参数，工况二验证核心和表面温度 | 3 |
| [电池温度与热阻双 EKF](dekf/README.md) | 三个热阻初值下分别复位，比较温度和参数轨迹 | 1 |

## 从库根复跑

```powershell
# 先切换到已复制的案例工作目录
matlab -batch "addpath('cases/matlab-scenes'); run_scene('mpc2')"
matlab -batch "addpath('cases/matlab-scenes'); run_scene('soc','','fixed','SOCekf.m')"
matlab -batch "addpath('cases/matlab-scenes'); run_scene('all'); check_mpc; verify_scenes;"
```

`run_scene` 默认跑对应目录全部修正版。第二参数可指定其他素材根目录，第三参数用 `original` 回放未改的原件，或用 `backup` 回放历史快照。单个模型会编译、仿真至指定终点，并从实际端口写出轨迹。函数由案例适配器提供输入，S-function 通过 Simulink 宿主执行。各程序在临时目录工作，可能被前一脚本改写的标定会在下一项前恢复。对照成绩中的原版为配齐源数据和初始化参数后的执行结果。[运行入口](run_scene.m) · [数值复核](verify_scenes.m)

交付证据位于 `runs/matlab-scenes/<场景>/fixed`。逐文件 JSON 记录版本、摘要、运行耗时、输入输出读数或错误。模型的 CSV 首列是仿真秒数，随后各列按所标端口通道顺序排列。辨识、滤波和调参另有 MAT，内含参数或状态轨迹。MATLAB 中用 `whos('-file',文件名)` 检查变量。只核已有结果时运行 `matlab -batch "addpath('cases/matlab-scenes'); verify_scenes;"`。如需用原件重新生成已注释和修复的 SLX，可运行 `prepare_models`，随后重跑受影响场景和数值验收。[模型修复程序](prepare_models.m)

## MPC 如何验收

`check_mpc` 从两份独立 .m 程序及六个 MATLAB Function 图表提取实际生成的矩阵。给未来五拍控制增量逐列施加单位脉冲，在保持输入的条件下独立递推状态，对比每个预测步的每个通道。原版 8 项中的 7 项在第 3 步第 1 通道首次出现差异。修正后的 8 项最大误差均为零。这里核验的是增量传播；闭环跟踪、执行器约束与外层权重搜索各有独立证据。[矩阵验收代码](check_mpc.m) · [逐项数值](../../runs/matlab-scenes/mpc-matrix.json) · [有幅值与速率硬约束的 MPC 案例](../native-mpc/README.md)

## 历史快照与使用范围

7 个旧快照中 2 个完成历史回放。另 5 个分别遇到旧发动机摩擦标定 K3 等未提供、旧版二阶/电增压辨识表缺失、两个电池旧模型引用机器绝对路径下的 udds.mat 或 SOC80.mat，以及一份未写完的松弛计算草稿。正式场景提供可运行对应程序。旧轨迹所需原件缺失时，保留实际缺项。SLXC、slprj 及图像不作为另一个可编辑控制程序计数。热模型目录附有限制原样转售的作者声明；此处是本机研究和验证用的整理副本。[历史版本清单](backups.json) · 版权说明（外部 `素材/遗传算法热模型参数辨识/版权声明.txt`，[来源清单](../../sources.json)）

所有指标来自 MATLAB/Simulink R2024b 软件仿真或素材数据计算。要换现场对象，需重新确认输入单位、信号是否可测、实际采样与允许控制幅值，尤其是电增压转速的物理基准，以及电池示例所用的容量和电流约定。
