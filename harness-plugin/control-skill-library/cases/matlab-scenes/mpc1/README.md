# 一阶传递环节与增量 MPC

原闭环跟踪从 105 到 120 的参考。独立 .m 以 [x,u上一拍] 增广并调用 quadprog；SLX 内有两份独立的控制函数。

## 如何运行

从库根执行。依赖为 MATLAB、Simulink；独立 QP 函数需要 Optimization Toolbox。

```powershell
matlab -batch "addpath('cases/matlab-scenes'); run_scene('mpc1')"
```

如需单独运行，给 run_scene 第四个参数传入文件名。调用脚本前装载配套数据，函数收到与接口匹配的实参，SLX 按记录端口输出 CSV。

## 程序与实跑结果

| 程序 | 用途 | 修正后实测 | 逐文件记录 |
|---|---|---|---|
| [LMPC_1st_order_Kang_2.slx](LMPC_1st_order_Kang_2.slx) | 一阶传递对象与图内增量控制器 | 20 s，末端 阶跃参考 sid1654 = 120；一阶对象输出 sid1737 = 120；控制器输出 sid1676 = 160 | [修正后](../../../runs/matlab-scenes/mpc1/fixed/LMPC_1st_order_Kang_2.slx.json) · [原版](../../../runs/matlab-scenes/mpc1/original/LMPC_1st_order_Kang_2.slx.json) |
| [MPC_QP_1st.m](MPC_QP_1st.m) | 独立 QP 求得当前最优增量 | 输出 [1, 1]，末值 2.045962 | [修正后](../../../runs/matlab-scenes/mpc1/fixed/MPC_QP_1st.m.json) · [原版](../../../runs/matlab-scenes/mpc1/original/MPC_QP_1st.m.json) |

## 修复和适用条件

修正两份图内代码对后续控制增量的传播。用五步时域的每一列单独施加单位增量，核对矩阵与状态递推。不同图表对权重是否平方的约定沿用原版；移植时分别核实。

图内 chart 与独立函数的矩阵可以单独检查。[MATLAB 核验入口](../check_mpc.m) · [原版与修正后的逐图结果](../../../runs/matlab-scenes/mpc-matrix.json) · [原生硬约束 MPC](../../native-mpc/README.md)

## 原件和配套输入

以下相对路径以 `D:\projects\素材` 为根。程序与数据分别保留原件摘要。[程序清单](../inventory.json) · [数据清单](../inputs.json)

- `LMPC_1st_order_Kang_2.slx`
- `MPC_QP_1st.m`
