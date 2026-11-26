---
name: verify-mpc-prediction
description: 核验离散 MPC 的增量预测矩阵，定位未来输出与逐步递推首次不一致的步和通道。用于第二个或后续控制增量的索引、时序和直接通道检查。
---

从项目根目录运行 `python -m controlkit.prediction cases/prediction/input.json --out runs/prediction/pass.json`。验证用户矩阵时附加 `--matrix candidate.json`，其形状为 (N·ny,N·nu)，按时间优先排列。JSON 示例定义 A/B/C/D/E、x0、上一动作、N 行控制增量和同长扰动。约定 y[k+1]=C x[k+1]+D u[k]，传入离散矩阵，核对提供方是否使用相同采样时刻。保持原坐标；使用本次控制前的输出时先转换时序。

输出包括逐步预测、递推输出、最大误差和从 1 起算的首次失败步与通道。至少使第二步增量非零，并分别激励每个输入；一条脉冲通过只能覆盖被激励的列。失败时先查增量累加下三角块和 D 的时间对齐，再检查离散化。返回码 1 表示验证发现差异。[案例](../../../cases/prediction/README.md) · [预测与约束](../../../knowledge/mpc.md)

MATLAB 入口为 [verify_increment_matrix.m](scripts/verify_increment_matrix.m)。传入结构体含 A、B、C、D、E，候选矩阵 P，初态、上一实际动作、N×nu 的未来控制增量和 N×nd 的扰动。约定每拍末端输出为 C*x(k+1)+D*u(k)。返回逐点矩阵与递推值以及从 1 起算的首个错误步、通道和有符号误差。只改最后一拍的某一通道也能复核报错位置。

```matlab
addpath('.agents/skills/verify-mpc-prediction/scripts')
s=struct('A',0,'B',1,'C',1,'D',0,'E',zeros(1,0));
P=tril(ones(3)); P(3,2)=0;
r=verify_increment_matrix(s,P,0,0,[0;1;0],zeros(3,0));
assert(~r.passed && r.first_step==3 && r.first_channel==1 && r.signed_error==-1)
```

要复查素材中现成的控制器，在库根执行 `matlab -batch "addpath('cases/matlab-scenes'); check_mpc;"`。该命令读取实际嵌入图表并逐列比较，再调用同一 Skill 入口验证混合控制增量。[真实控制器的逐项数值](../../../runs/matlab-scenes/mpc-matrix.json)
