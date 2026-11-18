# MPC 仅设计

`synthesize_mpc` 交付预测模型、控制代价、约束和调用程序，不要求先提供评价 Plant 或三组调参工况。旧 `design_controller` 保留为需要闭环搜索的独立操作。一个设计产物的 `verified` 表示包及本次检查通过；闭环结论仍须单独读取。

## 输入路线

- `inputs.model`：使用登记的可微模型。`config.prediction.method=direct` 保留非线性预测函数，直接执行 NMPC；不强制生成 A/B。
- `config.prediction.model_spec`：显式离散/连续方程；支持的连续仿射模型使用精确 ZOH，记录离散化方法和采样时间。
- `inputs.dataset`：`identify_affine` 只支持完整状态转移数据。提供输出矩阵 C 和预先确定的 `acceptance.atol/rtol`；仅训练分区拟合，在独立分区验证单步及整轨迹。仅有输入输出测量不等于状态已知。
- `linearize`：明确 `operating_point.x/u/d` 与 `validity_bounds.x/u/d`，保留偏置，检查中心和各轴端点。该局部检查不构成整个盒区间的误差证明。不能表达的输出偏置直接拒绝。
- `inputs.description`：弹簧振子可交付 `template=spring_damper`。实例化需要 mass、damping、stiffness、sample_time、state_bounds、input_bounds、output_bounds 七项；缺项或 null 保持参数化草稿。自然语言未给出的数值不会被自动补成真实物理参数。

`inputs.model/dataset/description` 互斥。无模型、无完整参数的草稿不要求数值确认；数值设计确认 signals，辨识另确认 training。确认使用本 Session 的原生命令及 `config.plan_refs`；请求 JSON 自行写 confirmed 无效。

## 控制参数与检查

`config.controller` 接受 horizon、max_iterations、solver_tolerance、input_rate_bounds；`config.weights` 的 q、r、s 分别作用于按边界宽度归一化的跟踪、输入幅值和输入增量代价。当前是标量权重；输入幅值代价不代表已建模的真实能耗。输入变化率边界按每个采样周期的增量定义。

默认 `solver_check` 只做一次预测求解，不运行闭环或 GA。可提供 x/reference/disturbance/previous；未提供时生成并明确标记诊断点。检查失败不会发布成通过的设计。超过决策变量或雅可比预算时仍可交付程序，但求解检查标记 not_run/resource_bound。

## 产物

`controller_design` 使用 bundle 材料接口。物理接口、实例化状态、求解检查及闭环状态在 `design.json` 中；草稿的物理接口为 null。

Python 包含 `controller.py`、原数值 MPC 求解实现、预测器、参数/调用 schema、来源清单及 README。支持的声明式、学习模型及残差模型可在仓库外新进程使用；不支持独立导出的任意 Python 模型会明确拒绝导出。

仿射模型与弹簧模板另外提供 `prediction_model.m`、`predict_step.m`、`mpc_cost.m`、`mpc_control.m` 和 `selfcheck_controller.m`。MATLAB 控制器使用 Optimization Toolbox 的 quadprog，与 Python SLSQP 是不同求解器；安装和许可在实际执行时检查。非线性直接预测仍走 Python NMPC，不伪称已导出等价 MATLAB 仿射控制器。

本轮验收实际执行了六条原生路线及五个 MATLAB 检查；四个已实例化 MATLAB 控制器的固定控制序列预测/代价/约束与 Python 一致，单点动作最大差异约 5.36e-8。草稿实际拒绝缺参数执行。所有这些检查的 `closed_loop` 均为 `not_run`、GA 次数为 0，不能据此声称原对象闭环性能达标。
