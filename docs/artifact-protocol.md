# 产物协议 v1

工作区由 `manifest.json` 标识，包含 `schema_version`、`project_id`、创建时间、按类型选择的 `active` 和完整 `artifacts` 索引。计算阶段先写 `.staging/<type>-<id>`，发布后移动到不可变 `artifacts/<type>-<id>`。失败阶段保留在 `failed/`，其中可包含检查点和失败说明。失败不会替换原有成功产物。

每个产物包含 `artifact.json`。字段包括 `id`、`kind`、`schema_version`、`created_at`、`interface`、`interface_hash`、`parents`、`files`、`validation`、`preprocessing`、`source`、`runtime`。文件清单保存 SHA-256，manifest 另外记录元数据哈希。读取模型、控制器、数据集或策略前，会核验文件和递归父依赖，拒绝缺失权重、内容修改、未经接受或已过期的产物。

`interface` 明确状态 `states`、控制 `controls`、输出 `outputs`、扰动 `disturbances` 的有序变量列表。每个变量包含名称、单位和上下界；另有正数 `sample_time` 和 `time_semantics: discrete_state_transition`。维度由列表推导。接口兼容性要求变量顺序、单位、采样时间、边界和时间语义一致，不自动变换单位或时间步长。

声明式动力学支持 `discrete_polynomial` 的 A/B/E/C、偏置与多项式项，以及连续多项式源的显式 Euler/RK4/仿射 ZOH 适配。独立 Python 源支持 `python_torch_transition` / `python_torch_derivative`，用冻结源码和明确的 Tensor 函数执行，输出可由 C 矩阵或声明的可微输出函数计算。各适配器对外统一提供 `x_next = f(x,u,d)`。连续源另保存原始声明与转换记录，详见[连续模型](continuous-dynamics.md)与[Python 动力学](python-dynamics.md)。校正模型执行 `f_nominal + residual`；网络输出单位为每个采样步的状态增量，不是导数。当前残差训练为带状态界限和可选线性平衡约束的监督学习，不是通用 PDE PINN。

| 类型 | 关键文件 | 下游使用 |
|---|---|---|
| nominal_model / reference_model | model.json；Python另有source.py、python-validation.json；连续源另有transition_adapter.json | 名义对象 / 明确指定的评价仿真对象 |
| identified_model | model.json、report.json | 满秩状态转移数据的最小二乘辨识模型 |
| transition_dataset | transitions.npz、split.json | 残差训练 |
| corrected_model | nominal.json、network.json、scalers.json、model.pt、report.json | 校正动力学 |
| controller | controller.json、report.json | 在线 MPC 与专家采集 |
| estimator | estimator.json、report.json | KF/EKF 测量反馈控制接口 |
| expert_dataset | expert.npz、split.json、collection.json | 模仿训练与策略评价 |
| imitation_policy / optimized_policy | network.json、scalers.json、policy.json、policy.pt、report.json | 神经控制策略 |

具体文件名以产物的 `files` 清单为准。数据划分按轨迹/工况执行；归一化统计仅来自训练集。调参、候选复验和最终测试工况不得重复。相同工况改名不能规避重复校验。

跨工作区交接使用 `import_artifact`，输入 `artifact_path` 为含 `artifact.json` 的完整产物目录。导入会校验并复制完整依赖闭包，保持 ID、元数据和文件哈希，之后不依赖源目录仍存在。上游活动产物被替换时，受影响的下游索引变为 `stale`；重新验证前不作为活动产物使用。

加载示例（工作区和 ID 换为实际值）：

```python
from control_harness.artifacts import Workspace
from control_harness.modeling import load_model
from control_harness.control import load_controller
from control_harness.policy import load_policy

workspace = Workspace("D:/projects/control-agent-harness/workspaces/project-01")
model = load_model(workspace, "corrected_model-实际ID")
x_next = model.step(x, u, d)
controller = load_controller(workspace, "controller-实际ID")
u_mpc, mpc_diagnostics = controller.act(x, reference, d, previous_u)
policy = load_policy(workspace, "imitation_policy-实际ID")
u_nn, inference_diagnostics = policy.act(x, reference, d, previous_u)
```

MLP 输入为当前状态、输出参考和扰动。当前 RNN/LSTM 适配器使用明确声明的恒定未来扰动预览，并非历史窗口模型；输入维度还包含预览长度乘扰动维度，部署时必须提供相同信息。

原生 Harness 作业 ID 是运行期句柄；持久 attempt 另存请求快照、事件、结果与恢复关系。服务器重启后可读取结果或明确的 interrupted 状态，但不会谎称旧进程仍在运行。三类训练均提供带配置/依赖/运行时签名的 `resume_checkpoint`，恢复 Adam、最佳验证状态、轮数及随机状态。工作流接续另用 `resume_run`。参见 [工作流](workflows.md)、[训练检查点](training-checkpoints.md) 和 [Harness 作业恢复](../harness-plugin/JOB-HISTORY.md)。

声明式模型接口的状态、输入和输出上下界均参与控制约束验收。输入变化率若由专家控制器声明，则由神经策略继承；报告的 `constraint_checks.input_rate` 区分启用和未配置，不以未配置时的数值 0 宣称约束通过。见 [神经策略约束](neural-policy-action-constraints.md)。
