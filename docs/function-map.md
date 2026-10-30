# 功能与复用对应表

本集成复用算法并替换智能体运行基础设施，没有嵌入原网页或启动三个旧服务。原项目保留在原路径。实际提取类、源行号和文件 SHA-256 见 `reuse-manifest.json`；新增适配器、约束校验和接力协议位于本集成工程。

| 原项目能力与代码入口 | 集成实现 | 当前边界 |
| --- | --- | --- |
| AIagent：`pgac_auto_train_agent_v1/app/skills/input_assets_skill.py`、`env_adaptive_skills.py` 的材料、接口与环境分析 | `model_inspection.py`、`contracts.py`、Web 接口确认、`python_model.py` | 静态检查与显式执行分开；未知信号语义需用户补充 |
| AIagent：`kernels/pgac_core.py` 的 `PerceptionNet`、物理模型与误差学习 | 源网络提取到 `reused/perception.py`；`residual.py` 与 `CorrectedDynamics` 训练并接入误差项 | 通用离散次态残差监督学习和声明的物理约束；不冒充通用 PDE PINN |
| AIagent：`pgac_skills.py`、`pgac_report_skills.py`、`artifact_register_skill.py` 的训练/评价/报告/登记 | Harness 作业 + 训练检查点 + 独立验证报告 + 不可变产物协议 | 原专用热管理 Actor/MPCCritic 完整训练环不是原样迁入；策略阶段采用有明确名称的可微 rollout 梯度 |
| mpc_agent_v5：`model_reading.py`、`simulink_inspection.py`、`interface_adaptation.py` 的对象核对 | JSON/Python/SLX/MDL 静态检查、显式接口、连续/离散适配、源码快照 | SLX/MDL 尚无原生 MATLAB 仿真适配器；数学重建与原模型执行分别表述 |
| mpc_agent_v5：`controller_design.py`、`simulation.py`、`analysis.py` | `control.py` 的可执行 SLSQP MPC/NMPC、输出/状态/动作/可选速率约束与闭环报告 | 以已注册的可调用模型为前提，不凭场景名称猜物理规律 |
| mpc_agent_v5：`tuning.py` 的遗传算子与独立复验 | 提取的 `reused/genetic.py`、真实候选仿真、独立验证选优；`identification.py` 的有条件最小二乘 | 无独立改善时保留基线；LS 用于满秩状态转移辨识，不套作所有参数的万能优化器 |
| mpc_agent_v5：`estimator_design.py`、`estimator_pipeline.py` | `estimation.py` 的 KF/EKF、Joseph 协方差、可观性检查、估计状态实际驱动 MPC | 测量映射必须明确；不宣称仅 I/O 的任意隐状态辨识 |
| NMPC-light：`generic_data_collector.py`、`generic_dataset.py` 的工况采集与划分 | `policy.py` 的专家闭环采集；按独立工况拆分，保存状态/参考/扰动/动作 | 工况需显式给定；不把随机仿真称为实测 |
| NMPC-light：`plugins/networks/{mlp,rnn,lstm}.py`、`train_nn.py` | 原网络结构提取，训练集预处理、早停、权重与训练状态保存、闭环评价 | 当前循环网络使用未来恒定扰动预览，不是历史观测堆栈；输入协议随产物保存 |
| NMPC-light：`pac_finetune.py`、`imitation_closed_loop.py`、`compare_archs.py` | 可微模型 rollout 策略更新；同一测试工况比较专家、模仿与优化策略的误差/约束/平滑性/耗时 | 不是 PPO/SAC，也无独立学习 critic；支持逐个训练三种架构，尚未提供原批量架构比较工作台 |
| 三个项目各自的 Planner、模型客户端、HTTP/SSE/WebSocket、任务记录与网页 | 固定上游 Harness 的 agent loop、原生 preset/tools/session/jobs/attachments、Web bundle | 一套运行框架；角色工具有作用域限制，公共工作流工具可显式串联 |
| 原记录文件与跨智能体登记 | `manifest.json`、版本化产物、哈希依赖、外部目录导入、下游失效、明确交接 | 旧私有格式需显式转换，不能仅凭相同维度直接混用 |

尚未迁入本版本的附加工作台包括：NMPC-light 动态 Plant/Controller 插件生成、安装/版本回滚、工况推荐与补采诊断；MPC 项目的 MATLAB 源码生成/编辑修复与原生 Simulink 运行包；AIagent 专用热管理完整联合训练及其环境代码补丁。这些原功能仍可在原仓库使用，不作为本集成的已验证能力。

新增集成能力包括：跨进程工作区互斥、Host 内数值并发限制、真实取消/强制退出恢复、精确训练续训、依赖闭包事务导入、源码和配置快照、只读产物文件预览，以及 Harness 三角色的实际 DeepSeek 对话与八阶段编排。相应证据见 `acceptance.md`。
