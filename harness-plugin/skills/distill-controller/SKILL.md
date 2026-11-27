---
name: distill-controller
description: 用已有MPC专家及对象或已有专家数据进行采集、神经网络模仿和可选PGAC微调，交付轻量控制器及同工况比较。适用于控制算法轻量化入口。
---

# 控制算法轻量化

对已确认的采集方案，登记获准的工况和训练轨迹数。自行编写采集脚本也按这个数量执行。若已有数据暴露工况覆盖不足，说明拟补的轨迹、预期收益和额外代价，等用户同意后增采。

用 `control_library_search` 查询原专家、观测和所需交付物。采数可加载 `collect-expert-trajectories`，策略模仿可加载 `clone-control-policy`，具有可微对象和已验证模仿权重时可查 `warmstart-pgac`。阅读搜索结果里的训练代码和闭环对照报告，再为本次任务记录相同工况下专家与学生的约束违例、跟踪误差和推理耗时。

用户指定独立部署接口时，训练脚本生成目标语言的可调用策略，验证直接调用部署目录中的版本。如果用户开放策略形式，可以从现有记录拟合低计算量基线，用整条留出轨迹判断是否值得扩大网络或搜索超参数。分别计算离线动作误差与学生自行反馈的闭环误差。用公开输入重训时核对依赖、路径、动作单位与约束。需要登记专家或发起领域作业时，可查 [执行与确认](../references/execution.md) 和 `control_help` 中该操作的要求。

收到 `controller_design` 时，请设计角色按用户工况核验 `workspace_controller` 交付。收到已核验控制器后，用 `prepare_handoff` 取得精确预测器及来源，复核 `collect_expert` 工况。存在直接绑定的原始 `python_callable` 时可在该适配器上闭环采集。报告中写明实际执行的对象版本。跨工作区先导入完整依赖，再由目标会话确认本次采集。

## 数据与模仿

- 已有专家数据：按 [材料与信号](../references/materials.md) 确认观测、参考、扰动和动作。可直接规划模仿，不虚构对象闭环验证。
- 已有对象和专家：核对reset、动作、内部记忆、控制周期、单位及求解失败状态。按 [采样](../references/sampling.md) 推荐或接收工况，确认后在本地数值程序内闭环采数；不要逐采样点让LLM调用MCP。

外部程序先登记材料，再用 `register_expert` 保存六输入/三输出适配声明；Python/MATLAB 函数都接收 observation/reference/disturbance/previous_action/memory/parameters，返回 action/next_memory/status。返回动作按专家坐标映射到对象控制通道；previous_action 和 rate bounds 始终是对象控制通道顺序。先查实际 `control_help`，原生工具保留其支持边界。

`collect_external_expert` 直接选择 expert 和 plant，确认 signals/sampling 后执行。普通模式先根据对象性质、范围及专家周期提出参考目标、扰动和初值候选；专家模式编辑同一 scenarios 配置。每个工况明确 initial_previous_action、控制决策步数；对象在相邻决策间按原 dt 保持动作。内部记忆每轨重置并保存诊断，失败状态和越界会终止并保留合法前缀。默认学生特征使用已声明可获得的观测、参考、扰动和上一动作；显式选用 simulation_only 观测时保留仿真使用范围，原始真状态与专家记忆不自动成为输入。

已有 CSV/XLSX 专家记录用 `import_expert_data`，确认 DataInterface、当前动作目标、因果历史、动作约束和来源。当前目标动作只可作为过去时刻的特征；独立 previous_action 列已经是上一实际动作，需声明其映射并保持原时间含义。没有原控制器保留工况时记录新的整轨划分，无内部 controller ID 的数据仍可进入后续训练。采数和数据登记完成不代表已经训练了神经网络。

读 [网络与物理训练](../references/training.md)，提出或接收网络与超参摘要，确认后训练。策略输入必须在部署时可获得；历史窗口、隐藏记忆和未来扰动预览是不同含义。用户只要求模仿，交付完整策略包后结束，不强制PGAC。

外部 `external_expert_dataset` 用 `train_external_imitation`，内置旧专家 `expert_dataset` 才用 `train_imitation`，不要把外部数据伪装成内部控制器的后代。网络摘要列出隐藏层宽、激活、学习率、batch、累计epochs、早停、动作NRMSE尺度及所请求的门限。最终报告分别列出原始NN输出和实际幅值/变化率投影后的动作误差，投影不隐藏原始越界比例。缩放和动作约束随独立策略包交付；变化率推理使用实际上一动作。

需要原对象比较时，先对选定策略调用只读 `prepare_imitation_evaluation`。它从实际数据祖先解析原专家、原对象及验证/测试工况；展示其 `config_suggestions` 的具体工况、计时设置并确认signals/sampling后，交给 `evaluate_external_imitation`。专家与NN分别使用自己的状态和上一动作闭环，不能把记录数据上的预测误差当作自主闭环结果。数据记录没有可执行原专家/对象或缺少历史反馈适配时，保留不适用原因，不编造控制曲线。比较为独立报告，不改写已训练策略、不选择新权重；如退化如实报告。

## 可选策略微调

用户要求时优先使用已验证可微对象。简单方程改写须通过原模型的单步、多步、batch及梯度对照；否则可用明确标记的可微代理。Torch包装或requires_grad不能恢复断开的梯度链。复杂Simulink工程不承诺无损转换。

从实际模仿权重初始化受支持优化器，记录其真实算法，不声称未实现的Critic/PPO/SAC。在原对象上独立复验；代理外推或控制退化时保留模仿基线。

外部模仿的受限微调用三个独立领域操作：`validate_differentiable_model`、`optimize_external_policy`、`evaluate_external_pgac`。先用原对象真实采样得到的 `sampled_dataset` 验证分区，对明确的 Torch 下一状态模型做单步、混合 batch、自主递推和完整 Jacobian 对照。Matlab/Simulink 原对象通过已有本地批仿真提供记录，不用逐点MCP。有训练覆盖域及明确误差预算的代理必须标记 `bounded_surrogate`，近似合格与精确等价分开报告；PINN条件轨迹不能代替状态转移。

`optimize_external_policy` 输入实际 `external_imitation_policy`、模型和绑定它的 `differentiable_validation`。此首期受限路线要求全部特征当前可用、完整状态可观测及明确物理映射；不能从特征宽度推断状态数。先展示实际学习率、累计epochs、batch、horizon、内部q/r/s、模仿保护、域，以及 `original_evaluation` 中冻结的原对象验证/测试工况和独立评价权重。可以用已有 `prepare_imitation_evaluation` 解析原对象工况，再按本操作的q/r/s评价格式组成摘要。确认后只训练策略参数，模型参数固定且保留输入梯度。

该训练输出非激活 `external_pgac_candidate`。用 `prepare_handoff` 取得精确的冻结评价配置，确认后执行 `evaluate_external_pgac`：原对象验证只接受一个合格候选或保留可行模仿基线，先保存选择再进行测试；测试只接受/拒绝该版本，不再选择epoch。基线自身不合格时报告无合格回退。保存专家、模仿、候选和实际选择，分别列出预测模型验证与原对象结论。训练、原对象验证或最终测试中的代理越界均不能当作合格优化结果。

按 [评价与报告](../references/evaluation.md) 比较专家、模仿和优化策略的跟踪、约束、动作及推理耗时。保存所有版本，交付权重、缩放、输入顺序、记忆接口、推理样例和报告。

## 在线闭环微调任务

已提供完整 JSON 训练或评价配置时，把该操作的 JSON 对象保存为工作区根目录中的普通文件。在 `control_preflight`、`control_prepare_review` 和 `control_lightweight` 使用同一个 `config_file` 文件名，并省略内联 `config`。实际内容进入工程摘要。文件更新后重新审阅，避免在对话中逐字转录时序数据。

用户提供可执行对象和 MPC 并要求模仿后继续在线 PGAC 时，加载本技能，查询 `warmstart-pgac` 的采样与验收案例。在受支持的 Python 对象上使用正式操作。依次登记原对象和专家，`collect_external_expert` 采集专家，`train_external_imitation` 冻结模仿包，`validate_differentiable_model` 核验滚动模型，`optimize_external_policy` 训练在线候选，`prepare_handoff` 和 `evaluate_external_pgac` 在原对象上验收。可复用同一对象已有的模型资格和已登记的专家材料。用户只要求模仿时，交付模仿包即可。

在训练摘要中列出 `replay.mode=online`、原对象和滚动模型版本、训练工况族与独立验证/盲测工况、学生闭环刷新周期、经验池步数、学生片段占比、每轮更新数，以及新增 MPC 求解和学生仿真的预算。用 `control_help` 取得当前操作 schema 和适配边界。静态记录、映射不全或隐藏动态未能恢复时，说明可完成的阶段。交付时分别说明验证是否选中、独立测试是否通过，以及是否确有改善。

用户优先改善跟踪 RMSE 时，在训练前确认输出名称、物理单位、相对模仿基线允许的动作与动作增量 RMS 增幅，以及最差工况门限。将这些要求写入 `original_evaluation.evaluation`。使用同一目标选 checkpoint、做原对象验证和盲测。最终按族报告 RMSE 的变化、两项预算余量、越界峰值和交付版本。配置字段及计算口径见[评价与报告](../references/evaluation.md)。

需要独立复验交付时，先对验收选定的产物使用 `prepare_bundle_archive`，核对回执、ZIP 和登记成员摘要。在工作区新建临时目录解包，以有推理依赖的 `python -I -B` 进程从解压目录加载策略并重放冻结工况。导入 Python 模块可能写入字节码缓存，封存的 `artifacts` 目录只作只读核对。报告 ZIP 的哈希、加载路径类型、原对象指标及逐拍差异。

回答“实际求解多少次”时，按采集、验收、计时、学生训练分列。专家的独立控制决策看成功行数；Python 适配器做重复和新模块一致性检查，实际专家函数执行看 `expert.source_invocations_this_adapter`。计时直接调用专家的上下文预检、预热和测量另行计算。区分闭式 MPC 求解、调用约束优化器以及整个专家函数的执行。原程序未对优化器分支计数时说明可核实的层级。
