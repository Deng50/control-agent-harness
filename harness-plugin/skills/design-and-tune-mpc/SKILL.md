---
name: design-and-tune-mpc
description: 为实际对象设计和调试控制器，按跟踪、用力、平滑和超调等要求评价闭环。覆盖受约束 MPC、已有控制器调参与独立验收。
---

# 控制算法设计与调优

先按对象、控制周期、约束和运行环境查询 `control_library_search`。例如 MATLAB Function 原生控制器可加载 `build-native-mpc`，预测一致性可加载 `verify-mpc-prediction`，有幅值或变化率约束的调参可加载 `tune-constrained-mpc`。从返回的案例、代码和报告核对预测步长、控制序列、代价与验收工况，记录本次改动的参数和实际闭环结果。

用户指定独立控制函数时，在目标语言的实际调用入口维护可用控制器；用同一函数进行闭环调参，保留已经满足约束的版本。独立启动后从初态重置控制器记忆，按对象的采样和执行时序复算跟踪误差与用力。需要启动领域作业时，可查 [执行与确认](../references/execution.md) 和 `control_help` 中该操作的要求。

收到模型交接时，用 `control_preflight` 检查实际选中的版本。原 `python_callable` 需要已有完整轨迹并完成 `convert_plant_predictor`。仅设计时可交付 `controller_design`。用户需要独立运行或供下一角色采数时，用 `verify_design_delivery` 明确目标、状态、参考和所需证据，分别报告文件、加载、独立运行、单点和闭环状况。

## 预测模型

需要识别文件时读 [材料与信号](../references/materials.md)。按实际材料选择显式状态方程、运行点线性化、受支持辨识或工程简化模型。非线性MPC可以直接使用非线性预测函数；块列表和端口宽度不等于恢复全部方程。

用户只描述弹簧振子等经典对象时，提出状态、输出、控制和必要参数；无数值可交付参数化模板并标明未实例化。仅设计不要求评价Plant、GA和调参场景。代码/求解自洽检查不等于原对象闭环验证。

实际仅设计操作为 `synthesize_mpc`。输入 model、dataset、description 三选一；预测分支支持 direct、equations、identify_affine、linearize，按工具参数约束选择。弹簧模板实例化需质量、阻尼、刚度、采样时间、状态/输入/输出边界七项，缺项保持草稿。数值设计确认 signals，完整状态辨识另确认 training；不为参数化草稿补造确认或物理数值。`controller_design` 的 verified 不能代替 design.json 中的 model_instantiated、solver_check 和 closed_loop 状态。

## 目标与调参

要调参时读 [评价与报告](../references/evaluation.md)。把用户侧重点整理为外部评价配方、正尺度、权重和硬门，与MPC内部Q/R/S分开。跟踪、能耗和动作平滑必须实际进入计算；无功率模型时注明动作代理，不能把动作平方当真实能耗。

有可执行评价对象时，用独立调参/验证/测试工况比较候选。需要采集新工况时按 [采样](../references/sampling.md) 确认。未量化的目标可给方向性改善与取舍，不能宣称满足未定义阈值。保留baseline/best/last和失败候选，未改善保留最佳可行基线。

偏好调参使用 `tune_controller`，明确 inputs.model/evaluation_model，config.parameters 内部 q/r/s，以及独立的 config.evaluation。recipe 可选 tracking/energy/smoothness；四个 scales 必须显式为正并在搜索前固定，自定义 weights 要给齐 rmse/energy/tv/overshoot。energy_model 当前只有 action_proxy。要限制超调/调节时间等，用 hard_gates 或按精确输出名设置 per_output；不要把不可用的无阶跃指标写为通过。读取真实报告中的 metric_definitions、逐段明细、候选失败原因和 evaluation_hash。

## 交付

导出预测方程、参数、单位/采样、代价与约束、求解器、调用样例及真实报告。标明设计完成、实例化状态和闭环验证范围。用户只要求设计或调参时到此结束，不自动进入专家采样或轻量化。
