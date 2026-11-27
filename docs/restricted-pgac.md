# 受限可微改写与策略微调

本路线复用真实模仿网络，通过可微下一状态模型计算多步控制代价对策略参数的梯度。实现是带模仿保护的物理 rollout 策略梯度；报告不声明已经实现学习型 Critic、PPO 或 SAC。当前验收状态以 [验收记录](acceptance.md) 为准。

## 模型资格

`validate_differentiable_model` 使用 `inputs.model` 指定 Torch 候选，`inputs.evaluation_model` 指定原对象，`inputs.dataset` 指定绑定原对象的真实 `sampled_dataset`。原 Python/MATLAB/Simulink 的采样复用既有路径和确认机制；此验证阶段读取不可变记录，不再次调用原仿真器。

配置明确 `mode`、验证分区、轨迹前缀长度、原数组中的探针行、数值容差、三个物理有限差分步长、域及已知开关面。验证包括所有选定行的单步、不同输入的混合 batch、自主状态递推，以及下一状态和输出对 x/u/d 的完整 Jacobian。初期冻结夹具使用50步、等价 `atol=1e-6/rtol=1e-5`，梯度 `atol=1e-5/rtol=1e-3`，步长 `1e-4/1e-5/1e-6`。

`exact_rewrite` 和 `existing_torch` 保留精确对照结果。`bounded_surrogate` 另需训练数据覆盖域及 `proxy_budget` 的单步/递推状态和输出 RMSE 物理预算；近似资格不能更改精确等价标记。域是训练边缘范围内的明确盒，不证明联合覆盖或全域物理精度。代理的梯度始终标记为代理梯度。脱离计算图、错误数值和错误 batch 广播会被拒绝；真实饱和零梯度可通过，开关边界单列报告。

资格回执 `differentiable_validation` 保持非激活，并绑定原对象、候选模型和参考数据的确切版本。条件 ODE PINN 的轨迹接口没有任意状态转移能力，不能用于本路线。

## 训练候选

`optimize_external_policy` 输入 `policy/model/validation`。策略必须是实际外部模仿产物，并能解析到可执行的原专家和原对象；仅有 CSV/XLSX 记录的普通模仿模型不凭空获得物理闭环。此首期路线使用当前因果特征、完整状态观测和明确的观测/状态映射；历史窗口及部分可观测策略继续使用其原有模仿接口。

可设置 `epochs`（累计目标）、`learning_rate`、`batch_size`、`horizon`、`seed`、`q/r/s`、`imitation_weight`、`constraint_weight` 和 `domain{x,u,d}`。保持原网络结构、Scaler、实际上一动作、幅值及每次控制决策的变化率投影。控制周期包含多个对象步时保持动作/扰动，逐对象步推进已声明的下一状态模型，不额外乘 dt。

`original_evaluation` 在训练前冻结，包含所有明确的原对象验证/测试 `scenarios`，以及独立的 `evaluation{q,r,s,comparison_atol}`；比较精度固定为 `1e-12`。原对象评价使用下一控制时刻输出误差、动作和动作增量的范围归一化二次代价，按步取均值、按工况等权。它不包含训练中的模仿保护项。需要计时时另提供 `benchmark`，保持 CPU 单线程、batch1 并区分预热。

策略从实际模仿张量初始化。仅策略网络参与 Adam 更新，可微模型参数固定并保留状态/动作梯度。预测模型验证集选择一个检查点，初始模仿基线也参与选择；测试不更新或选择权重。域、来源、Scaler、动作约束、代价和原对象评价配置纳入恢复合同。越界或非有限结果保留失败记录、已提交 checkpoint 和原基线。

训练输出 `external_pgac_candidate`，保存初始/选中权重、Adam/RNG、日志和独立包；它仍是非激活候选。

## 原对象验收

`prepare_handoff` 返回候选绑定的 `evaluate_external_pgac` 输入和冻结配置。评价不能更换原对象、工况、目标或权重。专家、模仿和候选分别以自己的状态和实际上一动作闭环运行，报告每个版本的真实轨迹。

原对象验证接受可行且更优的候选，或者保留可行基线；两者均不合格时报告无合格回退。选择在任何测试运行前保存。最终测试只接受或拒绝该固定版本；候选退化时可保留本来已通过原验证且本次测试可行的模仿基线，不能转选另一个 epoch。

Python 在本地运行原对象。MATLAB 使用一个原生请求绑定的本地批次：先完成三个版本的验证，保存选择，再完成测试，由 Python 归档并独立复算。没有逐对象步 MCP 或隐式 Engine。候选的覆盖域检查必须覆盖原对象内部步；控制时刻的边界记录不能证明未记录内部步始终在域内。

最终 `external_optimized_policy` 仅在原对象门通过后激活，明确是优化候选还是保留基线。独立推理包和报告保留操作域及使用范围；本阶段自动回退结论针对实际训练和验收过程，不构成未来任意域外运行的实时安全保证。

## 学生在线采集与片段回放

拿到原对象、专家、实际模仿包及可微模型资格后，可在 `optimize_external_policy` 的 `config.replay` 中选择 `online`。它在可执行 Python 原对象上用当前学生重新采集训练回合，利用冻结的可微模型从经验池起点重新展开策略。不传 replay 时沿用原专家起点模式。`expert_segments` 可以在同一组训练示范上检验片段采样产生的影响。

| 训练选项 | 工程含义 |
|---|---|
| `mode` | `online` 刷新学生经验，`expert_segments` 使用专家片段。 |
| `mode=balanced_starts` | 只改变专家起点在工况族与阶段间的配比，保留旧版固定外生量与预测时域，供单因素消融。 |
| `capacity_steps`、`refresh_every` | 学生池的控制步数容量、采集刷新周期。 |
| `student_fraction`、`updates_per_epoch` | 学生片段占比、每轮更新预算。 |
| `scenarios` | 每项给出名称、族、初态、上一实际动作、步数、参考和扰动。可使用控制步时序 `reference_sequence` 和 `disturbance_sequence`，用 `observation_noise_std` 和 `noise_seed` 指定传感器噪声。 |
| `family_weights`、`expert_families` | 训练族的相对权重，以及专家训练轨迹 ID 到工况族的映射。 |
| `phase_weights` | 可选的正权重，按实际出现的 initial、interior、near_boundary、reference_change、reversal 决策阶段配比。未指定的阶段权重为 1。 |

保留真状态、实际动作与上一动作。时序在每个控制决策点推进，策略只读取当拍部署可得的信号。对象有无法在片段起点恢复的隐藏状态时，应先建立可验证的回放接口；受限入口继续要求当前全状态映射。模型域外或资格缺失按既有失败规则处理。

`original_evaluation.condition_source=fresh_holdout` 使用新验证与盲测场景，且与已使用的专家分区独立。省略时保持原有绑定条件。`original_evaluation.evaluation.min_relative_improvement` 和 `max_worst_case_relative_regression` 分别约束整体改善和最差单一工况退化，均在拟合前冻结。保留共同验证/测试种子。学生在原对象上的训练采集与原对象独立验收使用不同工况。比较各组时同时报告闭环 J、物理约束可行性、逐工况代价、峰值和执行成本。

## 复现轻量化角色会话

启动配置好模型的本地 Harness Web。以下命令适用于弹簧公开测试材料，模型提供方接收工作区内的材料和角色工具结果。测试人员须先授权该材料与服务。运行时选择 `control-lightweight` 和已配置的 `deepseek-official`。第一次运行应使用一个尚未创建的测试目录。准备步骤生成物理接口、冻结预算和验收文件；观察器只提交普通用户请求，工程操作由真实模型选择。它使用本地测试夹具确认和冻结配置逐项相同的原生确认卡。实际用户仍通过界面确认。

```powershell
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -B scripts/bench_online_pgac.py prepare --cases spring --root test-output/dialogue-fixtures
$task = 'test-output/dialogue-spring-new'
$material = 'test-output/dialogue-fixtures/spring/materials'
$startupLog = 'harness-web.log'
node scripts/run-online-pgac-dialogue.mjs --action start --workspace $task --materials $material --web-log $startupLog
node scripts/run-online-pgac-dialogue.mjs --action drive --workspace $task --web-log $startupLog
node scripts/run-online-pgac-dialogue.mjs --action verify --workspace $task
```

`inspect` 读取已发生的模型调用、技能和作业。`drive` 在当前模型回合结束后返回；待澄清时可通过会话界面回复，或用 `--action reply --text '…'` 回复后继续运行 `drive`。完整观察记录和复核摘要分别写在工作区同级的 `<名称>-observer.json` 与 `<名称>-verified.json`。最后一步仅使用本地文件，以隔离 Python 载入本次会话交付包，在原对象及冻结盲测工况上逐步比较动作、状态与 J。原生 smoke/recovery 单独检验工具合同和中断恢复。数值基准及自然语言实测记录见[弹簧和电机案例](../examples/pgac-online/README.md)与[会话证据](../examples/pgac-online/dialogue-evidence.json)。

本次实测中，测试夹具代为确认六份冻结配置卡。初次观察器过早创建了 Windows 工况卡目录，测试人员移开这个夹具目录，取消一次额外权限请求并通知角色复用原对象和专家继续。观察器现等待角色创建工况账本，再写测试确认标识。首次独立加载将 Python 字节码缓存写入包，严格打包检查拒绝归档；角色排查并重试成功，复核程序现禁用字节码写入。原会话、归档失败及成功回执均保存在测试记录中。

角色应向用户明确区分可用专家决策数、实际专家源函数执行数和计时调用。Python 适配器每个成功决策执行三次一致性核验，优先回读报告的 `expert.source_invocations_this_adapter`。本例采集 720 个决策共调用 2,160 次，训练分区的 432 个决策占 1,296 次。验证/盲测另有 576 个专家决策、1,728 次执行；两次计时配对的预检、预热和采样共 104 次原函数执行。学生采集 1,152 个对象控制步，额外 MPC 调用为零。原答复把 1,296 个采集及验收决策写成了求解数，归档中保留该原答复及成本勘误。这些数字记录专家函数执行，识别其中多少次进入约束优化器需由专家程序另行记录。
