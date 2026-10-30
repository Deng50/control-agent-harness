# 实施记录与验收清单

目标是完整实现用户提出的三个独立、可接力的智能体，不把算法 CLI 当作最终交付。

本文件保留开发过程记录。GitHub 发布只包含源码、说明与 `validation-summary.json`；以下引用的原始运行 JSON、截图和测试日志留在本机或由检查脚本重新生成，不随仓库复制。

## 架构与复用

- 原始四个仓库保持不变。集成工程独立位于 `D:/projects/control-agent-harness`。
- Harness 使用本地源码版本 `f3803c96974ad4a94195fb97e5c637e9c2a3afe3`；采用外部 Cordis bundle/profile、领域 tools、原生 jobs 与 Web Client 扩展，不修改 agent-loop。
- Python 只提供确定性计算与产物操作，不引入另一套 LLM planner、聊天服务或智能体执行循环。
- 从 AIagent 提取 Perception 残差网络结构，从 NMPC-light 提取策略网络结构，从 mpc_agent_v5 提取遗传算法算子；源位置与哈希写入 `reuse-manifest.json`。移除原应用注册依赖，保留数学行为。
- 模型运行协议明确为离散状态转移 `x_next = f_nominal(x,u,d) + residual(x,u,d)`；连续多项式源通过显式 Euler/RK4/ZOH 适配，保存原声明与适配器；数据必须标记实测或具体仿真来源，不混用导数与状态残差单位。
- 先实现声明式、可微动力学对象和数值 MPC 适配器；后续接入 Python 工程材料/现有插件和 MATLAB，明确区分原始 Simulink 与重建方程的执行证据。
- 产物采用不可变目录、SHA-256 文件清单、父依赖与接口指纹；阶段选择与上游变更影响在 manifest 中管理。
- 前端入口对应三个独立 Harness Session，共享用户选定的项目产物目录；结构化表单和聊天 tools 使用同一领域接口。

## 分阶段实施

1. 统一协议、工作区事务、显式失败与来源验证。
2. 残差训练、预测接入、MPC 闭环与 GA 独立复验、专家采集、模仿与物理梯度微调；在线性和非线性对象上真实运行。
3. 外部 Harness bundle、原生 tools/jobs、工作区锁、全局资源限制、取消、恢复与独立训练检查点。
4. Harness Web 三入口、上下文保留、文件上传/路径选择、参数表单、日志/指标/文件及接力操作。
5. 模型导入与适配、原项目功能保留核对、完整故障与端到端测试、启动文档和演示。

## 本版本核心验收

- [x] 三阶段端到端真实计算，线性与非线性对象、不同于原三维对象的维度。
- [x] 独立阶段使用外部路径及兼容性检查；上游变化标记下游失效（协议、工作流与完整导入事务测试）。
- [x] 残差进入 MPC 与物理梯度微调，训练/调参/最终评价工况隔离。
- [x] MLP/RNN/LSTM 输入时序语义与部署一致（当前明确为恒定未来扰动预览适配器）。
- [x] 原生 Harness 工具、会话、作业、取消、重试、恢复经实际启动验证（见 acceptance.md 的各自证据）。
- [x] Web 三入口、上传/目录/参数/日志/曲线/产物接力经界面验证，含真实对话训练参数。
- [x] 训练 checkpoint、实验来源、缺失权重/非法文件/并发/失败回滚测试。
- [x] 启动命令、三类示例、协议、测试结果、依赖限制与逐条验收报告已提供；附加原项目能力边界另列 function-map.md。

## 已有实测证据（2026-10-29）

- 原生 Harness 源码固定在上述提交；8 项 `build:artifacts` 和 2 项 `build:check` 通过，未修改上游包源码。
- 14 项 Python 测试通过：接口顺序/单位/时间步、独立轨迹划分、校正项实际改变 MPC 控制量、三类网络可微、文件篡改、父依赖失效、失败保留、锁冲突、外部依赖闭包复制等。
- 4 项 worker 协议测试通过，其中实际训练取消验证来自真实 Python 进程，但该组使用测试中的进程适配器，不等于 Harness 全链路取消验收。
- `test-output/linear-001` 和 `test-output/nonlinear-001` 完成真实八阶段数值链，指标保存到 `docs/*-validation.json`。残差测试 MSE 分别下降约 98.04% 和 98.50%；轻量化策略没有达到专家全部跟踪性能，非线性微调保留初始模仿策略，没有声称提升。
- `scripts/smoke-harness.mjs` 通过真实 Web 认证、原生 RPC、人机命令、作业、Windows 沙箱、附件上传/接收以及 Python 模型导入；非法操作被拒绝。
- `scripts/smoke-harness.mjs nonlinear` 通过真实 Harness 作业完成非线性八阶段链，运行证据在 `test-output/harness-nonlinear-evidence.json`。这是框架内的真实计算，不涉及模型 API。
- Web 三入口已通过实际浏览器操作验证：独立会话与表单切换、提交工作区检查、日志/状态、MLP 与 LSTM 参数切换、上传 JSON 自动填入路径、原生目录浏览。报告/接力入口已接入，尚需完整交互验收与自动回归。
- 已提供统一启动/安装脚本、模型与阶段配置示例、README、产物协议。Windows 工作区权限辅助脚本只为已有所有者补齐沙箱前置权限，不关闭 Harness 沙箱。

## 交付范围与后续扩展

1. 更广泛的用户对象及原项目可选功能迁移核对。声明式连续时间转换、独立 PyTorch 源码快照执行已实现；任意工程依赖和原始 Simulink 仍需专用适配，不用重建方程冒充原对象执行。
2. 完整业务对话及参数调整已补充实测：真实 DeepSeek 通过原生工具完成 Python 对象八阶段链，Web 对话完成指定网络与学习率训练。继续逐条核对目标，不把单一成功案例当成任意对象自动适配证明。
3. 模型接口编辑、跨角色依赖接力、报告图表、文件预览、训练取消续训及硬退出恢复均有实际界面证据。未逐一穷举所有界面错误组合；确定性后端覆盖非法接口、文件篡改、来源不符、失败发布和恢复校验。
4. `acceptance.md` 为本版本逐项验收，`function-map.md`明确附加原功能未迁入的范围。以下保留开发期间的历史证据，最新状态以验收表为准。

## 新增实测证据（2026-10-30，UTC+8）

- 用户自定义流程支持 schema_version/steps、已有artifact与上游stage引用、文件内容快照、整体工作区锁、类型/接口/源完整性预检、失败停止和已完成阶段接续。输入变化或来源不符时拒绝恢复。
- `scripts/smoke-harness.mjs workflow-linear` 与 `workflow-nonlinear` 都通过真实 Harness Web/RPC、原生作业、Windows沙箱及八阶段数值计算；完成后再次 resume_run 保持恰好8个原始产物。证据在 `docs/harness-workflow-*-evidence.json`。
- 71项Python回归通过（67.017秒）：原14项、估计器11项、LS辨识7项、静态材料检查10项、三网络rate约束7项、训练恢复11项、工作流9项、输出硬约束2项。所有训练/数值测试实际调用NumPy/SciPy/Torch。
- 三类训练恢复网络/Adam/最佳验证状态/日志/随机状态；连续训练与中断后恢复逐tensor一致。配置、依赖、损坏文件、错误SHA均拒绝。见 `docs/training-checkpoints.md`。
- 14项Node worker/原生job持久化回归通过（48.8秒），包括真实Python取消续训、5路并发恢复只启动1次、独立host硬杀后interrupted及后续工作区写入、原生强制settlement但尚未确认清理时禁止恢复。
- KF/EKF 使用实际测量更新和Joseph协方差，后验估计状态实际驱动MPC；LS从完整状态转移数据识别A/B/E/bias，有训练-only缩放、满秩/条件数和独立评价门，不能当成仅输入输出的隐状态辨识。
- MPC新增输出硬约束。神经策略继承专家input_rate_bounds，模仿/物理训练/部署使用同一动作投影和前一动作；专家违反约束时被拒绝。未配置rate明确标记为未启用。
- Web实际验证了服务器重启后从持久任务记录读取旧的完整八阶段预检结果，原native job已不存在。界面新增对象材料检查、接口编辑、工作流文件入口和训练checkpoint选项；这些新入口的全部交互尚待逐项验证。
- 最终Python回归扩展至84项并全部通过（43.459秒），新增13项完整依赖闭包导入事务测试，覆盖I/O中断、孤立目录恢复、路径逃逸、junction与循环依赖。异常保留诊断，不覆盖原有成功产物。
- Web模型检查→接口编辑→导入已实际验证：在独立测试工作区把状态名改为 `reviewed_x1` 后提交，真实保存的 model.json 使用该名称，原输入文件未改。重启历史读取与模型导入验证不需要模型API。
- 最后修正了读取 rejected 产物报告的作业语义：检查本身成功，仍保留产物的 rejected 状态；只有实际计算验收失败才返回失败作业。5项worker协议回归重新通过（51.9秒），包含读取真实训练拒绝报告的测试。
- 两个通用流程的最新数值报告在 `docs/harness-workflow-linear-validation.json` 与 `docs/harness-workflow-nonlinear-validation.json`，由明确run id提取，并重新校验全部产物与依赖。查看报告不重新训练、不按目录时间选“最新”。

## 暂停后续开发的新增证据（2026-10-30，UTC+8）

- `test-output-regression-resumed-elevated.log`：整套106项Python回归通过，61.860秒。外层受限测试中的两个junction权限失败，已在支持该Windows文件操作的测试进程中复验13项导入事务全部通过，没有修改或跳过用例。
- 连续适配测试扩展为10项并通过，包括新增“LS拟合转移系数不得再次积分”。它是在整套106项发现测试后新增的1项；不把重跑的已有9项重复计入测试总数。
- 强制退出恢复5项测试包含真实 `os._exit` 与精确CPU续训；16项Node用例（worker 5、history 9、roles 2）分别通过，原生作用域隔离测试实际加载上游Tools/SystemPrompt/NativeExecutor。
- 真实Web/RPC创建三个 `control-*` 预设会话，完成4类依赖接力、报告读取与文件预览。证据 `harness-role-handoff-evidence.json`。修复原生预设缺少上游必填配置造成的激活失败。
- Web实际操作验证校正模型→控制器、控制器→轻量化：自动填入确切预测/评价模型，列出待设置工况；切换到独立角色草稿后参数保留。`web-controller-handoff.png` 为真实截图。
- Web从重启前的持久记录自动恢复控制器报告；从已校验清单读取controller.json成功。策略对比切换condition-400/410后表格和曲线同步，保留优化没有提升的真实结论。截图 `web-artifact-file-preview.png` 与 `web-policy-comparison.png`。
- 用户提供的凭据通过原生credentials接口存入忽略的私有目录，没有写入代码或测试配置。真实DeepSeek官方请求返回401；`harness-model-dialogue-evidence.json` 明确记录passed:false、0次领域工具执行。未宣称模型对话验收通过。
- Web真实训练取消后，从第838轮检查点续训到840轮，优化器累计更新15084→15120，产物verified且工作区锁释放。数值与来源证据见 `web-cancel-resume-evidence.json`。这是通过表单明确设置总目标轮数的新任务，不冒充原进程继续运行。
- 硬退出恢复界面已实测：实际子进程在第2轮 `os._exit(73)`；Web通过原生作业核验死亡、归档并采用检查点，另一个训练作业从2轮恢复到25轮，产物verified。证据 `web-hard-exit-recovery-evidence.json`。
- 最新整套108项Python回归通过（57.779秒），日志 `test-output-regression-coverage.log`。新适用范围报告只统计训练分区；验证/测试极值不会改变该范围。另经真实Harness外部产物导入→25轮残差训练→报告检查验证，证据 `harness-coverage-evidence.json`；它记录288条训练样本的4个变量范围，不宣称外推有效性。
- Web已实际展开训练覆盖表，显示288条训练数据的状态/控制/扰动实际范围、声明边界及训练/验证/测试MSE。截图 `web-model-applicability.png`；此独立25轮案例测试MSE下降96.26%，与早先100轮以上案例分别记录。
- 用户澄清先前UUID为tracking ID，提供实际API Key后，经原生凭据接口更新，无需重启。三个角色均经真实DeepSeek模型调用各自control工具创建inspect_workspace原生作业、等待结果并读取8个真实产物；`harness-model-dialogue-evidence.json`三项passed:true，401鉴权问题已解决。此验证不替代完整训练业务对话。

## Python 对象与业务对话验收

- 独立 PyTorch 转移/导数函数接入统一模型协议，保存源码、参数和运行依赖。注册检查 CPU float64、批量/单样本、输入纯度、重复性、独立导入随机性以及转移/输出自动微分与有限差分一致性；静态检查和工作流预检仍不执行源码。详见 `python-dynamics.md`。
- `scripts/smoke-business-dialogue.mjs` 经真实 DeepSeek 对话预检、运行并等待八阶段 Python 对象流程，核对8个已验证产物；代理读取生成的报告，未将神经策略较差的测试表现说成提升。证据 `harness-business-dialogue-evidence.json`。quiet配置下再次通过后，`harness-python-workflow-validation.json`另记录精确依赖校验、执行来源与权重重新加载验证。
- Web 完成静态检查→接口采用→显式 Python 注册→参考仿真采集→真实对话调参，保存网络6→12→12→3、学习率0.008、100轮、400次更新。`web-python-parameter-evidence.json`和`web-python-dialogue-report.png`记录实际参数与测试误差。
- 该界面联调暴露了上游默认作业完成会唤醒模型的问题。三个角色改用原生 quiet 完成通知，并要求对话主动等待自己的作业；`harness-quiet-forms-evidence.json`验证三个表单均未开启模型回合，随后三角色真实模型工具对话仍全部通过。
- 对话测试前多余的默认参数训练被取消，只有用户明确指定的训练发布产物，未将被取消的尝试算作成功。持久历史保留两次尝试以便审计。
- 补充 NPZ 数组头形状/字节预算检查，防止小ZIP成员声明巨大数组导致预览过量分配。控制器来源报告现在分别记录预测与评价模型的真实执行适配器，兼容声明式、连续源和 Python 源码。
- 最终回归：Python116项通过（111.056秒），Node16项通过（39.325秒）；日志 `test-output-release-regression.log`、`test-output-release-node.log`。原生上游 Git 工作树无跟踪文件修改。
- 最终启动脚本使用本工程 `config/web.overlay.yml` 配置原生目录选择器，不依赖上游测试目录；实际重启后原生工作区检查通过，`harness-startup-evidence.json`记录此次交付启动。
