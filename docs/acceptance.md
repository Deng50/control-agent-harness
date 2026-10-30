# 当前验收状态

本文件记录当前仿真集成版本的逐项证据与适配边界。真实 DeepSeek 已完成三个角色的工具对话，以及独立 Python 对象的八阶段业务编排；Web 对话调参、取消续训和硬退出恢复均有实测记录。原项目全部附加工作台不等于已迁入，详见[功能对应表](function-map.md)。

GitHub 版本只提交[脱敏数值摘要](validation-summary.json)。下表中的原始运行 JSON、截图、工作区和日志默认保留在开发机器，不随源码上传；它们的文件名用于说明本机验收来源。克隆仓库后可按 README 的检查命令生成新的本机记录，原训练权重与 API 凭据需要另行准备。

| 要求 | 当前证据 | 结论与边界 |
| --- | --- | --- |
| 保留Harness架构，统一智能体运行 | 固定上游提交；外部bundle、原生preset/tools/jobs/session/attachments；`harness-model-dialogue-evidence.json` | 已证明三个角色的真实模型请求、领域工具与作业读取，不另建LLM循环或聊天后端 |
| 三角色独立与接力 | `harness-role-handoff-evidence.json`、`harness-business-dialogue-evidence.json`，原生作用域测试，真实Web截图 | 预设会话、依赖接力、真实工具调用和完整八阶段业务对话已验证 |
| 识别并确认对象接口 | JSON/Python AST/SLX/MDL检查、接口编辑后导入、`test_model_inspection.py`、`test_python_model.py` | 检查不执行源码；声明式模型或经显式注册的独立 PyTorch 源码可执行；Simulink 仍为静态检查 |
| 校正动力学实际进入下游 | 线性/非线性8阶段报告，`test_contracts_and_handoff.py`、连续模型真实闭环测试 | 已验证名义+离散状态残差进入MPC与策略梯度；并非通用PDE PINN |
| 连续与离散时间语义 | `test_continuous.py`共10项；解析解、独立ODE、梯度、KF、LS | Euler/RK4、仿射精确ZOH可执行；源与转换记录分开保存；残差为状态单位 |
| 真实MPC/NMPC及有界GA调参 | `control.py`与两类原生工作流数值报告、输出/rate约束测试 | 使用真实候选闭环和独立复验，未改善保留基线；最小二乘另用于满足条件的辨识 |
| 状态不全测量与辨识 | `test_estimation.py`、`test_identification.py` | KF/EKF估计实际驱动MPC；LS需要完整状态转移数据，不宣称仅I/O隐状态辨识 |
| 专家→模仿→策略优化 | 真实专家采集、三种网络、动作约束与续训测试、策略比较图 | 支持MLP/RNN/LSTM；循环网络使用明确的未来恒定扰动预览；物理梯度法准确命名，不宣称PPO/SAC |
| 统一文件协议、外部目录接力 | `artifact-protocol.md`、13项导入事务测试、10项inspection测试 | 文件哈希、依赖、顺序/单位/dt、失效传播与事务回滚已验证；NPZ 先核对数组头与字节预算再分配 |
| 长任务资源与冲突控制 | WorkerSlots、内核workspace mutex，worker/history实际子进程测试 | 同一Host跨工作区有界并发；相同工作区跨进程排斥；不同Host不共享一台机器的全局计算配额 |
| 取消、失败重试、任务恢复、训练续训 | 11项训练恢复、5项硬退出恢复、9项native history测试；`web-cancel-resume-evidence.json`、`web-hard-exit-recovery-evidence.json` | Web已实际验证取消后838→840轮，以及硬退出后归档再2→25轮；均为新作业与真实权重恢复 |
| 前端目录/上传/参数/日志 | 实际Web上传与目录、三角色表单；`web-python-parameter-evidence.json`、`web-python-dialogue-report.png` | 真实对话设置网络6→12→12→3、学习率0.008、100轮；保存配置与报告已核验 |
| 前端曲线、报告、文件与阶段接力 | `web-policy-comparison.png`、`web-artifact-file-preview.png`、`web-controller-handoff.png` | 实际保存的报告值；长报告抽样明确标注；文件预览受清单与预算限制 |
| 交付与复现 | README、安装/启动脚本、3类examples、协议、复用哈希与测试日志；`harness-startup-evidence.json` | 最终项目配置实际启动并执行原生作业；用户新对象须满足已声明适配接口，未宣称任意工程自动迁移 |
| 表单与对话运行边界 | `harness-quiet-forms-evidence.json`、随后重新通过的三角色模型对话 | 三角色表单完成各自0次模型回合；显式对话仍能等待并读取作业，修复多余唤醒与提问 |

## 原项目能力的保留边界

已复用的算法源位置与哈希在 `reuse-manifest.json`。原项目自己的Planner、FastAPI/SSE服务、任务管理和聊天前端由Harness能力替代。不能把保留原目录等同于所有功能已迁移。

NMPC-light的动态Plant/Controller插件生成、安装与版本回滚，架构批量比较、工况推荐与补采诊断；mpc_agent_v5的原生MATLAB/Simulink执行与代码编辑修复；AIagent的专用热管理PGAC完整训练环，都需要进一步逐项核对与适配。当前没有把它们包装成已验证的通用能力。原仓库仍可独立使用，集成层不修改这些源码。

完整数值演示使用明确的数学参考仿真器；没有以随机权重、Mock模型服务或文件存在冒充算法成功。神经策略在现有案例中更快，但没有达到专家MPC的全部跟踪性能，报告保留该限制。

最新整套 Python 测试 **116项通过（111.056秒）**，日志 `test-output-release-regression.log`；Node worker/history/roles **16项通过（39.325秒）**，日志 `test-output-release-node.log`。7项 Python 对象测试包括真实八阶段闭环、源码冻结与导入、独立ODE对照、梯度/随机性/运行依赖拒绝。残差报告只统计训练分区的实际范围；`harness-coverage-evidence.json`另记录原生导入、训练和报告检查。

完整 Python 案例的数据与工况均为显式数学仿真。残差测试总 MSE 约下降14.01%，个别状态略变差；GA候选通过独立复验；神经策略推理更快但测试跟踪仍弱于专家，短程物理梯度未提升测试跟踪。报告中的 verified 表示通过本次配置的工程验收门，不代表实物认证、全域约束证明或部署性能达标。

最终 quiet 配置下再次完成真实八阶段业务对话，恰好一次流程运行，8个产物均通过依赖和文件校验。`harness-python-workflow-validation.json` 记录准确的 Session/Job/Run/产物ID，统一接口指纹、真实 Python 执行来源及重新加载模型/策略后的有限预测与动作；`verify-business-artifacts.py` 可复核这份已完成的数学案例。
