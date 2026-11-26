# 控制技能库使用与验收

在工作台选定对象模型、控制器或轻量化角色后，可以直接用自然语言描述对象、手头的材料及要交付的结果。三个角色都能检索同一份随包资产。换电脑只需恢复这个仓库及相应的运行环境。已有项目的表单仍按所选配置提交数值作业。

| 在专业会话中提出的请求 | 可以采用的技能与证据 | 在当前对象上应核对什么 |
| --- | --- | --- |
| “我只有温度和功率的时序 CSV，按整条工况训练并留出测试。” | `audit-train-tabular-model`，表格训练源码、列映射和防泄漏检查 | 轨迹划分、特征与标签的时序、仅在训练集拟合的缩放。给定可信热平衡方程时，还可比较条件 PINN 和普通 NN |
| “用 Simulink 搭二阶闭环 MPC，与 ADRC 比较，检查增量预测矩阵。” | `build-native-mpc` 和 `verify-mpc-prediction`，二阶模型、图内程序和独立 MATLAB 递推校验 | 端口、采样周期、上一拍实际动作、控制时域、闭环跟踪和动作总变化 |
| “把耗时的 MPC 蒸馏成小网络，用原对象评价跟踪和约束。” | `clone-control-policy`，专家轨迹、模仿策略及原对象闭环对照。需要继续优化时可查 `warmstart-pgac` | 专家与学生的输入及动作时序、同一对象的权重交接、留出工况的可行性和推理耗时 |

## 一次专业会话如何使用资产

1. 角色用 `control_library_search` 检索对象、方法、运行语言和交付物。每个命中给出技能名、适用角色、命中词、技能说明、知识条目、案例、源码和历史证据的位置。跨角色任务可以省略角色筛选。检索返回空列表时，角色还可按具体方法重查，或继续检查用户材料。
2. 原生 `skill` 工具加载专项技能。`control_library_read` 按库内相对路径读取案例、源码和报告，核验快照哈希，并支持分页和跟随文档中的相对链接。真实的 SLX 可以从对应案例目录打开。三个同名角色入口优先加载 Harness 版本，保留原有的确认、工作区和阶段交付约定。
3. 角色用 `control_help` 查看当前会话支持的操作和参数，确定哪些案例步骤适用于这次输入。运行阶段仍使用原生作业、Python worker 或已有 MATLAB/Simulink 工具。库中的旧报告可作为预期值，用本次数据、模型和工况生成的结果才进入当次验收。参考实现产生的权重需适配并验证接口后才能作为 Harness 工件接力。

表中的三类请求已在隔离 Web profile 的三个真实专业会话中验证。通过 Web RPC 创建会话后，验收程序依次调用领域检索、原生技能加载和源码读取，并核对结果哈希。三个首选技能分别为 `audit-train-tabular-model`、`build-native-mpc` 和 `clone-control-policy`。此次验收固定了工具调用顺序，未调用语言模型。[可复跑的原生会话验收](../scripts/smoke-native-capabilities.mjs)

本机 Web profile 安装了插件市场和 `dsh-knowledge 0.5.4`。市场供角色查询缺失的软件能力。控制插件将九篇知识文档同步到专属的“控制工程技能库”，支持 `knowledge_list_bases` 和 `knowledge_search`。该库使用本地词法索引。原生知识检索遵循用户的启用名单和服务配置，之后手动启用的全局重排也适用于该服务。专用的 `control_library_search` 和 `control_library_read` 始终在本地查找代码和工程证据。

在本机重新打开这一知识库后，核对了 9 篇文档和 35 个分块。用 `Toeplitz` 查询命中 `mpc.md`。同一配置下直接查“增量预测矩阵”返回空集，可以用领域检索定位 `verify-mpc-prediction`，再按返回路径读 `knowledge/mpc.md`。

## 从干净目录复跑

在仓库根目录用 PowerShell 执行。每次运行都会在指定目录下创建独立子目录，校验并复制所需源码和输入，从空的 `runs` 开始。运行目录中有命令、资产指纹和退出状态的 `replay-receipt.json`，以及新生成的模型、轨迹或报告。

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py scripts/replay-library-case.py pinn --workspace test-output/library-replays
& $py scripts/replay-library-case.py mpc  --workspace test-output/library-replays
& $py scripts/replay-library-case.py pgac --workspace test-output/library-replays
& $py scripts/replay-library-case.py matlab --scene mpc1 --workspace test-output/library-replays
```

`mpc1` 使用 MATLAB、Simulink 和 Optimization Toolbox，可在独立的 `matlab -batch` 进程运行。其他 MATLAB 场景按清单使用原始标定数据，可用 `--materials <素材根目录>` 指定，入口会核对数据版本。一个电池热辨识场景保留说明、指标和来源指纹，其三份第三方程序及模型需从原库取得。完整工程所需的外部热管理对象、存档、MAT 文件和现场数据分别见案例说明。[MATLAB 场景目录](../harness-plugin/control-skill-library/cases/matlab-scenes/README.md) · [标定输入清单](../harness-plugin/control-skill-library/cases/matlab-scenes/inputs.json) · [来源与授权说明](../THIRD_PARTY_NOTICES.md)

## 本次复跑结果

在 Windows、Python 3.10.22、NumPy 2.2.6、SciPy 1.15.3、PyTorch 2.7.1 和 MATLAB R2024b 环境中，从四个独立工作目录取得以下结果。

| 任务 | 独立验收与读数 |
| --- | --- |
| 恒功率热过程建模 | 14 条轨迹按 9/2/3 分入训练、验证和测试。普通 NN 与 PINN 的测试 RMSE 分别为 0.13750 K 和 0.15528 K，最大初值误差分别为 0.09219 K 和 0.02087 K。列序互换、重复工况泄漏、导出重载检查通过 |
| 有约束弹簧 MPC | 测试位移 RMSE 从 0.10138 m 降至 0.08171 m。控制力总变化从 5.98 N 增至 14.54 N，两组的力、位移和速度越界量均为 0。历史错误矩阵首次偏差定位到第 3 步、第 1 通道，误差 −0.00072 |
| 专家模仿及 PGAC | 学生的留出动作 RMSE 为 0.16766 N。独立闭环中，验证集选出的感知耦合策略在两条测试轨迹上的位移 RMSE 为 0.06998 m 和 0.04882 m，模仿基线为 0.06986 m 和 0.04865 m。此例保留了轻量化后跟踪略差的结果 |
| 原生一阶 Simulink MPC | 独立进程实际运行 SLX 和 QP 函数。20 s 闭环生成 2,001 个采样点，终点参考为 120，对象输出为 119.9999999999988。独立 QP 程序也返回有限值 |

全量回归在 Windows 默认临时目录下完成，Python 686/686 项通过，Node 插件 255/255 项通过。覆盖了原有数值流程、独立产物加载、工作区权限、任务恢复，以及新增的技能检索、文档读取和知识库同步。可在仓库根目录分别执行 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` 和 `node --test --test-concurrency=2 "harness-plugin/tests/*.test.mjs"`。

**与通用 Harness 的比较。** 这些检查覆盖任务路由、资产完整性与数值复跑。源库还有两组历史首交对照。三项跨角色子任务的无库组与显式调用组均为 9/9，完整阀门 MPC 均为 2/2。复算原提交可以检验评分器与交付物，无法从这些平局推断本次集成提高了真实对话成功率。[历史对照协议与结果](../tests/fixtures/control-library-evaluation/evaluation/README.md)

要衡量相对收益，可用两个独立 profile 运行同一批未见任务。基线使用固定上游的 `standard` 预设、干净技能目录和默认工具，实验组加载控制插件及相应角色。两组固定模型版本、预算、材料、许可工具箱及评分器，每次开新会话和独立工作区。分别统计首次交付通过率、返工次数、超预算或超时、产物重载成功率，并记录实际调用的技能、源码及库版本。评分时检查对象多步留出误差、控制约束与动作代价、轻量化后的原对象闭环和延迟。评审题、隐藏参数与首交答案放在维护端，评测参与者只接收题面和公开材料。

## 项目职责与代码审查

| 层 | 负责的工作 | 本次接入核对 |
| --- | --- | --- |
| 固定上游与安装启动 | 管理会话、角色预设、工具注册、权限、作业生命周期和插件市场 | 从固定上游生成三份预设，同时加载业务和专项技能根，保留用户级技能。通过有效 Web 配置核对实际加载的 bundle |
| 控制插件与工作台 | 接收用户任务，构建工程摘要、确认、提交作业，协调角色和 MATLAB 会话 | 将搜索和读取注册为原生工具，在角色入口写入选取专项技能、阅读证据、核对操作的步骤 |
| Python 数值层与 MATLAB 适配器 | 数据准备、建模、仿真、设计、训练、评价、导出与接力 | 查询接口只需要 Python 标准库。业务执行使用对应数值环境，MATLAB 的求解器、端口和工况可结合案例和实测输出核对 |
| 技能与研究资产 | 提供可改用的算法、代码、案例、来源和历史证据 | 补齐原索引未单列的 MATLAB 场景，根据资源类型提供归档结果、二进制模型链接和可读源码。维护端评分材料单独存放 |

审查覆盖两库自维护源码、测试、示例、配置、安装脚本、技能和研究资料。除对运行链的源码审阅与实测外，还对两库文本作了清点、Python 语法和 JSON 解析，对 SLX 容器中的 XML 作了结构检查。构建缓存、第三方依赖与工作区私有会话没有纳入逐行代码审查。复跑结果只覆盖上表四个案例，其余场景可查原库逐文件执行记录。

## 更新与核验资产

快照保存源库工作树的内容和逐文件 SHA-256，也纳入源库尚未提交的 MATLAB 技能和案例。以下命令分别校验当前快照、对照相邻源库，以及同步指定工作树。安装或离线检索期间无需相邻源库。

```powershell
& .\.venv\Scripts\python.exe scripts/sync-control-library.py --check
& .\.venv\Scripts\python.exe scripts/sync-control-library.py --source ..\control-skill-library --check
& .\.venv\Scripts\python.exe scripts/sync-control-library.py --source ..\control-skill-library
node scripts/sync-control-presets.mjs --check
```

插件快照、来源指纹和维护端评分清单可独立核对。[领域快照](../harness-plugin/control-skill-library/snapshot.json) · [历史来源清单](../harness-plugin/control-skill-library/sources.json) · [维护端复评方法与结果](control-library-evaluation.md)
