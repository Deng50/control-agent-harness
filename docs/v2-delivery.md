# V2 安装、三个入口与独立产物迁移

本轮按已批准 V2 方案在 `control-agent-harness` 增量实现。一套固定原生 Harness、三个专业入口及共享领域插件共同工作；技能引导路线，适配器和算法工具执行计算。原生会话、模型配置、规划、压缩、子智能体、权限、作业、文件服务和工作流继续由 Harness 提供。

本轮已完成真实数值、GUI、全量 Python 629 项、Node 120 项、三个专业角色的 13 个业务回合及 1 个解释更正回合，以及新目录独立加载验收。源码归档及所有必需门的最终发布状态以 [acceptance.md](acceptance.md) 和机器可读验收清单为准；本文区分较早真实训练/评价与 S14 新执行。

## 环境与安装

固定上游为 Harness `0.2.2-alpha.2`，提交 `f3803c96974ad4a94195fb97e5c637e9c2a3afe3`；业务包版本 `0.1.0`，增量内容另由交付源码清单和 SHA256 标识，未另发布版本标签。上游目录与独立上游副本均保持该提交，不修改 agent-loop。

| 组件 | 本机实际验收版本 | 使用范围 |
| --- | --- | --- |
| Windows / Node.js / pnpm | Windows；Node 26.13.0；pnpm 11.28.5 | Harness 和本地网页；项目要求 Node ≥24.9、pnpm 11 |
| Python | 3.10.22 | CPU 数值任务与独立推理 |
| PyTorch / NumPy / SciPy | 2.7.1+cu128 / 2.2.6 / 1.15.3 | 实际验收使用 CPU，不因安装 CUDA 包自动启用 GPU |
| openpyxl | 3.1.5 | XLSX 数据读取 |
| MATLAB / Simulink | R2024b、对应已许可产品 | 可选的 MATLAB 函数及受支持 SLX 路线 |
| MATLAB MCP / Simulink Agentic Toolkit | 0.16.0 / 2026.11.a | Harness 独立注册的工具连接与 Toolkit skills |
| MATLAB Engine API | 未安装、未采用 | 当前经过验证的 MCP 与批处理路线不依赖它 |

在目标机器重新创建 Python 环境和依赖，不复制本机 `.venv`：

```powershell
Set-Location <项目源码目录>
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c constraints-v2-validated.txt -e .
.\scripts\setup-harness.ps1
.\scripts\prepare-workspace.ps1 -Prepare -Workspace "$PWD\workspaces\project-01"
.\scripts\start-web.ps1 -NoOpen
```

`setup-harness.ps1` 准备固定上游及插件构建。模型服务和凭据在 Harness 原生配置中设置；表单领域计算不要求额外 LLM 服务，对话需要可用的模型服务。第一次使用启动终端提供的带认证链接，之后在同一 `127.0.0.1` 来源继续使用浏览器会话。

`constraints-v2-validated.txt` 对齐本轮五个关键 Python 包；其中 Torch 约束允许已验证的 `2.7.1+cu128` 本地构建，不会自动启用 GPU。这不是完整操作系统或所有二进制依赖的锁文件，换机后仍须执行独立数值加载检查。

需要 MATLAB 时，先按 [MATLAB 连接说明](matlab-existing-session.md) 显式建立工程专用会话，再通过 `start-web.ps1 -Matlab` 指定 MCP、Toolkit、私有会话 APPDATA 和共享资源目录。三个入口共享同一 MATLAB 资源锁，Python-only 部署可以不启用该配置。超时不代表 MATLAB 已停止；正常恢复方式和现有进程保护以该说明为准。

上述安装命令是部署步骤。本轮额外实际验证的是**复用已审计本机运行时的新目录迁移**，没有声称已在另一台机器或全新虚拟环境安装验证。

## 三个入口如何选择

侧栏选择“控制工程”，选择新项目目录，再选择相应入口。三个入口有独立的原生专业 Session；普通模式与专家模式修改同一份生效配置。

| 入口 | 用户可要求的局部结果 | 主要路线及边界 |
| --- | --- | --- |
| 模型与数据 | 仅分析、仅采数、NN、残差模型、条件 PINN | 有受支持对象时先确认接口和工况再采数；只有数据时确认列映射后直接训练，不强求对象程序 |
| 控制器设计 | 仅设计草稿、数值 MPC、偏好调参 | 缺参数的文字任务保留未实例化草稿；参数完整时可作基础设计及单点求解，调参需另有可执行评价对象与工况 |
| 控制器轻量化 | 专家采数、已有数据仅模仿、可选受限 PGAC | 不强制重做 MPC；已有专家数据可以直接模仿并结束；PGAC 另需合格可微模型及原对象评价 |

每个请求先明确停止点。训练摘要可配置网络宽度、隐藏层、激活、学习率、batch、epochs、早停和随机种子；输入、输出宽度须与所确认的信号接口一致，不能把特征宽度当作物理状态数。

先“生成摘要”，核对信号语义、单位、工况及训练预算，再“确认当前摘要”并执行。修改关键内容后重新审阅新版本。实际用户确认关联当前 Session 和版本；模型不能替用户确认。开发验收中的确认明确记录为 `test_fixture`，不冒充真实用户授权。

只读材料检查不执行上传程序。路径与点击上传都登记不可变快照；文件中的指示文字不构成用户的新授权。网络输入、输出、列位置、时间及行程等语义仍需明确核对。

## 已验证的支持范围

| 事项 | 当前能力 | 使用条件 |
| --- | --- | --- |
| Python 对象 | 独立、明确注册的对象合同与可微 PyTorch 适配；采数与 NN/残差训练 | 显式状态、输入、输出、扰动、dt、边界、复位及执行入口；不是任意工程自动迁移 |
| MATLAB 函数 | 已声明合同的实际执行、采数及参考评价 | 专用 R2024b 会话；未知全局状态和不明依赖须先适配 |
| Simulink | 实际执行平坦已知模块、完整 UnitDelay 状态可读的受支持 SLX 子集 | 快照、依赖、状态/输出、采样合同须通过检查；任意隐藏状态、未知模块或任意项目不在此范围 |
| CSV / XLSX | 独立表格数据接口、列位置/工作表选择、因果窗口、按行程或时间块划分、训练集 scaler | 不猜测单位、电流符号或不可观测状态；样例为合成数据，未冒充实车测量 |
| NN | 真实优化、验证集选权重、一次最终测试、检查点及独立推理 | 无物理对象时不展示伪物理基线；数据泄漏及错误维度在入口检查 |
| 条件 ODE PINN | 热过程 `C*dT/dt=-G*(T-Ta)+P` 的真实自动微分残差、配点和初值损失 | 输入时间/初温/整轨迹恒定功率；当前选定网络有明确仿射条件先验；不提供任意 Plant step 或通用 PDE 求解 |
| MPC 设计与调参 | 基础模板、已登记预测模型、状态估计、约束、偏好权重及有界 GA | 偏好可侧重跟踪、动作平方代理、平滑度等；没有跳变时不伪造超调/调节时间，有限仿真不构成稳定性证明 |
| 外部专家模仿 | Python/MATLAB/SLX 专家路线及 CSV/XLSX 专家记录；真实 NN 与动作投影 | 离线 teacher-forced 误差与自主闭环分别报告；只有记录时不编造闭环结果 |
| 受限 PGAC | 从真实模仿权重开始，经资格验证的有界可微动力学 rollout 梯度；原对象验证固定选择后做最终测试 | 如实记录代理模型/覆盖范围、可行性和候选拒绝；没有独立 Critic、PPO/SAC、在线 OOD 自动回退或硬实时证明 |

详细合同见 [对象采样](plant-sampling.md)、[数据-only NN](data-only-training.md)、[预测模型](predictive-models.md)、[仅设计](design-only.md)、[外部专家](external-experts.md)、[模仿](external-imitation.md)、[PINN](conditional-pinn.md) 和 [PGAC](restricted-pgac.md)。

## 可重复的输入示例

`examples/v2/` 提供相对路径输入及明确的合成来源：

- `f03-synthetic.csv` 与 `model-nn.json`：三输入二输出数据-only NN，不需要物理对象。
- `mpc-draft.json`：先生成缺参基础草稿；`mpc-design.json` 再显式给出质量、阻尼、刚度、采样周期和边界，进行仅设计及单点求解。
- `external-qp-records.csv` 与 `imitation.json`：来自实际 QP 专家的九特征、两动作记录，仅模仿即可结束。

在相应入口可对话：“读取上述配置和数据，先解释输入、输出、单位与训练参数，展示实际摘要，等我确认后只完成当前任务。”这些文件已经随源码提供；运行示例不需要旧 `test-output` 目录。`provenance.json` 记录准备来源，不是运行时的数据位置。

旧 `examples/linear`、`examples/nonlinear` 的完整八阶段流程继续保留，是可选兼容案例。温控旧案例和手动接力见 [原使用手册](user-guide-zh.md)，不是每个新任务的必经顺序。

## 独立下载与新目录加载

网页可下载经过文件清单校验的完整 bundle，也可单独获取报告及原始数组。报告标明来源、单位、训练划分、实际配置、验证级别以及未执行项；长数组的预览抽样会标明，完整文件可下载。

本机迁移验收目录为 `D:\projects\control-agent-v2-delivery-20261118`，主要内容如下：

| 路径 | 内容 |
| --- | --- |
| `downloads/` | 实际浏览器下载的 NN、MPC、PGAC 三个完整 ZIP |
| `packages/model/bundle/` | data-only NN 权重、scaler、信号顺序及 `predict.py` |
| `packages/controller/bundle/` | 已调参 MPC 参数、预测器及 Python/MATLAB 实现 |
| `packages/lightweight/bundle/` | 已验收 PGAC 策略权重、因果特征、动作约束及 `infer.py` |
| `reports/{model,controller,lightweight}/report.json` | 三类未经改写的原始报告 |
| `packages/pinn/bundle/`、`reports/pinn/report.json` | S11 实际选定的条件热过程 PINN 独立包和原始报告 |
| `additional-models-manifest.json` | PINN 补充交付清单，独立于原三角色固定清单 |
| `tools/verify-v2-migration.py` | 三种独立 API 的新进程加载、文件哈希及固定推理检查 |
| `delivery-manifest.json` | 相对路径、原始 SHA256、固定输入/期望值与支持范围 |
| `conversation-results/`、`dialogue-products-manifest.json` | S14 三条真实角色对话新生成的 NN、基础 MPC、仅模仿产物，另列精确版本 |
| `report-data-manifest.json` | 四组原始报告的 42 份文件、8 个 NPZ 数组档案及来源 SHA |

复制上述文件到一个新目录后，使用已安装 NumPy、SciPy、PyTorch 的兼容 Python 运行：

```powershell
Set-Location <新的产物目录>
python -I .\tools\verify-v2-migration.py `
  --manifest .\delivery-manifest.json `
  --output .\migration-check-new.json
```

输出文件须使用新名称，检查程序保留已有结果，不覆盖或默默重复执行。该程序实际加载三个独立 API，并以负控确认原项目文件读取及 `control_harness` 导入被阻断。它不训练、不调参、不调用 MATLAB、不重新运行原对象闭环。包中的原始报告路径字符串用于追溯，推理不读取那些原路径。

独立包也能分别运行 `predict.py`、`controller.py`、`infer.py`；输入格式见各包 README，固定示例保存在 `migration-*-input.json` 的 `case.probe.input` 字段。无需复制 `.dsh`、凭据、旧 Session、签名确认、`.control-jobs` 或本机 `.venv`。独立推理包与完整原生 artifact registry 是两种交付形式，此目录的三个 bundle 没有冒称完整 registry 导入。

本轮已在该仓库外新目录创建三个原生专业会话并执行只读初始化；三个便携包的独立推理误差分别约 `1.67e-16`、`0`、`2.78e-17`。这是同权重加载一致性验证，不是新训练精度、闭环改善或实时性结论。

补充 PINN 包也已在独立 `python -I` 进程中阻断原项目文件读取和导入，复现前三条已保存查询，温度结果误差为 0。示例可运行 `python -I .\packages\pinn\bundle\predict.py --input .\pinn-example-input.json --output .\temperature-new.json`；功率必须表示整条轨迹恒定驱动，不可当作通用状态转移控制器。

## 评价与已有结果的解释

NN 报告没有对象模型时标记物理基线不适用。MPC 报告区分仅设计、单点求解、预测模型闭环和原对象闭环；所附 MPC 包未冒称已在原 SLX 上闭环复验。模仿策略在部分工况下跟踪会弱于 MPC 专家，报告保留这种退化。PGAC 的改善仅针对固定工况与明确的评分，验证集固定版本后才评价测试集，不代表所有对象、所有工况或统计显著性。

计时报告保留 CPU、线程数、batch、预热/重复次数及控制器/对象范围；平均或分位耗时低于采样周期不等于 WCET、硬实时或部署认证。真实实车、硬件在环、任意工程迁移以及部署平台验证，应作为后续明确的独立项目。

## 开发验收入口

完整测试与实际证据保留在 `test-output/v2-development/20261117T002006/`。S14 的 `scripts/smoke-v2-user-paths.mjs` 用真实已配置模型，保留原生工具、作业、提案和明确测试确认；`scripts/smoke-v2-legacy-compatibility.mjs` 验证旧线性/非线性原配方与已完成阶段复用。自动测试确认仅用于预先隔离的 `test_fixture` 工作区，不能用于生产用户。

需要复现实验时使用新工作区，先查当前 `control_help` 和摘要要求。不要把旧式不带当前工程确认的 smoke 命令或历史证据文件，直接当成本轮新验收结果。
