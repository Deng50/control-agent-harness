# 控制工程智能体平台

基于固定版本的 DeepSeek Harness，通过外部 Cordis bundle 接入模型残差训练、MPC 设计与遗传调参、专家数据采集和策略轻量化。原四个仓库不作修改。Python 执行领域计算，Harness 负责会话、工具、人机命令、作业、附件与沙箱。

当前版本已完成线性、非线性及独立 Python 对象的数值闭环、Web 三入口和真实 DeepSeek 业务对话验证。具体范围与证据见[验收记录](docs/acceptance.md)，原项目能力的迁移对应见[功能表](docs/function-map.md)。仿真集成验收不等于实物部署认证。

第一次使用请阅读[中文使用手册](docs/user-guide-zh.md)，包含当前电脑快速启动、从 GitHub 首次安装、温控案例的完整流程、三个智能体的手动接力、对话示例及常见问题。

## 环境与启动

要求 Node.js ≥ 24.9、pnpm 11、Python ≥ 3.10，以及 NumPy、SciPy、PyTorch。当前机器使用 Python 3.10.22 / Torch 2.7.1 / SciPy 1.15.3。集成目录的 `.venv` 继承现有 `mpc_light` 环境；复制到另一台机器时需重新建环境，不要复制此虚拟环境。

在当前机器打开 PowerShell：

```powershell
Set-Location D:\projects\control-agent-harness
# 当前目录已经完成 Harness 安装与构建。首次建立或重新建立时执行：
.\scripts\setup-harness.ps1
# 建立由当前用户拥有的项目目录，并满足 Harness Windows 沙箱 ACL 前置条件：
.\scripts\prepare-workspace.ps1 -Workspace D:\projects\control-agent-harness\workspaces\project-01
.\scripts\start-web.ps1
```

启动终端会显示带本地认证令牌的 URL。默认绑定 `127.0.0.1:5288`，不会公开到局域网。停止时在启动终端按 Ctrl+C。`-NoOpen` 可跳过打开浏览器。模型凭据通过 Harness 自己的设置配置；确定性表单操作不调用模型 API，对话操作需要可用的模型服务。不要把启动令牌或 API Key 提交到 Git。

已有 Harness 构建仅更新外部插件时：

```powershell
.\scripts\setup-harness.ps1 -SkipHarnessBuild
```

新建 Python 环境可使用已有兼容解释器：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

从 GitHub 在另一台 Windows 机器恢复时，先安装上文要求的 Git、Node.js、pnpm 和 Python，再执行：

```powershell
git clone https://github.com/Deng50/control-agent-harness.git
Set-Location control-agent-harness
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\scripts\setup-harness.ps1
.\scripts\prepare-workspace.ps1 -Workspace "$PWD\workspaces\project-01"
.\scripts\start-web.ps1
```

安装脚本优先使用相邻的 `deepseek-harness` 源码仓库；没有本地副本时，从官方 GitHub 获取并检出固定提交。因此 GitHub 仓库不包含 `vendor/`，也不需要安装三个旧项目的服务。已提取的算法在 `src/control_harness/reused/` 中；只有重新提取算法时才需要原仓库。模型凭据需在新机器的 Harness 设置中重新配置。

## GitHub 上传范围

| 内容 | 是否上传 | 原因 |
| --- | --- | --- |
| `src/`、`harness-plugin/src/`、插件描述与依赖锁文件 | 上传 | 领域算法、原生工具、角色与前端源代码 |
| `config/`、`scripts/`、`tests/`、`harness-plugin/tests/` | 上传 | 安装启动、配置与可复现检查 |
| `examples/` | 上传 | 明确的数学模型、参数和工作流配方，可重新生成数据与权重 |
| README、`docs/*.md`、`docs/validation-summary.json`、来源清单与第三方声明 | 上传 | 使用说明、协议、脱敏验证摘要与源码溯源 |
| `.dsh/`、`.env*`、凭据文件 | 不上传 | 本机 API Key、认证、会话与个人设置 |
| `workspaces/`、`test-output/`、`.control-jobs/`、日志、锁与暂存目录 | 不上传 | 用户上传材料、运行记录、失败诊断与实验产物 |
| `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.npz`、`*.npy` 等 | 不上传 | 训练权重、检查点与生成数据；需另行做实验备份 |
| `.venv/`、`node_modules/`、`.pnpm-store/`、`vendor/`、`harness-plugin/lib/` | 不上传 | 可根据源码与锁文件重建的环境、依赖和构建输出 |
| `docs/` 中原始验收 JSON 与本机截图 | 默认不上传 | 可能包含本机路径、会话/作业标识；保留本机，仓库提供脱敏摘要 |

这些排除规则已写入 `.gitignore`，不会删除本机文件。GitHub 源码备份不包含实验数据与训练成果；若需要保存它们，应另做独立的工作区备份。上游许可及提取来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

Windows 的 Harness 沙箱需要工作目录由当前用户拥有，且该用户拥有完全控制，以便写入沙箱低完整性标签。辅助脚本只处理集成目录以内的当前用户自有目录；不会接管所有权、授权其他用户或关闭沙箱。外部工作路径需要具备相同前置条件。当前 Windows 后端仅限制写入，不提供完整读隔离或网络隔离，保留上游的能力边界。

## 使用方式

侧栏进入“控制工程”，选择项目目录，再选择“模型校正”“控制器设计”或“控制器轻量化”。每个项目和角色使用不同的 Harness Session，表单草稿在当前浏览器中保留。输入区填写产物 ID 或绝对文件路径；模型 JSON、独立 Python 文件和转移数据 NPZ 可以通过 Harness 附件服务上传。参数区的结构化选项与 JSON 配置同步。

点击“执行阶段”会立即返回原生作业，右侧显示进度、日志和验证损失曲线。作业可取消。产物可查看校验后的报告与文件清单，成功产物可选入下阶段；评估对象、独立测试工况等额外前提仍须填写。“持久任务记录与恢复”在服务重启后仍可读取结果，并提供按原请求重试、训练检查点续训或流程阶段接续；界面上另外的“按当前配置重试”仍是一次新运行。

“检查对象材料”支持 JSON、Python AST 和 Simulink 静态材料检查，列出已声明接口、缺失项与不确定项。有完整候选时可进入接口表单修改名称、单位、边界和采样周期，再显式导入。检查阶段不执行源码；独立、确定、可微的 PyTorch 对象可随后通过“运行并导入 Python 动力学”执行、保存源码快照并校验梯度，详见[Python 动力学适配](docs/python-dynamics.md)。SLX 静态查看不等于运行原始 Simulink。模型角色另有满秩线性最小二乘辨识，控制器角色提供真实估计状态驱动 MPC 的 KF/EKF 验证。

“用对话调整任务”将明确的工作路径和请求发送到所选角色的 Harness 对话，由 `control_model`、`control_controller`、`control_lightweight` 工具执行。表单通过同一领域提交函数及 Harness 原生命令执行，不另设聊天后端。三个角色使用 quiet 完成通知：表单作业结束不会自行开启模型对话；对话任务由模型主动等待原生作业并读取结果。

`examples/linear` 为 4 状态、2 控制、2 输出、1 扰动对象；`examples/nonlinear` 为 2 状态、1 控制、1 输出、1 扰动对象。两者都是明确声明的数学仿真案例。每个文件夹包含对象、各阶段配置和可编辑的 `workflow.json`。“用户项目串联工作流”可以预检、运行这些文件或用户自行编写的流程；已有产物和上游阶段引用均可接力。界面的旧“完整流程演示”仍须使用全新工作目录。

也可直接运行领域计算以诊断算法：

```powershell
.\scripts\run-worker.ps1 --demo nonlinear --workspace D:\projects\control-agent-harness\test-output\new-nonlinear
```

这条命令只测试计算链，不代表 Harness 或前端验收。一般操作通过 `--request request.json` 提交，格式如下：

```json
{
  "operation": "train_residual",
  "workspace": "D:/projects/control-agent-harness/workspaces/project-01",
  "inputs": {"nominal_model": "nominal_model-实际ID", "dataset": "transition_dataset-实际ID"},
  "config": {"epochs": 150, "hidden_dims": [32, 32], "learning_rate": 0.002, "seed": 42}
}
```

## 检查与文档

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --test harness-plugin/tests/worker.test.mjs
node --test harness-plugin/tests/job-history.test.mjs
# Web 已启动且日志保存到 harness-web.log 时，运行真实无模型 API 集成验收：
node scripts/smoke-harness.mjs
# 附加完整的原生 Harness 数值流程：
node scripts/smoke-harness.mjs nonlinear
# 通用用户流程入口及已完成阶段接续验证：
node scripts/smoke-harness.mjs workflow-linear
node scripts/smoke-harness.mjs workflow-nonlinear
# 已配置真实 DeepSeek 凭据时，验证完整八阶段业务对话：
node scripts/smoke-business-dialogue.mjs
# 对该次已完成的 Python 案例重新校验完整依赖并加载模型/策略（不重训）：
.\.venv\Scripts\python.exe scripts/verify-business-artifacts.py
# 验证表单完成不会自行开启模型对话（不需要 API Key）：
node scripts/smoke-quiet-forms.mjs
```

协议见 [artifact-protocol.md](docs/artifact-protocol.md)。算法来源及原文件哈希记录在 [reuse-manifest.json](reuse-manifest.json)。线性、非线性、Python 业务对话和温控案例的已执行结果见[脱敏验证摘要](docs/validation-summary.json)。神经策略目前没有达到专家 MPC 的全部跟踪性能，非线性案例的物理梯度微调未带来提升；摘要保留真实结果。开发期间的原始验收 JSON、截图和日志默认仅留本机，运行对应检查后可重新生成。

新能力说明：[用户工作流](docs/workflows.md)、[训练检查点](docs/training-checkpoints.md)、[状态估计](docs/state-estimation.md)、[线性辨识](docs/linear-identification.md)、[对象材料检查](docs/model-inspection.md)、[策略动作约束](docs/neural-policy-action-constraints.md)。

逐项进度、实际证据及剩余限制见[当前验收状态](docs/acceptance.md)。原生角色接力验证为 `node scripts/smoke-role-handoff.mjs`；真实 DeepSeek 对话验证为 `node scripts/smoke-model-dialogue.mjs`。完整 Python 对象八阶段业务对话见 `docs/harness-business-dialogue-evidence.json`；精确产物与重新加载验证见 `docs/harness-python-workflow-validation.json`；Web 对话设置 `6→12→12→3` 网络、学习率和训练轮数的实际记录见 `docs/web-python-parameter-evidence.json`。凭据在 Harness 设置中管理；`scripts/configure-deepseek.mjs` 也可通过隐藏终端输入写入同一个原生存储。

连续动力学已支持显式 Euler/RK4 和仿射精确 ZOH 转换，见[连续模型适配](docs/continuous-dynamics.md)。强制退出后的未发布检查点可经 OS 存活判定和完整性校验归档，见[检查点说明](docs/training-checkpoints.md)。三个角色使用原生预设、工具帮助及依赖接力，见[角色与报告](docs/role-relay.md)。

当前支持声明式离散/连续动力学及独立 PyTorch 可微对象。SLX/MDL 仍为静态检查；任意 Python 工程迁移、原始 Simulink 执行和通用 PDE PINN 不在已实现能力内。新对象仍需明确接口、模型来源、约束与独立评价工况；神经策略达到部署性能需要进一步工程验证。
