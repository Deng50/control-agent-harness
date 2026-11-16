# 二阶质量—弹簧—阻尼系统：三个智能体接力案例

本案例从连续时间方程开始，依次完成模型校正、MPC 设计与遗传调参、专家数据采集、控制策略模仿学习和物理模型驱动的策略优化。参考对象为显式数学仿真器，数据不是实测数据。配置可重复运行，训练效果和控制性能以本次实际报告为准。

## 1. 被控对象与接口

连续模型为：

\[
m\ddot p+c\dot p+kp=u+d,\qquad
x=\begin{bmatrix}p\\v\end{bmatrix},\quad v=\dot p,\quad y=p.
\]

相应状态方程为：

\[
\dot x=\begin{bmatrix}0&1\\-k/m&-c/m\end{bmatrix}x+
\begin{bmatrix}0\\1/m\end{bmatrix}u+
\begin{bmatrix}0\\1/m\end{bmatrix}d,\qquad y=[1\;0]x.
\]

| 参数 | 名义模型 | 参考对象 | 单位 |
| --- | ---: | ---: | --- |
| 质量 m | 1.0 | 1.1 | kg |
| 阻尼 c | 0.8 | 1.0 | N·s/m |
| 刚度 k | 2.0 | 2.3 | N/m |

两个 JSON 中的 A/B/E 是上述连续导数系数。`continuous_polynomial` 配合 `integration.method=zoh`、`substeps=1`，将它们精确转换为零阶保持下的离散转移；采样周期为 **0.2 s**。控制量和扰动在一个采样周期内保持不变。不要把连续 A 直接当成离散状态转移矩阵。

| 类别 | 顺序与单位 | 下界 | 上界 |
| --- | --- | ---: | ---: |
| 状态 1 | displacement，m | -1 | 1 |
| 状态 2 | velocity，m/s | -1.5 | 1.5 |
| 控制 | force，N | -4 | 4 |
| 输出 | displacement，m | -1 | 1 |
| 扰动 | disturbance_force，N | -0.2 | 0.2 |

导数单位按照当前接口规范声明为 `m/s` 和 `m/s/s`。MPC 默认获得完整状态 `[p,v]` 和已知扰动 d；本案例没有执行状态估计或真实传感器辨识。目标位移和扰动在每个闭环工况中均为常数。

## 2. 配置与三个角色的接力

| 阶段 | 配置文件 | 输出及用途 |
| --- | --- | --- |
| 登记名义模型 | `nominal.json` | `nominal_model`，保留连续源方程与离散适配声明 |
| 登记参考对象 | `reference.json` | `reference_model`，提供仿真数据与独立闭环评价 |
| 仿真采集 | `data.json` | 30 条轨迹×最多 35 步，产生 `transition_dataset` |
| 模型校正 | `residual.json` | 产生 `corrected_model`，交给控制器角色 |
| MPC 设计调参 | `controller.json` | 产生 `controller`，交给轻量化角色 |
| 专家采集 | `expert.json` | 12 个工况×36 步，产生 `expert_dataset` |
| 模仿学习 | `imitation.json` | 产生 `imitation_policy` |
| 物理策略优化 | `optimize.json` | 产生 `optimized_policy`，报告专家/模仿/优化策略对比 |

模型角色负责前四个阶段；控制器角色接收校正模型和参考对象，设计并验证 MPC；轻量化角色接收该控制器、参考对象及同一校正模型，完成后三个阶段。所有阶段通过同一工作目录中的产物 ID 接力，文件路径用于登记源模型和工作流。切换角色后须明确填入实际 ID，不要把文件名或旧实验的 ID 当成本次产物。

模型校正网络输入为 `[p,v,u,d]`（4 维），输出为下一状态残差 `[Δp,Δv]`（2 维）；校正模型等于名义模型离散预测加该残差。模仿策略输入为 `[p,v,目标位移,d]`（4 维），输出为控制力（1 维）。这两个网络虽然输入维数相同，输入含义不同。

## 3. 在前端运行

首先按项目根目录的中文使用手册安装环境、构建 Harness。当前电脑可在 PowerShell 中执行：

```powershell
Set-Location D:\projects\control-agent-harness
.\scripts\prepare-workspace.ps1 -Workspace D:\projects\control-agent-harness\workspaces\spring-damper-my-run
.\scripts\start-web.ps1
```

使用启动终端输出的本地链接进入前端。对话执行需先在 Harness 的“设置 → 模型”中配置模型服务；确定性工作流本身不消耗对话模型 API。启动链接中的认证信息及 API Key 不应写入案例配置。

1. 打开“控制工程”，工作目录填写 `D:/projects/control-agent-harness/workspaces/spring-damper-my-run`。
2. 使用“用户项目串联工作流”，工作流路径填写 `D:/projects/control-agent-harness/examples/spring-damper/workflow.json`。
3. 先执行预检，确认八个阶段均就绪；再运行完整工作流。预检验证文件、接口与依赖，数值可行性和性能在实际运行中验证。
4. 在作业面板查看进度、训练曲线和日志。结束后查看工作流记录及各产物的报告。八阶段编号依次为 `nominal`、`reference`、`data`、`corrected`、`controller`、`expert`、`imitation`、`optimized`。
5. 若要逐角色演示，先运行 `model-phase-workflow.json`（前四阶段），记录名义模型、参考对象、数据和校正模型 ID；切换“控制器设计”角色，选择“设计与调参”，输入校正模型 ID 与参考对象 ID，配置区粘贴 `controller.json` 内容。
6. 控制器完成后切换“控制器轻量化”角色：先“采集专家数据”，输入控制器 ID 和参考对象 ID，使用 `expert.json`；再“训练模仿策略”，输入专家数据 ID，使用 `imitation.json`；最后“物理策略优化”，输入模仿策略 ID 和原控制器使用的同一校正模型 ID，使用 `optimize.json`。

完整工作流内嵌了各阶段配置，便于作为固定配方执行。单独的六份配置供表单逐步演示使用；修改参数时，须同步修改工作流内对应阶段的 `config`，两者不会自动读取彼此。

## 4. 调参与评价规则

MPC 采用 horizon=6，即 1.2 s 预测窗口，初始 q=1、r=0.18、s=0.06。该保守初值便于观察跟踪与控制代价的权衡。GA 预算为 12 个候选，只搜索 r/s；质量、阻尼、刚度及预测窗口保持固定。目标误差、力与力变化项分别按接口范围归一化。本案例设置状态、输出及力幅值约束；没有额外的力变化率硬约束，s 对控制变化作软惩罚。

| 用途 | 初始位移/速度 | 目标位移/m | 扰动/N | 步数 |
| --- | --- | ---: | ---: | ---: |
| 调参 | [-0.25, 0.05] | 0.35 | 0.08 | 40 |
| 独立复验 | [0.28, -0.10] | -0.30 | -0.10 | 40 |
| 最终测试 | [-0.18, 0.12] | 0.37 | -0.06 | 40 |

三组工况相互独立，最终测试不参与调参。候选需通过独立复验并优于初始方案才被采用，否则保留初始方案。12 个专家工况与上述复验、测试工况不同；专家数据按照完整工况拆分为训练、验证和测试集，避免把同一条轨迹随机拆到不同集合。专家划分种子预先固定为 149，得到 8 个训练、2 个验证和 2 个测试工况；验证集和测试集各包含一个正目标和一个负目标，训练集覆盖四种目标。模型数据同样按轨迹拆分，归一化参数仅由训练集拟合。

仿真采集在状态出界后停止该条轨迹，因此实际样本数可能少于 1050，应使用产物记录中的实际数量。残差网络最多训练 300 轮，模仿策略最多 400 轮，并使用验证集早停及最佳检查点；物理策略优化执行 35 轮，按验证集选择检查点。训练轮数和执行预算不等于性能保证。

## 5. 结果保存与 PPT 素材

运行结果写入所选工作目录的 `artifacts/`，工作流进度写入 `workflows/`；通过前端产物文件清单查看实际文件和校验后的报告。建议保留模型/接口声明、仿真数据来源、各阶段配置、训练日志、产物 ID 与父子依赖，供复现和制作素材使用。

适合放入 PPT 的内容包括：连续方程与参数差异；名义/校正/参考对象的独立预测对比；位移与速度分别统计的测试误差；训练/验证损失；GA 候选得分和独立复验；初始/调参 MPC 的位移、速度与力曲线；专家/模仿/优化策略的同工况控制曲线；误差、约束和推理耗时对照。每张曲线应标明工况、单位、采样周期和数据来源。

本项目还提供三个真实角色会话的执行和素材导出脚本。先启动本地 Harness 并配置 DeepSeek 官方模型，在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[presentation]"
node scripts/run-spring-dialogue.mjs workspaces/spring-damper-my-run
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe scripts/summarize_spring.py workspaces/spring-damper-my-run
.\.venv\Scripts\python.exe scripts/plot_spring.py workspaces/spring-damper-my-run
.\.venv\Scripts\python.exe scripts/package_spring.py workspaces/spring-damper-my-run
```

三个会话使用 `control-model`、`control-controller`、`control-lightweight` 角色，分别预检、执行自己的阶段，并通过已验证产物 ID 接力。数值预算与上文配置一致，对话会消耗已配置模型的 API 调用；请为一次完整运行预留几十分钟。脚本可用同一工作目录接着观察已提交会话，不会自动重交同一阶段；失败后应先检查已有作业和记录。新的独立实验使用新目录。

`presentation/` 中保存三角色可见回复、阶段作业记录、全部曲线的 CSV、约4000像素宽 PNG、SVG、PDF，以及中文 PPT 大纲和说明。`spring-damper-PPT-materials.zip` 是素材压缩包；训练权重仍在原工作目录的 `artifacts/`，没有放入素材包。`plot_spring.py` 的多面板图审计默认使用本机 nature-figure 审计工具；可通过 `--skill-scripts` 指定该工具的 scripts 目录。PDF 字体与碰撞检查结果单独保存在 `presentation/qa/`。

界面运行完整八阶段可复现数值流程；上面的自动素材导出以三会话脚本生成的 `case-run.json` 为入口，不能直接处理只有界面工作流记录的目录。采集完成后也可先使用 `summarize_spring.py ... --model-only` 和 `plot_spring.py ... --model-only` 导出前三幅模型图，后续全阶段导出会保留完整结果。

## 6. 能力边界

- 这里的物理信息模型使用已知动力学骨架、离散下一状态残差及状态边界惩罚。它没有训练连续 ODE/PDE 方程残差，不能将其报告为完整方程残差 PINN。
- `optimize_policy` 实际支持 `epochs`、`horizon`、`learning_rate`、`batch_size`、`imitation_weight`、`constraint_weight`、`seed`；本案例已按这些参数配置。算法是可微物理 rollout 策略梯度，没有学习 Critic，不能称为 PPO 或 SAC。
- 控制约束由控制器传递至轻量策略，不应在优化阶段覆盖。当前模型和策略均为仿真演示，没有提供实物稳定性或安全性保证。
- 一步预测改善、长时滚动预测改善、闭环性能改善是不同结论，须分别根据实际结果判断。未改善、约束不通过或保留基线的结果也应如实保存。
