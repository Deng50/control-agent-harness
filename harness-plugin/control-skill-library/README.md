# 控制智能体技能库、知识库和案例库

> Harness 内置资料。下文的运行指标来自源库的历史记录。复跑前复制到独立工作区，命令会重写 runs/。基础 Python 案例使用 requirements.txt。完整工程训练及 Simulink 场景所需原始材料见 sources.json 与场景 inputs.json。历史盲测仅在维护端按 docs/control-library-evaluation.md 复评分。


按物理对象查找原生 MATLAB/Simulink 程序，可从 15 个场景中直接打开源码、带中文图注的模型及逐文件报告。在库根运行 `matlab -batch "addpath('cases/matlab-scenes'); run_scene('mpc2')"` 可复跑二阶 MPC 与 ADRC 对照及闭环权重整定。[MATLAB 场景目录](cases/matlab-scenes/README.md) · [逐文件验收](runs/matlab-scenes/acceptance.json)

从项目根打开 Codex 会话，用 `python -m controlkit.lookup "核验MPC预测矩阵"` 查看适用 Skill、代码与证据。库中有 3 个角色入口和 14 个专业 Skill。[任务索引](index.json) · [工程验收](runs/field-acceptance.json) · [基础回归](runs/acceptance.json) · 未见任务对照与首次提交（维护端文档 `docs/control-library-evaluation.md`）

| 真实交付请求 | 工程判断、代码及案例 | 从库根复跑 |
|---|---|---|
| 二阶对象分段保持，做状态转移训练 | [fit-state-flow](.agents/skills/fit-state-flow/SKILL.md) · [选型依据](knowledge/model-decisions.md) · [SLX 与训练工件](cases/object-plant/README.md) | `python replay.py object-case` |
| 把原生MPC接入Simulink控制器并闭环整定 | [build-native-mpc](.agents/skills/build-native-mpc/SKILL.md) · [建模与求解决策](knowledge/controller-decisions.md) · [原生控制器](cases/native-mpc/README.md) | `python replay.py native-mpc` |
| 用热管理专家做NMPC模仿和感知耦合训练 | [distill-thermal-expert](.agents/skills/distill-thermal-expert/SKILL.md) · [轻量化决策](knowledge/distillation-decisions.md) · [原热管理工况](cases/heat-policy/README.md) | `python replay.py heat-policy` |
| 从含积分器的 SLX 移植 Python，查找复位、输入保持及延时差异 | [port-dynamic-plant](.agents/skills/port-dynamic-plant/SKILL.md) · [原对象三组轨迹与对齐](cases/plant-transfer/README.md) | `python replay.py transfer` |

运行 `python replay.py field` 可重建以上工程产物。二阶对象 8 s 自由递推的输出 RMSE 从 NN 的 0.0006001 降至 PINN 的 0.00005282（原输出单位）。原生 MPC 在最终测试上将位置 RMSE 从 0.14844 降至 0.10263，执行器 RMS 从 0.30044 增至 0.36060。热管理从同项目备份导入 16 条原专家轨迹，经原对象逐拍复核后训练，验证集选出的交付策略为模仿 Actor。三个基础案例另用于单测物理损失、增量矩阵和感知滚动。

| 角色 | 可直接使用的请求 | 单独复跑 |
|---|---|---|
| [对象模型自动化训练](.agents/skills/train-object-model/SKILL.md) | 用给定热过程采数，用相同的留出工况比较 NN、PINN；如果只有 CSV 也交付基线。 | `python replay.py pinn` |
| [控制算法设计与参数调优](.agents/skills/design-and-tune-mpc/SKILL.md) | 给二阶对象设计有约束的 MPC，找出预测矩阵中第一个错误，并报告调参是否增加动作变化。 | `python replay.py mpc` |
| [控制算法轻量化](.agents/skills/distill-controller/SKILL.md) | 采集专家闭环轨迹、训练模仿策略，再用同一对象的权重暖启动 PGAC，比较原对象闭环。 | `python replay.py pgac` |

## 运行环境与命令

本机验证环境为 Windows、Python 3.10.22、NumPy 2.2.6、SciPy 1.15.3、PyTorch 2.7.1，训练用 CPU。Python 核心依赖在 [requirements.txt](requirements.txt)。本机可执行以下命令；其他环境用安装这些依赖的 Python 替换解释器。全量回放另需 MATLAB R2024b + Simulink、Node.js、本地素材和相邻 control-agent-harness。归档热 Critic 的原始内核还导入 matplotlib，本机版本 3.10.9。

```powershell
# 先切换到已复制的案例工作目录
$py = 'python'  # 使用已安装 requirements.txt 依赖的环境
& $py replay.py field
& $py replay.py verify
& $py replay.py benchmark
```

`field` 从原对象采数、训练、执行原生联合仿真，最后用另一批 Simulink 轨迹核验导出的神经模型。`benchmark` 对两项未见任务试验的首次提交重新评分。`verify` 从磁盘复核基础及工程产物。案例数值门限与各自冻结的输入配套。

`python replay.py all` 可重跑基础案例、独立反例、历史材料、MATLAB 和基础验收。单独运行 `archive` 可复跑原对象线性化、原热管理 Critic 调用检查、二阶辨识、已有专家数据训练和 Harness 签名核验。`matlab` 单独回放原始 SLX。各命令刷新对应的 runs 文件。

## 从请求到验收的首例

请求“给定未来控制增量，定位预测矩阵首次出错的步和通道”。路由到 [verify-mpc-prediction](.agents/skills/verify-mpc-prediction/SKILL.md)，输入 [离散模型与三步增量](cases/prediction/input.json)，由 [prediction.py](controlkit/prediction.py) 独立计算两列输出。正确矩阵最大误差 1.39e−17，阈值 1e−12。换成历史分块索引方式构造的反例，第 3 步通道 1 矩阵值 0.099627028，递推值 0.100347028，误差 −0.00072。另一组含直接通道和非零扰动的 MIMO 测试覆盖六个独立增量，在另一个注入错误中定位到第 3 步通道 2。[正确结果](runs/prediction/pass.json) · [反例](runs/prediction/legacy.json) · [独立核验](runs/prediction/independent.json)

## 本次实跑结果

| 任务 | 工况与读数 | 交付产物 |
|---|---|---|
| 热过程 PINN | 14 条轨迹按 9/2/3 划分。NN / PINN 测试温度 RMSE 为 0.13750 / 0.15528 K，最大初值误差 0.09219 / 0.02087 K。PINN 方程残差 0.76459 W，独立有限差分复核差 1.22e−5 W。换列顺序结果一致，两类跨集合泄漏被拒绝。 | [训练对照](runs/thermal/paired/report.json)、[模型接口](runs/thermal/paired/pinn/interface.json)、[缺方程分支](runs/thermal/data-only/report.json)、[复核](runs/thermal/independent.json) |
| 弹簧有约束 MPC | 六组权重，用独立对象与三套工况。测试位移 RMSE 从 0.10138 降到 0.08171 m，同时力总变化由 5.98 增至 14.54 N，力 RMS 从 0.934 增至 1.331 N。输入、位移和速度越界均为零。错误周期被拒绝，重离散后完成复验。 | [配置](runs/mpc/controller.json)、[调参报告](runs/mpc/report.json)、[动作与状态轨迹](runs/mpc/selected.csv) |
| 专家模仿到 PGAC | 12 条轨迹、432 个样本，按 8/2/2 整轨划分。模仿动作测试 RMSE 0.16766 N。两分支导入权重后的动作差均为 0 N。留出工况 07/02 的位移 RMSE：专家约 0.06828/0.04770 m，模仿 0.06986/0.04865 m，名义滚动 0.06968/0.04846 m，验证集选出的感知耦合版本 0.06998/0.04882 m。耦合版本满足测试约束，两条测试轨迹跟踪均较模仿略差。 | [专家数据](runs/expert/data.json)、[学生](runs/imitation/report.json)、[PGAC 对照与耗时](runs/pgac/report.json)、[部署接口](runs/pgac/selected/interface.json)、[反例与独立闭环](runs/pgac/independent.json) |
| 原生与专项复核 | 原 SLX 回放 2001 点，输出最大差 4.44e−16；四/六参数误用被拒绝。归档专家数据重训动作 RMSE 0.15692 N。原热管理 Critic 实际滚动三步，更改感知参数前后 J 同为 0.383109，感知网络被调用 0 次。 | [Simulink](runs/matlab/report.json)、[参数辨识](runs/identification/report.json)、[热对象线性化](runs/linearization/report.json)、[存档数据重训](runs/archive-import/imitation/report.json)、[热 Critic 探针](runs/thermal-critic/report.json) |

## 来源与适用范围

**本次实跑**包括以上核验及现有 Harness 的 9 个原生工具签名、27 个操作 schema 查询。文件级来源清单记录所用本地原件和摘要，案例 README 说明哪些实现为自编、哪些配置来自存档。[Harness 适配](knowledge/harness.md) · [版本与状态](cases/legacy/README.md)

**历史存档**包括既有热管理及弹簧策略权重、原 NMPC 与历史训练报告、电池 RLS/EKF/DEKF、PID/ADRC/LQR 参考实现。基础回放中更改感知参数的探针验证了原 Critic 的实际连接；完整热对象训练另见热管理工程案例。[案例状态](cases/legacy/README.md) · [热管理训练](cases/heat-policy/README.md)

**仍待验证**包括客户现场数据与阈值、受限材料的外发授权、真实设备时延以及部署回退。热管理专家动作来自历史采数，本轮复验其物理轨迹。逐拍变化输入的二阶神经转移已另行训练，适用范围见导出接口。[神经转移](cases/object-plant/README.md) · [热策略](cases/heat-policy/README.md)
