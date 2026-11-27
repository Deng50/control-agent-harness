# 从模仿策略到闭环微调的验收记录

## 历史实验复核

历史弹簧工作区 `workspaces/test_11_27_02` 的 `train_pgac.py` 从模仿参数开始，对固定初态的 241 个控制步整轨求梯度。回合内确实有策略状态反馈。下一轮从同一组初态重新开始，没有保留当前学生在中途访问的状态用于抽取短窗口。63 个增强条目按有效初态、参考和真正生效的噪声去重后只有 32 种。

第 150 轮的训练 J 下降 1.40%，验证 J 上升 2.01%。在归档的 `runs/pgac/compare3.json` 中，T08 负阶跃由 0.228862849 升至 0.241916786，T09 正阶跃由 0.072814625 降至 0.065830662。两条等权，净增加 0.003034987。工况权重和轨迹起点分布足以让训练均值变好、验证变差。按验证目标，最好的仍是更新前的模仿权重。历史指标与下文的新基准采用不同的物理接口和目标定义，数值仅在各自实验内可比。

第一版 AIagent 的 `simu_pac_align_v5_delay_5s.py` 将状态、动作、下一状态、参考、时刻、回合号写入经验池。抽到时刻 t 后，在同一回合内读取 t+k 的参考和车速，让预测状态反馈给 Actor，累积有限时域跟踪、功耗和衰减代价，再用自动微分更新 Actor。`kernels/pgac_core.py` 的对应循环反复读取起点时刻 t。两份实现的 Actor rollout 虽然接收 perception 参数，却没有将该感知残差用于状态更新。新实现按每拍推进外生量，并显式绑定已验证且实际参与反传的模型。

## 三条执行路径

| 入口 | 实际用途 |
|---|---|
| 内置策略 `optimize_policy` | 仓内配套模型与策略的优化入口，保留原调用方式。 |
| 外部策略 `optimize_external_policy` | 公开产品入口。从已登记的真实模仿包继承网络、特征顺序、单位、缩放和投影。选择 `replay.mode=online` 后，交替执行学生在原对象上的闭环采集和冻结可微模型上的片段更新。 |
| 会话自编的 `train_pgac.py` | 历史回归材料。新案例通过角色工具和正式作业完成。 |

在新工作区选择 `control-lightweight`，提供原对象、MPC、显式信号映射、工况与验收要求。角色加载 `distill-controller`，按需要检索 `warmstart-pgac`。登记支持材料的公开操作为 `register_material`、`register_plant`、`register_expert`。通过 `collect_external_expert` 和 `train_external_imitation` 得到模仿包。可微模型通过 `register_model`、`sample_model`、`validate_differentiable_model` 资格检查后，角色调用 `optimize_external_policy`，通过 `prepare_handoff` 取得冻结的评价合同，最后调用 `evaluate_external_pgac`。通过验收才产生可交付的 `external_optimized_policy/bundle`。教学数据仅为静态记录时，运行到离线模仿及动作误差评估。

## 双对象数值实验

每个对象的三组 PGAC 共用一份实际模仿包、验证及盲测工况和 192 次 Adam 更新。预设门限要求验证和盲测的等权均值 J 至少改善 0.5%，任何单一工况最多退化 5%，硬约束满足后才能发布。T08–T12 均未进入这次盲测。弹簧有 432 个专家训练决策，全部分区共 720 个。电机分别为 360 个和 648 个。每个有效 Python 专家决策还包含两次一致性重算，因此两例全部分区实际执行专家入口 2,160 和 1,944 次。两例在线组各增加 1,152 个学生原对象控制步，PGAC 拟合新增专家调用为零。评价与计时的专家执行另计。

| 原对象 | MPC J | 模仿 J | 原 PGAC J | 仅配平 J | 在线 PGAC J | 在线流程交付 |
|---|---:|---:|---:|---:|---:|---|
| 弹簧 | 0.00334529 | 0.00335680 | 0.00315904 | 0.00317228 | 0.00315465 | 第 9 轮在线权重 |
| 直流电机 | 0.00714039 | 0.00712527 | 0.00697969 | 0.00703640 | 0.00714768 | 模仿权重 |

弹簧相对模仿的 J 按族等权计算，正阶跃改善 4.20%，负阶跃改善 5.61%，跨零改善 7.51%，中途接管改善 2.88%。8 条均改善。整体 J 下降 6.02%，较原 PGAC 再下降 0.14%。力 RMS 下降 6.89%，位置跟踪 RMSE 上升 5.68%。该物理代价体现了用户预先指定的动作权重。此机上在线策略单次推理中位数为 267 µs，凝聚型 MPC 为 82 µs。

电机在线候选的均值 J 比模仿高 0.315%，最差工况高 5.32%。原对象验证阶段已选择模仿，盲测确认交付基线。旧 PGAC 与仅配平组通过了各自的预设门限。各组、各对象盲测硬约束越界均为零。每个对象只有一个训练种子，结论限定在记录的工况和部署接口。六个发布包均经隔离 Python 进程在原对象逐拍复算，最大状态和动作差为零。[协议、分组指标和复现命令](../examples/pgac-online/README.md) · [逐工况数据及冻结哈希](../examples/pgac-online/results.json)

## 真实角色任务

真实会话 `session-8d5bc5e9-347d-4ee4-af4d-2efd4ffe3d05` 在全新工作区运行。用户提供弹簧程序、专家、物理接口和冻结工况，模型实际读取两个技能并自行发起各项训练与评价操作。测试观察器按原生确认卡核对冻结内容，任务中修复过一次观察器的 Windows 目录权限。此次任务使用与数值基准相同的冻结验收材料，验证从自然语言到策略包的实际执行链。

会话选中第 9 轮候选 `external_pgac_candidate-0fb1f9a7f8ea`，归档 `external_optimized_policy-cb3943a33ca4`。独立进程载入本会话的交付包后，在 8 个原对象盲测工况复得 J = 0.003154648286140705，逐拍最大动作和状态偏差为 0。实际操作回执、模型答复纠正和详细复现说明见[角色验收记录](../examples/pgac-online/dialogue-evidence.json)与[运行命令](restricted-pgac.md)。

发布选择冻结后，另做贴近物理边界的镜像压力检查。每个对象 144 拍，起始状态为相应限制的 93%–94%，上一动作为幅值限制的 95%。两个方向均实际触及每拍动作变化率上限，两个对象均无越界。该检查没有参与训练或选择。[压力工况和原始结果](../examples/pgac-online/boundary-stress.json)

## 原生作业与恢复

在运行中的本地 Harness Host 上，`scripts/smoke-online-pgac.mjs` 通过真实作业命令触发采集，观察到持久化进度后终止该训练 worker，用 `/control-recover` 恢复。与从头连续运行的对照相比，8 轮、96 次更新、144 个学生控制步的最终权重、最优权重、Adam、随机状态、经验池和训练日志逐项完全一致。随后经交接进入原对象三方验证。

```powershell
node scripts/smoke-online-pgac.mjs --web-log <本地Host启动日志>
```

旧 `smoke-restricted-pgac.mjs` 与 `smoke-restricted-pgac-recovery.mjs` 的 S12 注册文件固定了历史产品源码哈希。实测两者都在源码哈希预检处停止。历史注册与结果原样保留。上述新 smoke 覆盖本版 Python 原对象在线路径的真实作业和恢复。MATLAB 动态外生量及在线学生采集没有在本版开放。

Windows Host 上的 Python 全量回归通过 695 项，插件全量回归通过 259 项。最后修改的经验恢复、子步、成本和门限检查还分别运行了 9 项回放测试、32 项专家与发布测试和 6 项插件确认测试。知识快照核验了 516 个资产及 104 个评测夹具，三个角色的已安装预设通过同步检查。

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
node --test --test-concurrency=2 'harness-plugin/tests/*.test.mjs'
.\.venv\Scripts\python.exe -B scripts/sync-control-library.py --check
node scripts/sync-control-presets.mjs --check
```
