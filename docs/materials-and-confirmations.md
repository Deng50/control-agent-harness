# 材料、数据接口与工程确认

此页描述 S03 接口。采样、训练及控制器分支以当前 `control_help` 和验收清单的实际支持范围为准。

## 路径与上传

`inspect_material` 读取原件及显式依赖，不运行源码。`register_material` 将字节快照和来源写入不可变 `material` 产物。支持 CSV、XLSX、Python、MATLAB `.m`、SLX、MDL 和 JSON；格式识别不代表程序已被验证可执行。

路径入口传 `inputs.local_path`。上传入口由原生 `fileUploads/upload` 和 `/control-upload` 接收文件，再将返回路径传给同一个 `register_material`，指定 `source_kind:upload`、`original_name` 和上传来源。缓存名不用于猜原始格式。相同字节和依赖具有相同内容哈希，两种入口的来源分别保留。

依赖必须在显式 `dependency_root` 内，不沿符号链接或 Windows Junction 展开。复制前后核对清单、原件及哈希，变化时拒绝发布。默认上限为 2048 个文件、单文件 64 MiB、合计 256 MiB、目录深度 32。多份独立材料共存，登记另一原件不会替换已有材料。

`.m` 静态检查可提出函数头和数值字面量候选，不执行 MATLAB，也不据此承诺完整维度。实际执行由可选 MCP 和对应适配器完成，见[连接说明](matlab-existing-session.md)。

## 数据接口

`tabular_supervised` 独立于物理 `ModelInterface`，不虚构 nx/nu/ny。

- `signals`：名称、角色、单位、列位置、可观测性及 `delay_steps`。
- `column`：零基 `index`、原始 `header`、明确的 `worksheet`（CSV 为 null）；同名列靠位置区分。
- `features`：`signal/window/lag`；窗口内由最旧到最新排列。
- `targets`：`signal/offset`；0 为同一行，正数为预测标签。
- `time_column/trip_column`：秒制数值时间及行程/车辆分组。
- 可选 `state_mapping`：仅关联显式 state 角色；特征宽度不代表物理状态数。

在锚点 k，特征使用 `k-lag-delay_steps-(window-1)` 到 `k-lag-delay_steps`，标签为 `k+offset`。delay 表示信息可用性，不是自动辨识的物理传输延迟。不可观测特征、负 lag、当前标签回填特征及重复信号/时刻均被拒绝。

静态回归使用同一行；动态预测要求时间列和正 sample_time。当前不自动重采样、计算 Excel 公式、转换单位或判断电流符号。请求/实测通道、单位及可用时间应在信号摘要中明确。

`prepare_supervised_dataset` 消费材料及确认的接口。原始行先按独立行程或时间块划分，再在各份内建窗口；窗口不跨行程或分区，Scaler 只 fit 训练集。整组划分下实际比例可能受行程数和长度影响，实际组数、样本数、行索引及拟合统计均落盘。

产物包括原始/缩放 X/y、分区证据、接口、scaler、统计及预处理报告；此步骤不训练网络，没有 Plant 时不生成物理基线。

## 确认版本

普通用户在对话或侧栏查看同一张工程确认卡，核对后点击“确认以上摘要”。点击只签署显示的方案。侧栏可继续执行阶段，对话可发送“继续”。模型使用 `control_prepare_review` 生成卡片。执行时提交原请求和顶层 `card_id`，Host 选择对应阶段的版本。工作区按 Session、角色和任务标识隔离工程意图，同一任务的修改使用原 `intent_id`，旧快照可审计。

数据准备卡展示来源、工作表、列与时间映射、划分和缩放。选择“同时审阅训练方案”会把具体网络与训练预算加入同一次审阅。准备作业为实际生成的数据及配置签发派生凭证。后续训练同时核验当前工作区和 Session、原料父链、数据字节、信号版本及网络配置。单独准备、导入旧数据或修改方案各有独立的确认边界。

两个页面都通过原生 `control-confirm` 命令签署卡片。`control-card-status` 读取该 Session 内的卡片状态。旧版单计划命令和 `control-plan-status` 继续支持历史会话。高级命令每次只提交一个标准 JSON 对象，示例 `/control-confirm {"workspace":"D:/project","card_id":"<当前卡片标识>"}`。用户口头回复、模型生成的标记和预选项均不能签署。历史 v1 确认只能按旧范围执行，新卡重新核对工程语义。

执行配置以 `plan_refs` 选版本。Host 在创建历史记录和数值 job 前检查版本、依赖和 Session 确认；worker 再校验绑定原始请求的 Host receipt。材料、语义、关键数值参数变化需要新版本；已验证兼容的 checkpoint/resume_run 作为恢复状态，复用未变的工程摘要。

升级前没有版本引用的旧作业，可在侧栏点击“载入原请求以审阅”，在原会话签署卡片，再重试或续接。Host 比对原请求、卡片、作业归属和已有 checkpoint，成功后只创建一个新的恢复作业。旧脚本也可向 `/control-recover` 提供已签的 `plan_refs`。确认引用不参与优化器数值轨迹签名，学习率、数据、模型、接口等实际设置仍严格校验。

工作流按所选步骤联合所需 scope；只登记或只导入不要求无关采样、训练确认。受保护工作流的文件和依赖按版本固化，再交给已有领域执行器。

工程确认保护注册领域入口，原生执行权限仍由 Harness 管理。既有独立 Python/CLI 数值接口保留给明确调用它们的开发者，不冒充原生 Session 确认。自动化用例使用项目 test-output 内的隔离标记及 `test_fixture` 来源，不计为真实用户授权。
