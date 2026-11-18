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

`propose_case_plan` 创建 `signals/sampling/training` 摘要并绑定材料、数据与依赖版本；相同内容复用版本，变更使本 scope 及相关下游摘要失效。`read_case_plan` 读取提案，提案状态不等于某个 Session 的用户确认。

用户核对后通过原生 `/control-confirm {workspace,plan_id,version_hash}` 确认，`/control-plan-status` 读取本 Session 状态。模型不能替用户执行人工确认命令。生产确认关联真实原生用户命令记录，JSON 的 `confirmed`、`confirmation_source` 或预选答案均不是授权。

执行配置以 `plan_refs` 选版本。Host 在创建历史记录和数值 job 前检查版本、依赖和 Session 确认；worker 再校验绑定原始请求的 Host receipt。材料、语义、关键数值参数变化需要新版本；已验证兼容的 checkpoint/resume_run 作为恢复状态，复用未变的工程摘要。

升级前没有 plan_refs 的旧作业，可在 `/control-recover` 中显式提供当前已确认的 `plan_refs`。恢复仍检查原作业和 checkpoint 兼容性，未确认时不创建恢复 attempt 或 job。确认引用不参与优化器数值轨迹签名；学习率、数据、模型、接口等实际设置仍严格校验。

工作流按所选步骤联合所需 scope；只登记或只导入不要求无关采样、训练确认。受保护工作流的文件和依赖按版本固化，再交给已有领域执行器。

工程确认保护注册领域入口，原生执行权限仍由 Harness 管理。既有独立 Python/CLI 数值接口保留给明确调用它们的开发者，不冒充原生 Session 确认。自动化用例使用项目 test-output 内的隔离标记及 `test_fixture` 来源，不计为真实用户授权。
