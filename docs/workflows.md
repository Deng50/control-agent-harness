# 用户项目工作流

`plan_workflow` 和 `run_workflow` 使用同一版本化流程定义。它们在 Harness 的一个后台作业内顺序调用领域阶段，不另建智能体循环或任务调度器。可只包含一个阶段，也可组合模型导入、数据导入/仿真、残差学习、LS 辨识、MPC、估计器验证、专家采集和策略训练。

请求示例：

```json
{
  "agent": "workflow",
  "operation": "plan_workflow",
  "workspace": "D:/projects/control-agent-harness/workspaces/project-01",
  "inputs": {"workflow_path": "D:/projects/control-agent-harness/examples/linear/workflow.json"}
}
```

预检后把 `operation` 改为 `run_workflow`。前端“用户项目串联工作流”提供相同入口。`examples/linear/workflow.json` 和 `examples/nonlinear/workflow.json` 都是完整可修改的八阶段文件；示例数据明确来自数学仿真。

流程文件：

```json
{
  "schema_version": 1,
  "name": "用户项目",
  "steps": [
    {"id": "nominal", "operation": "register_model",
     "inputs": {"model_path": "nominal.json"},
     "config": {"source": {"kind": "user_nominal", "description": "用户声明的离散方程"}}},
    {"id": "data", "operation": "import_transitions",
     "inputs": {"nominal_model": {"stage": "nominal"}, "data_path": "measurements.npz"},
     "config": {"source": {"kind": "measured", "description": "实际实验批次及采样条件"}, "seed": 42}},
    {"id": "corrected", "operation": "train_residual",
     "inputs": {"nominal_model": {"stage": "nominal"}, "dataset": {"stage": "data"}},
     "config": {"epochs": 150, "hidden_dims": [32,32], "learning_rate": 0.002, "seed": 42}}
  ]
}
```

`{"stage":"id"}` 只能引用前面的有效阶段。字符串也可以直接填写当前工作区的已验证产物 ID；外部产物须先用 `import_artifact` 复制完整依赖。文件中的相对文件/目录路径相对于流程文件所在目录；内联 `config.steps` 的路径必须绝对化。不会在执行途中改用新的 active 产物。

预检检查源文件、产物与父依赖哈希、阶段类型、接口兼容性、已知训练参数及工况声明。它不承诺后续数值可行或训练合格；每个阶段仍实际执行并验收。任一阶段 rejected，后面的阶段保持 pending。失败和取消不会覆盖此前的有效产物。

## 三种恢复的区别

1. **记录读取**：Harness `/control-history` 与 `/control-status` 读取持久 attempt。服务器重启后原生 job ID 可能不存在，仍能查看请求、结果和中断状态。
2. **工作流接续**：`config.resume_run` 填 `workflow_run`。严格核对流程、源文件和依赖，复用已完成且来源标记一致的产物。未完成阶段重新执行；产物发布后记录尚未落盘的情况，通过精确的 workflow 来源标记恢复，不能按最新目录猜测。
3. **训练续训**：单阶段 `config.resume_checkpoint` 恢复网络、优化器、最佳验证状态和随机状态。见 [训练检查点](training-checkpoints.md)。工作流中的显式 checkpoint 输入还会把预检 SHA-256 传给实际加载器。

在前端持久任务列表中，“按原请求重试”生成新原生 job；“从检查点续训”或“从已完成阶段续接”根据已记录的任务类型使用上述语义。已完成流程也可显式用 resume_run 再次核验；不会重复生成全部产物。

整个串联运行持有工作区互斥锁，内部阶段共享该锁。锁由操作系统持有，进程退出会释放；诊断文件只有在操作系统确认旧 PID 已不存在/退出后才自动回收。无法确认旧所有者时会拒绝恢复，不按文件年龄判定。跨工作区计算并发仍由 Harness 插件的 worker slots 限制。

强制终止可能留下 `.staging` 中的未发布训练文件；当前不自动把这些文件当作可用 checkpoint。此类任务能显示 interrupted 并重新执行，已有 `failed/` 或 `artifacts/` 中通过签名验证的检查点才可续训。
