# 只有数据时的神经网络训练

此路线不要求对象模型、名义模型或物理状态方程。路径和上传使用相同的材料快照：`register_material` → `inspect_table` → 确认信号语义 → `prepare_supervised_dataset` → 确认训练摘要 → `train_supervised`。CSV 和 XLSX 都按实际列位置选择变量，保留同名列、工作表、单位和原始来源；合成数据不能标作实车测量。

变量选择、历史窗口、可用延迟和预测目标在数据接口中声明。`input_dim/output_dim` 是特征和目标宽度，与物理 `nx/nu/ny` 独立；用户覆盖的维度与已准备数据不符时拒绝执行，不补零或截断。动态任务先按行程/时间分区再构造窗口，不跨分区借用历史；归一化仅由训练数据拟合。

`train_supervised` 只输入 `dataset` 产物 ID。可编辑 `hidden_dims`、`activation`（tanh/relu/silu）、`learning_rate`、`batch_size`、`epochs`、`patience`、`seed`、`weight_decay`、`min_delta`、`grad_clip` 和可选的最终 `max_test_nrmse`。当前训练使用 CPU/float64 MLP，线性输出；运行预算在实际配置中明确，超出单次内存/网络容量时先拒绝，不先分配巨量数组。学习率、层宽、参数量与解析后的完整设置写入产物。

训练只使用 train，最佳权重只按 validation 选择，test 在训练和选模结束后评价一次。报告包括真实 loss、最佳轮次、逐输出 RMSE/NRMSE、训练目标的总体标准差以及数据来源。NRMSE 使用训练目标尺度；没有物理模型时不生成虚假的物理模型对比，也不把普通数据损失称为 PINN。未声明业务精度门限时，报告标记 `acceptance:not_requested`，不宣称满足未给定的精度目标；不达预声明门限的产物保留为 rejected，不替换先前有效模型。

`epochs` 表示累计目标总轮次。每个完整 epoch 使用共享原子 checkpoint 保存网络、Adam、最佳权重、随机状态和日志。取消后可从工作区失败目录中的 `latest.pt` 创建新作业；请求的网络、学习率、数据/接口版本和预处理等必须匹配。相同工程参数与总预算的恢复复用原确认；续训路径和恢复位置不构成新授权。更换关键参数不能强行续接旧优化轨迹。

`supervised_model` 包括实际权重、checkpoint、解析配置、训练与测试报告及最终测试预测数据。其 `bundle/` 是独立推理目录：权重、网络定义、scaler、数据接口、特征/目标顺序、窗口/延迟/单位、推理程序、文件哈希和运行说明随包保存。按包内说明提供已组装的原始特征矩阵，程序进行训练时的缩放并输出原目标单位。动态窗口须按清单组装，不能从网络宽度推断物理状态。独立目录不依赖原材料路径、`control_harness`、MATLAB 或 MCP。

用户仅要求网络训练时在报告和导出后结束。一个输入输出回归网络不自动成为 MPC 预测模型；后续控制器任务需要另行验证状态映射和动力学接口。
