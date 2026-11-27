# 网络与物理训练

输入/输出由确认的变量、窗口和目标决定；维度冲突解释并拒绝，不静默截断、补零或广播。提供层数/宽度、学习率、batch、epochs、早停和种子，保存实际生效配置。Scaler仅fit训练集，验证集用于选模，测试集不参与选参；按行程/时间/工况独立划分。

区分四类结果：普通NN使用数据损失；物理约束NN列出具体约束；机理加残差学习对齐偏差并保留组合接口；方程PINN需要可信方程、变量/参数映射、真实方程残差及初/边条件。仿真来源、L2和输出clamp不是PINN的充分条件。

条件ODE PINN明确时间、初值、输入驱动条件、配点、自动导数和各loss尺度。多初值/驱动不能混成单一无条件时间函数；导数残差不能直接加到下一状态。没有方程先做NN，不编造物理损失。

当前热过程模板通过 `prepare_pinn_problem` 和 `train_pinn` 执行。网络学习 `theta(s,a,b)`，其中 `s=(t-time_origin_s)/time_scale_s`、`a=(T0-Ta)/state_scale_K`、`b=P/power_scale_W`；重建 `T=Ta+state_scale_K*theta`。方程项实际为 `[C*(state_scale_K/time_scale_s)*dtheta/ds+G*state_scale_K*theta-P]/physics_scale_W`，初值项为 `theta(0,a,b)-a`。数据误差指标尺度由训练参考温度确定，精确常量训练标签使用事先声明的正状态工程尺度，并记录来源；不能拿测试数据定尺度。

每个完整 epoch 按全样本均值分块累积三项梯度后进行一次 Adam 更新；此处 batch_size 控制计算分块。保留实际 seeded 权重、best/last、Adam/RNG、分项梯度与精确恢复签名。物理项关闭对照保持非激活，算法与报告明确为 physics-off；它与有方程梯度的 PINN 分开标记。未请求工程门限时只报告实际误差，不宣称达到某个业务精度。

热过程的 `conditional_parameterization` 默认 `free_mlp`。可选 `affine_initial_power` 利用恒定 C/G 热方程对初温与恒定功率的仿射性质：同一时间在声明域的三个条件角点实际运行同一个 MLP，再按初温/功率坐标仿射组合。时间函数由网络学习，方程残差和初值项仍实际参与训练；不使用解析温度曲线，也不把初值直接固定为零损失。该模式必须出现在训练确认、checkpoint 和独立包声明中，不可默默用于其他非线性方程。

训练前汇总配置确认，之后按实际预算执行。保存best/last、checkpoint、分项loss、数据划分和预处理。导出含变量顺序、单位、窗口/记忆、权重、完整配置及推理脚本。仅在已验证rollout接口适用时才把训练模型交给MPC/PGAC。

数据-only 使用 `prepare_supervised_dataset` 后的 `train_supervised`，只输入 dataset，不要求 nominal_model。先查 control_help 并展示网络/层宽、学习率、划分、累计目标 epochs 和产物摘要，确认 signals/training 版本后执行。best 仅由 validation 选择，test 最终一次评价；无业务门限时 acceptance:not_requested，不能宣称达到未给定精度。结果的 bundle/ 可独立推理，历史特征依照其明确顺序组装；用户只要 NN 时到此结束。

单个对象的 `sampled_dataset` 使用 `train_predictive_model`，特征为当前完整 [x,u,d]，标签为下一状态，不要求补造名义模型。C 必须显式并能对上源对象输出，缺失或非线性映射不猜单位阵。报告区分 teacher-forced 一步与仅用初态的多步开环；源对象是模拟数据时不能称实车真值。完整状态是否可在部署时测量/估计是后续控制器接口条件，不能由采样时读得到状态就宣称硬件全测量。

外部专家数据使用 `train_external_imitation`，训练设置放在network，动作NRMSE门限和尺度在外层；恢复checkpoint也放在外层。复用数据原有整轨划分和S05训练，验证集选择best，复用最终一次测试前向结果评价原始/投影动作。因果窗口可作为明确的平铺输入训练；自主闭环需要独立可用的历史组装接口，不能用零补齐。独立bundle含权重、Scaler、输入输出顺序和动作投影。旧内部MLP/RNN/LSTM导出保留原数值；其RNN/LSTM为声明的常值未来扰动预览，不冒充历史序列模型。

在线 PGAC 使用两个对象入口。当前学生在可执行原对象上产生轨迹；模型资格回执绑定的可微下一状态模型负责从池中起点反传。抽片段时恢复真实状态与上一实际动作，沿同一回合的控制时钟读取参考、扰动和观测噪声，并用当前 Actor 逐拍重新决策。策略只能看到当前部署接口声明的信号。动作在对象子步间保持。训练和报告注明实际的学生采集步数、MPC 调用数、更新次数和入池策略版本。全状态映射、模型域及资格有效性均须满足外部 PGAC 合同。
