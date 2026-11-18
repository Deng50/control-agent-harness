# 条件热过程 ODE PINN

对象模型入口提供 `prepare_pinn_problem` 和 `train_pinn`。当前实现支持显式的一阶热过程方程 `C*dT/dt=-G*(T-Ta)+P`，输入是时间、初始温度和整条轨迹保持不变的功率，输出是温度。时间用秒，温度用绝对开尔文，功率用瓦；`C` 用 J/K，`G` 用 W/K。不具备方程的数据使用普通 NN 训练路线。

## 材料与确认

先通过已有路径或上传机制登记一个不可变材料，包含问题声明、参考 CSV 和单独的配点 CSV。两个 CSV 在准备配置中使用材料根目录下的相对路径，例如 `source/reference.csv` 和 `source/collocation.csv`。列映射同时指定位置和精确表头。

准备配置明确方程参数、三维适用域、单位、归一化、每个工况的初温/恒定功率/分区，以及参考数据来源。训练、验证、测试按不同的完整 `(初温, 功率)` 工况划分，同一物理工况不能通过改名进入多个分区。配点只属于训练工况，不使用解析导数标签。

`prepare_pinn_problem` 的完整配置先作 signals 摘要确认，输出 `conditional_ode_problem`。`train_pinn` 使用该问题，并在执行前确认 signals 和 training 摘要。更改方程、单位、域、材料版本、网络或损失参数必须采用对应的新版本确认。已有同版本授权和兼容恢复使用原记录。请求 JSON 中的 `confirmed` 不能构成授权。

## 网络与损失

网络是 CPU float64 的平滑 MLP，输入维度为 3、输出维度为 1。用户可设置隐藏层、学习率、batch_size、总 epochs、patience、seed、weight_decay、min_delta 和 grad_clip；具体范围由工具 schema 验证。

`conditional_parameterization` 默认为 `free_mlp`。当前常系数线性热方程还支持 `affine_initial_power`：同一时间分别计算条件域三个角点的真实 MLP 输出，再用初温和恒定功率的归一化比例作仿射组合。该结构表达已知方程对两个条件的仿射性质，时间演化仍由网络学习；初值损失和物理损失仍通过实际梯度优化。此模式在训练摘要、报告、恢复签名和独立推理包中明确保存，激活内存预算包含三次网络计算。

令 `s=(t-t_origin)/time_scale`、`a=(T0-Ta)/state_scale`、`b=P/power_scale`，网络输出 `theta`，温度为 `Ta+state_scale*theta`。实际训练使用自动微分计算 `dT/dt=(state_scale/time_scale)*dtheta/ds`，保留物理损失对网络参数的导数。

损失由数据、方程残差和初值三部分组成，权重通过 `loss_weights.data/physics/initial` 设置。残差为 `[C*(state_scale/time_scale)*dtheta/ds+G*state_scale*theta-P]/physics_scale`，初值残差为 `theta(0,a,b)-a`。每个 epoch 累加完整各项均值梯度后进行一次 Adam 更新，batch_size 控制计算分块。

验证集的加权损失选择 checkpoint；测试集只在最终冻结权重后评估。数据 NRMSE 使用训练集总体标准差；训练目标精确恒定时，明确记录并使用声明的正 `state_scale_K` 工程尺度。不会从测试集拟合归一化参数。

`acceptance` 可省略或为 null，也可提供工程门限 `max_test_trajectory_nrmse`、`max_normalized_physics_rms`、`max_initial_normalized_error`。未提供门限时报告 `not_requested`，不能宣称精度通过。候选未满足所请求门限时保存拒绝报告并保留已有有效模型。

物理项为零仅允许显式 `ablation: true` 的对照训练；它标记为 physics-off ablation，记录损失与梯度对照，始终不激活、不替换有效 PINN。固定 F04 验收使用额外的已冻结工况和门限，这些数值不是所有用户项目的默认标准。

## 产物、恢复与使用范围

产物包含问题与单位声明、网络参数、选中权重、Adam/RNG checkpoint、训练日志、梯度探针、逐工况指标和测试预测。已有恢复机制识别 `pinn_model`；继续训练绑定问题、网络、优化器、归一化及损失合同，总 epochs 表示最终目标轮数。

`bundle/` 包含独立运行所需的权重、配置、manifest 和推理代码。将整个 bundle 复制到新目录，在有兼容 NumPy/PyTorch 的 Python 环境中从 `pinn_runtime.py` 导入 `load_pinn(bundle_path)`。`predict(time_s, initial_temperature_K, power_W)` 返回温度；`temperature(...)` 与 `residual(..., create_graph=True)` 提供 Torch 接口及物理残差。支持标量、等长一维数组或 N×1 输入以及显式标量广播；超域、形状冲突和非有限输入会被拒绝。

该模型表示给定初温和恒定驱动下的条件轨迹，尚未验证任意状态下一步转移、时变驱动或通用 PDE。它不提供 Plant 的 `step` 接口，`prepare_handoff` 不自动交给 MPC/PGAC；模型加载器会明确拒绝将其当作控制预测动力学。报告的解析物理轨迹比较和有限工况误差不等于全域精度或闭环稳定性证明。

验收状态以 [验收记录](acceptance.md) 和对应运行目录内的实际证据为准。
