# 连续动力学适配

`register_model` 接受 `continuous_polynomial`。A/B/E/bias/terms 定义状态导数，C 定义输出。必须显式提供：

```json
{
  "representation": "continuous_polynomial",
  "time_unit": "s",
  "derivative_units": ["K/s"],
  "integration": {"method": "zoh", "substeps": 1, "input_hold": "zero_order_hold"},
  "interface": {
    "schema_version": 1, "name": "first-order-temperature",
    "sample_time": 0.2, "time_semantics": "continuous_state_derivative",
    "states": [{"name":"temperature_deviation","unit":"K","lower":-2,"upper":2}],
    "controls": [{"name":"power_deviation","unit":"W","lower":-1.5,"upper":1.5}],
    "outputs": [{"name":"temperature_deviation","unit":"K","lower":-2,"upper":2}],
    "disturbances": []
  },
  "A": [[-1.4]], "B": [[0.8]], "C": [[1]], "E": [[]], "bias": [0], "terms": []
}
```

`zoh` 用增广矩阵指数计算仿射对象的精确离散映射，允许奇异 A，要求无非线性 terms 且 substeps=1。Euler/RK4 支持非线性多项式与 1–1024 子步；每个采样周期内控制和扰动保持恒定。固定步积分不提供自适应误差保证。

注册产物同时保存原始 `model.json` 和 `transition_adapter.json`。产物协议与加载器公开离散状态转移，`forward/step` 返回完整采样周期后的 next_state；`derivative` 单独返回状态导数。仿射对象公开的 A/B/E 是离散矩阵，KF 使用同一离散时间语义。读取时重新计算并核对适配器声明。

残差监督训练目标为实际 next_state 与积分后的名义 next_state 之差，单位为状态单位。残差加在完整积分步之后，不按子步重复累加。显式请求 derivative 残差会被拒绝。LS 从转移数据拟合的结果保存为 `discrete_polynomial`，避免把拟合结果再次积分。

单位检查只核对显式声明，不证明各系数的物理量纲，也不自动换算单位。Python/SLX/MDL 检查仍是静态材料分析；本适配器执行确认过的声明式方程，不执行上传的任意程序或原始 Simulink 模型。

新残差报告的 `applicability` 记录训练分区的状态、控制、扰动实际最小/最大值、声明边界和样本数。验证/测试样本不参与这些统计。逐变量范围不证明联合工况覆盖或外推有效性；声明边界也不等同已验证的模型适用范围。

`tests/test_continuous.py` 使用解析解、独立 SciPy ODE 求解、输入梯度、KF 与真实残差→约束 MPC 闭环验证数学语义。
