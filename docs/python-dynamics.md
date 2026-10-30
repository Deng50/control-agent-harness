# 独立 Python 动力学

`register_python_model` 可将一个明确声明接口的独立 PyTorch 文件接入同一条残差训练、MPC、专家采集和策略优化链。原文件保持不变，产物内保存 `source.py`、源哈希、模型声明、运行依赖与数值探测报告。后续执行读取这份源码快照；跨目录导入会复制完整依赖闭包。

先使用“检查对象材料”。这一步只读 AST，不导入或执行 Python。确认接口后选择“运行并导入 Python 动力学”，才会在 Harness 原生数值作业中执行源码。源码必须经过检查；AST 限制不是安全沙箱，实际权限由 Harness 的子进程与沙箱配置决定。

## 函数约定

文件中提供字面量 `MODEL_SPEC`，或在配置中显式提供完整 `model_spec`。参考可运行文件 `examples/python-tensor/model.py`。

```python
def transition(x, u, d, parameters):
    # CPU float64 Tensor，支持单样本和任意前置批量维度。
    # 返回下一状态，最后一维长度必须等于状态数。
    ...

def observe(x, parameters):
    # 返回输出向量；也可以用明确的 C 矩阵替代此函数。
    ...
```

- `python_torch_transition`：函数返回下一时刻状态，接口为 `discrete_state_transition`。
- `python_torch_derivative`：函数返回状态导数，必须声明时间/导数单位、采样周期和 Euler/RK4 积分配置；外部统一提供离散转移接口。残差仍以状态单位训练，不会重复积分。
- `parameters` 为有限 JSON 常量。状态、控制、输出、扰动的顺序、名称、单位和边界由接口声明，不能只凭函数参数名猜测。

注册检查单样本/批量形状、有限输出、重复调用、输入未被修改、独立导入的随机种子一致性，以及自动微分与有限差分的转移/输出梯度。探测只覆盖记录的点，不证明全域可微、物理真实性或全工况安全。模型进入下游时重新检查源码完整性与运行依赖。

当前支持已安装的数学/Torch 库及独立、可微、确定性的函数。相对工程导入、外部数据/权重文件和训练时随机初始化不属于这个适配器；冻结物理网络需要明确的固定参数且禁用参数梯度。脱离计算图的 NumPy 转换不能用于这条可微链。不要把此入口称为任意 Python 工程自动迁移。

## 实际案例

`examples/python-tensor/workflow.json` 包含 3 状态、2 控制、2 非线性输出、1 扰动的八阶段案例，动力学含 `tanh`、`sin` 和状态耦合。名义与参考参数不同，数据明确来自数学仿真。可在界面“用户项目串联工作流”预检后运行，也可执行 `node scripts/smoke-business-dialogue.mjs`，由真实 DeepSeek 对话规划、提交并读取原生作业。

`docs/harness-business-dialogue-evidence.json` 记录已完成的实际运行及原始报告：残差测试总 MSE 下降约 14.01%，但第二个状态的误差略增；GA 候选通过独立复验。神经策略推理更快，测试跟踪仍弱于专家，短程物理梯度优化没有改善测试跟踪。该小规模案例验证工程接通，不代表训练预算充分或已达到部署性能。
