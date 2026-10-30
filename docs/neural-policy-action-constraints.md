# 神经策略的动作约束和前一动作

MLP、RNN、LSTM 的输入幅值边界来自模型接口 `controls`；输入变化率边界只继承专家控制器 `controller.json.parameters.input_rate_bounds`。策略训练和物理优化配置不能自行设置 `input_rate_bounds` 或 `action_constraints`，需要先修改并重新验收专家控制器。没有声明变化率时不增加默认约束。

变化率的单位是每个采样步的动作增量，不除以 `sample_time`。网络预测动作以后，训练和部署均投影到以下逐维交集：

```
lower = max(input_lower, previous_action + rate_lower)
upper = min(input_upper, previous_action + rate_upper)
action = clamp(network_prediction, lower, upper)
```

交集为空或前一动作不是正确维度的有限数值时，执行报错；不会静默丢弃其中一个约束。未启用变化率时只投影到幅值区间。

`NeuralPolicy.forward(x, reference, disturbance, previous=...)` 在启用变化率时必须传入与批量输出形状相同的 `previous`。其余情况下保留三参数调用。`act(x, reference, disturbance=None, previous=None)` 保持现有接口；未给 `previous` 时按一个新轨迹开始处理，使用投影到幅值范围内的零动作。连续部署时调用者必须传入实际执行的前一动作，策略不维护隐含动作历史。

专家采集会保存 `expert.npz.previous_u`。各轨迹首行使用幅值内零动作，后续行使用该轨迹上一行的真实控制动作。旧数据没有该字段时根据轨迹顺序重建；已经保存但不匹配的历史会被拒绝。模仿训练、验证和离线动作误差使用该前一动作；闭环验证使用策略自己实际执行的前一动作。物理训练从采样点的专家前一动作开始，每一步传入上一步网络动作，模仿保护项也使用采样点的真实前一动作。

`network.json`、`policy.json` 和 `report.json` 保存 `action_constraints`，包括专家控制器版本、继承的变化率、单位、投影算法版本和初始动作语义。加载产物时核对网络配置、策略描述和专家来源。旧策略若专家配置了变化率但产物没有继承记录，会要求重新训练。继承配置和训练动作语义进入断点签名；改变专家、变化率或训练语义不能续接原有优化器轨迹。

`constraint_checks` 明确列出状态、输入幅值、输出和输入变化率是否参与验收。未配置变化率时 `input_rate` 为 `false`；保留的 `max_rate_violation: 0` 仅为指标兼容，不是该约束已通过的证据。神经投影直接实施输入幅值和已声明的变化率约束；状态和输出约束仍需真实闭环验收，物理训练对它们使用惩罚项。专家数据、模仿策略及优化策略只有全部已声明约束通过时才发布为 `verified`；输出违规的专家数据发布为 `rejected`，下游不能消费。

验证：`tests/test_policy_rate.py` 使用真实 SLSQP 专家控制器和真实 Torch 训练，覆盖三网络、交集和可微前一动作、专家历史、输出违规拒收、未声明变化率、带约束的中断续训全状态一致性，以及变化率来源改变后的恢复拒绝。`tests/test_training_resume.py` 保持原有无变化率训练的恢复回归。
