# 外部专家与数据

三个入口共用原生 Harness 作业和材料协议。轻量化入口可直接接入已有专家，不要求先调用本系统的 MPC 设计工具。原 `collect_expert` 路线继续支持已在本系统产生的控制器；以下接口处理外部来源。

## 程序接口

通过 `register_material` 固化源码，再调用 `register_expert`，得到不激活的 `external_expert`。登记检查声明、源文件与接口，不执行源码。`expert.json` 保留材料版本、物理接口、观测与动作映射、控制周期、初始记忆及失败策略。

Python callable 和 MATLAB 函数使用相同的六项输入：观测、参考、扰动、上一实际动作、记忆、参数；输出为动作、下一记忆和状态。仅 `status='ok'` 允许使用动作。输出按 `action_mapping` 从专家坐标转换到对象控制通道；上一动作输入及 rate bounds 按对象控制通道排列。

观测必须说明来自 state 或 output 的哪个通道、名称、单位和可用性。当前支持零声明延迟；额外滤波/延迟需要显式适配。每条轨迹使用独立记忆及源码实例。Python 源码使用冻结快照和支持的 NumPy/SciPy 数值导入，不是任意工程的自动解析器或安全沙箱；MATLAB 函数使用明确的静态数值子集。复杂工程需要经过验证的薄适配器。

## 闭环采集

`collect_external_expert` 输入为 expert/plant 两个真实产物 ID。scenarios 每项提供 initial_state、reference、disturbance、steps 和 initial_previous_action，name 可选。控制周期必须为对象 dt 的整数倍；steps 是专家决策次数，对象在中间子步保持输入，每个子步都检查实际约束。变化率界限按每次控制更新的动作增量定义。

专家模式直接编辑工况；普通模式由原生智能体分析源码和接口，提出参考、扰动及初值候选。二者确认并执行同一配置。signals 摘要包括物理接口、观测、周期、记忆及动作映射，sampling 摘要绑定完整采集配置。源码与材料版本另由不可变依赖绑定。

支持 Python 专家配 Python/已登记 Torch 对象，以及 MATLAB 专家配 MATLAB 函数/受支持平坦 Simulink 对象。MATLAB 同一次本地批处理完成整组轨迹，使用共享专用会话及资源锁；跨后端需要显式桥接，不默认安装 Engine。复杂 Simulink 工程不在平坦已知模块子集内。

学生特征默认包含已声明可用的观测、参考、扰动和上一动作，显式 student_features 可选择支持的通道。simulation_only 不默认选入，显式选择时保留仿真范围。原始状态和内部专家记忆作为诊断数据，不自动成为部署特征。专家失败、非有限值、动作/对象/变化率越界或取消保留合法前缀，不发布成可训练的完整数据集。

显式 scenario.split 必须覆盖 train/validation/test；否则记录新生成的整轨划分。没有内部设计保留工况时，不伪造原控制器的验证/测试来源。

## 已有数据

`import_expert_data` 直接处理 CSV/XLSX 材料，复用 S03 的列索引、时间窗口、先划分后建窗和训练集缩放。输入接口为 DataInterface，信号追加 expert_role：observation/reference/disturbance/previous_action/expert_action/diagnostic_memory。

目标必须为当前专家动作。动作历史允许显式 lag≥1；同一时刻的目标不能作为输入。单独上一动作列需要 source_mapping 声明 `already_previous_applied_action` 及对应 action_target，不再次移位。内部记忆和不可用信号不能作为默认特征。

动作幅值和变化率由用户显式提供，不用数据极值推断硬件约束。物理接口可以为空；输入宽度不会被改写成对象状态维度。来源明确区分已有外部记录和声明的仿真，没有内部 controller ID。

导入仍只需要一次 signals 确认，摘要覆盖完整的实际导入配置：DataInterface、外置信号映射、动作约束、物理上下文、来源、划分/缩放及文件解析设置。修改其中任一设置会使旧版本确认失效；plan_refs 和兼容恢复字段不构成新的工程设置。普通 S03 数据准备的确认范围保持不变。

## 产物与验收范围

新数据类型为 `external_expert_dataset`，包含 expert.npz、DataInterface、scaler/split、来源映射、物理上下文、约束和诊断。`read_external_expert_dataset` 在读取时再次检查文件/依赖、数组预算、分区和训练集缩放一致性。神经网络训练、独立包与原对象比较见[外部专家模仿](external-imitation.md)。

本轮实际采集了 Python 外部 QP 与两个控制周期、MATLAB 外部 QP 与函数/SLX 对象；CSV/XLSX 导入使用真实已采动作。MATLAB 接受的两个批次各有一次业务工具执行和两次固定身份/空闲探针，所有相关原生 Session 没有模型回合。首个未加载计数观察器的运行只保留为历史记录，不作为调用次数证明；独立参考求解器的收敛修复没有改变专家、数据或数值门限。
