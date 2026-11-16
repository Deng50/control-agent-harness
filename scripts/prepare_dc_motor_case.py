"""Prepare explicit DC motor simulation inputs; does not train or run stages."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
EX=ROOT/'examples'/'dc-motor'
WS=ROOT/'workspaces'/'dc-motor-web-20261109'
EX.mkdir(parents=True,exist_ok=True)
(WS/'presentation'/'screenshots').mkdir(parents=True,exist_ok=True)
def write(name,data):
    (EX/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def variable(name,unit,low,high): return dict(name=name,unit=unit,lower=low,upper=high)
def model(p):
    R,L,J,b,Ke,Kt=[p[k] for k in ['R','L','J','b','Ke','Kt']]
    return dict(representation='continuous_polynomial',interface=dict(schema_version=1,name='dc-motor-speed',sample_time=.1,time_semantics='continuous_state_derivative',
        states=[variable('armature_current','A',-12,12),variable('angular_speed','rad/s',-40,40)],
        controls=[variable('armature_voltage','V',-24,24)],outputs=[variable('angular_speed','rad/s',-40,40)],
        disturbances=[variable('load_torque','N*m',-.1,.1)]),time_unit='s',derivative_units=['A/s','rad/s/s'],
        integration=dict(method='zoh',substeps=1,input_hold='zero_order_hold'),
        A=[[-R/L,-Ke/L],[Kt/J,-b/J]],B=[[1/L],[0]],E=[[0],[-1/J]],C=[[0,1]],bias=[0,0],terms=[])
nom=dict(R=2,L=.5,J=.02,b=.02,Ke=.1,Kt=.1)
ref=dict(R=2.3,L=.55,J=.024,b=.024,Ke=.105,Kt=.095)
write('nominal.json',model(nom));write('reference.json',model(ref))
write('physical-parameters.json',dict(equations=['L di/dt = U - R i - Ke omega','J domega/dt = Kt i - b omega - tau_load'],nominal=nom,reference=ref,source='illustrative_mathematical_simulator_not_measured_motor',units=dict(R='ohm',L='H',J='kg*m^2',b='N*m*s/rad',Ke='V*s/rad',Kt='N*m/A')))
write('data.json',dict(trajectories=30,steps=40,seed=241))
write('residual.json',dict(input_dim=4,output_dim=2,epochs=300,hidden_dims=[24,24],learning_rate=.005,batch_size=64,patience=50,physics_weight=.1,seed=242))
def scenario(name,x,target,d,steps=30):return dict(name=name,initial_state=x,reference=[target],disturbance=[d],steps=steps)
write('controller.json',dict(parameters=dict(horizon=6,q=1,r=.15,s=.04),parameter_space=dict(r=dict(lower=.0005,upper=.2),s=dict(lower=.0001,upper=.08)),budget=12,seed=243,
    tuning_scenarios=[scenario('motor-tuning',[0,-5],14,.04)],verification_scenarios=[scenario('motor-verification',[1,6],-12,-.05)],
    test_scenarios=[scenario('motor-final-test',[-1,-3],16,.06)]))
scenes=[scenario(f'motor-expert-{j+1:02}',x,t,d) for j,(x,t,d) in enumerate([
    ([0,0],12,.02),([1,4],-10,-.03),([-1,-4],16,.04),([0,5],-14,-.05),
    ([2,2],10,.06),([-2,-2],-16,-.02),([0,-6],14,.03),([1,6],-12,-.04),
    ([-1,-5],8,.05),([0,3],-8,-.06),([2,-3],18,.02),([-2,3],-18,-.03)])]
write('expert.json',dict(scenarios=scenes,seed=249))
write('imitation.json',dict(architecture='mlp',input_dim=4,output_dim=1,epochs=600,hidden_dims=[64,48],learning_rate=.002,batch_size=64,patience=100,dropout=0,seed=245))
write('optimize.json',dict(epochs=40,horizon=6,learning_rate=.0003,batch_size=24,imitation_weight=.1,constraint_weight=100,seed=246))
(EX/'README.md').write_text('''# 直流电机转速控制：网页逐阶段演示

被控对象为教学用数学参考仿真，不是实测电机。模型为 L di/dt=U-Ri-Keω，J dω/dt=Kt i-bω-τL。参数与单位见 physical-parameters.json。

状态顺序：[电流 A，转速 rad/s]；控制：电枢电压 V；输出：转速 rad/s；扰动：负载转矩 N·m；采样 0.1 s，精确零阶保持。电压 ±24 V，电流 ±12 A，转速 ±40 rad/s，负载转矩 ±0.1 N·m。假设完整状态与扰动可得，未运行观测器。

模型校正：准备 nominal.json 和 reference.json；确认模型角色、状态顺序、单位、边界、采样、数据来源。检查材料、确认接口、导入名义与参考模型，再按 data.json 采集和 residual.json 训练。输出转移数据、校正模型、残差权重、归一化、独立测试报告。若用实测数据替代 reference.json，需提供 x/u/d/next_x/trajectory_id 的同周期对齐数据。

控制器设计：输入校正模型 ID、参考模型 ID 和 controller.json。确认转速目标、初始状态、负载、约束、预测步数和 r/s 调参范围；调参、复验、测试工况独立，测试不参与选择。输出 controller.json、候选搜索和闭环报告；无额外电压变化率硬约束，s 为变化软惩罚。

控制器轻量化：输入专家控制器与参考模型，按 expert.json 采集；确认工况覆盖和按工况划分；按 imitation.json 训练，输入 [i,ω,目标转速,负载]，输出电压。再输入模仿策略 ID 与同版本校正模型 ID，按 optimize.json 做可微物理 rollout 策略优化。输出 policy.pt、结构、归一化、检查点、专家/模仿/优化三方报告。物理优化不含学习型 Critic，不命名为 PPO/SAC。

所有数值以本次真实报告为准；不得因任务完成推断效果改善。
''',encoding='utf8')
print(str(EX));print(str(WS))
