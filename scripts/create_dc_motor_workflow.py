"""Build replay recipe from checked case inputs, without executing numerical stages."""
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
p=root/'examples/dc-motor'
load=lambda name:json.loads((p/name).read_text(encoding='utf8'))
ref=lambda stage:{'stage':stage}
steps=[
    dict(id='nominal',operation='register_model',inputs=dict(model_path='nominal.json'),config=dict(source=dict(kind='user_nominal',description='Declared DC motor continuous nominal equations; exact ZOH at 0.1 s'))),
    dict(id='reference',operation='register_model',inputs=dict(model_path='reference.json'),config=dict(source=dict(kind='reference_simulator',description='Separate mathematical DC motor reference; not measured data'))),
    dict(id='data',operation='simulate_transitions',inputs=dict(nominal_model=ref('nominal'),reference_model=ref('reference')),config=load('data.json')),
    dict(id='corrected',operation='train_residual',inputs=dict(nominal_model=ref('nominal'),dataset=ref('data')),config=load('residual.json')),
    dict(id='controller',operation='design_controller',inputs=dict(model=ref('corrected'),evaluation_model=ref('reference')),config=load('controller.json')),
    dict(id='expert',operation='collect_expert',inputs=dict(controller=ref('controller'),evaluation_model=ref('reference')),config=load('expert.json')),
    dict(id='imitation',operation='train_imitation',inputs=dict(dataset=ref('expert')),config=load('imitation.json')),
    dict(id='optimized',operation='optimize_policy',inputs=dict(policy=ref('imitation'),model=ref('corrected')),config=load('optimize.json')),
]
(p/'workflow.json').write_text(json.dumps(dict(schema_version=1,name='DC motor speed: residual model, tuned MPC and lightweight policy',steps=steps),ensure_ascii=True,indent=2),encoding='utf8')
print('Wrote DC motor replay workflow (not executed).')
