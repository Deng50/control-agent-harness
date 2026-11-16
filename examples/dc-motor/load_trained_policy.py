"""Read-only inference on an existing DC motor workspace; never retrains."""
import argparse,json
from pathlib import Path
import torch
from control_harness.artifacts import Workspace
from control_harness.policy import load_policy

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path)
    p.add_argument('--current',type=float,default=0);p.add_argument('--speed',type=float,default=0)
    p.add_argument('--target',type=float,default=12);p.add_argument('--load',type=float,default=.02)
    a=p.parse_args();torch.set_num_threads(1)
    w=Workspace(a.workspace.resolve());m=json.loads((w.path/'manifest.json').read_text(encoding='utf8'))
    policies=[i for i,v in m['artifacts'].items() if v['kind']=='optimized_policy' and v['status']=='verified']
    if len(policies)!=1:raise ValueError('Select a workspace with exactly one verified optimized policy')
    policy=load_policy(w,policies[0]);u,diagnostic=policy.act([a.current,a.speed],[a.target],[a.load])
    print(json.dumps(dict(policy_id=policies[0],input=dict(current_A=a.current,speed_rad_s=a.speed,target_rad_s=a.target,load_Nm=a.load),voltage_V=u.tolist(),diagnostics=diagnostic),indent=2))
