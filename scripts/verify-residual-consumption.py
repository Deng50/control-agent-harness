"""N05 actual MPC/policy consumers; explicit synthetic weight interventions.

No load_model replacement or proxy loss is used. Every policy probe calls the
existing optimize_policy, including its real backward, clipping and Adam step.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid
from unittest.mock import patch

import numpy as np
import torch

from control_harness.artifacts import Workspace,atomic_json,read_json,file_hash
from control_harness.contracts import ContractError
from control_harness.control import MPCPolicy
from control_harness.data import simulate_transitions
from control_harness.modeling import DeclarativeDynamics,ResidualNetwork,CorrectedDynamics,load_model,register_model,tensor
from control_harness.policy import collect_expert,train_imitation,load_policy,optimize_policy,NeuralPolicy
from control_harness.residual import train_residual

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'test-output/v2-development/20261117T002006/fixtures'
STEPS=[1e-4,1e-5,1e-6]


def residual(nominal,bias):
    i=nominal.interface
    config={'hidden_dims':[6],'residual_semantics':'next_state_residual_in_state_units'}
    scalers={'input_mean':[0.]*(i.nx+i.nu+i.nd),'input_scale':[1.]*(i.nx+i.nu+i.nd),'residual_scale':[1.]*i.nx}
    net=ResidualNetwork(i,config,scalers)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net.network.net[-1].bias.copy_(tensor(bias))
    return net,config,scalers


def mpc_probe(spec):
    nominal=DeclarativeDynamics(spec);i=nominal.interface
    bias=[.02]*i.nx
    models=[CorrectedDynamics(copy.deepcopy(nominal),residual(copy.deepcopy(nominal),b)[0]) for b in ([0.]*i.nx,bias)]
    x=[.12,-.04] if i.nx==2 else [.12,-.04,.06,-.02]
    reference=[0.]*i.ny;d=[.01]*i.nd;previous=[0.]*i.nu
    seq=tensor([[.03]*i.nu]*4)
    records=[]
    for model in models:
        policy=MPCPolicy(model,{'horizon':4,'q':1.,'r':.03,'s':.005,'solver_tolerance':1e-12,'max_iterations':100})
        z=seq.clone().requires_grad_(True)
        value,constraints=policy.rollout(z,tensor(x),tensor(reference),tensor(d),tensor(previous))
        grad,=torch.autograd.grad(value,z)
        coordinate=int(torch.argmax(grad.abs()).item());flat=seq.flatten().clone()
        fd=[]
        for h in STEPS:
            plus=flat.clone();minus=flat.clone();plus[coordinate]+=h;minus[coordinate]-=h
            a=policy.rollout(plus.reshape_as(seq),tensor(x),tensor(reference),tensor(d),tensor(previous))[0].item()
            b=policy.rollout(minus.reshape_as(seq),tensor(x),tensor(reference),tensor(d),tensor(previous))[0].item()
            fd.append({'h':h,'value':(a-b)/(2*h),'autograd':grad.flatten()[coordinate].item()})
        action,diagnostics=policy.act(x,reference,d,previous)
        margin=np.minimum(action-policy.lo,policy.hi-action)
        assert np.min(margin)>.01,'Intervention MPC action must stay away from saturation'
        assert np.isfinite(action).all() and torch.isfinite(grad).all() and constraints.min()>0
        records.append({'prediction':model(tensor(x),tensor(previous),tensor(d)).detach().tolist(),'fixed_sequence_cost':value.item(),
            'sequence_gradient':grad.detach().tolist(),'finite_difference':fd,'action':action.tolist(),'action_bound_margin':margin.tolist(),'solve':diagnostics})
    observed=np.array(records[1]['prediction'])-records[0]['prediction']
    np.testing.assert_allclose(observed,bias,rtol=0,atol=1e-15)
    assert abs(records[1]['fixed_sequence_cost']-records[0]['fixed_sequence_cost'])>1e-7
    assert np.max(np.abs(np.array(records[1]['action'])-records[0]['action']))>1e-4
    for row in records:
        for fd in row['finite_difference']:np.testing.assert_allclose(fd['value'],fd['autograd'],atol=1e-8,rtol=1e-4)
    return {'nominal_spec_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),'sample_time':i.sample_time,
        'residual_semantics':'next_state_residual_in_state_units','intervention_bias':bias,'observed_prediction_delta':observed.tolist(),
        'extra_dt_applied':False,'consumers':'existing MPCPolicy.rollout and SciPy-SLSQP MPCPolicy.act','variants':records}


def publish_corrected(ws,nominal_id,bias):
    nominal=load_model(ws,nominal_id);net,config,scalers=residual(nominal,bias)
    with ws.stage('corrected_model',nominal.interface,[nominal_id],{'test_fixture':'constant residual intervention'}) as stage:
        atomic_json(stage.root/'nominal.json',nominal.spec);atomic_json(stage.root/'network.json',config);atomic_json(stage.root/'scalers.json',scalers)
        torch.save({'network_state':net.state_dict()},stage.root/'model.pt')
        return stage.publish({'status':'verified','checks':['explicit_synthetic_weight_intervention']},source={
            'nominal_model':nominal_id,'algorithm':'test_fixture_constant_residual_intervention','residual_semantics':config['residual_semantics'],
            'intervention_bias':bias,'learned_improvement_claimed':False},activate=False)


def publish_actor(ws,template_id,actor,tag):
    root,meta=ws.resolve(template_id);i=actor.interface
    with ws.stage('imitation_policy',i,[template_id],{'test_fixture':tag}) as stage:
        for name in ('network.json','policy.json'):atomic_json(stage.root/name,read_json(root/name))
        atomic_json(stage.root/'scalers.json',{key:getattr(actor,key).tolist() for key in ('input_mean','input_scale','output_mean','output_scale')})
        torch.save({'policy_state':actor.state_dict()},stage.root/'policy.pt')
        return stage.publish({'status':'verified','checks':['explicit_actor_parameter_fixture']},source={
            'algorithm':'test_fixture_actor_parameter_intervention','template':template_id,'learned_improvement_claimed':False},activate=False)


def policy_fixture(path,spec,bias):
    ws=Workspace(path);nominal=register_model(ws,spec,{'kind':'user_nominal','description':'Frozen synthetic N05 nominal'})
    reference=register_model(ws,spec,{'kind':'reference_simulator','description':'Same frozen nominal simulator for both interventions'})
    model_id=publish_corrected(ws,nominal,bias);model=load_model(ws,model_id);i=model.interface
    params={'horizon':1,'q':100.,'r':1e-5,'s':0.,'solver_tolerance':1e-10}
    def scenario(k):
        sign=-1 if k%2 else 1
        x=[sign*(.70+.02*k),0.] if i.nx==2 else [sign*(.75+.02*k)]*i.nx
        return {'name':f'explicit-{k}','initial_state':x,'reference':[0.]*i.ny,'disturbance':[.01]*i.nd,'steps':1}
    config={'parameters':params,'verification_scenarios':[scenario(10)],'test_scenarios':[scenario(11)]}
    with ws.stage('controller',i,[model_id,reference],config) as stage:
        atomic_json(stage.root/'controller.json',{'prediction_model':model_id,'evaluation_model':reference,'parameters':params,
            'solver':'scipy_slsqp_with_torch_derivatives','test_fixture':'Explicit controller parameters; no tuning/improvement claim'})
        controller=stage.publish({'status':'verified','checks':['explicit_test_fixture_controller']},activate=False)
    expert=collect_expert(ws,controller,reference,{'scenarios':[scenario(k) for k in range(6)],'seed':37})
    trained=train_imitation(ws,expert,{'epochs':1,'hidden_dims':[5],'batch_size':2,'seed':83,'learning_rate':.001,'patience':3})
    actor=load_policy(ws,trained)
    with torch.no_grad():
        for parameter in actor.parameters():parameter.zero_()
        actor.input_mean.zero_();actor.input_scale.fill_(1.);actor.output_mean.zero_();actor.output_scale.fill_(1.)
    controlled=publish_actor(ws,trained,actor,'same zero actor and canonical 0/1 scaler fixture after actual 1-epoch imitation; not a trained-statistics claim')
    expert_root,_=ws.resolve(expert)
    with np.load(expert_root/'expert.npz',allow_pickle=False) as a:expert_arrays={key:a[key].copy() for key in a.files}
    return ws,model_id,controlled,expert_arrays


def optimize_capture(ws,model_id,actor_id):
    capture={'losses':[],'raw_gradients':[],'optimizer_gradients':[],'actions':[],'actual_adam_steps':0}
    original_backward=torch.Tensor.backward;original_clip=torch.nn.utils.clip_grad_norm_;original_step=torch.optim.Adam.step;original_forward=NeuralPolicy.forward
    def backward(value,*args,**kwargs):
        capture['losses'].append(float(value.detach()));return original_backward(value,*args,**kwargs)
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters)
        capture['raw_gradients'].append(torch.cat([p.grad.detach().flatten() for p in parameters]).numpy().copy())
        return original_clip(parameters,*args,**kwargs)
    def step(optimizer,*args,**kwargs):
        capture['optimizer_gradients'].append(torch.cat([p.grad.detach().flatten() for group in optimizer.param_groups for p in group['params']]).numpy().copy())
        capture['actual_adam_steps']+=1
        return original_step(optimizer,*args,**kwargs)
    def forward(policy,*args,**kwargs):
        result=original_forward(policy,*args,**kwargs)
        if result.requires_grad:capture['actions'].append(result.detach().numpy().copy())
        return result
    config={'epochs':1,'horizon':3,'batch_size':1,'seed':43,'learning_rate':1e-5,'imitation_weight':0.,'constraint_weight':0.}
    with patch.object(torch.Tensor,'backward',backward),patch.object(torch.nn.utils,'clip_grad_norm_',clip),patch.object(torch.optim.Adam,'step',step),patch.object(NeuralPolicy,'forward',forward):
        artifact=optimize_policy(ws,actor_id,model_id,config)
    root=ws.path/ws.manifest['artifacts'][artifact]['path'];report=read_json(root/'report.json')
    assert report['corrected_dynamics_used']==model_id and report['algorithm']=='differentiable_physical_rollout_policy_gradient'
    assert capture['actual_adam_steps']==report['optimizer_updates']>0
    i=load_model(ws,model_id).interface;low=np.array([v.lower for v in i.controls]);high=np.array([v.upper for v in i.controls])
    margin=min(float(np.minimum(a-low,high-a).min()) for a in capture['actions'])
    assert margin>1e-5,'Actor rollout must remain away from amplitude saturation'
    return {'loss':capture['losses'][0],'gradient':capture['raw_gradients'][0].tolist(),
        'optimizer_gradient':capture['optimizer_gradients'][0].tolist(),'actual_adam_steps':capture['actual_adam_steps'],
        'action_margin':margin,'artifact_id':artifact,'model_id':model_id,'actor_id':actor_id}


def policy_probe(path,spec):
    i=DeclarativeDynamics(spec).interface
    variants=[policy_fixture(path/f'v{k}',spec,bias) for k,bias in enumerate(([0.]*i.nx,[.02]*i.nx))]
    a,b=variants
    assert a[3].keys()==b[3].keys()
    for key in a[3]:
        if key!='u':np.testing.assert_array_equal(a[3][key],b[3][key],err_msg=f'expert {key} changed; causal comparison would be confounded')
    action_roundoff=float(np.max(np.abs(a[3]['u']-b[3]['u'])))
    assert action_roundoff<1e-12,'Expert targets differ beyond solver roundoff; imitation_weight=0 excludes targets from the tested gradient'
    actors=[load_policy(ws,actor_id) for ws,_,actor_id,_ in variants]
    for key,value in actors[0].state_dict().items():assert torch.equal(value,actors[1].state_dict()[key]),'Controlled actors differ'
    records=[optimize_capture(ws,model,actor) for ws,model,actor,_ in variants]
    g0,g1=np.asarray(records[0]['gradient']),np.asarray(records[1]['gradient'])
    assert np.isfinite(g0).all() and np.isfinite(g1).all() and max(np.linalg.norm(g0),np.linalg.norm(g1))>1e-8
    assert np.max(np.abs(g1-g0))>1e-8 and abs(records[1]['loss']-records[0]['loss'])>1e-7
    coordinate=int(np.argmax(np.maximum(np.abs(g0),np.abs(g1))))
    for index,(ws,model_id,actor_id,_) in enumerate(variants):
        actor=load_policy(ws,actor_id);base_hashes={name:file_hash(root/name) for root,meta in [ws.resolve(model_id)] for name in meta['files']}
        fd=[]
        for h in STEPS:
            losses=[]
            for sign in (1,-1):
                candidate=copy.deepcopy(actor);offset=coordinate
                with torch.no_grad():
                    for parameter in candidate.parameters():
                        if offset<parameter.numel():parameter.flatten()[offset]+=sign*h;break
                        offset-=parameter.numel()
                perturbed=publish_actor(ws,actor_id,candidate,f'directional parameter FD h={h} sign={sign}')
                losses.append(optimize_capture(ws,model_id,perturbed)['loss'])
            value=(losses[0]-losses[1])/(2*h);expected=records[index]['gradient'][coordinate]
            np.testing.assert_allclose(value,expected,rtol=1e-3,atol=1e-5)
            fd.append({'h':h,'value':value,'autograd':expected})
        root,meta=ws.resolve(model_id)
        assert all(file_hash(root/name)==value for name,value in base_hashes.items()),'Published model bytes were edited'
        records[index]['finite_difference']=fd
    return {'same_expert_inputs':True,'expert_action_roundoff_max':action_roundoff,'expert_action_loss_weight':0.,'same_actor_state_and_scalers':True,'nominal_unchanged':True,
        'intervention':'separately published inactive genuine CorrectedDynamics residual state_dict bias 0 vs 0.02; no load_model mock',
        'consumer':'existing optimize_policy actual inline rollout/backward/clipping/Adam; instrumentation calls original implementations',
        'first_gradient_max_change':float(np.max(np.abs(g1-g0))),'variants':records}


def legacy_probe(path,spec,reference):
    ws=Workspace(path);nominal=register_model(ws,spec,{'kind':'user_nominal','description':'Frozen F02 legacy route'})
    actual=register_model(ws,reference,{'kind':'reference_simulator','description':'Frozen separate F02 reference'})
    dataset=simulate_transitions(ws,nominal,actual,{'trajectories':6,'steps':6,'seed':20261119})
    corrected=train_residual(ws,nominal,dataset,{'epochs':4,'hidden_dims':[8],'batch_size':8,'learning_rate':.005,'seed':67,'physics_weight':0.,'patience':10})
    row=ws.manifest['artifacts'][corrected];root=ws.path/row['path'];report=read_json(root/'report.json')
    ckpt=torch.load(root/'latest.pt',weights_only=True)
    assert ckpt['epoch']==4 and report['algorithm']=='supervised_discrete_residual_with_declared_physical_constraints'
    assert report['residual_semantics']=='next_state_residual_in_state_units'
    try:train_residual(ws,nominal,dataset,{'residual_semantics':'derivative_residual'})
    except ContractError:rejected=True
    else:raise AssertionError('A derivative residual was silently accepted as a next-state residual')
    return {'nominal_id':nominal,'reference_id':actual,'dataset_id':dataset,'corrected_id':corrected,
        'trained_epochs':ckpt['epoch'],'optimizer_updates':ckpt['updates'],'artifact_status':row['status'],
        'domain_acceptance':report['acceptance'],'execution_completed':True,'algorithm_name_unchanged':report['algorithm'],
        'semantics':report['residual_semantics'],'derivative_residual_rejected':rejected,'improvement_claimed':False}


def run_evidence(output,fixtures=FIXTURES):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('F*/v*/manifest.json')):raise ContractError('Use a fresh N05 evidence directory; do not mix intervention runs')
    expected={'F01_spring_2x1.json':'22525d4196bda85f3e3ddba0de0493859a91f515fe301f99cd54f40a84e55824',
        'F02_existing_linear_4x2.json':'ac6364b26915a81abc790a88c6346e2fe91fdd62950898ecac0cc0c67f6ae7d9'}
    for name,digest in expected.items():assert file_hash(Path(fixtures)/name)==digest,'Frozen engineering fixture changed'
    f1=read_json(Path(fixtures)/'F01_spring_2x1.json');f2=read_json(Path(fixtures)/'F02_existing_linear_4x2.json')
    threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        results={name:{'mpc':mpc_probe(spec),'policy':policy_probe(output/name,spec)} for name,spec in [('F01',f1['source_model']),('F02',f2['nominal'])]}
        legacy=legacy_probe(output/'legacy',f2['nominal'],f2['reference'])
    finally:torch.set_num_threads(threads)
    record={'schema_version':1,'stage':'S06','group':'N05','status':'PASS','scope':'Synthetic causal consumption interventions, actual numerical consumers; no user performance-improvement claim',
        'fixture_provenance':'Frozen F01/F02; F01 continuous nominal is explicitly ZOH-integrated before a state-unit next-state residual is added.',
        'source':'ResidualNetwork reuses AIagent PerceptionNet; CorrectedDynamics adds nominal(x,u,d)+residual(x,u,d) without extra dt.',
        'finite_difference_steps':STEPS,'results':results,'legacy_route':legacy,'no_product_files_edited':True,'no_model_loader_mock':True,
        'fixture_sha256':expected,'source_sha256':{name:file_hash(ROOT/name) for name in ('src/control_harness/modeling.py','src/control_harness/control.py','src/control_harness/policy.py','src/control_harness/residual.py','src/control_harness/reused/perception.py','scripts/verify-residual-consumption.py')},
        'limits':['Finite visited smooth non-saturated fixtures; not a proof for all plants or saturation points.','Expert collection uses saturated one-step actions: loss inputs are identical; <=1e-12 target solver roundoff is recorded and excluded by imitation_weight=0. Tested MPC and actor rollout consumption probes separately require non-saturated actions.','Weight interventions and controlled actor/canonical-scaler initialization are explicit test_fixture artifacts, not trained-statistics or learned-improvement evidence.']}
    atomic_json(output/'residual-consumption-result.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--fixtures',default=str(FIXTURES));args=p.parse_args()
    result=run_evidence(args.output,args.fixtures)
    print(json.dumps({'status':result['status'],'evidence':str(Path(args.output)/'residual-consumption-result.json')}))
