"""Bounded recorded-Plant equivalence and real Torch Jacobian validation.

The public operation never starts/imports a new original simulator. Its numeric
reference is an immutable sampled_dataset validation partition. A separate
pure component API can execute an explicit reference callable in unit tests.
Neither route converts NumPy/MATLAB execution into an autograd graph.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import math

import numpy as np
import torch

from .artifacts import atomic_json,canonical_hash,read_json
from .contracts import ContractError,MaterialInterface,ModelInterface
from .runtime import Cancelled,OperationContext


MAX_CASE_VALUES=2_000_000
MAX_JACOBIAN_VALUES=1_000_000
MAX_FD_CALLS=50_000
MAX_MODEL_VALUES=10_000_000
_GROUPS={'x':'states','u':'controls','d':'disturbances'}
_KINDS={'smooth','saturation_interior','switch_boundary'}


@dataclass
class ValidationResult:
    report:dict
    arrays:dict


def _number(value,label,positive=False):
    try:valid=type(value) in {int,float} and math.isfinite(value) and value>=0 and (not positive or value>0)
    except (OverflowError,ValueError):valid=False
    if not valid:
        raise ContractError(label+' must be an explicit finite '+('positive' if positive else 'nonnegative')+' number')
    return float(value)


def _array(value,shape,label):
    try:
        raw=np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.shape!=shape or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):raise ValueError()
        result=raw.astype(np.float64)
    except (TypeError,ValueError,OverflowError) as error:raise ContractError(label+' requires exact finite numeric shape '+str(shape)) from error
    if not np.isfinite(result).all():raise ContractError(label+' contains nonfinite values')
    return result


def _bounds(variables):return np.asarray([[v.lower,v.upper] for v in variables],dtype=np.float64).reshape(len(variables),2)


def _domain(raw,interface):
    if raw is not None and (not isinstance(raw,dict) or set(raw)!=set(_GROUPS)):raise ContractError('domain must declare ordered physical x/u/d bounds')
    result={}
    for key,group in _GROUPS.items():
        physical=_bounds(getattr(interface,group))
        if raw is None:value=physical.copy()
        elif not len(physical) and isinstance(raw[key],list) and not raw[key]:value=np.empty((0,2),dtype=np.float64)
        else:value=_array(raw[key],physical.shape,'domain.'+key)
        if np.any(value[:,0]>=value[:,1]) or np.any(value[:,0]<physical[:,0]) or np.any(value[:,1]>physical[:,1]):raise ContractError('Validation domain must stay inside physical bounds: '+key)
        result[key]=value.tolist()
    return result


def _within(value,bounds):
    bounds=np.asarray(bounds,dtype=np.float64).reshape(value.shape[-1],2)
    return bool(np.all(value>=bounds[:,0]-1e-12) and np.all(value<=bounds[:,1]+1e-12))


def resolve_validation_config(config,interface,*,live=False):
    """Validate all declarations/resource sizes before invoking any dynamics."""
    allowed={'schema_version','mode','partition','trajectory_ids','steps','probes','numeric','gradient','domain','proxy_budget','switching_planes','plan_refs'}|({'trajectories'} if live else set())
    if not isinstance(config,dict) or set(config)-allowed or type(config.get('schema_version',1)) is not int or config.get('schema_version',1)!=1:raise ContractError('Explicit supported version-1 differentiable validation config required')
    mode=config.get('mode','exact_rewrite')
    if not isinstance(mode,str) or mode not in {'exact_rewrite','existing_torch','bounded_surrogate'}:raise ContractError('Unsupported differentiable validation mode')
    if config.get('partition','validation')!='validation':raise ContractError('Candidate validation uses only the original validation partition, never train/test selection')
    steps=config.get('steps',50)
    if type(steps) is not int or not 1<=steps<=2048:raise ContractError('steps must be an integer1..2048 for one bounded validation')
    numeric=config.get('numeric',{'atol':1e-6,'rtol':1e-5});gradient=config.get('gradient',{'atol':1e-5,'rtol':1e-3,'finite_difference_steps':[1e-4,1e-5,1e-6]})
    if not isinstance(numeric,dict) or set(numeric)!={'atol','rtol'} or not isinstance(gradient,dict) or set(gradient)!={'atol','rtol','finite_difference_steps'}:raise ContractError('Declare fixed numeric and gradient tolerances before validation')
    numeric={key:_number(numeric[key],'numeric.'+key) for key in ('atol','rtol')}
    hs=gradient['finite_difference_steps']
    if not isinstance(hs,list) or len(hs)!=3:raise ContractError('Use three explicit finite-difference step sizes')
    hs=[_number(h,'finite difference step',True) for h in hs]
    if not hs[0]>hs[1]>hs[2]:raise ContractError('Finite-difference step sizes must be distinct and decreasing')
    gradient={key:_number(gradient[key],'gradient.'+key) for key in ('atol','rtol')}|{'finite_difference_steps':hs}
    probes=config.get('probes')
    if not isinstance(probes,list) or not 1<=len(probes)<=32:raise ContractError('Declare1..32 actual original reference rows for full Jacobian probes')
    normalized=[];seen=set()
    for probe in probes:
        if (not isinstance(probe,dict) or set(probe)-{'row','kind','note'} or type(probe.get('row')) is not int or probe['row']<0
                or probe['row'] in seen or not isinstance(probe.get('kind'),str) or probe['kind'] not in _KINDS):raise ContractError('Each gradient probe requires a unique original row and explicit differentiability kind')
        note=probe.get('note','')
        if not isinstance(note,str) or probe['kind']=='switch_boundary' and not note.strip():raise ContractError('Switch-boundary probes need an explicit physical-switch explanation')
        seen.add(probe['row']);normalized.append({'row':probe['row'],'kind':probe['kind'],'note':note})
    if not any(p['kind']!='switch_boundary' for p in normalized):raise ContractError('Switch boundaries alone cannot establish smooth/saturation gradient acceptance')
    width=interface.nx+interface.nu+interface.nd;targets=interface.nx+2*interface.ny
    if len(probes)*targets*width>MAX_JACOBIAN_VALUES or len(probes)*(6*width+targets+1)>MAX_FD_CALLS:raise ContractError('Full-Jacobian validation exceeds its single-job logical/call budget')
    ids=config.get('trajectory_ids')
    if ids is not None and (not isinstance(ids,list) or not ids or len(ids)>64 or any(type(v) is not int for v in ids) or len(set(ids))!=len(ids)):raise ContractError('trajectory_ids must be explicit distinct validation IDs')
    domain=_domain(config.get('domain'),interface)
    if mode=='bounded_surrogate' and config.get('domain') is None:raise ContractError('A surrogate requires an explicit bounded physical domain')
    budget=config.get('proxy_budget')
    if budget is not None:
        sizes={'one_step_state_rmse':interface.nx,'one_step_output_rmse':interface.ny,'rollout_state_rmse':interface.nx,'rollout_output_rmse':interface.ny}
        if not isinstance(budget,dict) or set(budget)!=set(sizes):raise ContractError('proxy_budget must explicitly declare four per-variable physical RMSE limits')
        budget={key:[_number(v,key) for v in values] for key,values in budget.items() if isinstance(values,list) and len(values)==sizes[key]}
        if set(budget)!=set(sizes):raise ContractError('Proxy budgets must preserve ordered state/output widths')
    planes=config.get('switching_planes',[]);names=set();parsed=[]
    if not isinstance(planes,list) or len(planes)>128:raise ContractError('switching_planes must be an explicit bounded list')
    for plane in planes:
        if (not isinstance(plane,dict) or set(plane)!={'name','axis','index','value','unit'} or not isinstance(plane['name'],str) or not plane['name'].strip() or plane['name'] in names
                or not isinstance(plane['axis'],str) or plane['axis'] not in _GROUPS or type(plane['index']) is not int):raise ContractError('Switch planes require unique name and explicit ordered physical axis/index/value/unit')
        variables=getattr(interface,_GROUPS[plane['axis']]);j=plane['index']
        if not 0<=j<len(variables) or plane['unit']!=variables[j].unit or type(plane['value']) not in {int,float} or not math.isfinite(plane['value']) or not variables[j].lower<=plane['value']<=variables[j].upper:raise ContractError('Switch plane order/units/value conflict with the physical interface')
        names.add(plane['name']);parsed.append(copy.deepcopy(plane))
    if any(p['kind']=='switch_boundary' for p in normalized) and not parsed:raise ContractError('Boundary probes require explicitly disclosed physical switching planes')
    return {'schema_version':1,'mode':mode,'partition':'validation','steps':steps,'trajectory_ids':ids,'probes':normalized,'numeric':numeric,'gradient':gradient,
        'domain':domain,'proxy_budget':budget,'switching_planes':parsed,'reference_surface':'explicit_callable_component' if live else 'recorded_original_sampled_dataset'}


def _case_budget(interface,count,steps):
    if count<1 or count>64 or count*steps*(4*interface.nx+interface.nu+interface.nd+4*interface.ny+4)>MAX_CASE_VALUES:raise ContractError('Validation cases exceed one-job logical values; use smaller batches')


def _reference_cases(arrays,split,interface,resolved):
    chosen=resolved['trajectory_ids'] or list(split['validation'])
    if not set(chosen).issubset(set(split['validation'])):raise ContractError('Only originally reserved validation trajectories may qualify a candidate')
    _case_budget(interface,len(chosen),resolved['steps']);cases=[];selected=[]
    for group in chosen:
        rows=np.flatnonzero(arrays['trajectory_id']==group)
        if len(rows)<resolved['steps']:raise ContractError('Original validation trajectory is shorter than the requested own-rollout horizon')
        # Never silently reorder the original reference material.
        if len(rows)>1 and (not np.all(np.diff(arrays['time'][rows])>0) or not np.allclose(np.diff(arrays['time'][rows]),interface.sample_time,atol=1e-12,rtol=1e-8)
                or not np.allclose(arrays['next_x'][rows[:-1]],arrays['x'][rows[1:]],atol=1e-12,rtol=1e-10)
                or not np.allclose(arrays['next_y'][rows[:-1]],arrays['y'][rows[1:]],atol=1e-12,rtol=1e-10)):
            raise ContractError('Original recorded trajectory time/state/output order is inconsistent')
        rows=rows[:resolved['steps']]
        for key in ('x','next_x','u','d'):
            domain_key='x' if key=='next_x' else key
            if not _within(arrays[key][rows],resolved['domain'][domain_key]):raise ContractError('Reference validation points leave the declared candidate domain: '+key)
        cases.append({'trajectory_id':int(group),'rows':rows});selected.extend(rows.tolist())
    selected=np.asarray(selected,dtype=np.int64)
    if len(selected)<2 or len(np.unique(np.column_stack((arrays['x'][selected],arrays['u'][selected],arrays['d'][selected])),axis=0))<2:raise ContractError('Mixed-batch acceptance requires distinct physical rows, not duplicate copies of one point')
    if any(p['row'] not in set(selected.tolist()) for p in resolved['probes']):raise ContractError('Every Jacobian probe must address an actually selected validation row')
    z=np.column_stack((arrays['x'],arrays['u'],arrays['d']));low=np.concatenate([np.asarray(resolved['domain'][key]).reshape(-1,2)[:,0] for key in _GROUPS]);high=np.concatenate([np.asarray(resolved['domain'][key]).reshape(-1,2)[:,1] for key in _GROUPS])
    offsets={'x':0,'u':interface.nx,'d':interface.nx+interface.nu};h=resolved['gradient']['finite_difference_steps'][0]
    for probe in resolved['probes']:
        point=z[probe['row']]
        if probe['kind']!='switch_boundary' and (np.any(point-h<low) or np.any(point+h>high)):raise ContractError('Smooth/saturation probes require room for every declared central-difference step inside the physical domain')
        nearby=[plane for plane in resolved['switching_planes'] if abs(point[offsets[plane['axis']]+plane['index']]-plane['value'])<=h]
        if probe['kind']=='switch_boundary' and not any(abs(point[offsets[p['axis']]+p['index']]-p['value'])<=1e-12*max(1.,abs(p['value'])) for p in nearby):raise ContractError('Declared switch-boundary probe does not lie on a disclosed physical switching plane')
        if probe['kind']!='switch_boundary' and nearby:raise ContractError('Smooth/saturation probe is too close to a declared switching plane; label the boundary separately')
    return cases,selected


def _checked_tensor(value,shape,label):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float64 or tuple(value.shape)!=shape or not torch.isfinite(value).all():raise ContractError(label+' must return finite CPU float64 Tensor'+str(shape))
    return value


def _call(candidate,x,u,d,interface):
    before=[v.detach().clone() for v in (x,u,d)]
    nxt=_checked_tensor(candidate(x,u,d),(*x.shape[:-1],interface.nx),'transition')
    current=_checked_tensor(candidate.output(x),(*x.shape[:-1],interface.ny),'output')
    after=_checked_tensor(candidate.output(nxt),(*x.shape[:-1],interface.ny),'next output')
    if any(not torch.equal(a,b.detach()) for a,b in zip(before,(x,u,d))):raise ContractError('Candidate mutates its physical input tensors')
    return nxt,current,after


def _numpy_call(candidate,x,u,d,interface):
    with torch.no_grad():values=_call(candidate,*[torch.tensor(v,dtype=torch.float64) for v in (x,u,d)],interface)
    return tuple(v.detach().numpy().copy() for v in values)


def _metrics(predicted,reference,tolerance):
    if not len(predicted):return {'rmse':[],'max_abs_error':[],'within_tolerance':False,'rows':0}
    delta=predicted-reference
    return {'rmse':np.sqrt(np.mean(delta**2,axis=0)).tolist(),'max_abs_error':np.max(np.abs(delta),axis=0).tolist(),
        'within_tolerance':bool(np.all(np.abs(delta)<=tolerance['atol']+tolerance['rtol']*np.abs(reference))),'rows':len(predicted)}


@contextmanager
def _candidate_state(candidate):
    from .python_model import preserve_random_state
    states=candidate.state_dict();total=sum(v.numel() for v in states.values())
    if total>MAX_MODEL_VALUES:raise ContractError('Candidate state exceeds the one-validation tensor snapshot budget')
    snapshot={key:value.detach().clone() for key,value in states.items()};modes=[(module,module.training) for module in candidate.modules()]
    candidate.eval()
    try:
        with preserve_random_state():yield snapshot
    finally:
        for module,training in modes:module.training=training
        current=candidate.state_dict()
        if set(current)==set(snapshot) and any(not torch.equal(value,snapshot[key]) for key,value in current.items()):candidate.load_state_dict(snapshot,strict=True)


def _jacobians(candidate,interface,arrays,resolved,context):
    width=interface.nx+interface.nu+interface.nd;targets=interface.nx+2*interface.ny
    hs=resolved['gradient']['finite_difference_steps'];settings=resolved['gradient'];reports=[];jacs=[];fds=[];present=[];fd_present=[]
    domain=np.vstack([np.asarray(resolved['domain'][key]).reshape(-1,2) for key in _GROUPS])
    for probe in resolved['probes']:
        context.check();row=probe['row'];point=np.concatenate([arrays[key][row] for key in _GROUPS]);z=torch.tensor(point,dtype=torch.float64,requires_grad=True)
        def function(value):return torch.cat(_call(candidate,value[:interface.nx],value[interface.nx:interface.nx+interface.nu],value[interface.nx+interface.nu:],interface))
        record={**probe,'input':point.tolist(),'evaluated':False,'passed':False,'physical_switch_smoothing':'not_performed_by_validator'}
        jac=np.zeros((targets,width));finite=np.zeros((3,targets,width));computed=np.zeros(3,dtype=bool)
        try:
            output=function(z);connected=bool(output.requires_grad)
            if connected:
                for index in range(targets):
                    context.check();gradient=torch.autograd.grad(output[index],z,retain_graph=True,allow_unused=True)[0]
                    if gradient is not None:jac[index]=gradient.detach().numpy()
            base=output.detach().numpy();asym=[]
            for step_index,h in enumerate(hs):
                if np.any(point-h<domain[:,0]) or np.any(point+h>domain[:,1]):asym.append(None);continue
                largest=0.
                for column in range(width):
                    context.check();plus=point.copy();minus=point.copy();plus[column]+=h;minus[column]-=h
                    denominator=plus[column]-minus[column]
                    if not np.isfinite(denominator) or denominator==0:raise ContractError('Finite-difference perturbation is not representable in the declared physical units')
                    with torch.no_grad():a=function(torch.tensor(plus,dtype=torch.float64)).numpy();b=function(torch.tensor(minus,dtype=torch.float64)).numpy()
                    finite[step_index,:,column]=(a-b)/denominator
                    largest=max(largest,float(np.max(np.abs((a-base)/(plus[column]-point[column])-(base-b)/(point[column]-minus[column])))))
                computed[step_index]=True;asym.append(largest)
            if not np.isfinite(jac).all() or not np.isfinite(finite).all():raise ContractError('Jacobian/finite differences are nonfinite')
            errors=[None if not computed[j] else float(np.max(np.abs(jac-finite[j]))) for j in range(3)]
            agreement=[bool(computed[j] and np.allclose(jac,finite[j],atol=settings['atol'],rtol=settings['rtol'])) for j in range(3)]
            threshold=settings['atol']+settings['rtol']*max(float(np.max(np.abs(jac))),float(np.max(np.abs(finite))))
            kink=bool(all(computed) and asym[-1]>threshold and asym[-1]>.2*asym[0])
            physical_zero=bool(probe['kind']=='saturation_interior' and all(computed) and np.max(np.abs(finite))<=settings['atol'])
            passed=bool(all(agreement) and not kink and (connected or physical_zero))
            record.update(evaluated=True,autograd_connected=connected,gradient_max_abs_error_by_h=errors,agreement_by_h=agreement,left_right_derivative_difference_by_h=asym,
                nonsmoothness_detected=kink,physical_zero_saturation_accepted=physical_zero,passed=passed if probe['kind']!='switch_boundary' else None,
                boundary_excluded_from_smooth_acceptance=probe['kind']=='switch_boundary')
        except (Cancelled,TimeoutError):raise
        except (ContractError,RuntimeError,ValueError,FloatingPointError) as error:record['error']={'type':type(error).__name__,'message':str(error)}
        reports.append(record);jacs.append(jac);fds.append(finite);present.append(record.get('autograd_connected',False));fd_present.append(computed)
    eligible=[row for row in reports if row['kind']!='switch_boundary']
    passed=bool(eligible and all(row['passed'] is True for row in eligible))
    return passed,reports,{'autograd_jacobian':np.stack(jacs),'finite_difference_jacobian':np.stack(fds),'autograd_connected':np.asarray(present,dtype=bool),'finite_difference_evaluated':np.stack(fd_present)}


def _validate_recorded(candidate,interface,arrays,cases,selected,resolved,context):
    evidence={};errors=[];numeric={};tolerance=resolved['numeric'];rollouts=[];rollout_rows=[];case_reports=[]
    target={'next_x':arrays['next_x'][selected],'y':arrays['y'][selected],'next_y':arrays['next_y'][selected]}
    with _candidate_state(candidate) as state:
        try:
            single=[_numpy_call(candidate,arrays['x'][row],arrays['u'][row],arrays['d'][row],interface) for row in selected]
            repeated=_numpy_call(candidate,arrays['x'][selected[0]],arrays['u'][selected[0]],arrays['d'][selected[0]],interface)
            if any(not np.array_equal(a,b) for a,b in zip(single[0],repeated)):raise ContractError('Candidate is not repeatable at identical physical state/input')
            batch=_numpy_call(candidate,arrays['x'][selected],arrays['u'][selected],arrays['d'][selected],interface)
            for j,key in enumerate(('next_x','y','next_y')):
                values=np.stack([item[j] for item in single]);numeric['single_'+key]=_metrics(values,target[key],tolerance)
                numeric['mixed_batch_'+key]=_metrics(batch[j],target[key],tolerance);numeric['batch_vs_single_'+key]=_metrics(batch[j],values,{'atol':1e-12,'rtol':0.})
                evidence['single_'+key]=values;evidence['batch_'+key]=batch[j]
                bounds=resolved['domain']['x'] if key=='next_x' else _bounds(interface.outputs)
                if not _within(values,bounds) or not _within(batch[j],bounds):errors.append({'stage':'candidate_point_bounds','message':key+' leaves the declared state/output domain'})
            for case in cases:
                x=arrays['x'][case['rows'][0]].copy();completed=0;reason=None
                for row in case['rows']:
                    context.check()
                    nxt,_,out=_numpy_call(candidate,x,arrays['u'][row],arrays['d'][row],interface)
                    rollouts.append((nxt,out));rollout_rows.append(int(row));completed+=1;x=nxt
                    if not _within(nxt,resolved['domain']['x']) or not _within(out,_bounds(interface.outputs)):
                        reason='candidate_left_declared_state_or_output_domain';break
                case_reports.append({'trajectory_id':case['trajectory_id'],'requested_steps':len(case['rows']),'completed_steps':completed,'completed':completed==len(case['rows']) and reason is None,'failure':reason,
                    'initial_state':arrays['x'][case['rows'][0]].tolist(),'teacher_forcing_after_initial_state':False})
            ids=np.asarray(rollout_rows,dtype=np.int64);states=np.asarray([r[0] for r in rollouts],dtype=np.float64).reshape(-1,interface.nx);outputs=np.asarray([r[1] for r in rollouts],dtype=np.float64).reshape(-1,interface.ny)
            numeric['rollout_state']=_metrics(states,arrays['next_x'][ids],tolerance);numeric['rollout_output']=_metrics(outputs,arrays['next_y'][ids],tolerance)
            evidence.update(rollout_source_row=ids,rollout_state=states,rollout_output=outputs)
        except (Cancelled,TimeoutError):raise
        except (ContractError,RuntimeError,ValueError,FloatingPointError) as error:errors.append({'stage':'numeric_comparison','type':type(error).__name__,'message':str(error)})
        gradient_pass,probes,gradient_arrays=_jacobians(candidate,interface,arrays,resolved,context);evidence.update(gradient_arrays)
        unchanged=all(key in candidate.state_dict() and torch.equal(value,candidate.state_dict()[key]) for key,value in state.items()) and set(state)==set(candidate.state_dict())
        if not unchanged:errors.append({'stage':'candidate_state','message':'Candidate changed a registered tensor/buffer during validation'})
    complete=bool(len(case_reports)==len(cases) and all(c['completed'] for c in case_reports))
    exact=bool(not errors and complete and numeric and all(row['within_tolerance'] for row in numeric.values()))
    proxy_pass=None
    if resolved['mode']=='bounded_surrogate' and resolved['proxy_budget'] is not None:
        mapping={'one_step_state_rmse':'single_next_x','one_step_output_rmse':'single_next_y','rollout_state_rmse':'rollout_state','rollout_output_rmse':'rollout_output'}
        proxy_pass=bool(not errors and complete and all(key in numeric and len(numeric[key]['rmse'])==len(resolved['proxy_budget'][limit]) and np.all(np.asarray(numeric[key]['rmse'])<=resolved['proxy_budget'][limit]) for limit,key in mapping.items())
            and all(row['within_tolerance'] for key,row in numeric.items() if key.startswith('batch_vs_single_')))
    qualified=bool(gradient_pass and (proxy_pass is True if resolved['mode']=='bounded_surrogate' else exact))
    evidence.update(reference_source_row=selected,reference_x=arrays['x'][selected],reference_u=arrays['u'][selected],reference_d=arrays['d'][selected],reference_next_x=target['next_x'],reference_y=target['y'],reference_next_y=target['next_y'])
    report={'schema_version':1,'algorithm':'recorded_reference_and_full_jacobian_validation','mode':resolved['mode'],'qualified_for_pgac':qualified,'exact_equivalence_pass':exact,
        'bounded_proxy_acceptance':proxy_pass,'gradient_checks_passed':gradient_pass,'numeric':numeric,'trajectories':case_reports,'gradient_probes':probes,'errors':errors,'resolved_config':resolved,
        'physical_interface':interface.to_dict(),'gradient_semantics':'surrogate_gradient_not_physical_true_gradient' if resolved['mode']=='bounded_surrogate' else 'autograd_of_declared_candidate_verified_at_recorded_probes',
        'coverage':{'domain':resolved['domain'],'coverage_kind':'declared_bounded_box_and_recorded_validation_points','joint_coverage_proved':False,'extrapolation_validated':False,
            'reference_rows':len(selected),'trajectory_ids':[c['trajectory_id'] for c in cases],'sample_time':interface.sample_time},
        'reference_partition':'validation','test_used_for_selection':False,'original_plant_closed_loop_evaluated':False,'source_code_rewritten_or_switches_smoothed':False,
        'resource_boundary':{'max_case_values':MAX_CASE_VALUES,'max_jacobian_values':MAX_JACOBIAN_VALUES,'max_fd_calls':MAX_FD_CALLS,'max_model_snapshot_values':MAX_MODEL_VALUES,'scope':'one validation batch, not an overall engineering goal limit'}}
    return ValidationResult(report,evidence)


def _proxy_coverage(workspace,candidate_id,resolved,interface):
    if resolved['mode']!='bounded_surrogate':return None
    root,meta=workspace.resolve(candidate_id)
    if meta['kind']!='learned_model' or 'report.json' not in meta['files']:raise ContractError('This bounded-surrogate route requires an existing learned_model with bound training-only coverage')
    report=read_json(root/'report.json');coverage=report.get('coverage')
    if not isinstance(coverage,dict) or coverage.get('source')!='training_partition_only' or not isinstance(coverage.get('marginal_ranges'),list):raise ContractError('A bounded surrogate requires bound candidate training-only marginal coverage')
    for key,group in _GROUPS.items():
        for index,variable in enumerate(getattr(interface,group)):
            rows=[row for row in coverage['marginal_ranges'] if row.get('group')==group and row.get('name')==variable.name and row.get('unit')==variable.unit]
            if len(rows)!=1:raise ContractError('Surrogate training coverage order/name/units incomplete')
            lo,hi=rows[0].get('observed_min'),rows[0].get('observed_max')
            if type(lo) not in {int,float} or type(hi) not in {int,float} or not math.isfinite(lo) or not math.isfinite(hi) or not lo<=resolved['domain'][key][index][0]<resolved['domain'][key][index][1]<=hi:raise ContractError('Surrogate validation domain leaves observed training marginal ranges')
    return {'candidate':meta['id'],'report_sha256':meta['files']['report.json'],'training_coverage':coverage,'joint_coverage_proved':False}


def validate_differentiable_model(workspace,original_id,candidate_id,config,context=None,*,reference_dataset_id):
    """Publish an inactive validation receipt for existing candidate/source IDs."""
    from .modeling import load_model
    from .predictive import _read_sampled
    context=context or OperationContext();original_root,original=workspace.resolve(original_id)
    candidate_root,candidate_meta=workspace.resolve(candidate_id);interface=ModelInterface.parse(original['interface']);interface.compatible(ModelInterface.parse(candidate_meta['interface']))
    resolved=resolve_validation_config(config,interface)
    arrays,_,split,recorded_interface,dataset=_read_sampled(workspace,reference_dataset_id);interface.compatible(recorded_interface)
    if original_id not in {p['id'] for p in dataset['parents']} or dataset.get('source',{}).get('kind')!='plant_simulator' or dataset['source'].get('model')!=original_id or dataset['source'].get('measured') is not False:
        raise ContractError('Reference dataset must bind the exact original simulator artifact; user-supplied numeric JSON is not original-Plant evidence')
    cases,selected=_reference_cases(arrays,split,interface,resolved);training_coverage=_proxy_coverage(workspace,candidate_id,resolved,interface)
    context.check();candidate=load_model(workspace,candidate_id);candidate.interface.compatible(interface)
    result=_validate_recorded(candidate,interface,arrays,cases,selected,resolved,context)
    bindings={key:{'id':meta['id'],'metadata_hash':workspace.manifest['artifacts'][meta['id']]['metadata_hash']} for key,meta in [('original',original),('candidate',candidate_meta),('dataset',dataset)]}
    result.report.update(bindings=bindings,reference_backend=dataset['source'].get('backend'),original_execution_surface='immutable_recorded_original_validation_trajectories_no_new_simulator_execution',
        original_model_source=original.get('source',{}),proxy_training_coverage=training_coverage,confirmation_sources=context.case_confirmation_sources)
    parents=list(dict.fromkeys([original_id,candidate_id,reference_dataset_id]));context.check()
    with workspace.stage('differentiable_validation',MaterialInterface('bundle','Bounded differentiable model validation'),parents,config,compatible_parents=False) as stage:
        atomic_json(stage.root/'report.json',result.report);np.savez_compressed(stage.root/'validation_arrays.npz',**result.arrays)
        atomic_json(stage.root/'bindings.json',bindings)
        (stage.root/'README.md').write_text('Recorded original-Plant validation only. Candidate own-rollout uses actual original inputs and its own predicted state after initialization. Full per-output Jacobians are compared at three physical step sizes; switch boundaries are disclosed separately. No original simulator/MATLAB execution, model fitting, switch smoothing, controller optimization or physical closed-loop/stability proof is performed. Proxy gradients remain proxy gradients; domain bounds are marginal boxes and do not prove joint coverage.\n',encoding='utf-8')
        return stage.publish({'status':'verified' if result.report['qualified_for_pgac'] else 'rejected','qualified_for_pgac':result.report['qualified_for_pgac'],
            'exact_equivalence_pass':result.report['exact_equivalence_pass'],'bounded_proxy_acceptance':result.report['bounded_proxy_acceptance'],'gradient_checks_passed':result.report['gradient_checks_passed']},
            source={'kind':'bounded_differentiable_validation','bindings':bindings,'mode':resolved['mode'],'coverage':result.report['coverage'],'reference_partition':'validation'},activate=False)


def read_differentiable_validation(workspace,validation_id,*,candidate_id=None,original_id=None,require_qualified=True):
    root,metadata=workspace.resolve(validation_id,kind='differentiable_validation',allow_rejected=not require_qualified);report=read_json(root/'report.json');bindings=read_json(root/'bindings.json')
    if report.get('bindings')!=bindings or metadata.get('source',{}).get('bindings')!=bindings or set(bindings)!={'original','candidate','dataset'} or {row['id'] for row in metadata['parents']}!={row['id'] for row in bindings.values()}:raise ContractError('Validation receipt binding differs from its exact parents')
    for row in bindings.values():
        _,meta=workspace.resolve(row['id'])
        if workspace.manifest['artifacts'][row['id']]['metadata_hash']!=row['metadata_hash']:raise ContractError('Validated source/candidate version changed')
    if candidate_id is not None and candidate_id!=bindings['candidate']['id'] or original_id is not None and original_id!=bindings['original']['id']:raise ContractError('Validation receipt is for a different candidate/original model')
    if require_qualified and (report.get('qualified_for_pgac') is not True or metadata['validation'].get('qualified_for_pgac') is not True):raise ContractError('Model validation did not qualify this candidate for PGAC')
    return report


def validate_pair(original,candidate,config,context=None):
    """Component-only explicit callable reference; no artifact/Host/MATLAB API.

    Original must expose interface, __call__(x,u,d), output(x). NumPy
    callables receive NumPy values; Torch modules receive float64 tensors.
    config.trajectories contains {id,initial_state,u[T,nu],d[T,nd]}.
    """
    context=context or OperationContext();interface=candidate.interface;interface.compatible(original.interface)
    resolved=resolve_validation_config(config,interface,live=True);trajectories=config.get('trajectories')
    if not isinstance(trajectories,list) or not trajectories:raise ContractError('Component reference requires explicit actual trajectories')
    _case_budget(interface,len(trajectories),resolved['steps']);prepared=[];ids=set()
    for item in trajectories:
        if not isinstance(item,dict) or set(item)!={'id','initial_state','u','d'} or type(item['id']) is not int or item['id'] in ids:raise ContractError('Explicit unique component trajectory IDs/initial_state/u/d required')
        ids.add(item['id']);x=_array(item['initial_state'],(interface.nx,),'initial_state');u=_array(item['u'],(resolved['steps'],interface.nu),'controls');d=_array(item['d'],(resolved['steps'],interface.nd),'disturbances')
        if not all(_within(value,resolved['domain'][key]) for key,value in [('x',x),('u',u),('d',d)]):raise ContractError('Reference declaration is outside validation domain')
        prepared.append((item['id'],x,u,d))
    values={key:[] for key in ('x','u','d','next_x','y','next_y','trajectory_id','time')}
    def invoke(fn,*args):
        inputs=[torch.tensor(value,dtype=torch.float64) for value in args] if isinstance(original,torch.nn.Module) else [value.copy() for value in args]
        with torch.no_grad():value=fn(*inputs)
        return value.detach().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value)
    for identifier,x,us,ds in prepared:
        for step,(u,d) in enumerate(zip(us,ds)):
            context.check();y=_array(invoke(original.output,x),(interface.ny,),'original output');nxt=_array(invoke(original,x,u,d),(interface.nx,),'original next state');next_y=_array(invoke(original.output,nxt),(interface.ny,),'original next output')
            for key,value in [('x',x),('u',u),('d',d),('next_x',nxt),('y',y),('next_y',next_y),('trajectory_id',identifier),('time',step*interface.sample_time)]:values[key].append(copy.deepcopy(value))
            x=nxt
    arrays={key:np.asarray(value,dtype=np.int64 if key=='trajectory_id' else np.float64) for key,value in values.items()}
    cases,selected=_reference_cases(arrays,{'validation':list(ids)},interface,resolved)
    result=_validate_recorded(candidate,interface,arrays,cases,selected,resolved,context);result.report['reference_source']='explicit_component_reference_callable'
    return result
