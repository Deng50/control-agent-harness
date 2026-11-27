"""One validation-selected PGAC candidate, independently tested on its Plant.

The original source is resolved through the immutable S10 baseline ancestry.
Validation fixes a version before any test rollout. Test may reject that one
candidate and retain an already feasible baseline; it never selects an epoch.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch
import threadpoolctl

from .artifacts import atomic_json,read_json,file_hash
from .contracts import ContractError,DataInterface,canonical_hash
from .runtime import OperationContext,Cancelled
from . import external_policy_evaluation as ep
from .external_expert import _collect_python,ExpertCollectionStopped,ExpertCollectionCancelled
from .external_policy import load_external_policy
from .policy_export import export_external_policy_state
from .pgac_scenarios import ensure_new_physical_conditions, scenario_series

NAMES=('expert','imitation','candidate')


class _CoveragePlant:
    """Guard the actual original adapter's every physical step, without a proxy."""
    def __init__(self,plant,domain):
        from .matlab_expert import validate_coverage_domain
        self.plant=plant;self.interface=plant.interface;self.domain=validate_coverage_domain(domain,self.interface)
        self.guard={'schema_version':1,'enabled':True,'domain_hash':canonical_hash(self.domain),'before_checks':0,'after_checks':0,'passed_subticks':0,'violations':0,'status':'passed'}
    def __getattr__(self,name):return getattr(self.plant,name)
    def _inside(self,key,value):
        raw=np.asarray(value,dtype=np.float64);bounds=np.asarray(self.domain[key],dtype=np.float64).reshape(-1,2)
        return raw.shape==(len(bounds),) and np.isfinite(raw).all() and np.all(raw>=bounds[:,0]) and np.all(raw<=bounds[:,1])
    def _failed(self,phase,key,value):
        raw=np.asarray(value,dtype=np.float64);bounds=np.asarray(self.domain[key],dtype=np.float64).reshape(-1,2)
        details={'phase':phase,'signal_group':key}
        if raw.shape==(len(bounds),) and np.isfinite(raw).all():
            details.update(value=raw.tolist(),max_excess=float(np.maximum(bounds[:,0]-raw,raw-bounds[:,1]).max()))
        self.guard.update(violations=1,status='failed',first_observed_violation=details)
        raise ContractError('Original Plant leaves qualified x/u/d coverage '+phase+' an actual physical subtick; baseline is not replaced')
    def step(self,u,d=None):
        d=np.empty(0) if d is None and not self.interface.nd else d;self.guard['before_checks']+=1
        for key,value in [('x',self.plant.state),('u',u),('d',d)]:
            if not self._inside(key,value):self._failed('before',key,value)
        value=self.plant.step(u,d);self.guard['after_checks']+=1
        if not self._inside('x',value):self._failed('after','x',value)
        self.guard['passed_subticks']+=1;return value
    def describe(self):return {**self.plant.describe(),'coverage_guard':copy.deepcopy(self.guard),'coverage_scope':'before x/u/d and after x for every actual original Plant dt step'}


def evaluation_config(value):
    gates = {'min_relative_improvement', 'max_worst_case_relative_regression'}
    if not isinstance(value,dict) or set(value)-gates-{'q','r','s','comparison_atol'} or not {'q','r','s'}<=value.keys():raise ContractError('Original-Plant evaluation requires explicit normalized quadratic q/r/s')
    result={**value,'comparison_atol':value.get('comparison_atol',1e-12)}
    if any(type(v) not in {int,float} or not math.isfinite(v) for v in result.values()) or result['q']<=0 or result['r']<0 or result['s']<0 or result['comparison_atol']!=1e-12:
        raise ContractError('q must be positive, r/s nonnegative, and frozen comparison_atol must be 1e-12')
    if any(result[key] < 0 for key in gates & result.keys()) or result.get('min_relative_improvement', 0) > 1:
        raise ContractError('Relative acceptance thresholds must be nonnegative fractions, with minimum improvement at most 1')
    return {key:float(number) for key,number in result.items()}


def _selection(workspace,candidate_id):
    root,meta=workspace.resolve(candidate_id,kind='external_pgac_candidate');descriptor=read_json(root/'policy.json')
    names=('imitation_policy','rollout_model','model_validation','external_expert_dataset','expert','plant')
    if any(not isinstance(descriptor.get(key),str) for key in names):raise ContractError('PGAC candidate must bind its actual baseline/model/validation/dataset/expert/Plant IDs')
    closure=ep._closure(workspace,candidate_id)
    if any(descriptor[key] not in closure for key in names):raise ContractError('A candidate source is not an immutable ancestor')
    selection=ep._selection(workspace,descriptor['imitation_policy'])
    if not selection['applicable']:raise ContractError('PGAC original-source evaluation is not applicable: '+selection['reason'])
    if any(descriptor[key]!=selection[target] for key,target in [('external_expert_dataset','dataset_id'),('expert','expert_id'),('plant','plant_id')]):raise ContractError('PGAC source differs from the exact baseline dataset ancestry')
    from .differentiable_validation import read_differentiable_validation
    qualification=read_differentiable_validation(workspace,descriptor['model_validation'],candidate_id=descriptor['rollout_model'],original_id=selection['plant_id'])
    columns,reason=ep._features(selection)
    if columns is None:raise ContractError(reason)
    report=read_json(root/'report.json');frozen=report.get('original_evaluation_config')
    if not isinstance(frozen,dict) or set(frozen)-{'scenarios','evaluation','benchmark','condition_source'} or not {'scenarios','evaluation'}<=frozen.keys():raise ContractError('Candidate must retain its prospectively frozen original_evaluation_config')
    if meta['interface']!=selection['policy_meta']['interface']:raise ContractError('PGAC candidate changed its original action DataInterface')
    for name in ['network.json','scaler.json','data_interface.json','features.json','action_constraints.json','physical_context.json']:
        if read_json(root/'bundle'/name)!=read_json(selection['policy_root']/'bundle'/name):raise ContractError('PGAC candidate changed baseline deployment semantics: '+name)
    baseline_runtime=load_external_policy(selection['policy_root']);candidate_runtime=load_external_policy(root)
    if candidate_runtime.binding!=baseline_runtime.binding or candidate_runtime.deployment_mapping!=baseline_runtime.deployment_mapping or candidate_runtime.deployment_scope!=baseline_runtime.deployment_scope:
        raise ContractError('PGAC candidate changed the actual effective baseline deployment mapping or binding')
    domain=report.get('domain');coverage=qualification.get('coverage',{}).get('domain')
    if not isinstance(domain,dict) or set(domain)!={'x','u','d'} or not isinstance(coverage,dict):raise ContractError('Candidate must retain its qualified rollout x/u/d domain')
    from .matlab_expert import validate_coverage_domain
    domain=validate_coverage_domain(domain,selection['physical']);report['domain']=domain
    for key,variables in [('x',selection['physical'].states),('u',selection['physical'].controls),('d',selection['physical'].disturbances)]:
        box=np.asarray(domain[key],dtype=np.float64).reshape(-1,2);bound=np.asarray(coverage[key],dtype=np.float64).reshape(-1,2)
        if box.shape!=(len(variables),2) or bound.shape!=box.shape or not np.isfinite(box).all() or np.any(box[:,0]>=box[:,1]) or np.any(box[:,0]<bound[:,0]) or np.any(box[:,1]>bound[:,1]):raise ContractError('Candidate domain differs from qualified model coverage: '+key)
    return {**selection,'candidate_id':candidate_id,'candidate_root':root,'candidate_meta':meta,'candidate_descriptor':descriptor,
        'candidate_report':report,'frozen_evaluation':frozen,'columns':columns}


def normalize_original_evaluation(selection,config):
    """Static pre-training normalization; no Plant, expert or NN is executed."""
    if not isinstance(config,dict) or set(config)-{'scenarios','evaluation','benchmark','plan_refs','condition_source'}:raise ContractError('Unknown original PGAC evaluation setting')
    if not isinstance(config.get('scenarios'),list) or not config['scenarios']:raise ContractError('Review all explicit original validation/test scenarios before evaluation')
    objective=evaluation_config(config.get('evaluation'))
    plan,reason=ep._evaluation_plan(selection,config);bound,_=ep._evaluation_plan(selection,{})
    if plan is None:raise ContractError(reason)
    identity=lambda row:canonical_hash({'condition':ep._condition_hash(row),'steps':row['steps'],'split':row['split']})
    if not plan['split']['validation'] or not plan['split']['test']:
        raise ContractError('PGAC requires independent validation and test conditions')
    mode = config.get('condition_source', 'exact_bound')
    if mode == 'fresh_holdout':
        collection = read_json(selection['dataset_root']/'collection-plan.json')
        ensure_new_physical_conditions(collection['trajectories'], plan['trajectories'])
    elif mode != 'exact_bound' or sorted(map(identity,plan['trajectories']))!=sorted(map(identity,bound['trajectories'])):
        raise ContractError('PGAC comparison must contain every exact original bound validation/test condition and duration')
    i=selection['physical'];logical=3*sum(row['steps'] for row in plan['trajectories'])*(2*i.nx+3*i.nu+3*i.ny+i.nd+len(selection['expert']['observations'])+8)
    if logical>5_000_000:raise ContractError('One three-controller original evaluation exceeds the five-million-value batch boundary; use smaller confirmed condition batches')
    return {'scenarios':copy.deepcopy(config['scenarios']),'evaluation':objective,
            **({'condition_source':mode} if 'condition_source' in config else {}),
            **({'benchmark':ep.benchmark_config(config['benchmark'])} if 'benchmark' in config else {})}


def _plan(selection,config):
    normalized=normalize_original_evaluation(selection,config)
    if normalized!=selection['frozen_evaluation']:raise ContractError('Original evaluation cases/objective changed after PGAC training')
    plan,_=ep._evaluation_plan(selection,normalized)
    return plan,normalized['evaluation'],normalized.get('benchmark')


def _subplan(plan,part,spec=None):
    result=copy.deepcopy(plan);result['trajectories']=[row for row in result['trajectories'] if row['split']==part]
    result['split']={name:[row['trajectory_id'] for row in result['trajectories'] if row['split']==name] for name in ['train','validation','test']}
    if spec is not None:result['expert_spec_hash']=canonical_hash(spec)
    result.pop('plan_hash',None);result['plan_hash']=canonical_hash(result);return result


def quadratic_result(result,physical,plan,objective,domain=None):
    """Recompute score and hard feasibility from each controller's own rows."""
    objective=evaluation_config(objective);a=result['arrays'];records=[]
    yr=np.asarray([v.upper-v.lower for v in physical.outputs]);ur=np.asarray([v.upper-v.lower for v in physical.controls])
    constraints=read_json(Path(result['constraints_path'])) if 'constraints_path' in result else result.get('action_constraints')
    if not isinstance(constraints,dict):raise ContractError('Original evaluation must retain inherited action constraints')
    rate=constraints['rate_bounds'];complete=result['status']=='completed';expected=sum(row['steps'] for row in plan['trajectories'])
    if len(a['time'])!=expected:complete=False
    checks={'completed_all_trajectories':bool(complete),'state':True,'output':True,'input':True,'input_rate':True,'causal_chain':True,'finite':True,'qualified_domain':True,'subtick_coverage':True}
    if domain is not None:
        for name,key in [('x','x'),('next_x','x'),('u','u'),('d','d')]:
            bounds=np.asarray(domain[key],dtype=np.float64).reshape(-1,2);raw=np.asarray(a[name])
            checks['qualified_domain']&=bool(np.isfinite(raw).all() and np.all(raw>=bounds[:,0]) and np.all(raw<=bounds[:,1]))
        guard=result.get('coverage_guard',result.get('report',{}).get('coverage_guard',{}));ticks=expected*plan['plant_substeps']
        checks['subtick_coverage']=bool(guard.get('enabled') is True and guard.get('domain_hash')==canonical_hash(domain) and guard.get('status')=='passed' and guard.get('violations')==0
            and all(type(guard.get(key)) is int and guard[key]==ticks for key in ['before_checks','after_checks','passed_subticks']))
        # An empty safe prefix is not evidence that the complete original run
        # remained in coverage; include the actual subtick guard outcome.
        checks['qualified_domain']&=checks['subtick_coverage']
    for name,variables in [('x',physical.states),('next_x',physical.states),('plant_y',physical.outputs),('next_y',physical.outputs),('u',physical.controls)]:
        raw=np.asarray(a[name]);lo=np.asarray([v.lower for v in variables]);hi=np.asarray([v.upper for v in variables]);finite=np.isfinite(raw).all();checks['finite']&=bool(finite)
        key='state' if name in {'x','next_x'} else 'output' if name in {'plant_y','next_y'} else 'input'
        checks[key]&=bool(finite and np.all(raw>=lo-1e-6) and np.all(raw<=hi+1e-6))
    for condition in plan['trajectories']:
        indexes=np.flatnonzero(a['trajectory_id']==condition['trajectory_id']);count=len(indexes)
        if count!=condition['steps']:complete=False
        if not count:records.append({'trajectory_id':condition['trajectory_id'],'score':None,'steps':0});continue
        u,previous=np.asarray(a['u'][indexes]),np.asarray(a['previous_u'][indexes]);x=np.asarray(a['x'][indexes]);xn=np.asarray(a['next_x'][indexes])
        checks['causal_chain']&=bool(np.allclose(x[0],condition['x0'],atol=1e-12,rtol=0) and np.allclose(previous[0],condition['initial_action'],atol=1e-12,rtol=0)
            and np.allclose(x[1:],xn[:-1],atol=1e-12,rtol=0) and np.allclose(previous[1:],u[:-1],atol=1e-12,rtol=0)
            and np.allclose(a['time'][indexes],condition['time'][:count],atol=1e-12,rtol=0)
            and np.allclose(a['reference'][indexes],scenario_series(condition,'reference')[:count],atol=1e-12,rtol=0)
            and np.allclose(a['d'][indexes],scenario_series(condition,'disturbance')[:count],atol=1e-12,rtol=0))
        if rate is not None:
            bounds=np.asarray(rate);delta=u-previous;checks['input_rate']&=bool(np.all(delta>=bounds[:,0]-1e-6) and np.all(delta<=bounds[:,1]+1e-6))
        tracking=np.mean(((a['next_y'][indexes]-a['reference'][indexes])/yr)**2,axis=1)
        action=np.mean((u/ur)**2,axis=1);increment=np.mean(((u-previous)/ur)**2,axis=1)
        terms={'tracking':float(np.mean(tracking)),'action':float(np.mean(action)),'increment':float(np.mean(increment))}
        score=objective['q']*terms['tracking']+objective['r']*terms['action']+objective['s']*terms['increment']
        checks['finite']&=math.isfinite(score)
        records.append({'trajectory_id':condition['trajectory_id'],'family':condition.get('family'),'noise_seed':condition.get('noise_seed'),
                        'steps':count,'terms':terms,'score':score if math.isfinite(score) else None,
                        'tracking_rmse_physical':np.sqrt(np.mean((a['next_y'][indexes]-a['reference'][indexes])**2,axis=0)).tolist(),
                        'peak_abs_state':np.maximum(np.abs(x).max(axis=0),np.abs(xn).max(axis=0)).tolist(),
                        'peak_abs_action':np.abs(u).max(axis=0).tolist(),'peak_abs_increment':np.abs(u-previous).max(axis=0).tolist()})
    checks['completed_all_trajectories']&=bool(complete)
    feasible=all(checks.values());score=float(np.mean([row['score'] for row in records])) if feasible else None
    return {'feasible':bool(feasible),'score':score,'checks':checks,'per_scenario':records,
        'aggregation':'mean per-control-step normalized q/r/s, equal mean across complete scenarios','internal_subticks':'S09 collector checks every physical state/output subtick; rows record the next control tick',
        'qualified_domain_scope':'actual original before x/u/d and after x at every physical Plant dt subtick, plus logged control endpoints; not a proof of joint coverage' if domain is not None else 'not_applied_to_expert_or_retained_imitation'}


def _passes_performance_gate(baseline, candidate, objective, *, validation):
    tol = objective['comparison_atol']
    improvement = baseline['score'] - candidate['score']
    minimum = baseline['score'] * objective.get('min_relative_improvement', 0.)
    if (validation and improvement <= tol) or improvement + tol < minimum:
        return False
    if 'max_worst_case_relative_regression' in objective:
        base = {row['trajectory_id']:row['score'] for row in baseline['per_scenario']}
        limit = objective['max_worst_case_relative_regression']
        return all(row['score'] <= base[row['trajectory_id']] * (1.+limit) + tol
                   for row in candidate['per_scenario'])
    return True


def validation_choice(scores,objective,*,candidate_eligible=True):
    baseline,candidate=scores['imitation'],scores['candidate'];tol=evaluation_config(objective)['comparison_atol']
    if candidate_eligible and candidate['feasible'] and (not baseline['feasible'] or _passes_performance_gate(baseline,candidate,evaluation_config(objective),validation=True)):choice='candidate';reason='candidate_strictly_better_on_original_validation' if baseline['feasible'] else 'candidate_only_feasible_original_validation'
    elif baseline['feasible']:choice='imitation';reason='feasible_imitation_retained_without_required_improvement'
    else:choice=None;reason='no_approved_validation_fallback'
    return {'choice':choice,'reason':reason,'candidate_eligible':bool(candidate_eligible),'scores':scores,'comparison_atol':tol,'selection_partition':'validation','test_used':False}


def final_acceptance(selection,scores,objective):
    """Binary gate for the already frozen version; no alternate epoch search."""
    chosen=selection['choice'];baseline,candidate=scores['imitation'],scores['candidate'];objective=evaluation_config(objective)
    baseline_approved=baseline['feasible'] and selection['scores']['imitation']['feasible']
    if chosen=='candidate':
        accepted=candidate['feasible'] and (not baseline['feasible'] or _passes_performance_gate(baseline,candidate,objective,validation=False))
        if accepted:return {'selected_version':'candidate','candidate_accepted':True,'approved':True,'reason':'fixed_candidate_passed_original_test'}
        return {'selected_version':'imitation' if baseline_approved else None,'candidate_accepted':False,'approved':bool(baseline_approved),
            'reason':'fixed_candidate_rejected_existing_feasible_baseline_retained' if baseline_approved else 'fixed_candidate_rejected_no_approved_fallback'}
    if chosen=='imitation' and baseline_approved:return {'selected_version':'imitation','candidate_accepted':False,'approved':True,'reason':'fixed_validation_baseline_passed_original_test'}
    return {'selected_version':None,'candidate_accepted':False,'approved':False,'reason':'no_approved_fallback_no_test_driven_candidate_selection'}


def _candidate_eligible(selection):
    report=selection['candidate_report']
    # The trainer must explicitly attest its verified domain contract; absence
    # cannot turn an unchecked rollout into a deployment candidate.
    return report.get('rollout_domain_checks_passed') is True


def _save_results(stage,phase,results):
    hashes={}
    for name,result in results.items():
        path=stage.root/(phase+'-'+name+'-rollout.npz');np.savez_compressed(path,**result['arrays']);hashes[name]=file_hash(path)
        atomic_json(stage.root/(phase+'-'+name+'-diagnostics.json'),result['diagnostics'])
    return hashes


def _scores(selection,results,plan,objective):
    constraints=read_json(selection['dataset_root']/'action_constraints.json')
    return {name:quadratic_result({**result,'action_constraints':constraints},selection['physical'],plan,objective,
        selection['candidate_report']['domain'] if name=='candidate' else None) for name,result in results.items()}


def _publish(stage,selection,config,context,plan,objective,results,decision,selection_sha,acceptance,benchmark=None):
    phase_scores={part:_scores(selection,results[part],_subplan(plan,part),objective) for part in ['validation','test']}
    reports={part:{name:{'status':value['status'],'report':value['report'],'metrics':ep._metrics({**value,'rate_bounds':selection['expert'].get('input_rate_bounds')},selection['physical'],_subplan(plan,part),{}),
        'quadratic':phase_scores[part][name],'rollout_file':part+'-'+name+'-rollout.npz','feedback':'own states and actual previous applied action; no teacher forcing'} for name,value in results[part].items()} for part in results}
    for part,group in results.items():
        for name,value in group.items():
            reports[part][name]['plant_timing']=value.get('plant_timing',{'status':'not_separately_instrumented','included_in_controller_benchmark':False})
            reports[part][name]['coverage_guard']=value.get('coverage_guard',value['report'].get('coverage_guard'))
    selected=acceptance['selected_version'];source_root=selection['candidate_root'] if selected=='candidate' else selection['policy_root']
    if selected is not None:
        policy=load_external_policy(source_root)
        export_external_policy_state(stage.root/'bundle',policy.network.state_dict(),read_json(source_root/'bundle/network.json'),selection['policy_meta']['interface'],
            read_json(source_root/'bundle/scaler.json'),read_json(source_root/'bundle/action_constraints.json'),read_json(source_root/'bundle/physical_context.json'),
            provenance={'source_mapping':selection['data'].get('source_mapping',{}),'selected_version':selected,'candidate':selection['candidate_id'],'imitation_policy':selection['policy_id'],
                'selection_sha256':selection_sha,'algorithm':'differentiable_physical_rollout_policy_gradient','learned_critic':False,
                'pgac_candidate_domain':selection['candidate_report']['domain'],'online_ood_fallback_implemented':False})
    descriptor={'schema_version':1,'representation':'external_optimized_policy','bundle':'bundle' if selected is not None else None,'selected_version':selected,
        'candidate':selection['candidate_id'],'imitation_policy':selection['policy_id'],'external_expert_dataset':selection['dataset_id'],'expert':selection['expert_id'],'plant':selection['plant_id'],
        'rollout_model':selection['candidate_descriptor']['rollout_model'],'model_validation':selection['candidate_descriptor']['model_validation']}
    report={'schema_version':1,'status':'completed' if acceptance['approved'] else 'rejected','algorithm':'differentiable_physical_rollout_policy_gradient','learned_critic':False,
        **descriptor,'original_evaluation_config':selection['frozen_evaluation'],'plan':plan,'selection':decision,'selection_sha256':selection_sha,
        'final_acceptance':acceptance,'closed_loop':reports,'versions':{'expert':selection['expert_id'],'imitation':selection['policy_id'],'optimized_candidate':selection['candidate_id']},
        'controller_benchmark':benchmark or {'status':'not_requested','scope':'No new latency claim; prior S10 timing remains separate'},
        'test_used_for_checkpoint_or_hyperparameter_selection':False,'final_test_rule':'Accept/reject only the validation-fixed version; existing feasible baseline may be retained after candidate rejection, never another epoch',
        'original_plant':'exact immutable baseline external-expert-dataset ancestor','action_rate_semantics':'per control decision; hold action over all original Plant subticks',
        'physical_degradation_reported':True,'improvement_required_for_stage_completion':False,'stability_proof':False,'confirmation_sources':context.case_confirmation_sources}
    report['deployment_scope']={'candidate_domain':selection['candidate_report']['domain'],'original_domain_checks':'logged control-tick x/next_x/u/d',
        'original_physical_bounds':'checked at every Plant subtick','original_candidate_domain':'before x/u/d and after x at every actual Plant dt subtick',
        'online_ood_fallback':'not_implemented_by_standalone_MLP; an external domain guard is required for deployment','joint_coverage_or_global_stability_proved':False}
    atomic_json(stage.root/'policy.json',descriptor);atomic_json(stage.root/'report.json',report)
    (stage.root/'README.md').write_text('Original-Plant PGAC comparison\n\nThe validation selection was fixed before final tests. Failed/degraded candidates do not replace an actually feasible original imitation baseline. Expert, imitation and candidate IDs, conditions, raw autonomous trajectories and the exact selected weight bundle are retained in report.json.\n\nCandidate training and original-source validation/test check qualified x/u/d before, and x after, each actual Plant dt step. These checks cover the recorded finite scenarios only; marginal boxes do not prove joint coverage or stability. The exported MLP itself has no future online OOD fallback. Deployment requires an external actual-state/action/disturbance domain guard using the declared report domain; never infer state/history from feature width or assume a missing signal is zero.\n',encoding='utf-8')
    return stage.publish({'status':'verified' if acceptance['approved'] else 'rejected','original_plant_test_accepted':acceptance['approved'],
        'candidate_accepted':acceptance['candidate_accepted'],'baseline_retained':selected=='imitation','stability_proof':False},
        source={'algorithm':report['algorithm'],**descriptor,'selection_sha256':selection_sha},activate=bool(acceptance['approved']))


def _python_evaluation(workspace,selection,config,context,plan,objective,settings):
    from .expert_adapter import load_external_expert
    from .model_adapters import load_python_adapter
    from .plants import _TorchAdapter
    expert=load_external_expert(selection['expert_root'],selection['expert'],expected_file_hashes=selection['expert_material']['files'])
    baseline=load_external_policy(selection['policy_root']);candidate=load_external_policy(selection['candidate_root'])
    controllers={'expert':expert,'imitation':ep._PolicyAdapter(baseline,selection['expert'],selection['columns']),'candidate':ep._PolicyAdapter(candidate,selection['expert'],selection['columns'])}
    parents=[selection[key] for key in ['candidate_id','policy_id','dataset_id','expert_id','plant_id']]
    old_threads=torch.get_num_threads();torch.set_num_threads(1);results={};benchmark=None
    try:
        with workspace.stage('external_optimized_policy',DataInterface.parse(selection['policy_meta']['interface']),parents,config,compatible_parents=False) as stage:
            try:
                with threadpoolctl.threadpool_limits(limits=1):
                    for phase in ['validation','test']:
                        subplan=_subplan(plan,phase);results[phase]={}
                        for name,controller in controllers.items():
                            context.check();original=load_python_adapter(selection['plant_root'],selection['plant_spec'],expected_file_hashes=selection['plant_material']['files']) if selection['plant_spec'] is not None else _TorchAdapter(workspace,selection['plant_id'])
                            plant=ep._TimedPlant(_CoveragePlant(original,selection['candidate_report']['domain']) if name=='candidate' else original)
                            try:
                                try:result=_collect_python(plant,controller,subplan,context)
                                except ExpertCollectionCancelled:raise
                                except ExpertCollectionStopped as error:result={'status':error.report['status'],'arrays':error.arrays,'report':error.report,'diagnostics':error.diagnostics}
                                result['plant_timing']=plant.describe()
                                if name=='candidate':result['coverage_guard']=copy.deepcopy(result['plant_timing']['coverage_guard'])
                                results[phase][name]=result
                            finally:plant.close()
                        hashes=_save_results(stage,phase,results[phase])
                        if phase=='validation':
                            decision=validation_choice(_scores(selection,results[phase],subplan,objective),objective,candidate_eligible=_candidate_eligible(selection))
                            snapshot={'schema_version':1,'phase':'frozen_before_any_test_rollout','candidate':selection['candidate_id'],'imitation_policy':selection['policy_id'],
                                'plan_hash':plan['plan_hash'],'evaluation':objective,'decision':decision,'validation_rollout_sha256':hashes,
                                'candidate_weight_sha256':file_hash(selection['candidate_root']/'bundle/model.pt'),'baseline_weight_sha256':file_hash(selection['policy_root']/'bundle/model.pt')}
                            atomic_json(stage.root/'selection.json',snapshot);selection_sha=file_hash(stage.root/'selection.json');context.progress('pgac_original_validation_selection_frozen',selection_sha256=selection_sha,choice=decision['choice'])
                    acceptance=final_acceptance(decision,_scores(selection,results['test'],_subplan(plan,'test'),objective),objective)
                    if settings:
                        left=ep._benchmark_python(expert,baseline,selection['columns'],results['test']['expert'],settings,context)
                        right=ep._benchmark_python(expert,candidate,selection['columns'],results['test']['expert'],settings,context)
                        median=right['imitation']['median']
                        benchmark={**left,'candidate':right['imitation'],'expert_repeated_for_candidate_pair':right['expert'],
                            'candidate_speedup_median':None if median<=0 else left['expert']['median']/median,
                            'scope':left['scope']+' Both neural versions use exactly the same recorded physical expert contexts. Expert was measured again for the candidate pair; both raw measurements are retained, displayed ratios use the displayed expert median.'}
                    return _publish(stage,selection,config,context,plan,objective,results,decision,selection_sha,acceptance,benchmark)
            except Exception as error:
                for part,value in results.items():_save_results(stage,part,value)
                atomic_json(stage.root/'report.json',{'status':'cancelled' if isinstance(error,Cancelled) else 'failed','failure':{'type':type(error).__name__,'message':str(error)},'candidate':selection['candidate_id'],'policy_replacement':'not_performed','test_selection':'not_performed'})
                raise
    finally:expert.close();torch.set_num_threads(old_threads)


def evaluate_external_pgac(workspace,candidate_id,config,context=None,request=None):
    context=context or OperationContext();context.check();selection=_selection(workspace,candidate_id);plan,objective,settings=_plan(selection,config)
    if selection['expert']['backend']=='matlab_function':return _matlab_evaluation(workspace,selection,config,context,plan,objective,settings)
    return _python_evaluation(workspace,selection,config,context,plan,objective,settings)


def _matlab_evaluation(workspace,selection,config,context,plan,objective,settings):
    from .matlab_expert import prepare_matlab_expert_manifest,load_matlab_expert_batch
    adapters={'expert':selection['expert']};roots={'expert':selection['expert_root'],'imitation':selection['policy_root'],'candidate':selection['candidate_root']}
    for name,meta in [('imitation',selection['policy_meta']),('candidate',selection['candidate_meta'])]:
        adapter,reason=ep._matlab_adapter({**selection,'policy_root':roots[name],'policy_meta':meta})
        if adapter is None:raise ContractError('Original MATLAB PGAC comparison is not applicable: '+str(reason))
        adapters[name]=adapter
    phase=os.environ.get('CONTROL_MATLAB_PHASE');raw=context._case_raw_request;digest=os.environ.get('CONTROL_MATLAB_REQUEST_SHA256','');nonce=os.environ.get('CONTROL_MATLAB_NONCE','')
    if not context._case_receipt_body or phase not in {'prepare','finalize'} or not raw or hashlib.sha256(raw.encode()).hexdigest()!=digest or not re.fullmatch(r'[a-f0-9-]{36}',nonce):
        raise ContractError('Original MATLAB PGAC evaluation requires one exact-request-bound native Host batch, not Engine or pointwise MCP')
    exchange=Path(os.environ.get('CONTROL_MATLAB_EXCHANGE_DIR','')).resolve()
    if exchange.parent!=(workspace.path/'.case-plans/matlab-exchanges').resolve() or exchange.name!=nonce:raise ContractError('PGAC exchange escaped the confirmed workspace')
    versions={name:ep._version(workspace,selection[key]) for name,key in [('policy_version','policy_meta'),('candidate_version','candidate_meta'),('dataset_version','dataset_meta'),
        ('expert_version','expert_meta'),('expert_material_version','expert_material'),('plant_version','plant_meta'),('plant_material_version','plant_material')]}
    spec_hash=canonical_hash({'controllers':adapters,'plant':selection['plant_spec'],'evaluation':objective,'benchmark':settings,'candidate_eligible':_candidate_eligible(selection),'candidate_domain':selection['candidate_report']['domain'],**versions})
    expected={'schema_version':1,'nonce':nonce,'request_sha256':digest,'spec_hash':spec_hash,**versions}
    common={'request_sha256':digest,'nonce':nonce,'plant_version':versions['plant_version'],'plant_material_version':versions['plant_material_version']}
    child_versions={'expert':{'expert_version':versions['expert_version'],'expert_material_version':versions['expert_material_version']},
        'imitation':{'expert_version':versions['policy_version'],'expert_material_version':versions['policy_version']},
        'candidate':{'expert_version':versions['candidate_version'],'expert_material_version':versions['candidate_version']}}
    with workspace.lock('evaluate_external_pgac'):
        if phase=='prepare':
            exchange.mkdir(parents=True,exist_ok=False);children={}
            for part in ['validation','test']:
                children[part]={}
                for name in NAMES:
                    children[part][name]=prepare_matlab_expert_manifest(adapters[name],roots[name],selection['plant_spec'],selection['plant_root'],_subplan(plan,part,adapters[name]),
                        exchange_dir=exchange/part/name,**common,**child_versions[name],coverage_domain=selection['candidate_report']['domain'] if name=='candidate' else None)
            document={**expected,'kind':'control-matlab-pgac-evaluation-v1','plan':plan,'evaluation':objective,'benchmark':settings,'candidate_eligible':_candidate_eligible(selection),'candidate_domain':selection['candidate_report']['domain'],
                'children':children,'selection_path':str(exchange/'selection.json'),'result_path':str(exchange/'result.json'),'batch_kind':'external_pgac_evaluation'}
            path=exchange/'manifest.json';atomic_json(path,document)
            return {'status':'verified','matlab_phase':'prepared','manifest_path':str(path),'manifest_sha256':file_hash(path),'exchange_dir':str(exchange),
                'result_path':document['result_path'],'request_sha256':digest,'nonce':nonce,'plan_hash':plan['plan_hash'],'batch_kind':'external_pgac_evaluation'}
        path=Path(os.environ.get('CONTROL_MATLAB_RESULT_PATH','')).resolve();digest_result=os.environ.get('CONTROL_MATLAB_RESULT_SHA256','')
        if path!=exchange/'result.json' or not re.fullmatch('[a-f0-9]{64}',digest_result) or path.stat().st_size>256*1024*1024 or file_hash(path)!=digest_result:raise ContractError('PGAC result differs from the native Host-observed bytes')
        document=read_json(exchange/'manifest.json');result=read_json(path)
        if (document.get('kind')!='control-matlab-pgac-evaluation-v1' or result.get('kind')!='control-matlab-pgac-evaluation-result-v1' or
            any(document.get(k)!=v or result.get(k)!=v for k,v in expected.items()) or document.get('plan')!=plan or document.get('evaluation')!=objective or document.get('benchmark')!=settings or
            document.get('candidate_eligible')!=_candidate_eligible(selection) or document.get('candidate_domain')!=selection['candidate_report']['domain'] or document.get('result_path')!=str(path) or result.get('plan_hash')!=plan['plan_hash']):
            raise ContractError('PGAC native source versions, objective, conditions or request identity changed')
        results={};selection_path=exchange/'selection.json'
        if not selection_path.is_file() or selection_path.stat().st_size>4*1024*1024:raise ContractError('PGAC native validation selection snapshot is missing or oversized')
        selection_sha=file_hash(selection_path)
        if result.get('selection_sha256')!=selection_sha or result.get('test_started_after_selection') is not True or result.get('test_selection_sha256')!={name:selection_sha for name in NAMES}:
            raise ContractError('PGAC test batches were not bound to the frozen pre-test selection bytes')
        frozen=read_json(selection_path)
        if frozen.get('phase')!='frozen_before_any_test_rollout' or frozen.get('evaluation')!=objective or frozen.get('plan_hash')!=plan['plan_hash'] or frozen.get('request_sha256')!=digest or frozen.get('nonce')!=nonce:
            raise ContractError('PGAC native validation selection snapshot has a different confirmed contract')
        for part in ['validation','test']:
            results[part]={}
            for name in NAMES:
                child_manifest=exchange/part/name/'manifest.json';prepared=document.get('children',{}).get(part,{}).get(name,{})
                if prepared.get('manifest_path')!=str(child_manifest) or prepared.get('manifest_sha256')!=file_hash(child_manifest):raise ContractError('PGAC child manifest changed')
                child=exchange/part/name/'result.json';child_sha=result.get('children',{}).get(part,{}).get(name)
                if not isinstance(child_sha,str) or file_hash(child)!=child_sha:raise ContractError('PGAC original-source child result hash changed')
                results[part][name]=load_matlab_expert_batch(adapters[name],selection['plant_spec'],_subplan(plan,part,adapters[name]),result_path=child,result_sha256=child_sha,**common,**child_versions[name],
                    coverage_domain=selection['candidate_report']['domain'] if name=='candidate' else None)
        scores=_scores(selection,results['validation'],_subplan(plan,'validation'),objective);decision=validation_choice(scores,objective,candidate_eligible=_candidate_eligible(selection))
        native_decision=frozen.get('decision',{});native_choice=native_decision.get('choice');native_choice=None if native_choice==[] else native_choice
        if native_choice!=decision['choice'] or native_decision.get('reason')!=decision['reason'] or native_decision.get('candidate_eligible')!=decision['candidate_eligible']:
            raise ContractError('Native original validation choice disagrees with independent Python recomputation')
        for name in NAMES:
            native=native_decision.get('scores',{}).get(name,{})
            if native.get('feasible')!=scores[name]['feasible']:raise ContractError('Native original validation hard feasibility differs')
            value=native.get('score');value=None if value==[] else value
            if scores[name]['score'] is None and value is not None or scores[name]['score'] is not None and (type(value) not in {int,float} or not math.isclose(value,scores[name]['score'],abs_tol=1e-12,rel_tol=1e-12)):
                raise ContractError('Native q/r/s objective differs from independent physical-row recomputation')
        acceptance=final_acceptance(decision,_scores(selection,results['test'],_subplan(plan,'test'),objective),objective)
        all_complete=all(child['status']=='completed' for group in results.values() for child in group.values())
        expected_status='completed' if all_complete else 'partial' if any(len(child['arrays']['time']) for group in results.values() for child in group.values()) else 'failed'
        if result.get('status')!=expected_status:raise ContractError('PGAC native terminal status contradicts its actual valid prefixes')
        benchmark=result.get('controller_benchmark',{'status':'not_requested'})
        if settings:
            if benchmark.get('status')!='completed' or benchmark.get('settings')!=settings or benchmark.get('matlab_max_computational_threads')!=1:raise ContractError('PGAC timing did not complete under the frozen settings')
            for name in NAMES:
                values=ep._stats(benchmark.get(name,{}).get('raw_seconds'))
                if values['samples']!=settings['repeats']:raise ContractError('PGAC timing repeat count differs')
                benchmark[name]=values
        parents=[selection[key] for key in ['candidate_id','policy_id','dataset_id','expert_id','plant_id']]
        with workspace.stage('external_optimized_policy',DataInterface.parse(selection['policy_meta']['interface']),parents,config,compatible_parents=False) as stage:
            for part,values in results.items():_save_results(stage,part,values)
            (stage.root/'selection.json').write_bytes(selection_path.read_bytes())
            return _publish(stage,selection,config,context,plan,objective,results,decision,selection_sha,acceptance,benchmark)
