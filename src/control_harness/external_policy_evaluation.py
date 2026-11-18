"""Original-source autonomous expert/imitation comparisons and isolated timing.

Evaluation publishes an inactive report, never a replacement policy. A missing
executable source chain is explicitly not applicable; tabular feature counts do
not define physical state, observations, a plant or controller execution time.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
import torch
import threadpoolctl

from .artifacts import atomic_json,read_json,file_hash
from .contracts import ContractError,DataInterface,MaterialInterface,ModelInterface,canonical_hash
from .runtime import OperationContext,Cancelled
from .external_expert import (_expert_selection,_plant_selection,_collect_python,_vector,_bounded,
    ExpertCollectionStopped,ExpertCollectionCancelled)


def benchmark_config(value=None):
    value=value or {}
    if not isinstance(value,dict) or set(value)-{'warmup','repeats','batch_size','threads'}:raise ContractError('Unknown controller benchmark setting')
    result={'warmup':5,'repeats':50,'batch_size':1,'threads':1,**value}
    if any(type(result[k]) is not int or result[k]<1 for k in result):raise ContractError('Benchmark counts and fixed execution settings must be positive integers')
    if result['batch_size']!=1 or result['threads']!=1:raise ContractError('Comparable controller benchmark requires CPU one thread and batch size one')
    if result['repeats']>1_000_000:raise ContractError('One benchmark timing array exceeds its storage budget; use separate benchmark batches')
    return result


def prepare_external_imitation_evaluation(workspace,policy_id,config=None):
    """Read-only domain proposal: resolve actual held-out cases before approval.

    No user module, policy network, expert, Plant or MATLAB service is executed.
    The returned explicit config can be reviewed by the native confirmation path.
    """
    config=config or {}
    if not isinstance(config,dict) or set(config)-{'scenarios','benchmark','metrics','plan_refs'}:raise ContractError('Unknown evaluation proposal setting')
    from .objectives import normalize_evaluation
    settings=benchmark_config(config.get('benchmark'));metric_settings=normalize_evaluation({'recipe':'tracking','scales':{'rmse':1,'energy':1,'tv':1,'overshoot':1},'metrics':config.get('metrics',{})})['metrics']
    selection=_selection(workspace,policy_id)
    versions={name:_version(workspace,selection[key]) for name,key in [('policy','policy_meta'),('dataset','dataset_meta')]}
    result={'schema_version':1,'inputs':{'policy':policy_id},'source_executed':False,'training':'not_run','bound_source_versions':versions,
        'dataset_source':selection['dataset_meta'].get('source',{}),'confirmation_required':[]}
    if not selection['applicable']:return {**result,'status':'not_applicable','reason':selection['reason'],'config_suggestions':None}
    for name,key in [('expert','expert_meta'),('expert_material','expert_material'),('plant','plant_meta')]:versions[name]=_version(workspace,selection[key])
    if selection['plant_material'] is not None:versions['plant_material']=_version(workspace,selection['plant_material'])
    columns,reason=_features(selection)
    if columns is None:return {**result,'status':'not_applicable','reason':reason,'config_suggestions':None}
    plan,reason=_evaluation_plan(selection,config)
    if plan is None:return {**result,'status':'not_applicable','reason':reason,'config_suggestions':None}
    scenarios=[{'name':'bound-'+row['split']+'-trajectory-'+str(row['trajectory_id']),'initial_state':row['x0'],'reference':row['reference'],
        'disturbance':row['disturbance'],'steps':row['steps'],'initial_previous_action':row['initial_action'],'split':row['split']} for row in plan['trajectories']]
    return {**result,'status':'proposal','confirmation_required':['signals','sampling'],
        'config_suggestions':{'scenarios':scenarios,'benchmark':settings,'metrics':metric_settings},
        'resolved_source_trajectory_ids':[row['trajectory_id'] for row in plan['trajectories']],
        'control_period':plan['control_period'],'plant_dt':plan['plant_dt'],'plant_substeps':plan['plant_substeps'],
        'observations':selection['expert']['observations'],'action_constraints':read_json(selection['dataset_root']/'action_constraints.json'),
        'scope':'Same immutable original Plant and expert, independent own-action autonomous rollouts; numerical outcomes and speed are not predicted by this proposal.'}


def _closure(workspace,artifact_id):
    seen=set();pending=[artifact_id]
    while pending:
        current=pending.pop()
        if current in seen:continue
        _,metadata=workspace.resolve(current);seen.add(current);pending.extend(row['id'] for row in metadata['parents'])
    return seen


def _selection(workspace,policy_id):
    root,metadata=workspace.resolve(policy_id,kind='external_imitation_policy');definition=read_json(root/'policy.json')
    dataset_id=definition.get('external_expert_dataset')
    if not isinstance(dataset_id,str) or dataset_id not in _closure(workspace,policy_id):raise ContractError('Policy must name its exact bound external expert dataset ancestor')
    dataset_root,dataset_meta=workspace.resolve(dataset_id,kind='external_expert_dataset')
    source=dataset_meta.get('source',{});expert_id,plant_id=source.get('expert'),source.get('plant')
    result={'policy_id':policy_id,'policy_root':root,'policy_meta':metadata,'dataset_id':dataset_id,'dataset_root':dataset_root,'dataset_meta':dataset_meta,
        'data':read_json(dataset_root/'dataset.json'),'physical_context':read_json(dataset_root/'physical_context.json')}
    if expert_id is None and plant_id is None:return {**result,'applicable':False,'reason':'No bound executable external expert and original plant; recorded actions are not executable controllers.'}
    ancestors=_closure(workspace,dataset_id)
    if not isinstance(expert_id,str) or not isinstance(plant_id,str) or expert_id not in ancestors or plant_id not in ancestors:
        raise ContractError('Dataset expert/plant source IDs must be real immutable ancestors, never unrelated active artifacts')
    expert,expert_root,expert_meta,expert_material=_expert_selection(workspace,expert_id)
    physical,plant_spec,plant_root,plant_meta,plant_material=_plant_selection(workspace,plant_id)
    physical.compatible(ModelInterface.parse(expert['interface']))
    result.update(applicable=True,expert_id=expert_id,plant_id=plant_id,expert=expert,expert_root=expert_root,expert_meta=expert_meta,expert_material=expert_material,
        physical=physical,plant_spec=plant_spec,plant_root=plant_root,plant_meta=plant_meta,plant_material=plant_material)
    native_plant=plant_spec is not None and plant_spec['backend'] in {'matlab_function','simulink'}
    if (expert['backend']=='matlab_function')!=native_plant:
        result.update(applicable=False,reason='Mixed Python/MATLAB original expert and Plant have no declared co-located evaluation bridge; no Engine or pointwise MCP fallback is invented.')
    return result


def _condition_hash(row):
    fields={key:[0. if float(v)==0 else float(v) for v in row[key]] for key in ('x0','reference','disturbance','initial_action')}
    return canonical_hash(fields)


def _evaluation_plan(selection,config):
    root=selection['dataset_root'];source_plan=read_json(root/'collection-plan.json') if (root/'collection-plan.json').is_file() else None
    expert=selection['expert'];i=selection['physical'];period=expert['control_period'];substeps=int(round(period/i.sample_time))
    training={_condition_hash(row) for row in (source_plan or {}).get('trajectories',[]) if row.get('split')=='train'}
    custom=config.get('scenarios')
    if source_plan is None and custom is not None:
        raise ContractError('Custom independent evaluation requires the bound collection plan to check training physical prefixes')
    if custom is None:
        if source_plan is None:return None,'No bound collection operating conditions; explicit evaluation scenarios are required.'
        trajectories=copy.deepcopy([row for row in source_plan['trajectories'] if row['split'] in {'validation','test'}]);origin='bound_dataset_validation_and_test'
        if not trajectories:raise ContractError('Bound dataset has no independent validation/test conditions')
    else:
        if not isinstance(custom,list) or not custom:raise ContractError('Custom evaluation scenarios must be a nonempty list')
        trajectories=[]
        for index,row in enumerate(custom):
            if not isinstance(row,dict) or set(row)-{'name','initial_state','reference','disturbance','steps','initial_previous_action','split'} or type(row.get('steps')) is not int or row['steps']<1:
                raise ContractError('Invalid explicit evaluation scenario')
            group=row.get('split','test')
            if group not in {'validation','test'}:raise ContractError('Autonomous evaluation does not label training conditions as independent held-out data')
            condition={'trajectory_id':index,'steps':row['steps'],'x0':_vector(row.get('initial_state'),i.nx,'initial_state'),
                'reference':_vector(row.get('reference'),i.ny,'reference'),'disturbance':_vector(row.get('disturbance'),i.nd,'disturbance'),
                'initial_action':_vector(row.get('initial_previous_action'),i.nu,'initial_previous_action'),'split':group}
            if _condition_hash(condition) in training:raise ContractError('Custom scenario overlaps a training physical prefix; labels or duration do not make it independent')
            trajectories.append(condition)
        origin='explicit_custom_independent_conditions'
    total=0;groups={'train':[],'validation':[],'test':[]}
    for row in trajectories:
        for key,variables in [('x0',i.states),('reference',i.outputs),('disturbance',i.disturbances),('initial_action',i.controls)]:_bounded(row[key],variables,key)
        total+=row['steps']
        if total>100000 or total*substeps>500000 or total*(2*i.nx+3*i.nu+3*i.ny+i.nd+len(expert['observations'])+8)>5_000_000:
            raise ContractError('One autonomous evaluation batch exceeds its numerical storage boundary; evaluate separate batches')
        row['time']=[k*period for k in range(row['steps'])];groups[row['split']].append(row['trajectory_id'])
    plan={'schema_version':1,'interface':i.to_dict(),'control_period':period,'plant_substeps':substeps,'plant_dt':i.sample_time,
        'trajectories':trajectories,'split':groups,'seed':(source_plan or {}).get('seed',0),'expert_spec_hash':canonical_hash(expert),
        'condition_source':origin,'independent_of_training':True,'resource_boundary':{'max_diagnostic_bytes':8*1024*1024}}
    # ModelInterface.to_dict retains dataclass tuples. The native manifest is
    # JSON, so freeze the same JSON value before both hashing and phase replay.
    plan=json.loads(json.dumps(plan,allow_nan=False))
    plan['plan_hash']=canonical_hash(plan);return plan,None


def _features(selection):
    data=selection['data'];interface=DataInterface.parse(data['interface']);mapping=data.get('source_mapping',{}).get('signals',{})
    signals={row['name']:row for row in data['interface']['signals']}
    if any(row.get('lag',0)!=0 or row.get('window',1)!=1 or signals[row['signal']].get('delay_steps',0)!=0 for row in data['interface']['features']):
        return None,'History/window/delayed features need an explicit autonomous history adapter; no missing history is invented.'
    columns=[]
    for entry in data['feature_order']:
        if entry.get('lag',0)!=0 or entry.get('delay',0)!=0 or entry.get('window',1)!=1:
            return None,'History/window features require a declared autonomous history adapter; zero or teacher-forced history is not invented.'
        item=mapping.get(entry['signal'],{});name=item.get('array');column=item.get('column')
        if name not in {'observation','reference','d','previous_u','actual_previous_u'} or type(column) is not int or column<0:
            return None,'No verified current observation/reference/disturbance/previous-action mapping for an autonomous rollout.'
        columns.append((name,column))
    if len(columns)!=interface.input_dim:raise ContractError('Feature mapping differs from the actual trained input width')
    return columns,None


def _feature_vector(columns,observation,reference,disturbance,previous):
    values={'observation':observation,'reference':reference,'d':disturbance,'previous_u':previous,'actual_previous_u':previous}
    try:result=np.asarray([values[name][column] for name,column in columns],dtype=np.float64)
    except (IndexError,TypeError) as error:raise ContractError('Current feature selector is outside the bound physical signal order') from error
    if not np.isfinite(result).all():raise ContractError('Current autonomous features are nonfinite')
    return result


class _PolicyAdapter:
    def __init__(self,model,spec,columns):self.model,self.spec,self.columns=model,spec,columns;self.count=0
    def reset(self):self.count=0;self.model.reset() if hasattr(self.model,'reset') else None
    def act(self,observation,reference,disturbance,previous):
        features=_feature_vector(self.columns,observation,reference,disturbance,previous)
        with torch.no_grad():action=self.model(torch.as_tensor(features,dtype=torch.float64),previous_action=torch.as_tensor(previous,dtype=torch.float64)).detach().cpu().numpy()
        i=ModelInterface.parse(self.spec['interface']);action=_bounded(action,i.controls,'imitation action')
        if self.spec.get('input_rate_bounds') is not None:
            rate=np.asarray(self.spec['input_rate_bounds']);delta=action-previous
            if np.any(delta<rate[:,0]-1e-12) or np.any(delta>rate[:,1]+1e-12):raise ContractError('Imitation action increment violates its inherited constraints')
        self.count+=1
        return action,{'status':'ok','controller':'imitation','memory_before':{'count':self.count-1},'memory_after':{'count':self.count},'autonomous_own_previous_action':True}
    def describe(self):return {'backend':'exported_external_policy_torch','input':'actual current own-state observations and own applied previous action','teacher_forcing':False}
    def close(self):pass


class _TimedPlant:
    def __init__(self,plant):self.plant=plant;self.interface=plant.interface;self.elapsed_ns=0;self.calls=0
    def __getattr__(self,name):return getattr(self.plant,name)
    def _call(self,name,*args):
        start=time.perf_counter_ns()
        try:return getattr(self.plant,name)(*args)
        finally:self.elapsed_ns+=time.perf_counter_ns()-start;self.calls+=1
    def reset(self,*args):return self._call('reset',*args)
    def step(self,*args):return self._call('step',*args)
    def output(self):return self._call('output')
    def describe(self):return {**self.plant.describe(),'separate_plant_timing_seconds':self.elapsed_ns/1e9,'timed_plant_calls':self.calls}


def _stats(values):
    from .matlab_adapter import _native_numeric
    a=_native_numeric(values,'controller timing',max_rank=1).astype(np.float64).reshape(-1)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or np.any(a<0):raise ContractError('Invalid controller-only timing samples')
    return {'unit':'seconds','samples':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),
        'minimum':float(a.min()),'maximum':float(a.max()),'std':float(a.std()),'raw_seconds':a.tolist()}


def _benchmark_python(expert,policy,columns,reference_result,settings,context):
    arrays=reference_result['arrays'];diagnostics=reference_result['diagnostics'];count=len(arrays['time'])
    if not count:return {'status':'not_applicable','reason':'No valid expert operating contexts for comparable timings'}
    contexts=[]
    for index in range(min(count,settings['repeats'])):
        obs=arrays['observation'][index].copy();r=arrays['reference'][index].copy();d=arrays['d'][index].copy();p=arrays['previous_u'][index].copy()
        args=[obs,r if expert.spec['reference_available'] else np.empty(0),d if expert.spec['disturbance_available'] else np.empty(0),
            p if expert.spec['previous_action_available'] else np.empty(0),copy.deepcopy(diagnostics[index]['memory_before']),copy.deepcopy(expert.spec['parameters'])]
        contexts.append((args,(obs.copy(),r.copy(),d.copy(),p.copy())))
    expert.reset();timings={'expert':[],'imitation':[]}
    def invoke(which,c):
        args,physical=c
        if which=='expert':return expert.function(*args)
        obs,r,d,p=physical
        features=_feature_vector(columns,obs,r,d,p)
        return policy.predict(features,previous_action=p)
    def validate(which,result,c):
        i=expert.interface;previous=c[1][3]
        if which=='expert':
            if type(result) is not tuple or len(result)!=3 or result[2]!='ok':raise ContractError('Raw expert benchmark did not return a successful explicit result')
            raw=np.asarray(result[0]);action=np.empty(i.nu)
            if raw.dtype.kind not in 'iuf' or raw.shape!=(i.nu,) or not np.isfinite(raw).all():raise ContractError('Raw expert benchmark action has an invalid finite width/type')
            for item in expert.spec['action_mapping']:action[item['plant_control_index']]=raw[item['action_index']]
            if not isinstance(result[1],dict):raise ContractError('Raw expert benchmark memory is not explicit JSON')
            try:json.dumps(result[1],allow_nan=False)
            except (ValueError,TypeError) as error:raise ContractError('Raw expert benchmark memory is nonfinite') from error
        else:action=np.asarray(result)
        _bounded(action,i.controls,'timed controller action')
        if expert.spec.get('input_rate_bounds') is not None:
            rate=np.asarray(expert.spec['input_rate_bounds']);delta=action-previous
            if np.any(delta<rate[:,0]-1e-12) or np.any(delta>rate[:,1]+1e-12):raise ContractError('Timed controller action violates its per-update rate constraint')
    with expert.runtime.active():
        for c in contexts:
            # Resolve lazy snapshot helpers before timing any representative input.
            validate('expert',invoke('expert',copy.deepcopy(c)),c);validate('imitation',invoke('imitation',c),c)
        # A source may load SciPy/another supported numerical library lazily.
        # Re-enter limits AFTER all representative contexts have been preflighted,
        # so a library loaded after the outer rollout context is constrained too.
        with threadpoolctl.threadpool_limits(limits=1):
            pools_before=threadpoolctl.threadpool_info()
            if any(pool.get('num_threads')!=1 for pool in pools_before):raise ContractError('A benchmark numerical runtime did not honor one-thread limits')
            for which in timings:
                for index in range(settings['warmup']):context.check();c=copy.deepcopy(contexts[index%len(contexts)]);validate(which,invoke(which,c),c)
                for index in range(settings['repeats']):
                    context.check();c=copy.deepcopy(contexts[index%len(contexts)]);start=time.perf_counter_ns();result=invoke(which,c);elapsed=time.perf_counter_ns()-start
                    validate(which,result,c);timings[which].append(elapsed/1e9)
            pools_after=threadpoolctl.threadpool_info()
            if any(pool.get('num_threads')!=1 for pool in pools_after):raise ContractError('A timed numerical runtime changed its one-thread setting')
    result={name:_stats(values) for name,values in timings.items()};median=result['imitation']['median']
    return {'status':'completed',**result,'settings':settings,'speedup_median':None if median<=0 else result['expert']['median']/median,
        'scope':'Single raw expert function call vs NumPy-facing NN predict including current feature assembly, tensor conversion, normalization and amplitude/rate projection; imports, S09 repeated/fresh validation, source observation preparation and Plant simulation excluded.',
        'input_contexts':'Same recorded expert held-out physical contexts; prepared outside timing, explicit memory before that decision',
        'not_a_deployment_latency_guarantee':True,'hardware':{'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
            'python':platform.python_version(),'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'threadpoolctl':threadpoolctl.__version__,
            'threadpools':pools_after,'threadpools_before_timing':pools_before}}


def _metrics(result,physical,plan,settings):
    from .objectives import normalize_evaluation,trajectory_metrics
    definition=normalize_evaluation({'recipe':'tracking','scales':{'rmse':1,'energy':1,'tv':1,'overshoot':1},'metrics':settings})
    metric_interface=ModelInterface.parse({**physical.to_dict(),'sample_time':plan['control_period']})
    arrays=result['arrays'];records=[]
    for summary in result['report']['trajectories']:
        trajectory=next(row for row in plan['trajectories'] if row['trajectory_id']==summary['trajectory_id']);indices=np.flatnonzero(arrays['trajectory_id']==summary['trajectory_id'])
        if not len(indices):records.append({'trajectory_id':summary['trajectory_id'],'status':summary['status'],'metrics':None});continue
        rows=[{'step':k,'x':arrays['x'][n].tolist(),'next_x':arrays['next_x'][n].tolist(),'u':arrays['u'][n].tolist(),'output':arrays['next_y'][n].tolist(),
            'reference':arrays['reference'][n].tolist()} for k,n in enumerate(indices)]
        run={'scenario':{'initial_state':trajectory['x0']},'trajectory':rows,'initial_output':arrays['plant_y'][indices[0]].tolist(),
            'initial_control':trajectory['initial_action'],'input_rate_bounds':result.get('rate_bounds')}
        records.append({'trajectory_id':summary['trajectory_id'],'status':summary['status'],'metrics':trajectory_metrics(run,metric_interface,definition)})
    return records


def _publish(workspace,selection,config,report,results=None):
    parents=[selection['policy_id'],selection['dataset_id']]
    if selection.get('applicable'):parents.extend([selection['expert_id'],selection['plant_id']])
    with workspace.stage('policy_evaluation',MaterialInterface('bundle','Original-plant expert/imitation evaluation'),list(dict.fromkeys(parents)),config,compatible_parents=False) as stage:
        if results:
            for name,result in results.items():np.savez_compressed(stage.root/(name+'-rollout.npz'),**result['arrays']);atomic_json(stage.root/(name+'-diagnostics.json'),result['diagnostics'])
        atomic_json(stage.root/'report.json',report)
        valid=report['status'] in {'completed','not_applicable'}
        return stage.publish({'status':'verified' if valid else 'rejected','scope':'evaluation_report_only','candidate_feasible':report.get('candidate_feasible'),
            'policy_replacement':'not_performed','stability_proof':False},source={'kind':'external_policy_evaluation','policy':selection['policy_id'],
            'external_expert_dataset':selection['dataset_id'],'expert':selection.get('expert_id'),'plant':selection.get('plant_id')},activate=False)


def evaluate_external_imitation(workspace,policy_id,config,context=None,request=None):
    context=context or OperationContext();context.check()
    if not isinstance(config,dict) or set(config)-{'scenarios','benchmark','metrics','plan_refs'}:raise ContractError('Unknown original-plant evaluation setting')
    from .objectives import normalize_evaluation
    normalize_evaluation({'recipe':'tracking','scales':{'rmse':1,'energy':1,'tv':1,'overshoot':1},'metrics':config.get('metrics',{})})
    settings=benchmark_config(config.get('benchmark'));selection=_selection(workspace,policy_id)
    base={'schema_version':1,'policy':policy_id,'external_expert_dataset':selection['dataset_id'],'teacher_forced_offline':'separate training report, not autonomous closed-loop evidence',
        'policy_activation':'not_changed','policy_replacement':'not_performed','stability_proof':False,'confirmation_sources':context.case_confirmation_sources}
    def not_applicable(reason):return _publish(workspace,selection,config,{**base,'status':'not_applicable','reason':reason,
        'closed_loop':{'status':'not_applicable','plant_execution':'not_run'},'controller_benchmark':{'status':'not_applicable','expert_timing':'not_inferred_from_recorded_actions'}})
    if not selection['applicable']:return not_applicable(selection['reason'])
    columns,reason=_features(selection)
    if columns is None:return not_applicable(reason)
    plan,reason=_evaluation_plan(selection,config)
    if plan is None:return not_applicable(reason)
    if selection['expert']['backend']=='matlab_function':return _matlab_evaluation(workspace,selection,plan,columns,settings,config,context,base)
    from .external_policy import load_external_policy
    from .expert_adapter import load_external_expert
    from .model_adapters import load_python_adapter
    from .plants import _TorchAdapter
    model=load_external_policy(selection['policy_root']);expert=load_external_expert(selection['expert_root'],selection['expert'],expected_file_hashes=selection['expert_material']['files'])
    def load_plant():return _TimedPlant(load_python_adapter(selection['plant_root'],selection['plant_spec'],expected_file_hashes=selection['plant_material']['files'])
        if selection['plant_spec'] is not None else _TorchAdapter(workspace,selection['plant_id']))
    old_threads=torch.get_num_threads();torch.set_num_threads(1);results={};benchmark=None
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            for name,controller in [('expert',expert),('imitation',_PolicyAdapter(model,selection['expert'],columns))]:
                plant=load_plant()
                try:
                    try:result=_collect_python(plant,controller,plan,context)
                    except ExpertCollectionCancelled:raise
                    except ExpertCollectionStopped as error:result={'status':error.report['status'],'arrays':error.arrays,'report':error.report,'diagnostics':error.diagnostics}
                    result['rate_bounds']=selection['expert'].get('input_rate_bounds');result['plant_timing']=plant.describe();results[name]=result
                finally:plant.close()
            try:benchmark=_benchmark_python(expert,model,columns,results['expert'],settings,context)
            except Cancelled:raise
            except Exception as error:benchmark={'status':'failed','reason':str(error),'error_type':type(error).__name__,'speedup_median':None}
    except Exception as error:
        if isinstance(error,ExpertCollectionStopped):
            results[name]={'status':error.report['status'],'arrays':error.arrays,'report':error.report,'diagnostics':error.diagnostics}
        _archive_failure(workspace,selection,config,base,plan,results,error)
        raise
    finally:expert.close();torch.set_num_threads(old_threads)
    return _finish(workspace,selection,config,context,base,plan,results,benchmark)


def _archive_failure(workspace,selection,config,base,plan,results,error):
    parents=[selection[key] for key in ['policy_id','dataset_id','expert_id','plant_id']]
    with workspace.stage('policy_evaluation',MaterialInterface('bundle','Cancelled/failed original-Plant evaluation'),parents,config,compatible_parents=False) as stage:
        for name,result in results.items():
            np.savez_compressed(stage.root/(name+'-partial-rollout.npz'),**result['arrays']);atomic_json(stage.root/(name+'-diagnostics.json'),result['diagnostics'])
        atomic_json(stage.root/'report.json',{**base,'status':'cancelled' if isinstance(error,Cancelled) else 'failed','plan':plan,
            'failure':{'type':type(error).__name__,'message':str(error)},'valid_prefixes':{name:result['report'] for name,result in results.items()},
            'controller_benchmark':{'status':'not_completed'},'recovery':'Evaluation has no training checkpoint; rerun the unchanged confirmed conditions explicitly.'})
        raise error


def _finish(workspace,selection,config,context,base,plan,results,benchmark):
    try:context.check()
    except Exception as error:_archive_failure(workspace,selection,config,base,plan,results,error)
    metrics={name:_metrics(result,selection['physical'],plan,config.get('metrics',{})) for name,result in results.items()}
    complete=all(result['status']=='completed' for result in results.values()) and benchmark.get('status') in {'completed','not_applicable'};by_id={name:{row['trajectory_id']:row for row in records} for name,records in metrics.items()};comparisons=[]
    for trajectory in plan['trajectories']:
        left=by_id['expert'].get(trajectory['trajectory_id']);right=by_id['imitation'].get(trajectory['trajectory_id']);available=left and right and left['metrics'] and right['metrics']
        comparison={'trajectory_id':trajectory['trajectory_id'],'comparison_complete':bool(available and left['status']=='completed' and right['status']=='completed')}
        if available:comparison['imitation_minus_expert']={key:None if left['metrics'][key] is None or right['metrics'][key] is None else right['metrics'][key]-left['metrics'][key] for key in ['rmse','energy','tv','overshoot']}
        comparisons.append(comparison)
    feasible=results['imitation']['status']=='completed' and all(row['metrics'] is not None and row['metrics']['hard_constraint_violations']==0 for row in metrics['imitation'])
    report={**base,'status':'completed' if complete else 'partial','expert':selection['expert_id'],'plant':selection['plant_id'],
        'condition_source':plan['condition_source'],'independent_of_training':plan['independent_of_training'],'plan':plan,
        'original_plant':'exact bound dataset ancestor, no unrelated active replacement','physical_interface':selection['physical'].to_dict(),
        'closed_loop':{name:{'status':result['status'],'report':result['report'],'metrics':metrics[name],'plant_timing':result.get('plant_timing'),
            'feedback':'own past applied actions and own plant states; no teacher forcing','rollout_file':name+'-rollout.npz'} for name,result in results.items()},
        'comparison':comparisons,'candidate_feasible':bool(feasible),'physical_degradation_reported':True,'effect_improvement_required':False,
        'degradation':{'constraint_or_rollout_failure':not bool(feasible),'raw_tracking_rmse_increase_on_conditions':[row['trajectory_id'] for row in comparisons
            if row.get('imitation_minus_expert',{}).get('rmse') is not None and row['imitation_minus_expert']['rmse']>0],
            'interpretation':'Raw measured differences, not a statistical-significance or improvement guarantee'},
        'controller_benchmark':benchmark,'metric_scoring':'raw physical/coordinate diagnostics only; no new preference optimization or test-driven policy selection',
        'action_rate_semantics':'per control update; hold across original plant subticks'}
    for name in [selection['policy_id'],selection['dataset_id'],selection['expert_id'],selection['plant_id']]:workspace.resolve(name)
    return _publish(workspace,selection,config,report,results)


def _version(workspace,metadata):return {'id':metadata['id'],'metadata_hash':workspace.manifest['artifacts'][metadata['id']]['metadata_hash']}


def _matlab_adapter(selection):
    from .expert_adapter import validate_expert_spec
    bundle=selection['policy_root']/'bundle';manifest=read_json(bundle/'policy_manifest.json');matlab=manifest.get('matlab',{})
    if matlab.get('status')!='supported':return None,matlab.get('reason','This policy has no supported autonomous numeric MATLAB adapter.')
    source=matlab.get('source_file')
    if not isinstance(source,str) or Path(source).is_absolute() or '..' in Path(source).parts:raise ContractError('Policy MATLAB source must be a verified bundle member')
    if 'bundle/'+source.replace('\\','/') not in selection['policy_meta']['files']:raise ContractError('Policy MATLAB source is not bound to the published policy artifact')
    adapter=validate_expert_spec({**matlab['adapter'],'source_file':'bundle/'+source.replace('\\','/')})
    selection['physical'].compatible(ModelInterface.parse(adapter['interface']))
    if adapter['control_period']!=selection['expert']['control_period']:raise ContractError('Policy and original expert control periods disagree')
    if adapter.get('input_rate_bounds')!=selection['expert'].get('input_rate_bounds'):raise ContractError('Policy MATLAB action-rate constraints differ from their original source')
    if any(row['action_index']!=row['plant_control_index'] for row in adapter['action_mapping']):raise ContractError('Imitation targets and MATLAB return actions must remain in Plant order')
    return adapter,None


def _matlab_evaluation(workspace,selection,plan,columns,settings,config,context,base):
    from .matlab_expert import prepare_matlab_expert_manifest,load_matlab_expert_batch
    adapter,reason=_matlab_adapter(selection)
    if adapter is None:return _publish(workspace,selection,config,{**base,'status':'not_applicable','reason':reason,
        'closed_loop':{'status':'not_applicable','plant_execution':'not_run'},'controller_benchmark':{'status':'not_applicable'}})
    phase=os.environ.get('CONTROL_MATLAB_PHASE');raw=context._case_raw_request;digest=os.environ.get('CONTROL_MATLAB_REQUEST_SHA256','');nonce=os.environ.get('CONTROL_MATLAB_NONCE','')
    if not context._case_receipt_body or phase not in {'prepare','finalize'} or not raw or hashlib.sha256(raw.encode()).hexdigest()!=digest or not re.fullmatch(r'[a-f0-9-]{36}',nonce):
        raise ContractError('Original MATLAB policy evaluation requires one exact-request-bound native Host batch, never Engine or pointwise MCP')
    exchange=Path(os.environ.get('CONTROL_MATLAB_EXCHANGE_DIR','')).resolve()
    if exchange.parent!=(workspace.path/'.case-plans/matlab-exchanges').resolve() or exchange.name!=nonce:raise ContractError('MATLAB policy evaluation exchange escaped its fixed request directory')
    versions={name:_version(workspace,selection[key]) for name,key in [('policy_version','policy_meta'),('dataset_version','dataset_meta'),
        ('expert_version','expert_meta'),('expert_material_version','expert_material'),('plant_version','plant_meta'),('plant_material_version','plant_material')]}
    nnplan=copy.deepcopy(plan);nnplan['expert_spec_hash']=canonical_hash(adapter);nnplan.pop('plan_hash');nnplan['plan_hash']=canonical_hash(nnplan)
    spec_hash=canonical_hash({'expert':selection['expert'],'imitation':adapter,'plant':selection['plant_spec'],'benchmark':settings,**versions})
    expected={'schema_version':1,'nonce':nonce,'request_sha256':digest,'spec_hash':spec_hash,**versions}
    common={'request_sha256':digest,'nonce':nonce,'plant_version':versions['plant_version'],'plant_material_version':versions['plant_material_version']}
    child_versions={'expert':{'expert_version':versions['expert_version'],'expert_material_version':versions['expert_material_version']},
        'imitation':{'expert_version':versions['policy_version'],'expert_material_version':versions['policy_version']}}
    with workspace.lock('evaluate_external_imitation'):
        if phase=='prepare':
            exchange.mkdir(parents=True,exist_ok=False);children={}
            for name,spec,source,subplan in [('expert',selection['expert'],selection['expert_root'],plan),('imitation',adapter,selection['policy_root'],nnplan)]:
                children[name]=prepare_matlab_expert_manifest(spec,source,selection['plant_spec'],selection['plant_root'],subplan,
                    exchange_dir=exchange/name,**common,**child_versions[name])
            document={**expected,'kind':'control-matlab-policy-evaluation-v1','plan':plan,'nn_plan':nnplan,'benchmark':settings,'children':children,
                'result_path':str(exchange/'result.json'),'batch_kind':'external_policy_evaluation'}
            path=exchange/'manifest.json';atomic_json(path,document)
            for key in ['policy_id','dataset_id','expert_id','plant_id']:workspace.resolve(selection[key])
            return {'status':'verified','matlab_phase':'prepared','manifest_path':str(path),'manifest_sha256':file_hash(path),'exchange_dir':str(exchange),
                'result_path':document['result_path'],'request_sha256':digest,'nonce':nonce,'plan_hash':plan['plan_hash'],'batch_kind':'external_policy_evaluation'}
        path=Path(os.environ.get('CONTROL_MATLAB_RESULT_PATH','')).resolve();sha=os.environ.get('CONTROL_MATLAB_RESULT_SHA256','')
        if path!=exchange/'result.json' or not re.fullmatch(r'[a-f0-9]{64}',sha) or path.stat().st_size>256*1024*1024 or file_hash(path)!=sha:
            raise ContractError('MATLAB policy evaluation did not return the Host-observed fixed result bytes')
        document=read_json(exchange/'manifest.json');result=read_json(path)
        if (document.get('kind')!='control-matlab-policy-evaluation-v1' or result.get('kind')!='control-matlab-policy-evaluation-result-v1' or
                any(document.get(k)!=v or result.get(k)!=v for k,v in expected.items()) or document.get('plan')!=plan or document.get('nn_plan')!=nnplan or
                document.get('benchmark')!=settings or result.get('plan_hash')!=plan['plan_hash'] or document.get('result_path')!=str(path)):
            raise ContractError('MATLAB evaluation result/manifest/source versions/conditions no longer match the confirmed request')
        if result.get('status') not in {'completed','partial','failed'}:raise ContractError('MATLAB evaluation requires an explicit native terminal status')
        results={}
        for name,spec,subplan in [('expert',selection['expert'],plan),('imitation',adapter,nnplan)]:
            prepared_child=document.get('children',{}).get(name,{})
            child_manifest=exchange/name/'manifest.json'
            if prepared_child.get('manifest_path')!=str(child_manifest) or prepared_child.get('manifest_sha256')!=file_hash(child_manifest):
                raise ContractError('A child MATLAB evaluation manifest changed after preparation')
            child=exchange/name/'result.json';expected_sha=result.get(name+'_result_sha256')
            if not isinstance(expected_sha,str) or file_hash(child)!=expected_sha:raise ContractError('MATLAB evaluation child result changed after the native batch')
            results[name]=load_matlab_expert_batch(spec,selection['plant_spec'],subplan,result_path=child,result_sha256=expected_sha,**common,**child_versions[name])
            results[name]['rate_bounds']=selection['expert'].get('input_rate_bounds')
            results[name]['plant_timing']={'status':'not_separately_instrumented','included_in_controller_benchmark':False,
                'closed_loop_batch_seconds':result.get(name+'_batch_seconds'),'batch_scope':'Whole S09 guarded loop including expert repeated validation and original Plant; not a pure Plant latency.'}
        benchmark=result.get('controller_benchmark')
        if not isinstance(benchmark,dict):raise ContractError('Native policy evaluation omitted controller-only benchmark evidence')
        if benchmark.get('status')=='completed':
            if benchmark.get('settings')!=settings or benchmark.get('matlab_max_computational_threads')!=1:raise ContractError('Native controller timing did not use its frozen one-thread/batch-one settings')
            for name in ['expert','imitation']:
                values=benchmark.get(name,{}).get('raw_seconds');computed=_stats(values)
                if computed['samples']!=settings['repeats']:raise ContractError('Native timing repeats differ from the confirmed benchmark')
                # Recompute all statistics in Python; MATLAB's percentile method
                # is intentionally not used as its own p95 acceptance oracle.
                benchmark[name]=computed
            nnmedian=benchmark['imitation']['median'];benchmark['speedup_median']=None if nnmedian<=0 else benchmark['expert']['median']/nnmedian
        elif benchmark.get('status') not in {'not_applicable','failed'}:raise ContractError('Native controller benchmark has no explicit terminal status')
        completed=all(child['status']=='completed' for child in results.values()) and benchmark['status']!='failed'
        expected_status='completed' if completed else ('partial' if any(len(child['arrays']['time']) for child in results.values()) else 'failed')
        if result['status']!=expected_status:raise ContractError('Native evaluation terminal status contradicts its child valid prefixes or timing result')
    return _finish(workspace,selection,config,context,base,plan,results,benchmark)
