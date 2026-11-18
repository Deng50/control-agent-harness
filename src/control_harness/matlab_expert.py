"""Authenticated one-native-call exchanges for an external MATLAB expert and plant.

This module never imports MATLAB Engine, executes a user source, or publishes a
trainable artifact. The native Host owns dispatch/identity/permissions; the
external collector owns confirmed features, predetermined splits and publishing.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat

import numpy as np

from .contracts import ContractError,ModelInterface,canonical_hash
from .matlab_adapter import (_json,_relative,_name,_sha,_version,_native_numeric,
    validate_matlab_spec,validate_simulink_source)
from .matlab_source import validate_matlab_sources

_EXPERT_BUILTINS=('quadprog','optimoptions','speye','spdiags')
_SHA=re.compile(r'[a-f0-9]{64}')
_MAX_BYTES=256*1024*1024


def validate_matlab_expert_source(root,spec,expected_files):
    """Static six-input/three-output supported-subset check; never execution."""
    if not isinstance(spec,dict) or spec.get('backend')!='matlab_function':
        raise ContractError('An explicit matlab_function external expert is required')
    source=_relative(spec.get('source_file'));entry=_name(spec.get('entrypoint'),'expert entrypoint',function=True)
    suffix=[f'+{part}' for part in entry.split('.')[:-1]]+[entry.split('.')[-1]+'.m']
    if list(Path(source).parts[-len(suffix):])!=suffix:
        raise ContractError('Expert MATLAB entrypoint/package must match the frozen .m source')
    if entry in {'control_harness_expert_batch','control_harness_collect_batch'}:
        raise ContractError('Expert entrypoint cannot shadow a trusted batch helper')
    return validate_matlab_sources(root,spec,expected_files,signature=(6,3),extra_numeric_builtins=_EXPERT_BUILTINS)


def _specs(expert_spec,plant_spec):
    # Common validation is shared with the Python expert route, not inferred
    # from MATLAB source or a preview. Import is local to avoid a module cycle.
    from .expert_adapter import validate_expert_spec
    expert=validate_expert_spec(expert_spec)
    if expert['backend']!='matlab_function':raise ContractError('This batch requires a MATLAB external expert')
    plant=validate_matlab_spec(plant_spec)
    interface=ModelInterface.parse(plant['interface']);interface.compatible(ModelInterface.parse(expert['interface']))
    if plant['backend']=='simulink':
        reserved={plant['simulink']['sample_time_variable'],*[row['initial_condition_variable'] for row in plant['simulink']['states']]}
        if reserved&set(plant['parameters']):raise ContractError('SLX parameters cannot override approved dt/x0 variables')
    return expert,plant,interface


def _vector(value,width,label):
    raw=_native_numeric(value,label,max_rank=1)
    if raw.shape!=(width,) or not np.isfinite(raw).all():raise ContractError(label+' must be an explicit finite vector of width '+str(width))
    return raw.astype(float)


def _bounded(value,variables,label):
    if any(v<item.lower-1e-12 or v>item.upper+1e-12 for v,item in zip(value,variables)):
        raise ContractError(label+' is outside the declared physical bounds')


def _plan(plan,expert,plant,interface):
    value=_json(plan,'external expert collection plan')
    if not isinstance(value,dict) or type(value.get('schema_version')) is not int or value['schema_version']!=1:
        raise ContractError('An explicit version-1 external expert plan is required')
    interface.compatible(ModelInterface.parse(value.get('interface',{})))
    period=value.get('control_period');substeps=value.get('plant_substeps')
    if (type(period) not in (int,float) or not math.isfinite(period) or period<=0 or
            type(substeps) is not int or not 1<=substeps<=10000 or
            not math.isclose(period,substeps*interface.sample_time,rel_tol=0,abs_tol=1e-12) or period!=expert['control_period']):
        raise ContractError('Control period must equal the declared integer plant-step multiple')
    if plant['backend']=='simulink' and (substeps+1)*(interface.nx+interface.ny+interface.nu+interface.nd+1)>5_000_000:
        raise ContractError('One SLX control-interval log exceeds its 5e6-value memory budget')
    boundary=value.get('resource_boundary',{})
    limit=boundary.get('max_diagnostic_bytes',8*1024*1024) if isinstance(boundary,dict) else None
    if type(limit) is not int or not 1<=limit<=8*1024*1024:raise ContractError('Expert diagnostic storage must respect the common per-batch 8MiB limit')
    trajectories=value.get('trajectories')
    if not isinstance(trajectories,list) or not trajectories:raise ContractError('Explicit independent trajectories are required')
    scalar_count=0;ids=[];split={'train':[],'validation':[],'test':[]}
    width=2*interface.nx+2*interface.ny+3*interface.nu+interface.nd+len(expert['observations'])+interface.ny+3
    for row in trajectories:
        if (not isinstance(row,dict) or type(row.get('trajectory_id')) is not int or row['trajectory_id'] in ids or
                type(row.get('steps')) is not int or row['steps']<1 or row.get('split') not in split):
            raise ContractError('Trajectory IDs/counts/split must be explicit and unique')
        ids.append(row['trajectory_id']);split[row['split']].append(row['trajectory_id']);scalar_count+=row['steps']*width
        if scalar_count>5_000_000:raise ContractError('Single external-expert batch exceeds its 5e6-value expansion budget; collect in batches')
        x=_vector(row.get('x0'),interface.nx,'initial state');u=_vector(row.get('initial_action'),interface.nu,'initial action')
        d=_vector(row.get('disturbance'),interface.nd,'disturbance');_vector(row.get('reference'),interface.ny,'reference')
        _bounded(x,interface.states,'initial state');_bounded(u,interface.controls,'initial action');_bounded(d,interface.disturbances,'disturbance')
        times=_vector(row.get('time'),row['steps'],'control time grid')
        if not np.allclose(times,np.arange(row['steps'])*period,atol=1e-12,rtol=0):raise ContractError('Expert time grid differs from its explicit control period')
    if value.get('split')!=split:raise ContractError('Predetermined trajectory split differs from the declared trajectory roles')
    if value.get('plan_hash')!=canonical_hash({key:item for key,item in value.items() if key!='plan_hash'}):
        raise ContractError('External expert plan hash does not cover its complete definition')
    return value


def _snapshot(root,destination,material_version,spec,expert=False):
    root=Path(root).resolve(strict=True);metadata_bytes=(root/'artifact.json').read_bytes();metadata=json.loads(metadata_bytes.decode('utf-8'))
    if metadata.get('id')!=material_version['id'] or canonical_hash(metadata)!=material_version['metadata_hash'] or not isinstance(metadata.get('files'),dict):
        raise ContractError('MATLAB source is not the bound immutable material artifact')
    files=dict(metadata['files']);files['artifact.json']=hashlib.sha256(metadata_bytes).hexdigest()
    support=(validate_matlab_expert_source(root,spec,metadata['files']) if expert else
        validate_matlab_sources(root,spec,metadata['files']) if spec['backend']=='matlab_function' else validate_simulink_source(root/spec['source_file'],spec))
    destination.mkdir();inventory={};total=0
    for source in sorted(root.rglob('*')):
        info=source.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:raise ContractError('MATLAB source snapshots cannot contain filesystem aliases')
        if source.is_dir():continue
        relative=source.relative_to(root).as_posix()
        if not stat.S_ISREG(info.st_mode) or relative not in files or info.st_size>64*1024*1024:
            raise ContractError('MATLAB material has an unregistered/unsupported file')
        total+=info.st_size
        if len(inventory)>=2048 or total>_MAX_BYTES:raise ContractError('MATLAB source snapshot exceeds its file/byte budget')
        before=_sha(source)
        if before!=files[relative]:raise ContractError('MATLAB source differs from its bound artifact bytes')
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if _sha(source)!=before or _sha(target)!=before:raise ContractError('MATLAB source changed during snapshot')
        inventory[relative]=before
    if inventory!=files or spec['source_file'] not in inventory:raise ContractError('MATLAB source inventory changed or entrypoint is absent')
    source=destination/spec['source_file']
    result={'spec':spec,'source_root':str(destination),'source_file':str(source),'source_sha256':inventory[spec['source_file']],
        'source_inventory':inventory,'source_files':[{'relative_path':name,'sha256':digest} for name,digest in inventory.items()],'source_validation':support}
    if spec['backend']=='matlab_function':result['function_path_root']=str(source.parents[len(spec['entrypoint'].split('.'))-1])
    return result


def validate_coverage_domain(domain,interface):
    """Optional trusted per-Plant-subtick box; never inferred from dimensions."""
    if domain is None:return None
    if not isinstance(domain,dict) or set(domain)!={'x','u','d'}:raise ContractError('Coverage domain must explicitly contain x/u/d boxes')
    result={}
    for key,variables in [('x',interface.states),('u',interface.controls),('d',interface.disturbances)]:
        value=_native_numeric(domain[key],'coverage domain '+key)
        if not variables and value.size==0:value=np.empty((0,2))
        if value.shape!=(len(variables),2) or not np.isfinite(value).all() or np.any(value[:,0]>=value[:,1]):raise ContractError('Invalid coverage box '+key)
        physical=np.asarray([[v.lower,v.upper] for v in variables]).reshape(-1,2)
        if np.any(value[:,0]<physical[:,0]) or np.any(value[:,1]>physical[:,1]):raise ContractError('Coverage box exceeds original physical bounds: '+key)
        result[key]=value.astype(float).tolist()
    return result


def prepare_matlab_expert_manifest(expert_spec,expert_source_root,plant_spec,plant_source_root,plan,*,exchange_dir,
        request_sha256,nonce,expert_version,expert_material_version,plant_version,plant_material_version,coverage_domain=None):
    """Private native prepare phase. No source code or MATLAB is executed."""
    expert,plant,interface=_specs(expert_spec,plant_spec);plan=_plan(plan,expert,plant,interface)
    if not isinstance(request_sha256,str) or not _SHA.fullmatch(request_sha256) or not isinstance(nonce,str) or not re.fullmatch(r'[a-f0-9-]{36}',nonce):
        raise ContractError('Host-bound request SHA and random batch nonce are required')
    versions={name:_version(value,name) for name,value in [('expert_version',expert_version),('expert_material_version',expert_material_version),
        ('plant_version',plant_version),('plant_material_version',plant_material_version)]}
    exchange=Path(exchange_dir)
    if not exchange.is_absolute() or any(exchange.resolve().is_relative_to(Path(root).resolve()) for root in (expert_source_root,plant_source_root)):
        raise ContractError('Private exchange must be absolute and outside both immutable source trees')
    exchange.mkdir(parents=True,exist_ok=False)
    expert_source=_snapshot(expert_source_root,exchange/'expert-input',versions['expert_material_version'],expert,True)
    plant_source=_snapshot(plant_source_root,exchange/'plant-input',versions['plant_material_version'],plant)
    collisions=set(expert_source['source_validation'].get('function_inventory',[]))&set(plant_source['source_validation'].get('function_inventory',[]))
    if collisions:raise ContractError('Expert and plant function namespaces collide; use explicit distinct thin adapters')
    manifest={'schema_version':1,'kind':'control-matlab-expert-batch-v1','nonce':nonce,'request_sha256':request_sha256,**versions,
        'spec_hash':canonical_hash({'expert':expert,'plant':plant}),'expert':expert_source,'plant':plant_source,'plan':plan,
        'result_path':str(exchange/'result.json'),'batch_kind':'external_expert'}
    domain=validate_coverage_domain(coverage_domain,interface)
    if domain is not None:manifest.update(coverage_domain=domain,coverage_guard_hash=canonical_hash(domain))
    path=exchange/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return {'status':'verified','matlab_phase':'prepared','manifest_path':str(path),'manifest_sha256':_sha(path),'exchange_dir':str(exchange),
        'result_path':manifest['result_path'],'request_sha256':request_sha256,'nonce':nonce,'plan_hash':plan['plan_hash'],'backend':plant['backend'],'batch_kind':'external_expert'}


def _matrix(raw,key,count,width):
    try:
        value=_native_numeric(raw[key],key)
        if value.ndim==2 and value.shape!=(count,width):raise ValueError('matrix shape')
        result=value.astype(float).reshape(count,width)
    except (KeyError,ValueError,TypeError) as error:raise ContractError('Native expert '+key+' has the wrong shape') from error
    if not np.isfinite(result).all():raise ContractError('Native expert '+key+' contains nonfinite values')
    return result


def load_matlab_expert_batch(expert_spec,plant_spec,plan,*,result_path,result_sha256,request_sha256,nonce,
        expert_version,expert_material_version,plant_version,plant_material_version,coverage_domain=None):
    """Authenticate terminal results; partial returns remain explicitly failed prefixes."""
    expert,plant,interface=_specs(expert_spec,plant_spec);plan=_plan(plan,expert,plant,interface)
    path=Path(result_path)
    if not path.is_absolute() or path.name!='result.json' or not _SHA.fullmatch(result_sha256 or '') or path.stat().st_size>_MAX_BYTES:
        raise ContractError('Expert result must be this Host-observed bounded exchange file')
    raw_bytes=path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest()!=result_sha256:raise ContractError('Native expert result differs from the Host-observed bytes')
    result=json.loads(raw_bytes.decode('utf-8'));manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
    domain=validate_coverage_domain(coverage_domain,interface)
    if domain is None:
        if manifest.get('coverage_domain') is not None:raise ContractError('Unexpected unrequested coverage guard in native expert manifest')
    elif manifest.get('coverage_domain')!=domain or manifest.get('coverage_guard_hash')!=canonical_hash(domain):raise ContractError('Native expert coverage domain differs from the bound request')
    versions={name:_version(value,name) for name,value in [('expert_version',expert_version),('expert_material_version',expert_material_version),
        ('plant_version',plant_version),('plant_material_version',plant_material_version)]}
    bound={'schema_version':1,'nonce':nonce,'request_sha256':request_sha256,**versions,'spec_hash':canonical_hash({'expert':expert,'plant':plant})}
    if (result.get('kind')!='control-matlab-expert-batch-result-v1' or manifest.get('kind')!='control-matlab-expert-batch-v1' or
            any(result.get(key)!=value or manifest.get(key)!=value for key,value in bound.items()) or result.get('plan_hash')!=plan['plan_hash'] or
            manifest.get('plan')!=plan or Path(manifest.get('result_path','')).resolve()!=path.resolve()):
        raise ContractError('Native expert result/manifest is not bound to this request, source versions and plan')
    for name,spec in [('expert',expert),('plant',plant)]:
        source=manifest.get(name,{});root=(path.parent/(name+'-input')).resolve()
        if source.get('spec')!=spec or Path(source.get('source_root','')).resolve()!=root:raise ContractError('Expert source copy escaped its fixed exchange')
        inventory=source.get('source_inventory')
        if not isinstance(inventory,dict) or spec['source_file'] not in inventory or source.get('source_sha256')!=inventory[spec['source_file']]:raise ContractError('Expert source has no bound byte inventory')
        copied_metadata=(root/'artifact.json').read_bytes();copied=json.loads(copied_metadata.decode('utf-8'))
        material=versions[name+'_material_version'];expected_inventory={**copied.get('files',{}),'artifact.json':hashlib.sha256(copied_metadata).hexdigest()}
        if (copied.get('id')!=material['id'] or canonical_hash(copied)!=material['metadata_hash'] or inventory!=expected_inventory or
                Path(source.get('source_file','')).resolve()!=(root/spec['source_file']).resolve() or
                source.get('source_files')!=[{'relative_path':relative,'sha256':digest} for relative,digest in inventory.items()]):
            raise ContractError('Expert source paths/inventory no longer match the bound material version')
        for relative,digest in inventory.items():
            current=(root/_relative(relative)).resolve(strict=True)
            if not current.is_relative_to(root) or _sha(current)!=digest:raise ContractError('Expert/plant source changed during native execution')
    raw=result.get('arrays')
    if not isinstance(raw,dict):raise ContractError('Native expert arrays are required')
    ids=_native_numeric(raw.get('trajectory_id'),'trajectory_id',max_rank=1)
    if ids.size==0:ids=np.asarray([],dtype=np.int64)
    if ids.dtype.kind not in 'iu':raise ContractError('Native expert trajectory IDs must be integers')
    ids=ids.reshape(-1);count=len(ids);arrays={'trajectory_id':ids.astype(np.int64)}
    for key,width in {'observation':len(expert['observations']),'reference':interface.ny,'d':interface.nd,'previous_u':interface.nu,'u':interface.nu,
            'x':interface.nx,'next_x':interface.nx,'plant_y':interface.ny,'next_y':interface.ny}.items():arrays[key]=_matrix(raw,key,count,width)
    arrays['time']=_matrix(raw,'time',count,1).reshape(count)
    summaries=result.get('trajectories');diagnostics=result.get('diagnostics')
    if not isinstance(summaries,list) or not 1<=len(summaries)<=len(plan['trajectories']) or not isinstance(diagnostics,list) or len(diagnostics)!=count:
        raise ContractError('Native expert summaries/diagnostics differ from their valid row prefix')
    offset=0
    for index,summary in enumerate(summaries):
        expected=plan['trajectories'][index];steps=summary.get('completed_steps');status=summary.get('status')
        if (summary.get('trajectory_id')!=expected['trajectory_id'] or type(steps) is not int or not 0<=steps<=expected['steps'] or
                status!=('completed' if steps==expected['steps'] else 'partial' if steps else 'failed') or
                (status!='completed' and index!=len(summaries)-1)):
            raise ContractError('Native expert trajectory completion contradicts its ordered plan')
        end=offset+steps
        if end>count or not np.all(ids[offset:end]==expected['trajectory_id']):raise ContractError('Native expert rows are not an ordered trajectory prefix')
        if steps:
            for key,constant in [('reference',expected['reference']),('d',expected['disturbance'])]:
                if not np.allclose(arrays[key][offset:end],np.asarray(constant),rtol=0,atol=1e-12):raise ContractError('Native expert condition differs from the approved plan')
            if not np.allclose(arrays['time'][offset:end],expected['time'][:steps],rtol=0,atol=1e-12):raise ContractError('Native expert time differs from the approved control grid')
            if not np.allclose(arrays['x'][offset],expected['x0'],rtol=0,atol=1e-12) or not np.allclose(arrays['previous_u'][offset],expected['initial_action'],rtol=0,atol=1e-12):
                raise ContractError('Native expert did not reset complete state and previous action')
            if (not np.allclose(arrays['x'][offset+1:end],arrays['next_x'][offset:end-1],rtol=0,atol=1e-12) or
                    not np.allclose(arrays['plant_y'][offset+1:end],arrays['next_y'][offset:end-1],rtol=0,atol=1e-12) or
                    not np.allclose(arrays['previous_u'][offset+1:end],arrays['u'][offset:end-1],rtol=0,atol=1e-12)):
                raise ContractError('Native expert state/output/action history is discontinuous')
            for row in range(offset,end):
                for key,variables in [('x',interface.states),('next_x',interface.states),('u',interface.controls),('previous_u',interface.controls),('d',interface.disturbances),('plant_y',interface.outputs),('next_y',interface.outputs)]:_bounded(arrays[key][row],variables,key)
                observation=np.array([arrays['x' if item['source']=='state' else 'plant_y'][row,item['index']] for item in expert['observations']])
                if not np.allclose(arrays['observation'][row],observation,rtol=0,atol=1e-12):raise ContractError('Native observation contradicts the explicit available signal mapping')
                if expert.get('input_rate_bounds') is not None:
                    rate=np.asarray(expert['input_rate_bounds']);delta=arrays['u'][row]-arrays['previous_u'][row]
                    if np.any(delta<rate[:,0]-1e-12) or np.any(delta>rate[:,1]+1e-12):raise ContractError('Native expert action increment violates its physical contract')
                diagnostic=diagnostics[row]
                if not isinstance(diagnostic,dict) or diagnostic.get('trajectory_id')!=expected['trajectory_id'] or diagnostic.get('step_index')!=row-offset or diagnostic.get('status')!='ok':
                    raise ContractError('Native expert success/memory diagnostics do not bind each valid action')
                _json(diagnostic,'expert diagnostic memory')
                action=_native_numeric(diagnostic.get('expert_action'),'expert action',max_rank=1)
                if action.size!=interface.nu or not np.isfinite(action).all():raise ContractError('Native expert action has an invalid diagnostic shape')
                action=action.reshape(interface.nu);applied=np.zeros(interface.nu)
                for item in expert['action_mapping']:applied[item['plant_control_index']]=action[item['action_index']]
                if not np.allclose(applied,arrays['u'][row],atol=1e-12,rtol=0):raise ContractError('Native applied action contradicts the explicit expert action mapping')
                if not isinstance(diagnostic.get('memory_before'),dict) or not isinstance(diagnostic.get('memory_after'),dict):raise ContractError('Expert memory must remain an explicit finite JSON object')
                if any(len(json.dumps(diagnostic[name],ensure_ascii=False,allow_nan=False).encode('utf-8'))>65536 for name in ['memory_before','memory_after']):
                    raise ContractError('Expert diagnostic memory exceeds its 64KiB contract')
                before=expert['initial_memory'] if row==offset else diagnostics[row-1]['memory_after']
                if diagnostic['memory_before']!=before:raise ContractError('Expert memory did not reset/continue explicitly')
        offset=end
    if offset!=count:raise ContractError('Native expert summary count differs from its rows')
    diagnostic_limit=plan.get('resource_boundary',{}).get('max_diagnostic_bytes',8*1024*1024)
    if sum(len(json.dumps(item,ensure_ascii=False,allow_nan=False).encode('utf-8')) for item in diagnostics)>diagnostic_limit:
        raise ContractError('Native expert diagnostics exceed the approved per-batch memory budget')
    complete=len(summaries)==len(plan['trajectories']) and all(row['status']=='completed' for row in summaries)
    if (result.get('status')=='completed')!=complete or result.get('status') not in {'completed','partial','failed'}:
        raise ContractError('Native expert terminal status contradicts the actual complete trajectories')
    failure=result.get('failure')
    if complete:
        if failure not in (None,[]):raise ContractError('Complete native expert batch cannot include a failure')
    elif (not isinstance(failure,dict) or summaries[-1]['status']=='completed' or failure.get('trajectory_id')!=summaries[-1]['trajectory_id'] or
            failure.get('step_index')!=summaries[-1]['completed_steps'] or result['status']!=('partial' if count else 'failed')):
        raise ContractError('Native expert failure is not bound to its first invalid action prefix')
    if 'C' in plant:
        c=np.asarray(plant['C'])
        if not np.allclose(arrays['plant_y'],arrays['x']@c.T,atol=1e-12,rtol=1e-10) or not np.allclose(arrays['next_y'],arrays['next_x']@c.T,atol=1e-12,rtol=1e-10):raise ContractError('Native physical outputs contradict the declared C mapping')
    if domain is not None:
        guard=result.get('coverage_guard');fields={'schema_version','enabled','domain_hash','before_checks','after_checks','passed_subticks','violations','status'}
        if not isinstance(guard,dict) or set(guard)!=fields or guard['schema_version']!=1 or guard['enabled'] is not True or guard['domain_hash']!=canonical_hash(domain):raise ContractError('Native expert omitted or changed actual subtick coverage evidence')
        if any(type(guard[name]) is not int or guard[name]<0 for name in ['before_checks','after_checks','passed_subticks','violations']):raise ContractError('Native coverage counters must be explicit nonnegative integers')
        passed,before,after,violations=(guard[name] for name in ['passed_subticks','before_checks','after_checks','violations'])
        if (not count*plan['plant_substeps']<=passed<=(count+1)*plan['plant_substeps'] or not passed<=after<=before<=passed+1 or violations not in {0,1}
                or guard['status']!=('failed' if violations else 'passed')):raise ContractError('Native coverage counters contradict their valid physical prefix')
        if complete and (violations or before!=after or after!=passed or passed!=count*plan['plant_substeps']):raise ContractError('Complete native batch did not check every original Plant subtick')
        if violations and complete:raise ContractError('A coverage-violating native batch cannot report completed')
    report={key:copy.deepcopy(value) for key,value in result.items() if key not in {'arrays','diagnostics'}}
    report.update(source='external_matlab_expert_and_declared_plant',training='not_run',hardware_execution=False,features='assembled_by_common_external_collector_only')
    return {'status':result['status'],'arrays':arrays,'report':report,'diagnostics':diagnostics}
