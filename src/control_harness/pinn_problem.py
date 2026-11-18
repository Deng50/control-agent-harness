"""Immutable, static preparation of conditional thermal ODE reference inputs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .artifacts import atomic_json,canonical_hash,confined,read_json
from .contracts import ContractError,MaterialInterface
from .pinn_contract import (MAX_CONDITIONS,MAX_LOGICAL_VALUES,MAX_NPZ_BYTES,MAX_POINTS,SPLIT_IDS,
    pinn_signals,validate_pinn_config,validate_pinn_definition)
from .runtime import OperationContext


FLOAT_COLUMNS=('time_s','initial_temperature_K','power_W','temperature_K')
COLLOCATION_FLOATS=('collocation_time_s','collocation_initial_temperature_K','collocation_power_W')
INITIAL_FLOATS=('initial_time_s','initial_value_K','initial_power_W')
INTEGER_ARRAYS=('condition_index','split_id','collocation_condition_index','initial_condition_index','initial_split_id')
ARRAY_NAMES=set(FLOAT_COLUMNS+COLLOCATION_FLOATS+INITIAL_FLOATS+INTEGER_ARRAYS)


def _boundaries():
    return {'max_conditions':MAX_CONDITIONS,'max_points_per_table':MAX_POINTS,'max_logical_values':MAX_LOGICAL_VALUES,'max_npz_bytes':MAX_NPZ_BYTES,
        'scope':'one problem artifact; not a limit on the overall engineering objective'}


def _member(root,files,relative):
    if relative not in files:raise ContractError('Reference/collocation CSV must belong to the complete immutable material inventory')
    path=confined(root,relative)
    if not path.is_file() or path.stat().st_size>MAX_NPZ_BYTES:raise ContractError('PINN CSV member exceeds the 64MiB input budget or is missing')
    payload=path.read_bytes()
    if len(payload)>MAX_NPZ_BYTES or hashlib.sha256(payload).hexdigest()!=files[relative]:raise ContractError('PINN source bytes differ from their bound material version')
    return payload


def _table(payload,columns,label):
    try:text=payload.decode('utf-8-sig')
    except UnicodeError as error:raise ContractError('First PINN CSV boundary requires explicit UTF-8 text') from error
    stream=io.StringIO(text,newline='');reader=csv.reader(stream)
    try:headers=next(reader)
    except (StopIteration,csv.Error) as error:raise ContractError(label+' needs an explicit header row') from error
    if not headers or len(headers)>65536:raise ContractError('Invalid PINN CSV header count')
    for name,selector in columns.items():
        if selector['index']>=len(headers) or headers[selector['index']]!=selector['header']:raise ContractError(label+' column position/header conflicts with declared '+name)
    unused=[{'index':index,'header':header} for index,header in enumerate(headers) if index not in {v['index'] for v in columns.values()}]
    def records():
        count=0
        try:
            for row in reader:
                count+=1
                if count>MAX_POINTS:raise ContractError('PINN table exceeds its per-problem point budget')
                if len(row)!=len(headers):raise ContractError(label+' has missing, ragged or extra fields at row '+str(count+1))
                result={}
                for name,selector in columns.items():
                    value=row[selector['index']]
                    if name in {'condition_id','split'}:result[name]=value
                    else:
                        try:number=float(value)
                        except (ValueError,OverflowError) as error:raise ContractError(label+' needs finite SI numeric '+name+' at row '+str(count+1)) from error
                        if not np.isfinite(number):raise ContractError(label+' has nonfinite '+name+' at row '+str(count+1))
                        result[name]=number
                yield result
        except csv.Error as error:raise ContractError('Malformed static PINN CSV: '+label) from error
        finally:stream.close()
    return records(),unused


def _reference(payload,columns,definition,context):
    rows,unused=_table(payload,columns,'reference');conditions=definition['conditions'];lookup={c['id']:j for j,c in enumerate(conditions)}
    values={key:[] for key in FLOAT_COLUMNS+('condition_index','split_id')}
    for index,row in enumerate(rows):
        if index%1000==0:context.check()
        if row['condition_id'] not in lookup:raise ContractError('Reference row identifies an undeclared thermal condition')
        j=lookup[row['condition_id']];condition=conditions[j]
        if row['split']!=condition['split']:raise ContractError('Reference split column disagrees with its complete declared condition partition')
        for key in FLOAT_COLUMNS:values[key].append(row[key])
        values['condition_index'].append(j);values['split_id'].append(SPLIT_IDS[condition['split']])
    arrays={key:np.asarray(values[key],dtype=np.float64).reshape(-1,1) for key in FLOAT_COLUMNS}
    arrays.update(condition_index=np.asarray(values['condition_index'],dtype=np.int64),split_id=np.asarray(values['split_id'],dtype=np.int8))
    return arrays,unused


def _collocation(payload,columns,definition,context):
    rows,unused=_table(payload,columns,'collocation');conditions=definition['conditions'];lookup={c['id']:j for j,c in enumerate(conditions)}
    values={key:[] for key in ('time_s','initial_temperature_K','power_W','condition_index')}
    for index,row in enumerate(rows):
        if index%1000==0:context.check()
        if row['condition_id'] not in lookup:raise ContractError('Collocation row identifies an undeclared thermal condition')
        j=lookup[row['condition_id']]
        if conditions[j]['split']!='train':raise ContractError('Physics collocation is restricted to training conditions; validation/test condition inputs cannot enter training')
        for key in ('time_s','initial_temperature_K','power_W'):values[key].append(row[key])
        values['condition_index'].append(j)
    arrays={'collocation_'+key:np.asarray(values[key],dtype=np.float64).reshape(-1,1) for key in ('time_s','initial_temperature_K','power_W')}
    arrays['collocation_condition_index']=np.asarray(values['condition_index'],dtype=np.int64)
    return arrays,unused


def _initial(definition):
    rows=definition['conditions'];count=len(rows)
    return {'initial_time_s':np.full((count,1),definition['domain']['time_s'][0],dtype=np.float64),
        'initial_value_K':np.asarray([[c['initial_temperature_K']] for c in rows],dtype=np.float64),
        'initial_power_W':np.asarray([[c['power_W']] for c in rows],dtype=np.float64),
        'initial_condition_index':np.arange(count,dtype=np.int64),
        'initial_split_id':np.asarray([SPLIT_IDS[c['split']] for c in rows],dtype=np.int8)}


def _training_normalization(targets,state_scale):
    # Exact equality precedes std: summing identical decimal floats can otherwise
    # manufacture a tiny positive variance (for example 293.15 K).
    constant=bool(np.all(targets==targets[0]))
    population=0. if constant else float(targets.std(ddof=0))
    if not np.isfinite(population) or not constant and population<=0:raise ContractError('Nonconstant training reference temperatures require a finite positive population standard deviation')
    return {'data_nrmse_scale_K':float(state_scale) if constant else population,'fit_split':'train','fit_samples':len(targets),'ddof':0,
        'data_population_std_K':population,'data_nrmse_scale_source':'engineering_state_scale_for_exact_constant_training_targets' if constant else 'training_population_std',
        'data_nrmse_fallback_K':float(state_scale) if constant else None,'exact_constant_training_targets':constant}


def validate_pinn_arrays(definition,arrays):
    """Validate physical arrays and exact condition partitions, without fitting."""
    if not isinstance(arrays,dict) or set(arrays)!=ARRAY_NAMES:raise ContractError('Conditional ODE arrays differ from the exact physical point schema')
    conditions=definition['conditions'];count=len(conditions);n=len(arrays['time_s']);m=len(arrays['collocation_time_s'])
    if not count<=n<=MAX_POINTS or not 1<=m<=MAX_POINTS:raise ContractError('Every explicit condition needs reference labels and every training condition needs collocation')
    for key in FLOAT_COLUMNS+COLLOCATION_FLOATS+INITIAL_FLOATS:
        value=arrays[key];rows=n if key in FLOAT_COLUMNS else m if key in COLLOCATION_FLOATS else count
        if not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.shape!=(rows,1) or not np.isfinite(value).all():raise ContractError('PINN physical arrays must be finite float64 [points,1]: '+key)
    for key in INTEGER_ARRAYS:
        value=arrays[key];rows=n if key in {'condition_index','split_id'} else m if key=='collocation_condition_index' else count
        if not isinstance(value,np.ndarray) or value.dtype.kind not in 'iu' or value.shape!=(rows,):raise ContractError('PINN condition/split arrays must be ordered integer vectors: '+key)
    if sum(v.size for v in arrays.values())>MAX_LOGICAL_VALUES or sum(v.nbytes for v in arrays.values())>MAX_NPZ_BYTES:raise ContractError('PINN arrays exceed the single-problem logical/byte resource budget')
    initial=np.asarray([c['initial_temperature_K'] for c in conditions]);power=np.asarray([c['power_W'] for c in conditions]);split=np.asarray([SPLIT_IDS[c['split']] for c in conditions])
    lo,hi=definition['domain']['time_s']
    for prefix,index_key in [('', 'condition_index'),('collocation_','collocation_condition_index')]:
        ids=arrays[index_key]
        if np.any(ids<0) or np.any(ids>=count):raise ContractError('Out-of-range PINN condition index')
        if not np.allclose(arrays[prefix+'initial_temperature_K'][:,0],initial[ids],atol=1e-12,rtol=0) or not np.allclose(arrays[prefix+'power_W'][:,0],power[ids],atol=1e-12,rtol=0):raise ContractError('Thermal row changes its declared initial condition or constant forcing')
        times=arrays[prefix+'time_s'][:,0]
        if np.any(times<lo-1e-12) or np.any(times>hi+1e-12):raise ContractError('PINN point lies outside the declared trajectory time domain')
        if len(set(zip(ids.tolist(),times.tolist())))!=len(ids):raise ContractError('Duplicate condition/time points cannot silently change data/physics weighting')
    ids=arrays['condition_index']
    if set(ids.tolist())!=set(range(count)) or not np.array_equal(arrays['split_id'],split[ids]):raise ContractError('Reference points must cover all exact declared whole-condition partitions')
    if np.any(arrays['temperature_K']<=0):raise ContractError('Reference state must be finite positive absolute kelvin')
    for j in range(count):
        times=arrays['time_s'][ids==j,0]
        if np.any(np.diff(times)<=0):raise ContractError('Reference times must be strictly increasing within each condition; no sorting conceals source order conflicts')
    training=set(np.flatnonzero(split==0).tolist());collocation=set(arrays['collocation_condition_index'].tolist())
    if collocation!=training:raise ContractError('Fixed collocation must cover training conditions only, never validation/test conditions')
    expected=_initial(definition)
    for key,value in expected.items():
        if not np.array_equal(arrays[key],value):raise ContractError('Initial-condition points must be the exact declared physical initial values/splits')
    population=arrays['temperature_K'][arrays['split_id']==0,0];norm=definition['normalization']
    fitted=_training_normalization(population,norm['state_scale_K'])
    # Old nonconstant records remain readable, but their previous implicit
    # positive-std convention must never reinterpret an exact constant target.
    explicit='exact_constant_training_targets' in norm
    keys=fitted.keys() if explicit else ('data_nrmse_scale_K','fit_split','fit_samples','ddof')
    if (not explicit and fitted['exact_constant_training_targets'] or any(norm.get(key)!=fitted[key] for key in keys)):
        raise ContractError('Data NRMSE scale/source/fallback is not the exact training-only population statistic or declared engineering scale for exact constant targets')
    return {'reference_rows':n,'collocation_rows':m,'initial_conditions':count,'reference_rows_by_split':{name:int((arrays['split_id']==index).sum()) for name,index in SPLIT_IDS.items()},
        'conditions_by_split':{name:int((split==index).sum()) for name,index in SPLIT_IDS.items()},**fitted,
        'training_collocation_condition_indices':sorted(training),'logical_values':sum(v.size for v in arrays.values())}


def _load_points(path,definition,expected_sha):
    if not path.is_file() or path.stat().st_size>MAX_NPZ_BYTES:raise ContractError('PINN point archive is missing or exceeds its byte budget')
    payload=path.read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha:raise ContractError('PINN point archive differs from its immutable artifact hash')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries=archive.infolist()
            if len(entries)!=len(ARRAY_NAMES) or {e.filename for e in entries}!={key+'.npy' for key in ARRAY_NAMES}:raise ContractError('PINN point archive has duplicate, missing or unexpected members')
            logical=uncompressed=0
            for entry in entries:
                with archive.open(entry) as stream:
                    version=np.lib.format.read_magic(stream);reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:raise ContractError('Unsupported PINN NPY header')
                    shape,order,dtype=reader(stream);key=entry.filename[:-4];rank=1 if key in INTEGER_ARRAYS else 2
                    if len(shape)!=rank or any(type(v) is not int or not 1<=v<=MAX_POINTS for v in shape) or rank==2 and shape[1]!=1 or dtype.hasobject or dtype.kind not in ('iu' if rank==1 else 'f') or rank==2 and dtype!=np.dtype('float64'):raise ContractError('PINN NPY header has invalid declared shape/type')
                    size=1
                    for width in shape:size*=width
                    logical+=size;uncompressed+=size*dtype.itemsize
                    if logical>MAX_LOGICAL_VALUES or uncompressed>MAX_NPZ_BYTES or stream.tell()+size*dtype.itemsize!=entry.file_size:raise ContractError('PINN NPY logical size/bytes exceed budget or disagree with member payload')
        with np.load(io.BytesIO(payload),allow_pickle=False) as loaded:result={key:loaded[key].copy() for key in loaded.files}
    except ContractError:raise
    except (OSError,ValueError,KeyError,EOFError,zipfile.BadZipFile) as error:raise ContractError('Invalid bounded PINN point archive') from error
    validate_pinn_arrays(definition,result)
    return result


def prepare_pinn_problem(workspace,material_id,config,context=None):
    context=context or OperationContext();declaration=validate_pinn_config(config)
    source_root,source_meta=workspace.resolve(material_id,kind='material');material=read_json(source_root/'material.json')
    reference=_member(source_root,material['file_hashes'],declaration['reference_file']);collocation=_member(source_root,material['file_hashes'],declaration['collocation_file'])
    definition=declaration['definition'];arrays,unused_reference=_reference(reference,declaration['reference_columns'],definition,context)
    more,unused_collocation=_collocation(collocation,declaration['collocation_columns'],definition,context);arrays.update(more);arrays.update(_initial(definition))
    train=arrays['temperature_K'][arrays['split_id']==0,0]
    definition['normalization'].update(_training_normalization(train,definition['normalization']['state_scale_K']))
    definition['resource_boundary']=_boundaries();definition['source']={**definition['source'],'material':{'id':material_id,'metadata_hash':canonical_hash(source_meta),'content_hash':material['content_hash']},
        'reference':{'file':declaration['reference_file'],'sha256':hashlib.sha256(reference).hexdigest(),'columns':declaration['reference_columns']},
        'collocation':{'file':declaration['collocation_file'],'sha256':hashlib.sha256(collocation).hexdigest(),'columns':declaration['collocation_columns']},'provenance_is_user_declared':True}
    definition['source']['pinn_signals']=pinn_signals(definition);definition=validate_pinn_definition(definition);statistics=validate_pinn_arrays(definition,arrays)
    context.check();workspace.resolve(material_id,kind='material')
    with workspace.stage('conditional_ode_problem',MaterialInterface('bundle','Conditional constant-forcing thermal ODE problem'),[material_id],config,compatible_parents=False) as stage:
        atomic_json(stage.root/'problem.json',definition);np.savez_compressed(stage.root/'points.npz',**arrays)
        if (stage.root/'points.npz').stat().st_size>MAX_NPZ_BYTES:raise ContractError('Final PINN point archive exceeds the one-problem byte budget')
        _load_points(stage.root/'points.npz',definition,hashlib.sha256((stage.root/'points.npz').read_bytes()).hexdigest())
        atomic_json(stage.root/'source.json',definition['source'])
        atomic_json(stage.root/'report.json',{'kind':'conditional_ode_problem_preparation','status':'prepared','source_executed':False,'training':'not_run','capability':definition['capability'],
            'statistics':statistics,'unused_reference_columns':unused_reference,'unused_collocation_columns':unused_collocation,
            'reference_derivative_columns':'Not copied to points.npz; actual equation residual requires model time autodifferentiation',
            'initial_conditions':'All declared splits retained; only initial_split_id==0 may contribute training gradients',
            'confirmation_sources':context.case_confirmation_sources})
        (stage.root/'README.md').write_text('Conditional thermal trajectory problem\n\nC*dT/dt=-G*(T-Ta)+P, with constant P and explicit initial T0 on each trajectory. Units are seconds, kelvin and watts; C is J/K and G W/K. The raw point archive contains physical values. Normalization and the time-derivative chain rule are in problem.json/source.pinn_signals. Data-error scale uses training-only population std; for exactly constant training targets it uses the predeclared positive state_scale_K engineering fallback and records zero population std. Validation/test labels never fit this scale. Collocation contains training conditions only. Initial points retain all split labels; training must select the training subset.\n\nNo source code, solver or neural network is executed during preparation. Supplied derivative columns are not consumed. This bundle is not a Plant, a generic one-step transition, an arbitrary-forcing solution or a general PDE solver. Preparation does not establish accuracy or generalization to unobserved conditions; explicit downstream acceptance belongs to training/evaluation.\n',encoding='utf-8')
        return stage.publish({'status':'verified','checks':['explicit_equation_SI_units','immutable_reference_and_collocation','whole_condition_split','training_only_collocation','train_only_nrmse_scale','finite_bounded_physical_arrays'],
            'training':'not_run','accuracy':'not_assessed'},source={'kind':'conditional_ode_problem_source','material':material_id,'content_hash':material['content_hash'],
            'pinn_signals':definition['source']['pinn_signals'],'source_executed':False,'provenance_is_user_declared':True},activate=False)


def read_pinn_problem(workspace,artifact_id):
    root,metadata=workspace.resolve(artifact_id,kind='conditional_ode_problem');definition=validate_pinn_definition(read_json(root/'problem.json'))
    source=definition['source'];material_id=source['material']['id']
    if {p['id'] for p in metadata['parents']}!={material_id}:raise ContractError('Conditional ODE problem must bind its exact one source material parent')
    material_root,material_meta=workspace.resolve(material_id,kind='material');material=read_json(material_root/'material.json')
    if source['material']['metadata_hash']!=canonical_hash(material_meta) or source['material']['content_hash']!=material['content_hash']:raise ContractError('Conditional ODE source material identity changed')
    for role in ['reference','collocation']:
        record=source[role]
        if material['file_hashes'].get(record['file'])!=record['sha256']:raise ContractError('Conditional ODE CSV version differs from bound source material')
    if metadata['source'].get('pinn_signals')!=source['pinn_signals'] or read_json(root/'source.json')!=source:raise ContractError('Artifact/source PINN signals differ from actual problem semantics')
    arrays=_load_points(root/'points.npz',definition,metadata['files']['points.npz'])
    return {'definition':definition,'arrays':arrays,'root':root,'metadata':metadata}
