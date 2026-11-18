"""Pure SI contract for a conditional, constant-forcing thermal ODE problem.

This module does not open files, import a model, run a solver, or infer a Plant
interface. Portable PINN runtimes may vendor these pure validation functions.
"""
from __future__ import annotations

import json
import math
from pathlib import PurePosixPath,PureWindowsPath
import re

from .contracts import ContractError


SPLIT_IDS={'train':0,'validation':1,'test':2}
MAX_CONDITIONS=4096
MAX_POINTS=500000
MAX_LOGICAL_VALUES=5000000
MAX_NPZ_BYTES=64*1024*1024
UNITS={'time_s':'s','initial_temperature_K':'K','power_W':'W','temperature_K':'K'}
CAPABILITY={'kind':'conditional_trajectory_only','constant_forcing':True,'one_step_transition':False,'general_pde':False}
REFERENCE_COLUMNS={'condition_id','split','time_s','initial_temperature_K','power_W','temperature_K'}
COLLOCATION_COLUMNS={'condition_id','time_s','initial_temperature_K','power_W'}
BASE_NORMALIZATION={'time_origin_s','time_scale_s','state_offset_K','state_scale_K','power_scale_W','physics_scale_W'}
LEGACY_PREPARED_NORMALIZATION=BASE_NORMALIZATION|{'data_nrmse_scale_K','fit_split','fit_samples','ddof'}
PREPARED_NORMALIZATION=LEGACY_PREPARED_NORMALIZATION|{'data_population_std_K','data_nrmse_scale_source','data_nrmse_fallback_K','exact_constant_training_targets'}


def _json(value,label):
    try:
        text=json.dumps(value,allow_nan=False,ensure_ascii=False)
        if len(text.encode('utf-8'))>4*1024*1024:raise ValueError('definition byte budget')
        return json.loads(text)
    except (TypeError,ValueError,RecursionError,OverflowError) as error:raise ContractError(label+' must be bounded finite JSON') from error


def _number(value,label,positive=False):
    if type(value) not in {int,float} or not math.isfinite(value) or positive and value<=0:raise ContractError(label+' must be an explicit finite'+(' positive' if positive else '')+' SI number')
    return 0. if float(value)==0 else float(value)


def _range(value,label):
    if not isinstance(value,list) or len(value)!=2:raise ContractError(label+' requires explicit [lower,upper]')
    result=[_number(v,label) for v in value]
    if result[0]>=result[1]:raise ContractError(label+' must have increasing finite endpoints')
    return result


def _relative_csv(value):
    if not isinstance(value,str) or not value or PureWindowsPath(value).drive or ':' in value:raise ContractError('PINN source files must be explicit material-relative CSV paths')
    path=PurePosixPath(value.replace('\\','/'))
    if path.is_absolute() or '..' in path.parts or path.suffix.lower()!='.csv':raise ContractError('PINN source files must remain relative CSV members of one immutable material')
    return path.as_posix()


def _columns(value,required,label):
    if not isinstance(value,dict) or set(value)!=required:raise ContractError(label+' requires all explicit semantic column selectors')
    result={};seen=set()
    for name,selector in value.items():
        if (not isinstance(selector,dict) or set(selector)!={'index','header'} or type(selector['index']) is not int or not 0<=selector['index']<=65535
                or not isinstance(selector['header'],str) or not selector['header'].strip() or selector['index'] in seen):raise ContractError('PINN columns need unambiguous distinct positions and exact headers; names alone are insufficient')
        seen.add(selector['index']);result[name]=dict(selector)
    return result


def _physical(raw,prepared):
    equation=raw.get('equation')
    if (not isinstance(equation,dict) or equation.get('template')!='thermal_first_order' or equation.get('forcing')!='constant_per_condition'
            or set(equation)-{'template','parameters','forcing','expression'}):
        raise ContractError('PINN requires the explicit thermal_first_order equation and constant forcing; use ordinary supervised fitting when no equation is supplied')
    parameters=equation.get('parameters')
    if not isinstance(parameters,dict) or set(parameters)!={'C_J_per_K','G_W_per_K','Ta_K'}:raise ContractError('Thermal equation needs explicit SI C[J/K], G[W/K] and Ta[K]')
    parameters={key:_number(value,key,positive=True) for key,value in parameters.items()}
    expression='C*dT/dt=-G*(T-Ta)+P'
    if 'expression' in equation and equation['expression']!=expression:raise ContractError('The supported thermal equation expression cannot be silently replaced')
    domain=raw.get('domain')
    if not isinstance(domain,dict) or set(domain)!={'time_s','initial_temperature_K','power_W'}:raise ContractError('Explicit time, initial-temperature and constant-power domains required')
    domain={key:_range(value,key) for key,value in domain.items()}
    if domain['initial_temperature_K'][0]<=0:raise ContractError('Initial temperatures must be strictly positive absolute kelvin')
    norm=raw.get('normalization');key_sets=(LEGACY_PREPARED_NORMALIZATION,PREPARED_NORMALIZATION) if prepared else (BASE_NORMALIZATION,)
    if not isinstance(norm,dict) or set(norm) not in key_sets:raise ContractError('Explicit time/state/power/physics SI normalization required; data error scale uses training population statistics or the declared engineering fallback for exact constant targets')
    normalization={key:_number(norm[key],key,positive=key not in {'time_origin_s','state_offset_K'}) for key in BASE_NORMALIZATION}
    if normalization['time_origin_s']!=domain['time_s'][0] or normalization['state_offset_K']!=parameters['Ta_K']:raise ContractError('Thermal normalization must use the declared initial time and ambient Ta offset')
    if prepared:
        if norm.get('fit_split')!='train' or type(norm.get('fit_samples')) is not int or norm['fit_samples']<1 or type(norm.get('ddof')) is not int or norm['ddof']!=0:raise ContractError('PINN data NRMSE normalization must identify training-only population statistics')
        normalization.update(data_nrmse_scale_K=_number(norm['data_nrmse_scale_K'],'data_nrmse_scale_K',positive=True),fit_split='train',fit_samples=norm['fit_samples'],ddof=0)
        if set(norm)==PREPARED_NORMALIZATION:
            population=_number(norm['data_population_std_K'],'data_population_std_K')
            constant=norm['exact_constant_training_targets'];fallback=norm['data_nrmse_fallback_K'];source=norm['data_nrmse_scale_source']
            if population<0 or type(constant) is not bool:raise ContractError('Training population scale must be nonnegative and exact-constant status explicit')
            if constant:
                if (population!=0 or type(fallback) not in {int,float} or fallback!=normalization['state_scale_K']
                        or normalization['data_nrmse_scale_K']!=fallback or source!='engineering_state_scale_for_exact_constant_training_targets'):
                    raise ContractError('Exact constant training targets require zero population std and the declared positive state_scale_K engineering fallback')
                fallback=_number(fallback,'data_nrmse_fallback_K',positive=True)
            elif population<=0 or fallback is not None or source!='training_population_std' or normalization['data_nrmse_scale_K']!=population:
                raise ContractError('Nonconstant target normalization must use positive training population std without a fallback')
            normalization.update(data_population_std_K=population,data_nrmse_scale_source=source,data_nrmse_fallback_K=fallback,exact_constant_training_targets=constant)
    if raw.get('units')!=UNITS:raise ContractError('First thermal PINN boundary requires explicit seconds, absolute kelvin and watts; no implicit unit conversion')
    conditions=raw.get('conditions')
    if not isinstance(conditions,list) or not 3<=len(conditions)<=MAX_CONDITIONS:raise ContractError('Explicit bounded train/validation/test conditions are required')
    normalized=[];names=set();physical=set();parts=set()
    for row in conditions:
        if (not isinstance(row,dict) or set(row)!={'id','split','initial_temperature_K','power_W'} or not isinstance(row['id'],str)
                or not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,127}',row['id']) or row['id'] in names
                or not isinstance(row['split'],str) or row['split'] not in SPLIT_IDS):raise ContractError('Conditions require unique IDs, explicit split, initial temperature and constant power')
        t0=_number(row['initial_temperature_K'],'condition initial temperature');power=_number(row['power_W'],'condition constant power')
        if not domain['initial_temperature_K'][0]<=t0<=domain['initial_temperature_K'][1] or not domain['power_W'][0]<=power<=domain['power_W'][1]:raise ContractError('Condition lies outside the declared initial-temperature/power domain')
        if (t0,power) in physical:raise ContractError('Names or split labels cannot turn the same physical thermal condition into independent data')
        names.add(row['id']);physical.add((t0,power));parts.add(row['split']);normalized.append({'id':row['id'],'split':row['split'],'initial_temperature_K':t0,'power_W':power})
    if parts!=set(SPLIT_IDS):raise ContractError('All three complete condition partitions must be nonempty')
    return {'schema_version':1,'problem_kind':'conditional_ode','template':'thermal_first_order',
        'equation':{'template':'thermal_first_order','expression':expression,'parameters':parameters,'forcing':'constant_per_condition'},
        'domain':domain,'normalization':normalization,'units':dict(UNITS),'conditions':normalized,'capability':dict(CAPABILITY)}


def validate_pinn_config(config):
    """Pure user declaration normalization; no CSV or supplied code is run."""
    allowed={'schema_version','equation','domain','normalization','units','conditions','reference_file','reference_columns','collocation_file','collocation_columns','source','plan_refs'}
    if not isinstance(config,dict) or type(config.get('schema_version')) is not int or config['schema_version']!=1 or set(config)-allowed:raise ContractError('An explicit version-1 thermal PINN preparation declaration is required')
    config=_json(config,'PINN configuration');definition=_physical(config,prepared=False)
    source=config.get('source')
    if (not isinstance(source,dict) or set(source)!={'kind','description'} or not isinstance(source.get('kind'),str)
            or source['kind'] not in {'synthetic_analytic','provided_reference','measured'} or not isinstance(source.get('description'),str) or not source['description'].strip()):raise ContractError('Reference provenance requires an explicit kind and engineering description')
    definition['source']=dict(source)
    reference=_relative_csv(config.get('reference_file'));collocation=_relative_csv(config.get('collocation_file'))
    if reference==collocation:raise ContractError('Reference labels and training-only collocation inputs require separate explicitly bound CSV members')
    return {'definition':definition,'reference_file':reference,'reference_columns':_columns(config.get('reference_columns'),REFERENCE_COLUMNS,'reference_columns'),
        'collocation_file':collocation,'collocation_columns':_columns(config.get('collocation_columns'),COLLOCATION_COLUMNS,'collocation_columns')}


def pinn_signals(definition):
    """Engineering confirmation summary; conditional trajectories, never a Plant."""
    source=definition['source'];n=definition['normalization']
    return {'schema_version':1,'problem_kind':'conditional_ode','equation':definition['equation'],'domain':definition['domain'],'units':definition['units'],
        'coordinates':[{'name':'time_s','role':'independent_variable','unit':'s','transform':'s=(time_s-time_origin_s)/time_scale_s'},
            {'name':'initial_temperature_K','role':'initial_condition','unit':'K','transform':'a=(initial_temperature_K-state_offset_K)/state_scale_K'},
            {'name':'power_W','role':'constant_driving_condition','unit':'W','transform':'b=power_W/power_scale_W','semantics':'constant for the complete trajectory'}],
        'state':{'name':'temperature_K','unit':'K','network_output':'theta','reconstruction':'temperature_K=state_offset_K+state_scale_K*theta'},
        'time_derivative':{'unit':'K/s','chain_rule':'dT/dt=(state_scale_K/time_scale_s)*dtheta/ds'},
        'equation_parameter_units':{'C_J_per_K':'J/K','G_W_per_K':'W/K','Ta_K':'K'},
        'normalized_residual':'[C_J_per_K*(state_scale_K/time_scale_s)*dtheta/ds+G_W_per_K*state_scale_K*theta-power_W]/physics_scale_W',
        'normalized_initial_residual':'theta(s=0,a,b)-a','normalization':dict(n),'conditions':definition['conditions'],
        'condition_split_semantics':'whole distinct (initial_temperature_K, constant power_W) conditions; no renamed overlap',
        'collocation_semantics':'fixed physical inputs for training conditions only; no target temperature or analytic derivative supplied to the residual',
        'source':{'kind':source['kind'],'description':source['description'],'material':source['material'],'reference':source['reference'],'collocation':source['collocation']},
        'capability':definition['capability']}


def validate_pinn_definition(raw):
    """Validate the complete portable problem declaration without filesystem IO."""
    allowed={'schema_version','problem_kind','template','equation','domain','normalization','units','conditions','source','capability','resource_boundary'}
    if not isinstance(raw,dict) or set(raw)!=allowed or type(raw.get('schema_version')) is not int or raw['schema_version']!=1 or raw.get('problem_kind')!='conditional_ode' or raw.get('template')!='thermal_first_order':raise ContractError('Complete conditional thermal ODE problem definition required; not a transition/Plant interface')
    raw=_json(raw,'PINN problem definition');definition=_physical(raw,prepared=True)
    if raw['capability']!=CAPABILITY:raise ContractError('Conditional trajectory capability cannot be widened to arbitrary forcing, one-step dynamics or a general PDE')
    source=raw['source']
    if not isinstance(source,dict) or set(source)!={'kind','description','material','reference','collocation','pinn_signals','provenance_is_user_declared'} or source.get('provenance_is_user_declared') is not True:raise ContractError('Problem source must identify the bound material and both static CSV inputs')
    if not isinstance(source.get('kind'),str) or source['kind'] not in {'synthetic_analytic','provided_reference','measured'} or not isinstance(source.get('description'),str) or not source['description'].strip():raise ContractError('Reference provenance is required')
    material=source.get('material')
    if not isinstance(material,dict) or set(material)!={'id','metadata_hash','content_hash'} or not isinstance(material.get('id'),str) or not material['id']:raise ContractError('Explicit immutable material identity required')
    for key in ['metadata_hash','content_hash']:
        if not isinstance(material[key],str) or not re.fullmatch('[a-f0-9]{64}',material[key]):raise ContractError('Explicit source material SHA256 required')
    for role,columns in [('reference',REFERENCE_COLUMNS),('collocation',COLLOCATION_COLUMNS)]:
        record=source[role]
        if not isinstance(record,dict) or set(record)!={'file','sha256','columns'} or not isinstance(record.get('sha256'),str) or not re.fullmatch('[a-f0-9]{64}',record['sha256']):raise ContractError('Source CSV identity/columns must be explicit')
        record['file']=_relative_csv(record['file']);record['columns']=_columns(record['columns'],columns,role+' columns')
    if source['reference']['file']==source['collocation']['file']:raise ContractError('Reference and collocation files must be distinct')
    definition['source']=source;definition['resource_boundary']=raw['resource_boundary']
    expected_boundary={'max_conditions':MAX_CONDITIONS,'max_points_per_table':MAX_POINTS,'max_logical_values':MAX_LOGICAL_VALUES,'max_npz_bytes':MAX_NPZ_BYTES,
        'scope':'one problem artifact; not a limit on the overall engineering objective'}
    if definition['resource_boundary']!=expected_boundary:raise ContractError('Unsupported conditional problem resource boundary')
    if source['pinn_signals']!=pinn_signals(definition):raise ContractError('Saved PINN signal summary differs from the actual equation/units/conditions/scales/source versions')
    return definition
