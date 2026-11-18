"""Independent conditional PINN delivery using the actual numerical runtime."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path,PureWindowsPath

import torch

from .contracts import ContractError
from . import pinn_contract
from . import pinn_runtime
from .supervised_export import build_supervised_network,validate_network_config


def _pinn_portable(value):
    private={'workspace','source_root','artifact_path','input_path','local_path','model_path','data_path','source_file_path',
        'checkpoint_path','resume_checkpoint','resume_checkpoint_sha256','plan_refs','bundle_root','manifest_path'}
    if isinstance(value,dict):return {key:_pinn_portable(item) for key,item in value.items() if key not in private}
    if isinstance(value,(list,tuple)):return [_pinn_portable(item) for item in value]
    if isinstance(value,str) and (PureWindowsPath(value).drive or Path(value).is_absolute()):return '<original-storage-location-omitted>'
    return value


def _pinn_write(path,value):
    try:text=json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    except (TypeError,ValueError,OverflowError) as error:raise ContractError('PINN export metadata must be finite JSON') from error
    if len(text.encode('utf-8'))>4*1024*1024:raise ContractError('PINN export metadata exceeds the bounded member size')
    path.write_text(text,encoding='utf-8')


def _pinn_training_metadata(training_config,provenance):
    config=_pinn_portable(training_config or {});source=_pinn_portable(provenance or {})
    if not isinstance(config,dict) or not isinstance(source,dict):raise ContractError('PINN training/provenance declarations must be JSON objects')
    declaration=pinn_runtime._pinn_training_declaration(source)
    mode=pinn_runtime._pinn_parameterization(config.get('conditional_parameterization','free_mlp'))
    metadata={'config':config,**declaration,'conditional_parameterization':mode,'training_provenance_is_producer_declared':True,
        'export_performs_training':False,'export_certifies_accuracy_or_stability':False}
    return metadata,source


def _pinn_standalone_source():
    contract=inspect.getsource(pinn_contract);runtime=inspect.getsource(pinn_runtime)
    contract_import='from .contracts import ContractError\n'
    if contract.count(contract_import)!=1:raise ContractError('Pure PINN contract source no longer matches its portable import boundary')
    contract=contract.replace(contract_import,'')
    for line in ['from .contracts import ContractError\n','from .pinn_contract import validate_pinn_definition\n',
                 'from .supervised_export import build_supervised_network, validate_network_config\n']:
        if runtime.count(line)!=1:raise ContractError('Actual PINN runtime source no longer matches its portable import boundary')
        runtime=runtime.replace(line,'')
    contract='\n'.join(line for line in contract.splitlines() if not line.startswith('from __future__ import '))
    runtime='\n'.join(line for line in runtime.splitlines() if not line.startswith('from __future__ import '))
    return ('"""Self-contained conditional thermal NN. NumPy/Torch only; no original CSV/model/Workspace."""\n'
        'from __future__ import annotations\nimport torch\nfrom torch import nn\nContractError=ValueError\n\n'+contract+'\n\n'
        +inspect.getsource(validate_network_config)+'\n\n'+inspect.getsource(build_supervised_network)+'\n\n'+runtime+'\n')


_PINN_PREDICT=r'''"""Evaluate T(t;T0,P) under constant P, never a one-step Plant."""
import argparse
import importlib.util
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='JSON time_s,initial_temperature_K,power_W; scalar,[N] or matching[N,1]')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent;path=Path(args.input)
    if path.stat().st_size>64*1024*1024:raise ValueError('Conditional input document exceeds its 64MiB budget')
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict) or set(value)!={'time_s','initial_temperature_K','power_W'}:raise ValueError('Provide explicit ordered physical query fields')
    spec=importlib.util.spec_from_file_location('_portable_thermal_pinn',root/'pinn_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);model=module.load_pinn(root)
    result=model.predict(value['time_s'],value['initial_temperature_K'],value['power_W'])
    Path(args.output).write_text(json.dumps({'temperature_K':result.tolist(),'capability':model.capability,
        'forcing':'Each P is constant for the entire trajectory from the declared initial time; this is not a stepper.'},allow_nan=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
'''


def export_pinn(root,model,definition,network_config,*,training_config=None,provenance=None):
    """Export actual smooth MLP weights and SI conditional/equation semantics.

    No analytic temperature solution, original source, samples or collocation
    arrays are delivered or evaluated by this function. Training declarations
    do not turn export itself into accuracy/physics/stability certification.
    """
    definition=pinn_contract.validate_pinn_definition(definition);config=pinn_runtime._pinn_network_config(network_config)
    wrapped=isinstance(model,(pinn_runtime.ConditionalThermalPINN,pinn_runtime._AffineInitialPowerNetwork))
    network=model.network if wrapped else model
    if wrapped and model.definition!=definition:raise ContractError('PINN model and exported physical definition differ')
    inspected=pinn_runtime._pinn_inspect_network(network)
    if inspected['hidden_dims']!=config['hidden_dims'] or config['hidden_dims'] and inspected!=config:raise ContractError('PINN export config differs from its actual Linear/activation layers')
    training,source=_pinn_training_metadata(training_config,provenance)
    mode=training['conditional_parameterization']
    if wrapped and model.parameterization!=mode:raise ContractError('Export parameterization must equal the actual trained conditional representation')
    state={name:value.detach().cpu().clone() for name,value in network.state_dict().items()}
    with torch.random.fork_rng(devices=[]):check=build_supervised_network(config)
    check.load_state_dict(state,strict=True);pinn_runtime.ConditionalThermalPINN(check,definition,parameterization=mode,freeze=True)
    target=Path(root).resolve();target.mkdir(parents=True,exist_ok=True)
    if any(target.iterdir()):raise ContractError('PINN export requires a new empty bundle directory')
    torch.save(state,target/'model.pt')
    for name,value in [('network',config),('definition',definition),('training',training),('provenance',source)]:_pinn_write(target/(name+'.json'),value)
    (target/'pinn_runtime.py').write_text(_pinn_standalone_source(),encoding='utf-8')
    (target/'predict.py').write_text(_PINN_PREDICT,encoding='utf-8')
    (target/'README.md').write_text(
        '# Conditional thermal trajectory model\n\n'
        'Dependencies: NumPy and PyTorch, CPU float64. Load pinn_runtime.load_pinn(directory), or run python predict.py --input query.json --output temperature.json. No project package, original CSV, material, Workspace, MATLAB or MCP is needed.\n\n'
        'Queries explicitly supply time_s, initial_temperature_K and power_W. Scalars, [N] vectors, or consistently shaped [N,1] columns are supported; use scalars for shared conditions. Inputs must be finite and within definition.json domain. Kelvin and seconds are required, with no implicit Celsius/time conversion.\n\n'
        'Network inputs are s=(t-time_origin_s)/time_scale_s, a=(T0-Ta)/state_scale_K and b=P/power_scale_W. The actual smooth MLP produces theta, and T=Ta+state_scale_K*theta. The exported network contains no analytic solution.\n\n'
        'The explicit conditional_parameterization in training.json and pinn_manifest.json is free_mlp by default. Optional affine_initial_power evaluates the same actual raw MLP at three declared condition corners (s,alo,blo), (s,ahi,blo), (s,alo,bhi), then computes f00+alpha*(f10-f00)+beta*(f01-f00). This is an affine-in-initial-temperature/constant-power prior for the declared constant-coefficient ODE; time remains learned, with no analytic trajectory or hard initial condition. The IC loss remains necessary. This mode requires three raw network evaluations per query.\n\n'
        'temperature/forward keep Torch gradients of t/T0/P. residual(...,create_graph=True) retains derivatives for physics-gradient work; it computes [C*dT/dt+G*(T-Ta)-P]/physics_scale_W with the full time chain rule. Set normalized=False for a residual in W. predict is NumPy inference.\n\n'
        'P is constant throughout the complete trajectory from time_origin_s. Independent batched conditions are allowed; successive changing P values are not a simulated time-varying input. There is no state/reset/step method, arbitrary one-step transition, Plant interface, general PDE capability, or automatic MPC/PGAC handoff.\n\n'
        'training.json records producer-declared physics/ablation semantics. A physics-off ablation is not labeled physics-trained merely because a residual can be evaluated. Export does not train, test, prove accuracy, establish stability or authorize a new control task. Unit/domain/scales/capabilities and member SHA256 are explicit; library/hardware migration still requires numerical verification.\n',encoding='utf-8')
    files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in target.iterdir()}
    manifest={'schema_version':1,'kind':pinn_runtime.PINN_BUNDLE_KIND,'files':files,'input_order':['time_s','initial_temperature_K','power_W'],
        'output':{'name':'temperature_K','unit':'K'},'capability':definition['capability'],'domain':definition['domain'],
        'runtime':{'entrypoint':'pinn_runtime.py:load_pinn','dependencies':['numpy','torch'],'device':'cpu','dtype':'float64'},
        'training_classification':training['classification'],'physics_loss_enabled':training['physics_loss_enabled'],
        'conditional_parameterization':mode,
        'resource_boundary':{'max_parameters':pinn_runtime.PINN_MAX_PARAMETERS,'max_activation_values':pinn_runtime.PINN_MAX_ACTIVATION_VALUES,
            'raw_network_evaluations_per_query':3 if mode=='affine_initial_power' else 1,
            'max_member_bytes':pinn_runtime.PINN_MAX_MEMBER_BYTES,'max_bundle_bytes':pinn_runtime.PINN_MAX_BUNDLE_BYTES,'scope':'one model/inference batch, not an overall engineering objective limit'}}
    _pinn_write(target/'pinn_manifest.json',manifest)
    # Structural load only: no made-up zero condition and no data/test forward.
    pinn_runtime.load_pinn(target)
    return manifest
