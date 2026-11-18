"""Self-contained external imitation MLP bundles and numeric MATLAB adapters."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path, PureWindowsPath
import shutil
import uuid

import numpy as np
import torch

from .contracts import ContractError, DataInterface
from . import external_policy as runtime
from .supervised_export import build_supervised_network, validate_network_config, _feature_metadata


_MATLAB_SOURCE=r'''function [action,next_memory,status] = external_policy_nn(observation,reference,disturbance,previous_action,memory,p)
% Pure numeric frozen MLP. No file reads, dynamic calls, source Plant or toolbox.
assert(numel(observation)==p.observation_dim,'Policy:Shape','Observation width differs.');
assert(numel(reference)==p.reference_dim,'Policy:Shape','Reference width differs.');
assert(numel(disturbance)==p.disturbance_dim,'Policy:Shape','Disturbance width differs.');
assert(numel(previous_action)==p.previous_dim,'Policy:Shape','Previous action width differs.');
z=zeros(p.input_dim,1);
for j=1:p.input_dim
    k=p.feature_index(j);group=p.feature_group(j);
    if group==0,z(j)=observation(k);
    elseif group==1,z(j)=reference(k);
    elseif group==2,z(j)=disturbance(k);
    else,z(j)=previous_action(k);end
end
assert(all(isfinite(z)),'Policy:Finite','Raw features must be finite.');
z=(z-p.x_location(:))./p.x_scale(:);
assert(all(isfinite(z)),'Policy:Finite','Feature scaling overflowed.');
for k=1:numel(p.layer_input)
    weights=p.layers(k).W;bias=p.layers(k).b;
    z=reshape(weights,p.layer_output(k),p.layer_input(k))*z+bias(:);
    if k<numel(p.layer_input)
        if p.activation_code==1,z=tanh(z);
        elseif p.activation_code==2,z=max(z,0);
        else,z=z./(1+exp(-z));end
    end
end
raw=z.*p.y_scale(:)+p.y_location(:);
assert(all(isfinite(raw)),'Policy:Finite','Raw action is nonfinite; no failed inference can be hidden by projection.');
low=p.action_low(:);high=p.action_high(:);
if p.has_rate==1
    previous=previous_action(:);
    assert(all(isfinite(previous)) && all(previous>=low-1e-12 & previous<=high+1e-12),'Policy:Previous','Actual previous action must be finite and amplitude feasible.');
    low=max(low,previous+p.rate_low(:));high=min(high,previous+p.rate_high(:));
    assert(all(low<=high),'Policy:Intersection','Amplitude/rate intersection is empty.');
end
action=max(min(raw,high),low);
next_memory=struct('calls',memory.calls+1);
status='ok';
end
'''


_INFER_SOURCE=r'''"""Infer from explicitly assembled raw feature rows; no implicit history or prior."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='JSON {features:[[...]],previous_action:[[...]],feature_labels:[...]} or NPZ with numeric features/previous_action')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('_standalone_external_policy',root/'policy_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);policy=module.load_external_policy(root)
    path=Path(args.input)
    if path.stat().st_size>64*1024*1024:raise ValueError('Input exceeds the 64MiB inference document budget')
    if path.suffix.lower()=='.npz':
        import io,zipfile,math
        payload=path.read_bytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries=archive.infolist()
            if len(entries)>3 or len({e.filename for e in entries})!=len(entries):raise ValueError('Unexpected/duplicate inference NPZ members')
            for entry in entries:
                if entry.filename not in {'features.npy','previous_action.npy','feature_labels.npy'}:raise ValueError('Unknown inference NPZ field')
                with archive.open(entry) as stream:
                    version=np.lib.format.read_magic(stream)
                    reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:raise ValueError('Unsupported NPY header')
                    shape,order,dtype=reader(stream);count=math.prod(shape)
                    if dtype.hasobject or len(shape)>2 or count>5000000 or stream.tell()+count*dtype.itemsize!=entry.file_size:raise ValueError('Invalid/oversized NPY shape')
        with np.load(io.BytesIO(payload),allow_pickle=False) as values:
            features=values['features'].copy();previous=values['previous_action'].copy() if 'previous_action' in values else None
            labels=values['feature_labels'].tolist() if 'feature_labels' in values else None
    else:
        value=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict):raise ValueError('JSON input must declare features and actual previous action explicitly')
        features=value['features'];previous=value.get('previous_action');labels=value.get('feature_labels')
    raw=policy.predict_raw(features,labels);action=policy.project_actions(raw,previous)
    Path(args.output).write_text(json.dumps({'raw_action':raw.tolist(),'action':action.tolist(),'units':[item['unit'] for item in policy.layout['target_order']],
        'projection':'amplitude_rate_intersection','feature_labels':policy.layout['feature_labels']},allow_nan=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
'''


def _portable(value):
    storage={"workspace","source_root","artifact_path","input_path","local_path","model_path","data_path","source_file_path","checkpoint_path","bundle_root","manifest_path"}
    if isinstance(value,dict):return {key:_portable(item) for key,item in value.items() if key not in storage}
    if isinstance(value,(list,tuple)):return [_portable(item) for item in value]
    if isinstance(value,str) and (PureWindowsPath(value).drive or Path(value).is_absolute()):return "<original-storage-location-omitted>"
    return value


def _write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def _runtime_source():
    functions=[runtime._read_json,runtime._numeric,runtime._effective_deployment_mapping,runtime._feature_layout,runtime._policy_metadata,runtime._verify_files,runtime.project_actions,
        runtime.ExternalPolicy,runtime.load_external_policy]
    return ('"""Standalone external policy: NumPy + Torch; no Harness or original source."""\n'
        'from __future__ import annotations\nimport hashlib\nimport json\nimport math\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nContractError=ValueError\n\n'
        +inspect.getsource(validate_network_config)+'\n'+inspect.getsource(build_supervised_network)+'\n'
        +'\n\n'.join(inspect.getsource(function) for function in functions))


def _source_bundle(supervised_root):
    source=Path(supervised_root).resolve(strict=True)
    if not (source/'bundle_manifest.json').is_file():source=(source/'bundle').resolve(strict=True)
    manifest=runtime._read_json(source/'bundle_manifest.json')
    required={'model.pt','network.json','scaler.json','data_interface.json','features.json','training_config.json','inference.py','predict.py','README.md'}
    if manifest.get('kind')!='control-supervised-mlp-bundle-v1' or set(manifest.get('files',{}))!=required:raise ContractError('An exact S05 supervised model export is required')
    for name,digest in manifest['files'].items():
        path=(source/name).resolve(strict=True)
        if path.parent!=source or not path.is_file() or path.stat().st_size>64*1024*1024 or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ContractError('Supervised source member differs: '+name)
    return source


def _matlab_assets(settings,state,data,scaler,constraints,context,binding):
    if binding['status']!='supported_current_mapping':return {'status':'unsupported','reason':binding['reason'],'history':'caller-managed assembled-feature runtime remains available'}
    physical=context['interface'];entries=binding['current_features'];observations=binding['observations']
    uses={role:any(entry['kind']==role for entry in entries) for role in ['reference','disturbance','previous_action']}
    has_rate=constraints['rate_bounds'] is not None;previous=uses['previous_action'] or has_rate
    dimensions=[settings['input_dim'],*settings['hidden_dims'],settings['output_dim']]
    layers=[]
    for j in range(len(dimensions)-1):
        layers.append({'W':state[str(2*j)+'.weight'].detach().cpu().numpy().tolist(),'b':state[str(2*j)+'.bias'].detach().cpu().numpy().tolist()})
    parameters={'layers':layers,'layer_input':dimensions[:-1],'layer_output':dimensions[1:],
        'activation_code':{'tanh':1,'relu':2,'silu':3}[settings['activation']],
        'input_dim':settings['input_dim'],'output_dim':settings['output_dim'],'observation_dim':len(observations),
        'reference_dim':len(physical['outputs']) if uses['reference'] else 0,'disturbance_dim':len(physical['disturbances']) if uses['disturbance'] else 0,
        'previous_dim':settings['output_dim'] if previous else 0,
        'feature_group':[{'observation':0,'reference':1,'disturbance':2,'previous_action':3}[entry['kind']] for entry in entries],
        'feature_index':[entry['index']+1 for entry in entries],
        'x_location':scaler['X']['location'],'x_scale':scaler['X']['scale'],'y_location':scaler['y']['location'],'y_scale':scaler['y']['scale'],
        'action_low':[pair[0] for pair in constraints['bounds']],'action_high':[pair[1] for pair in constraints['bounds']],
        'has_rate':1 if has_rate else 0,'rate_low':[pair[0] for pair in constraints['rate_bounds']] if has_rate else [],
        'rate_high':[pair[1] for pair in constraints['rate_bounds']] if has_rate else []}
    adapter={'schema_version':1,'backend':'matlab_function','source_file':'source/external_policy_nn.m','entrypoint':'external_policy_nn',
        'interface':physical,'parameters':parameters,'time_unit':'s','control_period':binding['control_period'],'observations':observations,
        'reference_available':uses['reference'],'disturbance_available':uses['disturbance'],'previous_action_available':previous,
        'initial_memory':{'calls':0},'action_mapping':[{'action_index':j,'plant_control_index':j,'plant_control_name':v['name'],'unit':v['unit']} for j,v in enumerate(physical['controls'])],
        'failure_policy':'stop'}
    if has_rate:adapter['input_rate_bounds']=constraints['rate_bounds']
    return {'status':'supported','source_file':'external_policy_nn.m','entrypoint':'external_policy_nn','parameters_file':'matlab_parameters.json',
        'parameters':parameters,'adapter':adapter,'dependencies':['MATLAB numeric operations; no additional toolbox needed for this NN function'],
        'calling_convention':'[action,next_memory,status]=external_policy_nn(observation,reference,disturbance,previous_action,memory,parameters)',
        'status_success':'ok','memory':'calls is diagnostic; MLP has no dynamic hidden state; reset initial_memory per trajectory',
        'action_order':'physical Plant control order; previous and rate bounds use that same order','actual_matlab_execution':'not_run_by_export'}


def export_external_policy(root,supervised_root,data_interface,action_constraints,physical_context,*,provenance=None):
    """Export the actual selected S05 weights, with no new training/forward loop.

    ``root`` is a new bundle directory. Inputs may be arbitrary causal action
    DataInterface widths; physical .act/MATLAB are offered only when explicit
    S09 mappings can be validated. Generic history inference stays caller-managed.
    """
    source=_source_bundle(supervised_root)
    data=DataInterface.parse(data_interface.to_dict() if hasattr(data_interface,'to_dict') else data_interface).to_dict()
    original=runtime._read_json(source/'data_interface.json')
    if data!=original:raise ContractError('External policy DataInterface must equal the actually trained supervised model interface')
    settings=validate_network_config(runtime._read_json(source/'network.json'));features=runtime._read_json(source/'features.json');scaler=runtime._read_json(source/'scaler.json')
    state=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    return _export_external_policy_state(root,state,settings,data,scaler,action_constraints,physical_context,
        provenance=provenance,features=features,source_weight=source/'model.pt')


def export_external_policy_state(root,state,network_config,data_interface,scaler,action_constraints,physical_context,*,provenance=None):
    """Export actual raw optimized weights without fabricating a training artifact.

    ``state`` is the finite CPU float64 S05 Sequential state dictionary, with
    0/2/... layer keys. The original training-only scaler and ordered causal
    DataInterface remain unchanged. This function performs no optimization or
    reference-data forward, and does not establish control improvement.
    """
    data=DataInterface.parse(data_interface.to_dict() if hasattr(data_interface,'to_dict') else data_interface).to_dict()
    settings=validate_network_config(network_config)
    features=_feature_metadata(data)
    return _export_external_policy_state(root,state,settings,data,scaler,action_constraints,physical_context,
        provenance=provenance,features=features)


def _export_external_policy_state(root,state,settings,data,scaler,action_constraints,physical_context,*,provenance,features,source_weight=None):
    context={'interface':None,'physical_mapping':None,'inferred_from_feature_width':False} if physical_context is None else json.loads(json.dumps(physical_context,allow_nan=False))
    constraints=json.loads(json.dumps(action_constraints,allow_nan=False))
    if provenance is not None and not isinstance(provenance,dict):raise ContractError('Policy provenance must be an explicit JSON object')
    source_mapping=(provenance or {}).get('source_mapping')
    layout,transforms,bounds,rate,binding=runtime._policy_metadata(data,features,scaler,constraints,context,source_mapping)
    if settings['input_dim']!=data['input_dim'] or settings['output_dim']!=data['output_dim']:raise ContractError('Trained network dimensions differ from action interface')
    if not isinstance(state,dict) or not state or any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in state.values()):raise ContractError('Policy state must contain finite CPU float64 tensor weights only')
    if sum(v.numel() for v in state.values())>5_000_000:raise ContractError('Policy state exceeds its single-network parameter budget')
    state={key:value.detach().clone() for key,value in state.items()}
    with torch.random.fork_rng(devices=[]):network=build_supervised_network(settings)
    network.load_state_dict(state,strict=True)
    matlab=_matlab_assets(settings,state,data,scaler,constraints,context,binding)
    selected={entry['signal'] for entry in layout['feature_order']}
    deployment_scope='simulation_only' if any(s.get('availability')=='simulation_only' and s['name'] in selected for s in data['signals']) else 'user_declared_available_not_certified'
    descriptor={'schema_version':1,'representation':'external_imitation_mlp','target_semantics':'current_action_in_target_units',
        'feature_labels':layout['feature_labels'],'target_labels':layout['target_labels'],'physical_binding':binding,
        'deployment_scope':deployment_scope,
        'action_projection':'amplitude_rate_intersection_v1','previous_action':'actual applied command; required for rate projection; never implicitly zero',
        'history':layout['history_management'],'hidden_state':'none','provenance':_portable(provenance or {}),
        'source_weight_sha256':hashlib.sha256(source_weight.read_bytes()).hexdigest() if source_weight else None,
        'closed_loop_validation':'not_run_by_export','training':'performed_by_supervised_training_owner_not_export'}
    if source_weight is None:descriptor.update(weight_source='explicit_actual_raw_state_dict',training='performed_by_declared_training_owner_not_export')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ContractError('Export external policy into a new empty bundle directory')
    temporary=root/f'.policy-export-{uuid.uuid4().hex}';temporary.mkdir()
    try:
        if source_weight is None:
            torch.save(state,temporary/'model.pt');descriptor['source_weight_sha256']=hashlib.sha256((temporary/'model.pt').read_bytes()).hexdigest()
        else:shutil.copyfile(source_weight,temporary/'model.pt')
        for name,value in [('policy',descriptor),('network',settings),('scaler',scaler),('data_interface',data),('features',features),('action_constraints',constraints),('physical_context',context)]:_write(temporary/(name+'.json'),_portable(value))
        (temporary/'policy_runtime.py').write_text(_runtime_source(),encoding='utf-8');(temporary/'infer.py').write_text(_INFER_SOURCE,encoding='utf-8')
        if matlab['status']=='supported':
            (temporary/'external_policy_nn.m').write_text(_MATLAB_SOURCE,encoding='utf-8');_write(temporary/'matlab_parameters.json',matlab['parameters'])
        readme=('External imitation policy\n\nDependencies: Python, numpy and torch; CPU float64. This bundle imports no Harness modules or original model/material/workspace files. MATLAB numeric export needs MATLAB only; the original .M/SLX evaluation backend may have its own product prerequisites.\n\n'
            'Run: python infer.py --input inputs.json --output actions.json\nInput: {"features":[[...]],"feature_labels":[...],"previous_action":[[...]]}. Features are raw physical-unit rows assembled in exactly features.json order. Histories and availability delays are explicit caller-managed same-trajectory data; no zero padding, implicit reset state or expert-internal memory is fabricated. Raw vector width does not imply physical nx.\n\n'
            'Inference: scale raw X with training-only X.location/X.scale; apply the actual frozen MLP; inverse-scale y into action units; project onto amplitude and per-control-update rate bounds. Rate projection requires actual previous applied action in target/Plant order, with exactly the same batch shape. No implicit zero or broadcast is used. Clamped directions can have zero input gradient.\n\n'
            'Runtime APIs: predict_raw/features -> raw NumPy actions; predict(features,previous_action) -> projected NumPy actions; raw_forward and forward keep Torch input gradients while weights are frozen. project_actions projects saved predictions without another NN forward. reset clears diagnostic call count only: the MLP has no hidden state. Physical act(x,reference,disturbance,previous) is available only for a validated current state-observation mapping. Other mappings must use assembled features. The evaluator holds actions for the declared Plant subticks; policy inference does not advance a Plant.\n\n'
            'See policy.json for scope, data_interface.json for roles/units/columns, action_constraints.json for explicit bounds, physical_context.json for optional Plant context, and policy_manifest.json for member SHA256 and MATLAB support. Export itself does not train/evaluate a controller or establish stability, timing on target hardware, extrapolation validity or a performance improvement.\n')
        (temporary/'README.md').write_text(readme,encoding='utf-8')
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in temporary.iterdir()}
        manifest={'schema_version':1,'kind':'control-external-imitation-policy-v1','representation':'external_imitation_mlp','files':files,
            'input_dim':data['input_dim'],'output_dim':data['output_dim'],'feature_labels':layout['feature_labels'],'target_labels':layout['target_labels'],
            'deployment_scope':deployment_scope,
            'runtime':{'entrypoint':'policy_runtime.py:load_external_policy','dependencies':['numpy','torch'],'dtype':'float64','device':'cpu'},
            'physical_act':{'supported':binding['status']=='supported_current_mapping' and binding['state_act_supported'],'binding':binding},
            'matlab':_portable(matlab),'resource_boundary':{'max_parameters':5_000_000,'max_activation_values':5_000_000,'scope':'one network/inference batch; not an overall engineering objective limit'}}
        _write(temporary/'policy_manifest.json',manifest)
        # Structural loading verifies actual state keys and all copied metadata;
        # no test-set forward or fake zero-history example is performed here.
        runtime.load_external_policy(temporary)
        for file in sorted(temporary.iterdir(),key=lambda p:p.name=='policy_manifest.json'):file.rename(root/file.name)
        return manifest
    finally:shutil.rmtree(temporary,ignore_errors=True)
