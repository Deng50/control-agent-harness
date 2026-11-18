"""Independent codec for the existing internal imitation policy numerics.

The numerical network/feature/constraint bodies are copied from their actual
loaded source. This adds a delivery format without replacing the old trainer.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json
from .contracts import ContractError, ModelInterface


_LOADER = r'''
class ContractError(ValueError):
    pass

def tensor(value):
    return torch.as_tensor(value,dtype=torch.float64)

def _vector(value,width,name):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=(width,) or not np.isfinite(result).all():raise ValueError(name+' requires a finite ordered vector')
    return result

def _interface(raw):
    if raw.get('schema_version')!=1:raise ValueError('Unsupported physical interface version')
    groups={}
    for group in ('states','controls','outputs','disturbances'):
        rows=raw.get(group)
        if not isinstance(rows,list) or len(rows)>65536:raise ValueError('Invalid physical interface')
        values=[]
        for row in rows:
            if not isinstance(row,dict) or not isinstance(row.get('name'),str) or not isinstance(row.get('unit'),str):raise ValueError('Named physical variables and units required')
            bounds=np.asarray([row['lower'],row['upper']],dtype=np.float64)
            if not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:raise ValueError('Invalid physical bounds')
            values.append(SimpleNamespace(**row))
        groups[group]=values
    if not groups['states'] or not groups['controls'] or not groups['outputs']:raise ValueError('Nonempty physical dimensions required')
    return SimpleNamespace(**groups,nx=len(groups['states']),nu=len(groups['controls']),ny=len(groups['outputs']),nd=len(groups['disturbances']),sample_time=raw['sample_time'])

class PortablePolicy:
    def __init__(self,module,document):
        self.module,self.document=module,document
        self.interface,self.config,self.rate=module.interface,module.config,module.rate
        self._previous=None
        # Check the raw network before the original projection can hide Inf/NaN.
        def finite_output(_module,_args,output):
            raw=output*self.module.output_scale+self.module.output_mean
            if not torch.isfinite(output).all() or not torch.isfinite(raw).all():raise FloatingPointError('Neural network produced nonfinite values before action projection')
        self._finite_hook=module.network.register_forward_hook(finite_output)
    def reset(self,previous_action=None):
        if previous_action is None and self.rate is not None:raise ValueError('Rate-constrained reset requires the actual previous action')
        self._previous=None if previous_action is None else self._previous_vector(previous_action)
    def _previous_vector(self,value):
        result=_vector(value,self.interface.nu,'previous action')
        lo=np.array([item.lower for item in self.interface.controls]);hi=np.array([item.upper for item in self.interface.controls])
        if np.any(result<lo) or np.any(result>hi):raise ValueError('Previous action is outside physical actuator bounds')
        return result
    def _check_activation_budget(self,batch):
        config=self.config
        if config['architecture']=='mlp':
            estimated=batch*4*sum([config['input_dim'],*config.get('hidden_dims',[]),config['output_dim']])
        else:
            gates=4 if config['architecture']=='lstm' else 1
            estimated=batch*config['preview_len']*4*(config['static_features_dim']+config['preview_features_dim']+(gates+1)*config['hidden_size']*config['num_layers'])
        if estimated>5_000_000:raise ValueError('Inference exceeds the single-call activation budget; reduce batch size or the declared preview/network dimensions')
    def act(self,x,reference,disturbance=None,previous=None):
        self._check_activation_budget(1)
        x=_vector(x,self.interface.nx,'state');reference=_vector(reference,self.interface.ny,'reference')
        if disturbance is None and self.interface.nd==0:disturbance=[]
        disturbance=_vector(disturbance,self.interface.nd,'disturbance')
        previous=self._previous if previous is None else self._previous_vector(previous)
        if previous is None and self.rate is not None:raise ValueError('Supply/reset actual previous action before rate-constrained inference')
        action,diagnostics=self.module.act(x,reference,disturbance,previous)
        self._previous=action.copy()
        return action,diagnostics
    def forward(self,x,reference,disturbance,previous=None):
        if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[-1]!=self.interface.nx:raise ValueError('Batched forward requires ordered [batch,nx] states')
        self._check_activation_budget(x.shape[0])
        for name,value,width in [('state',x,self.interface.nx),('reference',reference,self.interface.ny),('disturbance',disturbance,self.interface.nd)]:
            if not isinstance(value,torch.Tensor) or value.shape!=(x.shape[0],width) or value.dtype!=torch.float64 or value.device.type!='cpu' or not torch.isfinite(value).all():raise ValueError(name+' forward input must be a finite CPU float64 tensor with the declared batch and ordered width')
        if previous is not None:
            if not isinstance(previous,torch.Tensor) or previous.shape!=(x.shape[0],self.interface.nu) or previous.dtype!=torch.float64 or previous.device.type!='cpu' or not torch.isfinite(previous).all():raise ValueError('Previous action requires the matching finite CPU float64 batch')
            if torch.any(previous<self.module.action_low) or torch.any(previous>self.module.action_high):raise ValueError('Previous action is outside physical actuator bounds')
        elif self.rate is not None:raise ValueError('Rate-constrained forward requires actual previous actions')
        return self.module(x,reference,disturbance,previous=previous)

def load_policy(directory=None):
    root=Path(__file__).resolve().parent if directory is None else Path(directory).resolve()
    manifest=json.loads((root/'policy_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='control-legacy-policy-bundle-v1':raise ValueError('Unsupported policy bundle')
    files=manifest.get('files')
    if not isinstance(files,dict) or len(files)>10000:raise ValueError('Invalid file inventory')
    total=0
    for name,digest in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Policy member escapes or is missing')
        total+=path.stat().st_size
        if total>256*1024*1024:raise ValueError('Policy bundle exceeds the inference byte budget')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Policy member hash changed: '+name)
    document=json.loads((root/'policy.json').read_text(encoding='utf-8'))
    interface=_interface(document['interface']);config=document['network'];scalers=document['scalers']
    width=config.get('input_dim');outputs=config.get('output_dim')
    if type(width) is not int or not 1<=width<=65536 or outputs!=interface.nu:raise ValueError('Invalid network input/output contract')
    architecture=config.get('architecture')
    if architecture=='mlp':
        hidden=config.get('hidden_dims',[])
        if not isinstance(hidden,list) or len(hidden)>16 or any(type(n) is not int or not 1<=n<=65536 for n in hidden):raise ValueError('Invalid/beyond-budget MLP dimensions')
        dims=[width,*hidden,outputs]
        if sum((a+1)*b for a,b in zip(dims,dims[1:]))+2*sum(hidden)>5_000_000:raise ValueError('Configured policy exceeds the tensor budget')
    elif architecture in {'rnn','lstm'}:
        for key,limit in [('hidden_size',4096),('num_layers',16),('preview_len',65536),('static_features_dim',65536),('preview_features_dim',65536)]:
            if type(config.get(key)) is not int or not 1<=config[key]<=limit:raise ValueError('Invalid/beyond-budget recurrent dimensions')
        hidden=config['hidden_size'];gates=4 if architecture=='lstm' else 1;per_step=config['static_features_dim']+config['preview_features_dim']
        recurrent=gates*hidden*(per_step+hidden+2)+(config['num_layers']-1)*gates*hidden*(2*hidden+2)
        head=2*hidden+(hidden+1)*(hidden//2)+(hidden//2+1)*outputs
        if recurrent+head>5_000_000:raise ValueError('Configured policy exceeds the tensor budget')
    else:raise ValueError('Unknown neural architecture')
    saved=torch.load(root/'policy.pt',weights_only=True,map_location='cpu')
    state=saved.get('policy_state') if isinstance(saved,dict) else None
    if not isinstance(state,dict) or sum(value.numel() for value in state.values() if isinstance(value,torch.Tensor))>5_000_000:raise ValueError('Policy state exceeds the supported tensor budget')
    if any(not isinstance(value,torch.Tensor) or value.dtype!=torch.float64 or not torch.isfinite(value).all() for value in state.values()):raise ValueError('Policy state must contain finite float64 tensors')
    expected={key:np.asarray(scalers[key],dtype=np.float64) for key in ('input_mean','input_scale','output_mean','output_scale')}
    expected.update(action_low=np.asarray([v.lower for v in interface.controls]),action_high=np.asarray([v.upper for v in interface.controls]))
    for key,value in expected.items():
        n=width if key.startswith('input_') else interface.nu
        if value.shape!=(n,) or not np.isfinite(value).all() or key.endswith('scale') and np.any(value<=0):raise ValueError('Invalid scaler/bounds: '+key)
        if key not in state or not np.array_equal(state[key].numpy(),value):raise ValueError('Saved buffers differ from declared scaler/constraints: '+key)
    # Build metadata only before allocating parameters. A small saved tensor
    # file must not authorize a huge configuration-driven allocation.
    with torch.random.fork_rng(devices=[]),torch.device('meta'):
        skeleton=NeuralPolicy(interface,config,scalers)
    if sum(value.numel() for value in skeleton.state_dict().values())>5_000_000:raise ValueError('Configured policy exceeds the tensor budget')
    expected_shapes={key:tuple(value.shape) for key,value in skeleton.state_dict().items()}
    if set(expected_shapes)!=set(state) or any(tuple(state[key].shape)!=shape for key,shape in expected_shapes.items()):raise ValueError('Network configuration and saved tensor shapes differ')
    with torch.random.fork_rng(devices=[]):module=NeuralPolicy(interface,config,scalers,state)
    module.eval()
    return PortablePolicy(module,document)
'''


def export_legacy_policy(directory, interface, config, scalers, state, *, provenance=None):
    from .policy import NeuralPolicy, build_network, features
    from .reused.mlp import MLPNetwork
    from .reused.rnn import RNNNetwork
    from .reused.lstm import LSTMNetwork
    physical=interface if isinstance(interface,ModelInterface) else ModelInterface.parse(interface)
    if not isinstance(state,dict) or sum(value.numel() for value in state.values() if isinstance(value,torch.Tensor))>5_000_000:
        raise ContractError('Legacy standalone export exceeds its tensor budget')
    if any(not isinstance(value,torch.Tensor) or value.dtype!=torch.float64 or not torch.isfinite(value).all() for value in state.values()):
        raise ContractError('Legacy policy export requires finite float64 state tensors')
    target=Path(directory)
    if target.exists() and any(target.iterdir()):raise ContractError('Policy export refuses to overwrite an existing bundle')
    target.mkdir(parents=True,exist_ok=True)
    numerical=(MLPNetwork,RNNNetwork,LSTMNetwork,build_network,features,NeuralPolicy)
    header='from __future__ import annotations\nimport copy,hashlib,json,math,random,time\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom typing import Sequence\nimport numpy as np\nimport torch\nfrom torch import nn\nDTYPE=torch.float64\n\n'
    (target/'policy_runtime.py').write_text(header+'\n\n'.join(inspect.getsource(item) for item in numerical)+'\n\n'+_LOADER,encoding='utf-8')
    document={'schema_version':1,'kind':'legacy_internal_imitation','interface':physical.to_dict(),'network':copy.deepcopy(config),
        'scalers':copy.deepcopy(scalers),'input_semantics':config['input_semantics'],
        'history_semantics':'external actual state/reference/disturbance; recurrent architecture consumes explicitly declared constant future disturbance preview, not a history window',
        'memory_interface':'reset(actual_previous_action) or act(...,previous=actual_previous_action); outputs are remembered only when caller omits previous after reset',
        'state_availability':'All listed states must be supplied by the deployment measurement/estimation system; this export does not implement sensors or an observer',
        'source':provenance or {},'producer':{'torch':torch.__version__,'numpy':np.__version__,'dtype':'float64','device':'cpu'}}
    atomic_json(target/'policy.json',document)
    torch.save({'policy_state':{key:value.detach().cpu() for key,value in state.items()}},target/'policy.pt')
    (target/'infer.py').write_text('from pathlib import Path\nimport json,sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom policy_runtime import load_policy\nif __name__=="__main__":\n request=json.load(sys.stdin);policy=load_policy();u,diagnostics=policy.act(request["x"],request["reference"],request.get("disturbance"),request.get("previous"));print(json.dumps({"action":u.tolist(),"diagnostics":diagnostics},allow_nan=False))\n',encoding='utf-8')
    (target/'requirements.txt').write_text('numpy\ntorch>=2.1\n',encoding='utf-8')
    (target/'README.md').write_text('# Internal imitation policy export\n\nRun `python infer.py` with a JSON object containing x/reference/disturbance/previous on stdin, or import policy_runtime.load_policy. No project package or original model is required. Ordered physical units, scalers, architecture and actual action constraints are in policy.json. Rate-constrained inference requires the actual previous action or an explicit reset. RNN/LSTM retain the original declared-constant disturbance preview semantics. They are not history-window encoders.\n\nThe numerical network/forward/projection source is copied from the original loaded implementation. Weights load with weights_only=True. File hashes detect changes and are not digital signatures. Physical-model simulation and hardware observers are not included. Producer versions are recorded; reproduce and verify numerical behavior when migrating libraries or hardware.\n',encoding='utf-8')
    files={path.relative_to(target).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in target.rglob('*') if path.is_file()}
    manifest={'schema_version':1,'kind':'control-legacy-policy-bundle-v1','files':files,
        'numerical_sources':{item.__name__:{'module':item.__module__,'sha256':hashlib.sha256(inspect.getsource(item).encode()).hexdigest()} for item in numerical}}
    atomic_json(target/'policy_manifest.json',manifest)
    return {'status':'exported','directory':'bundle','manifest':'bundle/policy_manifest.json','kind':manifest['kind'],'numerical_sources':manifest['numerical_sources']}
