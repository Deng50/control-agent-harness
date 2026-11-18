"""Conditional constant-power thermal trajectories, never a Plant transition.

The network maps dimensionless time/initial-temperature/power to theta. All
temperature derivatives retain the declared SI scaling through autograd.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import ContractError
from .pinn_contract import validate_pinn_definition
from .supervised_export import build_supervised_network, validate_network_config


PINN_MAX_PARAMETERS = 2_000_000
PINN_MAX_ACTIVATION_VALUES = 5_000_000
PINN_MAX_MEMBER_BYTES = 64 * 1024 * 1024
PINN_MAX_BUNDLE_BYTES = 128 * 1024 * 1024
PINN_BUNDLE_KIND = 'control-conditional-thermal-pinn-v1'
PINN_BUNDLE_MEMBERS = {'model.pt','network.json','definition.json','training.json','provenance.json','pinn_runtime.py','predict.py','README.md'}


def _pinn_parameterization(value='free_mlp'):
    if not isinstance(value,str) or value not in {'free_mlp','affine_initial_power'}:
        raise ContractError('conditional_parameterization must be free_mlp or affine_initial_power')
    return value


def _pinn_network_config(config):
    value = validate_network_config(config)
    if value['input_dim'] != 3 or value['output_dim'] != 1 or value['activation'] not in {'tanh','silu'}:
        raise ContractError('Conditional thermal networks require three inputs, one theta output and a smooth tanh/silu activation')
    if len(value['hidden_dims']) > 8 or any(width > 4096 for width in value['hidden_dims']):
        raise ContractError('Conditional thermal network exceeds the supported layer boundary')
    widths = [3,*value['hidden_dims'],1]
    if sum((a+1)*b for a,b in zip(widths,widths[1:])) > PINN_MAX_PARAMETERS:
        raise ContractError('Conditional thermal network exceeds its single-model parameter budget')
    return value


def _pinn_inspect_network(network):
    if type(network) is not nn.Sequential or len(network) < 1 or len(network) % 2 != 1:
        raise ContractError('Use the actual supported S05 Linear/smooth-activation MLP sequence')
    hidden=[];activation=None;previous=3
    for index,layer in enumerate(network):
        if index % 2 == 0:
            if type(layer) is not nn.Linear or layer.in_features != previous or layer.bias is None:
                raise ContractError('Conditional MLP has an incompatible ordered Linear layer')
            previous=layer.out_features
            if index < len(network)-1:hidden.append(previous)
        else:
            kind='tanh' if type(layer) is nn.Tanh else ('silu' if type(layer) is nn.SiLU else None)
            if kind is None or activation is not None and activation != kind:
                raise ContractError('Only uniform smooth tanh/silu hidden activations are supported')
            activation=kind
    if previous != 1:raise ContractError('Conditional thermal MLP must output theta with width one')
    config=_pinn_network_config({'schema_version':1,'architecture':'mlp','input_dim':3,'output_dim':1,
        'hidden_dims':hidden,'activation':activation or 'tanh','dtype':'float64'})
    for value in network.state_dict().values():
        if not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float64 or not torch.isfinite(value).all():
            raise ContractError('Conditional thermal weights must be finite CPU float64 tensors')
    return config


class _AffineInitialPowerNetwork(nn.Module):
    """Three actual MLP evaluations impose the known affine condition prior.

    Time remains a learned smooth function. No trajectory solution or hard
    initial value is inserted; the initial-condition objective is still needed.
    """
    def __init__(self,network,definition):
        super().__init__()
        config=_pinn_inspect_network(network);self.definition=validate_pinn_definition(definition)
        self.network=network;self._width_sum=sum([3,*config['hidden_dims'],1])
        d,n=self.definition['domain'],self.definition['normalization']
        self._alo=(d['initial_temperature_K'][0]-n['state_offset_K'])/n['state_scale_K']
        self._ahi=(d['initial_temperature_K'][1]-n['state_offset_K'])/n['state_scale_K']
        self._blo=d['power_W'][0]/n['power_scale_W'];self._bhi=d['power_W'][1]/n['power_scale_W']
        self._slo=(d['time_s'][0]-n['time_origin_s'])/n['time_scale_s'];self._shi=(d['time_s'][1]-n['time_origin_s'])/n['time_scale_s']
        if not all(math.isfinite(v) for v in [self._alo,self._ahi,self._blo,self._bhi,self._slo,self._shi]) or self._ahi<=self._alo or self._bhi<=self._blo or self._shi<=self._slo:
            raise ContractError('Affine condition bounds must remain finite and distinct after SI normalization')

    @property
    def parameterization(self):return 'affine_initial_power'

    def forward(self,z):
        if not isinstance(z,torch.Tensor) or z.device.type!='cpu' or z.dtype!=torch.float64 or z.ndim not in {1,2} or z.shape[-1]!=3 or not z.numel() or not torch.isfinite(z).all():
            raise ContractError('Affine conditional inputs require finite CPU float64 normalized [3] or [N,3] coordinates')
        count=1 if z.ndim==1 else len(z)
        if count*self._width_sum*4*3>PINN_MAX_ACTIVATION_VALUES:raise ContractError('Three-query affine representation exceeds its activation budget; use smaller batches')
        s,a,b=z[...,0:1],z[...,1:2],z[...,2:3]
        for value,low,high in [(s,self._slo,self._shi),(a,self._alo,self._ahi),(b,self._blo,self._bhi)]:
            if torch.any(value<low) or torch.any(value>high):raise ContractError('Affine normalized query is outside the declared time/condition domain')
        z00=torch.cat((s,torch.full_like(s,self._alo),torch.full_like(s,self._blo)),dim=-1)
        z10=torch.cat((s,torch.full_like(s,self._ahi),torch.full_like(s,self._blo)),dim=-1)
        z01=torch.cat((s,torch.full_like(s,self._alo),torch.full_like(s,self._bhi)),dim=-1)
        f00=self.network(z00);f10=self.network(z10);f01=self.network(z01)
        if any(value.shape!=(*z.shape[:-1],1) or not torch.isfinite(value).all() for value in [f00,f10,f01]):
            raise FloatingPointError('An affine corner MLP evaluation returned a wrong shape or nonfinite value')
        alpha=(a-self._alo)/(self._ahi-self._alo);beta=(b-self._blo)/(self._bhi-self._blo)
        result=f00+alpha*(f10-f00)+beta*(f01-f00)
        if not torch.isfinite(result).all():raise FloatingPointError('Affine conditional reconstruction is nonfinite')
        return result


def conditional_network(network,definition,parameterization='free_mlp'):
    """Return the same raw MLP by default or a parameter-sharing affine view."""
    mode=_pinn_parameterization(parameterization)
    return network if mode=='free_mlp' else _AffineInitialPowerNetwork(network,definition)


def _pinn_tensor_inputs(time_s,initial_temperature_K,power_W):
    values=[time_s,initial_temperature_K,power_W];styles=set();lengths=set()
    for value in values:
        if not isinstance(value,torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float64:
            raise ContractError('Thermal autograd inputs must be CPU float64 tensors; use predict for NumPy inputs')
        if value.ndim not in {0,1,2} or value.ndim == 2 and value.shape[1] != 1 or not value.numel():
            raise ContractError('Thermal inputs require scalar, [N], or consistently ordered [N,1] queries')
        if not torch.isfinite(value).all():raise ContractError('Thermal inputs must be finite')
        if value.ndim:styles.add(value.ndim);lengths.add(value.shape[0])
    if len(styles)>1 or len(lengths)>1:
        raise ContractError('Batch layouts/lengths must match; use scalar tensors for shared condition values')
    style=next(iter(styles),0)
    flattened=[value.reshape(-1) if value.ndim == 2 else value for value in values]
    return (*torch.broadcast_tensors(*flattened),style)


def _pinn_numpy(value,label):
    def boolean(item):
        return isinstance(item,(bool,np.bool_)) or isinstance(item,(list,tuple)) and any(boolean(v) for v in item)
    array=np.asarray(value)
    if boolean(value) or array.dtype.kind not in 'iuf' or array.ndim>2 or array.ndim==2 and array.shape[1]!=1:
        raise ContractError(label+' must be numeric scalar, [N] or [N,1], without booleans or implicit coercion')
    if not array.size or array.size > PINN_MAX_ACTIVATION_VALUES or not np.isfinite(array).all():raise ContractError(label+' is empty, nonfinite or exceeds one inference input budget')
    return torch.as_tensor(array,dtype=torch.float64)


def _pinn_training_declaration(source):
    if not isinstance(source,dict):raise ContractError('PINN training provenance must be an object')
    enabled=source.get('physics_loss_enabled');ablation=source.get('physics_off_ablation');graph=source.get('create_graph_for_physics_training')
    for value in [enabled,ablation,graph]:
        if value is not None and type(value) is not bool:raise ContractError('Physics/ablation/derivative declarations must be booleans')
    if ablation is True and enabled is not False:raise ContractError('A physics-off ablation must explicitly disable physics loss')
    if enabled is True and (ablation is True or graph is not True):raise ContractError('Physics training requires a connected create_graph derivative declaration')
    if source.get('is_pinn') is True and enabled is not True:raise ContractError('Physics-off or undeclared training cannot be labeled as PINN training')
    if 'is_pinn' in source and (type(source['is_pinn']) is not bool or source['is_pinn']!=(enabled is True)):
        raise ContractError('PINN label must agree with the declared physics objective')
    if enabled is False and graph is True:raise ContractError('Physics-off training cannot claim a derivative in its training objective')
    classification=('conditional_pinn_training_declared' if enabled is True else
        ('conditional_mlp_physics_off_ablation' if ablation is True else 'conditional_neural_trajectory_training_not_declared'))
    return {'classification':classification,'physics_loss_enabled':enabled,'physics_off_ablation':ablation,'create_graph_for_physics_training':graph}


class ConditionalThermalPINN(nn.Module):
    """T(t;T0,P) for one constant P over the entire trajectory from t_origin.

    Batched rows are independent conditional queries. Calling at successive
    powers does not simulate a time-varying input. There is deliberately no
    state/reset/step interface or arbitrary one-step transition capability.
    """
    def __init__(self,network,definition,*,parameterization='free_mlp',freeze=True):
        super().__init__()
        if type(freeze) is not bool:raise ContractError('freeze is a trusted boolean runtime option')
        self.definition=validate_pinn_definition(definition)
        self.network_config=_pinn_inspect_network(network)
        self.network=network;self.capability=copy.deepcopy(self.definition['capability'])
        self._parameterization=_pinn_parameterization(parameterization)
        # Keep only the raw network registered, so raw checkpoint/state keys do
        # not acquire a second representation.network prefix.
        object.__setattr__(self,'_representation',conditional_network(network,self.definition,self.parameterization))
        self._width_sum=sum([3,*self.network_config['hidden_dims'],1])
        if freeze:self.network.requires_grad_(False);self.network.eval()

    @property
    def parameterization(self):return self._parameterization

    def _checked(self,time_s,initial_temperature_K,power_W):
        time,initial,power,style=_pinn_tensor_inputs(time_s,initial_temperature_K,power_W)
        factor=3 if self.parameterization=='affine_initial_power' else 1
        if time.numel()*self._width_sum*4*factor > PINN_MAX_ACTIVATION_VALUES:
            raise ContractError('One conditional query batch exceeds its activation budget; use smaller batches')
        for name,value in [('time_s',time),('initial_temperature_K',initial),('power_W',power)]:
            lower,upper=self.definition['domain'][name]
            if torch.any(value<lower) or torch.any(value>upper):raise ContractError(name+' is outside the declared conditional trajectory domain')
        if any(p.device.type!='cpu' or p.dtype!=torch.float64 for p in self.network.parameters()):
            raise ContractError('Conditional thermal runtime must remain on CPU float64')
        return time,initial,power,style

    def _temperature_flat(self,time,initial,power):
        n=self.definition['normalization']
        features=torch.stack(((time-n['time_origin_s'])/n['time_scale_s'],
            (initial-n['state_offset_K'])/n['state_scale_K'],power/n['power_scale_W']),dim=-1)
        if not torch.isfinite(features).all():raise FloatingPointError('Conditional thermal normalization overflowed')
        representation=self._representation
        if (self.parameterization=='free_mlp' and representation is not self.network or
                self.parameterization=='affine_initial_power' and representation.network is not self.network):
            representation=conditional_network(self.network,self.definition,self.parameterization);object.__setattr__(self,'_representation',representation)
        theta=representation(features).squeeze(-1)
        temperature=n['state_offset_K']+n['state_scale_K']*theta
        if not torch.isfinite(theta).all() or not torch.isfinite(temperature).all():
            raise FloatingPointError('Conditional thermal network produced a nonfinite temperature')
        return temperature

    @staticmethod
    def _layout(value,style):return value.unsqueeze(-1) if style==2 else value

    def temperature(self,time_s,initial_temperature_K,power_W):
        time,initial,power,style=self._checked(time_s,initial_temperature_K,power_W)
        return self._layout(self._temperature_flat(time,initial,power),style)

    def forward(self,time_s,initial_temperature_K,power_W):
        """Alias for a conditional temperature query, not x/u/d dynamics."""
        return self.temperature(time_s,initial_temperature_K,power_W)

    def residual(self,time_s,initial_temperature_K,power_W,*,create_graph=False,normalized=True):
        if type(create_graph) is not bool or type(normalized) is not bool:raise ContractError('Residual graph/normalization flags must be booleans')
        with torch.enable_grad():
            time,initial,power,style=self._checked(time_s,initial_temperature_K,power_W)
            # Broadcast time entries are independent differentiation coordinates.
            # Differentiating a sum with respect to the original shared scalar
            # would incorrectly add derivatives from different conditions.
            if not time.requires_grad:time=time.clone().requires_grad_(True)
            temperature=self._temperature_flat(time,initial,power)
            derivative=torch.autograd.grad(temperature.sum(),time,create_graph=create_graph,retain_graph=True)[0]
            p=self.definition['equation']['parameters']
            value=p['C_J_per_K']*derivative+p['G_W_per_K']*(temperature-p['Ta_K'])-power
            if normalized:value=value/self.definition['normalization']['physics_scale_W']
            if not torch.isfinite(value).all():raise FloatingPointError('Conditional thermal equation residual is nonfinite')
            return self._layout(value,style)

    def predict(self,time_s,initial_temperature_K,power_W):
        inputs=[_pinn_numpy(value,label) for value,label in [(time_s,'time_s'),(initial_temperature_K,'initial_temperature_K'),(power_W,'power_W')]]
        with torch.no_grad():return self.temperature(*inputs).detach().cpu().numpy()


def _pinn_read_json(path):
    if path.stat().st_size > 4*1024*1024:raise ContractError('PINN metadata exceeds its bounded JSON member size')
    def invalid(value):raise ContractError('PINN metadata cannot contain nonfinite JSON constants')
    value=json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)
    if not isinstance(value,dict):raise ContractError('PINN metadata must be an explicit JSON object')
    return value


def _pinn_verify_bundle(root):
    manifest=_pinn_read_json(root/'pinn_manifest.json')
    if manifest.get('schema_version')!=1 or manifest.get('kind')!=PINN_BUNDLE_KIND or set(manifest.get('files',{}))!=PINN_BUNDLE_MEMBERS:
        raise ContractError('An exact version-1 conditional thermal bundle inventory is required')
    total=0
    for name,digest in manifest['files'].items():
        path=root/name;resolved=path.resolve(strict=True)
        if path.is_symlink() or resolved.parent!=root or not resolved.is_file():raise ContractError('PINN member escapes or aliases its bundle')
        size=path.stat().st_size;total+=size
        if size>PINN_MAX_MEMBER_BYTES or total>PINN_MAX_BUNDLE_BYTES:raise ContractError('PINN inference bundle exceeds its numerical byte budget')
        if not isinstance(digest,str) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ContractError('PINN inference bundle member hash changed: '+name)
    return manifest


def load_pinn(directory):
    """Load the independent conditional runtime, never an executable Plant."""
    root=Path(directory).resolve(strict=True)
    if not (root/'pinn_manifest.json').is_file():
        descriptor=_pinn_read_json(root/'pinn.json')
        if descriptor.get('bundle')!='bundle':raise ContractError('PINN artifact must identify its fixed bundle directory')
        root=(root/'bundle').resolve(strict=True)
        if root.parent!=Path(directory).resolve(strict=True):raise ContractError('PINN bundle escaped its artifact')
    manifest=_pinn_verify_bundle(root);definition=validate_pinn_definition(_pinn_read_json(root/'definition.json'))
    config=_pinn_network_config(_pinn_read_json(root/'network.json'))
    if manifest.get('capability')!=definition['capability'] or manifest.get('domain')!=definition['domain'] or manifest.get('input_order')!=['time_s','initial_temperature_K','power_W'] or manifest.get('output')!={'name':'temperature_K','unit':'K'}:
        raise ContractError('PINN bundle cannot widen its conditional trajectory contract')
    declaration=_pinn_training_declaration(_pinn_read_json(root/'provenance.json'));training=_pinn_read_json(root/'training.json')
    if any(training.get(key)!=value for key,value in declaration.items()) or manifest.get('training_classification')!=declaration['classification'] or manifest.get('physics_loss_enabled')!=declaration['physics_loss_enabled']:
        raise ContractError('PINN training/ablation labels contradict their declared source objective')
    training_config=training.get('config',{})
    if not isinstance(training_config,dict):raise ContractError('PINN training config must be an explicit JSON object')
    mode=_pinn_parameterization(training.get('conditional_parameterization','free_mlp'))
    if mode!=_pinn_parameterization(manifest.get('conditional_parameterization','free_mlp')) or mode!=_pinn_parameterization(training_config.get('conditional_parameterization','free_mlp')):
        raise ContractError('PINN inference parameterization differs from its training/manifest declaration')
    resource=manifest.get('resource_boundary',{})
    if not isinstance(resource,dict):raise ContractError('PINN inference resource declaration must be an object')
    evaluations=3 if mode=='affine_initial_power' else 1
    if resource.get('raw_network_evaluations_per_query',1)!=evaluations:
        raise ContractError('PINN resource declaration differs from its actual conditional representation')
    state=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or sum(value.numel() for value in state.values() if isinstance(value,torch.Tensor))>PINN_MAX_PARAMETERS:
        raise ContractError('PINN weight tensor inventory exceeds its supported model boundary')
    if any(not isinstance(value,torch.Tensor) or value.dtype!=torch.float64 or value.device.type!='cpu' or not torch.isfinite(value).all() for value in state.values()):
        raise ContractError('PINN weights must be finite CPU float64 tensors')
    with torch.random.fork_rng(devices=[]):network=build_supervised_network(config)
    network.load_state_dict(state,strict=True)
    model=ConditionalThermalPINN(network,definition,parameterization=mode,freeze=True)
    # A linear network's unused activation label may be tanh or silu; weights
    # and the actual module structure remain identical in that special case.
    if model.network_config['hidden_dims']!=config['hidden_dims'] or config['hidden_dims'] and model.network_config!=config:
        raise ContractError('PINN network definition differs from the actual runtime layers')
    model.network_config=config;model.training_declaration=declaration
    return model
