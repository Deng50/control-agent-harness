"""Frozen differentiable action MLPs with explicit causal feature contracts.

Raw feature rows are caller assembled in declared oldest-to-newest window
order. This runtime never guesses physical state dimensions, primes missing
history with zeros, or recovers expert-internal memory as student features.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import ContractError
from .supervised_export import build_supervised_network, validate_network_config


def _read_json(path):
    if path.stat().st_size > 64*1024*1024: raise ContractError("Policy JSON exceeds its member byte budget")
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(value, shape, label, positive=False):
    def boolean(item):
        return isinstance(item,(bool,np.bool_)) or isinstance(item,(list,tuple)) and any(boolean(child) for child in item)
    raw=np.asarray(value)
    if boolean(value) or raw.dtype.kind not in "iuf" or raw.shape!=tuple(shape): raise ContractError(label+" has the wrong numeric shape/type")
    result=raw.astype(np.float64)
    if not np.isfinite(result).all() or positive and not (result>0).all(): raise ContractError(label+" must be finite"+(" and positive" if positive else ""))
    return result


def _effective_deployment_mapping(data,external=None):
    """Combine bound deployment metadata without changing a training interface.

    S09 permits source_mapping to live in dataset.json separately from its
    DataInterface. Only new fields may enrich an embedded declaration; two
    conflicting declarations cannot silently authorize a different signal.
    """
    embedded=data.get('source_mapping',{})
    if embedded is None:embedded={}
    if not isinstance(embedded,dict) or external is not None and not isinstance(external,dict):raise ContractError('Deployment source mappings must be explicit JSON objects')
    try:
        left=json.loads(json.dumps(embedded,allow_nan=False));right={} if external is None else json.loads(json.dumps(external,allow_nan=False))
    except (TypeError,ValueError,RecursionError) as error:raise ContractError('Deployment source mapping must contain finite JSON values') from error
    def merge(base,additional,prefix):
        for key,value in additional.items():
            if key not in base:base[key]=value
            elif isinstance(base[key],dict) and isinstance(value,dict):merge(base[key],value,prefix+'.'+key)
            elif type(base[key]) is not type(value) or base[key]!=value:raise ContractError('Conflicting embedded and bound deployment source mapping at '+prefix+'.'+key)
        return base
    result=merge(left,right,'source_mapping')
    if not isinstance(result.get('signals',{}),dict) or any(not isinstance(name,str) or not isinstance(value,dict) for name,value in result.get('signals',{}).items()):raise ContractError('Deployment source_mapping.signals requires named explicit mapping objects')
    return result


def _feature_layout(data,source_mapping=None):
    if not isinstance(data,dict) or data.get("interface_kind")!="tabular_supervised" or type(data.get("schema_version")) is not int or data["schema_version"]!=1:
        raise ContractError("A version-1 explicit DataInterface is required")
    if any(key in data for key in ["nx","nu","ny","nd","states","controls"]): raise ContractError("Feature widths do not define physical state dimensions")
    signals=data.get("signals")
    if not isinstance(signals,list) or not signals or len(signals)>65536: raise ContractError("Explicit ordered feature signals required")
    by_name={}
    roles={"observation","reference","disturbance","previous_action","expert_action","diagnostic_memory"}
    for signal in signals:
        if (not isinstance(signal,dict) or not isinstance(signal.get("name"),str) or not signal["name"] or signal["name"] in by_name
                or not isinstance(signal.get("unit"),str) or not signal["unit"] or not isinstance(signal.get("expert_role"),str) or signal["expert_role"] not in roles
                or type(signal.get("observable")) is not bool or type(signal.get("delay_steps")) is not int or signal["delay_steps"]<0):
            raise ContractError("Expert signal name/unit/role/observability/delay must be explicit and unambiguous")
        by_name[signal["name"]]=signal
    deployment_mapping=_effective_deployment_mapping(data,source_mapping)
    targets=data.get("targets");features=data.get("features")
    if not isinstance(targets,list) or not targets or not isinstance(features,list) or not features: raise ContractError("Policy features/targets must be explicit nonempty lists")
    target_names=[];target_order=[]
    for target in targets:
        signal=by_name.get(target.get("signal")) if isinstance(target,dict) else None
        if (signal is None or signal["expert_role"]!="expert_action" or signal.get("role")!="control" or type(target.get("offset")) is not int
                or target["offset"]!=0 or signal["name"] in target_names): raise ContractError("Action labels must be unique current expert actions at offset zero")
        target_names.append(signal["name"]);target_order.append({"signal":signal["name"],"offset":0,"unit":signal["unit"]})
    order=[]
    for feature in features:
        signal=by_name.get(feature.get("signal")) if isinstance(feature,dict) else None
        if (signal is None or signal["expert_role"]=="diagnostic_memory" or not signal["observable"]
                or type(feature.get("window")) is not int or not 1<=feature["window"]<=65536
                or type(feature.get("lag")) is not int or feature["lag"]<0): raise ContractError("Only explicit observable causal features are supported; expert memory is diagnostic")
        if signal["expert_role"]=="expert_action" and (signal["name"] not in target_names or feature["lag"]+signal["delay_steps"]<1):
            raise ContractError("Same-current/future action target leakage is forbidden")
        if signal['expert_role']=='previous_action':
            mapping=deployment_mapping.get('signals',{}).get(signal['name'],{})
            if mapping.get('time_semantics')!='already_previous_applied_action' or mapping.get('action_target') not in target_names:raise ContractError('Previous-action feature needs an explicit actual-applied prior/action-target mapping; never shift or infer it')
        if len(order)+feature["window"]>65536: raise ContractError("Expanded policy feature width exceeds the single-network interface budget")
        for history in reversed(range(feature["window"])):
            order.append({"signal":signal["name"],"lag":feature["lag"]+signal["delay_steps"]+history,
                "unit":signal["unit"],"availability_delay_steps":signal["delay_steps"]})
    if type(data.get("input_dim")) is not int or type(data.get("output_dim")) is not int or data["input_dim"]!=len(order) or data["output_dim"]!=len(target_order):
        raise ContractError("Network dimensions disagree with expanded ordered feature/target semantics")
    if data.get('task')=='static_regression':
        if any(f['window']!=1 or f['lag']!=0 or by_name[f['signal']]['delay_steps']!=0 for f in features):raise ContractError('Static regression cannot silently acquire dynamic windows/delays')
    elif data.get('task')=='dynamic_prediction':
        if data.get('time_column') is None or type(data.get('sample_time')) not in {int,float} or not math.isfinite(data['sample_time']) or data['sample_time']<=0:raise ContractError('Dynamic policy features require explicit positive time grid semantics')
    else:raise ContractError('Unknown explicit supervised task')
    labels=[item["signal"]+"@lag="+str(item["lag"]) for item in order]
    if len(set(labels))!=len(labels): raise ContractError("Overlapping/duplicate feature labels are ambiguous")
    return {"feature_order":order,"target_order":target_order,"feature_labels":labels,
        "target_labels":[item["signal"]+"@offset=0" for item in target_order],"history_order":"feature declaration order; oldest to newest within each window; lag includes availability delay",
        "history_management":"caller supplies complete same-trajectory assembled raw feature rows; no implicit zero padding or priming"}


def _policy_metadata(data,features,scaler,constraints,physical_context,source_mapping=None):
    deployment_mapping=_effective_deployment_mapping(data,source_mapping)
    layout=_feature_layout(data,deployment_mapping)
    for key in ["feature_order","target_order","feature_labels","target_labels"]:
        if features.get(key)!=layout[key]: raise ContractError("Supervised feature/target order differs from the expert-data contract: "+key)
    ni,no=data["input_dim"],data["output_dim"]
    if not isinstance(scaler,dict) or scaler.get("method") not in {"standard","minmax","none"} or scaler.get("fit_split")!="train" or type(scaler.get("fit_samples")) is not int or scaler["fit_samples"]<1:
        raise ContractError("Policy normalization must retain the training-only fitted scaler")
    transforms={}
    for key,width in [("X",ni),("y",no)]:
        group=scaler.get(key)
        if not isinstance(group,dict): raise ContractError("Missing policy scaler group "+key)
        loc=_numeric(group.get("location"),(width,),key+" scaler location");scale=_numeric(group.get("scale"),(width,),key+" scaler scale",positive=True)
        if scaler["method"]=="none" and (np.any(loc!=0) or np.any(scale!=1)): raise ContractError("none scaler must be the identity")
        transforms[key]=(loc,scale)
    if not isinstance(constraints,dict) or constraints.get("rate_semantics")!="per_control_step" or "rate_bounds" not in constraints:
        raise ContractError("Action amplitude/rate constraints and per-control-update timing must be explicit")
    bounds=_numeric(constraints.get("bounds"),(no,2),"action bounds")
    if np.any(bounds[:,0]>=bounds[:,1]): raise ContractError("Action amplitude bounds must be ordered")
    rate=None if constraints["rate_bounds"] is None else _numeric(constraints["rate_bounds"],(no,2),"action rate bounds")
    if rate is not None and np.any(rate[:,0]>=rate[:,1]): raise ContractError("Action rate bounds must be ordered")
    if "target_order" in constraints and constraints["target_order"]!=layout["target_order"]: raise ContractError("Action-constraint target order differs")
    context=physical_context
    if not isinstance(context,dict) or "interface" not in context or context.get("inferred_from_feature_width",False) is not False:
        raise ContractError("Physical context must be explicit, with null interface for data-only inference")
    binding={"status":"unavailable","reason":"No explicit physical interface; use assembled features","current_features":[],"state_act_supported":False}
    physical=context["interface"]
    if physical is None:return layout,transforms,bounds,rate,binding
    if not isinstance(physical,dict) or physical.get("time_semantics")!="discrete_state_transition": raise ContractError("Only explicit discrete Plant context is supported")
    for group in ["states","controls","outputs","disturbances"]:
        if not isinstance(physical.get(group),list) or (group!="disturbances" and not physical[group]): raise ContractError("Physical variable groups must be explicit ordered lists")
        names=[]
        for variable in physical[group]:
            if (not isinstance(variable,dict) or not isinstance(variable.get("name"),str) or not variable["name"] or variable["name"] in names
                    or not isinstance(variable.get("unit"),str) or not variable["unit"]): raise ContractError("Physical channel names/units must be explicit")
            values=_numeric([variable.get("lower"),variable.get("upper")],(2,),"physical bounds")
            if values[0]>=values[1]:raise ContractError("Physical bounds must be ordered")
            names.append(variable["name"])
    mapping=context.get("physical_mapping")
    if not isinstance(mapping,dict):return layout,transforms,bounds,rate,{**binding,"reason":"Physical feature-to-channel mapping is not supplied"}
    dt,period,substeps=context.get("plant_dt"),context.get("control_period"),context.get("plant_substeps")
    if (type(dt) not in {int,float} or type(period) not in {int,float} or type(substeps) is not int or dt<=0 or period<=0 or not math.isfinite(dt) or not math.isfinite(period)
            or not 1<=substeps<=10000 or not math.isclose(dt,physical.get("sample_time",-1),rel_tol=1e-12,abs_tol=1e-12)
            or not math.isclose(period,dt*substeps,rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(period,data.get("sample_time",-1),rel_tol=1e-12,abs_tol=1e-12)):
        raise ContractError("Plant dt, expert period, subticks and policy data timing disagree")
    nu=len(physical["controls"])
    if no!=nu:raise ContractError("Physical actuator count differs from policy target width")
    for j,control in enumerate(physical["controls"]):
        target=layout["target_order"][j]
        if target!={"signal":"expert_action::"+control["name"],"offset":0,"unit":control["unit"]}:raise ContractError("Physical policy targets must preserve exact Plant control order/units")
        if not np.array_equal(bounds[j],np.asarray([control["lower"],control["upper"]])):raise ContractError("Physical action limits disagree with expert constraints")
    if any(item["lag"]!=0 for item in layout["feature_order"]):return layout,transforms,bounds,rate,{**binding,"reason":"Physical act/MATLAB adapter requires current features; caller must explicitly assemble declared history windows"}
    observations=mapping.get("observations")
    if not isinstance(observations,list) or not observations:return layout,transforms,bounds,rate,{**binding,"reason":"Explicit observation mapping is absent"}
    for obs in observations:
        if not isinstance(obs,dict) or obs.get("source") not in {"state","output"} or type(obs.get("index")) is not int:raise ContractError("Physical observations need explicit state/output coordinate mappings")
        variables=physical["states"] if obs["source"]=="state" else physical["outputs"]
        if not 0<=obs["index"]<len(variables) or obs.get("plant_name")!=variables[obs["index"]]["name"] or obs.get("unit")!=variables[obs["index"]]["unit"] or obs.get("delay_steps")!=0:
            raise ContractError("Physical observation mapping index/name/unit/delay differs")
    signal_map=deployment_mapping.get("signals",{})
    by_name={s["name"]:s for s in data["signals"]};entries=[]
    for item in layout["feature_order"]:
        signal=by_name[item["signal"]];role=signal["expert_role"];source=signal_map.get(signal["name"],{});index=source.get("column");array=source.get("array")
        if type(index) is not int:return layout,transforms,bounds,rate,{**binding,"reason":"Feature mapping lacks explicit logical array coordinates"}
        if role=="observation" and array=="observation" and 0<=index<len(observations):
            obs=observations[index]
            if signal["name"]!="observation::"+obs["name"] or signal["unit"]!=obs["unit"]:raise ContractError("Observation feature identity/unit differs from declared physical observation")
            entries.append({"kind":"observation","index":index,"source":obs["source"],"physical_index":obs["index"]})
        elif role in {"reference","disturbance","previous_action"}:
            group,prefix,expected={"reference":("outputs","reference","reference"),"disturbance":("disturbances","disturbance","d"),"previous_action":("controls","previous_action","previous_u")}[role]
            variables=physical[group]
            if array!=expected or not 0<=index<len(variables) or signal["name"]!=prefix+"::"+variables[index]["name"] or signal["unit"]!=variables[index]["unit"]:
                raise ContractError("Physical reference/disturbance/previous-action feature order/units differ")
            if role=="previous_action" and (source.get("time_semantics")!="already_previous_applied_action" or source.get("action_target")!="expert_action::"+variables[index]["name"]):raise ContractError("Previous action mapping must identify actual applied prior command without a second shift")
            entries.append({"kind":role,"index":index})
        else:return layout,transforms,bounds,rate,{**binding,"reason":"Selected feature requires an unsupported explicit physical adapter"}
    binding={"status":"supported_current_mapping","current_features":entries,"observations":observations,
        "state_act_supported":all(e.get("source")!="output" for e in entries),"control_period":period,"plant_dt":dt,"plant_substeps":substeps,
        "availability_scope":"simulation_only" if any(by_name[f["signal"]].get("availability")=="simulation_only" for f in layout["feature_order"]) else "user_declared_available_not_certified"}
    return layout,transforms,bounds,rate,binding


def _verify_files(root,manifest):
    if not isinstance(manifest,dict) or manifest.get("kind")!="control-external-imitation-policy-v1" or manifest.get("schema_version")!=1 or not isinstance(manifest.get("files"),dict):raise ContractError("Unsupported external policy manifest")
    required={"policy.json","model.pt","network.json","scaler.json","data_interface.json","features.json","action_constraints.json","physical_context.json","policy_runtime.py","infer.py","README.md"}
    actual=set(manifest["files"])
    if actual not in [required,required|{"external_policy_nn.m","matlab_parameters.json"}]:raise ContractError("External policy member inventory differs")
    for relative,digest in manifest["files"].items():
        path=(root/relative).resolve(strict=True)
        if path.parent!=root or not path.is_file() or path.stat().st_size>64*1024*1024 or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ContractError("Invalid external policy bundle member")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ContractError("External policy member hash differs: "+relative)


def project_actions(raw_actions,previous_action=None,action_constraints=None):
    """Project saved physical-unit predictions without executing a network."""
    raw=np.asarray(raw_actions)
    if raw.ndim not in {1,2} or raw.shape[-1]<1 or raw.size>5_000_000 or raw.ndim==2 and not len(raw):raise ContractError('Raw actions need a nonempty vector or samples-by-actions matrix')
    raw=_numeric(raw_actions,raw.shape,'raw actions')
    constraints=action_constraints
    if not isinstance(constraints,dict) or constraints.get('rate_semantics')!='per_control_step' or 'rate_bounds' not in constraints:raise ContractError('Explicit per-control-update action constraints required')
    bounds=_numeric(constraints.get('bounds'),(raw.shape[-1],2),'action bounds');low,high=bounds[:,0],bounds[:,1]
    if np.any(low>=high):raise ContractError('Action amplitude bounds must be ordered')
    if constraints['rate_bounds'] is not None:
        rate=_numeric(constraints['rate_bounds'],bounds.shape,'action rate bounds')
        if np.any(rate[:,0]>=rate[:,1]):raise ContractError('Action rate bounds must be ordered')
        if previous_action is None:raise ContractError('Rate projection requires actual previous applied action; no implicit zero')
        previous=_numeric(previous_action,raw.shape,'actual previous action')
        if np.any(previous<low-1e-12) or np.any(previous>high+1e-12):raise ContractError('Actual previous action violates amplitude bounds')
        low=np.maximum(low,previous+rate[:,0]);high=np.minimum(high,previous+rate[:,1])
        if np.any(low>high):raise ContractError('Amplitude/rate intersection is empty')
    elif previous_action is not None:_numeric(previous_action,raw.shape,'actual previous action')
    return np.maximum(np.minimum(raw,high),low)


class ExternalPolicy(nn.Module):
    def __init__(self,directory):
        super().__init__();self.root=Path(directory).resolve(strict=True)
        manifest=_read_json(self.root/'policy_manifest.json');_verify_files(self.root,manifest)
        self.descriptor=_read_json(self.root/'policy.json');self.data_interface=_read_json(self.root/'data_interface.json')
        self.features=_read_json(self.root/'features.json');self.scaler=_read_json(self.root/'scaler.json')
        self.action_constraints=_read_json(self.root/'action_constraints.json');self.physical_context=_read_json(self.root/'physical_context.json')
        self.config=validate_network_config(_read_json(self.root/'network.json'))
        provenance=self.descriptor.get('provenance',{})
        if not isinstance(provenance,dict):raise ContractError('Policy provenance must be an explicit JSON object')
        source_mapping=provenance.get('source_mapping')
        self.deployment_mapping=_effective_deployment_mapping(self.data_interface,source_mapping)
        layout,transforms,bounds,rate,binding=_policy_metadata(self.data_interface,self.features,self.scaler,self.action_constraints,self.physical_context,source_mapping)
        selected={entry['signal'] for entry in layout['feature_order']}
        self.deployment_scope='simulation_only' if any(s.get('availability')=='simulation_only' and s['name'] in selected for s in self.data_interface['signals']) else 'user_declared_available_not_certified'
        if self.config['input_dim']!=self.data_interface['input_dim'] or self.config['output_dim']!=self.data_interface['output_dim']:raise ContractError("Policy network/feature dimensions differ")
        if self.descriptor.get('representation')!='external_imitation_mlp' or self.descriptor.get('feature_labels')!=layout['feature_labels'] or self.descriptor.get('target_labels')!=layout['target_labels'] or self.descriptor.get('physical_binding')!=binding:
            raise ContractError("Policy descriptor does not match actual feature/physical metadata")
        if self.descriptor.get('deployment_scope')!=self.deployment_scope or manifest.get('deployment_scope')!=self.deployment_scope:raise ContractError('Policy deployment scope differs from selected signal availability')
        if manifest.get('input_dim')!=self.config['input_dim'] or manifest.get('output_dim')!=self.config['output_dim'] or manifest.get('feature_labels')!=layout['feature_labels'] or manifest.get('target_labels')!=layout['target_labels']:raise ContractError('Policy manifest dimensions/feature labels differ')
        self.input_dim,self.output_dim=self.config['input_dim'],self.config['output_dim'];self.binding=binding;self.layout=layout
        with torch.random.fork_rng(devices=[]):self.network=build_supervised_network(self.config)
        state=torch.load(self.root/'model.pt',map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in state.values()):raise ContractError("Policy weights must be finite float64 tensor state_dict entries")
        self.network.load_state_dict(state,strict=True);self.network.requires_grad_(False);self.network.eval()
        for key,(location,scale) in transforms.items():
            self.register_buffer(key+'_location',torch.as_tensor(location,dtype=torch.float64));self.register_buffer(key+'_scale',torch.as_tensor(scale,dtype=torch.float64))
        self.register_buffer('action_low',torch.as_tensor(bounds[:,0],dtype=torch.float64));self.register_buffer('action_high',torch.as_tensor(bounds[:,1],dtype=torch.float64))
        self.rate_enabled=rate is not None
        if rate is not None:
            self.register_buffer('rate_low',torch.as_tensor(rate[:,0],dtype=torch.float64));self.register_buffer('rate_high',torch.as_tensor(rate[:,1],dtype=torch.float64))
        self.decisions=0;self.eval()

    def _tensor(self,value,width,label):
        if isinstance(value,torch.Tensor):
            if value.dtype!=torch.float64 or value.is_complex():raise ContractError(label+' Tensor must be real float64')
            result=value
        else:
            raw=np.asarray(value)
            def boolean(v):return isinstance(v,(bool,np.bool_)) or isinstance(v,(list,tuple)) and any(boolean(c) for c in v)
            if boolean(value) or raw.dtype.kind not in 'fiu':raise ContractError(label+' must be numeric without booleans/strings')
            result=torch.as_tensor(raw,dtype=torch.float64,device=self.action_low.device)
        if result.ndim not in {1,2} or result.shape[-1]!=width or result.numel()>5_000_000 or (result.ndim==2 and not len(result)) or not torch.isfinite(result).all():raise ContractError(label+' has invalid finite vector/batch shape')
        if result.device!=self.action_low.device:raise ContractError(label+' device differs from policy buffers')
        return result

    def raw_forward(self,features,feature_labels=None):
        if feature_labels is not None and list(feature_labels)!=self.layout['feature_labels']:raise ContractError('Provided feature labels/order differ')
        x=self._tensor(features,self.input_dim,'features');rows=1 if x.ndim==1 else len(x)
        if rows*sum([self.input_dim,*self.config['hidden_dims'],self.output_dim])>5_000_000:raise ContractError('Policy inference exceeds its single-batch activation resource budget')
        scaled=(x-self.X_location)/self.X_scale
        if not torch.isfinite(scaled).all():raise ContractError('Policy input normalization overflowed')
        raw=self.network(scaled)*self.y_scale+self.y_location
        if not torch.isfinite(raw).all():raise ContractError('Policy raw action is nonfinite; projection cannot hide a failed network')
        return raw

    def project_tensor(self,raw_actions,previous_action=None):
        raw=self._tensor(raw_actions,self.output_dim,'raw action');low,high=self.action_low,self.action_high
        if self.rate_enabled:
            if previous_action is None:raise ContractError('Rate-constrained policy requires actual previous applied action; no implicit zero')
            previous=self._tensor(previous_action,self.output_dim,'actual previous action')
            if previous.shape!=raw.shape:raise ContractError('Previous action must have exactly the same vector/batch shape; no implicit broadcast')
            if torch.any(previous<low-1e-12) or torch.any(previous>high+1e-12):raise ContractError('Actual previous action violates amplitude bounds')
            low=torch.maximum(low,previous+self.rate_low);high=torch.minimum(high,previous+self.rate_high)
            if torch.any(low>high):raise ContractError('Amplitude/rate intersection is empty; no silent fallback')
        elif previous_action is not None:
            previous=self._tensor(previous_action,self.output_dim,'actual previous action')
            if previous.shape!=raw.shape:raise ContractError('Previous action shape differs from raw action')
        return torch.maximum(torch.minimum(raw,high),low)

    def forward(self,features,previous_action=None,feature_labels=None):
        return self.project_tensor(self.raw_forward(features,feature_labels),previous_action)

    def predict_raw(self,features,feature_labels=None):
        with torch.no_grad():return self.raw_forward(features,feature_labels).detach().cpu().numpy().copy()

    def predict(self,features,previous_action=None,feature_labels=None):
        with torch.no_grad():return self(features,previous_action,feature_labels).detach().cpu().numpy().copy()

    def project_actions(self,raw_actions,previous_action=None):
        with torch.no_grad():return self.project_tensor(raw_actions,previous_action).detach().cpu().numpy().copy()

    def numpy_raw(self,features,feature_labels=None):
        x=self._tensor(features,self.input_dim,'features').detach().cpu().numpy()
        rows=1 if x.ndim==1 else len(x)
        if rows*sum([self.input_dim,*self.config['hidden_dims'],self.output_dim])>5_000_000:raise ContractError('NumPy policy inference exceeds its single-batch activation resource budget')
        if feature_labels is not None and list(feature_labels)!=self.layout['feature_labels']:raise ContractError('Provided feature labels/order differ')
        x=(x-self.X_location.cpu().numpy())/self.X_scale.cpu().numpy()
        if not np.isfinite(x).all():raise ContractError('NumPy policy input normalization overflowed')
        for index,module in enumerate(self.network):
            if isinstance(module,nn.Linear):x=x@module.weight.detach().cpu().numpy().T+module.bias.detach().cpu().numpy()
            elif isinstance(module,nn.Tanh):x=np.tanh(x)
            elif isinstance(module,nn.ReLU):x=np.maximum(x,0.)
            elif isinstance(module,nn.SiLU):
                with np.errstate(over='ignore'):x=x/(1.+np.exp(-x))
            else:raise ContractError('Unknown network activation in numeric reference')
        result=x*self.y_scale.cpu().numpy()+self.y_location.cpu().numpy()
        if not np.isfinite(result).all():raise ContractError('NumPy raw policy prediction is nonfinite')
        return result

    def act(self,x,reference,disturbance,previous):
        if self.binding['status']!='supported_current_mapping' or not self.binding['state_act_supported']:raise ContractError('Physical act unavailable: explicit current state-observation mapping required; use assembled features')
        physical=self.physical_context['interface'];x=self._tensor(x,len(physical['states']),'physical state')
        r=self._tensor(reference,len(physical['outputs']),'reference');d=self._tensor(disturbance,len(physical['disturbances']),'disturbance')
        p=self._tensor(previous,self.output_dim,'actual previous action') if previous is not None else None
        if r.shape[:-1]!=x.shape[:-1] or d.shape[:-1]!=x.shape[:-1] or p is not None and p.shape[:-1]!=x.shape[:-1]:raise ContractError('Physical act channel batch shapes differ')
        values=[]
        for entry in self.binding['current_features']:
            if entry['kind']=='observation':value=x[...,entry['physical_index']]
            elif entry['kind']=='reference':value=r[...,entry['index']]
            elif entry['kind']=='disturbance':value=d[...,entry['index']]
            else:
                if p is None:raise ContractError('Previous-action feature requires actual previous action')
                value=p[...,entry['index']]
            values.append(value)
        action=self(torch.stack(values,dim=-1),p);self.decisions+=1
        return action

    def reset(self):
        self.decisions=0
        return {'hidden_state':'none','history':'caller_managed; no implicit priming'}

    def describe(self):
        return {'representation':'external_imitation_mlp','input_dim':self.input_dim,'output_dim':self.output_dim,'data_interface':self.data_interface,
            'physical_binding':self.binding,'deployment_scope':self.deployment_scope,'action_projection':'physical amplitude and actual previous action rate intersection',
            'rate_semantics':'per_control_step','history_management':self.layout['history_management'],'hidden_state':'none','weights_frozen':True,
            'input_gradients':'connected through scaling/network/inverse scaling/projection; clamped directions may have zero derivative'}


def load_external_policy(root):
    root=Path(root).resolve(strict=True)
    if not (root/'policy_manifest.json').is_file():
        descriptor=_read_json(root/'policy.json')
        if descriptor.get('bundle')!='bundle':raise ContractError('Policy artifact must select its fixed relative bundle')
        bundle=(root/'bundle').resolve(strict=True)
        if bundle.parent!=root:raise ContractError('Policy bundle escapes artifact')
        root=bundle
    return ExternalPolicy(root)
