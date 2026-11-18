"""Portable basic-design packages using the existing MPCPolicy numerical source.

No tuning or closed-loop execution occurs during export. The MATLAB affine
QP is a readable independent solver template; Optimization Toolbox availability
and cross-solver numerical agreement require actual MATLAB validation.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path, PureWindowsPath

import numpy as np

from .contracts import ContractError
from .control import MPCPolicy, SolverFailure
from .modeling import tensor

SPRING_KEYS = ('mass','damping','stiffness','sample_time','state_bounds','input_bounds','output_bounds')
MATLAB_SOURCES = {
    'quadprog':'https://www.mathworks.com/help/optim/ug/quadprog.html',
    'optimoptions':'https://www.mathworks.com/help/optim/ug/optimoptions.html',
    'expm':'https://www.mathworks.com/help/matlab/ref/expm.html',
}


def _portable(value):
    """Keep engineering JSON while omitting original absolute storage locations."""
    if isinstance(value,dict):
        return {key:_portable(item) for key,item in value.items() if key not in
            {'workspace','source_root','artifact_path','bundle_root','manifest_path','local_path','input_path','model_path','data_path'}}
    if isinstance(value,(list,tuple)):return [_portable(item) for item in value]
    if isinstance(value,str) and (Path(value).is_absolute() or PureWindowsPath(value).is_absolute()):return '<original-location-omitted>'
    return value


def _write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def _runtime_source():
    header='''"""Exact project MPCPolicy source; no Harness/workspace services are bundled."""
from __future__ import annotations
import copy,math,time
import numpy as np
import torch
from scipy.optimize import minimize
class ContractError(ValueError): pass
class OperationContext:
    """Standalone adapter; callers may supply an object with check()."""
    def check(self): pass
DTYPE=torch.float64
'''
    return header+'\n\n'.join(inspect.getsource(obj) for obj in (tensor,SolverFailure,MPCPolicy))+'\n'


_CONTROLLER = r'''"""Standalone parameterized design. Solving one point is not closed-loop validation."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.linalg import expm

class NotInstantiated(ValueError): pass
_SPRING=('mass','damping','stiffness','sample_time','state_bounds','input_bounds','output_bounds')

def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def _verify(root):
    manifest=json.loads((root/'bundle_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='control-controller-design-bundle-v1' or manifest.get('schema_version')!=1:raise ValueError('Unsupported controller-design bundle')
    for name,digest in manifest['files'].items():
        file=(root/name).resolve(strict=True)
        if not file.is_relative_to(root) or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:raise ValueError('Controller bundle member changed: '+name)
    return manifest

def _number(value,name,positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (positive and value<=0):raise ValueError(name+' requires a finite '+('positive' if positive else 'nonnegative')+' number')
    return float(value)

def _bounds(value,width,name):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.shape!=(width,2) or any(isinstance(item,bool) for item in np.asarray(value,dtype=object).flat):raise ValueError(name+' must be an explicit numeric '+str(width)+' by 2 matrix')
    raw=raw.astype(float)
    if not np.isfinite(raw).all() or np.any(raw[:,0]>=raw[:,1]):raise ValueError(name+' requires finite ordered bounds')
    return raw

class SpringPredictor(torch.nn.Module):
    def __init__(self,parameters):
        super().__init__()
        missing=[key for key in _SPRING if key not in parameters or parameters[key] is None]
        if missing:raise NotInstantiated('Missing physical parameters: '+', '.join(missing))
        if set(parameters)-set(_SPRING):raise ValueError('Unknown physical parameter keys')
        m=_number(parameters['mass'],'mass',True);c=_number(parameters['damping'],'damping');k=_number(parameters['stiffness'],'stiffness');dt=_number(parameters['sample_time'],'sample_time',True)
        xb=_bounds(parameters['state_bounds'],2,'state_bounds');ub=_bounds(parameters['input_bounds'],1,'input_bounds');yb=_bounds(parameters['output_bounds'],1,'output_bounds')
        generator=np.zeros((3,3));generator[:2,:2]=[[0.,1.],[-k/m,-c/m]];generator[:2,2]=[0.,1./m];transition=expm(generator*dt)
        for name,value in [('A',transition[:2,:2]),('B',transition[:2,2:3]),('E',np.zeros((2,0))),('bias',np.zeros(2)),('C',np.array([[1.,0.]]))]:self.register_buffer(name,torch.as_tensor(value,dtype=torch.float64))
        variable=lambda name,unit,b:SimpleNamespace(name=name,unit=unit,lower=float(b[0]),upper=float(b[1]))
        self.interface=SimpleNamespace(nx=2,nu=1,ny=1,nd=0,sample_time=dt,time_semantics='discrete_state_transition',
            states=(variable('position','m',xb[0]),variable('velocity','m/s',xb[1])),controls=(variable('force','N',ub[0]),),outputs=(variable('position','m',yb[0]),),disturbances=())
    def forward(self,x,u,d=None):
        if x.shape[-1]!=2 or u.shape[-1]!=1 or x.shape[:-1]!=u.shape[:-1] or (d is not None and (d.shape[-1]!=0 or d.shape[:-1]!=x.shape[:-1])):raise ValueError('Spring x/u/d shape mismatch')
        return x@self.A.T+u@self.B.T+self.bias
    def output(self,x):return x@self.C.T

def _config(design,override):
    config={**design.get('controller_config',{}),**design.get('weights',{})}
    if override:config.update(override)
    if set(config)-{'horizon','q','r','s','input_rate_bounds','solver_tolerance','max_iterations'}:raise ValueError('Unknown controller setting')
    for key in ('horizon','max_iterations'):
        if type(config.get(key)) is not int or config[key]<1:raise ValueError(key+' requires an explicit positive integer')
    if config['horizon']>100:raise ValueError('MPC horizon must be at most 100')
    for key in ('q','r','s','solver_tolerance'):_number(config.get(key),key,key in ('q','solver_tolerance'))
    return config

class GuardedPolicy:
    """Preserve the original optimizer; bound one dense solve/Jacobian call."""
    def __init__(self,policy):self._policy=policy
    def __getattr__(self,name):return getattr(self._policy,name)
    @property
    def warm(self):return self._policy.warm
    @warm.setter
    def warm(self,value):self._policy.warm=value
    def act(self,*args,**kwargs):
        p=self._policy;i=p.interface;variables=p.horizon*i.nu
        margins=p.horizon*(2*i.nx+2*i.ny+(2*i.nu if p.rate is not None else 0))
        if variables>256 or variables*margins>5_000_000:raise ValueError('Single-point MPC resource guard exceeded; reduce horizon/width before numerical solve')
        return p.act(*args,**kwargs)

class Design:
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True);self.manifest=_verify(self.root)
        self.definition=json.loads((self.root/'design.json').read_text(encoding='utf-8'))
        self.model_instantiated=bool(self.definition.get('model_instantiated'))
    def instantiate(self,parameters=None,controller_config=None,context=None):
        definition=self.definition
        if definition.get('template')=='spring_damper' and (parameters is not None or not self.model_instantiated):
            merged={**definition.get('parameters',{}),**(parameters or {})};model=SpringPredictor(merged)
        elif not self.model_instantiated:raise NotInstantiated('Design is not instantiated; provide an explicit supported predictor and complete engineering parameters')
        elif parameters:raise ValueError('This frozen predictor cannot be changed by a physical-parameter override')
        else:
            source=self.root/'prediction';loader=_module(source/'predictor.py','_design_predictor')
            model=loader.load_predictor(source)
        runtime=_module(self.root/'mpc_runtime.py','_design_mpc_runtime')
        return GuardedPolicy(runtime.MPCPolicy(model,_config(definition,controller_config),context))
    def solve(self,x,reference,disturbance=None,previous=None,parameters=None):
        return self.instantiate(parameters).act(x,reference,disturbance,previous)

def load_design(directory=None):return Design(directory or Path(__file__).resolve().parent)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='One MPC solve; no GA or closed-loop simulation')
    parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--parameters')
    args=parser.parse_args();case=json.loads(Path(args.input).read_text(encoding='utf-8'));parameters=json.loads(Path(args.parameters).read_text(encoding='utf-8')) if args.parameters else None
    u,diagnostics=load_design().solve(case['x'],case['reference'],case.get('disturbance'),case.get('previous'),parameters)
    Path(args.output).write_text(json.dumps({'u':u.tolist(),'diagnostics':diagnostics,'operation':'single_point_solve','closed_loop':'not_run','ga_trials':0},allow_nan=False,indent=2)+'\n',encoding='utf-8')
'''


_MATLAB_MODEL = r'''function model = prediction_model(parameters)
% Explicit spring parameters or frozen affine predictor; never an example default.
if nargin < 1, parameters = struct(); end
root = fileparts(mfilename('fullpath'));
saved = jsondecode(fileread(fullfile(root,'matlab_model.json')));
if strcmp(saved.kind,'affine') && isempty(fieldnames(parameters))
    model = saved;
elseif strcmp(saved.template,'spring_damper')
    required = {'mass','damping','stiffness','sample_time','state_bounds','input_bounds','output_bounds'};
    design = jsondecode(fileread(fullfile(root,'design.json')));
    values = design.parameters; names = fieldnames(parameters);
    for index=1:numel(names), values.(names{index})=parameters.(names{index}); end
    assert(all(isfield(values,required)),'ControlDesign:NotInstantiated','All seven physical parameters are required; no numerical example defaults.');
    assert(numel(fieldnames(values))==numel(required),'ControlDesign:Parameters','Unknown physical parameters.');
    m=values.mass; c=values.damping; k=values.stiffness; dt=values.sample_time;
    assert(isnumeric(m)&&isscalar(m)&&isfinite(m)&&m>0 && isnumeric(c)&&isscalar(c)&&isfinite(c)&&c>=0 && isnumeric(k)&&isscalar(k)&&isfinite(k)&&k>=0 && isnumeric(dt)&&isscalar(dt)&&isfinite(dt)&&dt>0,'ControlDesign:Parameters','Invalid SI physical parameters.');
    assert(isequal(size(values.state_bounds),[2,2])&&isequal(size(values.input_bounds),[1,2])&&isequal(size(values.output_bounds),[1,2]),'ControlDesign:Bounds','Explicit state/input/output bound matrices are required.');
    bounds=[values.state_bounds;values.input_bounds;values.output_bounds];
    assert(isreal(bounds)&&all(isfinite(bounds),'all')&&all(bounds(:,1)<bounds(:,2)),'ControlDesign:Bounds','Bounds must be finite ordered real values.');
    generator=zeros(3);generator(1:2,1:2)=[0,1;-k/m,-c/m];generator(1:2,3)=[0;1/m];transition=expm(generator*dt);
    model=struct('kind','affine','nx',2,'nu',1,'ny',1,'nd',0,'sample_time',dt,'A',transition(1:2,1:2),'B',transition(1:2,3),'E',zeros(2,0),'bias',zeros(2,1),'C',[1,0], ...
        'state_bounds',values.state_bounds,'input_bounds',values.input_bounds,'output_bounds',values.output_bounds);
else
    error('ControlDesign:PredictorUnsupported','MATLAB numerical template requires an explicit affine predictor; use the exported Python NMPC for supported direct nonlinear models.');
end
if model.nd==0, model.E=zeros(model.nx,0); end
model.A=reshape(model.A,model.nx,model.nx);model.B=reshape(model.B,model.nx,model.nu);model.E=reshape(model.E,model.nx,model.nd);model.C=reshape(model.C,model.ny,model.nx);model.bias=reshape(model.bias,model.nx,1);
end
'''

_MATLAB_STEP = r'''function [next_x,y] = predict_step(x,u,d,parameters)
if nargin<3,d=[];end
if nargin<4,parameters=struct();end
model=prediction_model(parameters);
assert(isnumeric(x)&&isreal(x)&&numel(x)==model.nx&&all(isfinite(x),'all'),'ControlDesign:Shape','x width/finite mismatch');
assert(isnumeric(u)&&isreal(u)&&numel(u)==model.nu&&all(isfinite(u),'all'),'ControlDesign:Shape','u width/finite mismatch');
assert(isnumeric(d)&&isreal(d)&&numel(d)==model.nd&&all(isfinite(d),'all'),'ControlDesign:Shape','d width/finite mismatch');
next_x=model.A*x(:)+model.B*u(:)+model.E*d(:)+model.bias;
y=model.C*next_x;
end
'''

_MATLAB_CONTROL = r'''function [u,diagnostics,U] = mpc_control(x,reference,d,previous,parameters)
% Same normalized q/r/s and hard constraints as the Python MPCPolicy.
% Independent affine QP solver: Optimization Toolbox quadprog is required.
% This function performs one solve only; no GA, simulation or validation claim.
if nargin<3,d=[];end
if nargin<4,previous=[];end
if nargin<5,parameters=struct();end
assert(exist('quadprog','file')==2,'ControlDesign:ToolboxRequired','Optimization Toolbox quadprog is required; license availability was not certified during export.');
model=prediction_model(parameters);root=fileparts(mfilename('fullpath'));design=jsondecode(fileread(fullfile(root,'design.json')));
cfg=design.controller_config;weights=design.weights;H=cfg.horizon;n=model.nu;nx=model.nx;ny=model.ny;
V=H*n;M=H*(2*nx+2*ny+2*n*(isfield(cfg,'input_rate_bounds')&&~isempty(cfg.input_rate_bounds)));
assert(V<=256&&V*M<=5000000,'ControlDesign:ResourceGuard','Reduce horizon/width before a dense single-point solve.');
x=x(:);reference=reference(:);d=d(:);
assert(numel(x)==nx&&numel(reference)==ny&&numel(d)==model.nd&&isreal([x;reference;d])&&all(isfinite([x;reference;d])),'ControlDesign:Shape','State/reference/disturbance width or values mismatch.');
lo=model.input_bounds(:,1);hi=model.input_bounds(:,2);
if isempty(previous),previous=min(max(zeros(n,1),lo),hi);else,previous=previous(:);end
assert(numel(previous)==n&&isreal(previous)&&all(isfinite(previous)),'ControlDesign:Shape','Previous action mismatch.');
assert(H==floor(H)&&H>=1&&H<=100&&weights.q>0&&weights.r>=0&&weights.s>=0,'ControlDesign:Config','Invalid horizon/q/r/s.');
XP=zeros(H*nx,H*n);XF=zeros(H*nx,1);YP=zeros(H*ny,H*n);YF=zeros(H*ny,1);P=zeros(nx,H*n);f=x;
for k=1:H
    f=model.A*f+model.E*d+model.bias;P=model.A*P;P(:,(k-1)*n+(1:n))=P(:,(k-1)*n+(1:n))+model.B;
    XP((k-1)*nx+(1:nx),:)=P;XF((k-1)*nx+(1:nx))=f;YP((k-1)*ny+(1:ny),:)=model.C*P;YF((k-1)*ny+(1:ny))=model.C*f;
end
yrange=model.output_bounds(:,2)-model.output_bounds(:,1);urange=hi-lo;
 % Keep diagonal weights sparse: an accepted V*M budget must not allocate (H*ny)^2 values.
 Q=kron(speye(H),spdiags(weights.q./yrange.^2,0,ny,ny));R=kron(speye(H),spdiags(weights.r./urange.^2,0,n,n));S=kron(speye(H),spdiags(weights.s./urange.^2,0,n,n));
D=eye(H*n);for k=2:H,D((k-1)*n+(1:n),(k-2)*n+(1:n))=-eye(n);end
p0=[previous;zeros((H-1)*n,1)];error0=YF-repmat(reference,H,1);
Hessian=2*(YP'*Q*YP+R+D'*S*D);Hessian=(Hessian+Hessian')/2;linear=2*(YP'*Q*error0-D'*S*p0);
A=[XP;-XP;YP;-YP];b=[repmat(model.state_bounds(:,2),H,1)-XF;XF-repmat(model.state_bounds(:,1),H,1);repmat(model.output_bounds(:,2),H,1)-YF;YF-repmat(model.output_bounds(:,1),H,1)];
if isfield(cfg,'input_rate_bounds')&&~isempty(cfg.input_rate_bounds)
    rate=cfg.input_rate_bounds;assert(isequal(size(rate),[n,2])&&all(isfinite(rate),'all')&&all(rate(:,1)<rate(:,2)),'ControlDesign:Rate','Rate bounds are per-sample action increments.');
    A=[A;D;-D];b=[b;repmat(rate(:,2),H,1)+p0;-repmat(rate(:,1),H,1)-p0];
end
options=optimoptions('quadprog','Display','off','OptimalityTolerance',cfg.solver_tolerance,'MaxIterations',cfg.max_iterations);
initial=repmat(previous,H,1);[flat,~,exitflag,output]=quadprog(Hessian,linear,A,b,[],[],repmat(lo,H,1),repmat(hi,H,1),initial,options);
assert(exitflag>0&&numel(flat)==H*n&&all(isfinite(flat)),'ControlDesign:SolveFailed','quadprog did not return a finite converged solution.');
violation=max([0;A*flat-b;repmat(lo,H,1)-flat;flat-repmat(hi,H,1)]);
assert(violation<=1e-6,'ControlDesign:Infeasible','Predicted hard-constraint violation exceeds 1e-6.');
tracking=YP*flat+error0;delta=D*flat-p0;objective=tracking'*Q*tracking+flat'*R*flat+delta'*S*delta;
U=reshape(flat,n,H).';u=U(1,:).';diagnostics=struct('solver','quadprog','exitflag',exitflag,'iterations',output.iterations,'predicted_objective',objective,'constraint_violation',violation,'closed_loop','not_run','ga_trials',0);
end
'''

_MATLAB_COST = r'''function [cost,margins] = mpc_cost(U,x,reference,d,previous,parameters)
% Per-step normalization/order matches the exported Python MPCPolicy.rollout.
if nargin<4,d=[];end
if nargin<5,previous=[];end
if nargin<6,parameters=struct();end
model=prediction_model(parameters);root=fileparts(mfilename('fullpath'));design=jsondecode(fileread(fullfile(root,'design.json')));cfg=design.controller_config;w=design.weights;
H=cfg.horizon;n=model.nu;x=x(:);reference=reference(:);d=d(:);
assert(isequal(size(U),[H,n])&&isreal(U)&&all(isfinite(U),'all'),'ControlDesign:Shape','Sequence must be H-by-nu real finite actions.');
assert(numel(x)==model.nx&&numel(reference)==model.ny&&numel(d)==model.nd&&isreal([x;reference;d])&&all(isfinite([x;reference;d])),'ControlDesign:Shape','State/reference/disturbance mismatch.');
lo=model.input_bounds(:,1);hi=model.input_bounds(:,2);if isempty(previous),previous=min(max(zeros(n,1),lo),hi);else,previous=previous(:);end
assert(numel(previous)==n&&isreal(previous)&&all(isfinite(previous)),'ControlDesign:Shape','Previous action mismatch.');
yscale=model.output_bounds(:,2)-model.output_bounds(:,1);uscale=hi-lo;states=zeros(H,model.nx);outputs=zeros(H,model.ny);increments=zeros(H,n);cost=0;
for k=1:H
    action=U(k,:).';x=model.A*x+model.B*action+model.E*d+model.bias;y=model.C*x;delta=action-previous;
    cost=cost+w.q*sum(((y-reference)./yscale).^2)+w.r*sum((action./uscale).^2)+w.s*sum((delta./uscale).^2);
    states(k,:)=x.';outputs(k,:)=y.';increments(k,:)=delta.';previous=action;
end
margins=[reshape((states-model.state_bounds(:,1).').',[],1);reshape((model.state_bounds(:,2).'-states).',[],1);reshape((outputs-model.output_bounds(:,1).').',[],1);reshape((model.output_bounds(:,2).'-outputs).',[],1)];
if isfield(cfg,'input_rate_bounds')&&~isempty(cfg.input_rate_bounds)
    rate=cfg.input_rate_bounds;margins=[margins;reshape((increments-rate(:,1).').',[],1);reshape((rate(:,2).'-increments).',[],1)];
end
assert(isfinite(cost)&&all(isfinite(margins)),'ControlDesign:Nonfinite','Nonfinite cost/constraints.');
end
'''

_MATLAB_SELFCHECK = r'''function result = selfcheck_controller(caseJsonPath,physicalParametersJsonPath)
% Fixed one-point entrypoint for independent caller-controlled acceptance.
% Does not simulate a plant, tune weights or certify closed-loop behaviour.
if nargin<1,caseJsonPath='';end
if nargin<2,physicalParametersJsonPath='';end
root=fileparts(mfilename('fullpath'));design=jsondecode(fileread(fullfile(root,'design.json')));
result=struct('status','not_run','model_instantiated',false,'prediction',[],'cost',[],'action',[],'constraints',[], ...
    'closed_loop','not_run','ga_trials',0,'solver','quadprog','optimization_toolbox_license','not_checked');
% Loading the declared function headers makes MATLAB parse each source file.
names={'prediction_model','predict_step','mpc_cost','mpc_control'};
for index=1:numel(names),assert(strcmp(fileparts(which(names{index})),root),'ControlDesign:FunctionIdentity','A helper resolves outside this design bundle.');end
result.function_headers=[nargin('prediction_model'),nargin('predict_step'),nargin('mpc_cost'),nargin('mpc_control')];
parameters=struct();if ~isempty(physicalParametersJsonPath),parameters=jsondecode(fileread(physicalParametersJsonPath));end
try
    model=prediction_model(parameters);
catch problem
    if ~design.model_instantiated && (strcmp(problem.identifier,'ControlDesign:NotInstantiated')||strcmp(problem.identifier,'ControlDesign:PredictorUnsupported'))
        result.status='not_instantiated';result.missing_parameters_rejected=true;result.error_identifier=problem.identifier;return
    end
    rethrow(problem)
end
result.model_instantiated=true;
assert(~isempty(caseJsonPath),'ControlDesign:ExplicitCase','Provide an explicit nonzero case JSON for a numerical check.');
data=jsondecode(fileread(caseJsonPath));x=data.x(:);reference=data.reference(:);
d=zeros(model.nd,1);if isfield(data,'disturbance'),d=data.disturbance(:);elseif model.nd>0,error('ControlDesign:ExplicitDisturbance','Provide declared disturbances.');end
previous=min(max(zeros(model.nu,1),model.input_bounds(:,1)),model.input_bounds(:,2));if isfield(data,'previous'),previous=data.previous(:);end
U=repmat(previous.',design.controller_config.horizon,1);if isfield(data,'sequence'),U=data.sequence;end
[next_x,next_y]=predict_step(x,U(1,:).',d,parameters);[cost,margins]=mpc_cost(U,x,reference,d,previous,parameters);
result.prediction=struct('next_x',next_x,'next_y',next_y,'sample_time',model.sample_time,'semantics','discrete_next_state_no_extra_dt');result.cost=cost;result.constraints=margins;
available=license('test','Optimization_Toolbox');result.optimization_toolbox_license=logical(available);
if ~available,result.status='blocked_toolbox_license';return;end
[action,diagnostics,sequence]=mpc_control(x,reference,d,previous,parameters);
result.action=action;result.solve=diagnostics;result.solution_sequence=sequence;result.status='single_point_solved';
end
'''


def _matlab_model(design,prediction):
    empty={'kind':'not_instantiated','template':design.get('template','generic_description')}
    if prediction is None:return empty
    model=getattr(prediction,'runtime',getattr(prediction,'model',None))
    if model is None or not all(hasattr(model,key) for key in ('A','B','E','C','bias')) or getattr(model,'terms',[]):
        return {**empty,'kind':'unsupported_nonlinear_predictor','reason':'No affine certificate; Python native MPCPolicy remains available'}
    i=model.interface
    return {'kind':'affine','template':design.get('template','generic_description'),'nx':i.nx,'nu':i.nu,'ny':i.ny,'nd':i.nd,'sample_time':i.sample_time,
        **{key:getattr(model,key).detach().cpu().tolist() for key in ('A','B','E','C','bias')},
        **{key:[[v.lower,v.upper] for v in values] for key,values in [('state_bounds',i.states),('input_bounds',i.controls),('output_bounds',i.outputs)]}}


def export_controller_design(root,design,prediction=None):
    """Write a new self-contained basic-design bundle without any numerical run."""
    if not isinstance(design,dict) or type(design.get('schema_version')) is not int or design.get('schema_version')!=1:raise ContractError('Version-1 design definition required')
    if type(design.get('model_instantiated')) is not bool or not isinstance(design.get('parameters',{}),dict):raise ContractError('Explicit model_instantiated boolean and physical parameter object required')
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ContractError('Controller export requires a new empty directory')
    document=_portable(copy.deepcopy(design))
    if bool(document.get('model_instantiated'))!=(prediction is not None):raise ContractError('Instantiated design and predictor presence disagree')
    merged={**document.get('controller_config',{}),**document.get('weights',{})}
    required_controller={'horizon','q','r','s','solver_tolerance','max_iterations'}
    if not required_controller.issubset(merged):raise ContractError('Design requires explicit horizon/q/r/s/solver tolerance/max iterations')
    _write(root/'design.json',document)
    (root/'controller.py').write_text(_CONTROLLER,encoding='utf-8')
    numerical=_runtime_source();(root/'mpc_runtime.py').write_text(numerical,encoding='utf-8')
    assets=None
    if prediction is not None:
        from .predictors import export_predictor_assets
        assets=export_predictor_assets(prediction,root/'prediction')
        if not (root/'prediction/predictor.py').is_file():raise ContractError('Self-contained predictor.py/load_predictor assets required')
    _write(root/'parameter_schema.json',{'template':document.get('template'),'physical_parameters':{
        'mass':{'unit':'kg','exclusive_minimum':0},'damping':{'unit':'N*s/m','minimum':0},'stiffness':{'unit':'N/m','minimum':0},
        'sample_time':{'unit':'s','exclusive_minimum':0},'state_bounds':{'shape':[2,2],'units':['m','m/s']},
        'input_bounds':{'shape':[1,2],'units':['N']},'output_bounds':{'shape':[1,2],'units':['m']}},
        'required_when_instantiating_spring':list(SPRING_KEYS),'missing':document.get('required_parameters',[]),'physical_example_defaults':False})
    _write(root/'case_schema.json',{'x':'finite [nx] in state order','reference':'finite [ny] in output order','disturbance':'finite [nd]; omit only for nd=0','previous':'optional finite [nu]; default matches MPCPolicy amplitude-clipped zero'})
    (root/'example.py').write_text('''"""Use explicit caller files; no numerical example parameters are supplied."""
import argparse,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--parameters');p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('portable_controller',root/'controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
parameters=json.loads(Path(a.parameters).read_text(encoding='utf-8')) if a.parameters else None;case=json.loads(Path(a.input).read_text(encoding='utf-8'));u,diag=m.load_design(root).solve(case['x'],case['reference'],case.get('disturbance'),case.get('previous'),parameters)
Path(a.output).write_text(json.dumps({'u':u.tolist(),'diagnostics':diag,'closed_loop':'not_run','ga_trials':0},allow_nan=False)+'\\n',encoding='utf-8')
''',encoding='utf-8')
    _write(root/'matlab_model.json',_matlab_model(document,prediction))
    for name,content in [('prediction_model.m',_MATLAB_MODEL),('predict_step.m',_MATLAB_STEP),('mpc_control.m',_MATLAB_CONTROL),('mpc_cost.m',_MATLAB_COST),('selfcheck_controller.m',_MATLAB_SELFCHECK)]:
        (root/name).write_text(content,encoding='utf-8')
    (root/'README.md').write_text('''# Basic controller design

This package exports a design, not closed-loop validation. No GA, evaluation-plant rollout, deployment or effect-improvement claim was performed by export.

Python requires NumPy, SciPy and Torch. `controller.py` loads a bundle with hashes and returns the existing project MPCPolicy numerical implementation, copied verbatim into mpc_runtime.py. The same normalized q/r/s cost, state/output/input bounds, per-sample input increments, SLSQP gradients, feasibility checks and warm start apply. Source object hashes are recorded in bundle_manifest.json. No Harness, job/chat implementation or original workspace is loaded.

`load_design(bundle).instantiate(parameters)` returns a policy; `policy.act(x,reference,d,previous)` solves one point. `python -I controller.py --input case.json --output result.json [--parameters physical.json]` is the command form. Read design.json and case_schema.json for order, units, equations and constraints. This is not a simulator loop.

Missing physical parameters remain **not instantiated**. A spring template requires all seven named physical parameters in parameter_schema.json; only values explicitly provided in design.json or physical.json are used. No numerical example defaults are supplied. Generic descriptions remain documentation/contracts until an explicit supported predictor is supplied.

MATLAB prediction_model.m / predict_step.m support parameterized spring exact-ZOH models and explicitly affine frozen predictors. mpc_control.m condenses the same affine normalized quadratic cost and hard constraints into quadprog. This is an independent QP solver template, not the Python SLSQP implementation. Nonlinear predictors remain available through the Python package and are not silently linearized. MATLAB numeric/gradient/cross-solver agreement is **not tested by export**. Optimization Toolbox quadprog/optimoptions and an available license are required; availability and licensing are **not checked** here. `expm` constructs the parameterized spring ZOH map.

Official syntax references: [quadprog](https://www.mathworks.com/help/optim/ug/quadprog.html), [optimoptions](https://www.mathworks.com/help/optim/ug/optimoptions.html), [expm](https://www.mathworks.com/help/matlab/ref/expm.html). Hashes detect modifications; they are not a digital signature. Python predictor source is executable model code.
''',encoding='utf-8')
    files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    sources={obj.__name__:{'module':obj.__module__,'source_sha256':hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()} for obj in (MPCPolicy,SolverFailure,tensor)}
    manifest={'schema_version':1,'kind':'control-controller-design-bundle-v1','files':files,'model_instantiated':bool(document.get('model_instantiated')),
        'closed_loop':'not_run','ga_trials':0,'numerical_reuse':sources,'runtime_mpc_sha256':hashlib.sha256(numerical.encode()).hexdigest(),
        'predictor_assets':_portable(assets),'solver_resource_guard':{'max_variables':256,'max_dense_jacobian_values':5_000_000},
        'matlab':{'numerical_validation':'not_run','toolbox':'Optimization Toolbox','license':'not_checked','sources':MATLAB_SOURCES,'selfcheck':'selfcheck_controller(caseJsonPath,physicalParametersJsonPath_optional)'}}
    _write(root/'bundle_manifest.json',manifest)
    return {'directory':'bundle','manifest':'bundle/bundle_manifest.json','files':files,'model_instantiated':manifest['model_instantiated'],
        'closed_loop':'not_run','ga_trials':0,'numerical_reuse':sources,'matlab':manifest['matlab']}
