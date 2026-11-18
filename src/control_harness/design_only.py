"""MPC design packages, independently of tuning or original-plant validation.

Missing physical parameters remain symbolic. A numeric predictor can support
one diagnostic optimization, which is explicitly separate from closed-loop
verification and never invokes the existing genetic tuning workflow.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import torch

from .artifacts import Workspace, atomic_json
from .contracts import ContractError, MaterialInterface
from .control import MPCPolicy, SolverFailure
from .runtime import OperationContext


_SPRING_KEYS = ("mass", "damping", "stiffness", "sample_time", "state_bounds", "input_bounds", "output_bounds")
_METHODS = {"direct", "equations", "identify_affine", "linearize"}
_MAX_DIAGNOSTIC_VARIABLES = 256
_MAX_DIAGNOSTIC_JACOBIAN = 5_000_000


def _json(value):
    try: return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error: raise ContractError("Design declarations require finite JSON values") from error


def _number(value, label, *, positive=False):
    try: result = float(value) if type(value) in {int, float} else float("nan")
    except (ValueError, OverflowError): result = float("nan")
    if not math.isfinite(result) or result < 0 or positive and result == 0:
        raise ContractError(label + " must be a finite " + ("positive" if positive else "nonnegative") + " number")
    return result


def _integer(value, label, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ContractError(f"{label} must be an integer in [{low},{high}]")
    return value


def _has_bool(value):
    return isinstance(value, (bool, np.bool_)) or isinstance(value, (list, tuple)) and any(_has_bool(item) for item in value)


def _bounds(value, shape, label):
    try: raw = np.asarray(value); result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error: raise ContractError(label + " must contain explicit finite lower/upper pairs") from error
    if (_has_bool(value) or raw.dtype.kind not in "fiu" or result.shape != shape or not np.isfinite(result).all() or np.any(result[:,0] >= result[:,1])):
        raise ContractError(label + " must contain explicit finite ordered lower/upper pairs")
    return result.tolist()


def _partial_spring(parameters):
    if not isinstance(parameters, dict) or set(parameters) - set(_SPRING_KEYS):
        raise ContractError("Spring parameters use only mass,damping,stiffness,sample_time,state_bounds,input_bounds,output_bounds")
    result = {}
    for name, value in parameters.items():
        if value is None: continue  # Unknown physical parameters stay required.
        if name in {"mass", "sample_time"}: result[name] = _number(value, name, positive=True)
        elif name in {"damping", "stiffness"}: result[name] = _number(value, name)
        else: result[name] = _bounds(value, (2 if name == "state_bounds" else 1, 2), name)
    return result


def _source(inputs, config):
    if not isinstance(inputs, dict) or set(inputs) - {"model", "dataset", "description"}:
        raise ContractError("Design inputs contain one model artifact, sampled dataset or description; no evaluation_model is required")
    present = [key for key in inputs if inputs[key] is not None]
    if len(present) > 1 or any(not isinstance(inputs[key], str) or not inputs[key].strip() for key in present):
        raise ContractError("Choose one explicit nonempty model, dataset or description")
    template = config.get("template")
    if template is not None and (not isinstance(template,str) or template not in {"spring_damper", "generic"}):
        raise ContractError("Supported description templates are spring_damper or generic")
    if present and present[0] in {"model", "dataset"}:
        if template is not None or config.get("parameters") not in (None,{}): raise ContractError("Template parameters cannot silently override a supplied model/dataset")
        return present[0], None
    if isinstance(config.get("prediction"),dict) and config["prediction"].get("model_spec") is not None:
        if config.get("parameters") not in (None,{}): raise ContractError("Template parameters cannot override an explicit model_spec")
        return "equations",None
    if present:
        text = inputs["description"].lower()
        template = template or ("spring_damper" if "spring" in text or "弹簧" in text else "generic")
        return "description", template
    if template is not None: return "description", template
    raise ContractError("Provide a model, sampled dataset, description or explicit template/equation contract")


def _controller(config, nu=None):
    raw, weights = config.get("controller", {}), config.get("weights", {})
    if not isinstance(raw, dict) or set(raw) - {"horizon", "max_iterations", "solver_tolerance", "input_rate_bounds"}:
        raise ContractError("controller supports horizon,max_iterations,solver_tolerance,input_rate_bounds; q/r/s belong in weights")
    if not isinstance(weights, dict) or set(weights) - {"q", "r", "s"}: raise ContractError("weights supports only actual scalar q,r,s from MPCPolicy")
    result = {"horizon": _integer(raw.get("horizon",5),"horizon",1,100),
        "max_iterations": _integer(raw.get("max_iterations",80),"max_iterations",1,1000),
        "solver_tolerance": _number(raw.get("solver_tolerance",1e-8),"solver_tolerance",positive=True)}
    if result["solver_tolerance"] > 1: raise ContractError("solver_tolerance must be <= 1")
    if raw.get("input_rate_bounds") is not None:
        value = raw["input_rate_bounds"]
        if nu is None:
            if not isinstance(value, list) or not value: raise ContractError("input_rate_bounds requires explicit control width")
            nu = len(value)
        result["input_rate_bounds"] = _bounds(value,(nu,2),"input_rate_bounds")
    resolved_weights = {key: _number(weights.get(key,default),key,positive=key=="q") for key,default in (("q",1.),("r",.01),("s",.01))}
    return result, resolved_weights, {"controller": [key for key in ("horizon","max_iterations","solver_tolerance") if key not in raw],
        "weights": [key for key in ("q","r","s") if key not in weights], "scope": "declared design defaults, not plant measurements or tuned optima"}


def _template(template):
    if template == "spring_damper":
        return {"equations": {"continuous": ["d(position)/dt = velocity", "d(velocity)/dt = (force-damping*velocity-stiffness*position)/mass"],
            "output": "y = position", "continuous_matrices": {"A_c":[["0","1"],["-stiffness/mass","-damping/mass"]],
                "B_c":[["0"],["1/mass"]],"C":[[1,0]]},
            "discretization": "exact ZOH for constant force over the declared sample_time, after all numeric parameters are supplied"},
            "signal_order": {"states":[{"name":"position","unit":"m"},{"name":"velocity","unit":"m/s"}],
                "controls":[{"name":"force","unit":"N"}],"outputs":[{"name":"position","unit":"m"}],"disturbances":[]},
            "parameter_contract": {"mass":"kg, >0","damping":"N*s/m, >=0","stiffness":"N/m, >=0","sample_time":"s, >0",
                "state_bounds":"[[position_min_m,position_max_m],[velocity_min_mps,velocity_max_mps]]",
                "input_bounds":"[[force_min_N,force_max_N]]","output_bounds":"[[position_min_m,position_max_m]]"}}
    return {"equations":{"discrete":"x[k+1] = f_d(x[k],u[k],d[k];user_parameters)","output":"y[k] = g(x[k];user_parameters)"},
        "signal_order":None,"parameter_contract":{"prediction_contract":"Explicit supported model artifact or numeric model_spec; text is not executed",
            "interface":"Ordered states/controls/outputs/disturbances with units, sample time and finite applicability bounds"}}


def _constraints(interface, controller):
    return {"state":"x_min <= predicted_x <= x_max", "control":"u_min <= u <= u_max", "output":"y_min <= predicted_y <= y_max",
        "input_rate":"delta_u_min <= u[k]-u[k-1] <= delta_u_max" if "input_rate_bounds" in controller else "not_requested",
        "input_rate_units":"action increment per sample, not per second",
        "bounds":None if interface is None else {group:[[item.lower,item.upper] for item in getattr(interface,group)] for group in ("states","controls","outputs","disturbances")}}


def _cost(interface, weights):
    ranges = None
    if interface is not None:
        with np.errstate(over="ignore",invalid="ignore"):
            y = np.asarray([item.upper-item.lower for item in interface.outputs]); u = np.asarray([item.upper-item.lower for item in interface.controls])
        if not np.isfinite(y).all() or not np.isfinite(u).all() or np.any(y<=0) or np.any(u<=0):
            raise ContractError("MPC normalization requires finite positive output/control ranges; rescale extreme units/bounds")
        ranges = {"output":y.tolist(),"control":u.tolist()}
    return {"expression":"sum_h q*|| (y_pred-y_ref)/output_range ||^2 + r*||u/control_range||^2 + s*||delta_u/control_range||^2",
        "weights":weights,"range_normalization":ranges,"Q":"q*diag(output_range^-2)","R":"r*diag(control_range^-2)","S":"s*diag(control_range^-2)",
        "reference_semantics":"current reference held over the horizon","scope":"internal predicted-horizon objective; external overshoot/settling/stability have not been evaluated"}


def _vector(value,width,label):
    try: raw=np.asarray(value); result=np.asarray(value,dtype=np.float64)
    except (TypeError,ValueError,OverflowError) as error: raise ContractError(label+" must be a finite ordered vector") from error
    if _has_bool(value) or raw.dtype.kind not in "fiu" or result.shape!=(width,) or not np.isfinite(result).all():
        raise ContractError(label+" must be a finite ordered vector")
    return result


def _single_check(prediction, controller, weights, requested, context):
    if requested is False: return {"status":"not_run","reason":"not_requested","solves":0}
    if requested is not True and not isinstance(requested,dict): raise ContractError("solver_check must be true, false or an explicit diagnostic point")
    interface, model = prediction.interface,prediction.runtime
    point = {} if requested is True else requested
    if set(point)-{"x","reference","disturbance","previous"}: raise ContractError("solver_check point supports x,reference,disturbance,previous")
    variables=controller["horizon"]*interface.nu
    inequalities=controller["horizon"]*(2*interface.nx+2*interface.ny+(2*interface.nu if "input_rate_bounds" in controller else 0))
    if variables>_MAX_DIAGNOSTIC_VARIABLES or variables*inequalities>_MAX_DIAGNOSTIC_JACOBIAN:
        return {"status":"not_run","reason":"single_point_diagnostic_resource_budget","solves":0,
            "decision_variables":variables,"constraint_jacobian_scalars":variables*inequalities,
            "scope":"one diagnostic only; design generation is not a limit on the engineering task"}
    # Midpoints are generated diagnostic values, never alleged user conditions.
    midpoint=lambda group: np.asarray([item.lower/2+item.upper/2 for item in group])
    x=_vector(point.get("x",midpoint(interface.states)),interface.nx,"diagnostic x")
    d=_vector(point.get("disturbance",midpoint(interface.disturbances)),interface.nd,"diagnostic disturbance")
    previous=_vector(point.get("previous",np.clip(np.zeros(interface.nu),[v.lower for v in interface.controls],[v.upper for v in interface.controls])),interface.nu,"diagnostic previous")
    for value,group,label in ((x,interface.states,"x"),(d,interface.disturbances,"disturbance"),(previous,interface.controls,"previous")):
        if np.any(value<np.asarray([item.lower for item in group])) or np.any(value>np.asarray([item.upper for item in group])):
            raise ContractError("Diagnostic "+label+" is outside the declared applicability bounds")
    with torch.no_grad(): current=model.output(torch.from_numpy(x)).detach().cpu().numpy()
    reference=_vector(point.get("reference",current),interface.ny,"diagnostic reference")
    generated=[key for key in ("x","reference","disturbance","previous") if key not in point]
    details={"point":{"x":x.tolist(),"reference":reference.tolist(),"disturbance":d.tolist(),"previous":previous.tolist()},
        "point_source":"diagnostic_generated" if generated else "explicit_diagnostic_point","generated_fields":generated,
        "purpose":"solver/constraint self-consistency only; not an original-plant closed-loop condition"}
    if np.any(current<np.asarray([v.lower for v in interface.outputs])) or np.any(current>np.asarray([v.upper for v in interface.outputs])):
        return {**details,"status":"not_run","reason":"diagnostic_state_output_outside_bounds; provide another explicit point","solves":0}
    policy=MPCPolicy(model,{**controller,**weights},context)
    try:
        action,diagnostics=policy.act(x,reference,d,previous)
        if not np.isfinite(action).all(): raise FloatingPointError("MPC diagnostic action is non-finite")
        return {**details,"status":"passed","solves":1,"action":action.tolist(),"diagnostics":diagnostics,
            "solver":"scipy_slsqp_with_torch_derivatives","optimality":"local numerical solver diagnostic, no global optimum certificate"}
    except (SolverFailure,FloatingPointError) as error:
        return {**details,"status":"failed","solves":1,"error":str(error),"solver":"scipy_slsqp_with_torch_derivatives"}


def synthesize_mpc(workspace: Workspace, inputs: dict, config: dict,
                   context: OperationContext | None = None) -> str:
    """Produce a design package with zero GA trials and zero plant closed loops."""
    from .design_export import export_controller_design
    from .predictors import resolve_predictor, spring_damper_spec

    context=context or OperationContext()
    if not isinstance(config,dict) or set(config)-{"prediction","controller","weights","template","parameters","solver_check","plan_refs"}:
        raise ContractError("Unsupported design-only configuration; evaluation/tuning belongs to its separate operation")
    config=_json(config); inputs=_json(inputs)
    source,template=_source(inputs,config)
    prediction_config=config.get("prediction",{})
    if not isinstance(prediction_config,dict): raise ContractError("prediction must be an explicit method/config object")
    default={"model":"direct","dataset":"identify_affine","description":"equations","equations":"equations"}[source]
    prediction_config={"method":default,**prediction_config}
    method=prediction_config["method"]
    if not isinstance(method,str) or method not in _METHODS: raise ContractError("prediction.method must be direct,equations,identify_affine or linearize")
    allowed_prediction={"direct":{"method"},"equations":{"method","model_spec"},"identify_affine":{"method","C","fit","acceptance"},
        "linearize":{"method","operating_point","validity_bounds"}}
    if set(prediction_config)-allowed_prediction[method]: raise ContractError("Unsupported fields for the selected prediction method")
    if ((source=="model" and method not in {"direct","linearize"}) or (source=="dataset" and method!="identify_affine") or
        (source in {"description","equations"} and method!="equations")):
        raise ContractError("Prediction method conflicts with the explicit input source")
    raw_parameters=config.get("parameters")
    if raw_parameters is None: raw_parameters={}
    if not isinstance(raw_parameters,dict): raise ContractError("parameters must be an explicit parameter object")
    parameters=_partial_spring(raw_parameters) if template=="spring_damper" else _json(raw_parameters)
    required=[key for key in _SPRING_KEYS if key not in parameters] if template=="spring_damper" else (["prediction_contract","interface"] if template=="generic" else [])
    controller,weights,defaults=_controller(config,1 if template=="spring_damper" else None)
    prediction=None
    with workspace.lock("synthesize_mpc"):
        context.check()
        if not required:
            if template=="spring_damper":
                if any(value is not None for key,value in prediction_config.items() if key!="method"): raise ContractError("Spring template model_spec is constructed only from its supplied parameters")
                prediction_config={"method":"equations","model_spec":spring_damper_spec(parameters)}
            prediction=resolve_predictor(workspace,{key:value for key,value in inputs.items() if key in {"model","dataset"}},prediction_config,context)
            if not prediction.instantiated or prediction.interface is None or prediction.runtime is None:
                raise ContractError("The selected predictor did not provide an instantiated numeric model")
            controller,weights,defaults=_controller(config,prediction.interface.nu)
        interface=None if prediction is None else prediction.interface
        info=_template(template) if template is not None else {"equations":{"source":"explicit predictor descriptor"},
            "signal_order":None,"parameter_contract":{}}
        if interface is not None:
            info["signal_order"]={group:[{"name":item.name,"unit":item.unit} for item in getattr(interface,group)] for group in ("states","controls","outputs","disturbances")}
        requested=config.get("solver_check",True)
        if requested is not True and requested is not False and not isinstance(requested,dict): raise ContractError("solver_check must be boolean or an explicit point object")
        if isinstance(requested,dict) and set(requested)-{"x","reference","disturbance","previous"}: raise ContractError("Unknown diagnostic point field")
        cost=_cost(interface,weights)
        check={"status":"not_run","reason":"physical_parameters_or_predictor_contract_required","solves":0} if prediction is None else _single_check(prediction,controller,weights,requested,context)
        failed=check["status"]=="failed"
        design={"schema_version":1,"mode":"design_only","status":"parameterized_draft" if prediction is None else ("solver_check_failed" if failed else "instantiated_unverified"),
            "template":template,"description":inputs.get("description"),"parameters":parameters,"required_parameters":required,
            "model_instantiated":prediction is not None,"physical_interface":None if interface is None else interface.to_dict(),
            "prediction":None if prediction is None else prediction.to_dict(),"prediction_config":prediction_config,
            "controller_config":controller,"weights":weights,"design_defaults":defaults,
            **info,"constraints":_constraints(interface,controller),"cost":cost,"solver_check":check,
            "closed_loop":{"status":"not_run","evaluation_model":None,"ga_trials":0,"plant_rollout_steps":0},
            "closed_loop_validation":"not_run","controller_parameters_tuned":False,
            "validation_scope":"design-package integrity and optional single-point diagnostic, not closed-loop controller acceptance"}
        parents=[]
        for value in [inputs.get("model"),inputs.get("dataset"),None if prediction is None else prediction.artifact_id,
                      *([] if prediction is None else prediction.parents)]:
            if value is not None and value not in parents: parents.append(value)
        with workspace.stage("controller_design",MaterialInterface("bundle","MPC design package"),parents,config,compatible_parents=False) as stage:
            atomic_json(stage.root/"design.json",design)
            manifest=export_controller_design(stage.root/"bundle",design,prediction=prediction)
            report={**design,"algorithm":"mpc_design_only","confirmation_sources":context.case_confirmation_sources,
                "export":{"directory":"bundle","manifest":"bundle/bundle_manifest.json", "details":{key:value for key,value in manifest.items() if key not in {"bundle_root","manifest_path","directory"}}}}
            atomic_json(stage.root/"report.json",report)
            context.check()
            return stage.publish({"status":"rejected" if failed else "verified","checks":["explicit_design_contract","no_fabricated_physical_parameters","standalone_design_export"],
                "design_status":design["status"],"model_instantiated":design["model_instantiated"],"solver_check":check["status"],
                "closed_loop":"not_run","status_semantics":"package integrity only, not controller acceptance"},
                source={"algorithm":"mpc_design_only","prediction_method":method,"model_instantiated":design["model_instantiated"],
                    "controller_parameters_tuned":False,"ga_trials":0,"closed_loop":"not_run"},activate=False)
