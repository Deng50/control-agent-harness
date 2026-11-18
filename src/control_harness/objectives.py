"""Fixed external evaluation recipes, independent of MPC's internal q/r/s.

Per-channel tracking/constant-reference segmentation and signed overshoot are
adapted from mpc_agent_v5/app/performance.py::canonical_tracking_metrics.
Source SHA256: f99b0da7b8e44882b1812756d88b8cfdef4b0a152a739fd0615e3d3d7d3d8c89.
Only pure numerical logic is reused: no Pydantic/FastAPI/application imports.
The first segment uses an explicit *pre-step* initial output, correcting the
old post-first-output baseline assumption. Settling/crossing definitions below
are explicit finite-record diagnostics, not physical stability certificates.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .contracts import ContractError, ModelInterface


ATTRIBUTION = {"source":"mpc_agent_v5/app/performance.py", "function":"canonical_tracking_metrics",
    "sha256":"f99b0da7b8e44882b1812756d88b8cfdef4b0a152a739fd0615e3d3d7d3d8c89",
    "adaptation":"pure per-output errors, reference segmentation and signed step overshoot; explicit pre-step initial output"}
_KEYS=("rmse","energy","tv","overshoot")
_RECIPES={"tracking":{"rmse":10.,"energy":.1,"tv":.1,"overshoot":0.},
    "energy":{"rmse":.1,"energy":10.,"tv":.1,"overshoot":0.},
    "smoothness":{"rmse":.1,"energy":.1,"tv":10.,"overshoot":0.}}
_GATES={"max_rmse":"rmse","max_energy":"energy","max_tv":"tv","max_overshoot":"overshoot",
    "max_settling_time":"settling_time","max_oscillation_crossings":"oscillation_crossings"}
_OUTPUT_GATES={"max_rmse":"rmse","max_overshoot":"overshoot_ratio",
    "max_settling_time":"settling_time","max_oscillation_crossings":"oscillation_crossings"}
_METRICS={"settling_abs_band":0.,"settling_rel_band":.02,"settling_hold_steps":3,"step_atol":1e-12,
    "oscillation_abs_deadband":0.,"oscillation_rel_deadband":.02}
_MAX_SEGMENTS=10_000


def _number(value,label,positive=False):
    try: result=float(value) if type(value) in {int,float} else float("nan")
    except (OverflowError,ValueError):result=float("nan")
    if not math.isfinite(result) or result<0 or positive and result==0:
        raise ContractError(label+" must be a finite "+("strictly positive" if positive else "nonnegative")+" number")
    return result


def _gate(value,label):
    if value is None:return None
    if label.endswith("oscillation_crossings"):
        if type(value) is not int or value<0:raise ContractError(label+" must be a nonnegative integer or null")
        return value
    return _number(value,label)


def normalize_evaluation(config):
    """Return an idempotent JSON-only contract fixed before any candidate search."""
    if not isinstance(config,dict) or set(config)-{"schema_version","recipe","weights","scales","energy_model","hard_gates","metrics"}:
        raise ContractError("Evaluation requires a strict recipe/weights/scales/hard_gates/metrics object")
    if type(config.get("schema_version",1)) is not int or config.get("schema_version",1)!=1:
        raise ContractError("Evaluation schema_version must be 1")
    recipe=config.get("recipe")
    if not isinstance(recipe,str) or recipe not in _RECIPES:raise ContractError("recipe must be tracking, energy or smoothness")
    weights=config.get("weights",_RECIPES[recipe]);scales=config.get("scales")
    if not isinstance(weights,dict) or set(weights)!=set(_KEYS):raise ContractError("weights must explicitly contain rmse,energy,tv,overshoot when supplied")
    if not isinstance(scales,dict) or set(scales)!=set(_KEYS):raise ContractError("All four positive scales must be declared before search; candidate-fitted scales are unsupported")
    weights={key:_number(weights[key],"weights."+key) for key in _KEYS}
    if not any(weights.values()):raise ContractError("At least one evaluation weight must be positive")
    scales={key:_number(scales[key],"scales."+key,True) for key in _KEYS}
    if config.get("energy_model","action_proxy")!="action_proxy":raise ContractError("Only explicit action_proxy energy is supported; no physical power model or Joule claim is inferred")
    raw=config.get("hard_gates",{})
    if not isinstance(raw,dict) or set(raw)-{"constraint_tolerance","per_output",*_GATES}:
        raise ContractError("Unknown hard gate; use supported explicit metric ceilings")
    gates={"constraint_tolerance":_number(raw.get("constraint_tolerance",1e-6),"constraint_tolerance")}
    gates.update({name:_gate(raw.get(name),name) for name in _GATES})
    per_output=raw.get("per_output",{})
    if not isinstance(per_output,dict):raise ContractError("hard_gates.per_output must map exact output names to ceilings")
    gates["per_output"]={}
    for name,values in sorted(per_output.items(),key=lambda pair:str(pair[0])):
        if not isinstance(name,str) or not name.strip() or not isinstance(values,dict) or set(values)-set(_OUTPUT_GATES):
            raise ContractError("Each named output gate must use known per-output metric ceilings")
        gates["per_output"][name]={key:_gate(values.get(key),key) for key in _OUTPUT_GATES}
    raw_metrics=config.get("metrics",{})
    if not isinstance(raw_metrics,dict) or set(raw_metrics)-set(_METRICS):raise ContractError("Unknown metric definition setting")
    definitions={}
    for name,default in _METRICS.items():
        value=raw_metrics.get(name,default)
        if name=="settling_hold_steps":
            if type(value) is not int or not 1<=value<=1_000_000:raise ContractError("settling_hold_steps must be an integer in 1..1000000")
            definitions[name]=value
        else:definitions[name]=_number(value,"metrics."+name)
    return {"schema_version":1,"recipe":recipe,"weights":weights,"scales":scales,"energy_model":"action_proxy",
        "hard_gates":gates,"metrics":definitions}


def _boolean(value):
    return isinstance(value,(bool,np.bool_)) or isinstance(value,(list,tuple)) and any(_boolean(child) for child in value)


def _numeric(value,shape,label):
    try:raw=np.asarray(value);result=np.asarray(value,dtype=np.float64)
    except (ValueError,TypeError,OverflowError) as error:raise ContractError(label+" has an invalid numeric shape/type") from error
    if _boolean(value) or raw.dtype.kind not in "fiu" or result.shape!=shape:raise ContractError(label+" has an invalid numeric shape/type")
    return result


def _all_finite(value):
    if isinstance(value,dict):return all(_all_finite(item) for item in value.values())
    if isinstance(value,(list,tuple)):return all(_all_finite(item) for item in value)
    if isinstance(value,(float,np.floating)):return math.isfinite(value)
    return True


def _segments(values,references,initial,dt,settings):
    boundaries=[0]+[k for k in range(1,len(references)) if not math.isclose(float(references[k]),float(references[k-1]),rel_tol=1e-12,abs_tol=settings["step_atol"])]+[len(values)]
    records=[];central=[]
    for segment,(start,end) in enumerate(zip(boundaries,boundaries[1:])):
        target=float(references[start]);origin=initial if segment==0 else float(references[start-1])
        error=values[start:end]-target;length=end-start
        quarter=length//4;central.extend((values[start:end]-references[start:end])[quarter:length-quarter].tolist())
        known=origin is not None
        applicable=known and not math.isclose(target,float(origin),rel_tol=1e-12,abs_tol=settings["step_atol"])
        amplitude=abs(target-float(origin)) if known else None
        direction=(1. if target>origin else -1.) if applicable else (0. if known else None)
        overshoot=max(0.,float(np.max(direction*error)))/amplitude if applicable else None
        band=max(settings["settling_abs_band"],settings["settling_rel_band"]*amplitude) if known else None
        settled=None
        if applicable:
            inside=np.abs(error)<=band
            for k in range(length):
                if length-k>=settings["settling_hold_steps"] and bool(np.all(inside[k:])):
                    settled=(k+1)*dt;break
        settling_reason="available" if settled is not None else ("missing_initial_output" if not known else "no_step" if not applicable else "not_settled_in_observed_segment")
        if length<3:
            crossings=None;crossing_reason="insufficient_constant_target_samples";deadband=None
        elif not known and settings["oscillation_rel_deadband"]>0:
            crossings=None;crossing_reason="missing_amplitude_for_relative_deadband";deadband=None
        else:
            deadband=max(settings["oscillation_abs_deadband"],settings["oscillation_rel_deadband"]*(amplitude or 0.))
            signs=np.sign(error[np.abs(error)>deadband])
            crossings=int(np.count_nonzero(signs[1:]!=signs[:-1]));crossing_reason="available"
        records.append({"start_step":start,"end_step_exclusive":end,"target":target,"origin":origin,
            "step_applicable":bool(applicable),"step_amplitude":amplitude,"step_direction":direction,
            "overshoot_ratio":overshoot,"overshoot_reason":"available" if applicable else "missing_initial_output" if not known else "no_step",
            "settling_time":settled,"settling_band":band,"settling_reason":settling_reason,
            "oscillation_crossings":crossings,"oscillation_deadband":deadband,"oscillation_reason":crossing_reason})
    applicable=[record for record in records if record["step_applicable"]]
    unknown=any(record["origin"] is None for record in records)
    overshoot=None if unknown or not applicable else max(record["overshoot_ratio"] for record in applicable)
    settling=None if unknown or not applicable or any(record["settling_time"] is None for record in applicable) else max(record["settling_time"] for record in applicable)
    crossings=None if any(record["oscillation_crossings"] is None for record in records) else sum(record["oscillation_crossings"] for record in records)
    return {"segments":records,"central_segment_rmse":math.sqrt(float(np.mean(np.square(central)))),
        "overshoot_ratio":overshoot,"settling_time":settling,"oscillation_crossings":crossings,
        "step_metrics_reason":"missing_initial_output" if unknown else "no_step" if not applicable else "observed_step_segments"}


def _definitions(interface):
    return {"rmse":"sqrt(mean((output-reference)^2)) over all post-action samples and output coordinates",
        "energy":"sum_samples sum_controls u^2 * sample_time; action-coordinate effort proxy, not measured power or Joules",
        "tv":"sum absolute control increments, including the first jump from supplied actual initial_control; no invented terminal jump",
        "overshoot":"max observed signed step excursion beyond target / abs(target-origin); no-step is null, not zero",
        "step_origin":"explicit initial output for first segment; previous target for later constant-reference segments",
        "central_segment_rmse":"central 50% of each reference segment by integer-quarter trimming; v5 diagnostic, not a settling proof",
        "settling_time":"first post-action time (k-start+1)*dt whose remaining segment suffix stays within max(abs_band,rel_band*step_amplitude), with at least hold_steps samples; no-step/unsettled is null",
        "oscillation_crossings":"error sign changes outside the fixed deadband within each constant-reference segment, summed per output; not an oscillation-frequency or stability certificate",
        "stability":"not_assessed; finite observation and small error do not establish physical or closed-loop stability",
        "aggregate_units":"RMSE/energy/TV aggregates are numeric coordinate indices, particularly with mixed physical units; fixed scales are explicit user choices, not physical unit conversions",
        "output_units":[item.unit for item in interface.outputs],"control_units":[item.unit for item in interface.controls],
        "mixed_output_units":len({item.unit for item in interface.outputs})>1,
        "mixed_control_units":len({item.unit for item in interface.controls})>1,
        "constraint_scope":"supplied initial values when available plus all recorded post-action states/outputs and held controls; every channel/sample breach is counted"}


def trajectory_metrics(run,interface,evaluation):
    """Compute raw metrics from a recorded run; never trust its legacy score."""
    evaluation=normalize_evaluation(evaluation)
    if not isinstance(interface,(ModelInterface,dict)):raise ContractError("Explicit physical ModelInterface required")
    interface=interface if isinstance(interface,ModelInterface) else ModelInterface.parse(interface)
    unknown=set(evaluation["hard_gates"]["per_output"])-{item.name for item in interface.outputs}
    if unknown:raise ContractError("Per-output hard gates reference unknown outputs: "+", ".join(sorted(unknown)))
    if not isinstance(run,dict) or not isinstance(run.get("trajectory"),list) or not run["trajectory"]:
        raise ContractError("A nonempty recorded closed_loop trajectory is required")
    rows=run["trajectory"];n=len(rows)
    if n*(interface.nx+interface.nu+2*interface.ny)>5_000_000:raise ContractError("Evaluation trajectory exceeds its single-run numeric budget")
    if any(not isinstance(row,dict) for row in rows):raise ContractError("Trajectory rows must be explicit objects")
    for index,row in enumerate(rows):
        if "step" in row and (type(row["step"]) is not int or row["step"]!=index):raise ContractError("Trajectory steps must be ordered consecutive post-action records")
    arrays={key:_numeric([row.get(key) for row in rows],(n,width),key) for key,width in
        (("next_x",interface.nx),("u",interface.nu),("output",interface.ny),("reference",interface.ny))}
    scenario=run.get("scenario",{})
    if not isinstance(scenario,dict):raise ContractError("scenario must be an object")
    initial_y=run.get("initial_output",scenario.get("initial_output"))
    initial_u=run.get("initial_control",scenario.get("initial_control"))
    initial_x=scenario.get("initial_state",rows[0].get("x"))
    initial_y=None if initial_y is None else _numeric(initial_y,(interface.ny,),"initial_output")
    initial_u=None if initial_u is None else _numeric(initial_u,(interface.nu,),"initial_control")
    initial_x=None if initial_x is None else _numeric(initial_x,(interface.nx,),"initial_state")
    definitions=_definitions(interface)
    base={"schema_version":1,"samples":n,"sample_time":interface.sample_time,"duration":n*interface.sample_time,
        "metric_definitions":definitions,"attribution":copy.deepcopy(ATTRIBUTION),"energy_kind":"action_proxy",
        "initial_output_available":initial_y is not None,"initial_control_available":initial_u is not None}
    finite=all(np.isfinite(value).all() for value in [*arrays.values(),*[value for value in (initial_y,initial_u,initial_x) if value is not None]])
    if not finite:
        return {**base,**{key:None for key in (*_KEYS,"settling_time","oscillation_crossings")},"finite":False,
            "hard_constraints_complete":False,"hard_constraint_violations":1,"per_output":[],"per_control":[],"metric_error":"nonfinite_recorded_values"}
    changes=interface.ny
    for index in range(interface.ny):
        reference_column=arrays["reference"][:,index]
        # Match math.isclose in _segments exactly: NumPy's atol+rtol rule
        # differs and could undercount many near-tolerance reference changes.
        with np.errstate(over="ignore",invalid="ignore"):
            differences=np.abs(reference_column[1:]-reference_column[:-1])
            limits=np.maximum(evaluation["metrics"]["step_atol"],1e-12*np.maximum(np.abs(reference_column[1:]),np.abs(reference_column[:-1])))
        changes+=int(np.count_nonzero(differences>limits))
    if changes>_MAX_SEGMENTS:raise ContractError("Evaluation exceeds the 10,000 reference-segment single-run detail budget; partition the evaluation cases")
    y,u,x,reference=(arrays[key] for key in ("output","u","next_x","reference"))
    tolerance=evaluation["hard_gates"]["constraint_tolerance"]
    constraints={};total=0
    for key,values,group,initial in (("state",x,interface.states,initial_x),("input",u,interface.controls,initial_u),("output",y,interface.outputs,initial_y)):
        checked=values if initial is None else np.vstack((initial,values))
        low=np.asarray([item.lower for item in group]);high=np.asarray([item.upper for item in group])
        with np.errstate(over="ignore",invalid="ignore"):violation=np.maximum(low-checked,0)+np.maximum(checked-high,0)
        count=int(np.count_nonzero(violation>tolerance));total+=count
        constraints[key]={"available":True,"requested":True,"count":count,"max_violation":float(violation.max()),"initial_included":initial is not None}
    delta=None if initial_u is None else np.diff(np.vstack((initial_u,u)),axis=0)
    checks=run.get("constraint_checks",{})
    if not isinstance(checks,dict):raise ContractError("constraint_checks must be an object")
    rate=run.get("input_rate_bounds")
    requested_rate=rate is not None or checks.get("input_rate") is True
    complete=True
    if rate is not None:
        rate=_numeric(rate,(interface.nu,2),"input_rate_bounds")
        if not np.isfinite(rate).all() or np.any(rate[:,0]>=rate[:,1]):raise ContractError("input_rate_bounds require finite ordered increment bounds")
    if requested_rate and (rate is None or delta is None):
        complete=False;constraints["input_rate"]={"requested":True,"available":False,"count":None,"max_violation":None}
    elif requested_rate:
        violation=np.maximum(rate[:,0]-delta,0)+np.maximum(delta-rate[:,1],0)
        count=int(np.count_nonzero(violation>tolerance));total+=count
        constraints["input_rate"]={"requested":True,"available":True,"count":count,"max_violation":float(violation.max())}
    else:constraints["input_rate"]={"requested":False,"available":False,"count":None,"max_violation":None}
    error=y-reference
    with np.errstate(over="ignore",invalid="ignore"):
        rmse_per=np.sqrt(np.mean(error**2,axis=0));rmse=float(np.sqrt(np.mean(error**2)))
        effort=(u**2).sum(axis=0)*interface.sample_time
        tv=None if delta is None else np.abs(delta).sum(axis=0)
    outputs=[]
    for index,variable in enumerate(interface.outputs):
        step=_segments(y[:,index],reference[:,index],None if initial_y is None else float(initial_y[index]),interface.sample_time,evaluation["metrics"])
        outputs.append({"name":variable.name,"unit":variable.unit,"rmse":float(rmse_per[index]),
            "max_abs_error":float(np.max(np.abs(error[:,index]))),"mean_abs_error":float(np.mean(np.abs(error[:,index]))),**step})
    step_known=all(item["step_metrics_reason"]!="missing_initial_output" for item in outputs)
    applicable=[item for item in outputs if item["step_metrics_reason"]=="observed_step_segments"]
    overshoot=max(item["overshoot_ratio"] for item in applicable) if applicable and step_known else None
    settling=max(item["settling_time"] for item in applicable) if applicable and step_known and all(item["settling_time"] is not None for item in applicable) else None
    oscillation=max(item["oscillation_crossings"] for item in outputs) if all(item["oscillation_crossings"] is not None for item in outputs) else None
    with np.errstate(over="ignore",invalid="ignore"):
        energy_total=float(effort.sum());tv_total=None if tv is None else float(tv.sum())
    numeric=[rmse,energy_total,*rmse_per.tolist(),*effort.tolist(),*[value for value in (overshoot,settling,oscillation,tv_total) if value is not None]]
    if tv is not None:numeric.extend(tv.tolist())
    finite=all(math.isfinite(float(value)) for value in numeric) and all(math.isfinite(value["max_violation"]) for value in constraints.values() if value["max_violation"] is not None)
    finite=finite and _all_finite(outputs)
    if not finite:
        return {**base,**{key:None for key in (*_KEYS,"settling_time","oscillation_crossings")},"finite":False,
            "hard_constraints_complete":False,"hard_constraint_violations":max(1,total),"per_output":[],"per_control":[],"metric_error":"numeric_metric_overflow"}
    return {**base,"finite":True,"rmse":rmse,"rmse_per_output":rmse_per.tolist(),"energy":energy_total,
        "tv":tv_total,"overshoot":overshoot,"settling_time":settling,"oscillation_crossings":oscillation,
        "hard_constraint_violations":total,"hard_constraints_complete":complete,"constraints":constraints,
        "per_output":outputs,"per_control":[{"name":item.name,"unit":item.unit,"energy_proxy":float(effort[index]),
            "energy_proxy_unit":item.unit+"^2*s","tv":None if tv is None else float(tv[index])} for index,item in enumerate(interface.controls)],
        "max_state_violation":constraints["state"]["max_violation"],"max_input_violation":constraints["input"]["max_violation"],
        "max_output_violation":constraints["output"]["max_violation"],"max_rate_violation":constraints["input_rate"]["max_violation"]}


def _actual(value):
    if type(value) not in {int,float}:return None
    try:result=float(value)
    except (OverflowError,ValueError):return None
    return result if math.isfinite(result) and result>=0 else None


def score_metrics(metrics,evaluation):
    """Fixed weighted sum with fail-closed hard/availability gates; no fitting."""
    evaluation=normalize_evaluation(evaluation)
    if not isinstance(metrics,dict):raise ContractError("Scoring requires a metric object")
    checks=[];terms={}
    count=metrics.get("hard_constraint_violations")
    valid_count=type(count) is int and count>=0
    checks.append({"name":"hard_constraint_violations","actual":count if valid_count else None,"limit":0,"available":valid_count,"passed":valid_count and count==0})
    complete=metrics.get("hard_constraints_complete",True)
    checks.append({"name":"hard_constraints_complete","actual":complete,"available":type(complete) is bool,"passed":complete is True})
    finite=metrics.get("finite",True)
    checks.append({"name":"finite","actual":finite,"available":type(finite) is bool,"passed":finite is True})
    for key in _KEYS:
        raw=metrics.get(key);value=_actual(raw);weight=evaluation["weights"][key];scale=evaluation["scales"][key]
        available=value is not None
        checks.append({"name":"metric."+key,"available":available,"passed":available or raw is None and weight==0})
        normalized=None if value is None else value/scale
        normalized=normalized if normalized is not None and math.isfinite(normalized) else None
        weighted=0. if weight==0 else None if normalized is None else weight*normalized
        weighted=weighted if weighted is not None and math.isfinite(weighted) else None
        checks.append({"name":"normalization."+key,"available":normalized is not None,"passed":weight==0 or weighted is not None})
        terms[key]={"raw":value,"weight":weight,"scale":scale,"normalized":normalized,"weighted":weighted,"included":weight>0}
    for gate,key in _GATES.items():
        ceiling=evaluation["hard_gates"][gate]
        if ceiling is None:continue
        actual=_actual(metrics.get(key));checks.append({"name":gate,"actual":actual,"limit":ceiling,
            "available":actual is not None,"passed":actual is not None and actual<=ceiling})
    channels=metrics.get("per_output",[])
    if not isinstance(channels,list):raise ContractError("per_output metrics must be a list")
    by_name={item.get("name"):item for item in channels if isinstance(item,dict)}
    for name,gates in evaluation["hard_gates"]["per_output"].items():
        for gate,key in _OUTPUT_GATES.items():
            ceiling=gates[gate]
            if ceiling is None:continue
            actual=_actual(by_name.get(name,{}).get(key));checks.append({"name":name+"."+gate,"actual":actual,"limit":ceiling,
                "available":actual is not None,"passed":actual is not None and actual<=ceiling})
    passed=all(item["passed"] for item in checks)
    try:score=math.fsum(item["weighted"] for item in terms.values()) if passed else None
    except OverflowError:
        checks.append({"name":"finite_score","available":True,"passed":False});passed=False;score=None
    if score is not None and not math.isfinite(score):
        checks.append({"name":"finite_score","available":True,"passed":False});passed=False;score=None
    return {"score":score,"hard_constraints_passed":passed,"terms":terms,"checks":checks,
        "scoring":"sum fixed weight*raw_metric/fixed_scale; no candidate-dependent normalization",
        "energy_kind":"action_proxy"}
