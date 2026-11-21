"""Explicit acquisition of design predictors without evaluation plants or GA."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, ModelInterface
from .modeling import DeclarativeDynamics, load_model, tensor
from .runtime import OperationContext


@dataclass
class PredictionResult:
    runtime: object | None
    interface: ModelInterface | None
    spec: dict | None
    artifact_id: str | None
    route: str
    report: dict
    instantiated: bool
    source_root: Path | None = None
    parents: tuple[str, ...] = ()

    @property
    def model(self): return self.runtime

    @property
    def descriptor(self): return self.spec

    def to_dict(self):
        return {"route": self.route, "artifact_id": self.artifact_id, "instantiated": self.instantiated,
            "interface": None if self.interface is None else self.interface.to_dict(), "spec": self.spec,
            "report": self.report, "parents": list(self.parents)}


def _number(value, name, *, minimum=None, positive=False):
    if type(value) not in {int, float} or not math.isfinite(value) or (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ContractError(name + " must be an explicit finite number in the supported range")
    return float(value)


def _array(value, shape, name):
    if math.prod(shape) > 5_000_000:
        raise ContractError(name + " exceeds the single-predictor numeric budget")
    def boolean(item):
        return isinstance(item, (bool, np.bool_)) or isinstance(item, (list, tuple)) and any(boolean(child) for child in item)
    raw = np.asarray(value)
    if boolean(value) or raw.dtype.kind not in "fiu" or raw.shape != shape:
        if shape == (0, 2) and isinstance(value, list) and value == []:
            return np.empty(shape, dtype=np.float64)
        raise ContractError(name + " requires an explicit numeric array of shape " + str(shape))
    result = raw.astype(np.float64)
    if not np.isfinite(result).all(): raise ContractError(name + " must be finite")
    return result


def spring_damper_spec(parameters):
    """SI spring/mass/damper with all physical/timing/bounds inputs required."""
    required = {"mass", "damping", "stiffness", "sample_time", "state_bounds", "input_bounds", "output_bounds"}
    if not isinstance(parameters, dict) or set(parameters) != required:
        raise ContractError("Spring template requires exactly: " + ", ".join(sorted(required)))
    mass = _number(parameters["mass"], "mass", positive=True)
    damping = _number(parameters["damping"], "damping", minimum=0)
    stiffness = _number(parameters["stiffness"], "stiffness", minimum=0)
    dt = _number(parameters["sample_time"], "sample_time", positive=True)
    states = _array(parameters["state_bounds"], (2, 2), "state_bounds")
    controls = _array(parameters["input_bounds"], (1, 2), "input_bounds")
    outputs = _array(parameters["output_bounds"], (1, 2), "output_bounds")
    if any(np.any(bounds[:, 0] >= bounds[:, 1]) for bounds in (states, controls, outputs)):
        raise ContractError("Template bounds must be ordered lower/upper pairs")
    variable = lambda name, unit, bounds: {"name": name, "unit": unit, "lower": float(bounds[0]), "upper": float(bounds[1])}
    return {"representation": "continuous_polynomial", "interface": {"schema_version": 1, "name": "Explicit spring mass damper",
        "time_semantics": "continuous_state_derivative", "sample_time": dt,
        "states": [variable("position", "m", states[0]), variable("velocity", "m/s", states[1])],
        "controls": [variable("force", "N", controls[0])], "outputs": [variable("position", "m", outputs[0])], "disturbances": []},
        "A": [[0., 1.], [-stiffness / mass, -damping / mass]], "B": [[0.], [1. / mass]], "E": [[], []],
        "C": [[1., 0.]], "bias": [0., 0.], "terms": [], "time_unit": "s", "derivative_units": ["m/s", "(m/s)/s"],
        "integration": {"method": "zoh", "substeps": 1, "input_hold": "zero_order_hold"}}


def _declarative(spec):
    if not isinstance(spec, dict) or spec.get("representation") not in {"discrete_polynomial", "continuous_polynomial"}:
        raise ContractError("Explicit supported polynomial equations are required")
    raw = copy.deepcopy(spec.get("interface", {})); raw["time_semantics"] = "discrete_state_transition"
    interface = ModelInterface.parse(raw)
    if interface.nx * (interface.nx + interface.nu + interface.nd + interface.ny) > 5_000_000:
        raise ContractError("Equation predictor exceeds the single-predictor matrix budget")
    model = DeclarativeDynamics(spec)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def _metrics(predicted, actual, atol, rtol):
    return {"rows": len(actual), "passed": bool(np.all(np.abs(predicted - actual) <= atol + rtol * np.abs(actual))),
        "max_abs_error": np.max(np.abs(predicted - actual), axis=0).tolist()}


def compare_predictor_trajectories(model, arrays, masks, split, metric, context):
    """在相同的独立工况计算单步与开环误差。metric 按量纲、检验类型计算门限。"""
    C = model.C.detach().numpy()
    one_step, rollouts = {}, {}
    for part in ("validation", "test"):
        context.check()
        mask = masks[part]
        prediction = model.step(arrays["x"][mask], arrays["u"][mask], arrays["d"][mask])
        one_step[part] = {"state": metric(prediction, arrays["next_x"][mask], "state", "one_step"),
                          "output": metric(prediction @ C.T, arrays["next_y"][mask], "output", "one_step")}
        records = []
        for trajectory in split[part]:
            rows = np.flatnonzero(arrays["trajectory_id"] == trajectory)
            rows = rows[np.argsort(arrays["time"][rows], kind="stable")]
            state = arrays["x"][rows[0]].copy()
            states = []
            for row in rows:
                # 仅首拍用记录的真实状态，以后的每一拍都用预测器自己的上一拍状态。
                context.check()
                state = model.step(state, arrays["u"][row], arrays["d"][row])
                states.append(state)
            predicted = np.asarray(states)
            records.append({"trajectory_id": trajectory, "steps": len(rows),
                "state": metric(predicted, arrays["next_x"][rows], "state", "rollout"),
                "output": metric(predicted @ C.T, arrays["next_y"][rows], "output", "rollout"),
                "teacher_forcing_after_initial_state": False})
        rollouts[part] = records
    passed = all(value["passed"] for part in one_step.values() for value in part.values()) and all(
        record[key]["passed"] for records in rollouts.values() for record in records for key in ("state", "output"))
    return one_step, rollouts, passed


def _identify(workspace, dataset_id, config, context):
    from .identification import fit_affine_transitions
    from .predictive import _read_sampled
    arrays, masks, split, interface, source = _read_sampled(workspace, dataset_id)
    C = _array(config.get("C"), (interface.ny, interface.nx), "C")
    if not np.allclose(arrays["y"], arrays["x"] @ C.T, rtol=1e-9, atol=1e-11) or not np.allclose(arrays["next_y"], arrays["next_x"] @ C.T, rtol=1e-9, atol=1e-11):
        raise ContractError("Explicit C disagrees with recorded y/next_y; C is not inferred")
    acceptance = config.get("acceptance")
    if not isinstance(acceptance, dict) or set(acceptance) != {"atol", "rtol"}:
        raise ContractError("Identification requires explicit acceptance:{atol,rtol}")
    atol = _number(acceptance["atol"], "acceptance.atol", minimum=0)
    rtol = _number(acceptance["rtol"], "acceptance.rtol", minimum=0)
    if atol == 0 and rtol == 0: raise ContractError("At least one identification error tolerance must be positive")
    fit = config.get("fit", {})
    if not isinstance(fit, dict) or set(fit) - {"fit_bias", "rank_relative_tolerance", "max_condition_number"}:
        raise ContractError("Unsupported affine fit configuration")
    coefficients, report = fit_affine_transitions(interface, arrays, masks, fit, context)
    spec = {"representation": "discrete_polynomial", "interface": interface.to_dict(), **coefficients, "C": C.tolist(), "terms": []}
    model = _declarative(spec)
    one_step, rollouts, passed = compare_predictor_trajectories(
        model, arrays, masks, split, lambda predicted, actual, *_: _metrics(predicted, actual, atol, rtol), context)
    report.update({"route": "identify_affine", "acceptance": {"atol": atol, "rtol": rtol, "passed": passed},
        "one_step": one_step, "rollout": rollouts, "sampled_dataset": dataset_id,
        "source_provenance": source.get("source", {}), "C_validation": "consistent_with_recorded_data_not_uniqueness_proof",
        "scope": "fully_observed_affine_fit_on_recorded_training_and_heldout_trajectory_region_only",
        "global_linearity_proved": False, "controller_parameters_modified": False, "evaluation_plant_required": False,
        "prediction_time_semantics": "source.time+sample_time; model already returns next_x in state units",
        "observed_training_ranges": {key: {"lower": arrays[key][masks["train"]].min(0).tolist(), "upper": arrays[key][masks["train"]].max(0).tolist()} for key in ("x", "u", "d")}})
    with workspace.stage("identified_model", interface, [dataset_id], config) as stage:
        atomic_json(stage.root / "model.json", spec); atomic_json(stage.root / "report.json", report)
        artifact_id = stage.publish({"status": "verified" if passed else "rejected", "full_training_rank": True,
            "heldout_one_step_and_rollout": passed, "scope": report["scope"]},
            preprocessing=report["preprocessing"], source={"kind": "identified_affine_state_transition", "dataset": dataset_id,
                "data_source": source.get("source", {}), "global_linearity_proved": False}, activate=False)
    if not passed: raise ContractError("Affine predictor failed declared held-out gates; inspect " + artifact_id)
    root, _ = workspace.resolve(artifact_id)
    return PredictionResult(model, interface, spec, artifact_id, "identify_affine", report, True, root, (artifact_id,))


def _linearize(model, config, context):
    interface = model.interface
    operating, bounds = config.get("operating_point"), config.get("validity_bounds")
    if not isinstance(operating, dict) or set(operating) != {"x", "u", "d"} or not isinstance(bounds, dict) or set(bounds) != {"x", "u", "d"}:
        raise ContractError("Linearization requires explicit operating_point{x,u,d} and validity_bounds{x,u,d}")
    point, ranges = [], []
    local_interface = copy.deepcopy(interface.to_dict())
    for key, group in (("x", "states"), ("u", "controls"), ("d", "disturbances")):
        variables = getattr(interface, group); width = len(variables)
        value = _array(operating[key], (width,), "operating_point." + key)
        interval = _array(bounds[key], (width, 2), "validity_bounds." + key)
        if np.any(interval[:, 0] >= interval[:, 1]) or np.any(value < interval[:, 0]) or np.any(value > interval[:, 1]):
            raise ContractError("Operating point must lie in ordered explicit validity bounds")
        if np.any(interval[:, 0] < [item.lower for item in variables]) or np.any(interval[:, 1] > [item.upper for item in variables]):
            raise ContractError("Local validity bounds must lie inside the declared physical interface")
        for item, pair in zip(local_interface[group], interval): item["lower"], item["upper"] = map(float, pair)
        point.append(tensor(value)); ranges.append(interval)
    context.check()
    with torch.enable_grad():
        A, B, E = torch.autograd.functional.jacobian(lambda x, u, d: model(x, u, d), tuple(point))
        C = torch.autograd.functional.jacobian(model.output, point[0])
        f0 = model(*point).detach(); h0 = model.output(point[0]).detach()
    if not all(torch.isfinite(value).all() for value in (A, B, E, C, f0, h0)):
        raise ContractError("Operating-point Jacobians are nonfinite")
    output_bias = h0 - C @ point[0]
    if not torch.allclose(output_bias, torch.zeros_like(output_bias), rtol=0, atol=1e-10):
        raise ContractError("Nonzero local output bias is unsupported by C@x; use the direct nonlinear predictor")
    bias = f0 - A @ point[0] - B @ point[1] - E @ point[2]
    spec = {"representation": "discrete_polynomial", "interface": local_interface, "A": A.detach().numpy().tolist(),
        "B": B.detach().numpy().tolist(), "E": E.detach().numpy().tolist(), "C": C.detach().numpy().tolist(),
        "bias": bias.numpy().tolist(), "terms": []}
    linear = _declarative(spec)
    points = [[value.detach().numpy().copy() for value in point]]
    for group, interval in enumerate(ranges):
        for index, pair in enumerate(interval):
            for endpoint in pair:
                probe = [value.detach().numpy().copy() for value in point]; probe[group][index] = endpoint; points.append(probe)
    comparisons = []
    for probe in points:
        context.check()
        actual = model.step(*probe); predicted = linear.step(*probe)
        with torch.no_grad(): actual_y = model.output(tensor(probe[0])).numpy(); predicted_y = linear.output(tensor(probe[0])).numpy()
        comparisons.append({"x": probe[0].tolist(), "u": probe[1].tolist(), "d": probe[2].tolist(),
            "state_abs_error": np.abs(actual - predicted).tolist(), "output_abs_error": np.abs(actual_y - predicted_y).tolist()})
    return linear, spec, {"route": "linearize", "scope": "local_jacobian_not_global_linear_model",
        "operating_point": {key: value.detach().numpy().tolist() for key, value in zip(("x", "u", "d"), point)},
        "f0": f0.numpy().tolist(), "h0": h0.numpy().tolist(), "bias": bias.numpy().tolist(), "output_bias": output_bias.numpy().tolist(),
        "sample_time": interface.sample_time, "validity_bounds": bounds, "original_physical_interface": interface.to_dict(),
        "validation_method": "operating_point_plus_each_axis_endpoint; not a full-box error bound", "local_comparisons": comparisons,
        "max_state_abs_error": np.max([row["state_abs_error"] for row in comparisons], axis=0).tolist(),
        "max_output_abs_error": np.max([row["output_abs_error"] for row in comparisons], axis=0).tolist(),
        "continuous_derivative_used": False, "evaluation_plant_required": False}


def resolve_predictor(workspace, inputs, prediction_config, context=None):
    context = context or OperationContext(); context.check()
    if not isinstance(inputs, dict) or not isinstance(prediction_config, dict): raise ContractError("Explicit predictor inputs/config required")
    method = prediction_config.get("method", "direct")
    allowed = {"direct": {"method"}, "equations": {"method", "model_spec"},
        "identify_affine": {"method", "C", "fit", "acceptance"}, "linearize": {"method", "operating_point", "validity_bounds"}}
    if not isinstance(method, str) or method not in allowed or set(prediction_config) - allowed[method]:
        raise ContractError("Unsupported prediction method/configuration")
    if method == "equations":
        model = _declarative(prediction_config.get("model_spec"))
        return PredictionResult(model, model.interface, copy.deepcopy(model.spec), None, method,
            {"route": method, "source": "explicit_user_equations", "transition_adapter": model.transition_adapter,
             "scope": "declared_equations_not_independently_certified", "evaluation_plant_required": False}, True)
    if method == "identify_affine":
        if not isinstance(inputs.get("dataset"), str): raise ContractError("identify_affine requires inputs.dataset")
        return _identify(workspace, inputs["dataset"], prediction_config, context)
    if not isinstance(inputs.get("model"), str): raise ContractError(method + " requires inputs.model")
    root, metadata = workspace.resolve(inputs["model"]); model = load_model(workspace, inputs["model"])
    source = metadata.get("source", {})
    # 对转换得到的对象，设计包需带走原始采样范围；可执行性并不能扩展物理有效域。
    applicability = source.get("applicability") if source.get("kind") == "plant_predictor_conversion" else None
    if method == "direct":
        spec = copy.deepcopy(getattr(model, "spec", {"representation": "nominal_plus_learned_residual"}))
        return PredictionResult(model, model.interface, spec, inputs["model"], method,
            {"route": method, "source_kind": metadata["kind"], "source": source,
             "nonlinear_MPC_supported_without_AB": True, "evaluation_plant_required": False,
             "scope": applicability["scope"] if applicability else
                      "registered_differentiable_discrete_predictor; source fidelity not independently certified"}, True, root, (inputs["model"],))
    linear, spec, report = _linearize(model, prediction_config, context)
    if applicability:
        # 线性化误差是相对于原预测器的局部误差；原对象与预测器间的误差仍受转换报告限制。
        report["source_conversion"] = source
    linearization_source = {"kind": "operating_point_linearization", "source_model": inputs["model"],
                            "operating_point": report["operating_point"], "validity_bounds": report["validity_bounds"]}
    if applicability:
        linearization_source["source_conversion"] = source
    with workspace.stage("identified_model", linear.interface, [inputs["model"]], prediction_config, compatible_parents=False) as stage:
        atomic_json(stage.root / "model.json", spec); atomic_json(stage.root / "report.json", report)
        artifact_id = stage.publish({"status": "verified", "scope": report["scope"], "local_error_bound_proved": False},
            source=linearization_source, activate=False)
    root, _ = workspace.resolve(artifact_id)
    return PredictionResult(linear, linear.interface, spec, artifact_id, method, report, True, root, (artifact_id,))


_ASSET_VERIFY = '''
def _verify_assets(root):
    manifest=json.loads((root/'predictor_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='control-predictor-assets-v1':raise ValueError('Unsupported predictor manifest')
    for name,digest in manifest['files'].items():
        path=(root/name).resolve(strict=True)
        if path.parent!=root or path.stat().st_size>64*1024*1024 or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Predictor asset hash/path mismatch')
    return manifest
'''


def _portable_evidence(value):
    """Keep semantic IDs/hashes/numbers without local workspace storage details."""
    storage = {"workspace", "source_root", "codeRoot", "pythonPath", "checkpoint", "resume_checkpoint"}
    if isinstance(value, dict):
        return {key: _portable_evidence(item) for key, item in value.items()
                if key not in storage and not key.endswith(("_path", "_root", "_directory"))
                and not (isinstance(item, str) and Path(item).is_absolute())}
    if isinstance(value, (list, tuple)):
        return [_portable_evidence(item) for item in value if not (isinstance(item, str) and Path(item).is_absolute())]
    return value


def _execution_spec(spec):
    return {key: copy.deepcopy(value) for key, value in spec.items() if key in {
        "representation", "interface", "A", "B", "E", "C", "bias", "terms", "time_unit", "derivative_units", "integration"}}


def export_predictor_assets(result, destination):
    """Write standalone predictor.py/load_predictor assets for supported runtimes."""
    if not isinstance(result, PredictionResult) or not result.instantiated:
        raise ContractError("An instantiated PredictionResult is required for numerical export")
    representation = result.spec.get("representation")
    corrected = representation == "nominal_plus_learned_residual"
    if corrected:
        from .modeling import CorrectedDynamics, ResidualNetwork
        if (not isinstance(result.runtime, CorrectedDynamics) or not isinstance(result.runtime.nominal, DeclarativeDynamics)
                or not isinstance(result.runtime.residual, ResidualNetwork)):
            raise ContractError("Portable residual predictor currently requires its actual supported declarative nominal and ResidualNetwork")
    if representation not in {"discrete_polynomial", "continuous_polynomial", "learned_discrete_mlp", "nominal_plus_learned_residual"}:
        raise ContractError("Portable S07 export supports declarative polynomial/ZOH or learned MLP; this source requires an explicit portable adapter")
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()): raise ContractError("Predictor assets require a new empty directory")
    if representation == "learned_discrete_mlp":
        if result.source_root is None: raise ContractError("Learned predictor has no frozen bundle root")
        from .learned_dynamics import load_learned_model, _verify_members
        load_learned_model(result.source_root)
        source = result.source_root / "bundle"
        physical_manifest = read_json(source / "dynamics_manifest.json")
        expected = {"learned.json", "model.pt", "network_config.json", "scaler.json", "data_interface.json", "features.json", "dynamics.py", "simulate.py", "README.md"}
        if physical_manifest.get("kind") != "control-learned-dynamics-bundle-v1" or physical_manifest.get("schema_version") != 1:
            raise ContractError("Unsupported frozen learned predictor bundle")
        _verify_members(source.resolve(), physical_manifest.get("files"), expected)
        load_learned_model(source)
        for name in expected:
            shutil.copyfile(source / name, destination / name)
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != physical_manifest["files"][name]:
                raise ContractError("Frozen learned predictor changed while copying assets")
        atomic_json(destination / "dynamics_manifest.json", physical_manifest)
        learned = read_json(destination / "learned.json")
        learned["provenance"] = _portable_evidence(learned.get("provenance", {}))
        atomic_json(destination / "learned.json", learned)
        physical_manifest = read_json(destination / "dynamics_manifest.json")
        physical_manifest["files"]["learned.json"] = hashlib.sha256((destination / "learned.json").read_bytes()).hexdigest()
        atomic_json(destination / "dynamics_manifest.json", physical_manifest)
        loader = ("import hashlib,importlib.util,json\nfrom pathlib import Path\n" + _ASSET_VERIFY + '''
def load_predictor(directory):
    root=Path(directory).resolve(strict=True);_verify_assets(root)
    spec=importlib.util.spec_from_file_location('_portable_learned_predictor',root/'dynamics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.load_dynamics(root)
''')
    else:
        from .contracts import Variable
        from .continuous import affine_discrete_matrices, continuous_contract, integrate_polynomial
        from .modeling import matrix
        header = ('import copy,hashlib,json,math\nfrom pathlib import Path\nfrom dataclasses import dataclass,asdict\n'
            'import numpy as np\nimport torch\nfrom torch import nn\nContractError=ValueError\nSCHEMA_VERSION=1\nDTYPE=torch.float64\n'
            'def canonical_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(",",":")).encode()).hexdigest()\n')
        classes = [Variable, ModelInterface, tensor, matrix, continuous_contract, affine_discrete_matrices, integrate_polynomial, DeclarativeDynamics]
        if corrected:
            from .reused.perception import PerceptionNet
            classes.extend([PerceptionNet, ResidualNetwork, CorrectedDynamics])
            atomic_json(destination / "residual-network.json", {key: value for key, value in result.runtime.residual.config.items()
                if key in {"hidden_dims", "residual_semantics"}})
            atomic_json(destination / "residual-scalers.json", {key: getattr(result.runtime.residual, key).detach().cpu().numpy().tolist()
                for key in ("input_mean", "input_scale", "residual_scale")})
            torch.save({key: value.detach().cpu().clone() for key, value in result.runtime.residual.state_dict().items()}, destination / "residual.pt")
        loader = header + '\n\n'.join(inspect.getsource(item) for item in classes) + _ASSET_VERIFY + '''
def load_predictor(directory):
    root=Path(directory).resolve(strict=True);_verify_assets(root)
    model=DeclarativeDynamics(json.loads((root/'model.json').read_text(encoding='utf-8')))
'''
        if corrected:
            loader += '''
    residual=ResidualNetwork(model.interface,json.loads((root/'residual-network.json').read_text(encoding='utf-8')),json.loads((root/'residual-scalers.json').read_text(encoding='utf-8')))
    state=torch.load(root/'residual.pt',map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or any(not isinstance(value,torch.Tensor) or value.dtype!=torch.float64 or not torch.isfinite(value).all() for value in state.values()):raise ValueError('Residual weights must be finite float64 tensors')
    residual.load_state_dict(state,strict=True);residual.eval();model=CorrectedDynamics(model,residual)
'''
        loader += '''
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model
'''
        atomic_json(destination / "model.json", _execution_spec(result.runtime.nominal.spec if corrected else result.spec))
    (destination / "predictor.py").write_text(loader, encoding="utf-8")
    atomic_json(destination / "prediction_report.json", _portable_evidence(result.report))
    manifest = {"schema_version": 1, "kind": "control-predictor-assets-v1", "representation": representation,
        "interface": result.interface.to_dict(), "source_workspace_required": False,
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir() if path.is_file()}}
    atomic_json(destination / "predictor_manifest.json", manifest)
    return manifest
