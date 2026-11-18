"""Frozen differentiable next-state MLPs with explicit physical bindings.

The network predicts next_x in state units. It is never a derivative and no
extra sample-time multiplier is applied. Source plants are provenance only.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .contracts import ContractError, DataInterface, ModelInterface
from .supervised_export import build_supervised_network, validate_network_config


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _json(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ContractError("Learned-model JSON exceeds the 64 MiB member budget")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_members(root, hashes, expected):
    if not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ContractError("Learned-model member inventory differs")
    for name, digest in hashes.items():
        path = (root / name).resolve(strict=True)
        if (path.parent != root or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ContractError("Invalid learned-model file or SHA256")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            raise ContractError("Learned-model file hash mismatch: " + name)


def _physical(raw):
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ContractError("Explicit physical interface version 1 required")
    if raw.get("time_semantics") != "discrete_state_transition":
        raise ContractError("Learned dynamics requires discrete next-state semantics")
    if type(raw.get("sample_time")) not in {int, float} or not math.isfinite(raw["sample_time"]) or raw["sample_time"] <= 0:
        raise ContractError("Physical sample_time must be finite and positive")
    if not isinstance(raw.get("name"), str) or not raw["name"].strip():
        raise ContractError("Physical interface name must be explicit")
    if set(raw) - {"schema_version", "name", "time_semantics", "sample_time", "states", "controls", "outputs", "disturbances"}:
        raise ContractError("Physical interface has unsupported/inferred dimension fields")
    value = {"schema_version": 1, "name": raw["name"], "time_semantics": "discrete_state_transition",
             "sample_time": float(raw["sample_time"])}
    for group in ("states", "controls", "outputs", "disturbances"):
        variables = raw.get(group, [])
        if not isinstance(variables, (list, tuple)) or (group != "disturbances" and not variables):
            raise ContractError("Physical variables must be explicit ordered lists")
        if len(variables) > 65536:
            raise ContractError("Physical variable count exceeds the interface budget")
        result = []
        for item in variables:
            if not isinstance(item, dict) or set(item) != {"name", "unit", "lower", "upper"}:
                raise ContractError("Physical variables require name/unit/finite lower/upper")
            if any(not isinstance(item[key], str) or not item[key].strip() for key in ("name", "unit")):
                raise ContractError("Physical variable names and units must be nonempty text")
            if any(type(item[key]) not in {int, float} or not math.isfinite(item[key]) for key in ("lower", "upper")) or item["lower"] >= item["upper"]:
                raise ContractError("Physical bounds must be finite ordered numbers")
            result.append({"name": item["name"], "unit": item["unit"], "lower": float(item["lower"]), "upper": float(item["upper"])})
        if len({item["name"] for item in result}) != len(result):
            raise ContractError("Duplicate physical variable names within a group")
        value[group] = result
    return value


def _numeric(value, shape, label, *, positive=False):
    if math.prod(shape) > 5_000_000:
        raise ContractError(label + " exceeds the five-million-value numeric budget")
    def boolean(item):
        return isinstance(item, (bool, np.bool_)) or (isinstance(item, (list, tuple)) and any(boolean(child) for child in item))
    raw = np.asarray(value)
    if boolean(value) or raw.dtype.kind not in "iuf" or raw.shape != tuple(shape):
        raise ContractError(label + " has an invalid numeric shape/type")
    raw = raw.astype(np.float64)
    if not np.isfinite(raw).all() or (positive and not (raw > 0).all()):
        raise ContractError(label + " must be finite" + (" and positive" if positive else ""))
    return raw


def _binding(descriptor, settings, data_interface, features, scaler):
    if (not isinstance(descriptor, dict) or type(descriptor.get("schema_version")) is not int
            or descriptor["schema_version"] != 1 or descriptor.get("representation") != "learned_discrete_mlp"
            or descriptor.get("target_semantics") != "next_state_in_state_units"
            or descriptor.get("input_semantics") != "current_full_states_controls_disturbances"
            or descriptor.get("transition_adapter") is not None
            or any(key in descriptor for key in ("integration", "derivative_units", "time_unit", "A", "B", "E"))):
        raise ContractError("Unsupported learned discrete next-state declaration")
    physical = _physical(descriptor.get("interface"))
    if descriptor.get("interface_hash") != _hash({key: value for key, value in physical.items() if key != "name"}):
        raise ContractError("Physical interface order/unit/dt/bounds hash differs")
    nx, nu, ny, nd = [len(physical[group]) for group in ("states", "controls", "outputs", "disturbances")]
    if settings["input_dim"] != nx + nu + nd or settings["output_dim"] != nx:
        raise ContractError("Network widths must equal [all x,u,d] -> next_x")
    if (not isinstance(data_interface, dict) or data_interface.get("interface_kind") != "tabular_supervised"
            or type(data_interface.get("schema_version")) is not int or data_interface["schema_version"] != 1
            or data_interface.get("task") != "dynamic_prediction"
            or type(data_interface.get("sample_time")) not in {int, float}
            or data_interface.get("sample_time") != physical["sample_time"]
            or type(data_interface.get("input_dim")) is not int or type(data_interface.get("output_dim")) is not int
            or data_interface.get("input_dim") != nx + nu + nd or data_interface.get("output_dim") != nx):
        raise ContractError("Supervised data interface timing/dimensions differ from physical next-state binding")
    expected_signals, expected_features, expected_order, physical_order = [], [], [], []
    for group, role in (("states", "state"), ("controls", "control"), ("disturbances", "disturbance")):
        for index, item in enumerate(physical[group]):
            signal = group + "::" + item["name"]
            expected_signals.append((signal, role, item["unit"]))
            expected_features.append({"signal": signal, "window": 1, "lag": 0})
            expected_order.append({"signal": signal, "lag": 0, "unit": item["unit"], "availability_delay_steps": 0})
            physical_order.append({"group": group, "index": index, "name": item["name"], "unit": item["unit"], "signal": signal})
    signals = data_interface.get("signals")
    if not isinstance(signals, list) or len(signals) != len(expected_signals):
        raise ContractError("Every physical state/control/disturbance must have exactly one explicit signal")
    for signal, expected in zip(signals, expected_signals):
        if (not isinstance(signal, dict) or tuple(signal.get(key) for key in ("name", "role", "unit")) != expected
                or type(signal.get("delay_steps")) is not int or signal["delay_steps"] != 0
                or signal.get("observable") is not True):
            raise ContractError("Signal order/role/unit/availability differs from full physical [x,u,d]")
    targets = [{"signal": "states::" + item["name"], "offset": 1} for item in physical["states"]]
    target_order = [{**target, "unit": item["unit"]} for target, item in zip(targets, physical["states"])]
    mapping = {item["name"]: "states::" + item["name"] for item in physical["states"]}
    if (_hash(data_interface.get("features")) != _hash(expected_features) or _hash(data_interface.get("targets")) != _hash(targets)
            or data_interface.get("state_mapping") != mapping):
        raise ContractError("Only explicit full current-state features and one-step next-state targets are supported; no history/hidden state/preview")
    if _hash(descriptor.get("feature_order")) != _hash(physical_order) or _hash(descriptor.get("target_order")) != _hash(target_order):
        raise ContractError("Learned descriptor feature/target order differs")
    if (not isinstance(features, dict) or _hash(features.get("feature_order")) != _hash(expected_order)
            or _hash(features.get("target_order")) != _hash(target_order)
            or features.get("feature_labels") != [item["signal"] + "@lag=0" for item in expected_order]
            or features.get("target_labels") != [item["signal"] + "@offset=1" for item in targets]
            or features.get("input_units") != [item["unit"] for item in expected_order]
            or features.get("output_units") != [item["unit"] for item in physical["states"]]):
        raise ContractError("Saved supervised feature labels/order/units differ")
    C = _numeric(descriptor.get("C"), (ny, nx), "C")
    if not isinstance(scaler, dict) or scaler.get("method") not in {"standard", "minmax", "none"} or scaler.get("fit_split") != "train":
        raise ContractError("A finite training-only scaler is required")
    transforms = {}
    for key, width in (("X", nx + nu + nd), ("y", nx)):
        group = scaler.get(key)
        if not isinstance(group, dict):
            raise ContractError("Missing learned-model scaler group")
        location = _numeric(group.get("location"), (width,), key + " location")
        scale = _numeric(group.get("scale"), (width,), key + " scale", positive=True)
        if scaler["method"] == "none" and (np.any(location != 0) or np.any(scale != 1)):
            raise ContractError("Identity scaler is not identity")
        transforms[key] = (location, scale)
    return physical, C, transforms


class LearnedDynamics(nn.Module):
    """Stateless differentiable next-state map; frozen parameters retain input grads."""
    def __init__(self, descriptor, settings, network, data_interface, features, scaler):
        super().__init__()
        physical, C, transforms = _binding(descriptor, settings, data_interface, features, scaler)
        self.spec = json.loads(json.dumps(descriptor, allow_nan=False))
        self.physical_interface = physical
        self.interface = SimpleNamespace(**{**physical, **{key: tuple(SimpleNamespace(**item) for item in physical[key]) for key in ("states", "controls", "outputs", "disturbances")},
            "nx": len(physical["states"]), "nu": len(physical["controls"]), "ny": len(physical["outputs"]), "nd": len(physical["disturbances"])})
        self.transition_adapter = None
        self.representation = "learned_discrete_mlp"
        self.network_config = dict(settings)
        self.network = network
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("C", torch.as_tensor(C, dtype=torch.float64))
        for key, (location, scale) in transforms.items():
            self.register_buffer(key + "_location", torch.as_tensor(location, dtype=torch.float64))
            self.register_buffer(key + "_scale", torch.as_tensor(scale, dtype=torch.float64))
        self.eval()

    def _tensor(self, value, width, label):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.device.type != "cpu"
                or value.ndim not in (1, 2) or value.shape[-1] != width or not torch.isfinite(value).all()
                or (value.ndim == 2 and not value.shape[0])):
            raise ContractError(label + " must be finite CPU/float64 [width] or [batch,width]")
        return value

    def forward(self, x, u, d=None):
        x = self._tensor(x, self.interface.nx, "x")
        u = self._tensor(u, self.interface.nu, "u")
        if d is None:
            if self.interface.nd:
                raise ContractError("Declared disturbances must be provided explicitly")
            d = x.new_empty((*x.shape[:-1], 0))
        d = self._tensor(d, self.interface.nd, "d")
        if x.shape[:-1] != u.shape[:-1] or x.shape[:-1] != d.shape[:-1]:
            raise ContractError("x/u/d batch shapes must match exactly; no implicit broadcast")
        rows = 1 if x.ndim == 1 else x.shape[0]
        if rows * sum([self.network_config["input_dim"], *self.network_config["hidden_dims"], self.network_config["output_dim"]]) > 5_000_000:
            raise ContractError("Learned dynamics exceeds the 5,000,000 activation-scalar batch budget")
        features = torch.cat((x, u, d), dim=-1)
        scaled = (features - self.X_location) / self.X_scale
        if not torch.isfinite(scaled).all():
            raise FloatingPointError("Learned input scaling is nonfinite")
        result = self.network(scaled) * self.y_scale + self.y_location
        if result.shape != (*x.shape[:-1], self.interface.nx) or not torch.isfinite(result).all():
            raise FloatingPointError("Learned next-state prediction is nonfinite or dimensionally invalid")
        return result

    def output(self, x):
        x = self._tensor(x, self.interface.nx, "x")
        rows = 1 if x.ndim == 1 else x.shape[0]
        if rows * (self.interface.nx + self.interface.ny) > 5_000_000:
            raise ContractError("Learned output exceeds the five-million-value batch budget")
        result = x @ self.C.T
        if not torch.isfinite(result).all():
            raise FloatingPointError("Learned output is nonfinite")
        return result

    def derivative(self, *args):
        raise ContractError("This is an already discrete next-state model, not a derivative")

    def step(self, x, u, d=None):
        with torch.no_grad():
            x = torch.as_tensor(_numeric(x, np.asarray(x).shape, "x"), dtype=torch.float64)
            u = torch.as_tensor(_numeric(u, np.asarray(u).shape, "u"), dtype=torch.float64)
            d = None if d is None else torch.as_tensor(_numeric(d, np.asarray(d).shape, "d"), dtype=torch.float64)
            return self(x, u, d).detach().cpu().numpy().copy()


def _components(root):
    root = Path(root).resolve(strict=True)
    descriptor = _json(root / "learned.json")
    expected = {"model.pt", "network_config.json", "scaler.json", "data_interface.json", "features.json"}
    _verify_members(root, descriptor.get("files"), expected)
    settings = validate_network_config(_json(root / "network_config.json"))
    interface, features, scaler = [_json(root / name) for name in ("data_interface.json", "features.json", "scaler.json")]
    _binding(descriptor, settings, interface, features, scaler)
    with torch.random.fork_rng(devices=[]):
        network = build_supervised_network(settings)
    weights = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or not torch.isfinite(value).all() for value in weights.values()):
        raise ContractError("Learned weights must be a pure finite float64 state_dict")
    try:
        network.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise ContractError("Learned network/state_dict keys or dimensions differ") from error
    return descriptor, settings, network, interface, features, scaler


def load_learned_model(root):
    """Load entirely from this model's files; never open a source plant or session."""
    model = LearnedDynamics(*_components(root))
    model.interface = ModelInterface.parse(model.physical_interface)
    return model


_SIMULATOR = r'''
class DynamicsSimulator:
    def __init__(self,model): self.model=model; self._state=None
    def reset(self,x0):
        value=_numeric(x0,(self.model.interface.nx,),'x0')
        self._state=value.copy(); return self.state
    @property
    def state(self):
        if self._state is None: raise ValueError('Call reset(x0) first')
        return self._state.copy()
    def step(self,u,d=None):
        value=self.model.step(self.state,u,d); self._state=value.copy(); return self.state
    def output(self):
        with torch.no_grad(): return self.model.output(torch.as_tensor(self.state,dtype=torch.float64)).numpy().copy()

def load_dynamics(directory):
    root=Path(directory).resolve(strict=True)
    manifest=_json(root/'dynamics_manifest.json')
    expected={'learned.json','model.pt','network_config.json','scaler.json','data_interface.json','features.json','dynamics.py','simulate.py','README.md'}
    if manifest.get('kind')!='control-learned-dynamics-bundle-v1' or manifest.get('schema_version')!=1: raise ValueError('Unsupported dynamics bundle')
    _verify_members(root,manifest.get('files'),expected)
    return LearnedDynamics(*_components(root))

def load_simulator(directory): return DynamicsSimulator(load_dynamics(directory))
'''


_SIMULATE_CLI = r'''"""Simulate the learned discrete map; input is physical x0/u/d, not scaled data."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,help='JSON {x0,controls:[[nu]],disturbances:[[nd]]}; d omitted only when nd=0')
    parser.add_argument('--output',required=True,help='JSON states/outputs at T+1 grid points in physical units')
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    sys.dont_write_bytecode=True
    spec=importlib.util.spec_from_file_location('_portable_dynamics',root/'dynamics.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    simulator=module.load_simulator(root);model=simulator.model
    source=Path(args.input)
    if source.stat().st_size>64*1024*1024: raise ValueError('Input exceeds 64 MiB')
    data=json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or set(data)-{'x0','controls','disturbances'}:raise ValueError('Unsupported simulation input fields')
    controls=np.asarray(data.get('controls'))
    if controls.ndim!=2 or not len(controls) or len(controls)>100000:raise ValueError('controls must contain 1..100000 steps')
    controls=module._numeric(data['controls'],(len(controls),model.interface.nu),'controls')
    raw_d=data.get('disturbances')
    if model.interface.nd and raw_d is None:raise ValueError('Declared disturbances required')
    disturbances=np.zeros((len(controls),0)) if raw_d is None else module._numeric(raw_d,(len(controls),model.interface.nd),'disturbances')
    if (len(controls)+1)*(model.interface.nx+model.interface.ny)>5000000:raise ValueError('Simulation output exceeds the five-million-value budget')
    states=[simulator.reset(data.get('x0')).tolist()];outputs=[simulator.output().tolist()]
    for u,d in zip(controls,disturbances):states.append(simulator.step(u,d).tolist());outputs.append(simulator.output().tolist())
    physical=model.physical_interface
    result={'schema_version':1,'semantics':'discrete_next_state_no_extra_dt','sample_time':model.interface.sample_time,
        'time':(np.arange(len(states))*model.interface.sample_time).tolist(),'states':states,'outputs':outputs,
        'state_order':[v['name'] for v in physical['states']],'state_units':[v['unit'] for v in physical['states']],
        'output_order':[v['name'] for v in physical['outputs']],'output_units':[v['unit'] for v in physical['outputs']]}
    Path(args.output).write_text(json.dumps(result,allow_nan=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
'''


def _standalone_source():
    definitions = (_hash, _json, _verify_members, _physical, _numeric, _binding,
                   validate_network_config, build_supervised_network, LearnedDynamics, _components)
    return ('"""Independent frozen physical next-state model; NumPy/Torch only."""\n'
            'import hashlib,json,math\nfrom pathlib import Path\nfrom types import SimpleNamespace\n'
            'import numpy as np\nimport torch\nfrom torch import nn\nContractError=ValueError\n\n'
            + '\n\n'.join(inspect.getsource(function) for function in definitions) + '\n' + _SIMULATOR)


def export_learned_model(root, supervised_root, interface, C, *, provenance=None):
    """Copy a verified S05 bundle into a new, physically bound model artifact.

    The caller resolves supervised_root through Workspace first. This helper
    checks its bundle hashes again and never imports exported executable code.
    """
    root = Path(root).resolve()
    source = (Path(supervised_root) / "bundle").resolve(strict=True)
    manifest = _json(source / "bundle_manifest.json")
    expected = {"model.pt", "network.json", "scaler.json", "data_interface.json", "features.json", "training_config.json", "inference.py", "predict.py", "README.md"}
    if manifest.get("kind") != "control-supervised-mlp-bundle-v1" or manifest.get("schema_version") != 1:
        raise ContractError("A verified S05 supervised inference bundle is required")
    _verify_members(source, manifest.get("files"), expected)
    physical = _physical(interface.to_dict() if hasattr(interface, "to_dict") else interface)
    ModelInterface.parse(physical)
    settings = validate_network_config(_json(source / "network.json"))
    data_interface = DataInterface.parse(_json(source / "data_interface.json")).to_dict()
    features, scaler = _json(source / "features.json"), _json(source / "scaler.json")
    descriptor = {"schema_version": 1, "representation": "learned_discrete_mlp", "interface": physical,
        "interface_hash": _hash({key: value for key, value in physical.items() if key != "name"}),
        "C": _numeric(C, (len(physical["outputs"]), len(physical["states"])), "C").tolist(),
        "input_semantics": "current_full_states_controls_disturbances", "target_semantics": "next_state_in_state_units",
        "transition_adapter": None, "linear_estimator_support": "EKF; no affine model certificate or A matrix",
        "feature_order": [{"group": group, "index": index, "name": item["name"], "unit": item["unit"], "signal": group + "::" + item["name"]}
            for group in ("states", "controls", "disturbances") for index, item in enumerate(physical[group])],
        "target_order": [{"signal": "states::" + item["name"], "offset": 1, "unit": item["unit"]} for item in physical["states"]],
        "provenance": json.loads(json.dumps(provenance or {}, allow_nan=False)),
        "source_plant_required_at_runtime": False}
    _binding(descriptor, settings, data_interface, features, scaler)
    copied = {"model.pt": "model.pt", "network_config.json": "network.json", "scaler.json": "scaler.json",
              "data_interface.json": "data_interface.json", "features.json": "features.json"}
    names = {*copied, "learned.json", "bundle"}
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in names):
        raise ContractError("Learned-model export filenames already exist; use a new stage")
    temporary = root / (".learned-export-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        for target, original in copied.items():
            shutil.copyfile(source / original, temporary / target)
        descriptor["files"] = {name: hashlib.sha256((temporary / name).read_bytes()).hexdigest() for name in copied}
        for target, original in copied.items():
            if descriptor["files"][target] != manifest["files"][original]:
                raise ContractError("Source supervised bundle changed during export")
        (temporary / "learned.json").write_text(json.dumps(descriptor, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        load_learned_model(temporary)
        bundle = temporary / "bundle"
        bundle.mkdir()
        for name in [*copied, "learned.json"]:
            shutil.copyfile(temporary / name, bundle / name)
        (bundle / "dynamics.py").write_text(_standalone_source(), encoding="utf-8")
        (bundle / "simulate.py").write_text(_SIMULATE_CLI, encoding="utf-8")
        (bundle / "README.md").write_text(
            "# Learned discrete physical dynamics\n\nRequires NumPy and Torch; CPU/float64. Copy this whole directory. No repository, original source plant, MATLAB, Engine or MCP session is loaded.\n\n"
            "`python -I simulate.py --input case.json --output rollout.json`\n\n"
            "Input: `{\"x0\":[...],\"controls\":[[...]],\"disturbances\":[[...]]}`. Disturbances may be omitted only when nd=0. Features are current full states, controls, disturbances in learned.json order. No history, hidden state or preview is inferred.\n\n"
            "The network produces **next_x in physical state units** after inverse scaling. This is already a discrete transition: never multiply by dt or integrate it again. Output is C@x. Physical bounds/units/dt are declarations, not independent physical-fidelity certification; inference does not clip states or certify closed-loop stability.\n\n"
            "Python: load `dynamics.py` by importlib.util.spec_from_file_location; `load_dynamics(bundle)` returns a differentiable frozen Torch module; `load_simulator(bundle)` returns reset(x0), state, step(u,d), output().\n\n"
            "Hashes in dynamics_manifest.json detect content changes; they are not a digital signature. This directory contains executable Python.\n", encoding="utf-8")
        bundle_manifest = {"schema_version": 1, "kind": "control-learned-dynamics-bundle-v1",
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in bundle.iterdir()},
            "representation": descriptor["representation"], "interface_hash": descriptor["interface_hash"],
            "source_plant_required_at_runtime": False}
        (bundle / "dynamics_manifest.json").write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name in sorted(names - {"learned.json"}):
            (temporary / name).rename(root / name)
        (temporary / "learned.json").rename(root / "learned.json")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return descriptor
