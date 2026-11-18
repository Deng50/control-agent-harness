"""Explicit-memory external expert callables over S04 immutable source snapshots."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np

from .contracts import ContractError, ModelInterface, canonical_hash
from .model_adapters import _CodeStore, _SnapshotRuntime, _vector, _identifier


class ExpertFailure(RuntimeError):
    def __init__(self, message, *, status=None):
        super().__init__(message)
        self.status = status


def _finite_json(value, label, *, max_bytes=65536):
    try:
        text = json.dumps(value, allow_nan=False, ensure_ascii=False)
        if len(text.encode("utf-8")) > max_bytes: raise ValueError("JSON budget exceeded")
        normalized = json.loads(text)
    except (TypeError, ValueError, RecursionError, OverflowError) as error:
        raise ContractError(label + " must be bounded finite JSON") from error
    return normalized


def validate_expert_spec(spec):
    """Static declaration only. Neither Python nor MATLAB code is run here."""
    allowed = {"schema_version", "backend", "source_file", "entrypoint", "interface", "parameters", "time_unit", "control_period",
        "observations", "reference_available", "disturbance_available", "previous_action_available", "initial_memory", "action_mapping",
        "failure_policy", "input_rate_bounds"}
    if (not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or spec["schema_version"] != 1
            or not isinstance(spec.get("backend"), str) or spec["backend"] not in {"python_callable", "matlab_function"} or set(spec) - allowed):
        raise ContractError("Explicit version-1 external expert callable declaration required")
    value = _finite_json(spec, "expert specification", max_bytes=1024 * 1024)
    source = value.get("source_file")
    if not isinstance(source, str) or not source or PureWindowsPath(source).drive:
        raise ContractError("Expert source_file must be a confined relative material path")
    path = PurePosixPath(source.replace("\\", "/"))
    suffix = ".py" if value["backend"] == "python_callable" else ".m"
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != suffix:
        raise ContractError("Expert source_file/backend must match its frozen source")
    value["source_file"] = path.as_posix(); _identifier(value.get("entrypoint"), "entrypoint")
    raw = value.get("interface")
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or type(raw.get("sample_time")) not in {int, float}:
        raise ContractError("Expert requires the explicit discrete physical Plant interface")
    for key in ("states", "controls", "outputs", "disturbances"):
        if not isinstance(raw.get(key, []), list): raise ContractError("Physical groups must be explicit ordered lists")
        for variable in raw.get(key, []):
            if (not isinstance(variable, dict) or any(not isinstance(variable.get(name), str) or not variable[name].strip() for name in ("name", "unit"))
                    or any(type(variable.get(name)) not in {int, float} for name in ("lower", "upper"))):
                raise ContractError("Physical names/units/bounds must be explicit and typed")
    interface = ModelInterface.parse(raw)
    period = value.get("control_period")
    if value.get("time_unit") != "s" or type(period) not in {int, float} or not math.isfinite(period) or period <= 0:
        raise ContractError("Expert control_period must be an explicit positive number in seconds")
    ratio = period / interface.sample_time
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=1e-10, abs_tol=1e-10) or round(ratio) > 10000:
        raise ContractError("Expert control_period must be an integer multiple of Plant sample_time (1..10000 subticks)")
    if value.get("failure_policy") != "stop": raise ContractError("Only explicit failure_policy='stop' is supported; no silent clamp/fallback")
    if not isinstance(value.get("parameters"), dict) or not isinstance(value.get("initial_memory"), dict):
        raise ContractError("parameters and initial_memory must be explicit finite JSON objects")
    value["initial_memory"] = _finite_json(value["initial_memory"], "initial_memory")
    for flag in ("reference_available", "disturbance_available", "previous_action_available"):
        if type(value.get(flag)) is not bool: raise ContractError(flag + " must be an explicit availability boolean")
    observations = value.get("observations")
    if not isinstance(observations, list) or not observations or len(observations) > 65536:
        raise ContractError("Explicit bounded observation mappings required; full state is never assumed")
    names = set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {"name", "source", "index", "plant_name", "unit", "availability", "delay_steps"}:
            raise ContractError("Each observation requires name/source/index/plant_name/unit/availability/delay_steps")
        if not isinstance(item["name"], str) or not item["name"].strip() or item["name"] in names: raise ContractError("Observation names must be explicit and unique")
        names.add(item["name"])
        if not isinstance(item["source"], str) or item["source"] not in {"state", "output"}: raise ContractError("Observation source must explicitly be state or output")
        variables = interface.states if item["source"] == "state" else interface.outputs
        index = item["index"]
        if type(index) is not int or not 0 <= index < len(variables) or item["plant_name"] != variables[index].name or item["unit"] != variables[index].unit:
            raise ContractError("Observation index/name/unit disagrees with the physical Plant interface")
        if not isinstance(item["availability"], str) or item["availability"] not in {"measured", "estimated", "simulation_only"}:
            raise ContractError("Observation availability must be declared; it is not proof of measured data")
        if type(item["delay_steps"]) is not int or item["delay_steps"] != 0:
            raise ContractError("This callable boundary requires current available observations; delayed channels need an explicit supported history adapter")
    mapping = value.get("action_mapping")
    if not isinstance(mapping, list) or len(mapping) != interface.nu: raise ContractError("An explicit complete action_mapping is required")
    actions, controls = set(), set()
    for item in mapping:
        if not isinstance(item, dict) or set(item) != {"action_index", "plant_control_index", "plant_control_name", "unit"}:
            raise ContractError("Action mapping requires explicit action/Plant indices, Plant name and unit")
        a, p = item["action_index"], item["plant_control_index"]
        if type(a) is not int or type(p) is not int or not 0 <= a < interface.nu or not 0 <= p < interface.nu:
            raise ContractError("Action mapping indices must address every control exactly once")
        if a in actions or p in controls or item["plant_control_name"] != interface.controls[p].name or item["unit"] != interface.controls[p].unit:
            raise ContractError("Action mapping permutation/name/unit conflict")
        actions.add(a); controls.add(p)
    if "input_rate_bounds" in value:
        bounds = value["input_rate_bounds"]
        if not isinstance(bounds, list) or len(bounds) != interface.nu or any(not isinstance(pair, list) or len(pair) != 2 or
                any(type(number) not in {int, float} or not math.isfinite(number) for number in pair) or pair[0] >= pair[1] for pair in bounds):
            raise ContractError("Expert input_rate_bounds are ordered Plant-control increments per control update")
    value["control_period"] = float(period)
    return value


def validate_python_expert_source(root, spec, expected_files):
    spec = validate_expert_spec(spec)
    store = _CodeStore(root, spec["source_file"], expected_files)
    store.code(store.entry)  # S04 import/IO policy, hash before AST/execute.
    payload = store.cache[store.entry][0]
    tree = ast.parse(payload)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == spec["entrypoint"]), None)
    if (function is None or len(function.args.posonlyargs) + len(function.args.args) != 6
            or function.args.vararg or function.args.kwarg or function.args.kwonlyargs):
        raise ContractError("External expert entrypoint must be a top-level six-positional-argument function")
    if any(isinstance(node, (ast.Global, ast.Nonlocal)) for node in ast.walk(tree)):
        raise ContractError("Expose expert mutable state in explicit memory, not global/nonlocal state")
    return {"status": "supported_explicit_callable", "source_executed": False,
        "import_policy": "S04 immutable mathematical source subset; not a Python security sandbox", "arity": 6,
        "return_contract": "exact tuple(action,next_memory,status); status='ok' only success"}


class ExternalExpertAdapter:
    def __init__(self, root, spec, *, expected_file_hashes):
        self.spec = validate_expert_spec(spec)
        if self.spec["backend"] != "python_callable": raise ContractError("Use the fixed native batch for MATLAB experts")
        self.interface = ModelInterface.parse(self.spec["interface"])
        self.source_support = validate_python_expert_source(root, self.spec, expected_file_hashes)
        self.store = _CodeStore(root, self.spec["source_file"], expected_file_hashes)
        self.runtime = None; self.memory = None; self.calls = 0; self.resets = 0; self._failed = False

    def reset(self):
        if self.runtime is not None: self.runtime.close()
        self.runtime = _SnapshotRuntime(self.store)
        self.function = self.runtime.function(self.spec["entrypoint"])
        self.memory = copy.deepcopy(self.spec["initial_memory"]); self._failed = False; self.resets += 1
        return copy.deepcopy(self.memory)

    def _invoke(self, runtime, function, args):
        result = runtime.call(function, *copy.deepcopy(args))
        if type(result) is not tuple or len(result) != 3: raise ExpertFailure("Expert must return exact tuple(action,next_memory,status)")
        action, memory, status = result
        if not isinstance(status, str) or status != "ok": raise ExpertFailure("External expert returned non-success status", status=status if isinstance(status, str) else "invalid_status_type")
        try:
            action = _vector(action, self.interface.nu, "expert action", source=True)
            if not isinstance(memory, dict): raise ContractError("next_memory must be an explicit JSON object")
            memory = _finite_json(memory, "next_memory")
        except (ContractError, FloatingPointError) as error: raise ExpertFailure(str(error), status=status) from error
        return action, memory, status

    def act(self, observation, reference, disturbance, previous_action):
        if self.runtime is None or self._failed: raise ContractError("Reset the expert before use; failed trajectories cannot continue implicitly")
        previous = _vector(previous_action, self.interface.nu, "actual previous action")
        args = [_vector(observation, len(self.spec["observations"]), "observation"),
            _vector(reference, self.interface.ny, "reference") if self.spec["reference_available"] else np.empty(0, dtype=np.float64),
            _vector(disturbance, self.interface.nd, "disturbance") if self.spec["disturbance_available"] else np.empty(0, dtype=np.float64),
            previous if self.spec["previous_action_available"] else np.empty(0, dtype=np.float64),
            copy.deepcopy(self.memory), copy.deepcopy(self.spec["parameters"])]
        try:
            first = self._invoke(self.runtime, self.function, args)
            again = self._invoke(self.runtime, self.function, args)
            fresh = _SnapshotRuntime(self.store)
            try: independent = self._invoke(fresh, fresh.function(self.spec["entrypoint"]), args)
            finally: fresh.close()
            for compared in (again, independent):
                if not np.allclose(first[0], compared[0], rtol=1e-10, atol=1e-10) or canonical_hash(first[1]) != canonical_hash(compared[1]):
                    raise ExpertFailure("Expert behavior depends on undeclared hidden state or nondeterminism; expose it through memory")
            action = np.empty(self.interface.nu, dtype=np.float64)
            for mapping in self.spec["action_mapping"]: action[mapping["plant_control_index"]] = first[0][mapping["action_index"]]
            low, high = np.array([v.lower for v in self.interface.controls]), np.array([v.upper for v in self.interface.controls])
            if np.any(action < low - 1e-12) or np.any(action > high + 1e-12): raise ExpertFailure("Expert action violates declared amplitude bounds; action was not clamped")
            if "input_rate_bounds" in self.spec:
                rate = np.asarray(self.spec["input_rate_bounds"]); delta = action - previous
                if np.any(delta < rate[:, 0] - 1e-12) or np.any(delta > rate[:, 1] + 1e-12): raise ExpertFailure("Expert action violates declared per-update rate bounds; action was not clamped")
            before = copy.deepcopy(self.memory); self.memory = copy.deepcopy(first[1]); self.calls += 1
            return action.copy(), {"status": "ok", "memory_before": before, "memory_after": copy.deepcopy(self.memory), "repeated_and_fresh_calls_checked": True}
        except Exception:
            self._failed = True
            raise

    def describe(self):
        return {"backend": self.spec["backend"], "entrypoint": self.spec["entrypoint"], "control_period": self.spec["control_period"],
            "time_unit": "s", "failure_policy": "stop", "memory_semantics": "explicit JSON; deep reset plus fresh private module per trajectory",
            "valid_policy_calls": self.calls, "resets": self.resets, "repeatability": "same-input repeat plus independent fresh module at visited calls",
            "source_hashes": {str(path.relative_to(self.store.root).as_posix()): hashlib.sha256(row[0]).hexdigest() for path, row in self.store.cache.items()},
            "source_support": self.source_support, "memory_is_student_feature": False}

    def close(self):
        if self.runtime is not None: self.runtime.close(); self.runtime = None
        self.memory = None


def load_external_expert(snapshot_root, spec, *, expected_file_hashes):
    return ExternalExpertAdapter(snapshot_root, spec, expected_file_hashes=expected_file_hashes)
