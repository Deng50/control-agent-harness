"""Native-MCP batch exchange for explicit MATLAB/Simulink plant adapters.

Registration is static. Only the trusted Host job may create a prepare/finalize
exchange; model JSON cannot select a precomputed result or declare it trusted.
``source_root`` is the immutable material artifact root, and ``source_file`` is
relative to that root (normally ``source/<file>.m`` or ``source/<file>.slx``).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PureWindowsPath
from xml.etree import ElementTree

import numpy as np

from .contracts import ContractError, ModelInterface, canonical_hash


BACKENDS = {"matlab_function", "simulink"}
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_FUNCTION = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*")
_SHA = re.compile(r"[a-f0-9]{64}")
_MAX_BYTES = 256 * 1024 * 1024
_SLX_BLOCKS = {"Inport", "Outport", "UnitDelay", "Gain", "Sum", "Constant", "Product", "Saturate",
               "Switch", "RelationalOperator", "Logic", "MinMax", "Abs", "Signum", "Math", "Trigonometry",
               "Mux", "Demux", "DataTypeConversion", "SignalConversion", "Terminator", "Bias"}


def _json(value, label):
    try:
        return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label} must contain finite JSON values") from error


def _has_bool(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return any(_has_bool(item) for item in value)
    return False


def _native_numeric(value, label, max_rank=2):
    raw = np.asarray(value)
    if _has_bool(value) or raw.dtype.kind not in "iuf" or raw.ndim > max_rank:
        raise ContractError(f"Native batch {label} must use real numeric MATLAB scalar/vector/matrix JSON")
    return raw


def _relative(value):
    if (not isinstance(value, str) or not value or Path(value).is_absolute() or
            PureWindowsPath(value).drive or any(part in {"", ".", ".."} for part in value.replace("\\", "/").split("/")) or
            ":" in value):
        raise ContractError("source_file must be an explicit relative snapshot file")
    return value.replace("\\", "/")


def _name(value, label, *, function=False):
    expression = _FUNCTION if function else _IDENTIFIER
    if not isinstance(value, str) or not expression.fullmatch(value):
        raise ContractError(f"{label} must be an explicit MATLAB identifier")
    return value


def _sid(value):
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*(?::[1-9][0-9]*)*", value):
        raise ContractError("Simulink SID must be an explicit numeric SID string")
    return value


def _port(value):
    if type(value) is not int or value < 1:
        raise ContractError("Simulink port numbers are explicit positive integers")
    return value


def validate_matlab_spec(spec: dict) -> dict:
    """Validate declarations only: never start MATLAB, import or invoke source."""
    if not isinstance(spec, dict) or spec.get("schema_version") != 1 or spec.get("backend") not in BACKENDS:
        raise ContractError("Version-1 matlab_function or simulink adapter required")
    allowed = {"schema_version", "backend", "source_file", "interface", "parameters", "source_semantics",
               "integration", "time_unit", "derivative_units", "entrypoint", "output_entrypoint", "C", "simulink"}
    if set(spec) - allowed:
        raise ContractError(f"Unknown MATLAB adapter fields: {sorted(set(spec) - allowed)}")
    value = _json(spec, "MATLAB adapter")
    raw_interface = value.get("interface")
    if (not isinstance(raw_interface, dict) or type(raw_interface.get("schema_version")) is not int or
            type(raw_interface.get("sample_time")) not in {int, float}):
        raise ContractError("Explicit numeric discrete ModelInterface required")
    for group in ("states", "controls", "outputs", "disturbances"):
        variables = raw_interface.get(group, [])
        if not isinstance(variables, list):
            raise ContractError("Interface variable groups must be ordered JSON lists")
        for variable in variables:
            if (not isinstance(variable, dict) or not isinstance(variable.get("name"), str) or
                    not isinstance(variable.get("unit"), str) or
                    any(type(variable.get(key)) not in {int, float} for key in ("lower", "upper"))):
                raise ContractError("Every ordered variable requires explicit name/unit and numeric bounds")
    interface = ModelInterface.parse(raw_interface)
    value["interface"] = _json(interface.to_dict(), "interface")
    source = _relative(value.get("source_file"))
    value["source_file"] = source
    parameters = value.setdefault("parameters", {})
    if not isinstance(parameters, dict):
        raise ContractError("parameters must be an explicit JSON object")

    def parameter_names(item):
        if isinstance(item, dict):
            for key, child in item.items():
                _name(key, "MATLAB parameter field")
                parameter_names(child)
        elif isinstance(item, list):
            for child in item:
                parameter_names(child)

    parameter_names(parameters)
    semantics = value.get("source_semantics")
    if semantics not in {"transition", "derivative"}:
        raise ContractError("Explicit source_semantics transition or derivative required")
    if semantics == "derivative":
        integration = value.get("integration")
        if (value["backend"] != "matlab_function" or value.get("time_unit") != "s" or
                not isinstance(value.get("derivative_units"), list) or len(value["derivative_units"]) != interface.nx or
                any(not isinstance(unit, str) or not unit.strip() for unit in value["derivative_units"])):
            raise ContractError("Derivative MATLAB functions require time_unit s and ordered derivative_units")
        for state, unit in zip(interface.states, value["derivative_units"]):
            state_unit = state.unit.replace(" ", "")
            expected_units = {f"{state_unit}/s", f"({state_unit})/s"}
            if state_unit == "1":
                expected_units.add("s^-1")
            if unit.replace(" ", "") not in expected_units:
                raise ContractError(f"Derivative unit for {state.name} must explicitly be ({state.unit})/s; no unit conversion is inferred")
        if (not isinstance(integration, dict) or set(integration) != {"method", "substeps", "input_hold"} or
                integration.get("method") not in {"euler", "rk4"} or type(integration.get("substeps")) is not int or
                not 1 <= integration["substeps"] <= 1024 or integration.get("input_hold") != "zero_order_hold"):
            raise ContractError("Derivative integration requires Euler/RK4, 1..1024 substeps and zero_order_hold")
    elif any(key in value for key in ("integration", "time_unit", "derivative_units")):
        raise ContractError("Transition adapters must not supply derivative integration declarations")
    if value["backend"] == "matlab_function":
        if Path(source).suffix.lower() != ".m" or "simulink" in value:
            raise ContractError("matlab_function requires a .m source and no Simulink mapping")
        entrypoint = _name(value.get("entrypoint"), "entrypoint", function=True)
        suffix = [f"+{part}" for part in entrypoint.split(".")[:-1]] + [entrypoint.split(".")[-1] + ".m"]
        if list(Path(source).parts[-len(suffix):]) != suffix:
            raise ContractError("MATLAB entrypoint/package must match the declared .m snapshot filename")
        if ("C" in value) == ("output_entrypoint" in value):
            raise ContractError("Declare exactly one state-only output_entrypoint(x,parameters) or C")
        if "output_entrypoint" in value:
            _name(value["output_entrypoint"], "output_entrypoint", function=True)
        else:
            try:
                c = np.asarray(value["C"])
                if c.dtype.kind not in "iuf" or _has_bool(value["C"]):
                    raise ValueError("numeric entries required")
                c = c.astype(np.float64)
            except (TypeError, ValueError) as error:
                raise ContractError("C must be a finite ny-by-nx matrix") from error
            if c.shape != (interface.ny, interface.nx) or not np.isfinite(c).all():
                raise ContractError("C must be a finite ny-by-nx matrix")
    else:
        if Path(source).suffix.lower() != ".slx" or any(key in value for key in ("entrypoint", "output_entrypoint", "C")):
            raise ContractError("simulink requires an SLX source and explicit port/state mappings")
        mapping = value.get("simulink")
        if not isinstance(mapping, dict) or set(mapping) != {"controls", "disturbances", "outputs", "states", "sample_time_variable", "solver"}:
            raise ContractError("Simulink mapping requires controls/disturbances/outputs/states, sample_time_variable and solver")
        all_inputs = []
        for group, count in (("controls", interface.nu), ("disturbances", interface.nd), ("outputs", interface.ny)):
            channels = mapping.get(group)
            if not isinstance(channels, list) or len(channels) != count:
                raise ContractError(f"Simulink {group} mapping width differs from the declared interface")
            for channel in channels:
                if not isinstance(channel, dict) or set(channel) != {"port", "sid"}:
                    raise ContractError("Each supported scalar port requires only explicit port and SID")
                _port(channel["port"])
                _sid(channel["sid"])
            if len({channel["port"] for channel in channels}) != len(channels):
                raise ContractError("Duplicate scalar port mapping")
            if group != "outputs":
                all_inputs.extend(channels)
        if sorted(channel["port"] for channel in all_inputs) != list(range(1, interface.nu + interface.nd + 1)):
            raise ContractError("Control/disturbance inports must be disjoint and cover every root port")
        if sorted(channel["port"] for channel in mapping["outputs"]) != list(range(1, interface.ny + 1)):
            raise ContractError("Output mapping must cover every declared scalar root outport")
        states = mapping.get("states")
        if not isinstance(states, list) or len(states) != interface.nx:
            raise ContractError("Every declared state needs an explicit scalar logged-state SID")
        for state in states:
            if not isinstance(state, dict) or set(state) != {"sid", "initial_condition_variable"}:
                raise ContractError("Each scalar state requires SID and model-workspace initial_condition_variable")
            _sid(state["sid"])
            _name(state["initial_condition_variable"], "initial_condition_variable")
        if len({state["sid"] for state in states}) != interface.nx or len({state["initial_condition_variable"] for state in states}) != interface.nx:
            raise ContractError("State SIDs and scalar initial-condition variables must be unambiguous")
        _name(mapping["sample_time_variable"], "sample_time_variable")
        if mapping["sample_time_variable"] in {state["initial_condition_variable"] for state in states}:
            raise ContractError("Sample-time and initial-state variables cannot alias")
        if mapping["solver"] != "FixedStepDiscrete":
            raise ContractError("The S04 SLX adapter supports explicit fixed-step discrete scalar-state models")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(value, label):
    if (not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"] or
            not isinstance(value.get("metadata_hash"), str) or not _SHA.fullmatch(value["metadata_hash"])):
        raise ContractError(f"Explicit immutable {label} id and metadata_hash required")
    return _json(value, label)


def validate_simulink_source(path: str | Path, spec: dict) -> dict:
    """Static, flat built-in subset; state cuts are declared, never inferred."""
    spec = validate_matlab_spec(spec)
    if spec["backend"] != "simulink":
        raise ContractError("SLX static validation requires a Simulink adapter")
    blocks, parents = {}, {}
    callbacks = {"PreLoadFcn", "PostLoadFcn", "InitFcn", "StartFcn", "StopFcn", "CloseFcn", "PreSaveFcn",
                 "PostSaveFcn", "LoadFcn", "MaskInitialization", "MaskDisplay", "OpenFcn", "DeleteFcn", "CopyFcn"}
    try:
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) > 2048 or sum(item.file_size for item in archive.infolist()) > _MAX_BYTES:
                raise ContractError("SLX exceeds its static ZIP budget")
            for item in archive.infolist():
                if not item.filename.lower().endswith(".xml"):
                    continue
                raw = archive.read(item)
                declarations = raw.replace(b"\x00", b"").lower()
                if b"<!doctype" in declarations or b"<!entity" in declarations:
                    raise ContractError("SLX XML entity declarations are unsupported")
                document = ElementTree.fromstring(raw)
                for node in document.iter():
                    tag = node.tag.rsplit("}", 1)[-1]
                    if tag == "P" and node.get("Name") in callbacks and (node.text or "").strip():
                        raise ContractError("SLX callbacks/masks are opaque; expose a thin supported numerical model")
                    if tag == "Block":
                        sid, kind = node.get("SID"), node.get("BlockType")
                        if not sid or sid in blocks or kind not in _SLX_BLOCKS:
                            raise ContractError("SLX uses an opaque/hierarchical/stateful block outside the S04 built-in subset")
                        blocks[sid] = kind
                    elif tag == "Line":
                        source = next((child.text for child in node if child.tag.rsplit("}", 1)[-1] == "P" and child.get("Name") == "Src"), None)
                        if not source:
                            continue
                        match = re.match(r"^([1-9][0-9]*)#out:", source)
                        if not match:
                            raise ContractError("Unsupported SLX source-port encoding")
                        for target in node.iter():
                            if target.tag.rsplit("}", 1)[-1] == "P" and target.get("Name") == "Dst":
                                destination = re.match(r"^([1-9][0-9]*)#in:", target.text or "")
                                if not destination:
                                    raise ContractError("Unsupported SLX destination-port encoding")
                                parents.setdefault(destination[1], set()).add(match[1])
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as error:
        raise ContractError(f"Cannot inspect SLX built-in subset: {error}") from error
    states = {item["sid"] for item in spec["simulink"]["states"]}
    if {sid for sid, kind in blocks.items() if kind == "UnitDelay"} != states:
        raise ContractError("Every Unit Delay state must be explicitly declared; hidden states are unsupported")
    for group, kind in (("controls", "Inport"), ("disturbances", "Inport"), ("outputs", "Outport")):
        for channel in spec["simulink"][group]:
            if blocks.get(channel["sid"]) != kind:
                raise ContractError("Declared port SID does not identify the required scalar block type")

    def direct(sid, visiting):
        if sid in states or blocks.get(sid) == "Constant":
            return False
        if sid not in blocks or sid in visiting:
            raise ContractError("Unresolved/algebraic SLX dependency cannot establish state-only outputs")
        if blocks[sid] == "Inport":
            return True
        incoming = parents.get(sid, set())
        if not incoming:
            raise ContractError("Unconnected SLX output dependency is unsupported")
        return any(direct(parent, visiting | {sid}) for parent in incoming)

    for output in spec["simulink"]["outputs"]:
        if direct(output["sid"], set()):
            raise ContractError("SLX output has direct control/disturbance feedthrough; state-only output is required")
    return {"status": "supported_subset", "source_executed": False, "output_semantics": "state_only",
            "state_cut_sids": sorted(states), "block_types": sorted(set(blocks.values())),
            "limitations": "flat built-in scalar blocks, explicitly mapped Unit Delays, no opaque callbacks or direct feedthrough"}


def prepare_matlab_manifest(spec: dict, source_root: str | Path, plan: dict, *, exchange_dir: str | Path,
                            request_sha256: str, nonce: str, model_version: dict,
                            material_version: dict) -> dict:
    """Called by the normal sandboxed worker's private prepare phase, not JSON."""
    spec = validate_matlab_spec(spec)
    if not isinstance(request_sha256, str) or not _SHA.fullmatch(request_sha256) or not isinstance(nonce, str) or not re.fullmatch(r"[a-f0-9-]{36}", nonce):
        raise ContractError("Host request SHA and random batch nonce required")
    model_version, material_version = _version(model_version, "model"), _version(material_version, "material")
    root = Path(source_root).resolve(strict=True)
    metadata_bytes = (root / "artifact.json").read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    if (metadata.get("id") != material_version["id"] or canonical_hash(metadata) != material_version["metadata_hash"] or
            not isinstance(metadata.get("files"), dict)):
        raise ContractError("MATLAB source root is not the bound immutable material artifact")
    expected_files = dict(metadata["files"])
    expected_files["artifact.json"] = hashlib.sha256(metadata_bytes).hexdigest()
    if spec["backend"] == "matlab_function":
        from .matlab_source import validate_matlab_sources
        source_validation = validate_matlab_sources(root, spec, metadata["files"])
    else:
        source_validation = validate_simulink_source(root / spec["source_file"], spec)
    exchange = Path(exchange_dir)
    if not exchange.is_absolute() or not root.is_dir() or exchange.resolve().is_relative_to(root):
        raise ContractError("Private MATLAB exchange must be an absolute directory outside the immutable source tree")
    exchange.mkdir(parents=True, exist_ok=False)
    copied = exchange / "input"
    copied.mkdir()
    inventory, total = {}, 0
    try:
        for path in sorted(root.rglob("*")):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ContractError("MATLAB source snapshots cannot contain filesystem aliases")
            if path.is_dir():
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_size > 64 * 1024 * 1024:
                raise ContractError("MATLAB snapshot contains an unsupported/oversize source file")
            relative = path.relative_to(root).as_posix()
            if relative not in expected_files:
                raise ContractError("MATLAB material contains an unregistered source file")
            total += info.st_size
            if len(inventory) >= 2048 or total > _MAX_BYTES:
                raise ContractError("MATLAB source snapshot exceeds its fixed file/byte budget")
            target = copied / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = _sha(path)
            if before != expected_files[relative]:
                raise ContractError("MATLAB material source differs from its bound artifact file hash")
            shutil.copyfile(path, target)
            if _sha(target) != before or _sha(path) != before:
                raise ContractError("MATLAB source changed during private snapshot copy")
            inventory[relative] = before
        if inventory != expected_files:
            raise ContractError("MATLAB material file inventory changed before its frozen copy")
        source = copied / spec["source_file"]
        if spec["source_file"] not in inventory or not source.is_file():
            raise ContractError("Declared MATLAB source is absent from its immutable material snapshot")
        if not isinstance(plan, dict) or plan.get("schema_version") != 1 or not isinstance(plan.get("trajectories"), list) or not plan["trajectories"]:
            raise ContractError("An explicit prevalidated sampling plan is required")
        interface = ModelInterface.parse(spec["interface"])
        interface.compatible(ModelInterface.parse(plan["interface"]))
        if plan.get("dt") != interface.sample_time or not _SHA.fullmatch(plan.get("plan_hash", "")):
            raise ContractError("Sampling plan time/hash differs from the declared adapter")
        manifest = {"schema_version": 1, "kind": "control-matlab-batch-v1", "nonce": nonce,
                    "request_sha256": request_sha256, "model_version": model_version, "material_version": material_version,
                    "spec": spec, "spec_hash": canonical_hash(spec), "plan": _json(plan, "sampling plan"), "source_root": str(copied),
                    "source_file": str(source), "source_sha256": inventory[spec["source_file"]],
                    "source_inventory": inventory, "source_files": [{"relative_path": name, "sha256": digest}
                        for name, digest in inventory.items()], "source_validation": source_validation,
                    "result_path": str(exchange / "result.json")}
        if spec["backend"] == "matlab_function":
            manifest["function_path_root"] = str(source.parents[len(spec["entrypoint"].split(".")) - 1])
        path = exchange / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return {"status": "verified", "matlab_phase": "prepared", "manifest_path": str(path),
                "exchange_dir": str(exchange), "manifest_sha256": _sha(path), "result_path": manifest["result_path"], "request_sha256": request_sha256,
                "nonce": nonce, "plan_hash": plan["plan_hash"], "backend": spec["backend"]}
    except BaseException:
        # Private exchange evidence remains for the parent job to diagnose.
        raise


class MatlabBatchError(RuntimeError):
    """An actual synchronous batch stopped; previously complete rows are usable."""


class MatlabBatchAdapter:
    """Replay only authenticated native batch rows in the generic sampler API."""
    def __init__(self, spec, plan, result):
        self.spec, self.plan, self.result = spec, plan, result
        self.interface = ModelInterface.parse(spec["interface"])
        self._trajectory = -1
        self._position = 0
        self._state = self._output = None
        self._closed = False
        raw = result.get("arrays")
        if not isinstance(raw, dict):
            raise ContractError("Native MATLAB batch did not return arrays")
        ids = np.asarray(raw.get("trajectory_id"))
        if ids.size == 0:
            ids = np.asarray([], dtype=np.int64)
        if ids.dtype.kind not in "iu" or ids.ndim > 1:
            raise ContractError("Native batch trajectory IDs must be integers")
        ids = ids.reshape(-1)
        count = len(ids)
        self.arrays = {"trajectory_id": ids.astype(np.int64)}
        widths = {"x": self.interface.nx, "u": self.interface.nu, "d": self.interface.nd,
                  "next_x": self.interface.nx, "y": self.interface.ny, "next_y": self.interface.ny}
        for key, width in widths.items():
            try:
                source = _native_numeric(raw[key], key)
                if source.ndim == 2 and source.shape != (count, width):
                    raise ContractError(f"Native batch {key} matrix shape is incorrect")
                value = source.astype(np.float64).reshape(count, width)
            except (KeyError, ValueError, TypeError) as error:
                raise ContractError(f"Native batch {key} has the wrong shape") from error
            if not np.isfinite(value).all():
                raise ContractError(f"Native batch {key} is nonfinite")
            self.arrays[key] = value
        try:
            self.arrays["time"] = _native_numeric(raw["time"], "time", max_rank=1).astype(np.float64).reshape(count)
        except (KeyError, ValueError, TypeError) as error:
            raise ContractError("Native batch time has the wrong shape") from error
        if not np.isfinite(self.arrays["time"]).all():
            raise ContractError("Native batch time is nonfinite")
        summaries = result.get("trajectories")
        if not isinstance(summaries, list) or len(summaries) > len(plan["trajectories"]):
            raise ContractError("Native batch trajectory summaries conflict with the approved plan")
        self._groups = []
        offset = 0
        for index, summary in enumerate(summaries):
            expected = plan["trajectories"][index]
            steps = summary.get("completed_steps")
            if (summary.get("trajectory_id") != expected["trajectory_id"] or type(steps) is not int or
                    not 0 <= steps <= plan["steps"] or summary.get("status") not in {"completed", "partial", "failed"}):
                raise ContractError("Native batch trajectory identity/count is invalid")
            end = offset + steps
            if end > count or not np.all(ids[offset:end] == expected["trajectory_id"]):
                raise ContractError("Native rows are not an ordered trajectory prefix")
            for key in ("u", "d"):
                if not np.allclose(self.arrays[key][offset:end], np.asarray(expected[key], dtype=float)[:steps], rtol=0, atol=1e-12):
                    raise ContractError(f"Native {key} differs from the approved sampling plan")
            if not np.allclose(self.arrays["time"][offset:end], np.asarray(expected["t"], dtype=float)[:steps], rtol=0, atol=1e-12):
                raise ContractError("Native times differ from the approved sampling plan")
            if steps:
                if not np.allclose(self.arrays["x"][offset], expected["x0"], rtol=0, atol=1e-12):
                    raise ContractError("Native trajectory initial state differs from its approved x0")
                if (not np.allclose(self.arrays["x"][offset + 1:end], self.arrays["next_x"][offset:end - 1], rtol=0, atol=1e-12) or
                        not np.allclose(self.arrays["y"][offset + 1:end], self.arrays["next_y"][offset:end - 1], rtol=0, atol=1e-12)):
                    raise ContractError("Native state/output rows are discontinuous within a trajectory")
            if summary["status"] == "completed" and steps != plan["steps"]:
                raise ContractError("Completed native trajectory is incomplete")
            if summary["status"] != "completed" and index != len(summaries) - 1:
                raise ContractError("Native batch must stop at the first incomplete trajectory")
            self._groups.append((offset, end, summary))
            offset = end
        if offset != count or (result.get("status") == "completed" and
                               (len(summaries) != len(plan["trajectories"]) or any(row[2]["status"] != "completed" for row in self._groups))):
            raise ContractError("Native completion/count contradicts its trajectory evidence")
        if result.get("status") not in {"completed", "partial", "failed"}:
            raise ContractError("Native batch has no explicit terminal status")
        all_complete = (len(summaries) == len(plan["trajectories"]) and
                        all(row[2]["status"] == "completed" for row in self._groups))
        if all_complete != (result["status"] == "completed"):
            raise ContractError("Native terminal status contradicts completed trajectories")
        if not all_complete:
            failure = result.get("failure")
            if not self._groups or not isinstance(failure, dict):
                raise ContractError("Incomplete native batch requires a bound failure record")
            last = self._groups[-1][2]
            if (last["status"] == "completed" or last["completed_steps"] >= plan["steps"] or
                    failure.get("trajectory_id") != last["trajectory_id"] or
                    failure.get("step_index") != last["completed_steps"] or
                    result["status"] != ("partial" if count else "failed")):
                raise ContractError("Native failure/prefix/terminal evidence is inconsistent")
        elif result.get("failure") not in (None, []):
            raise ContractError("Completed native batch must not contain a failure")
        if "C" in spec:
            c = np.asarray(spec["C"], dtype=np.float64)
            if (not np.allclose(self.arrays["y"], self.arrays["x"] @ c.T, rtol=1e-10, atol=1e-12) or
                    not np.allclose(self.arrays["next_y"], self.arrays["next_x"] @ c.T, rtol=1e-10, atol=1e-12)):
                raise ContractError("Native outputs differ from the explicitly declared C map")

    def reset(self, x0):
        if self._closed:
            raise ContractError("MATLAB batch adapter is closed")
        value = np.asarray(x0, dtype=np.float64)
        if value.shape != (self.interface.nx,) or not np.isfinite(value).all():
            raise ContractError("MATLAB adapter reset requires a finite nx-vector")
        self._trajectory += 1
        if self._trajectory >= len(self._groups):
            raise MatlabBatchError("MATLAB batch stopped before this trajectory")
        expected = self.plan["trajectories"][self._trajectory]
        if not np.allclose(value, expected["x0"], rtol=0, atol=1e-12):
            raise ContractError("Sampler reset differs from approved MATLAB x0")
        start, end, summary = self._groups[self._trajectory]
        self._position = start
        self._state = np.asarray(expected["x0"], dtype=np.float64).copy()
        self._output = None
        if start == end:
            raise MatlabBatchError(summary.get("error_message", "MATLAB trajectory failed before its first sample"))
        self._output = self.arrays["y"][start].copy()
        return self.state

    @property
    def state(self):
        if self._state is None:
            raise ContractError("Reset the MATLAB batch adapter first")
        return self._state.copy()

    def output(self):
        if self._output is None:
            raise ContractError("Reset the MATLAB batch adapter first")
        return self._output.copy()

    def step(self, u, d=None):
        if self._state is None:
            raise ContractError("Reset the MATLAB batch adapter first")
        _, end, summary = self._groups[self._trajectory]
        if self._position >= end:
            raise MatlabBatchError(summary.get("error_message", "MATLAB trajectory batch prefix is exhausted"))
        index = self._position
        u = np.asarray(u, dtype=np.float64)
        d = np.zeros(self.interface.nd) if d is None else np.asarray(d, dtype=np.float64)
        if u.shape != (self.interface.nu,) or d.shape != (self.interface.nd,) or not np.isfinite(u).all() or not np.isfinite(d).all():
            raise ContractError("MATLAB adapter step requires finite nu/nd vectors")
        if (not np.allclose(u, self.arrays["u"][index], rtol=0, atol=1e-12) or
                not np.allclose(d, self.arrays["d"][index], rtol=0, atol=1e-12)):
            raise ContractError("Sampler control/disturbance differs from the approved native batch")
        self._state, self._output = self.arrays["next_x"][index].copy(), self.arrays["next_y"][index].copy()
        self._position += 1
        return self.state

    def describe(self):
        return {"backend": self.spec["backend"], "source_semantics": self.spec["source_semantics"],
                "integration": self.spec.get("integration"), "source_file": self.spec["source_file"],
                "dt": self.interface.sample_time, "reset_strategy": "approved_native_batch_trajectory_x0",
                "stop_detection": self.result.get("stop_detection"),
                "hardware_execution": False,
                "matlab_batch": {key: self.result.get(key) for key in
                    ("status", "request_sha256", "plan_hash", "model_version", "material_version", "failure",
                     "trajectories", "matlab_release", "matlab_pid", "session_id", "helper_file")}}

    def close(self):
        """No MATLAB process/session ownership is transferred to this replay."""
        self._state = self._output = None
        self._closed = True


def load_matlab_batch_adapter(spec: dict, plan: dict, *, result_path: str | Path, result_sha256: str,
                              request_sha256: str, nonce: str, model_version: dict,
                              material_version: dict) -> MatlabBatchAdapter:
    """Private finalize API; callers obtain these values only from Host env."""
    spec = validate_matlab_spec(spec)
    path = Path(result_path)
    if not path.is_absolute() or path.name != "result.json" or not _SHA.fullmatch(result_sha256 or ""):
        raise ContractError("Native MATLAB result does not match the Host-observed file hash")
    if path.stat().st_size > _MAX_BYTES:
        raise ContractError("Native MATLAB result exceeds the exchange byte budget")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != result_sha256:
        raise ContractError("Native MATLAB result does not match the Host-observed file hash")
    result = json.loads(raw.decode("utf-8"))
    expected = {"schema_version": 1, "kind": "control-matlab-batch-result-v1", "nonce": nonce,
                "request_sha256": request_sha256, "plan_hash": plan["plan_hash"],
                "model_version": _version(model_version, "model"), "material_version": _version(material_version, "material"),
                "spec_hash": canonical_hash(spec)}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ContractError("Native MATLAB result is not bound to this request/model/material/plan")
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if (any(manifest.get(key) != value for key, value in expected.items() if key not in {"kind", "plan_hash"}) or
            manifest.get("kind") != "control-matlab-batch-v1" or manifest.get("spec") != spec or
            manifest.get("plan") != _json(plan, "sampling plan") or Path(manifest.get("result_path", "")).resolve() != path.resolve()):
        raise ContractError("Native MATLAB finalize exchange no longer matches its prepared manifest")
    source_root = (path.parent / "input").resolve()
    if Path(manifest.get("source_root", "")).resolve() != source_root:
        raise ContractError("Native MATLAB input copy escaped its fixed exchange")
    inventory = manifest.get("source_inventory")
    if (not isinstance(inventory, dict) or spec["source_file"] not in inventory or
            manifest.get("source_sha256") != inventory[spec["source_file"]] or
            Path(manifest.get("source_file", "")).resolve() != (source_root / spec["source_file"]).resolve()):
        raise ContractError("Native MATLAB exchange has no bound source-file inventory")
    for relative, digest in inventory.items():
        relative = _relative(relative)
        source = (source_root / relative).resolve(strict=True)
        if not source.is_relative_to(source_root) or _sha(source) != digest:
            raise ContractError("Native MATLAB source copy changed during execution")
    return MatlabBatchAdapter(spec, plan, result)
