"""Versioned numerical interfaces shared by all three domain stages."""
from __future__ import annotations

import hashlib
import json
import math
import copy
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1


class ContractError(ValueError):
    """An input or artifact cannot satisfy the declared interface."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Variable:
    name: str
    unit: str
    lower: float
    upper: float

    @classmethod
    def parse(cls, item: dict) -> "Variable":
        try:
            result = cls(str(item["name"]), str(item["unit"]),
                         float(item["lower"]), float(item["upper"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("Variables require name, unit, finite lower/upper bounds") from exc
        if not result.name.strip() or not result.unit.strip():
            raise ContractError("Variable names and units must be explicit")
        if not all(map(math.isfinite, (result.lower, result.upper))) or result.lower >= result.upper:
            raise ContractError(f"Invalid bounds for {result.name}")
        return result


@dataclass(frozen=True)
class ModelInterface:
    name: str
    sample_time: float
    states: tuple[Variable, ...]
    controls: tuple[Variable, ...]
    outputs: tuple[Variable, ...]
    disturbances: tuple[Variable, ...] = ()
    time_semantics: str = "discrete_state_transition"
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def parse(cls, raw: dict) -> "ModelInterface":
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ContractError("Unsupported interface schema_version")
        if raw.get("time_semantics") != "discrete_state_transition":
            raise ContractError("Explicit discrete_state_transition required; derivatives need an adapter")
        try:
            groups = {k: tuple(Variable.parse(v) for v in raw.get(k, []))
                      for k in ("states", "controls", "outputs", "disturbances")}
            result = cls(name=str(raw["name"]), sample_time=float(raw["sample_time"]), **groups)
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"Invalid interface: {exc}") from exc
        if not math.isfinite(result.sample_time) or result.sample_time <= 0:
            raise ContractError("sample_time must be finite and positive")
        if not all(groups[k] for k in ("states", "controls", "outputs")):
            raise ContractError("At least one state, control and output are required")
        for key, values in groups.items():
            if len({v.name for v in values}) != len(values):
                raise ContractError(f"Duplicate variable names in {key}")
        return result

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        # Name is descriptive; ordering, units, bounds and timing are functional.
        return canonical_hash({k: v for k, v in self.to_dict().items() if k != "name"})

    @property
    def nx(self) -> int:
        return len(self.states)

    @property
    def nu(self) -> int:
        return len(self.controls)

    @property
    def ny(self) -> int:
        return len(self.outputs)

    @property
    def nd(self) -> int:
        return len(self.disturbances)

    def compatible(self, other: "ModelInterface") -> None:
        if self.fingerprint != other.fingerprint:
            raise ContractError("Interface mismatch: variable order, dimensions, units, bounds or sample time differ")


def validate_network_dimensions(interface: ModelInterface, config: dict, *, residual: bool) -> tuple[int, int]:
    expected = ((interface.nx + interface.nu + interface.nd, interface.nx) if residual
                else (interface.nx + interface.ny + interface.nd, interface.nu))
    for name, value in zip(("input_dim", "output_dim"), expected):
        if name in config and config[name] != value:
            raise ContractError(f"{name}={config[name]} conflicts with model metadata; expected {value}")
    return expected


@dataclass(frozen=True)
class MaterialInterface:
    """Source bytes have a format, not inferred plant state dimensions."""
    format: str
    name: str = "material"

    @classmethod
    def parse(cls, raw: dict) -> "MaterialInterface":
        if raw.get("schema_version") != 1 or raw.get("interface_kind") != "material":
            raise ContractError("Explicit version-1 material interface required")
        if raw.get("format") not in {"csv", "xlsx", "py", "m", "slx", "mdl", "json", "bundle"}:
            raise ContractError("Unsupported material format")
        return cls(raw["format"], str(raw.get("name", "material")))

    def to_dict(self) -> dict:
        return {"schema_version": 1, "interface_kind": "material", "format": self.format, "name": self.name}

    @property
    def fingerprint(self) -> str:
        return canonical_hash({key: value for key, value in self.to_dict().items() if key != "name"})

    def compatible(self, other) -> None:
        if not isinstance(other, MaterialInterface) or self.fingerprint != other.fingerprint:
            raise ContractError("Material interface mismatch")


@dataclass(frozen=True)
class DataInterface:
    """Causal supervised features/targets, independently of physical nx/nu/ny."""
    definition_json: str

    @classmethod
    def parse(cls, raw: dict) -> "DataInterface":
        if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("interface_kind", raw.get("kind")) != "tabular_supervised":
            raise ContractError("Explicit version-1 tabular_supervised interface required")
        if any(key in raw for key in ("nx", "nu", "ny", "nd", "states")):
            raise ContractError("Supervised feature widths do not define plant nx/nu/ny/nd")
        value = copy.deepcopy(raw)
        value.pop("kind", None)
        value["interface_kind"] = "tabular_supervised"
        if value.get("task") not in {"static_regression", "dynamic_prediction"}:
            raise ContractError("Declare static_regression or dynamic_prediction")
        dt = value.get("sample_time")
        if dt is not None and (isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0):
            raise ContractError("sample_time must be finite and positive when supplied")

        def integer(item, label, minimum=0):
            if type(item) is not int or item < minimum:
                raise ContractError(f"{label} must be an integer >= {minimum}")
            return item

        def text(item, label):
            if not isinstance(item, str) or not item.strip():
                raise ContractError(f"{label} must be explicit nonempty text")

        def column(item):
            if not isinstance(item, dict) or "header" not in item or "worksheet" not in item:
                raise ContractError("Column mapping requires index, original header and worksheet (CSV:null)")
            integer(item.get("index"), "column.index")
            if item["header"] is not None and not isinstance(item["header"], str):
                raise ContractError("Original column header must be a string or null")
            if item["worksheet"] is not None:
                text(item["worksheet"], "worksheet")
            return item["worksheet"], item["index"]

        signals = value.get("signals")
        if not isinstance(signals, (list, tuple)) or not signals:
            raise ContractError("Explicit signal definitions required")
        by_name, positions = {}, set()
        for signal in signals:
            if not isinstance(signal, dict):
                raise ContractError("Signal must be an object")
            text(signal.get("name"), "signal.name")
            text(signal.get("unit"), "signal.unit")
            if signal.get("role") not in {"state", "control", "output", "disturbance", "auxiliary"}:
                raise ContractError("Signal role is unknown")
            if type(signal.get("observable")) is not bool:
                raise ContractError("Signal observability must be explicit")
            integer(signal.get("delay_steps"), "signal availability delay")
            position = column(signal.get("column"))
            if signal["name"] in by_name or position in positions:
                raise ContractError("Signal name or source column is duplicated")
            by_name[signal["name"]] = signal
            positions.add(position)
        for key in ("time_column", "trip_column"):
            if value.get(key) is not None:
                column(value[key])
        mapping = value.get("state_mapping")
        mapping = {} if mapping is None else mapping
        if not isinstance(mapping, dict) or any(not isinstance(item, str) for item in mapping.values()) or len(set(mapping.values())) != len(mapping):
            raise ContractError("state_mapping must be an unambiguous object")
        for state, signal in mapping.items():
            text(state, "plant state mapping name")
            if signal not in by_name or by_name[signal]["role"] != "state":
                raise ContractError("state_mapping must refer to declared state-role signals")
        features, targets = value.get("features"), value.get("targets")
        if not isinstance(features, (list, tuple)) or not features or not isinstance(targets, (list, tuple)) or not targets:
            raise ContractError("Explicit nonempty features and targets required")
        feature_times, target_times = set(), set()
        for feature in features:
            if not isinstance(feature, dict) or feature.get("signal") not in by_name:
                raise ContractError("Feature references an unknown signal")
            signal = by_name[feature["signal"]]
            if not signal["observable"]:
                raise ContractError("Unobservable signals cannot be inference features")
            width = integer(feature.get("window"), "feature.window", 1)
            if width > 65536 or len(feature_times) + width > 65536:
                raise ContractError("Feature width exceeds the 65536-element interface budget")
            lag = integer(feature.get("lag"), "feature.lag") + signal["delay_steps"]
            for offset in range(width):
                key = signal["name"], -(lag + offset)
                if key in feature_times:
                    raise ContractError("Feature definitions overlap at the same signal/time")
                feature_times.add(key)
        for target in targets:
            if not isinstance(target, dict) or target.get("signal") not in by_name:
                raise ContractError("Target references an unknown signal")
            key = target["signal"], integer(target.get("offset"), "target.offset")
            if key in target_times or key in feature_times:
                raise ContractError("Target duplicates a target or leaks directly into features")
            target_times.add(key)
        temporal = any(time != 0 for _, time in feature_times | target_times)
        if value["task"] == "static_regression" and temporal:
            raise ContractError("Static regression requires same-row features and targets without history/delay")
        if value["task"] == "dynamic_prediction" and (value.get("time_column") is None or dt is None):
            raise ContractError("Dynamic data requires an explicit time column and sample_time")
        if value.get("time_column") is None and temporal:
            raise ContractError("History/prediction offsets require an explicit time column")
        dims = (len(feature_times), len(target_times))
        for key, expected in zip(("input_dim", "output_dim"), dims):
            if key in value and (type(value[key]) is not int or value[key] != expected):
                raise ContractError(f"{key} conflicts with declared feature/target width; expected {expected}")
            value[key] = expected
        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ContractError("Data interface must contain finite JSON values") from error
        return cls(encoded)

    def to_dict(self) -> dict:
        return json.loads(self.definition_json)

    @property
    def fingerprint(self) -> str:
        return canonical_hash({key: value for key, value in self.to_dict().items() if key != "name"})

    @property
    def input_dim(self) -> int:
        return self.to_dict()["input_dim"]

    @property
    def output_dim(self) -> int:
        return self.to_dict()["output_dim"]

    def compatible(self, other) -> None:
        if not isinstance(other, DataInterface) or self.fingerprint != other.fingerprint:
            raise ContractError("Supervised interface mismatch: signal semantics/order/units/timing/features differ")


def parse_interface(raw: dict):
    if not isinstance(raw, dict):
        raise ContractError("Artifact interface must be a JSON object")
    kind = raw.get("interface_kind")
    if kind == "material":
        return MaterialInterface.parse(raw)
    if kind == "tabular_supervised":
        return DataInterface.parse(raw)
    if kind is not None:
        raise ContractError(f"Unsupported artifact interface_kind: {kind}")
    return ModelInterface.parse(raw)
