"""Versioned numerical interfaces shared by all three domain stages."""
from __future__ import annotations

import hashlib
import json
import math
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
