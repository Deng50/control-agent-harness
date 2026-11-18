"""Explicit NumPy plant adapters over immutable material snapshots.

Validation never imports user source. Loading is source execution and belongs
only to an authorized trial/sampling operation. This is a supported interface,
not a Python security sandbox or an arbitrary-project parser. Registered Torch
artifacts continue to use their existing load_model path.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import sys
import types
import uuid

import numpy as np

from .contracts import ContractError, ModelInterface

_BACKENDS = {"python_callable", "python_factory"}
_IMPORTS = {"numpy", "scipy", "math", "typing", "collections", "dataclasses", "functools",
            "itertools", "operator", "statistics", "copy", "enum", "__future__"}
_DYNAMIC = {"open", "eval", "exec", "compile", "__import__"}
_IO = {"load", "loadtxt", "fromfile", "save", "savez", "savez_compressed", "read_text", "read_bytes", "write_text", "write_bytes"}


def _identifier(value, label):
    if not isinstance(value, str) or not value.isidentifier() or value.startswith("__"):
        raise ContractError(f"{label} must name one explicit non-dunder Python function or method")


def _has_bool(value):
    return isinstance(value, (bool, np.bool_)) or (isinstance(value, (list, tuple)) and any(_has_bool(item) for item in value))


def validate_adapter_spec(spec: dict) -> dict:
    """Return a JSON-only normalized spec, without reading/importing its source."""
    if not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or spec.get("schema_version") != 1 or spec.get("backend") not in _BACKENDS:
        raise ContractError("Explicit version-1 python_callable or python_factory adapter required")
    allowed = {"schema_version", "backend", "source_file", "interface", "parameters", "source_semantics",
               "entrypoint", "factory", "reset_strategy", "C", "output_entrypoint", "integration",
               "time_unit", "derivative_units", "D", "feedthrough", "output_depends_on_control"}
    if set(spec) - allowed:
        raise ContractError("Unknown Python adapter field; use the declared thin protocol rather than an inferred project interface")
    try:
        value = json.loads(json.dumps(spec, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ContractError("Adapter declarations and parameters must contain finite JSON values") from error
    source = value.get("source_file")
    if not isinstance(source, str) or not source or PureWindowsPath(source).drive:
        raise ContractError("source_file must be a relative Python file in the selected material snapshot")
    source = source.replace("\\", "/")
    path = PurePosixPath(source)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ContractError("source_file must remain inside the selected material snapshot")
    value["source_file"] = path.as_posix()
    raw_interface = value.get("interface")
    if (not isinstance(raw_interface, dict) or type(raw_interface.get("schema_version")) is not int
            or type(raw_interface.get("sample_time")) not in {int, float}):
        raise ContractError("A complete discrete runtime ModelInterface and numeric sample_time are required")
    for group in ("states", "controls", "outputs", "disturbances"):
        variables = raw_interface.get(group, [])
        if not isinstance(variables, list): raise ContractError("Ordered interface variable groups must be explicit lists")
        for variable in variables:
            if (not isinstance(variable, dict) or not isinstance(variable.get("name"), str)
                    or not isinstance(variable.get("unit"), str)
                    or any(type(variable.get(key)) not in {int, float} for key in ("lower", "upper"))):
                raise ContractError("Each ordered variable requires explicit text name/unit and numeric bounds")
    interface = ModelInterface.parse(raw_interface)
    if not isinstance(value.get("parameters", {}), dict):
        raise ContractError("parameters must be a JSON object")
    value.setdefault("parameters", {})
    semantics = value.get("source_semantics")
    if semantics not in {"transition", "derivative"}:
        raise ContractError("Declare source_semantics=transition or derivative; derivatives are never next states implicitly")
    if value["backend"] == "python_callable":
        _identifier(value.get("entrypoint"), "entrypoint")
        if "factory" in value or "reset_strategy" in value:
            raise ContractError("A callable uses explicit x, not a factory reset strategy")
    else:
        _identifier(value.get("factory"), "factory")
        if "entrypoint" in value or value.get("reset_strategy") not in {"recreate", "reset"}:
            raise ContractError("A factory requires reset_strategy=recreate or reset and no callable entrypoint")
        if semantics != "transition":
            raise ContractError("Stateful factory derivatives are not supported; expose a pure callable derivative adapter")
    if ("C" in value) == ("output_entrypoint" in value):
        raise ContractError("Declare exactly one state-only output: C or output_entrypoint; direct feedthrough is not inferred")
    if "C" in value:
        try:
            raw = np.asarray(value["C"])
            matrix = np.asarray(value["C"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ContractError("C must be a finite numeric [ny,nx] matrix") from error
        if _has_bool(value["C"]) or raw.dtype.kind not in "fiu" or matrix.shape != (interface.ny, interface.nx) or not np.isfinite(matrix).all():
            raise ContractError("C must be a finite numeric [ny,nx] matrix")
    else:
        _identifier(value["output_entrypoint"], "output_entrypoint")
    if any(key in value for key in ("D", "feedthrough", "output_depends_on_control")):
        raise ContractError("This adapter supports declared state-only outputs; provide an explicit supported wrapper for feedthrough")
    if semantics == "transition":
        if any(key in value for key in ("integration", "time_unit", "derivative_units")):
            raise ContractError("Transition sources cannot carry derivative/integration declarations")
    else:
        if value.get("time_unit") != "s":
            raise ContractError("Derivative integration requires explicit time_unit='s'")
        declared = value.get("derivative_units")
        if not isinstance(declared, list) or len(declared) != interface.nx:
            raise ContractError("derivative_units must declare one unit per ordered state")
        for state, unit in zip(interface.states, declared):
            expected = {f"{state.unit.replace(' ', '')}/s", f"({state.unit.replace(' ', '')})/s"}
            if state.unit.replace(" ", "") == "1": expected.add("s^-1")
            if not isinstance(unit, str) or unit.replace(" ", "") not in expected:
                raise ContractError(f"Derivative unit for {state.name} must explicitly be ({state.unit})/s; no unit conversion is inferred")
        integration = value.get("integration")
        if (not isinstance(integration, dict) or set(integration) != {"method", "substeps", "input_hold"}
                or integration["method"] not in {"euler", "rk4"}
                or type(integration["substeps"]) is not int or not 1 <= integration["substeps"] <= 1024
                or integration["input_hold"] != "zero_order_hold"):
            raise ContractError("Derivative integration requires explicit Euler/RK4, 1..1024 substeps and zero_order_hold")
    return value


def _vector(value, size: int, label: str, *, source=False) -> np.ndarray:
    if _has_bool(value):
        raise ContractError(f"{label} cannot contain boolean values")
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu" or (source and isinstance(value, np.ndarray) and raw.dtype != np.float64):
            raise ContractError(f"{label} must use real float64 values, not bool/object/complex or lower-precision source arrays")
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ContractError(f"{label} must be a float64 vector with shape ({size},)") from error
    if result.shape != (size,):
        raise ContractError(f"{label} must have exact shape ({size},), got {result.shape}; broadcasting is not supported")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{label} contains non-finite values")
    return result.copy()


class _CodeStore:
    """Private package loader: only explicit relative snapshot dependencies and math imports."""
    def __init__(self, root, source_file, expected_file_hashes=None):
        self.root = Path(root).resolve()
        self.expected_file_hashes = None
        if expected_file_hashes is not None:
            if not isinstance(expected_file_hashes, dict):
                raise ContractError("Expected material file hashes must be a relative-path SHA256 map")
            self.expected_file_hashes = {}
            for relative, digest in expected_file_hashes.items():
                if not isinstance(relative, str) or not relative or PureWindowsPath(relative).drive:
                    raise ContractError("Expected material file hashes require confined relative paths")
                path = PurePosixPath(relative.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ContractError("Expected material file hashes require confined relative paths")
                if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
                    raise ContractError("Expected material file hashes require SHA256 digests")
                key = path.as_posix()
                if key in self.expected_file_hashes:
                    raise ContractError("Expected material file hashes contain duplicate normalized paths")
                self.expected_file_hashes[key] = digest.lower()
        self.entry = self.checked(self.root / source_file, file=True)
        self.package_root = self.entry.parent
        self.cache = {}

    def checked(self, path, *, file=False):
        path = Path(path)
        if not path.is_relative_to(self.root): raise ContractError("Python dependency escapes its material snapshot")
        for part in [self.root, *reversed([p for p in path.parents if p != self.root and p.is_relative_to(self.root)]), path]:
            try: info = part.lstat()
            except OSError as error: raise ContractError("Python snapshot dependency is missing") from error
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ContractError("Python snapshot dependencies cannot follow links/reparse points")
        if file and not path.is_file(): raise ContractError("Python snapshot source must be a regular file")
        if not path.resolve().is_relative_to(self.root): raise ContractError("Python dependency resolves outside its snapshot")
        return path

    def code(self, path):
        path = self.checked(path, file=True)
        if path not in self.cache:
            payload = path.read_bytes()
            # Hash the same immutable bytes that are parsed and executed.  This
            # check also covers lazy relative imports, before their first code
            # runs, rather than relying on a later artifact publish check.
            if self.expected_file_hashes is not None:
                relative = path.relative_to(self.root).as_posix()
                expected = self.expected_file_hashes.get(relative)
                if expected is None:
                    raise ContractError(f"Python source is outside the bound material file closure: {relative}")
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ContractError(f"Python material source hash changed before execution: {relative}")
            if not payload or len(payload) > 1024 * 1024 or len(self.cache) >= 256 or sum(len(row[0]) for row in self.cache.values()) + len(payload) > 8 * 1024 * 1024:
                raise ContractError("Python source/dependency budget exceeded (1 MiB/file, 8 MiB/256 modules total)")
            try: tree = ast.parse(payload, filename=str(path))
            except (SyntaxError, UnicodeError) as error: raise ContractError("Python snapshot source cannot be parsed") from error
            if sum(1 for _ in ast.walk(tree)) > 50000: raise ContractError("Python snapshot AST budget exceeded")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): roots = [alias.name.split('.')[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level: roots = [str(node.module).split('.')[0]]
                else: roots = []
                if any(root not in _IMPORTS for root in roots):
                    raise ContractError("Use declared installed math imports or explicit relative snapshot imports; arbitrary project/environment imports are unsupported")
                if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id in _DYNAMIC)
                        or (isinstance(node.func, ast.Attribute) and node.func.attr in _IO)):
                    raise ContractError("Python adapters cannot discover external files or dynamically execute source; use frozen dependencies and JSON parameters")
            self.cache[path] = (payload, compile(tree, str(path), "exec"))
        return self.cache[path][1]


class _Loader(importlib.abc.Loader):
    def __init__(self, store, path): self.store, self.path = store, path
    def create_module(self, spec): return None
    def exec_module(self, module):
        module.__file__ = str(self.path)
        exec(self.store.code(self.path), module.__dict__)


class _SnapshotRuntime(importlib.abc.MetaPathFinder):
    def __init__(self, store):
        self.store = store
        self.namespace = f"_control_numpy_{uuid.uuid4().hex}"
        package = types.ModuleType(self.namespace)
        package.__path__ = [str(store.package_root)]
        package.__package__ = self.namespace
        package.__spec__ = importlib.machinery.ModuleSpec(self.namespace, loader=None, is_package=True)
        sys.modules[self.namespace] = package
        try:
            name = f"{self.namespace}._entry"
            spec = importlib.util.spec_from_loader(name, _Loader(store, store.entry), origin=str(store.entry))
            self.module = importlib.util.module_from_spec(spec)
            sys.modules[name] = self.module
            with self.active(): spec.loader.exec_module(self.module)
        except BaseException:
            self.close()
            raise

    @contextlib.contextmanager
    def active(self):
        sys.meta_path.insert(0, self)
        try:
            with contextlib.redirect_stdout(sys.stderr): yield
        finally:
            sys.meta_path.remove(self)

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + '.'): return None
        parts = fullname[len(self.namespace) + 1:].split('.')
        relative = self.store.package_root.joinpath(*parts)
        module_file, init_file = relative.with_suffix('.py'), relative / '__init__.py'
        if module_file.is_file(): return importlib.util.spec_from_loader(fullname, _Loader(self.store, self.store.checked(module_file, file=True)), origin=str(module_file))
        if init_file.is_file():
            self.store.checked(init_file, file=True)
            spec = importlib.util.spec_from_loader(fullname, _Loader(self.store, init_file), origin=str(init_file), is_package=True)
            spec.submodule_search_locations = [str(relative)]
            return spec
        if relative.is_dir():
            self.store.checked(relative)
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(relative)]
            return spec
        raise ModuleNotFoundError("Declared relative Python dependency is missing from the selected snapshot")

    def call(self, function, *args):
        with self.active(): return function(*args)

    def function(self, name):
        with self.active(): function = getattr(self.module, name, None)
        if not callable(function): raise ContractError(f"Declared Python entrypoint {name} is not callable")
        return function

    def member(self, instance, name):
        # Properties may perform lazy relative imports too; keep them inside the
        # same snapshot loader instead of falling through to PathFinder.
        with self.active(): return getattr(instance, name, None)

    def close(self):
        for name in list(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + '.'): sys.modules.pop(name, None)


class _Adapter:
    def __init__(self, snapshot_root, spec, *, expected_file_hashes=None):
        self.spec = validate_adapter_spec(spec)
        self.interface = ModelInterface.parse(self.spec["interface"])
        self.store = _CodeStore(snapshot_root, self.spec["source_file"], expected_file_hashes)
        self.runtime = _SnapshotRuntime(self.store)
        self._x = None
        self._failed = False
        self._closed = False
        self.C = np.asarray(self.spec["C"], dtype=np.float64) if "C" in self.spec else None

    def _ready(self):
        if self._closed or self._failed or self._x is None: raise ContractError("Adapter must be reset before use; closed/failed adapters cannot continue implicitly")

    @property
    def state(self):
        self._ready()
        return self._x.copy()

    def describe(self):
        return {"backend": self.spec["backend"], "source_file": self.spec["source_file"],
            "source_semantics": self.spec["source_semantics"], "sample_time": self.interface.sample_time,
            "runtime_time_semantics": "discrete_state_transition", "boundary_dtype": "float64",
            "state_order": [v.name for v in self.interface.states], "control_order": [v.name for v in self.interface.controls],
            "output_order": [v.name for v in self.interface.outputs], "disturbance_order": [v.name for v in self.interface.disturbances],
            "integration": copy.deepcopy(self.spec.get("integration")), "reset_strategy": self.spec.get("reset_strategy", "fresh_private_module"),
            "time_unit": self.spec.get("time_unit"), "derivative_units": copy.deepcopy(self.spec.get("derivative_units")),
            "dt_argument": "declared_sample_time; integration_substep_is_internal",
            "output_semantics": "state_only", "hidden_memory": "rejected_when_it_changes_visited_state_transitions_or_outputs",
            "units": "user_declared_equation_consistency_not_symbolically_proved",
            "material_hash_verification": "every_loaded_module_before_execution" if self.store.expected_file_hashes is not None else "not_supplied_by_caller",
            "source_hashes": {str(path.relative_to(self.store.root).as_posix()): hashlib.sha256(row[0]).hexdigest() for path, row in self.store.cache.items()}}

    def _inputs(self, u, d):
        return (_vector(u, self.interface.nu, "control"),
                _vector([] if d is None and self.interface.nd == 0 else d, self.interface.nd, "disturbance"))

    def close(self): self.runtime.close(); self._closed = True
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class PythonCallableAdapter(_Adapter):
    def reset(self, x0):
        if self._closed: raise ContractError("Cannot reset a closed adapter")
        x = _vector(x0, self.interface.nx, "initial state")
        self.runtime.close()
        self.runtime = _SnapshotRuntime(self.store)
        self._x, self._failed = x, False
        return x.copy()

    def _raw(self, runtime, x, u, d):
        value = runtime.call(runtime.function(self.spec["entrypoint"]), x.copy(), u.copy(), d.copy(),
                             self.interface.sample_time, copy.deepcopy(self.spec["parameters"]))
        return _vector(value, self.interface.nx, "source state/derivative", source=True)

    def _evaluate(self, x, u, d):
        first, second = self._raw(self.runtime, x, u, d), self._raw(self.runtime, x, u, d)
        reference = _SnapshotRuntime(self.store)
        try: fresh = self._raw(reference, x, u, d)
        finally: reference.close()
        if not np.array_equal(first, second) or not np.array_equal(first, fresh):
            raise ContractError("Callable has hidden memory/nondeterministic dynamics; declare all dynamic state or provide an explicit thin adapter")
        return first

    def step(self, u, d=None):
        self._ready()
        try:
            u, d = self._inputs(u, d)
            x = self._x.copy()
            if self.spec["source_semantics"] == "transition": x = self._evaluate(x, u, d)
            else:
                integration = self.spec["integration"]
                h = self.interface.sample_time / integration["substeps"]
                for _ in range(integration["substeps"]):
                    if integration["method"] == "euler": x = x + h * self._evaluate(x, u, d)
                    else:
                        k1 = self._evaluate(x, u, d); k2 = self._evaluate(x + .5 * h * k1, u, d)
                        k3 = self._evaluate(x + .5 * h * k2, u, d); k4 = self._evaluate(x + h * k3, u, d)
                        x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    x = _vector(x, self.interface.nx, "integrated next state", source=True)
            self._x = x.copy()
            return x.copy()
        except Exception:
            self._failed = True
            raise

    def _output(self, runtime):
        if self.C is not None: return self.C @ self._x
        return _vector(runtime.call(runtime.function(self.spec["output_entrypoint"]), self._x.copy(), copy.deepcopy(self.spec["parameters"])), self.interface.ny, "source output", source=True)

    def output(self):
        self._ready()
        try:
            first, second = self._output(self.runtime), self._output(self.runtime)
            reference = _SnapshotRuntime(self.store)
            try: fresh = self._output(reference)
            finally: reference.close()
            if not np.array_equal(first, second) or not np.array_equal(first, fresh): raise ContractError("Output has hidden memory; outputs must be state-only")
            return _vector(first, self.interface.ny, "output", source=True)
        except Exception:
            self._failed = True
            raise


class PythonFactoryAdapter(_Adapter):
    def __init__(self, snapshot_root, spec, *, expected_file_hashes=None):
        super().__init__(snapshot_root, spec, expected_file_hashes=expected_file_hashes)
        self.instance = None

    def _new(self, runtime):
        instance = runtime.call(runtime.function(self.spec["factory"]), copy.deepcopy(self.spec["parameters"]))
        if not callable(runtime.member(instance, "reset")) or not callable(runtime.member(instance, "step")):
            raise ContractError("Factory must expose reset(x0), state and step(u,d)")
        declared_dt = runtime.member(instance, "sample_time")
        if declared_dt is not None and (isinstance(declared_dt, (bool, np.bool_)) or declared_dt != self.interface.sample_time):
            raise ContractError("Factory sample_time differs from the declared interface")
        return instance

    def _state(self, instance, runtime=None): return _vector((runtime or self.runtime).member(instance, "state"), self.interface.nx, "factory state", source=True)

    def _reset(self, runtime, instance, x):
        returned = runtime.call(runtime.member(instance, "reset"), x.copy())
        if not np.array_equal(self._state(instance, runtime), x) or (returned is not None and not np.array_equal(_vector(returned, self.interface.nx, "factory reset result", source=True), x)):
            raise ContractError("Factory reset must restore the exact declared initial state")

    def reset(self, x0):
        if self._closed: raise ContractError("Cannot reset a closed adapter")
        x = _vector(x0, self.interface.nx, "initial state")
        try:
            if self.instance is None or self.spec["reset_strategy"] == "recreate":
                self.runtime.close(); self.runtime = _SnapshotRuntime(self.store)
                self.instance = self._new(self.runtime)
            self._reset(self.runtime, self.instance, x)
            self._x, self._failed = x, False
            return x.copy()
        except Exception:
            self._failed = True
            raise

    def _advance(self, runtime, instance, u, d):
        result = _vector(runtime.call(runtime.member(instance, "step"), u.copy(), d.copy()), self.interface.nx, "factory next state", source=True)
        if not np.array_equal(result, self._state(instance, runtime)): raise ContractError("Factory step result differs from its declared state")
        return result

    def step(self, u, d=None):
        self._ready()
        try:
            u, d = self._inputs(u, d)
            if not np.array_equal(self._state(self.instance), self._x): raise ContractError("Factory state changed outside a declared step")
            value = self._advance(self.runtime, self.instance, u, d)
            reference = _SnapshotRuntime(self.store)
            try:
                fresh_instance = self._new(reference); self._reset(reference, fresh_instance, self._x)
                fresh = self._advance(reference, fresh_instance, u, d)
            finally: reference.close()
            if not np.array_equal(value, fresh): raise ContractError("Factory hidden memory affects dynamics; include it in explicit state or provide a thin adapter")
            self._x = value.copy()
            return value.copy()
        except Exception:
            self._failed = True
            raise

    def _output(self, runtime, instance):
        if self.C is not None: return self.C @ self._x
        function = runtime.member(instance, self.spec["output_entrypoint"])
        if not callable(function): raise ContractError("Factory output_entrypoint must be a state-only no-argument method")
        value = _vector(runtime.call(function), self.interface.ny, "factory output", source=True)
        if not np.array_equal(self._state(instance, runtime), self._x): raise ContractError("Factory output must not mutate state")
        return value

    def output(self):
        self._ready()
        try:
            first, second = self._output(self.runtime, self.instance), self._output(self.runtime, self.instance)
            reference = _SnapshotRuntime(self.store)
            try:
                instance = self._new(reference); self._reset(reference, instance, self._x)
                fresh = self._output(reference, instance)
            finally: reference.close()
            if not np.array_equal(first, second) or not np.array_equal(first, fresh): raise ContractError("Factory hidden memory affects output; outputs must depend only on declared state")
            return _vector(first, self.interface.ny, "output", source=True)
        except Exception:
            self._failed = True
            raise


def load_python_adapter(snapshot_root, spec: dict, *, expected_file_hashes=None):
    """Execute an authorized snapshot, optionally enforcing its bound file hashes.

    Production material callers must supply artifact metadata ``files``.  None
    preserves the standalone numerical adapter API; an empty map permits no
    source.  All actually loaded modules, including lazy helpers, must belong
    to the supplied closure and match before parsing or execution.
    """
    normalized = validate_adapter_spec(spec)
    implementation = PythonCallableAdapter if normalized["backend"] == "python_callable" else PythonFactoryAdapter
    return implementation(snapshot_root, normalized, expected_file_hashes=expected_file_hashes)
