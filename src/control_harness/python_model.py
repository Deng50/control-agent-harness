"""Explicitly executed, frozen standalone PyTorch dynamics for native Harness workers.

Static inspection calls only the declaration readers. Execution happens only in
the registration operation or when consuming an already selected model artifact.
The host's normal subprocess permissions remain the execution boundary.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import random
import sys
import tokenize
import types
import uuid
from pathlib import Path

import numpy as np
import scipy
import torch
from torch import nn

from .artifacts import Workspace, atomic_json, read_json
from .contracts import ContractError, ModelInterface, canonical_hash
from .continuous import continuous_contract, integrate_polynomial

REPRESENTATIONS = {"python_torch_transition", "python_torch_derivative"}
MAX_SOURCE_BYTES = 1024 * 1024
ALLOWED_IMPORTS = {"torch", "numpy", "scipy", "math", "typing", "collections", "dataclasses",
                   "functools", "itertools", "operator", "statistics", "__future__"}
UNSUPPORTED_CALLS = {"open", "eval", "exec", "compile", "__import__"}
UNSUPPORTED_IO = {"load", "loadtxt", "fromfile", "save", "savez", "savez_compressed", "read_text",
                  "read_bytes", "write_text", "write_bytes", "getenv"}


def read_source(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read(MAX_SOURCE_BYTES + 1)


def runtime_dependencies():
    return {"python": list(sys.version_info[:3]), "torch": str(torch.__version__),
            "numpy": str(np.__version__), "scipy": str(scipy.__version__), "tensor_boundary": "cpu_float64"}


def source_tree(payload: bytes) -> ast.Module:
    if not payload or len(payload) > MAX_SOURCE_BYTES:
        raise ContractError("Standalone Python dynamics source must be nonempty and at most 1 MiB")
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(payload).readline)
        tree = ast.parse(payload.decode(encoding))
    except (SyntaxError, UnicodeError, LookupError) as exc:
        raise ContractError(f"Python dynamics source cannot be parsed: {exc}") from exc
    if sum(1 for _ in ast.walk(tree)) > 50000:
        raise ContractError("Python dynamics source exceeds the AST budget")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = [alias.name.split('.')[0] for alias in node.names] if isinstance(node, ast.Import) else [str(node.module).split('.')[0]]
            if (isinstance(node, ast.ImportFrom) and node.level) or any(root not in ALLOWED_IMPORTS for root in roots):
                raise ContractError("Standalone dynamics permits declared installed math/Torch imports only; bundle external project dependencies explicitly")
        if isinstance(node, ast.Call):
            if ((isinstance(node.func, ast.Name) and node.func.id in UNSUPPORTED_CALLS)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr in UNSUPPORTED_IO)):
                raise ContractError("Standalone dynamics cannot read external resources or dynamically execute source; provide fixed JSON parameters")
    return tree


def python_spec(payload: bytes, override: dict | None = None) -> dict:
    """Read a literal MODEL_SPEC or an explicit override, without executing Python."""
    tree = source_tree(payload)
    if override is None:
        declarations = []
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(target, ast.Name) and target.id == "MODEL_SPEC" for target in targets):
                try:
                    declarations.append(ast.literal_eval(node.value))
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise ContractError("MODEL_SPEC must be a finite literal declaration or an explicit reviewed model_spec override") from exc
        if len(declarations) != 1:
            raise ContractError("Provide exactly one literal MODEL_SPEC or an explicit reviewed model_spec override")
        override = declarations[0]
    if not isinstance(override, dict):
        raise ContractError("Python model_spec must be a JSON object")
    spec = copy.deepcopy(override)
    try:
        canonical_hash(spec)
    except (ValueError, TypeError) as exc:
        raise ContractError("Python model specification and parameters must contain finite JSON values") from exc
    spec = json.loads(json.dumps(spec, allow_nan=False))
    validate_python_spec(spec)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for field in ("entrypoint", "output_entrypoint"):
        name = spec.get(field)
        if name is None:
            continue
        if name not in functions:
            raise ContractError(f"{field} must name a top-level synchronous function in this source")
        function = functions[name]
        expected = 4 if field == "entrypoint" else 2
        if (len(function.args.posonlyargs) + len(function.args.args) != expected
                or function.args.vararg or function.args.kwarg or function.args.kwonlyargs):
            raise ContractError(f"{field} requires exactly {expected} positional arguments")
    return spec


def validate_python_spec(spec: dict) -> tuple[ModelInterface, dict | None]:
    """Validate only declarations; no source import, function call or Jacobian."""
    representation = spec.get("representation")
    if representation not in REPRESENTATIONS:
        raise ContractError("Python dynamics requires python_torch_transition or python_torch_derivative")
    if not isinstance(spec.get("entrypoint"), str) or not spec["entrypoint"].isidentifier():
        raise ContractError("Python entrypoint must be an explicit function identifier")
    if not isinstance(spec.get("parameters", {}), dict):
        raise ContractError("Python dynamics parameters must be a JSON object")
    canonical_hash(spec.get("parameters", {}))
    if representation == "python_torch_derivative":
        interface, adapter = continuous_contract(spec)
        if adapter["method"] not in {"euler", "rk4"}:
            raise ContractError("Python derivative adapters support explicit Euler/RK4; exact ZOH needs declared affine matrices")
        adapter["source_representation"] = representation
    else:
        interface, adapter = ModelInterface.parse(spec["interface"]), None
        if any(key in spec for key in ("integration", "derivative_units", "time_unit")):
            raise ContractError("Python transition functions already return next_state and cannot carry integration settings")
    for group in ("states", "controls", "outputs", "disturbances"):
        for variable in spec["interface"].get(group, []):
            if any(not isinstance(variable.get(field), str) or not variable[field].strip() for field in ("name", "unit")):
                raise ContractError("Python model variable names and units must be explicit strings")
    output_name = spec.get("output_entrypoint")
    if output_name is not None and (not isinstance(output_name, str) or not output_name.isidentifier()):
        raise ContractError("output_entrypoint must be a function identifier")
    if ("C" in spec) == (output_name is not None):
        raise ContractError("Declare exactly one output map: C or output_entrypoint")
    if "C" in spec:
        C = np.asarray(spec["C"], dtype=float)
        if C.shape != (interface.ny, interface.nx) or not np.isfinite(C).all():
            raise ContractError("Python model C must be finite with shape [ny,nx]")
    return interface, adapter


@contextlib.contextmanager
def preserve_random_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state, dtype = torch.get_rng_state(), torch.get_default_dtype()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_default_dtype(dtype)


class PythonTensorDynamics(nn.Module):
    def __init__(self, spec: dict, source_path: Path):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.representation = spec["representation"]
        self.interface, self.transition_adapter = validate_python_spec(spec)
        payload = read_source(Path(source_path))
        if hashlib.sha256(payload).hexdigest() != spec.get("source_sha256"):
            raise ContractError("Python model source differs from its frozen source hash")
        if spec.get("runtime_dependencies") != runtime_dependencies():
            raise ContractError("Python model runtime dependencies differ from the verified registration environment")
        python_spec(payload, spec)
        module_name = f"_control_dynamics_{uuid.uuid4().hex}"
        module = types.ModuleType(module_name)
        module.__file__ = str(source_path)
        # This is explicit source execution, not a claim of a Python security sandbox.
        # Redirect model prints away from the worker's NDJSON stdout protocol.
        with preserve_random_state(), contextlib.redirect_stdout(sys.stderr):
            sys.modules[module_name] = module
            try:
                exec(compile(payload, str(source_path), "exec"), module.__dict__)
            finally:
                sys.modules.pop(module_name, None)
        for value in module.__dict__.values():
            if isinstance(value, nn.Module) and any(parameter.requires_grad for parameter in value.parameters()):
                raise ContractError("Pretrained modules in physical dynamics must explicitly freeze their parameters before registration")
        self._function = getattr(module, spec["entrypoint"])
        self._output_function = getattr(module, spec["output_entrypoint"]) if spec.get("output_entrypoint") else None
        self.fixed_parameters = copy.deepcopy(spec.get("parameters", {}))
        if "C" in spec:
            self.register_buffer("C", torch.as_tensor(spec["C"], dtype=torch.float64))

    def _checked_call(self, function, args, width):
        before = [value.detach().clone() for value in args]
        with contextlib.redirect_stdout(sys.stderr):
            result = function(*args, copy.deepcopy(self.fixed_parameters))
        if any(not torch.equal(old, value.detach()) for old, value in zip(before, args)):
            raise ContractError("Python model modified an input tensor; pure dynamics functions are required")
        expected = (*args[0].shape[:-1], width)
        if (not isinstance(result, torch.Tensor) or result.dtype != torch.float64
                or result.device.type != "cpu" or tuple(result.shape) != expected):
            raise ContractError(f"Python model must return a CPU float64 Tensor with shape {expected}")
        if not torch.isfinite(result).all():
            raise ContractError("Python model returned non-finite values")
        return result

    def derivative(self, x, u, d=None):
        if self.transition_adapter is None:
            raise ContractError("A Python transition model does not expose a continuous derivative")
        return self._checked_call(self._function, (x, u, self._disturbances(x, d)), self.interface.nx)

    def _disturbances(self, x, d):
        if d is None:
            if self.interface.nd:
                raise ContractError("Disturbances must be provided explicitly")
            d = x.new_empty((*x.shape[:-1], 0))
        return d

    def forward(self, x, u, d=None):
        d = self._disturbances(x, d)
        if self.transition_adapter is not None:
            return integrate_polynomial(self.derivative, x, u, d, self.transition_adapter)
        return self._checked_call(self._function, (x, u, d), self.interface.nx)

    def output(self, x):
        return self._checked_call(self._output_function, (x,), self.interface.ny) if self._output_function else x @ self.C.T

    def step(self, x, u, d=None):
        with torch.no_grad():
            return self(torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(u, dtype=torch.float64),
                        None if d is None else torch.as_tensor(d, dtype=torch.float64)).detach().numpy()


def validate_python_numerics(model: PythonTensorDynamics) -> dict:
    i = model.interface
    variables = (*i.states, *i.controls, *i.disturbances)
    lower, upper = np.array([v.lower for v in variables]), np.array([v.upper for v in variables])
    probes = []
    def transition(z):
        return model(z[..., :i.nx], z[..., i.nx:i.nx+i.nu], z[..., i.nx+i.nu:])
    with preserve_random_state():
        for fraction in (.37, .5, .63):
            point = lower + (upper-lower)*fraction
            z = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            predicted = transition(z)
            repeated = transition(z.clone())
            if not torch.equal(predicted.detach(), repeated.detach()):
                raise ContractError("Python dynamics is not repeatable; random/stateful models require an explicit state adapter")
            jacobian = torch.autograd.functional.jacobian(transition, z).detach().numpy()
            finite = np.empty_like(jacobian)
            for index in range(len(point)):
                epsilon = min(1e-5*max(1., abs(point[index])), (upper[index]-lower[index])*1e-3)
                plus, minus = z.detach().clone(), z.detach().clone()
                plus[index] += epsilon
                minus[index] -= epsilon
                denominator = float(plus[index]-minus[index])
                if denominator == 0:
                    raise ContractError("Variable scale is too narrow for numerical gradient verification")
                finite[:, index] = ((transition(plus)-transition(minus))/denominator).detach().numpy()
            if not np.isfinite(jacobian).all() or not np.allclose(jacobian, finite, rtol=5e-3, atol=1e-5):
                raise ContractError("Python dynamics autograd disagrees with finite differences; detached/NumPy dynamics cannot drive physical gradients")
            batch = z.detach().repeat(2, 1)
            if not torch.allclose(transition(batch), predicted.detach().repeat(2, 1), rtol=0, atol=1e-12):
                raise ContractError("Python dynamics has different batch and single-sample semantics")
            x = z[:i.nx].detach().clone().requires_grad_(True)
            output = model.output(x)
            output_jac = torch.autograd.functional.jacobian(model.output, x).detach().numpy()
            output_fd = np.empty_like(output_jac)
            for index in range(i.nx):
                epsilon = min(1e-5*max(1., abs(point[index])), (upper[index]-lower[index])*1e-3)
                plus, minus = x.detach().clone(), x.detach().clone()
                plus[index] += epsilon
                minus[index] -= epsilon
                denominator = float(plus[index]-minus[index])
                if denominator == 0:
                    raise ContractError("State scale is too narrow for output gradient verification")
                output_fd[:, index] = ((model.output(plus)-model.output(minus))/denominator).detach().numpy()
            if not np.isfinite(output_jac).all() or not np.allclose(output_jac, output_fd, rtol=5e-3, atol=1e-5):
                raise ContractError("Python output-map autograd disagrees with finite differences")
            if not torch.allclose(model.output(x.detach().repeat(2, 1)), output.detach().repeat(2, 1), rtol=0, atol=1e-12):
                raise ContractError("Python output map has inconsistent batch semantics")
            probes.append({"input": point.tolist(), "next_state": predicted.detach().tolist(), "output": output.detach().tolist(),
                           "max_gradient_error": float(np.max(np.abs(jacobian-finite))),
                           "max_output_gradient_error": float(np.max(np.abs(output_jac-output_fd)))})
    return {"schema_version": 1, "source_sha256": model.spec["source_sha256"], "runtime_dependencies": runtime_dependencies(),
            "execution_surface": "frozen_python_source_with_declared_tensor_functions", "probes": probes,
            "scope": "Finite shape/repeatability/autograd probes, not a proof across the full operating domain"}


def register_python_model(workspace: Workspace, path: Path, source: dict, override: dict | None = None) -> str:
    payload = read_source(path)
    spec = python_spec(payload, override)
    if source.get("kind") not in {"user_nominal", "reference_simulator", "reconstructed_equations"}:
        raise ContractError("Python model source kind must be explicitly declared")
    spec.update(source_file="source.py", source_sha256=hashlib.sha256(payload).hexdigest(), runtime_dependencies=runtime_dependencies())
    interface, _ = validate_python_spec(spec)
    kind = "reference_model" if source["kind"] == "reference_simulator" else "nominal_model"
    with workspace.stage(kind, interface, [], {"source": source, "model_spec": spec}) as stage:
        snapshot = stage.root / "source.py"
        snapshot.write_bytes(payload)
        # Distinct import seeds expose unfrozen random initialization while the
        # caller's own training/collection random streams remain unchanged.
        with preserve_random_state():
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            model = PythonTensorDynamics(spec, snapshot)
            validation = validate_python_numerics(model)
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            second = PythonTensorDynamics(spec, snapshot)
            again = validate_python_numerics(second)
        for before, after in zip(validation["probes"], again["probes"]):
            if not np.allclose(before["next_state"], after["next_state"], rtol=0, atol=1e-12) or not np.allclose(before["output"], after["output"], rtol=0, atol=1e-12):
                raise ContractError("Independent Python imports produced different model behavior")
        validation["import_seed_independence_checked"] = True
        atomic_json(stage.root / "model.json", spec)
        atomic_json(stage.root / "python-validation.json", validation)
        if model.transition_adapter is not None:
            atomic_json(stage.root / "transition_adapter.json", model.transition_adapter)
        return stage.publish({"status": "verified", "checks": ["source_hash", "runtime_dependencies", "finite_shapes", "repeatability", "autograd_finite_differences", "batch_semantics"],
                              "physical_fidelity": "user_supplied_not_independently_certified", "execution_surface": validation["execution_surface"]},
                             source={**source, "input_path": str(path.resolve()), "input_sha256": spec["source_sha256"],
                                     "source_representation": spec["representation"], "explicit_source_execution": True})


def load_python_model(root: Path, spec: dict) -> PythonTensorDynamics:
    if spec.get("source_file") != "source.py":
        raise ContractError("Python model must use its frozen artifact source.py")
    model = PythonTensorDynamics(spec, root / "source.py")
    saved = read_json(root / "python-validation.json")
    if saved.get("schema_version") != 1:
        raise ContractError("Unsupported Python model validation schema_version")
    current = validate_python_numerics(model)
    if saved.get("source_sha256") != current["source_sha256"] or saved.get("runtime_dependencies") != current["runtime_dependencies"]:
        raise ContractError("Python model validation does not match its source and runtime")
    if len(saved.get("probes", [])) != len(current["probes"]):
        raise ContractError("Python model has an incomplete saved validation record")
    for before, after in zip(saved["probes"], current["probes"]):
        if not np.allclose(before["input"], after["input"], rtol=0, atol=0) or not np.allclose(before["next_state"], after["next_state"], rtol=0, atol=1e-12) or not np.allclose(before["output"], after["output"], rtol=0, atol=1e-12):
            raise ContractError("Python dynamics behavior differs from its saved registration probes")
    if model.transition_adapter is not None and read_json(root / "transition_adapter.json") != model.transition_adapter:
        raise ContractError("Python integration adapter differs from its saved declaration")
    return model
