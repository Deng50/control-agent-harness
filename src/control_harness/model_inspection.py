"""Read-only model-material inspection; uploaded Python is parsed, never executed."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

from .contracts import ContractError
from .modeling import DeclarativeDynamics
from .runtime import OperationContext

MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_XML_BYTES = 2 * 1024 * 1024
MANIFEST_NAMES = {"ENV_SPEC", "MANIFEST", "MODEL_SPEC", "DYNAMICS_SPEC"}
_SENSITIVE = re.compile(r"(?:api_?key|secret|password|access_?token|auth_?token|private_?key|(?:^|_)(?:token|credential|authorization)(?:_|$))", re.I)


def _json_literal(value, *, depth=0):
    if depth > 30:
        raise ValueError("literal nesting limit exceeded")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_literal(v, depth=depth + 1) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: ("[redacted]" if _SENSITIVE.search(k) else _json_literal(v, depth=depth + 1))
                for k, v in value.items()}
    raise ValueError("only finite JSON-compatible literals are inspected")


def _candidate_report(spec, evidence: dict) -> dict:
    missing, errors = [], []
    if not isinstance(spec, dict):
        return {"evidence": evidence, "valid": False, "missing": ["model object"],
                "errors": ["A declarative model must be a JSON object"]}
    is_python = spec.get("representation") in {"python_torch_transition", "python_torch_derivative"}
    required = ("representation", "interface", "entrypoint") if is_python else ("representation", "interface", "A", "B", "C")
    for name in required:
        if name not in spec:
            missing.append(name)
    if spec.get("representation") in {"continuous_polynomial", "python_torch_derivative"}:
        for name in ("time_unit", "derivative_units", "integration"):
            if name not in spec:
                missing.append(name)
        if isinstance(spec.get("integration"), dict):
            for name in ("method", "substeps", "input_hold"):
                if name not in spec["integration"]:
                    missing.append(f"integration.{name}")
    interface = spec.get("interface")
    if isinstance(interface, dict):
        for name in ("schema_version", "name", "sample_time", "time_semantics", "states", "controls", "outputs"):
            if name not in interface:
                missing.append(f"interface.{name}")
        for group in ("states", "controls", "outputs", "disturbances"):
            variables = interface.get(group)
            if group != "disturbances" and isinstance(variables, list) and not variables:
                missing.append(f"interface.{group} (at least one declared variable)")
            if isinstance(variables, list):
                for index, variable in enumerate(variables):
                    if not isinstance(variable, dict):
                        errors.append(f"interface.{group}[{index}] must be an object")
                        continue
                    for key in ("name", "unit", "lower", "upper"):
                        if key not in variable:
                            missing.append(f"interface.{group}[{index}].{key}")
                    for key in ("name", "unit"):
                        if key in variable and (not isinstance(variable[key], str) or not variable[key].strip()):
                            errors.append(f"interface.{group}[{index}].{key} must be an explicit nonempty string")
    elif "interface" in spec:
        errors.append("interface must be an object")
    parsed, model, adapter = None, None, None
    if not missing and not errors:
        try:
            if is_python:
                from .python_model import validate_python_spec
                parsed, adapter = validate_python_spec(spec)
            else:
                model = DeclarativeDynamics(spec)
                parsed, adapter = model.interface, model.transition_adapter
        except (ContractError, KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    result = {"evidence": evidence, "valid": parsed is not None, "missing": missing, "errors": errors,
              "model_spec": spec,
              "physical_fidelity": "declared_equations_not_independently_validated"}
    if parsed is not None:
        result.update(interface=parsed.to_dict(), interface_hash=parsed.fingerprint,
                      dimensions={"states": parsed.nx, "controls": parsed.nu,
                                  "outputs": parsed.ny, "disturbances": parsed.nd})
        if adapter is not None:
            result.update(source_interface=spec["interface"], transition_adapter=adapter)
        if is_python:
            result.update(registration_operation="register_python_model", runtime_execution_verified=False)
    return result


def _declared_candidates(value, evidence: dict) -> list[dict]:
    if not isinstance(value, dict):
        return []
    candidates = []
    # These are explicit schema containers, not variable-name semantic guesses.
    if "interface" in value or "representation" in value:
        candidates.append(_candidate_report(value, evidence))
    for field in ("model", "model_spec", "dynamics_spec"):
        if isinstance(value.get(field), dict):
            candidates.append(_candidate_report(value[field], {**evidence, "field": field}))
    return candidates


class _SourceVisitor(ast.NodeVisitor):
    def __init__(self):
        self.scope = []
        self.functions, self.classes, self.constants, self.manifests, self.candidates, self.uncertain = [], [], [], [], [], []

    def _name(self, name):
        return ".".join([*self.scope, name])

    def visit_ClassDef(self, node):
        self.classes.append({"name": self._name(node.name), "line": node.lineno,
                             "end_line": node.end_lineno, "bases": [ast.unparse(base)[:160] for base in node.bases]})
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self.functions.append({"name": self._name(node.name), "line": node.lineno, "end_line": node.end_lineno,
            "async": isinstance(node, ast.AsyncFunctionDef),
            "arguments": [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]],
            "keyword_only_arguments": [arg.arg for arg in node.args.kwonlyargs],
            "vararg": node.args.vararg.arg if node.args.vararg else None,
            "kwarg": node.args.kwarg.arg if node.args.kwarg else None})
        # Function bodies are never evaluated. Nested callable declarations can
        # be inventoried without interpreting local assignments as model facts.
        self.scope.append(node.name)
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(statement)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._assignment(target.id, node.value, node.lineno)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.value is not None:
            self._assignment(node.target.id, node.value, node.lineno)

    def _assignment(self, name, value, line):
        is_manifest = name.upper() in MANIFEST_NAMES
        if not is_manifest and not name.isupper():
            return
        evidence = {"kind": "python_ast_literal_assignment", "name": self._name(name), "line": line}
        record = {**evidence, "expression_type": type(value).__name__}
        if _SENSITIVE.search(name):
            record.update(literal=False, value="[redacted]")
            self.constants.append(record)
            return
        try:
            literal = _json_literal(ast.literal_eval(value))
            record["literal"] = True
            size = len(json.dumps(literal, ensure_ascii=False))
            record["value"] = literal if size <= 32768 else {"omitted": "literal exceeds report display budget", "characters": size}
            if is_manifest:
                self.candidates.extend(_declared_candidates(literal, evidence))
        except (ValueError, TypeError, SyntaxError, RecursionError):
            record.update(literal=False, reason="not_a_finite_static_literal")
            if is_manifest:
                self.uncertain.append({"field": self._name(name), "line": line,
                    "reason": "Manifest depends on names/calls/computation; no source code was executed"})
        self.constants.append(record)
        if is_manifest:
            self.manifests.append(record)


def matlab_runtime_status() -> dict:
    """Filesystem/PATH discovery only. Installation is not a license check."""
    commands = [os.environ.get("SIMULINK_MPC_MATLAB_COMMAND", ""), shutil.which("matlab") or ""]
    executable = None
    for value in commands:
        if value and Path(value).is_file():
            executable = str(Path(value).resolve())
            break
    if executable is None:
        standard_root = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "MATLAB"
        if standard_root.is_dir():
            found = sorted(standard_root.glob("*/bin/matlab.exe"))
            executable = str(found[-1].resolve()) if found else None
    return {"matlab_executable_found": executable is not None, "matlab_executable": executable,
            "matlab_started": False, "license_status": "not_checked",
            "required_products": ["MATLAB", "Simulink"], "product_availability": "not_checked",
            "native_execution_supported": False,
            "blockers": ["native_simulink_execution_adapter_not_implemented", "MATLAB_and_Simulink_licenses_not_verified"]}


def _slx_structure(content: bytes, context: OperationContext) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            infos = archive.infolist()
            if len(infos) > 5000:
                raise ContractError("SLX archive has too many entries for static inspection")
            names = [info.filename for info in infos]
            selected = [info for info in infos if info.filename == "simulink/blockdiagram.xml"
                        or (info.filename.startswith("simulink/systems/") and info.filename.endswith(".xml"))]
            if not selected:
                raise ContractError("Archive contains no recognized Simulink block-diagram/system XML")
            if sum(info.file_size for info in selected) > 8 * MAX_XML_BYTES:
                raise ContractError("SLX XML content exceeds static inspection budget")
            blocks = []
            for info in selected:
                context.check()
                if info.file_size > MAX_XML_BYTES:
                    raise ContractError("SLX XML entry exceeds static inspection budget")
                raw = archive.read(info)
                if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                    raise ContractError("SLX XML DTD/entity declarations are not accepted by the static inspector")
                root = ET.fromstring(raw)
                for block in root.iter("Block"):
                    properties = {p.get("Name"): p.text for p in block.findall("P")
                                  if p.get("Name") in {"Port", "SampleTime"}}
                    blocks.append({"name": block.get("Name"), "block_type": block.get("BlockType"),
                        "sid": block.get("SID"), "properties": properties,
                        "evidence": {"archive_entry": info.filename, "kind": "slx_xml_block_declaration"}})
                    if len(blocks) > 5000:
                        raise ContractError("SLX contains too many blocks for static inspection")
            return {"archive_entries": len(names), "inspected_xml_entries": [info.filename for info in selected],
                    "blocks": blocks, "declared_port_blocks": [block for block in blocks
                        if block["block_type"] in {"Inport", "Outport"}],
                    "compiled": False, "executed": False,
                    "scope": "static_archive_declarations_not_resolved_dimensions_units_or_runtime_interfaces"}
    except (zipfile.BadZipFile, ET.ParseError, RuntimeError) as exc:
        raise ContractError(f"Unable to inspect SLX archive: {exc}") from exc


def _matlab_source_structure(content: bytes, context: OperationContext) -> dict:
    """Recognize literal declarations only; this is not a MATLAB interpreter."""
    text = content.decode("utf-8-sig")
    functions, literals, headers = [], [], []
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    in_block_comment = False
    logical, first_line = "", 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        context.check()
        stripped = raw.strip()
        if stripped == "%{":
            in_block_comment = True
            continue
        if stripped == "%}":
            in_block_comment = False
            continue
        if in_block_comment:
            continue
        if stripped.startswith("%"):
            if not functions and not literals and len(headers) < 8:
                header = stripped[1:].strip()
                headers.append({"line": line_number,
                                "text": "[redacted]" if _SENSITIVE.search(header) else header[:240]})
            continue
        # Only signatures and numeric literal assignments are recognized below.
        # They contain no string literals, so an inline % starts a comment.
        clean = raw.split("%", 1)[0].strip()
        if not logical:
            first_line = line_number
        continued = clean.endswith("...")
        logical += " " + (clean[:-3] if continued else clean)
        if continued:
            continue
        declaration, logical = logical.strip(), ""
        function = re.fullmatch(r"function\s+(?:(\[[^\]]*\]|[A-Za-z]\w*)\s*=\s*)?"
                                r"([A-Za-z]\w*)\s*(?:\(([^)]*)\))?\s*;?", declaration)
        if function:
            outputs, name, arguments = function.groups()
            functions.append({"name": name, "line": first_line, "end_line": line_number,
                "arguments": [value.strip() for value in (arguments or "").split(",") if value.strip()],
                "outputs": [value for value in re.split(r"[\s,]+", (outputs or "").strip("[] ")) if value],
                "evidence": "matlab_function_header_only"})
            continue
        assignment = re.fullmatch(r"([A-Za-z]\w*)\s*=\s*(.*?)\s*;?", declaration)
        if not assignment:
            continue
        name, expression = assignment.groups()
        expression = expression.rstrip("; ")
        if _SENSITIVE.search(name):
            literals.append({"name": name, "line": first_line, "value": "[redacted]",
                             "kind": "matlab_assignment_candidate"})
            continue
        value = None
        if re.fullmatch(number, expression):
            value = float(expression)
        elif expression.startswith("[") and expression.endswith("]"):
            rows = [row.strip() for row in expression[1:-1].split(";")]
            tokens = [re.split(r"[\s,]+", row) for row in rows]
            if (tokens and all(row and all(re.fullmatch(number, token) for token in row) for row in tokens)
                    and len({len(row) for row in tokens}) == 1 and sum(map(len, tokens)) <= 256):
                value = [[float(token) for token in row] for row in tokens]
        if value is not None:
            try:
                value = _json_literal(value)
            except ValueError:
                continue
            literals.append({"name": name, "line": first_line, "value": value,
                             "kind": "matlab_numeric_literal_assignment_candidate",
                             "semantic_role": "unconfirmed", "scope": "not_resolved"})
    return {"text_decoded": True, "compiled": False, "executed": False,
            "functions": functions, "numeric_literals": literals, "comment_header": headers,
            "scope": "Static function headers and numeric literals only; equations, scopes and dimensions are not inferred"}


def inspect_model_source(model_path: str | Path, context: OperationContext | None = None) -> dict:
    """Return evidence and editable candidates without registering or running source."""
    context = context or OperationContext()
    context.check()
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise ContractError(f"Model source file does not exist: {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ContractError("Model inspection source exceeds 8 MiB")
    suffix = path.suffix.lower()
    if suffix not in {".json", ".py", ".m", ".slx", ".mdl"}:
        raise ContractError("Static model inspection supports .json, .py, .m, .slx and .mdl")
    content = path.read_bytes()
    if len(content) > MAX_SOURCE_BYTES:
        raise ContractError("Model inspection source exceeds 8 MiB")
    report = {"source": {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(),
                         "size_bytes": len(content), "format": suffix[1:]},
              "source_executed": False, "source_modified": False, "registered": False,
              "candidates": [], "missing": [], "uncertain": [], "errors": []}
    if suffix == ".json":
        try:
            parsed = _json_literal(json.loads(content.decode("utf-8-sig")))
            report["candidates"] = _declared_candidates(parsed, {"kind": "explicit_json_model_schema", "json_path": "$"})
            if not report["candidates"]:
                report["candidates"] = [_candidate_report(parsed, {"kind": "json_object", "json_path": "$"})]
        except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as exc:
            report["errors"].append(f"Cannot parse JSON: {exc}")
    elif suffix == ".py":
        try:
            # ast.parse accepts source bytes and respects Python's encoding cookie.
            tree = ast.parse(content, filename=str(path))
            if sum(1 for _ in ast.walk(tree)) > 100000:
                raise ContractError("Python AST exceeds static inspection node budget")
            visitor = _SourceVisitor()
            visitor.visit(tree)
            report.update(functions=visitor.functions, classes=visitor.classes, constants=visitor.constants,
                          manifests=visitor.manifests, candidates=visitor.candidates)
            report["uncertain"].extend(visitor.uncertain)
            report["uncertain"].append({"field": "callable_semantics", "reason":
                "Function/argument names do not prove state/control/output semantics; arbitrary Python was not imported or executed"})
        except (SyntaxError, UnicodeDecodeError, RecursionError) as exc:
            report["errors"].append(f"Cannot parse Python: {exc}")
    else:
        report["runtime"] = matlab_runtime_status()
        if suffix == ".m":
            report["runtime"]["required_products"] = ["MATLAB"]
            try:
                report["matlab_structure"] = _matlab_source_structure(content, context)
            except UnicodeDecodeError:
                report["errors"].append("MATLAB text encoding is not UTF-8; explicit conversion is needed")
        elif suffix == ".slx":
            try:
                report["simulink_structure"] = _slx_structure(content, context)
            except ContractError as exc:
                report["errors"].append(str(exc))
        else:
            try:
                text = content.decode("utf-8-sig")
                report["simulink_structure"] = {"text_decoded": True, "compiled": False, "executed": False,
                    "scope": "MDL source recognized; no native block/interface resolution performed"}
                if not re.search(r"\b(?:Model|Library)\s*\{", text):
                    report["errors"].append("No Model/Library header found in textual MDL source")
            except UnicodeDecodeError:
                report["errors"].append("MDL text encoding is not UTF-8; explicit conversion is needed")
        report["uncertain"].append({"field": "matlab_runtime_interface" if suffix == ".m" else "simulink_runtime_interface", "reason":
            "Compiled dimensions, units, initial conditions, equations and licenses are unverified; no MATLAB session was started"})
    for candidate in report["candidates"]:
        if candidate.get("registration_operation") != "register_python_model":
            continue
        try:
            if suffix != ".py":
                raise ContractError("Python execution declarations require the corresponding standalone .py source")
            from .python_model import python_spec
            python_spec(content, candidate["model_spec"])
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            candidate["valid"] = False
            candidate["errors"].append(str(exc))
    if not report["candidates"]:
        report["missing"].extend(["explicit ordered states/controls/outputs with units and bounds",
            "sample_time and explicit discrete transition or continuous derivative with integration semantics",
            "executable declarative A/B/C and optional E/bias/terms"])
    else:
        report["missing"] = sorted({item for candidate in report["candidates"] for item in candidate["missing"]})
    valid = [candidate for candidate in report["candidates"] if candidate["valid"]]
    report["registration_ready"] = len(valid) == 1 and not report["errors"]
    report["status"] = "ready_to_register" if report["registration_ready"] else "needs_information"
    if len(valid) > 1:
        report["uncertain"].append({"field": "model_selection", "reason": "Multiple valid model declarations require an explicit selection"})
    if report["registration_ready"]:
        report["editable_model_spec"] = valid[0]["model_spec"]
        report["registration_operation"] = valid[0].get("registration_operation", "register_model")
    report["next_action"] = ("Review/edit the complete model spec, save it as JSON, then explicitly register_model"
                             if report["registration_ready"] else
                             "Supply or correct an explicit complete model schema; do not infer missing signal semantics from names")
    if report.get("registration_operation") == "register_python_model":
        report["next_action"] = "Review the standalone Python source and interface; explicitly register_python_model to execute and numerically validate it in the native worker"
    context.progress("model_source_inspection", format=suffix[1:], candidates=len(report["candidates"]),
                     registration_ready=report["registration_ready"], executed=False)
    return report
