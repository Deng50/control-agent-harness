"""Worker-side recheck of native admission scopes and immutable input versions.

Workflow confirmations describe only the selected stages. Uploaded workflow
JSON and explicitly declared resources are read as data, never executed here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from .artifacts import Workspace, canonical_hash
from .contracts import ContractError
from .runtime import OperationContext

PROTECTED = {
    "convert_plant_predictor": ("signals",), "verify_design_delivery": ("signals",),
    "check_matlab_design": ("signals",),
    "prepare_supervised_dataset": ("signals",),
    "train_supervised": ("signals", "training"),
    "train_predictive_model": ("signals", "training"),
    "sample_model": ("signals", "sampling"),
    "simulate_transitions": ("signals", "sampling"), "collect_expert": ("signals", "sampling"),
    "validate_estimator": ("signals", "sampling"),
    "train_residual": ("signals", "training"), "train_imitation": ("signals", "training"),
    "optimize_policy": ("signals", "training"), "identify_linear_model": ("signals", "training"),
    "design_controller": ("signals", "training"),
    "tune_controller": ("signals", "training"),
    "collect_external_expert": ("signals", "sampling"),
    "import_expert_data": ("signals",),
    "train_external_imitation": ("signals", "training"),
    "evaluate_external_imitation": ("signals", "sampling"),
    "prepare_pinn_problem": ("signals",),
    "train_pinn": ("signals", "training"),
    "validate_differentiable_model": ("signals",),
    "optimize_external_policy": ("signals", "training"),
    "evaluate_external_pgac": ("signals", "sampling"),
    "run_workflow": (),  # Selected STAGES determine the actual union below.
    "demo": ("signals", "sampling", "training"),
}
SCOPES = ("signals", "sampling", "training")
RESUME_FIELDS = {"plan_refs", "resume_checkpoint", "resume_checkpoint_sha256", "resume_run"}


def synthesis_scopes(request: dict) -> tuple[str, ...]:
    """Pure JSON boundary: parameterized code generation is not a model run."""
    inputs, config = request.get("inputs", {}), request.get("config", {})
    prediction = config.get("prediction")
    prediction = prediction if isinstance(prediction, dict) else {}
    method = prediction.get("method")
    numeric = (inputs.get("model") is not None or inputs.get("dataset") is not None
               or prediction.get("model_spec") is not None)
    if not numeric:
        template = config.get("template")
        if template is None:
            description = inputs.get("description")
            description = description if isinstance(description, str) else ""
            template = "spring_damper" if "spring" in description.lower() or "弹簧" in description else "generic"
        params = config.get("parameters")
        required = ("mass", "damping", "stiffness", "sample_time", "state_bounds", "input_bounds", "output_bounds")
        numeric = (template == "spring_damper" and isinstance(params, dict)
                   and all(key in params and params[key] is not None for key in required))
    if not numeric:
        return ()
    if method == "identify_affine" or method is None and inputs.get("dataset") is not None:
        return ("signals", "training")
    return ("signals",)


def _config(config: dict) -> dict:
    return copy.deepcopy({key: value for key, value in config.items() if key not in RESUME_FIELDS})


def _file(path, *, content=False):
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ContractError("Declared workflow resources require absolute paths")
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ContractError(f"Declared workflow resource is not a file: {path}")
    before = path.stat()
    digest, chunks, count = hashlib.sha256(), [], 0
    signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    with path.open("rb") as stream:
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ContractError("Workflow source changed before reading")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
            if content:
                chunks.append(chunk)
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ContractError("Workflow source changed while reading")
    if signature(path.stat()) != signature(before) or count != before.st_size:
        raise ContractError("Workflow source changed while reading")
    return {"kind": "file", "path": str(path), "sha256": digest.hexdigest(), "size_bytes": count}, b"".join(chunks)


def _closure(workspace: Workspace, artifact_id: str) -> dict:
    versions = {}

    def visit(item_id):
        if item_id in versions:
            return
        root, meta = workspace.resolve(item_id)
        row = workspace.manifest["artifacts"][item_id]
        record, raw = _file(root / "artifact.json", content=True)
        if canonical_hash(json.loads(raw.decode("utf-8"))) != row["metadata_hash"]:
            raise ContractError("Case artifact metadata changed while reading")
        versions[item_id] = {"id": item_id, "metadata_hash": row["metadata_hash"],
            "metadata_file_sha256": record["sha256"], "content_hash": meta.get("source", {}).get("content_hash"),
            "files": dict(meta["files"]), "parents": copy.deepcopy(meta["parents"])}
        for parent in meta["parents"]:
            visit(parent["id"])

    visit(artifact_id)
    return versions


def resolve_workflow_case(workspace: Workspace, request: dict) -> dict:
    """Read one complete definition and bind its explicit files/artifact closure."""
    from .workflow import MODELS, STAGES
    inputs, config = request.get("inputs", {}), request.get("config", {})
    bindings, file_sources, external_sources = {}, {}, {}
    if inputs.get("workflow_path"):
        if "steps" in config:
            raise ContractError("Choose either workflow_path or inline steps")
        record, raw = _file(inputs["workflow_path"], content=True)
        bindings["workflow_file"] = record
        try:
            definition = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise ContractError("Workflow source must be valid UTF-8 JSON") from error
        base = Path(record["path"]).parent
    else:
        definition = copy.deepcopy({key: config[key] for key in ("schema_version", "name", "steps") if key in config})
        base = None
    if (not isinstance(definition, dict) or definition.get("schema_version") != 1
            or not isinstance(definition.get("steps"), list) or not 1 <= len(definition["steps"]) <= 50):
        raise ContractError("Workflow requires schema_version 1 and 1-50 explicit stages")
    selected_scopes, versions, seen = set(), {}, set()
    for step in definition["steps"]:
        if not isinstance(step, dict) or not isinstance(step.get("id"), str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", step["id"]) or step["id"] in seen:
            raise ContractError("Workflow stage ids must be unique valid identifiers")
        operation = step.get("operation")
        if operation not in STAGES:
            raise ContractError(f"Unsupported workflow operation: {operation}")
        required = STAGES[operation][0]
        supplied = step.get("inputs", {})
        if not isinstance(supplied, dict) or not isinstance(step.get("config", {}), dict) or set(required) - supplied.keys():
            raise ContractError("Workflow stage inputs/config are incomplete")
        selected_scopes.update(PROTECTED.get(operation, ()))
        for name, value in supplied.items():
            expected = required.get(name)
            if operation == "import_transitions" and name == "reference_model":
                expected = MODELS
            if expected is None:
                raise ContractError(f"Unknown workflow input {name} for {operation}")
            key = f"{step['id']}.{name}"
            if isinstance(value, dict):
                if not isinstance(expected, set) or set(value) != {"stage"} or value["stage"] not in seen:
                    raise ContractError("Workflow reference must name an earlier stage output")
                continue
            if not isinstance(value, str) or not value.strip():
                raise ContractError("Workflow input must be a declared path, artifact id or stage reference")
            if expected in ("file", "artifact_directory"):
                path = Path(value).expanduser()
                if not path.is_absolute() and base is not None:
                    path = base / path
                if not path.is_absolute():
                    raise ContractError("Inline workflow paths must be absolute")
                path = path.resolve(strict=True)
                supplied[name] = str(path)
                if expected == "file":
                    record, _ = _file(path)
                    versions[key] = record
                    file_sources[str(path)] = record
                else:
                    meta_path = path / "artifact.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not (path.parent.parent / "manifest.json").is_file():
                        raise ContractError("External artifact source manifest is missing")
                    source = Workspace(path.parent.parent)
                    resolved, _ = source.resolve(meta["id"])
                    if resolved != path:
                        raise ContractError("External artifact directory does not match its source manifest")
                    closure = _closure(source, meta["id"])
                    versions[key] = {"kind": "artifact_directory", "path": str(path), "workspace": str(source.path),
                        "project_id": source.manifest["project_id"], "id": meta["id"], "artifacts": closure}
                    external_sources[str(path)] = versions[key]
            else:
                meta = workspace.resolve(value)[1]
                if meta["kind"] not in expected:
                    raise ContractError("Workflow artifact kind does not match its declared stage input")
                versions[key] = {"kind": "workspace_artifact", "id": value, "artifacts": _closure(workspace, value)}
        seen.add(step["id"])
    bindings["workflow_inputs"] = versions
    definition = copy.deepcopy(definition)
    return {"definition": definition, "bindings": bindings,
        "scopes": [scope for scope in SCOPES if scope in selected_scopes],
        "payload": {"inputs": copy.deepcopy(inputs), "config": _config(config), "workflow_definition": definition},
        "sources": {"files": file_sources, "external_artifacts": external_sources},
        "request": copy.deepcopy(request)}


def freeze_workflow_case(workspace: Workspace, case: dict, snapshot_directory: str | Path,
                         context: OperationContext | None = None) -> dict:
    """Use only for a verified Host receipt or explicit test context, under its workspace lock.

    Existing artifact import transactions snapshot external dependency closures.
    Direct files are copied and hash-checked. No source code or numerical stage
    runs here. The caller passes returned inputs/config to the unchanged engine.
    """
    context = context or OperationContext()
    current = resolve_workflow_case(workspace, case["request"])
    if current["bindings"] != case["bindings"] or current["definition"] != case["definition"]:
        raise ContractError("Workflow source changed after confirmation verification")
    directory = Path(snapshot_directory)
    if not directory.is_absolute() or not directory.resolve().is_relative_to(workspace.path.resolve()):
        raise ContractError("Frozen workflow inputs must remain inside the authorized workspace")
    directory.mkdir(parents=True, exist_ok=True)
    frozen = copy.deepcopy(case["definition"])
    replacements = {}
    for original, expected in case["sources"]["files"].items():
        context.check()
        target = directory / "files" / f"{expected['sha256']}{Path(original).suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".copy-{uuid.uuid4().hex}")
        try:
            shutil.copyfile(original, temporary)
            copied, _ = _file(temporary)
            source, _ = _file(original)
            if (copied["sha256"] != expected["sha256"] or copied["size_bytes"] != expected["size_bytes"]
                    or source != expected):
                raise ContractError("Workflow input changed while creating its frozen copy")
            if target.exists():
                if _file(target)[0]["sha256"] != expected["sha256"]:
                    raise ContractError("Previously frozen workflow input was changed")
            else:
                temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
        replacements[original] = str(target)
    for original, expected in case["sources"]["external_artifacts"].items():
        context.check()
        cache_key = canonical_hash({"project_id": expected["project_id"], "artifacts": expected["artifacts"]})
        # Short namespace prevents normal final filenames from inheriting the
        # import transaction's long temporary paths. Full versions are still
        # verified below, so the prefix is never an authority check.
        cache_path = directory / "a" / cache_key[:24]
        # Reuse the existing import transaction without exceeding legacy Win32
        # MAX_PATH in its nested temporary metadata filenames. This is the same
        # authorized workspace path, not a different storage location.
        if os.name == "nt" and not str(cache_path).startswith("\\\\?\\"):
            text = str(cache_path.resolve())
            cache_path = Path("\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text)
        cache = Workspace(cache_path)
        selected = cache.import_artifact(original)
        copied_versions = _closure(cache, selected)
        # Artifact import reserializes wrapper metadata canonically; declared
        # file bytes, canonical metadata and complete parent versions must match.
        def semantic(versions):
            return {key: {name: value for name, value in row.items() if name != "metadata_file_sha256"}
                    for key, row in versions.items()}
        if semantic(copied_versions) != semantic(expected["artifacts"]):
            raise ContractError("External artifact changed while freezing its dependency closure")
        replacements[original] = str(cache.resolve(selected)[0])
    current = resolve_workflow_case(workspace, case["request"])
    if current["bindings"] != case["bindings"] or current["definition"] != case["definition"]:
        raise ContractError("Workflow source changed during freezing; no stage may execute")
    from .workflow import STAGES
    for step in frozen["steps"]:
        for name, kind in STAGES[step["operation"]][0].items():
            if kind in ("file", "artifact_directory"):
                step["inputs"][name] = replacements[step["inputs"][name]]
    settings = copy.deepcopy(case["request"].get("config", {}))
    settings.update({key: frozen[key] for key in ("schema_version", "name", "steps") if key in frozen})
    context.check()
    return {"inputs": {}, "config": settings, "sourcedigest": canonical_hash(case["bindings"]),
            "definition_hash": canonical_hash(case["definition"])}


def _mlp_settings(config, interface) -> dict:
    from .supervised import _resolved
    values, _, _ = _resolved(config, interface)
    # 只显示并封存会决定网络结构、数值训练或验收的实际参数。未填写字段使用训练器同一套默认值。
    return {key: values[key] for key in ("input_dim", "output_dim", "hidden_dims", "activation", "epochs",
            "learning_rate", "batch_size", "patience", "seed", "weight_decay", "min_delta", "grad_clip", "max_test_nrmse")}


def validate_review_settings(requirements: list[dict]) -> dict:
    """在写入版本前复用实际执行器的静态检查。此处不执行上传的模型或求解控制器。"""
    from .contracts import DataInterface
    from .datasets import _fractions
    interface, effective = None, {}
    for item in requirements:
        payload = item.get("reviewPayload")
        if not isinstance(payload, dict):
            raise ContractError("Review payload must be a JSON object")
        if payload.get("operation") == "synthesize_mpc" and item["scope"] == "signals":
            from .design_only import _source, _partial_spring, _controller
            config = payload["config"]
            _, template = _source(payload["inputs"], config)
            if template == "spring_damper":
                _partial_spring(config.get("parameters", {}))
            # 已登记预测器来自摘要中的物理接口，明确方程的接口来自 model_spec。
            raw_interface = config.get("prediction", {}).get("model_spec", {}).get("interface", payload["summary"])
            axes = raw_interface.get("controls")
            _controller(config, len(axes) if isinstance(axes, list) else None)
        if payload.get("operation") == "prepare_supervised_dataset":
            config = payload.get("config")
            if not isinstance(config, dict) or not isinstance(config.get("interface"), dict):
                raise ContractError("Provide a complete tabular data interface")
            interface = DataInterface.parse(config["interface"])
            split, source = config.get("split"), config.get("source")
            if (not isinstance(split, dict) or split.get("mode") not in {"trip", "time"}
                    or split["mode"] == "trip" and config["interface"].get("trip_column") is None):
                raise ContractError("Review a time split or explicit trip grouping column")
            _fractions(split)
            if (config.get("scaler", "standard") not in {"standard", "minmax", "none"}
                    or not isinstance(source, dict) or source.get("kind") not in {"measured", "synthetic", "external_expert"}
                    or not isinstance(source.get("description"), str) or not source["description"].strip()):
                raise ContractError("Review a supported scaler and explicit data provenance")
        if payload.get("operation") == "train_supervised":
            if item["scope"] == "signals" and isinstance(payload.get("summary"), dict):
                interface = DataInterface.parse(payload["summary"])
            if item["scope"] == "training" and interface:
                effective["training"] = _mlp_settings(payload.get("config", payload.get("summary")), interface)
    return effective


def confirmation_requirements(workspace: Workspace, request: dict, *, workflow_case: dict | None = None,
                              context=None, structured: bool = False) -> list[dict]:
    # 老接口继续返回 v1 的原始结构。Worker 依磁盘计划版本选择 v2，不能由请求中的布尔值切换协议。
    references = request.get("config", {}).get("plan_refs", {})
    if not structured and context and isinstance(references, dict):
        from .case_plans import read_plan
        if references:
            records = read_plan(workspace)["plans"]
            structured = any(isinstance(ref, dict) and records.get(ref.get("plan_id"), {}).get("schema_version") == 2
                             for ref in references.values())
    if request["operation"] == "run_workflow":
        case = workflow_case if workflow_case is not None else resolve_workflow_case(workspace, request)
        return [{"scope": scope, "expected_payload": case["payload"], "bindings": case["bindings"],
                 **({"review_payload": {"operation": request["operation"], "summary": case["payload"]},
                     "review_bindings": case["bindings"]} if structured else {})}
                for scope in case["scopes"]]
    scopes = synthesis_scopes(request) if request["operation"] == "synthesize_mpc" else PROTECTED.get(request["operation"], ())
    if not scopes:
        return []
    index, bindings, metadata = workspace.manifest["artifacts"], {}, {}
    for key, value in request.get("inputs", {}).items():
        if not isinstance(value, str) or value not in index:
            continue
        _, meta = workspace.resolve(value)
        bindings[key] = {"id": value, "metadata_hash": index[value]["metadata_hash"],
                         "content_hash": meta.get("source", {}).get("content_hash")}
        metadata[key] = meta
    config = _config(request.get("config", {}))
    if request["operation"] == "evaluate_external_imitation" and (not isinstance(config.get("scenarios"), list) or not config["scenarios"]):
        raise ContractError("Prepare and review explicit evaluation scenarios before confirming an original-Plant comparison")
    if request["operation"] == "sample_model":
        # Only select trial versus full within the same declared plan and trial
        # prefix. Both ranges remain in the reviewed configuration; the signed
        # receipt still binds the exact execution request and selection.
        config.pop("sampling_run", None)
    blueprint = {"inputs": request.get("inputs", {}), "config": config}
    selected = next((metadata[key]["interface"] for key in
        ("nominal_model", "model", "dataset", "controller", "policy", "evaluation_model") if key in metadata), None)
    signals = config.get("interface") if request["operation"] == "prepare_supervised_dataset" else selected or blueprint
    if request["operation"] in {"convert_plant_predictor", "verify_design_delivery"}:
        signals = blueprint
    if request["operation"] in {"import_expert_data", "prepare_pinn_problem"}:
        # One existing signals review covers the complete declared import
        # contract: causal mappings, actuator limits, provenance and data prep.
        # Resume/plan references were removed above; they are not engineering
        # changes and must not recursively enter the confirmed payload.
        signals = config
    if request["operation"] == "train_pinn":
        problem = metadata.get("problem", {})
        signals = problem.get("source", {}).get("pinn_signals")
        if problem.get("kind") != "conditional_ode_problem" or not isinstance(signals, dict):
            raise ContractError("Select an explicit conditional ODE problem with equation, time/condition units and normalization before PINN training")
    if request["operation"] == "validate_differentiable_model":
        signals = {"candidate_interface": metadata.get("model", {}).get("interface"),
                   "original_interface": metadata.get("evaluation_model", {}).get("interface"), "validation_config": config}
    if request["operation"] == "optimize_external_policy":
        signals = {"rollout_model_interface": metadata.get("model", {}).get("interface"),
                   "policy_signal_interface": metadata.get("policy", {}).get("interface"),
                   "validated_source": metadata.get("validation", {}).get("source")}
    if request["operation"] == "evaluate_external_pgac":
        if not isinstance(config.get("scenarios"), list) or not config["scenarios"]:
            raise ContractError("Review the frozen original-Plant validation/test scenarios before confirming PGAC evaluation")
        signals = metadata.get("candidate", {}).get("interface")
    if request["operation"] == "collect_external_expert":
        if "plant" not in metadata or metadata.get("expert",{}).get("kind") != "external_expert":
            raise ContractError("Select bound external expert and plant artifacts before confirming collection")
        expert_root, expert_meta = workspace.resolve(request["inputs"]["expert"])
        payload = (expert_root / "expert.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != expert_meta.get("files",{}).get("expert.json"):
            raise ContractError("External expert declaration changed before confirmation")
        spec = json.loads(payload)["adapter_spec"]
        fields = ("interface","control_period","time_unit","observations","reference_available","disturbance_available",
                  "previous_action_available","initial_memory","action_mapping","failure_policy","input_rate_bounds")
        signals = {"plant_interface":metadata["plant"]["interface"],"expert_adapter":{key:spec[key] for key in fields if key in spec}}
    operation = request["operation"]
    proof = None
    if structured and operation == "train_supervised" and "dataset" in metadata:
        from .preparation_proof import verified_preparation
        proof = verified_preparation(workspace, metadata["dataset"], request, context)
    requirements = []
    for scope in scopes:
        summary = signals if scope == "signals" else blueprint if operation == "demo" else config
        review, semantic_bindings = {"operation": operation, "summary": summary}, bindings
        if operation == "prepare_supervised_dataset":
            review = {"operation": operation, "config": config}
        elif proof:
            review = ({"operation": "prepare_supervised_dataset", "config": proof["config"]}
                      if scope == "signals" else {"operation": operation, "config": config, "preparation": proof["config"]})
            semantic_bindings = {"material": proof["material"]}
        elif operation == "synthesize_mpc" and scope == "signals":
            review.update(config=config, inputs=request.get("inputs", {}))
        requirements.append({"scope": scope, "expected_payload": summary, "bindings": bindings,
                             **({"review_payload": review, "review_bindings": semantic_bindings} if structured else {})})
        if structured and operation == "train_supervised" and scope == "training":
            from .contracts import DataInterface
            requirements[-1]["effective_settings"] = _mlp_settings(config, DataInterface.parse(metadata["dataset"]["interface"]))
    return requirements
