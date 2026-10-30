"""Deterministic domain stage composition inside one Harness job, not an agent loop."""
from __future__ import annotations

import copy
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from .artifacts import Workspace, atomic_json, canonical_hash, file_hash, read_json
from .contracts import ContractError, ModelInterface, validate_network_dimensions
from .control import partition_scenarios, validate_scenario
from .data import validate_transitions
from .modeling import DeclarativeDynamics
from .runtime import Cancelled, OperationContext

MODELS = {"nominal_model", "reference_model", "corrected_model", "identified_model"}
# Inputs are explicit. An active artifact is never silently substituted mid-workflow.
STAGES = {
    "register_model": ({"model_path": "file"}, "nominal_model"),
    "register_python_model": ({"model_path": "file"}, "nominal_model"),
    "import_artifact": ({"artifact_path": "artifact_directory"}, None),
    "import_transitions": ({"nominal_model": {"nominal_model"}, "data_path": "file"}, "transition_dataset"),
    "simulate_transitions": ({"nominal_model": {"nominal_model"}, "reference_model": MODELS}, "transition_dataset"),
    "train_residual": ({"nominal_model": {"nominal_model"}, "dataset": {"transition_dataset"}}, "corrected_model"),
    "design_controller": ({"model": MODELS, "evaluation_model": MODELS}, "controller"),
    "collect_expert": ({"controller": {"controller"}, "evaluation_model": MODELS}, "expert_dataset"),
    "train_imitation": ({"dataset": {"expert_dataset"}}, "imitation_policy"),
    "optimize_policy": ({"policy": {"imitation_policy", "optimized_policy"}, "model": MODELS}, "optimized_policy"),
    "identify_linear_model": ({"nominal_model": {"nominal_model"}, "dataset": {"transition_dataset"}}, "identified_model"),
    "validate_estimator": ({"model": MODELS, "controller": {"controller"}, "evaluation_model": MODELS}, "estimator"),
}


def workflow_definition(inputs: dict, config: dict) -> dict:
    if inputs.get("workflow_path"):
        if "steps" in config:
            raise ContractError("Choose either workflow_path or inline steps")
        path = Path(inputs["workflow_path"]).expanduser()
        if not path.is_absolute():
            raise ContractError("workflow_path must be absolute")
        raw = read_json(path)
        for step in raw.get("steps", []):
            required = STAGES.get(step.get("operation"), ({}, None))[0]
            for name, kind in required.items():
                value = step.get("inputs", {}).get(name)
                if kind in ("file", "artifact_directory") and isinstance(value, str) and not Path(value).is_absolute():
                    step["inputs"][name] = str((path.parent / value).resolve())
    else:
        raw = {k: config[k] for k in ("schema_version", "name", "steps") if k in config}
    if raw.get("schema_version") != 1 or not isinstance(raw.get("steps"), list) or not raw["steps"]:
        raise ContractError("Workflow requires schema_version: 1 and a nonempty steps array")
    if len(raw["steps"]) > 50:
        raise ContractError("A workflow supports at most 50 explicit stages")
    return copy.deepcopy(raw)


def _artifact_info(workspace, artifact_id):
    _, metadata = workspace.resolve(artifact_id)
    return {"kind": metadata["kind"], "interface": metadata["interface"],
            "metadata_hash": canonical_hash(metadata)}


def plan_workflow(workspace: Workspace, inputs: dict, config: dict) -> dict:
    """No training or publication; validate references, source files and known interfaces."""
    definition = workflow_definition(inputs, config)
    known, resources, checks, errors = {}, {}, [], []
    for step in definition["steps"]:
        stage_id = step.get("id") if isinstance(step, dict) else None
        try:
            if not isinstance(stage_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", stage_id):
                raise ContractError("Each stage requires an id of 1–64 letters/digits/_/- starting with a letter")
            if stage_id in known:
                raise ContractError(f"Duplicate stage id: {stage_id}")
            operation = step.get("operation")
            if operation not in STAGES:
                raise ContractError(f"Unsupported workflow operation: {operation}")
            required, output_kind = STAGES[operation]
            supplied, settings = step.get("inputs", {}), step.get("config", {})
            if not isinstance(supplied, dict) or not isinstance(settings, dict):
                raise ContractError("Stage inputs and config must be objects")
            if set(required) - supplied.keys():
                raise ContractError(f"Missing inputs: {sorted(set(required) - supplied.keys())}")
            interface = None
            for name, value in supplied.items():
                expected = required.get(name)
                if name == "reference_model" and operation == "import_transitions":
                    expected = MODELS
                if expected is None:
                    raise ContractError(f"Unknown input {name} for {operation}")
                if isinstance(value, dict):
                    if set(value) != {"stage"} or value["stage"] not in known:
                        raise ContractError(f"{name}: reference must name a valid earlier stage (no forward references or cycles)")
                    if not isinstance(expected, set):
                        raise ContractError(f"{name} expects a path, not a stage output")
                    info = known[value["stage"]]
                elif isinstance(value, str) and value.strip():
                    if isinstance(expected, set):
                        info = _artifact_info(workspace, value)
                        resources[f"artifact:{value}"] = info["metadata_hash"]
                    elif expected == "file":
                        path = Path(value).expanduser()
                        if not path.is_absolute() or not path.is_file():
                            raise ContractError(f"{name} requires an existing absolute file path")
                        resources[f"file:{path.resolve()}"] = file_hash(path)
                        continue
                    else:
                        path = Path(value).expanduser()
                        if not path.is_absolute() or not (path / "artifact.json").is_file():
                            raise ContractError("artifact_path requires an absolute artifact directory")
                        source = Workspace(path.parent.parent)
                        meta = read_json(path / "artifact.json")
                        resolved, _ = source.resolve(meta["id"])
                        if resolved != path.resolve():
                            raise ContractError("Artifact directory is inconsistent with source manifest")
                        info = _artifact_info(source, meta["id"])
                        output_kind = info["kind"]
                        resources[f"external:{path.resolve()}"] = info["metadata_hash"]
                else:
                    raise ContractError(f"{name} requires an artifact id, absolute path, or {{stage: id}}")
                if isinstance(expected, set) and info["kind"] not in expected:
                    raise ContractError(f"{name} expects {sorted(expected)}, received {info['kind']}")
                current = ModelInterface.parse(info["interface"])
                if interface:
                    interface.compatible(current)
                interface = current
            if operation in {"register_model", "register_python_model"}:
                spec = settings.get("model_spec")
                if operation == "register_python_model":
                    from .python_model import python_spec, read_source, validate_python_spec
                    spec = python_spec(read_source(Path(supplied["model_path"])), spec)
                    interface, _ = validate_python_spec(spec)
                else:
                    interface = DeclarativeDynamics(spec if spec is not None else read_json(Path(supplied["model_path"]))).interface
                source = settings.get("source", {})
                if source.get("kind") not in {"user_nominal", "reference_simulator", "reconstructed_equations"}:
                    raise ContractError("Model source kind must be explicitly declared")
                output_kind = "reference_model" if source["kind"] == "reference_simulator" else "nominal_model"
            if operation == "import_transitions":
                source = settings.get("source", {})
                if source.get("kind") not in {"measured", "reference_simulator"} or not source.get("description"):
                    raise ContractError("Transition source kind and description are required")
                if source["kind"] == "reference_simulator" and "reference_model" not in supplied:
                    raise ContractError("Simulated transitions require the exact reference_model")
                with np.load(supplied["data_path"], allow_pickle=False) as data:
                    validate_transitions(interface, {key: data[key] for key in data.files})
            if operation == "design_controller":
                partition_scenarios(settings, interface)
            if operation == "collect_expert":
                scenarios = settings.get("scenarios", [])
                if len(scenarios) < 5:
                    raise ContractError("Expert collection requires at least five independent scenarios")
                for scenario in scenarios:
                    validate_scenario(interface, scenario)
            if operation == "train_residual":
                validate_network_dimensions(interface, settings, residual=True)
            if operation == "train_imitation":
                from .policy import network_config
                network_config(interface, settings)
            if operation in {"train_residual", "train_imitation", "optimize_policy"}:
                if "epochs" in settings and (type(settings["epochs"]) is not int or settings["epochs"] < 1):
                    raise ContractError("epochs must be a positive integer")
                if "resume_checkpoint" in settings:
                    path = Path(settings["resume_checkpoint"])
                    if not path.is_absolute() or not path.is_file():
                        raise ContractError("resume_checkpoint must be an existing absolute file")
                    resources[f"file:{path.resolve()}"] = file_hash(path)
            known[stage_id] = {"kind": output_kind, "interface": interface.to_dict()}
            checks.append({"id": stage_id, "operation": operation, "output_kind": output_kind,
                           "interface": interface.to_dict(), "status": "ready"})
        except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
            errors.append({"id": stage_id, "message": str(exc)})
    return {"schema_version": 1, "status": "ready" if not errors else "invalid", "definition": definition,
            "definition_hash": canonical_hash(definition), "resources": resources, "stages": checks, "errors": errors,
            "validation_scope": "Source integrity, stage ordering, artifact types, numerical interfaces and declared scenarios. Numerical feasibility and acceptance are checked during execution."}


def run_workflow(workspace: Workspace, inputs: dict, config: dict, context: OperationContext) -> dict:
    from .operations import execute_stage
    with workspace.lock("run_workflow"):
        plan = plan_workflow(workspace, inputs, config)
        if plan["errors"]:
            raise ContractError(f"Workflow preflight failed: {plan['errors']}")
        run_id = config.get("resume_run") or uuid.uuid4().hex
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ContractError("resume_run must be a workflow run id")
        path = workspace.path / "workflows" / f"{run_id}.json"
        if config.get("resume_run"):
            if not path.is_file():
                raise ContractError("Workflow run does not exist in this workspace")
            record = read_json(path)
            if record["definition_hash"] != plan["definition_hash"] or record["resources"] != plan["resources"]:
                raise ContractError("Workflow definition or input content changed; create a new run instead of resuming")
        else:
            record = {"schema_version": 1, "run_id": run_id, "created_at": time.time(),
                      "definition": plan["definition"], "definition_hash": plan["definition_hash"],
                      "resources": plan["resources"], "steps": {s["id"]: {"status": "pending"} for s in plan["stages"]}}
        snapshots = {}
        input_files = {str(Path(step["inputs"][name]).resolve()) for step in plan["definition"]["steps"]
                       for name, expected in STAGES[step["operation"]][0].items() if expected == "file"}
        for resource, expected in record["resources"].items():
            if not resource.startswith("file:") or resource[5:] not in input_files:
                continue
            original = Path(resource[5:])
            snapshot = workspace.path / "workflows" / f"{run_id}.inputs" / f"{expected}{original.suffix}"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                shutil.copyfile(original, snapshot)
            if file_hash(snapshot) != expected:
                raise ContractError(f"Workflow input changed while creating snapshot: {original}")
            snapshots[str(original)] = str(snapshot)
        record.update(status="running", updated_at=time.time())
        atomic_json(path, record)
        outputs = {}
        try:
            for index, step in enumerate(plan["definition"]["steps"]):
                context.check()
                stage_id, row = step["id"], record["steps"][step["id"]]
                resolved = {key: outputs[value["stage"]] if isinstance(value, dict) else value
                            for key, value in step.get("inputs", {}).items()}
                for name, expected_type in STAGES[step["operation"]][0].items():
                    if expected_type == "file":
                        resolved[name] = snapshots[str(Path(resolved[name]).resolve())]
                signature = canonical_hash({"operation": step["operation"], "inputs": resolved, "config": step.get("config", {})})
                marker = {"run_id": run_id, "stage_id": stage_id, "request_hash": signature}
                # Publication may have succeeded immediately before a process exit.
                # Only adopt an exact immutable provenance match, never a guessed active artifact.
                if row["status"] != "completed":
                    matches = []
                    for artifact_id, artifact_row in workspace.manifest["artifacts"].items():
                        if artifact_row["status"] != "verified":
                            continue
                        meta_path = workspace.path / artifact_row["path"] / "artifact.json"
                        if read_json(meta_path).get("workflow") == marker:
                            _, metadata = workspace.resolve(artifact_id)
                            matches.append((artifact_id, canonical_hash(metadata)))
                    if len(matches) > 1:
                        raise ContractError(f"Ambiguous committed workflow stage: {stage_id}")
                    if matches:
                        row.update(status="completed", artifact_id=matches[0][0], metadata_hash=matches[0][1],
                                   request_hash=signature, recovered_publication=True)
                if row["status"] == "completed":
                    _, metadata = workspace.resolve(row["artifact_id"])
                    if canonical_hash(metadata) != row["metadata_hash"] or row.get("request_hash") != signature:
                        raise ContractError(f"Completed stage changed: {stage_id}")
                    if step["operation"] == "import_artifact":
                        expected = plan["resources"][f"external:{Path(resolved['artifact_path']).resolve()}"]
                        if canonical_hash(metadata) != expected:
                            raise ContractError(f"Completed import does not match its selected source: {stage_id}")
                    elif metadata.get("workflow") != marker:
                        raise ContractError(f"Completed stage provenance mismatch: {stage_id}")
                    outputs[stage_id] = row["artifact_id"]
                    context.progress("workflow", workflow_run=run_id, step=stage_id, step_status="reused_verified", index=index + 1, total=len(record["steps"]))
                    continue
                row.update(status="running", started_at=time.time(), request_hash=signature)
                atomic_json(path, record)
                context.progress("workflow", workflow_run=run_id, step=stage_id, step_status="running", index=index + 1, total=len(record["steps"]))
                workspace.workflow_stage = marker
                try:
                    settings = copy.deepcopy(step.get("config", {}))
                    if settings.get("resume_checkpoint"):
                        checkpoint = Path(settings["resume_checkpoint"]).resolve()
                        settings["resume_checkpoint_sha256"] = record["resources"][f"file:{checkpoint}"]
                    artifact_id = execute_stage(workspace, step["operation"], resolved, settings, context)
                finally:
                    workspace.workflow_stage = None
                status = workspace.manifest["artifacts"][artifact_id]["status"]
                row.update(status="completed" if status == "verified" else "rejected", artifact_id=artifact_id, finished_at=time.time())
                if status != "verified":
                    record.update(status="rejected", updated_at=time.time())
                    atomic_json(path, record)
                    return {"status": "rejected", "workflow_run": run_id, "record_path": str(path), "steps": record["steps"], "artifacts": outputs}
                _, metadata = workspace.resolve(artifact_id)
                row["metadata_hash"] = canonical_hash(metadata)
                outputs[stage_id] = artifact_id
                atomic_json(path, record)
            record.update(status="completed", updated_at=time.time())
            atomic_json(path, record)
            return {"status": "verified", "workflow_run": run_id, "record_path": str(path), "steps": record["steps"], "artifacts": outputs}
        except BaseException as exc:
            record.update(status="cancelled" if isinstance(exc, Cancelled) else "interrupted", updated_at=time.time(),
                          error={"type": type(exc).__name__, "message": str(exc)})
            atomic_json(path, record)
            raise
