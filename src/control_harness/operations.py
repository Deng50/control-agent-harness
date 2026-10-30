"""One domain dispatch shared by Harness tools and deterministic acceptance scripts."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .artifacts import Workspace, atomic_json, read_json, file_hash
from .contracts import ContractError
from .control import design_controller
from .data import register_transitions, simulate_transitions
from .examples import example
from .modeling import register_model
from .policy import collect_expert, optimize_policy, train_imitation
from .residual import train_residual
from .runtime import OperationContext

OPERATIONS = ["inspect_workspace", "inspect_artifact", "import_artifact", "register_model", "import_transitions", "simulate_transitions", "train_residual",
              "design_controller", "collect_expert", "train_imitation", "optimize_policy", "demo",
              "plan_workflow", "run_workflow", "validate_estimator", "identify_linear_model", "inspect_model_source",
              "prepare_handoff", "read_artifact_file", "recover_interrupted_stage", "register_python_model"]


def run_operation(request: dict, context: OperationContext | None = None) -> dict:
    context = context or OperationContext()
    operation = request["operation"]
    if operation not in OPERATIONS:
        raise ContractError(f"Unknown operation {operation}")
    workspace = Workspace(request["workspace"])
    config = request.get("config", {})
    inputs = request.get("inputs", {})
    if operation == "inspect_workspace":
        return workspace.manifest
    if operation == "prepare_handoff":
        from .inspection import prepare_handoff
        return prepare_handoff(workspace, inputs["artifact_id"])
    if operation == "read_artifact_file":
        from .inspection import read_artifact_file
        return read_artifact_file(workspace, inputs["artifact_id"], inputs["file"], config)
    if operation == "recover_interrupted_stage":
        from .recovery import recover_interrupted_stage
        return recover_interrupted_stage(workspace, inputs["stage_id"], context)
    if operation == "inspect_model_source":
        from .model_inspection import inspect_model_source
        return inspect_model_source(inputs["model_path"], context)
    if operation == "inspect_artifact":
        from .inspection import report_preview
        root, metadata = workspace.resolve(inputs["artifact_id"], allow_stale=True, allow_rejected=True)
        return {"artifact_id": metadata["id"], "kind": metadata["kind"], "status": workspace.manifest["artifacts"][metadata["id"]]["status"],
                "artifact_path": str(root), "metadata": metadata,
                "report": report_preview(root / "report.json") if "report.json" in metadata["files"] else None,
                "files": [{"name": name, "path": str(root / name)} for name in metadata["files"]]}
    if operation == "demo":
        with workspace.lock("demo"):
            return run_demo(workspace, config.get("case", "linear"), context)
    if operation in {"plan_workflow", "run_workflow"}:
        from .workflow import plan_workflow, run_workflow
        return (plan_workflow(workspace, inputs, config) if operation == "plan_workflow"
                else run_workflow(workspace, inputs, config, context))
    with workspace.lock(operation):
        artifact_id = execute_stage(workspace, operation, inputs, config, context)
    row = workspace.manifest["artifacts"][artifact_id]
    return {"artifact_id": artifact_id, "kind": row["kind"], "status": row["status"], "workspace": str(workspace.path),
            "artifact_path": str(workspace.path / row["path"])}


def execute_stage(workspace: Workspace, operation: str, inputs: dict, config: dict, context: OperationContext) -> str:
    """Shared numerical stage dispatch. Caller owns the operation-wide workspace lock."""
    if operation == "import_artifact":
        artifact_id = workspace.import_artifact(inputs["artifact_path"])
    elif operation == "register_python_model":
        from .python_model import register_python_model
        artifact_id = register_python_model(workspace, Path(inputs["model_path"]), config["source"], config.get("model_spec"))
    elif operation == "register_model":
        path = Path(inputs["model_path"])
        spec = config.get("model_spec")
        edited = spec is not None
        if not edited:
            spec = read_json(path)
        source = {**config["source"], "input_path": str(path.resolve()), "input_sha256": file_hash(path),
                  "explicit_model_override": edited}
        artifact_id = register_model(workspace, spec, source)
    elif operation == "import_transitions":
        with np.load(Path(inputs["data_path"]), allow_pickle=False) as loaded:
            data = {k: loaded[k] for k in loaded.files}
        artifact_id = register_transitions(workspace, inputs["nominal_model"], data, config["source"],
                                           int(config.get("seed", 42)), inputs.get("reference_model"))
    elif operation == "simulate_transitions":
        artifact_id = simulate_transitions(workspace, inputs["nominal_model"], inputs["reference_model"], config, context)
    elif operation == "train_residual":
        artifact_id = train_residual(workspace, inputs["nominal_model"], inputs["dataset"], config, context)
    elif operation == "design_controller":
        artifact_id = design_controller(workspace, inputs["model"], inputs["evaluation_model"], config, context)
    elif operation == "collect_expert":
        artifact_id = collect_expert(workspace, inputs["controller"], inputs["evaluation_model"], config, context)
    elif operation == "train_imitation":
        artifact_id = train_imitation(workspace, inputs["dataset"], config, context)
    elif operation == "optimize_policy":
        artifact_id = optimize_policy(workspace, inputs["policy"], inputs["model"], config, context)
    elif operation == "validate_estimator":
        from .estimation import validate_estimator
        artifact_id = validate_estimator(workspace, inputs["model"], inputs["controller"], inputs["evaluation_model"], config, context)
    elif operation == "identify_linear_model":
        from .identification import identify_linear_model
        artifact_id = identify_linear_model(workspace, inputs["nominal_model"], inputs["dataset"], config, context)
    else:
        raise AssertionError(operation)
    return artifact_id


def run_demo(workspace: Workspace, case: str, context: OperationContext) -> dict:
    if workspace.manifest["artifacts"]:
        raise ContractError("Use a new workspace for a demonstration; previous experiments will not be overwritten")
    config = example(case)
    atomic_json(workspace.path / "demo-config.json", config)
    artifacts = {}
    artifacts["nominal"] = register_model(workspace, config["nominal"], {"kind": "user_nominal", "description": "Declared demonstration nominal model"})
    artifacts["reference"] = register_model(workspace, config["reference"], {"kind": "reference_simulator", "description": "Known separate demonstration simulator; not experimental data"})
    artifacts["data"] = simulate_transitions(workspace, artifacts["nominal"], artifacts["reference"], config["data"], context)
    artifacts["corrected"] = train_residual(workspace, artifacts["nominal"], artifacts["data"], config["residual"], context)
    artifacts["controller"] = design_controller(workspace, artifacts["corrected"], artifacts["reference"], config["controller"], context)
    artifacts["expert"] = collect_expert(workspace, artifacts["controller"], artifacts["reference"], config["collection"], context)
    artifacts["imitation"] = train_imitation(workspace, artifacts["expert"], config["imitation"], context)
    artifacts["optimized"] = optimize_policy(workspace, artifacts["imitation"], artifacts["corrected"], config["optimization"], context)
    statuses = {key: workspace.manifest["artifacts"][value]["status"] for key, value in artifacts.items()}
    summary = {"case": case, "source": "explicit_demonstration_simulators", "artifacts": artifacts,
               "status": "verified" if all(value == "verified" for value in statuses.values()) else "rejected",
               "stage_statuses": statuses}
    atomic_json(workspace.path / "demo-result.json", summary)
    return summary
