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
              "prepare_handoff", "read_artifact_file", "read_report_arrays", "prepare_bundle_archive", "recover_interrupted_stage", "register_python_model",
              "inspect_material", "register_material", "inspect_table", "prepare_supervised_dataset",
              "propose_case_plan", "read_case_plan", "register_plant", "recommend_sampling", "sample_model", "train_supervised", "train_predictive_model", "synthesize_mpc", "tune_controller",
              "register_expert", "collect_external_expert", "import_expert_data",
              "train_external_imitation", "prepare_imitation_evaluation", "evaluate_external_imitation",
              "prepare_pinn_problem", "train_pinn", "validate_differentiable_model", "optimize_external_policy", "evaluate_external_pgac"]


def run_operation(request: dict, context: OperationContext | None = None) -> dict:
    context = context or OperationContext()
    operation = request["operation"]
    if operation not in OPERATIONS:
        raise ContractError(f"Unknown operation {operation}")
    workspace = Workspace(request["workspace"])
    config = request.get("config", {})
    inputs = request.get("inputs", {})
    if operation == "inspect_material":
        from .materials import inspect_material
        from .material_registration import material_options
        return inspect_material(inputs["local_path"], context=context, **material_options(config))
    if operation == "inspect_table":
        from .datasets import inspect_table
        return inspect_table(inputs["data_path"], **{key: config[key] for key in
            ("worksheet", "header_row", "preview_rows", "delimiter", "encoding") if key in config})
    if operation == "propose_case_plan":
        from .case_plans import propose_plan
        return propose_plan(workspace, config["scope"], config["payload"], config.get("bindings"), config.get("dependencies"))
    if operation == "read_case_plan":
        from .case_plans import read_plan
        return read_plan(workspace, inputs.get("plan_id"), config.get("scope"))
    if operation == "recommend_sampling":
        from .plants import recommend_sampling
        return recommend_sampling(workspace, inputs["model"], config)
    if operation == "prepare_imitation_evaluation":
        from .external_policy_evaluation import prepare_external_imitation_evaluation
        return prepare_external_imitation_evaluation(workspace, inputs["policy"], config)
    # Existing standalone numerical scripts remain compatible. All native Host
    # executions carry a request-bound receipt; new data-only semantics always
    # require a trusted confirmation context, including direct Python callers.
    workflow_case = None
    trusted_case_context = bool(getattr(context, "_case_receipt_body", None) or getattr(context, "_test_fixture_workspace", None))
    if operation in {"prepare_supervised_dataset", "sample_model", "train_supervised", "train_predictive_model", "synthesize_mpc", "tune_controller", "collect_external_expert", "import_expert_data",
                     "train_external_imitation", "evaluate_external_imitation", "prepare_pinn_problem", "train_pinn",
                     "validate_differentiable_model", "optimize_external_policy", "evaluate_external_pgac"} or trusted_case_context:
        from .case_requirements import confirmation_requirements, resolve_workflow_case
        if operation == "run_workflow":
            workflow_case = resolve_workflow_case(workspace, request)
        context.require_case_confirmation(request, confirmation_requirements(workspace, request, workflow_case=workflow_case))
    if operation == "inspect_workspace":
        return workspace.manifest
    if operation == "prepare_handoff":
        from .inspection import prepare_handoff
        return prepare_handoff(workspace, inputs["artifact_id"])
    if operation == "read_artifact_file":
        from .inspection import read_artifact_file
        return read_artifact_file(workspace, inputs["artifact_id"], inputs["file"], config)
    if operation == "read_report_arrays":
        from .reporting import read_report_arrays
        return read_report_arrays(workspace, inputs["artifact_id"], inputs["file"], config)
    if operation == "prepare_bundle_archive":
        from .reporting import prepare_bundle_archive
        if config:
            raise ContractError("Bundle download does not accept configuration overrides")
        return prepare_bundle_archive(workspace, inputs["artifact_id"])
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
        if workflow_case is not None:
            from .case_requirements import freeze_workflow_case
            with workspace.lock("confirmed_workflow"):
                frozen = freeze_workflow_case(workspace, workflow_case,
                    workspace.path / ".case-plans" / "workflow-cache", context)
                return run_workflow(workspace, frozen["inputs"], frozen["config"], context)
        return (plan_workflow(workspace, inputs, config) if operation == "plan_workflow"
                else run_workflow(workspace, inputs, config, context))
    with workspace.lock(operation):
        if operation in {"sample_model", "collect_external_expert", "evaluate_external_imitation", "evaluate_external_pgac"}:
            if operation == "sample_model":
                from .plants import sample_model
                artifact_id = sample_model(workspace, inputs["model"], config, context, request)
            elif operation == "collect_external_expert":
                from .external_expert import collect_external_expert
                artifact_id = collect_external_expert(workspace, inputs["expert"], inputs["plant"], config, context, request)
            elif operation == "evaluate_external_imitation":
                from .external_policy_evaluation import evaluate_external_imitation
                artifact_id = evaluate_external_imitation(workspace, inputs["policy"], config, context, request)
            else:
                from .external_pgac_evaluation import evaluate_external_pgac
                artifact_id = evaluate_external_pgac(workspace, inputs["candidate"], config, context, request)
            if isinstance(artifact_id, dict):
                return artifact_id
        else:
            artifact_id = execute_stage(workspace, operation, inputs, config, context)
    row = workspace.manifest["artifacts"][artifact_id]
    return {"artifact_id": artifact_id, "kind": row["kind"], "status": row["status"], "workspace": str(workspace.path),
            "artifact_path": str(workspace.path / row["path"])}


def execute_stage(workspace: Workspace, operation: str, inputs: dict, config: dict, context: OperationContext) -> str:
    """Shared numerical stage dispatch. Caller owns the operation-wide workspace lock."""
    if operation == "register_material":
        from .material_registration import register_material
        artifact_id = register_material(workspace, inputs["local_path"], config, context)
    elif operation == "register_expert":
        from .external_expert import register_expert
        artifact_id = register_expert(workspace, inputs["material"], config, context)
    elif operation == "import_expert_data":
        from .external_expert_data import import_expert_data
        artifact_id = import_expert_data(workspace, inputs["material"], config, context)
    elif operation == "train_external_imitation":
        from .external_imitation import train_external_imitation
        artifact_id = train_external_imitation(workspace, inputs["dataset"], config, context)
    elif operation == "prepare_pinn_problem":
        from .pinn_problem import prepare_pinn_problem
        artifact_id = prepare_pinn_problem(workspace, inputs["material"], config, context)
    elif operation == "train_pinn":
        from .pinn import train_pinn
        artifact_id = train_pinn(workspace, inputs["problem"], config, context)
    elif operation == "validate_differentiable_model":
        from .differentiable_validation import validate_differentiable_model
        artifact_id = validate_differentiable_model(workspace, inputs["evaluation_model"], inputs["model"], config, context,
                                                  reference_dataset_id=inputs["dataset"])
    elif operation == "optimize_external_policy":
        from .restricted_pgac import train_restricted_pgac
        artifact_id = train_restricted_pgac(workspace, inputs["policy"], inputs["model"], config, context,
                                            validation_id=inputs["validation"])
    elif operation == "evaluate_external_pgac":
        from .external_pgac_evaluation import evaluate_external_pgac
        artifact_id = evaluate_external_pgac(workspace, inputs["candidate"], config, context)
    elif operation == "evaluate_external_imitation":
        from .external_policy_evaluation import evaluate_external_imitation
        artifact_id = evaluate_external_imitation(workspace, inputs["policy"], config, context)
    elif operation == "register_plant":
        from .plants import register_plant
        artifact_id = register_plant(workspace, inputs["material"], config, context)
    elif operation == "prepare_supervised_dataset":
        from .material_registration import prepare_supervised_dataset
        artifact_id = prepare_supervised_dataset(workspace, inputs["material"], config, context)
    elif operation == "train_supervised":
        from .supervised import train_supervised
        artifact_id = train_supervised(workspace, inputs["dataset"], config, context)
    elif operation == "train_predictive_model":
        from .predictive import train_predictive_model
        artifact_id = train_predictive_model(workspace, inputs["dataset"], config, context)
    elif operation == "synthesize_mpc":
        from .design_only import synthesize_mpc
        artifact_id = synthesize_mpc(workspace, inputs, config, context)
    elif operation == "import_artifact":
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
    elif operation in {"design_controller", "tune_controller"}:
        if operation == "tune_controller" and not isinstance(config.get("evaluation"), dict):
            raise ContractError("Preference-driven tuning requires an explicit evaluation recipe, positive scales and optional hard gates")
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
