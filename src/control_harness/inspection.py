"""Verified artifact inspection and explicit cross-role input preparation."""
from __future__ import annotations

from pathlib import Path
import math
import zipfile
import numpy as np

from .artifacts import Workspace, canonical_hash, confined, read_json
from .contracts import ContractError, ModelInterface, validate_network_dimensions

MODEL_KINDS = {"nominal_model", "reference_model", "corrected_model", "identified_model", "learned_model"}


def report_preview(path: Path) -> dict:
    if path.stat().st_size > 64 * 1024 * 1024:
        return {"display_summary": {"omitted": "Report exceeds the 64 MiB preview budget; inspect the saved report locally."}}
    report = read_json(path)
    reductions = []

    def reduce(value, location="report"):
        if isinstance(value, dict):
            return {key: reduce(item, f"{location}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            cap = 240 if location.endswith(".trajectory") else 500 if location.endswith(".log") else 24 if location.endswith((".runs", ".comparison")) else None
            if cap and len(value) > cap:
                reductions.append({"field": location, "original_count": len(value), "displayed_count": cap})
                indexes = np.linspace(0, len(value) - 1, cap, dtype=int)
                value = [value[index] for index in indexes]
            return [reduce(item, f"{location}[]") for item in value]
        return value

    preview = reduce(report)
    if reductions:
        preview["display_summary"] = {"sampled_fields": reductions, "metrics_recomputed": False,
                                      "full_report": path.name}
    # Keep the native worker's bounded JSON-line protocol usable for large runs.
    import json
    if len(json.dumps(preview, ensure_ascii=False).encode("utf-8")) > 650000:
        return {key: value for key, value in preview.items() if key in {
            "algorithm", "acceptance", "test_used_for_training", "trained_epochs", "parameter_count"}} | {
            "display_summary": {"omitted": "Detailed report exceeds the native preview budget; use the saved report file."}}
    return preview


def prepare_handoff(workspace: Workspace, artifact_id: str) -> dict:
    root, metadata = workspace.resolve(artifact_id)
    if metadata["kind"] == "differentiable_validation":
        from .differentiable_validation import read_differentiable_validation
        report = read_differentiable_validation(workspace, artifact_id)
        return {"artifact_id": artifact_id, "kind": metadata["kind"], "status": "needs_input", "handoff": {
            "agent": "lightweight", "operation": "optimize_external_policy",
            "inputs": {"model": report["bindings"]["candidate"]["id"], "validation": artifact_id},
            "missing_inputs": ["policy"], "missing_config": ["Review the restricted PGAC training and frozen original-Plant evaluation configuration"],
            "interface": report["physical_interface"], "config_suggestions": {"domain": report["coverage"]["domain"]},
            "scope": "This receipt qualifies only the bound model and sampled validation/domain checks; a bounded surrogate retains proxy gradients and unproved joint coverage."},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "external_pgac_candidate":
        report = read_json(root / "report.json")
        frozen = report.get("original_evaluation_config")
        if not isinstance(frozen, dict):
            raise ContractError("PGAC candidate lacks its prospectively frozen original-Plant evaluation configuration")
        return {"artifact_id": artifact_id, "kind": metadata["kind"], "status": "needs_input", "handoff": {
            "agent": "lightweight", "operation": "evaluate_external_pgac", "inputs": {"candidate": artifact_id},
            "missing_inputs": [], "missing_config": ["Review the exact frozen original-Plant validation/test comparison before execution"],
            "interface": metadata["interface"], "config_suggestions": frozen,
            "scope": "Inactive numerical candidate. Original-Plant acceptance and any feasible baseline fallback are still pending."},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "external_optimized_policy":
        return {"artifact_id": artifact_id, "kind": metadata["kind"], "status": "complete", "handoff": {
            "agent": "lightweight", "operation": None, "inputs": {"policy": artifact_id}, "missing_inputs": [], "missing_config": [],
            "interface": metadata["interface"], "config_suggestions": {},
            "scope": "Read the original-Plant comparison and selection record to distinguish an accepted optimized candidate from a feasible retained imitation baseline; the bounded simulation result is not a stability or hardware certificate."},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "conditional_ode_problem":
        signals = metadata.get("source", {}).get("pinn_signals")
        if not isinstance(signals, dict):
            raise ContractError("Conditional ODE problem lacks its explicit equation/signal summary")
        return {"artifact_id": artifact_id, "kind": metadata["kind"], "status": "needs_input", "handoff": {
            "agent": "model", "operation": "train_pinn", "inputs": {"problem": artifact_id},
            "missing_inputs": [], "missing_config": ["Review and confirm the smooth conditional network, actual physics/initial loss weights and training summary"],
            "interface": signals, "config_suggestions": {"network": {"input_dim": 3, "output_dim": 1}},
            "scope": "Known bounded constant-forcing thermal ODE; explicit time, initial condition and driving input. Not a general PDE or Plant transition."},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "pinn_model":
        descriptor = read_json(root / "pinn.json")
        problem_id = descriptor.get("problem")
        if problem_id not in {row["id"] for row in metadata["parents"]}:
            raise ContractError("Conditional PINN must bind its exact ODE problem ancestor")
        _, problem = workspace.resolve(problem_id, kind="conditional_ode_problem")
        report = read_json(root / "report.json")
        return {"artifact_id": artifact_id, "kind": "pinn_model", "status": "complete", "handoff": {
            "agent": "model", "operation": None, "inputs": {"pinn_model": artifact_id}, "missing_inputs": [], "missing_config": [],
            "interface": problem.get("source", {}).get("pinn_signals"), "config_suggestions": {},
            "is_pinn": report.get("is_pinn"), "training_regime": report.get("training_regime"),
            "capabilities": {"conditional_trajectory": True, "constant_forcing_only": True,
                "arbitrary_state_transition": False, "automatic_mpc_pgac_handoff": False},
            "scope": "Independent bounded time/initial/constant-drive trajectory inference. No verified arbitrary one-step dynamics interface; MPC/PGAC handoff is unavailable."},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)},
                {"id": problem_id, "metadata_hash": canonical_hash(problem)}]}
    if metadata["kind"] == "material":
        descriptor = read_json(root / "material.json")
        tabular = descriptor["format"] in {"csv", "xlsx"}
        return {"artifact_id": artifact_id, "kind": "material", "status": "ready", "handoff": {
            "agent": "model", "operation": "inspect_table" if tabular else "inspect_model_source",
            "inputs": {"data_path" if tabular else "model_path": str(root / descriptor["primary_file"])},
            "missing_inputs": [], "missing_config": [], "interface": metadata["interface"], "config_suggestions": {}},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "supervised_dataset":
        from .contracts import DataInterface
        interface = DataInterface.parse(metadata["interface"])
        return {"artifact_id": artifact_id, "kind": "supervised_dataset", "status": "needs_input", "handoff": {
            "agent": "model", "operation": "train_supervised", "inputs": {"dataset": artifact_id}, "missing_inputs": [],
            "missing_config": ["Review and confirm the requested network and training summary"],
            "interface": interface.to_dict(), "config_suggestions": {"input_dim": interface.input_dim, "output_dim": interface.output_dim}},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "supervised_model":
        from .contracts import DataInterface
        interface = DataInterface.parse(metadata["interface"])
        return {"artifact_id": artifact_id, "kind": "supervised_model", "status": "complete", "handoff": {
            "agent": "model", "operation": None, "inputs": {}, "missing_inputs": [], "missing_config": [],
            "interface": interface.to_dict(), "config_suggestions": {},
            "scope": "Standalone supervised inference export. Physical state mapping and an executable dynamics contract are separate prerequisites for optional controller use."},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "external_expert":
        descriptor = read_json(root / "expert.json")
        return {"artifact_id":artifact_id,"kind":"external_expert","status":"needs_input","handoff":{
            "agent":"lightweight","operation":"collect_external_expert","inputs":{"expert":artifact_id,"plant":""},
            "missing_inputs":["plant"],"missing_config":["Explicit scenarios, initial previous actions and confirmed observation/collection summary"],
            "interface":metadata["interface"],"expert_adapter":descriptor["adapter_spec"],"config_suggestions":{},
            "scope":"Choose an explicit compatible plant; do not redesign or invent an internal MPC controller."},
            "source_versions":[{"id":artifact_id,"metadata_hash":canonical_hash(metadata)}]}
    if metadata["kind"] == "external_expert_dataset":
        from .contracts import DataInterface
        interface = DataInterface.parse(metadata["interface"])
        return {"artifact_id":artifact_id,"kind":"external_expert_dataset","status":"data_ready","handoff":{
            "agent":"lightweight","operation":"train_external_imitation","inputs":{"dataset":artifact_id},"missing_inputs":[],
            "missing_config":["Review and confirm network/training and predeclared action-error metrics"],
            "interface":interface.to_dict(),"config_suggestions":{"network":{"input_dim":interface.input_dim,"output_dim":interface.output_dim}},
            "scope":"Causal external expert recordings; network training and physical closed-loop verification are separate stages."},
            "source_versions":[{"id":artifact_id,"metadata_hash":canonical_hash(metadata)}]}
    if metadata["kind"] == "external_imitation_policy":
        from .contracts import DataInterface
        interface = DataInterface.parse(metadata["interface"])
        descriptor = read_json(root / "policy.json")
        dataset_id = descriptor.get("external_expert_dataset")
        if dataset_id not in {parent["id"] for parent in metadata["parents"]}:
            raise ContractError("External imitation policy must bind its exact external expert dataset")
        _, source = workspace.resolve(dataset_id, kind="external_expert_dataset")
        return {"artifact_id":artifact_id,"kind":"external_imitation_policy","status":"complete","handoff":{
            "agent":"lightweight","operation":None,"inputs":{"policy":artifact_id},"missing_inputs":[],"missing_config":[],
            "interface":interface.to_dict(),"config_suggestions":{},
            "scope":"Requested imitation export is complete. Offline recorded-action error is separate from optional autonomous evaluation; PGAC is not automatic."},
            "optional_handoffs":[{"agent":"lightweight","operation":"prepare_imitation_evaluation","inputs":{"policy":artifact_id},
                "missing_inputs":[],"missing_config":[],"config_suggestions":{},
                "scope":"Read-only preparation resolves concrete independent original-Plant conditions or reports not_applicable; no evaluation runs until reviewed and confirmed."}],
            "source_versions":[{"id":artifact_id,"metadata_hash":canonical_hash(metadata)},
                               {"id":dataset_id,"metadata_hash":canonical_hash(source)}]}
    if metadata["kind"] == "policy_evaluation":
        policy_id = metadata.get("source", {}).get("policy")
        if policy_id not in {parent["id"] for parent in metadata["parents"]}:
            raise ContractError("Policy evaluation must bind the exact evaluated policy")
        workspace.resolve(policy_id, kind="external_imitation_policy")
        report = read_json(root / "report.json")
        return {"artifact_id":artifact_id,"kind":"policy_evaluation","status":"complete","handoff":{
            "agent":"lightweight","operation":None,"inputs":{"evaluation":artifact_id,"policy":policy_id},"missing_inputs":[],
            "missing_config":[],"interface":metadata["interface"],"config_suggestions":{},
            "evaluation_status":report.get("status"),"scope":"Evaluation report only; policy activation is unchanged and no improvement or stability proof is implied."},
            "source_versions":[{"id":artifact_id,"metadata_hash":canonical_hash(metadata)}]}
    if metadata["kind"] == "controller_design":
        design = read_json(root / "design.json")
        optional = []
        prediction = design.get("prediction") or {}
        predictor_id = prediction.get("artifact_id")
        if design.get("model_instantiated") and predictor_id:
            if predictor_id not in {parent["id"] for parent in metadata["parents"]}:
                raise ContractError("Design predictor is not a bound artifact dependency")
            workspace.resolve(predictor_id)
            optional.append({"agent":"controller", "operation":"tune_controller",
                "inputs":{"model":predictor_id,"evaluation_model":""},"missing_inputs":["evaluation_model"],
                "missing_config":["evaluation","tuning_scenarios","verification_scenarios","test_scenarios"],
                "config_suggestions":{"parameters":{**design.get("controller_config",{}),**design.get("weights",{})}},
                "scope":"Optional preference tuning requires a separately selected compatible executable evaluation model and independent conditions."})
        return {"artifact_id": artifact_id, "kind": "controller_design", "status": "design_complete", "handoff": {
            "agent": "controller", "operation": None, "inputs": {"design": artifact_id}, "missing_inputs": [],
            "missing_config": [], "config_suggestions": {}, "interface": metadata["interface"],
            "scope": "Design delivery only. Parameterization, numerical solve checks and closed-loop evaluation are separate states.",
            "design_status": design.get("status"), "model_instantiated": design.get("model_instantiated", False)},
            "optional_handoffs": optional,
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "plant_model":
        return {"artifact_id": artifact_id, "kind": "plant_model", "status": "ready", "handoff": {
            "agent": "model", "operation": "recommend_sampling", "inputs": {"model": artifact_id},
            "missing_inputs": [], "missing_config": [], "interface": metadata["interface"], "config_suggestions": {}},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "sampling_trial":
        config = read_json(root / "config.json")
        model = metadata["source"]["model"]
        if model not in [item["id"] for item in metadata["parents"]]:
            raise ContractError("Sampling trial does not bind its declared plant")
        workspace.resolve(model)
        return {"artifact_id": artifact_id, "kind": "sampling_trial", "status": "ready", "handoff": {
            "agent": "model", "operation": "sample_model", "inputs": {"model": model},
            "missing_inputs": [], "missing_config": [], "interface": metadata["interface"],
            "config_suggestions": {**config, "sampling_run": "full"}},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "sampled_dataset":
        interface = ModelInterface.parse(metadata["interface"])
        suggestions = {"network": {"input_dim": interface.nx + interface.nu + interface.nd, "output_dim": interface.nx}}
        model_id = metadata.get("source", {}).get("model")
        if model_id not in [row["id"] for row in metadata["parents"]]:
            raise ContractError("Sampled dataset does not bind its declared plant")
        model_root, model_meta = workspace.resolve(model_id)
        C = None
        if model_meta["kind"] == "plant_model":
            C = read_json(model_root / "plant.json")["adapter_spec"].get("C")
        elif "model.json" in model_meta["files"]:
            C = read_json(model_root / "model.json").get("C")
        elif "nominal.json" in model_meta["files"]:
            C = read_json(model_root / "nominal.json").get("C")
        if C is not None:
            suggestions["C"] = C
        return {"artifact_id": artifact_id, "kind": "sampled_dataset", "status": "needs_input", "handoff": {
            "agent": "model", "operation": "train_predictive_model", "inputs": {"dataset": artifact_id}, "missing_inputs": [],
            "missing_config": ["Review and confirm the prediction training summary"] + ([] if C is not None else ["Explicit state-only output matrix C consistent with sampled outputs"]),
            "interface": metadata["interface"], "config_suggestions": suggestions},
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    selected_interface = ModelInterface.parse(metadata["interface"])
    ancestors = {}

    def visit(item_id):
        if item_id in ancestors:
            return
        item_root, item = workspace.resolve(item_id)
        # Learned physical models carry supervised datasets and immutable source
        # materials in their provenance closure. Only physical interfaces share
        # nx/nu/ny; dimensionless/data interfaces must retain their own meaning.
        if item["interface"].get("interface_kind") is None:
            selected_interface.compatible(ModelInterface.parse(item["interface"]))
        ancestors[item_id] = (item_root, item)
        for parent in item["parents"]:
            visit(parent["id"])

    visit(artifact_id)

    def bound(value, kinds):
        if value not in ancestors or ancestors[value][1]["kind"] not in kinds:
            raise ContractError("Selected artifact descriptor refers to an incompatible or unbound dependency")
        return value

    def reference_model():
        # The source of the selected training data is evidence. A workspace's
        # active reference_model is not evidence about this selected result.
        candidates = set()
        for _, item in ancestors.values():
            if item["kind"] == "sampled_dataset":
                original = item.get("source", {}).get("model")
                if original in ancestors and ancestors[original][1]["kind"] in MODEL_KINDS:
                    candidates.add(bound(original, MODEL_KINDS))
                continue
            if item["kind"] != "transition_dataset":
                continue
            source = item["source"]
            if source.get("kind") == "reference_simulator":
                if source.get("reference_artifact"):
                    candidates.add(bound(source["reference_artifact"], MODEL_KINDS))
                else:
                    candidates.update(p["id"] for p in item["parents"] if ancestors[p["id"]][1]["kind"] == "reference_model")
        return next(iter(candidates)) if len(candidates) == 1 else None

    kind = metadata["kind"]
    suggestions, missing_config, optional_handoffs = {}, [], []
    if kind in {"nominal_model", "corrected_model", "identified_model", "learned_model"}:
        agent, operation = "controller", "synthesize_mpc"
        inputs = {"model": artifact_id}
        suggestions = {"prediction": {"method": "direct"}}
        evaluation = reference_model()
        optional_handoffs.append({"agent": "controller", "operation": "tune_controller",
            "inputs": {"model": artifact_id, "evaluation_model": evaluation or ""},
            "missing_inputs": [] if evaluation else ["evaluation_model"],
            "missing_config": ["evaluation", "tuning_scenarios", "verification_scenarios", "test_scenarios"],
            "scope": "Optional external-preference tuning; not required for a design-only request"})
    elif kind == "controller":
        agent, operation = "lightweight", "collect_expert"
        descriptor = read_json(root / "controller.json")
        inputs = {"controller": artifact_id, "evaluation_model": bound(descriptor["evaluation_model"], MODEL_KINDS)}
        missing_config = ["scenarios (at least five independent conditions)"]
    elif kind == "expert_dataset":
        agent, operation = "lightweight", "train_imitation"
        inputs = {"dataset": artifact_id}
        input_dim, output_dim = validate_network_dimensions(selected_interface, {}, residual=False)
        suggestions = {"architecture": "mlp", "input_dim": input_dim, "output_dim": output_dim}
    elif kind == "imitation_policy":
        agent, operation = "lightweight", "optimize_policy"
        descriptor = read_json(root / "policy.json")
        controller_id = bound(descriptor["expert_controller"], {"controller"})
        controller_root = ancestors[controller_id][0]
        controller = read_json(controller_root / "controller.json")
        inputs = {"policy": artifact_id, "model": bound(controller["prediction_model"], MODEL_KINDS)}
    elif kind == "transition_dataset":
        agent, operation = "model", "train_residual"
        nominal = [p["id"] for p in metadata["parents"] if ancestors[p["id"]][1]["kind"] == "nominal_model"]
        if len(nominal) != 1:
            raise ContractError("Dataset must identify one exact nominal model")
        inputs = {"dataset": artifact_id, "nominal_model": nominal[0]}
        input_dim, output_dim = validate_network_dimensions(selected_interface, {}, residual=True)
        suggestions = {"input_dim": input_dim, "output_dim": output_dim}
    else:
        raise ContractError(f"No automatic next-stage mapping for {kind}; choose an explicit operation")
    missing = [key for key, value in inputs.items() if value is None]
    return {"artifact_id": artifact_id, "kind": kind,
            "status": "needs_input" if missing or missing_config else "ready",
            "handoff": {"agent": agent, "operation": operation,
                        "inputs": {key: value or "" for key, value in inputs.items()},
                        "missing_inputs": missing, "missing_config": missing_config,
                        "config_suggestions": suggestions, "interface": selected_interface.to_dict()},
            "source_versions": [{"id": item_id, "metadata_hash": canonical_hash(item)} for item_id, (_, item) in ancestors.items()],
            "optional_handoffs": optional_handoffs}


def read_artifact_file(workspace: Workspace, artifact_id: str, name: str, config: dict) -> dict:
    root, metadata = workspace.resolve(artifact_id, allow_stale=True, allow_rejected=True)
    if not isinstance(name, str) or name not in metadata["files"]:
        raise ContractError("Select a file from this artifact's verified file list")
    limit = config.get("max_chars", 16000)
    if type(limit) is not int or not 1 <= limit <= 32000:
        raise ContractError("max_chars must be an integer between 1 and 32000")
    path = confined(root, name)
    preview = {"name": name, "bytes": path.stat().st_size, "sha256": metadata["files"][name],
               "format": "binary", "truncated": False}
    if path.suffix.lower() in {".json", ".jsonl", ".md", ".txt", ".csv", ".py", ".m"}:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            text = stream.read(limit + 1)
        preview.update(format="text", text=text[:limit], truncated=len(text) > limit)
    elif path.suffix.lower() == ".npz":
        # Inspect bounded numeric datasets without object/pickle deserialization.
        with zipfile.ZipFile(path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
                preview.update(format="array_archive", text="Dataset exceeds the 64 MiB preview budget; use the local file for analysis.", truncated=True)
                return {"artifact_id": artifact_id, "kind": metadata["kind"],
                        "status": workspace.manifest["artifacts"][artifact_id]["status"], "file_preview": preview}
            entries = archive.infolist()
            if len({item.filename for item in entries}) != len(entries):
                raise ContractError("Dataset preview does not accept duplicate archive entries")
            # A small ZIP member can advertise a huge ndarray shape. Check the
            # NPY header before np.load is allowed to allocate that array.
            for item in entries[:32]:
                if not item.filename.endswith(".npy"):
                    raise ContractError("Dataset preview requires NPY array members")
                with archive.open(item) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ContractError("Unsupported NPY header version for bounded preview")
                    shape, _, dtype = reader(stream)
                    if dtype.hasobject:
                        raise ValueError("Object arrays cannot be previewed without pickle")
                    payload_bytes = math.prod(shape) * dtype.itemsize
                    if payload_bytes > 64 * 1024 * 1024 or stream.tell() + payload_bytes != item.file_size:
                        raise ContractError("NPY shape/dtype exceeds the preview budget or disagrees with stored bytes")
        arrays = []
        with np.load(path, allow_pickle=False) as data:
            for key in data.files[:32]:
                value = data[key]
                row = {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
                if np.issubdtype(value.dtype, np.number) and value.size:
                    finite = value[np.isfinite(value)]
                    if finite.size and not np.iscomplexobj(finite):
                        row.update(min=float(finite.min()), max=float(finite.max()))
                    row["nonfinite"] = int(value.size - finite.size)
                arrays.append(row)
            preview.update(format="array_archive", arrays=arrays, truncated=len(data.files) > 32)
    return {"artifact_id": artifact_id, "kind": metadata["kind"],
            "status": workspace.manifest["artifacts"][artifact_id]["status"], "file_preview": preview}
