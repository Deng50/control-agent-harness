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


def artifact_capabilities(workspace: Workspace, artifact_id: str, *, _closure=None) -> dict:
    """从选中分支的描述和文件推导可选入口；不运行上传代码，也不产生运行证明。"""
    closure = _closure if _closure is not None else workspace.resolve_closure(artifact_id)
    root, meta = closure[artifact_id]
    kind, members = meta["kind"], meta["files"]
    representation, backend, missing = None, None, []
    prediction = False
    if kind in MODEL_KINDS:
        required = {"nominal_model": ["model.json"], "reference_model": ["model.json"], "identified_model": ["model.json"],
                    "corrected_model": ["nominal.json", "network.json", "scalers.json", "model.pt"],
                    "learned_model": ["learned.json", "model.pt", "features.json", "scaler.json", "network_config.json", "data_interface.json"]}[kind]
        if "model.json" in members:
            representation = read_json(root / "model.json").get("representation")
            if representation in {"python_torch_derivative", "python_torch_transition"}:
                required = [*required, "source.py", "python-validation.json"]
            if representation in {"continuous_polynomial", "python_torch_derivative"}:
                required = [*required, "transition_adapter.json"]
        else:
            representation = {"corrected_model": "residual", "learned_model": "learned_state_transition"}.get(kind)
        missing = [name for name in required if name not in members]
        prediction = not missing and representation in {"discrete_polynomial", "continuous_polynomial", "python_torch_transition",
                                                         "python_torch_derivative", "residual", "learned_state_transition"}
    if kind == "plant_model":
        if "plant.json" in members:
            from .plants import validate_plant_spec
            definition = read_json(root / "plant.json")
            spec = validate_plant_spec(definition["adapter_spec"])
            ModelInterface.parse(meta["interface"]).compatible(ModelInterface.parse(spec["interface"]))
            material_id = definition.get("material")
            if material_id not in {parent["id"] for parent in meta["parents"]} or closure[material_id][1]["kind"] != "material":
                raise ContractError("对象描述中的来源材料与已验证的父产物不一致")
            material_root, material = closure[material_id]
            if (spec["source_file"] not in material["files"] or
                    read_json(material_root / "material.json")["content_hash"] != definition.get("material_content_hash")):
                raise ContractError("对象适配器与原材料快照的内容版本不一致")
            backend = spec["backend"]
        else:
            missing = ["plant.json"]
    descriptor = read_json(root / "controller.json") if kind == "controller" and "controller.json" in members else {}
    if kind == "controller_design":
        missing = [name for name in ("design.json", "bundle/bundle_manifest.json") if name not in members]
    if kind == "controller" and not descriptor:
        missing = ["controller.json"]
    design = kind == "controller_design" and not missing
    for field in ("prediction_model", "evaluation_model"):
        bound = descriptor.get(field)
        if bound is not None and (not isinstance(bound, str) or bound not in closure or closure[bound][1]["kind"] not in MODEL_KINDS):
            raise ContractError(f"控制器的 {field} 未绑定可追溯的预测模型版本")
    predictor = descriptor.get("prediction_model")
    executable = (kind == "controller" and isinstance(descriptor.get("parameters"), dict) and bool(predictor)
                  and artifact_capabilities(workspace, predictor, _closure=closure)["capabilities"]["prediction"])
    from .plants import LEGACY_MODELS
    capabilities = {"sampling": bool(backend) or (prediction and kind in LEGACY_MODELS), "prediction": prediction,
                    "original_plant": bool(backend), "reference_dynamics": prediction and kind == "reference_model",
                    "controller_design": design, "executable_controller": executable,
                    "expert_dataset": (kind == "external_expert_dataset" and "expert.npz" in members) or
                                     (kind == "expert_dataset" and "expert.npz" in members)}
    return {"artifact_id": artifact_id, "kind": kind, "workspace": str(workspace.path),
            "project_id": workspace.manifest["project_id"], "capabilities": capabilities,
            "evidence": "verified_files_and_descriptor", "load_check": "not_run",
            "backend": backend, "representation": representation, "missing_files": missing, "interface": meta["interface"],
            "source_versions": [{"id": item["id"], "kind": item["kind"], "metadata_hash": canonical_hash(item)}
                                for _, item in closure.values()]}


def prepare_handoff(workspace: Workspace, artifact_id: str) -> dict:
    """交接只返回精确版本和下一步草案；迁入工作区与下一角色仍要经过各自会话的门禁。"""
    closure = workspace.resolve_closure(artifact_id)
    root, metadata = closure[artifact_id]
    info = artifact_capabilities(workspace, artifact_id, _closure=closure)
    result = _handoff_routes(workspace, artifact_id, root, metadata, closure, info)
    result.update(source_versions=info["source_versions"], artifact_capabilities=info["capabilities"],
                  transfer={"mode": "explicit_artifact_closure", "workspace": str(workspace.path),
                            "project_id": info["project_id"], "artifact_path": str(root),
                            "artifact_id": artifact_id, "metadata_hash": canonical_hash(metadata),
                            "target_session_review": "required_for_confirmed_operations"})
    return result


def _conversion_routes(plant_id, adapter, dataset_id=None, candidates=()):
    """按不可变来源给出转换草案；从对象出发时所有数据集都由用户显式选择。"""
    if adapter["backend"] != "python_callable":
        return []
    routes = []
    for method in ("declared_equations", "sampled_affine"):
        inputs = {"plant": plant_id, "dataset": dataset_id or ""}
        missing = [] if dataset_id else ["dataset"]
        if method == "declared_equations":
            inputs["equations"] = ""
            missing.append("equations")
            scope = "选择已登记且单位、采样周期、输出定义相符的方程，用原对象独立轨迹核验。"
        else:
            scope = "仅在确认采样覆盖范围内可用仿射近似时选择。独立轨迹检验单步和多步预测，保留原对象。"
        questions = [] if dataset_id else ["选择此对象版本的完整采样数据（含状态、动作、独立验证和测试轨迹）"]
        if method == "sampled_affine" and adapter.get("C") is None:
            questions.append("缺少输出矩阵 C。请补充状态到输出的已知线性映射并重新登记采样，或提供明确方程。")
        routes.append({"agent": "model", "operation": "convert_plant_predictor", "inputs": inputs,
                       "missing_inputs": missing, "missing_config": questions, "config_suggestions": {"method": method},
                       "candidate_datasets": list(candidates), "scope": scope,
                       "blocked": method == "sampled_affine" and adapter.get("C") is None})
    return routes


def _handoff_routes(workspace, artifact_id, root, metadata, closure, info):
    kind = metadata["kind"]
    needed = "prediction" if kind in MODEL_KINDS else {
        "controller": "executable_controller", "plant_model": "sampling", "controller_design": "controller_design"}.get(kind)
    if needed and not info["capabilities"][needed]:
        missing = info["missing_files"] or ["核对描述文件中的表示、模型来源及执行参数"]
        return {"artifact_id": artifact_id, "kind": kind, "status": "unsupported", "handoff": {
            "agent": "controller" if kind.startswith("controller") else "model", "operation": None, "inputs": {},
            "missing_inputs": [], "missing_config": missing, "scope": "该版本缺少对应执行接口的证据。"}}
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
            optional.append({"agent":"controller", "operation":"tune_controller",
                "inputs":{"model":predictor_id,"evaluation_model":""},"missing_inputs":["evaluation_model"],
                "missing_config":["evaluation","tuning_scenarios","verification_scenarios","test_scenarios"],
                "config_suggestions":{"parameters":{**design.get("controller_config",{}),**design.get("weights",{})}},
                "scope":"Optional preference tuning requires a separately selected compatible executable evaluation model and independent conditions."})
        spec = prediction.get("spec") or {}
        portable = spec.get("representation") in {"continuous_polynomial", "discrete_polynomial"}
        if design.get("model_instantiated") and portable:
            physical = ModelInterface.parse(design["physical_interface"])
            parent_ok = predictor_id and closure[predictor_id][1]["kind"] in {"nominal_model", "reference_model", "identified_model"}
            targets = ["standalone_python"] + (["workspace_controller"] if parent_ok else [])
            for target in targets:
                optional.append({"agent": "controller", "operation": "verify_design_delivery",
                    "inputs": {"design": artifact_id}, "missing_inputs": [],
                    "missing_config": ["case.x", "case.reference"] + (["case.disturbance"] if physical.nd else []),
                    "config_suggestions": {"target": target, "case": {}},
                    "scope": "请填写验收工况和用户要求的证据层级，可指定期望控制量与容差。"})
        return {"artifact_id": artifact_id, "kind": "controller_design", "status": "design_complete", "handoff": {
            "agent": "controller", "operation": None, "inputs": {"design": artifact_id}, "missing_inputs": [],
            "missing_config": design.get("required_parameters", []) + (["当前交付验收器支持声明式离散或连续多项式方程；此表示需要相应的运行校验器。"]
                                if design.get("model_instantiated") and not portable else []),
            "config_suggestions": {}, "interface": metadata["interface"],
            "scope": "Design delivery only. Parameterization, numerical solve checks and closed-loop evaluation are separate states.",
            "design_status": design.get("status"), "model_instantiated": design.get("model_instantiated", False)},
            "optional_handoffs": optional,
            "source_versions": [{"id": artifact_id, "metadata_hash": canonical_hash(metadata)}]}
    if metadata["kind"] == "plant_model":
        adapter = read_json(root / "plant.json")["adapter_spec"]
        # 清单中的候选仅用于导航。选择后重新验证完整闭包；从单个对象不能推出唯一采集工况。
        candidates = [item_id for item_id, row in workspace.manifest["artifacts"].items()
                      if row["status"] == "verified" and row["kind"] == "sampled_dataset" and row["parents"] == [artifact_id]]
        return {"artifact_id": artifact_id, "kind": "plant_model", "status": "ready", "handoff": {
            "agent": "model", "operation": "recommend_sampling", "inputs": {"model": artifact_id},
            "missing_inputs": [], "missing_config": [], "interface": metadata["interface"], "config_suggestions": {}},
            "optional_handoffs": _conversion_routes(artifact_id, adapter, candidates=sorted(candidates))}
    if metadata["kind"] == "sampling_trial":
        config = read_json(root / "config.json")
        # 试采的确认和恢复句柄属于原会话。新工作区/角色要自行准备完整计划。
        config = {key: value for key, value in config.items() if key not in
                  {"plan_refs", "resume_run", "resume_checkpoint", "resume_checkpoint_sha256"}}
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
        conversion = []
        if model_meta["kind"] == "plant_model":
            adapter = read_json(model_root / "plant.json")["adapter_spec"]
            C = adapter.get("C")
            if ([parent["id"] for parent in metadata["parents"]] == [model_id]
                    and metadata["source"].get("kind") == "plant_simulator"
                    and metadata["source"].get("backend") == adapter["backend"]
                    and {"transitions.npz", "split.json"}.issubset(metadata["files"])):
                conversion = _conversion_routes(model_id, adapter, dataset_id=artifact_id)
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
            "optional_handoffs": conversion}
    if metadata["kind"] == "delivery_validation":
        return {"artifact_id": artifact_id, "kind": kind, "status": "complete", "handoff": {
            "agent": "controller", "operation": None, "inputs": {"validation": artifact_id},
            "missing_inputs": [], "missing_config": [], "scope": "验收记录的逐项结果见交付报告。"}}
    selected_interface = ModelInterface.parse(metadata["interface"])
    ancestors = closure

    def bound(value, kinds):
        if value not in ancestors or ancestors[value][1]["kind"] not in kinds:
            raise ContractError("Selected artifact descriptor refers to an incompatible or unbound dependency")
        # 只对要一起执行的输入核对物理接口。局部线性化后的状态范围可以严于原对象，
        # 数据集和材料又使用其他接口；祖先作为来源不等于它们都可直接参与当前运算。
        selected_interface.compatible(ModelInterface.parse(ancestors[value][1]["interface"]))
        return value

    def reference_model():
        # The source of the selected training data is evidence. A workspace's
        # active reference_model is not evidence about this selected result.
        candidates = set()
        for _, item in ancestors.values():
            if item["kind"] == "sampled_dataset":
                original = item.get("source", {}).get("model")
                if (original in ancestors and ancestors[original][1]["kind"] in MODEL_KINDS and
                        ModelInterface.parse(ancestors[original][1]["interface"]).fingerprint == selected_interface.fingerprint):
                    candidates.add(bound(original, MODEL_KINDS))
                continue
            if item["kind"] != "transition_dataset":
                continue
            source = item["source"]
            if source.get("kind") == "reference_simulator":
                if source.get("reference_artifact"):
                    candidate = source["reference_artifact"]
                    if candidate not in ancestors:
                        raise ContractError("Reference simulator is missing from the selected data branch")
                    if ModelInterface.parse(ancestors[candidate][1]["interface"]).fingerprint == selected_interface.fingerprint:
                        candidates.add(bound(candidate, MODEL_KINDS))
                else:
                    candidates.update(p["id"] for p in item["parents"] if ancestors[p["id"]][1]["kind"] == "reference_model"
                                      and ModelInterface.parse(ancestors[p["id"]][1]["interface"]).fingerprint == selected_interface.fingerprint)
        return next(iter(candidates)) if len(candidates) == 1 else None

    kind = metadata["kind"]
    suggestions, missing_config, optional_handoffs = {}, [], []
    scope = None
    applicability = metadata.get("source", {}).get("applicability")
    if kind in MODEL_KINDS:
        agent, operation = "controller", "synthesize_mpc"
        inputs = {"model": artifact_id}
        suggestions = {"prediction": {"method": "direct"}}
        if applicability:
            scope = "预测能力仅在转换报告记录的工况及误差门限内核验。"
        evaluation = reference_model()
        optional_handoffs.append({"agent": "controller", "operation": "tune_controller",
            "inputs": {"model": artifact_id, "evaluation_model": evaluation or ""},
            "missing_inputs": [] if evaluation else ["evaluation_model"],
            "missing_config": ["evaluation", "tuning_scenarios", "verification_scenarios", "test_scenarios"],
            "scope": "Optional external-preference tuning; not required for a design-only request"})
        optional_handoffs.append({"agent": "controller", "operation": "synthesize_mpc", "inputs": inputs,
            "missing_inputs": [], "missing_config": ["prediction.operating_point", "prediction.validity_bounds"],
            "config_suggestions": {"prediction": {"method": "linearize"}},
            "scope": "需要局部线性 MPC 时，指定实际平衡点或工作点及有效范围。保留原非线性预测器和线性化依据。"})
    elif kind == "controller":
        agent, operation = "lightweight", "collect_expert"
        descriptor = read_json(root / "controller.json")
        evaluation = descriptor.get("evaluation_model")
        from .plants import controller_callable_source
        original = controller_callable_source(artifact_id, closure)
        inputs = {"controller": artifact_id,
                  "evaluation_model": original or (bound(evaluation, MODEL_KINDS) if evaluation else None)}
        missing_config = ["至少五个独立采集工况（初始状态、参考目标、扰动、步数）"]
        scope = ("使用转换时直接绑定的原 python_callable 对象闭环采集；每条轨迹从其声明的初态重新开始。"
                 if original else "选择接口相符的评价对象。登记方程产生的仿真按该方程评价，原对象实测需另行验证。")
        applicability = descriptor.get("applicability")
    elif kind == "expert_dataset":
        agent, operation = "lightweight", "train_imitation"
        inputs = {"dataset": artifact_id}
        input_dim, output_dim = validate_network_dimensions(selected_interface, {}, residual=False)
        suggestions = {"architecture": "mlp", "input_dim": input_dim, "output_dim": output_dim}
    elif kind == "imitation_policy":
        agent, operation = "lightweight", None
        descriptor = read_json(root / "policy.json")
        controller_id = bound(descriptor["expert_controller"], {"controller"})
        controller_root = ancestors[controller_id][0]
        controller = read_json(controller_root / "controller.json")
        inputs = {"policy": artifact_id}
        optional_handoffs.append({"agent": agent, "operation": "optimize_policy",
            "inputs": {"policy": artifact_id, "model": bound(controller["prediction_model"], MODEL_KINDS)},
            "missing_inputs": [], "missing_config": ["确认优化目标、训练预算与独立评价工况"],
            "scope": "模仿学习交付可以结束，只有进一步优化的任务才启动此阶段。"})
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
                        **({"applicability": applicability} if applicability else {}),
                        **({"scope": scope} if scope else {}),
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
