"""Declared single-plant registration and confirmed deterministic sampling.

Source registration is static. Execution stays in a numerical worker or one
native-MCP MATLAB batch; the same sampling validator checks both routes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

import numpy as np

from .artifacts import Workspace, atomic_json, confined, file_hash, read_json
from .contracts import ContractError, ModelInterface
from .data import trajectory_split, validate_transitions, training_coverage
from .runtime import OperationContext

MATLAB_BACKENDS = {"matlab_function", "simulink"}
LEGACY_MODELS = {"nominal_model", "reference_model", "identified_model", "corrected_model"}


def validate_plant_spec(spec):
    if not isinstance(spec, dict):
        raise ContractError("A declared adapter specification is required")
    if spec.get("backend") in MATLAB_BACKENDS:
        from .matlab_adapter import validate_matlab_spec
        return validate_matlab_spec(spec)
    from .model_adapters import validate_adapter_spec
    return validate_adapter_spec(spec)


def register_plant(workspace: Workspace, material_id: str, config: dict, context: OperationContext) -> str:
    root, metadata = workspace.resolve(material_id, kind="material")
    material = read_json(root / "material.json")
    spec = validate_plant_spec(config.get("adapter"))
    relative = spec["source_file"]
    if relative not in material["file_hashes"]:
        raise ContractError("Adapter source_file must name a file in the bound material snapshot")
    path = confined(root, relative)
    expected_suffix = {"python_callable": {".py"}, "python_factory": {".py"},
                       "matlab_function": {".m"}, "simulink": {".slx", ".mdl"}}[spec["backend"]]
    if path.suffix.lower() not in expected_suffix or file_hash(path) != material["file_hashes"][relative]:
        raise ContractError("Adapter backend or source hash disagrees with the frozen material")
    support = {"scope": "declared_adapter_only", "source_executed": False}
    if spec["backend"] == "simulink":
        from .matlab_adapter import validate_simulink_source
        support = validate_simulink_source(path, spec)
    interface = ModelInterface.parse(spec["interface"])
    context.check()
    with workspace.stage("plant_model", interface, [material_id], config, compatible_parents=False) as stage:
        atomic_json(stage.root / "plant.json", {"schema_version": 1, "adapter_spec": spec,
            "material": material_id, "material_content_hash": material["content_hash"]})
        atomic_json(stage.root / "report.json", {"kind": "plant_registration", "backend": spec["backend"],
            "source_executed": False, "interface": interface.to_dict(),
            "source_support": support,
            "validation_scope": "declared_interface_and_immutable_source_only",
            "physical_fidelity": "not_independently_certified", "state_scope": "all_dynamic_state_must_be_explicit",
            "limitations": ["Supported explicit adapters require declared signal order, units and bounds.",
                "State-only outputs; direct feedthrough, hidden memory and arbitrary projects require a supported thin adapter."]})
        return stage.publish({"status": "verified", "checks": ["static_adapter_declaration", "source_hash", "interface"],
            "execution": "not_run", "physical_fidelity": "not_independently_certified"},
            source={"kind": "declared_plant_source", "backend": spec["backend"], "material": material_id,
                "content_hash": material["content_hash"], "source_executed": False}, activate=False)


def _selection(workspace, model_id):
    root, metadata = workspace.resolve(model_id)
    interface = ModelInterface.parse(metadata["interface"])
    if metadata["kind"] == "plant_model":
        definition = read_json(root / "plant.json")
        spec = validate_plant_spec(definition["adapter_spec"])
        interface.compatible(ModelInterface.parse(spec["interface"]))
        material_id = definition["material"]
        if material_id not in [item["id"] for item in metadata["parents"]]:
            raise ContractError("Plant material is not a bound parent")
        material_root, material_meta = workspace.resolve(material_id, kind="material")
        material = read_json(material_root / "material.json")
        if material["content_hash"] != definition["material_content_hash"]:
            raise ContractError("Plant material content identity changed")
        return interface, spec, material_root, metadata, material_meta
    if metadata["kind"] not in LEGACY_MODELS:
        raise ContractError("Sampling requires one registered plant or executable Torch dynamics artifact")
    return interface, None, None, metadata, None


def recommend_sampling(workspace, model_id, config):
    from .sampling import recommend_sampling as recommend
    interface, spec, _, metadata, _ = _selection(workspace, model_id)
    candidate = recommend(interface, config.get("overrides", {}))
    return {"status": "proposal", "source_executed": False, "model": model_id,
        "backend": spec["backend"] if spec else "registered_torch", "sampling": candidate,
        "interface": interface.to_dict(), "confirmation_required": ["signals", "sampling"],
        "scope": "Deterministic bounded candidate; engineering excitation adequacy and model fidelity are not proved."}


class _TorchAdapter:
    def __init__(self, workspace, model_id):
        from .modeling import load_model
        self.model = load_model(workspace, model_id)
        self.model.eval()
        self.interface = self.model.interface
        self._state = None

    @property
    def state(self):
        if self._state is None:
            raise ContractError("Reset the plant before accessing its state")
        return self._state.copy()

    def reset(self, x0):
        self._state = np.asarray(x0, dtype=np.float64).copy()
        return self.state

    def step(self, u, d):
        import torch
        from .modeling import tensor
        with torch.no_grad():
            self._state = self.model(tensor(self.state), tensor(u), tensor(d)).detach().cpu().numpy().copy()
        return self.state

    def output(self):
        import torch
        from .modeling import tensor
        with torch.no_grad():
            return self.model.output(tensor(self.state)).detach().cpu().numpy().copy()

    def describe(self):
        from .modeling import describe_execution
        return {"backend": "registered_torch", "execution": describe_execution(self.model), "reset": "explicit_state_copy"}

    def close(self):
        pass


def _version(workspace, metadata):
    return {"id": metadata["id"], "metadata_hash": workspace.manifest["artifacts"][metadata["id"]]["metadata_hash"]}


def _matlab_phase(workspace, request, context, spec, source_root, plan, metadata, material_meta):
    from .matlab_adapter import prepare_matlab_manifest, load_matlab_batch_adapter
    phase = os.environ.get("CONTROL_MATLAB_PHASE")
    if not context._case_receipt_body or phase not in {"prepare", "finalize"}:
        raise ContractError("MATLAB sampling requires the native Host MCP batch job; direct Python does not start MATLAB")
    raw = context._case_raw_request
    digest, nonce = os.environ.get("CONTROL_MATLAB_REQUEST_SHA256", ""), os.environ.get("CONTROL_MATLAB_NONCE", "")
    if not raw or hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest or not re.fullmatch(r"[a-f0-9-]{32,64}", nonce):
        raise ContractError("MATLAB batch phase is not bound to this exact native request")
    exchange = Path(os.environ.get("CONTROL_MATLAB_EXCHANGE_DIR", "")).resolve()
    exchange_parent = (workspace.path / ".case-plans" / "matlab-exchanges").resolve()
    if exchange.parent != exchange_parent or not re.fullmatch(r"[a-f0-9-]{32,36}", exchange.name):
        raise ContractError("MATLAB exchange must remain in this workspace's private batch directory")
    versions = {"request_sha256": digest, "nonce": nonce,
        "model_version": _version(workspace, metadata), "material_version": _version(workspace, material_meta)}
    if phase == "prepare":
        prepared = prepare_matlab_manifest(spec, source_root, plan, exchange_dir=exchange, **versions)
        # The copy must still correspond to the versions approved by the user,
        # not merely be internally consistent bytes from a racing edit.
        workspace.resolve(metadata["id"])
        workspace.resolve(material_meta["id"], kind="material")
        return prepared
    result_path = Path(os.environ.get("CONTROL_MATLAB_RESULT_PATH", "")).resolve()
    if result_path.parent != exchange:
        raise ContractError("MATLAB result must belong to this batch exchange")
    return load_matlab_batch_adapter(spec, plan, result_path=result_path,
        result_sha256=os.environ.get("CONTROL_MATLAB_RESULT_SHA256", ""), **versions)


def sample_model(workspace: Workspace, model_id: str, config: dict, context: OperationContext,
                 request: dict | None = None):
    from .sampling import build_sampling_plan, prefix_plan, collect, SamplingStopped, SamplingCancelled
    interface, spec, source_root, metadata, material_meta = _selection(workspace, model_id)
    # Every condition is expanded and checked before source is imported or an
    # external simulator is called. Trials use a prefix of this same plan.
    full_plan = build_sampling_plan(interface, config["sampling"])
    run_kind = config.get("sampling_run", "trial" if "trial" in config else "full")
    if run_kind not in {"trial", "full"}:
        raise ContractError("sampling_run must be trial or full")
    trial = config.get("trial") if run_kind == "trial" else None
    if run_kind == "trial" and trial is None:
        raise ContractError("A trial requires an approved explicit trial prefix")
    plan = prefix_plan(full_plan, trial) if trial is not None else full_plan
    if trial is None and len(plan["trajectories"]) < 5:
        raise ContractError("Full sampled datasets require at least five independent trajectories; use an explicit trial for a short check")
    if spec and spec["backend"] in MATLAB_BACKENDS:
        adapter = _matlab_phase(workspace, request, context, spec, source_root, plan, metadata, material_meta)
        if isinstance(adapter, dict):
            return adapter
    elif spec:
        from .model_adapters import load_python_adapter
        adapter = load_python_adapter(source_root, spec, expected_file_hashes=material_meta["files"])
    else:
        adapter = _TorchAdapter(workspace, model_id)
    kind = "sampling_trial" if trial is not None else "sampled_dataset"
    try:
        with workspace.stage(kind, interface, [model_id], config) as stage:
            atomic_json(stage.root / "sampling-plan.json", plan)
            atomic_json(stage.root / "adapter.json", adapter.describe())
            try:
                arrays, report = collect(adapter, plan, context)
            except (SamplingStopped, SamplingCancelled) as error:
                # Capture dependencies first loaded while stepping, while
                # retaining the original failure if diagnostics are unavailable.
                try:
                    actual_adapter = adapter.describe()
                    atomic_json(stage.root / "adapter.json", actual_adapter)
                    error.report["adapter"] = actual_adapter
                except Exception as diagnostic_error:
                    error.report["adapter_evidence_error"] = {"type": type(diagnostic_error).__name__,
                        "message": str(diagnostic_error)}
                np.savez_compressed(stage.root / "partial-transitions.npz", **error.arrays)
                atomic_json(stage.root / "report.json", {**error.report, "kind": kind,
                    "model": model_id, "plan_hash": plan["plan_hash"], "data_source": "plant_simulator",
                    "confirmation_sources": context.case_confirmation_sources})
                raise
            actual_adapter = adapter.describe()
            atomic_json(stage.root / "adapter.json", actual_adapter)
            report["adapter"] = actual_adapter
            np.savez_compressed(stage.root / "transitions.npz", **arrays)
            if trial is None:
                validate_transitions(interface, arrays)
                split = trajectory_split(arrays["trajectory_id"], plan["seed"])
                atomic_json(stage.root / "split.json", split)
                coverage = training_coverage(interface, arrays, np.isin(arrays["trajectory_id"], split["train"]))
            else:
                split, coverage = None, {"scope": "trial_only_no_training_split"}
            report.update({"kind": kind, "model": model_id, "data_source": "plant_simulator",
                "plan_hash": plan["plan_hash"], "full_plan_hash": full_plan["plan_hash"],
                "trial": trial, "training": "not_run", "coverage": coverage,
                "confirmation_sources": context.case_confirmation_sources})
            atomic_json(stage.root / "report.json", report)
            return stage.publish({"status": "verified", "checks": ["confirmed_plan", "dimensions", "finite",
                "bounds", "reset", "trajectory_split" if split else "trial_prefix"], "samples": len(arrays["x"])},
                source={"kind": "plant_simulator", "model": model_id, "measured": False,
                    "backend": spec["backend"] if spec else "registered_torch", "sampling_plan_hash": plan["plan_hash"],
                    "confirmation_sources": context.case_confirmation_sources}, activate=trial is None)
    finally:
        adapter.close()
