"""Verified artifact inspection and explicit cross-role input preparation."""
from __future__ import annotations

from pathlib import Path
import math
import zipfile
import numpy as np

from .artifacts import Workspace, canonical_hash, confined, read_json
from .contracts import ContractError, ModelInterface, validate_network_dimensions

MODEL_KINDS = {"nominal_model", "reference_model", "corrected_model", "identified_model"}


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
    selected_interface = ModelInterface.parse(metadata["interface"])
    ancestors = {}

    def visit(item_id):
        if item_id in ancestors:
            return
        item_root, item = workspace.resolve(item_id)
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
    suggestions, missing_config = {}, []
    if kind in {"nominal_model", "corrected_model", "identified_model"}:
        agent, operation = "controller", "design_controller"
        inputs = {"model": artifact_id, "evaluation_model": reference_model()}
        missing_config = ["tuning_scenarios", "verification_scenarios", "test_scenarios"]
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
            "source_versions": [{"id": item_id, "metadata_hash": canonical_hash(item)} for item_id, (_, item) in ancestors.items()]}


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
