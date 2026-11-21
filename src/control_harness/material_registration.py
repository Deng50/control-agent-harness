"""Immutable material and data-only artifacts over the existing Workspace."""
from __future__ import annotations

import numpy as np

from .artifacts import atomic_json, read_json
from .contracts import ContractError, DataInterface, MaterialInterface
from .datasets import prepare_dataset
from .materials import inspect_material, register_material as snapshot_material


def material_options(config: dict) -> dict:
    return {key: config[key] for key in ("original_name", "dependency_root", "dependencies") if key in config}


def register_material(workspace, local_path, config: dict, context) -> str:
    description = inspect_material(local_path, context=context, **material_options(config))
    source_kind = config.get("source_kind", "path")
    if source_kind not in {"path", "upload"}:
        raise ContractError("Material source_kind must be path or upload")
    interface = MaterialInterface(description["format"], config.get("name", "material"))
    with workspace.stage("material", interface, [], config) as stage:
        material = snapshot_material(local_path, stage.root, source_kind=source_kind,
            source_metadata=config.get("source_metadata"), expected_manifest=config.get("expected_manifest", description),
            context=context, **material_options(config))
        atomic_json(stage.root / "material.json", material)
        return stage.publish({"status": "verified", "checks": ["source_snapshot", "all_file_hashes", "dependencies_confined"],
            "execution": "not_run", "semantics": "format_only"},
            source={"kind": "material", "source_kind": source_kind, "content_hash": material["content_hash"],
                    "input_path": str(local_path), "source_executed": False}, activate=False)


def prepare_supervised_dataset(workspace, material_id: str, config: dict, context) -> str:
    root, metadata = workspace.resolve(material_id, kind="material")
    material = read_json(root / "material.json")
    if material["format"] not in {"csv", "xlsx"}:
        raise ContractError("Supervised data preparation requires CSV or XLSX material")
    interface = DataInterface.parse(config["interface"])
    source = config.get("source", {})
    if source.get("kind") not in {"measured", "synthetic", "external_expert"} or not isinstance(source.get("description"), str) or not source["description"].strip():
        raise ContractError("Dataset provenance requires measured/synthetic/external_expert and an explicit description")
    context.check()
    prepared = prepare_dataset(root / material["primary_file"], interface, config["split"],
        scaler=config.get("scaler", "standard"), **{key: config[key] for key in
            ("worksheet", "header_row", "delimiter", "encoding") if key in config})
    with workspace.stage("supervised_dataset", interface, [material_id], config, compatible_parents=False) as stage:
        np.savez_compressed(stage.root / "supervised.npz", **prepared.arrays)
        for name, value in (("dataset", prepared.metadata), ("split", prepared.split),
                            ("scaler", prepared.scaler), ("statistics", prepared.statistics)):
            atomic_json(stage.root / f"{name}.json", value)
        confirmations = getattr(context, "case_confirmation_sources", [])
        atomic_json(stage.root / "report.json", {"kind": "supervised_data_preparation", "training": "not_run",
            "physical_model": "not_provided", "dimensions": {"input_dim": interface.input_dim, "output_dim": interface.output_dim},
            "split": prepared.split, "statistics": prepared.statistics, "confirmation_sources": confirmations})
        from .case_requirements import _config
        from .preparation_proof import sign_preparation
        parent = {"id": material_id, "metadata_hash": stage.parents[0]["metadata_hash"],
                  "content_hash": material["content_hash"]}
        provenance = {**source, "material_artifact": material_id, "content_hash": material["content_hash"],
                      "provenance_is_user_declared": True, "confirmation_sources": confirmations}
        # 用户可填写来源说明，派生数据凭证只能由已验证原生收据的本次作业签发。
        provenance.pop("preparation_proof", None)
        proof = sign_preparation(stage, context, parent, _config(config))
        if proof is not None:
            provenance["preparation_proof"] = proof
        return stage.publish({"status": "verified", "checks": ["explicit_signal_semantics", "causal_windows",
            "split_before_windows", "train_only_scaler", "finite_arrays"], "training": "not_run"},
            preprocessing=prepared.scaler, source=provenance)
