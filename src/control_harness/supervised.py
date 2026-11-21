"""Data-only CPU regression with train/validation/test and shared checkpoints.

Feature/target dimensions come only from DataInterface.  No physical state
dimension, nominal dynamics or controller is inferred or required here.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
from pathlib import Path
import re
import zipfile

import numpy as np
import torch

from .artifacts import Workspace, atomic_json, read_json
from .checkpoints import load_checkpoint, restore_training, save_checkpoint, seed_training, training_contract
from .contracts import ContractError, DataInterface, canonical_hash
from .runtime import OperationContext


_ARRAYS = {"X", "y", "X_scaled", "y_scaled", "split_id"}
_MAX_NPZ_BYTES = 256 * 1024 * 1024
_MAX_LOGICAL_VALUES = 8_000_000
_MAX_PARAMETERS = 2_000_000
_MAX_BATCH_ACTIVATIONS = 5_000_000


def _integer(value, label, minimum=1, maximum=100_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractError(f"{label} must be an integer in [{minimum},{maximum}]")
    return value


def _number(value, label, *, minimum=0., positive=False, maximum=None):
    try: number = float(value) if type(value) in {int, float} else float("nan")
    except (ValueError, OverflowError): number = float("nan")
    if (not math.isfinite(number) or number < minimum
            or (positive and number == minimum) or (maximum is not None and number > maximum)):
        raise ContractError(f"{label} must be a finite {'positive' if positive else 'nonnegative'} number" + (f" <= {maximum}" if maximum is not None else ""))
    return number


def _resolved(config, interface):
    if not isinstance(config, dict): raise ContractError("Supervised training config must be a JSON object")
    allowed = {"hidden_dims", "input_dim", "output_dim", "activation", "epochs", "learning_rate", "batch_size",
        "patience", "seed", "weight_decay", "min_delta", "grad_clip", "max_test_nrmse",
        "resume_checkpoint", "resume_checkpoint_sha256", "plan_refs"}
    if set(config) - allowed: raise ContractError(f"Unknown supervised training fields: {sorted(set(config) - allowed)}")
    if "resume_checkpoint" in config and (not isinstance(config["resume_checkpoint"], str) or not config["resume_checkpoint"].strip()):
        raise ContractError("resume_checkpoint must be an explicit nonempty workspace checkpoint path")
    if "resume_checkpoint_sha256" in config and (not isinstance(config["resume_checkpoint_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", config["resume_checkpoint_sha256"])):
        raise ContractError("resume_checkpoint_sha256 must be an explicit lowercase SHA256")
    for key, width in (("input_dim", interface.input_dim), ("output_dim", interface.output_dim)):
        if type(config.get(key, width)) is not int or config.get(key, width) != width:
            raise ContractError(f"{key} must equal the declared DataInterface width {width}; physical dimensions are not inferred")
    hidden = config.get("hidden_dims", [32, 32])
    if not isinstance(hidden, list) or len(hidden) > 8:
        raise ContractError("hidden_dims must be an explicit list of at most 8 layer widths")
    hidden = [_integer(width, "hidden layer width", maximum=4096) for width in hidden]
    activation = config.get("activation", "tanh")
    if not isinstance(activation, str) or activation not in {"tanh", "relu", "silu"}: raise ContractError("activation must be tanh, relu or silu")
    widths = [interface.input_dim, *hidden, interface.output_dim]
    parameters = sum((left + 1) * right for left, right in zip(widths, widths[1:]))
    if parameters > _MAX_PARAMETERS: raise ContractError("Supervised network exceeds the 2,000,000-parameter CPU budget; use smaller explicit layers")
    network = {"schema_version": 1, "architecture": "mlp", "input_dim": interface.input_dim,
        "output_dim": interface.output_dim, "hidden_dims": hidden, "activation": activation, "dtype": "float64"}
    result = {"input_dim": interface.input_dim, "output_dim": interface.output_dim, "hidden_dims": hidden,
        "activation": activation, "epochs": _integer(config.get("epochs", 200), "epochs"),
        "epochs_semantics": "target_total", "learning_rate": _number(config.get("learning_rate", .003), "learning_rate", positive=True, maximum=1),
        "batch_size": _integer(config.get("batch_size", 128), "batch_size", maximum=1_000_000),
        "patience": _integer(config.get("patience", 30), "patience"),
        "seed": _integer(config.get("seed", 42), "seed", minimum=0, maximum=2**32 - 1),
        "weight_decay": _number(config.get("weight_decay", 1e-6), "weight_decay"),
        "min_delta": _number(config.get("min_delta", 0), "min_delta"),
        "grad_clip": _number(config.get("grad_clip", 5), "grad_clip", positive=True),
        "device": "cpu", "dtype": "float64", "loss": "scaled_target_mse", "optimizer": "adam"}
    threshold = config.get("max_test_nrmse")
    if threshold is not None:
        if isinstance(threshold, list):
            if len(threshold) != interface.output_dim: raise ContractError("max_test_nrmse list must have one limit per ordered output")
            threshold = [_number(value, "max_test_nrmse") for value in threshold]
        else: threshold = _number(threshold, "max_test_nrmse")
    result["max_test_nrmse"] = threshold
    return result, network, parameters


def _load_bounded_npz(path: Path, interface: DataInterface, expected_sha256=None):
    """Inspect ZIP/.npy headers before NumPy can allocate a declared shape."""
    if path.stat().st_size > _MAX_NPZ_BYTES: raise ContractError("Supervised NPZ exceeds its 256 MiB byte budget")
    try:
        with path.open("rb") as stream:
            payload = stream.read(_MAX_NPZ_BYTES + 1)
        if len(payload) > _MAX_NPZ_BYTES: raise ContractError("Supervised NPZ exceeds its 256 MiB byte budget")
        if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ContractError("Supervised NPZ bytes changed after artifact version verification")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (len(entries) > 32 or len(set(names)) != len(names) or
                    any(not name.endswith(".npy") or "/" in name or "\\" in name for name in names) or
                    sum(entry.file_size for entry in entries) > _MAX_NPZ_BYTES):
                raise ContractError("Supervised NPZ has an unsupported inventory or expanded-byte budget")
            if not {name + ".npy" for name in _ARRAYS}.issubset(names): raise ContractError("Supervised NPZ requires X,y,X_scaled,y_scaled,split_id")
            shapes, logical = {}, 0
            for key in sorted(_ARRAYS):
                entry = archive.getinfo(key + ".npy")
                with archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0): shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0): shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    else: raise ContractError("Supervised NPZ requires version-1/2 numeric NPY members")
                    header_bytes = stream.tell()
                expected_rank = 1 if key == "split_id" else 2
                if len(shape) != expected_rank or any(type(width) is not int or width < 1 for width in shape):
                    raise ContractError(f"Supervised {key} has an invalid declared rank/shape")
                if dtype.kind not in ("iu" if key == "split_id" else "fiu") or dtype.hasobject:
                    raise ContractError(f"Supervised {key} must contain real numeric values, not bool/object/complex")
                count = math.prod(shape); logical += count
                if logical > _MAX_LOGICAL_VALUES or count * dtype.itemsize != entry.file_size - header_bytes:
                    raise ContractError("Supervised NPZ declared logical shape exceeds the supported budget or member bytes")
                shapes[key] = shape
            rows = shapes["X"][0]
            if any(shape[0] != rows for shape in shapes.values()): raise ContractError("Supervised arrays have different row counts")
            for key, width in (("X", interface.input_dim), ("X_scaled", interface.input_dim), ("y", interface.output_dim), ("y_scaled", interface.output_dim)):
                if shapes[key][1] != width: raise ContractError(f"Supervised {key} width disagrees with DataInterface")
        # Deserialize the exact bytes whose headers were bounded above, rather
        # than reopening a path that could have changed after inspection.
        with np.load(io.BytesIO(payload), allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key], dtype=np.int64 if key == "split_id" else np.float64) for key in _ARRAYS}
    except ContractError: raise
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        raise ContractError(f"Invalid supervised NPZ: {error}") from error
    if any(not np.isfinite(value).all() for value in arrays.values()): raise ContractError("Supervised arrays must be finite")
    if set(np.unique(arrays["split_id"]).tolist()) != {0, 1, 2}: raise ContractError("Supervised split_id must contain nonempty train=0,validation=1,test=2 only")
    return arrays


def _scaler_vector(value, width, label):
    if (not isinstance(value, list) or len(value) != width or
            any(type(item) not in {int, float} or not math.isfinite(item) for item in value)):
        raise ContractError(f"{label} must contain one finite numeric value per declared column")
    return np.asarray(value, dtype=np.float64)


def _read_data(workspace, dataset_id):
    root, metadata = workspace.resolve(dataset_id, kind="supervised_dataset")
    interface = DataInterface.parse(metadata["interface"])
    arrays = _load_bounded_npz(root / "supervised.npz", interface, metadata["files"].get("supervised.npz"))
    masks = {name: arrays["split_id"] == index for index, name in enumerate(("train", "validation", "test"))}
    scaler = read_json(root / "scaler.json")
    if not isinstance(scaler, dict): raise ContractError("Supervised scaler must be an explicit transform object")
    method = scaler.get("method")
    if (not isinstance(method, str) or method not in {"standard", "minmax", "none"} or scaler.get("fit_split") != "train" or
            type(scaler.get("fit_samples")) is not int or scaler["fit_samples"] != int(masks["train"].sum())):
        raise ContractError("Supervised scaler must identify its exact training-only fit and sample count")
    for key, width in (("X", interface.input_dim), ("y", interface.output_dim)):
        item = scaler.get(key)
        if not isinstance(item, dict): raise ContractError("Supervised scaler requires explicit X and y transforms")
        location = _scaler_vector(item.get("location"), width, key + ".location")
        scale = _scaler_vector(item.get("scale"), width, key + ".scale")
        train = arrays[key][masks["train"]]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            expected_location = train.mean(0) if method == "standard" else (train.min(0) if method == "minmax" else np.zeros(width))
            expected_scale = train.std(0, ddof=0) if method == "standard" else (train.max(0) - train.min(0) if method == "minmax" else np.ones(width))
            constant = np.flatnonzero(expected_scale == 0).tolist()
            expected_scale[expected_scale == 0] = 1.
            expected_scaled = (arrays[key] - location) / scale
        if (np.any(scale <= 0) or not np.isfinite(expected_location).all() or not np.isfinite(expected_scale).all() or
                not np.allclose(location, expected_location, rtol=1e-12, atol=1e-12) or
                not np.allclose(scale, expected_scale, rtol=1e-12, atol=1e-12) or
                item.get("constant_columns", constant) != constant):
            raise ContractError("Supervised scaler disagrees with training-only population statistics")
        if not np.isfinite(expected_scaled).all() or not np.allclose(arrays[key + "_scaled"], expected_scaled, rtol=1e-12, atol=1e-12):
            raise ContractError("Supervised scaled arrays disagree with their declared train-only transform")
    dataset = read_json(root / "dataset.json")
    if (not isinstance(dataset, dict) or type(dataset.get("input_dim")) is not int or type(dataset.get("output_dim")) is not int or
            dataset["input_dim"] != interface.input_dim or dataset["output_dim"] != interface.output_dim):
        raise ContractError("Supervised dataset description dimensions disagree with DataInterface")
    interface.compatible(DataInterface.parse(dataset.get("interface")))
    declaration = interface.to_dict()
    signals = {item["name"]: item for item in declaration["signals"]}
    feature_order = [{"signal": item["signal"], "lag": item["lag"] + signals[item["signal"]]["delay_steps"] + history,
        "unit": signals[item["signal"]]["unit"], "availability_delay_steps": signals[item["signal"]]["delay_steps"]}
        for item in declaration["features"] for history in reversed(range(item["window"]))]
    target_order = [{"signal": item["signal"], "offset": item["offset"], "unit": signals[item["signal"]]["unit"]}
        for item in declaration["targets"]]
    if dataset.get("feature_order") != feature_order or dataset.get("target_order") != target_order:
        raise ContractError("Supervised dataset feature/target order disagrees with its declared signal units and timing")
    with np.errstate(over="ignore", invalid="ignore"):
        population_std = arrays["y"][masks["train"]].std(0, ddof=0)
    if not np.isfinite(population_std).all(): raise ContractError("Training output population standard deviation is non-finite")
    metric_scale = population_std.copy(); metric_scale[metric_scale == 0] = 1.
    return arrays, masks, scaler, dataset, metadata, interface, population_std, metric_scale


def _evaluate(network, features, truth, scale, location, metric_scale, output_order, batch_size):
    rows = []
    with torch.no_grad():
        for batch in features.split(batch_size): rows.append(network(batch).cpu().numpy())
    prediction = np.concatenate(rows, axis=0) * scale + location
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rmse = np.sqrt(np.mean(np.square(prediction - truth), axis=0))
        nrmse = rmse / metric_scale
    if not np.isfinite(prediction).all() or not np.isfinite(rmse).all() or not np.isfinite(nrmse).all():
        raise FloatingPointError("Supervised final predictions/metrics are non-finite")
    return {"samples": len(truth), "rmse": rmse.tolist(), "nrmse": nrmse.tolist(),
        "per_output": [{**order, "rmse": float(rmse[index]), "nrmse": float(nrmse[index])} for index, order in enumerate(output_order)]}, prediction


def train_supervised(workspace: Workspace, dataset_id: str, config: dict,
                     context: OperationContext | None = None, *, activate: bool = True) -> str:
    """Train only on train, select only on validation, evaluate test once at end."""
    # Internal composition only; request JSON has no activation override.
    # A candidate NN must not invalidate an accepted physical wrapper before
    # that wrapper's own prediction checks finish.
    if type(activate) is not bool:
        raise ContractError("activate must be a trusted explicit boolean")
    from .supervised_export import build_supervised_network, export_supervised

    context = context or OperationContext()
    arrays, masks, scaler, dataset, data_meta, interface, population_std, metric_scale = _read_data(workspace, dataset_id)
    resolved, network_config, parameter_count = _resolved(config, interface)
    # Parameter count does not bound hidden activations: a narrow input/output
    # may still feed a wide hidden layer with a very large batch.  Bound every
    # actual train/validation/final-evaluation chunk before network/tensor setup.
    layer_width_sum = sum([interface.input_dim, *network_config["hidden_dims"], interface.output_dim])
    for part, mask in masks.items():
        batch_rows = min(resolved["batch_size"], int(mask.sum()))
        if batch_rows * layer_width_sum > _MAX_BATCH_ACTIVATIONS:
            raise ContractError(f"Supervised {part} batch exceeds the 5,000,000 logical activation-scalar budget; reduce batch_size")
    seed_training(resolved["seed"])
    generator = torch.Generator(device="cpu").manual_seed(resolved["seed"])
    network = build_supervised_network(network_config)
    if sum(parameter.numel() for parameter in network.parameters()) != parameter_count:
        raise ContractError("Supervised network builder differs from the resolved MLP parameter contract")
    optimizer = torch.optim.Adam(network.parameters(), lr=resolved["learning_rate"], weight_decay=resolved["weight_decay"])
    X = torch.from_numpy(arrays["X_scaled"]).double()
    y = torch.from_numpy(arrays["y_scaled"]).double()
    train_idx = torch.from_numpy(np.flatnonzero(masks["train"]))
    validation_idx = torch.from_numpy(np.flatnonzero(masks["validation"]))
    contract = training_contract(workspace, "supervised_training", [dataset_id], resolved, interface.fingerprint)
    signature = canonical_hash(contract)
    start, wait, updates, log, best_state, best_loss = 0, 0, 0, [], None, float("inf")
    if config.get("resume_checkpoint"):
        checkpoint = load_checkpoint(workspace, config["resume_checkpoint"], contract, resolved["epochs"], expected_sha256=config.get("resume_checkpoint_sha256"))
        restore_training(checkpoint, network, optimizer, generator)
        start, wait, updates, log = checkpoint["epoch"], checkpoint["wait"], checkpoint["updates"], checkpoint["log"]
        best_state, best_loss = checkpoint["best_state"], checkpoint["best_metric"]
    with workspace.stage("supervised_model", interface, [dataset_id], config) as stage:
        def checkpoint_epoch(epoch):
            return save_checkpoint(stage.root, contract, target_epochs=resolved["epochs"], model=network, optimizer=optimizer, epoch=epoch, updates=updates,
                best_state=best_state, best_metric=best_loss, wait=wait, log=log, generator=generator)

        atomic_json(stage.root / "network_config.json", network_config)
        atomic_json(stage.root / "resolved_config.json", resolved)
        atomic_json(stage.root / "scaler.json", scaler)
        saved = checkpoint_epoch(start)
        context.progress("supervised_initial_checkpoint", epoch=start, optimizer_updates=updates, **saved)
        for epoch in range(start, resolved["epochs"]):
            if wait >= resolved["patience"]: break
            context.check()
            network.train()
            order = train_idx[torch.randperm(len(train_idx), generator=generator)]
            training_sum = 0.
            for batch in order.split(resolved["batch_size"]):
                context.check()
                loss = (network(X[batch]) - y[batch]).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError("Supervised training loss is non-finite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), resolved["grad_clip"], error_if_nonfinite=True)
                optimizer.step(); updates += 1
                training_sum += loss.item() * len(batch)
            network.eval()
            validation_sum = 0.
            with torch.no_grad():
                for batch in validation_idx.split(resolved["batch_size"]):
                    context.check()
                    validation_sum += (network(X[batch]) - y[batch]).square().mean().item() * len(batch)
            validation_loss = validation_sum / len(validation_idx)
            if not math.isfinite(validation_loss): raise FloatingPointError("Supervised validation loss is non-finite")
            selected = validation_loss < best_loss - resolved["min_delta"]
            if selected: best_loss, best_state, wait = validation_loss, copy.deepcopy(network.state_dict()), 0
            else: wait += 1
            event = {"epoch": epoch + 1, "training_loss": training_sum / len(train_idx), "validation_loss": validation_loss,
                     "selected_by_validation": selected, "patience_wait": wait}
            log.append(event)
            saved = checkpoint_epoch(epoch + 1)
            context.progress("supervised_training", **event, optimizer_updates=updates,
                **saved)
        if best_state is None: raise ContractError("No completed supervised training epoch is available")
        context.check()
        network.load_state_dict(best_state, strict=True); network.eval()
        torch.save({"network_state": best_state, "training_signature": signature}, stage.root / "model.pt")
        target_order = dataset.get("target_order")
        if not isinstance(target_order, list) or len(target_order) != interface.output_dim:
            raise ContractError("Dataset target_order must contain every declared output")
        reports, predictions = {}, {}
        for part in ("train", "validation", "test"):
            context.check()
            reports[part], predictions[part] = _evaluate(network, X[masks[part]], arrays["y"][masks[part]],
                np.asarray(scaler["y"]["scale"]), np.asarray(scaler["y"]["location"]), metric_scale, target_order, resolved["batch_size"])
        np.savez_compressed(stage.root / "test_predictions.npz", prediction=predictions["test"], truth=arrays["y"][masks["test"]],
                            row_index=np.flatnonzero(masks["test"]))
        limits = resolved["max_test_nrmse"]
        passed = limits is None or bool(np.all(np.asarray(reports["test"]["nrmse"]) <= np.asarray(limits)))
        acceptance = "not_requested" if limits is None else ("passed" if passed else "failed")
        export = export_supervised(stage.root / "bundle", network, network_config, interface.to_dict(), scaler, resolved,
                                  feature_order=dataset.get("feature_order"), target_order=target_order)
        best_epoch = next(event["epoch"] for event in reversed(log) if event["selected_by_validation"])
        provenance = {"dataset_id": dataset_id, "dataset_metadata_hash": canonical_hash(data_meta), "data_source": data_meta.get("source", {}),
            "feature_order": dataset.get("feature_order"), "target_order": target_order, "physical_model_required": False}
        report = {"algorithm": "supervised_training", "network_config": network_config, "resolved_config": resolved,
            "parameter_count": parameter_count, "trained_epochs": len(log), "target_total_epochs": resolved["epochs"],
            "best_epoch": best_epoch, "best_validation_loss": best_loss, "optimizer_updates": updates,
            "resumed_from_epoch": start, "training_signature": signature,
            "resource_boundary": {"max_npz_bytes": _MAX_NPZ_BYTES, "max_required_array_values": _MAX_LOGICAL_VALUES,
                "max_parameters": _MAX_PARAMETERS, "max_batch_activation_scalars": _MAX_BATCH_ACTIVATIONS,
                "batch_activation_estimate": "min(batch_size,split_rows)*sum(input,hidden,output widths); each split",
                "scope": "one training job; not a limit on the overall engineering task"},
            "stopping_reason": "patience" if wait >= resolved["patience"] else "target_total_epochs",
            "selection": "validation_scaled_target_mse_with_declared_min_delta", "test_used_for_training_or_selection": False,
            "test_evaluations": 1, "normalization": {"source": "training_partition_only", "ddof": 0,
                "y_population_std": population_std.tolist(), "nrmse_scale": metric_scale.tolist(),
                "zero_std_fallback": 1., "zero_std_outputs": np.flatnonzero(population_std == 0).tolist()},
            **reports, "acceptance": acceptance, "max_test_nrmse": limits, "data_provenance": provenance,
            "export": {"directory": "bundle", "manifest": "bundle/bundle_manifest.json",
                       "details": {key: value for key, value in export.items() if key not in {"bundle_root", "manifest_path"}}},
            "confirmation_sources": context.case_confirmation_sources, "log": log}
        atomic_json(stage.root / "report.json", report)
        atomic_json(stage.root / "training_summary.json", {key: report[key] for key in
            ("algorithm", "parameter_count", "trained_epochs", "best_epoch", "best_validation_loss", "optimizer_updates",
             "resumed_from_epoch", "stopping_reason", "test_used_for_training_or_selection", "test_evaluations", "acceptance")})
        context.check()
        return stage.publish({"status": "verified" if passed else "rejected", "checks": ["data_interface", "train_only_scaler",
            "finite_training", "validation_only_selection", "final_test_once", "independent_inference_exports"],
            "acceptance": acceptance, "test": reports["test"]}, preprocessing=scaler,
            source={"algorithm": "supervised_training", **provenance, "confirmation_sources": context.case_confirmation_sources}, activate=passed and activate)
