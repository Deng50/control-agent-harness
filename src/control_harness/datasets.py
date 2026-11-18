"""Read-only tabular inspection and causal supervised-data preparation.

This module neither infers a plant state dimension nor trains a model. Column
indices are zero based; recorded source-row numbers are one-based table records
(the header is ``header_row``), not physical CSV lines with quoted newlines.
``delay_steps`` describes availability, not a physical transport delay.
"""
from __future__ import annotations

import csv
import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .contracts import ContractError


SPLITS = ("train", "validation", "test")
ROLES = {"state", "control", "output", "disturbance", "auxiliary"}
MAX_BYTES = 64 * 1024 * 1024
MAX_ROWS = 1_000_000


@dataclass
class PreparedDataset:
    arrays: dict[str, np.ndarray]
    metadata: dict
    split: dict
    statistics: dict
    scaler: dict


@dataclass
class _Table:
    worksheet: str | None
    headers: list[str | None]
    rows: list[list[Any]]
    header_row: int


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be explicit and nonempty")
    return value


def _source(path: str | Path) -> tuple[Path, dict]:
    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in {".csv", ".xlsx"}:
        raise ContractError("Select an existing CSV or XLSX file")
    if source.stat().st_size > MAX_BYTES:
        raise ContractError("Table exceeds the 64 MiB source-file budget")
    raw = source.read_bytes()
    return source, {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw), "format": source.suffix.lower()[1:]}


def _table(sheet: str | None, rows: list[list[Any]], header_row: int) -> _Table:
    if len(rows) < header_row:
        raise ContractError("header_row is outside the table")
    headers = [None if value is None else str(value) for value in rows[header_row - 1]]
    if not headers:
        raise ContractError("Table header is empty")
    data = rows[header_row:]
    while data and all(value is None or value == "" for value in data[-1]):
        data.pop()
    if not data:
        raise ContractError("Table has no data records")
    if len(data) > MAX_ROWS:
        raise ContractError("Table exceeds the one-million-row budget")
    if any(len(row) != len(headers) for row in data):
        raise ContractError("Ragged records conflict with the positional header mapping")
    return _Table(sheet, headers, data, header_row)


def _xlsx_preflight(source: Path) -> None:
    """Bound ZIP expansion and reject XML entity declarations before parsing."""
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) > 10000 or sum(item.file_size for item in entries) > 512 * 1024 * 1024:
            raise ContractError("XLSX exceeds the entry-count or 512 MiB expanded-file budget")
        names = set()
        for item in entries:
            normalized = item.filename.replace("\\", "/")
            parts = PurePosixPath(normalized).parts
            if (not parts or normalized.startswith("/") or ".." in parts or
                    any(":" in part for part in parts)):
                raise ContractError("XLSX archive member path is unsafe")
            name = "/".join(parts)
            if name in names:
                raise ContractError("XLSX archive contains duplicate member paths")
            names.add(name)
            if item.flag_bits & 1:
                raise ContractError("Encrypted XLSX members are not supported")
            if item.is_dir() or not normalized.lower().endswith((".xml", ".rels")):
                continue
            # XML in UTF-8/16/32 is checked without constructing a DOM. Carry
            # covers declarations straddling chunks, including UTF-32 NULs.
            with archive.open(item) as stream:
                carry, observed = b"", 0
                while chunk := stream.read(64 * 1024):
                    observed += len(chunk)
                    if observed > item.file_size:
                        raise ContractError("XLSX member expansion conflicts with its declared size")
                    sample = (carry + chunk).replace(b"\x00", b"").lower()
                    if b"<!doctype" in sample or b"<!entity" in sample:
                        raise ContractError("XLSX XML DTD/ENTITY declarations are forbidden")
                    carry = chunk[-128:]


def _read_tables(source: Path, *, worksheet: str | None, header_row: int,
                 delimiter: str, encoding: str) -> list[_Table]:
    _integer(header_row, "header_row", 1)
    if source.suffix.lower() == ".csv":
        if worksheet is not None:
            raise ContractError("CSV columns cannot specify a worksheet")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise ContractError("CSV delimiter must be one character")
        try:
            with source.open("r", encoding=encoding, newline="") as stream:
                rows = list(csv.reader(stream, delimiter=delimiter))
        except (UnicodeError, csv.Error, LookupError) as exc:
            raise ContractError(f"Cannot read CSV with the declared encoding/delimiter: {exc}") from exc
        return [_table(None, rows, header_row)]
    try:
        import openpyxl
    except ImportError as exc:
        raise ContractError("XLSX inspection requires the optional openpyxl dependency") from exc
    try:
        _xlsx_preflight(source)
        workbook = openpyxl.load_workbook(source, read_only=True, data_only=False, keep_links=False)
        try:
            names = [worksheet] if worksheet is not None else workbook.sheetnames
            if not names or any(name not in workbook.sheetnames for name in names):
                raise ContractError("The explicitly selected worksheet does not exist")
            tables = []
            for name in names:
                sheet = workbook[name]
                if sheet.max_row > MAX_ROWS + header_row:
                    raise ContractError("Worksheet exceeds the one-million-row budget")
                tables.append(_table(name, [list(row) for row in sheet.iter_rows(values_only=True)], header_row))
            return tables
        finally:
            workbook.close()
    except (OSError, ValueError, RuntimeError, NotImplementedError, zipfile.BadZipFile) as exc:
        raise ContractError(f"Cannot read XLSX: {exc}") from exc


def _same_source(source: Path, metadata: dict) -> None:
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ContractError("Source file changed while it was being inspected/prepared")


def inspect_table(path: str | Path, worksheet: str | None = None, *, header_row: int = 1,
                  preview_rows: int = 5, delimiter: str = ",", encoding: str = "utf-8-sig") -> dict:
    """Return a read-only inventory, preserving duplicate/blank header positions."""
    _integer(preview_rows, "preview_rows")
    if preview_rows > 100:
        raise ContractError("Inspection preview_rows must be <= 100")
    source, metadata = _source(path)
    tables = _read_tables(source, worksheet=worksheet, header_row=header_row,
                          delimiter=delimiter, encoding=encoding)
    inventory = []
    for table in tables:
        duplicates = []
        for header in dict.fromkeys(table.headers):
            positions = [i for i, value in enumerate(table.headers) if value == header]
            if len(positions) > 1:
                duplicates.append({"header": header, "indices": positions})
        inventory.append({"worksheet": table.worksheet, "header_row": header_row,
                          "row_count": len(table.rows), "column_count": len(table.headers),
                          "columns": [{"index": i, "header": value, "worksheet": table.worksheet}
                                      for i, value in enumerate(table.headers)],
                          "duplicate_headers": duplicates,
                          "preview": [[_json_cell(cell) for cell in row] for row in table.rows[:preview_rows]]})
    _same_source(source, metadata)
    return {"schema_version": 1, "read_only": True, "source": metadata, "worksheets": inventory,
            "inferred_signal_roles": False, "inferred_plant_dimensions": False}


def _json_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return str(value)


def _column(selector: Any, table: _Table, description: str) -> int:
    if not isinstance(selector, dict) or "index" not in selector or "header" not in selector:
        raise ContractError(f"{description} requires explicit column index and original header")
    index = _integer(selector["index"], f"{description}.index")
    if index >= len(table.headers):
        raise ContractError(f"{description} column index is outside the table")
    if selector.get("worksheet") != table.worksheet:
        raise ContractError(f"{description} worksheet conflicts with the selected table")
    if selector["header"] != table.headers[index]:
        raise ContractError(f"{description} header conflicts with its column position")
    return index


def _numeric(rows: list[list[Any]], column: int, description: str) -> np.ndarray:
    values = []
    for row_index, row in enumerate(rows):
        value = row[column]
        if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
            raise ContractError(f"Missing/conflicting numeric {description} at data record {row_index + 1}")
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ContractError(f"Non-numeric {description} at data record {row_index + 1}; formulas are not evaluated") from exc
        if not math.isfinite(number):
            raise ContractError(f"Nonfinite {description} at data record {row_index + 1}")
        values.append(number)
    return np.asarray(values, dtype=np.float64)


def _fractions(split: dict) -> tuple[float, float, float]:
    values = []
    for name, default in zip(SPLITS, (0.6, 0.2, 0.2)):
        value = split.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ContractError("Split fractions must be finite and positive")
        values.append(float(value))
    if not math.isclose(sum(values), 1, abs_tol=1e-10):
        raise ContractError("Split train/validation/test fractions must sum to one")
    return tuple(values)


def _counts(length: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if length < 3:
        raise ContractError("At least three independent groups/rows are needed for three splits")
    train = min(length - 2, max(1, int(length * fractions[0])))
    validation = min(length - train - 1, max(1, int(length * fractions[1])))
    return train, validation, length - train - validation


def _summary(values: np.ndarray) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        summary = {"min": values.min(axis=0), "max": values.max(axis=0),
                   "mean": values.mean(axis=0), "std": values.std(axis=0)}
    if not all(np.isfinite(value).all() for value in summary.values()):
        raise ContractError("Numeric range overflows finite summary statistics")
    return {"samples": len(values), **{key: value.tolist() for key, value in summary.items()}}


def _fit_scaler(values: np.ndarray, method: str) -> dict:
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "standard":
            location, scale = values.mean(axis=0), values.std(axis=0)
        elif method == "minmax":
            location = values.min(axis=0)
            scale = values.max(axis=0) - location
        else:
            location, scale = np.zeros(values.shape[1]), np.ones(values.shape[1])
    if not np.isfinite(location).all() or not np.isfinite(scale).all():
        raise ContractError("Numeric range overflows finite scaler statistics")
    constant = np.flatnonzero(scale == 0).tolist()
    scale[scale == 0] = 1
    return {"location": location.tolist(), "scale": scale.tolist(), "constant_columns": constant}


def prepare_dataset(path: str | Path, interface: dict, split: dict, *, scaler: str = "standard",
                    worksheet: str | None = None, header_row: int = 1,
                    delimiter: str = ",", encoding: str = "utf-8-sig") -> PreparedDataset:
    """Prepare arrays without writes, fitting transformations on train only.

    Features are ordered by declaration, then oldest to newest within each
    window. At anchor k, feature samples are k-lag-delay-(window-1)..k-lag-delay;
    targets are k+offset. Rows are partitioned *before* constructing windows.
    """
    if hasattr(interface, "to_dict"):
        interface = interface.to_dict()
    if not isinstance(interface, dict) or interface.get("schema_version") != 1 or interface.get("interface_kind", interface.get("kind")) != "tabular_supervised":
        raise ContractError("A version-1 tabular_supervised DataInterface is required")
    if any(name in interface for name in ("nx", "nu", "ny", "states")):
        raise ContractError("A data-only interface cannot declare/infer plant dimensions")
    if scaler not in {"standard", "minmax", "none"}:
        raise ContractError("scaler must be standard, minmax or none")
    task = interface.get("task")
    if task not in {"static_regression", "dynamic_prediction"}:
        raise ContractError("Explicit static_regression or dynamic_prediction task is required")
    source, source_metadata = _source(path)
    time_selector = interface.get("time_column")
    raw_signals = interface.get("signals")
    if not isinstance(raw_signals, (list, tuple)) or not raw_signals or any(not isinstance(item, dict) for item in raw_signals):
        raise ContractError("Explicit signal definitions are required")
    selectors = [item.get("column", {}) for item in raw_signals]
    if any(not isinstance(item, dict) for item in selectors):
        raise ContractError("Each signal needs an explicit column selector")
    if worksheet is None and source.suffix.lower() == ".xlsx":
        sheet_names = {item.get("worksheet") for item in selectors}
        if len(sheet_names) != 1 or None in sheet_names:
            raise ContractError("All signals must name one explicit worksheet; cross-sheet alignment is not inferred")
        worksheet = next(iter(sheet_names))
    tables = _read_tables(source, worksheet=worksheet, header_row=header_row, delimiter=delimiter, encoding=encoding)
    if len(tables) != 1:
        raise ContractError("Prepare one explicitly selected worksheet at a time")
    table = tables[0]
    signals, columns, series = {}, {}, {}
    for signal in raw_signals:
        if not isinstance(signal, dict):
            raise ContractError("Each signal must be a mapping")
        name = _nonempty(signal.get("name"), "Signal name")
        _nonempty(signal.get("unit"), f"{name}.unit")
        if name in signals or signal.get("role") not in ROLES:
            raise ContractError("Signal names must be unique and roles explicit")
        if type(signal.get("observable")) is not bool:
            raise ContractError(f"{name}.observable must be explicitly true or false")
        _integer(signal.get("delay_steps"), f"{name}.delay_steps")
        column = _column(signal.get("column"), table, name)
        if column in columns:
            raise ContractError("Two signal names alias the same source column; resolve the mapping conflict")
        signals[name], columns[column] = signal, name
        series[name] = _numeric(table.rows, column, name)
    mapping = interface.get("state_mapping")
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict) or any(name not in signals or signals[name]["role"] != "state" for name in mapping.values()):
        raise ContractError("state_mapping may only reference explicitly declared state-role signals")
    features, targets = interface.get("features"), interface.get("targets")
    if not isinstance(features, (list, tuple)) or not features or not isinstance(targets, (list, tuple)) or not targets:
        raise ContractError("At least one explicit feature and target are required")
    feature_order, target_order = [], []
    for feature in features:
        if not isinstance(feature, dict) or feature.get("signal") not in signals:
            raise ContractError("Feature references an unknown signal")
        signal = signals[feature["signal"]]
        if not signal["observable"]:
            raise ContractError("Feature signal is not observable at inference time")
        window = _integer(feature.get("window"), "Feature window", 1)
        if window > len(table.rows):
            raise ContractError("No valid windows: a feature window exceeds all available records")
        lag = _integer(feature.get("lag"), "Feature lag")
        for history in reversed(range(window)):
            feature_order.append({"signal": signal["name"], "lag": lag + signal["delay_steps"] + history,
                                  "unit": signal["unit"], "availability_delay_steps": signal["delay_steps"]})
    if len({(item["signal"], item["lag"]) for item in feature_order}) != len(feature_order):
        raise ContractError("Overlapping feature definitions duplicate the same signal/time")
    for target in targets:
        if not isinstance(target, dict) or target.get("signal") not in signals:
            raise ContractError("Target references an unknown signal")
        offset = _integer(target.get("offset"), "Target offset")
        signal = signals[target["signal"]]
        if any(item["signal"] == signal["name"] and -item["lag"] >= offset for item in feature_order):
            raise ContractError("Target/output input leakage: feature observes its target at the same or future time")
        target_order.append({"signal": signal["name"], "offset": offset, "unit": signal["unit"]})
    if len({(item["signal"], item["offset"]) for item in target_order}) != len(target_order):
        raise ContractError("Duplicate signal/offset targets")
    for name, expected in (("input_dim", len(feature_order)), ("output_dim", len(target_order))):
        if name in interface and (type(interface[name]) is not int or interface[name] != expected):
            raise ContractError(f"{name} conflicts with the explicit feature/target definitions")
    if task == "static_regression" and (any(item["lag"] for item in feature_order) or
                                        any(item["window"] != 1 for item in features) or
                                        any(item["offset"] for item in target_order)):
        raise ContractError("Static regression requires same-row features/targets without windows or delay")
    sample_time = interface.get("sample_time")
    if sample_time is not None and (isinstance(sample_time, bool) or not isinstance(sample_time, (int, float)) or
                                    not math.isfinite(sample_time) or sample_time <= 0):
        raise ContractError("sample_time must be finite and positive")
    if task == "dynamic_prediction" and (time_selector is None or sample_time is None):
        raise ContractError("Dynamic prediction requires a time column and explicit sample_time")
    times = _numeric(table.rows, _column(time_selector, table, "time_column"), "time") if time_selector is not None else None
    trip_selector = interface.get("trip_column")
    if not isinstance(split, dict) or split.get("mode") not in {"trip", "time"}:
        raise ContractError("Split mode must explicitly be trip or time; random-row splitting is forbidden")
    if split["mode"] == "trip" and trip_selector is None:
        raise ContractError("Trip split requires an explicit trip_column")
    fractions = _fractions(split)
    if trip_selector is not None:
        trip_index = _column(trip_selector, table, "trip_column")
        trips = [row[trip_index] for row in table.rows]
        if any(value is None or not str(value).strip() or isinstance(value, bool) or
               (isinstance(value, float) and not math.isfinite(value)) for value in trips):
            raise ContractError("Trip/group identifiers must be explicit and nonmissing")
        # Type prefix prevents numeric 1 and text '1' from being silently merged.
        group_ids = [f"{type(value).__name__}:{value}" for value in trips]
    else:
        group_ids = ["single_table"] * len(table.rows)
    grouped_rows: dict[str, list[int]] = {}
    for row_index, group in enumerate(group_ids):
        grouped_rows.setdefault(group, []).append(row_index)
    groups = {name: np.asarray(indices, dtype=np.int64) for name, indices in grouped_rows.items()}
    for name, indices in groups.items():
        if times is not None:
            differences = np.diff(times[indices])
            if np.any(differences <= 0):
                raise ContractError(f"Time must be strictly increasing within group {name}")
            if sample_time is not None and not np.allclose(differences, sample_time, rtol=1e-6, atol=1e-9):
                raise ContractError(f"Time gaps conflict with sample_time within group {name}; resampling is not inferred")
    partitions: dict[str, list[tuple[str, np.ndarray]]] = {name: [] for name in SPLITS}
    if split["mode"] == "trip":
        if trip_selector is None:
            raise ContractError("Trip split requires an explicit trip_column")
        counts = _counts(len(groups), fractions)
        names = list(groups)
        start = 0
        for label, count in zip(SPLITS, counts):
            partitions[label] = [(name, groups[name]) for name in names[start:start + count]]
            start += count
    else:
        for group, indices in groups.items():
            counts = _counts(len(indices), fractions)
            start = 0
            for label, count in zip(SPLITS, counts):
                partitions[label].append((group, indices[start:start + count]))
                start += count
    X, y, feature_rows, target_rows, anchors, prepared_groups, labels = [], [], [], [], [], [], []
    raw_rows, group_membership = {}, {}
    max_history = max(item["lag"] for item in feature_order)
    max_future = max(item["offset"] for item in target_order)
    for split_id, label in enumerate(SPLITS):
        raw_rows[label], group_membership[label] = [], []
        before = len(X)
        for group, indices in partitions[label]:
            raw_rows[label].extend((indices + header_row + 1).tolist())
            group_membership[label].append(group)
            for anchor in range(max_history, len(indices) - max_future):
                input_indices = [int(indices[anchor - item["lag"]]) for item in feature_order]
                output_indices = [int(indices[anchor + item["offset"]]) for item in target_order]
                X.append([series[item["signal"]][row] for item, row in zip(feature_order, input_indices)])
                y.append([series[item["signal"]][row] for item, row in zip(target_order, output_indices)])
                feature_rows.append([row + header_row + 1 for row in input_indices])
                target_rows.append([row + header_row + 1 for row in output_indices])
                anchors.append(int(indices[anchor]) + header_row + 1)
                prepared_groups.append(group)
                labels.append(split_id)
        if len(X) == before:
            raise ContractError(f"{label} has no valid windows after splitting; supply more independent data")
    arrays = {"X": np.asarray(X, dtype=np.float64), "y": np.asarray(y, dtype=np.float64),
              "split_id": np.asarray(labels, dtype=np.int8), "anchor_row": np.asarray(anchors, dtype=np.int64),
              "feature_rows": np.asarray(feature_rows, dtype=np.int64), "target_rows": np.asarray(target_rows, dtype=np.int64),
              "group_id": np.asarray(prepared_groups, dtype=str)}
    train = arrays["split_id"] == 0
    transform = {"method": scaler, "fit_split": "train", "fit_samples": int(train.sum()),
                 "X": _fit_scaler(arrays["X"][train], scaler), "y": _fit_scaler(arrays["y"][train], scaler)}
    for key in ("X", "y"):
        with np.errstate(over="ignore", invalid="ignore"):
            arrays[f"{key}_scaled"] = (arrays[key] - transform[key]["location"]) / transform[key]["scale"]
        if not np.isfinite(arrays[f"{key}_scaled"]).all():
            raise ContractError("Numeric range overflows finite transformed arrays")
    split_metadata = {"mode": split["mode"], "fractions": dict(zip(SPLITS, fractions)),
                      "ordering": "first-seen trip order" if split["mode"] == "trip" else "chronological prefix within each trip",
                      "split_before_windowing": True, "raw_rows": raw_rows, "groups": group_membership,
                      "sample_counts": {name: int((arrays["split_id"] == i).sum()) for i, name in enumerate(SPLITS)}}
    statistics = {name: {key: _summary(arrays[key][arrays["split_id"] == i]) for key in ("X", "y")}
                  for i, name in enumerate(SPLITS)}
    metadata = {"schema_version": 1, "interface_kind": "tabular_supervised", "task": task,
                "source": source_metadata | {"worksheet": table.worksheet, "header_row": header_row,
                                               "delimiter": delimiter if table.worksheet is None else None,
                                               "encoding": encoding if table.worksheet is None else None},
                "interface": interface, "input_dim": len(feature_order), "output_dim": len(target_order),
                "feature_order": feature_order, "target_order": target_order,
                "row_semantics": "one-based table record; header_row identifies the header",
                "delay_semantics": "sample at k becomes available at k+delay_steps",
                "plant_dimensions_inferred": False, "trained": False}
    _same_source(source, source_metadata)
    return PreparedDataset(arrays, metadata, split_metadata, statistics, transform)
