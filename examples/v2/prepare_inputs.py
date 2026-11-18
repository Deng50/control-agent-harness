"""Freeze portable synthetic examples from approved bytes; never train or call a service.

The two CSVs produced beside this file are distributable example inputs.
The preparation source roots are used once for provenance, never at runtime.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source-run", type=Path, required=True)
parser.add_argument("--evidence", type=Path, required=True)
args = parser.parse_args()
run = args.source_run.resolve()
out = Path(__file__).resolve().parent
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
ref = lambda path: {"path": str(path.resolve()), "sha256": sha(path)}


def put(name, value):
    path = out / name
    raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists():
        assert path.read_bytes() == raw, "Refuse changed portable input: " + name
    else:
        path.write_bytes(raw)
    return {"path": name, "sha256": sha(path), "bytes": len(raw)}


f03 = run / "fixtures/F03_synthetic_3in2out.csv"
assert sha(f03) == "11a7547ecd08d46b7c04f9b02f707753479f18a4d062c2a2499206fa20343715"
s05 = read(run / "S05/native-supervised.json")
s05report = Path(s05["stages"]["csv:training"]["result"]["artifact_path"]) / "report.json"
assert sha(s05report) == "fd4876a4789b2b55e7cd40a37b86d2545b856fe218811aa38599ac3838ba4ed9"
training = s05["preRegisteredExpertConfigs"]["csv"]
assert training["hidden_dims"] == [24, 16] and training["learning_rate"] == .005
headers = next(csv.reader(f03.read_text().splitlines()))
column = lambda index: {"index": index, "header": headers[index], "worksheet": None}
interface = {"schema_version": 1, "interface_kind": "tabular_supervised", "name": "Frozen synthetic F03 static 3in2out; no plant", "task": "static_regression", "sample_time": None, "time_column": None, "trip_column": column(0),
    "signals": [{"name": name, "role": "auxiliary" if index < 3 else "output", "unit": "1", "column": column(index + 4), "delay_steps": 0, "observable": index < 3} for index, name in enumerate(headers[4:])],
    "features": [{"signal": name, "window": 1, "lag": 0} for name in headers[4:7]], "targets": [{"signal": name, "offset": 0} for name in headers[7:]], "input_dim": 3, "output_dim": 2}
nn = {"data_file": "f03-synthetic.csv", "data_config": {"interface": interface, "split": {"mode": "trip", "train": .625, "validation": .1875, "test": .1875}, "scaler": "standard", "source": {"kind": "synthetic", "description": "Frozen mathematical F03 table; not vehicle measurements or a physical Plant"}}, "training_config": training,
    "expected": {"input_dim": 3, "output_dim": 2, "rows": 1024, "split_rows": {"train": 640, "validation": 192, "test": 192}, "test_nrmse_max": .15, "physical_model_required": False}}

record = run / "S10/i3/csv/artifacts/external_expert_dataset-7d3e2ae1544c"
csv_source = run / "S10/i3/csv/artifacts/material-1ec836bd3459/source/actual-expert.csv"
import_cfg = read(record / "config.json")
import_cfg.pop("plan_refs", None)
import_cfg["source"] = {"kind": "recorded_external_expert", "description": "Actual recorded external fixed-QP actions from declared S09 simulator collection; not measured hardware or replacement labels", "previous_action_semantics": "actual_applied_action_before_current_decision", "original_source_kind": "declared_simulation", "measured": False}
teacher = run / "S09/native-external-expert-python/f05/artifacts/external_expert_dataset-5b579ff6a853"
with np.load(record / "expert.npz", allow_pickle=False) as a, np.load(teacher / "expert.npz", allow_pickle=False) as b:
    assert a["X"].shape == (1280, 9) and a["y"].shape == (1280, 2)
    assert np.array_equal(a["X"], b["X"]) and np.array_equal(a["y"], b["y"])
    assert np.array_equal(a["split_id"], b["split_id"])
    actual = np.asarray(list(csv.reader(csv_source.read_text().splitlines()))[1:], dtype=float)
    assert np.array_equal(actual[:, 2:11], a["X"]) and np.array_equal(actual[:, 11:13], a["y"])
s10report = run / "S10/n2/python-f05/artifacts/external_imitation_policy-cdc97200cf55/report.json"
assert sha(s10report) == "64d6e331ebde97f2f6221180c9fc55d042cfb26a5ddb7f1db839e382287b1da7"
report = read(s10report)
assert report["network"]["network_config"]["input_dim"] == 9 and report["network"]["network_config"]["output_dim"] == 2
assert report["network"]["parameter_count"] == 98 and report["acceptance"] == "passed"
selected = run / "S10/native-selected-training-registration.json"
policy_cfg = read(selected)["cases"][0]["training_config"]
assert policy_cfg["network"]["hidden_dims"] == [8] and policy_cfg["network"]["seed"] == 20261122
lw = {"data_file": "external-qp-records.csv", "data_config": import_cfg, "training_config": policy_cfg,
    "expected": {"input_dim": 9, "output_dim": 2, "rows": 1280, "split_rows": {"train": 800, "validation": 240, "test": 240}, "test_nrmse_max": .2, "parameter_count": 98, "physical_context": None, "pgac": "not_run", "closed_loop": "not_run"}}

f01 = read(run / "fixtures/F01_spring_2x1.json")
p = f01["physical_parameters"]
mpc = {"description": "质量弹簧阻尼系统，仅设计MPC；状态为位移和速度，力为输入，位移为输出，无额外扰动通道。未知参数保持未实例化。", "draft_config": {"template": "spring_damper", "controller": {"horizon": 5, "max_iterations": 120, "solver_tolerance": 1e-9}, "weights": {"q": 1, "r": .01, "s": .01}},
    "parameters": {"mass": p["mass_kg"], "damping": p["damping_Ns_per_m"], "stiffness": p["spring_N_per_m"], "sample_time": .2, "state_bounds": [[-1, 1], [-1.5, 1.5]], "input_bounds": [[-4, 4]], "output_bounds": [[-1, 1]]},
    "solver_check": {"x": [.1, 0], "reference": [0], "disturbance": [], "previous": [0]}, "expected": {"ga_trials": 0, "closed_loop": "not_run", "numeric_solver_status": "passed"}, "scope": "F01 mass/damping/stiffness/dt/bounds reused; this explicit spring template declares nd=0, not the full F01 disturbance fixture"}
files = [put("f03-synthetic.csv", f03.read_bytes()), put("external-qp-records.csv", csv_source.read_bytes()), put("model-nn.json", nn), put("mpc-design.json", mpc), put("imitation.json", lw)]
provenance = {"schema_version": 1, "source_kinds": {"f03-synthetic.csv": "synthetic mathematical reference, not measured", "external-qp-records.csv": "recorded external QP commands in a declared simulator; all 1280 actual rows preserved"}, "runtime_old_run_paths_required": False, "fixtures": files, "gates": {"F03_per_output_NRMSE": .15, "F05_per_action_NRMSE": .2, "same_export_atol": 1e-6, "same_export_rtol": 1e-5}, "selection": "Fixed before S14 execution using previously accepted explicit configurations, not new test-driven tuning"}
files.append(put("provenance.json", provenance))
files.append(put("mpc-draft.json", {"description": mpc["description"], "draft_config": mpc["draft_config"], "missing_parameters": ["mass", "damping", "stiffness", "sample_time", "state_bounds", "input_bounds", "output_bounds"], "model_instantiated": False}))
evidence = {"schema_version": 1, "status": "FROZEN_INPUTS_NO_TRAINING_NO_LLM", "public_root": str(out), "files": files, "lineage": {"F03": ref(f03), "S05_passed_report": ref(s05report), "S05_native_evidence": ref(run / "S05/native-supervised.json"), "expert_CSV": ref(csv_source), "expert_dataset_meta": ref(record / "artifact.json"), "S09_teacher_meta": ref(teacher / "artifact.json"), "S10_passed_report": ref(s10report), "S10_training_registration": ref(selected)}, "expert_arrays_equal_to_actual_S09_X_y_split": True, "S10_network_dims": [9, 8, 2], "S10_parameter_count": 98, "actions": {"LLM": 0, "training": 0, "RPC": 0}}
assert not args.evidence.exists()
args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": evidence["status"], "evidence": str(args.evidence), "sha256": sha(args.evidence), "files": len(files)}))
