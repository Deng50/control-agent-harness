"""Verify the exact completed dialogue run without choosing files by mtime."""
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from control_harness.artifacts import Workspace, atomic_json, file_hash, read_json
from control_harness.modeling import load_model
from control_harness.policy import load_policy

source = read_json(PROJECT / "docs/harness-business-dialogue-evidence.json")
workspace = Workspace(source["workspace"])
artifacts, interfaces = {}, set()
for stage, artifact_id in source["artifacts"].items():
    root, metadata = workspace.resolve(artifact_id)
    interfaces.add(metadata["interface_hash"])
    artifacts[stage] = {"id": artifact_id, "kind": metadata["kind"],
        "status": workspace.manifest["artifacts"][artifact_id]["status"],
        "file_count": len(metadata["files"]), "parents": [item["id"] for item in metadata["parents"]]}
    assert artifacts[stage]["status"] == "verified"
assert len(artifacts) == 8 and len(interfaces) == 1
model = load_model(workspace, source["artifacts"]["corrected"])
prediction = model.step([.2, .1, 0], [.1, -.2], [.05])
assert np.isfinite(prediction).all()
policy = load_policy(workspace, source["artifacts"]["optimized"])
action, _ = policy.act([.2, .1, 0], [.1, -.1], [.05])
assert action.shape == (2,) and np.isfinite(action).all()
root, _ = workspace.resolve(source["artifacts"]["controller"])
controller = read_json(root / "controller.json")
assert controller["prediction_execution"]["nominal"]["frozen_python_source"]
assert controller["evaluation_execution"]["source_representation"] == "python_torch_transition"
reports = source["reports"]
assert reports["corrected"]["test_used_for_training_or_selection"] is False
assert reports["controller"]["test_used_for_tuning"] is False
assert reports["optimized"]["test_used_for_updates_or_checkpoint_selection"] is False
comparison = {}
for strategy in ("expert", "imitation", "optimized"):
    rows = [row[strategy]["metrics"] for row in reports["optimized"]["comparison"]]
    comparison[strategy] = {"conditions": len(rows),
        "aggregation": "arithmetic_mean_of_saved_per_condition_metrics",
        **{key: float(np.mean([row[key] for row in rows])) for key in
           ("rmse", "normalized_rmse", "input_total_variation", "mean_control_seconds")},
        **{key: max(row[key] for row in rows) for key in
           ("max_state_violation", "max_input_violation", "max_output_violation")}}
result = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
    "workspace": str(workspace.path), "session_id": source["sessionId"], "job_id": source["jobId"],
    "workflow_run": source["workflowRun"], "workflow_source_sha256": file_hash(Path(source["workflowPath"])),
    "source": "explicit_standalone_python_mathematical_benchmark_not_measurements",
    "artifacts": artifacts, "interface_hash": next(iter(interfaces)),
    "fresh_model_prediction": prediction.tolist(), "fresh_policy_action": action.tolist(),
    "execution": {key: controller[key] for key in
        ("execution_surface", "prediction_execution", "evaluation_execution")},
    "residual_test": reports["corrected"]["test"],
    "residual_relative_test_mse_reduction": 1 - reports["corrected"]["test"]["corrected_mse"] / reports["corrected"]["test"]["nominal_mse"],
    "tuned_candidate_accepted": reports["controller"]["tuned_candidate_accepted"],
    "constraint_checks": [row["constraint_checks"] for row in reports["controller"]["test"]["runs"]],
    "policy_comparison": comparison,
    "scope": "Fresh loading verifies hashes, dependencies and executable weights. Times describe this small local run, not a controlled deployment benchmark."}
atomic_json(PROJECT / "docs/harness-python-workflow-validation.json", result)
print({"verified_artifacts": len(artifacts), "compatible_interfaces": len(interfaces),
       "model_and_policy_loaded": True, "residual_test_mse_reduction": result["residual_relative_test_mse_reduction"]})
