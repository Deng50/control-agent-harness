"""Extract factual acceptance metrics from an already executed demo workspace."""
import argparse
import json
from pathlib import Path
import numpy as np
from control_harness.artifacts import Workspace, read_json

parser = argparse.ArgumentParser()
parser.add_argument("workspace", type=Path)
parser.add_argument("--workflow-run", help="Explicit completed workflow run id; no latest-file guessing")
args = parser.parse_args()
workspace = Workspace(args.workspace)
if args.workflow_run:
    import re
    if not re.fullmatch(r"[a-f0-9]{32}", args.workflow_run):
        parser.error("Invalid workflow run id")
    record = read_json(workspace.path / "workflows" / f"{args.workflow_run}.json")
    if record["status"] != "completed":
        parser.error("Workflow is not completed")
    result = {"case": record["definition"].get("name", args.workflow_run),
              "artifacts": {key: row["artifact_id"] for key, row in record["steps"].items()}}
else:
    result = read_json(workspace.path / "demo-result.json")
for artifact_id in result["artifacts"].values():
    workspace.resolve(artifact_id)

def report(key):
    root, _ = workspace.resolve(result["artifacts"][key])
    return read_json(root / "report.json")

residual, controller, policy = report("corrected"), report("controller"), report("optimized")
summaries = {}
for key in ("expert", "imitation", "optimized"):
    metrics = [c[key]["metrics"] for c in policy["comparison"]]
    summaries[key] = {name: float((np.max if name.startswith("max_") else np.mean)([m[name] for m in metrics]))
                      for name in ("normalized_rmse", "rmse", "max_state_violation", "max_input_violation",
                                   "input_total_variation", "mean_control_seconds")}
    for name in ("max_output_violation", "max_rate_violation"):
        summaries[key][name] = max(m[name] for m in metrics) if all(name in m for m in metrics) else None
output = {"case": result["case"], "workspace": str(workspace.path), "residual_test": residual["test"],
          "residual_test_mse_improvement": 1 - residual["test"]["corrected_mse"] / residual["test"]["nominal_mse"],
          "controller_tuned_candidate_accepted": controller["tuned_candidate_accepted"],
          "controller_test": {k: v for k, v in controller["test"].items() if k != "runs"},
          "held_out_policy_comparison": summaries,
          "all_artifact_files_and_dependencies_verified": True}
print(json.dumps(output, ensure_ascii=False, indent=2))
