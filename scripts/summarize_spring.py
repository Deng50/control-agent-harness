"""Export traceable slide data from a completed spring benchmark without retraining.

Usage: PYTHONPATH=src python scripts/summarize_spring.py <workspace>
The additional reserved-test comparison evaluates frozen artifacts only. It never
changes artifacts, controller parameters, datasets, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch

from control_harness.artifacts import Workspace, atomic_json, read_json
from control_harness.control import evaluate_controller
from control_harness.data import read_transitions
from control_harness.modeling import describe_execution, load_model, tensor
from control_harness.policy import read_expert
from control_harness.runtime import OperationContext


STAGES = ("nominal", "reference", "data", "corrected", "controller", "expert", "imitation", "optimized")
CONSTRAINT_KEYS = ("max_state_violation", "max_input_violation", "max_output_violation", "max_rate_violation")
METRIC_KEYS = ("rmse", "normalized_rmse", "mean_control_seconds", "p95_control_seconds",
               "input_total_variation", "control_effort", *CONSTRAINT_KEYS)


def csv_file(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Preserve every supplied row; UTF-8 BOM supports the Windows Excel default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def partition_rows(data: dict, masks: dict) -> tuple[list[str], dict]:
    labels = np.full(len(data["trajectory_id"]), "", dtype=object)
    summary = {}
    for part, mask in masks.items():
        if np.any(labels[mask] != ""):
            raise ValueError("Dataset partitions overlap")
        labels[mask] = part
        summary[part] = {"samples": int(mask.sum()),
                         "trajectory_ids": np.unique(data["trajectory_id"][mask]).astype(int).tolist()}
    if np.any(labels == ""):
        raise ValueError("Some dataset rows do not belong to a declared partition")
    return labels.tolist(), summary


def trajectory_rows(run: dict, method: str, purpose: str, dt: float) -> list[dict]:
    scenario = run["scenario"]
    rows = []
    for value in run["trajectory"]:
        step = int(value["step"])
        diagnostics = value.get("diagnostics", {})
        rows.append({"purpose": purpose, "scenario": scenario["name"], "method": method,
                     "step": step, "time_s": step * dt, "next_time_s": (step + 1) * dt,
                     "q_m": value["x"][0], "v_m_per_s": value["x"][1],
                     "u_N": value["u"][0], "d_N": value["d"][0],
                     "reference_m": value["reference"][0],
                     "next_q_m": value["next_x"][0], "next_v_m_per_s": value["next_x"][1],
                     "next_output_m": value["output"][0],
                     "solver_seconds": diagnostics.get("solve_seconds", ""),
                     "solver_iterations": diagnostics.get("iterations", "")})
    return rows


def metric_row(run: dict, method: str, purpose: str) -> dict:
    checks = run.get("constraint_checks", {})
    return {"purpose": purpose, "scenario": run["scenario"]["name"], "method": method,
            **{key: run["metrics"].get(key) for key in METRIC_KEYS},
            "state_checked": checks.get("state"), "input_checked": checks.get("input"),
            "output_checked": checks.get("output"), "rate_checked": checks.get("input_rate"),
            "steps": len(run["trajectory"])}


def free_decay_data(models: dict, interface, dt: float, sources: Path) -> list[dict]:
    """Independent own-state rollouts of already frozen models, with no tuning."""
    free_states = {name: np.array([.6, .0]) for name in models}
    free_values = {name: [state.copy()] for name, state in free_states.items()}
    free_rows = []
    for step in range(81):
        for name in models:
            free_rows.append({"step": step, "time_s": step * dt, "method": name,
                              "q_m": float(free_states[name][0]), "v_m_per_s": float(free_states[name][1]),
                              "u_N": 0., "d_N": 0.})
        if step == 80:
            break
        for name, model in models.items():
            free_states[name] = model.step(free_states[name], [0.], [0.])
            free_values[name].append(free_states[name].copy())
    csv_file(sources / "free-decay-trajectories.csv", free_rows)
    free_rmse = []
    free_actual = np.asarray(free_values["reference"])[1:]
    for name in ("nominal", "corrected"):
        error = np.asarray(free_values[name])[1:] - free_actual
        for column, state in enumerate(interface.states):
            free_rmse.append({"method": name, "state": state.name, "unit": state.unit,
                              "rmse": float(np.sqrt(np.mean(error[:, column] ** 2))), "steps": 80,
                              "metric": "independent_autonomous_multistep_state_rmse_excluding_initial_state"})
    csv_file(sources / "free-decay-rmse.csv", free_rmse)
    atomic_json(sources / "free-decay-config.json", {"initial_state": [.6, 0.], "control_N": 0.,
                "disturbance_N": 0., "steps": 80, "sample_time_s": dt,
                "scope": "additional_independent_simulation_not_used_for_training_tuning_or_selection",
                "semantics": "each_model_uses_its_own_previous_predicted_state"})
    return free_rmse


def export(workspace_path: Path, model_only: bool = False) -> dict:
    if not (workspace_path / "manifest.json").is_file():
        raise ValueError("Select an existing completed workspace containing manifest.json")
    record = read_json(workspace_path / "case-run.json")
    allowed_statuses = {"running", "verified"} if model_only else {"verified"}
    if record.get("case") != "spring-damper" or record.get("status") not in allowed_statuses:
        raise ValueError("case-run.json must identify an admissible spring-damper execution")
    artifact_ids = record["artifacts"]
    selected_stages = STAGES[:4] if model_only else STAGES
    if set(selected_stages) - artifact_ids.keys():
        raise ValueError("case-run.json does not contain all required explicit artifact IDs")
    workspace = Workspace(workspace_path)
    roots, metadata = {}, {}
    for name in selected_stages:
        roots[name], metadata[name] = workspace.resolve(artifact_ids[name])
        if metadata[name]["validation"].get("status") != "verified":
            raise ValueError(f"Artifact {name} did not pass validation")

    output = workspace.path / "presentation"
    sources = output / "source-data"
    sources.mkdir(parents=True, exist_ok=True)
    report_stages = ("corrected",) if model_only else ("corrected", "controller", "imitation", "optimized")
    reports = {name: read_json(roots[name] / "report.json") for name in report_stages}
    models = {name: load_model(workspace, artifact_ids[name]) for name in ("nominal", "reference", "corrected")}
    interface = models["nominal"].interface
    if (interface.nx, interface.nu, interface.ny, interface.nd) != (2, 1, 1, 1):
        raise ValueError("This presentation exporter requires spring dimensions 2/1/1/1")
    if not np.isclose(interface.sample_time, .2):
        raise ValueError("This benchmark is declared at a 0.2-second sample period")
    for model in models.values():
        model.interface.compatible(interface)
    dt = float(interface.sample_time)
    model_summary = {"equation": "m*q_ddot+c*q_dot+k*q=u+d", "sample_time_s": dt,
                     "source_kind": "explicit_reference_simulation_not_measurements", "models": {}}
    physical_parameters = {}
    for name, model in models.items():
        skeleton = model.nominal if name == "corrected" else model
        entry = {"artifact_id": artifact_ids[name], "interface": model.interface.to_dict(),
                 "execution": describe_execution(model),
                 "continuous_linear_skeleton": {key: getattr(skeleton, f"derivative_{key}").tolist()
                                                  for key in ("A", "B", "E", "bias")},
                 "discrete_zoh_linear_skeleton": {key: getattr(skeleton, key).tolist()
                                                    for key in ("A", "B", "E", "C", "bias")}}
        if name == "corrected":
            entry.update(residual_equation="x_next=Ad_nom*x+Bd_nom*u+Ed_nom*d+NN([q,v,u,d])",
                         linear_matrix_warning="The displayed matrices are the nominal backbone; the corrected map also includes a neural residual, so it has no single global corrected A/B/E matrix.",
                         network=read_json(roots[name] / "network.json"))
        else:
            entry["source_model"] = read_json(roots[name] / "model.json")
            mass = 1.0 / float(skeleton.derivative_B[1, 0])
            physical_parameters[name] = {
                "mass_kg": mass,
                "damping_N_s_per_m": -float(skeleton.derivative_A[1, 1]) * mass,
                "stiffness_N_per_m": -float(skeleton.derivative_A[1, 0]) * mass,
            }
        model_summary["models"][name] = entry
    atomic_json(sources / "models.json", model_summary)

    # Full transition table and all held-out one-step predictions, with no row subsampling.
    data, masks, _ = read_transitions(workspace, artifact_ids["data"])
    labels, transition_split = partition_rows(data, masks)
    csv_file(sources / "transition-data.csv", [{
        "row": index, "trajectory_id": int(data["trajectory_id"][index]), "partition": labels[index],
        "q_m": float(data["x"][index, 0]), "v_m_per_s": float(data["x"][index, 1]),
        "u_N": float(data["u"][index, 0]), "d_N": float(data["d"][index, 0]),
        "next_q_m": float(data["next_x"][index, 0]), "next_v_m_per_s": float(data["next_x"][index, 1])
    } for index in range(len(data["x"]))])
    test_indexes = np.flatnonzero(masks["test"])
    with torch.no_grad():
        args = [tensor(data[key][masks["test"]]) for key in ("x", "u", "d")]
        predictions = {name: models[name](*args).cpu().numpy() for name in ("nominal", "corrected")}
    actual = data["next_x"][masks["test"]]
    residual_rows = []
    for local, index in enumerate(test_indexes):
        residual_rows.append({"source_row": int(index), "trajectory_id": int(data["trajectory_id"][index]),
            "true_next_q_m": float(actual[local, 0]), "true_next_v_m_per_s": float(actual[local, 1]),
            **{f"{name}_next_q_m": float(predictions[name][local, 0]) for name in predictions},
            **{f"{name}_next_v_m_per_s": float(predictions[name][local, 1]) for name in predictions}})
    csv_file(sources / "residual-test-predictions.csv", residual_rows)
    residual_metrics = []
    for name, prediction in predictions.items():
        for column, state in enumerate(interface.states):
            residual_metrics.append({"method": name, "state": state.name, "unit": state.unit,
                "rmse": float(np.sqrt(np.mean((prediction[:, column] - actual[:, column]) ** 2))),
                "samples": len(actual), "metric": "held_out_one_step_next_state_rmse"})
    csv_file(sources / "residual-test-rmse.csv", residual_metrics)
    log_files = (("corrected", "residual-training.csv"),) if model_only else (
        ("corrected", "residual-training.csv"), ("imitation", "imitation-training.csv"),
        ("optimized", "policy-optimization.csv"))
    for stage, filename in log_files:
        csv_file(sources / filename, reports[stage]["log"])
    free_rmse = free_decay_data(models, interface, dt, sources)
    if model_only:
        summary = {
            "case": "spring-damper", "scope": "first_four_verified_model_artifacts_only",
            "full_workflow_completed": record["status"] == "verified",
            "artifacts": {name: artifact_ids[name] for name in selected_stages},
            "source": "explicit_mathematical_reference_simulator_not_measured_plant_data",
            "all_artifact_files_and_dependencies_verified": True,
            "sample_time_s": dt, "physical_parameters": physical_parameters,
            "dimensions": {"state": 2, "control": 1, "output": 1, "disturbance": 1,
                           "residual_network": [4, 2]},
            "transition_split": transition_split,
            "model_one_step_test_rmse": residual_metrics, "free_decay_multistep_rmse": free_rmse,
            "residual": {"trained_epochs": reports["corrected"]["trained_epochs"],
                         "optimizer_updates": reports["corrected"]["optimizer_updates"],
                         "acceptance": reports["corrected"]["acceptance"],
                         "test": reports["corrected"]["test"],
                         "test_used_for_training_or_selection": reports["corrected"]["test_used_for_training_or_selection"]},
            "source_files": sorted(path.name for path in sources.iterdir() if path.is_file()),
            "integrity_notes": [
                "Verification is limited to the four declared model/data artifacts and their dependencies.",
                "Controller and policy artifacts are not loaded or evaluated in model-only mode.",
                "Each state retains its own RMSE unit; all test rows are retained.",
                "Free decay uses each model's own predicted state at every subsequent sample.",
                "No training, tuning, checkpoint selection or artifact mutation occurs."]}
        atomic_json(output / "metrics-summary-model.json", summary)
        return summary

    controller = reports["controller"]
    controller_config = read_json(roots["controller"] / "config.json")
    descriptor = read_json(roots["controller"] / "controller.json")
    csv_file(sources / "controller-ga-trials.csv", [{"trial": int(row["index"]) + 1,
        "generation": row["generation"], "score": row["score"],
        "hard_constraints_passed": row["hard_constraints_passed"],
        **{f"parameter_{key}": value for key, value in row["parameters"].items()},
        "lineage_json": json.dumps(row.get("lineage", {}), ensure_ascii=False),
        "error": row.get("error", "")} for row in controller["trials"]])
    control_metrics, control_trajectories = [], []
    for report_key, method in (("verification_baseline", "baseline"), ("verification_candidate", "candidate")):
        for run in controller[report_key]["runs"]:
            control_metrics.append(metric_row(run, method, "independent_verification"))
            control_trajectories.extend(trajectory_rows(run, method, "independent_verification", dt))
    csv_file(sources / "controller-verification-metrics.csv", control_metrics)
    csv_file(sources / "controller-verification-trajectories.csv", control_trajectories)

    # This is a post-selection comparison on exactly the same reserved conditions.
    # Every controller parameter and model/checkpoint is already frozen.
    final_evaluations = {}
    final_trajectories, final_metrics = [], []
    for method, parameters in (("baseline", controller_config["parameters"]), ("chosen", descriptor["parameters"])):
        evaluation = evaluate_controller(models["corrected"], models["reference"], parameters,
                                         controller_config["test_scenarios"], OperationContext())
        final_evaluations[method] = evaluation
        for run in evaluation["runs"]:
            final_metrics.append(metric_row(run, method, "reserved_final_test_frozen_comparison"))
            final_trajectories.extend(trajectory_rows(run, method, "reserved_final_test_frozen_comparison", dt))
    atomic_json(sources / "controller-final-test.json", {
        "scope": "post_selection_frozen_parameters_comparison_no_tuning_or_checkpoint_selection",
        "baseline_parameters": controller_config["parameters"], "chosen_parameters": descriptor["parameters"],
        "evaluations": final_evaluations})
    csv_file(sources / "controller-final-test-trajectories.csv", final_trajectories)
    csv_file(sources / "controller-final-test-metrics.csv", final_metrics)

    expert_data, expert_masks, collection, _ = read_expert(workspace, artifact_ids["expert"])
    expert_labels, expert_split = partition_rows(expert_data, expert_masks)
    csv_file(sources / "expert-data.csv", [{
        "row": index, "trajectory_id": int(expert_data["trajectory_id"][index]),
        "scenario": collection["scenarios"][int(expert_data["trajectory_id"][index])]["scenario"]["name"],
        "partition": expert_labels[index], "q_m": float(expert_data["x"][index, 0]),
        "v_m_per_s": float(expert_data["x"][index, 1]), "reference_m": float(expert_data["reference"][index, 0]),
        "d_N": float(expert_data["d"][index, 0]), "u_N": float(expert_data["u"][index, 0])
    } for index in range(len(expert_data["u"]))])
    policy_rows, policy_metrics, speedups = [], [], []
    comparisons = reports["optimized"]["comparison"]
    if not comparisons:
        raise ValueError("Optimized policy report has no held-out comparisons")
    for comparison in comparisons:
        expert_time = comparison["expert"]["metrics"]["mean_control_seconds"]
        for method in ("expert", "imitation", "optimized"):
            run = comparison[method]
            policy_rows.extend(trajectory_rows(run, method, "held_out_expert_scenario", dt))
            policy_metrics.append(metric_row(run, method, "held_out_expert_scenario"))
            if method != "expert":
                speedups.append({"scenario": comparison["scenario"]["name"], "method": method,
                    "expert_mean_control_ms": expert_time * 1000,
                    "policy_mean_control_ms": run["metrics"]["mean_control_seconds"] * 1000,
                    "mean_time_ratio_expert_over_policy": expert_time / run["metrics"]["mean_control_seconds"]})
    csv_file(sources / "policy-heldout-trajectories.csv", policy_rows)
    csv_file(sources / "policy-heldout-metrics.csv", policy_metrics)
    csv_file(sources / "policy-heldout-timing.csv", speedups)
    policy_summary = {}
    for method in ("expert", "imitation", "optimized"):
        values = [comparison[method]["metrics"] for comparison in comparisons]
        aggregate = {key: float((np.max if key.startswith("max_") else np.mean)([row[key] for row in values]))
                     for key in METRIC_KEYS}
        aggregate["scenarios"] = len(values)
        aggregate["samples"] = sum(len(comparison[method]["trajectory"]) for comparison in comparisons)
        aggregate["aggregation"] = "unweighted_mean_of_scenario_metrics_except_max_violations"
        aggregate["rate_constraint_declared"] = all(comparison[method]["constraint_checks"].get("input_rate")
                                                     for comparison in comparisons)
        policy_summary[method] = aggregate
    for method in ("imitation", "optimized"):
        policy_summary[method]["mean_time_ratio_expert_over_policy"] = (
            policy_summary["expert"]["mean_control_seconds"] / policy_summary[method]["mean_control_seconds"])

    lineage = []
    for stage in STAGES:
        meta = metadata[stage]
        lineage.append({"stage": stage, "artifact_id": artifact_ids[stage], "kind": meta["kind"],
                        "status": meta["validation"]["status"], "parents": meta["parents"],
                        "file_sha256": meta["files"]})
    atomic_json(sources / "artifact-lineage.json", {"artifacts": lineage})
    summary = {
        "case": "spring-damper", "artifacts": artifact_ids,
        "source": "explicit_mathematical_reference_simulator_not_measured_plant_data",
        "all_artifact_files_and_dependencies_verified": True,
        "sample_time_s": dt,
        "physical_parameters": physical_parameters,
        "dimensions": {"state": 2, "control": 1, "output": 1, "disturbance": 1,
                       "residual_network": [4, 2], "policy_network": [4, 1]},
        "transition_split": transition_split, "expert_split": expert_split,
        "model_one_step_test_rmse": residual_metrics, "free_decay_multistep_rmse": free_rmse,
        "residual": {"trained_epochs": reports["corrected"]["trained_epochs"],
                     "optimizer_updates": reports["corrected"]["optimizer_updates"],
                     "acceptance": reports["corrected"]["acceptance"],
                     "test": reports["corrected"]["test"],
                     "test_used_for_training_or_selection": reports["corrected"]["test_used_for_training_or_selection"]},
        "controller": {"baseline_parameters": controller_config["parameters"],
                       "chosen_parameters": descriptor["parameters"],
                       "tuned_candidate_accepted": controller["tuned_candidate_accepted"],
                       "trials": len(controller["trials"]),
                       "verification_baseline_score": controller["verification_baseline"]["score"],
                       "verification_candidate_score": controller["verification_candidate"]["score"],
                       "verification_metrics": control_metrics,
                       "final_test": {method: {key: value for key, value in evaluation.items() if key != "runs"}
                                      for method, evaluation in final_evaluations.items()},
                       "final_test_metrics": final_metrics,
                       "post_selection_comparison_did_not_change_parameters": True},
        "policy": {"held_out_comparison": policy_summary,
                   "held_out_per_scenario": policy_metrics,
                   "parameter_count": reports["imitation"]["parameter_count"],
                   "imitation_epochs": reports["imitation"]["trained_epochs"],
                   "optimization_epochs": reports["optimized"]["trained_epochs"],
                   "optimization_algorithm": reports["optimized"]["algorithm"],
                   "learned_critic": reports["optimized"]["learned_critic"],
                   "test_used_for_updates_or_checkpoint_selection": reports["optimized"]["test_used_for_updates_or_checkpoint_selection"],
                   "paired_timing": speedups},
        "timing_scope": {"device": "CPU", "torch_num_threads": torch.get_num_threads(),
                         "python": platform.python_version(), "torch": torch.__version__,
                         "meaning": "wall_clock_control_call_durations_from_the_same_saved_held_out_scenarios_in_this_run",
                         "limitation": "one_run_on_this_machine_with_no_hardware_independent_speedup_or_repeated_run_confidence_interval_claim"},
        "integrity_notes": [
            "All transition and expert rows are exported; no visual subsampling is applied.",
            "Data are partitioned by independent trajectory/scenario; held-out tests do not train or select checkpoints.",
            "State-specific RMSE retains each state's physical unit; the mixed-state scalar MSE is not labelled with one physical unit.",
            "Each closed-loop row is x(k),u(k),x(k+1); current time is k*dt and output time is (k+1)*dt.",
            "No input-rate constraint is asserted when none was declared, even though max_rate_violation is numerically zero.",
            "Residual learning corrects the integrated next-state map, not a full differential-equation PINN.",
            "Policy improvement is differentiable physical-rollout policy gradient without a learned critic, not PPO/SAC.",
            "Single declared seeds are used; no confidence intervals, statistical significance or universal stability guarantees are claimed."
        ],
        "source_files": sorted(path.name for path in sources.iterdir() if path.is_file())}
    atomic_json(output / "metrics-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--model-only", action="store_true",
                        help="Export only the first four verified model/data artifacts while later stages run")
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = export(args.workspace.expanduser().resolve(), model_only=args.model_only)
    print(json.dumps({"case": summary["case"], "verified": summary["all_artifact_files_and_dependencies_verified"],
                      "presentation": str(args.workspace.resolve() / "presentation"),
                      "source_file_count": len(summary["source_files"]),
                      "scope": summary.get("scope", "full_workflow"),
                      **({} if args.model_only else {
                          "controller_candidate_accepted": summary["controller"]["tuned_candidate_accepted"],
                          "policy_test_scenarios": summary["policy"]["held_out_comparison"]["expert"]["scenarios"]})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
