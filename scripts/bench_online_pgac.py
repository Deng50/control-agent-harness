"""Train all registered ablation arms, then evaluate frozen versions on the original Plant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from bench_pgac_cases import PROTOCOL, ROOT, freeze, prepare_materials, read
from control_harness.artifacts import Workspace, atomic_json, file_hash
from control_harness.case_plans import FIXTURE_KIND, propose_plan
from control_harness.case_requirements import confirmation_requirements
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext


class Run:
    def __init__(self, root, case):
        self.case = case
        self.root = root / case
        self.materials = prepare_materials(self.root, case)
        self.workspace = Workspace(self.root / "workspace")
        self.log = self.root / "operations.json"
        self.ids = read(self.log) if self.log.exists() else {}
        # Product benchmark harness, explicitly labeled fixture confirmations; no model dialogue claimed.
        atomic_json(self.workspace.path / ".case-plans/test-fixture.json",
                    {"kind": FIXTURE_KIND, "workspace": str(self.workspace.path), "confirmation_source": "test_fixture"})

    def config(self, name):
        return read(self.materials / (name + ".json"))

    def stage(self, key, operation, inputs, config):
        self.workspace = Workspace(self.workspace.path)
        if key in self.ids:
            rec = self.ids[key]
            if rec["operation"] != operation or rec["config"] != config or rec["inputs"] != inputs:
                raise ValueError(f"Existing stage differs from frozen request: {key}")
            self.workspace.resolve(rec["artifact_id"], allow_rejected=True)
            return rec["artifact_id"]
        print(f"{self.case}  {key}: {operation}", flush=True)
        request = dict(operation=operation, workspace=str(self.workspace.path), inputs=inputs, config=dict(config))
        refs = {}
        for row in confirmation_requirements(self.workspace, request):
            proposal = propose_plan(self.workspace, row["scope"], row["expected_payload"], row["bindings"], dependencies=refs.copy())
            refs[row["scope"]] = {k: proposal[k] for k in ("plan_id", "version_hash")}
        if refs:
            request["config"]["plan_refs"] = refs
        result = run_operation(request, OperationContext.test_fixture(self.workspace))
        self.ids[key] = dict(operation=operation, inputs=inputs, config=config,
                             artifact_id=result["artifact_id"], status=result["status"])
        atomic_json(self.log, self.ids)
        self.workspace = Workspace(self.workspace.path)
        return result["artifact_id"]

    def id(self, key):
        return self.ids[key]["artifact_id"]

    def artifact(self, key):
        return self.workspace.resolve(self.id(key), allow_rejected=True)[0]

    def prepare(self):
        for name, filename in (("plant", "plant.py"), ("expert", "mpc.py")):
            mat = self.stage(name + "_material", "register_material", {"local_path": str(self.materials / filename)}, {})
            self.stage(name, "register_" + name, {"material": mat}, {"adapter": self.config(name + "-adapter")})
        self.stage("model", "register_model", {"model_path": str(self.materials / "model.json")},
                   {"source": {"kind": "reconstructed_equations", "description": "Exact discrete ZOH of checked-in physical example"}})
        self.stage("sample", "sample_model", {"model": self.id("plant")}, {"sampling": self.config("sampling")})
        sample = self.artifact("sample")
        split = read(sample / "split.json")
        with np.load(sample / "transitions.npz") as arr:
            row = int(np.flatnonzero(np.isin(arr["trajectory_id"], split["validation"]))[3])
        validation = {"mode": "exact_rewrite", "steps": 50, "probes": [{"row": row, "kind": "smooth"}]}
        self.stage("receipt", "validate_differentiable_model", dict(evaluation_model=self.id("plant"),
                   model=self.id("model"), dataset=self.id("sample")), validation)
        self.stage("expert_data", "collect_external_expert", dict(expert=self.id("expert"), plant=self.id("plant")), self.config("collection"))
        self.stage("imitation", "train_external_imitation", dict(dataset=self.id("expert_data")), self.config("imitation"))
        report = read(self.artifact("imitation") / "report.json")
        metrics = report["offline"]["test"]["projected"]
        lim = self.config("plant-adapter")["interface"]["controls"][0]["upper"]
        quality = float(np.mean(metrics["rmse"])) / (2 * lim)
        print(f"{self.case} independent imitation action RMSE/span = {quality:.5f}", flush=True)
        if quality >= .05:
            raise ValueError("Failed preregistered baseline quality floor")

    def train(self, arm):
        self.stage(arm, "optimize_external_policy", dict(policy=self.id("imitation"), model=self.id("model"),
                   validation=self.id("receipt")), self.config(arm))

    def evaluate(self, arm):
        self.stage("evaluate_" + arm, "evaluate_external_pgac", dict(candidate=self.id(arm)), self.config("evaluation"))

    def summary(self):
        result = {}
        physical = self.config("plant-adapter")["interface"]
        rates = np.array(self.config("expert-adapter")["input_rate_bounds"]).T
        def bounds(kind):
            return tuple(np.asarray([[v["lower"], v["upper"]] for v in physical[kind]]).T)
        def excess(values, lower, upper):
            return np.maximum(np.maximum(lower - values, values - upper), 0).max(axis=1)
        for arm in PROTOCOL["arms"][1:]:
            artifact = self.artifact("evaluate_" + arm)
            report = read(artifact / "report.json")
            train_report = read(self.artifact(arm) / "report.json")
            entry = dict(selected=report["final_acceptance"]["selected_version"], reports=str(artifact),
                         baseline_weights=train_report["initial_weights_sha256"], optimizer_updates=train_report["optimizer_updates"],
                         expert_training_decisions=sum(r["steps"] for r in self.config("collection")["scenarios"] if r["split"] == "train"),
                         expert_collection_decisions=sum(r["steps"] for r in self.config("collection")["scenarios"]),
                         extra_training_mpc_calls=0, validation_selection_sha256=report["selection_sha256"],
                         reuse={"imitation": self.id("imitation"), "qualified_model": self.id("receipt")}, phases={})
            for split in ("validation", "test"):
                conditions = [c for c in self.config("evaluation")["scenarios"] if c["split"] == split]
                rows = {}
                for policy in ("expert", "imitation", "candidate"):
                    data = report["closed_loop"][split][policy]["quadratic"]
                    rec = dict(J=data["score"], feasible=data["feasible"], per_condition=[])
                    with np.load(artifact / f"{split}-{policy}-rollout.npz") as raw:
                        for scenario, score in zip(conditions, data["per_scenario"], strict=True):
                            idx = raw["trajectory_id"] == score["trajectory_id"]
                            du = raw["u"][idx] - raw["previous_u"][idx]
                            err = raw["next_y"][idx] - raw["reference"][idx]
                            x = raw["next_x"][idx]
                            violation = np.maximum.reduce([excess(x, *bounds("states")), excess(raw["next_y"][idx], *bounds("outputs")),
                                                           excess(raw["u"][idx], *bounds("controls")),
                                                           excess(du, *rates)])
                            rec["per_condition"].append(dict(name=scenario["name"], family=scenario["family"], noise_seed=scenario["noise_seed"],
                                    J=score["score"], rmse=float(np.sqrt(np.mean(err**2))), delta_u_rms=float(np.sqrt(np.mean(du**2))),
                                    action_rms=float(np.sqrt(np.mean(raw["u"][idx]**2))), normalized_cost_terms=score["terms"],
                                    peak_abs_states=np.max(np.abs(x), axis=0).tolist(), violations=int(np.sum(violation > 1e-6)),
                                    max_violation=float(violation.max())))
                    for field in ("rmse", "action_rms", "delta_u_rms"):
                        rec[field] = np.mean([row[field] for row in rec["per_condition"]])
                    rec["median_call_us"] = report["controller_benchmark"][policy]["median"] * 1e6
                    rows[policy] = rec
                entry["phases"][split] = rows
            online = train_report.get("replay", {})
            entry["replay_accounting"] = online
            entry["student_steps"] = online.get("student_control_steps", 0)
            entry["extra_training_mpc_calls"] = online.get("extra_mpc_calls", 0)
            entry["candidate_epoch"] = max([0] + [event["epoch"] for event in train_report["log"] if event["selected_by_validation"]])
            entry["selected_bundle"] = str(artifact / "bundle")
            entry["deployment"] = verify(self, artifact, report) if report["final_acceptance"]["approved"] else None
            result[arm] = entry
        return result


def verify(run, artifact, report):
    command = [sys.executable, "-I", "-B", str(ROOT / "scripts/bench_pgac_verify.py"), str(artifact),
               str(run.materials), report["final_acceptance"]["selected_version"]]
    return json.loads(subprocess.check_output(command, encoding="utf-8"))


def publish_summary(directory, runs, root):
    """Keep the Git-tracked result portable; absolute execution paths belong to the run directory."""
    result = {"schema_version": 1, "protocol_sha256": file_hash(root / "spring/materials/protocol.json"),
              "frozen_weights": read(root / "frozen-candidates.json"), "frozen_inputs": read(root / "frozen-inputs.json"), "cases": {}}
    for run in runs:
        arms = read(run.root / "summary.json")
        offline = read(run.artifact("imitation") / "report.json")["offline"]["test"]["projected"]
        span = 2 * run.config("plant-adapter")["interface"]["controls"][0]["upper"]
        workspace = run.root.relative_to(ROOT).as_posix() if run.root.is_relative_to(ROOT) else run.case
        case = dict(imitation_rmse_over_action_span=float(np.mean(offline["rmse"])) / span,
                    baseline_tensors_sha256=arms["online"]["baseline_weights"], workspace=workspace, policies={})
        for name, arm, policy in [("mpc", "online", "expert"), ("imitation", "online", "imitation"),
                                  *[(key, key, "candidate") for key in PROTOCOL["arms"][1:]]]:
            source = arms[arm]
            scores = {phase: source["phases"][phase][policy] for phase in ("validation", "test")}
            result_row = dict(scores=scores)
            if policy == "candidate":
                base = source["phases"]["test"]["imitation"]
                result_row.update(selected=source["selected"], selected_epoch=source["candidate_epoch"],
                    cost=dict(common_expert_training_decisions=source["expert_training_decisions"], total_common_expert_collection_decisions=source["expert_collection_decisions"],
                              expert_control_function_evaluations_for_training=3*source["expert_training_decisions"],
                              expert_control_function_evaluations_for_entire_collection=3*source["expert_collection_decisions"],
                              expert_function_calls_per_decision=3, slsqp_solver_calls="not_counted; feasible unconstrained QP solutions return directly",
                              extra_training_mpc_calls=source["extra_training_mpc_calls"], student_plant_steps=source["student_steps"],
                              optimizer_updates=source["optimizer_updates"]),
                    deployment=source["deployment"], selection_sha256=source["validation_selection_sha256"],
                    relative_test_improvement=1 - scores["test"]["J"] / base["J"],
                    worst_test_relative_regression=max(x["J"] / b["J"] - 1 for x, b in
                                                       zip(scores["test"]["per_condition"], base["per_condition"], strict=True)),
                    artifact_id=run.id("evaluate_" + arm))
            case["policies"][name] = result_row
        result["cases"][run.case] = case
    atomic_json(directory / "results.json", result)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=["prepare", "train", "evaluate", "summary", "all"])
    parser.add_argument("--root", type=Path, default=ROOT / "test-output/pgac-online-benchmark-v2")
    parser.add_argument("--cases", nargs="+", choices=["spring", "motor"], default=["spring", "motor"])
    parser.add_argument("--arms", nargs="+", choices=PROTOCOL["arms"][1:], default=PROTOCOL["arms"][1:])
    parser.add_argument("--report-dir", type=Path, help="For summary/all, also write portable results.json here")
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(1)
    runs = [Run(root, case) for case in args.cases]
    for phase in (["prepare", "train", "evaluate", "summary"] if args.phase == "all" else [args.phase]):
        if phase == "evaluate":
            # Freeze the actual tensors of every arm before any blind execution across either case.
            all_runs = [Run(root, case) for case in PROTOCOL["case_sources"]]
            versions = {run.case: {arm: file_hash(run.artifact(arm) / "bundle/model.pt") for arm in PROTOCOL["arms"][1:]} for run in all_runs}
            freeze(root / "frozen-candidates.json", versions)
            inputs = {run.case: {f.name: file_hash(f) for f in sorted(run.materials.iterdir()) if f.is_file()}
                      for run in all_runs}
            freeze(root / "frozen-inputs.json", inputs)
        for run in runs:
            if phase == "prepare":
                run.prepare()
            elif phase in ("train", "evaluate"):
                for arm in args.arms:
                    getattr(run, phase)(arm)
            else:
                atomic_json(run.root / "summary.json", run.summary())
        if phase == "summary" and args.report_dir is not None:
            publish_summary(args.report_dir.resolve(), runs, root)
    print(f"Results in {root}", flush=True)


if __name__ == "__main__":
    main()
