"""Run the tracking-primary comparison through public Harness operations with a prospective release freeze."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bench_online_pgac import Run, verify
from bench_pgac_verify import autonomous_trajectory, load_module
from bench_pgac_cases import ROOT, read, freeze
from bench_tracking_cases import PROTOCOL, TRACKING_ASSETS, make_materials
from control_harness.artifacts import atomic_json, file_hash
from control_harness.contracts import ModelInterface
from control_harness.external_pgac_evaluation import quadratic_result, _subplan, validation_choice, final_acceptance
from control_harness.pgac_tracking import tracking_gate
from control_harness.operations import run_operation
from control_harness.runtime import OperationContext
from control_harness.restricted_pgac import _state_hash


def baseline_identity(run):
    weights = torch.load(run.artifact("imitation") / "bundle/model.pt", map_location="cpu", weights_only=True)
    actual = _state_hash(weights)
    original = read(ROOT / "examples/pgac-online/results.json")["cases"][run.case]["baseline_tensors_sha256"]
    if actual != original:
        raise ValueError(f"Different imitation weights in {run.case}")
    return actual


def freeze_inputs(root, runs):
    material = {}
    for run in runs:
        material[run.case] = {f.name: file_hash(f) for f in sorted(run.materials.iterdir())
                              if f.is_file() and f.name != "online.json"}
    freeze(root / "frozen-inputs.json", material)


def select(runs, root, previous):
    """Read registered training logs only. No original evaluation operation has been admitted."""
    result = {}
    for run in runs:
        history = {}
        for design in PROTOCOL["designs"]:
            artifact = run.artifact(design)
            report = read(artifact / "report.json")
            best = report["best_epoch"]
            epoch = next((r for r in report["log"] if r["epoch"] == best), None)
            metrics = report["selected_predictor_validation"]["tracking"]
            history[design] = dict(epoch=best, qualified=bool(epoch and epoch["eligible_for_best_checkpoint"]),
                tracking_rmse=metrics["tracking_rmse"], action_rms=metrics["action_rms"],
                increment_rms=metrics["increment_rms"], baseline=report["baseline_predictor_validation"]["tracking"]["tracking_rmse"],
                eligible_epochs=[r["epoch"] for r in report["log"] if r["eligible_for_best_checkpoint"]],
                candidate_id=run.id(design), weights_sha256=file_hash(artifact / "bundle/model.pt"))
        design = min(history, key=lambda k: (not history[k]["qualified"], history[k]["tracking_rmse"]))
        freeze(run.materials / "online.json", run.config(design))
        result[run.case] = dict(design=design, predictor_comparison=history, imitation_tensor_sha256=baseline_identity(run),
                                prior_online_comparator=historical_policy(previous, run.case)[1],
                                config_sha256=file_hash(run.materials / "online.json"))
    freeze(root / "frozen-candidates.json", result)
    return result


def evaluate(run, frozen):
    choice = frozen[run.case]
    name = choice["design"]
    if file_hash(run.artifact(name) / "bundle/model.pt") != choice["predictor_comparison"][name]["weights_sha256"]:
        raise ValueError("Frozen candidate changed")
    if file_hash(run.materials / "online.json") != choice["config_sha256"]:
        raise ValueError("Frozen design configuration changed")
    run.stage("evaluate", "evaluate_external_pgac", dict(candidate=run.id(name)), run.config("evaluation"))


def historical_policy(previous, case):
    ops = read(previous / case / "operations.json")
    artifact = previous / case / "workspace/artifacts" / ops["online"]["artifact_id"]
    past_selection = read(previous / case / "workspace/artifacts" / ops["evaluate_online"]["artifact_id"] / "report.json")
    weight = file_hash(artifact / "bundle/model.pt")
    registered = read(ROOT / "examples/pgac-online/results.json")["frozen_weights"][case]["online"]
    if weight != read(previous / "frozen-candidates.json")[case]["online"] or weight != registered:
        raise ValueError("Historical comparison weights have changed")
    return artifact, dict(model_sha256=weight, original_selection=past_selection["selected_version"],
        purpose="Fixed prior online candidate on the new inputs. Report only; excluded from selection and retuning.")


def historical_scores(run, previous, binding, conditions, interface, rates):
    artifact, identity = historical_policy(previous, run.case)
    if identity != binding:
        raise ValueError("Historical comparator differs from pre-blind binding")
    physical = run.config("plant-adapter")
    policy = load_module(artifact / "bundle/policy_runtime.py", "old_runtime").load_external_policy(artifact / "bundle")
    plant = load_module(run.materials / "plant.py", "original_source")
    span = interface["controls"][0]["upper"] - interface["controls"][0]["lower"]
    yspan = interface["outputs"][0]["upper"] - interface["outputs"][0]["lower"]
    qrs = run.config("evaluation")["evaluation"]
    rows = []
    for s in conditions:
        trace = autonomous_trajectory(policy, plant, physical, s)
        rmse = float(np.sqrt(np.mean((trace["next_y"]-trace["reference"])**2)))
        u = float(np.sqrt(np.mean(trace["u"]**2)))
        du = float(np.sqrt(np.mean((trace["u"]-trace["previous_u"])**2)))
        rows.append(dict(name=s["name"], family=s["family"], seed=s["noise_seed"],
            tracking_rmse=rmse, action_rms=u, increment_rms=du,
            J=qrs["q"]*(rmse/yspan)**2 + qrs["r"]*(u/span)**2 + qrs["s"]*(du/span)**2,
            constraints=violations(trace, interface, rates)))
    feasible = all(g["count"] == 0 for r in rows for g in r["constraints"].values())
    return {**{k:float(np.mean([r[k] for r in rows])) for k in ("tracking_rmse","action_rms","increment_rms","J")},
            "feasible":feasible,"per_condition":rows,"median_call_us":None}


def violations(trace, interface, rate):
    groups = {}
    for key, data in (("states", trace["next_x"]), ("outputs", trace["next_y"]), ("controls", trace["u"]),
                      ("increments", trace["u"] - trace["previous_u"])):
        bounds = np.array(rate if key == "increments" else [[v["lower"], v["upper"]] for v in interface[key]])
        excess = np.maximum(np.maximum(bounds[:, 0] - data, data - bounds[:, 1]), 0)
        groups[key] = dict(count=int(np.count_nonzero(excess > 1e-6)), peak=float(np.max(excess)))
    return groups


def recheck_release(release, report, interface, rate):
    """Recompute all six original rollouts and both decisions without training or checkpoint search."""
    objective = report["original_evaluation_config"]["evaluation"]
    scores, audit = {}, {}
    physical = ModelInterface.parse(interface)
    for phase in ("validation", "test"):
        scores[phase] = {}
        observed = {}
        for who, record in report["closed_loop"][phase].items():
            path = release / record["rollout_file"]
            with np.load(path, allow_pickle=False) as raw:
                result = {"arrays":raw, "status":record["status"], "action_constraints":{"rate_bounds":rate},
                          "coverage_guard":record["coverage_guard"]}
                score = quadratic_result(result, physical, _subplan(report["plan"], phase), objective,
                    report["deployment_scope"]["candidate_domain"] if who == "candidate" else None)
            if score != record["quadratic"]:
                raise ValueError(f"Frozen original raw metrics disagree with the published report: {who} {phase}")
            observed[who] = dict(raw_sha256=file_hash(path), feasible=score["feasible"])
            scores[phase][who] = score
        pair = scores[phase]
        gate = tracking_gate(pair["imitation"]["tracking"], pair["candidate"]["tracking"], objective,
                             validation=phase == "validation")
        if gate["checks"] != report["engineering_objective"]["gates"][phase]["checks"]:
            raise ValueError("Current paired RMS gates differ from the frozen report")
        audit[phase] = dict(rollouts=observed, gates=gate["checks"], passed=gate["passed"])
    choice = validation_choice(scores["validation"], objective, candidate_eligible=report["selection"]["candidate_eligible"])
    accepted = final_acceptance(choice, scores["test"], objective)
    if choice["choice"] != report["selection"]["choice"] or accepted != report["final_acceptance"]:
        raise ValueError("Current selection disagrees with original release")
    return {**audit, "validation_choice":choice["choice"], "final":accepted}


def summarize(run, frozen, previous):
    release = run.artifact("evaluate")
    report = read(release / "report.json")
    design = frozen[run.case]["design"]
    training = read(run.artifact(design) / "report.json")
    interface = run.config("plant-adapter")["interface"]
    limits = run.config("expert-adapter")["input_rate_bounds"]
    span = interface["controls"][0]["upper"] - interface["controls"][0]["lower"]
    result = dict(frozen=frozen[run.case], selected=report["selected_version"], artifact=run.id("evaluate"),
                  training_updates_per_design=training["optimizer_updates"],
                  student_steps_per_design=training["replay"]["student_control_steps"],
                  total_design_updates=sum(read(run.artifact(k) / "report.json")["optimizer_updates"] for k in PROTOCOL["designs"]),
                  total_design_student_steps=sum(read(run.artifact(k) / "report.json")["replay"]["student_control_steps"]
                                               for k in PROTOCOL["designs"]),
                  extra_training_mpc_source_invocations=training["replay"]["extra_mpc_calls"],
                  selected_weights_sha256=file_hash(release / "bundle/model.pt") if report["selected_version"] else None,
                  gates=report["engineering_objective"]["gates"], phases={})
    for phase in ("validation", "test"):
        policies = {}
        conditions = [r for r in run.config("evaluation")["scenarios"] if r["split"] == phase]
        for who in ("expert", "imitation", "candidate"):
            metric = report["closed_loop"][phase][who]["quadratic"]
            tracking = metric["tracking"]
            if tracking is None:
                policies[who] = dict(tracking_rmse=None, action_rms=None, increment_rms=None,
                                     J=metric["score"], feasible=False, checks=metric["checks"],
                                     partial=report["closed_loop"][phase][who]["report"])
                continue
            rows = []
            with np.load(release / f"{phase}-{who}-rollout.npz") as data:
                for s, t, q in zip(conditions, tracking["per_scenario"], metric["per_scenario"], strict=True):
                    mask = data["trajectory_id"] == t["trajectory_id"]
                    if int(mask.sum()) != s["steps"]:
                        raise ValueError("Missing original-Plant rows")
                    trace = {k: data[k][mask] for k in ("next_x", "next_y", "u", "previous_u")}
                    rows.append(dict(name=s["name"], family=s["family"], seed=s["noise_seed"],
                        tracking_rmse=t["tracking_rmse"], action_rms=span*t["action_rms"],
                        increment_rms=span*t["increment_rms"], J=q["score"], constraints=violations(trace, interface, limits)))
            policies[who] = dict(tracking_rmse=tracking["tracking_rmse"], action_rms=span*tracking["action_rms"],
                                 increment_rms=span*tracking["increment_rms"], J=metric["score"], feasible=metric["feasible"],
                                 median_call_us=report["controller_benchmark"][who]["median"] * 1e6, per_condition=rows)
        policies["prior_online_candidate"] = historical_scores(run, previous, frozen[run.case]["prior_online_comparator"],
                                                               conditions, interface, limits)
        result["phases"][phase] = policies
        if all(policies[w]["feasible"] for w in ("imitation", "candidate")):
            groups = {}
            pairs = zip(policies["imitation"]["per_condition"], policies["candidate"]["per_condition"], strict=True)
            for im, pg in pairs:
                if (im["name"], im["seed"]) != (pg["name"], pg["seed"]):
                    raise ValueError("Mismatched trajectory IDs or exogenous seeds")
                groups.setdefault(im["family"], []).append((im, pg))
            metrics = {}
            for family, paired in groups.items():
                before, after = map(np.mean, ([a["tracking_rmse"] for a,_ in paired], [b["tracking_rmse"] for _,b in paired]))
                worst = max(paired, key=lambda pair: pair[1]["tracking_rmse"] / pair[0]["tracking_rmse"])
                metrics[family] = dict(imitation_rmse=before, candidate_rmse=after,
                    mean_gain=1-after/before, worst_single_gain=1-worst[1]["tracking_rmse"]/worst[0]["tracking_rmse"],
                    worst_seed=worst[0]["seed"], paired_seeds=[a["seed"] for a,_ in paired])
            result.setdefault("paired_families", {})[phase] = metrics
    result["independent_delivery"] = verify(run, release, report) if report["final_acceptance"]["approved"] else None
    result["frozen_raw_release_recheck"] = recheck_release(release, report, interface, limits)
    if result["independent_delivery"]:
        request = dict(operation="prepare_bundle_archive", workspace=str(run.workspace.path),
                       inputs={"artifact_id":run.id("evaluate")}, config={})
        descriptor = run_operation(request, OperationContext.test_fixture(run.workspace))
        result["delivery_zip"] = dict(path=Path(descriptor["host_path"]).relative_to(run.workspace.path).as_posix(),
                                      sha256=descriptor["sha256"], size=descriptor["bytes"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "freeze", "evaluate", "summary", "all"))
    parser.add_argument("--root", type=Path, default=ROOT / "test-output/pgac-tracking-v1")
    parser.add_argument("--previous", type=Path, default=ROOT / "test-output/pgac-online-benchmark-v2",
                        help="Frozen prior reference inputs (recreate with bench_online_pgac.py if necessary)")
    parser.add_argument("--report-dir", type=Path, help="Optional location for a portable results file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    runs = [Run(root, c, material_builder=lambda d, c: make_materials(d, c, args.previous)) for c in PROTOCOL["objects"]]
    freeze_inputs(root, runs)
    phases = ("prepare", "train", "freeze", "evaluate", "summary") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "prepare":
            for run in runs:
                run.prepare()
                baseline_identity(run)
        elif phase == "train":
            for run in runs:
                baseline_identity(run)
                for design in PROTOCOL["designs"]:
                    run.train(design)
        elif phase == "freeze":
            frozen = select(runs, root, args.previous)
            chronology = root / "preblind.json"
            if not chronology.exists():
                if any("evaluate" in r.ids for r in runs):
                    raise ValueError("Already evaluated; no prospective freeze possible")
                freeze(chronology, dict(utc=datetime.now(timezone.utc).isoformat(),
                     inputs_sha256=file_hash(root / "frozen-inputs.json"),
                     candidates_sha256=file_hash(root / "frozen-candidates.json")))
            print({k: r["design"] for k, r in frozen.items()})
        elif phase in ("evaluate", "summary"):
            chronology = read(root / "preblind.json")
            if (chronology["inputs_sha256"] != file_hash(root / "frozen-inputs.json")
                or chronology["candidates_sha256"] != file_hash(root / "frozen-candidates.json")):
                raise ValueError("Pre-blind record differs")
            frozen = read(root / "frozen-candidates.json")
            if phase == "evaluate":
                for run in runs:
                    evaluate(run, frozen)
            else:
                results = dict(protocol_sha256=file_hash(TRACKING_ASSETS / "protocol.json"), freeze=chronology,
                    cases={r.case: summarize(r, frozen, args.previous) for r in runs})
                atomic_json(root / "results.json", results)
                if args.report_dir:
                    atomic_json(args.report_dir / "results.json", results)


if __name__ == "__main__":
    main()
