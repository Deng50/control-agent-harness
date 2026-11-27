"""Read the previous benchmark's frozen training logs and original-plant comparisons."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from bench_pgac_cases import ROOT, read
from control_harness.artifacts import atomic_json, file_hash


def diagnostics(root, case):
    base = root / case
    ops = read(base / "operations.json")
    train = base / "workspace/artifacts" / ops["online"]["artifact_id"] / "report.json"
    delivery = base / "workspace/artifacts" / ops["evaluate_online"]["artifact_id"] / "report.json"
    kernel, original = read(train), read(delivery)
    settings = original["original_evaluation_config"]["evaluation"]
    weights = {term: settings[key] for term, key in (("tracking", "q"), ("action", "r"), ("increment", "s"))}
    sums = {a: Counter() for a in ("source", "family", "phase")}
    epochs = []
    for row in kernel["log"]:
        for s in row["sampled_strata"]:
            for key in sums:
                sums[key][s[key]] += s["count"]
        epochs.append(dict(epoch=row["epoch"], hold_in_J=row["post_update_training_J"],
                   validation_J=row["predictor_validation_score"], selected=row["selected_by_validation"],
                   action_probe_change=row["post_update_action_delta_max"], updates=row["optimizer_updates"]))
    phases = {}
    for part in ("validation", "test"):
        scores = original["closed_loop"][part]
        def arr(who):
            return scores[who]["quadratic"]["per_scenario"]
        groups = {}
        for im, pg in zip(arr("imitation"), arr("candidate"), strict=True):
            assert im["trajectory_id"] == pg["trajectory_id"]
            family = im["family"]
            terms = {k: weights[k] * (pg["terms"][k] - im["terms"][k]) for k in weights}
            delta = pg["score"] - im["score"]
            if abs(sum(terms.values()) - delta) > 1e-12:
                raise ValueError("Inconsistent loss decomposition")
            groups.setdefault(family, []).append(dict(delta_J=delta, **terms,
                imitation_J=im["score"], candidate_J=pg["score"],
                imitation_rmse=im["tracking_rmse_physical"][0], candidate_rmse=pg["tracking_rmse_physical"][0]))
        def reduce(rows):
            reduced = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            reduced["rmse_change_percent"] = 100 * (reduced["candidate_rmse"] / reduced["imitation_rmse"] - 1)
            reduced["J_change_percent"] = 100 * (reduced["candidate_J"] / reduced["imitation_J"] - 1)
            return reduced
        phases[part] = {family: reduce(rows) for family, rows in groups.items()}
        phases[part]["overall"] = reduce([r for rows in groups.values() for r in rows])
    return dict(old_candidate=ops["online"]["artifact_id"], log_sha256=file_hash(train),
        old_evaluation=ops["evaluate_online"]["artifact_id"], evaluation_sha256=file_hash(delivery),
        old_release=original["selected_version"],
        baseline_predictor_validation_J=kernel["baseline_predictor_validation"]["score"],
        weights=weights, epochs=epochs, sampled_starts=sums, phases=phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=ROOT / "test-output/pgac-online-benchmark-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, {c: diagnostics(args.previous, c) for c in ("spring", "motor")})


if __name__ == "__main__":
    main()
