"""Render slide figures from the frozen spring benchmark's exported CSV tables.

Run summarize_spring.py first. No model training, data smoothing, row exclusion,
interpolation, or controller selection occurs in this plotting program.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "reference": "#343A40", "nominal": "#9A9A9A", "corrected": "#2166AC",
    "baseline": "#999999", "candidate": "#2166AC", "chosen": "#2166AC",
    "expert": "#234C73", "imitation": "#D18C3C", "optimized": "#269589",
}
LABELS = {
    "reference": "Reference object", "nominal": "Nominal model", "corrected": "Corrected model",
    "baseline": "Initial MPC", "candidate": "GA candidate", "chosen": "Selected MPC",
    "expert": "MPC expert", "imitation": "Imitation policy", "optimized": "Optimized policy",
}


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def subset(rows: list[dict], **matches) -> list[dict]:
    return [row for row in rows if all(row.get(key) == value for key, value in matches.items())]


def unique(rows: list[dict], key: str) -> list[str]:
    return list(dict.fromkeys(row[key] for row in rows))


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 18, "axes.titlesize": 19, "axes.labelsize": 17,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .8, "lines.linewidth": 2.6,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "text.usetex": False,
    })


def canvas(title: str, rows: int = 1, cols: int = 1):
    fig, axes = plt.subplots(rows, cols, figsize=(13.333333, 7.5), squeeze=False)
    fig.subplots_adjust(left=.12, right=.965, bottom=.13, top=.77,
                        hspace=.55 if rows == 3 else .45, wspace=.42)
    fig.suptitle(title, x=.12, y=.965, ha="left", va="top", fontsize=22, fontweight="bold")
    fig.text(.12, .025, "Declared reference simulation  |  Single seed  |  Sample time 0.2 s",
             ha="left", va="bottom", fontsize=11, color="#666666")
    for index, ax in enumerate(axes.flat):
        ax.grid(axis="y", color="#E8E8E8", linewidth=.7)
        ax.set_axisbelow(True)
        if rows * cols > 1:
            ax.annotate(chr(97 + index), xy=(0, 1), xycoords="axes fraction",
                        xytext=(-76 if rows == 3 else -42, 9), textcoords="offset points",
                        va="bottom", ha="left", fontweight="bold", fontsize=16)
    return fig, axes


def legend(fig, ax, columns: int = 3):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.54, .90),
                   ncol=columns, handlelength=2.2, columnspacing=1.5)


def line(ax, x, y, method, **kwargs):
    return ax.plot(x, y, color=COLORS[method], label=LABELS[method],
                   linestyle="--" if method == "reference" else "-", **kwargs)


def save(fig, key, title, sources, caption, output, qa, index, audit):
    # Measure the final rendered axes rectangles before exporting any figure.
    fig.canvas.draw()
    audit(fig, json_out=qa / f"{key}.alignment.json",
          overlay_svg=qa / f"{key}.alignment.svg",
          tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(output / f"{key}.png", dpi=300)
    fig.savefig(output / f"{key}.svg")
    fig.savefig(output / f"{key}.pdf")
    index.append({"id": key, "title": title,
                  "files": {suffix: f"figures/{key}.{suffix}" for suffix in ("png", "svg", "pdf")},
                  "source_data": [f"source-data/{name}" for name in sources], "caption": caption,
                  "figure_contract": {
                      "archetype": "quantitative grid" if len(fig.axes) > 1 else "quantitative single panel",
                      "question": title, "evidence": caption,
                      "simulation": True, "measured_data": False, "single_seed": True,
                      "uncertainty": "No repeated-run uncertainty or significance inference; all declared rows retained.",
                      "dimensions_inches": [13.333333, 7.5], "png_dpi": 300,
                      "editable_vector_text": True,
                      "alignment_audit": f"qa/{key}.alignment.json",
                  }})
    plt.close(fig)


def response_figure(rows, methods, title, key, source, output, qa, index, audit,
                    include_velocity=False):
    nrows = 3 if include_velocity else 2
    fig, axes = canvas(title, nrows)
    axes = axes[:, 0]
    for method in methods:
        values = sorted(subset(rows, method=method), key=lambda row: int(row["step"]))
        if not values:
            raise ValueError(f"No trajectory for method {method}")
        t = np.r_[float(values[0]["time_s"]), numeric(values, "next_time_s")]
        q = np.r_[float(values[0]["q_m"]), numeric(values, "next_q_m")]
        line(axes[0], t, q, method)
        if include_velocity:
            v = np.r_[float(values[0]["v_m_per_s"]), numeric(values, "next_v_m_per_s")]
            line(axes[1], t, v, method)
        # Extend the last held action to the endpoint without changing its value.
        tu = np.r_[numeric(values, "time_s"), float(values[-1]["next_time_s"])]
        force = np.r_[numeric(values, "u_N"), float(values[-1]["u_N"])]
        axes[-1].step(tu, force, where="post", color=COLORS[method], label=LABELS[method])
    first = sorted(subset(rows, method=methods[0]), key=lambda row: int(row["step"]))
    target = float(first[0]["reference_m"])
    axes[0].axhline(target, color="#555555", linestyle="--", linewidth=1.3, label="Target")
    axes[0].set_ylabel("Displacement (m)")
    if include_velocity:
        axes[1].set_ylabel("Velocity (m/s)")
    axes[-1].set_ylabel("Force (N)")
    axes[-1].set_xlabel("Time (s)")
    axes[-1].axhline(4., color="#B8B8B8", linewidth=1., linestyle=":")
    axes[-1].axhline(-4., color="#B8B8B8", linewidth=1., linestyle=":")
    axes[-1].set_ylim(-4.6, 4.6)
    for ax in axes:
        ax.set_xlim(0., float(first[-1]["next_time_s"]))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    legend(fig, axes[0], len(methods) + 1)
    scenario = first[0]["scenario"]
    caption = (f"Frozen-method comparison on {scenario}; target={target:g} m and constant disturbance="
               f"{float(first[0]['d_N']):g} N. State points include the initial condition at t=0 and "
               "next-state outputs at (k+1)*dt. Force is held on [k*dt,(k+1)*dt). Dotted force lines are declared +/-4 N bounds.")
    save(fig, key, title, [source], caption, output, qa, index, audit)


def write_index(presentation: Path, index: list[dict], model_only: bool) -> dict:
    result = {"case": "spring-damper", "scope": "model_only" if model_only else "complete_three_role_case",
              "figure_count": len(index), "figures": index,
              "integrity": "All figures use complete frozen result tables. No smoothing, simulated observations added by the plotting program, subsampling, confidence intervals, or significance claims.",
              "source_preflight_warning_review": {
                  "EXPORT-RASTER": "PNG is the requested slide raster; SVG/PDF supply vectors. No journal TIFF submission is being prepared.",
                  "RASTER-DPI": "A 300-dpi, 4000-by-2250 pixel 16:9 PNG is the intended PPT export; the editable SVG/PDF are also retained.",
                  "FINAL-WIDTH": "Slide-sized 16:9 export is intentional; this is a PPT asset bundle, not a journal-width figure.",
                  "UNCERTAINTY-ENCODING": "Only one declared run/seed is available. Per-scenario points and their recorded timing summaries are displayed without repeated-run uncertainty or significance inference.",
              },
              "qa_status": "Render-time panel alignment complete; final PDF collision/text audit and visual review are separate required checks."}
    filename = "figures-index-model.json" if model_only else "figures-index.json"
    (presentation / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def render(workspace: Path, skill_scripts: Path, model_only: bool = False) -> dict:
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    audit = require_matplotlib_panel_alignment
    presentation = workspace / "presentation"
    data = presentation / "source-data"
    output, qa = presentation / "figures", presentation / "qa"
    output.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    summary_name = "metrics-summary-model.json" if model_only else "metrics-summary.json"
    summary = json.loads((presentation / summary_name).read_text(encoding="utf-8"))
    if summary.get("case") != "spring-damper" or not summary.get("all_artifact_files_and_dependencies_verified"):
        raise ValueError("Use a verified spring result summary")
    style()
    index = []

    free = load_rows(data / "free-decay-trajectories.csv")
    title = "Model correction: independent free decay"
    fig, axes = canvas(title, 2)
    for method in ("nominal", "corrected", "reference"):
        values = sorted(subset(free, method=method), key=lambda row: int(row["step"]))
        line(axes[0, 0], numeric(values, "time_s"), numeric(values, "q_m"), method)
        line(axes[1, 0], numeric(values, "time_s"), numeric(values, "v_m_per_s"), method)
    axes[0, 0].set_ylabel("Displacement (m)")
    axes[1, 0].set_ylabel("Velocity (m/s)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[0, 0].tick_params(labelbottom=False)
    for ax in axes.flat:
        ax.set_xlim(0., max(numeric(free, "time_s")))
    legend(fig, axes[0, 0])
    save(fig, "01-model-free-decay", title, ["free-decay-trajectories.csv", "free-decay-rmse.csv"],
         "All three models start at q=0.6 m, v=0 m/s and evolve their own predicted states for 80 steps under u=d=0. This is independent autonomous multi-step prediction, not one-step teacher forcing.",
         output, qa, index, audit)

    # Per-state RMSE panels keep displacement and velocity units separate.
    errors = load_rows(data / "residual-test-rmse.csv")
    title = "Held-out model error after residual learning"
    fig, axes = canvas(title, 1, 2)
    for ax, state in zip(axes.flat, unique(errors, "state")):
        values = subset(errors, state=state)
        ordered = [next(row for row in values if row["method"] == method) for method in ("nominal", "corrected")]
        ax.bar([0, 1], numeric(ordered, "rmse"), width=.55,
               color=[COLORS["nominal"], COLORS["corrected"]])
        ax.set_xticks([0, 1], ["Nominal", "Corrected"])
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(numeric(ordered, "rmse")) * .35, top=max(numeric(ordered, "rmse")) * 3)
        ax.set_ylabel(f"{state.capitalize()} RMSE ({ordered[0]['unit']})")
    save(fig, "02-model-heldout-rmse", title, ["residual-test-rmse.csv"],
         f"Next-state one-step RMSE over all {summary['transition_split']['test']['samples']} held-out transitions; partitions are independent trajectories. Separate axes retain m and m/s units. Logarithmic axes display small residual errors; no repeated-seed uncertainty is implied.",
         output, qa, index, audit)

    training = load_rows(data / "residual-training.csv")
    title = "Residual network learning curve"
    fig, axes = canvas(title)
    ax = axes[0, 0]
    for field, label, color in (("training_loss", "Training", "#2166AC"), ("validation_loss", "Validation", "#D18C3C")):
        value = numeric(training, field)
        if np.any(value <= 0):
            raise ValueError("Logarithmic training loss requires positive values")
        ax.plot(numeric(training, "epoch"), value, color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized loss")
    legend(fig, ax, 2)
    save(fig, "03-residual-learning", title, ["residual-training.csv"],
         "Recorded losses for every trained epoch. Training includes the declared physical constraint penalty, whereas validation is normalized prediction error; the two objectives are related but not numerically identical.",
         output, qa, index, audit)

    if model_only:
        return write_index(presentation, index, model_only=True)

    trials = load_rows(data / "controller-ga-trials.csv")
    title = "Genetic search over MPC weights"
    fig, axes = canvas(title)
    ax = axes[0, 0]
    scores = np.asarray([float(row["score"]) if row["score"] else np.nan for row in trials])
    feasible = np.asarray([row["hard_constraints_passed"].lower() == "true" for row in trials])
    trial = numeric(trials, "trial")
    ax.plot(trial, scores, "o", color="#9A9A9A", markersize=7, label="Candidate score")
    best = np.minimum.accumulate(np.where(feasible & np.isfinite(scores), scores, np.inf))
    ax.step(trial, np.where(np.isfinite(best), best, np.nan), where="post", color="#2166AC", label="Best feasible so far")
    ax.set_xlabel("Candidate evaluation")
    ax.set_ylabel("Tuning score (lower is better)")
    ax.set_xticks(trial.astype(int))
    legend(fig, ax, 2)
    save(fig, "04-mpc-ga-search", title, ["controller-ga-trials.csv"],
         "Every requested genetic candidate is retained. Scores use tuning conditions only; the best candidate must still pass independent verification before adoption. Score combines normalized tracking RMSE and 0.01 times normalized control effort; it is not an error in metres.",
         output, qa, index, audit)

    for source, methods, prefix, heading, velocity in (
        ("controller-verification-trajectories.csv", ["baseline", "candidate"], "05-mpc-verification", "MPC weights: independent verification", False),
        ("controller-final-test-trajectories.csv", ["baseline", "chosen"], "06-mpc-final-test", "MPC tracking on the reserved final test", True),
    ):
        rows = load_rows(data / source)
        scenarios = unique(rows, "scenario")
        for count, scenario in enumerate(scenarios, 1):
            key = prefix if len(scenarios) == 1 else f"{prefix}-{count:02d}"
            response_figure(subset(rows, scenario=scenario), methods, heading, key, source,
                            output, qa, index, audit, include_velocity=velocity)

    policy = load_rows(data / "policy-heldout-trajectories.csv")
    for count, scenario in enumerate(unique(policy, "scenario"), 1):
        response_figure(subset(policy, scenario=scenario), ["expert", "imitation", "optimized"],
                        f"Controller compression: {scenario}", f"07-policy-heldout-{count:02d}",
                        "policy-heldout-trajectories.csv", output, qa, index, audit)

    metrics = load_rows(data / "policy-heldout-metrics.csv")
    scenarios = unique(metrics, "scenario")
    title = "Accuracy and computation on the same held-out cases"
    fig, axes = canvas(title, 1, 2)
    for method in ("expert", "imitation", "optimized"):
        values = [next(row for row in metrics if row["scenario"] == scenario and row["method"] == method) for scenario in scenarios]
        line(axes[0, 0], np.arange(len(scenarios)), numeric(values, "rmse"), method, marker="o", markersize=7)
        line(axes[0, 1], np.arange(len(scenarios)), numeric(values, "mean_control_seconds") * 1000,
             method, marker="o", markersize=7)
    for ax in axes.flat:
        ax.set_xticks(np.arange(len(scenarios)), [f"Case {number + 1}" for number in range(len(scenarios))])
        ax.set_xlim(-.25, len(scenarios) - .75)
        ax.set_xlabel("Held-out scenario")
    axes[0, 0].set_ylabel("Tracking RMSE (m)")
    axes[0, 1].set_ylabel("Mean control-call time (ms)")
    axes[0, 1].set_yscale("log")
    sample_period_ms = float(summary["sample_time_s"]) * 1000
    period_line = axes[0, 1].axhline(
        sample_period_ms, color="#555555", linewidth=1.3, linestyle="--",
        label=f"Sample period ({sample_period_ms:g} ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend([*handles, period_line], [*labels, period_line.get_label()],
               loc="upper center", bbox_to_anchor=(.54, .90), ncol=4,
               handlelength=2.2, columnspacing=1.5)
    save(fig, "08-policy-accuracy-and-time", title, ["policy-heldout-metrics.csv", "policy-heldout-timing.csv"],
         "Each connected point uses the same independent test scenario for all methods; left is tracking RMSE, right is this run's mean CPU control-call wall time on a log axis. Case order: "
         + ", ".join(f"{i+1}={name}" for i, name in enumerate(scenarios))
         + f". The dashed {sample_period_ms:g} ms sample-period line is a time-scale reference; a mean call time below the sample period does not establish a hard real-time guarantee."
         + " Lines connect matched scenario summaries, not a continuous-time trend. Single-machine timing is not a hardware-independent speedup claim.",
         output, qa, index, audit)

    imitation = load_rows(data / "imitation-training.csv")
    optimization = load_rows(data / "policy-optimization.csv")
    title = "Learning the expert, then improving physical rollouts"
    fig, axes = canvas(title, 1, 2)
    for key, label, color in (("training_loss", "Training", "#2166AC"), ("validation_loss", "Validation", "#D18C3C")):
        values = numeric(imitation, key)
        if np.any(values <= 0):
            raise ValueError("Logarithmic imitation loss requires positive values")
        axes[0, 0].plot(numeric(imitation, "epoch"), values, label=label, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Imitation epoch")
    axes[0, 0].set_ylabel("Normalized action loss")
    values = np.asarray([float(row["validation_score"]) if row["validation_score"] else np.nan for row in optimization])
    axes[0, 1].plot(numeric(optimization, "epoch"), values, color=COLORS["optimized"], label="Physical optimization")
    axes[0, 1].set_xlabel("Physical optimization epoch")
    axes[0, 1].set_ylabel("Validation selection score")
    legend(fig, axes[0, 0], 2)
    save(fig, "09-policy-learning", title, ["imitation-training.csv", "policy-optimization.csv"],
         "All logged imitation epochs and physical policy-optimization epochs are shown. The imitation training loss is the logged final minibatch of each epoch, whereas validation uses the held-out validation partition. Physical rollout optimization selects a shared checkpoint using validation conditions; testing does not select weights.",
         output, qa, index, audit)

    return write_index(presentation, index, model_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--model-only", action="store_true",
                        help="Render only model-stage figures from metrics-summary-model.json")
    parser.add_argument("--skill-scripts", type=Path,
                        default=Path.home() / ".agents/skills/nature-figure/scripts")
    args = parser.parse_args()
    result = render(args.workspace.resolve(), args.skill_scripts.resolve(), model_only=args.model_only)
    print(json.dumps({"figure_count": result["figure_count"], "formats": ["PNG 300 dpi", "SVG", "PDF"]}))


if __name__ == "__main__":
    main()
