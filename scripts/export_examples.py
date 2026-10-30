"""Export reproducible, explicitly simulated objects and configuration templates."""
from pathlib import Path
from control_harness.artifacts import atomic_json
from control_harness.examples import example

root = Path(__file__).resolve().parents[1] / "examples"
for case in ("linear", "nonlinear"):
    config = example(case)
    for key, value in config.items():
        if isinstance(value, dict):
            atomic_json(root / case / f"{key}.json", value)
    ref = lambda stage: {"stage": stage}
    steps = [
        {"id": "nominal", "operation": "register_model", "inputs": {"model_path": "nominal.json"},
         "config": {"source": {"kind": "user_nominal", "description": "Declared demonstration nominal equations"}}},
        {"id": "reference", "operation": "register_model", "inputs": {"model_path": "reference.json"},
         "config": {"source": {"kind": "reference_simulator", "description": "Separate demonstration simulator, not measured data"}}},
        {"id": "data", "operation": "simulate_transitions", "inputs": {"nominal_model": ref("nominal"), "reference_model": ref("reference")}, "config": config["data"]},
        {"id": "corrected", "operation": "train_residual", "inputs": {"nominal_model": ref("nominal"), "dataset": ref("data")}, "config": config["residual"]},
        {"id": "controller", "operation": "design_controller", "inputs": {"model": ref("corrected"), "evaluation_model": ref("reference")}, "config": config["controller"]},
        {"id": "expert", "operation": "collect_expert", "inputs": {"controller": ref("controller"), "evaluation_model": ref("reference")}, "config": config["collection"]},
        {"id": "imitation", "operation": "train_imitation", "inputs": {"dataset": ref("expert")}, "config": config["imitation"]},
        {"id": "optimized", "operation": "optimize_policy", "inputs": {"policy": ref("imitation"), "model": ref("corrected")}, "config": config["optimization"]},
    ]
    atomic_json(root / case / "workflow.json", {"schema_version": 1, "name": f"{case} explicit simulator pipeline", "steps": steps})
print(f"Exported simulator examples to {root}")
