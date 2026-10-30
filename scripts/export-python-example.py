"""Export explicit non-polynomial benchmark conditions; never execute its source."""
import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from control_harness.artifacts import atomic_json
from control_harness.python_model import python_spec

root = Path(__file__).resolve().parents[1] / "examples/python-tensor"
spec = python_spec((root / "model.py").read_bytes())
reference = copy.deepcopy(spec)
reference["parameters"] = {"gains": [.24, .18], "bias": [.01, -.008, .004]}
rng = np.random.default_rng(54)


def scenarios(prefix, count, steps):
    return [{"name": f"{prefix}-{index}", "initial_state": rng.uniform(-.4,.4,3).tolist(),
             "reference": rng.uniform(-.25,.25,2).tolist(), "disturbance": rng.uniform(-.15,.15,1).tolist(),
             "steps": steps} for index in range(count)]


workflow = {"schema_version": 1, "name": "Reviewed standalone PyTorch benchmark; explicit simulation only", "steps": [
    {"id":"nominal","operation":"register_python_model","inputs":{"model_path":"model.py"},"config":{"source":{"kind":"user_nominal","description":"Reviewed standalone non-polynomial mathematical model"}}},
    {"id":"reference","operation":"register_python_model","inputs":{"model_path":"model.py"},"config":{"model_spec":reference,"source":{"kind":"reference_simulator","description":"Explicit separate benchmark parameters; not measured plant truth"}}},
    {"id":"data","operation":"simulate_transitions","inputs":{"nominal_model":{"stage":"nominal"},"reference_model":{"stage":"reference"}},"config":{"trajectories":12,"steps":16,"seed":54}},
    {"id":"corrected","operation":"train_residual","inputs":{"nominal_model":{"stage":"nominal"},"dataset":{"stage":"data"}},"config":{"epochs":100,"hidden_dims":[12,12],"learning_rate":.008,"batch_size":32,"patience":30,"seed":55}},
    {"id":"controller","operation":"design_controller","inputs":{"model":{"stage":"corrected"},"evaluation_model":{"stage":"reference"}},"config":{"parameters":{"horizon":3,"q":1,"r":.01,"s":.005},"budget":8,"seed":56,"tuning_scenarios":scenarios('tune',1,6),"verification_scenarios":scenarios('verify',1,7),"test_scenarios":scenarios('test',1,8)}},
    {"id":"expert","operation":"collect_expert","inputs":{"controller":{"stage":"controller"},"evaluation_model":{"stage":"reference"}},"config":{"scenarios":scenarios('expert',6,6),"seed":57}},
    {"id":"imitation","operation":"train_imitation","inputs":{"dataset":{"stage":"expert"}},"config":{"architecture":"mlp","epochs":50,"hidden_dims":[16,12],"batch_size":16,"learning_rate":.005,"seed":58}},
    {"id":"optimized","operation":"optimize_policy","inputs":{"policy":{"stage":"imitation"},"model":{"stage":"corrected"}},"config":{"epochs":2,"horizon":4,"learning_rate":.0002,"seed":59}},
]}
atomic_json(root / "workflow.json", workflow)
print(root / "workflow.json")
