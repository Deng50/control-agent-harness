"""Replay the selected standalone plant in the invoking Harness shell sandbox.

Use the project Python with -I -B. Each input row is held for one step.
No MATLAB connection or model provider is needed.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types

import numpy as np


def finite_array(value, ndim, field):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{field} requires a finite real array with {ndim} dimensions")
    return array.astype(np.float64)


def replay(source, request, expected_sha256=None):
    dt = request["sampleTime"]
    tolerance = request["absTolerance"]
    for key, value in (("sampleTime", dt), ("absTolerance", tolerance)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    parameters = request["parameters"]
    if not isinstance(parameters, dict) or not request["trajectories"]:
        raise ValueError("Supply explicit parameters and at least one trajectory")

    # Execute the same bytes that identify the upload, including paths without a .py suffix.
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Source bytes differ from the selected upload")
    plant = types.ModuleType("uploaded_plant")
    plant.__file__ = str(source)
    sys.modules[plant.__name__] = plant
    exec(compile(payload, str(source), "exec"), plant.__dict__)
    trajectories = []
    for series in request["trajectories"]:
        initial = finite_array(series["initialCondition"], 1, "initialCondition")
        controls = finite_array(series["controls"], 2, "controls")
        disturbances = finite_array(series["disturbances"], 2, "disturbances")
        if not initial.size or not len(controls) or len(controls) != len(disturbances):
            raise ValueError("An initial state and equally many control/disturbance rows are required")
        if controls.shape[1] + disturbances.shape[1] == 0:
            raise ValueError("A trajectory must drive at least one input")
        inputs = np.column_stack((controls, disturbances))
        states = [initial.tolist()]
        state = initial
        for u, d in zip(controls, disturbances):
            state = finite_array(plant.step(state.copy(), u.copy(), d.copy(), dt,
                                            copy.deepcopy(parameters)), 1, "step return")
            if state.shape != initial.shape:
                raise ValueError("step must return the complete next state in its original order")
            states.append(state.tolist())
        trajectories.append({"initialCondition": initial.tolist(), "inputs": inputs.tolist(), "states": states})
    return {"source_sha256": digest, "parameters": parameters,
            "sampleTime": dt, "absTolerance": tolerance, "trajectories": trajectories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expect-sha256", help="Verify the selected source bytes before executing step")
    args = parser.parse_args()
    request = json.load(sys.stdin) if str(args.request) == "-" else json.loads(args.request.read_text(encoding="utf-8-sig"))
    result = replay(args.source, request, args.expect_sha256)
    # Publish only a complete JSON file, and preserve any existing output.
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        temp_path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        os.link(temp_path, args.output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(json.dumps({"path": str(args.output.resolve()), "source_sha256": result["source_sha256"],
                      "samples": sum(len(t["states"]) for t in result["trajectories"])}))


if __name__ == "__main__":
    main()
