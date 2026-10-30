# Measurement-driven KF/EKF control

`control_harness.estimation` adds an observer to the existing constrained MPC. The controller's optimization state is the posterior estimate, not the simulator's true state. It does not tune MPC weights or replace genetic controller tuning.

## Runtime interfaces and timing

```python
from control_harness.estimation import EstimatedMPCPolicy, StateEstimator
from control_harness.control import MPCPolicy

policy = EstimatedMPCPolicy(MPCPolicy(model, controller_parameters), {
    "initial_estimate": [0.0, 0.0],
    "measurement_output_indices": [0],
    "Q": [[1e-5, 0], [0, 1e-5]],
    "R": [[1e-4]],
    "P0": [[0.5, 0], [0, 0.5]],
})
u, diagnostics = policy.act_measurement(y_k, reference_k, disturbance_k)
```

The measured vector follows `measurement_output_indices` in the declared output order, with its declared units. It is `y(k)=h(x(k))`: first correct the prior estimate using the current measurement, then calculate `u(k)` from `x_hat(k|k)`, then predict `x_hat(k+1|k)` and covariance using that command. The convenience policy assumes commanded and applied control agree. For actuator overrides or a separately managed sensor loop, use `StateEstimator.correct(y_k)` and `predict(actual_applied_control, disturbance_k)` explicitly, alongside the controller.

The production `act_measurement` method has no true-state argument. `EstimatorSimulationAdapter` is explicitly the simulation sensor boundary: it takes `plant.output(x)`, selects only configured measurements, and adds seeded Gaussian sensor noise. Only evaluation reports see the true state. The model and controller share the exact same prediction model, including a residual-corrected model if supplied.

## Numerical contracts

- `method="auto"` chooses KF for an affine declarative model without nonlinear terms, and EKF otherwise. An explicit KF request for nonlinear or residual-corrected dynamics is rejected.
- EKF Jacobians use Torch automatic differentiation of the executable transition and output functions.
- Q/P0 must be finite symmetric positive semidefinite matrices. R must be positive definite. Joseph-form posterior covariance avoids the cancellation of a subtractive update; innovation solves use `numpy.linalg.solve`, not an explicit inverse. Only negative eigenvalues within floating-point roundoff are corrected.
- State, control, measurement and disturbance dimensions are checked. At least one unique declared output must be measured.
- Linear models must pass a state/output-scale-normalized observability rank check. Validation also builds the time-varying observability matrix along the executed estimated trajectory. Nonlinear rank is a local numerical result, not a global nonlinear observability proof.
- `initial_estimate` is required and never initialized from the simulator's true state. Missing Q/R/P0 use range-scaled engineering assumptions recorded in provenance, not claimed measured noise statistics.

## Validation and artifacts

```python
from control_harness.estimation import validate_estimator, load_estimator_controller

artifact_id = validate_estimator(workspace, model_id, controller_id, reference_id, {
    "initial_estimate": [0.0, 0.0],
    "verification_scenarios": [verification_scenario],
    "test_scenarios": [independent_test_scenario],
    "sensor_noise_covariance": [[1e-4]],
    "seed": 42,
    "max_normalized_estimation_rmse": 0.1,
})
deployed = load_estimator_controller(workspace, artifact_id)
```

Scenario fields are the existing `initial_state`, `reference`, `disturbance`, `steps`, optional `name`. Verification and test scenarios cannot be identical after removing their names, and each must contain at least `nx` transitions. Parameters are fixed before either group; test scenarios do not select Q/R/P0. Each scenario receives a deterministic distinct seed. Sensor covariance can differ from the assumed R; Q describes estimator process uncertainty, and this validation does not inject process noise into the plant.

The `estimator` artifact binds the exact model/controller/reference versions, contains `estimator.json`, `config.json`, and `report.json`, and can be imported through the existing artifact handoff protocol. Reports include measured/estimated/true trajectories, open-loop estimator baseline, posterior covariance, innovations, local observability, real MPC trajectories, hard constraints and explicit error gates. A failed gate publishes a rejected, non-loadable artifact without replacing an existing verified observer. The default normalized estimation-error gate is 0.25 of declared state ranges; set an explicit engineering threshold for an application.

This is simulator validation of declared discrete equations. It does not certify physical sensor noise or run an original Simulink diagram. Validation is bounded and restartable; no resumable training checkpoint is claimed.

## Dispatcher integration

Register `validate_estimator` and dispatch `inputs.model`, `inputs.controller`, `inputs.evaluation_model`, `config` and `context` to the function above. Its result is an artifact ID compatible with the standard result envelope. The module does not require modifications to `control.py` or `modeling.py`.

## Tests

```powershell
$env:PYTHONPATH = 'D:\projects\control-agent-harness\src'
.venv\Scripts\python.exe -m unittest discover -s tests -p test_estimation.py -v
```

Tests compare Joseph updates to a Gaussian conditioning calculation, reject invalid covariance and unobservable systems, prove MPC's action equals an independent solve at the posterior estimate and differs from a true-state solve, exercise real linear and nonlinear closed loops with partial measurements, check reproducible sensor noise, cancellation, independent holdouts, rejection and artifact reload.
