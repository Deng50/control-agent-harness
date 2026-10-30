# Full-state least-squares system identification

`control_harness.identification.identify_linear_model` identifies the discrete affine transition

`x(k+1) = A x(k) + B u(k) + E d(k) + bias`.

This operation estimates a dynamics model from transition data. Genetic tuning still owns controller parameters; identification does not change controller weights or horizons. It is not output-only/subspace identification and does not infer unmeasured states. The output map C and all dimensions, units, ordering and sample time remain the declared source contract.

## Data and fitting

Input is a verified `transition_dataset` artifact containing `x`, `u`, `d`, `next_x`, and integer `trajectory_id`, with the existing train/validation/test partition. The dataset must bind the exact nominal model version. The three partitions must be nonempty, disjoint, exhaustive, and cannot share a trajectory.

Only training rows determine centering, scaling and coefficients. `numpy.linalg.lstsq` solves the scaled regression using an SVD-based rank criterion. The training design matrix must have full column rank and more rows than parameters; zero-variance regressors, insufficient excitation and excessive condition numbers are rejected. Coefficients are converted back into physical units before being saved.

Validation and test data are evaluated independently after fitting. Neither chooses coefficients, feature scaling, regression rank or hyperparameters. The report includes errors by state, normalized prediction errors, nominal-model comparison, the training residual covariance, matrix rank/condition/singular values and exact trajectory groups. The covariance is a regression residual statistic, not a claim of experimentally measured process noise.

```python
from control_harness.identification import identify_linear_model

identified_id = identify_linear_model(workspace, nominal_id, dataset_id, {
    "fit_bias": True,
    "rank_relative_tolerance": 1e-10,
    "max_condition_number": 1e8,
    "max_validation_normalized_rmse": 0.05,
    "max_test_normalized_rmse": 0.05,
})
```

Normalization uses each declared state's range. The defaults above are engineering gates and should be replaced by application-specific limits. `max_spectral_radius` is an optional explicit stability gate; unstable model dynamics are not categorically impossible and are otherwise reported. If the declared source contains nonlinear terms, `allow_linear_approximation=true` is required before replacing them with an affine approximation.

## Artifact and integration

The `identified_model` artifact contains standard executable `model.json`, provenance, `config.json`, and the independent validation `report.json`. Its parents are the exact nominal model and transition dataset. It deliberately does not replace `nominal_model`, so the source data lineage remains valid. Failed numerical or error gates never produce an active usable model. Verified identified models may be handed off through the existing immutable artifact protocol and consumed by MPC, residual training or estimation after the standard model loader recognizes the kind.

Dispatcher patch: import `identify_linear_model`, add `identify_linear_model` to operation names, and dispatch `inputs.nominal_model`, `inputs.dataset`, `config`, `context`. Model-loader patch: add `identified_model` to the existing nominal/reference model kinds that read `model.json`. `load_identified_model` is also available directly without that common-loader change.

## Tests

```powershell
$env:PYTHONPATH = 'D:\projects\control-agent-harness\src'
.venv\Scripts\python.exe -m unittest discover -s tests -p test_identification.py -v
```

The tests identify A/B/E/bias from seeded trajectories, verify that changing held-out data cannot change fitted coefficients or preprocessing, reject rank-deficient and overlapping datasets, reject a poor test-set result, preserve the original nominal source, and run a real MPC closed loop using the identified model.
