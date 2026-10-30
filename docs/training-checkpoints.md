# Numerical training checkpoints

`train_residual`, `train_imitation`, and `optimize_policy` accept
`config.resume_checkpoint`. The operation's inputs must identify the same model,
dataset, and (for policy optimization) starting policy artifacts as the original
run. `config.epochs` is the **total target epoch count**, not the number of extra
epochs. It may increase; it cannot be below the checkpoint's completed epoch.

Example configuration change after an interruption:

```json
{
  "epochs": 150,
  "resume_checkpoint": "failed/imitation_policy-<attempt-id>/latest.pt"
}
```

Keep the remaining configuration from the original request. The path can be
absolute or relative to the workspace. It must resolve to
`artifacts/<id>/latest.pt` or `failed/<id>/latest.pt` inside that workspace.
The alternative `failures/<id>/latest.pt` spelling is also accepted for retained
failure packages. The worker currently writes failures under `failed/`.
Active `.staging` files and external checkpoint paths are rejected.

Every completed epoch atomically replaces `latest.pt`, then writes its integrity
sidecar `latest.checkpoint.json`. An epoch-zero checkpoint is saved before the
training loop, including before physical optimization's baseline validation.
The checkpoint contains current model weights, Adam state, best validation
weights and score, completed epoch and optimizer-update counts, early-stopping
counter, the complete training log, Python/NumPy/Torch random states and the
separate shuffle/sampling generator states.

Cancellation or failure during an epoch leaves the last completed epoch as the
recovery point. Any uncommitted minibatch/condition updates are intentionally
replayed. Failure of a progress callback after checkpoint commit preserves that
completed epoch. The stage moves to `failed/`, preserving its failure record and
checkpoint. Training progress events include `checkpoint` pointing to this
post-failure path and `optimizer_updates`. On normal publication the checkpoint
is under the returned artifact directory instead.

Recovery verifies the sidecar digest and checkpoint format, resolves and hashes
all exact artifact dependencies, and checks the effective training configuration,
interface, algorithm version and numerical runtime versions. Residual signatures
include weight decay, patience, physical constraints and acceptance threshold;
policy signatures include architecture/forecast semantics and optimization loss
weights/horizon, inherited expert action constraints and projection semantics.
Only `epochs`, `resume_checkpoint`, and `resume_checkpoint_sha256` are excluded
from the signature. An early-stopped run stays early-stopped on resume; changing its
patience is a new training experiment and is rejected as continuation.

Old incomplete checkpoint formats, missing/corrupt sidecars, changed dependency
files, stale dependencies, incompatible settings, and a different numerical
runtime are rejected. Loading always uses `torch.load(weights_only=True)`.
Unpublished failed stages do not become accepted artifacts just because their
checkpoint can be resumed. Likewise a completed but rejected training output can
supply a checkpoint for a larger epoch budget; the resumed output still undergoes
the normal final acceptance evaluation. Published checkpoint hashes are checked
against the immutable artifact manifest.

Workflows can supply `config.resume_checkpoint_sha256` from resource preflight.
All three workers compare that digest with the loaded bytes before restoring any
training state. Deserialization uses the same byte buffer that was hashed, so a
later replacement of the checkpoint path cannot switch the validated payload.
This hash pins the recovery input; it does not alter the optimization settings.

The checkpoint/sidecar pair is fail-closed: if the process dies between replacing
the checkpoint and updating its sidecar, a digest mismatch is reported rather
than loading an unverified state. It does not implement a multi-generation
checkpoint backup or mid-minibatch continuation.

Reports retain all training log rows across attempts and expose `trained_epochs`,
`optimizer_updates`, `resumed_from_epoch`, and `training_signature`. The published
model/policy uses the best validation weights; the continuation checkpoint uses
the latest optimizer weights. Test scenarios never select checkpoints.

## Verification

Run from the repository using the bundled project virtual environment:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_training_resume.py -v
```

The tests perform real CPU training with a real MPC expert and simulated data.
They compare uninterrupted and interrupted/resumed runs tensor-for-tensor,
including Adam states, best weights, random states, counters, and complete logs.
They also cover mid-epoch cancellation, failed-stage recovery, total-budget
extension, early stopping, and invalid configuration/dependency/checkpoint
rejection. They do not mock successful training or solver output.

## OS-confirmed hard-exit recovery

For a worker that exits without running normal cleanup, use
`recover_interrupted_stage` with the exact `.staging/<stage_id>` directory name.
The operation takes the workspace kernel mutex, requires the OS to positively
confirm the original PID exited, verifies stage intent, config, full dependency
closure, runtime, checkpoint contract and sidecar hashes, then moves the stage
to `failed/<stage_id>` without overwriting another directory. Live or unknown
owners, aliases, publication metadata and manifest-referenced stages are refused.
No age threshold is used.

The returned `checkpoint` and `checkpoint_sha256` can be copied to the original
matching request as `resume_checkpoint` and `resume_checkpoint_sha256`. This
archival operation does not start training. Web exposes both steps separately.
`tests/test_interrupted_stage_recovery.py` executes an actual `os._exit`, then
checks numerical continuation against uninterrupted CPU training.
