# Numerical training checkpoints

`train_residual`, `train_supervised`, `train_imitation`, `optimize_policy`
and the PINN and restricted PGAC trainers accept
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
absolute or relative to the workspace. Use the checkpoint path and SHA-256
returned by the worker. Supported locations inside the workspace are
`artifacts/<id>/`, `failed/<id>/` and the historical `failures/<id>/`.

每个完整轮次写入独立的 `checkpoint-<32位十六进制代号>.pt` 和同名 `.json`。
两者刷新到磁盘后，训练器读取实际字节，核对哈希、契约、计数和张量，再原子替换
`checkpoint-head.json`。指针保留当前和前一个已提交代号。未提交的文件可能留在失败
现场供排查，恢复器只接受指针列出的代号。训练正常发布或失败归档时生成一份
`latest.pt` 和 `latest.checkpoint.json` 兼容快照。历史版本写成的旧文件对仍按原格式只读校验。
若当前代号损坏，失败现场与孤儿阶段恢复器可选择同一指针记录的上一代，返回
`checkpoint_selection=previous`、失败原因、实际轮次和摘要；此时保留原始指针与损坏字节。
已经指定路径和 SHA-256 的续训请求始终只接受指定字节。

An epoch-zero checkpoint is saved before the
training loop, including before physical optimization's baseline validation.
The checkpoint contains current model weights, Adam state, best validation
weights and score, completed epoch and optimizer-update counts, early-stopping
counter, the complete training log, Python/NumPy/Torch random states and the
separate shuffle/sampling generator states.

Cancellation or failure during an epoch leaves the last completed epoch as the
recovery point. Any uncommitted minibatch/condition updates are intentionally
replayed. Failure of a progress callback after checkpoint commit preserves that
completed epoch. Progress reports the generation digest, epoch, optimizer updates,
target epochs and prospective failed-stage path with status `committed_in_staging`.
The final error reports the verified location and actual archive state. When a
rename fails, `stage_failure.retained_path` identifies the retained staging directory
and `archive_errors` describes the failure. Use the guarded hard-exit recovery
operation after the worker exits. A stage interrupted during publication requires
publication review.

Recovery verifies the sidecar digest and checkpoint format, resolves and hashes
all exact artifact dependencies, and checks the effective training configuration,
interface, algorithm version and numerical runtime versions. Residual signatures
include weight decay, patience, physical constraints and acceptance threshold;
policy signatures include architecture/forecast semantics and optimization loss
weights/horizon, inherited expert action constraints and projection semantics.
Only `epochs`, `resume_checkpoint`, `resume_checkpoint_sha256`, and `plan_refs` are excluded
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
The trainers compare that digest with the loaded bytes before restoring any
training state. Deserialization uses the same byte buffer that was hashed, so a
later replacement of the checkpoint path cannot switch the validated payload.
This hash pins the recovery input; it does not alter the optimization settings.

指针替换遇到 Windows 共享或锁冲突（32、33）最多尝试三次，间隔 25 和 75 毫秒。
拒绝访问（5）立即报告 `CHECKPOINT_ACCESS_DENIED`。持续锁冲突返回
`CHECKPOINT_FILE_BUSY`，其他写入错误返回 `CHECKPOINT_IO_ERROR`。CLI 错误事件包含
`io_failure` 中的操作、路径和系统错误码。第一次提交就失败时没有可继续的轮次。

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
