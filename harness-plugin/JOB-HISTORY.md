# Durable control attempts

The host plugin retains execution evidence in the selected workspace's `.control-jobs/<attemptId>/`. Harness `jobs` remains the only live job registry and controller. `WorkerSlots` remains the existing bounded admission queue; history never starts work on its own.

Each attempt contains an immutable `request.json` (Session owner, role, request snapshot, host/runtime identity and recovery lineage), followed by numbered JSON events. Writers serialize events and publish a complete temporary file atomically, so readers never consume half-written JSON. The append-only events include the native job id, worker progress/result and producer/native settlement. This supports process restarts, not a guarantee against disk loss or filesystem corruption; malformed records fail visibly.

## Commands

- `/control {agent,operation,workspace,inputs?,config?}` returns `{jobId,attemptId,workspace}`. The snapshot is persisted before spawning Python. Native admission failures remain visible as failed attempts.
- `/control-status <jobId>` retains the original native-only interface.
- `/control-status {workspace,attemptId}` returns `{record,job,output}`. `job/output` are `null` when the native registry no longer owns the execution; the durable result is still in `record.result`.
- `/control-history {workspace}` returns `{workspace,attempts}` for the exact current Session owner.
- `/control-recover {workspace,attemptId,mode,checkpoint?}` explicitly creates a replacement native job. `mode` is `retry` or `resume`; the response adds `recoveredFrom`. Concurrent repeated submissions atomically claim one replacement and return its identity with `existing:true`. Recover the replacement attempt itself if it later fails. A lost host between the recovery claim and admission leaves that replacement visible as interrupted, rather than silently starting it later.

Important record fields are `attemptId`, `owner`, `agentName`, `workspace`, `request`, `nativeJobId`, `state`, `authority`, `detail`, `progress`, `result`, `outcome`, `checkpoint`, `workflowRun`, `recoveredFrom`, `recoveryMode`, `recovery`, and optional `recoveryBlocked`. Timestamps are presentation metadata only.

## Recovery semantics

Live native `running/stopping/completed/failed/killed` status always takes precedence. A persisted nonterminal attempt missing from the native registry projects `state:interrupted, authority:ownership_lost`; this means lost execution ownership, **not proof that its process stopped**. A still-existing old host PID, a PID lookup that cannot establish absence, or another hostname blocks recovery. PID reuse therefore errs toward blocking; it is never used to kill a process. No age or file timestamp decides liveness. A native forced settlement without producer cleanup acknowledgement, or a producer that could not confirm process-range cleanup, also blocks recovery. Native process containment handles teardown; Python's workspace kernel mutex and dead-owner verification fence mutation after abrupt termination.

`retry` sends the original request again, including any explicit original recovery configuration. It can create new artifact versions and is never automatic. `resume` for `train_residual/train_imitation/optimize_policy` sets `config.resume_checkpoint` to an explicitly selected or recorded checkpoint inside the same workspace. Python validates its content signature, dataset/model versions, optimizer/RNG compatibility and allowed stage location. Host history does not inspect Torch binaries or pretend the original process survived. If cancellation occurred before the first checkpoint progress event, the user must select the checkpoint explicitly; history never guesses the newest file.

`resume` for `run_workflow` sets `config.resume_run` from the recorded `workflow_run` progress/result field and preserves the original steps. This asks the Python workflow to verify and skip already completed stages. It is distinct from resuming a training checkpoint. Other operations support retry only. Completed attempts cannot be recovered through this command.

Every execution or recovery resolves the current Session sandbox policy again. Host persistence requires a writable workspace, validates physical ancestors/junctions against the policy roots, and rejects redirected history directories. Read access checks the exact persisted Session owner; copied or forked chat history does not transfer job ownership. Numerical work still uses Harness sandbox confinement and native subprocess handles.

## Verification

`node --test harness-plugin/tests/job-history.test.mjs` covers policy roots and junctions, atomic concurrent events, Session isolation, native-status precedence, conservative liveness, first-wins recovery, checkpoint scope, cancellation before admission, real native job/subprocess execution, restart after completion, concurrent recovery, and abrupt host termination during real Python training. It creates isolated workspaces under `test-output/job-history-tests` and never opens or restarts the Web service. On Windows it may need an unsandboxed test runner because the outer execution sandbox denies Node `realpath` for newly created test directories; the product path checks remain enabled.
