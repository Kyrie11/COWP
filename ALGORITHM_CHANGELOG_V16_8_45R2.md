# V16.8.45R2 — RCRSO Sidecar Performance / Observability Repair

**Classification:** engineering-only performance and observability revision. The scientific method remains V16.8.45 RCRSO; no proposal/hard-verifier/gate/dataset semantics are changed.

## Motivation

V16.8.45R1 fixes schema/support fidelity, but full `build_sidecar_train_parallel4` is still expensive and silent because the launcher runs four CPU workers with stdout/stderr redirected to per-shard logs. Static inspection and synthetic profiling show that sidecar generation is dominated by repeated hard-verifier work rather than RCRSO neural forward: responder/ego/environment collision compatibility and per-edge Waymax inverse-kinematic checks are the largest repeated components.

## Semantics-preserving performance changes

1. **Scene-level verifier/cache reuse.** The V44R1 semantic caches now live across candidate/root evaluations within one immutable scene instead of being recreated as `{}` for each verifier call. Cache keys remain trajectory/control-semantic identities; no boolean predicate is skipped.
2. **Vectorized Waymax kinematic audit.** `_trajectory_waymax_kinematic_safe_np` evaluates the H transitions in one batch through the unchanged `_waymax_kinematic_transition_np`. A literal edge-by-edge implementation is retained for randomized equivalence regression.
3. **Environment event reuse.** V44 analytic completion already computes root↔environment current/shift unsafe-event indices. RCRSO feature construction consumes those exact cached indices instead of recomputing the same collision events.
4. **Partial NPZ loading.** Sidecar construction asks `COWPNpzDataset` only for fields it actually uses rather than decompressing every tensor-cache field.
5. **Scene context reuse.** Successor environment state, blocker ordering, and immutable environment contexts are computed once per scene/blocker/horizon where valid.
6. **Optional uncompressed sidecar writes.** `SIDECAR_SAVE_MODE=uncompressed` trades disk space for lower per-file write CPU; default remains `compressed`.

## Observability

- Every worker is unbuffered and streamed through `tee` with a shard prefix.
- Default progress cadence is 30 seconds (`SIDECAR_PROGRESS_EVERY_SECONDS`).
- Progress includes scanned/processed scenes, groups/examples, proposals/verified proposals, elapsed time, scene throughput and ETA.
- Each shard records timing buckets: load/schema, environment, analytic completion, proposal verification, feature/context and write.
- Launcher prints aggregate timing/cache counters after all shards finish.
- `profile_sidecar_train8` runs the real train-side scientific settings on 8 scenes (4 shards × 2 scenes) for server-side wall-clock attribution before a full 5000-scene build.

## Parallelism / GPU policy

- Recommended sidecar execution is CPU multiprocessing: start with `profile_sidecar_train8`, then `build_sidecar_train_parallel8` or `build_sidecar_train_auto` depending on server CPU/RAM/NVMe behavior.
- Do **not** make CUDA the authoritative hard verifier in R2. The remaining verifier is exact/branch-heavy RSS/OBB/TTC + inverse dynamics over small trajectories; a naive GPU port risks changing hard-set membership at floating-point boundaries and may lose to transfer/launch overhead.
- The A30 should be used for `train_rcrso` and Stage-0 model inference. V45 RCRSO is small enough that one A30 is the default; dual-GPU DDP is not enabled because it may be input-pipeline-bound and changing effective batch semantics requires a separate equivalence/profile study.

## Local component benchmarks

Synthetic repeated-work benchmark only; not a server end-to-end claim:

- vectorized kinematics: ~46.0× component speedup;
- shared verifier cache: ~8.38× repeated-candidate verifier speedup;
- cached environment-event feature construction: ~4.86× speedup.

## Validation

- V25→V45R2 focused semantic/integrity suite: **139/139 passed**.
- New performance-equivalence tests cover vectorized-vs-literal kinematics, shared-cache verifier equality, and precomputed-event feature equality.
- Python compile and launcher shell syntax pass.

## Scientific status

RCRSO remains **UNRESOLVED**. R2 changes only runtime/observability and does not authorize any new algorithm branch. The frozen sequence remains Stage-0 support → equivalence16 → lost7 → retained3 → induced9 → remaining29/CF48 → fresh37 → development exact200.
