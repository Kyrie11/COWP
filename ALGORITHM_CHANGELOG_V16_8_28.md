# COWP Algorithm Changelog — v16.8.28

## Scope

v16.8.28 is an **online execution-integrity repair only**. It does not change the dataset, cached labels, checkpoint, natural-root construction, RCOT, BCOT, protected-priority certificate, candidate families, learned heads, frontier ranking, fallback score, or any paper-level planning mechanism.

The v16.8.27 exact-200 Waymax results must **not** be promoted to a physical algorithm conclusion because code review found a second execution-semantic bug on the common no-valid-candidate path.

## Triggering evidence from v16.8.27

The v16.8.27 conventional-safety repair itself worked: the old `NEUTRAL_EGO` conventional-audit bypass is gone, the ordinary/fallback-outcome metadata is method-local, all four methods use the same exact 200 IDs, and the merged statistics reproduce from per-scenario rows.

However, first-event provenance exposed a suspicious `PAD` pattern:

- COWP: `PAD` immediately before first offroad in 11/20 episodes and before first kinematics violation in 13/30;
- COWP + fallback outcome: 14/21 offroad and 11/28 kinematics;
- conventional safety: 10/20 offroad and 12/24 kinematics;
- planner-score-only: 9/19 offroad and 15/28 kinematics.

This pattern occurs across COWP and both baselines, so the shared execution path was audited.

## Critical bug: no-valid state executed zero-padded candidate slot 0

Online candidate tensors are allocated as zero-padded arrays. When no dynamically valid candidate exists, `cand_valid.any()` is false. In v16.8.27 both the COWP path and baseline fast path then used `selected = 0` as a sentinel.

The sentinel was not kept diagnostic-only. The code subsequently executed:

```python
traj = batch_np["cowp/candidates/trajectory"][0, selected]
return self._trajectory_to_action(..., traj)
```

Thus an all-zero `PAD` trajectory was converted into a **valid Waymax ego action**. `_trajectory_to_action()` always marks the SDC action valid. `_consistent_one_step_target()` then interprets the zero waypoint relative to the current world-frame ego pose; when the padded desired velocity is zero it can infer desired speed from distance to the origin, then jerk/acceleration/yaw-limit that nonsensical command into a real action.

This is an execution-semantics error, not an algorithmic result. It directly contaminates offroad/kinematics events and can alter all later closed-loop states after the first no-valid step.

## Repair 1: padding is never executable

A new pure resolver `_resolve_execution_trajectory()` separates **selection** from **execution**:

- if a valid candidate exists, the exact selected candidate trajectory is returned unchanged;
- if the valid pool is empty, no candidate slot is treated as selected or executable;
- execution uses a bounded `smooth_stop_trajectory()` synthesized from the **current ego state**, using the existing `fallback_decel_mps2` and the existing one-step jerk/yaw-rate-limited action projection.

The emergency trajectory is execution-only. It is not inserted into the proposal bank, not marked conventional-safe, not passed through the COWP certificate, and not counted as a certified candidate.

## Repair 2: no-valid provenance is explicit

When the valid pool is empty:

- `selected_candidate = -1` in diagnostics;
- `selected_candidate_valid = false`;
- `selected_candidate_conventional_safe = false`;
- `selected_macro_name = EMERGENCY_BOUNDED_STOP`;
- `emergency_action_used = true`;
- `execution_trajectory_source = bounded_smooth_stop`.

The baseline fast path now uses `baseline_no_valid_emergency_stop` instead of conflating zero-valid states with `baseline_use_valid`.

Episode diagnostics additionally report:

- emergency-action step/episode rate;
- zero-valid and zero-conventional candidate step rates from actual candidate counts;
- emergency execution/source immediately before the first collision, offroad, or kinematics event.

The physical comparison script is upgraded to schema `cowp_v16_8_28_waymax_physical_compare_v3` and reports these fields.

## Repair 3: remove unreachable fallback branch

The old COWP fallback ordering checked `cand_valid.any()` before `(cand_valid & stop_like).any()`. Therefore the later `emergency_stop_like` branch was mathematically unreachable: every valid stop-like candidate already makes `cand_valid.any()` true.

v16.8.28 removes this dead branch. This does **not** alter behavior for any state that had a valid candidate; it only removes misleading code and makes the true no-valid transition explicit.

## Scientific disposition

The v16.8.27 simulator outputs are factual outcomes for the v16.8.27 controller, and the exact-ID pairing/integrity checks pass. They are **not reliable enough for full physical algorithm attribution** because the common execution bug lies directly on many first-event paths and changes subsequent closed-loop state evolution.

Therefore v16.8.28 does **not** promote or reject Recovery Certificate, Execution-Viability Certificate, proposal redesign, BCOT retuning, or any other new planning mechanism. The same exact 200 IDs and all four methods must be rerun first.

Learned-offline conclusions that do not use this online no-valid execution path remain outside the scope of this repair; this version makes no new claim about them.

## Regression contract

New tests require that:

1. a no-valid state cannot return zero padding as the execution trajectory;
2. the emergency trajectory is finite, remains anchored to the current ego world pose, and monotonically decelerates under the trajectory primitive;
3. a valid selected candidate is returned bit-exactly by the resolver;
4. the dead `emergency_stop_like` selector branch is absent;
5. emergency execution provenance reaches first-event episode diagnostics.

Packaged `sanity`: **15/15 passed**.

Focused v16.8.26--v16.8.28 + Waymax diagnostic set: **17/17 passed**.

Full repository: **265 passed / 5 skipped / 8 historical failures**. The 8 failures are the same repository-history classes as before: six tests reference legacy launcher scripts absent from the supplied archive, and two tests hard-code an old semantic fingerprint. No new functional regression was introduced.

## Required rerun

Do not retrain or rebuild the dataset/cache. Reuse the exact 200-ID manifest with logical SHA256:

`3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`

Rerun all four strict methods because the faulty action path is shared:

- `cowp`
- `cowp_fallback_outcome`
- `conventional_safety`
- `planner_score_only`

Only after the repaired first-event provenance is available should physical bottleneck attribution resume.
