# ALGORITHM_CHANGELOG V16.8.38 — Shift-Closed Control-Reachable Recovery Tube

## Triggering evidence: V16.8.37 is reliable but fails Stage-1

V16.8.37 passes an independent 71/71 hard-check reliability audit, so its result is valid for algorithm attribution. It fails the unchanged six-item counterfactual48 promotion gate:

- historical RVR rescues retained: 1/10 (fail; minimum 5/10);
- historical RVR-induced collisions avoided: 9/9 (pass; minimum 7/9);
- net COWP collisions removed: 1 (fail; minimum 3);
- kinematics net regression: +1 scene (pass; maximum +1);
- paired mean EP delta: -0.065162 (fail; minimum -0.05);
- intervention: 8/3840 policy steps (pass; nonzero).

`cowp_recourse_returnability_bridge` is archived and must not run fresh37.

The only collision rescue, `fccd9a25a2a57a73`, simultaneously induces kinematics infeasibility and changes EP from 3.67633 to 0.51512. Removing this single development scene leaves mean EP delta about +0.00071 across the other 47 scenes, but the preregistered Gate is not recomputed after excluding it.

## V37 mechanism attribution

V37 is active and causal, not dead code:

- 144 returnability probes;
- 7 bridge entries;
- 1 direct entry;
- 2 actual bridge executions;
- 1 abort;
- 8 recovery switch steps.

The branch is high precision but unusably low recall. Per probe, roughly 35 distinct emitted-action classes are available/evaluated, but only 0.28–0.31 restoring semantic macros are witnessed on average. Exact return to a non-empty unchanged full-conventional bank within the direct/one-new-replan bridge is therefore almost always an empty-set comparison.

V37 validates the need for explicit terminal/returnability semantics and actual emitted-action reasoning, but falsifies exact short-horizon full-conventional restoration as the sole recovery-entry certificate.

The dominant bottleneck is updated to:

**Constructive Control-Reachable Backup Support under Zero-Conventional Collapse**

Paper-level name:

**Shift-Closed Control-Reachable Physical Feasibility**.

## New method

`cowp_shift_closed_control_reachable_tube`

The method is active only when the full conventional set is empty and at least one valid candidate exists. It does not change data, checkpoint, candidate semantic families, Natural roots, RCOT, BCOT, protected-priority certificate, certificate-compatible set preservation, the 8 s conventional contract, common controller limits, or accepted-path behavior.

### 1. Existing semantic parent support

Use the historical roadgraph-first recovery parent pool. If no valid+roadgraph-safe parent exists, use all valid parents and re-audit every realized tube. Exclude PAD.

Deduplicate by `(semantic macro, actual emitted first target)` so one macro may retain multiple physically distinct actions while duplicate labels cannot fabricate support.

### 2. Constructive control lifting

For every parent geometry, propagate three longitudinal controller realizations:

- mode 0: unchanged nominal stateful controller;
- mode -1: lower endpoint of the existing acceleration/jerk reachable interval at every edge;
- mode +1: upper endpoint of the same interval at every edge.

For previous acceleration `a_{t-1}`:

```text
lo_t = max(-max_decel, a_{t-1} - max_jerk*dt)
hi_t = min( max_accel, a_{t-1} + max_jerk*dt)
```

These are endpoints of the already-frozen controller envelope, not relaxed constraints or tuned proposal offsets. Lateral/yaw realization continues to use the unchanged online projection.

Historical callers of `_project_candidate_bank_through_controller_np` use the original `None` defaults. A regression requires the mode-0 first target to match the unchanged online one-step controller within 2e-5.

### 3. Full physical tube certificate

A control-realized tube is admissible only if it passes, over the full frozen horizon:

- finite-state check;
- unchanged roadgraph drivable screen;
- unchanged causal constant-velocity collision screen;
- Waymax-aligned inverse acceleration/steering-curvature feasibility on every edge.

The tube is not relabeled conventional-safe or NCF. Social feasibility semantics remain untouched.

### 4. One-step shift closure

After executing the actual first target:

1. construct the causal successor state;
2. shift the realized tube left by one edge;
3. append only a constant-velocity terminal edge;
4. carry the emitted longitudinal acceleration as successor controller memory;
5. re-project the shifted reference through the same controller envelope mode;
6. require the complete shifted tube to pass the same physical certificate.

This is an invariant-like one-replanning closure test on the same backup tube, not V2/V3/V4 returnability-tree depth or future logged-state access.

### 5. Hard-certified selection

If no shift-closed tube exists, execute the unchanged COWP least-coercive-valid fallback.

Otherwise select:

```text
hard shift-closed physical set
→ minimum frozen COWP fallback score
→ minimum absolute first-acceleration deviation
→ prefer nominal mode on exact tie
→ deterministic parent/mode index
```

No risk/progress/collision scalarization, profile area/AUC, horizon discount, relaxed threshold, fixed dwell time, mode commitment, or learned viability head is introduced.

The selected projected trajectory/first target/acceleration is passed to execution through an explicit override; the mechanism is not analysis-only.

### 6. No persistent recovery mode

Each changed action must own a complete shift-closed tube certificate at the current observed state. The next real planning step reconstructs and recertifies. Continuity is supplied by shift closure, not heuristic hysteresis or unconditional commitment.

## Added diagnostics

- recovery-tube probe/certificate/action-change rates;
- parent pool and distinct action classes;
- generated tube hypotheses and unique physical first actions;
- full-safe and shift-closed counts;
- nominal/lower/upper envelope shift-closed counts;
- lifted-only parent support;
- selected lifted rate, envelope mode, parent candidate/macro;
- selected first-acceleration delta;
- current and shifted collision margins;
- selected fallback-score delta.

A nonzero lifted-selection or lifted-only-support signal is required to attribute any successful outcome specifically to constructive support expansion, but it is not a post-hoc seventh promotion Gate.

## Frozen layers

Freeze:

- compact-5k data/labels/checkpoint;
- Natural roots;
- RCOT and BCOT;
- protected-priority hard certificate;
- certificate-compatible set-preservation frontier;
- outcome head/settings;
- 8 s conventional-safe definition;
- V27/V28 integrity fixes;
- semantic candidate families and map/Frenet primitives;
- common controller limits and ordinary execution path;
- accepted/certified-path selection.

Accepted-path execution viability remains a separate secondary bottleneck.

## Newly prohibited directions

All previous prohibitions remain. Add:

1. Do not relax the V37 rescue/EP Gate or exclude `fccd...` after seeing the result.
2. Do not continue direct/indirect returnability as a 2/3/4-step bridge tree; this repeats failed horizon stacking.
3. Do not replace exact restoration with a weighted proxy over conventional count, prefix, macro count, time-to-return, risk, collision, or progress.
4. Do not continue the base-vs-RVR or same-bank selector/frontier family after V36/V37.
5. Do not relax the full V38 tube certificate, shorten the 8 s collision contract, or widen controller limits after a failed Gate.
6. Do not introduce fixed dwell, hysteresis epsilon, minimum commitment, or release threshold.
7. Do not use nominal first-waypoint exact reachability as executable support.
8. Do not globally retune the common controller or mix accepted-path kinematics into V38.
9. Do not train a neural reachable-tube/returnability head before the analytic tube target is validated.
10. Do not claim strong social-burden causality from logged-replay Waymax outcomes.

## Promotion protocol

Stage-1 retains the exact six gates from V16.8.33–37:

- retain at least 5/10 old RVR rescues;
- avoid at least 7/9 old RVR-induced collisions;
- remove at least 3 net COWP collisions;
- kinematics net regression no worse than +1 scene;
- paired mean EP delta at least -0.05;
- nonzero action-changing intervention.

Only Stage-1 pass may launch fresh37. The launcher is fail-closed. Fresh37 then requires no net collision/CR harm, offroad and kinematics regressions no worse than +1 scene, mean EP delta at least -0.03, and nonzero intervention before historical exact200 development confirmation.

All panels remain development evidence, not publication holdout.

## Validation

- V16.8.38 dedicated tests: 7/7 passed;
- V16.8.25→38 focused semantic/integrity suite: 68/68 passed;
- Python compilation: passed;
- launcher bash syntax: passed;
- exact200/equivalence16/counterfactual48/fresh37 manifest hashes: passed;
- analyzer smoke: passed;
- direct fresh37 without Stage-1 analysis exits with code 4 before rollout;
- conventional-safety bypass grep: none;
- V38 support-construction helper does not consume logged future.

## CCF-A positioning

The tube and shift-closure device is not claimed as standalone novelty. The candidate paper contribution remains **Orthogonal Option-Set Feasibility**:

- social axis: preserve protected agents' natural low-burden response sets;
- physical axis: preserve ego control-reachable, shift-closed backup support during uncertified recovery.

The shared principle remains: **safety must not be obtained through critical option-set collapse**.
