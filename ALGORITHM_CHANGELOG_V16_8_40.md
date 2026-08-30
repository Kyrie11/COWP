# ALGORITHM CHANGELOG — V16.8.40

## Version

**V16.8.40 — Shift-Closed First-Action Viability Interval (SC-FAVI)**

Method:

```text
cowp_shift_closed_first_action_viability_interval
```

## Reliability prerequisite and V16.8.39 decision

The uploaded V16.8.39 code/results pass an independent reliability audit: **222/222 hard checks passed**. The audit independently verifies archive/release provenance, frozen manifests, 24+24 counterfactual shards, exact shard-to-merged reconstruction, all standard metrics and fallback aggregation, 16-scene common-path equivalence, the V28 no-valid execution invariant, causal protocol metadata, and a pristine-source analyzer replay with zero recursive mismatch at tolerance `1e-12`.

V16.8.39 fails the unchanged six-item Stage-1 Gate:

- historical RVR rescues retained: **3/10** (required `>=5/10`; FAIL);
- historical RVR-induced collisions avoided: **9/9** (required `>=7/9`; PASS);
- net COWP collisions removed: **4** (required `>=3`; PASS);
- kinematics net regression: **0** (required `<=+1`; PASS);
- paired mean EP delta: **-0.011079** (required `>=-0.05`; PASS);
- action-changing intervention: **87 steps** (required `>0`; PASS).

Decision:

```text
V16.8.39 POLICY = ARCHIVE / NO FRESH37
```

The recall threshold is not changed after seeing the result. V39 is not promoted even though it has no induced collision/offroad/kinematics failure on counterfactual48.

## V16.8.39 mechanism evidence

Headline:

```text
COWP: 34/48 collision, 1/48 offroad, 6/48 kinematics, EP 1.002512
V39 : 30/48 collision, 1/48 offroad, 6/48 kinematics, EP 0.991434
```

Paired against COWP:

- collision: **4 rescue / 0 induced**;
- offroad: 0 rescue / 0 induced;
- kinematics: 0 rescue / 0 induced;
- mean EP delta: `-0.0110785`, bootstrap interval crosses zero;
- McNemar exact collision `p=0.125`.

Pooled mechanism counts:

- 2632 recovery probes;
- 88 hard certificates;
- 87 actual action changes;
- 83/88 certified selections use event-release schedules;
- 593253 future tube hypotheses;
- 157205 unique first-action hypotheses;
- 1844 full physically safe witnesses;
- 1754 shift-closed witnesses;
- 1271 event-release shift-closed witnesses;
- 506 event-release-only parent supports;
- shift closure retains `1754/1844 = 95.12%` of full-safe witnesses.

Relative to V38, V39 increases hypotheses per probe from `96.71` to `225.40`, full-safe witnesses per probe from `0.194` to `0.701`, and shift-closed witnesses per probe from `0.180` to `0.666`. However, unique actual first actions per probe remain essentially unchanged (`59.92` to `59.73`). Event-release changes future control tails, but every first action is still one of the same nominal/lower/upper values.

Six of the seven lost historical RVR rescues receive zero V39 certificate/action-change steps. The seventh (`fccd9a25a2a57a73`) receives four event-release interventions but still collides. Therefore the remaining dominant recall failure is not another release-time shortage.

## Dominant bottleneck

**Shift-closed first-action viability support under ternary control quantization.**

Paper-level formulation:

**Control-Interval Reachable Backup Support.**

The actionable question is now:

> When the fixed nominal/lower/upper first actions have no shift-closed certificate, does the interior of the already-existing jerk/acceleration-reachable first-action interval contain a hard-certified action that preserves the unchanged V39 future witness?

This replaces further hand-enumerated release schedules, selector tuning, certificate relaxation, or deeper lookahead as the P0 branch.

## Frozen contracts

V16.8.40 does not change:

- compact-5k data, labels, split, checkpoint, or training;
- natural roots, same-root RCOT, BCOT;
- protected-priority hard non-coercive certificate;
- certificate-compatible post-certificate set preservation;
- outcome head/settings;
- the 8 s conventional-safety contract;
- candidate geometries, semantic macro families, map/Frenet primitives;
- common controller acceleration, deceleration, jerk, yaw and dynamics limits;
- accepted/certified COWP path;
- conventional fallback and no-valid emergency semantics;
- V38/V39 full physical tube certificate and one-step shift closure.

V40 only adds support inside the zero-conventional + valid branch after V39 proves its own hard set empty.

## Algorithm

### 1. Exact V39 nesting

V40 first calls the unchanged V39 conflict-window tube constructor.

```text
if V39 hard set is non-empty:
    return the exact V39 selected record
    do not evaluate V40 interval support
```

The selected parent, trajectory, emitted target, acceleration, fallback ordering, certificate, and execution override are preserved. This prevents the new recall experiment from perturbing V39's observed high-precision layer.

### 2. Existing first-action reachable interval

Only when V39 has no hard-certified tube, V40 reuses each non-nominal V39 future witness and exposes the current first acceleration as an interval.

For previous longitudinal acceleration `a_prev`, unchanged limits define:

```text
a_lo = max(-max_decel, a_prev - max_jerk * dt)
a_hi = min( max_accel, a_prev + max_jerk * dt)
```

For a lower or upper V39 witness, the interval is the segment from the parent's unchanged nominal emitted acceleration to the corresponding reachable endpoint. The lower and upper segments together cover the existing `[a_lo, a_hi]` support; no controller bound is widened.

Schedules that differ only at edge zero are deduplicated because edge zero is replaced by the interval action. Candidate geometry and every later V39 schedule edge remain unchanged.

### 3. Hard-margin action proposals

For each interval basis, V40 first evaluates the canonical fractions:

```text
0.0  nominal endpoint
0.5  interval midpoint
1.0  reachable endpoint
```

Their causal hard collision margins generate deterministic algebraic proposals:

- secant boundary roots on `[0, 0.5]` or `[0.5, 1]` when the hard margin changes sign;
- the interior maximizer and real roots of the unique quadratic interpolant through the three margins.

These values only propose actions. They never certify or rank them. Every proposed acceleration is clipped by the unchanged reachable interval, projected through the unchanged stateful controller, and subjected to the unchanged hard certificates.

No outcome label, logged future, learned score, arbitrary resolution grid, tuned fraction, margin threshold, or scalar risk/progress trade-off is used.

### 4. Unchanged full physical certificate

Every current tube must satisfy across the complete frozen horizon:

```text
finite
AND frozen roadgraph safe
AND frozen causal-CV collision safe
AND Waymax-aligned kinematics safe
```

The recovery is never relabeled conventional-safe, NCF, or socially certified.

### 5. Unchanged one-step shift closure

After the actual first action, V40 constructs the causal successor, shifts/appends the realized reference, carries emitted longitudinal acceleration as controller memory, shifts the unchanged V39 future schedule, reprojects through the same controller, and requires the same full physical certificate.

### 6. New emitted-action requirement

A certificate whose first target is already one of the existing parent nominal/end-point targets is retained only as a diagnostic. It cannot trigger V40.

```text
V40 intervention requires:
    actual emitted first target is new
    AND current full certificate passes
    AND shifted full certificate passes
```

This prevents a different hypothetical future tail from being counted as new closed-loop support when the action executed now is unchanged.

### 7. Hard-set selection and execution

If no genuinely new shift-closed action exists, V40 fails closed to unchanged COWP fallback.

Otherwise select inside the hard set:

```text
minimum frozen COWP fallback score
→ minimum |first acceleration - parent nominal acceleration|
→ fewer nonnominal future edges
→ event-release before all-horizon endpoint on exact tie
→ deterministic parent/policy/boundary identity
```

The selected control-realized trajectory, first target, and acceleration are passed through the explicit execution override.

## Added diagnostics

- nested V39 certificate/selection rate;
- interval-completion attempt steps;
- interval bases;
- seed evaluations and algebraic boundary proposals;
- total interval hypotheses;
- certified unique actions and genuinely new first actions;
- full-safe and shift-closed interval witnesses;
- interval-only parent support;
- selected interval/new-first-action rates;
- selected first-acceleration fraction and boundary source;
- pooled attempt/certificate/action-change denominators.

The new-action diagnostics are attribution checks, not a post-hoc seventh Gate.

V40 also fixes a V39 diagnostic-only omission: the generic `no_conventional_step_rate` reason set now includes the V39 and V40 fallback reasons. V39 outcome, method-specific tube diagnostics, zero-conventional exposure, and every Gate value were unaffected.

## Preregistered promotion protocol

Stage-1 keeps the exact six prior gates:

- retain `>=5/10` old RVR rescues;
- avoid `>=7/9` old RVR-induced collisions;
- remove `>=3` net COWP collisions;
- kinematics net regression `<=+1` scene;
- paired mean EP delta `>=-0.05`;
- nonzero action-changing intervention.

Only Stage-1 pass may launch fresh37. The launcher is fail-closed. Fresh37 must then pass its frozen no-harm/generalization Gate before historical exact200 development confirmation.

Attribution discipline:

- an outcome Gate pass with nonzero new-first-action selection supports the interval mechanism;
- an outcome Gate pass with approximately zero new-first-action selection cannot be attributed to V40 interval completion;
- a Gate failure with nonzero new first actions falsifies this analytic interval model and triggers the interaction-aware reachable-response branch;
- an empty interval support set triggers constrained reachable geometry/proposal construction.

## Newly prohibited directions

All prior prohibitions remain. Add:

1. Do not add more hand-enumerated release times or arbitrary switch-time grids.
2. Do not tune midpoint fractions, interpolation roots, numerical margins, or action grids on counterfactual48 outcomes.
3. Do not relax the full physical certificate or shift closure.
4. Do not shorten the 8 s conventional contract.
5. Do not change common controller limits to create support.
6. Do not add collision/risk/progress/kinematics/profile scalar weights.
7. Do not resume selector/frontier, returnability-depth, horizon-stacking, fixed-dwell, or hysteresis-margin patches.
8. Do not rebuild compact-5k, change RCOT/BCOT, or expand map/Frenet primitives in V40.
9. Do not mix accepted-path kinematics repair into this recovery iteration.
10. Do not train a learned viability head before the analytic support object is validated.
11. If V40 fails with nonzero interval intervention, do not add more interpolation seeds; move to an interaction-aware reachable response envelope.

## Validation

- V16.8.40 dedicated tests: **7/7 passed**;
- V16.8.25→40 focused semantic/integrity suite: **82/82 passed**;
- Python compilation: passed;
- launcher bash syntax: passed;
- all frozen manifest hashes: passed;
- exact nested-V39 preservation: tested;
- interior controller action realization: tested;
- interval-only new first-action support: tested;
- non-new-action fail-closed behavior: tested;
- missing shifted certificate fail-closed behavior: tested;
- no logged-future request: tested;
- V40 analyzer smoke: passed;
- direct fresh37 without Stage-1 result: exits with code 4 before rollout.

The full repository still contains the historical collection failure in `tests/test_v16_8_29_recovery_viability.py`, which imports the archived `_recovery_bridge_viability_mask`. The same failure exists in the unmodified uploaded V39 archive and is not a V40 regression.

## CCF-A positioning

SC-FAVI is a falsifiable physical-support probe, not standalone novelty.

The paper candidate remains **Orthogonal Option-Set Feasibility**:

- social axis: protect other critical agents' natural low-burden option sets;
- physical axis: protect ego's control-reachable, shift-closed backup set during uncertified recovery.

Shared principle:

> **Safety must not be obtained through critical option-set collapse.**

V40 tests whether the physical option-set object must include the continuous control-reachable first-action interval rather than only a finite list of future trajectory schedules.
