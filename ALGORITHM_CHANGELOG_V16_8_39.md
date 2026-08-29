# ALGORITHM CHANGELOG — V16.8.39

## Version

**V16.8.39 — Conflict-Window Control-Reachable Recovery Tube (CW-CRT)**

Method:

```text
cowp_conflict_window_control_reachable_tube
```

## Reliability prerequisite and V16.8.38 decision

V16.8.38 results pass independent reliability audit: **169/169 hard checks passed**. The counterfactual48 and equivalence16 shards, frozen manifests, merged rows, headline metrics, fallback counts, common-path equivalence, release hashes, execution invariants, and analyzer replay are internally consistent. Attribution is allowed.

V16.8.38 fails the unchanged six-item Stage-1 Gate:

- historical RVR rescues retained: **0/10** (required >=5/10; FAIL);
- historical RVR-induced collisions avoided: **9/9** (required >=7/9; PASS);
- net COWP collisions removed: **1** (required >=3; FAIL);
- kinematics net regression: **0** (required <=+1; PASS);
- paired mean EP delta: **-0.004746** (required >=-0.05; PASS);
- action-changing intervention: **55 steps** (required >0; PASS).

Decision:

```text
V16.8.38 POLICY = ARCHIVE / NO FRESH37
```

The policy is not promoted. Its hard tube construction and shift-closure mechanism are retained as evaluators/support primitives.

## V16.8.38 mechanism evidence

Pooled over 48x80 policy steps:

- 2695 recovery probes;
- 56 hard tube certificates;
- 55 action changes;
- 55/56 certified selections use controller-lifted support;
- 260640 generated tube hypotheses;
- 522 full physically safe witnesses;
- 486 shift-closed witnesses;
- shift-closure retention of full-safe witnesses: 486/522 = 93.10%;
- nominal/lower-all/upper-all shift-closed witnesses: 3/425/58.

The dominant sparsity is therefore before shift closure. V38's three constant all-horizon control sequences are too coarse to cover finite collision windows. The clean rescue `6992366c5c998d00` proves constructive support can remove collision without offroad/kinematics conversion and with positive EP, but policy recall is inadequate.

The V38 analyzer's conditional lifted-selection mean was scenario-macro averaged. The correct pooled value is 55/56 = 98.21%. This reporting correction does not change any outcome or Gate. The V39 analyzer uses pooled counts with explicit denominators.

## Dominant bottleneck

**Finite-duration control-reachable backup support under zero-conventional collapse.**

Paper-level formulation:

**Constructive control-sequence support for shift-closed physical feasibility.**

This replaces selector tuning, exact return-to-existing-bank, deeper bridge trees, or certificate relaxation as the P0 branch.

## Frozen contracts

V16.8.39 does not change:

- compact-5k data, labels, split, checkpoint, or training;
- natural roots, same-root RCOT, BCOT;
- protected-priority hard non-coercive certificate;
- certificate-compatible post-certificate set preservation;
- outcome head/settings;
- the 8 s conventional-safety contract;
- candidate geometries, semantic macro families, map/Frenet primitives;
- common controller acceleration, deceleration, jerk, yaw and dynamics limits;
- accepted/certified COWP path;
- no-valid emergency and V27/V28 integrity fixes.

V39 only changes the control-sequence support constructed inside the zero-conventional + valid recovery branch.

## Algorithm

### 1. Parent/action-class support

Use valid and nominal-roadgraph-safe parents when available; otherwise use valid parents, while requiring every realized tube to pass the full roadgraph certificate. Exclude PAD. Deduplicate by semantic macro plus actual emitted first action.

### 2. Nominal causal conflict window

For each parent, first realize the nominal trajectory through the unchanged controller. Under the frozen sampled causal-CV collision contract, identify the first and last violating edges:

```text
E_i = {h : nominal control-realized parent violates collision screen at h}
h_first = min E_i
h_last  = max E_i
```

No logged future or outcome is read.

### 3. Nested schedule family

Retain V38 exactly:

```text
NOMINAL
LOWER_ALL
UPPER_ALL
```

Add only four event-derived endpoint-release schedules:

```text
LOWER_TO_FIRST_CONFLICT
UPPER_TO_FIRST_CONFLICT
LOWER_TO_LAST_CONFLICT
UPPER_TO_LAST_CONFLICT
```

The lower/upper endpoint is used from the current edge through the event edge, then the unchanged nominal controller resumes. Exact duplicate schedules are removed. No arbitrary switch-time grid, tuned margin, fixed dwell, learned release, or weighted objective is used.

### 4. Full physical certificate

Every candidate emitted action must own a complete horizon tube satisfying:

```text
finite
AND frozen roadgraph safe
AND frozen causal-CV collision safe
AND Waymax-aligned kinematics safe
```

The backup is not relabeled conventional-safe, NCF, or socially certified.

### 5. One-step shift closure

After the actual first action, construct the causal successor, shift the realized tube, append a constant-velocity terminal edge, carry emitted acceleration as controller memory, shift the schedule, re-project, and require the same full certificate. Event-release schedules append nominal control; all-horizon V38 endpoints retain their endpoint at the terminal edge, preserving exact nesting.

### 6. Hard-set selection and execution

If no shift-closed witness exists, use unchanged COWP least-coercive-valid fallback.

Otherwise select lexicographically:

```text
minimum frozen COWP fallback score
→ minimum absolute first-acceleration deviation
→ fewer nonnominal edges
→ nominal, then event-release, then all-horizon endpoint on exact tie
→ deterministic parent/policy id
```

The selected realized first target and acceleration are passed through explicit execution override.

## Added diagnostics

- pooled probe/certificate/action-change counts and rates;
- parents with nominal conflict;
- mean first/last conflict edge;
- generated and unique-action hypotheses;
- full-safe and shift-closed witnesses;
- nested V38 nominal/lower-all/upper-all witnesses;
- lower/upper event-release witnesses;
- event-release-only parent support;
- selected event-release rate, release edge and nonnominal duration;
- first-acceleration deviation, current/shifted collision margins, fallback-score delta.

Event-release-only support and selection are attribution diagnostics, not a post-hoc seventh outcome Gate.

## Preregistered promotion protocol

Stage-1 keeps the exact prior six gates:

- retain >=5/10 old RVR rescues;
- avoid >=7/9 old RVR-induced collisions;
- remove >=3 net COWP collisions;
- kinematics net regression <=+1 scene;
- paired mean EP delta >=-0.05;
- nonzero action-changing intervention.

Only Stage-1 pass may launch fresh37. The launcher is fail-closed. Fresh37 must then pass its frozen no-harm/generalization Gate before historical exact200 development confirmation. All are development evidence, not publication holdout.

## Falsification branch

If event-release-only support is near zero or the six-item Gate fails, stop finite hand-enumerated schedule patches. Do not search arbitrary release times after seeing outcomes. Move to constrained reachable-set optimization and/or interaction-aware response envelopes.

## Newly prohibited directions

All previous prohibitions remain. Add:

1. Do not relax the V38 full physical certificate or one-step shift closure to raise recall.
2. Do not change the 8 s contract or controller limits.
3. Do not tune first/last conflict release definitions or enumerate an outcome-selected switch grid.
4. Do not add risk/progress/collision/kinematics/profile weights.
5. Do not add fixed dwell, commitment duration, hysteresis epsilon, or release margins.
6. Do not resume selector/frontier, exact-returnability depth, or horizon-stacking families.
7. Do not rebuild compact-5k or expand map/Frenet primitives in V39.
8. Do not mix accepted-path execution viability into this recovery experiment.
9. Do not train a neural tube/reachable-set head before the analytic target is validated.
10. Do not claim strong social-burden causality from logged-replay Waymax.

## Validation

- V16.8.39 dedicated tests: **7/7 passed**;
- V16.8.25→39 focused semantic/integrity suite: **75/75 passed**;
- Python compilation: passed;
- launcher bash syntax: passed;
- all frozen manifest hashes: passed;
- V38 nested schedule identity: tested;
- event-window exact dedup and event-release-only support: tested;
- first-target controller equivalence and shifted fail-closed behavior: tested;
- pooled analyzer smoke: passed;
- promotion launcher fail-closed: passed;
- no conventional-safety bypass or logged-future dependency.

Full repository collection still stops at the historical `tests/test_v16_8_29_recovery_viability.py` import of deleted `_recovery_bridge_viability_mask`. The same error reproduces on the unmodified uploaded V38 tree and is not a V39 regression.

## CCF-A positioning

CW-CRT is a mechanism probe, not standalone novelty. The paper candidate remains **Orthogonal Option-Set Feasibility**:

- social axis preserves protected agents' natural low-burden option sets;
- physical axis preserves ego control-reachable, shift-closed backup support under uncertified recovery.

Shared principle: **safety must not be obtained through critical option-set collapse**.
