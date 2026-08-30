# ALGORITHM CHANGELOG — V16.8.41

## Version

**V16.8.41 — Shift-Closure Semantic Fidelity Repair (SC-FAVI-R)**

Online method alias remains:

```text
cowp_shift_closed_first_action_viability_interval
```

Release role:

```text
RELIABILITY REPAIR ONLY
NOT A NEW ALGORITHM HYPOTHESIS
REQUIRES EQUIVALENCE16 + COUNTERFACTUAL48 RERUN
```

## Why this release exists

The uploaded V16.8.40 artifacts pass code provenance, manifest, shard, merged-summary, standard-metric, common-path equivalence, no-valid invariant and analyzer-replay checks. However, independent mechanism-fidelity review finds one hard discrepancy between the frozen V39 shift-closure semantics and the V40 interval-completion implementation.

Audit verdict:

```text
243 hard checks
242 passed
1 failed
mechanism fidelity = FAIL
algorithm attribution allowed = false
```

V39 shifted schedule semantics:

```text
finite event-release / nominal: append 0
LOWER_ALL:                    append -1
UPPER_ALL:                    append +1
```

Uploaded V40 interval completion always appended `0`, including `LOWER_ALL/UPPER_ALL` bases.

Because the successor shifted certificate is a hard set-membership condition, this can create false-negative interval support. The uploaded V40 outcome and provisional 5/6 Stage-1 result must not be used for algorithm attribution.

## Scientific decision

No V16.8.41 research mechanism is introduced. The user's reliability-first rule is applied literally:

```text
unreliable result
→ no algorithm attribution
→ no dominant-bottleneck update
→ no next-branch selection
→ code repair and exact rerun only
```

The repaired result must first pass a new reliability audit. Only then may the unchanged six-item Gate be interpreted.

## Code change

### Shared schedule-shift helper

Added:

```python
_shift_longitudinal_envelope_schedule_np(schedule, policy_id)
```

It implements the literal frozen V39 rule:

```text
shifted[:-1] = schedule[1:]

if policy_id == -1:
    shifted[-1] = -1
elif policy_id == +1:
    shifted[-1] = +1
else:
    shifted[-1] = 0
```

Both constructors now call this helper:

```text
_construct_conflict_window_control_reachable_tube_np
_construct_shift_closed_first_action_viability_interval_np
```

The change centralizes policy identity and prevents future divergence between nested V39 and interval completion.

### Fail-closed launcher

Added:

```text
NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh
```

The script:

- writes to a new V41 output root;
- preserves the original online method alias;
- runs the same frozen manifests;
- uses the same six Stage-1 Gate values;
- requires repaired Stage-1 pass before fresh37;
- exits with code 4 on a missing or failed promotion Gate;
- creates no rollout shards after a failed Gate.

### Analyzer

Added:

```text
cowp/scripts/94_analyze_shift_closure_semantic_fidelity_repair.py
```

It keeps the existing paired and mechanism aggregations and writes:

```text
preregistered_gate.shift_closure_semantic_fidelity_repair
```

The Gate is numerically unchanged:

- retain at least 5/10 historical RVR rescues;
- avoid at least 7/9 historical RVR-induced collisions;
- remove at least 3 net COWP collisions;
- kinematics net regression at most +1 scene;
- paired mean EP delta at least -0.05;
- nonzero action-changing intervention.

## Frozen contracts

V16.8.41 changes none of the following:

- compact-5k data, labels or splits;
- checkpoint or training;
- natural roots, RCOT or BCOT;
- protected-priority non-coercive certificate;
- certificate-compatible set-preservation frontier;
- outcome head/settings;
- 8 s conventional-safety contract;
- proposal geometry or semantic macro families;
- controller acceleration/deceleration/jerk/yaw limits;
- roadgraph or causal-CV collision audit;
- Waymax-aligned kinematics adapter;
- full current tube certificate;
- one-step successor construction;
- interval seeds, secant or quadratic proposals;
- hard-set ordering;
- actual execution override;
- no-valid emergency semantics.

## Information boundary

The repair consumes only the current schedule and policy identity. It does not use:

- Waymax logged future states;
- future outcome labels;
- online mechanism ground truth;
- real counterfactual trajectories for unexecuted policies.

## New tests

Added `tests/test_v16_8_41_shift_closure_semantic_fidelity_repair.py`:

1. all-horizon lower/upper endpoint retention;
2. finite event-release terminal nominal behavior;
3. shared-helper use by both V39 and V40 constructors;
4. exact randomized equivalence to the literal frozen V39 rule across several horizons and policy IDs;
5. no control-limit widening.

## Validation

```text
V41 dedicated repair tests          4/4 passed
V16.8.25→41 focused suite          86/86 passed
Python compile                      passed
launcher bash syntax                passed
four manifest hashes                passed
analyzer smoke                      passed
failed-Gate fresh37 exit code       4
rollout artifacts after failed Gate 0
```

Full repository collection still stops at the pre-existing historical test importing absent `_recovery_bridge_viability_mask`. The same error is reproduced in pristine uploaded V40; it is not a V41 regression and the archived API is not restored.

## Required rerun

```bash
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh make_ids
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_41_SHIFT_CLOSURE_SEMANTIC_FIDELITY_REPAIR_CN.sh analyze_counterfactual48
```

Stop after repaired Stage-1. Do not run fresh37 unless:

```text
preregistered_gate.shift_closure_semantic_fidelity_repair.pass == true
```

## Attribution discipline after rerun

1. Audit V41 result reliability before reading the algorithm result.
2. If unreliable, repair only again.
3. If reliable, apply the unchanged six-item conjunction Gate.
4. Only after a reliable Gate result may mechanism attribution or a subsequent algorithm branch be selected.
5. Do not compare the repaired result against uploaded V40 as though V40 were a valid scientific baseline; use V39 and frozen references for attribution.

## Newly prohibited post-hoc changes

In addition to all historical prohibitions:

- do not interpret uploaded V40's provisional 3/10 rescue retention as a mechanism failure;
- do not modify any Gate value during the repaired rerun;
- do not add action-grid points, interpolation orders, release schedules or switch times in V41;
- do not alter full certificate or shift closure to increase coverage;
- do not mix a new V42 mechanism into the repair rollout;
- do not claim this semantic repair as a paper contribution.
