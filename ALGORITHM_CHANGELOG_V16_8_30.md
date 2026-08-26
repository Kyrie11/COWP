# COWP Algorithm Changelog — v16.8.30

## Scope

v16.8.30 is a **mechanism-attribution release** built on the clean v16.8.29 exact-200 evidence. It does not rebuild the dataset, retrain the checkpoint, change natural roots, RCOT, BCOT, protected-priority certificate semantics, the set-preservation frontier, the learned outcome head, the conventional-safe definition, or the no-valid execution-integrity repair.

The release retires unconditional v16.8.29 Recursive Viability Recovery (RVR) as a promotion candidate and introduces two opt-in zero-conventional branches:

1. `cowp_rvr_pareto_guard` — a diagnostic branch only;
2. `cowp_successor_option_viability` — the primary next mechanism probe.

Both branches act **only** when the full conventional-safe candidate set is empty while dynamically valid candidates still exist. All certified and conventional-safe paths remain identical to COWP.

---

## v16.8.29 result integrity: accepted for algorithm attribution

The uploaded v16.8.29 exact-200 outputs pass the attribution integrity audit:

- exact manifest: 200 unique IDs, logical SHA256 `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`;
- COWP and RVR scenario sets both equal the manifest;
- each method is merged from two 100-scene shards;
- the 64 dev scenes are an exact subset of confirm200;
- all 64 COWP scenario rows and all 64 RVR scenario rows are identical between the dev64 run and confirm200 at tolerance `1e-9`;
- the packaged v16.8.29 COWP-vs-v16.8.28 equivalence check passed with zero mismatches on the 64 dev scenes;
- no new conventional-audit bypass, zero-padding execution, or method-metadata corruption was found in the active execution path.

The uploaded result zip did not include the two JSON files normally emitted by `analyze_confirm200`, but the merged exact-200 outputs are complete and the shipped analysis scripts reproduce those derived summaries. This is a packaging omission, not an evaluator or algorithm-integrity blocker.

Machine audit: `V16_8_29_RESULT_INTEGRITY_AND_MECHANISM_AUDIT.json`.

---

## v16.8.29 exact-200 result: RVR fails promotion

COWP:

- CR = **19.5%**
- collision = **17.0%**
- offroad = **3.0%**
- kinematics infeasible = **12.5%**
- EP = **1.04607**

RVR:

- CR = **19.5%**
- collision = **16.5%**
- offroad = **3.5%**
- kinematics infeasible = **14.5%**
- EP = **1.00470**

Paired exact-200 collision transitions:

- COWP collision -> RVR safe: **10 scenes**;
- COWP safe -> RVR collision: **9 scenes**;
- shared collision: **24 scenes**;
- McNemar exact `p = 1.0`.

RVR therefore removes only one net collision episode while adding offroad and kinematics failures. Paired finite-scene EP delta is `-0.04137`, with bootstrap 95% CI approximately `[-0.0984, 0.0061]`.

**Disposition: do not promote unconditional max-prefix RVR. Do not tune a prefix weight or shorten the conventional horizon to rescue it.**

---

## Why dev64 looked positive but confirm200 did not

The development panel contained all 34 v16.8.28 COWP collision scenes plus 30 high-zero-conventional non-collision scenes. On this selected panel, RVR rescued 10 collision scenes and induced zero new collision scenes.

All ten rescues are real: they reproduce exactly in confirm200. The failure is that confirm200 contains **nine additional RVR-induced collisions, all outside dev64**. Thus the discrepancy is not stochasticity or shard merging. It is a development-set selection blind spot: the old dev64 was designed to measure rescue of known failures, not counterfactual harm on previously safe scenes.

Future development gates must explicitly contain induced-failure counterexamples and ordinary stable controls.

---

## Mechanism conclusion: safe prefix is informative but not sufficient viability

The zero-conventional decomposition is now decisive for the collision-side operating regime:

- mean zero-conventional step rate: **55.69%**;
- `collision_empty`: **52.53%**;
- `roadgraph_empty`: **0.11%**;
- `road_and_collision_empty`: **2.66%**;
- `intersection_empty`: **0.39%**.

The dominant issue is therefore **collision-screen feasibility collapse**, not map/roadgraph support.

RVR acts on **53.68%** of all policy steps, but raises the mean selected collision-safe prefix only from 39.23 to 40.70 raw steps, i.e. about **+1.48 sampled raw steps on average**. This is a very large intervention footprint for a small immediate local-survival gain.

The paired collision groups expose the real mechanism:

### 10 rescued scenes

Compared with COWP, RVR:

- reduces zero-conventional exposure by **14.1 pp**;
- increases mean conventional candidate count by **+2.12**;
- increases mean max prefix by **+16.33 steps**;
- increases mean selected prefix by **+20.03 steps**;
- reduces fallback exposure by **8.0 pp**;
- but sacrifices mean EP by **-0.7735**.

These scenes prove that the unchanged proposal bank sometimes contains an uncertified action whose execution moves the system into a materially more recoverable future state. Therefore current failures cannot yet be attributed solely to an immutable proposal-support ceiling.

### 9 induced-collision scenes

Compared with COWP, RVR:

- increases zero-conventional exposure by **22.5 pp**;
- reduces mean conventional candidate count by **-4.15**;
- reduces mean max prefix by **-16.11 steps**;
- reduces mean selected prefix by **-14.32 steps**;
- increases fallback exposure by **13.3 pp**;
- increases mean selected action risk by **+0.0715**;
- increases mean selected rule risk by **+0.1278**.

The first RVR collisions occur at a median policy step of 61, not immediately after the initial RVR intervention. Their first-event macros are mostly conservative-looking (`STOP_BEFORE_CONFLICT` 5/9, `YIELD` 2/9), so the failure is not reducible to “RVR chose aggressive acceleration”. The closed-loop state has been driven into a later support collapse.

**Key conclusion:** v16.8.29 maximizes a property of the current open-loop candidate. It does not evaluate whether the *executed one-step action preserves a viable option set at the successor replanning state*. Thus the name “Recursive Viability Recovery” overstates what the implemented statistic actually measures. Safe-prefix is useful evidence, but not a sufficient statistic of closed-loop recoverability.

---

## What is frozen after v16.8.29

The following layers remain protected from algorithm changes in v16.8.30:

- current compact-5k dataset contract / split;
- natural-root construction and natural option basis;
- same-root RCOT;
- BCOT structured protected-priority certificate;
- protected-priority hard feasibility semantics;
- certificate-compatible set-preservation frontier (CTU is a negative ablation);
- learned outcome head as diagnostic only (fallback-outcome is a clean negative);
- 8 s causal conventional-safe definition and its roadgraph/collision decomposition;
- no-valid bounded execution fallback and conventional-audit integrity invariants.

The main unsolved collision-side layer is **closed-loop physical option preservation after uncertified recovery actions**.

The accepted-path kinematics issue remains a separate secondary bottleneck: in the clean COWP exact-200 baseline, 16/25 first kinematics events occur on `accepted_priority_ncf`, and 17/25 immediately preceding candidates are conventional-safe. v16.8.30 does not mix that issue into the recovery experiment.

---

## New diagnostic branch: `cowp_rvr_pareto_guard`

This is intentionally **not a paper contribution**.

When the conventional set is empty, compute:

- the original COWP fallback candidate;
- the v16.8.29 RVR candidate.

RVR is allowed to replace COWP only if:

1. it has a strictly longer current collision-safe prefix; and
2. it is no worse than the COWP fallback on every already-available recovery signal:
   - transport UCB;
   - rule decision risk;
   - action decision risk;
   - pressure decision risk.

This is a parameter-free Pareto no-regret probe. Its only purpose is to test whether v16.8.29's induced failures were primarily caused by letting prefix lexicography override existing execution/rule/interaction evidence.

If this branch fixes the induced failures while retaining most rescues, the next paper mechanism should be a **set/dominance recovery frontier**, not a learned successor module.

---

## Primary branch: `cowp_successor_option_viability`

### Motivation

The correct receding-horizon question is not:

> Which failed open-loop candidate stays collision-free longest now?

It is:

> After the action that will actually be executed, which choice leaves a better feasible option set at the next replanning state?

### Counterfactual successor construction

Only when COWP fallback and RVR propose different actions:

1. use the existing jerk/acceleration/yaw-rate-limited **emitted one-step target**, not the raw candidate waypoint;
2. propagate other valid agents one causal step using the same constant-velocity assumption already used by the conventional collision screen;
3. regenerate the unchanged physical online proposal bank at that successor state;
4. evaluate the successor option set without invoking RCOT/BCOT or relabeling anything safe.

### Parameter-free successor signature

Compare COWP successor vs RVR successor lexicographically by:

1. existence of any full conventional option;
2. number of distinct conventional macro types;
3. number of conventional candidates;
4. best collision-safe prefix among drivable valid successor candidates.

Switch to RVR **only when its successor signature strictly dominates** the COWP fallback successor. Exact ties keep COWP.

This is a deliberately small two-counterfactual probe rather than K-way lookahead, both for attribution and runtime. If it succeeds, a later version can replace expensive online regeneration with a learned successor-viability estimator after the target is proven.

### Scientific status

This implementation is a **model-relative successor-option probe**, not a formal invariant-set or recursive-feasibility guarantee. The other-agent successor uses the current causal CV model. Publication-level promotion would require stronger reactive-agent validation and a formalized definition of the physical successor option set.

---

## Why this direction is more paper-like than another fallback penalty

Recursive feasibility, backup plans, predictive safety filters and trajectory repair already exist in prior work. Therefore “look one step ahead” or “maximize viability” alone is not sufficient novelty.

The CCF-A-level hypothesis to test is instead an **orthogonal option-set feasibility architecture**:

- **social option preservation:** protected-priority same-root non-coercive feasibility (RCOT/BCOT);
- **physical option preservation:** action-conditioned successor recoverability of the ego's future feasible option set.

The scientific contribution would be the explicit separation and composition of these two different feasibility notions, rather than another scalar mixture of social cost, collision risk and utility.

v16.8.30 does not claim this contribution is proven. It builds the minimum mechanism probe needed to decide whether the second axis is real.

---

## Faster experimental protocol

The old dev64 is retired as the only promotion gate because it missed all nine v16.8.29 induced collisions.

### Stage A — equivalence16

16 scenes, COWP only. This is a cheap common-path equivalence gate against the bundled v16.8.29 COWP reference. It must pass before interpreting new methods.

### Stage B — counterfactual48

48 development-selected scenes:

- all 10 RVR-rescued collision scenes;
- all 9 RVR-induced collision scenes;
- all 24 shared-collision scenes;
- 5 stable-safe controls spanning zero-conventional exposure.

Run only the two new branches. This directly tests rescue retention vs induced-failure suppression.

### Stage C — balanced dev96

Only if Stage B is favorable. 96 scenes = old dev64 + all 9 induced counterexamples + 23 extra stable controls spanning zero-conventional exposure.

### Stage D — exact200 promotion

Only if Stage C is favorable. Run **only the promoted new method**. The unchanged v16.8.29 exact200 COWP and RVR outputs are bundled as immutable references. This removes the redundant ~53.5 min COWP rerun from each promotion attempt.

The uploaded v16.8.29 wall times were:

- dev64 COWP: 1213 s;
- dev64 RVR: 1239 s;
- exact200 COWP: 3209 s;
- exact200 RVR: 3678 s.

Thus workflow reduction, rather than another Torch-forward optimization, is the immediate speed win. The earlier profiler already localized ~88% of policy time to CPU candidate construction.

The exact200 set is itself now a **development strict set**, not final paper evidence, because it has been repeatedly used for mechanism selection. After algorithm freeze, final claims require a new untouched scenario set plus the paper's paired/multi-seed uncertainty protocol.

---

## Development manifests

- `waymax_v16_8_30_equivalence16_ids.txt`: 16, SHA256 `81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760`;
- `waymax_v16_8_30_counterfactual48_ids.txt`: 48, SHA256 `ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0`;
- `waymax_v16_8_30_balanced_dev96_ids.txt`: 96, SHA256 `8ca509bd1263aec10e31fbd4a4ff2df21ae22b83287efce61b072897de8e7783`;
- existing exact200: 200, SHA256 `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`.

All are development-selected and explicitly not publication test sets.

---

## Regression

- new v16.8.30 + v16.8.29 focused helper tests: **9/9 passed**;
- packaged semantic/integrity `sanity`: **24/24 passed**;
- a full repository run completed during construction with **274 passed / 5 skipped / 8 failed**; the 8 failures matched the historical repository classes (six missing legacy launchers and two stale semantic fingerprints).
- the final pre-delivery re-run was also started, but the current 120 s command window expired around 25% of the suite. The directly re-verified acceptance gates are therefore the **24/24 sanity** and **9/9 focused** suites; the 274/5/8 count is retained as the earlier completed construction-run record rather than presented as a newly completed final re-run.

---

## Next execution order

```bash
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh make_ids
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh analyze_counterfactual48
```

Stop there and inspect the counterexample-retention result. Do **not** automatically run 96 or 200.

Promotion logic:

- if Pareto guard removes most induced failures and retains rescues, study recovery dominance/frontier;
- if successor-option viability clearly outperforms both COWP and guard, promote physical successor option preservation;
- if both successor candidates frequently have empty successor support and neither branch helps, move to structured proposal refinement;
- do not touch accepted-path kinematics until the collision/recovery mechanism is resolved, unless the new branch materially worsens it.
