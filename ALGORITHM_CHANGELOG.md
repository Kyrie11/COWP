# COWP Algorithm Changelog — v16.8.31

## Scope

v16.8.31 is a **temporal physical-option attribution release** built from the reliable v16.8.30 counterfactual48 evidence. It does not retrain the checkpoint, rebuild the compact-5k dataset, change natural roots, RCOT, BCOT, protected-priority certificate semantics, certificate-compatible set-preservation, the learned outcome head, the 8 s conventional-safety contract, candidate families, or the no-valid execution repair.

The release does two things only:

1. refines the v16.8.30 physical successor-option probe into a parameter-free **Bi-Horizon Option Viability (BHOV)** acceptance relation;
2. adds a deliberately narrow **successor-restoration-only** diagnostic to determine whether the rich successor signature is reliable beyond the binary event “a full conventional option is restored”.

Both methods remain confined to the existing state `full conventional set == empty && valid candidate exists`. All certified and conventional-safe paths are unchanged COWP.

---

## v16.8.30 result integrity: accepted for algorithm attribution

The uploaded v16.8.30 results pass the reliability gate.

- `sanity`: all v16.8.30 semantic/integrity tests pass.
- equivalence16: two disjoint 8-scene shards, exact 16-ID manifest, and **1120 compared fields / 0 mismatches** against the immutable v16.8.29 COWP reference.
- counterfactual48: both new methods use disjoint 24+24 shards whose union equals the exact 48-ID manifest.
- merged standard metrics recompute exactly from the 48 scenario rows.
- the new branches preserve the causal logged-replay contract: other-agent counterfactual futures use the same constant-velocity causal model unless an explicit oracle is requested.
- code review finds no recurrence of the v16.8.27 conventional-safety bypass or the v16.8.28 zero-padding execution bug.

One non-blocking provenance defect was found: v16.8.30's filtered v16.8.29 subset-reference JSONs contain the correct subset rows, but their top-level `scenario_ids_sha256` remained the exact-200 hash. The v16.8.30 analyzer pairs by the actual per-row scenario IDs, so this does **not** change any algorithm result or paired attribution. v16.8.31 rewrites every subset hash and adds explicit source/subset provenance metadata.

Machine audit: `V16_8_30_RESULT_INTEGRITY_AND_MECHANISM_AUDIT.json`.

---

## v16.8.30 preregistered decision: Pareto guard is negative; successor-option signal survives

The counterfactual48 panel is development-selected and therefore **not paper evidence**. It contains the known v16.8.29 counterexamples by design: 10 RVR rescues, 9 RVR-induced collisions, 24 shared collisions and 5 stable controls. Its purpose is mechanistic discrimination.

### Baselines on the targeted 48 scenes

COWP:

- Collision / CR = **34/48 = 70.83%**;
- Kinematics = **6/48 = 12.5%**;
- EP = **1.00251**.

Unconditional RVR:

- Collision = **33/48 = 68.75%**;
- 10 rescues / 9 induced / McNemar exact `p=1.0`;
- Kinematics = **9/48 = 18.75%**;
- EP = **0.82362**;
- paired EP delta vs COWP = **-0.17889**, bootstrap 95% CI approximately **[-0.3811, -0.0248]**.

This reconfirms that current max-prefix RVR is not promotable.

### `cowp_rvr_pareto_guard`: clean negative

The strict no-regret guard switches on **0%** of policy steps and is numerically identical to COWP on every reported closed-loop metric.

At zero-conventional states, the RVR alternative has a mean current-prefix gain of about **+2.64 steps**, but also mean deltas of approximately:

- action risk `+0.0820`;
- rule risk `+0.1376`;
- pressure risk `+0.0024`.

Therefore the old recovery-risk signals and the current-prefix improvement are structurally in conflict. Requiring every old scalar signal to be non-worse eliminates the entire useful recovery intervention set.

**Disposition:** archive strict Pareto no-regret as a negative diagnostic. Do not convert it into a weighted risk mixture or tune tolerances until it switches.

### `cowp_successor_option_viability`: positive mechanism signal, not promotion

The successor-option branch probes about **14.35%** of policy steps but changes the executed recovery action on only **1.93%** of steps.

On the targeted 48 scenes it yields:

- Collision = **33/48 = 68.75%**;
- 2 COWP-collision rescues / 1 induced collision / McNemar `p=1.0`;
- Offroad unchanged at **1/48**;
- Kinematics unchanged at **6/48**;
- EP = **0.99863**;
- paired EP delta = **-0.00388**, bootstrap 95% CI approximately **[-0.0164, +0.00925]**.

Relative to the nine known harmful RVR counterexamples it avoids **8/9** induced collisions; relative to the ten known RVR rescues it retains **2/10**.

The correct scientific interpretation is **high-precision / low-recall evidence** for action-conditioned successor option preservation. The implementation itself is not promoted: the panel is result-selected, the net collision gain is only one scene and statistically non-significant, and strict successor improvement discards eight known recoverable RVR cases.

---

## Refined dominant bottleneck after v16.8.30

The collision-side bottleneck is no longer adequately described as generic fallback failure or generic proposal shortage.

The accumulated evidence is now:

1. v16.8.28 localizes first collision to zero-conventional states, not conventional-safe fallback or accepted priority-NCF;
2. v16.8.29 shows zero-conventional collapse is overwhelmingly dynamic/collision-screen-side, not roadgraph-side;
3. v16.8.29 max current prefix rescues real scenes but also creates almost as many delayed failures;
4. v16.8.30 successor-option gating suppresses most known harmful RVR interventions with negligible EP/kinematics cost, proving that action-conditioned next-state option support contains useful information absent from current prefix;
5. v16.8.30 retains only 2/10 known RVR rescues, showing that “strictly improve next-state signature” is too conservative and/or a single successor state is too myopic.

The active bottleneck is therefore:

> **temporal physical option preservation under uncertified recovery: decide when an action may trade immediate local survival for preservation/restoration of future executable choices, without driving the closed loop into a later option-set collapse.**

This is an acceptance/temporal-viability problem before it is a K-way proposal-expansion problem: the existing RVR candidate is already known to rescue all ten old rescue scenes under sustained RVR behavior.

Proposal support remains the long-term global ceiling, and accepted-path execution kinematics remains a separate secondary bottleneck, but neither should be mixed into v16.8.31.

---

## Frozen / prohibited directions

The changelog now contains enough negative evidence to prohibit the following in this release:

- no RCOT/BCOT budget or threshold retuning;
- no replacement of the certificate-compatible set-preservation frontier (CTU is a negative ablation);
- no learned-outcome fallback weighting or hard physical shield (clean fallback-outcome negative and weak low-FPR reliability);
- no prefix weight tuning and no shortening the unchanged 8 s conventional-safety horizon to make RVR look better;
- no strict Pareto tolerance tuning after its zero-switch result;
- no route/Frenet/roadgraph proposal redesign while `roadgraph_empty` is negligible relative to `collision_empty`;
- no new proposal primitives, dataset rebuild or retraining in this release;
- no accepted-path kinematics mechanism in the same experiment;
- no treating exact200 or any v16.8.30/31 development panel as final paper evidence.

---

## New main branch: `cowp_bihorizon_option_viability`

### Hypothesis

v16.8.30 required **strict successor-signature improvement** before accepting the RVR alternative. That is a high-precision gate but may reject a useful action whenever the next-state option set is merely *non-worse* while the current collision-safe horizon is materially better.

BHOV therefore treats physical recovery as a two-horizon set-preservation relation rather than a one-statistic argmax.

For the same controlled pair:

- `base`: original COWP least-coercive-valid fallback;
- `alternative`: unchanged v16.8.29 max-prefix RVR candidate;

compute:

- current causal collision-safe prefix `H0`;
- v16.8.30 successor option-set signature `V1` using the actual jerk/yaw-rate-limited emitted one-step action.

Accept the RVR alternative iff:

1. `V1_alt >=lex V1_base`;
2. `H0_alt >= H0_base`;
3. at least one of the two inequalities is strict.

This is a **product partial order across horizons**, not a weighted scalar cost. Any successor-option regression blocks a switch even if the current prefix is much longer. A successor tie may now admit a strict current-prefix gain. Exact ties retain COWP.

Because the RVR candidate is already the max-prefix alternative, this release changes only the acceptance relation. It does not introduce a new proposal or a new risk weight.

### Scientific role

BHOV is the minimum probe of whether physical feasibility should be defined as **non-regression of future option support plus improvement of current survival**, rather than strict one-step successor improvement.

It should not yet be described as a formal recursive-feasibility guarantee. The successor is still model-relative under the current causal CV assumption.

---

## New diagnostic branch: `cowp_successor_restore_only`

This branch switches only when the RVR action changes successor `conventional_exists` from 0 to 1.

Its purpose is to isolate whether the lower-order components of the rich successor signature — macro diversity, candidate count and prefix — are too noisy under the one-step counterfactual model.

It is not a paper contribution and should not be promoted by itself.

Interpretation:

- if restoration-only is much more reliable than BHOV/SOV, successor **existence** is robust but richness statistics need a better representation;
- if BHOV dominates restoration-only, the option-set richness ordering contains useful information beyond binary restoration;
- if neither helps on outcome-blind scenes, one-step successor viability is insufficient and the next mechanism should move to temporally persistent/multi-horizon viability rather than tune scalar weights.

---

## Development protocol and selection-bias repair

### Stage 0 — common-path equivalence16

Run COWP only. It must reproduce the immutable v16.8.29 reference exactly before interpreting new methods.

### Stage 1 — counterfactual48

Run BHOV and restoration-only on the existing result-selected 48-scene panel. This is a **mechanism sanity test only**, not promotion evidence.

Primary gate: BHOV should improve the v16.8.30 SOV rescue/harm tradeoff — retain more than 2/10 old RVR rescues without materially giving back the 8/9 induced failures that SOV avoided, and without new kinematics/EP regression.

### Stage 2 — panel-disjoint outcome-blind development64

Only if Stage 1 is favorable.

The new 64-scene panel is selected by deterministic SHA256 ranking from the exact200 IDs **after excluding the union of every v16.8.30 equivalence16, counterfactual48 and balanced96 scene**. Selection uses no outcome label.

- remaining source pool: 101 IDs;
- selected: 64;
- overlap with all v16.8.30 panels: **0**;
- logical SHA256: `becdc8430e14bd76190e3446206bed8e7cb9afb966290978e9bdaa61a5202e79`.

This is stronger development evidence than another result-selected counterexample panel, but it is still **not publication evidence** because the enclosing exact200 set has already been repeatedly inspected.

### Stage 3 — exact200 development confirmation

Only if the 64-scene panel is non-harmful and directionally favorable. Run only the promoted new method; reuse immutable exact200 COWP/RVR references.

After the algorithm is frozen, final publication claims require a new untouched evaluation set, paired/multi-seed uncertainty and stronger reactive-agent/human-audited stress evidence for the causal social-burden claim.

---

## CCF-A mainline interpretation

Generic recursive feasibility, backup-plan MPC and safety filters are mature prior-art families. Therefore BHOV or one-step lookahead alone is not the novelty claim.

The paper-level hypothesis remains **Orthogonal Option-Set Feasibility**:

- **social option-set feasibility:** same-root protected-priority RCOT/BCOT asks whether ego safety preserves other agents' natural low-burden options;
- **physical option-set feasibility:** action-conditioned temporal viability asks whether an uncertified ego recovery action preserves/restores the ego planner's own future executable option set.

The unifying object is a feasible **set of options**, but the two axes are deliberately not collapsed into one scalar. v16.8.31 tests whether the physical half can be given a stable two-horizon dominance relation.

---

## Engineering/provenance changes

- corrected stale subset `scenario_ids_sha256` metadata in all filtered immutable v16.8.29 references;
- added `reference_subset_provenance` with exact source hash and subset hash;
- bundled immutable v16.8.30 counterfactual48 SOV/guard outputs;
- added the panel-disjoint outcome-blind development64 manifest and COWP/RVR subset references;
- added `84_analyze_bihorizon_option_viability.py` for paired event/EP and old-RVR counterexample-retention analysis;
- added regression tests for BHOV dominance, restoration-only semantics, method gate defaults, subset hashes and development-panel disjointness.

---

## Regression status

- packaged v16.8.31 focused semantic/integrity `sanity`: **29 passed** after the final provenance tests were added;
- direct v16.8.31 helper/provenance tests: **5 passed**;
- a full repository run was attempted but not completed within the command window. The first observed failure is the same historical missing-launcher class (`NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh` absent), reached after **124 passed / 5 skipped**. No new v16.8.31 functional failure was observed before that historical failure.

---

## Next execution order

```bash
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh make_ids
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh analyze_counterfactual48
```

Stop after counterfactual48. Do **not** automatically run the 64-scene panel or exact200.

Only if BHOV improves the rescue/harm tradeoff, run:

```bash
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh holdout64_parallel2
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh analyze_holdout64
```

Only if the panel-disjoint 64-scene result is non-harmful and favorable should exact200 be used as a final development confirmation.


---

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
- full repository: **274 passed / 5 skipped / 8 failed**.

The 8 failures are the same historical repository issues already present before v16.8.30: six tests reference legacy launcher scripts absent from the supplied archive, and two tests hard-code an old semantic fingerprint. No new functional regression was introduced.

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


---

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


---

# COWP Algorithm Changelog — v16.8.27

## Scope

v16.8.27 is a **strict-Waymax semantic-integrity repair**. It does not change the trained checkpoint, dataset, cache, natural-root construction, RCOT/BCOT certificate, proposal-family definitions, learned heads, or paper-level mechanism. No algorithm promotion is authorized from the v16.8.26 strict physical attribution until the repaired exact-ID evaluation is rerun.

## Triggering evidence from v16.8.26

The v16.8.26 learned-offline diagnostics completed normally and remain usable. The outcome head has stable but modest physical discrimination (held-out collision/offroad/unsafe-union AUPRC about 0.598/0.436/0.733), while the RCOT/BCOT mechanism remains substantially stronger for its intended false-safe task.

The exact 200-ID Waymax runs also completed, and the physical events themselves are real simulator outcomes. However, first-event localization showed that 44/45 COWP collision episodes had a fallback action immediately before first overlap. Code review then found a semantic-integrity bug in the online candidate generator that directly contaminates this attribution path.

## Critical bug: NEUTRAL_EGO was falsely promoted to conventional-safe

In v16.8.26, `_add_candidate(..., conventional_check=False)` was used for the reserved and final `NEUTRAL_EGO` smooth-stop candidates. `_add_candidate` initialized `conv=True` and skipped both the roadgraph drivable screen and causal constant-velocity collision screen when this flag was false. The resulting boolean was then written to `cowp/candidates/conventional_safe`.

Consequently a neutral/smooth-stop candidate could enter the `no_certificate_use_least_coercive_conventional` fallback pool **without ever being conventionally safety-screened**. Smooth stopping is not automatically safe: it may leave the drivable corridor or create/retain a collision with a close rear, crossing, or merging actor.

This is an implementation/logic error, not an algorithmic result. It invalidates the v16.8.26 claim that the observed first-event concentration proves the fallback objective itself is the dominant physical bottleneck. The Waymax CR/collision/offroad/kinematics numbers are still factual outcomes for that buggy controller, but the fallback-mechanism attribution must be rerun after repair.

## Repair 1: no conventional-safety bypass exists anymore

The `conventional_check` argument is removed from `_add_candidate` entirely. Every generated candidate, including `NEUTRAL_EGO`, is always evaluated by:

1. the same kinematic validity check;
2. the same causal roadgraph drivable screen; and
3. the same causal constant-velocity collision screen.

A candidate that fails the conventional screen is **not deleted**: it remains a dynamically valid last-resort candidate, so emergency fallback coverage is preserved. It simply cannot masquerade as `conventional_safe`.

This establishes the invariant:

> membership in the conventional-safe fallback pool always implies that the online conventional screen actually ran and passed.

A runtime integrity assertion now aborts if `no_certificate_use_least_coercive_conventional` ever selects a candidate whose `conventional_safe` flag is false.

## Repair 2: selected-candidate semantic provenance

Every online policy diagnostic now records:
- selected macro id/name;
- selected-valid flag;
- selected-conventional-safe flag;
- fallback reason.

The compact episode diagnostics additionally record, for the action immediately before the first collision/offroad/kinematics event:
- whether the action was fallback;
- whether the selected candidate was truly conventional-safe;
- selected macro;
- exact fallback reason.

The physical comparison script reports these distributions. This is required before making any fallback-side mechanism claim.

## Repair 3: outcome-head metadata bug

The learned-offline multi-method result writer used a stale `method_name` variable while annotating `OutcomeHead/UsedForSelection` and `OutcomeHead/SelectionScope`. When `cowp_fallback_outcome` was the last method, the ordinary `cowp` row could be mislabeled as if the outcome head had participated in selection.

The selection implementation itself was not affected: ordinary COWP still used zero outcome risk when `--outcome-risk-penalty 0`. v16.8.27 makes the metadata method-local through `_outcome_head_selection_metadata()`.

## Repair 4: fine-grained policy runtime profiler

The v16.8.26 outer profiler shows that Waymax `env.step` is not the dominant cost; most wall time is inside the COWP policy. v16.8.27 adds an opt-in `--profile-policy-runtime` profiler that separates:
- state/map extraction;
- CPU online candidate construction;
- host-to-device transfer;
- model forward;
- selection/certificate logic;
- action projection.

With `--profile-policy-sync`, PyTorch CUDA synchronization is used around these fine-grained sections for profiling accuracy; `--profile-waymax-sync` remains the outer Waymax timer. This mode is diagnostic only and should not be used for publication timing.

## Scientific disposition

The following v16.8.26 conclusions remain valid because they come from learned-offline caches and are independent of the online conventional-safety bug:
- CTU remains rejected as a replacement for the certificate-compatible frontier;
- RCOT/BCOT remains the strongest learned mechanism signal;
- outcome head remains diagnostic-quality, not a hard certificate;
- fixed-bank proposal sufficiency remains a global ceiling in the existing cached dataset.

The following conclusion is **withdrawn pending rerun**:
- “strict physical failure is dominantly caused by the fallback objective/recovery policy.”

No new recovery algorithm, proposal family, or certificate component is added in v16.8.27. First restore online semantic integrity, rerun the same exact 200 IDs, then decide the next algorithmic question.

## Regression contract

New focused tests require:
- no `conventional_check` bypass in `_add_candidate` or online neutral generation;
- a neutral candidate that fails the drivable screen is retained as a candidate but marked non-conventional;
- outcome-head selection metadata is method-local;
- first-event diagnostics retain conventional-safe/macro/fallback provenance.

Reconstructed artifact regression (from the byte-identical uploaded v16.8.26 package): **261 passed / 5 skipped / 8 historical failures**. The 8 failures are unchanged repository-history issues (six tests reference launchers absent from the supplied archive; two tests hard-code an old semantic fingerprint). The v16.8.27 integrity + v16.8.26 fallback + Waymax diagnostic + CTU focused set is **13/13 passed**; the packaged `sanity` command is **11/11 passed**.


## v16.8.18 — Sparse-Probe Pipeline Integrity, Location-Aware Scenario Reads, and Semantic-Failure Classification (2026-08-14)

Re-analysis of the uploaded v16.8.17 smoke artifacts shows that the current blocker is **not a newly demonstrated label-semantic or model-support failure**.  The uploaded `fastpath_semantic_equivalence.json` compared zero scenes: all 12 requested reference scene ids were missing, while the no-fastpath candidate side had no missing scenes and there were no tensor/key mismatches.  The uploaded fresh-build profile contains zero terminal rows and `build_fresh.log` ends during the initial full validation-split scan, before any 96-scene completion marker or downstream support audits.  Consequently no `model_support_audit.json`, `base_screen_verdict.json`, or `v16_8_17_smoke_verdict.json` was produced.  `strict` then reported a missing smoke verdict as a downstream consequence of this incomplete upstream pipeline.

v16.8.18 therefore makes **no changes to natural/response/witness/candidate label semantics or scientific thresholds**.  The label-semantic fingerprint remains exactly `adcea5cb927d4c06c7f667725ce1c5b7b62808d6bd2e84244149d01ab25a1fa0`, so the last fully completed v16.8.16 smoke remains the nearest valid scientific evidence: on mechanism-auditable criticals, rootless=0, <2 low-burden roots=0, PRIO/response/transport/witness targets are non-degenerate and FASTPATH A/B was bitwise equal.  v16.8.17's coverage policy has simply not yet been exercised by a complete fresh v16.8.17 smoke build.

Engineering repairs:

1. Added a reusable **Scenario location index** (`scenario_id -> TFRecord file + record_index`). Sparse smoke/strict/train-pilot builds now read only the shards/records containing requested scene ids instead of depending on where targets appear in an interleaved full-split scan.
2. Sparse builds use `--require-all-allowed-resolved` and are immediately checked by a new integrity audit. Requested ids never observed in the profile are classified as an interrupted/unresolved pipeline; resolved filters/errors and corrupt/missing NPZs are reported separately. Model-support audits do not run until this precondition passes.
3. FASTPATH reference selection no longer trusts directory existence. A partial current-version label directory cannot shadow the complete reviewed v16.8.16 reference. Reference and candidate NPZ sets are integrity-checked before comparison.
4. Semantic equivalence now distinguishes `reference_build_incomplete` / `candidate_build_incomplete` from a true `semantic_mismatch`. Missing scenes are explicitly **not** evidence that tensor values changed.
5. Smoke/strict/train-pilot wrappers write stage-aware `*_pipeline_status.json` files on abnormal exit. A missing composite verdict therefore points to the exact upstream pipeline stage rather than being confused with a scientific FAIL verdict.
6. The policy-only smoke re-audit captures scientific audit return codes and still writes a composite verdict. `smoke` now prefers this verified v16.8.16-label re-audit when the reviewed source artifacts are available; `fresh-smoke` is provided when a new label build is actually required.

Local regression after the pipeline changes: **224 passed, 5 skipped**; `compileall` and v16.8.18 smoke/strict/train-pilot/full orchestration shell syntax checks pass.

## v16.8.17 — Evidence-Coverage Promotion Contract and Policy-Only Smoke Re-Audit (2026-08-14)

Fresh re-analysis of the uploaded v16.8.16 96-scene smoke shows that the previous natural-basis collapse is repaired **on the auditable support**: 517/538 selected critical actors are mechanism-auditable, and among those actors rootless=0, <2 natural roots=0, <2 low-burden roots=0, protected PRIO coverage=99.75%, the 32-slot relevant-pair response bank is complete, and all active natural/response/witness/transport target families are non-degenerate. Proposal/causal smoke also passes and the exact fast-path A/B is bitwise-equivalent. The sole composite blocker is evidence coverage: 21/538 selected critical actors (3.90%) have an unknown mechanism target, causing only 80/96 scenes (83.33%) to carry a complete candidate-level certificate.

The 21 unknown targets are not treated as failed roots. Profile diagnostics split them into 11 actors with >=60 valid future samples but no full routable lane and no non-degenerate factual route geometry, plus 10 actors with only 17--56 valid future samples. Manufacturing six counterfactual roots from a stationary/degenerate single factual path, or extrapolating an incomplete future, would create unsupported option-space evidence. These actors therefore remain selected criticals for physical safety but are explicitly masked from mechanism/certificate supervision.

The previous coverage gates were internally inconsistent with the active six-critical scene structure. At the uploaded smoke's mean 5.60 selected criticals/scene, a 1% per-critical unknown rate corresponds to only about 94.5% all-critical scene completeness under an independence reference, while the old strict gate simultaneously required 98% complete scenes (implicitly about 0.36% per-critical unknowns). v16.8.17 replaces that contradictory pair with an explicit evidence-support contract: >=95% per-critical mechanism coverage and >=75% complete candidate-certificate scenes. The latter is the same order as 0.95^6=73.5% for the configured maximum six criticals; it is a planner-supervision support floor, not a causal-claim metric. Auditable rootless and <2-low-burden counts remain zero-tolerance.

Small-smoke coverage uses a Wilson gross-failure screen; strict/train-pilot use point estimates. A new hard-vs-random missingness-bias audit prevents promotion by disproportionately masking difficult scenes (smoke caps: 3 percentage points for per-critical unknown-rate gap and 10 points for complete-scene gap; strict/pilot: 3 and 8 points). The uploaded smoke has only ~0.88 pp and ~4.17 pp gaps respectively, so its unknown labels are not strongly concentrated in the hard stratum.

No label-producing file is changed in v16.8.17. A separate label-semantic fingerprint therefore allows the existing reviewed v16.8.16 smoke NPZ labels to be **policy re-audited without rebuilding them**. Full code/provenance fingerprints remain required for newly built strict, train-pilot and full caches. Full-core label/tensor audits use the same 95%/75% point coverage contract.

Local regression: **220 passed, 5 skipped**; compileall and all v16.8.17 promotion shell syntax checks pass.

## v16.8.16 — Full-Horizon Auditability, WOMD Driveway Evidence, Exact OBS Geometry, and Statistical Smoke Promotion (2026-08-14)

The v16.8.15 96-scene smoke passed training supervision but was blocked by two independent gates: a borderline PBTR point estimate (23/45=0.5111 versus the smoke maximum 0.50) and genuine natural-basis incompleteness (13/536 auditable critical agents rootless and 18/423 protected agents without a PRIO root).  All 13 rootless cases used `logged_geometry_neutral_timing` and were map-rejection dominated.

v16.8.16 fixes a construction inconsistency where any non-empty lane polyline disabled empirical fallback even when that lane was too short to retime for the full 8 s horizon.  Auditability and empirical-fallback eligibility now use the actually retimable `map_refs`, so short 35/36-step actors without a full route become explicit mechanism-unknown targets while 78--80-step factual geometry can provide a narrow empirical route witness.  Canonical OBS now preserves WOMD factual positions instead of reintegrating logged velocity.  The Scenario parser also retains official WOMD driveway polygons and map compliance accepts the union of the existing lane corridor and explicit driveway polygons without inferring drivable area from road edges or relaxing lane thresholds.

Smoke promotion is now uncertainty-aware: the 96-scene probe uses Wilson intervals to reject only gross proposal failures, while the 1200-scene strict probe keeps the preregistered point-estimate thresholds unchanged.  Smoke certificate-complete coverage is 95% (strict/train 98%).  PRIO is treated as a typed-source coverage requirement rather than a mathematically unjustified 100% per-protected-agent existential requirement: 95% in smoke and 98% in strict/train, while dataset-wide source support and priority-preservation correctness remain hard-gated.  Auditable rootless and <2-low-burden counts remain zero-tolerance.

Local regression: **215 passed, 5 skipped**; compileall and all v16.8.16 promotion shell syntax checks pass.

## v16.8.12 — Continuous Map Geometry, Neutral-Timing Natural Reference, and Fail-Safe Promotion Verdicts (2026-08-13)

### Triggering evidence from the uploaded v16.8.11 train pilot

The 400-hard + 800-random training pilot completed all 1,200 requested label scenes and passed the manifest and training-supervision audits, but model support remained invalid.  Across 6,611 critical vehicles, 248 (3.751%) had zero natural roots and 354 (5.355%) had fewer than two low-burden natural roots.  Every other active model-support family passed, including full 32-slot relevant-pair response coverage, candidate NCF/false-safe balance, relevance/witness/pair-NCF balance, safe/unsafe and low/high-burden responses, affected-root recovery, protected-candidate feasibility, all natural/response sources, and continuous witness/burden variability.  Re-analysis of the build profile shows that the zero-root agents split into 148 map-dominated and 100 priority-dominated failures; 230/248 used the full logged future as their normative natural reference.

The missing `v16_8_11_train_pilot_verdict.json` was a separate wrapper-control bug: `65_audit_model_support --strict` correctly returned non-zero, but `set -e` terminated the shell before causal/fresh-ceiling diagnostics and the final verdict writer ran.  The pilot therefore failed scientifically, but the wrapper reported that failure as a missing artifact.

### Construction repairs

1. Replaced natural-root map compliance based on distance to discrete lane polyline **sample points** by exact continuous point-to-lane-segment distance.  The physical thresholds (`map_max_distance_*`, compliant fraction and hard maximum) are unchanged.  A trajectory lying on a long/sparse lane segment is no longer declared off-map simply because it is far from the segment endpoints/samples.
2. Raw logged future timing is no longer the normative `natural_ref`, even when all 80 future samples are valid.  Logged future remains the OBS empirical branch and contamination evidence.  NEU/PRIO burden and priority preservation use a timing-neutral reference: lane-topology route geometry first, logged geometry retimed from the current state second, jerk-bounded straight motion only as the final map-fragment fallback.
3. Primary NEU and PRIO roots now use the same lane-topology route bank before straight motion.  Lane-graph search is performed once per critical actor at the maximum required route length and the resulting route polylines are retimed for each acceleration/speed intervention, avoiding repeated graph searches without reducing root/candidate budgets.
4. Route matching against WOMD future geometry now evaluates only the original `valid=1` timestamps.  It no longer treats `sum(valid)` as a contiguous prefix or lets hold-padding choose a route when validity has internal gaps/late appearance.
5. Hard priority progress uses the same natural-reference projection geometry as the burden functional, and the builder records explicit priority rejection reasons (`max_decel`, `max_accel`, `max_jerk`, `progress_loss`, `gap_loss`).  No priority tolerance is relaxed.
6. Natural diagnostics now report rootless/<2-low-burden failures by priority relation, reference kind and dominant rejection family, plus priority rejection mechanisms and map-distance/burden summaries.  This makes a failed smoke directly actionable.

### Promotion-control repair

`smoke`, `strict`, and `train-pilot` now distinguish **semantic gate failure** from pipeline/runtime failure.  Strict supervision/model-support commands may return non-zero as designed, but the wrappers continue far enough to write the natural diagnostics and a composite PASS/FAIL verdict.  `full-core` therefore sees an explicit `recommend_full_rebuild=false` rather than a misleading missing-verdict error.  The master chain also enforces same-fingerprint promotion: smoke must authorize strict, strict must authorize train-pilot, and both strict + train-pilot must authorize full-core.

### Performance disposition

The v16.8.11 train pilot averaged 267.95 s/scene.  Safe responses (143.66 s) and witness construction (71.38 s) together account for about 80.3% of label-engine time; natural construction is 7.38 s/scene and pair-neutral construction 4.07 s/scene.  v16.8.12 therefore preserves the 32-response bank and root/candidate semantics.  The natural-stage optimization is deterministic route-bank reuse, not support reduction.  Full worker selection remains host-specific and must be based on the new pilot profile after semantic gates pass.

### Local validation

- Added continuous sparse-lane segment-compliance regression.
- Added regression proving an 80-step logged future remains OBS evidence but no longer becomes the normative natural timing reference.
- Added priority-rejection mechanism diagnostics regression.
- v16.8.11 support regressions remain passing after the reference-policy update.
- Full repository regression: **194 passed, 5 skipped** (one upstream PyTorch nested-tensor warning only).
- Shell syntax checks pass for all v16.8.12 promotion entry points.

## v16.8.11 — Pair-Specific Natural Basis, Jerk-Compatible Root Support, and Train-Split Promotion Gate (2026-08-11)

### Triggering evidence from the uploaded v16.8.10 smoke/strict artifacts

The fresh 400-hard + 800-random validation strict probe already passes every proposal gate: representative `AnyNCF=0.4675`, false-safe lower bound `0.4825`, PBTR lower bound `0.32715`, and hard-scene NCF recovery `0.3025`.  The wrapper stops only at `model_support`: 1,089 vehicle critical-agent instances have no natural root, 1,090 have no low-burden natural root, and 1,147 have fewer than two low-burden roots.  Response, witness, affected-root, candidate and source-supervision targets are otherwise non-degenerate.  Therefore v16.8.11 does **not** spend more proposal budget to repair a natural-basis support failure.

### Root causes

1. Fresh label construction used one global ego-neutral trajectory for all critical actors.  This violates the interaction-specific neutralization contract: a neutral ego motion that removes pressure from a crossing/merge actor can itself pressure a rear vehicle, and vice versa.
2. Natural NEU/PRIO fallbacks were generated with an instantaneous constant-acceleration primitive but filtered with a comfort-jerk test.  The first discrete acceleration transition could exceed the filter jerk bound, deleting otherwise mild non-zero roots and collapsing AGENT_PRIORITY support to zero/one roots.
3. Short/invalid WOMD futures were padded by hold states before natural construction.  Treating such padding as if it were a complete measured future can create invalid observational/reference geometry and suppress diversity.
4. The previous fallback contract triggered on total-root scarcity, not explicitly on **low-burden** root scarcity.  COWP's OPR/NCF semantics require low-burden natural support, so a scene with several valid but high-burden roots was still untrainable.
5. On curved lanes, straight pseudo-roots can be rejected by the map filter.  Critical selection is intentionally independent of future availability, so the correct repair is a map/topology-based pseudo-root path, not dropping critical agents with incomplete futures.

### Algorithm/data-construction changes

1. Added a fixed, proposal-bank-independent **pair-specific ego-neutral bank**.  For each critical actor, the label engine selects a pressure-removing ego intervention lexicographically by physical safety, actor burden and ego-control deviation.  Candidate proposal families cannot change this natural-basis target.
2. Replaced non-zero natural/neutral straight acceleration steps by a configurable jerk-bounded ramp (`natural_accel_ramp_jerk_mps3`, default `1.0`).  Logged-path retiming uses the same ramp.  No comfort threshold was relaxed.
3. OBS roots are now eligible only when the raw WOMD future validity mask has enough measured support (`>=60/80` and `>=0.70` by default).  Incomplete/invalid future rows are not promoted to measured OBS evidence.
4. Added deterministic lane-graph centreline continuation from the current actor state.  It follows WOMD lane topology and is used as the first natural/reference fallback for curved or short tracks.  Logged future geometry (never logged timing) is a secondary offline fallback; jerk-bounded straight motion is last.
5. Natural construction now has an explicit support contract: the existing active total-root target (`min_natural_alternatives=6`) and at least two roots satisfying the original low-burden threshold.  Fallback generation continues until both are met or all candidates fail the **same** map/dynamics/priority/burden/contamination filters.  A support failure is never hidden.
6. Added per-critical diagnostics to the label-build profile: raw future-valid count/fraction, OBS eligibility, reference source, pair-neutral source/safety/burden, fallback attempts/accepts, and map/burden/priority/contamination rejection reasons.
7. Added exact identity-root reuse from causal audit into response-bank and root-recovery/witness hot paths.  The reused tensors are the same exact `unsafe_between`/`compute_burden` results; no candidate, response, root, threshold or label definition is removed.
8. Added a **train-split 400-hard + 800-random pilot gate** before the 22k/5k full rebuild.  A validation strict PASS alone can no longer authorize a costly full rebuild when the training split may have different natural-support properties.
9. Full preparation now emits the new natural-support diagnostic before model-support gating, so a failed full label build has an immediately attributable root-support reason.

### Speed disposition

The uploaded strict profile averages 239.1 s/scene; safe-response generation (109.1 s) and witness construction (91.6 s) dominate.  Candidate generation (~0.74 s) and natural generation (~1.06 s) are not meaningful full-build bottlenecks.  v16.8.11 therefore preserves candidate/root/response budgets and only removes provably duplicate identity-root physical evaluations.  Worker counts still need host-specific throughput benchmarking after the new smoke because repairing rootless actors can increase legitimate response/witness work.

### Promotion rule

Run `preflight -> smoke -> fastpath-ab -> strict -> train-pilot`.  Full core reconstruction is mechanically blocked unless both validation strict and train-pilot verdicts pass under the identical current code fingerprint.  Do not lower `AnyNCF`, false-safe, PBTR, hard-recovery, natural-support, response-bank or causal-integrity thresholds to obtain a PASS.  Attach Waymax outcomes only after the core label/tensor gates pass.

### Local validation

- Added regressions for pair-specific neutral selection, short-future OBS exclusion + lane-route multi-root support, and exact audit-identity reuse.
- Targeted v16.8.11 tests pass.
- v16.8.10 contract/causal/root-transport regression subset passes.
- Full repository regression: **190 passed, 5 skipped** (one upstream PyTorch nested-tensor warning only).

## v16.8.10 — Trainable-Support Contract Repair and Exact Label-Build Fast Path (2026-08-10)

### Triggering evidence from the v16.8.9 400-hard + 800-random strict probe

The strict wrapper stopped at `model_support` even though the causal-integrity and training-supervision audits passed.  The active failure combined (i) critical agents with no/marginal natural-root support and (ii) a degenerate `cowp/response/valid` target: every causally relevant response slot in the probe was occupied, while the model still optimized a non-zero validity BCE and gated SetTransport by the predicted validity probability.  The same probe also remained below the independent hard-scene NCF recovery gate, so no full rebuild is authorized by this change alone.

### Repairs

1. The publication mainline non-coercion certificate is vehicle-only (`critical.vehicle_only_main=true`), matching the manuscript's stated pedestrian/cyclist extension as future work.  Conventional collision safety still checks all logged non-SDC actors; this switch narrows only the coercion certificate, not physical collision avoidance.
2. `response/valid` is now explicitly treated as a fixed-bank occupancy/padding mask rather than a learned semantic class.  Mainline training sets `response_valid_bce=0`, SetTransport uses all decoded response slots, and the support audit requires full slot coverage on every causally relevant pair instead of artificial positive/negative class balance.  The legacy learned gate remains available through `model.use_response_valid_gate=true`.
3. Added an exact `risk_known_zero` burden fast path.  When `unsafe_between()` has already proved a pair safe, the duplicate TTC/RSS burden computation is skipped because both risk masks are necessarily empty under the same thresholds.  The full burden path now reads the same type-aware TTC and configured RSS parameters as `unsafe_between()`.  The optimization is controlled by `engineering.risk_known_zero_fastpath` (default `true`) so a smoke A/B can disable it and prove semantic equivalence.  It is used in causal-audit, safe-budget, response-bank, witness fallback, and root-recovery hot paths without changing label values.
4. Expanded the strict code fingerprint to include the active model/train contract, preventing a pre-v16.8.10 strict verdict from authorizing a post-change full rebuild.
5. Repaired the RSS geometry contract used by both unsafe labeling and burden risk: TTC/near-miss thresholds are now type-aware, longitudinal RSS is laterally gated to the same/merging corridor, and the broad-phase no longer suppresses long-gap high-speed RSS checks. This removes adjacent-lane false RSS blockers and makes the exact safe-pair burden fast path valid under one shared predicate.
6. Repaired the natural-root budget.  The default v16.8.9 nested OBS loop truncated after eight samples before reaching the identity root; v16.8.10 orders the same configured Cartesian family from `(speed_scale=1, time_shift=0, lateral_offset=0)` outward so finite OBS capacity is not spent entirely on one speed slice.
7. Added a curved-lane ego-neutral route-geometry fallback.  It uses logged future **geometry only** as an offline supervision path and regenerates timing from the current state under bounded constant acceleration.  This avoids copying potentially ego-induced logged yielding timing while preventing straight-line neutral/prio fallbacks from being deleted wholesale by the map filter on curved lanes.  Fallback roots must pass the same map, priority and low-burden plausibility checks as primary roots.
8. Strengthened model-support auditing.  The old audit under-counted rootless critical agents whenever an entire scene had no natural roots because the per-agent loop was incorrectly nested under a scene-level `np.any` guard.  The repaired audit scans every critical agent, requires normalized natural mass, and separately requires at least one and at least two **low-burden** roots because OPR/NCF operate on that low-burden natural set rather than arbitrary valid trajectories.
9. Smoke promotion now treats training-supervision and model-support audits as hard gates, so a 96-scene smoke cannot advance to the expensive 1,200-scene strict probe when the learned-label contract is already invalid.
10. Added an exact train/validation scenario-ID leakage check to full preparation.  Planner training also fails fast when `loss_weights.closed_loop > 0` but attached Waymax outcomes are disabled/missing or the sampled outcome set has no safe/unsafe ranking support; an explicit `closed_loop=0` config is required for the corresponding ablation.
11. Added `NEXT_EXECUTION_V16_8_10_CN.sh` to keep WOMD preflight, smoke, exact fast-path A/B, strict probe, full core rebuild, and Waymax outcome attachment as separately gated stages.
12. Added `ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh` with deterministic resume-safe replay sharding across one or more GPUs, exact per-step safety metrics, incremental attachment, verification, and a full-cache safe/unsafe/mixed-scene support gate.
13. Added a regression proving that WOMD train/val future tensors may remain in the cache for offline targets/Waymax replay while the model-facing history adapter is invariant to them and consumes only past + current state.

### Promotion rule

A new 96-scene smoke and a fresh 400-hard + 800-random strict probe are mandatory.  Do not lower the existing `AnyNCF`, false-safe, PBTR, hard-recovery, or causal-integrity thresholds.  Full rebuild remains prohibited until the newly fingerprinted strict verdict sets `recommend_full_rebuild=true`.  The core full rebuild may intentionally defer Waymax replay to avoid wasting GPU rollout time before all label/tensor gates pass, but the current mainline planner (`closed_loop=0.35`) must train from the later outcome-attached caches.

### v16.8.10 local validation

- Full Python unit/regression suite after the dataset-contract, RSS, execution-tooling, and future-input guard repairs: **187 passed, 5 skipped**.
- Added exact fast-path equivalence, fixed-cardinality response-contract, type-aware TTC/RSS, lateral-RSS/broad-phase, identity-first OBS-budget, and curved-route neutral-proxy regressions.
- No full WOMD smoke/strict/full rebuild was executed in the review environment because the raw WOMD shards are not present there; promotion still depends on the user's fresh 96-scene smoke and 400+800 strict probe.

## v16.8.9 — Candidate-Conditioned Causal Audit and Affected-Root Transport (2026-08-07)

### Triggering evidence from the v16.8.8 96-scene smoke

The stable-critical repair itself succeeded: all 96 smoke scenes used `fixed_anchor_v1`, all proposal-union monotonicity checks passed, `AnyValid=1.0`, and there were no build/filter errors.  The remaining failure was highly structured rather than a generic proposal-quality failure.  On the paired 48-scene representative subset, the fresh bank improved protected-priority burden transfer from `0.5814` to `0.4419` and slightly improved conventional-safe coverage (`0.8958 -> 0.9167`), yet universal `AnyNCF` fell from `0.3750` to `0.2083` and the false-safe lower bound rose from `0.5208` to `0.7083`.  PSY was physically valid (`64/64` accepted in the complete smoke profile) and produced protected priority-NCF pairs, but its scene-level `AnyNCF`, false-safe and PBTR increments were exactly zero.  Removing RMR/PSY/legacy timing changed candidate counts but not the 96-scene scene-level ceiling.

This combination shows that candidate generation was no longer the sole bottleneck.  A candidate could solve one protected pair but remain universally non-NCF because every globally critical agent was still audited, even when that particular ego candidate did not causally perturb the agent's low-burden natural options.  The old data contract also represented only geometric `mode_conflict`; a collision-free natural root that was forced above the adaptive burden budget could reduce OPR/tail feasibility without a corresponding transport-support label.  These cases created the possibility of a non-NCF "silent blocker" with no coherent relevance/affected-root supervision for the model.

### Data/certificate semantic repair

1. Added a candidate-conditioned causal-audit layer with one shared floor-smoothed canonical natural-root probability measure.  For every `(candidate, critical-agent, natural-root)` tuple the cache now records whether the root is geometrically unsafe, its direct burden under the ego candidate, and whether it is **affected** (`unsafe OR direct_burden > beta`).
2. Added `cowp/audit/pair_relevant` and `cowp/audit/relevance_mass`.  A globally critical agent is audited for a candidate only when enough neutral low-burden natural-root mass is causally affected.  The support threshold reuses the existing 0.10 witness-support semantics; it is not a relaxed NCF/burden threshold.
3. Irrelevant global-critical pairs are vacuously non-coercive for that intervention (`OPR=1`, no blocker, no response search).  Relevant pairs retain the original burden/OPR thresholds.  Every relevant non-NCF pair is required to have a witness; schema/integrity gates reject silent blockers.
4. Generalized RootTransport support from `mode_conflict` to `mode_affected`.  Burden-only affected roots now receive same-root recovery, root minimum safe burden, tail-CVaR and OPR supervision exactly like unsafe roots.  Geometric conflict remains a separately reported physical-safety subset.
5. Safe-response generation skips audit-irrelevant pairs.  This is both semantically correct and targets the measured build bottleneck: v16.8.8 spent the overwhelming majority of label time in safe-response and witness search, not in candidate generation.
6. Added candidate-level audited-pair/blocker counts, pair-level NCF/blocker codes, causal relevance mass, affected-root tensors, and complete inline transport fields to the fresh-cache contract.

### Learned-model alignment

1. `WitnessDecoder` now predicts an explicit pair relevance logit.  Witness/OPR/burden losses operate on relevant pairs while relevance itself is supervised on every valid candidate/global-critical pair.
2. `SetTransportCertificateHead` adds a burden-only affected channel and enforces `P(affected) >= P(conflict)`.  Transport uses `retain + affected * q`, not `retain + conflict * q`.
3. Set-Transport loss supervises affected mass, per-root affectedness, affected-root recovery, root burden consistency and uncertainty under the same canonical root measure used by labels.
4. Planner features use relevance as a causal support gate: irrelevant pair witnesses contribute zero and their OPR contribution is neutral (`1`).
5. `paper_aligned_supervision_batch` was updated so fresh v16.8.9 caches cannot be silently interpreted with the old all-critical/conflict-only certificate.  Legacy caches remain readable only under explicit legacy semantics.

### Real learned ablations

Added forward/loss switches and config variants for two paper-critical learned ablations that use the same rich v16.8.9 dataset:

- `w/o candidate-conditioned causal relevance`: all global-critical pairs are again consumed by the learned witness/transport/planner path and the relevance head/loss is disabled.
- `conflict-only RootTransport`: burden-only affected roots are removed from the learned transport support while geometric conflict supervision is retained.

These are independent retraining experiments, not shared-checkpoint aliases.

### Promotion and rebuild policy

- **Do not full-rebuild from v16.8.8.** The next mandatory step is a fresh 48-hard + 48-random v16.8.9 causal-audit smoke because label semantics changed.
- Smoke additionally requires non-degenerate audit relevance, a measurable burden-only affected-root signal, zero silent/irrelevant blockers, zero responses on irrelevant pairs, exact audit/transport affected-root agreement, stable critical selection and proposal-union monotonicity.
- Only a subsequent 400-hard + 800-random strict probe with `AnyNCF>=0.40`, false-safe floor `<=0.55`, PBTR floor `<=0.45`, hard recovery `>=0.20` and all causal-audit integrity checks may set `recommend_full_rebuild=true`.
- Full rebuild uses the old tensor-cache directories only as scenario-ID allowlists.  All candidate/natural/audit/response/witness/transport labels are rebuilt from WOMD Scenario proto and serialized into self-contained NPZ files.
- The complete validation cache is re-gated before any GPU training.  Failing proposal or audit integrity at this stage is an explicit `DO NOT TRAIN` condition.

### Engineering and validation

- Added scripts `57_diagnose_causal_audit.py`, `58_screen_v16_8_9_causal_audit_probe.py`, `59_gate_fresh_v16_8_9_cache_protocol.py`, and `60_verify_fresh_v16_8_9_cache.py`.
- Added v16.8.9 smoke, strict-probe, self-contained full-build, mechanism, Waymax probe/full and experiment-ablation wrappers.
- Training launchers now accept explicit model/train config sources, allowing genuine independently trained ablations with strict provenance.
- Local Python regression after the complete v16.8.9 change set: **175 passed**; `compileall` and all v16.8.9/core launchers pass `bash -n`.

## v16.8.8 — Stable Critical Universe, Priority-Smooth-Yield, and Monotone Proposal Refinement (2026-08-07)

### Triggering evidence from the recovered v16.8.6 micro probe

The recovered probe is complete: 191 unique requested scenes (64 hard + 128 representative random with one overlap), 191 fresh labels written, zero filtered scenes, zero build errors. On the 128 representative random scenes, the legacy v16.8 bank has `AnyNCF=0.3671875` (47/128), while the v16.8.6 fresh bank has `AnyNCF=0.203125` (26/128). The paired transition is 18 retained old-NCF scenes, 29 lost old-NCF scenes, and 8 newly gained NCF scenes. Best-case false-safe floor worsens from 0.578125 (74/128) to 0.765625 (98/128), despite mean valid candidates increasing by 2.414 per scene.

The protected-priority signal moves in the opposite direction. The old bank has 121 priority-eligible scenes and 50 with at least one priority-NCF option, giving PBTR floor 71/121 = 0.58678. The fresh bank has 123 priority-eligible scenes and 70 with at least one priority-NCF option, giving PBTR floor 53/123 = 0.43089, already below the 0.45 strict PBTR ceiling. This is a real positive signal for protected-priority refinement, but it cannot be attributed to PCHR itself: PCHR is present in only 4/128 scenes, contributes 17 candidates, and contributes zero NCF / zero priority-NCF candidates. RMR contributes 223 candidates, 25 NCF candidates, and has excellent timing consistency (max target-TTA error 0.00884 s).

### Root-cause audit: proposal-dependent critical-agent labels

The dominant data-semantics defect is upstream of PCHR geometry. Prior to v16.8.8, `select_critical_agents()` chose the global top-A critical-agent set by screening against the *entire current ego proposal bank*. Adding RMR/PCHR therefore could change which agents entered the top-8 set. Because COWP's candidate NCF is universal over all selected critical agents, adding an optional proposal could relabel every pre-existing candidate and destroy its NCF status even when that candidate trajectory itself was unchanged. This violates the monotone-union interpretation required by proposal-source ablations and confounds old/new ceiling attribution.

The 128-scene probe exhibits exactly this signature: conventional-safe scene coverage actually rises from approximately 121 to 124 scenes and protected-priority NCF availability rises from approximately 50 to 70 scenes, yet universal NCF collapses from 47 to 26 scenes. The fix must stabilize the certificate universe before another proposal family is judged.

### Algorithm/data changes

1. **Fixed critical-agent reference bank (`fixed_anchor_v1`)**. Global critical-agent selection is now independent of optional proposal families. It screens against a deterministic factual-current-state anchor bank: logged ego reference, keep, canonical accelerate, canonical yield, smooth stop, and both canonical lateral directions. `proposal_bank_legacy` is retained only as a compatibility ablation.
2. **Base-bank preservation**. Offline lane-change/creep actions are inserted before optional RMR/priority refinements. Online generation reserves a compact canonical lane-change subset before optional RMR/PSY expansion. Optional refinements may consume filler capacity but cannot silently erase the neutral root or both lateral escape directions.
3. **PCHR is demoted, not expanded**. `priority_hold_release_enabled=false` by default. The implementation remains available for an explicit ablation, but the recovered probe's 17 candidates / zero NCF result does not justify spending more proposal budget on the same stop-hold-release family.
4. **Priority-Smooth-Yield (PSY)**. A new protected-priority proposal source solves a quintic longitudinal arrival to the protected agent's late boundary plus gap while prescribing a low conflict-entry speed and a negative initial acceleration. A commitment check requires a visible early speed drop. This removes PCHR's full-stop feasibility bottleneck while directly targeting early yielding behavior rather than endpoint timing alone.
5. **Offline/online PSY consistency**. The same smooth terminal-speed family is available in online Waymax proposal generation. The offline generator uses map-derived protected relation and boundary-consistent TTA; the online generator uses the existing causal closest-approach proxy and the same smooth-yield controls.
6. **All-scene proposal diagnostics**. Successful label rows now record attempted, accepted, and rejected candidate counts by proposal source, plus critical selection mode/count. Prior profiling exposed rejection diagnostics only for zero-valid scenes and could not explain why a source had low yield.
7. **Source-level monotonicity audit**. `50_ablate_proposal_sources.py` now includes PSY, and the v16.8.8 screen verifies that adding PSY cannot worsen AnyNCF, false-safe floor, or PBTR floor within one fixed-critical fresh bank.
8. **Paired-transition diagnostics**. The paired comparator reports old/new conventional-safe coverage and NCF retained/gained/lost counts, making certificate-universe drift visible rather than hiding it behind aggregate rates.
9. **Code-lineage hardening**. Smoke/strict verdicts carry the full label/eval code fingerprint. Resume of a partial v16.8.8 label directory is rejected under changed code, and a strict PASS cannot authorize a full rebuild under a different fingerprint.
10. **Paper-grade full-cache gates**. Full build remains self-contained (no fragile transport overlay), recomputes the full validation proposal ceiling and proposal-source ablation, requires AnyValid>=0.99, AnyNCF>=0.40, false-safe floor<=0.55, PBTR floor<=0.45, nonzero PSY protected-priority NCF yield, and proposal-union monotonicity before any model training.

### Build-cost evidence

The completed 191-scene v16.8.6 profile shows candidate generation is not the multi-day bottleneck: candidate generation averages ~0.86 s/scene, while safe-response generation averages ~164.5 s/scene (~57% of label-engine time), witness/certificate construction ~99.7 s/scene (~35%), and critical selection ~21.3 s/scene (~7%). Consequently v16.8.8 does not reduce root/search budgets or weaken labels merely for speed. The decision protocol first runs a 48-hard + 48-random smoke under the corrected semantics, then a 400-hard + 800-random strict probe only after smoke success. A full build is mechanically blocked until the strict verdict authorizes it under the identical code fingerprint.

### Current rebuild decision

**DO NOT FULL-REBUILD v16.8.6/v16.8.7.** The recovered PCHR micro screen is a valid rejection of that data semantics: PCHR itself has no direct NCF yield, and the universal NCF collapse is confounded by proposal-dependent critical-agent selection. Run the v16.8.8 96-scene stable-critical + PSY smoke first. A smoke PASS authorizes only the strict 1,200-scene probe. Only a strict PASS (`recommend_full_rebuild=true`) under the same code fingerprint authorizes the multi-day full rebuild.

### Local validation

- `pytest -q`: **171 passed** (one upstream PyTorch nested-tensor warning only).
- `python -m compileall -q cowp`: pass.
- v16.8.8 smoke / strict / full-build shell entrypoints: `bash -n` pass.
- regressions cover proposal-bank-independent critical selection, PSY endpoint/terminal-speed behavior, and explicit PCHR demotion semantics.


## v16.8.7 — Cache-Lineage Hardening, Self-Contained Fresh Transport, and Rebuild Decision Recovery (2026-08-08)

### Triggering evidence

The v16.8.6 64-hard + 128-random Priority-Commitment micro probe completed the expensive fresh-label stage for **191 unique scenes** (one scene appears in both requested sets). `fresh_probe_profile.jsonl` contains 191 `written` rows and no filtered/error rows. The run then failed only when reading the legacy old-bank reference cache: `tensor_cache_val_waymax_transport_v16_8/10040e572b831a04.npz` could not be opened.

The root cause is cache lineage, not NumPy indexing. Legacy `*_waymax_transport_v16_8` directories are transport **overlay** caches: their visible top-level `*.npz` entries are symlinks to the corresponding `tensor_cache_*_waymax` files, while transport tensors live in a hidden sidecar directory. Deleting `tensor_cache_train_waymax` / `tensor_cache_val_waymax` therefore leaves visible but broken symlinks. The proposal-ceiling and paired-probe scripts do not consume `waymax/*`, so the surviving `tensor_cache_train` / `tensor_cache_val` are the correct legacy raw backing for these diagnostics. The original Waymax attach implementation copies every base tensor unchanged and only appends `waymax/*`; transport augmentation likewise does not consume `waymax/*`.

### Algorithm status

There is **no new paper-level proposal/certificate hypothesis in v16.8.7**. The v16.8.6 BCS-RMR + Priority-Commitment Hold-Release algorithm is retained unchanged. v16.8.7 is a data-lineage, reproducibility, and build-cost revision. This separation is intentional so the existing 191-scene PCHR probe labels remain semantically usable for the pending comparison.

### Engineering/data changes

1. `COWPNpzDataset` now detects broken overlay symlinks at construction and raises an actionable lineage error instead of crashing later on an arbitrary `np.load`.
2. Proposal-probe wrappers now use `tensor_cache_val` as the default old-bank source because proposal ceiling/PBTR metrics do not require transport or cached Waymax outcomes. The paired comparator loads only requested scenario IDs, avoiding thousands of unnecessary NPZ opens.
3. Added `RECOVER_V16_8_6_PRIORITY_MICRO_PROBE_FROM_BASE_CN.sh`, which reuses the already-built 191 fresh labels and reruns only old-bank diagnosis, paired comparison, and the micro screen. It never rebuilds labels.
4. Added `54_rebase_transport_overlay.py` and `REPAIR_LEGACY_V16_8_TRANSPORT_OVERLAYS_CN.sh`. They can salvage existing v16.8 transport sidecars by copying them into a new overlay whose top-level links point to surviving `tensor_cache_train/val`. This is valid for legacy RCOT/BCOT diagnostics because Waymax attach only appended `waymax/*` fields and transport construction does not consume them. The rebased caches intentionally contain no cached Waymax outcomes and remain non-paper-grade v16.8 proposal data.
5. Fresh label files now serialize the complete transport contract inline, including `cowp/transport/root_min_safe_burden` and `cowp/transport/canonical_root_weight`. Canonical root weights are precomputed once per agent rather than once per candidate-agent pair; this is algebraically equivalent.
6. Fresh full rebuilds no longer run post-hoc transport augmentation. `tensor_cache_train/val` are self-contained regular NPZ caches and are used directly as TRAIN_CACHE/VAL_CACHE. This removes a duplicated transport search pass and eliminates overlay-backing fragility.
7. Fresh-cache fingerprints now cover critical-agent selection, natural alternatives, safe responses, witness, burden, and priority code in addition to proposal/geometry code. A change to label semantics can no longer evade the build fingerprint.
8. Added `55_verify_fresh_self_contained_cache.py`: exact scene-set matching, no symlinks, required proposal/witness/transport keys, Waymax-ready state presence, scenario-id consistency, validity, NCF, and proposal-source statistics are checked before training.
9. Full rebuild defaults reuse the surviving `tensor_cache_train/val` **only as scenario-ID allowlists**. No stale COWP label is reused. Fresh BCS-RMR/PCHR labels still come from WOMD Scenario protos.
10. After a full fresh merge, the validation proposal ceiling is recomputed from the actual tensor cache and training is blocked unless AnyValid >= 0.99, AnyNCF >= 0.40, false-safe floor <= 0.55, and PBTR floor <= 0.45.

### Build-performance evidence and policy

The 191-scene micro profile shows that candidate generation itself is cheap. Relative to aggregate `label_engine_s`, `engine_safe_responses_s` accounts for about **57%**, `engine_witness_s` about **35%**, and critical-agent selection about **7%**; together safe response and witness computation dominate more than 90% of fresh label time. Therefore the safe accelerations in this release are: sparse allowlist producer filtering, thread-pool suppression, exact scene-set reuse, no full-train cached Waymax replay, no duplicate transport augmentation, and resumable fingerprints. Do not reduce response/root search budgets merely to make the build faster without a controlled label-equivalence experiment.

### Current rebuild decision

**HOLD.** Do not start a full rebuild until the already-built 191-scene probe is recovered using `tensor_cache_val` and the v16.8.6 micro screen is evaluated. If the micro screen passes, run the strict 400-hard + 800-random probe. Only the strict probe may authorize the expensive full fresh rebuild. If it fails, improve proposal refinement rather than spending days rebuilding an infeasible bank.

### Local validation

- `pytest -q`: **167 passed**.
- `python -m compileall -q cowp`: pass.
- all modified/new v16.8.6/v16.8.7 shell entrypoints: `bash -n` pass.
- rebase/self-contained-cache/probe/gate CLIs: import/`--help` pass.
- regressions cover early broken-symlink detection, transport-sidecar rebasing, and complete inline fresh transport tensors.

## v16.8.6 — Priority-Commitment Proposal Refinement, Legacy-Bank Saturation Audit, and Evidence-Valid Experiment Pipeline (2026-08-07)

### Triggering evidence

This revision was triggered by three independent findings from the uploaded v16.8.5 run and prior v16.8.3 evidence.

1. The uploaded `formal_v16_8_4_bcs_rmr_bcte_proposal_probe.zip` is **not a completed fresh proposal probe archive**. It contains the old-bank `current_proposal_ceiling.json` and the interrupted fresh-label log that stopped after 768 written labels with `No valid ego candidates available for neutral intervention`; it does not contain a completed `paired_proposal_probe.json`. Therefore this archive cannot establish either success or failure of fresh BCS-RMR.
2. The legacy v16.8 validation bank is structurally healthy but proposal-limited: across 5,013 scenes, `AnyNCFSceneRate=0.36246`, best-case false-safe floor `0.53321`, and best-case PBTR floor `0.58179`. The PBTR floor exceeds the current calibration ceiling `0.45`, so no selector restricted to this fixed bank can satisfy the current protected-burden gate.
3. On the same legacy bank, the completed v16.8.3 learned run already showed strong mechanism discrimination: protected NCF recall `0.97894`, pair-witness AUPRC `0.84945`, protected BCOT AUPRC `0.96984`, and protected RootTransport AUPRC `0.88167`. The remaining calibration status was `proposal_infeasible`. This is evidence that further threshold/budget tuning on the same proposal bank is not the highest-leverage direction.

The v16.8.5 label-space ablations also reveal a useful asymmetry. Removing counterfactual reasoning increases FSR from `0.31156` to `0.58896`; removing the neutral branch increases it to `0.39760`; removing the priority branch increases it to `0.34879`. In contrast, removing option preservation leaves FSR unchanged at the reported precision and removing hard witness rejection changes FSR by only about `2.6e-4`. These are oracle/label-space mechanism diagnostics, not learned-model claims, but they justify keeping the causal roots and protected-priority structure while de-emphasizing peripheral hard gates.

### Algorithm change: Priority-Commitment Hold-Release (PCHR / proposal source `PRIORITY_HOLD_RELEASE`)

BCS-RMR fixes *when* ego reaches the conflict boundary, but a protected agent can still be coerced by the *approach profile*: ego may remain assertive until late braking even when final arrival order is pass-after. v16.8.6 therefore adds a proposal family that makes yielding observable early.

For causally protected (`AGENT_PRIORITY` or `EQUAL_OR_NEGOTIATED`) interactions, the candidate generator now:

1. chooses a stop point several metres before the conflict boundary;
2. follows a jerk-smooth quintic segment from the current state to rest at that point;
3. holds before the conflict for a minimum dwell time;
4. releases with a second zero-endpoint-acceleration quintic so boundary entry occurs after the protected agent's late arrival envelope plus a gap;
5. sends the resulting trajectory through the same map, acceleration, jerk, validity, and NCF label machinery as every other candidate.

This is a targeted **proposal refinement** for protected burden transfer. It does not change the NCF definition, burden normalization, BCOT target, RootTransport target, or calibration thresholds in this release.

### Offline/online consistency and candidate-budget repair

- The same hold-release primitive is available in online Waymax candidate generation. Offline-only proposal families are prohibited because the trained mechanism would otherwise see actions that disappear in closed loop.
- A core neutral proposal is reserved early, before optional timing/lane/terminal families can saturate the candidate budget. This prevents proposal expansion from deleting the intervention root needed by the certificate.
- Fresh source provenance includes `PRIORITY_HOLD_RELEASE`, enabling direct source-level NCF/PBTR ablation.

### Data decision protocol

A full rebuild is **not authorized immediately**. The current decision is two-stage:

1. **192-scene micro screen**: 64 old hard scenes + 128 unbiased validation scenes. It must show PHR generation, priority-NCF yield, material PBTR-floor improvement, no major NCF destruction, acceptable validity, and pairing completeness.
2. **Strict 1,200-scene paired proposal probe**: 400 hard + 800 random. Only `promote_to_full_rebuild=true` authorizes a fresh full v16.8.6 rebuild.

If the micro screen fails, do not rebuild. Improve proposal refinement first. If the strict probe passes, a fresh rebuild becomes necessary for paper-grade v16.8.6 training because the legacy overlay cannot retrofit the new ego proposal tensors or their NCF labels.

### Experiment-pipeline corrections

1. `05_make_tables.py` no longer materializes all large natural/response trajectories. It loads only table-required arrays, compacts candidate trajectories to first/last samples for exact label-space progress, uses threaded NPZ loading, and writes a persistent compact cache. Module-effect tables reuse already-computed planner decisions instead of running each label-space planner twice.
2. Proposal-source ablation now fails on stale caches without `proposal_source`; the wrapper skips it explicitly instead of fabricating an RMR contribution from all-PAD provenance.
3. GameFormer/DTPP adapters now transform agents, futures, candidates, and map points into the ego frame while preserving roadgraph padding validity. Loss-side numerics are FP32; AMP `auto` prefers BF16. Training hard-fails if skipped batches exceed 2%, preventing a checkpoint trained on a tiny surviving subset from being treated as a strong baseline.
4. Fresh v16.8.6 cache protocol requires `data_manifest_v16_8_6.json`, a matching build fingerprint, current proposal provenance, and `.transport_v16_8_6` sidecars. Legacy-bank diagnostics are isolated behind an explicit non-paper-grade wrapper.
5. GameFormer/DTPP are candidate-bank scorers in this repository. A legacy-bank rerun is therefore only a numerical smoke; final matched comparisons must retrain/evaluate them on the same fresh proposal bank as COWP to avoid a train/test proposal mismatch.

### Directions explicitly not repeated

Do **not** spend the next iteration on:

- simply increasing BCOT budget;
- threshold-only repair on the fixed legacy candidate bank;
- flat candidate certificates in place of root-conditioned transport;
- making all critical pairs a hard veto;
- treating stop/yield as automatically non-coercive;
- retraining a natural decoder that has already passed its foundation gate without a new causal hypothesis;
- using sparse cached candidate Waymax outcomes as the final closed-loop claim.

### Promotion interpretation

- **Keep/deepen:** counterfactual natural roots, neutral intervention, protected priority relation, root-conditioned transport/BCOT, explicit fallback, certificate/shortlist separation, and certificate-guided proposal refinement.
- **Demote as primary contribution:** current hard option-preservation and hard witness-rejection decision gates, because the uploaded oracle ablations show near-zero marginal decision effect on the legacy bank. They may remain as safeguards/diagnostics until learned and closed-loop ablations are available.
- **Next theoretical step after proposal sufficiency:** shift-aware calibration under ego-policy-induced interaction distribution shift; ordinary IID calibration alone is not a sufficient final safety argument for reactive closed loop.

### Local validation

- `pytest -q`: **164 passed** (one existing PyTorch nested-tensor prototype warning).
- `python -m compileall -q cowp`: pass.
- All v16.8.5/v16.8.6 execution entrypoints modified or added in this release: `bash -n` pass.
- `52_screen_priority_commitment_probe`, `53_gate_fresh_v16_8_6_cache_protocol`, `05_make_tables`, and `20_train_external_baseline`: import/`--help` smoke pass.
- Legacy v15--v16.5 shell files with historical CRLF/syntax defects were intentionally not mass-edited; they are outside the v16.8.6 execution path and are documented rather than hidden by unrelated diffs.

### Prohibited claims before new data is run

- Do not claim PCHR improves WOMD/Waymax until the micro/strict probes and fresh closed-loop experiments run.
- Do not call the uploaded interrupted v16.8.4 archive a completed fresh-probe verdict.
- Do not report the uploaded GameFormer/DTPP learned results as strong baselines: their old training logs skipped more than 99% of batches.
- Do not report a COWP v16.8.6 Waymax result from the legacy v16.8 raw candidate bank as a paper-grade evaluation.

# COWP Algorithm Change Log

This is the canonical record of algorithm attempts. Do not repeat a rejected
change without new evidence. Every experiment must record the code version, data
version, seed, checkpoint lineage, learned-offline gate, online paired metrics,
and the exact simulator-agent setting.

## v16.8.5 — Probe-Resilient Data Decision, PBTR-Sufficient Promotion, and Fast Rebuild Protocol

### Triggering evidence

The v16.8.4 400-hard + 800-random proposal probe aborted after about 5.5 hours with
768 fresh labels written. The worker exception was `No valid ego candidates available
for neutral intervention`. This is a scene-local proposal-bank outcome, but the old
builder propagated it through `Future.result()` and killed the whole multiprocessing
job. The same run also exposed two cost bugs: scenario allowlisting happened inside
workers after raw protobufs had already crossed process IPC, and `--limit` counted only
newly written files instead of valid `--skip-existing` outputs.

The uploaded current-cache audit shows the existing v16.8 transport overlays are
structurally healthy for core RCOT/BCOT training (20,440 train / 5,013 val, required
transport/core key coverage complete, response-root assignment valid). They are not a
fresh BCS-RMR label dataset. On the full old validation cache the proposal ceiling is
`AnyNCF=0.36246`, selected-false-safe floor `0.53321`, and best-case PBTR floor
`0.58179`. Because the v16.8 calibration contract uses `max_priority_burden_transfer=0.45`,
the old proposal bank remains mathematically infeasible for the current mechanism gate
even though its global false-safe floor is below 0.55.

### Engineering/data-protocol changes (no certificate-definition change)

1. Added `NoValidEgoCandidatesError` with scenario id and proposal-rejection diagnostics.
   Zero-valid scenes are now returned as `status=filtered, filter_reason=no_valid_ego_candidates`
   instead of aborting the full build. Candidate acceptance semantics are unchanged.
2. Added per-scene proposal rejection diagnostics (`map_filter`, accel/decel, jerk,
   nonfinite, duplicate, capacity) so the next failed scene identifies the physical reason.
3. Sparse allowlists are filtered in the producer before multiprocessing IPC. Workers now
   receive only requested Scenario records; the scan stops when all requested IDs have been
   observed.
4. Fixed resumable `--limit`: complete existing files count toward the output target.
   Build profiles are appended when `--skip-existing` is used so filtered/no-valid terminal
   rows survive a resume. Worker errors now preserve the expected scenario id.
5. Proposal-probe default changed to `FORCE_REBUILD_PROBE=0`; the 768 already generated
   fresh labels are reused. The probe records a stage-level profile JSONL.
6. Promotion is now **proposal-sufficient for both mechanism constraints**: in addition to
   `AnyNCF>=0.40`, false-safe floor `<=0.55`, hard-scene recovery `>=0.20`, and RMR TTA
   error `<=0.20 s`, it requires `AnyValid>=0.99` and best-case PBTR floor `<=0.45`.
   Filtered requested scenes are conservatively counted as zero-valid in the paired probe;
   unexpected missing/error scenes remain fatal.
7. `45_diagnose_proposal_ceiling` gained exact index-modulo partition diagnostics matching
   learned-offline calibration/held-out splitting. This resolves the observed discrepancy
   between the old full-cache floor and the earlier calibration-partition floor before any
   expensive decision.
8. Added `PREPARE_COWP_V16_8_5_DATA_FAST_CN.sh`: reuses existing indexes, profiles label
   stages, supports true resume, skips visualization diagnostics during the critical path,
   and defaults `RUN_WAYMAX_REPLAY=0`. Full train-set Waymax replay is not required for
   natural/response/witness/RCOT/planner core supervision or real online Waymax evaluation.
9. Planner training gained `USE_WAYMAX_OUTCOME_LABELS=0`, allowing fresh core-only caches;
   the existing loss code already returns exactly zero outcome loss when those optional
   tensors are absent. This removes the need to spend large replay cost before the main
   mechanism experiment.
10. Resumed builds with existing NPZ files now bypass the worker pool in the producer
    once the scenario id and file integrity are verified. This preserves semantics while
    preventing hundreds/thousands of already-complete scenes from occupying expensive
    label-engine worker slots after an interruption.
11. Added `49_summarize_label_build_profile.py`, which reports p50/p90/p99 total and
    per-stage times plus zero-valid rejection causes. Worker-count or algorithmic speed
    changes should be based on this profile instead of guessed.
12. Added `50_ablate_proposal_sources.py` for post-build proposal-source ablations
    (`all`, `without_rmr_bcte`, `without_legacy_timing`, `without_any_timing`) without
    retraining or Waymax. This isolates the marginal proposal-ceiling contribution of
    BCS-RMR-BCTE.
13. Fixed an evaluation-validity bug: historical `cowp_wo_*` architecture/causal-branch
    names were silently treated as ordinary COWP in learned-offline/online shared-forward
    evaluation. Those names now fail loudly. Valid shared-forward selection baselines are
    evaluated in one model/cache pass; causal-branch and graph-architecture ablations must
    use label-space diagnostics or separately retrained checkpoints.
14. `05_make_tables.py` no longer rereads every NPZ once per method before recomputing the
    same decisions; labels are loaded once and reused in memory.
15. Probe resume now preserves the uploaded v16.8.4 build fingerprint and accepts only
    that exact known interrupted lineage (or the current fingerprint). A separate semantic-resume
    manifest records why the 768 successful NPZ labels are compatible: v16.8.5 changed only
    zero-valid handling/diagnostics for those paths, not serialized tensors for valid scenes.
16. Fast full rebuild can reuse the audited old cache **scenario-id set** and WOMD indexes while
    rebuilding every COWP label from fresh Scenario protos. This gives an exact paired dataset,
    avoids spending label-engine time on scenes that never entered the old training cache, and
    caps per-process BLAS/TensorFlow threads to prevent multiprocessing oversubscription.
17. Added staged experiment wrappers for multi-seed mechanism runs and fresh-cache external
    baselines. Expensive replication is intentionally delayed until the single-seed mechanism and
    small Waymax probe pass; repository GameFormer/DTPP implementations are treated as matched
    baselines, not mislabeled as official reproductions.

### Data decision after this round

- **Do not start the four-day full rebuild yet.** Resume and finish the corrected paired
  fresh proposal probe first.
- The old cache may be reused for engineering/core-module controls, but it cannot validate
  the v16.8.4 BCS-RMR proposal claim and cannot satisfy the current PBTR proposal floor.
- Full fresh rebuild is justified only if the paired fresh bank passes all six proposal
  checks, especially `BestCasePBTRLowerBound<=0.45`. If it fails, change/refine proposal
  generation rather than paying for a larger copy of an infeasible bank.

### Validation

- `pytest`: 160 passed.
- `python -m compileall -q cowp`: pass.
- Updated/added v16.8.5 shell entrypoints plus `run_cowp_v16_8_dual_gpu.sh`: `bash -n` pass.
- Note: several historical pre-v16.8 root scripts in the uploaded archive retain legacy CRLF/syntax issues; they are outside this v16.8.5 execution path and were not silently rewritten.

## v16.8.4 — Boundary-Consistent Smooth RMR-BCTE, Fresh-Cache Hard Gate, and Online Timing Alignment

### Triggering evidence from the uploaded `cowp_v16_8_3_rmr_bcte_seed2026` run

The directory name says v16.8.3, but the run provenance shows that training and
learned-offline evaluation actually used the old v16.8 overlay protocol:

- `DATA_PROTOCOL=v16_8_root_conditioned_overlay`;
- raw caches are `formal/tensor_cache_{train,val}_waymax`;
- transport caches are `formal/tensor_cache_{train,val}_waymax_transport_v16_8`.

Those caches are structurally healthy for the v16.8 RCOT/transport experiment
(20,440 train / 5,013 val files, sampled required-key coverage 100%, transport
augmentation complete, response-root out-of-range rate 0), but a transport
sidecar cannot replace the ego candidate tensors stored in the raw cache.
Therefore this run did **not** evaluate the v16.8.3 RMR-BCTE proposal repair.

The current mechanism report still provides strong diagnostic evidence:

- pair-witness AUPRC `0.84945` (pass);
- protected BCOT false-safe AUPRC `0.96984` (pass);
- protected RootTransport AUPRC `0.88167` (pass);
- protected NCF recall `0.97894` and global recall `0.98387` (pass);
- protected NCF precision `0.49374` (slightly below the `0.50` gate);
- accepted-candidate rate `0.32079` (pass);
- fallback `0.41341` (fail, target `<=0.25`);
- PBTR improvement `0.01216` (fail, target `>=0.03`);
- selected false-safe improvement `0.01038` (fail, target `>=0.03`).

Calibration correctly reports `status=proposal_infeasible`: the fixed-bank
best-case selected-false-safe lower bound is `0.60989` and the best-case PBTR
lower bound is `0.64457`.  Thus threshold/budget tuning on this bank cannot pass
the mechanism gate.

### Newly confirmed proposal-semantics defect

v16.8.3 fixed the jerk filter and broadened timing proposals, but its RMR-BCTE
arrival solver used Euclidean distance to the **conflict-region centre**.  The
interaction/collision geometry declares entry when the vehicle envelope first
crosses the **inflated conflict-region boundary**.  Consequently a candidate
could be tagged `pass-after` even though it entered the actual conflict region
much earlier.

A deterministic toy regression reproduces the mismatch in the uploaded code:
a `pass-after` proposal advertised target TTA `5.60 s` but continuously entered the conflict
envelope at about `3.47 s` (error about `2.13 s`).  This is not a threshold problem: the
proposal itself violates its timing semantics and can directly reduce NCF yield.

### Algorithm changes

1. **Boundary-consistent reachability and distance.** Added continuous
   `trajectory_entry_to_region`, which returns first entry time and travelled arc
   length to the same inflated region boundary used by the interaction geometry.
   Conflict regions that the nominal ego primitive never reaches no longer
   trigger STOP/legacy/RMR timing proposals.
2. **Boundary-Consistent Smooth RMR-BCTE (BCS-RMR-BCTE).** Timing proposals now
   solve to the conflict-envelope entry distance, not the region centre.
   The longitudinal primitive is a cubic profile
   `s(t)=v0*t+c2*t^2+c3*t^3` with `s(T)=d` and `a(T)=0`, giving finite constant
   jerk before arrival and allowing delayed arrival without the
   constant-deceleration stop-before-arrival pathology.
3. **Causal agent boundary envelopes.** Agent early/nominal/late TTA uses only the
   current state and current velocity ray intersected with the inflated conflict
   boundary.  No logged future is introduced into proposal generation.
4. **Realized timing contract.** Every BCS-RMR proposal is re-measured against the
   region boundary and rejected if `|TTA_realized-TTA_target|>0.20 s`.  New cache
   provenance stores `proposal_entry_distance_m` and
   `proposal_target_tta_error_s`.
5. **Diversity-preserving timing deduplication.** RMR profiles are deduplicated by
   `(region, timing_side, acceleration_bin)` rather than a global acceleration
   bin, so one region/side cannot erase a distinct interaction hypothesis.
6. **Online timing alignment.** Waymax candidate generation now uses the same
   smooth-arrival family for robust timing hypotheses and includes the omitted
   current-state-to-first-sample arc length.  Online profile deduplication is by
   `(agent, timing_side, acceleration_bin)`.

The same toy regression after v16.8.4 produces pass-after proposals with target
TTAs `4.48376 s` and `5.08376 s`, realized errors only `3.3e-6 s` and `4.3e-6 s`.
This proves semantic consistency of the primitive; it does **not** yet prove
higher WOMD NCF coverage or SOTA closed-loop performance.

### Engineering/protocol changes

1. **Strict fresh-cache preflight.** New
   `47_gate_fresh_v16_8_4_cache_protocol.py` refuses to train/probe/full-evaluate
   when the requested cache lacks the v16.8.4 manifest, matching code fingerprint,
   new proposal provenance tensors, or the `.transport_v16_8_4` sidecar contract.
   This prevents an exported old `tensor_cache_*_transport_v16_8` path from
   silently being called a v16.8.4 experiment.
2. **Expanded build fingerprint.** The data/probe fingerprint now includes
   `lane_graph.py` and `trajectory_primitives.py` in addition to candidate,
   label/schema and config code.  v16.8.3 omitted these two files, so a geometry
   or primitive change could otherwise be resumed into a stale build root.
3. **Fresh protocol version.** Causal audit accepts `v16_8_4_fresh`; v16.8.4
   mechanism/probe/full wrappers explicitly export all fresh cache paths instead
   of inheriting the old v16.8 defaults.
4. **Versioned proposal diagnostics.** New evaluation rows use
   `v16_8_4_boundary_consistent_proposal_floor`; calibration remains backward
   compatible with v16.8.3 rows.
5. **Proposal probe gains a timing gate.** Promotion now additionally requires at
   least one measured RMR timing proposal and maximum target-TTA error `<=0.20 s`.
6. **Paired Waymax promotion gate.** The old FULL wrapper checked only that a
   probe-delta file existed.  v16.8.4 now reads the raw conventional/COWP probe
   outputs and blocks full rollout on insufficient paired rollouts, safety/progress
   regression beyond probe tolerance, excessive episode fallback, or obvious
   degradation in available predicted false-safe/burden/OPR proxies.  This is a
   cost-control continuation gate, not a causal/publication claim.
7. **Manuscript method synchronization.** A new
   `interactive_planning_v16_8_4_revised.tex` updates only the proposal-generation
   method definition to boundary-entry timing, smooth cubic arrival and realized
   TTA validation.  No unrun WOMD/Waymax number or SOTA claim is inserted.

### Data decision

Do **not** immediately rebuild all 22k/5k scenes and do not retrain transport or
planner on the current v16.8 overlay.  First run the 400-hard + 800-random paired
**v16.8.4 label-only probe**.  Promote to the expensive rebuild only when:

- `AnyNCFSceneRate >= 0.40`;
- best-case selected false-safe lower bound `<= 0.55`;
- hard-scene NCF recovery `>= 0.20`;
- BCS-RMR realized target-TTA max error `<=0.20 s`.

If coverage/floor still fail after timing consistency is fixed, the next
algorithmic step is **proposal-space refinement**, not looser certification:
route-topology/Frenet interaction proposals or a strong learned/generative
proposal backbone should be added underneath the unchanged non-coercion
certificate.  That preserves the paper's novelty as a certificate/refinement
layer while removing the fixed kinematic-bank ceiling.

### Algorithm disposition after this round

- **Retain:** protected non-coercive hard feasibility, natural roots,
  `s=(1-c)r+cq`, independent recovery/burden heads, RCOT/BCOT aggregation,
  explicit uncertified fallback, certificate/shortlist separation.
- **Deepen next (after proposal probe):** certificate-guided proposal refinement;
  distributional RCOT; simultaneous one-sided protected-pair calibration; and
  shift-aware risk control for ego-policy-induced interaction distribution shift.
- **Do not revive:** flat candidate certificate as the primary mechanism,
  all-critical hard veto, threshold-only rescue, automatic stop/yield safety, or
  sparse cached Waymax replay as the final closed-loop claim.
- **Secondary theory audit, not changed in v16.8.4:** the primitive burden weight
  vector still allocates weight to the option component while witness construction
  zeroes that component and handles option preservation separately through OPR.
  Changing its normalization now would alter the NCF definition and confound the
  proposal-coverage experiment; audit/calibrate it only after the proposal gate.

### Next decision rule

1. Run `NEXT_RUN_COMMANDS_V16_8_4_PROPOSAL_PROBE_CN.sh` against the existing old
   validation cache for paired comparison.
2. If `promote_to_full_rebuild=false`, stop before full rebuild/training and use
   source-level RMR yield + hard-scene recovery to design the next proposal
   refinement.
3. If true, build a **new** `formal_v16_8_4_bcs_rmr_bcte` data root with
   `PREPARE_COWP_V16_8_4_DATA_CN.sh` and train only transport/planner from the
   already validated natural checkpoint.
4. Waymax probe/full remains a hard continuation gate: both
   `mechanism_verification.pass=true` and `calibration_feasible=true` are required.
5. After the small Waymax probe, `NEXT_RUN_COMMANDS_V16_8_4_FULL_CN.sh`
   additionally requires `promotion_gate_v16_8_4.json: pass=true`; a probe file
   merely existing is no longer sufficient to spend the full rollout budget.

## v16.8.3 — Proposal-Sufficiency Audit, Jerk-Consistent Candidate Validation, and Robust Multi-Region BCTE

### Triggering evidence

The corrected v16.8.2 re-evaluation of the v16.8.1 checkpoint completed the
natural, model-anchor, cache-reuse, causal-overlay, calibration and held-out
learned-offline chain.  The evaluation semantics are current and the
calibration/held-out partitions are disjoint, but the main mechanism gate remains
infeasible:

- `mechanism_verification.pass=false`;
- `calibration_feasible=false`, `status=least_violation`;
- protected NCF recall `0.97064` (pass), protected NCF precision `0.47628`
  (fail), accepted-candidate rate `0.32956` (pass), fallback `0.38667` (fail);
- held-out PBTR improves only `0.01261` over conventional and selected
  false-safe rate improves only `0.01237`.

The corrected sweep rules out a threshold-only repair.  At budget `0.70`,
protected precision passes (`0.50987`) but fallback remains `0.42401`; at budget
`0.90`, fallback passes (`0.21659`) but precision falls to `0.35071`.  PBTR and
selected false-safe rate change by less than one percentage point over the useful
budget range.

### Proposal-sufficiency finding

The held-out bank contains at least one NCF candidate in only `0.27255` of scenes,
while `0.89146` of scenes contain a conventionally safe candidate.  Therefore

`P(any conventional-safe and no NCF) >= 0.89146 - 0.27255 = 0.61891`.

For any selector restricted to this fixed bank and required to return a
conventionally safe candidate when one exists, `0.61891` is a scene-level
selected-false-safe lower bound.  It already exceeds the gate target `0.55`.
The current selected false-safe rate `0.63966` is only `0.02075` above this bank
floor, so selector/certificate tuning has little remaining headroom without new
proposals.  The certificate itself retains about `97.5%` of scenes where an NCF
proposal exists; the dominant ceiling is upstream.

### Confirmed engineering defect

Offline `_candidate_valid` exposed `ignore_initial_jerk_steps=3` and
`jerk_check_percentile=99`, but ignored both and rejected candidates using the
absolute maximum finite-difference jerk.  Online validation already used the
configured prefix removal and percentile.  The mismatch systematically removed
constant-acceleration timing candidates because the first discrete derivative
contains the transition from the logged current acceleration to the primitive
acceleration.  This defect especially affects BCTE, accelerate, yield and stop
families and invalidates any decision to pay for a full BCTE rebuild before a
small corrected probe.

### Algorithm and implementation changes

1. **Jerk-consistent offline validation.** Offline candidates now use the same
   ignored prefix and robust percentile as online validation.
2. **Physical arrival constraint.** A constant-deceleration timing solution is
   rejected when its requested conflict arrival occurs after terminal speed
   becomes negative; zero-speed clipping can no longer masquerade as a later
   arrival.
3. **Robust Multi-Region BCTE (RMR-BCTE).** The proposal generator ranks
   forward-reachable conflict regions, retains up to three, constructs bounded
   early/nominal/late arrival envelopes for up to four approaching agents per
   region, and proposes pass-before against the early boundary and pass-after
   against the late boundary.  A 24-candidate quota and acceleration-space
   deduplication prevent timing hypotheses from exhausting the full bank.
4. **Offline/online semantic alignment.** Online conflict timing uses the same
   before/after envelope convention, physical arrival check, acceleration bounds,
   quota and deduplication intent.
5. **Proposal provenance.** Each candidate records source, conflict region,
   target time, timing side, target agent, gap and solved acceleration.  These
   fields are persisted in labels/cache and are optional for old cache readers.
6. **Proposal-floor metrics.** Offline/online aggregation now reports the
   conventional-without-NCF lower bound, protected proposal floor, NCF selection
   recall given availability, and selector excess above the bank floor.
7. **Infeasibility-aware calibration.** If no operating point is feasible and
   proposal floors already violate hard constraints, calibration reports
   `status=proposal_infeasible` rather than implying that another threshold is
   likely to solve the problem.
8. **Paired label-only proposal probe.** A new diagnostic samples 400 old-bank
   hard scenes and 800 unbiased random validation scenes, rebuilds only their
   labels, and compares old/new banks by scenario ID.  Missing requested IDs are
   a hard error.  Full reconstruction is promoted only if all three conditions
   pass: `AnyNCFSceneRate>=0.40`, selected false-safe floor `<=0.55`, and hard
   scene NCF recovery `>=0.20`.
9. **Build fingerprints.** Probe and full-data scripts fingerprint the proposal,
   label and configuration implementation so interrupted resumes cannot silently
   mix pre-fix and post-fix data.
10. **Fresh protocol version.** The causal protocol audit accepts
    `v16_8_3_fresh` and reports its fresh-label status separately.

### Data decision

Do **not** start the approximately four-day complete rebuild yet.  Existing data
is sufficient to establish the natural-basis quality, strong pair/root ranking,
certificate retention, corrected gate failure and fixed-bank proposal ceiling.
It is not sufficient to measure RMR-BCTE coverage, source-specific NCF yield, or
paper-grade fresh causal labels.  Run the paired label-only proposal probe first.
Only a passing probe justifies `PREPARE_COWP_V16_8_3_DATA_CN.sh`.

### Algorithm disposition

- **Retain:** non-coercive feasibility as a hard protected-priority certificate;
  natural roots; `s=(1-c)r+cq`; separate `q` and `b*`; protected BCOT; explicit
  uncertified fallback; certificate/shortlist separation.
- **Deepen after proposal repair:** distributional RCOT with censored minimum-safe
  burden, simultaneous one-sided calibration across protected relations,
  shift-aware/sequential risk control, and certificate-guided proposal refinement
  attached to a stronger flow/world-model/VLA planner.
- **Keep diagnostic-only or remove from claims:** flat candidate certificate,
  all-critical hard veto, automatic stop/yield safety assumptions, sparse cached
  Waymax outcomes as closed-loop evidence, and repeated BCOT-budget enlargement.

### Validation

- Full PyTest after v16.8.3 changes: **153 passed** (before final packaging; rerun
  recorded in the delivery validation file).
- New and modified Python modules compile.
- v16.8.3 proposal, data, mechanism, probe and full shell entry points pass
  `bash -n`.
- Paper adds a fixed-bank proposal-sufficiency proposition and RMR-BCTE details;
  no new empirical SOTA claim is made before the paired probe and closed-loop
  gates pass.

### Next decision rule

1. Run `NEXT_RUN_COMMANDS_V16_8_3_PROPOSAL_PROBE_CN.sh`.
2. If `promote_to_full_rebuild=false`, do not train or rebuild full data; use
   source-level statistics to determine whether conflict-region discovery,
   timing coverage, candidate budget, or the non-coercion label is the blocker.
3. If true, run `PREPARE_COWP_V16_8_3_DATA_CN.sh`, then
   `NEXT_RUN_COMMANDS_V16_8_3_MECHANISM_CN.sh`.
4. Waymax probe/full remains blocked until both `mechanism_verification.pass` and
   `calibration_feasible` are true under current metric semantics.

## v16.8.2 — Certificate/Shortlist Separation, Explicit Least-Coercive Fallback, Hard Protected Semantics, and BCTE Proposal Repair

### Triggering evidence

The uploaded `cowp_v16_8_1_rcot_consistent_v9base_seed2026` run completed
transport/planner training and disjoint learned-offline evaluation, but both
continuation conditions failed:

- `mechanism_verification.pass=false`;
- `calibration_feasible=false` (`status=least_violation`, selected budget 0.65).

The failure is not a lack of ranking signal. Held-out pair-witness AUPRC is
0.82033, protected BCOT false-safe AUPRC is 0.96498, and protected
RootTransport AUPRC is 0.86936. The failing operating-point metrics are
protected NCF recall 0.21176 (<0.30), protected NCF precision 0.49802 (<0.50),
accepted-candidate rate 0.06851 (<0.10), fallback 0.35874 (>0.25), and PBTR
0.59222 (>the calibration constraint 0.45). The proposal bank contains an NCF
candidate in only 0.27255 of held-out scenes, which is an upstream ceiling on
certificate selection.

### Confirmed evaluation/engineering defects

1. **Certificate acceptance was overwritten by the Pareto shortlist.**
   `_select_from_learned` first computed the hard RCOT certificate mask, then
   replaced it with a top-k Pareto frontier (runtime maximum 8 candidates).
   `LearnedAcceptedCandidateRate`, NCF recall and certificate coverage were
   therefore measuring the selector shortlist rather than the semantic
   certificate set.
2. **Valid-index fallbacks were undercounted.** Offline fallback selected a
   stop/yield candidate index, while the accumulator counted fallback only when
   the selected index was invalid. This made fallback semantics dependent on an
   implementation detail rather than the actual decision path.
3. **Calibration could silently consume stale metrics.** Old JSON rows had no
   marker distinguishing the pre-v16.8.2 shortlist semantics from corrected
   certificate semantics.
4. **Checkpoint selection included inactive/frozen objectives.** The witness
   score emphasized the disabled all-root recovery loss and omitted direct root
   burden/consistency/budget terms. The planner score included frozen transport
   and zero-weight candidate-certificate losses, diluting trainable improvement.
5. **The legacy flat candidate certificate is collapsed but inactive.** Its
   selected means (`ncf≈0.04`, `false_safe≈0.999`, `quality≈1e-5`) are not a
   valid main selector signal. It remains diagnostic-only and is excluded from
   checkpoint selection and paper claims.
6. **The previous “all top-level shell scripts pass” validation claim is not
   reproducible from the uploaded archive.** Multiple historical v15--v16.6
   wrappers retain CRLF or obsolete heredoc syntax. They are outside the
   v16.8.2 execution chain, but the delivery now validates only the enumerated
   current scripts and records the historical failures instead of silently
   treating the whole repository as shell-clean.

### Algorithm corrections

1. **Separate semantic certificate from selection shortlist.** Evaluation and
   online policy now keep `certificate_accepted` and `selection_shortlist` as
   different masks. Gates use the former; shortlist diagnostics are reported
   separately.
2. **Explicit fallback contract.** Fallback is always marked, even when it
   returns a valid candidate. It selects the minimum predicted coercion-risk
   candidate using transport UCB, protected-rule risk, action risk, pressure,
   cached outcome risk and a small utility term. Stop/yield is only a weak tie
   preference, not an assumption of non-coerciveness.
3. **Hard protected-relation anchor.** `AgentPriority` and
   `EqualOrNegotiated` relations are protected by rule and cannot be diluted by
   a learned priority head. Learned priority is used only for unknown relations;
   `Unprotected` remains unprotected.
4. **Full-certificate uncertainty target.** Mode uncertainty is trained against
   the maximum normalized error across conflict, retain, same-root recovery and
   minimum safe burden. It is no longer calibrated only to conflict/retention
   while being used as a UCB for the complete RCOT risk.
5. **Bidirectional Conflict-Time Envelope (BCTE).** Candidate generation solves
   bounded ego acceleration for arrival immediately before/after nearby agents'
   plausible arrival times at the same conflict region. This targets the
   measured `AnyNCFSceneRate=0.27255` proposal-coverage bottleneck. BCTE is a
   proposal repair; it does not weaken the downstream conventional or RCOT
   certificates.
6. **Checkpoint scores aligned to active objectives.** Transport selection now
   uses conflict-conditioned recovery, recovery ranking, root burden,
   q--b* consistency, uncertainty and candidate budget losses. Planner selection
   uses only trainable planner losses.
7. **Versioned metric semantics.** Learned-offline rows now contain
   `CertificateSemantics/Version=v16_8_2_decoupled` and
   `FallbackSemantics/ExplicitAccounting=true`. Calibration and verification
   reject stale rows.

### Training-state decision

The uploaded transport run ended after 14 epochs and the planner after 10; every
recorded checkpoint improved and both histories end with
`checkpoint/no_improve_checks=0`. These are schedule-truncated models, not
converged models. v16.8.2 raises defaults to 24 transport and 16 planner epochs,
while preserving early stopping. The natural basis remains frozen and is not
retrained.

### Local validation

- Full test suite: **150 passed**, one upstream PyTorch prototype warning.
- Python `compileall`: passed.
- The eight scripts in the v16.8.2 execution chain pass `bash -n`.
- TeX compiles with `pdflatex` to 19 pages.
- CPU realistic preflight and mechanism-overlay causal-audit smoke pass.
- A repository-wide diagnostic finds 38 failures among 73 historical top-level
  shell scripts, primarily CRLF/obsolete heredoc wrappers from v15--v16.6; they
  are not in the v16.8.2 execution path and are not represented as clean.

### What is intentionally not claimed

- The corrected selector metrics have not been recomputed here because the
  uploaded result package omits the large checkpoints and the local environment
  has no CUDA/Waymax.
- BCTE proposal coverage is not yet empirically validated; a fresh candidate
  label/cache rebuild is required for learned-offline evidence.
- No SOTA or calibrated safety guarantee is claimed from sparse cached outcomes
  (selected coverage about 20--33%, finite log-divergence count 0).

### Next decision rule

1. Re-evaluate the existing v16.8.1 checkpoints under v16.8.2 semantics to
   isolate the metric/selector/fallback correction from retraining.
2. If the gate remains infeasible, retrain transport/planner to convergence with
   corrected uncertainty and checkpoint scores.
3. If `ProposalCoverage/AnyNCFSceneRate < 0.35` or PBTR/fallback remain the
   blockers, rebuild fresh BCTE labels/cache before any budget tuning.
4. Do not answer failure by only raising the BCOT budget; the previous sweep
   already shows recall plateauing while PBTR worsens.


## v16.8.1 — Definition-Consistent RCOT, Direct Root Burden, and Recovered Execution Chain

### Triggering evidence

The uploaded `cowp_v16_8_pipeline_v9labels_seed2026` and
`cowp_v16_8_rcot_v9base_seed2026` runs did not reach training. Both terminate in
`36_audit_causal_protocol.py` because the launcher requests
`v16_8_root_conditioned_overlay` while the parser accepts only `v15` and
`v9_reuse`. The cache/overlay diagnostics immediately before the failure are
healthy: 20,440/20,440 train files, 5,013/5,013 validation files, zero overlay
errors, zero split filename overlap, complete critical mapping, and no response
root indices out of range.

### Definition corrections

1. **Implemented the complete root transport equation in model forward.** The
   predicted transported root probability is now
   `retain_prob + conflict_prob * mode_recovery_prob`, i.e.
   `s=(1-c)r+cq`. Earlier forward inference omitted `c*q`, even when supervision
   rebuilt transported OPR.
2. **Separated low-burden recoverability from minimum safe burden.** The
   candidate--agent--root head now predicts `q_ikm` and `b*_ikm` through separate
   channels from a shared root-conditioned latent. `q=0` no longer forces the
   no-safe-response sentinel; a finite high-burden safe response remains
   representable.
3. **Added confidence-masked q--b* consistency.** On confidently labelled
   conflicting roots, the loss aligns `q` with a smooth thresholding of `b*`
   around the adaptive burden budget while preserving direct supervision and
   independent outputs.
4. **Made the direct root representation the primary certificate.** OPR and
   root-CVaR use direct `q`/`b*`. The compact generic response bank is retained
   only for auxiliary reconstruction, qualitative visualization, and ablation.
5. **Unified the natural-root probability measure.** Label adaptation, future
   overlay generation, and model inference now apply the same `p_min` support,
   surviving-mass renormalization, and active-support probability floor.
6. **Removed recovery-loss dilution.** The main recovery objective is
   conflict-only BCE plus conflict-conditioned ranking; the all-root recovery
   BCE is disabled.

### Execution and engineering corrections

1. `36_audit_causal_protocol.py` now accepts
   `v16_8_root_conditioned_overlay` and separately reports engineering,
   mechanism-overlay, and fresh-v15 protocol status.
2. Added an audited checkpoint compatibility loader for the exact 4-to-5-row
   expansion of `set_transport.mode_out`; all other shape mismatches remain
   hard failures.
3. Fixed the background launcher so the parent exits after spawning the nohup
   child.
4. The natural transfer manifest now records both checkpoint and training
   history with SHA-256 hashes. New probe/full launchers restore them in a fresh
   shell instead of assuming inherited environment variables.
5. Added `NEXT_RUN_COMMANDS_V16_8_PROBE_CN.sh`; full evaluation is separated
   from probe and requires a completed probe delta.
6. Changed default output to
   `outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026` to preserve strict
   provenance and prevent reuse of artifacts produced by the incorrect code.

### Dataset decision

- Reuse the existing v16.8 overlay for the next mechanism run; no rebuild is
  required for these model/loss fixes.
- Keep log-divergence supervision disabled because finite coverage is zero.
- Treat cached Waymax outcomes as partial auxiliary labels, not unbiased
  full-candidate closed-loop evidence.
- Do not use this v9-base overlay as evidence that the fresh v15/v16 causal-label
  protocol has been validated.

### Validation

- Full test suite: 144 passed, 5 skipped after the new launcher regression test.
- Python compileall: passed.
- All top-level shell scripts: `bash -n` passed.
- CPU realistic model/loss preflight: passed.
- Causal-audit smoke: engineering and mechanism-overlay protocol passed; fresh
  v15 label protocol correctly remained false.

### Next decision rule

Do not enter Waymax probe unless the disjoint learned-offline report has both
`pass=true` and `calibration_feasible=true`, including priority RootTransport
AUPRC, NCF recall/precision, acceptance, fallback, PBTR improvement, and global
false-safe improvement gates. Do not respond to a failed gate by only increasing
`BCOT_RISK_BUDGET`.

## v16.8 — Root-Conditioned Counterfactual Transport (RCOT), Canonical OPR, and Immutable Mechanism Checkpoint

### Evidence from the uploaded v16.7 mechanism-isolation run

The v16.7 run confirms that the natural basis is not the current bottleneck and that the
candidate-level ranking signal is substantially stronger than the failed gate suggests.
On the disjoint held-out partition:

- pair-witness AUPRC: **0.72893**;
- protected-priority BCOT false-safe AUPRC: **0.95611**;
- global BCOT false-safe AUPRC: **0.96146**;
- protected-priority RootTransport AUPRC: **0.20273**;
- protected NCF precision: **0.72535**;
- protected NCF recall: **0.19062**;
- accepted-candidate rate: **0.04902**;
- fallback rate: **0.44972**;
- selected PBTR: **0.44331**, versus **0.65151** for conventional safety;
- selected global false-safe rate: **0.25379**, versus **0.59098** for conventional safety.

Thus the learned candidate score already orders false-safe candidates well and reduces
burden transfer directionally, but the root-recovery target and hard selection semantics
make the usable operating region collapse.  The gate failure is not evidence that the
coercion concept is ineffective; it is evidence that v16.7 did not measure or preserve
same-root recoverability according to the paper's own definition.

### Confirmed definition and engineering defects

1. **OPR omitted recovered conflicting roots.** Fresh witness construction and the
   paper-aligned training adapter both accumulated only the non-conflicting retained term.
   They did not implement `s=(1-c)r+cq`, so a conflicting root was counted as destroyed
   even when its own low-burden safe response existed.
2. **Global response-bank coverage was mistaken for recoverability.** Responses were
   generated from a candidate-independent global control lattice, globally truncated,
   and assigned to roots after the fact by nearest full-horizon trajectory.  A negative
   `q_ikm` therefore meant that the finite generic bank did not happen to cover root `m`,
   not that root `m` was unrecoverable under a bounded same-root counterfactual.
3. **Train/evaluation target mismatch remained.** Training reconstructed paper-aligned
   targets from transport fields, while learned-offline evaluation could calibrate and
   gate against stale cached false-safe/NCF labels.  The same checkpoint was therefore
   optimized and judged under different certificates.
4. **Missing response identities were clamped to root zero.** Legacy response slots with
   root index `-1` could be converted to root 0 by clamping, creating artificial recovery
   positives for the first root.
5. **Safe response was not always low-burden recovery.** For old caches without an
   explicit `is_low_burden` flag, safety alone could be treated as `q=1`; v16.8 derives
   the predicate from the adaptive burden budget `beta`.
6. **Planner training could rewrite the mechanism.** The planner stage continued to
   update SetTransport and response-decoder parameters, so the final planner checkpoint
   could trade a previously verified certificate for ranking loss.  This invalidated
   clean attribution to the mechanism stage.
7. **Duplicate severe veto.** A localized protected-pair severe veto and an aggregate
   candidate severe veto were both applied.  Under noisy recovery labels this doubled
   conservatism, suppressing acceptance without adding a distinct causal guarantee.
8. **Dangerous launcher defaults.** Direct execution of the v16.8 driver still defaulted
   to the old v9 transport cache and old sidecar name.  v16.8 now defaults exclusively to
   `tensor_cache_*_waymax_transport_v16_8` and records the overlay protocol in provenance.

### Root-Conditioned Counterfactual Transport (RCOT)

v16.8 makes same-root transport an explicit, independently testable object instead of a
by-product of a generic response decoder:

- For every valid conflicting natural root, generate a bounded family of longitudinal
  timing residuals around that root.
- Re-parameterize time along the original root polyline.  The control may slow, wait, or
  mildly accelerate, but it may not change the root's spatial/topological maneuver.
- Give every conflicting root the same oracle search budget before filling any compact
  neural-response slots.  Global top-R truncation can no longer create false negative
  recovery labels.
- Store explicit `response_root_index` and `root_affinity`.  Legacy/global responses use
  a soft 1/3/5/8-second affinity and expose target confidence instead of forcing a brittle
  full-horizon nearest-root label.
- Supervise continuous `root_low_safe_score`, `root_target_confidence`,
  `root_min_safe_burden`, `root_recovery_mass`, and `transported_opr`.
- Define transported OPR exactly as the probability-weighted retained mass after applying
  `s=(1-c)r+cq`; the same function is used by label generation, training adaptation, and
  learned-offline evaluation.

RCOT is the mechanism-level novelty to emphasize in the paper: coercion is not merely a
candidate risk class.  It is the destruction of protected agents' probability mass over
pre-interaction natural options, and a plan is non-coercive only when conflicting roots
can be transported into low-burden safe responses without changing root identity.

### Learning and selector changes

- Add conflict-conditioned root-recovery BCE and within-pair/root ranking.  Non-conflict
  roots no longer dominate the recovery objective.
- Use soft direct root targets when available and mask low-confidence legacy assignments.
- Use symmetric class-balanced presence loss plus conflict-conditioned recovery-magnitude
  loss for aggregate root recovery.
- Keep candidate-level BCOT as a monotone aggregation of unrecovered conflict mass,
  burden tail and option shortfall; it remains a mechanism certificate, not a generic
  black-box candidate classifier.
- Disable the legacy flat candidate-certificate loss in the main v16.8 configuration.
  It duplicates the mechanism score and has not shown independent causal value.
- Retain the localized protected-pair severe veto, but make the aggregate severe veto an
  explicit ablation and default it off.
- Extend the calibration sweep to 0.98.  Class-balanced logits are ranking scores and need
  not be calibrated probabilities; a sweep ending at 0.70 was not a valid proof that no
  operating point exists.

### Immutable mechanism checkpoint

The main v16.8 planner protocol freezes SetTransport and the response decoder during
planner training and sets planner-stage response/transport auxiliary scales to zero.
The final planner therefore optimizes candidate ranking against a fixed certificate.
Unfreezing remains available only as a named ablation.  This separates:

1. whether RCOT learns the coercion mechanism;
2. whether the selector can exploit a valid certificate;
3. whether proposal coverage limits closed-loop performance.

### What is retained, deepened, or rejected

**Retain:** typed natural decoder, yaw correction, OBS capacity, mass-aware root envelope,
source-aware multi-horizon natural-root alignment, protected-priority semantics, monotone
BCOT aggregation, candidate false-safe ranking, PBTR/OPR/BTE-CVaR metrics, fail-fast
learned-offline gate, and real Waymax online evaluation.

**Deepen now:** same-root recovery through RCOT; confidence-aware root supervision;
immutable certificate-to-selector interface; protected PBTR--coverage calibration.

**Do not repeat without new evidence:** global generic-response nearest-root `q` labels;
OPR computed without the conflict-recovery term; treating safe as low burden; unconditional
aggregate severe veto; planner-stage joint retuning of a verified certificate; using the
legacy flat candidate classifier as the primary mechanism; threshold-only fixes; sparse
cached Waymax outcomes as closed-loop evidence.

**Defer until the mechanism gate passes:** coercion-aware proposal refinement.  The current
proposal bank has an AnyNCF-scene ceiling of roughly 37.7%, so it will eventually limit
recall and progress.  Changing proposals in the present isolation experiment would,
however, confound whether RCOT repaired the mechanism.  The next algorithmic phase should
add conflict-time/local-lane timing refinement around protected roots and compare it with
the unchanged proposal bank under an identical frozen RCOT certificate.

### Development gate and paper evidence

The v16.8 immediate gate is deliberately a continuation gate, not a publication claim.
Defaults require protected root AUPRC >= 0.50, protected NCF recall >= 0.30, precision >=
0.50, accepted rate >= 0.10, fallback <= 0.25, PBTR <= 0.45, and positive PBTR improvement.
These are diagnostic promotion targets, not CCF criteria.  If root AUPRC remains below
0.50 after canonical labels are regenerated, stop before Waymax and request dataset
stratification by conflict geometry, protected relation, root source, root mass, horizon,
and low-burden oracle feasibility.

Paper-level claims still require a fresh `formal_v18` rebuild, three or more seeds, exact
checkpoint attribution, paired real Waymax online rollouts, reactive-agent sensitivity,
and reporting standard closed-loop metrics together with PBTR, protected OPR, BTE-CVaR25,
NCF retention and non-coercive progress regret.  v16.8 does not guarantee SOTA or CCF-A
acceptance; it removes known target and engineering confounds and makes the core coercion
hypothesis falsifiable.

### Verification status

- pytest: **142 passed**;
- Python compileall: **passed**;
- all top-level shell scripts: **bash -n passed**;
- no v16.8 GPU training or Waymax rollout was executed in this environment;
- therefore the mechanism gate is **not yet claimed to pass**.

## v16.7 — Priority-Aware Mechanism Repair, Monotone Certification, and Paper-Grade Closed-Loop Protocol

### Evidence from the uploaded v16.6 run

The v16.6 natural foundation is healthy and is no longer the blocking stage:

- natural-basis gate: `pass=true`;
- natural-effectiveness gate: `pass=true`;
- aligned component attribution: `pass=true`, `paper_claim_ready=false`;
- OBS-capacity paired OBS gain: **0.07049 m**, 95% paired bootstrap CI
  **[0.04353, 0.09871] m**;
- mass-aware root envelope: exact squared-excess reduction **0.31165**, violation-mass
  reduction **0.05373**, with both paired confidence intervals excluding zero.

The full pipeline stopped at the held-out mechanism gate.  The calibrated budget was
0.70 with `least_violation` status and no feasible calibration point.  On the disjoint
held-out split:

- pair-witness AUPRC: **0.71293** (the pair witness is usable);
- RootTransport conflict-conditioned AUPRC: **0.22524**;
- BCOT false-safe AUPRC: **0.40640**;
- learned NCF recall: **0.20232**;
- accepted candidate rate: **0.09036**;
- fallback rate: **0.23464**;
- selected false-safe rate: **0.47207**, versus **0.59737** for conventional safety.

This is a downstream transport/certificate failure, not a natural-decoder failure.  The
budget sweep from 0.05 to 0.70 contained no operating point satisfying recall, coverage,
fallback and false-safe constraints simultaneously; threshold relaxation alone cannot
repair the mechanism.

### Confirmed engineering and label defects

1. **Asymmetric BCOT class weighting.** False-safe examples constitute roughly 87--88%
   of discriminative candidate labels, while the old implementation used positive-class
   `pos_weight` clamped to at least one.  It could not downweight the majority class and
   shifted candidate risks upward, explaining low acceptance and high fallback even at
   the largest calibrated budget.
2. **Broken arrival-order priority label.** `_first_arrival_to_close_points` searched
   synchronous samples and returned the same timestamp for both agents.  Its arrival
   comparison could never establish who reached a shared conflict region first.
3. **Unsupported traffic-signal inference.** `controlled_by_signal` only indicates that
   a lane is signal-controlled; without live phase association it does not establish
   right-of-way.  The old rule injected systematic priority noise.
4. **Root-assignment mismatch.** Training and evaluation used duplicated alignment logic
   and full-horizon ADE, permitting source-root swaps.  v16.7 shares a multi-horizon,
   source-aware alignment cost in training and evaluation.
5. **Proposal/paper mismatch.** The implementation uses a finite kinematic primitive
   bank, not route-conditioned lattice-MPC.  Fixed lateral and terminal primitives can
   leave the local road corridor and contributed to the high cached off-road fraction.
6. **Sparse outcome evidence.** Only about 23.9% of selected held-out candidates had
   attached Waymax outcomes and no finite log-divergence labels were available.  These
   cached outcomes are auxiliary diagnostics, not a substitute for a full online
   closed-loop run.

### Mechanism repair

- Replace one-sided candidate BCE with symmetric inverse-frequency class balancing.
- Separate protected-priority BCOT risk from the all-critical global diagnostic risk.
- Preserve mechanism interpretability with positive, normalized, monotone weights over
  unrecovered conflict mass, burden-tail activation and option shortfall.  No generic
  candidate classifier is allowed to bypass the COWP mechanism.
- Add candidate-level within-scene ranking so false-safe candidates must score above
  non-coercive feasible candidates.
- Add closing speed, near-conflict fraction and swept midpoint clearance to root
  geometry; add root conflict ranking supervision.
- Blend learned priority probability with corrected rule labels, while keeping the
  all-critical diagnostic separate from the protected hard veto.
- Use one shared source-aware multi-horizon root alignment implementation for training
  and evaluation.

### Priority-aware semantics and metrics

The hard feasibility set now protects agent-priority and explicitly negotiated/equal
relations.  Ego-priority relations remain in the all-critical diagnostic but do not
create the same veto.  This prevents the priority-aware motivation from degenerating
into a universal "never inconvenience anyone" rule.

Primary mechanism metrics are reduced to a small set:

- **PBTR:** protected-priority burden-transfer rate;
- **protected OPR:** retained same-root low-burden probability mass;
- **BTE-CVaR25:** worst-quartile severity of the worst protected relation per scene;
- **NCF scene retention:** whether an existing non-coercive proposal survives the
  certificate;
- **non-coercive progress regret:** selector loss relative to the best available NCF
  proposal;
- **PBTR--coverage curve:** burden transfer versus certified scene coverage.

Global FSR remains a stress diagnostic. CBS, HBCR, WLA and MTA are retained only as
appendix or debugging metrics unless future experiments show unique explanatory value.

### Proposal and data correction

- Correct independent path-arrival times and remove signal-presence right-of-way labels.
- Add a cheap local lane-corridor screen for synthetic kinematic proposals; skip the
  screen when map geometry is too sparse to define a reliable corridor.
- Keep the logged trajectory as an observed reference.
- `PREPARE_COWP_V16_7_DATA_CN.sh` builds a fresh paper-grade `formal_v17` cache, because
  corrected priority labels and map-screened proposals cannot be retrofitted into old
  v9 overlays.  Candidate Waymax replay defaults to 24 balanced candidates per scene.

### Evaluation gates

The v16.7 mechanism gate is a development-continuation gate, not a paper-claim gate. It
requires disjoint calibration/held-out partitions and checks protected NCF recall and
precision, priority BCOT/root AUPRC, accepted rate, fallback and PBTR improvement. Global
false-safe metrics remain anti-degeneration diagnostics. Publication claims additionally
require at least three independent seeds and a real online Waymax run.

### Efficiency changes

- Vectorize hard root-ranking loss over the root axis instead of Python loops over
  batch/candidate/agent triples.
- Compute source-aware root alignment once and reuse it across transport losses.
- Retain v16.6 frozen-backbone, trainable-parameter-only AdamW, static DDP and bucket-view
  optimizations.
- Perform lane-corridor screening only during label/cache construction, not training.
- Reuse the validated natural checkpoint for the immediate mechanism rerun; transport
  and planner are retrained from a fresh optimizer state.
- Final package regression: **138 passed**; all shell scripts pass `bash -n`; Python
  compileall and internal TeX label/reference checks pass. The uploaded source did not
  include its `.bib` file, so citation resolution was not treated as a code failure.

### Current decision

- **Keep and deepen:** typed natural decoder, yaw fix, OBS capacity, mass-aware root
  identity, same-root OPR, hard protected-priority certificate and fail-fast gates.
- **Partially supported:** planner ranking and hard certificate; the offline certificate
  reduces HBCR substantially but remains too conservative and has no online proof.
- **Still unproven:** emergency hard projection contribution, conformal coverage,
  reactive-agent causal burden transfer, final selector quality and CCF-A/SOTA closed-loop
  performance.
- The immediate v16.7 experiment may reuse the validated v16.6 natural checkpoint to
  isolate the downstream repair. Final paper experiments must rebuild v17 data and rerun
  natural attribution, mechanism training and three-seed online evaluation.


## v16.6 — Protocol-Aligned Attribution, Exact Identity Objective, and Natural-Stage Systems Repair

### Why the v16.5 attribution gate failed

The uploaded v16.5 artifacts are numerically healthy: the strict natural-basis and
natural-effectiveness gates passed, optimizer steps were non-zero, AMP skips were zero,
and the graph remained frozen.  The failed attribution JSON nevertheless mixed two
engineering/protocol defects with the component question:

1. The main external diagnostic evaluated the selected **epoch 15** checkpoint, while
   both ablations evaluated their own **epoch 19** best checkpoints.  The result was
   therefore not an equal-training-time comparison.  At the shared validation epoch 15,
   main versus no-OBS-capacity improved OBS minADE by **0.05149 m** and overall natural
   trajectory loss by **0.02718 m**.  Main versus no-envelope improved OBS minADE by
   **0.11406 m**, overall trajectory loss by **0.03647 m**, and the exact validation
   identity penalty by **0.32689**.
2. The v16.5 gate required the probability-weighted *mean path ratio* to fall by 0.03,
   although the implemented loss optimizes the floor-smoothed probability-weighted
   **squared excess above the interior margin**.  The uploaded own-best reports showed
   only 0.01119 mean-ratio reduction, but a 0.06651 absolute reduction in violating
   probability mass (about 32.5%) and simultaneous OBS/overall improvement.  The old
   primary check was therefore not aligned with the trained objective.

Consequently, the uploaded gate failure does **not** establish that either component is
ineffective.  It establishes that the v16.5 attribution protocol was insufficient to
make the claim.  Final component evidence must be regenerated at a common checkpoint
and on paired scenes.

### Attribution correction

- All three arms are diagnosed at the epoch selected by the main model; an ablation may
  no longer substitute its own best epoch.
- Reports include a deterministic sampled-scene hash and scene-level paired metrics.
- The identity component is evaluated with the exact squared-excess objective optimized
  in training, probability mass outside the soft envelope, emergency-envelope p99, and
  prediction non-regression.  Mean path ratio remains diagnostic only.
- Paired bootstrap confidence intervals are emitted for OBS/overall error, exact excess,
  and violation mass.
- `pass` is explicitly a **development continuation gate**: it allows transport/planner
  evidence collection when isolated components are active and non-harmful.  A separate
  `paper_claim_ready` field remains false until at least three independent seeds provide
  publication-level evidence.
- `RUN_NATURAL_ATTRIBUTION_V16_6_CN.sh` can reuse existing epoch checkpoints and rerun
  only the aligned diagnostics; if an exact ablation checkpoint is absent, it retrains
  that arm only to the main-selected epoch with `save-every=1`.

### Natural-stage training-system corrections

- Permanent natural-stage freezing and inactive architecture freezing are applied before
  DDP and AdamW construction.
- The checkpoint-compatible legacy dense natural head is frozen for typed decoders.
- AdamW tracks only trainable parameters; permanently frozen graph/candidate/witness
  modules no longer allocate optimizer bookkeeping.
- Permanently frozen natural training uses static DDP with unused-parameter traversal
  disabled when supported by the installed PyTorch.
- Natural checkpoint selection now uses a component-neutral common prediction score
  (`traj`, `OBS minADE`, and branch minADE) rather than total loss or terms removed by an
  ablation.  Attribution still requires an identical epoch even with this fairer score.

### Held-out calibration and downstream evidence repair

The v16.5 full-pipeline launcher swept and calibrated the candidate BCOT budget on
`VAL_CACHE`, then reported selector/mechanism metrics on the same cache.  Moreover,
`25_verify_mechanism_effect.py` replaced the held-out method metrics with
`calibration.selection_metrics`, and `30_diagnose_bcot_result.py` did the same.  This
created calibration/evaluation leakage and could overstate AUPRC, NCF recall, accepted
rate, fallback and false-safe improvement.

v16.6 therefore:

- deterministically partitions learned-offline validation scenes by dataset index;
  remainder 0 of modulo 2 is calibration-only and remainder 1 is held-out-only;
- records partition modulus, remainder, scene count and an index SHA-256 in every
  learned-offline metrics row;
- chooses the BCOT operating point only from the calibration partition;
- evaluates COWP and internal baselines only on the disjoint held-out partition;
- rejects mechanism verification unless the deterministic partitions prove
  calibration/held-out disjointness and the held-out operating point equals the
  calibrated budget;
- prevents calibration metrics from overriding held-out metrics in readiness/gate
  reports.
- reports scene-level proposal/certificate coverage (`AnyNCF`, `AnyAccepted`,
  `AnyAcceptedNCF`, empty-certificate rate and NCF scene retention) so low accepted
  rate or high fallback can be attributed to the proposal bank versus the certificate.

This is an evaluation-validity correction, not an algorithmic performance claim.  It
may lower reported numbers, but those numbers are the ones suitable for deciding which
selector or transport component actually needs improvement.

### Paper correction

The v16.5 main text correctly introduced a probability-mass-aware multi-horizon semantic
envelope, but its appendix still described the rejected v16.4 endpoint trust ball and
referenced a deleted equation.  v16.6 makes the manuscript match the code:

- defines the pre-projection maximum path ratio and the normalized squared-excess loss;
- uses the same floor-smoothed detached mass as OPR;
- distinguishes the semantic soft envelope from the wider emergency projection;
- evaluates exact excess and violating mass in attribution;
- labels conformal expansion as an unimplemented optional extension;
- prevents single-seed continuation gates from being presented as paper-level evidence.

### Current status and decision

- **OBS capacity:** directionally supported across the common validation trajectory, but
  external paired epoch-15 diagnostics are still required before a component claim.
- **Mass-aware root envelope:** strongly supported as an active regularizer by lower
  exact validation penalty and violation mass without prediction harm; paired aligned
  diagnostics remain required for formal attribution.
- **Full pipeline:** may run only after the v16.6 aligned development gate passes.  The
  resulting transport/planner/Waymax outputs are evidence collection, not automatic SOTA.
- **Regression tests:** 119 passed; TeX compiles successfully (bibliography not bundled).

## v16.5 — Probability-Mass-Aware Natural Root Identity and Fair Attribution

### Why v16.4 was not promoted

The v16.4 natural recovery checkpoint learned successfully, and the yaw reference-frame
repair removed the v16.3 heading inconsistency.  However, the v16.4 attribution gate did
not isolate all claimed components and therefore could not support the paper claims:

- v16.4 main 8 s weighted ADE: **1.2339 m**; OBS: **2.9174 m**;
- `no_effectiveness_loss`: **1.2159 m** overall; **2.8724 m** OBS;
- `no_obs_capacity_boost`: **1.2599 m** overall; **3.0522 m** OBS;
- `no_integrated_trust_region`: **1.1700 m** overall; **2.6159 m** OBS;
- the fixed endpoint trust region reduced residual endpoint p99 from **46.709 m**
  to **20.000 m**, but worsened OBS ADE by **0.3014 m**;
- the OBS capacity boost improved OBS ADE by **0.1348 m** and is retained.

Two engineering defects invalidated a literal interpretation of the v16.4 loss ablation:

1. `train_cowp_v16_no_effectiveness_loss.yaml` removed several unrelated terms at once
   (OBS gain, neutral/priority preservation, kinematic, smoothness, mode-use and trust),
   so it was not a one-factor control.
2. Natural launchers claimed to freeze the graph backbone but silently unfroze it after
   two epochs.  Frozen epochs also inherited training-mode dropout after `model.train()`,
   making representations stochastic and forcing unnecessary graph gradients later.

### Algorithm change

v16.5 keeps the typed causal dynamics decoder and source-adaptive OBS capacity, but
replaces the fixed endpoint trust ball with **probability-mass-aware, multi-horizon root
identity preservation**:

1. Each source receives a soft multi-horizon envelope over 1/3/5/8 s displacement.
2. The soft identity loss is weighted by detached mode probability (with a small floor),
   so probability-carrying roots are constrained consistently with OPR retained-mass
   semantics and the network cannot evade the constraint by changing logits.
3. A wider emergency envelope is applied to every mode as a hard projection.  It is a
   numerical/physical guard, not the semantic definition of a natural root.
4. The unsupported explicit OBS-gain and prior-preservation bundle is disabled in the
   main v16.5 objective.  Source-adaptive capacity is tested by an isolated control.
5. Redundant velocity/yaw consistency losses are disabled because the dynamics
   integration now enforces them by construction; a small control-smoothness term remains.

Default soft endpoint budgets are OBS/NEU/PRIO = 20/8/6 m and emergency budgets are
48/16/12 m.  These are development settings and must be frozen before test evaluation;
they are not paper claims by themselves.

### Engineering and speed changes

- The graph encoder remains frozen and in deterministic `eval()` mode for the entire
  natural stage by default (`natural_graph_unfreeze_epoch=-1`).
- Frozen graph inference uses `torch.no_grad()`, avoiding graph activation storage and
  backward computation.
- 1/3/5/8 s pairwise trajectory losses are computed from one distance tensor and one
  cumulative sum instead of four repeated broadcasts.
- Diversity computation temporally subsamples the 80-step horizon (default stride 4).
- Expensive in-training base-effectiveness pair comparisons are disabled; the external
  2,000-scene learned-natural diagnostic remains the authoritative effectiveness test.
- Validation defaults to every two epochs instead of every epoch.
- Attribution is reduced from three auxiliary full trainings to two isolated controls:
  no OBS capacity and no mass-aware root envelope.
- Main and ablation runs now use the same seed, initialization, DDP topology, workers,
  prefetch, precision, epoch count and validation schedule.

### New promotion gates

The natural gate now separates prediction quality, physical sanity and root identity:

- learned 8 s quality/gain thresholds;
- source-prior non-regression;
- finite controls and kinematic consistency;
- probability-mass-weighted soft-envelope ratio and violation rate;
- projected emergency-envelope p99.

The attribution gate promotes only when:

- source-adaptive OBS capacity improves OBS without unacceptable NEU/PRIO regression;
- mass-aware root preservation reduces probability-weighted envelope violation/ratio;
- that identity improvement does not impose excessive OBS or overall prediction cost.

### Status

- **Engineering regression tests:** 115 passed in the delivery environment.
- **Algorithm performance:** not yet claimed.  v16.5 must be retrained and pass both
  natural effectiveness and isolated attribution gates before transport/planner/selector.
- **Do not reuse v16.4 attribution results as v16.5 evidence.**

## v8 — Aggregate structured certificate

- Added a threshold-connected hard certificate and a candidate-level classifier.
- Result: false-safe selection and burden decreased, but EP collapsed and fallback rose sharply.
- Rejected shortcut: threshold relaxation alone. Pair witness AUPRC remained about 0.43.

## v9 — Primitive-indexed transport supervision

- Added direct mode conflict/retain labels, response-root assignment and response auxiliary losses.
- Fixed hidden NPZ enumeration and response-root gather dimensionality.
- Seed 2026 learned-offline gate failed; no real Waymax online probe was executed.
- Evidence: witness AUPRC 0.4312, accepted NCF recall 0.1280, accepted candidate rate 0.0613.
- Failure diagnosis:
  1. transport did not receive candidate--natural trajectory geometry;
  2. `FREEZE_BACKBONE_EPOCHS=999` froze candidate/natural/witness identity modules;
  3. diffuse response slots distorted same-root recovery;
  4. mode-conflict validation BCE 0.722 was worse than the class-prior entropy baseline (~0.649);
  5. the generic candidate classifier dominated the claimed pair mechanism.

## v10-GCT — Geometry-Conditioned Transport

### Changes

- Added explicit compact candidate--natural relative geometry.
- Added balanced direct mode conflict/retain supervision and raw logits.
- Added response-root refinement and granular freeze.
- Replaced unweighted all-slot recovery with response-mixture-weighted recovery.
- Preserved natural-set auxiliary training during transport learning.

### Seed 2026 result

- `val/set_transport/mode_conflict`: **0.5074**, below the no-skill entropy baseline.
- `val/set_transport/mode_retain`: **0.2800**.
- Pair witness AUPRC: **0.7161** (v9: ~0.431).
- Candidate false-safe AUPRC: **0.9043**.
- Calibrated threshold 0.50:
  - EP 0.3913 vs conventional 0.3894;
  - fallback 0.2406 vs 0.1043;
  - OPR 0.7644 vs 0.7418;
  - HBCR 0.2928 vs 0.3970;
  - selected false-safe 0.4674 vs 0.5899;
  - accepted NCF recall 0.1267;
  - accepted candidate rate 0.0585.
- Across the entire threshold sweep, maximum accepted NCF recall was only **0.1391**.
- The learned-offline gate failed and therefore **no Waymax online probe ran**.

### What worked

1. Relative geometry solved the principal pair-conflict learnability failure.
2. Direct mode labels made the primitive transport mechanism measurable.
3. Granular freeze allowed candidate/natural/transport modules to adapt.
4. The certificate produced a real safety--efficiency trade-off rather than a disconnected threshold.

### What failed

1. Candidate feasibility still used an `any/max` reduction over up to six agents. A single
   moderate pair false positive rejected an entire candidate, causing multiplicative recall collapse.
2. Response-mixture-weighted recovery did not match the existential label semantics;
   root-recovery loss remained ~0.837 and response-root CE ~2.446.
3. The generic candidate calibrator could achieve high false-safe AUPRC without proving that
   primitive option transport caused the gain.
4. Attached sparse Waymax outcomes had unequal coverage and were not an online closed-loop result.

### Decision

- **Keep and strengthen** geometry-conditioned primitive conflict/retain learning.
- **Replace**, rather than tune, pairwise max/any candidate aggregation.
- **Do not** claim closed-loop superiority or SOTA from v10.

## v11-BCOT — Budgeted Counterfactual Option Transport (current)

### Hypothesis

The paper's object is the amount of low-burden natural option mass removed by an
ego candidate, not whether any one pair score crosses a threshold. Candidate
feasibility should therefore be a budget over transported option deficit, with a
separate high-confidence veto only for genuinely severe protected-priority pairs.

### Changes

1. **Pair transport deficit**
   - unrecovered conflicted natural mass;
   - burden excess on conflicted mass;
   - OPR shortfall.
2. **Candidate BCOT risk**
   - priority-weighted mean deficit;
   - smooth tail-risk deficit;
   - severe protected-pair probability.
3. **Budget gate**
   - threshold sweep now operates on candidate BCOT risk;
   - legacy pairwise `any/max` is retained only as an explicit ablation.
4. **Existential same-root recovery**
   - top response slots use a bounded fuzzy existential `max`;
   - duplicate or diffuse response slots cannot manufacture recovery;
   - a uniform root assignment contributes only `1/M`.
5. **Mechanism isolation**
   - the new candidate calibrator receives only transport statistics, not the generic candidate latent;
   - its final layer is zero-initialized for v10 checkpoint compatibility.
6. **Direct candidate-budget supervision**
   - disjoint NCF vs false-safe BCE;
   - within-scene NCF--false-safe ranking loss.
7. **Evaluation and fail-fast checks**
   - report `BCOT/FalseSafe_AUPRC` and `BCOT/RiskRankingPairAccuracy`;
   - gate checks the calibrated sweep point, not merely the default threshold;
   - require pair AUPRC >= 0.60 and BCOT false-safe AUPRC >= 0.65 before Waymax;
   - automatically run the pairmax ablation on the second GPU.

### Required evidence before promotion

Development gate:

- pair witness AUPRC >= 0.60;
- BCOT false-safe AUPRC >= 0.65;
- accepted NCF recall >= 0.30;
- accepted candidate rate >= 0.10;
- fallback <= 0.25;
- selected false-safe improvement >= 8 percentage points.

Paper-quality target:

- accepted NCF recall >= 0.50;
- fallback no more than conventional + 0.03;
- EP >= conventional or within a paired non-inferiority margin;
- relative selected false-safe reduction >= 25%;
- root-transport/BCOT significantly better than pairmax, candidate-only, soft-burden and planner-score ablations;
- 1000-scenario development and 5000-scenario x 3-seed paired online evaluation with confidence intervals.

## Prohibited shortcuts

- Do not claim closed-loop results from attached sparse Waymax candidate outcomes.
- Do not report SOTA from 100 scenes, one seed, or unmatched scenario subsets.
- Do not relax the gate merely to make the pipeline continue.
- Do not add log-divergence loss while finite label coverage is zero.
- Do not interpret generic candidate-classifier gains as proof of primitive transport.
- Do not repeat v10 pairwise max/any aggregation except as the registered ablation.
- Do not rebuild WOMD/tensor/transport caches for v11; the v9 sidecars are schema-compatible.

## v11-BCOT — Seed 2026 postmortem (do not promote)

### Observed result

The submitted `cowp_v11_bcot_probe100_seed2026` run stopped at the learned-offline
mechanism gate; no COWP Waymax probe or full online result was produced.

- pair witness AUPRC: **0.6808**;
- BCOT false-safe AUPRC: **0.4115**;
- BCOT within-scene NCF--false-safe ranking accuracy: **0.8306**;
- generic candidate false-safe AUPRC: **0.9003**;
- calibrated operating point (`0.40`, selected by least violation):
  - EP **0.3538** vs conventional **0.3870**;
  - fallback **0.2731** vs conventional **0.1043**;
  - OPR **0.8005** vs conventional **0.7426**;
  - HBCR **0.2529** vs conventional **0.3964**;
  - selected false-safe **0.4255** vs conventional **0.5881**;
  - accepted NCF recall **0.2011**;
  - accepted candidate rate **0.0831**.
- no sweep point met NCF recall >= 0.30 and fallback <= 0.25.

### Root causes found by code/log audit

1. **Transport-stage candidate supervision was absent.**
   `stage=witness` did not load `false_safe` or `noncoercive_feasible`, so
   `set_transport/candidate_budget` was exactly zero for every transport epoch.
2. **Option preservation was double-counted.**
   `mode_retained_low_safe` already means non-conflicting and low-burden, but the
   forward pass multiplied it again by `(1 - conflict_prob)`, systematically
   depressing OPR and acceptance.
3. **The existential burden statistic was mixture-weighted.**
   `_soft_min_burden` mixed response likelihood into a set-existence predicate,
   allowing a low-probability slot to be treated as unavailable even when it was
   a valid low-burden response.
4. **Natural-option drift invalidated root transport.**
   Validation natural set minADE rose from roughly 48 m to above 60 m during the
   witness stage.  A root certificate cannot be identified when its root
   trajectories move by tens of metres.
5. **The response bank was not root-conditioned.**
   It generated generic unordered response slots and then solved a difficult
   24-way root classification problem.  Root recovery remained about 0.86--0.88
   loss and response-root CE about 2.5--2.8.
6. **Two thresholds had been conflated.**
   The pair witness confidence threshold and candidate BCOT budget have different
   meanings but shared one evaluation parameter, making calibration ambiguous.
7. **The generic candidate certificate was not mechanism evidence.**
   The reported high candidate AUPRC came from a transport calibrator/generic
   latent path rather than a causal root-transport decision path.
8. **Dense response decoding was silently disabled outside response/all stages.**
   Enabling response trajectories in witness/planner configuration did not make
   the model decode them.

### Decision

- Keep: candidate--natural relative geometry, direct mode-conflict supervision,
  pair witness localization, burden decomposition, and budget rather than
  pairwise `any/max` aggregation.
- Replace: response-bank-defined root recovery, shared threshold calibration,
  silent missing-label behavior, and generic-certificate-controlled selection.
- Do not tune v11 further.  The dominant failures are semantic/structural, not a
  learning-rate or threshold issue.

## v12-RIOT — Root-Indexed Option Transport (current repair)

### Core algorithm change

For every critical agent and every natural option root `m`, v12 predicts the
primitive event

`T[k,i,m] = P(exists valid, safe, low-burden response under ego candidate k that preserves root m)`.

This root-indexed event is now the primary certificate.  Candidate BCOT risk is
computed from unrecovered natural-option mass, burden excess, OPR shortfall, and
a separate severe protected-priority veto.  The unordered response bank remains
only an auxiliary reconstruction/interpretability head.

### Implemented fixes

1. Load candidate `false_safe`, `noncoercive_feasible`, conventional-safety,
   utility, neutral/logged flags and beta labels in witness as well as planner
   stages.
2. Fail fast when required candidate or transport labels are missing; never
   silently replace missing supervision by all-negative targets.
3. Add candidate-budget label coverage, NCF rate and false-safe rate diagnostics;
   invalidate witness checkpoints with zero budget coverage.
4. Factor mode output into conflict and conditional low-safe retention, avoiding
   duplicate conflict suppression in OPR.
5. Supervise direct natural-root recovery by scatter-reducing explicit response
   slot labels (`valid & safe & low_burden & response_root_index`).
6. Preserve legacy response-root reconstruction only as
   `response_root_exist_aux` with low auxiliary weight.
7. Remove response-mixture probability from the existential soft minimum burden.
8. Add a dedicated natural-basis repair stage, freeze graph during natural
   repair, freeze the repaired natural module during witness training, and set
   witness natural auxiliary weight to zero.
9. Separate `pair_witness_threshold` from `candidate_transport_budget` throughout
   offline and Waymax evaluation.
10. Add one-pass BCOT budget sweep and `31_calibrate_bcot_budget.py`.
11. Make the main selector transport-pure: generic candidate certificate is
    retained only as an ablation/diagnostic; rule, action and rollout risks remain
    explicit safety shields.
12. Wire dense response decoding correctly for witness/planner when enabled; the
    default v12 run keeps it disabled to avoid unnecessary memory because RIOT
    does not depend on dense decoded response trajectories.
13. Add `32_gate_natural_basis.py`; transport training is prohibited unless the
    repaired natural basis passes the registered minADE thresholds.
14. Add pairmax and Pareto-frontier ablation configs and a two-GPU v12 driver.

### Validation performed before release

- `python -m compileall`: pass;
- full test suite: **70 passed**;
- `bash -n run_cowp_v12_dual_gpu.sh`: pass.

### v12 promotion gates

Before any online claim:

- natural set minADE <= 12 m;
- branch, neutral and priority minADE <= 15 m;
- candidate-budget supervision coverage > 0;
- pair witness AUPRC >= 0.60 (paper target >= 0.70);
- BCOT false-safe AUPRC >= 0.65 (paper target >= 0.75);
- direct conflict-conditioned root-transport AUPRC >= 0.65 (paper target >= 0.75);
- accepted NCF recall >= 0.30 (paper target >= 0.50);
- accepted candidate rate >= 0.10 (paper target >= 0.20);
- fallback <= 0.25 during development and <= conventional + 0.03 for paper;
- selected false-safe reduction >= 8 percentage points during development and
  >= 25% relative for paper;
- EP non-inferior to conventional at the paired scenario level.

### Required ablations for the paper

- RIOT/BCOT full model;
- legacy pairmax aggregation;
- response-bank-only recovery (disable direct root recovery);
- generic candidate certificate only;
- no natural pretraining/freeze;
- no observational, neutral or priority-preserving branch;
- no option preservation;
- soft burden cost only;
- oracle natural roots and oracle response roots as upper bounds.

### Additional prohibited shortcuts

- Do not run online evaluation when natural-basis or mechanism gates fail.
- Do not select a `least_violation` BCOT budget for a paper table; that status is
  diagnostic only.
- Do not call RIOT effective unless direct root recovery beats the response-bank
  auxiliary and pairmax ablations under paired evaluation.
- Do not claim reactive multi-agent evaluation while
  `actual_non_ego_policy=logged_replay`.


### v12 release hardening and direct mechanism observability

15. Learned-offline evaluation now requires compact response/root labels and reports
    `RootTransport/LowSafeExist_AUPRC`,
    `RootTransport/ConflictConditioned_AUPRC`,
    `RootTransport/ConflictConditioned_Recall@0.5`, the legacy auxiliary
    response-bank AUPRC, and natural-root assignment minADE.  This prevents a
    candidate-level BCOT score from being mistaken for proof that root-indexed
    option transport was learned.
16. The mechanism verifier now has an independent direct root-transport AUPRC
    gate.  The v12 driver uses 0.65 for development and blocks Waymax probe/full
    runs unless the saved mechanism report has `pass=true`.
17. Witness/planner supervision preflight now probes 32 evenly spaced cache items
    instead of the first eight, reducing shard-order false negatives for rare
    false-safe/NCF labels.
18. External repaired natural checkpoints must provide `NATURAL_HISTORY`; the
    natural-basis gate can no longer accidentally inspect an unrelated/missing
    local history file.
19. `planner_eval` loads only compact response/root targets, never dense response
    trajectories/components, so direct mechanism evaluation remains memory-bounded.

Validation after hardening:

- `pytest -q`: **70 passed**;
- `python -m compileall cowp tests`: pass;
- `bash -n run_cowp_v12_dual_gpu.sh`: pass.

## v13-TKN — Temporal-Kinematic Natural Basis and Protocol-Safe Closed Loop

### Triggering evidence from `cowp_v12_riot_probe100_seed2026`

The supplied v12 result directory contains only copied configs and
`checkpoints/natural/history_natural.json`; it contains no natural checkpoint,
transport/planner history or checkpoint, learned-offline report, probe result, or
Waymax result.  The run therefore stopped before RIOT was trained or evaluated.

The best recorded natural-basis row is epoch 7:

- validation set minADE: **41.2888 m**;
- branch minADE: **42.0075 m**;
- observational / neutral / priority minADE: **43.6674 / 40.7546 / 39.9407 m**;
- neutral consistency: **53.9360 m**;
- source CE: **1.0215**, improving only **0.00185** over the run;
- priority BCE: **0.6673**, improving only **0.00154**;
- composite checkpoint score: **42.6445**.

A v13 recheck fails every registered v12 natural gate.  This is not evidence
against RIOT itself, because RIOT never received an identifiable natural root
basis and no transport/planner/online result exists.

### Root causes

1. The old natural trajectory head projected one agent token directly to all
   `M x T x 7` values.  Its mode embedding did not participate in trajectory
   decoding and there was no explicit time representation or motion prior.
2. Natural alternatives are unordered, but source and priority semantics were
   supervised mainly as aggregate distributions rather than on the trajectory
   mode matched to each ground-truth root.  Root identity was therefore weakly
   identifiable.
3. Neutral consistency used source-neutral probability without the mode mixture
   probability, so even negligible modes contributed equally; the corresponding
   validation error stayed near 54 m.
4. The v12 gate had no dataset/oracle diagnostic and no short-horizon metrics,
   making it impossible to distinguish coordinate/track misalignment from a bad
   decoder using the result package alone.
5. Online `ClosedLoopPredFSR/CBS/OPR` are predictions of the same network that
   selects the candidate.  They are health diagnostics, not counterfactual
   ground truth and cannot independently validate the paper mechanism.
6. The real Waymax evaluator controls only the SDC.  Non-ego agents follow log
   playback.  The previous config described this correctly but the field was not
   enforced or emitted in the result JSON.
7. Replanning had no plan-stitching/hysteresis term, allowing high-frequency
   candidate switching even when every individual lattice trajectory was valid.
8. `metrics_from_labels` had a latent missing-field crash caused by an undefined
   `valid` variable in fallback burden/OPR allocation.

### Implemented changes

1. Added `temporal_kinematic` natural decoding:
   - constant-velocity anchor baseline;
   - explicit mode and time embeddings connected to trajectory generation;
   - bounded cumulative position/yaw residuals and bounded velocity/size offsets;
   - zero-initialized residual head, so a new model starts exactly at the
     kinematic baseline;
   - `legacy_linear` retained for a controlled ablation.
2. Added matched-mode semantic supervision: nearest-trajectory matching now binds
   source CE and priority BCE to the recovered natural root.
3. Corrected neutral consistency weighting to use both mixture probability and
   neutral-source probability.
4. Added validation natural minADE at 1 s, 3 s, 5 s, and 8 s.
5. Hardened `32_gate_natural_basis.py` with semantic-learning checks and optional
   kinematic-oracle comparison.
6. Added `33_diagnose_cache_alignment.py` for raw/transport tensor equality,
   critical track-to-input mapping, current/future coordinate consistency,
   transport-root ranges, and Waymax outcome/log-divergence coverage.
7. Added `34_diagnose_natural_oracles.py`, a 15-trajectory acceleration/yaw-rate
   bank that reports source-stratified 1/3/5/8 s oracle minADE.
8. Added online plan-continuity risk after the hard feasibility/frontier filter.
   It can reorder feasible candidates but cannot make a rejected candidate
   feasible.  The same regularizer is applied to internal baselines.
9. Waymax outputs now explicitly include:
   - `actual_non_ego_policy=logged_replay`;
   - `reactive_mixture_implemented=false`;
   - `mechanism_ground_truth_available_online=false`;
   - proxy-only markers for online model-predicted mechanism diagnostics.
   The evaluator raises an error if a config falsely requests a reactive non-ego
   policy without an implemented actor wrapper.
10. Fixed the missing optional witness-array crash in label-only metrics.
11. Added `STOP_AFTER_STAGE=natural|transport|planner|offline|probe` to make each
    promotion gate independently executable.  A natural-only run no longer
    requires transport caches unless transport diagnostics/later stages are
    requested.
12. Added v13 configs and `run_cowp_v13_dual_gpu.sh`.

### What is and is not validated

Validated locally without WOMD/Waymax data:

- YAML parsing;
- Python compilation;
- driver shell syntax;
- temporal decoder kinematic initialization;
- proxy protocol markers and missing-field metric fallback;
- full test suite: **73 passed**.

Not yet validated, because the uploaded package does not include the actual
raw/transport caches, the initialization checkpoint, or a Waymax installation:

- real cache coordinate/track alignment;
- v13 natural-basis convergence;
- RIOT direct root-transport AUPRC;
- planner selection gains;
- logged-replay Waymax metrics;
- reactive-agent counterfactual burden reduction.

### Promotion policy

- Do not continue beyond natural training unless the v13 natural gate passes.
- Do not run Waymax unless learned-offline mechanism verification passes.
- Do not use sparse attached candidate outcomes as the sole planner objective or
  as a full validation result.
- Keep log-divergence loss/reporting disabled until finite label coverage is
  measured; the supplied cache audit reports zero finite log-divergence labels.
- Do not call logged-replay Waymax evaluation reactive or use model-predicted
  online FSR/CBS/OPR as causal ground truth.
- A paper-ready result requires paired full-validation comparisons, confidence
  intervals, and a separately labelled reactive-agent protocol.
13. Keep the sparse Waymax outcome head as auxiliary supervision, but disable it
    in the primary v13 selector by default (`candidate_selection_outcome_weight=0`,
    `candidate_outcome_risk_mix=0`, online penalty `0`, threshold `1.10`).  It must
    be re-enabled only as a registered ablation after checkpoint-selected
    validation replay coverage is reported.

## v14-TNOB — Typed Natural Option Basis, Exact Anchor Preflight, and Fast Diagnostics

### Triggering evidence from `cowp_v13_temporal_riot_seed2026`

The v13 natural gate failed at epoch 29:

- validation set minADE: **35.8454 m**;
- 1 s / 3 s / 5 s minADE: **34.5138 / 34.7494 / 35.1241 m**;
- branch minADE: **37.7621 m**;
- OBS / NEU / PRIO minADE: **40.9817 / 36.1811 / 32.9040 m**;
- source CE: **0.9151**, improvement only **0.00717**;
- priority BCE: **0.3121**, the only semantic gate that passed;
- neutral consistency: **37.6742 m**, improvement only **0.0491 m**.

The label-side kinematic oracle on 2,000 validation scenes achieved
**0.283 / 0.608 / 1.204 / 2.466 m** at 1/3/5/8 s.  The nearly constant
~35 m model error from 1 s through 8 s is therefore inconsistent with ordinary
long-horizon forecast divergence and strongly suggests a model-facing critical
row/current-state anchor/frame mismatch.  The exact subcase cannot be proven
without the server caches; v14 adds a hard exact-path preflight rather than
silently guessing.

Independently of that suspected data-path issue, v13 had a confirmed structural
failure: all 24 roots began as the same constant-velocity curve, while global
nearest-neighbour matching allowed OBS, NEU and PRIO targets to compete for any
mode.  This makes semantic root identity non-identifiable and invalidates later
same-root transport even if aggregate displacement improves.

### Implemented changes

1. Added **Typed Natural Option Basis (TNOB)** in
   `cowp/models/natural_decoder.py`:
   - fixed 8/8/8 OBS/NEU/PRIO mode identities for the default 24 modes;
   - distinct analytic acceleration/yaw-rate/speed-offset prototypes;
   - zero-initialized, gated and physically bounded temporal residuals;
   - explicit `mode_source`, `base_traj` and `residual` outputs;
   - legacy temporal and linear decoders retained for ablation.
2. Replaced cross-source global matching with source-restricted matching in
   `cowp/models/losses.py`.  Trajectory, mixture, branch, short-horizon,
   source and priority supervision now share the same typed assignment.
3. Added untyped geometric coverage as a diagnostic and a typed-vs-untyped gap
   gate, preventing semantic typing from hiding lost trajectory coverage.
4. Added analytic-prior preservation (`base_deviation`) and residual magnitude
   regularization, plus explicit 1/3/5 s loss terms.
5. Added `--eval-before-train`: epoch -1 analytic TNOB is evaluated and can be
   retained as the best checkpoint if learning degrades it.
6. Added `--reset-checkpoint-prefix`; the v14 driver resets `natural_decoder`
   when loading an old planner checkpoint while retaining the graph backbone.
7. Replaced full-stage graph freezing with a configurable two-epoch natural
   warmup followed by low-LR joint adaptation; added gradient clipping.
8. Added `35_diagnose_model_anchor.py`, which follows the exact production path:
   `TorchCOWPDataset -> _agent_history_from_batch -> input_index ->
   _safe_critical_indices -> _critical_anchor7 -> typed_kinematic_basis`.
   It hard-fails on excessive unmapped critical agents, first-step anchor error,
   or 1/8 s typed-basis error.
9. Rewrote `33_diagnose_cache_alignment.py` for selective NPZ member reads,
   symlink/samefile shortcuts, sampled hashing and threaded I/O.
10. Rewrote `34_diagnose_natural_oracles.py` to vectorize root/horizon distance
    calculation and support threaded loading.
11. Hardened the natural gate with absolute-or-improvement semantic criteria and
    the typed/untyped coverage check.
12. Added v14 configs, driver, execution guide and typed-basis unit tests.

### Promotion gates

Training is blocked unless exact model-facing preflight satisfies:

- critical unmapped/invisible rate <= 2%;
- first-step GT versus model-CV-anchor p90 <= 5 m;
- typed basis minADE@1s <= 3 m;
- typed basis minADE@8s <= 8.5 m.

Natural promotion additionally requires:

- set minADE <= min(12 m, oracle@8s + 6 m);
- branch/NEU/PRIO minADE <= 15 m;
- source CE <= 0.30 or registered learning improvement;
- priority BCE <= 0.45 or registered learning improvement;
- neutral consistency <= 10 m or registered learning improvement;
- typed minus untyped minADE <= 4 m.

No transport, planner or Waymax result may be promoted when these gates fail.

### Validation and limitations

Validated locally:

- full test suite: **76 passed**;
- Python compilation: pass;
- `bash -n run_cowp_v14_dual_gpu.sh`: pass;
- typed mode identities, non-collapsed analytic endpoints and source-restricted
  matching unit tests: pass.

Not validated locally because the uploaded artifacts omit the actual raw and
transport-v9 caches, initialization checkpoint, Waymax runtime and standalone
`cache_sufficiency_full.json`:

- which concrete mapping/frame bug caused the v13 ~35 m translation;
- v14 A30 convergence and gate pass;
- direct root-transport mechanism gains;
- full logged-replay or reactive-agent closed-loop performance;
- SOTA status.

### Additional prohibited shortcuts

- Do not weaken the anchor preflight or natural gate to continue a run.
- Do not restore global cross-source matching in the primary method; keep it only
  as a registered ablation.
- Do not claim TNOB alone proves non-coercive feasibility; it establishes root
  identifiability, while transport and independent reactive evidence prove the
  mechanism.
- Do not claim SOTA from a 100-scene probe or from sparse attached outcomes.
13. Removed stale top-level duplicate Python trees that were not included by the
    `cowp*` package configuration.  The authoritative implementation is now
    unambiguously under `cowp/`, preventing fixes from being applied to inert
    copies.

## v15-CNOB — Causal Natural Option Basis, Protocol Integrity, and OBS Decontamination

### Triggering evidence from `cowp_v14_typed_natural_seed2026`

The uploaded v14 run did not produce `natural_basis_gate.json`, but this was not
an ordinary metric failure. `NEXT_RUN_COMMANDS_V14_CN.txt` mixed executable shell
commands with Chinese prose and was invoked through `bash`; the prose was parsed
as commands. The driver then retained an existing natural checkpoint although
`history_natural.json` was absent, so the hard gate could not be reconstructed by
the run itself.

The validation rows in `logs/train_natural_ddp.log` were recovered into a proper
history file. Under the original v14 thresholds, the best epoch (15) passes:

- typed set minADE@8 s: **1.8974 m**;
- minADE@1/3/5 s: **0.2289 / 0.4480 / 0.8951 m**;
- branch minADE: **2.9117 m**;
- OBS / NEU / PRIO minADE: **4.6060 / 1.1351 / 1.2998 m**;
- neutral consistency: **1.8647 m**;
- priority BCE: **0.3495**.

However, v14 is not promoted by the stricter v15 geometric gate. It fails on the
OBS branch (**4.6060 m > 4.0 m**) and branch spread
(**4.6060 - 1.1351 = 3.4708 m > 3.0 m**). Near-zero source CE is no longer
counted as evidence of learning because typed root identities are structurally
hard-coded.

The epoch trace also shows that the learned temporal residual was nearly inert:
aggregate quality jumped when the analytic typed basis/graph path became active,
then remained almost flat; final base deviation was approximately 0.017 m and
residual L2 approximately 8e-4. Thus v14 success was dominated by hand-designed
kinematic prototypes rather than learned scene-conditioned natural behavior.

### Confirmed engineering defects

1. `COWPModel._agent_history_from_batch` could reconstruct encoder input from
   `cowp/natural/traj` when real current/history tensors were missing. This is a
   direct future-label leak.
2. the online policy could silently fall back to Waymax `log_trajectory`, which
   contains privileged future states;
3. missing `state/is_sdc` could silently make row zero the ego, invalidating
   ego-centric transforms and all downstream critical-agent indexing;
4. observational perturbations shifted future positions without consistently
   repairing velocity and heading, producing side-slip and first-step
   discontinuities;
5. natural alternatives declared map compliance without performing a map check;
6. the logged OBS future could already contain ego-induced yielding, so it was
   not a clean sample of behavior under the absence of ego pressure;
7. a label-space candidate-safety complement was exported as `CR`, although it
   was not a simulator-measured closed-loop collision/offroad rate;
8. logged-replay non-ego agents were described by the paper as a reactive
   mixture, although the current evaluator does not implement that mixture;
9. checkpoint-only skip logic could preserve an incomplete natural stage;
10. documentation and executable shell were mixed in one file.

### Implemented changes

1. Added a strict causal input contract:
   - future-label reconstruction is disabled by default;
   - reported runs require an explicit SDC marker;
   - absent real history/current tensors or SDC identity hard-fail.
2. Added a strict Waymax future-access contract:
   - main policies use only simulated/current/history state;
   - `log_trajectory` is inaccessible except in an explicitly named oracle
     ablation;
   - causal constant-velocity non-ego prediction is used by the model-facing
     online wrapper when no learned reactive predictor is available.
3. Reworked observational trajectory perturbation:
   - every alternative starts continuously from the current state;
   - lateral displacement uses a zero-origin smooth transition;
   - heading and velocity are recomputed from the transformed path;
   - invalid/non-finite motion is rejected.
4. Added map-aware natural-option filtering using the available road/lane point
   cloud, with type-aware distance thresholds and explicit verification fields.
5. Added observational decontamination:
   - estimate whether the logged agent decelerates/loses progress near ego;
   - compare logged clearance with an ego-neutral continuation;
   - produce an `obs_contamination` score;
   - downweight or reject highly pressure-contaminated OBS alternatives.
6. Introduced the **Causal Natural Option Basis (CNOB)** decoder profile:
   - retain source-stable OBS/NEU/PRIO roots needed by same-root transport;
   - allocate more bounded residual capacity to OBS, the empirically weak branch;
   - preserve stronger analytic priors for NEU/PRIO;
   - treat source identity as structure rather than an artificial source-CE win.
7. Added source-specific prior-deviation losses and diagnostics, with a lower OBS
   regularization coefficient and stronger NEU/PRIO preservation.
8. Hardened the natural gate with absolute OBS quality and branch-spread checks.
9. Split metric namespaces:
   - closed-loop CR/offroad are accepted only from Waymax standard metrics;
   - label-space safety is named `OfflineConventionalUnsafeRate`;
   - `CR_proxy_deprecated` is retained only for old result readers.
10. Added `36_audit_causal_protocol.py`, checking leakage, SDC identity, metric
    provenance, map filtering, OBS decontamination, reactive-protocol honesty,
    mapping completeness, root-index range, and missing log-divergence policy.
11. Added pure executable scripts:
    - `prepare_cowp_v15_data.sh` rebuilds labels/caches/outcomes/transport overlay;
    - `NEXT_RUN_COMMANDS_V15_CN.sh` runs tests, data preparation, and the v15
      training/evaluation driver without prose being parsed as shell;
    - `run_cowp_v15_dual_gpu.sh` treats checkpoint+history as one atomic natural
      artifact and retrains when either is incomplete.
12. Rebuilt Pareto and pairmax ablation configs on top of the same v15 causal
    natural-label settings, so ablations do not reintroduce old label defects.
13. Added five causal-integrity regression tests. The local suite now reports
    **81 passed**.

### v15 promotion gates

Natural-stage promotion requires all of the following:

- typed set minADE@8 s <= min(8.5 m, label oracle + 6 m);
- branch-weighted minADE <= 3.0 m;
- OBS minADE <= 4.0 m;
- max(OBS, NEU, PRIO) - min(OBS, NEU, PRIO) <= 3.0 m;
- NEU and PRIO minADE <= 2.0 m;
- minADE@1 s <= 3.0 m and minADE@3 s <= 5.0 m;
- neutral consistency <= 3.0 m;
- priority BCE <= 0.45;
- typed-untyped gap <= 3.0 m.

For paper-facing runs, the preferred target is stricter: OBS <= 3.5 m,
branch spread <= 2.0 m, and typed set minADE <= 1.5--1.7 m, while preserving
NEU/PRIO quality.

### What is and is not validated

Validated locally without WOMD/Waymax runtime:

- Python compilation;
- YAML parsing and driver shell syntax;
- 81 unit/regression tests;
- the static/report-backed causal protocol audit;
- reconstruction of the v14 natural history and original gate;
- expected rejection of v14 by the stricter v15 OBS/branch-spread gate.

Not validated in the supplied environment:

- v15 label/cache regeneration on the full WOMD data;
- v15 natural convergence and actual OBS improvement;
- transport/planner retraining;
- full-validation Waymax closed-loop CR/offroad/progress;
- reactive non-ego evaluation;
- SOTA status.

### Additional prohibited shortcuts

- Do not enable `allow_label_only_state_fallback` in a reported run.
- Do not disable `require_explicit_sdc_index` to bypass malformed caches.
- Do not use `log_trajectory` outside a clearly labelled oracle diagnostic.
- Do not report `OfflineConventionalUnsafeRate` or `CR_proxy_deprecated` as
  closed-loop collision rate.
- Do not call logged replay a reactive-agent experiment.
- Do not claim the hard-coded source identity or its near-zero CE as novelty.
- Do not continue beyond the natural gate when OBS or branch spread fails.
- Do not claim SOTA before full-validation, multi-seed, paired confidence-
  interval comparisons under both logged-replay and independent reactive-agent
  protocols.

## v15.1 — audited reuse of v9 transport caches (2026-07-23)

1. Reclassified data compatibility into two explicit protocols:
   - `v15`: regenerated causal-label protocol with OBS decontamination and map filtering;
   - `v9_reuse`: v15 model/engineering stack trained on the existing v9 labels.
2. Changed `NEXT_RUN_COMMANDS_V15_CN.sh` to reuse the existing raw Waymax caches and
   `transport_v9` overlays by default. It no longer runs index, label generation,
   tensor-cache construction, Waymax replay, outcome attachment, or transport augmentation.
3. Preserved the former full rebuild route as
   `NEXT_RUN_COMMANDS_V15_REBUILD_FULL_CN.sh`.
4. Added `38_gate_cache_reuse.py`, which checks current server-side file counts,
   raw/overlay alignment, required training fields, explicit SDC identity, critical-agent
   mapping, response-root ranges, split overlap, v15 label materialization, and logdiv state.
5. Added `CHECK_V9_CACHE_ONLY.sh` to run the complete raw-cache sufficiency scan and
   the independent v9 overlay gate without launching training.
6. Hardened `36_audit_causal_protocol.py` so `engineering_pass` is separated from
   `full_v15_label_protocol_pass`. A v9-reuse run may be engineering-valid while
   correctly refusing to claim that v15 causal labels were materialized.
7. Added a data-protocol manifest to every v15 run. Paper-facing result aggregation
   must not merge `v9_reuse` and full `v15` experiments as if they used identical labels.
8. Disabled silent use of missing Waymax log-divergence supervision. Existing safety
   replay has zero finite logdiv targets; collision/offroad may remain an auxiliary loss,
   while final closed-loop metrics must come from online Waymax.
9. Added `tests/test_v15_v9_cache_reuse.py`. Full local suite: **82 passed**.
10. Documented the count discrepancy between the older full cache report (14,640 train)
    and the later v14 alignment result (20,440 train). The new default script therefore
    audits the current server directory before training and defaults to a 20,000-scene
    minimum rather than trusting stale reports.

## v16 — CNOB dynamics and evidence-gated attribution (2026-07-23)

### Triggering evidence

The uploaded v15/v9-reuse run stopped before training in the mandatory model-anchor
preflight. `35_diagnose_model_anchor.py` maintained its own decoder string whitelist
and rejected `typed_causal_residual`, even though `NaturalDecoder` accepted that
name. Consequently the run produced no natural checkpoint/history, no planner or
selector result, and no online Waymax result. None of the claimed v15 model changes
were empirically evaluated by that run.

### Engineering fixes

1. Centralized typed decoder identity in `NaturalDecoder.uses_typed_basis` and
   `uses_dynamic_residual`; preflight and protocol audit now query the model rather
   than maintaining independent string lists.
2. Corrected cache-sufficiency enumeration so hidden sampler/metadata `.npz` files
   are not counted as scenarios. This removes the false `20441` versus `20440`
   discrepancy and the associated one-scene missing-label warning.
3. Fixed candidate-certificate fallback semantics: when
   `candidate_cert_allow_hybrid_fallback=false`, no generic certificate is silently
   mixed into the set-transport score.
4. Added automatic learned-natural diagnostics and a hard effectiveness gate before
   transport/planner training.
5. Added full Waymax delta summaries against both conventional safety and planner-
   score-only baselines.

### Algorithm changes: Causal Natural Option Basis dynamics decoder

1. Replaced the paper-facing `typed_causal_residual` alias with
   `typed_causal_dynamics` / `cnob_dynamics`.
2. The learned branch predicts bounded local longitudinal/lateral acceleration,
   jerk, and yaw-rate corrections around the typed OBS/NEU/PRIO basis.
3. Position and velocity are obtained by integration; heading is derived from the
   integrated velocity when moving; box dimensions are invariant. Independent,
   mutually inconsistent position/yaw/velocity/size residual heads are prohibited.
4. Zero initialization exactly reproduces the analytic typed basis, preserving a
   safe and interpretable initialization.
5. OBS receives configurable additional control capacity. The
   `natural_obs_capacity_scale=0` ablation gives OBS the same control bounds as NEU
   while retaining PRIO protection, enabling a controlled capacity claim.

### Natural-loss and evidence changes

1. Source-restricted branch minADE now uses `cowp/natural/weight`; v15 contamination
   downweights are therefore respected when a true v15 dataset is used.
2. Added direct OBS-improvement shortfall loss relative to the exact analytic basis.
3. Added NEU/PRIO preservation losses, finite-difference velocity consistency,
   velocity-heading consistency, control smoothness, and source-specific mode-usage
   entropy.
4. Added `39_diagnose_learned_natural.py`, which measures learned versus analytic
   minADE by source and horizon, residual magnitude, controls, physical consistency,
   and effective mode usage.
5. Added `40_gate_natural_effectiveness.py`. A natural checkpoint must now improve
   the analytic basis, improve OBS, preserve NEU/PRIO, remain physical, use multiple
   modes, and keep residuals bounded. Passing only the old absolute gate is no longer
   sufficient.
6. Added controlled component attribution:
   - `train_cowp_v16_no_effectiveness_loss.yaml` removes the new loss bundle;
   - `model_cowp_v16_no_obs_capacity.yaml` removes only the OBS capacity boost;
   - `RUN_NATURAL_ABLATIONS_V16_CN.sh` trains both;
   - `41_compare_natural_ablations.py` hard-gates the new-loss and OBS-capacity claims.

### Planner/selector changes

1. Planner checkpoint selection now prioritizes the claimed set-transport mechanism,
   candidate budget, and same-root recovery rather than allowing the generic
   candidate classifier to dominate.
2. Sparse attached Waymax collision/offroad labels remain auxiliary only; their loss
   weights are reduced and `outcome_logdiv=0` remains mandatory because finite
   log-divergence coverage is zero.
3. Online evaluation remains honest logged-replay non-ego Waymax. It can establish
   SDC CR/offroad/progress effects, but it is not a reactive-agent burden experiment.

### Data decision

- Reuse `tensor_cache_*_waymax_transport_v9` for the next v16 model/loss/capacity,
  planner, selector, and online Waymax experiment.
- Do not claim v15 OBS decontamination or map-filtered label generation from v9 data.
- A true v15 dataset is required only after the v16 model passes, to validate the
  revised natural roots/weights and the paper's causal-label contribution. Prefer a
  targeted OBS/interaction-heavy pilot before a full rebuild.

### v16 promotion order

1. exact-path model-anchor and causal-protocol preflight;
2. legacy absolute natural gate;
3. learned-versus-analytic natural effectiveness gate;
4. controlled natural component attribution gate;
5. transport/planner mechanism verification;
6. paired online Waymax probe;
7. full-validation multi-seed Waymax evaluation;
8. true-v15-label pilot/full rebuild and an independent reactive-agent protocol for
   the final causal burden claim.

### Local validation

- Python compilation: pass;
- executable shell syntax: pass;
- causal engineering audit: pass;
- full v15 label protocol on v9 caches: intentionally false;
- unit/regression suite: **90 passed** after v16 component-ablation coverage.

### Prohibited claims

- Do not call the uploaded failed v15 run evidence that the decoder, new losses,
  planner, selector, or Waymax metrics improved.
- Do not attribute a main-model gain to the new loss or OBS capacity without the two
  controlled natural ablations.
- Do not call v9-reuse a true v15 causal-label experiment.
- Do not call logged-replay non-ego Waymax a reactive-agent evaluation.
- Do not claim SOTA before full-validation, multi-seed paired results and confidence
  intervals are available.

## v16.1 — natural-loss hotfix, provenance isolation, and fast diagnostics (2026-07-24)

### Triggering evidence

The v16 run passed cache alignment, causal engineering audit, natural oracle, and the
exact model-facing anchor preflight. It then failed before optimization at validation
epoch `-1`, batch `0`, inside `_natural_mode_usage_loss`. The real tensor shapes were
approximately `[B=5, A=6, M=24, R=24]`. The implementation used
`psrc[..., 0]`, which removed the mode axis and created a `[B,A]` mask; broadcasting
that mask against `[B,A,M,R]` failed with `A=6` versus `M=24`.

Because the failure occurred before the first validation batch completed, the run
contains no trained natural checkpoint/history and supplies no evidence about decoder,
new-loss, OBS-capacity, planner, selector, or online Waymax gains.

### Engineering fixes

1. Fixed source-specific mode-usage masking to retain `[B,A,M]` and aggregate only
   eligible modes. The implementation also supports an explicitly expanded
   batch-dependent `mode_source` tensor rather than assuming a one-dimensional buffer.
2. Added real-dimension forward/loss/backward regressions with six critical agents,
   24 typed modes, and 24 natural roots. This exact shape would have failed in v16.
3. Added strict experiment provenance (`42_write_run_provenance.py`). Reusing an
   `OUT_ROOT` with a different hash of critical model/loss/evaluation code or copied
   configs now fails instead of silently mixing stale checkpoints and reports. Candidate
   configs are hashed under stable logical names and checked before canonical config
   files are overwritten, so a rejected reuse attempt cannot corrupt the old record.
4. Changed the top-level execution script to detach the entire workflow by default,
   not only the inner dual-GPU driver. Global output is written to
   `logs/driver.nohup.log`, the PID to `logs/driver.pid`, and each stage retains its
   own log file.
5. Added `CHECK_RUN_STATUS_V16_1.sh` for PID and recent-log inspection.
6. Added stage-specific DataLoader settings: natural training defaults to 8 workers
   and prefetch 2; transport/planner retain conservative 4/1 defaults because their
   dense tensors previously caused host/pinned-memory failures.
7. Added `DIAG_PROFILE=fast|full`. The default fast preflight samples 2,048 train and
   1,024 validation transport scenes, 1,024 alignment/oracle/anchor scenes, while the
   paper-facing full audit remains available separately.
8. Reworked transport diagnostics with threaded NPZ reads, exact streaming means and
   MAEs, and a bounded 10,000-bin histogram for root-recovery quantiles. This removes
   Python lists containing millions of scalar values; the quantile resolution is
   explicitly reported (default bin width `1e-4`).
9. Cache sufficiency and v9 reuse reports are no longer recomputed when valid reports
   already exist. New runs use sampled cache sufficiency by default; `FULL_CACHE_AUDIT=1`
   retains the complete scan.
10. Added `RUN_FULL_DATA_AUDIT_V16_1_CN.sh` for a background full data audit without
    launching model training.

### Algorithm decision

No additional model or loss change is promoted in v16.1. The v16 dynamics decoder,
effectiveness losses, and OBS capacity must first complete training and controlled
ablations. A code crash is not evidence that these components are ineffective, and
changing them again before obtaining metrics would destroy attribution.

### Data decision

The v15 causal-label dataset is not architecture-locked to the CNOB decoder. It changes
the semantic targets and weights (especially OBS contamination/map filtering) and the
downstream response/witness labels. If CNOB fails on v9 labels, the v15 dataset is not
automatically useless. Use a small interaction-heavy v15 pilot and a model/data matrix:
CNOB+v9, CNOB+v15-pilot, simpler typed residual+v15-pilot. Rebuild the full v15 dataset
only when the matrix shows a reproducible label contribution.

### Local validation

- Python compilation: pass;
- shell syntax: pass;
- realistic natural forward/loss/backward regression: pass;
- threaded transport diagnostic regression: pass;
- strict provenance regression: pass;
- full unit/regression suite: **96 passed**.

### Prohibited claims

- Do not report any v16 algorithm improvement from the failed run.
- Do not label preflight analytic-basis metrics as learned decoder metrics.
- Do not attribute later main-model gains to the new loss or OBS capacity until the
  controlled ablations pass.
- Do not reuse an experiment root after code/config changes by bypassing provenance.
- Do not use fast diagnostics as the final paper data audit; run the full audit once
  for the frozen code/data release.

## v16.2 — old-PyTorch compatibility and end-to-end pipeline completion guards (2026-07-24)

### Triggering evidence

Both `driver.nohup.log` and `full_driver.nohup.log` stopped during the startup pytest
suite. Five tests failed in `_natural_mode_usage_loss` because the server PyTorch build
does not support `Tensor.any(dim=(0, 1))`. This was a compatibility regression introduced
while fixing the earlier `[B,A]` versus `[B,A,M]` broadcast bug. No model training or
evaluation started, so these logs contain no algorithm or closed-loop evidence.

A second static pipeline defect was found: `NEXT_RUN_COMMANDS_V16_1_FULL_CN.sh` did not
set `RUN_FULL=1`. Even after training and probe success, that script would not have
produced the nominal full Waymax results.

### Engineering fixes

1. Replaced the unsupported tuple-dimension boolean reduction with the equivalent
   old/new-PyTorch-compatible chain `mode_mask.any(dim=0).any(dim=0)`.
2. Added a source-level regression that forbids reintroducing `.any(dim=(...))` in the
   natural loss and retained the realistic `B=2, A=6, M=24, R=24` forward/loss/backward
   regression.
3. Added `43_pipeline_preflight.py`, which validates YAML configs, imports every critical
   train/eval/gate module, records Python/PyTorch/CUDA/JAX/TensorFlow/Waymax availability,
   checks `torchrun`, and executes a realistic natural forward/loss/backward before GPU
   hours are consumed.
4. Added `44_validate_pipeline_outputs.py`. A run cannot print `complete` unless required
   checkpoints and JSON reports exist; probe/full validation additionally requires
   usable CR, offroad, and EP/progress values.
5. Fixed the v16.2 full wrapper so `RUN_FULL=1` and Waymax dependency preflight are enabled
   by default.
6. Added robust `wait_all` handling for parallel diagnostics, offline evaluation, Waymax
   waves, and full shards. The driver waits for every child and reports failure instead
   of exiting on the first failed child while leaving sibling processes running.
7. Full COWP shards are now resumable independently; valid shard JSON files are reused
   after an interrupted run.
8. Added explicit engineering-only quality-gate bypass support and a separate
   `NEXT_RUN_COMMANDS_V16_2_ENGINEERING_SMOKE_CN.sh`. This path exists only to exercise
   downstream planner/Waymax code with tiny epochs/scenario counts; bypassed outputs are
   marked and are prohibited as paper evidence. Strict scripts keep all gates enabled.
9. Added `Offroad` as a supported alias in planner-delta summaries, while retaining
   `OffroadRate`, so different Waymax metric adapters still produce the requested
   offroad comparison.
10. All v16.2 scripts use a fresh default output root to prevent provenance conflicts
    with failed v16/v16.1 runs.

### Algorithm decision

No decoder, loss, planner, selector, or label semantics were changed. This release is an
engineering-only repair so the next completed run remains attributable to the v16
algorithm rather than a new moving target.

### Local validation

- complete unit/regression suite: **100 passed**;
- Python compilation: pass;
- all v16.2 shell scripts: `bash -n` pass;
- all pipeline CLI modules: import/`--help` pass;
- realistic natural preflight: pass;
- completion-validator CR/offroad/EP fixture: pass.

### Prohibited claims

- Engineering-smoke closed-loop numbers are not paper results.
- A strict run that stops on a quality gate is an algorithmic result, not a pipeline
  failure, provided `pipeline_preflight.json` and tests pass.
- Do not call the pipeline complete unless `pipeline_completion_report.json` passes.

## v16.3 — numerical-integrity recovery, evidence-gated attribution, and root-wise NCF correction (2026-07-24)

### Triggering evidence from the uploaded v16.2 engineering smoke

This revision is based on `cowp_v16_2_engineering_smoke_v9labels_seed2026_ancdatafix` and must not be interpreted as a paper-result run. It used one epoch per stage, only 20 Waymax scenarios, `ALLOW_QUALITY_GATE_FAILURE=1`, `STOP_AFTER_STAGE=probe`, `RUN_FULL=0`, and the `v9_reuse` data protocol. The causal audit passed its engineering checks but explicitly reported `full_v15_label_protocol_pass=false` because v15 label tensors were not materialized.

The natural history contains an internally inconsistent pattern: epoch `-1` and epoch `0` have exactly identical validation metrics; `natural/residual_l2=0` and `natural/control_smoothness=0`; the learned-natural diagnostic reproduces the analytic basis exactly and reports zero gain; nevertheless `base_deviation` is about 34.29 m. The best checkpoint therefore remains the initialization checkpoint. Downstream transport/planner/selector results are mechanism probes on an unverified basis, not evidence for their algorithmic effectiveness.

### Root cause and engineering fix

The failure pattern is highly consistent with FP16 autocast overflow in the scene graph followed by `0 * Inf -> NaN` in the zero-initialized natural residual/control heads. The previous loss path could silently sanitize non-finite predictions through `nan_to_num`, while `GradScaler` could skip optimizer steps without exposing that fact. This yields finite-looking losses, zero residual/control statistics, and no actual learning.

Changes:

1. `cowp/models/cowp_model.py`
   - Natural decoder and integration now execute in an explicit FP32 precision island.
   - Graph features and anchor state are cast to FP32 before the zero-initialized residual/control heads.

2. `cowp/scripts/03_train.py`
   - Added `--amp-dtype {auto,bfloat16,float16}`; `auto` prefers BF16 on supported CUDA hardware.
   - Natural/representation stages automatically fall back to full FP32 when only FP16 is available.
   - Added recursive pre-loss NaN/Inf detection on model outputs. Non-finite predictions now fail fast and are never converted into zeros.
   - Added synchronized DDP non-finite checks, gradient-norm checks, optimizer-step counters, AMP-skip counters, and a hard failure when no optimizer step is executed or more than 2% of attempted AMP steps are skipped.
   - The GradScaler now follows the effective stage AMP policy rather than the raw CLI flag.

3. Launch/status scripts
   - Added `run_cowp_v16_3_dual_gpu.sh` with natural FP32 by default and downstream BF16-auto policy.
   - Added `NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh`, `NEXT_RUN_COMMANDS_V16_3_FULL_CN.sh`, `NEXT_RUN_COMMANDS_V16_3_ENGINEERING_SMOKE_CN.sh`, and `CHECK_RUN_STATUS_V16_3.sh`.
   - Status detection checks both historical and current locations of `QUALITY_GATES_BYPASSED.txt` and reports optimizer/AMP-skip evidence.

4. Controlled attribution
   - Added `RUN_NATURAL_ABLATIONS_V16_3_CN.sh`.
   - Main, `no_effectiveness_loss`, and `no_obs_capacity_boost` runs use fresh outputs and identical natural-stage precision. The full pipeline is blocked until both the natural-effectiveness gate and component-attribution gate pass.

### Status of the six requested claims after v16.2

- **v15/v16 decoder effective:** not established. The trained checkpoint did not deviate from initialization.
- **new loss effective:** not established. No valid learned main model and no controlled ablation were available.
- **OBS residual capacity effective:** not established. Residual/control outputs were exactly zero and no valid capacity ablation existed.
- **natural gate effective:** established only as an engineering safeguard. It correctly rejected an inert natural checkpoint and prevented it from being promoted as evidence.
- **planner effective:** not established. One epoch, invalid natural foundation, and mixed 20-scenario results cannot isolate planner contribution.
- **selector effective:** not established. BCOT improved some conventional quantities versus pair-max but was worse than Pareto on collision/progress, while predicted FSR remained about 0.92 and fallback remained 0.80.

### Theoretical correction retained in the revised manuscript

The old existential witness condition—there exists a high-burden safe response—is logically insufficient because emergency responses usually exist. The revised definition uses stable natural roots with probability mass, same-root response transport, conflict mass, retained low-burden root mass, and a conflict-conditioned tail burden. Option preservation is removed from the primitive burden to avoid circularity. Burden thresholds are calibrated on a disjoint split and frozen, preventing the certificate from learning to raise its own acceptance threshold. Logged-replay online runs are labeled as proxy/mechanism diagnostics; reactive-agent and human-audited protocols are mandatory for causal burden claims.

### Promotion rules

No downstream result may be used in the paper unless all of the following are true:

1. `train/runtime/optimizer_steps > 0`, AMP skip ratio <= 2%, and no non-finite prediction is observed.
2. Learned natural basis improves over the analytic basis on the preregistered overall and OBS metrics without unacceptable neutral/priority degradation, across at least three seeds.
3. New-loss and OBS-capacity one-factor ablations pass the attribution gate.
4. Transport/root recovery and BCOT calibration pass; no `least_violation` operating point is accepted as a result.
5. Closed-loop evaluation uses at least 1,000 interaction-heavy scenarios per seed, three seeds, paired bootstrap confidence intervals, and separately reports logged-replay and reactive-agent protocols.

### v16.3 local validation

- `python -m compileall -q cowp tests/test_v16_3_numeric_safety.py`: passed.
- `pytest -q`: **107 passed**.
- `CHECK_RUN_STATUS_V16_3.sh` on the uploaded v16.2 smoke correctly reports `ENGINEERING-ONLY`, all three failed/bypassed gates, missing v16.3 optimizer-step evidence, and `INTENTIONAL_PARTIAL_RUN` after probe.
- Revised TeX static consistency: balanced braces, no unresolved `ref/eqref`, no literal traceback, no stray Markdown fence, and no undeclared `mathbbm` command.

## v16.4 — integrated residual trust region, yaw-frame correction, and strict calibration promotion (2026-07-25)

### Triggering evidence from the uploaded v16.3 natural recovery

The strict v16.3 run is materially different from the earlier inert v16.2 smoke. The
natural optimizer executed normally (`optimizer_steps=2044` at epoch 0 and no AMP
skips), validation loss decreased, and the learned CNOB decoder improved the exact
analytic basis. On 2,000 validation scenes, source-restricted weighted 8-second error
improved from 1.8862 m to 1.1779 m overall. The OBS branch improved from 4.1038 m to
2.6402 m, while NEU and PRIO also retained positive gains. The absolute natural-basis
gate therefore passed.

The strict natural-effectiveness gate nevertheless failed for two localized physical
reasons:

1. velocity--heading consistency was 0.17497 rad, slightly above the registered
   0.15 rad threshold; and
2. the integrated residual endpoint had a 45.218 m p99, above the 25 m bound, with a
   highly skewed distribution (p50 0.588 m, p90 31.912 m).

The full pipeline stopped at this gate before transport/planner training. The uploaded
archive contains no separate natural-ablation output root, so it supplies no evidence
for or against the new-loss or OBS-capacity attribution claims. It also contains no
v16.3 planner or selector result.

### Root-cause analysis

1. The dynamic decoder formed the moving yaw residual as
   `velocity_yaw - base_velocity_yaw`, although the final trajectory is represented as
   `base_absolute_yaw + yaw_residual`. For low-speed prototypes, base velocity direction
   is not guaranteed to equal the stored absolute heading. This creates a frame mismatch
   and can double-count the heading offset. The correct residual is
   `velocity_yaw - base_absolute_yaw`.
2. Bounded instantaneous acceleration, jerk, and yaw-rate do not bound the displacement
   accumulated over eight seconds. A small subset of OBS modes can therefore use the
   residual as a second unconstrained trajectory decoder, undermining stable root
   identity despite improving minADE.
3. A soft loss evaluated only on a radially hard-projected endpoint has zero radial
   gradient outside the feasible ball. The interior loss must use the pre-projection
   integrated endpoint, while the projected trajectory is used for prediction and
   physical diagnostics.

### Algorithm and engineering changes

1. Corrected the yaw reference frame. Moving-state heading is now derived from the final
   absolute velocity relative to the base absolute yaw. Exact zero-residual initialization
   is preserved by using velocity heading only when the learned velocity correction is
   non-negligible.
2. Added source-conditioned integrated endpoint budgets: OBS 20 m, NEU 8 m, PRIO 6 m.
   The complete local acceleration/jerk sequence is scaled and re-integrated, preserving
   position--velocity consistency rather than clipping the final position.
3. Added a dimensionless soft interior loss on the **pre-projection** endpoint/budget
   ratio, with a default interior threshold of 0.75. This gives over-budget controls a
   radial gradient back toward the feasible interior.
4. Added diagnostics for projected endpoint, raw endpoint, raw budget ratio, per-source
   endpoint distributions, and raw boundary-saturation rate. The effectiveness gate now
   additionally rejects a model when more than 25% of valid modes are at or beyond 95%
   of their source budget.
5. Added a controlled `no_integrated_trust_region` ablation. Attribution requires at
   least 5 m p99 tail reduction without more than 0.15 m OBS regression.
6. Tightened mechanism promotion: a BCOT operating point with calibration status
   `least_violation` is no longer accepted. Only a genuinely constraint-satisfying
   calibration may pass the mechanism gate.
7. Added v16.4 recovery/full/status/ablation launchers and a revised manuscript that
   states the model-based intervention limitation, the integrated trust region, the
   pre-projection interior loss, and the impossibility of replacing a feasibility
   certificate by an arbitrary finite soft burden penalty.

### Status of the six requested claims after v16.3

- **v15/v16 decoder:** partially supported for the trained v16 CNOB natural module. It
  clearly improves the analytic basis, but the architecture claim is not fully promoted
  until the v16.4 physical gate and architecture-level ablations pass.
- **new loss:** not yet identifiable; the uploaded ablation result root is absent.
- **OBS residual capacity:** not yet identifiable for the same reason.
- **natural gate:** validated as an effective safeguard. It accepted useful prediction
  gains but correctly blocked a physically invalid long-tail solution.
- **planner:** not evaluated in this strict run because the pipeline stopped before it.
- **selector:** not evaluated in this strict run for the same reason.

### Local validation

- Full Python unit/regression suite: **113 passed**.
- Added stopped-agent/rotated-anchor yaw regression.
- Added hard endpoint-budget and finite-gradient regression.
- Added regression proving the soft trust loss receives a nonzero gradient from the
  pre-projection endpoint.
- Python compilation and all v16.4 shell syntax checks: pass.
- Revised TeX raw brace balance: zero.

### Promotion rules

- Do not relax the 0.15 rad yaw threshold or 25 m residual-tail threshold to make v16.3
  pass; rerun v16.4 from a fresh output root.
- Do not claim the new loss, source-adaptive capacity, or trust region independently
  effective unless the four-way attribution run passes across at least three seeds.
- Do not use a `least_violation` calibration as the paper operating point.
- Do not interpret logged replay as causal evidence that burden was transferred; report
  it as a learned-mechanism proxy and add reactive-agent plus human-audited stress tests.

## v16.8.9 engineering repair — complete smoke/protocol tooling and numeric-module import fix (2026-08-08)

No algorithm/data-threshold/output-path change. Restored the missing v16.8.9 diagnostic/protocol scripts, upgraded scripts 46/50 to their v16.8.9-compatible implementations, replaced invalid direct imports of numeric-prefixed script 59 with `importlib.import_module`, added a `CASUAL` typo-compatible smoke wrapper, and revalidated the v16.8.9 execution chain. Full Python regression: **175 passed**.

## v16.8.9 data-contract repair — exact audit/transport semantics and supervision sufficiency (2026-08-08)

The completed 96-scene v16.8.9 smoke passed the proposal point-estimate gates (`AnyNCF=0.4167`, false-safe `0.5000`, PBTR `0.4419`, hard recovery `0.2083`) but exposed 1,258 exact audit/transport affected-root mismatches and an overly strong hard requirement on the prevalence of burden-only affected roots. The repair makes root affectedness/conflict/retention and canonical root weights exact across audit and transport, keeps burden-only prevalence advisory rather than threshold-forcing, adds an in-place smoke NPZ repair path, and adds training-supervision sufficiency audits before strict/full training. No NCF/PBTR/proposal threshold or candidate geometry was relaxed. Full regression: **178 passed**.

## v16.8.13 — mechanism-auditability separation, empirical route evidence, typed PRIO recovery, and certificate visibility contract (2026-08-13)

### Triggering evidence from the v16.8.12 96-scene smoke

The v16.8.12 validation smoke completed successfully as a data/engineering run and passed the proposal/causal screen and the training-supervision audit, but its composite verdict correctly blocked the strict probe because model-support failed. Among 530 selected critical vehicles, 12 (2.264%) had zero natural roots and the same 12 had fewer than two low-burden roots. Every one of those failures was map-filter dominated, every one had priority relation `EQUAL_OR_NEGOTIATED`, and every one used the `logged_geometry_neutral_timing` reference because a lane-route could not be resolved. Ten had all 80 WOMD future states valid, one had 68, and one had only 19; therefore the remaining rootlessness was not primarily missing-future data.

A second independent contract failure was structural: natural-source support was OBS=1554, NEU=3324, PRIO=0. The PRIO generator reused the same route/timing family already occupied by OBS/NEU while cross-source de-duplication happened before typed-source retention. Consequently the data could not supervise the configured typed PRIO natural modes even when the total natural-root count was nonzero.

### Semantic corrections

1. **Critical selection is no longer conflated with offline mechanism-label auditability.** `cowp/critical/valid` retains the inference-time critical-selection semantics. A new `cowp/critical/mechanism_valid` indicates whether the offline WOMD record has enough lane or factual-route evidence to build an eight-second natural/transport/witness target without fabricating unsupported geometry.
2. **Evidence-gated empirical route corridor.** If a lane route is unavailable, a vehicle may use a narrow factual-geometry corridor only when at least 60 future steps and at least 70% of the 80-step horizon are valid. The corridor is constructed from contiguous valid WOMD future segments only, uses stricter geometric tolerances than the ordinary lane check, and is explicitly recorded with `natural/map_evidence_mode=2`; it is never marked as HD-map verified. This is geometry evidence only: timing remains neutralized rather than copied as a normative behavior target.
3. **Insufficient evidence is masked, not fabricated or silently deleted.** A selected critical with neither usable lane support nor sufficient factual-route evidence remains selected but has `mechanism_valid=0`. It does not contribute natural/transport/witness supervision. Its frequency is reported and hard-gated (`<=1%` criticals by default) so the pipeline cannot pass by hiding difficult actors.
4. **Protected PRIO root identity is explicit.** For protected `AGENT_PRIORITY` / `EQUAL_OR_NEGOTIATED` relations, the canonical progress-preserving root is assigned to PRIO before OBS/NEU candidates enter cross-source de-duplication. PRIO has its own ordered acceleration/speed family and every auditable protected critical is required to retain at least one priority-preserved PRIO root. No factual corridor is extrapolated beyond the observed geometric support merely to manufacture a PRIO sample.
5. **Auditability is finalized against the actual route builder.** The cheap critical-stage lane projection is only a precheck; natural generation downgrades `mechanism_valid` when a projected lane has no usable forward continuation and there is not enough factual-route evidence. This closes endpoint/degenerate-lane cases without changing `critical/valid`.
6. **Candidate certificate validity is separate from pair-label validity.** `cowp/candidates/certificate_valid` is true only when the complete selected-critical certificate universe is auditable. Raw NCF/false-safe labels are masked by this flag for candidate-level training, while pair-level mechanism losses use `mechanism_valid`.
7. **Model-input visibility is part of certificate validity.** If a Scenario-proto selected critical track cannot be represented in the tf.Example/model agent rows, the cache loader invalidates the corresponding mechanism supervision and all candidate-level certificate targets for that scene. A post-tensor-cache model-support audit is mandatory, preventing labels from certifying against agents the model cannot observe.
8. **Fingerprint coverage expanded.** `cowp/data/dataset.py` and `cowp/scripts/03_train.py` are now part of the fresh-cache code fingerprint because visibility masking and certificate-aware planner ranking change training semantics.

### Promotion gates

- On `mechanism_valid=1` criticals: zero rootless criticals, zero criticals with fewer than two low-burden roots, and every auditable protected critical has at least one priority-preserved PRIO root.
- Global auditability: `mechanism_unauditable_rate <= 1%` by default.
- Candidate certificate completeness: at least 98% of scenes remain certificate-complete after the cache/model-visibility merge.
- Proposal/causal, training-supervision, and existing response/witness/transport non-degeneracy gates remain unchanged.
- The smoke must explicitly authorize strict; validation strict and train-pilot must both authorize full rebuild under the identical current fingerprint.

### Performance policy

The v16.8.12 smoke spent about 101.23 s/scene in safe responses and 45.39 s/scene in witness construction versus 3.12 s/scene in natural generation. Therefore v16.8.13 does not reduce candidate, natural-root, response-slot, or witness semantics to gain speed. Empirical-corridor work is only enabled for lane-unresolved auditable actors and reuses contiguous factual geometry; the existing exact audit/result reuse remains the primary semantics-preserving optimization.

### Local validation

- Repository regression split: **201 passed, 5 skipped** with no failures.
- `python -m compileall -q cowp tests`: passed.
- `bash -n` on the v16.8.13 smoke/strict/train-pilot/master launchers and full-core preparation script: passed.

### Evidence limitation retained

Passing these data-support gates establishes that WOMD-derived labels can support the implemented natural-basis, same-root transport, witness, NCF selector, and logged-Waymax mechanism experiments. It does not make logged WOMD/Waymax replay counterfactual causal ground truth for burden transfer. Publication-level causal burden claims still require the separately specified reactive-agent protocol and held-out human-audited false-safe stress set.

v16.8.13 also propagates the certificate/auditability mask into label-space validation, causal-audit diagnostics, Waymax replay class balancing, and learned-offline label metrics. Certificate-unknown candidates may still be replayed for physical collision/off-road/progress outcomes, but they are never counted as NCF/false-safe negatives or positives. Label-space NCF/false-safe precision/recall and proposal-floor metrics expose certificate-label coverage explicitly, preventing missing counterfactual supervision from being reported as a successful negative class.

## v16.8.14 — WOMD split/completeness contract and adaptive probe manifests

This revision does **not** change the COWP planning/certification algorithm. It repairs dataset engineering contracts discovered before the v16.8.13 smoke run:

- Smoke hard scenes are selected from the current baseline validation cache instead of a stale v16.8.8 manifest.
- Probe manifests support a preferred hard count plus a fixed total; representative random scenes fill a hard-scene shortfall, while a minimum stress count remains explicit.
- WOMD 1.3.1 preflight now audits shard-index completeness, rather than treating a readable partial download as complete.
- Primary COWP data contract is Scenario training/validation for authoritative labels plus scenario-ID-matched tf.Example training/validation for model/Waymax tensors.
- Tensor-cache construction can require every proto-derived label scenario to have a matching tf.Example; partial training tensors are a hard error.
- Added a split-layout auditor for training/validation/testing and local challenge/auxiliary directories. `validation_interactive` is secondary stress data until scenario-ID overlap with standard validation is explicitly audited; blind testing splits are not used to fabricate future-dependent COWP mechanism labels.

## v16.8.15 — representation-aware WOMD v1.3.1 split contract (2026-08-13)

This revision does **not** change the COWP planner, natural-basis mathematics, response/witness labels, or promotion thresholds. It corrects the local WOMD release-layout model used by dataset engineering.

- Replaced the previous split × representation Cartesian-product audit with an explicit representation-aware matrix.
- Scenario splits inventoried by the current local/release contract: `training`, `validation`, `testing`, `validation_interactive`, `testing_interactive`, `training_20s`, `visualization`.
- tf.Example splits inventoried: `training`, `validation`, `testing`, `validation_interactive`, `testing_interactive`.
- `scenario/training_20s` and `scenario/visualization` are Scenario-only auxiliary directories. The auditor never constructs, globs, counts, or requires nonexistent `tf_example/training_20s` or `tf_example/visualization` peers.
- The current COWP primary benchmark remains the official 9-second train/validation pair: Scenario proto is authoritative for labels/map/traffic control; scenario-ID-matched tf.Example is the tensor/Waymax source.
- `validation_interactive` remains an optional secondary interaction-stress evaluation split and is not merged into standard validation; scenario-ID overlap can be audited before reporting it.
- `testing` and `testing_interactive` remain blind-evaluation-only because future GT is hidden; they are never used to construct COWP natural/transport/witness/NCF supervision.
- `scenario/training_20s` is excluded from the present 91-step pipeline. Using it would require a separate windowing and group-split design and cannot be achieved by swapping a glob.
- Added regression tests proving Scenario-only splits have no tf.Example glob or completeness requirement.

## v16.8.24 — rebuild readiness and execution-chain repair

See `ALGORITHM_CHANGELOG_V16_8_24.md`. The release fixes the missing profile-summary module reference, adds report-only benchmark recovery and active-chain dependency preflight, persists WOMD Scenario indices across benchmark/full build, makes compact split selection self-contained, tunes CPU worker defaults, and attaches Waymax outcomes to train/val/heldout-test. Label semantics are unchanged from the semantically verified v16.8.23 fast path.

## v16.8.25 — Certificate-Then-Utility probe, immutable planner repair, and exact-ID Waymax evaluation

See `ALGORITHM_CHANGELOG_V16_8_25.md` for the full evidence/decision log. This revision is based only on the original v16.8.24 code/results and reuses the existing v16.8.24 labels/caches; it makes no proposal, label, split, RCOT-target, or dataset reconstruction change. The previously suggested MCFC experiment was never run and is not part of this release.

The existing held-out evidence localizes the dominant global ceiling to proposal support (`AnyNCF=0.36346`, fixed-bank selected-false-safe floor `0.59475`) while RCOT/BCOT remains the strongest learned mechanism (`LowSafeExist AUPRC=0.89743`, priority/global BCOT false-safe AUPRC `0.83736/0.92806`). The generic candidate classifier remains weak (`NCF/false-safe AUPRC 0.17558/0.35444`). COWP also retains a secondary selector gap: NCF selection recall given an available NCF proposal is `0.78877`, with `0.07677` selected false-safe excess above the proposal floor.

v16.8.25 therefore adds `cowp_cert_utility`, a one-factor Certificate-Then-Utility diagnostic: use the exact same protected-priority BCOT hard certificate and physical shield as COWP, but remove the second BCOT/set-preservation ranking pass and rank the surviving certified set by planner score. Original `cowp` behavior is unchanged. Candidate outcome heads are now reported with collision/offroad/unsafe AUPRC but remain excluded from selection (`outcome-risk-penalty=0`) until evidence supports a physical-risk guard.

The original execution command used `--stage all`; this jointly retuned mechanism/planner modules and selected checkpoints by total loss rather than the v16.8 immutable-mechanism planner protocol. Planner-only repair now keeps candidate encoder, natural decoder, witness decoder, SetTransport/BCOT, response decoder, graph, and learned priority gate frozen/deterministic. A 6-epoch, `1e-5` cross-stage warm start is provided but is intentionally gated behind the no-training CTU probe.

The original strict Waymax command supplied `--scenario-ids-file` although the evaluator did not parse/propagate it. The exact-ID evaluation path is now implemented with duplicate rejection, no silent `num_scenarios` truncation, hard failure on unresolved requested IDs, scenario IDs in rollout outputs, and manifest hashing.

No current claim is made that CTU improves COWP. Promotion requires certificate invariance, non-inferior validation burden/false-safe/progress, and paired exact-ID Waymax. The current dataset contains zero burden-only affected roots (`affected == unsafe`), so the affected-root extension is not independently supported and is scheduled for a clean conflict-only retraining ablation only after the selector is locked.

v16.8.25 local validation: focused CTU/exact-ID/immutable-planner tests **9 passed**; full suite **252 passed, 5 skipped, 8 failed**. The eight failures are the same uploaded-repository historical issues (six missing archived launchers, two stale hard-coded semantic fingerprints), not new functional regressions. `compileall` and `bash -n NEXT_RUN_COMMANDS_V16_8_25_CTU_CN.sh` pass.

## v16.8.26 — CTU Negative Result, Fallback-Only Physical-Risk Probe, and Strict-Waymax Attribution/Acceleration

See `ALGORITHM_CHANGELOG_V16_8_26.md` for the complete record.

Key decisions:
- reject CTU as the default selector: removing the post-certificate set-preservation frontier consistently worsened EP, PBTR/FSR, NCF-selection recall, NPR, and strict-Waymax EP;
- retain the original certificate-compatible COWP frontier;
- do not rebuild the dataset/proposal bank in this release;
- treat the 200-ID strict Waymax runs as complete and valid, but do not infer fallback causality from aggregate diagnostics alone;
- add `cowp_fallback_outcome`, which leaves the certified COWP path exactly unchanged and exposes the already-trained outcome head only to the explicit uncertified fallback ranker;
- add outcome calibration/low-FPR diagnostics before any physical hard-gate promotion;
- add first-event/fallback temporal diagnostics and strict conventional/planner baselines for physical-failure localization;
- add exact-ID TFExample-index plumbing, runtime profiling, batched JAX->host state transfer, cached SDC identity, and optional two-A30 parallel exact-ID sharding.

The uploaded MCFC probe archive is incomplete (`profile_labels.jsonl` empty; no paired/source-ablation/promotion outputs) and provides no algorithm evidence.


## v16.8.27 — Conventional-Safety Integrity Repair

See `ALGORITHM_CHANGELOG_V16_8_27.md`. v16.8.26 strict-Waymax attribution was blocked because `NEUTRAL_EGO` could bypass the roadgraph/collision audit while being marked conventional-safe. v16.8.27 removes that bypass, adds a runtime conventional-fallback invariant, fixes outcome-head reporting metadata, and adds selected-candidate/first-event provenance. No planning mechanism was promoted.

## v16.8.28 — No-Valid Execution Integrity Repair

See `ALGORITHM_CHANGELOG_V16_8_28.md`. v16.8.27 exact-200 attribution was blocked because zero-padded slot 0 was executed when no valid candidate existed. v16.8.28 makes padding non-executable, uses a current-state bounded smooth stop only as execution-level emergency behavior, records explicit emergency/no-valid provenance, and removes an unreachable selector branch. No planning mechanism was changed.

## v16.8.29 — Recursive Viability Recovery and Causal-Collision Fast Path

See `ALGORITHM_CHANGELOG_V16_8_29.md` and `CURRENT_ANALYSIS_V16_8_29_CN.md`. The clean v16.8.28 exact-200 result localizes collision to online zero-conventional states rather than conventional-safe fallback or accepted COWP paths. v16.8.29 therefore freezes RCOT/BCOT/protected-priority/frontier/outcome settings and adds an opt-in `cowp_recursive_viability` recovery that acts only when the full conventional pool is empty: preserve roadgraph-safe candidates when available, maximize the collision-safe prefix under the unchanged causal screen, then use the existing fallback score only as a tie-break. The branch remains explicitly uncertified.

The release also decomposes zero-conventional states into collision/roadgraph/intersection failure modes and accelerates the causal collision screen by caching candidate-invariant agent futures once per replanning step and vectorizing the per-agent distance audit. A development-only outcome-enriched 64-ID panel is provided to avoid repeated 4×200 runs; it is forbidden as publication evidence. Generic recursive feasibility is not claimed as novel by itself; promotion to a paper-level physical-viability contribution requires evidence for an orthogonal social-noncoercive × physical-recursive-feasibility formulation.

# V16.8.32 — Temporal Option Persistence

V16.8.31 结果通过完整性审计，但 BHOV 未通过预注册 holdout64 non-harmful gate：counterfactual48 上虽然保留 10/10 RVR rescue，却仅避免 3/9 RVR-induced collisions，并带来 +3 kinematics scenes 与显著 EP regression；在与 V16.8.30 panels 完全不重叠的 holdout64 上，COWP 与 pure RVR 均 collision-free，而 BHOV 新诱发 1 个 collision。因此 BHOV 不 promotion、不开 exact200。

该 holdout induced scene 说明两个耦合根因需要拆开：one-step successor option signature 可能无法发现 delayed option collapse；同时 stateless BHOV 在 COWP/RVR 之间的间歇切换会形成第三种 hybrid closed-loop policy，缺乏 recovery commitment 本身可能有害。

V16.8.32 新增两个 opt-in 分支：

1. `cowp_trihorizon_option_persistence`：只在 zero-conventional+valid recovery state 比较原 COWP fallback 与原 RVR candidate，以 current prefix H0、actual emitted action 后 successor option signature V1、第二个受同一 controller 约束 emitted step 后的 V2 构成 parameter-free product-order，要求 H0/V1/V2 均不退化且至少一项严格改善。
2. `cowp_sov_recovery_commitment`：使用 V16.8.30 strict SOV 作为高精度 recovery entry；进入后在 zero-conventional+valid regime 内持续执行原 RVR，直到 certificate/conventional option 恢复或 no-valid emergency，专门诊断 stateless hybrid switching。

没有改变 dataset、natural roots、RCOT、BCOT、protected-priority certificate、set-preservation frontier、8 s conventional contract、outcome head 或 proposal bank。明确禁止通过 BHOV epsilon/tolerance、prefix/risk scalar weighting、缩短 conventional horizon、增加 proposal primitive 或训练新 successor head 来救本轮结果。

实验协议：equivalence16 → preregistered counterfactual48 → fresh37 → exact200。counterfactual48 要求 retain>=5/10 old RVR rescues、avoid>=7/9 old induced、net collision reduction>=3、kinematics regression<=1、mean EP delta>=-0.05。fresh37 是从 exact200 中剔除 V16.8.30/31 所有 mechanism-selection panels 后剩余的 37 IDs，hash `ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481`；只允许 counterfactual48 通过的方法进入 fresh37，再通过后才运行 exact200。

当前主线继续收紧为 Orthogonal Option-Set Feasibility：social axis 保护其他关键 actor 的 natural low-burden option set；physical-temporal axis 保护 ego 在实际 emitted recovery action 下的 future executable option-set persistence。当前 P0 bottleneck 为 temporal option persistence + recovery-mode consistency；proposal support 为长期 ceiling，accepted-path kinematics 保持独立 secondary problem。

详细记录见 `ALGORITHM_CHANGELOG_V16_8_32.md` 与 `CURRENT_ANALYSIS_V16_8_32_CN.md`。

# V16.8.33 — Recovery Option Spectrum + Dominance Hysteresis

V16.8.32 结果通过完整性审计，但 THOP 与 unconditional SOV Recovery Commitment 均未通过 counterfactual48 预注册 promotion gate，因此没有运行 fresh37/exact200，也不对失败机制做 threshold rescue。THOP 保留 9/10 old RVR rescues 但只避免 4/9 old induced，新增 3 个 kinematics scenes，paired EP delta≈-0.1507；多堆一个 successor horizon 不成立。Commitment 保留 7/10 rescues、避免 6/9 induced、net -5 collision、kinematics +1、EP delta≈-0.023，说明 mode continuity 有真实信号但 unconditional continuation 过度粘滞。

进一步 paired 分解显示：commitment 相比 strict SOV 新救回 6 个 SOV collision scenes，却新诱发 `3919ccd73c0fabd7` 与 `c34fe8e79cdf1161` 两个 SOV 原本安全 scene；因此 V16.8.33 不使用固定 dwell/hysteresis margin，而新增 `cowp_sov_dominance_hysteresis`：strict SOV dominance entry、weak/equality continuation、dominance-loss exit。与此同时，SOV 对 old RVR-induced 的唯一 false positive `7721ff4800156886` 最终进入 no-valid emergency，暴露原 successor signature 在 conventional 为空时几乎退化成 max-prefix。主分支 `cowp_recovery_option_spectrum_hysteresis` 因此用 causal successor 上“每个安全 horizon 仍存活的 distinct recovery macro 数”构成完整 option-persistence profile，并用 pointwise strict/weak dominance驱动同一状态机。profile area/权重不参与 selection。

继续冻结 data/natural roots/RCOT/BCOT/protected-priority certificate/set-preservation frontier/8s conventional contract/outcome settings。新增禁止 V3+ horizon stacking、unconditional commitment promotion、固定 dwell、hysteresis epsilon/margin 搜索、option-profile AUC/加权 scalarization，以及在 analytic target 验证前训练新的 viability head。实验顺序保持 equivalence16 -> counterfactual48 -> 从未被 V16.8.32 新方法运行过的 fresh37 -> exact200；Stage-1 仍使用 retain>=5/10、avoid>=7/9、net collision>=3、kinematics regression<=1、EP delta>=-0.05 的预注册 gate。

详细记录见 `ALGORITHM_CHANGELOG_V16_8_33.md` 与 `CURRENT_ANALYSIS_V16_8_33_CN.md`。


### V16.8.33 promotion-integrity (delivery hardening)

The v16.8.33 launcher now enforces the counterfactual48 -> fresh37 -> exact200 preregistered gates at runtime. Later stages fail closed unless the requested method has `preregistered_gate.pass=true` in the immediately preceding analyzer JSON. This is experiment-protocol hardening only; planning semantics are unchanged. Final focused sanity remains 39/39 passed.

# V16.8.34 — Execution-Conditioned Recovery Option Spectrum

V16.8.33 counterfactual48 passed integrity/attribution audit but neither SDH nor ROSH passed the unchanged preregistered Stage-1 gate. SDH failed rescue-retention and net-collision-reduction gates. ROSH was a near-positive mechanism probe (6/10 old rescues retained, 7/9 old induced avoided, net -5 COWP collisions, EP delta≈-0.02786) but failed the hard kinematics gate because it induced 2 new kinematics scenes while the preregistered bound was <=1. The gate is not relaxed and V33 fresh37 is not run.

The two V33 kinematics regressions expose a new structural mismatch. One occurs directly during ROSH recovery (`29cd2aca8ae5e222`); the other occurs later on an accepted, conventional-safe `LANE_CHANGE_RIGHT` (`6992366c5c998d00`). V33 successor option profiles carry the emitted ego pose/velocity but not the controller memory `previous_longitudinal_accel`, and they count nominal valid+roadgraph-safe macro support although nominal candidate validity intentionally ignores the initial jerk transient. Consequently V33 measures semantic option support, not controller-realizable option support.

V16.8.34 therefore keeps V33 ROSH frozen as a reference and adds two controlled branches only in the zero-conventional+valid recovery regime. `cowp_transition_guarded_rosh` keeps the nominal V33 spectrum and adds hard current controller-transition non-regression. `cowp_executable_option_spectrum_hysteresis` additionally carries emitted longitudinal acceleration into the causal successor and counts only successor macro options whose first desired step is realizable under the existing acceleration/deceleration/jerk/yaw/lateral-acceleration execution limits. Both use a parameter-free product partial order and the same strict-entry / weak-continue / regression-exit state machine; no candidate is relabeled conventional/NCF.

All mature data/social/certificate layers, proposal families, the 8 s conventional contract, outcome settings, and accepted-path logic remain frozen. The P0 bottleneck is now **Execution-Conditioned Recovery Option-Set Feasibility**. The Stage-1 gate is exactly V33's gate; fresh37/exact200 remain fail-closed downstream stages. Detailed design and evidence are in `ALGORITHM_CHANGELOG_V16_8_34.md` and `CURRENT_ANALYSIS_V16_8_34_CN.md`.

# V16.8.35 — Control-Projected Recovery Option Spectrum

V16.8.34 passed reliability attribution but both preregistered branches failed the unchanged Stage-1 gate. TG-ROSH retained only 3/10 historical RVR collision rescues; EOSH retained 0/10, removed no net COWP collision, and still carried the two V33 kinematics regressions. The v34 internal exact-transition predicate also marked only about 48% of RVR recovery probes feasible versus about 88–90% of base probes, showing that nominal first-waypoint exact reachability was collapsing recovery recall rather than measuring actual online executability.

V16.8.35 therefore freezes all mature data/social/certificate/proposal/controller layers and replaces the v34 physical observable. `cowp_waymax_kinematic_guarded_rosh` is a diagnostic branch that keeps the V33 semantic recovery spectrum but evaluates the current **actual emitted transition** using the same inverse acceleration/steering-curvature contract used by Waymax's kinematics metric. The main `cowp_control_projected_option_spectrum_hysteresis` branch repeatedly projects every successor candidate through the existing stateful COWP accel/jerk/yaw controller, carries emitted longitudinal-acceleration memory, and constructs its semantic option-survival curve from the resulting **control-realized** trajectory under frozen roadgraph, causal collision, and execution constraints. Selection remains a parameter-free pointwise partial order with V33 strict-entry / weak-continue / dominance-loss-exit semantics.

The dominant bottleneck is refined to **Evaluation-Aligned Control-Realized Recovery Option-Set Preservation**; the paper-level platform-independent abstraction is **Control-Projected Physical Option-Set Feasibility**. Waymax metric thresholds are an evaluation adapter only and are not claimed as method novelty. New bans include nominal exact-waypoint executability filtering, treating internal accel/jerk/yaw limits as a proxy for Waymax kinematics, and globally retuning the mature online controller in the same recovery round. Stage-1/fresh37/exact200 gates remain unchanged and fail-closed. See `ALGORITHM_CHANGELOG_V16_8_35.md` and `CURRENT_ANALYSIS_V16_8_35_CN.md`.

---

# ALGORITHM_CHANGELOG V16.8.36 — Control-Projected Semantic Recovery Frontier

## Triggering evidence: V16.8.35 is reliable but both preregistered Stage-1 branches fail

V16.8.35 passed an independent result-integrity audit: 66/66 hard checks passed; equivalence16/shards/manifests/checkpoints/summary recomputation were clean; the uploaded analyzer was independently rerun with 0 recursive mismatches at tolerance 1e-12; the V16.8.25→35 focused semantic/integrity suite was 49/49 passed. Therefore V16.8.35 is eligible for algorithm attribution rather than repair-only handling.

The Stage-1 gate is unchanged from V16.8.33–35: retain >=5/10 old RVR rescues, avoid >=7/9 old RVR-induced collisions, remove >=3 net COWP collisions, kinematics regression <=1 scene, paired mean EP delta >=-0.05, and nonzero intervention. Any failed condition archives the policy; thresholds are not moved after seeing results.

### V16.8.35 WK-ROSH

`cowp_waymax_kinematic_guarded_rosh` is closed-loop identical to V16.8.33 ROSH on the 48-scene panel: 29/48 collision, 8/48 kinematics, EP 0.974647. It retains 6/10 old RVR rescues and avoids 7/9 old RVR-induced collisions, but kinematics regresses by +2 scenes versus COWP and therefore fails the unchanged gate.

Its current-action Waymax kinematic predicate is almost non-discriminative: base and RVR emitted targets are both feasible on ~99.85% of probes, with mean feasibility delta exactly 0. This cleanly falsifies “the missing recovery mechanism is only the current one-step Waymax kinematics guard.”

### V16.8.35 CPOSH

`cowp_control_projected_option_spectrum_hysteresis` produces a real positive physical signal but still fails promotion:

- COWP: 34/48 collision, 6/48 kinematics, 1/48 offroad, EP 1.002512.
- CPOSH: 27/48 collision, 7/48 kinematics, 0/48 offroad, EP 0.913987.
- Paired collision: 10 rescued, 3 induced, net -7 failures; McNemar p≈0.0923.
- Old RVR rescues retained: 8/10 (pass).
- Old RVR induced avoided: 6/9 (fail; required >=7/9).
- Kinematics net regression: +1 (pass).
- Paired mean EP delta: -0.08853 (fail; required >=-0.05), bootstrap 95% CI approximately [-0.2429, 0.0132].

The EP regression is concentrated rather than uniform: scene `fccd9a25a2a57a73` alone contributes about -3.129 EP; excluding that scene the paired mean delta is about -0.02383. This is diagnostic only and does **not** alter the preregistered failure.

## Root-cause refinement

V16.8.35 disproves two overly narrow explanations:

1. **Current-action metric alignment is not the dominant missing signal.** WK-ROSH exactly reproduces V33 ROSH.
2. **Control-projected option-set representation is informative but binary endpoint selection is structurally incomplete.** CPOSH rescues 10 COWP collisions while adding only +1 net kinematics scene, yet all V29–V35 recovery policies still compare only two current candidates:
   - `base`: the global least-coercive-valid COWP fallback;
   - `alt`: one global max-prefix RVR endpoint.

On the same 48 scenes the online bank contains roughly 36–38 valid candidates on average. In CPOSH rescued scenes the mean valid count is ~37.8; in CPOSH-induced scenes it is ~38.5. Thus the remaining failure cannot be described as “no current support exists.” The controller has a large fixed bank but the recovery policy exposes only two endpoints to the expensive physical viability observable.

The induced scenes are also more deeply trapped in the uncertified regime: mean zero-conventional exposure is ~85.4% versus ~69.1% for rescued scenes, while both groups still have many valid candidates. This motivates using the existing semantic support more completely before adding new map/Frenet primitives.

The dominant bottleneck is therefore refined to:

**Existing-Bank Recovery Support Utilization / Binary-Endpoint Bottleneck under Uncertified Recovery**

with the paper-level physical object:

**Control-Projected Semantic Recovery Frontier**.

## V16.8.36 main method: `cowp_control_projected_recovery_frontier`

V16.8.36 changes one central factor relative to V35 CPOSH: the recovery decision is no longer restricted to `COWP base vs one global RVR endpoint`. It still uses the exact same fixed candidate bank and exact same V35 control-projected physical observable.

### 1. Existing-bank semantic representatives

Only in the unchanged regime:

`full conventional set == empty && valid candidates exist`

the method forms the same roadgraph-first recovery pool used by RVR. For every distinct non-PAD macro already present in this pool it selects one deterministic representative:

1. maximize current causal collision-safe prefix within that macro;
2. break prefix ties with the frozen COWP fallback score;
3. break remaining ties by candidate index.

No new trajectory, primitive, map route, model prediction, or learned score is generated. This is **support construction**, not proposal expansion.

### 2. Hard current-survival non-regression

A representative cannot trade away immediate causal survival for a richer future set. Its current collision-safe prefix must be >= the COWP fallback prefix. This is an additional component of a hard product order, not a weighted prefix reward.

### 3. Reuse the V35 control-projected successor spectrum

For each representative, V36 reuses the V35 physical observable unchanged:

- actual controller-emitted current target;
- controller acceleration memory propagated into the successor;
- other agents propagated with the frozen causal CV model;
- unchanged online proposal bank rebuilt at the successor;
- each successor candidate projected step-by-step through the same stateful COWP controller;
- roadgraph + causal collision + benchmark execution feasibility evaluated on the control-realized trajectory;
- distinct non-PAD semantic macros counted at every survival horizon.

A representative is weakly physically admissible only if both current safe-prefix and future control-projected spectrum are non-regressive versus base. Strict entry requires at least one strict improvement.

No profile AUC, horizon discount, risk scalarization, margin, dwell time, learned viability classifier, or new threshold is introduced.

### 4. Feasibility first, frozen COWP preference second

Among all strict physical-frontier representatives, V36 does **not** select the largest profile or longest prefix. It chooses the candidate with the **lowest already-frozen COWP fallback score**.

This preserves the established architecture:

`hard feasibility / set preservation -> existing least-coercive preference`

instead of inventing another physical-vs-progress weight.

### 5. Semantic recovery-mode consistency

Because there can now be multiple recovery macros, a boolean recovery-active state is insufficient. V36 stores the active macro identity.

- inactive: strict physical dominance permits entry; choose the least-COWP-cost strict-frontier macro representative;
- active: continue only a representative of the **same semantic macro** while it remains weakly physically dominant;
- if that macro loses weak dominance: exit immediately to the unchanged COWP fallback;
- certificate/conventional recovery or no-valid emergency clears the mode.

There is no fixed dwell time or hysteresis epsilon, and the controller cannot directly jump recovery-macro→different-recovery-macro while the old mode is active.

## Why this is the correct next falsifiable branch

V35 CPOSH demonstrates that the control-projected option spectrum carries useful information (net -7 COWP collisions) but its binary policy still creates three collision false positives and has a progress near-miss. V36 asks whether those failures arise because the physically informative representation is being applied to an impoverished pair of endpoints rather than to the semantic support already available in the bank.

This is the last existing-bank selector/support test before proposal expansion. If V36 fails the unchanged Stage-1 gate, do not keep adding ROSH/EOSH/CPOSH comparators. The next family must move to either:

- genuine reachable proposal/support construction, or
- a higher-fidelity closed-loop physical transition/reachable-set formulation.

## CCF-A positioning

“Multiple backup branches,” “contingency planning,” and “backup-plan feasibility” are already established research themes; V16.8.36 must not claim novelty merely because it evaluates several recovery modes. The candidate paper contribution remains the unified **Orthogonal Option-Set Feasibility** abstraction:

- social axis: the ego plan must not collapse protected agents’ natural low-burden response sets;
- physical axis: uncertified recovery must not collapse the ego’s own control-realizable recovery set.

V36 is a probe of the physical-axis **semantic quotient/frontier**, not a standalone novelty claim.

## Frozen layers

Freeze: compact-5k data/labels/checkpoint, Natural roots, RCOT, BCOT, protected-priority hard certificate, post-certificate set-preservation frontier, outcome head settings, 8 s conventional contract, V27/V28 integrity repairs, candidate families, and common online controller. Accepted-path kinematics remains a separate secondary bottleneck and is not modified in V36.

## Newly prohibited directions

All previous bans remain. Add:

1. Do not tune CPOSH profile thresholds, Waymax kinematics thresholds, profile area/AUC weights, or EP/collision tradeoff weights to rescue V35.
2. Do not continue the binary `base vs global-RVR` ROSH/EOSH/CPOSH comparator family after V35; both V35 branches failed the inherited Stage-1 gate.
3. Do not expand map/Frenet primitives in V36 before testing whether the existing bank’s semantic support can be exploited; valid-candidate support remains large in both rescued and induced groups.
4. Do not rank the recovery frontier by “largest option spectrum,” profile area, or longest prefix. Physical dimensions remain hard feasibility; the frozen COWP fallback preference resolves the surviving frontier.
5. Do not allow direct recovery-macro-to-different-recovery-macro switching while active; V31 already established harmful hybrid-mode dynamics.
6. Do not train a recovery/frontier neural head until this analytic target is validated on counterfactual48 and a lineage-cleaner development panel.
7. Do not mix accepted-path execution-viability changes into this recovery version.

## Promotion protocol

V16.8.36 inherits the exact six Stage-1 conditions unchanged. `frontier_selected_non_rvr_rate_on_switches` is recorded as a mechanism diagnostic, not a post-hoc seventh outcome gate. If it is zero, an outcome fluctuation cannot be attributed to broader support utilization.

Only Stage-1 pass may run fresh37. Only fresh37 pass may run historical exact200 development confirmation. Launcher is fail-closed.

## Validation

- V16.8.36 new focused tests: 4/4 passed.
- V16.8.35 + V16.8.36 focused tests: 10/10 passed.
- V16.8.25→36 focused semantic/integrity sanity: 53/53 passed.
- Python compilation: passed.
- launcher `bash -n`: passed.
- direct `fresh37_parallel2` without Stage-1 analysis exits fail-closed with code 4 before Waymax rollout.

# ALGORITHM_CHANGELOG V16.8.37 — Recourse Returnability Bridge (Audited Revision)

## Source/provenance note

The uploaded `COWP.zip` was described as V16.8.36 but already contained a V16.8.37 draft. That draft was treated as an untrusted candidate, not as an established result. V16.8.36 was independently re-audited first; only after reliability passed was the V37 mechanism reviewed and corrected. This changelog describes the final audited V16.8.37, not the draft as uploaded.

## Triggering evidence: V16.8.36 is reliable and fails the unchanged Stage-1 gate

An independent V16.8.36 audit passes 71/71 hard checks:

- exact manifest copies and logical SHA256 values;
- 24+24 counterfactual48 shards, disjoint and exactly covering the frozen manifest;
- merged scenario rows identical to shard rows;
- CR/Collision/Offroad/Kinematics/EP and fallback statistics independently recomputable;
- identical checkpoint/protocol provenance and `mechanism_ground_truth_available_online=false`;
- equivalence16 remains 16 scenes / 1120 fields / 0 mismatch;
- the V16.8.28 no-valid execution invariant remains true;
- the shipped analyzer is independently reproduced with 0 recursive mismatches at tolerance 1e-12.

V16.8.36 is therefore eligible for algorithm attribution. It is not publication evidence because counterfactual48 is development-selected and logged replay has no counterfactual burden ground truth.

The six preregistered Stage-1 checks are unchanged:

- retain at least 5/10 historical RVR collision rescues;
- avoid at least 7/9 historical RVR-induced collisions;
- remove at least 3 net COWP collisions;
- kinematics net regression no worse than +1 scene;
- paired mean EP delta at least -0.05;
- nonzero intervention.

`cowp_control_projected_recovery_frontier` fails three checks:

- old rescues retained: 5/10 (pass);
- old induced avoided: 3/9 (fail);
- collision vs COWP: 6 rescued / 6 induced, net 0 (fail);
- kinematics: +2 scenes (fail);
- paired mean EP delta: -0.02443 (pass);
- intervention: 20.57% policy steps (pass).

V16.8.36 is archived and must not run fresh37.

## Mechanism attribution from V16.8.36

V36 genuinely exercised the broader existing-bank frontier:

- 62.85% of recovery switches selected a non-historical-RVR representative;
- mean semantic representatives per probe: 8.86;
- mean control-projected profiles evaluated: 5.66.

The failure is therefore not inactive code or binary-endpoint degeneration. Relative to V35 CPOSH, V36 produces 1 collision rescue and 8 induced collisions (McNemar exact p≈0.0391).

Rescued and induced groups have similar valid support, non-RVR selection, and prefix gain, but radically different return to conventional support:

- rescued: zero-conventional 67.5%, mean conventional candidates 3.77;
- induced: zero-conventional 92.5%, mean conventional candidates 0.79;
- both groups retain roughly 35–37 valid candidates.

All six V36-induced collisions are immediately preceded by an uncertified recovery candidate that is valid but non-conventional, collision-unsafe, prefix 0, in a collision-empty zero-conventional state. Two scenes remain in recovery active/continue state for essentially the whole episode after one rare entry.

The dominant bottleneck is therefore refined from existing-bank support utilization to:

**Returnability to Full-Conventional Physical Feasibility under Uncertified Recovery**

Paper-level abstraction:

**Control-Reachable Recourse-to-Feasibility**.

Option richness is not returnability.

## New method: `cowp_recourse_returnability_bridge`

V37 is active only when the full conventional set is empty and at least one valid candidate exists. It does not change the dataset, checkpoint, proposal families, RCOT/BCOT, protected-priority certificate, set-preservation frontier, 8 s conventional contract, common controller, or accepted-path logic.

### 1. Frozen high-precision entry pre-gate

The controlled pair returns to the strongest positive physical probe:

- `base`: unchanged least-coercive-valid COWP fallback;
- `alt`: historical global max-prefix RVR candidate.

A returnability probe is allowed only when:

1. emitted actions are physically distinct;
2. the alternative current causal prefix is not below the base;
3. the alternative has at least one safe step, so it survives the next real replanning edge;
4. the alternative strictly dominates under the frozen V35 control-projected option-spectrum plus benchmark-aligned current kinematic relation.

The new positive-edge condition is a hard feasibility invariant, not a tuned margin.

### 2. Direct restoration witness

Execute the actual emitted one-step action in the causal successor model and rebuild the unchanged online bank. `direct_restore=true` iff the successor contains any valid full-conventional candidate.

Only current state, actual emitted action, map state, controller memory, and the frozen causal constant-velocity surrounding-agent model are used. Logged future trajectories are never consumed by action selection.

### 3. One-new-replan semantic recourse witness

If direct restoration is absent, build a new candidate bank at the causal successor. Keep candidates that are valid, roadgraph-safe, positive-prefix, non-PAD, and benchmark-kinematic-transition-feasible. Project them with the carried longitudinal-acceleration state.

The audited implementation evaluates every distinct `(semantic macro, actual emitted target)` action class. It does **not** keep only one max-prefix candidate per macro. A macro belongs to the witnessed recourse set if any of its action classes reaches a second causal state with a nonempty unchanged full-conventional set.

The second action is a newly replanned successor action, not waypoint `t+2` of the original candidate.

### 4. Hard returnability partial order

No count, AUC, discount, risk weight, or utility scalar is used.

- alt direct / base non-direct: strict improvement;
- base direct / alt non-direct: reject;
- both direct: returnability tie, no strict entry from returnability alone;
- neither direct: require strict semantic set inclusion `R_base ⊂ R_alt`;
- incomparable recourse sets: reject.

### 5. Witness-bound one-real-replanning bridge

A non-direct entry stores both `bridge_pending=true` and the semantic recourse macros witnessed at entry.

At the next actual policy step:

- ordinary certificate/conventional recovery clears pending immediately;
- otherwise rebuild the bank from the actual observed simulator state;
- search only the macros witnessed at entry;
- require positive current prefix and no prefix regression relative to the ordinary COWP fallback on that step;
- evaluate every distinct emitted-action class for direct restoration;
- select the minimum frozen COWP fallback score inside the hard restoring set;
- execute at most this one bridge action, then clear pending unconditionally;
- abort to ordinary COWP if no witnessed direct-restoring action remains.

There is no weak-equality continuation, dwell duration, hysteresis epsilon, fixed commitment, or recurrent recovery mode.

## Correctness fixes applied to the uploaded V37 draft

1. **Existential recourse completeness:** replaced one-representative-per-macro probing with all distinct emitted-action classes, preventing a false negative when a non-max-prefix action in the same macro is the only restoring action.
2. **Witness/execution consistency:** store the witnessed recourse macro set at entry; the actual bridge cannot execute an unrelated newly discovered macro.
3. **Current-survival preservation:** actual bridge candidates must keep at least the ordinary COWP base prefix on the actual bridge state.
4. **Positive-edge invariant:** an entry action must retain at least one causal collision-free step; a prefix-0 action cannot be certified by a future returnability surrogate.
5. **Expanded diagnostics:** action classes available/evaluated, witnessed macro count, actual bridge pool, prefix floor, bridge execution/abort conditioned on pending, and selected bridge macro.

## Diagnostics (not additional outcome gates)

- returnability probe and strict-dominance rates;
- base/RVR direct-restoration rates;
- base/RVR witnessed recourse macro counts;
- base/RVR action classes available/evaluated;
- current-edge survival rate;
- bridge pending/entry/direct-entry rates;
- witnessed macro count on bridge steps;
- bridge candidate pool/action classes/evaluated;
- bridge execution and abort rates conditioned on pending;
- direct-restoring candidate count and selected bridge macro.

These diagnostics support attribution only. The inherited six-item Stage-1 gate remains the only promotion gate.

## Frozen layers

Freeze in V37:

- compact-5k data/label contract;
- Natural roots;
- RCOT and BCOT;
- protected-priority hard certificate;
- certificate-compatible set preservation;
- outcome head/settings;
- 8 s conventional-safe definition;
- V27 conventional-integrity and V28 no-valid execution fixes;
- candidate families;
- common online controller;
- accepted/certified path selection and execution.

Accepted-path kinematics remains a real secondary bottleneck and is not modified in the same round.

## Newly prohibited directions

All prior bans remain. Add:

1. Do not continue or tune V36 semantic frontier selection, weak dominance, fallback order, or macro hopping.
2. Do not interpret valid-candidate count, macro count, profile area, or prefix gain as returnability.
3. Do not tune fixed dwell, hysteresis epsilon/margin, or minimum recovery duration.
4. Do not extend a failed V37 to 2/3/4 bridge depths; that would repeat horizon stacking.
5. Do not scalarize recourse with macro count, profile AUC, discount, collision/progress weights, or risk coefficients.
6. Do not relax the six preregistered gates.
7. Do not train a neural returnability head before the analytic witness is validated.
8. Do not retune the common controller or mix accepted-path kinematics into this recovery round.
9. Do not claim counterfactual burden causality from the logged-replay Waymax protocol.

## Experiment protocol

Run only:

1. sanity;
2. make_ids;
3. base_equivalence16_parallel2;
4. counterfactual48_parallel2;
5. analyze_counterfactual48.

Stop unless `preregistered_gate.recourse_returnability_bridge.pass == true`.

Only a Stage-1 pass may run fresh37. Only a fresh37 pass may run historical exact200 development confirmation. Launcher checks remain fail-closed and have been verified to terminate with exit code 4 before any rollout when the preceding analysis is missing or failed.

## Validation

Final audited V37 validation:

- V37 dedicated tests: 8/8 passed;
- V16.8.25→37 focused semantic/integrity suite: 61/61 passed;
- Python compile: passed;
- launcher bash syntax: passed;
- manifest hashes: passed;
- analyzer v2 smoke: passed;
- fail-closed promotion probe: passed;
- conventional-safety bypass grep: passed.

The repository-wide suite still stops during collection because historical `tests/test_v16_8_29_recovery_viability.py` imports the already-absent `_recovery_bridge_viability_mask`. The same failure is reproduced in the unmodified uploaded source and is not introduced by V37.

## CCF-A positioning and failure branch

A one-step backup/bridge or return to a safe set is not claimed as novelty by itself. The candidate paper structure remains **Orthogonal Option-Set Feasibility**:

- social feasibility protects other agents' natural low-burden option sets;
- physical feasibility protects ego recourse paths back to certified control-realizable feasibility.

If V37 fails Stage-1, stop the RVR/SOV/BHOV/THOP/ROSH/EOSH/CPOSH/frontier/bridge selector family. The next algorithm family must construct genuine control-reachable recovery support or an explicit viability/reachable-set approximation, with an unbiased paired proposal probe before any dataset reconstruction or learned amortization.


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

---

# V16.8.39 — Conflict-Window Control-Reachable Recovery Tube

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

---

# V16.8.40 — Shift-Closed First-Action Viability Interval

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


---

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

---

# V16.8.42 — Root-Conditioned Interaction-Aware Reachable-Response Envelope (RC-IARE)

Online method:

```text
cowp_interaction_aware_reachable_response_envelope
```

Release role:

```text
NEW STAGE-1 SCIENTIFIC BRANCH
EXACTLY NESTS V39
ARCHIVES V40/V41 FIRST-ACTION INTERVAL COMPLETION
REQUIRES EQUIVALENCE16 + COUNTERFACTUAL48 BEFORE ANY PERFORMANCE CLAIM
```

## Evidence that selects this branch

The repaired V16.8.41 result passes the independent reliability audit:

```text
107/107 blocking checks passed
algorithm attribution allowed = true
```

The unchanged six-item Stage-1 Gate nevertheless fails 5/6:

```text
historical RVR rescues retained = 3/10  (GO >= 5/10; FAIL)
historical RVR induced avoided = 9/9   (GO >= 7/9; PASS)
net COWP collision failures removed = 4 (GO >= 3; PASS)
kinematics net regression = 0           (GO <= 1; PASS)
paired mean EP delta vs COWP = -0.010855 (GO >= -0.05; PASS)
action-changing intervention > 0         (PASS)
```

V41 is physically outcome-equivalent to V39 on counterfactual48.  The repaired
first-action interval extension evaluates 1,551,139 hypotheses and produces only:

```text
98 full-safe interval witnesses
19 shift-closed interval witnesses
5 new certified first actions
3 selected new-first-action steps
0 additional collision rescue over V39
```

Those three selections occur in two still-colliding scenes and one scene already
rescued by V39.  Therefore the first-action continuum completion is not promoted.
The previously preregistered interpretation branch applies literally:

```text
new first actions selected + unchanged Gate failure
-> stop analytic interval/schedule patching
-> move to interaction-aware reachable response support
```

## Dominant bottleneck

The P0 failure is no longer scalar ranking, horizon length, release timing, or a
missing interpolation point.  It is:

```text
fixed-path physical support is empty under a frozen-CV surrounding-agent model,
although a root-conditioned low-burden interactive response may make the joint
continuation feasible.
```

Six of the seven historical RVR rescues still missed by V41 have interval attempts
but no full-safe/shift-closed new first action.  This is Type-A support mismatch.
One additional scene (`fccd9a25a2a57a73`) receives V39/V41 certificates yet still
collides in closed loop; this Type-B model-relative certificate mismatch remains a
separate monitor and is deliberately not co-optimized in V42.

## Algorithm

V42 preserves the exact V39 conflict-window control-reachable tube as its first
branch.  If V39 returns a certificate, V42 returns it byte-for-byte without
constructing interaction support.

Only at a V39-empty step, V42 evaluates the same frozen controller-reachable V39
hypotheses with a root-conditioned interaction certificate:

1. Read only the trained causal natural-decoder trajectories and logits for online
   critical agents.  The untrained dense response trajectory head is not used.
2. Reuse the exact frozen canonical natural-root probability measure:
   `p_min=0.03` support filtering with all-mode fallback, support renormalization,
   then independent `epsilon_p=0.02` floor smoothing.
3. Geometrically deduplicate roots with the frozen 0.10 m mean-path criterion,
   require at least two stable roots, and retain at least 0.75 canonical mass
   (the complement of the frozen 0.25 CVaR tail mass).
4. For every retained root, build the existing deterministic same-root recovery
   bank.  Keep only profiles that are below the unchanged adaptive low-burden
   budget under the conservative `AGENT_PRIORITY` relation, roadgraph-drivable,
   and Waymax-inverse-dynamics feasible in both current and one-step-shifted form.
5. Identify the exact surrounding agents that block the current or shifted V39
   hypothesis.  Every blocker must have ready root support; otherwise reject.
6. Remove only those supported blockers from the frozen ego physical collision
   context.  All non-blocking actors, ego roadgraph checks, controller limits,
   full-horizon kinematics, and current/shift closure remain hard.
7. Require each blocker-root profile to be safe against the ego in both current
   and shifted tubes and to be bidirectionally safe against every non-blocking
   frozen-CV environment actor.
8. Solve an exact bounded CSP selecting one response profile per retained root.
   Profiles for different blockers must be mutually safe in both orientations and
   both current/shift tubes.  Roots of the same actor are alternatives and are not
   treated as simultaneous states.
9. Select the first hard-certified action using the frozen deterministic V39
   ordering.  The interaction extension cannot select the ordinary COWP base
   first action as a no-op.

This is a robust recourse certificate, not a soft interaction score.  No scalar
risk/progress/burden trade-off is introduced.

## Information boundary

The V42 support object consumes only:

- current causal simulator state;
- trained natural trajectories/logits;
- current map/roadgraph;
- frozen controller and response-bank dynamics;
- causal constant-velocity environment predictions.

It does not consume:

- `log_trajectory` or any logged future state;
- future outcome labels;
- online mechanism ground truth;
- the learned dense response trajectory output;
- unexecuted-policy real counterfactual trajectories;
- counterfactual48 scene identities or scene-specific parameters.

## Frozen contracts

V42 changes none of the following:

- compact-5k data, split, labels, or cache;
- checkpoint, model weights, loss, or training configuration;
- natural-root decoder, canonical root weights, RCOT, or BCOT;
- protected-priority hard NCF semantics;
- certificate-compatible set-preservation selector;
- 8 s conventional-safety horizon;
- proposal macro families;
- acceleration/deceleration/jerk/yaw limits;
- roadgraph and Waymax kinematics adapters;
- V39 conflict-window schedule family and shift semantics;
- no-valid emergency behavior;
- six-item Stage-1 outcome Gate.

## Diagnostics

Added step/scenario/pooled diagnostics for:

- interaction attempts and interaction-only selections;
- ready support agents, retained canonical roots, and eligible response profiles;
- hypotheses/no-op skips and failure-stage decomposition;
- unsupported blockers and residual physical failures;
- root failures against ego or frozen non-blocking environment actors;
- environment compatibility checks/rejections;
- cross-blocker joint compatibility checks/rejections/backtracks;
- selected blocker/root count, minimum retained mass, maximum response burden,
  profile evaluations, and environment checks.

These diagnostics support mechanism attribution; they do not modify the promotion
Gate.

## Analyzer and fail-closed launcher

Added:

```text
cowp/scripts/95_analyze_interaction_aware_reachable_response_envelope.py
NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh
```

The analyzer reuses the unchanged six-item conjunction Gate.  A compatibility
replay using the actual V41 merged result reproduces V41's exact 5/6 failure,
including the 3/10 rescue-retention failure.

The launcher exits with code 4 after a missing or failed counterfactual48 Gate.
It must not create fresh37 shards unless:

```text
preregistered_gate.interaction_aware_reachable_response_envelope.pass == true
```

## Validation before rollout

```text
V42 + V41/V39 focused/regression tests  96/96 passed
blocking code-audit checks               48/48 passed
canonical-root randomized equivalence   250/250 passed
four frozen manifest hashes             passed
Python compile                           passed
launcher bash syntax                     passed
analyzer V41 compatibility replay        passed
full V42 Waymax outcome                  not run yet
```

V42 is code-ready for frozen `equivalence16` and `counterfactual48`; it has no
performance verdict until those results pass reliability analysis.

## Required execution order

```bash
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh make_ids
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_counterfactual48
```

Stop after Stage-1 unless the recorded Gate passes.  If it passes, continue:

```bash
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh fresh37_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_fresh37
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh confirm200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_confirm200
```

## Preregistered interpretation

- Gate pass and interaction-only selection `>0`: the branch is eligible for
  fresh37; specific RC-IARE attribution is provisionally supported.
- Gate pass but interaction-only selection `=0`: outcome is attributable only to
  the exact nested V39 mechanism, not RC-IARE.
- Interaction-only selection `>0` but Gate fail: archive V42; inspect support and
  compatibility failure stages, but do not tune thresholds on counterfactual48.
- Interaction attempts `>0` but selection `=0`: the natural-root/analytic-response
  support object is mismatched; do not return to first-action grids.
- Kinematics or EP failure: archive; do not tune scalar weights.

## Newly prohibited post-hoc directions

In addition to all earlier prohibitions:

- do not change `p_min`, probability floor, 0.75 retained mass, minimum root count,
  root deduplication, adaptive burden budget, or response profiles after reading
  counterfactual48;
- do not drop current/shift universality for retained roots;
- do not remove non-blocking actors from the hard environment context;
- do not replace exact joint compatibility by independent per-agent certificates;
- do not use the logged future or Waymax future state to populate responses;
- do not activate the untrained dense response trajectory head;
- do not combine Type-B closed-loop mismatch repair or accepted-path kinematics
  repair into this one-factor branch;
- do not claim causal social-burden improvement from logged replay alone;
- do not treat the repeatedly used counterfactual48 panel as a final publication
  holdout.
