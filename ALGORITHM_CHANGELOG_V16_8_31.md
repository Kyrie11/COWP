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
