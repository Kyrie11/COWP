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
