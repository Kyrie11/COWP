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
