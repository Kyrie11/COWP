# ALGORITHM_CHANGELOG V16.8.37 — Recourse Returnability Bridge

## Triggering evidence: V16.8.36 is reliable but the existing-bank semantic frontier fails decisively

V16.8.36 passed a fresh result-integrity/attribution audit before any mechanism interpretation. The two 24-scene shards are disjoint and exactly cover the frozen counterfactual48 manifest; merged standard metrics are exactly reproducible from scenario rows; equivalence16 remains 16 scenes / 1120 fields / 0 mismatch; online ground-truth leakage flags remain false; the v16.8.28 no-valid execution invariant remains intact; the shipped v16.8.36 analyzer was independently rerun with zero recursive mismatches; and the focused semantic/integrity suite is clean. The result is therefore eligible for algorithm attribution rather than repair-only handling.

The inherited Stage-1 gate is unchanged: retain >=5/10 old RVR collision rescues, avoid >=7/9 old RVR-induced collisions, remove >=3 net COWP collisions, kinematics net regression <=1 scene, paired mean EP delta >=-0.05, and nonzero intervention. No threshold is changed after observing v16.8.36.

`cowp_control_projected_recovery_frontier` fails that gate:

- COWP: 34/48 collision, 6/48 kinematics, 1/48 offroad, EP 1.002512.
- V35 CPOSH reference: 27/48 collision, 7/48 kinematics, 0/48 offroad, EP 0.913987.
- V36 frontier: 34/48 collision, 8/48 kinematics, 0/48 offroad, EP 0.978086.
- versus COWP collision: 6 rescued / 6 induced, net 0, McNemar p=1.0.
- old RVR rescues retained: 5/10 (pass at the exact lower bound).
- old RVR induced avoided: 3/9 (fail; required >=7/9).
- kinematics: +2 scenes (fail; allowed <=1).
- paired mean EP delta versus COWP: approximately -0.02443 (pass).
- intervention is nonzero (pass).

Relative to V35 CPOSH, V36 is not merely neutral: it produces 1 collision rescue and 8 induced collisions, a net +7 failure regression, with exact McNemar p≈0.0391. Therefore broadening the same fixed bank from a binary endpoint to a semantic recovery frontier is *not* the missing dominant mechanism.

## What V16.8.36 did successfully falsify

V36 genuinely used the broader fixed-bank support. Approximately 62.85% of its recovery switches selected a representative that was **not** the historical global-RVR endpoint. Thus the negative result cannot be dismissed as an implementation that silently collapsed back to `base vs RVR`.

However, the support statistics do not separate beneficial from harmful switches:

- rescued scenes select a non-historical RVR representative on roughly 77.4% of switches;
- induced scenes do so on roughly 75.0%;
- selected safe-prefix improvement is ~+5.20 steps in rescued scenes and ~+5.36 in induced scenes.

So `more semantic support` and `larger current prefix` are not sufficient physical viability observables.

The strongest separating evidence is **returnability to the conventional feasible region**. In V36-induced scenes:

- mean zero-conventional exposure is ~92.5%;
- mean conventional candidate count is only ~0.79;
- mean valid candidate count remains ~37.2.

The controller therefore remains in a state with many nominal/un-certified choices while almost never recovering a full-conventional option. All six V36-induced collisions occur while the immediately preceding action is still `no_conventional_use_control_projected_recovery_frontier`; the selected candidate is valid but conventional=false and collision-safe=false, with selected safe-prefix already 0. The failures are not downstream accepted-path accidents after successful restoration.

Two examples make the mode problem explicit:

- `d632f1919fe4bab`: recovery active for essentially the whole episode, entry ~1.25%, continuation ~98.75%, zero-conventional 100%, first collision around step 37.
- `f8d4c735825e5d81`: recovery active 100%, entry ~1.25%, continuation ~98.75%, zero-conventional 100%, first collision around step 19.

A single strict entry can therefore create an almost episode-long weak-dominance mode even though the system never returns to certified physical support.

## Dominant bottleneck after V16.8.36

The prior bottleneck `Existing-Bank Recovery Support Utilization / Binary-Endpoint Bottleneck` is falsified as the dominant next fix.

The new P0 bottleneck is:

**Returnability to Certified Physical Feasibility under Uncertified Recovery**

or, in the paper-level abstraction:

**Control-Reachable Recourse-to-Feasibility**.

The important distinction is:

- **option richness:** how many semantic recovery modes remain nominally/control-realizably available;
- **returnability:** whether executing a current recovery action preserves a causal replanning route back to the unchanged full-conventional feasible set.

V36 shows that the former can stay large while the latter is effectively absent.

## V16.8.37 main method: `cowp_recourse_returnability_bridge`

V16.8.37 follows the branch explicitly preregistered in the v16.8.36 changelog: after an existing-bank selector/frontier failure, stop adding ROSH/EOSH/CPOSH comparator variants and move to a higher-fidelity closed-loop returnability/reachable-set formulation.

It does **not** add map/Frenet primitives, retrain a model, change RCOT/BCOT, change certificate semantics, tune the common controller, shorten the 8 s conventional horizon, or modify accepted-path execution.

### 1. Frozen high-precision entry pre-gate

The current controlled alternatives return to the clean V35 pair:

- `base`: unchanged least-coercive-valid COWP fallback;
- `alt`: historical global max-prefix RVR candidate.

This is deliberate. V36 demonstrated that broad current semantic frontier expansion is harmful; V35 CPOSH remains the strongest positive physical signal so far.

An expensive returnability probe is considered only when:

1. base and RVR emit physically distinct current actions;
2. RVR current causal collision-safe prefix is not below base;
3. RVR strictly dominates base under the frozen V35 control-projected successor option-spectrum relation, including the benchmark-aligned emitted-action kinematics contract.

This preserves the V35 high-recall signal while asking a new question only for its strict-positive entries.

### 2. Direct restoration witness

For each current alternative, execute its *actual emitted one-step action* in the same causal successor model used by previous recovery probes and rebuild the unchanged online physical candidate bank.

If that successor contains any full-conventional candidate, the branch has a **direct restoration witness**.

This uses no future Waymax logged trajectory; surrounding agents use the same frozen causal constant-velocity propagation as the conventional collision audit.

### 3. One-replan semantic recourse witness

If direct restoration is absent at the first successor, V37 generates a **new candidate bank at that successor**, because the real planner will replan there. This is the key difference from THOP: the second edge is not waypoint `t+2` from the original candidate.

From that successor bank, V37 forms one deterministic representative per non-PAD semantic macro from candidates that are valid, roadgraph-safe, have positive current causal safe-prefix, and whose actual emitted transition satisfies the Waymax kinematics contract. Each representative is projected through the unchanged stateful controller using the carried emitted longitudinal acceleration.

For each representative, V37 performs one more causal successor step and asks whether that state contains any full-conventional candidate. The set of semantic macros that succeed is the **witnessed recourse set** `R(a)`.

This is a finite one-replanning recourse witness, not an indefinite horizon score and not a new proposal primitive.

### 4. Returnability partial order

No count weight, area, discount, risk coefficient, or learned score is used.

- alt direct-restores while base does not -> strict returnability improvement;
- base direct-restores while alt does not -> reject;
- both direct-restore -> returnability tie, so returnability itself cannot create a strict entry;
- neither direct-restores -> require strict semantic set inclusion `R_base ⊂ R_alt`;
- incomparable recourse sets -> reject.

Thus a candidate cannot win merely because it has the same number of different recovery macros; it must preserve every witnessed base recourse macro and add at least one new one.

### 5. One-real-replanning bridge, not hysteresis continuation

When an RVR entry wins because it has a strict non-direct recourse witness, V37 stores only `bridge_pending=True`.

At the **next actual policy step**:

- if the conventional/certificate path has already recovered, pending is cleared and ordinary COWP takes over;
- otherwise, V37 recomputes the candidate bank from the actual observed simulator state;
- one semantic representative per macro is tested for whether its actual emitted one-step transition produces a successor with any full-conventional candidate;
- among those direct-restoring representatives, choose the one with minimum frozen COWP fallback score;
- execute at most this one bridge action, then clear pending unconditionally;
- if no direct-restoring representative exists, abort the bridge and use the unchanged COWP fallback immediately.

There is no weak-equality continuation, dwell time, hysteresis epsilon, minimum recovery duration, or recurrent recovery mode. This specifically removes the v36 failure mode in which a single entry can continue for almost an entire zero-conventional episode.

### 6. Diagnostics

New step/episode diagnostics include:

- returnability probe rate;
- base/RVR direct-restoration rates;
- base/RVR witnessed recourse macro counts;
- strict/weak returnability relation;
- action classes evaluated;
- bridge pending/entry/direct-entry rates;
- bridge recourse execution rate;
- bridge abort rate;
- direct-restoring representative count on the actual bridge step.

These are mechanism diagnostics, **not** post-hoc additional promotion conditions. The inherited six-item Stage-1 gate remains the only outcome gate.

## Why this is not THOP or another tuned horizon

V16.8.32 THOP evaluated the same incomplete viability representation at additional time indices along/derived from candidate continuation. V16.8.37 instead explicitly branches through a *new replanning action* at the causal successor and asks whether a full-conventional set becomes non-empty.

The one-replan depth is fixed as the mechanism definition of this diagnostic probe; it is not tuned after seeing outcomes. If it fails, V37 is not extended to “2/3/4 bridge steps.” The next family must construct a genuine reachable-support/viability object rather than horizon-stack this probe.

## CCF-A positioning

Backup-plan MPC, contingency planning, recursive feasibility, safe-set recovery, and reachability are established research families. V16.8.37 must not claim novelty for “one backup action” or “returning to a safe set.”

The candidate paper contribution remains **Orthogonal Option-Set Feasibility**:

- social feasibility: an ego plan must not obtain safety by collapsing protected road users’ natural low-burden option sets;
- physical feasibility: an uncertified ego recovery must not obtain short-term survival by collapsing the ego’s own **recourse paths back to certified control-realizable feasibility**.

V37 is a mechanism probe that sharpens the physical axis from `option richness` to `recourse-to-feasibility`.

## Frozen layers

Freeze throughout V37:

- compact-5k data and label contract;
- Natural roots;
- RCOT and BCOT;
- protected-priority hard certificate;
- post-certificate set-preservation frontier;
- outcome head/settings;
- 8 s conventional-safe definition;
- V27 conventional-integrity and V28 no-valid execution fixes;
- current candidate families;
- common online action controller;
- accepted/certified path selection and execution.

Accepted-path kinematics remains a real but secondary independent bottleneck. V36 fails primarily on collision recovery, so it is not mixed into V37.

## Newly prohibited directions

All previous bans remain. Add:

1. **Do not continue V36 semantic frontier selection** or tune weak-dominance continuation. V36 significantly regresses collision relative to V35 CPOSH.
2. **Do not interpret valid-candidate richness as physical viability.** V36 induced scenes retain ~37 valid candidates while conventional support is almost absent.
3. **Do not tune recovery dwell/continuation thresholds.** V36 shows harmful near-episode-long continuation after rare entries; prior unconditional commitment was also negative.
4. **Do not add V3/V4 returnability horizons if V37 fails.** The next branch must be genuine reachable-support construction, not another horizon stack.
5. **Do not select recourse by number of macros, profile AUC, returnability count weight, or progress/collision scalarization.** V37 uses hard set inclusion and frozen COWP preference only.
6. **Do not broaden the current candidate frontier in V37.** V36 already falsified that as the dominant fix.
7. **Do not train a neural returnability head before the analytic witness is validated.**
8. **Do not modify accepted-path kinematics in the same version.**

## Promotion protocol

Stage-1 counterfactual48 uses the exact same six conditions as V16.8.33–36. No returnability diagnostic is introduced as a seventh outcome gate.

Only Stage-1 `pass=true` may run fresh37; only fresh37 pass may run historical exact200 development confirmation. The launcher remains fail-closed.

If V37 fails Stage-1, stop the ROSH/EOSH/CPOSH/frontier/bridge selector family. The next algorithm family should be a **genuine control-reachable recovery support / viability construction**, potentially with explicit reachable tubes/sets or learned dynamics only after an honest analytic target is defined.
