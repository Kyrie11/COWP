# ALGORITHM_CHANGELOG V16.8.35 — Control-Projected Recovery Option Spectrum

## Triggering evidence: V16.8.34 failed its unchanged preregistered Stage-1 gate

V16.8.34 results passed reliability/integrity audit (32/32 hard checks, 16-scene COWP equivalence 1120 fields / 0 mismatch, shard/manifest/summary/analyzer reproduction all clean), so the failure is algorithmic rather than an engineering attribution blocker.

On the same 48-scene counterfactual panel:

- COWP: 34 collisions, 6 kinematics, EP 1.00251.
- V33 ROSH: 29 collisions, 8 kinematics, EP 0.97465.
- V34 TG-ROSH: 32 collisions, 7 kinematics, EP 0.98470; retained only 3/10 old RVR rescues and therefore failed the rescue-retention and net-collision gates.
- V34 EOSH: 34 collisions, 8 kinematics, EP 0.99558; retained 0/10 old RVR rescues, removed no net COWP collision, and regressed kinematics by +2.

The Stage-1 contract remains unchanged: retain >=5/10 old RVR rescues, avoid >=7/9 old RVR induced collisions, remove >=3 net COWP collisions, kinematics regression <=1 scene, paired mean EP delta >=-0.05, nonzero intervention. V34 failed and therefore must not run fresh37/exact200.

## Root-cause refinement

V34 defined an executable option using nominal first-waypoint exact realizability under COWP's internal accel/decel/jerk/yaw-rate/lateral-acceleration limits. This is structurally too strong: the online controller is explicitly designed to project nominal candidates into bounded emitted actions, so “requires projection” is not equivalent to “not executable.” On V34 recovery probes the base transition passed this exact-realizability predicate about 88–90% of the time while the RVR alternative passed only about 48%, consistent with the collapse from V33's recovery recall to EOSH's 0/10 old-rescue retention.

A second mismatch is evaluation semantics. Waymax `KinematicsInfeasibilityMetric` evaluates inverse acceleration and steering curvature of consecutive realized states; COWP's v34 internal transition check is not the same predicate. A v35 regression case explicitly shows an internally accepted 2 m/s, 0.1 rad/0.1 s transition whose inverse curvature is about 0.5 m^-1 and therefore violates the current Waymax default 0.3 m^-1 bound.

The dominant recovery bottleneck is consequently refined from nominal controller-state executability to **Evaluation-Aligned Control-Realized Recovery Option-Set Preservation**. The paper-level, platform-independent formulation is **Control-Projected Physical Option-Set Feasibility**; the Waymax metric is only the current benchmark adapter, not the method contribution.

## New diagnostic branch: `cowp_waymax_kinematic_guarded_rosh`

- Acts only in the unchanged `zero-conventional && valid` recovery regime.
- Compares only the original COWP fallback and original RVR candidate.
- Keeps the V33 semantic recovery-option spectrum unchanged.
- Replaces the V34 internal exact-waypoint predicate with a current-action guard matching Waymax's evaluated inverse acceleration / steering-curvature transition contract on the **actual emitted targets**.
- Alternative must itself be kinematically feasible; `base=false, alt=false` is not treated as admissible equality.
- Uses the frozen V33 strict-entry / weak-continue / dominance-loss-exit state machine.

This branch isolates whether v34 mainly checked the wrong current physical contract.

## New main branch: `cowp_control_projected_option_spectrum_hysteresis`

The main branch changes the physical viability observable, not the actual controller, proposal generator, certificate, or conventional-safe label.

For each base/RVR actual emitted current action:

1. Propagate the ego causally to the successor state and carry the **emitted longitudinal acceleration** as controller memory.
2. Propagate other agents using the same causal constant-velocity model as the frozen conventional collision screen; never read logged future ground truth.
3. Rebuild the unchanged online physical proposal bank on the successor.
4. For every successor candidate, repeatedly apply the **same stateful COWP acceleration/jerk/yaw projection used online** across the horizon rather than testing nominal waypoint exactness. Longitudinal acceleration memory is propagated step by step.
5. Evaluate the projected trajectory under the roadgraph screen, the same causal collision audit, and the benchmark execution adapter (currently Waymax inverse acceleration/steering curvature).
6. Define each candidate's control-realized survival prefix from the conjunction of those constraints.
7. Count distinct non-PAD semantic macro types surviving each horizon, yielding a control-projected recovery option spectrum `P_ctrl(h)`.
8. Use pointwise partial-order dominance only. No profile AUC, horizon discount, risk weight, candidate-count weight, margin, dwell time, or learned viability classifier is introduced.

A new regression test requires the first step produced by the horizon projector to match the existing online `_consistent_one_step_targets_np` target and acceleration exactly, preventing model/diagnostic controller drift.

## Frozen layers

No changes to compact-5k data/labels/checkpoint, natural roots, RCOT, BCOT, protected-priority hard certificate, set-preservation frontier, outcome settings, 8 s conventional contract, candidate families, V27 conventional integrity repair, V28 no-valid execution repair, or accepted-path logic. The common online controller is also frozen; v35 only reuses it inside the physical-option observable.

## Newly prohibited directions

In addition to all previous bans:

1. Do not continue using **nominal first-waypoint exact reachability** as the definition of an executable option; v34 EOSH destroyed recovery recall (0/10 old RVR rescues retained).
2. Do not use COWP's internal accel/jerk/yaw-rate/lateral-accel predicate as a proxy for Waymax KinematicsInfeasibility; v34 did not repair either known V33 kinematics regression and the contracts are structurally different.
3. Do not globally tighten or retune the mature common controller in the same round to improve kinematics; that would break common-path attribution and mix accepted-path execution with recovery.
4. Do not add a kinematics penalty weight, curvature score weight, profile AUC, or threshold search to rescue v35.
5. The two known V33 kinematics counterexamples remain diagnostics only; do not convert them into outcome-tuned hard promotion gates.

## Promotion protocol

The counterfactual48 Stage-1 gate is **identical** to V33/V34. Any failed condition archives the method. Only a passing method may enter fresh37, and only a fresh37 pass may enter historical exact200 development confirmation. The launcher enforces this fail-closed.

If both WK-ROSH and CPOSH fail Stage-1, stop the ROSH/EOSH/guard family rather than tuning thresholds. The next branch should move to reachable proposal/support construction or a higher-fidelity physical-transition representation. Accepted-path kinematics remains an independent secondary problem for a later isolated version.

## Validation

- V16.8.35 new tests: 6/6 passed.
- V16.8.25→35 focused semantic/integrity suite: 49/49 passed.
- Python compilation: passed.
- launcher shell syntax: passed.
- full repository currently stops at an external-baseline collection import error (`candidate_geometry_finite`) that is reproduced unchanged in the uploaded V16.8.34 source tree, so it is historical and not attributed to V35.
