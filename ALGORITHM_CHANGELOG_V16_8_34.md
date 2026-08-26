# ALGORITHM_CHANGELOG V16.8.34 — Execution-Conditioned Recovery Option Spectrum

## Triggering evidence

V16.8.33 results are reliable for attribution. `equivalence16` preserves the mature COWP common path (16 scenes / 1120 fields / 0 mismatch); SDH/ROSH counterfactual48 shards are 24+24 disjoint and merged summaries recompute exactly. Re-running the V33 analyzer from immutable V29–V32 references yields 0 recursive field mismatches against the uploaded analyzer JSON.

The V33 preregistered Stage-1 gate is unchanged: retain >=5/10 old RVR rescues, avoid >=7/9 old RVR induced, net remove >=3 COWP collisions, kinematics regression <=1 scene, paired mean EP delta >=-0.05, intervention >0. Any failed check archives the branch.

### V33 SDH — archive

`cowp_sov_dominance_hysteresis` retains 4/10 old rescues and removes only 2 net COWP collisions. Although it avoids 7/9 old induced, has no net kinematics regression and EP delta≈-0.0070, it fails two hard gates. Mode hysteresis alone is insufficient.

### V33 ROSH — near-positive mechanism, failed policy

`cowp_recovery_option_spectrum_hysteresis`:

- Collision 34/48 → 29/48 (7 rescue / 2 induced, net -5);
- old RVR rescue retained 6/10;
- old RVR induced avoided 7/9;
- Kinematics 6/48 → 8/48 (**+2 scenes, hard gate FAIL**);
- paired mean EP delta≈-0.02786, bootstrap95≈[-0.08755,+0.02881];
- recovery switch step rate≈7.84%.

Therefore the V33 implementation is **not promoted** and fresh37 is not run. The +1 kinematics threshold is not relaxed after observing this near-miss.

## New root-cause evidence

The two V33 kinematics regressions have different causes:

1. `29cd2aca8ae5e222`: first kinematics occurs while directly executing `no_conventional_use_recovery_option_spectrum_hysteresis`, macro `MERGE_AHEAD` — a direct recovery-interface problem.
2. `6992366c5c998d00`: first kinematics occurs later on `accepted_priority_ncf`, `LANE_CHANGE_RIGHT`, with `conventional_safe=true`, `collision_safe=true`, prefix=80 — recovery changed closed-loop/controller state and a later nominally certified candidate became execution-infeasible.

Code audit shows why V33 can miss both. The real controller is stateful in `previous_longitudinal_accel`, but V33 successor spectrum only conditions on simulator `agent_state + emitted_target`. In addition, the V33 spectrum counts nominal `valid & roadgraph_safe` candidate macro survival; historical candidate validity deliberately ignores the initial jerk transient, so nominal semantic support is not equivalent to immediate controller-realizable support.

The dominant bottleneck is therefore refined from semantic Recovery Option Spectrum to:

**Control-Realizable / Execution-Conditioned Recovery Option-Set Feasibility.**

## Frozen components

No change to:

- compact-5k data/labels/checkpoint;
- natural roots;
- RCOT / BCOT;
- protected-priority hard certificate;
- certificate-compatible set-preservation frontier;
- 8 s conventional safety contract;
- proposal families;
- outcome settings;
- V27 conventional-integrity repair;
- V28 no-valid bounded execution repair;
- accepted-path execution/certificate logic.

## Branch A — Transition-Guarded ROSH (TG-ROSH)

Method: `cowp_transition_guarded_rosh`.

This is a root-cause diagnostic. It preserves the V33 nominal successor recovery spectrum but adds a hard current controller-transition feasibility component for base and RVR candidates. Feasibility reuses existing acceleration/deceleration, jerk, yaw-rate/Waymax delta-yaw and lateral-acceleration limits; no new tunable threshold is introduced.

The recovery alternative may enter only when the product `(current transition feasibility, future nominal profile)` weakly dominates base and at least one component is strict. Active mode continues under weak dominance and exits on any regression.

## Branch B — Executable Option-Spectrum Hysteresis (EOSH)

Method: `cowp_executable_option_spectrum_hysteresis`.

EOSH keeps the same current transition product constraint and upgrades the successor representation:

1. execute the actual controller-projected base/RVR target;
2. carry the emitted longitudinal acceleration into the successor as `previous_longitudinal_accel`;
3. regenerate the unchanged online physical proposal bank;
4. before a future candidate contributes its macro to the recovery survival curve, require its first desired step to be reachable from the carried controller state under the existing hard controller limits;
5. count distinct non-PAD executable/roadgraph-safe macros whose collision-safe prefix survives each horizon;
6. use the same pointwise partial order + strict-entry / weak-continue / regression-exit state machine.

EOSH does **not** change conventional-safe or NCF labels, proposal geometry, RCOT/BCOT, or training data. It does not use logged future, profile area weighting, horizon discount, risk scalarization, learned viability heads, dwell time, or hysteresis margins.

## New diagnostics

V34 records:

- base/RVR current controller-transition feasibility;
- transition feasibility delta;
- executable-profile usage;
- successor transition-feasible candidate count;
- roadgraph-safe candidates rejected by transition realizability;
- selected candidate transition feasibility;
- TG/EOSH recovery switch and hysteresis statistics.

The analyzer also reports whether the two known V33 kinematics-induced scenes are avoided, but this is diagnostic only and is **not added to the preregistered gate**.

## Preregistered protocol

Stage-1 counterfactual48 gate is exactly the V33 gate; it is deliberately not weakened after the ROSH +2 kinematics near-miss. Only Stage-1 pass methods may run fresh37. The launcher remains fail-closed.

Fresh37 uses the pre-existing V33 rule: no net collision/CR harm, offroad and kinematics regression <=1 scene, paired mean EP delta>=-0.03, intervention>0. It is development evidence, not publication holdout. exact200 remains development confirmation and is allowed only after fresh37 pass.

## Prohibited rescue directions

In addition to prior prohibitions, do not:

- relax the kinematics Stage-1 bound from +1 to +2;
- add a weighted kinematics penalty to V33 ROSH;
- treat current action-risk gating as sufficient for the downstream `6992366c...` failure;
- relabel controller-transition-filtered candidates as conventional/NCF;
- tune the controller-transition test; it must reuse existing hard execution limits.

## Local validation

- V34 tests: 4/4 passed.
- V34+V33 focused tests: 9/9 passed.
- V16.8.25→34 focused semantic/integrity sanity: **43/43 passed**.
- Manifest hashes: exact200/equivalence16/counterfactual48/fresh37 all pass.
- `py_compile`: pass.
- launcher `bash -n`: pass.
- analyzer static smoke: pass and correctly preserves the V33 +2-kinematics gate failure when fed the V33 ROSH result as a synthetic V34 candidate.
- Full repository `pytest -x` reached ~23% before the 120 s execution limit with no failure observed; this incomplete run is not reported as a full-suite pass.
