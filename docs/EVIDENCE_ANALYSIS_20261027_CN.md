# RIDE current evidence audit (evidence(1).zip)

## Executive decision

The submitted evidence does **not** pass either scientific gate. It is an engineering/smoke-test checkpoint, not evidence for the RIDE claims:

- RQ1 (strict cross-response-model risk-ranking reversal): **not demonstrated**. The aggregate gate says `NO_CROSS_FAMILY_RANKING_REVERSAL_IN_THIS_PANEL`.
- RQ2 (ego-induced, progress-matched, avoidable third-party risk): **not demonstrated**. The aggregate gate says `NO_STABLE_PAIRED_THIRD_PARTY_COLLISION_EFFECT_IN_THIS_PANEL`.
- RQ3: **not run** (`rq3_strong_baselines_executed=false`; `H4_baseline_run=false`). Do not start algorithm design or claim a residual baseline gap from this panel.

The only usable CARLA data are eight valid convoy trials (one anchor, one seed, one repetition, two response families). They yield only two eligible paired contrasts (both `mild_early` vs `mild_late`), both with zero collision difference and zero proxy-effect difference. This is insufficient sample size and does not support a negative claim either.

## CARLA trial accounting

`all_scenes.jsonl` contains 81 trials across four archetypes; 73 fail immediately with the same software exception:

```
AttributeError: 'Rotation' object has no attribute 'y'
ride/carla_scenarios.py:264
return min(ahead, key=lambda v: yaw_diff(v.transform.rotation.y, ...))
```

The CARLA 0.9.16 API exposes `Rotation.yaw`, not `Rotation.y`; therefore `cut_in` (8/8 failed), `intersection` (17/17 failed), and `support_gap` (48/48 failed) have no executed trajectory and no causal result. `convoy` happened to run through a code path that avoided this call (8/8 valid), but this does not repair the other scenes.

The aggregate audit reports:

| quantity | value |
|---|---:|
| total CARLA trials | 81 |
| preregistered paired contrasts | 16 |
| eligible task-matched contrasts | 2 |
| invalid contrasts | 12 (`invalid_trial`) |
| other ineligible contrasts | 2 (`progress_unmatched`, `initial_speed_mismatch:follower_b`) |
| strict RQ1 reversals | 0 |
| stable RQ2 collision effects | 0 |
| cross-family matched effects | 0 |
| bootstrap clusters | 1 anchor |

The two eligible convoy contrasts are:

- TM, mild early vs mild late: progress delta 2.316 m; no ego/non-ego collision in either arm; both have hard-braking proxy true; signed collision/braking/TTC effects all 0.
- BehaviorAgent, mild early vs mild late: progress delta 2.967 m; same no-collision and hard-braking result; signed effects all 0.

The early/late pair is excluded because the ego progress differs by 12.68 m for TM and the BehaviorAgent arm has `initial_speed_mismatch:follower_b`. Thus the apparent panel is not 16 usable comparisons; it is two valid comparisons from one geometry/seed/repetition.

A cluster bootstrap with one cluster returns estimate 0 and CI [0, 0]. This is a degenerate resampling artifact, not statistical confirmation. Increase anchors, seeds, and repetitions before reporting uncertainty; a zero-width interval must not be presented as evidence of a stable null.

The convoy traces also show a likely metric-design problem: every valid profile has `hard_brake=true` for both followers, with minimum acceleration often exactly about -48.15 m/s², regardless of the profile. This suggests a controller/simulation transient or threshold artifact rather than a discriminative third-party burden signal. It should be debugged with action-independent baseline runs, acceleration clipping/physics logging, and a sensitivity analysis of the hard-brake threshold.

## Four-scene construction audit

### `convoy`

The straight-lane three-car geometry is physically executable and all eight trials complete, but it is an explicitly designed stress test, not a WOMD replay. One anchor/seed/repetition cannot establish a population-level effect. The response families are two CARLA policy implementations; neither is an independently calibrated human counterfactual model. The current result is therefore: *the implementation can run a convoy and its two families agree on the selected proxy ordering in this panel*.

### `cut_in`

The manifest has a plausible two-lane setup (ego on road 43/lane 5, followers on lane 6, target lane 6), but all eight trials abort before the first trace step because of `Rotation.y`. No lane change, induced braking, collision, or ranking result has been observed. After fixing the API call, verify that an actual lateral cut-in occurs (lane occupancy/lateral offset and time of merge), rather than only a target-waypoint change.

### `intersection`

The manifest lists distinct road IDs and crossing-like coordinates, but all 17 trials abort at the same waypoint-selection exception. In addition, `traffic_light_legality_protocol` is `unverified_red_trigger_motion_proxy_only`; no claim about legal red-light interaction can be made. After fixing execution, log junction entry/exit, conflict-point occupancy, signal state, and route-arc progress. Initial-heading projection alone is not valid through a turn.

### `support_gap`

The support-gap manifest uses the **same ego/follower coordinates and waypoint identities as `convoy`** (Town04 straight-lane anchor). As currently constructed it is not an independent gap/merge geometry. All 48 trials abort at the same exception, and the bank ablation explicitly reports `comparison_valid=false`, `all_candidate_executions_recorded=false`, and small/large-bank no-witness status `UNKNOWN`. No finite-bank miss and no positive safe witness are established. To support the intended claim, construct a real gap/merge or interaction geometry, execute every candidate under identical initial state, and record complete traces for every candidate; a missing file/exception cannot be interpreted as UNSAT.

## WOMD/Waymax evidence

`observational_response_fit.json` has 320 attempted and 310 eligible development records, but it explicitly conditions on the logged ego future and says `No measured do-effect; fit quality does not establish human counterfactual validity`. Aggregate 1 s ADE (m): constant velocity 0.154 (median 0.143), cautious 0.294, assertive 0.336, heterogeneous 0.302, TTC-gap 0.163. Constant velocity is better than cautious/assertive/heterogeneous on 282/310, 290/310, and 288/310 scenes respectively (TTC-gap 148/310). These are observational fit results, not evidence that any model is the correct response under a changed ego action; they should be used to reject premature calibration claims and to design the counterfactual protocol.

`native_panel.json` attempts 22 WOMD records, completes 17, and has 5 missing-file errors. `any_all_regime_matched=0` and `scientific_claim_allowed=false`. The selected IDMRoutePolicy follows logged spatial routes; non-ego actors are log experts; selected trajectories are reported off-road (`selected_road_valid=false` and often hundreds of off-road counts); progress is initial-heading projection rather than route arc length; only one verified v1.3.1 TFExample record is documented; no GameFormer or RA-CBF reproduction is present. Some rows show collision differences between cautious/assertive, but these cannot be interpreted as RQ1/RQ2 because the regime action matching and map/route validity gates fail.

## What the current evidence can and cannot claim

Can claim: the audit layer is conservative, catches invalid CARLA executions and task-mismatch, and prevents invalidating model differences from being mislabeled as human causality. One convoy smoke panel is executable. The observational WOMD fit pipeline runs and demonstrates that simple observational fit does not identify a counterfactual response model.

Cannot claim: four verified failure scenarios; a human behavioral model; a cross-model ranking reversal; a stable avoidable third-party collision effect; a finite response-bank impossibility; or a strong-baseline failure. The current absence of a positive result is mostly due to invalid/underpowered trials, not evidence that the RIDE hypothesis is false.

## Minimum evidence repair before RQ3

1. Fix `Rotation.y` -> `Rotation.yaw` and add a unit/integration assertion that every scenario emits at least one trace step and no `simulation_exception`.
2. Re-run each scene with strict fresh-world reload (or prove actor reset restores all hidden state), >=10 independent anchors per archetype, >=5 seeds and >=3 repetitions. Keep all metadata frozen.
3. Make task matching explicit: same initial state, same ego progress tolerance, same route-arc metric, and equal action duration; exclude any arm with initial-speed mismatch.
4. Replace the support-gap anchor with a genuine gap/merge/crossing geometry, and record all finite-bank candidate traces. Report `UNKNOWN` for missing executions.
5. Define RQ1 as a planner-choice reversal on identical candidate actions (not post-hoc ranking of outcomes) and use a pre-specified scalar/lexicographic risk with uncertainty. Require strict reversal plus bootstrap CI/paired test across independent scenes.
6. Define RQ2 with a counterfactual triple: baseline ego action induces third-party burden/risk, a task-matched alternative avoids it, and both are safe/valid under the same response model. Report collision, TTC, jerk/deceleration, progress, and route completion separately.
7. Only after RQ1/RQ2 positive cases survive independent WOMD/ReactSim reactive evaluation should RQ3 compare locked, properly adapted strong baselines.
