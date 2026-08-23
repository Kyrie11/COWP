# COWP Algorithm Changelog — v16.8.26

## Scope

v16.8.26 is deliberately a **no-data-rebuild** release. It reuses the existing v16.8.24 train/val/developer-held-out caches and the existing `cowp_all_best.pt`. It does not change label semantics, natural-root construction, RCOT/BCOT targets, proposal generation, or candidate-cache schema.

The release answers the v16.8.25 CTU experiment, repairs missing strict-Waymax attribution observability, adds a narrowly scoped fallback-only physical-risk probe, and removes avoidable JAX↔host synchronization overhead in online evaluation.

## Evidence received from v16.8.25

### CTU is rejected as the default selector

The paired learned-offline comparison kept the certificate invariant but worsened every relevant selected-plan metric.

Validation:
- EP: `0.642943 -> 0.522353` (`-0.120590`).
- PBTR: `0.465432 -> 0.482716` (`+0.017284`).
- FSR: `0.707524 -> 0.717233` (`+0.009709`).
- NCF-selection recall given available: `0.795380 -> 0.768977` (`-0.026403`).
- non-coercive progress regret: `0.295410 -> 0.449539` (`+0.154129`).
- certificate/fallback metrics were invariant.

Developer-held-out:
- EP: `0.615471 -> 0.500354` (`-0.115117`).
- PBTR: `0.471074 -> 0.490702` (`+0.019628`).
- FSR: `0.700811 -> 0.715010` (`+0.014199`).
- NCF-selection recall given available: `0.788770 -> 0.751337` (`-0.037433`).

Strict exact-ID Waymax, same 200 IDs and 80 steps/episode:
- COWP: CR `0.285`, collision `0.225`, offroad `0.095`, kinematics infeasible `0.105`, EP `0.936086`.
- CTU: CR `0.300`, collision `0.230`, offroad `0.100`, kinematics infeasible `0.100`, EP `0.859208`.
- Paired finite-EP delta CTU-COWP: mean `-0.076878`, bootstrap 95% CI approximately `[-0.161, -0.018]`.
- CR discordance: 3 COWP-safe episodes became unsafe under CTU, 0 reversed (exact McNemar p=0.25 on only 200 scenes; not a physical-safety significance claim).

**Decision:** keep the original certificate-compatible set-preservation frontier. The BCOT/transport quantity has useful within-feasible-set ordering information; it is not merely duplicated punishment. The paper should not describe the implementation as a pure `hard certificate -> unconstrained planner-score argmin`. The correct hierarchy is `semantic certificate -> certificate-compatible risk/utility frontier -> selection`.

### Strict Waymax run is complete, not truncated

Both uploaded strict JSON files contain exactly 200 requested/resolved unique IDs, the same ID order/hash, and 80 simulator steps for every episode (16,000 policy steps per method). The terminal/network interruption therefore did not invalidate those saved results.

### Strict online fallback is now the most urgent no-rebuild diagnostic

For COWP on the exact 200 scenes:
- fallback episode rate: `0.915`;
- fallback step rate: `0.62625`;
- accepted priority-NCF steps: `0.37375`;
- no-certificate -> least-coercive conventional fallback: `0.586875`;
- no-conventional -> least-coercive valid: `0.00775`;
- no-valid candidate: `0.031625`;
- mean valid candidates: `30.83`;
- mean conventionally safe candidates: `6.74`;
- mean certificate-accepted candidates: `4.05`.

This does **not yet prove** that collision/offroad is caused by fallback, because v16.8.25 JSONs aggregate fallback and physical events at different granularities. Since >90% of episodes contain some fallback, episode-level co-occurrence is uninformative. v16.8.26 adds first-event temporal attribution before changing the main algorithm.

### Outcome head: promising diagnostic, not yet a main certificate

Existing checkpoint, val/held-out:
- collision AUPRC about `0.585 / 0.598`;
- offroad AUPRC about `0.429 / 0.436`;
- collision-or-offroad AUPRC about `0.728 / 0.733`.

This is stable enough to justify an **opt-in fallback-only ranking probe**, but insufficient for a hard physical certificate. In particular, the current evidence lacks low-FPR recall and probability calibration, and offroad discrimination is modest.

## Algorithm probe: Fallback-Only Outcome Risk (`cowp_fallback_outcome`)

The new method is intentionally not a replacement for COWP.

### Invariant certified path

When at least one candidate survives the original protected-priority RCOT/BCOT certificate and COWP frontier, `cowp_fallback_outcome` is required to be identical to `cowp`:
- same certificate;
- same physical action/rule shield;
- same set-preservation frontier;
- same selected candidate.

Outcome risk is explicitly zeroed inside the certified-path frontier for this method. This prevents the physical-outcome head from contaminating the paper's non-coercion mechanism.

### Changed uncertified path only

If the semantic certificate is empty, the existing explicit uncertified fallback objective can now observe the trained outcome head:

`fallback_score = transport-UCB + rule/action/pressure terms + fallback_outcome_weight * outcome_risk + weak utility/stop-like tie terms`.

The default COWP method remains unchanged: with `--outcome-risk-penalty 0`, its outcome risk remains zero. Only `cowp_fallback_outcome` evaluates collision/offroad probability while keeping main-path outcome influence disabled.

This is a **diagnostic repair of the recovery policy**, not a new causal certificate and not a CCF-A contribution claim unless strict paired evidence supports it.

## New physical-failure attribution observability

Waymax standard-metric accumulation now records the first positive step for event metrics. The evaluator emits compact per-scenario diagnostics:
- fallback step rate and first fallback policy step;
- no-certificate / no-conventional / no-valid fractions;
- accepted-priority-NCF fraction;
- mean accepted/conventional/valid candidate counts;
- selected certificate/action/rule/outcome risk means;
- collision/offroad/kinematics event flags;
- fallback fraction before the first physical event;
- whether the action immediately before the first event came from fallback.

This allows the next experiment to distinguish:
1. failures occurring primarily after the certificate has emptied (fallback-side bottleneck);
2. failures occurring while an accepted COWP candidate is being executed (accepted-selector / physical-shield bottleneck);
3. similarly high failure rates for conventional/planner baselines (common candidate/action-projection/online dynamics bottleneck).

## Outcome-head diagnostics expanded

Learned-offline evaluation now additionally reports, for collision, offroad, and unsafe-union:
- positive prevalence;
- Brier score;
- ECE-10;
- recall at FPR <= 5%;
- recall at FPR <= 10%.

Promotion to any hard or all-path physical guard is prohibited until these diagnostics are acceptable on validation and stable on developer-held-out data.

## Waymax runtime engineering

### Exact-ID TFExample index plumbing

`04_eval_closed_loop.py` now accepts `--tfexample-index-jsonl` and passes it to `waymax_state_generator_for_sids`. This lets exact-ID evaluation scan only TFRecord shards known to contain requested scene IDs. `78_build_tfexample_id_index.py` builds/reuses this index. This is an index-only operation and does not reconstruct labels/caches.

### Runtime profiler

`--profile-waymax-runtime` records per-scenario:
- TFExample `next()`/decode time;
- environment construction/reset;
- policy time;
- environment step time;
- standard metric time;
- total scenario time.

`--profile-waymax-sync` forces device synchronization around timing sections and is intended only for small profiling subsets.

### Batched JAX -> host state transfer

The online policy previously called `jax.device_get` separately for trajectory `x/y/yaw/vx/vy/length/width/height/valid`, and also repeated trajectory/timestep extraction. v16.8.26:
- fetches trajectory leaves as one pytree/device-get;
- extracts the trajectory/timestep once per policy call;
- caches the SDC index for the scenario.

This is intended to be exact-equivalent and targets the serial cross-framework synchronization overhead repeated over 16,000 policy calls per 200-scene method.

### Two-A30 throughput mode

The launcher provides three small profiling modes before any long run:
1. current stable split: PyTorch on A30-0, JAX on A30-1;
2. one-GPU co-location: PyTorch+JAX on the same A30;
3. two parallel exact-ID shards: one process per A30, each co-locating PyTorch+JAX and processing half the scenarios.

Do not assume parallel co-location is faster until the 12-scene profile passes memory and throughput checks. JAX preallocation remains disabled.

## Historical-attempt guardrails

The following conclusions remain active and must not be retried without new evidence:
- CTU / pure certificate-then-planner-score is rejected.
- universal/all-critical NCF is too conservative as the default hard gate.
- soft burden cost alone does not replace protected hard feasibility.
- threshold/budget tuning cannot overcome a fixed proposal-sufficiency floor.
- PCHR did not generate useful scene-level NCF recovery in prior probes.
- PSY/RMR-style primitive additions previously failed to move the scene-level proposal ceiling sufficiently.
- affected-root vs unsafe-root has no independent burden-only supervision in the current dataset and is not a supported headline claim.
- the uploaded `v16_8_25_mcfc_proposal_probe_val.zip` is **not MCFC evidence**: `profile_labels.jsonl` is empty and there is no paired proposal probe/source-ablation/promotion output.

## Required next evidence

No planner repair and no dataset/proposal rebuild yet.

1. `outcome_diag`: measure prevalence/calibration/low-FPR recall and verify fallback-only method keeps learned-offline certificate behavior.
2. runtime profiles on the same small exact-ID subset.
3. strict paired 200-ID Waymax for:
   - `cowp`;
   - `cowp_fallback_outcome`;
   - `conventional_safety`;
   - `planner_score_only`.
4. compare failure timing against fallback and compare COWP against conventional/planner physical rates.

Only after this localization should the next main algorithm be chosen.
