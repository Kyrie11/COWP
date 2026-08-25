# COWP Algorithm Changelog — v16.8.29

## Scope

v16.8.29 is the first algorithmic probe after the v16.8.26 conventional-safety repair and v16.8.28 no-valid execution repair restored clean strict-Waymax attribution. It makes **one preregistered planning change only**: a recovery-viability bridge used exclusively when the full-horizon conventional-safe set is empty while dynamically valid candidates still exist.

The dataset, cached labels, checkpoint, natural-root construction, RCOT, BCOT, protected-priority certificate, all-critical diagnostic, main COWP certificate-compatible frontier, candidate families, and learned heads are unchanged. No retraining or cache/data rebuild is authorized.

A separate engineering-only optimization caches candidate-invariant conventional-screen state once per replanning step. Regression requires bit-exact candidate-bank equality relative to v16.8.28 semantics.

## Triggering evidence: v16.8.28 exact-200 is attribution-clean

The repaired parallel2 run passes the execution-integrity audit:

- exact manifest: 200 unique IDs, logical SHA256 `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`;
- four methods use exactly the same scenario-ID set and checkpoint;
- each pair of 100-scene shards is disjoint and covers the exact manifest;
- merged standard metrics reproduce exactly from per-scenario rows;
- physical attribution regenerates from the merged files;
- v16.8.28 bounded-stop provenance is explicit and PAD execution is absent.

Split-mode outputs are not present in the uploaded result package. This is not a scientific blocker because the successful parallel2 run already contains the complete paired exact-200 evidence. `parallel2` and `split` are alternate execution modes, not two required experiments.

## What the preregistered branch test resolved

v16.8.28 COWP:

- CR = 19.5%, collision = 17.0%, offroad = 3.0%, kinematics infeasible = 12.5%, EP = 1.0461;
- fallback step rate = 71.675%;
- zero-conventional-candidate step rate = 55.69375%;
- zero-valid-candidate step rate = 2.65625%;
- mean valid candidates ≈ 33.27, mean full-horizon conventional candidates ≈ 6.64.

For all 34 collision episodes, the immediately preceding action is fallback. Crucially:

- 32/34 are `no_conventional_use_least_coercive_valid`;
- 2/34 are the explicit no-valid bounded emergency stop;
- 0/34 preceding selected candidates are conventional-safe;
- 0/34 collisions are immediately preceded by an accepted protected-priority COWP candidate.

The same structural failure exists in the shared baselines. Conventional-safety has 44 collision episodes, 43 immediately preceded by `baseline_use_stop_like` after its full-horizon conventional set has collapsed. Planner-score-only has 49 collisions, with 47 accepted dynamic-valid candidates and 0/49 immediately preceding candidates conventional-safe.

Thus the preregistered decision does **not** support “the certified COWP path is the dominant collision source” and does **not** support “a truly conventional-safe recovery selector is choosing the wrong candidate.” It supports the common lower-layer branch:

> the online physical-feasibility/proposal interface loses full-horizon conventional support during receding-horizon closed-loop operation, then falls through to candidates that are dynamically valid but not physically certified over the full primitive.

## Negative result archived: fallback-only OutcomeHead

`cowp_fallback_outcome` does not earn promotion:

- collision changes only 17.0% → 16.5% (paired McNemar p = 1.0);
- CR changes 19.5% → 18.5% (p ≈ 0.774);
- kinematics slightly worsens 12.5% → 13.0%;
- EP decreases by ≈ 0.0222 with paired bootstrap 95% CI approximately [-0.0390, -0.0073].

This is now a clean negative probe rather than a result contaminated by the v16.8.26/27 execution bugs. Do not tune the outcome weight or promote the current outcome head to a hard physical certificate.

## Existing core mechanisms remain frozen

The historical learned-offline evidence remains stronger for RCOT/BCOT than for generic candidate classifiers. CTU also remains a negative ablation: replacing the certificate-compatible set-preservation frontier with planner-score argmin reduces progress and NCF retention. Therefore v16.8.29 does not change RCOT, BCOT, protected-priority semantics, or the post-certificate robustness frontier.

## New mechanism: Receding-Horizon Recovery-Viability Bridge (RVB)

### Motivation

The current online conventional screen asks whether an entire ~8 s primitive passes the causal collision and drivable-road audit. Waymax, however, executes only one 0.1 s action before replanning. In v16.8.28 the full-horizon conventional set is empty on ~55.7% of COWP policy steps even though the dynamic-valid set is empty on only ~2.7% of steps. This is a structural discontinuity, not raw candidate-count starvation.

### Definition

The main full-horizon COWP certificate is unchanged. RVB is evaluated only when:

1. no protected-priority certified candidate is selected;
2. the full-horizon conventional-safe set is empty; and
3. at least one dynamically valid candidate exists.

For each dynamic-valid, non-conventional candidate, RVB:

1. takes a short executable prefix using `online_recovery_commit_steps` (default 8 = 0.8 s, intentionally aligned with the existing action-risk horizon);
2. constructs a bounded smooth-stop continuation from the prefix endpoint with the existing `fallback_decel_mps2` primitive;
3. splices prefix + stopping continuation over the original horizon;
4. requires the spliced trajectory to pass the **same** roadgraph-drivability and causal constant-velocity/logged collision screens;
5. also requires the existing hard action-risk and rule-risk shields at selection time.

Only candidates satisfying this hard recourse test enter the recovery bridge set. Within the set, the existing fallback score is retained. No learned outcome score, no new scalar ranking weight, and no macro-name privilege can promote a candidate.

Fallback order becomes:

`certified COWP -> full-horizon conventional fallback -> recovery-viability bridge -> unrestricted dynamic-valid fallback -> bounded no-valid emergency stop`.

The original unrestricted-valid branch is deliberately retained after RVB. It is needed both as an emergency fallback and as a clean diagnostic: if RVB availability is low, the result should reveal proposal/geometry support failure rather than silently changing coverage.

## Why this is not “just another stopping trick”

The contribution being tested is not the smooth-stop primitive. That primitive already existed. The scientific object is an explicit **recourse set** that bridges two different feasibility horizons without weakening the social certificate:

- full-horizon protected non-coercive feasibility for other road users;
- short-horizon ego execution conditioned on preserving an explicitly audited physical recovery continuation.

Generic contingency/fallback planning already exists in the literature, so RVB is not to be claimed as novel merely because it carries a backup trajectory. Promotion requires evidence that the dual-feasibility decomposition explains and fixes the specific COWP closed-loop support collapse.

## Preregistered interpretation of v16.8.29

Only COWP and `cowp_recovery_bridge` need the next exact-200 comparison.

1. **RVB available/used frequently and collision falls with comparable EP:** promote the bridge as a candidate paper-level dual-feasibility component; then run multi-seed/publication-scale validation.
2. **RVB availability is near zero:** the dominant root is physical proposal/geometry support; redesign online physical proposal support rather than tuning selection weights.
3. **RVB is available but collision does not improve:** the common causal screen or candidate-to-action projection is mismatched to closed-loop physics; audit that interface rather than adding another fallback score/head.
4. **Kinematics remains concentrated on accepted protected-priority actions:** treat this as an orthogonal Execution-Viability Certificate branch in a later version; do not mix it into the collision experiment now.

## Secondary bottleneck retained for later

Kinematics is not the dominant collision source, but it is not solved. In COWP, 16/25 first kinematics violations occur immediately after `accepted_priority_ncf`; 17/25 preceding candidates are conventional-safe. This is evidence that the social/non-coercive certificate is not an execution-dynamics certificate. v16.8.29 intentionally does not solve this second problem, to preserve attribution.

## Engineering-only acceleration

v16.8.27 profiling showed CPU candidate construction consumed roughly 88% of policy time. v16.8.29 therefore caches, once per replanning step:

- lane-centerline mask;
- nearby-agent priority/nearest ranking;
- logged-causal / constant-velocity future arrays;
- collision radii and fixed collision-screen indices.

Each candidate then reuses this context. The equations, thresholds, agent ordering, and candidate screen results are unchanged. A regression compares cached and uncached candidate-bank outputs bit-exactly.

A local 32-agent + lane-map microbenchmark including cache construction measured about 0.282 s -> 0.254 s per synthetic candidate build (~9.9% reduction). This is a local CPU microbenchmark only, not a server wall-clock guarantee.

The recommended Waymax execution remains two independent scenario shards: one Torch+JAX process per A30. The next experiment evaluates only two methods instead of the four already needed for v16.8.28 attribution, so do not rerun fallback-outcome/conventional/planner-only.

## Regression contract

New focused tests require:

1. cached and uncached conventional-screen candidate banks are bit-exact;
2. a candidate rejected only for a far-horizon collision can enter RVB only when its short prefix + bounded-stop recourse passes the unchanged physical audit;
3. invalid or already-conventional candidates are not falsely promoted into RVB;
4. recovery-bridge usage/availability and first-event provenance reach scenario diagnostics;
5. selector ordering keeps RVB strictly after full-horizon conventional fallback and before unrestricted-valid fallback.

Packaged `sanity`: **20/20 passed**.

Full repository: **270 passed / 5 skipped / 8 historical failures**. The eight failures are unchanged from v16.8.28: six tests reference legacy launcher scripts absent from the supplied archive and two tests hard-code an old label-semantic fingerprint. No v16.8.29 functional regression is present.

## Required next run

No retraining. No dataset/cache/label rebuild. No BCOT/frontier/outcome retuning.

Recommended:

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh make_ids
# only if the existing TFExample index is missing:
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh build_tfindex
# optional 12-scene performance check:
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh profile_parallel2
# exact 200, two A30s, only COWP + RVB:
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh waymax_recovery200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh analyze_parallel2
```

Only if two-process co-location OOMs, use `waymax_recovery200_split` + `analyze_split` instead. Do **not** run both modes after one succeeds.
