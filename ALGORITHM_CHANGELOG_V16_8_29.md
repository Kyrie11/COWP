# COWP Algorithm Changelog — v16.8.29

## Scope

v16.8.29 is the first algorithmic branch authorized after the v16.8.27/v16.8.28 integrity repairs. It is based on the clean v16.8.28 exact-200 physical attribution. It **does not** rebuild the dataset/cache, retrain the checkpoint, alter natural roots, RCOT, BCOT, the protected-priority certificate, the set-preservation frontier, or the full-horizon conventional-safe definition.

The release has two separable changes:

1. an execution-equivalent CPU acceleration of the online causal collision audit; and
2. an opt-in `cowp_recursive_viability` method that changes selection **only when the full conventional-safe set is empty**.

## v16.8.28 result integrity

Independent audit of the uploaded parallel-2 result package found:

- exact manifest: 200/200 unique IDs, logical SHA256 `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`;
- every method has two disjoint 100-ID shards whose union is exactly the manifest;
- all four merged result files contain the same exact scenario set;
- CR/collision/offroad/kinematics/EP recompute exactly from the 200 scenario rows (`max_abs_diff=0`);
- EP is finite on the same 196 scenes for all four methods;
- on every scenario row, `emergency_action_step_rate == zero_valid_candidate_step_rate == no_valid_step_rate`, confirming the v16.8.28 no-valid repair semantics;
- first-event indexing remains action-consistent: policy action is emitted, Waymax steps, then the 1-indexed first-positive metric is mapped back to the action at index `first_step-1`.

The uploaded archive does **not** contain the user's separately requested single-process `split` rerun outputs or runtime profile JSONs. Therefore parallel-2 vs split numerical replication and a fresh v16.8.28 wall-time comparison cannot be verified from this archive. This does not invalidate the internally consistent parallel-2 exact-200 attribution.

## Clean pre-registered branch decision

The previous integrity versions pre-registered three interpretations:

1. failure from `fallback=true && selected_conventional_safe=true` → Recovery Certificate;
2. failure from `accepted_priority_ncf` → accepted-path execution viability;
3. comparable failure in common baselines → common online proposal/action interface.

v16.8.28 selects the third branch for collision, more specifically **online conventional-feasible support collapse**.

For COWP, 34 collision episodes localize as:

- 32/34: `no_conventional_use_least_coercive_valid` immediately before first collision;
- 2/34: `no_valid_candidate` emergency bounded stop;
- 0/34: conventional-safe fallback;
- 0/34: accepted certified path.

For the conventional baseline, 44 collision episodes localize as:

- 43/44: `baseline_use_stop_like` after the conventional pool is empty;
- 1/44: no-valid emergency;
- 0/44: a conventional-safe action immediately before first collision.

COWP collision episodes spend on average `0.80993` of their policy steps in zero-conventional states versus `0.50512` for non-collision episodes. All 34 COWP collision episodes encounter a zero-conventional state no later than the action preceding first collision. The first collision is not an immediate execution artifact: median first-positive overlap step is 50.5 (mean 53.35, range 32--78).

Zero-conventional exposure is also highly correlated across methods (`r≈0.87--0.97` pairwise), establishing that this is primarily a common online feasibility-support property rather than a COWP-only certificate pathology.

## What is frozen

The following components remain protected from opportunistic retuning:

- current dataset/labels/cache and checkpoint;
- typed natural roots and natural basis;
- RCOT same-root transport;
- BCOT structured protected-priority certificate;
- protected-priority hard feasibility with all-critical diagnostic;
- certificate-compatible set-preservation frontier (CTU remains a negative ablation);
- outcome head as diagnostic/probe only.

The clean fallback-outcome rerun now makes the last item stronger: relative to COWP it changes collision `0.170→0.165` (McNemar `p=1.0`) and CR `0.195→0.185` (`p=0.774`) while reducing paired EP by `-0.02218`, bootstrap 95% CI `[-0.03896,-0.00726]`. It is not promoted to recovery or hard physical certification.

## Secondary bottleneck retained for later

Kinematics is not localized the same way as collision. Of 25 COWP kinematics-infeasible episodes, 16 first events follow `accepted_priority_ncf`, and 17/25 preceding candidates are conventional-safe. This is evidence for a secondary accepted-path/action-projection execution-viability issue. It is deliberately **not** mixed into the collision-recovery change in v16.8.29. Once the dominant collision bottleneck is resolved, this should receive its own orthogonal probe.

## Remaining ambiguity inside zero-conventional states

`conventional_safe` is currently the intersection of a local roadgraph screen and an 8-second causal constant-velocity collision screen. Therefore an empty conventional set can mean different things:

- no candidate survives the collision screen (`collision_empty`);
- no candidate survives the roadgraph screen (`roadgraph_empty`);
- neither side has any survivor (`road_and_collision_empty`);
- both sides have survivors but no same candidate passes both (`intersection_empty`).

This is scientifically important. A long open-loop constant-velocity screen can be conservative relative to a controller that replans every 0.1 s; alternatively the candidate bank may genuinely lack a useful safe action. Direct proposal expansion or direct horizon shortening before distinguishing these cases would be confounded.

v16.8.29 records the two screen components and their decomposition without changing the full conventional-safe boolean.

## New algorithm probe: Recursive Viability Recovery (RVR)

New opt-in method: `cowp_recursive_viability`.

The certified path and conventional-safe fallback path are identical to COWP. RVR activates only when:

```text
certificate empty
AND conventional-safe pool empty
AND at least one dynamically valid candidate exists
```

For each dynamically valid candidate, the planner evaluates the first violation time under the **same causal collision model and same horizon/stride/buffers** already used by the conventional screen. Let `h_k` be the collision-safe prefix length.

Recovery is lexicographic:

1. if any dynamically valid candidate passes the roadgraph screen, restrict recovery to that set; otherwise do not invent road safety and retain the valid emergency pool;
2. retain only candidates attaining the maximum `h_k` in that pool;
3. use the existing COWP fallback score only as a tie-break inside the maximal-prefix set.

No new scalar weight, learned head, threshold, certificate relaxation, or candidate primitive is introduced. A candidate selected by RVR remains explicitly **uncertified**. It is never counted as conventional-safe or NCF.

The purpose is falsifiable: determine whether many closed-loop failures are caused by binary full-horizon feasibility collapse even though some candidates preserve substantially more causal time-to-violation for the next replanning step. If not, the next move should be proposal support / map-topology generation rather than another recovery ranker.

## New decomposition diagnostics

Per policy step and first physical event, v16.8.29 records:

- valid / roadgraph-safe / collision-safe / conventional candidate counts;
- maximum collision-safe prefix and selected prefix;
- selected roadgraph-safe and collision-safe flags;
- minimum selected collision-clearance margin;
- zero-conventional reason (`collision_empty`, `roadgraph_empty`, `road_and_collision_empty`, `intersection_empty`);
- RVR recovery step rate.

`81_summarize_recursive_viability.py` produces paired screen-decomposition summaries. `80_compare_waymax_physical_methods.py` now accepts `--recursive`.

## Exact-equivalent online speed optimization

v16.8.28 rebuilt nearby-agent ranking and constant-velocity futures for every candidate although they depend only on the current simulator state. v16.8.29:

1. builds the causal collision context once per policy step;
2. stacks the same nearby-agent futures and thresholds;
3. evaluates all agents for a candidate using a NumPy broadcast rather than a Python agent loop.

The conventional boolean remains the exact v16.8.28 inequality on the same sampled indices. A randomized reference regression compares the cached path against a literal v16.8.28 implementation over 64 random scenes/candidates. Focused sanity passes.

A local synthetic 64-agent/48-candidate microbenchmark shows about `7.3x` acceleration of the collision-audit component. This is **not** claimed as whole-Waymax speedup; the server-side 12-scene profiler remains the source of wall-time evidence.

## Faster experiment policy

Repeated 4-method × 200-scene runs are no longer required during mechanism maturation.

A deterministic development-only 64-ID panel contains:

- all 34 v16.8.28 COWP collision scenes;
- 30 non-collision scenes with the highest zero-conventional exposure.

Logical SHA256: `0b271cb30febe3feac3a35bb08bb8b9506b048cd6559d5d49bc046bc80c91567`.

This panel is **outcome-selected and therefore forbidden as publication evidence**. It is only a high-information debugging/promotion screen. Running COWP + RVR on it requires 128 scene-method rollouts versus the previous 800, a 6.25x experiment-level reduction before any runtime acceleration.

`analyze_diag64` first compares the newly run unmodified `cowp` path to a bundled v16.8.28 64-scene reference. Any mismatch aborts before RVR interpretation.

Only after a favorable mechanism signal should exact-200 COWP + RVR be run (400 scene-method rollouts, 2x fewer than the previous four-method panel).

## Novelty guard

RVR/maximal survival prefix by itself is **not** promoted as a CCF-A-level contribution. Recursive feasibility, predictive safety filters, reachability, and backup-set ideas are established fields. If RVR succeeds, the paper-level direction should be formalized as an orthogonal two-feasibility architecture:

```text
social feasibility: protected-priority same-root non-coercion (RCOT/BCOT)
×
physical recursive viability: causal recoverability under closed-loop replanning
```

A publishable extension should define a model-relative recursive viability/recovery condition, prove or calibrate the relevant guarantee under explicit assumptions, and show that it improves physical safety without weakening the non-coercive certificate. If the RVR probe fails, do not polish it; use the decomposition to move to proposal-space refinement or roadgraph/action-interface repair.

## Regression status

- new v16.8.29 tests: `5/5 passed`;
- packaged focused v16.8.25--v16.8.29 sanity: `20/20 passed`;
- the randomized v16.8.28 collision-boolean reference equivalence is included in the test set;
- a full-repository run was attempted in this environment but exceeds the available per-command execution window, so no new full-suite count is claimed here. The uploaded v16.8.28 baseline recorded `265 passed / 5 skipped / 8 historical failures` before these localized changes.

## Required next run

Run:

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh make_ids

# only if the TFExample index is absent
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh build_tfindex

# fast development gate; not paper evidence
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh viability_diag64_parallel2
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh analyze_diag64

# only after the dev64 mechanism signal is favorable
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh confirm200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh analyze_confirm200
```

Optional server-side speed measurement:

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh profile_parallel2
```
