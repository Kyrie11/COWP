## v16.8.18 — Sparse-Probe Pipeline Integrity, Location-Aware Scenario Reads, and Semantic-Failure Classification (2026-08-14)

Re-analysis of the uploaded v16.8.17 smoke artifacts shows that the current blocker is **not a newly demonstrated label-semantic or model-support failure**.  The uploaded `fastpath_semantic_equivalence.json` compared zero scenes: all 12 requested reference scene ids were missing, while the no-fastpath candidate side had no missing scenes and there were no tensor/key mismatches.  The uploaded fresh-build profile contains zero terminal rows and `build_fresh.log` ends during the initial full validation-split scan, before any 96-scene completion marker or downstream support audits.  Consequently no `model_support_audit.json`, `base_screen_verdict.json`, or `v16_8_17_smoke_verdict.json` was produced.  `strict` then reported a missing smoke verdict as a downstream consequence of this incomplete upstream pipeline.

v16.8.18 therefore makes **no changes to natural/response/witness/candidate label semantics or scientific thresholds**.  The label-semantic fingerprint remains exactly `adcea5cb927d4c06c7f667725ce1c5b7b62808d6bd2e84244149d01ab25a1fa0`, so the last fully completed v16.8.16 smoke remains the nearest valid scientific evidence: on mechanism-auditable criticals, rootless=0, <2 low-burden roots=0, PRIO/response/transport/witness targets are non-degenerate and FASTPATH A/B was bitwise equal.  v16.8.17's coverage policy has simply not yet been exercised by a complete fresh v16.8.17 smoke build.

Engineering repairs:

1. Added a reusable **Scenario location index** (`scenario_id -> TFRecord file + record_index`). Sparse smoke/strict/train-pilot builds now read only the shards/records containing requested scene ids instead of depending on where targets appear in an interleaved full-split scan.
2. Sparse builds use `--require-all-allowed-resolved` and are immediately checked by a new integrity audit. Requested ids never observed in the profile are classified as an interrupted/unresolved pipeline; resolved filters/errors and corrupt/missing NPZs are reported separately. Model-support audits do not run until this precondition passes.
3. FASTPATH reference selection no longer trusts directory existence. A partial current-version label directory cannot shadow the complete reviewed v16.8.16 reference. Reference and candidate NPZ sets are integrity-checked before comparison.
4. Semantic equivalence now distinguishes `reference_build_incomplete` / `candidate_build_incomplete` from a true `semantic_mismatch`. Missing scenes are explicitly **not** evidence that tensor values changed.
5. Smoke/strict/train-pilot wrappers write stage-aware `*_pipeline_status.json` files on abnormal exit. A missing composite verdict therefore points to the exact upstream pipeline stage rather than being confused with a scientific FAIL verdict.
6. The policy-only smoke re-audit captures scientific audit return codes and still writes a composite verdict. `smoke` now prefers this verified v16.8.16-label re-audit when the reviewed source artifacts are available; `fresh-smoke` is provided when a new label build is actually required.

Local regression after the pipeline changes: **224 passed, 5 skipped**; `compileall` and v16.8.18 smoke/strict/train-pilot/full orchestration shell syntax checks pass.

## v16.8.17 — Evidence-Coverage Promotion Contract and Policy-Only Smoke Re-Audit (2026-08-14)

Fresh re-analysis of the uploaded v16.8.16 96-scene smoke shows that the previous natural-basis collapse is repaired **on the auditable support**: 517/538 selected critical actors are mechanism-auditable, and among those actors rootless=0, <2 natural roots=0, <2 low-burden roots=0, protected PRIO coverage=99.75%, the 32-slot relevant-pair response bank is complete, and all active natural/response/witness/transport target families are non-degenerate. Proposal/causal smoke also passes and the exact fast-path A/B is bitwise-equivalent. The sole composite blocker is evidence coverage: 21/538 selected critical actors (3.90%) have an unknown mechanism target, causing only 80/96 scenes (83.33%) to carry a complete candidate-level certificate.

The 21 unknown targets are not treated as failed roots. Profile diagnostics split them into 11 actors with >=60 valid future samples but no full routable lane and no non-degenerate factual route geometry, plus 10 actors with only 17--56 valid future samples. Manufacturing six counterfactual roots from a stationary/degenerate single factual path, or extrapolating an incomplete future, would create unsupported option-space evidence. These actors therefore remain selected criticals for physical safety but are explicitly masked from mechanism/certificate supervision.

The previous coverage gates were internally inconsistent with the active six-critical scene structure. At the uploaded smoke's mean 5.60 selected criticals/scene, a 1% per-critical unknown rate corresponds to only about 94.5% all-critical scene completeness under an independence reference, while the old strict gate simultaneously required 98% complete scenes (implicitly about 0.36% per-critical unknowns). v16.8.17 replaces that contradictory pair with an explicit evidence-support contract: >=95% per-critical mechanism coverage and >=75% complete candidate-certificate scenes. The latter is the same order as 0.95^6=73.5% for the configured maximum six criticals; it is a planner-supervision support floor, not a causal-claim metric. Auditable rootless and <2-low-burden counts remain zero-tolerance.

Small-smoke coverage uses a Wilson gross-failure screen; strict/train-pilot use point estimates. A new hard-vs-random missingness-bias audit prevents promotion by disproportionately masking difficult scenes (smoke caps: 3 percentage points for per-critical unknown-rate gap and 10 points for complete-scene gap; strict/pilot: 3 and 8 points). The uploaded smoke has only ~0.88 pp and ~4.17 pp gaps respectively, so its unknown labels are not strongly concentrated in the hard stratum.

No label-producing file is changed in v16.8.17. A separate label-semantic fingerprint therefore allows the existing reviewed v16.8.16 smoke NPZ labels to be **policy re-audited without rebuilding them**. Full code/provenance fingerprints remain required for newly built strict, train-pilot and full caches. Full-core label/tensor audits use the same 95%/75% point coverage contract.

Local regression: **220 passed, 5 skipped**; compileall and all v16.8.17 promotion shell syntax checks pass.

## v16.8.16 — Full-Horizon Auditability, WOMD Driveway Evidence, Exact OBS Geometry, and Statistical Smoke Promotion (2026-08-14)

The v16.8.15 96-scene smoke passed training supervision but was blocked by two independent gates: a borderline PBTR point estimate (23/45=0.5111 versus the smoke maximum 0.50) and genuine natural-basis incompleteness (13/536 auditable critical agents rootless and 18/423 protected agents without a PRIO root).  All 13 rootless cases used `logged_geometry_neutral_timing` and were map-rejection dominated.

v16.8.16 fixes a construction inconsistency where any non-empty lane polyline disabled empirical fallback even when that lane was too short to retime for the full 8 s horizon.  Auditability and empirical-fallback eligibility now use the actually retimable `map_refs`, so short 35/36-step actors without a full route become explicit mechanism-unknown targets while 78--80-step factual geometry can provide a narrow empirical route witness.  Canonical OBS now preserves WOMD factual positions instead of reintegrating logged velocity.  The Scenario parser also retains official WOMD driveway polygons and map compliance accepts the union of the existing lane corridor and explicit driveway polygons without inferring drivable area from road edges or relaxing lane thresholds.

Smoke promotion is now uncertainty-aware: the 96-scene probe uses Wilson intervals to reject only gross proposal failures, while the 1200-scene strict probe keeps the preregistered point-estimate thresholds unchanged.  Smoke certificate-complete coverage is 95% (strict/train 98%).  PRIO is treated as a typed-source coverage requirement rather than a mathematically unjustified 100% per-protected-agent existential requirement: 95% in smoke and 98% in strict/train, while dataset-wide source support and priority-preservation correctness remain hard-gated.  Auditable rootless and <2-low-burden counts remain zero-tolerance.

Local regression: **215 passed, 5 skipped**; compileall and all v16.8.16 promotion shell syntax checks pass.

## v16.8.12 — Continuous Map Geometry, Neutral-Timing Natural Reference, and Fail-Safe Promotion Verdicts (2026-08-13)

### Triggering evidence from the uploaded v16.8.11 train pilot

The 400-hard + 800-random training pilot completed all 1,200 requested label scenes and passed the manifest and training-supervision audits, but model support remained invalid.  Across 6,611 critical vehicles, 248 (3.751%) had zero natural roots and 354 (5.355%) had fewer than two low-burden natural roots.  Every other active model-support family passed, including full 32-slot relevant-pair response coverage, candidate NCF/false-safe balance, relevance/witness/pair-NCF balance, safe/unsafe and low/high-burden responses, affected-root recovery, protected-candidate feasibility, all natural/response sources, and continuous witness/burden variability.  Re-analysis of the build profile shows that the zero-root agents split into 148 map-dominated and 100 priority-dominated failures; 230/248 used the full logged future as their normative natural reference.

The missing `v16_8_11_train_pilot_verdict.json` was a separate wrapper-control bug: `65_audit_model_support --strict` correctly returned non-zero, but `set -e` terminated the shell before causal/fresh-ceiling diagnostics and the final verdict writer ran.  The pilot therefore failed scientifically, but the wrapper reported that failure as a missing artifact.

### Construction repairs

1. Replaced natural-root map compliance based on distance to discrete lane polyline **sample points** by exact continuous point-to-lane-segment distance.  The physical thresholds (`map_max_distance_*`, compliant fraction and hard maximum) are unchanged.  A trajectory lying on a long/sparse lane segment is no longer declared off-map simply because it is far from the segment endpoints/samples.
2. Raw logged future timing is no longer the normative `natural_ref`, even when all 80 future samples are valid.  Logged future remains the OBS empirical branch and contamination evidence.  NEU/PRIO burden and priority preservation use a timing-neutral reference: lane-topology route geometry first, logged geometry retimed from the current state second, jerk-bounded straight motion only as the final map-fragment fallback.
3. Primary NEU and PRIO roots now use the same lane-topology route bank before straight motion.  Lane-graph search is performed once per critical actor at the maximum required route length and the resulting route polylines are retimed for each acceleration/speed intervention, avoiding repeated graph searches without reducing root/candidate budgets.
4. Route matching against WOMD future geometry now evaluates only the original `valid=1` timestamps.  It no longer treats `sum(valid)` as a contiguous prefix or lets hold-padding choose a route when validity has internal gaps/late appearance.
5. Hard priority progress uses the same natural-reference projection geometry as the burden functional, and the builder records explicit priority rejection reasons (`max_decel`, `max_accel`, `max_jerk`, `progress_loss`, `gap_loss`).  No priority tolerance is relaxed.
6. Natural diagnostics now report rootless/<2-low-burden failures by priority relation, reference kind and dominant rejection family, plus priority rejection mechanisms and map-distance/burden summaries.  This makes a failed smoke directly actionable.

### Promotion-control repair

`smoke`, `strict`, and `train-pilot` now distinguish **semantic gate failure** from pipeline/runtime failure.  Strict supervision/model-support commands may return non-zero as designed, but the wrappers continue far enough to write the natural diagnostics and a composite PASS/FAIL verdict.  `full-core` therefore sees an explicit `recommend_full_rebuild=false` rather than a misleading missing-verdict error.  The master chain also enforces same-fingerprint promotion: smoke must authorize strict, strict must authorize train-pilot, and both strict + train-pilot must authorize full-core.

### Performance disposition

The v16.8.11 train pilot averaged 267.95 s/scene.  Safe responses (143.66 s) and witness construction (71.38 s) together account for about 80.3% of label-engine time; natural construction is 7.38 s/scene and pair-neutral construction 4.07 s/scene.  v16.8.12 therefore preserves the 32-response bank and root/candidate semantics.  The natural-stage optimization is deterministic route-bank reuse, not support reduction.  Full worker selection remains host-specific and must be based on the new pilot profile after semantic gates pass.

### Local validation

- Added continuous sparse-lane segment-compliance regression.
- Added regression proving an 80-step logged future remains OBS evidence but no longer becomes the normative natural timing reference.
- Added priority-rejection mechanism diagnostics regression.
- v16.8.11 support regressions remain passing after the reference-policy update.
- Full repository regression: **194 passed, 5 skipped** (one upstream PyTorch nested-tensor warning only).
- Shell syntax checks pass for all v16.8.12 promotion entry points.

## v16.8.11 — Pair-Specific Natural Basis, Jerk-Compatible Root Support, and Train-Split Promotion Gate (2026-08-11)

### Triggering evidence from the uploaded v16.8.10 smoke/strict artifacts

The fresh 400-hard + 800-random validation strict probe already passes every proposal gate: representative `AnyNCF=0.4675`, false-safe lower bound `0.4825`, PBTR lower bound `0.32715`, and hard-scene NCF recovery `0.3025`.  The wrapper stops only at `model_support`: 1,089 vehicle critical-agent instances have no natural root, 1,090 have no low-burden natural root, and 1,147 have fewer than two low-burden roots.  Response, witness, affected-root, candidate and source-supervision targets are otherwise non-degenerate.  Therefore v16.8.11 does **not** spend more proposal budget to repair a natural-basis support failure.

### Root causes

1. Fresh label construction used one global ego-neutral trajectory for all critical actors.  This violates the interaction-specific neutralization contract: a neutral ego motion that removes pressure from a crossing/merge actor can itself pressure a rear vehicle, and vice versa.
2. Natural NEU/PRIO fallbacks were generated with an instantaneous constant-acceleration primitive but filtered with a comfort-jerk test.  The first discrete acceleration transition could exceed the filter jerk bound, deleting otherwise mild non-zero roots and collapsing AGENT_PRIORITY support to zero/one roots.
3. Short/invalid WOMD futures were padded by hold states before natural construction.  Treating such padding as if it were a complete measured future can create invalid observational/reference geometry and suppress diversity.
4. The previous fallback contract triggered on total-root scarcity, not explicitly on **low-burden** root scarcity.  COWP's OPR/NCF semantics require low-burden natural support, so a scene with several valid but high-burden roots was still untrainable.
5. On curved lanes, straight pseudo-roots can be rejected by the map filter.  Critical selection is intentionally independent of future availability, so the correct repair is a map/topology-based pseudo-root path, not dropping critical agents with incomplete futures.

### Algorithm/data-construction changes

1. Added a fixed, proposal-bank-independent **pair-specific ego-neutral bank**.  For each critical actor, the label engine selects a pressure-removing ego intervention lexicographically by physical safety, actor burden and ego-control deviation.  Candidate proposal families cannot change this natural-basis target.
2. Replaced non-zero natural/neutral straight acceleration steps by a configurable jerk-bounded ramp (`natural_accel_ramp_jerk_mps3`, default `1.0`).  Logged-path retiming uses the same ramp.  No comfort threshold was relaxed.
3. OBS roots are now eligible only when the raw WOMD future validity mask has enough measured support (`>=60/80` and `>=0.70` by default).  Incomplete/invalid future rows are not promoted to measured OBS evidence.
4. Added deterministic lane-graph centreline continuation from the current actor state.  It follows WOMD lane topology and is used as the first natural/reference fallback for curved or short tracks.  Logged future geometry (never logged timing) is a secondary offline fallback; jerk-bounded straight motion is last.
5. Natural construction now has an explicit support contract: the existing active total-root target (`min_natural_alternatives=6`) and at least two roots satisfying the original low-burden threshold.  Fallback generation continues until both are met or all candidates fail the **same** map/dynamics/priority/burden/contamination filters.  A support failure is never hidden.
6. Added per-critical diagnostics to the label-build profile: raw future-valid count/fraction, OBS eligibility, reference source, pair-neutral source/safety/burden, fallback attempts/accepts, and map/burden/priority/contamination rejection reasons.
7. Added exact identity-root reuse from causal audit into response-bank and root-recovery/witness hot paths.  The reused tensors are the same exact `unsafe_between`/`compute_burden` results; no candidate, response, root, threshold or label definition is removed.
8. Added a **train-split 400-hard + 800-random pilot gate** before the 22k/5k full rebuild.  A validation strict PASS alone can no longer authorize a costly full rebuild when the training split may have different natural-support properties.
9. Full preparation now emits the new natural-support diagnostic before model-support gating, so a failed full label build has an immediately attributable root-support reason.

### Speed disposition

The uploaded strict profile averages 239.1 s/scene; safe-response generation (109.1 s) and witness construction (91.6 s) dominate.  Candidate generation (~0.74 s) and natural generation (~1.06 s) are not meaningful full-build bottlenecks.  v16.8.11 therefore preserves candidate/root/response budgets and only removes provably duplicate identity-root physical evaluations.  Worker counts still need host-specific throughput benchmarking after the new smoke because repairing rootless actors can increase legitimate response/witness work.

### Promotion rule

Run `preflight -> smoke -> fastpath-ab -> strict -> train-pilot`.  Full core reconstruction is mechanically blocked unless both validation strict and train-pilot verdicts pass under the identical current code fingerprint.  Do not lower `AnyNCF`, false-safe, PBTR, hard-recovery, natural-support, response-bank or causal-integrity thresholds to obtain a PASS.  Attach Waymax outcomes only after the core label/tensor gates pass.

### Local validation

- Added regressions for pair-specific neutral selection, short-future OBS exclusion + lane-route multi-root support, and exact audit-identity reuse.
- Targeted v16.8.11 tests pass.
- v16.8.10 contract/causal/root-transport regression subset passes.
- Full repository regression: **190 passed, 5 skipped** (one upstream PyTorch nested-tensor warning only).

## v16.8.10 — Trainable-Support Contract Repair and Exact Label-Build Fast Path (2026-08-10)

### Triggering evidence from the v16.8.9 400-hard + 800-random strict probe

The strict wrapper stopped at `model_support` even though the causal-integrity and training-supervision audits passed.  The active failure combined (i) critical agents with no/marginal natural-root support and (ii) a degenerate `cowp/response/valid` target: every causally relevant response slot in the probe was occupied, while the model still optimized a non-zero validity BCE and gated SetTransport by the predicted validity probability.  The same probe also remained below the independent hard-scene NCF recovery gate, so no full rebuild is authorized by this change alone.

### Repairs

1. The publication mainline non-coercion certificate is vehicle-only (`critical.vehicle_only_main=true`), matching the manuscript's stated pedestrian/cyclist extension as future work.  Conventional collision safety still checks all logged non-SDC actors; this switch narrows only the coercion certificate, not physical collision avoidance.
2. `response/valid` is now explicitly treated as a fixed-bank occupancy/padding mask rather than a learned semantic class.  Mainline training sets `response_valid_bce=0`, SetTransport uses all decoded response slots, and the support audit requires full slot coverage on every causally relevant pair instead of artificial positive/negative class balance.  The legacy learned gate remains available through `model.use_response_valid_gate=true`.
3. Added an exact `risk_known_zero` burden fast path.  When `unsafe_between()` has already proved a pair safe, the duplicate TTC/RSS burden computation is skipped because both risk masks are necessarily empty under the same thresholds.  The full burden path now reads the same type-aware TTC and configured RSS parameters as `unsafe_between()`.  The optimization is controlled by `engineering.risk_known_zero_fastpath` (default `true`) so a smoke A/B can disable it and prove semantic equivalence.  It is used in causal-audit, safe-budget, response-bank, witness fallback, and root-recovery hot paths without changing label values.
4. Expanded the strict code fingerprint to include the active model/train contract, preventing a pre-v16.8.10 strict verdict from authorizing a post-change full rebuild.
5. Repaired the RSS geometry contract used by both unsafe labeling and burden risk: TTC/near-miss thresholds are now type-aware, longitudinal RSS is laterally gated to the same/merging corridor, and the broad-phase no longer suppresses long-gap high-speed RSS checks. This removes adjacent-lane false RSS blockers and makes the exact safe-pair burden fast path valid under one shared predicate.
6. Repaired the natural-root budget.  The default v16.8.9 nested OBS loop truncated after eight samples before reaching the identity root; v16.8.10 orders the same configured Cartesian family from `(speed_scale=1, time_shift=0, lateral_offset=0)` outward so finite OBS capacity is not spent entirely on one speed slice.
7. Added a curved-lane ego-neutral route-geometry fallback.  It uses logged future **geometry only** as an offline supervision path and regenerates timing from the current state under bounded constant acceleration.  This avoids copying potentially ego-induced logged yielding timing while preventing straight-line neutral/prio fallbacks from being deleted wholesale by the map filter on curved lanes.  Fallback roots must pass the same map, priority and low-burden plausibility checks as primary roots.
8. Strengthened model-support auditing.  The old audit under-counted rootless critical agents whenever an entire scene had no natural roots because the per-agent loop was incorrectly nested under a scene-level `np.any` guard.  The repaired audit scans every critical agent, requires normalized natural mass, and separately requires at least one and at least two **low-burden** roots because OPR/NCF operate on that low-burden natural set rather than arbitrary valid trajectories.
9. Smoke promotion now treats training-supervision and model-support audits as hard gates, so a 96-scene smoke cannot advance to the expensive 1,200-scene strict probe when the learned-label contract is already invalid.
10. Added an exact train/validation scenario-ID leakage check to full preparation.  Planner training also fails fast when `loss_weights.closed_loop > 0` but attached Waymax outcomes are disabled/missing or the sampled outcome set has no safe/unsafe ranking support; an explicit `closed_loop=0` config is required for the corresponding ablation.
11. Added `NEXT_EXECUTION_V16_8_10_CN.sh` to keep WOMD preflight, smoke, exact fast-path A/B, strict probe, full core rebuild, and Waymax outcome attachment as separately gated stages.
12. Added `ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh` with deterministic resume-safe replay sharding across one or more GPUs, exact per-step safety metrics, incremental attachment, verification, and a full-cache safe/unsafe/mixed-scene support gate.
13. Added a regression proving that WOMD train/val future tensors may remain in the cache for offline targets/Waymax replay while the model-facing history adapter is invariant to them and consumes only past + current state.

### Promotion rule

A new 96-scene smoke and a fresh 400-hard + 800-random strict probe are mandatory.  Do not lower the existing `AnyNCF`, false-safe, PBTR, hard-recovery, or causal-integrity thresholds.  Full rebuild remains prohibited until the newly fingerprinted strict verdict sets `recommend_full_rebuild=true`.  The core full rebuild may intentionally defer Waymax replay to avoid wasting GPU rollout time before all label/tensor gates pass, but the current mainline planner (`closed_loop=0.35`) must train from the later outcome-attached caches.

### v16.8.10 local validation

- Full Python unit/regression suite after the dataset-contract, RSS, execution-tooling, and future-input guard repairs: **187 passed, 5 skipped**.
- Added exact fast-path equivalence, fixed-cardinality response-contract, type-aware TTC/RSS, lateral-RSS/broad-phase, identity-first OBS-budget, and curved-route neutral-proxy regressions.
- No full WOMD smoke/strict/full rebuild was executed in the review environment because the raw WOMD shards are not present there; promotion still depends on the user's fresh 96-scene smoke and 400+800 strict probe.

## v16.8.9 — Candidate-Conditioned Causal Audit and Affected-Root Transport (2026-08-07)

### Triggering evidence from the v16.8.8 96-scene smoke

The stable-critical repair itself succeeded: all 96 smoke scenes used `fixed_anchor_v1`, all proposal-union monotonicity checks passed, `AnyValid=1.0`, and there were no build/filter errors.  The remaining failure was highly structured rather than a generic proposal-quality failure.  On the paired 48-scene representative subset, the fresh bank improved protected-priority burden transfer from `0.5814` to `0.4419` and slightly improved conventional-safe coverage (`0.8958 -> 0.9167`), yet universal `AnyNCF` fell from `0.3750` to `0.2083` and the false-safe lower bound rose from `0.5208` to `0.7083`.  PSY was physically valid (`64/64` accepted in the complete smoke profile) and produced protected priority-NCF pairs, but its scene-level `AnyNCF`, false-safe and PBTR increments were exactly zero.  Removing RMR/PSY/legacy timing changed candidate counts but not the 96-scene scene-level ceiling.

This combination shows that candidate generation was no longer the sole bottleneck.  A candidate could solve one protected pair but remain universally non-NCF because every globally critical agent was still audited, even when that particular ego candidate did not causally perturb the agent's low-burden natural options.  The old data contract also represented only geometric `mode_conflict`; a collision-free natural root that was forced above the adaptive burden budget could reduce OPR/tail feasibility without a corresponding transport-support label.  These cases created the possibility of a non-NCF "silent blocker" with no coherent relevance/affected-root supervision for the model.

### Data/certificate semantic repair

1. Added a candidate-conditioned causal-audit layer with one shared floor-smoothed canonical natural-root probability measure.  For every `(candidate, critical-agent, natural-root)` tuple the cache now records whether the root is geometrically unsafe, its direct burden under the ego candidate, and whether it is **affected** (`unsafe OR direct_burden > beta`).
2. Added `cowp/audit/pair_relevant` and `cowp/audit/relevance_mass`.  A globally critical agent is audited for a candidate only when enough neutral low-burden natural-root mass is causally affected.  The support threshold reuses the existing 0.10 witness-support semantics; it is not a relaxed NCF/burden threshold.
3. Irrelevant global-critical pairs are vacuously non-coercive for that intervention (`OPR=1`, no blocker, no response search).  Relevant pairs retain the original burden/OPR thresholds.  Every relevant non-NCF pair is required to have a witness; schema/integrity gates reject silent blockers.
4. Generalized RootTransport support from `mode_conflict` to `mode_affected`.  Burden-only affected roots now receive same-root recovery, root minimum safe burden, tail-CVaR and OPR supervision exactly like unsafe roots.  Geometric conflict remains a separately reported physical-safety subset.
5. Safe-response generation skips audit-irrelevant pairs.  This is both semantically correct and targets the measured build bottleneck: v16.8.8 spent the overwhelming majority of label time in safe-response and witness search, not in candidate generation.
6. Added candidate-level audited-pair/blocker counts, pair-level NCF/blocker codes, causal relevance mass, affected-root tensors, and complete inline transport fields to the fresh-cache contract.

### Learned-model alignment

1. `WitnessDecoder` now predicts an explicit pair relevance logit.  Witness/OPR/burden losses operate on relevant pairs while relevance itself is supervised on every valid candidate/global-critical pair.
2. `SetTransportCertificateHead` adds a burden-only affected channel and enforces `P(affected) >= P(conflict)`.  Transport uses `retain + affected * q`, not `retain + conflict * q`.
3. Set-Transport loss supervises affected mass, per-root affectedness, affected-root recovery, root burden consistency and uncertainty under the same canonical root measure used by labels.
4. Planner features use relevance as a causal support gate: irrelevant pair witnesses contribute zero and their OPR contribution is neutral (`1`).
5. `paper_aligned_supervision_batch` was updated so fresh v16.8.9 caches cannot be silently interpreted with the old all-critical/conflict-only certificate.  Legacy caches remain readable only under explicit legacy semantics.

### Real learned ablations

Added forward/loss switches and config variants for two paper-critical learned ablations that use the same rich v16.8.9 dataset:

- `w/o candidate-conditioned causal relevance`: all global-critical pairs are again consumed by the learned witness/transport/planner path and the relevance head/loss is disabled.
- `conflict-only RootTransport`: burden-only affected roots are removed from the learned transport support while geometric conflict supervision is retained.

These are independent retraining experiments, not shared-checkpoint aliases.

### Promotion and rebuild policy

- **Do not full-rebuild from v16.8.8.** The next mandatory step is a fresh 48-hard + 48-random v16.8.9 causal-audit smoke because label semantics changed.
- Smoke additionally requires non-degenerate audit relevance, a measurable burden-only affected-root signal, zero silent/irrelevant blockers, zero responses on irrelevant pairs, exact audit/transport affected-root agreement, stable critical selection and proposal-union monotonicity.
- Only a subsequent 400-hard + 800-random strict probe with `AnyNCF>=0.40`, false-safe floor `<=0.55`, PBTR floor `<=0.45`, hard recovery `>=0.20` and all causal-audit integrity checks may set `recommend_full_rebuild=true`.
- Full rebuild uses the old tensor-cache directories only as scenario-ID allowlists.  All candidate/natural/audit/response/witness/transport labels are rebuilt from WOMD Scenario proto and serialized into self-contained NPZ files.
- The complete validation cache is re-gated before any GPU training.  Failing proposal or audit integrity at this stage is an explicit `DO NOT TRAIN` condition.

### Engineering and validation

- Added scripts `57_diagnose_causal_audit.py`, `58_screen_v16_8_9_causal_audit_probe.py`, `59_gate_fresh_v16_8_9_cache_protocol.py`, and `60_verify_fresh_v16_8_9_cache.py`.
- Added v16.8.9 smoke, strict-probe, self-contained full-build, mechanism, Waymax probe/full and experiment-ablation wrappers.
- Training launchers now accept explicit model/train config sources, allowing genuine independently trained ablations with strict provenance.
- Local Python regression after the complete v16.8.9 change set: **175 passed**; `compileall` and all v16.8.9/core launchers pass `bash -n`.

## v16.8.8 — Stable Critical Universe, Priority-Smooth-Yield, and Monotone Proposal Refinement (2026-08-07)

### Triggering evidence from the recovered v16.8.6 micro probe

The recovered probe is complete: 191 unique requested scenes (64 hard + 128 representative random with one overlap), 191 fresh labels written, zero filtered scenes, zero build errors. On the 128 representative random scenes, the legacy v16.8 bank has `AnyNCF=0.3671875` (47/128), while the v16.8.6 fresh bank has `AnyNCF=0.203125` (26/128). The paired transition is 18 retained old-NCF scenes, 29 lost old-NCF scenes, and 8 newly gained NCF scenes. Best-case false-safe floor worsens from 0.578125 (74/128) to 0.765625 (98/128), despite mean valid candidates increasing by 2.414 per scene.

The protected-priority signal moves in the opposite direction. The old bank has 121 priority-eligible scenes and 50 with at least one priority-NCF option, giving PBTR floor 71/121 = 0.58678. The fresh bank has 123 priority-eligible scenes and 70 with at least one priority-NCF option, giving PBTR floor 53/123 = 0.43089, already below the 0.45 strict PBTR ceiling. This is a real positive signal for protected-priority refinement, but it cannot be attributed to PCHR itself: PCHR is present in only 4/128 scenes, contributes 17 candidates, and contributes zero NCF / zero priority-NCF candidates. RMR contributes 223 candidates, 25 NCF candidates, and has excellent timing consistency (max target-TTA error 0.00884 s).

### Root-cause audit: proposal-dependent critical-agent labels

The dominant data-semantics defect is upstream of PCHR geometry. Prior to v16.8.8, `select_critical_agents()` chose the global top-A critical-agent set by screening against the *entire current ego proposal bank*. Adding RMR/PCHR therefore could change which agents entered the top-8 set. Because COWP's candidate NCF is universal over all selected critical agents, adding an optional proposal could relabel every pre-existing candidate and destroy its NCF status even when that candidate trajectory itself was unchanged. This violates the monotone-union interpretation required by proposal-source ablations and confounds old/new ceiling attribution.

The 128-scene probe exhibits exactly this signature: conventional-safe scene coverage actually rises from approximately 121 to 124 scenes and protected-priority NCF availability rises from approximately 50 to 70 scenes, yet universal NCF collapses from 47 to 26 scenes. The fix must stabilize the certificate universe before another proposal family is judged.

### Algorithm/data changes

1. **Fixed critical-agent reference bank (`fixed_anchor_v1`)**. Global critical-agent selection is now independent of optional proposal families. It screens against a deterministic factual-current-state anchor bank: logged ego reference, keep, canonical accelerate, canonical yield, smooth stop, and both canonical lateral directions. `proposal_bank_legacy` is retained only as a compatibility ablation.
2. **Base-bank preservation**. Offline lane-change/creep actions are inserted before optional RMR/priority refinements. Online generation reserves a compact canonical lane-change subset before optional RMR/PSY expansion. Optional refinements may consume filler capacity but cannot silently erase the neutral root or both lateral escape directions.
3. **PCHR is demoted, not expanded**. `priority_hold_release_enabled=false` by default. The implementation remains available for an explicit ablation, but the recovered probe's 17 candidates / zero NCF result does not justify spending more proposal budget on the same stop-hold-release family.
4. **Priority-Smooth-Yield (PSY)**. A new protected-priority proposal source solves a quintic longitudinal arrival to the protected agent's late boundary plus gap while prescribing a low conflict-entry speed and a negative initial acceleration. A commitment check requires a visible early speed drop. This removes PCHR's full-stop feasibility bottleneck while directly targeting early yielding behavior rather than endpoint timing alone.
5. **Offline/online PSY consistency**. The same smooth terminal-speed family is available in online Waymax proposal generation. The offline generator uses map-derived protected relation and boundary-consistent TTA; the online generator uses the existing causal closest-approach proxy and the same smooth-yield controls.
6. **All-scene proposal diagnostics**. Successful label rows now record attempted, accepted, and rejected candidate counts by proposal source, plus critical selection mode/count. Prior profiling exposed rejection diagnostics only for zero-valid scenes and could not explain why a source had low yield.
7. **Source-level monotonicity audit**. `50_ablate_proposal_sources.py` now includes PSY, and the v16.8.8 screen verifies that adding PSY cannot worsen AnyNCF, false-safe floor, or PBTR floor within one fixed-critical fresh bank.
8. **Paired-transition diagnostics**. The paired comparator reports old/new conventional-safe coverage and NCF retained/gained/lost counts, making certificate-universe drift visible rather than hiding it behind aggregate rates.
9. **Code-lineage hardening**. Smoke/strict verdicts carry the full label/eval code fingerprint. Resume of a partial v16.8.8 label directory is rejected under changed code, and a strict PASS cannot authorize a full rebuild under a different fingerprint.
10. **Paper-grade full-cache gates**. Full build remains self-contained (no fragile transport overlay), recomputes the full validation proposal ceiling and proposal-source ablation, requires AnyValid>=0.99, AnyNCF>=0.40, false-safe floor<=0.55, PBTR floor<=0.45, nonzero PSY protected-priority NCF yield, and proposal-union monotonicity before any model training.

### Build-cost evidence

The completed 191-scene v16.8.6 profile shows candidate generation is not the multi-day bottleneck: candidate generation averages ~0.86 s/scene, while safe-response generation averages ~164.5 s/scene (~57% of label-engine time), witness/certificate construction ~99.7 s/scene (~35%), and critical selection ~21.3 s/scene (~7%). Consequently v16.8.8 does not reduce root/search budgets or weaken labels merely for speed. The decision protocol first runs a 48-hard + 48-random smoke under the corrected semantics, then a 400-hard + 800-random strict probe only after smoke success. A full build is mechanically blocked until the strict verdict authorizes it under the identical code fingerprint.

### Current rebuild decision

**DO NOT FULL-REBUILD v16.8.6/v16.8.7.** The recovered PCHR micro screen is a valid rejection of that data semantics: PCHR itself has no direct NCF yield, and the universal NCF collapse is confounded by proposal-dependent critical-agent selection. Run the v16.8.8 96-scene stable-critical + PSY smoke first. A smoke PASS authorizes only the strict 1,200-scene probe. Only a strict PASS (`recommend_full_rebuild=true`) under the same code fingerprint authorizes the multi-day full rebuild.

### Local validation

- `pytest -q`: **171 passed** (one upstream PyTorch nested-tensor warning only).
- `python -m compileall -q cowp`: pass.
- v16.8.8 smoke / strict / full-build shell entrypoints: `bash -n` pass.
- regressions cover proposal-bank-independent critical selection, PSY endpoint/terminal-speed behavior, and explicit PCHR demotion semantics.


## v16.8.7 — Cache-Lineage Hardening, Self-Contained Fresh Transport, and Rebuild Decision Recovery (2026-08-08)

### Triggering evidence

The v16.8.6 64-hard + 128-random Priority-Commitment micro probe completed the expensive fresh-label stage for **191 unique scenes** (one scene appears in both requested sets). `fresh_probe_profile.jsonl` contains 191 `written` rows and no filtered/error rows. The run then failed only when reading the legacy old-bank reference cache: `tensor_cache_val_waymax_transport_v16_8/10040e572b831a04.npz` could not be opened.

The root cause is cache lineage, not NumPy indexing. Legacy `*_waymax_transport_v16_8` directories are transport **overlay** caches: their visible top-level `*.npz` entries are symlinks to the corresponding `tensor_cache_*_waymax` files, while transport tensors live in a hidden sidecar directory. Deleting `tensor_cache_train_waymax` / `tensor_cache_val_waymax` therefore leaves visible but broken symlinks. The proposal-ceiling and paired-probe scripts do not consume `waymax/*`, so the surviving `tensor_cache_train` / `tensor_cache_val` are the correct legacy raw backing for these diagnostics. The original Waymax attach implementation copies every base tensor unchanged and only appends `waymax/*`; transport augmentation likewise does not consume `waymax/*`.

### Algorithm status

There is **no new paper-level proposal/certificate hypothesis in v16.8.7**. The v16.8.6 BCS-RMR + Priority-Commitment Hold-Release algorithm is retained unchanged. v16.8.7 is a data-lineage, reproducibility, and build-cost revision. This separation is intentional so the existing 191-scene PCHR probe labels remain semantically usable for the pending comparison.

### Engineering/data changes

1. `COWPNpzDataset` now detects broken overlay symlinks at construction and raises an actionable lineage error instead of crashing later on an arbitrary `np.load`.
2. Proposal-probe wrappers now use `tensor_cache_val` as the default old-bank source because proposal ceiling/PBTR metrics do not require transport or cached Waymax outcomes. The paired comparator loads only requested scenario IDs, avoiding thousands of unnecessary NPZ opens.
3. Added `RECOVER_V16_8_6_PRIORITY_MICRO_PROBE_FROM_BASE_CN.sh`, which reuses the already-built 191 fresh labels and reruns only old-bank diagnosis, paired comparison, and the micro screen. It never rebuilds labels.
4. Added `54_rebase_transport_overlay.py` and `REPAIR_LEGACY_V16_8_TRANSPORT_OVERLAYS_CN.sh`. They can salvage existing v16.8 transport sidecars by copying them into a new overlay whose top-level links point to surviving `tensor_cache_train/val`. This is valid for legacy RCOT/BCOT diagnostics because Waymax attach only appended `waymax/*` fields and transport construction does not consume them. The rebased caches intentionally contain no cached Waymax outcomes and remain non-paper-grade v16.8 proposal data.
5. Fresh label files now serialize the complete transport contract inline, including `cowp/transport/root_min_safe_burden` and `cowp/transport/canonical_root_weight`. Canonical root weights are precomputed once per agent rather than once per candidate-agent pair; this is algebraically equivalent.
6. Fresh full rebuilds no longer run post-hoc transport augmentation. `tensor_cache_train/val` are self-contained regular NPZ caches and are used directly as TRAIN_CACHE/VAL_CACHE. This removes a duplicated transport search pass and eliminates overlay-backing fragility.
7. Fresh-cache fingerprints now cover critical-agent selection, natural alternatives, safe responses, witness, burden, and priority code in addition to proposal/geometry code. A change to label semantics can no longer evade the build fingerprint.
8. Added `55_verify_fresh_self_contained_cache.py`: exact scene-set matching, no symlinks, required proposal/witness/transport keys, Waymax-ready state presence, scenario-id consistency, validity, NCF, and proposal-source statistics are checked before training.
9. Full rebuild defaults reuse the surviving `tensor_cache_train/val` **only as scenario-ID allowlists**. No stale COWP label is reused. Fresh BCS-RMR/PCHR labels still come from WOMD Scenario protos.
10. After a full fresh merge, the validation proposal ceiling is recomputed from the actual tensor cache and training is blocked unless AnyValid >= 0.99, AnyNCF >= 0.40, false-safe floor <= 0.55, and PBTR floor <= 0.45.

### Build-performance evidence and policy

The 191-scene micro profile shows that candidate generation itself is cheap. Relative to aggregate `label_engine_s`, `engine_safe_responses_s` accounts for about **57%**, `engine_witness_s` about **35%**, and critical-agent selection about **7%**; together safe response and witness computation dominate more than 90% of fresh label time. Therefore the safe accelerations in this release are: sparse allowlist producer filtering, thread-pool suppression, exact scene-set reuse, no full-train cached Waymax replay, no duplicate transport augmentation, and resumable fingerprints. Do not reduce response/root search budgets merely to make the build faster without a controlled label-equivalence experiment.

### Current rebuild decision

**HOLD.** Do not start a full rebuild until the already-built 191-scene probe is recovered using `tensor_cache_val` and the v16.8.6 micro screen is evaluated. If the micro screen passes, run the strict 400-hard + 800-random probe. Only the strict probe may authorize the expensive full fresh rebuild. If it fails, improve proposal refinement rather than spending days rebuilding an infeasible bank.

### Local validation

- `pytest -q`: **167 passed**.
- `python -m compileall -q cowp`: pass.
- all modified/new v16.8.6/v16.8.7 shell entrypoints: `bash -n` pass.
- rebase/self-contained-cache/probe/gate CLIs: import/`--help` pass.
- regressions cover early broken-symlink detection, transport-sidecar rebasing, and complete inline fresh transport tensors.

## v16.8.6 — Priority-Commitment Proposal Refinement, Legacy-Bank Saturation Audit, and Evidence-Valid Experiment Pipeline (2026-08-07)

### Triggering evidence

This revision was triggered by three independent findings from the uploaded v16.8.5 run and prior v16.8.3 evidence.

1. The uploaded `formal_v16_8_4_bcs_rmr_bcte_proposal_probe.zip` is **not a completed fresh proposal probe archive**. It contains the old-bank `current_proposal_ceiling.json` and the interrupted fresh-label log that stopped after 768 written labels with `No valid ego candidates available for neutral intervention`; it does not contain a completed `paired_proposal_probe.json`. Therefore this archive cannot establish either success or failure of fresh BCS-RMR.
2. The legacy v16.8 validation bank is structurally healthy but proposal-limited: across 5,013 scenes, `AnyNCFSceneRate=0.36246`, best-case false-safe floor `0.53321`, and best-case PBTR floor `0.58179`. The PBTR floor exceeds the current calibration ceiling `0.45`, so no selector restricted to this fixed bank can satisfy the current protected-burden gate.
3. On the same legacy bank, the completed v16.8.3 learned run already showed strong mechanism discrimination: protected NCF recall `0.97894`, pair-witness AUPRC `0.84945`, protected BCOT AUPRC `0.96984`, and protected RootTransport AUPRC `0.88167`. The remaining calibration status was `proposal_infeasible`. This is evidence that further threshold/budget tuning on the same proposal bank is not the highest-leverage direction.

The v16.8.5 label-space ablations also reveal a useful asymmetry. Removing counterfactual reasoning increases FSR from `0.31156` to `0.58896`; removing the neutral branch increases it to `0.39760`; removing the priority branch increases it to `0.34879`. In contrast, removing option preservation leaves FSR unchanged at the reported precision and removing hard witness rejection changes FSR by only about `2.6e-4`. These are oracle/label-space mechanism diagnostics, not learned-model claims, but they justify keeping the causal roots and protected-priority structure while de-emphasizing peripheral hard gates.

### Algorithm change: Priority-Commitment Hold-Release (PCHR / proposal source `PRIORITY_HOLD_RELEASE`)

BCS-RMR fixes *when* ego reaches the conflict boundary, but a protected agent can still be coerced by the *approach profile*: ego may remain assertive until late braking even when final arrival order is pass-after. v16.8.6 therefore adds a proposal family that makes yielding observable early.

For causally protected (`AGENT_PRIORITY` or `EQUAL_OR_NEGOTIATED`) interactions, the candidate generator now:

1. chooses a stop point several metres before the conflict boundary;
2. follows a jerk-smooth quintic segment from the current state to rest at that point;
3. holds before the conflict for a minimum dwell time;
4. releases with a second zero-endpoint-acceleration quintic so boundary entry occurs after the protected agent's late arrival envelope plus a gap;
5. sends the resulting trajectory through the same map, acceleration, jerk, validity, and NCF label machinery as every other candidate.

This is a targeted **proposal refinement** for protected burden transfer. It does not change the NCF definition, burden normalization, BCOT target, RootTransport target, or calibration thresholds in this release.

### Offline/online consistency and candidate-budget repair

- The same hold-release primitive is available in online Waymax candidate generation. Offline-only proposal families are prohibited because the trained mechanism would otherwise see actions that disappear in closed loop.
- A core neutral proposal is reserved early, before optional timing/lane/terminal families can saturate the candidate budget. This prevents proposal expansion from deleting the intervention root needed by the certificate.
- Fresh source provenance includes `PRIORITY_HOLD_RELEASE`, enabling direct source-level NCF/PBTR ablation.

### Data decision protocol

A full rebuild is **not authorized immediately**. The current decision is two-stage:

1. **192-scene micro screen**: 64 old hard scenes + 128 unbiased validation scenes. It must show PHR generation, priority-NCF yield, material PBTR-floor improvement, no major NCF destruction, acceptable validity, and pairing completeness.
2. **Strict 1,200-scene paired proposal probe**: 400 hard + 800 random. Only `promote_to_full_rebuild=true` authorizes a fresh full v16.8.6 rebuild.

If the micro screen fails, do not rebuild. Improve proposal refinement first. If the strict probe passes, a fresh rebuild becomes necessary for paper-grade v16.8.6 training because the legacy overlay cannot retrofit the new ego proposal tensors or their NCF labels.

### Experiment-pipeline corrections

1. `05_make_tables.py` no longer materializes all large natural/response trajectories. It loads only table-required arrays, compacts candidate trajectories to first/last samples for exact label-space progress, uses threaded NPZ loading, and writes a persistent compact cache. Module-effect tables reuse already-computed planner decisions instead of running each label-space planner twice.
2. Proposal-source ablation now fails on stale caches without `proposal_source`; the wrapper skips it explicitly instead of fabricating an RMR contribution from all-PAD provenance.
3. GameFormer/DTPP adapters now transform agents, futures, candidates, and map points into the ego frame while preserving roadgraph padding validity. Loss-side numerics are FP32; AMP `auto` prefers BF16. Training hard-fails if skipped batches exceed 2%, preventing a checkpoint trained on a tiny surviving subset from being treated as a strong baseline.
4. Fresh v16.8.6 cache protocol requires `data_manifest_v16_8_6.json`, a matching build fingerprint, current proposal provenance, and `.transport_v16_8_6` sidecars. Legacy-bank diagnostics are isolated behind an explicit non-paper-grade wrapper.
5. GameFormer/DTPP are candidate-bank scorers in this repository. A legacy-bank rerun is therefore only a numerical smoke; final matched comparisons must retrain/evaluate them on the same fresh proposal bank as COWP to avoid a train/test proposal mismatch.

### Directions explicitly not repeated

Do **not** spend the next iteration on:

- simply increasing BCOT budget;
- threshold-only repair on the fixed legacy candidate bank;
- flat candidate certificates in place of root-conditioned transport;
- making all critical pairs a hard veto;
- treating stop/yield as automatically non-coercive;
- retraining a natural decoder that has already passed its foundation gate without a new causal hypothesis;
- using sparse cached candidate Waymax outcomes as the final closed-loop claim.

### Promotion interpretation

- **Keep/deepen:** counterfactual natural roots, neutral intervention, protected priority relation, root-conditioned transport/BCOT, explicit fallback, certificate/shortlist separation, and certificate-guided proposal refinement.
- **Demote as primary contribution:** current hard option-preservation and hard witness-rejection decision gates, because the uploaded oracle ablations show near-zero marginal decision effect on the legacy bank. They may remain as safeguards/diagnostics until learned and closed-loop ablations are available.
- **Next theoretical step after proposal sufficiency:** shift-aware calibration under ego-policy-induced interaction distribution shift; ordinary IID calibration alone is not a sufficient final safety argument for reactive closed loop.

### Local validation

- `pytest -q`: **164 passed** (one existing PyTorch nested-tensor prototype warning).
- `python -m compileall -q cowp`: pass.
- All v16.8.5/v16.8.6 execution entrypoints modified or added in this release: `bash -n` pass.
- `52_screen_priority_commitment_probe`, `53_gate_fresh_v16_8_6_cache_protocol`, `05_make_tables`, and `20_train_external_baseline`: import/`--help` smoke pass.
- Legacy v15--v16.5 shell files with historical CRLF/syntax defects were intentionally not mass-edited; they are outside the v16.8.6 execution path and are documented rather than hidden by unrelated diffs.

### Prohibited claims before new data is run

- Do not claim PCHR improves WOMD/Waymax until the micro/strict probes and fresh closed-loop experiments run.
- Do not call the uploaded interrupted v16.8.4 archive a completed fresh-probe verdict.
- Do not report the uploaded GameFormer/DTPP learned results as strong baselines: their old training logs skipped more than 99% of batches.
- Do not report a COWP v16.8.6 Waymax result from the legacy v16.8 raw candidate bank as a paper-grade evaluation.

# COWP Algorithm Change Log

This is the canonical record of algorithm attempts. Do not repeat a rejected
change without new evidence. Every experiment must record the code version, data
version, seed, checkpoint lineage, learned-offline gate, online paired metrics,
and the exact simulator-agent setting.

## v16.8.5 — Probe-Resilient Data Decision, PBTR-Sufficient Promotion, and Fast Rebuild Protocol

### Triggering evidence

The v16.8.4 400-hard + 800-random proposal probe aborted after about 5.5 hours with
768 fresh labels written. The worker exception was `No valid ego candidates available
for neutral intervention`. This is a scene-local proposal-bank outcome, but the old
builder propagated it through `Future.result()` and killed the whole multiprocessing
job. The same run also exposed two cost bugs: scenario allowlisting happened inside
workers after raw protobufs had already crossed process IPC, and `--limit` counted only
newly written files instead of valid `--skip-existing` outputs.

The uploaded current-cache audit shows the existing v16.8 transport overlays are
structurally healthy for core RCOT/BCOT training (20,440 train / 5,013 val, required
transport/core key coverage complete, response-root assignment valid). They are not a
fresh BCS-RMR label dataset. On the full old validation cache the proposal ceiling is
`AnyNCF=0.36246`, selected-false-safe floor `0.53321`, and best-case PBTR floor
`0.58179`. Because the v16.8 calibration contract uses `max_priority_burden_transfer=0.45`,
the old proposal bank remains mathematically infeasible for the current mechanism gate
even though its global false-safe floor is below 0.55.

### Engineering/data-protocol changes (no certificate-definition change)

1. Added `NoValidEgoCandidatesError` with scenario id and proposal-rejection diagnostics.
   Zero-valid scenes are now returned as `status=filtered, filter_reason=no_valid_ego_candidates`
   instead of aborting the full build. Candidate acceptance semantics are unchanged.
2. Added per-scene proposal rejection diagnostics (`map_filter`, accel/decel, jerk,
   nonfinite, duplicate, capacity) so the next failed scene identifies the physical reason.
3. Sparse allowlists are filtered in the producer before multiprocessing IPC. Workers now
   receive only requested Scenario records; the scan stops when all requested IDs have been
   observed.
4. Fixed resumable `--limit`: complete existing files count toward the output target.
   Build profiles are appended when `--skip-existing` is used so filtered/no-valid terminal
   rows survive a resume. Worker errors now preserve the expected scenario id.
5. Proposal-probe default changed to `FORCE_REBUILD_PROBE=0`; the 768 already generated
   fresh labels are reused. The probe records a stage-level profile JSONL.
6. Promotion is now **proposal-sufficient for both mechanism constraints**: in addition to
   `AnyNCF>=0.40`, false-safe floor `<=0.55`, hard-scene recovery `>=0.20`, and RMR TTA
   error `<=0.20 s`, it requires `AnyValid>=0.99` and best-case PBTR floor `<=0.45`.
   Filtered requested scenes are conservatively counted as zero-valid in the paired probe;
   unexpected missing/error scenes remain fatal.
7. `45_diagnose_proposal_ceiling` gained exact index-modulo partition diagnostics matching
   learned-offline calibration/held-out splitting. This resolves the observed discrepancy
   between the old full-cache floor and the earlier calibration-partition floor before any
   expensive decision.
8. Added `PREPARE_COWP_V16_8_5_DATA_FAST_CN.sh`: reuses existing indexes, profiles label
   stages, supports true resume, skips visualization diagnostics during the critical path,
   and defaults `RUN_WAYMAX_REPLAY=0`. Full train-set Waymax replay is not required for
   natural/response/witness/RCOT/planner core supervision or real online Waymax evaluation.
9. Planner training gained `USE_WAYMAX_OUTCOME_LABELS=0`, allowing fresh core-only caches;
   the existing loss code already returns exactly zero outcome loss when those optional
   tensors are absent. This removes the need to spend large replay cost before the main
   mechanism experiment.
10. Resumed builds with existing NPZ files now bypass the worker pool in the producer
    once the scenario id and file integrity are verified. This preserves semantics while
    preventing hundreds/thousands of already-complete scenes from occupying expensive
    label-engine worker slots after an interruption.
11. Added `49_summarize_label_build_profile.py`, which reports p50/p90/p99 total and
    per-stage times plus zero-valid rejection causes. Worker-count or algorithmic speed
    changes should be based on this profile instead of guessed.
12. Added `50_ablate_proposal_sources.py` for post-build proposal-source ablations
    (`all`, `without_rmr_bcte`, `without_legacy_timing`, `without_any_timing`) without
    retraining or Waymax. This isolates the marginal proposal-ceiling contribution of
    BCS-RMR-BCTE.
13. Fixed an evaluation-validity bug: historical `cowp_wo_*` architecture/causal-branch
    names were silently treated as ordinary COWP in learned-offline/online shared-forward
    evaluation. Those names now fail loudly. Valid shared-forward selection baselines are
    evaluated in one model/cache pass; causal-branch and graph-architecture ablations must
    use label-space diagnostics or separately retrained checkpoints.
14. `05_make_tables.py` no longer rereads every NPZ once per method before recomputing the
    same decisions; labels are loaded once and reused in memory.
15. Probe resume now preserves the uploaded v16.8.4 build fingerprint and accepts only
    that exact known interrupted lineage (or the current fingerprint). A separate semantic-resume
    manifest records why the 768 successful NPZ labels are compatible: v16.8.5 changed only
    zero-valid handling/diagnostics for those paths, not serialized tensors for valid scenes.
16. Fast full rebuild can reuse the audited old cache **scenario-id set** and WOMD indexes while
    rebuilding every COWP label from fresh Scenario protos. This gives an exact paired dataset,
    avoids spending label-engine time on scenes that never entered the old training cache, and
    caps per-process BLAS/TensorFlow threads to prevent multiprocessing oversubscription.
17. Added staged experiment wrappers for multi-seed mechanism runs and fresh-cache external
    baselines. Expensive replication is intentionally delayed until the single-seed mechanism and
    small Waymax probe pass; repository GameFormer/DTPP implementations are treated as matched
    baselines, not mislabeled as official reproductions.

### Data decision after this round

- **Do not start the four-day full rebuild yet.** Resume and finish the corrected paired
  fresh proposal probe first.
- The old cache may be reused for engineering/core-module controls, but it cannot validate
  the v16.8.4 BCS-RMR proposal claim and cannot satisfy the current PBTR proposal floor.
- Full fresh rebuild is justified only if the paired fresh bank passes all six proposal
  checks, especially `BestCasePBTRLowerBound<=0.45`. If it fails, change/refine proposal
  generation rather than paying for a larger copy of an infeasible bank.

### Validation

- `pytest`: 160 passed.
- `python -m compileall -q cowp`: pass.
- Updated/added v16.8.5 shell entrypoints plus `run_cowp_v16_8_dual_gpu.sh`: `bash -n` pass.
- Note: several historical pre-v16.8 root scripts in the uploaded archive retain legacy CRLF/syntax issues; they are outside this v16.8.5 execution path and were not silently rewritten.

## v16.8.4 — Boundary-Consistent Smooth RMR-BCTE, Fresh-Cache Hard Gate, and Online Timing Alignment

### Triggering evidence from the uploaded `cowp_v16_8_3_rmr_bcte_seed2026` run

The directory name says v16.8.3, but the run provenance shows that training and
learned-offline evaluation actually used the old v16.8 overlay protocol:

- `DATA_PROTOCOL=v16_8_root_conditioned_overlay`;
- raw caches are `formal/tensor_cache_{train,val}_waymax`;
- transport caches are `formal/tensor_cache_{train,val}_waymax_transport_v16_8`.

Those caches are structurally healthy for the v16.8 RCOT/transport experiment
(20,440 train / 5,013 val files, sampled required-key coverage 100%, transport
augmentation complete, response-root out-of-range rate 0), but a transport
sidecar cannot replace the ego candidate tensors stored in the raw cache.
Therefore this run did **not** evaluate the v16.8.3 RMR-BCTE proposal repair.

The current mechanism report still provides strong diagnostic evidence:

- pair-witness AUPRC `0.84945` (pass);
- protected BCOT false-safe AUPRC `0.96984` (pass);
- protected RootTransport AUPRC `0.88167` (pass);
- protected NCF recall `0.97894` and global recall `0.98387` (pass);
- protected NCF precision `0.49374` (slightly below the `0.50` gate);
- accepted-candidate rate `0.32079` (pass);
- fallback `0.41341` (fail, target `<=0.25`);
- PBTR improvement `0.01216` (fail, target `>=0.03`);
- selected false-safe improvement `0.01038` (fail, target `>=0.03`).

Calibration correctly reports `status=proposal_infeasible`: the fixed-bank
best-case selected-false-safe lower bound is `0.60989` and the best-case PBTR
lower bound is `0.64457`.  Thus threshold/budget tuning on this bank cannot pass
the mechanism gate.

### Newly confirmed proposal-semantics defect

v16.8.3 fixed the jerk filter and broadened timing proposals, but its RMR-BCTE
arrival solver used Euclidean distance to the **conflict-region centre**.  The
interaction/collision geometry declares entry when the vehicle envelope first
crosses the **inflated conflict-region boundary**.  Consequently a candidate
could be tagged `pass-after` even though it entered the actual conflict region
much earlier.

A deterministic toy regression reproduces the mismatch in the uploaded code:
a `pass-after` proposal advertised target TTA `5.60 s` but continuously entered the conflict
envelope at about `3.47 s` (error about `2.13 s`).  This is not a threshold problem: the
proposal itself violates its timing semantics and can directly reduce NCF yield.

### Algorithm changes

1. **Boundary-consistent reachability and distance.** Added continuous
   `trajectory_entry_to_region`, which returns first entry time and travelled arc
   length to the same inflated region boundary used by the interaction geometry.
   Conflict regions that the nominal ego primitive never reaches no longer
   trigger STOP/legacy/RMR timing proposals.
2. **Boundary-Consistent Smooth RMR-BCTE (BCS-RMR-BCTE).** Timing proposals now
   solve to the conflict-envelope entry distance, not the region centre.
   The longitudinal primitive is a cubic profile
   `s(t)=v0*t+c2*t^2+c3*t^3` with `s(T)=d` and `a(T)=0`, giving finite constant
   jerk before arrival and allowing delayed arrival without the
   constant-deceleration stop-before-arrival pathology.
3. **Causal agent boundary envelopes.** Agent early/nominal/late TTA uses only the
   current state and current velocity ray intersected with the inflated conflict
   boundary.  No logged future is introduced into proposal generation.
4. **Realized timing contract.** Every BCS-RMR proposal is re-measured against the
   region boundary and rejected if `|TTA_realized-TTA_target|>0.20 s`.  New cache
   provenance stores `proposal_entry_distance_m` and
   `proposal_target_tta_error_s`.
5. **Diversity-preserving timing deduplication.** RMR profiles are deduplicated by
   `(region, timing_side, acceleration_bin)` rather than a global acceleration
   bin, so one region/side cannot erase a distinct interaction hypothesis.
6. **Online timing alignment.** Waymax candidate generation now uses the same
   smooth-arrival family for robust timing hypotheses and includes the omitted
   current-state-to-first-sample arc length.  Online profile deduplication is by
   `(agent, timing_side, acceleration_bin)`.

The same toy regression after v16.8.4 produces pass-after proposals with target
TTAs `4.48376 s` and `5.08376 s`, realized errors only `3.3e-6 s` and `4.3e-6 s`.
This proves semantic consistency of the primitive; it does **not** yet prove
higher WOMD NCF coverage or SOTA closed-loop performance.

### Engineering/protocol changes

1. **Strict fresh-cache preflight.** New
   `47_gate_fresh_v16_8_4_cache_protocol.py` refuses to train/probe/full-evaluate
   when the requested cache lacks the v16.8.4 manifest, matching code fingerprint,
   new proposal provenance tensors, or the `.transport_v16_8_4` sidecar contract.
   This prevents an exported old `tensor_cache_*_transport_v16_8` path from
   silently being called a v16.8.4 experiment.
2. **Expanded build fingerprint.** The data/probe fingerprint now includes
   `lane_graph.py` and `trajectory_primitives.py` in addition to candidate,
   label/schema and config code.  v16.8.3 omitted these two files, so a geometry
   or primitive change could otherwise be resumed into a stale build root.
3. **Fresh protocol version.** Causal audit accepts `v16_8_4_fresh`; v16.8.4
   mechanism/probe/full wrappers explicitly export all fresh cache paths instead
   of inheriting the old v16.8 defaults.
4. **Versioned proposal diagnostics.** New evaluation rows use
   `v16_8_4_boundary_consistent_proposal_floor`; calibration remains backward
   compatible with v16.8.3 rows.
5. **Proposal probe gains a timing gate.** Promotion now additionally requires at
   least one measured RMR timing proposal and maximum target-TTA error `<=0.20 s`.
6. **Paired Waymax promotion gate.** The old FULL wrapper checked only that a
   probe-delta file existed.  v16.8.4 now reads the raw conventional/COWP probe
   outputs and blocks full rollout on insufficient paired rollouts, safety/progress
   regression beyond probe tolerance, excessive episode fallback, or obvious
   degradation in available predicted false-safe/burden/OPR proxies.  This is a
   cost-control continuation gate, not a causal/publication claim.
7. **Manuscript method synchronization.** A new
   `interactive_planning_v16_8_4_revised.tex` updates only the proposal-generation
   method definition to boundary-entry timing, smooth cubic arrival and realized
   TTA validation.  No unrun WOMD/Waymax number or SOTA claim is inserted.

### Data decision

Do **not** immediately rebuild all 22k/5k scenes and do not retrain transport or
planner on the current v16.8 overlay.  First run the 400-hard + 800-random paired
**v16.8.4 label-only probe**.  Promote to the expensive rebuild only when:

- `AnyNCFSceneRate >= 0.40`;
- best-case selected false-safe lower bound `<= 0.55`;
- hard-scene NCF recovery `>= 0.20`;
- BCS-RMR realized target-TTA max error `<=0.20 s`.

If coverage/floor still fail after timing consistency is fixed, the next
algorithmic step is **proposal-space refinement**, not looser certification:
route-topology/Frenet interaction proposals or a strong learned/generative
proposal backbone should be added underneath the unchanged non-coercion
certificate.  That preserves the paper's novelty as a certificate/refinement
layer while removing the fixed kinematic-bank ceiling.

### Algorithm disposition after this round

- **Retain:** protected non-coercive hard feasibility, natural roots,
  `s=(1-c)r+cq`, independent recovery/burden heads, RCOT/BCOT aggregation,
  explicit uncertified fallback, certificate/shortlist separation.
- **Deepen next (after proposal probe):** certificate-guided proposal refinement;
  distributional RCOT; simultaneous one-sided protected-pair calibration; and
  shift-aware risk control for ego-policy-induced interaction distribution shift.
- **Do not revive:** flat candidate certificate as the primary mechanism,
  all-critical hard veto, threshold-only rescue, automatic stop/yield safety, or
  sparse cached Waymax replay as the final closed-loop claim.
- **Secondary theory audit, not changed in v16.8.4:** the primitive burden weight
  vector still allocates weight to the option component while witness construction
  zeroes that component and handles option preservation separately through OPR.
  Changing its normalization now would alter the NCF definition and confound the
  proposal-coverage experiment; audit/calibrate it only after the proposal gate.

### Next decision rule

1. Run `NEXT_RUN_COMMANDS_V16_8_4_PROPOSAL_PROBE_CN.sh` against the existing old
   validation cache for paired comparison.
2. If `promote_to_full_rebuild=false`, stop before full rebuild/training and use
   source-level RMR yield + hard-scene recovery to design the next proposal
   refinement.
3. If true, build a **new** `formal_v16_8_4_bcs_rmr_bcte` data root with
   `PREPARE_COWP_V16_8_4_DATA_CN.sh` and train only transport/planner from the
   already validated natural checkpoint.
4. Waymax probe/full remains a hard continuation gate: both
   `mechanism_verification.pass=true` and `calibration_feasible=true` are required.
5. After the small Waymax probe, `NEXT_RUN_COMMANDS_V16_8_4_FULL_CN.sh`
   additionally requires `promotion_gate_v16_8_4.json: pass=true`; a probe file
   merely existing is no longer sufficient to spend the full rollout budget.

## v16.8.3 — Proposal-Sufficiency Audit, Jerk-Consistent Candidate Validation, and Robust Multi-Region BCTE

### Triggering evidence

The corrected v16.8.2 re-evaluation of the v16.8.1 checkpoint completed the
natural, model-anchor, cache-reuse, causal-overlay, calibration and held-out
learned-offline chain.  The evaluation semantics are current and the
calibration/held-out partitions are disjoint, but the main mechanism gate remains
infeasible:

- `mechanism_verification.pass=false`;
- `calibration_feasible=false`, `status=least_violation`;
- protected NCF recall `0.97064` (pass), protected NCF precision `0.47628`
  (fail), accepted-candidate rate `0.32956` (pass), fallback `0.38667` (fail);
- held-out PBTR improves only `0.01261` over conventional and selected
  false-safe rate improves only `0.01237`.

The corrected sweep rules out a threshold-only repair.  At budget `0.70`,
protected precision passes (`0.50987`) but fallback remains `0.42401`; at budget
`0.90`, fallback passes (`0.21659`) but precision falls to `0.35071`.  PBTR and
selected false-safe rate change by less than one percentage point over the useful
budget range.

### Proposal-sufficiency finding

The held-out bank contains at least one NCF candidate in only `0.27255` of scenes,
while `0.89146` of scenes contain a conventionally safe candidate.  Therefore

`P(any conventional-safe and no NCF) >= 0.89146 - 0.27255 = 0.61891`.

For any selector restricted to this fixed bank and required to return a
conventionally safe candidate when one exists, `0.61891` is a scene-level
selected-false-safe lower bound.  It already exceeds the gate target `0.55`.
The current selected false-safe rate `0.63966` is only `0.02075` above this bank
floor, so selector/certificate tuning has little remaining headroom without new
proposals.  The certificate itself retains about `97.5%` of scenes where an NCF
proposal exists; the dominant ceiling is upstream.

### Confirmed engineering defect

Offline `_candidate_valid` exposed `ignore_initial_jerk_steps=3` and
`jerk_check_percentile=99`, but ignored both and rejected candidates using the
absolute maximum finite-difference jerk.  Online validation already used the
configured prefix removal and percentile.  The mismatch systematically removed
constant-acceleration timing candidates because the first discrete derivative
contains the transition from the logged current acceleration to the primitive
acceleration.  This defect especially affects BCTE, accelerate, yield and stop
families and invalidates any decision to pay for a full BCTE rebuild before a
small corrected probe.

### Algorithm and implementation changes

1. **Jerk-consistent offline validation.** Offline candidates now use the same
   ignored prefix and robust percentile as online validation.
2. **Physical arrival constraint.** A constant-deceleration timing solution is
   rejected when its requested conflict arrival occurs after terminal speed
   becomes negative; zero-speed clipping can no longer masquerade as a later
   arrival.
3. **Robust Multi-Region BCTE (RMR-BCTE).** The proposal generator ranks
   forward-reachable conflict regions, retains up to three, constructs bounded
   early/nominal/late arrival envelopes for up to four approaching agents per
   region, and proposes pass-before against the early boundary and pass-after
   against the late boundary.  A 24-candidate quota and acceleration-space
   deduplication prevent timing hypotheses from exhausting the full bank.
4. **Offline/online semantic alignment.** Online conflict timing uses the same
   before/after envelope convention, physical arrival check, acceleration bounds,
   quota and deduplication intent.
5. **Proposal provenance.** Each candidate records source, conflict region,
   target time, timing side, target agent, gap and solved acceleration.  These
   fields are persisted in labels/cache and are optional for old cache readers.
6. **Proposal-floor metrics.** Offline/online aggregation now reports the
   conventional-without-NCF lower bound, protected proposal floor, NCF selection
   recall given availability, and selector excess above the bank floor.
7. **Infeasibility-aware calibration.** If no operating point is feasible and
   proposal floors already violate hard constraints, calibration reports
   `status=proposal_infeasible` rather than implying that another threshold is
   likely to solve the problem.
8. **Paired label-only proposal probe.** A new diagnostic samples 400 old-bank
   hard scenes and 800 unbiased random validation scenes, rebuilds only their
   labels, and compares old/new banks by scenario ID.  Missing requested IDs are
   a hard error.  Full reconstruction is promoted only if all three conditions
   pass: `AnyNCFSceneRate>=0.40`, selected false-safe floor `<=0.55`, and hard
   scene NCF recovery `>=0.20`.
9. **Build fingerprints.** Probe and full-data scripts fingerprint the proposal,
   label and configuration implementation so interrupted resumes cannot silently
   mix pre-fix and post-fix data.
10. **Fresh protocol version.** The causal protocol audit accepts
    `v16_8_3_fresh` and reports its fresh-label status separately.

### Data decision

Do **not** start the approximately four-day complete rebuild yet.  Existing data
is sufficient to establish the natural-basis quality, strong pair/root ranking,
certificate retention, corrected gate failure and fixed-bank proposal ceiling.
It is not sufficient to measure RMR-BCTE coverage, source-specific NCF yield, or
paper-grade fresh causal labels.  Run the paired label-only proposal probe first.
Only a passing probe justifies `PREPARE_COWP_V16_8_3_DATA_CN.sh`.

### Algorithm disposition

- **Retain:** non-coercive feasibility as a hard protected-priority certificate;
  natural roots; `s=(1-c)r+cq`; separate `q` and `b*`; protected BCOT; explicit
  uncertified fallback; certificate/shortlist separation.
- **Deepen after proposal repair:** distributional RCOT with censored minimum-safe
  burden, simultaneous one-sided calibration across protected relations,
  shift-aware/sequential risk control, and certificate-guided proposal refinement
  attached to a stronger flow/world-model/VLA planner.
- **Keep diagnostic-only or remove from claims:** flat candidate certificate,
  all-critical hard veto, automatic stop/yield safety assumptions, sparse cached
  Waymax outcomes as closed-loop evidence, and repeated BCOT-budget enlargement.

### Validation

- Full PyTest after v16.8.3 changes: **153 passed** (before final packaging; rerun
  recorded in the delivery validation file).
- New and modified Python modules compile.
- v16.8.3 proposal, data, mechanism, probe and full shell entry points pass
  `bash -n`.
- Paper adds a fixed-bank proposal-sufficiency proposition and RMR-BCTE details;
  no new empirical SOTA claim is made before the paired probe and closed-loop
  gates pass.

### Next decision rule

1. Run `NEXT_RUN_COMMANDS_V16_8_3_PROPOSAL_PROBE_CN.sh`.
2. If `promote_to_full_rebuild=false`, do not train or rebuild full data; use
   source-level statistics to determine whether conflict-region discovery,
   timing coverage, candidate budget, or the non-coercion label is the blocker.
3. If true, run `PREPARE_COWP_V16_8_3_DATA_CN.sh`, then
   `NEXT_RUN_COMMANDS_V16_8_3_MECHANISM_CN.sh`.
4. Waymax probe/full remains blocked until both `mechanism_verification.pass` and
   `calibration_feasible` are true under current metric semantics.

## v16.8.2 — Certificate/Shortlist Separation, Explicit Least-Coercive Fallback, Hard Protected Semantics, and BCTE Proposal Repair

### Triggering evidence

The uploaded `cowp_v16_8_1_rcot_consistent_v9base_seed2026` run completed
transport/planner training and disjoint learned-offline evaluation, but both
continuation conditions failed:

- `mechanism_verification.pass=false`;
- `calibration_feasible=false` (`status=least_violation`, selected budget 0.65).

The failure is not a lack of ranking signal. Held-out pair-witness AUPRC is
0.82033, protected BCOT false-safe AUPRC is 0.96498, and protected
RootTransport AUPRC is 0.86936. The failing operating-point metrics are
protected NCF recall 0.21176 (<0.30), protected NCF precision 0.49802 (<0.50),
accepted-candidate rate 0.06851 (<0.10), fallback 0.35874 (>0.25), and PBTR
0.59222 (>the calibration constraint 0.45). The proposal bank contains an NCF
candidate in only 0.27255 of held-out scenes, which is an upstream ceiling on
certificate selection.

### Confirmed evaluation/engineering defects

1. **Certificate acceptance was overwritten by the Pareto shortlist.**
   `_select_from_learned` first computed the hard RCOT certificate mask, then
   replaced it with a top-k Pareto frontier (runtime maximum 8 candidates).
   `LearnedAcceptedCandidateRate`, NCF recall and certificate coverage were
   therefore measuring the selector shortlist rather than the semantic
   certificate set.
2. **Valid-index fallbacks were undercounted.** Offline fallback selected a
   stop/yield candidate index, while the accumulator counted fallback only when
   the selected index was invalid. This made fallback semantics dependent on an
   implementation detail rather than the actual decision path.
3. **Calibration could silently consume stale metrics.** Old JSON rows had no
   marker distinguishing the pre-v16.8.2 shortlist semantics from corrected
   certificate semantics.
4. **Checkpoint selection included inactive/frozen objectives.** The witness
   score emphasized the disabled all-root recovery loss and omitted direct root
   burden/consistency/budget terms. The planner score included frozen transport
   and zero-weight candidate-certificate losses, diluting trainable improvement.
5. **The legacy flat candidate certificate is collapsed but inactive.** Its
   selected means (`ncf≈0.04`, `false_safe≈0.999`, `quality≈1e-5`) are not a
   valid main selector signal. It remains diagnostic-only and is excluded from
   checkpoint selection and paper claims.
6. **The previous “all top-level shell scripts pass” validation claim is not
   reproducible from the uploaded archive.** Multiple historical v15--v16.6
   wrappers retain CRLF or obsolete heredoc syntax. They are outside the
   v16.8.2 execution chain, but the delivery now validates only the enumerated
   current scripts and records the historical failures instead of silently
   treating the whole repository as shell-clean.

### Algorithm corrections

1. **Separate semantic certificate from selection shortlist.** Evaluation and
   online policy now keep `certificate_accepted` and `selection_shortlist` as
   different masks. Gates use the former; shortlist diagnostics are reported
   separately.
2. **Explicit fallback contract.** Fallback is always marked, even when it
   returns a valid candidate. It selects the minimum predicted coercion-risk
   candidate using transport UCB, protected-rule risk, action risk, pressure,
   cached outcome risk and a small utility term. Stop/yield is only a weak tie
   preference, not an assumption of non-coerciveness.
3. **Hard protected-relation anchor.** `AgentPriority` and
   `EqualOrNegotiated` relations are protected by rule and cannot be diluted by
   a learned priority head. Learned priority is used only for unknown relations;
   `Unprotected` remains unprotected.
4. **Full-certificate uncertainty target.** Mode uncertainty is trained against
   the maximum normalized error across conflict, retain, same-root recovery and
   minimum safe burden. It is no longer calibrated only to conflict/retention
   while being used as a UCB for the complete RCOT risk.
5. **Bidirectional Conflict-Time Envelope (BCTE).** Candidate generation solves
   bounded ego acceleration for arrival immediately before/after nearby agents'
   plausible arrival times at the same conflict region. This targets the
   measured `AnyNCFSceneRate=0.27255` proposal-coverage bottleneck. BCTE is a
   proposal repair; it does not weaken the downstream conventional or RCOT
   certificates.
6. **Checkpoint scores aligned to active objectives.** Transport selection now
   uses conflict-conditioned recovery, recovery ranking, root burden,
   q--b* consistency, uncertainty and candidate budget losses. Planner selection
   uses only trainable planner losses.
7. **Versioned metric semantics.** Learned-offline rows now contain
   `CertificateSemantics/Version=v16_8_2_decoupled` and
   `FallbackSemantics/ExplicitAccounting=true`. Calibration and verification
   reject stale rows.

### Training-state decision

The uploaded transport run ended after 14 epochs and the planner after 10; every
recorded checkpoint improved and both histories end with
`checkpoint/no_improve_checks=0`. These are schedule-truncated models, not
converged models. v16.8.2 raises defaults to 24 transport and 16 planner epochs,
while preserving early stopping. The natural basis remains frozen and is not
retrained.

### Local validation

- Full test suite: **150 passed**, one upstream PyTorch prototype warning.
- Python `compileall`: passed.
- The eight scripts in the v16.8.2 execution chain pass `bash -n`.
- TeX compiles with `pdflatex` to 19 pages.
- CPU realistic preflight and mechanism-overlay causal-audit smoke pass.
- A repository-wide diagnostic finds 38 failures among 73 historical top-level
  shell scripts, primarily CRLF/obsolete heredoc wrappers from v15--v16.6; they
  are not in the v16.8.2 execution path and are not represented as clean.

### What is intentionally not claimed

- The corrected selector metrics have not been recomputed here because the
  uploaded result package omits the large checkpoints and the local environment
  has no CUDA/Waymax.
- BCTE proposal coverage is not yet empirically validated; a fresh candidate
  label/cache rebuild is required for learned-offline evidence.
- No SOTA or calibrated safety guarantee is claimed from sparse cached outcomes
  (selected coverage about 20--33%, finite log-divergence count 0).

### Next decision rule

1. Re-evaluate the existing v16.8.1 checkpoints under v16.8.2 semantics to
   isolate the metric/selector/fallback correction from retraining.
2. If the gate remains infeasible, retrain transport/planner to convergence with
   corrected uncertainty and checkpoint scores.
3. If `ProposalCoverage/AnyNCFSceneRate < 0.35` or PBTR/fallback remain the
   blockers, rebuild fresh BCTE labels/cache before any budget tuning.
4. Do not answer failure by only raising the BCOT budget; the previous sweep
   already shows recall plateauing while PBTR worsens.


## v16.8.1 — Definition-Consistent RCOT, Direct Root Burden, and Recovered Execution Chain

### Triggering evidence

The uploaded `cowp_v16_8_pipeline_v9labels_seed2026` and
`cowp_v16_8_rcot_v9base_seed2026` runs did not reach training. Both terminate in
`36_audit_causal_protocol.py` because the launcher requests
`v16_8_root_conditioned_overlay` while the parser accepts only `v15` and
`v9_reuse`. The cache/overlay diagnostics immediately before the failure are
healthy: 20,440/20,440 train files, 5,013/5,013 validation files, zero overlay
errors, zero split filename overlap, complete critical mapping, and no response
root indices out of range.

### Definition corrections

1. **Implemented the complete root transport equation in model forward.** The
   predicted transported root probability is now
   `retain_prob + conflict_prob * mode_recovery_prob`, i.e.
   `s=(1-c)r+cq`. Earlier forward inference omitted `c*q`, even when supervision
   rebuilt transported OPR.
2. **Separated low-burden recoverability from minimum safe burden.** The
   candidate--agent--root head now predicts `q_ikm` and `b*_ikm` through separate
   channels from a shared root-conditioned latent. `q=0` no longer forces the
   no-safe-response sentinel; a finite high-burden safe response remains
   representable.
3. **Added confidence-masked q--b* consistency.** On confidently labelled
   conflicting roots, the loss aligns `q` with a smooth thresholding of `b*`
   around the adaptive burden budget while preserving direct supervision and
   independent outputs.
4. **Made the direct root representation the primary certificate.** OPR and
   root-CVaR use direct `q`/`b*`. The compact generic response bank is retained
   only for auxiliary reconstruction, qualitative visualization, and ablation.
5. **Unified the natural-root probability measure.** Label adaptation, future
   overlay generation, and model inference now apply the same `p_min` support,
   surviving-mass renormalization, and active-support probability floor.
6. **Removed recovery-loss dilution.** The main recovery objective is
   conflict-only BCE plus conflict-conditioned ranking; the all-root recovery
   BCE is disabled.

### Execution and engineering corrections

1. `36_audit_causal_protocol.py` now accepts
   `v16_8_root_conditioned_overlay` and separately reports engineering,
   mechanism-overlay, and fresh-v15 protocol status.
2. Added an audited checkpoint compatibility loader for the exact 4-to-5-row
   expansion of `set_transport.mode_out`; all other shape mismatches remain
   hard failures.
3. Fixed the background launcher so the parent exits after spawning the nohup
   child.
4. The natural transfer manifest now records both checkpoint and training
   history with SHA-256 hashes. New probe/full launchers restore them in a fresh
   shell instead of assuming inherited environment variables.
5. Added `NEXT_RUN_COMMANDS_V16_8_PROBE_CN.sh`; full evaluation is separated
   from probe and requires a completed probe delta.
6. Changed default output to
   `outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026` to preserve strict
   provenance and prevent reuse of artifacts produced by the incorrect code.

### Dataset decision

- Reuse the existing v16.8 overlay for the next mechanism run; no rebuild is
  required for these model/loss fixes.
- Keep log-divergence supervision disabled because finite coverage is zero.
- Treat cached Waymax outcomes as partial auxiliary labels, not unbiased
  full-candidate closed-loop evidence.
- Do not use this v9-base overlay as evidence that the fresh v15/v16 causal-label
  protocol has been validated.

### Validation

- Full test suite: 144 passed, 5 skipped after the new launcher regression test.
- Python compileall: passed.
- All top-level shell scripts: `bash -n` passed.
- CPU realistic model/loss preflight: passed.
- Causal-audit smoke: engineering and mechanism-overlay protocol passed; fresh
  v15 label protocol correctly remained false.

### Next decision rule

Do not enter Waymax probe unless the disjoint learned-offline report has both
`pass=true` and `calibration_feasible=true`, including priority RootTransport
AUPRC, NCF recall/precision, acceptance, fallback, PBTR improvement, and global
false-safe improvement gates. Do not respond to a failed gate by only increasing
`BCOT_RISK_BUDGET`.

## v16.8 — Root-Conditioned Counterfactual Transport (RCOT), Canonical OPR, and Immutable Mechanism Checkpoint

### Evidence from the uploaded v16.7 mechanism-isolation run

The v16.7 run confirms that the natural basis is not the current bottleneck and that the
candidate-level ranking signal is substantially stronger than the failed gate suggests.
On the disjoint held-out partition:

- pair-witness AUPRC: **0.72893**;
- protected-priority BCOT false-safe AUPRC: **0.95611**;
- global BCOT false-safe AUPRC: **0.96146**;
- protected-priority RootTransport AUPRC: **0.20273**;
- protected NCF precision: **0.72535**;
- protected NCF recall: **0.19062**;
- accepted-candidate rate: **0.04902**;
- fallback rate: **0.44972**;
- selected PBTR: **0.44331**, versus **0.65151** for conventional safety;
- selected global false-safe rate: **0.25379**, versus **0.59098** for conventional safety.

Thus the learned candidate score already orders false-safe candidates well and reduces
burden transfer directionally, but the root-recovery target and hard selection semantics
make the usable operating region collapse.  The gate failure is not evidence that the
coercion concept is ineffective; it is evidence that v16.7 did not measure or preserve
same-root recoverability according to the paper's own definition.

### Confirmed definition and engineering defects

1. **OPR omitted recovered conflicting roots.** Fresh witness construction and the
   paper-aligned training adapter both accumulated only the non-conflicting retained term.
   They did not implement `s=(1-c)r+cq`, so a conflicting root was counted as destroyed
   even when its own low-burden safe response existed.
2. **Global response-bank coverage was mistaken for recoverability.** Responses were
   generated from a candidate-independent global control lattice, globally truncated,
   and assigned to roots after the fact by nearest full-horizon trajectory.  A negative
   `q_ikm` therefore meant that the finite generic bank did not happen to cover root `m`,
   not that root `m` was unrecoverable under a bounded same-root counterfactual.
3. **Train/evaluation target mismatch remained.** Training reconstructed paper-aligned
   targets from transport fields, while learned-offline evaluation could calibrate and
   gate against stale cached false-safe/NCF labels.  The same checkpoint was therefore
   optimized and judged under different certificates.
4. **Missing response identities were clamped to root zero.** Legacy response slots with
   root index `-1` could be converted to root 0 by clamping, creating artificial recovery
   positives for the first root.
5. **Safe response was not always low-burden recovery.** For old caches without an
   explicit `is_low_burden` flag, safety alone could be treated as `q=1`; v16.8 derives
   the predicate from the adaptive burden budget `beta`.
6. **Planner training could rewrite the mechanism.** The planner stage continued to
   update SetTransport and response-decoder parameters, so the final planner checkpoint
   could trade a previously verified certificate for ranking loss.  This invalidated
   clean attribution to the mechanism stage.
7. **Duplicate severe veto.** A localized protected-pair severe veto and an aggregate
   candidate severe veto were both applied.  Under noisy recovery labels this doubled
   conservatism, suppressing acceptance without adding a distinct causal guarantee.
8. **Dangerous launcher defaults.** Direct execution of the v16.8 driver still defaulted
   to the old v9 transport cache and old sidecar name.  v16.8 now defaults exclusively to
   `tensor_cache_*_waymax_transport_v16_8` and records the overlay protocol in provenance.

### Root-Conditioned Counterfactual Transport (RCOT)

v16.8 makes same-root transport an explicit, independently testable object instead of a
by-product of a generic response decoder:

- For every valid conflicting natural root, generate a bounded family of longitudinal
  timing residuals around that root.
- Re-parameterize time along the original root polyline.  The control may slow, wait, or
  mildly accelerate, but it may not change the root's spatial/topological maneuver.
- Give every conflicting root the same oracle search budget before filling any compact
  neural-response slots.  Global top-R truncation can no longer create false negative
  recovery labels.
- Store explicit `response_root_index` and `root_affinity`.  Legacy/global responses use
  a soft 1/3/5/8-second affinity and expose target confidence instead of forcing a brittle
  full-horizon nearest-root label.
- Supervise continuous `root_low_safe_score`, `root_target_confidence`,
  `root_min_safe_burden`, `root_recovery_mass`, and `transported_opr`.
- Define transported OPR exactly as the probability-weighted retained mass after applying
  `s=(1-c)r+cq`; the same function is used by label generation, training adaptation, and
  learned-offline evaluation.

RCOT is the mechanism-level novelty to emphasize in the paper: coercion is not merely a
candidate risk class.  It is the destruction of protected agents' probability mass over
pre-interaction natural options, and a plan is non-coercive only when conflicting roots
can be transported into low-burden safe responses without changing root identity.

### Learning and selector changes

- Add conflict-conditioned root-recovery BCE and within-pair/root ranking.  Non-conflict
  roots no longer dominate the recovery objective.
- Use soft direct root targets when available and mask low-confidence legacy assignments.
- Use symmetric class-balanced presence loss plus conflict-conditioned recovery-magnitude
  loss for aggregate root recovery.
- Keep candidate-level BCOT as a monotone aggregation of unrecovered conflict mass,
  burden tail and option shortfall; it remains a mechanism certificate, not a generic
  black-box candidate classifier.
- Disable the legacy flat candidate-certificate loss in the main v16.8 configuration.
  It duplicates the mechanism score and has not shown independent causal value.
- Retain the localized protected-pair severe veto, but make the aggregate severe veto an
  explicit ablation and default it off.
- Extend the calibration sweep to 0.98.  Class-balanced logits are ranking scores and need
  not be calibrated probabilities; a sweep ending at 0.70 was not a valid proof that no
  operating point exists.

### Immutable mechanism checkpoint

The main v16.8 planner protocol freezes SetTransport and the response decoder during
planner training and sets planner-stage response/transport auxiliary scales to zero.
The final planner therefore optimizes candidate ranking against a fixed certificate.
Unfreezing remains available only as a named ablation.  This separates:

1. whether RCOT learns the coercion mechanism;
2. whether the selector can exploit a valid certificate;
3. whether proposal coverage limits closed-loop performance.

### What is retained, deepened, or rejected

**Retain:** typed natural decoder, yaw correction, OBS capacity, mass-aware root envelope,
source-aware multi-horizon natural-root alignment, protected-priority semantics, monotone
BCOT aggregation, candidate false-safe ranking, PBTR/OPR/BTE-CVaR metrics, fail-fast
learned-offline gate, and real Waymax online evaluation.

**Deepen now:** same-root recovery through RCOT; confidence-aware root supervision;
immutable certificate-to-selector interface; protected PBTR--coverage calibration.

**Do not repeat without new evidence:** global generic-response nearest-root `q` labels;
OPR computed without the conflict-recovery term; treating safe as low burden; unconditional
aggregate severe veto; planner-stage joint retuning of a verified certificate; using the
legacy flat candidate classifier as the primary mechanism; threshold-only fixes; sparse
cached Waymax outcomes as closed-loop evidence.

**Defer until the mechanism gate passes:** coercion-aware proposal refinement.  The current
proposal bank has an AnyNCF-scene ceiling of roughly 37.7%, so it will eventually limit
recall and progress.  Changing proposals in the present isolation experiment would,
however, confound whether RCOT repaired the mechanism.  The next algorithmic phase should
add conflict-time/local-lane timing refinement around protected roots and compare it with
the unchanged proposal bank under an identical frozen RCOT certificate.

### Development gate and paper evidence

The v16.8 immediate gate is deliberately a continuation gate, not a publication claim.
Defaults require protected root AUPRC >= 0.50, protected NCF recall >= 0.30, precision >=
0.50, accepted rate >= 0.10, fallback <= 0.25, PBTR <= 0.45, and positive PBTR improvement.
These are diagnostic promotion targets, not CCF criteria.  If root AUPRC remains below
0.50 after canonical labels are regenerated, stop before Waymax and request dataset
stratification by conflict geometry, protected relation, root source, root mass, horizon,
and low-burden oracle feasibility.

Paper-level claims still require a fresh `formal_v18` rebuild, three or more seeds, exact
checkpoint attribution, paired real Waymax online rollouts, reactive-agent sensitivity,
and reporting standard closed-loop metrics together with PBTR, protected OPR, BTE-CVaR25,
NCF retention and non-coercive progress regret.  v16.8 does not guarantee SOTA or CCF-A
acceptance; it removes known target and engineering confounds and makes the core coercion
hypothesis falsifiable.

### Verification status

- pytest: **142 passed**;
- Python compileall: **passed**;
- all top-level shell scripts: **bash -n passed**;
- no v16.8 GPU training or Waymax rollout was executed in this environment;
- therefore the mechanism gate is **not yet claimed to pass**.

## v16.7 — Priority-Aware Mechanism Repair, Monotone Certification, and Paper-Grade Closed-Loop Protocol

### Evidence from the uploaded v16.6 run

The v16.6 natural foundation is healthy and is no longer the blocking stage:

- natural-basis gate: `pass=true`;
- natural-effectiveness gate: `pass=true`;
- aligned component attribution: `pass=true`, `paper_claim_ready=false`;
- OBS-capacity paired OBS gain: **0.07049 m**, 95% paired bootstrap CI
  **[0.04353, 0.09871] m**;
- mass-aware root envelope: exact squared-excess reduction **0.31165**, violation-mass
  reduction **0.05373**, with both paired confidence intervals excluding zero.

The full pipeline stopped at the held-out mechanism gate.  The calibrated budget was
0.70 with `least_violation` status and no feasible calibration point.  On the disjoint
held-out split:

- pair-witness AUPRC: **0.71293** (the pair witness is usable);
- RootTransport conflict-conditioned AUPRC: **0.22524**;
- BCOT false-safe AUPRC: **0.40640**;
- learned NCF recall: **0.20232**;
- accepted candidate rate: **0.09036**;
- fallback rate: **0.23464**;
- selected false-safe rate: **0.47207**, versus **0.59737** for conventional safety.

This is a downstream transport/certificate failure, not a natural-decoder failure.  The
budget sweep from 0.05 to 0.70 contained no operating point satisfying recall, coverage,
fallback and false-safe constraints simultaneously; threshold relaxation alone cannot
repair the mechanism.

### Confirmed engineering and label defects

1. **Asymmetric BCOT class weighting.** False-safe examples constitute roughly 87--88%
   of discriminative candidate labels, while the old implementation used positive-class
   `pos_weight` clamped to at least one.  It could not downweight the majority class and
   shifted candidate risks upward, explaining low acceptance and high fallback even at
   the largest calibrated budget.
2. **Broken arrival-order priority label.** `_first_arrival_to_close_points` searched
   synchronous samples and returned the same timestamp for both agents.  Its arrival
   comparison could never establish who reached a shared conflict region first.
3. **Unsupported traffic-signal inference.** `controlled_by_signal` only indicates that
   a lane is signal-controlled; without live phase association it does not establish
   right-of-way.  The old rule injected systematic priority noise.
4. **Root-assignment mismatch.** Training and evaluation used duplicated alignment logic
   and full-horizon ADE, permitting source-root swaps.  v16.7 shares a multi-horizon,
   source-aware alignment cost in training and evaluation.
5. **Proposal/paper mismatch.** The implementation uses a finite kinematic primitive
   bank, not route-conditioned lattice-MPC.  Fixed lateral and terminal primitives can
   leave the local road corridor and contributed to the high cached off-road fraction.
6. **Sparse outcome evidence.** Only about 23.9% of selected held-out candidates had
   attached Waymax outcomes and no finite log-divergence labels were available.  These
   cached outcomes are auxiliary diagnostics, not a substitute for a full online
   closed-loop run.

### Mechanism repair

- Replace one-sided candidate BCE with symmetric inverse-frequency class balancing.
- Separate protected-priority BCOT risk from the all-critical global diagnostic risk.
- Preserve mechanism interpretability with positive, normalized, monotone weights over
  unrecovered conflict mass, burden-tail activation and option shortfall.  No generic
  candidate classifier is allowed to bypass the COWP mechanism.
- Add candidate-level within-scene ranking so false-safe candidates must score above
  non-coercive feasible candidates.
- Add closing speed, near-conflict fraction and swept midpoint clearance to root
  geometry; add root conflict ranking supervision.
- Blend learned priority probability with corrected rule labels, while keeping the
  all-critical diagnostic separate from the protected hard veto.
- Use one shared source-aware multi-horizon root alignment implementation for training
  and evaluation.

### Priority-aware semantics and metrics

The hard feasibility set now protects agent-priority and explicitly negotiated/equal
relations.  Ego-priority relations remain in the all-critical diagnostic but do not
create the same veto.  This prevents the priority-aware motivation from degenerating
into a universal "never inconvenience anyone" rule.

Primary mechanism metrics are reduced to a small set:

- **PBTR:** protected-priority burden-transfer rate;
- **protected OPR:** retained same-root low-burden probability mass;
- **BTE-CVaR25:** worst-quartile severity of the worst protected relation per scene;
- **NCF scene retention:** whether an existing non-coercive proposal survives the
  certificate;
- **non-coercive progress regret:** selector loss relative to the best available NCF
  proposal;
- **PBTR--coverage curve:** burden transfer versus certified scene coverage.

Global FSR remains a stress diagnostic. CBS, HBCR, WLA and MTA are retained only as
appendix or debugging metrics unless future experiments show unique explanatory value.

### Proposal and data correction

- Correct independent path-arrival times and remove signal-presence right-of-way labels.
- Add a cheap local lane-corridor screen for synthetic kinematic proposals; skip the
  screen when map geometry is too sparse to define a reliable corridor.
- Keep the logged trajectory as an observed reference.
- `PREPARE_COWP_V16_7_DATA_CN.sh` builds a fresh paper-grade `formal_v17` cache, because
  corrected priority labels and map-screened proposals cannot be retrofitted into old
  v9 overlays.  Candidate Waymax replay defaults to 24 balanced candidates per scene.

### Evaluation gates

The v16.7 mechanism gate is a development-continuation gate, not a paper-claim gate. It
requires disjoint calibration/held-out partitions and checks protected NCF recall and
precision, priority BCOT/root AUPRC, accepted rate, fallback and PBTR improvement. Global
false-safe metrics remain anti-degeneration diagnostics. Publication claims additionally
require at least three independent seeds and a real online Waymax run.

### Efficiency changes

- Vectorize hard root-ranking loss over the root axis instead of Python loops over
  batch/candidate/agent triples.
- Compute source-aware root alignment once and reuse it across transport losses.
- Retain v16.6 frozen-backbone, trainable-parameter-only AdamW, static DDP and bucket-view
  optimizations.
- Perform lane-corridor screening only during label/cache construction, not training.
- Reuse the validated natural checkpoint for the immediate mechanism rerun; transport
  and planner are retrained from a fresh optimizer state.
- Final package regression: **138 passed**; all shell scripts pass `bash -n`; Python
  compileall and internal TeX label/reference checks pass. The uploaded source did not
  include its `.bib` file, so citation resolution was not treated as a code failure.

### Current decision

- **Keep and deepen:** typed natural decoder, yaw fix, OBS capacity, mass-aware root
  identity, same-root OPR, hard protected-priority certificate and fail-fast gates.
- **Partially supported:** planner ranking and hard certificate; the offline certificate
  reduces HBCR substantially but remains too conservative and has no online proof.
- **Still unproven:** emergency hard projection contribution, conformal coverage,
  reactive-agent causal burden transfer, final selector quality and CCF-A/SOTA closed-loop
  performance.
- The immediate v16.7 experiment may reuse the validated v16.6 natural checkpoint to
  isolate the downstream repair. Final paper experiments must rebuild v17 data and rerun
  natural attribution, mechanism training and three-seed online evaluation.


## v16.6 — Protocol-Aligned Attribution, Exact Identity Objective, and Natural-Stage Systems Repair

### Why the v16.5 attribution gate failed

The uploaded v16.5 artifacts are numerically healthy: the strict natural-basis and
natural-effectiveness gates passed, optimizer steps were non-zero, AMP skips were zero,
and the graph remained frozen.  The failed attribution JSON nevertheless mixed two
engineering/protocol defects with the component question:

1. The main external diagnostic evaluated the selected **epoch 15** checkpoint, while
   both ablations evaluated their own **epoch 19** best checkpoints.  The result was
   therefore not an equal-training-time comparison.  At the shared validation epoch 15,
   main versus no-OBS-capacity improved OBS minADE by **0.05149 m** and overall natural
   trajectory loss by **0.02718 m**.  Main versus no-envelope improved OBS minADE by
   **0.11406 m**, overall trajectory loss by **0.03647 m**, and the exact validation
   identity penalty by **0.32689**.
2. The v16.5 gate required the probability-weighted *mean path ratio* to fall by 0.03,
   although the implemented loss optimizes the floor-smoothed probability-weighted
   **squared excess above the interior margin**.  The uploaded own-best reports showed
   only 0.01119 mean-ratio reduction, but a 0.06651 absolute reduction in violating
   probability mass (about 32.5%) and simultaneous OBS/overall improvement.  The old
   primary check was therefore not aligned with the trained objective.

Consequently, the uploaded gate failure does **not** establish that either component is
ineffective.  It establishes that the v16.5 attribution protocol was insufficient to
make the claim.  Final component evidence must be regenerated at a common checkpoint
and on paired scenes.

### Attribution correction

- All three arms are diagnosed at the epoch selected by the main model; an ablation may
  no longer substitute its own best epoch.
- Reports include a deterministic sampled-scene hash and scene-level paired metrics.
- The identity component is evaluated with the exact squared-excess objective optimized
  in training, probability mass outside the soft envelope, emergency-envelope p99, and
  prediction non-regression.  Mean path ratio remains diagnostic only.
- Paired bootstrap confidence intervals are emitted for OBS/overall error, exact excess,
  and violation mass.
- `pass` is explicitly a **development continuation gate**: it allows transport/planner
  evidence collection when isolated components are active and non-harmful.  A separate
  `paper_claim_ready` field remains false until at least three independent seeds provide
  publication-level evidence.
- `RUN_NATURAL_ATTRIBUTION_V16_6_CN.sh` can reuse existing epoch checkpoints and rerun
  only the aligned diagnostics; if an exact ablation checkpoint is absent, it retrains
  that arm only to the main-selected epoch with `save-every=1`.

### Natural-stage training-system corrections

- Permanent natural-stage freezing and inactive architecture freezing are applied before
  DDP and AdamW construction.
- The checkpoint-compatible legacy dense natural head is frozen for typed decoders.
- AdamW tracks only trainable parameters; permanently frozen graph/candidate/witness
  modules no longer allocate optimizer bookkeeping.
- Permanently frozen natural training uses static DDP with unused-parameter traversal
  disabled when supported by the installed PyTorch.
- Natural checkpoint selection now uses a component-neutral common prediction score
  (`traj`, `OBS minADE`, and branch minADE) rather than total loss or terms removed by an
  ablation.  Attribution still requires an identical epoch even with this fairer score.

### Held-out calibration and downstream evidence repair

The v16.5 full-pipeline launcher swept and calibrated the candidate BCOT budget on
`VAL_CACHE`, then reported selector/mechanism metrics on the same cache.  Moreover,
`25_verify_mechanism_effect.py` replaced the held-out method metrics with
`calibration.selection_metrics`, and `30_diagnose_bcot_result.py` did the same.  This
created calibration/evaluation leakage and could overstate AUPRC, NCF recall, accepted
rate, fallback and false-safe improvement.

v16.6 therefore:

- deterministically partitions learned-offline validation scenes by dataset index;
  remainder 0 of modulo 2 is calibration-only and remainder 1 is held-out-only;
- records partition modulus, remainder, scene count and an index SHA-256 in every
  learned-offline metrics row;
- chooses the BCOT operating point only from the calibration partition;
- evaluates COWP and internal baselines only on the disjoint held-out partition;
- rejects mechanism verification unless the deterministic partitions prove
  calibration/held-out disjointness and the held-out operating point equals the
  calibrated budget;
- prevents calibration metrics from overriding held-out metrics in readiness/gate
  reports.
- reports scene-level proposal/certificate coverage (`AnyNCF`, `AnyAccepted`,
  `AnyAcceptedNCF`, empty-certificate rate and NCF scene retention) so low accepted
  rate or high fallback can be attributed to the proposal bank versus the certificate.

This is an evaluation-validity correction, not an algorithmic performance claim.  It
may lower reported numbers, but those numbers are the ones suitable for deciding which
selector or transport component actually needs improvement.

### Paper correction

The v16.5 main text correctly introduced a probability-mass-aware multi-horizon semantic
envelope, but its appendix still described the rejected v16.4 endpoint trust ball and
referenced a deleted equation.  v16.6 makes the manuscript match the code:

- defines the pre-projection maximum path ratio and the normalized squared-excess loss;
- uses the same floor-smoothed detached mass as OPR;
- distinguishes the semantic soft envelope from the wider emergency projection;
- evaluates exact excess and violating mass in attribution;
- labels conformal expansion as an unimplemented optional extension;
- prevents single-seed continuation gates from being presented as paper-level evidence.

### Current status and decision

- **OBS capacity:** directionally supported across the common validation trajectory, but
  external paired epoch-15 diagnostics are still required before a component claim.
- **Mass-aware root envelope:** strongly supported as an active regularizer by lower
  exact validation penalty and violation mass without prediction harm; paired aligned
  diagnostics remain required for formal attribution.
- **Full pipeline:** may run only after the v16.6 aligned development gate passes.  The
  resulting transport/planner/Waymax outputs are evidence collection, not automatic SOTA.
- **Regression tests:** 119 passed; TeX compiles successfully (bibliography not bundled).

## v16.5 — Probability-Mass-Aware Natural Root Identity and Fair Attribution

### Why v16.4 was not promoted

The v16.4 natural recovery checkpoint learned successfully, and the yaw reference-frame
repair removed the v16.3 heading inconsistency.  However, the v16.4 attribution gate did
not isolate all claimed components and therefore could not support the paper claims:

- v16.4 main 8 s weighted ADE: **1.2339 m**; OBS: **2.9174 m**;
- `no_effectiveness_loss`: **1.2159 m** overall; **2.8724 m** OBS;
- `no_obs_capacity_boost`: **1.2599 m** overall; **3.0522 m** OBS;
- `no_integrated_trust_region`: **1.1700 m** overall; **2.6159 m** OBS;
- the fixed endpoint trust region reduced residual endpoint p99 from **46.709 m**
  to **20.000 m**, but worsened OBS ADE by **0.3014 m**;
- the OBS capacity boost improved OBS ADE by **0.1348 m** and is retained.

Two engineering defects invalidated a literal interpretation of the v16.4 loss ablation:

1. `train_cowp_v16_no_effectiveness_loss.yaml` removed several unrelated terms at once
   (OBS gain, neutral/priority preservation, kinematic, smoothness, mode-use and trust),
   so it was not a one-factor control.
2. Natural launchers claimed to freeze the graph backbone but silently unfroze it after
   two epochs.  Frozen epochs also inherited training-mode dropout after `model.train()`,
   making representations stochastic and forcing unnecessary graph gradients later.

### Algorithm change

v16.5 keeps the typed causal dynamics decoder and source-adaptive OBS capacity, but
replaces the fixed endpoint trust ball with **probability-mass-aware, multi-horizon root
identity preservation**:

1. Each source receives a soft multi-horizon envelope over 1/3/5/8 s displacement.
2. The soft identity loss is weighted by detached mode probability (with a small floor),
   so probability-carrying roots are constrained consistently with OPR retained-mass
   semantics and the network cannot evade the constraint by changing logits.
3. A wider emergency envelope is applied to every mode as a hard projection.  It is a
   numerical/physical guard, not the semantic definition of a natural root.
4. The unsupported explicit OBS-gain and prior-preservation bundle is disabled in the
   main v16.5 objective.  Source-adaptive capacity is tested by an isolated control.
5. Redundant velocity/yaw consistency losses are disabled because the dynamics
   integration now enforces them by construction; a small control-smoothness term remains.

Default soft endpoint budgets are OBS/NEU/PRIO = 20/8/6 m and emergency budgets are
48/16/12 m.  These are development settings and must be frozen before test evaluation;
they are not paper claims by themselves.

### Engineering and speed changes

- The graph encoder remains frozen and in deterministic `eval()` mode for the entire
  natural stage by default (`natural_graph_unfreeze_epoch=-1`).
- Frozen graph inference uses `torch.no_grad()`, avoiding graph activation storage and
  backward computation.
- 1/3/5/8 s pairwise trajectory losses are computed from one distance tensor and one
  cumulative sum instead of four repeated broadcasts.
- Diversity computation temporally subsamples the 80-step horizon (default stride 4).
- Expensive in-training base-effectiveness pair comparisons are disabled; the external
  2,000-scene learned-natural diagnostic remains the authoritative effectiveness test.
- Validation defaults to every two epochs instead of every epoch.
- Attribution is reduced from three auxiliary full trainings to two isolated controls:
  no OBS capacity and no mass-aware root envelope.
- Main and ablation runs now use the same seed, initialization, DDP topology, workers,
  prefetch, precision, epoch count and validation schedule.

### New promotion gates

The natural gate now separates prediction quality, physical sanity and root identity:

- learned 8 s quality/gain thresholds;
- source-prior non-regression;
- finite controls and kinematic consistency;
- probability-mass-weighted soft-envelope ratio and violation rate;
- projected emergency-envelope p99.

The attribution gate promotes only when:

- source-adaptive OBS capacity improves OBS without unacceptable NEU/PRIO regression;
- mass-aware root preservation reduces probability-weighted envelope violation/ratio;
- that identity improvement does not impose excessive OBS or overall prediction cost.

### Status

- **Engineering regression tests:** 115 passed in the delivery environment.
- **Algorithm performance:** not yet claimed.  v16.5 must be retrained and pass both
  natural effectiveness and isolated attribution gates before transport/planner/selector.
- **Do not reuse v16.4 attribution results as v16.5 evidence.**

## v8 — Aggregate structured certificate

- Added a threshold-connected hard certificate and a candidate-level classifier.
- Result: false-safe selection and burden decreased, but EP collapsed and fallback rose sharply.
- Rejected shortcut: threshold relaxation alone. Pair witness AUPRC remained about 0.43.

## v9 — Primitive-indexed transport supervision

- Added direct mode conflict/retain labels, response-root assignment and response auxiliary losses.
- Fixed hidden NPZ enumeration and response-root gather dimensionality.
- Seed 2026 learned-offline gate failed; no real Waymax online probe was executed.
- Evidence: witness AUPRC 0.4312, accepted NCF recall 0.1280, accepted candidate rate 0.0613.
- Failure diagnosis:
  1. transport did not receive candidate--natural trajectory geometry;
  2. `FREEZE_BACKBONE_EPOCHS=999` froze candidate/natural/witness identity modules;
  3. diffuse response slots distorted same-root recovery;
  4. mode-conflict validation BCE 0.722 was worse than the class-prior entropy baseline (~0.649);
  5. the generic candidate classifier dominated the claimed pair mechanism.

## v10-GCT — Geometry-Conditioned Transport

### Changes

- Added explicit compact candidate--natural relative geometry.
- Added balanced direct mode conflict/retain supervision and raw logits.
- Added response-root refinement and granular freeze.
- Replaced unweighted all-slot recovery with response-mixture-weighted recovery.
- Preserved natural-set auxiliary training during transport learning.

### Seed 2026 result

- `val/set_transport/mode_conflict`: **0.5074**, below the no-skill entropy baseline.
- `val/set_transport/mode_retain`: **0.2800**.
- Pair witness AUPRC: **0.7161** (v9: ~0.431).
- Candidate false-safe AUPRC: **0.9043**.
- Calibrated threshold 0.50:
  - EP 0.3913 vs conventional 0.3894;
  - fallback 0.2406 vs 0.1043;
  - OPR 0.7644 vs 0.7418;
  - HBCR 0.2928 vs 0.3970;
  - selected false-safe 0.4674 vs 0.5899;
  - accepted NCF recall 0.1267;
  - accepted candidate rate 0.0585.
- Across the entire threshold sweep, maximum accepted NCF recall was only **0.1391**.
- The learned-offline gate failed and therefore **no Waymax online probe ran**.

### What worked

1. Relative geometry solved the principal pair-conflict learnability failure.
2. Direct mode labels made the primitive transport mechanism measurable.
3. Granular freeze allowed candidate/natural/transport modules to adapt.
4. The certificate produced a real safety--efficiency trade-off rather than a disconnected threshold.

### What failed

1. Candidate feasibility still used an `any/max` reduction over up to six agents. A single
   moderate pair false positive rejected an entire candidate, causing multiplicative recall collapse.
2. Response-mixture-weighted recovery did not match the existential label semantics;
   root-recovery loss remained ~0.837 and response-root CE ~2.446.
3. The generic candidate calibrator could achieve high false-safe AUPRC without proving that
   primitive option transport caused the gain.
4. Attached sparse Waymax outcomes had unequal coverage and were not an online closed-loop result.

### Decision

- **Keep and strengthen** geometry-conditioned primitive conflict/retain learning.
- **Replace**, rather than tune, pairwise max/any candidate aggregation.
- **Do not** claim closed-loop superiority or SOTA from v10.

## v11-BCOT — Budgeted Counterfactual Option Transport (current)

### Hypothesis

The paper's object is the amount of low-burden natural option mass removed by an
ego candidate, not whether any one pair score crosses a threshold. Candidate
feasibility should therefore be a budget over transported option deficit, with a
separate high-confidence veto only for genuinely severe protected-priority pairs.

### Changes

1. **Pair transport deficit**
   - unrecovered conflicted natural mass;
   - burden excess on conflicted mass;
   - OPR shortfall.
2. **Candidate BCOT risk**
   - priority-weighted mean deficit;
   - smooth tail-risk deficit;
   - severe protected-pair probability.
3. **Budget gate**
   - threshold sweep now operates on candidate BCOT risk;
   - legacy pairwise `any/max` is retained only as an explicit ablation.
4. **Existential same-root recovery**
   - top response slots use a bounded fuzzy existential `max`;
   - duplicate or diffuse response slots cannot manufacture recovery;
   - a uniform root assignment contributes only `1/M`.
5. **Mechanism isolation**
   - the new candidate calibrator receives only transport statistics, not the generic candidate latent;
   - its final layer is zero-initialized for v10 checkpoint compatibility.
6. **Direct candidate-budget supervision**
   - disjoint NCF vs false-safe BCE;
   - within-scene NCF--false-safe ranking loss.
7. **Evaluation and fail-fast checks**
   - report `BCOT/FalseSafe_AUPRC` and `BCOT/RiskRankingPairAccuracy`;
   - gate checks the calibrated sweep point, not merely the default threshold;
   - require pair AUPRC >= 0.60 and BCOT false-safe AUPRC >= 0.65 before Waymax;
   - automatically run the pairmax ablation on the second GPU.

### Required evidence before promotion

Development gate:

- pair witness AUPRC >= 0.60;
- BCOT false-safe AUPRC >= 0.65;
- accepted NCF recall >= 0.30;
- accepted candidate rate >= 0.10;
- fallback <= 0.25;
- selected false-safe improvement >= 8 percentage points.

Paper-quality target:

- accepted NCF recall >= 0.50;
- fallback no more than conventional + 0.03;
- EP >= conventional or within a paired non-inferiority margin;
- relative selected false-safe reduction >= 25%;
- root-transport/BCOT significantly better than pairmax, candidate-only, soft-burden and planner-score ablations;
- 1000-scenario development and 5000-scenario x 3-seed paired online evaluation with confidence intervals.

## Prohibited shortcuts

- Do not claim closed-loop results from attached sparse Waymax candidate outcomes.
- Do not report SOTA from 100 scenes, one seed, or unmatched scenario subsets.
- Do not relax the gate merely to make the pipeline continue.
- Do not add log-divergence loss while finite label coverage is zero.
- Do not interpret generic candidate-classifier gains as proof of primitive transport.
- Do not repeat v10 pairwise max/any aggregation except as the registered ablation.
- Do not rebuild WOMD/tensor/transport caches for v11; the v9 sidecars are schema-compatible.

## v11-BCOT — Seed 2026 postmortem (do not promote)

### Observed result

The submitted `cowp_v11_bcot_probe100_seed2026` run stopped at the learned-offline
mechanism gate; no COWP Waymax probe or full online result was produced.

- pair witness AUPRC: **0.6808**;
- BCOT false-safe AUPRC: **0.4115**;
- BCOT within-scene NCF--false-safe ranking accuracy: **0.8306**;
- generic candidate false-safe AUPRC: **0.9003**;
- calibrated operating point (`0.40`, selected by least violation):
  - EP **0.3538** vs conventional **0.3870**;
  - fallback **0.2731** vs conventional **0.1043**;
  - OPR **0.8005** vs conventional **0.7426**;
  - HBCR **0.2529** vs conventional **0.3964**;
  - selected false-safe **0.4255** vs conventional **0.5881**;
  - accepted NCF recall **0.2011**;
  - accepted candidate rate **0.0831**.
- no sweep point met NCF recall >= 0.30 and fallback <= 0.25.

### Root causes found by code/log audit

1. **Transport-stage candidate supervision was absent.**
   `stage=witness` did not load `false_safe` or `noncoercive_feasible`, so
   `set_transport/candidate_budget` was exactly zero for every transport epoch.
2. **Option preservation was double-counted.**
   `mode_retained_low_safe` already means non-conflicting and low-burden, but the
   forward pass multiplied it again by `(1 - conflict_prob)`, systematically
   depressing OPR and acceptance.
3. **The existential burden statistic was mixture-weighted.**
   `_soft_min_burden` mixed response likelihood into a set-existence predicate,
   allowing a low-probability slot to be treated as unavailable even when it was
   a valid low-burden response.
4. **Natural-option drift invalidated root transport.**
   Validation natural set minADE rose from roughly 48 m to above 60 m during the
   witness stage.  A root certificate cannot be identified when its root
   trajectories move by tens of metres.
5. **The response bank was not root-conditioned.**
   It generated generic unordered response slots and then solved a difficult
   24-way root classification problem.  Root recovery remained about 0.86--0.88
   loss and response-root CE about 2.5--2.8.
6. **Two thresholds had been conflated.**
   The pair witness confidence threshold and candidate BCOT budget have different
   meanings but shared one evaluation parameter, making calibration ambiguous.
7. **The generic candidate certificate was not mechanism evidence.**
   The reported high candidate AUPRC came from a transport calibrator/generic
   latent path rather than a causal root-transport decision path.
8. **Dense response decoding was silently disabled outside response/all stages.**
   Enabling response trajectories in witness/planner configuration did not make
   the model decode them.

### Decision

- Keep: candidate--natural relative geometry, direct mode-conflict supervision,
  pair witness localization, burden decomposition, and budget rather than
  pairwise `any/max` aggregation.
- Replace: response-bank-defined root recovery, shared threshold calibration,
  silent missing-label behavior, and generic-certificate-controlled selection.
- Do not tune v11 further.  The dominant failures are semantic/structural, not a
  learning-rate or threshold issue.

## v12-RIOT — Root-Indexed Option Transport (current repair)

### Core algorithm change

For every critical agent and every natural option root `m`, v12 predicts the
primitive event

`T[k,i,m] = P(exists valid, safe, low-burden response under ego candidate k that preserves root m)`.

This root-indexed event is now the primary certificate.  Candidate BCOT risk is
computed from unrecovered natural-option mass, burden excess, OPR shortfall, and
a separate severe protected-priority veto.  The unordered response bank remains
only an auxiliary reconstruction/interpretability head.

### Implemented fixes

1. Load candidate `false_safe`, `noncoercive_feasible`, conventional-safety,
   utility, neutral/logged flags and beta labels in witness as well as planner
   stages.
2. Fail fast when required candidate or transport labels are missing; never
   silently replace missing supervision by all-negative targets.
3. Add candidate-budget label coverage, NCF rate and false-safe rate diagnostics;
   invalidate witness checkpoints with zero budget coverage.
4. Factor mode output into conflict and conditional low-safe retention, avoiding
   duplicate conflict suppression in OPR.
5. Supervise direct natural-root recovery by scatter-reducing explicit response
   slot labels (`valid & safe & low_burden & response_root_index`).
6. Preserve legacy response-root reconstruction only as
   `response_root_exist_aux` with low auxiliary weight.
7. Remove response-mixture probability from the existential soft minimum burden.
8. Add a dedicated natural-basis repair stage, freeze graph during natural
   repair, freeze the repaired natural module during witness training, and set
   witness natural auxiliary weight to zero.
9. Separate `pair_witness_threshold` from `candidate_transport_budget` throughout
   offline and Waymax evaluation.
10. Add one-pass BCOT budget sweep and `31_calibrate_bcot_budget.py`.
11. Make the main selector transport-pure: generic candidate certificate is
    retained only as an ablation/diagnostic; rule, action and rollout risks remain
    explicit safety shields.
12. Wire dense response decoding correctly for witness/planner when enabled; the
    default v12 run keeps it disabled to avoid unnecessary memory because RIOT
    does not depend on dense decoded response trajectories.
13. Add `32_gate_natural_basis.py`; transport training is prohibited unless the
    repaired natural basis passes the registered minADE thresholds.
14. Add pairmax and Pareto-frontier ablation configs and a two-GPU v12 driver.

### Validation performed before release

- `python -m compileall`: pass;
- full test suite: **70 passed**;
- `bash -n run_cowp_v12_dual_gpu.sh`: pass.

### v12 promotion gates

Before any online claim:

- natural set minADE <= 12 m;
- branch, neutral and priority minADE <= 15 m;
- candidate-budget supervision coverage > 0;
- pair witness AUPRC >= 0.60 (paper target >= 0.70);
- BCOT false-safe AUPRC >= 0.65 (paper target >= 0.75);
- direct conflict-conditioned root-transport AUPRC >= 0.65 (paper target >= 0.75);
- accepted NCF recall >= 0.30 (paper target >= 0.50);
- accepted candidate rate >= 0.10 (paper target >= 0.20);
- fallback <= 0.25 during development and <= conventional + 0.03 for paper;
- selected false-safe reduction >= 8 percentage points during development and
  >= 25% relative for paper;
- EP non-inferior to conventional at the paired scenario level.

### Required ablations for the paper

- RIOT/BCOT full model;
- legacy pairmax aggregation;
- response-bank-only recovery (disable direct root recovery);
- generic candidate certificate only;
- no natural pretraining/freeze;
- no observational, neutral or priority-preserving branch;
- no option preservation;
- soft burden cost only;
- oracle natural roots and oracle response roots as upper bounds.

### Additional prohibited shortcuts

- Do not run online evaluation when natural-basis or mechanism gates fail.
- Do not select a `least_violation` BCOT budget for a paper table; that status is
  diagnostic only.
- Do not call RIOT effective unless direct root recovery beats the response-bank
  auxiliary and pairmax ablations under paired evaluation.
- Do not claim reactive multi-agent evaluation while
  `actual_non_ego_policy=logged_replay`.


### v12 release hardening and direct mechanism observability

15. Learned-offline evaluation now requires compact response/root labels and reports
    `RootTransport/LowSafeExist_AUPRC`,
    `RootTransport/ConflictConditioned_AUPRC`,
    `RootTransport/ConflictConditioned_Recall@0.5`, the legacy auxiliary
    response-bank AUPRC, and natural-root assignment minADE.  This prevents a
    candidate-level BCOT score from being mistaken for proof that root-indexed
    option transport was learned.
16. The mechanism verifier now has an independent direct root-transport AUPRC
    gate.  The v12 driver uses 0.65 for development and blocks Waymax probe/full
    runs unless the saved mechanism report has `pass=true`.
17. Witness/planner supervision preflight now probes 32 evenly spaced cache items
    instead of the first eight, reducing shard-order false negatives for rare
    false-safe/NCF labels.
18. External repaired natural checkpoints must provide `NATURAL_HISTORY`; the
    natural-basis gate can no longer accidentally inspect an unrelated/missing
    local history file.
19. `planner_eval` loads only compact response/root targets, never dense response
    trajectories/components, so direct mechanism evaluation remains memory-bounded.

Validation after hardening:

- `pytest -q`: **70 passed**;
- `python -m compileall cowp tests`: pass;
- `bash -n run_cowp_v12_dual_gpu.sh`: pass.

## v13-TKN — Temporal-Kinematic Natural Basis and Protocol-Safe Closed Loop

### Triggering evidence from `cowp_v12_riot_probe100_seed2026`

The supplied v12 result directory contains only copied configs and
`checkpoints/natural/history_natural.json`; it contains no natural checkpoint,
transport/planner history or checkpoint, learned-offline report, probe result, or
Waymax result.  The run therefore stopped before RIOT was trained or evaluated.

The best recorded natural-basis row is epoch 7:

- validation set minADE: **41.2888 m**;
- branch minADE: **42.0075 m**;
- observational / neutral / priority minADE: **43.6674 / 40.7546 / 39.9407 m**;
- neutral consistency: **53.9360 m**;
- source CE: **1.0215**, improving only **0.00185** over the run;
- priority BCE: **0.6673**, improving only **0.00154**;
- composite checkpoint score: **42.6445**.

A v13 recheck fails every registered v12 natural gate.  This is not evidence
against RIOT itself, because RIOT never received an identifiable natural root
basis and no transport/planner/online result exists.

### Root causes

1. The old natural trajectory head projected one agent token directly to all
   `M x T x 7` values.  Its mode embedding did not participate in trajectory
   decoding and there was no explicit time representation or motion prior.
2. Natural alternatives are unordered, but source and priority semantics were
   supervised mainly as aggregate distributions rather than on the trajectory
   mode matched to each ground-truth root.  Root identity was therefore weakly
   identifiable.
3. Neutral consistency used source-neutral probability without the mode mixture
   probability, so even negligible modes contributed equally; the corresponding
   validation error stayed near 54 m.
4. The v12 gate had no dataset/oracle diagnostic and no short-horizon metrics,
   making it impossible to distinguish coordinate/track misalignment from a bad
   decoder using the result package alone.
5. Online `ClosedLoopPredFSR/CBS/OPR` are predictions of the same network that
   selects the candidate.  They are health diagnostics, not counterfactual
   ground truth and cannot independently validate the paper mechanism.
6. The real Waymax evaluator controls only the SDC.  Non-ego agents follow log
   playback.  The previous config described this correctly but the field was not
   enforced or emitted in the result JSON.
7. Replanning had no plan-stitching/hysteresis term, allowing high-frequency
   candidate switching even when every individual lattice trajectory was valid.
8. `metrics_from_labels` had a latent missing-field crash caused by an undefined
   `valid` variable in fallback burden/OPR allocation.

### Implemented changes

1. Added `temporal_kinematic` natural decoding:
   - constant-velocity anchor baseline;
   - explicit mode and time embeddings connected to trajectory generation;
   - bounded cumulative position/yaw residuals and bounded velocity/size offsets;
   - zero-initialized residual head, so a new model starts exactly at the
     kinematic baseline;
   - `legacy_linear` retained for a controlled ablation.
2. Added matched-mode semantic supervision: nearest-trajectory matching now binds
   source CE and priority BCE to the recovered natural root.
3. Corrected neutral consistency weighting to use both mixture probability and
   neutral-source probability.
4. Added validation natural minADE at 1 s, 3 s, 5 s, and 8 s.
5. Hardened `32_gate_natural_basis.py` with semantic-learning checks and optional
   kinematic-oracle comparison.
6. Added `33_diagnose_cache_alignment.py` for raw/transport tensor equality,
   critical track-to-input mapping, current/future coordinate consistency,
   transport-root ranges, and Waymax outcome/log-divergence coverage.
7. Added `34_diagnose_natural_oracles.py`, a 15-trajectory acceleration/yaw-rate
   bank that reports source-stratified 1/3/5/8 s oracle minADE.
8. Added online plan-continuity risk after the hard feasibility/frontier filter.
   It can reorder feasible candidates but cannot make a rejected candidate
   feasible.  The same regularizer is applied to internal baselines.
9. Waymax outputs now explicitly include:
   - `actual_non_ego_policy=logged_replay`;
   - `reactive_mixture_implemented=false`;
   - `mechanism_ground_truth_available_online=false`;
   - proxy-only markers for online model-predicted mechanism diagnostics.
   The evaluator raises an error if a config falsely requests a reactive non-ego
   policy without an implemented actor wrapper.
10. Fixed the missing optional witness-array crash in label-only metrics.
11. Added `STOP_AFTER_STAGE=natural|transport|planner|offline|probe` to make each
    promotion gate independently executable.  A natural-only run no longer
    requires transport caches unless transport diagnostics/later stages are
    requested.
12. Added v13 configs and `run_cowp_v13_dual_gpu.sh`.

### What is and is not validated

Validated locally without WOMD/Waymax data:

- YAML parsing;
- Python compilation;
- driver shell syntax;
- temporal decoder kinematic initialization;
- proxy protocol markers and missing-field metric fallback;
- full test suite: **73 passed**.

Not yet validated, because the uploaded package does not include the actual
raw/transport caches, the initialization checkpoint, or a Waymax installation:

- real cache coordinate/track alignment;
- v13 natural-basis convergence;
- RIOT direct root-transport AUPRC;
- planner selection gains;
- logged-replay Waymax metrics;
- reactive-agent counterfactual burden reduction.

### Promotion policy

- Do not continue beyond natural training unless the v13 natural gate passes.
- Do not run Waymax unless learned-offline mechanism verification passes.
- Do not use sparse attached candidate outcomes as the sole planner objective or
  as a full validation result.
- Keep log-divergence loss/reporting disabled until finite label coverage is
  measured; the supplied cache audit reports zero finite log-divergence labels.
- Do not call logged-replay Waymax evaluation reactive or use model-predicted
  online FSR/CBS/OPR as causal ground truth.
- A paper-ready result requires paired full-validation comparisons, confidence
  intervals, and a separately labelled reactive-agent protocol.
13. Keep the sparse Waymax outcome head as auxiliary supervision, but disable it
    in the primary v13 selector by default (`candidate_selection_outcome_weight=0`,
    `candidate_outcome_risk_mix=0`, online penalty `0`, threshold `1.10`).  It must
    be re-enabled only as a registered ablation after checkpoint-selected
    validation replay coverage is reported.

## v14-TNOB — Typed Natural Option Basis, Exact Anchor Preflight, and Fast Diagnostics

### Triggering evidence from `cowp_v13_temporal_riot_seed2026`

The v13 natural gate failed at epoch 29:

- validation set minADE: **35.8454 m**;
- 1 s / 3 s / 5 s minADE: **34.5138 / 34.7494 / 35.1241 m**;
- branch minADE: **37.7621 m**;
- OBS / NEU / PRIO minADE: **40.9817 / 36.1811 / 32.9040 m**;
- source CE: **0.9151**, improvement only **0.00717**;
- priority BCE: **0.3121**, the only semantic gate that passed;
- neutral consistency: **37.6742 m**, improvement only **0.0491 m**.

The label-side kinematic oracle on 2,000 validation scenes achieved
**0.283 / 0.608 / 1.204 / 2.466 m** at 1/3/5/8 s.  The nearly constant
~35 m model error from 1 s through 8 s is therefore inconsistent with ordinary
long-horizon forecast divergence and strongly suggests a model-facing critical
row/current-state anchor/frame mismatch.  The exact subcase cannot be proven
without the server caches; v14 adds a hard exact-path preflight rather than
silently guessing.

Independently of that suspected data-path issue, v13 had a confirmed structural
failure: all 24 roots began as the same constant-velocity curve, while global
nearest-neighbour matching allowed OBS, NEU and PRIO targets to compete for any
mode.  This makes semantic root identity non-identifiable and invalidates later
same-root transport even if aggregate displacement improves.

### Implemented changes

1. Added **Typed Natural Option Basis (TNOB)** in
   `cowp/models/natural_decoder.py`:
   - fixed 8/8/8 OBS/NEU/PRIO mode identities for the default 24 modes;
   - distinct analytic acceleration/yaw-rate/speed-offset prototypes;
   - zero-initialized, gated and physically bounded temporal residuals;
   - explicit `mode_source`, `base_traj` and `residual` outputs;
   - legacy temporal and linear decoders retained for ablation.
2. Replaced cross-source global matching with source-restricted matching in
   `cowp/models/losses.py`.  Trajectory, mixture, branch, short-horizon,
   source and priority supervision now share the same typed assignment.
3. Added untyped geometric coverage as a diagnostic and a typed-vs-untyped gap
   gate, preventing semantic typing from hiding lost trajectory coverage.
4. Added analytic-prior preservation (`base_deviation`) and residual magnitude
   regularization, plus explicit 1/3/5 s loss terms.
5. Added `--eval-before-train`: epoch -1 analytic TNOB is evaluated and can be
   retained as the best checkpoint if learning degrades it.
6. Added `--reset-checkpoint-prefix`; the v14 driver resets `natural_decoder`
   when loading an old planner checkpoint while retaining the graph backbone.
7. Replaced full-stage graph freezing with a configurable two-epoch natural
   warmup followed by low-LR joint adaptation; added gradient clipping.
8. Added `35_diagnose_model_anchor.py`, which follows the exact production path:
   `TorchCOWPDataset -> _agent_history_from_batch -> input_index ->
   _safe_critical_indices -> _critical_anchor7 -> typed_kinematic_basis`.
   It hard-fails on excessive unmapped critical agents, first-step anchor error,
   or 1/8 s typed-basis error.
9. Rewrote `33_diagnose_cache_alignment.py` for selective NPZ member reads,
   symlink/samefile shortcuts, sampled hashing and threaded I/O.
10. Rewrote `34_diagnose_natural_oracles.py` to vectorize root/horizon distance
    calculation and support threaded loading.
11. Hardened the natural gate with absolute-or-improvement semantic criteria and
    the typed/untyped coverage check.
12. Added v14 configs, driver, execution guide and typed-basis unit tests.

### Promotion gates

Training is blocked unless exact model-facing preflight satisfies:

- critical unmapped/invisible rate <= 2%;
- first-step GT versus model-CV-anchor p90 <= 5 m;
- typed basis minADE@1s <= 3 m;
- typed basis minADE@8s <= 8.5 m.

Natural promotion additionally requires:

- set minADE <= min(12 m, oracle@8s + 6 m);
- branch/NEU/PRIO minADE <= 15 m;
- source CE <= 0.30 or registered learning improvement;
- priority BCE <= 0.45 or registered learning improvement;
- neutral consistency <= 10 m or registered learning improvement;
- typed minus untyped minADE <= 4 m.

No transport, planner or Waymax result may be promoted when these gates fail.

### Validation and limitations

Validated locally:

- full test suite: **76 passed**;
- Python compilation: pass;
- `bash -n run_cowp_v14_dual_gpu.sh`: pass;
- typed mode identities, non-collapsed analytic endpoints and source-restricted
  matching unit tests: pass.

Not validated locally because the uploaded artifacts omit the actual raw and
transport-v9 caches, initialization checkpoint, Waymax runtime and standalone
`cache_sufficiency_full.json`:

- which concrete mapping/frame bug caused the v13 ~35 m translation;
- v14 A30 convergence and gate pass;
- direct root-transport mechanism gains;
- full logged-replay or reactive-agent closed-loop performance;
- SOTA status.

### Additional prohibited shortcuts

- Do not weaken the anchor preflight or natural gate to continue a run.
- Do not restore global cross-source matching in the primary method; keep it only
  as a registered ablation.
- Do not claim TNOB alone proves non-coercive feasibility; it establishes root
  identifiability, while transport and independent reactive evidence prove the
  mechanism.
- Do not claim SOTA from a 100-scene probe or from sparse attached outcomes.
13. Removed stale top-level duplicate Python trees that were not included by the
    `cowp*` package configuration.  The authoritative implementation is now
    unambiguously under `cowp/`, preventing fixes from being applied to inert
    copies.

## v15-CNOB — Causal Natural Option Basis, Protocol Integrity, and OBS Decontamination

### Triggering evidence from `cowp_v14_typed_natural_seed2026`

The uploaded v14 run did not produce `natural_basis_gate.json`, but this was not
an ordinary metric failure. `NEXT_RUN_COMMANDS_V14_CN.txt` mixed executable shell
commands with Chinese prose and was invoked through `bash`; the prose was parsed
as commands. The driver then retained an existing natural checkpoint although
`history_natural.json` was absent, so the hard gate could not be reconstructed by
the run itself.

The validation rows in `logs/train_natural_ddp.log` were recovered into a proper
history file. Under the original v14 thresholds, the best epoch (15) passes:

- typed set minADE@8 s: **1.8974 m**;
- minADE@1/3/5 s: **0.2289 / 0.4480 / 0.8951 m**;
- branch minADE: **2.9117 m**;
- OBS / NEU / PRIO minADE: **4.6060 / 1.1351 / 1.2998 m**;
- neutral consistency: **1.8647 m**;
- priority BCE: **0.3495**.

However, v14 is not promoted by the stricter v15 geometric gate. It fails on the
OBS branch (**4.6060 m > 4.0 m**) and branch spread
(**4.6060 - 1.1351 = 3.4708 m > 3.0 m**). Near-zero source CE is no longer
counted as evidence of learning because typed root identities are structurally
hard-coded.

The epoch trace also shows that the learned temporal residual was nearly inert:
aggregate quality jumped when the analytic typed basis/graph path became active,
then remained almost flat; final base deviation was approximately 0.017 m and
residual L2 approximately 8e-4. Thus v14 success was dominated by hand-designed
kinematic prototypes rather than learned scene-conditioned natural behavior.

### Confirmed engineering defects

1. `COWPModel._agent_history_from_batch` could reconstruct encoder input from
   `cowp/natural/traj` when real current/history tensors were missing. This is a
   direct future-label leak.
2. the online policy could silently fall back to Waymax `log_trajectory`, which
   contains privileged future states;
3. missing `state/is_sdc` could silently make row zero the ego, invalidating
   ego-centric transforms and all downstream critical-agent indexing;
4. observational perturbations shifted future positions without consistently
   repairing velocity and heading, producing side-slip and first-step
   discontinuities;
5. natural alternatives declared map compliance without performing a map check;
6. the logged OBS future could already contain ego-induced yielding, so it was
   not a clean sample of behavior under the absence of ego pressure;
7. a label-space candidate-safety complement was exported as `CR`, although it
   was not a simulator-measured closed-loop collision/offroad rate;
8. logged-replay non-ego agents were described by the paper as a reactive
   mixture, although the current evaluator does not implement that mixture;
9. checkpoint-only skip logic could preserve an incomplete natural stage;
10. documentation and executable shell were mixed in one file.

### Implemented changes

1. Added a strict causal input contract:
   - future-label reconstruction is disabled by default;
   - reported runs require an explicit SDC marker;
   - absent real history/current tensors or SDC identity hard-fail.
2. Added a strict Waymax future-access contract:
   - main policies use only simulated/current/history state;
   - `log_trajectory` is inaccessible except in an explicitly named oracle
     ablation;
   - causal constant-velocity non-ego prediction is used by the model-facing
     online wrapper when no learned reactive predictor is available.
3. Reworked observational trajectory perturbation:
   - every alternative starts continuously from the current state;
   - lateral displacement uses a zero-origin smooth transition;
   - heading and velocity are recomputed from the transformed path;
   - invalid/non-finite motion is rejected.
4. Added map-aware natural-option filtering using the available road/lane point
   cloud, with type-aware distance thresholds and explicit verification fields.
5. Added observational decontamination:
   - estimate whether the logged agent decelerates/loses progress near ego;
   - compare logged clearance with an ego-neutral continuation;
   - produce an `obs_contamination` score;
   - downweight or reject highly pressure-contaminated OBS alternatives.
6. Introduced the **Causal Natural Option Basis (CNOB)** decoder profile:
   - retain source-stable OBS/NEU/PRIO roots needed by same-root transport;
   - allocate more bounded residual capacity to OBS, the empirically weak branch;
   - preserve stronger analytic priors for NEU/PRIO;
   - treat source identity as structure rather than an artificial source-CE win.
7. Added source-specific prior-deviation losses and diagnostics, with a lower OBS
   regularization coefficient and stronger NEU/PRIO preservation.
8. Hardened the natural gate with absolute OBS quality and branch-spread checks.
9. Split metric namespaces:
   - closed-loop CR/offroad are accepted only from Waymax standard metrics;
   - label-space safety is named `OfflineConventionalUnsafeRate`;
   - `CR_proxy_deprecated` is retained only for old result readers.
10. Added `36_audit_causal_protocol.py`, checking leakage, SDC identity, metric
    provenance, map filtering, OBS decontamination, reactive-protocol honesty,
    mapping completeness, root-index range, and missing log-divergence policy.
11. Added pure executable scripts:
    - `prepare_cowp_v15_data.sh` rebuilds labels/caches/outcomes/transport overlay;
    - `NEXT_RUN_COMMANDS_V15_CN.sh` runs tests, data preparation, and the v15
      training/evaluation driver without prose being parsed as shell;
    - `run_cowp_v15_dual_gpu.sh` treats checkpoint+history as one atomic natural
      artifact and retrains when either is incomplete.
12. Rebuilt Pareto and pairmax ablation configs on top of the same v15 causal
    natural-label settings, so ablations do not reintroduce old label defects.
13. Added five causal-integrity regression tests. The local suite now reports
    **81 passed**.

### v15 promotion gates

Natural-stage promotion requires all of the following:

- typed set minADE@8 s <= min(8.5 m, label oracle + 6 m);
- branch-weighted minADE <= 3.0 m;
- OBS minADE <= 4.0 m;
- max(OBS, NEU, PRIO) - min(OBS, NEU, PRIO) <= 3.0 m;
- NEU and PRIO minADE <= 2.0 m;
- minADE@1 s <= 3.0 m and minADE@3 s <= 5.0 m;
- neutral consistency <= 3.0 m;
- priority BCE <= 0.45;
- typed-untyped gap <= 3.0 m.

For paper-facing runs, the preferred target is stricter: OBS <= 3.5 m,
branch spread <= 2.0 m, and typed set minADE <= 1.5--1.7 m, while preserving
NEU/PRIO quality.

### What is and is not validated

Validated locally without WOMD/Waymax runtime:

- Python compilation;
- YAML parsing and driver shell syntax;
- 81 unit/regression tests;
- the static/report-backed causal protocol audit;
- reconstruction of the v14 natural history and original gate;
- expected rejection of v14 by the stricter v15 OBS/branch-spread gate.

Not validated in the supplied environment:

- v15 label/cache regeneration on the full WOMD data;
- v15 natural convergence and actual OBS improvement;
- transport/planner retraining;
- full-validation Waymax closed-loop CR/offroad/progress;
- reactive non-ego evaluation;
- SOTA status.

### Additional prohibited shortcuts

- Do not enable `allow_label_only_state_fallback` in a reported run.
- Do not disable `require_explicit_sdc_index` to bypass malformed caches.
- Do not use `log_trajectory` outside a clearly labelled oracle diagnostic.
- Do not report `OfflineConventionalUnsafeRate` or `CR_proxy_deprecated` as
  closed-loop collision rate.
- Do not call logged replay a reactive-agent experiment.
- Do not claim the hard-coded source identity or its near-zero CE as novelty.
- Do not continue beyond the natural gate when OBS or branch spread fails.
- Do not claim SOTA before full-validation, multi-seed, paired confidence-
  interval comparisons under both logged-replay and independent reactive-agent
  protocols.

## v15.1 — audited reuse of v9 transport caches (2026-07-23)

1. Reclassified data compatibility into two explicit protocols:
   - `v15`: regenerated causal-label protocol with OBS decontamination and map filtering;
   - `v9_reuse`: v15 model/engineering stack trained on the existing v9 labels.
2. Changed `NEXT_RUN_COMMANDS_V15_CN.sh` to reuse the existing raw Waymax caches and
   `transport_v9` overlays by default. It no longer runs index, label generation,
   tensor-cache construction, Waymax replay, outcome attachment, or transport augmentation.
3. Preserved the former full rebuild route as
   `NEXT_RUN_COMMANDS_V15_REBUILD_FULL_CN.sh`.
4. Added `38_gate_cache_reuse.py`, which checks current server-side file counts,
   raw/overlay alignment, required training fields, explicit SDC identity, critical-agent
   mapping, response-root ranges, split overlap, v15 label materialization, and logdiv state.
5. Added `CHECK_V9_CACHE_ONLY.sh` to run the complete raw-cache sufficiency scan and
   the independent v9 overlay gate without launching training.
6. Hardened `36_audit_causal_protocol.py` so `engineering_pass` is separated from
   `full_v15_label_protocol_pass`. A v9-reuse run may be engineering-valid while
   correctly refusing to claim that v15 causal labels were materialized.
7. Added a data-protocol manifest to every v15 run. Paper-facing result aggregation
   must not merge `v9_reuse` and full `v15` experiments as if they used identical labels.
8. Disabled silent use of missing Waymax log-divergence supervision. Existing safety
   replay has zero finite logdiv targets; collision/offroad may remain an auxiliary loss,
   while final closed-loop metrics must come from online Waymax.
9. Added `tests/test_v15_v9_cache_reuse.py`. Full local suite: **82 passed**.
10. Documented the count discrepancy between the older full cache report (14,640 train)
    and the later v14 alignment result (20,440 train). The new default script therefore
    audits the current server directory before training and defaults to a 20,000-scene
    minimum rather than trusting stale reports.

## v16 — CNOB dynamics and evidence-gated attribution (2026-07-23)

### Triggering evidence

The uploaded v15/v9-reuse run stopped before training in the mandatory model-anchor
preflight. `35_diagnose_model_anchor.py` maintained its own decoder string whitelist
and rejected `typed_causal_residual`, even though `NaturalDecoder` accepted that
name. Consequently the run produced no natural checkpoint/history, no planner or
selector result, and no online Waymax result. None of the claimed v15 model changes
were empirically evaluated by that run.

### Engineering fixes

1. Centralized typed decoder identity in `NaturalDecoder.uses_typed_basis` and
   `uses_dynamic_residual`; preflight and protocol audit now query the model rather
   than maintaining independent string lists.
2. Corrected cache-sufficiency enumeration so hidden sampler/metadata `.npz` files
   are not counted as scenarios. This removes the false `20441` versus `20440`
   discrepancy and the associated one-scene missing-label warning.
3. Fixed candidate-certificate fallback semantics: when
   `candidate_cert_allow_hybrid_fallback=false`, no generic certificate is silently
   mixed into the set-transport score.
4. Added automatic learned-natural diagnostics and a hard effectiveness gate before
   transport/planner training.
5. Added full Waymax delta summaries against both conventional safety and planner-
   score-only baselines.

### Algorithm changes: Causal Natural Option Basis dynamics decoder

1. Replaced the paper-facing `typed_causal_residual` alias with
   `typed_causal_dynamics` / `cnob_dynamics`.
2. The learned branch predicts bounded local longitudinal/lateral acceleration,
   jerk, and yaw-rate corrections around the typed OBS/NEU/PRIO basis.
3. Position and velocity are obtained by integration; heading is derived from the
   integrated velocity when moving; box dimensions are invariant. Independent,
   mutually inconsistent position/yaw/velocity/size residual heads are prohibited.
4. Zero initialization exactly reproduces the analytic typed basis, preserving a
   safe and interpretable initialization.
5. OBS receives configurable additional control capacity. The
   `natural_obs_capacity_scale=0` ablation gives OBS the same control bounds as NEU
   while retaining PRIO protection, enabling a controlled capacity claim.

### Natural-loss and evidence changes

1. Source-restricted branch minADE now uses `cowp/natural/weight`; v15 contamination
   downweights are therefore respected when a true v15 dataset is used.
2. Added direct OBS-improvement shortfall loss relative to the exact analytic basis.
3. Added NEU/PRIO preservation losses, finite-difference velocity consistency,
   velocity-heading consistency, control smoothness, and source-specific mode-usage
   entropy.
4. Added `39_diagnose_learned_natural.py`, which measures learned versus analytic
   minADE by source and horizon, residual magnitude, controls, physical consistency,
   and effective mode usage.
5. Added `40_gate_natural_effectiveness.py`. A natural checkpoint must now improve
   the analytic basis, improve OBS, preserve NEU/PRIO, remain physical, use multiple
   modes, and keep residuals bounded. Passing only the old absolute gate is no longer
   sufficient.
6. Added controlled component attribution:
   - `train_cowp_v16_no_effectiveness_loss.yaml` removes the new loss bundle;
   - `model_cowp_v16_no_obs_capacity.yaml` removes only the OBS capacity boost;
   - `RUN_NATURAL_ABLATIONS_V16_CN.sh` trains both;
   - `41_compare_natural_ablations.py` hard-gates the new-loss and OBS-capacity claims.

### Planner/selector changes

1. Planner checkpoint selection now prioritizes the claimed set-transport mechanism,
   candidate budget, and same-root recovery rather than allowing the generic
   candidate classifier to dominate.
2. Sparse attached Waymax collision/offroad labels remain auxiliary only; their loss
   weights are reduced and `outcome_logdiv=0` remains mandatory because finite
   log-divergence coverage is zero.
3. Online evaluation remains honest logged-replay non-ego Waymax. It can establish
   SDC CR/offroad/progress effects, but it is not a reactive-agent burden experiment.

### Data decision

- Reuse `tensor_cache_*_waymax_transport_v9` for the next v16 model/loss/capacity,
  planner, selector, and online Waymax experiment.
- Do not claim v15 OBS decontamination or map-filtered label generation from v9 data.
- A true v15 dataset is required only after the v16 model passes, to validate the
  revised natural roots/weights and the paper's causal-label contribution. Prefer a
  targeted OBS/interaction-heavy pilot before a full rebuild.

### v16 promotion order

1. exact-path model-anchor and causal-protocol preflight;
2. legacy absolute natural gate;
3. learned-versus-analytic natural effectiveness gate;
4. controlled natural component attribution gate;
5. transport/planner mechanism verification;
6. paired online Waymax probe;
7. full-validation multi-seed Waymax evaluation;
8. true-v15-label pilot/full rebuild and an independent reactive-agent protocol for
   the final causal burden claim.

### Local validation

- Python compilation: pass;
- executable shell syntax: pass;
- causal engineering audit: pass;
- full v15 label protocol on v9 caches: intentionally false;
- unit/regression suite: **90 passed** after v16 component-ablation coverage.

### Prohibited claims

- Do not call the uploaded failed v15 run evidence that the decoder, new losses,
  planner, selector, or Waymax metrics improved.
- Do not attribute a main-model gain to the new loss or OBS capacity without the two
  controlled natural ablations.
- Do not call v9-reuse a true v15 causal-label experiment.
- Do not call logged-replay non-ego Waymax a reactive-agent evaluation.
- Do not claim SOTA before full-validation, multi-seed paired results and confidence
  intervals are available.

## v16.1 — natural-loss hotfix, provenance isolation, and fast diagnostics (2026-07-24)

### Triggering evidence

The v16 run passed cache alignment, causal engineering audit, natural oracle, and the
exact model-facing anchor preflight. It then failed before optimization at validation
epoch `-1`, batch `0`, inside `_natural_mode_usage_loss`. The real tensor shapes were
approximately `[B=5, A=6, M=24, R=24]`. The implementation used
`psrc[..., 0]`, which removed the mode axis and created a `[B,A]` mask; broadcasting
that mask against `[B,A,M,R]` failed with `A=6` versus `M=24`.

Because the failure occurred before the first validation batch completed, the run
contains no trained natural checkpoint/history and supplies no evidence about decoder,
new-loss, OBS-capacity, planner, selector, or online Waymax gains.

### Engineering fixes

1. Fixed source-specific mode-usage masking to retain `[B,A,M]` and aggregate only
   eligible modes. The implementation also supports an explicitly expanded
   batch-dependent `mode_source` tensor rather than assuming a one-dimensional buffer.
2. Added real-dimension forward/loss/backward regressions with six critical agents,
   24 typed modes, and 24 natural roots. This exact shape would have failed in v16.
3. Added strict experiment provenance (`42_write_run_provenance.py`). Reusing an
   `OUT_ROOT` with a different hash of critical model/loss/evaluation code or copied
   configs now fails instead of silently mixing stale checkpoints and reports. Candidate
   configs are hashed under stable logical names and checked before canonical config
   files are overwritten, so a rejected reuse attempt cannot corrupt the old record.
4. Changed the top-level execution script to detach the entire workflow by default,
   not only the inner dual-GPU driver. Global output is written to
   `logs/driver.nohup.log`, the PID to `logs/driver.pid`, and each stage retains its
   own log file.
5. Added `CHECK_RUN_STATUS_V16_1.sh` for PID and recent-log inspection.
6. Added stage-specific DataLoader settings: natural training defaults to 8 workers
   and prefetch 2; transport/planner retain conservative 4/1 defaults because their
   dense tensors previously caused host/pinned-memory failures.
7. Added `DIAG_PROFILE=fast|full`. The default fast preflight samples 2,048 train and
   1,024 validation transport scenes, 1,024 alignment/oracle/anchor scenes, while the
   paper-facing full audit remains available separately.
8. Reworked transport diagnostics with threaded NPZ reads, exact streaming means and
   MAEs, and a bounded 10,000-bin histogram for root-recovery quantiles. This removes
   Python lists containing millions of scalar values; the quantile resolution is
   explicitly reported (default bin width `1e-4`).
9. Cache sufficiency and v9 reuse reports are no longer recomputed when valid reports
   already exist. New runs use sampled cache sufficiency by default; `FULL_CACHE_AUDIT=1`
   retains the complete scan.
10. Added `RUN_FULL_DATA_AUDIT_V16_1_CN.sh` for a background full data audit without
    launching model training.

### Algorithm decision

No additional model or loss change is promoted in v16.1. The v16 dynamics decoder,
effectiveness losses, and OBS capacity must first complete training and controlled
ablations. A code crash is not evidence that these components are ineffective, and
changing them again before obtaining metrics would destroy attribution.

### Data decision

The v15 causal-label dataset is not architecture-locked to the CNOB decoder. It changes
the semantic targets and weights (especially OBS contamination/map filtering) and the
downstream response/witness labels. If CNOB fails on v9 labels, the v15 dataset is not
automatically useless. Use a small interaction-heavy v15 pilot and a model/data matrix:
CNOB+v9, CNOB+v15-pilot, simpler typed residual+v15-pilot. Rebuild the full v15 dataset
only when the matrix shows a reproducible label contribution.

### Local validation

- Python compilation: pass;
- shell syntax: pass;
- realistic natural forward/loss/backward regression: pass;
- threaded transport diagnostic regression: pass;
- strict provenance regression: pass;
- full unit/regression suite: **96 passed**.

### Prohibited claims

- Do not report any v16 algorithm improvement from the failed run.
- Do not label preflight analytic-basis metrics as learned decoder metrics.
- Do not attribute later main-model gains to the new loss or OBS capacity until the
  controlled ablations pass.
- Do not reuse an experiment root after code/config changes by bypassing provenance.
- Do not use fast diagnostics as the final paper data audit; run the full audit once
  for the frozen code/data release.

## v16.2 — old-PyTorch compatibility and end-to-end pipeline completion guards (2026-07-24)

### Triggering evidence

Both `driver.nohup.log` and `full_driver.nohup.log` stopped during the startup pytest
suite. Five tests failed in `_natural_mode_usage_loss` because the server PyTorch build
does not support `Tensor.any(dim=(0, 1))`. This was a compatibility regression introduced
while fixing the earlier `[B,A]` versus `[B,A,M]` broadcast bug. No model training or
evaluation started, so these logs contain no algorithm or closed-loop evidence.

A second static pipeline defect was found: `NEXT_RUN_COMMANDS_V16_1_FULL_CN.sh` did not
set `RUN_FULL=1`. Even after training and probe success, that script would not have
produced the nominal full Waymax results.

### Engineering fixes

1. Replaced the unsupported tuple-dimension boolean reduction with the equivalent
   old/new-PyTorch-compatible chain `mode_mask.any(dim=0).any(dim=0)`.
2. Added a source-level regression that forbids reintroducing `.any(dim=(...))` in the
   natural loss and retained the realistic `B=2, A=6, M=24, R=24` forward/loss/backward
   regression.
3. Added `43_pipeline_preflight.py`, which validates YAML configs, imports every critical
   train/eval/gate module, records Python/PyTorch/CUDA/JAX/TensorFlow/Waymax availability,
   checks `torchrun`, and executes a realistic natural forward/loss/backward before GPU
   hours are consumed.
4. Added `44_validate_pipeline_outputs.py`. A run cannot print `complete` unless required
   checkpoints and JSON reports exist; probe/full validation additionally requires
   usable CR, offroad, and EP/progress values.
5. Fixed the v16.2 full wrapper so `RUN_FULL=1` and Waymax dependency preflight are enabled
   by default.
6. Added robust `wait_all` handling for parallel diagnostics, offline evaluation, Waymax
   waves, and full shards. The driver waits for every child and reports failure instead
   of exiting on the first failed child while leaving sibling processes running.
7. Full COWP shards are now resumable independently; valid shard JSON files are reused
   after an interrupted run.
8. Added explicit engineering-only quality-gate bypass support and a separate
   `NEXT_RUN_COMMANDS_V16_2_ENGINEERING_SMOKE_CN.sh`. This path exists only to exercise
   downstream planner/Waymax code with tiny epochs/scenario counts; bypassed outputs are
   marked and are prohibited as paper evidence. Strict scripts keep all gates enabled.
9. Added `Offroad` as a supported alias in planner-delta summaries, while retaining
   `OffroadRate`, so different Waymax metric adapters still produce the requested
   offroad comparison.
10. All v16.2 scripts use a fresh default output root to prevent provenance conflicts
    with failed v16/v16.1 runs.

### Algorithm decision

No decoder, loss, planner, selector, or label semantics were changed. This release is an
engineering-only repair so the next completed run remains attributable to the v16
algorithm rather than a new moving target.

### Local validation

- complete unit/regression suite: **100 passed**;
- Python compilation: pass;
- all v16.2 shell scripts: `bash -n` pass;
- all pipeline CLI modules: import/`--help` pass;
- realistic natural preflight: pass;
- completion-validator CR/offroad/EP fixture: pass.

### Prohibited claims

- Engineering-smoke closed-loop numbers are not paper results.
- A strict run that stops on a quality gate is an algorithmic result, not a pipeline
  failure, provided `pipeline_preflight.json` and tests pass.
- Do not call the pipeline complete unless `pipeline_completion_report.json` passes.

## v16.3 — numerical-integrity recovery, evidence-gated attribution, and root-wise NCF correction (2026-07-24)

### Triggering evidence from the uploaded v16.2 engineering smoke

This revision is based on `cowp_v16_2_engineering_smoke_v9labels_seed2026_ancdatafix` and must not be interpreted as a paper-result run. It used one epoch per stage, only 20 Waymax scenarios, `ALLOW_QUALITY_GATE_FAILURE=1`, `STOP_AFTER_STAGE=probe`, `RUN_FULL=0`, and the `v9_reuse` data protocol. The causal audit passed its engineering checks but explicitly reported `full_v15_label_protocol_pass=false` because v15 label tensors were not materialized.

The natural history contains an internally inconsistent pattern: epoch `-1` and epoch `0` have exactly identical validation metrics; `natural/residual_l2=0` and `natural/control_smoothness=0`; the learned-natural diagnostic reproduces the analytic basis exactly and reports zero gain; nevertheless `base_deviation` is about 34.29 m. The best checkpoint therefore remains the initialization checkpoint. Downstream transport/planner/selector results are mechanism probes on an unverified basis, not evidence for their algorithmic effectiveness.

### Root cause and engineering fix

The failure pattern is highly consistent with FP16 autocast overflow in the scene graph followed by `0 * Inf -> NaN` in the zero-initialized natural residual/control heads. The previous loss path could silently sanitize non-finite predictions through `nan_to_num`, while `GradScaler` could skip optimizer steps without exposing that fact. This yields finite-looking losses, zero residual/control statistics, and no actual learning.

Changes:

1. `cowp/models/cowp_model.py`
   - Natural decoder and integration now execute in an explicit FP32 precision island.
   - Graph features and anchor state are cast to FP32 before the zero-initialized residual/control heads.

2. `cowp/scripts/03_train.py`
   - Added `--amp-dtype {auto,bfloat16,float16}`; `auto` prefers BF16 on supported CUDA hardware.
   - Natural/representation stages automatically fall back to full FP32 when only FP16 is available.
   - Added recursive pre-loss NaN/Inf detection on model outputs. Non-finite predictions now fail fast and are never converted into zeros.
   - Added synchronized DDP non-finite checks, gradient-norm checks, optimizer-step counters, AMP-skip counters, and a hard failure when no optimizer step is executed or more than 2% of attempted AMP steps are skipped.
   - The GradScaler now follows the effective stage AMP policy rather than the raw CLI flag.

3. Launch/status scripts
   - Added `run_cowp_v16_3_dual_gpu.sh` with natural FP32 by default and downstream BF16-auto policy.
   - Added `NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh`, `NEXT_RUN_COMMANDS_V16_3_FULL_CN.sh`, `NEXT_RUN_COMMANDS_V16_3_ENGINEERING_SMOKE_CN.sh`, and `CHECK_RUN_STATUS_V16_3.sh`.
   - Status detection checks both historical and current locations of `QUALITY_GATES_BYPASSED.txt` and reports optimizer/AMP-skip evidence.

4. Controlled attribution
   - Added `RUN_NATURAL_ABLATIONS_V16_3_CN.sh`.
   - Main, `no_effectiveness_loss`, and `no_obs_capacity_boost` runs use fresh outputs and identical natural-stage precision. The full pipeline is blocked until both the natural-effectiveness gate and component-attribution gate pass.

### Status of the six requested claims after v16.2

- **v15/v16 decoder effective:** not established. The trained checkpoint did not deviate from initialization.
- **new loss effective:** not established. No valid learned main model and no controlled ablation were available.
- **OBS residual capacity effective:** not established. Residual/control outputs were exactly zero and no valid capacity ablation existed.
- **natural gate effective:** established only as an engineering safeguard. It correctly rejected an inert natural checkpoint and prevented it from being promoted as evidence.
- **planner effective:** not established. One epoch, invalid natural foundation, and mixed 20-scenario results cannot isolate planner contribution.
- **selector effective:** not established. BCOT improved some conventional quantities versus pair-max but was worse than Pareto on collision/progress, while predicted FSR remained about 0.92 and fallback remained 0.80.

### Theoretical correction retained in the revised manuscript

The old existential witness condition—there exists a high-burden safe response—is logically insufficient because emergency responses usually exist. The revised definition uses stable natural roots with probability mass, same-root response transport, conflict mass, retained low-burden root mass, and a conflict-conditioned tail burden. Option preservation is removed from the primitive burden to avoid circularity. Burden thresholds are calibrated on a disjoint split and frozen, preventing the certificate from learning to raise its own acceptance threshold. Logged-replay online runs are labeled as proxy/mechanism diagnostics; reactive-agent and human-audited protocols are mandatory for causal burden claims.

### Promotion rules

No downstream result may be used in the paper unless all of the following are true:

1. `train/runtime/optimizer_steps > 0`, AMP skip ratio <= 2%, and no non-finite prediction is observed.
2. Learned natural basis improves over the analytic basis on the preregistered overall and OBS metrics without unacceptable neutral/priority degradation, across at least three seeds.
3. New-loss and OBS-capacity one-factor ablations pass the attribution gate.
4. Transport/root recovery and BCOT calibration pass; no `least_violation` operating point is accepted as a result.
5. Closed-loop evaluation uses at least 1,000 interaction-heavy scenarios per seed, three seeds, paired bootstrap confidence intervals, and separately reports logged-replay and reactive-agent protocols.

### v16.3 local validation

- `python -m compileall -q cowp tests/test_v16_3_numeric_safety.py`: passed.
- `pytest -q`: **107 passed**.
- `CHECK_RUN_STATUS_V16_3.sh` on the uploaded v16.2 smoke correctly reports `ENGINEERING-ONLY`, all three failed/bypassed gates, missing v16.3 optimizer-step evidence, and `INTENTIONAL_PARTIAL_RUN` after probe.
- Revised TeX static consistency: balanced braces, no unresolved `ref/eqref`, no literal traceback, no stray Markdown fence, and no undeclared `mathbbm` command.

## v16.4 — integrated residual trust region, yaw-frame correction, and strict calibration promotion (2026-07-25)

### Triggering evidence from the uploaded v16.3 natural recovery

The strict v16.3 run is materially different from the earlier inert v16.2 smoke. The
natural optimizer executed normally (`optimizer_steps=2044` at epoch 0 and no AMP
skips), validation loss decreased, and the learned CNOB decoder improved the exact
analytic basis. On 2,000 validation scenes, source-restricted weighted 8-second error
improved from 1.8862 m to 1.1779 m overall. The OBS branch improved from 4.1038 m to
2.6402 m, while NEU and PRIO also retained positive gains. The absolute natural-basis
gate therefore passed.

The strict natural-effectiveness gate nevertheless failed for two localized physical
reasons:

1. velocity--heading consistency was 0.17497 rad, slightly above the registered
   0.15 rad threshold; and
2. the integrated residual endpoint had a 45.218 m p99, above the 25 m bound, with a
   highly skewed distribution (p50 0.588 m, p90 31.912 m).

The full pipeline stopped at this gate before transport/planner training. The uploaded
archive contains no separate natural-ablation output root, so it supplies no evidence
for or against the new-loss or OBS-capacity attribution claims. It also contains no
v16.3 planner or selector result.

### Root-cause analysis

1. The dynamic decoder formed the moving yaw residual as
   `velocity_yaw - base_velocity_yaw`, although the final trajectory is represented as
   `base_absolute_yaw + yaw_residual`. For low-speed prototypes, base velocity direction
   is not guaranteed to equal the stored absolute heading. This creates a frame mismatch
   and can double-count the heading offset. The correct residual is
   `velocity_yaw - base_absolute_yaw`.
2. Bounded instantaneous acceleration, jerk, and yaw-rate do not bound the displacement
   accumulated over eight seconds. A small subset of OBS modes can therefore use the
   residual as a second unconstrained trajectory decoder, undermining stable root
   identity despite improving minADE.
3. A soft loss evaluated only on a radially hard-projected endpoint has zero radial
   gradient outside the feasible ball. The interior loss must use the pre-projection
   integrated endpoint, while the projected trajectory is used for prediction and
   physical diagnostics.

### Algorithm and engineering changes

1. Corrected the yaw reference frame. Moving-state heading is now derived from the final
   absolute velocity relative to the base absolute yaw. Exact zero-residual initialization
   is preserved by using velocity heading only when the learned velocity correction is
   non-negligible.
2. Added source-conditioned integrated endpoint budgets: OBS 20 m, NEU 8 m, PRIO 6 m.
   The complete local acceleration/jerk sequence is scaled and re-integrated, preserving
   position--velocity consistency rather than clipping the final position.
3. Added a dimensionless soft interior loss on the **pre-projection** endpoint/budget
   ratio, with a default interior threshold of 0.75. This gives over-budget controls a
   radial gradient back toward the feasible interior.
4. Added diagnostics for projected endpoint, raw endpoint, raw budget ratio, per-source
   endpoint distributions, and raw boundary-saturation rate. The effectiveness gate now
   additionally rejects a model when more than 25% of valid modes are at or beyond 95%
   of their source budget.
5. Added a controlled `no_integrated_trust_region` ablation. Attribution requires at
   least 5 m p99 tail reduction without more than 0.15 m OBS regression.
6. Tightened mechanism promotion: a BCOT operating point with calibration status
   `least_violation` is no longer accepted. Only a genuinely constraint-satisfying
   calibration may pass the mechanism gate.
7. Added v16.4 recovery/full/status/ablation launchers and a revised manuscript that
   states the model-based intervention limitation, the integrated trust region, the
   pre-projection interior loss, and the impossibility of replacing a feasibility
   certificate by an arbitrary finite soft burden penalty.

### Status of the six requested claims after v16.3

- **v15/v16 decoder:** partially supported for the trained v16 CNOB natural module. It
  clearly improves the analytic basis, but the architecture claim is not fully promoted
  until the v16.4 physical gate and architecture-level ablations pass.
- **new loss:** not yet identifiable; the uploaded ablation result root is absent.
- **OBS residual capacity:** not yet identifiable for the same reason.
- **natural gate:** validated as an effective safeguard. It accepted useful prediction
  gains but correctly blocked a physically invalid long-tail solution.
- **planner:** not evaluated in this strict run because the pipeline stopped before it.
- **selector:** not evaluated in this strict run for the same reason.

### Local validation

- Full Python unit/regression suite: **113 passed**.
- Added stopped-agent/rotated-anchor yaw regression.
- Added hard endpoint-budget and finite-gradient regression.
- Added regression proving the soft trust loss receives a nonzero gradient from the
  pre-projection endpoint.
- Python compilation and all v16.4 shell syntax checks: pass.
- Revised TeX raw brace balance: zero.

### Promotion rules

- Do not relax the 0.15 rad yaw threshold or 25 m residual-tail threshold to make v16.3
  pass; rerun v16.4 from a fresh output root.
- Do not claim the new loss, source-adaptive capacity, or trust region independently
  effective unless the four-way attribution run passes across at least three seeds.
- Do not use a `least_violation` calibration as the paper operating point.
- Do not interpret logged replay as causal evidence that burden was transferred; report
  it as a learned-mechanism proxy and add reactive-agent plus human-audited stress tests.

## v16.8.9 engineering repair — complete smoke/protocol tooling and numeric-module import fix (2026-08-08)

No algorithm/data-threshold/output-path change. Restored the missing v16.8.9 diagnostic/protocol scripts, upgraded scripts 46/50 to their v16.8.9-compatible implementations, replaced invalid direct imports of numeric-prefixed script 59 with `importlib.import_module`, added a `CASUAL` typo-compatible smoke wrapper, and revalidated the v16.8.9 execution chain. Full Python regression: **175 passed**.

## v16.8.9 data-contract repair — exact audit/transport semantics and supervision sufficiency (2026-08-08)

The completed 96-scene v16.8.9 smoke passed the proposal point-estimate gates (`AnyNCF=0.4167`, false-safe `0.5000`, PBTR `0.4419`, hard recovery `0.2083`) but exposed 1,258 exact audit/transport affected-root mismatches and an overly strong hard requirement on the prevalence of burden-only affected roots. The repair makes root affectedness/conflict/retention and canonical root weights exact across audit and transport, keeps burden-only prevalence advisory rather than threshold-forcing, adds an in-place smoke NPZ repair path, and adds training-supervision sufficiency audits before strict/full training. No NCF/PBTR/proposal threshold or candidate geometry was relaxed. Full regression: **178 passed**.

## v16.8.13 — mechanism-auditability separation, empirical route evidence, typed PRIO recovery, and certificate visibility contract (2026-08-13)

### Triggering evidence from the v16.8.12 96-scene smoke

The v16.8.12 validation smoke completed successfully as a data/engineering run and passed the proposal/causal screen and the training-supervision audit, but its composite verdict correctly blocked the strict probe because model-support failed. Among 530 selected critical vehicles, 12 (2.264%) had zero natural roots and the same 12 had fewer than two low-burden roots. Every one of those failures was map-filter dominated, every one had priority relation `EQUAL_OR_NEGOTIATED`, and every one used the `logged_geometry_neutral_timing` reference because a lane-route could not be resolved. Ten had all 80 WOMD future states valid, one had 68, and one had only 19; therefore the remaining rootlessness was not primarily missing-future data.

A second independent contract failure was structural: natural-source support was OBS=1554, NEU=3324, PRIO=0. The PRIO generator reused the same route/timing family already occupied by OBS/NEU while cross-source de-duplication happened before typed-source retention. Consequently the data could not supervise the configured typed PRIO natural modes even when the total natural-root count was nonzero.

### Semantic corrections

1. **Critical selection is no longer conflated with offline mechanism-label auditability.** `cowp/critical/valid` retains the inference-time critical-selection semantics. A new `cowp/critical/mechanism_valid` indicates whether the offline WOMD record has enough lane or factual-route evidence to build an eight-second natural/transport/witness target without fabricating unsupported geometry.
2. **Evidence-gated empirical route corridor.** If a lane route is unavailable, a vehicle may use a narrow factual-geometry corridor only when at least 60 future steps and at least 70% of the 80-step horizon are valid. The corridor is constructed from contiguous valid WOMD future segments only, uses stricter geometric tolerances than the ordinary lane check, and is explicitly recorded with `natural/map_evidence_mode=2`; it is never marked as HD-map verified. This is geometry evidence only: timing remains neutralized rather than copied as a normative behavior target.
3. **Insufficient evidence is masked, not fabricated or silently deleted.** A selected critical with neither usable lane support nor sufficient factual-route evidence remains selected but has `mechanism_valid=0`. It does not contribute natural/transport/witness supervision. Its frequency is reported and hard-gated (`<=1%` criticals by default) so the pipeline cannot pass by hiding difficult actors.
4. **Protected PRIO root identity is explicit.** For protected `AGENT_PRIORITY` / `EQUAL_OR_NEGOTIATED` relations, the canonical progress-preserving root is assigned to PRIO before OBS/NEU candidates enter cross-source de-duplication. PRIO has its own ordered acceleration/speed family and every auditable protected critical is required to retain at least one priority-preserved PRIO root. No factual corridor is extrapolated beyond the observed geometric support merely to manufacture a PRIO sample.
5. **Auditability is finalized against the actual route builder.** The cheap critical-stage lane projection is only a precheck; natural generation downgrades `mechanism_valid` when a projected lane has no usable forward continuation and there is not enough factual-route evidence. This closes endpoint/degenerate-lane cases without changing `critical/valid`.
6. **Candidate certificate validity is separate from pair-label validity.** `cowp/candidates/certificate_valid` is true only when the complete selected-critical certificate universe is auditable. Raw NCF/false-safe labels are masked by this flag for candidate-level training, while pair-level mechanism losses use `mechanism_valid`.
7. **Model-input visibility is part of certificate validity.** If a Scenario-proto selected critical track cannot be represented in the tf.Example/model agent rows, the cache loader invalidates the corresponding mechanism supervision and all candidate-level certificate targets for that scene. A post-tensor-cache model-support audit is mandatory, preventing labels from certifying against agents the model cannot observe.
8. **Fingerprint coverage expanded.** `cowp/data/dataset.py` and `cowp/scripts/03_train.py` are now part of the fresh-cache code fingerprint because visibility masking and certificate-aware planner ranking change training semantics.

### Promotion gates

- On `mechanism_valid=1` criticals: zero rootless criticals, zero criticals with fewer than two low-burden roots, and every auditable protected critical has at least one priority-preserved PRIO root.
- Global auditability: `mechanism_unauditable_rate <= 1%` by default.
- Candidate certificate completeness: at least 98% of scenes remain certificate-complete after the cache/model-visibility merge.
- Proposal/causal, training-supervision, and existing response/witness/transport non-degeneracy gates remain unchanged.
- The smoke must explicitly authorize strict; validation strict and train-pilot must both authorize full rebuild under the identical current fingerprint.

### Performance policy

The v16.8.12 smoke spent about 101.23 s/scene in safe responses and 45.39 s/scene in witness construction versus 3.12 s/scene in natural generation. Therefore v16.8.13 does not reduce candidate, natural-root, response-slot, or witness semantics to gain speed. Empirical-corridor work is only enabled for lane-unresolved auditable actors and reuses contiguous factual geometry; the existing exact audit/result reuse remains the primary semantics-preserving optimization.

### Local validation

- Repository regression split: **201 passed, 5 skipped** with no failures.
- `python -m compileall -q cowp tests`: passed.
- `bash -n` on the v16.8.13 smoke/strict/train-pilot/master launchers and full-core preparation script: passed.

### Evidence limitation retained

Passing these data-support gates establishes that WOMD-derived labels can support the implemented natural-basis, same-root transport, witness, NCF selector, and logged-Waymax mechanism experiments. It does not make logged WOMD/Waymax replay counterfactual causal ground truth for burden transfer. Publication-level causal burden claims still require the separately specified reactive-agent protocol and held-out human-audited false-safe stress set.

v16.8.13 also propagates the certificate/auditability mask into label-space validation, causal-audit diagnostics, Waymax replay class balancing, and learned-offline label metrics. Certificate-unknown candidates may still be replayed for physical collision/off-road/progress outcomes, but they are never counted as NCF/false-safe negatives or positives. Label-space NCF/false-safe precision/recall and proposal-floor metrics expose certificate-label coverage explicitly, preventing missing counterfactual supervision from being reported as a successful negative class.

## v16.8.14 — WOMD split/completeness contract and adaptive probe manifests

This revision does **not** change the COWP planning/certification algorithm. It repairs dataset engineering contracts discovered before the v16.8.13 smoke run:

- Smoke hard scenes are selected from the current baseline validation cache instead of a stale v16.8.8 manifest.
- Probe manifests support a preferred hard count plus a fixed total; representative random scenes fill a hard-scene shortfall, while a minimum stress count remains explicit.
- WOMD 1.3.1 preflight now audits shard-index completeness, rather than treating a readable partial download as complete.
- Primary COWP data contract is Scenario training/validation for authoritative labels plus scenario-ID-matched tf.Example training/validation for model/Waymax tensors.
- Tensor-cache construction can require every proto-derived label scenario to have a matching tf.Example; partial training tensors are a hard error.
- Added a split-layout auditor for training/validation/testing and local challenge/auxiliary directories. `validation_interactive` is secondary stress data until scenario-ID overlap with standard validation is explicitly audited; blind testing splits are not used to fabricate future-dependent COWP mechanism labels.

## v16.8.15 — representation-aware WOMD v1.3.1 split contract (2026-08-13)

This revision does **not** change the COWP planner, natural-basis mathematics, response/witness labels, or promotion thresholds. It corrects the local WOMD release-layout model used by dataset engineering.

- Replaced the previous split × representation Cartesian-product audit with an explicit representation-aware matrix.
- Scenario splits inventoried by the current local/release contract: `training`, `validation`, `testing`, `validation_interactive`, `testing_interactive`, `training_20s`, `visualization`.
- tf.Example splits inventoried: `training`, `validation`, `testing`, `validation_interactive`, `testing_interactive`.
- `scenario/training_20s` and `scenario/visualization` are Scenario-only auxiliary directories. The auditor never constructs, globs, counts, or requires nonexistent `tf_example/training_20s` or `tf_example/visualization` peers.
- The current COWP primary benchmark remains the official 9-second train/validation pair: Scenario proto is authoritative for labels/map/traffic control; scenario-ID-matched tf.Example is the tensor/Waymax source.
- `validation_interactive` remains an optional secondary interaction-stress evaluation split and is not merged into standard validation; scenario-ID overlap can be audited before reporting it.
- `testing` and `testing_interactive` remain blind-evaluation-only because future GT is hidden; they are never used to construct COWP natural/transport/witness/NCF supervision.
- `scenario/training_20s` is excluded from the present 91-step pipeline. Using it would require a separate windowing and group-split design and cannot be achieved by swapping a glob.
- Added regression tests proving Scenario-only splits have no tf.Example glob or completeness requirement.

## v16.8.24 — rebuild readiness and execution-chain repair

See `ALGORITHM_CHANGELOG_V16_8_24.md`. The release fixes the missing profile-summary module reference, adds report-only benchmark recovery and active-chain dependency preflight, persists WOMD Scenario indices across benchmark/full build, makes compact split selection self-contained, tunes CPU worker defaults, and attaches Waymax outcomes to train/val/heldout-test. Label semantics are unchanged from the semantically verified v16.8.23 fast path.

## v16.8.25 — Exact-ID Waymax repair + experimental MCFC (2026-08-23)

See `ALGORITHM_CHANGELOG_V16_8_25.md`. Current v16.8.24 evidence attributes the dominant learned-offline ceiling to proposal support (`AnyNCF` about 0.35--0.36; oracle selected false-safe floor about 0.59--0.60) while protected/global BCOT discrimination is already strong. This revision therefore (1) repairs the previously non-functional exact-ID Waymax held-out path and records exact scenario coverage, and (2) adds an **opt-in, unpromoted** Multi-Conflict Feasibility Corridor proposal family that replaces JR's single-acceleration timing profile with piecewise protected timing knots plus post-conflict recovery. The canonical v16.8 label config remains behaviorally unchanged; MCFC is enabled only by `configs/label_cowp_v16_8_25_mcfc.yaml`.

MCFC must pass a paired validation proposal probe before any full rebuild/retraining. If it fails, disable it; do not repeat PCHR/threshold-only/flat-certificate attempts. The currently inspected 1,200-scene held-out set is now diagnostic rather than final-blind for subsequent algorithm selection; final paper evaluation must use a new untouched split sampled from unused WOMD validation IDs after configuration freeze. The current data also contain zero burden-only affected roots, so affected-root-vs-conflict-only superiority is not an empirically supported headline claim yet.

### v16.8.25 evidence protocol addendum

- Added `cowp/scripts/85_screen_v16_8_25_mcfc_probe.py`: MCFC now requires a
  source-attributed effect-size gate (not just an aggregate fresh-bank pass)
  before full rebuild/retraining.
- Added `NEXT_RUN_COMMANDS_V16_8_25_MCFC_CN.sh`: exact-ID v16.8 strict-Waymax
  diagnosis -> validation-only MCFC probe -> blind-final ID freeze -> gated
  rebuild/retrain -> one-shot final-blind evaluation.
- The already-inspected v16.8.24 1,200-scene held-out set is development-only
  for future algorithm decisions; a new content-blind ID-hash holdout is
  required for final claims.
