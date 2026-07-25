# COWP Algorithm Change Log

This is the canonical record of algorithm attempts. Do not repeat a rejected
change without new evidence. Every experiment must record the code version, data
version, seed, checkpoint lineage, learned-offline gate, online paired metrics,
and the exact simulator-agent setting.

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
