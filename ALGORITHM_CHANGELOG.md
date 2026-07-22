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
