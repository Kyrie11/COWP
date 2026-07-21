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
