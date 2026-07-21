# COWP Algorithm Change Log

This file is the canonical record of algorithm attempts. Do not repeat a rejected
change without new evidence. Each run must record code version, data version, seed,
checkpoint lineage, offline gate and online paired metrics.

## v8 — Aggregate structured certificate
- Added threshold-connected hard certificate and candidate-level classifier.
- Result: false-safe and burden reduced, but EP collapsed and fallback rose sharply.
- Rejected approach: solving the problem by threshold relaxation alone. Witness AUPRC stayed about 0.43.

## v9 — Primitive-indexed transport supervision
- Added mode conflict/retain labels, root assignment and response auxiliary losses.
- Fixed hidden NPZ enumeration and response-root gather dimensionality.
- Seed 2026 result: learned-offline gate failed; no real Waymax probe was executed.
- Key evidence: witness AUPRC 0.4312, accepted NCF recall 0.1280, accepted candidate rate 0.0613.
- Failure diagnosis:
  1. set-transport head did not receive candidate--natural trajectory geometry;
  2. FREEZE_BACKBONE_EPOCHS=999 froze candidate/natural/witness identity modules;
  3. root recovery used unweighted product over diffuse response slots and overpredicted sparse recovery;
  4. mode-conflict validation BCE (0.722) was worse than the class-prior entropy baseline (~0.649);
  5. candidate classifier AUPRC dominated pair witness AUPRC, so the claimed mechanism was not isolated.

## v10-GCT — Geometry-Conditioned Transport (current)
### Code changes
- Explicit compact relative geometry between each ego candidate and natural primitive.
- Raw mode logits and dynamically balanced direct conflict/retain BCE.
- Response-mixture-weighted same-root recovery instead of unweighted product-of-complements.
- Slot-specific residual root queries, zero initialized for v9 checkpoint compatibility.
- Granular freeze policy: graph warm-up only in transport stage; candidate/natural/witness heads remain trainable.
- Small natural-set auxiliary loss during transport learning to preserve primitive identity.
- Weighted recovery-presence + positive-magnitude objective for sparse recovery labels.
- Added no-skill baseline diagnostic (`cowp.scripts.29_diagnose_v10_readiness`).

### Required evidence before promotion
- mode-conflict validation BCE below its no-skill entropy baseline and decreasing;
- witness AUPRC >= 0.55 in the first development gate, target >= 0.65 for paper;
- accepted NCF recall >= 0.30 in development, target >= 0.50;
- accepted candidate rate >= 0.10 without fallback > 0.25;
- selected false-safe reduction >= 8 percentage points offline before a 100-scene probe;
- paired 1000/5000-scene online evaluation before any SOTA claim.

## Prohibited shortcuts
- Do not claim closed-loop results from attached sparse Waymax candidate outcomes.
- Do not report SOTA from 100 scenes or one random seed.
- Do not relax the gate merely to make the pipeline continue.
- Do not add log-divergence loss while finite label coverage is zero.
- Do not interpret candidate-certificate gains as proof of primitive transport without mechanism ablations.
