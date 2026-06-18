# COWP optimization notes

This package was revised after reviewing the paper, label diagnostics, and implementation.

## Algorithm/data fixes

- Propagated critical-agent priority relation (`rho`) into ego-conditioned safe-response burden computation and typed safe-budget search.
- Added progress-loss based inference of delay/gap loss inside `compute_burden`, enabling PA/GS mechanism tokens instead of collapsing normative violations into AY/OR.
- Normalized OPR by the mass of low-burden natural alternatives, and stored `cowp/witness/natural_mass_by_source` for source-resolved branch ablations.
- Made branch-ablation planning recompute OPR as a ratio under enabled natural branches.
- Changed offline fallback selection to prefer neutral/stop-like conventional plans and to avoid reporting a coercive candidate as selected when only a conservative fallback is appropriate.
- Changed label-only EP to a normalized progress metric, with `EP_m` retained for meter-scale debugging and `FallbackRate` exposed separately.

## Configuration updates

- Expanded ego candidate terminal speed/progress lattice, merge timing offsets, and lane-change durations/delays.
- Reduced endpoint dedup tolerance to increase endpoint diversity.
- Raised the diagnostic expectation for mechanism-token diversity.

## Remaining limitations

- The current code remains a practical label/certification engine plus training/evaluation interface, not a full neural reproduction of the paper's graph decoder and diffusion-style natural alternative generator.
- Closed-loop claims still require generating Waymax-compatible rollout datasets and running simulator metrics; label diagnostics alone cannot prove closed-loop performance.

## Second-pass fixes in this package

- Changed natural-alternative trajectory supervision from slotwise L1 to weighted set-minADE so OBS/NEU/PRIO alternatives are trained as an unordered counterfactual set, matching the paper formulation better and avoiding arbitrary construction-order overfitting.
- Added online COWP policy diagnostics during Waymax rollout: selected candidate, accepted-candidate count, fallback flag, selected-candidate witness probability, OPR, predicted burden and C_i.
- Added closed-loop diagnostic aggregation in `04_eval_closed_loop.py` through `policy_diagnostic_summary`, so Waymax mode now reports model-predicted FSR/CBS/OPR-style signals in addition to raw rollout steps and optional official Waymax metrics.
- Added robust scalar aggregation for official Waymax metric outputs via `aggregate_waymax_standard_metrics`.
- Added regression tests for natural set-order invariance and closed-loop policy diagnostic aggregation.

## Data recommendation from uploaded diagnostics

The current train/val labels are adequate to start training. The endpoint-spread warning is distributional rather than a schema or supervision failure: 5.46 m is below the 6.0 m soft target, but candidate count, critical-agent coverage, witness positive rate, false-safe rate, NCF candidate rate, natural alternatives, response safety, and train/val consistency are all strong enough for model training. Rebuilding the full label set solely to move endpoint spread from 5.46 m to 6.0 m is not the most cost-effective next step unless the first trained model shows poor candidate selection diversity or weak stress-set ablations.
