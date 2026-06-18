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
