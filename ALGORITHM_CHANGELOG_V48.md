# v48 OC-TRAC-SR Algorithm Changelog

## Motivation

v47 improved candidate-level positive-recovery AUC but failed policy-level top-1 selection and scene-disjoint risk verification, especially in Contact.

## Algorithm

- Added tri-state candidate-vs-nominal supervision: positive, dead-zone, harmful.
- Added nominal as an explicit setwise abstention class.
- Added independent harmful-switch head.
- Added conservative two-expert aggregation: lower confidence for gain/opportunity, upper confidence for harm.
- Added asymmetric expert specialization without hidden regime routing.
- Unfroze shared observation encoder with layer-wise learning rate.
- Added direct-only fast path.
- Aligned sampler, loss and calibration to exact teacher PCD.
- Added scene-balanced exact positive-group sampling.
- Added policy-level joint opportunity/harm/gain/macro calibration.
- Disabled historical handwritten rescue certificates in the v48 main-policy evaluation path.

## Data and protocol

- Added clean WOMD training-based Near/Contact builder.
- Added dedicated standard-validation calibration builder.
- Added filtering to exclude all existing val/test scenes from dedicated calibration roots.
- Added scene overlap audit.
- Added scene-disjoint low-cost split from existing val roots.
- Added positive-group and positive-scene minimum quality gates.
- Preserved existing user val/test roots.

## Engineering

- Added BF16/TF32, pinned/persistent data loading and prefetch.
- Added partial checkpoint loading for shape-changed heads.
- Fixed sequential two-GPU data-worker synchronization.
- Added background controller and centralized logs.
- Added harm prediction propagation through selector, offline evaluator and closed-loop runner.

## Validation

- Python compileall passed.
- Shell syntax checks passed.
- 97 tests passed, 2 non-failing warnings.
- No v48 WOMD/GPU experiment has yet been executed in this environment.
