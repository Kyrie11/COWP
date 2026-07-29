# OC-RAP v48.10 COPE Change Manifest

## Algorithm

- Added **Conditional Option Preference (COP)**: conditional ranking over recovery candidates only; nominal is removed from option ranking and conditional rank-margin calculation.
- Added ambiguity-aware conditional acceptable-set mass and exact expected-recovery-regret losses.
- Added lower-weight no-op/harm-group conditional preference supervision so these groups teach least-bad recovery ordering without deciding intervention.
- Added **Monotone Ordinal Evidence (MOE)** with ordered cumulative benefit/non-harm logits and focal policy-top1 tri-state supervision.
- Added `ordinal_evidence` as a unified risk source for checkpoint metrics, calibration, evaluator, selector, and closed-loop runner.
- Added conditional recovery rank margin, which compares the best recovery to the runner-up recovery rather than to nominal.

## Engineering and attribution

- Added `training.strict_init_prefixes`; Stage E aborts if Stage-P preference geometry is not loaded exactly.
- Added `STAGE_ARCHITECTURE.json` and immutable staged checkpoint SHA256 completion metadata.
- Fixed stage-width propagation in the v48.10 staged and ablation scripts.
- Added resumable 4×2 causal ablation scheduling with one task per A30 and immutable `TASK_COMPLETE.json` markers.
- Added multi-seed and dedicated-calibration scripts for ordinal evidence.
- Added v48.10 layered learning-gate diagnostics.

## Modified source files

- `src/ocrap/models/ocrap.py`
- `src/ocrap/models/losses.py`
- `src/ocrap/models/inference.py`
- `src/ocrap/cli/train.py`
- `src/ocrap/config/defaults.py`
- `src/ocrap/planning/selector.py`
- `src/ocrap/evaluation/evaluator.py`
- `src/ocrap/evaluation/baselines.py`
- `src/ocrap/simulation/closed_loop_runner.py`
- `tools/calibrate_policy_risk_v48.py`
- `scripts/train_ocrap_v48_trac_sr.sh`
- `run_v48_two_gpu_fast_commands.txt`

## New executable/support files

- `scripts/train_ocrap_v48_10_cope.sh`
- `scripts/run_v48_10_parallel_ablations.sh`
- `scripts/recalibrate_v48_10_multiseed.sh`
- `scripts/recalibrate_v48_10_on_dedicated_set.sh`
- `tools/check_v48_10_learning_gates.py`
- `tests/test_v48_10_cope.py`

## Documentation and audit

- `OC-RAP-v48.9-results-audit-summary.json`
- `OC-RAP-v48.9-results-audit-and-v48.10-COPE-plan-ZH.md`
- `OC-RAP-v48.10-run-commands-ZH.txt`
- root `ALGORITHM_CHANGELOG.md` updated with v48.10 evidence, design, ablations, and non-repetition record.
