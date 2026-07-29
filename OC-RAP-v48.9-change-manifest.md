# OC-RAP v48.9 PACER Change Manifest

## Algorithm and training

- `src/ocrap/models/losses.py`
  - Added partial-label acceptable-set mass loss.
  - Added nominal-only supervision for no-opportunity groups.
  - Added weak dead-zone versus strong harmful intervention margins.
  - Added policy-selected top-1 relative-gain regression and sign losses.
- `src/ocrap/cli/train.py`
  - Wires the new PACER loss controls into training.
- `scripts/train_ocrap_v48_trac_sr.sh`
  - Exposes the new loss controls as environment/config overrides.
- `scripts/train_ocrap_v48_9_pacer.sh`
  - New two-stage PACER training: intervention-aware Stage P followed by frozen-policy Stage C.

## Calibration and diagnostics

- `tools/calibrate_policy_risk_v48.py`
  - Added `policy_top1` conformal residual scope.
  - Preserves diagnostic frontiers when probability bounds saturate.
  - Adds policy-top1 gain/harm AUC, gain MAE, false-switch and activation diagnostics.
- `tools/check_v48_9_learning_gates.py`
  - Separately audits Preference, policy-induced Relative gain, and deployment Certificate.
- `scripts/recalibrate_v48_9_multiseed.sh`
  - Fixed-checkpoint proxy recalibration for seeds 4801/4802/4803.

## Ablations and execution

- `scripts/run_v48_9_parallel_ablations.sh`
  - Runs four causal ablations with at most one task per A30 and two tasks concurrently.
- `run_v48_two_gpu_fast_commands.txt`
  - Passes conformal scope and uses a configurable run label.
- `tests/test_v48_9_pacer.py`
  - Tests partial-label set mass, policy-top1 gradient targeting, and conformal scope.

## Documentation

- `ALGORITHM_CHANGELOG.md`
  - Adds the v48.9 evidence, design, non-repetition note, validation protocol, and local status.
- `OC-RAP-v48.8-results-audit-and-v48.9-PACER-plan-ZH.md`
- `OC-RAP-v48.9-run-commands-ZH.txt`
- `OC-RAP-v48.8-results-audit-summary.json`
