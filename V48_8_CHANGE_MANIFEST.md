# OC-RAP v48.8 SCOPE Change Manifest

## Version

- New version: **v48.8 OC-TRAC-SCOPE**
- Base: uploaded v48.7 SPIRE code
- Goal: isolate Preference, relative gain, and deployability evidence while fixing experimental attribution and runtime consistency.

## Core model and loss changes

### `src/ocrap/models/ocrap.py`

- Added `direct_recovery_relative_features_include_absolute`.
- Preference/delta adapters can consume invariant relative-only features.
- Added configurable fixed initial delta log-variance.
- Preserved zero-initialized residual behavior for checkpoint-compatible warm start.

### `src/ocrap/models/losses.py`

- Added nominal-inclusive all-group set preference objective.
- Added replacement mode so set-valued supervision does not conflict with legacy single-winner/listwise losses.
- Added explicit nominal margin and harmful-recovery margin.
- Added direct-delta sign supervision.
- Added no-opportunity/harm-group ranking supervision.

### `src/ocrap/cli/train.py`

- Strict best-checkpoint improvement with configurable minimum delta.
- Added support-aware fold aggregation and worst-K supported-fold metrics.
- Preference checkpoint risk now includes harmful top-1 and false intervention terms.
- Added certificate fold-robust metric.
- Training summaries retain trainable prefixes and best-metric tolerance.

## Calibration and runtime consistency

### `tools/calibrate_policy_risk_v48.py`

- Added `conformal_delta` risk source.
- Added one-sided finite-sample split-conformal gain bounds.
- Added near-miss rules evaluated on verify scenes.
- No-rule runs still emit unconstrained diagnostic rows.
- Fixed duplicate negative-gain CLI declaration.

### `src/ocrap/models/inference.py`
### `src/ocrap/evaluation/evaluator.py`
### `src/ocrap/simulation/closed_loop_runner.py`

- Added selector/runtime support for conformal residual quantile and temperature.
- Unified calibration, offline evaluation, and closed-loop certificate semantics.

## New training and experiment scripts

### `scripts/train_ocrap_v48_8_scope.sh`

- Stage P trains only a small relative-only preference context adapter.
- Stage C freezes ranking and trains only robust direct-delta mean/sign evidence.
- Learned variance NLL is disabled by default.
- Writes staged checkpoint SHA256 completion marker.

### `scripts/prepare_v48_8_shared_assets.sh`

- Builds proxy scene split once.
- Builds exact teacher-PCD index once.
- Writes reusable asset manifest and overlap audit.

### `scripts/run_v48_8_parallel_ablations.sh`

- Submits all four ablations to a two-GPU queue.
- Runs at most one training process per A30.
- Reuses shared split and teacher index.
- Produces one structured ablation summary.

### `scripts/recalibrate_v48_8_multiseed.sh`

- Recalibrates immutable completed checkpoints on seeds 4801/4802/4803.
- Uses conformal-delta risk source.
- Rechecks checkpoint SHA256 after all seeds.

### `run_v48_two_gpu_fast_commands.txt`

- Added single-variant controller support.
- Added shared/prebuilt split and teacher-index support.
- Added conformal calibration arguments.
- GPU locks now apply only to requested variants.

## Safe evaluation corrections

### `scripts/run_ocrap_v48_trac_sr.sh`

- Fixed duplicate `for` syntax error in the legacy summary block.
- Added paired scalar/reference and model Safe execution on two GPUs.
- Added configurable Safe WOMD source.

### `scripts/run_v48_7_safe_noninferiority.sh`

- Enables paired Safe reference/model run by default.

### `tools/analyze_safe_paired_noninferiority_v48_8.py`

- Matches results by scene ID.
- Computes paired bootstrap confidence intervals for collision, offroad, NUP, and intervention episode rate.
- Explicitly reports unavailable route/jerk/yaw metrics rather than substituting proxies.

## New analysis tools

### `tools/check_v48_8_learning_gates.py`

- Separately reports Stage P, Stage C discrimination, and Natural-gate status.

### `tools/summarize_v48_8_parallel_ablations.py`

- Aggregates four ablations and both variants.

## Tests

### `tests/test_v48_8_scope.py`

Covers:

- nominal-inclusive all-group preference;
- invariant relative-only feature construction;
- support-aware fold aggregation;
- finite-sample conformal quantile behavior.

## Validation

- 137 tests passed.
- Python compileall passed.
- Modified shell scripts passed `bash -n`.
- Real WOMD/Waymax/A30 experiments were not run in the local environment.
