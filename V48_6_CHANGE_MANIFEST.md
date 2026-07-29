# v48.6 OC-TRAC-RPGC Change Manifest

## Algorithm and model

- `src/ocrap/models/ocrap.py`
  - Adds zero-initialized preference-only relative context.
  - Adds a direct candidate-minus-nominal PCD gain mean/log-variance head.
  - Keeps legacy shared NASC optional and disabled in the v48.6 main path.
- `src/ocrap/models/losses.py`
  - Adds exact relative-gain Gaussian NLL.
  - Adds confidence-paced listwise preference and rank-gap calibration losses.
- `src/ocrap/models/inference.py`
  - Saves, restores, and exports the new preference-context and direct-delta outputs.

## Training and checkpoint selection

- `src/ocrap/cli/train.py`
  - Uses direct relative gain for admission diagnostics when enabled.
  - Adds scene-hash fold policy metrics and worst-fold checkpoint selection.
  - Adds teacher-best-macro-balanced positive-group sampling.
  - Adds explicit output-mode handling and preference confidence metrics.
- `scripts/train_ocrap_v48_trac_sr.sh`
  - Selects `direct_policy_risk_fold_worst` as the best metric.
  - Enables RPGC defaults and disables legacy shared NASC/GroupDRO in the main path.
- `configs/default.yaml`, `src/ocrap/config/defaults.py`
  - Add configuration fields for RPGC and rank-margin abstention.

## Calibration and deployment

- `tools/calibrate_policy_risk_v48.py`
  - Adds `direct_delta` risk source.
  - Uses preference rank for within-group selection and direct delta for admission.
  - Jointly calibrates a minimum rank margin.
  - Enforces macro concentration on the calibration fit fold.
  - Separates direct-risk, legacy-harm-head, and rank-margin diagnostics.
- `src/ocrap/planning/selector.py`
  - Adds runtime rank-margin abstention.
- `src/ocrap/evaluation/evaluator.py`, `src/ocrap/evaluation/baselines.py`
  - Propagate direct-delta and rank-margin policy settings.
- `src/ocrap/simulation/closed_loop_runner.py`
  - Supports direct-delta admission in closed-loop execution.
- `scripts/recalibrate_v48_on_dedicated_set.sh`
  - Uses the same direct-delta contract for dedicated calibration.

## Experiment automation

- `scripts/recalibrate_v48_6_multiseed.sh`
  - Fixed-checkpoint calibration on seeds 4801/4802/4803 with independent output roots.
- `scripts/run_v48_6_core_ablations.sh`
  - Runs the four controlled v48.6 attribution experiments.
- `tools/audit_v48_6_completion.py`
  - Audits immutable checkpoint and calibration completeness.
- `tools/summarize_v48_6_multiseed.py`
  - Summarizes seed stability and gate outcomes.
- `tools/summarize_v48_6_ablations.py`
  - Summarizes module attribution.
- `run_v48_two_gpu_fast_commands.txt`
  - Propagates v48.6 variables through the two-GPU controller.

## Tests and documentation

- `tests/test_v48_6_rpgc.py`
  - Tests warm-start identity, nominal anchoring, permutation equivariance, and exact direct-delta supervision.
- `ALGORITHM_CHANGELOG.md`
  - Adds the evidence, design, attribution protocol, gates, and non-repetition record for v48.6.
- `OC-RAP-v48.5-results-audit-and-v48.6-RPGC-plan-ZH.md`
  - Full v48.5 result audit and v48.6 rationale.
- `OC-RAP-v48.6-run-commands-ZH.txt`
  - Ordered main, audit, multi-seed, ablation, dedicated-calibration, and closed-loop commands.
