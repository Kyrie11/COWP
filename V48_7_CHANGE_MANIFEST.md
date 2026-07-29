# OC-RAP v48.7 OC-TRAC-SPIRE change manifest

## Algorithm and training

- Added ambiguity-aware set-valued exact-PCD preference targets with regime-specific tie tolerances.
- Added acceptable-set KL, acceptable-vs-worse margin, tie-aware regret, and preference-gap supervision.
- Added staged SPIRE training:
  - Stage P trains only independent preference adapters while freezing encoder/value/risk paths.
  - Stage C freezes ranking and trains only the direct candidate-minus-nominal certificate adapter.
- Added `training.trainable_param_prefixes` for auditable positive allow-list parameter freezing.
- Replaced the v48.6 checkpoint-risk approximation with deployment-aligned Gaussian-CDF admission metrics.
- Added missed-opportunity cost so an always-abstain checkpoint cannot win early stopping trivially.

## Calibration and evaluation

- Added conditional harmful-selected confidence-bound constraints separately from population harmful exposure.
- Added strict single-winner and tie-aware acceptable-set ranking metrics.
- Added fit/verify controls for conditional harmful-switch UCB.
- Updated teacher-index quality gates to count deployable-macro opportunities separately from all macros.
- Added automatic multi-seed and ablation summaries plus a submission-gate checker.

## Engineering isolation

- Removed the silent fallback to `runs/ocrap_v48_trac_sr_regime_balanced` from closed-loop execution.
- Natural-gate rejection now writes `GATE_FAILED.json` and blocks stress closed loop explicitly.
- Added a nominal-locked Safe-only non-inferiority runner independent of Near/Contact policy promotion.
- Added partial/complete dedicated calibration merging with manifest checks, scene filtering, atomic installation, and overlap audit.
- Added v48.7 completion auditing and immutable completed-checkpoint validation for recalibration.

## New or materially changed files

- `src/ocrap/models/losses.py`
- `src/ocrap/cli/train.py`
- `tools/calibrate_policy_risk_v48.py`
- `tools/build_teacher_pcd_index_v48.py`
- `run_v48_two_gpu_fast_commands.txt`
- `scripts/train_ocrap_v48_7_spire.sh`
- `scripts/recalibrate_v48_7_multiseed.sh`
- `scripts/run_v48_7_core_ablations.sh`
- `scripts/recalibrate_v48_7_on_dedicated_set.sh`
- `scripts/run_v48_7_safe_noninferiority.sh`
- `scripts/merge_v48_calibration_regimes_to_eval_root.sh`
- `scripts/run_ocrap_v48_trac_sr.sh`
- `tools/audit_v48_7_completion.py`
- `tools/summarize_v48_7_multiseed.py`
- `tools/summarize_v48_7_ablations.py`
- `tools/check_v48_7_submission_gates.py`
- `ALGORITHM_CHANGELOG.md`
