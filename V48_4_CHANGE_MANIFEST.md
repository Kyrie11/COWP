# v48.4 Change Manifest

## Algorithm
- `src/ocrap/models/ocrap.py`: zero-initialized NASC residual and more conservative gate.
- `src/ocrap/models/losses.py`: DRA-RCD ranking/admission separation, soft opportunity/harm targets, pseudo-environment GroupDRO.
- `src/ocrap/cli/train.py`: exact teacher-PCD group policy metrics, worst-regime regret checkpointing, strict missing-metric failure, unified direct loss path.

## Training and experiments
- `scripts/train_ocrap_v48_trac_sr.sh`: v48.4 defaults and correct best metric.
- `scripts/run_v48_4_core_ablations.sh`: four explicit core ablations.
- `scripts/recalibrate_v48_4_multiseed.sh`: fixed-checkpoint 4801/4802/4803 recalibration.
- `tools/summarize_v48_4_ablations.py`: ablation summary.
- `tools/summarize_v48_4_multiseed.py`: multi-seed summary.
- `run_v48_4_main_commands.txt`: complete command sequence.

## Evidence and documentation
- `OC-RAP-v48.3-results-audit-and-v48.4-plan-ZH.md`
- `V48_3_RESULTS_AND_BASELINES_SUMMARY.json`
- `ALGORITHM_CHANGELOG.md`
- `V48_4_VALIDATION_STATUS.txt`

## Tests
- Added ZI-NASC identity, DRA-RCD decoupling, and policy-regret metric tests to `tests/test_v48_trac_sr.py`.
