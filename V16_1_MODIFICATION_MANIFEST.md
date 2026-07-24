# v16.1 Modification Manifest

## Modified

- `cowp/models/losses.py`
  - fixed real-batch mode-usage mask broadcasting;
  - robust eligible-mode aggregation.
- `cowp/scripts/27_diagnose_transport_labels.py`
  - threaded NPZ reads;
  - streaming exact aggregate metrics;
  - bounded-memory histogram quantiles.
- `run_cowp_v16_1_dual_gpu.sh`
  - fast/full diagnostic profiles;
  - cached diagnostics;
  - stage-specific DataLoader settings;
  - diagnose-only stop;
  - strict run provenance checked before canonical configs are overwritten.
- `NEXT_RUN_COMMANDS_V16_1_CN.sh` / `NEXT_RUN_COMMANDS_V16_CN.sh`
  - entire workflow background by default;
  - sampled cache audit by default;
  - cached cache-reuse gate.
- `ALGORITHM_CHANGELOG.md`
  - v16.1 diagnosis and decisions.

## Added

- `cowp/scripts/42_write_run_provenance.py`
- `tests/test_v16_1_engineering.py`
- realistic shape regressions in `tests/test_v16_cnob_dynamics.py`
- `NEXT_RUN_COMMANDS_V16_1_FULL_CN.sh`
- `RUN_NATURAL_ABLATIONS_V16_1_CN.sh`
- `RUN_FULL_DATA_AUDIT_V16_1_CN.sh`
- `CHECK_RUN_STATUS_V16_1.sh`
- `COWP_V16_RESULT_DIAGNOSIS_AND_V16_1_EXECUTION_CN.md`
- `PAPER_METHOD_ALIGNMENT_V16_1_CN.md`
