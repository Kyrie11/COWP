# EICDRR evidence-first COWP patch (2026-10-13)

This is a RESEARCH EXPERIMENT PATCH, NOT a new verified safety controller.

## What changed

- `research_probe/native_womd.py`: compact SDC-first NPZ and source record provenance (no longer wrongly aliases track 0 as SDC).
- `research_probe/verify_cohort_identity.py`: verify exported identity and optional raw TFExample ID/xy.
- `research_probe/matched_response_test.py`: progress-matched, smooth ego interventions; pair-specific collision and braking-proxy contrasts; same-model ghost control.
- `research_probe/waymax_paired_idm.py`: actual SDC Waymax ID preflight. This is a PILOT, not a human counterfactual or GameFormer result.
- `tests/test_native_sdc_order_regression.py`, `tests/test_matched_response_test.py`, `tests/test_cohort_identity_patch.py`: new regression coverage.
- `NEXT_SCIENTIFIC_GATE_20261013_CN.sh`: step-by-step commands.

Read `SCIENTIFIC_AUDIT_AND_NEXT_STEPS_CN.md` for gates, caveats and full experiment protocol.

## Quick start

```bash
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh tests
export RAW_WOMD_ROOT=/path/to/WOMD/tf_example
export EICDRR_DATA_ROOT=/new/empty/results/root
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh cohort_dev
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh identity_npz
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh identity_raw12
bash NEXT_SCIENTIFIC_GATE_20261013_CN.sh matched_dev
```

Do not overwrite existing cohort.  `tests/test_core.py` depends on a missing legacy package `ocrap` and is intentionally excluded from the 65-passing-test report. TensorFlow+Waymax were not available to execute a real scenario here. See reports for exact limitations.
