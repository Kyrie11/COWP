# COWP v13-TKN Modification Manifest

## Modified files

- `ALGORITHM_CHANGELOG.md`
- `cowp/models/natural_decoder.py`
- `cowp/models/cowp_model.py`
- `cowp/models/losses.py`
- `cowp/scripts/04_eval_closed_loop.py`
- `cowp/scripts/32_gate_natural_basis.py`
- `cowp/waymax_eval/metrics_cowp.py`
- `cowp/waymax_eval/policy_wrapper.py`

## Added files

- `configs/train_cowp_v13.yaml`
- `configs/label_cowp_v13.yaml`
- `configs/eval_cowp_v13.yaml`
- `cowp/scripts/33_diagnose_cache_alignment.py`
- `cowp/scripts/34_diagnose_natural_oracles.py`
- `run_cowp_v13_dual_gpu.sh`
- `tests/test_v13_natural_decoder.py`
- `COWP_V12_DIAGNOSIS_AND_V13_EXECUTION_CN.md`
- `NEXT_RUN_COMMANDS_CN.txt`
- `V12_NATURAL_GATE_RECHECK.json`

## Validation completed in the delivery environment

- `python -m compileall -q cowp`: passed
- `bash -n run_cowp_v13_dual_gpu.sh`: passed
- `pytest -q`: 73 passed

## Not validated locally

Actual WOMD/raw cache/transport cache execution, GPU training, Waymax rollout, reactive non-ego simulation, and SOTA performance require the user's server-side datasets and environment.
