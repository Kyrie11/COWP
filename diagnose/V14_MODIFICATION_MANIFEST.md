# COWP v14 Modification Manifest

## Algorithm
- Typed Natural Option Basis: `cowp/models/natural_decoder.py`
- Source-restricted matching and prior preservation: `cowp/models/losses.py`
- Natural base trajectory anchoring: `cowp/models/cowp_model.py`

## Training and gates
- Epoch -1 evaluation, checkpoint-prefix reset, graph warmup, grad clip: `cowp/scripts/03_train.py`
- Natural promotion gate: `cowp/scripts/32_gate_natural_basis.py`
- Config: `configs/train_cowp_v14.yaml`

## Diagnostics
- Fast raw/transport alignment: `cowp/scripts/33_diagnose_cache_alignment.py`
- Vectorized natural oracle: `cowp/scripts/34_diagnose_natural_oracles.py`
- Exact model-facing anchor preflight: `cowp/scripts/35_diagnose_model_anchor.py`

## Execution
- Driver: `run_cowp_v14_dual_gpu.sh`
- Commands: `NEXT_RUN_COMMANDS_V14_CN.txt`
- Diagnosis/report: `COWP_V13_DIAGNOSIS_AND_V14_EXECUTION_CN.md`
- Changelog: `ALGORITHM_CHANGELOG.md`

## Tests
- `tests/test_v14_typed_natural_basis.py`
- Full local suite: 76 passed.

## Source layout cleanup
The stale top-level duplicate Python trees (`models/`, `scripts/`, `core/`, etc.)
were removed. `pyproject.toml` packages only `cowp*`, and all documented commands
use `python -m cowp...`; the authoritative source is therefore unambiguous.
