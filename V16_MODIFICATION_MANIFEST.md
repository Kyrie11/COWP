# COWP v16 Modification Manifest

## Core model

- `cowp/models/natural_decoder.py`
  - central decoder registry;
  - dynamics-consistent CNOB residual integration;
  - configurable OBS capacity scale;
  - exact analytic zero initialization.
- `cowp/models/cowp_model.py`
  - propagates `natural_obs_capacity_scale`.
- `cowp/models/losses.py`
  - weighted source-restricted branch minADE;
  - OBS gain, prior preservation, physical consistency, smoothness and mode-usage losses.

## Training and evaluation

- `cowp/scripts/03_train.py`
  - mechanism-aligned natural/planner checkpoint composites.
- `cowp/scripts/35_diagnose_model_anchor.py`
  - uses centralized decoder capabilities; fixes v15 preflight failure.
- `cowp/scripts/36_audit_causal_protocol.py`
  - dynamic decoder and explicit selector-fallback audit.
- `cowp/scripts/39_diagnose_learned_natural.py`
  - learned-vs-analytic/source/horizon/physical/mode-use report.
- `cowp/scripts/40_gate_natural_effectiveness.py`
  - hard evidence gate for decoder usefulness.
- `cowp/scripts/41_compare_natural_ablations.py`
  - controlled new-loss and OBS-capacity attribution gate.
- `cowp/scripts/19_diagnose_waymax_cache_sufficiency.py`
  - excludes hidden NPZ metadata from scenario counts.
- `cowp/waymax_eval/rollout.py`
  - honors disabled generic-certificate fallback.

## Configs

- `configs/model_cowp_v16.yaml`
- `configs/train_cowp_v16.yaml`
- `configs/label_cowp_v16.yaml`
- `configs/eval_cowp_v16.yaml`
- `configs/label_cowp_v16_pareto_ablation.yaml`
- `configs/label_cowp_v16_pairmax_ablation.yaml`
- `configs/train_cowp_v16_no_effectiveness_loss.yaml`
- `configs/model_cowp_v16_no_obs_capacity.yaml`

## Drivers

- `NEXT_RUN_COMMANDS_V16_CN.sh`: v9 reuse, default stop after natural gates.
- `RUN_NATURAL_ABLATIONS_V16_CN.sh`: two controlled natural ablations.
- `NEXT_RUN_COMMANDS_V16_FULL_CN.sh`: continue transport/planner/Waymax.
- `run_cowp_v16_dual_gpu.sh`: integrated v16 pipeline and full online deltas.

## Tests and validation

- `tests/test_v16_cnob_dynamics.py`
- `tests/test_hidden_npz_exclusion.py`
- `validation/causal_protocol_audit_v16.json`
- `V16_LOCAL_VALIDATION.json`

## Documentation

- `COWP_V15_FAILURE_AND_V16_EXECUTION_CN.md`
- `ALGORITHM_CHANGELOG.md`
