# COWP fixes v3

## Critical-agent/input-row alignment

- Added `cowp/critical/track_id` to label generation from Scenario protos.
- Added tensor-cache merge-time id remapping from Scenario critical agent id to WOMD tf.Example input row.
- Added `cowp/critical/input_index` as the model gather index, while preserving `cowp/critical/track_index` and `cowp/critical/track_index_original` for diagnostics.
- Dataset/model now prefer `input_index` and still mask invalid/invisible critical slots.
- `11_diagnose_tensor_cache_visibility` now diagnoses `input_index` when present and reports id-mapping coverage.

## Training robustness and speed

- Fixed CUDA pin-memory thread failures by auto-testing pinning once and disabling `pin_memory` when the local PyTorch/CUDA build raises `CUDA error: invalid argument`.
- Added `--pin-memory` and `--no-pin-memory` overrides to `03_train.py`.
- Added stage-aware dataset loading so Stage A no longer loads huge response/witness/planner labels.
- Natural/representation stage no longer conditions the graph on ego candidates, matching the paper's scene-conditioned natural-alternative design and reducing compute.

## Model/loss safety

- Model forward prefers `cowp/critical/input_index` over Scenario `track_index`.
- Candidate macro ids are clamped to the embedding range.
- Response/witness/planner loss masks remain tied to model-visible critical slots.

## Validation

- `python -m compileall -q cowp`
- `pytest -q`: 20 passed
