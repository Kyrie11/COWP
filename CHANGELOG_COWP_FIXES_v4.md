# COWP v4 fixes

This patch addresses the latest Stage-A AMP training crash and tightens the natural-branch objective.

## Fixed

- Fixed AMP dtype mismatch in `cowp.models.losses._natural_source_distribution_loss`.
  - Under `--amp`, decoder logits can be fp16/bf16 while label weights remain fp32.
  - `scatter_add_` requires source and destination tensors to share dtype.
  - The source-distribution target is now accumulated in fp32 with matching dtype.
- Made natural set-coverage and mixture-NLL terms compute distances/log-probabilities in fp32 for AMP stability.
- Made natural priority expectation loss compute in fp32 for AMP stability.
- Updated natural branch supervision so OBS / neutral / priority-preserving branch minADE uses the configured paper-style relative weights: `obs_prediction`, `neutral`, and `priority_rule`.

## Still preserved from v3

- Runtime safety for old tensor caches with out-of-range or invisible critical agents.
- New-cache support for `cowp/critical/track_id -> cowp/critical/input_index` mapping through WOMD `state/id`.
- Stage-aware dataset loading.
- `--no-pin-memory` / pin-memory compatibility guard.
- AMP GradScaler compatibility across PyTorch versions.

## Validation

- `python -m compileall -q cowp`
- `pytest -q` -> 21 passed
