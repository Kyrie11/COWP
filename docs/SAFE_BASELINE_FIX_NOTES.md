# Safe baseline fix notes

This patch fixes a DDP unused-parameter failure observed during `scripts/run_safe_regime_external_baselines.sh`.

## Root cause

The BeTopNet-lite adapter returns topology decoder outputs at every fusion layer:

- `actor_topo_logits_levels`
- `map_topo_logits_levels`

The previous training loss used only the final `actor_topo_logits` and `map_topo_logits`. Therefore the earlier topology decoder parameters did not receive gradients in DDP, which caused PyTorch to abort on the next iteration with `Expected to have finished reduction in the prior iteration`.

## Fix

- `src/ocrap/external_baselines/train.py` now applies BeTop focal topology loss to every topology decoder level, not only the final level.
- DDP construction now supports `find_unused_parameters: auto`; auto enables unused-parameter detection for BeTop-style models while keeping the default efficient path for other models.
- Safe baseline YAML configs now include `find_unused_parameters: auto`.

## Checked

- `python -m compileall -q src/ocrap` passes.
- Wayformer-style BC, GameFormer-level-k, and BeTopNet-lite all have finite forward/loss/backward in smoke checks, and all trainable parameters receive gradients in the checked configurations.
