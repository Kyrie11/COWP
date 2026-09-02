# Safe regime v2 build notes

This branch changes safe/normal dataset construction so that safe shards are used for nominal preservation and no-safety-regression claims, not anti-oracle claims.

Key changes:

1. `assign_regimes()` now determines the scene-time `normal` anchor from the nominal prefix/sample. Deliberately generated candidate prefixes no longer make the nominal scene-time non-normal merely because their deployable recovery is low.
2. `dataset_quality.min_deployable_score_per_sample`, `dataset_quality.min_oracle_score_per_sample`, and `dataset_quality.drop_scene_time_if_under_min_quality` add strict safe-shard gates. With the recommended values `0.0, 0.0, true`, underfull scene-times are dropped instead of backfilled by low-quality candidates.
3. `prefix_macro_whitelist` lets safe shards restrict prefix candidates to nominal-like macros such as `nominal, keep, stabilize, perturb_nominal, brake`, while near/contact shards can still use the full macro bank.

Recommended build script:

```bash
bash tools/build_safe_v2.sh
```

Expected diagnostics after build:

- `artifact_fraction`: 0 or < 1%.
- `negative_deployable_fraction`: much lower than the previous ~0.41 safe shards.
- `oracle_recoverable_fraction`: clearly higher than the previous ~0.58.
- `future_count_mean`: no longer 3 because non-artifact targeted futures are included.
- `valid_root_count_mean`: no longer 2 because visible perturbation roots are enabled.
- `regimes.normal`: present and dominant in safe shards.
- `regimes.oracle_artifact`, `prefix_collision`, `prefix_contact`: 0 after the group gate.

Safe v2 should be reported as a nominal-preservation / no-regression shard. Keep deployable-vs-oracle recovery claims on near-contact/contact shards.
