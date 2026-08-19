# COWP v16.8.23 — Fast Full-Rebuild / Compact Evidence Dataset

## Purpose
v16.8.22 already passes strict and train-pilot six-layer support gates. v16.8.23 does **not** relax the mechanism-support contract. It makes the next full rebuild cheaper and safer by (1) adding semantics-preserving label fast paths, (2) forcing an exact NPZ equivalence benchmark before a large build, and (3) fixing the v16.8.22 compact-size bug in historical-scene-set reuse mode.

## Main engineering changes
- Added boolean-only unsafe predicates to avoid materializing event masks/min-distance diagnostics in inner-loop screening.
- Precompute candidate-independent safe-response burden terms and use an exact early-stop path for Top-B safe budget search.
- Precompute root-recovery safe-case burden and support exact `min_only` first-safe search for witness construction.
- Cache candidate/root time-to-arrival values for conflict-region localization instead of recomputing TTA for every positive pair.
- Added fast-path flags to `configs/label_cowp_v16_8.yaml`; a reference config with the new fast paths disabled is retained.
- Added `BENCHMARK_V16_8_23_FASTPATHS_CN.sh`: 64-scene default, exact tensor equivalence against existing v16.8.21/v22 labels, plus matched profile comparison when the historical profile is available.
- Added `PREPARE_COWP_V16_8_23_FAST_DATA_CN.sh`: compact train/val/heldout-test build with hard split caps and post-build six-layer audits.
- Added `NEXT_EXECUTION_V16_8_23_CN.sh` wrapper; full-core is blocked until the benchmark passes semantic equivalence (and >=1.25 matched worker-time speedup when measurable, unless explicitly overridden).

## Size fix
In v16.8.22, `REUSE_OLD_SCENE_SET=1` wrote every historical NPZ ID into train/val allowlists and then omitted `--limit`; therefore `TRAIN_LIMIT`/`VAL_LIMIT` did not cap the build. v16.8.23 samples the requested number of historical scenario IDs first, so caps apply in reuse mode.

## Compact split policy
Default: train=6000, internal validation=1200, heldout test=1500. Train comes from official WOMD training. Internal validation and heldout test are deterministic scenario-disjoint subsets of official WOMD validation. The official WOMD test split is not used to generate COWP transport/witness labels because its future ground truth is hidden.

## Validation performed in this environment
- `python -m py_compile` on modified Python modules: PASS.
- `bash -n` on benchmark/full-builder/wrapper: PASS.
- Targeted support/fast-path tests: 31 passed.
- No WOMD full build was run here; actual speedup must be measured on the user's machine by the benchmark before full-core.
