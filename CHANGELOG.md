# Changelog

## 0.2.0 - OC-RAP paper-aligned implementation

### Added

- `src/ocrap` package layout with dedicated `cli`, `config`, `data`, `planning`, `simulation`, `roots`, `models`, `algorithms`, `evaluation`, and `utils` modules.
- Pure Python TFRecord reader for WOMD shards, including CRC validation, gzip support, glob expansion, shard iteration, and resume state.
- SDC-first WOMD parser with metadata for original SDC index and agent remapping.
- Expanded agent, map, dynamic-map, route, BEV observation, prefix, future, root, teacher, and sample schemas.
- Route-local candidate prefix generation with controls `[accel, steer, jerk, steer_rate]`, feasibility checks, and macro diversity.
- Replay, reactive IDM-style route-following, and targeted hidden-spawn future generators.
- Seven-channel BEV observation renderer with dynamic occlusion shadows and observation compatibility labels.
- Recovery-option teacher with controller-dependent rollouts and active-mask corrected margins.
- Recovery-signature root clustering and lower-tail LCVaR root aggregation.
- Corrected OC-MERO implementation using `normalize(C_ij * p_j)` observation-consistent weights.
- `papercheck` CLI and implementation for artifact, hidden-spawn, candidate, root, and observation-compatibility validation.
- Train, calibrate, evaluate, deploy, diagnose, and dataset-build CLIs.
- Unit tests and smoke tests for algorithms, WOMD reader/parser, observation, future mining, split stability, builder, and papercheck.

### Changed

- Removed old TensorFlow runtime parser path from the WOMD ingestion code.
- Replaced Python `hash()`-based randomness with SHA1-based deterministic seeding.
- Made artifact regime detection depend on observation/unknown route corridor geometry rather than hidden metadata.
- Default root margin aggregation is now LCVaR instead of mean.
- README commands now use `PYTHONPATH=src` and the package CLI `python -m ocrap.cli`.

### Fixed

- Fixed package-shadowing risk from the old root-level `ocrap` layout.
- Fixed degenerate dynamic occlusion geometry by using an ego-centered grid containing `(0, 0)`.
- Fixed OC-MERO masking and removed unsafe unconditional `R_orc >= R_dep` assumptions.
- Fixed hidden-agent spawning so it is restricted to unknown, drivable, non-visible cells and delayed after prefix execution.
- Fixed paper artifact fixture generation so artifact fraction, negative deployability, oracle recoverability, hidden emergence counts, and off-diagonal observation compatibility are all non-degenerate.

## v48.1

See `ALGORITHM_CHANGELOG_V48_1.md`. The default workflow now reuses the existing datasets, isolates proxy calibration from development validation by scene, seals test roots, and permits later dedicated recalibration without retraining.
