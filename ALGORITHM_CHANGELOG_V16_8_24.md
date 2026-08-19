# COWP v16.8.24 — Rebuild-readiness / execution-chain repair

This release does **not** change the Scenario→COWP label semantics validated by the v16.8.23 64-scene bitwise/array semantic-equivalence benchmark. It repairs the build/replay execution chain and removes avoidable full-split rescans.

## Fixes

1. Replaced the nonexistent `cowp.scripts.44_summarize_label_build_profile` with the existing `cowp.scripts.49_summarize_label_build_profile` in the supported benchmark/full-build paths.
2. Added `REPORT_ONLY=1` benchmark recovery so a completed v23 benchmark can regenerate `benchmark_report.json` without rebuilding labels.
3. Added `77_audit_active_execution_chain.py`, which statically verifies all Python modules, configs and shell entrypoints used by the supported v16.8.24 chain before a long job starts.
4. Scenario-location indexes now default to the persistent WOMD-local cache `WOMD_ROOT/.cowp_v131_indices`; benchmark and full build therefore reuse the same full-split scan.
5. Full-build split selection no longer requires a sufficiently large historical COWP cache. It prefers a promoted historical scene set when available and deterministically falls back to the authoritative WOMD Scenario index.
6. Worker defaults are physical-core aware: at most 40 label workers and 12 tensor-cache workers; BLAS/OpenMP nested threads are pinned to one.
7. Waymax outcome attachment now covers **train, internal validation, and held-out test**, rather than only train/val.
8. Official WOMD blind test remains excluded from future-dependent COWP mechanism labels. Held-out mechanism test is a scenario-disjoint subset of WOMD validation, whose future ground truth is available.

## Validation

- Uploaded v23 benchmark semantic equivalence: 64/64 scenes, PASS, zero unexpected tensor mismatches.
- Recovered matched worker-time speedup vs the historical v21 pilot profile: ~2.185x.
- Supported v16.8.24 active execution-chain audit: PASS.
- v16.8.24 rebuild-readiness unit tests: 5/5 PASS.
