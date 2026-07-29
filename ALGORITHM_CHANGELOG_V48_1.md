# v48.1 Existing-Data-First / Calibration Isolation

- Decision changed from mandatory train/contact rebuild to existing-data-first screening.
- Added scene-disjoint 50/50 proxy calibration/development split for Safe, Near and Contact.
- Test roots are not read by the screening controller.
- Added expensive rebuild double opt-in guards.
- Teacher-PCD coverage target supports strict/warn/off; screening defaults to warn with a zero-opportunity hard floor.
- Added dataset role manifest, overlap audits and screening status output.
- Added dedicated calibration construction from the reserved tail of standard WOMD validation.
- Added Safe dedicated calibration and scene-count gates.
- Added no-retraining dedicated recalibration script.
- Tuned two-concurrent-job CPU defaults for 2×A30 + Xeon Gold 5220R: 6 workers/job, prefetch 2, 4 intra-op threads.

## 2026-07-24 regression completion

- Restored tri-state dead-zone handling: tied teacher-PCD candidates are no longer forced negative.
- Restored observation-consistent router pooling (`shared_raw`, `ego_shared_raw`) and robust-routing aliases.
- Restored exact teacher-PCD sampler helper and checkpoint/inference router-pooling persistence.
- Reconnected configurable macro/variant scheduling to candidate generation.
- Fixed closed-loop aggregation of embedded max/min metric names and explicit collision/offroad aliases.
- Final regression: 112 tests passed; compile and shell syntax checks passed.

## v48.1.1 manifest preflight repair

- Added `tools/ensure_manifest_v48.py` and `tools/manifest_repair_v48.py`.
- Existing NPZ samples can now reconstruct a missing/stale `manifest.csv` without rebuilding datasets.
- `run_v48_two_gpu_fast_commands.txt` performs this metadata-only preflight before scene-overlap auditing and proxy calibration splitting.
- Reconstruction is atomic and aborts if the dataset changes during scanning, avoiding a silent partial snapshot of an active build.

## v48.2 audit correction

The uploaded v48.1 screening logs revealed a pre-epoch `NameError` (`os` not imported) and a sampler configuration-key mismatch. Consequently, v48.1 did not experimentally validate its model-level modifications. These defects and the follow-on OC-TRAC-SRC algorithm are documented in the root `ALGORITHM_CHANGELOG.md`.
