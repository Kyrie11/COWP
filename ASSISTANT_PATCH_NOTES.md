# Assistant patch notes

This patch fixes three implementation issues found while auditing the current uploaded paper/code/results:

1. **CLI override preservation**: common flags such as `--set` are accepted before and after the subcommand. Previously, putting `--set` before the subcommand could be silently overwritten by the subparser defaults.
2. **DRS metric masking**: `deployable_recovery_success` now supports `root_valid` and masks padded/invalid roots before normalizing root probabilities. `evaluate` passes `root_valid` when present.
3. **Calibration auditability**: `calibrate` now reports `negative_fraction` and `valid_for_fra_calibration`; when no negative deployability samples exist it explicitly warns that the calibrated threshold cannot control FRA.

Tests added:
- `tests/test_cli_overrides_and_metrics.py`

Validation:
- `PYTHONPATH=src pytest -q` passes: 30 tests.
