# RC-NCF Patch Usage

## Files added/changed

- `cowp/models/coordinate.py`: ego-centric SE(2) normalization and SDC identity.
- `cowp/models/natural_decoder.py`: exposes root-scene natural latent without dense trajectory decoding during planner inference.
- `cowp/models/witness_decoder.py`: cross-world natural-conditioned evidential witness.
- `cowp/models/losses.py`: balanced/ranking/candidate-consistency/evidential losses.
- `cowp/waymax_eval/policy_wrapper.py`: kinematically consistent control, Frenet lane changes, critical/token caps, lane-only map checks.
- `cowp/scripts/17_merge_waymax_shards.py`: correct shard merge.
- `cowp/scripts/18_calibrate_witness_threshold.py`: validation threshold calibration.
- `run_rc_ncf_2gpu.sh`: end-to-end two-GPU pipeline.
- `tests/test_rc_ncf_patches.py`: regression tests for new functionality.

## Full run

```bash
cd COWP_RC_NCF
bash run_rc_ncf_2gpu.sh
```

## Common variants

```bash
# Caches already contain rc24 safety+logdiv outcomes
RUN_OUTCOME_REPLAY=0 bash run_rc_ncf_2gpu.sh

# Evaluate an existing trained checkpoint/output tree
RUN_TESTS=0 RUN_OUTCOME_REPLAY=0 RUN_TRAIN=0 RUN_EVAL=1 bash run_rc_ncf_2gpu.sh

# Fast online smoke run
TOTAL_ONLINE_SCENARIOS=100 bash run_rc_ncf_2gpu.sh

# Re-run full online evaluation only
RUN_TESTS=0 RUN_OUTCOME_REPLAY=0 RUN_TRAIN=0 RUN_EVAL=1 \
TOTAL_ONLINE_SCENARIOS=2000 FORCE_EVAL=1 bash run_rc_ncf_2gpu.sh
```

All path variables can be overridden from the environment: `WOMD_ROOT`, `COWP_ROOT`, `BASE_TRAIN_CACHE`, `BASE_VAL_CACHE`, `TRAIN_CACHE`, `VAL_CACHE`, `OUT_ROOT`, and `WAYMAX_VAL`.
