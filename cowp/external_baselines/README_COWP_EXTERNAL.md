# COWP external baselines: GameFormer and DTPP adapters

This patch adds COWP tensor-cache adapters and training/evaluation scripts for two external learning-based planners:

- `gameformer`: hierarchical level-k Transformer prediction/planning baseline.
- `dtpp`: query-centric ego-conditioned prediction plus learned cost-evaluation baseline.

The code reuses COWP tensor caches and the online Waymax evaluation path. It does not change the COWP model.

## Files

- `cowp/external_baselines/adapters.py`: converts COWP npz tensor-cache batches to GameFormer/DTPP-style tensors.
- `cowp/external_baselines/gameformer_cowp.py`: GameFormer core encoder, level-0 decoder, level-k interaction decoders, GMM heads, and loss.
- `cowp/external_baselines/dtpp_cowp.py`: DTPP-style Transformer encoder/decoder, ego-plan query, score/cost decoder, and loss.
- `cowp/external_baselines/waymax_policy.py`: external baseline policy wrapper for real Waymax closed-loop rollout.
- `cowp/scripts/20_train_external_baseline.py`: trains one external baseline on COWP cache.
- `cowp/scripts/21_eval_external_baseline.py`: evaluates one external baseline offline or online in Waymax.
- `run_external_baselines.sh`: end-to-end driver similar to the COWP training/evaluation shell script.

## Smoke run

```bash
MODE=smoke BASELINES="gameformer dtpp" RUN_TRAIN=1 RUN_OFFLINE_EVAL=1 RUN_ONLINE_EVAL=1 \
  bash run_external_baselines.sh
```

## Full run

```bash
MODE=full BASELINES="gameformer dtpp" RUN_TRAIN=1 RUN_OFFLINE_EVAL=1 RUN_ONLINE_EVAL=1 \
  TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax \
  VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax \
  WAYMAX_VAL=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150 \
  bash run_external_baselines.sh
```

## Notes

- The learned-offline evaluation uses the same candidate labels and attached Waymax candidate outcomes as COWP.
- The online evaluation calls the real Waymax closed-loop rollout path and generates candidates at every step using COWP's existing online candidate generator, then scores them with the external model.
- The COWP cache diagnostics show log-divergence labels are missing/degenerate in the current safety-only replay, so this patch does not train or report log-divergence as a main external-baseline objective.

## V2 raw-WOMD / performance note

For the causal closed-loop data boundary, speed profiler, fidelity labels and current commands, see the repository-root file `README_5_SOTA_BASELINES_COWP_V2_RAW_WOMD_SPEED_CN.md`.  In particular, final Waymax rollout uses raw WOMD 1.3.1 validation TFExamples; NPZ files remain training/mechanism caches, and external policies must not consume logged future states.
