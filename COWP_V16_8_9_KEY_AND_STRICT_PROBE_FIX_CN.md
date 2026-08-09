# COWP v16.8.9：Smoke Contract Repair 与 Strict Probe 工程修复

## 结论

本轮发现的两个问题均为工程工具链错误，不改变 candidate / NCF / PBTR / causal relevance / response / witness 的算法语义。

1. `61_repair_v16_8_9_audit_transport_contract.py` 错误地假设所有 NPZ 都使用 tensor-cache 的 `__` key 编码；实际上 `01_build_labels_from_proto` 产生的 label NPZ 保留 `/` key。修复工具现同时支持两种编码并保持原文件风格。
2. strict wrapper 将随机 ID 文件路径保存在 Bash 特殊变量 `RANDOM` 中。`$RANDOM` 每次读取会产生新的整数，因此 `--random-scene-ids` 被传成数字文件名，下一次 `cat` 又得到另一个数字并在 `set -e` 下终止。现改为 `RANDOM_IDS_FILE`。
3. strict wrapper 在 required stage 失败时现在也会生成 `v16_8_9_strict_verdict.json`，其 `recommend_full_rebuild=false` 并记录 `failure_stage`，避免“没有 verdict”被误解为算法结论。

## 当前数据性质判断

v16.8.9 96-scene smoke 中无偏 48 scenes 的点估计：AnyNCF=0.4167、false-safe floor=0.50、PBTR floor=0.4419、hard recovery=0.2083；proposal point gates 已通过。之前的 transport mismatch 属于 root-level serialization contract 错误，不改变 candidate NCF。

因此数据设计存在正向信号，但 48 random scenes 的统计不确定性仍不足以授权数日 full rebuild。必须用修复后的 400 hard + 800 random strict probe。

## 执行顺序

### A. 先修当前 96-scene smoke，不重建 Scenario labels

```bash
cd /path/to/COWP_v16_8_9_fixed
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export SMOKE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_smoke
bash RECOVER_V16_8_9_CAUSAL_AUDIT_SMOKE_CONTRACT_CN.sh
```

只有 `v16_8_9_smoke_verdict.json` 中 `recommend_strict_probe=true` 才继续。

### B. 重跑 strict probe

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe_contract_fixed
export HARD_COUNT=400
export RANDOM_COUNT=800
export LABEL_WORKERS=24
export SEED=2026
export FORCE_REBUILD_PROBE=1
bash NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh
```

这次应生成：

- `representative_random_scene_ids.txt`
- `labels_val_v16_8_9/*.npz`
- `fresh_probe_profile.jsonl`
- `paired_proposal_probe.json`
- `proposal_source_ablation.json`
- `causal_audit_diagnostic.json`
- `training_supervision_audit.json`
- `v16_8_9_strict_verdict.json`

仅当 `recommend_full_rebuild=true` 才执行 full rebuild。

### C. strict PASS 后 full rebuild

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_contract_fixed
export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE=$SOURCE_DATA_ROOT/tensor_cache_train
export OLD_SCENESET_VAL_CACHE=$SOURCE_DATA_ROOT/tensor_cache_val
export STRICT_VERDICT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe_contract_fixed/v16_8_9_strict_verdict.json
export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0
export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8
bash PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
```

## 数据完备性

Fresh label generator 已显式序列化 candidate provenance、stable critical agents、natural roots、candidate-conditioned causal relevance、root affected/unsafe/direct burden、responses、witness/pair NCF/blocker、inline RootTransport。Tensor cache 构建再加入 WOMD model/state/map tensors。训练 loss 已消费 `pair_relevant` 与 `mode_affected`，因此 schema 层面能够支撑当前 learned architecture。

最终训练前仍必须依靠 `60_verify_fresh_v16_8_9_cache.py`、`62_audit_training_supervision.py` 和 full-validation proposal ceiling gate，确认字段不仅存在，而且正负监督不退化。

## 构建加速原则

目前不要为了速度缩减 natural roots、response search 或 witness search，这些会改变 ground truth。保留的无损优化包括：精确复用旧 scene-ID set、producer allowlist 过滤、BLAS/TF 单线程 worker、inline transport、跳过 full-train cached Waymax replay、关闭构建期 visualization，并使用 fingerprint-safe resume。
