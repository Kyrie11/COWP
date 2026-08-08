# COWP v16.8.7：当前恢复、数据决策与后续重建指令

## 结论先行

1. 当前 `tensor_cache_*_waymax_transport_v16_8` 报错的根因是 **broken symlink**：顶层 NPZ 是指向已删除 `tensor_cache_*_waymax` 的链接，不是实体文件。
2. v16.8.6 micro probe 的 fresh label 阶段已经完成：64 hard + 128 random 有 1 个重合，所以是 **191 unique scenes**；191 个全部 `written`，无需重跑。
3. 当前先用 `tensor_cache_val` 恢复 paired compare；不要重建 fresh labels，更不要直接 full rebuild。
4. 若最终 strict 1200-scene probe 通过，paper-grade fresh rebuild 必须从 WOMD Scenario proto 重新生成 labels；旧 `tensor_cache_train/val` 只用作 scenario-ID allowlist。
5. 新 full-build 不再创建 transport overlay；transport 直接内嵌 fresh tensor cache。

## A. 先确认 broken symlink（只诊断）

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal

ls -l "$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8/10040e572b831a04.npz"
readlink "$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8/10040e572b831a04.npz" || true
test -L "$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8/10040e572b831a04.npz" && echo symlink
test -e "$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8/10040e572b831a04.npz" || echo BROKEN_LINK

echo "base val file:"
ls -l "$COWP_ROOT/tensor_cache_val/10040e572b831a04.npz"
```

## B. 现在最应该执行：恢复已经完成的 191-scene micro probe

**不要重新执行 fresh label build。**

```bash
cd /path/to/COWP_v16_8_7

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_micro_probe

bash RECOVER_V16_8_6_PRIORITY_MICRO_PROBE_FROM_BASE_CN.sh
```

关键文件：

```text
$PROBE_ROOT/paired_proposal_probe.json
$PROBE_ROOT/priority_commitment_micro_screen.json
```

如果 `screen_pass=false`：不要 1200 probe，不要 full rebuild。

如果 `screen_pass=true`：进入严格 400+800 probe。

## C. micro 通过后：严格 1200-scene probe

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_proposal_probe
export HARD_COUNT=400
export RANDOM_COUNT=800
export LABEL_WORKERS=24
export SEED=2026
export FORCE_REBUILD_PROBE=0

bash NEXT_RUN_COMMANDS_V16_8_6_PRIORITY_COMMITMENT_PROPOSAL_PROBE_CN.sh
```

只有 `paired_proposal_probe.json` 中：

```text
promote_to_full_rebuild = true
```

才允许 full rebuild。正式阈值仍为：AnyValid >= 0.99、AnyNCF >= 0.40、false-safe floor <= 0.55、PBTR floor <= 0.45、hard recovery >= 0.20、RMR TTA max error <= 0.20 s。

## D. 如果仍需复用旧 v16.8 transport 数据做 legacy diagnostic

### 优先：复用现有 sidecar，rebase 到 surviving base cache

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
bash REPAIR_LEGACY_V16_8_TRANSPORT_OVERLAYS_CN.sh
```

得到：

```text
tensor_cache_train_transport_v16_8_rebased
tensor_cache_val_transport_v16_8_rebased
```

使用时必须：

```bash
export TRAIN_CACHE=$COWP_ROOT/tensor_cache_train_transport_v16_8_rebased
export VAL_CACHE=$COWP_ROOT/tensor_cache_val_transport_v16_8_rebased
export USE_WAYMAX_OUTCOME_LABELS=0
```

它们只用于旧-bank RCOT/BCOT/selector diagnostic，不是 v16.8.6 paper-grade 数据。

### 如果旧 hidden transport sidecar 也缺失/不完整

无需先恢复 Waymax cache，直接从 base tensor cache 重算 transport：

```bash
python -u -m cowp.scripts.26_augment_transport_labels \
  --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml \
  --input-dir "$COWP_ROOT/tensor_cache_train" \
  --output-dir "$COWP_ROOT/tensor_cache_train_transport_v16_8_recomputed" \
  --num-workers 12 --chunksize 2 --storage-mode overlay \
  --sidecar-subdir .transport_v16_8 --force

python -u -m cowp.scripts.26_augment_transport_labels \
  --data-config configs/data.yaml --label-config configs/label_cowp_v16_8.yaml \
  --input-dir "$COWP_ROOT/tensor_cache_val" \
  --output-dir "$COWP_ROOT/tensor_cache_val_transport_v16_8_recomputed" \
  --num-workers 6 --chunksize 2 --storage-mode overlay \
  --sidecar-subdir .transport_v16_8 --force
```

## E. strict probe 通过后的 paper-grade full rebuild

这里**不能从旧 tensor cache 补 PCHR/BCS-RMR 标签**。需要从 Scenario proto fresh build；但 index 和 scene set 可复用。

```bash
cd /path/to/COWP_v16_8_7

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_7_priority_commitment

export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE=$SOURCE_DATA_ROOT/tensor_cache_train
export OLD_SCENESET_VAL_CACHE=$SOURCE_DATA_ROOT/tensor_cache_val

export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0
export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8

bash PREPARE_COWP_V16_8_6_DATA_FAST_CN.sh
```

新数据的正式训练路径是自包含 cache：

```text
RAW_TRAIN_CACHE=$COWP_ROOT/tensor_cache_train
RAW_VAL_CACHE=$COWP_ROOT/tensor_cache_val
TRAIN_CACHE=$COWP_ROOT/tensor_cache_train
VAL_CACHE=$COWP_ROOT/tensor_cache_val
USE_WAYMAX_OUTCOME_LABELS=0
```

build 会在训练前自动做：完整 scene-set 对齐、实体 NPZ 检查、proposal provenance、witness/transport required keys、Waymax-ready state、scenario-id 一致性，以及完整 validation proposal ceiling gate。

## F. fresh dataset 通过后主实验

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_7_priority_commitment
export RAW_TRAIN_CACHE=$DATA_ROOT/tensor_cache_train
export RAW_VAL_CACHE=$DATA_ROOT/tensor_cache_val
export TRAIN_CACHE=$DATA_ROOT/tensor_cache_train
export VAL_CACHE=$DATA_ROOT/tensor_cache_val
export USE_WAYMAX_OUTCOME_LABELS=0
export OUT_ROOT=outputs/cowp_v16_8_7_priority_commitment_seed2026
export TRAIN_SEED=2026
export CUDA_VISIBLE_DEVICES=0,1
export BACKGROUND=0

bash NEXT_RUN_COMMANDS_V16_8_6_MECHANISM_CN.sh
```

mechanism gate 通过后再 Waymax probe/full，不要跳级。
