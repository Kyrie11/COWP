# COWP v16.8.4：候选库审计、三臂配对 Probe 与证书引导精化

本包基于已上传的论文、建议文档和 `cowp_v16_8_3_rmr_bcte_seed2026` 结果包生成。

## 重要边界

本轮附件中没有 COWP 主源码压缩包、原始 `ALGORITHM_CHANGELOG.md`、执行 README 或 checkpoint 权重。结果包只有配置、日志、评估 JSON、训练 history，以及源码文件哈希。因此本包没有伪造对主仓库的逐行补丁：

- 已实现：缓存 schema 探测、fixed-bank 理论下界审计、严格配对三臂 proposal probe、promotion 判定、证书引导 proposal refinement 独立模块。
- 尚未集成：把 `cowp_extensions/certificate_guided_refinement.py` 接入你主仓库的 candidate generator、label engine 和 online planner。
- 不作声明：RMR-BCTE 已通过 gate、闭环指标提升、达到 SOTA。

## 0. 环境变量

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=$DATA_ROOT
export RAW_TRAIN_CACHE=$DATA_ROOT/tensor_cache_train_waymax
export RAW_VAL_CACHE=$DATA_ROOT/tensor_cache_val_waymax
export TRAIN_CACHE=$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8
export VAL_CACHE=$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SCENARIO_VAL="$WOMD_ROOT/uncompressed/scenario/validation/*.tfrecord*"
export PROBE_ROOT=$DATA_ROOT/proposal_probe_v16_8_4
mkdir -p "$PROBE_ROOT"
```

## 1. 先分析当前缓存 schema 与 fixed-bank 上限

从本包根目录执行：

```bash
python tools/probe_cache_schema.py \
  --cache-dir "$VAL_CACHE" \
  --sample 16 \
  --output "$PROBE_ROOT/current_val_schema.json"

python tools/analyze_proposal_cache.py \
  --cache-dir "$VAL_CACHE" \
  --sample 0 \
  --promotion-config configs/proposal_promotion_v16_8_4.yaml \
  --output "$PROBE_ROOT/current_val_proposal_audit.json"
```

如果第一个命令提示字段歧义，把报告中的真实 key 通过如下参数显式传给两个脚本：

```text
--valid-key
--conventional-safe-key 或 --conventional-unsafe-key
--ncf-key
--priority-eligible-key
--priority-ncf-key
--source-key
```

脚本拒绝把 pair/root tensor 静默压成 candidate label，避免得到“看起来合理但语义错误”的 gate。

## 2. 构建 400 hard + 800 unbiased-random 的配对 index

```bash
python tools/select_probe_scenarios.py \
  --old-cache "$VAL_CACHE" \
  --index-jsonl "$COWP_ROOT/index_val.jsonl" \
  --hard-count 400 \
  --random-count 800 \
  --seed 2026 \
  --output-index-jsonl "$PROBE_ROOT/probe_index_val.jsonl" \
  --output-manifest "$PROBE_ROOT/probe_manifest.json"
```

`hard_ids` 仅用于恢复率；总体 proposal rate 只使用 `random_ids`，两者不能直接混合计算总体比例。

## 3. 构建两个新 label-only arm

两个 arm 使用相同的 jerk 修复和其他配置，只改变可达冲突区域数：

- 单区域控制组：`configs/label_cowp_v16_8_4_single_region_control.yaml`
- RMR-BCTE 组：`configs/label_cowp_v16_8_4_rmr_bcte.yaml`

```bash
python -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml \
  --label-config /path/to/COWP_v16_8_4_audit_refinement/configs/label_cowp_v16_8_4_single_region_control.yaml \
  --proto-glob "$SCENARIO_VAL" \
  --output-dir "$PROBE_ROOT/labels_single_region" \
  --index-jsonl "$PROBE_ROOT/probe_index_val.jsonl" \
  --limit 1200 \
  --num-workers 24 \
  --start-method forkserver \
  --max-pending-multiplier 2 \
  --no-compress \
  --skip-diagnostics \
  --cpu-only

python -m cowp.scripts.01_build_labels_from_proto \
  --data-config configs/data.yaml \
  --label-config /path/to/COWP_v16_8_4_audit_refinement/configs/label_cowp_v16_8_4_rmr_bcte.yaml \
  --proto-glob "$SCENARIO_VAL" \
  --output-dir "$PROBE_ROOT/labels_rmr_bcte" \
  --index-jsonl "$PROBE_ROOT/probe_index_val.jsonl" \
  --limit 1200 \
  --num-workers 24 \
  --start-method forkserver \
  --max-pending-multiplier 2 \
  --no-compress \
  --skip-diagnostics \
  --cpu-only
```

必须使用全新输出目录。不要在旧目录上 `--skip-existing`，否则会混合修复前后的标签。

## 4. 三臂配对判定

```bash
python tools/compare_proposal_caches.py \
  --old-cache "$VAL_CACHE" \
  --control-cache "$PROBE_ROOT/labels_single_region" \
  --new-cache "$PROBE_ROOT/labels_rmr_bcte" \
  --manifest "$PROBE_ROOT/probe_manifest.json" \
  --promotion-config configs/proposal_promotion_v16_8_4.yaml \
  --output "$PROBE_ROOT/paired_proposal_probe_v16_8_4.json"
```

只在以下两个字段同时为 `true` 时进入全量重建：

```text
promote_to_full_rebuild
algorithm_increment_demonstrated
```

前者判断最终候选库是否有希望通过 mechanism gate；后者判断 RMR-BCTE 相对同 jerk 语义的单区域控制是否有独立算法增量。

## 5. 通过后才进行的工作

1. 用 RMR-BCTE 配置重建 train/val labels 和 tensor cache，使用新目录与 build fingerprint。
2. Waymax attached outcomes 不需要先对全部训练候选重放；自然/transport/planner 的 label-based 阶段可以先训练。
3. `logdiv` 当前没有有效监督，保持禁用。
4. 复用已通过 natural checkpoint，优先重训 transport 与 planner；若新标签改变 natural-root tensors，才重新跑 natural gate。
5. 在独立 calibration/held-out 上重新 calibration 与 mechanism gate。
6. gate 通过后再跑真实 online Waymax closed loop；论文最终 burden/causal claim 仍需 reactive-agent 与 human-audited stress set。

## 6. 本地回归

```bash
python -m compileall tools cowp_extensions
pytest -q
```
