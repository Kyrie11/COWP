# COWP v16.8.3 执行说明：先验证 proposal，再决定全量重建

## 唯一当前优先事项

不要先重训，不要先全量重建，不要继续调 BCOT budget。先运行：

```bash
cd /path/to/COWP_v16_8_3

WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OLD_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v16_8 \
PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_proposal_probe \
HARD_COUNT=400 RANDOM_COUNT=800 LABEL_WORKERS=24 SEED=2026 \
FORCE_REBUILD_PROBE=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_PROPOSAL_PROBE_CN.sh
```

该脚本只重建约 1200 个 validation labels，不训练模型、不跑 Waymax。

## Promotion 判定

读取：

```text
/data0/senzeyu2/dataset/COWP/formal_v16_8_3_proposal_probe/paired_proposal_probe.json
```

只有：

```text
promote_to_full_rebuild = true
pairing_completeness.complete = true
```

且三项均通过时，才进入 full rebuild：

```text
new.any_ncf_scene_rate >= 0.40
new.best_case_selected_false_safe_lower_bound <= 0.55
paired.hard_scene_ncf_recovery_rate >= 0.20
```

若为 false，停止。优先检查：

```text
current_proposal_ceiling.json -> proposal_source_stats / macro_stats
paired_proposal_probe.json -> scene_with_rmr_bcte_rate
paired_proposal_probe.json -> scene_with_rmr_bcte_ncf_rate
paired_proposal_probe.json -> ncf_loss_rate
```

## Probe 通过后：完整数据重建

```bash
cd /path/to/COWP_v16_8_3

WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
TRAIN_LIMIT=22000 VAL_LIMIT=5000 RUN_WAYMAX_REPLAY=1 \
MAX_REPLAY_CANDIDATES=24 CUDA_VISIBLE_DEVICES=0 \
bash PREPARE_COWP_V16_8_3_DATA_CN.sh
```

脚本使用 fingerprint 防止混合旧/新候选文件。fingerprint 不一致时必须换新的
`COWP_ROOT`，不要强行 resume。

## 完整数据完成后：训练

```bash
cd /path/to/COWP_v16_8_3

DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
OUT_ROOT=outputs/cowp_v16_8_3_rmr_bcte_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
TRANSPORT_EPOCHS=24 PLANNER_EPOCHS=16 \
CUDA_VISIBLE_DEVICES=0,1 BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_MECHANISM_CN.sh
```

本轮不重训 natural decoder；使用已通过 basis/effectiveness gate 的 checkpoint。

## Learned-offline gate

只有以下两个值同时为 true 才进入 Waymax：

```text
mechanism_verification.pass
mechanism_verification.calibration_feasible
```

若 calibration 输出 `status=proposal_infeasible`，说明新 bank 的理论 floor 仍不满足
约束，不能通过调 threshold 解决。

## Waymax probe/full

```bash
DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
OUT_ROOT=outputs/cowp_v16_8_3_rmr_bcte_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_PROBE_CN.sh
```

probe 合格后：

```bash
DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
OUT_ROOT=outputs/cowp_v16_8_3_rmr_bcte_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_FULL_CN.sh
```
