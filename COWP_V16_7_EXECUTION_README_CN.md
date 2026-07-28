# COWP v16.7 执行说明

## 0. 解压与环境

```bash
unzip COWP_v16_7_optimized.zip
cd COWP_v16_7_optimized
```

## 1. 快速机制修复实验：复用 v16.6 natural

要求服务器上保留 v16.6 natural checkpoint 和 aligned attribution gate。

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
BACKGROUND=0 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
TRANSPORT_AMP=1 \
PLANNER_AMP=1 \
bash NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```

该步骤：

- 验证 v16.6 natural basis/effectiveness/attribution gate；
- 复用选中的 natural checkpoint；
- 从新 optimizer state 训练 v16.7 transport 和 planner；
- 在互斥 calibration/held-out split 上重新校准并验证 mechanism；
- 不运行 online Waymax。

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
bash CHECK_RUN_STATUS_V16_7.sh

python -m json.tool \
outputs/cowp_v16_7_mechanism_v9labels_seed2026/eval/learned_offline/mechanism_verification.json
```

必须重点检查：

- `pass=true`；
- `calibration_feasible=true`，而不是 `least_violation`；
- priority RootTransport/BCOT AUPRC；
- protected NCF recall 与 precision；
- accepted rate、fallback、PBTR improvement；
- global false-safe 没有反向恶化。

## 2. mechanism 通过后运行真实 online Waymax

先跑 100 场 probe，再跑 1000 场 full：

```bash
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
BACKGROUND=1 \
RUN_PROBE=1 \
PROBE_SCENARIOS=100 \
RUN_FULL=1 \
FULL_SCENARIOS=1000 \
ROLLOUT_HORIZON=80 \
bash NEXT_RUN_COMMANDS_V16_7_FULL_CN.sh
```

查看：

```bash
OUT_ROOT=outputs/cowp_v16_7_mechanism_v9labels_seed2026 \
bash CHECK_RUN_STATUS_V16_7.sh
```

不要在论文实验中设置 `ALLOW_QUALITY_GATE_FAILURE=1`。

## 3. 论文级 fresh-data rebuild

priority arrival-order、signal rule 和 map-screened candidates 已改变，最终论文结果不能只复用 v9 labels：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v17 \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
CUDA_VISIBLE_DEVICES=0 \
MAX_REPLAY_CANDIDATES=24 \
RUN_WAYMAX_REPLAY=1 \
bash PREPARE_COWP_V16_7_DATA_CN.sh
```

输出：

```text
/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax
/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax
/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax_transport_v17
/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax_transport_v17
```

## 4. fresh v17 natural recovery

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v17 \
RAW_TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax \
RAW_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax \
TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax_transport_v17 \
VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax_transport_v17 \
OUT_ROOT=outputs/cowp_v16_7_natural_recovery_seed2026 \
TRAIN_SEED=2026 \
BACKGROUND=0 \
STOP_AFTER_STAGE=natural \
RUN_NATURAL=1 RUN_TRANSPORT=0 RUN_PLANNER=0 RUN_OFFLINE=0 RUN_PROBE=0 RUN_FULL=0 \
FORCE_TRAIN=1 FORCE_EVAL=1 \
bash NEXT_RUN_COMMANDS_V16_7_CN.sh
```

## 5. fresh v17 aligned attribution

`RUN_NATURAL_ATTRIBUTION_V16_7_CN.sh` 会在缺少对齐 checkpoint 时，自动从同一 `INIT_CKPT` 将两个 ablation 训练到 main 选择的同一 epoch；无需先手工运行旧版消融脚本。直接执行：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v17 \
RAW_TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax \
RAW_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax \
MAIN_OUT_ROOT=outputs/cowp_v16_7_natural_recovery_seed2026 \
SOURCE_ABL_ROOT=outputs/cowp_v16_7_natural_ablations_seed2026 \
ATTR_OUT_ROOT=outputs/cowp_v16_7_natural_attribution_aligned_seed2026 \
FORCE_RETRAIN_ABLATIONS=1 \
bash RUN_NATURAL_ATTRIBUTION_V16_7_CN.sh
```

## 6. fresh v17 full pipeline

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_7_natural_recovery_seed2026 \
ATTR_GATE=outputs/cowp_v16_7_natural_attribution_aligned_seed2026/natural_component_attribution_gate.json \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v17 \
RAW_TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax \
RAW_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax \
TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_train_waymax_transport_v17 \
VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v17/tensor_cache_val_waymax_transport_v17 \
OUT_ROOT=outputs/cowp_v16_7_full_seed2026 \
BACKGROUND=0 FORCE_TRAIN=1 FORCE_EVAL=1 \
bash NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh
```

机制通过后再运行 `NEXT_RUN_COMMANDS_V16_7_FULL_CN.sh`。

## 7. 三 seed

对 `TRAIN_SEED=2026,2027,2028` 重复第 4--6 步；保持同一数据、同一 split、同一 calibration protocol。主表报告均值、标准差、paired bootstrap/seed-level CI，并单独报告 online Waymax 结果。
