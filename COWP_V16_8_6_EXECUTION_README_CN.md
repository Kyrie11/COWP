# COWP v16.8.6 下一步执行说明

## 结论先行

当前**不要直接全量重建**，也不要继续把旧 `tensor_cache_*_waymax_transport_v16_8` 当作 v16.8.6 paper-grade 主训练数据。

旧 cache 结构是健康的，适合旧 bank 饱和度诊断、旧版本复现和修复后的 matched baseline；但旧 bank 的 best-case PBTR floor 约为 0.5818，高于当前 0.45 gate，因此它已经是最终机制上限。与此同时，你本次上传的 proposal 压缩包不是完整 fresh probe：没有最终 `paired_proposal_probe.json`，fresh build 日志仍停在 768 labels 的旧异常。因此先用新 proposal 做便宜 screen，再决定是否花数天重建。

---

## Step 1 — 192-scene Priority-Commitment micro-probe（现在先跑这个）

```bash
cd /path/to/COWP_v16_8_6

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8

export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_micro_probe
export HARD_COUNT=64
export RANDOM_COUNT=128
export LABEL_WORKERS=24
export SEED=2026

bash NEXT_RUN_COMMANDS_V16_8_6_PRIORITY_COMMITMENT_MICRO_PROBE_CN.sh
```

关键输出：

```text
$PROBE_ROOT/paired_proposal_probe.json
$PROBE_ROOT/priority_commitment_micro_screen.json
$PROBE_ROOT/fresh_probe_profile_summary.json
```

micro screen 默认要求：

```text
pairing complete && build_error_count == 0
AnyValidSceneRate >= 0.97
AnyNCFSceneRate >= 0.38
FalseSafe proposal floor <= 0.58
PBTR proposal floor <= 0.52
old->new PBTR floor improvement >= 0.04
HardSceneNCFRecoveryRate >= 0.12
NCFLossRate <= 0.05
PHR generated scene rate >= 0.02
PHR priority-NCF scene rate >= 0.005
```

**如果 micro screen 失败：停止。不要 full rebuild。** 上传上述 3 个 JSON 给下一轮分析，重点检查 PHR 生成率、priority-NCF yield 和 rejection/profile。

---

## Step 2 — micro 通过后再跑严格 400+800 proposal probe

```bash
export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_proposal_probe
export HARD_COUNT=400
export RANDOM_COUNT=800
export LABEL_WORKERS=24
export SEED=2026

bash NEXT_RUN_COMMANDS_V16_8_6_PRIORITY_COMMITMENT_PROPOSAL_PROBE_CN.sh
```

严格 gate 仍要求 fresh proposal bank 满足：

```text
AnyValidSceneRate >= 0.99
AnyNCFSceneRate >= 0.40
BestCaseSelectedFalseSafeLowerBound <= 0.55
BestCasePBTRLowerBound <= 0.45
HardSceneNCFRecoveryRate >= 0.20
RMRTargetTTAErrorMax <= 0.20 s
no unexpected build errors
```

只有 `paired_proposal_probe.json` 中：

```text
promote_to_full_rebuild=true
```

才进入 Step 3。

---

## Step 3 — strict probe 通过后才做 full fresh rebuild

这一步重新计算 candidate/natural/response/witness/NCF 标签；它不能被旧 transport overlay 代替。但可以安全复用 WOMD index、复用旧 train/val scenario ID 集、关闭可选的 full-train candidate Waymax replay 和可视化 diagnostics。

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment

export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE=$SOURCE_DATA_ROOT/tensor_cache_train_waymax
export OLD_SCENESET_VAL_CACHE=$SOURCE_DATA_ROOT/tensor_cache_val_waymax

export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0
export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8
export AUG_WORKERS_TRAIN=12
export AUG_WORKERS_VAL=6

bash PREPARE_COWP_V16_8_6_DATA_FAST_CN.sh
```

该 wrapper 默认把 BLAS/TF 内部线程限制为 1，避免 24--32 个 Python worker 再各自开多线程造成 oversubscription；同时生成 build fingerprint、profile、alignment report 和 `data_manifest_v16_8_6.json`。

---

## Step 4 — fresh data 完成后先跑单 seed 机制链

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment
export OUT_ROOT=outputs/cowp_v16_8_6_priority_commitment_seed2026
export TRAIN_SEED=2026
export CUDA_VISIBLE_DEVICES=0,1
export BACKGROUND=0

bash NEXT_RUN_COMMANDS_V16_8_6_MECHANISM_CN.sh
```

不要手工把旧 cache 填进这个 wrapper。它会用 `53_gate_fresh_v16_8_6_cache_protocol.py` 检查 proposal provenance、fingerprint、manifest 和 `.transport_v16_8_6` sidecar。

只有 mechanism gate 真正通过才进入 Waymax。

---

## Step 5 — Waymax 先 100-scene probe，再 1000-scene full

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment
export OUT_ROOT=outputs/cowp_v16_8_6_priority_commitment_seed2026
export PROBE_SCENARIOS=100
export BACKGROUND=0
bash NEXT_RUN_COMMANDS_V16_8_6_PROBE_CN.sh
```

probe promotion 通过后：

```bash
export FULL_SCENARIOS=1000
export BACKGROUND=0
bash NEXT_RUN_COMMANDS_V16_8_6_FULL_CN.sh
```

之后再扩 2027/2028 两个 seed；不要在结构性 gate 尚未通过时提前烧三倍训练/闭环预算。

---

## 旧 cache 现在还能做什么

旧数据：

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export TRAIN_CACHE=$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8
export VAL_CACHE=$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8
```

可以用于：旧 bank 饱和度/negative-control、label-space oracle ablation、v16.8 RCOT/BCOT 复现、修复后的 matched baseline。

**不能用于：** v16.8.6 paper-grade COWP 主训练、PHR/BCS-RMR proposal-source ablation、v16.8.6 COWP Waymax paper claim。

若必须做旧-bank learned diagnostic，显式标记：

```bash
export ALLOW_LEGACY_V16_8_BANK=1
bash NEXT_RUN_COMMANDS_V16_8_6_LEGACY_BANK_DIAGNOSTIC_CN.sh
```

这个 wrapper 会把输出标记为非 paper-grade，不会绕过成正式闭环证据。

---

## `Load labels for tables` 加速后的重跑方式

旧 label-space ablation 仍可以重跑，但 proposal-source 部分会在 stale cache 上正确 SKIP，而不是伪造 RMR contribution。

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export LABELS_VAL=$DATA_ROOT/labels_val
export VAL_CACHE=$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8
export OUT_ROOT=outputs/cowp_v16_8_5_bcs_rmr_fast_seed2026
export LABEL_TABLE_LOAD_WORKERS=8

bash RUN_LABEL_AND_PROPOSAL_ABLATIONS_V16_8_5_CN.sh
```

第一次会写：

```text
$OUT_ROOT/eval/ablation/label_space/compact_label_table_cache.pkl
```

后续完全相同 labels 的重跑直接复用 compact cache。新 loader 不再加载 natural/response 大轨迹 bank，且 module-effect 不再重复第二遍 planner selection。

---

## 外部 baseline：旧结果作废；旧 bank 只做数值 smoke，最终必须 fresh-bank 重训

旧 GameFormer/DTPP 训练日志跳过了超过 99% batch，不能作为强论文 baseline。修复版把输入变换到 ego frame、修复 roadgraph padding、FP32 loss-side，并在 skip fraction >2% 时直接报错。

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export TRAIN_CACHE=$DATA_ROOT/tensor_cache_train_waymax_transport_v16_8
export VAL_CACHE=$DATA_ROOT/tensor_cache_val_waymax_transport_v16_8
export OUT_ROOT=outputs/external_baselines_v16_8_6_fixed_oldbank
export MODE=smoke

bash RUN_EXTERNAL_BASELINES_V16_8_6_FIXED_CN.sh
```

每个 epoch 必须看到：

```text
skip_fraction <= 0.02
```

这个旧-bank smoke 只验证“修复后不再大量 skip / loss 有限 / checkpoint 能正常训练”，**不要把其 learned Waymax 当最终对比**。本仓库的 GameFormer/DTPP 会对 candidate bank 打分，因此最终 paper baseline 要等 fresh rebuild 后，在同一 fresh proposal bank 上重新训练：

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment
export TRAIN_CACHE=$COWP_ROOT/tensor_cache_train_transport_v16_8_6
export VAL_CACHE=$COWP_ROOT/tensor_cache_val_transport_v16_8_6
export OUT_ROOT=outputs/external_baselines_v16_8_6_fixed_freshbank
export MODE=full
export RUN_ONLINE_EVAL=1

bash RUN_EXTERNAL_BASELINES_V16_8_6_FIXED_CN.sh
```

这样 COWP 和 matched GameFormer/DTPP 看到同一个 proposal space，避免 baseline 被旧 bank 人为压低。代码仓中的 GameFormer/DTPP 仍应在论文里写作 matched implementation baseline，除非后续严格对齐官方代码/权重与协议。
