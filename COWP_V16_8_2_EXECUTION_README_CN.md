# COWP v16.8.2 分阶段执行说明

## 目标

执行链严格区分四件事：

1. 用旧 checkpoint 重评估，隔离评估/selector/fallback 修复；
2. 在旧 overlay 上重训 corrected transport/planner；
3. 必要时重建 fresh BCTE 候选与 Waymax cache；
4. 只有两个 mechanism gate 同时通过后才运行 probe/full。

不要覆盖旧 `OUT_ROOT`，不要从旧 JSON 继续 calibration。

---

## A. 最优先：重评估 v16.8.1 checkpoint

服务器必须仍存在：

- natural checkpoint/history；
- `checkpoints/transport/cowp_witness_best.pt`；
- `checkpoints/planner/cowp_planner_best.pt`。

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

SOURCE_RUN=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_2_reeval_v9base_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_REEVAL_CURRENT_CN.sh
```

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_8_2_reeval_v9base_seed2026 \
bash CHECK_RUN_STATUS_V16_8_2.sh
```

必须确认新结果包含：

```text
CertificateSemantics/Version = v16_8_2_decoupled
FallbackSemantics/ExplicitAccounting = true
heldout_certificate_semantics_current = true
calibration_certificate_semantics_current = true
```

如果旧服务器 checkpoint 已被删除，直接进入 B。

---

## B. 在现有 v16.8 overlay 上重训

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_2_certificate_consistent_v9base_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
TRANSPORT_EPOCHS=24 \
PLANNER_EPOCHS=16 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh
```

状态：

```bash
OUT_ROOT=outputs/cowp_v16_8_2_certificate_consistent_v9base_seed2026 \
bash CHECK_RUN_STATUS_V16_8_2.sh
```

本阶段能验证 corrected uncertainty、checkpoint score、hard protected semantics、certificate/shortlist 和 fallback；不能验证 BCTE，因为旧候选已经缓存。

---

## C. Gate 判定

必须同时满足：

```text
mechanism_verification.pass = true
mechanism_verification.calibration_feasible = true
```

并检查主要硬条件：

```text
priority_accept_ncf_recall >= 0.30
priority_accept_ncf_precision >= 0.50
learned_accepted_candidate_rate >= 0.10
fallback_rate <= 0.25
priority_burden_transfer_rate <= calibration constraint
```

如果 gate 未通过，优先查看：

```text
ProposalCoverage/AnyNCFSceneRate
CertificateCoverage/AnyAcceptedSceneRate
CertificateCoverage/NCFSceneRetention
SelectionShortlist/CandidateRate
FallbackSelected/PBTR
```

若 `AnyNCFSceneRate < 0.35`，不要继续调大 BCOT budget，进入 D。

---

## D. 重建 fresh BCTE 数据

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
TRAIN_LIMIT=22000 \
VAL_LIMIT=5000 \
RUN_WAYMAX_REPLAY=1 \
MAX_REPLAY_CANDIDATES=24 \
CUDA_VISIBLE_DEVICES=0 \
bash PREPARE_COWP_V16_8_2_BCTE_DATA_CN.sh
```

输出应包含：

```text
tensor_cache_train_waymax
tensor_cache_val_waymax
tensor_cache_train_waymax_transport_v16_8_2
tensor_cache_val_waymax_transport_v16_8_2
data_manifest_v16_8_2.json
cache_alignment_train.json
cache_alignment_val.json
```

随后训练：

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
RAW_TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_train_waymax \
RAW_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_val_waymax \
TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_train_waymax_transport_v16_8_2 \
VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_val_waymax_transport_v16_8_2 \
DATA_PROTOCOL=v16_8_2_fresh \
OUT_ROOT=outputs/cowp_v16_8_2_bcte_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
TRANSPORT_EPOCHS=24 \
PLANNER_EPOCHS=16 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh
```

---

## E. Gate 通过后运行 Waymax probe

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
OUT_ROOT=outputs/cowp_v16_8_2_bcte_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_PROBE_CN.sh
```

人工检查：

- conventional safety 不退化；
- collision/offroad；
- PBTR 与 false-safe；
- fallback 及 fallback-selected PBTR；
- progress/comfort；
- probe sample 是否覆盖 protected interaction。

---

## F. Probe 合格后运行 full

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
OUT_ROOT=outputs/cowp_v16_8_2_bcte_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_FULL_CN.sh
```

`FULL` 脚本会强制检查 mechanism gate、当前 metric semantics 与 probe delta 是否存在。

---

## G. 结果回传清单

下一轮分析请至少打包：

```text
configs/
logs/
checkpoints/transport/history_witness.json
checkpoints/planner/history_planner.json
eval/learned_offline/
eval/probe/
eval/waymax/
eval/causal_protocol_audit.json
eval/pipeline_completion_report.json
```

若压缩体积允许，也保留两个 best checkpoint；这样可以在离线环境复算 selector/gate，而不需要重新训练。

---

## H. 禁止事项

- 不复用 v16.8.1 的 learned-offline JSON；
- 不降低 gate threshold；
- 不只提高 `BCOT_RISK_BUDGET`；
- 不跳过 probe 直接 full；
- 不把 fallback 计入 certified acceptance；
- 不用 sparse cached outcomes 作为完整闭环 SOTA 证据。
