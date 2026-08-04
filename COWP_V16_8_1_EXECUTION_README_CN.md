# COWP v16.8.1 分阶段执行指令

## 1. 版本与目录要求

请使用本次交付的新代码目录，不要覆盖旧输出。默认数据目录：

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
```

本轮复用：

```text
$COWP_ROOT/tensor_cache_train_waymax
$COWP_ROOT/tensor_cache_val_waymax
$COWP_ROOT/tensor_cache_train_waymax_transport_v16_8
$COWP_ROOT/tensor_cache_val_waymax_transport_v16_8
```

使用新的输出根：

```bash
export OUT_ROOT=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026
```

不要复用旧的：

```text
outputs/cowp_v16_8_pipeline_v9labels_seed2026
outputs/cowp_v16_8_rcot_v9base_seed2026
```

严格 provenance 会主动阻止旧代码与新代码混合。

## 2. 是否需要重建 overlay

上传结果已证明现有 train/val v16.8 overlay 完整、无错误且 split disjoint。本次修复发生在模型、loss、audit 与执行链，**默认不需要重建数据**。

只有服务器上的 overlay 被移动、缺文件或 `transport_augmentation_summary.json` 不完整时，才执行：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
AUG_WORKERS_TRAIN=12 \
AUG_WORKERS_VAL=8 \
bash PREPARE_COWP_V16_8_OVERLAY_CN.sh
```

## 3. 阶段 A：训练 transport/planner 并做 learned-offline mechanism gate

代码默认复用已验证的 v16.6 natural checkpoint。若你的实际路径不同，覆盖 `SOURCE_NATURAL_ROOT` 和 `ATTR_GATE`。

```bash
cd /path/to/COWP_v16_8_1

COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_MECHANISM_CN.sh
```

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026 \
bash CHECK_RUN_STATUS_V16_8_1.sh
```

持续看日志：

```bash
tail -f outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026/logs/driver.nohup.log
```

阶段 A 正常结束位置：

```text
eval/learned_offline/mechanism_verification.json
```

检查硬门禁：

```bash
python - <<'PY'
import json
p='outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026/eval/learned_offline/mechanism_verification.json'
x=json.load(open(p))
print(json.dumps(x,indent=2,ensure_ascii=False))
assert x['pass'] is True
assert x['calibration_feasible'] is True
PY
```

若失败，不要运行 Waymax。优先查看：

```text
priority_root_transport_auprc
priority_accept_ncf_recall
priority_accept_ncf_precision
learned_accepted_candidate_rate
fallback_rate
priority_transfer_improvement
global_false_safe_improvement
```

## 4. 阶段 B：真实 Waymax probe

仅在阶段 A 两个 gate 都通过后执行：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_PROBE_CN.sh
```

该脚本会从 transfer manifest 自动恢复 external natural checkpoint/history，并复用同一 OUT_ROOT 的 transport/planner checkpoint。不会重新训练。

完成后检查：

```text
eval/probe/cowp_root_transport_*.json
eval/probe/conventional_safety_*.json
eval/probe/delta_conventional_vs_root_transport.json
eval/pipeline_completion_report.json
```

probe 主要判断：

- Waymax 环境和 checkpoint wrapper 是否健康；
- collision/offroad/progress 是否出现明显退化；
- PBTR/false-safe 改善方向是否与 learned-offline 一致；
- fallback 是否异常；
- pairmax/Pareto ablation 是否符合预期。

probe 样本量不足以形成论文结论，必须人工检查趋势后再运行 full。

## 5. 阶段 C：真实 Waymax full

`NEXT_RUN_COMMANDS_V16_8_FULL_CN.sh` 会强制要求 probe delta 文件存在，并且不重复 probe：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_FULL_CN.sh
```

主要输出：

```text
eval/waymax/cowp_root_transport_merged.json
eval/waymax/conventional_safety_merged.json
eval/waymax/planner_score_only_merged.json
eval/waymax/delta_conventional_vs_cowp.json
eval/waymax/delta_planner_vs_cowp.json
eval/pipeline_completion_report.json
```

## 6. 常见失败与对应处理

### 6.1 再次出现 data-protocol invalid choice

说明运行的不是本次交付代码。确认：

```bash
grep -n v16_8_root_conditioned_overlay cowp/scripts/36_audit_causal_protocol.py
```

### 6.2 checkpoint shape mismatch: mode_out 4 vs 5

确认新文件存在并被 provenance 记录：

```bash
ls cowp/utils/checkpoint_compat.py
grep -R load_checkpoint_compatible -n cowp/scripts cowp/waymax_eval
```

不要删除新 `b*` 行，也不要使用普通 `strict=False` 代替迁移器。

### 6.3 probe/full 报 missing natural checkpoint/history

检查：

```bash
cat "$OUT_ROOT/configs/natural_attribution_transfer_manifest.json"
```

路径必须在当前服务器仍存在。若 checkpoint 被移动，重新运行 mechanism launcher 并设置正确的 `SOURCE_NATURAL_ROOT`。

### 6.4 learned-offline 无可行 calibration point

不要先增大 BCOT budget。按顺序检查：

1. `RootTransport/PriorityConflict_AUPRC`；
2. direct `q` 与 `b*` 的 calibration；
3. OPR predicted/target 分布；
4. accepted/fallback frontier；
5. target confidence 分层；
6. root source/type 分层。

### 6.5 finite logdiv 为 0

这是当前数据已知限制。保持 logdiv loss/selection 禁用，不要填 0 伪造监督。
