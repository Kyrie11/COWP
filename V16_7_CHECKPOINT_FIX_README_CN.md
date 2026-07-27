# COWP v16.7 checkpoint 修复说明

## 1. 报错根因

`NEXT_RUN_COMMANDS_V16_7_MECHANISM_CN.sh` 有意复用已通过门禁的 v16.6 natural checkpoint，随后重新训练 v16.7 transport 和 planner。v16.7 的 `SetTransportCertificateHead` 新增了 7 个可学习参数：

- `candidate_risk_raw_weight`
- `candidate_risk_threshold_logit`
- `candidate_risk_log_scale`
- `global_risk_raw_weight`
- `global_risk_threshold_logit`
- `global_risk_log_scale`
- `pair_deficit_raw_weight`

这些参数属于后续 BCOT/transport 证书头，`stage="natural"` 的诊断前向不会使用它们。但原 `39_diagnose_learned_natural.py` 对整个模型做全局严格检查，因此把“预期的下游新增参数”误判为 natural checkpoint/config 不匹配。

## 2. 修复策略

1. natural 诊断只允许上述 7 个键缺失；graph、natural decoder 或任何其他键缺失仍立即报错。
2. natural -> transport 使用 `v16_7_natural_to_transport` 严格迁移策略：只初始化上述 7 个新增参数，其余参数必须逐键、逐形状匹配。
3. transport -> planner 使用 `strict` 策略，并要求来源 checkpoint 的 `stage=witness`。
4. learned-offline/Waymax 前运行最终 planner checkpoint 精确验证，禁止用旧 checkpoint 静默随机初始化新头。
5. FULL 在线续跑不再重新要求 external natural checkpoint/history，只复用同一 `OUT_ROOT` 中已经通过的 mechanism gate、BCOT calibration 和 v16.7 planner checkpoint。
6. 如果旧运行恰好在本次报错处中断、尚未生成 transport/planner checkpoint，机制 wrapper 会以 amendment 方式保留并更新 provenance；不会覆盖旧 provenance。

## 3. 机制训练与离线门禁

原命令可直接使用：

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

正常日志中应出现：

```text
Learned-natural checkpoint migration: initialized unused v16.7 downstream keys ...
Strict checkpoint migration initialized approved v16.7 keys ...
Checkpoint load policy=v16_7_natural_to_transport ...
Checkpoint load policy=strict ...
```

## 4. Waymax probe + full 评测

机制门禁通过后，原命令可直接使用：

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

该流程会先验证：

- `mechanism_verification.json` 为 `pass=true`；
- gate role 为 `development_continuation_not_paper_claim`；
- planner checkpoint 的 stage、参数名和张量形状与当前 v16.7 模型完全一致；
- BCOT calibration 与 planner checkpoint 均位于同一 provenance root。

## 5. 验证结果

- 仓库回归：`142 passed`
- Python：`compileall` 通过
- Shell：全部 `bash -n` 通过
- 真实 `COWPModel` 模拟 v16.6 checkpoint：7 个新增参数迁移通过
- 7 个新增参数在 witness/transport 阶段均为 trainable
- online-only 分支：`NEED_NATURAL_PIPELINE=0`、`NEED_TRANSPORT_CHECKPOINT=0`、`NEED_PLANNER_CHECKPOINT=1`
