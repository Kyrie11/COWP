# COWP v16.8.8 执行说明：先修稳定证书集合，再决定是否全量重建

## 0. 当前结论

当前 **不建议 full rebuild**。已恢复的 v16.8.6 micro probe 证明：fresh bank 的 PBTR floor 从 0.5868 降至 0.4309，但 AnyNCF 从 0.3672 降至 0.2031，false-safe floor 从 0.5781 恶化至 0.7656；PCHR 仅产生 17 条候选且 0 NCF。代码审计进一步发现旧 `select_critical_agents()` 使用整个 optional proposal bank 选择 top-A critical agents，导致新增 proposal 会重定义所有旧候选的 NCF 审计集合。因此 v16.8.6 的 universal-NCF 退化不能被干净归因为候选几何。

v16.8.8 的顺序是：

`96-scene smoke -> 1200-scene strict probe -> strict PASS 才允许 full rebuild -> post-build full-val gate -> 才训练`

## 1. 快速 smoke（现在执行）

```bash
cd /path/to/COWP_v16_8_8

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val

# 复用已经验证过的 hard/random ID 池，只重新构建 v16.8.8 的小子集。
export SOURCE_PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_6_priority_commitment_micro_probe
export SMOKE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke

export HARD_COUNT=48
export RANDOM_COUNT=48
export LABEL_WORKERS=24
export FORCE_REBUILD_SMOKE=1

bash NEXT_RUN_COMMANDS_V16_8_8_REFINEMENT_SMOKE_CN.sh
```

重点查看：

```text
$SMOKE_ROOT/v16_8_8_smoke_verdict.json
$SMOKE_ROOT/paired_probe.json
$SMOKE_ROOT/proposal_source_ablation.json
$SMOKE_ROOT/fresh_profile_summary.json
```

Smoke gate 是开发筛选，不是 full-rebuild 授权。主要条件：AnyValid>=0.99、AnyNCF>=0.30、false-safe floor<=0.65、PBTR floor<=0.50、hard recovery>=0.12，并要求 PSY 有实际生成/priority-NCF yield、critical mode 全部为 `fixed_anchor_v1`、proposal union 对 AnyNCF/false-safe/PBTR 三项单调不退化。

- `screen_pass=false`：停止；不要跑 1200，更不要 full rebuild。把上述四个 JSON 和 logs 上传继续分析。
- `screen_pass=true`：只允许进入 strict probe，仍然不能 full rebuild。

如 smoke 因服务器中断需要续跑，在**代码完全没改**时使用 `FORCE_REBUILD_SMOKE=0`。脚本会验证 `v16_8_8_code_fingerprint.sha256`；代码变更后拒绝混合 resume。

## 2. 1200-scene strict probe（仅 smoke PASS 后）

```bash
cd /path/to/COWP_v16_8_8

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_strict_probe

export HARD_COUNT=400
export RANDOM_COUNT=800
export LABEL_WORKERS=24
export SEED=2026
export FORCE_REBUILD_PROBE=1

bash NEXT_RUN_COMMANDS_V16_8_8_STRICT_PROPOSAL_PROBE_CN.sh
```

重点结果：

```text
$PROBE_ROOT/v16_8_8_strict_verdict.json
$PROBE_ROOT/paired_proposal_probe.json
$PROBE_ROOT/proposal_source_ablation_v16_8_8.json
$PROBE_ROOT/fresh_probe_profile_summary.json
```

正式授权条件：AnyValid>=0.99、AnyNCF>=0.40、false-safe floor<=0.55、PBTR floor<=0.45、hard recovery>=0.20；PSY scene rate>=0.05、PSY priority-NCF scene rate>=0.01、PSY accepted/attempted>=0.02；critical selection 必须全部是 `fixed_anchor_v1`；proposal union 必须单调；pairing/build error 必须为零。

只有：

```json
"recommend_full_rebuild": true
```

才允许下一阶段。

## 3. Full rebuild（仅 strict PASS 且代码 fingerprint 未变化）

```bash
cd /path/to/COWP_v16_8_8

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_stable_critical_psy

# 旧 base tensor cache 只用于确定严格 paired 的 scene-ID 集，不复用旧 COWP labels。
export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE=$SOURCE_DATA_ROOT/tensor_cache_train
export OLD_SCENESET_VAL_CACHE=$SOURCE_DATA_ROOT/tensor_cache_val

export STRICT_VERDICT=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_strict_probe/v16_8_8_strict_verdict.json

# 主训练不依赖 full-train cached Waymax outcome。
export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0

export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8

bash PREPARE_COWP_V16_8_8_DATA_FAST_CN.sh
```

脚本启动前会比较 strict verdict 的 code fingerprint 与当前代码。任何 label/proposal/eval 语义代码变化都会阻止 full rebuild，并要求重新跑 strict probe。

最终 fresh 数据为自包含实体 NPZ：

```bash
export RAW_TRAIN_CACHE=$COWP_ROOT/tensor_cache_train
export RAW_VAL_CACHE=$COWP_ROOT/tensor_cache_val
export TRAIN_CACHE=$COWP_ROOT/tensor_cache_train
export VAL_CACHE=$COWP_ROOT/tensor_cache_val
export USE_WAYMAX_OUTCOME_LABELS=0
```

不需要 `tensor_cache_*_waymax_transport_*` overlay。

Full build 完成后，脚本自动运行完整 cache integrity、full-val proposal ceiling 和 proposal-source ablation。在训练前再次要求：AnyValid>=0.99、AnyNCF>=0.40、false-safe<=0.55、PBTR<=0.45、PSY 有实际 protected-priority NCF、proposal union 单调。失败即停止，不进入 GPU 训练。

## 4. 数据构建速度原则

已完成 191-scene profile 中，candidate generation 约 0.86 s/scene；safe-response 约 164.5 s/scene；witness 约 99.7 s/scene；critical selection 约 21.3 s/scene。真正的重构成本在 response/witness，而不是 PCHR/PSY candidate generation。

当前版本只使用不改变标签质量的加速：producer allowlist、精确 scene-set 复用、BLAS/TF 单线程 worker、跳过不必要的 full-train Waymax replay、自包含 inline transport、断点 fingerprint。不要为了速度自行缩小 root-conditioned transport / witness / safe-response 搜索预算；那会直接改变 q/OPR/NCF ground truth。

## 5. Smoke/strict 失败后需要上传什么

最有诊断价值的是 `paired_probe/paired_proposal_probe.json`、`proposal_source_ablation.json`、`fresh_profile_summary.json` 和 `v16_8_8_*_verdict.json`。v16.8.8 profile 已经对**所有成功场景**统计每个 proposal source 的 attempted/accepted/rejection，因此能区分“PSY 没尝试”“动力学/map 过滤”“生成了但不是 NCF”“priority 方向有效但 global NCF 失败”等不同根因。
