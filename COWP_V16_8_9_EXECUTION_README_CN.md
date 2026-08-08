# COWP v16.8.9 下一步执行说明

## A. 当前唯一建议先跑的实验：96-scene causal-audit smoke

**当前状态：DO NOT FULL REBUILD。**

```bash
cd /path/to/COWP_v16_8_9

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val

# 复用 v16.8.8 smoke 的 hard/random ID 池，保证可配对比较
export SOURCE_PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke

export SMOKE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_smoke
export HARD_COUNT=48
export RANDOM_COUNT=48
export LABEL_WORKERS=24
export FORCE_REBUILD_SMOKE=1

# 可选：若 v16.8.8 fresh label 目录仍保留，可额外生成新旧 fresh paired comparison
export PREV_FRESH_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_8_refinement_smoke/labels_val_v16_8_8

bash NEXT_RUN_COMMANDS_V16_8_9_CAUSAL_AUDIT_SMOKE_CN.sh
```

重点输出：

```text
$SMOKE_ROOT/v16_8_9_smoke_verdict.json
$SMOKE_ROOT/paired_probe.json
$SMOKE_ROOT/causal_audit_diagnostic.json
$SMOKE_ROOT/proposal_source_ablation.json
$SMOKE_ROOT/fresh_profile_summary.json
```

如果 `screen_pass=false`：**停止，不 strict、不 full rebuild。**

如果 `screen_pass=true`：只进入 B；仍不要直接 full rebuild。

---

## B. Smoke 通过后：400 hard + 800 random strict probe

```bash
cd /path/to/COWP_v16_8_9

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export OLD_VAL_CACHE=$COWP_ROOT/tensor_cache_val

export PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe
export HARD_COUNT=400
export RANDOM_COUNT=800
export LABEL_WORKERS=24
export SEED=2026
export FORCE_REBUILD_PROBE=1

bash NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh
```

Full rebuild 的**唯一授权条件**：

```text
$PROBE_ROOT/v16_8_9_strict_verdict.json
recommend_full_rebuild = true
```

Strict 核心阈值：

```text
AnyValid >= 0.99
AnyNCF >= 0.40
false-safe floor <= 0.55
PBTR floor <= 0.45
hard recovery >= 0.20
causal-audit integrity = PASS
stable critical = PASS
proposal-union monotonic = PASS
```

---

## C. Strict PASS 后：full fresh rebuild

```bash
cd /path/to/COWP_v16_8_9

export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1

export SOURCE_DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit

# 旧 cache 只提供 scene-ID allowlist；不复用旧 COWP 标签
export REUSE_OLD_SCENE_SET=1
export OLD_SCENESET_TRAIN_CACHE=$SOURCE_DATA_ROOT/tensor_cache_train
export OLD_SCENESET_VAL_CACHE=$SOURCE_DATA_ROOT/tensor_cache_val

export STRICT_VERDICT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit_strict_probe/v16_8_9_strict_verdict.json

# 不影响标签质量的加速
export RUN_WAYMAX_REPLAY=0
export RUN_LABEL_DIAGNOSTICS=0
export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8

bash PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
```

构建后正式使用：

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export RAW_TRAIN_CACHE=$DATA_ROOT/tensor_cache_train
export RAW_VAL_CACHE=$DATA_ROOT/tensor_cache_val
export TRAIN_CACHE=$DATA_ROOT/tensor_cache_train
export VAL_CACHE=$DATA_ROOT/tensor_cache_val
export USE_WAYMAX_OUTCOME_LABELS=0
```

**不要再创建 transport overlay。** Fresh transport 已内嵌在实体 NPZ 中。

Full build 最后会自动对完整 val cache 再做 proposal + causal-audit hard gate；失败即停止训练。

---

# D. Full data PASS 后：seed 2026 主机制实验

```bash
cd /path/to/COWP_v16_8_9

export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export OUT_ROOT=outputs/cowp_v16_8_9_causal_audit_seed2026
export TRAIN_SEED=2026
export CUDA_VISIBLE_DEVICES=0,1
export BACKGROUND=0

bash NEXT_RUN_COMMANDS_V16_8_9_MECHANISM_CN.sh
```

这个入口首先：

1. 验证 v16.8.9 fresh cache lineage/schema；
2. 在**实际 fresh val cache**上重新验证之前 transfer 的 natural decoder；
3. natural gate 通过后才进入 transport/planner/calibration/held-out mechanism。

如果 fresh natural gate 失败：**停止，不强行训练 transport/planner。** 这意味着 fixed-critical/audit 数据改变了自然根评估分布，需要单独决定是否在 fresh data 上重新训练 natural decoder。

---

# E. Main mechanism 通过后：便宜的选择/标签/Proposal 消融

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export OUT_ROOT=outputs/cowp_v16_8_9_causal_audit_seed2026

bash RUN_OFFLINE_SELECTION_ABLATIONS_V16_8_9_CN.sh
bash RUN_LABEL_AND_PROPOSAL_ABLATIONS_V16_8_9_CN.sh
```

第一条比较 shared model forward 下真正的 selector/certificate decision：

- planner-score only
- IDM/lattice
- conventional safety
- soft burden cost only
- universal NCF oracle
- full COWP

第二条输出：

- label-space mechanism tables；
- proposal-source ablation；
- proposal ceiling；
- causal-audit diagnostics。

这些不等同于 learned architecture ablation。

---

# F. 必须独立重训的关键 learned ablation

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export BASE_OUT_ROOT=outputs/cowp_v16_8_9_causal_audit_seed2026
export CUDA_VISIBLE_DEVICES=0,1

bash RUN_LEARNED_CAUSAL_ABLATIONS_V16_8_9_CN.sh
```

包括：

1. **w/o candidate-conditioned causal relevance**；
2. **conflict-only RootTransport**（去掉 burden-only affected-root support）。

两项都使用相同 rich fresh dataset、但独立 checkpoint；不是 shared-checkpoint 伪消融。

---

# G. Waymax closed-loop：先 100 scenes

仅当主 mechanism gate 通过：

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export OUT_ROOT=outputs/cowp_v16_8_9_causal_audit_seed2026
export PROBE_SCENARIOS=100
export BACKGROUND=0

bash NEXT_RUN_COMMANDS_V16_8_9_PROBE_CN.sh
```

只有 paired Waymax promotion gate 通过才进入 full。

---

# H. Waymax full

```bash
export FULL_SCENARIOS=1000
export BACKGROUND=0
bash NEXT_RUN_COMMANDS_V16_8_9_FULL_CN.sh
```

---

# I. 多 seed

仅在 seed 2026 的数据 gate、mechanism、Waymax probe 都无结构性错误后：

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export SEEDS="2026 2027 2028"
export CUDA_VISIBLE_DEVICES=0,1

bash RUN_MULTI_SEED_MECHANISM_V16_8_9_CN.sh
```

最终 paper 建议报告 3 seeds 的 offline mechanism 与闭环指标均值/方差；Waymax full 可在确认 3 个 mechanism runs 合格后分别执行。

---

# J. External baselines（必须使用同一个 fresh proposal bank）

先做数值 smoke：

```bash
export DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_9_causal_audit
export OUT_ROOT=outputs/external_baselines_v16_8_9
MODE=smoke RUN_ONLINE_EVAL=0 bash RUN_EXTERNAL_BASELINES_V16_8_9_CN.sh
```

要求 skipped batch fraction <= 2%。

通过后：

```bash
MODE=full RUN_ONLINE_EVAL=1 bash RUN_EXTERNAL_BASELINES_V16_8_9_CN.sh
```

仓库内 GameFormer/DTPP 是 matched implementation baseline；除非另外验证官方 repo/权重，不应称为 official reproduction。

---

## 最小决策树

```text
v16.8.9 96-scene smoke
  FAIL -> STOP / 上传 5 个 JSON / 不 full rebuild
  PASS
    -> 1200-scene strict
       FAIL -> STOP / 不 full rebuild
       PASS + recommend_full_rebuild=true
          -> full fresh rebuild
             full-val gate FAIL -> DO NOT TRAIN
             PASS
               -> seed2026 mechanism
                  mechanism FAIL -> 不 Waymax
                  PASS
                    -> ablations + 100-scene Waymax probe
                       probe PASS -> 1000-scene Waymax full
                       -> 多 seed + matched baselines
```
