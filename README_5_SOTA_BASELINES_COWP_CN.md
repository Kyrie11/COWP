# COWP v16.8.33：5 个外部规划 Baseline 的 WOMD/Waymax 统一复现与闭环评测

## 1. 先说明“严格复现”的边界

这个工程把 **GameFormer、DTPP、PLUTO、PlanT 2.0、PDM-Closed** 接入了 COWP 的 WOMD v1.3.1 数据与 Waymax 闭环评测。

这里必须区分两种“复现”：

1. **Native/source reproduction**：论文作者原生数据集与仿真器上的官方实现。GameFormer 原生含 WOMD interaction/open-loop，但官方仓库明确没有发布 WOMD closed-loop；DTPP/PLUTO/PDM-Closed 原生面向 nuPlan；PlanT 2.0 原生面向 CARLA。
2. **COWP/WOMD source-faithful adaptation**：保留公开论文/源码的主要模型与 planner 机制，把输入、route/map、轨迹输出和闭环 action 映射到当前 COWP/WOMD/Waymax contract。跨数据集之后不能诚实地称为“逐字节/逐 benchmark 的原生严格复现”，因此所有结果应标注为 **WOMD adaptation**。

官方参考：

- GameFormer: https://github.com/MCZhi/GameFormer ; https://arxiv.org/abs/2303.05760
- DTPP: https://github.com/MCZhi/DTPP ; https://arxiv.org/abs/2310.05885
- PLUTO: https://github.com/jchengai/pluto ; https://arxiv.org/abs/2404.14327
- PlanT 2.0: https://github.com/autonomousvision/plant2 ; https://arxiv.org/abs/2511.07292
- PDM-Closed / tuPlan Garage: https://github.com/autonomousvision/tuplan_garage ; https://arxiv.org/abs/2306.07962

## 2. 为什么选这 5 个

| 方法 | 代表性 | Native domain | COWP 里的执行方式 |
|---|---|---|---|
| GameFormer | hierarchical game-theoretic interactive prediction/planning | WOMD + nuPlan | **direct**：最终层 ego trajectory head |
| DTPP | ego-conditioned prediction + differentiable learned cost + tree policy | nuPlan | **candidate/tree**：保留 tree/proposal planning 逻辑 |
| PLUTO | 强 imitation planner；longitudinal/lateral-aware + auxiliary + contrastive | nuPlan | **direct**：最高概率 multimodal ego trajectory |
| PlanT 2.0 | 2025 object-centric planning transformer | CARLA | **direct**：autoregressive waypoint trajectory |
| PDM-Closed | 2023 nuPlan challenge winner，强 rule-based predictive planner | nuPlan | **predictive rule proposals**，无需训练 |

这 5 个方法覆盖 game reasoning、prediction-planning coupling、imitation planning、object-centric transformer 和强 rule-based closed-loop planner，适合作为 COWP 的外部比较组。

## 3. 公平性 contract

所有方法统一：

- 同一 held-out scenario manifest：`reference_manifests/formal_v16_8_24_compact5k_heldout1200_ids.txt`
- 场景数：1200
- manifest logical SHA256：`134fa919582e64bf2b315be474890456be36af0be81a6d24364033e24456494f`
- WOMD history：current + past 共 11 帧
- planning horizon：80 帧 / 8 s
- route：优先使用 WOMD v1.3.1 SDC `path_samples/on_route`，缺失时回退 roadgraph route proxy
- map：WOMD roadgraph/vector polyline
- action projection：`absolute_xy_yaw`
- 非 ego agent：Waymax logged replay（与论文 primary protocol 一致）
- 标准闭环 metric adapter：统一 Waymax evaluator

**禁止训练泄漏：** 外部 baseline 的 `planner_inputs` 不包含 COWP 的 `false_safe`、witness、OPR、burden、NCF 等机制标签；这些标签只可在统一机制 audit 中用于评价。

## 4. 新增/修改的关键代码

- `cowp/external_baselines/pluto_cowp.py`
- `cowp/external_baselines/plant2_cowp.py`
- `cowp/external_baselines/gameformer_cowp.py`
- `cowp/external_baselines/dtpp_cowp.py`
- `cowp/external_baselines/adapters.py`
- `cowp/external_baselines/waymax_policy.py`
- `cowp/external_baselines/rule_based.py`
- `cowp/scripts/20_train_external_baseline.py`
- `cowp/scripts/21_eval_external_baseline.py`
- `cowp/scripts/22_eval_rule_baseline.py`
- `cowp/scripts/24_summarize_sota_closed_loop.py`
- `RUN_5_SOTA_BASELINES_COWP.sh`
- `NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh`（新增 heldout1200 mode）
- `tests/test_external_sota_baselines.py`

## 5. 数据路径

runner 默认沿用当前 formal v16.8.24 compact 5k 的历史路径：

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k/tensor_cache_train
export VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k/tensor_cache_val
export HELDOUT_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k/labels_heldout_test
```

如果你的机器路径不同，只覆盖上述环境变量，不要改代码。

## 6. 训练 4 个 learning-based baseline

PDM-Closed 无训练。

### 一键训练全部

```bash
bash RUN_5_SOTA_BASELINES_COWP.sh train all
```

### 单独训练

```bash
bash RUN_5_SOTA_BASELINES_COWP.sh train gameformer
bash RUN_5_SOTA_BASELINES_COWP.sh train dtpp
bash RUN_5_SOTA_BASELINES_COWP.sh train pluto
bash RUN_5_SOTA_BASELINES_COWP.sh train plant2
```

默认 source-oriented training recipe：

- GameFormer：20 epochs, batch 32, lr 1e-4；公开源码 MultiStepLR `[10,12,14,16,18]`, gamma 0.5。
- DTPP：30 epochs, batch 16, lr 2e-4；公开源码 StepLR(step=5, gamma=0.5)。
- PLUTO：25 epochs, batch 32, lr 1e-3, weight decay 1e-4；对应官方 README full-data recipe。CIL/auxiliary 机制在 WOMD adapter 中保留，但原生 nuPlan augmentation pipeline 不可逐字节迁移。
- PlanT2：WOMD adapter 默认 30 epochs, batch 16, lr 1e-4；CARLA 原生训练 pipeline/专家数据不可直接复用于 WOMD，因此这组超参属于适配预算，不标注为 native exact。

任何全局覆盖都可以这样设置：

```bash
EPOCHS=30 BATCH_SIZE=16 LR=2e-4 SEED=3407 \
  bash RUN_5_SOTA_BASELINES_COWP.sh train dtpp
```

论文级比较建议至少 3 seeds，并保持同一 split、同一 Waymax IDs、同一 evaluator。例如：

```bash
for seed in 3407 3408 3409; do
  SEED=$seed OUT_ROOT=outputs/external_sota5_v16_8_33_seed${seed} \
    bash RUN_5_SOTA_BASELINES_COWP.sh train all
done
```

## 7. 外部 baseline 的机制 audit

这是 **heldout1200 上的 cached-label candidate-projection audit**，用于统一计算 COWP 定义的 PBTR/FSR/OPR/BTE/NCF-Ret/NPR。它不是 reactive-agent 反事实 ground truth。

```bash
bash RUN_5_SOTA_BASELINES_COWP.sh offline all
```

单个方法：

```bash
bash RUN_5_SOTA_BASELINES_COWP.sh offline gameformer
bash RUN_5_SOTA_BASELINES_COWP.sh offline dtpp
bash RUN_5_SOTA_BASELINES_COWP.sh offline pluto
bash RUN_5_SOTA_BASELINES_COWP.sh offline plant2
bash RUN_5_SOTA_BASELINES_COWP.sh offline pdm_closed
```

## 8. 5 个 baseline 的 Waymax 1200-scene 闭环

默认和你现有 workflow 一样采用两张 GPU、2 shards，并自动 exact-shard merge：

```bash
PARALLEL2=1 GPU0=0 GPU1=1 \
  bash RUN_5_SOTA_BASELINES_COWP.sh waymax all
```

单独跑：

```bash
bash RUN_5_SOTA_BASELINES_COWP.sh waymax gameformer
bash RUN_5_SOTA_BASELINES_COWP.sh waymax dtpp
bash RUN_5_SOTA_BASELINES_COWP.sh waymax pluto
bash RUN_5_SOTA_BASELINES_COWP.sh waymax plant2
bash RUN_5_SOTA_BASELINES_COWP.sh waymax pdm_closed
```

每个方法最终输出：

```text
outputs/external_sota5_v16_8_33/<method>/waymax.json
outputs/external_sota5_v16_8_33/<method>/offline.json
```

## 9. COWP v16.8.33 当前模型闭环

你已有的两个开发 gate 命令保持不变：

```bash
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh analyze_counterfactual48

PROMOTED_METHODS=cowp_recovery_option_spectrum_hysteresis \
  bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh fresh37_parallel2
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh analyze_fresh37
```

只有 fresh37 的 preregistered gate 通过后，才能进入新增的 heldout1200：

```bash
WAYMAX_STANDARD_METRIC_NAMES=OverlapMetric,OffroadMetric,WrongWayMetric,ProgressionMetric,OffRouteMetric,KinematicsInfeasibilityMetric,LogDivergenceMetric \
PROMOTED_METHODS=cowp_recovery_option_spectrum_hysteresis \
  bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh heldout1200_parallel2
```

默认输出：

```text
outputs/v16_8_33_recovery_option_spectrum/heldout1200_v33_cowp_recovery_option_spectrum_hysteresis_merged.json
```

## 10. 最终汇总 5 baseline + COWP

当上面的 COWP heldout1200 JSON 已生成：

```bash
COWP_JSON=outputs/v16_8_33_recovery_option_spectrum/heldout1200_v33_cowp_recovery_option_spectrum_hysteresis_merged.json \
  bash RUN_5_SOTA_BASELINES_COWP.sh summary
```

输出：

```text
outputs/external_sota5_v16_8_33/summary_5_baselines_plus_cowp.json
outputs/external_sota5_v16_8_33/summary_5_baselines_plus_cowp.csv
```

## 11. 最终表中的指标

### 真正的 Waymax closed-loop 指标（可以在 5 baseline 与 COWP 之间直接比较）

- `CollisionRate` ↓
- `OffroadRate` ↓
- `CUR` ↓：collision ∪ offroad；注意工程里历史字段 `CR` 实际保存的是这个 union，不等同论文中的 collision-only CR。
- `WrongWayRate` ↓
- `OffRouteRate` ↓
- `EgoProgress` ↑
- `KinematicsInfeasibilityRate` ↓
- `LogDivergence` ↓

### COWP mechanism audit（必须保留 protocol 标签）

- `PBTR_offline_audit` ↓
- `FSR_offline_audit` ↓
- `OPR_offline_audit` ↑
- `BTE_CVaR25_offline_audit` ↓
- `NCF_Ret_offline_audit` ↑
- `NPR_offline_audit` ↓

对外部 baseline，这些是 `cached_label_candidate_projection`；**不能写成 logged-replay 下的 ground-truth counterfactual burden**。论文中的最终 causal burden claim 仍需独立 reactive-agent protocol 与 human-audited false-safe stress set。

## 12. 当前包中已经存在的 COWP 参考数值（只作开发 sanity，不与 1200 baseline 直接比较）

`reference_results/v16_8_32_fresh37_cowp_reference.json`：

- n = 37
- CollisionRate = 0.000000
- OffroadRate = 0.081081
- CUR = 0.081081
- EgoProgress = 1.201146
- KinematicsInfeasibilityRate = 0.216216
- ClosedLoopFallbackStepRate = 0.530068

这是 fresh37 development reference，不是 heldout1200 最终结果。

## 13. 数据质量 gate 提醒

上传的 formal 5k 分析包显示 train cache 可读性/SDC path contract 正常，但当前 `verify_cache_train.json` 的总 gate 仍为 false，原因之一是存在大量 irrelevant-pair blockers。工程接入和 smoke test 可以继续，但论文级最终数字不建议在未解释/修复该 gate 的情况下直接宣称为 publication evidence。

## 14. 本地完整性测试

```bash
PYTHONPATH=. pytest -q \
  tests/test_external_sota_baselines.py \
  tests/test_v16_8_6_speed_and_baseline_integrity.py \
  tests/test_v16_8_33_recovery_option_spectrum.py
```

此外：

```bash
python -m py_compile \
  cowp/scripts/20_train_external_baseline.py \
  cowp/scripts/21_eval_external_baseline.py \
  cowp/scripts/22_eval_rule_baseline.py \
  cowp/scripts/24_summarize_sota_closed_loop.py
bash -n RUN_5_SOTA_BASELINES_COWP.sh
bash -n NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh
```

## 15. 最重要的论文表述建议

如果最终写论文，外部方法名称建议写成：

- `GameFormer (WOMD adaptation)`
- `DTPP (WOMD adaptation)`
- `PLUTO (WOMD adaptation)`
- `PlanT 2.0 (WOMD adaptation)`
- `PDM-Closed (WOMD adaptation)`

并在实验设置里说明：保持公开方法的核心 architecture/planning mechanism；由于 native simulator/domain 不同，统一改接相同 WOMD observation contract 与 Waymax action/metric adapter。这样是可复现且科学诚实的跨域外部 baseline，而不是把适配实现误称为作者原生 benchmark 的 exact reproduction。
