# COWP 本轮代码修改说明与推荐指令

## 1. 本轮主要修复

### Waymax online policy

修改文件：`cowp/waymax_eval/policy_wrapper.py`

关键修复：

1. 在线输入不再只用当前一帧，而是从 Waymax `SimulatorState` 中提取最近 `history_steps` 帧，并转换成训练 cache 使用的模型状态格式 `[x,y,z,length,width,height,heading,vx,vy,speed,valid]`。
2. 在线候选不再只有少数 constant acceleration primitive，而是加入：
   - nearest-roadgraph/lane-heading keep-lane；
   - yield / accelerate timing lattice；
   - stop-before-conflict；
   - merge-ahead / merge-behind timing；
   - lane-change / cut-in lateral proposals。
3. 在线 batch 现在会尽力从 Waymax roadgraph 提取局部 roadgraph token，并构造非零 `map/conflict_regions`，避免 conflict-query 分支在闭环里全零。
4. critical agents 不再只是最近邻，而是用 candidate-agent closest approach、当前距离、closing/TTC proxy 共同排序。
5. 在线 candidate 会计算 lightweight `conventional_safe` mask，包括动态可行性、roadgraph 近邻 drivable proxy、与其他车 constant-velocity projection 的距离检查。
6. 修复 Waymax action conversion：candidate future 的 `traj[0]` 才是下一步 action 目标，原先使用 `traj[1]` 会放大每步位移，容易触发 kinematic/offroad/wrong-way 爆炸。
7. 对 witness、OPR、planner score 做 `nan_to_num`，不再让 NaN 直接造成 100% fallback 和 NaN diagnostic。
8. 闭环 diagnostic 增加：`valid_candidates`、`conventional_candidates`、`critical_agents`、`conflict_tokens`、`fallback_reason`、`mean_witness_prob`。

### Learned offline / witness / ranking diagnostics

修改文件：

- `cowp/waymax_eval/rollout.py`
- `cowp/scripts/04_eval_closed_loop.py`
- `cowp/waymax_eval/metrics_cowp.py`
- `cowp/scripts/05_make_tables.py`

新增/增强指标：

1. `SelectedNCFRate`：learned planner 选中的 candidate 中 NCF 比例。
2. `SelectedFalseSafeRate`：learned planner 选中的 candidate 中 false-safe 比例。
3. `SelectedConventionalSafeRate`：选中 candidate 的 conventional safe 比例。
4. `LearnedAcceptedCandidateRate`：模型 hard filter 接受的有效候选比例。
5. `LearnedAcceptNCFRecall`：真实 NCF candidate 被模型接受的召回率。
6. `LearnedAcceptFalseSafeRate`：真实 false-safe candidate 被模型错误接受的比例。
7. `WitnessQuality/AUPRC`：threshold-free witness pair AP，避免只看 0.5 threshold。
8. `PlannerRankingPairAccuracy`：同一 root scene 中 learned score 是否把 NCF 排在 false-safe 前面。
9. `--witness-threshold-sweep`：learned offline 可一次跑多个 witness threshold。
10. `module_effects.csv`：对 full COWP 与各消融方法的 decision change、accepted-mask Jaccard 和 COWP 指标差异做表。

### Training config

修改文件：`configs/train.yaml`

把 `witness_focal_alpha` 从 `0.25` 改为 `0.75`。当前 positive witness pair 稀疏，原先配置会在 focal BCE 中相对压低 positive 权重，容易学成全负或极低召回。

### Label config

修改文件：`configs/label.yaml`

增加 `planning.online_lane_change_offsets_m`，供 Waymax online proposal 使用。

---

## 2. 快速诊断 / 快速训练指令

目的：低 epoch 检查 idea 是否有学习信号，以及闭环 smoke 是否不再被工程问题击穿。建议先不要追求最终 SOTA。

### 2.1 快速 response 阶段

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage response \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --resume outputs/checkpoints/representation/cowp_representation_best.pt \
  --output-dir outputs/checkpoints/quick_response
```

### 2.2 快速 witness 阶段

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage witness \
  --epochs 3 \
  --batch-size 32 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --resume outputs/checkpoints/quick_response/cowp_response_best.pt \
  --output-dir outputs/checkpoints/quick_witness
```

### 2.3 快速 planner 阶段

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage planner \
  --epochs 2 \
  --batch-size 32 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --resume outputs/checkpoints/quick_witness/cowp_witness_best.pt \
  --output-dir outputs/checkpoints/quick_planner
```

### 2.4 快速 learned offline eval，带 threshold sweep

```bash
python -m cowp.scripts.04_eval_closed_loop \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --mode learned_offline \
  --method cowp \
  --checkpoint outputs/checkpoints/quick_planner/cowp_planner_best.pt \
  --batch-size 64 \
  --witness-threshold 0.5 \
  --witness-threshold-sweep 0.1,0.2,0.3,0.5,0.7 \
  --output outputs/eval/quick_learned_offline_cowp_val.json
```

重点看：

- `WitnessQuality/AUPRC` 是否显著高于正例基率；
- `WitnessQuality/WitnessRecall` 是否不再为 0；
- `SelectedFalseSafeRate` 是否下降；
- `PlannerRankingPairAccuracy` 是否明显高于 0.5；
- `LearnedAcceptFalseSafeRate` 是否低于 ablation / soft-only。

### 2.5 Waymax smoke test

先用 CPU Waymax/JAX 避免 JAX 与 PyTorch 抢 GPU：

```bash
python -m cowp.scripts.04_eval_closed_loop \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --mode waymax \
  --method cowp \
  --checkpoint outputs/checkpoints/quick_planner/cowp_planner_best.pt \
  --num-scenarios 100 \
  --rollout-horizon-steps 80 \
  --waymax-standard-metrics \
  --witness-threshold 0.5 \
  --waymax-device cpu \
  --waymax-action-mode delta_xy_yaw \
  --output outputs/eval/quick_cowp_waymax_smoke_100.json
```

如果 `KinematicsInfeasibilityRate / WrongWayRate / OffroadRate` 仍异常高，请立即对比：

```bash
# 绝对状态 action 语义排查
python -m cowp.scripts.04_eval_closed_loop \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --mode waymax \
  --method cowp \
  --checkpoint outputs/checkpoints/quick_planner/cowp_planner_best.pt \
  --num-scenarios 30 \
  --rollout-horizon-steps 80 \
  --waymax-standard-metrics \
  --witness-threshold 0.5 \
  --waymax-device cpu \
  --waymax-action-mode absolute_xy_yaw \
  --output outputs/eval/quick_cowp_waymax_abs_smoke_30.json
```

---

## 3. 完整训练指令

### Stage A：representation / natural pretraining

如果你的数据 cache 包含完整 natural supervision，建议先跑 representation 或 natural，而不是把 Stage A 写成 response。

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage natural \
  --epochs 8 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --amp \
  --fused-adamw \
  --output-dir outputs/checkpoints/natural
```

### Stage B：response supervised training

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage response \
  --epochs 12 \
  --batch-size 48 \
  --num-workers 8 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --resume outputs/checkpoints/natural/cowp_natural_best.pt \
  --output-dir outputs/checkpoints/response
```

### Stage C：witness training

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage witness \
  --epochs 15 \
  --batch-size 48 \
  --num-workers 8 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --resume outputs/checkpoints/response/cowp_response_best.pt \
  --output-dir outputs/checkpoints/witness
```

### Stage D：planner / ranking training

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage planner \
  --epochs 8 \
  --batch-size 48 \
  --num-workers 8 \
  --prefetch-factor 1 \
  --amp \
  --fused-adamw \
  --resume outputs/checkpoints/witness/cowp_witness_best.pt \
  --output-dir outputs/checkpoints/planner
```

---

## 4. 离线 ablation / module-effect 表

```bash
python -m cowp.scripts.05_make_tables \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --labels-dir outputs/cowp/formal/labels_val \
  --output-dir outputs/tables
```

除了原来的 `main_results.csv / ablation.csv / stress_test.csv / witness_quality.csv`，现在会额外输出：

```text
outputs/tables/module_effects.csv
```

重点看：

- `DecisionChangeVsFull`：关闭模块后选择是否真的变化；
- `AcceptedJaccardVsFull`：关闭模块后 accepted candidate set 是否明显不同；
- `FSR/CBS/OPR/HBCR/EP` 是否朝预期方向恶化。

如果 `cowp_wo_neutral_branch`、`cowp_wo_priority_branch` 与 full 几乎完全一样，说明 counterfactual branch 标签没有起作用，论文里不能声称该模块被证明。

---

## 5. 当前仍需注意

1. 本补丁让 online policy 更接近训练/离线分布，但无法凭空恢复 Scenario proto 级别的 route、traffic light、priority、natural alternatives。最终论文闭环表仍建议在 selected trajectory 上做 proto/label counterfactual recertification。
2. 如果 Waymax smoke 仍出现接近 100% wrong-way/offroad/kinematic infeasibility，优先判定为 Waymax dynamics action 语义或 route/path 提取未对齐，而不是论文 idea 失败。
3. learned witness 若 `AUPRC`、`Recall` 仍接近 0，应先检查 positive pair ratio、token distribution、`witness_focal_alpha`、threshold sweep，而不是直接改成 soft burden only。
