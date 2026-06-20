# COWP：基于 WOMD + Waymax 的 Non-Coercive Planning 复现工程

本工程实现论文 **Not All Collision-Free Plans Are Safe: Non-Coercive Feasibility for Interactive Autonomous Driving** 的数据构造、诊断、模型训练、候选规划、闭环/离线评测、消融实验和结果表生成流程。实现重点不是普通 collision prediction，而是同一 WOMD root scene 下多个 ego intervention candidate 对 critical road users 的 **natural alternatives、safe response set、burden、option preservation ratio、coercion witness** 的影响。

工程遵循以下职责边界：

- **WOMD `Scenario` proto**：权威标签来源，用于解析 tracks、map features、traffic controls、lane topology、priority、conflict regions，并生成 counterfactual labels。
- **WOMD `tf.Example` / tensor cache**：训练期输入容器；重型几何与标签计算已在 Scenario proto 阶段完成，训练时只读 cache。
- **Waymax dataloader / rollout / metrics**：用于 simulator state 初始化、closed-loop rollout 和标准指标；COWP-specific metrics 由本项目自定义 evaluator 在 candidate/rollout 轨迹上重新 certification。

> 单元测试使用 toy scenes 检查几何、priority、burden、witness、cache schema 和 eval wrapper；真实论文实验需要你本地提供 WOMD Scenario proto TFRecord、WOMD tf.Example TFRecord 以及 Waymax/Waymo 依赖。

---

## 1. 项目结构

```text
cowp_project/
  configs/
    data.yaml                 # WOMD 路径、输出路径、split
    label.yaml                # 所有阈值、shape、candidate/natural/response 配置
    model.yaml                # 模型维度与结构
    train.yaml                # 训练阶段、loss 权重、batch 设置
    eval.yaml                 # 评测方法、指标表开关
  cowp/
    core/                     # config、常量、dataclass
    data/                     # Scenario proto 解析、tf.Example 解析、cache、dataset、diagnostics
    geometry/                 # OBB、collision、near-miss、TTC、RSS-like、lane projection
    label/                    # scene filter、critical agents、priority、candidates、alternatives、responses、burden、witness、stress set
    models/                   # graph/candidate/natural/response/witness/planner 模块
    planning/                 # proposal lattice、NCF filter、fallback、planner wrapper
    waymax_eval/              # Waymax dataloader wrapper、standard/COWP metrics、rollout、baseline/ablation、visualization
    scripts/                  # 端到端命令行脚本
  tests/                      # 单元测试
```

---

## 2. 安装

建议 Python 3.10+。当前测试环境使用可选依赖的 lazy import，因此没有 WOMD 数据也能跑核心单测。

```bash
cd cowp_project
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements.txt
```

真实 WOMD/Waymax 运行还需要安装与你的 CUDA/JAX/TensorFlow 环境匹配的包，例如：

```bash
pip install waymax
pip install tensorflow
# 按你的 Python / TensorFlow 版本安装 Waymo Open Dataset 支持包。
# 例如许多环境会使用 waymo-open-dataset-tf-2-x 系列；具体版本以你本机 TF/Python ABI 为准。
```

验证核心逻辑：

```bash
pytest -q
```

本代码包第二轮生成时的单测结果：`15 passed`。

---

## 3. 配置 WOMD 路径

编辑 `configs/data.yaml`：

```yaml
womd:
  scenario_proto_glob: /path/to/womd/scenario/training/*.tfrecord*
  tfexample_glob: /path/to/womd/tfexample/training/*.tfrecord*
  validation_proto_glob: /path/to/womd/scenario/validation/*.tfrecord*
  validation_tfexample_glob: /path/to/womd/tfexample/validation/*.tfrecord*
  include_sdc_paths: true
  waymax_config_name: WOD_1_1_0_TRAINING

outputs:
  root: outputs/cowp
  index_jsonl: outputs/cowp/cowp_index.jsonl
  labels_dir: outputs/cowp/labels
  tensor_cache_dir: outputs/cowp/tensor_cache
  stress_dir: outputs/cowp/stress_set
  diagnostics_dir: outputs/cowp/diagnostics
```

核心阈值都在 `configs/label.yaml` 中，包括：

- `limits.max_candidates / max_critical_agents / max_natural_alternatives / max_safe_responses`
- `unsafe.collision_inflation_m / near_miss_distance_* / ttc_* / rss_*`
- `burden.weights / beta0_* / comfort-hard acceleration thresholds`
- `ncf.alpha_opr / gamma / positive_min_natural_conflict_mass`
- `ablation.use_obs_branch / use_neutral_branch / use_priority_branch / use_option_preservation / use_hard_witness_rejection`

---
## 4. 端到端流程

### 4.1 构建 Scenario proto 索引

```bash
python cowp/scripts/00_index_womd.py \
  --data-config configs/data.yaml \
  --proto-glob '/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/scenario/training/*.tfrecord*' \
  --output outputs/cowp/cowp_index.jsonl
```

索引用于快速检查 scenario id、track 数量、map feature 数量等。完整标签构造不依赖索引，但建议先跑索引确认路径和 proto 依赖正确。

### 4.2 从 WOMD Scenario proto 构造 COWP labels

```bash
python cowp/scripts/01_build_labels_from_proto.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --proto-glob '/path/to/womd/scenario/training/*.tfrecord*' \
  --output-dir outputs/cowp/labels
```

该阶段对每个 scenario 执行：

1. 解析 SDC、agents、timestamps、valid mask。
2. 解析 map features、lane graph、traffic controls、conflict regions。
3. 筛选 interaction-heavy 场景并选择 critical agents。
4. 生成同 root 多个 ego intervention candidates：keep lane、yield、stop before conflict、merge ahead/behind、lane change、accelerate/decelerate cross、logged/neutral ego。
5. 为每个 critical agent 构造 observational、ego-neutral、priority-preserving 三类 natural alternatives。
6. 为每个 candidate-agent pair 构造 safe response set：pred-like、burden-minimizing primitives、emergency primitives。
7. 计算 `Unsafe`、burden components、adaptive beta、OPR、C_i、witness label、mechanism token、conflict interval。
8. 写出 `.npz` label artifacts，并自动运行 dataset diagnostics。

输出包含论文需要的关键字段：

```text
cowp/candidates/conventional_safe
cowp/candidates/false_safe
cowp/candidates/noncoercive_feasible
cowp/natural/traj, weight, source, valid, burden_neutral, priority_preserved
cowp/response/traj, valid, is_safe, is_low_burden, burden_total, burden_components
cowp/witness/exists, token, burden_total, min_safe_burden, natural_conflict_mass, opr, c_i, conflict_interval
```

### 4.3 数据集诊断

```bash
python cowp/scripts/06_diagnose_dataset.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --labels-dir outputs/cowp/labels \
  --output-dir outputs/cowp/diagnostics
```

诊断输出：

```text
outputs/cowp/diagnostics/dataset_diagnostics.csv
outputs/cowp/diagnostics/dataset_diagnostics_summary.json
outputs/cowp/diagnostics/validation_errors.json
```

重点检查：

- `candidate_valid`：每个 scene 是否有足够候选。
- `critical_valid`：interaction-heavy 场景是否有 critical agent。
- `positive_pair_ratio`：stress augmentation 后 witness positive pair ratio 是否约在 5%–30%。
- `false_safe_candidate_ratio`：stress set 中 false-safe candidate ratio 是否约在 10%–40%。
- `ncf_candidate_ratio`：noncoercive candidate 是否约在 20%–60%。
- `mean_opr / mean_max_cbs`：OPR 与 burden 是否合理。
- `mechanism_token_counts`：HB/AY/PA/GS/SR/OR 是否存在，避免全部集中到 HB 或 OR。
- `validation_error_files`：schema、finite、mask、OPR、burden、witness consistency 是否有错误。

### 4.4 构建 false-safe stress set manifest

```bash
python cowp/scripts/07_build_stress_set.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --labels-dir outputs/cowp/labels \
  --output outputs/cowp/stress_set/cowp_stress_manifest.jsonl
```

stress manifest 只保留同一 root scene 中同时存在：

- 至少一个 `noncoercive_feasible` candidate。
- 至少一个 `false_safe` candidate。

这用于论文 stress table：`Accept Non-Coercive↑`、`Accept False-Safe↓`、`Witness Recall↑`、`Witness Precision↑`。

### 4.5 合并 WOMD tf.Example 与 proto-derived labels 为 tensor cache

```bash
python cowp/scripts/02_build_tensor_cache.py \
  --data-config configs/data.yaml \
  --tfexample-glob '/path/to/womd/tfexample/training/*.tfrecord*' \
  --labels-dir outputs/cowp/labels \
  --output-dir outputs/cowp/tensor_cache \
  --tfexample-index-jsonl outputs/cowp/tfexample_training_index.jsonl \
  --build-tfexample-index
```

该阶段把原始 WOMD tf.Example tensors 与 `cowp/*` labels 合并到 `.npz` cache。训练期只读 cache，不重复执行几何/priority/witness label generation。若进度条长期显示 `matched=0`，优先检查 labels 与 tf.Example 是否来自同一 WOMD split/version；`--tfexample-index-jsonl` 会把 scenario id 映射到 shard，后续只扫描命中 shard。

### 4.6 可选：构建 Waymax rollout-augmented label 数据集

```bash
python cowp/scripts/09_build_waymax_rollout_dataset.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --labels-dir outputs/cowp/labels \
  --output-dir outputs/cowp/labels_waymax_rollout \
  --tfexample-glob '/path/to/womd/tfexample/training/*.tfrecord*' \
  --candidate-selection all \
  --background-policy expert \
  --profile-jsonl outputs/cowp/waymax_rollout_profile.jsonl
```

该脚本现在已经补全，并写出统一的 `waymax/*` 字段与 `waymax_rollout_manifest.json`，包括 `waymax/candidate_selected_for_rollout`、`waymax/candidate_rollout_valid`、`waymax/candidate_collision`、`waymax/candidate_offroad`、`waymax/candidate_log_divergence`、`waymax/background_policy` 和 `waymax/rollout_status`。默认情况下，如果本机未安装 Waymax/JAX，脚本会生成状态为 `waymax_unavailable` 的可诊断 label copy，避免 README 中的 rollout dataset 入口缺失；若希望强制真实 Waymax 环境可用，请加：

```bash
  --require-waymax
```

真实候选 replay actor 的插入点在 `cowp/scripts/09_build_waymax_rollout_dataset.py::build_rollout_dataset`，输出 schema 已固定，后续替换为完整 Waymax candidate rollout 不会影响训练与表格读取接口。

---

## 5. 训练

训练脚本支持论文中的四阶段训练。为了避免 witness positive 稀疏，建议使用 `configs/train.yaml` 中的 batch composition、scene-level positive oversampling，以及新增的 pair-level witness mining。`witness_mining_max_pos_per_scene`、`witness_mining_max_neg_per_scene`、`witness_mining_neg_pos_ratio` 会在 `witness_loss` 内保留 positive witness pairs 并选择 hardest negative pairs。

### Stage A：representation pretraining

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage representation \
  --epochs 5 \
  --batch-size 64 \
  --output-dir outputs/checkpoints/representation
```

### Stage B：natural / response supervised training

```bash
python -m cowp.scripts.03_train \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train \
  --val-cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val \
  --stage response \
  --epochs 10 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 1 \
  --amp \
  --compile \
  --fused-adamw \
  --resume outputs/checkpoints/representation/cowp_representation_best.pt \
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
  --epochs 10 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --amp \
  --compile \
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
  --epochs 5 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 1 \
  --amp \
  --compile \
  --fused-adamw \
  --resume outputs/checkpoints/witness/cowp_witness_best.pt \
  --output-dir outputs/checkpoints/planner
```

一次性联合训练也支持：

```bash
python cowp/scripts/03_train.py \
  --data-config configs/data.yaml \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --cache-dir outputs/cowp/tensor_cache \
  --stage all \
  --epochs 20 \
  --output-dir outputs/checkpoints/all
```

---

## 6. 评测与论文表格

### 6.1 Label-only offline eval
确认 rule certificate 的上界/ sanity check
```bash
python -m cowp.scripts.04_eval_closed_loop \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --labels-dir outputs/cowp/formal/labels_val \
  --mode offline \
  --method cowp \
  --output outputs/eval/offline_cowp_val.json
```

### Learned offline eval
看模型是否学到了 witness / ranking:

```bash
python -m cowp.scripts.04_eval_closed_loop \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --cache-dir outputs/cowp/formal/tensor_cache_val \
  --mode learned_offline \
  --method cowp \
  --checkpoint outputs/checkpoints/planner/cowp_planner_best.pt \
  --batch-size 64 \
  --witness-threshold 0.5 \
  --output outputs/eval/learned_offline_cowp_val.json
```

### Waymax closed-loop smoke test
```bash
python -m cowp.scripts.04_eval_closed_loop \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --mode waymax \
  --method cowp \
  --checkpoint outputs/checkpoints/planner/cowp_planner_best.pt \
  --num-scenarios 100 \
  --rollout-horizon-steps 80 \
  --waymax-standard-metrics \
  --witness-threshold 0.5 \
  --output outputs/eval/cowp_waymax_smoke_100.json
```

`--mode waymax --checkpoint` 会使用 `cowp/waymax_eval/policy_wrapper.py` 中的 `COWPWaymaxPolicy`：从 Waymax `SimulatorState` 提取当前 agent state，在线生成轻量 candidate lattice，调用 COWP 模型预测 witness/OPR/planner score，并转成 Waymax action。若你的 Waymax dynamics 使用不同 action 语义，可用 `--waymax-action-mode absolute_xy_yaw`，或继续通过 `--policy-fn module:function` 接入自定义 actor。


闭环输出现在会包含两类 summary：

- `standard_metric_summary`：在 `--waymax-standard-metrics` 开启时，对官方 Waymax metrics 做 JSON scalar 聚合。
- `policy_diagnostic_summary`：COWP policy 在每个 closed-loop step 的在线诊断聚合，包括 `ClosedLoopPredFSR`、`ClosedLoopCBS_pred`、`ClosedLoopOPR_min`、`ClosedLoopFallbackStepRate`。这些是模型预测的 closed-loop certification 信号，用于 smoke/debug；最终论文仍应结合 proto/label counterfactual certification 和官方 Waymax metrics 汇报。


支持的方法名：

```text
idm_lattice
cowp
cowp_wo_counterfactual
cowp_wo_neutral_branch
cowp_wo_priority_branch
cowp_wo_option_preservation
cowp_wo_witness_rejection
soft_burden_cost_only
cowp_wo_dual_edge
cowp_wo_conflict_query
```

### 6.2 生成论文结果表

```bash
python cowp/scripts/05_make_tables.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --labels-dir outputs/cowp/labels \
  --output-dir outputs/tables
```

输出：

```text
outputs/tables/main_results.csv
outputs/tables/ablation.csv
outputs/tables/stress_test.csv
outputs/tables/witness_quality.csv
```

表格列与论文 Experiments 对齐：

- Main：`Method | CR↓ | EP↑ | FSR↓ | CBS↓ | OPR↑ | HBCR↓`
- Stress：`Method | Accept Non-Coercive↑ | Accept False-Safe↓ | Witness Recall↑ | Witness Precision↑`
- Ablation：`FSR↓ | CBS↓ | OPR↑ | EP↑`
- Witness quality：`WLA↑ | MTA↑ | HB-F1↑ | AY-F1↑ | PA-F1↑ | GS-F1↑`

---

## 7. 消融实验开关

以下开关已经做成真实参数，而不是只改 method 名称：

```text
use_obs_branch / use_neutral_branch / use_priority_branch
use_dual_edge / use_conflict_query / use_option_preservation
use_hard_witness_rejection / soft_burden_cost_only
```

其中 natural branch 与 option preservation 会影响 label/certificate；`use_dual_edge` 与 `use_conflict_query` 会影响 `GraphEncoder`；`use_hard_witness_rejection` 与 `soft_burden_cost_only` 会影响 planner selection。新生成的 labels 还包含 `cowp/witness/natural_conflict_mass_by_source` 与 `cowp/witness/low_safe_mass_by_source`，因此 `cowp_wo_neutral_branch` / `cowp_wo_priority_branch` 在离线评测时可以按 source 重新计算 witness，而不是无效开关。

### 7.1 标签构造阶段消融

```bash
# 只使用 observed/logged branch，关闭 ego-neutral 与 priority-preserving branch
python cowp/scripts/01_build_labels_from_proto.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --proto-glob '/path/to/womd/scenario/validation/*.tfrecord*' \
  --output-dir outputs/cowp_ablation/obs_only_labels \
  --no-neutral-branch \
  --no-priority-branch

# 关闭 option preservation
python cowp/scripts/01_build_labels_from_proto.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --proto-glob '/path/to/womd/scenario/validation/*.tfrecord*' \
  --output-dir outputs/cowp_ablation/no_opr_labels \
  --no-option-preservation
```

模型/规划阶段对应开关：

```bash
# GraphEncoder 消融：关闭 conditioned/natural dual-edge 或 conflict query token
python cowp/scripts/03_train.py \
  --model-config configs/model.yaml \
  --train-config configs/train.yaml \
  --data-config configs/data.yaml \
  --cache-dir outputs/cowp/tensor_cache \
  --stage all
# 在 configs/model.yaml 的 ablation.use_dual_edge / use_conflict_query 中切换

# 评测阶段：硬 witness rejection 与 soft burden cost only
python cowp/scripts/04_eval_closed_loop.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --eval-config configs/eval.yaml \
  --labels-dir outputs/cowp/labels \
  --method soft_burden_cost_only \
  --output outputs/eval/soft_burden_cost_only.json
```

### 7.2 评测阶段消融

```bash
for method in \
  idm_lattice \
  cowp \
  cowp_wo_counterfactual \
  cowp_wo_neutral_branch \
  cowp_wo_priority_branch \
  cowp_wo_option_preservation \
  cowp_wo_witness_rejection \
  soft_burden_cost_only \
  cowp_wo_dual_edge \
  cowp_wo_conflict_query
  do
    python cowp/scripts/04_eval_closed_loop.py \
      --data-config configs/data.yaml \
      --label-config configs/label.yaml \
      --eval-config configs/eval.yaml \
      --labels-dir outputs/cowp/labels \
      --method "$method" \
      --output "outputs/eval/${method}.json"
  done
```

---

## 8. Waymax rollout 集成说明

`cowp/waymax_eval/dataloader.py` 提供 lazy Waymax wrapper：

- `make_default_config(...)`
- `simulator_state_generator(...)`
- `simulator_state_from_womd_dict(...)`

`cowp/waymax_eval/rollout.py` 提供两类入口：

- `offline_candidate_eval(...)`：直接使用预生成 labels 做 planner/baseline/ablation 评测，便于快速复现实验表。
- `waymax_closed_loop_rollout(...)`：从 Waymax `SimulatorState` 和 planner 函数执行闭环 rollout。真实环境中可接入 `waymax.env`、`waymax.dynamics`、`waymax.agents`，并在 selected ego trajectory 上用 `metrics_cowp.py` 重新 certification。

标准 Waymax metrics 由 `metrics_standard.py` 封装，包括 overlap/collision、offroad、wrong-way、route-following、kinematic infeasibility、log divergence；COWP metrics 由 `metrics_cowp.py` 计算：

```text
CR, EP, FSR, CBS, OPR, HBCR, WLA, MTA, mechanism-token F1
```

---

## 9. 实现细节对照论文

### Burden-Oriented Interaction Graph

模型输入包含 ego、agents、candidate trajectory、critical agent mask、conflict regions 与 map/traffic-control 摘要。`models/graph_encoder.py` 现在不再只是 type embedding + transformer，而是在 transformer 前注入 typed edge message：candidate-to-agent、agent-to-candidate、candidate/agent-to-conflict、natural/conditioned dual-edge，并可加入 learned conflict-query token。`configs/model.yaml` 中的 `ablation.use_dual_edge`、`ablation.use_conflict_query`、`ablation.use_typed_edges` 是真实结构开关。

### Ego Candidate Generation

`label/ego_candidates.py` 按配置生成 keep/yield/stop/merge/lane-change/cross/logged/neutral candidate，并检查动力学约束、valid horizon 和 ego utility prior。

### Counterfactual Natural Alternatives

`label/natural_alternatives.py` 实现三分支：

- `OBS`：logged future + speed/time/lateral perturbation。
- `NEU`：ego-neutral intervention 下的 constant acceleration / lane-following primitives。
- `PRIO`：priority-preserving branch，保持 arrival order、target-lane gap 或 mainline priority。

权重 `mu` 使用 branch source weight、与 logged trajectory 的距离、neutral burden 归一化得到。`models/natural_decoder.py` 与 `models/losses.py` 已加入 branch source classification、branch-specific minADE、priority preservation loss、neutral branch consistency 与 diversity loss。

### Safe Response Set

`label/safe_responses.py` 实现：

- `R_pred`：logged / natural / mild-yield variants。
- `R_opt`：候选条件化 typed safe-budget search（`label/safe_budget_search.py`），包含 preserve、comfort-yield、yield-recover、hard-yield 和 small-lateral-slack profile，并按 safety → burden → hard-profile cost 排序。
- `R_emg`：emergency braking primitives，用于判断是否只有高负担 response 才能避险。

### Unsafe / Burden / Witness

- `geometry/collision.py`：OBB overlap、near-miss、TTC、RSS-like gap violation、offroad severe。
- `label/burden.py`：ACC/JERK/PROG/RISK/OPTION/NORM 六分量和 adaptive beta。
- `label/witness.py`：`natural_conflict_mass >= threshold AND min_safe_burden > beta` 生成 witness；candidate-level `false_safe` 与 `noncoercive_feasible` 按论文定义计算；mechanism token 使用 HB/AY/PA/GS/SR/OR 规则优先级。

---

## 10. 质量标准

一次完整数据构造后，建议至少满足：

```text
candidate_valid.mean >= 8
critical_valid.mean >= 1 for interaction-heavy scenes
positive witness pair ratio after stress augmentation: 5% - 30%
false-safe candidate ratio in stress set: 10% - 40%
noncoercive candidate ratio: 20% - 60%
mechanism tokens include HB/AY/PA/GS/SR/OR, not all HB/OR
validation_error_files == 0
```

若 positive 太少，增大 aggressive ego candidates 的采样范围，例如更早 merge、更小 gap、更快 crossing。若 positive 太多，优先检查 near-miss/RSS thresholds 与 natural alternatives 是否过激。

---

## 11. 重要限制与复现实验注意

1. 本仓库不包含 WOMD 数据，也不会生成 synthetic-only 数据集；真实数据集必须由 WOMD Scenario proto 和 tf.Example 构造。
2. 单元测试中的 toy scene 只用于验证函数正确性，不用于论文结论。
3. WOMD/Waymax 的包版本和 TensorFlow/JAX/CUDA 组合经常受本地环境影响；建议先用 `--limit 100` 跑标签构造和诊断，再扩大到完整 train/val/test。
4. Waymax closed-loop 的非 ego policy mixture 可在 `configs/eval.yaml` 调整；FSR/OPR/HBCR 必须用本项目 counterfactual evaluator certification，不应直接用 rollout 中实际他车是否刹车替代。

---

## 12. 快速 smoke test

```bash
pytest -q
python cowp/scripts/01_build_labels_from_proto.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --proto-glob '/path/to/womd/scenario/training/*.tfrecord*' \
  --output-dir outputs/cowp/labels_smoke \
  --limit 100
python cowp/scripts/06_diagnose_dataset.py \
  --data-config configs/data.yaml \
  --label-config configs/label.yaml \
  --labels-dir outputs/cowp/labels_smoke \
  --output-dir outputs/cowp/diagnostics_smoke
```

## Notes for v3 critical-agent alignment

After this version, labels should be regenerated before rebuilding tensor cache so that each critical agent carries `cowp/critical/track_id`. The tensor-cache builder then maps Scenario track ids to WOMD tf.Example input rows and writes `cowp/critical/input_index`, which is what the model uses for gathering agent embeddings.

Recommended rebuild order:

```bash
python -m cowp.scripts.01_build_labels_from_proto ... --output-dir <labels_train_new> ...
python -m cowp.scripts.01_build_labels_from_proto ... --output-dir <labels_val_new> ...
python -m cowp.scripts.02_build_tensor_cache ... --labels-dir <labels_train_new> --output-dir <tensor_cache_train_new> ...
python -m cowp.scripts.02_build_tensor_cache ... --labels-dir <labels_val_new> --output-dir <tensor_cache_val_new> ...
python -m cowp.scripts.11_diagnose_tensor_cache_visibility --cache-dir <tensor_cache_val_new> --output <visibility_val_new.json>
```

If `files_with_id_mapping` is close to `num_files` and `visible_critical_slot_ratio` is still low, the remaining issue is true WOMD input exclusion/current invisibility rather than a Scenario-index/input-row mismatch.

For Stage-B response training, `cowp/response/traj` is a dense `[K,A,R,T,7]` target and can make each batch very large. `03_train.py` therefore keeps DataLoader `pin_memory=False` by default and uses `prefetch_factor=1` by default for response/planner/all stages. This does not change model learning or predictions; it only avoids CUDA pinned-memory failures. You may still force pinning with `--pin-memory` after confirming your host/CUDA setup is stable.
