# COWP v16.8 执行说明

## 一、重要原则

1. 必须使用新的 `OUT_ROOT`，不得 resume v16.7 transport/planner。
2. immediate mechanism isolation 可以复用已验证的 v16.6 natural checkpoint 和原 raw Waymax cache，但必须重新生成 v16.8 RCOT transport overlay。
3. mechanism gate 未通过前，不运行 full Waymax。
4. immediate run 不修改 proposal bank，确保结果只归因于 RCOT/certificate 修复。

---

## 二、解压代码

```bash
unzip COWP_v16_8_rcot_optimized.zip
cd COWP_v16_8_rcot_optimized
```

建议先检查：

```bash
pytest -q
python -m compileall -q cowp tests
bash -n PREPARE_COWP_V16_8_OVERLAY_CN.sh
bash -n NEXT_RUN_COMMANDS_V16_8_MECHANISM_CN.sh
bash -n NEXT_RUN_COMMANDS_V16_8_FULL_CN.sh
```

---

## 三、第一阶段：重建 v16.8 RCOT overlay

把 `COWP_ROOT` 指向当前包含以下 raw cache 的目录：

- `tensor_cache_train_waymax`
- `tensor_cache_val_waymax`

执行：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
AUG_WORKERS_TRAIN=12 \
AUG_WORKERS_VAL=8 \
bash PREPARE_COWP_V16_8_OVERLAY_CN.sh
```

默认输出：

```text
/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_transport_v16_8
/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v16_8
```

查看标签诊断：

```bash
python -m json.tool \
/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_transport_v16_8/transport_diagnostics_v16_8.json

python -m json.tool \
/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v16_8/transport_diagnostics_v16_8.json
```

必须首先确认：

- `error_count = 0`；
- `pass = true`；
- canonical transported OPR consistency error 接近 0；
- root target confidence 有有效覆盖；
- train/val conflict-root recovery 分布没有异常断裂；
- cache alignment gate 通过。

不要只比较 `root_recovery_mean` 是否变大。正确目标是 per-root RCOT oracle 语义一致，不是人为提高正类比例。

---

## 四、第二阶段：隔离训练 mechanism

以下路径沿用上一轮已验证 natural 结果；按服务器实际路径修改：

```bash
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_transport_v16_8 \
VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v16_8 \
OUT_ROOT=outputs/cowp_v16_8_rcot_v9base_seed2026 \
BACKGROUND=0 \
FORCE_TRAIN=1 \
FORCE_EVAL=1 \
TRANSPORT_AMP=1 \
PLANNER_AMP=1 \
bash NEXT_RUN_COMMANDS_V16_8_MECHANISM_CN.sh
```

说明：

- natural 不重训；
- transport 从 natural checkpoint 新训练；
- planner 默认冻结 SetTransport 和 response decoder；
- calibration budget sweep 扩展到 0.98；
- proposal bank 保持不变。

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_8_rcot_v9base_seed2026 \
bash CHECK_RUN_STATUS_V16_8.sh
```

查看机制报告：

```bash
python -m json.tool \
outputs/cowp_v16_8_rcot_v9base_seed2026/eval/learned_offline/mechanism_verification.json
```

重点字段：

```text
pass
calibration_feasible
priority_root_transport_auprc
priority_accept_ncf_recall
priority_accept_ncf_precision
learned_accepted_candidate_rate
fallback_rate
priority_burden_transfer_rate
priority_transfer_improvement
```

### 继续运行 Waymax 的最低条件

必须同时看到：

```text
pass = true
calibration_feasible = true
```

不要把 `least_violation` 当成通过。

若 `priority_root_transport_auprc < 0.50`，先停止，不要跑 full Waymax。此时需要上传下一节所列 dataset diagnose 输出。

---

## 五、第三阶段：真实 Waymax probe/full

mechanism gate 通过后：

```bash
OUT_ROOT=outputs/cowp_v16_8_rcot_v9base_seed2026 \
BACKGROUND=1 \
RUN_PROBE=1 \
PROBE_SCENARIOS=100 \
RUN_FULL=1 \
FULL_SCENARIOS=1000 \
ROLLOUT_HORIZON=80 \
bash NEXT_RUN_COMMANDS_V16_8_FULL_CN.sh
```

状态：

```bash
OUT_ROOT=outputs/cowp_v16_8_rcot_v9base_seed2026 \
bash CHECK_RUN_STATUS_V16_8.sh
```

主结果必须至少报告：

- overlap/collision；
- offroad；
- wrong-way；
- route progression；
- kinematic infeasibility；
- log divergence；
- PBTR；
- protected OPR；
- BTE-CVaR25；
- NCF scene retention；
- non-coercive progress regret；
- fallback/coverage。

---

## 六、若 mechanism 仍失败：生成数据集 diagnose

### 1. RCOT transport label 诊断

```bash
python -u -m cowp.scripts.27_diagnose_transport_labels \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_train_waymax_transport_v16_8 \
  --workers 8 \
  --output /tmp/transport_train_v16_8_diagnose.json

python -u -m cowp.scripts.27_diagnose_transport_labels \
  --cache-dir /data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v16_8 \
  --workers 8 \
  --output /tmp/transport_val_v16_8_diagnose.json
```

### 2. 完整 label dataset 诊断

若服务器仍保留原 labels 目录：

```bash
python -u -m cowp.scripts.06_diagnose_dataset \
  --data-config configs/data.yaml \
  --label-config configs/label_cowp_v16_8.yaml \
  --labels-dir /data0/senzeyu2/dataset/COWP/formal/labels_train \
  --output-dir /tmp/cowp_v16_8_dataset_diagnose_train
```

建议上传：

- `transport_train_v16_8_diagnose.json`；
- `transport_val_v16_8_diagnose.json`；
- mechanism verification JSON；
- BCOT calibration JSON；
- transport/planner history JSON；
- dataset diagnose 汇总 JSON；
- 若可行，10–20 个 q false-positive / false-negative 场景可视化。

下一轮应重点按以下维度分层：priority relation、conflict geometry、natural source、root mass、time-to-conflict、object type、beta、oracle profile、target confidence。

---

## 七、第四阶段：paper-grade fresh data

priority/candidate/map screening 和 RCOT label 都需要最终 fresh rebuild：

```bash
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v18 \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
CUDA_VISIBLE_DEVICES=0 \
MAX_REPLAY_CANDIDATES=24 \
RUN_WAYMAX_REPLAY=1 \
bash PREPARE_COWP_V16_8_DATA_CN.sh
```

之后重新运行：

1. natural recovery；
2. exact aligned attribution；
3. RCOT transport；
4. frozen-certificate planner；
5. learned-offline mechanism；
6. real Waymax online；
7. seeds `2026/2027/2028`。

最终 paper 不应使用 v9base overlay 作为主表数据；它只用于隔离验证机制修复。

---

## 八、下一项算法实验顺序

只有 RCOT mechanism gate 通过后才执行：

1. 固定 RCOT checkpoint；
2. baseline proposal bank；
3. coercion-aware conflict-time proposal refinement；
4. aggregate severe hard-veto ablation；
5. scene-adaptive PBTR/coverage calibration；
6. 100-scene probe；
7. 1000-scene full；
8. 3-seed fresh data。

不要同时解冻 RCOT、修改 proposal、修改 threshold；否则无法归因。
