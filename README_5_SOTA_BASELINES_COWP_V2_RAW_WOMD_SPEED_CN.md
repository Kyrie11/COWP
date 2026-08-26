# COWP v16.8.33 + 5 外部 Planner：Raw-WOMD 闭环与性能优化版

## 1. 最终建议：训练用 NPZ cache，闭环仿真用原始 WOMD validation TFRecord

这两种数据不是二选一，而应承担不同职责。

- **训练 / validation**：继续使用从 WOMD training / validation 场景构建的 `tensor_cache_train` / `tensor_cache_val` NPZ。它们是离线 feature cache，可以显著减少每个 epoch 重复解析 TFRecord、构图和 map tokenization 的成本；只要 split、历史窗口、map/route 信息和监督 future 与原 WOMD 保持一致，使用 cache 不会让训练“不原生”。
- **最终 Waymax closed-loop**：必须以原始 WOMD 1.3.1 validation `tf_example` 作为 simulator source-of-truth。本版默认路径已经改成：

```text
/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/
```

- **held-out 1200**：仍使用 `reference_manifests/formal_v16_8_24_compact5k_heldout1200_ids.txt` 精确选择官方 validation 内的 1200 个 scenario ID。也就是说，测试场景来源仍是原始 WOMD validation，只是用你的 held-out manifest 做子集选择。
- **PBTR / FSR / OPR / BTE 等 COWP mechanism audit**：可以继续使用 `labels_heldout_test/*.npz`，前提是它们与上述 1200 scenario ID 一一对应。它们是离线 mechanism diagnostics，不应被描述成 Waymax logged-replay 下的 counterfactual ground truth。

### 为什么 closed-loop 不应只读你的 NPZ

Waymax 的 `SimulatorState` 会随每一步 action 更新；raw TFExample 中还包含 roadgraph、对象轨迹结构和 WOMD 1.3.1 的 `sdc_paths`。WrongWay / Progression / OffRoute 等 route metrics 需要这类 route/path 信息。静态 NPZ 适合训练缓存，但不应替代 simulator state。

本版 `configs/data.yaml` 明确增加：

```yaml
validation_waymax_config_name: WOD_1_3_1_VALIDATION
```

并且 `RUN_5_SOTA_BASELINES_COWP.sh` 默认从上面的 raw validation 目录构建 exact-ID TFExample index。

## 2. 本版修复的闭环正确性问题

### 2.1 去除 DTPP / PDM-Closed 的 logged-future leakage

旧版 candidate path 会从 `SimulatorState` 提取 logged future agent trajectories，然后用于候选碰撞筛选。Waymax 为 replay/evaluation 保存完整 logged trajectory，并不意味着 planner 在时刻 t 可以读取 t 之后的真实轨迹。

本版完全移除了外部 baseline 对 `_extract_logged_future_agent_trajs` 的依赖：

- planner 只读取当前及之前的 11-frame history；
- DTPP 周车未来来自它自己的 ego-conditioned predictor；
- PDM-Closed-style 使用当前/历史观测和自己的 predictive/rule scoring；
- non-ego logged replay 仍由 Waymax environment 自己执行，planner 不获得未来 GT。

这同时提高了公平性和速度。

### 2.2 direct planner 真正使用 Waymax `sdc_paths`

GameFormer / PLUTO / PlanT2 direct mode 现在从 `state.sdc_paths` 读取 route token，而不是仅从 roadgraph 近邻点近似 route。`sdc_paths` 和 roadgraph 对同一个 scenario 是静态的，host array 会按 scenario 缓存；每一步只按 ego 当前 pose 重做 ego-frame transform。

### 2.3 direct planner 不再生成无用 COWP 候选 / witness / conflict feature

GameFormer、PLUTO、PlanT2 的 direct trajectory head 不需要 COWP candidate bank。旧版每 0.1 s 仍会运行 proposal generation、critical pair ranking 和 conflict token extraction。本版 direct path 使用最小 observation adapter，完全跳过这些步骤。

DTPP 和 PDM-Closed-style 因自身属于 candidate/tree/proposal planner，仍保留 proposal generation。

## 3. 训练和闭环的主要速度瓶颈与优化

| 瓶颈 | 旧行为 | 本版 | 是否改变算法逻辑 |
|---|---|---|---|
| NPZ 解压/IPC | 每种方法都加载所有外部 + COWP key | baseline-specific key whitelist | 否 |
| feature 构造 | 每种方法同时构造 GF/DTPP/PLUTO/PlanT2 输入 | 只构造本模型实际消费的 tensor | 否；有精确 tensor 等价测试 |
| roadgraph GPU copy | 最多 30k 点先搬 GPU 再截断 | CPU/host 先截断后 non-blocking copy | 否 |
| direct closed-loop | 每步生成 COWP candidate/critical/conflict | 最小 observation-only path | 否 |
| static map/path | 每 0.1 s JAX -> NumPy 重复 copy | scenario-local cache | 否 |
| DTPP branch scorer | Python 循环最多 K=30 | B×K vectorization | 否；与旧 loop 数值等价 |
| checkpoint I/O | 每 epoch 保存 numbered checkpoint | best 始终保存；普通 checkpoint 默认每 5 epoch | 否 |
| DataLoader | 保守 worker/prefetch | pin + persistent workers + prefetch | 否 |
| 两 GPU 总 wall time | 4 个 learning baselines 顺序训练 | 两个独立 baseline 各占一张 GPU | 否；每个 run seed/optimizer/batch 不变 |

### 合成 CPU adapter microbenchmark

`BASELINE_ADAPTER_CPU_BENCHMARK.json` 是本地合成 tensor 的 **adapter-only** 测试，B=4, N=64, H=11, T=80, K=30, roadgraph=30000，另外含 45×800 SDC paths。它不包括 NPZ 磁盘 I/O、GPU model forward、TensorFlow 或 Waymax，因此只能用于确认被删掉的 CPU feature-work 确实存在，不能当成真实训练/闭环 FPS。

本次运行的 mean adapter latency：

| planner | 旧 full-adapter | baseline-specific | synthetic speedup |
|---|---:|---:|---:|
| GameFormer | 7.96 ms | 1.49 ms | 5.33× |
| DTPP | 6.80 ms | 3.15 ms | 2.16× |
| PLUTO | 7.85 ms | 2.97 ms | 2.64× |
| PlanT2 | 6.27 ms | 3.18 ms | 1.97× |

DTPP scorer 的独立合成 CPU 测试见 `DTPP_SCORE_CPU_BENCHMARK.json`：旧 Python branch loop 54.33 ms，新 B×K vectorized 46.17 ms，约 1.18×。GPU 实际收益必须用你的机器上的 profiler 测量，本报告不外推 CPU 数字。

## 4. 真实 raw-WOMD profiler

先训练好 learning baselines，然后跑少量 exact held-out IDs：

```bash
PROFILE_NUM_SCENARIOS=24 \
WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
GPU0=0 GPU1=1 \
bash RUN_5_SOTA_BASELINES_COWP.sh profile all
```

每个方法输出：

```text
outputs/external_sota5_v16_8_33/<method>/profile_waymax.json
```

会汇总 `timing_ms/*` mean / P95。主要观察：

- `observation`
- `static_map_host`
- `proposal_generation`（DTPP/PDM）
- `adapter_and_model` 或 `rule_scoring`
- `total_before_action`

建议先用这个 profiler 再调整 `NUM_WORKERS`。如果 `/data0` 是共享 HDD/NFS，worker 过多反而会变慢；如果是本地 NVMe，可从 8 增到 12/16 做对比。

## 5. 复现忠实度审计

必须区分“论文/官方 benchmark 原生实现”和“跨域 WOMD adaptation”。当前 5 个方法并不具备同一等级的 strict reproduction。

| 方法 | 官方原生 domain | 当前 WOMD 实现忠实度 | 论文表格建议名称 |
|---|---|---|---|
| GameFormer | WOMD open-loop / interaction prediction；官方不发布 WOMD closed-loop | network 结构高；Waymax closed-loop wrapper 为本工程适配 | `GameFormer (WOMD/Waymax adaptation)` |
| DTPP | nuPlan | encoder / ego-conditioned prediction / learned cost 较高；native spline tree 被 shared WOMD proposal tree 替代 | `DTPP (WOMD shared-tree adaptation)` |
| PLUTO | nuPlan | clean-room、保留 vector/query/IL/aux/CIL 思路，但不是官方代码逐行移植 | `PLUTO-style WOMD adaptation` |
| PlanT 2.0 | CARLA | 保留 object-centric transformer abstraction；CARLA token/control/training stack 未复制 | `PlanT2-style WOMD adaptation` |
| PDM-Closed | nuPlan | 保留 predictive rule-based proposal scoring；native centerline/BatchIDM/observation stack 未完整复制 | `PDM-Closed-style WOMD adaptation` |

因此，如果论文正文写“strictly reproduced official planners”，当前说法仍然过强。更严谨的是写 **source-faithful cross-domain adaptations under a unified WOMD/Waymax protocol**，并把上表放在 appendix/protocol 中。

默认模型参数量（当前 WOMD adaptation）：GameFormer 15.37M，DTPP 5.90M，PLUTO 1.05M，PlanT2 1.00M。PLUTO/PlanT2 参数量本身也提示它们是较轻的 clean-room adaptation，而不是对官方 native stack 的逐层复刻。

## 6. `.bib` 审计

上传的 `interactive planning.bib` 已包含：

- `huang2023gameformer`
- `huang2024dtpp`
- `renz2022plant`（原始 PlanT，不是 PlanT 2.0）
- WOMD / Waymax

但缺少：

- `cheng2024pluto`
- `dauner2023parting`
- `gerstenecker2025plant20exposingbiases`

本版提供 `docs/bib/interactive_planning_baseline_refs_additions.bib`。

另外，你上传的 `.bib` 中 `gulino2023waymax` 出现了两次相同 citation key。Biber/BibLaTeX 前建议只保留一个，否则可能产生 duplicate-entry warning/error。

## 7. 推荐运行命令

### 7.1 路径

```bash
export WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1
export WOMD_VALIDATION_TFEXAMPLE_DIR=$WOMD_ROOT/uncompressed/tf_example/validation
```

### 7.2 两 GPU 并行训练 4 个 learning baselines

```bash
GPU0=0 GPU1=1 \
NUM_WORKERS=8 VAL_NUM_WORKERS=4 \
PREFETCH_FACTOR=4 VAL_PREFETCH_FACTOR=2 \
CHECKPOINT_EVERY=5 \
bash RUN_5_SOTA_BASELINES_COWP.sh train_parallel2 all
```

如果单模型需要单独训练：

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash RUN_5_SOTA_BASELINES_COWP.sh train gameformer
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash RUN_5_SOTA_BASELINES_COWP.sh train dtpp
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash RUN_5_SOTA_BASELINES_COWP.sh train pluto
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda:0 bash RUN_5_SOTA_BASELINES_COWP.sh train plant2
```

PDM-Closed-style 不训练。

### 7.3 先 profile 再全量 closed-loop

```bash
PROFILE_NUM_SCENARIOS=24 \
bash RUN_5_SOTA_BASELINES_COWP.sh profile all
```

### 7.4 held-out 1200 的离线 mechanism audit

```bash
bash RUN_5_SOTA_BASELINES_COWP.sh offline all
```

### 7.5 raw WOMD validation 上的 1200-scene Waymax closed-loop

```bash
PARALLEL2=1 GPU0=0 GPU1=1 \
WOMD_VALIDATION_TFEXAMPLE_DIR=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation \
bash RUN_5_SOTA_BASELINES_COWP.sh waymax all
```

脚本内部默认：

```text
TFEXAMPLE_GLOB=$WOMD_VALIDATION_TFEXAMPLE_DIR/*.tfrecord*
rollout_horizon_steps=80
waymax_action_mode=absolute_xy_yaw
```

并自动构建 `scenario_id -> TFExample shard/record` exact-ID index；后续复用该 index，避免每个 baseline 重扫 150 个 validation shards。

### 7.6 COWP 同 split raw-WOMD closed-loop

保持 v16.8.33 的 fail-closed gate：

```bash
WAYMAX_STANDARD_METRIC_NAMES=OverlapMetric,OffroadMetric,WrongWayMetric,ProgressionMetric,OffRouteMetric,KinematicsInfeasibilityMetric,LogDivergenceMetric \
PROMOTED_METHODS=cowp_recovery_option_spectrum_hysteresis \
bash NEXT_RUN_COMMANDS_V16_8_33_RECOVERY_OPTION_SPECTRUM_CN.sh heldout1200_parallel2
```

### 7.7 汇总

```bash
COWP_JSON=outputs/v16_8_33_recovery_option_spectrum/heldout1200_v33_cowp_recovery_option_spectrum_hysteresis_merged.json \
bash RUN_5_SOTA_BASELINES_COWP.sh summary
```

## 8. 我没有改变的实验逻辑

为避免“为了提速改变结果”，本版没有默认做以下事情：

- 不减少 80-step horizon；
- 不减少 DTPP/PDM candidate 数；
- 不改变 loss 权重；
- 不改变 optimizer / scheduler / seed；
- 不默认启用可能改变 numerical path 的 `torch.compile`；
- 不为了速度改成更低精度的 optimizer/state；
- 不把 COWP witness/OPR/burden 标签泄漏给外部 baseline 训练。

本版加速主要来自少做无关工作、缓存 scenario-static 数据、减少 I/O 和等价向量化。
