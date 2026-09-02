# OC-RAP 数据集构建加速与论文一致性检查

本文档记录本次代码修改的目标、修改点、复现实验建议，以及目前代码相对论文主张仍需要补齐的部分。

## 1. 计时日志定位出的瓶颈

根据 `stress_val_time.txt` 与 `strict_val_time.txt` 的 per-prefix profiling：

| validation set | 样本数 | mean total | mean future | future 占比 | mean teacher | teacher 占比 | obs/root/OC-MERO |
|---|---:|---:|---:|---:|---:|---:|---:|
| stress/contact val | 12 | 39.70s | 14.74s | 37.1% | 24.69s | 62.2% | <1% |
| strict/near-contact val | 6 | 33.06s | 22.43s | 67.9% | 10.48s | 31.7% | <1% |

结论：慢的不是 OC-MERO、root clustering 或 observation kernel，而是 Waymax counterfactual future rollout 与 hybrid teacher 的 Waymax recovery metric rollout。

## 2. 已实现的语义保持型加速

这些修改不改变候选 prefix、future 类型、prior、teacher margin 定义、筛选阈值或保存字段，只减少重复 rollout、重复结构化 margin 计算和 JAX dispatch。

### 2.1 默认缓存 Waymax environment

文件：`configs/default.yaml`, `src/ocrap/config/defaults.py`, `src/ocrap/simulation/waymax_rollout.py`

`waymax.cache_env_objects` 默认改为 `true`。缓存键包含 dynamics 名称、object 数、init steps、allow_new、metrics 列表、采样率和控制限，因此仅复用等价的 stateless `BaseEnvironment`。这能让 `_rollout_bicycle_controls_scan` 的 JIT 函数更稳定复用，避免每个 prefix/branch 构造新 env 后触发额外 compile/cache miss。

### 2.2 post-prefix future rollout 精确缓存

文件：`src/ocrap/simulation/waymax_rollout.py`

新增 `waymax.cache_postprefix_rollouts=true`。同一个 `(state_after_prefix, env, post_steps, coast_accel)` 的 constant-control future 只 rollout 一次。stress/strict 构建中 targeted fill 常出现重复 `-2.0` 与 `1.2` accel，这部分 trajectory 本来完全相同，缓存只复用 deterministic state，不改变样本分布。

### 2.3 hybrid teacher 的 Waymax metric rollout 精确缓存

文件：`src/ocrap/simulation/waymax_rollout.py`

新增 `waymax.cache_teacher_metric_rollouts=true`。replay/reactive/部分 targeted future 会共享同一个 post-prefix simulator state 和 env；它们的 Waymax recovery metric rollout 对同一 recovery option 是 deterministic 且相同。现在仅缓存 Waymax metric 部分，仍然逐 future 重新计算 branch-specific structural margin，并继续使用 `min(waymax_metric_margin, structural_margin)`。因此不会把 reactive/targeted roots 错误折叠成同一行，也不会擦除 oracle-to-deployable gap。

### 2.4 hidden/visible reference augmentation 向量化

文件：`src/ocrap/simulation/waymax_rollout.py`

原来 hidden/visible augmentation 对每个时间步调用 `jnp.ndarray.at[...].set`，在 Python 循环中产生大量 device dispatch。现在改成 host NumPy 数组一次性向量化修改，再 `jnp.asarray` 回写。轨迹公式、spawn/slot 选择、valid mask 和 metadata 保持不变。

### 2.5 避免无 override 时重复 structural teacher 计算

文件：`src/ocrap/simulation/waymax_rollout.py`

hybrid teacher 原本对每个 `(future, option)` 都调用两次 `teacher_margin`：一次 no-override 用于 top-k screening，一次 label-side cfg。当前 README 的 strict/stress 主命令中 `artifact.use_margin_override=false`，这两次结果完全相同。现在仅当 artifact override 可能触发时才做第二次计算，否则复用第一次结果。

### 2.6 README 指令修正

文件：`README.md`

- build 命令显式加入三个 cache flag。
- post-contact train shard 命令补上行尾 `&`，否则 `for i in 0 1 2 3` 实际串行。
- “内部 test split 评估”从误写的 `calibrate` 改为 `evaluate --dataset "$TRAIN_MIX" --split test`。
- “外部 validation set 评估”从误用 `$TRAIN_MIX --split test` 改为 `$VAL_MIX --split val`。

## 3. 继续生成剩余 30% strict/stress train 的建议

为了不改变已经生成 70% 的分片边界，继续使用原来的 `scenario_stride=4` 与 `scenario_worker_index=0..3`，加上 `--skip-existing`。两张 A30 上建议把四个 shard 分配到两张卡：

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((i % 2)) python -m ocrap.cli build-dataset \
    --skip-existing \
    ...原 README 中该 regime 的其它参数保持不变... \
    --set waymax.cache_env_objects=true \
    --set waymax.cache_postprefix_rollouts=true \
    --set waymax.cache_teacher_metric_rollouts=true \
    --output ${OCRAP_ROOT}/train_xxx_v1_w${i} &
done
wait
```

如果显存或 host RAM 抖动，建议一次只跑两个进程：`w0,w1` 完成后再跑 `w2,w3`。不要在续跑阶段把现有四分片改成 `scenario_stride=2`，否则会改变 shard 覆盖方式，容易和已有产物重复或漏场景。

## 4. 仍会影响论文主张论证的主要问题

### 4.1 当前 evaluate 不是论文中真正的闭环 replanning protocol

论文描述的是 planner 在每个控制间隔重新观测、重新选择 1s prefix，再推进 simulator。当前 `src/ocrap/evaluation/evaluator.py` 是离线 one-shot prefix selection：按 `(scene_id, time_index)` 分组，在预生成 candidates 中选一个，然后用该 sample 的 teacher label 统计 FRA/DRS/ODG/NUP。它适合做 dataset-level selection benchmark，但还不能证明“闭环管道”里的长期安全与二次碰撞恢复。

建议后续新增独立 `closed_loop_evaluate`：输入 WOMD/Waymax scenario 与 checkpoint；每个 replanning tick 重新构造 SceneHistory、生成 candidates、模型选择 prefix、Waymax 推进控制若干步；累计 collision/offroad/secondary collision/progression/recovery success。论文主表如果写 closed-loop，就应使用该 evaluator，而不是当前离线 evaluate。

### 4.2 strict/stress README 里 balanced_two_pass 可能没有真正生效

代码中 `dataset_quality.balanced_two_pass` 只有在 `dataset_quality.artifact_pair_mode=balanced` 时才进入双 pass 逻辑。当前 README strict/stress 命令设置了 `balanced_two_pass=true` 和 quota，但没有显式设置 `artifact_pair_mode=balanced`，默认仍是 `tag`。这不会影响你已经生成的数据继续补齐，但会削弱“严格控制 artifact/non-artifact 比例”的论证。若要做最终论文主实验，建议重新生成或至少报告当前数据的 `papercheck` 统计，并在新的主实验命令中显式加入：

```bash
--set dataset_quality.artifact_pair_mode=balanced
```

### 4.3 `artifact.use_margin_override` 相关实验需要严格分离

论文主张需要证明 oracle-to-deployable gap 来自隐藏/观测不可区分与真实 recovery teacher，而不是人工 override。README 主命令已经把 `artifact.use_margin_override=false`，这是正确方向。若使用 override 生成 sanity/stress 数据，应单独标注为 diagnostic 或 sanity set，不能混作主表证据。

### 4.4 当前 learned model 评估容易被 teacher fallback 掩盖

`predict_sample()` 在 checkpoint 不存在时会返回 teacher prediction；`evaluate()` 的 `source` 会标成 `teacher_fallback`。如果误把 fallback 结果当成模型结果，会高估 OC-RAP。最终表格中必须确认 checkpoint path 存在，且输出 JSON 中 `source == "model"`。

### 4.5 DRS 对 baseline 的含义偏上界

当前 DRS 对非 OC-RAP 方法使用 teacher OC-MERO 的 shared recovery option 来评估 deployable recovery success。这让 branchwise backup/contingency baseline 在 recovery execution 上带有 shared-option 上界色彩，而不是它们自己实际会执行的 branch-specific option。建议论文中把该指标称为 selected-prefix deployable recovery headroom，或者为每个 baseline 明确实现其 recovery option selection 后再算执行 DRS。

### 4.6 post-contact / near-contact regime 仍要靠 papercheck/diagnose 证明覆盖率

数据生成逻辑有 post-contact、secondary-collision、visible perturbation、hidden artifact 等 metadata，但论文主张需要报告每个 regime 的覆盖率：`near_contact`、`post_contact`、`occluded`、artifact fraction、negative deployable fraction、ODG 正值均值、candidate count per scene-time。当前已有 `papercheck`/`diagnose`，最终实验必须把这些统计作为 dataset table 或 appendix，否则主张难以复核。

## 5. 本次验证

在当前环境中运行：

```bash
PYTHONPATH=src pytest -q
```

结果：`27 passed`。Waymax/WOMD 全量生成未在当前环境运行，因此加速幅度需要在你的 A30 机器上用同一条 profiling command 对比。预期收益主要来自 teacher metric cache 与 env/JIT cache；stress/contact validation 中 teacher 占比约 62%，strict/near-contact validation 中 future 占比约 68%，两者都会受益。
