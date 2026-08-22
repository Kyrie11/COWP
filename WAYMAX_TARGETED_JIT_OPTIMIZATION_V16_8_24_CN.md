# COWP v16.8.24 Waymax outcomes：基于实测 probe 的针对性无损优化

## 1. Probe 结论

本次优化依据用户上传的 `20260822_154331/timing_summary.json` 与既有 623-scene scene profile，而不是静态猜测。

同步（`block_until_ready`）probe 的每 candidate 均值：

| Stage | s/candidate | 占所列阶段 |
|---|---:|---:|
| Waymax `OverlapMetric` | 8.1330 | 54.1% |
| Waymax `OffroadMetric` | 3.5804 | 23.8% |
| Waymax `env.step` | 3.1927 | 21.2% |
| done check | 0.0535 | 0.36% |
| policy/action | 0.0485 | 0.32% |
| env.reset | 0.0233 | 0.15% |

前三项合计约 99.2%。因此继续优化 NPZ I/O、JSONL、done-check 或 policy 对总时间几乎没有意义。

623-scene 历史 profile 同样表明 `rollout_candidates_s` 完全主导 scene wall time；load/build/write 不构成瓶颈。

## 2. 本版本采用的优化

### 2.1 `env.step` JIT

沿用已有 `--jit-env-step`，但正式使用前必须通过新的 eager-vs-optimized gate。

### 2.2 新增 `--jit-safety-metrics`

这是本轮最关键改动。

`_FastSafetyMetricAccumulator` 原本虽然避免了每 step host copy，但 **Waymax 自己的 `OverlapMetric.compute()` 与 `OffroadMetric.compute()` 仍然 eager 执行**。Probe 显示二者合计约 78% candidate 时间。

新路径：

```text
Waymax 原始 OverlapMetric.compute
        +
原始 SDC scalar 选择 / valid mask / nan_to_num / episode max
        -> jax.jit（每 worker 只创建一次）

Waymax 原始 OffroadMetric.compute
        +
同一 SDC scalar / episode max
        -> jax.jit（每 worker 只创建一次）
```

没有改变：

- metric class；
- metric 数学公式；
- per-step 评估频率；
- collision/offroad episode aggregation；
- candidate selection；
- horizon；
- Waymax StateDynamics/action mode；
- non-ego logged replay。

JIT wrapper 若在本机 Waymax/JAX 上失败，会永久 fallback 到原 eager function；gate 会检测到 JIT 没真正 active，从而阻止误以为已经加速。

### 2.3 不启用 `env.reset` JIT

Probe 中 reset 只有约 0.023 s/candidate，因此正式推荐：

```bash
WAYMAX_JIT_ENV_RESET=0
```

不为不足 0.2% 的热点增加额外编译路径。

### 2.4 JAX persistent compilation cache

每张物理 GPU 使用稳定独立目录：

```text
$COWP_ROOT/.jax_compilation_cache_waymax_v16_8_24/gpu0
$COWP_ROOT/.jax_compilation_cache_waymax_v16_8_24/gpu1
```

这样 train -> val -> heldout 以及任务中断后的重新启动可以复用已有 JIT executable/autotune cache（是否命中取决于 JAX cache key 和 shape）。

### 2.5 resume semantic manifest

每个正式 shard JSONL 旁边新增：

```text
*.jsonl.semantics.json
```

锁定：

- cache_dir
- `candidate_selection=balanced`
- `max_candidates_per_scene=24`
- `horizon_steps=80`
- `action_mode=absolute_xy_yaw`
- `metric_set=safety`
- `metric_eval_mode=step`
- `metric_eval_interval=1`
- `done_check_interval=1`
- `state_source=cache`
- `num_shards=2`
- shard index

JIT、profile、compilation cache 不属于 dataset semantics，因此不会阻止 eager 旧结果与经过等价 gate 的 JIT 新结果连续 resume。

旧 v24 JSONL 没有 manifest 时，第一次 resume 会在校验旧 row 已有的 semantic fields 后 adopt；以后误改关键参数将 fail-fast。

## 3. 为什么这轮暂时不做 SDC-only metric

Waymax 标准 `OverlapMetric` 计算 all-object pairwise overlap，`OffroadMetric` 对全部 objects 的 bbox corners 查询 roadgraph；当前 COWP 最终只使用 SDC 值，因此确实还有进一步降复杂度的空间。

但 SDC-only 需要复写 Waymax 几何子逻辑，虽然理论上可以做到严格等价，它比“JIT 同一个 metric.compute”多一层语义风险。本轮先使用最低风险 exact-JIT。如果 exact-JIT 后仍然慢，再做逐 step equality-gated 的 SDC-only 第二阶段。

## 4. 正确执行顺序

### A. 不删除任何已有正式结果

保留：

```text
$COWP_ROOT/waymax_replay_v16_8_24/*.jsonl
$COWP_ROOT/tensor_cache_train_waymax/*.npz
$COWP_ROOT/tensor_cache_val_waymax/*.npz
$COWP_ROOT/tensor_cache_heldout_test_waymax/*.npz
```

### B. 先跑 exact semantic + speed gate

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export MAX_REPLAY_CANDIDATES=24
export REPLAY_HORIZON=80
export WAYMAX_JIT_CHECK_GPUS=0,1
export WAYMAX_JIT_CHECK_SCENES=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

bash CHECK_WAYMAX_JIT_EQUIVALENCE_V16_8_24_CN.sh
```

两张 A30 会并行：GPU0 eager reference，GPU1 optimized。默认完整使用每 scene 24 candidates × 80 steps。

必须看到 `compare.json`：

```json
{
  "semantic_pass": true,
  "jit_active_pass": true,
  "pass": true
}
```

同时检查 `speedup_x`。第一次 optimized candidate 可能包含编译时间；8 scenes 用于摊薄编译启动开销。

若希望更强的 correctness gate，可改：

```bash
export WAYMAX_JIT_CHECK_SCENES=8
```

### C. Gate 通过后，从原 JSONL 断点继续三 split

最推荐直接：

```bash
export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
bash CONTINUE_WAYMAX_OUTCOMES_OPTIMIZED_V16_8_24_CN.sh
```

该脚本会强制：

```text
REPLAY_NUM_SHARDS=2
MAX_REPLAY_CANDIDATES=24
REPLAY_HORIZON=80
WAYMAX_JIT_ENV_STEP=1
WAYMAX_JIT_SAFETY_METRICS=1
WAYMAX_JIT_ENV_RESET=0
```

并调用原 `NEXT_EXECUTION_V16_8_24_CN.sh outcomes`，因此仍按：

```text
training replay -> final attach -> verify
validation replay -> final attach -> verify
heldout replay -> final attach -> verify
final train/val/heldout outcome-support gate
```

完成。

## 5. GPU/CPU 建议

当前结构是一张 GPU 一个 replay worker；两张空闲 A30 最合适的配置仍是：

```text
GPU0 -> shard 0/2
GPU1 -> shard 1/2
```

不要把 `REPLAY_NUM_SHARDS` 改成 4。它既会改变现有 shard mapping，也会让一张 GPU 同时承载多个 JAX process，通常不会提高这一 workload 的吞吐。

CPU 不是瓶颈，因此不建议再增加 CPU worker。scene load/state build 远小于 rollout。

`XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以继续保留。它不是当前 15 s/candidate 的根因。若 exact-JIT 后 GPU memory 使用稳定且希望实验 allocator，可单独 benchmark，但不需要为本轮 full build 改。

## 6. NVIDIA driver / PTX warning

Probe log 显示：driver CUDA 12.8 低于 PTX compiler 12.9.86，XLA 因此关闭 parallel compilation。它主要影响 JIT **首次编译时间**，而不是 dataset semantics。

由于本版本启用了 persistent compilation cache，而且 full build 会复用同一批 compiled shapes，因此这不是阻塞项。若服务器维护窗口允许，更新 NVIDIA driver 或安装 NVIDIA CUDA forward-compat package 可以改善 compile latency；不建议为了这次数据构建临时冒险改驱动。

## 7. Resume 连续性

从旧 eager JSONL 停止，然后用 gate 已证明等价的 JIT execution 继续：

- 同一个 `(scenario_id, candidate_index)` 已成功 row 被 skip；
- 一个尚未 append 完的 scene 会重新计算未持久化部分；
- truncated JSONL tail 由 resume repair 清理；
- incremental NPZ 是 atomic write；
- 最后 scripts/12 仍做 authoritative reconciliation；
- scripts/14 仍验证 Waymax cache。

因此 execution 可以中断/恢复；关键是不要改变 semantic manifest 锁定的参数。
