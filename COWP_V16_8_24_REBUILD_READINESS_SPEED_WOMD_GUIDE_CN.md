# COWP v16.8.24：Full-Rebuild 就绪性、性能与 WOMD 1.3.1 使用说明

## 结论

当前数据机制已经可以进入 compact full rebuild。v16.8.23 的 64-scene fast-path benchmark 已完成 64/64 场景的 label semantic equivalence，且无非预期 tensor mismatch；缺失 `benchmark_report.json` 的直接原因只是 benchmark wrapper 调用了不存在的 `cowp.scripts.44_summarize_label_build_profile`。仓库实际存在并应调用的是 `49_summarize_label_build_profile.py`。

用同一批 64 scene 对齐历史 v16.8.21 train-pilot profile 后：

- old mean: 281.038 s/scene
- v23 fast mean: 128.639 s/scene
- matched worker-time speedup: 2.1847x
- safe-response: 2.4355x
- witness: 4.4296x
- semantic equivalence: PASS

因此无需重跑 v23 的 64-scene labels 来证明 fast-path 正确性；v16.8.24 支持 `REPORT_ONLY=1` 从已经落盘的 profile + semantic-equivalence 结果直接恢复 benchmark report。

## v23 执行链错误审计

全仓库静态扫描发现 v23 唯一缺失的 `cowp.scripts.*` Python module 是 `44_summarize_label_build_profile`，出现在 benchmark 和 full builder。除此之外，历史 v13–v22 wrapper 中存在若干早期 shell/config 缺失，但它们不是 v24 的支持入口，不应再调用，也不应该伪造旧文件来“补齐历史”。

v24 新增 `77_audit_active_execution_chain.py`，只审计真正支持的 4 个入口：

1. `NEXT_EXECUTION_V16_8_24_CN.sh`
2. `BENCHMARK_V16_8_24_FASTPATHS_CN.sh`
3. `PREPARE_COWP_V16_8_24_FAST_DATA_CN.sh`
4. `ATTACH_WAYMAX_OUTCOMES_V16_8_24_CN.sh`

它同时检查：shell 文件、shell syntax、Python module 文件、Python AST syntax、config 文件和本地 shell 引用。当前自检 PASS。

## v23 另一个很大的性能浪费：Scenario location index

上传 benchmark 的文件时间戳显示：`benchmark_scene_ids.txt` 约 03:15，training Scenario index 约 04:44，label profile/semantic equivalence 约 04:56。也就是说一次 64-scene benchmark 中，大约 1.5 小时耗在扫描完整 WOMD training 来建立 location index，而真正 64-scene fast labels 只占后面的十几分钟。

v23 又把 index 放在 `BENCH_ROOT` 下，因此换一个新的 full-root 会再次扫描整个 training split。

v24 默认将 training/validation Scenario index 放到：

```text
$WOMD_ROOT/.cowp_v131_indices/
```

benchmark 与 full build 共用同一份 index。你现有 v23 benchmark 的 training index 可以直接复制到这个目录后复用，不必重新扫描。

## Full builder 的健壮性修复

v24 还做了以下修复：

- 历史 promoted COWP cache 不再是 full rebuild 的硬依赖。若其 scene IDs 足够，则优先沿用以保持与已经通过 pilot 的场景分布连续性；不足时，自动从 WOMD Scenario index 做确定性 hash sampling。
- `TRAIN_LIMIT/VAL_LIMIT/TEST_LIMIT` 是实际 hard cap，不再因为 reuse old scene set 而失效。
- final full-core verdict 现在把 `causal_audit_{train,val,heldout_test}.integrity` 全部作为 hard gate；v23/v22 虽然运行 causal audit，但最终 composite verdict 没有真正消费这些 integrity flags。
- Waymax outcomes 现在覆盖 train / internal-val / heldout-test 三个 split。历史 `ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh` 只处理 train/val，这对于最终 held-out test 不完整。

## 服务器 worker 建议

服务器为 2 sockets × 24 physical cores，2 threads/core，共 48 physical / 96 logical CPUs。NUMA node 0 为 `0-23,48-71`，node 1 为 `24-47,72-95`。因此 CPU 0–47 正好覆盖两个 socket 的 48 个不同 physical cores 的第一 SMT thread。

推荐 label build：

```bash
export COWP_CPUSET=0-47
export LABEL_WORKERS_TRAIN=40
export LABEL_WORKERS_VAL=40
export LABEL_WORKERS_TEST=40
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
```

不要直接用 80/96 label workers。当前 workload 是 scene-level multiprocessing + 大量小型 NumPy/geometry kernel，SMT 带来的收益通常低于 memory/cache/NUMA contention。40 workers 留出 8 physical cores，同时 OS 还可以在 sibling hardware threads 上调度其它轻任务。

Tensor cache 是 TFRecord decode / NPZ merge / I/O 更重的阶段，推荐：

```bash
export CACHE_WORKERS=12
```

如果 full build 时 `free -h` 显示 RAM 紧张或出现 swap，label workers 降到 32；如果没有 swap、CPU utilization 稳定且 iowait 很低，40 是首选。没有必要为了追求数字继续上 44/48，除非一个 32-scene worker benchmark 明确证明更快。

## GPU

2 × A30 (24 GB) 对当前 Scenario→COWP labels 和 TFExample→tensor-cache 没有直接收益，因为这些路径是 CPU NumPy/TensorFlow I/O，v24 继续 `CUDA_VISIBLE_DEVICES=-1`。

GPU 应用在 full-core PASS 之后的 Waymax/JAX candidate replay：

```bash
export WAYMAX_GPUS=0,1
export REPLAY_NUM_SHARDS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

每块 A30 一个 replay shard，避免在单 GPU 上堆多个竞争进程。

## 是否继续做算法级速度优化

不建议此时继续修改 K/A/M、natural roots、response primitives、collision/RSS envelope 或 witness 定义。v23 fast path 已经把两个主要重复计算热点（safe responses / witness）大幅降低，并通过真实旧 NPZ 的 semantic equivalence。继续靠减少 proposal/root/response 数量换速度会直接改变六层机制的数据性质，收益不值得风险。

下一步只保留低风险工程优化：persistent index、物理核 worker 数、单线程 BLAS、支持 chain preflight、完整三 split Waymax outcomes。等 full build profile 再看是否出现新的显著热点；没有 evidence 就不继续“硬优化”。

## 推荐数据规模

时间敏感、但需要足够的机制监督与评测稳定性时，推荐：

- train: 5,000 scenes
- internal validation: 1,000 scenes
- held-out test: 1,200 scenes

总计 7,200 scenes。此前 1,200-scene train-pilot 已经给出大量 protected ranking pairs、same-root viability/recovery switch 和 OPR switch；5k train 的主要作用是增加 topology/interaction diversity，而不是修复监督稀缺。如果后续 learning curve 明显仍 data-limited，再把 train 单独扩到 6k–10k，val/test 不需要重建。

以当前 64-scene matched worker mean 128.64 s 粗略估算，7,200 scenes / 40 workers 的理想 CPU worker-time 下界约 6.4 小时；考虑 straggler、TensorFlow cache、validation index、审计等，实际应按一次 overnight job 规划，而不是按理想下界承诺完成时间。

## WOMD 目录使用合同

当前 COWP primary pipeline 只需要标准 9-second：

```text
uncompressed/scenario/training
uncompressed/scenario/validation
uncompressed/tf_example/training
uncompressed/tf_example/validation
```

Scenario 是 label-authoritative source：tracks、vector map、dynamic map states/traffic controls、SDC index 等用于 causal scene construction、natural roots、candidate/root viability、recovery/witness pseudo-label。tf.Example 是 model/Waymax tensor source：128-object state tensors、sampled roadgraph、traffic lights、scenario id，以及 v1.3.1 `path_samples/*` SDC paths。

官方 WOMD blind testing 的 future GT 隐藏，不能按与 train/validation 相同的定义构造依赖 8-second factual future 的 Natural/transport/witness labels。因此当前论文 test 是 **WOMD validation 内 scenario-id disjoint 的 held-out test**，必须这样报告，不能称为 Waymo official test score。

### 可以删除（当前 COWP 不需要）

如果你没有准备参加/提交官方 blind challenge：

```text
uncompressed/scenario/testing
uncompressed/tf_example/testing
uncompressed/scenario/testing_interactive
uncompressed/tf_example/testing_interactive
```

当前 COWP full build、六层 mechanism audit、Waymax outcome cache 都不读取这些目录。

`uncompressed/scenario/training_20s` 也不被当前 91-step COWP pipeline 引用；如果没有计划做 20-second scene-generation/long-context 扩展，可以删除。当前本地 release 也没有与之对应的 `tf_example/training_20s`，v24 明确不会把它混进 9-second primary pipeline。

### 建议保留但不是 primary 必需

```text
uncompressed/scenario/validation_interactive
uncompressed/tf_example/validation_interactive
```

论文主 full train/val/test 不需要它们，但 COWP 的题目本身是 interaction-heavy false-safe planning，因此它们很适合作为**单独报告的 secondary interaction stress set**。不要默认把它当成与 standard validation 独立的新 test population；如要发表相关结果，先做 scenario-id overlap audit。

如果磁盘非常紧，也可以删除这两个 interactive validation 目录，当前 active pipeline 仍可完整运行；只是会失去一个很有价值的后续 stress-evaluation 资源。
