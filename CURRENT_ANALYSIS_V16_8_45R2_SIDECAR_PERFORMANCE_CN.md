# V16.8.45R2：RCRSO Sidecar 性能与可观测性工程审计

## 结论

`build_sidecar_train_parallel4` 慢和“终端完全没输出”是两个独立工程问题。原 V45R1 launcher 启动 4 个 CPU Python worker，并把每个 worker 的 stdout/stderr 全部重定向到日志，因此用户侧看不到进度；真正的时间热点则在**sidecar hard-label 生成**，不是 RCRSO 网络训练/forward。

这版定义为 **V16.8.45R2 engineering-only**。RCRSO scientific method、compact-5k、root/RCOT/BCOT、V42–V44 hard verifier、Stage-0/lost7/CF48 gate 全部不变。

## 1. Sidecar 调用链与热点

每个 train scene 会经历：NPZ/schema load → candidate/blocker/retained-root enumeration → frozen response teachers → V44 analytic completion → deterministic rich control proposals → frozen hard verifier → RCRSO feature construction → NPZ write。

hard verifier 对每条 response 包含 burden、roadgraph、Waymax inverse kinematics、ego current/shift、environment current/shift 双向 collision compatibility。`environment_cap=24`、80-step horizon、每个 root 多 proposal，使 collision/kinematics 被重复执行大量次数。

合成 micro-profile（只用于定位，不代表服务器端到端）显示：

- Waymax trajectory kinematics 原 80 次 Python edge 调用是明显热点；
- environment/ego `unsafe_between_bool` 是另一个主要热点；
- V45R1 sidecar 在同一 scene/candidate/root 之间反复传新的 `{}` compatibility cache，丢掉 V44R1 已证明安全的语义缓存；
- analytic completion 已经算过 root↔environment unsafe event，feature builder 又重复计算；
- dataset loader 原来用 `wanted=None`，解压了 sidecar 根本不用的 cache tensor。

## 2. 已落地的零科学语义优化

### 2.1 scene-level semantic cache

compatibility/cache 生命周期提升到 scene。缓存仍以真实 response/control geometry identity 命名，ego/candidate-dependent predicate 不会被错误复用。fresh-cache 与 shared-cache verified-set 的 regression 完全一致。

### 2.2 Waymax kinematics batch 化

保留 literal helper 作为 oracle；新 fast helper 一次把 H 个 prior→current transition 传给原 `_waymax_kinematic_transition_np`。它没有改变 threshold、inverse-dynamics 公式或 first-failure 语义。随机轨迹 regression 比较 boolean、failure step、max acceleration/steering diagnostics。

### 2.3 conflict-event reuse

V44 analytic completion 的 `control_reachable_environment_events` cache 已经保存 frozen root 与 environment actor 在 current/shift 下的 exact unsafe-event indices。RCRSO feature builder 现在直接消费这些 event indices，只重新计算 candidate-dependent ego current/shift event。

### 2.4 partial NPZ loading/context reuse

sidecar builder 只加载使用到的 WOMD/COWP keys，并复用 per-scene successor/environment/blocker ordering。减少无关压缩解码和 Python object reconstruction。

### 2.5 sidecar write mode

默认仍 `compressed`。服务器 CPU 写压缩成为瓶颈时，可设置：

```bash
export SIDECAR_SAVE_MODE=uncompressed
```

它不改变读取后的数组语义，只增加磁盘占用以减少压缩 CPU。应先看真实 profile 的 `write` 占比再决定。

## 3. 本地 component benchmark

`V16_8_45R2_SIDECAR_MICROBENCHMARK.json`：

- kinematics literal 1000 calls: 3.8028 s
- kinematics vectorized: 0.08264 s → **46.02×**
- verifier 10 repeated candidates fresh cache: 3.3072 s
- shared scene cache: 0.39446 s → **8.38×**
- feature environment-event recompute: 0.07697 s
- event reuse: 0.01584 s → **4.86×**

这些数字是合成重复热点，不是 5000-scene wall-clock speedup 承诺。真实收益取决于服务器 scene composition、cache hit rate、CPU、内存带宽与 NVMe。

## 4. 进度与性能日志

R2 worker 使用 `PYTHONUNBUFFERED=1`，launcher 通过 `sed -u + tee` 把每个 shard 同时输出到终端和日志。默认每 30 s 输出：

```text
scan / scenes / groups / examples / proposals / verified
elapsed / scene_rate / ETA
timing(load, environment, analytic, verify, features, write)
```

可调：

```bash
export SIDECAR_PROGRESS_EVERY_SECONDS=15
```

所有 shard 结束后还会输出 aggregate timing percentage 和 cache counters。

## 5. 服务器推荐顺序

不要立刻再次跑全部 5000 scenes。先跑：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R2_RCRSO_SIDECAR_PERF_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_45R2_RCRSO_SIDECAR_PERF_CN.sh sidecar_smoke
bash NEXT_RUN_COMMANDS_V16_8_45R2_RCRSO_SIDECAR_PERF_CN.sh profile_sidecar_train8
```

`profile_sidecar_train8` 使用与正式 train 相同的 24 examples/scene、24 Sobol proposal 设置，只限制为 4 shards × 2 processed scenes，目标是获得服务器真实性能分解，不用于 scientific promotion。

若 CPU/RAM/NVMe 足够，优先：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R2_RCRSO_SIDECAR_PERF_CN.sh build_sidecar_train_parallel8
```

若机器 CPU 核数很多，可：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R2_RCRSO_SIDECAR_PERF_CN.sh build_sidecar_train_auto
```

`auto` 根据 CPU count 默认选择 4/8/12 shards。不要盲目提高到几十进程；每个 worker 都有 NumPy/NPZ/geometry 工作，过度并发会变成 RAM/IO contention。用 profile8 的 scene/s 和 timing 再决定。

## 6. 两张 A30 怎么用

### Sidecar build：当前不推荐让 CUDA 成为 authoritative hard verifier

原因不是 GPU 算力不够，而是计算形态和实验可靠性：

1. 单条轨迹很小（约 80 steps），大量分支、cache lookup、RSS/OBB/TTC 和 early-exit；
2. CPU↔GPU 小批量搬运/launch 可能吃掉矩阵并行收益；
3. hard verifier 决定 certificate membership，CPU NumPy 与 CUDA float reduction 在边界处的微小差异可能改变 hard boolean；
4. R2 已经通过 cache/vectorization 消除了更大的确定性重复工作。

因此 R2 不引入未经服务器 equivalence 证明的 CUDA verifier。

如果 `profile_sidecar_train8` 后仍显示 collision/interaction >70–80%，下一工程版才值得做 GPU batch prototype，但必须采用：GPU 只做 conservative prefilter/order，最终 hard membership 仍由 CPU exact verifier；或者先建立大规模 CPU↔CUDA boolean/first-failure equivalence gate 后再考虑 authoritative GPU。

### RCRSO train / Stage-0：使用 A30

这两步是标准 PyTorch Transformer/batched inference，适合 CUDA。当前 launcher 默认单卡：

```bash
export RCRSO_TRAIN_GPU=0
export RCRSO_STAGE0_GPU=0
```

RCRSO 第一版模型很小（d_model=128、2 layers、batch 64），单张 A30 通常足够。直接上双卡 DDP 可能被成千上万小 NPZ 的 DataLoader/I/O 限制，而且若改变 global batch 还会改变训练语义。建议先单卡 profile；只有 GPU utilization 长期高、epoch 仍耗时明显，且 DataLoader 不是主瓶颈时，再做保持 global batch/seed 合同的 DDP 版本。

两张卡现阶段更适合保留 GPU1 给 Waymax parallel2 / 独立实验，而不是强行让 sidecar CUDA 化。

## 7. 不允许为了速度改变的内容

- 不减少 80-step closed-loop horizon；
- 不降低 `max-examples-per-scene=24` 或 `rich-sobol-proposals=24` 来制造“快版本”作为正式 sidecar；
- 不放宽 burden/root/roadgraph/kinematics/current+shift/environment/CSP；
- 不以 GPU approximate collision 替换 hard verifier；
- 不重构 compact-5k；
- 不改变 RCRSO Stage-0/lost7/CF48 preregistered gate。

## 8. Validation

- focused semantic/integrity: **139/139 PASS**；
- vectorized-vs-literal kinematics randomized equivalence: PASS；
- fresh-vs-shared verifier cache verified-set equality: PASS；
- recomputed-vs-precomputed environment event RCRSO feature equality: PASS；
- Python compile: PASS；
- launcher `bash -n`: PASS。

RCRSO scientific status remains **UNRESOLVED**。R2 只加速/增强可观测性，不能作为 V46 算法证据。
