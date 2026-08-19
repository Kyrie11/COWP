# COWP v16.8.23：Full Rebuild 可行性与构建加速分析

## 结论
上传的 v16.8.22 strict re-audit 与 train-pilot re-audit 已同时授权 full rebuild；六层机制 contrast audit 也通过。因此不存在已知的数据性质 blocker。下一步风险主要来自计算成本，而不是数据语义不足。

但不建议直接执行原 v16.8.22 full-core。原因有二：
1. v16.8.22 在 `REUSE_OLD_SCENE_SET=1` 时会把历史 cache 的全部 NPZ ID 写入 allowlist，并且该分支不传 `--limit`，导致 `TRAIN_LIMIT`/`VAL_LIMIT` 被忽略。
2. v16.8.22 profile 显示每 scene 的主要成本在 safe-response 与 witness 内循环；直接按旧规模构建会非常昂贵。

v16.8.23 因此采用“64-scene exact semantic-equivalence benchmark -> compact full build -> split-wise six-layer audit”的执行顺序。

## 当前数据是否足以进入 full rebuild
### Strict
- pipeline complete: true
- recommend_full_rebuild: true
- failed_checks: []

### Train-pilot
- 1200 scenes
- pipeline complete: true
- recommend_full_rebuild: true
- failed_checks: []
- protected eligible scenes: 994
- protected NCF scenes: 561
- P(NCF | protected eligible): 0.5644
- best-case PBTR lower bound: 0.4356
- global auxiliary NCF candidate positives: 7219
- protected NCF candidate positives: 12413
- protected false-safe candidate positives: 16543

### Six-layer within-scene/root contrast
- protected rankable scenes: 287; rank pairs: 56,734
- candidate-induced viability switch: 1,163 scenes / 45,106 roots
- same-root recovery switch: 1,137 scenes / 26,198 roots
- option-mass switch: 803 scenes / 1,475 agents
- partial OPR pairs: 39,361
All checks pass. These are directly relevant to Layers 2-5 and are stronger readiness evidence than population prevalence alone.

## v16.8.22 性能瓶颈
1200-scene label profile:
- total scene mean: 291.74 s; p90: 601.49 s; p99: 1080.06 s
- label engine mean: 288.94 s
- safe responses: 114.23 s mean
- witness: 107.05 s mean
- audit relevance: 25.76 s mean
- critical agents: 19.60 s mean
- candidates: 16.18 s mean
- TFRecord proto parse: 0.010 s mean
- NPZ write: 0.056 s mean

因此主要是 CPU geometry / root-response combinatorics，不是 WOMD TFRecord I/O 或 NPZ 写盘。

## v16.8.23 等价 fast paths
### 1. Boolean unsafe predicate
Inner-loop 只需要安全/不安全时，不再生成完整 collision/near-miss/TTC/RSS mask 和诊断量；保留与完整 predicate 相同的逻辑短路。

### 2. Safe budget precompute + exact early stop
对确认安全的 response，physical/progress/norm burden 可预计算；candidate-dependent risk 只在 unsafe case 需要。安全候选按静态 burden 排序后，达到需要的 Top-B safe 数量即可停止；若 safe 数不足，则回退计算完整 unsafe burden，保持旧 Top-B 语义。

### 3. Same-root recovery `min_only`
Witness 只消费最小安全 recovery burden 和 low-burden flag，因此可预计算每个 root-profile 的安全情形 burden，按 burden 从小到大测试，找到第一个安全 response 即得到精确最小值。

### 4. Conflict TTA cache
候选 ego 到 conflict regions 的 TTA、natural root 到 conflict regions 的 TTA 在 scene 内缓存，positive witness 不再为每个 pair 重复扫描并计算最多 64 个 region 的 TTA。

### 5. 防止 worker 线程过订阅
ProcessPool 外层并行时固定 OMP/OpenBLAS/MKL/NumExpr 单 worker 内线程为 1。实际 worker 数量由 benchmark 决定，而不是盲目设置最大值。

## 为什么必须先 benchmark
优化修改了实现代码，因此 label semantic fingerprint 会变化。即使设计为数学等价，也必须用已有真实 V22 NPZ 做逐字段 exact-equivalence 检查，而不是只依赖单元测试。

`BENCHMARK_V16_8_23_FASTPATHS_CN.sh` 默认抽 64 个已有 pilot scene：
1. 用 fast path 从 WOMD Scenario 重新生成 labels；
2. 与 reference NPZ 做 exact semantic equivalence；
3. 汇总 fast profile；
4. 若存在旧 profile，则计算 matched worker-time speedup。

full-core 默认要求 exact equivalence PASS；若 speedup 可测，默认还要求 >=1.25 后才推荐 full build。

## 推荐数据规模
- train: 6000 scenes
- internal validation: 1200 scenes
- heldout test: 1500 scenes

这是“compact research full dataset”，不是要覆盖整个 WOMD。现有 1200-scene pilot 已有大量 pair/root/contrast supervision；6k train 用于扩大 scene diversity 和防止过拟合，而 val/test 保持足够大以稳定评估六层机制、planner 与 Waymax outcomes。如果训练后仍表现为明显 data-limited，可只把 train 扩到约 10k，不需要重建 val/test。

## Test split 定义
COWP transport/witness pseudo-label 需要 observed future。官方 WOMD test 不提供 future GT，因此不能构造与 train/val 同语义的 COWP labels。v16.8.23 将官方 WOMD validation 按 scenario ID 确定性拆为 internal val + heldout test，两者不重叠；最终论文中必须称为 held-out WOMD-validation test，而不是 official WOMD test benchmark。

## Full build 后仍必须通过的 gate
每个 train/val/heldout-test split 独立执行：
- training supervision audit
- model support audit
- six-layer mechanism contrast audit
- causal audit
- Waymax-ready + `sdc_paths` + all-labels-matched cache verification
- scenario-ID disjointness
任何一项失败，`full_core_support_verdict_v16_8_23.json` 都会阻止训练/论文证据使用。
