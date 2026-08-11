# COWP v16.8.11 数据集支撑审计与重构说明

日期：2026-08-11

## 1. 结论先行

上传的 v16.8.10 strict probe 已经证明 **proposal bank 不是当前 full rebuild 的主要阻塞项**。400-hard + 800-random 的代表集上：

- AnyValid = 1.0000；
- AnyNCF = 0.4675（门槛 >= 0.40）；
- selected false-safe lower bound = 0.4825（门槛 <= 0.55）；
- PBTR lower bound = 0.32715（门槛 <= 0.45）；
- hard-scene NCF recovery = 0.3025（门槛 >= 0.20）。

真正阻塞 full rebuild 的是 `model_support` 中的 **natural-basis per-critical support**：

- 1,089 个 vehicle critical-agent 没有 natural root；
- 1,090 个没有任何 low-burden natural root；
- 1,147 个少于 2 个 low-burden natural roots；
- 仅 51 个是“有 natural root 但少于 2 个 total roots”，说明主要问题不是总体 root 数量，而是大量 critical-agent 直接 rootless / low-burden-rootless。

与此同时 response/witness/transport/candidate 的监督并不退化：response 32-slot coverage 完整，safe/unsafe、low/high burden、pair relevance、witness、NCF、root recovery 等均有正负类。因此继续扩大 ego proposal bank 不会直接修复当前 blocker。

v16.8.11 的核心策略是：**不放宽任何 NCF、burden、map、priority、response 或 proposal 门槛，而是修复 natural-root generator 与自身过滤器/中和语义之间的不一致，并增加 train-split pilot 硬门。**

## 2. COWP 当前代码真正需要的数据性质

当前代码已经不只是论文最初的“候选 + witness”描述，而是一个可训练的同根反事实 transport/planner 管线。数据必须同时满足以下层次。

### 2.1 场景/事实层

1. 历史输入与当前状态完整；future 只作为离线标签、自然基、Waymax replay / closed-loop outcome 的监督来源，不能泄漏进模型历史 encoder。
2. priority/right-of-way、lane topology、traffic controls 必须来自完整 Scenario proto 语义。
3. critical-agent universe 必须 candidate-independent，否则添加 proposal 会改变旧 candidate 的证书 universe，破坏 source ablation 和 proposal monotonicity。
4. mainline certificate 当前为 vehicle-only； conventional collision 仍检查所有有效 actor。

### 2.2 Natural basis 层

对每个有效 critical agent，至少需要：

- >= 1 natural root 是数学定义的最低存在性条件；
- active config 继续要求 **至少 6 个 total natural roots**；
- **>= 2 low-burden roots** 是当前 model-support / OPR / transport 的最低可学习支撑条件；
- root weight 有效归一化，source identity 稳定；
- root 必须通过同一套 map、dynamic/comfort、priority、burden、contamination 过滤；
- OBS 只能来自真实有效 future states，不能把 `valid=0` 的 hold padding 当测量；
- future 不足时不能简单删除 critical agent，而要用 map/topology 和 current-state 动力学产生可审计的 pseudo-root；
- neutral intervention 必须以“对该 critical pair 去除 ego pressure”为目标，而不是一个场景级统一 ego trajectory。

### 2.3 Candidate / causal relevance 层

需要同时有：

- valid / invalid physical proposals；
- conventional safe / unsafe；
- NCF / coercive；
- relevant / irrelevant critical pairs；
- false-safe candidate（conventional safe but non-NCF）；
- protected-priority feasible / infeasible。

严格 probe 的现有 proposal threshold 应保持不变，避免用改门槛代替数据修复。

### 2.4 RootTransport / response / witness 层

对 causally relevant pair 必须有：

- 固定 32-slot response occupancy；
- root-indexed response assignment；
- same-root recovery positive / negative；
- safe / unsafe response；
- low / high burden；
- affected/conflict/retain 根状态；
- minimum safe burden、recovery score、OPR、tail burden；
- relevant non-NCF pair 必须可解释为 witness，不能出现 silent blocker。

当前 strict probe 这些项已经具备非退化支撑，说明它们不是 v16.8.10 full rebuild 的首要结构性失败。

### 2.5 Planner / closed-loop 层

当前 mainline planner loss 含非零 closed-loop 权重，因此 full labels + tensor cache 通过后仍不能直接开始 mainline planner training。必须再附加 Waymax outcome labels，并验证：

- collision/offroad/route/kinematic 等 outcome 可读；
- safe 与 unsafe candidate 均有支持；
- mixed-scene / ranking signal 不退化；
- train/val scenario IDs 不泄漏。

## 3. v16.8.10 smoke / strict 为什么失败

### 3.1 全局 ego-neutral 与 pair-specific neutralization 冲突

原 label engine 从 ego candidate bank 选一条全局 `ego_neutral`，然后所有 critical agent 共用。

这在定义上不稳：

- 为 crossing / merge actor 刹车让行，可能反而给 ego 后方 vehicle 制造 RSS/TTC/进度压力；
- 对 rear vehicle 保持速度可能又继续压迫 crossing actor；
- natural filtering 的 burden 又是相对这个 neutral ego 计算，因此错误 neutral 会把本来合理的 natural root 判成高 burden。

v16.8.11 改为每个 critical actor 从固定、proposal-bank-independent neutral family 中选自己的 pressure-removing ego intervention。

### 3.2 Generator / filter 自相矛盾：瞬时 constant acceleration 被 jerk comfort filter 删除

原 NEU/PRIO/fallback 使用数学 constant-acceleration primitive。离散实现中从当前加速度到目标加速度是一步跳变；而 `priority_preserved()` 又计算 finite-difference jerk 并施加 comfort 限制。

例如 0 -> 0.5 m/s² 在 0.1 s 采样下就形成约 5 m/s³ 的离散 jerk。于是本来很温和的 ±0.5 m/s² natural root 也可能因“自己的生成器不满足自己的 filter”被删除。

AGENT_PRIORITY 场景尤其容易只剩 `a=0`，再叠加 map/burden/contamination 后就形成 singleton 或 rootless。

v16.8.11 不放宽 jerk threshold，而是把 natural/neutral timing 改为 jerk-bounded acceleration ramp。

### 3.3 不完整 future 被 padding 后误当完整 observational/reference evidence

`future_states_to_traj7()` 会把 invalid/missing future 延续为 hold 状态以满足固定 tensor shape。这对张量化是合理的，但不能把这些 hold rows 当真实观测 natural behavior。

旧 natural path 在部分流程里只看到固定 H=80 轨迹，导致短 future 可能：

- 被当 OBS 根；
- 被用作完整 route geometry；
- 产生不自然的 progress/burden；
- 在曲线道路上与 map filter 联合作用造成 root collapse。

v16.8.11 在 padding 前直接读取 raw `valid` mask：默认 >=60/80 且 >=70% 才允许 OBS；reference 使用也有独立有效率门槛。

### 3.4 旧 fallback 只补 total roots，不保证 low-burden roots

COWP 的 NCF/OPR 依赖 low-burden natural option mass。仅仅拥有多个“valid 但高 burden”的 roots 并不能给 transport / NCF 提供所需监督。

v16.8.11 保留 active config 的 `min_natural_alternatives=6`，并新增 `min_low_burden_alternatives=2`，fallback 直到两者都满足或者所有候选均按原过滤器失败。

### 3.5 曲线道路的 straight fallback 结构性吃亏

当 OBS 不足时，straight NEU/PRIO 在弯道上很容易被 map filter 删除。critical selection 本身并不要求 future 80 步完整，所以单靠 logged future 不能保证可构造 natural basis。

v16.8.11 新增 lane-graph centreline continuation：从 actor 当前 lane 投影出发，沿 lane exit topology 构造多个合法 route continuation，再用 jerk-bounded timing 生成 roots。它是短/缺 future 和曲线道路的第一 fallback。

## 4. v16.8.11 代码修改

主要文件：

- `cowp/label/natural_alternatives.py`
  - pair-specific ego neutral；
  - jerk-bounded natural timing；
  - raw future validity gate；
  - lane-graph route fallback；
  - active 6-total-root + >=2 low-burden root support contract；
  - per-critical rejection diagnostics。
- `cowp/label/label_engine.py`
  - 使用 pair-specific neutral bank；
  - 把 neutral/natural diagnostics 写入 label-build profile。
- `cowp/label/audit_relevance.py`
  - 缓存 audit identity-root 的 exact collision/burden 结果。
- `cowp/label/safe_responses.py`
  - identity response 复用 audit exact result。
- `cowp/label/witness.py`
  - root recovery 的 identity evaluation 复用 audit exact result。
- `configs/label_cowp_v16_8.yaml`
  - 新增 raw future / low-burden support / pair-neutral / map-route / jerk-ramp 参数。
- `cowp/scripts/68_summarize_natural_support_diagnostics.py`
  - 新的 rootless 原因聚合器。
- `NEXT_TRAIN_PILOT_V16_8_11_CN.sh`
  - full rebuild 前新增 training split 400-hard + 800-random 支撑 pilot。
- `NEXT_EXECUTION_V16_8_11_CN.sh`
  - promotion chain 变为 `preflight -> smoke -> fastpath-ab -> strict -> train-pilot -> full-core -> outcomes`。
- `PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh`
  - full train/val label build 后先输出 natural support diagnostic 再做硬门。

## 5. 为什么这些修改不是“为了过 gate 篡改性质”

没有做以下任何操作：

- 没降低 AnyNCF / PBTR / false-safe / hard-recovery threshold；
- 没降低 burden threshold；
- 没放宽 map compliance；
- 没放宽 priority/comfort jerk filter；
- 没减少 critical-agent 数量来躲 rootless；
- 没减少 32 response slots；
- 没减少 root recovery search；
- 没把 high-burden root 标成 low-burden；
- 没把 invalid future 标成 valid OBS；
- 没删除 hard scenes。

修复的是“生成器能否产生满足既有定义的证据”和“同一个 exact physical quantity 是否被重复算多次”。

## 6. Full rebuild 性能分析

v16.8.10 strict profile（1200 场景）的平均：

- total ~239.11 s/scene；
- safe responses ~109.15 s/scene；
- witness ~91.61 s/scene；
- causal relevance ~19.61 s/scene；
- critical selection ~16.24 s/scene；
- natural ~1.06 s/scene；
- candidate ~0.74 s/scene。

safe response + witness 约占 label-engine 的 84%。因此：

1. **不要**为了速度删 candidate/natural roots；收益小，性质损失大。
2. 本版本先做 exact identity-root memoization：audit 已算过的相同 root/ego pair 直接复用到 response/witness。
3. full rebuild 使用 Scenario-ID allowlist 和 code fingerprint，可安全 resume，不必重做已经由同一 fingerprint 完成的场景。
4. label 输出继续 `--no-compress`，减少 CPU 压缩热路径；需要归档时可在 promotion 后再压缩。
5. Waymax replay 后置，并支持 deterministic multi-GPU sharding，避免 label gate 失败前浪费 GPU。
6. worker 数必须在你的主机上测吞吐。旧 strict 平均耗时按理想线性并行粗算，22k train@32 + 5k val@24 约 59.5 worker-normalized wall-hours；这只是旧 workload 的理想估计，不是新代码承诺。v16.8.11 修复 rootless 后会增加一部分真实 response/witness 工作，同时 exact reuse 会减少重复工作，净值必须以新 smoke/strict profile 为准。
7. 建议固定 96~192 个场景 A/B 测 `LABEL_WORKERS=24/32/40`，比较 scenes/hour、p90、RSS 内存、load average、I/O wait，再决定 full worker 数。不要只看 CPU 核数。

## 7. Full rebuild 之前必须看到什么

### Smoke 必须证明

- rootless critical agents = 0；
- critical agents with <2 low-burden roots = 0；
- pair-neutral unsafe rate 不出现系统性异常；
- response bank、witness、transport、candidate supervision 继续非退化；
- proposal gates 不回退；
- fastpath ON/OFF semantic equivalence PASS。

### Validation strict 必须证明

继续使用 400 hard + 800 random，并要求原有 proposal + causal + model-support 全部门通过。

### 新增 Train pilot 必须证明

training split 上独立 400 hard + 800 random：

- supervision audit PASS；
- model support PASS；
- rootless=0；
- <2 low-burden roots=0；
- no silent/irrelevant blockers；
- AnyValid >= 0.99；
- AnyNCF >= 0.30（仅 train support gate，不替代 publication validation metric）。

只有 validation strict 和 train pilot 同时 PASS 且 fingerprint 匹配，`full-core` 才允许运行。

## 8. WOMD v1.3.1 使用注意事项

1. WOMD motion scenario 是 9 s 窗口：1 s history + 8 s future，10 Hz；代码中的 10 past + current + 80 future 与该协议一致。
2. Scenario proto 的 `ObjectState.valid=false` 明确表示 state invalid/missing。因此任何 natural/reference 逻辑都必须读取 validity mask，不能把填充值解释成观测事实。
3. `tracks_to_predict` 在 training 中只是建议训练对象；COWP critical universe 不应被它限制。`objects_of_interest` 也只是交互组标记，不等于完整的 COWP burden-critical set。
4. tf.Example traffic-light tensor 存在最多 16 状态的历史格式限制，官方维护者说明它可能丢失仍有效的 traffic-light information，而 Scenario proto 保留所有有效 traffic-light data。priority/right-of-way 标签因此必须继续以 Scenario proto 为权威源。
5. v1.3.1 新增 `sdc_paths`，可用于 Waymax route-related support；本版本 preflight 要求其存在。
6. 不同 scenario 的坐标系 origin 可不同，不允许跨 scenario 拼接全局几何。

## 9. 本地验证

- `pytest -q`: **190 passed, 5 skipped**；唯一 warning 来自 PyTorch nested-tensor prototype API。
- `python -m compileall -q cowp tests`: PASS。
- `bash -n NEXT_EXECUTION_V16_8_11_CN.sh NEXT_TRAIN_PILOT_V16_8_11_CN.sh PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh`: PASS。
- 注意：当前环境没有你的原始 WOMD shards，因此无法在这里声称新的 96-scene smoke / 1200-scene strict 已经通过；必须在你的数据机上按下一节执行。

## 10. 下一步

严格按 `NEXT_EXECUTION_V16_8_11_COMMANDS_CN.txt` 执行。最重要的是：**不要直接 full rebuild。** 新 smoke 的 `natural_support_diagnostic.json` 会首次给出真正可操作的 rootless 原因分解；若仍有 rootless，它应明确显示是 map、burden、priority、contamination、future support 还是 pair-neutral 问题，再针对那个具体分支修，而不是继续扩大候选或放宽阈值。
