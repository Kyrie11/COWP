# COWP v16.8.13 数据集支撑审计与修复说明

## 1. 结论

v16.8.12 的 `SMOKE DOES NOT AUTHORIZE STRICT PROBE` 是正确的科学门控，不是 strict 脚本本身异常。上传的 96-scene smoke 中，proposal/causal screen 与 training-supervision audit 均通过；阻断 promotion 的唯一主层级仍是 natural-basis/model-support。

但 v16.8.12 的失败已经非常集中，不能再通过扩大 ego candidate bank 或降低 response/witness 门槛解决：

- 530 个 selected critical vehicles 中，12 个完全没有 natural root，rootless rate = 2.2642%；这 12 个也恰好全部少于 2 个 low-burden roots。
- 12/12 的 dominant rejection 都是 `map`；它们全部属于 `rho=3 (EQUAL_OR_NEGOTIATED)`，全部使用 `logged_geometry_neutral_timing` reference。
- 这 12 个中 10 个拥有 80 个有效 future steps，1 个有 68 steps，只有 1 个只有 19 steps。因此“大部分 rootless 是 WOMD future 太短”不成立。
- natural source 统计为 OBS=1554、NEU=3324、PRIO=0。PRIO=0 是独立于 12 个 rootless 的结构性 model-support failure。
- response bank、same-root transport、witness、candidate NCF 等其余监督均为非退化分布，并且所有 relevant pairs 都具有完整的 32-response slots。

因此 v16.8.12 不适合 full rebuild 的根本原因不是“缺样本量”，而是 **natural-basis 的可审计性、typed-source 覆盖和训练/推理证书有效性还没有形成闭环**。

## 2. v16.8.12 smoke 实际通过了什么

`v16_8_12_smoke_verdict.json` 中：

- `proposal_causal_screen_pass=true`
- `training_supervision_pass=true`
- `model_support_pass=false`
- `natural_rootless_zero=false`
- `natural_lt2_low_burden_zero=false`

Proposal/causal smoke 指标本身是健康的：

- representative AnyNCF = 17/48 = 0.3542，smoke 门槛 >=0.30；
- false-safe floor = 27/48 = 0.5625，门槛 <=0.65；
- PBTR lower bound = 19/43 = 0.4419，门槛 <=0.50；
- hard-scene NCF recovery = 19/48 = 0.3958，门槛 >=0.12；
- pair relevance rate = 0.4260；
- silent blocker = 0；irrelevant blocker = 0；
- audit root 与 transport root 的 affected/conflict/retain identity 检查全部通过。

训练监督也不是退化的：

- candidate NCF: 731/5226 positive = 0.1399；
- candidate false-safe: 1926/2657 = 0.7249；
- relevant pair: 12347/28983 = 0.4260；
- witness | relevant = 8262/12347 = 0.6692；
- pair NCF | relevant = 4085/12347 = 0.3308；
- response safe = 218590/395104 = 0.5532；
- response low-burden = 209475/395104 = 0.5302；
- affected-root recovery = 29004/73359 = 0.3954；
- response slots expected/present = 395104/395104，incomplete relevant pair = 0。

所以此时继续扩大候选轨迹或 response bank，不会解决 strict promotion 的科学缺口。

## 3. 问题一：剩余 rootless 的本质是“lane-route 证据缺失”，不是普通 map-distance 阈值

v16.8.12 已经把旧版 point-cloud map distance 修成连续 point-to-segment distance，因此这次的 12 个 rootless 不是 v16.8.11 的同一个 bug。

新的诊断显示：

- 全部 12 个 rootless 的 reference 都是 `logged_geometry_neutral_timing`；
- 即当前状态无法得到可信的 lane-graph route，构建器只能借 factual geometry；
- 这些失败轨迹到 lane-centerline network 的最优 max-distance 仍约 5.83–20.72 m，部分候选可达到更大距离；
- smoke 全局只有 23/530 个 critical 使用 logged-geometry reference，却贡献了全部 12 个 rootless。

这说明“lane-centerline graph”对少数 WOMD actor 并不是充分的 drivable-route 证据。典型来源包括无结构区域、停车场/driveway、复杂交叉口、lane topology 局部缺失或 centerline 表达与 factual path 不完全重合。

如果把普通 `map_max_distance` 从 5 m 直接放宽到 20 m，会破坏正常 lane-resolved 场景的 map compliance，属于为了过 smoke 改物理语义。因此 v16.8.13 不这么做。

### v16.8.13 修复

新增两层 route evidence：

1. **HD-map lane route**：能从 current state 解析 lane route 时，继续使用原本的 lane compliance；原阈值不放宽。
2. **Empirical route corridor**：只有 lane route 无法可靠解析、且 WOMD factual future 提供充分真实观测时才允许。默认需要 >=60 个 valid future steps 且 valid fraction >=70%；vehicle corridor distance 1.5 m、compliant fraction >=90%、hard max 3 m。它只证明“这条事实路径几何在数据中真实出现过”，不把它标记成 HD-map verified，也不把 future timing 当成 counterfactual causal truth。

factual geometry 必须按原始 valid timestep 的连续片段使用，不能跨 `valid=0` 缺口插值或让 hold/padding 成为路线证据。

## 4. 问题二：PRIO=0 是构造顺序 + 跨 source 去重造成的结构性缺类

v16.8.12 的 model-support audit 中：

- OBS = 1554
- NEU = 3324
- **PRIO = 0**

但自然生成日志确实尝试过：

- `primary_prio_map_route` 1540 次；
- `fallback_map_prio` 485 次；
- `primary_prio_logged_geometry` 69 次；

最终 accepted PRIO 仍为 0。

根因是旧 PRIO branch 没有形成独立 typed family：它使用和 OBS/NEU 高度重合的 route/timing/acceleration family，而 `try_keep()` 在 source retention 前做跨 source 几何去重。canonical progress-preserving trajectory 往往先被 OBS/NEU 占据，随后 PRIO 被标记为 duplicate。

这会直接破坏当前模型的训练与论文消融：模型配置存在 OBS/NEU/PRIO typed roots，但标签集中 PRIO 永远为空，priority-preserving decoder/source prior 无法获得真实 positive target；论文宣称的三源 natural basis 也无法被数据支持。

### v16.8.13 修复

对于 protected relation，natural generation 改为：

**PRIO -> OBS -> NEU**

而不是让 OBS/NEU 先占 canonical root。PRIO 的 canonical root 定义为 progress-preserving / priority-preserving reference，再由温和正向 commitment variants 扩充，且每个 accepted PRIO 都必须显式通过 `priority_preservation_check`。

默认 PRIO family 使用：

- acceleration: `[0.0, 0.25, 0.75] m/s^2`
- speed offsets: `[0.0, 0.75, 1.5] m/s`
- protected auditable critical 至少需要 1 个 PRIO root；
- 总 natural roots 仍至少 6；
- low-burden roots 仍至少 2。

没有降低 burden、comfort、map、priority gate。

## 5. 为什么不能简单删除这 12 个 critical actors

COWP 的 inference critical universe 是规划时根据当前/历史/map 可见信息选择出来的；WOMD future 只在离线构造监督时可用。

如果因为“future 不够好”就直接把 critical/valid 删除，会产生 train/inference semantic mismatch：训练数据说这个 actor 不需要审计，而推理阶段模型并没有 future 可以做同样删除。

v16.8.13 因此把三个概念明确拆开：

- `critical/valid`：算法在推理时选中的 critical universe，不因 future availability 改变；
- `critical/mechanism_valid`：离线数据是否具备足够证据构造可信 natural/transport/witness 机制监督；
- `candidates/certificate_valid`：candidate-level NCF/false-safe certificate 是否在完整 selected-critical universe 上可定义；还要求相关 critical 对模型 tensor input 可见。

一个 selected critical 如果没有足够 lane/factual route 证据，会保留在 `critical/valid`，但 `mechanism_valid=0`。它不会被伪造成“非 coercive”，也不会进入 natural/transport/witness loss；同时它会计入 auditability coverage loss。

Auditability 采用两阶段确认：critical selection 只做便宜的 current-lane/future-support 预检；natural builder 再使用实际 lane-route builder 最终确认。如果 current state 虽能投影到 lane、但该 lane 已到末端/退化而无法形成可用 continuation，同时 factual future 又不足，`mechanism_valid` 会在生成机制标签前降为 false。这避免“投影到 lane = 一定有 8 秒可审计路线”的错误假设。

v16.8.13 promotion 默认要求：

- `mechanism_unauditable_rate <= 1%`
- `certificate-complete scene rate >= 98%`
- 对所有 mechanism-valid critical：rootless=0；<2 low-burden roots=0；
- protected auditable critical：至少 1 个 PRIO root 且 priority preserved。

这使 gate 不能通过“静默删难例”获得 PASS。

## 6. Tensor cache 还存在一个 full rebuild 前必须封闭的一致性风险

Scenario proto 中选中的 critical track，可能因为 tf.Example/model agent-row 上限、row mapping 或 object-ID 对齐问题，在真正的 tensor input 中不可见。

旧路径为了防 gather 越界可能把该 critical mask 掉，但 candidate NCF/false-safe label 仍来自 proto label engine。这样模型会被要求学习一个由“它看不见的 agent”决定的 certificate target。

v16.8.13 在 `COWPNpzDataset` 中加入 object-ID 对齐和 `critical/input_visible`：

- selected critical 不可见时，模型 gather mask 安全失效；
- 对应 `mechanism_valid` 在模型监督路径失效；
- candidate `certificate_valid` 全部失效，而不是把未知 certificate 当负例；
- proto label-space 的原始 selected/mechanism state保留为诊断字段，用于区分“离线不可审计”和“tensor 不可见”。

`65_audit_model_support.py` 现在既能审 labels，也能审 tensor cache；full-core 会对 train/val labels 与 train/val tensor caches 都执行 model-support/visibility gate。

## 7. 对训练、测试和论文论证的影响

### 训练

- natural/source-restricted loss 只使用 `mechanism_valid` roots；
- response/witness/root-transport loss 只使用 mechanism-valid critical pairs；
- candidate NCF/false-safe、budget/consistency、planner non-coercive ranking 使用 `certificate_valid`；
- 模型仍可编码 selected/input-visible critical，即使该样本没有机制监督，避免改变 inference critical definition。

### 测试 / Waymax

`certificate_valid=0` 不意味着 candidate 物理无效。因此它仍可参与 Waymax collision/off-road/progress replay。

但 NCF、false-safe、witness、OPR/FSR/PBTR、label-space precision/recall 等机制指标只在 certificate-valid 样本上统计；mechanism-invalid critical 不进入机制 pair 指标。输出同时报告 certificate-label coverage。

这避免把“counterfactual label unknown”错误统计为“non-NCF/negative”，从而虚增/稀释论文指标。

### 论文论证边界

通过 v16.8.13 smoke/strict/train-pilot/full-core 能支持当前代码的 natural basis、root transport、witness、BCOT/NCF planner、logged-Waymax mechanism evaluation 的训练与内部论证。但论文中更强的真实 counterfactual causal burden claim 仍应按论文既定 evidence protocol 使用独立 reactive-agent protocol + held-out human-audited false-safe stress set + multi-seed paired evaluation；普通 logged replay 不能替代该层证据。

## 8. 速度分析

v16.8.12 smoke fresh profile 平均：

- label engine: 181.16 s/scene
- safe responses: 101.23 s/scene
- witness: 45.39 s/scene
- critical selection: 17.15 s/scene
- causal audit: 11.38 s/scene
- natural generation: 3.12 s/scene
- pair-neutral: 2.03 s/scene
- candidate generation: 0.87 s/scene

safe responses + witness 约占 label-engine 80%。因此为了速度减少 natural roots、32 response slots 或 candidate semantics 是错误优化：既伤模型支撑，又几乎没有攻击主要瓶颈。

v16.8.13 保留现有 exact causal-audit/result reuse，不缩减 candidate/root/response/witness 定义；empirical corridor 只在极少数 lane-unresolved actor 上启用。真正的 worker 配置应在 v16.8.13 train-pilot PASS 后比较 24/32（必要时 40）workers 的 scenes/hour、p90、RAM 和 I/O wait 再确定。

## 9. Promotion 顺序

不要复用 v16.8.12 labels/cache，因为 label、supervision、certificate 与 fingerprint 都变化。

顺序必须为：

`preflight -> fresh smoke -> fastpath A/B -> validation strict -> train split pilot -> full-core -> Waymax outcomes`

其中：

- smoke 只有 `recommend_strict_probe=true` 才允许 strict；
- strict 只有明确授权才允许 train-pilot；
- strict + train-pilot 必须都 PASS 且 fingerprint 与当前代码一致，才允许 full-core；
- full-core 会在 labels 和 tensor-cache 两层审 certificate/model visibility；
- core gates 全部通过后才开始 Waymax GPU replay。

如果 v16.8.13 smoke 仍失败，不要上传 NPZ，只需提供 smoke verdict、natural support、model support、training supervision、base screen 和 fresh profile；v16.8.13 的诊断字段已经可以继续定位到 auditability reason、map/priority/burden rejection、PRIO coverage、empirical corridor 使用量与 input visibility。

## 10. 本地代码验证

在不具备用户 WOMD 原始 shards 的当前环境中完成的是代码级回归，而不是声称真实 v16.8.13 smoke 已 PASS：

- repository regression: **201 passed, 5 skipped**；
- `python -m compileall -q cowp tests`: PASS；
- v16.8.13 smoke/strict/train-pilot/master/full-core launchers `bash -n`: PASS；
- 新增 regression 覆盖：lane-unresolved empirical corridor、短 future mechanism invalid、protected PRIO retention、candidate certificate mask、tensor input-invisible critical、Waymax certificate-aware replay/metrics。

必须在真实 WOMD 数据机上重新 fresh-build v16.8.13 smoke 后，才能判断本轮数据性质是否真正通过。
