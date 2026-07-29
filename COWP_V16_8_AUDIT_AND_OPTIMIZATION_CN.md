# COWP v16.8 严格审计、机制失败归因与算法优化报告

## 0. 结论先行

我完整核对了论文 TeX、上一轮“大模型建议”、COWP 源码、根目录算法日志，以及本轮 `cowp_v16_7_mechanism_v9labels_seed2026` 的 learned-offline、calibration 和 transport diagnostic 结果。

本轮 mechanism gate 没有通过，**不能简单归因于模型容量不足，也不能继续靠调高 BCOT budget 解决**。当前证据显示：

1. natural basis、typed natural decoder 和 pair witness 已经具备可用信号；
2. candidate-level BCOT 排序非常强，说明模型已经能分辨“哪类候选更可能 coercive”；
3. 真正断裂发生在论文核心链条中的
   `natural root -> same-root counterfactual transport q -> transported OPR -> candidate certificate -> selector`；
4. 断裂既包含真实算法问题，也包含定义、标签和训练/评估数据流不一致；
5. v16.7 的 root-recovery target 并没有严格实现论文公式，因此旧实验不能用于否定 coercive 核心 idea。

v16.8 的主修复不是再增加一个黑箱分类头，而是提出并实现：

> **Root-Conditioned Counterfactual Transport（RCOT）**

它对每个冲突 natural root 独立构造保持空间/拓扑根身份的纵向时序反事实，并用统一的 `s=(1-c)r+cq` 定义生成、训练和评估 transported OPR。这样，coercion 被定义为“保护对象原有自然选择概率质量是否被迫消失”，而不是“一个有限通用刹车库是否碰巧找到安全响应”。

代码已完成修改并通过：

- `pytest`: **142 passed**；
- Python `compileall`: passed；
- 所有顶层 shell 脚本 `bash -n`: passed。

本环境没有服务器数据、GPU checkpoint 和 Waymax 运行条件，因此我不声称 v16.8 mechanism gate 已经通过。当前交付的价值是：**已经消除会污染算法归因的确认性工程错误，并给出可证伪、直接服务论文核心概念的下一版机制算法。**

---

## 1. 我对论文核心 idea 与 pipeline 的理解

论文要解决的不是传统的 collision-free feasibility，而是：

> ego 候选轨迹即使不碰撞，也可能依赖其他交通参与者承担异常减速、让行、路径放弃或选项坍缩，从而形成 false-safe / coercive plan。

论文的机制链条可概括为：

1. **Natural behavior basis**：在不施加某个 ego 候选交互影响时，为关键 agent 构造多源、稳定、带概率质量的 natural roots；
2. **Conflict relation**：判断 ego candidate 与每个 natural root 是否冲突；
3. **Same-root recovery**：对于冲突 root，判断是否存在一个仍属于该 root、但通过低负担调整后可安全通过的 response；
4. **Option Preservation Ratio (OPR)**：统计保护对象原有 natural option probability mass 中有多少被保留；
5. **Burden tail / BCOT**：结合未恢复冲突质量、负担尾部和 option shortfall，形成 protected-priority candidate certificate；
6. **Selector**：在满足标准运动学/道路/碰撞约束的候选中，拒绝 coercive false-safe plan，并在可行集合内优化进度；
7. **Closed-loop verification**：在真实 Waymax rollout 中验证 collision、offroad、route progression 等标准指标，同时验证 PBTR、protected OPR、BTE-CVaR25、NCF retention 和 progress regret。

论文的创新性不应被表述成“新增一个风险分类器”。真正可形成 CCF-A 论文主线的是：

- 从 ego-only collision-free 扩展到 **interaction-feasible / non-coercive feasible**；
- 用 natural option mass 表达 agent 在交互前的自由度；
- 用 same-root counterfactual transport 区分“合理微调”与“被迫换意图/放弃路径”；
- 用 protected-priority semantics 避免退化成“ego 永远不能让别人不舒服”；
- 将机制证书与闭环 planner 严格分离、可归因验证。

---

## 2. 本轮结果实际说明了什么

### 2.1 当前 held-out 结果

本轮上传结果的关键值为：

| 指标 | 当前值 | 判断 |
|---|---:|---|
| Pair witness AUPRC | 0.72893 | 已有可用 pair-level coercion 信号 |
| Priority BCOT false-safe AUPRC | 0.95611 | candidate 风险排序很强 |
| Global BCOT false-safe AUPRC | 0.96146 | 全局诊断排序同样很强 |
| Priority RootTransport AUPRC | 0.20273 | 核心失败点 |
| Priority NCF precision | 0.72535 | 被接受候选多数确实较干净 |
| Priority NCF recall | 0.19062 | 过度拒绝 |
| Accepted candidate rate | 0.04902 | 过低 |
| Fallback rate | 0.44972 | 过高 |
| Selected PBTR | 0.44331 | 相比 conventional 0.65151 有明显改善 |
| Selected global false-safe | 0.25379 | 相比 conventional 0.59098 有明显改善 |

这组结果有一个重要含义：

> **模型不是完全没有学会 coercion。**

BCOT AUPRC 接近 0.96、priority risk ranking pair accuracy 约 0.902，说明 candidate-level 排序信号非常强。硬证书也确实降低了 PBTR 和 false-safe。问题在于：root-recovery 表达错误和重复 hard veto 将大量本来可能 non-coercive 的候选判为不可接受，导致 precision 高、recall 和 coverage 崩溃。

### 2.2 RootTransport 的异常形态

transport diagnostic 显示：

- train root recovery mean：0.02575；
- validation root recovery mean：0.02388；
- validation p90：0.03895；
- response-root assignment coverage：1.0。

“assignment coverage=1”但 recovery mass 约 2% 是一个强烈警报：

- 不是没有把 response 分配给 root；
- 而是旧标签几乎总在说“冲突 root 没有低负担同根恢复”；
- 这与 candidate-level BCOT 能很好排序形成矛盾；
- 因此应首先检查 q 的定义和生成，而不是继续扩大网络。

### 2.3 Proposal bank 的独立上限

当前：

- `ProposalCoverage/AnyNCFSceneRate = 0.37655`；
- `PriorityCertificate/NCFSceneRetention = 0.83934`；
- `NonCoerciveProgressRegret = 0.46885`。

这说明两个不同问题必须分开：

1. certificate 确实保留了较多已经存在的 NCF scene；
2. 但候选库本身只有约 37.7% 的场景出现过 NCF proposal，且 selector 在有 NCF 候选时仍有较高 progress regret。

因此 proposal bank 是后续闭环 SOTA 的第二瓶颈，但现在不能同时修改，否则无法知道 mechanism 提升来自 q 修复还是候选变多。

---

## 3. mechanism gate 未通过的确认性原因

## 3.1 OPR 实现漏掉冲突根的恢复项 q

论文定义为：

`retained root s = (1-c)r + c q`

其中：

- 不冲突 root 保留其原始 retained 状态；
- 冲突 root 是否保留，取决于 same-root low-burden recovery `q`。

但 v16.7 的两个关键位置都只累加了非冲突 retained root：

- fresh witness label 构建；
- `paper_aligned_supervision_batch()` 训练适配器。

结果是：即使某个冲突 root 已经找到合法、低负担、同根 response，也仍会被 OPR 当作丢失。这直接把 witness、candidate false-safe 和 selector 推向过度保守。

**v16.8 修复：**标签生成、旧缓存 overlay 适配、训练和 learned-offline eval 均统一使用 transported OPR，并显式加入 q。

## 3.2 旧 q 标签不是“可恢复性”，而是“有限通用响应库覆盖率”

旧流程是：

1. 为 agent 生成全局通用 response primitive；
2. 对所有 root 共享同一个有限 response bank；
3. 全局截断到前 R 条；
4. 再按整段轨迹 ADE 把 response 最近匹配到某个 root；
5. 某个 root 没有匹配到低负担安全 response，就令 q=0。

这与论文“transport each conflicting root into its own response family”不一致。

负标签可能由以下因素产生：

- 该 root 在全局 top-R 中没有得到搜索预算；
- constant-heading primitive 无法保持转弯/合流 root 的空间结构；
- response 对错了 root；
- 两个 root 长时域端点接近而发生 source/identity swap；
- response 安全但高负担，本来就不应算 q；
- 该 root 只需要局部 timing shift，但全局 primitive 没有对应参数。

所以旧 RootTransport AUPRC 低，不一定是模型学不会，而可能是 target 本身不稳定、极稀疏且不符合理论定义。

## 3.3 训练与评估使用了不同的证书定义

v16.7 训练阶段会从 transport 字段重建 paper-aligned target，但 learned-offline evaluation 仍可能使用缓存中的旧 `false_safe/NCF/OPR` 标签进行 calibration 和 gate。

这意味着：

- 模型被要求拟合新定义；
- gate 却按旧定义判定；
- threshold sweep 的不可行不能解释为模型失败。

**v16.8 修复：**offline rollout 在计算任何 selection/calibration 指标前，先调用同一个 `paper_aligned_supervision_batch()` 重建 canonical target。

## 3.4 `root_index=-1` 被 clamp 到 root 0

旧缓存的 generic/unknown response root identity 可能是 -1。若直接 `clamp(0, M-1)`，它会被错误映射到 root 0，并制造虚假的 root-0 recovery。

**v16.8 修复：**单独维护 `root_in_range` mask，-1 不参与 scatter；只有显式 identity 或足够置信的 soft affinity 才参与 q supervision。

## 3.5 safe 不等于 low-burden recovery

对缺少 `is_low_burden` 的旧 cache，旧适配逻辑可能把 `is_safe=true` 当作 recovery。这会把“通过急刹、强制停车才安全”的 response 错当成 non-coercive recovery。

**v16.8 修复：**缺失显式标签时，用 agent-adaptive `beta` 和 burden total 重建 low-burden predicate。

## 3.6 planner 会重新破坏 mechanism

原 planner stage 继续更新：

- SetTransport；
- response decoder；
- 相关辅助损失。

因此即使 transport checkpoint 已经过 mechanism gate，最终 planner checkpoint 也可能为了 ranking/imitation 再改变 q 和 certificate。之后闭环提升或退化无法归因。

**v16.8 修复：**主配置在 planner 阶段冻结 transport 与 response decoder，并将 planner-stage transport/response/witness auxiliary scale 设为 0。联合 fine-tuning 只保留为显式 ablation。

## 3.7 重复 severe veto 造成二次保守

旧 selector 同时使用：

- localized protected-pair severe veto；
- aggregate candidate severe veto。

两者高度相关，在 q 标签偏低时会重复拒绝同一候选。

**v16.8 修复：**保留具有因果定位能力的 protected-pair veto；aggregate severe 只作为 soft ranking/audit，hard veto 默认关闭并保留 ablation。

## 3.8 旧 budget sweep 不能证明“无 operating point”

当前使用 symmetric class balancing 后，risk logit 更适合解释为排序分数，不必天然概率校准。只扫到 0.70 不能证明不存在可行 coverage-recall 区间。

**v16.8 修复：**扩展到 0.98，同时仍要求 PBTR、precision、recall、accepted rate、fallback 联合约束，避免简单放宽阈值换 coverage。

---

## 4. v16.8 核心算法：RCOT

## 4.1 定义

对于 agent `i`、ego candidate `k`、natural root `m`：

1. 先判断 root 与 ego candidate 是否冲突 `c_ikm`；
2. 若冲突，则在 root 自身的空间路径上搜索一个有限纵向时序控制族；
3. 控制只允许改变沿 root 的时间参数，例如轻度减速、短暂等待、轻度加速；
4. 不允许改变其空间/拓扑 maneuver identity；
5. 若存在安全且 burden 不超过 `beta_i` 的响应，则得到 q；
6. transported OPR 由每个 root 的 probability mass 和 `s=(1-c)r+cq` 计算。

## 4.2 为什么这比旧方法更符合 coercive 概念

假设 agent 的 natural root 是“保持原车道通过冲突区”：

- 轻微提前或延后通过，仍是 same-root recovery；
- 被迫急刹停车，可能是高 burden，不应算 q；
- 被迫换道或放弃转弯，已改变 root identity，不应算 q；
- 通用 response bank 没采到合适 timing，不代表该 root 不可恢复。

RCOT 将这四种情况明确区分，使“选项是否被剥夺”成为可计算、可监督的机制变量。

## 4.3 具体实现

### Root-preserving residual

`_root_residual_trajectory()` 不再沿每帧局部 tangent 平移点，而是：

- 计算原 root polyline 的 arc length；
- 积分纵向速度 residual；
- 在原 polyline 上进行 time-warp 重采样；
- 超出末端时只沿终端 tangent 合理外推；
- 保留 root 的转弯/合流几何。

### Per-root oracle budget

`root_conditioned_recovery_search()` 对每个冲突 root 独立运行同样的 profile 搜索。核心 q 不再依赖 compact response tensor 的 slot 数量。

### Explicit identity + confidence-aware fallback

fresh data 直接写：

- `response_root_index`；
- `root_affinity`。

旧 response 若没有 identity，则使用 1/3/5/8 秒多时域 soft affinity；低置信 assignment 通过 `root_target_confidence` mask 掉，而不是强制负标签。

### Continuous targets

新增：

- `transport/root_low_safe_score`；
- `transport/root_target_confidence`；
- `transport/root_min_safe_burden`；
- `transport/root_recovery_mass`；
- `transport/transported_opr`。

这些字段让模型既能学是否存在 recovery，也能学 recovery mass/quality，而不是只学极端稀疏二值标签。

---

## 5. 值得保留、继续深化和停止的算法

## 5.1 值得保留

### Typed natural decoder

natural basis/effectiveness 已通过，当前没有证据支持推倒重来。它是 coercive 机制的前提，应冻结作为机制实验的稳定输入。

### OBS capacity

上一轮严格 paired attribution 已显示 OBS 误差改善，且没有对 neutral/priority 造成明显有害退化。应保留，最终在 fresh data 上做至少 3 seeds。

### Mass-aware root envelope

显著降低 identity excess 和 violation mass，符合“稳定 natural root”需求。保留 soft envelope；hard projection 仍只作为 safeguard，不作为已证实贡献。

### Pair witness

AUPRC 0.729，证明 pair-level conflict/burden witness 已具有信息。它应作为局部 causal evidence，不应被替换成全局 candidate 黑箱。

### Protected-priority semantics

只把 AgentPriority 和 EqualOrNegotiated 纳入硬保护是正确方向，避免 COWP 退化为普遍保守。EgoPriority 可留作 global anti-degeneration diagnostic。

### Monotone BCOT aggregator

candidate BCOT AUPRC 0.956 表明“unrecovered conflict mass + burden tail + option shortfall”的单调组合有价值。应保留其可解释性，不引入绕过机制的自由 MLP classifier。

### Fail-fast gate

机制未通过时阻止昂贵 full Waymax 是正确工程策略。

## 5.2 需要继续深化

### RCOT representation

当前版本先使用可解释、有限纵向 timing control 验证机制。若 root AUPRC 显著提升但仍不足，可按以下顺序深化：

1. continuous time-shift / speed-profile regression；
2. root-specific feasible interval prediction；
3. conflict-time-conditioned residual policy；
4. differentiable root-preserving optimal control；
5. uncertainty/conformal interval，仅在基本 target 稳定后加入。

不要立刻上复杂 diffusion response model，否则很难证明提升来自 same-root mechanism 而非容量。

### Selector calibration

当前 BCOT 排序强、progress regret 高。机制通过后，应把 selector 从固定 hard threshold 深化为：

- protected PBTR-constrained frontier；
- scene-adaptive risk budget；
- coverage-progress Pareto calibration；
- calibrated abstention/fallback cost。

但不能在 root target 未修复前继续调 selector。

### Coercion-aware proposal refinement

机制通过后，再增加：

- conflict-time centered longitudinal timing proposals；
- local lane-corridor Frenet residuals；
- protected-root-aware yield/pass timing pairs；
- proposal diversity over NCF certificate margin。

目标是提高 `AnyNCFSceneRate`，而不是直接让 selector 更宽松。

## 5.3 当前看起来无效或不应继续作为主算法

- 全局 generic response bank + 事后 nearest-root assignment 作为 q 真值；
- 不含 q 的 OPR；
- safe response 直接等价于 low-burden recovery；
- unconditional aggregate severe hard veto；
- planner 阶段联合修改已验证 certificate；
- legacy flat candidate certificate 作为主机制；
- 单纯调高 `BCOT_RISK_BUDGET`；
- 用稀疏 cached Waymax outcome 宣称闭环性能；
- 在当前 mechanism isolation 同时修改 proposal bank。

---

## 6. 工程层面修复与归因保证

### 6.1 Canonical target dataflow

同一套 paper-aligned target 现在用于：

- fresh label generation；
- v9 raw cache 的 v16.8 transport overlay；
- training batch adaptation；
- learned-offline evaluation；
- mechanism calibration/gate。

这避免“训练一个定义、评估另一个定义”。

### 6.2 Fresh provenance root

v16.8 脚本要求新 `OUT_ROOT`，并记录：

- config hash；
- source code hash；
- raw/transport cache path；
- natural checkpoint hash；
- attribution transfer manifest；
- data protocol。

不得 resume v16.7 transport/planner optimizer。

### 6.3 Correct cache default

`run_cowp_v16_8_dual_gpu.sh` 已修复为默认使用：

- `tensor_cache_train_waymax_transport_v16_8`；
- `tensor_cache_val_waymax_transport_v16_8`；
- `.transport_v16_8` sidecar；
- `v16_8_root_conditioned_overlay` protocol。

即使用户绕过 wrapper 直接运行 driver，也不会静默回退到 v9 transport label。

### 6.4 Immutable certificate

planner optimizer 默认不包含 transport/response 参数。最终闭环 checkpoint 的机制能力与 transport checkpoint 一致，便于将变化归因于 selector。

### 6.5 Regression tests

新增测试覆盖：

1. 冲突 root 的 q 会进入 OPR 并改变 witness；
2. soft root recovery target 被训练读取；
3. root residual 保持原始空间 root identity；
4. planner 默认冻结 transport/response；
5. legacy safe-but-high-burden response 不会被误标为 recovery。

---

## 7. 下一轮结果应该如何判读

### 7.1 先看标签，不要先看模型

overlay 构建后首先要求：

- `error_count = 0`；
- diagnostic `pass = true`；
- canonical aggregate OPR consistency error 接近 0；
- conflict root target confidence 有足够覆盖；
- conflict-root positive rate 不再由全局 response slot 截断决定；
- train/val 分布没有异常漂移。

不要预设 positive rate 必须很高。正确值取决于数据集中真实 conflict 和 burden budget；关键是它必须来自 per-root oracle，而不是 response bank coverage。

### 7.2 再看 RootTransport

若 RCOT 生效，最先应变化的是：

- priority RootTransport conflict-conditioned AUPRC；
- root recovery calibration；
- OPR calibration；
- protected NCF recall；
- accepted rate/fallback。

candidate BCOT AUPRC 已经很高，不应把它作为主要优化目标。

### 7.3 Development gate

v16.8 默认 continuation gate：

- priority root AUPRC >= 0.50；
- priority NCF recall >= 0.30；
- precision >= 0.50；
- accepted rate >= 0.10；
- fallback <= 0.25；
- PBTR <= 0.45；
- PBTR 相比 conventional 有正改善。

这些是内部诊断阈值，不是 CCF-A 官方标准。

### 7.4 若仍失败，如何归因

#### 情况 A：root AUPRC 仍 < 0.50

说明问题仍在 root representation/label separability。下一步运行数据集分层诊断，按以下维度统计：

- priority relation；
- conflict geometry；
- natural source；
- root probability mass；
- time-to-conflict；
- object type；
- beta/burden 区间；
- RCOT oracle profile；
- q positive/negative；
- assignment confidence。

#### 情况 B：root AUPRC 提升，但 recall/coverage 仍低

说明 certificate/selector calibration 是主因。此时做 aggregate severe veto ablation、scene-adaptive budget 和 PBTR-coverage frontier。

#### 情况 C：offline gate 通过但 Waymax progress 差

说明 proposal bank/selector objective 是主因。此时再做 coercion-aware proposal refinement。

#### 情况 D：offline 和 probe 好，full rollout 差

检查 reactive agent model、long-horizon compounding、offroad/wrong-way 和 rollout action conversion；不能再靠 offline certificate 推断 closed-loop SOTA。

---

## 8. 面向 CCF-A / 闭环 SOTA 的推荐研究路线

### 阶段 1：机制可证伪

- 固定 natural checkpoint；
- 只重建 RCOT labels；
- 训练 transport；
- 冻结 mechanism 训练 planner；
- 通过 learned-offline gate。

### 阶段 2：小规模真实闭环

- 100-scene paired probe；
- conventional / v16.7 / v16.8 RCOT 同候选同 seed；
- 标准 Waymax + PBTR/OPR/BTE-CVaR25；
- 检查 progress regression 是否可接受。

### 阶段 3：proposal ceiling 修复

- 固定 RCOT checkpoint；
- 引入 coercion-aware proposal refinement；
- 报告 AnyNCFSceneRate、NCF retention 和 progress regret 分解；
- 做 unchanged proposal bank ablation。

### 阶段 4：paper-grade rebuild

- fresh `formal_v18`；
- seeds 2026/2027/2028；
- 1000+ full Waymax；
- exact paired bootstrap CI；
- reactive agent sensitivity；
- stress set 可视化和失败案例分类。

### 阶段 5：论文表述

建议把贡献压缩为三条：

1. 提出 collision-free 之外的 non-coercive feasibility，并给出 protected burden-transfer / option-preservation 定义；
2. 提出 RCOT，在保持 natural root identity 的条件下估计冲突 root 可恢复性，形成可解释的 monotone certificate；
3. 在真实 closed-loop 中证明其能在有限 progress cost 下同时降低 standard safety failure 和 protected burden transfer，并通过 proposal/certificate 分解解释收益来源。

不建议把每个 loss、每个 heuristic 都写成独立贡献。

---

## 9. 已修改的主要文件

- `cowp/label/safe_responses.py`
- `cowp/label/witness.py`
- `cowp/label/label_engine.py`
- `cowp/data/cache_schema.py`
- `cowp/scripts/26_augment_transport_labels.py`
- `cowp/scripts/27_diagnose_transport_labels.py`
- `cowp/models/losses.py`
- `cowp/scripts/03_train.py`
- `cowp/waymax_eval/rollout.py`
- `cowp/waymax_eval/policy_wrapper.py`
- `configs/label_cowp_v16_8.yaml`
- `configs/train_cowp_v16_8.yaml`
- `configs/model_cowp_v16_8.yaml`
- `configs/eval_cowp_v16_8.yaml`
- `run_cowp_v16_8_dual_gpu.sh`
- `PREPARE_COWP_V16_8_OVERLAY_CN.sh`
- `NEXT_RUN_COMMANDS_V16_8_*.sh`
- `PREPARE_COWP_V16_8_DATA_CN.sh`
- `CHECK_RUN_STATUS_V16_8.sh`
- `tests/test_v16_8_root_conditioned_transport.py`
- `ALGORITHM_CHANGELOG.md`

---

## 10. 当前不能声称的内容

- 不能声称 v16.8 mechanism gate 已通过；
- 不能声称已经取得 closed-loop SOTA；
- 不能声称 CCF-A 录用概率；
- 不能把旧 sparse cached outcome 当作真实 Waymax 主结果；
- 不能把单 seed 结果写成 paper-ready；
- 不能在 RCOT 未验证前把 conformal、复杂生成模型或更多 loss 堆叠成贡献。

当前最合理的判断是：

> v16.7 已证明 coercion-aware candidate ranking 有强信号，但旧 same-root transport target 没有忠实实现论文机制。v16.8 修复后，下一次实验才是对 coercive core idea 的首次相对干净验证。
