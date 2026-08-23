# COWP v16.8.24 原始结果的当前算法状态与 v16.8.25 小步优化建议

## 结论摘要

这轮不应该重构数据集，也不应该大改 RCOT/BCOT。现有 5k 数据已经足够回答一轮模型/selector 问题。

当前瓶颈按优先级可分为：

1. **固定 proposal bank 的 NCF support ceiling（最大，但本轮暂不动）**；
2. **post-certificate selector / planner 排序（当前最值得用现有 cache 优化）**；
3. **严格在线 physical safety / offroad 风险尚未被正确 exact-ID Waymax 证实**；
4. **训练 protocol 使用 stage=all，导致机制与 planner 的归因不够干净**；
5. **RCOT/BCOT 本身不是当前第一瓶颈**；
6. **affected-root 的 burden-only 扩展在当前数据上没有独立正例，暂不能作为被论证的机制贡献。**

因此 v16.8.25 只做两项算法/训练收紧：

- `cowp_cert_utility`（CTU）：相同 protected BCOT certificate + 相同 physical shield，但 certificate 后只按 planner score 排序，不再第二次用 BCOT frontier；
- planner-only repair 时真正冻结 mechanism-side representation / priority gate，保证 certificate-to-selector interface immutable。

同时修复 exact-ID Waymax evaluator，并输出 outcome-head AUPRC，但 outcome head 暂不参与决策。

---

## 1. 论文主线应该继续收紧到什么

目前最有 CCF-A 潜力且与真实代码一致的主线不是“很多社会机制的组合”，而是：

**False-safe planning → Non-coercive feasibility → natural-root counterfactual representation → Root-Conditioned Counterfactual Transport (RCOT) → monotone BCOT protected-priority certificate → hard admissibility → utility selection。**

真正的核心区别应强调：不是预测“别人会让吗”，也不是给“别人付出的代价”加一个 finite soft penalty，而是问：**同一个非受迫 natural root 在 ego intervention 后，是否还能 transport 成同拓扑、低负担、安全的 response；被保留的 natural option probability mass 是否足够。**

这使 COWP 的 novelty 集中在“root-indexed option transport as feasibility”，而不是 generic safety classification。

---

## 2. 目前哪些机制已经有相对可靠的正证据

### 2.1 RCOT / root recovery：有效信号，当前不是主要瓶颈

Held-out：

- LowSafeExist AUPRC = **0.8974**；
- conflict-conditioned AUPRC = **0.8033**；
- priority-conflict AUPRC = **0.7820**。

这说明模型确实学到了同 root recovery / conflict-conditioned response 的可分辨信号，而不是完全依靠 candidate label。

### 2.2 BCOT：当前最强的 false-safe 判别模块

Held-out：

- priority false-safe AUPRC = **0.8374**；
- global false-safe AUPRC = **0.9281**；
- priority risk pair-ranking accuracy = **0.8384**。

Generic candidate classifier：

- NCF AUPRC = **0.1756**；
- false-safe AUPRC = **0.3544**；
- risk ranking = **0.5366**。

因此论文中应继续把 BCOT/RCOT 放在机制核心；generic candidate classifier 应保持 diagnostic/ablation 身份。

### 2.3 protected-priority hard feasibility：方向上有效

Held-out aggregate：

| Method | EP | Fallback | PBTR | FSR | Selected NCF |
|---|---:|---:|---:|---:|---:|
| COWP | 0.6155 | 0.3267 | **0.4711** | 0.7008 | **0.2867** |
| Conventional safety | 0.4442 | 0.0450 | 0.5196 | 0.7394 | 0.2498 |
| Planner only | 0.4410 | 0 | 0.5202 | 0.7370 | 0.2449 |
| Soft burden only | 0.4431 | 0.0450 | 0.5103 | 0.7302 | 0.2585 |
| Universal NCF | 0.5233 | **0.5625** | 0.4835 | 0.7059 | 0.2818 |

可支持的结论是：

- 单纯 planner / conventional selection 不足以降低 priority burden transfer；
- finite soft burden ranking 的方向弱于 hard protected feasibility；
- universal all-critical hard veto 明显增加 fallback，且 PBTR 并未优于 protected-priority COWP，因此不适合作为主定义。

但目前只有一个 checkpoint / 没有 paired CI，所以还不能把这些差异写成最终 publication-level 因果结论。

---

## 3. 当前最大的算法瓶颈：proposal ceiling，但本轮不重构

Held-out：

- Any conventional-safe scene = **95.5%**；
- Any NCF scene = **36.35%**；
- fixed-bank best-case global false-safe floor = **59.48%**；
- best-case PBTR lower bound = **41.32%**。

Validation 同样稳定：AnyNCF **35.11%**，global false-safe floor **60.37%**。

这解释了为什么 validation calibration 报：

`status = proposal_infeasible`

而不是“BCOT threshold 还没调好”。你要求 selected global false-safe <=0.55，但固定 bank 的 oracle floor 已经 0.6037，任何只改 threshold / rank / fallback 的算法都不可能达到该门槛。

### 为什么现在仍然不应该重构 proposal/data

因为你当前的约束是合理的：proposal bank 是大 ceiling，但重建 label/cache 成本很高；同时现有 selector 还有可测的 7--8 pp gap，并且 strict Waymax 还没正确运行。因此在动 proposal 前应该先把“固定 bank 上能做到的事情”做干净，否则新的 proposal 改善会和 selector/training protocol 混在一起。

### Fallback 进一步证明 proposal ceiling 是主因

COWP held-out fallback = **32.67%**；fallback-conditioned PBTR = **82.41%**，看起来很糟。

但关键诊断是：fallback scene 中只有 **7.91%** 原本存在任何 NCF candidate。

所以约 92% fallback 并不是“selector 明明有 NCF 却没选”，而是 bank/certificate universe 中根本没有 NCF proposal。现在专门优化 fallback heuristic 的上限很低，而且可能把 uncertified fallback 错写成“safe”。

---

## 4. 不重建数据时，最值得优化的是 post-certificate selection

Held-out：

- NCF available 时当前 selector 选中 NCF 的概率 = **78.88%**；
- selected false-safe 比固定-bank oracle floor 高 **7.68 pp**；
- Planner ranking pair accuracy = **61.32%**。

这部分是当前最明确、且不需要重新生成任何 label/cache 的剩余空间。

### 原 selector 的结构问题

当前 `cowp` 实际做了两层 BCOT 使用：

1. BCOT risk / pair severe witness 决定 hard admissibility；
2. 已经 accepted 后，再把 BCOT/non-coercive risk 送入 `select_set_preservation_frontier_*` 做第二次 shortlist / ranking。

这与论文正文“certificate 定义 feasible set，然后 ego utility 在 feasible set 中排序”的抽象并不完全一致。

第二次 BCOT 排序可能是有帮助的，也可能在已通过硬证书后过度压制 utility / physical-quality 信息。现有结果无法凭空判断。因此最合理的下一步不是直接删除它，而是做**单因素 paired probe**。

### v16.8.25 CTU

`cowp_cert_utility`：

- certificate 与 COWP 完全相同；
- action/rule/outcome physical shield 与 COWP 相同；
- 唯一变化：不再第二次用 BCOT frontier，在剩余 candidates 中直接按 planner score 选。

如果 CTU 变好，说明当前二次 BCOT ranking 是一个真实瓶颈；如果 CTU 变差，反而证明 BCOT 在 threshold 之外仍包含有用 ordering information，论文就应诚实保留“certificate + risk-aware tie/ranking”而不是硬套纯 hard-first 表述。

---

## 5. 当前 physical safety 是第二个必须尽快确认的问题

原 held-out 的 attached candidate Waymax outcomes 只有 623/1200 selected scenes 有效（52% coverage）：

- COWP collision ≈ **2.73%**；
- offroad ≈ **11.24%**；
- unsafe ≈ **13.96%**。

Conventional-safety 的 partial unsafe 约 **7.47%**。

这不能当 strict closed-loop 结论，因为：

1. coverage 只有约一半；
2. 它是 cache 中 attach 的 candidate replay outcome；
3. 原 strict Waymax 命令实际上没有真正的 exact-ID CLI plumbing。

但它足够构成一个**风险警报**：selector/planner 可能为了 progress / NCF 选到了更容易 offroad 的 candidate。

因此本轮代码新增 candidate-level outcome-head AUPRC diagnostics，但保持 `outcome-risk-penalty=0`。只有当 outcome head 对 collision/offroad/unsafe 的 AUPRC 足够强，并且 paired strict Waymax 证明 physical safety 是 selector-side 可预测问题时，下一轮才值得试一个很轻的 outcome guard。

---

## 6. 原训练 protocol 的问题：stage=all 不能提供干净机制归因

你的原命令使用 `--stage all`。

原代码中：

- `all` 同时优化 natural / response / witness / SetTransport / planner；
- `all` 不走 planner-stage freeze policy；
- `all` 的 best checkpoint 按 total loss 选择；
- `history_all.json` 25 个记录的 `checkpoint/kind` 全是 `loss`。

训练并没有失败：

- natural obs minADE 约 **3.77 -> 2.19 m**；
- transport recovery相关 loss 明显下降；
- planner ranking loss 约 **0.641 -> 0.527**；
- outcome cls 也下降。

所以不能说当前 checkpoint “没学到东西”。问题是：**最终 planner 性能与 mechanism quality 是 joint optimized 的，无法证明 planner 没有通过共同表示调整改变 certificate。**

v16.8 changelog 本来就要求 immutable mechanism checkpoint。因此下一次如果需要训练，本轮代码把 planner-only freeze 做严格：candidate encoder、natural、witness、SetTransport/BCOT、response、learned priority gate 全部固定，然后只做短程 downstream planner repair。

这不需要重建数据。

---

## 7. 哪些机制目前无效、冗余或证据不足

### 7.1 Generic candidate certificate：当前明显弱，不应作为主机制

AUPRC / ranking 接近弱分类器，远弱于 BCOT。保留为 ablation，不能替代 RCOT/BCOT。

### 7.2 Threshold-only BCOT tuning：无效

Validation budget 从 0.20/0.25 一直到 0.50，certificate acceptance / fallback 巨幅变化，但 selected global false-safe 基本仍在约 0.672--0.676，PBTR 也只在约 0.45--0.47 范围。说明 tighten gate 主要把场景推入 fallback，而不是创造 NCF solution。

### 7.3 Universal NCF hard veto：过度保守

Fallback 从 COWP 32.7% 上升到 56.3%，EP 下降，而且 PBTR 并未更好。应继续作为 stricter diagnostic，不做主 hard gate。

### 7.4 Soft burden cost only：不够

在同 bank/checkpoint 下 PBTR 仍高于 COWP，支持“coercion 不是一个普通 finite soft cost”这一论文 motivation。

### 7.5 affected-root 的 burden-only 通道：当前无法独立论证

现有 train/val/held-out 中：

- `affected_root == unsafe_root`；
- burden-only affected roots = 0。

所以虽然 v16.8.9 代码支持 `unsafe OR burden>beta`，实际本数据没有第二类正例。当前同时训练 conflict/affected 两套相关 loss 很可能只是重复监督，而不是学到额外机制。

本轮**不直接删**，因为当前 checkpoint 已按 affected 语义训练，删除会混入重训变量。后续应做一个当前-config 的 one-factor conflict-only retrain：只关闭 `use_affected_root_transport` 和 affected loss，其他 max critical agents、priority mix、seed、init、data order 全部保持一致。若无显著退化，论文主算法应删掉 affected-only novelty，直到有真正 burden-only 数据。

### 7.6 历史 proposal 尝试不要重复

Changelog 已记录：PCHR 在旧 probe 中候选量极少且贡献 0 NCF / 0 priority-NCF；PSY 虽物理有效并产生 pair-level priority-NCF，但 scene-level AnyNCF / false-safe / PBTR 增量为 0；删除 RMR/PSY/legacy timing 改变 candidate 数量但不改当时 scene-level ceiling。

这些不能简单外推到当前 5k 的所有 source，但足以说明下一轮如果最终进入 proposal redesign，不能再做“多加一个 stop/yield/timing primitive”式重复尝试，必须针对 **scene-level support floor** 做 mechanism-attributed proposal repair。

---

## 8. 下一轮最小实验矩阵

### Phase A：完全不训练

1. CTU validation budget sweep；
2. COWP vs CTU validation paired shared-pass；
3. 读取 `OutcomeHead/*_AUPRC`；
4. validation 非劣后，再跑 held-out paired learned-offline；
5. SHA 固定 200 个 held-out IDs，COWP/CTU 做 exact-ID strict Waymax。

**必须满足的 sanity invariant：** COWP 和 CTU 的 `ProposalCoverage/*`、`PriorityCertificate/Accept*`、`BCOT/*`、`RootTransport/*`、certificate coverage 必须相同。否则实现有 bug，不能解释 selected metric 差异。

CTU engineering screen（仅作为是否值得 strict probe 的预注册门槛，不是 paper statistical claim）：

- PBTR 不恶化 >1 pp；
- SelectedFalseSafe 不恶化 >1 pp；
- EP 至少保留 COWP 的 95%；
- certificate invariants 全部通过。

### Phase B：只有 Phase A 显示 planner-side room 才训练

用当前 `cowp_all_best.pt` warm start，现有 train/val cache：

- stage = planner；
- 6 epochs；
- lr = 1e-5；
- mechanism-side modules immutable；
- planner-specific checkpoint composite；
- outcome labels继续使用已有 attached cache；
- 不修改任何 labels/proposals。

目标不是“再训久一点”，而是回答：**在 fixed RCOT/BCOT certificate 上重新优化 planner，能否降低 NCF progress regret / physical unsafe，同时保持 PBTR。**

### Phase C：selector 锁定后再做机制消融

1. affected-root vs conflict-only（clean current one-factor retrain）；
2. causal relevance on/off（clean current one-factor retrain）；
3. protected-priority vs universal hard gate（当前已有强方向证据，可补 multi-seed/paired CI）；
4. hard certificate vs soft burden（multi-seed/paired CI）；
5. outcome guard 仅在 outcome-head AUPRC 和 strict Waymax 支持时才测试。

---

## 9. 什么时候才应该重新构建数据/改 proposal

如果出现以下组合：

- CTU 与原 COWP 基本相同；
- planner-only repair 也基本相同；
- BCOT/RCOT AUPRC 保持强；
- strict Waymax physical safety不是主要可修复的 selector bug；
- selected false-safe 仍紧贴 59--60% proposal oracle floor；

那么就可以很有把握地说：**剩余 dominant bottleneck 已经是 proposal support / dataset-generated candidate universe，而不是 model head。**

到那时再承担一次高成本 rebuild 是值得的，而且新 proposal 的目标应该是降低 scene-level oracle floor，而不是单纯增加 candidate 数量。

---

## 10. CCF-A 论文证据边界

当前结果可以支撑“机制方向选择”和“下一轮算法决策”，但还不够最终投稿结论：

- 当前 held-out 已被用于算法分析，后续应视为 developer held-out，不再是真正 blind final test；
- strict exact-ID Waymax 尚未跑；
- 单 checkpoint，缺 multi-seed paired CI；
- logged replay 不能作为 counterfactual burden causal ground truth；
- affected burden-only 机制当前无独立正例。

因此当前最正确的 CCF-A 策略不是增加更多 novelty，而是**把 RCOT/BCOT 的主线做薄、做强、做可归因**：保留已经被数据支持的机制，删除/降级无独立证据的机制，把 proposal ceiling、selector excess、certificate discrimination 三层分别报告。
