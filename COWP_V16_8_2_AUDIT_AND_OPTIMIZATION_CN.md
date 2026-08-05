# COWP v16.8.2 联合审计、Gate 根因分析与算法优化报告

## 0. 审计范围、版本与证据边界

本报告联合审阅了以下材料：

- 论文：`interactive_planning_v16_7_revised.tex`；
- 原始代码：`COWP.zip`；
- 既有建议：`大模型建议(1).md`；
- 算法历史：`ALGORITHM_CHANGELOG.md` 及各版本 changelog；
- 本轮结果：`cowp_v16_8_1_rcot_consistent_v9base_seed2026.zip`；
- 训练历史、calibration sweep、held-out mechanism verification、共享模型方法对比、数据与因果协议诊断。

本次输出代码版本命名为 **COWP v16.8.2 Certificate-Consistent**。它是在 v16.8.1 已完成的 RCOT 定义一致性修复基础上，进一步修正证书评估语义、fallback、protected relation、不确定性和候选覆盖问题。

必须先说明证据边界：上传的结果压缩包包含日志、历史与 JSON 指标，但不包含服务器上的大体积 `cowp_witness_best.pt` / `cowp_planner_best.pt`。当前环境也没有 CUDA 和 Waymax。因此，本报告能够严格判断旧运行为什么失败、修改源码和执行链并完成 CPU/静态回归，但不能在本地重新计算修复后的 GPU gate 数值，也不会声称新版本已经通过 gate 或达到 SOTA。

---

## 1. 两个 gate 是否通过；若失败，根本原因是什么

### 1.1 结论

用户当前阶段要求的两个 continuation gate 均未通过：

| Gate | 当前结果 | 要求 | 结论 |
|---|---:|---:|---|
| `mechanism_verification.pass` | `false` | `true` | 未通过 |
| `mechanism_verification.calibration_feasible` | `false`，`status=least_violation` | `true` | 未通过 |

这两个失败并不等价于“COWP 的 coercion/false-safe 思路无效”。相反，held-out 排序指标较强：

| 指标 | 当前值 | 解释 |
|---|---:|---|
| Pair-witness AUPRC | 0.82033 | witness 信号有效 |
| Protected BCOT false-safe AUPRC | 0.96498 | false-safe 排序很强 |
| Protected RootTransport AUPRC | 0.86936 | RCOT 排序已具有辨识力 |
| Global NCF recall | 0.21618 | 超过 0.18 的全局反退化下限 |
| PBTR 相对 conventional 改善 | 0.12367 | 方向性改善成立 |
| Selected global false-safe 改善 | 0.25978 | 方向性改善成立 |

真正失败的是把强排序信号转化为一个同时满足 recall、precision、接受率、fallback 与 PBTR 的可行 operating point：

| 失败项 | 当前值 | Gate |
|---|---:|---:|
| Protected NCF recall | 0.21176 | ≥ 0.30 |
| Protected NCF precision | 0.49802 | ≥ 0.50 |
| Learned accepted-candidate rate | 0.06851 | ≥ 0.10 |
| Fallback rate | 0.35874 | ≤ 0.25 |
| PBTR | 0.59222 | calibration 约束 ≤ 0.45 |

### 1.2 第一层根因：评估语义把“证书集合”错误地当成“有限 shortlist”

旧代码先构造硬 RCOT certificate mask，随后用最多 8 个候选的 Pareto frontier 覆盖该 mask，并继续用覆盖后的 shortlist 计算：

- `LearnedAcceptedCandidateRate`；
- NCF recall / precision；
- certificate coverage；
- calibration feasibility。

这使 gate 实际测量的是“selector 最终保留多少候选”，而不是“有多少候选满足语义证书”。在每场景约 50 个有效候选、shortlist 最多 8 个时，accepted-rate 会被结构性压低；同时 NCF candidate recall 也会因 shortlist 截断而降低。结果中的一个直接佐证是：

- candidate-level learned NCF recall 只有 0.216；
- 但有 NCF 场景中的 certificate scene retention 达 0.92094；
- protected scene retention 也达 0.91455。

也就是说，大多数“存在可接受 NCF 候选”的场景并没有被整体丢失，低 candidate recall 很大一部分来自 shortlist 计数语义，而不是证书完全无法识别 NCF。

**v16.8.2 修复：**

- `certificate_accepted`：语义硬证书集合，只用于 gate、coverage、recall/precision；
- `selection_shortlist`：在已认证候选中用于计算量控制和最终排序；
- 两套 mask 分别记录，互不覆盖；
- 新指标写入 `CertificateSemantics/Version=v16_8_2_decoupled`；
- calibration 与 verification 拒绝没有该版本标记的旧 JSON。

### 1.3 第二层根因：fallback 语义与计数错误

旧 learned-offline fallback 会返回一个有效的 stop/yield candidate index，但聚合器只在 index 无效时增加 fallback 计数。于是“是否计为 fallback”取决于返回值编码，而不是是否走了 fallback 分支。

更严重的是，旧 fallback 假设 stop/yield/neutral 比较保守。但对后车、汇入车辆、具有优先权的交互者而言，ego 急停或过度让行仍可能迫使对方急制动、改道或失去可行根，因此不能天然视为 non-coercive。

**v16.8.2 修复：**

- 只要进入 fallback 路径，无论最终 candidate index 是否有效，都显式记为 fallback；
- fallback 不再限定 stop-like，而是在常规安全候选内最小化预测 coercion risk：transport UCB、protected-rule 风险、action/pressure、稀疏 outcome 风险和小权重 utility 的组合；
- stop/yield 仅作为弱 tie preference，而不是非胁迫先验；
- 单独报告 fallback-selected PBTR/false-safe，防止总体指标掩盖 fallback 质量。

注意：修复诚实计数后，fallback 数值可能先升高。这不是回归，而是把此前漏计的 valid-index fallback 显式化。判断优化是否有效应看后续新训练与 BCTE 是否能让显式 fallback 真正下降。

### 1.4 第三层根因：真实的上游 proposal coverage 上限

共享模型评估显示：

- `ProposalCoverage/AnyNCFSceneRate = 0.27255`；
- 即只有约 27.25% 场景的候选库中存在至少一个 label-space NCF 候选。

这是真实算法瓶颈，不是单纯 gate 统计 bug。即便证书分类完美，也不可能在没有 NCF proposal 的场景中选择 NCF。现有候选主要由固定控制格、legacy timing offsets 和有限 Waymax rollout 覆盖构成；它们没有围绕“ego 与 protected agent 到达同一冲突区域的相对时序”系统构造 pass-before / pass-after 解。

**v16.8.2 修复：Bidirectional Conflict-Time Envelope（BCTE）**

对附近关键 agent 和共享 conflict region：

1. 估计 agent 到达冲突区域的时间区间；
2. 构造 ego 在该区间前后保留安全时间裕量的目标到达时刻；
3. 解有界常加速度，使 ego 到达时间接近 `t_agent - gap` 或 `t_agent + gap`；
4. 保留 pass-before 与 pass-after 两个方向的可行方案；
5. 仍由 conventional safety 与 RCOT certificate 决定是否接受，BCTE 本身不绕过安全门。

BCTE 已落代码，但必须重建 fresh candidate labels/cache 才能验证 proposal coverage 提升。建议把 `AnyNCFSceneRate >= 0.35` 作为下一轮的最低 upstream 目标，而不是通过放松证书阈值伪造 gate 通过。

### 1.5 第四层根因：protected relation 被 learned head 稀释

旧 offline 规则中，`AgentPriority` 权重为 1、`EqualOrNegotiated` 只有 0.65、`Unprotected` 仍有 0.1，再与 learned priority 做 50/50 混合。结果是：

- 本应硬保护的 equal/negotiated relation 可能掉到 `priority_hard_threshold=0.55` 以下；
- 本应不保护的 relation 可能被 learned score 抬高；
- 论文定义、label、offline selector 与 online policy 不完全同构。

**v16.8.2 修复：**

- `AgentPriority`、`EqualOrNegotiated` 由规则硬锚定为 protected；
- `Unprotected` 硬锚定为 unprotected；
- learned priority 只补充 unknown relation；
- offline、set head、online policy 使用同一语义。

### 1.6 第五层根因：用于 UCB 的 uncertainty 没有覆盖完整证书误差

旧 `mode_uncertainty` 只学习 conflict/retain 误差，却被用于包含 `q` 与 `b*` 的完整 RCOT 风险 UCB。这会产生两种误判：

- recovery/burden 预测不准但 uncertainty 偏低，错误接受 coercive candidate；
- conflict/retain 偏保守导致 uncertainty 偏高，错误拒绝可恢复 root。

**v16.8.2 修复：** uncertainty target 改为 conflict、retain、same-root recovery `q`、归一化 minimum safe burden `b*` 的最大误差，并在有效/高置信标签上监督。

### 1.7 为什么只增大 BCOT budget 不能解决问题

旧 calibration sweep 已给出充分反例：budget 增大时，接受率上升、fallback 下降，但 protected NCF recall 很快在约 0.22–0.23 附近平台化，同时 PBTR 持续恶化。例如：

- budget 0.35：recall 约 0.183，precision 约 0.739，accepted 约 0.0355，fallback 约 0.507，PBTR 约 0.504；
- budget 0.60：recall 约 0.220，precision 约 0.504，accepted 约 0.0625，fallback 约 0.370，PBTR 约 0.566；
- budget 0.90：recall 仍约 0.229，accepted 约 0.099，fallback 约 0.186，但 PBTR 约 0.655。

因此，问题不是 budget 太严，而是：

1. certificate metric 与 shortlist 混淆；
2. proposal 中缺少 NCF；
3. fallback 与 protected semantics 不一致；
4. uncertainty 和训练选择目标错位。

---

## 2. 为实现论文核心 idea 并达到理论上有竞争力的结果，当前主要算法问题及优化方向

### 2.1 论文核心 idea 是成立的，但系统还缺少“闭合三角形”

COWP 的核心不是另一个普通 trajectory backbone，而是：

- 识别其他参与者自然可行行为根；
- 判断 ego candidate 是否破坏 protected agent 的高概率自然根；
- 对冲突根检验 same-root、低负担、安全恢复；
- 用 protected OPR、root-specific burden/CVaR 和 BCOT 给出 non-coercion certificate；
- 在常规安全约束内优先选择有证书的高效方案。

v16.8.1 已修复完整 transport 方程 `s=(1-c)r+cq`、直接 root-conditioned `q/b*`、global response bank 与证书解耦、active-root probability 归一化等定义问题。当前缺的不是再增加一个 planner loss，而是让以下三部分闭合：

1. **Proposal coverage：** 必须生成足够多真正可被认证的交互时序候选；
2. **Certificate calibration：** 证书必须对全部 protected pairs 同时可校准，而不是只提供 point estimate；
3. **Fallback contract：** 无证书时不能退回一个语义不受约束的“保守动作”。

### 2.2 建议的理论主线：Distributional RCOT，而不是继续堆独立 head

当前 `q_ikm` 与 `b*_ikm` 虽已分开，但仍是两个点预测 head，再用 consistency loss 约束。更强、更统一的方向是预测 root-conditioned minimum safe burden 的条件分布：

\[
F_{ikm}(b)=P(B^{\star}_{ikm}\le b\mid x,\tau_i,k,m).
\]

由一个分布同时得到：

- `q_ikm(β_i)=F_ikm(β_i)`；
- `b*_ikm` 的中位数/高分位；
- burden tail risk 与 CVaR；
- one-sided uncertainty bound。

这可称为 **Distributional RCOT (D-RCOT)**。它比“两个 head + consistency penalty”更容易形成理论贡献：恢复概率、最低负担和风险界来自同一条件分布，避免内部不一致。

推荐实现顺序：

1. 用离散 burden bins 或 monotone quantile head 预测 `B*` 分布；
2. 对 no-safe-response 使用右删失/censoring，而不是极大 sentinel 回归；
3. 用 proper scoring rule（例如 censored NLL / CRPS）训练；
4. 由分布计算 `q(β)`、UCB(`b*`) 和 CVaR；
5. 保留现有 point heads 作为 ablation，不直接删除，直到 D-RCOT 在同一数据/seed 上稳定优于它们。

### 2.3 建议的理论证书：同时单侧校准，而不是单一 BCOT budget sweep

论文要形成 CCF-A 级别的理论卖点，建议把 accepted candidate 定义为对所有 protected `(i,k,m)` 同时满足：

- `LCB(q_ikm) >= q_min`；
- `UCB(b*_ikm) <= β_i`；
- `LCB(OPR_i) >= 1-ε_p`；
- `UCB(CVaR_i) <= β_i`；
- conventional safety 通过。

校准对象不是单个平均指标，而是“候选内全部 protected relations 的 family-wise false acceptance risk”。可采用：

- calibration/held-out 严格分离；
- Bonferroni/Holm 作为保守基线；
- learn-then-test / risk-controlling prediction sets；
- 按场景 difficulty、relation type 和 proposal family 做 group-conditional 校准；
- 对 ego policy 改变所诱导的 covariate shift 使用 importance-weighted / robust calibration；
- 部署时做 sequential risk monitoring，检测已校准保证是否在分布漂移下失效。

重要措辞：v16.8.2 只为这一路线清理定义和不确定性 target，**尚未实现或证明有限样本 guarantee**。在论文中应写成 next theorem/extension，而不是把当前 point-estimate gate 描述成已校准安全证书。

### 2.4 Proposal refinement 应由证书梯度/代理风险引导

BCTE 是必要的第一步，但仍是解析候选扩充。后续可做 **Certificate-Guided Proposal Refinement (CGPR)**：

1. 强 proposal planner 给出多模态轨迹；
2. RCOT 网络给出 protected root risk 与 burden UCB；
3. 在轨迹参数空间做少量可微/采样 refinement，优化：
   - conventional safety barrier；
   - RCOT certificate margin；
   - progress/comfort；
4. 保留硬 certificate 作为最后 accept/reject，refinement 只提高可认证候选覆盖。

这样 COWP 可以外挂到 flow/diffusion/VLA/world-model planner 上，并与当前强 planner 的 proposal quality 解耦。论文定位应是“可插拔的交互公平/非胁迫机制与统计证书”，而不是试图用当前轻量 planner backbone 直接在所有 NAVSIM/Bench2Drive 指标上击败专门的大模型 planner。

### 2.5 何谓“理论上达到 SOTA”

目前不能诚实声称理论 SOTA。一个可投稿的强版本至少要同时具备：

1. **新定义/机制：** root-preserving counterfactual transport，已经具备；
2. **统一分布建模：** D-RCOT，使 `q/b*/CVaR` 同源；
3. **有限样本或高概率错误接受界：** simultaneous one-sided calibration；
4. **分布漂移处理：** interaction-induced shift 下的加权/在线风险控制；
5. **proposal-independent claim：** 在多个强 proposal planner 上作为 plugin 均改善 false-safe/PBTR，且不显著损失闭环安全与效率；
6. **闭环证据：** fresh、足够覆盖、无 selector bias 的 Waymax/交互仿真结果和多 seed 统计。

---

## 3. 会误判算法优劣的工程问题

| 问题 | 会造成的误判 | v16.8.2 处理 | 是否需重训 |
|---|---|---|---|
| 硬 certificate mask 被 top-8 shortlist 覆盖 | 接受率/NCF recall 被结构性低估 | certificate 与 shortlist 分离 | 否，先重评估即可 |
| valid-index fallback 未计数 | fallback 被低估，方法看起来更稳定 | 显式 fallback flag | 否，先重评估 |
| stop/yield 被假设为 non-coercive | fallback PBTR 被掩盖 | 最小预测 coercion-risk fallback | 否可重评，最佳效果需重训/BCTE |
| stale JSON 可进入 calibration | 修复后仍可能读取旧语义并假通过/假失败 | metric semantics version + fail-fast | 否 |
| protected relation 与 learned priority 混合 | protected pair 被漏保护或误保护 | rule hard anchor，learned 仅 unknown | 最好重训/重评 |
| uncertainty 只监督 c/r 却给 q/b* UCB | 错误接受/拒绝不可解释 | full-certificate error target | 是 |
| transport checkpoint score 使用 disabled/遗漏 loss | 选到并非最佳 RCOT checkpoint | active objective composite | 是 |
| planner score包含 frozen/zero-weight diagnostics | planner 改善被稀释 | 仅 trainable planner losses | 是 |
| flat candidate certificate 已塌缩却留在 score | 无效 head 可能干扰模型选择 | 降级为 diagnostic-only | 需重新选/训 checkpoint |
| 只有约 23.7% 候选有 cached rollout | outcome-based结论受 selector bias | 只作 auxiliary，不作完整闭环证明 | 需 fresh replay |
| finite log-divergence coverage 为 0 | 训练/报告 logdiv 会产生伪监督 | 继续禁用该 loss | 数据重建前不启用 |
| 历史 shell 脚本 CRLF/旧 heredoc | “全仓库 shell clean”不可复现 | 当前 v16.8.2 执行链单独强门禁；遗留脚本明确标注 | 不影响当前训练 |

此外，结果包的 provenance/source SHA-256 与上传源码中关键 rollout/config/train 文件一致，因此上述 selector/fallback 问题不是“结果来自另一份代码”的猜测，而是当前运行代码真实存在的行为。

---

## 4. 哪些部分保留、深化、删除或修改

### 4.1 应保留

- False-safe / coercion 的问题定义；
- protected-priority relation 语义；
- typed natural decoder、OBS capacity、stable root probability mass；
- active-root support 与 mass-aware envelope；
- pair witness 与 source-aware alignment；
- RCOT 的 complete transport：`s=(1-c)r+cq`；
- direct root-conditioned recovery 与 burden；
- protected OPR、BTE-CVaR25、monotone BCOT；
- conventional-safety-first 的 hard hierarchy；
- transport 在 planner 阶段冻结；
- calibration/held-out 分离与 fail-fast mechanism gate；
- PBTR、false-safe、coverage、efficiency 的多目标评价。

### 4.2 应继续深化

- BCTE / interaction-timing proposal family；
- conflict-conditioned root ranking；
- D-RCOT 的 burden distribution/censoring；
- simultaneous one-sided calibration；
- proposal coverage–certificate precision–efficiency frontier；
- fallback-conditional PBTR；
- protected relation type 的 group calibration；
- certificate-guided proposal refinement；
- 在多个强 proposal planner 上做 plugin transfer；
- 多 seed、fresh labels、足量 Waymax rollout 与 reactive-agent 闭环验证。

### 4.3 应删除或降级

- 用 global response bank nearest-root assignment 定义 `q`；
- 不含 `c*q` 的 OPR；
- 把 `safe response` 等同于 `low-burden same-root recovery`；
- all-critical 无条件 severe veto；
- planner 联合更新已验证 certificate mechanism；
- 把 Pareto shortlist 当成 certificate set；
- 把 stop/yield 当成默认 non-coercive fallback；
- 单纯增大 BCOT budget；
- 用稀疏 cached outcomes 证明完整闭环 SOTA；
- flat candidate certificate 作为主 selector 或 checkpoint 目标。

---

## 5. 当前模型状态、训练截断和主要上游问题

### 5.1 Natural foundation

Natural basis 与 effectiveness gate 均通过：

- set minADE 1.1799 m；
- learned 8 s error 1.1765 m；
- OBS 8 s error 2.7116 m；
- overall gain 0.7098 m；
- OBS gain 1.3922 m；
- effective modes 4.51；
- velocity/yaw consistency 通过。

因此自然根建模不是本轮的首要阻塞。继续反复重训 natural decoder 会增加变量，却不会直接修复当前 mechanism feasibility。

### 5.2 Transport stage：明显 schedule-truncated

Transport 只训练 14 个 epoch，epoch 0 到 13 的每次 checkpoint 都改善，最后：

- `checkpoint/no_improve_checks=0`；
- composite 3.1502 → 2.5937；
- mode conflict loss 0.4334 → 0.3221；
- conflict recovery 0.6016 → 0.4606；
- root burden 0.7533 → 0.3761；
- uncertainty 0.1974 → 0.1002；
- OPR loss 0.2160 → 0.1371；
- candidate priority budget 0.3537 → 0.2624。

这不是已收敛模型，而是执行计划截断。v16.8.2 默认提高到 24 epoch，并保留 early stopping。由于 uncertainty target 和 checkpoint score 已改变，transport 必须重训才能完整验证修复。

### 5.3 Planner stage：同样 schedule-truncated

Planner 只训练 10 个 epoch，每次 checkpoint 均改善，最后 `no_improve_checks=0`：

- val loss 1.5555 → 1.3354；
- imitation 0.6126 → 0.3152；
- ranking 0.3498 → 0.3184；
- checkpoint composite 1.9825 → 1.9596。

transport 指标在 planner 历史中保持完全不变，说明 freeze 生效，这是正确的。planner 默认提高到 16 epoch。

### 5.4 flat candidate certificate 已塌缩

共享模型中，COWP selected candidate 的 flat certificate 输出约为：

- NCF probability 0.0445；
- false-safe probability 0.9994；
- quality probability 1.46e-5；
- risk 3.704。

该 head 没有形成可用判别器，而且其 loss weight 已为 0。它不应被解释为 COWP 主机制失败；当前主 selector 是 transport/BCOT。v16.8.2 将它明确降级为 diagnostics，并从 checkpoint score 移除。

### 5.5 主要上游问题排序

1. **候选覆盖：** AnyNCFSceneRate 27.25%，首要瓶颈；
2. **fallback contract：** 当前 35.87%，且旧计数不完整；
3. **证书 operating region：** precision/recall/PBTR 不可同时满足；
4. **训练截断与 checkpoint 目标错位；**
5. **稀疏 outcome：** selected Waymax coverage 约 20%，全候选约 23.7%，无法支撑完整闭环结论；
6. **fresh protocol 尚未物化：** 当前 overlay 可做机制开发，但不是 fresh v15/v16 causal-label paper protocol；
7. **reactive non-ego 闭环仍不足：** logged replay/constant-velocity proxy 不能等价于真正交互响应分布。

---

## 6. 已落地的算法和代码修改

### 6.1 核心源码

- `cowp/waymax_eval/rollout.py`
  - certificate/shortlist 分离；
  - 显式 fallback；
  - least-coercive fallback score；
  - protected hard anchor；
  - semantics version 与 shortlist/fallback diagnostics。
- `cowp/waymax_eval/policy_wrapper.py`
  - 在线策略同步相同语义；
  - online BCTE candidate；
  - 不再将 stop-like 视为自动认证。
- `cowp/models/set_transport_head.py`
  - protected relation rule anchor。
- `cowp/models/losses.py`
  - full-certificate uncertainty target。
- `cowp/scripts/03_train.py`
  - transport/planner checkpoint score 与 active objective 对齐。
- `cowp/label/ego_candidates.py`
  - target-arrival-time 常加速度求解；
  - agent/conflict-region TTA；
  - bidirectional BCTE proposals。
- `configs/label_cowp_v16_8.yaml`
  - BCTE 与 fallback 风险权重。

### 6.2 评估、校准与协议

- `cowp/scripts/31_calibrate_bcot_budget.py`
  - 拒绝旧 certificate/fallback semantics。
- `cowp/scripts/25_verify_mechanism_effect.py`
  - 显式检查 calibration 与 held-out 均为当前语义。
- `cowp/scripts/36_audit_causal_protocol.py`
  - 支持 `v16_8_2_fresh`；
  - overlay 与 fresh protocol 分开报告。

### 6.3 执行链

新增：

- `NEXT_RUN_COMMANDS_V16_8_2_REEVAL_CURRENT_CN.sh`；
- `NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh`；
- `NEXT_RUN_COMMANDS_V16_8_2_PROBE_CN.sh`；
- `NEXT_RUN_COMMANDS_V16_8_2_FULL_CN.sh`；
- `CHECK_RUN_STATUS_V16_8_2.sh`；
- `PREPARE_COWP_V16_8_2_BCTE_DATA_CN.sh`。

执行器保留外部 `DATA_PROTOCOL`，默认输出 root 更新为 v16.8.2，transport/planner 默认 epoch 更新为 24/16。

### 6.4 论文与日志

- 新论文：`interactive_planning_v16_8_2_revised.tex`；
- 新 changelog：`ALGORITHM_CHANGELOG_v16_8_2.md`；
- canonical `ALGORITHM_CHANGELOG.md` 同步更新；
- 明确 certificate set 与 shortlist、protected-only gate、uncertified fallback、BCTE 与 future calibration 路线；
- 没有把尚未验证的 BCTE 或 calibration guarantee 写成已获得结果。

---

## 7. 下一步执行指令与决策树

详细命令见 `COWP_V16_8_2_EXECUTION_README_CN.md`。推荐严格按以下顺序执行。

### 阶段 0：先用旧 checkpoint 重评估，隔离工程语义修复

目的：不重训，先确认 certificate/shortlist、fallback 计数和 protected semantics 修复对 gate 的纯影响。

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

SOURCE_RUN=outputs/cowp_v16_8_1_rcot_consistent_v9base_seed2026 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_2_reeval_v9base_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_REEVAL_CURRENT_CN.sh
```

前提：服务器上的 `SOURCE_RUN` 必须保留自然、transport 和 planner checkpoint；上传到本地的结果压缩包没有这些大文件。

状态：

```bash
OUT_ROOT=outputs/cowp_v16_8_2_reeval_v9base_seed2026 \
bash CHECK_RUN_STATUS_V16_8_2.sh
```

检查：

- `CertificateSemantics/Version == v16_8_2_decoupled`；
- `FallbackSemantics/ExplicitAccounting == true`；
- 比较 `LearnedAcceptedCandidateRate` 与 `SelectionShortlist/CandidateRate`，两者不应再相同；
- 不要因 fallback 诚实计数后变高而立即否定模型，需同时看 fallback-selected PBTR；
- 仍要求 `pass=true` 且 `calibration_feasible=true` 才可进入 probe。

### 阶段 1：若重评估仍失败，在现有 overlay 上重训 corrected mechanism

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OUT_ROOT=outputs/cowp_v16_8_2_certificate_consistent_v9base_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
TRANSPORT_EPOCHS=24 \
PLANNER_EPOCHS=16 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh
```

现有 overlay 足够验证：

- corrected uncertainty；
- active checkpoint score；
- certificate/shortlist 语义；
- protected hard anchor；
- fallback selector。

但它不能验证 BCTE proposal coverage，因为候选和 labels 已在旧 cache 中物化。

### 阶段 2：若 proposal coverage 仍低或 PBTR/fallback 仍阻塞，重建 fresh BCTE 数据

触发条件建议：

- `ProposalCoverage/AnyNCFSceneRate < 0.35`；或
- calibration 主要违反 PBTR/fallback；或
- accepted scene coverage 仍因无 NCF proposal 受限。

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
RUN_WAYMAX_REPLAY=1 \
MAX_REPLAY_CANDIDATES=24 \
CUDA_VISIBLE_DEVICES=0 \
bash PREPARE_COWP_V16_8_2_BCTE_DATA_CN.sh
```

随后训练：

```bash
cd /path/to/COWP_v16_8_2_certificate_consistent

DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
RAW_TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_train_waymax \
RAW_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_val_waymax \
TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_train_waymax_transport_v16_8_2 \
VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte/tensor_cache_val_waymax_transport_v16_8_2 \
DATA_PROTOCOL=v16_8_2_fresh \
OUT_ROOT=outputs/cowp_v16_8_2_bcte_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
TRANSPORT_EPOCHS=24 \
PLANNER_EPOCHS=16 \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_MECHANISM_CN.sh
```

### 阶段 3：gate 通过后才做 Waymax probe/full

Probe：

```bash
OUT_ROOT=outputs/cowp_v16_8_2_bcte_seed2026 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_PROBE_CN.sh
```

人工检查 probe 的 conventional safety、collision/offroad、PBTR、false-safe、fallback 与 progress，再运行 full：

```bash
OUT_ROOT=outputs/cowp_v16_8_2_bcte_seed2026 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_2_bcte \
CUDA_VISIBLE_DEVICES=0,1 \
BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_2_FULL_CN.sh
```

### 不允许的“修复”

- 不降低 protected recall/precision gate 来让结果通过；
- 不只增大 BCOT budget；
- 不把 shortlist rate 重新命名成 certificate rate；
- 不把 fallback candidate 当作 certified candidate；
- 不在 mechanism 失败时直接跑 full Waymax；
- 不用旧 cached outcome 的稀疏 selected subset 声称闭环 SOTA。

---

## 8. 推荐的后续实验矩阵

### 最小必要消融

1. v16.8.1 old semantics vs v16.8.2 reevaluation，同一 checkpoint；
2. soft protected head vs hard protected anchor；
3. old uncertainty target vs full-certificate target；
4. old fixed timing proposals vs BCTE；
5. stop-like fallback vs least-coercive fallback；
6. BCOT vs pairmax；
7. point `q/b*` vs D-RCOT distribution；
8. uncalibrated point certificate vs simultaneous one-sided calibration。

### 必报曲线

- PBTR–accepted rate；
- protected NCF precision–recall；
- false-safe–progress；
- fallback–proposal coverage；
- AnyNCFSceneRate–number of BCTE candidates；
- certificate set size–shortlist size；
- calibration coverage/error acceptance under shift。

### 论文 claim 层次

- **当前可 claim：** 方法定义、RCOT mechanism、强离线排序信号、旧运行方向性改善、代码一致性修复；
- **gate 通过后可 claim：** disjoint learned-offline mechanism effectiveness；
- **fresh Waymax multi-seed 后可 claim：** 闭环经验结果；
- **完成 simultaneous calibration theorem + empirical coverage 后可 claim：** calibrated non-coercion certificate；
- **只有多 backbone、强 benchmark、统计显著后才讨论 SOTA。**

---

## 9. 本地验证结果

- PyTest：**150 passed**；
- Python `compileall`：通过；
- v16.8.2 实际执行链 shell：全部 `bash -n` 通过；
- TeX：`pdflatex` 编译成功，19 页；
- CPU realistic model/loss preflight：`pass=true`；
- causal audit smoke：
  - `pass=true`；
  - `engineering_pass=true`；
  - `mechanism_overlay_protocol_pass=true`；
  - `fresh_v16_8_2_label_protocol_pass=false`，符合尚未重建 fresh data 的事实。

全仓库历史 shell 并非全部 clean：若干 v15–v16.6 旧脚本含 CRLF/obsolete heredoc。它们不在 v16.8.2 执行链中，但不再做“所有顶层脚本都通过”的不实声明。

---

## 10. 最终判断

1. 两个 mechanism gate 均未通过；
2. 失败由**评估语义 bug + fallback 语义 bug + 真实 proposal coverage/PBTR 瓶颈 + 训练截断**共同造成；
3. 排序 AUPRC 已足以证明核心 coercion/RCOT 信号值得保留；
4. 最优先不是再堆 planner backbone，也不是放松 budget，而是：
   - 重评估旧 checkpoint；
   - 重训 full-certificate uncertainty；
   - 用 BCTE 提高 NCF proposal coverage；
   - 建立 D-RCOT 与 simultaneous calibrated certificate；
5. v16.8.2 已完成代码和论文层面的必要修复，但新 gate 数值必须在服务器 checkpoint/Waymax 环境中重新产生。
