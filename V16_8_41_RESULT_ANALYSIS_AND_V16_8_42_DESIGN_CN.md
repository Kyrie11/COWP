# V16.8.41 结果可靠性、算法归因与 V16.8.42 设计报告

## 0. 执行结论

本轮结论分成两个完全独立的问题：

1. **V16.8.41 的实验结果是否可信？——可信。** 独立可靠性审计的 107 项阻断性检查全部通过，因此允许做算法归因。
2. **V16.8.41 的算法是否成功？——没有通过预注册 Stage-1 GO。** 六项冻结 Gate 通过 5/6，唯一失败项仍是历史 RVR rescue 召回，仅保留 3/10。V41 必须归档，不能运行 `fresh37`。

V41 的修复已经消除了 V40 的机制归因阻断，但修复后结果证明：**first-action interval completion 不是当前值得继续扩展的主机制。** 它在 1,551,139 次区间假设评估中只产生 5 个新认证首拍动作、实际只选择 3 step，且没有在 V39 之上新增任何 collision rescue。

按照上一轮已经预注册的判断分支——“新首拍动作被真实选择但 Gate 仍失败”——V16.8.42 应停止首拍网格、插值、release schedule 和 switch-time 修补，进入：

> **Root-Conditioned Interaction-Aware Reachable-Response Envelope（RC-IARE）**

其研究问题不是“还能不能找到一个更细的 ego 首拍动作”，而是：

> 当固定周车轨迹/CV 假设令 V39 的 hard physical support 为空时，能否在不放松 ego 物理证书、不改变成熟 selector 和控制器的前提下，以 surrounding-agent 的高概率 natural root 为条件，证明每个冲突参与者仍保有低负担、同 root、current/shift 均可执行且多车联合兼容的响应，从而恢复真实的 joint recourse support？

V42 代码已落地，并通过 48/48 项阻断性代码审计、96/96 focused/regression tests 和 250/250 组 canonical root 权重随机等价检查；但尚未执行完整 V42 Waymax rollout，因此**本文档不声称 V42 已取得性能提升**。

---

## 1. 对论文研究方向的理解

论文的核心不是一般意义上的“更礼貌规划”，而是定义并解决一个硬安全缺陷：

- ego 轨迹可以在回放或预测下无碰撞；
- 但无碰撞依赖其他交通参与者急刹、强制让行、放弃合法 gap 或承担过高负担；
- 因此这种轨迹是 **false-safe / safety-by-coercion**；
- coercion 不能只作为有限权重 soft cost，因为足够大的 ego utility 优势仍可压过任何有限惩罚；
- 正确的算法对象是 **non-coercive feasibility**：被保护的优先参与者必须保有足够的低负担安全选项。

论文主线可以抽象为五个层次：

1. **Natural basis**：构建 observational、ego-neutral、priority-preserving 的稳定语义 roots，并用概率质量而非单一 endpoint 维持 root identity。
2. **Same-root response transport（RCOT）**：对于候选 ego 轨迹，判断每个 natural root 是否仍可通过同 root 的低负担响应实现；不能跨 root 偷换语义。
3. **Burden/CVaR transport（BCOT）**：将可恢复 root 的概率质量、尾部 burden 和 option preservation 组合成候选级机制量。
4. **Protected-priority hard NCF**：对 protected-priority relations 使用 hard feasibility；all-critical 只作为更严格诊断，避免 ego-priority 关系被错误地全局 hard veto。
5. **Hard-first selector**：先构建满足 hard physical + non-coercive certificate 的候选集，再在集合内优化 progress/utility；无认证候选时显式 fallback，不能让 scalar score 绕过证书。

本次 V42 只修改第 4 层与第 5 层之间的 **online proposal/support interface**：它不重训 natural model，不改 RCOT/BCOT/P-NCF，不改 selector，而是让 hard physical support 从“周车固定轨迹”升级为“对高概率 stable roots 存在低负担 joint response 的 robust recourse support”。

这与论文中的“proposal sufficiency 必须与 certificate/selector 分开论证”一致：V39–V41 已经说明，证书和 selector 可能正确，但 proposal/support object 不足会让机制几乎无法介入。

---

## 2. V16.8.41 可靠性审计

### 2.1 审计结论

| 审计块 | 结论 | 核心证据 |
|---|---:|---|
| manifest 与场景集合 | PASS | exact200、equivalence16、counterfactual48、fresh37 的数量、唯一性和逻辑 SHA256 全部匹配冻结值 |
| shard 完整性 | PASS | 两个 counterfactual48 shard 各 24 场景，互斥并集等于 48 场景 manifest |
| merged 重构 | PASS | merged 中每个场景的标准指标与 diagnostics 可由 shard 精确重构 |
| 指标重算 | PASS | 所有 standard metric summary 和 fallback rate 由逐场景数据独立重算一致 |
| common-path equivalence | PASS | equivalence16：16 场景、1120 字段、0 mismatch |
| analyzer 复跑 | PASS | 独立 analyzer 与上传 analysis 递归 0 mismatch，文件 SHA256 相同 |
| V41 修复语义 | PASS | V39/V41 共用统一 shift helper；LOWER_ALL/UPPER_ALL terminal identity 被保留 |
| causality | PASS | repaired constructor 不读取 logged future、online GT 或未执行策略真实 future |
| focused tests | PASS | V41 repair suite 与历史 focused suite 通过 |
| 总体 | **PASS** | **107/107 blocking checks** |

审计产物：`V16_8_41_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT_INDEPENDENT.json`。

### 2.2 仍需明确的证据边界

这些边界不否定 V41 结果的工程可靠性，但限制论文结论：

- 结果包没有包含 checkpoint bytes，只能审计逻辑 checkpoint 路径，无法从结果包单独重建权重字节级 provenance。
- 结果 JSON 没有嵌入 source-tree content hash；本轮通过代码语义、shard 和输出一致性做了独立交叉审计，但服务器运行代码的精确字节 provenance 仍不是密码学闭环。
- 非 ego agent 是 logged replay；它可支持冻结协议下的 closed-loop physical outcome 归因，不能单独证明真实 counterfactual social burden 或周车会采取所认证响应。
- counterfactual48 已经多轮用于 mechanism selection，是 development panel，不能再作为最终论文 holdout。

因此：V41 可以用于选择下一算法分支，但不能直接支撑最终 CCF-A 级 causal social claim。

---

## 3. 按冻结预注册 Gate 判断 V41

### 3.1 六项 Gate

| 条件 | GO 阈值 | V41 | 判断 |
|---|---:|---:|---:|
| old RVR rescues retained | ≥ 5/10 | **3/10** | **FAIL** |
| old RVR induced avoided | ≥ 7/9 | **9/9** | PASS |
| 相对 COWP 净减少 collision | ≥ 3 scenes | **4** | PASS |
| Kinematics 净退化 | ≤ 1 scene | **0** | PASS |
| paired mean EP Δ vs COWP | ≥ −0.05 | **−0.010855** | PASS |
| action-changing intervention | > 0 | **90 steps** | PASS |

Gate 是 conjunction；任意一项失败即失败，不能看到结果后下调 5/10 阈值。

最终状态：

```text
V16.8.41 = RELIABLE
           + STAGE-1 FAIL
           + ARCHIVE
           + NO FRESH37
```

### 3.2 Headline 指标

| 方法 | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 = 0.7083 | 1/48 | 6/48 | 1.002512 |
| RVR | 33/48 = 0.6875 | 1/48 | 9/48 | 0.823619 |
| V39 conflict-window tube | 30/48 = 0.6250 | 1/48 | 6/48 | 0.991434（约） |
| V41 repaired interval | **30/48 = 0.6250** | **1/48** | **6/48** | **0.991657** |

V41 与 V39 在 48 场景的 collision/offroad/kinematics outcome **完全相同**。V41 的 paired EP 相对 V39 仅增加约 `+0.000223`；几乎所有场景完全一致，唯一可见变化集中在一个已经被 V39 rescue 的场景。

---

## 4. V41 机制级归因

### 4.1 真正成功、应保留的机制

#### A. V39 controller-lifted conflict-window tube

这是目前最成熟、最值得保留的 recovery 核心：

- 相对 COWP 净 rescue 4 个 collision 场景；
- 0 个 collision induced；
- 0 个 kinematics 净退化；
- EP 退化在预注册容忍范围内；
- current full-horizon physical certificate 与 one-step shift closure 均保持 hard；
- 90 个 action-changing steps 中，88/91 个 certificate 仍由 nested V39 分支选择。

结论：**V39 是 V42 的不可破坏 mature base，不应重写、调权或被新的 support object 取代。** V42 必须 exact-nest V39：只在 V39 空时扩展。

#### B. V41 shift semantic repair

V41 的研究贡献不是性能，而是归因可信度：

- 它修复了 V40 对 LOWER_ALL/UPPER_ALL 终端 shift identity 的错误；
- V39 与 interval constructor 共享同一 helper；
- 排除了“V40 低召回只是 shift bug 造成”的替代解释。

这部分应长期保留为基础设施，但不应写成论文的算法创新。

#### C. hard physical + shift closure + actual execution override

V41 结果说明，完整证书与实际 emitted action 的闭环仍可工作，且没有以 kinematics 为代价获得 collision 改善。应继续冻结：

- 8 s conventional horizon；
- controller projection；
- acceleration/deceleration/jerk/yaw limits；
- roadgraph；
- Waymax inverse-dynamics check；
- current/shift 双证书；
- no-valid emergency semantics。

#### D. hard-first selector

失败证据指向 support sparsity，而不是 selector 选错：interval extension 只有 3 个新首拍被选择，且没有形成新增 rescue。继续改 selector、frontier 或 risk/progress 权重没有针对当前瓶颈，反而可能破坏 V39 的 clean precision。

### 4.2 失败且应停止的机制

#### First-action interval completion

聚合统计：

```text
policy steps                         3,840
V39/V41 recovery probes             2,629
interval attempts                   2,541
interval basis points             490,114
seed evaluations                1,470,342
boundary proposals                 80,797
interval hypotheses             1,551,139
full physically safe                  98
shift closed                           19
unique certified actions               10
new certified first actions             5
selected new first actions              3 steps
additional collision rescue over V39    0
```

其转化链极端稀疏：

- `full-safe / hypothesis ≈ 6.3e-5`；
- `shift-closed / hypothesis ≈ 1.2e-5`；
- `selected new action / interval attempt ≈ 0.00118`；
- 即使选择新首拍，也没有形成新的场景级 rescue。

因此失败不是“还缺几个 interpolation fractions”。继续增加区间点只会继续在相同 fixed-future certificate 上做更密集搜索，没有证据表明会跨越 support ceiling。

### 4.3 三个新首拍选择的反事实解释

| 场景 | 新首拍 step | V39 Collision | V41 Collision | EP 变化 | 解释 |
|---|---:|---:|---:|---:|---|
| `1c03d20597561d94` | 1 | 1 | 1 | 0 | 有局部首拍差异，无 outcome 改善 |
| `a6f8b2b348015743` | 1 | 1 | 1 | 0 | 有局部首拍差异，无 outcome 改善 |
| `f8d5d2f0f7cf5825` | 1 | 0 | 0 | +0.010712 | 已由 V39 rescue；不是新增召回 |

这构成清晰证据：**V41 的新增动作支持是真实执行的，但不是当前缺失 rescues 的主要限制。**

---

## 5. Dominant bottleneck 收紧

### 5.1 Type A：fixed-path support mismatch（P0）

七个仍未保留的历史 RVR rescues 中，六个具有大量 interval attempts，却没有任何 full-safe/shift-closed 新首拍：

- `c9b1c562b6ff31e5`：66 attempts，0 certificate；
- `9e3e5f19ee38f2e3`：80 attempts，0 certificate；
- `ad7d72d8adca3e25`：80 attempts，0 certificate；
- `b85168f48c8c9970`：80 attempts，0 certificate；
- `9ccf60966ec93c20`：74 attempts，0 certificate；
- `2c2395ec28c6a158`：80 attempts，0 certificate。

这些场景中，ego 首拍连续化并没有产生可通过 fixed surrounding-agent CV/full-physical certificate 的路径。最合理的下一假设是：

> 冲突不是仅由 ego 固定轨迹几何决定；只要受影响周车保留合理、低负担且同 natural root 的响应，joint system 可能可行。当前 certificate 把周车冻结，因此把“可交互解决”误判为“无 support”。

这正是 interaction-aware reachable response envelope 要验证的内容。

### 5.2 Type B：certificate-to-closed-loop mismatch（P1 monitor）

`fccd9a25a2a57a73` 是不同类型：V39/V41 产生 4 certificate 和 4 action changes，但最终仍 collision。说明在至少一个场景中：

- model-relative current/shift certificate 有局部支持；
- 后续 replanning、周车行为、状态误差或证书模型仍未保证实际闭环避免碰撞。

这是重要问题，但目前只有一个主要 counterexample；如果同时在 V42 修 accepted path、dwell、kinematics、state uncertainty，会破坏一因子归因。V42 应只记录它是否仍失败，并保留 trajectory/first-collision diagnostics，暂不合并修复。

### 5.3 优先级判断

当前应先解决 **Type A dominant bottleneck**，原因：

1. 它覆盖至少六个缺失 rescue，是 Stage-1 唯一失败 Gate 的主要来源；
2. 它可通过“仅在 V39 空时”的严格嵌套设计解决，不伤害成熟层；
3. 它直接对应论文的 non-coercive recourse 主线，研究价值高于继续工程化首拍搜索；
4. Type B 可在 V42 输出中被更清晰地隔离，并作为后续 uncertainty/closed-loop invariance 分支。

---

## 6. 各模型/算法层成熟度

| 层 | 成熟度 | 当前证据 | V42 策略 |
|---|---|---|---|
| 数据 split / cache 基础 | 较成熟，冻结 | 5000/1000/1200；全量可读；场景 ID 固定 | 不重建、不换 split |
| natural root coverage | 较成熟，冻结 | train/val/heldout rootless=0，<2 low-burden roots=0，protected root coverage≈99.4% | 复用训练输出，不改 decoder |
| natural audit cleanliness | 部分成熟 | mechanism unauditable≈4.1–4.5%；cache verifier 因 58,243 irrelevant pair blockers 标记 fail，但 canonical/affected/retained mismatch 均 0 | 作为论文边界，非本轮重建理由 |
| root identity / canonical probability | 成熟，冻结 | train/label/SetTransport 已有统一 p_min + floor measure | V42 直接调用同一 canonical helper |
| RCOT / BCOT | 较成熟，冻结 | same-root、burden/CVaR 语义已稳定，论文主机制依赖 | 不改阈值、预算、训练 |
| protected-priority hard NCF | 成熟，冻结 | hard feasibility 与 soft courtesy 明确分离 | V42 blocker response 采用保守 AGENT_PRIORITY |
| conventional proposal bank | 中等成熟 | 能产生 4 个 V39 rescue，但大量场景 conventional support collapse | 不增加宏动作/网格；只改变 joint support 解释 |
| V39 conflict-window tube | 成熟正机制 | 4 rescue/0 induced/0 kin regression | exact nested，优先返回 |
| V40/V41 interval completion | 成熟负结论 | 新动作真实选择但无新增 rescue | archive，不再扩展 |
| interaction-aware response support | **新层，未验证** | 由 V41 预注册分支选择；尚无 V42 outcome | V42 唯一新机制 |
| hard-first selector | 成熟，冻结 | 当前失败不是 selector 覆盖错误 | 不改 frontier/score |
| controller projection / shift semantics | 成熟，冻结 | V41 reliability repair + equivalence | 不改 limits 和 helper |
| certificate→长期 closed-loop invariance | 不成熟 | `fccd...` 有证书仍 collision | P1 monitor，下一独立分支 |
| social causal evaluation | 不成熟 | logged replay 无真实 counterfactual response | 后续 reactive-agent + human-audited stress |
| publication statistics | 不成熟 | 当前是 development panel、单协议 | fresh holdout、≥3 seeds、paired CI/uncertainty |

### 下一步模型最应该学习/解决的内容

不是继续学习“更激进/更保守的 ego scalar preference”，而是：

1. **Root-conditioned recourse availability**：给定 ego candidate 和周车 natural root，哪些低负担响应仍可执行？
2. **Joint compatibility**：不同 blocker 的 root-conditioned responses 能否同时实现，而不是每车独立可行、合起来冲突？
3. **Shift-stable response support**：响应不只在当前 tube 可行，也要在实际 emitted first action 后仍可行。
4. **Support calibration/uncertainty**：未来若 analytic bank 证明方向有效，再学习 response viability/uncertainty，而不是先上黑盒 viability head。
5. **Closed-loop certificate calibration**：针对 Type B，后续独立研究多步 uncertainty tube 或 receding-horizon invariance。

---

## 7. V16.8.42 机制设计

### 7.1 名称与方法 ID

```text
V16.8.42 — Root-Conditioned Interaction-Aware Reachable-Response Envelope
简称：RC-IARE
method: cowp_interaction_aware_reachable_response_envelope
```

### 7.2 核心不变量

V42 必须同时满足：

1. **V39 exact nesting**：V39 有 certificate 时直接返回，V42 extension 不运行。
2. **No relaxation**：ego roadgraph、inverse dynamics、full horizon、current/shift 和 non-blocking actors 全部保持 hard。
3. **Same-root universality**：被保留的每个 root 都必须存在 response；不能只证明平均 root 或最有利 root。
4. **Low burden**：response 必须低于冻结 adaptive β；不能通过急刹/priority surrender 获得“可行”。
5. **Joint realizability**：多个 blocker 的 responses 必须联合兼容；不能把独立证书错误拼接。
6. **Causality**：只使用当前状态、natural model、地图和 analytic dynamics；不得读取 logged future/GT。
7. **One-factor attribution**：不改训练、不改 selector、不改 controller、不修 Type B。

### 7.3 Canonical root support

对 critical agent `i` 的 natural logits 做 softmax 得原始质量 `p_im`。使用与标签/SetTransport 完全相同的概率测度：

1. 支持过滤：`p_im >= p_min`，其中 `p_min=0.03`；若全部低于阈值，则保留所有 valid modes，避免空集合。
2. 在支持内重新归一化。
3. 独立 floor smoothing：

```text
p_tilde_im = (1 - epsilon_p) * p_norm_im + epsilon_p / |support|,
epsilon_p = 0.02.
```

4. 按冻结 0.10 m mean-path 距离去重，合并 canonical mass。
5. 至少保留 2 个 roots，并按质量从高到低覆盖至少 0.75 canonical mass。

关键修正：`p_min=0.03` 与 `epsilon_p=0.02` 是两种不同操作，不能取 max 后混为一个阈值。V42 代码直接调用已有 `canonical_root_weights` helper，消除 train/label/online 语义漂移。

### 7.4 Same-root response envelope

对每个 retained root：

- 复用现有 `build_root_recovery_trajectory_bank`；
- 复用 `prepare_root_recovery_burden_bank`；
- 按 `PriorityRelation.AGENT_PRIORITY` 计算 adaptive β，采用保守优先权；
- profile 必须：
  - burden ≤ β；
  - current trajectory roadgraph-drivable；
  - shifted trajectory roadgraph-drivable；
  - current Waymax inverse-dynamics feasible；
  - shifted Waymax inverse-dynamics feasible。

natural decoder 是训练过的 causal root basis；V42 不启用当前训练权重下没有直接监督保证的 dense response trajectory head。

### 7.5 Blocker decomposition

对每个 V39 controller-reachable hypothesis，分别在 current 和 shifted collision contexts 中找 exact blockers：

```text
B = blockers_current ∪ blockers_shifted
```

- `B` 为空：不是 interaction extension 的目标，拒绝。
- 任一 blocker 没有 ready response support：拒绝。
- 只从 ego collision context 中移除 `B`，构造 residual context；所有其他 actors 仍按冻结 CV/physical certificate 检查。
- current/shift residual physical certificate 都必须通过。

这不是“忽略周车碰撞”，而是把 exact blocker 从固定轨迹假设替换为更严格的 root-conditioned response certificate。

### 7.6 Ego—responder 与 responder—environment 约束

对每个 blocker、每个 retained root、每个候选 profile：

- profile 与 current ego tube 安全；
- shifted profile 与 shifted ego tube 安全；
- profile 与每个 non-blocking frozen-CV actor 双向安全；
- shifted profile 与其 shifted CV trajectory 双向安全。

“双向”是因为 COWP unsafe predicate 包含方向相关的 following/RSS/TTC 语义；只检查一个方向可能漏掉 responder 迫使背景车承担负担的情况。

### 7.7 多 blocker exact CSP

构造节点 `(agent, retained_root)`，每个节点的 domain 是通过 ego 与 environment 检查的 profiles。

使用 minimum-domain-first deterministic backtracking，选择每个节点一个 profile。不同 agent 的已选 profiles 必须在 current/shift、两个方向均安全。相同 agent 的不同 roots 表示互斥的自然模式，不要求同时存在，因此不做 pairwise collision。

若 CSP 有解，则得到一个“每个 root 有对应 response 且所有跨车 root 组合均有兼容响应”的有限 robust recourse envelope；无解则拒绝。

### 7.8 选择与执行

- 候选假设仍是 V39 冻结 conflict-window schedule family；V42 不新增 action grid。
- 排序仍用 V39 的 deterministic fallback key。
- extension 只允许选择与普通 COWP base first target 不同的动作，避免把 no-op 计为 mechanism intervention。
- actual execution override、Waymax kinematic check 和 diagnostics 与既有路径一致。

### 7.9 复杂度

设 blocker 数 `B`、每车 retained roots `R`、每 root profiles `Q`、非 blocker 环境 actors `E`：

- profile 过滤约为 `O(B R Q E H)`；
- exact CSP 最坏为 `O(Q^(B R))`。

但所有量均受已有 frozen bank/critical-agent limits 限制，且 V42 只在 V39 空的 fallback steps 执行。代码记录 compatibility checks、rejects 和 backtracks，若运行成本过高，应首先用结果证明瓶颈，再设计有证书的 branch-and-bound；不能事后改成不完整 top-k 而不说明证书弱化。

---

## 8. V42 预注册判定与分支

### 8.1 Outcome Gate（完全不变）

V42 的 Stage-1 promotion 仍只由六项冻结 outcome Gate 决定：

```text
old RVR rescues retained                  >= 5/10
old RVR induced collisions avoided       >= 7/9
net COWP collision failures removed      >= 3
kinematics net regression                 <= 1 scene
paired mean EP delta vs COWP              >= -0.05
nonzero action-changing intervention      true
```

达到 5/10 意味着：在不丢失 V39 已保留的三个 rescue 的理想情况下，V42 至少还需恢复七个缺失 rescue 中的两个。

### 8.2 Mechanism attribution 条件（不是 promotion Gate）

必须同时读取：

- `interaction_attempt_steps`；
- `interaction_selected_certificate_steps`；
- support agent readiness；
- retained roots / eligible profiles；
- unsupported blocker rejects；
- ego/root/environment rejects；
- joint compatibility rejects/backtracks；
- selected minimum root mass / maximum burden；
- V39 nested selection count。

解释分支：

| 结果 | 结论 | 下一动作 |
|---|---|---|
| Gate pass 且 interaction selection > 0 | V42 extension 有初步因果归因资格 | 运行 fresh37，再做 exact200 |
| Gate pass 但 interaction selection = 0 | outcome 只能归因于 nested V39，不能宣称 RC-IARE 成功 | 检查是否运行/是否真正介入 |
| selection > 0 但 Gate fail | interaction support 被实际采用但没有解决主 Gate | archive；按 reject decomposition 选择全新分支，不在 CF48 调阈值 |
| attempts > 0、selection = 0、ready≈0 | natural root online support 不可用/索引错位 | 修语义或训练证据，非 action search |
| ready>0 但 profile/root rejects 主导 | analytic response bank 与 learned roots 不匹配 | 下一步考虑经独立数据监督的 root-conditioned viability/response model |
| environment/joint rejects 主导 | 独立 per-agent recourse 不可联合实现 | 研究 joint response factorization/scene-level root coupling |
| residual physical rejects 主导 | ego proposal 本身仍不成立 | proposal family 是瓶颈，但不得回到密集首拍区间；需结构化 interaction-conditioned ego proposal |
| `fccd...` 仍 certificate→collision | Type B 独立存在 | 后续 multi-step uncertainty/closed-loop invariance 分支 |

### 8.3 Fail-closed 执行

`analyze_counterfactual48` 若 Gate 缺失或失败，launcher 退出码为 4。不得手动绕过后运行 fresh37。

---

## 9. 后续禁止方向

结合 V28–V41 的历史证据与 changelog，以下方向应明确禁止，除非出现与当前证据矛盾的新独立数据：

1. 再加 release schedule、switch time、horizon stack、dwell/hysteresis patch。
2. 再加 first-action grid、interval fractions、插值阶数、secant/quadratic proposal 或更密边界搜索。
3. 放松 full physical certificate、shift closure、8 s horizon、roadgraph 或 kinematic limits 来提高召回。
4. 调 scalar risk/progress/kinematics/profile 权重来“修” hard support 缺失。
5. 改 selector/frontier 以掩盖 candidate/support 不足。
6. 在 counterfactual48 上事后调 `p_min`、floor、0.75 mass、root count、dedup、β、RCOT/BCOT 阈值或 response profiles。
7. 重建 compact-5k、扩 map/Frenet 或改 split；本轮数据性质只作为证据边界，不是 V42 的一因子变量。
8. 激活未验证的 dense response trajectory head，或直接上黑盒 learned viability head。
9. 把多个单车可行 response 当成天然联合可行，省略 environment/joint compatibility。
10. 用 logged future、未来 Waymax state 或场景 ID 生成 response；这会造成信息泄漏。
11. 在 V42 同时修 accepted-path kinematics 或 Type-B long-horizon mismatch。
12. 把 logged replay 的 collision 改善写成 causal burden 改善。
13. 将 counterfactual48 当作论文最终 holdout，或只报单 seed 点估计。

---

## 10. CCF-A 级论文主线建议

V42 值得继续的原因不是它“更复杂”，而是它把论文最重要的两个概念连接起来：

> **hard non-coercive feasibility** 与 **proposal/support sufficiency**。

一个更有研究价值的论文主命题可表述为：

> Collision-free ego feasibility under fixed surrounding trajectories is neither necessary nor sufficient for safe interactive planning. A candidate should be accepted only when it preserves a high-mass set of root-consistent, low-burden and jointly realizable recourse responses under control and replanning shift closure.

若 V42 通过 Stage-1，其论文贡献可以升级为：

1. false-safe / coercion 是 hard feasibility defect；
2. root-conditioned option preservation 定义周车自然语义；
3. controller- and shift-closed ego recourse 保证执行一致性；
4. **joint interaction-aware reachable-response envelope** 解决 fixed-future false negatives；
5. proposal sufficiency 与 certificate soundness 分离评估；
6. 以 explicit witness / blocker-root-response assignment 提供可解释性。

要达到 CCF-A 级证据，仍需：

- 新的真正 holdout，不再用于 branch selection；
- 至少三独立 seeds；
- 场景级 paired confidence intervals / bootstrap；
- reactive-agent protocol，验证认证 response 在交互闭环中的可实现性和 burden transfer；
- human-audited false-safe stress set；
- response support calibration、coverage/uncertainty 或 conformal guarantee；
- 与同 candidate bank 的 controlled ablations，以及可匹配输入假设的外部 planner baselines；
- runtime/complexity 与 failure taxonomy；
- 对 Type-B certificate mismatch 单独给出 limitation 或后续理论修复。

如果 V42 未通过，仍可形成有价值的负证据：它将精确指出 failure 位于 natural support、response bank、joint compatibility、ego proposal 还是 multi-step closed-loop mismatch，而不是再次停留在“碰撞率没有改善”的不可解释结论。

---

## 11. 数据集性质与本轮使用方式

冻结数据包：`formal_v16_8_24_compact_full_5k`。

| split | scenes | critical agents | mechanism unauditable | protected-prio root coverage | rootless | <2 low-burden roots | pair-neutral unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 5000 | 25,828 | 4.07% | 99.45% | 0 | 0 | 14.89% |
| val | 1000 | 5,153 | 4.34% | 99.36% | 0 | 0 | 14.92% |
| heldout_test | 1200 | 6,120 | 4.46% | 99.47% | 0 | 0 | 15.05% |

数据支持 “root basis 基本不空” 的判断，但仍有两个限制：

- 约 4% critical agents 在当前机制下 unauditable；
- `verify_cache_train.json` 因 58,243 个 irrelevant-pair blockers 标记 `pass=false`，尽管 5000/5000 可读、silent blockers=0、affected/conflict/retained/canonical mismatch 均为 0。

因此本轮不重建数据；V42 通过在线 natural decoder 与冻结 canonical semantics 验证 interaction support。若后续 support readiness 极低，应先判断 online indexing/model calibration，而不是立即重做数据集。

---

## 12. 已落地代码与验证

### 12.1 主要文件

- `cowp/waymax_eval/policy_wrapper.py`
  - RC-IARE support/certificate/constructor；
  - exact V39 nesting；
  - canonical root helper 复用；
  - non-blocker environment 与 multi-blocker CSP。
- `cowp/waymax_eval/metrics_cowp.py`
  - 场景级 interaction mechanism 聚合。
- `cowp/waymax_eval/rollout.py`
  - 新方法注册与 rollout metadata。
- `cowp/scripts/04_eval_closed_loop.py`
  - CLI 方法注册。
- `cowp/scripts/95_analyze_interaction_aware_reachable_response_envelope.py`
  - 原六项 Gate + 新机制诊断。
- `tests/test_v16_8_42_interaction_aware_reachable_response_envelope.py`
  - causal、nested、canonical、environment、joint、aggregation tests。
- `NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh`
  - fail-closed 分阶段运行。
- `ALGORITHM_CHANGELOG.md`
  - 已同步 V42 设计、冻结项、禁止项和命令。

### 12.2 本地代码验证

```text
V42 + V39/V41 focused/regression tests     96/96 PASS
V42 blocking code audit                    48/48 PASS
canonical root randomized exact match     250/250 PASS
frozen manifest checks                      4/4 PASS
analyzer replay of actual V41 outcome           PASS
V42 full Waymax rollout                     NOT RUN
```

代码审计产物：`V16_8_42_CODE_VALIDATION.json`。

---

## 13. 下一步执行命令

在 V42 代码目录中严格按顺序执行：

```bash
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh make_ids
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_counterfactual48
```

然后停止并检查：

```text
preregistered_gate.interaction_aware_reachable_response_envelope.pass
```

只有为 `true` 才继续：

```bash
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh fresh37_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_fresh37
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh confirm200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_42_INTERACTION_AWARE_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_confirm200
```

若 Gate fail，脚本应退出 4；不要手动运行 fresh37。下一轮分析应先重复 V41 的可靠性顺序：manifest/shard/merge/analyzer/执行语义通过后，再读取 outcome 和 mechanism diagnostics。
