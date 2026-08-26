# V16.8.30 实验可靠性、机制归因与 V16.8.31 设计

## 0. 最终结论先行

V16.8.30 当前上传结果 **通过可靠性审计，可以做算法归因**。没有发现会要求停止算法分析、改为 repair-only 的工程阻断。

唯一发现的是一个 **non-blocking provenance/reporting bug**：V16.8.30 随代码保存的 V16.8.29 subset reference JSON，`scenario_results` 确实是正确的 16/48/96 子集，但顶层 `scenario_ids_sha256` 没有从 exact200 hash 改写为 subset hash。V16.8.30 analyzer 实际按每条 row 的 `scenario_id` 做 paired comparison，因此它不会改变任何本轮指标或 rescue/induced attribution。V16.8.31 已把 subset hash 与 provenance 全部修正，并加入 regression test。

V16.8.30 的算法判断是：

- `cowp_rvr_pareto_guard`：**clean negative，归档，不 promotion**；
- `cowp_successor_option_viability`（SOV）：**positive mechanism evidence，但当前 implementation 不 promotion**；
- 可吸收到论文主机制的不是“one-step lookahead”这个实现，而是 **action-conditioned future option-set preservation** 这一物理 feasibility 轴；
- 当前 dominant bottleneck 应进一步收紧为 **uncertified recovery 下的 temporal physical option preservation / recovery commitment problem**；
- 当前最不应该做的事是调 prefix weight、缩短 8 s horizon、重调 RCOT/BCOT/outcome、或立刻扩 proposal；
- V16.8.31 的最有信息量分支是 **Bi-Horizon Option Viability (BHOV)** + 一个 restoration-only diagnostic。

---

# 1. V16.8.30 结果可靠性审计

## 1.1 Equivalence16 通过

V16.8.30 的第一层 preregistered gate 是：先证明新增 successor-option 代码没有伤害普通 COWP common path。

实际结果：

- manifest：16 个唯一 scene；
- 两个 shard：8 + 8；
- shard overlap = 0；
- merged scene set 精确等于 manifest；
- 与 bundled V16.8.29 COWP reference 比较：**1120 fields / 0 mismatches**；
- tolerance = `1e-7`。

所以 V16.8.30 common COWP path 没有因为新 successor helper 或 method plumbing 被改变。

## 1.2 Counterfactual48 paired protocol 通过

两个新分支均满足：

- manifest = 48 unique scenes；
- shard = 24 + 24；
- shard overlap = 0；
- shard union = exact manifest；
- merged rows = 48；
- guard 与 SOV scene set 完全一致；
- merged `standard_metric_summary` 从 per-scenario rows 独立重算最大误差 = **0**。

因此 rescue/induced transition 是严格 paired 的。

## 1.3 Active execution semantics 没有发现新污染

代码检查确认：

- V16.8.27 的 conventional-safety bypass 没有复发；
- V16.8.28 的 no-valid PAD execution bug 没有复发；
- SOV/Pareto 只在 `full conventional set == empty && valid exists` 中介入；
- certified path / conventional-safe fallback 与 COWP 保持一致；
- successor ego state 使用的是 **实际经过 acceleration/jerk/yaw-rate projection 后的 emitted one-step target**，而不是 raw nominal waypoint；
- surrounding agents 的 counterfactual successor 使用当前 causal CV contract，而不是 silent logged-future oracle。

因此 V16.8.30 的 mechanism attribution 可以成立。

## 1.4 唯一 non-blocking 问题：subset reference hash stale

V16.8.30 中 V16.8.29 exact200 被过滤为 16/48/96 reference subset 后：

- `scenario_results` rows：正确；
- row IDs 与各 subset manifest：完全一致；
- 但顶层 `scenario_ids_sha256`：仍写 exact200 的 `3fb2...`。

因为 analyzer 不使用这个 stale hash 做 pairing，而是显式比较 row ID set，所以本轮结果不受影响。

V16.8.31 修复为：

- subset JSON 顶层 hash = subset manifest 的真实 logical SHA256；
- 保留 `source_scenario_ids_sha256`；
- 新增 `reference_subset_provenance`；
- regression 验证 row order、hash、manifest 三者一致。

**可靠性结论：PASS。**

---

# 2. 按 V16.8.30 上一轮 preregistered condition 判断结果

上一轮对 48-scene counterfactual panel 的判断条件本质上是：

1. Pareto guard 如果能避免 RVR induced failures 且保留 rescues，则研究 recovery dominance/frontier；
2. SOV 如果明显优于 COWP/guard，则 successor-option preservation 有资格继续；
3. 如果 successor candidate 也普遍无 support / 两支都无效，再转 structured proposal refinement；
4. 48 panel 是 mechanism-selected，只能作为机制 gate，不能作为 paper/promotion result。

当前结果给出的分支非常清楚。

---

# 3. Pareto guard：明确失败，但这个失败很有信息量

## 3.1 结果

`cowp_rvr_pareto_guard` 在 48 scenes 上：

- `recovery_switch_step_rate = 0.0`；
- CR / Collision / Offroad / Kinematics / EP 全部与 COWP 完全一致；
- 0/10 old RVR rescues retained；
- 9/9 old RVR induced collisions avoided。

它不是“安全但略保守”，而是 **完全不发生 algorithmic intervention**。

## 3.2 为什么 0 switch 很关键

RVR alternative 在 zero-conventional states 的平均 prefix gain ≈ **+2.64 steps**，但相对 COWP fallback 同时：

- action risk ≈ `+0.0820`；
- rule risk ≈ `+0.1376`；
- pressure risk ≈ `+0.0024`。

所以“longer prefix”与已有 fallback risk signals 不是天然一致的。

严格要求：

`prefix better AND transport/rule/action/pressure all non-worse`

会把 recovery action set 全部清空。

这直接排除了下一步两个看似容易但实际错误的方向：

- 不应把 Pareto guard 换成“加一点 tolerance”然后继续调；
- 不应把 prefix/rule/action/pressure 全混成一个 weighted scalar 再 grid search。

那会重新回到 paper 主线一直避免的 scalar soup。

**结论：Pareto no-regret 是 clean negative ablation。**

---

# 4. Successor Option Viability：机制有效，但 implementation 还没成熟

## 4.1 总体表现

针对 48 scenes：

| Method | Collision | Kinematics | EP | switch rate |
|---|---:|---:|---:|---:|
| COWP | 34/48 = 70.83% | 6/48 = 12.5% | 1.00251 | — |
| RVR | 33/48 = 68.75% | 9/48 = 18.75% | 0.82362 | large/unconditional zero-conv intervention |
| Pareto | 34/48 | 6/48 | 1.00251 | 0% |
| SOV | **33/48 = 68.75%** | **6/48 = 12.5%** | **0.99863** | **1.93%** |

SOV 相对 COWP：

- collision rescue = **2**；
- collision induced = **1**；
- shared collision = 32；
- McNemar exact `p=1.0`；
- EP paired delta = **-0.00388**；
- bootstrap 95% CI ≈ **[-0.0164, +0.00925]**；
- Offroad 无变化；
- Kinematics 净变化 0。

所以不能说“算法已经提高 collision”。统计上远远不够。

## 4.2 但 counterexample retention 很强

相对 V16.8.29 已知的 10 rescue / 9 induced：

- 保留 **2/10** RVR rescues；
- 避免 **8/9** RVR-induced collision。

也就是说：

> 当前 successor-option statistic 对“什么时候不应该跟 RVR”有很强的判别力，但对“什么时候值得跟 RVR”召回不足。

这是典型的 **high precision / low recall gate**。

## 4.3 为什么这个结果值得吸收进主机制

它只在约 **14.35%** policy steps 做 successor probe，真正 switch 只有 **1.93%**，却几乎把 V16.8.29 的 induced-collision 反例过滤掉。

这说明：

> `current prefix` 没包含的关键信息，确实存在于 `actual action -> successor state -> future option set` 中。

因此值得吸收的是 **physical option-set preservation** 这个机制对象，而不是当前 strict `rvr_sig > base_sig` 的代码形式。

## 4.4 为什么当前 SOV 还不能 promotion

有四个原因：

1. 48 scenes 是 result-selected counterexample panel；
2. collision 净收益仅 1 scene，`p=1.0`；
3. 只保留 2/10 已知 RVR rescues，明显过保守；
4. 仍有 1 个 induced collision `7721ff4800156886`。

这个 induced scene 尤其重要：它最后在较晚 step 才 collision，并且 SOV 闭环最终出现更高 zero-conventional exposure、更低 conventional-candidate support。说明“一步 successor signature 严格变好”仍不等价于长期闭环 viability。

**结论：SOV implementation 不 promotion；successor option preservation hypothesis promotion 到下一轮。**

---

# 5. V16.8.25 → V16.8.30 完整证据链

## 5.1 Social feasibility 主干已经相对成熟

历史 learned-offline：

- RCOT LowSafeExist AUPRC ≈ 0.897；
- BCOT priority/global false-safe AUPRC ≈ 0.837 / 0.928；
- generic candidate false-safe AUPRC ≈ 0.354。

因此主要问题不是“RCOT/BCOT 没学会”。

## 5.2 CTU 负结果保护了 post-certificate set preservation

V16.8.25 CTU：certificate invariant，但移除原 frontier、直接 certificate → planner-score argmin 后：

- learned-offline EP / PBTR / NCF recall 变差；
- strict Waymax EP 也变差。

因此 BCOT/option-transport 不仅含 threshold 信息，还含 feasible-set 内的 robustness ordering 信息。

这使：

`protected-priority certificate -> certificate-compatible set-preservation frontier`

应继续 freeze。

## 5.3 Outcome fallback 已是 clean negative

V16.8.28 修复完工程污染后：

- outcome fallback physical gain 不可信/几乎为零；
- EP 有明确小幅 regression；
- low-FPR recall 也不足以支持 hard shield。

因此 outcome head 保留 diagnostic，不能继续调 fallback weight。

## 5.4 V16.8.28 clean physical attribution 把 collision 定位到 zero-conventional

COWP first collision：

- 32/34 来自 `no_conventional_use_least_coercive_valid`；
- 0/34 来自 conventional-safe fallback；
- 0/34 来自 accepted priority-NCF。

Conventional baseline 也出现相同结构。

所以 collision 不是 RCOT/BCOT certificate 把 planner 搞坏，而是 shared online physical-feasibility regime 的问题。

## 5.5 V16.8.29 又把 zero-conventional 拆成 collision-side，而不是 map-side

平均 step decomposition：

- zero conventional ≈ 55.69%；
- `collision_empty` ≈ **52.53%**；
- `roadgraph_empty` ≈ **0.11%**；
- both empty ≈ 2.66%；
- intersection empty ≈ 0.39%。

因此当前不应该进入 route/Frenet/roadgraph proposal repair。

## 5.6 RVR 证明 current prefix 有信息，但不是 recursive viability

V16.8.29 exact200：

- 10 collision rescues；
- 9 induced collisions；
- collision 17.0% → 16.5%，无统计意义；
- offroad / kinematics 变差；
- EP 下降。

10 rescues 证明 bank 中存在能把系统带回更好 physical state 的 action；9 induced 证明 greedy max-prefix 会把系统带入后来更差的 support state。

所以 proposal bank 不是当前唯一瓶颈，selection/temporal state transition 仍然有可优化空间。

## 5.7 V16.8.30 第一次证明 successor option support 是独立有效信号

SOV 避免 8/9 known induced，同时几乎不损 EP/kinematics；Pareto 风险交集则完全 0 switch。

因此 current dominant question 从：

> 哪个 candidate 当前更安全？

进一步变成：

> 哪个 recovery action 在当前 survival 与 future feasible-option support 之间形成可靠的 temporal dominance？

---

# 6. 当前 dominant bottleneck：进一步收紧

当前最准确的定义是：

## **Temporal Physical Option Preservation under Uncertified Recovery**

当 full conventional set 已经为空时，planner 必须做 uncertified action。真正难点不是给这个 action 再加一个 risk score，而是：

1. 当前短期 survival 是否改善；
2. actual emitted action 是否使下一次 replanning 保留/恢复足够的 conventional options；
3. 这种改善是否能持续，而不是一拍变好、随后再次 collapse；
4. 何时需要连续几拍 commit 到 recovery，何时应退回 COWP conservative fallback。

SOV 目前解决了一部分第 2 点，但对第 1/3/4 点仍不完整。

---

# 7. 当前各层成熟度

| Layer | 状态 | 证据 / 下一步 |
|---|---|---|
| compact-5k data contract / split | **Freeze** | 当前不重建 |
| Natural roots | **Mature / Freeze** | 不动 natural basis |
| RCOT same-root transport | **Mature / Freeze** | 强 held-out signal，不追小 AUPRC |
| BCOT | **Mature / Freeze** | 强 false-safe signal，不调 budget |
| Protected-priority certificate | **Freeze** | 保持 hard feasibility |
| Set-preservation frontier | **Freeze** | CTU 已否定 planner-score replacement |
| Outcome head | **Diagnostic-only freeze** | fallback outcome clean negative |
| 8 s conventional screen | **Attribution contract / Freeze** | 不缩 horizon |
| Conventional-audit / no-valid execution integrity | **Solved / Freeze** | v27/v28 repair 不再碰 |
| max current prefix RVR | **Negative as policy** | 仅保留为 controlled recovery alternative |
| strict Pareto guard | **Negative diagnostic** | 0 switch，归档 |
| one-step SOV | **Promising mechanism signal** | high precision / low recall，不 promotion implementation |
| temporal recovery viability | **Main active bottleneck** | V31 主攻 |
| proposal support | **Long-term/global ceiling** | 当前先不扩 bank |
| accepted-path kinematics | **Secondary unresolved** | recovery collision 收敛后单独设计 Execution-Viability |

---

# 8. 当前模型已经学会什么、还没学会什么

## 8.1 已经学会的核心

当前 learned mechanism 已经比较可靠地学会：

- natural behavioral roots；
- same-root counterfactual response transport；
- protected-priority relation 下的 non-coercive feasibility；
- BCOT 对 false-safe/coercion risk 的 structured discrimination；
- certificate-compatible feasible-set robustness ordering。

所以不能把当前失败解释为“模型整体没学好”。

## 8.2 当前最缺的 representation

模型没有任何专门 supervision 去学习：

`current online state + actual projected ego action -> future executable option-set survival/restoration`

具体缺：

- successor conventional existence；
- successor macro diversity；
- successor candidate count；
- successor safe-prefix distribution；
- 更重要的，多步 persistence / time-to-option-collapse。

当前 SOV 是用 analytic causal simulator surrogate 在线计算这个 target，而不是 learned head。

**现在仍然不应马上训练 successor head。**

先用 analytic probe 证明哪一种 physical option-set target 真正与 closed-loop rescue/harm 对齐，再把它蒸馏成 learned estimator，才有论文级方法学意义。否则只是在训练一个新的 classifier 来拟合尚未验证的 target。

---

# 9. CCF-A 主线应该怎样收紧

单独的 recursive feasibility、backup-plan MPC、one-step lookahead、safety filter 都不够新。

当前最有潜力的统一论文叙事是：

## **Orthogonal Option-Set Feasibility**

### Social axis

`same-root natural alternatives -> protected-priority RCOT/BCOT -> preserve other agents' low-burden option set`

回答：

> Ego 的 collision-free plan 是否通过压缩他人的 natural safe options 才成立？

### Physical axis

`actual emitted action -> causal successor state -> preserve/restore ego planner's future executable option set`

回答：

> 当 certificate 已空、必须 recovery 时，这个 action 是否把自己的未来可执行选择空间也压垮？

两者都围绕 **option-set feasibility**，但 semantic object 不同，不能揉成一个 scalar cost。

这比“counterfactual + safety head + outcome head + fallback weight”更干净，也更容易形成 CCF-A 级 mechanism story。

V16.8.31 不是最终 contribution，它是在验证 physical axis 的最小结构。

---

# 10. V16.8.31：Bi-Horizon Option Viability (BHOV)

## 10.1 为什么不是直接上 multi-step tree search

V16.8.30 的结果首先暴露了一个更小、更可归因的问题：

SOV 当前要求：

`successor_signature_RVR > successor_signature_COWP`

才 switch。

这可能过严。一个 RVR action 可以：

- 当前 collision-safe prefix 明显更长；
- successor option support 与 COWP **相等**而不是更差；

这种 action 在 SOV 中仍被拒绝。

在 10 个 old RVR rescues 中，SOV 只保留 2 个，因此先验证这个 acceptance relation 比直接做 multi-step/K-way search 更有科研信息量。

## 10.2 BHOV 定义

同一个 zero-conventional state，仍然只比较：

- base = 原 COWP fallback；
- alt = 原 V16.8.29 RVR max-prefix candidate。

得到：

- `H0_base, H0_alt`：current causal safe prefix；
- `V1_base, V1_alt`：actual emitted action 后的 successor option signature。

只有当：

`V1_alt >=lex V1_base`

且

`H0_alt >= H0_base`

且至少一个 strict improvement，才允许 alt。

因此：

- 当前 prefix 变好但 successor 变差：**禁止**；
- successor 变好但 current prefix 变差：**禁止**；
- successor 相同、current prefix 变好：**允许**；
- 两边完全相同：保持 COWP。

它是跨两个时间层的 product partial order，没有新 weight，没有阈值调参。

## 10.3 为什么这个 probe 比调 SOV threshold 更合理

它直接测试一个机制命题：

> physical recovery 的 feasibility 是否应当要求“future option non-regression + current survival improvement”，而不是 future option strictly better。

如果成立，它可以成为未来 physical option-set feasibility 的基础 ordering；如果失败，就证明 one-step successor statistic 仍然太短，需要进入 temporal persistence / multi-horizon viability，而不是继续改 threshold。

---

# 11. 第二分支：Successor Restoration Only

为了判断 rich successor signature 的 lower-order components 是否噪声过大，增加 diagnostic：

只在：

`base successor conventional_exists = 0`

且

`alt successor conventional_exists = 1`

时 switch。

这将给出一个非常干净的分叉：

- restoration-only 好、BHOV 差：binary restoration 稳定，macro/count/prefix richness ordering 需要重构；
- BHOV 好、restoration-only 太保守：rich option-set ordering 确实含额外信息；
- 两者都差：one-step successor 不足，进入 temporal/multi-horizon；
- 两者都常遇到 successor no-support：proposal support 开始升级为直接 bottleneck。

---

# 12. 新实验协议：避免继续针对已知 counterexample 过拟合

## Stage A：equivalence16

重新跑普通 COWP 16 scenes。0 mismatch 才继续。

## Stage B：counterfactual48

先跑 BHOV + restoration-only。

这是 mechanism-selected panel，只用于判断：

- BHOV 能否比 SOV 的 2/10 rescue retention 更高；
- 是否仍保持接近 SOV 的 8/9 induced avoidance；
- 是否出现新的 kinematics/EP regression。

**这里不要直接宣称 promotion。**

## Stage C：panel-disjoint outcome-blind development64

如果 48 favorable，再跑 64。

为了避免我在构造新 64 时再次使用 V16.8.30 outcome：

- 从 exact200 中排除所有 V16.8.30 `equivalence16 ∪ counterfactual48 ∪ balanced96` scenes；
- union exclusion = 99 IDs；
- 剩余 101 IDs；
- 使用固定 salt `v16.8.31_outcome_blind_holdout64_disjoint` 对 scene ID 做 SHA256 rank；
- 取前 64；
- 与所有 V16.8.30 development panels overlap = **0**；
- logical hash = `becdc8430e14bd76190e3446206bed8e7cb9afb966290978e9bdaa61a5202e79`。

它仍然不是论文 holdout，因为 enclosing exact200 已被多轮看过；这里只叫 outcome-blind development panel。

## Stage D：exact200 development confirmation

只有 64 非伤害且 favorable，才跑 exact200。默认只跑 BHOV，复用 immutable COWP/RVR exact200 reference，不重复浪费 baseline 时间。

算法最终 freeze 后，必须新建 never-used final evaluation set。

---

# 13. 下一步最该解决的问题，而不是其它问题

当前 priority：

**P0：验证 two-horizon physical option dominance 是否能解决 SOV low recall，同时保持 induced-harm suppression。**

只有在 P0 有效后，才考虑：

- 把 analytic successor-option target formalize；
- 更进一步做 multi-horizon option persistence；
- 或学习一个 action-conditioned successor viability estimator，减少在线 proposal regeneration 开销。

若 P0 无效，则应停止 BHOV/SOV tie-breaking 变体，转向：

- multi-step temporal viability / option-set persistence；或
- 如果 successor support 本身系统性为空，再转 structured proposal refinement。

当前不应：

- 改 RCOT/BCOT；
- 改 8 s conventional definition；
- 调 outcome；
- 新增 proposal primitive；
- 同时修 accepted-path kinematics。

---

# 14. 回归与工程状态

V16.8.31 最终 package：

- focused semantic/integrity sanity：**29 passed**；
- V16.8.31 helper/provenance tests：**5 passed**；
- manifest hashes 全部通过；
- holdout64 已验证与全部 V16.8.30 development panels overlap = 0；
- conventional bypass grep = clean。

Full repository 尝试运行，但在命令窗口内没有完成；第一处观测到的 failure 是已有历史问题：缺失旧 launcher `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`。在该 failure 前为 **124 passed / 5 skipped**，没有看到 V16.8.31 新功能 failure。

---

# 15. 建议下一步命令

```bash
cd COWP_v16_8_31_BIHORIZON_OPTION_VIABILITY

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/你的旧COWP目录/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh make_ids

# TFExample index 缺失时才执行
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh base_equivalence16_parallel2

bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh analyze_counterfactual48
```

**到这里停止。**

若 BHOV 在 48 上明显提高 old rescue retention，同时没有明显放回 RVR-induced failures，再执行：

```bash
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh holdout64_parallel2
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh analyze_holdout64
```

只有 outcome-blind development64 也 favorable，才进入：

```bash
PROMOTED_METHODS=cowp_bihorizon_option_viability \
  bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh confirm200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_31_BIHORIZON_OPTION_VIABILITY_CN.sh analyze_confirm200
```

exact200 仍然只能作为 development confirmation，不是最终论文 test set。
