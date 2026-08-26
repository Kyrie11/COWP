# COWP V16.8.29 结果审计与 V16.8.30 Successor Option Viability 设计

## 0. 结论先行

本轮对论文、`大模型建议.md`、`formal_v16_8_24_compact_full_5k`、V16.8.29 代码与 `v16_8_29_recursive_viability` 结果进行了交叉审计。

**结论一：V16.8.29 结果正确、内部一致，可以做算法归因；没有发现需要停止算法分析的工程阻断。**

**结论二：V16.8.29 的 unconditional RVR 不应 promotion。** exact-200 上它确实 rescue 了 10 个 COWP collision 场景，但同时诱发 9 个原本不 collision 的场景，Collision 仅从 17.0% 到 16.5%，McNemar exact p=1.0；同时 Offroad 3.0%→3.5%、Kinematics 12.5%→14.5%、EP 1.0461→1.0047。因此不能通过调 prefix weight、缩短 conventional horizon 或继续堆 fallback penalty 来“救”RVR。

**结论三：RVR 的实验不是无效实验。它给出了比“RVR 成/败”更关键的机制证据：当前 collision-safe prefix 有真实信息，但不是 closed-loop recoverability 的充分统计量。** 10 个 rescue 证明相同 proposal bank 中确实存在能把系统带回更富可行选项区域的 recovery action；9 个 induced collision 又证明贪心最大化当前 open-loop survival 会把另一些场景推入未来更严重的 support collapse。

**结论四：下一版最应该学习/显式估计的是 action-conditioned successor feasible-option preservation。** 即在实际 jerk/yaw-rate-limited one-step action 执行后，下一个 replanning state 是否仍保留 conventional option、多少不同 macro、多少 conventional candidates、以及可恢复的 collision-safe margin，而不是继续评价当前 candidate 自己“还能撑多久”。

因此 V16.8.30 落地两个可归因分支：

1. `cowp_rvr_pareto_guard`：diagnostic-only，用来检验 V16.8.29 的 induced failure 是否主要来自 prefix lexicography 覆盖了已有 transport/rule/action/pressure 风险证据；
2. `cowp_successor_option_viability`：主分支，用实际将要发给 Waymax 的 one-step target 构造 causal successor state，并比较 COWP fallback 与 RVR fallback 在下一 replanning state 的 feasible option set，只有 successor option signature 严格占优时才切到 RVR action。

所有 certified / conventional-safe 路径均保持 COWP 不变。

---

## 1. 与论文研究主线的对齐

论文真正的研究对象不是“把 courtesy/social cost 加得更强”，而是 **false-safe planning / safety-by-coercion**：ego rollout 虽 collision-free，但其成立依赖其他 road user 通过高 burden braking/yield/gap surrender 来吸收冲突。COWP 用 natural alternatives、same-root counterfactual response transport、protected-priority non-coercive feasibility 与 coercion witness，把这种现象从 soft preference 提升到 feasibility defect。

这决定了当前算法演进必须遵循两条原则：

- social non-coercive feasibility 与 physical execution viability 不应重新揉成一个任意 scalar cost；
- proposal sufficiency、certificate quality、post-certificate selection、uncertified recovery、execution consistency 必须分层归因。

所以 V16.8.30 不动已经有强证据的 RCOT/BCOT/certificate/frontier，而只研究 V16.8.29 暴露出来的 zero-conventional physical recovery state transition。

---

## 2. 数据集理解与本轮是否需要重建

`formal_v16_8_24_compact_full_5k` 的 contract 稳定：

- train 5000 / val 1000 / held-out 1200；
- held-out 是 official WOMD validation 的独立子集，因为 official WOMD test future 不可见；
- train/val/held-out profile 均无 malformed row；
- critical-agent 数量约 5.34–5.39/scene，p50/p90 均为 6；
- natural mechanism unauditable rate：train 4.07%、val 4.34%、held-out 4.46%；
- protected-priority root coverage：train 99.45%、val 99.36%、held-out 99.47%；
- auditable critical agent 中 rootless=0；
- proposal source acceptance 在 split 间总体稳定，例如 ROBUST_BCTE 约 98%，JOINT_ROUTE_NCF 约 94–98%，TERMINAL 约 54–55%，PRIORITY_SMOOTH_YIELD 约 20–22%。

这说明本轮没有证据指向 dataset construction drift 或 natural-root coverage 崩溃。**当前不重建数据是正确的。** proposal support 仍是长期 global ceiling，但 V16.8.29 已证明相同 bank 内 action choice 能显著改变后续 feasible support，因此现在还不能把所有 collision 简化成“bank 里没 proposal”。

---

## 3. V16.8.29 完整性 / 工程可靠性审计

### 3.1 exact-ID 与 shard

- exact manifest：200 unique IDs；
- logical SHA256：`3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`；
- COWP / RVR 均为 2 shard × 100 scene；
- shard 互不重叠，union 精确等于 manifest；
- dev64 是 exact200 真子集。

### 3.2 dev64 与 confirm200 可重复性

我逐场景递归比较了 dev64 与 confirm200 中重叠的 64 个 scene row：

- COWP：0 mismatch；
- RVR：0 mismatch；
- tolerance 1e-9。

此外随包的 V16.8.29 COWP-vs-V16.8.28 equivalence gate：64 scenes、3384 fields，0 mismatch，passed=true。

所以 dev64/confirm200 差异不是 GPU、parallel2、merge、随机数或 evaluator non-determinism。

### 3.3 active semantics

代码审计确认：

- RVR 只在 `full conventional set == empty && valid candidate exists` 介入；
- full conventional definition 没有被改短，仍由 roadgraph screen ∩ full causal CV collision screen 构成；
- RVR candidate 不被 relabel 为 conventional-safe/NCF；
- certified path、RCOT、BCOT、protected-priority gate、set-preservation frontier 不变；
- V16.8.27 conventional bypass repair 仍在；
- V16.8.28 zero-valid PAD execution repair 仍在，no-valid 使用 current-state bounded smooth stop。

未发现新的 reporting/selection 语义污染。

### 3.4 一个非阻断的 packaging issue

结果 zip 没有带上 `confirm_exact200_physical_compare.json` 和 `confirm_exact200_mechanism_summary.json`，但 merged exact200 outputs 完整；我用随代码的 analyzer 可重复生成了二者。因此这是打包遗漏，不是 experiment integrity blocker。

机器审计：`V16_8_29_RESULT_INTEGRITY_AND_MECHANISM_AUDIT.json`。

---

## 4. 按 V16.8.28 的预注册顺序判断 V16.8.29

V16.8.28 预注册的关键 gate 是：

- `collision_empty` 大量且 RVR 有效 → 才考虑升级 physical recursive viability；
- `collision_empty` 大量但 prefix 很短 → proposal support；
- `roadgraph_empty` → map/Frenet proposal；
- `intersection_empty` → geometry/dynamics structured proposal refinement；
- **RVR 提高 prefix 但 collision 不降 → prefix 不是 sufficient viability statistic，停止该方向。**

V16.8.29 精确落在最后一个分支。

### 4.1 exact200 主结果

| Metric | COWP | RVR | 结论 |
|---|---:|---:|---|
| CR | 19.5% | 19.5% | 无净收益 |
| Collision | 17.0% (34) | 16.5% (33) | 10 rescue / 9 induced，p=1.0 |
| Offroad | 3.0% | 3.5% | 变差 |
| Kinematics | 12.5% | 14.5% | 变差 |
| EP | 1.04607 | 1.00470 | Δ=-0.04137，95% CI 约 [-0.0984, 0.0061] |
| Fallback step rate | 71.68% | 72.55% | 略高 |

**RVR promotion = fail。**

### 4.2 dev64 为什么看起来很成功

dev64 是由 34 个旧 COWP collision scene + 30 个高 zero-conventional non-collision scene 组成。它的设计偏向“能否 rescue 已知 failure”。

- dev64：RVR rescue 10，induce 0；
- confirm200：同 10 个 rescue 全部精确复现；
- 但其余 136 个 scene 中出现 9 个新的 RVR-induced collision。

所以 dev64 不是结果错，而是 **counterfactual harm coverage 不足**。以后 development set 必须同时包含：rescued、induced、shared-failure、stable-safe controls。

---

## 5. zero-conventional 根因已经进一步收紧

COWP exact200 平均：

- zero-conventional step rate = **55.69%**；
- `collision_empty` = **52.53%**；
- `roadgraph_empty` = **0.11%**；
- `road_and_collision_empty` = **2.66%**；
- `intersection_empty` = **0.39%**。

所以 collision-side support collapse 几乎完全是 dynamic collision screen 侧，而不是 map/roadgraph。

这意味着现在不应该：

- 做 route/Frenet/map repair；
- 扩大 roadgraph heuristic；
- 缩短 8s conventional horizon 来“制造”safe candidate；
- 把 roadgraph candidate density 当主矛盾。

真正的问题是：**一次 uncertified recovery action 会把 closed-loop state 带向更容易还是更难重新获得 conventional option 的区域。**

---

## 6. 10 个 rescue 与 9 个 induced collision 告诉了什么

### 6.1 10 个 rescue：RVR 的 signal 是真的

RVR vs COWP：

- zero-conventional exposure：-14.1 pp；
- conventional candidates：+2.12；
- max safe prefix：+16.33 steps；
- selected prefix：+20.03 steps；
- fallback：-8.0 pp；
- 但 EP：**-0.7735**。

这些 scene 证明：

1. fixed bank 并非处处无解；
2. 当前 fallback action 会影响以后 proposal/certificate availability；
3. 某些长-prefix action 确实能把系统送回更可恢复区域；
4. 但当前策略可能通过非常保守的 state transition 买到安全，因此 utility/progress 不可忽略。

### 6.2 9 个 induced collision：为什么 current prefix 不够

RVR vs COWP：

- zero-conventional：+22.5 pp；
- conventional candidates：-4.15；
- max prefix：-16.11 steps；
- selected prefix：-14.32 steps；
- fallback：+13.3 pp；
- action risk：+0.0715；
- rule risk：+0.1278；
- first RVR collision median step = 61。

first-event macro 主要是 `STOP_BEFORE_CONFLICT` (5/9) 与 `YIELD` (2/9)，并非单纯“太激进”。

这说明 RVR 的错误不是一个简单的 aggressive/conservative 参数问题：**当前时刻看起来更长的 open-loop collision-free prefix，可能通过改变 gap、相对速度、冲突到达顺序，把未来的 candidate bank 压缩得更严重。**

### 6.3 关键算法结论

V16.8.29 所谓 Recursive Viability 实际只测量：

> current candidate 在当前 CV obstacle rollout 中能持续多少步不撞。

真正的 recursive viability 至少需要回答：

> 执行实际 action 后，下一 replanning state 是否仍有一组可行 continuation / backup options？

因此当前模型最应该补的不是一个更强 candidate classifier，而是 **action-conditioned successor option preservation**。

---

## 7. 当前各层成熟度与 freeze 策略

| Layer | 状态 | V16.8.30 策略 |
|---|---|---|
| compact-5k data contract | 稳定 | Freeze，不重建 |
| natural roots / natural basis | 成熟 | Freeze |
| RCOT same-root transport | 强信号 | Freeze，不追小 AUPRC |
| BCOT structured witness/certificate | 强信号 | Freeze，不调 budget |
| protected-priority hard feasibility | 有一致证据 | Freeze |
| certificate-compatible set-preservation frontier | CTU 负消融支持 | Freeze |
| learned outcome head | 有诊断信号但 hard-safety 不够 | diagnostic-only freeze |
| full conventional definition | 当前归因基准 | Freeze，不缩 horizon |
| execution-integrity / no-valid stop | 已修复 | Freeze |
| zero-conventional recovery transition | **未解决/当前主瓶颈** | V16.8.30 主攻 |
| accepted-path kinematics | **secondary bottleneck** | 暂时隔离 |
| proposal support | 长期 ceiling | 先做 successor-support discriminator |
| final unseen evaluation | 尚未完成 | 算法 freeze 后另建 |

历史证据仍支持保护前半部分：held-out RCOT LowSafeExist 约 0.897；BCOT priority/global false-safe AUPRC 约 0.837/0.928；generic candidate false-safe classifier 仅约 0.354；CTU 已证明 certificate→planner argmin 会显著伤害 EP/PBTR/NCF recall。

---

## 8. 当前模型最应该学习但还没学到的内容

### 8.1 Successor feasible-set preservation

现有网络和 selector 都没有直接表示：

`state + actually_emitted_action -> next-state option-set richness / recoverability`。

这是当前最明显的 representation gap。

### 8.2 Execution-conditioned viability

COWP/RCOT/BCOT 操作的是候选 trajectory；Waymax 真正执行的是经过 acceleration/jerk/yaw-rate clipping 的 one-step action。一个物理 viability 模块应该以 **emitted action** 为条件，而不是 nominal waypoint 为条件。

### 8.3 Recovery 的状态价值，而不是当前轨迹风险

当前 action/rule/outcome/prefix 多数是“这个 candidate 自己看起来有多危险”；下一步真正需要的是“这个 action 会不会把系统带到仍有多种安全退出方式的状态”。这更接近 control-relevant state viability / option preservation。

### 8.4 暂时不要让模型学的东西

- 不要再训练 generic safety classifier 替代 RCOT/BCOT；
- 不要把 outcome head 升格 hard shield；
- 不要为了 online collision 重新训练 natural root；
- 不要把 kinematics 与 recovery 同轮混合训练，否则再次失去归因。

---

## 9. V16.8.30 新算法：Successor Option Viability Recovery

### 9.1 设计边界

只有在：

`full conventional set == empty && valid candidate exists`

时介入。

存在 certificate 或 conventional candidate 时，行为保持原 COWP。

### 9.2 两个 counterfactual recovery candidate

- `base`: 原 COWP least-coercive-valid fallback；
- `rvr`: V16.8.29 roadgraph-first + max collision-safe-prefix candidate。

V16.8.30 不对 K 个候选全部做 successor rollout，只比较这两个机制上有明确解释的 counterfactual，以控制运行时间和归因维度。

### 9.3 构造 action-conditioned successor

对于 base/RVR 各自：

1. 使用当前已有 one-step action projection 产生的 **实际 emitted target**；
2. ego 进入该 target；
3. 其他 valid agents 用与 conventional collision screen 一致的 one-step CV causal propagation；
4. 在 successor state 重新调用**完全相同的 online physical proposal generator**；
5. 不调用 RCOT/BCOT，不把该 surrogate 当 formal certificate，只测 option support。

### 9.4 parameter-free successor signature

对 successor proposal bank 定义 lexicographic signature：

1. 是否存在至少一个 full conventional option；
2. conventional macro type 数量；
3. conventional candidate 数量；
4. drivable-valid candidates 中最大 collision-safe prefix。

仅当 RVR successor signature **严格优于** base successor signature 时切换到 RVR；tie 保留 COWP base。

这使 V16.8.30 测的不是“当前谁更安全”，而是“哪个 action 更能保留下一个时刻的退出选择”。

### 9.5 为什么这是更合理的论文方向

单纯 predictive safety filter、backup trajectory、recursive feasibility、trajectory repair 都已有成熟工作，因此“做一步 lookahead/backup”本身不能作为 CCF-A novelty。

真正值得继续验证、并有可能形成论文级统一机制的是：

**Orthogonal Option-Set Feasibility**

- Social axis：protected-priority same-root non-coercive option preservation（RCOT/BCOT）；
- Physical axis：action-conditioned successor executable option preservation；
- 两者都用 set feasibility / option preservation，而不是重新 scalarize 成一个 social+physical cost。

如果 successor branch 被实验支持，下一步才值得把这个结构升格成完整 contribution，并进一步把 expensive online successor regeneration distill 成 learned successor-viability head / mechanism token。

---

## 10. 第二分支：RVR Pareto Guard（diagnostic-only）

`cowp_rvr_pareto_guard` 只在 RVR prefix 严格更长且以下已有指标全部不劣于 COWP base 时允许切换：

- transport UCB；
- rule decision risk；
- action decision risk；
- pressure decision risk。

无新权重、无调参。

它回答一个单独问题：

> V16.8.29 的 9 个 induced failures，是否主要因为“max prefix first”覆盖了已有 risk evidence？

如果 guard 已经能保留大多数 rescue 且消掉 induced failure，则真正的下一步可能不是 successor model，而是 **recovery set-preservation / Pareto dominance frontier**。因此保留这个分支能避免过早认定唯一根因。

---

## 11. 速度优化：以后不要每轮 2×200/4×200

V16.8.29 实测 wall time：

- dev64 COWP：1213s；
- dev64 RVR：1239s；
- exact200 COWP：3209s ≈ 53.5 min；
- exact200 RVR：3678s ≈ 61.3 min。

历史 profiler 已显示 CPU candidate construction 约占 policy 87.8–88.6%，model forward 约 5.8%，selection 约 4–5%。V16.8.29 已优化 collision-audit 子组件，因此眼下最大的进一步收益来自 **实验协议**，不是再抠 Torch forward。

V16.8.30 改为：

### Stage A — equivalence16

只跑 COWP 16 scene，与 bundle 的 V16.8.29 COWP reference 比较。目的是防止代码修改改变 common path。

### Stage B — counterfactual48

48 scene：

- 10 RVR-rescued；
- 9 RVR-induced；
- 24 shared collision；
- 5 stable-safe controls。

只跑两个新方法。直接回答“保留 rescue / 避免 induced”的机制问题。

### Stage C — balanced dev96

只有 48 favorable 才跑。old dev64 + 9 induced counterexamples + 23 stable controls。

### Stage D — exact200 promotion

只有 96 favorable 才跑，并且 **只跑 promoted new method**。COWP/RVR exact200 已作为 immutable reference 打包，不再每次重跑 COWP，单次 promotion 直接省掉约 53.5 min baseline wall time。

注意：16/48/96/200 都已经参与算法选择，只能作为 development evidence。最终论文算法 freeze 后需要新建从未参与机制选择的 final evaluation set，并执行 multi-seed + paired CI；false-safe causal burden 的强 claim 还必须遵循论文里的 reactive-agent / human-audited stress protocol。

---

## 12. 下一轮严格执行顺序

设置旧 checkpoint，不训练、不重建 cache：

```bash
cd COWP_v16_8_30_SUCCESSOR_OPTION_VIABILITY

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/你的旧COWP目录/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh make_ids

# 只有 TFExample index 不存在时才运行
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh build_tfindex

# common-path engineering equivalence
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh base_equivalence16_parallel2

# 第一轮真正的机制实验
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_30_SUCCESSOR_OPTION_VIABILITY_CN.sh analyze_counterfactual48
```

**跑到这里停止。不要自动跑 96/200。**

下一轮优先看：

- old 10 RVR rescue 保留多少；
- old 9 RVR-induced collision 避免多少；
- successor probe 的使用率 / switch rate；
- base vs RVR successor conventional-existence、macro-diversity、candidate-count；
- guard 与 successor branch 谁更强；
- EP / kinematics 是否出现新 regression。

Promotion gate：

- Pareto guard 明显更好 → recovery dominance/frontier；
- successor-option 明显最好 → promotion successor physical option preservation；
- 两者 successor support 都经常 empty 且 collision 不改善 → structured proposal refinement；
- 不要同时修改 accepted-path kinematics，除非新方法本身明显把 kinematics 恶化。

---

## 13. 回归与交付

V16.8.30：

- 新 v16.8.30 + v16.8.29 focused helper tests：9/9 passed；
- launcher `sanity`：24/24 passed；
- 在本轮构建过程中完成的一次 full repository run 记录为 274 passed / 5 skipped / 8 failed；8 个 failure 与历史类别一致（6 个缺失 legacy launcher，2 个 stale semantic fingerprint）。
- 交付前我再次触发 full suite，但当前单命令 120s 窗口在约 25% 处超时；因此交付验收以刚刚重新跑通的 24/24 sanity 与 9/9 focused 为直接复核，完整回归数字保留为先前构建记录，不把超时误报成新的 full-suite 完成。

交付包含：

- V16.8.30 完整代码；
- V16.8.29→V16.8.30 patch；
- 下一轮 launcher；
- 本分析；
- `ALGORITHM_CHANGELOG_V16_8_30.md`；
- 同步更新后的总 `ALGORITHM_CHANGELOG.md`；
- V16.8.29 machine audit；
- 16/48/96 development manifests；
- immutable V16.8.29 COWP/RVR reference results；
- SHA256。
