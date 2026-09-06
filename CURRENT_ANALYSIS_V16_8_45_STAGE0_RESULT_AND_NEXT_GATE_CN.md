# V16.8.45 RCRSO Stage-0 结果分析与下一证据门

## 0. 判决摘要

本轮 V16.8.45R3 Stage-0 结果通过独立可靠性审计，允许进行算法归因。

但预注册是分阶段的，因此科学状态必须精确写成：

> **V16.8.45 RCRSO = Stage-0 GO / continue; full-policy verdict pending.**

它不是 STOP，也还不是可以写入论文主方法的 full GO。当前唯一合法的下一科学动作是：

`equivalence16 -> progressive lost7 (2+2+3)`。

在 lost7 之前设计或调优 V46，会违反上一轮冻结的决策树。

---

# 1. 可靠性审计

## 1.1 代码完整性

上传的 R3 ZIP SHA256：

`08d0bf4bf4610e4734ca1b591e07e9c81e879e445b3aedafa88e870f845e51fa`

内部 release 清单共检查 1133 个文件，0 missing / 0 mismatch。原包 focused sanity 为 143/143 passed。

R3 自身已经做过三组 science-field fidelity regression：legacy R2 vs R3 single、legacy R2 vs R3 parallel2 merge、cross-hypothesis roadgraph/environment namespace regression，均为 17/17 scientific fields exact。

## 1.2 Stage-0 两 shard 完整性

- shard 0: 12,846/12,846 examples, 841 groups, 495 scenarios
- shard 1: 12,389/12,389 examples, 771 groups, 475 scenarios
- scenario overlap: 0
- union: 970 scenarios
- total examples: 25,235
- total hypothesis groups: 1,612
- oracle-positive roots: 6,044
- hard verifier calls: 403,760

两个 shard 的 checkpoint SHA 一致：

`9bf347331981f147c6c6fed1484ff7a727c6e0867286074472ec4131a2ea5c81`

两个 shard 的 sidecar summary SHA 一致：

`0eac90aeb0666c656f69bdd3310589063b33f52df8fab6569552e2abaf198240`

Stage-0 并行 wall time 约 11,195 s（约 3.11 h），两个日志均正常 DONE。

970 而不是 1000 raw validation scenes 不是丢 shard：Stage-0 的对象是 val sidecar 中实际产生的 eligible hypothesis groups；全部 25,235 assigned examples 均完成，且 `structural_ineligible_hypothesis_groups=0`。

## 1.3 独立 raw-count 重算

Frozen baselines：

| Method | root hits / 6044 | full groups / 1612 | CSP groups / 1612 |
|---|---:|---:|---:|
| fixed bank | 3401 | 11 | 11 |
| V44 analytic | 4817 | 12 | 12 |

RCRSO：

| K | root hits | full groups | CSP groups | FHR | CSP rate |
|---:|---:|---:|---:|---:|---:|
| 2 | 5015 | 97 | 96 | 6.0174% | 5.9553% |
| 4 | 5179 | 103 | 102 | 6.3896% | 6.3275% |
| 8 | 5379 | 134 | 132 | 8.3127% | 8.1886% |
| 16 | 5552 | 156 | 154 | **9.6774%** | **9.5533%** |

以上均由两个 partial JSON 的整数计数独立重算，与 merged audit 逐项一致。

## 1.4 provenance 边界

当前结果包没有包含服务器端 `rcrso_stage0_selected.pt` bytes，所以本地不能重新计算 selected checkpoint 文件本身的 SHA，也不能重新执行完整 Stage-0 forward。

但两个 partial 中记录的 unselected checkpoint SHA 完全一致，merged audit 的 checkpoint provenance 与之相同；所有 raw counts、K curve 和 Gate 都能独立重建。因此这个边界记为 **publication provenance caveat，不阻断本轮 Stage-0 attribution**。

最终 publication artifact 应让闭环结果 JSON 同时携带 source tree hash、base checkpoint hash、RCRSO checkpoint hash、manifest hash。

---

# 2. 严格按预注册条件：Stage-0 GO

冻结条件：

1. `K in {2,4,8,16}`；
2. 在 validation FHR curve 上选择达到最大 observed FHR 的 95% plateau 的最小 K；
3. selected FHR 相对 best frozen baseline 的绝对提升 >= 3 pp；
4. selected VerifiedRootRecall > 0；
5. K 不允许用 lost7/CF48 调。

当前最大 FHR 为 K16 的 9.6774%，95% target 为 9.1935%。K8 只有 8.3127%，因此 frozen rule 选择：

> **K = 16**

best frozen baseline 是 V44 analytic extension：

`FHR = 12/1612 = 0.7444%`

RCRSO K16：

`FHR = 156/1612 = 9.6774%`

绝对提升：

> **+8.9330 pp > +3 pp → PASS**

VerifiedRootRecall：

- V44 analytic: 79.6989%
- RCRSO K16: **91.8597%**

因此：

> **Stage-0 support gate = GO**

但这只授权进入 closed-loop causal gate；它不等价于 full algorithm promotion。

---

# 3. 本轮最关键机制解释

## 3.1 真正成功的是 proposal completeness，不是 neural safety score

RCRSO 的核心正信号不是 loss、AUC 或可行性 logit，而是：

> neural set proposer 产生了大量以前不存在的候选；这些候选经过完全冻结的 hard verifier 后，确实把 universal retained-root support 从 12 个 hypothesis groups 提升到 156 个。

网络没有 certificate 权力。hard soundness 仍由：

`burden beta -> roadgraph -> Waymax kinematics -> ego current/shift -> responder/environment -> exact multi-root/multi-blocker CSP`

决定。

因此 V45 首次对 V44 的负结论给出直接正回应：

`finite hand bank + scalar analytic completion` 的 completeness floor 可以被一个 context-conditioned set-valued proposer显著抬高，而不需要放宽 hard semantics。

这支持 **soundness–completeness separation**：

- learned operator = completeness engine；
- frozen verifier = soundness authority。

## 3.2 Stage-0 上 joint CSP 不是 dominant rejection

K16：

- full root coverage groups = 156
- exact CSP completion groups = 154

即：

`154 / 156 = 98.72%`

一旦所有 retained roots 都至少有一个 hard-verified response，exact CSP 只额外淘汰 2 个 group。

因此当前 validation proxy 下，不能把下一步重心放到“放宽 multi-agent CSP / compatibility”。这会直接违反证据。

## 3.3 但绝对 completeness 仍非常低

K16 仍有：

`1456 / 1612 = 90.32%`

的 hypothesis groups 没有做到 universal retained-root support。

所以本轮不是“recourse completeness solved”，而是：

> **learned completeness direction被验证为有效，但 residual root holes 仍非常大。**

另外 `learned_only_root_nonempty_rate≈23.21%`，说明新 proposer 对全部 sidecar root contexts 的 verified hit 仍很稀疏。

## 3.4 K=16 是当前 tested maximum

K selection 合法 PASS，但 K16 正好是 preregistered 最大 query budget。当前 curve 没有在 K<16 明确进入 plateau。

这意味着：

- 不能声称 architecture/query set 已饱和；
- 也不能在 lost7 后事后把 K 改成 32/64 来救 outcome；
- 如果未来 current architecture closed-loop STOP，是否增加 proposal budget必须作为一个新的 preregistered architecture，而不是 V45 内调参。

---

# 4. Main Stack 当前应该是什么

需要区分 **promoted backbone** 与 **Stage-0-GO candidate extension**。

## A. Social hard-feasibility backbone — Freeze

1. compact-5k data/label contract；
2. Natural alternatives / stable canonical roots；
3. same-root RCOT；
4. BCOT；
5. protected-priority hard non-coercive certificate；
6. certificate-compatible set preservation；
7. hard-first selection；
8. explicit uncertified fallback。

这些层已有长期正证据和负消融保护，不因 V45 修改。

## B. Physical-interactive high-precision backbone — Retain / Freeze

1. V39 actual-controller-realized conflict-window recovery tube；
2. full-horizon physical hard certificate；
3. one-step shift closure；
4. exact blocker discovery；
5. exact blocker -> natural root support；
6. same-root low-burden responder hard verification；
7. responder-environment 双向 safety；
8. exact multi-root/multi-blocker CSP。

## C. Proposal-completeness extension — Stage-0 GO, 尚未 promotion

9. RCRSO K16 verified proposal augmentation。

RCRSO 现在只能标：

> **candidate Main-Stack layer / Stage-0 GO**

只有 equivalence16 + lost7 + 后续 frozen gates 通过后，才可以正式从 candidate 升到 Main Stack。

统一论文抽象仍然是：

> **Orthogonal Option-Set Feasibility**
>
> Social: safety 不能建立在他人的 natural low-burden option-set collapse 上。
>
> Physical-interactive: recovery 不能建立在 ego backup support 或 exact-blocker reasonable recourse-set collapse 上。

---

# 5. dominant bottleneck 是否应该收紧

## 可以做的“offline provisional tightening”

Stage-0 现在支持把 offline P0 从宽泛的：

`Root-Conditioned Recourse-Set Completeness`

临时收紧为：

> **Universal Retained-Root Verified Proposal Coverage**

尤其是 **teacher-sparse / hard-root holes**。

理由：

- proposer lift 很大；
- full→CSP 几乎不掉；
- 绝对 FHR 仍低；
- query budget 到上限。

## 不能做的收紧

不能把 closed-loop dominant bottleneck 直接改成“teacher sparsity”或“query K 不够”。Stage-0 是 outcome-blind support proxy，而实际 online blocker distribution、selector使用率、replanning 和 late failure 必须由 lost7 检验。

因此 closed-loop P0 仍保持 **unresolved pending lost7**。

---

# 6. 模型距离内部收敛最大的差距

按重要性排序：

1. **还没有 closed-loop causal conversion evidence。** Stage-0 的 +8.93 pp 是否能救 lost7 是最大缺口。
2. **绝对 FHR 仍只有 9.68%。** 说明 universal coverage far from solved。
3. **K=16 处在 tested maximum。** 当前 architecture 的 query-budget saturation 未证明。
4. **teacher positive support 稀疏。** 6,044 oracle-positive roots / 25,235 eligible examples；训练 set-matching 只在 `target_valid.any()` 的 context 上得到正 set supervision。teacher 没找到 positive 不代表真实 infeasible。
5. equivalence16 未执行：必须确保 RCRSO fail-closed extension不改变成熟 common path。
6. lost7 后仍有 rescue10、induced9、remaining29/CF48、fresh37、development exact200。
7. publication 尚缺真正 untouched final holdout、>=3 seeds、paired CI、reactive-agent evaluation、human-audited false-safe stress set。
8. compact-5k publication provenance caveat (`verify_cache_train.json pass=false / irrelevant pair blockers`) 仍需闭环。

所以当前状态应描述为：

> **mechanism validated offline, policy not internally converged.**

---

# 7. 下一步最该回答的算法问题

不是“RCRSO 的 FHR 能不能再高一点”，而是：

> **这 8.93 pp 的 hard-verified support lift，是否会在真正 closed-loop exact-blocker states 上转化成 RCRSO-exclusive certificate/action，并可靠救回 historical lost rescues，而不破坏 high-precision backbone？**

lost7 应专门回答四层归因：

1. online 是否真的出现 RCRSO verified profiles？
2. profiles 是否填补了某个 retained-root 空 domain？
3. exact CSP 是否实际使用这些 learned profiles并产生 RCRSO-exclusive certificate/action？
4. action change 是否最终形成 collision rescue，还是出现 late closed-loop/certificate-model mismatch？

最关键硬条件仍是：

> **lost7 new rescue >= 2/7**

未达到则当前 RCRSO architecture STOP。

---

# 8. 模型真正应该学会什么

当前网络最应该学习的不是一个“feasible probability”。

目标应该继续是一个 set-valued conditional correspondence：

`(natural root, exact blocker, ego current/shift tube, controller state, environment)`

`-> diverse recourse proposal set`

且这个 set 的价值由：

> **是否覆盖 hard-feasible recourse regions**

而不是 likelihood/AUC 决定。

尤其需要学到：

- multimodal controls，而非同一 mode 的小扰动；
- rare but hard-feasible root-specific responses；
- teacher-sparse contexts 中的 transferable geometry；
- 与 ego current/shift conflict geometry 对齐的 temporally structured residual sequence；
- diversity 应服务 verified support coverage，而不是单纯欧氏距离多样性。

---

# 9. 相关文献与理论边界

截至 2026-09-06 的检索给出几个清晰边界：

1. Hsu, Hu, Fisac, *The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems*, Annual Review 2024：learning + model-based verifier/filter 的模块化分离已是成熟安全控制范式。所以“神经网络外面包 hard verifier”本身不是 novelty。
2. Kim et al., *Generalized Backup Plan-Constrained MPC*, JGCD 2026：backup-plan feasibility / multi-horizon backup 已有直接工作。所以“备用轨迹可行性”本身不是 novelty。
3. Zheng et al., *Contingency Planning for Safety-Critical Autonomous Vehicles: A Review and Perspectives*, 2026：external interaction uncertainty increasingly uses proactive recourse/branching。因此 contingency/recourse 也不能单独包装。
4. Yang et al., *Safe and Nonconservative Contingency Planning ... Reachable Set Barriers*, 2025/2026：learned human control-intent sets + reachable sets + hard barriers已用于 AV contingency safety。
5. Ichter et al., *Learning Sampling Distributions for Robot Motion Planning*: learned biased sampling can improve finite-sample efficiency while retaining planner guarantees when combined with a base sampler/support-complete process。
6. Sacks & Boots, *Learning Sampling Distributions for MPC*, CoRL 2023：learned control sampling distributions improving sample efficiency is established territory。
7. CoverNet / MultiPath / DPP diverse forecasting：finite set coverage、mode diversity也是成熟思想。
8. 2025 counterexample-guided robust CBF synthesis：learner↔verifier 的 counterexample-guided loop已有理论/实践先例。

因此 CCF-A novelty 不能写成：

- Transformer outputs K controls；
- hard verifier；
- backup feasibility；
- diverse proposals；
- learned sampler；
- CEGIS 本身。

更强的贡献仍应是组合后的问题结构：

> **false-safe as hard feasibility defect + natural-root semantics + low-burden option preservation + control/shift-closed ego support + exact-blocker recourse correspondence + verifier-wrapped completeness learning**。

---

# 10. “下一算法分支”现在为什么不应该落地 V46

上一轮预注册明确规定：

`Stage-0 PASS -> equivalence16 -> progressive lost7`

因此现在直接设计并运行 V46 会把 outcome-blind Stage-0 机制信号误当成 closed-loop conclusion。

本轮落地的是 **V16.8.45R4 Stage-1 Causal Gate harness**，不是新 scientific method：

- 独立从两个 partial raw counts 复算 Stage-0；
- fail-closed 绑定 Stage-0 selected checkpoint；
- 复用完全相同 V45 RCRSO online method；
- 只允许按 frozen order 进入 equivalence16 和 progressive lost7。

这比提前造 V46 更符合科研纪律。

---

# 11. 如果 V45 lost7 STOP，下一 scientific branch 应是什么（预设计，不启用）

如果当前 RCRSO Stage-0 GO、但 lost7 <2/7，且日志显示主要失败是“online retained roots 仍无 verified learned proposal”，下一预注册 architecture 应考虑：

## Verifier-Guided Adaptive Recourse Cover (VG-ARC)

核心不是把 K 从16改成32，而是把一次性 amortized set prediction 改成 **proposal–verification–counterexample refinement**：

1. RCRSO 先产生 amortized initial set；
2. frozen hard verifier 对每个失败 proposal返回结构化 failure witness（burden / roadgraph / kinematics / ego-current / ego-shift / environment / CSP conflict）；
3. refinement proposer 条件化于 `(z, previous proposals, failure witnesses)`，专门生成与失败边界互补的新 controls；
4. 每轮输出仍必须重新过完整 frozen verifier；
5. 根 domain 一旦 nonempty 即停止该 root refinement，优先把预算给 unresolved roots；
6. exact CSP 仍只在 hard-verified domains上工作。

理论上可以写成：

- **soundness**：所有最终返回 profiles 均属于 frozen hard set，因此 learning 永远不能制造 false-positive certificate；
- **finite-budget completeness objective**：优化 unresolved-root cover，而不是平均 reconstruction loss；
- 如果将来引入一个对 bounded same-root control domain 具有 full support 的 base sampler，并与 learned sampler 混合，则在 feasible recourse set 具有正测度的假设下，可以获得 sampling-style asymptotic completeness；learned sampler只提高 finite-sample效率。

这与 sampling-based planning / CEGIS 有理论联系，但论文 novelty仍必须落在 **root-conditioned low-burden recourse correspondence + option-set feasibility**，而非“用了 CEGIS”。

**VG-ARC 当前不进入代码、不参与结果选择。** 是否激活由 V45 lost7 结果决定。

---

# 12. 下一步命令

服务器必须保留 R3 Stage-0 生成的 selected checkpoint：

```bash
cd COWP_V16_8_45R4_STAGE1_CAUSAL_GATE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all

# 指向你服务器真正的 R3 Stage-0 输出目录；必须包含：
# stage0_val_support_audit.json
# rcrso_stage0_selected.pt
# stage0_partials/
export RCRSO_STAGE0_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_45r3_stage0_runtime_observability

bash NEXT_RUN_COMMANDS_V16_8_45R4_STAGE1_CAUSAL_GATE_CN.sh audit_stage0_result
bash NEXT_RUN_COMMANDS_V16_8_45R4_STAGE1_CAUSAL_GATE_CN.sh sanity

# E1: common path fidelity
bash NEXT_RUN_COMMANDS_V16_8_45R4_STAGE1_CAUSAL_GATE_CN.sh base_equivalence16_parallel2
```

只有 equivalence16 `passed=true`，再执行：

```bash
bash NEXT_RUN_COMMANDS_V16_8_45R4_STAGE1_CAUSAL_GATE_CN.sh lost7_batch1_parallel2
bash NEXT_RUN_COMMANDS_V16_8_45R4_STAGE1_CAUSAL_GATE_CN.sh analyze_lost7_progressive
```

查看：

`outputs/v16_8_45r4_stage1_causal_gate/lost7_v45_progressive_gate.json`

- `continue_progressive=true` 才跑 batch2；
- 再分析；仍为 true 才跑 batch3；
- 一旦 `mathematically_impossible=true`，当前 RCRSO architecture STOP；
- 一旦 `lost7_gate.pass=true`，才进入 retained3 / rescue10 / induced9。

不要缩短 80-step rollout，不要根据 batch1/2 结果重排 IDs，不要改 K。
