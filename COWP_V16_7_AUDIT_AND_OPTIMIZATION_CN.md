# COWP v16.7：v16.6 机制验证失败审计、算法修复与闭环实验方案

## 1. 审计范围与结论边界

本报告仅依据本轮上传的：

- `COWP(5).zip` 当前代码；
- `cowp_v16_6_natural_recovery_v9labels_seed2026.zip`；
- `cowp_v16_6_natural_attribution_aligned_v9labels_seed2026.zip`；
- `cowp_v16_6_full_pipeline_v9labels_seed2026.zip`；
- `interactive_planning_v16_6_revised(1).tex`。

没有把旧对话中的记忆当作论文或结果证据。当前结果只包含 learned-offline/稀疏缓存候选回放；由于 mechanism gate 失败，真正的完整 online Waymax closed-loop 没有执行。因此本报告不宣称达到 SOTA，也不把缓存中的 `CR=0` 当成真实闭环碰撞率。

## 2. 总结性判断

### 2.1 Natural foundation 已成立为开发基础

本轮：

- `natural_basis_gate=true`；
- `natural_effectiveness_gate=true`；
- aligned attribution `pass=true`，但 `paper_claim_ready=false`；
- OBS capacity 在 1,881 个可配对 OBS 场景上的平均改善为 **0.07049 m**，95% paired bootstrap CI 为 **[0.04353, 0.09871] m**；
- Mass-aware root envelope 将直接优化的 squared-excess objective 降低 **0.31165**，将 violation mass 降低 **0.05373**；两项置信区间均不跨 0。

因此，本轮失败已不是 natural decoder、yaw 或 root identity 造成。

### 2.2 Mechanism verification 失败是真实的 downstream 瓶颈

校准集与 held-out 集互斥，预算扫描有 15 个不同选择点，但不存在满足所有约束的预算。校准最终只能选 `least_violation` 的 0.70：

| 指标 | v16.6 held-out | 当前内部目标 | 判定 |
|---|---:|---:|---|
| Pair witness AUPRC | **0.71293** | 约 ≥0.70 | 已达到 |
| RootTransport conflict AUPRC | **0.22524** | 约 ≥0.70 | 严重不足 |
| BCOT false-safe AUPRC | **0.40640** | 约 ≥0.70 | 明显不足 |
| Learned NCF recall | **0.20232** | 约 ≥0.60 | 严重不足 |
| Accepted candidate rate | **0.09036** | 约 ≥0.20 | 不足 |
| Fallback rate | **0.23464** | 约 ≤0.15 | 过高 |
| Selected false-safe rate | **0.47207** | 越低越好 | 比 conventional 改善 20.98% |

结论是：pair witness 已有辨识力，但从 natural roots 到 candidate certificate 的压缩失真，导致证书过度保守、NCF 召回低。

## 3. 算法缺陷与工程错误的分离

### 3.1 已确认的工程/监督错误

#### 3.1.1 Candidate BCOT 类别权重方向错误

训练和验证可判别候选中，false-safe 占比约 87%--88%。原代码使用 `pos_weight` 且下限为 1，只能放大正类，不能对多数正类降权。这会系统性抬高风险分数，和“风险预算增加到 0.70 仍只有 9% 接受率、23.5% fallback”一致。

v16.7 改为对正负类都做逆频率加权的 symmetric class-balanced BCE，并增加同场景 NCF/false-safe 排序损失。

#### 3.1.2 Priority arrival-order 标签实际失效

旧 `_first_arrival_to_close_points` 只比较同步时间点，并把同一个索引同时返回给 ego 和 agent。因此 `agent_t + margin < ego_t` 与相反分支几乎永远无法触发。

v16.7 改为在两条路径的 pairwise geometry 上寻找最早共享冲突区域，分别返回 ego 与 agent 的到达时间。

#### 3.1.3 `controlled_by_signal` 被错误当成实时路权

数据中的 lane 被 signal 控制不等于当前相位拥有或失去通行权。当前 `ScenarioData` 没有将每条 lane 与实时信号相位可靠关联，原规则会制造 priority label noise。v16.7 删除该推断，只保留有数据支持的 stop、lane ownership、arrival-order 等规则。

#### 3.1.4 Root alignment 训练/评估逻辑重复且会交换 source identity

旧实现多处重复计算 root assignment，并主要按 full-horizon ADE 对齐，可能让 OBS root 被 neutral/priority root 替代。v16.7 增加统一的 source-aware multi-horizon alignment：1/3/5/8 秒共同参与，并对 source mismatch 添加有限惩罚。训练和评估共用同一实现。

#### 3.1.5 论文和候选生成实现不一致

论文 v16.6 把候选写成 route-conditioned lattice-MPC；实际代码是有限的恒加速度、平滑停车/蠕行、冲突定时、固定 lateral offset、terminal-state primitives。该不一致会削弱可信度，并掩盖固定 lateral primitive 的 off-road 问题。

v16.7 论文改为真实的 `map-screened kinematic primitive bank`；代码在标签构建时加入轻量局部 lane-corridor 筛选。macro 只作为 proposal family descriptor，不作为真实意图、路权或谈判成功证明。

### 3.2 真正的算法缺陷

#### 3.2.1 RootTransport 没有学到 conflict-conditioned recoverability

Pair witness AUPRC 已达 0.713，但 RootTransport conflict-conditioned AUPRC 只有 0.225，说明问题不是“是否有 witness”，而是 root 级机制估计：哪些 natural root 与 ego candidate 冲突，以及同一 root 是否保留低 burden response。

v16.7 增加：

- near-conflict fraction；
- closing speed；
- midpoint swept clearance；
- root conflict within-pair ranking；
- source-aware root alignment；
- 对未恢复冲突质量、burden tail 和 option shortfall 的单调可学习组合。

#### 3.2.2 一个混合风险分数承担了两种不同定义

论文核心是 priority-aware risk transfer，但旧 BCOT 同时混合所有 critical agents。这样会把“agent-priority 被迫高 burden”与“ego-priority 场景下对方轻微调整”放在同一硬 veto 中，导致证书泛化为 universal deference。

v16.7 分为：

- protected-priority risk：agent-priority 和明确 equal/negotiated 关系，用于硬证书；
- all-critical global risk：只作 anti-degeneration 与压力测试诊断。

#### 3.2.3 固定手工聚合器无法适应真实边界

旧 candidate risk 使用固定加权组合，难以同时校准 RootTransport、burden 和 option loss。v16.7 使用正数、归一化、单调可学习权重。任何机制 deficit 增大都不能令风险下降，因此保留可解释性和论文 novelty，而不是换成通用黑箱候选分类器。

#### 3.2.4 Proposal bank 的 NCF 上限不足

只有 **34.84%** held-out 场景存在至少一个 NCF proposal；即使 certificate 完美，scene-level recall 仍受 proposal bank 上限限制。当前 certificate 的 NCF scene retention 为 **89.23%**，说明它并非把大多数“已经存在的 NCF 场景”全部删掉，但 candidate-level NCF recall 仍很低，表明 NCF 候选的风险排序与接纳仍不充分。

v16.7 先修地图筛选、priority label 和 mechanism calibration。后续若 proposal NCF scene rate 仍低，应增加 route/frenet-conditioned lateral primitives，而不是继续放宽 certificate。

## 4. OBS capacity 与 Mass-aware root envelope 的本轮证据

### OBS capacity

- paired OBS gain：0.07049 m，CI 不跨 0；
- paired overall gain：0.02556 m，CI 不跨 0；
- neutral、priority 分支没有显示有害退化。

判定：**在单 seed、对齐 checkpoint/场景协议下生效，值得保留；尚未达到论文级证明。** 最终论文需要至少 3 seeds，并在重建 v17 labels 后复现。

### Mass-aware root envelope

- exact squared-excess 相对降低约 27.05%；
- violation mass 相对降低约 28.51%；
- OBS 改善约 0.113 m；overall 改善约 0.032 m；
- emergency hard projection active mass 为 0。

判定：**soft semantic envelope 明显生效，应保留。** 当前不能把性能归因于 emergency hard projection；后者只作为物理/数值 safeguard。

## 5. 上一轮七个“继续深化”点的复核

| 设计 | 本轮证据 | v16.7 决策 |
|---|---|---|
| Typed causal dynamics decoder | natural 两个 gate 均通过 | 保留 |
| Yaw reference-frame 修复 | yaw error 约 7.7e-8 rad | 保留 |
| OBS capacity | paired CI 排除 0 | 保留，做 3 seeds |
| Mass-aware root identity | excess/violation 显著下降 | 保留 |
| Same-root retained-mass OPR | natural identity 成立，但 downstream transport AUPRC 低 | 保留定义，深化 transport |
| Hard non-coercive certificate | HBCR 下降 25.44%，但接受率低/FSR仅下降7.96% | 部分有效，改为 protected-priority certificate |
| Fail-fast gates | 成功阻止无效 online 消耗 | 保留，并区分 development/paper gate |

## 6. 五个上一轮尚未证明点的状态

1. **Planner：部分证据。** Pair ranking accuracy 为 0.7939，但没有 online closed-loop，不能证明最终规划收益。
2. **Selector/calibrator：未通过。** false-safe 比 conventional 低，但 mechanism gate 失败，coverage/fallback 不合格。
3. **Emergency projection：仍未证明。** active mass=0。
4. **Conformal coverage：未实现。** 论文只应写 future/optional extension。
5. **Reactive-agent burden transfer：未证明。** 本轮没有完整在线 reactive Waymax 结果。

## 7. 本轮是否有真正闭环结果

没有。`SelectedWaymaxCollisionRate=0` 和 `SelectedWaymaxOffroadRate=0.16` 只来自附着在缓存候选上的稀疏 outcome：

- outcome coverage 仅 **23.94%**；
- finite log-divergence count 为 0；
- mechanism gate 失败后，完整 online probe/full 没有运行。

这些值只能用于定位候选质量，不能作为论文主表的闭环 CR/OR。

## 8. 对内部 CCF-A promotion targets 的差距

| 目标 | 当前 | 差距/原因 |
|---|---:|---|
| RootTransport AUPRC ≥0.70 | 0.225 | root geometry、alignment、recovery压缩不足 |
| BCOT AUPRC ≥0.70 | 0.406 | 类别失衡、混合风险定义、固定聚合器 |
| NCF recall ≥0.60 | 0.202 | risk过高、root判别差、proposal ceiling |
| NCF precision ≥0.60 | 旧代码未报告 protected precision | v16.7 新增 |
| Accepted rate ≥0.20 | 0.090 | certificate 过度保守 |
| Fallback ≤0.15 | 0.235 | 无可行 operating point |
| FSR/HBCR 降低25%--30% | 7.96% / 25.44% | 只在高 burden 子集有效，广义 false-safe 未控制 |
| Progress regression ≤1--2 pp | normalized EP +1.45 pp，但 EP_m -15.57% | 指标口径/高 fallback，需在线 route progression |
| 至少3 seeds | 1 | 尚无论文级统计 |

这些是内部目标，不是 CCF 官方录用阈值。

## 9. 建议保留的精简指标体系

### 必须作为主结果

1. 标准 online Waymax：overlap/collision、offroad、wrong-way、route progression、kinematic infeasibility、log divergence；
2. **PBTR**：protected-priority burden-transfer rate，最直接对应论文核心命题；
3. **Protected OPR**：同 root、低 burden retained probability mass；必须与 progress/coverage 联合报告；
4. **BTE-CVaR25**：保护关系中最严重 burden transfer 的尾部强度；
5. **NCF scene retention**：区分 proposal 不足与 certificate false rejection；
6. **Non-coercive progress regret**：判断 selector 是否在已有 NCF 候选中选择了不必要的保守方案；
7. **PBTR--coverage curve**：避免单预算 cherry-pick；
8. NCF precision/recall：证明 certificate 不是只靠拒绝一切取得低 PBTR。

### 次要或附录指标

- global FSR：作为 all-critical stress test；
- CBS、HBCR：保留历史连续性，但不宜与 PBTR 重复占主表；
- WLA、MTA：只在有人工 witness 标注的 stress set 上有意义，否则放附录。

## 10. v16.7 代码修改

### 机制算法

- symmetric class-balanced candidate BCE；
- protected/global 双风险头；
- 单调机制聚合器；
- candidate ranking；
- swept conflict geometry；
- root conflict ranking；
- source-aware multi-horizon root alignment；
- protected-priority calibration/gate；
- PBTR、BTE-CVaR25、protected NCF precision/recall、NCF retention 和 progress regret。

### 标签与候选

- 修复独立 arrival time；
- 删除无实时相位支持的 signal-priority 推断；
- synthetic proposals 加局部 lane-corridor 筛选；
- 论文如实改为 finite map-screened kinematic primitive bank。

### 工程效率

- root rank loss 向量化，去掉 B×K×A Python 循环；
- root alignment 一次计算、多 loss 复用；
- 保留 frozen-backbone/static-DDP/trainable-only AdamW；
- 地图筛选只在数据构建执行，不进入训练热路径；
- immediate mechanism rerun 复用已验证 natural checkpoint，只重训 transport/planner。

## 11. 论文修订原则

v16.7 TeX 基于本轮上传的 v16.6 TeX 直接修改：

- hard feasibility 从“所有 critical agents”改为 protected-priority agents；
- all-critical FSR 保留为压力诊断，防止 burden 转移到非保护对象；
- 删除不真实的 route-conditioned lattice-MPC 描述；
- macro 改为 proposal descriptor；
- beta 明确为冻结 heuristic table，不伪称 conformal；
- optimized response 改为有限 typed response beam；
- PBTR/BTE-CVaR25/NCF retention/progress regret 成为核心自造指标；
- 未通过机制和 online gate 前不写 `demonstrate/outperform/SOTA`。

## 12. 实验顺序

### A. 立即隔离验证 downstream 修复

复用已通过的 v16.6 natural checkpoint，只重训 v16.7 transport/planner。若机制 gate 通过，再运行 100 场 probe 和 1000 场 full online Waymax。

### B. 论文级数据与三 seed

由于 priority 标签和候选 bank 已变，最终论文必须使用 `PREPARE_COWP_V16_7_DATA_CN.sh` 新建 `formal_v17`，然后重新运行 natural、aligned attribution、mechanism 和 online closed-loop，至少 seeds 2026/2027/2028。


## 13. 交付回归验证

最终压缩包独立解压后：`pytest` **138 passed**；全部 shell 脚本通过 `bash -n`；Python `compileall` 通过；TeX 内部 label/reference 静态检查无缺失。上传文件未包含 `interactive_planning_revised.bib`，因此 LaTeX 可生成正文 PDF，但外部文献引用无法在本环境完成解析。
