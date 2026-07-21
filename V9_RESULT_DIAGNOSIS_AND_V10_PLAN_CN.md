# COWP v9 结果诊断与 v10-GCT 优化方案

## 1. 结论

### 1.1 论文核心 idea 是否得到佐证

只能得到**部分、弱证据**，不能称为已经证明。

支持证据：
- 相对 conventional_safety，COWP 将 SelectedFalseSafeRate 从 0.5931 降至 0.4897，绝对下降 10.33 个百分点，约为 17.4% 相对下降；
- HBCR 从 0.3984 降至 0.3232；
- OPR 从 0.7411 提升至 0.7558；
- 9 个阈值产生 9 个不同 selection points，说明证书确实连接到了 selector。

反证/不足：
- Witness AUPRC 仅 0.4312；
- LearnedAcceptNCFRecall 仅 0.1280；
- LearnedAcceptedCandidateRate 仅 0.0613；
- Fallback 从 conventional 的 0.1043 上升到 0.2210；
- EP 从 0.3893 降至 0.3676；
- mechanism gate 为 false；
- 100 场景 Waymax probe 没有执行。

因此 v9 证明的是“硬筛选会改变选择并降低部分 false-safe”，没有证明“primitive-indexed same-root transport 被准确学习并是性能提升的原因”。

### 1.2 是否达到闭环 SOTA

没有。当前结果包没有真实 online Waymax rollout 结果，总控在 learned-offline mechanism gate 失败后停止。`SelectedWaymaxUnsafeRate` 来自稀疏附着的候选 replay outcome，COWP 的 coverage 仅 0.2472，不能替代闭环评估，也不能与方法间覆盖率不同的数值直接比较。

## 2. v9 关键数值

| 方法 | EP ↑ | Fallback ↓ | OPR ↑ | HBCR ↓ | Selected False-Safe ↓ |
|---|---:|---:|---:|---:|---:|
| Planner score only | 0.4516 | 0.0000 | 0.7237 | 0.4666 | 0.5823 |
| Conventional safety | 0.3893 | 0.1043 | 0.7411 | 0.3984 | 0.5931 |
| Soft burden only | 0.3868 | 0.1043 | 0.7440 | 0.3964 | 0.5871 |
| Universal NCF | 0.1042 | 0.4652 | 0.8149 | 0.1745 | 0.2914 |
| COWP v9 | 0.3676 | 0.2210 | 0.7558 | 0.3232 | 0.4897 |

COWP v9 相对 conventional：
- false-safe：-0.1033（有意义）；
- HBCR：-0.0752（有意义）；
- OPR：+0.0147（偏小）；
- EP：-0.0217，约 -5.6%（不满足高水平规划方法要求）；
- fallback：+0.1167（过度保守）。

## 3. 训练机制诊断

### 3.1 Mode conflict 没有超过无技能基线

transport 数据中 mode_conflict_rate 约为 0.353。只预测类别先验的 BCE 熵基线约为 0.649。

v9 最后一个 epoch：
- val/set_transport/mode_conflict = 0.7222；
- val/set_transport/mode_retain = 0.6167；
- val/set_transport/root_recovery = 0.1268。

mode-conflict BCE 比先验基线更差，说明模型没有可靠识别候选与具体 natural primitive 的冲突。

### 3.2 Set-transport head 缺少实际冲突几何

v9 的 mode feature 只由 candidate embedding、agent embedding、graph embedding 和 mode latent 相加得到。它没有直接输入 candidate trajectory 与 natural trajectory 的最小距离、相对速度、航向、接近时刻或 footprint clearance。

标签由轨迹几何冲突构造，模型却没有显式几何输入，导致 direct supervision 的任务条件不足。

### 3.3 冻结策略阻断了 primitive identity 适配

运行参数 `FREEZE_BACKBONE_EPOCHS=999` 使 graph、candidate_encoder、natural_decoder、witness_decoder 在 transport 的全部 10 个 epoch 冻结。v9 checkpoint 加载时又存在新增/不兼容层，因此 primitive identity 与 witness proxy 无法根据新 direct labels 适配。

日志中 legacy witness loss/metric 几乎恒定，是冻结而不是“已经收敛”。

### 3.4 Root recovery 的概率计算不守恒

v9 对每个 response slot 使用 `root_prob * low_safe_prob`，再对 32 个 slot 做 product-of-complements，但没有乘 response mixture weight。若 root 分布较均匀，多个低置信 slot 会被当成多个独立恢复事件，显著高估 root recovery。

数据标签 root_recovery_mean 约 0.024–0.025，而训练预测的绝对误差长期约 0.126，说明该结构与稀疏标签不匹配。

### 3.5 Candidate classifier 压过论文机制

- CandidateCertificate/FalseSafe_AUPRC = 0.8219；
- WitnessQuality/AUPRC = 0.4312。

因此 selector 的有效信息主要可能来自 candidate-level classifier，而不是论文主张的 primitive transport certificate。若不做严格消融，审稿人会认为核心机制只是附加解释头。

## 4. CCF-A 研发门槛

以下是项目门槛，不是 CCF 官方统一数值标准。

### 4.1 机制门槛

| 指标 | 当前 | 允许进入 100 场景 probe | 论文目标 |
|---|---:|---:|---:|
| Mode conflict val BCE | 0.722 | < 数据先验熵约 0.649 | <= 0.55 |
| Witness AUPRC | 0.431 | >= 0.55 | >= 0.65 |
| Accepted NCF recall | 0.128 | >= 0.30 | >= 0.50 |
| Accepted candidate rate | 0.061 | >= 0.10 | >= 0.20 |
| Fallback | 0.221 | <= 0.25 | <= conventional + 0.03 |
| False-safe absolute improvement | 0.103 | >= 0.08 | relative reduction >= 25–30% |

### 4.2 离线 selector 门槛

- EP 至少达到 conventional 的 97%，最好不下降；
- fallback 不得比 conventional 高超过 3 个百分点；
- OPR 至少提升 3 个百分点；
- HBCR 至少相对下降 20%；
- coverage-matched sparse Waymax unsafe 不得恶化；
- root-transport 必须显著优于 semantic-only、aggregate/Pareto 和 planner-score-only。

### 4.3 闭环门槛

- 100 场景仅作 smoke/probe，不用于 SOTA；
- 1000 个 paired scenarios 用于开发决策；
- 最终至少 5000 个 paired scenarios × 3 seeds；
- 报告 paired bootstrap 95% CI；
- Waymax overlap、offroad、wrong-way、route progression、kinematics 均不得出现显著退化；
- 在 logged playback、IDM/rule-reactive 或论文实际声明的 reactive setting 下分别报告结果；
- 与 strong planner、conventional safety、soft burden、universal NCF、planner-score-only 及核心消融比较。

## 5. v10-GCT 修改

### 5.1 Geometry-Conditioned Transport

对每个 ego candidate–natural primitive 直接编码：
- 最小、平均和终点距离；
- 最接近时刻；
- 接近距离变化；
- 平均航向一致性；
- 平均相对速度；
- 最小 footprint clearance。

这些特征经过 MLP 加入 mode conflict/retain head。

### 5.2 质量守恒的 same-root recovery

改为：

`sum_r p(response_r) * p(low-safe_r) * p(root=m | response_r)`

不再把 32 个 response slots 当成独立完整概率事件。这样均匀 root 分布只会分配约 1/M 的恢复质量。

### 5.3 Response slot identity

新增 slot-specific response embedding 和零初始化 root residual head。加载 v9 checkpoint 时保持原 root logits，新增残差从零开始训练，避免完全随机重置。

### 5.4 Direct loss 平衡

- mode conflict/retain 使用动态 class-balanced BCE；
- root recovery 使用 weighted presence BCE + positive magnitude L1；
- 增加 no-skill entropy baseline 诊断。

### 5.5 Granular freeze

transport stage：
- graph 仅前 2 epoch warm-up 冻结；
- candidate_encoder、natural_decoder、witness_decoder 始终可训练；
- set_transport、response_decoder 始终可训练；
- 添加 0.08 权重的 natural-set auxiliary loss，防止 primitive identity 漂移。

planner stage：
- graph/natural/witness 固定；
- candidate encoder warm-up 后可适配；
- transport/response/candidate certificate 继续训练。

### 5.6 更严格 gate

v10 默认进入 online probe 前要求：
- NCF recall >= 0.30；
- witness AUPRC >= 0.55；
- accepted rate >= 0.10；
- fallback <= 0.25；
- false-safe absolute improvement >= 0.08。

不得通过放松 gate 来制造闭环结果。

## 6. 验证状态

已完成：
- Python compileall；
- v10 shell syntax；
- geometry transport、weighted root mass、response root refinement、新旧 transport、response、natural、witness、cache 相关定向测试：19 passed；
- 额外 v9 stale same-root test 已更新并通过。

全量 pytest 在旧测试集合中运行超过 5 分钟后超时，超时前输出 29 个通过点且未显示失败；因此部署后仍建议按指令分组运行定向测试，不声称全量测试已完成。

## 7. 预期判读

如果 v10 的 mode-conflict BCE 仍高于先验熵，说明 natural mode identity/标签噪声仍是主因，下一步应升级 response-root assignment 和 natural primitive matching，而不是继续调 selector threshold。

如果 offline mechanism gate 通过但 online 下降，下一步应做 selected-candidate online hard-negative mining 或 simulator-in-the-loop fine-tuning，而不是继续扩大静态分类器。
