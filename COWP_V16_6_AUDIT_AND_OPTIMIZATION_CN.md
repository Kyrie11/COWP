# COWP v16.6 严格复核、修复与后续闭环实验方案

## 1. 本次复核范围

本报告只以本轮实际上传内容为依据：

- 当前代码：`COWP(4).zip`；
- strict natural recovery：`cowp_v16_5_natural_recovery_v9labels_seed2026(1).zip`；
- 两个严格消融：`cowp_v16_5_natural_ablations_v9labels_seed2026(1).zip`；
- 前一版论文：`interactive_planning_v16_4_revised(2).tex`；
- 上轮修改稿：`interactive_planning_v16_5_revised(2).tex`。

没有用未上传论文的记忆补写方法，也没有把旧版本 smoke 结果当作本轮证据。

## 2. 最终审计结论

当前 `natural_component_attribution_gate.json` 的 `pass=false` 不能直接解释为 OBS capacity 和 mass-aware root envelope 都无效。失败由两类问题叠加造成：

1. **归因协议工程错误**：主模型外部诊断使用 epoch 15，而两个消融均使用各自 epoch 19 的最佳 checkpoint；三者不是同一训练时刻。旧报告也没有记录采样场景哈希和逐场景配对指标，无法证明比较完全配对。
2. **gate 与训练目标错配**：mass-aware envelope 的训练目标是 probability-floor 加权的、超过 interior margin 的归一化平方超额，而旧 gate 的主判据却是未直接优化的全局平均 path ratio 下降至少 0.03。

因此本轮失败首先是**实验归因协议错误**，其次才可能包含组件强度不足。正确做法不是放宽阈值或强制把 gate 改成 true，而是对同一 epoch、同一场景重新诊断，并检查模型实际优化的量。

## 3. strict natural recovery 是否健康

主模型的 natural basis 和 natural effectiveness gate 均已通过：

| 指标 | v16.5 main |
|---|---:|
| 8 s weighted ADE | 1.18093 m |
| OBS 8 s ADE | 2.71854 m |
| overall gain vs analytic basis | 0.70531 m |
| OBS gain vs analytic basis | 1.38523 m |
| yaw consistency error | 7.41e-08 rad |
| velocity consistency error | 0.00208 m/s |
| probability-weighted soft-envelope violation mass | 0.13825 |
| emergency envelope p99 ratio | 0.91890 |
| effective modes | 4.48890 |
| optimizer steps / epoch | 2044 |
| AMP skipped steps | 0 |

这排除了 v16.2/v16.3 早期的“没有真实 optimizer update”“FP16 非有限值被静默处理”等失效链。当前可以把重点放在公平归因和算法强度上。

## 4. 原 attribution gate 为什么失败

旧 gate 使用各自 best checkpoint 得到：

| 对照 | 8 s overall | OBS 8 s | soft ratio | violation mass |
|---|---:|---:|---:|---:|
| main, epoch 15 | 1.18093 | 2.71854 | 0.38925 | 0.13825 |
| no OBS capacity, epoch 19 | 1.18674 | 2.74204 | 0.39780 | 0.20585 |
| no mass envelope, epoch 19 | 1.20423 | 2.80524 | 0.40044 | 0.20475 |

旧 gate 失败于：

- OBS capacity 的 OBS 改善只有 0.02351 m，小于人为设置的 0.05 m；
- mean path ratio 只下降 0.01119，小于人为设置的 0.03。

但这两个数字都来自不公平的 epoch 15 对 epoch 19 比较。

### 4.1 同一 epoch 15 的结果

训练历史在共同 epoch 15 上显示：

| 组件 | 对照值减 main，正值表示 main 更优 |
|---|---:|
| OBS capacity：OBS minADE 改善 | 0.05149 m |
| OBS capacity：overall trajectory 改善 | 0.02718 m |
| mass envelope：OBS minADE 改善 | 0.11406 m |
| mass envelope：overall trajectory 改善 | 0.03647 m |
| mass envelope：实际 trust/excess objective 下降 | 0.32689 |

这说明旧 gate 的结论会随着 checkpoint 选择发生改变，不能作为组件无效证据。

## 5. 两个核心组件的严谨判定

### 5.1 Source-adaptive OBS capacity

**当前判定：有方向性证据，但尚未达到论文级证明。**

证据：

- 在共同 epoch 15，OBS minADE 改善 0.05149 m，overall trajectory 改善 0.02718 m；
- 在大多数共同验证 epoch 上，main 相对 no-capacity 的 OBS 指标为正；
- own-best 外部结果仍为正向 0.02351 m，只是小于旧硬阈值。

尚不能直接下论文结论的原因：

- 外部 2,000 场报告不是同一 epoch；
- 没有逐场景 paired CI；
- 单 seed 无法区分稳定增益和训练波动；
- 后期增益有衰减，说明固定 source capacity 可能被其他 decoder 自由度部分吸收。

v16.6 不伪造新的 adaptive-capacity 复杂模块，而是先做协议正确的配对诊断。若 aligned report 仍显示 point estimate 为负，则才应把它认定为算法问题，并做容量倍率敏感性实验；若为正但 CI 跨 0，则允许继续闭环收集机制证据，但论文中不能声称该组件已被统计证明。

### 5.2 Probability-mass-aware multi-horizon root envelope

**当前判定：组件明显处于工作状态，证据强于 OBS capacity，但仍需对齐诊断完成正式归因。**

证据：

- own-best 外部诊断中，违反 soft envelope 的概率质量由 0.20475 降到 0.13825，绝对下降 0.06651，约为 32.5%；
- 同时 OBS 8 s ADE 改善 0.08671 m，overall 改善 0.02330 m；
- 同一 epoch 15，训练所实际优化的 trust/excess objective 下降 0.32689；
- emergency p99 ratio 小于 1，物理 guard 未被突破。

旧 gate 的平均 path ratio 不是训练目标。模型可以主要消除高概率根上的超界尾部，而平均 ratio 变化很小；这种行为正符合 retained probability mass 的论文定义。

需要明确：`projection_active_mass=0`。因此本轮改善来自 **soft semantic identity loss**，不是 emergency hard projection。论文不能把 hard projection 写成已经贡献了性能的实证模块，它只是未触发的数值/物理 safeguard。

## 6. v16.6 attribution 修复

### 6.1 同一 checkpoint epoch

- 三个 arm 强制使用 main 选中的 checkpoint epoch；
- 不允许消融选择自己的 best epoch；
- 若服务器没有精确 epoch checkpoint，只重训缺失 arm 到该 epoch，并每 epoch 保存。

### 6.2 同一场景与逐场景配对

新的 natural diagnostic 写入：

- sampled scene index SHA-256；
- scene index；
- scene-level overall/OBS/NEU/PRIO 8 s error；
- scene-level mass ratio、violation mass、exact squared excess。

归因脚本拒绝 epoch、采样哈希、场景数、decoder family 不一致的报告。

### 6.3 检查真实优化目标

mass envelope 的主判据改为：

1. exact probability-weighted squared excess 的相对下降；
2. soft-envelope violation probability mass 的相对下降；
3. OBS/overall 非劣；
4. emergency envelope p99 合法。

mean path ratio 保留为诊断指标，不再作为主 gate。

### 6.4 continuation gate 与论文证据分离

- `pass=true`：两个组件在单 seed、对齐协议下方向有效且无明显伤害，可以继续 transport/planner/closed-loop 收集证据；
- `paper_claim_ready=false`：直到至少三个独立 seed、配对置信区间和 held-out closed-loop 完成。

这避免了“为了运行完整 pipeline 而修改门槛”的问题。

## 7. 已修复的训练效率问题

v16.5 已经有：冻结 graph、`no_grad`、四个 horizon 共用距离张量、diversity 时间降采样、零权重分支跳过、每两 epoch 验证一次。v16.6 又修复：

1. **冻结未使用的 legacy natural dense head**：typed decoder 为兼容旧 checkpoint 保留的 `natural_decoder.head` 不参与 forward，但之前仍被 DDP/optimizer 跟踪。
2. **AdamW 仅接收 `requires_grad=True` 参数**：不再为永久冻结模块建立 optimizer bookkeeping。
3. **冻结在 DDP 和 optimizer 构造前完成**：避免每 batch 的 unused-parameter graph traversal。
4. **static DDP**：永久冻结的 natural stage 使用 `find_unused_parameters=False`、`static_graph=True` 和 `gradient_as_bucket_view=True`，旧 PyTorch 自动回退。
5. **component-neutral checkpoint score**：只使用所有 arm 共有的 trajectory、OBS minADE 和 branch minADE，不再把被消融的 loss 项写入 best-checkpoint 选择。

本地环境不能代表服务器 GPU，因此不虚构“加速几倍”。代码会打印 trainable/total 参数和 DDP policy；服务器应记录 epoch wall time、samples/s、GPU utilization 和 peak memory，对比 v16.5 才能形成速度结论。

## 8. 下游 full pipeline 中发现的评价泄漏

旧 full pipeline 在同一 `VAL_CACHE` 上：

1. 扫描 BCOT risk budget；
2. 选择 calibrated budget；
3. 报告同一批场景上的 selector/mechanism 指标。

更严重的是：旧 `25_verify_mechanism_effect.py` 与 `30_diagnose_bcot_result.py` 会用 `calibration.selection_metrics` 覆盖方法比较中的指标。这会把 calibration-set performance 当成 held-out performance，可能高估 AUPRC、NCF recall、accepted rate、false-safe gain 和降低 fallback 的能力。

v16.6 已修复：

- validation dataset index `i % 2 == 0`：只做 calibration；
- `i % 2 == 1`：只做 held-out method comparison；
- 每个 metrics row 记录 modulo、remainder、scene count 和 index SHA-256；
- mechanism gate 必须验证两个 deterministic partition 互斥；
- calibration JSON 只提供 operating point，不得提供最终性能；
- held-out budget 必须与 calibrated budget 完全一致；
- 新增 scene-level proposal/certificate coverage，区分低 accepted/high fallback 到底来自候选覆盖不足，还是 certificate 过度拒绝。

这个修复可能使最终数字下降，但只有修复后的数字才可以写入论文。

## 9. 目前哪些算法设计有效、无效或未证实

### 9.1 有效或方向正确

- typed causal dynamics natural decoder：相对 analytic basis 有大幅、稳定学习增益；
- yaw reference-frame 修复：误差降到数值量级；
- graph 全程冻结与 deterministic feature：使 natural 消融真正集中于 decoder；
- mass-aware root identity：与 OPR retained-mass 语义一致，当前数据支持其作用；
- source-adaptive OBS capacity：有方向性证据，需 aligned paired report；
- fail-fast natural gates：阻止不合格 natural foundation 污染下游结果。

### 9.2 已否定或不应恢复

- v16.4 fixed 8 s endpoint ball：虽然削尾，但明显损伤 OBS，且忽略中间偏离；
- v16.4 compound `no_effectiveness_loss` 归因：一次关闭多个 loss，不是合法单因素实验；
- 将 logged replay 写成已识别因果效应；
- 用 calibration-set metrics 作为最终 selector/mechanism 结果。

### 9.3 当前未证实

- emergency projection 对性能的贡献：本轮未触发；
- conformal natural-set coverage：代码没有实现完整 coverage protocol；
- planner、selector、RootTransport、BCOT 的当前版本增益：完整 pipeline 尚未执行；
- reactive-agent 下的 burden transfer 因果结论；
- SOTA 或 CCF-A 级性能。

## 10. v16.4 与 v16.5 论文修改复核

v16.5 相对 v16.4 的核心理论方向基本正确：

1. fixed endpoint trust ball 改为 probability-mass-aware multi-horizon root identity；
2. 删除没有被合法消融支持的 explicit OBS-gain bundle；
3. 区分 soft semantic envelope 和 emergency guard；
4. 强调 logged data 只能支持 model-based intervention proxy；
5. 强调有限 soft burden cost 不能替代 non-coercive feasibility certificate。

这些修改让核心 idea 更集中：论文不是普通社会性 cost shaping，而是识别并阻止 ego 通过迫使 priority-relevant agent 承担高 burden 来实现表面安全。

但上传的 v16.5 TeX 存在必须修复的代码—论文不一致：

- 主文公式没有写代码实际使用的 probability floor；
- 主文少了 interior-margin normalization `(rho-eta)/(1-eta)`；
- 附录仍保留 v16.4 endpoint trust loss，并引用已删除的公式标签；
- conformal expansion 写得像已实现，当前代码和实验不支持；
- qualitative results 使用“demonstrate”，但闭环结果尚不存在；
- component evidence protocol 没有明确相同 checkpoint epoch 和相同场景。

`interactive_planning_v16_6_revised.tex` 已逐项修复。它以本轮上传的 v16.5 TeX 为直接基线，不是从记忆重建。

## 11. 为 CCF-A 级别证据仍需解决的代码/算法缺陷

CCF 目录是 venue 推荐分类，不提供自动驾驶统一数值录用线。以下是本项目的内部 promotion targets，不是官方标准。

### 11.1 RootTransport / BCOT AUPRC

建议目标：held-out conflict-conditioned RootTransport AUPRC 与 BCOT false-safe AUPRC 均达到约 0.70，并报告 class prevalence 与 bootstrap CI。

当前风险：

- 之前存在 calibration leakage，已修复；
- direct-root positive 稀少、easy negative 过多时，AUPRC 会被标签分布和 hard-negative 质量限制；
- candidate-level certificate 可能绕过 primitive/root mechanism，形成高 AUPRC 但弱机制证据。

完整 pipeline 后应先检查：direct root label prevalence、conflict-conditioned support、root assignment ADE、hard-negative slice、candidate certificate 与 root transport 的独立增益。没有这些诊断前不应继续加权重。

### 11.2 NCF recall、accepted rate 与 fallback

建议目标：NCF recall/precision 约 0.60、accepted candidate rate 至少约 0.20、normal fallback 不高于约 0.15。

当前风险：

- 多层 hard gate 可能产生乘法式拒绝；
- proposal bank 本身可能没有合格候选；
- certificate false rejection 与 proposal coverage 过去没有分开统计；
- `stop_like` fallback 会掩盖“证书为空”和“没有安全候选”的不同原因。

v16.6 新增 scene-level coverage 分解。后续只在数据证明瓶颈后修改：

- 若 `AnyNCFSceneRate` 低：扩展 proposal lattice/route-conditioned proposals；
- 若 AnyNCF 高但 `AnyAcceptedNCF` 低：改善 calibration、uncertainty 和 hard gate；
- 若 accepted 高但 false-safe 高：增强 RootTransport/BCOT hard negatives；
- 若 accepted 与 safety 都好但 progress 低：修改 certified frontier 内的 utility ranking，而不是放宽证书。

### 11.3 closed-loop collision/offroad/progress

建议：collision、offroad、wrong-way 对强 baseline 非劣；progress 下降不超过约 1--2 个百分点；至少三个 seeds、固定场景配对 CI。

当前代码支持 Waymax overlap/offroad/route/progression/kinematic 等指标，但当前没有本轮 full result。v1.3.1 `sdc_paths` 应用于 wrong-way 与 route progression。首次 full pipeline 结果应作为机制诊断，不可直接宣称 SOTA。

### 11.4 FSR/HBCR 与因果有效性

建议：reactive/human-audited stress set 上 FSR/HBCR 相对下降约 25--30%，paired CI 不跨 0。

当前最大缺口不是阈值，而是 simulator-agent protocol：logged replay 无法证明 ego action 导致其他 agent 让步。Waymax 提供 log playback 与 IDM，但当前代码没有实现并验证多反应类型的 actor wrapper。论文应继续使用 model-based proxy 表述，直到增加至少 IDM、cooperative 和 aggressive/less-yielding stress policies，或人工审核的干预对。

## 12. 本次交付的执行原则

1. 先运行 v16.6 aligned attribution；
2. 只有 development `pass=true` 才运行完整 pipeline；
3. 不允许 `ALLOW_QUALITY_GATE_FAILURE=1` 用于论文实验；
4. full pipeline 使用新的 v16.6 输出根目录，显式导入已验证 v16.5 natural checkpoint，避免修改后代码与旧 provenance 混在一个实验目录；
5. calibration 与 held-out evaluation 必须互斥；
6. single-seed full pipeline 只用于暴露后续瓶颈，正式论文结论仍需多 seed。
