# COWP v16.3→v16.4 重新审计与 v16.5 优化报告

## 1. 审计范围与结论边界

本报告完全放弃上一轮未经过完整代码—结果对照的 v16.4 判断，重新以以下材料为证据：

- v16.3 代码与 `cowp_v16_3_natural_recovery_v9labels_seed2026`；
- 当前 v16.4 代码 `COWP(2).zip`；
- v16.4 主 natural recovery；
- v16.4 三组 natural ablation 与 `natural_component_attribution_gate.json`；
- v16.3/v16.4 训练日志、配置、checkpoint history 与 learned-natural diagnostics。

本地环境不能复现服务器 GPU 全量训练，因此 v16.5 的结论分为两层：

1. **已经证实的工程结论**：由代码、日志、配置、结果 JSON 和回归测试直接支持；
2. **待新实验验证的算法结论**：v16.5 的设计方向合理且接口/梯度测试通过，但不能在没有新 GPU 结果时声称性能已提高。

## 2. v16.3 与 v16.4 的真实结果差异

| 版本/消融 | 8 s weighted ADE | OBS 8 s | NEU 8 s | PRIO 8 s | yaw error | residual endpoint p99 |
|---|---:|---:|---:|---:|---:|---:|
| v16.3 main | 1.1779 m | 2.6402 m | 1.0618 m | 0.1103 m | 0.1750 rad | 45.2182 m |
| v16.4 main | 1.2339 m | 2.9174 m | 1.0487 m | 0.1119 m | ~0 | 20.0000 m |
| v16.4 no-effectiveness bundle | 1.2159 m | 2.8724 m | 1.0381 m | 0.1027 m | ~0 | 20.0000 m |
| v16.4 no OBS capacity | 1.2599 m | 3.0522 m | 1.0465 m | 0.0996 m | ~0 | 20.0000 m |
| v16.4 no fixed trust | 1.1700 m | 2.6159 m | 1.0578 m | 0.1087 m | ~0 | 46.7090 m |

由此得到三个不能混淆的事实：

- v16.4 的 yaw reference-frame 修复是有效的工程修复；
- v16.4 main 相比 v16.3 并未提高预测质量，退化主要来自新加入的约束/损失，而不是 decoder 不再学习；
- 无固定 trust 的 v16.4 保留了 yaw 修复，并在 overall/OBS 上略优于 v16.3，说明固定 endpoint trust 是当前主要性能冲突源。

## 3. attribution gate 为什么失败

v16.4 gate 的五个核心判断为：

- new loss improves OBS：false；
- new loss improves overall：false；
- OBS capacity improves OBS：true；
- trust reduces residual tail：true；
- trust does not harm OBS：false。

数值上：

- main 相对 `no_effectiveness_loss`：OBS **恶化 0.04494 m**，overall **恶化 0.017996 m**；
- OBS capacity 相对 no-capacity：OBS **改善 0.134783 m**；
- fixed trust 将 endpoint p99 降低 **26.70899 m**，但 OBS **恶化 0.301442 m**。

不过，“new loss 无效”不能按字面直接成立，因为对应消融存在严重归因错误。

## 4. 会污染归因的工程问题

### 4.1 `no_effectiveness_loss` 不是单变量消融

v16.4 的 `train_cowp_v16_no_effectiveness_loss.yaml` 同时关闭了：

- OBS gain；
- neutral/priority preservation；
- kinematic velocity/yaw；
- control smoothness；
- mode usage；
- residual trust region。

因此它比较的是“一整组 regularization bundle”，不是 effectiveness loss。当前结果只能说明：

> v16.4 main 的整组新增损失不优于被整体删除后的模型。

不能说明其中每一项都无效，也不能为任何单项建立因果归因。

### 4.2 natural graph 并没有按声明全程冻结

配置写有 `freeze_graph_during_natural: true`，但启动参数将 graph warmup 设为 2。训练日志明确显示：

- epoch 0：`graph_frozen=True`；
- epoch 2：`graph_frozen=False`。

这造成：

1. natural 归因不再只比较 decoder/loss，四组实验都在重新学习共享 graph；
2. graph 反向传播占用大量时间和显存；
3. 不同 loss 通过 graph 改变共享表示，使组件归因更难解释。

### 4.3 冻结参数时 graph 仍可能处于 train/dropout 模式

训练循环调用 `model.train()` 后，冻结 graph 的 dropout 会重新开启。即使参数不更新，四组实验得到的 graph feature 仍含随机扰动。这会增加单 seed、小差值消融的不确定性。

### 4.4 主实验与消融运行参数没有完全统一

v16.4 主实验和消融脚本使用了不同的 DataLoader worker/prefetch 配置。它不一定改变期望值，但会改变吞吐、batch timing 和系统资源竞争；若结合非完全确定性算子，会进一步降低可重复性。

### 4.5 单 seed 不能支撑小幅归因

0.018–0.045 m 的差异可能小于训练方差。v16.4 只能把 0.1348 m 的 OBS capacity 效应视为“当前 seed 的正证据”，不能当作论文级确定结论；最终仍需至少 3 seeds 和置信区间。

## 5. v16.4 四项论文理论增强的复核

### 5.1 有限 soft burden penalty 不等于 non-coercive feasibility guarantee

**保留。** 这是优化与约束之间的逻辑区别，不依赖本轮指标。有限权重的 penalty 可以被 imitation/progress 等收益抵消，因此 COWP 的 novelty 应放在结构化证书或硬可行性筛选，而不是“再加一个 courtesy cost”。这使核心 idea 更集中。

### 5.2 logged WOMD 上的 `do(...)` 改称 model-based intervention proxy

**保留。** 当前数据和 learned response model 不能单独识别真实因果效应。论文应把 logged replay 定位为机制诊断，把更强因果表述留给 reactive-agent protocol、受控 stress set 和人工审核。

### 5.3 natural root 需要 identity/trust preservation

**原则保留，实现否定。** same-root retained mass 若允许 decoder 把 root 任意移动几十米，就失去可解释性；但 v16.4 的固定 8 秒 endpoint 球把“较大但合理的交互响应”和“身份漂移”一并裁掉，OBS 退化 0.3014 m，说明实现过于粗糙。

v16.5 改为：

- 1/3/5/8 秒多时间尺度 envelope；
- 对概率质量较大的 root 施加更强软身份约束；
- 对所有 mode 仅保留更宽的 emergency hard envelope；
- 语义约束和数值安全约束分离。

这样直接对应 OPR 的 retained probability mass，而不是把所有 root 视为同等重要的几何点。

### 5.4 retained mass 下界与 burden 上界

**理论合理，暂未验证。** 它有助于把 epistemic uncertainty 纳入证书，但当前代码/实验尚未形成 calibration coverage 证据。可保留为方法或后续实验目标，不应写成已被本轮证明的贡献。

## 6. 当前有效、无效和未验证的算法设计

### 6.1 已有证据支持

1. **typed causal dynamics natural decoder**：v16.3/v16.4 相对 analytic basis 均有显著学习增益；
2. **yaw reference-frame 修复**：将明显错误降至数值误差量级；
3. **OBS source-adaptive capacity**：唯一相对干净的 v16.4 单因素对照，OBS 改善 0.1348 m；
4. **natural quality gate**：正确阻止存在语义/归因问题的模型进入昂贵 full pipeline。

需要增强的部分：OBS capacity 不应只靠永久固定的大预算；后续可根据 conflict intensity、relative geometry 和 source uncertainty 进行条件化，但必须用独立消融证明。

### 6.2 当前设计应废弃或重做

1. **固定 endpoint trust ball**：保留 tail-control 需求，废弃其作为 root identity 的具体实现；
2. **v16.4 effectiveness/preservation bundle**：当前整体无正证据，且消融错误；v16.5 主训练先关闭显式 OBS gain 和 prior-preservation 项，再逐项添加；
3. **重复的 kinematic velocity/yaw loss**：在轨迹由同一 dynamics integration 生成时，它们主要检查实现一致性，不应持续占用主训练目标；保留为 diagnostic gate；
4. **只看 minADE 的 natural decoder 评价**：必须同时看概率质量、root identity、diversity 和 physical feasibility。

### 6.3 当前没有新证据

- planner；
- selector/calibrator；
- RootTransport/BCOT 在修复后的 natural foundation 上的效果；
- Waymax 闭环性能。

v16.4 没有运行这些阶段，不能沿用 v16.2 smoke 结果来判断新版本有效性。

## 7. 距离 CCF-A 级论文还缺什么

CCF-A 没有统一的数值录用门槛；真正需要的是方法新颖性、机制证据、强基线、严谨 protocol 和统计显著性。建议把以下指标作为项目内部 promotion gate，而非“官方标准”：

- 至少 3 个独立 seed；
- interaction-heavy 固定测试集每 seed 至少 1,000 场，另有更广泛 WOMD 测试；
- paired bootstrap 95% confidence interval；
- collision/offroad/wrong-way 相对最强匹配 baseline 不显著恶化；
- progress 下降不超过 1–2 个百分点；
- normal fallback 不高于 10–15%；
- accepted candidate rate 至少约 20%；
- conflict-conditioned RootTransport AUPRC 与 BCOT false-safe AUPRC 目标至少约 0.70；
- NCF recall 目标至少约 0.60，同时 precision 不低于约 0.60；
- reactive/human-reviewed stress set 上 FSR/HBCR 相对下降 25–30%，且置信区间不跨 0。

这些阈值是工程决策线，不应写成社区官方门槛。

## 8. v16.5 算法修改

### 8.1 Probability-mass-aware multi-horizon root envelope

对每个 mode 的 raw residual path，在 1/3/5/8 秒计算相对 source envelope 的最大比值。软 identity loss 使用 detached mode probability 加权，并设置小概率 floor：

- 约束真正承载 OPR probability mass 的 roots；
- 防止模型只降低违规 mode 的 logit 来逃避几何约束；
- 不再强迫每一个低概率 mode 都落在同一固定 endpoint 球中。

### 8.2 Soft semantic envelope 与 emergency hard envelope 分离

- soft envelope：OBS/NEU/PRIO 默认 20/8/6 m；
- emergency envelope：48/16/12 m；
- hard projection 只保证不发生数值/物理逃逸；
- soft envelope 才用于 root identity learning。

这些是 dev 配置，必须在 calibration split 上冻结后再进入正式 test。

### 8.3 精简主 loss

v16.5 暂时关闭：

- explicit OBS gain；
- neutral/priority preservation；
- velocity/yaw consistency loss；
- 训练内重复的 base-effectiveness pair comparison。

保留：

- multi-horizon trajectory likelihood/minADE；
- source-adaptive capacity；
- small control smoothness；
- mode usage；
- probability-mass-aware root identity loss。

外部 2,000-scene diagnostic 仍是 effectiveness 的权威判定。

## 9. 训练加速修改

### 9.1 graph 全 natural 阶段冻结并 `no_grad`

默认 `natural_graph_unfreeze_epoch=-1`：

- graph 始终 `eval()`；
- 不存 graph backward activations；
- 避免 epoch 2 后昂贵且污染归因的共享 encoder 更新。

### 9.2 合并多 horizon 距离计算

旧实现分别为 1/3/5/8 秒展开 prediction×GT 距离。v16.5 只展开一次，再用 cumulative sum 取各 horizon，保持数学等价。

### 9.3 diversity 时间降采样

80-step mode diversity 默认 stride=4，仅用于多样性 regularizer，不改变主 trajectory loss。

### 9.4 删除无用计算分支

权重为 0 时跳过 kinematic、effectiveness 和其他对应 tensor 构造，而不是“算完再乘 0”。

### 9.5 降低验证和消融总成本

- validation 从每 epoch 一次改为每 2 epoch 一次；
- attribution 从 3 个 auxiliary full runs 减为 2 个严格单因素 full runs；
- 所有 arm 使用同一 init/seed/DDP/workers/prefetch/precision/schedule。

无法在本地无 GPU 环境给出可信的端到端加速倍数。服务器上应记录：samples/s、GPU utilization、peak memory、epoch wall time，并用相同硬件和 batch 做 v16.4/v16.5 profile。

## 10. 新 gate 逻辑

### Natural effectiveness gate

同时检查：

- 8 秒绝对质量和相对 analytic basis 增益；
- NEU/PRIO 不退化；
- finite/kinematic/mode-use；
- probability-mass-weighted soft-envelope ratio 与 violation rate；
- projected emergency-envelope p99。

### Attribution gate

只保留两个真正独立的核心问题：

1. OBS capacity 是否改善 OBS 且不破坏 NEU/PRIO；
2. mass-aware envelope 是否降低 probability-weighted identity violation，同时不造成不可接受的 OBS/overall 退化。

## 11. 代码验证状态

交付前执行：

- Python compileall；
- 所有 shell 脚本 `bash -n`；
- pytest 全量回归。

最终结果见交付说明。代码正确性测试不等同于新算法性能结论；只有服务器新训练通过两个 gate 后，才能继续完整 pipeline。
