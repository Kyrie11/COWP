# OC-RAP v48.4结果复核与新v48.5 ECPR设计

## 1. 复核范围与结果完整性

本报告仅使用本轮重新上传的v48.4代码、主实验包、消融包和三seed复校准包。上一版v48.5草案及其结论不作为本轮设计依据。

需要先区分“服务器实验可能已经结束”和“本次上传压缩包中能被审计的内容”。本次包内可确认：

- 三个proxy calibration seed（4801、4802、4803）的Near/Contact校准JSON完整；
- `A_src_reference`消融的两个variant训练摘要和screening状态完整；
- 主实验日志包含7个完整epoch，随后进入epoch 8后停止记录；包内没有主实验`train_summary.json`、`best.pt`和主校准JSON；
- `B_zi_nasc_only`只有balanced分支的3个完整epoch，epoch 4中断；C、D消融没有出现在压缩包内。

因此，可以可靠判断v48.4的跨seed离线表现和A基线，但无法把B/C/D模块逐项归因成完整消融结论。新代码加入完成标记、checkpoint哈希和完成性审计，避免以后再把中间checkpoint与最终checkpoint混用。

## 2. v48.4核心结果

### 2.1 三seed结果

| Variant | Regime | Candidate AUC均值 | Top-1相关性均值 | Top-1范围 | Harm AUC范围 | Verify选择 |
|---|---|---:|---:|---:|---:|---:|
| Balanced | Near | 0.7764 | 0.0134 | -0.0443–0.0851 | 0.5456–0.5611 | 0 |
| Balanced | Contact | 0.8246 | -0.0864 | -0.0927–-0.0775 | 0.5216–0.5435 | 0 |
| Precision | Near | 0.7775 | 0.0303 | -0.0254–0.1151 | 0.5438–0.5612 | 0 |
| Precision | Contact | 0.8085 | -0.0813 | -0.0943–-0.0724 | 0.5120–0.5346 | 0 |

结论：候选级恢复信号稳定存在，但Contact组内排序在三个seed、两个variant中全部反向；Near只在seed 4802出现局部正相关，跨划分不稳定。Harm head仍接近随机，所有规则均退化为零选择。

### 2.2 与A_src_reference比较

A基线结果：

| Variant | Near AUC | Near top-1 | Contact AUC | Contact top-1 |
|---|---:|---:|---:|---:|
| Balanced | 0.7113 | 0.0363 | 0.7833 | -0.0354 |
| Precision | 0.7227 | 0.0514 | 0.7974 | -0.0800 |

v48.4完整配置相较A基线显著提高了候选AUC，尤其是Balanced的Near和Contact；但没有稳定提高策略top-1。Balanced Contact从-0.035恶化到约-0.086，Precision Contact基本仍为负。说明v48.4新增模块更擅长区分“可能有恢复价值的候选”，没有形成可靠的“同组最佳恢复候选”排序。

## 3. 六个问题的当前状态

1. **候选特征包含恢复信号：成立且较v48.3更稳定。** Near约0.75–0.79，Contact最高0.88。
2. **组内排序接近随机或反向：仍成立。** Contact在所有seed均为负。
3. **Harm head接近随机：仍成立。** AUC约0.51–0.56。
4. **联合规则选择0动作：仍成立。** Natural gate正确拒绝了不可验证策略。
5. **Candidate AUC与policy top-1脱节：仍成立且Contact最明显。**
6. **没有学会候选相对nominal和其他候选的最优关系：仍成立。**

## 4. 工程问题

### 4.1 训练目标与校准目标不一致

v48.4训练和validation checkpoint指标使用可微soft shared-success近似；calibration使用exact OC-MERO q表选择一个全局共享option，再在`m_star`上计算hard DRS和PCD。多root、多option的Contact组中，两种teacher可能给出不同甚至相反排序。这是高AUC、负top-1的重要解释。

### 4.2 排序与准入在运行时没有完全分离

v48.4虽然在loss中提出DRA-RCD，但calibration和selector的top-1仍按`pred_adv`（value）执行；同一标量同时承担候选AUC、组内排序、相对nominal准入和风险阈值。一个标量很难同时校准这四种任务。

### 4.3 Harm head标签合同漂移

Near/Contact validation/test的`harm_proxy`退化，而train存在非零harm标签。继续把Harm head作为主要准入概率，会学习train中特有严重度或macro捷径。v48.4的Harm AUC已经给出直接证据。

### 4.4 Checkpoint和实验完成状态不可验证

旧脚本没有不可变checkpoint清单，multi-seed可以读取仍在更新的`best.pt`；主实验、消融可能并发占用同一GPU；输出目录也缺乏互斥。新版本增加输出锁、GPU锁、SHA256、`TRAINING_COMPLETE.json`和完成性审计。

### 4.5 结果字段错误

候选选择脚本读取`harmful_rate_selected`，而calibration实际输出`harmful_selected_rate`。当将来出现有效规则时，这会影响variant排序。已修复。

## 5. v48.4修改的有效性判断

### 有效或有正向证据

- **ZI-NASC零初始化修复**：相比v48.3，warm start不再注入随机集合残差，这是确定的工程改进。
- **候选级表征能力**：完整配置相较A基线提高Near/Contact candidate AUC，说明set context、soft监督或额外策略目标至少有部分表征收益。
- **严格best-metric存在性检查**：不再静默回退total loss，工程隔离正确。
- **scene-disjoint proxy split和三seed独立目录**：没有目录覆盖，暴露了Near不稳定和Contact系统性反向。
- **Natural gate**：在风险与排序未验证时保持nominal，保护作用有效。

### 无效、作用不大或未被完整证明

- **DRA-RCD**：没有恢复Contact top-1；原因是其teacher仍为soft近似，且运行时top-1仍用value。
- **Harm head强化**：Harm AUC接近随机，较大loss权重没有形成可迁移风险估计。
- **Soft opportunity/harm labels**：只能缓和margin附近跳变，不能解决exact teacher排序、正候选之间偏好和标签合同漂移。
- **minibatch pseudo-environment GroupDRO**：B/C/D消融不完整；稀疏环境可能放大单group噪声。本轮默认关闭，而不是继续把它作为主贡献。
- **传统pairwise/listwise/top-rank的叠加**：过去多轮已尝试，仍通过同一个value标量优化，不能解决任务冲突。

## 6. 新v48.5：OC-TRAC-ECPR

### 6.1 Exact Policy Contract

训练、validation early stopping、calibration和诊断统一使用同一exact teacher-PCD：

1. teacher OC-MERO产生q；
2. 按root概率成功质量和value tie-break选择一个全局共享option；
3. 在teacher `m_star`上计算hard DRS；
4. 使用相同PCD公式得到候选目标。

### 6.2 独立Preference head

Value head只建模候选相对nominal的收益均值和方差；独立Preference head只负责同组候选排序。新head最后一层严格零初始化，所以载入v48.1/v48.4 checkpoint时初始rank与旧value排序完全一致，再通过exact preference监督逐步修正。

### 6.3 Confidence-Paced Preference Regret

只对teacher-best与runner-up差距足够大的组施加强监督；near-tie按exact PCD差距降权。损失包括：

- teacher-best对其他恢复候选的best-vs-rest margin；
- 正机会组的expected exact-teacher regret；
- 只在正恢复组计算的排序指标。

它直接优化“同一scene-time group里选谁”，而不是再次优化候选二分类。

### 6.4 Distributional gain admission

令候选与nominal的预测增益为：

`Delta_mu = mu_candidate - mu_nominal`

`Delta_sigma² = sigma_candidate² + sigma_nominal²`

由此直接计算：

- `P(Delta >= positive_gain)`作为opportunity；
- `P(Delta <= -negative_gain)`作为harm/downside。

主策略不再依赖随机Harm head。旧opportunity/harm heads保留为低权重辅助诊断，便于做ablation。

### 6.5 Risk-focused checkpoint

early stopping优化Near/Contact中更差的：

`positive-group regret + 0.35 × harmful selected-candidate rate + 0.15 × false-intervention rate`

只统计所有group的平均regret容易被大量无机会group稀释；新指标不能被always-nominal策略伪装成好结果。

## 7. 对三个regime指标的预期作用

### Safe

ECPR不主动扩大Safe干预。Safe继续由nominal-preservation和独立非劣闭环验证约束，目标是collision/offroad不增加、route progression下降不超过0.5%、NUP和动态舒适性非劣。

### Near-contact

Exact排序和delta admission应优先改善：positive recall、minimum clearance、p05 TTC、DRS、FRA和ODG。Near目前主要问题是seed不稳定，confidence-paced teacher gap可减少near-tie换序。

### Contact

Contact的主要缺陷是系统性负top-1。独立Preference head与exact PCD统一首先应把top-1相关性转正；只有排序通过后才可能改善secondary overlap、recontact、stable-stop、time-to-stable-stop和失控位移。模型仍应表述为contact-conditioned counterfactual recovery，而非真实碰撞动力学控制。

## 8. CCF-A内部目标

首轮开发门：Near/Contact top-1相关性均>0.10，positive-group top-1 regret显著下降，并至少产生非零verify selection。

投稿准备门：

- Near/Contact top-1相关性>=0.20，理想>=0.30；
- verify precision Wilson LCB90>=0.60；
- positive recall>=0.35，主结果>=0.50；
- harmful selection UCB90<=0.10，主结果<=0.05；
- candidate AUC不较v48.4下降超过0.03；
- 三seed中至少2个通过同一Natural gate；
- Safe严格非劣，Near/Contact闭环相对最强外部baseline取得scene-paired显著改善。

这些是内部工程目标，不是CCF-A会议的官方录用线，也不能保证投稿结果。
