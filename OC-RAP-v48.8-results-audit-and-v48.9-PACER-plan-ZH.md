# OC-RAP v48.8完整结果审计与v48.9 PACER设计

## 1. 审计范围与结论

本轮联合审计了：

- v48.8 SCOPE主实验；
- 四组消融、balanced/precision共8个任务；
- Safe paired non-inferiority结果；
- 训练损失、checkpoint指标、calibration和Natural gate代码路径。

结论不是“阈值太严”，而是两个学习对象仍未形成足够可靠的证据：

1. Preference能提取候选级顺序信号，但无机会group中的恢复误切换仍很高，真正的policy top-1不足；
2. Relative gain主要在所有候选上学习，而Natural gate只评估Preference最终挑出的一个候选，训练分布和部署分布不一致；
3. Split-conformal在所有候选对上拟合残差，半径被大量未使用候选支配，导致所有候选机会概率为0、伤害概率为1；
4. 因此Natural gate全部拒绝，没有Near/Contact stress closed-loop结果，不能声称SCOPE提升了闭环性能。

Natural gate必须保持不变。下一版的目标是让算法产生足够证据通过gate，而不是降低gate制造coverage。

## 2. v48.8主实验表现

| Variant | Regime | Candidate positive AUC | Candidate harm AUC | Group top-1 corr | Acceptable top-1 acc | Verify selections |
|---|---|---:|---:|---:|---:|---:|
| Balanced | Near | 0.7303 | 0.6292 | 0.0527 | 0.5918 | 0 |
| Balanced | Contact | 0.7850 | 0.5322 | 0.1251 | 0.5758 | 0 |
| Precision | Near | 0.6432 | 0.6247 | 0.1627 | 0.5510 | 0 |
| Precision | Contact | 0.7652 | 0.5743 | 0.1846 | 0.5909 | 0 |

主要判断：

- Contact和部分Near仍有候选级恢复信号；
- Precision的top-1已接近0.20，但acceptable accuracy仍低于0.60；
- Balanced排序更弱；
- Stage C gain discrimination未达到预设门槛；
- 所有certificate均为0选择，Natural gate未通过。

训练日志进一步显示，无机会group中的rank false-switch rate约0.53–0.71，harmful top-1约0.28–0.44。这说明Preference不仅要在正机会group中“选对”，还必须在多数无机会group中把nominal排在第一。

## 3. 四项SCOPE设计是否发挥作用

### 3.1 无冲突、包含nominal的set preference：没有按预期生效

消融结果：

| Variant | Reference Near/Contact | Conflict-free Near/Contact |
|---|---:|---:|
| Balanced | 0.0221 / 0.0787 | 0.0058 / 0.0579 |
| Precision | 0.0478 / 0.1024 | 0.0058 / 0.0579 |

问题不在“set-valued”思想本身，而在具体目标：

- 旧实现使用对acceptable set的均匀KL，强制集合内候选具有相同logit；
- 无正机会group把nominal和dead-zone recovery都视为可接受；
- 这在训练语义上允许不必要恢复动作，与低intervention/低false-switch目标冲突。

因此这项设计的出发点成立，但损失定义不成立。

### 3.2 纯相对、低容量Preference adapter：方向合理，当前实现未证明有效

相对特征可以减少severity/macro捷径，但旧adapter仍有196,297个可训练参数，面对少量有效group并不算真正低容量。消融B没有提升top-1，说明仅排除绝对特征不足以解决监督目标错误和无机会误切换。

值得保留的是：

- candidate-minus-nominal；
- candidate-minus-recovery-mean/max；
- 冻结继承Preference、只训练零初始化残差。

需要进一步降低容量并修改目标，而不是回到共享NASC或绝对特征。

### 3.3 稳健Relative gain：存在粗粒度信号，但没有学到部署候选证书

Candidate AUC仍有信息，但旧Stage C对所有恢复候选等权回归。部署时Natural gate只检查Preference挑出的一个候选，因此模型优化的样本分布和真正使用的样本分布不同。

这会出现：

- 全体候选AUC尚可；
- policy top-1候选的gain误差仍大；
- near-miss precision和harm控制不足；
- 最终仍然0 coverage。

### 3.4 Split-conformal Certificate：本轮没有发挥正作用

主实验单侧overprediction quantile约为0.57–0.61，而teacher advantage本身约在±0.62范围内。所有校准行最终表现为：

```text
opportunity = 0
harm = 1
```

原因是conformal residual用所有候选对拟合，而不是只用冻结Preference策略实际产生的top-1候选。大量不会被执行的候选支配了半径。

所以本轮conformal并没有带来可用安全证书，反而使Natural gate在进入精度、召回和support检验之前就全部为空。它不应继续作为主实验默认风险源。

## 4. 是否提升了闭环结果

没有Near/Contact stress closed-loop被合法执行，因此没有证据证明v48.8提升了：

- Near collision、clearance、TTC、near-contact exposure、DRS、PCD、FRA、ODG；
- Contact secondary overlap、recontact、overlap duration、stable stop、失控位移和route rejoin。

Safe probe只有8个paired scenes：collision/offroad、bounded NUP和intervention与nominal完全相同。它证明nominal lock生效，但route progression、jerk p95和yaw-rate p95缺失，且样本量不足，`paper_safe_claim_ready=false`。

## 5. 投稿目标差距

### Safe

尚不能形成论文级非劣结论。需要至少100个paired scenes，并真实输出route progression、jerk p95、yaw-rate p95。当前8场景结果只可作为smoke test。

### Near-contact

当前0执行，因此所有恢复型目标均未验证。最先需要达到的离线先决条件是：

- top-1 corr稳定超过0.10，向0.20靠近；
- 无机会false-switch明显下降；
- policy-top1 positive/harm AUC形成可分性；
- verify产生非零选择；
- precision LCB90、conditional harm UCB、recall和macro-share同时通过。

### Contact

Precision top-1 corr达到0.1846，已经接近0.20，但acceptable accuracy不足，证书完全失败。Contact最有希望优先通过Preference门，但不能用高candidate AUC代替相对nominal证书。

## 6. 工程与算法修复

### 6.1 Partial-label set mass

新损失最大化acceptable set的总概率质量，不再强制集合内均匀分布。它保留teacher模糊性，同时允许模型在可接受候选内部形成有依据的偏好。

### 6.2 无机会group nominal-only

无材料恢复机会时，唯一部署目标为nominal：

- dead-zone recovery只施加较弱intervention-cost margin；
- harmful recovery施加强margin；
- 不再把“无明显伤害”等同于“值得执行”。

### 6.3 Policy-induced certificate learning

Stage C冻结Preference，然后：

- 对Preference实际选中的candidate施加高权重relative-gain回归；
- 对该候选施加正收益/伤害三状态sign监督；
- 所有其他候选只保留低权重正则。

这让训练分布和Natural gate部署分布一致。

### 6.4 Policy-top1 conformal scope

可选conformal只使用每个group中冻结Preference产生的一个候选拟合残差。主实验仍使用`direct_delta`，conformal作为独立消融，只有证明不再饱和且产生verify coverage后才能恢复为主方法。

### 6.5 诊断增强

新增：

- policy-top1 positive AUC；
- policy-top1 harm AUC；
- policy-top1 gain MAE；
- non-positive false-switch rate；
- harmful ranked-switch rate；
- positive-group activation rate；
- probability-bound失败时仍输出near-miss frontier。

这使下一轮可以区分：Preference失败、gain失败还是统计证书失败。

## 7. v48.9核心idea与novelty

新版本命名为：

> **OC-TRAC-PACER：Policy-Aligned Candidate Evidence for Recovery**

核心idea不是另一个候选分类器，而是：

> 在ambiguity-aware的候选集合中学习恢复偏好，再在该偏好策略诱导出的候选分布上学习相对nominal的可执行证据。

论文贡献可组织为：

1. intervention-aware partial-label recovery preference；
2. nominal-dominant no-op supervision；
3. frozen-policy-induced relative-gain certification；
4. held-out statistical Natural gate with explicit abstention。

它与已经失败的共享NASC、联合rank/gain梯度、强化Harm head、GroupDRO、阈值放宽和手工rescue不同。

## 8. 如何判断v48.9是否有效

第一层，Preference：

- Near/Contact top-1 corr均≥0.10；
- acceptable top-1 accuracy≥0.60；
- non-positive false-switch≤0.45，并相较v48.8显著下降。

第二层，Relative gain：

- policy-top1 positive AUC：Near≥0.70、Contact≥0.75；
- policy-top1 harm AUC≥0.60；
- gain MAE下降。

第三层，Certificate：

- verify选择非零；
- precision LCB90、conditional harmful UCB、recall、support和macro-share全部通过原Natural gate；
- 只有这一层通过才允许stress closed-loop。

不能保证一次训练就全部通过，但新实验能够明确告诉下一轮应该继续修Preference、gain还是证书，不再只得到“0选择”这一条信息。
