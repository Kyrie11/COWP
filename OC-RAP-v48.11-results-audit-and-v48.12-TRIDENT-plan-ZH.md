# v48.11 CASTER完整结果审计与v48.12 TRIDENT设计

## 1. 结论

v48.11没有通过Natural gate，因此没有stress closed-loop是正确行为。失败并非单一阈值问题，而是由三个层次共同造成：

1. Near恢复候选组内排序没有形成稳定正相关；Contact虽有改善，但仍低于可投稿的策略级门槛。
2. Contact收益识别较强，但harmful-vs-dead证据在scene-disjoint verify fold上接近随机甚至反转，导致fit规则无法迁移。
3. 旧的绝对macro占比约束没有考虑teacher正机会本身高度集中于macro 5，因而混淆了“数据可用机会集中”和“模型额外走捷径”。

此外发现两个影响算法归因的工程错误：Stage-T checkpoint选择使用了非条件式策略风险；消融调度器会在第一个失败的`wait`处退出。

v48.12 TRIDENT不降低precision、harm、recall和scene-disjoint verify门槛，而是修复训练—评估合同，并分别深化恢复排序、伤害证据和支持集证书。

## 2. v48.11主要结果

### 2.1 主实验

| Variant | Regime | Candidate positive AUC | Candidate harm AUC | Candidate rank corr | Group top-1 corr | Policy-top1 benefit AUC | Policy-top1 harm AUC | Verify选择 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.6775 | 0.6116 | 0.1099 | 0.0135 | 0.6562 | 0.5866 | 0 |
| Balanced | Contact | 0.8385 | 0.5501 | 0.1967 | 0.0735 | 0.7994 | 0.4898 | 0 |
| Precision | Near | 0.6761 | 0.6074 | 0.1303 | -0.0243 | — | — | 0 |
| Precision | Contact | 0.8188 | 0.5490 | 0.2273 | 0.0208 | — | — | 0 |

三个seed下：

- Balanced Near top-1 corr均值-0.0100，范围-0.0311～0.0135；
- Balanced Contact均值0.0870，范围0.0735～0.1026；
- Precision Near均值-0.0186，范围-0.0319～0.0004；
- Precision Contact均值0.0480，范围0.0208～0.0777。

因此，Recovery Set Tournament对Contact产生真实改善，但仍未达到内部建议的0.20；Near没有解决。

### 2.2 Near中存在有价值的near-miss

Balanced Near的一条规则迁移到verify后：

- 选择10个group；
- point precision 0.70；
- harmful selection为0；
- positive recall 0.28；
- 平均exact-teacher advantage +0.1457；
- 但10个动作全部属于macro 5；
- precision Wilson LCB90只有0.4417，harmful UCB90仍为0.2129。

这说明Near并非完全没有可执行信号，但统计支持仍不足，而且旧macro约束会直接拒绝它。

### 2.3 Contact的主要失败是fit→verify迁移

Balanced Contact的代表性fit规则：

- 29个选择；
- precision 0.586；
- harmful rate 0.172；
- recall 0.515；
- 平均teacher advantage +0.051。

迁移到verify后，代表性规则会下降到约：

- precision 0.25；
- harmful rate 0.45；
- 平均teacher advantage转负。

Contact的benefit AUC足够高，真正瓶颈是harmful tail与scene-fold迁移，而不是继续增加普通candidate classifier。

## 3. v48.11哪些设计有效

### Recovery-only Set Tournament：部分有效，必须保留并深化

它彻底替换了旧value rank，而不是继续用小残差修补错误排序。Contact三seed top-1全部为正，是v48.11最明确的算法进步。

不足在于原set likelihood对teacher可接受集合内部基本不施加排序梯度，Near中大量近似并列候选使top-1仍接近随机。v48.12增加exact-PCD gap-weighted recovery pair监督：只有teacher差距超过阈值的候选对才参与排序，近似并列不强制排序。

### Policy-first/no-fallback：有效且必须保留

训练、校准和运行时均先由Preference确定唯一候选，再判断证据；证据不足时abstain，禁止落到未训练的runner-up。该合同没有导致Natural gate失败，应继续保留。

### Proper ordered three-state evidence：结构正确，但仅局部NLL不足

有序三类概率比两个独立BCE更符合harmful/dead/beneficial标签结构；Contact benefit识别保持较强。但局部NLL不直接优化跨scene的harm排序，导致Contact harm AUC仍约0.54，verify甚至反转。

v48.12保留有序三类NLL，同时加入regime内跨group的benefit和harm pairwise AUC surrogate，重点提高harmful-vs-nonharmful尾部排序。

## 4. 工程错误

### 4.1 Stage-T early stopping语义错误

v48.11的训练脚本设置了`CONDITIONAL_RECOVERY_RANKING=true`，却没有设置`PREFERENCE_CONDITIONAL_MODE=true`。因此checkpoint指标仍加入：

- nominal false-switch；
- harmful switch；
- admission相关项。

但Recovery Set Tournament只比较recovery，并且分数组内中心化，必然至少有一个recovery为正。使用nominal false-switch选择checkpoint在语义上错误。

v48.12已显式设置`PREFERENCE_CONDITIONAL_MODE=true`，Stage-R checkpoint只按条件式恢复排序regret和top-1选择。

### 4.2 消融调度器提前退出

旧脚本在`set -e`下依次执行：

```bash
wait "$p0"
wait "$p1"
```

C-balanced失败后脚本立即退出，造成C-precision和D-precision缺失，无法完整归因。v48.12记录每个任务状态并继续全部8个任务；只有全部完成才生成`ABLATIONS_COMPLETE.json`。

### 4.3 绝对macro约束与数据机会分布冲突

训练正机会中macro 5约占88%。旧约束要求selected最大macro share≤0.85，即使策略完全复制oracle机会分布也可能失败。

v48.12同时报告：

- raw selected macro share；
- oracle-positive macro share；
- selected excess share。

主实验按“相对oracle机会分布的额外集中度”约束，而不是删除macro检查。precision、harm、recall、正平均收益和scene-disjoint verify要求完全不变。

## 5. v48.12 TRIDENT

TRIDENT：Teacher-gap Recovery tournament with Inter-regime Discriminative Evidence and Normalized-support cerTification。

### 5.1 Teacher-gap Recovery Tournament

- Recovery-only set self-attention保留；
- exact PCD materially ordered pair使用gap-weighted margin；
- Near/Contact分别保留不同tie epsilon；
- near tie不制造伪winner；
- 清晰候选对直接优化真实top-1。

### 5.2 Bipolar Inter-regime Evidence

冻结Stage-R后，在每个regime内部收集policy-top1候选：

- beneficial vs nonbeneficial进行pairwise排序；
- harmful vs nonharmful进行更高权重pairwise排序；
- 与proper ordered NLL联合训练。

它针对v48.11最严重的Contact harm inversion。

### 5.3 Normalized-support Certificate

选择分布若与teacher正机会分布同样集中，不再被错误视为额外捷径；只有超出oracle-positive concentration的部分才计入约束。

这不是放宽Natural gate。模型仍必须满足：

- fit/verify最小选择数；
- precision Wilson LCB；
- harmful group exposure UCB；
- conditional harmful-switch UCB；
- positive recall；
- 平均exact teacher advantage为正；
- scene完全不泄漏。

## 6. 三个regime投稿目标差距

### Safe

v48.11结果包没有新的大样本paired Safe闭环，不能判断collision/offroad非增、route progression、NUP、jerk和yaw-rate置信区间。TRIDENT不改变Safe默认nominal保护逻辑；Safe需要独立扩大paired scenes。

### Near-contact

当前仍未获得合法coverage。距离投稿准备门槛主要是：

- top-1 corr约0，目标至少0.20；
- benefit AUC约0.66，目标至少0.70；
- harm AUC约0.58，需稳定≥0.60；
- verify recall为0，目标至少0.35；
- 无有效precision LCB和harm UCB。

若TRIDENT能把现有Near near-miss从LCB 0.44提高到≥0.60，并增加独立scene支持，才有机会验证clearance、TTC、DRS、PCD、FRA和ODG目标。

### Contact

Contact candidate benefit AUC约0.83已是优点，但：

- top-1 corr均值0.05～0.09，目标至少0.20；
- harm AUC约0.54；
- fit→verify harm明显反转；
- 当前没有合法coverage。

因此secondary overlap、recontact、stable stop、post-contact clearance、失控位移和route-rejoin目标仍不能声称实现。

## 7. 不能承诺“保证通过gate”

Natural gate是安全证书，不能通过修改阈值保证通过。v48.12修复的是造成错误拒绝或错误训练的根因；是否通过必须由scene-disjoint实验决定。如果D-full只改善fit不改善verify，应停止继续调阈值，转向数据合同或跨fold evidence稳健性。
