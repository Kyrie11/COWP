# v48.9 PACER完整结果审计与v48.10 COPE设计

## 1. 最终判断

本轮上传的v48.9主实验、三seed复校准和policy-top1 conformal诊断均可用于分析。主实验完成性审计通过，但四组消融并不完整，且发现一个会破坏最终消融归因的阶段结构错误。

`runs/ocrap_v48_9_pacer_proxy_4801/NEXT_COMMANDS.txt`没有生成不是漏写文件，而是controller的预期行为：只有某个variant同时通过Near与Contact的scene-disjoint Natural gate，才会写出`chosen_base_run.txt`和`NEXT_COMMANDS.txt`。本轮balanced、precision在Near和Contact均为`valid_for_deployment=false`，因此不应运行stress closed-loop。

总体上，PACER证明了两个局部方向：

1. intervention-aware preference能显著抑制无机会误切换和有害候选高排；
2. policy-top1对齐的certificate训练能提高收益识别AUC，尤其是Contact。

但PACER没有解决两个核心瓶颈：

- recovery候选之间的条件排序依然接近随机，Contact三seed均略为负；
- 连续relative-gain回归不能可靠区分beneficial/dead-zone/harmful，导致harm AUC偏低、conformal半径接近整个teacher优势范围、最终零coverage。

因此v48.10不降低Natural gate，而是把论文核心方法重构为：

> **先在恢复选项空间内学习条件偏好，再对冻结策略实际选出的候选学习有序三状态执行证据。**

新方法命名为 **OC-TRAC-COPE：Conditional Option Preference with Monotone Ordinal Evidence**。

---

## 2. 为什么没有NEXT_COMMANDS.txt

controller的实际逻辑是：

1. 分别训练balanced与precision；
2. 对Near和Contact做fit/verify calibration；
3. 只有同一variant的两个regime均`valid_for_deployment=true`，才进入`valid_candidates`；
4. 从有效候选中选出`chosen_base_run`；
5. 写入`NEXT_COMMANDS.txt`。

本轮`screening_status.json`中的`valid_candidates`为空，因此没有文件是正确安全行为。若强行手工运行closed-loop，只会绕过Natural gate，无法把结果归因于经过验证的策略。

---

## 3. v48.9主实验结果

### 3.1 Preference层

| Variant | Regime | group top-1 corr | acceptable top-1 acc | 无机会false switch | harmful ranked switch | 正机会恢复激活率 |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.0148 | 0.5918 | 0.1481 | 0.0711 | 0.5102 |
| Balanced | Contact | -0.0126 | 0.5758 | 0.1233 | 0.0546 | 0.3182 |
| Precision | Near | -0.0007 | 0.5918 | 0.1111 | 0.0521 | 0.4286 |
| Precision | Contact | -0.0251 | 0.5606 | 0.1267 | 0.0492 | 0.2727 |

解释：

- PACER确实让模型更少在无机会group中选择恢复动作；
- harmful recovery被排到第一的比例也明显降低；
- 但是这些改进主要来自“把nominal排高”，并不等价于学会“多个recovery中哪个最好”；
- Contact条件恢复排序依然为负，说明Stage P把“是否干预”和“恢复选项排序”混合在同一个目标中，抑制干预的梯度压过了恢复候选间的细粒度偏好。

### 3.2 Policy-top1 Relative gain层

| Variant | Regime | policy-top1 benefit AUC | policy-top1 harm AUC | gain MAE | candidate benefit AUC | mean regret |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.6752 | 0.5580 | 0.2090 | 0.7160 | 0.1083 |
| Balanced | Contact | 0.7422 | 0.4932 | 0.2224 | 0.7676 | 0.2028 |
| Precision | Near | 0.6698 | 0.5668 | 0.2045 | 0.7091 | 0.1201 |
| Precision | Contact | 0.7320 | 0.5102 | 0.2242 | 0.7530 | 0.2118 |

解释：

- policy-aligned训练保留了可用的收益信号；
- Contact benefit AUC达到约0.73–0.74，说明“冻结策略选中的候选是否可能有益”并非完全不可学；
- harm AUC只有约0.49–0.57，无法承担安全证书；
- MAE约0.20–0.22，而exact teacher advantage的离散跨度可达约±0.62，连续回归明显向零收缩。

### 3.3 Natural gate层

四个`variant × regime`组合均为：

- `verify_selected=0`；
- `positive_recall=0`；
- `precision_lcb90=null`；
- `harmful_selected_ucb90=1.0`；
- `valid_for_deployment=false`。

这不是“Natural gate过严”这一单一问题。Near的某些fit规则可以在8个动作上得到0.75点估计precision，但转移到verify后precision降至约0.38–0.50、recall仅约0.08–0.12，而且动作几乎全部来自macro 5。Contact转移更差，部分near-miss规则在verify中会出现大多数动作有害、平均teacher advantage为负。

---

## 4. 三seed稳定性

| Variant | Regime | candidate AUC均值 | top-1 corr均值 | top-1范围 | harm AUC均值 | verify选择 |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.7161 | 0.0137 | -0.0125～0.0389 | 0.5630 | 0 |
| Balanced | Contact | 0.7945 | -0.0057 | -0.0126～-0.0002 | 0.5183 | 0 |
| Precision | Near | 0.6999 | 0.0049 | -0.0155～0.0310 | 0.5588 | 0 |
| Precision | Contact | 0.7755 | -0.0162 | -0.0251～-0.0029 | 0.5264 | 0 |

结论：

- candidate-level信号跨seed存在；
- recovery top-1排序没有跨seed稳定提升；
- Contact不是偶然一个split失败，而是三个seed均略微反向；
- 所有seed均零coverage，说明不能通过更换calibration seed解决根因。

---

## 5. Conformal诊断为什么仍失败

即使把作用域改成`policy_top1`，单侧overprediction quantile仍然为：

| Variant | Near | Contact |
|---|---:|---:|
| Balanced | 0.6057 | 0.6241 |
| Precision | 0.5999 | 0.6236 |

这接近exact teacher advantage的完整范围。根本原因是目标分布与模型假设不匹配：

- 预测`pred_adv`集中在零附近，标准差约0.04；
- teacher advantage标准差约0.30，且呈明显三模态；
- Near与Contact中大量样本恰好为0，同时存在约±0.622的边界质量；
- 平滑连续回归在不确定样本上以均值最小化损失，必然向0坍缩；
- conformal只能诚实反映大残差，不能创造分类可分性。

因此conformal本身不是主失败源；主失败源是continuous delta evidence模型不适合当前teacher标签结构。继续调`alpha`或temperature只会改变保守程度，不会改善benefit/harm辨识。

---

## 6. v48.9消融的有效证据与工程错误

### 6.1 消融不完整

上传包只有：

- A balanced；
- B balanced；
- C balanced。

缺失：D balanced以及A/B/C/D precision。没有`ABLATIONS_COMPLETE.json`，不能声称完成四组消融。

### 6.2 Stage P到Stage C结构丢失

A/B/C的Stage-C日志显示：

```text
Stage P preference hidden = 32
Stage C preference hidden = 128
```

加载时`direct_preference_context_adapter`发生shape mismatch，Stage-P学到的adapter被丢弃。最终A/B结果完全相同，因此A/B最终性能不能用于判断intervention-aware preference是否有效。

主实验没有这个错误，因为主环境持续传入`PREFERENCE_CONTEXT_HIDDEN=32`。不过消融错误说明旧阶段脚本缺少严格架构合同，必须修复后才能发表消融结论。

### 6.3 可用的Stage-P审计证据

尽管最终消融失效，Stage-P审计本身仍可比较：

| 设计 | Near top-1 | Contact top-1 | Near false switch | Contact false switch | Near harmful rank | Contact harmful rank |
|---|---:|---:|---:|---:|---:|---:|
| 旧uniform/all-candidate | 0.0979 | 0.1117 | 0.6481 | 0.7067 | 0.2133 | 0.2295 |
| intervention-aware set preference | 0.0148 | -0.0126 | 0.1481 | 0.1233 | 0.0711 | 0.0546 |

这说明v48.9设计不是完全无效，而是优化了错误的混合目标：它成功学会“不轻易干预”，却牺牲了“recovery选项之间怎么排序”。

### 6.4 可用的Certificate证据

在可用balanced结果中，policy-aligned certificate相对all-candidate certificate：

- Near policy-top1 benefit AUC约0.551→0.571；
- Contact约0.716→0.760；
- candidate benefit AUC也有提升；
- harm AUC没有稳定改善；
- 最终仍没有非零verify coverage。

因此应保留“policy-induced candidate distribution”的训练思想，但要替换连续回归目标。

---

## 7. 哪些v48.9设计保留、深化或放弃

### 保留并深化

1. **两阶段冻结训练**：避免证书梯度破坏排序。
2. **nominal-relative低容量上下文**：减少严重度和macro捷径。
3. **policy-top1对齐的证书采样**：训练分布更接近部署分布。
4. **scene-disjoint fit/verify Natural gate**：继续作为闭环授权条件。
5. **宏观集中度、support、precision/harm置信界**：不能为了产生coverage而降低。

### 修改后保留

1. **partial-label acceptable set**：保留teacher模糊性，但只用于recovery选项空间；nominal不参与“哪个recovery更好”的排序。
2. **无机会group监督**：不再用nominal压制recovery排序，而是以较低权重训练“least-bad recovery”条件顺序；是否执行交给证书。

### 不再作为主路径

1. **连续relative-gain Gaussian回归**：只保留为消融基线。
2. **conformal作为默认主风险源**：在evidence本身不可分时无效；以后仅作为已具备判别力模型的附加校准诊断。
3. **单一标量同时承担排序和准入**：彻底分开。

---

## 8. 三个regime投稿目标差距

### Safe

本轮上传没有新的Safe paired closed-loop结果，因此不能更新以下目标：collision/offroad非增、paired 95% CI、route progression、NUP、jerk/yaw-rate p95和intervention episode。此前8-scene probe只能证明nominal lock和零干预，仍不是paper-ready Safe结果。

### Near-contact

当前零Natural-gate coverage，因此下列闭环目标均未验证：

- collision相对下降15%–25%；
- clearance p05提高0.20 m；
- TTC p05提高0.20 s；
- near-contact exposure下降15%；
- DRS提高8个百分点；
- PCD提高0.03；
- FRA下降30%；
- ODG下降25%；
- harmful switch控制在5%–10%。

离线差距主要是：top-1 corr约0.00而内部准备目标为≥0.20；policy-top1 harm AUC仅约0.56；verify recall为0而目标≥0.35；fit规则跨scene和macro迁移失败。

### Contact

Contact candidate benefit AUC较好，但条件恢复top-1三seed仍略为负，harm证据接近随机，零coverage。因此secondary overlap、recontact、overlap duration、stable-stop、time-to-stable-stop、post-contact clearance、uncontrolled displacement和route-rejoin均无闭环证据。

最核心的Contact差距不是候选完全无信号，而是：

- `beneficial recovery存在`与`具体选哪个recovery`没有被分别建模；
- harm与dead-zone的continuous target不可分；
- fit规则在verify中的错误候选比例过高。

---

## 9. v48.10 COPE算法

### 9.1 Conditional Option Preference

Stage P只回答：

> 在已经考虑恢复动作的条件下，哪一个recovery option最好？

实现：

- nominal从条件偏好loss和conditional rank margin中移除；
- exact teacher-PCD定义可接受recovery集合；
- 最大化可接受集合概率质量；
- 直接最小化exact expected recovery regret；
- 正机会group权重为1；无机会/有害group使用默认0.30权重，只学习least-bad recovery顺序；
- 是否应该留在nominal不由Stage P决定。

这避免v48.9中nominal抑制梯度淹没Contact选项排序。

### 9.2 Monotone Ordinal Evidence

Stage E只回答：

> 冻结Preference选择的top-1 recovery，相对nominal属于beneficial、dead-zone还是harmful？

模型输出两个有序累计logit：

```text
nonharm_logit = center + width / 2
benefit_logit = center - width / 2
```

由于`width>=0`，结构上保证：

```text
P(beneficial) <= P(non-harm)
P(harmful) = 1 - P(non-harm)
```

训练重点放在frozen policy-top1候选，使用focal ordinal BCE；所有候选只作为低权重正则。这样直接适配teacher的三状态结构，避免连续回归向零坍缩。

### 9.3 执行证书

Calibration和runtime统一使用：

- opportunity：`P(beneficial)`；
- harm：`P(harmful)`；
- evidence score：`P(beneficial)-P(harmful)`；
- conditional recovery rank margin；
- fit/verify precision和harm置信界；
- positive recall；
- selected macro share和支持数。

Natural gate门槛不降低。v48.10的目标是提高可分性，使模型在原门槛下产生可信coverage，而不是“首先让所有gate机械通过”。

### 9.4 工程隔离

- `training.strict_init_prefixes`：Stage E若不能完整加载Stage-P preference adapter则立即失败；
- `STAGE_ARCHITECTURE.json`固定hidden width、delta mode和条件排序模式；
- staged completion marker记录两个checkpoint SHA256；
- 八个消融任务各自生成不可变`TASK_COMPLETE.json`；
- 消融脚本支持断点续跑，缺任意任务时拒绝写总完成标记；
- calibration、checkpoint选择、离线评估与closed-loop使用同一ordinal evidence语义。

---

## 10. v48.10实验判定顺序

第一步不看Natural gate，先看两个学习层是否成立：

### Stage P诊断门

- Near/Contact conditional recovery top-1 corr均≥0.10；
- positive recovery top-1 accuracy均≥0.60；
- Contact不再跨seed持续为负；
- expected recovery regret低于v48.9。

### Stage E判别门

- Near policy-top1 benefit AUC≥0.70；
- Contact≥0.75；
- Near/Contact policy-top1 harm AUC均≥0.60；
- evidence score与teacher状态方向一致。

### Natural gate

只有前两层有判别力后，再要求：

- Near/Contact均有非零verify selection；
- paper-ready最终目标：precision LCB90≥0.60、recall≥0.35、conditional harmful UCB趋近≤0.10；
- macro share受控；
- 通过后才运行stress closed-loop。

---

## 11. 结论边界

COPE具有可写成论文核心方法的清晰结构：**条件恢复排序与单调有序执行证据的策略诱导解耦**。它针对v48.9结果中已被实验证实的两个失败源，而不是继续叠加阈值或普通pairwise loss。

但目前仅完成代码和本地测试，尚未在WOMD/Waymax/A30上运行。能否达到CCF-A内部门槛，必须由主实验、完整四组消融、三seed以及最终paired closed-loop共同证明。
