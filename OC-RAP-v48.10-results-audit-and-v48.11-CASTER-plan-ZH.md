# v48.10 COPE完整实验审计与v48.11 CASTER设计

## 1. 实验完整性

本轮主实验完成性审计通过，balanced与precision均存在完整训练摘要、固定checkpoint、Near/Contact calibration结果和Natural-gate输出。四组消融A/B/C/D、两个variant共8个任务均存在`TASK_COMPLETE.json`，因此可以进行算法归因。

Natural gate未通过是正确结果，不是controller漏生成文件。两个variant都没有同时获得Near和Contact的可部署规则，所以没有生成`NEXT_COMMANDS.txt`，也不应执行stress closed loop。

## 2. v48.10主实验

| Variant | Regime | Candidate benefit AUC | Candidate harm AUC | Group top-1 corr | Positive top-1 acc | Positive regret | Policy-top1 benefit AUC | Policy-top1 harm AUC | Verify selected |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.661 | 0.626 | -0.003 | 0.571 | 0.135 | 0.623 | 0.602 | 0 |
| Balanced | Contact | 0.834 | 0.610 | 0.012 | 0.561 | 0.200 | 0.808 | 0.537 | 0 |
| Precision | Near | 0.687 | 0.608 | 0.013 | 0.571 | 0.158 | 0.666 | 0.581 | 0 |
| Precision | Contact | 0.814 | 0.547 | 0.001 | 0.636 | 0.176 | 0.768 | 0.494 | 13 |

Precision Contact虽然出现13个verify动作，但并未通过Natural gate：precision为0.308、LCB90为0.146；6个动作有害，harmful rate为0.462、UCB90为0.675；positive recall仅0.121；平均teacher advantage为-0.180；11/13动作来自macro 5。它说明模型可以制造coverage，但coverage方向错误，不能进入闭环。

## 3. v48.10哪些设计有效

### 3.1 Monotone Ordinal Evidence有效，尤其对Contact收益识别

消融C仅加入ordinal evidence，保持A的Preference：

- Balanced Contact candidate benefit AUC：0.724 → 0.806；harm AUC：0.514 → 0.583。
- Precision Contact candidate benefit AUC：0.670 → 0.826；harm AUC：0.526 → 0.597。
- Near的benefit/harm AUC也有小幅提升。

这证明exact teacher advantage的三状态结构确实更适合ordinal建模，而不是连续delta回归。该设计应保留并深化。

### 3.2 两阶段冻结仍然正确

Stage P和Stage E没有共享梯度，证书训练不会直接重写排序网络。当前失败可以分别归因到排序与证据，而不是联合训练的梯度冲突。这是后续论文方法可解释性的基础，应保留。

### 3.3 Policy-top1 evidence sampling方向成立

Contact policy-top1 benefit AUC达到0.768–0.808，显著高于随机，说明证书在冻结策略实际诱导的候选分布上训练，比所有候选等权训练更贴近部署问题。该方向应继续保留。

### 3.4 Natural gate有效

Natural gate拒绝了低precision、高harm、负平均优势和单macro集中的规则。不能通过放宽阈值制造“通过”。

## 4. v48.10哪些设计无效

### 4.1 Conditional Option Preference没有真正替换旧排序

B消融的top-1仅从A的近零/负值变成近零：

- Balanced Near/Contact：0.011/-0.012 → -0.003/0.012。
- Precision Near/Contact：-0.004/-0.009 → 0.013/0.001。

同时，non-positive false switch从约0.13–0.15上升到0.37–0.53，harmful ranked switch从约0.06–0.07上升到0.14–0.19。

根因在模型实现：最终排序仍是`rank_base + residual`。`rank_base`来自candidate-level value head，而历史实验已反复证明它的candidate AUC较好、group top-1错误。Stage P只训练低容量residual，无法推翻冻结的错误基线。当前Conditional Preference不是独立策略，而是错误value排序的微调项。

### 4.2 两个独立BCE不能充分区分harm与dead-zone

Ordinal结构保证benefit不超过non-harm，但损失仍是benefit BCE和harm BCE的平均。它能改善“是否像benefit”，却没有直接最大化三状态联合似然。结果是Contact benefit AUC较强，policy-top1 harm AUC仍只有0.49–0.54。

### 4.3 单一evidence expert混合了Near与Contact边界

Near与Contact的teacher分布和难例不同。共享evidence adapter必须同时拟合两个regime，容易优先学习Contact显著收益，却无法学习Near的细微收益和Contact的harm/dead边界。

### 4.4 Macro捷径依然存在

Precision Contact唯一非零规则中macro 5占84.6%。训练正机会group也高度集中于macro 5。现有温和逆频率采样不足以消除这种捷径。

## 5. 工程错误：训练策略与部署策略不一致

Stage E训练时，每个group先由冻结Preference选择一个top-1 recovery，再对这个候选训练evidence。

但v48.10 calibration和runtime selector采用：

1. 先用opportunity/harm阈值过滤候选；
2. 再在过滤后的候选中选最高rank。

如果Preference真实top-1证据不足，系统会落到rank-2或rank-3候选。Stage E从未针对这种fallback分布训练，可能正是Precision Contact出现13个高harm动作的原因。

v48.11统一为policy-first、no-fallback：

1. 在物理可执行候选中选Preference top-1；
2. 计算它相对runner-up的rank margin；
3. 只检查这个候选的evidence；
4. 不通过就abstain，不允许落到runner-up。

该语义已同步到calibration JSON、selector和closed-loop配置。

## 6. 三个regime投稿目标差距

### Safe

本轮没有新的paired Safe closed-loop包，因此无法更新collision/offroad非增、paired CI、route progression、NUP、jerk/yaw-rate和intervention结论。Safe必须单独使用足够scene的paired nominal对照验证，不依赖Near/Contact Natural gate。

### Near-contact

当前没有有效verify coverage，闭环目标全部未验证。离线差距：

- top-1 corr约0，而内部准备门槛至少0.20；
- policy-top1 benefit AUC 0.62–0.67，目标至少0.70；
- harm AUC 0.58–0.60，勉强接近最低诊断线，但不能产生规则；
- recall为0；precision LCB和harm UCB无法成立；
- macro集中明显。

因此collision下降、clearance/TTC p05、DRS/PCD、FRA/ODG均没有可归因证据。

### Contact

Contact候选收益识别是当前优点：candidate benefit AUC约0.81–0.83，policy-top1 benefit AUC约0.77–0.81。但：

- top-1 corr约0；
- harm AUC约0.49–0.54；
- 唯一coverage规则平均teacher advantage为负；
- harmful UCB和macro concentration均严重超标。

因此secondary overlap、recontact、stable stop、time-to-stop、post-contact clearance、uncontrolled displacement和route-rejoin均不能进入闭环验证。

## 7. v48.11 CASTER

CASTER全称Conditional Attention Set Tournament with Evidence Routing。

### 7.1 Recovery-only Set Tournament

新排序器不再使用`rank_base`，而是直接替换旧排序：

- 对每个recovery使用candidate-minus-nominal、recovery mean/max相对token；
- 在同一scene-time group内使用小型self-attention交互；
- nominal固定为0且不进入tournament；
- recovery分数组内中心化，只学习相对次序；
- 参数量低，避免在有限正机会group上过拟合。

这直接针对candidate AUC与group top-1脱节。

### 7.2 Policy-conditioned Regime Evidence

Stage E冻结整个set tournament。Near和Contact使用独立evidence expert，并额外输入：

- 候选Preference分数；
- 该候选相对runner-up的rank gap。

证书学习的对象因此是“这个regime下，冻结策略选出的候选及其排序置信度”，而不是混合分布中的任意候选。

### 7.3 Proper Ordered Three-state NLL

保留单调ordinal参数化，但使用合法三类概率：

- harmful；
- dead-zone；
- beneficial。

损失改为class-weighted三类NLL，其中harmful权重最高。它直接优化harm-vs-dead边界，而不是两个独立BCE。

### 7.4 Policy-first no-fallback Certificate

训练、calibration、离线评估和closed-loop采用同一候选分布。不再因evidence gate改变Preference策略。

## 8. 下一轮如何判断设计是否有效

按层判断，不应只看Natural gate：

1. Stage T：Near、Contact top-1 corr均应超过0.10，且positive regret下降；若仍约0，说明set tournament仍未学会候选竞争。
2. Stage E：Near benefit AUC≥0.70、Contact≥0.75；两者harm AUC≥0.60。若benefit强而harm弱，应继续优化证据/数据漂移，而不是排序。
3. Natural gate：必须出现非零verify coverage、正平均teacher advantage、合理macro分散，并满足原有置信界。
4. 三seed：至少两个seed方向一致，Contact不能再次持续负相关。
5. 只有controller生成`NEXT_COMMANDS.txt`才运行Near/Contact closed loop。

v48.11理论上解决了v48.10最关键的两个归因错误，但真实A30/WOMD实验前不能保证Natural gate一定通过。
