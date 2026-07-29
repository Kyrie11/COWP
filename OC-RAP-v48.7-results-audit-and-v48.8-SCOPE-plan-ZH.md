# OC-RAP v48.7 完整结果审计与 v48.8 SCOPE 设计

## 1. 审计范围与结论

本轮审计基于以下上传内容：

- 当前 v48.7 代码；
- `ocrap_v48_7_spire_proxy_4801` 主实验；
- v48.7 四组消融；
- Safe nominal-locked non-inferiority probe。

本轮没有上传 v48.7 的 4801/4802/4803 固定 checkpoint 复校准结果，因此不能判断 v48.7 的跨 seed 稳定性。当前结论只针对 seed 4801 主实验与四组消融。

核心判断：

1. **Stage P 没有稳定学会“同组候选应该选谁”。** 候选级排序相关性为正，但真正执行所需的 group top-1 相关性仍略为负。
2. **Stage C 没有学会“是否值得执行”。** 风险区分只达到中等水平，所有 Natural-gate 规则仍选择 0 个动作。
3. **v48.7 的两个出发点各有局部证据：** staged optimization 对 Contact 有帮助；set-valued target 在联合训练中有帮助。但两者直接组合后反而退化，说明实现中存在目标冲突、checkpoint 噪声和证书建模问题。
4. Safe probe 只包含 8 个 nominal-locked scene，不能支撑论文级 Safe 非劣结论。

因此，v48.7 不能进入 Near/Contact stress closed loop；Natural gate 拒绝是正确的保护结果。

---

## 2. Stage P：是否学会“选谁”

### 2.1 主实验结果

| Variant | Regime | Candidate rank corr | Group top-1 corr | Acceptable top-1 acc | Strict top-1 acc | Positive regret |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.1462 | -0.0155 | 0.5306 | 0.4490 | 0.1430 |
| Balanced | Contact | 0.1187 | -0.0117 | 0.6364 | 0.5909 | 0.1927 |
| Precision | Near | 0.1576 | -0.0214 | 0.5306 | 0.4082 | 0.1210 |
| Precision | Contact | 0.1339 | -0.0132 | 0.6212 | 0.5758 | 0.1936 |

这组结果说明：

- 模型能在所有候选上学到弱的单调排序信号，candidate-level rank correlation 为 0.12–0.16；
- 但真正重要的组内第一名仍然选错，group top-1 correlation 在四种组合下全部为负；
- Near acceptable-set accuracy 只有约 53%，接近困难二分类水平；
- Contact acceptable-set accuracy约 62%–64%，但正机会组的平均 regret 仍接近 0.19，无法支撑策略执行。

所以 Stage P 只学到了“候选整体大概怎样排序”的弱信号，没有可靠学会“本组第一名是谁”。

### 2.2 消融归因

| 消融 | Balanced Near | Balanced Contact | Precision Near | Precision Contact |
|---|---:|---:|---:|---:|
| A joint single-winner | 0.0030 | -0.0694 | -0.0374 | -0.0529 |
| B staged single-winner | -0.0178 | 0.0198 | -0.0286 | 0.0484 |
| C joint set-valued | 0.0105 | **0.0771** | **0.0290** | **0.0685** |
| D full SPIRE | -0.0155 | -0.0117 | -0.0214 | -0.0132 |

可得出三个可信结论：

- **Staging 有局部价值：** B 相比 A，使 Contact top-1 从负值转为小幅正值。
- **Set-valued supervision 有价值：** C 是四组中最好的排序结果，Near和Contact均为正。
- **v48.7 的组合方式无效：** D 将 staged 与 set-valued 合并后重新退化为负，说明两项思路不是简单叠加关系。

### 2.3 为什么 D 退化

代码审计发现以下原因：

1. **Set target 与 single-winner margin 冲突。** Near 的 tie epsilon 为 0.025，而部分 best-vs-rest margin 从约 0.01 就开始强制排序。PCD 差距位于 0.01–0.025 的候选，一项损失要求“等价”，另一项损失要求“严格分出高低”。
2. **只在正机会group训练偏好。** 无机会和有害group没有被充分监督“nominal应该排第一”，导致策略在不该干预的group中仍可产生高恢复排名。
3. **参数量相对监督规模过大。** Stage P约有80万可训练参数，而exact deployable positive group只有362个，极易过拟合。
4. **绝对特征泄漏严重度与macro捷径。** Preference上下文仍包含绝对candidate表征，模型可以利用train特有严重度或macro分布，而不是学习候选间不变关系。
5. **checkpoint由稀疏最差fold支配。** 某些fold正机会过少，单个group即可改变worst-fold指标，best epoch不稳定。
6. **相同指标会覆盖较早best。** v48.7使用`<=`判断改善；指标完全相等时，后续epoch仍覆盖best checkpoint。B组Stage P存在指标多epoch恒定的情况。

---

## 3. Stage C：是否学会“是否执行”

### 3.1 主实验结果

| Variant | Regime | Positive AUC | Risk-harm AUC | Legacy Harm-head AUC | Verify selections |
|---|---:|---:|---:|---:|---:|
| Balanced | Near | 0.6871 | 0.6083 | 0.5789 | 0 |
| Balanced | Contact | 0.7707 | 0.5490 | 0.5252 | 0 |
| Precision | Near | 0.6624 | 0.6142 | 0.5763 | 0 |
| Precision | Contact | 0.7625 | 0.5630 | 0.5234 | 0 |

Stage C存在一定粗粒度收益信号，但没有形成可部署证书：

- Contact positive AUC约0.76–0.77，说明仍能识别部分“可能有收益”的候选；
- downside/harm AUC只有约0.55–0.61；
- 所有fit规则均未通过，verify选择全部为0；
- 因此不能将零执行解释为harmful switch达标，它只表示模型完全abstain。

### 3.2 Near-miss规则暴露的问题

最接近通过的规则仍存在严重缺陷：

- Balanced Near：33个fit选择，precision 0.182，harmful rate 0.152，teacher advantage均值为-0.048，macro最大占比0.939；
- Balanced Contact：31个选择，precision 0.387，harmful rate 0.323，teacher advantage均值为-0.079，macro最大占比0.871；
- Precision Near：仅8个选择，precision 0.375，虽然样本内无harm，但UCB90仍为0.253，macro最大占比1.0；
- Precision Contact：20个选择，precision 0.400，harmful rate 0.150，teacher advantage均值接近0，macro最大占比0.85。

这说明准入失败不是“门槛太严”这么简单。模型选出的高置信候选本身仍混入大量无收益和有害动作，而且高度依赖单一macro。

### 3.3 Stage C的算法和工程缺陷

1. **学习方差崩溃。** v48.7 heteroscedastic NLL在训练集上可以通过缩小方差获得很低甚至负loss，但validation loss明显变大，学到的std不是可靠epistemic confidence。
2. **相对收益与nominal共享误差未正确处理。** 证书模型仍易把绝对严重度当作相对收益。
3. **准入和排序仍缺少完整因果隔离。** 即使Stage C冻结Preference，证书训练数据的top候选仍由不稳定Stage P产生，输入标签本身噪声较大。
4. **No-rule时诊断不完整。** v48.7规则搜索失败后rows为空，near-miss fit规则没有在verify上被统一重评，难以判断是fit过拟合还是统计支持不足。

---

## 4. 数据问题仍然构成上限

Teacher index显示：

- 3,800个scene-time group中，deployable positive group只有362个；
- Near只有166个deployable positive group、72个scene，低于原建议的200/80；
- Contact有196个positive group、68个scene；
- Near正机会中macro 5占87.95%；Contact占88.78%；
- quality gate明确标记为`marginal_debug_only`。

在不重构数据集的前提下，模型仍可以用于筛选架构，但不能期待仅靠扩大网络稳定达到论文门槛。v48.8通过相对特征、低容量adapter、所有group监督、macro平衡和conformal校准减轻漂移影响；若v48.8仍无法使Near/Contact top-1稳定转正，下一步最合理的动作将是有限定向补建，而不是继续堆叠loss：优先补充非macro-5正机会和Near正机会scene，不必先全面重构全部train set。

---

## 5. Safe投稿目标完成情况

上传的Safe probe包含：

- 8个scene、320次decision；
- intervention rate与episode rate均为0；
- bounded NUP为1；
- collision scene rate为0.125；
- offroad scene rate为0.125。

这些collision/offroad数值不能单独判断模型变差，因为没有相同scene上的nominal/reference paired结果。当前只能确认：Safe selector被nominal lock，没有主动干预。

论文目标状态：

| 目标 | 当前状态 |
|---|---|
| collision/offroad不增加 | 未验证，缺paired baseline和CI |
| paired scene 95% CI上界≤+0.1～0.2个百分点 | 未验证 |
| route progression下降≤0.5% | 结果未输出该指标 |
| NUP增加≤1% | 仅8个scene诊断性通过，不能作为论文结论 |
| jerk/yaw-rate p95增加≤5% | 结果未输出对应scene-level统计 |
| intervention episode增加≤2%～3% | 8个scene上为0，样本不足 |

v48.8新增scene-paired Safe runner和bootstrap non-inferiority分析；如果runner没有输出route/jerk/yaw，报告会明确标为不可用，不会使用其他指标替代。

---

## 6. Near-contact投稿目标完成情况

由于verify执行数为0，以下闭环目标均未验证：

- collision相对下降15%～25%；
- minimum clearance p05提高0.20m；
- minimum TTC p05提高0.20s；
- near-contact exposure下降15%；
- DRS提高8个百分点；
- PCD提高0.03；
- FRA下降30%；
- ODG下降25%；
- harmful switch≤5%～10%。

当前离线差距：

- group top-1仍为负；
- acceptable top-1 accuracy仅0.53；
- candidate positive AUC只有0.66～0.69；
- Near数据中的teacher winner近似并列较多；
- macro-5捷径严重；
- near-miss precision LCB远低于0.60；
- 正机会样本与scene数量本身不足。

因此Near首先需要解决排序和正负相对收益辨识，再讨论闭环改善。

---

## 7. Contact投稿目标完成情况

由于verify执行数为0，以下指标均没有闭环证据：

- secondary overlap下降20%；
- recontact count与overlap duration下降20%；
- stable-stop rate提高10个百分点；
- time-to-stable-stop下降10%；
- post-contact clearance提高0.20m；
- uncontrolled displacement下降15%；
- route-rejoin提高5个百分点。

Contact比Near更接近可用：candidate AUC约0.77，acceptable-set accuracy约0.62～0.64；但group top-1仍为负，positive regret约0.19，near-miss harmful rate仍过高。当前最重要的不是提高candidate AUC，而是让rank和certificate分别具有可泛化、可验证的语义。

---

## 8. v48.8：OC-TRAC-SCOPE

**SCOPE = Support-aware Conflict-free Ordinal Preference with Conformal Evidence。**

### 8.1 Conflict-free nominal-inclusive set preference

v48.8不再把set-valued loss和single-winner loss同时叠加：

- material-positive group的目标是teacher等价恢复集合；
- no-opportunity group的目标是nominal以及仅处于dead-zone的等价候选；
- harmful recovery被显式压到nominal以下；
- 开启该目标时，替换旧single-winner/listwise家族，而不是与其相加。

这消除了Near tie区间中的直接梯度冲突，并让模型学习“什么时候nominal应该排第一”。

### 8.2 低容量、纯相对Preference上下文

Stage P只训练小型context residual：

- 输入只保留candidate−nominal、recovery mean和recovery max；
- 不输入绝对candidate block，降低严重度与macro捷径；
- inherited pointwise preference完全冻结；
- hidden width降到48；
- residual保持严格zero-init。

这把可训练自由度从大规模Preference路径压缩到小型相对修正器，更匹配当前只有数百个正机会group的数据规模。

### 8.3 Support-aware checkpoint

- 相同指标不再覆盖更早checkpoint，只有超过`best_metric_min_delta`才算改善；
- fold正机会少于最低支持数时，不参与worst-fold选择；
- 采用最差K个有支持fold的均值，而不是单一稀疏fold最大值；
- Preference risk同时加入harmful top-1和非机会group错误切换惩罚。

### 8.4 Robust relative-gain learning

Stage C只训练direct-delta adapter：

- smooth-L1学习candidate−nominal exact PCD增益；
- soft sign loss区分正收益和负收益；
- 默认关闭heteroscedastic NLL，避免方差崩溃；
- log variance固定在保守初值，不再冒充learned uncertainty。

### 8.5 Split-conformal execution evidence

在proxy calibration的fit scene上估计有限样本单侧残差分位数，得到relative gain lower confidence bound。规则搜索、verify、selector、offline evaluator和closed-loop均使用相同语义。

Conformal证书的价值在于：

- 不要求网络在漂移数据上准确学习方差；
- 能将“预测收益”和“统计支持”分开；
- dedicated calibration完成后可以对同一checkpoint重新估计，不必重新训练。

### 8.6 诊断完整性

- no-rule时也写出unconstrained top-1 rows；
- 所有near-miss fit规则在verify fold重新评价；
- 报告区分conditional harmful selected rate与all-group harmful exposure；
- Stage P、Stage C和Natural gate分别输出通过状态。

---

## 9. 实验加速与双A30调度

四组消融不能安全地以四个完整训练进程同时占用两张A30。这样会造成每张GPU两个模型竞争显存和计算，可能OOM，也会破坏wall-time可比性。

v48.8采用：

- 四组任务一次提交；
- 任意时刻最多两个GPU训练任务；
- 每张A30仅运行一个variant；
- 8个任务按4个wave自动排队；
- 不需要人工逐组启动。

此外：

- proxy split只构建一次；
- exact teacher-PCD index只构建一次；
- 消融通过硬链接/共享路径复用；
- controller支持单variant任务，不再为每个消融强制启动两variant；
- BF16、TF32、persistent workers和prefetch保留；
- Safe paired probe可在两张GPU上同时运行reference和candidate。

这些优化不改变模型目标和数据，只减少重复I/O、teacher index构建及GPU空闲时间。

---

## 10. 分阶段决策规则

v48.8不要求每次都跑完所有昂贵实验。

### Stage P gate

至少一个variant应同时满足：

- Near top-1 corr ≥0.10；
- Contact top-1 corr ≥0.10；
- Near/Contact acceptable top-1 accuracy ≥0.60；
- 两个regime均不再出现系统性negative top-1。

未达到时：只分析Preference消融，不运行stress closed loop，也不把精力投入准入阈值。

### Stage C discrimination gate

- Near positive AUC ≥0.75；
- Contact positive AUC ≥0.80；
- positive regret相较v48.7下降；
- conformal gain与harm具有清晰分离。

Stage P和Stage C discrimination通过后，才值得做三seed复校准。

### Natural gate

开发阶段至少要求：

- Near和Contact均有非零verify selection；
- precision LCB90达到开发阈值；
- positive recall不低于0.35；
- conditional harmful UCB受控；
- max selected macro share≤0.85。

论文准备阶段仍采用更严格门槛：top-1 corr≥0.20、precision LCB90≥0.60、harmful switch趋近5%～10%，并用dedicated calibration和paired closed loop确认。

---

## 11. 当前calibration策略

在`calibration_contact`没有完成之前，继续统一使用：

```text
CALIBRATION_MODE=proxy_val_split
CALIBRATION_FRACTION=0.50
```

不要混用dedicated Safe/Near与proxy Contact，因为三个regime的数据合同和统计支持会不一致。Contact完成后，再将三套dedicated calibration一起用于同一个固定checkpoint重新校准，不需要重新训练v48.8。

---

## 12. 验证状态

本地完成：

- 137项pytest通过；
- Python compileall通过；
- 主要Shell脚本语法检查通过；
- 新增all-group set preference、relative-only context、support-aware fold和conformal quantile测试；
- 修复Safe summary块中的重复`for`语法错误；
- 新增Safe scene-paired bootstrap报告工具。

本地没有真实WOMD、Waymax或A30环境，不能提前保证v48.8通过Natural gate或达到投稿闭环目标。
