# OC-RAP v48.3结果审计、外部基线对比与v48.4优化说明

日期：2026-07-25

## 1. 结论摘要

本轮上传的`ocrap_v48_3_nasc_rcd_proxy`已经完成训练和proxy calibration，但没有产生可部署候选。四个“variant × regime”组合的fit和verify均选择0个恢复动作。Natural gate拒绝是正确的保护行为，但这意味着当前OC-RAP在Near-contact和Contact上没有形成任何可验证的恢复收益，也不能据此宣称超过外部baseline。

v48.3没有解决两个核心问题：

1. 同一个scene-time group中，哪个恢复候选真正优于nominal和其他候选；
2. candidate-level AUC与policy-level top-1选择脱节。

代码审计进一步发现，v48.3的实验结论受到两个工程问题影响：NASC不是严格零初始化，载入v48.1 checkpoint后会立即注入随机集合残差；配置的worst-regime best metric没有被验证循环产出，训练程序静默退回total loss做early stopping。因此，v48.3不能被视为对“NASC/RCD思想本身”的充分否定，但现有实现确实没有取得策略级改善。

新代码升级为**v48.4 OC-TRAC-SRGR（Shift-Robust Groupwise Recovery）**，核心包括：

- ZI-NASC：严格零初始化的nominal-anchored set context；
- DRA-RCD：将组内排序与是否执行的风险准入解耦；
- soft opportunity/downside labels：减轻阈值附近标签翻转；
- pseudo-environment GroupDRO：降低train中特有严重度和macro分布的支配；
- exact teacher-PCD policy-regret checkpointing：真正按Near/Contact最差组级regret选择checkpoint，不再静默回退total loss。

## 2. Calibration seed与输出目录

### 2.1 seed的理解

是的，三种calibration seed就是分别设置：

```bash
CALIBRATION_SEED=4801
CALIBRATION_SEED=4802
CALIBRATION_SEED=4803
```

这里的seed控制的是从原val按scene划分proxy calibration/dev的具体scene集合，不应被理解为模型训练随机种子。

### 2.2 输出目录不能相同

三个seed绝对不应使用同一个`OUTPUTDIR`。同一目录会覆盖：

- `dataset_splits/calibration_*`和`development_*`；
- split manifest；
- calibration rows和JSON；
- candidate selection与screening status；
- 日志和exit code。

正确做法是固定同一组训练checkpoint，只重新建立三套scene split并分别校准：

```text
runs/ocrap_v48_4_srgr_proxy_multiseed/seed_4801
runs/ocrap_v48_4_srgr_proxy_multiseed/seed_4802
runs/ocrap_v48_4_srgr_proxy_multiseed/seed_4803
```

本次代码新增`scripts/recalibrate_v48_4_multiseed.sh`，会自动执行上述流程，不会重复训练模型。

### 2.3 为什么不建议为每个calibration seed重新训练

本阶段要测量的是“同一个模型对calibration scene抽样是否敏感”。如果训练也一起重跑，结果同时混入训练随机性和calibration抽样随机性，无法定位波动来源。建议分两层进行：

1. 固定checkpoint，测试4801/4802/4803 calibration robustness；
2. 模型结构确定后，再使用3个训练seed × 3个calibration seed做论文级统计。

## 3. v48.3相对v48.1是否提升

### 3.1 离线策略指标

| Variant | Regime | v48.1 Positive AUC | v48.3 Positive AUC | v48.1 Top-1 corr | v48.3 Top-1 corr | v48.3 Harm AUC |
|---|---:|---:|---:|---:|---:|---:|
| Balanced | Near | 0.696 | 0.7249 | -0.019 | -0.0190 | 0.5320 |
| Balanced | Contact | 0.786 | 0.7634 | -0.110 | -0.1397 | 0.5262 |
| Precision | Near | 0.729 | 0.7268 | +0.034 | -0.0208 | 0.5541 |
| Precision | Contact | 0.822 | 0.7906 | -0.079 | -0.0678 | 0.5551 |

判断：

- Balanced-Near candidate AUC有约+0.029的提升，但top-1完全没有改善；
- Precision-Contact top-1从-0.079变为-0.068，只是负相关程度略减，仍然错误；
- Precision-Near从小幅正相关退化为负相关；
- 两个Contact AUC均下降，Precision-Contact下降约0.031；
- Harm AUC最高仅0.555，仍接近随机排序；
- 四个组合的fit/verify全部选择0个动作，positive recall均为0。

因此，v48.3没有跨过任何关键策略门槛。

### 3.2 与内部CCF-A readiness gate的差距

| 指标 | 当前最佳v48.3 | 建议最低门槛 | 结论 |
|---|---:|---:|---|
| Near positive AUC | 0.7268 | ≥0.78 | 未达到 |
| Contact positive AUC | 0.7906 | ≥0.82 | 未达到；v48.1曾达到0.822 |
| Near group top-1 corr | -0.019 | ≥0.20 | 明显未达到 |
| Contact group top-1 corr | -0.068 | ≥0.20 | 明显未达到 |
| Verify precision LCB90 | 无选择 | ≥0.60 | 无法计算 |
| Verify positive recall | 0 | ≥0.35 | 未达到 |
| Harmful-switch UCB90 | 选择数为0，统计无意义 | ≤0.10 | 不能用“零选择”冒充安全成功 |
| 每fold有效选择 | 0 | ≥15–20 | 未达到 |

这里的数值是本项目的内部投稿准备线，不是CCF官方录用阈值。CCF-A能否投稿还取决于创新完整性、统计显著性、baseline公平性、闭环真实性和论文论证。

## 4. 为什么v48.3仍未达到门槛

### 4.1 NASC warm-start并未真正保留旧模型

v48.3的set adapter最后一层为随机初始化，gate为`sigmoid(-1.5)≈0.182`。因此即使旧checkpoint的106个兼容key全部成功加载，首个forward也会在旧特征上叠加约18%的随机LayerNorm残差。日志显示仅新set adapter相关key缺失，且没有shape mismatch，这说明继承机制本身已经正确，但新分支的初始化破坏了继承值。

这可以解释：

- v48.1 Contact AUC被破坏；
- epoch 1就可能偏离原selector；
- 小规模正机会监督不足以快速修复随机组上下文。

### 4.2 Early stopping没有按策略指标工作

脚本配置：

```text
training.best_metric=loss_direct_recovery_value_worst
```

但验证结果从未输出这个key。旧train loop使用：

```python
va.get(best_metric_name, va.get("loss"))
```

因此它静默使用total validation loss选best checkpoint。`train_summary.json`也证实各epoch没有`loss_direct_recovery_value_worst`字段。

这意味着模型训练目标虽然包含RCD，checkpoint却不是按Near/Contact最差组级排序或regret选出的。候选级回归loss降低，并不保证组内top-1改善。

### 4.3 RCD排序梯度被弱harm head污染

v48.3使用的policy logits同时混合：

- value delta；
- opportunity log-probability；
- inverse harm log-probability。

RCD的teacher distribution distillation和expected regret均作用于这一复合分布。当前harm AUC只有0.53–0.56，导致不可迁移的harm预测直接参与“哪个候选最优”的排序梯度。结果可能是：

- value head本来有候选信号，但被harm噪声重新排序；
- 训练趋向保守、所有候选分数被压低；
- calibration最终没有任何joint rule通过。

### 4.4 数据合同漂移仍然存在

已知Near/Contact的train与val/test在`r_dep_star`、hard violation、harm_proxy和macro机会集中度上不一致。即使当前harm loss主要使用teacher relative advantage而不是直接使用`harm_proxy`，这些变化仍会导致：

- 正/负margin附近的hard label翻转；
- 模型利用train中特有严重度作为捷径；
- 少数macro和scene主导正机会；
- 专家学习到训练集严重度分层，而不是可迁移的相对恢复规律。

### 4.5 正机会数量可用于筛选，但不足以支撑强泛化

当前exact teacher-PCD index包含：

- 3,800个scene-time groups；
- 442个positive-advantage groups；
- Near约210个positive groups、84个scenes；
- Contact约232个positive groups、76个scenes。

这些数量足以做架构方向筛选，但对于多macro、多严重度、多交互类型的稳定策略学习仍偏少，尤其当正机会集中于少数macro时。

## 5. 不完全重构train set时，如何减轻数据漂移

完全“去除”数据漂移做不到，但可以把现有数据用于可靠的算法筛选，并显著降低捷径风险。

### 5.1 本轮已经实现的策略

#### A. Pseudo-environment GroupDRO

v48.4把每个group划分到：

```text
(regime, nominal severity bin, opportunity state, teacher-best macro)
```

并对各环境平均loss做soft worst-case聚合，再与ERM混合。它的作用是防止样本量大的轻/重严重度或macro 5支配梯度。

它没有使用calibration/test分布来反推训练权重，因此不会产生目标集泄漏。

#### B. Soft opportunity/downside labels

原hard label在`+0.015/-0.010`附近会因teacher轻微变化突然翻转。v48.4在margin附近使用连续sigmoid target，使小的合同漂移只导致小的监督变化。

#### C. 排序与准入解耦

value-only distribution负责学习组内排序；harm/opportunity只负责是否离开nominal。这样即便harm head迁移较差，也不会直接破坏top-1 ranking。

#### D. Nominal-relative和group-relative表示

继续保留nominal delta、set mean/max以及完整scene-time group推理。绝对严重度变化更难成为唯一捷径。

#### E. Worst-regime policy regret early stopping

checkpoint按Near/Contact中更差的exact teacher-PCD group regret选择，而不是按总样本loss选择。它可以避免一个regime改善、另一个regime崩溃。

#### F. 多calibration seed稳定性

同一checkpoint必须在4801/4802/4803上表现一致。若仅一个split有非零选择，说明模型仍在利用小样本校准偶然性。

### 5.2 暂不建议的做法

- 不要使用test分布做importance weighting；
- 不要用专用Safe calibration与proxy Near/Contact混合调联合阈值；
- 不要降低Natural gate来制造非零选择；
- 不要仅追加旧合同数据，因为可能继续放大原有severity/macro捷径；
- 不要用harmful-switch=0且selection=0作为“模型非常安全”的证据。

### 5.3 何时必须重构train set

你的计划“先让模型明显跑赢baseline，再重构”适合开发阶段，但正式论文之前仍应至少完成一次合同统一的数据重构或提供等价的shift-controlled实验。建议触发条件：

1. current-data screening中Near和Contact top-1 corr均≥0.15，且非零verify selection；
2. 4801/4802/4803至少2个seed通过Natural gate；
3. development closed loop明显优于最强外部baseline的核心trade-off；
4. 消融证明改进来自模型而非单一阈值。

此时再投入重构成本最合理。若v48.4仍然top-1≤0，则应先继续修模型，不值得立即重构全部数据。

## 6. 是否需要从头训练或回到v47

不需要，也不建议。

本轮日志显示从v48.1 precision checkpoint载入：

- loaded keys = 106；
- shape mismatch = 0；
- 缺失项仅为v48.3新增的set-context参数。

因此上一轮从v47到v48发生的direct-head维度不兼容问题，在本轮v48.1→v48.3继承中已经解决。

v48.4推荐继续从：

```text
runs/ocrap_v48_1_existing_data_screening/candidates/precision/model_v48_trac_sr/best.pt
```

初始化，而不是：

- 从v47再次触发head shape mismatch；
- 从v48.3继承已经被随机NASC残差扰动的参数；
- 从scratch丢掉已有0.822 Contact AUC信号。

v48.4的ZI-NASC会保证新set branch初始输出严格等于继承的pointwise输出，然后再逐步学习集合修正。

## 7. 三个regime目前的结果与外部baseline

### 7.1 Safe

外部offline baseline中：

- nominal/log replay：DRS=1、FRA=0、ODG=0、NUP=1、intervention=0、yaw violation=0；
- Wayformer-BC与nominal相同；
- BeTopNet保持DRS/NUP，但intervention和yaw violation均为17.65%；
- GameFormer intervention和yaw violation均为39.22%，NUP降至0.9468。

这说明Safe最强基线非常接近“不要干预”。当前OC-RAP因为Near/Contact Natural gate失败，没有生成完整OC-RAP closed-loop结果。若部署逻辑保持nominal，理论上会接近Safe强基线，但目前尚无paired-scene非劣置信区间，不能宣称达到Safe投稿门槛。

v48.4增加了Safe-only non-inferiority probe指令，可在Near/Contact未通过时独立验证Safe，但不能用Safe通过来授权Near/Contact。

### 7.2 Near-contact

offline external baseline中，`predictive_safety_filter`表现出当前最好的综合trade-off：

- DRS=0.9726；
- FRA_exec=0.1268；
- ODG=0.1743；
- NUP=0.9882；
- intervention=0.4457；
- yaw violation=0.0942；
- stable stop=0.8841；
- route rejoin=0.9964。

Oracle recovery虽然DRS=0.9783，但FRA/ODG和干预更高，不是可直接采用的实用上界。MARC/RACP的干预、FRA和ODG明显偏高。

v48.3没有任何verify selection，因此：

- 正机会召回为0；
- 不能测得有效precision；
- 无法产生clearance/TTC/DRS/FRA/ODG闭环改进；
- 与predictive safety filter相比没有恢复收益。

上传的Near closed-loop baseline只完成30或31个/50个scene，尚未完成，不能用于最终论文排名。当前部分结果中predictive safety filter平均DRS约0.746、FRA约0.254、NUP约0.875、intervention约0.270；这些数值只能作为开发诊断。

### 7.3 Contact

offline baseline中：

- post-impact MPC-lite：DRS=0.9271、FRA=0.1648、ODG=0.1287、stable-stop=0.8396，恢复效果最强；但intervention=0.9376、yaw violation=0.5011、NUP=0.6293、route rejoin=0.686，代价很大；
- severity minimization：FRA=0.0557、ODG=0.0284最低，但DRS只有0.6699，stable-stop=0.5813；
- restoration：DRS=0.9144、route rejoin=0.9955，但FRA/ODG较高；
- post-crash braking总体较弱。

因此Contact的投稿亮点不应只是提高DRS，而应证明在保持post-impact MPC恢复能力的同时，显著降低过度干预、yaw violation和NUP损失。

v48.3没有动作选择，所以未产生secondary overlap、recontact、stable stop、time-to-stable-stop或uncontrolled displacement结果。上传的Contact closed loop只完成18/50 scene，仍在运行，不能作为最终比较。

## 8. v48.3哪些改动有效、哪些无效

### 8.1 有效或部分有效

- 完整group batched calibration已经生效，训练/校准上下文不一致问题被修复；
- v48.1 checkpoint无shape mismatch继承成功；
- exact teacher-PCD sampler提供了足够的方向筛选正机会；
- Natural gate正确阻止了不可靠策略进入闭环；
- Balanced-Near candidate AUC得到局部提升；
- explicit nominal、tri-state supervision和harm诊断仍是合理的安全框架。

这些应继续保留。

### 8.2 无效或作用不大

#### NASC原实现

思想合理，但随机残差warm-start破坏旧模型。v48.4改为严格zero-init，而不是继续增大set hidden或gate。

#### RCD原实现

完整teacher distribution和expected regret是正确方向，但作用于包含弱harm head的复合policy分布，使排序目标受到风险分类噪声污染。v48.4将ranking和admission分开。

#### Harm head

当前AUC接近随机，不能承担候选排序任务。它仍可作为保守准入证据，但需要soft downside label、独立校准和更弱的排序耦合。

#### Robust expert specialization

两个expert的disagreement在训练后持续下降，说明它们可能趋同而没有形成真正的风险态度互补。v48.4把specialization weight从0.40降到0.30，优先修复group ranking。后续只有在消融证明双expert有效时再增强；否则可简化为单value ranker + 独立risk head。

#### Calibration grid

当前无joint rule通过不是grid太窄，而是模型top-1和risk ordering错误。扩大grid或降低约束不会解决根因。

## 9. v48.4代码修改

### 9.1 ZI-NASC

- set residual最后投影严格zero-init；
- gate初值从-1.5改为-2.5；
- 多候选group首次forward与旧pointwise模型严格一致；
- 保留排列等变和singleton fallback。

### 9.2 DRA-RCD

- value-only logits负责teacher distribution distillation和expected regret；
- opportunity/harm只进入admission logits；
- 可使用小权重admission distillation，避免完全脱节；
- SRC harmful mass和coverage仍基于admission policy。

### 9.3 Soft labels

新增：

```text
direct_value_opportunity_soft_label_temperature
direct_value_harm_soft_label_temperature
```

默认均为0.02。

### 9.4 Pseudo-environment GroupDRO

新增：

```text
direct_value_group_dro_weight=0.35
direct_value_group_dro_temperature=0.35
direct_value_group_dro_severity_thresholds=0.25,0.55
```

### 9.5 Policy-level validation和early stopping

新增验证指标：

```text
direct_group_regret_mean_near
direct_group_regret_mean_contact
direct_group_regret_mean_worst
direct_group_top1_accuracy_{near,contact}
direct_positive_recall_{near,contact}
direct_harmful_switch_rate_{near,contact}
```

训练脚本使用：

```text
training.best_metric=direct_group_regret_mean_worst
```

若该key没有生成，程序会直接报错，不再静默回退total loss。

### 9.6 训练路径统一

旧代码direct-only fast path和完整path对新loss参数支持不一致。本轮统一通过`_direct_value_loss_from_outputs()`，避免关闭fast path时算法悄然变化。

## 10. 四组消融如何运行

新增脚本：

```text
scripts/run_v48_4_core_ablations.sh
```

它按顺序执行四组实验，每组内部同时在GPU0/GPU1训练balanced和precision：

| 目录 | Set context | DRA-RCD | Soft labels | GroupDRO | 目的 |
|---|---:|---:|---:|---:|---|
| A_src_reference | 关 | 关 | 关 | 关 | v48.2/SRC参考 |
| B_zi_nasc_only | 开 | 关 | 关 | 关 | ZI-NASC独立贡献 |
| C_dra_rcd_only | 关 | 开 | 关 | 关 | 排序/准入解耦独立贡献 |
| D_full_srgr | 开 | 开 | 开 | 开 | v48.4完整模型 |

即使Natural gate拒绝，脚本也会继续下一组，并保留`screening_status.json`。最终生成：

```text
runs/ocrap_v48_4_ablations/ablation_summary.json
```

完整命令已写入`run_v48_4_main_commands.txt`。

## 11. 下一步实验顺序

### 阶段A：v48.4主筛选

先运行full SRGR，仍使用proxy val split seed 4801。第一优先级不是立即通过Natural gate，而是观察：

- `direct_group_regret_mean_worst`是否随epoch下降；
- Near/Contact top-1 correlation是否都转正；
- candidate AUC是否基本保留；
- 至少一个variant是否出现非零verify selection。

建议screening gate：

```text
Near top-1 corr > 0.10
Contact top-1 corr > 0.10
至少一个variant在Near和Contact均有非零verify selection
candidate AUC较v48.1下降不超过0.03
```

### 阶段B：四组消融

无论full是否完全通过投稿门槛，都应运行四组消融来定位增益来源。若资源紧张，可在full模型top-1转正后再运行，但本轮已经提供了完整指令。

### 阶段C：固定checkpoint多calibration seed

对同一个SOURCE_RUN执行4801/4802/4803，要求：

- 至少2/3 seed在两regime有非零选择；
- top-1 correlation方向一致；
- verify precision/harm没有极端波动；
- 不能只有一个seed通过Natural gate。

### 阶段D：Safe-only probe

可以独立运行Safe non-inferiority，检查：

- collision/offroad不增加；
- route progression/NUP无明显退化；
- intervention/jerk/yaw-rate接近nominal。

### 阶段E：Near/Contact closed loop

只有两regime Natural gate均通过时，才执行controller生成的`NEXT_COMMANDS.txt`。不能因为external baseline已完成就跳过selector certificate。

### 阶段F：专用calibration完成后复校准

Near/Contact dedicated calibration全部构建完成后，用最佳checkpoint重新校准，不需要重新训练。最终论文结果不应混用dedicated Safe和proxy Near/Contact。

## 12. 本地验证状态

- 122项pytest通过；
- Python compileall通过；
- 主要shell脚本`bash -n`通过；
- 新增测试覆盖：
  - ZI-NASC初始严格pointwise identity；
  - DRA-RCD排序loss不受harm logits污染；
  - worst-regime policy regret指标正确生成；
  - 原有排列等变、singleton fallback、SRC和RCD测试继续通过。

当前环境没有真实WOMD/JAX/GPU训练条件，因此v48.4是否改善top-1和闭环指标必须由下一轮服务器实验确认。
