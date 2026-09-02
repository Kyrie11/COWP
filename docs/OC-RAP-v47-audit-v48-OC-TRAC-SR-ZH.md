# OC-RAP v47 联合审计与 v48 OC-TRAC-SR 优化报告

## 0. 审计范围、版本完整性与结论边界

本报告联合检查了：

- `post-collision.tex`：论文理论、算法定义、评价协议与拟写实验；
- `OC-RAP.zip`：可用的代码本体；
- `ocrap_v47_trac_balanced.zip`：v47 训练日志、checkpoint 摘要及 Near/Contact 风险校准结果；
- `reports.zip`：Safe、Near-contact、Contact 三个 regime 的 train/val/test 数据诊断；
- `大模型建议.md`：上一轮失败分析与 v47 设计思路。

必须先说明一个完整性问题：上传的 `OC-RAP.zip` 中主运行链、脚本命名和算法日志仍对应 v45 代码树，并不是生成当前 v47 结果的逐行源码；v47 压缩包主要包含结果和日志。因此，本次 v48 是在**可获得的 v45 完整源码上，依据 v47 结果文件、checkpoint 接口与既有建议继续实现**。我已加入兼容旧 checkpoint 的“按名称且按形状加载”逻辑，但无法声称这是对缺失的精确 v47 源码做出的逐行补丁。

### 核心结论

1. **v47 只有部分修改有效，尚未解决核心问题。**候选级正恢复可分性明显提高，但仍不能转化为同一 scene-time 候选集合中的正确 top-1 策略选择；Natural gate 拒绝 v47 是正确的保护行为。
2. **需要补充/重建 `train_contact`，不能只靠继续调 loss 或阈值解决。**当前问题不只是样本总量不足，而是训练与 val/test 的构建合同明显漂移、正恢复机会集中于少数 scene/macro、`harm_proxy` 在 val/test 中退化为全零。
3. **需要 calibration dataset。**开发阶段可立即从现有 `val_*` 按 scene 无交叉划出 calibration/dev；论文最终实验更推荐从标准 WOMD validation 额外构建专用 calibration roots，并显式排除所有现有 val/test scene。
4. **WOMD 选择应为：标准 training 构建训练集，标准 validation 构建开发/校准/内部测试。**官方 testing 和 testing_interactive 隐藏未来真值，不能用于 teacher-PCD 标签、threshold calibration 或本地闭环真值评估；validation_interactive 只适合作为额外 OOD/interaction stress，不应混入主 IID 结果。
5. 已生成下一版 **v48 OC-TRAC-SR**：Observation-Consistent Tri-state Risk-Calibrated Recovery Admission with Scene-disjoint Calibration。它直接优化“恢复候选或 nominal abstain”的策略级选择，加入独立 harm 预测、保守专家聚合、scene 级校准、数据质量门、训练/校准/评估隔离和 direct-only 加速链。
6. **v48 代码已经通过静态编译、Shell 语法检查和 97 项测试，但尚未在用户 GPU/WOMD 环境中跑出结果。**因此本报告只承诺修复了已识别的目标错位和工程缺口，不承诺 v48 已达到 SOTA。

---

## 1. v47 是否有效：逐项判定

### 1.1 v47 的正向结果

v47 的 exact teacher-PCD sampler 确实工作了，而不是上一版本中的“配置存在、实际未采样”：训练日志记录 3,800 个 scene-time group、357 个正优势 group、replacement=true、positive boost=4.0。候选级分类能力也比 v46 明显好：

| Regime | candidate positive AUC | candidate harm AUC | 候选预测与 teacher 优势相关 | 无约束 group top-1 相关 |
|---|---:|---:|---:|---:|
| Near-contact | 0.7745 | 0.5881 | 0.1215 | **-0.0263** |
| Contact | 0.7853 | 0.5839 | 0.0818 | **-0.0188** |

因此，v47 证明了“观测和候选特征中存在可学习的恢复信号”，也证明 exact PCD 采样比历史 `R_dep` proxy 更合理。

### 1.2 v47 没有解决的核心问题

候选 AUC 与策略 top-1 出现明显脱节：模型大致知道哪些候选可能有正机会，但在同一 scene-time 集合内真正选一个动作时，排序方向仍接近反向。这正是上一轮所说的：

> candidate-level learnability 不等于 policy-level deployability。

#### Near-contact

- 校准使用 409 个有效 group、172 个 scene；
- 没有任何 fit-fold 联合规则同时满足机会、gain、harm、覆盖和 precision 约束；
- fit/verify 均选择 0 个动作；
- Natural gate 失败，不应进入 learned-policy closed-loop。

#### Contact

Contact 更典型地暴露了 fit-fold 过拟合：

| Contact | 选择数 | 正例精度 | 有害选择率 | teacher 优势均值 |
|---|---:|---:|---:|---:|
| fit | 13 | 100.0% | 0% | +0.3085 |
| verify | 31 | **38.7%** | **41.9%** | **-0.1352** |
| all | 44 | 56.8% | 29.5% | -0.0041 |

规则在 fit 上几乎完美，但 verify 上变成负收益，说明它记住了少量局部 scene/macro 形态，而不是学到了可迁移的恢复接纳原则。

### 1.3 训练工程仍限制了 v47 上限

v47 日志确认：

- 初始化仍来自 `runs/ocrap_v39_ocrac_balanced/model_v39_ocrac/best.pt`；
- 冻结参数约 1,746,563，实际可训练参数约 485,192；
- 冻结范围包括 encoder、root attention、主要 certificate/option 表示；
- 最优 epoch 为 2，6 个 epoch 后早停；
- A30 上总训练约 3,785 秒，每 epoch 约 604–730 秒；
- 许多权重为 0 的历史分支仍被前向和统计，造成无效计算；
- 冻结分支对应的验证 loss 在多个 epoch 中完全不变，说明它们没有参与适应新数据合同。

因此 v47 不能被解释成“核心表示已经充分、只差再调阈值”。更合理的解释是：新 head 从旧表示中挖到了一些点级信号，但旧表示和训练目标没有被允许为了集合决策而重塑。

### 1.4 v47 修改项最终判定

| v47 修改 | 判定 | 依据 |
|---|---|---|
| exact teacher-PCD sampler | **有效** | 正组采样实际启用，357 个正 group 被识别并增采样 |
| tri-state/机会信号 | **部分有效** | candidate positive AUC 达 0.77–0.79，但 top-1 失败 |
| risk/harm head | **弱有效** | harm AUC 仅约 0.58，且旧 selector 未完整把 harm 作为最终接纳条件 |
| 风险态度专家 | **未得到有效验证** | 候选信号提高，但集合内排序和 Contact verify 仍失败 |
| 联合校准 gate | **作为保护机制有效** | 正确拒绝了不可部署规则；算法本身未通过 gate |
| Safe nominal preservation | **未验证** | learned checkpoint 未过 gate，不能用当前结果证明 Safe 闭环不损失 |
| 速度优化 | **未解决** | 仍约 10 分钟/epoch，且有大量无效分支计算 |

---

## 2. 是否需要补充 `train_contact`

### 2.1 结论：需要，但应新建而不是在旧 root 上盲目续增

当前 `train_contact` 已有 16,790 个 sample、2,000 个 scene-time group、500 个 scene。从绝对数量看，它并非“小到完全无法学习”；真正的问题是**训练集与 val/test 的构建分布不同**：

| Contact 指标 | train | val | test |
|---|---:|---:|---:|
| samples | 16,790 | 5,723 | 5,514 |
| scene-time groups | 2,000 | 639 | 616 |
| unique scenes | 500 | 187 | 171 |
| artifact fraction | 0.166 | 0.218 | 0.218 |
| negative deployable fraction | 0.543 | 0.461 | 0.445 |
| feasible fraction | 0.870 | 0.919 | 0.890 |
| hard violation mean | 0.0936 | 0.0105 | 0.0210 |
| harm_proxy mean | 0.0283 | **0.0000** | **0.0000** |
| `r_dep_star` mean | -1.79 | -0.56 | -0.54 |
| candidates/group mean | 8.40 | 8.96 | 8.95 |

Near-contact 有相同漂移：训练 `r_dep_star` 均值约 -1.79，而 val/test 约 -0.80/-0.69；训练 hard violation 均值约 0.089，而 val/test 约 0.009/0.016。

这表明旧训练集更“硬”、负例更极端、候选前沿更窄。模型可以利用 hard、harm_proxy、宏类型等捷径，在训练上区分极端样本，却无法迁移到 val/test 的细粒度候选排序。

### 2.2 `harm_proxy` 不应再作为主要 harm 监督

val/test Near 和 Contact 的 `harm_proxy` 均值、分位数和最大值全部为 0。无论这是构建配置差异还是字段写入问题，它都意味着：

- 使用旧 `harm_proxy` 训练、使用 teacher PCD 验收，目标不一致；
- 模型可在训练中学习一个在验证域不存在的特征；
- 独立 harm head 应直接由“candidate teacher-PCD 相对 nominal 显著为负”或真实二次接触/后接触事件监督，而不是依赖该字段。

v48 已把 harmful switch 定义为 candidate-vs-nominal teacher-PCD 的负优势，并在最终 selector 中显式使用预测 harm 上界。

### 2.3 正恢复机会的集中性

v47 分析显示：

- Near 的正机会约 58 个候选，来自约 32 个 scene；前 10 个 scene 占约 48%；
- Contact 的正机会约 29 个候选，来自约 16 个 scene；前 5 个 scene 占约 55%，前 10 个约 79%；
- Contact 校准支持最终只剩 macro 5，44 个被选动作也全部来自 macro 5。

这正是 fit-fold 规则容易“记住 macro 5 + 少数 scene”的原因。仅在旧 root 上继续追加同一构建分布，会增加行数，但未必增加独立恢复机会或宏多样性。

### 2.4 推荐的新训练集合同

v48 新建：

- `/data0/senzeyu2/dataset/OCRAP_v48_train/train_near_contact`
- `/data0/senzeyu2/dataset/OCRAP_v48_train/train_contact`

原则：

1. 从标准 WOMD `training/training_tfexample.tfrecord@1000` 构建；
2. 与 val/test 使用相同的候选数量、targeted futures、artifact pair、teacher-PCD 参数和 regime 定义；
3. 两张 GPU 同时扫描两个互斥 shard partition；Near 完成后再构建 Contact，避免 4–6 个 JAX worker 抢 GPU/编译缓存；
4. 旧 `train_contact` 不修改、不直接混合；
5. 开训前按 exact teacher-PCD 统计真正的正优势 group，而不是只检查 artifact 数量。

默认质量门：

- Near：至少 200 个正优势 group、覆盖至少 80 个 scene；
- Contact：至少 120 个正优势 group、覆盖至少 60 个 scene；
- 同时输出正机会 macro 分布、最大 macro 占比和前 10 scene 占比；
- 论文最终建议把 Contact 提高到至少 300 个正 group、100–150 个正 scene，并让 top-10 scene 占比尽量低于 30%–40%。

若新 root 仍达不到这些标准，应优先扩大 raw scenario budget、改善候选前沿和接触事件采样，不应降低 Natural gate。

---

## 3. Calibration dataset 是否必要、如何构建

### 3.1 必要性

必要。当前论文的 calibration 公式对单个绝对 `R_dep` 阈值取负样本分位数，但实际部署决策是：

1. 同一 scene-time 中是否存在正恢复机会；
2. 候选相对 nominal 的 gain 是否足够；
3. harmful-switch 风险是否足够低；
4. 在支持的 macro 内选择 top-1；
5. 若没有可靠正机会，选择 nominal/abstain。

因此 calibration 必须针对**最终策略选择规则**，而不是针对独立候选分数。v48 联合搜索：

- opportunity threshold；
- harm upper threshold；
- gain/score threshold；
- supported macro set；
- scene-disjoint fit/verify；
- precision Wilson lower bound；
- harmful selection Wilson upper bound；
- selected teacher advantage；
- group top-1 correlation 和 macro concentration。

### 3.2 立即可运行的低成本方案

保留原始路径不变，从现有：

- `/data0/senzeyu2/dataset/OCRAP/val_near_contact`
- `/data0/senzeyu2/dataset/OCRAP/val_contact`

按 `original_scenario_id/scene_id` 哈希划分：45% calibration、55% development validation。只建立 hardlink/symlink，不复制大文件，不改原 manifest。训练早停仅看 dev 55%，threshold 仅看 calibration 45%。

这适合当前算法筛查，但现有 Contact val 只有 187 个 scene，若最终被选动作很少，无法严谨证明 1% 或 5% 的风险界。

### 3.3 论文最终推荐方案

从标准 WOMD validation 单独构建：

- `OCRAP_v48_calibration/calibration_near_contact`
- `OCRAP_v48_calibration/calibration_contact`

新脚本会在构建后排除现有 `val_safe/test_safe/val_near/test_near/val_contact/test_contact` 中出现的所有 scene，再生成 calibration root。这样 calibration、development 和 test 都按 scene 隔离。

建议最终至少：

- 每个 stress regime 500–1,000 个 calibration scene；
- verify fold 至少 50–100 次实际接纳事件；
- 报告场景配对 bootstrap CI；
- 在 Natural gate 通过前，不运行 held-out test；
- test 结果不得反过来调阈值或选择 checkpoint。

---

## 4. WOMD split 的正确选择

Waymo 官方说明：Motion Dataset 约 70% training、15% validation、15% testing；training 和 validation 提供未来真值，而 testing 为挑战隐藏未来，仅提供历史。因此：

| 目的 | 应使用的数据 | 原因 |
|---|---|---|
| `train_safe/train_near/train_contact` | standard `training` | 数据最多，有 future GT，可生成 teacher roots/PCD |
| early-stopping dev | standard `validation` 的 scene 子集 | 有 future GT，可做离线和 Waymax 闭环 |
| calibration | standard `validation` 的独立 scene 子集 | 有 teacher label，可拟合风险阈值 |
| internal test | standard `validation` 的第三个独立 scene 子集 | 未来可见但不参与调参 |
| official `testing` | 只用于官方 challenge/submission | future GT 隐藏，不能本地算 teacher-PCD |
| `testing_interactive` | 只用于官方 interactive challenge | 同样不适合本地 teacher calibration |
| `validation_interactive` | 额外 OOD/interaction stress | 可作为补充，但分布被 interaction mining 改变，不应替代主 IID validation |

WOMD v1.3.1 的 `sdc_paths` 应保留，这使 Waymax 的 route-following、wrong-way、progression 等指标更完整。Waymax 官方支持 overlap、offroad、wrong-way、route-following、kinematic infeasibility 和 log divergence，可作为闭环基础指标。

官方资料：

- Waymo Motion Dataset: https://waymo.com/open/data/motion/
- Waymo Open Dataset overview: https://waymo.com/open/about/
- Waymax: https://github.com/waymo-research/waymax
- Waymo Open Dataset code/schema: https://github.com/waymo-research/waymo-open-dataset

---

## 5. 三个 regime 应如何定义目标和指标

### 5.1 Safe：严格 nominal-preservation / non-inferiority

Safe 不应被当作“也要主动寻找 recovery”的 regime。主目标是：当 nominal 可行时锁定 nominal，证明恢复机制不会损失正常驾驶效用。

建议主指标：

- collision scene/step rate；
- offroad、wrong-way、off-route；
- route progression / net displacement / progress efficiency；
- NUP；
- intervention rate 与 nominal trajectory deviation；
- hard brake、纵向加速度、jerk、yaw-rate；
- paired-scene delta 和非劣效置信区间。

论文表述应是词典序目标：首先满足 Safe 的 nominal 非劣；只有 Near/Contact 才优化 recovery gain。

### 5.2 Near-contact：预防接触并保留恢复余量

建议把结果分成三层：

**物理安全：**

- collision rate；
- scene minimum clearance 的均值和 p05；
- minimum TTC；
- near-contact exposure time（低于距离/TTC阈值的步数占比）；
- offroad/route violation。

**恢复性质：**

- candidate-vs-nominal teacher PCD；
- DRS；
- FRA；
- ODG；
- positive opportunity recall；
- selected positive precision / harmful switch rate；
- group top-1 teacher advantage；
- selector abstain rate 和错失正机会率。

**效用与舒适性：**

- NUP、progress、intervention；
- hard brake、jerk、yaw-rate；
- 为提高 clearance 付出的 utility cost。

要特别报告“最小接触距离提高但 collision 不变”的场景，因为这正能体现 recovery-aware planning 在稀有事件前的空间余量收益。

### 5.3 Contact：以首次接触为条件的 post-event outcome

Contact 已经以接触发生为条件，单纯比较 collision rate 没有区分力。需要把首次 overlap/contact 时刻定义为 `t0`，评价 `t>t0`：

- secondary overlap/collision event rate；
- overlap episode count、重接触次数；
- post-contact overlap duration；
- stable-stop rate；
- time-to-stable-stop；
- post-contact minimum clearance；
- t0 后 offroad/wrong-way/off-route；
- post-contact displacement / runaway distance；
- route-rejoin success 和 rejoin time；
- 最大 yaw-rate、jerk、横摆稳定性；
- 涉及不同 collision partner 的数量；
- 若动力学近似可信，再报告 Δv/impact severity。

当前代码已经具有 `secondary_overlap_event`、overlap episode、stable stop/time、minimum clearance/TTC、FRA/DRS/ODG/PCD 等基础统计；v48 保留这些闭环输出。

重要论文边界：当前数据诊断将 Contact 标为 `post_contact_counterfactual`，且禁止 `post_contact_observed`。因此现阶段更准确的表述是“counterfactual contact-surrogate recovery”，不能把结果写成真实碰撞动力学下的 post-impact control。若要强主张“撞后控制”，必须补充 observed contact 或高可信动力学仿真数据。

---

## 6. 论文当前需要同步修正的出发点

### 6.1 绝对候选证书改为策略级接纳

论文 Appendix 当前用负样本 `R_dep` 的分位数得到单一阈值。v47 已经实证说明：候选 AUC 可以较高，但同组 top-1 仍错误。论文应改为：

- nominal 是显式可选类别；
- 无正机会时 nominal 是正确标签；
- 有正机会时才要求恢复候选胜过 nominal；
- 接纳规则由 opportunity、gain、harm 和 uncertainty 联合定义；
- 风险界针对“最终被策略选择的动作”，不是所有独立候选。

### 6.2 Contact 的理论语义收紧

论文目前将 post-contact 和真实二次碰撞作为主叙事，但现有数据是 counterfactual surrogate。建议分两级 claim：

1. 当前版本：在 contact-conditioned counterfactual stress 下，恢复接纳减少 harmful secondary evolution；
2. 后续增强：在真实/高保真 observed post-impact dynamics 下验证稳定停车、二次碰撞和 route rejoin。

### 6.3 模型架构文字与代码不一致

论文写的是 agent-map transformer + ego-prefix encoder + 可选 BEV occupancy/occlusion CNN；当前代码实际输入主要是结构化扁平 token 和 structured transformer，没有完整实现论文所述的 vector polyline + BEV 双通道。因此需要二选一：

- 修改论文，准确描述当前 structured observation encoder；或
- 真正实现论文声称的 agent/map polyline 与 BEV 分支并做消融。

投稿 CCF-A 时，架构和实现合同不一致会是高风险问题。

### 6.4 候选前沿是算法上限的一部分

如果生成器没有产生真正有效的 brake/yield/merge/stabilize 候选，再强的 selector 也无法恢复。论文应报告：

- oracle candidate-frontier recovery coverage；
- 每种 macro 的正机会率；
- 去重后的有效候选数；
- frontier miss rate；
- selector regret 相对 oracle-in-frontier，而不是相对不可达的全动作空间 oracle。

### 6.5 Novelty 应收敛

不应把“MoE”“部分可观测规划”“多未来规划”本身作为首创。更可信的 novelty 是：

> observation-consistent、nominal-relative、tri-state、policy-level、risk-calibrated recovery admission。

---

## 7. v48 OC-TRAC-SR 已实施的算法修改

### 7.1 策略级 tri-state 监督

每个 candidate-vs-nominal 分为：

- positive recovery：teacher PCD 优势超过正阈值；
- dead-zone/tie：差异太小，不强行当负例；
- harmful switch：显著劣于 nominal。

### 7.2 nominal 作为显式类别

setwise loss 不再只在 recovery candidates 中排 top-1：

- 有可靠正恢复时，正确类别是最佳 recovery candidate；
- 没有可靠正恢复时，正确类别是 nominal/abstain；
- 训练目标与部署“恢复或不切换”完全一致。

### 7.3 独立 harm head，并贯穿 selector/离线/闭环

网络同时输出：

- predicted gain；
- opportunity probability；
- harmful-switch probability。

接纳必须同时满足 gain 下界、opportunity 下界和 harm 上界。此前模型即使产生 harm 预测，旧 selector 也未完整消费；v48 已把该字段传到 inference、baseline evaluator、offline evaluator 和 closed-loop runner。

### 7.4 风险态度专家与保守聚合

两个 expert 都看 Near 和 Contact，而不是预测隐藏 regime：

- recovery-seeking expert 更强调正机会召回；
- harm-averse expert 更强调 false-positive/harm；
- gain/opportunity 用 `mean - λ·std`；
- harm 用 `mean + λ·std`。

专家分歧成为保守不确定性，而不是隐式 regime router。

### 7.5 解冻共享表示，使用分层学习率

- 不再冻结 encoder；
- encoder 使用主 head LR 的 0.12–0.18；
- 从 v47 checkpoint 兼容初始化；
- 新增/变形 head 按形状安全跳过旧权重；
- early stopping 使用 Near/Contact 中较差的 `loss_direct_recovery_value_worst`。

### 7.6 统一 exact teacher-PCD

数据采样、训练标签和 Natural gate 都使用相同 teacher-PCD 参数；开训前构建 index，并检查正组数量和 scene 覆盖。

### 7.7 联合 scene-disjoint 校准

校准器按 scene 拆 fit/verify，并联合搜索 opportunity/harm/gain/macro。只有 Near 和 Contact 都通过的 checkpoint 才会进入离线评估和下一步闭环。

### 7.8 旧手工证书从 v48 主结果中关闭

v48 policy 只启用新 direct-value risk certificate。历史 relative recovery、PCD rescue、brake-tail 等逻辑保留为消融，但默认 false，避免多个规则混合后无法判断真正贡献。

---

## 8. 工程与速度优化

### 8.1 训练

已实施：

- direct-only fast path：权重为 0 的 root/certificate/option 旧损失不再完整前向；
- BF16 autocast（不支持时可改 FP16）；
- TF32、high matmul precision、cuDNN benchmark；
- batch size 默认 96；
- pinned memory、persistent workers、prefetch；
- 两个训练 variant 各占一张 GPU 并行；
- teacher-PCD 预计算 index，避免每个 epoch 重复 teacher 计算；
- 不保存每个 epoch，只保存 best/latest；
- 训练进度和所有日志输出到 `$OUTPUTDIR/logs`。

这些修改预计会明显快于 v47 的 604–730 秒/epoch，但尚未在用户 A30 环境实测，不给出虚假倍数承诺。首先应查看新日志中的：data loading time、GPU utilization、epoch seconds 和 direct-only 分支命中情况。

### 8.2 数据构建

- 每次只运行一对 GPU worker；
- Near 完成后再开始 Contact；
- 修复旧重建脚本中未等待 Safe/Near worker、PID 被覆盖的问题；
- 启用 JAX scan、环境对象/rollout/teacher metric 缓存；
- 不压缩 NPZ、不逐文件 fsync，以换取构建速度；
- merge 使用 hardlink；
- 支持 resume；
- 原始 roots 不被改写。

### 8.3 可进一步做但本轮未冒险启用

- 只对预筛 top-k candidate 执行昂贵 teacher rollout；
- 降低 targeted future 数或 option 数；
- 近似稀疏 compatibility matrix；
- 多进程预解析 TFRecord。

这些会改变 teacher/候选合同，可能贬损结果，因此当前默认保持完整 teacher 语义。应先用 profiler 找到瓶颈，再做成受控消融，而不是直接为了速度改主实验。

---

## 9. 推荐执行顺序

完整命令见独立文件 `OC-RAP-v48-run-instructions-ZH.md`。推荐：

1. 解压 v48 包；
2. 新建 distribution-matched train Near/Contact；
3. 首轮用现有 val scene-split calibration 筛查；
4. 若算法开始稳定，再构建专用 calibration roots；
5. Natural gate 通过后运行 development closed-loop probe；
6. 3 seeds 均通过后才运行 held-out test；
7. 最后同步修改 TeX 和补 baseline。

---

## 10. v48 成功/失败的判据

### 必须先通过的离线门

- Near candidate AUC 不低于 v47；
- Near/Contact group-top1 correlation 必须转正，建议 >0.10；
- verify selected teacher advantage mean >0；
- Contact verify harmful-selected rate 显著低于 v47 的 41.9%；
- fit→verify precision 不应出现 100%→38.7% 的坍塌；
- 正机会不再只由单一 macro/少数 scene 支撑；
- 至少有有限非零选择，不能靠全 abstain 过 gate。

### 闭环门

- Safe：collision/offroad/route/comfort/NUP 对 nominal 非劣；
- Near：minimum clearance/TTC、collision、PCD/DRS/FRA 至少有一致改善；
- Contact：secondary overlap、stable-stop、time-to-stop、post-contact clearance 改善；
- paired-scene bootstrap CI 支持结论；
- 至少 3 个随机种子方向一致。

### 若 v48 仍失败，如何定位

- candidate AUC 低：表示或标签仍不足；
- AUC 高、top1 低：setwise/候选特征或重复前沿仍有问题；
- fit 好 verify 差：scene 数和机会集中仍不足；
- gate 全 abstain：机会 head 或 harm 上界过保守，先看 oracle frontier coverage，不要先放宽 gate；
- Contact 有改善但论文 claim 不成立：需要真实 observed post-contact 数据，而不是继续包装 surrogate。

---

## 11. 验证状态

在当前容器中完成：

- `python -m compileall -q src tools`：通过；
- v48 主要 Shell 脚本 `bash -n`：通过；
- `PYTHONPATH=src pytest -q`：**97 passed，2 个非失败 warning**；
- 新增 selector harm gate 与保守专家聚合测试：通过；
- 尚未执行真实 WOMD/JAX/GPU 长实验，也未生成 v48 实验指标。

