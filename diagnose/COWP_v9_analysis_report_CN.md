# COWP 闭环 SOTA / CCF-A 投稿诊断与 v9 优化报告

## 0. 核心结论

### 0.1 当前能否声称闭环 SOTA？

不能。上传的 v8 结果包中只有完整的 `learned_offline` 结果和训练日志；`probe/waymax` 目录为空，说明执行在 offline mechanism gate 失败后停止。因而当前没有 v8 的 100 场景、1000 场景或 5000 场景新闭环结果。现有结果最多说明：

- witness threshold 已经能够改变最终候选集合，v7 的“阈值与决策断开”问题被修复；
- candidate-level certificate 有较好的 false-safe 排序能力；
- COWP 在离线选择上可以降低 FSR、CBS 和 HBCR，但代价是显著降低 EP、提高 fallback；
- 核心的 same-root option-set transport 仍没有被可靠学到，offline gate 因 NCF recall 过低而失败。

### 0.2 论文的核心 idea 是否成立？

**概念上成立，并且有投稿价值；当前 v8 实现尚不足以证明它成立。**

论文真正有价值的主张不是“更礼貌的规划”或“给他车负担加一个软成本”，而是：

> 对同一根场景和同一组他车自然行为模式，ego intervention 会把他车的自然选项集合运输、压缩或破坏。若一个 ego 候选只有依赖他车高负担让行才保持无碰撞，则该候选存在安全可行性缺陷，应被硬拒绝。

这个 setting 的关键是“集合的干预前后对应关系”，而不是一个普通的 pair/candidate 分类分数。只要把 supervised object、hard-first selector 和闭环验证做实，该 idea 具备成为 CCF-A 论文核心贡献的潜力。

### 0.3 当前最根本的问题

v8 的主要问题不是阈值没调好，而是以下五类机制错配：

1. **训练监督错配**：planner 阶段的 response set auxiliary loss 在全部 epoch 中为 0；模型没有学会“是否存在低负担安全响应”。
2. **集合支持错配**：OPR 分母在模型中包含所有 decoder slot，而标签只对有效、低自然负担模式归一化。
3. **负担目标错配**：`min_safe_burden` 曾被监督到 option-loss-adjusted `burden_total`；`c_i` 还曾被映射为 conflict mass，造成语义混淆和重复计数。
4. **不可识别的聚合监督**：仅用 aggregate OPR/conflict/source mass，许多完全不同的 mode-level transport 都能产生相同聚合量，模型可以绕开核心机制。
5. **评估口径错配**：v8 把最终 1–3 个 frontier short-list 当作“accepted set”计算 NCF recall。即使语义 gate 接受了大量 NCF 候选，后续 top-k 也会数学上把 recall 压得极低。

v9 已针对这些根因做结构性修复，而不是继续堆 selector 权重。

---

## 1. 论文 idea、算法 pipeline 与目标

### 1.1 核心定义

论文定义一种标准 collision-free 指标无法识别的失败：

- ego 候选自身在 rollout 中无碰撞；
- 但其无碰撞依赖某个关键他车硬刹、突让、放弃优先权、放弃合法 gap 或显著偏离自然驾驶模式；
- 因而该候选是 **false-safe**，其安全来自把冲突解决负担转移给他车。

对于每个关键 agent `i`，论文构建：

- 干预前自然选项集合 `A_nat_i`；
- 给定 ego candidate 后的安全响应集合 `R_safe_i(tau_e)`；
- 低负担安全响应集合 `R_low_i(tau_e)`；
- Option Preservation Ratio（OPR），衡量自然选项中有多少仍属于低负担安全集合；
- 最小安全响应负担及其相对自然状态的 coercion increment；
- 当自然低负担选项发生冲突且最低安全响应负担过高时形成 coercion witness。

### 1.2 应当成立的 hard-first 决策顺序

1. 物理可行性：候选轨迹、道路边界、碰撞与动力学检查；
2. 非强迫可行性：每个关键 agent 至少有低负担安全响应，OPR 高于阈值，coercion increment 不超过阈值；
3. 只在前两层均通过的候选内使用 ego utility / progress 排序；
4. 若集合为空，执行明确的 conservative fallback；
5. uncertain witness 可以进入校准/惩罚分支，但不能被高 utility 完全覆盖。

### 1.3 数据 pipeline

当前构建流程总体合理：

- WOMD `scenario` proto 用于场景结构、candidate、natural alternatives、safe responses 和 witness 伪标签；
- WOMD `tf_example` 用于模型输入 tensor cache 和 Waymax-ready state；
- 对每场景有限候选进行 Waymax candidate replay；
- 将 collision/offroad 等物理 outcome 附回 tensor cache；
- 分阶段或 warm-start 训练 response / witness / planner；
- learned-offline 做机制校准；
- Waymax 做真正闭环。

问题不在数据源组合，而在 v8 cache 缺少可识别的 per-natural-mode transport target，以及 planner 数据加载没有让 response auxiliary 真正非零。

---

## 2. v8 实验结果的证据

### 2.1 offline gate 明确失败

`mechanism_verification.json`：

- threshold points：9；
- unique selection points：9；
- threshold connected：true；
- witness AUPRC：0.429995；
- reported learned accept NCF recall：0.072716；
- pass：false。

这说明上一轮修改确实修复了 threshold 与 selection 断开的工程问题，但模型仍未达到机制有效性要求。

### 2.2 v8 COWP 与 conventional 的离线对比

| 指标 | Conventional | COWP v8 @ 0.30 | 变化 | 判断 |
|---|---:|---:|---:|---|
| EP | 0.3891 | 0.2218 | -0.1673 | 进展损失过大 |
| FallbackRate | 0.1043 | 0.3585 | +0.2541 | 过度拒绝 |
| FSR | 0.6624 | 0.5899 | -0.0725 | 有改善 |
| CBS | 0.9345 | 0.7879 | -0.1466 | 有改善 |
| OPR | 0.7413 | 0.7905 | +0.0492 | 有改善 |
| HBCR | 0.3980 | 0.2320 | -0.1660 | 有改善 |
| SelectedFalseSafeRate | 0.5933 | 0.3784 | -0.2148 | 筛选更保守 |
| SelectedWaymaxUnsafeRate* | 0.09350 | 0.09340 | -0.00010 | 基本相同 |
| Waymax outcome coverage* | 0.3008 | 0.1965 | -0.1043 | 覆盖不足 |

`*` 这里是 cache 中候选 replay outcome 的离线覆盖，不是真正在线闭环结果。

COWP 的负担指标改善是真实信号，但当前方式主要依赖“拒绝更多、fallback 更多”。要达到论文目标，需要在保持低 FSR/HBCR 的同时恢复 EP，而不是靠停车式保守策略获得指标。

### 2.3 threshold sweep 揭示明显 trade-off

| threshold | EP | fallback | Selected FSR | reported NCF recall |
|---:|---:|---:|---:|---:|
| 0.20 | 0.1609 | 0.4474 | 0.3202 | 0.0554 |
| 0.30 | 0.2218 | 0.3585 | 0.3784 | 0.0727 |
| 0.36 | 0.3019 | 0.2438 | 0.4696 | 0.0831 |
| 0.45 | 0.3817 | 0.1426 | 0.5568 | 0.0892 |
| 0.50 | 0.3979 | 0.1251 | 0.5725 | 0.0900 |

在 0.50 时 EP/fallback 接近 conventional，但 false-safe 改善已明显缩小。这说明目前模型还没有形成“同等进展下识别 coercive candidate”的判别边界，只是在 threshold 上做保守程度调节。

### 2.4 candidate certificate 比 pairwise witness 强

- Candidate false-safe AUPRC：0.82435；
- Candidate NCF AUPRC：0.56566；
- Candidate quality AUPRC：0.55882；
- Candidate risk pair ranking：0.80168；
- Pairwise witness AUPRC：0.429995。

candidate certificate 的排序能力是 v8 最明确起效的部分。但如果论文最终性能主要来自 candidate generic classifier，而不是 same-root transport certificate，审稿人会认为核心机制只是解释包装。因此 v9 将 candidate head 保留为**小幅有界校准项**，同时加强可识别的 mode-level 机制监督。

### 2.5 训练日志证明 response/set mechanism 没有真正学到

v8 共记录 17 个 epoch，并在第 16 epoch 后停止。关键现象：

- `train/set_transport/response = 0.0`；
- `val/set_transport/response = 0.0`；
- `val/set_transport/witness` 基本不下降；
- `val/set_transport/opr` 只有有限下降；
- backbone 从 epoch 0 起被冻结，配置为 `FREEZE_BACKBONE_EPOCHS=999`。

这与“大模型建议”中“planner response auxiliary supervision 已建立”的结论不一致。代码路径看似存在，但数据加载没有提供相应字段，最终 loss 是零。冻结全部 backbone 还使新证书只能在旧表示上拟合，无法学习低负担自然支持和 candidate-conditioned set transport。

---

## 3. 对上一轮大模型修改的逐项评价

### 3.1 明确有效，应保留

1. **threshold 连接到 selector**：9 个阈值产生 9 个不同选择点，修复有效。
2. **candidate certificate 与 planner score 分离**：false-safe AUPRC 和 ranking 较好，说明证书分支可用。
3. **hard-first 主路径与 Pareto ablation 分离**：将 Pareto 保留为 ablation 是正确方向。
4. **offline mechanism verifier**：阻止无效模型直接跑大规模闭环，工程设计正确。
5. **双卡 DDP 与 probe/full shard 并行**：运行组织合理。
6. **candidate replay outcome supervision**：能对 collision/offroad 等物理风险提供辅助监督。

### 3.2 部分有效，但需增强

1. **aggregate set-transport head**：比 proxy witness 更接近论文，但 aggregate OPR/conflict/source mass 不可识别，需要 per-mode transport label。
2. **compact response bank**：避免显存爆炸，但 v8 planner 数据未加载 response fields，导致辅助监督为零。
3. **analytic certificate + residual**：结构正确，但根支持、OPR 分母、负担 target 和 c_i 映射有误，解析项本身不可靠。
4. **冻结 backbone**：短暂 warm-up 有意义，冻结全部训练阶段无效。v9 改为 2 epoch warm-up 后解冻。
5. **mechanism gate 指标**：原 verifier 使用最终 short-list 的 recall，指标定义错误；需要单独统计 semantic feasible set。

### 3.3 未解决

- per-natural-mode 对应关系；
- mode collapse；
- intervention-conditioned burden field；
- semantic acceptance 与 final shortlist 的评估分离；
- 新机制表示学习；
- 新 v8 闭环结果；
- 在线 hard-negative mining；
- 外部强基线、统计显著性和 held-out test protocol。

---

## 4. v9 代码修改

代码目录：`COWP_v9_optimized`。

### 4.1 可兼容现有标签的修复

#### 数据加载

`cowp/data/dataset.py`

- planner/witness 阶段加载 compact response fields：valid、safe、low-burden、burden、source；
- planner/witness 阶段加载 natural 与 transport 字段；
- 使 `set_transport/response` 和 response auxiliary 不再恒为零。

#### 自然模式支持

`cowp/models/natural_decoder.py`

新增：

- `valid_logits`；
- `low_neutral_logits`；
- `neutral_burden`；
- mode latent。

OPR 的根集合现在由“预测 mode probability × valid probability × low-neutral probability”定义，不再把 padding/high-burden slot 计入分母。

#### 结构化 set certificate

`cowp/models/set_transport_head.py`

- 显式预测 mode conflict；
- 显式预测 intervention-conditioned mode burden；
- `mode retained = no-conflict × burden-below-beta`，而不是不受约束的 retain classifier；
- source-wise natural/conflict/retained mass；
- OPR 只在低负担自然支持上归一化；
- response existence 使用 valid-aware differentiable OR；
- min safe burden 使用 support-normalized soft minimum；
- `coercion_increment = min_safe_burden - natural_min_burden`；
- calibration residual 只在 logit space 做有界校准，不能覆盖解析证书。

#### loss target 修复

`cowp/models/losses.py`

- `min_safe_burden` 对应正确标签；
- conflict mass 不再 fallback 到 `c_i`；
- `c_i` 监督 coercion increment；
- 增加 root support、response existence、source mass、mode conflict、mode retention、mode burden 监督；
- 增加 per-mode label 对齐。

#### selector / metric 修复

`cowp/waymax_eval/rollout.py`

同时保留：

- `semantic_accepted_mask`：hard semantic gate 后的可行集合；
- `selected/frontier_mask`：最终用于排序的短名单。

新增指标：

- `SemanticFeasibleCandidateRate`；
- `SemanticAcceptNCFRecall`；
- `SemanticAcceptFalseSafeRate`。

mechanism verifier 改用 semantic set，而不是 final top-k。

### 4.2 新的 per-natural-mode transport labels

`cowp/label/witness.py` 与 `cowp/label/label_engine.py` 新增：

- `cowp/transport/mode_support [A,M]`；
- `cowp/transport/mode_conflict [K,A,M]`；
- `cowp/transport/mode_retained [K,A,M]`；
- `cowp/transport/mode_burden_under [K,A,M]`。

这些标签直接揭示：

- 哪个自然 mode 在根场景中属于低负担支持；
- 哪个 ego candidate 使哪个 mode 冲突；
- 哪个 mode 在干预后仍保留；
- 干预后的 mode burden 是多少。

这使聚合 OPR/conflict 具有可追溯的 mode-level decomposition。

### 4.3 Balanced Same-Root Optimal Transport（主要 novelty 增强）

v8/v9 初版曾用每个 predicted mode 独立最近邻匹配 GT mode。这允许多个 predicted modes 全部匹配同一个容易模式，造成 mode collapse，也不保留集合概率质量。

最终 v9 改为带熵正则的 balanced Sinkhorn transport：

- predicted natural mode probability 是源边缘质量；
- GT natural weight 是目标边缘质量；
- cost 由 trajectory ADE + source compatibility 构成；
- transport plan 同时满足两侧边缘质量；
- mode support/conflict/retained/burden label 通过 transport plan 形成 soft target。

因此“set transport”不再只是名称，而是训练目标中的真实质量守恒映射。该部分建议成为论文方法章节的主要技术创新之一：

> **Mass-conserving same-root option-set transport certificate**：在同一场景根节点下，将非受迫自然模式分布运输到 ego intervention 条件下的安全/低负担响应状态，并用 transport-induced conflict、retention 与 burden certificate 进行 hard feasibility filtering。

### 4.4 新工具与脚本

- `cowp/scripts/26_augment_transport_labels.py`：从现有 label NPZ 增量补齐 mode transport labels，无需重新做 Scenario proto 全流程；
- `cowp/scripts/27_verify_transport_cache.py`：重构 aggregate conflict/OPR，验证 mode label 与原标签一致；
- `prepare_cowp_v9_data.sh`：并行构建 train/val v9 labels/cache，并复用原 Waymax outcomes；
- `run_cowp_v9_dual_gpu.sh`：双卡主训练、offline gate、双卡 probe、双 shard full evaluation；
- `run_cowp_v9_ablation_dual_gpu.sh`：GPU0 运行 aggregate-only，GPU1 运行 nearest-match，对比主方法 Sinkhorn OT；
- `configs/train_cowp_v9*.yaml`：主方法和两项关键 ablation；
- 测试：54 项全部通过。

---

## 5. v9 为什么更可能让核心 idea 生效

### 5.1 从“可拟合聚合统计”变为“可识别机制”

aggregate OPR=0.5 可以由完全不同的模式组合产生：

- 一半 mode 被完整保留；
- 所有 mode 都以 0.5 概率保留；
- 高权重 mode 被破坏，低权重 mode 被保留；
- 多个预测 slot 塌缩到同一个 GT mode。

这些情况对 planning 含义不同。per-mode labels + balanced OT 能区分它们，减少 shortcut learning。

### 5.2 从“保留分类”变为“负担结构”

直接预测 retain probability 可以与预测 burden、beta threshold 相矛盾。v9 预测 mode burden，再通过 burden threshold 得到 low-burden probability，使证书可审计并与论文公式一致。

### 5.3 从“最终选中率”分离“语义可行率”

最终 planner 必须只选择少数候选，因此 final top-k recall 天然很低。v9 用 semantic set 衡量机制 recall，再单独衡量 shortlist/selected quality。这避免 mechanism gate 因指标定义错误而误判。

### 5.4 新表示可以被学习

v9 只冻结 2 epoch，让新 head 先稳定；随后解冻 natural/graph representation。v8 冻结 999 epoch 无法让 backbone 编码 root support、source 与 intervention burden。

---

## 6. 仍未解决、需要实验验证的问题

### 6.1 不能保证 v9 自动达到闭环 SOTA

本环境没有完整 WOMD cache、GPU 和 checkpoint，无法运行真实训练/Waymax。因此代码修改通过了静态、单元和机制一致性测试，但实际效果必须由用户机器上的实验验证。

### 6.2 online semantic domain gap

离线标签使用 counterfactual candidate rollouts，在线闭环状态会逐步偏离 log root。即使 offline gate 通过，online PredFSR/HBCR 仍可能升高。根本解决方式是：

1. 先用 v9 跑 100/1000 场景；
2. 收集在线被选择但发生高 PredFSR/HBCR、碰撞、offroad 或 fallback 的 root/candidate；
3. 重新 replay 这些 online frontier candidates；
4. 将其作为 hard negatives 回填 cache；
5. 只微调 set certificate / response / candidate residual，而不是大幅重写 planner。

### 6.3 Waymax outcome 覆盖仍不足

v8 selected outcome coverage 仅约 0.20–0.30。balanced 12 candidates 并不一定覆盖在线 selector 的边界。后续 replay 应从每场景增加：

- planner top candidates；
- semantic threshold 附近 candidates；
- high-progress/high-conflict candidates；
- fallback predecessor candidates；
- mode-conflict 最大的 candidates。

建议让“进入 semantic set 或 frontier 的候选”Waymax outcome coverage 达到至少 70%，再依赖 outcome head 证明安全提升。

### 6.4 候选生成器与论文不一致

论文写的是 graph-conditioned lattice-MPC，但代码实际是规则/运动学 primitive lattice，并没有求解短时域 MPC。论文还写了 conditional denoising diffusion neutral branch 和 conformal calibration，而当前代码没有实现对应模块。

投稿前必须二选一：

- **推荐**：将论文表述改成实际可复现的 learned multimodal natural decoder + kinodynamic primitive lattice + validation-calibrated certificate；
- 或真正实现 diffusion/MPC/conformal，并做相应 ablation。

不要保留无法由代码和实验支持的强描述，否则会成为 reproducibility/reviewer attack point。

### 6.5 外部基线不足

内部 baselines（conventional、planner-only、soft burden、universal NCF）只能验证机制，不足以支持 CCF-A 的 SOTA 主张。至少需要：

- Waymax 官方 IDM / log-playback 相关基线；
- 一个强 imitation/trajectory planner；
- 一个 ego-conditioned prediction-planning baseline；
- 一个 social/courtesy 或 risk-aware baseline；
- 同候选集上的“强 classifier”基线，证明收益不是 generic false-safe classification；
- oracle / label upper bound。

若外部 planner 无法完全接入相同候选空间，应报告两套结果：

1. same-candidate controlled study：公平比较 selector/certificate；
2. end-to-end closed-loop study：比较完整系统。

### 6.6 “SOTA”协议必须重新定义

Waymax 是闭环模拟器，不等同于一个统一的 ego-planning leaderboard。WOSAC/Sim Agents 的官方主指标是 realism meta-metric，主要评估多智能体仿真，不是本文的 ego non-coercive planning。论文更稳妥的 claim 应是：

- 在固定 Waymax closed-loop planning protocol 上，COWP 在标准 safety/progress 指标保持竞争力；
- 在新提出的 false-safe stress set 和 non-coercive metrics 上取得最佳结果；
- 通过 mechanism ablation 证明提升来自 same-root option-set transport，而非单纯保守或 generic classifier。

---

## 7. 投稿级实验设计

### 7.1 数据拆分

- train：训练 representation、response、transport certificate；
- validation：选择 threshold、alpha/beta/gamma、early stopping；
- held-out test：只运行一次最终结果；
- stress set：按 scenario id 与 train/val/test 严格去重；
- interaction strata：merge、unprotected turn、crossing、lane change、gap competition、priority conflict。

### 7.2 主表指标

标准指标：

- collision/overlap；
- offroad；
- wrong-way；
- route-following/off-route；
- kinematic infeasibility；
- progress；
- fallback；
- log divergence（若可用）。

核心指标：

- FSR；
- CBS；
- OPR；
- HBCR；
- Witness recall/precision/AUPRC；
- semantic NCF recall；
- semantic false-safe accept rate；
- stress set accept NCF / reject false-safe；
- calibration ECE/Brier 或 risk-coverage curve。

### 7.3 必做 ablation

1. Full v9 Sinkhorn set transport；
2. aggregate-only（无 per-mode supervision）；
3. nearest-mode matching（无质量守恒）；
4. w/o intervention mode burden；
5. w/o response existence；
6. w/o ego-neutral branch；
7. w/o priority-preserving branch；
8. soft burden cost only；
9. universal NCF gate；
10. Pareto frontier 替代 hard-first exact frontier；
11. candidate generic classifier only；
12. oracle label certificate。

其中 1–3 是最能证明新 novelty 的主消融，已提供双卡并行脚本。

### 7.4 统计报告

- 至少 3 个训练随机种子；
- 相同 scenario IDs 上做 paired bootstrap；
- 报告均值、95% CI 和相对变化；
- 对 FSR/HBCR 等 episode rate 使用 paired proportion bootstrap 或 McNemar-style paired test；
- 对 EP/OPR/CBS 使用 paired bootstrap；
- 阈值只在 validation 选择，不能在 test 上 sweep 后取最好值。

---

## 8. v9 go/no-go 标准

### 8.1 训练日志

必须同时满足：

- `train/val set_transport/response` 非零；
- `mode_conflict`、`mode_retain`、`mode_support`、`mode_burden` 非零；
- val mode losses 有下降；
- root support 不是全 0 或全 1；
- threshold sweep 至少产生多个 semantic selection points。

### 8.2 offline mechanism gate

建议最低标准：

- `SemanticAcceptNCFRecall >= 0.70`；
- `SemanticAcceptFalseSafeRate <= 0.35`；
- fallback <= 0.30；
- witness AUPRC 相比 v8 的 0.430 有明确提升，目标至少 0.52–0.55；
- candidate NCF/false-safe AUPRC 不低于 v8；
- 同等 EP 区间内 FSR/HBCR 优于 conventional 和 universal gate。

### 8.3 100 场景 probe

- collision/offroad/wrong-way/kinematics 不得劣于 conventional 的置信区间；
- PredFSR/HBCR 必须显著下降；
- EP 相对 conventional 的损失不超过约 10–15%；
- fallback 最好低于 0.25；
- 主方法优于 Pareto ablation，证明 hard-first 机制有效。

### 8.4 1000 / 5000 场景

100 场景只用于 smoke/probe，不能用于 SOTA 结论。1000 场景用于开发，5000+ 场景和 3 seeds 用于主表。若可使用完整 held-out validation/test，应优先覆盖更大规模和固定 scenario list。

---

## 9. 论文必须立即修正的内容

1. 删除文件末尾 stray triple backticks，否则 LaTeX 会报错；
2. 使用了 `\mathbbm{1}`，但 preamble 没有加载 `bbm`/`dsfont`；可以统一改用已定义的 `\ind`；
3. 主结果表和消融表仍是 `--`，摘要却声称“substantially reduces”，在真实结果产生前应改成中性表述；
4. 修改不一致的 diffusion/MPC/conformal 描述；
5. 明确 pseudo-label teacher 使用 privileged future 的边界，online planner 不可读取 future；
6. OPR 需要明确定义 mode measure、natural weight 和 source；
7. 说明 balanced OT 的 cost、marginal、temperature、iterations 和 mode masking；
8. 区分 semantic feasible set、frontier shortlist 和 final selected candidate；
9. 明确 false-safe stress set 的构建规则与去重；
10. 将 SOTA claim 限定到固定 protocol，不要把 WOSAC realism leaderboard 与 ego planning 混为一谈。

---

## 10. 文件完整性说明

本轮实际可访问的上传中没有单独挂载 `cache_sufficiency_full.json` 和 `大模型建议.md`。代码包中的 `COWP_v8_analysis_report.md` 已包含上一轮大模型的分析与修改说明，因此本报告据此核对了建议是否落实；但没有重新核验 `cache_sufficiency_full.json` 的原始统计。该缺失不影响本次最关键结论，因为 response loss 为零、backbone 全冻结、gate 失败和无闭环结果都可由实际代码、日志和结果 JSON 直接确认。

---

## 11. 最终判断

- **idea**：有潜力，且比一般 courtesy cost 更具理论和实证价值；
- **v8**：修复了 threshold connectivity 和 candidate certificate，但核心 same-root set mechanism 只部分生效；
- **v8 结果**：显示负担指标改善，但主要通过过度拒绝，且没有新闭环结果；
- **v9**：把 supervised object 改为 per-mode、mass-conserving、burden-structured transport，并修复 response、target、selector metric 和冻结策略；
- **下一关键证据**：不是继续手调 selector，而是 v9 offline semantic gate、100 场景闭环 probe、在线 hard-negative mining，以及 1000/5000 场景多 seed 统计；
- **投稿策略**：主张“新型非强迫可行性 + 同根选项集合最优传输证书 + false-safe benchmark”，标准闭环指标保持竞争力，核心指标达到最佳，而不是在不统一的 Waymax 自定义协议上笼统声称全局 SOTA。
