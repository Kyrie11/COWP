# COWP v13 失败诊断、v14 算法修改与 CCF-A 执行方案

## 0. 结论先行

本轮最重要的判断不是“natural decoder 容量不够”，而是：

1. **v13 natural gate 的失败首先表现为模型输入路径上的近常量空间平移。** 验证集 1 s / 3 s / 5 s / 8 s 的误差约为 34.51 / 34.75 / 35.12 / 35.85 m，误差几乎不随时域增长。这不符合正常预测误差随时间累积的形态，更接近关键体索引、当前状态锚点或坐标帧不一致造成的整条轨迹平移。
2. **数据标签本身并不难。** 同一验证数据上的运动学 oracle 在 1/3/5/8 s 的平均 minADE 为 0.283/0.608/1.204/2.466 m；所以不能把 35.85 m 归因于“自然轨迹天生不可预测”。
3. **即使锚点问题被排除，v13 的 root 学习结构仍然不成立。** 24 个 mode 从完全相同的 constant-velocity 曲线初始化，再使用跨 OBS/NEU/PRIO 的全局最近邻匹配，会让多个异质 GT root 抢占同一预测 mode；轨迹覆盖与语义身份相互冲突，RIOT 后续的“同 root transport”失去可辨识性。
4. **当前结果不能评价 RIOT 有效或无效。** natural basis gate 未过，transport、planner、完整闭环均未运行；因此只能证明 v13 的自然选项基础不可用，不能据此否定论文核心机制。
5. **v14 的目标不是直接追一个更低 ADE，而是先建立可识别的 Typed Natural Option Basis（TNOB）。** 每个 root slot 拥有固定来源身份、解析运动学先验和小幅有界残差，训练与评估均进行同源匹配；在此基础上才允许训练 direct root transport 和 planner。
6. **“CCF-A 指标门槛”不存在统一数值。** CCF 的 A/B/C 是会议/期刊推荐类别，不是某个 planning benchmark 的固定录用分数。本文需要的是：同协议强基线、完整验证集、多随机种子、置信区间、机制消融和独立闭环证据，而不是把某个碰撞率当成通用门槛。

---

## 1. 论文核心 idea、目标与可证伪命题

### 1.1 核心问题

论文提出 **false-safe planning**：ego 轨迹在 rollout 中无碰撞，但安全成立的原因是另一交通参与者被迫硬刹、突然让行、放弃合法优先权或让出原本可用的 gap。传统 ego-centric collision/safety 指标会把这种方案判为安全，但它实际上把冲突负担转移给了他人。

### 1.2 核心定义

论文把这种问题从“礼貌程度的软代价”提升为 **non-coercive feasibility**：

- 对每个 critical agent，先构造其不受 ego 压迫时合理存在的 natural alternatives；
- 在给定 ego candidate 后，求该 agent 仍然安全且低负担的 response set；
- 若自然的低负担选项被 candidate 消灭，且仅剩高负担 ceding response，则 candidate 存在 coercion witness，应被硬拒绝，而不是仅加一点 social cost。

### 1.3 COWP pipeline

论文和代码对应的主 pipeline 为：

1. **Burden-oriented interaction graph**：识别谁与谁冲突，以及谁被迫吸收冲突负担；
2. **Ego candidate generation**：产生候选 ego 轨迹；
3. **Counterfactual natural alternatives**：为 critical agents 构造 OBS、NEU、PRIO 三类自然 root；
4. **Ego-conditioned response prediction**：预测每个 candidate 下、每个 natural root 对应的安全/低负担响应；
5. **Root-indexed option transport**：估计哪些 root 被保留、破坏或仅能高负担恢复；
6. **Non-coercive feasibility certificate**：按照 per-agent / per-root option preservation 和 burden budget 判定候选；
7. **Coercion witness**：输出 burdened agent、冲突区间、让行机制和负担分量；
8. **Hard-first selection/fallback**：先过滤物理不安全和 coercive candidate，再按进度、舒适、连续性排序；无可行项时执行明确 fallback。

### 1.4 投稿必须成立的四个可证伪命题

仅“闭环碰撞率降低”不足以支撑论文。至少需要同时证明：

- **P1：false-safe 是传统指标遗漏的独立失败模式。** 在 collision-free candidate 中，存在可重复识别的 induced hard brake / priority abandonment / option collapse。
- **P2：自然 root 是可识别且稳定的。** OBS/NEU/PRIO 不是训练后才任意交换的无序 mode，而是可跨 candidate 对齐的机制坐标。
- **P3：direct same-root transport 是有效机制，而不是 candidate-level 分类器的别名。** 它应显著优于 pairmax、candidate-only certificate 和 response-bank-only recovery。
- **P4：机制收益不靠明显牺牲常规驾驶质量。** 在同一闭环协议下，false-safe/induced burden 显著下降，同时 collision/offroad 不恶化，进度、舒适和 fallback 满足非劣约束。

当前 v13 首先卡在 P2，所以 P3/P4 暂时没有被验证。

---

## 2. 数据集构建与数据性质

### 2.1 构建链路

根据数据构建指令和代码，数据由两条 WOMD 数据源合并：

- `uncompressed/scenario/...` proto：用于场景索引、critical relations、natural roots、responses、witness 和 planner 标签；
- `uncompressed/tf_example/...`：用于模型实际输入的 actor history、map 和 Waymax-ready state；
- 先建立 scenario index，再由 proto 建标签；
- 将标签与 tf.Example tensor cache 合并；
- 对候选轨迹做 Waymax replay，附加 collision/offroad 等 outcome；
- 后续 `tensor_cache_*_waymax_transport_v9` 以 overlay/sidecar 方式增加 root transport 标签。

### 2.2 当前可确认的数据性质

从 v13 的 `natural_oracle_val.json`：

| 指标 | 1 s | 3 s | 5 s | 8 s |
|---|---:|---:|---:|---:|
| 全部 natural root kinematic minADE | 0.283 m | 0.608 m | 1.204 m | 2.466 m |
| OBS | 0.590 m | 0.801 m | 1.441 m | 3.010 m |
| NEU | 0.182 m | 0.559 m | 1.111 m | 2.113 m |
| PRIO | 0.058 m | 0.417 m | 1.074 m | 2.594 m |

这说明：

- 标签在短时域与简单运动学高度一致；
- 8 s 仍可由小型运动学 bank 达到约 2.47 m；
- OBS 比 NEU/PRIO 略难，但没有难到 35 m；
- model gate 设为 oracle + 6 m（约 8.47 m）是宽松的工程门槛。

### 2.3 尚未独立核验的部分

用户提到的 `cache_sufficiency_full.json` 本轮没有作为独立文件出现在挂载目录或两个压缩包中，因此本报告不能重新计算其中的每个统计量。上一轮诊断文档记录：natural/response/witness/planner 关键字段覆盖基本完整，但 attached Waymax candidate outcome 对 valid candidates 的覆盖约为 train 23.44%、val 23.70%。该结论目前仅作为历史二手证据，v14 不把稀疏 outcome 当作完整闭环真值。

此外，实际服务器上的 `tensor_cache_*_waymax_transport_v9` 未随压缩包提供；raw 与 transport overlay 的实体对齐必须由新的快速诊断在服务器上重新验证。

---

## 3. v13 natural gate 为什么失败

### 3.1 实际结果

v13 最优 epoch 29：

| 指标 | v13 | gate/参考 | 结论 |
|---|---:|---:|---|
| set minADE@8s | 35.845 m | oracle+6 = 8.466 m | 失败 |
| minADE@1s | 34.514 m | 3.0 m | 严重失败 |
| minADE@3s | 34.749 m | 8.0 m | 失败 |
| branch minADE | 37.762 m | 15.0 m | 失败 |
| OBS / NEU / PRIO | 40.982 / 36.181 / 32.904 m | 各 15.0 m | 失败 |
| source CE | 0.915 | improvement ≥0.01 | 仅改善 0.0072 |
| priority BCE | 0.312 | improvement ≥0.005 | 通过 |
| neutral consistency | 37.674 m | improvement ≥0.5 m | 仅改善 0.049 m |

训练集 natural trajectory loss 大致在 24–28 m，而验证集长期停留约 36 m，说明还存在明显的分布/索引路径差异或过拟合；但最强证据仍是 1 s 和 8 s 几乎同样差。

### 3.2 第一层根因：模型实际使用的 agent anchor/索引/坐标路径高度可疑

正常的时域预测错误应随时间增长。当前：

- 1 s：34.51 m；
- 8 s：35.85 m；
- 增量仅 1.34 m。

这非常接近“GT 和 prediction 各自形状合理，但整体被平移约 35 m”。最可能的子原因包括：

1. `cowp/critical/track_index` 是 scenario proto 中的原始 track index，而 tf.Example actor tensor 可能经过筛选/重排；模型应使用已映射的 `cowp/critical/input_index`；
2. 某些 cache 中 `input_index` 缺失、错误或在 merge/overlay 后被覆盖；
3. model-facing history 当前帧和 natural label 起点使用不同坐标系或不同时间索引；
4. critical row 虽在范围内，但指向另一个可见 actor，因此简单的 in-range check 无法发现；
5. train 和 val cache 的字段版本或 mapping 生成逻辑不一致。

在没有实际 cache 的情况下不能武断指定其中某一个为已证实根因。为此 v14 新增 `35_diagnose_model_anchor.py`，它严格复现训练路径：

`TorchCOWPDataset → _agent_history_from_batch → input_index → _safe_critical_indices → _critical_anchor7 → typed_kinematic_basis`

并同时报告：

- critical unmapped/invisible rate；
- label 首帧与 model CV anchor 的误差分布；
- `input_index` 与原始 `track_index` 对应 anchor 的空间差；
- exact model-facing typed basis 的 1/3/5/8 s source-stratified minADE。

这个预检失败时禁止训练，它会把问题定位到真实模型输入路径，而不是继续用 label-only oracle 猜测。

### 3.3 第二层根因：v13 的 24-mode 表示存在确定的结构性退化

即使数据锚点完全正确，v13 仍有以下必然问题：

- 24 个 mode 的初始轨迹都是同一条 constant-velocity curve；
- GT natural alternatives 是 OBS/NEU/PRIO 异质集合；
- 全局 nearest-mode assignment 允许任意 predicted mode 匹配任意 source；
- 一个 mode 可以被多个 GT roots 抢占，其他 mode 没有有效梯度；
- source/priority 语义靠辅助 head 后验贴标签，不是 root slot 的结构属性；
- mode identity 可在不同 scene、epoch、candidate 之间交换；
- RIOT 所需的 “same root before/after candidate” 因而没有稳定语义。

这解释了为什么 priority BCE 能学到一个 aggregate 信号，而 source CE、neutral consistency 和轨迹恢复几乎不动。

### 3.4 训练策略问题

- v13 从旧 checkpoint warm-start，但 natural head 的结构已经变化；部分旧 natural 参数继续加载会污染新 root basis；
- natural stage 全程冻结 graph，使新 decoder 无法修复 encoder 对 critical-agent/root 任务不适配的问题；
- 没有 epoch -1 解析基线验证，训练可能把一个本来合理的运动学 prior 训坏；
- 只用“相对初始 improvement”做 semantic gate，会错误惩罚一开始已正确的 typed prior；
- 缺少 prior preservation、residual magnitude 和短时域强监督；
- 缺少 typed-vs-untyped coverage gap，模型可能通过 source 限制“虚假过 gate”，但几何覆盖已经崩溃。

---

## 4. 为什么原 raw/transport alignment 与 oracle 特别慢

### 4.1 `33_diagnose_cache_alignment.py` 的原瓶颈

- 每个 scene 同时完整解压 raw 和 transport NPZ；
- 对大型 state/map 数组做全量 SHA1；
- transport v9 常是 overlay/symlink，同一 raw payload 被重复读取和 hash；
- natural/root 统计使用 Python 多层循环；
- 单进程串行，没有进度反馈；
- 即使只需要少量 key，也读取整个 NPZ。

### 4.2 `34_diagnose_natural_oracles.py` 的原瓶颈

- 每个 NPZ 完整加载；
- scene × agent × root × horizon 使用 Python 循环；
- 1/3/5/8 s 反复重建 bank、重复计算相同距离；
- 无并行、无批量向量化。

### 4.3 v14 优化

- 只读取诊断需要的 NPZ members；
- 对 symlink / samefile payload 直接判同，避免重复解压和 hash；
- 默认 sampled hash，只有审计争议时才启用 full hash；
- 使用线程池并行 I/O；
- root 与 horizon 距离向量化，一次计算 cumulative distance 后切 1/3/5/8 s；
- 增加 `--workers`、`--max-scenes` 和进度输出；
- 推荐先 256/500 scene smoke，再 2000 scene promotion audit，而不是每次全量扫描。

注意：NPZ 解压受磁盘和 CPU 影响，线程数不是越大越好。NVMe 常用 8–12 workers；共享 HDD/NFS 建议 4–8。

---

## 5. v14 算法：Typed Natural Option Basis（TNOB）

### 5.1 结构

默认 24 roots 固定分成：

- mode 0–7：OBS；
- mode 8–15：NEU；
- mode 16–23：PRIO。

每类内部使用不同的解析加速度、yaw-rate 和速度偏置原型。模型不再从 24 条相同 CV 曲线开始，而是从具有覆盖性和语义的 kinematic basis 开始。

### 5.2 学习形式

预测轨迹为：

`typed analytic basis + zero-initialized bounded residual`

特性：

- 初始网络精确等于解析 basis；
- 残差门控初始接近关闭；
- position/yaw/velocity/size 残差均有物理范围约束；
- `base_deviation` 和 `residual_l2` 防止训练抹掉先验；
- 保留 legacy temporal decoder 作为论文消融。

### 5.3 训练匹配

- OBS GT 只能匹配 OBS slots；
- NEU GT 只能匹配 NEU slots；
- PRIO GT 只能匹配 PRIO slots；
- PAD 可忽略/回退；
- source CE 与 priority BCE 监督 matched typed mode；
- branch minADE 和 1/3/5 s minADE 同样 source-restricted；
- 额外报告 untyped set minADE；typed-untyped gap 过大则 gate 失败。

这不是把答案硬编码进网络。固定的是 **root 类型坐标系**，具体 scene 中的轨迹、概率、残差和 candidate-conditioned transport 仍由模型学习。它的作用是让论文所需的 counterfactual root identity 可辨识。

### 5.4 训练策略

- 从旧 checkpoint 只继承 graph/backbone，显式 reset `natural_decoder`；
- epoch -1 先评估解析 TNOB，并保存为 best baseline；
- natural graph 仅 warmup 2 epochs 冻结，随后小学习率联合微调；
- gradient clip = 1.0；
- 1/3/5 s loss 强化短时锚点；
- 训练后用绝对 gate 或 improvement gate，避免解析 prior 已好却因“改善不够”被拒绝。

### 5.5 novelty 是否被削弱

不会。论文 novelty 仍应写成：

1. false-safe / non-coercive feasibility 的问题定义；
2. natural option preservation 的可行性认证；
3. direct root-indexed intervention transport；
4. coercion witness 与 hard-first planner。

TNOB 是实现这些机制的 **identifiability layer**。不要把论文包装成“一个更好的多模态轨迹预测器”；要强调如果 root identity 不稳定，counterfactual transport 就没有机制含义。

---

## 6. 哪些算法保留、增强、修改或停止

### 6.1 应保留并增强

- false-safe 与 non-coercive feasibility 定义；
- burden-oriented graph 和 critical relation mining；
- OBS / NEU / PRIO 的多来源自然选项设计；
- candidate–natural relative geometry；
- direct root transport；
- per-agent option preservation 与 candidate-level budget，而不是跨 agent 的简单 any/max；
- witness agent / interval / mechanism 的可解释输出；
- hard safety first、机制 feasibility second、效率/舒适 third；
- Waymax collision/offroad/wrong-way/kinematic/log-divergence 等标准物理指标；
- plan continuity 只能在可行 frontier 内排序，不能把 coercive plan 重新变为可行。

可增强点：

- 将 burden 分量分为 induced deceleration、jerk、gap surrender、priority abandonment、safe-option mass loss；
- 对 critical agents 使用 uncertainty-aware budget，而不是所有场景固定阈值；
- 对 root transport 做 monotonicity / intervention consistency 正则；
- 对 witness 做 interval IoU、mechanism token accuracy 和 calibration；
- 构造 interaction-heavy stress split，并做人工抽检。

### 6.2 必须修改或只保留为消融

- 24 条相同 CV 初始化；
- 跨 source 的全局 nearest matching；
- aggregate-only source/priority supervision；
- pairmax/any 作为正式聚合器；
- candidate-only false-safe classifier 作为主机制；
- response-bank-only recovery 代替 same-root recovery；
- 用同一模型预测的 FSR/CBS/OPR 同时筛选和证明自身机制；
- 把稀疏 attached Waymax outcomes 当完整 planner ground truth；
- natural gate 未过就上 RL、planner 或全量 Waymax；
- 只调 threshold/budget 来制造更低 false-safe，但 fallback 和 progress 大幅恶化。

### 6.3 需要警惕的标签循环论证

如果 natural roots、response burden 和 witness 都由同一套启发式规则生成，模型在离线标签上高分只证明拟合了规则。论文需要至少一种独立证据：

- reactive simulator 中 non-ego induced braking / jerk / gap loss；
- 与 ego candidate intervention 配对的 counterfactual rollouts；
- 小规模专家人工审计；
- 规则阈值外的 continuous burden metric；
- 另一套 agent policy 下结论稳定。

---

## 7. 当前阻碍论文核心 idea 成立的问题

1. **Natural root identity 未建立。** v13 的核心阻塞。
2. **独立反事实真值不足。** logged replay 中 non-ego 不响应 ego 偏离，不能直接证明“被迫刹车”。
3. **同模型自评风险。** online predicted mechanism metrics 只能作为健康诊断。
4. **候选覆盖可能限制上界。** 若 candidate generator 中没有非胁迫且有进度的候选，certificate 只能 fallback，无法展示机制优势。
5. **critical relation/priority 标签偏差。** 需要人工审计和规则敏感性实验。
6. **root 数量固定与场景复杂度不匹配。** 24 个 root 需报告 coverage/saturation；可做 adaptive top-K 或 slot pruning 消融。
7. **transport supervision 稀有且不平衡。** false-safe/NCF positive 需报告有效样本数和每场景分布。
8. **没有 oracle transport / oracle planner 上界。** 无法判断瓶颈在 natural、transport、selector 还是 candidate set。

---

## 8. 当前阻碍闭环 SOTA / CCF-A 投稿的问题

1. **还没有可比较的完整闭环结果。** v13 gate 失败后未运行后续阶段。
2. **协议不够强。** Waymax logged playback 适合物理安全与轨迹执行，但不够验证交互因果；至少增加 IDM，最好再增加 learned reactive agent protocol。
3. **没有统一复现的强基线。** 应在完全相同 candidate set、控制频率、horizon、fallback、traffic policy 下比较 conventional safety、soft social cost、pairmax、candidate-only、response-bank-only、完整 COWP，以及一个近期强 planner/interaction baseline。
4. **样本量不足。** 100-scene 只能查 bug，不能宣称 SOTA；需要完整 5,013 val 或预注册的大规模固定 split。
5. **缺少多随机种子和 paired CI。** 至少 3 seeds；collision 等低频指标使用 paired bootstrap 或 Wilson/Clopper-Pearson CI。
6. **没有效率与稳定性报告。** 需报告 planner latency、P95、GPU/CPU、candidate count、fallback、plan switch rate。
7. **机制与常规性能没有联合 Pareto。** 需要展示 false-safe/burden reduction 与 collision/progress/comfort 的 frontier，而不是单点阈值。
8. **reactive 闭环中的 distribution shift 未处理。** 训练只使用 logged counterfactual labels 时，闭环响应可能超出支持集；需要 uncertainty/fallback 或 DAgger-like data aggregation 作为后续增强，而不是第一步就上 RL。

---

## 9. “CCF-A 指标门槛”的正确评估

CCF 最新目录仍将会议/期刊分为 A/B/C，但官方明确说明该目录是推荐列表，不应简单作为单篇论文评价依据。因此不存在“collision < X% 就能投 CCF-A”的官方门槛。

Waymax 官方提供 collision、offroad、wrong-way、kinematic infeasibility、log divergence 等指标，并支持 log playback 和 IDM。不同 non-ego policy、action space、候选生成器和 horizon 会显著改变绝对数值，因此跨论文直接比较单一 collision rate通常无效。nuPlan 的价值也在于统一场景、闭环 simulator 和多维指标。近期研究进一步表明，换成 learned reactive traffic agent 后，多数 planner 的分数和排序都会变化。

因此建议把以下数值作为 **本项目的内部 promotion/paper-readiness gate，而非官方录用门槛**。

### 9.1 Natural basis gate

- exact model-facing typed minADE@1s ≤ 3.0 m；
- typed minADE@3s ≤ 8.0 m；
- typed set minADE@8s ≤ min(12 m, oracle@8s + 6 m)；
- OBS/NEU/PRIO branch minADE 各 ≤ 15 m；
- source CE ≤ 0.30，或相对 epoch -1 有可信改善；
- priority BCE ≤ 0.45；
- neutral consistency ≤ 10 m；
- typed − untyped minADE gap ≤ 4 m；
- critical unmapped rate ≤ 2%，首帧 CV anchor error p90 ≤ 5 m。

### 9.2 Mechanism gate

开发 gate：

- pair witness AUPRC ≥ 0.60；
- candidate false-safe / BCOT AUPRC ≥ 0.65；
- direct conflict-conditioned root transport AUPRC ≥ 0.65；
- accepted NCF recall ≥ 0.30；
- accepted rate ≥ 0.10；
- fallback ≤ 0.25；
- 相对 conventional selected false-safe 至少下降 8 个百分点。

论文 gate：

- pair witness AUPRC ≥ 0.70；
- direct root transport AUPRC ≥ 0.75；
- direct root transport 相对 pairmax 与 response-bank-only 至少 +0.05 AUPRC；
- accepted NCF recall ≥ 0.50；
- accepted rate ≥ 0.20；
- selected false-safe 相对下降 ≥25%；
- option preservation 至少 +0.05；
- high-burden ceding rate 相对下降 ≥20%；
- fallback ≤ conventional +3 percentage points。

### 9.3 Closed-loop paper gate

在同协议、同场景的 paired comparison 下：

- collision 至少非劣；理想目标相对 strongest baseline 下降 15–20%；
- offroad/wrong-way 增幅不超过 1 percentage point；
- route progress 相对下降不超过 2%；
- comfort/jerk 不显著恶化；
- fallback ≤ conventional +3 percentage points；
- reactive protocol 下 non-ego induced hard braking / burden / option collapse 显著下降；
- 结果在 ≥3 seeds 和至少一个独立 traffic policy 下稳定；
- 报告 paired 95% CI 和 effect size。

“15–20% collision relative reduction”是工程上足够可感知的目标，不是 CCF 规定。若 strongest baseline 已极低碰撞，更合理的是 collision 非劣 + 机制指标显著改善。

---

## 10. 推荐实验顺序

### Phase 0：数据与 exact model path

1. 快速 256-scene raw/transport alignment；
2. 快速 500-scene oracle；
3. exact model-anchor preflight；
4. 通过后扩大到 2,000 scenes；
5. 任一 hard check 失败，修 cache/mapping，不训练。

### Phase A：TNOB natural-only

- reset natural decoder；
- epoch -1 保存解析 prior；
- 2 epochs frozen graph warmup，随后 joint fine-tune；
- gate 通过才 promotion。

必要消融：

- v13 identical-CV + global matching；
- typed slots + global matching；
- typed slots + typed matching；
- analytic-only；
- analytic + bounded residual；
- 去掉 base-preservation；
- OBS/NEU/PRIO 分支分别去除。

### Phase B：transport / RIOT

- 固定已通过的 natural basis；
- 首先看 natural-root assignment、direct root recovery 和 conflict-conditioned transport；
- 必做 pairmax、candidate-only、response-bank-only 消融；
- direct root transport 不优于这些消融时，不进入 planner。

### Phase C：planner + learned-offline

- calibrate budget 只能在 validation calibration split；
- final test split 不再调 threshold；
- 报告 false-safe、NCF recall、accepted rate、fallback、progress、burden、calibration；
- 做 oracle natural / oracle transport / oracle selector 上界分解。

### Phase D：100-scene Waymax smoke

只检查：

- action frame、yaw、SDC index；
- collision/offroad 数值是否异常；
- replanning 是否抖动；
- baseline 与 COWP 是否跑相同场景。

### Phase E：完整闭环

- 先完整 5,013 val logged replay，验证物理安全和轨迹执行；
- 再运行 Waymax IDM；
- 再增加 learned reactive traffic agent 或独立 multi-agent simulator；
- logged playback 与 reactive 结果分表报告，严禁把前者称为 reactive evaluation。

---

## 11. v14 已修改的代码

核心文件：

- `cowp/models/natural_decoder.py`：TNOB、解析原型、有界 residual、固定 mode source；
- `cowp/models/cowp_model.py`：同时 anchor `base_traj`；
- `cowp/models/losses.py`：typed matching、source-restricted branch/horizon loss、prior preservation、typed/untyped diagnostic；
- `cowp/scripts/03_train.py`：epoch -1 eval、reset checkpoint prefix、graph warmup、grad clip；
- `cowp/scripts/32_gate_natural_basis.py`：绝对/改善双 gate、typed-untyped gap；
- `cowp/scripts/33_diagnose_cache_alignment.py`：选择性读取、samefile shortcut、sampled hash、并行；
- `cowp/scripts/34_diagnose_natural_oracles.py`：向量化、多 horizon 一次计算、并行；
- `cowp/scripts/35_diagnose_model_anchor.py`：exact model-facing hard preflight；
- `configs/train_cowp_v14.yaml`；
- `run_cowp_v14_dual_gpu.sh`；
- `tests/test_v14_typed_natural_basis.py`。

本地验证：

- `pytest -q`：76 passed；
- Python compile：通过；
- driver `bash -n`：通过。

本地未能验证：

- 服务器真实 raw/transport cache；
- v10 init checkpoint；
- A30 上训练收敛；
- Waymax/JAX runtime；
- reactive agent 闭环；
- v14 是否最终达到 SOTA。

因此本版本是“消除已知结构缺陷并增加硬诊断”的可执行修复，不是已取得 SOTA 的结果包。

---

## 12. 论文写作建议

- 不要在 abstract 中提前写“substantially reduces”直到完整闭环与 CI 得到；当前可改为“is designed to reduce”或在结果完成后填写量化数值。
- 把 TNOB 描述为 counterfactual root identifiability，而非一般 prediction trick。
- 明确区分：natural alternative、ego-conditioned response、safe/low-burden response、root preservation、coercion witness。
- 给出 false-safe 的最小形式化例子和 theorem/proposition：collision-free 不蕴含 non-coercive feasibility。
- 说明 hard feasibility 与 soft courtesy 的差别，并做软代价 baseline。
- 把 model-predicted online FSR/CBS/OPR 标为 proxy；真正机制结论来自 direct transport labels、reactive rollout 和人工审计。
- 结果表必须按 protocol 分开：offline mechanism、Waymax logged playback、Waymax IDM、learned reactive。
- 主表包含 strongest baseline；消融表证明每个 novelty component 的必要性；附录报告阈值敏感性和 failure cases。

