# COWP / RC-NCF 自动驾驶规划算法诊断与优化报告

## 0. 结论先行

当前结果不佳不是单一超参数造成的，而是四类问题叠加：

1. **闭环控制接口失真（最高优先级）**：Waymax 在线评估中 `KinematicsInfeasibilityRate=0.57–0.62`，说明超过一半场景曾出现运动学不可行动作。此时碰撞率、进度、log-divergence 都被控制器错误严重污染，不能用于判断规划算法优劣。
2. **witness 概率塌缩**：所有候选—关键体对的预测概率几乎集中于 `0.698–0.709`，阈值扫描基本不改变候选接受集合；AUPRC 仅 `0.3082`，WLA 约 `0.4964`，接近随机定位。
3. **论文核心机制未真正进入原训练/推理链**：原脚本直接从 response 阶段训练，没有 natural 阶段；原 planner 在线前向也不读取 natural decoder。因此现有结果不能验证论文声称的“观测、ego-neutral、priority-preserving 三分支反事实自然行为集”。
4. **评估与标签链存在硬错误/不足**：原在线分片合并器读取不存在的 `rollouts` 字段，使所有 merged 文件变成 `num_rollouts=0`；Waymax outcome 仅回放 12/64 个候选且只计算 safety，导致 log-divergence 全为 0；400 个在线场景也不足以支撑 CCF-A 级别结论。

因此，合理顺序不是继续调 `witness_threshold`，而是：

> **修评估与控制器 → 恢复跨世界机制 → 修 witness 校准 → 扩展 outcome 覆盖 → 再做大规模闭环与消融。**

本补丁实现了第一轮关键修复，并将方法升级为 **RC-NCF（Risk-Calibrated Non-Coercive Feasibility）**：用根场景自然行为 latent 与 ego 条件行为 latent 构造跨世界 witness，并以 Beta evidential uncertainty 构造不确定性上界证书。

---

## 1. 对论文 idea 与目标的完整理解

### 1.1 论文研究问题

论文识别了一类传统 collision-free 指标无法捕获的规划失败：

- ego 轨迹本身未碰撞；
- 但只有在其他交通参与者急刹、突然让行、放弃合法间隙或优先权时才能维持无碰撞；
- 这种“由他人承担冲突负担换来的安全”被定义为 **false-safe planning**。

这与普通 courtesy cost 的区别在于：论文不是把“给别人造成负担”当成一个可与效率交换的软代价，而是把它提升为**可行性缺陷**。

### 1.2 核心定义

对 ego 候选轨迹 \(\tau_e^k\) 和每个 critical agent \(i\)，论文构造：

- 自然替代行为集 \(\mathcal N_i\)：没有被当前 ego 候选施压时，周车合理采取的行为；
- ego-conditioned 安全响应集 \(\mathcal R_i^k\)：给定 ego 候选后，周车还能避免冲突的响应；
- 低负担安全响应集；
- option preservation ratio（OPR）：自然行为自由度在 ego 介入后保留了多少。

若某个自然、低负担行为被 ego 候选变得不安全，且剩余安全行为只能是高负担让步，则形成 coercion witness，并拒绝该候选。

### 1.3 方法管线

论文的方法逻辑可概括为：

1. **Burden-oriented heterogeneous graph**：编码 ego、交通参与者、车道、冲突区域和控制元素，边表示谁可能承担冲突。
2. **Ego candidate lattice**：生成 keep/yield/stop/merge/lane-change/cut-in 等候选。
3. **Counterfactual natural alternatives**：
   - observational branch；
   - ego-neutral intervention branch；
   - priority-preserving rule branch。
4. **Ego-conditioned safe responses**：对候选—关键体构造响应集合并计算 collision、burden、low-burden 状态。
5. **Witness certification**：判断是否存在“自然行为失效 + 仅剩高负担让步”的证据。
6. **Hard-first planning**：先做 conventional safety 与 NCF 可行性过滤，再依据 ego utility 排序。

### 1.4 论文目标与 CCF-A 要求之间的关系

这项工作的潜在价值在于提出一个传统 closed-loop planner benchmark 没有显式测量的机制级失败模式。它具备 CCF-A 潜力，但前提是：

- 证明 false-safe 不是人为标签定义出来的伪问题；
- 在强 baseline 上显著降低 false-safe/burden，同时不牺牲传统 closed-loop safety 和 progress；
- 证明 witness 具有跨场景校准性和解释性；
- 使用真正 reactive 的非 ego agent 验证“其他车辆被迫让步”，而不仅是在 logged replay 下输出一个模型预测的 FSR。

当前版本尚未达到这些证据要求。

---

## 2. 数据集构建与训练/测试逻辑

### 2.1 原始数据

当前流程同时使用 WOMD 的两种数据形式：

- `scenario/*.tfrecord*`：用于构造交互、自然替代、响应、burden、witness 等结构化标签；
- `tf_example/*.tfrecord*`：用于构建训练 tensor cache，并直接被 Waymax 加载进行在线闭环。

时域为 1 秒历史 + 当前帧 + 8 秒未来，10 Hz，未来 80 步。这与代码中的 `history_steps=11`、`future_steps=80` 一致。

### 2.2 原构建流程

原指令执行了：

1. 为训练/验证 Scenario proto 建索引；
2. 生成 22,000 个训练标签、5,000 个验证标签；
3. 数据诊断；
4. 从 tf.Example 构建训练/验证 tensor cache；
5. 每场景 balanced 选 12 个候选做 Waymax safety replay；
6. 将 candidate outcome 附加到 cache；
7. 分阶段训练 response → witness → planner；
8. learned-offline 与 Waymax online 评估。

### 2.3 数据层主要问题

#### 2.3.1 仅使用 WOMD 的小子集

22k train / 5k val 对验证方法原型足够，但对 CCF-A 级别的闭环 planner 训练偏小。尤其 false-safe 是较稀有且机制多样的交互事件，模型容易学习标签先验而非场景差异。

#### 2.3.2 outcome 覆盖过稀

每场景最多 64 个候选，却只回放 12 个。未回放候选在 outcome head 中要么被 mask，要么形成非常稀疏监督。模型很难学会候选之间细粒度的 closed-loop risk 排序。

#### 2.3.3 `metric-set safety` 导致 log-divergence 全缺失

当前 learned-offline 文件中所有方法的 `SelectedWaymaxMeanLogDivergence=0.0`。这不是模型达到了零偏差，而是候选回放没有生成该标签。

#### 2.3.4 interaction-heavy 筛选需报告选择偏差

当前标签构建开启 `require_interaction_heavy=true`。这对训练 witness 有利，但主表需要同时包含：

- 全验证集 closed-loop；
- interaction-heavy 子集；
- false-safe stress set。

否则审稿人会质疑模型只在人工筛选的高交互分布上有效。

#### 2.3.5 验证路径命名不一致

原数据指令 attach 输出为 `tensor_cache_train_waymax`，验证命令却检查 `tensor_cache_train_waymax_bal12_safety`。这会造成验证了另一个旧缓存或直接找不到目录。

---

## 3. 当前结果的定量诊断

## 3.1 Learned-offline（5,013 场景）

| 方法 | EP | FSR | CBS | OPR | HBCR | Selected False-safe | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| IDM/Lattice | 0.7604 | 0.7619 | 1.1293 | 0.6629 | 0.4905 | 0.6824 | 0.1043 |
| Conventional safety | 0.3900 | 0.6608 | 0.9352 | 0.7407 | 0.3990 | 0.5919 | 0.1043 |
| Planner score only | 0.4301 | 0.6577 | 0.9606 | 0.7288 | 0.4580 | 0.5781 | 0.0000 |
| Soft burden cost | 0.3919 | 0.6608 | 0.9352 | 0.7404 | 0.3992 | 0.5919 | 0.1043 |
| Universal NCF | 0.0001 | 0.5582 | 0.7462 | 0.8262 | 0.1031 | 0.1751 | 0.6862 |
| COWP | 0.3919 | 0.6608 | 0.9352 | 0.7404 | 0.3992 | 0.5919 | 0.1043 |

关键现象：

- **COWP 与 soft burden / conventional safety 几乎完全相同**，说明 witness gate 没有产生有效的候选区分。
- universal NCF 通过拒绝几乎所有候选降低 FSR，但 EP 接近 0、fallback 68.6%，属于“停车换安全”，不能作为有效结果。
- planner ranking pair accuracy 为 `0.7476`，说明 candidate score 具有一定排序能力；真正塌缩的是 witness 校准与 hard gate。

### 3.1.1 Witness 质量

- Recall：`0.9198`
- Precision：`0.2864`
- AUPRC：`0.3082`
- WLA：`0.4964`
- MTA：`0.3554`
- 概率分位数：
  - p10 = `0.69817`
  - p50 = `0.70722`
  - p90 = `0.70894`
  - p99 = `0.70912`

概率范围只有约 0.011，且整体约为 0.70。这是典型的**输出塌缩/先验拟合**。阈值 0.1–0.6 时所有 pair 基本都判正，0.8–0.9 时基本都判负，无法形成可用的风险排序。

### 3.1.2 “Outcome oracle”并非 NCF oracle

原 `outcome_oracle` 只使用 collision/offroad/logdiv 排序，并不观察 false-safe、burden 或 OPR。因此它即使在传统安全标签上最优，也可能选择高度 coercive 的候选。其高 FSR 不是理论矛盾，而是说明该 baseline 名称不准确：应改名为 **physical-outcome oracle**，并另设真正的 **NCF label oracle** 作为上界。

## 3.2 Waymax online（实际仅完成两个方法）

修复分片合并后，planner-score-only 的 400 场景结果为：

- CR：`0.385`
- CollisionRate：`0.335`
- EP：`0.6231`
- KinematicsInfeasibilityRate：`0.610`
- OffroadRate：`0.060`
- LogDivergence：`10.9343`

分片结果：

| 方法/分片 | CR | Collision | EP | Kinematic infeasible | Offroad | LogDiv |
|---|---:|---:|---:|---:|---:|---:|
| Planner-only shard 0 | 0.34 | 0.305 | 0.6127 | 0.62 | 0.04 | 10.5487 |
| Planner-only shard 1 | 0.43 | 0.365 | 0.6334 | 0.60 | 0.08 | 11.3200 |
| Conventional shard 0 | 0.34 | 0.295 | 0.6093 | 0.61 | 0.05 | 10.5346 |
| Conventional shard 1 | 0.44 | 0.375 | 0.5928 | 0.57 | 0.085 | 10.8637 |

这组结果首先暴露的是控制器问题，而不是 planner 高层策略问题。超过一半场景触发运动学不可行，任何方法间 1–3 个百分点的差异都没有解释价值。

同时在线特征存在饱和：

- 平均 critical agents：约 `5.85–5.91`，几乎填满上限 6；
- conflict tokens：每步恒为 `64`；
- PredFSR_episode = 1；
- PredHBCR_episode = 1；
- witness mean ≈ 0.697。

这表明在线构造器给模型的输入分布与离线标签分布严重不一致，并放大了 max-over-agents 的错误拒绝。

---

## 4. 根因分析

## P0：闭环动作不是动态一致的状态更新

原 `absolute_xy_yaw` 接口直接使用候选第一个点的位置和 yaw，再用 `dx/dt, dy/dt` 计算速度。候选轨迹经过 repair、截断和 lane-change 叠加后，位置、速度、yaw 不一定满足同一个运动学模型。Waymax 的 kinematics metric 会把这种状态判为不可行。

此外，原动作每一步独立生成，没有跨步 acceleration/jerk 记忆，可能出现：

- 速度在相邻步突变；
- yaw 与速度方向不一致；
- 横向位移与 yaw-rate 不一致；
- absolute state 与上一帧状态不满足积分关系。

## P0：在线合并器丢失真实 rollout

每个分片 JSON 中保存的是 `standard_metrics: list[dict]`，而原 merge 代码读取 `rollouts`。结果是 merged 文件中：

- `num_rollouts=0`；
- `steps=[]`；
- summary 为空或错误。

这是一个确定性代码 bug。

## P1：witness loss 被类别先验主导

原训练同时使用：

- focal BCE；
- `alpha=0.75`；
- 正样本 oversampling；
- mined positive/negative pair。

这些机制叠加后，训练分布的有效正样本率远高于真实验证分布。模型用接近常数的高概率即可获得较低训练损失和高 recall，却没有校准性。

原 loss 还缺少：

- 全 pair 分布上的 balanced BCE；
- 同场景 positive-negative pair ranking；
- pair witness 与 candidate false-safe/NCF 标签的一致性；
- 不确定性监督或 OOD 保守证书。

## P1：核心 natural mechanism 没有进入 planner

原 shell：

- 没有训练 natural stage；
- response 阶段关闭 response trajectory 和 components；
- planner 前向不调用 natural decoder；
- 在线只使用一个直接从 candidate-conditioned feature 输出的 witness MLP。

因此原实现更接近“候选—周车 pair 分类器”，而不是论文定义的 counterfactual natural-vs-conditioned certification。

## P1：全局坐标导致模型学习无意义的场景原点

WOMD 场景使用全局坐标。原模型把绝对 `x,y,yaw,vx,vy` 直接输入线性层/GRU，没有做以 SDC 为原点、SDC yaw 为朝向的刚体归一化，也没有显式 ego type embedding。这会：

- 增加跨场景函数复杂度；
- 破坏平移/旋转不变性；
- 让图网络浪费容量学习场景坐标系；
- 加剧 train/online domain gap。

## P1：在线 critical/token flooding

原在线 critical selector只排序，不设真正的风险阈值，常把所有 6 个槽填满；conflict builder 再把 64 个 token 全部填满，其中很多只是附近 roadgraph 点或重复的 candidate-agent closest approach。

witness gate 对 critical agent 做 `any/max` 聚合，slot 越多，至少一个假阳性的概率越高。即使单 pair FPR 只有 0.15，6 个独立 pair 的候选假拒绝概率也约为：

\[
1-(1-0.15)^6 \approx 0.623.
\]

## P1：候选 outcome 标签既稀疏又缺少 log-divergence

仅 12 个 balanced 候选有标签，而 planner 在 64 个候选上排序。outcome head 大量面对未监督候选，且 logdiv 恒为 0，无法学习“物理上无碰撞但偏离真实交互”的风险。

## P2：候选生成过于弱、地图约束过于粗糙

原在线候选主要是：

- constant acceleration；
- smooth stop；
- 固定横向 offset lane change；
- 简化 merge timing。

但没有显式：

- route polyline / goal；
- lane connectivity；
- traffic light state；
- stop line / right-of-way；
- curvature-aware Frenet reference；
- optimization-based feasibility refinement。

原 drivable check 只检查离“任意 roadgraph point”是否近，可能把 lane boundary、crosswalk 等误当可行驶中心线。

## P2：所谓闭环 social 指标仍主要是模型自评

当前 Waymax environment 仅控制 SDC；非 ego agent 沿 log/default 动态演化，`eval.yaml` 中的 `logged_replay/rule_reactive/learned_reactive` mixture 没有真正被使用。在线 FSR、CBS、OPR 是 policy diagnostic 的预测值，不是从其他 agent 的实际反应中测得。

要证明“强迫他人让步”，最终实验必须至少包含：

1. log replay：保证传统 safety 可比性；
2. IDM/rule-reactive agents：测试因 ego 行为产生的 braking/yield；
3. learned sim agents：测试更真实的交互响应；
4. 多 agent model 的 sensitivity analysis，防止 planner 过拟合某一种模拟器。

---

## 5. 已完成的代码优化

## 5.1 Ego-centric SE(2) 归一化

新增 `cowp/models/coordinate.py`：

- 从 `state/is_sdc` 或 `womd/state/is_sdc` 获取 ego 索引；
- 将 agent history、candidate trajectory、conflict center 平移到 ego 原点；
- 按 `-ego_yaw` 旋转；
- yaw 做角度 wrapping；
- velocity 同步旋转；
- decoder 的绝对轨迹标签保持原坐标语义。

这使编码器具备刚体平移/旋转不变性。

## 5.2 显式 ego type embedding

GraphEncoder 不再把 ego 与普通 vehicle 当成完全同类型节点，加入 ego mask/type embedding，避免网络依靠数组位置猜测 SDC。

## 5.3 Cross-world natural latent witness

这是本次最重要的算法升级。

原 witness 只接收：

\[
[z_{candidate}, z_{agent}^{cond}, z_{graph}^{cond}].
\]

新 witness 接收：

\[
[z_{candidate}, z_{agent}^{cond}, z_{graph}^{cond}, z_{agent}^{natural}].
\]

其中 `natural_latent` 来自**不输入 ego candidate 的根场景图**和 natural decoder。planner/witness 阶段会计算该 latent，但不解码昂贵的 `24×80×7` dense trajectory，从而兼顾机制一致性和在线速度。

这使 witness 真正成为“natural world 与 ego-conditioned world 的差分证书”，而非单纯 pair classifier。

## 5.4 RC-NCF：Beta evidential witness certificate

WitnessDecoder 新增 Beta evidence head，输出：

- `evidential_prob`；
- `epistemic_uncertainty = 2/(alpha+beta)`；
- 用于 hard gate 的风险证书：

\[
\hat p_{cert} = \operatorname{clip}(\hat p + \lambda_u u,0,1).
\]

候选排序使用均值风险，硬可行性使用 uncertainty upper confidence bound。其意义是：

- 对分布内高置信度非 coercive 候选不额外惩罚；
- 对 OOD/不确定场景保守拒绝；
- 不再依赖固定阈值解释所有场景。

## 5.5 修复 witness 监督

新增：

- full-pair class-balanced BCE；
- hard-mined focal BCE 与 balanced BCE 混合；
- 场景内 positive-vs-negative pair ranking；
- pair witness 到 candidate false-safe 的 noisy-OR consistency；
- pair OPR 到 candidate NCF 的一致性；
- evidential Beta-Bernoulli loss 与 evidence KL；
- planner 微调时 witness loss 缩放为 0.2，降低校准漂移。

同时将 focal alpha 从 0.75 调整为 0.50。

## 5.6 动态一致的一步 Waymax 控制器

新增 `_consistent_one_step_target`：

1. 从候选下一点得到 desired speed/yaw；
2. 对 longitudinal acceleration 做上下限裁剪；
3. 用上一时刻 acceleration 做 jerk limit；
4. 对 yaw-rate 做限制；
5. 用梯形积分同时生成 `x,y,yaw,vx,vy`；
6. 场景切换或 step=0 时重置 controller memory。

这直接针对 57%–62% 的 kinematic infeasibility。

## 5.7 切向一致的 quintic Frenet lane change

固定横向 offset 改为 quintic lateral profile，并由轨迹切向导出 yaw 与 velocity，避免位置向左变换但 yaw/速度仍向前的非完整约束冲突。

## 5.8 更严格的地图与动力学过滤

- roadgraph 仅使用 vehicle lane centerline type 1/2；
- 排除 bike lane、边界、crosswalk 等非中心线元素；
- 增加 sideslip、yaw rate、lateral acceleration、jerk 检查；
- drivable 距离阈值从 7 m 收紧到 5.5 m。

## 5.9 在线上下文限流

默认：

- online critical agents 最多 4；
- 需要同时满足当前距离、候选最近距离和风险分数阈值；
- pair conflict tokens 最多 24；
- map tokens 最多 12，且要求空间间隔；
- 在线 batch 显式包含 `state/is_sdc`。

## 5.10 修复闭环分片合并

新增 `cowp/scripts/17_merge_waymax_shards.py`：

- 保留并合并 `standard_metrics`；
- 正确聚合 `num_rollouts` 和 `steps`；
- 按 policy step/episode 数对诊断 summary 加权；
- 已用现有两个 planner shard 验证，正确得到 400 个 rollout。

## 5.11 自动阈值校准

新增 `cowp/scripts/18_calibrate_witness_threshold.py`：

- 在 calibration cache 上扫描阈值；
- 先约束 NCF recall 和 fallback；
- 再按 FSR、HBCR、EP 选择阈值；
- 若无阈值满足约束，选择最小约束违反解。

注意：最终论文必须使用独立 calibration subset，不能在同一验证场景调阈值再报告结果。

---

## 6. 一键执行流程

主脚本：

```bash
bash run_rc_ncf_2gpu.sh
```

默认执行：

1. 38 项回归测试；
2. 两卡并行重放 24 个 balanced candidate，计算 safety + logdiv；
3. attach 新 outcome cache；
4. `natural → response → witness → planner` 两卡 DDP 训练；
5. witness threshold calibration；
6. learned-offline baseline 两卡并行；
7. 每个 Waymax 方法两卡分片并行；
8. 正确合并在线结果。

常用方式：

```bash
# 已经构建好 rc24 outcome cache，只训练和评估
RUN_OUTCOME_REPLAY=0 bash run_rc_ncf_2gpu.sh

# 只评估已有 checkpoint
RUN_TESTS=0 RUN_OUTCOME_REPLAY=0 RUN_TRAIN=0 RUN_EVAL=1 bash run_rc_ncf_2gpu.sh

# 先做小规模在线 smoke test
TOTAL_ONLINE_SCENARIOS=100 bash run_rc_ncf_2gpu.sh

# 正式 2000 场景评估
RUN_OUTCOME_REPLAY=0 RUN_TRAIN=0 TOTAL_ONLINE_SCENARIOS=2000 FORCE_EVAL=1 bash run_rc_ncf_2gpu.sh

# 最终资源允许时，回放全部 64 候选
OUTCOME_CANDIDATES=0 FORCE_REPLAY=1 bash run_rc_ncf_2gpu.sh
```

---

## 7. 下一轮结果的硬性验收门槛

不要直接以“比旧结果稍好”作为成功标准。建议分三道门：

### Gate A：闭环管线正确

必须先达到：

- merged `num_rollouts` 与计划场景数一致；
- `KinematicsInfeasibilityRate < 0.05`，理想 `<0.01`；
- `OffroadRate < 0.05`；
- 每场景 candidate valid 均值 ≥ 8；
- critical agents 均值在 1.5–4.0；
- conflict token 均值不应固定等于上限；
- 无 NaN/Inf；
- 两个 shard 指标差异可解释。

若 Gate A 不通过，不应继续解释 COWP vs baseline。

### Gate B：witness 可区分、可校准

最低目标：

- AUPRC > 0.50；
- WLA > 0.65；
- MTA > 0.55；
- ECE < 0.08；
- p10/p90 至少有 0.25 的间隔；
- threshold sweep 应形成连续的 recall/precision/acceptance trade-off，而不是分段常数；
- `LearnedAcceptFalseSafeRate` 明显低于 `LearnedAcceptNCFRecall`。

### Gate C：planner 有有效 Pareto 改善

相对于 planner-score-only / conventional safety：

- FSR 相对下降至少 20%；
- HBCR 相对下降至少 15%；
- CBS 显著下降；
- EP 绝对下降不超过 0.03–0.05；
- collision/offroad 不劣化；
- fallback < 0.15；
- 95% bootstrap CI 不跨越 0；
- 至少 3 个随机种子。

只有通过 Gate C 才适合开始声称“接近 SOTA”。

---

## 8. 逐步达到 SOTA 的实验路线

### Phase 1：验证本补丁是否修复根问题

先运行 100 场景 smoke：

- planner-score-only；
- conventional safety；
- cowp。

重点只看 kinematic/offroad/candidate/token/critical diagnostics，不看论文主结论。

### Phase 2：修 witness 与 candidate outcome

- outcome 从 12 扩至 24；
- 再扩至全部 64 或采用 active replay：优先重放模型不确定度最高、排序 margin 最小的候选；
- 加入 logdiv；
- 训练 natural → response → witness → planner；
- 单独报告 pair calibration 和 candidate-level consistency。

### Phase 3：增强候选生成

当前 primitive lattice 很难达到强 planner 的闭环性能。建议下一版加入：

1. route/lane centerline polyline encoder；
2. Frenet longitudinal-lateral lattice；
3. traffic light / stop line hard constraints；
4. terminal state sampling + differentiable kinematic refinement；
5. candidate diversity loss或 DPP/coverage objective；
6. top-M candidate 局部优化。

一个可形成新贡献的方向是 **NCF-guided candidate refinement**：

\[
\min_{\tau_e} J_{ego}(\tau_e)+\lambda_r R_{phys}(\tau_e)
\]

subject to

\[
\max_i p_{cert}(W_{ei}|\tau_e) \le \delta,
\quad
\min_i OPR_i(\tau_e)\ge \alpha.
\]

将 witness certificate 的梯度用于候选局部优化，而不仅用于离散过滤，可显著增强算法性和 novelty。

### Phase 4：真正 reactive multi-agent evaluation

至少构建三种环境：

- log replay；
- IDM/rule reactive；
- learned reactive/sim-agent。

实际测量：

- other-agent hard braking；
- time headway loss；
- gap surrender；
- priority violation；
- induced acceleration/jerk；
- response-set contraction。

论文当前的 FSR/CBS/OPR 大多来自模型预测，审稿人会要求外部行为证据。

### Phase 5：强 baseline 与统计学

必须补充同一 candidate/action/Waymax 管线下的：

- IDM/lattice；
- conventional safety；
- learned score-only；
- soft social/burden cost；
- universal hard NCF；
- strongest available learning planner；
- physical-outcome oracle；
- NCF-label oracle。

若 PDM-Closed、PlanT、GameFormer 无法原样移植到 WOMD/Waymax，不能只在表中写名称和横线。要么完成可靠适配，要么选择在相同 benchmark 上可复现的公开 baseline，并明确评价边界。

### Phase 6：最终论文证据

- 3–5 seeds；
- 1,000–5,000 个独立在线场景；
- bootstrap 95% CI；
- calibration 与 test 场景分离；
- 全集、interaction-heavy、stress set 三张表；
- failure taxonomy 与定性视频；
- runtime、显存、候选数、模型参数量；
- false-safe 人工复核与 inter-rater agreement；
- simulator policy sensitivity。

---

## 9. 对 CCF-A 投稿成熟度的判断

### 当前成熟度

- **Idea：7.5/10**。false-safe / non-coercive feasibility 具有辨识度。
- **理论定义：7/10**。概念完整，但 burden、priority 与 natural alternative 的可识别性还需更严格讨论。
- **实现一致性（原版本）：3/10**。核心 natural mechanism 未进入 planner。
- **实验有效性（原结果）：2/10**。控制器和 merge bug 使闭环结果不可作为论文证据。
- **当前补丁后的工程基础：6/10**。关键接口、机制和损失已重构，但尚未在真实 WOMD GPU 环境重新训练验证。
- **CCF-A 投稿就绪度：约 35%–45%**。

### 最关键的投稿风险

1. 方法被认为是“定义新的标签 + 分类器 + hard filter”；
2. false-safe 指标由本模型自身定义和预测，缺乏外部验证；
3. 只在 logged non-ego 下评价，无法证明 coercion；
4. 主表仍为空，外部强 baseline 未实现；
5. 传统 closed-loop collision 仍偏高；
6. 结果规模和统计不够。

RC-NCF 的 cross-world latent、evidential certificate 和未来的 certificate-guided candidate refinement，可以把方法从 heuristic filter 提升为一个更完整的可学习反事实可行性框架。

---

## 10. 本次可验证范围与限制

在当前沙箱中没有用户的 WOMD 数据、Waymax GPU 环境和训练 checkpoint，因此无法实际完成几十小时级的候选重放与 DDP 训练，也不能声称补丁已经提高了真实闭环指标。

本次已经完成并验证的是：

- 静态代码审查；
- eval 文件定量解析；
- 架构与损失重构；
- 动作接口与候选生成修复；
- 分片合并器修复；
- 一键两卡脚本；
- 全量单元测试：**38 passed**；
- 使用现有 shard 文件验证修复后的 merge：`num_rollouts=400`，不再为 0。

下一步应严格按 Gate A → Gate B → Gate C 执行，而不是一次跑完整训练后只看最终 CR/EP。
