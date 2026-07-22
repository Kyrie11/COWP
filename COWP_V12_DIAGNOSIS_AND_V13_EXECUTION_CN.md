# COWP v12 结果诊断、核心论证审计与 v13 执行方案

## 1. 审计范围与结论

本次审计覆盖：

- 论文 `interactive planning(1).tex`；
- COWP 代码及根目录 `ALGORITHM_CHANGELOG.md`；
- 数据构建指令；
- `cache_sufficiency_full(1).json`；
- `cowp_v12_riot_probe100_seed2026(1).zip`；
- v12 之前的 v8--v11 算法修改和失败记录。

当前上传材料中没有找到用户提到的“快速评估指令.txt”，也没有实际的 raw/transport tensor cache、初始化 checkpoint 或 Waymax Python 环境。因此，我能够完成代码级审计、结果解释、静态修复和可执行诊断工具，但不能在本地代跑 WOMD/Waymax 大规模实验，也不能诚实地宣称已经达到 SOTA。

**核心结论：当前 v12 结果既不能证明论文核心 idea 成立，也不能证明其不成立。原因不是 RIOT 已被实验否定，而是 v12 在自然选项基础阶段就停止了，RIOT、planner、learned-offline 和 Waymax 闭环都没有真正运行。**

结果压缩包中只有配置和 `checkpoints/natural/history_natural.json`，没有 `.pt` checkpoint、transport/planner history、learned-offline JSON、probe JSON 或 Waymax JSON。最佳自然基础验证 set minADE 为 41.2888 m，远高于 v12 的 12 m gate；source/priority 语义几乎未学习，neutral consistency 约 53.94 m 且基本不下降。

## 2. 论文真正需要成立的核心命题

论文的核心并非“再加一个社会成本”，而是：

> 一个 ego 候选即使自身无碰撞，只要它的安全依赖某个关键交通参与者采取高负担让行、急刹、放弃优先权或放弃合法间隙，该候选就是 false-safe；这种问题应作为可行性缺陷，而不是软代价。

COWP 的理论链条是：

1. 识别关键交互对象；
2. 为每个关键对象构造不受 ego 强迫时的自然选项集合；
3. 对每个 ego 候选构造 ego-conditioned response；
4. 判断每个自然 root 是否仍存在安全、低负担、同 root 的响应；
5. 用 option preservation / burden / witness 构成 non-coercive feasibility；
6. 先做硬可行性筛选，再按 ego utility 排序。

要让该命题成立，实验必须证明的不只是 candidate-level 分类好，而是以下因果链条都可识别：

- natural root 有稳定、可解释的身份；
- response 确实是对指定 ego candidate、指定 natural root 的响应；
- direct root transport 比 generic candidate classifier、pairmax 和 response-bank-only 更有效；
- 选出的计划降低真实或可信反事实下的高负担让行，而不是仅降低模型自己预测的 witness 概率；
- 安全改善不能以明显牺牲进度或大幅增加 fallback 为代价。

## 3. v12 为什么无法支撑核心 idea

### 3.1 自然选项基础完全未达到可识别状态

v12 最佳记录：

| 指标 | 最佳值 |
|---|---:|
| val natural set minADE | 41.2888 m |
| branch minADE | 42.0075 m |
| observational minADE | 43.6674 m |
| neutral minADE | 40.7546 m |
| priority minADE | 39.9407 m |
| neutral consistency | 53.9360 m |
| source CE | 1.0215 |
| priority BCE | 0.6673 |
| checkpoint score | 42.6445 |

8 个 epoch 中：

- source CE 只改善约 0.00185；
- priority BCE 只改善约 0.00154；
- neutral consistency 只改善约 0.0013 m。

这意味着模型虽然在降低一个很大的轨迹误差，但没有可靠恢复 natural root 的来源和优先权语义。RIOT 的监督对象正是“指定 natural root 是否被保留”，因此 root 本身不可辨识时，后续 transport 结果没有因果含义。

### 3.2 原 natural decoder 的结构不适合 80 步多模态轨迹

原实现从一个 agent latent 通过一个大线性层一次性输出 `24 × 80 × 7`，其 mode embedding 没有进入轨迹头，也没有显式 time embedding、运动学基线或逐时刻平滑结构。这会产生三个问题：

- 全时域坐标回归初始误差极大；
- 各 mode 的轨迹生成缺少独立的模式条件；
- 80 步误差通过一个投影耦合，短期可控性与长期分叉无法分离。

### 3.3 自然 root 的语义监督与轨迹 mode 没有正确绑定

natural alternatives 是无序集合。原 loss 主要约束 aggregate source distribution 和 aggregate priority expectation，并未首先把每个 GT root 匹配到最近的 predicted mode，再对该 matched mode 做 source/priority 监督。因此，即使 aggregate 统计正确，也不能说明某条 predicted trajectory 具有正确 root 语义。

neutral consistency 还遗漏了 mixture probability 权重，导致几乎不会被选择的低概率 mode 与高概率 mode 等权进入 consistency，解释了该项长期维持在约 54 m。

### 3.4 当前闭环机制指标存在循环论证风险

真实 Waymax 路径确实 step 了环境，并可计算 collision、offroad、route progress、wrong-way、kinematic infeasibility 等标准指标。但在线 `ClosedLoopPredFSR/CBS/OPR` 来自同一个 COWP 模型的预测头；模型用这些输出筛选候选，再用相同输出评价自己，不能作为独立反事实真值。

因此必须区分：

- **真实闭环物理指标**：collision/offroad/progress/kinematics；
- **模型健康代理量**：PredFSR、predicted CBS/OPR；
- **机制证据**：离线带 root/response 标签的 direct transport 指标，或由独立 reactive simulator / human audit 产生的反事实 burden 指标。

v13 已在输出 JSON 中显式标注 proxy-only 和 ground-truth unavailable。

### 3.5 当前 Waymax 是真实 SDC 闭环，但非 reactive 多智能体闭环

当前 environment 只控制 SDC，非 ego agent 沿 Waymax/log playback 演化。这种协议可用于比较 ego 物理安全和轨迹执行质量，但不足以验证“另一辆车被迫急刹/让行”这一行为命题，因为 logged agent 不会对 ego 偏离做真实响应。

这不是代码假闭环：ego 确实闭环重规划并执行；但它是 **closed-loop ego + non-reactive traffic**，不是 **reactive multi-agent closed loop**。论文必须准确命名。

## 4. 数据与缓存审计

现有 `tensor_cache_*_waymax` 审计显示：

- train 14,640 个场景，val 5,013 个场景；
- natural/response/witness/planner 核心键结构覆盖接近 100%；
- 每场约 50.6 个 valid candidate；
- 实际只 replay 约 12 个 candidate；
- Waymax outcome 对全部 valid candidate 的覆盖仅 train 23.44%、val 23.70%；
- replay-valid candidate 中 physical unsafe 约 train 52.19%、val 53.50%；
- finite log-divergence 覆盖为 0，不能训练或报告 log-divergence；
- train/val 分布总体接近，scenario_id overlap 为 0。

由此可得：

- 这些缓存足以继续 natural/response/witness/planner 的标签训练；
- sparse collision/offroad outcome 可以作为辅助损失，但不能作为唯一 planner 目标；
- learned-offline selected Waymax outcome 只能作为有覆盖分母的部分诊断；
- real online Waymax 不依赖把所有训练 candidate 重放一遍；
- 不要为缺失 log-divergence 重新跑全部训练集，应先对 val 和 checkpoint-selected candidate 做定向 replay。

后新增的 `tensor_cache_*_waymax_transport_v9` 实体文件未上传，无法在本地核验它与 raw cache 的 track ordering、坐标系和 base tensor 是否完全一致。v13 新增的 `33_diagnose_cache_alignment.py` 专门完成这一验证。

## 5. 既有算法中应保留、增强和停止使用的部分

### 5.1 应保留并增强

- burden-oriented problem formulation 和 non-coercive feasibility；
- conventional safety 作为硬 shield；
- candidate--natural relative geometry；
- pair witness localization，作为解释和辅助任务；
- direct mode conflict / retain primitive supervision；
- BCOT/RIOT 的 budget 思路，优于跨 agent 的 pairwise `any/max`；
- hard-first selection 与 fallback；
- pair threshold 与 candidate transport budget 分离；
- pairmax、generic certificate、response-bank-only 作为注册消融；
- Waymax 标准物理指标与逐步重规划路径；
- sparse Waymax collision/offroad outcome 作为辅助风险头；v13 主 selector 默认禁用该风险头，待 checkpoint-selected val replay 覆盖率明确后再做注册消融。

### 5.2 必须修改或降级为消融

- 原 single-linear natural decoder；
- aggregate-only source/priority supervision；
- response-bank-defined root recovery；
- generic candidate certificate 作为核心机制证据；
- pairwise max/any 作为主 selector；
- mixture-weighted existential burden；
- 用同一个 threshold 表示 pair confidence 和 candidate budget；
- sparse replay outcome 作为完整 offline benchmark；
- log-divergence loss/metric；
- model-predicted online FSR/CBS/OPR 作为 ground truth；
- 只放宽 threshold 来换 recall；
- 在 mechanism gate 未过时直接上 RL 或全量 Waymax。

### 5.3 v10/v11 的有效证据

v10 表明 candidate--natural relative geometry 是有效的：pair witness AUPRC 从约 0.43 提升到 0.7161，candidate false-safe AUPRC 达 0.9043。但 accepted NCF recall 仅 0.1267，fallback 0.2406，说明 pairwise aggregation 造成可行候选召回崩塌。

v11 的 ranking accuracy 达 0.8306，OPR/HBCR/selected false-safe 都比 conventional 好，但 EP 降至 0.3538、fallback 升至 0.2731，BCOT false-safe AUPRC 仅 0.4115。根因是 witness stage 缺 candidate budget supervision、double-count option preservation、错误的 existential burden、natural drift 和非 root-conditioned response bank。这些失败已经在 v12 设计中部分修复，但 v12 没有跑到验证这些修复的阶段。

## 6. v13 已完成的代码修改

修改包名：`COWP_v13_natural_closedloop`。

### 6.1 Natural basis

- 新增 `temporal_kinematic` decoder；
- constant-velocity baseline；
- explicit mode/time embeddings；
- bounded cumulative XY/yaw residual；
- zero-init residual head；
- 保留 `legacy_linear` 作为消融；
- matched-mode source CE 和 priority BCE；
- 修正 neutral consistency 权重；
- 新增 1/3/5/8 s minADE。

### 6.2 数据诊断

- `33_diagnose_cache_alignment.py`：
  - raw 与 transport base tensor 一致性；
  - critical track id 到 model input index 的映射和可见性；
  - current state、第一 future step、CV anchor 的坐标一致性；
  - natural displacement/speed/source 分布；
  - response root index 范围；
  - Waymax rollout/logdiv coverage。
- `34_diagnose_natural_oracles.py`：
  - 15 条 acceleration × yaw-rate kinematic bank；
  - source-stratified 1/3/5/8 s oracle minADE；
  - 为 natural gate 提供数据难度参照。

### 6.3 Planner/Waymax

- 新增上一周期计划 shift 后的 continuity risk；
- continuity 只在通过硬可行性后的候选间排序，不能让 rejected candidate 重新可行；
- 内部 baseline 同样使用 continuity，避免不公平；
- Waymax JSON 显式记录 logged replay、reactive 未实现、online mechanism GT 不可用；
- 配置虚假声明 reactive policy 时直接报错；
- 修复 label metric 缺字段时的未定义变量崩溃。

### 6.4 运行工程

- 新增 `STOP_AFTER_STAGE=natural|transport|planner|offline|probe`；
- natural-only 可单独运行；
- 每个阶段都保留硬 gate；
- 完整测试：73 passed；
- YAML、compileall、bash syntax 均通过。

## 7. “CCF-A 指标门槛”的正确理解与建议门槛

CCF-A 是会议/期刊推荐类别，不存在统一的 collision rate 或 planning score 录用线。第七版 CCF 目录仍按 A/B/C 分类，人工智能 A 类包含 AAAI、NeurIPS、CVPR、ICCV、ICML、IJCAI 等；不同 venue 和评审对 novelty、因果论证、实验完整性、可复现性要求不同。

Waymax 官方也没有一个统一 SOTA leaderboard 对应当前自定义 planning protocol。官方基准显示 collision/offroad 对 non-ego policy 和 action space 极其敏感。例如原 Waymax 论文中，同一类 DQN 在 playback/playback 与 playback/IDM 等协议下 collision 可从约 4.91% 到 8.67% 甚至 25.15%；因此跨协议直接比绝对数字没有意义。V-Max 也明确指出 log replay 在 ego 偏离日志后可能产生不真实交互。

因此建议使用两级目标：

### 7.1 论文核心机制 gate

以下是工程建议，不是官方录用线：

- pair witness AUPRC >= 0.70；
- direct conflict-conditioned root-transport AUPRC >= 0.75；
- direct root transport 相对 response-bank-only 和 pairmax 至少 +0.05 AUPRC；
- accepted NCF recall >= 0.50；
- accepted candidate rate >= 0.20；
- selected false-safe 相对 conventional 至少降低 25%；
- OPR 至少 +0.05；
- HBCR 相对降低至少 20%；
- fallback <= conventional + 0.03；
- EP paired non-inferiority margin 不低于 -0.02；
- 所有主要差异给出 scenario-paired bootstrap 95% CI，且核心改善 CI 不跨 0。

### 7.2 Waymax paper-ready gate

仍以同协议、同场景、同 candidate generator 的相对比较为主：

- 至少在预注册的 1,000 个场景上开发，最终跑完整 5,013 val 场景；
- collision 相对最强内部 baseline 降低 15--20%，或至少不差且机制指标显著更优；
- offroad 绝对恶化不超过 1 percentage point；
- route progress ratio/EP 下降不超过 2--3%；
- kinematic infeasibility、wrong-way、comfort 不得明显恶化；
- 报告 3 个种子或至少对 scenario 做 paired bootstrap；
- 100-scene probe 只用于查 bug。若观测 collision rate 为 5/100，其 Wilson 95% CI 约为 2.15%--11.18%，不足以宣称 SOTA。

近期公开工作在 Waymax 上通常强调相对碰撞下降而非单一绝对门槛，例如 CorrectionPlanner 报告超过 20% 的 collision reduction；这可以作为“安全提升需要达到可感知量级”的参考，但不能直接当作你的同协议目标。

## 8. 正确的下一步实验顺序

### Phase A：只做数据和 natural basis

先运行 raw/transport alignment 和 oracle。任何 track/coordinate hard failure 都应先修数据，禁止继续训练。

然后只训练 natural，使用 `STOP_AFTER_STAGE=natural`。不要一条命令直接跑到 Waymax。

通过标准：

- alignment hard checks 全过；
- 1 s / 3 s / 8 s minADE 同时改善；
- source CE、priority BCE、neutral consistency 有真实学习；
- gate 通过。

### Phase B：只做 transport/RIOT

冻结已通过的 natural basis，训练 witness/transport。首先看：

- candidate-budget supervision coverage；
- direct root-transport AUPRC；
- response-bank auxiliary AUPRC；
- natural-root assignment minADE；
- pair witness AUPRC。

若 direct root transport 不优于 auxiliary/pairmax，不要训练 planner。

### Phase C：planner + learned-offline

先跑 budget sweep，再按明确约束选 operating point。不得使用 `least_violation` 结果进入论文表。核心关注 NCF recall、fallback、selected false-safe、EP 非劣性，而不是仅看 generic candidate AUPRC。

### Phase D：100-scene Waymax smoke test

只用于确认：

- action mode/坐标正确；
- collision/offroad 指标能正常累计；
- candidate starvation、fallback、plan switching 不异常；
- COWP 不比 conventional 明显差。

### Phase E：全 val logged-replay + reactive protocol

完整 5,013 val 做真实 SDC closed loop。之后另建 reactive-agent protocol：Waymax IDM/sim-agent actor 或 V-Max multi-agent WOMD。二者必须分表报告，不能把 logged replay 写成 reactive。

对论文核心命题，至少需要一组 reactive test 显示：COWP 在物理安全不恶化时，降低 non-ego hard braking、induced deceleration、priority abandonment 或 option collapse。当前 v13 没有假装实现这一部分，因为本地无 Waymax 安装，无法可靠验证其 API；代码已经设置 protocol guard，避免误报。

## 9. 运行指令

完整指令见 `NEXT_RUN_COMMANDS_CN.txt`。推荐按以下四次独立运行：

1. cache alignment + oracle；
2. natural-only；
3. transport-only；
4. planner + learned-offline；
5. mechanism gate 通过后再 probe/full Waymax。

不要覆盖 v12 输出目录，以免将新 checkpoint 与旧 history 混在一起。

## 10. 论文需要同步修改的地方

在新实验完成前，论文中以下内容不能作为已证实结论：

- abstract 中“Experiments ... show ...”的确定性表述；
- main table 和 stress/ablation 的空值；
- 将当前 online PredFSR/CBS/OPR 描述为 closed-loop ground truth；
- 将 log replay 描述为 reactive interaction；
- 将旧 pairwise witness 算法写成当前 RIOT 实现。

最终论文方法应以最佳代码为准，建议把当前主方法重写为 **Root-Indexed Option Transport + hard-first selection**，把 pair witness 和 generic candidate certificate 放到 auxiliary/ablation。自然 alternatives 的实现也要从旧的泛化描述更新为可复现的多源 label、matched semantic supervision 和 temporal-kinematic decoder。

## 11. 当前可诚实给出的状态判断

- **论文问题定义：有潜力且区别于一般 courtesy cost。**
- **当前 v12 证据：不足，核心算法未运行。**
- **数据结构：基本可用，但 transport_v9 对齐必须在服务器实测。**
- **v13 修复：针对当前最直接的根因，静态测试通过。**
- **SOTA 状态：未知，不能从现有结果推断。**
- **最重要的下一步：先证明 natural root 和 direct root transport 可识别，再谈 RL 和全量闭环。**
