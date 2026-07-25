# COWP v6 结果复核与 v7 优化报告

## 1. 结论

当前 v6 已经纠正了 v5 最严重的证书监督错误，并第一次使 candidate-level non-coercive certificate 在离线数据上形成有效排序。v6 的证书并未再次塌缩：FalseSafe AUPRC 为 0.8125，NCF AUPRC 为 0.5583，候选风险 pair ranking accuracy 为 0.7841。COWP 相比 conventional safety 的离线 SelectedFalseSafeRate、CBS、OPR、HBCR 均有小幅改善，100 场景 Waymax 中碰撞率从 0.41 降至 0.34，EP 从 0.5080 升至 0.6515。

但 v6 还不能支撑“闭环 SOTA”或完整证明论文的核心机制。最主要的原因是：

1. pairwise witness 虽然没有饱和到 1，但固定 0.5 阈值下几乎完全失效，真实 recall 仅 0.000199；
2. v6 只阻断了 certificate loss 到 witness 输出的直接梯度，没有阻断 candidate/planner loss 经共享候选和图表示间接改写 witness；
3. 离线和在线分别维护了两套 frontier/selector 实现，权重和 priority anchor 存在偏差；
4. action risk 与真正发送给 Waymax 的 action 不一致，导致选中候选的预测 action risk 很低，但 KinematicsInfeasibilityRate 从 0.13 恶化到 0.23；
5. 在线路径读取 `sim_trajectory/log_trajectory` 的未来状态，形成 privileged-future/oracle leakage；
6. 训练在 epoch 17 后明显崩坏并重复输出相同验证指标，best checkpoint 避免了最终结果退化，但仍浪费大量训练时间并暴露优化不稳定性。

v7 因此不是简单继续增加 selector 权重，而是形成“Gradient-Isolated Causal Set-Preservation Planning”版本：语义证书、物理风险、执行动作和 utility 在结构上分离，并通过共享的 epsilon-Pareto frontier 统一离线和在线选择。

---

## 2. 四个既有根因是否已经被 v6 修正

| 根因 | v6 状态 | 证据与判断 | v7 处理 |
|---|---|---|---|
| 非胁迫证书监督标签相互矛盾 | **已修正** | certificate loss 只使用互斥的 `noncoercive_feasible` 与 `false_safe`，碰撞/越界监督进入 outcome head。FalseSafe AUPRC 0.8125、ranking 0.7841，说明修正有效 | 保留，并继续禁止 physical-safe 被解释为 NCF-positive |
| candidate/planner loss 反向污染 witness | **部分修正** | v6 对 witness probability、OPR、burden 等做 detach，因此 witness 不再全 1；但 certificate/planner/outcome head 仍通过共享 `z_cand/z_agent/z_graph` 间接更新 witness 表示 | v7 增加强 gradient firewall：planner/certificate/outcome/priority 均消费 detached backbone features；推荐从 v6 best 做 head-only fine-tune，并在 planner 阶段冻结图、候选和 witness stack |
| 离线与在线不是同一算法 | **部分修正** | v6 已关闭 generic hybrid fallback，二者都使用 certificate frontier；但 selector 分别写在 `rollout.py` 和 `policy_wrapper.py`，tie 权重不同，offline 使用 cached rho、online 使用 live heuristic | v7 新增共享 `set_preservation_selector.py`，离线和在线调用同一 frontier、Pareto、risk-budget 和 tie-break 实现。priority evidence 的输入形式仍因数据环境不同，但决策函数统一 |
| replay 与在线 action mode 不一致 | **已修正** | v6 主脚本显式使用 `absolute_xy_yaw` | v7 同时把 CLI、policy factory 和 dataclass 默认值改为 `absolute_xy_yaw`，避免绕过主脚本时再次静默退回 delta mode |

因此不能简单说“v6 已经把四个问题全部解决”。准确说法是：第 1、4 项已修正；第 2、3 项只修正了一半。v7 对第 2、3 项做了结构性补全。

---

## 3. v6 中真正起效的算法优化

### 3.1 互斥语义监督与 structured-residual certificate

这是 v6 最明确的成功。最终离线结果为：

- CandidateCertificate/FalseSafe_AUPRC = 0.8125
- CandidateCertificate/NCF_AUPRC = 0.5583
- CandidateCertificate/RiskRankingPairAccuracy = 0.7841

这说明“analytic set-preservation risk + learned residual calibration”的方向有效。它比普通 safety classifier 更适合论文，因为 certificate 明确描述其他交通参与者可保留的响应集合、OPR、burden excess 与 coercion intensity，而不是只预测 ego collision。

建议继续保留这个模块，并把论文主线从“又一个 candidate safety scorer”明确改写为：

> 通过结构化 response-set evidence 构造 candidate-level set-preservation certificate，再用闭环 outcome 作为独立物理盾牌，而不是用 outcome 替代 non-coercion 定义。

### 3.2 exact-cardinality frontier

在线 COWP 平均 accepted candidates 为 1.7225，而 conventional safety 为约 7.96。它证明 COWP 没有退化为 ordinary safety filter，候选集合确实被重新构造。

该优化有效，但 exact top-k 本身 novelty 不够强。v7 将其保留为 ablation，并新增 epsilon-Pareto frontier 作为主设置。Pareto frontier 分别处理：

1. semantic non-coercive risk；
2. physical/execution risk；
3. utility regret。

只有在一个候选在三轴上均不差并至少一轴明显更好时，才支配另一个候选。这比加权和更符合论文 setting，也能避免通过增大 action/outcome 权重把 COWP 再次变成 generic safety optimizer。

### 3.3 utility/progress guard

`universal_ncf` 的离线 EP 只有约 0.0053，fallback 约 0.672；v6 COWP EP 为 0.394，在线 EP 为 0.6515。这证明 utility-regret/progress guard 是必要的。

该设计应继续增强，但不能回到 raw planner score 主导。推荐采用 v7 的顺序：

1. 先构造 semantic/physical/utility Pareto feasible set；
2. 在 semantic risk 最优值附近设置 risk budget；
3. planner score、progress 和 physical risk 只在该预算内做 bounded tie-break。

### 3.4 outcome head 与物理盾牌

100 场景中碰撞率从 0.41 降到 0.34，说明 outcome/physical branch 对闭环安全有帮助。它应继续保留，但必须作为独立盾牌，不应参与 non-coercive label 或替代证书。

当前样本下 collision 改善比 CR 改善更明确，但 100 场景仍不足以支撑 SOTA。需要 1000/5000 场景、至少 3 个种子和置信区间。

### 3.5 best-checkpoint composite selection

v6 最优 checkpoint 出现在 epoch 9。epoch 17 后 val loss 从约 5.73 上升到 8.42，planner imitation 从约 0.24 上升到 3.88，此后 epoch 17–23 指标完全重复。若继续使用 final checkpoint，结果会再次被破坏。

best checkpoint 机制实际挽救了 v6，但还不够。v7 增加 plateau scheduler、early stopping、稀疏 epoch 保存，并默认进行 12 epoch 的低学习率 head-only fine-tune。

---

## 4. 没有起效或仍有明显副作用的设计

### 4.1 pairwise witness 没有承担论文核心角色

v6 witness probability 分布为：

- p10 = 0.2465
- p50 = 0.2896
- p90 = 0.3243
- p99 = 0.3625

固定阈值 0.5 下，WitnessRecall 约 0.000199，WLA 为 0。也就是说，v6 的结果主要由 candidate-level certificate 支撑，而不是 pairwise witness 的硬判定。

这不意味着 pairwise witness 完全无信息。其 AUPRC 为 0.3609，连续排序信号仍可能有价值。根本解决方式不是把 threshold 武断改成 0.3，而是：

1. 在独立 validation/calibration split 上做 threshold sweep；
2. 约束 NCF recall 和 fallback 后选阈值；
3. 主选择使用连续 priority-weighted witness evidence；
4. hard threshold 只用于强证据 veto 或解释性统计；
5. 报告 calibration curve、ECE、risk-coverage，而不仅是固定 0.5 precision/recall。

v7 脚本已把 threshold sweep 与校准接入 shared offline pass，在线自动读取校准结果。

### 4.2 action shield 与实际 Waymax action 不一致

v6 的 selected action risk 平均仅约 0.055，但 KinematicsInfeasibilityRate 从 0.13 上升到 0.23。这不是简单“action 权重不够大”，而是 action risk 评估 raw candidate trajectory，而最终发送动作又经过 jerk/yaw-limited controller 投影，两者不是同一个对象。

v7 的修正：

- 用 NumPy 向量化同时计算全部候选的 horizon risk；
- 对每个候选预计算真正会发送的 consistent one-step target；
- 加入 projection risk，即原始 waypoint 被 controller 修正的程度；
- selector 使用 execution-aligned action risk；
- 选中候选直接复用预计算 target/acceleration，不再重新计算。

这比继续增大 `candidate_selection_action_weight` 更根本。

### 4.3 COWP 在线语义诊断仍不理想

v6 在线 PredCBS_episode 约 1.126、PredOPR_min_episode 约 0.094、PredHBCR_episode 为 1.0；而离线核心指标只有小幅改善。这表明在线生成的候选、critical-agent 分布和训练 cache 仍存在 domain gap。

下一步需要按场景类型分析：intersection、merge、lane change、following、VRU 等。最关键的不是继续调一个全局阈值，而是确认哪些 macro candidate 和 critical-agent 类型导致：

- certificate 离线排序正确，但在线候选分布外失效；
- OPR 被系统性低估；
- 高进展候选触发 route/kinematics 问题。

建议在 1000 场景运行中输出 per-scenario candidate table，并按 macro type、interaction type、速度区间做 failure taxonomy。

### 4.4 privileged future leakage

v6 每个在线 step 从 `sim_trajectory` 或 `log_trajectory` 提取未来 80 步其他车辆轨迹，用于 conventional checks、rule risk 和 pressure prior。这种实现可能读取当前决策时不可获得的 logged future。即使 Waymax state 暴露这些字段，它也不应作为主方法输入，否则论文中的“闭环规划”会混入 oracle future 信息。

v7 默认：

```yaml
planning:
  online_other_future_source: constant_velocity
```

主实验不读取 future trajectory。`logged_oracle` 只作为上界消融。该改动可能使首次 v7 结果低于 v6，但这是必须接受的，因为它恢复了因果、可投稿的实验设置。

---

## 5. v7 新算法设计与 novelty

建议将 v7 方法概括为：

## Gradient-Isolated Causal Set-Preservation Planning

### 5.1 Semantic certificate 与 physical outcome 双分支

- Semantic branch：预测 response-set preservation / false-safe / NCF；
- Physical branch：预测 collision/offroad 等 rollout outcome；
- 两者标签、梯度和 selector 角色分离。

### 5.2 Strong gradient firewall

v7 中 certificate、planner、priority、outcome heads 使用 detached `z_candidate/z_agent/z_graph`。planner 阶段默认冻结 graph、candidate encoder、natural decoder 和 witness decoder，防止 candidate-level 大损失重新改写 pairwise witness。

这使“证书不塌缩”从依赖损失权重的经验现象，变为模型结构保证。

### 5.3 Causal response evidence

主闭环不读取 logged future。其他车辆未来只来自当前状态的 causal fallback 或 learned response branch。oracle future 单独报告，不能与主结果混合。

### 5.4 Epsilon-Pareto set-preservation frontier

v7 不再把 semantic、physical、utility 全部压成一个大权重和。候选在三个维度上构成 epsilon-Pareto set，然后再执行 bounded risk-budget selection。

这一设计能够成为论文新的算法贡献点：

- non-coercion 是一级决策语义；
- physical safety 是 feasibility shield；
- ego utility 是受约束的 performance objective；
- 三者不因量纲和权重调节而相互替代。

exact top-k 仍保留为 ablation，以验证 Pareto 结构是否确实改善 kinematics/offroad，同时维持 certificate 指标。

### 5.5 Execution-aligned action shield

选择阶段评估的 action 与真正执行的 action 使用同一个 jerk/yaw-limited controller。该设计直接服务于闭环 kinematics 指标，不再依赖 raw trajectory proxy。

---

## 6. 闭环加速优化

### 6.1 已实施且不删除任何指标的优化

1. **状态只提取一次**：v6 history 和 logged future 各自复制完整 trajectory tree；v7 统一成一个 online state bundle。
2. **历史帧向量化 gather**：不再逐帧调用 `_state11_at`。
3. **主设置不提取 logged future**：去除第二次完整 device-to-host trajectory copy 和 80 步 Python loop。
4. **action risk 向量化**：K×H Python loop 改为 NumPy broadcasting。
5. **复用 action target**：selector 已经计算的 consistent target 直接用于 Waymax action。
6. **共享 selector**：离线/在线不再维护两套实现，减少重复逻辑和分支漂移。
7. **训练 early stop + plateau LR**：避免 v6 epoch 17–23 的无效计算。
8. **双 GPU sharding**：full COWP 两卡各跑一个 shard，合并时保留全部标准指标。
9. **继续启用 env reuse、prefilter、JIT env 和 JIT metrics**：不删除 Waymax standard metrics。

### 6.2 微基准

在 K=64、action-risk horizon=8 的合成输入上，向量化 action risk 与 v6 loop 版本的最大绝对差约 `2.2e-6`；中位耗时从约 `2.74 ms` 降为 `0.218 ms`，约 12.6 倍加速。

这只是 policy helper 微基准，不是 Waymax 整体速度。v6 中 COWP 约比 conventional 慢 11%，而整体仍主要受 Waymax/JAX step 和标准指标计算控制。因此不能在没有真实 GPU 重跑前声称 end-to-end 加速比例。预计优化主要降低 policy overhead、CPU/JAX 同步和长时间运行中的重复开销，而不会改变指标定义。

### 6.3 不建议做的“加速”

- 不应关闭 Collision/Offroad/WrongWay/Kinematics/LogDivergence 等标准指标；
- 不应缩短主论文 rollout horizon；
- 不应减少候选到无法覆盖关键 macro；
- 不应以 logged future cache 代替因果在线预测；
- 不应在不同方法间使用不同场景集合。

---

## 7. 下一轮实验判定标准

### 第一阶段：100 场景 probe

v7 必须至少满足：

- certificate FalseSafe AUPRC 不低于 v6 的 0.81 太多；
- ranking accuracy 保持在约 0.75 以上；
- witness threshold 不再固定 0.5，且 hard/priority gate 有非零有效 recall；
- CollisionRate 不高于 conventional；
- KinematicsInfeasibilityRate 必须显著低于 v6 的 0.23，优先目标接近 conventional 的 0.13；
- EP 不发生 universal-NCF 式塌陷；
- causal COWP 与 logged-oracle ablation 必须明确分开。

### 第二阶段：1000 场景

需要同时比较：

1. conventional safety；
2. planner score only；
3. COWP exact-topk；
4. COWP epsilon-Pareto；
5. no-gradient-firewall；
6. no-structured-certificate；
7. logged-oracle future，仅作为上界。

报告 bootstrap CI 和 paired scenario wins/losses，而不是只报告均值。

### 第三阶段：5000 场景与 3 seeds

只有在 1000 场景通过后再扩展。最终 SOTA 结论至少要求：

- 标准指标总体不劣化，尤其 collision、offroad、wrong-way、kinematics；
- non-coercive 指标具有稳定显著改善；
- Pareto/frontier、certificate、witness、gradient firewall 都有清晰消融；
- 主实验严格 causal；
- 公开/强基线使用同一场景、horizon 和 metric implementation。

---

## 8. 验证状态

本次代码在当前容器中完成：

- Python compile：通过；
- shell syntax：通过；
- pytest：50/50 通过；
- vectorized action-risk 数值一致性微基准：通过；
- shared offline/online selector 一致性测试：通过；
- planner/certificate/outcome 对 graph/candidate/witness backbone 的梯度隔离测试：通过。

本环境没有你的 WOMD cache、v6 checkpoint 和两张实际 GPU，因此没有伪造 v7 训练或闭环结果。v7 的 Pareto frontier、因果 future 和 execution-aligned shield 必须由下一轮真实实验确认。
