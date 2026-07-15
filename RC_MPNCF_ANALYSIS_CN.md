# RC-MPNCF 本轮结果分析与下一轮补丁说明

## 1. 本轮结果的关键信号

上一轮机制感知 P-NCF 补丁已经解决了最初的候选数量不足问题，但没有解决闭环安全问题：

- `cowp` online valid candidates 从上一轮的约 7--8 提升到 `28.64`，达到预期的 20--32 区间。
- `cowp` online conventional candidates 从约 1.8 提升到 `13.71`，超过预期的 4--8。
- 但是 `cowp` 的闭环结果明显变差：`CR=0.477`、`CollisionRate=0.432`、`OffroadRate=0.080`、`KinematicsInfeasibilityRate=0.191`。
- `conventional_safety` 反而更安全：`CR=0.370`、`CollisionRate=0.329`、`OffroadRate=0.063`、`KinematicsInfeasibilityRate=0.124`。
- learned-offline 中 candidate certificate 虽然已有指标输出，但 `SelectedRiskMean=1.875` 对所有方法完全相同，说明候选证书在选择层仍然接近常数。

结论：上一轮不再是 candidate starvation，而是 **candidate set 被放大后，COWP 的选择准则缺少可靠的反事实安全校准**。

## 2. 当前导致论文算法 idea 不能成立的主要问题

### 问题 A：logged future conventional check 太乐观

上一轮用 logged/simulator future 替代 constant velocity，确实释放了更多候选；但这把 “日志中其他车让行/减速后的轨迹” 当成了 ego 当前动作下仍然成立的反事实轨迹。对于 cut-in、merge-ahead、加速穿越等动作，这会把 coercive plan 错判为 conventional-safe。

这解释了为什么 `cowp` conventional candidates 增加到 `13.71` 后，collision 反而从 conventional_safety 的 `0.329` 升到 `0.432`。

### 问题 B：candidate certificate 没有形成有效排序

`CandidateCertificate/NCF_AUPRC=0.1843`、`FalseSafe_AUPRC=0.3716`、`RiskRankingPairAccuracy=0.0`，且 selected risk 恒为 `1.875`。这基本等价于：candidate head 目前不能作为闭环选择的主证书。

原因可能包括：候选标签本身噪声较强、false-safe/NCF 标签分布冲突、planner stage 训练时 certificate head 与 planner score 共同被多个损失牵制、以及在线候选分布与 offline root candidates 分布偏移较大。

### 问题 C：pressure prior 能选出低 coercion 候选，但不能保证物理安全

`cowp` selected pressure prior 约 `0.439`，mean pressure prior 约 `0.752`，说明 pressure prior 的确在选低压力候选。但低 pressure 不等于低 collision/offroad/kinematic risk。因此它不能单独支撑 SOTA 闭环安全。

### 问题 D：kinematic infeasibility 是新硬瓶颈

`cowp` 的 `KinematicsInfeasibilityRate=0.191`，比 baseline 更差。这说明 terminal primitive 放大候选后，部分候选虽然通过了候选动态检查，但第一步 action adapter 与 Waymax StateDynamics 的约束仍然不一致。

### 问题 E：闭环评估慢的瓶颈

从日志看，full online rollout 每个 shard 跑 500 个场景需要数小时。主要瓶颈不是训练，而是 Waymax online：

1. 每个 scenario 80 steps，每步都做 PyTorch forward + JAX env step + standard metrics update。
2. 上一轮脚本每个 scenario 后都 `--clear-accelerator-cache`，会触发昂贵的 JAX/PyTorch cache 清理，严重拖慢。
3. 每步每个候选都要做 roadgraph drivable check，原实现会把候选采样点和大量 roadgraph points 做全量距离计算。
4. 三个 online methods 顺序跑，等于 full online 成本乘以 3。

## 3. 本补丁的算法改动：RC-MPNCF

本轮补丁把方法从单纯 Mechanism-aware P-NCF 改成：

> Risk-Calibrated Mechanism-aware P-NCF Frontier, RC-MPNCF

核心思想是保留论文的 P-NCF / non-coercive frontier 主线，但在 candidate certificate 尚未可靠时，引入一个低容量、可解释的反事实风险校准层，防止 logged-future false-safe plan 进入闭环。

### 3.1 Dual-envelope conventional safety

online conventional check 不再只看 logged future，而是：

- 所有候选都必须对 logged/simulator future 无碰撞；
- 对前方/近距离/低 TTC 的 priority-like agent，还必须对短时 constant-velocity envelope 安全；
- 这样避免把“日志中别人让了你”误当成“你的计划非胁迫安全”。

### 3.2 Rule risk prior

新增 `cowp/candidates/rule_risk`，刻画候选相对于 logged future 与 CV future 的 clearance、priority-like agent、logged-vs-CV gap。这个 risk 只作为 certificate flat 时的反事实安全校准，不替代 P-NCF。

### 3.3 Action risk prior

新增 one-step action risk，用候选第一步需要的 acceleration jerk、yaw-rate、acceleration excess 来估计 Waymax StateDynamics 不可行风险，降低 `KinematicsInfeasibilityRate`。

### 3.4 Scene-normalized risk fusion

不再把 raw candidate certificate risk 直接加入分数。因为当前 raw risk 近似恒定 `1.875`，直接使用没有信息量。现在每个 risk 在 scene 内做归一化，若 std 太小则自动置零。

融合后的 frontier risk：

```text
R_frontier = normalized(CertRisk)
           + lambda_pair * PairRisk
           + lambda_pressure * normalized(PressurePrior)
           + lambda_rule * normalized(RuleRisk)
           + lambda_action * normalized(ActionRisk)
```

这使算法仍然是 P-NCF frontier，而不是退回纯规则 planner；规则项只是校准层。

## 4. 运行时间优化

本补丁做了三类加速：

1. roadgraph drivable check 改为 candidate-local bbox 过滤，避免每个候选全量 roadgraph 距离计算。
2. conventional/rule risk 的碰撞检查默认 stride=4、horizon=50，而不是 80 帧全量逐帧检查。
3. 新脚本默认关闭 `--clear-accelerator-cache`，只在 OOM 诊断时再打开。

调参阶段建议只跑 `ONLINE_METHODS=cowp`，确认信号有效后再跑完整三方法对比。

## 5. 下一轮优先判断

先跑 fast probe，看：

- `ClosedLoopMean/valid_candidates` 是否仍在 20--32 附近；
- `ClosedLoopMean/conventional_candidates` 是否回落到 4--10，而不是 13+；
- `ClosedLoopMean/selected_candidate_rule_risk` 是否低于 mean rule risk；
- `ClosedLoopMean/selected_candidate_action_risk` 是否低；
- `CollisionRate` 是否接近或低于 conventional_safety；
- `KinematicsInfeasibilityRate` 是否从 0.191 回落。

若 fast probe 有效，再 full retrain/full eval。若仍无效，下一步需要把 candidate certificate 从离线标签学习，升级为 offline replay/Waymax outcome contrastive certificate，否则 CCF-A 级别的闭环 SOTA 很难成立。
