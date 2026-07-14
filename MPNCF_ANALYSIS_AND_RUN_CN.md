# Mechanism-aware P-NCF 下一轮补丁说明

## 结论

当前结果证明 Dynamics-aware Quantile P-NCF 没有真正改变决策边界：learned-offline 中 `cowp == conventional_safety == soft_burden_cost_only`，online Waymax 中 `cowp_merged.json` 与 `conventional_safety_merged.json` 的 CR、Collision、Offroad、EP、fallback、候选数全部一致。

这不是论文核心 idea 必然失败，而是当前在线候选生成和在线 conventional 过滤让 COWP 没有足够候选可选：offline cache 每个场景约 50.6 个 valid candidates、12 个 Waymax replay candidates；online 每步只有约 7.6 个 valid candidates、约 1.8 个 conventional candidates。quantile frontier 在 1--2 个 conventional candidate 上基本退化为 conventional safety。

## A/B 是否解决上一轮问题

- A：修复开头离散 jerk 检查是必要的，但从当前结果看不充分。online candidate starvation 仍然存在，真实瓶颈还包括在线 primitive 太窄、endpoint-only dedup、低速/回退状态候选坍缩，以及 constant-velocity conventional check 过度拒绝。
- B：quantile frontier 当前没有起效。它依赖候选内相对排序，但当前 conventional set 太小，且 candidate certificate 诊断没有输出，无法证明 certificate 学到了有效排序；结果上 COWP 完全等于 conventional safety。

## 本补丁做了什么

### 1. Logged-future-aware conventional check

在 Waymax online 中优先使用 simulator/log trajectory 的非 ego future 做碰撞过滤，缺失时才回退到 constant velocity，减少把“会让行/会转弯/会离开冲突区”的交通参与者误判成静态 CV 障碍。

### 2. Terminal position-speed candidate frontier

新增 cubic-longitudinal / quintic-lateral terminal primitive，默认把在线候选填充到 32 个左右，并用“endpoint + terminal speed + macro”共同去重，避免多个语义不同但 endpoint 接近的候选被折叠。

### 3. Mechanism-aware P-NCF pressure prior

新增 candidate pressure prior：对 accelerate-through、merge-ahead、lane-change cut-in、近距离 TTC/closest approach 等高 coercion 机制加弱惩罚；对 yield/stop/neutral/merge-behind 等让行机制降权。它只作为 prior 与 candidate certificate、pair witness 混合，不替代学习模块。

论文上可表述为：

> Mechanism-aware quantile P-NCF frontier: a set-valued non-coercive feasibility frontier that combines learned candidate certificates with an interaction-mechanism pressure prior for online candidate ranking.

### 4. learned-offline 与 online 对齐

`rollout.py` 也加入了 pressure prior，使 learned-offline 的 COWP 选择逻辑和 online policy 的机制 prior 一致。

### 5. 新诊断指标

输出 online pressure 指标：

- `ClosedLoopMean/selected_candidate_pressure_prior`
- `ClosedLoopMean/mean_candidate_pressure_prior`

learned-offline 也输出：

- `CandidateCertificate/SelectedPressurePriorMean`
- `CandidateCertificate/AcceptedPressurePriorMean`

## 下一轮首先看什么

1. `ClosedLoopMean/valid_candidates`：应从约 7--8 明显升高，理想目标 20--32。
2. `ClosedLoopMean/conventional_candidates`：应从约 1.8 升高，理想目标至少 4--8。
3. `cowp` 是否不再与 `conventional_safety` 完全一致。
4. `SelectedFalseSafeRate`、`FSR` 是否低于 planner_score_only/conventional_safety，同时 EP 不明显塌陷。
5. `CandidateCertificate/*_AUPRC` 和 `RiskRankingPairAccuracy` 是否出现并有区分度。
6. `KinematicsInfeasibilityRate` 仍需重点看；如果仍在 0.10+，下一轮要继续修 action adapter。
