# COWP v16.8.12 数据支撑审计与构建修复

## 1. 结论

本次上传的 `formal_v16_8_11_train_pilot.zip` 不是“没有构建完成”，而是 **1200 场 train-pilot 已完整构建，但 natural-basis model-support 没过；同时 shell 的 `set -e` 在写最终 verdict 之前提前退出**。

因此：

- v16.8.11 的 `Missing train-pilot verdict` 是执行控制流 bug；
- 真正的科学门槛失败仍然存在，不能进入 full-core；
- 当前失败已经集中到 natural roots 的构建/过滤，不是 candidate bank、response bank 或 witness 标签退化；
- v16.8.12 不降低任何 root、low-burden、NCF、false-safe、PBTR、response-bank 或 causal-integrity 门槛，而是修复 natural basis 的几何与规范参考逻辑。

## 2. v16.8.11 train-pilot 的实际结果

Pilot manifest：400 hard + 800 random，0 overlap，union=1200，PASS。

`model_support_audit.json` 唯一失败的四类 hard checks：

- `every_critical_has_natural_root = false`
- `every_critical_has_multi_root_support = false`
- `every_critical_has_low_burden_natural_root = false`
- `every_critical_has_multi_low_burden_root_support = false`

精确计数（6611 个 critical vehicles）：

- zero natural roots：248，3.7513%
- fewer than two natural roots：101
- zero low-burden roots：250
- fewer than two low-burden roots：354，5.3547%

其余 active support 全部 PASS，包括：

- full 32-slot response coverage on every relevant pair
- candidate NCF / false-safe 非退化
- pair relevance / witness / pair-NCF 非退化
- response safe/unsafe、low/high-burden 非退化
- mode conflict / affected / retained 非退化
- affected-root recovery 非退化
- protected-candidate feasibility 非退化
- OBS/NEU/PRIO 与 PRED/OPT/EMG source support
- witness OPR/tail/min-safe-burden 和 response/root continuous targets 均有变化量

说明训练对象总体是可学习的，当前 blocker 是 natural basis coverage，而不是“模型没有正负例”。

## 3. 为什么 natural basis 仍然失败

### 3.1 Map filter 的离散点代理会误杀真实 lane-following 轨迹

v16.8.11 `_trajectory_map_compliance()` 把所有 lane polyline 当成离散 sample point cloud，然后计算轨迹点到这些 sample points 的距离。

这并不等价于“点到 lane polyline 的距离”。如果某个 lane segment 较长，一个轨迹点即使严格位于 segment 上，也可能离 segment 两端/采样点超过 5 m，于是被错误判为 off-map。

对 248 个 zero-root critical agents 重新分析：

- 148/248 的 dominant rejection 是 `map`
- 很多 full-80-future actor 的 OBS/NEU/fallback 共 31 次尝试全部被 map filter 拒绝

这与“WOMD 没 future”不一致，而与 map proxy 假阴性一致。

**v16.8.12 修复：**使用连续 point-to-segment distance；`map_max_distance_vehicle_m=5`、80% compliant fraction、12 m hard max 等物理阈值完全不变。

### 3.2 Raw logged future 被当成 NEU/PRIO 的规范 natural reference

v16.8.11 中，只要 future 足够完整，`natural_ref` 就直接采用 logged future timing。

但 COWP 的论文语义是：OBS、ego-neutral、priority-preserving 是不同来源。Logged future 可能已经包含 ego-induced yield，也可能包含其他交互导致的速度变化。它可以作为 OBS empirical evidence，却不能无条件成为所有 NEU/PRIO roots 的 normative progress/burden baseline。

248 个 zero-root 中：

- 230 使用 `logged` reference
- 18 使用 `logged_geometry`

其中 dominant rejection：

- map：148
- priority：100

`<2 low-burden roots` 的 354 个失败中：

- priority-dominated：201
- map-dominated：153

这说明 map 和 priority-reference 是两条主要失败链。

**v16.8.12 修复：**

- raw logged timing 继续作为 OBS branch；
- NEU/PRIO 的 natural reference 改为 timing-neutral reference：lane-topology route + current-state timing 优先；无可用 route 时只借 logged geometry 重新 timing；最后才 jerk-bounded straight；
- priority hard filter 的 progress geometry 与 burden functional 对齐到 natural-reference projected progress；
- 不提高 `priority_progress_loss_tolerance_m`，不放宽 jerk/acc/gap threshold。

### 3.3 Pair-specific neutral 已经不是主要 rootless 原因

v16.8.11 的 pair-specific neutral 修复有效。

248 个 zero-root agents 中，仅 5 个 pair-neutral `neutral_actor_unsafe=true`；243/248 的 pair-neutral 本身不 unsafe。

所以继续扩大 neutral candidate family 不是当前主方向。

### 3.4 Failure 不是由 future 缺失主导

按 future valid steps：

- 1–19：6/95 rootless
- 20–39：12/166
- 40–59：8/402
- 60–79：30/820
- **80：192/5128**

绝大多数 rootless 样本有完整 80-step future。继续只修 padding/short-future 不能解决当前 blocker。

### 3.5 Validity route matching 仍有一个隐藏错误

旧逻辑用 `valid_steps = sum(valid_mask)`，然后比较 `logged[:valid_steps]`。这隐含假设有效 future 一定从 t+1 开始连续出现。

v16.8.12 改成只在 raw WOMD `valid=1` 的真实 timestamp 上比较 route 与 future geometry；内部 gap / late appearance 不再由 hold padding 参与 route selection。

## 4. v16.8.12 代码修改

### Natural geometry

- `_lane_point_cloud` → `_lane_segment_cloud`
- map compliance 使用 exact continuous point-to-segment distance
- lane segment geometry 每场景只构建一次

### Natural reference

- full logged future 不再自动成为 normative timing reference
- map-route neutral timing first
- logged geometry + new timing second
- jerk-bounded straight last
- OBS branch仍保留 raw logged empirical evidence

### Route generation

- lane graph search 每个 critical actor 只按最大 required length 做一次
- route polylines 复用，再对不同 accel/speed-offset 重新 timing
- primary NEU/PRIO 直接 route-conditioned，不再先花大量预算在 straight roots 上

### Priority audit

新增明确 rejection reason：

- `max_decel`
- `max_accel`
- `max_jerk`
- `progress_loss`
- `gap_loss`

### Natural support diagnostics

新增：

- rootless / `<2 low-burden` by priority relation
- by reference kind
- dominant rejection family
- priority rejection mechanism counts
- rejected map max-distance summaries
- best rejected burden summaries

### Promotion wrapper

v16.8.11 的问题：`model_support --strict` 返回 2 + `set -e` → shell 退出 → verdict 没写。

v16.8.12：

- supervision/model-support/screen 属于 semantic gates，允许非零返回后继续写 verdict；
- build/manifest 等真正 runtime failures 仍 fail-fast；
- smoke/strict/train-pilot 都写 composite verdict；
- full-core 区分“verdict missing（流水线真的没完成）”和“verdict=false（科学门没过）”；
- smoke → strict → train-pilot → full-core 逐级检查相同 code fingerprint。

## 5. 什么条件才允许 full rebuild

### Validation smoke（48 hard + 48 random）

必须同时：

- proposal/causal screen pass
- training supervision pass
- model support pass
- rootless critical = 0
- critical with <2 low-burden roots = 0

### Validation strict（400 hard + 800 random）

必须同时保留既有 strict proposal gates 与上面的 natural/model-support gates。

### Train pilot（400 hard + 800 random）

必须：

- rootless=0
- `<2 low-burden=0`
- supervision/model-support/causal integrity PASS
- AnyValid >=0.99
- train AnyNCF support >=0.30

### Full core

只有 validation strict 和 train-pilot 都 `recommend_full_rebuild=true`，而且两者 fingerprint 与当前代码完全一致，才允许 22k/5k labels + tensor cache。

### Waymax outcome attachment

当前 planner 的 closed-loop loss 仍需要真实 attached Waymax candidate outcomes。Core cache PASS 并不等价于 mainline training dataset 完成；必须继续 `outcomes` 并运行现有 outcome support verification。

## 6. “完全支撑论文论证”仍需要注意的边界

即使 v16.8.12 core + Waymax outcome 全 PASS，WOMD logged replay 本身仍不能提供真实的 counterfactual causal burden ground truth。

代码/论文自己的 evidence protocol 已经区分：

1. WOMD/Waymax logged replay：physical safety、progress、mechanism proxy；
2. independently reactive non-ego protocol：因果 burden-transfer evidence；
3. held-out human-audited false-safe stress set：false-safe/witness validity；
4. publication-level claims：multi-seed + paired scenario-level confidence intervals。

因此，“数据足够训练当前代码”与“数据足够证明最终 causal claim”是两个门。不要把 logged replay 的 burden proxy 写成 causal ground truth。

## 7. v16.8.11 性能结论与 v16.8.12 加速策略

train-pilot 平均 267.95 s/scene；label-engine 266.97 s/scene。

平均阶段：

- safe responses：143.66 s
- witness：71.38 s
- critical selection：22.46 s
- causal relevance：17.06 s
- natural：7.38 s
- pair-neutral：4.07 s
- candidates：0.97 s

safe responses + witness 占 label-engine 时间约 80.3%。

因此：

- 不应通过减少 32-response bank、critical agents、natural roots 或 candidate sources 来加速；那会直接损伤监督性质；
- v16.8.12 只做 route-bank deterministic reuse，避免 natural/pair-neutral 对同一 actor 重复 lane graph search；
- v16.8.11 已有 audit identity exact reuse，继续保留；
- full rebuild worker 数仍应按新 pilot 的 scenes/hour、p90、RAM 和 I/O wait 实测选择，不能根据 CPU 核数盲目增加。

## 8. 本地代码验证

- 新增连续 sparse-lane segment map-compliance regression
- 新增 full-80 logged future 仍为 OBS、但不再作为 normative timing reference regression
- 新增 raw validity timestamp route-match regression
- 新增 priority rejection reason regression
- v16.8.11 regression 继续通过
- full repository：**194 passed, 5 skipped**
- `compileall` PASS
- v16.8.12 smoke/strict/train-pilot/master shell `bash -n` PASS

本环境没有用户机器上的 WOMD shards，因此不能声称 v16.8.12 的真实 smoke/strict 已 PASS。真实 promotion 必须以用户机器重新生成的 verdict 为准。
