# COWP v16.8.3 结果审计与 v16.8.4 优化建议

## 1. 总结结论

当前失败不是训练崩溃，也不是 BCOT/RootTransport 没学到排序信号。根本瓶颈是：**旧 fixed proposal bank 在大量场景中没有任何 NCF candidate，因而在数学上无法同时满足 false-safe/PBTR、precision 与 fallback gate。**

当前缓存是 `v16_8_root_conditioned_overlay`：在旧 base cache 上补充 RCOT/transport tensors。审计明确显示它不是 fresh v16.8.3 label protocol；配置中的 offline jerk 修复与 RMR-BCTE 没有被物化到现有候选标签中。

因此：

- 不要继续在同一候选库上反复训练 natural decoder。
- 不要把 threshold、budget 或 selector loss 当作主修复路径。
- 不要立即进行四天级别全量重建。
- 先执行 1200 场景、三臂、严格配对的 label-only probe。
- 只有 proposal floor、protected PBTR floor、hard recovery、旧 NCF retention 与算法增量全部通过，才完整重建。

## 2. Gate 判定（上传结果包中的真实数值）

| Gate / 指标 | 结果 | 判定 |
|---|---:|---|
| Pipeline preflight | pass | 通过 |
| Model anchor preflight | pass | 通过 |
| Natural basis | pass | 通过 |
| Natural effectiveness | pass | 通过 |
| Cache/overlay integrity | pass | 通过 |
| Causal engineering audit | pass | 工程通过 |
| Fresh v16.8.3 label protocol | false | 失败 |
| Calibration/Held-out split | 2507 / 2506，互斥 | 通过 |
| Pair-witness AUPRC | 0.84945 | 通过 |
| Protected BCOT AUPRC | 0.96984 | 通过 |
| Protected RootTransport AUPRC | 0.88167 | 通过 |
| Global NCF recall | 0.98387 | 通过 |
| Priority NCF recall | 0.97894 | 通过 |
| Accepted-candidate rate | 0.32079 | 通过 |
| Priority NCF precision | 0.49374，要求 0.50 | 失败 |
| Fallback rate | 0.41341，要求 0.25 | 失败 |
| PBTR improvement | 0.01216，要求 0.03 | 失败 |
| Global FSR improvement | 0.01038，要求 0.03 | 失败 |
| Calibration feasible | false | 失败 |
| Mechanism verification | false | 失败 |
| Paper claim ready | false | 失败 |

## 3. 数学不可行性

Held-out：

- `AnyConventionalSafeSceneRate = 0.89146`
- `AnyNCFSceneRate = 0.27255`
- `BestCaseSelectedFalseSafeLowerBound = 0.61891 > 0.55`
- `BestCasePBTRLowerBound = 0.66276 > 0.45`

即使有 oracle selector，只能从当前候选库选轨迹，也无法通过这两个 gate。实际 COWP 的 selected global FSR 为 0.63567，只比 fixed-bank floor 高约 0.01676；这说明 selector 不是主要误差来源。

Calibration partition 也得到：

- selected-FSR floor = 0.60989
- PBTR floor = 0.64457
- calibration status = `proposal_infeasible`

## 4. 为什么 fallback 没有救回来

COWP held-out fallback rate 为 0.41341。fallback 场景中几乎没有可选 NCF candidate；calibration 中 fallback 的 protected burden-transfer rate 约 0.9906。换言之，fallback 当前是在“不存在证书解”时挑 least-coercive 候选，而不是生成新的 NCF 解。它可以改善排序，但不能突破 proposal support。

这也是新增 **Certificate-Guided Proposal Refinement (CGPR)** 的原因：证书不只做拒绝，还应该把 protected-pair 的 root deficit、arrival envelope 与 burden mechanism 反馈给 proposal generator，生成针对性 pass-before/pass-after/stop-before-region 修复。

## 5. 工程问题与数据问题

### 5.1 新配置没有进入旧数据

现有 label 配置已经包含：

- `ignore_initial_jerk_steps: 3`
- `jerk_check_percentile: 99.0`
- RMR-BCTE 最多 3 个 region、每 region 4 个 approaching agents、最多 24 个 timing candidates

但当前数据协议仍是旧 base + transport overlay。运行结果本身也标记：

- `fresh_v16_8_2_label_protocol_pass = false`
- `fresh_v16_8_3_label_protocol_pass = false`
- `v15_label_tensors_materialized = false`

因此不能用当前训练结果判断 jerk 修复或 RMR-BCTE 是否有效。

### 5.2 Waymax attached outcome 只适合作为稀疏辅助监督

每个场景平均约 50.6 个 valid candidates，但只 replay 12 个，valid-candidate outcome coverage 约 23.7%。`finite_logdiv = 0`。当前 selected Waymax 指标必须带 coverage denominator，只能作为局部诊断；它不能替代真实 online Waymax，也不能作为反事实 burden ground truth。

### 5.3 没有数值崩溃证据

Natural、transport、planner histories 持续收敛，未看到 NaN/异常退出。Planner 中 flat candidate-certificate 相关指标跨 epoch 基本不变，与配置 `candidate_certificate: 0.0` 一致。该 flat head 应继续降级，而不是恢复为主机制。

## 6. 算法取舍

### 保留

- protected-priority 上的 hard non-coercive feasibility
- natural roots 与 source identity
- `s=(1-c)r+cq`、transported natural mass/OPR
- 独立的 response existence / minimum burden
- protected BCOT + all-critical diagnostic
- certificate/shortlist 分离
- 显式 uncertified fallback

### 删除或降级

- flat candidate certificate 作为主证书
- all-critical hard veto
- 默认 stop/yield 就是非强迫安全
- 继续仅增加 BCOT budget 或调阈值
- 重复训练已通过的 natural decoder
- 把 sparse cached Waymax outcome 写成闭环结论

### 深化

1. **Certificate-Guided Proposal Refinement**
   - 仅从 protected-pair deficit 触发；
   - 使用 agent early/late arrival envelope；
   - 生成 pass-before / pass-after / stop-before-region 修复；
   - 所有新候选重新经过 physical、map、RCOT 与 certificate；
   - 保存完整 provenance，支持 source-level novelty attribution。

2. **Distributional RCOT**
   - 对 conflict、same-root recovery、minimum burden 使用一侧不确定性界；
   - simultaneous control across all protected pairs，而不是逐 pair 独立阈值。

3. **Policy-shift-aware calibration**
   - calibration planner 与部署 planner 会改变场景分布；
   - 后续应使用 policy-state density ratio 或 importance-weighted LTT；
   - 将 coverage、PBTR、FSR 作为联合风险约束。

4. **强 proposal backbone + COWP 证书层**
   - COWP 的核心竞争力应是 non-coercion certificate/refinement，而非有限 kinematic bank 本身；
   - candidate generator 可替换为强 generative/flow planner，再由 COWP 提供 hard feasibility 与可解释 witness。

## 7. 为什么采用三臂 probe

两臂 old/new 无法区分：

- offline jerk 修复带来的工程恢复；
- 单区域到多区域带来的 RMR 算法恢复。

v16.8.4 使用：

- old fixed bank；
- single-region control（同 jerk 语义）；
- RMR-BCTE（三区域）。

全量重建 gate 与算法归因 gate 分开。这样即便最终 bank 可行，也不会把纯工程修复误当成论文 novelty。

## 8. 不能完成的源码级工作

本轮上传结果包不包含主源码、原始 changelog 或 checkpoint 权重。`run_provenance.json` 只有文件名、大小和 SHA-256，无法还原源代码。因此本包的 CGPR 是独立模块与集成接口，不声称已经接入主仓库；逐行 patch 需以实际 COWP 源码包为基准。
