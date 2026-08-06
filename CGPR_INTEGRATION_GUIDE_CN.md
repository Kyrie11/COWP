# CGPR 主仓库集成说明

由于主源码未上传，下面是必须在真实仓库完成的接口级改动，不是虚构的逐行 diff。

## 1. Candidate generation

在生成基础 kinematic/RMR-BCTE bank 后、达到候选预算前：

1. 从 transport/certificate head 取得每个 protected pair 的：
   - conflict mass
   - OPR / retained mass
   - minimum/tail burden
   - uncertainty
   - forward-reachable conflict regions 与 agent early/nominal/late TTA
2. 转换为 `ProtectedPairCertificate`。
3. 调用 `generate_certificate_guided_refinements`。
4. 通过 `PathProfileAdapter` 将 1D arc-length profile 映射到 ego reference path。
5. 重新执行：
   - dynamic/jerk filter
   - map/lane-corridor filter
   - collision/RSS margin
   - RCOT label generation
   - protected NCF certificate
6. 只在通过上述流程后进入 shortlist。

## 2. 防止训练/在线语义漂移

offline 与 online 必须共享同一个：

- jerk evaluator（ignore steps + percentile）
- constant-acceleration physical reachability helper
- conflict-region ranking
- arrival envelope estimator
- acceleration dedup
- provenance schema

建议把这些函数移到 `cowp/planning/proposal_common.py`，label builder 与 online policy 共同调用，不维护两份实现。

## 3. 候选预算

不要让新增 timing proposals 挤掉旧 NCF：

- 先保留每个已有 source 的 top-N；
- 再分配 protected-deficit repair quota；
- 最后做全局 utility/risk dedup；
- probe gate 要求 `OldNCFSceneRetention >= 0.95`。

## 4. 训练

第一轮 full rebuild 后：

- natural tensors 未变化：复用 natural checkpoint；
- transport labels 变化：重训 transport；
- planner ranking/selection targets 变化：重训 planner；
- flat candidate certificate 仍保持权重 0；
- 先禁用 logdiv；
- calibration 必须重新做，旧 budget 不可复用。

## 5. 论文归因

必须报告：

- jerk-only/single-region control
- RMR-BCTE
- RMR-BCTE + CGPR
- proposal floor、PBTR floor、hard recovery、old-NCF retention
- source-level unique NCF yield

只有 `RMR-BCTE + CGPR` 相对同工程语义 control 有稳定增量，才能作为算法贡献。
