# ALGORITHM_CHANGELOG

> 本文件是 v16.8.4 独立补充日志。原始 `ALGORITHM_CHANGELOG.md` 未包含在本轮可访问附件中，不能安全合并历史条目。

## v16.8.4 — Proposal Sufficiency, Attribution, and Certificate-Guided Refinement

### 状态

- `implemented_in_this_package`: 数据/候选库审计、三臂配对 probe、promotion gate、CGPR 独立模块。
- `not_integrated_into_main_repo`: CGPR 与主 candidate generator / label engine / online planner 的连接。
- `not_claimed`: 全量数据重建、mechanism gate 通过、Waymax 闭环提升、SOTA。

### 诊断协议修改

1. 将 proposal promotion 从三项扩展为：
   - `AnyNCFSceneRate >= 0.40`
   - `BestCaseSelectedFalseSafeLowerBound <= 0.55`
   - `BestCasePBTRLowerBound <= 0.45`
   - `HardSceneNCFRecoveryRate >= 0.20`
   - `OldNCFSceneRetention >= 0.95`
   - `ProposalProvenanceCoverage >= 0.95`

2. 新增算法归因 gate：
   - RMR 相对 single-region control 的随机集 AnyNCF 绝对增量；
   - hard-scene recovery 绝对增量；
   - 带 RMR provenance 的独占 hard recovery。

3. 总体 proposal rate 仅在 unbiased random stratum 计算；400 hard scenes 只用于条件恢复率，禁止与 random stratum 直接混合。

4. 任何 requested scenario ID 缺失、重复或 schema 语义歧义均 fail closed。

### 算法设计修改

新增 **Certificate-Guided Proposal Refinement (CGPR)**：

- 输入为 protected-pair 的 conflict mass、OPR shortfall、tail burden、uncertainty 与 robust arrival envelope。
- 按 certificate deficit 排序，优先修复真正导致 hard infeasibility 的关系。
- pass-before 使用 agent early arrival 减 gap；pass-after 使用 late arrival 加 gap。
- 常加速度求解增加“目标时刻前提前停车”拒绝条件。
- 使用 jerk-ramp proxy、加速度边界与 acceleration dedup。
- 输出完整 provenance，所有候选必须重新走 COWP certificate，不把 proposal generation 当作安全证明。

### 保留

- protected hard NCF
- natural roots / RCOT / OPR
- protected BCOT
- certificate/shortlist separation
- explicit uncertified fallback

### 降级或停止

- flat candidate certificate 主路径
- all-critical hard veto
- threshold-only repair
- 在 fixed bank 上重复训练 natural decoder
- sparse cached Waymax/logdiv 作为论文闭环证据
