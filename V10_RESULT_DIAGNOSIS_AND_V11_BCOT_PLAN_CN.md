# COWP v10-GCT 结果诊断与 v11-BCOT 优化方案

## 1. v10 是否超过基线

结论是：**v10 在核心机制学习和部分离线多目标指标上明显超过 v9，并在若干阈值下超过 conventional/soft-burden 基线，但没有形成整体支配，更没有闭环 SOTA 证据。**

### 校准点（threshold=0.50）

| 指标 | Conventional | v10-GCT | 变化 |
|---|---:|---:|---:|
| EP ↑ | 0.3894 | 0.3913 | +0.0019 |
| Fallback ↓ | 0.1043 | 0.2406 | +0.1362（显著变差） |
| OPR ↑ | 0.7418 | 0.7644 | +0.0226 |
| HBCR ↓ | 0.3970 | 0.2928 | -0.1041 |
| Selected False-Safe ↓ | 0.5899 | 0.4674 | -0.1225 |
| Accepted NCF Recall ↑ | 1.0000* | 0.1267 | 核心失败 |
| Accepted Candidate Rate ↑ | 0.5489* | 0.0585 | 核心失败 |

`*` conventional 不使用 learned certificate，其接受率定义不能作为相同机制的分类能力比较，只用于显示候选集规模。

阈值 0.70 时，v10 的 EP 达到 0.4630、selected false-safe 为 0.5388、fallback 为 0.1552；说明方法存在真实 trade-off，但所有阈值下 NCF recall 最高仅 0.1391，因此证书仍然错误拒绝绝大多数真正 NCF 候选。

## 2. v10 中哪些部分有效

### 2.1 几何条件化 transport 有效

`mode_conflict` 验证 BCE 从 v9 的约 0.722 降至 0.507，低于约 0.649 的类别先验熵基线。pair witness AUPRC 从约 0.431 提升到 0.716。由此可判定：显式 candidate--natural 相对几何和 direct mode supervision 是正确方向，应保留并作为论文机制的一部分。

### 2.2 granular freeze 有效

candidate、natural、witness 与 transport 模块能够随新监督更新后，pair-level 机制出现大幅提升。不能恢复 `FREEZE_BACKBONE_EPOCHS=999` 的旧策略。

### 2.3 false-safe 与负担指标改善是真实信号

相对 conventional，校准点 false-safe 下降 12.25 个百分点，HBCR 下降 10.41 个百分点，OPR 提升 2.26 个百分点。论文关于“普通 collision-free 仍可能压缩他车低负担选择”的核心动机得到部分支持。

## 3. v10 的根本瓶颈

### 3.1 pair AUPRC 高，但 candidate recall 极低：聚合器失败

v10 把最多六个 agent 的 pair certificate 通过 `any/max` 聚合。若每个 pair 均存在中等假阳性概率，候选被至少一个 pair 误拒绝的概率随 agent 数快速增加。结果是 pair AUPRC 已达 0.716，但 candidate NCF recall 上限仍只有 0.139。

这不是继续微调 witness threshold 能解决的问题；阈值扫描已经证明所有 operating point 都存在同一瓶颈。

### 3.2 same-root recovery 与标签语义不匹配

标签表示“是否存在一条低负担安全 response 能恢复某个 root”。v10 使用 response mixture weight 求和，相当于期望质量，而非存在性。root-recovery loss 约 0.837、response-root CE 约 2.446，表明该链路仍弱。

### 3.3 generic candidate classifier 威胁 novelty

Candidate false-safe AUPRC 为 0.904，显著高于 pair witness 0.716。若最终选择主要依赖 generic candidate latent，审稿人可以认为 option transport 只是辅助任务。下一版必须限制 candidate calibrator 只能使用可审计 transport statistics，并通过 pairmax/candidate-only 消融证明贡献来源。

### 3.4 没有真实闭环结果

learned-offline gate 失败后脚本停止，结果包中不存在 100-scene Waymax probe。附着的 candidate replay outcome 覆盖率不同，而且不是在线滚动选择，不能用于 SOTA 判断。

## 4. v11-BCOT 核心设计

### 4.1 Pair-level option deficit

对 candidate k、agent i，定义可审计 deficit：

- 被 candidate 冲突且没有 same-root 低负担恢复的 natural option mass；
- 冲突质量上的 burden excess；
- OPR 相对最低可接受水平的 shortfall。

这些量均由 primitive-indexed transport 产生，保持论文 novelty。

### 4.2 Candidate-level coherent budget

候选风险不再使用 `max/any`，而由以下三部分构成：

1. priority-weighted mean deficit：表示总体被压缩的选择质量；
2. smooth tail deficit：防止均值掩盖最受影响的 agent；
3. severe protected-pair probability：保留对真正严重优先权侵犯的敏感性。

主 gate 对该风险使用 budget。仅当 pair 同时满足高 witness、高优先权、低 OPR、低不确定性时，才执行 hard veto。

### 4.3 Existential root recovery

使用 top response slots 的 fuzzy max，而不是 mixture expectation 或 all-slot noisy-OR：

- 重复响应槽不会累计到 1；
- 均匀 root distribution 只贡献 1/M；
- 高置信、低负担、安全 response 可接近 1；
- 与“存在一条恢复路径”的标签一致。

### 4.4 机制隔离

candidate residual calibrator 不再读取通用 candidate embedding，只读取 BCOT 风险、均值/tail deficit、severe probability、uncertainty、OPR、burden、conflict 等 transport statistics。最终层零初始化，保证从 v10 checkpoint 载入时先保持解析证书，再逐步学习校准。

## 5. 下一轮判定逻辑

首先检查：

- `val/set_transport/candidate_budget` 是否下降；
- `BCOT/FalseSafe_AUPRC >= 0.65`；
- `BCOT/RiskRankingPairAccuracy` 是否高于 0.75；
- NCF recall 是否突破 v10 的 0.139，开发门槛为 0.30；
- pairmax ablation 是否重新出现高 fallback/低 recall。

只有离线 gate 全部通过，才运行 100 场景 Waymax。100 场景仅用于检查方向；通过后运行 1000 paired scenes，再进行在线 hard-negative mining和三随机种子，最终运行 5000 paired scenes。

## 6. 不能保证的事项

本版本是由 v10 结果直接支持的结构性修复，但代码修改本身不能保证 SOTA 或 CCF-A 接收。最终结论必须来自真实多场景、多种子、配对闭环实验及关键消融。
