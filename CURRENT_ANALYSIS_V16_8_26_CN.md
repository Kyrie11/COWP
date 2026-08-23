# COWP v16.8.26 当前证据链与下一步决策

## 结论先行

本轮 v16.8.25 CTU + strict Waymax 结果**可以用于可靠的部分算法归因**，没有发现会让已有结果整体作废的工程错误。两份 strict Waymax JSON 都完整覆盖相同的 200 个 exact IDs、每个 80 steps；网络/终端断线没有破坏已保存结果。

但当前结果只能可靠回答三个问题中的第 1 个，并能给第 3 个一个有边界的答案；第 2 个还缺 failure-conditioned temporal diagnostics。因此 v16.8.26 的目标不是继续大改模型，而是补齐这个归因缺口，并做一个只作用于 uncertified fallback 的小算法 probe。

## 1. CTU 是否解决 post-certificate ranking？

**否，而且结果足够稳定，可以把 CTU 作为负消融归档。**

Validation：certificate/fallback invariant，但 CTU 相比 COWP：
- EP `0.642943 -> 0.522353`；
- PBTR `0.465432 -> 0.482716`；
- FSR `0.707524 -> 0.717233`；
- NCF-selection recall `0.795380 -> 0.768977`；
- NPR `0.295410 -> 0.449539`。

Developer-held-out 同方向：
- EP `0.615471 -> 0.500354`；
- PBTR `0.471074 -> 0.490702`；
- FSR `0.700811 -> 0.715010`；
- NCF-selection recall `0.788770 -> 0.751337`。

Strict exact-ID Waymax 200 scenes：
- COWP: CR `0.285`, collision `0.225`, offroad `0.095`, kinematic infeasible `0.105`, EP `0.936086`；
- CTU: CR `0.300`, collision `0.230`, offroad `0.100`, kinematic infeasible `0.100`, EP `0.859208`。

196 个同时有有限 EP 的 paired scenes 中，CTU-COWP mean EP delta = `-0.076878`，bootstrap 95% CI 约 `[-0.161,-0.018]`。

### 方法论判断

原 post-certificate frontier 不是“BCOT 重复处罚”。它提供了有用的 feasible-set 内排序结构。论文方法应从过于简单的

`hard certificate -> planner utility argmin`

调整为

`protected-priority semantic certificate -> certificate-compatible set-preservation/risk-utility frontier -> selected plan`。

这实际上使论文主线更一致：BCOT 不仅定义绝对不可接受区域，还描述 feasible set 内 option-preservation robustness；CTU 证明把这个结构完全丢掉会导致更差的 progress/NCF trade-off。

## 2. physical unsafe/offroad 是否 selector-side bottleneck？

**当前不能可靠定性。**

COWP strict 200-ID：
- CR `28.5%`；
- collision `22.5%`；
- offroad `9.5%`；
- fallback step rate `62.625%`；
- fallback episode rate `91.5%`。

这说明 online execution 里 fallback 非常频繁，但由于 v16.8.25 只保存了 aggregate fallback 与 episode physical event，无法判断第一次 collision/offroad 发生前正在执行 accepted COWP 还是 fallback。91.5% episode 有过 fallback 时，简单的 episode-level correlation 几乎没有鉴别力。

因此目前只能说：**online certificate availability / fallback exposure 是当前最紧迫的 operational bottleneck 候选**，不能直接说“物理失败由 fallback 导致”。

v16.8.26 已加入：
- first collision/offroad/kinematic-infeasible step；
- first event 前 fallback step fraction；
- first event 前一个动作是否来自 fallback；
- 每 episode no-certificate/no-conventional/no-valid 比例；
- exact same IDs 下 conventional-safety / planner-score-only strict baselines。

下一轮即可区分：
1. fallback-side failure；
2. accepted-selector/physical shield failure；
3. 所有方法都高 failure，说明 common online candidate/action projection/dynamics 是瓶颈。

## 3. Outcome head 是否有资格进入下一轮算法？

**有资格进入“fallback-only soft ranking probe”，没有资格进入主 certificate / hard physical shield。**

现有 checkpoint：
- val collision AUPRC `~0.585`, offroad `~0.429`, unsafe union `~0.728`；
- held-out collision AUPRC `~0.598`, offroad `~0.436`, unsafe union `~0.733`。

优点：unsafe-union 信号稳定，不是随机 head。

不足：
- offroad 判别能力偏弱；
- 未报告 prevalence；
- 未报告 calibration/Brier/ECE；
- 未报告低 FPR 区域 recall；
- attached candidate Waymax outcomes 只覆盖约一半 scenes，且它对 CTU 的方向与 strict online Waymax 不一致，不能用于主算法决策。

因此 v16.8.26 新增 `cowp_fallback_outcome`：只有 certificate 为空进入 explicit uncertified fallback 时，outcome head 才参与 fallback ranking。certified COWP path 中 outcome risk 被显式置零，不进入 frontier。

## 当前模型状态

### 已经学得比较好的部分

1. **Root-conditioned transport / RCOT**：held-out root low-safe existence AUPRC 约 `0.897`，conflict-conditioned `0.803`，priority-conflict `0.782`。
2. **BCOT candidate certificate**：priority/global false-safe AUPRC 约 `0.837/0.928`。
3. **Protected-priority hard feasibility 的方向**：COWP 相比 soft burden 与 conventional selection 的 PBTR/FSR 更好；universal NCF fallback 明显更高，支持 protected-priority 而不是 all-critical hard veto。
4. **post-certificate set-preservation frontier**：CTU 负结果证明其有独立实际价值。

### 没学好或尚未被证明的部分

1. generic candidate classifier 很弱，NCF/false-safe AUPRC 约 `0.176/0.354`，不应转正为主 certificate。
2. outcome head 的 offroad 与 calibration 尚不够强，只能做 fallback probe。
3. affected-root 的额外语义没有数据证据：当前 affected 与 unsafe root 重合，burden-only root=0，不能声称已学到“无碰撞但高 burden affectedness”。
4. online repeated-replanning 状态上的 certificate availability / fallback recovery 没被当前 offline checkpoint protocol 充分验证。
5. 原训练 `stage=all` 中 mechanism/planner 共同更新、best by total loss，使 planner attribution 不够干净；但现在不应优先做 planner-only repair，因为 strict online fallback/physical attribution比它更紧迫。

## Dominant bottleneck 的层级

### 全局固定-bank理论瓶颈

仍然是 proposal support：held-out AnyNCF 约 36%，fixed-bank false-safe oracle floor 约 59.5%。任何 threshold/selector 都不能突破候选不存在的问题。

但按照当前“不重建数据”的约束，这不是本轮 actionable target。

### 当前最值得先解决的 no-rebuild bottleneck

**online feasibility availability + uncertified fallback physical quality。**

原因：strict online 62.6% policy steps 在 fallback，而 CR 28.5%。在线状态随 ego policy 演化，已经不再等价于 cache root 上的一次性 learned-offline selection。下一轮必须先确认 physical failures 是不是集中在 fallback。

## 下一步算法：Fallback-Only Outcome Risk

这是一个小改动，不改变论文核心方法。

- Certified path：完全保持 COWP。
- Uncertified path：在原 least-coercive fallback score 中启用 already-trained collision/offroad union risk。
- 不把 outcome probability 当作 non-coercion evidence。
- 不让 outcome head 改写 protected set、RCOT、BCOT、certificate threshold 或 certified frontier。

如果 strict paired 结果证明 fallback-only risk 能显著降低 CR/offroad，同时不造成明显 EP collapse，那么再考虑把它抽象成更漂亮的 **Feasibility-Preserving Recovery**：主 certificate 负责社会/机制可行性，certificate 为空时进入明确标记为 uncertified 的 physical-risk-minimal recovery；这比把所有风险塞进一个 scalar objective 更符合 COWP 的方法论。

如果它无效，则不继续堆 outcome penalty，而根据 failure localization：
- accepted path failure 高 -> 研究 execution viability shield；
- conventional/planner 也同样高 -> 修 online candidate/action projection/common dynamics；
- fallback failure高但 outcome guard无效 -> outcome head 不够好，fallback 应转向 analytic physical viability / short-horizon rollout，而不是继续训练 classifier。

## Waymax 为什么慢，以及 v16.8.26 的处理

上一版其实已经让 PyTorch 用 A30-0、Waymax/JAX 用 A30-1，即两张 GPU 都在工作。主要问题更可能是单 scene、逐 step 的串行跨框架 pipeline：

`JAX state -> host NumPy -> CPU feature/candidate generation -> PyTorch GPU -> host decision -> JAX action -> Waymax env -> JAX metrics`

200 scenes x 80 steps = 16,000 次完整 replanning / method。

代码审查发现明确冗余：trajectory 的九个字段此前逐字段 `jax.device_get`，且 timestep extraction 重复。v16.8.26 改为一次 batched device-get，并缓存 scenario-invariant SDC index。

另外加入：
- exact-ID TFExample index，避免启动时扫无关 raw shards；
- component runtime profiler；
- current split-GPU vs single-A30 co-location vs dual-A30 two-shard parallel benchmark；
- two-process / two-A30 exact-ID throughput launcher。

不要直接假设“两个并行 process 一定更快”：如果每张 A30 同时容纳 Torch+JAX 导致显存压力/上下文争用，可能反而慢。因此先跑 12-scene profiler，再决定 200-scene protocol。

## 本轮不应该做的事情

- 不重建 dataset/cache；
- 不重新设计 proposal；
- 不跑 planner-only repair；
- 不把 CTU 继续作为候选主方法；
- 不把 outcome head 加到 hard certificate；
- 不重新尝试 PCHR/PSY/RMR 式“再加一个 primitive”；
- 不使用上传的 MCFC probe 作为证据：其 `profile_labels.jsonl` 为 0 行，且没有 paired probe/source ablation/promotion 文件。

## 下一轮最关键的算法问题

> 当 protected-priority certificate 在线变为空时，当前 physical failures 是由“fallback 本身选择错误”引起，还是由“所有在线候选/动作投影都缺乏闭环可执行性”引起？

这个问题一旦回答，下一步才有资格决定是做：
1. Feasibility-Preserving Recovery；
2. accepted-path execution viability shield；
3. online candidate/action interface redesign。

这比继续改 BCOT head 或重新构造数据更接近当前真正的 bottleneck。
