# COWP v16.8.8：v16.8.6 Micro-Probe 审计、数据语义修复与 Proposal Refinement

## 一、结论

当前不建议 full rebuild。v16.8.6 的 191-scene fresh label 构建本身是完整的（191 written，0 filtered，0 error），但 128-scene representative comparison 暴露出两个不同方向的信号：protected-priority burden 显著改善，而 universal NCF 明显退化。PCHR 本身几乎没有有效产出，因此不能把 PBTR 改善归功于 PCHR；代码审计又发现 critical-agent 集合依赖整个 optional proposal bank，使新增 proposal 会改变所有旧候选的 NCF 审计对象。这是必须先修的数据/标签语义缺陷。

v16.8.8 因此不直接扩大 PCHR，而是先固定 critical universe、保护 base proposal bank，再以一个更可行的 Priority-Smooth-Yield (PSY) proposal 验证“早期让行承诺”方向。只有 96-scene smoke 和 1200-scene strict probe 分层通过，才允许 full rebuild。

## 二、Recovered micro-probe 的所有关键指标

128 个 representative random scenes：

| 指标 | old v16.8 bank | v16.8.6 fresh bank | 解释 |
|---|---:|---:|---|
| AnyValid | 1.0000 | 1.0000 | 数据没有 0-valid / build failure 问题 |
| mean valid candidates | 50.445 | 52.859 | 新 bank 候选数量反而增加 |
| AnyNCF | 0.36719 = 47/128 | 0.203125 = 26/128 | universal NCF 严重退化 |
| false-safe floor | 0.578125 = 74/128 | 0.765625 = 98/128 | strict gate 失败 |
| PBTR floor | 0.58678 = 71/121 | 0.43089 = 53/123 | protected-priority 已跨过 0.45 gate |
| hard recovery | — | 0.125 = 8/64 | micro threshold 勉强过，strict 0.20 未过 |
| NCF loss | — | 0.22656 = 29/128 | 旧 NCF 大量被新标签破坏 |

由 NCF transition 可进一步得到：旧 47 个 NCF 场景中仅 18 个被保留，29 个丢失；新 bank 另外获得 8 个新 NCF 场景，最终 26 个。

false-safe 与 NCF 是互补结构（NCF 是 conventional-safe 子集），所以 old 至少一个 conventional-safe 的场景约为 74+47=121；new 约为 98+26=124。也就是说**新 bank 的常规安全 coverage 并没有下降，反而略升；真正下降的是 non-coercion certificate coverage。**

protected-priority 层面更加明显：old 121 个 priority-eligible scenes 中 50 个有 priority-NCF；new 123 个中约 70 个有 priority-NCF。protected-priority 子问题实际上变好，而 universal NCF 变差。

## 三、PCHR 是否有效

不能说 PCHR 产生了 PBTR 的正向结果。它只在 4/128 场景出现，合计 17 条有效候选，NCF=0，priority-NCF=0。它通过“完整停住—hold—release”表达早期让行的方向在概念上合理，但现实 3–7 s interaction window 中物理可行条件过苛，候选产率太低。

因此 v16.8.8 将 PCHR 默认关闭，保留为 ablation，不继续扩大 stop margin / release speed / gap grid。继续扩大同一 family 会增加候选预算和 label 成本，却没有现有 NCF yield 证据支持。

## 四、真正的数据语义 bug：critical-agent universe 随 proposal bank 漂移

v16.8.7 及之前的 `select_critical_agents(scene, cfg, ego_candidates, ...)` 会把整个 ego candidate bank 用于 critical-agent scoring。新增 RMR/PCHR trajectory 可以使新的 agent 满足 min-future-distance/shared-conflict/TTA 条件并进入 top-8；因为 NCF 要对所有 selected critical agents 同时成立，critical set 一变，原来完全没有修改的旧 trajectory 也可能从 NCF 变成 non-NCF。

这违反了 proposal refinement 应有的基本实验语义：如果只是把 proposal set 从 B 扩展到 B∪R，oracle feasibility ceiling 不应该因为“新增了选项”而变差。过去的实现实际上同时修改了 proposal set 和 certificate universe，因此 v16.8.6 的 NCF collapse 被混杂了。

v16.8.8 默认 `critical.selection_reference_mode=fixed_anchor_v1`：critical selection 只看与 optional proposal 无关的固定 anchor bank（logged reference、keep、canonical accel、canonical yield、smooth stop、左右 canonical lateral anchors）。optional RMR/PSY 不再重定义 critical set。历史 `proposal_bank_legacy` 仅作为兼容消融。

## 五、为什么 protected-priority 是值得深化的正向信号

PBTR floor 0.587→0.431 是目前最强的新信号，但由于 PCHR 0 NCF、critical universe 又发生过漂移，不能把这个数字写成“PCHR 已有效”。更稳健的解读是：BCS-RMR / protected priority / response-root machinery 已经能找到更多“不迫使 protected agent 让步”的候选，说明论文核心的 protected burden 方向值得保留；下一步需要在稳定 certificate universe 下验证一个更可行的 early-yield proposal family。

RMR 本身也应保留：223 条候选中 25 条 NCF，且 target-TTA error mean≈0.00070 s、max≈0.00884 s，远低于 0.20 s gate。RMR 的 boundary-consistent timing 语义已经不是当前主要缺陷。

## 六、v16.8.8 Priority-Smooth-Yield (PSY)

PCHR 要求 ego 先完全停住，因此可行窗口太窄。PSY 改为求解 quintic longitudinal trajectory：当前速度/给定初始负加速度出发，在 protected agent `late_tta + gap` 时达到 conflict-entry distance，并指定较低 terminal speed，同时 terminal acceleration=0。候选还必须在约 1 s commitment window 内出现最低速度下降，随后仍经过统一 map/dynamics/jerk/TTA validator。

默认搜索：terminal speed {1,2,3} m/s、initial decel {-0.8,-1.4,-2.0} m/s²、gap {0.8,1.4,2.0}s，每 scene 最多 8 条。它不是“再扩一个 timing grid”，而是显式控制 approach profile，在不强迫 full stop 的前提下表达 early yielding commitment。

## 七、base-bank preservation 与 offline/online 对齐

offline 中 neutral、lane-change、creep 等语义上独立的 base actions 在 optional RMR/PCHR/PSY 之前占位；online Waymax 也在 optional refinement 前保留 neutral 和一个小型左右 lateral reserve bank。这样新 proposal 只能替代 filler/冗余候选，不能因为 K 饱和把论文核心 neutral root 或两侧 escape action 删除。

PSY 同时落地到 offline 和 online generator。否则 fresh labels 中出现的行为在 Waymax 闭环消失，会产生 train/test proposal mismatch。

## 八、数据构建完整性与全量重建成本

191-scene profile：label engine mean≈287.33 s；safe responses≈164.48 s（约 57%）；witness≈99.70 s（约 35%）；critical selection≈21.32 s（约 7%）；candidate generation≈0.86 s（<1%）。因此当前多日成本不来自 RMR/PCHR proposal generation，而来自 response/witness counterfactual certification。

不建议为了速度降低 `root_conditioned_transport.label_search_profiles`、response search budget、natural roots 或 witness 阈值，因为这些直接定义 q/OPR/NCF ground truth。当前安全加速包括：只把 allowlisted Scenario 送给 worker、复用旧 base tensor cache 的 scenario IDs 而不复用旧 labels、限制每个 worker 的 BLAS/TF 线程、跳过 full-train cached Waymax replay、inline self-contained transport、严格 fingerprint resume。

## 九、数据能否支撑训练/测试/论文论证的 gate

v16.8.8 不只检查“文件存在”。96-scene smoke 首先要求 stable critical semantics、PSY generation/yield 和 proposal union monotonicity。通过后 1200 strict probe 要求 AnyValid>=0.99、AnyNCF>=0.40、false-safe<=0.55、PBTR<=0.45、hard recovery>=0.20，并要求 PSY 有非零且达到最低比例的 protected-priority NCF yield。

strict verdict 带 code/config fingerprint；full build 开始前重新核对。full data 构建完成后又对实际完整 validation tensor cache 重算 ceiling 和 source ablation，并再次要求 aggregate gates、PSY protected-priority yield 和 proposal-union monotonicity全部通过，才允许模型训练。因此 full rebuild 有两道前置防线和一道 post-build 防线。

## 十、当前决策

**不建议 full rebuild v16.8.6/v16.8.7。** 当前 `screen_pass=false` 不是“protected priority idea 失败”，而是“PCHR 直接 yield 失败 + critical-universe 数据语义存在混杂”。应先执行 v16.8.8 48-hard+48-random smoke。smoke fail 就继续 proposal/label refinement；smoke pass 才跑 400+800 strict。只有 strict JSON 中 `recommend_full_rebuild=true` 且 fingerprint 与当前代码一致时，才建议支付 full rebuild 成本。
