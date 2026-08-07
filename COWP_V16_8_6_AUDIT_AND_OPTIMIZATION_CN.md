# COWP v16.8.6 数据/算法/实验联合审计与优化报告

## 1. 数据是否现在就要全量重建

**当前结论：不立即重建，但旧 proposal/data bank 已经是最终模型上限；先验证升级后的 proposal，验证成功后再重建。**

这不是“旧 cache 损坏”。旧 train/val transport overlay 结构完整，可继续用于 v16.8 RCOT/BCOT 学习与工程诊断。真正的问题是它冻结了旧 ego proposal tensors 和旧 NCF 标签。

旧 val bank（5,013 scenes）当前 proposal ceiling：

- AnyValidSceneRate = 1.00000
- AnyNCFSceneRate = 0.36246
- best-case selected false-safe lower bound = 0.53321
- best-case PBTR lower bound = 0.58179

当前 PBTR gate 目标为 0.45，因此任何只能从这个固定 bank 选择的 planner，即使拥有 oracle selector，也无法通过这一项。

更重要的是，同一个旧 bank 上此前已完成的 v16.8.3 learned run表现为：protected NCF recall 0.97894、pair-witness AUPRC 0.84945、protected BCOT AUPRC 0.96984、protected RootTransport AUPRC 0.88167；最终 calibration 状态仍是 `proposal_infeasible`。因此“模型还没学会识别现有好 proposal”已经不是最主要矛盾。

不过，你本次上传的 `formal_v16_8_4_bcs_rmr_bcte_proposal_probe.zip` 并不是一个完成后的 fresh verdict archive：它没有 `paired_proposal_probe.json`，也没有新的 fresh completion summary；`build_fresh_probe.log` 仍以 768 个写出 labels 后的 `No valid ego candidates` 异常结束。因此不能根据这个压缩包声称 fresh BCS-RMR 已经 `promote=false`，也不能反过来声称 fresh BCS-RMR 足够好。

所以本轮不直接花数天重建，而是先让一个**更针对 PBTR 的 proposal family**接受 192-scene micro screen，再跑 1200-scene strict probe。只有 strict probe 通过才重建。

## 2. 当前算法哪些有效

### 强正向信号：必须保留并继续深化

在旧 label bank 的 oracle/label-space ablation 中：

- Full COWP FSR = 0.31156；w/o counterfactual = 0.58896，DecisionChangeVsFull = 34.25%。
- w/o neutral branch FSR = 0.39760，DecisionChange = 9.54%。
- w/o priority branch FSR = 0.34879，DecisionChange = 5.34%。

这说明 counterfactual natural roots、neutral intervention 和 protected priority 并不是装饰性模块；尤其 counterfactual branch 是目前最强的核心机制信号。

同时 learned pair/root scores 已经具有很强的排序能力，所以应保留：

- stable natural roots；
- root-conditioned counterfactual transport；
- protected relation；
- `s=(1-c)r+cq` / root-wise OPR；
- BCOT / RootTransport；
- certificate/shortlist separation；
- explicit uncertified fallback。

### 当前边际信号很弱：降级为 safeguard，而不是继续堆复杂度

- `w/o option preservation` 与 full FSR 在当前表格精度下相同，DecisionChange 约 0.02%。
- `w/o witness rejection` FSR 只从 0.31156 变到 0.31182，DecisionChange 同样约 0.02%。
- `soft burden cost only` 也只产生约 0.69% 决策变化。

这些结果不能证明相关理论定义“无意义”，但明确说明**当前 hard decision implementation 不是值得继续扩大复杂度的瓶颈**。在有 learned/closed-loop 证据前，应将其作为辅助 safeguard/diagnostic，而不是主增益来源。

## 3. BCS-RMR 的结构性缺口

v16.8.4 BCS-RMR 修正了 boundary-entry timing，但 non-coercion burden 并不只由最终 TTA 决定。

典型 failure mode：ego 最终确实 pass-after，但前半段仍高速逼近冲突区，直到很晚才明显制动。protected agent 从可观察行为上仍可能被迫提前制动/让行。因此：

> arrival-order correct ≠ early interaction commitment ≠ low burden transfer.

这与当前“false-safe floor 比 PBTR floor 更容易压下来”的现象是一致的：常规安全/到达顺序改善，不代表 protected burden transfer 同步消失。

## 4. v16.8.6：Priority-Commitment Hold-Release

新 proposal 只针对 causally protected interaction：

1. 在 conflict boundary 前留出 stop margin；
2. 用零端点加速度 quintic 早期平滑减速到静止；
3. 显式 hold；
4. 等 protected agent 的 late arrival envelope + gap 后再平滑 release；
5. 仍经过统一 map/accel/jerk/conventional/NCF 校验。

它不修改 NCF ground truth 或 BCOT loss，因此 micro probe 的收益/失败可以归因于 proposal geometry，而不是同时改变标签定义。

本轮同时保证 offline/online 同源，并提前保留 neutral proposal slot，避免新增候选反而挤掉核心 intervention root。

如果这个 proposal 仍不能明显降低 PBTR floor，下一步应进入更一般的 **certificate-guided adaptive proposal refinement**：证书指出哪一个 protected root/pair 失败，再针对该 pair优化 ego approach profile；而不是继续盲目扩 flat timing grid。

## 5. 为什么本次“单 seed 主机制实验”没有 checkpoint

`cowp_planner_best.pt` 缺失不是训练收敛问题，而是训练根本没有开始。

v16.8.5 mechanism wrapper 在 `set -e` 下首先执行 fresh-cache gate。你实际传入的是 v16.8 legacy transport overlay：

- 缺 `build_fingerprint.sha256`；
- 缺 `data_manifest_v16_8_4.json`；
- sampled scenes 全部缺 `proposal_source/region/target_time/...` fresh provenance；
- transport sidecar 是 `.transport_v16_8`，不是 `.transport_v16_8_4`。

因此 gate 先退出，natural/transport/planner 训练没有进入。后面的 offline selection ablation 和 Waymax wrapper继续找 planner checkpoint/检查 fresh protocol，自然连续失败。

这个 strict gate 本身是正确的：如果允许 v16.8.4/6 在旧 proposal tensors 上跑，目录名虽然是新版本，实际方法仍是旧 bank，会制造错误论文证据。

## 6. `Load labels for tables` 为什么慢、如何修复

旧 `05_make_tables.py` 对每个 `.npz` 使用普通 `np.load` 后把大量数组全部物化，包括 table 根本不需要的 natural/response trajectory banks；之后 `module_effect_metrics` 又把各个 label-space planner重复跑第二次。

v16.8.6 修复：

- 只读取 table/certificate 所需 key；
- candidate trajectory 从 `[K,T,D]` 压成 `[K,2,D]`，保留原实现所需的首/尾点 progress；
- `ThreadPoolExecutor` 并行 NPZ I/O；
- 写 `compact_label_table_cache.pkl`，相同文件 size/mtime fingerprint 下后续直接复用；
- module effect 复用第一遍已经算过的 decisions/metrics。

另外，旧 proposal-source ablation 在 stale cache 上把所有候选默认为 PAD，随后会产生虚假的“RMR increment”。现在 stale provenance 会明确 SKIP/FAIL，而不是输出误导结果。

## 7. 外部 baseline 当前结果不能用于论文

上传的旧训练日志中：

- DTPP epoch 1 train：5,099 个 batch 被跳过，只使用 48 samples；epoch 2 同样只使用 48 samples。
- GameFormer epoch 1 train：5,084 batch 被跳过，只使用 108 samples；epoch 2 只使用 104 samples。
- validation 同样跳过约 99% batch。

因此已有 GameFormer/DTPP checkpoint 不能作为 CCF-A 强 learned baseline，后续 50-scene Waymax 结果只能视为 smoke output。

代码审计发现旧 adapter 直接把 WOMD 全局坐标/未来/候选/roadgraph 喂入 matched baseline，并在 FP16 下进行敏感 GMM/trajectory loss；roadgraph padding 也缺显式的局部坐标保护。这些组合容易导致巨量尺度和 non-finite batch，而旧 trainer 又静默 skip。

v16.8.6 已修复 ego-frame transform、roadgraph padding、FP32 loss-side、BF16-auto，并规定 skip fraction >2% 直接失败。

另外，这两个 matched baseline 在本仓库中会对 candidate bank 评分。因此旧-bank修复重训只能作为数值 smoke；最终与 fresh COWP 比较时必须在同一个 fresh v16.8.6 proposal bank 上重新训练/离线评估，否则 baseline 会被旧 proposal ceiling 人为限制，而 Waymax online 又会看到当前候选生成器，形成 train/test proposal mismatch。

## 8. 实验策略

1. 先 PHR 192-scene micro screen。
2. micro 正向后才跑 400+800 strict proposal probe。
3. strict `promote_to_full_rebuild=true` 后 full fresh rebuild。
4. fresh seed-2026 mechanism gate；通过后 Waymax 100 probe。
5. probe promotion 后 Waymax 1000 full。
6. 结构稳定后扩 seed 2027/2028。
7. proposal-source ablation只在 fresh provenance 上做；architecture branch ablation必须独立重训，不能只改方法名复用完整模型 forward。
8. 修复版 GameFormer/DTPP 先 smoke，确认 skip fraction <=2%，再 full。

## 9. CCF-A 方向建议

COWP 的竞争点不应是“有限 kinematic bank 自身必须击败所有强 generative planner”。更可扩展的定位是：

> **backbone-agnostic causal non-coercion certificate + certificate-guided adaptive proposal refinement layer**。

v16.8.6 PHR 是这个方向的第一步：让 certificate 的失败类型决定 proposal geometry。若 PHR 只解决一部分 hard scenes，下一步不是再加更多固定 timing templates，而是把 protected pair/root risk 作为 refinement objective，必要时接到更强的 continuous/generative proposal backbone。

最终理论也需要处理 ego policy 改变交互分布后的 calibration shift，而不能把静态 held-out IID calibration直接解释为 closed-loop risk guarantee。
