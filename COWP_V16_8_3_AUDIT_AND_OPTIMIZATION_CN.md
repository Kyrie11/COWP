# COWP v16.8.3 阶段 A 结果审计、算法诊断与优化报告

## 0. 结论先行

本轮阶段 A **不是执行链故障**，也不是 natural decoder、pair witness 或
RootTransport 完全失效。重评估使用了当前 v16.8.2 certificate/fallback 语义，
校准集与 held-out 集严格分离，主要上游质量门均通过；但 mechanism gate 与
calibration feasibility 均失败，因而现在不能进入论文 claim、Waymax full，
也不能声称闭环或理论 SOTA。

当前最主要瓶颈是固定 proposal bank 的可行解覆盖，而不是再调一次 BCOT
threshold：held-out 中 `AnyNCFSceneRate=0.27255`，而
`AnyConventionalSafeSceneRate=0.89146`。由此得到固定候选库下的
selected false-safe 场景率下界至少为 `0.61891`，已经高于 gate 的 `0.55`。
现有 COWP 的实际值 `0.63966` 只比该理论下界高 `0.02075`，说明 selector 已
接近“当前候选库允许的上限”。

同时发现一个会直接压低 proposal coverage 的确定性工程错误：离线候选
过滤器没有使用配置中的 `ignore_initial_jerk_steps=3` 与
`jerk_check_percentile=99`，而在线路径使用了。很多由常加速度产生的合法
pass-before/pass-after、加速、yield/stop 轨迹，仅因为第一个有限差分的 jerk
脉冲被离线删除。因此不应立刻支付约四天全量重建成本；应先运行本轮新增的
1200-scene 配对 label-only probe。只有 probe 证明修复后的候选库确实越过
proposal floor，才进入完整数据重建。

---

## 1. 已阅读材料与论文核心 idea

审计基于：论文 TeX、完整 COWP 代码、算法修改日志、大模型建议、
v16.8.2 执行说明，以及上传的阶段 A 结果包。

论文的核心贡献不是一个普通 trajectory decoder，而是把以下失败模式变成
规划时的硬可行性问题：ego 轨迹虽然自身不碰撞，但其安全依赖其他道路参与者
硬刹、突然让行、放弃优先权或交出合理 gap。COWP 的正确主线是：

1. 形成不受当前 ego candidate 胁迫的 natural behavior roots；
2. 对 candidate--agent--root 估计冲突、same-root 低负担恢复概率 `q` 和最低
   安全负担 `b*`；
3. 用 `s=(1-c)r+cq` 计算 retained root mass / OPR；
4. 对 `AgentPriority` 与 `EqualOrNegotiated` 关系做 protected hard certificate；
5. 无证书时进入显式 uncertified fallback，而不是默认 stop/yield 就安全；
6. 把 COWP 定位为可外挂到更强 proposal planner 上的 non-coercion mechanism
   and certificate layer。

这条核心 idea 值得保留。阶段 A 的证据说明，强 pair/root 排序信号确实存在，
但当前 proposal--certificate--fallback 链没有形成可通过 gate 的 operating
point。

---

## 2. Gate 审计

### 2.1 通过的 gate

| Gate / 审计 | 结果 | 解释 |
|---|---:|---|
| Pipeline preflight | PASS | 配置、checkpoint 路径与执行链可运行 |
| Model anchor preflight | PASS | dataset→critical mapping→natural basis 的模型输入链一致 |
| Natural basis gate | PASS | typed causal dynamics basis、分支几何与 priority/neutral 语义达到阈值 |
| Natural effectiveness gate | PASS | learned residual 相比 analytic basis 有真实增益，且 root identity 未明显崩坏 |
| Cache reuse / overlay integrity | PASS | train 20440、val 5013 完整，零 overlay error、零跨 split 文件重叠 |
| Causal engineering audit | PASS | 已知 future leakage、metric naming、fallback 等工程协议检查通过 |
| Mechanism-overlay protocol | PASS | 现有 v9 base + v16.8 root overlay 足以做 isolated mechanism development |
| Calibration/held-out disjoint | PASS | 2507 / 2506 场景，索引 hash 不同且不重叠 |
| Current certificate semantics | PASS | calibration 与 held-out 均使用 v16.8.2 decoupled semantics |
| Threshold connected to selection | PASS | 23 个预算点形成 18 个不同 selection operating points |
| Pair witness AUPRC | 0.82033 PASS | witness 排序信号强 |
| Protected BCOT false-safe AUPRC | 0.96513 PASS | protected false-safe 排序信号很强 |
| Protected RootTransport AUPRC | 0.86936 PASS | root-conditioned transport 有辨别力 |
| Protected NCF recall | 0.97064 PASS | 有 NCF proposal 时，证书很少漏掉它 |
| Global NCF recall | 0.97252 PASS | 同上 |
| Accepted candidate rate | 0.32956 PASS | 不是 certificate 全空的退化解 |

### 2.2 未通过的 gate

| 指标 | 当前结果 | 要求 | 判定 |
|---|---:|---:|---|
| Calibration feasible | false (`least_violation`) | true | FAIL |
| Protected NCF precision | 0.47628 | ≥0.50 | FAIL |
| Fallback rate | 0.38667 | ≤0.25 | FAIL |
| PBTR improvement over conventional | 0.01261 | ≥0.03 | FAIL |
| Selected false-safe improvement | 0.01237 | ≥0.03 | FAIL |
| Mechanism verification | false | true | FAIL |
| Paper claim ready | false | true | FAIL |

此外 fresh causal-label protocol 仍未通过。这并不是当前 overlay 有错误，而是它
缺少完整 fresh 数据协议所需的 map-filtered / observationally decontaminated
materialization，因而只能支持机制开发，不能作为最终论文数据协议。

---

## 3. 根本原因：为什么不是继续调 threshold

### 3.1 Precision 与 fallback 存在不可兼得区间

在 calibration sweep 中：

- budget 0.70：protected precision `0.50987` 已过线，但 fallback `0.42401`；
- budget 0.75：fallback `0.36857`，precision `0.46037`；
- budget 0.90：fallback `0.21659` 过线，但 precision 降到 `0.35071`。

更关键的是，budget 从 0.70 提高到 0.90 时，PBTR 仅约
`0.68639→0.68817`，selected false-safe 仅约 `0.63024→0.63183`。阈值改变
主要是在“certificate accepted / fallback”的记账边界上移动，并没有改变
场景级选中结果的本质。

### 3.2 固定候选库存在严格 proposal floor

令：

- `A(x)=1`：场景中至少存在一个 conventionally safe candidate；
- `N(x)=1`：场景中至少存在一个 NCF candidate。

因为 NCF 是 conventional safe 的子集，所以对任何只能从固定 candidate bank
中选择、且在 `A=1` 时必须返回 conventional-safe candidate 的 selector：

`selected_false_safe(x) >= A(x) * (1-N(x))`。

当前 held-out：

- `P(A)=0.89146`；
- `P(N)=0.27255`；
- 因此 `P(A=1,N=0) >= 0.61891`。

这已经高于 gate `0.55`。实际 COWP selected false-safe `0.63966`，只比下界
高 `0.02075`。所以：

- classifier/selector 还有小幅优化空间；
- 但不增加真正 NCF proposal，就不可能通过该 false-safe gate；
- fallback-conditioned PBTR 接近 1，进一步说明 fallback 场景本身缺乏可用解，
  而非仅仅 fallback scoring 错了。

### 3.3 Certificate 并非主要丢失点

`CertificateCoverage/NCFSceneRetention` 约 0.975，protected NCF recall 约
0.971。即在候选库已经含 NCF 的场景，证书大多数能保留它。当前最优先不是
放松证书，也不是继续增加 flat candidate head，而是修复候选生成和验证。

### 3.4 确定性工程错误：offline jerk filter 与 online 不一致

原离线 `_candidate_valid` 直接检查 `max(abs(jerk))`，没有读取：

- `ignore_initial_jerk_steps: 3`；
- `jerk_check_percentile: 99.0`。

常加速度 primitive 的第一个离散差分包含从 logged current state 到目标加速度
的瞬时过渡。以 0.1 s 步长为例，2.5 m/s² 的目标加速度会产生约 25 m/s³ 的
首步 jerk；这不等价于整条轨迹持续违反 jerk 约束。在线生成器已经忽略初始
prefix 并使用 percentile，离线 label 却会删除这些候选，造成训练/离线评估与
在线闭环候选空间不一致。

该错误会系统性伤害 accelerate、yield、stop、legacy timing 和 BCTE，尤其会
让“新增了 BCTE 但 cache 中仍看不到 BCTE”的现象误判为算法失败。

---

## 4. 当前数据是否够用，以及是否重建

### 4.1 当前数据已经足够回答的问题

现有数据足以可靠判断：

- natural basis/effectiveness 已经通过，不需重复重训 natural decoder；
- pair witness、protected BCOT、RootTransport 有强排序信号；
- corrected certificate 语义与 fallback accounting 已生效；
- 现有 fixed bank 的 NCF scene coverage 太低；
- 仅调 threshold 或仅重训 selector 不可能越过 proposal floor；
- 现有 overlay 可继续用于同一 candidate bank 下的工程 ablation 与模型调试。

### 4.2 当前数据不能回答的问题

它不能证明：

- jerk 修复后有多少原本合法候选恢复；
- RMR-BCTE 的候选生成率、NCF yield、before/after 分布；
- 新候选是否挤掉已有 NCF candidate；
- full fresh causal-label protocol 是否通过；
- 新 proposal bank 训练后的 closed-loop PBTR/false-safe 是否改善。

### 4.3 本轮决定：先 probe，不立即全量重建

新增 `NEXT_RUN_COMMANDS_V16_8_3_PROPOSAL_PROBE_CN.sh`：

1. 在旧 val cache 上精确计算 proposal floor；
2. 抽取 400 个 `conventional-safe but no NCF` hard scenes；
3. 另抽 800 个无偏随机 val scenes；
4. 只重建这些场景的 fresh labels，不跑训练、不跑 Waymax；
5. 按 scenario ID 严格配对；任何 requested ID 缺失即报错；
6. 输出 source/macro 级 candidate、conventional-safe、NCF、false-safe 统计。

只有以下三项同时满足，才允许完整重建：

- representative `AnyNCFSceneRate >= 0.40`；
- representative false-safe proposal floor `<= 0.55`；
- old hard scenes 的 NCF recovery `>= 0.20`。

这三个阈值不是最终论文 gate，而是“是否值得支付四天成本”的工程 promotion
条件。

---

## 5. 已实施的算法/代码优化

### 5.1 Jerk-consistent validation

离线候选验证现在与在线一致：忽略配置的初始 jerk steps，并对剩余 jerk 使用
配置 percentile。该修复不放松持续 jerk 违规，只排除有限差分初始化伪脉冲。

### 5.2 Physical arrival check

求解 `a=2(d-v0*t)/t²` 后，若 `v0+a*t<0`，该 timing proposal 被拒绝。
否则速度截断为 0 会让车辆在目标时间之前已经停下，却被错误标记为“在目标
时刻到达冲突区”。

### 5.3 Robust Multi-Region BCTE（RMR-BCTE）

相对 v16.8.2 的单点/单区域 BCTE，本轮实现：

- 以 ego nominal TTA 排序 forward-reachable conflict regions，而不是仅取欧氏
  最近区域；
- 最多 3 个 region、每 region 最多 4 个 approaching agents；
- 对 agent 构造 bounded-acceleration early/nominal/late TTA envelope；
- pass-before 使用 `early - gap`，pass-after 使用 `late + gap`；
- gap `{0.8,1.4,2.0}` s；
- 最多 24 个 RMR-BCTE candidates；
- 以 0.1 m/s² acceleration bin 去重，保留 lane-change/stop/terminal 空间；
- offline/online 使用同一 before/after 语义与物理到达约束。

这是 proposal repair，不改变 conventional safety 或 RCOT certificate；因此它
不会通过“生成器名称”绕过安全门。

### 5.4 Proposal provenance 与上限诊断

新增每个候选的：

`proposal_source / region_id / target_time / timing_side / target_agent / gap / accel`。

新增场景级指标：

- `ConventionalWithoutNCFSceneRate`；
- `BestCaseSelectedFalseSafeLowerBound`；
- protected eligible-without-NCF floor；
- `BestCasePBTRLowerBound`；
- `NCFSelectionRecallGivenAvailable`；
- `FalseSafeExcessAboveProposalFloor`。

Calibration 在 proposal floor 已违反约束时会输出 `proposal_infeasible`，防止把
结构性不可行误写成“再找一个阈值即可”。

### 5.5 数据版本完整性

probe 和 full build 均写入代码/config fingerprint。若同一数据根目录已有不同
fingerprint，脚本直接终止，避免四天数据中混入修复前后的候选文件。

---

## 6. 算法取舍

### 6.1 应保留

1. **Non-coercive feasibility 作为 hard constraint**：这是与 courtesy soft cost
   的核心差异，也是论文 novelty 的中心。
2. **Protected relation 语义**：AgentPriority 与 EqualOrNegotiated 做硬保护；
   all-critical 只作 stress diagnostic。
3. **Natural roots + RCOT**：保留 observational / neutral / priority roots，以及
   同 root 的恢复定义。
4. **`s=(1-c)r+cq` 与 `q/b*` 分离**：定义一致且可扩展到 distributional form。
5. **Protected BCOT/witness**：AUPRC 很强，说明这一机制有继续深化价值。
6. **Explicit uncertified fallback**：不能把 stop/yield 当作天然 non-coercive。
7. **Certificate 与 shortlist 分离**：避免评估选择器 top-K 而不是证书集合。

### 6.2 应继续深化

1. **Distributional RCOT**：预测 minimum same-root safe burden 的条件分布，
   用同一分布得到 `q(β)`、`b*` quantile、CVaR 与 one-sided uncertainty；对
   no-safe-response 用 censoring，而非大 sentinel 回归。
2. **Simultaneous calibration**：对 candidate 内所有 protected pairs/roots 同时
   校准 `LCB(q)`、`UCB(b*)`、`LCB(OPR)`、`UCB(CVaR)`；当前单一 budget sweep
   不能形成理论安全保证。
3. **Shift-aware risk control**：ego policy 会改变其他 agent 的交互分布，最终
   需要 importance weighting、robust calibration 或 sequential monitor；不能
   直接宣称 IID conformal guarantee。
4. **Certificate-guided proposal refinement**：当 RMR-BCTE probe 仍不够时，
   在强 proposal planner 输出上沿 certificate surrogate 梯度/搜索局部修正
   crossing time、speed profile、gap 与 lateral timing。
5. **强 proposal backbone 插件化**：将 COWP 放在 flow/diffusion、world-model
   或 VLA proposal planner 后面，竞争 non-coercion 与安全推理，而不是用当前
   kinematic bank 单独竞争所有 planning 指标。

### 6.3 应删除、降级或停止重复尝试

- flat candidate certificate：当前已塌缩，保持 diagnostic-only；
- all-critical hard veto：会把 ego-priority 关系错误升级为同等否决；
- stop/yield 自动判定安全；
- 单纯继续增大 BCOT budget；
- 重复重训已通过 gate 的 natural decoder；
- 用稀疏 cached Waymax outcome 直接做 closed-loop/SOTA claim；
- 在 proposal floor 未改善前继续增加 selector loss/head。

---

## 7. CCF-A / SOTA 路线判断

当前结果不足以声称 SOTA，也无法通过代码修改“保证”理论或实证 SOTA。更可行
的 CCF-A 定位是：**COWP 是一个可插拔、可审计的 non-coercion mechanism and
certificate layer，解决强 planner 普遍未显式约束的 safety-by-coercion 问题。**

论文需要形成四个闭合贡献：

1. 明确定义与 fixed-bank proposal-sufficiency lower bound；
2. distributional same-root burden transport；
3. protected family-wise / shift-aware risk certificate；
4. 在强 proposal planner 上的 certificate-guided refinement，并用 reactive
   closed-loop、human witness audit 和多 seed 展示 conventional safety 不退化、
   PBTR/false-safe 显著下降。

RMR-BCTE 是修复当前实验的必要工程步骤，但不应被包装成最终主要 novelty；它
更适合作为“proposal sufficiency audit + interpretable timing repair”组成部分。

---

## 8. 下一步执行顺序

### 阶段 1：只运行 proposal probe

```bash
cd /path/to/COWP_v16_8_3

WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal \
OLD_VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal/tensor_cache_val_waymax_transport_v16_8 \
PROBE_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_proposal_probe \
HARD_COUNT=400 RANDOM_COUNT=800 LABEL_WORKERS=24 SEED=2026 \
FORCE_REBUILD_PROBE=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_PROPOSAL_PROBE_CN.sh
```

关键输出：

```text
formal_v16_8_3_proposal_probe/current_proposal_ceiling.json
formal_v16_8_3_proposal_probe/paired_proposal_probe.json
formal_v16_8_3_proposal_probe/logs/
```

### 阶段 2A：probe 不通过

不要运行 full rebuild。根据 `proposal_source_stats` 判断：

- RMR-BCTE candidate rate 低：检查 conflict-region discovery、approach filter、
  map screening 与 physical arrival bounds；
- RMR-BCTE 数量高但 NCF yield 低：调整 gap/envelope、引入 piecewise acceleration
  或 certificate-guided local refinement；
- 新 bank 丢失旧 NCF：减小 RMR quota、加强 source-aware quota，避免候选预算挤占；
- false-safe floor 仍 >0.55：当前解析 bank 仍不足，先实现更强 proposal refinement，
  不应训练 selector。

### 阶段 2B：probe 通过后全量重建

```bash
cd /path/to/COWP_v16_8_3

WOMD_ROOT=/data0/senzeyu2/dataset/WOMD/waymo_open_dataset_motion_v_1_3_1 \
COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
TRAIN_LIMIT=22000 VAL_LIMIT=5000 RUN_WAYMAX_REPLAY=1 \
MAX_REPLAY_CANDIDATES=24 CUDA_VISIBLE_DEVICES=0 \
bash PREPARE_COWP_V16_8_3_DATA_CN.sh
```

### 阶段 3：训练 corrected mechanism

```bash
cd /path/to/COWP_v16_8_3

DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
OUT_ROOT=outputs/cowp_v16_8_3_rmr_bcte_seed2026 \
SOURCE_NATURAL_ROOT=outputs/cowp_v16_6_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_6_natural_attribution_aligned_v9labels_seed2026/natural_component_attribution_gate.json \
TRANSPORT_EPOCHS=24 PLANNER_EPOCHS=16 \
CUDA_VISIBLE_DEVICES=0,1 BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_MECHANISM_CN.sh
```

### 阶段 4：gate 通过后才运行 closed-loop probe/full

必须同时满足：

```text
mechanism_verification.pass = true
mechanism_verification.calibration_feasible = true
```

然后：

```bash
DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
OUT_ROOT=outputs/cowp_v16_8_3_rmr_bcte_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_PROBE_CN.sh
```

probe conventional safety、PBTR、false-safe、fallback、progress/comfort 均合格后：

```bash
DATA_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_3_rmr_bcte \
OUT_ROOT=outputs/cowp_v16_8_3_rmr_bcte_seed2026 \
CUDA_VISIBLE_DEVICES=0,1 BACKGROUND=1 \
bash NEXT_RUN_COMMANDS_V16_8_3_FULL_CN.sh
```

---

## 9. 下一轮需回传

若 probe 不通过，回传：

```text
formal_v16_8_3_proposal_probe/current_proposal_ceiling.json
formal_v16_8_3_proposal_probe/paired_proposal_probe.json
formal_v16_8_3_proposal_probe/logs/
```

若进入训练，再回传：

```text
configs/
logs/
checkpoints/transport/history_witness.json
checkpoints/planner/history_planner.json
eval/learned_offline/
eval/causal_protocol_audit.json
eval/probe/  # 若已运行
```
