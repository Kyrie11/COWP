# ALGORITHM_CHANGELOG V16.8.32 — Temporal Option Persistence

## Status

V16.8.31 的结果通过完整性审计，可以做算法归因；但 **BHOV 未通过预注册 promotion gate**，因此 V16.8.32 不做阈值/权重修补，而是围绕暴露出的两个根因做最小、可证伪分解：

1. one-step successor option signature 可能太短视，无法发现 delayed option collapse；
2. 每 0.1 s 在 COWP/RVR 间重新判定会形成第三种 hybrid closed-loop policy，缺少 recovery-mode commitment 本身可能制造失败。

V16.8.32 新增两个 opt-in 方法：

- `cowp_trihorizon_option_persistence`（THOP）：主机制 probe；
- `cowp_sov_recovery_commitment`：mode-consistency diagnostic。

两者都只作用于 `full conventional set == empty && valid candidate exists`，不改变 COWP certified/common path。

---

## V16.8.31 结果结论

### 可靠性

- `sanity`: 29/29 passed；
- equivalence16 COWP vs V16.8.29 reference：1120 fields / 0 mismatch；
- counterfactual48 两个新方法均为 24+24 disjoint shards，union 精确等于 manifest；
- holdout64 为 32+32 disjoint shards，union 精确等于 manifest；
- merged summary 可由逐场景 rows 零误差重算；
- holdout64 与 V16.8.30 equivalence16/counterfactual48/balanced96 overlap=0；
- 所有运行复用同一 checkpoint；online 不使用 mechanism ground truth。

结论：**没有算法归因阻断。**

### Counterfactual48

- COWP collision = 34/48；
- RVR = 33/48；
- SOV = 33/48；
- BHOV = 29/48；
- restore-only = 34/48。

BHOV 相对 COWP：11 rescue / 6 induced，net -5 collision，但：

- 旧 RVR rescue 保留 10/10；
- 旧 RVR induced 仅避免 3/9；
- kinematics 额外 +3 scenes；
- paired EP delta ≈ -0.1500，95% bootstrap CI ≈ [-0.3277, -0.0198]。

因此 BHOV 是 **high-recall / low-precision recovery acceptance**，未达到 non-harmful promotion 条件。

Restore-only 仅极少介入，旧 RVR rescue 0/10、induced avoidance 9/9，说明单纯 `successor conventional exists: 0→1` 太稀疏，不能作为 recovery criterion。

### Outcome-blind holdout64

- COWP collision = 0/64；
- RVR collision = 0/64；
- BHOV collision = 1/64；
- BHOV vs COWP = 0 rescue / 1 induced；
- paired EP delta ≈ -0.0192，CI 跨 0。

唯一 induced scene：`3356dd85996d9c1d`。

该 scene 中：

- COWP safe；
- pure RVR safe；
- BHOV collision；
- BHOV 在 no-conventional regime 中间歇执行 RVR，形成 COWP/RVR 都不等价的 hybrid trajectory；
- first collision 发生在较晚 step 59，并非单步投影 bug。

结论：**V16.8.31 BHOV 未通过上一轮预注册的 holdout64 non-harmful GO gate，不运行 exact200，不 promotion。**

---

## 保护/冻结的方向

以下层已有稳定正证据，V16.8.32 不允许改变：

- compact-5k data/label contract；
- natural behavioral roots；
- RCOT same-root transport；
- BCOT structured certificate；
- protected-priority hard feasibility semantics；
- certificate-compatible set-preservation frontier；
- 8 s conventional collision-screen contract；
- V16.8.27 conventional-safety integrity；
- V16.8.28 no-valid bounded execution integrity。

Outcome head 保持 diagnostic-only；不进入 hard physical certificate。

---

## 明确禁止继续尝试的方向

基于 V16.8.25→31 的负证据，后续禁止：

1. CTU / `certificate -> planner-score argmin` 替代 set-preservation frontier；
2. outcome fallback weight 调参或把 outcome head 升级成 hard shield；
3. 通过缩短 8 s conventional horizon 来制造更多“safe”候选；
4. max-prefix RVR 直接 promotion；
5. Pareto guard tolerance/权重搜索；
6. BHOV comparator 放宽、epsilon/tolerance 调参；
7. 将 social / physical / utility 再揉成单个 scalar penalty；
8. 当前阶段扩 proposal primitive、route/Frenet/map repair（roadgraph_empty 并非 dominant）；
9. 当前阶段修改 RCOT/BCOT budget、threshold 或重训 generic candidate safety classifier；
10. 在 analytic physical target 未验证前训练新的 successor neural head；
11. 将 accepted-path kinematics 与 zero-conventional recovery 混在同一轮修改。

---

## 新机制 1：Tri-Horizon Option Persistence (THOP)

Method: `cowp_trihorizon_option_persistence`

### 目标

验证 V16.8.31 的 one-step successor statistic 是否因为短视而接受会在更晚时刻 option collapse 的 recovery action。

### Intervention scope

仅在：

```text
full conventional set == empty
AND valid candidate exists
```

时比较：

- base = 原 COWP least-coercive-valid fallback；
- alt = 原 V16.8.29 RVR max-prefix candidate。

certificate/common conventional path bit-semantically unchanged。

### 三层信息

- `H0`: current causal collision-safe prefix；
- `V1`: 执行第一步 **actual jerk/yaw-rate-limited emitted action** 后的 successor option signature；
- `V2`: 沿当前 recovery trajectory 再执行一个同样受控制器约束的 emitted step 后的 second-successor option signature。

`V1/V2` 均使用同一 online physical proposal generator 与同一 conventional screen，周围 agents 使用与现有 screen 一致的 causal CV rollout；不读取 future logs / Waymax ground truth。

### Acceptance

严格 product-order：

```text
H0_alt >= H0_base
V1_alt >=lex V1_base
V2_alt >=lex V2_base
AND 至少一项严格改善
```

不存在权重、epsilon、阈值放宽或新训练参数。

为降低成本，只在 V16.8.31 BHOV 的 H0+V1 pre-gate 已成立时计算 V2。

### 论文定位

THOP 本身只是 analytic mechanism probe，不宣称 formal viability kernel。若有效，吸收的是 **temporal physical option-set persistence** 这一机制对象，而不是“三步 lookahead”技巧。

---

## 新机制 2：SOV Recovery Commitment

Method: `cowp_sov_recovery_commitment`

### 目标

独立验证 V16.8.31 holdout64 暴露的第二个根因：stateless 每步 gate 是否因 COWP/RVR 间反复切换而形成 harmful hybrid policy。

### Entry

仅使用 V16.8.30 的 strict high-precision SOV 条件：

```text
V1_RVR >lex V1_COWP
```

进入 recovery mode。

### Continue

一旦进入，在仍然 `zero-conventional && valid` 的期间继续原始 RVR recovery，不在每 0.1 s 重新 gate。

### Exit

当：

- certificate/common conventional option 恢复；或
- no-valid emergency branch 出现

时清除 commitment。

没有固定 dwell time、没有 hysteresis threshold、没有 learned mode classifier。

### 定位

这是 diagnostic branch，不单独作为论文 novelty；它用于区分：

- successor target 本身短视；
- mode-switch inconsistency/chattering。

---

## 新实验协议

### Stage 0 — sanity / equivalence16

首先证明 common COWP path 未改变。

### Stage 1 — counterfactual48

同时测试 THOP 与 commitment。

对每个方法预注册 GO：

- retain >= 5/10 old RVR rescues；
- avoid >= 7/9 old RVR induced collisions；
- 相对 COWP net collision reduction >= 3 scenes；
- kinematics net regression <= 1 scene；
- paired mean EP delta >= -0.05；
- intervention rate > 0。

未通过即 archive，**禁止调阈值救结果**。

### Stage 2 — fresh37

V16.8.32 新增 `fresh37`：exact200 中剔除 V16.8.30/31 已用于 mechanism selection 的所有 163 个 scene 后，剩余精确 37 个 ID。

选择过程不读取 V16.8.32 outcome；manifest hash：

`ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481`

它仍非 publication holdout，因为 exact200 历史上已经被整体看过，但可作为当前 lineage 中最干净的 development generalization gate。

GO：

- no net collision harm；
- no net CR harm；
- offroad regression <= 1 scene；
- kinematics regression <= 1 scene；
- paired mean EP delta >= -0.03；
- intervention rate > 0。

### Stage 3 — exact200

只有 Stage 1 + Stage 2 都通过的方法才允许跑 exact200，且只跑 promoted new method，复用 immutable COWP/RVR references。

exact200 仍只是 development confirmation；算法 freeze 后必须另建未参与 mechanism selection 的 final evaluation set。

---

## 当前主线

V16.8.32 不改变论文的上层主线，而是进一步收紧为：

**Orthogonal Option-Set Feasibility**

- Social axis: preserve other critical actors' natural low-burden option sets；
- Physical-temporal axis: preserve ego's future executable option sets under actual emitted recovery actions。

当前 dominant bottleneck：

**Temporal Option-Set Persistence + Recovery-Mode Consistency under Uncertified Recovery**。

Proposal support 仍是长期 global ceiling；accepted-path kinematics 仍是独立 secondary bottleneck。

---

## Validation

- V16.8.32 focused semantic/integrity suite: **34 passed**；
- launcher `sanity`: **34 passed**；
- Python compile / bash syntax checks passed；
- full repository run 在当前执行时间窗口无法完成；`pytest -x` 的首个 failure 是历史缺失 launcher `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`，此前 **124 passed / 5 skipped**，没有发现 V16.8.32 新功能 regression。
