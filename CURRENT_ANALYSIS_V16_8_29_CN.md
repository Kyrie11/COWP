# V16.8.28 结果审计与 V16.8.29 下一步设计

## 1. 结论先行

这次和 v16.8.26 / v16.8.27 不同：**上传的 v16.8.28 parallel2 exact-200 结果已经足够干净，可以做当前层级的算法归因。** 我没有发现需要再次把整轮 strict Waymax 作废的工程阻断。

需要保留两个边界：

1. 上传结果包里没有 `waymax_diag200_split` 的单进程 split 输出，因此无法对 parallel2 与 split 做独立数值复现；
2. 包里也没有新 runtime profile JSON，所以速度判断继续以 v16.8.27 已得到的 CPU candidate-build 热点和本轮代码审计为基础，新版本提供 12-scene profile 做服务器实测。

这两个缺口不妨碍当前 parallel2 结果内部的一致性和 first-event attribution。

## 2. 结果完整性

四个方法均满足：

- 200 个唯一 exact IDs；
- manifest logical SHA256 = `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`；
- shard0/shard1 各 100，互斥且并集精确等于 manifest；
- merged summary 可由 200 scenario rows 对 CR / Collision / Offroad / Kinematics / EP 逐项精确复算，最大误差 0；
- 四方法 EP 都在同样 196 个 finite scenes 上比较；
- v16.8.28 emergency repair 的语义一致：每个场景的 `emergency_action_step_rate`、`zero_valid_candidate_step_rate`、`no_valid_step_rate` 完全一致。

因此这轮没有再次出现“统计是真的，但 controller 语义错了”的情况。

## 3. 主结果

| Method | CR | Collision | Offroad | Kin infeasible | EP | Fallback step |
|---|---:|---:|---:|---:|---:|---:|
| COWP | 0.195 | 0.170 | 0.030 | 0.125 | 1.0461 | 0.7168 |
| COWP + fallback outcome | 0.185 | 0.165 | 0.025 | 0.130 | 1.0239 | 0.7229 |
| Conventional safety | 0.240 | 0.220 | 0.020 | 0.085 | 0.7142 | 0.5907 |
| Planner-score-only | 0.270 | 0.245 | 0.025 | 0.100 | 0.8644 | 0.0213 |

### COWP vs planner-score-only

这是本轮最强的正证据：

- CR: `0.195 vs 0.270`, paired McNemar `p≈0.00813`;
- collision: `0.170 vs 0.245`, `p≈0.00592`;
- paired EP: planner - COWP = `-0.18166`, bootstrap 95% CI `[-0.26516,-0.11414]`.

因此不能把当前问题表述成“COWP certificate/selector 整体导致 physical failure”。恰恰相反，在同一 candidate/checkpoint/execution interface 下，COWP 相比裸 planner-score 显著改善 collision，同时进度也更高。

### COWP vs conventional safety

- collision: `0.170 vs 0.220`, conventional - COWP = +0.05，McNemar `p≈0.0755`；
- EP: conventional - COWP = `-0.33188`, bootstrap 95% CI `[-0.44294,-0.23644]`.

200 场景下 collision 差异只是趋势，不能写成显著安全优势，但 COWP 的 progress-efficiency 优势非常明确。

### fallback outcome：现在可以正式作为 clean negative

修掉前两轮的 conventional/PAD 污染后：

- collision 只从 `0.170→0.165`, `p=1.0`；
- CR `0.195→0.185`, `p≈0.774`；
- kinematics 反而 `0.125→0.130`；
- paired EP 下降 `-0.02218`, 95% CI `[-0.03896,-0.00726]`。

因此 outcome head 可以继续做 diagnostic，但不应该进入下一版 hard physical certificate，也不值得继续调 fallback outcome weight。

## 4. 按上一轮预注册顺序判定 physical bottleneck

上一轮要求 clean rerun 后在三条路中选择：

A. `fallback && conventional_safe` → Recovery Certificate；

B. `accepted_priority_ncf` → Execution-Viability Certificate；

C. conventional/planner 也暴露同类 failure → common online proposal/action interface。

### Collision 明确选择 C，但可进一步收紧

COWP 34 个 collision：

- 32 个 first collision 前一步 = `no_conventional_use_least_coercive_valid`；
- 2 个 = `no_valid_candidate` 的 bounded smooth stop；
- **0 个 conventional-safe fallback**；
- **0 个 accepted priority-NCF path**。

Conventional baseline 44 个 collision：

- 43 个发生在 conventional pool 已空后的 `baseline_use_stop_like`；
- 1 个 no-valid emergency；
- first collision 前同样 **0 个 conventional-safe action**。

因此当前 collision 的 dominant actionable bottleneck 不是“certificate 太严”，不是“BCOT 排错了”，也不是“conventional-safe recovery score 不好”，而是：

> **闭环进入大量没有任何 full-horizon conventional-safe proposal 的状态；之后 planner 被迫在已经失败 conventional audit 的候选中执行 uncertified action。**

这就是 **online conventional-feasible support collapse**。

COWP collision episode 的 zero-conventional step rate 平均 `0.80993`，非 collision episode 为 `0.50512`；34/34 collision episode 在 first collision 前都已经经历 zero-conventional。zero-conventional exposure 在四种方法之间相关系数约 `0.87–0.97`，说明它是共同 candidate/screen operating-regime 问题，而非 COWP-specific artifact。

### 但“proposal support 不足”还不能直接作为唯一解释

当前 `conventional_safe` = roadgraph surrogate ∩ 8 秒 causal constant-velocity collision screen，而控制器每 0.1 秒 replanning。

所以 zero-conventional 可能由四种机制产生：

1. 真正没有有用 proposal；
2. 8 秒 CV open-loop screen 对 receding-horizon control 过保守；
3. roadgraph centerline-distance surrogate 过保守/不准确；
4. roadgraph 与 collision 各自有 survivor，但没有同一 candidate 同时通过。

若现在直接扩 proposal 或直接把 8 秒改成 2 秒，都无法区分根因，且后者本质只是调 horizon。

## 5. 模型/算法各层状态

### 可以冻结

**Natural roots / natural basis**：历史机制门控已过，当前 physical failure 不在这里。

**RCOT**：held-out Root LowSafeExist AUPRC 约 0.897，是当前最强机制信号之一。

**BCOT**：priority/global false-safe AUPRC 约 0.837/0.928，明显强于 generic candidate classifier。

**Protected-priority hard feasibility**：仍比 universal hard veto 更符合已得到证据和论文 claim。

**Certificate-compatible set-preservation frontier**：CTU 已作为 clean negative；不要退回 certificate → planner-score argmin。

**Outcome head**：冻结为 diagnostic-only。clean strict negative 已经足够，不再花迭代调权重。

### 暂时不能冻结

**Online conventional feasibility representation**：这是当前最大不确定层，需要拆 collision/roadgraph/support。

**No-conventional uncertified recovery**：collision 几乎全部发生在这里，是当前第一改动对象。

**Proposal bank**：长期 fixed-bank ceiling 仍存在，但这轮先不扩 bank；必须先证明 zero-conventional 是 true support shortage 还是 screen/receding-horizon mismatch。

**Action projection / physical execution viability**：是 secondary bottleneck。COWP 25 个 kinematics first event 中 16 个来自 accepted_priority_ncf，17/25 前一步是 conventional-safe。这说明 collision 修完后还需要单独处理 accepted-path kinematic viability。

## 6. 当前 dominant bottleneck

分两层：

- **长期 global ceiling**：fixed-bank proposal support（历史 AnyNCF ≈ 36%，oracle false-safe floor ≈ 59.5%）；
- **当前 actionable dominant bottleneck**：online conventional-feasible support collapse，尤其 collision-side。

当前不要再追 RCOT/BCOT AUPRC 的小幅提升，也不要进入 outcome weight、BCOT budget、planner repair。

## 7. V16.8.29：Recursive Viability Recovery（RVR）

我实现的新方法 `cowp_recursive_viability` 严格保护已验证主线。

### 不变部分

只要存在 certificate candidate：完全等同 COWP。

certificate 空但存在 conventional-safe candidate：完全等同 COWP 的 conventional fallback。

只有 full conventional pool 为空才改变。

### 新 recovery

对每个 dynamically valid candidate，在完全相同的 causal collision screen 下记录 first violation 的 prefix length `h_k`。

选择顺序：

1. 若存在 roadgraph-safe valid candidates，先保留它们；
2. 在保留池中只留下 `h_k` 最大者；
3. 原 COWP fallback composite score 只在并列者之间排序。

它不降低 conventional threshold、不缩 horizon、不添加 weight、不训练新 head，也不把这些 candidate 改称 safe/NCF。

### 为什么这轮先做它

它直接检验一个关键假设：

> 当前 binary 8 秒 conventional test 在 0.1 秒 receding-horizon planner 中是否丢失了“谁能让系统保持更久可恢复”的 ordering information？

如果 RVR 能显著减少 collision，同时 selected safe-prefix 明显上升，那么下一步才值得把它升级为更正规的 **physical recursive viability** 层；如果无效，就说明 maximal temporal slack 也救不了，应该转向 proposal support / route-topology / roadgraph/action interface。

## 8. CCF-A novelty 边界

`max safe prefix` 本身不能作为论文 contribution。recursive feasibility、MPC safety filter、reachable/backup set 都是成熟方向。

如果这个 probe 成功，真正值得写成方法贡献的是：

> **Orthogonal Dual Feasibility**：把 COWP 的 protected-priority social non-coercive feasibility 与 physical recursive viability 正交组合，而不是把两种风险塞进一个 scalar cost。

可形成：

```text
natural roots
→ same-root RCOT
→ protected-priority BCOT social certificate
→ certificate-compatible robustness frontier
→ causal recursive physical viability / recoverability
→ explicit uncertified emergency
```

最终论文版应给出明确的 recursive viability/recovery condition 和保证/校准边界，并证明它不会通过放松 social certificate 换物理指标。

## 9. 为什么不应该每轮再跑 4×200

### Development gate

新代码带一个 outcome-enriched **dev64**：

- 34 个当前全部 COWP collision scenes；
- 30 个 zero-conventional exposure 最高的 non-collision scenes。

只跑 COWP + RVR：`2×64=128` scene-method rollouts，相对之前 `4×200=800` 少 **6.25 倍**。

这个集合因为使用了 v16.8.28 outcome 选择，**严禁用于论文 claim**。它的作用只是快速 falsify 新机制。

`analyze_diag64` 会先自动检查新代码中的普通 COWP 是否逐场景复现随包保存的 v16.8.28 dev64 reference。只要 speed refactor 改了旧 COWP 行为，就会直接 fail，不继续解释 RVR。

### Confirmation gate

dev64 有正信号后，再跑 exact200 的 COWP + RVR 两个方法：400 scene-method rollouts，仍比旧四方法整组少 2 倍。

不要每次重复 `fallback_outcome/conventional/planner`；它们在当前代码未触及的部分可以作为已冻结参照。算法最终锁定时再做完整 publication protocol。

### 最终论文实验纪律

这一组 exact-200 已经在多轮迭代中用于设计选择，后续应视为 development strict set。算法真正 freeze 后，需要从未参与调算法的场景建立新的 final evaluation set，并按论文自己的协议跑 ≥3 seeds、paired scenario CI。这个动作不等于重建训练数据。

## 10. Waymax 加速

历史 profiler 已经定位 CPU online candidate construction 占 policy 时间约 88%，model forward 约 6%，selection 约 4–5%。

v16.8.29 对 collision audit 做两层严格等价优化：

1. nearby-agent ranking + constant-velocity future 每 policy step 只构造一次；
2. 24 个 nearby agents 的距离检查改成 NumPy broadcast，去掉每 candidate 的 Python agent loop。

随机状态 regression 用 literal v16.8.28 reference 对照 conventional collision boolean；当前 focused sanity 20/20 passed。

本地 synthetic 64-agent/48-candidate microbenchmark 中，collision-audit 子组件约 `7.3x` faster。它不能替代 A30 服务器 wall-time profile，因此新 launcher 保留 `profile_parallel2`。

## 11. 下一步命令

```bash
cd COWP_v16_8_29_RECURSIVE_VIABILITY

bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh make_ids

# 仅脚本提示 index 缺失时执行
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh build_tfindex

# 推荐先跑：快速机制 gate
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh viability_diag64_parallel2
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh analyze_diag64
```

把 `viability_dev64_cowp_base_equivalence.json`、`viability_dev64_physical_compare.json`、`viability_dev64_mechanism_summary.json` 上传回来即可先决定是否值得跑 200。

只有 dev64 支持机制时：

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh confirm200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh analyze_confirm200
```

需要量服务器加速效果时单独：

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh profile_parallel2
```

如果 parallel2 co-location OOM 或实测变慢：

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh confirm200_split
bash NEXT_RUN_COMMANDS_V16_8_29_RECURSIVE_VIABILITY_CN.sh analyze_confirm200_split
```
