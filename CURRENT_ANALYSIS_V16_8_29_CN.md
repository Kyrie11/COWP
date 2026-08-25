# V16.8.28 结果审计与 V16.8.29 下一步设计

## 0. 结论先行

这一次 **V16.8.28 strict Waymax 结果可以用于算法归因**。没有发现类似 V16.8.26 的 conventional-safe 语义污染，也没有发现 V16.8.27 的 PAD execution 污染。parallel2 的 exact-200 证据链完整：四方法同一 200-ID set、同一 checkpoint、两个 100-scene shard 不重叠且完整覆盖、merged 指标能逐 scene 精确复算、physical attribution 可重现。

上传包中没有保留 split-mode 的最终 JSON。由于 parallel2 已完整成功，split 只是同一实验的备用执行方式，不是第二份必须的科学证据；以后 parallel2 成功后不要再跑 split。

本轮预注册分叉被明确收敛到：**common online physical-feasibility / proposal-action interface**。更具体地说，dominant actionable bottleneck 是 **full-horizon conventional support 在 receding-horizon closed-loop 中频繁塌缩**，随后 controller 从大量“动态有效但未通过完整物理 conventional audit”的候选里执行。

因此下一步不应再调 RCOT/BCOT、outcome 权重或 planner score，也不应重建数据。V16.8.29 只实现一个可证伪机制：**Receding-Horizon Recovery-Viability Bridge (RVB)**。

---

## 1. 论文主线与这一轮问题的关系

论文的真正核心不是 courtesy cost，而是把 false-safe planning 作为 feasibility defect：protected-priority road user 必须保留足够的 same-root low-burden safe response，BCOT/RCOT 用于结构化证据和 hard-first non-coercive certificate。主证书与 all-critical diagnostic 分开，proposal sufficiency 与 selector/certificate quality 也分开。

这意味着物理闭环问题不能用“把一个 physical outcome penalty 混进 COWP score”粗暴解决，否则会破坏论文最干净的贡献分解：

1. social/non-coercive feasibility；
2. physical execution feasibility；
3. ego utility/ranking。

V16.8.28 的 clean evidence 正好说明第二层在 closed-loop operating regime 下存在一个独立断层，所以这一轮应该补一个与主 social certificate 正交的 physical recovery-recourse object，而不是重写 COWP 主证书。

---

## 2. 数据集状态：本轮无需重建

formal_v16_8_24_compact_full_5k 的 train/val/heldout 规模为 5000/1000/1200。natural-support 结构在 split 间稳定：protected priority-root coverage 接近 99.4% 以上，rootless critical actors 为 0，低 burden roots 的可用性没有显示 split collapse。历史 cache verifier 的 irrelevant blocker 属于既知旧问题，而 affected/conflict/retained/root-weight 等核心 semantic mismatch 为 0。

这与历史 learned-offline 结果一致：RCOT/BCOT 当前不是主要的 representation-learning bottleneck。用户已明确不考虑重建数据，本轮也没有新证据推翻这一点。

长期的 **global ceiling** 仍是 fixed-bank proposal sufficiency：历史 held-out AnyNCF 约 36%，oracle false-safe floor 约 59.5%。但那是 false-safe 主任务的长期上限；当前 strict collision 的 actionable bottleneck 已经更具体地定位到 online physical support horizon，不应现在把两个问题混在一起。

---

## 3. V16.8.28 结果完整性

### 3.1 exact-ID / shard / merge

- exact manifest：200 unique IDs；逻辑 SHA256 = `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`。
- `cowp / cowp_fallback_outcome / conventional_safety / planner_score_only` 的 merged scene-ID set 与 manifest **精确一致**。
- 每方法 shard0/shard1 各 100 scenes，互不相交，union = exact 200。
- 四方法 checkpoint 相同。
- CR / Collision / Offroad / Kinematics / EP 均可从 200 per-scenario rows 精确复算。
- `waymax_200_physical_attribution.json` 可由当前 analyzer 重建。

注意：merged JSON 的 row order 与原 manifest order 不相同，因此不能用 merged row sequence 重算 manifest SHA。正确 integrity check 是 manifest 自身 hash + result ID set equality。交付的 `V16_8_28_RESULT_INTEGRITY_AUDIT.json` 已按此修正。

### 3.2 V16.8.28 execution repair 是否生效

生效。no-valid 时不再执行 zero PAD slot，而是明确记录：

- selected candidate = -1；
- selected candidate valid = false；
- `EMERGENCY_BOUNDED_STOP`；
- `execution_trajectory_source=bounded_smooth_stop`。

更重要的是，当前 collision 只有 2/34 紧邻 bounded emergency action，说明 v16.8.27 的 PAD execution 已不再是 collision dominant source。

---

## 4. 主结果与预注册判断

| Method | CR | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|---:|
| COWP | 19.5% | 17.0% | 3.0% | 12.5% | 1.0461 |
| COWP + fallback outcome | 18.5% | 16.5% | 2.5% | 13.0% | 1.0239 |
| Conventional safety | 24.0% | 22.0% | 2.0% | 8.5% | 0.7142 |
| Planner-score-only | 27.0% | 24.5% | 2.5% | 10.0% | 0.8644 |

### 4.1 不是 accepted COWP certificate collision

COWP 34 个 collision：

- 34/34 first collision 前一动作属于 fallback；
- 32/34 = `no_conventional_use_least_coercive_valid`；
- 2/34 = no-valid bounded emergency；
- 0/34 前一动作是 conventional-safe；
- 0/34 是 `accepted_priority_ncf`。

所以不支持“RCOT/BCOT certificate 接受了物理危险 candidate，导致 dominant collision”的假设。

### 4.2 也不是 conventional-safe recovery 排序失败

如果真正 conventional-safe fallback pool 中候选很多，但 selector 排错，应该看到 collision 紧邻 `selected_conventional_safe=true`。实际 COWP collision 是 0/34。

因此也不支持“只需要设计一个更好的 conventional recovery ranker”的假设。

### 4.3 common lower-layer interface 得到支持

Conventional-safety baseline 44 个 collision 中：

- 43/44 first collision 前是 `baseline_use_stop_like`；
- 0/44 前一 candidate conventional-safe；
- 仅 1/44 no-valid emergency。

Planner-score-only 49 个 collision 中：

- 47/49 是 accepted dynamic-valid baseline candidate；
- 0/49 前一 candidate conventional-safe；
- macro 极度集中在 MERGE_BEHIND（46/49）。

因此 collision 并非 COWP-specific；baseline 同样暴露出 candidate physical support / action interface 的问题。这正符合上一轮预注册的第三分支。

---

## 5. dominant bottleneck 为什么不是“proposal 数量少”

COWP online step 统计：

- fallback step rate = 71.675%；
- accepted priority-NCF ≈ 28.325%；
- no-certificate but conventional ≈ 15.981%；
- no-conventional but dynamic-valid ≈ 53.038%；
- no-valid ≈ 2.656%；
- zero full-horizon conventional candidate ≈ 55.694%；
- mean valid candidates ≈ 33.27；
- mean conventional candidates ≈ 6.64；
- mean accepted/certified ≈ 4.37。

也就是说，大多数失败状态不是“候选一个都没有”，而是：**有几十个 dynamic-valid proposal，但完整约 8 s primitive 没有一个通过 conventional physical screen。**

Waymax 只执行 0.1 s 后重新规划。当前实现却把一个 candidate 是否能进入 conventional pool，绑定到完整 primitive 对 causal-CV/logged future + roadgraph 的 8 s pass/fail。这使 physical feasibility 从一个应该具有 receding-horizon recourse 含义的对象，变成全时域二值过滤器。一旦全时域 set 空，controller 直接跳到 unrestricted dynamic-valid fallback。

这就是目前最有解释力的 structural gap：

> **full-horizon physical refusal -> no explicit short-horizon recoverability certificate -> unrestricted-valid execution**。

---

## 6. 各层当前状态

| Layer | 状态 | 处理建议 |
|---|---|---|
| Natural typed roots / protected-priority semantics | 证据稳定 | Freeze |
| RCOT same-root recovery transport | 强：历史 held-out low-safe AUPRC ~0.897 | Freeze |
| BCOT protected/global false-safe | 强：历史 ~0.837 / ~0.928 | Freeze |
| All-critical BCOT | 合理 diagnostic，不应 universal veto | Freeze as diagnostic |
| Main protected-priority hard certificate | 未被当前 collision 证伪 | Freeze |
| Certificate-compatible set-preservation frontier | CTU 负消融支持其价值 | Freeze |
| Generic flat candidate classifier | 弱 | 不升级 |
| Outcome head | clean strict probe 无显著安全收益、损失 EP | Archive negative |
| Planner score / utility | planner-only 安全和 EP 都更差 | 非 dominant bottleneck |
| Raw dynamic proposal count | ~33 valid/step | 非当前 bottleneck |
| Full-horizon conventional support | ~55.7% steps set 空 | **Dominant actionable bottleneck** |
| no-valid execution | v16.8.28 已修；collision 仅 2/34 紧邻 | Freeze repair |
| Accepted-path kinematic executability | 16/25 kinematics 紧邻 accepted priority-NCF | Secondary branch, later |
| Fixed-bank NCF proposal support | false-safe 主任务长期 ceiling | Later proposal work |

---

## 7. clean negative：不要再追 fallback outcome weight

COWP + fallback-outcome 相比 COWP：

- collision -0.5pp，McNemar p=1.0；
- CR -1.0pp，p≈0.774；
- kinematics +0.5pp；
- EP 平均 -0.02218，paired bootstrap 95% CI [-0.03896, -0.00726]。

这说明现有 outcome head 并没有解决当前 recovery support gap，而且以 progress 为代价。因为 v16.8.28 已修复两轮工程污染，这次可以把该方向正式归档为当前设计下的 negative probe，而不是再调 0.5/1.0/2.0 权重。

---

## 8. 为什么下一步不是简单 Recovery Certificate 或 Execution-Viability Certificate

上一轮设定三分叉：

1. conventional-safe fallback collision -> Recovery Certificate；
2. accepted COWP path collision -> Execution-Viability Certificate；
3. baselines 也高 collision -> common online proposal/action interface。

当前主 collision 属于 3；但 kinematics 中确有 16/25 紧邻 accepted priority-NCF，说明 2 作为**次级独立问题**存在。若现在同时加 recovery + accepted-path dynamics shield，会把两类 failure source 耦合，下一轮又无法知道哪个贡献有效。

因此 v16.8.29 只解决 collision dominant 的 branch 3；accepted-path kinematics 留到下一轮。如果 RVB 后 collision 明显下降但 kinematics 不变，正好获得干净证据支持后续的 orthogonal Execution-Viability Certificate。

---

## 9. V16.8.29：Receding-Horizon Recovery-Viability Bridge

### 9.1 机制

主 COWP 路径完全不变。只有当 full-horizon conventional set 为空、dynamic-valid set 非空时：

1. 对每个 dynamic-valid / non-conventional candidate 截取前 8 steps（默认 0.8 s）；
2. 从该 prefix endpoint 用已有 smooth-stop primitive 构造 bounded-stop continuation；
3. splice 成完整 horizon；
4. 使用**原有同一套** roadgraph drivable + causal logged/CV collision check 做硬筛选；
5. selection 时继续要求已有 hard action-risk / rule-risk shield；
6. bridge set 内使用原 fallback score；
7. bridge set 空才进入原 unrestricted-valid fallback。

新方法名：`cowp_recovery_bridge`。

### 9.2 它没有改变什么

- 不改 natural roots；
- 不改 RCOT / BCOT；
- 不改 protected set；
- 不改 certificate risk budget；
- 不改 main frontier；
- 不加 outcome head；
- 不加新 learned module；
- 不给 STOP/YIELD 宏天然 safe 标签；
- 不删除 original valid fallback，因此不会用 coverage 改变伪装效果。

### 9.3 论文级 novelty 应该如何理解

generic contingency / backup trajectory / recursive feasibility 本身已经有大量工作，因此不能把“我也拼一个 stop backup”写成 contribution。当前文献甚至直接在 receding-horizon 中维护共享 initial segment 后 branching 的 contingency trajectories，或联合优化 exploration/fallback trajectories，以及用 reachable-set barrier 保证 safety。

真正可能达到 CCF-A 机制标准的对象是 **dual feasibility**：

- `Non-Coercive Feasibility`：protected agents 保留 same-root low-burden options；
- `Recovery Viability`：当 full-horizon physical support 暂时空时，ego 只能执行仍保留显式 physical recourse 的短前缀。

即 planner 不再把“social feasibility / physical recoverability / utility”揉成一个 score，而是两个正交 hard-set + 后续 ordering。这个结构只有在 v16.8.29 strict paired evidence 有效后才值得升格为论文 contribution；现在仍应叫 preregistered mechanism probe。

---

## 10. 下一轮预注册判据

只比较同一 exact-200 的 `cowp` 与 `cowp_recovery_bridge`：

### A. Bridge available + used，collision 显著下降，EP 基本不损失

支持：closed-loop 主要问题确实是 full-horizon conventional support 与 short-horizon recourse 之间的 horizon mismatch。下一步可将 RVB 升级为主算法组件，并做 3 seeds / 更大 scene set / bridge availability-calibration / causal screen ablation。

### B. Bridge availability 很低

支持：当前 primitive family 或物理几何 support 本身缺少 recoverable prefix。下一步应做 proposal-support redesign，而不是调 fallback score；仍无需立刻重建 dataset labels。

### C. Bridge available/used，但 collision 不下降

支持：当前 causal CV/logged screen 或 candidate -> one-step action projection 与 Waymax physical evolution不一致。下一步做 projection/dynamics consistency mechanism，而不是再加 outcome/risk penalty。

### D. Collision 改善但 kinematics 仍由 accepted priority-NCF 主导

说明 dominant collision 与 secondary executability 成功解耦。下一版再单独设计 Execution-Viability Certificate，避免本轮混因。

---

## 11. 速度优化

v16.8.27 profiler 已显示 CPU candidate construction ≈87.8–88.6% policy time，model forward ≈5.8%，selection ≈4.2–4.8%。所以继续优化 GPU model forward 的收益很有限。

v16.8.29 在不改变 screen 数学定义的前提下，将每个 policy step 对所有候选重复的：

- lane-centerline mask；
- nearby agent priority/nearest ranking；
- logged/CV causal futures；
- collision radii；
- sample index；

提取成一次 cache，candidate 只做 candidate-dependent distance checks。

`tests/test_v16_8_29_recovery_viability.py` 对 cached/uncached candidate bank 做 bit-exact equality。新的本地 32-agent + lane-map microbenchmark（包含 cache build 自身）约为 0.282s -> 0.254s，约 9.9% candidate-build reduction。服务器端端到端收益必须由 `profile_parallel2` 实测，不承诺固定比例。

两张 A30 的正确用法仍是 scenario-level 2-process sharding：process0 Torch+JAX co-locate A30-0，process1 co-locate A30-1。场景独立，结果按 exact scenario ID merge。下一轮只跑两方法，不再跑四方法，因此相比 v16.8.28 的四方法 sweep，计算量本身也约减半。

---

## 12. 回归与工程状态

- v16.8.29 `sanity`: **20/20 passed**。
- 全仓：**270 passed / 5 skipped / 8 historical failures**。
- 8 failures 与 v16.8.28 相同：6 个压缩包缺失的旧 launcher；2 个旧 semantic fingerprint hard-code。
- 没有新功能 regression。

因此没有工程 blocker 需要把 V16.8.28 或 v16.8.29 probe 作废。

---

## 13. 下一步指令

```bash
cd COWP_v16_8_29_RECOVERY_VIABILITY

bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh make_ids

# 仅当旧 TFExample index 不存在时：
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh build_tfindex

# 可选，先测服务器实际速度：
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh profile_parallel2

# 推荐的唯一主实验：两张 A30，same exact-200，只跑 COWP + RVB
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh waymax_recovery200_parallel2
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh analyze_parallel2
```

如果、且仅如果，两进程 co-location OOM，再改为：

```bash
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh waymax_recovery200_split
bash NEXT_RUN_COMMANDS_V16_8_29_RECOVERY_VIABILITY_CN.sh analyze_split
```

**不要 parallel2 和 split 都跑。**

下一轮不要：retrain、dataset/cache rebuild、BCOT budget sweep、outcome-weight sweep、CTU、PCHR、universal all-critical hard veto、MCFC promotion、accepted-path execution shield。先让 RVB 这个单因素 probe 回答 dominant collision bottleneck。
