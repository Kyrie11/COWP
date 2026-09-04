# V16.8.44R1 工程可靠性修复：当前 V44 lost7 不可归因

## 0. 本轮最终结论

当前上传的 V16.8.44 `lost7` 结果**不能用于可靠算法归因**，因此按本轮约定：

- 不对 V44 下 GO / STOP；
- 不判断 RC-CRRS 是否值得 promotion；
- 不进一步收紧 dominant bottleneck；
- 不关闭新的算法族；
- 不设计新的科学算法版本；
- 不修改论文 method / abstract / introduction / contributions。

本轮只修复 correctness defect，并做不改变科学语义的 runtime work reuse。修复版本命名为 **V16.8.44R1**，科学 method ID 仍为 `cowp_root_conditioned_control_reachable_responder_support`，预注册 lost7 Gate 仍为 `>=2/7 newly rescued`。

---

## 1. 对论文主线的对齐理解

当前论文的核心问题不是 generic collision avoidance，而是 **false-safe / safety-by-coercion**：一个 ego plan 可以在几何上 collision-free，却只有在其他 road users hard brake、abrupt yield、priority abandonment 或 gap surrender 时才成立。论文将这种现象从 soft social cost 提升为 hard feasibility defect。

主 pipeline 是：

```text
natural alternatives / natural roots
→ ego-conditioned same-root responses (RCOT)
→ burden / option preservation / coercion witness
→ protected-priority hard non-coercive certificate
→ hard-first candidate selection
→ explicit uncertified fallback
```

从 V28→V43 的证据链，当前代码的 physical-recovery branch 又把问题扩展到：ego 自身不能通过让关键 interaction options 崩塌来获得所谓 recovery。上轮已形成的 CCF-A 级统一方向仍然是：

> **Orthogonal Option-Set Feasibility / Safety must not be obtained through critical option-set collapse.**

其中 social axis 保护其他 agent 的 low-burden natural option set；physical-interactive axis 要求 exact blocker 的 retained natural roots 存在 root-consistent、low-burden、control-reachable、environment-compatible、jointly realizable recourse。

本轮没有可靠新证据，因此不修改这条主线。

---

## 2. compact-5k benchmark 的独立性质复核

上传数据包本身给出的 split 为：train 5000 / val 1000 / heldout 1200；三者 ID 均唯一且两两零 overlap。heldout 是 official WOMD validation 的 held-out subset，因为 official WOMD test future hidden。

| 性质 | train | val | heldout |
|---|---:|---:|---:|
| scenes | 5000 | 1000 | 1200 |
| audit-relevant pair rate | 0.42970 | 0.42863 | 0.42949 |
| protected PRIO-root coverage | 0.99453 | 0.99363 | 0.99465 |
| rootless rate | 0 | 0 | 0 |
| `<2 low-burden roots` rate | 0 | 0 | 0 |
| mechanism unauditable rate | 0.04067 | 0.04344 | 0.04465 |
| critical-agent mean | 5.3846 | 5.3870 | 5.3383 |
| conflict-region selected-cap saturation | 0.9578 | 0.9550 | 0.9542 |

结论：跨 split 的关键 support / audit-relevance 统计非常稳定；当前没有证据支持“模型主要被 split drift、rootless、low-burden-root sparsity 卡住”，所以**当前不重构数据集**。

长期 watch items 仍有两个：

1. critical-agent observed cap=6，且 conflict-region selected-cap saturation 约 95.4–95.8%，说明后期可能存在 support ceiling；
2. `PRIORITY_SMOOTH_YIELD` proposal acceptance 约 20–22%，`TERMINAL` 约 54–55%，明显弱于 `ROBUST_BCTE` / `JOINT_ROUTE_NCF`，后期可作为 proposal-support ceiling 排查项。

另外 `verify_cache_train.json` 仍为 `pass=false`，唯一列出的原因是 58,243 个 `irrelevant pair blockers`。这不是当前 V44 online failure 的已证实解释，但投稿前必须修 cache serialization 或给出 cache→runtime semantic-equivalence 证明。

独立统计文件：`V16_8_24_COMPACT5K_DATASET_CHARACTERIZATION_INDEPENDENT_R1.json`。

---

## 3. 上传 V44 lost7 结果的结构可靠性

结构层面没有发现 manifest/shard/merge 错误：

- lost7 manifest = 7 unique IDs；
- shard = 4 + 3，零 overlap，并集精确等于 manifest；
- merged = 7 unique scenarios，精确等于 manifest；
- method = `cowp_root_conditioned_control_reachable_responder_support`；
- checkpoint path 指向 compact5k `cowp_all_best.pt`；
- merged standard metrics 可从 7 个 scenario row 逐项重算，最大误差 0。

原始 headline 观测是 `0/7 newly rescued`、CollisionRate=1.0。但**结构正确不等于算法归因正确**。

---

## 4. Blocking correctness defect：动态 profile 的 joint-cache aliasing

### 4.1 bug 本质

V44 的 control-reachable responder profile 是 candidate-conditioned：不同 ego hypothesis 会导致不同的 unsafe-event duration、不同的最小 residual acceleration，因此 response trajectory 会变化。

但是 V42/V43 遗留的 shared joint-CSP cache 仍按

```text
(agent_index, root_ordinal, profile_index)
```

识别 response profile。

V44 动态 completion 在不同 hypothesis 中会重用相同的 `profile_index_base + local_index`。于是：

```text
hypothesis A:
(agent 1, root 0, profile 10000) = trajectory A
→ pair-compatible = True
→ cache True

hypothesis B:
(agent 1, root 0, profile 10000) = trajectory B  # geometry changed
→ old key identical
→ stale True reused
```

这会产生 hard joint certificate 的 false positive 或 false negative。

### 4.2 为什么当前日志无法事后排除影响

当前 lost7 不是“机制没执行”：从 456 个 blocker-query policy steps 重构出：

- control-reachable completion attempts = **83,263**；
- dynamic profiles found = **26,485**；
- profile evaluations = **1,350,867**；
- selected roots with completion = **26,477**；
- joint-cache hits = **52,882**。

但 V44 日志没有标记每次 joint-cache hit 是 fixed-bank profile 还是 dynamic profile，因此无法证明 52,882 次命中中没有发生跨 hypothesis dynamic alias。

### 4.3 最小反例已经复现

我构造了两个 consecutive hypotheses，共享同一个 joint cache：

- hypothesis A：两个 responder trajectory 几何兼容 → PASS；
- hypothesis B：两个 responder trajectory 实际不兼容，但复用相同 dynamic `profile_index`。

原 V44：

```text
A = PASS
B = PASS            # 错误
B joint-cache hit=1
```

V44R1：

```text
A = PASS
B = no_jointly_compatible_response_envelope FAIL   # 正确
B stale joint-cache hit=0
```

因此这是一个真实 correctness bug，不是 style / diagnostic issue。

---

## 5. 为什么现在不能把 0/7 判成 V44 STOP

预注册条件是：

```text
lost7 newly rescued >= 2/7 -> continue
lost7 newly rescued <  2/7 -> STOP
```

但这个 Gate 的前提是实验实现忠实于注册的 hard certificate。当前 joint-CSP 可能读取错误的 compatibility result，所以 `0/7` 不能合法映射到 scientific STOP。

本轮科学状态只能是：

> **V44 = UNRESOLVED / attribution blocked by implementation correctness.**

旧 `lost7_failfast_gate.json` 必须 quarantine，不能进入 evidence chain、changelog family closure 或论文结论。

完整审计：`V16_8_44_RESULT_RELIABILITY_AUDIT_NONATTRIBUTABLE.json`。

---

## 6. V16.8.44R1 correctness repair

R1 不改任何科学约束，只修 identity：

- fixed bank profile：仍使用 frozen per-root `profile_index`；
- V44 dynamic profile：compatibility identity 改为 exact `(residual_accel_mps2, residual_duration_s)`，外层仍包含 immutable `(agent, root)`；
- environment compatibility cache 和 joint-CSP cache 都改用 semantic profile identity；
- selected-response diagnostics 额外记录 `control_reachable_extension`、residual acceleration、duration，便于下一轮追溯。

这些字段在固定 root + 固定 cfg 下确定 dynamic response geometry，因此不会把不同 hypothesis 的不同 response trajectory 混为一谈。

---

## 7. `lost7_parallel2` 为什么这么慢，以及本轮加速

上传结果 wall time = **30,368 s = 8.44 h / 7 scenes = 4,338 s/scene**。作为量级参考，V43R3 gate19 为 22,249 s / 19 = 1,171 s/scene，V44 每 scene 约慢 **3.70×**。

根因不是 Waymax 80-step 本身，而是 V44 responder completion 的 nested search：456 个 query-active policy steps 内执行 83k completion attempts 和 1.35M profile evaluations；每次 evaluation 原实现重复 trajectory build、burden、roadgraph、current/shift kinematics、ego checks 和 environment checks。

R1 只做 exact work reuse：

1. `(agent, root, duration, residual)` 相同的 trajectory / burden / roadgraph / responder current+shift kinematics 复用；
2. `(agent, root, environment actor)` 的 root↔frozen-environment unsafe-event support 复用；
3. dynamic responder↔frozen-environment compatibility 按修复后的 semantic profile identity 复用；
4. **ego-conditioned current/shift collision checks仍每个 hypothesis 重算**，没有为了速度放松 hard certificate。

新增 cache-hit diagnostics；R1 launcher 默认加 `--profile-policy-runtime`（不加 GPU sync），下一轮可以直接拿到 selection/runtime breakdown。

由于本地没有你的 WOMD/Waymax server 数据环境，本轮只声称“消除了可证明重复的计算”，**不提前声称端到端 speedup 倍数**。

---

## 8. 验证

当前 release 本地：

```text
V44 dedicated + cache-fidelity regressions: 5 passed
V16.8.25→V44R1 focused semantic/integrity sanity: 114 passed
Python compile: PASS
launcher bash syntax: PASS
conventional_check=False scan: PASS
```

其中包含：

- old-key stale joint-cache regression；
- static work reuse 不改变 logical profile-evaluation count；
- frozen-environment event support reuse；
- V42/V43R3 及 V25→V43 历史 focused regression。

---

## 9. 下一步唯一科学指令

不要运行 retained3 / induced9 / remaining29，也不要改 threshold、bank、root、selector、loss 或论文。

```bash
cd COWP_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR
bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh lost7_parallel2
bash NEXT_RUN_COMMANDS_V16_8_44R1_DYNAMIC_PROFILE_CACHE_FIDELITY_REPAIR_CN.sh analyze_lost7
```

把新的 R1 output 目录整体打包回传。下一轮才按原 preregistration：

- `<2/7`：可靠 STOP，再做 mechanism attribution / family closure / next scientific design；
- `>=2/7`：继续 retained3，然后才谈 promotion 证据。

---

## 10. 论文处理

本轮**不修改论文**。原因不是忽略论文/代码 mismatch，而是你已明确规定：如果当前实验不可靠，就只修代码错误。现在没有可靠 V44 scientific evidence，把 RC-CRRS 或 R1 runtime implementation 写进 Method/Abstract/Contributions 会把未经验证的工程修复误包装成论文贡献。

等 repaired lost7 通过可靠性审计后，再决定：

- RC-CRRS 是进入 Method 的核心 physical-interactive option-set component；还是
- 仅作为 Appendix / diagnostic analytic probe；或
- 若 STOP，则完全不进入主方法，只作为 ablation/falsification evidence。
