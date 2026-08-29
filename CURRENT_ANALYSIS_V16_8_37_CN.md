# V16.8.36 结果可靠性、机制归因与 V16.8.37 设计

## 0. 执行结论

本轮按实际实验链 **V16.8.36 → V16.8.37** 推进。结论分为五层。

1. **V16.8.36 结果可靠，可进入算法归因。** 独立审计完成 71/71 项 hard checks：manifest/hash、24+24 shard、merged-row 等价、标准指标与 fallback 统计零误差重算、checkpoint/协议一致、16-scene common-path 1120 字段零 mismatch、V28 no-valid 执行不变量、analyzer 独立复现均通过；未发现 online future-GT leakage。
2. **V16.8.36 严格按上一轮预注册 Gate 判 FAIL。** 它只满足 3/6 条：保留 5/10 old-RVR rescues、EP 与 intervention 通过；但仅避免 3/9 old-RVR induced、相对 COWP 是 6 rescue / 6 induced（净 0），Kinematics +2。不得 promotion，也不得运行 fresh37。
3. **V36 的 semantic frontier 确实被大量使用，但没有形成 returnability。** 62.85% recovery switches 选择了 non-historical-RVR representative；因此失败不是“frontier 没生效”。V36 相对 V35 CPOSH 为 1 rescue / 8 induced，McNemar exact p≈0.0391，说明 broader semantic support 反而破坏了已存在的 high-value signal。
4. **dominant bottleneck 收紧为：Returnability to Full-Conventional Physical Feasibility under Uncertified Recovery。** harmful scenes 并不缺 valid candidates，而是长期无法回到 unchanged full-conventional feasible region。option richness、prefix、spectrum 与 weak-dominance continuation 都不能替代“返回可行域”的证据。
5. **V16.8.37 采用 Recourse Returnability Bridge（RRB）。** 它不是继续调 ROSH/EOSH/CPOSH/frontier，而是验证一个 recovery action 是否：先安全跨过当前 replanning edge；随后直接恢复 conventional support，或保留一个在下一次真实 replanning 中可执行、可见证、且能恢复 full-conventional support 的 recourse action。bridge 只允许一条真实 replanning edge，不使用 dwell/epsilon/权重，也不读取 logged future。

重要 provenance 说明：上传的 `COWP.zip` 虽被描述为 V16.8.36，但压缩包内已混入一份 V16.8.37 草案。本文没有把草案结论当答案，而是先独立审计 V36 结果，再逐行审计并修订草案。最终交付的是 **audited V16.8.37**，不是原草案的原样重打包。

---

# 1. 对论文、研究对象和证据协议的理解

## 1.1 论文主问题

论文的核心不是“给 planner 增加 social cost”，而是识别一种传统 ego-centric safety 指标会漏掉的失败：

> ego trajectory 本身 collision-free，但只有在另一交通参与者 hard braking、abrupt yielding、priority abandonment 或 gap surrender 后才成立。

论文把这种 **false-safe planning / safety-by-coercion** 定义为 feasibility defect，而不是礼貌程度差异。有限 soft burden weight 无法保证剔除 coercive plan；hard non-coercive feasibility 才能让 ego utility 无法覆盖 feasibility violation。

## 1.2 COWP 的算法对象

当前成熟主线为：

1. burden-oriented interaction relation；
2. observational / ego-neutral / priority-preserving natural alternatives；
3. stable natural roots 与 retained natural mass；
4. same-root RCOT，搜索每个 conflicting root 的最低 burden safe response；
5. BCOT 对 root-level deficit、burden tail 与 OPR shortfall 做单调、预算化压缩；
6. protected-priority hard certificate；
7. certificate-compatible set-preservation frontier；
8. hard-first selection；certificate 为空时进入显式 uncertified fallback，而不是把 fallback 伪称为 NCF。

论文的主要理论结构是：

- **Social feasibility**：ego 不能通过压垮 protected road users 的 natural low-burden option set 获得所谓安全；
- **Proposal sufficiency 与 certificate/selector quality 分开审计**：fixed bank 没有 NCF proposal 时，threshold/ranking/fallback 无法消掉 bank-dependent false-safe floor。

## 1.3 数据集性质

`formal_v16_8_24_compact_full_5k` 的静态审计显示：

| split | scenarios | audit-relevant pair rate | protected PRIO root coverage | rootless | <2 low-burden roots |
|---|---:|---:|---:|---:|---:|
| train | 5000 | 0.42970 | 0.99453 | 0 | 0 |
| val | 1000 | 0.42863 | 0.99363 | 0 | 0 |
| heldout | 1200 | 0.42949 | 0.99465 | 0 | 0 |

三 split 的机制相关比例和 protected-root coverage 接近，没有支持“当前应重建数据”的新证据。历史 train-cache verifier 的 `pass=false` 仅来自 `irrelevant pair blockers=58243`；未观察到 missing keys、read errors、scenario-ID mismatch 或 root-contract mismatch。本轮遵守用户约束，不重建数据。

## 1.4 当前结果能论证什么、不能论证什么

当前 counterfactual48 是多轮机制选择中反复使用的 development panel；exact200 也已经参与历史算法选择。因此：

- 可以做 **mechanism falsification / development attribution**；
- 不能作为 final publication holdout；
- Waymax logged replay 能支持 closed-loop collision/offroad/kinematics/progress 证据；
- 不能把 learned burden quantity 当成真实 counterfactual burden GT；
- 最终 CCF-A claim 仍需：algorithm freeze 后的全新 final scene set、至少三 independent seeds、paired scene-level CI，以及 reactive-agent + held-out human-audited false-safe stress protocol。

---

# 2. V16.8.36 可靠性审计

## 2.1 审计结果

独立审计文件：

- `V16_8_36_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT_INDEPENDENT.json`
- `V16_8_36_ANALYZER_RECOMPUTED_INDEPENDENT.json`

总计 **71/71 hard checks passed**。

## 2.2 Manifest / shard / provenance

- counterfactual48：48 unique IDs；logical SHA256=`ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0`；两个 shard 各 24，0 overlap，union 精确等于 manifest。
- equivalence16：16 unique IDs；logical SHA256=`81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760`；两个 shard 各 8，0 overlap，union 精确等于 manifest。
- 两个 V36 shard 的 method、checkpoint、logged-replay protocol 与 online-GT flag 一致。
- merged scenario rows 与 shard rows 逐行一致。

## 2.3 指标重算与 analyzer 复现

从 scenario rows 独立重算：

- CR
- CollisionRate
- OffroadRate
- KinematicsInfeasibilityRate
- EP
- fallback rate

与 merged summary 最大误差为 0（fallback 仅有浮点表示末位差）。上传 analyzer 用相同冻结 references 独立重跑后，递归比较 **0 mismatch，tolerance=1e-12**。

## 2.4 Common-path 与 execution integrity

- equivalence16：16 scenes / 1120 fields / 0 mismatch。
- `emergency_action_step_rate == zero_valid_candidate_step_rate == no_valid_step_rate` 在 48 scenes 中保持成立。
- `mechanism_ground_truth_available_online=false`；内部 successor/returnability counterfactual 使用当前 simulator state、actual emitted action 与 frozen causal CV，未读取 Waymax logged future 来选 action。

因此没有出现 V26 conventional bypass、V27 conventional-integrity 或 V28 zero-PAD execution 级别的 attribution blocker。

---

# 3. 严格执行上一轮预注册 Gate

上一轮 Stage-1 六项 Gate 原封不动：

| 条件 | GO | V36 | 判定 |
|---|---:|---:|---|
| old RVR rescues retained | ≥5/10 | 5/10 | PASS |
| old RVR induced avoided | ≥7/9 | 3/9 | **FAIL** |
| COWP collision net removed | ≥3 | 0 | **FAIL** |
| Kinematics net regression | ≤1 scene | +2 | **FAIL** |
| paired mean EP Δ | ≥−0.05 | −0.02443 | PASS |
| intervention | >0 | 20.57% steps | PASS |

结论：

> **`cowp_control_projected_recovery_frontier` = Gate FAIL / Archive。**

不得运行 V36 fresh37；不得把 induced Gate 从 7/9 改成 3/9、把 kinematics bound 从 +1 改成 +2，也不得只凭 EP 改善或 non-RVR usage 宣布成功。

---

# 4. V36 closed-loop 结果与 paired attribution

## 4.1 Headline

| Method | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 1/48 | 6/48 | 1.002512 |
| V35 CPOSH | **27/48** | **0/48** | 7/48 | 0.913987 |
| V36 Frontier | 34/48 | **0/48** | 8/48 | 0.978086 |

V36 vs COWP：

- collision: 6 rescued / 6 induced / 28 shared failure；net 0；McNemar p=1.0；
- offroad: 1 rescued / 0 induced；
- kinematics: 0 rescued / 2 induced；
- paired EP Δ≈−0.02443，CI 跨 0。

V36 vs V35 CPOSH：

- collision: 1 rescued / 8 induced；net +7 failures；McNemar exact p≈0.0391；
- EP +0.0641 relative V35，但无法补偿 collision regression。

## 4.2 V36 不是“没用到 frontier”

- recovery switch step rate≈20.57%；
- frontier probe rate≈72.14%；
- switch 中 non-historical-RVR≈62.85%；
- mean semantic representatives≈8.86；
- mean profiles evaluated≈5.66；
- strict-admissible reps≈0.58；weak-admissible reps≈6.42。

所以 V36 已经实质性扩展了 same-bank support utilization。它的失败直接否定：

> “只要把更多 semantic branches 暴露给 control-projected spectrum，就能解决 recovery。”

## 4.3 rescued 与 induced 的真正差异

| diagnostic | V36 rescued (n=6) | V36 induced (n=6) |
|---|---:|---:|
| zero-conventional exposure | 67.5% | **92.5%** |
| mean conventional candidates | 3.77 | **0.79** |
| mean valid candidates | 35.23 | **37.19** |
| non-RVR switch rate | 77.4% | 75.0% |
| selected prefix delta | +5.20 | +5.36 |
| recovery active | 45.6% | **80.4%** |
| recovery continue | 40.6% | **76.7%** |

关键 separating signal 不是 valid support、non-RVR usage 或 prefix gain；它们在两组中接近。真正不同的是：induced scenes 长期回不到 full-conventional region。

## 4.4 first-collision 证据

六个 V36-induced IDs：

- `fe51445d725b8b8b`
- `3919ccd73c0fabd7`
- `d632f1919fe4bab`
- `c34fe8e79cdf1161`
- `f8d4c735825e5d81`
- `6418b0c9e2e4b093`

first collision 前一步全部为：

- fallback reason=`no_conventional_use_control_projected_recovery_frontier`；
- selected candidate valid=true；
- conventional=false；
- collision-safe=false；
- selected prefix=0；
- zero-conventional reason=`collision_empty`。

所以失败不是“已恢复 conventional 后 accepted path 又出事”；系统根本没有退出 uncertified basin。

两个最强 mode-trap 反例：

- `d632...`：zero-conventional 100%，active 100%，continue 98.75%，first collision step 37；
- `f8d4...`：zero-conventional 100%，active 100%，continue 98.75%，first collision step 19。

一次 rare strict entry 可以产生近乎全 episode 的 weak continuation，却没有明确 terminal target。

---

# 5. 机制成败、层级成熟度与 promotion 决策

## 5.1 真正成功、应继续保护的机制

| Layer | 证据/状态 | V37 原则 |
|---|---|---|
| compact-5k data/label contract | split 性质稳定；无重建证据 | Freeze |
| Natural roots | 历史 coverage/quality 稳定 | Freeze |
| RCOT same-root transport | held-out LowSafeExist 强于 generic classifier | Freeze |
| BCOT | protected/global false-safe 证据较强 | Freeze |
| protected-priority hard certificate | 符合论文 feasibility semantics | Freeze |
| post-certificate set preservation | CTU replacement 已负消融 | Freeze |
| 8 s conventional contract | 当前归因基准 | Freeze |
| V27/V28 integrity fixes | 已解决 | Freeze |
| outcome head | diagnostic-only；fallback weight 已负消融 | Freeze |
| SOV successor signal | 高精度过滤 harmful RVR | 保留 insight |
| semantic option spectrum | 比 max-prefix 更有信息 | 保留 representation insight |
| control-projected spectrum | V35 有明显 collision signal | 作为 V37 entry pre-gate |

## 5.2 失败/归档机制

- RVR policy：10 rescue / 9 induced；不能 promotion。
- Pareto guard：0 switch；不能调 tolerance。
- BHOV/THOP：one-step relaxation 与 horizon stacking 失败。
- unconditional commitment：over-commit。
- SDH：mode consistency 单独不足。
- V33 ROSH implementation：Gate fail；representation insight 保留。
- V34 EOSH/TG-ROSH：nominal exact-realizability 错误。
- WK-ROSH：current guard 无新增选择力。
- V35 CPOSH implementation：Gate fail；control-projected observable 保留。
- V36 semantic frontier：Gate fail，且显著伤害 V35 collision；archive。

## 5.3 secondary bottleneck

Accepted-path kinematics 仍是真问题：历史 clean COWP 的 25 个 kinematics episodes 中，16 个 first event 来自 `accepted_priority_ncf`，17/25 前一步甚至 conventional-safe。它必须在 recovery 稳定后单独设计 Execution-Viability Certificate；V37 不同时修改，避免破坏 collision recovery attribution。

---

# 6. dominant bottleneck 与模型下一步应学习的对象

## 6.1 bottleneck 递进

- V28：online conventional support collapse；
- V29：主要是 dynamic collision-side，而非 roadgraph；
- V30：successor option set 有独立信息；
- V31：stateless hybrid switching 会制造 failure；
- V32：horizon stacking / unconditional commitment 失败；
- V33：semantic spectrum 有效但不等于 execution；
- V34：nominal exact realizability 错误；
- V35：control-projected future option set 有信息；
- V36：same-bank semantic frontier 实际被利用，但 option richness 仍无法保证退出 uncertified basin。

当前 P0：

## **Returnability to Full-Conventional Physical Feasibility under Uncertified Recovery**

论文层更通用的名字：

## **Control-Reachable Recourse-to-Feasibility**

## 6.2 模型缺的不是“更多 options”，而是 terminal semantics

当前 physical modules 能近似回答：

> 当前 action 后还剩多少语义/控制可实现 options？

但不能回答：

> 这些 options 中是否存在一条在真实 replanning 后可执行、且能回到 unchanged full-conventional feasible set 的路径？

因此下一步应学习/验证的对象不是 `max prefix`、profile area 或 macro count，而是：

\[
(s_t,a_t^{\mathrm{executed}})
\rightarrow
\exists a_{t+1}^{\mathrm{replan}}
:\;\mathcal K_{\mathrm{conv}}(s_{t+2})\neq\emptyset.
\]

V37 只验证这个最小 causal witness，不训练 neural returnability head。

---

# 7. V16.8.37：Recourse Returnability Bridge（RRB）

方法名：

`cowp_recourse_returnability_bridge`

只在：

```text
full conventional set == empty
AND valid candidate exists
```

时介入。dataset、model、proposal families、certificate、8 s conventional definition 与 common controller 全部不变。

## 7.1 冻结的 high-precision entry pre-gate

V36 broader frontier 已被否定，所以 V37 回到受控 pair：

- `base`：ordinary COWP least-coercive-valid fallback；
- `alt`：historical global max-prefix RVR candidate。

只有同时满足：

1. emitted action 不同；
2. `H0_alt >= H0_base`；
3. **`H0_alt >= 1`，至少安全跨过下一次真实 replanning edge**；
4. alt 在冻结的 V35 control-projected spectrum + Waymax-aligned current transition relation上严格支配 base；

才执行 returnability probe。

第三条是本轮修订新增的 hard invariant：如果 action 在下一次 policy call 前就 collision-unsafe，任何“未来可返回”证据都无意义。

## 7.2 Direct restoration witness

对 base/alt 的 actual emitted action：

1. ego 用 emitted target 前进一步；
2. surrounding agents 用与 conventional audit 同源的 causal CV 前进一步；
3. 在 successor 上重新生成 unchanged online bank；
4. 若 `valid ∩ conventional` 非空，则 `direct_restore=true`。

不使用 Waymax logged future。

## 7.3 One-new-replan recourse witness

若 successor 仍无 conventional candidate：

1. 在 successor 生成一组**新的** replanning candidates；
2. 保留 valid、roadgraph-safe、positive-prefix、Waymax-current-transition-feasible、non-PAD candidates；
3. 按 `(semantic macro, actual emitted target)` 去重；
4. **评估每个不同 emitted-action class，而不是每个 macro 只看一条 max-prefix representative**；
5. carry 当前 emitted longitudinal acceleration，执行该新动作；
6. 在第二个 causal state 重新生成 unchanged bank；
7. 某 macro 只要有至少一个 action class 使 `K_conv` 非空，就加入 witnessed recourse set `R(a)`。

这是 existence witness。原草案“一 macro 一 representative”的做法会在同一 macro 内漏掉可返回 action，已修复。

## 7.4 Returnability partial order

不使用 count/AUC/discount/risk weight。

- alt direct，base non-direct → strict improve；
- base direct，alt non-direct → reject；
- 两者 direct → returnability tie，不能凭该项 entry；
- 两者 non-direct → 必须 `R_base ⊂ R_alt`；
- sets incomparable → reject。

因此相同 macro 数量但语义不同的 recourse sets 不会被 scalar count 误排序。

## 7.5 Witness-bound one-replan bridge

若 alt 依靠 non-direct recourse set 获胜：

- entry 时保存 `bridge_pending=true` 和**被见证的 semantic macro set**；
- 下一次实际 policy step 若 conventional/certificate 已恢复，直接 clear；
- 否则，从实际 observed state 重新生成 bank；
- 只在 entry 时见证过的 macros 内搜索；
- 必须 current prefix 不低于当拍 ordinary COWP base，且 >0；
- 对每个 distinct emitted-action class 验证 direct restoration；
- 在 hard-valid restoring set 内用 frozen COWP fallback score 选最小者；
- bridge 最多执行这一拍，然后无条件 clear；
- 没有真实 restoring action则 abort 并立刻用 ordinary COWP base。

它不是 fixed dwell，也不是 weak continuation；没有 `entry → indefinite continue` 状态。

## 7.6 现实性与无泄漏

V37 只使用：

- 当前 simulator state；
- 当前在线 candidate bank；
- actual projected/emitted action；
- carried controller acceleration memory；
- frozen map/roadgraph；
- 与 conventional audit 同源的 causal CV surrounding-agent model。

它不读取 future Waymax logged states；第二条 action 是 successor 上重新规划出来的 action，而不是偷看原 candidate 的第二 waypoint。因此它是 model-relative causal recourse probe，不是 oracle reachability claim。

## 7.7 V37 的论文地位

“一步 backup”“返回 safe set”“有限 recourse”本身不足以构成 CCF-A novelty。V37 仍是 physical-axis probe。

可持续的论文主线是：

## **Orthogonal Option-Set Feasibility**

- Social axis：ego safety 不得建立在 protected agents 的 natural low-burden option-set collapse 上；
- Physical axis：uncertified recovery 不得建立在 ego 自己返回 certified control-realizable feasibility 的 recourse collapse 上。

两轴对象不同、语义正交，但共享“safety 不应建立在 critical option-set collapse 上”的统一抽象。

当前环境无法进行 2026 年最新文献在线核验，因此本轮只确认内部逻辑、证据链和方法结构达到严谨研究迭代标准；正式投稿前仍需完成最新 related-work/novelty 检索。

---

# 8. 对上传 V37 草案的代码审计与修复

原上传 code tree 已包含 V37 草案。独立审计发现三处会破坏 witness semantics 的问题，并增加一项必要 invariant。

## 8.1 修复 1：recourse existence 不再“一 macro 一代表”

问题：同一 macro 中 max-prefix action 可能不 restore，而另一 actual emitted action 可以 restore。只看一个代表会产生 false-negative recourse set。

修复：保留每个 distinct `(macro, emitted target)` action class；仅去掉物理上等价的 emitted actions。macro existence 由任一 class 恢复 conventional support 决定。

## 8.2 修复 2：actual bridge 与 entry witness 绑定

问题：草案只保存 boolean pending。下一拍可能执行一个 entry 时从未见证的 newly discovered macro，导致“见证 A、执行 B”。

修复：entry 保存 `recovery_bridge_allowed_macros`；actual bridge 只能从该 witnessed set 选择。

## 8.3 修复 3：actual bridge 不得牺牲当前 survival

问题：草案 bridge 只要求 candidate positive-prefix，没有要求不低于当拍 ordinary COWP base。

修复：actual bridge 使用 `prefix_bridge >= prefix_base` 的 hard floor，同时仍要求 >0。

## 8.4 新增 invariant：entry action 至少跨过一个 safe edge

问题：若 base/alt prefix 同为 0，原草案仍可能根据 successor model 宣布 returnability，尽管当前 action 在下一次 replanning 前已经 unsafe。

修复：`_returnability_current_edge_admissible` 要求 alt prefix≥1 且不低于 base。

## 8.5 新 diagnostics

新增：

- base/RVR action classes available/evaluated；
- current-action survival；
- witnessed macro count before bridge；
- actual bridge candidate pool/action classes/evaluated；
- bridge minimum prefix；
- bridge execution/abort conditioned on pending；
- selected bridge macro。

这些量仅用于 attribution，不新增事后 outcome Gate。

---

# 9. 预注册实验协议

## 9.1 Stage-1：counterfactual48

仍使用 V33–36 原封不动的六项 Gate：

| condition | GO |
|---|---:|
| old RVR rescues retained | ≥5/10 |
| old RVR induced avoided | ≥7/9 |
| COWP collision net removed | ≥3 |
| Kinematics net regression | ≤1 scene |
| paired mean EP Δ | ≥−0.05 |
| intervention | >0 |

任一失败即 archive；不得修改 threshold。

机制 diagnostics 只帮助解释：

- returnability probe 是否非零；
- direct vs non-direct entry；
- bridge 是否在实际 state 可兑现；
- abort 是否过高；
- witnessed action-class search 是否真被使用。

它们不是第七个 promotion Gate。

## 9.2 fresh37

只有 Stage-1 `pass=true` 才允许运行 historical fresh37 panel。Gate：

- no net collision harm；
- no net CR harm；
- offroad/kinematics regression 各≤1 scene；
- mean EP Δ≥−0.03；
- intervention>0。

fresh37 仍不是 publication holdout，只是当前 lineage 中相对干净的 development generalization check。

## 9.3 exact200

只有 fresh37 pass 才允许 historical exact200 development confirmation。算法 freeze 后必须重新构建从未参与机制选择的 final evaluation set。

---

# 10. 下一步运行指令

```bash
cd COWP_v16_8_37_RECOURSE_RETURNABILITY_BRIDGE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh make_ids

# 仅当 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh analyze_counterfactual48
```

**到这里停止。**

只有 analyzer 中：

```text
preregistered_gate.recourse_returnability_bridge.pass == true
```

才运行：

```bash
PROMOTED_METHODS=cowp_recourse_returnability_bridge \
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh analyze_fresh37
```

launcher 已验证 fail-closed：缺少/失败的 Stage-1 analysis 时，fresh37 在启动 Waymax rollout 前以 exit code 4 停止。

---

# 11. 代码验证

最终 audited V37：

- V37 dedicated tests：8/8 passed；
- V16.8.25→37 focused semantic/integrity suite：61/61 passed；
- Python compile：passed；
- launcher `bash -n`：passed；
- exact200/equivalence16/counterfactual48/fresh37 manifest hashes：passed；
- analyzer v2 smoke：passed；
- fail-closed promotion path：passed（exit code 4, no rollout）；
- conventional bypass grep：passed。

完整仓库 `pytest -q` 仍在 collection 阶段遇到历史文件 `tests/test_v16_8_29_recovery_viability.py` 导入不存在的 `_recovery_bridge_viability_mask`。相同错误已在未修改的上传代码中独立复现，因此不是 V37 引入；本轮没有为了制造“全绿”而恢复已归档历史 API。

---

# 12. 明确禁止的后续方向

除历轮 changelog 禁止项外，V36/V37 新增：

1. 不继续调 V36 semantic frontier、weak dominance、fallback order 或 active-macro hopping。
2. 不把 valid candidate 数、macro 数、profile area 当 returnability。
3. 不通过 fixed dwell、minimum duration、hysteresis epsilon/margin 修 mode trap。
4. 不把 V37 失败后改成 2/3/4-step bridge；那会重复 horizon stacking。
5. 不按 recourse macro count/AUC/discount/risk/progress weighted sum 排序。
6. 不放宽六项预注册 Gate。
7. 不在 analytic returnability witness 未验证前训练 neural head。
8. 不全局 retune common controller。
9. 不在同一版本修改 accepted-path kinematics。
10. 不调整 RCOT/BCOT threshold/budget、protected-priority semantics 或 8 s conventional contract。
11. 不立即扩 map/Frenet primitive；当前 collapse 明确是 dynamic collision-side。
12. 不把 current logged replay 结果写成真实 causal burden 结论。

---

# 13. 若 V37 失败，下一条真正值得做的分支

若 Stage-1 FAIL，应正式停止：

`RVR → SOV/BHOV/THOP → ROSH/EOSH/CPOSH → semantic frontier → finite bridge selector`

这一 selector family。

下一版应进入更高一级的：

## **Genuine Control-Reachable Recovery Support / Viability Construction**

候选方向包括：

- 在 controller-state-augmented state space 中构造有限 reachable tubes / viability kernel approximation；
- 基于实际 low-level dynamics 生成可返回 full-conventional set 的 recovery primitives，而不是只重排现有 bank；
- 用 unbiased paired proposal probe 先证明新 support 能降低 `conventional-without-recoverable` floor，且不丢失原有 NCF scenes；
- analytic target 通过后，才考虑 amortized learned reachable-support predictor。

这是 proposal/reachable-support construction，而不是再加一个 selector score。只有这一层获得稳定证据，physical axis 才有资格与成熟 social axis 组合成 CCF-A 主算法。
