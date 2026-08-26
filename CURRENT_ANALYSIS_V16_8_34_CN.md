# V16.8.33 结果分析与 V16.8.34 设计：Execution-Conditioned Recovery Option Spectrum

## 0. 版本说明与总判断

当前上传代码的真实版本链是 **V16.8.33 → V16.8.34**；消息中出现的 “V12.8.34” 按前几轮同样规则视为版本号笔误。本轮没有重建数据、没有重训模型，也没有改变成熟的 social-feasibility 主链。

先给结论：

1. **V16.8.33 counterfactual48 结果通过完整性/实现可靠性审计，可以做算法归因。**
2. 按 V16.8.33 预注册 Stage-1 gate，**SDH 与 ROSH 都 FAIL，因此 V16.8.33 不能运行 fresh37，也不能 promotion 当前 implementation。**
3. SDH 是明确的 negative；ROSH 是 **near-positive mechanism / failed policy**：六个 gate 只失败 kinematics regression 一项（+2 scenes，而上限是 +1），科研纪律上仍必须判 FAIL。
4. ROSH 相对 SDH 的提升说明 **Recovery Option Spectrum representation 本身是真正有信息量的机制对象**；但两个新增 kinematics 回归暴露出 V33 spectrum 仍只是 nominal semantic option support，而不是 **controller-state-conditioned executable option support**。
5. 因此 V16.8.34 的 P0 不是继续调 ROSH comparator，也不是把 kinematics gate 放宽，而是把 physical feasibility 从 semantic option preservation 收紧为：

> **Control-Realizable Recovery Option-Set Preservation under Uncertified Recovery**

6. V16.8.34 落地两个解耦分支：
   - `cowp_transition_guarded_rosh`（TG-ROSH，diagnostic）：只增加当前 recovery action 的 controller-transition non-regression；
   - `cowp_executable_option_spectrum_hysteresis`（EOSH，main）：进一步把 successor spectrum 改为 controller-state-conditioned executable spectrum。

---

# 1. 对论文与研究主线的理解

论文当前最有价值的原始命题不是“增加一个 social cost”，而是：**collision-free 并不等于真正安全；如果 ego 的所谓安全依赖其他 road user 承担 hard braking / abrupt yielding / priority abandonment / gap surrender，则这是 false-safe planning，应作为 feasibility defect，而不是 courtesy penalty。**

现有 COWP social axis 的核心结构为：

`natural alternatives → same-root RCOT → BCOT → protected-priority non-coercive certificate → certificate-compatible set preservation`

它回答：

> ego 是否通过压缩其他 critical actor 的 natural low-burden response options 才获得 safety？

V16.8.28→33 的在线闭环证据又暴露出一个正交问题：即使 social certificate 本身是好的，当 full conventional set 为空、planner 被迫进入 uncertified recovery 时，ego 也可能通过当前动作把**自己的未来 executable options 压垮**。

因此当前最值得维护的论文统一抽象仍然是：

## Orthogonal Option-Set Feasibility

- **Social option-set feasibility**：保护别人的 natural low-burden options；
- **Physical/execution option-set feasibility**：保护 ego 自己在实际控制接口下的 future executable recovery options。

统一原则：

> **Safety should not be obtained through critical option-set collapse.**

V34 不把 safety filter / hysteresis / recursive feasibility 自身包装成 novelty；它们只是验证 physical axis observable 是否正确的工程/机制 probe。

---

# 2. compact-5k 数据集性质审计（不重建）

`formal_v16_8_24_compact_full_5k` 的 split contract：

- train = 5000；
- val = 1000；
- heldout_test = 1200；
- train/val 来自 historical promoted cache；
- heldout_test 是 official WOMD validation 的 held-out subset（official WOMD test future hidden）。

三 split 的性质高度一致：

| 项目 | train | val | heldout_test |
|---|---:|---:|---:|
| critical agents / scene mean | 5.3846 | 5.3870 | 5.3383 |
| audit-relevant pair rate | 0.42970 | 0.42863 | 0.42949 |
| protected PRIO root coverage | 0.99453 | 0.99363 | 0.99465 |
| rootless rate | 0 | 0 | 0 |
| <2 low-burden roots rate | 0 | 0 | 0 |
| mechanism-unauditable selected critical rate | 4.07% | 4.34% | 4.46% |
| pair-neutral-unsafe rate | 14.89% | 14.92% | 15.05% |

有代表性的 proposal acceptance rate 也稳定：`JOINT_ROUTE_NCF≈0.94–0.98`、`ROBUST_BCTE≈0.98`，而 `PRIORITY_SMOOTH_YIELD≈0.20–0.22`、`TERMINAL≈0.54–0.55`。这说明当前 compact set 并不是某一 split 上自然机制突然失配的异常数据。

`verify_cache_train.json` 的顶层 `pass=false` 仅由 historical verifier 把 `irrelevant pair blockers=58243` 作为失败原因；同时 missing keys、read errors、scenario mismatch、affected/conflict/retained root mismatch、canonical root weight mismatch 均为 0。因此本轮把它记录为 verifier-semantics 边界，而不是重建数据的证据。

结论：**按用户约束继续 Freeze compact-5k，不重建。** 当前 P0 来自 closed-loop recovery/interface，而不是 offline natural-data contract。

---

# 3. V16.8.33 结果可靠性审计

## 3.1 Manifest / shard / merged

全部 logical SHA256 与预注册 reference 一致：

- exact200: `3fb2e3607b4cd8ca977456bfc08f9d41aadf949f338549d4f1e16c92fea1529f`
- equivalence16: `81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760`
- counterfactual48: `ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0`
- fresh37: `ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481`

`counterfactual48`：SDH/ROSH 都是 24+24 shards、互不重叠、union 精确等于 manifest；merged 每个方法 48 unique IDs。

## 3.2 Summary 独立重算

对 SDH/ROSH merged 的 CR / Collision / Offroad / Kinematics / EP 等 standard metrics，从 48 个 scenario rows 独立重算，**最大绝对误差 = 0**。

## 3.3 Common path equivalence

`equivalence16_cowp_vs_v16_8_29.json` 继续通过：

- 16 scenes；
- 1120 fields；
- 0 mismatch。

说明 V33 没有伤害成熟 COWP common path。

## 3.4 Analyzer 独立复现

我用上传的 immutable V29/V30/V31/V32 references + V33 SDH/ROSH merged 重新运行 V33 analyzer，并对上传 analyzer JSON 做递归逐字段比较（1e-12 tolerance）：

- **0 mismatch**。

因此 gate 不是 analyzer 偶然/打包错误。

## 3.5 因果信息边界

V33 recovery successor/profile 只使用当前 simulator state、实际 emitted target 与 conventional collision model 同源的 causal constant-velocity surrounding-agent propagation；没有读取 logged future 作为 online ground truth。结果适合做 physical closed-loop mechanism attribution。

但 logged-replay Waymax 本身不能证明真实 counterfactual burden transfer；论文最终 burden 因果 claim 仍必须遵守 reactive-agent + human-audited false-safe stress protocol。

**可靠性结论：PASS，可以进行算法归因。**

---

# 4. 严格按 V33 预注册 GO 条件：两个分支都 FAIL

Stage-1 原 gate：

- retain old RVR rescue >= 5/10；
- avoid old RVR induced >= 7/9；
- COWP collision net reduction >= 3 scenes；
- kinematics regression <= 1 scene；
- paired mean EP delta >= -0.05；
- intervention > 0；
- 任意一项失败即 archive，禁止改阈值。

## 4.1 SDH (`cowp_sov_dominance_hysteresis`)

| Gate | 结果 | 判定 |
|---|---:|---|
| old RVR rescues retained | 4/10 | **FAIL** |
| old RVR induced avoided | 7/9 | PASS |
| COWP collision net reduction | 2 scenes | **FAIL** |
| kinematics regression | 0 | PASS |
| mean EP delta | -0.00703 | PASS |
| intervention | >0 | PASS |

总结果：Collision 32/48、Kinematics 6/48、EP 0.99548。

**SDH 是明确 negative。** 仅靠 strict-entry / equality-continue / dominance-loss-exit 的 mode semantics 不足以把 physical recovery 变成稳定机制。

## 4.2 ROSH (`cowp_recovery_option_spectrum_hysteresis`)

| Gate | 结果 | 判定 |
|---|---:|---|
| old RVR rescues retained | 6/10 | PASS |
| old RVR induced avoided | 7/9 | PASS |
| COWP collision net reduction | **5 scenes** | PASS |
| kinematics regression | **+2 scenes** | **FAIL** |
| mean EP delta | **-0.02786** | PASS |
| intervention | 7.84% policy steps | PASS |

总结果：

- COWP: Collision 34/48, Kinematics 6/48, EP 1.00251；
- ROSH: Collision **29/48**, Kinematics **8/48**, EP **0.97465**；
- Collision paired transition：7 rescue / 2 induced / net -5；McNemar p=0.1797；
- EP bootstrap 95% CI ≈ `[-0.08755, +0.02881]`。

**ROSH 仍然必须 FAIL。** 不能因为只差一个 kinematics scene 就把 gate 从 `<=1` 改成 `<=2`。

同时要区分“policy promotion”与“mechanism promotion”：

- 不 promotion V33 ROSH implementation；
- **promotion Recovery Option Spectrum 作为下一轮 physical-feasibility representation 的核心对象。**

---

# 5. V33 真正成功的机制

## 5.1 Recovery spectrum 明显优于 state-machine-only SDH

SDH 只净救 2 个 collision，retain 4/10 old rescues；ROSH 在相同 dominance-hysteresis mode semantics 下净救 5 个、retain 6/10，同时 old induced avoided 仍为 7/9。

因此提升不能主要归因于 hysteresis；**semantic option-spectrum representation 提供了额外有效信息。**

ROSH 的 profile probe 只发生在约 15.26% steps；strict pointwise dominance rate ≈27.25%，weak dominance ≈74.92%，实际 recovery switch 约 7.84%。这不是“全程强制新 policy”造成的粗糙变化，而是 sparse gate 对闭环状态的选择。

## 5.2 `7721ff4800156886` 是关键正反例

此前 V30 SOV / BHOV / THOP / commitment / SDH 对这个 old-RVR-induced scene 都仍有 false positive；V33 ROSH 首次保持 collision-free，EP≈1.433。

这说明：

> “有多少 distinct semantic recovery modes 在各 collision horizon 仍然存活”的确包含 max-prefix / conventional-count signature 缺失的信息。

因此不应退回单最长 trajectory，也不应 archive option-spectrum 概念本身。

---

# 6. V33 失败机制：两个 kinematics regression 不是同一种失败

ROSH 相对 COWP 新增的 kinematics scenes：

`6992366c5c998d00`
`29cd2aca8ae5e222`

## 6.1 `29cd2aca...`：direct recovery execution failure

- first kinematics = step 55；
- 前一步仍在 zero-conventional fallback；
- fallback reason = `no_conventional_use_recovery_option_spectrum_hysteresis`；
- macro = `MERGE_AHEAD`；
- valid=true、roadgraph-safe=true、collision-safe=false、prefix=0。

这是 recovery action 本身的 execution/interface 失败。

## 6.2 `6992366c...`：downstream controller-state failure

- first kinematics = step 17；
- first kinematics 前一步已经**不是 recovery**；
- reason=`accepted_priority_ncf`；
- candidate conventional-safe=true、collision-safe=true、prefix=80；
- macro=`LANE_CHANGE_RIGHT`。

也就是说 ROSH 先改变了 earlier closed-loop state，之后 planner 回到“成熟 certified/conventional path”，但此时 controller/internal transition state 已不同，导致一个 nominally valid/conventional candidate 在 actual interface 下出现 kinematics infeasibility。

因此不能用一个“只 guard 当前 fallback action”的补丁解释全部问题。

---

# 7. 代码层根因：V33 spectrum 不是 executable spectrum

代码审计发现 V33 physical profile 有两个结构性 representation gap。

## 7.1 Policy state 不只有 agent_state

实际 online action projection 维护：

`self._previous_longitudinal_accel`

下一步 emitted acceleration 会受到：

- max acceleration / deceleration；
- previous acceleration ± max jerk × dt；
- yaw-rate / Waymax max-delta-yaw；
- lateral-acceleration contract。

因此真实控制 policy 的 Markov state 至少是：

`(simulator agent_state, previous_longitudinal_accel)`。

但 V33 `_successor_recovery_option_profile(...)` 只传递 `agent_state + emitted_target`，没有把 emitted action 产生的 longitudinal acceleration 作为 successor controller state 继续传递。

## 7.2 V33 option support 是 nominal candidate support

V33 profile 对 successor bank 使用：

`valid & roadgraph_safe & collision_prefix`

统计 macro survival。

而 candidate nominal dynamics validity 为了不把 acceleration/yield primitives 全部删除，历史上刻意允许 initial jerk transient；这对 proposal validity contract 是合理的，但意味着：

> nominal `valid` 并不等价于“从当前 controller memory 下一拍无失真可执行”。

于是 V33 的 `P_s(h)` 可能把一个**几何上存在、collision prefix 也长，但下一拍需要 controller 大幅裁剪/扭曲的 macro**算作独立 option。

这正好解释为什么 semantic spectrum 能改善 collision，却仍出现 kinematics regression。

---

# 8. dominant bottleneck 再收紧

历史证据链：

1. V28：online conventional-feasible support collapse；
2. V29：主要是 dynamic collision-side zero-conventional，非 roadgraph；
3. V29 RVR：same bank 中确有 recovery action，故 proposal 不是当前唯一 bottleneck；
4. V30：successor option set 有高精度信息；
5. V31：stateless hybrid switching 会制造 harmful dynamics；
6. V32：horizon stacking 与 unconditional commitment 都失败；
7. V33：semantic option spectrum 显著提高 collision-side recovery precision/recall，但执行可实现性不完整。

所以当前 P0 应写成：

# **Execution-Conditioned Recovery Option-Set Feasibility**

更具体：

> 在 zero-conventional recovery 中，选择动作时必须保护的是“经过真实 controller state/action projection 后仍可执行的 recovery modes 的 survival structure”，而不是 nominal trajectory bank 中的 semantic options。

Proposal support 仍是长期 ceiling，但当前证据尚不支持立刻扩 bank：V29→33 已反复证明相同 bank 内 selector/representation 改变可以真实 rescue 多个 scene。

Accepted-path kinematics 仍是 secondary：如果 V34 能消除 recovery-induced controller-state pollution 但仍存在 clean accepted-path kinematics，再单独进入 Execution-Viability Certificate；不要和本轮混改。

---

# 9. 各层成熟度

| Layer | 当前状态 | V34 策略 |
|---|---|---|
| compact-5k data/label contract | Mature | **Freeze** |
| Natural roots | Mature | **Freeze** |
| RCOT same-root transport | Strong | **Freeze** |
| BCOT | Strong | **Freeze** |
| Protected-priority hard certificate | Mature | **Freeze** |
| Set-preservation frontier | Supported by CTU evidence | **Freeze** |
| Outcome head | Diagnostic-only | **Freeze** |
| 8 s conventional contract | Stable semantic reference | **Freeze** |
| V27 conventional integrity | Solved | **Freeze** |
| V28 no-valid execution integrity | Solved | **Freeze** |
| RVR max-prefix | Policy negative / controlled alt useful | reference only |
| SDH | Negative | Archive |
| ROSH semantic spectrum | Mechanism positive / policy gate FAIL | absorb representation |
| Controller-realizable recovery representation | **Immature / P0** | V34 main |
| Recovery mode state machine | useful but not sufficient | retain dominance semantics |
| Accepted-path kinematics | Secondary | do not co-modify |
| Proposal support | Long-term ceiling | do not expand now |

---

# 10. 后续明确禁止的算法方向

继续遵守已有 changelog 禁区，并新增 V33 证据约束：

1. certificate → planner-score argmin / CTU replacement；
2. outcome fallback weight tuning；
3. outcome head hard shield；
4. 缩短 8 s conventional horizon 来制造候选；
5. 直接 promotion RVR；
6. prefix/risk/Pareto scalar weight 搜索；
7. 放宽 BHOV/ROSH comparator、epsilon、margin；
8. V3/V4/V5 horizon stacking；
9. unconditional commitment / fixed dwell time；
10. profile AUC / discounted area 参与选择；
11. social + physical + utility 单 scalar；
12. 当前阶段扩 map/Frenet primitive；
13. 调 RCOT/BCOT threshold/budget；
14. analytic target 验证前训练 successor/viability neural head；
15. 同一版本修改 accepted-path execution certificate；
16. **把 V33 kinematics gate 从 +1 放宽到 +2；**
17. **只给 ROSH fallback 加 kinematics penalty，然后保留 nominal spectrum。** `6992366c...` 已证明 downstream controller-state effect 不是当前 action risk 能完全覆盖。

---

# 11. V16.8.34 分支 A：Transition-Guarded ROSH（TG-ROSH）

Method:

`cowp_transition_guarded_rosh`

角色：**diagnostic，不作为 paper contribution。**

保持 V33 的 nominal semantic successor profile 完全不变，只在 base/RVR 当前动作之间额外计算 hard controller-transition feasibility：

`E_t(a) ∈ {0,1}`

它复用现有 controller/candidate hard limits：acceleration、deceleration、jerk transition、yaw-rate / Waymax yaw delta、lateral acceleration，不新增 tuned threshold。

TG-ROSH 的 product order：

- future profile 必须 pointwise non-regressive；
- current transition feasibility 不允许 `1 → 0`；
- entry 至少一个维度 strict improvement；
- active mode continuation 使用 weak dominance；
- 任一 component regression 即 exit。

它回答：

> V33 的 kinematics regression 是否主要只是“当前 RVR action nominally 很好，但 controller transition 不可实现”？

若 TG-ROSH 已修复 +2 kinematics 且保持 V33 collision gain，则无需进一步复杂化 future spectrum。

---

# 12. V16.8.34 主分支：Executable Option-Spectrum Hysteresis（EOSH）

Method:

`cowp_executable_option_spectrum_hysteresis`

## 12.1 Successor state 加入 controller memory

对 base/RVR 都先使用已有 `_one_step_action_risk_np(..., return_targets=True)` / emitted controller 得到：

- actual emitted target；
- actual emitted longitudinal acceleration `a_t^emit`。

构造 causal successor simulator state 后，将 `a_t^emit` 作为 successor 的 `previous_longitudinal_accel`。

不读 logged future，无信息泄漏。

## 12.2 Executable recovery spectrum

在 unchanged successor proposal bank 中，一个 candidate 只有同时满足：

- existing `valid`；
- roadgraph-safe；
- **从 carried controller state 到该 candidate 第一目标步的 hard transition feasible**；
- collision-safe prefix >= h；

才允许对应 macro 计入 horizon h 的 option support：

`P_exec(h) = # distinct non-PAD recovery macro modes surviving h and immediately realizable under controller state`

同一 macro 的多个近似 trajectory 仍只计一次。

重要：这个 transition filter **不改 candidate 的 conventional/NCF label**，不改 training/data contract，只用于 zero-conventional recovery 的 physical option representation。

## 12.3 选择仍是 partial order，不 scalarize

EOSH product relation：

`(current transition feasibility, future executable spectrum)`

RVR 只有在两个 components 都 non-regressive，并至少一个 strict improvement 时才 entry；active 时 weak dominance continuation，任一 regression exit。

没有：

- risk weight；
- profile area weight；
- horizon discount；
- tuned hysteresis threshold；
- dwell time；
- new model head。

因此 V34 可以清楚地区分：

| 结果 | 归因 |
|---|---|
| TG 好，EOSH 无额外收益 | 当前 action controller transition 是主缺口 |
| EOSH 明显 > TG | carried controller state + executable future option spectrum 是关键 |
| 两者都过 Stage-1 | 最终 physical axis 应统一 execution-aware representation + dominance mode |
| 两者都失败 | 停止 ROSH-family refinement；转更高保真 causal dynamics / reachable-option construction，再考虑 proposal support |

---

# 13. V16.8.34 预注册实验协议

## Stage 0

`sanity → make_ids → equivalence16`

必须维持 common COWP 0 mismatch。

## Stage 1: counterfactual48

TG-ROSH / EOSH 使用 **与 V33 完全相同**的六项 gate，不根据 V33 near-miss 修改：

- old RVR rescues retained >=5/10；
- old RVR induced avoided >=7/9；
- net COWP collision reduction >=3；
- kinematics regression <=1；
- mean EP delta >=-0.05；
- intervention >0。

任一 fail 即 archive。

另外记录但**不作为新增 gate**：

- V33 两个 kinematics-induced IDs 是否被避免；
- EOSH vs TG paired transition；
- transition-rejected roadgraph candidates；
- executable profile probe/switch rates。

这样不会用已知 V33 counterexamples 重写 promotion gate。

## Stage 2: fresh37

只有 Stage-1 `preregistered_gate.pass=true` 的方法可运行。fresh37 是 V33 从未运行过的结果-unseen development stage，但仍属于历史 exact200 universe，不是 publication holdout。

保持 V33 fresh37 gate：

- no net collision harm；
- no net CR harm；
- offroad regression <=1；
- kinematics regression <=1；
- paired mean EP delta >=-0.03；
- intervention >0。

## Stage 3: exact200

fresh37 通过后才允许；只跑 promoted new method，复用 immutable exact200 COWP/RVR references。仍只算 development confirmation。

最终 paper freeze 后必须重新冻结从未参与算法选择的 final evaluation set，并按论文协议做 >=3 seeds + paired scenario CI；false-safe causal claim 还要 reactive-agent + human-audited stress evidence。

---

# 14. CCF-A 研究价值判断

不能把以下内容单独当 novelty：

- safety filter；
- backup trajectory / recursive feasibility；
- hysteresis；
- multi-horizon feasibility。

这些方向已有成熟工作。V34 值得继续的论文主线在于：

## Orthogonal Option-Set Feasibility with Execution-Conditioned Physical Options

Social axis 与 physical axis 不共享 actor，却共享同一个 feasibility principle：

- social：ego 不应把**别人**的 natural low-burden response set 压垮；
- physical：uncertified recovery 不应把**自己**在真实执行接口下的 future recovery set 压垮。

如果 EOSH 在 counterfactual48 + fresh37 稳定成立，论文可进一步发展为：

> non-coercive social feasibility × control-realizable physical feasibility，二者都通过 set preservation / hard partial order，而不是通过 score reweighting 获得 safety。

如果 EOSH 失败，则 physical axis 当前 observable 仍不够，不能硬塞进主算法；此时保留 social COWP 主线，把 recovery 作为 failure analysis，再转向 higher-fidelity reachable-option construction。

---

# 15. 本地代码验证

V16.8.34 当前交付：

- 新增 V34 unit/semantic tests：4/4 passed；
- V34 + V33 focused tests：9/9 passed；
- V16.8.25→34 focused semantic/integrity sanity：**43/43 passed**；
- exact200/equivalence16/counterfactual48/fresh37 manifest hashes：全部通过；
- `py_compile`：passed；
- `bash -n` launcher：passed；
- V34 analyzer 用 V33 ROSH 作为 synthetic static input：正确复现“kinematics +2 导致 gate FAIL”，证明 gate 没被意外放宽；
- full repository `pytest -x` 在本环境 120 s 限制下运行到约 23% 后超时，超时前未出现 failure；不把该不完整运行声称为 full-suite pass。

---

# 16. 下一步执行指令

只跑到 counterfactual48：

```bash
cd COWP_v16_8_34_EXECUTABLE_OPTION_SPECTRUM

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/你的旧COWP目录/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh make_ids

# 仅当 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh build_tfindex

# common path equivalence
bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh base_equivalence16_parallel2

# 两个解耦根因分支
bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh analyze_counterfactual48
```

**到这里停止。**

只有 analyzer 中某个方法的 `preregistered_gate.pass=true` 才运行 fresh37。例如 EOSH 通过：

```bash
PROMOTED_METHODS=cowp_executable_option_spectrum_hysteresis \
bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_34_EXECUTABLE_OPTION_SPECTRUM_CN.sh analyze_fresh37
```

launcher 保持 fail-closed；未通过上一阶段会直接退出，不会启动 Waymax rollout。
