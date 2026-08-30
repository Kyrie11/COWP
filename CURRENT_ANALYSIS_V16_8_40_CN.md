# V16.8.39 实验审计、机制归因与 V16.8.40 设计

## 0. 一页结论

### 可靠性

V16.8.39 结果通过独立可靠性审计：

```text
222 / 222 hard checks passed
reliability_verdict = PASS_ATTRIBUTION_ALLOWED
```

检查覆盖：

- 上传 ZIP 与 release SHA256；
- exact200 / equivalence16 / counterfactual48 / fresh37 冻结 manifest；
- counterfactual48 的 24+24 shards、无重叠与 manifest 精确覆盖；
- shard 与 merged 逐场景 row 精确一致；
- CR / Collision / Offroad / Kinematics / EP / fallback 独立重算；
- checkpoint logical provenance 与各 shard 运行协议；
- `mechanism_ground_truth_available_online=false`；
- V16.8.28 no-valid/emergency execution invariant；
- equivalence16：16 scenes / 1120 fields / 0 mismatch；
- 使用上传 ZIP 中 pristine V39 源码独立重跑 analyzer，递归比较 0 mismatch，容差 `1e-12`；
- V39 dedicated/focused tests、compile、launcher、final archive roundtrip。

因此可以做算法归因，不需要退回 repair-only。

### 冻结 Gate

V39 仅失败一项，但仍必须归档：

| Stage-1 条件 | GO | V39 | 结论 |
|---|---:|---:|---|
| old RVR rescues retained | >=5/10 | **3/10** | **FAIL** |
| old RVR induced avoided | >=7/9 | **9/9** | PASS |
| 相对 COWP 净减少 collision | >=3 | **4** | PASS |
| Kinematics 净退化 | <=1 scene | **0** | PASS |
| paired mean EP delta | >=-0.05 | **-0.011079** | PASS |
| action-changing intervention | >0 | **87 steps** | PASS |

最终：

```text
V16.8.39 POLICY = ARCHIVE
NO FRESH37
```

不能把 rescue retention 从 5/10 改成 3/10，也不能用“其余五项都过了”替代 conjunction Gate。

### dominant bottleneck

V39 将 future tube hypotheses/probe 从 V38 的约 96.71 提升到 225.40，full-safe 和 shift-closed witnesses/probe 分别提高约 3.62 倍和 3.70 倍；但 unique actual first actions/probe 从 59.92 变成 59.73，基本没有扩大。

因此当前 P0 已收紧为：

# **Shift-Closed First-Action Viability Support under Ternary Control Quantization**

论文级名称：

# **Control-Interval Reachable Backup Support**

V40 应扩大真实首拍可达控制支持，而不是继续增加 future release schedules。

---

# 1. V16.8.39 结果可靠性

## 1.1 Artifact 与 provenance

上传的 audited V39 ZIP：

```text
SHA256 =
b2e3afad585044d15cd3a6ca437b97edefd1df416c6ec485268449e890f2c92c
```

与上一轮发布的 archive hash 一致。`V16_8_39_RELEASE_SHA256.txt` 中 16 个 release files 全部在 pristine 解压树中匹配。

结果包包含：

- counterfactual48 两个 shards、merged、analysis；
- equivalence16 两个 shards、merged、equivalence report；
- 四组冻结 manifest；
- wall-time files。

没有 fresh37 / exact200 新 rollout，符合 Stage-1 失败后停止的 fail-closed 协议。

## 1.2 Manifest 与 shards

冻结 hashes：

| Manifest | Count | logical SHA256 |
|---|---:|---|
| exact200 | 200 | `3fb2e360...fea1529f` |
| equivalence16 | 16 | `81d0319d...571c760` |
| counterfactual48 | 48 | `ee3c231c...ddc40ab0` |
| fresh37 | 37 | `ecce3321...5a1481` |

counterfactual48：

```text
shard0 = 24 unique scenes
shard1 = 24 unique scenes
intersection = 0
union = exact frozen 48-ID manifest
```

equivalence16：

```text
shard0 = 8
shard1 = 8
intersection = 0
union = exact frozen 16-ID manifest
```

## 1.3 运行协议

两个 counterfactual shards 一致使用：

```text
method = cowp_conflict_window_control_reachable_tube
checkpoint = outputs/v16_8_24_compact5k_all/cowp_all_best.pt
ncf_gate_mode = priority
waymax_split = validation
rollout horizon = 80 steps
BCOT budget = 0.5
pair witness threshold = 0.7
reuse/prefilter/JIT = true
non-ego policy = logged_replay
reactive mixture = false
online mechanism GT = false
```

结果包没有 checkpoint bytes，因此只能验证 logical checkpoint provenance 和 shard 一致性，不能独立计算 `.pt` 文件内容 hash。这不阻断当前固定 checkpoint 协议下的归因。

## 1.4 merged 与指标重算

48 个 merged scenario rows 可以从 shards 精确重建。对 merged 中所有标准指标独立求均值，与 summary 的最大误差为 0；fallback rate 同样零误差。

V16.8.28 execution invariant 对 48/48 scenes 成立：

```text
emergency_action_step_rate
=
zero_valid_candidate_step_rate
=
no_valid_step_rate
```

equivalence16：

```text
scenarios = 16
fields_checked = 1120
mismatch = 0
passed = true
```

## 1.5 analyzer 独立重放

使用 pristine 上传 V39 代码重新执行：

```text
92_analyze_conflict_window_control_reachable_tube.py
```

上传 analysis 与独立重建 JSON：

```text
recursive mismatches = 0
tolerance = 1e-12
```

## 1.6 两个不阻断归因的边界

第一，Waymax 中 surrounding agents 使用 logged replay。当前结果可用于冻结协议下的 ego closed-loop physical attribution，但不能作为强 counterfactual social-burden ground truth。

第二，counterfactual48 已在多轮机制选择中使用，只是 development-selected panel，不是论文最终 holdout。

另外发现一个纯 diagnostics omission：V39 的 generic `no_conventional_step_rate` reason set 没加入 V39 fallback reason。method-specific tube rate、zero-conventional rate、outcomes 和六项 Gate 都不受影响。V40 已修复该 reporting-only omission。

完整机器审计见：

```text
V16_8_39_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT_INDEPENDENT.json
```

---

# 2. 按预注册 Gate 判断 V39 成败

## 2.1 Headline

| Method | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 1/48 | 6/48 | 1.002512 |
| RVR | 33/48 | 1/48 | 9/48 | 0.823619 |
| V33 ROSH | 29/48 | 1/48 | 8/48 | 0.974647 |
| V35 CPOSH | 27/48 | 0/48 | 7/48 | 0.913987 |
| V36 Frontier | 34/48 | 0/48 | 8/48 | 0.978086 |
| V37 RRB | 33/48 | 0/48 | 7/48 | 0.937350 |
| V38 SCRT | 33/48 | 1/48 | 6/48 | 0.997766 |
| **V39 CW-CRT** | **30/48** | **1/48** | **6/48** | **0.991434** |

V39 vs COWP：

```text
Collision: 4 rescue / 0 induced / net -4
Offroad:   0 rescue / 0 induced
Kin:       0 rescue / 0 induced
mean EP delta = -0.0110785
bootstrap 95% interval = [-0.02731, +0.00098]
McNemar collision exact p = 0.125
```

这不是 publication-level 显著性结果，但作为 development mechanism evidence 是干净的 high-precision improvement。

## 2.2 为什么仍然必须 FAIL

历史 RVR 10 个 rescue 中，V39 只保留：

```text
40f2d8b336a3dafc
7c6ac47c0deee2af
f8d5d2f0f7cf5825
```

另一个 V39 rescue：

```text
6992366c5c998d00
```

是 V38 已发现的 clean constructive-support rescue，不属于那 10 个 old RVR rescues。

因此：

```text
old rescue recall = 3/10
```

低于冻结的 5/10。Conjunction Gate 的意义正是防止一个机制只做成“几乎不犯错但也几乎不救人”的 recovery rejector。

V39 不能 promotion，也不能跑 fresh37。

---

# 3. V39 真正成功的机制

## 3.1 它不是 dead code

48×80=3840 policy steps 中：

| Mechanism | Pooled count |
|---|---:|
| recovery probes | 2632 |
| hard certificates | 88 |
| actual action changes | 87 |
| scenes with action change | 20 |
| event-release selected certificates | 83/88 |
| lifted selected certificates | 87/88 |

所以 V39 的 event-release branch 被真实执行，不是 diagnostics-only。

## 3.2 hard certificate + shift closure 仍是高精度层

V39：

- 没有新增 collision；
- 没有新增 offroad；
- 没有新增 Kinematics failure；
- 避免 9/9 historical RVR-induced collision；
- mean EP Gate 通过。

因此以下机制应保留：

1. actual controller-realized tube；
2. 完整 horizon finite/road/collision/Waymax-kinematics hard certificate；
3. one-step shift closure；
4. hard set first，frozen COWP preference second；
5. uncertified recovery 不重新标记为 conventional/NCF。

## 3.3 event-release future witness 有真实新增信息

V39 pooled witnesses：

```text
full-safe = 1844
shift-closed = 1754
event-release shift-closed = 1271
event-release-only parent support = 506
```

83/88 selected certificates 是 event-release。

V39 相对 V38 增加三个额外 collision rescue，同时仍保持 0 induced physical failures。因此不能得出“有限冲突窗口完全没有价值”。正确结论是：

> event-derived release 是有效的 future control-sequence witness expansion，但还不是完整的 current recovery support。

## 3.4 四个 clean collision rescues

```text
40f2d8b336a3dafc
6992366c5c998d00
7c6ac47c0deee2af
f8d5d2f0f7cf5825
```

其中：

- `40f2...`：5 个 certified/action-changing steps，EP `+0.03193`；
- `6992...`：5 个 certified/action-changing steps，EP `+0.05326`；
- `7c6a...`：15 个 steps，EP `-0.29376`；
- `f8d5...`：3 个 steps，EP `-0.10712`。

前两例说明 event/window-aware constructive support 可以产生 collision-free、无 offroad/kinematics conversion、且 progress 不降的正例。后两例说明更长 intervention 仍可能以 progress 换 collision avoidance，但总体 EP Gate 尚可。

---

# 4. V39 失败机制：future witness 丰富，first action 未丰富

## 4.1 V38→V39 的关键对照

| Pooled quantity | V38 | V39 |
|---|---:|---:|
| probes | 2695 | 2632 |
| hypotheses | 260640 | 593253 |
| unique first actions | 161472 | 157205 |
| full-safe witnesses | 522 | 1844 |
| shift-closed witnesses | 486 | 1754 |
| certificates | 56 | 88 |
| action changes | 55 | 87 |

Per probe：

| Quantity | V38 | V39 | Ratio |
|---|---:|---:|---:|
| hypotheses | 96.712 | 225.400 | 2.33× |
| unique first actions | 59.915 | 59.728 | 0.997× |
| full-safe | 0.194 | 0.701 | 3.62× |
| shift-closed | 0.180 | 0.666 | 3.70× |

这组数字把结构缺口暴露得很清楚：

> V39 组合了更多 future schedule tails，但每个 parent 在当前拍仍只发 nominal、reachable lower endpoint 或 reachable upper endpoint。

因此大量新的 hypotheses 只是把相同首拍控制连接到不同 future tails，不能解决首拍三值量化的 support hole。

## 4.2 lost old rescues

七个 lost historical rescues：

```text
c9b1c562b6ff31e5
9e3e5f19ee38f2e3
ad7d72d8adca3e25
fccd9a25a2a57a73
b85168f48c8c9970
9ccf60966ec93c20
2c2395ec28c6a158
```

其中六个场景 V39：

```text
certificate steps = 0
action changes = 0
```

这些场景每拍仍有约 38–44 个 valid candidates，却找不到一个通过 ternary-first-action hard tube 的 witness。主要问题不是 selector 从 hard set 中选错，而是 hard set 为空。

`fccd...` 是不同类型：

```text
certificates = 4
action changes = 4
first collision step: COWP 67 → V39 68
Collision remains 1
Offroad remains 1
```

它说明 hard CV/shift model 仍存在 residual mismatch；但 6/7 lost rescues 的主因仍是 support absence，而不是 false-positive selection。

## 4.3 shift closure 不是召回 collapse 的主因

V39：

```text
1754 / 1844 = 95.12%
```

full-safe witness 能通过 shift closure。

因此删除 shift closure最多只能恢复约 4.88% 的 full-safe witnesses，而且会破坏当前 high-precision evidence。不能通过放宽 shift closure 来“修 recall”。

## 4.4 不应继续增加 schedule

V39 已按 nominal conflict first/last edge增加四条 event-release schedules。看到结果后再搜索任意 switch time，会变成 outcome-selected schedule grid，既损害归因，也不解决 actual first action 仍为三值的问题。

---

# 5. dominant bottleneck 的递进证据链

```text
V28  online conventional support collapse
→ V29 dynamic collision-side collapse；same bank 可 rescue
→ V30 actual emitted action 的 successor option set 有独立信息
→ V31 stateless hybrid switching 会制造 closed-loop failure
→ V32 horizon stacking / unconditional commitment 失败
→ V33 semantic option spectrum 有信息
→ V34 nominal exact-realizability 定义错误
→ V35 control-projected spectrum 有强 recovery signal
→ V36 broader existing-bank semantic frontier 失败
→ V37 exact returnability 高精度、极低召回
→ V38 hard controller-lifted tube 有 clean positive，但全时域 N/L/U support 稀疏
→ V39 event-release 显著扩大 future witnesses，但 actual first-action support 不增
```

当前 P0：

## **Shift-Closed First-Action Viability Support under Ternary Control Quantization**

更一般的论文表述：

## **Control-Interval Reachable Backup Support**

模型下一步需要表示：

\[
(s_t, a_{t-1}^{long}, \mathcal I_t^{ctrl})
\rightarrow
\{a_t \in \mathcal I_t^{ctrl}:
\tau(a_t)\in\mathcal T_{\mathrm{phys}},
\operatorname{Shift}(\tau(a_t))\in\mathcal T_{\mathrm{phys}}\}.
\]

这里的关键对象不是“候选 trajectory 数”，而是：

> 当前 controller reachable interval 内，哪些真实 emitted actions 拥有 shift-closed hard backup support？

---

# 6. 当前各层成熟度

| Layer | 状态 | V40 原则 |
|---|---|---|
| compact-5k data/labels/checkpoint | Mature | **Freeze** |
| Natural roots | Mature | **Freeze** |
| same-root RCOT | Strong/Mature | **Freeze** |
| BCOT | Strong/Mature | **Freeze** |
| Protected-priority certificate | Mature | **Freeze** |
| Certificate-compatible set preservation | Mature | **Freeze** |
| Outcome head | Diagnostic-only | **Freeze** |
| 8 s conventional contract | Stable | **Freeze** |
| V27/V28 execution integrity | Solved | **Freeze** |
| Candidate geometry families | 当前无重建证据 | **Freeze in V40** |
| Common controller limits | Mature interface | **Freeze** |
| RVR/BHOV/THOP | Policy negatives | Archive |
| SOV/ROSH/CPOSH observable | Positive representation evidence | Retain insight |
| V36 frontier | Gate FAIL | Archive |
| V37 exact returnability | High precision / very low recall | Archive |
| V38 full tube + shift closure | High-precision evaluator | **Retain** |
| V39 event-release witness | Positive future support | **Retain, nested** |
| V39 policy | Gate FAIL | Archive |
| First-action control interval support | P0 / unvalidated | **V40** |
| Accepted-path execution viability | Secondary | Keep separate |
| Interaction-aware reachable response | Next ceiling | V40 fail branch |

## 为什么 accepted-path Kinematics 仍不是本轮 P0

历史上 accepted COWP path 的 Kinematics 是独立问题，但 V39 当前 collision recovery：

```text
Kinematics 6 → 6
```

没有产生新的 Kin regression。现在把 accepted-path execution repair塞进 V40只会破坏对 first-action support 的 attribution。它应继续作为后续独立版本。

---

# 7. 模型下一步应该重点“学”什么

当前 social half 已经学习：

> ego plan 是否依赖其他 protected agents 放弃 natural low-burden responses？

physical recovery 还没有学习：

> 在当前 controller memory 和真实 actuator envelope 下，哪一段 emitted-action control interval 能同时保持当前与一次 replan shift 后的硬物理可行 backup？

因此下一阶段的 target 不是 generic collision classifier，也不是 scalar risk score，而是：

## **Action-Conditioned Viability Set Membership**

可以将 analytic V40 视为 target discovery：

```text
input:
    current causal state
    previous longitudinal acceleration
    parent semantic geometry
    frozen surrounding-agent prediction

target:
    a first action is in/out of the shift-closed hard viability set
```

只有这一 analytic object 在多组场景上被验证，才有理由进一步训练 learned support predictor。

---

# 8. V16.8.40：Shift-Closed First-Action Viability Interval

方法：

```text
cowp_shift_closed_first_action_viability_interval
```

简称：

```text
SC-FAVI
```

## 8.1 介入范围

只在：

```text
full conventional set == empty
AND valid candidate exists
```

时工作。

accepted/certified path、conventional fallback、no-valid emergency、social certificate 均保持不变。

## 8.2 强嵌套：先让 V39 独占决策权

V40 首先原样调用 V39 constructor：

```text
selected_v39, detail_v39 = V39(...)
if selected_v39 exists:
    return selected_v39 exactly
```

这意味着 V40 不会重排或伤害任何 V39-certified decision。

只有：

```text
V39 hard set == empty
```

才进行 interval completion。

## 8.3 从 fixed ternary actions 扩成 existing control interval

冻结 controller 给出首拍纵向加速度区间：

\[
a_{\min}=\max(-a_{\mathrm{decel}},a_{t-1}-j_{\max}\Delta t),
\]

\[
a_{\max}=\min(a_{\mathrm{accel}},a_{t-1}+j_{\max}\Delta t).
\]

对每个 V39 lower/upper future witness：

```text
segment = [nominal emitted accel, corresponding reachable endpoint]
```

lower 和 upper 两段合并覆盖当前已有 `[a_min, a_max]`，但不放宽任何 limit。

V39 future schedule 的 edge1...H-1 不变；edge0 由 interval action 替换。若两个 basis 只在 edge0 不同，进行 exact dedup。

## 8.4 不做 outcome-selected grid

每个 interval basis 首先只检查三个 canonical seeds：

```text
0.0, 0.5, 1.0
```

三个 hard causal collision margins 用来生成：

- sign-change segment 的 secant boundary；
- 唯一 quadratic interpolant 的 interior margin maximizer；
- quadratic real boundary roots。

这些 action 只是 proposal。每个 proposal 必须重新：

1. clip 到 frozen accel/jerk interval；
2. 经过 unchanged controller projection；
3. 通过 current full physical tube certificate；
4. 通过 successor one-step shift certificate。

没有 outcome、未来 log、学习权重、任意网格、tuned margin 或 risk/progress scalarization。

## 8.5 必须产生新 actual emitted first action

V40 不允许“future tail 变了但当前发出的 action 没变”被当作 support expansion。

```text
new first target
AND full certificate
AND shift closure
```

三者必须同时成立。

如果 hard-certified record 的 first target 已经等于任一 existing nominal/end-point target，只保留诊断，不允许触发 policy。

## 8.6 Hard selection

没有新 first action：

```text
unchanged COWP fallback
```

有 hard set：

```text
min frozen COWP fallback score
→ min |first accel delta|
→ fewer nonnominal future edges
→ event-release on exact tie
→ deterministic identity
```

执行使用 selected projected trajectory/target/acceleration override。

## 8.7 无信息泄漏

V40 仅使用：

- current simulator state；
- current roadgraph；
- fixed checkpoint/current candidate bank；
- previous longitudinal acceleration；
- frozen controller limits；
- frozen causal-CV prediction；
- current/shift hard physical certificate。

不使用：

- Waymax `log_trajectory` future；
- future outcome；
- online mechanism GT；
- 未执行 policy 的真实反事实结果。

---

# 9. V40 预注册解释分叉

| 结果 | 结论 |
|---|---|
| 六项 Gate pass，new-first-action selection >0 | 支持 control-interval reachable support，进入 fresh37 |
| Gate pass，但 new-first-action≈0 | 不能归因于 V40；检查是否仅 nested V39 |
| new first actions 被真实执行，但 Gate fail | 停止 interpolation/schedule patch；转 interaction-aware reachable response envelope |
| interval support≈0 | fixed geometry/control family 是 ceiling；转 constrained reachable geometry/proposal construction |
| collision gain但 Kin/EP Gate fail | Archive；按 interval fraction、duration、downstream event拆因，不调权重 |
| `fccd...` 类 certificate false positive 增多 | causal-CV model mismatch；升级 robust interaction-aware response tube |

---

# 10. 后续明确禁止的方向

继续禁止：

1. certificate→planner-score argmin / CTU replacement；
2. outcome fallback weight；
3. outcome head hard shield；
4. 缩短 8 s horizon；
5. promotion max-prefix RVR；
6. Pareto tolerance / risk-weight 搜索；
7. BHOV comparator relaxation；
8. V3/V4/V5 horizon stacking；
9. unconditional commitment / fixed dwell；
10. hysteresis epsilon/margin；
11. profile AUC / horizon discount / weighted sum；
12. social+physical+utility 单 scalar；
13. 调 RCOT/BCOT threshold/budget；
14. 全局 retune common controller；
15. 同轮修改 accepted-path kinematics；
16. analytic target 验证前训练 neural head。

V39 后新增禁止：

17. 不再增加 hand-enumerated release schedules；
18. 不搜索任意 switch-time grid；
19. 不按结果调整 conflict first/last 定义；
20. 不放宽 full tube certificate 或 shift closure。

V40 后预注册新增：

21. 不按结果调 midpoint fraction；
22. 不增加更多 interpolation seeds；
23. 不按结果调 secant/quadratic root margin；
24. 不把 numerical resolution 当 algorithmic hyperparameter；
25. 若 interval intervention 非零但 Gate fail，直接转 interaction-aware reachable response envelope；
26. 若 interval support 为空，转 constrained reachable geometry/proposal construction，不再回 selector-family。

---

# 11. CCF-A 主线

V40 的 interval/root construction 本身不能包装成最终 novelty。

论文主线仍应是：

# **Orthogonal Option-Set Feasibility**

## Social axis

```text
natural roots
→ same-root RCOT
→ BCOT
→ protected-priority non-coercive certificate
```

不允许 ego safety 建立在其他 critical agents 的 natural low-burden option-set collapse 上。

## Physical axis

```text
actual controller state
→ control-reachable first-action set
→ full physical backup tube
→ shift closure
```

不允许 uncertified recovery 建立在 ego 自己 control-reachable backup-set collapse 上。

统一原则：

> **Safety must not be obtained through critical option-set collapse.**

V40 的研究角色是验证 physical axis 的真正 actionable object 是否应从“有限 trajectory schedule”升级成“连续 first-action reachable set”。

---

# 12. V40 代码验证

当前本地验证：

```text
V40 dedicated tests: 7/7 passed
V16.8.25→40 focused suite: 82/82 passed
Python compile: passed
launcher bash -n: passed
manifest hashes: passed
analyzer smoke: passed
fail-closed fresh37 probe: exit code 4
rollout artifacts after illegal promotion attempt: 0
```

专门覆盖：

- algebraic boundary proposal deterministic；
- interior first action 真实经过 controller realization；
- V39-certified decision exact nesting；
- interval-only new first-action support；
- non-new action 不可触发；
- shifted certificate 缺失时 fail closed；
- no logged future；
- priority Gate defaults。

本地没有用户服务器上的 Waymax dataset/index/checkpoint/GPU runtime，因此没有生成或虚构 V40 closed-loop outcome。

---

# 13. 下一步指令

```bash
cd COWP_v16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh make_ids

# 仅当 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh base_equivalence16_parallel2

bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh analyze_counterfactual48
```

到这里停止。

只有：

```text
preregistered_gate.shift_closed_first_action_viability_interval.pass == true
```

才运行：

```bash
PROMOTED_METHODS=cowp_shift_closed_first_action_viability_interval \
bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_40_SHIFT_CLOSED_FIRST_ACTION_VIABILITY_INTERVAL_CN.sh analyze_fresh37
```

fresh37 再通过才运行 historical exact200 development confirmation。

---

# 14. 下一轮最关键文件

```text
outputs/v16_8_40_shift_closed_first_action_viability_interval/
  equivalence16_cowp_vs_v16_8_29.json
  counterfactual48_v40_cowp_shift_closed_first_action_viability_interval_merged.json
  counterfactual48_v40_shift_closed_first_action_viability_interval_analysis.json
```

机制归因还需要 merged 中的 pooled diagnostics，尤其：

```text
nested_v39_selection_rate
interval_attempt_steps
interval_only_parent_support
new_first_action_selected_steps
selected_first_accel_fraction
boundary_source
full_safe → shift_closed retention
```
