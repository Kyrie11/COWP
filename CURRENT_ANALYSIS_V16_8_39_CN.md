# V16.8.38 结果可靠性、算法归因与 V16.8.39 设计

## 0. 结论先行

V16.8.38 `cowp_shift_closed_control_reachable_tube` 的结果通过独立可靠性审计，允许做算法归因；但它没有通过上一轮冻结的六项 Stage-1 GO Gate，必须归档，不能运行 `fresh37`。

独立审计共执行 **169 项 hard checks，169/169 通过**。V38 的算法结论不是 shard、merged、manifest、method alias、common-path regression 或 analyzer 错误造成的假失败。

V38 六项预注册 Gate 中通过 4 项、失败 2 项：

| Gate | 冻结 GO | V38 | 判断 |
|---|---:|---:|---|
| old RVR rescues retained | ≥5/10 | **0/10** | FAIL |
| old RVR induced avoided | ≥7/9 | **9/9** | PASS |
| 相对 COWP 净减少 collision | ≥3 scenes | **1** | FAIL |
| Kinematics 净退化 | ≤1 scene | **0** | PASS |
| paired mean EP Δ | ≥−0.05 | **−0.004746** | PASS |
| action-changing intervention | >0 | **55 steps** | PASS |

所以最终决策是：

```text
V16.8.38 = STAGE-1 FAIL / ARCHIVE / NO FRESH37
```

V38 policy implementation 不 promotion。但以下机制事实值得保留：

1. actual emitted action 上的 controller lifting 是真实有效分支，不是 dead code；
2. 完整 physical tube certificate 与 one-step shift closure 对 harmful recovery 具有高精度过滤作用；
3. 唯一 clean rescue 证明 constructive control support 不是原则上无效；
4. V38 的主要失败不是 shift closure 太严，而是只构造了三条过于粗糙的全时域纵向控制序列。

V16.8.39 因此不修改 selector、certificate、8 s contract 或 controller limits，而只升级 **constructive control-sequence support**：保留 V38 三条 baseline 序列，新增由当前 causal nominal collision-violation window 唯一确定的有限时长 endpoint lift，然后继续要求原封不动的 full physical certificate 和 shift closure。

新方法：

```text
cowp_conflict_window_control_reachable_tube
```

---

# 1. V16.8.38 可靠性审计

## 1.1 文件与协议完整性

独立核查结果：

- exact200、equivalence16、counterfactual48、fresh37 四组 manifest 数量、唯一性与 logical SHA256 全部匹配冻结值；
- counterfactual48 两个 shard 为 24+24、无 overlap，并集精确等于 48-ID manifest；
- equivalence16 两个 shard 为 8+8、无 overlap，并集精确等于 16-ID manifest；
- merged scenario rows 可由两个 shard 精确重建；
- CR、Collision、Offroad、Kinematics、EP 与 fallback 统计均可从逐场景/逐步字段零误差复算；
- checkpoint logical path/basename 在全部 shard 中一致；
- rollout horizon、Waymax standard metrics、JIT/reuse/prefilter、BCOT、NCF gate 和 action mode 协议一致；
- `mechanism_ground_truth_available_online=false`；
- V16.8.28 修复的 no-valid/emergency execution invariant 未复发；
- common-path equivalence 为 **16 scenes / 1120 fields / 0 mismatch**；
- 使用上传 V38 源码独立重跑 analyzer，和结果包 analyzer 递归比较 **0 mismatch，tolerance=1e-12**；
- V38 release hash 校验通过；
- V38 focused semantic/integrity suite 为 **68/68 passed**；
- 结果包没有 fresh37，符合 Stage-1 fail-closed 预注册协议。

完整机器审计：

`V16_8_38_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT_INDEPENDENT.json`

## 1.2 不阻断归因的证据边界

1. 结果 ZIP 未包含 checkpoint bytes，因此可以确认 logical checkpoint provenance 一致，但无法重新计算 `.pt` 文件内容 SHA256。
2. 当前 Waymax 采用 logged replay surrounding agents，足以支持冻结协议下的 ego closed-loop physical attribution，但不能作为强 counterfactual social-burden ground truth。
3. counterfactual48 已被 V30→38 多轮机制选择使用，只是 development-selected panel，不是 publication holdout。
4. 结果是单 checkpoint / 单 development panel；不能据此宣称统计显著或最终泛化。

## 1.3 一个不影响 Gate 的 analyzer 报表问题

V38 analyzer 的部分 conditional mechanism mean 使用“先按场景计算，再对有事件场景做宏平均”。因此它报告的：

```text
lifted_selection_rate_on_certified_steps = 94.1176%
```

不是所有 certificate steps 的 pooled ratio。独立逐步汇总得到：

```text
55 lifted selections / 56 certified steps = 98.2143%
```

这不改变任何 scenario outcome、paired comparison 或预注册 Gate。V39 analyzer 已将该类 mechanism count 改为 pooled aggregation，并保留明确分母。

---

# 2. V16.8.38 headline 与预注册 Gate

## 2.1 Headline

| Method | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 1/48 | 6/48 | 1.002512 |
| RVR | 33/48 | 1/48 | 9/48 | 0.823619 |
| V33 ROSH | 29/48 | 1/48 | 8/48 | 0.974647 |
| V35 CPOSH | 27/48 | 0/48 | 7/48 | 0.913987 |
| V36 Frontier | 34/48 | 0/48 | 8/48 | 0.978086 |
| V37 Returnability Bridge | 33/48 | 0/48 | 7/48 | 0.937350 |
| **V38 Shift-Closed Tube** | **33/48** | **1/48** | **6/48** | **0.997766** |

V38 相对 COWP：

- Collision：1 rescue / 0 induced，净减少 1；McNemar exact p=1.0；
- Offroad：0 rescue / 0 induced；
- Kinematics：0 rescue / 0 induced；
- paired mean EP Δ = −0.0047458；
- EP bootstrap 95% interval = [−0.011919, +0.001463]；
- 只有 9/48 scenes 的 EP 非零变化。

这组结果说明 V38 很保守、没有造成 headline physical harm，但不满足预注册的 recovery recall 和净 collision gain。

## 2.2 严格 Gate 决策

V38 只保留 **0/10** 个 historical RVR rescues，虽然避免 **9/9** 个 historical RVR-induced collisions。它是非常典型的：

```text
high precision / extremely low recall
```

不能因为无 induced、无 Kinematics regression、EP 几乎不变，就把“old rescue ≥5/10”或“净 collision gain ≥3”删掉。V38 必须 archive。

---

# 3. V38 机制归因

## 3.1 分支真实执行，并且 constructive lifting 是主导来源

48×80=3840 policy steps 中：

| Mechanism count | 数值 |
|---|---:|
| probe steps | 2695（70.1823%） |
| certified steps | 56（1.4583% all steps；2.0779% probes） |
| action-change steps | 55（1.4323%） |
| lifted selected | 55/56（98.2143% pooled） |
| total parent geometries | 98045 |
| distinct parent action classes | 86880 |
| generated hypotheses | 260640 |
| unique first actions | 161472 |

因此 V38 不是 nominal policy 的别名。新增的 lower/upper reachable endpoint 实际产生了原 nominal tube 没有的 hard-certified actions。

shift-closed witnesses 的构成：

| Witness type | Count |
|---|---:|
| nominal | 3 |
| lower-all | 425 |
| upper-all | 58 |
| total shift-closed | 486 |
| lifted-only parent support | 481 |

这证明“actual controller lifting”是 V38 真正激活的机制。

## 3.2 shift closure 不是 certificate 稀疏性的主因

在全部 probes 中：

- full physically safe witnesses：522；
- shift-closed witnesses：486；
- retention：486/522 = **93.1034%**；
- shift closure 只额外淘汰 36 个 full-safe witnesses。

所以不能把 V38 的 1.46% certificate step rate 归因于“一步 shift closure 太严格”。真正的稀疏发生在它之前：

```text
260640 generated hypotheses
→ only 522 full-safe witnesses
→ 486 shift-closed witnesses
```

## 3.3 真正结构问题：三条全时域控制序列过于贫乏

V38 对每个 parent geometry 只构造：

```text
NOMINAL:    每一拍执行普通 controller
LOWER_ALL:  80 个 edge 都取 accel/jerk 可达区间 lower endpoint
UPPER_ALL:  80 个 edge 都取 accel/jerk 可达区间 upper endpoint
```

这种 family 对有限冲突窗口缺少必要的“先干预、后释放”结构：

- `NOMINAL` 可能无法跨越冲突窗口；
- `LOWER_ALL` 能避开冲突，但会在冲突结束后继续最大制动，造成 progress basin；
- `UPPER_ALL` 同理可能长期保持不必要加速；
- 当前 bank 明明有约 32 个不同 action classes/probe，但 control sequence family 只有 3 种时间结构。

所以当前不是几何 parent 数量不够，也不是 selector 还不够复杂，而是：

> **existing geometry support 没有被足够丰富、仍可证书化的 control sequence support 所覆盖。**

## 3.4 clean constructive-support rescue

场景：

```text
6992366c5c998d00
```

| Metric | COWP | V38 |
|---|---:|---:|
| Collision | 1 | 0 |
| Offroad | 0 | 0 |
| Kinematics | 0 | 0 |
| EP | 0.334336 | 0.387596 |

该场景：

- 5 个 tube-certified steps；
- 5 个实际 action changes；
- 全部选择 lifted support；
- Collision 被救回；
- 没有 failure conversion；
- EP Δ=+0.053261。

所以不能得出“constructive tube support 本身错误”的结论。正确结论是：

> **support construction 有高精度正例，但当前 control-sequence family 的 coverage/recall 远远不够。**

## 3.5 进度损失也指向 all-horizon envelope

V38 最大的 EP regressions 集中在发生较长 lower-envelope intervention 的场景，例如：

- `1e015170763b9002`：ΔEP≈−0.103393，14 action changes；
- `da87744bcf7613db`：ΔEP≈−0.083955；
- `7721ff4800156886`：ΔEP≈−0.083527。

虽然总体 EP Gate 通过，但这些场景说明：让 endpoint policy 持续整个 8 s horizon 会把“有限时间避免冲突”错误地扩展为“无限制保持极端控制”。这也是 V39 应加入 event release，而不是继续 lower/upper-all 的直接证据。

---

# 4. 本轮真正成功、失败与 promotion 判断

## 4.1 值得保留/升级

### A. Actual-emitted-action control lifting

保留。它和 nominal exact-reachability 不同：不是把需要 projection 的 option 删除，而是在冻结 controller envelope 中真实构造可执行动作。

### B. Full physical tube certificate

保留。它在完整冻结 horizon 上同时要求：

```text
finite
∧ frozen roadgraph-safe
∧ frozen causal-CV collision-safe
∧ Waymax-aligned kinematics-safe
```

不能为了增加 certificate rate 放宽。

### C. One-step shift closure

保留。它不是主要稀疏源，而且为“下一次真实 replanning 后仍属于同类 feasible tube”提供最小 closed-loop consistency。

### D. Frozen COWP fallback preference inside hard set

保留。hard feasibility set 内仍以成熟 COWP fallback score 作为第一 utility ordering，避免重新设计 recovery scalar。

## 4.2 应归档

### A. V38 policy implementation

Gate FAIL，不能 promotion。

### B. 三条 constant all-horizon schedule 作为完整 support family

它们可以作为 V39 nested baseline 和 ablation，但不能继续假设足以覆盖 zero-conventional recovery。

### C. 继续调 selector/certificate

V36 已否定 same-bank semantic selector/frontier；V37 exact returnability 稀疏；V38 又显示当前 bottleneck 在 full-safe support 生成前。继续调 comparator、score 或 release threshold缺乏证据。

## 4.3 本轮不 promotion 的内容

- 不 promotion `cowp_shift_closed_control_reachable_tube`；
- 不把 1-scene collision gain写成统计性成功；
- 不把 9/9 induced avoided 当作覆盖 recall Gate 的理由；
- 不把 V38 的 Waymax execution adapter包装成论文 novelty。

---

# 5. dominant bottleneck 的进一步收紧

完整证据链：

```text
V28  online conventional-feasible support collapse
→ V29 dynamic collision-side collapse，而非 roadgraph；same bank 可 rescue
→ V30 actual emitted action 的 successor option set 有独立信息
→ V31 stateless hybrid switching 可制造 failure
→ V32 horizon stacking / unconditional commitment 失败
→ V33 semantic recovery-option spectrum 有信息
→ V34 nominal exact-realizability 不是 executable support
→ V35 control-projected option spectrum 有强 recovery signal
→ V36 broader existing-bank semantic frontier 仍失败
→ V37 exact return-to-conventional 高精度但极稀疏
→ V38 controller-lifted hard tube 有 clean positive，但三条全时域控制序列 support recall 极低
```

当前 P0 应定义为：

# **Finite-Duration Control-Reachable Backup Support under Zero-Conventional Collapse**

更适合论文的概括：

# **Constructive Control-Sequence Support for Shift-Closed Physical Feasibility**

当前最该解决的是这个 P0，而不是：

- 重建 compact-5k；
- 修改 RCOT/BCOT/certificate；
- map/Frenet proposal expansion；
- accepted-path kinematics；
- selector weights；
- return-to-original-bank depth。

accepted-path execution viability 仍是真实 secondary bottleneck，但继续独立，避免破坏 recovery attribution。

---

# 6. 模型各层成熟度

| Layer | 当前判断 | V39 原则 |
|---|---|---|
| compact-5k data/labels/checkpoint contract | Mature | **Freeze** |
| Natural roots | Mature | **Freeze** |
| same-root RCOT | Strong/Mature | **Freeze** |
| BCOT | Strong/Mature | **Freeze** |
| Protected-priority hard certificate | Mature | **Freeze** |
| Certificate-compatible set preservation | Mature；CTU 已有负消融 | **Freeze** |
| Outcome head/settings | Diagnostic-only | **Freeze** |
| 8 s conventional contract | Stable attribution contract | **Freeze** |
| V27/V28 integrity fixes | Solved | **Freeze** |
| Candidate geometry families | 当前无重建证据 | **Freeze in V39** |
| Common controller limits | Mature interface | **Freeze** |
| RVR/SOV/BHOV/THOP | policy negatives / partial signals | Archive implementations |
| ROSH/CPOSH representations | 有正信号 | 吸收 insight |
| V36 semantic frontier | Gate FAIL | Archive |
| V37 exact returnability bridge | Gate FAIL；高精度低召回 | Archive |
| V38 hard tube certificate/shift closure | 高精度 evaluator | **Retain** |
| V38 all-horizon N/L/U support family | 不成熟/低召回 | **V39 upgrade** |
| Conflict-window control sequence support | P0 / 未验证 | **V39 主攻** |
| Accepted-path execution viability | Secondary | 后续独立 |
| Interaction-aware reachable support | Long-term ceiling | V39 fail 后进入 |

---

# 7. 模型下一步应该重点“学”什么

social axis 已经能够较好回答：

> ego 是否通过压垮其他 protected agents 的 natural low-burden choices 来获得所谓安全？

physical axis 还没有学会：

> 在 zero-conventional 状态下，哪一段有限时长的真实可达控制 intervention 能跨过当前冲突窗口，同时在冲突结束后释放，保留一个完整、shift-closed、可继续 replanning 的 physical backup tube？

当前缺失的不是再一个 collision probability，而是：

```text
(state, controller memory, parent recovery geometry)
→ causal conflict window
→ finite-duration reachable control sequence family
→ full physical tube feasibility
→ one-replan shift closure
```

V39 仍先用解析构造验证 target。只有这个对象获得稳定 outcome evidence 后，才有理由训练 proposal/tube generator 或 neural viability head。

---

# 8. 后续明确禁止的算法方向

继续禁止：

1. 改 RCOT/BCOT threshold、budget、natural-root 或 protected-priority certificate；
2. certificate→planner-score argmin/CTU replacement；
3. outcome fallback weight 或 outcome head hard shield；
4. 缩短 8 s horizon来人为制造 conventional candidate；
5. 直接 promotion RVR、SOV/BHOV/THOP/ROSH/CPOSH/frontier/returnability/V38 implementations；
6. risk/collision/progress/kinematics/profile-AUC/horizon-discount scalarization；
7. 固定 dwell、minimum commitment、hysteresis epsilon 或 tuned release threshold；
8. 继续 V2/V3/V4 returnability tree 或 horizon stacking；
9. 用 nominal first-waypoint exact reachability定义 executable；
10. 放宽 full physical certificate、shift closure 或 controller limits；
11. 全局 retune common controller；
12. 本轮同时修改 accepted-path kinematics；
13. 当前扩 map/Frenet primitives或重建 compact-5k；
14. 在 analytic support target 未验证前训练 neural reachable-tube head；
15. 用 logged-replay Waymax 声称强 counterfactual social-burden causality。

V39 新增一条预注册禁区：

> **禁止在看到 V39 结果后搜索任意 switch time grid、增加更多手工 schedule 或调整 first/last conflict release 定义。**

如果这组最小 event-derived family 仍 FAIL，应停止 finite hand-enumerated schedule patch，进入真正 constrained reachable-set optimization / interaction-aware response envelope。

---

# 9. V16.8.39：Conflict-Window Control-Reachable Recovery Tube

## 9.1 介入范围

V39 只在：

```text
full conventional set == empty
AND valid candidate exists
```

时介入。

以下 common paths 不变：

- accepted/certified COWP path；
- conventional fallback 存在的 path；
- no-valid emergency；
- social certificate；
- common controller limits；
- 8 s conventional definition；
- candidate geometries与数据/checkpoint。

## 9.2 Parent support

复用 V38：

1. 优先 `valid & roadgraph_safe` parent pool；为空时使用全部 valid parents，生成后仍必须重新通过 full roadgraph audit；
2. 排除 PAD；
3. 按 `(semantic macro, actual emitted first target)` 去重；
4. 同 macro 不同物理动作可以保留，相同 emitted action 不因多个近似 candidate 重复计数。

## 9.3 因果冲突窗口

对每个 parent 的 **nominal controller-realized tube**，使用冻结的 sampled causal-CV collision model 计算：

```text
E_i = {h : nominal realized tube 在 edge h 违反 collision contract}
```

若 `E_i` 非空：

```text
h_first = min E_i
h_last  = max E_i
```

这些时刻只来自当前 observed state、当前 parent、frozen CV prediction 和 frozen collision audit，不读取 Waymax logged future 或真实 outcome。

## 9.4 Nested control schedule family

V39 完整保留 V38 baseline：

```text
NOMINAL
LOWER_ALL
UPPER_ALL
```

并新增四个 event-derived schedules：

```text
LOWER_TO_FIRST_CONFLICT
UPPER_TO_FIRST_CONFLICT
LOWER_TO_LAST_CONFLICT
UPPER_TO_LAST_CONFLICT
```

其中 lower/upper 表示每一拍使用已有 accel/jerk reachable interval 的 endpoint：

```text
lo_t = max(-max_decel, a_{t-1} - max_jerk·dt)
hi_t = min( max_accel, a_{t-1} + max_jerk·dt)
```

例如：

```text
LOWER_TO_LAST_CONFLICT:
  edge 0 ... h_last: lower endpoint
  edge h_last+1 ... H-1: unchanged nominal controller
```

first/last 冲突相同或 schedule 完全相同时做 exact dedup，避免虚增 support。

没有：

- arbitrary switch grid；
- tuned conflict margin；
- minimum intervention duration；
- learned release time；
- risk/progress weight。

## 9.5 Hard full-tube certificate

每条 realized tube 必须在完整冻结 horizon 上同时满足：

```text
finite
∧ frozen roadgraph-safe
∧ frozen causal-CV collision-safe
∧ Waymax-aligned inverse-dynamics kinematics-safe
```

它仍被标记为 uncertified recovery backup，不会重新标成 conventional-safe、NCF 或 social-certified。

## 9.6 One-step shift closure

执行实际 first target 后：

1. 用 causal successor state；
2. 将 realized tube 左移一个 edge；
3. 追加 constant-velocity terminal edge；
4. 携带 emitted longitudinal acceleration 作为 successor controller memory；
5. event-release schedule 同步左移，末端回到 nominal；
6. all-horizon lower/upper baseline 的末端继续保持原 endpoint，确保 V38 精确嵌套；
7. 从 successor 重新投影；
8. 再次要求完整 physical certificate。

这不是更深的 future-state tree，也不访问 logged future。

## 9.7 Hard-set selection

无 shift-closed witness：

```text
unchanged COWP least-coercive-valid fallback
```

有 witness：

```text
hard shift-closed set
→ minimum frozen COWP fallback score
→ minimum |first accel deviation|
→ fewer nonnominal edges
→ nominal, then event-release, then all-horizon endpoint on exact tie
→ deterministic parent/policy id
```

`fewer nonnominal edges` 只在相同成熟 COWP preference 与相同 first-action distortion 后做 deterministic witness tie-break，不会让不满足 hard certificate 的动作进入集合。

选中的 projected trajectory、first target 和 acceleration 通过 explicit execution override 真正发给 Waymax。

## 9.8 新 diagnostics

V39 记录 pooled counts：

- parent nominal conflict rate；
- first/last nominal conflict edge；
- event-release hypotheses；
- event-release full-safe/shift-closed witnesses；
- lower/upper event-release witness counts；
- event-release-only parent support；
- selected event-release rate；
- selected release edge 与 nonnominal duration；
- nested V38 baseline support；
- full-safe→shift-closed retention；
- action-change、collision margin、fallback-score delta。

以下只用于机制归因，不增加事后第七 Gate：

```text
event_release_only_parent_support > 0
selected_event_release > 0
```

若 outcome Gate 通过但这两项约为 0，只能归因于 V38 nested baseline/重实现，不能声称 conflict-window support 成功。

---

# 10. V39 预注册分叉

Stage-1 继续原封不动使用六项 Gate：

| Gate | GO |
|---|---:|
| old RVR rescues retained | ≥5/10 |
| old RVR induced avoided | ≥7/9 |
| 相对 COWP 净减少 collision | ≥3 scenes |
| Kinematics 净退化 | ≤1 scene |
| paired mean EP Δ | ≥−0.05 |
| action-changing intervention | >0 |

分叉：

| V39 结果 | 决策 |
|---|---|
| Gate pass，event-release-only support 与 selection 均非零 | 支持 finite conflict-window control support，进入 fresh37 |
| Gate pass，但 event-release 支持/选择≈0 | outcome 可 promotion，但不能归因于新 support family |
| certificate rate 仍≈V38，event-release-only≈0 | 当前 event-derived finite family不足；转 constrained reachable-set optimization |
| event-release support/selection 非零但 Gate fail | analytic CV conflict window或有限 schedule family存在 model mismatch；停止手工 patch |
| collision改善但 kinematics/EP fail | Archive；按 release duration、first accel、downstream event timing拆根因，不调权重 |
| overall fail | 停止 finite hand-enumerated schedules，升级 interaction-aware reachable support / response envelope |

只有 Stage-1 analyzer：

```text
preregistered_gate.conflict_window_control_reachable_tube.pass == true
```

才允许 launcher 启动 fresh37。

---

# 11. 代码验证

V39 当前交付验证：

- V39 dedicated tests：**7/7 passed**；
- V16.8.25→39 focused semantic/integrity suite：**75/75 passed**；
- Python compile：passed；
- launcher `bash -n`：passed；
- exact200/equivalence16/counterfactual48/fresh37 manifest hashes：passed；
- method registration / metrics aggregation / execution override：tested；
- V38 nested schedule identity：tested；
- conflict-window exact dedup：tested；
- event-release-only support construction：tested；
- first-target controller equivalence：tested；
- missing shifted certificate fail closed：tested；
- analyzer pooled aggregation smoke：passed；
- direct fresh37 without Stage-1 pass：fail closed；
- conventional-safety bypass grep：none；
- V39 helper does not consume logged future：confirmed。

完整 repository `pytest -q` 仍在历史测试 `tests/test_v16_8_29_recovery_viability.py` collection 阶段导入已删除的 `_recovery_bridge_viability_mask` 时停止。完全相同错误已在用户上传的未修改 V38 tree 中复现，因此不是 V39 regression；没有为了虚报 full-suite green 而恢复已归档 API。

本地环境没有服务器 Waymax dataset/index/checkpoint/GPU runtime，因此没有伪造 V39 closed-loop outcome。真实 Gate 必须由下一轮执行得到。

---

# 12. 运行指令

```bash
cd COWP_v16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh make_ids

# 仅当 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh analyze_counterfactual48
```

到这里停止。

仅 Gate pass 后：

```bash
PROMOTED_METHODS=cowp_conflict_window_control_reachable_tube \
bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_39_CONFLICT_WINDOW_CONTROL_REACHABLE_TUBE_CN.sh analyze_fresh37
```

fresh37 再通过，才运行 historical exact200 development confirmation。

---

# 13. CCF-A 主线判断

V39 的 conflict-window schedule 本身不应被包装成最终 novelty。当前值得维护的论文结构仍是：

# **Orthogonal Option-Set Feasibility**

Social axis：

> ego safety 不能建立在其他 protected agents 的 natural low-burden option-set collapse 上。

Physical axis：

> uncertified recovery 不能建立在 ego 自身 control-reachable、shift-closed backup option-set collapse 上。

统一原则：

> **Safety must not be obtained through critical option-set collapse.**

V39 的角色是验证 physical axis 中“有限冲突窗口上的 constructive control-sequence support”是否是正确 actionable object。最终 CCF-A claim 仍需要全新 final unseen set、多 independent seeds、paired uncertainty，以及对 strong social causality 的 reactive-agent/human-audited stress protocol。
