# COWP V16.8.37 结果审计与 V16.8.38 算法设计

## 0. 执行结论

本轮严格遵循“先可靠性、后算法归因”的顺序。

1. **V16.8.37 结果通过可靠性审计。** 独立审计共 71 项 hard checks，71/71 全部通过，没有发现会阻断算法归因的代码、分片、汇总、manifest、协议或 common-path regression。
2. **V16.8.37 没有通过冻结的 Stage-1 预注册 GO Gate。** 六项条件只通过三项，因此 `cowp_recourse_returnability_bridge` 必须 archive，不能运行 fresh37，也不能调整阈值救结果。
3. **V37 不是“代码完全没工作”，而是 high precision / unusably low recall。** 它避免了 9/9 个历史 RVR-induced collision，却只保留 1/10 个历史 RVR rescue；3840 个 policy steps 中有 144 次 returnability probe、7 次 entry、2 次真实 bridge execution，但只净减少 1 个 COWP collision。
4. **V37 的唯一 collision rescue 不是完整成功。** `fccd9a25a2a57a73` 从 collision+offroad 转成了 kinematics failure，EP 由 3.67633 降到 0.51512，单场景 ΔEP=-3.16121。剔除该场景，其余 47 scenes 的平均 ΔEP≈+0.00071。
5. 当前 dominant bottleneck 应收紧为：

   **Constructive Control-Reachable Backup Support / Feasibility-Basin Representation**

   不是继续调 selector、增加 bridge depth、放宽 restoration threshold，或把 exact return-to-conventional 当唯一终点。
6. V16.8.38 已落地为：

   **Shift-Closed Control-Reachable Recovery Tube（SCRT）**  
   方法名：`cowp_shift_closed_control_reachable_tube`

   它不再只从 fixed bank 中寻找“两拍内回到 full-conventional set”的稀疏见证，而是在已有 semantic parent geometry 上，使用真实 controller 的可达纵向控制端点构造显式 recovery tubes，并要求完整物理证书与 one-step shift closure。

---

# 一、V16.8.37 可靠性审计

## 1.1 审计结论

`reliability_verdict = PASS`，71/71 hard checks 全部通过。

本轮检查覆盖：

- exact200、equivalence16、counterfactual48、fresh37 四组冻结 manifest 的数量、唯一性、逻辑 SHA256 与代码内 reference manifest 一致；
- counterfactual48 为 24+24 shards，互不重叠，并集精确覆盖冻结的 48 IDs；
- equivalence16 为 8+8 shards，互不重叠，并集精确覆盖冻结的 16 IDs；
- merged scenario rows 与 shard rows 精确一致；
- CR、Collision、Offroad、Kinematics、EP 可从逐场景 rows 零误差重算；
- fallback step rate 可从 3840 个 policy steps 零误差重算；
- checkpoint logical path/basename、Waymax metric set、JIT/reuse/prefilter、BCOT/threshold 等运行协议在 shards 间一致；
- `mechanism_ground_truth_available_online=false`，没有把结果时的机制标签作为在线输入；
- V16.8.28 的 no-valid / emergency-action execution invariant 仍成立；
- equivalence16 独立验证为 **16 scenes / 1120 fields / 0 mismatch**；
- V37 analyzer 独立重跑，与上传 analyzer JSON 逐字段递归比较为 0 mismatch，tolerance=1e-12；
- V16.8.25→37 focused semantic/integrity sanity 为 61/61 passed；
- release SHA256 entries 校验通过；
- V37 method registration、explicit no-conventional branch、logged-future rejection 均存在。

完整机器审计：`V16_8_37_RESULT_RELIABILITY_AND_ATTRIBUTION_AUDIT_INDEPENDENT.json`。

## 1.2 不阻断归因、但必须明确的证据边界

1. 上传包没有 checkpoint bytes，因此 checkpoint identity 只能通过一致 logical path/basename 与协议 provenance 验证，不能独立重算 `.pt` 文件 SHA256。
2. Waymax 当前协议对非 ego agents 使用 logged replay。它可以支持冻结协议下的 closed-loop physical attribution，但不能作为 counterfactual social burden ground truth。
3. counterfactual48 已被多轮机制选择使用，只能视为 development-selected panel，不是 publication holdout。
4. 没有 fresh37 结果是正确的停止行为，因为 V37 Stage-1 Gate 已失败。

因此，本轮可以做算法归因，但不能把 48-scene 结果写成论文最终统计结论。

---

# 二、严格按冻结的六项 Stage-1 GO 条件判断 V37

冻结条件未做任何修改：

| 预注册条件 | GO | V37 | 结论 |
|---|---:|---:|---|
| old RVR rescues retained | ≥5/10 | **1/10** | FAIL |
| old RVR induced avoided | ≥7/9 | **9/9** | PASS |
| 相对 COWP 净减少 collision | ≥3 scenes | **1** | FAIL |
| Kinematics 净退化 | ≤1 scene | **+1** | PASS |
| paired mean EP Δ | ≥−0.05 | **−0.065162** | FAIL |
| intervention | >0 | **8/3840 steps = 0.2083%** | PASS |

所以：

> **`cowp_recourse_returnability_bridge` = Stage-1 FAIL / Archive。**

不能把 rescue threshold 从 5/10 改成 1/10，不能把 collision 净改善阈值从 3 改成 1，也不能因为 EP regression 集中在单个场景就事后删除该场景或重定义 Gate。

Headline：

| Method | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 1/48 | 6/48 | 1.002512 |
| RVR | 33/48 | 1/48 | 9/48 | 0.823619 |
| V33 ROSH | 29/48 | 1/48 | 8/48 | 0.974647 |
| V35 CPOSH | 27/48 | 0/48 | 7/48 | 0.913987 |
| V36 Frontier | 34/48 | 0/48 | 8/48 | 0.978086 |
| **V37 RRB** | **33/48** | **0/48** | **7/48** | **0.937350** |

V37 相对 COWP：

- collision：1 rescue / 0 induced，净改善 1；
- offroad：1 rescue / 0 induced；
- kinematics：0 rescue / 1 induced，净退化 1；
- paired mean EP Δ=-0.065162；
- McNemar 对 1/0 discordance 没有统计支持，不能宣称 collision improvement。

---

# 三、V37 的机制归因

## 3.1 V37 确实介入，不是 dead code

聚合 48×80=3840 个 policy steps：

| 机制事件 | 数量 |
|---|---:|
| returnability probes | 144 |
| recovery switches | 8 |
| bridge entries | 7 |
| direct entries | 1 |
| pending steps | 6 |
| actual bridge executions | 2 |
| bridge aborts | 1 |
| strict returnability-dominance probes | 7 |

所以失败不能解释为 implementation 从未运行。

## 3.2 精度高，但召回几乎归零

V37 在历史 RVR counterexamples 上：

- 9/9 old RVR-induced collision 全部避免；
- 只有 1/10 old RVR rescue 被保留；
- 其余 9 个 old rescue scenes 即使累计发生大量 probes，也没有 entry；
- 唯一保留 rescue 的场景仍转化为 kinematics failure 和 severe progress loss。

这说明 returnability witness 的约束方向不是随机的。它确实能强力过滤 harmful recovery，但作为 recovery policy 的 admissible set 太稀疏，几乎等价于“不恢复”。

## 3.3 “有约 35 个 action classes”与“有 restoring support”完全不是一回事

V37 每次 probe 平均：

| Diagnostic | Base | RVR |
|---|---:|---:|
| action classes available | 34.81 | 35.06 |
| action classes evaluated | 34.17 | 34.22 |
| restoring macro witnesses / probe | 0.278 | 0.306 |
| direct restore probes | 3/144 | 4/144 |

也就是说，大量 nominal/uncertified action classes 并不等于存在一个能在一拍或一个新 replanning 后**精确回到完整 conventional set**的见证。

V37 的 returnability target 在最关键的 zero-conventional operating regime 中，绝大多数时候是：

```text
R_base = ∅
R_alt  = ∅
```

这不是“需要再比较更多 score”的问题，而是 terminal target 本身造成的 near-empty support。

## 3.4 唯一 rescue 暴露的是 failure conversion，而不是完整恢复

`fccd9a25a2a57a73`：

- COWP：collision=1、offroad=1、kinematics=0、EP=3.67633；
- V37：collision=0、offroad=0、kinematics=1、EP=0.51512；
- ΔEP=-3.16121；
- first kinematics step=51。

该场景贡献了全部 headline EP regression。剔除它后，其余 47 scenes 平均 ΔEP≈+0.000711。

正确解释是：

> V37 找到了一条足以改变 collision/offroad outcome 的回避路径，但没有形成兼顾 execution feasibility 与 useful progress 的稳定 recovery basin。

不能把它宣传成成功 collision rescue，也不能简单为它增加一个 kinematics penalty。历史 V33/V34 已经证明 downstream execution failure 不一定发生在 entry action 当拍。

## 3.5 V37 保留的真实正信号

V37 policy implementation 不 promotion，但以下机制事实值得保留：

1. **Returnability/terminal semantics 必须显式存在。** V36 已证明 option richness 不等于可退出性；V37 进一步证明直接问“是否能回到 conventional set”具有很高过滤精度。
2. **实际 emitted action 与新 replanning edge 必须进入物理证据。** 不能再用 nominal candidate waypoint 或原 trajectory 的第二个 waypoint冒充 closed-loop recourse。
3. **硬可行性优先于 scalar utility。** 9/9 harmful RVR cases 被避免，说明 hard witness 能拒绝部分 current-prefix/option-spectrum false positives。
4. **但 exact full-conventional restoration 只能保留为 evaluator/diagnostic，不能继续作为唯一 entry certificate。** 它太稀疏，无法提供 recovery recall。

---

# 四、V16.8.28→37 的证据链与 dominant bottleneck

当前证据链已经足够收敛：

```text
V28: online conventional-feasible support collapse
  ↓
V29: collapse 主要来自 dynamic collision side，不是 roadgraph；same bank 可 rescue
  ↓
V30: actual emitted action → successor option set 有独立信息
  ↓
V31: stateless hybrid switching 会制造新的 closed-loop failure
  ↓
V32: horizon stacking 与 unconditional commitment 都失败
  ↓
V33: semantic recovery-option spectrum 明显有信息
  ↓
V34: nominal exact-realizability 不是 executable option，hard filtering 造成 false negatives
  ↓
V35: control-projected spectrum 增加 rescue recall，但 endpoint policy 有 false positives/EP basin
  ↓
V36: broader existing-bank semantic frontier 被真实利用仍失败；richness ≠ returnability
  ↓
V37: exact return-to-full-conventional 高精度但极端稀疏；recourse certificate ≠ constructive support
```

因此当前 P0 不是：

- 再调 selector；
- 再比较 base / RVR / more macros；
- 再加 V2/V3/V4 lookahead；
- 放宽 direct/indirect restore 条件；
- 调 profile AUC、prefix、risk 或 EP 权重；
- 立即重建 compact-5k 数据；
- 立即做 map/Frenet proposal expansion。

当前最准确的 dominant bottleneck 是：

## **Constructive Control-Reachable Backup Support under Zero-Conventional Collapse**

更论文级、可推广的表述：

## **Shift-Closed Control-Reachable Physical Feasibility**

核心问题是：

> 当 nominal fixed bank 的 full-conventional set 为空时，系统是否能在不放宽车辆约束、不读取未来真值的条件下，构造一条属于当前真实控制可达集、完整满足 physical contract，并在执行一拍后仍保留同类可行后继的 backup tube？

这与 V37 的区别是：

- V37 在 existing actions 上寻找“回到原 conventional bank”的稀疏路径；
- V38 在 existing semantic geometry 周围构造**真实 controller-reachable support**，并直接对 recovery tube 做 shift closure。

---

# 五、模型各层成熟度与保护策略

| Layer | 当前成熟度 | V38 原则 |
|---|---|---|
| compact-5k data / label contract | Mature | **Freeze** |
| Natural roots | Mature | **Freeze** |
| RCOT same-root transport | Strong / Mature | **Freeze** |
| BCOT | Strong / Mature | **Freeze** |
| Protected-priority hard certificate | Mature | **Freeze** |
| Certificate-compatible set preservation | Mature，CTU 已给负消融 | **Freeze** |
| Outcome head | Diagnostic-only | **Freeze** |
| 8 s conventional contract | Stable attribution contract | **Freeze** |
| V27 conventional integrity | Solved | **Freeze** |
| V28 no-valid execution integrity | Solved | **Freeze** |
| Common controller | Mature interface | **Freeze limits and nominal path** |
| RVR policy | Negative | Archive；只作历史 counterexample source |
| SOV signal | Positive mechanism signal | Retain as evidence |
| BHOV / THOP | Negative | Archive；禁止 horizon stacking |
| Unconditional commitment | Policy negative | Archive |
| ROSH semantic spectrum | Positive representation | Retain as evidence |
| EOSH exact-realizability filter | Negative | Archive |
| CPOSH observable | Positive | Retain as evidence |
| V36 semantic frontier | Gate FAIL | Archive |
| V37 exact returnability bridge | Gate FAIL，高精度/低召回 | Archive policy；保留 terminal evaluator insight |
| **Control-reachable support construction** | **P0 / immature** | **V38 主攻** |
| Accepted-path execution viability | Secondary bottleneck | 继续独立，不在 V38 修改 |
| True interaction-aware proposal/reachable support | Long-term ceiling | V38 fail 后升级 |

前半条 social feasibility 主线不能因为 physical recovery 尚未成熟而反向改动。

---

# 六、本轮以后继续禁止的修改方向

除历史 `ALGORITHM_CHANGELOG.md` 已记录的禁止项外，本轮新增或再次确认：

1. 不得把 V37 的 old-rescue threshold 从 5/10 改成 1/10，或把 EP Gate 从 −0.05 改成 −0.07。
2. 不得删除 `fccd...` 后重算 Gate 来声称 V37 成功。
3. 不得把 direct restoration 放宽成“conventional candidate count 增加”“prefix 更长”或其他 proxy；这会退回 V29/V35 已否定的 representation。
4. 不得继续做 2/3/4-step bridge depth。它只是用更昂贵方式重演 THOP horizon stacking。
5. 不得给 returnability macro count、profile area、time-to-return、EP 或 collision 加权形成 scalar score。
6. 不得用 fixed dwell、minimum commitment、hysteresis epsilon 或 hand-tuned exit margin补救稀疏 witness。
7. 不得继续 base-vs-global-RVR 或 same-bank selector family；V36/V37 已完成该分支的否证。
8. 不得把 nominal first-waypoint exact reachability重新定义为 executable support；V34 已 clean negative。
9. 不得放宽 common controller limits或全局 retune controller；V38 只能利用原 controller reachable set。
10. 不得改 RCOT/BCOT threshold、budget、natural roots、set-preservation frontier、8 s conventional contract。
11. 不得在 V38 同时修改 accepted/certified-path kinematics，避免破坏 recovery attribution。
12. 不得在 analytic control-reachable tube target 未被验证前训练 neural viability/returnability head。
13. 不得把 logged-replay Waymax 结果解释为强 counterfactual burden causality。
14. 不得把“有多个 backup tube”“shift closure”“safety filter”单独包装成 CCF-A novelty。

---

# 七、V16.8.38：Shift-Closed Control-Reachable Recovery Tube

方法名：

```text
cowp_shift_closed_control_reachable_tube
```

## 7.1 介入范围

只在：

```text
full conventional set == empty
AND valid candidate exists
```

时介入。

certificate/accepted path、存在 conventional fallback 的路径、no-valid emergency 路径全部保持 V37/COWP common behavior。

## 7.2 Parent geometry：冻结 candidate family，只使用已有语义结构

V38 不重建数据，不增加 map/Frenet primitive，不生成新的 lateral semantic family。

Parent pool：

1. 若有 `valid & roadgraph_safe`，使用该 pool；
2. 否则使用全部 valid；
3. 排除 PAD；
4. 按 `(semantic macro, actual emitted target)` 去重。

这样同一 macro 中物理动作不同的 candidate 不会被错误合并，同一实际动作也不会因为多个 macro label 虚增 support。

## 7.3 Constructive control lifting

对每个 parent geometry 构造三条纵向 control-realized tube：

```text
mode  0: unchanged nominal controller
mode -1: every edge uses current accel/jerk reachable interval lower endpoint
mode +1: every edge uses current accel/jerk reachable interval upper endpoint
```

这里的 lower/upper：

```text
lo_t = max(-max_decel, a_{t-1} - max_jerk * dt)
hi_t = min( max_accel, a_{t-1} + max_jerk * dt)
```

不是新 threshold，也不是放宽约束，而是已有 controller reachable interval 的端点。

所有 tube：

- 使用相同 stateful acceleration memory；
- 使用相同 yaw-rate / max-delta-yaw projection；
- 每一拍通过 Waymax-aligned inverse acceleration / steering-curvature check；
- mode 0 第一拍 target 必须与 online `_consistent_one_step_targets_np` 精确一致，否则 fail loudly。

因此 V38 真正扩展的是 fixed semantic geometry 上的**control support**，不是修改车辆物理限制。

## 7.4 完整 physical tube certificate

每条 control-realized tube 必须同时满足：

1. finite；
2. frozen roadgraph drivable screen；
3. frozen causal constant-velocity collision screen 在完整 horizon 上安全；
4. 每一拍满足 Waymax-aligned kinematics contract。

它不被重新标成 `conventional_safe` 或 NCF，也不修改 social certificate semantics。它仍是显式 uncertified-recovery branch 中的独立 physical backup certificate。

## 7.5 One-step shift closure

仅“当前 8 s tube 安全”仍可能是 open-loop artifact。

V38 对候选 tube：

1. 执行真实第一拍 emitted action；
2. 用 causal successor 更新 ego 与其他 agents；
3. 将已 realization 的 tube 左移一拍；
4. 在末端只追加一次 constant-velocity terminal edge；
5. 以第一拍 emitted acceleration 作为 successor controller memory；
6. 从 causal successor 重新投影整个 shifted tube；
7. 再次通过同一完整 physical certificate。

这定义了最小 shift closure：

\[
\tau_t \in \mathcal T_{phys}(s_t)
\quad\land\quad
\operatorname{Shift}(\tau_t) \in \mathcal T_{phys}(s_{t+1}).
\]

它不是 V2/V3/V4 horizon stacking：没有反复向更远 future tree 搜索，也不要求两拍内回到 original conventional bank；它检查的是同一个 backup tube family 对一次真实 replanning shift 的闭包性质。

## 7.6 Hard-certified set 内的选择

若没有任何 shift-closed tube：

```text
unchanged COWP least-coercive-valid fallback
```

若存在：

```text
hard physical tube certificate
→ min frozen COWP fallback score
→ min |first accel delta from nominal|
→ prefer nominal mode on exact tie
→ deterministic parent/mode index
```

没有：

- profile area/AUC；
- horizon discount；
- risk/progress/collision weight；
- relaxed threshold；
- dwell time；
- learned classifier。

选中的 tube 通过 execution override 真正将其 projected first target 与 acceleration 发给 Waymax，而不是只在 analyzer 中改变 score。

## 7.7 不使用 recovery mode state

V31/V32 证明 stateless hopping 与 unconditional commitment 都可能有害。

V38 不使用 persistent mode：每一拍真正改变动作之前，该动作自身必须拥有完整、shift-closed backup tube witness；下一拍基于真实 observed state 重新构造和认证。

因此 continuity 来自 certificate closure，而不是 heuristic dwell/hysteresis。

## 7.8 无信息泄漏

V38 只使用：

- 当前 simulator state；
- 当前 map/roadgraph；
- 当前 checkpoint output 与 fixed candidate bank；
- 原 common controller limits/memory；
- 与 frozen conventional screen 相同的 causal CV surrounding-agent model。

不读取：

- Waymax future `log_trajectory`；
- future outcome labels；
- online mechanism GT；
- counterfactual outcome of unexecuted policy；
- future human annotations。

## 7.9 机制归因 diagnostics

V38 记录：

- probe/certificate/action-change step rates；
- parent pool 与 distinct action classes；
- generated tubes 与 unique first actions；
- full physically safe 与 shift-closed tube 数；
- nominal/lower/upper envelope shift-closed 数；
- lifted-only parent support；
- selected lifted rate与 envelope mode；
- selected first-accel delta；
- current/shifted collision margins；
- selected fallback-score delta。

重要解释：

> `selected_lifted_support_rate > 0` 不是第七个 outcome Gate；但若 V38 outcome 通过而该值约为 0，则不能把成功归因于“constructive support expansion”，最多只能归因于更严格的 terminal tube certificate。

---

# 八、V38 的预注册分叉

V38 继续原封不动使用六项 Stage-1 Gate。

## 8.1 可能结果与下一分支

### A. Gate 通过，且 lifted support 实际被选择

支持：

> fixed semantic geometry 内确实存在 nominal bank 未显式提供、但属于真实 controller reachable set 的 backup support；shift closure 是有效 physical feasibility object。

之后才运行 fresh37。fresh37 仍通过，才考虑 exact200 development confirmation和后续论文级 final unseen evaluation。

### B. Gate 通过，但 lifted selection≈0

不能声称 constructive support 成功。说明结果主要来自 full-tube + shift-closure certificate 对 existing nominal action 的筛选。

论文机制应收紧为 terminal invariant-tube certificate，而不是 reachable support construction。

### C. certificate rate≈0

说明三端点纵向 lifting 仍不足以构造 backup support。禁止放宽 certificate或再加 bridge depth；下一步转向真正的 constrained reachable-set optimization、interaction-aware tube construction 或 proposal-support expansion。

### D. certificate 非零、action change 非零，但 collision/kinematics/EP Gate 失败

说明 frozen CV/road/kinematic shift-closure model与真实 logged-replay closed-loop outcome仍存在 model mismatch，或纵向-only support无法覆盖需要的 lateral/gap interaction。

下一步应研究 robust interaction-aware reachable tubes / game-aware response envelope，而不是继续调 selector或 score。

### E. collision 改善但 kinematics/EP 失败

仍判 archive。不得通过 penalty tuning救结果。应利用逐场景 tube mode、first-accel delta、current/shift margins与下游 event timing确定是 support geometry、physical adapter，还是 progress-feasible terminal set缺失。

---

# 九、CCF-A 主线判断

当前最值得继续维护的统一主线仍是：

## **Orthogonal Option-Set Feasibility**

### Social axis

```text
Natural roots
→ same-root RCOT
→ BCOT
→ protected-priority hard certificate
→ certificate-compatible set preservation
```

问题：

> ego 是否通过压垮其他 critical agents 的 natural low-burden option set 获得所谓 safety？

### Physical axis

V38 候选结构：

```text
existing semantic recovery geometry
→ controller-reachable support lifting
→ full physical tube certificate
→ one-step shift closure
→ hard-certified least-coercive execution
```

问题：

> zero-conventional recovery 是否仍保留 ego 自己可持续执行的 control-reachable backup set？

统一抽象：

> **Safety must not be obtained through critical option-set collapse.**

V38 的 tube/shift closure 本身不应单独宣称 CCF-A novelty。研究价值来自把：

- 他车 natural response-set preservation；
- ego control-reachable backup-set preservation；

放在同一 feasibility-first、非 scalarized 的规划框架内，并通过完整的负消融与分阶段证据协议证明两条轴语义正交、结构统一。

最终论文仍需要：

- 冻结算法后的从未参与选择的 final scene set；
- 多 independent seeds；
- paired confidence intervals；
- reactive-agent evaluation；
- held-out human-audited social-burden stress set；
- proposal sufficiency 与 selector/certificate quality 分开报告。

---

# 十、V38 代码与实验协议

## 10.1 运行顺序

```bash
cd COWP_v16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh make_ids

# 仅 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh analyze_counterfactual48
```

到这里停止。

只有：

```text
preregistered_gate.shift_closed_control_reachable_tube.pass == true
```

才运行：

```bash
PROMOTED_METHODS=cowp_shift_closed_control_reachable_tube \
  bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_38_SHIFT_CLOSED_CONTROL_REACHABLE_TUBE_CN.sh analyze_fresh37
```

fresh37 再通过才运行 exact200 development confirmation。

## 10.2 当前代码验证

- V38 dedicated tests：7/7 passed；
- V16.8.25→38 focused semantic/integrity suite：68/68 passed；
- Python compile：passed；
- launcher `bash -n`：passed；
- exact200/equivalence16/counterfactual48/fresh37 manifest hashes：passed；
- analyzer smoke：passed；
- fail-closed promotion probe：exit code 4，未启动 rollout；
- conventional-safety bypass grep：none；
- V38 helper logged-future use：none；
- nominal projector first-target equivalence invariant：covered by tests。

本地环境没有你的 Waymax dataset/index/checkpoint/GPU runtime，因此没有伪造 V38 closed-loop结果；V38 是否有效必须由上述服务器命令产生的新结果决定。
