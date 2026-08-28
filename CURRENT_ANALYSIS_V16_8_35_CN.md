# V16.8.34 结果分析与 V16.8.35 算法设计

> 实际版本链：**V16.8.34 → V16.8.35**。用户消息中的 `V12.8.35` 按历轮约定视为版本号笔误。
>
> 本文档遵守顺序：**结果可靠性审计通过后才做算法归因；预注册 gate 不因当前结果而修改；失败分支不通过调阈值“救活”。**

## 0. 总结结论

1. **V16.8.34 结果可靠，可以进行算法归因。** 独立机器审计 32/32 个 hard checks 全部通过；`equivalence16` 为 16 scenes / 1120 fields / 0 mismatch；两个 counterfactual48 方法均为 24+24 shards、无 overlap、union 精确等于 manifest；merged 标准指标可由逐场景 rows 零误差重算；上传 analyzer 可独立重跑并逐字段完全复现；online future-GT flag 均为 false。
2. **严格按 V33/V34 原封不动的六项预注册 Stage-1 gate，TG-ROSH 与 EOSH 都 FAIL。** 因此 V34 不允许进入 fresh37/exact200。
3. **V34 不是“差一个参数”，而是 representation 定义错了一层。** 它把“nominal first waypoint 能否在内部 controller hard limits 下无失真到达”当成 executable option；但真正在线执行本来就会对 nominal candidate 做 jerk/accel/yaw projection。需要 projection 不等于不可执行。
4. 同时，V34 的 internal transition predicate 与 Waymax `KinematicsInfeasibilityMetric` 的判定对象不一致：前者检查 acceleration/deceleration/jerk/yaw-rate/lateral-accel；后者按连续状态逆动力学得到 **acceleration + steering curvature** 并判阈值。因此 V34 既产生大量 false-negative recovery rejection，又没有真正约束当前 benchmark 的 kinematics failure mode。
5. V34 的结果把 dominant bottleneck 从 **Controller-State-Conditioned Executable Recovery Option Spectrum** 再收紧到：

   **Evaluation-Aligned Control-Realized Recovery Option-Set Preservation**

   更论文级、平台无关的表述是：

   **Control-Projected Physical Option-Set Feasibility**。

6. V16.8.35 不修改成熟 COWP controller/certificate/proposal，而新增两个严格解耦分支：
   - **WK-ROSH**：只把 current-action execution predicate 换成 Waymax 实际评测的 inverse acceleration / steering-curvature contract，future profile 仍是 V33 semantic spectrum；用于诊断“V34 是否只是检查错了 physical contract”。
   - **CPOSH**：在相同 current guard 上，把 successor candidate bank 的每条 trajectory **逐步经过现有 stateful COWP controller projection**，携带 longitudinal acceleration memory，并在 control-realized trajectory 上重新检查 roadgraph、causal collision survival、Waymax kinematics survival，再统计 distinct macro option spectrum；用于验证真正缺失的是否是 **control-realized future option set**。
7. Stage-1 六项 gate **完全不变**；已知 V33 两个 kinematics counterexamples 只作 diagnostic，禁止变成 outcome-tuned hard gate。
8. 若 V35 两个分支仍 FAIL，应停止继续做 ROSH/EOSH/guard 变体，下一主分支转向 **reachable proposal/support construction 或更高保真 physical transition model**；accepted-path kinematics 仍作为另一条独立 secondary bottleneck，不与 recovery 同轮修改。

---

# 1. 论文与 CCF-A 主线的理解

论文当前真正有研究价值的核心不是“再加一个 social cost”，而是把：

**false-safe / safety-by-coercion**

定义为 feasibility defect：一个 ego candidate 即使自身 collision-free，如果安全依赖 protected-priority road user 通过 hard braking、abrupt yielding、priority abandonment 或 gap surrender 承担异常 burden，则不应被认为是可行解。

当前主机制是：

`burden-oriented interaction graph`
→ `natural alternatives`
→ `same-root response/RCOT transport`
→ `BCOT aggregation`
→ `protected-priority non-coercive hard certificate`
→ `certificate-compatible set-preservation frontier`
→ `ego utility`

这条主线已经形成一个清晰的论文对象：**关键主体的 viable option set 不应因为 ego plan 而发生不可接受的 collapse。**

历轮 Waymax 结果又暴露了一个正交的 physical-temporal axis：当系统已经处于 uncertified recovery regime 时，一个 recovery action 也不应把 **ego 自己未来 control-realizable recovery choices** 压垮。

因此目前最值得继续维护的 CCF-A 论文主线是：

## Orthogonal Option-Set Feasibility

### Social axis

`natural roots → same-root RCOT → BCOT → protected-priority option preservation`

判断：ego 是否通过压缩**他人的 natural low-burden option set**来获得所谓 safety？

### Physical-temporal axis

`actual emitted action → controller-state causal successor → control-realized recovery option set → dominance-consistent mode`

判断：uncertified recovery 是否通过当前动作压缩**ego 自己未来真正能执行的 recovery option set**？

两个 axis 对象不同、语义正交，但统一于：

> **Safety should not be obtained by critical option-set collapse.**

这仍然比 `social score + collision score + outcome risk + utility` 的 scalarization 更像一条方法学论文主线。

但需要严格避免过度包装：hysteresis、safety filter、backup feasibility、multi-horizon lookahead、benchmark kinematics adapter 都不是 CCF-A novelty 本身。V35 仍是 physical-axis mechanism probe；只有经过 Stage-1、fresh37、最终新冻结 publication holdout、多 seed/paired CI 以及 reactive-agent/human-audited protocol 后，才有资格升格为论文 contribution。

---

# 2. V16.8.34 可靠性审计

独立审计文件：`V16_8_34_RESULT_INTEGRITY_AND_ATTRIBUTION_AUDIT.json`。

## 2.1 equivalence16

- manifest：16 unique IDs；logical SHA256 与预注册一致；
- shards：8 + 8，无 overlap；union 精确等于 manifest；
- merged rows：16；可由两个 shards 精确重建；
- standard summary 最大绝对重算误差：0；
- checkpoint 一致；
- online GT flag：全 false；
- COWP vs immutable V16.8.29 reference：**1120 fields / 0 mismatch**。

`shard_order_partition_matches_manifest=false` 不是 integrity failure：launcher 的 sharding 不是“按 manifest 前半/后半”划分，hard check 是 shard 无重叠且 union 精确等于 manifest，这一点通过。

## 2.2 counterfactual48

TG-ROSH 与 EOSH：

- 同一个 48-ID manifest；
- 每个方法 24 + 24 shards；
- shard overlap=0；
- union 精确等于 manifest；
- merged rows 精确来自 shards；
- CR / Collision / Offroad / Kinematics / EP 从 scenario rows 重算误差=0；
- checkpoint 一致；
- online GT flag=false。

## 2.3 analyzer 与代码语义

- V34 analyzer 独立重跑后与上传 JSON recursive exact match；
- V34/V33 focused tests：9/9 passed；
- EOSH successor profile 确实携带 emitted acceleration；
- EOSH 只读 current simulator state + current emitted action + causal CV surrounding model；没有使用 Waymax logged future。

**结论：V34 结果可以进入算法归因，不是 repair-only。**

---

# 3. 严格按上一轮预注册 GO 条件判定 V34

Stage-1 gate 沿用 V33，V34 明确承诺不改：

| Gate | GO |
|---|---:|
| old RVR rescues retained | >= 5/10 |
| old RVR induced avoided | >= 7/9 |
| 相对 COWP collision 净减少 | >= 3 scenes |
| Kinematics regression | <= 1 scene |
| paired mean EP delta | >= -0.05 |
| intervention | > 0 |

## 3.1 48-scene总结果

| Method | Collision | Kinematics | EP | recovery switch |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 6/48 | 1.00251 | -- |
| RVR | 33/48 | 9/48 | 0.82362 | historical |
| V33 ROSH | 29/48 | 8/48 | 0.97465 | 7.84% |
| V34 TG-ROSH | 32/48 | 7/48 | 0.98470 | 4.06% |
| V34 EOSH | 34/48 | 8/48 | 0.99558 | 2.89% |

## 3.2 TG-ROSH

- old RVR rescues retained：**3/10 — FAIL**
- old RVR induced avoided：**8/9 — PASS**
- COWP collision：3 rescue / 1 induced，净减少 **2 — FAIL**
- Kinematics：1 rescue / 2 induced，净 regression +1 — PASS
- EP delta：**-0.01781**，bootstrap 95% CI `[-0.04419, +0.00492]` — PASS
- intervention > 0 — PASS

所以 TG-ROSH FAIL。

它相对 V33 ROSH 反而是 collision **1 rescue / 4 induced**，即净差 3 个；只在 kinematics 上救了 1 个 scene。说明 internal transition non-regression 主要在**抑制 intervention**，而不是识别真正 physical recovery quality。

## 3.3 EOSH

- old RVR rescues retained：**0/10 — FAIL**
- old RVR induced avoided：**8/9 — PASS**
- COWP collision：1 rescue / 1 induced，净减少 **0 — FAIL**
- Kinematics：0 rescue / 2 induced，净 regression +2 — FAIL
- EP delta：**-0.00693**，95% CI `[-0.02810,+0.01137]` — PASS
- intervention >0 — PASS

所以 EOSH 明确 FAIL。

相对 V33 ROSH，EOSH 是 collision **1 rescue / 6 induced**，净差 5 个；而两个 V33 新增 kinematics counterexample 一个都没有修复。

**科研纪律结论：TG-ROSH/EOSH implementation 全部 archive；不能运行 V34 fresh37/exact200。**

---

# 4. V34 真正提供的新证据

## 4.1 “immediately realizable” hard filter 过度压制 recovery recall

在 recovery probes 上：

- TG：base internal-transition feasible ≈ **88.05%**，RVR ≈ **48.37%**；
- EOSH：base ≈ **89.52%**，RVR ≈ **48.32%**。

也就是说，RVR alternatives 有超过一半被 V34 internal exact-transition predicate 判为不可执行。

这与 EOSH `old RVR rescues retained = 0/10` 高度一致。

但在线 controller 的本职工作就是把 raw/nominal candidate 通过 acceleration/jerk/yaw-rate limits 投影成真正 emitted target。因此：

> **nominal waypoint 需要被 projection 修正，不等于这个 recovery mode 不可执行。**

V34 把“projection magnitude/necessity”错误地升级成 hard existence criterion，造成系统性的 false-negative option collapse。

这不是调 comparator 能修复的问题。

## 4.2 internal controller feasibility 不是 Waymax KinematicsInfeasibility 的同一个对象

V34 `_controller_transition_feasible_np` 检查：

- max acceleration / deceleration；
- jerk from `previous_longitudinal_accel`；
- yaw rate / delta yaw；
- lateral acceleration。

Waymax closed-loop 的 `KinematicsInfeasibilityMetric` 则对连续状态做 inverse bicycle-style transition，当前公开实现核心判据是：

- inverse longitudinal acceleration；
- inverse steering curvature；
- metric default threshold 约为 `|accel| <= 10.4 m/s^2`、`|curvature| <= 0.3 m^-1`。

这两个 predicate 不等价。

V35 regression test 构造了一个明确反例：2 m/s 时 0.1 s 内 yaw 变化 0.1 rad，V34 internal yaw-rate/lateral limits 可接受，但 Waymax inverse curvature 约 0.5 m^-1，超过 0.3，因此评测为 infeasible。

所以 V34 同时存在：

1. **false negative**：nominal candidate 需要 controller projection，却被 EOSH 从 option set 删除；
2. **false positive**：internal yaw-rate predicate可通过，但 evaluator inverse steering curvature 可失败。

## 4.3 两个 V33 kinematics counterexample 在 V34 仍全部存在

- `29cd2aca8ae5e222`：first kinematics 前直接是 recovery `MERGE_AHEAD`；V34 EOSH 的 base/RVR internal transition feasible rate 在这个 scene 的 probes 都是 1.0，说明 internal predicate 根本没有发现真正的 metric failure。
- `6992366c5c998d00`：first kinematics 前已回到 `accepted_priority_ncf / LANE_CHANGE_RIGHT / conventional-safe=true`；这仍证明 earlier recovery 改变 controller/closed-loop state 后可能在后续显现 physical defect。

因此绝不能回到“给当前 RVR action 加一个 kinematics penalty”。

---

# 5. dominant bottleneck 再收紧

历史链：

- V28：online conventional-feasible support collapse；
- V29：主要是 dynamic collision-side zero-conventional collapse，非 roadgraph；同 bank 可 rescue，proposal 不是唯一 bottleneck；
- V30：current prefix 不足，successor option set 有独立高精度信息；
- V31：stateless hybrid switching 可制造 failure；
- V32：horizon stacking 与 unconditional commitment 都失败；
- V33：semantic recovery-option spectrum 明显比 signature/max-prefix 有效，但 Kinematics gate +2 fail；
- V34：把 nominal exact first-step realizability 当 executable option 会严重杀 recall，且 internal predicate 与 evaluator kinematics 不一致。

因此 P0 应从：

`Controller-State-Conditioned Executable Recovery Option Spectrum`

进一步收紧为：

## **Evaluation-Aligned Control-Realized Recovery Option-Set Preservation**

论文级平台无关表达：

## **Control-Projected Physical Option-Set Feasibility**

真正需要回答的是：

> 给定当前 physical/controller state，执行一个 recovery action 后，重新生成的未来 recovery alternatives 在**实际闭环控制投影**下，仍有多少语义不同的分支能够持续满足道路、碰撞和平台执行约束？

关键区别：

- 不是 nominal candidate 是否“无需修正就能执行”；
- 不是 longest prefix；
- 不是一个 one-step risk scalar；
- 而是 **after-control-realization 的 option-set survival structure**。

目前最值得解决的仍是这个 dominant recovery bottleneck，而不是马上混入 accepted-path kinematics。原因是：V33 已证明 option spectrum 对 collision recovery 有强信号；V34 又表明 execution-conditioned representation 仍有明确可修复的定义错误。只有 V35 仍然失败，才应该停止该 family 并切更上游的 reachable proposal/support construction。

---

# 6. 当前每一层成熟度

| Layer | 当前判断 | V35 原则 |
|---|---|---|
| compact-5k data/label contract | Mature | **Freeze**；不重建 |
| Natural roots | Mature | **Freeze** |
| RCOT same-root transport | Strong/Mature | **Freeze** |
| BCOT | Strong/Mature | **Freeze** |
| protected-priority hard certificate | Mature | **Freeze** |
| set-preservation frontier | Mature，CTU 支持 | **Freeze** |
| Outcome head | diagnostic-only | **Freeze** |
| 8 s conventional-safety contract | stable attribution baseline | **Freeze** |
| V27 conventional integrity | solved | **Freeze** |
| V28 no-valid execution integrity | solved | **Freeze** |
| RVR max-prefix policy | negative | controlled alternative only |
| SOV successor signal | supported | preserve insight |
| BHOV/THOP horizon stacking | negative | **Archive / ban** |
| unconditional commitment | negative policy，mode signal positive | archive policy |
| V33 semantic option spectrum | **mechanism object supported** | preserve + upgrade |
| V33 dominance hysteresis semantics | useful state-machine relation | freeze semantics during representation test |
| V34 TG-ROSH | negative | Archive |
| V34 EOSH nominal exact-realizability | negative | Archive |
| control-projected physical option representation | **P0 / immature** | V35 main branch |
| accepted-path kinematics | real secondary bottleneck | **do not mix in V35** |
| proposal support | long-term global ceiling | do not expand yet |

compact-5k 既往独立 characterization 仍支持保持冻结：train/val/heldout audit-relevant pair rate 约 0.429，protected PRIO root coverage 约 99.4%，rootless 与 `<2 low-burden roots` 为 0；没有新证据要求重建数据。

---

# 7. 当前模型下一步应该“学”什么

social side 已经比较会回答：

> ego plan 是否通过压缩 protected actor 的 natural low-burden choices 才保持安全？

physical recovery side 还没有正确表示：

> **nominal recovery choices 在经过实际 controller/dynamics 后，还剩多少真正可以实现且能持续生存的语义分支？**

下一步最值得形成的 target 不是：

- `max safe prefix`；
- generic collision probability；
- nominal candidate count；
- exact first-waypoint reachability；
- weighted profile area。

而是：

`(physical state, controller memory, recovery mode)`
→ `actual emitted/control-projected trajectory family`
→ `road/collision/execution survival`
→ `distinct semantic option survival curve`。

V35 仍坚持 analytic target first。只有该 target 被闭环结果验证后，未来才有理由训练 learned viability head；否则只是在用 neural classifier 拟合一个未证实标签。

---

# 8. 后续明确禁止的方向

除总 `ALGORITHM_CHANGELOG.md` 已禁止的方向外，V34 结果新增两类 clean negative：

1. **禁止把 nominal first-waypoint exact-reachability hard filter 继续当成 executable-option definition。** EOSH old RVR rescues 0/10，明确杀死 recovery recall。
2. **禁止继续把 internal accel/jerk/yaw-rate/lateral-accel transition non-regression 当成 Waymax Kinematics proxy。** 它没有修复 V33 两个 kinematics counterexample，并与 evaluator steering-curvature contract 不同。

继续保留历史禁区：

- certificate → planner-score argmin / CTU replacement；
- fallback outcome weight tuning；
- outcome head hard shield；
- 缩短 8 s conventional horizon 制造“safe”candidate；
- 直接 promotion max-prefix RVR；
- Pareto tolerance / risk-weight search；
- BHOV comparator/epsilon relaxation；
- V3/V4/V5 horizon stacking；
- unconditional commitment / fixed N-step dwell；
- hysteresis epsilon/margin tuning；
- profile AUC / discounted horizon weighted sum；
- social + physical + utility scalarization；
- 当前扩 map/Frenet primitives；
- RCOT/BCOT threshold/budget tuning；
- analytic target 验证前训练 successor neural head；
- recovery 与 accepted-path kinematics 同轮修改；
- 为救 ROSH 把 kinematics gate 从 +1 改 +2；
- 只在 V33 ROSH 当前 action 上加 kinematics/risk penalty；
- **全局修改成熟 common controller 来让 V35 看起来更好**：这会破坏 common-path equivalence，并把 recovery 与 accepted-path execution 问题混在一起。

---

# 9. V16.8.35 算法设计

版本目录：`COWP_v16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM`

## 9.1 公共约束：只在 uncertified recovery regime 工作

两个新方法均只在：

```text
full conventional set == empty
AND valid candidate exists
```

时工作。

比较仍然只有两个历史受控 counterfactual：

```text
base = original COWP least-coercive-valid fallback
alt  = original V16.8.29 RVR max-prefix candidate
```

不新增 candidate family，不改变 conventional-safe / NCF 标签，不改 RCOT/BCOT/certificate/frontier，不读取 logged future。

mode semantics 继续冻结 V33 的 parameter-free dominance hysteresis：

- inactive：strict dominance 才 entry；
- active：weak/equality dominance 可 continue；
- 任一 component regression：immediate exit；
- certificate/conventional 恢复或 no-valid emergency：clear mode。

## 9.2 Diagnostic branch：WK-ROSH

方法：

`cowp_waymax_kinematic_guarded_rosh`

目的不是 paper contribution，而是隔离 V34 是否主要“检查错了 execution contract”。

### Current transition

对 base / RVR 已经经过在线 controller 投影得到的 **actual emitted target**，按 Waymax `KinematicsInfeasibilityMetric` 的 inverse-transition semantics 计算：

```text
inverse acceleration
inverse steering curvature
```

alternative 必须 current KIM-feasible；如果 alt infeasible，即使 future semantic profile 更好也不能 entry/continue。

### Future profile

仍使用 V33 原始 semantic recovery option spectrum，不做 control projection。

因此：

- 如果 WK-ROSH 明显恢复 V33 collision gain 且修复 kinematics，说明 V34 主要问题是 current execution contract mismatch；
- 如果它仍失败而 CPOSH 成功，则证明真正缺失的是 future control-realized option representation。

## 9.3 Main branch：CPOSH

方法：

`cowp_control_projected_option_spectrum_hysteresis`

完整名称：**Control-Projected Option-Spectrum Hysteresis**。

### Step A：current action

base/RVR 当前 action 都使用真实在线 controller 已得到的 emitted target + emitted longitudinal acceleration。

用 evaluator-aligned execution adapter 检查 current transition。

### Step B：causal successor

ego 用 actual emitted action 前进一步；surrounding agents 仍使用 conventional collision screen 同源的 causal constant-velocity propagation；不读取 Waymax future GT。

### Step C：重新生成同一个 successor proposal bank

candidate generator、candidate families、map/route logic 均保持冻结。

### Step D：每条 successor candidate 不是检查 nominal first step，而是**逐拍通过当前 COWP controller 投影**

对 horizon `t=1...H`：

1. 由当前 realized state 与 nominal desired waypoint 计算 desired speed/yaw；
2. 应用现有 accel/decel hard limit；
3. 应用 `previous_longitudinal_accel ± jerk*dt`，并把新 accel 作为下一拍 controller memory；
4. 应用既有 yaw-rate / Waymax delta-yaw projection；
5. 用 trapezoidal integration 得到 actually realized next position/velocity/yaw；
6. 对 realized transition 计算 evaluator execution feasibility；
7. 将 realized state 继续传播到下一拍。

V35 新增 regression test，验证 control-projection helper 的**第一步 target 与现有在线 `_consistent_one_step_targets_np` 精确一致**，避免 diagnostic rollout 与真实 online controller 漂移。

### Step E：在 control-realized trajectory 上重新构造 option survival

每个 successor candidate 的 effective survival prefix 由：

```text
roadgraph-safe on control-projected path
AND causal collision-free prefix on projected path
AND evaluator kinematics-feasible prefix on projected transitions
```

共同决定。

对每个 horizon `h`，定义：

```text
P_ctrl(h) = # distinct non-PAD recovery macro types
            whose control-realized trajectory survives at least h
```

同一 macro 的大量近似候选仍只计 1 个 option，避免 candidate multiplicity 冒充 option diversity。

### Step F：不做 scalar ranking，继续 pointwise partial order

```text
P_alt(h) >= P_base(h) for every h
```

且至少一个 horizon strict better，才构成 profile strict dominance。

同时 current alt execution 必须可行。

不用：

- profile AUC；
- horizon discount；
- candidate-count weight；
- kinematics penalty weight；
- risk weight；
- hysteresis margin；
- dwell time；
- learned classifier。

这使 V35 仍是可证伪的 structural hypothesis，而不是 hyperparameter search。

## 9.4 benchmark adapter 与论文 formulation 的边界

代码当前用 Waymax KinematicsInfeasibilityMetric 的 inverse acceleration/steering-curvature contract 来实例化 execution feasibility，是为了和当前 Waymax闭环评价完全对齐并验证 V34 mismatch。

**论文不能把 Waymax 默认数值当贡献。** 最终 paper formulation 应写为平台无关 execution envelope `E(s,a)=1`：

- 在 Waymax 中由官方 inverse-kinematics metric adapter 实例化；
- 在真实车辆/高保真 simulator 中应由 actuator/steering/bicycle dynamics constraint 或 safety-certified low-level controller envelope 实例化。

真正方法对象仍是 `control-realized option-set preservation`。

---

# 10. V35 的预注册实验分叉

Stage-1 `counterfactual48` 对 WK-ROSH/CPOSH **仍用完全相同六项 gate**：

| Condition | GO |
|---|---:|
| old RVR rescues retained | >=5/10 |
| old RVR induced avoided | >=7/9 |
| COWP collision net removed | >=3 |
| kinematics regression | <=1 |
| mean EP delta | >=-0.05 |
| intervention | >0 |

不加入“必须修复 `29cd.../699...`”这样的 hard gate；这两例只作 diagnostic，以避免 counterexample overfitting。

Stage-1 结果解释：

| 结果 | 下一步 |
|---|---|
| WK pass，CPOSH 无额外收益 | current execution-contract alignment 是主要缺口；physical axis 先保持简单 |
| CPOSH 明显超过 WK 且 pass | **control-realized future option set 是关键机制**；进入 fresh37 |
| 两者都 pass，CPOSH 更强 | representation + execution adapter 都有价值，主推 CPOSH |
| kinematics 改善但 rescue recall 仍 fail | hard execution gate 仍过度保守；archive family，不调阈值 |
| collision pass 但 kinematics >+1 | physical representation 仍不完整；archive |
| 两者都 fail | **停止 ROSH/EOSH/guard 变体**；转 reachable proposal/support 或 higher-fidelity dynamics construction |

Stage-2 fresh37 gate 保持历史：

- no net collision harm；
- no net CR harm；
- offroad regression <=1；
- kinematics regression <=1；
- mean EP delta >= -0.03；
- intervention >0。

fresh37 仍属于 historical exact200 development universe，不是 publication holdout。

exact200 也只是 development confirmation。算法最终 freeze 后必须重新冻结从未参与 mechanism selection 的 final evaluation scenes，并按论文要求进行 >=3 independent seeds + paired scene CI；strong causal burden claim 还必须走 reactive-agent + held-out human-audited stress protocol。

---

# 11. 下一步命令

复用现有 compact-5k checkpoint，不训练、不重建 dataset/cache：

```bash
cd COWP_v16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh make_ids

# 只有 TFExample index 缺失时
bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh build_tfindex

# common-path equivalence
bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh base_equivalence16_parallel2

# Stage-1：同时跑 diagnostic WK-ROSH 与主分支 CPOSH
bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh analyze_counterfactual48
```

**到这里停止。**

只有 analyzer 中某方法 `preregistered_gate.<method>.pass=true` 才运行 fresh37；launcher 是 fail-closed。

例如 CPOSH 通过：

```bash
PROMOTED_METHODS=cowp_control_projected_option_spectrum_hysteresis \
  bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh analyze_fresh37
```

fresh37 再通过才允许：

```bash
PROMOTED_METHODS=cowp_control_projected_option_spectrum_hysteresis \
  bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh confirm200_parallel2

bash NEXT_RUN_COMMANDS_V16_8_35_CONTROL_PROJECTED_OPTION_SPECTRUM_CN.sh analyze_confirm200
```

---

# 12. 代码验证

V16.8.35 当前本地验证：

- new V35 semantic/controller tests：**6/6 passed**；
- V16.8.25→V16.8.35 focused semantic/integrity suite：**49/49 passed**；
- Python compile：passed；
- launcher bash syntax：passed；
- common manifests hash：passed；
- V35 projection regression：first projected step 与 online controller target/acceleration exact match；
- full repository `pytest -x` 当前在 test collection 阶段遇到一个**V34 原包即可复现**的历史 external-baseline import error：`candidate_geometry_finite` 不存在于 `cowp.external_baselines.adapters`。该错误与 V35 新代码无关，因此没有把 full suite 虚报为通过。

---

# 13. 本轮最重要的研究结论

V34 的失败没有推翻 option-spectrum 方向，反而暴露了一个更精确的定义问题：

> **“nominal trajectory 看起来从 controller state 可立即无失真实现”不是 physical option；真正的 physical option 必须先经过实际闭环 controller/dynamics realization，再判断它是否仍保留足够的 future recovery support。**

因此 V35 的价值不是“换一个 kinematics threshold”，而是第一次把 physical option-set feasibility 从：

`nominal option existence`

改成：

`control-realized option existence and persistence`。

如果 CPOSH 能在不放宽任何预注册 gate 的情况下通过 counterfactual48 + fresh37，它会成为目前为止 physical-temporal axis 最有资格进入 CCF-A 主机制组合的候选；如果仍失败，就应该干净停止这个 family，把失败本身作为证据转向更上游的 reachable proposal/support 或 higher-fidelity physical state construction，而不是继续做阈值/权重工程。
