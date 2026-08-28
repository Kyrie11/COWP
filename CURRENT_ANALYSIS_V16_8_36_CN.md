# V16.8.35 结果分析与 V16.8.36 设计

## 结论摘要

本轮按实际版本链 **V16.8.35 → V16.8.36** 推进。V16.8.35 结果通过可靠性审计，因此可以做算法归因；不是 repair-only。

V16.8.35 两个分支都没有通过上一轮原封不动的 Stage-1 预注册 GO 条件：

- WK-ROSH：29/48 collision、8/48 kinematics、EP=0.974647；只失败 kinematics +2 这一项，不能 promotion。
- CPOSH：27/48 collision、7/48 kinematics、0 offroad、EP=0.913987；10 rescue / 3 induced，净减少 7 collision，但 old-RVR-induced avoided=6/9 < 7/9，paired mean EP Δ=-0.08853 < -0.05，因此不能 promotion。

科研纪律上，两者都必须 archive，不能因为 CPOSH collision 很好就越过 gate 跑 fresh37。

但是 CPOSH 的 **control-projected recovery option spectrum** 不是 clean negative：它是 V29→35 中到目前最强的 collision-side 正信号之一。应保留并升级的是 representation object，而不是 V35 binary policy implementation。

V35 进一步表明当前最关键的结构瓶颈不再是 current-action kinematics guard，而是：

> **在 zero-conventional recovery 中，有大量 existing-bank valid candidates，但 V29→35 始终只让 expensive viability observable 在 COWP base 与一个 global-RVR endpoint 之间二选一。**

因此 V16.8.36 进入 **Control-Projected Semantic Recovery Frontier**：不扩 proposal、不改成熟层，而是在完全相同 bank 中构造 semantic recovery representatives，对整个 hard physical frontier 做 feasibility-first selection。

---

# 1. V16.8.35 可靠性审计

## 1.1 结果包完整性

独立审计共 **66/66 hard checks passed**：

- counterfactual48 manifest：48 unique IDs；
- 两个 V35 新方法均为 24+24 shards；
- shard 之间无 overlap；
- union 精确等于同一个 48-scene manifest；
- declared logical SHA 与 manifest 一致；
- merged 结果 48 unique IDs；
- merged scene set 与 manifest 精确一致；
- checkpoint 一致；
- `mechanism_ground_truth_available_online=false`；
- CR / Collision / Offroad / Kinematics / EP 能从 scenario rows 零误差重算；
- fallback 汇总也可重算；
- equivalence16 仍为 16 scenes / 1120 fields / 0 mismatch。

## 1.2 analyzer 可复现

在上传的 V35 源码上重新执行 `88_analyze_control_projected_option_spectrum.py`，与上传 analyzer JSON 逐字段递归比较：

**0 mismatch，tolerance=1e-12。**

## 1.3 代码语义

重新运行 V16.8.25→35 focused semantic/integrity suite：

**49/49 passed。**

V35 control projection 的首拍 target/acceleration 与在线真实 COWP controller 继续有 exact regression 检查；没有发现 future Waymax logged GT 被 recovery observable 读取。

因此：

> **V16.8.35 结果可靠，可以进行算法归因。**

---

# 2. 严格按上一轮预注册 GO 条件判定

Stage-1 六项条件不变：

| 条件 | GO |
|---|---:|
| old RVR rescues retained | >=5/10 |
| old RVR induced avoided | >=7/9 |
| COWP collision 净减少 | >=3 scenes |
| Kinematics regression | <=1 scene |
| paired mean EP Δ | >=-0.05 |
| intervention | >0 |

任意一项失败即 archive，不允许事后改 gate。

## 2.1 WK-ROSH

`cowp_waymax_kinematic_guarded_rosh`：

- Collision：34 → 29；
- Kinematics：6 → 8；
- EP：1.002512 → 0.974647；
- old RVR rescue retained：6/10 PASS；
- old induced avoided：7/9 PASS；
- net collision removed：5 PASS；
- EP Δ=-0.02786 PASS；
- intervention>0 PASS；
- **kinematics +2 FAIL**。

因此 WK-ROSH **FAIL**。

更关键的是，它与 V33 ROSH 在整个 48 scenes 的 closed-loop outcome 上完全一致。current Waymax kinematic guard 的 base/RVR feasible rate 都约 99.85%，mean transition delta=0。这说明：

> **把 current emitted action 对齐到 Waymax kinematic metric，并没有提供新的 recovery selection information。**

current-action guard 不是 dominant missing mechanism。

## 2.2 CPOSH

`cowp_control_projected_option_spectrum_hysteresis`：

| 指标 | COWP | CPOSH |
|---|---:|---:|
| Collision | 34/48 | **27/48** |
| Offroad | 1/48 | **0/48** |
| Kinematics | 6/48 | **7/48** |
| EP | 1.002512 | **0.913987** |

Paired collision：

- rescued：10；
- induced：3；
- net removed：7；
- McNemar exact p≈0.0923。

Gate：

- old RVR rescues retained：8/10 PASS；
- old induced avoided：**6/9 FAIL**；
- net collision removed：7 PASS；
- kinematics net regression：+1 PASS；
- paired mean EP Δ：**-0.08853 FAIL**；
- intervention：PASS。

因此 CPOSH 也 **FAIL**。

不能把它 promotion，也不能跑 fresh37。

---

# 3. CPOSH 失败不是“没有价值”：真正应该 promotion 的是 representation object

CPOSH 不能 promotion 为 policy，但其核心 observable 值得保留：

> **actual controller projection 后的 semantic option survival structure。**

理由：

1. COWP 34 collision → CPOSH 27；
2. 10 rescue / 3 induced，而 Kinematics 只净退化 +1；
3. old RVR rescues 保留 8/10；
4. 它修复了 V33 一部分 execution issue；
5. 两个 endpoint 的 current Waymax feasibility 几乎完全相同，所以 gain 不是 current-action guard 造成的，而来自 future control-realized option representation。

因此本轮 promotion 判断应拆成：

- **不 promotion V35 CPOSH implementation；**
- **保留/升级 control-projected recovery option spectrum 作为 physical feasibility observable。**

---

# 4. 为什么 CPOSH 仍会失败

## 4.1 Binary-endpoint structural bottleneck

V29→V35 的 recovery family 虽然 observable 不断升级，但当前动作的反事实集合始终只有：

- base = global least-coercive-valid COWP fallback；
- RVR = global max current safe-prefix candidate。

CPOSH 的 expensive control-projected spectrum 仍然只回答：

> “这两个 endpoint 哪一个更好？”

而不是：

> “当前 fixed bank 中有哪些不同 semantic recovery branches 真正 physical-feasible，哪个应该进入？”

在当前 48 scenes：

- CPOSH rescued scenes 平均 valid candidates ≈ **37.80**；
- CPOSH induced scenes平均 valid candidates ≈ **38.48**。

因此 induced failure 不能解释为“bank 已经没东西可选”。

这是当前非常明确的 **support-utilization gap**。

## 4.2 Induced cases 更深陷 zero-conventional basin

CPOSH rescued scenes：

- mean zero-conventional exposure ≈ **69.1%**；
- mean conventional candidates ≈ **3.27**。

CPOSH induced scenes：

- mean zero-conventional exposure ≈ **85.4%**；
- mean conventional candidates ≈ **1.68**。

但后者仍有约 38.5 valid candidates。

说明 representation 还存在一个长期问题：

> **rich uncertified support 不等于已经回到 conventional-feasible region。**

V36 不尝试再堆 V2/V3 horizon，也不训练 returnability classifier；第一步先解决更直接、可证伪的 current support utilization：把整个 semantic fixed-bank frontier 暴露给已经验证有信号的 physical observable。

若 V36 仍失败，下一步才有充分理由升级到真正 reachable-set / returnability construction，而不是继续 selector patch。

## 4.3 EP 失败是集中式，而非全局所有场景变慢

CPOSH paired mean EP Δ=-0.08853。

其中 `fccd9a25a2a57a73` 单 scene EP Δ≈**-3.1293**；去掉这一 scene 后其余 paired mean Δ≈**-0.02383**。

这不能用来修改 gate：CPOSH 依然 FAIL。

但机制上说明 progress near-miss 不是简单的“所有 intervention 都变保守”，而是某些 binary endpoint rescue 把闭环送到极低-progress basin。这进一步支持：

> 在多个 physical-admissible recovery branch 存在时，应该保留 hard feasibility，然后用已经成熟的 COWP fallback preference 选择最小偏离的 branch，而不是永远使用 global max-prefix RVR endpoint。

---

# 5. dominant bottleneck 再收紧

证据链：

- V28：online conventional-feasible support collapse；
- V29：主要是 dynamic collision-side，不是 roadgraph；same bank 能 rescue；
- V30：successor option-set 是有效 high-precision signal；
- V31：stateless hybrid switching 可制造 failure；
- V32：horizon stacking 与 unconditional commitment 均失败；
- V33：semantic recovery-option spectrum 明显有效，但 execution realism 不完整；
- V34：nominal exact realizability hard filter 杀死 recovery recall；
- V35：control-projected spectrum 恢复并提升 collision recall，但 binary endpoint policy 仍有 false positives 与 progress cost；current-action Waymax guard 单独完全没有新收益。

因此 P0 从上一轮的 `Control-Projected Physical Option-Set Feasibility` 再收紧为：

# **Control-Projected Semantic Recovery Frontier under Uncertified Recovery**

工程上的具体 bottleneck 是：

# **Existing-Bank Recovery Support Utilization / Binary-Endpoint Bottleneck**

当前不应该优先处理 accepted-path kinematics，也不应该立即扩 proposal primitive。

原因：

1. recovery side 已经有直接、可操作的结构证据；
2. valid support 很大，先利用 existing bank 的科研归因更干净；
3. accepted-path kinematics 是真实 secondary bottleneck，但与 recovery failure 不同源；
4. 同轮修改会破坏归因并可能伤害成熟 certificate path。

---

# 6. 每一层当前成熟度

| Layer | 当前判断 | 下一步原则 |
|---|---|---|
| compact-5k data/label contract | 成熟 | **Freeze** |
| Natural roots | 成熟 | **Freeze** |
| RCOT same-root transport | 强/成熟 | **Freeze** |
| BCOT | 强/成熟 | **Freeze** |
| Protected-priority hard certificate | 成熟 | **Freeze** |
| Post-certificate set-preservation frontier | 有负消融支持 | **Freeze** |
| Outcome head | diagnostic-only | **Freeze** |
| 8 s conventional contract | 稳定语义基准 | **Freeze** |
| V27 conventional integrity | solved | **Freeze** |
| V28 no-valid execution integrity | solved | **Freeze** |
| Candidate families | 当前不证明不足 | V36 **Freeze** |
| Common controller | 成熟接口 | **Freeze** |
| RVR max-prefix policy | negative；alternative 有诊断价值 | 不 promotion |
| SOV successor signal | positive mechanism | absorbed |
| BHOV/THOP | negative | Archive |
| unconditional commitment | negative policy / positive mode insight | Archive policy |
| ROSH semantic spectrum | positive representation | 保留 |
| EOSH nominal exact filtering | negative | Archive |
| WK-ROSH current kinematic guard | clean no-added-value | Archive |
| CPOSH control-projected spectrum policy | Gate FAIL | Archive implementation |
| Control-projected spectrum observable | **positive** | 保留并用于 V36 |
| Existing-bank semantic recovery frontier | **未成熟 / P0** | V36 主攻 |
| Accepted-path kinematics | secondary | 后续独立版本 |
| True reachable proposal support | long-term ceiling | V36 gate fail 后提升优先级 |

---

# 7. 模型下一步应该学/表示什么

social axis 已经比较会回答：

> ego 是否通过压缩别人的 natural low-burden option set 获得所谓 safety？

physical recovery side 到 V35 已经开始会回答：

> 一个当前 action 的 causal successor 是否保留较丰富的 control-realizable semantic recovery options？

但仍然不会回答：

> **当前 fixed bank 中，哪些语义 recovery branches 构成 physical-feasible frontier；在满足 hard current/future feasibility 后，哪一个是对已成熟 COWP preference 最小扰动的 recovery branch？**

V36 因此先做 analytic frontier，不训练 neural head。

如果 analytic frontier 得到稳定证据，再考虑学习一个 amortized frontier predictor 才有意义；反之现在训练 classifier 只是在拟合未经验证的 target。

---

# 8. V16.8.36 机制设计

## 方法：`cowp_control_projected_recovery_frontier`

缩写：CPRF。

### 8.1 介入条件完全不变

只在：

```text
full conventional set == empty
AND valid candidate exists
```

时介入。

certificate/conventional path、no-valid emergency path 都完全不改。

### 8.2 existing-bank semantic representatives

先使用与 RVR 完全相同的 roadgraph-first recovery pool：

```text
pool = valid & roadgraph_safe, if nonempty
else valid
```

对每个 distinct non-PAD macro：

```text
representative(m)
= argmax current collision-safe prefix within macro m
  tie -> min frozen COWP fallback score
  tie -> min candidate index
```

没有增加 trajectory。

### 8.3 hard current-survival component

任何 alternative 必须满足：

```text
H0_alt >= H0_base
```

不允许用更差的 immediate collision survival 换 richer future profile。

### 8.4 future physical component

对每个 representative 复用 V35 的 control-projected option spectrum：

```text
P_ctrl_alt(h) >= P_ctrl_base(h), for every h
```

并在 current-prefix / future-profile 至少一个维度 strict better 才允许 inactive entry。

这仍然是 product partial order，不做 scalarization。

### 8.5 frontier 内排序

所有 hard-admissible representatives 构成 recovery frontier。

不选：

- profile area 最大；
- longest prefix 最大；
- collision score + utility weighted sum。

而是：

```text
chosen = argmin frozen COWP fallback score
         over hard-admissible semantic frontier
```

即：

> **physical feasibility first → existing COWP least-coercive preference second**。

这与当前 social certificate 后的 set-preservation 思想一致。

### 8.6 semantic mode consistency

inactive：strict product dominance 才 entry。

active：只允许当前 active macro 的 representative 在 weak dominance 下 continue。

不允许：

```text
YIELD recovery -> MERGE_AHEAD recovery -> STOP recovery
```

逐拍直接跳 macro。

一旦 active macro lose weak dominance，立即 exit base；conventional/certificate 恢复或 no-valid 时清 state。

没有 dwell time、epsilon、release margin。

---

# 9. V36 如何判断机制是否真的工作

Stage-1 outcome gate仍然一字不改。

额外记录、但不作为事后新增 GO 条件：

- mean semantic representative count；
- evaluated profile count；
- strict / weak admissible frontier size；
- current-prefix-admissible count；
- recovery switches 中选择 historical RVR 的比例；
- recovery switches 中选择 **non-RVR representative** 的比例；
- selected fallback-score delta；
- selected current-prefix delta。

非常重要：

> 如果 `selected_non_historical_rvr_rate_on_switches == 0`，即便 outcome 有波动，也不能声称“frontier support utilization”得到验证，因为方法事实上仍退化成原 binary endpoint。

但这一项不加入 inherited GO gate，避免看到结果后新增 outcome gate。

---

# 10. V36 之后的预注册分叉

### A. V36 Stage-1 通过，且 non-RVR frontier 真正被使用

进入 fresh37。

如果 fresh37 仍 non-harmful，说明 physical axis 可以从“spectrum observable”升级到：

**semantic recovery frontier / control-realizable option-set feasibility**。

### B. outcome 通过，但 non-RVR usage≈0

不把结果归因给 frontier。说明 candidate bank 在当前场景上 effective support 仍基本只有原 endpoints；应检查 representative construction / true reachable support，而不是宣传算法成功。

### C. V36 FAIL，尤其仍有 old-induced false positives

正式停止 ROSH/EOSH/CPOSH/frontier selector family。

下一步转向：

1. **reachable recovery proposal/support construction**；或
2. 更高保真 **closed-loop reachable set / returnability representation**。

不再做 profile threshold、AUC、horizon、penalty patch。

### D. collision-side恢复稳定后，Kinematics 仍主要来自 accepted path

单独开启 Execution-Viability Certificate 版本；不与 recovery 混做。

---

# 11. CCF-A 研究价值判断

V36 不能把“多个 backup branches”包装成 novelty。Contingency planning、backup-plan safety、multi-horizon backup feasibility、branching trajectory planning 都已有成熟研究。

当前值得维护的论文主线仍是：

# **Orthogonal Option-Set Feasibility**

统一原则：

> **Safety should not be obtained by collapsing a critical viable option set.**

Social axis：

```text
natural roots
-> same-root RCOT
-> BCOT
-> protected-priority low-burden option preservation
```

约束 ego 不压垮**其他 critical agents** 的 natural low-burden choices。

Physical axis：

```text
actual emitted action
-> control-conditioned causal successor
-> semantic recovery support
-> control-projected physical frontier
-> dominance-consistent recovery mode
```

约束 uncertified recovery 不压垮 **ego 自己** 的 future control-realizable choices。

真正可能达到 CCF-A standard 的不是某个 hysteresis 或 frontier trick，而是：

1. 两种 option-set feasibility 的统一形式；
2. social/physical 两轴语义正交；
3. hard feasibility 与 utility 分离；
4. same-root / causal / no-GT-leakage 的证据链；
5. proposal sufficiency 与 selector/certificate quality 的独立审计；
6. final unseen evaluation + multi-seed paired confidence intervals + reactive-agent/human-audited stress evidence。

---

# 12. 后续禁止方向

继续禁止历史 negative branches：

- CTU / certificate→planner-score replacement；
- outcome fallback weight search；
- outcome head hard shield；
- 缩短 8 s conventional horizon；
- direct RVR promotion；
- Pareto tolerance / risk weighting；
- BHOV comparator relaxation；
- V3/V4/V5 horizon stacking；
- unconditional commitment；
- fixed N-step dwell；
- hysteresis epsilon/margin tuning；
- profile AUC / horizon discount；
- social+physical+utility scalarization；
- RCOT/BCOT threshold/budget tuning；
- analytic target 未验证前训练 successor/recovery head；
- recovery 与 accepted-path kinematics 同轮修改；
- nominal first-waypoint exact reachability 作为 executable definition；
- 用 internal accel/jerk/yaw proxy 冒充 Waymax kinematics metric；
- 全局 retune common controller。

V36 新增禁止：

- 不再继续 binary base-vs-global-RVR comparator patch；
- 不通过 profile/curvature/EP 权重搜索救 CPOSH；
- V36 之前不扩 map/Frenet primitive；
- frontier 内禁止用 largest-profile / longest-prefix scalar ranking；
- active recovery 时禁止跨 semantic macro 直接跳转；
- V36 analytic frontier 未验证前不训练 frontier neural selector。

---

# 13. 下一步运行指令

```bash
cd COWP_v16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh make_ids

# only when TFExample index is missing
bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh base_equivalence16_parallel2

bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh analyze_counterfactual48
```

**到这里停止。**

只有：

```text
preregistered_gate.control_projected_recovery_frontier.pass == true
```

才执行：

```bash
PROMOTED_METHODS=cowp_control_projected_recovery_frontier \
bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_36_CONTROL_PROJECTED_RECOVERY_FRONTIER_CN.sh analyze_fresh37
```

fresh37 再 pass 才进入 exact200 development confirmation。

---

# 14. V16.8.36 代码验证

当前交付：

- V36 new tests：4/4 passed；
- V35+V36 focused tests：10/10 passed；
- V16.8.25→36 focused suite：**53/53 passed**；
- `py_compile` passed；
- launcher `bash -n` passed；
- manifest hashes passed；
- fail-closed promotion path passed：没有 Stage-1 analyzer 时直接运行 fresh37 返回 code 4，Waymax rollout 不会启动。

本轮最重要的研究判断可压缩为：

> **V35 证明 control-projected option spectrum 确实能提高 recovery recall，但也证明只在 COWP 与 global-RVR 两个 endpoint 之间做 sophisticated viability comparison 仍然结构性不足。V36 因而不再继续修 comparator，而是第一次把 existing bank 中的 semantic recovery support 显式变成 hard physical frontier。**
