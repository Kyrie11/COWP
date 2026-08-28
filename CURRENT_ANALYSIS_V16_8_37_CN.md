# V16.8.36 结果分析与 V16.8.37 设计

## 0. 结论先行

本轮按实际版本链 **V16.8.36 → V16.8.37** 推进。

结论分三层：

1. **V16.8.36 实验结果通过可靠性审计，可以进行算法归因。** 独立机器审计 53/53 hard checks passed；equivalence16 继续是 16 scenes / 1120 fields / 0 mismatch；counterfactual48 两个 24-scene shard 无重叠且精确覆盖 manifest；merged standard metrics 可由 scenario rows 零误差重算；V36 analyzer 重跑与上传结果 0 recursive mismatch；没有 online future-GT leakage；V28 no-valid execution invariant 没有复发。
2. **V16.8.36 按上一轮原封不动的预注册 GO gate 明确 FAIL。** 它只避免 3/9 old-RVR-induced collision，相对 COWP 6 rescue / 6 induced，净 collision gain=0，Kinematics +2。不能 promotion，也不能跑 fresh37。
3. **V36 的失败不是“frontier 没真的用起来”，而是更强的机制反证：option richness / semantic support utilization 不是 returnability。** V36 的 62.85% recovery switches 已经选择了 non-historical-RVR representative，但相对 V35 CPOSH 仍变成 1 rescue / 8 induced（McNemar exact p≈0.0391）。真正区分 harmful recovery 的不是“有多少 valid/recovery options”，而是能否回到原 full-conventional feasible region。

因此 V16.8.37 不继续调 V36 frontier、weak dominance、hysteresis 或 spectrum，而转向上一轮 changelog 已预注册的更高层分支：

> **Recourse Returnability Bridge：显式验证 current recovery action 是否保留一次真实 replanning 后返回 full-conventional feasible set 的 causal recourse witness。**

---

# 1. V16.8.36 可靠性审计

## 1.1 Manifest / shard

- equivalence16：两个 shard 各 8 scenes，0 overlap，union 精确等于冻结的 16-ID manifest。
- equivalence16 logical SHA256：`81d0319da0446d1452b4c3a0361ffa6941dfa226b2f14027cac5576f9571c760`。
- counterfactual48：两个 shard 各 24 scenes，0 overlap，union 精确等于冻结的 48-ID manifest。
- counterfactual48 logical SHA256：`ee3c231c240878d5d20020aec3c98efbb4932cdbf1f1e309b9b7b26bddc40ab0`。
- checkpoint / method / gate 配置在两个 shard 间一致。

## 1.2 Summary 可重算

对每个 scenario row 独立重算：

- CR
- CollisionRate
- OffroadRate
- KinematicsInfeasibilityRate
- EP

与 merged summary 的最大误差为 0。

## 1.3 Common-path equivalence

`equivalence16_cowp_vs_v16_8_29.json`：

- scenarios=16
- fields_checked=1120
- mismatches=0
- passed=true

所以 V36 的新 recovery branch 没有悄悄改变成熟的 ordinary COWP common path。

## 1.4 Analyzer 可复现

按 launcher 的相同参数重新执行 V36 analyzer，并与上传 analyzer JSON 递归比较：

- tolerance = 1e-12
- mismatch = 0

## 1.5 Execution integrity / leakage

- `mechanism_ground_truth_available_online=false`
- non-ego future source 仍是 honest logged-replay protocol 中用于 simulator 的标准设置；机制 counterfactual 内部仍使用 frozen causal CV，而不是读取未来 GT 来选 action。
- `emergency_action_step_rate == zero_valid_candidate_step_rate == no_valid_step_rate` 的 V28 invariant 在审计范围内保持成立。

因此没有出现 V26/V27/V28 那类会阻断 attribution 的工程问题。

完整机器审计：`V16_8_36_RESULT_INTEGRITY_AND_ATTRIBUTION_AUDIT.json`。

---

# 2. 按上一轮预注册 GO gate：V16.8.36 明确失败

Stage-1 gate 没有任何调整：

| 条件 | GO | V36 |
|---|---:|---:|
| old RVR rescues retained | >=5/10 | **5/10 PASS** |
| old RVR induced avoided | >=7/9 | **3/9 FAIL** |
| COWP collision net removed | >=3 | **0 FAIL** |
| Kinematics net regression | <=1 scene | **+2 FAIL** |
| paired mean EP delta | >=-0.05 | **-0.02443 PASS** |
| intervention | >0 | **PASS** |

所以科研纪律上结论只有一个：

**V16.8.36 `cowp_control_projected_recovery_frontier` 不 promotion；禁止运行 fresh37。**

不能因为 EP 比 V35 好、或者 old rescue 恰好达到 5/10，就忽略三个 hard failure。

---

# 3. V16.8.36 的 closed-loop 结果意味着什么

## 3.1 相对 COWP

| Method | Collision | Kinematics | Offroad | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 6/48 | 1/48 | 1.002512 |
| V35 CPOSH | **27/48** | 7/48 | **0/48** | 0.913987 |
| V36 Frontier | 34/48 | 8/48 | **0/48** | 0.978086 |

V36 vs COWP collision paired transition：

- 6 rescue
- 6 induced
- 28 shared failure
- net=0
- McNemar exact p=1.0

所以 V36 没有 closed-loop collision gain。

## 3.2 相对 V35 CPOSH：这是决定性负证据

V36 vs V35 CPOSH：

- collision rescue：1
- collision induced：8
- net regression：+7 failures
- exact McNemar p≈0.0391

也就是说，V36 扩大 current semantic frontier 后，不是“信号差不多”，而是在这组已知 high-information development scenes 上显著破坏 V35 已得到的 collision gain。

这条证据强到足以停止：

- 再调 frontier fallback order；
- 再放宽/收紧 weak dominance；
- 再增加 semantic reps；
- 再加 spectrum score。

---

# 4. V36 确实测试到了 broader support，不是代码没生效

这是本轮很重要的 attribution 边界。

V36 的 recovery switches 中：

- selected non-historical-RVR rate ≈ **62.85%**
- selected historical-RVR rate ≈ 37.15%

所以它没有退化回 V35 的 base-vs-RVR pair。

而且 rescued 与 induced 两组都大量使用 non-RVR：

- rescued：约 **77.4%**
- induced：约 **75.0%**

current prefix gain 也几乎一样：

- rescued：约 +5.20 steps
- induced：约 +5.36 steps

因此下面两个 hypothesis 都被否定：

1. “只要把更多 existing-bank semantic branches 暴露给 selector 就会更好”；
2. “prefix + semantic option-spectrum hard dominance 已经足够区分 recovery viability”。

---

# 5. 真正的 separating signal：returnability，而不是 richness

V36 rescued scenes：

- zero-conventional ≈ 67.5%
- mean conventional candidates ≈ 3.77
- mean valid candidates ≈ 35.23

V36 induced scenes：

- zero-conventional ≈ **92.5%**
- mean conventional candidates ≈ **0.79**
- mean valid candidates ≈ **37.19**

这是目前最干净的新证据之一：

> harmful state 并不是没有 proposal；它有很多 valid choices，但几乎没有真正回到 conventional-feasible region 的能力。

因此：

**valid/recovery option richness 与 certified feasibility returnability 是两个不同的对象。**

V33/V35 的 spectrum 捕获前者的一部分，V36 更完整地暴露 existing support，但仍没有显式建模后者。

---

# 6. V36 induced collision 的 first-event attribution

V36 相对 COWP 新产生的 6 个 collision：

- `fe51445d725b8b8b`
- `3919ccd73c0fabd7`
- `d632f1919fe4bab`
- `c34fe8e79cdf1161`
- `f8d4c735825e5d81`
- `6418b0c9e2e4b093`

共同点：first collision 前一步仍是：

`no_conventional_use_control_projected_recovery_frontier`

并且：

- candidate valid=true
- conventional=false
- collision_safe=false
- selected collision-safe prefix 已经为 0
- zero-conventional reason 仍以 collision-side collapse 为主
- 不是 no-valid emergency

所以不能把它解释成：

> recovery 成功把系统送回 safe region，后来 accepted path 又出事。

事实相反：

> **系统从未真正退出 uncertified recovery basin。**

因此 collision P0 仍然是 recovery physical feasibility，而不是 accepted-path kinematics。

---

# 7. Mode dynamics 的新证据

V36 overall：

- entry ≈3.23% policy steps
- active ≈34.4%
- continue ≈31.1%

V36-induced group：

- active ≈80.4%
- continue ≈76.7%

两个 smoking scenes：

### `d632f1919fe4bab`

- zero conventional ≈100%
- entry≈1.25%
- continue≈98.75%
- active≈100%
- first collision around step 37

### `f8d4c735825e5d81`

- zero conventional≈100%
- entry≈1.25%
- continue≈98.75%
- active≈100%
- first collision around step 19

这说明 V36 的 strict-entry / weak-continue semantic mode 仍有一个结构缺陷：

> 一次 rare strict entry 可以形成近乎整个 episode 的 recovery continuation，即使系统始终没有重新获得 conventional support。

但下一步**不能**简单把 weak continuation 改成 stronger threshold/hysteresis margin，因为：

- V31 已经证明 stateless switching 有 harmful hybrid dynamics；
- V32 unconditional commitment 又已经证明 over-commit；
- V36 再证明 weak dominance continuation 仍可以形成 long trap。

问题不是需要一个更好的 dwell/hysteresis 参数，而是 recovery mode 缺少**明确 terminal semantics：我要回到哪里？**

---

# 8. 代码层发现：V36 有一个非阻断但值得修正的语义不一致

V36 `_recovery_frontier_mode_choice_np` 的 written intent 是：emitted action exact-equality 不能作为新的 strict physical branch。

但 current-prefix strict improvement 在 exact-equal target 分支仍可以让 `strict_map[rep]=True`。这意味着在极端情况下：

- 当前实际执行 action 没变化；
- 但 recovery macro state 可以 strict-enter。

这没有破坏本次 V36 结果可靠性，因为：

- 代码与上传结果一致；
- analyzer/diagnostics 可重复；
- 它不是 manifest/data/execution bug；

但它进一步说明：**semantic mode state 与真实 control-state 之间仍可能脱节。**

V37 不修补这个 V36 branch，因为 V36 已 archive；V37 直接取消 weak/free-running recovery state，改成有明确 terminal witness 的 one-replan bridge。

---

# 9. 当前真正成功的机制

## 9.1 Mature / 应冻结

以下部分已有多轮正证据或 negative-control 保护：

- compact-5k data/label contract
- Natural roots
- RCOT same-root transport
- BCOT
- protected-priority hard non-coercive certificate
- post-certificate set-preservation frontier
- V27 conventional-safety integrity
- V28 no-valid execution integrity
- 8 s conventional-safe contract

RCOT/BCOT 的历史 held-out evidence 仍明显强于 generic candidate false-safe classifier；CTU 已证明删除 certificate-compatible set-preservation、直接 certificate→planner-score argmin 是错误方向。

这些层后续不能为了 physical recovery 的问题重新调 threshold/budget。

## 9.2 成功但尚未成熟的 mechanism object

### Successor / option-set information

V30 SOV 高精度挡住 harmful RVR，证明 successor option set 有独立信息。

### Semantic recovery option spectrum

V33 ROSH 明显超过只改 mode 的 SDH，说明“整个 semantic option survival structure”比 longest-prefix 更合理。

### Control-projected physical representation

V35 CPOSH 相对 COWP 10 rescue / 3 induced，net -7 collision，说明 actual control realization 后的 future option-set 信息是真正有效的 physical signal。

这些 representation insight 应保留。

## 9.3 本轮明确失败并 archive

### V36 existing-bank semantic frontier

它真正使用了 broader support，却相对 V35 明显恶化，因此不能加入主机制。

### Weak-dominance semantic continuation

在 induced scenes 形成很长 active trap；不能继续调 continuation comparator。

---

# 10. Layer-by-layer 成熟度

| Layer | 当前状态 | 下一步 |
|---|---|---|
| compact-5k data/labels | Mature | **Freeze** |
| Natural roots | Mature | **Freeze** |
| RCOT | Strong/Mature | **Freeze** |
| BCOT | Strong/Mature | **Freeze** |
| Protected-priority certificate | Mature | **Freeze** |
| Post-certificate set preservation | Mature | **Freeze** |
| Outcome head | Diagnostic only | **Freeze** |
| 8 s conventional contract | Stable semantic baseline | **Freeze** |
| V27/V28 integrity | Solved | **Freeze** |
| current candidate families | 尚未证明完全不足 | **V37 Freeze** |
| common action controller | Mature interface | **Freeze** |
| RVR | policy negative | Archive/reference only |
| SOV successor signal | Positive | retain insight |
| BHOV/THOP | Negative | Archive |
| unconditional commitment | Negative | Archive |
| ROSH spectrum | Positive representation | Retain insight |
| V34 executable hard filter | Negative | Archive |
| V35 CPOSH | policy FAIL, physical signal positive | Retain as pre-gate/reference |
| V36 semantic frontier | **FAIL** | Archive |
| **returnability representation** | **P0 / immature** | **V37** |
| accepted-path kinematics | Secondary independent bottleneck | 后续单独 |
| genuine reachable proposal support | next ceiling | V37 fail 后进入 |

---

# 11. 模型下一步真正应该“学”什么

当前 physical side 逐渐已经会描述：

> “这个 action 后我还有多少 recovery options？”

但它还不会描述：

> **“这个 action 后，我是否仍保留一条 control-realizable recourse path，把系统带回 certified physical feasibility？”**

这是两个不同的 target。

一个 state 可以有：

- 7 个 semantic macros；
- 很长的 prefix；
- 丰富的 control-projected spectrum；

但如果这些 branch 都在同一个 zero-conventional basin 内循环，仍然不是好的 recovery state。

所以下一步真正需要学习/验证的 object 是：

**Recourse-to-Feasibility / Returnability Witness**。

当前先用 analytic causal witness 验证 target，本轮仍不训练新 neural head。

---

# 12. V16.8.37：Recourse Returnability Bridge

方法名：

`cowp_recourse_returnability_bridge`

## 12.1 只在 unchanged P0 regime 介入

仍然只有：

```text
full conventional set == empty
AND valid candidate exists
```

才运行。

certificate path、conventional fallback、no-valid emergency 完全不改。

## 12.2 当前 controlled pair 回到 V35 clean pair

V36 已经 falsify broader semantic current frontier，所以 V37 不再继续 exposure expansion。

当前只保留：

```text
base = 原 COWP least-coercive-valid fallback
alt  = historical global RVR max-prefix candidate
```

但 V37 不再用 spectrum/hysteresis 决定 indefinite policy；它只把 alt 当成一个**当前 recovery entry proposal**。

## 12.3 V35 CPOSH strict signal 作为 coarse pre-gate

只有满足：

- emitted action physically distinct；
- current prefix `H0_alt >= H0_base`；
- V35 control-projected successor option profile strict-dominates base；
- current emitted transition 满足 benchmark-aligned kinematics contract；

才计算 expensive returnability witness。

这不是 promotion V35 implementation，而是保留目前最强的 positive physical representation，避免 V37 把搜索空间重新扩大成 V36 的失败模式。

## 12.4 Direct restoration

执行 current emitted action 得到 causal successor `s1`。

重新生成 unchanged online physical bank。

若：

```text
exists full-conventional candidate at s1
```

则记为 direct restoration。

## 12.5 One-replan recourse set

如果 `s1` 仍 zero-conventional：

1. 在 `s1` 真正重新生成 candidate bank；
2. 用 carried emitted acceleration 计算每个 candidate 真正将执行的 one-step controller target；
3. 保留 valid + roadgraph-safe + positive safe-prefix + Waymax-kinematic-feasible；
4. 每个 non-PAD semantic macro 只取一个 deterministic max-prefix representative；
5. 对这个 **new replanning action** 再执行一个 causal successor；
6. 如果新 successor 出现 full-conventional candidate，则该 macro 属于 witnessed recourse set `R(a)`。

重点：

**这里不是 original candidate waypoint t+2。**

所以与 THOP 的 horizon stacking 是不同机制。

## 12.6 Returnability set partial order

不使用 count/AUC/weight：

```text
alt direct restore, base not -> strict accept
base direct restore, alt not -> reject
both direct restore          -> returnability tie
neither direct restore:
    require R_base proper-subset R_alt
incomparable sets            -> reject
```

因此同样拥有两个 macro，但语义完全不同的两个 recourse set 不能靠 “count=2” 打平或排序。

## 12.7 One-real-replan bridge state

如果 alt strict returnability win，但不是 direct restore：

```text
bridge_pending = True
```

只允许存在到**下一个真实 policy step**。

下一真实 step：

- 若 certificate/conventional 已恢复：直接 ordinary COWP，clear pending；
- 否则重新从 actual simulator state 构建 bank；
- 找每个 semantic macro 的 representative，其 actual one-step successor 能 direct restore conventional support；
- 在这些 direct-restoring representatives 中选 frozen COWP fallback score 最低者；
- 执行一次；
- 无论是否成功找到，pending 都立刻 clear。

如果没找到 direct-restoring candidate：

**立即 abort → 原 COWP base。**

没有：

- weak equality continuation；
- minimum dwell；
- 5-step/10-step commitment；
- hysteresis epsilon；
- release threshold；
- learned mode classifier。

所以 V37 直接针对 V36 “entry 很少但 continue 近整个 episode” 的 failure mode。

---

# 13. 为什么 one-replan 不是另一个 temporal hyperparameter

这里必须提前规定解释边界。

V37 的 one-replan 是一个**mechanism probe definition**：

> current action 后，在下一次真实 replanning opportunity 是否存在一个 control-realizable bridge 回到 full-conventional set？

它不是为了寻找 “最佳 horizon=2”。

所以如果 V37 FAIL：

**禁止把它改成 2-step / 3-step / 4-step returnability bridge 继续搜索。**

那时下一步应该正式进入：

- backward/forward reachable support；
- viability kernel / safe-set return set；
- learned dynamics reachable tube（前提是 target 先定义清楚）；
- structured recovery proposal generation。

---

# 14. CCF-A 主线怎样收紧

Generic contingency planning、backup plans、recursive feasibility、reachable safe set 都不是新概念。

所以论文不能写成：

> “我们提出一个 one-step backup action。”

真正值得维护的统一主线仍然是：

# Orthogonal Option-Set Feasibility

## Social axis

Natural roots → same-root RCOT → BCOT → protected-priority certificate

回答：

> ego 的 safety 是否靠压缩 **other agents' natural low-burden option set** 得到？

## Physical axis

Actual emitted action → causal replanning successor → control-realizable recourse witness → return to certified feasibility

回答：

> uncertified recovery 的 short-term survival 是否以压垮 **ego 自己回到 certified feasible set 的 recourse options** 为代价？

两条轴对象不同，但统一抽象进一步从：

> safety should not depend on critical option-set collapse

收紧到：

> **safety should preserve critical agents' viable options and the ego's recourse to certified feasibility.**

这比 `social score + collision head + kinematics penalty + recovery score` 更符合 CCF-A 级方法结构。

---

# 15. 后续禁止方向

历史所有禁止项继续有效，包括：

- CTU / certificate→planner-score replacement；
- outcome fallback weight tuning；
- outcome head hard shield；
- 缩短 8 s conventional horizon；
- direct RVR promotion；
- Pareto/risk weighted scalarization；
- BHOV comparator relaxation；
- V3/V4/V5 horizon stacking；
- unconditional commitment；
- fixed dwell；
- profile AUC/discount；
- RCOT/BCOT threshold/budget tuning；
- nominal exact waypoint executable filter；
- globally retuning common controller；
- 同轮 accepted-path kinematics 修改。

V36 之后新增禁止：

1. 不再调 V36 frontier / weak dominance / active macro continuation。
2. 不再把 valid candidate count / semantic macro count 当 viability target。
3. 不再通过增加 current semantic reps 修 selector。
4. V37 fail 后禁止继续 2/3/4-step bridge horizon search。
5. returnability 不做 count/AUC/weighted ranking；只用 hard direct-restore + set inclusion。
6. analytic returnability target 没验证前，不训练 returnability head。

---

# 16. V16.8.37 预注册实验协议

Stage-1 仍是同一个 counterfactual48，六项 gate **完全不变**：

| GO 条件 | 阈值 |
|---|---:|
| old RVR rescues retained | >=5/10 |
| old RVR induced avoided | >=7/9 |
| COWP collision net removed | >=3 |
| Kinematics net regression | <=1 |
| paired mean EP delta | >=-0.05 |
| intervention | >0 |

新增 returnability diagnostics 只用于 attribution，不是第七个 post-hoc gate。

只有：

`preregistered_gate.recourse_returnability_bridge.pass=true`

才能进入 fresh37。

fresh37 再通过才允许 exact200 historical development confirmation。

整个 exact200 仍然已经被历史机制选择污染，不能冒充 publication holdout。

---

# 17. 下一步执行命令

```bash
cd COWP_v16_8_37_RECOURSE_RETURNABILITY_BRIDGE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh make_ids

# 仅当 TFExample index 缺失时：
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh base_equivalence16_parallel2
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh analyze_counterfactual48
```

**到这里停止。**

只有 Stage-1 pass 后：

```bash
PROMOTED_METHODS=cowp_recourse_returnability_bridge \
bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh fresh37_parallel2

bash NEXT_RUN_COMMANDS_V16_8_37_RECOURSE_RETURNABILITY_BRIDGE_CN.sh analyze_fresh37
```

---

# 18. 当前代码验证

V16.8.37 新 tests：4/4 passed。

V16.8.25→37 focused semantic/integrity suite：**57/57 passed**；最终 launcher `sanity` 已重新执行并通过。

代码新增了：

- `_returnability_relation`
- `_returnability_witness_signature`
- `_direct_restoring_representatives_np`
- `cowp_recourse_returnability_bridge`
- one-real-replan `bridge_pending` state
- V37 episode diagnostics
- `90_analyze_recourse_returnability_bridge.py`
- fail-closed V37 launcher
- V37 regression tests

---

# 19. 下一轮最值得看的量

不能只看 Collision 从多少到多少。

最关键的是同时看：

1. old RVR rescues 是否仍 >=5/10；
2. old induced avoided 能否从 V35 的 6/9 提升到 >=7/9；
3. returnability probe 是否真的非零；
4. strict returnability entry 是否集中在 rescued 而非 induced scenes；
5. bridge pending 后 direct-restoring representative 是否在 actual state 上真实存在；
6. bridge abort 是否频繁——若频繁，说明 causal one-replan surrogate 与 actual closed loop mismatch；
7. zero-conventional exposure 是否在 rescued scenes 真正下降；
8. EP 是否避免 V35 `fccd...` 那类 severe progress basin；
9. first collision 前是否仍是 prolonged uncertified recovery。

如果 V37 通过 counterfactual48 + fresh37，那么 physical half 第一次拥有比 “option richness” 更接近 closed-loop recursive feasibility 的 clean evidence。

如果 V37 Stage-1 仍失败，则停止当前 selector/recovery-comparator lineage，直接进入真正的 **control-reachable recovery support / viability-set construction**。


## 18.1 Full repository 边界

最终 `pytest -x -q` 在 collection 阶段仍停在历史 external-baseline 测试：`tests/test_external_v4_validity_and_numerics.py` 无法从 `cowp.external_baselines.adapters` 导入 `candidate_geometry_finite`。我在用户上传的原始 V16.8.36 代码上单独复现了相同错误，因此它不是 V37 引入的 regression。没有把这个历史 full-suite 阻断伪报成 V37 full-suite passed。
