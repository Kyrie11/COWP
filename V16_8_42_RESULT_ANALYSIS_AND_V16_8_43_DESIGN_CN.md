# V16.8.42 结果分析与 V16.8.43 设计：Blocker-Conditioned Reachable-Response Envelope

## 0. 结论先行

本轮遵守“可靠性通过后才归因”的规则。V16.8.42 独立可靠性审计 **63/63 blocking checks PASS**，因此算法归因允许。

V16.8.42 同时存在两个必须分开的结论：

1. **V42 policy 不通过冻结的 Stage-1 conjunction Gate，必须 archive，不得运行 fresh37。** 唯一失败项是历史 RVR rescue retention：3/10 < 5/10。
2. **RC-IARE 的 interaction-aware hard support object 是当前 lineage 中最强的新正机制信号之一。** 在同一 48-scene development panel 上，V42 相对普通 COWP 是 10 rescue / 0 induced collision，Offroad/Kinematics 无净退化，paired mean EP Δ≈+0.00022；其中 7 个 rescue 并不是历史 RVR rescue，而是新的 COWP collision rescue。

因此不能把 V42 简单归类为“算法没用”，也不能因为 headline 很好而事后修改 Gate。正确处理是：**archive V42 implementation as promoted policy；retain RC-IARE mechanism object；下一版本只修 V42 证据暴露的 support-indexing bottleneck。**

V16.8.43 的正式分支为：

> **Blocker-Conditioned Interaction-Aware Reachable-Response Envelope (BC-IARE)**  
> method: `cowp_blocker_conditioned_interaction_aware_reachable_response_envelope`

V43 exact-nests V42；只有 V42 hard set 为空时，才允许对冻结 collision context 中、但不在 scene-level critical set 中的潜在 exact blockers 进行 late-bound natural-root query。它不扩大社会 NCF critical set，不改 p_min / mass floor / retained mass / β / response bank / environment hard checks / joint CSP / V39 tube / controller / 8 s conventional contract。

---

## 1. 论文研究方向与本轮判断标准

论文的核心不是“social cost 更好”，而是 **false-safe / safety-by-coercion 是 feasibility defect**：ego 即使 collision-free，如果其安全依赖他车 hard braking、abrupt yielding、priority abandonment 或 gap surrender，也不能仅靠有限 soft burden weight 处理。COWP 的核心对象是 natural roots、same-root low-burden response preservation 和 protected-priority hard certificate。

论文还明确要求 **proposal/support sufficiency 与 certificate/selector quality 分开审计**。固定 bank 缺少支持时，继续调 threshold、ranking loss 或 fallback 无法突破 bank-dependent floor。当前多轮实验正是在沿这个原则逐层排除 selector、prefix、horizon、mode、nominal execution、existing-bank frontier、exact returnability、finite control schedules 和 first-action interval 等假设。

当前 compact-5k 的 split 仍稳定：train/val/heldout 的 audit-relevant pair rate 约 0.429，protected-priority root coverage 约 99.4%，rootless 和 <2 low-burden roots 为 0；本轮没有证据支持重建数据，且用户已明确本阶段不重建数据。

---

## 2. V16.8.42 可靠性审计

### 2.1 结果链通过

独立检查包括：

- exact200 / equivalence16 / counterfactual48 / fresh37 manifest 数量、唯一性、logical SHA256；
- counterfactual48 两 shard 24+24，无 overlap，union 精确等于冻结 48 IDs；
- equivalence16 两 shard 8+8，无 overlap；
- shard → merged scenario rows 精确重建；
- CR / Collision / Offroad / Kinematics / EP 可由逐场景 row 零误差重算；
- V28 `emergency_action_step_rate == zero_valid_candidate_step_rate == no_valid_step_rate` 未复发；
- equivalence16：16 scenes / 1120 fields / 0 mismatch；
- V42 key source files SHA256 与正式 V42 code-validation manifest 一致；
- pristine V42 analyzer 独立重跑后与上传 analyzer **0 recursive mismatch @1e-12**；
- pristine V42 dedicated tests **10/10 passed**；
- no logged future / no online GT / exact V39 nesting / residual physical certificate / joint CSP 等 release hard checks 均保持通过。

因此：

```text
V16.8.42 result reliability = PASS
algorithm attribution       = ALLOWED
```

### 2.2 非阻断证据边界

- 结果包未包含 checkpoint bytes，因此只能核对 logical checkpoint provenance，不能独立重算 `.pt` 内容 hash；
- result JSON 未嵌入 server runtime source-tree hash；上传源码与正式 release hash 一致，但仅凭结果包不能做服务器源码的密码学重建；
- non-ego 使用 Waymax logged replay，可以做冻结协议下的 ego closed-loop physical attribution，但不是 counterfactual social-burden GT；
- counterfactual48 已被多轮使用，是 development-selected panel，不是 publication holdout。

---

## 3. 严格按上一轮预注册 GO：V42 policy FAIL

冻结 Gate 一个数字不改：

| 条件 | GO | V42 | 判决 |
|---|---:|---:|---|
| old RVR rescues retained | >=5/10 | **3/10** | **FAIL** |
| old RVR induced avoided | >=7/9 | **9/9** | PASS |
| net COWP collisions removed | >=3 | **10** | PASS |
| Kinematics net regression | <=1 | **0** | PASS |
| paired mean EP Δ vs COWP | >=-0.05 | **+0.000219** | PASS |
| action-changing intervention | >0 | **318 steps** | PASS |

所以：

```text
V16.8.42 RC-IARE
= RELIABLE
+ STAGE-1 FAIL
+ ARCHIVE POLICY
+ NO FRESH37
```

### 3.1 headline 不能被 Gate FAIL 掩盖

| Method | Collision | Offroad | Kinematics | EP |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 1/48 | 6/48 | 1.002512 |
| V39 | 30/48 | 1/48 | 6/48 | ~0.991434 |
| **V42 RC-IARE** | **24/48** | **1/48** | **6/48** | **1.002731** |

V42 vs COWP：

- collision = **10 rescue / 0 induced / net -10**；
- development-panel McNemar exact p≈0.00195（只能解释本开发 panel，不能写成 publication claim）；
- Kinematics = 0 rescue / 0 induced；
- Offroad = 0 / 0；
- paired mean EP Δ≈+0.000219，bootstrap95≈[-0.02995,+0.03274]。

V42 只复现 3 个 historical RVR rescue，却产生 **7 个新的 COWP rescue**。因此 old-RVR-recall Gate 与总物理 outcome 在这一轮不再同义。科研纪律要求仍然判 Gate FAIL，但机制判断必须记录“new support mode 确有真实作用”。

---

## 4. V42 真正成功的机制

### 4.1 interaction branch 不是 dead code

48×80=3840 policy steps 中：

```text
recovery probes                         2626
hard certificate steps                  319
action-changing steps                   318
interaction attempts                   2517
nested V39 selected certificates        109
interaction selected certificates       210
```

约 65.8% 的 certified selections 来自 V42 interaction extension，而不是 nested V39。V42 的 10 个 COWP rescues 中有 7 个不是历史 RVR rescue，说明 root-conditioned interactive recourse 是一个新的 support source，而不是旧 heuristic 的重命名。

### 4.2 应保留的 V42 机制对象

- exact V39 control/shift-closed physical backbone；
- actual emitted action semantics；
- canonical natural-root probability measure；
- same-root low-burden response bank；
- current + shifted response feasibility；
- exact blocker replacement，而不是删除 collision audit；
- responder 对 non-blocking environment 的双向安全；
- multi-blocker exact joint compatibility CSP；
- hard support first，冻结 COWP preference second；
- logged future 禁止进入 certificate。

这些机制与论文“不能靠压垮 critical option set 获得 safety”的主线一致，应继续保留。

---

## 5. V42 失败在哪里：reject decomposition 给出 dominant bottleneck

V42 约评估 **495,101** 个 interaction hypotheses。按 hypothesis 数量分解：

| reject stage | count | share |
|---|---:|---:|
| unsupported blocker | 282,610 | **57.08%** |
| retained root unrecoverable | 173,029 | **34.95%** |
| residual physical | 38,600 | **7.80%** |
| no blocker | 510 | 0.10% |
| joint incompatibility | 142 | **0.029%** |

所以当前最不应该做的是：

- 放宽 joint CSP；
- 删除 environment safety；
- 调 burden β；
- 调 p_min / mass floor / 0.75 mass / root count；
- 增加 scalar risk 权重。

它们都不是当前最大 reject source。

### 5.1 为什么 unsupported blocker 是结构问题而不是随机数据现象

当前 online critical selection 默认：

```text
max_online_critical_agents = 4
```

而冻结 collision context 默认：

```text
online_collision_max_agents = 24
```

V42 在 interaction attempt 开始前只为 scene-level online critical agents 准备 natural-root response support；随后在更宽的 24-agent collision context 中才识别 exact blockers。于是一个 blocker 只要落在 active critical-4 之外，就会直接进入：

```text
unsupported_collision_blocker
```

即使冻结 NaturalDecoder 实际能够给这个 agent 生成 natural roots。

更强的场景级证据：7 个 historical RVR rescue 中 V42 没保留的 7 个场景，共评估 112,707 hypotheses，其中：

```text
unsupported blocker rejects = 75,470 ≈ 66.96%
root unrecoverable rejects  = 30,277 ≈ 26.86%
residual physical rejects   =  6,929 ≈  6.15%
joint incompatibility       =      1
```

这把 P0 从“interaction response support 还不够”进一步收紧为：

> **Blocker-Conditioned Natural-Option Support Indexing / Coverage under Interaction-Aware Recovery.**

注意这不等于扩大 social certificate 的 critical set。社会 NCF critical selection 已经成熟，不应因为 recovery blocker indexing 的不一致而扩大 hard social veto。

### 5.2 P0 的第二层

修复 blocker coverage 后，下一大类仍是：

> **same-root low-burden responder reachability / response support quality**

当前约 35% hypotheses 在 root response 侧失败。这是 V43 之后若仍失败时的优先分支，而不是现在同时修改。

### 5.3 独立 P1：certificate→长期闭环 mismatch

`fccd9a25a2a57a73` 仍属于 Type-B：局部 certificate/action change 存在，但长期 receding-horizon outcome 仍 collision。它需要 multi-step uncertainty/invariance 或 richer reactive-model mismatch 分支。本轮不与 blocker coverage 同时修，避免失去归因。

---

## 6. 当前每一层成熟度

| Layer | 状态 | V43 原则 |
|---|---|---|
| compact-5k data/label/split | Mature | **Freeze** |
| Natural roots / canonical root measure | Mature | **Freeze** |
| same-root RCOT | Strong/Mature | **Freeze** |
| BCOT | Strong/Mature | **Freeze** |
| protected-priority hard NCF | Mature | **Freeze** |
| certificate-compatible set preservation | Mature | **Freeze** |
| outcome head | Diagnostic-only | **Freeze** |
| 8 s conventional contract | Stable attribution contract | **Freeze** |
| V27/V28 integrity fixes | Solved | **Freeze** |
| common controller / shift semantics | Mature interface | **Freeze** |
| V39 hard physical tube + shift closure | high-precision backbone | **Retain exactly** |
| V40/V41 first-action interval | negative | Archive |
| V42 policy | Gate FAIL | Archive as promoted policy |
| V42 root-conditioned interaction support | **strong positive mechanism** | **Retain / exact nest** |
| blocker support indexing | **P0 / immature** | **V43 only variable** |
| root-conditioned responder reachable support | next P0 if V43 coverage works but recall fails | Do not modify yet |
| certificate→long-horizon invariance | P1 | separate later |
| accepted-path kinematics | secondary | separate later |
| final causal burden evaluation | immature | reactive + human audit |
| publication statistics | immature | unseen set + ≥3 seeds + paired CI |

---

## 7. V16.8.43：Blocker-Conditioned IARE

### 7.1 Exact V42 nesting

V43 先完整执行 V42：

```text
V42 certificate exists
    -> return V42 selection unchanged
    -> blocker support extension cannot replace it
```

这不是在声称 V42 已 promotion，而是为了做严格 one-factor attribution：V43 只测试 V42 hard set 为空时，support-index mismatch 是否造成 false reject。

### 7.2 不扩大 social critical set

V43 不修改：

```text
cowp/critical/track_index
RCOT/BCOT protected set
priority relation
hard NCF certificate
```

新增 query domain 只属于 **uncertified physical recovery certificate**。

### 7.3 late-bound blocker candidate pool

V43 从冻结的 current collision context 获取 model-visible、当前 valid、非 SDC、非原 critical agent 的 nearby agents；用 **root-scene graph latent** 调用冻结 NaturalDecoder，不能用 candidate-conditioned planner latent。

新增模型输出 `natural_scene_z_agent` 只是 already-computed tensor 的只读 view：没有新参数、没有 checkpoint change、没有 loss/training change。

### 7.4 冻结 natural response semantics

对 late-bound agents 完全复用 V42：

```text
p_min = 0.03
probability floor epsilon_p = 0.02
root mean-path dedup = 0.10 m
minimum stable roots = 2
retained canonical mass >= 0.75
AGENT_PRIORITY adaptive low-burden budget beta
same-root response bank
current + shift roadgraph / Waymax kinematics
responder <-> ego safety
responder <-> nonblocker environment safety
multi-blocker exact CSP
```

不调整任何阈值。

### 7.5 hard selection 不变

V43 仍使用冻结 V39/V42 action hypotheses 和 deterministic hard-set ordering。late-bound support 只改变“某个 exact blocker 是否拥有合法 natural low-burden recourse”，不生成新的 ego grid、schedule、controller limit 或 score。

### 7.6 无信息泄漏

V43 在线只使用：

- 当前 observed simulator state；
- 当前 map；
- current collision context；
- frozen checkpoint 的 root-scene natural latent/decoder；
- analytic root recovery bank / controller dynamics。

禁止：

- `log_trajectory` future；
- future Waymax state/outcome；
- online mechanism GT；
- scenario ID special case；
- dense learned response head；
- counterfactual48 outcome-conditioned parameter tuning。

---

## 8. V43 的预注册解释分叉

Stage-1 继续使用 **完全相同的六项 outcome Gate**，不是新增第七项。

机制 diagnostics：

```text
blocker_query_attempt_steps
blocker_query_selected_steps
query_agent_count / ready_agent_count
unsupported-blocker rejects after expansion
root-unrecoverable rejects after expansion
compatibility-cache hits
successor-context-cache hits
```

提前固定解释：

1. **Gate pass + blocker-query selection > 0**：support-indexing hypothesis 获支持，才运行 fresh37。
2. **Gate pass + query selection = 0**：不能把结果归因于 V43，最多是 nested V42。
3. **query 真正被选择、unsupported rejects 明显下降，但 Gate fail 且 root-unrecoverable 成为主导**：停止 indexing patch，进入 `root-conditioned control-reachable responder envelope`；不调 p_min/β。
4. **query agents 大量 decoder-not-ready**：审计 late-bound natural-root calibration / model-visible indexing；不在 CF48 上重训或调阈值。
5. **query selection >0 但出现 induced physical failure**：archive V43 scope；不放松 environment/joint constraints。
6. **`fccd...` 等 certificate→collision 继续存在**：下一独立版本研究 multi-step invariance / interaction-model uncertainty。
7. **V43 overall fail**：blocker indexing 不是主因；升级到真正 interaction-conditioned reachable response/support construction，不回到 ego selector/grid family。

---

## 9. V42 为什么慢，以及 V43 的语义等价加速

### 9.1 服务器事实

```text
counterfactual48_parallel2 wall = 55,440 s ≈ 15.4 h
equivalence16_parallel2 wall   = 343 s
```

V42 在 48 scenes 内部逻辑工作量：

```text
interaction hypotheses                 495,101
environment compatibility checks     7,813,492
joint compatibility checks              76,616
```

因此 V42 新增的主要 runtime 不是“多一次神经网络 forward”，而是大量 Python/NumPy responder-environment / responder-responder current+shift collision predicate 重复执行。

### 9.2 V43 只做 semantics-preserving memoization

在一个 policy step 内，以下对象对多个 ego hypotheses 不变：

- 某 `(blocker, root, response profile)` 与某 frozen environment actor 的 current/shift compatibility；
- 两个 response profiles 的 current/shift mutual compatibility；
- 相同 actual emitted first target + emitted accel 对应的 causal successor collision context。

V43 增加 policy-step local cache，但：

- 不跨真实 state/replan step；
- 不改变任何 safety predicate；
- logical check/reject counters 仍按“未缓存时会执行”的次数累计，便于与 V42 diagnostics 可比；
- cache hit 只作为额外性能诊断。

### 9.3 synthetic microbenchmark

针对重复 environment compatibility 子组件：

```text
80 repeated certificates
uncached: 0.9496 s
cached:   0.0853 s
subcomponent speedup: ~11.13x
unsafe predicate calls: 209,920 -> 7,680 (~27.33x reduction)
```

这只是 synthetic subcomponent benchmark，**不是服务器端整段 Waymax speedup 承诺**。V43 提供 `profile8_parallel2`，必须以服务器 profile8 的 end-to-end policy timing 为准。

---

## 10. 后续明确禁止的修改方向

继承 ALGORITHM_CHANGELOG 中全部历史禁区，并新增/强化：

- 不把 `max_online_critical_agents=4` 全局调大来“解决” V42；这会改变成熟 social certificate 的分布和 hard veto 语义；
- 不把 collision context 的所有 nearby actors 直接加入 social protected set；
- 不在 CF48 上调整 p_min、floor、retained mass、root count、dedup、β 或 response-bank primitives；
- 不因为 unsupported blocker 多就删除 blocker 的 universal root requirement；
- 不放松 non-blocker environment safety；
- 不放松 multi-agent joint CSP；
- 不重新启用 dense response head；
- 不增加 ego first-action grid、interval fractions、schedules、switch times、horizon stacking；
- 不缩短 8 s conventional horizon；
- 不做 social+physical+utility scalarization；
- 不重启 RVR/ROSH/CPOSH/frontier selector patch；
- 不同时修 `fccd...` Type-B long-horizon mismatch；
- 不同时修 accepted-path Kinematics；
- 不重建 compact-5k；
- 不把 logged-replay outcome 当 causal burden evidence；
- 不把 CF48/fresh37/exact200 development universe 当 final publication holdout。

---

## 11. CCF-A 主线与 novelty 边界

当前最统一的论文主线仍建议写成：

> **Orthogonal Option-Set Feasibility: safety must not be obtained through critical option-set collapse.**

Social axis：

> ego 不应通过压缩其他 protected agents 的 natural low-burden response set 获得所谓 safety。

Physical-interactive axis：

> uncertified recovery 不应依赖一个在真实 control/shift-closed ego tube 下、对 exact interaction blockers 没有 root-consistent low-burden joint recourse 的假可行 continuation。

V43 的“late-bound blocker query”本身不是 CCF-A novelty；generic safety filter、contingency/reachable-set planning、backup feasibility、action-space reduction 都已有广泛相关工作。真正有论文价值的是：

1. false-safe / safety-by-coercion 的 hard feasibility semantics；
2. stable natural roots + same-root transport / low-burden preservation；
3. protected-priority hard social feasibility；
4. control/shift-closed physical support；
5. exact blocker-conditioned root-consistent joint recourse witness；
6. proposal/support sufficiency 与 certificate soundness 的分离论证；
7. logged replay / reactive agent / human-audited causal evidence 的分层 protocol。

正式投稿前仍必须做最新 literature clearance。当前 literature 已表明 safety filters、proactive recourse/contingency planning、reachable-set barriers、Feasible Action-Space Reduction 都是活跃且拥挤的相关方向，所以不能把“有 backup / 有 reachable set / 有 option reduction”单独当 novelty。

---

## 12. 下一步执行顺序

```bash
cd COWP_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/home/senzeyu2/code/COWP/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh make_ids

# 只有 index 缺失时：
bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh base_equivalence16_parallel2

# 可选：先测服务器真实速度，不作为算法证据
bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh profile8_parallel2

# Stage-1
bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_43_BLOCKER_CONDITIONED_REACHABLE_RESPONSE_ENVELOPE_CN.sh analyze_counterfactual48
```

到这里停止。只有：

```text
preregistered_gate.blocker_conditioned_interaction_aware_reachable_response_envelope.pass == true
```

才运行 fresh37；fresh37 再过才运行 historical exact200 development confirmation。

最终算法 freeze 后，需要新建从未参与 V25–V43 mechanism selection 的 final evaluation set，至少 3 independent seeds + paired scenario CI；强 causal burden claim 继续要求 reactive-agent + held-out human-audited false-safe stress set。
