# V16.8.31 结果审计与 V16.8.32 Temporal Option Persistence 设计

> 用户消息写作 V12.8.31 / V12.8.32；上传包、代码 lineage 与此前版本均为 V16.8.31，因此本轮继续使用 V16.8.31 → V16.8.32 命名，避免破坏版本追踪。

## 0. 总结

V16.8.31 结果 **通过可靠性 gate，可以做算法归因**，没有需要 repair-only 停止分析的工程阻断。

但上一轮预注册的 promotion gate **没有通过**：

- counterfactual48 上，BHOV 从 COWP 的 34/48 collision 降到 29/48，保留 10/10 个旧 RVR rescue，并额外救回 1 个 COWP/RVR 都失败的场景；
- 但 BHOV 只避免 3/9 个旧 RVR induced collision，Kinematics 从 6/48 恶化到 9/48，EP 从 1.00251 降到 0.85246，paired EP delta=-0.15005，95% bootstrap CI 约 [-0.3277,-0.0198]；
- 更关键的是，与 V16.8.30 development panels 完全不重叠的 outcome-blind holdout64 上，COWP 0 collision，而 BHOV 新诱发 1 个 collision；因此 `holdout64 non-harmful` 条件失败，V16.8.31 不应进入 exact200 promotion；
- restoration-only 只在约 0.13% policy steps switch，0/10 old RVR rescue retained，几乎退化为 COWP，说明“是否立刻恢复至少一个 conventional option”这个二值统计过于稀疏。

因此：

**BHOV implementation 不 promotion；one-step successor option-set 这个机制信号继续保留；下一步必须拆分两个耦合根因：**

1. **one-step successor statistic 是否太短视，无法识别 delayed option collapse；**
2. **COWP/RVR 间歇式 hybrid switching 是否本身制造闭环失败，需要 recovery-mode commitment。**

V16.8.32 因此实现两个严格可归因分支：

- 主分支 `cowp_trihorizon_option_persistence`：BHOV 预筛 + 第二个 causal emitted-action successor option-set non-regression；
- 诊断分支 `cowp_sov_recovery_commitment`：严格 SOV 只负责进入 recovery mode，进入后持续 RVR，直到 full conventional feasibility 恢复；不使用固定 dwell time / tolerance。

两支都不改数据、checkpoint、RCOT/BCOT、protected-priority certificate、set-preservation frontier、outcome head、8 s conventional screen、candidate family 或执行投影。

---

# 1. V16.8.31 结果可靠性

## 1.1 代码语义与 focused regression

重新执行 V16.8.31 launcher `sanity`：

- **29/29 passed**；
- exact200 / equivalence16 / counterfactual48 / balanced96 / holdout64 manifest hash 全部通过；
- 没有 `conventional_check=False` safety bypass；
- V16.8.27 conventional integrity 与 V16.8.28 no-valid/PAD execution repair 未复发。

## 1.2 Equivalence16

- 16 unique IDs；
- shard 8+8；
- shard overlap=0；
- union=manifest；
- merged hash=`81d0319d...`，与 manifest 一致；
- 对 immutable V16.8.29 COWP reference：**1120 fields / 0 mismatch**。

所以新增 BHOV plumbing 没有伤害普通 COWP common path。

## 1.3 Counterfactual48

BHOV 与 restore-only：

- 均为 48 unique IDs；
- shard 24+24；
- overlap=0；
- shard union 精确等于 manifest；
- merged row IDs 精确等于 manifest；
- `scenario_ids_sha256=ee3c231c...`；
- CR / Collision / Offroad / Kinematics / EP 从 48 个逐场景 row 重算，与 merged summary 最大误差 **0**；
- checkpoint 在两个 shard / merged 中一致。

因此是严格 paired mechanism panel。

## 1.4 Holdout64

- 64 unique IDs；
- shard 32+32，overlap=0；
- union=manifest；
- hash=`becdc843...`；
- merged metrics 重算误差=0；
- 与 V16.8.30 equivalence16 / counterfactual48 / balanced96 overlap 均为 **0**；
- 全部属于 historical exact200 development pool。

所以它不是最终 paper holdout，但足以作为 V31 的“不要只在已知反例上优化”的 development non-harmfulness gate。

## 1.5 Waymax / causal interpretation 边界

这些结果中的 non-ego Waymax policy 仍是 logged replay；内部 physical successor probe 使用的是 current state + constant-velocity causal surrogate，并没有偷看 logged future。该设置足够用于本轮 planner/recovery 机制开发，但不能当作“ego 行为真实改变了他车反应”的强因果证据。最终论文仍需要 reactive-agent / human-audited stress protocol。

机器审计：`V16_8_31_RESULT_INTEGRITY_AND_ATTRIBUTION_AUDIT.json`。

**可靠性结论：PASS。**

---

# 2. 按上一轮预注册 GO 条件判定 V16.8.31

上一轮明确要求：BHOV 必须先在 counterfactual48 相对 SOV 提高 recovery recall，同时保持对 induced collision 的高 precision；之后必须在与 V30 panels 不重叠的 holdout64 上 non-harmful，才允许进入 exact200。

## 2.1 Counterfactual48：BHOV 有 signal，但不满足可 promotion 的整体质量

| Method | Collision | Kinematics | EP | recovery switch |
|---|---:|---:|---:|---:|
| COWP | 34/48 | 6/48 | 1.00251 | — |
| RVR | 33/48 | 9/48 | 0.82362 | unconditional zero-conv recovery |
| SOV | 33/48 | 6/48 | 0.99863 | 1.93% |
| **BHOV** | **29/48** | **9/48** | **0.85246** | **13.52%** |
| restore-only | 34/48 | 6/48 | 1.00263 | 0.13% |

BHOV vs COWP：

- collision rescue=11；
- induced=6；
- net -5 failures；
- McNemar exact p≈0.332；
- Kinematics：0 rescue / 3 induced；
- EP delta=-0.15005，bootstrap 95% CI≈[-0.3277,-0.0198]。

旧 RVR counterexamples：

- SOV：2/10 rescue retained，8/9 induced avoided；
- BHOV：**10/10 rescue retained，3/9 induced avoided**；
- restore-only：0/10 rescue retained，9/9 induced avoided。

所以 SOV→BHOV 的变化确实解决了“recall 太低”，但代价是基本回到了 RVR 的 harmful recovery regime。

BHOV 的正确结论不是“成功”，而是：

> one-step successor option set 具有判别力；但只要求它 non-regression，再让 current prefix 决定切换，仍然不足以预测 closed-loop delayed failure。

## 2.2 Restore-only：不值得 promotion

其 switch rate≈0.00130，几乎完全复制 COWP：

- 0/10 old RVR rescues retained；
- 9/9 old induced avoided；
- Collision 34/48 不变。

所以 binary `successor conventional exists 0→1` 太稀疏。下一步不应把 richer signature 丢掉，只保留 existence。

## 2.3 Holdout64：BHOV 明确失败 non-harmful GO gate

COWP：

- CR=1/64；
- Collision=0/64；
- Offroad=1/64；
- Kinematics=9/64；
- EP=1.08524。

BHOV：

- CR=2/64；
- **Collision=1/64**；
- Offroad=1/64；
- Kinematics=9/64；
- EP=1.06603；
- recovery switch≈10.94%。

BHOV 相对 COWP：0 rescue / **1 induced collision**。

该 scene 为 `3356dd85996d9c1d`。更关键的是：

- COWP：collision=0；
- pure RVR：collision=0；
- BHOV：collision=1，first collision step=59；
- BHOV fallback=100%，zero-conventional≈97.5%，recovery switch≈32.5%；
- first collision 前 action 是 uncertified `STOP_BEFORE_CONFLICT`，collision prefix=0。

因此这是一个 **hybrid-only induced collision**：不是 COWP endpoint policy 本身失败，也不是 pure RVR endpoint policy 本身失败，而是两者的时序混合进入了第三种闭环状态。

这条 evidence 很重要：V31 暴露的不仅是一拍 lookahead 太短，也可能有 **recovery-mode temporal consistency / commitment** 问题。

**预注册 verdict：NO-GO。不要运行 V31 exact200。**

---

# 3. 哪些机制真正成功，哪些失败

## 3.1 应继续 Freeze 的成熟主干

### Natural roots / RCOT
历史 held-out `LowSafeExist AUPRC≈0.897`，conflict/priority-conditioned 也有稳定 signal。没有任何 V28-V31 closed-loop 证据指向 natural-root 表征是当前 collision bottleneck。

### BCOT / protected-priority certificate
priority/global false-safe AUPRC≈0.837/0.928，显著强于 generic candidate false-safe classifier≈0.354。当前不要改 BCOT budget，也不要换成 generic safety classifier。

### Certificate-compatible set-preservation frontier
CTU 在 certificate invariant 的情况下，移除 frontier 并直接 planner-score argmin，使 learned-offline EP/PBTR/NCF recall 与 strict Waymax EP 都下降。该层是已获得正/负证据共同支持的成熟层。

### Outcome head
clean fallback-outcome experiment 没有 reliable physical gain，反而小幅伤害 EP；低-FPR recall 也不足以做 hard shield。继续 diagnostic-only freeze。

## 3.2 RVR：机制 signal 成功，policy 失败

RVR 的 10 rescues 证明 current collision-safe prefix 不是随机 statistic，且 fixed bank 中确有可改变未来物理状态的 recovery action。

但 9 induced + offroad/kinematics/EP regression 已经证明 greedy max-prefix 不能作为 policy。

保留它的角色：**controlled high-recall recovery alternative**，不再作为 candidate main policy。

## 3.3 SOV：值得吸收的是 successor option-set hypothesis

SOV 只 switch≈1.93%，却避免 8/9 old RVR induced，且几乎不伤 EP/kinematics。这是很强的 high-precision signal。

但 2/10 rescue recall 太低，所以 strict one-step successor improvement 不能直接 promotion。

## 3.4 BHOV：失败的 acceptance，不是否定 option-set representation

BHOV 放松为：current prefix non-worse + successor signature non-worse，至少一项 strict。

结果恢复了 10/10 old RVR rescues，却只挡 3/9 harmful RVR actions，并在 disjoint holdout 新诱发 collision。

所以失败点是：

- one-step option relation 不足以判断 delayed option collapse；
- current-prefix + V1 的 product order 仍然允许大量后续恶化；
- stateless per-step switch 可能产生 hybrid mode failure。

**不要再放松/tune BHOV。**

## 3.5 restore-only：binary restoration 太稀疏

0/10 rescue，0.13% switch。说明 future option richness 的 lower-order information（macro diversity / count / prefix）不能被粗暴压成一个 existence bit。

---

# 4. 完整证据链 V16.8.25 → V16.8.31

1. CTU negative：post-certificate option-transport/set-preservation 含有效 robustness ordering，不能删。
2. fallback-outcome negative：learned outcome risk 不能通过加权 fallback 解决 physical failure。
3. v27 repair：conventional-safe semantics 修复后，旧 fallback 归因才可信。
4. v28 repair：no-valid PAD execution 修复后，physical attribution 才真正 clean。
5. v28 exact200：collision first-event 主要发生在 `no_conventional_use_least_coercive_valid`，不是 conventional-safe fallback / accepted priority-NCF。
6. v29 decomposition：zero-conventional 主要是 `collision_empty≈52.5%`，roadgraph_empty≈0.1%，所以不是当前 map/Frenet bottleneck。
7. v29 RVR：相同 bank 能救 10 个 collision，但又造成 9 个新 collision，证明 bank 不是唯一 bottleneck，max-prefix 也不是 recursive viability。
8. v30 SOV：action-conditioned successor option set 能挡 8/9 harmful RVR，证明 future option support 是独立有效 signal；但只保留 2/10 rescue。
9. v31 BHOV：放松一拍 acceptance 可以恢复 10/10 rescue，但 precision、EP、kinematics 崩掉，并在 disjoint holdout 制造 hybrid-only collision。
10. v31 restore-only：单纯 conventional existence 太稀疏。

因此当前真正缺失的不是“再一个 risk score”，而是：

> **Temporal option-set persistence + recovery-mode consistency：一个 uncertified recovery action 不仅要让当前/下一拍看起来更可行，还必须让这种 option support 在闭环时间上持续，并且 recovery mode 的进入/退出不能制造 hybrid oscillation。**

---

# 5. Dominant bottleneck 再收紧

此前：`online conventional-feasible support collapse`。

V29 后：`collision-side recursive physical viability under zero-conventional recovery`。

V30 后：`temporal physical option preservation`。

V31 后进一步收紧为：

## **Temporal Option-Set Persistence and Recovery-Mode Consistency under Uncertified Recovery**

当前 P0 需要回答：

1. 一个 recovery action 的 option advantage 是否能跨多个 replanning interval 保持，而不是 V1 好、V2/V3 崩；
2. recovery action 是否需要成为一个显式 mode，而不能每 0.1s stateless 地 COWP/RVR 来回切；
3. 何时进入、何时退出 recovery mode，能否由 feasibility-set restoration 定义，而不是手工 dwell time / hysteresis threshold。

Proposal support 仍是长期 global ceiling；accepted-path kinematics 仍是独立 secondary bottleneck。当前不混入这两条，以保持归因。

---

# 6. 当前模型每层成熟度

| Layer | 状态 | 后续策略 |
|---|---|---|
| compact-5k data contract | Freeze | 不重建 |
| Natural roots | Mature / Freeze | 不动 |
| RCOT | Mature / Freeze | 不追小 AUPRC |
| BCOT | Mature / Freeze | 不调 budget |
| Protected-priority hard certificate | Freeze | 保持 feasibility semantics |
| Certificate-compatible frontier | Freeze | CTU 已否定 replacement |
| Outcome head | Diagnostic-only Freeze | 不做 fallback weight / hard shield |
| 8 s conventional contract | Freeze | 不缩 horizon |
| v27/v28 execution integrity | Solved / Freeze | 不碰 |
| RVR max-prefix policy | Negative policy | 仅保留 controlled alternative |
| SOV one-step strict gate | Positive representation signal | 不 promotion implementation |
| BHOV H0×V1 product order | Failed promotion | 不调 tolerance，不 exact200 |
| successor restoration bit | Too sparse | 归档 diagnostic |
| **multi-step option persistence** | Unlearned / P0 | V32 主分支 |
| **recovery-mode consistency** | Unlearned / P0 diagnostic | V32 第二分支 |
| proposal support | long-term ceiling | 暂不扩 bank |
| accepted-path kinematics | secondary unresolved | recovery 收敛后单独做 |

---

# 7. 模型现在学会了什么，还没学会什么

## 已学会

模型主干已经比较可靠地表示：

- natural behavioral alternatives；
- same-root counterfactual response；
- protected-priority non-coercive feasibility；
- BCOT structured false-safe/coercion evidence；
- certified feasible-set 内的 robustness ordering。

## 没学会 / 没有 representation 的内容

当前没有一个模型层或 analytic certificate 直接表示：

`(s_t, a_t^executed) -> option support 是否在 t+1, t+2, ... 持续存在`

也没有表示：

`recovery mode entered -> 什么时候应该持续，什么时候 feasibility 已恢复可以安全退出`

这两个缺失共同解释：

- SOV 太保守；
- BHOV 太容易跟 RVR；
- BHOV 在 `3356...` 中出现 COWP/RVR 都安全、hybrid 却失败的现象。

现在仍然不应该训练 neural viability head。先验证 analytic temporal target；否则只是在拟合一个还没被证明有意义的 supervision。

---

# 8. ALGORITHM_CHANGELOG 约束：V32 明确禁止

本轮不允许：

- RCOT/BCOT budget、threshold、priority hard gate retune；
- CTU-style `certificate -> planner argmin`；
- fallback outcome weight / hard outcome shield；
- prefix weight tuning；
- 缩短 8 s conventional screen；
- 把 BHOV 加 epsilon/tolerance 继续“救”；
- 继续做 strict Pareto tolerance；
- route/Frenet/roadgraph redesign；
- 新 proposal primitive / dataset rebuild / retraining；
- 同时改 accepted-path kinematics；
- 使用 logged future 做 successor probe；
- 把 holdout64/exact200 当 final paper evidence；
- 把 generic multi-horizon MPC、backup plan、hysteresis/commitment 本身包装成 novelty。

---

# 9. V16.8.32 主分支：Tri-Horizon Option Persistence (THOP)

方法名：`cowp_trihorizon_option_persistence`

## 9.1 Controlled pair 不变

仅在：

`full conventional set == empty && valid candidate exists`

比较：

- base = 原 COWP least-coercive-valid fallback；
- alternative = 原 V29 RVR max-prefix candidate。

其它所有 COWP path bit-for-bit 语义不变。

## 9.2 三个 horizon

### H0：current causal survival

使用现有 8 s causal collision-safe prefix：

`P0_alt >= P0_base`

### V1：one emitted action 后的 successor option set

与 V30/31 相同：

- ego 使用真正 jerk/yaw-rate-limited emitted target；
- other agents 只用 current state + CV；
- successor 重新生成同一 online physical bank；
- signature：`(conventional existence, macro diversity, conventional count, max prefix)`。

要求：

`V1_alt >=lex V1_base`

### V2：第二个 causal emitted action 后的 option set

仅当 BHOV H0×V1 pre-gate 已通过时，才支付第二层计算：

1. 从 V1 state 开始；
2. 对 base / RVR 各自沿同一个原 candidate 的第二 waypoint；
3. 再经过一次相同 jerk/yaw-rate-limited action projection，并把第一步 acceleration 作为 jerk state；
4. other agents 再做一步 CV；
5. 在 V2 state 重新生成相同 physical proposal bank，计算同样 signature。

要求：

`V2_alt >=lex V2_base`

最终接受 iff：

`P0_alt >= P0_base AND V1_alt >= V1_base AND V2_alt >= V2_base`

且至少一项 strict improvement。

没有新 weight、阈值或 oracle。

### 为什么它是当前最小正确 probe

V31 的问题正是 V1 non-regression 不保证 delayed state 不 collapse。THOP 不试图一次做复杂 learned reachability，而是只增加最小的第二个实际控制 interval，直接验证 temporal reversal 是否是 BHOV false positive 的根因。

THOP 本身不宣称 formal recursive-feasibility guarantee；它是为论文 physical option-set axis 验证 target semantics 的最小 mechanism probe。

---

# 10. V16.8.32 诊断分支：SOV-triggered Recovery Commitment (SRC)

方法名：`cowp_sov_recovery_commitment`

它专门测试 `3356...` 暴露的 hybrid-mode 问题。

## Entry

当零 conventional 且 base/RVR action 不同：

仅当 V30 的高精度条件：

`V1_RVR > V1_COWP`

才进入 recovery mode。

## Continue

一旦进入，只要仍为：

`zero conventional && valid exists`

就持续执行 unchanged RVR recovery alternative，不再每 0.1s 重新要求 SOV strict improvement。

## Exit

不设 fixed dwell time，也不设手工 hysteresis threshold。

只在状态语义上退出：

- certificate / full conventional option 恢复 → 立即回原 COWP；
- no-valid emergency → 清除 commitment。

因此它回答的单一问题是：

> SOV 的低 rescue recall 是否主要不是 entry classifier 错，而是 entry 后没有形成足够持续的 recovery mode？

该 branch 只是 attribution diagnostic。现有 autonomous-driving hysteresis / recovery-mode 文献很多，不能把“commitment/hysteresis”单独当论文 novelty。

新增 diagnostics：commitment active / enter / continue / clear step rate。

---

# 11. V16.8.32 新 development protocol

## Stage A：equivalence16

先证明普通 COWP 不变。

## Stage B：counterfactual48

同时跑 THOP + SRC，只用于旧 10 rescue / 9 induced 的机制 discrimination。

自动 preregistered gate（每个新 branch 单独判断）：

- old RVR rescue retained >= 5/10；
- old RVR induced avoided >= 7/9；
- 相对 COWP net collision removed >=3；
- Kinematics net regression <=1 scene；
- mean paired EP delta >= -0.05；
- 必须有 nonzero intervention。

失败就归档，不调 threshold/weight。

## Stage C：fresh37

V31 的 holdout64 已经被本轮看过，V32 不能再把它当 unbiased gate。

因此从 exact200 中排除：

- V30 equivalence16；
- V30 counterfactual48；
- V30 balanced96；
- V31 holdout64。

四组 union 后正好剩 **37** 个此前没有被这些 panel 用于机制选择的 ID。V32 在不知道新方法 outcome 的情况下用固定 salt 排序并冻结全部 37：

`logical_sha256 = ecce3321d8f4cd57bbd3189b3673784bec8fde185b882e9c11c38430265a1481`

它仍不是 paper holdout，因为 exact200 在历史上整体反复使用；但比复用 holdout64 更能约束 V32 对已知 counterexample 的过拟合。

fresh37 gate：

- no net collision harm；
- no net CR harm；
- Offroad/Kinematics 净回退各最多 1 scene；
- mean EP delta >= -0.03；
- nonzero intervention。

## Stage D：exact200

只有 fresh37 通过的 method 才跑 exact200，而且只跑新 method；COWP/RVR reference 不重复计算。

exact200 仍然只是 development confirmation。

最终 freeze 后必须重新冻结从未参与 V25-V32 mechanism selection 的 final evaluation scenes，并做 multi-seed + paired CI。

---

# 12. CCF-A 主线应该如何收紧

不能把 V32 的 novelty 写成：

- two-step lookahead；
- recursive feasibility；
- backup trajectory；
- hysteresis；
- commitment mode。

这些方向已有成熟工作。

当前真正有研究价值的统一主线应继续朝：

## **Orthogonal Option-Set Feasibility**

### Social axis

`natural roots -> same-root counterfactual transport -> protected-priority option preservation -> BCOT certificate`

问题：ego 是否靠压缩 **他车的 natural low-burden option set** 获得 collision-free 结果？

### Physical-temporal axis

`actual emitted action -> causal temporal successors -> ego executable option-set persistence`

问题：uncertified recovery 是否靠一个局部看似安全的动作压缩 **ego 自己未来的 executable option set**？

两个 axis 的对象不同，但核心抽象一致：

> **安全不是一个单一 risk scalar，而是不能通过不可接受地压缩关键参与者未来可行选择空间来获得。**

如果 THOP/SRC 后续证据支持 temporal physical half，这才是值得升级的 paper-level mechanism story；V32 两个具体 probe 都只是验证该抽象的必要步骤。

---

# 13. 下一步执行

```bash
cd COWP_v16_8_32_TEMPORAL_OPTION_PERSISTENCE

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal_v16_8_24_compact_full_5k
export BASE_RUN=/你的旧COWP目录/outputs/v16_8_24_compact5k_all
export BASE_CKPT="$BASE_RUN/cowp_all_best.pt"

bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh sanity
bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh make_ids

# 只有 index 缺失时再跑
bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh build_tfindex

bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh base_equivalence16_parallel2

# 第一阶段只跑 48；两个根因分支都跑
bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh counterfactual48_parallel2
bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh analyze_counterfactual48
```

到这里先停止。

只有 `counterfactual48_v32_temporal_option_analysis.json` 中某个 method 的 `preregistered_gate.pass=true`，才进入 fresh37：

```bash
PROMOTED_METHODS=cowp_trihorizon_option_persistence \
bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh fresh37_parallel2
bash NEXT_RUN_COMMANDS_V16_8_32_TEMPORAL_OPTION_PERSISTENCE_CN.sh analyze_fresh37
```

fresh37 通过后才 exact200。

---

# 14. 本地 validation

- V16.8.32 focused semantic/integrity suite：**34/34 passed**；
- V16.8.32 launcher `sanity`：**34/34 passed**，4 个 manifest hash 全部通过；
- `py_compile` / `bash -n`：pass；
- full-suite 在本环境单命令窗口内未完成；`-x` 运行到第一处失败为 **124 passed / 5 skipped / 1 historical failure**，第一处仍是仓库历史缺失 `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`，与 V32 功能无关。完整 run 在超时前可见 8 个 failure，数量与历史 baseline 一致，但因未完整结束，不把它报告成新的 full-suite verdict。
