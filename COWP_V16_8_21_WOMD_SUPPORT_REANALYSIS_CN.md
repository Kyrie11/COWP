# COWP v16.8.21：WOMD 支撑能力、strict 失败根因与 evidence-aligned 修复

## 0. 最终判断

本轮 v16.8.20 smoke **通过**，strict probe **只有 proposal causal screen 未通过**。重新追踪 `paired_proposal_probe.json -> 45_diagnose_proposal_ceiling.py -> 58_screen_v16_8_9_causal_audit_probe.py -> v16_8_18_strict_verdict.json` 后，可以确定：

1. **当前 strict FAIL 不是 WOMD 无法支撑 COWP，也不是 v16.8.20 natural/model supervision 再次失败。**
2. 当前真正 blocker 是 v16.8.20 promotion policy 把 `protected relation 在随机 WOMD scene 中出现的比例` 当成了 proposal/data quality 硬指标，并且又把 `priority NCF / all representative scenes` 与 PBTR 同时硬 gate；这在数学上重复计入了 eligibility prevalence。
3. WOMD 确实存在一个真实、不可通过数据构建器消除的边界：**一个 9 s WOMD window 只记录 factual trajectory；它没有同一真实场景在多个 ego intervention 下的人类反事实响应 ground truth。** `sdc_paths` 也只是 SDC 可行未来 route，不是 surrounding-agent counterfactual response label。
4. 这个 WOMD 边界**不要求推翻 COWP 主模型**。现有论文/实现最合理、也最可发表的解释应该是：学习的是 **root-indexed counterfactual feasibility / viability transport operator**，而不是声称从 WOMD 单独识别了真实人类 causal response distribution。最终 causal burden claim 继续由 independent reactive-agent protocol + held-out human-audited stress set 支撑。
5. v16.8.21 因此是 **promotion/evidence contract 修复**，不是 label semantic rewrite。v16.8.20 的 Scenario->label semantic fingerprint 保持不变：

```
c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51
```

因此本轮已经完成的 1200-scene strict 证据可以在严格 provenance 检查后直接 policy re-audit；不需要重新生成相同 NPZ。之后仍必须 fresh train-pilot，再决定 full rebuild。

---

## 1. 本轮 v16.8.20 strict 到底哪里失败

`v16_8_18_strict_verdict.json`：

- sparse label build complete: PASS
- training supervision: PASS
- model support: PASS
- auditability coverage: PASS
- certificate-complete scene coverage: PASS
- natural root support: PASS
- protected PRIO root coverage: PASS
- **proposal_causal_screen_pass: FAIL**

`base_screen_verdict.json` 中只有两个 proposal checks 为 false：

| 指标 | 观测 | v16.8.20 strict threshold | 结果 |
|---|---:|---:|---|
| any valid | 1.0000 | >=0.99 | PASS |
| priority eligible / all random scenes | 0.79875 | >=0.90 | FAIL |
| priority NCF / all random scenes | 0.49750 | >=0.50 | FAIL |
| PBTR floor / eligible scenes | 0.3771518 | <=0.45 | PASS |
| global any NCF | 0.315 | >=0.30 | PASS |
| global false-safe floor | 0.63875 | <=0.65 | PASS |
| hard-scene NCF recovery | 0.2525 | >=0.20 | PASS |

代表性 random slice 为 800 scenes：

- eligible = 639 scenes
- priority NCF = 398 scenes
- eligible but no NCF = 241 scenes

因此：

```
P(NCF | protected-eligible)
= 398 / 639
= 0.622848200312989

PBTR_floor
= 241 / 639
= 0.377151799687011

P(NCF | eligible) + PBTR_floor = 1.0
```

而旧 gate 使用：

```
priority_ncf_scene_rate
= 398 / 800
= (639 / 800) * (398 / 639)
= P(eligible) * P(NCF | eligible)
= 0.4975
```

因此 v16.8.20 同时硬 gate：

- `P(eligible) >= 0.90`
- `P(NCF over all scenes) >= 0.50`
- `PBTR <= 0.45`

实质是在同一 protected certificate 上重复乘入 `P(eligible)`。

### 为什么这是 promotion policy bug，而不是 proposal bug

proposal generator **不能也不应该**把没有 protected relation 的 WOMD scene 变成 protected scene。eligibility 来自当前 causal critical relation / priority semantics，是被采样 scene mixture 的属性。

论文定义 PBTR 时，分母本身就是“collision-free 且存在 protected relation 的场景”；因此 proposal/certificate 质量应在 eligible subset 上判断，而不是要求随机 WOMD slice 中 90% 场景必须拥有 protected relation。

此外，v16.8.20 causal selector 相比旧 future-assisted cache：

- protected eligibility prevalence 从约 0.8975 降到 0.79875；
- 但 PBTR floor 从约 0.5139 改善到 0.37715；
- conventional safe coverage 也改善。

这更符合“去掉 future oracle 后 protected scene 定义更干净”，而不是“WOMD 突然丢失模型支持”。

---

## 2. 当前 strict 数据实际上已经有很强的模型监督支撑

### 2.1 natural / auditability

1200 strict scenes：

- selected critical agents: 6531
- mechanism-auditable: 6245
- unauditable: 286 = 4.3791%
- certificate-complete scenes: 1029 / 1200 = 85.75%
- protected auditable agents: 4924
- protected PRIO root coverage: 99.4110%
- rootless auditable critical agents: 0
- `<2 low-burden roots`: 0

286 个 unauditable critical 中：

- 160：future support 足够，但没有足够 substantial route geometry
- 126：future support 本身不足

这种缺失应当被 `mechanism_valid` / evidence mask 表示为 **unknown / not auditable**，而不是强行伪造 NCF/false-safe 标签。

### 2.2 candidate / pair / response / transport 监督都不是退化分布

model support audit 已 PASS，典型 class support：

- candidate NCF: 7749 positive / 49307 negative
- priority candidate NCF: 13388 positive / 14999 negative
- priority candidate false-safe: 14999 positive / 13388 negative
- pair relevance: 152630 positive / 193806 negative
- relevant-pair witness: 105605 positive / 47025 negative
- response safe: 2,366,187 positive / 2,517,973 negative
- response low burden: 2,284,723 positive / 2,599,437 negative
- root recovery: 674455 positive / 1,009,251 negative

source support：

- natural OBS 17719
- natural NEU 32029
- natural PRIO 47751
- response PRED 2,194,818
- response OPT 2,655,281
- response EMG 34,061

因此从“模型有没有可训练的正负类/多 source/root transport supervision”角度，本轮不是数据匮乏。

---

## 3. proposal bank 是否仍有问题

有优化空间，但不是当前 strict blocker。

全 1200 scenes source ablation：

- all-bank any NCF = 0.29417
- all-bank protected eligible = 0.825
- all-bank protected NCF = 0.48417
- all-bank PBTR = 0.41313
- mean valid candidates = 55.26 / 64

JR-NCF 的增量：

- global NCF +0.001667
- protected NCF +0.001667
- PBTR -0.00202
- mean valid candidates +0.8333

PSY 增加候选，但本轮没有 scene-level NCF/PBTR 增量。

因此继续堆更多类似 timing/yield primitive 会出现明显边际收益递减。当前更值得优化的是训练后的 learned selector/certificate，而不是为了过一个错误 prevalence gate 继续扩 proposal family。

### conflict region cap

v16.8.20 已修 raw-order early return：

- ego reference used rate = 100%
- candidate-pool saturation rate = 0%
- selected cap saturation = 96.58%
- raw conflict regions mean ≈354
- deduplicated mean ≈353
- selected mean ≈63.13

selected C=64 经常饱和本身不是 bug，只要 ranking 前的 candidate pool 没饱和并且 selection 是 SDC-centric。当前这项通过。

---

## 4. `burden_only_affected = 0` 是否说明 WOMD 不支持 coercion？

不是当前主线 defect。

strict 中：

- affected roots = 1,719,992
- unsafe roots = 1,719,992
- budget-crossed roots = 9,389
- burden-only affected roots = 0

代码主定义：一个 factual/natural low-burden root 先被 ego candidate 激活，当它进入 collision / near-miss / TTC / RSS safety envelope violation 时，再对它执行 same-root recovery search，并计算恢复所需 minimum burden、OPR 和 tail burden。

这和论文的 `c_ikm` / conflicted-root transport 公式本身一致：核心问题不是“自然轨迹本身突然有了急刹负担”，而是自然轨迹被 candidate 阻断后，为了保持安全需要 transport 到 braking/yielding/recovery response；**负担应在 recovery response 上测量**。

因此本轮 `burden_only=0` 更说明当前 mechanism 是“root viability loss -> recovery burden”，而不是额外的 continuous soft-risk channel。后者可以作为未来 extension，但不应为了数据看起来丰富而硬造。

建议论文措辞把主机制固定为：

> candidate-induced natural-root viability loss + same-root minimum-burden recovery transport

而不是暗示 WOMD 给出了真实人类在每个 ego intervention 下的 counterfactual response label。

---

## 5. WOMD 到底能提供什么，不能提供什么

### 5.1 WOMD 能提供

官方 Motion Dataset：

- 103,354 个 20 s segments，并切为 9 s windows；
- train/validation window 含 1 s history、current sample、8 s factual future（10+1+80=91 samples）；
- vector map / dynamic map states / object tracks；
- `objects_of_interest` 是检测到 interactive behavior 的对象组；
- `tracks_to_predict` 是 prediction benchmark target/suggestion，不是在线 planner 可直接使用的 causal input；
- WOMD 1.3.1 新增 `sdc_paths`，描述 SDC 的 valid future routes，可支持 Waymax route/wrong-way/progress metrics。

这足以构建：

1. factual observational roots；
2. current/history/map-based critical relation；
3. map/topology/priority-constrained constructive NEU/PRIO roots；
4. ego candidate bank；
5. deterministic candidate-vs-root safety envelope；
6. same-root recovery primitive search；
7. OPR / root minimum burden / tail burden / witness pseudo-label；
8. Waymax physical closed-loop collision/offroad/route outcomes。

### 5.2 WOMD 不能单独提供

WOMD 每个 window 只有一个真实发生的 future，因此没有：

```
同一真实 initial state + 多个不同 ego intervention
-> 同一个 human driver 的真实 counterfactual response distribution
```

`sdc_paths` 也不是这个量，它只描述 SDC 可行 future routes。

Waymax 官方本身是 WOMD-based simulator，并提供 log-playback / IDM 等 agents。log/expert actor 的实现直接利用下一时刻 logged state 推导动作，因此它不是“在任意 ego intervention 下的人类反事实响应 ground truth”。

所以 **不能把 WOMD + logged replay 里的 constructed neutral/priority roots 叙述成 identified human causal effect**。

官方参考：

- https://waymo.com/open/data/motion/
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto
- https://github.com/waymo-research/waymax
- https://github.com/waymo-research/waymax/blob/main/waymax/agents/expert.py

---

## 6. 为了保持强 novelty，COWP 的机制主线应该怎么表述

我不建议为了 WOMD 的 limitation 去把 COWP 改成普通 social-cost planner；那会真正损害 novelty。

更稳的主线是：

### Evidence-Gated Root-wise Counterfactual Viability Transport

保留 COWP 名称，机制拆为：

1. **Natural Root Basis**
   - OBS：factual-distribution anchor
   - NEU：pressure-removed constructive intervention proxy
   - PRIO：rule/topology-constrained priority-preserving root

2. **Candidate-Induced Root Activation**
   - 不是预测“这个人真的会怎么反应”；
   - 而是判断 ego candidate 是否使一个原本低负担 root 进入 candidate-conditioned safety envelope violation。

3. **Same-Root Recovery Operator**
   - 在保持 root semantic identity 的约束下搜索 recovery bank；
   - 得到 `q_ikm`、minimum safe burden、root recovery mass。

4. **Protected Option Transport**
   - OPR 计算保留下来的自然概率质量；
   - tail burden 测量被阻断 roots 的 minimum recovery burden excess。

5. **Hard Protected-Priority Certificate + Global Diagnostic**
   - protected priority 是 primary hard certificate；
   - all-critical 是 global burden-transfer stress diagnostic。

6. **Evidence Gate**
   - factual observation、map topology、constructive dynamics labels明确区分；
   - `mechanism_valid` / root target confidence 控制哪些 pseudo-target 可用于 loss/certificate；
   - human causal claim 不从 offline pseudo-label 推出。

这套组合的贡献仍然是：

- false-safe failure mode；
- feasibility-level（不是 soft courtesy cost）certificate；
- same-root counterfactual option transport；
- priority-aware hard feasibility；
- evidence-gated causal-claim protocol。

这比强行声称 WOMD 给出真实 counterfactual human response 更严谨，也更容易经受高水平审稿人的 identifiability 质疑。任何会议等级/CCF-A 接收都无法保证，但从研究论证结构上，这种改法不会把工作退化为普通 socially-aware cost weighting。

---

## 7. v16.8.21 数据集“必须具备”的性质

### Hard promotion properties

1. causal input contract 100%：critical selector 不用 logged future / TTP / OOI oracle。
2. ego-relevant conflict ranking 100%。
3. conflict candidate pool 不系统截断。
4. any valid proposal >= 0.99。
5. **protected eligible evidence count >= 256 / strict random-800**。
6. **protected NCF evidence count >= 128 / strict random-800**。
7. **PBTR floor <= 0.45 on protected-eligible scenes**，并且 strict random slice 的 **95% Wilson upper bound <= 0.45**。
8. global any NCF >= 0.30，给 all-critical auxiliary BCOT 头保留正例支撑。
9. global false-safe floor <= 0.65。
10. hard-scene recovery >= 0.20。
11. natural auditable roots：rootless=0，至少 2 low-burden roots。
12. protected PRIO-root coverage >= 0.95/0.98（现有 gate 根据 stage）。
13. candidate/pair/response/transport labels non-degenerate。
14. certificate-complete / auditability coverage 通过已有 Wilson + stratum gates。

### Advisory / reporting properties，不再作为 random-scene hard gate

- protected eligibility scene rate；
- protected NCF / all-scenes rate；
- selected conflict C=64 saturation；
- JR/PSY marginal source gain；
- burden-only root prevalence。

尤其不能通过“人为扩大 protected relation”来追 90% eligible prevalence。

---

## 8. v16.8.21 代码修改

### `cowp/scripts/58_screen_v16_8_9_causal_audit_probe.py`

删除 strict hard checks：

```text
min_priority_eligible = 0.90
min_priority_ncf = 0.50
```

替换为：

```text
min_priority_eligible_count = 256
min_priority_ncf_count = 128
max_pbtr_floor = 0.45
max_pbtr_wilson_high = 0.45
```

同时保留：

```text
min_global_any_ncf = 0.30
max_global_false_safe_floor = 0.65
min_hard_recovery = 0.20
```

新增一致性检查：

```text
protected_ncf_given_eligible + PBTR_floor ~= 1
```

这可以直接捕获未来如果 `priority_eligible / priority_ncf / PBTR` 三套标签语义发生漂移。

### `cowp/scripts/73_reaudit_v16_8_20_strict_policy.py`

新增 provenance-verified strict policy re-audit：

- 要求 source strict full code fingerprint 精确等于 reviewed v16.8.20：
  `e9fcfab92ed8a24cac3215e6ca037897231ce59e74fd186ddd8200c6338b8172`
- 要求 current label semantic fingerprint 精确等于 v16.8.20：
  `c7f8a33f...0af51`
- 只重跑 policy screen；
- 复用同一 strict evidence bundle 中已经 PASS 的 supervision/model/natural/sparse audits；
- 输出当前代码 fingerprint 的兼容 verdict，供 train-pilot/full-core wrapper 使用。

### wrapper

新增：

- `NEXT_EXECUTION_V16_8_21_CN.sh`
- `NEXT_EXECUTION_V16_8_21_COMMANDS_CN.txt`

推荐路径：

```text
reaudit-strict -> fresh train-pilot -> full-core -> outcomes
```

---

## 9. 对你这次上传的 strict 直接重算结果

在**不重新构造标签**的情况下，用 v16.8.21 screen 对本轮 JSON evidence 重算：

```text
screen_pass = true
protected_eligible_support_count = true   (639 >= 256)
protected_ncf_support_count = true        (398 >= 128)
protected_partition_consistent = true
PBTR floor = 0.3771517997 <= 0.45
PBTR 95% Wilson upper = 0.4153605045 <= 0.45
NCF|eligible 95% Wilson lower = 0.5846394955
Global NCF = 0.315 >= 0.30
Global false-safe floor = 0.63875 <= 0.65
Hard recovery = 0.2525 >= 0.20
```

所有 causal selector / conflict ranking / audit integrity / source union checks 同样 PASS。

完整 policy re-audit 也 PASS：

```text
recommend_full_rebuild = true
failed_checks = []
```

但注意：这里的 `recommend_full_rebuild=true` 只表示 **validation strict gate 已授权进入 train-pilot**。现有 pipeline 仍要求 fresh train-pilot PASS 后，`full-core` 才会真正执行。

---

## 10. 下一阶段研究建议

### 10.1 现在不要再重构 natural labels

当前 natural support 已非常完整。继续为了提高 random-scene protected prevalence 调自然根或 critical selector，很容易重新引入 oracle / label inflation。

### 10.2 train-pilot 才是下一条真正的工程证据

严格数据支撑通过后，必须检查模型能否学习：

- protected candidate NCF / false-safe；
- pair relevance / witness；
- root recovery / OPR；
- BCOT protected/global heads；
- selector 是否在保留 progress 的情况下降低 PBTR。

如果 train-pilot 失败，届时问题就从“数据构建支持”转移到 model optimization/calibration，而不是继续重建 WOMD pseudo-label。

### 10.3 最终因果 burden claim

最终论文应该分三层证据：

1. **WOMD offline constructed-feasibility evidence**：机制监督、pseudo-label correctness、root transport。
2. **Waymax logged replay**：physical closed-loop CR/OR/wrong-way/progress；model mechanism quantity 只叫 diagnostic/proxy。
3. **Reactive + human stress protocol**：真正用于支持“在 ego intervention 下其他道路使用者被迫承担 burden”的 causal/behavioral claim。

不要把三层混成同一个 ground truth。

