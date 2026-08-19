# COWP v16.8.22：WOMD 支撑边界、train-pilot blocker 与六层机制一致性修复

## 结论摘要

本轮 v16.8.21 的 **validation strict re-audit 已经 PASS**。v16.8.21 的 evidence-aligned strict promotion policy 确实修正了 v16.8.20 的 denominator/prevalence 问题；当前仍提示“不建议 full rebuild”的直接原因来自 **train-pilot 仍调用 v16.8.18 的旧 verdict 逻辑**。

上传的 1200-scene train-pilot 中，唯一失败项为：

- `candidate_any_ncf_support = false`
- all-critical `any_ncf = 346/1200 = 0.288333 < 0.30`

但是同一 pilot 的 protected-primary 数据为：

- protected eligible scenes = `994/1200 = 0.82833`
- protected NCF scenes = `561/1200 = 0.46750`
- `P(priority NCF | protected eligible) = 561/994 = 0.564386`
- PBTR floor = `433/994 = 0.435614 < 0.45`

因此当前 blocker 不是“train split 没有 NCF 支撑”，而是 **旧 train gate 仍把 all-critical scene prevalence 当作主训练就绪条件**。

更深一层，v16.8.21 模型训练还存在 Layer-5 语义不完全一致：论文/label 已经是 protected-priority primary、all-critical global auxiliary，但 planner ranking/classification 仍主要使用 global labels；primary BCOT 聚合还允许 non-protected pair 通过 5% floor 和 global max deficit 进入主 certificate。这会让已经被降为诊断量的 all-critical NCF 再次污染主模型目标。v16.8.22 一并修复这一点。

WOMD 的固有限制是真实的：一个 Scenario 只提供一条 factual logged future，`objects_of_interest` / `tracks_to_predict` 也不是不同 ego interventions 下的真人 response ground truth；WOMD 1.3.1 的 `sdc_paths` 是 SDC 可行未来 route，而不是周车的人类反事实响应。因此 COWP 不应声称从 WOMD 单独识别真实 human causal response distribution。这个边界 **不会推翻六层机制**，前提是把 NEU/PRIO/recovery 统一定义成 constructive counterfactual feasibility evidence，并把最终 causal/human claim 放到 reactive-agent + human-audited evidence gate。

---

## 1. v16.8.21 strict 是否修正了 promotion-policy bug？

是。

对上传的 strict re-audit 证据按当前 v16.8.22 full fingerprint 重放后仍然 PASS，且 Scenario->label semantic fingerprint 未改变：

- current full fingerprint: `ab8c1c609267a5ebd0f975724546d87a7ccfe18a07635291efb2cef837b9707c`
- label semantic fingerprint: `c7f8a33f5e9fef04ac009d41806173369ddbfef6ac0b7e7c4ac0ca1edfc0af51`

strict representative random 800：

- any-valid = `1.000`
- global any-NCF = `252/800 = 0.315`
- global false-safe floor = `511/800 = 0.63875`
- protected eligible = `639/800 = 0.79875`
- protected NCF = `398/800`
- protected `NCF|eligible = 398/639 = 0.622848`
- PBTR = `241/639 = 0.377152`
- PBTR 95% Wilson upper = `0.415361 < 0.45`
- hard recovery = `101/400 = 0.2525`

所以 v16.8.21 strict 已经不再要求“随机 WOMD 的 protected prevalence >= 90%”，而是用 protected absolute support + conditional PBTR/NCF + global auxiliary support 来 promotion。这部分无需再削弱。

---

## 2. train-pilot 为什么还 FAIL？

### 2.1 直接代码原因

`NEXT_EXECUTION_V16_8_21_CN.sh train-pilot` 仍然委托给旧 `NEXT_EXECUTION_V16_8_18_CN.sh` / `NEXT_TRAIN_PILOT_V16_8_18_CN.sh`。

旧 verdict 中硬编码：

```python
candidate_any_ncf_support = rates["any_ncf"] >= 0.30
```

这仍然使用 **all-critical NCF / 全部 pilot scenes**。

而本轮 pilot 本身是刻意构造的：

- 400 hard scenes
- 800 representative random scenes
- hard/random overlap = 0

历史 hard 定义又是 `any conventional safe && no all-critical NCF`。所以 pilot 本身被人为富集了 global-NCF 困难场景，再要求这个混合分布复现 `global any-NCF >= 0.30`，不是训练 identifiability 条件，而是 sampling scheme 与 gate 的自相矛盾。

### 2.2 当前 pilot 实际拥有充足监督

candidate：

- global NCF: total `57,741`, positive `7,219`, negative `50,522`
- protected candidate NCF: total `28,956`, positive `12,413`, negative `16,543`
- protected false-safe: positive `16,543`

pair：

- relevant pairs `154,015 / 346,371`
- witness positives `107,437`
- pair-NCF positives `46,578`

root transport：

- affected roots `1,721,620`
- same-root recovery positive `685,468`
- same-root recovery negative `1,036,152`

natural/model coverage：

- selected critical agents `6,502`
- auditable `6,257`
- unauditable rate约 `3.77% < 5%`
- rootless auditable critical = `0`
- `<2` low-burden roots = `0`
- protected PRIO-root coverage = `0.994498`
- certificate-complete scenes = `1,043/1,200 = 0.86917`
- hard/random certificate gap = `0.02375`
- model-support audit = PASS
- training-supervision audit = PASS
- causal integrity read/silent/irrelevant/transport mismatch = 0

所以“缺类”不是当前问题。

---

## 3. 现有 audit 仍缺什么？

旧 audit 证明了 **marginal class support**，但没有证明六层 novelty 最关键的 **within-scene / same-root intervention contrast**。

v16.8.22 新增 `74_audit_mechanism_contrast.py`，硬检查：

1. protected 同 scene 至少同时存在 NCF candidate 与 false-safe candidate，可构造 planner ranking pair；
2. 同一 `(scene, critical-agent, natural-root)` 在不同 ego candidates 下存在 `affected ↔ unaffected` switch —— Layer 2；
3. 对同一 affected root，不同 ego candidates 下存在 recovery score 高/低 switch —— Layer 3；
4. 同一 protected agent 在不同 ego candidates 下 OPR 跨越 alpha —— Layer 4；
5. 足够多 0<OPR<1 的 partial option mass，避免 option-mass 退化成二值碰撞标签。

默认 1200-scene pilot gate：

- rankable scenes >= 128
- protected rank pairs >= 2,048
- viability-switch scenes >= 128 / roots >= 1,024
- recovery-switch scenes >= 32 / roots >= 128
- OPR-switch scenes >= 128
- partial OPR pairs >= 512

这组性质才真正回答“shared same-root response surface / transport 是否能从数据中学到”，而不是用全局 prevalence 替代。

上传的 compact train-pilot zip 不含 NPZ，因此这些 switch 不能从汇总 JSON 逆推出。必须在保留 NPZ 的机器上运行 v16.8.22 re-audit；如果这一 audit FAIL，那才是需要回到 label/proposal/机制构造层修复的真实证据。

---

## 4. WOMD 的边界与六层机制是否矛盾？

### WOMD 可以可靠提供

- history/current factual state；
- one factual future；
- HD map / dynamic map state；
- SDC identity；
- v1.3.1 `sdc_paths` 的 SDC future-route support；
- factual OBS anchor；
- 用 map/current state 构造 NEU/PRIO feasibility basis 的输入；
- ego candidate 与 root 的几何/safety interaction；
- analytic/primitive same-root recovery search；
- OPR / burden / witness pseudo-target 的确定性构造输入。

### WOMD 不能单独识别

- 同一 initial condition 下真人驾驶员面对多个不同 ego interventions 的真实 response distribution；
- “如果 ego 当时不切入，后车这个真人实际会怎样”的 ground-truth causal trajectory；
- 人类 normative burden 的唯一真值。

因此不能把 NEU/PRIO/recovery pseudo-target 描述成“WOMD 提供的 counterfactual human GT”。

### 六层 novelty 的可成立版本

1. **Natural Root Basis**：OBS factual-distribution anchor + NEU pressure-removed constructive proxy + PRIO topology/rule-preserving root；
2. **Candidate-induced root viability loss**：问 candidate 是否让原本 admissible/natural root 失去安全可行性；不是声称真人必然这样响应；
3. **Same-root recovery transport**：在语义 root identity 下搜索最低 burden safe recovery；
4. **Option-mass transport**：对 root mass 的 retained feasibility 做 OPR/transport；
5. **Protected-priority hard certificate**：protected set 是 primary hard veto，all-critical 仅 global burden-transfer diagnostic；
6. **Evidence gate**：factual / constructive / reactive-sim / human-audited evidence 分层，不跨层声称因果真值。

这套组合保留了 COWP 的主要技术辨识度：它不是把 social burden 加成 soft cost，而是建立 root-wise intervention feasibility + same-root recovery + option-mass + protected hard certificate。是否达到 CCF-A 录用标准无法保证，但这比“用 WOMD 单 factual future 冒充多干预 human causal GT”更审稿稳健。

---

## 5. v16.8.22 模型修复

### 5.1 planner primary target 与 Layer 5 对齐

新增 `primary_candidate_targets()`：

primary universe =

```text
certificate-valid AND conventional-safe AND priority-eligible
```

primary targets：

- `priority_noncoercive_feasible`
- `priority_false_safe`

global `noncoercive_feasible/false_safe` 只保留辅助 ranking。

train config：

```yaml
ranking: 0.75
global_ranking_aux: 0.10
```

### 5.2 BCOT primary/global 真正解耦

v16.8.21 的 primary BCOT 存在两个 leakage：

- non-protected pair 仍有 5% priority-weight floor；
- primary feature 的 max deficit 是 all-critical max。

v16.8.22：

- protected/unknown-supported pair 才进入 primary weight；
- ego-priority pair 对 primary risk 权重严格为 0；
- primary max deficit 仅 protected；
- global head 保留 all-critical mean/tail/severe/max；
- protected set 为空时 primary risk 严格为 0（vacuous protected-set semantics），global risk 仍可报告。

同时 `set_transport_priority_budget_mix: 0.65 -> 0.80`，让学习预算与论文 primary claim 对齐，但保留 20% global auxiliary。

### 5.3 modern training cache 必须显式含 priority labels

planner/planner_eval/all stage 的 dataset contract 增加 explicit priority candidate labels，防止新训练静默 fallback 到旧 global semantics。

---

## 6. v16.8.22 train-pilot promotion 修复

删除：

```text
all-critical any-NCF scene prevalence >= 0.30
```

替换为：

- global auxiliary NCF candidate positives >= 1,024
- global auxiliary NCF scenes >= 128
- protected NCF candidate positives >= 1,024
- protected false-safe candidate positives >= 1,024
- existing natural/model/audit/certificate gates all PASS
- six-layer mechanism-contrast audit PASS

这不是降低质量要求，而是从“混合场景中的出现率”改成“主/辅 head 的绝对监督量 + 机制可辨识 contrast”。

对于 fresh v16.8.22 pilot，hard-scene sampling 也新增 `--hard-definition protected`：

```text
priority-eligible AND no protected-priority NCF
```

而不是历史 all-critical hard 定义，使 stress sampling 和 Layer 5 primary claim 一致。

---

## 7. full-core 执行链的工程 blocker

v16.8.21 的 `full-core` 最后会调用：

```bash
PREPARE_COWP_V16_8_9_DATA_FAST_CN.sh
```

但该文件不在上传的代码包中。因此即使 train verdict PASS，旧 full-core 仍会在第一步失败。

v16.8.22 新增 `PREPARE_COWP_V16_8_22_DATA_CN.sh`，自包含执行：

1. 默认只从历史 promoted cache 读取 train/val scenario IDs；不复用任何旧 label tensor；
2. build location-aware Scenario index；
3. fresh WOMD Scenario -> COWP labels；
4. natural support audit；
5. fresh WOMD TFExample -> tensor cache；
6. `--require-waymax-ready --require-sdc-paths --require-all-labels-matched`；
7. full-cache self-contained protocol verify；
8. train/val supervision + model-support + causal + six-layer mechanism contrast；
9. 全部 PASS 才写 `full_core_support_verdict_v16_8_22.json: pass=true`；
10. Waymax candidate outcomes 后置到 `outcomes`。

transport supervision 已 inline 写入 fresh NPZ，所以 manifest 的 raw/transport 路径可以指向同一个 self-contained cache tree，不再依赖历史 overlay。

---

## 8. 当前能否直接 full rebuild？

**不能直接跳过 v16.8.22 train-pilot re-audit。**

能确认的：

- uploaded strict 在 v16.8.22 当前 fingerprint 下 re-audit PASS；
- uploaded v16.8.21 train pilot 的旧唯一 blocker 是错误的 global prevalence gate；
- aggregate natural/model/supervision support 足够；
- v16.8.22 相关 targeted/regression tests PASS。

尚未确认的：

- uploaded train-pilot NPZ 的 six-layer mechanism-contrast audit。

原因不是代码不可运行，而是上传 archive 里没有 NPZ。下一步必须在保留 `labels_train_v16_8_18/*.npz` 的机器上运行 `reaudit-train-pilot`。如果新 contrast PASS，就没有科学理由再重建同一 1200 labels，可以直接进入 full-core。若 contrast FAIL，则输出会明确是哪一层（viability / recovery / OPR / rankability）缺数据，再针对那一层改构造，而不是继续修改 population threshold。

---

## 9. 测试状态

v16.8.22 直接相关与 transport/planner 回归：`37 passed`。

完整 pytest：

- `237 passed`
- `5 skipped`
- `2 failed`

这 2 个失败都因为历史 wrapper 文件在原 v16.8.21 zip 中已经缺失：

- `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`
- `NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh`

在未修改的 v16.8.21 原包上单独复现相同 2 failures，因此不是 v16.8.22 算法回归。v16.8.22 不伪造历史文件来追求形式全绿。
