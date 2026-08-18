# COWP v16.8.20：WOMD / Waymax 数据支持诊断与修复方案

## 0. 结论

当前 `formal_v16_8_19_support_strict_probe` **不是 natural-basis、training supervision、model support 或 sparse-build 完整性失败**。复合 strict verdict 中唯一失败的是 `proposal_causal_screen_pass=false`。旧 `strict_pipeline_status.json` 的 `next_action="inspect strict natural/model-support diagnostics"` 是泛化/过时文案，不能代表真实失败项。

v16.8.19 strict 的直接失败数值是：

- random representative 800 scenes：`any_valid=1.0`；all-critical `any_ncf=0.35 < 0.40`；all-critical false-safe lower bound `0.60375 > 0.55`；PBTR `0.3763 <= 0.45`；hard-scene recovery `0.285 >= 0.20`。
- full 1200 source ablation：all-critical `any_ncf=0.32833`，false-safe floor `0.64083`；protected-priority diagnostic `any_priority_eligible=0.95833`、`any_priority_ncf=0.56583`、PBTR `0.40957`。
- natural support 已闭合：6663 selected critical，6417 mechanism-auditable；auditable critical 中 rootless=0，`<2` low-burden roots=0；protected PRIO root coverage `0.99223`。
- model/training support 已非退化：candidate NCF positive rate `0.1477`，pair relevance `0.41746`，pair NCF on relevant `0.3380`；training-supervision audit pass。
- 1200/1200 requested scenes 全部写出完整 NPZ，无 missing/corrupt/build error。

真正的问题是：**proposal / critical / conflict support 的因果和联合覆盖结构不够正确，而不是 natural response 数据不足。**

---

## 1. 论文真正需要数据证明什么

论文核心不是“collision-free planner”本身，而是区分：

1. ego candidate 在常规几何/动力学意义上安全；
2. 但它是否把碰撞避免责任转移给具有受保护优先权的其他道路使用者；
3. 对每个 natural root，是否仍存在 same-root、safe、low-burden response；
4. option-preservation mass 是否超过阈值；
5. hard-first planner 是否因此拒绝 false-safe candidate，同时保留足够的 non-coercive feasible candidate；
6. closed-loop / reactive evaluation 中，这种机制是否真实降低 other-agent hard braking、option collapse、tail burden，而不只是复述 logged future。

因此数据集必须同时覆盖 **candidate proposal support、critical pair support、natural-root support、root-indexed response support、witness/transport support、planner/Waymax outcome support**。只提升 natural root 数量但没有一个 ego candidate 能同时满足 protected constraints，模型仍然学不到/选不到正确行为。

---

## 2. WOMD 1.3.1 / Waymax 数据合同

官方资料核对结果：

- WOMD Motion v1.3.1（Oct 2025）新增 `sdc_paths`：表示从 SDC 起始位置按 roadgraph connectivity 得到的有效未来 route 示例。
- training/validation 的一个 9 s WOMD example 为 `10 history + 1 current + 80 future = 91` 个 10 Hz sample；tf.Example object tensor 以 `128` object 上限组织。
- `tracks_to_predict` 是 prediction challenge / training target annotation；`objects_of_interest` 是 interactive group annotation。它们不是部署时 planner 必然可获得的因果输入，因此不能作为 online critical selector 的默认打分特征。
- Waymax `DatasetConfig.include_sdc_paths=True` 才解析路径；`aggregate_timesteps=True` 是形成 `SimulatorState` 所需；`batch_by_scenario=True` 才按整个 scenario 而不是单 object trajectory 返回；`EnvironmentConfig.max_num_objects` 应与 DatasetConfig 一致，WOMD warmup/current 合计 `init_steps=11`。
- Waymax 官方 v1.3.1 preset 使用 `num_paths=45`, `num_points_per_path=800`。

对应 COWP 的正确用法是：

- **Scenario proto**：用于离线机制 pseudo-label（完整 vector map、track future、traffic context、scenario_id）。
- **tf.Example / tensor cache**：用于模型输入和 Waymax 对齐。
- **logged future**：可以用于离线监督、natural OBS、auditability、stress mining、label truth；不能影响模型实际消费的 critical set / ego proposal selection。
- **sdc_paths**：用于 Waymax route metrics / rollout route context；除非部署时有同等 route-intent 输入，否则不要把 WOMD 的 future-valid path set 当作训练 label proposal 的 oracle。当前修复版 ego proposal 仍从 current map topology 构造 route，不直接使用 `sdc_paths` 选 candidate。

官方来源：

- Waymo Motion: https://waymo.com/open/data/motion/
- Waymo tf.Example format: https://waymo.com/open/data/motion/tfexample/
- Waymo v1.3.1 download/version history: https://waymo.com/open/download/
- Waymax: https://github.com/waymo-research/waymax
- Waymax `config.py`: https://raw.githubusercontent.com/waymo-research/waymax/main/waymax/config.py

---

## 3. 为什么 v16.8.19 strict 不支持 full rebuild

### 3.1 不是 natural/model support 失败

`v16_8_18_strict_verdict.json`：

- `sparse_label_build_complete=true`
- `training_supervision_pass=true`
- `model_support_pass=true`
- `auditability_coverage=true`
- `certificate_complete_scene_coverage=true`
- `natural_rootless_zero_on_auditable=true`
- `natural_lt2_low_burden_zero_on_auditable=true`
- `protected_prio_coverage=true`
- **仅** `proposal_causal_screen_pass=false`

因此继续向 natural bank 塞更多 roots 不是当前最优方向。

### 3.2 旧 strict 用 all-critical candidate label 卡住了 primary protected-priority certificate

`witness.py` 中旧 `cowp/candidates/noncoercive_feasible` 是：对所有 mechanism-valid critical pair 做全称 AND；任一 critical pair 不满足就使 candidate global NCF=false。

但模型 `set_transport / BCOT` 的主 hard certificate 明确还有 protected mask：`rho in {AGENT_PRIORITY, EQUAL_OR_NEGOTIATED}`。论文主定义也以 protected-priority agents 为硬对象；all-critical 是更强的 global burden-transfer audit。

v16.8.19 结果已经显示二者差异巨大：

- all-critical any-NCF：0.328–0.350；
- protected-priority diagnostic any-NCF：0.566–0.586；
- PBTR 已通过 0.45 strict threshold。

因此旧 gate 把“global auxiliary head 的更严格目标”错误提升成了“primary hard-certificate 的 promotion veto”。不过 v16.8.19 的 priority-NCF 是旧脚本重构指标，不应直接当成最终 exact pair-NCF 证据；v16.8.20 已把 protected candidate labels 显式写入 cache，必须 fresh rebuild 后重测。

### 3.3 conflict-region bank 存在系统性 raw-order truncation

v16.8.19 `build_conflict_regions()` 在遍历 WOMD map lanes 时，一旦累计到 `max_conflict_regions=64` 就直接返回；没有按 SDC current pose / heading / reachable route 排序。

strict profile 的统计：

- 1166 / 1200 scenes 的 `num_conflict_regions == 64`，即 **97.17%** 场景命中上限；
- mean=63.24，median=64，P90=64。

这意味着“前 64 个”在绝大多数场景中不是一个无关紧要的实现细节，而是实际决定所有后续 proposal/critical/witness 的冲突图。由于 WOMD map feature 遍历顺序不是 COWP 的 ego-relevance ranking，这会把真实前方 conflict 排除出去，或者让 proposal solver 针对不重要的远端 conflict 做 timing。

### 3.4 critical selector 名义上 causal，实际使用 logged future oracle

v16.8.19 `critical_agents.py` 注释声称未来只用于 auditability，但实际 selection score 使用：

- agent logged future min distance；
- logged-future TTA；
- future-dependent priority；
- `objects_of_interest / tracks_to_predict` 额外加分；
- `fixed_anchor_v1` 甚至把 logged ego future 放入 selection anchor bank。

而 `COWPModel.forward` 直接消费 cache 中选出的 critical set。因此即使 raw model feature 没有 future tensor，**critical membership 本身已经泄露未来信息**。这会造成 train/offline evaluation 与 deployment 不一致，必须在 full rebuild 前修掉。

### 3.5 现有 proposal 解决的是单 conflict region，不是 scene-level joint feasibility

v16.8.19 RT-NCF/group pass-after 对 `(route, one conflict region)` 内的一组 protected agents 求 pass-after timing；但 candidate NCF 是 scene-level 的：同一 ego trajectory 必须同时满足所有 relevant/protected critical pairs。

strict critical count：

- `{1:18, 2:35, 3:43, 4:61, 5:56, 6:987}`；
- mean=5.5525，median=6，P90=6；**82.25% scenes 直接打满 A=6**。

因此“每个 region 都各有一个可行 candidate”并不推出“存在一个 candidate 同时对多个 region 可行”。

证据与此完全一致：

- PSY：3137 attempts → 909 valid candidates，但 1200-scene source ablation 的 scene-level any-NCF / PBTR **零增量**；
- RMR：2237 valid candidates，但 scene-level all-critical any-NCF 仅 `+0.000833`（约 1/1200 scene）；
- RMR target TTA error mean `0.00108 s`、max `0.05746 s`，远低于 0.2 s，所以核心不是 timing root solver 数值不准，而是 **联合约束覆盖不够**。

### 3.6 label/model A 维度不一致

label config `max_critical_agents=6`，model config 原为 8。即使 tensor loader 能 pad，这仍会导致模型容量、loss mask、audit 解释不一致。v16.8.20 已统一为 6。

---

## 4. 完整支撑当前代码模型的数据集必须具备的性质

下面是 full rebuild 之前应该固定的数据合同。

### 4.1 Causal input contract（硬条件）

1. `scenario_id` 唯一且 train/val/test/stress/human-audit 跨 split 不泄漏。
2. 模型输入只来自 history/current + map + deployment-available route context。
3. critical membership 不使用 logged future、TTP、OOI；修改 future 而保持 history/current 不变时，critical `track_index/score/base_priority` 必须不变。
4. conflict top-C 不依赖 map feature 插入顺序；必须基于 current SDC pose/heading relevance 后截断。
5. `critical/input_index` 对 tf.Example object axis 可见且 track-id 映射一致。

### 4.2 Candidate support（硬条件）

每 scene 固定 `K=64`，至少包含并保留 provenance：

- logged/keep/accel/yield/stop/creep/lane-change/neutral；
- legacy timing / RMR；
- PSY；
- **joint-route NCF**；
- terminal lattice 仅做覆盖 filler，不能提前挤掉 interaction-conditioned sources。

每 candidate 必须有：valid、map/dynamic checks、proposal source/region/target timing、certificate_valid、conventional_safe、global NCF/false-safe，以及 explicit protected-priority eligible/NCF/false-safe。

必须保证：

- any-valid 几乎全覆盖；
- protected-priority eligible 与 NCF 正类场景足够；
- PBTR 不退化；
- global NCF/false-safe 两类仍有训练支持，但不再作为 primary protected hard-veto；
- source ablation 中 interaction source 确实产生 valid candidate；边际增量用于算法消融而不是 dataset correctness 的必要条件。

### 4.3 Critical / conflict support（硬条件）

- `A=6` label/model 一致；
- current-map causal selection；
- 选中的 pair 有足够 `mechanism_valid` auditability；
- conflict candidate pool 在排序前足够大；允许 selected C=64 饱和，但不能出现“candidate pool 本身经常到上限”的再次截断；
- profile 必须写出 `ego_reference_used`, raw/dedup/selected region count, candidate_pool_saturated。

### 4.4 Natural root support（当前 v16.8.19 已基本满足）

每 auditable critical agent：

- 至少 2 个 constructive + low-burden natural roots；
- OBS / NEU / PRIO 来源可审计；
- protected agent 几乎总有 PRIO root；
- map compliance、priority preservation、beta、weight/source 均写 cache；
- empirical/logged future 只作为事实/监督证据，不作为 online selector oracle。

### 4.5 Root-indexed response / witness support（当前 strict 已基本满足）

每 relevant `(candidate, critical, natural-root)`：

- root affected / unsafe / direct burden / budget-crossed；
- response bank PRED/OPT/EMG；
- root_index / root_affinity；
- is_safe / is_low_burden / burden components；
- pair NCF、OPR、tail burden、witness、token、interval、conflict region；
- canonical root weights 与 transport tensors 精确一致；
- irrelevant pair 不生成 response bank。

### 4.6 Distribution support（full 数据集，不只 smoke gate）

当前 WOMD 本身已经是 interaction-mined 数据，但 COWP full training 若只保留高互动正例，仍可能使 risk/calibration head 缺少“普通非互动/容易安全”的 scene-level 控制组。建议 full cache 最终按 scenario_id 固定出：

- interaction core：主要训练 counterfactual mechanism；
- random/control slice：保持常规驾驶负例与 calibration；
- hard false-safe stress split：只用于 proposal/certificate evaluation，不回流 train；
- human-audited false-safe holdout：机制 claim 的最终人工校验集；
- Waymax reactive rollout split：closed-loop outcome，和训练 pseudo-label split 分离。

比例不是论文定理；建议先以现有 full split 为主，额外保留 20–30% random/control slice 做 calibration/negative support，并通过 class-support audit 决定是否调整，而不要写死为结论。

---

## 5. v16.8.20 代码修改

### A. `cowp/geometry/lane_graph.py`

- 不再在 raw map traversal 过程中到 64 即 return；
- 建立较大的 candidate pool（默认 4096）；
- current SDC xy/heading 驱动 lane preselection 与 conflict ranking；
- deduplicate 后再 top-C；
- selected 后重新分配 cache-local `conflict_id`；
- profile 输出 candidate-pool saturation / ego-reference diagnostics。

### B. `cowp/label/critical_agents.py`

- 默认改为 `causal_anchor_v2`；
- ego anchors 仅 current state 的 keep / accel / decel / stop / limited lateral anchor；
- agent interaction projection 用 current state causal kinematics；
- logged future 只在选人完成后用于 `mechanism_valid/auditability`；
- OOI/TTP 默认 score weight=0；
- profile 明确写 `logged_future_used_for_selection=false`。

### C. `cowp/label/ego_candidates.py`

新增 `ProposalSource.JOINT_ROUTE_NCF`：

- 沿 current-map route 取多个 ego-reachable conflict regions；
- 对每 region 收集 protected agents 的 causal TTA envelope；
- 构造一个统一的 route timing/deceleration bound；
- 最终用**同一 trajectory**对所有 active region 的 target TTA 做 hard validation；
- 在 terminal filler 之前插入 bank。

### D. `witness.py / label_engine.py / cache_schema.py / dataset.py / losses.py`

显式新增：

- `cowp/candidates/priority_eligible`
- `cowp/candidates/priority_false_safe`
- `cowp/candidates/priority_noncoercive_feasible`

模型 primary priority candidate budget 优先使用 exact cached labels；旧 cache 仅保留兼容 fallback。

### E. strict / audit scripts

- scripts 45/46/50/62/65 全部读取 explicit priority labels；
- source ablation 新增 `without_joint_route_ncf` / `base_plus_joint_route_ncf`；
- profile summary 汇总 conflict selection diagnostics；
- strict gate primary hard metrics 改为 protected-priority，global all-critical 保留为 non-degenerate auxiliary support；
- strict 强制 `causal_anchor_v2`、ego-relevant conflict ranking、candidate pool 不系统截断；
- JR-NCF 必须在 strict 中真实产出候选，但它的边际 scene-level gain 只做 attribution advisory；
- strict composite verdict 改成直接列 `failed_checks`，不再错误地统一提示 natural/model support。

### F. Waymax

`cowp/waymax_eval/dataloader.py` 显式设置：

- `aggregate_timesteps=True`
- `batch_by_scenario=True`
- `max_num_objects` 与 env/model 合同一致
- `include_sdc_paths` 按 route metric 需要开启

### G. 维度

`configs/model_cowp_v16_8.yaml` 的 `max_critical_agents` 从 8 改为 6，与 label 一致。

---

## 6. v16.8.20 promotion gate

strict 仍不是“放宽阈值以通过”，而是改变成与模型语义一致的双轨 gate：

**primary protected-priority（hard）**

- any valid >= 0.99
- priority eligible >= 0.90
- priority NCF >= 0.50
- PBTR <= 0.45

**global all-critical auxiliary support（hard non-degeneracy）**

- global any-NCF >= 0.30
- global false-safe floor <= 0.65

**data semantics（hard）**

- stable critical reference = 100% `causal_anchor_v2`
- ego-reference conflict ranking = 100%
- conflict candidate-pool saturation <= 5% strict
- JR-NCF source candidates > 0
- audit/transport invariants pass

**diagnostic/advisory**

- legacy global threshold `any_ncf>=0.40` / floor `<=0.55`
- JR-NCF scene-level marginal gain
- burden-only prevalence

注意：v16.8.19 cache 不能直接套新 gate，因为它缺少 causal selector/conflict diagnostics 与 explicit protected candidate labels；必须 fresh rebuild，避免“改 gate 就通过”的假 promotion。

---

## 7. 测试状态

补丁中的针对性测试：**24 passed**（最新组合）；其中新增：

- 修改 logged future 不改变 causal critical selection；
- map feature insertion order 不改变 conflict top-C；
- explicit priority labels 存在于 cache contract；
- label/model A 维度一致；
- JR-NCF source/config 存在；
- 原 natural/transport/set-transport 相关测试继续通过。

全套：**229 passed, 5 skipped, 2 failed**。两个 failure 在原始 v16.8.19 zip 上也同样存在，原因是原包缺少两个历史 wrapper 文件：

- `NEXT_RUN_COMMANDS_V16_8_14_CAUSAL_AUDIT_SMOKE_CN.sh`
- `NEXT_RUN_COMMANDS_V16_8_9_STRICT_PROPOSAL_PROBE_CN.sh`

它们与 v16.8.20 修改无关；没有为了“全绿”而伪造历史文件。

---

## 8. 下一步执行

直接按 `NEXT_EXECUTION_V16_8_20_COMMANDS_CN.txt`：

1. targeted unit tests
2. split-audit
3. preflight
4. **fresh** 96-scene smoke
5. 看 `base_screen + fresh_profile_summary + source_ablation`
6. fastpath-ab
7. smoke 授权后才 1200-scene strict
8. strict 授权后 train-pilot
9. train-pilot 授权后 full-core
10. 最后 Waymax outcomes

**不要**把 v16.8.19 NPZ 复制过来做 reaudit；critical membership、conflict ids、candidate priority labels 都已经变了。

---

## 9. 新 smoke 最值得先看的字段

如果 v16.8.20 smoke 仍 fail，按以下顺序定位：

1. `fresh_profile_summary.json -> critical_selection_reference_modes`
   - 必须全部 `causal_anchor_v2`。
2. `conflict_region_selection.ego_reference_used_rate`
   - 必须 1.0。
3. `conflict_region_selection.candidate_pool_saturation_rate`
   - smoke 应 <= 0.25，strict <= 0.05；若高，继续增大 pool 或先做 route/reachability pruning。
4. `proposal_source_ablation.proposal_source_candidate_counts.JOINT_ROUTE_NCF`
   - strict 必须 >0；如果为 0，检查 route attach / protected-agent TTA envelopes。
5. `base_screen.observed.new_priority_ncf_scene_rate`
   - 比 global NCF 更关键。
6. `new_pbtr_floor`
   - priority candidate 有但 PBTR 高，说明 joint proposal 仍在压缩 protected options。
7. `global_any_ncf/global_false_safe_floor`
   - 只要 auxiliary head 两类不退化即可；如果极差，说明 critical universe 或 all-critical head 太苛刻，需要另外分析，而不是牺牲 primary protected semantics。
8. natural/model/training audit
   - 只有相应 check 真的失败才回头修 natural/response/model support。

---

## 10. 不能在本地证明的部分

你没有上传庞大的 NPZ，也没有提供本环境可读的 WOMD 原始 1.3.1 shards，所以本次能完成的是：

- 论文/代码/JSON 结果的完整静态与诊断分析；
- 代码修复；
- unit/static contract test；
- WOMD/Waymax 官方语义核对；
- 新的 smoke/strict promotion protocol。

**不能诚实地宣称 v16.8.20 的新 smoke/strict 数值已经通过**。下一条实证证据必须来自你机器上 fresh v16.8.20 smoke 的新 profile/JSON；如果失败，新的 `failed_checks` 会比 v16.8.19 的泛化 `next_action` 精确很多。
