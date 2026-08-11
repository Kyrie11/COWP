# COWP v16.8.10：算法—数据契约审计、strict probe 失败分析与全量重构方案

## 0. 结论先行

当前 `formal_v16_8_9_causal_audit_strict_probe_optimized` **不应该进入 full rebuild**。它不是单一阈值问题，而是两个独立层面的失败叠加：

1. **模型监督契约失败**：`model_support_audit.json` 中 `cowp/response/valid` 共 4,830,368 个样本，正类 4,830,368、负类 0；但模型仍以非零 BCE 学习 validity，并用预测 validity 去门控 SetTransport。这是“固定容量 response bank 的占位掩码”被误当成“可学习语义类别”。
2. **困难场景 proposal/NCF 恢复仍不足**：400 个旧 hard scenes 上 `hard_scene_ncf_recovery_rate=0.1575`，低于 strict gate 的 0.20。与此同时，代表性 800 场景的 AnyNCF=0.415、PBTR lower bound=0.4312、false-safe floor=0.535 已经分别满足 0.40 / 0.45 / 0.55，所以不能靠简单放松总门槛解决。

本补丁将这两类问题分开处理：

- 修正 **数据/模型语义不一致**（response validity、natural-root support、审计漏检、RSS false blocker）；
- 修正 **hard-scene NCF 支撑不足的高概率根因**（主论文 certificate 的 vehicle-only 语义、OBS identity root、弯道 neutral/prio root、RSS 横向约束），但**不降低 strict 的 AnyNCF / false-safe / PBTR / hard-recovery 阈值**；
- 对最耗时的 safe-response / witness 路径加入可 A/B 证明的 **exact fast path**；
- full core 数据通过全部门后，再做 Waymax outcome replay，避免先花 GPU 时间再发现数据契约失败。

由于当前审查环境没有你的原始 WOMD shards，本地不能重新跑 96-scene smoke / 1200-scene strict / full rebuild。因此本补丁的最终 promotion 仍必须以你机器上的新 smoke + strict 结果为准。

---

## 1. 我对“当前代码真实算法”的理解

以下以代码 `ALGORITHM_CHANGELOG.md` 的 v16.8.9/v16.8.10 为准，而不是强行要求和论文早期文字逐字一致。

### 1.1 固定参考的 global critical universe

- `critical_agents.py` 先用与可选 proposal family 无关的 fixed-anchor bank 选择 global critical agents，避免“增加一个 proposal，反而改变所有旧 proposal 的被审计 agent 集合”。
- v16.8.10 主线设置 `critical.vehicle_only_main=true`：**非胁迫 certificate 只对车辆主线成立**；常规 collision safety 仍对所有 logged non-SDC actors 检查。
- 这是和当前论文结论一致的范围界定：pedestrian/cyclist 被论文放在 future work，而不是把尚未验证的 VRU burden model 混进主结论。

### 1.2 candidate-conditioned causal audit

对每个 `(ego candidate k, critical agent i, natural root m)`：

- `root_unsafe`：几何碰撞 / near-miss / RSS / severe margin；
- `root_direct_burden`：候选 ego 直接施加到 natural root 的 primitive burden；
- `root_affected = root_unsafe OR direct_burden > beta`；
- 用 floor-smoothed canonical natural root mass 求 `relevance_mass`；
- 只有超过 relevance support 的 pair 才进入 response search / witness / transport；irrelevant pair 对该 candidate 视为 vacuous NCF。

这解决了 v16.8.8 “global critical agent 对所有 candidate 一律硬审计”造成的 silent blocker。

### 1.3 三类 natural roots

当前 natural basis 是 typed root set：

- **OBS**：贴近 logged empirical future 的 observational roots；
- **NEU**：移除 candidate pressure 的 ego-neutral proxy；
- **PRIO**：priority-preserving rule roots。

OPR/NCF 的关键不是“有一条 natural trajectory”，而是**稳定 root + 归一化 probability mass + 同 root recovery**。因此一个 critical agent 只有 arbitrary root、不具备 low-burden root，仍然不能支撑当前算法。

### 1.4 fixed-cardinality response bank 与 same-root recovery

对 relevant pair，response bank 由 PRED / OPT / EMG 组成，并为每个 natural root 搜索同 root 的低 burden safe recovery。

v16.8.10 明确：

- `response/valid` = 固定 R=32 bank 的 occupancy/padding mask；
- 语义监督是 `is_safe`、`is_low_burden`、`burden_total/components`；
- SetTransport 不再把一个全为 1、不可辨识的 validity logit 当概率门。

### 1.5 witness / affected-root transport / BCOT

- witness 只在 relevant pairs 上成立；
- positive witness = 足够 affected mass 且 OPR shortfall 或 conflict-conditioned tail burden 超阈值；
- RootTransport 使用 `mode_affected`，不是只看 collision conflict；
- BCOT 用 protected-priority certificate 做主 hard gate，同时保留 all-critical 诊断，防止把 burden 转移到非 protected agent 后“指标看起来变好”。

### 1.6 planner supervision

当前主 train config `closed_loop=0.35`。因此主线 planner 训练需要真实 attached Waymax candidate outcomes；否则 `planner_outcome_supervision()` 会返回零，而训练过程仍可能继续，实际变成了另一个算法。

v16.8.10 已改为：

- planner/all + `closed_loop>0` 时，没有 `--with-waymax-outcome-labels` 直接 fail；
- outcome cache 必须至少有 valid、safe、unsafe，并至少存在一个同场景 safe-vs-unsafe ranking pair；
- 若要做 `closed_loop=0`，必须显式使用 ablation config，而不是“忘了附 outcomes”。

---

## 2. 当前算法要求的数据集性质

| 层级 | 必须满足的性质 | 为什么是硬要求 | v16.8.10 gate |
|---|---|---|---|
| WOMD 原始源 | train/val Scenario 为 91 帧（10 past + current + 80 future），SDC current valid | label engine 需要 future-visible offline supervision | `64_validate_womd_v131_contract` |
| tf.Example | 128-object tensor contract，past/current/future shape 完整 | tensor cache + Waymax state source | 同上 + `02_build_tensor_cache --require-waymax-ready` |
| WOMD 1.3.1 | `sdc_paths/path_samples` 存在 | route progression / wrong-way 等完整 Waymax metric | `--require-sdc-paths` |
| split | train/val scenario ID 零重叠 | future-visible pseudo label 下尤其不能有 leakage | `09_check_splits --fail-on-overlap` |
| model input | encoder 只使用 past+current；future 只可作 label/replay | 防止 offline future leakage | 新增 future-invariance regression |
| critical agent | 主论文 certificate 仅 vehicle；physical collision 仍覆盖所有 actor | 与论文当前 claim scope 对齐 | `vehicle_only_main=true` + per-type audit |
| natural root | 每个 valid critical 至少 1 root，且至少 2 个 **low-burden** roots | OPR 是 retained natural mass，不是 arbitrary trajectory 存在性 | `65_audit_model_support --strict` |
| natural mass | valid root 权重 finite、nonnegative、sum≈1 | OPR / affected mass / tail 都依赖统一 measure | strict model support |
| natural sources | OBS / NEU / PRIO 都有足够样本 | 支撑 counterfactual ablation 与 source-restricted learning | strict model support |
| candidate bank | AnyValid 高、conventional-safe 与 NCF 都存在；hard scene 有恢复 | planner 必须看到可比较 intervention | smoke/strict proposal gates |
| causal audit | conflict⊂affected；affected definition 与 transport 完全一致；无 silent/irrelevant blocker | witness 和 RCOT 必须监督同一个机制对象 | `57` + `58` + `60` |
| response | relevant pair 的固定 response slots 完整；safe/unsafe、low/high burden 非退化 | response classifier/transport 必须可学 | v16.8.10 model support |
| witness | relevant pairs 上 witness 有正负；机制 token 不止一个正类 | witness existence/token head 可辨识 | strict model support |
| transport | affected root 有 root-indexed responses、recovery 正负、root burden variability | BCOT/SetTransport 的核心监督 | strict model support |
| planner outcomes | attached Waymax outcome 有 safe/unsafe/mixed scenes | `closed_loop=0.35` 的 classification/ranking/expected-cost 才真正激活 | 新 outcome support gate |

**重要区分**：WOMD train/val 的 80 帧 future 真值用于“离线构标签”是合理的；它不能进入模型的在线 scene encoder。当前 `build_agent_history_from_womd()` 只读取 past/current，v16.8.10 新测试进一步把 future 任意改成极端值，要求 encoder history 完全不变。

---

## 3. 为什么当前 strict probe 没通过

### 3.1 真正触发 wrapper 停止的是 `model_support`

`v16_8_9_strict_verdict.json`：

- `recommend_full_rebuild=false`
- `failure_stage="model_support"`
- return code 2

这说明后续 screen 没有资格把它 promotion 成 full rebuild。

### 3.2 `response/valid` 是确定性的 1，却被当作 learnable target

`model_support_audit.json`：

- total = 4,830,368
- positive = 4,830,368
- negative = 0
- positive rate = 1.0

但其它 response label 并不退化：

- safe positive rate = 0.4522
- low-burden positive rate = 0.4346
- min-burden positive rate = 0.1762

因此问题不是“response 数据没有信息”，而是 **valid 这个字段本来只是固定 bank 占位语义，不应该独立学习**。

### 3.3 natural-root support 有缺口，而且旧审计还低估了缺口

旧 `model_support` 报：

- `every_critical_has_natural_root=false`
- `every_critical_has_multi_root_support=false`
- rootless critical = 1050（旧审计实现下的下界）
- `<2 natural roots = 24`

旧审计只有“该 scene 至少有某个 natural root”时才进入 per-critical rootless 统计；若整个 scene 的 critical 都 rootless，反而漏记。v16.8.10 修正该 loop，并新增 **low-burden root** 数量审计。

### 3.4 hard-scene recovery 是独立失败，不应被 `model_support` 修复掩盖

`paired_proposal_probe.json`：

- 代表性 800：AnyValid=1.000
- conventional-safe=0.950
- AnyNCF=0.415（pass ≥0.40）
- priority-NCF=0.5375
- false-safe floor=0.535（pass ≤0.55）
- PBTR lower bound=0.4312（pass ≤0.45）
- **hard recovery=0.1575（fail <0.20）**
- old NCF scenes=338，new=332，retained=252，gained=80，lost=86

这意味着不能只把 `response_valid` gate 删除就认为 full rebuild 安全。

### 3.5 proposal source ablation 说明“继续堆 candidate 数量”不是最优第一修复

1200 scenes 的 source ablation：

- all bank AnyNCF = 0.32917
- without RMR = 0.32917
- without PSY = 0.32833
- RMR 增加约 1.71 个 valid candidates/scene，但 scene-level AnyNCF 增量为 0

这说明当前瓶颈更像是 **certificate/root/audit semantics**，不是单纯 proposal 数量太少。v16.8.10 因此优先修 false blockers / natural basis，而不是继续无约束增加 proposal family。

### 3.6 causal-audit 内部一致性其实很好

`causal_audit_diagnostic.json`：

- relevant pair rate = 0.41824
- witness | relevant = 0.66011
- relevant pair response coverage = 1.0
- irrelevant pair response coverage = 0
- affected roots = 1,442,094
- unsafe roots = 1,441,979
- burden-only affected = 115
- affected/transport mismatch = 0
- silent blocker = 0
- irrelevant blocker = 0

所以 v16.8.9 的 causal-audit/affected-root transport 主结构没有必要推翻；应修其外围数据契约与几何误判。

---

## 4. v16.8.10 代码修复

### 4.1 fixed response validity contract

修改：

- `configs/train_cowp_v16_8.yaml`: `response_valid_bce=0`
- 两个 v16.8.9 learned ablation train config 同步
- `configs/model_cowp_v16_8.yaml`: `use_response_valid_gate=false`
- `set_transport_head.py`: 支持关闭 learned validity gate
- `65_audit_model_support.py`: 不再要求 validity 正负两类，而要求 relevant pair response bank 完整占位

不改变 safe / low burden / burden / root recovery 的语义标签。

### 4.2 vehicle-only main certificate

`configs/label_cowp_v16_8.yaml`：

```yaml
critical:
  vehicle_only_main: true
```

只缩小当前论文的 **coercion certificate** 范围；常规 collision/offroad 仍检查其他对象。这样避免用尚未建立 VRU priority/burden semantics 的标签去阻断车辆交互主结论。

### 4.3 RSS / TTC 共同几何契约

修复：

- type-aware TTC / near-miss 阈值；
- longitudinal RSS 增加 lateral corridor gate，避免相邻车道被当成纵向跟驰；
- broad phase 即使距离超过普通 TTC 半径，仍执行廉价的 long-gap high-speed RSS；
- burden 的 risk term 和 `unsafe_between()` 读取同一套 RSS/TTC config。

这是 hard-scene recovery 可能改善的重要来源：过去的 adjacent-lane false RSS 会把本可 NCF 的 candidate/root 错标为 affected/blocker。

### 4.4 natural root budget 修复

旧 OBS Cartesian enumeration 在固定 `max_obs_modes=8` 时先把 `speed_scale=0.85` 的组合塞满，甚至可能没有 `(1.0,0,0)` identity-like observational root。

新实现：从 `(speed=1,time_shift=0,lateral=0)` 向扰动幅度递增分配 OBS mode budget。

### 4.5 curved-lane neutral fallback

旧 neutral/prio 的直线常加速度 fallback 在弯道上容易被 map filter 全删。

新 fallback：

- 只复用 logged future 的**路径几何**；
- 从 current state 重新生成 bounded neutral timing；
- 不复制可能已经包含 ego-induced yield 的 logged timing；
- 仍必须通过 map / priority / low-burden plausibility。

### 4.6 model-support audit 修复

现在逐 critical agent 检查：

- any root；
- ≥2 roots；
- any low-burden root；
- ≥2 low-burden roots；
- root weights finite/nonnegative/normalized；
- per object type 计数，便于发现 scope 或 parser 问题。

### 4.7 planner outcome hard gate

`03_train.py` 新增：主线 planner/all 若 `closed_loop>0`，则：

- 未开启 outcome labels -> fail；
- keys 缺失 -> fail；
- valid outcomes=0 -> fail；
- 没有 safe/unsafe 双类或 mixed ranking scenes -> fail。

---

## 5. WOMD 1.3.1 `scenario/training` 的正确使用方式

根据 Waymo 官方 Motion / tf.Example 文档和 Waymax：

1. train/val 场景有 **10 history + 1 current + 80 future**；这正适合 COWP 离线构造 counterfactual pseudo-label、witness 和 natural targets。
2. Scenario proto 保留 tracks、map features、traffic controls、SDC index、objects_of_interest、tracks_to_predict、current_time_index 等结构化语义；tf.Example 提供同类信息的 tensorized 版本。
3. `tracks_to_predict` 最多 8 个，是 challenge/submission 的预测对象参考。官方明确说 training 中“可以自由选择其他 objects 训练”，因此**不能用它作为 COWP critical-agent universe**。
4. `objects_of_interest` 是交互兴趣对象，也只能作为 interaction prior/score，不能代替 COWP 的 burden-oriented critical selection。
5. 每个 scenario 有自己的坐标原点；不要跨场景把 x/y 当统一城市坐标直接拼接。
6. WOMD v1.3.1（2025-10）新增 `sdc_paths/path_samples`；它提供 future valid routes、arc length、road-part ID、on-route metadata，并使 Waymax 的 wrong-way / route progression 等完整指标可计算。
7. train/val future 真值可用于**标签**，但模型在线输入必须限制为 past/current。v16.8.10 已验证这一点。

代码里的 `64_validate_womd_v131_contract.py` 会在 full build 前抽样验证 Scenario/tf.Example train+val、91 帧、SDC current、128-object tensor 和 sdc_paths；`02_build_tensor_cache --require-sdc-paths` 再对实际匹配 scene 做逐场景约束。

---

## 6. full rebuild 速度分析

strict probe profile（1200 scenes）的每场景均值：

- total = **303.69 s**
- label engine = **302.88 s**
- safe responses = **139.31 s**
- witness = **122.26 s**
- audit relevance = 19.82 s
- critical selection = 19.52 s
- natural = 1.13 s
- candidates = 0.83 s
- proto parse = 0.010 s
- write NPZ = 0.059 s

safe responses + witness 占 label engine 时间约 **86.36%**。因此优化 TFRecord parsing、压缩、候选生成几乎不是第一收益点。

### 6.1 exact fast path：主要 CPU 优化

`unsafe_between()` 已经为一个 response/candidate pair 计算 collision/TTC/RSS 并证明 safe。旧代码随后 `compute_burden()` 再算一遍相同 TTC/RSS。

在同一阈值下，既然 `unsafe_between()==False`，risk mask 必为 0，因此 v16.8.10 可直接令 risk contribution=0，跳过重复几何计算。

这个 fast path：

- 不近似；
- 不改阈值；
- 不跳过 response candidate；
- 可通过 `engineering.risk_known_zero_fastpath=false` 关闭；
- `NEXT_EXECUTION... fastpath-ab` 会在相同 smoke IDs 上构建 slow reference，并用脚本 66 **逐 tensor exact equality** 比较。

只有 A/B `pass=true` 才建议进入 strict/full。

### 6.2 不牺牲性质的流程级加速

- `REUSE_OLD_SCENE_SET=1`：只复用旧 cache 的 **scenario ID allowlist**，不复用旧 labels；既保证 paired comparability，也避免扩大 scene set。
- index 在同一 WOMD release 下可复制/复用；labels 全部新建。
- `RUN_LABEL_DIAGNOSTICS=0`：full build 期间不画图，promotion 后再做。
- `RUN_WAYMAX_REPLAY=0`：先把 fresh label + tensor core 全部严格通过，再 replay；避免数据失败后白跑 GPU。
- `--no-compress`：构建阶段避免 CPU 压缩开销。
- scene-level multiprocessing + BLAS/TF thread=1，防止 24/32 个进程各自再开线程导致 oversubscription。
- `--skip-existing` + fingerprint：同一代码 lineage 可以安全 resume；fingerprint 变化必须新 root。

### 6.3 Waymax replay 的 exact 多 GPU 加速

新 `ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh`：

- `WAYMAX_GPUS=0,1,...`
- deterministic `--num-shards/--shard-index`
- 每 shard 独立 resume JSONL
- `state-source cache`
- `absolute_xy_yaw`
- horizon 80
- `metric-set safety`
- 每 step metric evaluation 保持默认 exact 语义，不使用 sampled/adaptive/final 近似模式
- attach 后 full-cache 检查 safe/unsafe/mixed support

当前主 config 的 `outcome_logdiv=0` 且 logdiv unsafe threshold 被显式关闭，因此 `metric-set safety` 足够支撑当前 closed-loop loss；无需为不参与优化的 log divergence 付出额外 replay 成本。

---

## 7. 新 promotion 流程

### Step 0：源码预检

```bash
cd /path/to/COWP_v16_8_10_optimized
bash NEXT_EXECUTION_V16_8_10_CN.sh preflight
```

必须得到 WOMD preflight `pass=true`。

### Step 1：96-scene smoke

```bash
export FORCE_REBUILD_SMOKE=1
bash NEXT_EXECUTION_V16_8_10_CN.sh smoke
```

必须同时通过 proposal screen、training supervision、model support；任何一个失败都停止。

### Step 2：exact fast-path A/B

```bash
export FASTPATH_AB_SCENES=12
bash NEXT_EXECUTION_V16_8_10_CN.sh fastpath-ab
```

必须：

```json
"pass": true
```

这是“加速不改数据性质”的直接证据。

### Step 3：1200-scene strict probe

```bash
export FORCE_REBUILD_PROBE=1
bash NEXT_EXECUTION_V16_8_10_CN.sh strict
```

不要手改 strict gate。必须最终：

```json
"recommend_full_rebuild": true
```

并重点确认：

- AnyValid ≥ 0.99
- AnyNCF ≥ 0.40
- false-safe floor ≤ 0.55
- PBTR floor ≤ 0.45
- hard recovery ≥ 0.20
- model support pass
- no silent/irrelevant blocker
- affected/transport exact match

### Step 4：full core rebuild

```bash
export LABEL_WORKERS_TRAIN=32
export LABEL_WORKERS_VAL=24
export CACHE_WORKERS=8
bash NEXT_EXECUTION_V16_8_10_CN.sh full-core
```

这一阶段只完成 fresh labels + fresh tensor cache + 全量数据 gates，不做 Waymax replay。

### Step 5：Waymax outcomes

单 GPU：

```bash
export WAYMAX_GPUS=1
bash NEXT_EXECUTION_V16_8_10_CN.sh outcomes
```

多 GPU：

```bash
export WAYMAX_GPUS=0,1,2,3
bash NEXT_EXECUTION_V16_8_10_CN.sh outcomes
```

脚本自动 deterministic shard、resume、attach、verify。

### Step 6：planner 训练时

主线 cache 应使用：

```bash
TRAIN_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_10_full/tensor_cache_train_waymax
VAL_CACHE=/data0/senzeyu2/dataset/COWP/formal_v16_8_10_full/tensor_cache_val_waymax
```

并必须带：

```bash
--with-waymax-outcome-labels
```

否则当前 `closed_loop=0.35` 主线会主动 fail，而不是静默变成无 closed-loop supervision 的模型。

---

## 8. 不应该做的事情

1. **不要**为了让 strict 通过把 hard-recovery 0.20 降到 0.15；这只会把已知失败 promotion 到 full。
2. **不要**把 `response/valid` 人工制造负样本；它本来就是 fixed-bank occupancy，不是可学习行为标签。
3. **不要**用 `tracks_to_predict` 过滤 critical agents；那是 challenge prediction reference，不是 burden universe。
4. **不要**把 train/val future trajectory 输入 encoder；future 只用于 offline target / audit / replay。
5. **不要**在 strict verdict 仍为 false 时开始 full label build。
6. **不要**在 core data gates 没通过前跑全量 Waymax replay。
7. **不要**把 logged Waymax replay 的 burden proxy 当作真实 causal ground truth；论文自己已经要求 reactive-agent + human-audited stress set 才能支撑最终 causal burden claim。

---

## 9. 本地代码验证结果

审查环境完成：

- `pytest -q`: **187 passed, 5 skipped**
- shell syntax：`NEXT_EXECUTION_V16_8_10_CN.sh` / `ATTACH_WAYMAX_OUTCOMES_V16_8_10_CN.sh` 均通过 `bash -n`
- 新增回归覆盖：
  - exact safe-pair risk fast path；
  - fixed-cardinality response contract；
  - TTC/RSS config consistency；
  - adjacent-lane RSS / long-gap high-speed RSS；
  - OBS identity-first budget；
  - curved-route neutral proxy；
  - future tensor 不进入 model history input。

**仍未完成且不能在本地伪造的证据**：新 96-scene smoke、400+800 strict、full rebuild 和 Waymax outcome 分布。它们必须在你的 WOMD 1.3.1 原始数据机器上运行后才能最终确认。
