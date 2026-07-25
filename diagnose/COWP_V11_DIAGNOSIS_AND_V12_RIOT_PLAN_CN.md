# COWP v11 结果诊断、v12-RIOT 修复与 CCF-A 投稿路线

> 审阅范围：`interactive planning.tex`、`COWP.zip`、`构建数据集指令.txt`、
> `cowp_v11_bcot_probe100_seed2026.zip`、`external_baselines.zip`。
>
> 未能审阅：用户提到的 `cache_sufficiency_full.json` 未出现在本次上传文件中。
> 因此，本文对缓存充分性的判断只能依据运行日志、结果 JSON 以及代码中的覆盖率统计，
> 不能替代该文件的完整扫描结论。

## 1. 结论先行

当前 v11 **不能支撑“闭环 SOTA”或 CCF-A 论文主张**，但原因不是核心 idea 本身被证伪，
而是核心机制在实现中没有被正确训练和正确使用：

1. transport/witness 阶段没有读取候选级 `false_safe` 与
   `noncoercive_feasible` 标签，导致 `set_transport/candidate_budget` 在整个
   transport 训练中恒为 0；
2. `mode_retained_low_safe` 已包含“无冲突且低负担”，前向又乘一次
   `(1-conflict_prob)`，重复惩罚冲突并系统性压低 OPR/接受率；
3. v11 用“通用、无根条件的响应槽 + 24 类 root 分类”间接恢复自然选项，
   但论文要证明的是“每一个自然选项是否存在同根、低负担、安全响应”，两者结构不匹配；
4. natural basis 在 witness 训练中漂移到几十米，导致 root transport 的“根”失去物理意义；
5. pair witness 阈值与 candidate BCOT budget 被当成同一个阈值扫描，统计语义混淆；
6. 高达 0.9003 的 CandidateCertificate AUPRC 不是核心运输机制已经成立的证据，
   而 BCOT 本身的 false-safe AUPRC 只有 0.4115。

因此，本次修复不是简单调学习率或放松阈值，而是进行了**局部结构重构**：
引入 **RIOT（Root-Indexed Option Transport，根索引选项运输）**，直接预测每个自然根
是否还能被安全、低负担地保留，并让这一运输事件成为候选可行性证书的主路径。

修复后的代码已经完成静态检查和单元测试，但**没有在用户 GPU/WOMD 环境中重新训练**。
所以不能诚实地保证“必然达到 SOTA”；它解决的是当前可确定的逻辑故障，并把下一次训练
变成一套可证伪、可定位、不会在 gate 失败后继续浪费闭环算力的流程。

---

## 2. 论文核心 idea 与目标

论文定义了一个常规 collision-free 指标遗漏的失败模式：

- ego 轨迹自身没有碰撞；
- 但安全成立依赖其他道路参与者硬刹、突然让行、放弃合法优先权或交出自然间隙；
- 因此这是 **false-safe planning**，不是“安全但不够礼貌”。

论文的核心目标是把这种情况从 soft courtesy cost 提升为**可行性缺陷**：

> 对每个关键参与者，都应保留足够丰富的低负担安全响应，而不是只剩一个脆弱、昂贵的
> 逃生动作。

论文 pipeline 可归纳为：

1. **Burden-Oriented Interaction Graph**：定位关键参与者、冲突时段、优先关系与承担冲突的一方；
2. **Ego candidate generation**：生成满足路线、车道、动力学、舒适性约束的候选；
3. **Counterfactual natural alternatives**：由 observed、ego-neutral、priority-preserving
   三个分支构建非胁迫自然选项；
4. **Ego-conditioned safe responses**：在给定 ego 候选后构建他车安全响应集合；
5. **Hybrid burden**：度量加速度、jerk、进度损失、gap loss、优先权放弃等负担；
6. **Non-coercive feasibility**：至少存在低负担安全响应，并满足 OPR；
7. **Coercion witness**：给出 burdened agent、冲突区间、机制 token 与负担强度；
8. **Hard-first selection/fallback**：先常规安全，再非胁迫可行性，最后才按 ego utility 排序。

核心因果链应当是：

`自然选项 -> ego 条件冲突 -> 同根低负担响应是否存在 -> 选项质量/质量损失 -> 候选 BCOT 风险 -> 硬可行性选择`

论文主张成立的必要条件不是单纯降低 FSR，而是证明：

- primitive/root transport 本身可学习；
- transport 的错误与 false-safe 有统计关联；
- 使用 transport 会改变候选选择；
- 相比 pairmax、candidate-only、soft burden 等替代解释，transport 是主要增益来源；
- 闭环中在降低 false-safe/HBCR/CBS 的同时，不牺牲常规安全与进度。

---

## 3. 数据集构建逻辑与风险

### 3.1 当前构建流程

`构建数据集指令.txt` 的流程是：

1. 读取 WOMD scenario proto，建立 train/val index；
2. 从 scenario proto 构建 counterfactual labels；
3. 从 `tf_example` 构建 tensor cache；
4. 在 Waymax 中对每个场景最多 12 个 balanced 候选进行 80 步 replay；
5. 将 collision/offroad 等候选 outcome 回挂到 tensor cache；
6. 在其上追加 transport sidecar/labels，用于训练 COWP。

该设计总体合理：scenario proto 用于复杂标签构造，tf.Example 用于 Waymax-ready 输入，
通过 scenario ID 对齐。

### 3.2 必须修正或核验的点

1. **验证路径命名不一致**：构建输出为
   `tensor_cache_train_waymax` / `tensor_cache_val_waymax`，验证命令却写成
   `tensor_cache_*_waymax_bal12_safety`。运行前应统一为真实目录，否则可能验证了旧缓存或直接漏验。
2. **只 replay 12 个候选，outcome 覆盖稀疏**：v11 结果中 selected outcome coverage 约
   20%--22%，且 finite log-divergence count 为 0。它只适合作为稀疏辅助监督，不能当成
   全候选闭环 ground truth，更不能替代真实 Waymax rollout。
3. **`--limit 22000/5000` 是“写够即停”**：场景按 TFRecord 遍历顺序处理并经过
   interaction-heavy filter。应验证 shard/order 是否近似随机；更稳妥的论文版本应保存固定
   scenario-ID manifest，并报告各交互类型分布。
4. **必须做 split leakage 检查**：scenario proto、tf.Example、transport sidecar、Waymax
   replay outcome 均应由同一 scenario ID split 约束。
5. **future-derived labels 只能用于监督**：natural/response/witness 标签可用未来轨迹构造，
   但模型在线输入不得泄漏 future state。应将该点写入 paper protocol 与 cache schema audit。
6. **需要上传/重跑 `cache_sufficiency_full.json`**：当前无法验证所有缓存的 outcome 覆盖、
   label completeness、log-divergence 有效性和异常文件分布。

建议先执行：

```bash
python -m cowp.scripts.09_check_splits \
  --train "$COWP_ROOT/index_train.jsonl" \
  --val "$COWP_ROOT/index_val.jsonl" \
  --output outputs/data_audit/split_check.json \
  --fail-on-overlap

python -m cowp.scripts.27_verify_transport_cache \
  --cache-dir "$COWP_ROOT/tensor_cache_train_waymax_transport_v9" \
  --output outputs/data_audit/transport_train_verify.json

python -m cowp.scripts.27_verify_transport_cache \
  --cache-dir "$COWP_ROOT/tensor_cache_val_waymax_transport_v9" \
  --output outputs/data_audit/transport_val_verify.json

python -m cowp.scripts.19_diagnose_waymax_cache_sufficiency \
  --train-cache "$COWP_ROOT/tensor_cache_train_waymax" \
  --val-cache "$COWP_ROOT/tensor_cache_val_waymax" \
  --workers 8 \
  --output-json outputs/data_audit/cache_sufficiency_full.json
```

---

## 4. v11 结果：哪些部分有效，哪些没有生效

### 4.1 v11 calibrated operating point

验证集共有 5013 个场景。v11 calibration 选择 0.40，但状态是
`least_violation`，不是满足约束的 operating point。

| 指标 | v11 BCOT | Conventional safety | 差值（v11-conv） |
|---|---:|---:|---:|
| CR（offline attached outcome） | 0.0000 | 0.0000 | 0.0000 |
| EP | 0.3538 | 0.3870 | -0.0332 |
| Fallback | 0.2731 | 0.1043 | +0.1688 |
| OPR | 0.8005 | 0.7426 | +0.0579 |
| HBCR | 0.2529 | 0.3964 | -0.1434 |
| Selected false-safe | 0.4255 | 0.5881 | -0.1626 |
| Accepted NCF recall | 0.2011 | 1.0000 | -0.7989 |
| Accepted candidate rate | 0.0831 | 0.5489 | -0.4658 |

该 operating point 表明：v11 确实能降低 false-safe 与高负担，但主要通过大面积拒绝候选与
fallback 完成，导致 EP 降低。这是“保守过滤器有效”，不是“非胁迫可行性证书有效”。

### 4.2 生效的模块

1. **candidate--natural relative geometry**：pair witness AUPRC 达到 0.6808，显著高于早期
   版本约 0.43，说明几何条件化是正确方向；
2. **pair witness localization/classification**：虽然未达到强论文门槛，但已经具备可用信号；
3. **BCOT 排序方向**：within-scene NCF--false-safe ranking accuracy 为 0.8306，说明风险的
   相对次序比绝对概率可靠；
4. **OPR/HBCR/selected false-safe 改善**：核心行为目标在选择层面确有响应，不是完全断连；
5. **budget aggregation 优于简单 pair any/max 的理论方向**：避免一个中等误报将整个候选
   否决，仍应保留并继续加强。

### 4.3 未生效或被其他路径替代的模块

1. **candidate budget transport loss**：transport 阶段完全未训练，日志每个 epoch 都是 0；
2. **root recovery**：loss 约 0.86--0.88，response-root CE 约 2.5--2.8，无法支持同根恢复；
3. **natural basis**：validation minADE 从约 48 m 恶化到 60 m 以上，根语义失真；
4. **BCOT 绝对校准**：AUPRC 只有 0.4115，远低于 generic candidate path 的 0.9003；
5. **候选覆盖率**：接受率 8.31%、NCF recall 20.11%，核心证书对真正可行候选召回不足；
6. **闭环验证**：机制 gate 失败后未执行 COWP Waymax probe，当前没有 COWP 闭环结果；
7. **reactive multi-agent claim**：当前真实 evaluator 控制 SDC，其他参与者是 logged/background
   evolution；代码没有实现 paper 可能暗示的 learned/rule reactive mixture。

---

## 5. 逻辑缺陷与修复策略

### 5.1 缺陷一：transport 阶段没有候选级监督

**缺陷大小：致命实现错误，直接修补。**

旧代码只在 planner/planner_eval/all 阶段加载候选 `false_safe` 与 NCF 标签，
而 v11 的 transport 训练使用 `stage=witness`。损失函数看不到标签后构造全零默认值，
`candidate_budget` 恒为 0，导致 BCOT calibrator 没有在 transport 阶段学习候选级语义。

修复：

- witness 阶段加载全部候选级监督；
- 缺标签时不再自动填全负样本；
- 训练入口检查 required prefixes；
- 记录 budget coverage/positive rates；
- coverage 为 0 的 checkpoint 加硬惩罚并阻止 promotion。

### 5.2 缺陷二：OPR 被重复乘冲突概率

**缺陷大小：中到大，语义修补。**

label 中 `mode_retained_low_safe` 已定义为：

`natural valid AND no conflict under candidate AND burden <= beta`。

旧前向再次计算 `retain_prob * (1 - conflict_prob)`，同一冲突被惩罚两次。
这会压低 OPR、提高 unrecovered mass、减少 accepted candidate。

修复：将输出因子化为：

- `conflict_prob`；
- `conditional_low_safe_retain_prob`；
- 最终 retain = `(1-conflict_prob) * conditional_retain`；
- OPR 只使用最终 retain 一次。

### 5.3 缺陷三：响应槽与自然根无条件关联

**缺陷大小：结构性瓶颈，需要部分重构。**

论文要求的是对每个 natural option root 的存在性判断。旧模型先生成 R 个通用响应槽，
再预测它属于 M 个根中的哪一个。这个后验 assignment 同时承担：

- 响应生成；
- root matching；
- low-burden/safe existence；
- existential aggregation。

错误被多次相乘，因此 root recovery 成为瓶颈。

修复后的 RIOT 直接预测：

```text
T[k,i,m] = P(存在一个 valid、安全、低负担、保持 natural root m 的响应)
```

监督 target 由缓存中明确的 `response_root_index` 对
`valid & safe & low_burden` 做 scatter-OR 得到。旧 response bank 只保留为辅助重建与可视化，
不再定义主证书。

这是一种“部分重构”，没有推翻 graph、candidate generator、burden、witness 与 BCOT，
但把最关键的因果变量改成与论文定义一致的 primitive-indexed event，既服务核心 idea，
也比继续堆叠 candidate classifier 更有 novelty。

### 5.4 缺陷四：自然选项在 transport 训练中漂移

**缺陷大小：致命训练流程问题，阶段化修复。**

根索引运输只有在 natural roots 物理准确且稳定时才可识别。v11 在 witness 阶段继续用较大
natural auxiliary 联合更新，导致 minADE 恶化。

修复：

1. 从强 v10/v11 checkpoint 初始化；
2. 单独训练 natural stage；
3. natural stage 冻结 graph backbone，避免破坏已有场景编码；
4. 通过 minADE hard gate；
5. witness stage 冻结 natural module，`witness_natural_scale=0`；
6. planner stage不再重写 transport backbone。

### 5.5 缺陷五：BCOT budget 与 pair witness threshold 混用

**缺陷大小：中等但影响所有 calibration 结论。**

- pair witness threshold：高置信度严重 pair 的 veto；
- BCOT budget：候选允许损失多少自然选项质量。

两者不能共用一个 scalar。修复后离线固定 pair threshold，单独 sweep BCOT budget；
在线显式传入两个参数。

### 5.6 缺陷六：generic candidate classifier 抢占核心机制

**缺陷大小：论文有效性风险，选择路径重构。**

generic candidate latent 可以直接学习 false-safe 标签，却不需要证明 natural option transport。
即使指标高，也不能证明论文 idea。

修复：

- 主 COWP selector 只使用 RIOT/BCOT 风险与不确定性；
- rule/action/outcome risk 仅作明确安全 shield；
- generic candidate certificate 作为 ablation/diagnostic；
- paper 必须报告 candidate-only 与 RIOT 的对照。

---

## 6. v12-RIOT 已修改的代码

主要改动文件：

- `cowp/data/dataset.py`
- `cowp/models/losses.py`
- `cowp/models/set_transport_head.py`
- `cowp/models/cowp_model.py`
- `cowp/scripts/03_train.py`
- `cowp/scripts/04_eval_closed_loop.py`
- `cowp/scripts/25_verify_mechanism_effect.py`
- `cowp/scripts/30_diagnose_bcot_result.py`
- `cowp/scripts/31_calibrate_bcot_budget.py`（新增）
- `cowp/scripts/32_gate_natural_basis.py`（新增）
- `cowp/waymax_eval/policy_wrapper.py`
- `cowp/waymax_eval/rollout.py`
- `configs/train_cowp_v12.yaml`（新增）
- `configs/label_cowp_v12.yaml`（新增）
- `configs/eval_cowp_v12.yaml`（新增）
- `configs/label_cowp_v12_pairmax_ablation.yaml`（新增）
- `configs/label_cowp_v12_pareto_ablation.yaml`（新增）
- `run_cowp_v12_dual_gpu.sh`（新增）
- `ALGORITHM_CHANGELOG.md`（已同步更新）

验证结果：

- `python -m compileall`：通过；
- `pytest -q`：69 passed；
- `bash -n run_cowp_v12_dual_gpu.sh`：通过。

---


### 6.6 新增：直接验证核心 idea 的根级指标

仅看 `BCOT/FalseSafe_AUPRC` 仍不足以证明论文核心机制成立，因为候选级聚合可能在根恢复错误时仍获得一定排序能力。修复版离线评估额外输出：

- `RootTransport/LowSafeExist_AUPRC`：所有有效自然根上的低负担安全响应存在性；
- `RootTransport/ConflictConditioned_AUPRC`：仅在被 ego 候选冲突的自然根上评估，是主机制指标；
- `RootTransport/ConflictConditioned_Recall@0.5`：冲突自然根被恢复的召回；
- `RootTransport/AuxConflictConditioned_AUPRC`：旧 response-bank + root 分类路线，仅作为辅助/消融；
- `RootTransport/NaturalAssignmentMinADE_m`：无序自然根对齐质量。

开发 gate 要求冲突条件 direct root-transport AUPRC >= 0.65，论文目标建议 >= 0.75，并且主头必须显著优于辅助 response-bank。若该指标不达标，禁止把候选级 BCOT 提升解释为“自然选项运输机制生效”。

## 7. CCF-A / SOTA 指标门槛与当前差距

不存在官方统一的“Waymax 规划 SOTA 分数”，且论文的 FSR/CBS/OPR/HBCR 是自定义新指标。
下面是**建议的投稿门槛**，不是官方 leaderboard cutoff。门槛分为：

- development gate：判断是否值得跑昂贵闭环；
- paper-quality target：判断核心 claim 是否足以支撑 CCF-A；
- strong target：为了更接近“闭环 SOTA”叙事而设定。

以 v11 calibrated point 0.40 为当前值：

| 指标 | 当前 v11 | development gate | paper-quality target | 当前到 paper target 的差距 |
|---|---:|---:|---:|---:|
| Pair witness AUPRC | 0.6808 | >=0.60 | >=0.70 | 0.0192 |
| BCOT false-safe AUPRC | 0.4115 | >=0.65 | >=0.75 | 0.3385 |
| Accepted NCF recall | 0.2011 | >=0.30 | >=0.50 | 0.2989 |
| Accepted candidate rate | 0.0831 | >=0.10 | >=0.20 | 0.1169 |
| Fallback | 0.2731 | <=0.25 | <=conv+0.03 = 0.1343 | 超出 0.1388 |
| EP | 0.3538 | 不明显劣于 conv | >=0.3870 或通过 3% 非劣检验 | 0.0332 |
| Selected false-safe | 0.4255 | 至少降 8pp | <=0.4411（相对降25%） | 已达到最低目标 |
| Selected false-safe（强目标） | 0.4255 | — | <=0.35 | 0.0755 |
| OPR | 0.8005 | >=0.78 | >=0.80 且 EP 非劣 | 数值达到 |
| HBCR | 0.2529 | <=0.32 | <=0.2973（相对降25%） | 数值达到 |
| COWP Waymax CR | 无结果 | 100 场景仅 smoke | 相比最强公平 baseline 相对降 >=10%，95% CI 支持 | 无法计算 |

关键判断：

- 当前最大差距不是 false-safe 降幅，而是 **BCOT calibration、NCF recall、fallback、EP**；
- OPR/HBCR/selected false-safe 已显示方向性优势，应继续加强，但不能以牺牲候选覆盖率为代价；
- CCF-A 论文需要显示 Pareto improvement：在常规安全与效率非劣的前提下显著降低胁迫；
- “SOTA”至少需要在相同场景、相同非 ego policy、相同 horizon、相同 candidate/action interface 下，
  对强 baseline 做 paired closed-loop comparison，而不是引用当前 50 场景结果。

---

## 8. 外部 baseline 的可用性判断

当前 `external_baselines.zip` 包含 DTPP、GameFormer、Frenet optimal、IDM lattice、
state lattice 的 adapted/reference-family 实现。

### 8.1 learned-offline 结果

| 方法 | CR | EP | FSR | CBS | OPR | HBCR | Selected FS |
|---|---:|---:|---:|---:|---:|---:|---:|
| DTPP | 0.3794 | 0.6203 | 0.6500 | 1.0426 | 0.6920 | 0.4967 | 0.4034 |
| GameFormer | 0.3712 | 0.2619 | 0.6063 | 0.9174 | 0.7409 | 0.4329 | 0.3812 |
| Frenet optimal | 0.1043 | 0.6486 | 0.7107 | 1.0454 | 0.6901 | 0.5007 | 0.6365 |
| IDM lattice | 0.1043 | 0.9410 | 0.7927 | 1.1632 | 0.6376 | 0.5625 | 0.7100 |
| State lattice | 0.1043 | 0.9445 | 0.7962 | 1.1667 | 0.6361 | 0.5651 | 0.7131 |

这些值用于同一缓存上的诊断有意义，但 attached outcome 覆盖不完整，不能当作真实闭环 SOTA 表。

### 8.2 Waymax 50 场景结果

| 方法 | CR | Collision | EP | Kinematic infeasible | Offroad | Log divergence |
|---|---:|---:|---:|---:|---:|---:|
| GameFormer | 0.52 | 0.52 | 0.4055 | 0.20 | 0.02 | 15.02 |
| IDM lattice | 0.30 | 0.24 | 0.9612 | 0.20 | 0.06 | 9.97 |
| Frenet optimal | 0.32 | 0.24 | 0.9505 | 0.24 | 0.08 | 7.87 |
| State lattice | 0.34 | 0.26 | 1.1126 | 0.18 | 0.08 | 9.49 |
| DTPP | 0.40 | 0.32 | 1.3377 | 0.22 | 0.10 | 11.27 |

只有 50 场景，置信区间非常宽，而且 COWP 指标在这些 baseline evaluator 中没有被真实计算
（诊断摘要为 0/1 常数）。因此不能据此宣布 COWP 超越或未超越它们。

投稿版本应做到：

1. 所有方法使用完全一致的 scenario IDs；
2. 固定 non-ego policy（首先 logged replay；再补 IDM/reactive robustness）；
3. 固定 rollout horizon、action mode、dynamics、route metric；
4. 所有方法都计算标准指标与 COWP 指标；
5. 对方法实现注明 faithful reproduction、official checkpoint 或 adapted family；
6. paired bootstrap 95% CI，而不是只给均值。

---

## 9. 下一轮实验顺序

### Phase A：数据与机制 sanity（必须先通过）

1. split/cache audit；
2. natural stage 训练并通过 minADE gate；
3. 检查 transport history：
   - `candidate_budget_coverage > 0`；
   - `candidate_budget` 不为 0；
   - direct `mode_recovery` 持续下降；
   - auxiliary root recovery 可以较差，但不能控制主证书；
4. BCOT budget sweep 至少产生 3 个不同选择点；
5. calibration 状态必须为 `constraints_satisfied`，不能是 `least_violation`。

### Phase B：100 场景 smoke

只用于排查：

- Waymax action/interface 是否稳定；
- CR/offroad/kinematic infeasibility 是否出现灾难；
- COWP 与 conventional 的场景集合是否完全一致；
- fallback 是否在连续步骤振荡；
- pairmax/RIOT/Pareto ablation 是否真的改变选择。

100 场景不能用于论文主表。

### Phase C：1000 场景 development

- full RIOT；
- conventional safety；
- planner score only；
- soft burden only；
- pairmax；
- response-bank-only；
- generic candidate-only；
- no natural branch / no OPR。

只有 full RIOT 在 CR/EP 非劣且 FSR/HBCR 显著优于核心 ablation 后，才进入 Phase D。

### Phase D：5000 场景 × 3 seeds

- seeds 建议 2026/2027/2028；
- paired scenario bootstrap 95% CI；
- collision/fallback 用 paired binary test 或 bootstrap；
- EP/CBS/OPR/HBCR 用 paired bootstrap/permutation；
- 报告总体与 merge、unprotected turn、lane change、dense following 等子集；
- 单独报告 stress set 的 Accept NCF / Accept False-Safe / witness precision-recall。

---

## 10. 推荐执行命令

### 10.1 首次运行：完整 v12 pipeline

从原先效果最好的 v10/v11 planner checkpoint 初始化。不要从零训练，除非先证明 natural basis
可以从零通过 gate。

```bash
cd /path/to/COWP_v12_RIOT_fixed

export COWP_ROOT=/data0/senzeyu2/dataset/COWP/formal
export INIT_CKPT=outputs/cowp_v10_gct_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt

OUT_ROOT=outputs/cowp_v12_riot_probe100_seed2026 \
RAW_TRAIN_CACHE="$COWP_ROOT/tensor_cache_train_waymax" \
RAW_VAL_CACHE="$COWP_ROOT/tensor_cache_val_waymax" \
TRAIN_CACHE="$COWP_ROOT/tensor_cache_train_waymax_transport_v9" \
VAL_CACHE="$COWP_ROOT/tensor_cache_val_waymax_transport_v9" \
INIT_CKPT="$INIT_CKPT" \
RUN_AUGMENT=0 \
RUN_DIAGNOSE=1 \
RUN_NATURAL=1 \
RUN_TRANSPORT=1 \
RUN_PLANNER=1 \
RUN_OFFLINE=1 \
RUN_PROBE=1 \
RUN_FULL=0 \
PROBE_SCENARIOS=100 \
TRAIN_SEED=2026 \
bash run_cowp_v12_dual_gpu.sh
```

运行脚本会按顺序：

1. natural repair；
2. natural hard gate；
3. root-indexed transport；
4. planner；
5. BCOT budget sweep；
6. BCOT calibration；
7. mechanism gate；
8. 只有 gate 通过才运行 100-scene Waymax probe。


外部提供自然基座 checkpoint 时，必须同时设置其训练历史：

```bash
RUN_NATURAL=0 \
NATURAL_CKPT=/path/to/cowp_natural_best.pt \
NATURAL_HISTORY=/path/to/history_natural.json \
bash run_cowp_v12_dual_gpu.sh
```

Waymax probe/full 现在会读取 `mechanism_verification.json` 并要求 `pass=true`，因此不能通过 `RUN_OFFLINE=0` 绕过机制 gate。

### 10.2 机制 gate 通过后跑 1000 场景

```bash
OUT_ROOT=outputs/cowp_v12_riot_probe100_seed2026 \
RUN_AUGMENT=0 \
RUN_DIAGNOSE=0 \
RUN_NATURAL=0 \
RUN_TRANSPORT=0 \
RUN_PLANNER=0 \
RUN_OFFLINE=0 \
RUN_PROBE=0 \
RUN_FULL=1 \
FULL_SCENARIOS=1000 \
bash run_cowp_v12_dual_gpu.sh
```

### 10.3 论文级 3 seeds

每个 seed 使用独立 `OUT_ROOT`，不要共享 optimizer/checkpoint：

```bash
for SEED in 2026 2027 2028; do
  OUT_ROOT="outputs/cowp_v12_riot_seed${SEED}" \
  TRAIN_SEED="$SEED" \
  INIT_CKPT="outputs/cowp_v10_gct_probe100_seed2026/checkpoints/planner/cowp_planner_best.pt" \
  RUN_AUGMENT=0 RUN_DIAGNOSE=0 \
  RUN_NATURAL=1 RUN_TRANSPORT=1 RUN_PLANNER=1 RUN_OFFLINE=1 \
  RUN_PROBE=1 RUN_FULL=1 PROBE_SCENARIOS=100 FULL_SCENARIOS=5000 \
  bash run_cowp_v12_dual_gpu.sh
done
```

在真正跑 5000×3 前，应先用 1000 场景确认 full RIOT 优于关键 ablation，否则应停止并继续定位。

---

## 11. 如何根据下一次日志继续判断

### natural gate 失败

不要放宽 gate 后继续。先看：

- branch source 分类是否塌缩；
- observed/neutral/priority 哪个分支 minADE 最大；
- v10 checkpoint 的 natural decoder 是否与 v12 shape 兼容并被成功加载；
- cache 中 future trajectory 单位、坐标系和 valid mask 是否一致；
- `sigma_traj_m=15` 是否过宽导致分支身份模糊。

若只有 priority branch 失败，可对 priority branch 做独立 decoder/残差修补；若所有分支都几十米，
优先排查坐标/加载，而不是增加 epoch。

### pair AUPRC 高、BCOT AUPRC 低

说明 pair conflict 可学，但 candidate aggregation/calibration 仍错。检查：

- direct root recovery recall/precision；
- unrecovered mass 的 class-conditional 分布；
- critical-agent priority weight 是否将大量普通 agent 放大；
- tail-risk temperature 是否过小；
- candidate budget label coverage 与 NCF/FS 比例；
- candidate risk 是否单调对应 false-safe label。

### BCOT AUPRC 高、NCF recall 低

说明分类好但 operating point/候选分布有问题。检查：

- candidate generator 是否缺少足够的低负担可行候选；
- conventional-safe 与 NCF 候选交集；
- OPR alpha 是否过严；
- fallback 之前是否有 near-feasible candidate 可通过轻微速度/时序修复；
- 是否需要加入“certificate-guided local candidate refinement”，而不是继续放宽 budget。

### 离线 gate 通过、Waymax CR 高

这时问题不再是 RIOT 核心分类，而是执行与闭环分布：

- absolute_xy_yaw action 的连续性；
- 每步重规划造成的候选跳变；
- route/path matching；
- kinematics infeasibility；
- logged replay 对 ego 干预不反应导致的 distribution mismatch；
- fallback 连续控制是否稳定。

优先加入 trajectory stitching、hysteresis 和短时承诺，而不是改变 false-safe label。

---

## 12. 论文需要同步修改的内容

若 v12 成为最终算法，论文必须从 v11 的 pairwise hard rejection 公式同步到 RIOT/BCOT：

1. 明确定义 root-indexed transport event；
2. 定义 candidate BCOT risk：平均选项缺失、tail deficit、OPR shortfall 与 priority veto；
3. pair witness threshold 与 candidate budget 分开；
4. 说明 response bank 是辅助解释头，不是主可行性判据；
5. 将 generic candidate classifier 明确列为 ablation，不能作为核心方法；
6. 将 non-ego policy 如实写为 logged replay；若增加 IDM/reactive robustness，再单独报告；
7. 将 attached candidate outcomes 描述为 sparse auxiliary supervision，不称为在线闭环；
8. 主表必须报告 confidence intervals、scenario count、seed、exact simulator config；
9. 增加 oracle natural-root / oracle response-root upper bound，证明性能上限与剩余瓶颈；
10. 论文 novelty 应表述为：
   - 从“预测他车会不会让”转为“自然行动空间是否被候选运输到低负担安全响应集”；
   - 从 courtesy score 转为 root-indexed feasibility certificate；
   - 从 worst-pair veto 转为 option-mass budget + protected-priority veto。

---

## 13. 最终判断

- **核心 idea 有研究价值且尚未被当前结果否定**；
- v11 的主要失败是核心机制未被正确监督、根恢复结构不匹配、自然选项漂移与阈值混用；
- 相对几何、pair witness、OPR/HBCR 方向和 BCOT 排序信号应保留；
- v12-RIOT 是必要的部分重构，不是为了追分而增加无关模块；
- 下一次结果能否达到 CCF-A，取决于它是否同时解决四个缺口：
  **BCOT AUPRC、NCF recall、fallback、EP 非劣**；
- 只有在 1000 场景 development 与 5000×3 paired closed-loop 中满足统计显著性后，
  才能使用“闭环 SOTA”或“CCF-A ready”表述。
