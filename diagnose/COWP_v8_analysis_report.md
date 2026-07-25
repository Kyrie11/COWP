# COWP 闭环 SOTA 诊断与 COWP-ST v8 优化报告

## 0. 结论先行

当前 v7 **不能支撑“论文核心机制已经在闭环中生效”**，也不能据此声称闭环 SOTA。它取得的主要正向结果是：100 场景 Waymax 中 Collision 从 conventional 的 0.42 降到 0.33，CR 从 0.44 降到 0.37，EP 从 0.506 上升到 0.735；但与此同时 Kinematics 从 0.11 恶化到 0.17，Offroad 从 0.03 恶化到 0.06，Wrong-way 从 0.04 恶化到 0.08，Off-route 从 0.0103 恶化到 0.0722。更关键的是，在线 PredFSR_episode=1.0、PredHBCR_episode=1.0、PredOPR_min_episode=0.0947，说明模型认为几乎所有闭环轨迹都在压缩他车选项，却仍然选择并执行这些轨迹。

v7 的闭环提升更可能来自以下机制的组合，而不是论文所主张的 pairwise coercion witness：

1. 更激进的候选前沿与 progress guard；
2. candidate-level certificate 的排序；
3. physical outcome/rule/action shield；
4. execution-aligned controller projection；
5. conventional fallback。

最根本的原因不是 witness threshold 取值不理想，而是**论文中的集合证书没有被实现成决策路径上的显式计算对象**：planner 阶段原先不解码 ego-conditioned response bank，natural alternatives 也没有以 mode-level latent 被 candidate 条件化；pairwise witness/OPR/burden 主要由一个 embedding 回归头直接预测。于是“自然选项集合 → 候选条件下保留/冲突 → 安全响应集合 → 最小负担 → witness”这条因果链被一个候选/成对代理分类器替代。

本次代码改动将主方法升级为 **COWP-ST（Causal Same-Root Option-Set Transport Certificate）**：显式预测每个 natural mode 在 ego candidate 下是否冲突、是否仍被保留为低负担安全选项，并与 candidate-conditioned response bank 一起计算 OPR、最低安全负担和单调的 coercion witness。候选级 learned head 只允许做小幅有界校准，不能覆盖解析集合证书。

这能让代码真正服务论文 setting，并形成比“exact top-k / Pareto 调权重”更强的算法 novelty。但由于本环境没有 WOMD、checkpoint 和双 GPU，**我只能验证代码、单元测试与机制连通性，不能承诺真实闭环结果已经达到 SOTA**。

---

## 1. 我理解的论文核心目标

论文的目标不是一般意义上的“更礼貌”或“社会成本更低”，而是定义一个新的交互规划可行性缺陷：

- ego 轨迹自身 collision-free；
- 但只有在某个他车硬刹、突让、放弃优先权或放弃合法 gap 时才成立；
- 因而该轨迹是 **false-safe**。

主方法必须满足以下硬逻辑：

1. 对每个关键他车构建非受迫自然选项集合 \(\mathcal A_i^{nat}\)；
2. 对每个 ego candidate 构建安全响应集合 \(\mathcal R_i^{safe}(\tau_e^k)\)；
3. 安全响应中至少存在低负担响应；
4. 自然选项中仍有足够概率质量保留为低负担安全响应，即 OPR 不低于阈值；
5. 当自然低负担行为被 ego candidate 变成不安全，而最低安全响应又超过负担阈值时，形成 pairwise coercion witness；
6. witness/OPR/最低负担首先决定可行性，ego utility 只能在可行集合内排序。

因此，论文的最强论点应是：

> COWP 不只是预测“他车会不会让”，而是估计同一根场景下 ego intervention 如何运输/压缩他车的自然选项集合，并以集合保留证书拒绝 safety-by-coercion。

---

## 2. 数据集与训练/评估 pipeline 判断

### 2.1 当前数据构建逻辑是合理的

数据由两类 WOMD 输入共同构建：

- `uncompressed/scenario/*.tfrecord*`：用于场景索引、自然替代、响应、witness 等伪标签构建；
- `uncompressed/tf_example/*.tfrecord*`：用于构建模型 tensor cache 和 Waymax-ready 状态。

随后对 train/val tensor cache 中每个场景的候选做 Waymax replay，并把 collision/offroad 等物理 outcome 附回 cache。这种设计允许语义证书和物理执行结果使用不同监督来源，是正确方向。

现有 cache 已包含本次 v8 所需的聚合监督：

- natural trajectory / mode weight / source / priority；
- response validity / safety / low-burden / burden / source；
- witness、OPR、最低负担、natural conflict mass；
- source-wise natural/conflict/low-safe mass；
- 部分 Waymax candidate outcomes。

所以 v8 第一轮可以不重建全量数据。后续为了投稿质量，建议增加 per-natural-mode transport labels，见第 9 节。

### 2.2 offline teacher 与 online causal student 要明确区分

伪标签阶段可以使用 logged future 做离线 counterfactual teacher，因为它不作为在线输入；但论文主方法的在线 planner 不能读取 future log。v7 默认改成 constant-velocity other-future source 是必要修复。

论文中应明确写成：

- privileged future 只参与离线 label generation / oracle upper bound；
- 主方法在线输入严格由 history、current state、map、candidate 和 causal response model 构成；
- `logged_oracle` 单独作为 upper-bound ablation，不进入主表。

### 2.3 Waymax outcome 覆盖不足

v7 offline 选中候选的 Waymax outcome coverage 只有约 0.286。物理 outcome head 因而主要在稀疏候选上监督，难以可靠覆盖在线候选分布。后续需要：

- 将 replay 从每场景 balanced 12 candidates 扩大为“随机 + 高风险 + 高进展 + 语义边界”的 hard-negative set；
- 对在线常被选中的 macro intent 做定向 replay；
- 至少让训练中会进入 selector 前沿的候选 outcome coverage 达到 70% 以上。

`cache_sufficiency_full.json` 没有出现在本轮实际挂载文件中，因此本报告没有重新核验它；但用户已说明其数量是旧快照，且当前问题并不依赖该数量差异。

---

## 3. v7 实验结果的证据链

### 3.1 Offline 5013 场景

| 指标 | Conventional | COWP v7 | 判断 |
|---|---:|---:|---|
| EP | 0.3860 | 0.4100 | 小幅提高 |
| FSR | 0.6628 | 0.6624 | 基本不变 |
| CBS | 0.9325 | 0.9197 | 小幅改善 |
| OPR | 0.7415 | 0.7454 | 小幅改善 |
| HBCR | 0.3976 | 0.3832 | 小幅改善 |
| SelectedFalseSafeRate | 0.5937 | 0.5839 | 仅下降 0.0098 |
| LearnedAcceptedCandidateRate | 0.5489 | 0.0612 | 过度拒绝 |
| LearnedAcceptNCFRecall | 1.0000 | 0.1089 | 核心失败 |
| SelectedWaymaxUnsafeRate | 0.0965 | 0.1165 | 变差 |

candidate certificate 的 FalseSafe AUPRC=0.8135、NCF AUPRC=0.5616、risk ranking accuracy=0.7890，说明候选级代理证书具备一定排序能力；但 pairwise witness AUPRC 只有 0.3609，远低于候选级 certificate。

### 3.2 threshold sweep 证明 witness 没接入选择

v7 在 9 个 witness thresholds 上：

- witness recall/precision/WLA 会变化；
- 但 EP、FSR、CBS、OPR、HBCR、accepted rate、NCF recall、SelectedFalseSafeRate 完全不变。

本次新增的 verifier 对 v7 输出得到：

```json
{
  "threshold_points": 9,
  "unique_selection_points": 1,
  "threshold_connected_to_selection": false,
  "learned_accept_ncf_recall": 0.10891195959508752,
  "pass": false
}
```

这不是“校准还不够好”，而是**阈值没有进入最终候选选择的有效控制路径**。

### 3.3 Waymax 100 场景

| 指标 | Conventional | COWP v7 | 变化 |
|---|---:|---:|---:|
| CR | 0.44 | 0.37 | -0.07 |
| Collision | 0.42 | 0.33 | -0.09 |
| EP | 0.5062 | 0.7350 | +0.2288 |
| Kinematics | 0.11 | 0.17 | +0.06，变差 |
| Offroad | 0.03 | 0.06 | +0.03，变差 |
| Wrong-way | 0.04 | 0.08 | +0.04，变差 |
| Off-route | 0.0103 | 0.0722 | +0.0619，显著变差 |

这组结果不能解释成“已经更安全且更高效”，因为 EP 的大幅提升伴随明显道路/方向/动力学违规。原实现使用候选终点相对起点的欧氏位移表示 progress，横向偏离、错向甚至大角度偏航都可能提高该值，容易形成 reward hacking。

同时：

- PredFSR_episode=1.0；
- PredHBCR_episode=1.0；
- PredOPR_min_episode=0.0947；
- accepted candidates 均值约 1.83；
- fallback episode rate=0.27。

这说明模型在线认为语义证书普遍失败，但 selector 仍靠其他分支选择并执行轨迹。因此核心 idea 没有闭环成立。

---

## 4. 上一轮大模型修改的逐项判定

### 4.1 明确起效

1. **语义标签与物理 outcome 分离**  
   避免 collision-free false-safe 候选同时被当作 NCF 正例，候选 certificate AUPRC/ranking 恢复，修改有效。

2. **梯度隔离和 planner backbone 冻结**  
   防止巨大 planner/certificate loss 反向污染 witness representation，结构上有效。

3. **统一离线/在线 selector 与 action mode**  
   消除两套选择器和 `delta_xy_yaw` / `absolute_xy_yaw` 不一致，工程上必要。

4. **execution-aligned action risk**  
   v6 Kinematics=0.23，v7 降到 0.17，说明 controller projection alignment 有作用；但仍未追平 conventional=0.11，需要增强。

5. **取消 logged future 作为主在线输入**  
   恢复因果实验有效性，必须保留。

6. **early stopping / best checkpoint**  
   v7 epoch 0 后几乎无改善并在 epoch 7 早停，避免继续无效训练，修改有效。

7. **utility/progress guard 的动机正确**  
   避免 `universal_ncf` 几乎停车的退化；但原 progress 定义需要修正。

### 4.2 只部分起效

1. **structured residual certificate**  
   候选级排序有效，但原解析项仍由 proxy witness/OPR/burden 组成，并不等同于显式集合证书；且 residual 权重过大，可能覆盖解析项。

2. **epsilon-Pareto frontier**  
   能改变行为，但把 semantic、physical、utility 放在同一个 Pareto 支配关系中，会允许 utility 或 physical 维度“挽救”语义上 coercive 的候选，和论文 hard-first 定义不完全一致。应只作为 ablation，不作为主方法。

3. **threshold sweep / calibration**  
   评估工具本身有用，但 v7 selector 后续覆盖了 semantic gate，导致校准阈值不改变选择。

### 4.3 没有解决

1. **pairwise witness 没有真实参与决策**；
2. **ego-conditioned safe response set 在 planner 阶段没有显式计算**；
3. **natural mode 没有 candidate-conditioned option transport**；
4. **在线 semantic domain gap 极大**；
5. **wrong-way/off-route/kinematics 未进入充分强的 physical feasibility shield**；
6. **强基线、统计显著性和机制消融仍不足**；
7. v7 `compare_probe` 调用参数错误，日志中比较脚本直接 argparse 失败，因此没有生成可靠 probe delta 报告。

---

## 5. 根因代码分析

### 根因 A：论文的“集合”被 proxy head 替代

原 planner forward 只在 `stage=response/all` 时运行 response decoder，planner 阶段未构建 response set。Natural decoder 也只向 witness 暴露一个 agent-level latent。Witness decoder 从 `z_agent/z_candidate/z_graph/natural_latent` 直接回归：

- witness existence；
- OPR；
- burden；
- conflict intensity。

这使得模型可以拟合聚合标签，却不需要学会：哪一个 natural option 被 candidate 冲突、哪一个 option 被保留、哪一个 response 是低负担安全响应。结果是论文声称的可解释证书和代码实际决策机制不一致。

### 根因 B：hard semantic gate 被后续 frontier base 覆盖

v7 先计算 `primary_bad / option_bad / severe_bad`，但之后 frontier base 重新由 `cand_valid & conventional & outcome` 构造，语义拒绝条件没有被保留。threshold 当然只改变 witness 统计，不改变 selection。

### 根因 C：candidate residual 覆盖解析证书

v7 配置中解析 structured logit 权重约 0.35，learned residual scale 约 1.0。即便解析证书包含 witness，它也可能被候选分类器覆盖。论文最终变成 generic candidate risk classifier，而不是机制证书。

### 根因 D：progress 指标可被错误轨迹利用

候选短期 progress 使用欧氏距离；在线标准 EP 也可能在明显 off-route/wrong-way 时上升。需要同时：

- selector 内使用 route/heading-aware signed progress；
- 论文主表报告 route progression，而不是只报告位移；
- 对 off-route/wrong-way 设硬 shield，而不是作为小权重成本。

### 根因 E：offline/online response distribution 不一致

离线伪标签有丰富 natural/response primitives 和 teacher future；在线只用 CV prior + learned embedding。缺少显式 set transport 后，模型只能依赖训练分布中的候选相关模式，在线候选一偏移就出现 PredFSR/HBCR 全 1。

---

## 6. 本次实现的 COWP-ST v8

### 6.1 Causal Same-Root Option-Set Transport

对 critical agent \(i\)、ego candidate \(k\)、natural mode \(m\)，新增两个 mode-level 概率：

- \(p^{conf}_{ikm}\)：该自然选项在 candidate 下发生冲突；
- \(p^{ret}_{ikm}\)：该选项仍能被保留为可接受的低负担选项。

自然模式权重为 \(\pi_{im}\)，则：

\[
O_{ik}=\sum_m \pi_{im}\,p^{ret}_{ikm}(1-p^{conf}_{ikm}),
\]

\[
M^{conf}_{ik}=\sum_m \pi_{im}\,p^{conf}_{ikm}.
\]

同时 compact response decoder 对每个 response slot 预测：

- valid；
- safe；
- low-burden；
- response mixture weight；
- burden；
- source branch。

由 response bank 计算：

- low-burden safe response existence；
- soft minimum safe burden；
- source-wise response statistics。

最终解析 witness 是一个单调组合：只有存在足够 natural conflict mass，且出现 burden excess / option collapse / no-low-safe-response 中至少一种时才升高。learned calibration 使用 `tanh` 限制在小幅残差范围内，不能取代集合逻辑。

### 6.2 决策层恢复 hard-first

主配置 `label_cowp_v8.yaml` 使用：

1. candidate validity；
2. conventional + execution-aligned physical feasibility；
3. high-confidence witness veto；
4. OPR veto；
5. severe witness + option collapse veto；
6. 在保留下来的 NCF 集合中按 semantic risk 形成 bounded exact-top-k；
7. 只在 semantic risk budget 内用 ego score、progress、action/rule/outcome risk 做 tie-break。

Pareto 版本保留在 `label_cowp_v8_pareto_ablation.yaml`，只用于验证“hard semantic hierarchy 是否优于三轴 Pareto”。

### 6.3 修复 progress reward hacking

候选 progress 改为沿候选初始 heading 的有符号纵向位移，并裁剪负值：

\[
p_k=\max(0,(x_T-x_0)^\top[\cos\psi_0,\sin\psi_0]).
\]

这不是完美的 route arc-length，但比欧氏距离更不容易奖励横移、错向和大偏航。最终论文实验仍应使用 route polyline 的弧长投影。

### 6.4 保持 representation 与新机制解耦

v8 从 v7 best checkpoint 初始化：

- 冻结 graph、candidate encoder、natural decoder、proxy witness decoder；
- 训练 set-transport head；
- 训练 compact response heads；
- 训练 candidate certificate/outcome/priority/planner heads；
- planner loss 不反向污染旧 witness stack。

Response decoder 同时接受逐响应 safe/low/burden/valid/source 辅助监督，防止只靠聚合 set loss 产生 slot collapse。

### 6.5 新增机制验证器

`cowp.scripts.25_verify_mechanism_effect` 会在跑昂贵 Waymax probe 前检查：

- threshold sweep 是否至少产生两个不同 selection operating points；
- LearnedAcceptNCFRecall 是否达到最低要求。

失败则脚本直接退出，避免继续消耗 GPU 得到一个“闭环看起来不错但核心机制仍未连接”的结果。

---

## 7. 代码修改清单

核心新增/修改：

- `cowp/models/set_transport_head.py`：显式 option-set transport certificate；
- `cowp/models/natural_decoder.py`：暴露 per-mode latent；
- `cowp/models/response_decoder.py`：增加 valid/mode/source heads，支持 compact decode；
- `cowp/models/cowp_model.py`：planner/witness 阶段运行 compact response bank，并用 set certificate 替换决策 proxy；
- `cowp/models/losses.py`：set-transport 聚合监督、response 额外监督；
- `cowp/scripts/03_train.py`：v8 loss、checkpoint composite、冻结策略；
- `cowp/planning/set_preservation_selector.py`：hard-first bounded frontier；
- `cowp/waymax_eval/rollout.py`：offline hard gate 与 signed progress；
- `cowp/waymax_eval/policy_wrapper.py`：online hard gate 与 signed progress；
- `configs/label_cowp_v8.yaml`：主方法；
- `configs/label_cowp_v8_pareto_ablation.yaml`：Pareto 消融；
- `configs/train_cowp_v8.yaml`：训练权重；
- `cowp/scripts/24_summarize_planner_delta.py`：修复 v7 比较脚本调用问题；
- `cowp/scripts/25_verify_mechanism_effect.py`：机制连通性 gate；
- `run_cowp_v8_dual_gpu.sh`：双卡训练、并行 probe、并行 full shards；
- `tests/test_set_transport_certificate.py`：compact decode 与 burden-monotonicity 测试。

本地验证：

- `pytest -q`：**52 passed**；
- Python compile：通过；
- `bash -n run_cowp_v8_dual_gpu.sh`：通过；
- 新 head synthetic forward：通过；
- v7 机制 verifier：按预期失败，证明工具能捕获 threshold-selection 断开。

---

## 8. 下一轮实验判定标准

### 8.1 Offline 必须先过机制 gate

建议最低门槛：

- `unique_selection_points >= 2`；
- Witness AUPRC ≥ 0.50，理想 ≥ 0.60；
- LearnedAcceptNCFRecall ≥ 0.25，理想 ≥ 0.35；
- LearnedAcceptedCandidateRate ≥ 0.12，避免再次过度拒绝；
- SelectedFalseSafeRate 比 conventional 至少下降 0.03；
- OPR 至少提升 0.02；
- HBCR 至少下降 0.03；
- fallback ≤ 0.25；
- selected Waymax unsafe 不得高于 conventional。

### 8.2 Waymax 100 场景 probe

最低门槛：

- Collision/CR 仍优于 conventional；
- Kinematics ≤ conventional + 0.02；
- Offroad、Wrong-way、Off-route 各自 ≤ conventional + 0.01；
- EP 提升不能依赖 wrong-way/off-route；
- PredFSR_episode 和 PredHBCR_episode 必须显著低于 1，建议 <0.60；
- PredOPR_min_episode 建议 >0.25；
- fallback episode rate ≤0.25；
- main hard-first 配置应优于 Pareto ablation 的 FSR/HBCR，并保持相近 CR/EP。

若 offline gate 不过，不要跑 1000/5000。若 100 probe 道路约束不过，也不要扩大规模。

---

## 9. 达到 CCF-A 投稿强度仍需做的工作

### 9.1 数据标签升级：per-mode transport supervision

v8 第一版利用已有 aggregate labels，能建立显式集合计算图，但 mode-level conflict/retain 仍存在可辨识性不足。最强版本应在 cache 中增加：

- `natural_mode_unsafe_under_candidate[K,A,M]`；
- `natural_mode_retained_low_safe[K,A,M]`；
- `natural_mode_conflict_type[K,A,M]`；
- `response_is_min_burden[K,A,R]`；
- per-source calibrated mass；
- scene/candidate/agent group id，用于 same-root contrastive loss。

这样可以直接训练“同一自然选项如何随 ego intervention 改变”，形成非常清晰的新颖性。

### 9.2 加入 same-root counterfactual ranking / invariance

同一 scene root 的候选应满足：更激进、更早占用冲突区的 candidate 不应比其邻近温和 intervention 获得更高 OPR 或更低最低负担，除非交通拓扑/优先权确实改变。建议加入：

- intervention monotonic ranking；
- neighboring-candidate Lipschitz regularization；
- source-preservation contrastive loss；
- pairwise burden attribution consistency。

这会比单纯增加网络规模更有 novelty。

### 9.3 不确定性：conformal lower-bound certificate

主论文可把 OPR 由点估计升级为保守下界：

\[
\underline O_{ik}=\widehat O_{ik}-q_{1-\delta}(\mathcal C),
\]

并用 burden 上界 \(\overline C_{ik}\) 做可行性判定。报告 risk-coverage、ECE、selective FSR，能明显增强顶会可信度。

### 9.4 物理 shield 必须覆盖全部闭环失败类型

增加或显式计算：

- kinematics infeasibility；
- wrong-way；
- off-route；
- offroad；
- collision；
- controller projection error。

这些都应是 hard feasibility，不应只作为小权重 cost。特别是 route arc-length 和 lane direction 必须来自 map topology，而不是 endpoint displacement。

### 9.5 在线 hard-negative mining

从 v8 1000 场景运行中收集：

- 被选中但 PredFSR/HBCR 高；
- collision/offroad/wrong-way/off-route；
- fallback；
- candidate certificate 与 set certificate 分歧；
- CV prior 与 logged oracle 分歧。

把这些场景按 macro intent/拓扑/速度/关键 agent 类型分层 replay，做一轮 closed-loop hard-negative fine-tune。这是修复在线 domain gap 的根本途径。

### 9.6 必须补齐的实验

1. 真实强基线：至少 GameFormer/DTPP 类交互规划器、rule lattice、planner-score-only、soft burden；
2. 主方法与以下消融：
   - w/o neutral branch；
   - w/o priority branch；
   - w/o option transport；
   - w/o response existence；
   - soft burden only；
   - proxy witness only；
   - Pareto frontier；
   - logged oracle vs CV vs learned causal response；
3. 3 个训练 seed，均值、95% CI、paired bootstrap；
4. 1000 场景开发集，5000 场景最终一次性测试；
5. merge/lane-change/intersection/following 分层结果；
6. 机制可视化：被拒候选、natural mode、冲突 mode、最低负担 response、被保留选项质量；
7. 人工审阅一小批 witness，验证伪标签不是规则自证。

---

## 10. 论文表述应同步调整

建议把主算法叙述从“graph predicts witness + rollout validates”改为更精确的三层结构：

1. **Root Natural Option Set**：学习同一根场景的非受迫选项分布；
2. **Candidate-Conditioned Set Transport**：估计每个 natural option 在 ego intervention 下的 conflict/retention；
3. **Response-Existence and Burden Certificate**：用 ego-conditioned response bank 证明是否仍存在低负担安全响应，并得到 OPR lower bound 与 burden upper bound。

候选 certificate 只能称为 residual calibration / auxiliary ranking，不能作为核心贡献。Exact top-k 和 Pareto 都只能是 selector implementation/ablation。核心 novelty 应落在“same-root causal option-set transport + hard non-coercive feasibility certificate”。

---

## 11. 风险说明

- v8 是结构性修复，不是结果保证；真实结果取决于旧 checkpoint 中 natural/response representation 的质量。
- aggregate set labels 可能不足以唯一确定 mode-level transport；若 witness AUPRC 或 OPR 仍不提升，应直接构建 per-mode labels，而不是继续调阈值。
- 100 场景统计波动很大，只能做机制 probe，不能用于 SOTA claim。
- WOMD/Waymax 与 nuPlan 的 planner baselines 不可直接无条件横向比较；需要统一输入、候选生成、闭环 agent policy 和 metric protocol。
- 在道路约束明显差于 conventional 时，即便 collision/EP 更好，也不应宣称总体 SOTA。
