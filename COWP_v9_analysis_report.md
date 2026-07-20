# COWP v8 实验诊断与 COWP-RCT v9 优化报告

## 1. 结论

当前 v8 **不能证明论文核心机制在真实闭环中成立，也不能支持闭环 SOTA 或 CCF-A 投稿结论**。这不是因为 v8 完全无效，而是因为它只解决了上一版的一部分结构性问题：

- 已解决：witness threshold 与最终 selection 的断链。v8 的 9 个阈值产生了 9 个不同 selection operating points。
- 部分解决：显式 Set-Transport 结构、hard-first selector、离线/在线 selector 复用、route-aware progress、execution-aligned action shield、语义证书与物理 outcome 分离。
- 未解决：逐 response 监督在实际 planner 训练中没有执行；per-natural-mode 分解没有显式标签；response bank 与 natural option root 没有关联；不确定性和自然多样性混淆；hard veto 在多个 critical agents 上累积假阳性；没有得到任何 v8 Waymax probe 结果。

v8 offline gate 的确按设计停止了 Waymax probe。实际 gate 结果为：

```json
{
  "threshold_points": 9,
  "unique_selection_points": 9,
  "threshold_connected_to_selection": true,
  "learned_accept_ncf_recall": 0.07271589216723144,
  "witness_auprc": 0.4299954898907559,
  "selected_false_safe_rate": 0.37841611809295833,
  "pass": false
}
```

这说明 v8 不再是“机制不参与决策”，而变成了“机制参与决策但证书过窄、召回严重塌缩”。COWP 默认阈值下只接受约 3.11% 的候选，NCF recall 只有 7.27%，fallback 达 35.85%，EP 从 conventional 的 0.3891 降到 0.2218。FSR、OPR、HBCR 的改善主要来自大面积拒绝，而不是准确区分 non-coercive 与 coercive candidates。

本报告给出的 v9 不再继续调 selector 的单一阈值，而是修复训练数据流、标签可辨识性和 same-root 结构，并把论文写出的“高置信 witness 硬拒绝、不确定 witness 软惩罚”真正落实到 offline/online selector。

---

## 2. 论文核心 idea 与必须成立的因果链

论文的核心不是 courtesy cost，也不是普通 collision-risk classifier。核心命题是：

> 一个 ego candidate 即使 collision-free，只要它使关键他车的自然低负担行为变得不安全，并且只有高负担让行响应能够消解冲突，那么它就是 false-safe，应从可行集合中排除。

因此算法必须完整实现并验证：

```text
natural alternatives
    -> candidate-conditioned natural-mode conflict
    -> ego-conditioned safe response set
    -> low-burden response / option preservation
    -> pairwise coercion witness
    -> high-confidence hard rejection
    -> utility optimization only within certified feasible candidates
```

论文预期的核心指标不是单独降低 collision，而是同时满足：

- CR 不恶化；
- EP 保持接近强基线；
- FSR、CBS、HBCR 明显下降；
- OPR 明显上升；
- witness AUPRC/WLA/MTA 有可信质量；
- non-coercive candidate acceptance 不能塌缩。

只要结果是通过“拒绝绝大多数候选”获得低 FSR，就不能证明核心 idea 成立。

---

## 3. 数据集是否是当前主要瓶颈

### 3.1 核心语义标签足够完整

`cache_sufficiency_full.json` 全量扫描显示：

- train 14,640 scenes；
- val 5,013 scenes；
- natural/response/witness/planner 标签覆盖接近 100%；
- train/val scenario ID overlap 为 0；
- train/val 候选数量、Waymax unsafe rate 和 replay coverage 分布一致。

所以当前不需要为了 v9 的 natural/response/witness/planner 核心训练重新构建整个 WOMD 数据集。

### 3.2 Waymax outcome 只适合作为辅助监督

valid candidates 中只有约：

- train 23.44%；
- val 23.70%

拥有有效 Waymax candidate rollout outcome。它足以作为 collision/offroad auxiliary loss，但不能作为唯一 planner objective，也不能把 learned-offline selected Waymax outcome 当作完整闭环结果。

### 3.3 log-divergence 完全不可用

train/val finite log-divergence coverage 都是 0，现有 cache 中 log-divergence 缺失或退化为常量。因此：

- v9 保持 `outcome_logdiv: 0.0`；
- 不应报告 `SelectedWaymaxMeanLogDivergence`；
- offline mechanism gate 默认 `outcome-risk-penalty=0`，避免稀疏物理 head 掩盖语义机制；
- online 可以使用较小 collision/offroad risk penalty，但必须做 penalty=0 的机制消融。

---

## 4. v8 到底解决了哪些问题

### 4.1 selector 与 witness 已经连接

v7 的 9 个 threshold 只有 1 个 selection point；v8 的 9 个 threshold 有 9 个 selection points。这证明 hard witness/OPR 条件已经真正进入 selector。

### 4.2 structured candidate certificate 有效

v8 offline 中：

- Candidate NCF AUPRC：0.5657；
- Candidate FalseSafe AUPRC：0.8243；
- Candidate risk ranking pair accuracy：约 0.802。

这部分明显强于 pairwise witness AUPRC 0.4300，说明 candidate-level certificate、rule/action/outcome shield 和 ranking head 确实学到了有效信号。

### 4.3 hard-first 可降低 false-safe selection

默认阈值下，SelectedFalseSafeRate：

- conventional safety：0.5933；
- COWP v8：0.3784。

同时 OPR 从 0.7413 提高到 0.7905，HBCR 从 0.3980 降到 0.2320。这说明“先做语义过滤再优化 utility”的方向是对的。

### 4.4 fail-fast gate 是正确修改

v8 没有在 mechanism gate 失败后继续浪费 Waymax 计算。这一实验纪律应继续保留。

---

## 5. v8 没有解决的根本问题

## 5.1 实际 planner 训练从未加载 response 标签

这是当前最直接的代码错误。

v8 的 `03_train.py` 虽然包含 planner/witness 阶段的 response auxiliary loss，但 `cowp/data/dataset.py::_wanted_keys_for_stage()` 只在 `stage in ("response", "all")` 时读取：

- `cowp/response/valid`；
- `cowp/response/is_safe`；
- `cowp/response/is_low_burden`；
- `cowp/response/burden_total`；
- `cowp/response/source`。

planner 阶段 batch 中不存在这些字段，所以 response auxiliary 分支始终跳过。训练日志对此有直接证据：

- 每个 epoch 的 `train/set_transport/response = 0.0`；
- 每个 epoch 的 `val/set_transport/response = 0.0`；
- 不存在任何 `response_aux/*` 指标。

因此上一轮报告中“planner 阶段继续接受逐 response supervision”的描述并没有在实际运行中实现。

## 5.2 per-mode Set-Transport 是不可辨识的

v8 预测每个 natural mode 的：

- conflict probability；
- retain probability；
- uncertainty。

但训练只有 aggregate OPR、aggregate conflict mass、aggregate burden 和 witness scalar。许多完全不同的 per-mode 分解都可以产生相同 aggregate loss，因此网络没有理由学到真实的 option transport。

结果是模型可能学会一个 aggregate candidate score，却无法回答论文真正要求的问题：“哪个 natural option 被 ego candidate 消灭，哪个 option 被保留？”

## 5.3 “same-root” 在 v8 中只是命名，不是结构约束

v8 response bank 只计算全局低负担安全响应是否存在。它没有预测 response 属于哪个 natural root，也没有要求冲突 natural mode 必须由同一 root 的低负担响应恢复。

这样会出现错误证书：

- natural mode A 被 ego candidate 消灭；
- response bank 中存在一个来自完全不同意图 B 的低负担响应；
- 模型仍可能认为 option 已被保留。

这不满足论文的 option preservation 定义。

## 5.4 v8 把自然多样性当成 epistemic uncertainty

丰富的 natural alternatives 是论文希望保留的对象，而 v8 将 natural-set entropy 混入 uncertainty。多模态越丰富，hard certificate 越不容易通过，等于惩罚论文倡导的 option richness。

v9 将 natural entropy 与模型错误不确定性彻底分开。

## 5.5 zero-conflict candidate 仍有背景 witness

v8 的 conflict gate 在 natural conflict mass 为 0 时仍可输出非零值，造成没有自然冲突的 pair 也带有背景 witness probability。多 critical-agent 场景中，任意 pair hard veto 会放大这种假阳性。

## 5.6 hard veto 对所有不确定 pair 一票否决

v8 对任意 protected pair 使用：

```text
witness >= threshold OR OPR < alpha
```

直接拒绝 candidate。若每个场景有 4–6 个 critical agents，即使单 pair 假阳性不高，candidate-level survival 也会指数式下降。这正是 accepted candidate rate 只有 2.3%–4.7% 的原因之一。

论文正文已经写明：高置信 witness 硬拒绝，不确定 witness 软惩罚。v8 实现没有正确区分这两类情况。

## 5.7 没有 v8 Waymax probe，不能判断闭环改善

`cowp_v8.zip` 只有 learned-offline 输出；offline verifier 失败后脚本退出，因此：

- 没有 conventional-vs-COWP 100-scene Waymax delta；
- 没有 kinematics/offroad/wrong-way/off-route 新结果；
- 没有依据判断 v8 是否接近闭环 SOTA。

## 5.8 论文与 evaluator 的 non-ego policy 设置不一致

论文实验部分声称 Waymax 中使用 logged replay、rule reactive、learned reactive 的 mixture，但当前 `04_eval_closed_loop` 的 real Waymax evaluator 只控制 SDC；`configs/eval.yaml` 中的 `non_ego_policy_mixture` 没有任何代码消费。

这意味着当前 online evaluation 实际是 SDC-controlled/logged-background protocol，而不是论文写出的 reactive mixture。v9 增加了 `configs/eval_cowp_v9.yaml`，明确标记当前真实 protocol，防止无意中报告未执行的实验。

要真正证明“他车被迫让行”的闭环因果效应，投稿前仍需实现并验证 reactive non-ego evaluation，或把论文主实验协议改成诚实的 log-replay closed-loop + 独立 counterfactual stress test。

## 5.9 训练标签与核心评价指标存在循环风险

FSR、OPR、HBCR 和 witness pseudo-label 都来自同一 burden/unsafe/response generator。模型可能只是在拟合标签引擎，而不是发现可泛化的 coercion mechanism。

投稿前至少需要一种独立评价：

- 人工核验的 false-safe stress subset；
- 不同规则参数生成的 held-out label engine；
- reactive simulator 中实际他车 deceleration/progress/gap loss；
- 或跨数据集/跨 simulator 的 transfer。

---

## 6. COWP-RCT v9：Root-Coupled Same-Root Option-Set Transport

v9 的核心修改不是放松阈值，而是使证书可辨识、可解释、可校准。

### 6.1 显式 transport 标签

对每个 ego candidate `k`、critical agent `i` 和 natural mode `m`，新增：

```text
cowp/transport/mode_valid[k,i,m]
cowp/transport/mode_conflict[k,i,m]
cowp/transport/mode_retained_low_safe[k,i,m]
```

对每个 ego-conditioned response `r`，新增：

```text
cowp/transport/response_root_index[k,i,r]
cowp/transport/response_is_min_burden[k,i,r]
```

并增加：

```text
cowp/transport/root_recovery_mass[k,i]
```

这些字段可由现有 cache 的 natural/response trajectories 直接增广，不需要重新解析全部 WOMD TFRecord。

### 6.2 unordered natural-mode 对齐

natural decoder 输出是无序集合，不能假设预测 mode `m` 与标签 mode `m` 对应。v9 在监督前用轨迹 ADE 将 GT natural modes 匹配到预测 modes，再施加：

- mode conflict BCE；
- mode retain BCE；
- uncertainty calibration；
- response-root CE。

这避免逐 index 监督破坏 set prediction。

当前实现使用 frozen natural decoder 下的 hard nearest-ADE matching。论文最终版建议增加 soft OT/Sinkhorn matching 消融，以增强 permutation-invariant transport 的理论表达。

### 6.3 compact same-root response bank

ResponseDecoder 在不解码大 trajectory tensor 的情况下输出：

- safe / low-burden / valid；
- burden；
- mixture weight；
- source；
- `root_logits[r,m]`；
- minimum-burden response marker。

只有被分配到同一 natural root 的低负担安全 response 才能恢复该 option。

### 6.4 root-coupled recovery certificate

对 conflict natural mass：

```text
M_conf = sum_m pi_m * p_conf_m
```

对每个 root 的低负担 response existence：

```text
E_root_m = 1 - product_r (1 - p_root(r,m) * p_low_safe_r)
```

conflict-conditioned same-root recovery：

```text
R_root = sum_m pi_m * p_conf_m * E_root_m / M_conf
```

若 `M_conf=0`，则不存在需要恢复的 displaced option，`R_root=1`，不会产生 response-absence witness。

OPR 仍由 natural options 直接计算：

```text
OPR = sum_m pi_m * p_retain_m * (1 - p_conf_m)
```

因此“natural option 本身仍被保留”和“冲突 option 存在同 root 低负担恢复”是两个可解释、可消融的量。

### 6.5 低冲突质量下 witness 精确为零

v9 的 conflict gate 乘以 conflict support，使 natural conflict mass 为零时 analytic witness 精确为零，消除背景 witness。

### 6.6 uncertainty-aware hard-first

v9 保留论文的 feasibility-level rejection，但只对低不确定性的 certified pair 做 hard veto：

```text
confident_pair = uncertainty <= u_hard
hard_bad = priority_claim AND confident_pair AND certified_coercion
```

对不确定 pair：

```text
soft_penalty = witness + lambda_u * uncertainty
```

OPR hard veto 使用 uncertainty upper bound：

```text
OPR_upper = OPR + lambda_opr * uncertainty
```

只有 `OPR_upper < alpha` 才能作为高置信 option collapse。这样不会通过简单放宽 witness threshold 牺牲语义，而是将“证据不足”与“已证明 coercive”分开。

### 6.7 两阶段训练

1. **transport/witness stage**：从 v8 best checkpoint warm start，冻结 graph/candidate/natural/proxy witness backbone，只训练 set transport、response root bank 和相关 calibration。
2. **planner stage**：从 transport best warm start，训练 candidate ranking、physical shield 和 planner head，同时保留较小 response/set-transport consistency loss。

这种训练比直接 joint fine-tuning 更容易把改进归因到核心机制。

---

## 7. v9 修改文件

主要修改：

- `cowp/data/dataset.py`
- `cowp/data/cache_schema.py`
- `cowp/label/witness.py`
- `cowp/label/label_engine.py`
- `cowp/models/response_decoder.py`
- `cowp/models/set_transport_head.py`
- `cowp/models/cowp_model.py`
- `cowp/models/losses.py`
- `cowp/scripts/03_train.py`
- `cowp/scripts/25_verify_mechanism_effect.py`
- `cowp/scripts/26_augment_transport_labels.py`
- `cowp/scripts/27_diagnose_transport_labels.py`
- `cowp/waymax_eval/rollout.py`
- `cowp/waymax_eval/policy_wrapper.py`
- `configs/label_cowp_v9.yaml`
- `configs/label_cowp_v9_pareto_ablation.yaml`
- `configs/train_cowp_v9.yaml`
- `configs/eval_cowp_v9.yaml`
- `run_cowp_v9_dual_gpu.sh`
- `tests/test_v9_transport_supervision.py`

---

## 8. 验证状态

本地已完成：

- Python compileall 通过；
- `bash -n run_cowp_v9_dual_gpu.sh` 通过；
- augmentation CLI、diagnostic CLI、mechanism verifier CLI 可解析；
- synthetic cache augmentation 单进程/多进程 smoke 通过；
- 受影响模块定向测试 17/17 通过；
- 额外回归测试 6/6 通过；
- unordered natural-mode alignment 测试通过。

未完成：

- 本环境没有你的完整 tensor cache、GPU checkpoint 和 Waymax runtime，因此没有运行 v9 训练或真实 Waymax closed-loop；
- 没有据此声称 v9 已达到 SOTA；
- full pytest 在当前打包环境中没有完整跑完，不能声称整个仓库所有测试均通过。

---

## 9. v9 gate 与实验决策

第一阶段必须同时满足：

- threshold sweep 至少 3 个不同 selection points；
- witness AUPRC >= 0.50；
- LearnedAcceptNCFRecall >= 0.25；
- LearnedAcceptedCandidateRate >= 0.08；
- fallback <= 0.30；
- SelectedFalseSafeRate 相比 conventional 至少下降 0.03。

建议更有投稿价值的目标：

- witness AUPRC >= 0.60；
- NCF recall >= 0.40；
- accepted candidate rate >= 0.15；
- fallback <= 0.20；
- EP >= conventional 的 90%；
- SelectedFalseSafeRate 降低 >= 0.10；
- OPR 提高 >= 0.03；
- HBCR 降低 >= 0.08。

若 explicit mode/root losses 明显下降，但 witness AUPRC 仍低，优先检查：

1. conflict/retain 标签噪声与 class imbalance；
2. nearest-ADE root matching 是否需要 soft OT；
3. beta/alpha 对不同 agent type 是否需要 calibration；
4. natural modes 的 recall 是否不足；
5. witness label 本身是否被 aggregate burden rule 主导。

若 offline 通过而 online PredFSR/HBCR 仍异常，才进行 Waymax closed-loop hard-negative mining。不要在 offline mechanism 仍失败时先扩大 selector 或继续跑 5000 场景。

---

## 10. CCF-A/SOTA 投稿前仍必须补齐

即便 v9 1000/5000 场景结果良好，投稿前仍需：

1. 至少 3 个训练随机种子，报告 mean/std 或 bootstrap 95% CI；
2. 真实强基线：IDM/Lattice、GameFormer、DTPP/PlanT 类 planner，且说明是否官方实现或公平重实现；
3. 核心消融：无 counterfactual、无 option preservation、无 same-root recovery、无 hard rejection、无 uncertainty split、soft burden only；
4. 机制隔离：offline/online outcome-risk penalty=0 与 full physical shield 对比；
5. reactive non-ego robustness，或明确修改论文，不再声称已执行 mixture；
6. 人工核验/独立 evaluator，避免 pseudo-label 与评价指标循环；
7. scenario-level paired significance，特别是 collision/FSR/EP trade-off；
8. 失败案例：过保守、natural mode miss、priority ambiguity、VRU 场景；
9. 运行时间、显存、每步 latency 与 candidate 数量消融；
10. 明确主张：若 CR/EP 不优于所有 planner，不要笼统写“闭环 SOTA”，应写“在保持 conventional safety/progress 的同时，non-coercive metrics 达到最佳”。

当前最可信的论文贡献定位是：

> **一种可验证的、root-coupled 的 non-coercive feasibility certificate，补充而不是替代 conventional collision safety。**

这一定位比把 COWP 表述成普通综合分数 planner 更有 novelty，也更容易通过机制实验和消融建立因果证据。
