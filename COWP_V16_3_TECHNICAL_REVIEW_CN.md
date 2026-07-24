# COWP v16.3 论文—算法—代码—实验联合审计与优化报告

> 审计对象：`interactive planning.tex`、`COWP.zip`、`cowp_v16_2_engineering_smoke_v9labels_seed2026_ancdatafix.zip`  
> 结论版本：v16.3，2026-07-24  
> 重要限制：本次可以完成代码静态审计、结果解析、CPU 单元/回归测试和工程修复；无法在当前容器复现服务器上的多 GPU 全量训练，因此所有“有效性”结论严格依据已上传结果，不把修复后的预期当作实验事实。

## 1. 论文真正的核心 idea 与必须守住的主张

论文最有价值的 idea 不是一般的 social planning，也不是再加一个礼貌代价，而是：

**一个 ego 轨迹即使自身不碰撞，也可能只是把冲突解决责任转移给了其他有优先权或正常行驶权的 agent；这种“靠别人急刹、弃权、让出合法间隙才安全”的轨迹应被视为 feasibility failure，而不是较差但仍可接受的 soft-cost 解。**

这可以被概括为 priority-aware non-coercive feasibility：

1. 先生成 ego candidate；
2. 对关键他车构建不受该 ego candidate 压迫时的 natural roots；
3. 在同一个 natural root 下，评估 ego 介入后该 agent 是否还存在低负担、安全、动力学可行的 continuation；
4. 估计自然行为概率质量中有多少仍被保留；
5. 若 ego 只靠对方高负担让行才避免冲突，则产生 coercion witness 并拒绝该 candidate；
6. 在物理安全和 non-coercive feasibility 均满足后再优化 ego utility。

论文要成立，必须证明的不只是“模型能预测 witness”，而是以下完整闭环链条：

- natural basis 真实覆盖了非胁迫行为，而不是复刻日志中的被迫让行；
- root transport 能判断同一自然意图在 ego 干预下的保留、冲突和恢复；
- certificate 与真实/独立构建的 false-safe 标签一致；
- selector 能减少 false-safe / burden transfer，同时不恶化 collision、off-road 和 progress；
- 改善在足够大的闭环 stress set、多个随机种子和置信区间下成立。

## 2. 当前代码和数据 pipeline 的实际状态

### 2.1 数据

代码当前使用 WOMD / Waymax 的 `tf.Example` 数据，历史 1 s、未来 8 s，并复用：

- raw train/val tensor cache；
- `transport_v9` overlay；
- `data_protocol=v9_reuse`。

上传的 causal audit 明确显示：

- 工程因果检查通过；
- non-ego online policy 实际是 `logged_replay`；
- `reactive_mixture_implemented=false`；
- `v15_label_tensors_materialized=false`；
- 因此 `full_v15_label_protocol_pass=false`。

所以论文中“logged replay + learned reactive + rule-based reactive mixture”的表述与当前实现不一致。v16.3 修订稿已改成：logged replay 只支持物理闭环指标和 learned mechanism diagnostics；真正的 burden 因果结论必须另做 reactive-agent protocol 和 human-audited stress set。

### 2.2 实际 pipeline

当前实现可归纳为：

1. shared scene / graph encoder；
2. natural decoder：OBS、ego-neutral、priority-preserving basis，加 dynamics-consistent residual/control；
3. root/set transport：学习 conflict、retain、recovery、low-safe-exist、OPR 等；
4. witness / BCOT certificate；
5. planner candidate scoring；
6. hard physical shield；
7. non-coercive selector、least-violation fallback；
8. offline learned evaluation；
9. Waymax logged-replay probe/full rollout。

这条 pipeline 与论文总 idea 一致，但旧论文中的若干数学定义和实验声称与代码不一致，已在修订稿中调整。

## 3. 上传的 v16.2 结果是否足以验证六项修改

### 3.1 运行性质

这次结果是 engineering smoke：

- natural / transport / planner 各 1 epoch；
- Waymax 仅 20 个场景；
- `ALLOW_QUALITY_GATE_FAILURE=1`；
- `STOP_AFTER_STAGE=probe`；
- `RUN_FULL=0`；
- 脚本按设计在 probe 后停止，而不是异常漏跑 full；
- `eval/QUALITY_GATES_BYPASSED.txt` 存在。

因此它可以验证“流程是否能走通、gate 是否会工作、输出字段是否生成”，不能证明论文效果，更不能证明 SOTA。

### 3.2 六项结论

| 修改 | 当前能否验证 | 结论与依据 |
|---|---:|---|
| 1. v15/v16 decoder | 否 | natural epoch -1 与 epoch 0 的所有验证指标完全相同；best checkpoint 仍是初始化；learned output 与 analytic basis 完全一致，gain=0。 |
| 2. 新 loss | 否 | 主模型没有发生可验证学习，也没有同 seed/同 batch/同初始化的一因素 loss ablation。 |
| 3. OBS residual capacity | 否 | residual L2 与 control smoothness 均为 0；无有效主模型和容量消融，无法归因。 |
| 4. natural gate | **仅工程层面有效** | gate 正确识别 learned basis 没有改善并拒绝了 checkpoint。它证明了门禁能防止错误结论，不证明 natural 算法本身有效。 |
| 5. planner | 否 | 仅 1 epoch，natural basis 无效，20 场结果混合，无法隔离 planner 贡献。 |
| 6. selector | 否 | BCOT 对 pair-max 有部分改善，但相对 Pareto 的 collision/progress 更差；FSR 仍约 0.92、fallback 0.80，样本数过小。 |

## 4. 关键结果与真实含义

### 4.1 Natural decoder

验证集在初始化与训练 1 epoch 后完全不变：

- overall trajectory minADE：8.5399 m；
- branch minADE：9.5111 m；
- OBS minADE：10.5468 m；
- neutral minADE：8.5961 m；
- priority minADE：7.3193 m；
- neutral consistency：38.6482；
- residual L2：0；
- control smoothness：0；
- base deviation：约 34.293 m。

独立 learned-natural diagnostic 又显示 learned 与 analytic trajectory 完全相同，overall/OBS gain 都为 0。因此 decoder、新 loss、OBS capacity 目前没有被训练结果验证。

### 4.2 Transport / witness

1 epoch validation 的关键值：

- candidate-budget coverage：0.5493；
- candidate false-safe rate：0.6716；
- NCF rate：0.3284；
- mode conflict loss：0.3968；
- mode retain loss：0.2354；
- mode recovery loss：0.8316；
- root recovery loss：1.0284；
- witness interval loss：17.2124；
- token loss：1.1486。

离线机制质量：

- CandidateCertificate false-safe AUPRC：0.4082；
- BCOT false-safe AUPRC：0.4310；
- RootTransport low-safe-exist AUPRC：0.5122；
- RootTransport conflict-conditioned AUPRC：0.1247；
- auxiliary conflict-conditioned AUPRC：0.0653；
- witness AUPRC：0.7590；
- BCOT ranking pair accuracy：0.8801。

含义：pair-level witness 有一定可分性，但 root transport 的核心 conflict-conditioned recovery 很弱。模型容易识别“某对 agent 看起来危险”，却还不能可靠回答“哪个自然 root 被 ego 阻断，以及同 root 是否有低负担恢复”。这正是论文新意所在，因此目前瓶颈不在分类头，而在 natural-root identity 与 transport supervision。

### 4.3 Selector / calibration

BCOT calibration 找不到满足全部约束的 operating point，只能返回 `least_violation`：

- accepted candidate rate：0.0925，低于 0.10 gate；
- NCF recall：0.2146，低于 0.30 gate；
- fallback：0.2180；
- selected false-safe：0.4608；
- EP：0.3646；
- OPR：0.7951；
- HBCR：0.2837。

它相对 conventional 的确降低 selected false-safe 和 HBCR、提高 OPR，但代价是 accepted rate 与 NCF recall 大幅下降，EP 也下降。这更像“高拒绝率保守过滤器”，还不是能用于论文主张的有效 selector。

### 4.4 20 场 Waymax probe

Root-transport/BCOT 相对 conventional：

- CR：0.40 → 0.40，无改善；
- EP：0.5992 → 0.7570；
- kinematic infeasibility：0.20 → 0.10；
- wrong-way：0.20 → 0.15；
- fallback：0.30 → 0.80；
- predicted FSR：0.9167。

BCOT 相对 pair-max：

- CR：0.45 → 0.40；
- EP：0.6698 → 0.7570；
- predicted FSR：0.9091 → 0.9167，略差；
- OPR 略差；
- fallback：1.00 → 0.80。

BCOT 相对 Pareto：

- CR：0.35 → 0.40，变差；
- EP：0.7735 → 0.7570，变差；
- FSR：0.9231 → 0.9167，极小改善；
- OPR：0.0587 → 0.0954，改善；
- fallback：0.85 → 0.80。

20 场数据的单场变化就是 5 个百分点，无法做可信显著性判断。更重要的是，predicted FSR 是模型自身证书输出，不是 reactive ground truth；不同 selector 间可比较，但 conventional 的 0 不是同口径真实“无 false-safe”。

## 5. 已发现并修复的工程错误

### 5.1 最可能的根因

natural residual/control head 是零初始化，输入来自 scene graph。原运行默认 CUDA FP16 autocast 时，graph feature 可能溢出为 Inf；在线性层中 `0 * Inf` 会产生 NaN。旧 loss 路径中的 `nan_to_num` 又可能把非法预测静默变成 0，随后 GradScaler 跳过 optimizer step，但日志没有记录跳步。

这能同时解释：

- residual/control 始终为 0；
- epoch 0 validation 与 epoch -1 完全相同；
- learned diagnostic 与 analytic basis 完全相同；
- loss 仍显示有限值；
- base deviation 却异常大；
- best checkpoint 一直是初始化。

在 CPU FP32 synthetic regression 中，decoder 零初始化能正确等于 base，执行一次优化后 residual/control 会非零。因此问题不是“结构上永远无法学习”，而是训练数值路径极可能阻断更新。

### 5.2 v16.3 修复

- natural decoder 和 dynamics integration 强制在 FP32 precision island 中执行；
- downstream AMP 的 `auto` 优先 BF16；
- natural stage 默认完全关闭 AMP；
- model output 在进入 loss 前递归检查 NaN/Inf，发现即同步终止 DDP；
- 不再允许非有限预测被静默清零；
- 检查 gradient norm；
- 记录 optimizer steps 与 AMP skipped steps；
- optimizer step 为 0 或 AMP skip 比例超过 2% 时拒绝保存“看似训练完成”的 checkpoint；
- 状态脚本同时识别 `eval/QUALITY_GATES_BYPASSED.txt` 和历史根目录 marker；
- precision policy 被写入 provenance manifest。

## 6. 论文中限制最终结果的理论问题与修订

### 6.1 旧 witness 的量词错误

旧定义要求：

- 存在一个 natural behavior 被 ego 变得 unsafe；
- 存在一个 burden 大于阈值的 safe response。

第二项几乎总能由 emergency hard brake 满足，它并不能证明“只有高负担响应才能安全”。正确问题应是：

- 有非忽略概率质量的 natural roots 被 ego 阻断；
- 对这些冲突 roots，其**最佳同 root 安全 continuation** 的负担仍高，或者不存在；
- 保留下来的低负担 root mass 太低。

v16.3 修订为：

- stable natural roots `r_im` 与质量 `p_im`；
- root-level conflict `c_ikm`；
- same-root safe-response probability / minimum burden；
- conflict mass；
- retained low-burden root mass OPR；
- conflict-conditioned CVaR burden；
- witness = conflict mass 超阈值，并且 tail burden 过高或 OPR 过低。

### 6.2 OPR 的集合交不成立

旧公式直接计算 independently sampled continuous trajectory sets 的交集，通常几乎处处为空，也与代码的 root assignment 不一致。修订后 OPR 是经过 root transport 后保留的自然概率质量，既可计算也符合实现。

### 6.3 Burden 与 option loss 的循环定义

旧 burden 包含 option loss，但 low-burden set 和 OPR 又由 burden 阈值定义，形成自指循环。修订后：

- primitive burden 只包含 acceleration、jerk、progress loss、risk exposure、priority/rule norm；
- option loss = `1 - OPR`，作为独立 feasibility/ranking 项；
- 不再放回 primitive burden。

### 6.4 阈值与证书联合学习会“解释掉” coercion

若 `beta_i(scene)` 与 witness 模型一起学习，模型可以通过提高阈值减少 positive witness。修订后阈值在独立 calibration split 上分层拟合并冻结，可加 conformal correction；test labels 不更新阈值。

### 6.5 不同 root 的最小值相减没有语义

旧 `C_i` 用 safe-response 全局最小 burden 减 natural-set 全局最小 burden，两个 minima 可能来自完全不同的意图，甚至产生负值。修订后只比较同一个 root，并对 conflicted roots 的 burden excess 做加权 tail aggregation。

### 6.6 自生成 pseudo-label 的闭环自证

natural basis、transport labels、witness 和最终指标若全部由同一个模型族生成，容易出现 self-confirmation。论文级证据至少需要：

- rule/optimization-based independent counterfactual evaluator；
- reactive non-ego simulator；
- 人工审核的 priority-aware false-safe stress set；
- 对 natural root、conflict interval、mechanism token 的抽样人工一致性评估。

## 7. 下一步算法优化优先级

### P0：先确认自然分支真的学习

在 v16.3 strict recovery 中必须看到：

- optimizer steps 明确大于 0；
- AMP skips = 0；
- residual/control 不再恒为 0；
- epoch 后 validation 与 initialization 不再逐位相同；
- learned 8 s error 优于 analytic basis；
- OBS 改善不能通过显著破坏 neutral / priority 得到。

若仍失败，应优先检查 graph feature scale、coordinate transform、anchor/base unit 与 mask，而不是继续改 planner。

### P1：把 natural roots 变成“可追踪身份”，而不是 unordered modes

建议每个 root 显式包含：

- source：OBS / neutral / priority；
- maneuver topology：keep / yield / go-first / lane-change / stop；
- route/lane anchor；
- conflict-order token；
- continuous motion residual。

transport 必须按 root identity 匹配，不能只依赖轨迹最近邻。加入 cycle consistency：neutral root → ego-conditioned response → 去除 ego 后应恢复到原 root 附近。

### P2：把 conflict 与 recovery 做成结构化两阶段任务

当前 conflict-conditioned AUPRC 只有 0.1247。建议拆成：

1. `P(conflict | ego, root)`；
2. `P(low-burden safe continuation | conflict, ego, root)`；
3. `best-safe burden distribution`；
4. 最后组合 certificate。

对 conflict roots 使用 focal / class-balanced loss 和 hard-negative mining；对 no-conflict roots 学 retain，而不是让所有标签竞争一个扁平 head。

### P3：用 distributional burden 代替单点回归

单一 burden regression 无法表达响应搜索的不确定性。建议预测：

- safe continuation existence；
- burden quantiles 或 ordinal bins；
- CVaR / upper-tail；
- epistemic uncertainty。

证书使用保守上界，而不是均值。阈值在 held-out calibration 上冻结。

### P4：selector 采用真正的 lexicographic constrained selection

顺序应固定为：

1. Waymax/geometry physical safety；
2. calibrated non-coercive feasibility；
3. 若无可行项，选最小 certificate violation；
4. 最后才按 progress/comfort/route utility 排序。

当前 fallback 0.80 表明 candidate generator 与 certificate 不匹配。不能只继续调 selector 阈值；应让 planner 生成更多“自我让步”的 candidate，例如更早减速、延迟 merge、让出 crossing order、选择后方 gap。

### P5：让 planner 接受 certificate-aware candidate augmentation

planner 的改善来源应是**候选空间更包含非胁迫解**，而不只是重新排序：

- 对每个 conflict region 生成 ego-yield / delayed-entry / alternative-gap candidates；
- 使用 certificate gradient 或 surrogate 引导局部 trajectory repair；
- 学习 proposal coverage loss：至少一个 candidate 同时满足 physical safety 与 NCF；
- 分开报告 candidate-bank oracle 与 selector regret。

如果 oracle NCF coverage 本身低，再强的 selector 也只会 fallback。

## 8. 建议的 CCF-A / SOTA 内部门槛

不存在一个官方的“CCF-A 自动驾驶规划指标门槛”，不同 venue、benchmark、action space 和闭环协议不可直接用单一数字比较。以下是本项目进入投稿写作前的**内部 go/no-go 标准**，不是官方标准：

### 8.1 Natural basis

- learned overall 8 s error 相对 analytic 至少改善 0.10 m；
- OBS 至少改善 0.20 m；
- neutral / priority 任一退化不超过 0.10–0.15 m；
- 三个 seeds 方向一致，paired CI 不跨 0；
- v15 label protocol 全量通过。

### 8.2 Mechanism / certificate

- RootTransport conflict-conditioned AUPRC ≥ 0.70；
- low-safe-exist AUPRC ≥ 0.75；
- BCOT false-safe AUPRC ≥ 0.70；
- witness AUPRC ≥ 0.80；
- WLA ≥ 0.60，MTA ≥ 0.65；
- calibration 必须存在 feasible operating point，不能是 `least_violation`。

### 8.3 Selector offline

- NCF recall ≥ 0.60；
- accepted candidate rate ≥ 0.20；
- selected false-safe ≤ 0.20；
- fallback ≤ 0.15；
- EP 相对 strongest matched selector 绝对下降不超过 0.02。

### 8.4 Closed loop

- 每 seed 至少 1,000 个 interaction-heavy scenes，至少 3 seeds；
- 与 strongest same-candidate baseline 做 paired bootstrap 95% CI；
- collision/off-road 不劣化，理想目标 CR 与 OR 各低于 5%–10%（具体受 scenario mining 难度影响）；
- reactive protocol 下 FSR / HBCR 至少相对下降 30%，且 EP 下降不超过 2 个百分点；
- logged replay 与 reactive results 分表；
- 报告 candidate-bank oracle、selector regret、fallback 原因分解。

当前距离：RootTransport conflict AUPRC 0.1247、BCOT AUPRC 0.4310、NCF recall 0.2146、accepted rate 0.0925、closed-loop CR 0.35–0.45、predicted FSR 约 0.91，仍处于“机制原型尚未成立”，不是“接近 SOTA”的阶段。

## 9. 推荐实验矩阵

### 阶段 A：数值恢复

- main v16.3 natural，FP32；
- 3 seeds；
- 检查 optimizer steps、AMP skips、non-finite、residual activation；
- strict natural basis/effectiveness gates。

### 阶段 B：自然分支归因

完全控制变量：

- main；
- no new effectiveness loss；
- no OBS capacity boost；
- no residual decoder / analytic only；
- no ego-neutral；
- no priority branch；
- full v15 labels vs v9 reuse。

### 阶段 C：transport 结构归因

- unordered mode matching；
- root identity matching；
- root identity + cycle consistency；
- point burden vs quantile burden；
- independent conflict/recovery heads；
- no mass floor / with mass floor。

### 阶段 D：planner / selector

固定同一 candidate bank 比较：

- conventional physical safety；
- score-only；
- soft burden；
- pair-max；
- epsilon-Pareto；
- COWP root-wise lexicographic；
- oracle certificate upper bound。

再固定 selector，比较 candidate generator 是否加入 certificate-aware ego-yield candidates。

## 10. 执行顺序

### 第一步：严格 natural recovery

```bash
cd COWP_v16_3_optimized
BACKGROUND=1 bash NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh
```

查看状态：

```bash
OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026 \
  bash CHECK_RUN_STATUS_V16_3.sh
```

必须先确认两个 natural gates 均通过，且 optimizer steps > 0、AMP skips = 0。

### 第二步：新 loss 与 OBS capacity 归因

```bash
MAIN_OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026 \
  bash RUN_NATURAL_ABLATIONS_V16_3_CN.sh
```

检查：

```bash
python -m json.tool \
  outputs/cowp_v16_3_natural_ablations_v9labels_seed2026/natural_component_attribution_gate.json
```

### 第三步：只有归因 gate 通过后才继续完整 pipeline

```bash
OUT_ROOT=outputs/cowp_v16_3_natural_recovery_v9labels_seed2026 \
ATTR_GATE=outputs/cowp_v16_3_natural_ablations_v9labels_seed2026/natural_component_attribution_gate.json \
BACKGROUND=1 bash NEXT_RUN_COMMANDS_V16_3_FULL_CN.sh
```

不要在 natural gate 或 attribution gate 失败时设置 `ALLOW_QUALITY_GATE_FAILURE=1` 后继续训练并把结果用于论文；该选项仅供工程 smoke。

## 11. 本次交付文件

- `interactive_planning_v16_3_revised.tex`：修订核心定义、实验协议和论文声称；
- `COWP_V16_3_TECHNICAL_REVIEW_CN.md`：本报告；
- `ALGORITHM_CHANGELOG.md`：已追加 v16.3 修改；
- `run_cowp_v16_3_dual_gpu.sh`：数值安全 pipeline；
- `NEXT_RUN_COMMANDS_V16_3_RECOVERY_CN.sh`：推荐首条命令；
- `RUN_NATURAL_ABLATIONS_V16_3_CN.sh`：controlled attribution；
- `NEXT_RUN_COMMANDS_V16_3_FULL_CN.sh`：通过 gate 后的完整训练/评估；
- `CHECK_RUN_STATUS_V16_3.sh`：状态和证据检查；
- `tests/test_v16_3_numeric_safety.py`：数值安全回归测试。
