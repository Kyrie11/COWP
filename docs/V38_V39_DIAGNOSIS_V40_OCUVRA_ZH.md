# OC-RAP 论文、v38/v39 结果诊断与 v40 OC-UVRA 优化方案

## 0. 结论

### 0.1 对当前论文与结果的总体判断

论文的核心问题设定是成立且有潜力的：**branch-wise oracle recovery 并不等价于真实可部署的 recovery**。当多个潜在隐状态在执行候选前缀后仍产生相容观测时，规划器不能依赖尚未获得的隐状态信息，为每个分支选择不同恢复动作；必须存在一个对该观测等价类共享、可执行的恢复选项。论文通过 observation compatibility、root-conditioned recovery margins、OC-MERO 与校准 CRISP selector 把这一点形式化，理论定位比普通的风险评分、backup feasibility 或 post-crash braking 更有辨识度。

但以目前上传的实验结果，**整体还没有达到 CCF-A 投稿所需的实证状态**：

- Safe 的 nominal preservation 已经方向性达标，但规模、随机种子和非劣性统计仍不足；
- Near-contact 的 v39 改动只降低了少量 audit miss，没有带来 PCD/FRA/DRS 或 clearance/TTC 的真实改善；
- Contact 在开发集上相对 v38 明显退化，更多干预换来了更低 NUP、更高 FRA 和更低 PCD；
- 只完成的扩展 seed 2027 暴露出 near-contact 在更大场景分布上远比 12-rollout 开发集困难；
- 当前证据仍高度依赖少量 audit 点和 selector heuristic，尚不能证明核心理论量在真实闭环中转化为物理安全收益。

### 0.2 v39 是否根本解决了 v38 的问题

没有。v39 的 OC-RAC 方向抓住了“同一 scene-time 内候选排序反转”这个现象，但实现方式把反事实 PCD 排序梯度重新压回 `R_dep`、DRS、oracle gap 三个原本承担证书与校准语义的量，导致目标冲突。结果是：

1. Near-contact 的 paper-PCD miss 从 0.050 降到 0.033，但 PCD、FRA、DRS 与实际物理轨迹完全不变；
2. Contact 的干预率从 0.025 上升到 0.0458，NUP 从 0.9916 降到 0.9759，FRA 从 0.20 恶化到 0.2333，PCD 从 0.4940 降到 0.4724；
3. near-heavy 版本没有解决 near，反而使 contact 干预更多、miss 更差；
4. 更大规模的 seed 2027 中，near PCD 只有 0.5051、FRA 0.1838、DRS 0.8162，说明 12-rollout 的近似达标不具备分布稳健性。

因此 v39 更像一个影响少量候选排序的局部修补，而不是模型层面的可靠修复。

### 0.3 本次代码优化

本次在 v39 基础上实现 v40 **OC-UVRA：Observation-Consistent Uncertainty-aware Value-guided Recovery Admission/ranking**。核心改变是把两个问题彻底拆开：

- **OC-MERO/CRISP 继续负责证书**：候选是否 observation-consistently recoverable、是否满足校准准入、物理/语义/预算约束；
- **新的 direct recovery value head 只负责偏好**：在已经被现有证书准入的候选中，哪个候选相对 nominal 有更高的反事实 deployable value。

新的价值头输出均值与方差，并通过 scene-time 组内 pointwise、listwise、nominal-versus-recovery advantage 损失训练。推理时使用验证集校准的选择后有效 lower-confidence bound：

```text
A_LCB(a, a_nom)
  = V_hat(a) - V_hat(a_nom)
    - z_cal * sqrt(sigma_hat(a)^2 + sigma_hat(a_nom)^2)
```

其中 `z_cal` 不是手工指定的高斯常数。校准工具对每个 scene-time 组取所有 recovery candidate 的最大标准化过估计误差，再做有限样本上分位数校准，因此该 LCB 针对“模型看完所有候选后再选择”的情形，而不是只对预先固定候选有效。

---

## 1. 对论文 idea、pipeline、数据和实验的理解

### 1.1 核心研究问题

论文关注 oracle-to-deployable gap：

- Branch-wise oracle 允许在不同潜在 root 下使用不同 recovery option；
- 真实车辆在候选前缀执行后只接收到一个观测，如果多个 roots 在该观测下不可区分，就不能依据真实系统尚未知晓的 root 身份选择不同恢复；
- 因而真正需要评估的是：在 observation-compatible roots 上，是否存在一个共享恢复动作，并且它在低尾部 root mass 上仍具备足够 margin。

这一问题与普通 collision probability、CVaR trajectory risk、state-wise viability 或预设 backup controller 不同。论文的潜在贡献不是“多一个风险分数”，而是把**观测可实现性**加入恢复存在性的定义。

### 1.2 算法 pipeline

代码与论文的主链路可以对应为：

1. **场景和候选前缀编码**：结构化 transformer 编码 scene、agents、map、route 与 candidate prefix；
2. **latent recovery roots**：root queries cross-attend 到 scene-prefix token，输出 root probability 与 root representation；
3. **post-prefix observation embedding**：每个 root 预测执行前缀后的观测 embedding，构造 compatibility kernel `C`；
4. **root-option recovery margin**：对语义 recovery option 预测每个 root 下的 signed margin；
5. **OC-MERO**：
   - `R_orc`：每个 root 可单独挑选最佳 recovery 的下尾部聚合；
   - `R_dep`：对 observation-compatible roots 使用共享 option 后的下尾部聚合；
   - `gap = R_orc - R_dep`：oracle 与可部署恢复的差距；
6. **CRISP selector**：校准 admission、hard/harm feasibility、nominal preservation、intervention budget/cooldown 与必要的 protective recovery channel；
7. **closed-loop audit**：执行候选，按 teacher recovery labels 计算 FRA、DRS、PCD、NUP 与物理指标。

### 1.3 数据集构建逻辑

三个数据 regime 的构建差异总体合理：

- **Safe**：不启用 targeted future 和 augmented hidden roots，主要验证正常场景不被错误干预；
- **Near-contact**：加入 hidden vehicle yields/accelerates、低附着制动、控制延迟噪声、augmented hidden roots、visible perturbation 与 artifact pair mining；
- **Contact/post-contact**：进一步加入 contact impulse surrogate 与 secondary-collision approach，强化接触后稳定与二次碰撞风险。

不过测试 near/contact 中的 artifact pass 会使用 margin override，并跳过部分 augmented Waymax 计算。它很适合测试“oracle artifact 检出能力”，但不应与自然场景物理收益混成一个主数字。正式论文应把结果拆成：

1. natural/interactive physical set；
2. oracle-artifact diagnostic set；
3. targeted perturbation stress set。

否则审稿人无法判断改进来自真实闭环状态，还是来自人为构造的 teacher margin 反例。

---

## 2. 三个 regime 的目标与应增加的指标

固定阈值只能作为开发 gate，不能直接等同于 CCF-A 证据。最终论文需要相对强基线的 paired effect、置信区间、多个随机种子和清晰的物理解释。

### 2.1 Safe regime

#### 现实目标

在正常可行驾驶中，OC-RAP 不应因不确定性、恢复评分或 artifact detector 而变得保守。它应保持 nominal 的轨迹、效率、舒适性和规则合规性，同时保留校准保证。

#### 开发目标

| 指标 | 建议 gate |
|---|---:|
| decision intervention rate | 0 |
| intervention episode rate | 0 |
| bounded NUP | >= 0.999，理想 1.0 |
| collision/overlap | 对 nominal 非劣 |
| offroad / kinematic infeasibility | 对 nominal 非劣 |
| log divergence / progress | 对 nominal 非劣 |
| jerk、加速度、yaw-rate | 对 nominal 非劣 |

#### 必须补充

- `FRA_admit`：所有 admitted candidates 中 teacher 不可部署的比例；
- `FRA_exec`：最终执行候选不可部署的比例；
- 在 `delta={1%,5%,10%}` 下的 reliability/coverage curve，而不只是一个 gamma；
- scene-paired non-inferiority CI；
- candidate-set size、admission coverage，防止通过全部拒绝获得低 FRA。

#### 当前状态

v38/v39 safe 均为全 nominal、NUP=1，方向上达标。但目前只证明了“这批 rollout 没有触发”，尚未证明多 seed、不同 safe 子分布和强基线下的统计非劣性。Safe 可进入论文主表，但还不能单独支撑总体方法已经成熟。

### 2.2 Near-contact regime

#### 现实目标

在尚未接触、但时空余量低的情况下，以极低干预成本提升 minimum clearance 和 TTC，减少从 near-contact 进入 contact 的概率。关键不是“多选几次 brake”，而是 recovery episode 确实改变了后续轨迹与危险暴露。

#### 开发 gate

| 指标 | 建议 gate |
|---|---:|
| paper-PCD selector miss | <= 0.034 |
| PCD | >= 0.54 |
| FRA_exec | <= 0.12 |
| DRS | >= 0.88 |
| bounded NUP | >= 0.995 |
| decision intervention | <= 0.020 |
| intervention episode rate | <= 0.012 |
| max consecutive intervention | <= 1 |

#### 论文级物理目标

| 指标 | 建议要求 |
|---|---:|
| paired delta of rollout min-clearance | >= +0.10 m，95% CI 下界 >= 0 |
| paired delta of finite min-TTC | >= +0.20 s，95% CI 下界 >= 0 |
| clearance < 2 m exposure | 显著下降 |
| TTC < 3 s exposure | 显著下降 |
| near-to-contact transition | 显著下降 |
| actual overlap/collision | 不增加 |

#### 新增“可作用机会”指标

上传结果显示部分 near intervention 发生时 ego 已经接近静止，brake 宏动作无法改变轨迹。因此还应报告：

- `recovery_opportunity_eligible_rate`：干预时 ego speed、控制差异和剩余 horizon 足以让候选产生物理作用的比例；
- eligible 子集上的 intervention rate 和 paired physical benefit；
- `control_delta_from_nominal`；
- 每个 recovery episode 的 clearance/TTC 改善，而不是只看全场景平均。

这可以区分“teacher label 上更优”与“闭环中可真正执行并产生效果”。

#### 当前状态

Near-contact **没有达到 CCF-A 状态**。v39 balanced 的 paper miss 降到 0.033，但 PCD/FRA/DRS 与 v38 完全相同；四次干预没有改变 aggregate physical trajectory。扩展 seed 2027 中 PCD/FRA/DRS 明显恶化，说明开发集的 miss 改善不是稳健收益。

### 2.3 Contact/post-contact regime

#### 现实目标

接触后允许短促、稀疏、具有明确恢复目的的动作，目标是避免二次碰撞、降低碰撞后伤害、抑制 yaw/offroad escalation，并进入稳定停车或可控 rejoin 状态。不能把“连续制动”本身当成成功。

#### 开发 gate

| 指标 | 建议 gate |
|---|---:|
| paper-PCD selector miss | <= 0.034 |
| PCD | >= 0.52 |
| FRA_exec | <= 0.16 |
| DRS | >= 0.84 |
| bounded NUP | >= 0.985 |
| decision intervention | <= 0.040 |
| intervention episode rate | <= 0.025 |
| max consecutive intervention | <= 2 |

#### 论文级物理目标

- secondary overlap/collision episode rate；
- new stable-stop rate：只统计 rollout 起点尚未稳定停车的 eligible scenes；
- time-to-stable-stop；
- collision/post-impact `Delta-v` 或 normalized impact severity；
- peak yaw rate、lateral acceleration；
- offroad duration；
- route-rejoin success 与 time-to-rejoin；
- recovery episode 的长度、宏动作切换率和 episode utility loss。

建议至少要求：

- paired secondary-overlap delta <= -0.02，CI 上界 <= 0；
- paired new-stable-stop delta >= +0.02，CI 下界 >= 0；
- NUP 与干预预算同时达标。

#### 当前状态

Contact **没有达到 CCF-A 状态**。v38 有真实但很小的改善；v39 balanced 在 12-rollout 开发集上退化，near-heavy 更差。扩展 seed 2027 的 contact 数字有所恢复，但只有一个 seed，而且场景级分析显示物理收益主要由一个连续多步 brake 场景驱动，不能证明普遍有效。

---

## 3. v38 与 v39 的实测对比

### 3.1 12-rollout 开发评测

| Regime / metric | v38 stateful-margin | v39 balanced | v39 near-heavy | 判断 |
|---|---:|---:|---:|---|
| Near intervention | 0.0000 | 0.0167 | 0.0125 | v39 开始干预 |
| Near NUP | 1.0000 | 1.0000 | 1.0000 | 无效用损失，但也可能是动作未产生作用 |
| Near PCD | 0.548116 | 0.548116 | 0.548116 | 完全无改善 |
| Near FRA | 0.116667 | 0.116667 | 0.116667 | 完全无改善 |
| Near DRS | 0.883333 | 0.883333 | 0.883333 | 完全无改善 |
| Near paper miss | 0.0500 | 0.0333 | 0.0333 | 少 1 个 miss |
| Near paper regret | 0.03120 | 0.02082 | 0.02082 | audit 排序有所改善 |
| Contact intervention | 0.0250 | 0.0458 | 0.0542 | 明显增加 |
| Contact NUP | 0.991639 | 0.975921 | 0.968657 | 明显退化 |
| Contact PCD | 0.494034 | 0.472369 | 0.472369 | 退化 |
| Contact FRA | 0.200000 | 0.233333 | 0.233333 | 退化 |
| Contact DRS | 0.800000 | 0.766667 | 0.766667 | 退化 |
| Contact paper miss | 0.0500 | 0.0667 | 0.1000 | 退化 |
| Contact paper regret | 0.01209 | 0.02338 | 0.04281 | 退化 |

v39 的 near 改进只体现在有限 audit 标签中的 one-miss reduction，没有影响主 deployability 指标。Contact 则是 clear regression。

### 3.2 未完成三 seed 中的 seed 2027

| Regime / metric | v39 balanced confirm s2027 |
|---|---:|
| Near scenes / decisions | 37 / 740 |
| Near intervention | 0.00946 |
| Near NUP | 1.0000 |
| Near PCD | 0.505125 |
| Near FRA | 0.183784 |
| Near DRS | 0.816216 |
| Near paper miss | 0.037838 |
| Near selector miss | 0.135135 |
| Contact scenes / decisions | 40 / 800 |
| Contact intervention | 0.0450 |
| Contact episode rate | 0.03375 |
| Contact NUP | 0.992309 |
| Contact PCD | 0.504646 |
| Contact FRA | 0.1850 |
| Contact DRS | 0.8150 |
| Contact paper miss | 0.0350 |

这组结果不能作为最终统计，但它非常有诊断价值：near-contact 的开发阈值明显过拟合于小 audit；contact 在大样本中接近开发目标，却仍缺少跨 seed 的 CI 和稳定物理收益。

### 3.3 v39 训练行为

Balanced 与 near-heavy 都在 epoch 2 选择 best，并在 epoch 6 early stop：

- balanced validation recovery-advantage loss：0.03682 -> 0.03670，之后无稳定下降；
- near-heavy：0.03679 -> 0.03653，改善同样极小；
- validation direct teacher-PCD loss 长期约 2.37，几乎未被修复；
- total validation loss 很大且波动，说明新增排序项没有建立稳定的可泛化表示。

更重要的是，v39 的 groupwise loss 需要完整 scene-time candidate set，但原 validation loader 是普通 batch，候选组会被拆散。以该 loss 选 best checkpoint 并不可靠。

---

## 4. v39 失败的根因

### 4.1 证书变量与偏好变量混在一起

`R_dep`、DRS、gap 本来各自有明确含义：

- `R_dep` 是 observation-consistent recoverability；
- DRS 是共享恢复动作在 roots 上的成功质量；
- gap 是 oracle 与 deployable 的差距。

v39 为修正候选 PCD 排序，同时要求 recovery 和 nominal 在这三个分量上做 anti-inversion。这样一个 candidate-level preference 目标会改变证书尺度，和 calibration、anti-oracle、margin regression 等目标竞争。Contact 中更多 brake 但 FRA/PCD 变差，正是这种语义污染的结果。

### 4.2 best-checkpoint 选择不可靠

组内 advantage loss 依赖同一 scene-time 中 nominal 与 recovery candidates 同批出现。训练用了 group sampler，但 validation 没有完整保持 candidate group。best epoch 因而依据不完整候选集合上的噪声 loss 选出。

### 4.3 near teacher gain 与真实可作用性不一致

场景级对比显示，v39 near 的物理指标与 scalar 完全一致；多次 brake 发生在 ego speed 近零的状态。也就是说，teacher PCD 可能根据 latent recovery root 给出候选优势，但在当前闭环 state/horizon 下，宏动作无法产生实际 clearance/TTC 变化。

这不是单纯 selector threshold 问题，而是数据与评价的 actionability mismatch：

- recovery candidate 是否控制上与 nominal 有实质差异；
- 当前 ego speed、距离、horizon 是否允许动作产生效果；
- teacher advantage 是否能在 Waymax closed-loop 中被实现。

### 4.4 旧 selector heuristic 仍占主导

v39 新增训练目标，但实际选择仍受到 stress nominal anchor、relative certificate、PCD rescue、brake tail、cooldown bypass 等多条历史通道影响。大量 heuristic 让结果难以归因，也使论文 novelty 容易被审稿人理解成阈值工程。

### 4.5 指标实现存在误导

v39 中：

- `max_intervention_run_length` 被按 scene 平均，因此出现小于 1 的“最大长度”；
- mean run length 是先按 scene 平均再平均，不是 pooled episode mean；
- per-scene p05/min 再平均被命名成全局 p05/min；
- `contact_exposure_rate` 实际是 clearance <= 0.05 m，不等同于真实 overlap；
- stable-stop 会把 rollout 起点已经停止的 scene 计为成功；
- 缺少与同一 scene scalar control 的 paired delta 和 CI。

### 4.6 小样本和 artifact 混合

60 个 audit 点下 miss rate 以 1/60=0.0167 跳变；0.033 与 0.050 只差一个点。没有 scene bootstrap CI 时，这类差异不能形成强结论。Artifact pair 还会改变 test distribution，必须与 natural physical set 分层报告。

---

## 5. v40 OC-UVRA 算法设计

### 5.1 Certificate–preference decoupling

对于候选前缀 `a`：

- 证书仍由 `R_dep`、DRS、gap、hard/harm、macro semantics 与 calibration 决定；
- 新 head 预测 `V(a)` 与 `sigma(a)`，用于已准入候选的排序和 nominal challenge；
- direct value 不进入 admission union，不能把原本 unadmitted candidate 变成 admitted。

这一结构可以形成清晰的论文命题：

> 如果原 OC-MERO/CRISP admission set 在校准分布下满足 false-recoverability admission 控制，那么任何只在该 admission set 内重新排序的 preference layer 不扩大 admission set，因此不会削弱原 admission guarantee。

这比通过修改 `R_dep`、DRS、gap 来做候选排序更容易证明，也更符合论文核心理论。

### 5.2 Direct uncertainty recovery value loss

新损失包含：

1. **heteroscedastic point regression**：拟合 observation-consistent teacher PCD，同时预测方差；
2. **listwise scene-time distillation**：保留全部候选的相对次序，不只看 best pair；
3. **nominal/recovery advantage**：teacher 显示 recovery 有显著优势时，要求 advantage LCB 为正；
4. **false-positive asymmetry**：nominal 更好时更强地惩罚 recovery 过估计；
5. **bucket/macro restriction**：只在 near/contact 与语义 recovery macros 上训练 intervention boundary。

### 5.3 Selection-valid conformal LCB

普通的 learned standard deviation 不等于校准置信区间。v40 新增验证集校准：

对 scene-time 组 `g` 中每个 recovery candidate `j` 计算

```text
s_gj = (A_pred_gj - A_true_gj) / sigma_pair_gj
```

再取

```text
s_g = max_j s_gj
```

并使用有限样本上分位数得到 `z_cal`。因为每个组先取了所有候选的最大过估计，模型在推理时根据自身预测选择 candidate 后，`A_LCB` 仍然受到组级控制。校准按 near/contact 分开进行。

### 5.4 两个训练候选

- **head_only（主候选）**：冻结 v39 encoder、root、margin、observation、utility 和 option heads，只训练 direct value/uncertainty head。它最大程度保留 v39 已有 OC-MERO certificate 尺度；
- **adapter_light（备选）**：冻结 encoder/root decoder，只允许 option/margin 小幅适配，并保留低权重 legacy loss。仅在 head-only 表示能力不足时使用。

### 5.5 推理约束

Direct-value challenge 需要同时满足：

- candidate 已通过现有 admission certificate；
- candidate macro 在 allowlist；
- hard/harm/feasible 约束；
- candidate direct value floor；
- candidate uncertainty ceiling；
- calibrated advantage LCB threshold；
- intervention budget/cooldown；
- near 最大连续 1，contact 最大连续 2。

### 5.6 修复的工程问题

- validation 采用完整 scene-time group、无 replacement、固定 group/order；
- paired scalar control 明确关闭 direct-value channel；
- scalar 与 v40 audit 使用完全一致的 targets/labels/rollouts；
- 修正 global max run length、pooled mean run length 和 metric aggregation；
- `new_stable_stop_event` 排除 rollout 起点已稳定停止的 scene；
- 增加 paired scene bootstrap 工具；
- medium confirmation 不通过时自动停止昂贵的三 seed 实验。

---

## 6. 下一步实验顺序

### Stage 1：双 GPU 并行训练

- GPU 0：head_only；
- GPU 1：adapter_light；
- 每个模型训练后分别进行 near/contact direct-advantage conformal calibration。

### Stage 2：12-rollout 开发评测

先评 head_only：

- 每个 scalar/v40 pair 使用两张 GPU 同步跑；
- 必须生成 paired near/contact bootstrap report；
- 若 head_only 未达到开发 gate，再评 adapter_light。

### Stage 3：24-rollout medium confirmation

这是新增的成本控制关口。它需要同时满足：

- Safe strict preservation；
- Near PCD/FRA/DRS/NUP/intervention；
- Contact PCD/FRA/DRS/NUP/intervention；
- paired clearance/TTC 和 secondary-overlap/stable-stop 的方向性改善。

失败则停止，不运行三 seed。

### Stage 4：三 seed publication confirmation

仅在 medium 通过后运行 2027/2028/2029：

- 每个 seed 约 48 rollouts、约 240 audit decisions；
- scalar 与 v40 使用同一 target list；
- 以 scene 为单位 bootstrap 10,000 次；
- 报告 seed-wise 结果和跨 seed hierarchical/bootstrap CI。

### Stage 5：消融

至少包含：

1. v39 selector-only / no direct value；
2. direct point estimate，`z=0`；
3. fixed `z=1`，无 validation conformal calibration；
4. calibrated simultaneous LCB（完整 v40）；
5. head_only vs adapter_light；
6. without observation kernel；
7. branch-wise oracle upper bound；
8. no anti-oracle；
9. no intervention episode cap。

---

## 7. 论文修改建议

### 7.1 把 v40 写成方法贡献，而不是 selector patch

推荐的贡献表述：

1. Observation-consistent recoverability 定义与 OC-MERO；
2. 校准的 recovery-preserving admission；
3. **certificate-preserving, selection-valid conformal recovery preference**：在不改变原 admission set 的前提下，学习并校准候选相对 nominal 的 deployable value。

避免把“brake-tail challenge”或多个阈值作为主要 novelty。

### 7.2 新增两个命题/定理

- **Admission preservation proposition**：preference layer 只在 admitted set 内排序，因此不扩大 false-admission event；
- **Selection-valid advantage coverage**：以 scene-time group 为 exchangeable unit，对 max-over-candidates nonconformity score 校准后，所选 candidate 的 true advantage 低于 LCB 的概率受 delta 控制。

### 7.3 主表结构

按 natural safe / natural near / natural contact 分表，并把 artifact diagnostics 独立成一张：

- Admission：FRA_admit、coverage、admission rate；
- Execution：FRA_exec、DRS、PCD、paper miss/regret；
- Utility：NUP、intervention decision/episode/run length；
- Physical：clearance、TTC、secondary overlap、stable stop、Delta-v/yaw/offroad/rejoin；
- Statistics：paired delta 与 95% CI。

### 7.4 外部基线

相同 candidate pool、相同物理预算下至少比较：

- nominal / backup filter；
- CVaR 或 distributionally robust planner；
- predictive safety filter / barrier-style backup；
- contingency/scenario-tree planner；
- post-impact braking / stable-stop controller；
- branch-wise oracle upper bound；
- learned direct risk ranking without observation consistency。

---

## 8. 交付代码

主要新增/修改：

- `src/ocrap/models/ocrap.py`：direct recovery value/uncertainty head；
- `src/ocrap/models/losses.py`：uncertainty-aware scene-time recovery value loss；
- `src/ocrap/cli/train.py`：loss 接入、checkpoint 配置、deterministic grouped validation；
- `src/ocrap/models/inference.py`：direct value/std 推理；
- `src/ocrap/planning/selector.py`：LCB preference channel，且不扩大 admission set；
- `src/ocrap/evaluation/evaluator.py`、`baselines.py`：offline selector 接入；
- `src/ocrap/simulation/closed_loop_runner.py`：物理/episode 指标修复；
- `tools/calibrate_direct_value_advantage.py`：scene-time simultaneous conformal calibration；
- `tools/compare_paired_closed_loop.py`：paired scene bootstrap CI；
- `tools/check_v40_regime_targets.py`：开发与 publication gate；
- `scripts/train_ocrap_v40_ocuvra.sh`；
- `scripts/run_ocrap_v40_ocuvra.sh`；
- `run_v40_two_gpu_commands.txt`；
- `run_v40_ablation_commands.txt`；
- `tests/test_v40_direct_value.py`。

本地静态与单元测试：

```text
python -m compileall -q src tools tests
bash -n scripts/train_ocrap_v40_ocuvra.sh
bash -n scripts/run_ocrap_v40_ocuvra.sh
bash -n run_v40_two_gpu_commands.txt
PYTHONPATH=src pytest -q
```

结果：**57 passed**。

当前环境未具备你的 WOMD/Waymax 数据和两张 GPU，因此没有实际训练 v40，也没有声称 v40 已取得数值提升。v40 的价值在于修正 v39 的目标结构、验证协议、pairing 和指标实现；真实效果必须按交付命令验证。
